"""Particle-simulation methods and data utilities."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "adaptive-gns"))
from gns.data_loader import load_manifest_data
from gns.device_utils import resolve_device, runtime_provenance, synchronize
from gns.model_io import load_for_evaluation
from research.budget_graph import candidates, directed, random_pairs, select_pairs
from research.full_training import RunLock

HISTORY = 6
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
RADIUS_FACTOR = 1.267
FRACTION = .25
TRACE_STEPS = (1, 10, 50, 200, 500, 995)
OFFICIAL_TEST_SHA256 = "b7f147c22e96fd3fbb8d595702cb8c85e412bfb449d7b4d761cc03eb76246c31"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def state_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class RolloutGuard(RuntimeError):
    def __init__(self, category, **details):
        self.details = {"category": category, **details}
        super().__init__(category)


def check_state(value, phase, max_abs_coordinate):
    if not np.isfinite(value).all():
        raise RolloutGuard("nonfinite_" + phase)
    largest = float(np.max(np.abs(value)))
    if largest > max_abs_coordinate:
        raise RolloutGuard("coordinate_resource_guard", phase=phase,
                           maximum_absolute_coordinate=largest, limit=max_abs_coordinate)


def checked_candidates(current, radius, max_candidate_pairs):
    """Count before allocating pairs; count and query use the same <= radius."""
    tree = cKDTree(current)
    count = int((tree.count_neighbors(tree, radius * RADIUS_FACTOR) - len(current)) // 2)
    if count > max_candidate_pairs:
        raise RolloutGuard("candidate_pair_resource_guard", candidate_pairs=count,
                           limit=max_candidate_pairs)
    graph = candidates(current, radius, RADIUS_FACTOR)
    if len(graph.base) + len(graph.extra) != count:
        raise RuntimeError("Candidate counting and pair materialization disagree")
    return graph


def boundary_metrics(position, bounds):
    """Geometric diagnostics, not pass/fail criteria or physical validation."""
    position = np.asarray(position, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    excursions = np.maximum(np.maximum(bounds[:, 0] - position,
                                       position - bounds[:, 1]), 0.)
    per_particle = excursions.max(axis=-1)
    return {
        "fraction_particles_outside": float(np.mean(per_particle > 0)),
        "fraction_particles_outside_by_more_than_1e-6": float(np.mean(per_particle > 1e-6)),
        "maximum_coordinate_excursion": float(excursions.max()),
        "mean_particle_maximum_excursion": float(per_particle.mean()),
        "coordinate_minimum": position.min(axis=0).tolist(),
        "coordinate_maximum": position.max(axis=0).tolist(),
    }


@torch.no_grad()
def predict_supplied_graph(model, history, particle_types, pairs, device):
    """Original feature construction and integrator; no ground truth argument."""
    sequence = torch.as_tensor(np.ascontiguousarray(history.transpose(1, 0, 2)),
                               dtype=torch.float32, device=device)
    types = torch.as_tensor(particle_types, dtype=torch.long, device=device)
    edge = torch.as_tensor(directed(pairs), dtype=torch.long, device=device)
    counts = torch.tensor([len(particle_types)], dtype=torch.long, device=device)
    prediction, raw_head = model._forward_with_edge_index(
        sequence, counts, types, edge[0], edge[1], None)
    # Check the raw head too: conversion must not hide nonfinite raw values.
    if not bool(torch.isfinite(raw_head).all()):
        raise RolloutGuard("nonfinite_risk")
    variance = model.head_to_variance(raw_head)
    synchronize(device)
    return prediction.detach().cpu().numpy(), variance.detach().cpu().numpy().reshape(-1)


def choose_pairs(graph, policy, history, previous_risk, rng):
    budget = math.floor(FRACTION * len(graph.extra))
    if policy == "base":
        return graph.base
    if policy == "dense":
        return np.concatenate((graph.base, graph.extra))
    if policy == "random25":
        return random_pairs(graph, budget, rng)
    if policy == "speed25":
        return select_pairs(graph, np.linalg.norm(history[-1] - history[-2], axis=-1), budget)
    if previous_risk is None:
        raise RuntimeError("Cached-risk forecast requires its explicit warmup")
    return select_pairs(graph, previous_risk, budget)


@torch.no_grad()
def rollout(model, positions, particle_types, metadata, policy, horizon, rng_seed,
            device="cpu", max_candidate_pairs=100000, max_abs_coordinate=10.,
            trace_steps=TRACE_STEPS):
    """One policy and trajectory. Return a JSON record and safe numeric traces."""
    if policy not in POLICIES or horizon < 1:
        raise ValueError("Known policy and positive horizon required")
    if max_candidate_pairs < 1 or max_abs_coordinate <= 0:
        raise ValueError("Positive resource guards required")
    if positions.ndim != 3 or len(positions) < horizon + HISTORY or positions.shape[-1] != 2:
        raise ValueError("Expected a 2D trajectory with six observed frames plus the horizon")
    types = np.array(particle_types, dtype=np.int64, copy=True)
    if types.ndim == 0:
        types = np.full(positions.shape[1], types, dtype=np.int64)
    if types.shape != (positions.shape[1],) or np.any(types == 3):
        raise ValueError("Autonomous WATERDROP evaluation requires one type per node and no prescribed kinematic particles")
    bounds = np.asarray(metadata["bounds"])
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Finite ordered 2D bounds required")
    radius = float(metadata["default_connectivity_radius"])
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("Positive finite radius required")
    if not math.isclose(float(model._connectivity_radius), radius, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Checkpoint and metadata radii differ")
    started = time.perf_counter()
    history = np.array(positions[:HISTORY], dtype=np.float32, copy=True)
    initial_hash = state_hash(history)
    rng = np.random.default_rng(rng_seed)
    previous_risk, failure, warmup = None, None, None
    attempts, mse, edges, optional, candidate_counts, base_counts, risk_means = [], [], [], [], [], [], []
    predicted_boundary, truth_boundary = [], []
    trace_indices, trace_prediction, trace_truth = [], [], []
    total_passes, forecast_passes = 0, 0
    graph_seconds, forward_seconds = 0., 0.
    initial_boundary = [boundary_metrics(frame, bounds) for frame in history] if np.isfinite(history).all() else None
    trace_steps = set(int(value) for value in trace_steps)
    model.eval()
    # Warmup never advances history and never reads frame 7. Other policies
    # receive no hidden pass. The first actual forecast always has exact K.
    if policy == "laggedrisk25":
        warmup = {"passes": 0, "history_sha256": initial_hash, "mean_prediction_discarded": True}
        warmup_start = time.perf_counter()
        phase = "state"
        try:
            check_state(history, "state", max_abs_coordinate)
            phase = "graph"
            graph_start = time.perf_counter()
            graph = checked_candidates(history[-1], radius, max_candidate_pairs)
            warmup["graph_build_and_selection_seconds"] = time.perf_counter() - graph_start
            warmup.update(candidate_pairs=len(graph.base) + len(graph.extra),
                          retained_pairs=len(graph.base), directed_edges=2 * len(graph.base))
            forward_start = time.perf_counter()
            total_passes += 1
            warmup["passes"] = 1
            phase = "forward"
            unused_prediction, previous_risk = predict_supplied_graph(model, history, types, graph.base, device)
            warmup["features_forward_decode_and_transfer_seconds"] = time.perf_counter() - forward_start
            if not np.isfinite(unused_prediction).all():
                raise RolloutGuard("nonfinite_warmup_prediction")
            if not np.isfinite(previous_risk).all():
                raise RolloutGuard("nonfinite_risk")
            warmup["mean_normalized_acceleration_variance"] = float(previous_risk.mean())
        except RolloutGuard as error:
            failure = {**error.details, "forecast_step": 0, "phase": "warmup_" + phase}
        warmup["total_wall_seconds"] = time.perf_counter() - warmup_start
    for step in range(1, horizon + 1):
        if failure is not None:
            break
        attempt = {"forecast_step": step, "accepted": False}
        phase = "state"
        try:
            check_state(history, "state", max_abs_coordinate)
            graph_start = time.perf_counter()
            phase = "graph"
            graph = checked_candidates(history[-1], radius, max_candidate_pairs)
            pairs = choose_pairs(graph, policy, history, previous_risk, rng)
            extra = len(pairs) - len(graph.base)
            budget = math.floor(FRACTION * len(graph.extra))
            if policy in POLICIES[2:] and extra != budget:
                raise RuntimeError("Exact optional pair budget was violated")
            attempt.update(candidate_pairs=len(graph.base) + len(graph.extra), base_pairs=len(graph.base),
                           available_annulus_pairs=len(graph.extra), optional_pair_budget=budget,
                           retained_optional_pairs=extra, directed_edges=2 * len(pairs),
                           graph_build_and_selection_seconds=time.perf_counter() - graph_start)
            graph_seconds += attempt["graph_build_and_selection_seconds"]
            forward_start = time.perf_counter()
            phase = "forward"
            total_passes += 1
            forecast_passes += 1
            prediction, risk = predict_supplied_graph(model, history, types, pairs, device)
            attempt["features_forward_decode_and_transfer_seconds"] = time.perf_counter() - forward_start
            forward_seconds += attempt["features_forward_decode_and_transfer_seconds"]
            if prediction.shape != history[-1].shape or risk.shape != (len(types),):
                raise RuntimeError("Predictor returned an incompatible shape")
            if np.isfinite(prediction).all():
                attempt["predicted_boundary"] = boundary_metrics(prediction, bounds)
            check_state(prediction, "prediction", max_abs_coordinate)
            if not np.isfinite(risk).all():
                raise RolloutGuard("nonfinite_risk")
            # This is the first future ground-truth access in the forecast loop.
            truth = np.asarray(positions[HISTORY + step - 1], dtype=np.float64)
            if not np.isfinite(truth).all():
                raise ValueError("Ground truth contains nonfinite coordinates")
            error = float(np.mean((prediction.astype(np.float64) - truth) ** 2))
            mse.append(error)
            edges.append(2 * len(pairs)); optional.append(extra)
            candidate_counts.append(len(graph.base) + len(graph.extra)); base_counts.append(len(graph.base))
            risk_means.append(float(risk.mean()))
            predicted_boundary.append(attempt["predicted_boundary"])
            truth_boundary.append(boundary_metrics(truth, bounds))
            attempt.update(accepted=True, coordinate_mse=error,
                           ground_truth_boundary=truth_boundary[-1], predicted_state_sha256=state_hash(prediction))
            if step in trace_steps:
                trace_indices.append(step); trace_prediction.append(prediction.copy()); trace_truth.append(truth.astype(np.float32))
            previous_risk = risk
            history = np.concatenate((history[1:], prediction[None]), axis=0)
        except RolloutGuard as error:
            failure = {**error.details, "forecast_step": step, "phase": phase}
            attempt["failure"] = failure
        attempts.append(attempt)
    synchronize(device)
    complete = failure is None and len(mse) == horizon
    if warmup is not None:
        graph_seconds += warmup.get("graph_build_and_selection_seconds", 0.)
        forward_seconds += warmup.get("features_forward_decode_and_transfer_seconds", 0.)
    result = {
        "status": "complete" if complete else "failed", "failure": failure,
        "policy": policy, "horizon": horizon, "completed_steps": len(mse), "rng_seed": int(rng_seed),
        "warmup": warmup, "total_network_passes": total_passes, "forecast_network_passes": forecast_passes,
        "mse_at_steps": {str(s): mse[s - 1] if len(mse) >= s else None for s in TRACE_STEPS if s <= horizon},
        "mse_at_final_horizon": mse[-1] if complete else None,
        "mean_rollout_mse": float(np.mean(mse)) if complete else None,
        "mean_directed_edges": float(np.mean(edges)) if complete else None,
        "prefix_mean_mse_if_failed": float(np.mean(mse)) if mse and not complete else None,
        "total_wall_seconds": time.perf_counter() - started,
        "graph_build_and_selection_seconds_including_warmup": graph_seconds,
        "features_forward_decode_and_transfer_seconds_including_warmup": forward_seconds,
        "initial_observed_state_sha256": initial_hash, "final_predicted_state_sha256": state_hash(history[-1]),
        "mse_per_step": mse, "directed_edges_per_step": edges, "retained_optional_pairs_per_step": optional,
        "candidate_pairs_per_step": candidate_counts, "base_pairs_per_step": base_counts,
        "mean_normalized_acceleration_variance_per_step": risk_means,
        "initial_observed_boundary": initial_boundary,
        "predicted_boundary_per_step": predicted_boundary, "ground_truth_boundary_per_step": truth_boundary,
        "attempts": attempts,
    }
    traces = {"forecast_steps": np.asarray(trace_indices, dtype=np.int64),
              "initial_observed_positions": np.asarray(positions[:HISTORY], dtype=np.float32),
              "predicted_positions": np.stack(trace_prediction) if trace_prediction else np.empty((0, len(types), 2), dtype=np.float32),
              "ground_truth_positions": np.stack(trace_truth) if trace_truth else np.empty((0, len(types), 2), dtype=np.float32),
              "particle_types": types, "bounds": np.asarray(bounds, dtype=np.float64)}
    return result, traces


def summarize(records, expected_ids, policies=POLICIES):
    """Equal trajectories, with undefined unconditional error after any failure."""
    expected_ids = set(expected_ids)
    if not expected_ids:
        raise ValueError("At least one expected trajectory required")
    result = {}
    for policy in policies:
        rows = [row for row in records if row["policy"] == policy]
        ids = [row["trajectory_id"] for row in rows]
        if len(ids) != len(set(ids)) or not set(ids) <= expected_ids:
            raise ValueError("Duplicate or unexpected trajectory in summary")
        completed = [row for row in rows if row["status"] == "complete"]
        all_complete = len(completed) == len(expected_ids)
        values = {name: float(np.mean([row[name] for row in rows])) if all_complete else None
                  for name in ("mean_rollout_mse", "mse_at_final_horizon", "mean_directed_edges")}
        result[policy] = {**values,
            "required_trajectories": len(expected_ids), "attempted_trajectories": len(rows),
            "completed_trajectories": len(completed), "failed_trajectories": len(rows) - len(completed),
            "missing_trajectories": sorted(expected_ids - set(ids)),
            "all_sample_full_horizon_error_defined": all_complete,
            "total_wall_seconds_including_warmup": float(sum(row["total_wall_seconds"] for row in rows)),
            "total_network_passes_including_warmup": sum(row["total_network_passes"] for row in rows),
            "mse_at_steps": {str(s): float(np.mean([row["mse_at_steps"][str(s)] for row in rows]))
                             if len(rows) == len(expected_ids) and all(row["mse_at_steps"].get(str(s)) is not None for row in rows)
                             else None for s in TRACE_STEPS},
        }
    return result


def select_records(manifest, scope, start_index=None, trajectory_ids=None, max_trajectories=None,
                   expected_source_sha256=OFFICIAL_TEST_SHA256):
    expected_split = "test" if scope == "locked_test" else "valid"
    if manifest.get("split") != expected_split:
        raise ValueError(f"{scope} requires the {expected_split} split")
    records = manifest["records"]
    if scope == "locked_test":
        source = manifest.get("source", {})
        if (len(records) != 30 or manifest.get("record_count") != 30 or source.get("record_count") != 30
                or source.get("CRC_verified") is not True):
            raise ValueError("Locked test requires the complete CRC-verified 30-trajectory official test manifest")
        if (expected_source_sha256 != OFFICIAL_TEST_SHA256
                or source.get("sha256") != OFFICIAL_TEST_SHA256):
            raise ValueError("Locked test requires the pinned official test-source SHA256")
    if any(row.get("source_index") != index for index, row in enumerate(records)):
        raise ValueError("Complete source-ordered manifests with explicit source_index are required")
    default_start = 3 if scope == "locked_test" else 0
    start_index = default_start if start_index is None else start_index
    if start_index < 0 or start_index >= len(records):
        raise ValueError("Trajectory start index is outside the manifest")
    if scope == "locked_test" and (start_index != 3 or trajectory_ids or max_trajectories is not None):
        raise ValueError("Locked test requires every trajectory from source index 3; subsets are validation-only")
    selected = list(range(start_index, len(records)))
    if trajectory_ids:
        requested = set(trajectory_ids)
        if len(requested) != len(trajectory_ids) or not requested <= {records[i]["id"] for i in selected}:
            raise ValueError("Unknown, excluded or duplicate trajectory ID")
        selected = [i for i in selected if records[i]["id"] in requested]
    if max_trajectories is not None:
        if max_trajectories < 1:
            raise ValueError("Positive maximum trajectory count required")
        selected = selected[:max_trajectories]
    return selected


def check_checkpoint(payload, scope, protocol_path, metadata_sha256=None):
    config = payload.get("simulator_config", {})
    required = {"particle_dimensions": 2, "latent_dim": 128, "nmessage_passing_steps": 10,
                "nmlp_layers": 2, "mlp_hidden_dim": 128, "uncertainty_parameterization": "variance"}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Full evaluation requires the original 128x10 architecture and corrected variance semantics")
    training = payload.get("training_config", {})
    if training.get("loss") not in {"nll", "faithful"}:
        raise ValueError("A corrected NLL or faithful checkpoint is required")
    if bool(config.get("detach_variance_features")) != (training["loss"] == "faithful"):
        raise ValueError("Checkpoint risk feature detachment differs from its objective")
    if scope == "locked_test":
        run = payload.get("run_config", {})
        digest = hashlib.sha256(json.dumps(run, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (payload.get("full_training_schema") != 1 or payload.get("completed_steps") != 100000
                or training.get("completed_optimizer_updates") != 100000 or run.get("steps") != 100000
                or run.get("scope") != "bounded_full_data_100k" or payload.get("run_config_sha256") != digest
                or run.get("research_protocol_sha256") != sha256(protocol_path)):
            raise ValueError("Locked test requires the complete fixed 100k checkpoint and matching research protocol hash")
        if (run.get("seed") not in (0, 1, 2) or run.get("objective") != training["loss"]
                or metadata_sha256 is None or run.get("metadata_sha256") != metadata_sha256):
            raise ValueError("Locked test requires paired seed 0..2, consistent objective and identical training/evaluation metadata SHA256")


def compact_record(row, row_path):
    """Keep the index small; full step diagnostics stay in a separate file."""
    fields = ("trajectory_id", "source_index", "policy", "status", "failure", "horizon", "n_particles",
              "rng_seed", "completed_steps", "mse_at_steps", "mse_at_final_horizon", "mean_rollout_mse",
              "mean_directed_edges", "total_wall_seconds", "total_network_passes", "forecast_network_passes",
              "trace_file", "trace_sha256")
    return {**{key: row[key] for key in fields}, "record_file": row_path.name,
            "record_sha256": sha256(row_path)}


def recover_record(directory, index, policy, trajectory_id, protocol_sha256, horizon, indexed=None):
    """Verify finished outcomes before reuse, including a just-written orphan.

    An orphan can occur if the process exits after the per-rollout row is
    committed but before the compact index is replaced. Its embedded protocol,
    identity and trace hash must still match. Guard-failed outcomes are retained.
    """
    path = Path(directory) / f"trajectory_{index:06d}_{policy}.json"
    if not path.exists():
        if indexed is not None:
            raise ValueError("Indexed rollout record is missing")
        return None
    row = json.loads(path.read_text())
    if (row.get("protocol_sha256") != protocol_sha256 or row.get("trajectory_id") != trajectory_id
            or row.get("source_index") != index or row.get("policy") != policy or row.get("horizon") != horizon
            or row.get("status") not in {"complete", "failed"}):
        raise ValueError("Saved rollout identity/protocol differs; refusing reuse")
    trace = Path(directory) / f"trajectory_{index:06d}_{policy}.npz"
    if row.get("trace_file") != trace.name or not trace.is_file() or sha256(trace) != row.get("trace_sha256"):
        raise ValueError("Saved rollout trace is missing or its SHA256 differs")
    if ((row["status"] == "complete" and (row["completed_steps"] != horizon or row["failure"] is not None))
            or (row["status"] == "failed" and (row["completed_steps"] >= horizon or row["failure"] is None))
            or len(row["mse_per_step"]) != row["completed_steps"]):
        raise ValueError("Saved rollout completion/failure record is inconsistent")
    compact = compact_record(row, path)
    if indexed is not None and compact != indexed:
        raise ValueError("Saved rollout record SHA256 or compact index differs")
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot_validation", "locked_test"), required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "research/protocols/full_waterdrop_100k.md")
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--horizon", type=int, default=995)
    parser.add_argument("--policies", choices=POLICIES, nargs="+", default=list(POLICIES))
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--trajectory-ids", nargs="+")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rng-seed", type=int, default=93000)
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    parser.add_argument("--max-abs-coordinate", type=float, default=10.)
    parser.add_argument("--expected-test-source-sha256", default=OFFICIAL_TEST_SHA256,
                        help="Pinned official full test TFRecord identity; locked scope rejects another source")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.horizon < 1 or len(set(args.policies)) != len(args.policies):
        parser.error("Positive threads/horizon and unique policies required")
    if args.scope == "locked_test" and (args.horizon != 995 or tuple(args.policies) != POLICIES
            or args.max_candidate_pairs != 100000 or args.max_abs_coordinate != 10. or args.rng_seed != 93000):
        parser.error("Locked test requires the fixed horizon, policies, guards and random seed")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest = json.loads(args.manifest.read_text())
    selected = select_records(manifest, args.scope, args.start_index, args.trajectory_ids, args.max_trajectories,
                              args.expected_test_source_sha256)
    trajectories = load_manifest_data(args.manifest, verify_hashes=True)
    metadata_path = args.manifest.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if manifest.get("metadata") != metadata:
        raise ValueError("Embedded manifest metadata differs from metadata.json")
    check_checkpoint(payload, args.scope, args.protocol, sha256(metadata_path))
    model, provenance = load_for_evaluation(args.checkpoint, metadata, device, radius_backend="scipy_host")
    if np.asarray(model._boundaries).tolist() != metadata["bounds"]:
        raise ValueError("Checkpoint and dataset bounds differ")
    records_description = [manifest["records"][index] for index in selected]
    seed = payload.get("run_config", {}).get("seed", payload.get("training_config", {}).get("seed"))
    source_paths = [Path(__file__), REPO / "research/budget_graph.py", REPO / "research/full_training.py"]
    source_paths += [REPO / "adaptive-gns/gns" / name for name in
                     ("data_loader.py", "device_utils.py", "model_io.py", "learned_simulator.py", "graph_network.py")]
    protocol = {
        "schema": 2, "scope": args.scope, "split": manifest["split"],
        "seed": seed, "objective": payload["training_config"]["loss"],
        "horizon": args.horizon, "context_frames": HISTORY, "policies": args.policies,
        "trajectory_ids": [row["id"] for row in records_description],
        "trajectory_records": records_description,
        "source_tfrecord": manifest["source"], "manifest_sha256": sha256(args.manifest),
        "pinned_test_source_sha256": args.expected_test_source_sha256 if args.scope == "locked_test" else None,
        "metadata_sha256": sha256(metadata_path), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_provenance": provenance, "research_protocol_sha256": sha256(args.protocol),
        "code_sha256": {str(path.relative_to(REPO)): sha256(path) for path in source_paths},
        "graph": {"base_radius": metadata["default_connectivity_radius"], "radius_factor": RADIUS_FACTOR,
                  "optional_fraction": FRACTION, "pairs": "uncapped undirected pairs with distance <= radius",
                  "message_edges": "both directions per pair; no self loops",
                  "priority": "max endpoint score; lexicographic pair-ID tie breaking",
                  "training_difference": "training graph is capped, self-looped and uses strict distance < radius",
                  "normalization": "original saved base-radius edge/boundary features and noise-adjusted acceleration decoder"},
        "risk": {"units": "per-coordinate normalized acceleration variance, converted with model.head_to_variance",
                 "initialization": "one explicit base-graph warmup on initial six observed frames; mean discarded; cost included",
                 "forecast": "exact floor(.25 * available annulus pairs) from first forecast; subsequently cache risk from own selected graph",
                 "other_policies": "no warmup network passes"},
        "autonomy": "only first six frames observed; no future truth or prescribed kinematic replacement in controller",
        "random_seed": {"base": args.rng_seed, "rule": "base + 1000*training_seed + source_index; same stream within paired seeds"},
        "guards": {"max_candidate_pairs": args.max_candidate_pairs, "max_absolute_coordinate": args.max_abs_coordinate,
                   "nonfinite": "state, prediction or raw/converted risk, including warmup",
                   "warmup_mean": "discarded mean must be finite, but absolute-coordinate guard applies only to accepted state and actual forecasts",
                   "interpretation": "computational limits, not physical-validity tests"},
        "metrics": {"mse": "mean squared position error over particles and coordinates; equal trajectories within checkpoint",
                    "failure": "full-horizon error undefined if any required trajectory fails or is missing; no survivors-only average",
                    "boundary": "exact fraction outside, fraction outside by >1e-6, and excursion magnitude, with same-frame truth reference",
                    "timing": "synchronized wall clock; host candidates, selection, tensor transfers, original forward/decoder, metrics and warmup included",
                    "timing_caveat": "single measurement in fixed policy order; first policy may include lazy device initialization; compare controlled repeated same-state timings separately"},
        "trace_forecast_steps": list(TRACE_STEPS), "runtime": runtime_provenance(device, "scipy_host"),
        "software": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__},
        "threads": args.threads,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(args.output_dir, clear_stale=args.clear_stale_lock):
        evaluate_to_directory(args, protocol, model, trajectories, manifest, selected, metadata, seed, device)


def evaluate_to_directory(args, protocol, model, trajectories, manifest, selected, metadata, seed, device):
    """Called under exclusive RunLock; verifies every reused row before forecasting."""
    protocol_path = args.output_dir / "protocol.json"
    result_path = args.output_dir / "result.json"
    indexed = {}
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing output requires --resume and exactly matching protocol/data/checkpoint/source/software hashes")
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("protocol_sha256") != sha256(protocol_path):
                raise ValueError("Saved result protocol SHA256 differs")
            for row in previous["records"]:
                key = (row["source_index"], row["policy"])
                if key in indexed or key[0] not in selected or key[1] not in args.policies:
                    raise ValueError("Duplicate or unexpected record in saved result")
                indexed[key] = row
    else:
        if args.resume or result_path.exists():
            raise ValueError("Resume requires the original protocol.json; existing results must be preserved")
        atomic_json(protocol_path, protocol)  # Frozen before any forecast outcomes.
    protocol_digest = sha256(protocol_path)
    recovered = {}
    for index in selected:
        for policy in args.policies:
            row = recover_record(args.output_dir, index, policy, manifest["records"][index]["id"],
                                 protocol_digest, args.horizon, indexed.get((index, policy)))
            if row is not None:
                if not args.resume:
                    raise ValueError("Existing per-rollout record requires --resume")
                recovered[index, policy] = row
    results = list(recovered.values())
    started = time.perf_counter()
    atomic_json(args.output_dir / "status.json", {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "verified_reused_policy_trajectories": len(recovered)})
    try:
        for index in selected:
            record = manifest["records"][index]
            if all((index, policy) in recovered for policy in args.policies):
                continue
            positions, types = trajectories[index]
            for policy in args.policies:
                if (index, policy) in recovered:
                    continue
                row, trace = rollout(model, positions, types, metadata, policy, args.horizon,
                                     args.rng_seed + 1000 * (seed or 0) + index, device,
                                     args.max_candidate_pairs, args.max_abs_coordinate)
                trace_path = args.output_dir / f"trajectory_{index:06d}_{policy}.npz"
                with trace_path.with_suffix(".npz.tmp").open("wb") as stream:
                    np.savez_compressed(stream, **trace)
                trace_path.with_suffix(".npz.tmp").replace(trace_path)
                row.update(trajectory_id=record["id"], source_index=index, n_particles=positions.shape[1],
                           trace_file=trace_path.name, trace_sha256=sha256(trace_path), protocol_sha256=protocol_digest)
                row_path = args.output_dir / f"trajectory_{index:06d}_{policy}.json"
                atomic_json(row_path, row)
                results.append(compact_record(row, row_path))
                atomic_json(result_path, {
                    "state": "partial", "protocol_sha256": protocol_digest,
                    "records": results, "summary": summarize(results, protocol["trajectory_ids"], args.policies)})
                print(record["id"], policy, row["status"], row["completed_steps"], round(row["total_wall_seconds"], 3), flush=True)
        results.sort(key=lambda row: (row["source_index"], args.policies.index(row["policy"])))
        result = {"state": "complete", "protocol_sha256": protocol_digest,
                  "checkpoint_sha256": protocol["checkpoint_sha256"], "seed": seed,
                  "objective": protocol["objective"], "invocation_wall_seconds": time.perf_counter() - started,
                  "total_recorded_rollout_wall_seconds": sum(row["total_wall_seconds"] for row in results),
                  "verified_reused_policy_trajectories": len(recovered),
                  "records": results, "summary": summarize(results, protocol["trajectory_ids"], args.policies)}
        atomic_json(result_path, result)
        atomic_json(args.output_dir / "status.json", {"state": "complete", "result_sha256": sha256(result_path),
                    "verified_reused_policy_trajectories": len(recovered)})
    except BaseException as error:
        atomic_json(args.output_dir / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                    "error_type": type(error).__name__, "error": str(error), "completed_policy_trajectories": len(results)})
        raise


if __name__ == "__main__":
    main()
