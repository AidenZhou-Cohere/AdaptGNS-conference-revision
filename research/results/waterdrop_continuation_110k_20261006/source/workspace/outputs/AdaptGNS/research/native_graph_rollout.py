"""Separate exploratory full rollout with native capped base and self edges.

The fixed full_rollout.py is unchanged. This module reuses its checkpoint,
failure, boundary, publication, locking and selection helpers, but owns the
native directed graph intervention and rollout loop. No launch occurs at import.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from research import full_rollout as full
from research import graph_convention_bridge as bridge

PROTOCOL = Path(__file__).parent / "protocols/native_graph_rollout_20261005.md"
POLICIES = full.POLICIES
DEADLINE = "2026-10-07T08:00:00+00:00"


def native_graph(history, policy, cached_risk, rng):
    """Preserve native directed base as an ordered prefix; append annulus edges.

    The native receiver cap applies only to the mandatory base. Optional pairs
    are geometric annulus pairs, not within-r neighbors excluded by the cap.
    Their two orientations are uncapped and never remove a native edge.
    """
    graph = bridge.strict_pairs(history[-1], .015)
    native_base = bridge.ordered_edges(history[-1], graph.base, True, 128)
    uncapped_base = bridge.ordered_edges(history[-1], graph.base, True)
    selected = full.choose_pairs(graph, policy, history, cached_risk, rng)
    if not np.array_equal(selected[:len(graph.base)], graph.base):
        raise RuntimeError("Selector changed the geometric base prefix")
    optional = bridge.same.canonical_pairs(selected[len(graph.base):])
    expected = 0 if policy == "base" else len(graph.extra) if policy == "dense" else int(.25 * len(graph.extra))
    if len(optional) != expected or not bridge.same.pair_set(optional) <= bridge.same.pair_set(graph.extra):
        raise RuntimeError("Exact optional annulus budget differs")
    extra_edges = bridge.ordered_edges(history[-1], optional, False)
    edges = np.concatenate((native_base, extra_edges), axis=1)
    if edges.shape[1] != native_base.shape[1] + 2 * expected:
        raise RuntimeError("Optional directed edge count differs")
    native_set = set(map(tuple, native_base.T.tolist()))
    nonself = {(a, b) for a, b in native_set if a != b}
    asymmetric = sum((b, a) not in nonself for a, b in nonself)
    degree = np.bincount(uncapped_base[1], minlength=graph.n_nodes)
    native_degree = np.bincount(native_base[1], minlength=graph.n_nodes)
    audit = {"candidate_pairs": len(graph.base) + len(graph.extra), "geometric_base_pairs": len(graph.base),
        "available_annulus_pairs": len(graph.extra), "optional_pair_budget": int(.25 * len(graph.extra)),
        "retained_optional_pairs": len(optional), "directed_edges": edges.shape[1],
        "native_base_directed_edges": native_base.shape[1], "native_base_self_edges": int(np.sum(native_base[0] == native_base[1])),
        "native_base_receivers_above_cap_before_capping": int(np.sum(degree > 128)),
        "native_base_edges_removed_by_cap": uncapped_base.shape[1] - native_base.shape[1],
        "native_base_max_receiver_degree": int(native_degree.max()),
        "native_base_asymmetric_directed_edges": asymmetric,
        "native_base_sha256": full.state_hash(native_base), "selected_optional_pair_sha256": full.state_hash(optional),
        "directed_edge_sha256": full.state_hash(edges), "native_base_prefix_preserved": bool(np.array_equal(edges[:, :native_base.shape[1]], native_base))}
    return graph, edges, optional, audit


@torch.no_grad()
def rollout(model, positions, particle_types, metadata, policy, horizon, rng_seed, device="cpu", trace_steps=full.TRACE_STEPS):
    if policy not in POLICIES or horizon < 1 or len(positions) < horizon + 6:
        raise ValueError("Known policy and full input/target horizon required")
    if positions.ndim != 3 or positions.shape[2] != 2 or float(model._connectivity_radius) != .015:
        raise ValueError("Native bridge requires the original 2D radius .015 model")
    types = np.asarray(particle_types, dtype=np.int64)
    if types.ndim == 0:
        types = np.full(positions.shape[1], types, dtype=np.int64)
    if types.shape != (positions.shape[1],) or np.any(types == 3):
        raise ValueError("One nonkinematic type per WaterDrop particle required")
    bounds = np.asarray(metadata["bounds"], dtype=np.float64)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Finite ordered 2D bounds required")
    if float(metadata["default_connectivity_radius"]) != .015 or not np.array_equal(np.asarray(model._boundaries), bounds):
        raise ValueError("Model and metadata graph/boundaries differ")
    model.eval()
    history = np.array(positions[:6], dtype=np.float32, copy=True)
    initial_hash = full.state_hash(history)
    all_started = time.perf_counter()
    traces = {"initial_observed_positions": history.copy(), "particle_types": types.copy(), "bounds": bounds}
    initial_failure, initial_phase, parity_passes = None, "initial_state", 0
    parity = {"passed": False, "not_attempted_reason": None}
    try:
        full.check_state(history, "initial_state", bridge.MAX_ABS)
        initial_phase = "initial_graph"
        initial_graph = bridge.strict_pairs(history[-1], .015)
        base_edges = bridge.ordered_edges(history[-1], initial_graph.base, True, 128)
        initial_phase = "native_parity"
        parity, parity_arrays, parity_supplied = bridge.native_parity(model, history, types, base_edges, device)
        parity_passes = 2
        traces.update({f"initial_parity_{key}": value for key, value in parity_arrays.items()})
        traces.update({f"initial_parity_supplied_{key}": parity_supplied[key] for key in ("prediction", "risk", "raw_risk")})
        traces["initial_parity_supplied_edges"] = base_edges
    except full.RolloutGuard as error:
        initial_failure = {**error.details, "forecast_step": 0, "phase": initial_phase}
        parity["not_attempted_reason"] = initial_failure
        traces["failed_input_history"] = history.copy()
        traces["rejected_history"] = history.copy()
    parity_seconds = time.perf_counter() - all_started
    started = time.perf_counter()
    rng = np.random.default_rng(rng_seed)
    failure = initial_failure or (None if parity["passed"] else {"category": "native_parity_failure", "forecast_step": 0, "phase": "native_parity"})
    cached_risk, warmup = None, None
    attempts, mse, edges_per_step, optional_per_step, candidates, geometric_base = [], [], [], [], [], []
    risk_means, predicted_boundary, truth_boundary = [], [], []
    selected_trace, trace_prediction, trace_truth, trace_history = [], [], [], []
    trace_current_risk, trace_cached_risk = [], []
    total_passes = forecast_passes = 0
    graph_seconds = forward_seconds = 0.
    if policy == "laggedrisk25" and failure is None:
        warmup_started = time.perf_counter()
        warmup = {"passes": 0, "mean_prediction_discarded": True, "history_sha256": initial_hash}
        try:
            graph_started = time.perf_counter()
            _, warmup_edges, _, graph_audit = native_graph(history, "base", None, rng)
            warmup["graph_build_and_selection_seconds"] = time.perf_counter() - graph_started
            warmup.update(graph_audit)
            total_passes += 1; warmup["passes"] = 1
            output = bridge.supplied(model, history, types, warmup_edges, device)
            for key in ("prediction", "raw_risk", "risk"):
                traces[f"warmup_{key}"] = output[key]
            warmup["features_forward_decode_and_transfer_seconds"] = output["operational_seconds"]
            bridge.validate_output(output, len(types))
            cached_risk = output["risk"]
        except full.RolloutGuard as error:
            failure = {**error.details, "forecast_step": 0, "phase": "warmup"}
            traces["failed_input_history"] = history.copy()
            traces["rejected_history"] = history.copy()
        warmup["total_wall_seconds"] = time.perf_counter() - warmup_started
    for step in range(1, horizon + 1):
        if failure is not None:
            break
        attempt = {"forecast_step": step, "accepted": False}
        phase = "state"
        try:
            full.check_state(history, "state", bridge.MAX_ABS)
            phase = "graph"
            graph_started = time.perf_counter()
            _, edge, optional, graph_audit = native_graph(history, policy, cached_risk, rng)
            duration = time.perf_counter() - graph_started
            attempt.update(graph_audit, graph_build_and_selection_seconds=duration)
            graph_seconds += duration
            phase = "forward"
            total_passes += 1; forecast_passes += 1
            output = bridge.supplied(model, history, types, edge, device)
            attempt["features_forward_decode_and_transfer_seconds"] = output["operational_seconds"]
            forward_seconds += output["operational_seconds"]
            prediction, risk = output["prediction"], output["risk"]
            if np.isfinite(prediction).all():
                attempt["predicted_boundary"] = full.boundary_metrics(prediction, bounds)
            try:
                bridge.validate_output(output, len(types))
            except full.RolloutGuard:
                for key in ("prediction", "raw_risk", "risk"):
                    traces[f"rejected_{key}"] = output[key]
                traces["rejected_history"] = history.copy()
                traces["rejected_edges"] = edge.copy()
                traces["rejected_optional_pairs"] = optional.copy()
                raise
            # Future truth is read only after the autonomous prediction.
            truth = np.asarray(positions[6 + step - 1], dtype=np.float64)
            if not np.isfinite(truth).all():
                raise ValueError("Ground-truth coordinates are nonfinite")
            error = float(np.mean((prediction.astype(np.float64) - truth) ** 2))
            mse.append(error); edges_per_step.append(edge.shape[1]); optional_per_step.append(len(optional))
            candidates.append(graph_audit["candidate_pairs"]); geometric_base.append(graph_audit["geometric_base_pairs"])
            risk_means.append(float(risk.mean()))
            predicted_boundary.append(attempt["predicted_boundary"])
            truth_boundary.append(full.boundary_metrics(truth, bounds))
            attempt.update(accepted=True, coordinate_mse=error, ground_truth_boundary=truth_boundary[-1],
                           predicted_state_sha256=full.state_hash(prediction))
            if step in trace_steps:
                selected_trace.append(step); trace_prediction.append(prediction.copy()); trace_truth.append(truth.astype(np.float32))
                trace_history.append(history.copy()); trace_current_risk.append(risk.copy())
                trace_cached_risk.append(np.full(len(types), np.nan, dtype=np.float32) if cached_risk is None else cached_risk.copy())
                traces[f"edges_forecast_{step:04d}"] = edge.copy()
                traces[f"optional_pairs_forecast_{step:04d}"] = optional.copy()
            cached_risk = risk
            history = np.concatenate((history[1:], prediction[None]), axis=0)
        except full.RolloutGuard as error:
            failure = {**error.details, "forecast_step": step, "phase": phase}
            attempt["failure"] = failure
            traces["failed_input_history"] = history.copy()
            traces.setdefault("rejected_history", history.copy())
            if cached_risk is not None:
                traces["failed_input_cached_risk"] = cached_risk.copy()
        attempts.append(attempt)
    full.synchronize(device)
    complete = failure is None and len(mse) == horizon
    if warmup:
        graph_seconds += warmup.get("graph_build_and_selection_seconds", 0.)
        forward_seconds += warmup.get("features_forward_decode_and_transfer_seconds", 0.)
    row = {"status": "complete" if complete else "failed", "failure": failure, "policy": policy,
        "horizon": horizon, "completed_steps": len(mse), "rng_seed": int(rng_seed), "warmup": warmup,
        "native_parity": parity, "native_parity_network_passes": parity_passes, "native_parity_operational_seconds": parity_seconds,
        "total_network_passes": total_passes, "forecast_network_passes": forecast_passes,
        "mse_at_steps": {str(s): mse[s-1] if len(mse) >= s else None for s in full.TRACE_STEPS if s <= horizon},
        "mse_at_final_horizon": mse[-1] if complete else None,
        "mean_rollout_mse": float(np.mean(mse)) if complete else None,
        "mean_directed_edges": float(np.mean(edges_per_step)) if complete else None,
        "prefix_mean_mse_if_failed": float(np.mean(mse)) if mse and not complete else None,
        "total_wall_seconds": time.perf_counter() - started,
        "total_wall_seconds_including_native_parity": time.perf_counter() - all_started,
        "graph_build_and_selection_seconds_including_warmup": graph_seconds,
        "features_forward_decode_and_transfer_seconds_including_warmup": forward_seconds,
        "initial_observed_state_sha256": initial_hash, "final_predicted_state_sha256": full.state_hash(history[-1]),
        "mse_per_step": mse, "directed_edges_per_step": edges_per_step,
        "retained_optional_pairs_per_step": optional_per_step, "candidate_pairs_per_step": candidates,
        "base_pairs_per_step": geometric_base, "mean_normalized_acceleration_variance_per_step": risk_means,
        "initial_observed_boundary": [full.boundary_metrics(frame, bounds) for frame in positions[:6]] if np.isfinite(positions[:6]).all() else None,
        "predicted_boundary_per_step": predicted_boundary, "ground_truth_boundary_per_step": truth_boundary,
        "attempts": attempts}
    traces.update(forecast_steps=np.asarray(selected_trace, dtype=np.int64),
        predicted_positions=np.stack(trace_prediction) if trace_prediction else np.empty((0, len(types), 2), dtype=np.float32),
        ground_truth_positions=np.stack(trace_truth) if trace_truth else np.empty((0, len(types), 2), dtype=np.float32),
        observed_or_predicted_histories=np.stack(trace_history) if trace_history else np.empty((0, 6, len(types), 2), dtype=np.float32),
        current_risk=np.stack(trace_current_risk) if trace_current_risk else np.empty((0, len(types)), dtype=np.float32),
        cached_risk_before_forecast=np.stack(trace_cached_risk) if trace_cached_risk else np.empty((0, len(types)), dtype=np.float32))
    return row, traces


def run(args, protocol, model, trajectories, manifest, selected):
    directory = args.output_dir
    if any(directory.glob("*.tmp")):
        raise ValueError("Existing temporary artifact requires preservation/review before publication or recovery")
    protocol_path, result_path = directory / "protocol.json", directory / "result.json"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing run requires resume and exactly matching provenance")
    elif args.resume or any(directory.glob("*.json")) or any(directory.glob("*.npz")):
        raise ValueError("Separate new output or original resume protocol required")
    else:
        full.atomic_json(protocol_path, protocol)
    digest = full.sha256(protocol_path)
    indexed = {}
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old["protocol_sha256"] != digest:
            raise ValueError("Previous result protocol differs")
        indexed = {(row["source_index"], row["policy"]): row for row in old["records"]}
        if len(indexed) != len(old["records"]):
            raise ValueError("Duplicate saved records")
    recovered, records = {}, []
    for index in selected:
        for policy in POLICIES:
            row_path = directory / f"trajectory_{index:06d}_{policy}.json"
            trace_path = row_path.with_suffix(".npz")
            if trace_path.exists() and not row_path.exists():
                raise ValueError("Orphan trace requires preservation/review")
            row = full.recover_record(directory, index, policy, manifest["records"][index]["id"], digest, 995, indexed.get((index, policy)))
            if row:
                if not args.resume:
                    raise ValueError("Existing trajectory requires explicit resume")
                if row["failure"] and row["failure"]["category"] == "native_parity_failure":
                    raise RuntimeError("Retained native parity failure requires review; no retry")
                recovered[index, policy] = row; records.append(row)
    if set(indexed) - set(recovered):
        raise ValueError("Saved index contains missing/unexpected outcomes")
    started = time.perf_counter()
    full.atomic_json(directory / "status.json", {"state": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "reused_outcomes": len(recovered)})
    try:
        for index in selected:
            for policy in POLICIES:
                if (index, policy) in recovered:
                    continue
                if datetime.now(timezone.utc) >= datetime.fromisoformat(DEADLINE):
                    raise TimeoutError("Research cutoff reached before next trajectory; partial outcomes preserved")
                positions, types = trajectories[index]
                row, trace = rollout(model, positions, types, manifest["metadata"], policy, 995,
                                     93000 + 1000 * protocol["seed"] + index, args.device)
                trace_path = directory / f"trajectory_{index:06d}_{policy}.npz"
                temporary = trace_path.with_suffix(".npz.tmp")
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, **trace); stream.flush(); os.fsync(stream.fileno())
                temporary.replace(trace_path)
                row.update(trajectory_id=manifest["records"][index]["id"], source_index=index, n_particles=positions.shape[1],
                           trace_file=trace_path.name, trace_sha256=full.sha256(trace_path), protocol_sha256=digest)
                row_path = trace_path.with_suffix(".json")
                full.atomic_json(row_path, row)
                records.append(full.compact_record(row, row_path))
                full.atomic_json(result_path, {"state": "partial", "protocol_sha256": digest, "records": records,
                    "summary": full.summarize(records, protocol["trajectory_ids"])})
                full.atomic_json(directory / "status.json", {"state": "running", "pid": os.getpid(), "committed_outcomes": len(records),
                    "last_source_index": index, "last_policy": policy, "updated_utc": datetime.now(timezone.utc).isoformat()})
                print(json.dumps({"source_index": index, "policy": policy, "status": row["status"], "completed_steps": row["completed_steps"]}), flush=True)
                if row["failure"] and row["failure"]["category"] == "native_parity_failure":
                    raise RuntimeError("Native parity failed; raw unsuccessful outputs preserved for review")
        for path, expected_hash in protocol["input_files_sha256"].items():
            if full.sha256(path) != expected_hash:
                raise RuntimeError("Pinned input/source bytes changed during evaluation")
        records.sort(key=lambda row: (row["source_index"], POLICIES.index(row["policy"])))
        full.atomic_json(result_path, {"state": "complete", "scope": protocol["scope"], "protocol_sha256": digest,
            "checkpoint_sha256": protocol["checkpoint_sha256"], "objective": protocol["objective"], "seed": protocol["seed"],
            "invocation_wall_seconds": time.perf_counter() - started, "records": records,
            "summary": full.summarize(records, protocol["trajectory_ids"])})
        full.atomic_json(directory / "status.json", {"state": "complete", "result_sha256": full.sha256(result_path), "completed_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        state = "deadline_stopped" if isinstance(error, TimeoutError) else "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        full.atomic_json(directory / "status.json", {"state": state, "error_type": type(error).__name__, "error": str(error),
            "committed_outcomes": len(records), "updated_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--training-protocol", type=Path, default=full.REPO / "research/protocols/full_waterdrop_100k.md")
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    args.output_dir = args.output_dir.resolve()
    for protected in (args.checkpoint.resolve().parent.parent, args.test_manifest.resolve().parent):
        if args.output_dir == protected or protected in args.output_dir.parents or args.output_dir in protected.parents:
            parser.error("A separate exploratory output tree is required")
    torch.set_num_threads(args.threads)
    device = full.resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with full.RunLock(args.output_dir, clear_stale=args.clear_stale_lock):
        manifest = json.loads(args.test_manifest.read_text())
        selected = full.select_records(manifest, "locked_test")
        trajectories = full.load_manifest_data(args.test_manifest, verify_hashes=True)
        metadata_path = args.test_manifest.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata != manifest["metadata"]:
            raise ValueError("Manifest/file metadata differs")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        full.check_checkpoint(payload, "locked_test", args.training_protocol, full.sha256(metadata_path))
        for relative, digest in payload["run_config"]["source_sha256"].items():
            if full.sha256(full.REPO / relative) != digest:
                raise ValueError("Original frozen training source differs")
        model, provenance = full.load_for_evaluation(args.checkpoint, metadata, device, radius_backend="scipy_host")
        if model._max_num_neighbors != 128:
            raise ValueError("Native base requires original neighbor cap128")
        paths = [Path(__file__), Path(bridge.__file__), Path(full.__file__), Path(bridge.same.__file__),
                 full.REPO / "research/budget_graph.py", full.REPO / "research/full_training.py",
                 args.protocol, args.training_protocol, bridge.PROTOCOL, args.checkpoint, args.test_manifest, metadata_path]
        paths += sorted((full.REPO / "adaptive-gns/gns").glob("*.py"))
        for record in manifest["records"]:
            paths += [args.test_manifest.parent / record[name]["path"] for name in ("positions", "particle_types")]
        protocol = {"schema": 1, "scope": "post-inspection exploratory native-graph autonomous bridge; not locked confirmation",
            "objective": payload["run_config"]["objective"], "seed": payload["run_config"]["seed"],
            "checkpoint_sha256": full.sha256(args.checkpoint), "checkpoint_provenance": provenance,
            "run_config_sha256": payload["run_config_sha256"], "native_bridge_protocol_sha256": full.sha256(args.protocol),
            "trajectory_ids": [manifest["records"][index]["id"] for index in selected], "source_indices": selected,
            "source_tfrecord": manifest["source"], "policies": list(POLICIES), "horizon": 995,
            "graph": "strict native float32 radius; native capped128 directed base+self-edge candidates; preserve base ordered prefix; append uncapped symmetric selected geometric annulus pairs; R=1.267r",
            "random_seed": "93000+1000*training_seed+source_index; same initial material as locked rollout; geometry-dependent subsequent draws differ",
            "risk": "one explicit native-base warmup, forecast1 exact budget, later cached risk from previous own selected graph and predicted history",
            "native_parity": "two extra verification passes per trajectory before policy timer; raw arrays saved; failure stops model job",
            "guards": {"max_candidate_pairs": bridge.MAX_PAIRS, "max_abs_coordinate": bridge.MAX_ABS},
            "trace_steps": list(full.TRACE_STEPS), "deadline_utc": DEADLINE, "runtime": full.runtime_provenance(device, "scipy_host"),
            "threads": args.threads, "software": {"numpy": np.__version__, "scipy": full.scipy.__version__, "python": full.platform.python_version()},
            "time_scope": "operational synchronized durations; fixed policy order and changing geometry; no causal speed comparison; native verification separately reported",
            "input_files_sha256": {str(path.resolve()): full.sha256(path) for path in paths}}
        run(args, protocol, model, trajectories, manifest, selected)


if __name__ == "__main__":
    main()
