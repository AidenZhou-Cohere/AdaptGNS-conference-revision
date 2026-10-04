"""Autonomous WaterDrop rollouts for the compact CPU pilot, not a benchmark.

The initial six frames are the only state inputs from ground truth. All later
histories, geometry, speed scores and lagged risk come from the policy's own
predictions. Ground truth is accessed subsequently only to compute error.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from research.budget_graph import candidates, select_pairs, random_pairs
from research.pilot import PilotGNS, feature, graph_tensors


POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
KINDS = ("faithful", "nll", "beta_nll")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """An incomplete process must not leave a valid-looking partial result."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def integrate(history, normalized_acceleration, metadata):
    """Invert the exact standardized second-difference training target.

    The target already uses displacements per stored frame. No extra dt or
    dt-squared factor belongs in this inverse transformation.
    """
    scale = np.asarray(metadata["acc_std"], dtype=np.float32)
    center = np.asarray(metadata["acc_mean"], dtype=np.float32)
    acceleration = normalized_acceleration * scale + center
    return 2 * history[-1] - history[-2] + acceleration


@torch.no_grad()
def rollout(model, positions, metadata, kind, policy, horizon, rng_seed,
            max_candidate_pairs=100000, max_abs_coordinate=10.):
    """One trajectory/policy with explicit failure accounting and prefix data."""
    if positions.shape[0] < horizon + 6:
        raise ValueError("Ground truth is too short for requested forecast horizon")
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy}")
    history = positions[:6].copy()
    rng = np.random.default_rng(rng_seed)
    previous_risk = None
    mse, edges, extra_pairs, candidate_counts = [], [], [], []
    graph_seconds, forward_seconds = [], []
    failure = None
    started = time.perf_counter()
    for step in range(horizon):
        if not np.isfinite(history).all():
            failure = {"category": "nonfinite_state", "forecast_step": step + 1}
            break
        if float(np.max(np.abs(history))) > max_abs_coordinate:
            failure = {"category": "coordinate_resource_guard", "forecast_step": step + 1}
            break
        graph_start = time.perf_counter()
        current = history[-1]
        # Count before materializing pairs, so the resource guard is effective.
        tree = cKDTree(current)
        available = int((tree.count_neighbors(tree, .015 * 1.267) - len(current)) // 2)
        if available > max_candidate_pairs:
            failure = {"category": "candidate_pair_resource_guard", "forecast_step": step + 1,
                       "candidate_pairs": available, "limit": max_candidate_pairs}
            break
        graph = candidates(current, .015, 1.267)
        budget = math.floor(.25 * len(graph.extra))
        if policy == "base" or (policy == "laggedrisk25" and previous_risk is None):
            pairs = graph.base
        elif policy == "dense":
            pairs = np.concatenate((graph.base, graph.extra))
        elif policy == "random25":
            pairs = random_pairs(graph, budget, rng)
        elif policy == "speed25":
            speed = np.linalg.norm(history[-1] - history[-2], axis=-1)
            pairs = select_pairs(graph, speed, budget)
        else:
            if not np.isfinite(previous_risk).all():
                failure = {"category": "nonfinite_lagged_risk", "forecast_step": step + 1}
                break
            pairs = select_pairs(graph, previous_risk, budget)
        graph_seconds.append(time.perf_counter() - graph_start)
        forward_start = time.perf_counter()
        edge_index, edge_features = graph_tensors(current, pairs)
        mean, risk = model(feature(history, metadata), edge_index, edge_features,
                           kind == "faithful")
        forward_seconds.append(time.perf_counter() - forward_start)
        next_position = integrate(history, mean.numpy(), metadata)
        previous_risk = risk.numpy()
        if not np.isfinite(next_position).all() or not np.isfinite(previous_risk).all():
            failure = {"category": "nonfinite_prediction", "forecast_step": step + 1}
            break
        if float(np.max(np.abs(next_position))) > max_abs_coordinate:
            failure = {"category": "coordinate_resource_guard", "forecast_step": step + 1,
                       "maximum_absolute_coordinate": float(np.max(np.abs(next_position))),
                       "limit": max_abs_coordinate}
            break
        residual = next_position.astype(np.float64) - positions[step + 6]
        mse.append(float(np.mean(residual ** 2)))
        edges.append(2 * len(pairs))
        extra_pairs.append(len(pairs) - len(graph.base))
        candidate_counts.append(available)
        history = np.concatenate((history[1:], next_position[None]), axis=0)
    elapsed = time.perf_counter() - started
    complete = failure is None and len(mse) == horizon
    return {
        "status": "complete" if complete else "failed",
        "failure": failure, "horizon": horizon, "completed_steps": len(mse),
        "mse_at_steps": {str(s): mse[s - 1] if len(mse) >= s else None
                         for s in (1, 10, 50, 200, 500, 995) if s <= horizon},
        "mse_at_final_horizon": mse[-1] if complete else None,
        "mean_rollout_mse": float(np.mean(mse)) if complete else None,
        "mean_directed_edges": float(np.mean(edges)) if complete else None,
        "prefix_mean_mse_if_failed": float(np.mean(mse)) if mse and not complete else None,
        "total_wall_seconds": elapsed,
        "graph_build_and_selection_seconds": float(sum(graph_seconds)),
        "features_and_forward_seconds": float(sum(forward_seconds)),
        "final_predicted_state_sha256": hashlib.sha256(history[-1].tobytes()).hexdigest(),
        "mse_per_step": mse, "directed_edges_per_step": edges,
        "incremental_pairs_per_step": extra_pairs,
        "candidate_pairs_per_step": candidate_counts,
    }


def summarize(records, expected_trajectories):
    """Do not turn survivor-only accuracy into an unconditional comparison."""
    result = {}
    for policy in POLICIES:
        rows = [r for r in records if r["policy"] == policy]
        complete = [r for r in rows if r["status"] == "complete"]
        all_complete = len(complete) == expected_trajectories
        result[policy] = {
            "attempted_trajectories": len(rows), "completed_trajectories": len(complete),
            "failed_trajectories": len(rows) - len(complete),
            "mse_at_final_horizon": (float(np.mean([r["mse_at_final_horizon"] for r in complete]))
                                     if all_complete else None),
            "mean_rollout_mse": (float(np.mean([r["mean_rollout_mse"] for r in complete]))
                                 if all_complete else None),
            "mean_directed_edges": (float(np.mean([r["mean_directed_edges"] for r in complete]))
                                    if all_complete else None),
            "total_wall_seconds": float(sum(r["total_wall_seconds"] for r in rows)),
            "mse_at_steps": {
                str(s): (float(np.mean([r["mse_at_steps"][str(s)] for r in rows]))
                         if len(rows) == expected_trajectories and all(
                             r["mse_at_steps"].get(str(s)) is not None for r in rows) else None)
                for s in (50, 200)
            },
        }
    return result


def load_checkpoint(path):
    # Locally produced pilot checkpoints include torch.__version__ in protocol.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint["protocol"]["architecture"]
    model = PilotGNS(width=architecture["latent_width"], depth=architecture["processor_depth"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--objectives", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    parser.add_argument("--max-abs-coordinate", type=float, default=10.)
    parser.add_argument("--wait-checkpoints", action="store_true")
    parser.add_argument("--wait-limit-seconds", type=float, default=3600.)
    args = parser.parse_args()
    if args.horizon < 1 or args.threads < 1 or args.max_candidate_pairs < 1 or args.max_abs_coordinate <= 0:
        parser.error("Positive horizon, thread count and resource guards required")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.data_dir / "metadata.json").read_text())
    data_path = args.data_dir / "test-pilot.npz"
    trajectories = {}
    with np.load(data_path, allow_pickle=False) as data:
        for key in sorted(k for k in data.files if k.startswith("position_")):
            if np.any(data[key.replace("position_", "type_")] == 3):
                raise ValueError("The compact pilot cannot model prescribed kinematic particles")
            trajectories[key] = data[key]
            if len(data[key]) < args.horizon + 6:
                raise ValueError(f"{key} lacks requested ground-truth horizon")
    protocol = {
        "scope": "Autonomous rollout extension of small CPU WaterDrop pilot; not full-paper reproduction",
        "split": "first three official test trajectories", "trajectories": list(trajectories),
        "horizon": args.horizon, "context_frames": 6, "seeds": args.seeds,
        "objectives": args.objectives, "policies": list(POLICIES),
        "integration": "x_next=2*x_current-x_previous + acc_std*normalized_mean + acc_mean; no extra dt",
        "policy": {"base_radius": .015, "radius_factor": 1.267,
                   "allocation_fraction": .25, "candidate_rule": "uncapped undirected pairs within <= radius",
                   "pair_score": "max of endpoint scores; lexicographic tie break",
                   "directions": "both directions, no self-loops",
                   "first_step": "base/dense/random/speed apply immediately; lagged risk uses base then previous actual policy risk",
                   "budget_caveat": "Exact quarter annulus within each predicted state; counts differ across policy rollouts"},
        "failure_rules_predeclared": {
            "nonfinite": "any state, prediction or risk value is nonfinite",
            "max_absolute_coordinate": args.max_abs_coordinate,
            "max_candidate_undirected_pairs": args.max_candidate_pairs,
            "interpretation": "Resource guards are not physical-validity thresholds",
            "reporting": "all failures retained; incomplete trajectory final/mean accuracy null; group accuracy null if any trajectory incomplete"},
        "timing": "Single-thread CPU wall time includes candidate counting/building, selection, features, forward, integration and metrics; concurrent training may affect timing",
        "hardware": platform.platform(), "torch": str(torch.__version__), "threads": args.threads,
        "data_sha256": sha256(data_path), "metadata_sha256": sha256(args.data_dir / "metadata.json"),
        "code_sha256": sha256(__file__),
        "model_source_sha256": sha256(Path(__file__).with_name("pilot.py")),
        "graph_source_sha256": sha256(Path(__file__).with_name("budget_graph.py")),
    }
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output protocol differs; use a new directory")
    write_json(protocol_path, protocol)  # Written before any rollout outcome.
    pending = [(seed, kind) for seed in args.seeds for kind in args.objectives]
    started = time.perf_counter()
    while pending:
        made_progress = False
        for seed, kind in pending[:]:
            name = f"{kind}_seed{seed}"
            checkpoint_path = args.checkpoint_dir / (name + ".pt")
            output_path = args.output_dir / (name + ".json")
            if output_path.exists():
                if checkpoint_path.exists() and json.loads(output_path.read_text())["checkpoint_sha256"] != sha256(checkpoint_path):
                    raise ValueError(f"Checkpoint changed after rollout: {checkpoint_path}")
                pending.remove((seed, kind))
                made_progress = True
                continue
            if not checkpoint_path.exists():
                continue
            model, checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint["seed"] != seed or checkpoint["objective"] != kind:
                raise ValueError("Checkpoint identity mismatch")
            records = []
            checkpoint_start = time.perf_counter()
            for trajectory_index, (trajectory_id, positions) in enumerate(trajectories.items()):
                for policy in POLICIES:
                    row = rollout(model, positions, metadata, kind, policy, args.horizon,
                                  93000 + 100 * seed + trajectory_index,
                                  args.max_candidate_pairs, args.max_abs_coordinate)
                    row.update(trajectory_id=trajectory_id, policy=policy,
                               n_particles=positions.shape[1])
                    records.append(row)
                    print(name, trajectory_id, policy, row["status"], row["completed_steps"],
                          round(row["total_wall_seconds"], 3), flush=True)
            result = {"seed": seed, "objective": kind,
                      "training_steps": checkpoint["protocol"]["steps"],
                      "checkpoint_sha256": sha256(checkpoint_path),
                      "checkpoint_wall_seconds": time.perf_counter() - checkpoint_start,
                      "records": records, "summary": summarize(records, len(trajectories))}
            write_json(output_path, result)
            print("SAVED", output_path.name, round(result["checkpoint_wall_seconds"], 3), flush=True)
            pending.remove((seed, kind))
            made_progress = True
        if pending and not made_progress:
            if not args.wait_checkpoints or time.perf_counter() - started > args.wait_limit_seconds:
                write_json(args.output_dir / "missing_checkpoints.json",
                           {"missing": [f"{kind}_seed{seed}.pt" for seed, kind in pending]})
                raise RuntimeError("Expected checkpoints are not yet available")
            time.sleep(5)
    print("COMPLETE", len(args.seeds) * len(args.objectives), "checkpoints", flush=True)


if __name__ == "__main__":
    main()
