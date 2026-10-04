#!/usr/bin/env python3
"""Equal-state, exact-budget diagnostics for a full LearnedSimulator checkpoint.

All policies receive the same ground-truth histories. This is a teacher-forced
allocation experiment, not a free rollout or a production GPU timing benchmark.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "adaptive-gns"))

from gns import data_loader, reading_utils
from gns.model_io import load_for_evaluation
from research.budget_graph import candidates, select_pairs, random_pairs, directed


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def predict_pairs(simulator, history, particle_types, material, pairs):
    """Use uncapped undirected radius pairs, both directions, plus self-loops."""
    edge_index = directed(pairs)
    self_edges = np.tile(np.arange(len(history)), (2, 1))
    edge_index = np.concatenate((edge_index, self_edges), axis=1)
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=history.device)
    prediction, head = simulator._forward_with_edge_index(
        history, [len(history)], particle_types, edges[0], edges[1], material)
    return prediction, head, edge_index.shape[1]


@torch.no_grad()
def evaluate_state(simulator, history, target, particle_types, material,
                   previous_risk, rng, radius_factor=1.267,
                   extra_fraction=.25, extra_budget=None):
    """Measure allocation at identical positions and a common pair budget."""
    device = history.device
    dynamic = particle_types != 3
    if not bool(dynamic.any()):
        raise ValueError("Allocation evaluation requires a dynamic particle")
    synchronize(device)
    start = time.perf_counter()
    graph = candidates(history[:, -1].cpu().numpy(), simulator._connectivity_radius,
                       radius_factor)
    graph_seconds = time.perf_counter() - start
    budget = min(len(graph.extra), math.floor(extra_fraction * len(graph.extra))
                 if extra_budget is None else extra_budget)
    speed = torch.linalg.vector_norm(history[:, -1] - history[:, -2], dim=-1).cpu().numpy()
    selections = {
        "base": lambda: graph.base,
        "full": lambda: np.concatenate((graph.base, graph.extra)),
        "random": lambda: random_pairs(graph, budget, rng),
        "speed": lambda: select_pairs(graph, speed, budget),
        "lagged_risk": lambda: select_pairs(graph, previous_risk, budget),
    }
    rows, next_risk = {}, None
    # Rotate evaluation order to avoid always charging cold caches to one policy.
    policies = list(selections)
    for policy in rng.permutation(policies):
        start = time.perf_counter()
        pairs = selections[policy]()
        selection_seconds = time.perf_counter() - start
        synchronize(device)
        start = time.perf_counter()
        prediction, head, count = predict_pairs(simulator, history, particle_types, material, pairs)
        synchronize(device)
        forward_seconds = time.perf_counter() - start
        error = (prediction[dynamic] - target[dynamic]).square().mean().item()
        if not math.isfinite(error):
            raise ValueError(f"Nonfinite prediction for policy {policy}")
        rows[policy] = {
            "position_mse": error, "directed_edges_including_self": count,
            "incremental_undirected_pairs": len(pairs) - len(graph.base),
            "selection_seconds": selection_seconds,
            "forward_with_features_seconds": forward_seconds,
            "reference_step_seconds": graph_seconds + selection_seconds + forward_seconds,
        }
        if policy == "lagged_risk":
            next_risk = head.cpu().numpy()
    matched_counts = [rows[p]["directed_edges_including_self"]
                      for p in ("random", "speed", "lagged_risk")]
    if len(set(matched_counts)) != 1:
        raise AssertionError("Allocation controls must have identical edge budgets")
    return {"base_pairs": len(graph.base), "candidate_annulus_pairs": len(graph.extra),
            "extra_pair_budget": budget, "candidate_build_seconds": graph_seconds,
            "policies": rows}, next_risk


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--start_step", type=int, default=2,
                        help="One-based forecast step after six history frames; >=2 for lag warmup")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius_factor", type=float, default=1.267)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--extra_fraction", type=float, default=.25)
    budgets.add_argument("--extra_budget", type=int, default=None)
    parser.add_argument("--normalization_noise_std", type=float, default=None)
    parser.add_argument("--nmessage_passing_steps", type=int, default=None)
    args = parser.parse_args()
    if (args.start_step < 2 or args.max_steps < 1 or not 0 <= args.extra_fraction <= 1
            or (args.extra_budget is not None and args.extra_budget < 0)
            or (args.max_trajectories is not None and args.max_trajectories < 1)):
        parser.error("Invalid start step, limits or pair budget")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = reading_utils.read_metadata(str(args.data_path), "rollout")
    simulator, provenance = load_for_evaluation(
        args.checkpoint, metadata, device,
        normalization_noise_std=args.normalization_noise_std,
        nmessage_passing_steps=args.nmessage_passing_steps)
    data_file = args.data_path / (args.split + ".npz")
    loader = data_loader.get_data_loader_by_trajectories(data_file)
    rng = np.random.default_rng(args.seed)
    records = []
    with torch.no_grad():
        for trajectory_id, features in enumerate(loader):
            if args.max_trajectories is not None and trajectory_id >= args.max_trajectories:
                break
            positions, types = features[0].to(device), features[1].to(device)
            material = features[2].to(device) if len(features) == 4 else None
            first = 6 + args.start_step - 1
            if first >= positions.shape[1]:
                continue
            # Initialize lag on the preceding true state using the base graph.
            history = positions[:, first - 7:first - 1]
            warm_graph = candidates(history[:, -1].cpu().numpy(),
                                    simulator._connectivity_radius, args.radius_factor)
            _, previous, _ = predict_pairs(simulator, history, types, material, warm_graph.base)
            previous_risk = previous.cpu().numpy()
            for target_index in range(first, min(positions.shape[1], first + args.max_steps)):
                row, previous_risk = evaluate_state(
                    simulator, positions[:, target_index - 6:target_index],
                    positions[:, target_index], types, material, previous_risk, rng,
                    args.radius_factor, args.extra_fraction, args.extra_budget)
                row.update(trajectory_id=trajectory_id, target_frame=target_index,
                           forecast_step=target_index - 5)
                records.append(row)
    if not records:
        raise ValueError("No eligible trajectory windows")
    result = {
        "protocol": {
            "evaluation": "teacher_forced_equal_state_allocation",
            "split": args.split, "seed": args.seed,
            "radius_factor": args.radius_factor, "extra_fraction": args.extra_fraction,
            "extra_budget": args.extra_budget,
            "graph": "SciPy CPU uncapped pairs at <= radius; both directions and self-loops",
            "graph_caveat": "Reference graph can differ from strict-radius, capped PyG at boundaries or high density",
            "lag": "previous true state, risk predicted on the risk policy's previous graph; initial base warmup",
            "timing": "synchronized graph search + selection + tensor/features/forward; CPU reference, warmup excluded",
            "claim_limit": "One-step position error at fixed true states; no long-horizon or production speed claim",
        },
        "checkpoint_sha256": sha256(args.checkpoint), "data_sha256": sha256(data_file),
        "provenance": provenance, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved {len(records)} equal-state windows to {args.output}")


if __name__ == "__main__":
    main()
