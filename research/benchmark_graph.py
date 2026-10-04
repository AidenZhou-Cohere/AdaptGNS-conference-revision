#!/usr/bin/env python3
"""Same-state CPU graph-construction benchmark; no model or learned risk scores.

The implementation under test is research/budget_graph.py (SciPy cKDTree).
This measures graph construction and pair selection, not GNN/rollout runtime,
and does not benchmark the original torch_cluster GPU implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy

from budget_graph import candidates, directed, random_pairs, select_pairs


METHODS = (
    "fixed_base",
    "fixed_dense",
    "exact_optional25_random",
    "exact_optional25_synthetic_score",
    "percentile70_synthetic_score",
)
LABELS = {
    "fixed_base": "Fixed r=0.015",
    "fixed_dense": "Fixed r=0.019005",
    "exact_optional25_random": "25% optional pairs, random",
    "exact_optional25_synthetic_score": "25% optional pairs, synthetic score",
    "percentile70_synthetic_score": "70th-percentile node expansion, synthetic score",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_scores(position: np.ndarray) -> np.ndarray:
    """Arbitrary spatial hotspot; not fitted and not interpreted as true risk."""
    span = np.ptp(position, axis=0)
    scaled = (position - position.min(axis=0)) / np.maximum(span, 1e-12)
    center = np.asarray([0.65, 0.8])
    return np.exp(-np.sum((scaled - center) ** 2, axis=1) / (2 * 0.2**2))


def build(method: str, position: np.ndarray, score: np.ndarray, radius: float, factor: float, rng: np.random.Generator) -> np.ndarray:
    if method == "fixed_base":
        return directed(candidates(position, radius, 1.0).base)
    if method == "fixed_dense":
        return directed(candidates(position, radius * factor, 1.0).base)
    graph = candidates(position, radius, factor)
    # Matches the pilot: budget is 25% OF OPTIONAL ANNULUS PAIRS,
    # not 25% of the base graph and not 25% of the nodes.
    budget = int(np.floor(0.25 * len(graph.extra)))
    if method == "exact_optional25_random":
        pairs = random_pairs(graph, budget, rng)
    elif method == "exact_optional25_synthetic_score":
        pairs = select_pairs(graph, score, budget)
    elif method == "percentile70_synthetic_score":
        threshold = np.quantile(score, 0.70)
        high = score > threshold
        keep = high[graph.extra[:, 0]] | high[graph.extra[:, 1]]
        pairs = np.concatenate((graph.base, graph.extra[keep]), axis=0)
    else:
        raise ValueError(method)
    return directed(pairs)


def describe_ms(values: list[float]) -> dict:
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "median_ms": float(median),
        "q25_ms": float(q25),
        "q75_ms": float(q75),
        "iqr_ms": float(q75 - q25),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "repeats": len(values),
        "raw_ms": values,
    }


def validate_edges(edges: np.ndarray, expected_base: set[tuple[int, int]], allowed: set[tuple[int, int]], expected_pairs: int | None) -> int:
    if edges.shape[0] != 2 or edges.shape[1] % 2:
        raise ValueError("Expected symmetric directed edge array")
    directed_set = set(map(tuple, edges.T.tolist()))
    if len(directed_set) != edges.shape[1] or any(i == j for i, j in directed_set):
        raise ValueError("Duplicate edges or unexpected self loops")
    if any((j, i) not in directed_set for i, j in directed_set):
        raise ValueError("Graph is not symmetric")
    pairs = {(min(i, j), max(i, j)) for i, j in directed_set}
    if not expected_base <= pairs <= allowed:
        raise ValueError("Graph violates mandatory/allowed pairs")
    if expected_pairs is not None and len(pairs) != expected_pairs:
        raise ValueError("Exact pair budget was not satisfied")
    return len(pairs)


def report_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Same-state CPU reference graph benchmark",
        "",
        "This is a fresh timing experiment on nine fixed observed WATERDROP states. It measures the complete SciPy reference graph build and pair selection, including conversion to symmetric directed edges. It does **not** measure a GNN, autoregressive rollout, original torch-cluster graph builder, GPU implementation, learned-risk inference, or end-to-end simulation speedup.",
        "",
        "## Protocol",
        "",
        f"- Three official test-prefix trajectories, recorded frame indices 100, 400, and 700 (zero-based indices in each 1001-frame array). {result['protocol']['warmups_per_state_method']} warmup calls per method per state and {result['protocol']['repeats_per_state_method']} timed calls per method per state.",
        "- Every method receives identical positions within a state. cKDTree is rebuilt on each call. Input arrays and synthetic node scores are prepared before timing; tree construction, neighbor queries, lexicographic sorting, optional-pair filtering, policy selection, allocations, and directed-edge output are timed.",
        "- Base radius r=0.015; candidate radius R=1.267r=0.019005. No self-loops; each retained undirected pair is emitted twice. All particle types are included in construction.",
        "- Exact-budget methods retain K=floor(0.25 × number of optional annulus pairs), matching the pilot. They retain all base pairs. The percentile rule selects optional pairs incident to at least one node above the 70th percentile; it has no exact edge-count budget.",
        "- The synthetic score is a fixed Gaussian-shaped hotspot in normalized scene coordinates. It is neither trained nor an estimate of actual model error. Random pair permutation is generated inside the timed selection call; synthetic score computation is outside timing as an already available controller input.",
        "- State order and method order within each repeat are randomized with a recorded seed. Garbage collection and normal host scheduling are unchanged. The machine is not isolated from other work; medians and interquartile ranges summarize timing variability and do not remove resource-contention effects.",
        "- Untimed validation checks base preservation, candidate containment, symmetry, absence of duplicates/self-loops, and exact retained-pair counts for the budgeted rules on every state.",
        f"- Platform: {result['platform']['system']} {result['platform']['release']}, {result['platform']['machine']}; CPU {result['platform']['cpu_model']}; Python {result['platform']['python']}; NumPy {result['platform']['numpy']}; SciPy {result['platform']['scipy']}.",
        "- Source NPZ and both benchmark/reference scripts are identified by SHA-256 in `research/results/graph_benchmark.json`; raw per-call durations are saved there.",
        "",
        "## Construction and selection timings",
        "",
        "Times are milliseconds. Brackets give [25th percentile, 75th percentile], not a confidence interval. Pair counts are undirected; directed counts are exactly twice these values.",
        "",
        "| Trajectory | Frame | Nodes | Policy | Candidate pairs queried | Retained pairs | Median ms [Q25, Q75] |",
        "|---:|---:|---:|---|---:|---:|---|",
    ]
    for state in sorted(result["states"], key=lambda s: (s["trajectory_index"], s["recorded_frame_index_zero_based"])):
        for method in METHODS:
            timing = state["methods"][method]
            lines.append(f"| {state['trajectory_index']} | {state['recorded_frame_index_zero_based']} | {state['n_nodes']} | {LABELS[method]} | {timing['candidate_pairs_queried']} | {timing['retained_undirected_pairs']} | {timing['median_ms']:.5f} [{timing['q25_ms']:.5f}, {timing['q75_ms']:.5f}] |")
    lines += [
        "",
        "## Relative construction overhead",
        "",
        "Each ratio divides a method's median by the base-radius median on the same state; the table summarizes those nine ratios. It does not estimate a ratio of full-simulation runtimes.",
        "",
        "| Method | Median of nine state ratios | Smallest state ratio | Largest state ratio |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        summary = result["relative_to_base_same_state"] [method]
        lines.append(f"| {LABELS[method]} | {summary['median_ratio']:.3f}× | {summary['min_ratio']:.3f}× | {summary['max_ratio']:.3f}× |")
    lines += [
        "",
        "A retained-edge budget limits subsequent message-passing work, but this implementation first enumerates the entire larger-radius candidate graph. It therefore still pays for that search, filtering, and selection. Any claim of an end-to-end speedup requires the measured savings in the GNN to outweigh this added cost; the present experiment cannot establish that outcome.",
        "",
        "## Reproduction",
        "",
        "From the repository root, run `python research/benchmark_graph.py --data ../../work/data/test-pilot.npz`. The command needs NumPy and SciPy. Default outputs are `research/results/graph_benchmark.json` and the sibling `graph_benchmark.md`. Timings are intentionally machine- and load-dependent; graph counts and seeded selections are reproducible for matching versions and data.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.repeats < 20 or args.warmups < 1:
        parser.error("At least 20 timed repeats and one warmup are required")
    repo = Path(__file__).resolve().parents[1]
    output = args.output or repo / "research/results/graph_benchmark.json"
    report = args.report or repo.parent / "graph_benchmark.md"
    data_path = args.data.resolve()
    radius, factor = 0.015, 1.267
    with np.load(data_path, allow_pickle=False) as archive:
        states = []
        for trajectory in range(3):
            source = archive[f"position_{trajectory}"]
            types = archive[f"type_{trajectory}"]
            for frame in (100, 400, 700):
                position = np.ascontiguousarray(source[frame], dtype=np.float64)
                if position.ndim != 2 or position.shape[1] != 2 or not np.isfinite(position).all():
                    raise ValueError("Expected finite 2D positions")
                states.append((trajectory, frame, position, int(np.count_nonzero(types == 3))))
    order_rng = np.random.default_rng(args.seed)
    load_start = os.getloadavg() if hasattr(os, "getloadavg") else None
    state_results = []
    for state_index in order_rng.permutation(len(states)):
        trajectory, frame, position, kinematic_count = states[int(state_index)]
        score = synthetic_scores(position)
        graph = candidates(position, radius, factor)
        base_set = set(map(tuple, graph.base.tolist()))
        all_set = base_set | set(map(tuple, graph.extra.tolist()))
        budget = int(np.floor(0.25 * len(graph.extra)))
        # Independent streams prevent random-policy draws from changing trial order.
        method_rng = {m: np.random.default_rng(args.seed + 1000 * trajectory + frame + k) for k, m in enumerate(METHODS)}
        counts = {}
        for method in METHODS:
            edges = build(method, position, score, radius, factor, method_rng[method])
            expected = len(base_set) if method == "fixed_base" else len(all_set) if method == "fixed_dense" else len(base_set) + budget if method.startswith("exact_") else None
            counts[method] = validate_edges(edges, base_set, all_set, expected)
        for _ in range(args.warmups):
            for method in order_rng.permutation(METHODS):
                build(str(method), position, score, radius, factor, method_rng[str(method)])
        elapsed = {m: [] for m in METHODS}
        for _ in range(args.repeats):
            for method in order_rng.permutation(METHODS):
                method = str(method)
                start = time.perf_counter_ns()
                edges = build(method, position, score, radius, factor, method_rng[method])
                duration_ms = (time.perf_counter_ns() - start) / 1e6
                elapsed[method].append(float(duration_ms))
                # Size check intentionally outside the timed interval.
                if edges.shape[1] != 2 * counts[method]:
                    raise ValueError("Retained count changed between repeats")
        state_results.append({
            "trajectory_index": trajectory,
            "recorded_frame_index_zero_based": frame,
            "n_nodes": len(position),
            "n_kinematic_nodes": kinematic_count,
            "position_float64_sha256": hashlib.sha256(position.tobytes()).hexdigest(),
            "base_undirected_pairs": len(base_set),
            "optional_annulus_pairs": len(graph.extra),
            "exact_extra_pair_budget": budget,
            "methods": {
                method: {**describe_ms(elapsed[method]), "retained_undirected_pairs": counts[method], "retained_directed_edges": 2 * counts[method], "candidate_pairs_queried": len(base_set) if method == "fixed_base" else len(all_set)}
                for method in METHODS
            },
        })
    try:
        cpu_model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        cpu_model = platform.processor() or "unavailable"
    result = {
        "experiment_type": "same_observed_state_cpu_reference_graph_build_and_selection",
        "measures_gnn_runtime": False,
        "uses_learned_risk": False,
        "source_data_name": data_path.name,
        "source_data_sha256": sha256(data_path),
        "benchmark_script_sha256": sha256(Path(__file__).resolve()),
        "reference_graph_script_sha256": sha256(Path(__file__).resolve().with_name("budget_graph.py")),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "cpu_model": cpu_model, "logical_cpu_count": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "load_average_start": load_start, "load_average_end": os.getloadavg() if hasattr(os, "getloadavg") else None, "cpu_isolated": False},
        "protocol": {"base_radius": radius, "radius_factor": factor, "candidate_radius": radius * factor, "extra_budget_definition": "floor(0.25 * optional_annulus_pair_count)", "warmups_per_state_method": args.warmups, "repeats_per_state_method": args.repeats, "seed": args.seed, "randomized_state_order": True, "randomized_method_order_each_repeat": True, "includes_tree_rebuild": True, "includes_directed_conversion": True, "self_loops": False, "position_dtype": "float64", "score_description": "fixed synthetic Gaussian hotspot at (0.65,0.8) in scene min-max coordinates, width0.2", "score_construction_timed": False, "random_pair_permutation_timed": True},
        "states": state_results,
        "relative_to_base_same_state": {},
    }
    for method in METHODS:
        ratios = [s["methods"][method]["median_ms"] / s["methods"]["fixed_base"]["median_ms"] for s in state_results]
        result["relative_to_base_same_state"][method] = {"median_ratio": float(np.median(ratios)), "min_ratio": float(min(ratios)), "max_ratio": float(max(ratios)), "n_states": len(ratios)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report_markdown(result, report)
    print(f"Measured {len(state_results)} states, {len(METHODS)} methods, {args.repeats} timed repeats per combination.")
    print(output)
    print(report)
    print(json.dumps(result["relative_to_base_same_state"], indent=2))


if __name__ == "__main__":
    main()
