#!/usr/bin/env python3
"""Audit committed rollout arrays without loading models or rerunning simulation.

Only NumPy is required. Intervals describe trajectory sampling conditional on
the saved checkpoints. They are NOT intervals over training seeds. Paired
comparisons assume the same trajectory ordering, which the arrays do not prove.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np


SELECTED_NAME = "adaptive_rollout_mse_model-500000_p70_r1.267"
UPSTREAM_REF = "5cf0d7cb9d927f4277c4538917ae570d8dc9d7b6"
CAUTIONS = [
    "This is a reanalysis of committed arrays, not a fresh simulation or training experiment.",
    "Each artifact represents one checkpoint; training seeds and checkpoint hashes are absent.",
    "Paired resampling assumes row i denotes the same trajectory across artifacts. IDs are absent; shuffle=False in the committed loader supports but does not prove this assumption.",
    "95% percentile bootstrap intervals resample whole trajectories, conditional on these checkpoints and the assumed evaluation population. They are not multi-seed uncertainty estimates.",
    "Intervals are exploratory, pointwise, and unadjusted for multiple models, horizons, or hyperparameter selection.",
    "Upstream adaptive evaluation code at commit 5cf0d7cb9d927f4277c4538917ae570d8dc9d7b6 loads test.npz and exposes no split selector. The Sand grid is therefore reported as a post-hoc test-artifact sensitivity audit; independent validation-only selection is not documented.",
    "Only 314 predicted Sand steps and 995 predicted WaterDrop steps are stored after the six-frame history. MSE@1000 cannot be computed from these artifacts.",
    "Edge counts measure realized rollout graphs on different predicted geometries, not runtime or equal-input graph cost.",
    "WaterDrop dense MSE baseline uses a 400000-step checkpoint; the other WaterDrop artifacts use 500000-step checkpoints.",
    "The paper's Sand MSE-trained r=0.0176 result and WaterDrop MLP result have no matching committed rollout arrays.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: np.ndarray, indices: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite per-trajectory metric values")
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "trajectory_sd": float(values.std(ddof=1)),
        "n_trajectories": int(len(values)),
        "conditional_trajectory_bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def classify(path: Path) -> str:
    if path.stem.startswith("adaptive_rollout"):
        return "adaptive"
    if path.parent.name == "adaptive_gns":
        return "fixed_nll_mid" if "r0.0176" in path.stem else "fixed_nll_base"
    return {"baseline_k5": "fixed_mse_base", "baseline_k10": "fixed_mse_dense", "baseline_mlp": "mlp"}[path.parent.name]


def read_artifact(path: Path, repo: Path, indices: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    mse = np.asarray(data["mse_per_trajectory"], dtype=np.float64)
    edges = np.asarray(data["edge_counts_per_trajectory"], dtype=np.float64)
    steps = data["steps"]
    if mse.ndim != 2 or edges.shape != mse.shape:
        raise ValueError(f"Mismatched shapes: {path}")
    if not np.isfinite(mse).all() or not np.isfinite(edges).all():
        raise ValueError(f"Non-finite values require an explicit censoring policy: {path}")
    if (mse < 0).any() or (edges < 0).any():
        raise ValueError(f"Negative errors or edge counts: {path}")
    if not np.array_equal(steps, np.arange(1, mse.shape[1] + 1)):
        raise ValueError(f"Unexpected horizon indexing: {path}")
    if int(data["n_trajectories"]) != len(mse):
        raise ValueError(f"Trajectory count mismatch: {path}")
    if indices.shape[1] != len(mse):
        raise ValueError(f"Cannot pair different numbers of trajectories: {path}")
    horizon = mse.shape[1]
    wanted = sorted({s for s in (1, 10, 50, 100, 200, 300, 500, 750, 1000, horizon) if s <= horizon})
    tail_start = int(np.floor(0.8 * horizon))
    vectors = {f"mse_at_{s}": mse[:, s - 1] for s in wanted}
    vectors.update({
        "rollout_mean_mse": mse.mean(axis=1),
        "last_20_percent_mean_mse": mse[:, tail_start:].mean(axis=1),
        "mean_edges_per_step": edges.mean(axis=1),
    })
    companion = path.with_suffix(".json")
    summary = json.loads(companion.read_text()) if companion.exists() else {}
    checks = {
        "mse_mean_max_absolute_difference": float(np.max(np.abs(mse.mean(axis=0) - data["mse_mean"]))),
        "mse_std_max_absolute_difference": float(np.max(np.abs(mse.std(axis=0, ddof=0) - data["mse_std"]))),
        "edge_mean_max_absolute_difference": float(np.max(np.abs(edges.mean(axis=0) - data["mean_edges_by_step"]))),
        "global_edge_mean_absolute_difference": float(abs(edges.mean() - data["mean_edges_per_step"])),
        "n_at_step_matches": bool(np.array_equal(data["n_at_step"], np.full(horizon, len(mse)))),
        "json_summary_mse_at_200_absolute_difference": abs(float(summary.get("mse_at_steps", {}).get("200", mse[:, 199].mean())) - float(mse[:, 199].mean())),
    }
    if any(value > 1e-10 for key, value in checks.items() if key != "n_at_step_matches") or not checks["n_at_step_matches"]:
        raise ValueError(f"Saved aggregates disagree with arrays: {path}: {checks}")
    record = {
        "id": path.relative_to(repo).as_posix(),
        "dataset": path.relative_to(repo).parts[0],
        "kind": classify(path),
        "selected_in_paper": path.stem == SELECTED_NAME,
        "sha256": sha256(path),
        "companion_json_sha256": sha256(companion) if companion.exists() else None,
        "checkpoint_filename": summary.get("model_file"),
        "checkpoint_sha256": None,
        "trajectory_ids_present": False,
        "n_trajectories": len(mse),
        "last_predicted_step": horizon,
        "tail_mean_first_included_step": tail_start + 1,
        "sigma_percentile": float(data["sigma_percentile"]) if "sigma_percentile" in data else None,
        "radius_factor": float(data["radius_factor"]) if "radius_factor" in data else None,
        "aggregate_consistency_checks": checks,
        "metrics": {key: describe(value, indices) for key, value in vectors.items()},
        "mse_mean_by_step": mse.mean(axis=0).tolist(),
        "mean_edges_by_step": edges.mean(axis=0).tolist(),
    }
    return record, vectors


def compare(adaptive: dict, baseline: dict, a: dict, b: dict, indices: np.ndarray, independent_indices: np.ndarray) -> dict:
    metrics = {}
    for key in sorted(a.keys() & b.keys()):
        difference = a[key] - b[key]
        result = describe(difference, indices)
        result["adaptive_mean"] = float(a[key].mean())
        result["baseline_mean"] = float(b[key].mean())
        result["adaptive_minus_baseline_relative_percent"] = float(100 * difference.mean() / b[key].mean())
        result["trajectories_adaptive_lower"] = int(np.count_nonzero(difference < 0))
        result["trajectories_tied"] = int(np.count_nonzero(difference == 0))
        samples_a = a[key][indices].mean(axis=1)
        samples_b = b[key][indices].mean(axis=1)
        result["relative_percent_conditional_paired_bootstrap_95ci"] = np.quantile(100 * (samples_a - samples_b) / samples_b, [0.025, 0.975]).tolist()
        independent_b = b[key][independent_indices].mean(axis=1)
        result["unpaired_bootstrap_95ci_pairing_sensitivity_only"] = np.quantile(samples_a - independent_b, [0.025, 0.975]).tolist()
        metrics[key] = result
    return {"dataset": adaptive["dataset"], "adaptive_id": adaptive["id"], "baseline_id": baseline["id"], "baseline_kind": baseline["kind"], "difference_convention": "adaptive minus baseline; negative is lower", "metrics": metrics}


def pareto_ids(records: list[dict], error_key: str) -> list[str]:
    points = [(r["id"], r["metrics"]["mean_edges_per_step"]["mean"], r["metrics"][error_key]["mean"]) for r in records if r["kind"] != "mlp"]
    return [name for name, edge, error in points if not any(e <= edge and m <= error and (e < edge or m < error) for other, e, m in points if other != name)]


def label(record: dict) -> str:
    if record["kind"] == "adaptive":
        return f"Adaptive p={record['sigma_percentile']:g}, R/r={record['radius_factor']:g}"
    return {"fixed_nll_base": "Fixed NLL r=0.015", "fixed_nll_mid": "Fixed NLL r=0.0176", "fixed_mse_base": "Fixed MSE r=0.015", "fixed_mse_dense": "Fixed MSE r=0.019", "mlp": "MLP"}[record["kind"]]


def write_report(result: dict, path: Path) -> None:
    lines = [
        "# Reanalysis of the committed AdaptGNS rollout results",
        "",
        "These results come from saved rollout arrays; no checkpoint was trained or simulation rerun for this audit. The original narrow WATERDROP MSE@200 improvement is uncertain across the 30 saved trajectories, and its sign reverses at longer horizons against the sparse MSE baseline.",
        "",
        "## Statistical scope and provenance",
        "",
        *[f"- {item}" for item in CAUTIONS],
        f"- Bootstrap: {result['bootstrap']['resamples']:,} resamples; NumPy PCG64 seed {result['bootstrap']['seed']}; equal weight per trajectory; all time samples within a trajectory kept together. Paired and pairing-sensitive unpaired intervals are both saved.",
        "- Every input NPZ and companion JSON has a SHA-256 digest in `research/results/reanalysis.json`. All saved aggregate means/std/edge totals were checked against the underlying arrays.",
        "- Prediction-step indices start at 1 after the six input frames. Full-rollout mean MSE is the arithmetic mean over every stored predicted step, then over trajectories; it is not a physical-time integral.",
        "",
        "## Headline comparison: selected adaptive rule versus sparse MSE baseline",
        "",
        "Negative deltas favor Adaptive. Intervals below are **conditional paired trajectory bootstrap intervals**, assuming aligned row ordering, and are not seed statistics.",
        "",
        "| Dataset | Metric | Adaptive | Fixed MSE r=0.015 | Delta | 95% interval for delta | Adaptive wins |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for comparison in result["comparisons"]:
        if comparison["baseline_kind"] != "fixed_mse_base":
            continue
        final_step = 995 if comparison["dataset"] == "WaterDrop" else 314
        keys = ["mse_at_200"] + (["mse_at_500"] if final_step >= 500 else []) + [f"mse_at_{final_step}", "rollout_mean_mse", "mean_edges_per_step"]
        for key in keys:
            m = comparison["metrics"][key]
            lo, hi = m["conditional_trajectory_bootstrap_95ci"]
            lines.append(f"| {comparison['dataset']} | {key} | {m['adaptive_mean']:.9g} | {m['baseline_mean']:.9g} | {m['mean']:+.9g} | [{lo:+.9g}, {hi:+.9g}] | {m['trajectories_adaptive_lower']}/{m['n_trajectories']} |")
    water = next(c for c in result["comparisons"] if c["dataset"] == "WaterDrop" and c["baseline_kind"] == "fixed_mse_base")
    m200 = water["metrics"]["mse_at_200"]
    lines += [
        "",
        f"The unrounded WATERDROP MSE@200 effect is {m200['mean']:+.12g} ({m200['adaptive_minus_baseline_relative_percent']:+.6f}%). Rounding the two means to four decimals makes the difference look larger. The conditional interval contains zero. This does not establish a strict accuracy improvement.",
        "",
        "The long-horizon comparisons are exploratory and use the same selected checkpoint/configuration. They show that the original claim cannot be generalized from MSE@200 to all horizons. The edge reduction is real in the stored rollouts, but cannot be interpreted as a measured runtime reduction or an equal-geometry budget saving.",
        "",
        "## Full saved-horizon summaries",
        "",
        "| Dataset | Model | Predicted steps | MSE@200 | MSE at last step | Mean MSE over rollout | Mean edges/step |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in result["artifacts"]:
        if record["kind"] == "adaptive" and not record["selected_in_paper"]:
            continue
        metrics = record["metrics"]
        last = record["last_predicted_step"]
        lines.append(f"| {record['dataset']} | {label(record)} | {last} | {metrics['mse_at_200']['mean']:.9g} | {metrics[f'mse_at_{last}']['mean']:.9g} | {metrics['rollout_mean_mse']['mean']:.9g} | {metrics['mean_edges_per_step']['mean']:,.3f} |")
    lines += [
        "",
        "The stored Sand MLP edge counts reflect graph construction despite zero message-passing steps; they should not be compared as proportional neural-message cost. The MLP row is retained to expose this implementation detail, not to manufacture an efficiency comparison.",
        "",
        "## Selected adaptive rule versus the fixed NLL checkpoint",
        "",
        "| Dataset | Metric | Delta | Relative change | Conditional paired 95% interval |",
        "|---|---|---:|---:|---|",
    ]
    for comparison in result["comparisons"]:
        if comparison["baseline_kind"] != "fixed_nll_base":
            continue
        for key in ("mse_at_200", "rollout_mean_mse", "mean_edges_per_step"):
            m = comparison["metrics"][key]
            lo, hi = m["conditional_trajectory_bootstrap_95ci"]
            lines.append(f"| {comparison['dataset']} | {key} | {m['mean']:+.9g} | {m['adaptive_minus_baseline_relative_percent']:+.6f}% | [{lo:+.9g}, {hi:+.9g}] |")
    lines += [
        "",
        "A graph-expansion policy cannot have fewer edges than the base graph on the same positions when it preserves all base edges. The reported decreases therefore include changes in the rollout geometry. Save equal-state graph counts and compare policies at a hard incremental edge budget before attributing savings to placement.",
        "",
        "## Post-hoc Sand hyperparameter sensitivity",
        "",
        "This table audits existing test artifacts. It is not a validation sweep, a new hyperparameter search, or evidence for selecting a replacement configuration on the test set. Only one WaterDrop adaptive configuration is committed, so WaterDrop sensitivity cannot be estimated.",
        "",
        "| Percentile | Radius factor | MSE@200 | MSE@314 | Rollout mean MSE | Mean edges/step | Paper selection |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["artifacts"]:
        if record["dataset"] != "Sand" or record["kind"] != "adaptive":
            continue
        m = record["metrics"]
        lines.append(f"| {record['sigma_percentile']:g} | {record['radius_factor']:g} | {m['mse_at_200']['mean']:.9g} | {m['mse_at_314']['mean']:.9g} | {m['rollout_mean_mse']['mean']:.9g} | {m['mean_edges_per_step']['mean']:,.3f} | {'Yes' if record['selected_in_paper'] else ''} |")
    lines += [
        "",
        "The configuration with the lowest MSE@200 need not have the lowest rollout mean MSE or final-step MSE. Higher expansion radii can even reduce realized rollout edge counts because the predicted geometry changes; this is further evidence that the table is not an equal-input cost study.",
        "",
        "## Reproducibility discrepancies requiring repair",
        "",
        "- Sand fixed-NLL mean edges are 28,950.8607 in both its NPZ and JSON, versus 28,968 in the paper. Regenerate tables directly from arrays.",
        "- The claimed Sand MSE r=0.0176 budget match is not reproducible from committed arrays; its reported 23,498 edges are about 2.35% more than Adaptive's 22,958, so even the published match is approximate.",
        "- Save trajectory IDs, dataset file hashes, checkpoint hashes, training seeds, package versions, physical units/normalization, policy parameters, and graph direction/self-edge conventions with every rollout.",
        "- Re-run the preselected comparison across independent training seeds and a locked held-out test set. Report both prediction horizons and total dataset frames explicitly.",
        "- Measure graph construction, model forward time, total rollout time, and peak memory on identical hardware, with warmup and synchronization; include equal-ground-truth-geometry timing as well as autoregressive timing.",
        "",
        "## Reproduce this audit",
        "",
        "From the repository root, run `python research/reanalyze.py`. Only NumPy is needed. Default output is `research/results/reanalysis.json`; the Markdown audit is written beside the repository in `legacy_results_audit.md`. These files are deterministic for the same inputs, Python/NumPy versions, seed, and resample count.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--bootstrap-resamples", type=int, default=100000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1000:
        parser.error("Use at least 1000 bootstrap resamples")
    repo = args.repo.resolve()
    output = args.output or repo / "research/results/reanalysis.json"
    report = args.report or repo.parent / "legacy_results_audit.md"
    paths = sorted(repo.glob("*/models/*/*.npz"))
    if not paths:
        parser.error(f"No committed model rollout NPZ files found under {repo}")
    # Reuse identical row resamples for all comparisons within each dataset.
    # Reset per-dataset RNG for reproducible headline intervals independently
    # of the number or order of artifacts in other datasets.
    paired_indices, independent_indices = {}, {}
    for dataset in sorted({p.relative_to(repo).parts[0] for p in paths}):
        first = next(p for p in paths if p.relative_to(repo).parts[0] == dataset)
        with np.load(first, allow_pickle=False) as z:
            count = len(z["mse_per_trajectory"])
        rng = np.random.default_rng(args.seed)
        paired_indices[dataset] = rng.integers(0, count, size=(args.bootstrap_resamples, count))
        independent_indices[dataset] = rng.integers(0, count, size=(args.bootstrap_resamples, count))
    artifacts, vectors = [], {}
    for path in paths:
        dataset = path.relative_to(repo).parts[0]
        record, per_trajectory = read_artifact(path, repo, paired_indices[dataset])
        artifacts.append(record)
        vectors[record["id"]] = per_trajectory
    comparisons = []
    for adaptive in artifacts:
        if not adaptive["selected_in_paper"]:
            continue
        for baseline in artifacts:
            if baseline["dataset"] == adaptive["dataset"] and baseline["kind"] != "adaptive":
                comparisons.append(compare(adaptive, baseline, vectors[adaptive["id"]], vectors[baseline["id"]], paired_indices[adaptive["dataset"]], independent_indices[adaptive["dataset"]]))
    sources = {"research/reanalyze.py": sha256(Path(__file__).resolve())}
    upstream_sources = {}
    for relative in ("adaptive-gns/scripts/evaluate_adaptive_rollout.py", "adaptive-gns/scripts/evaluate_rollout_mse.py", "adaptive-gns/gns/data_loader.py", "jobs/eval_wd_adaptive.sh", "jobs/eval_adaptive_rollout.sh"):
        if (repo / relative).exists():
            sources[relative] = sha256(repo / relative)
        try:
            historical_source = subprocess.check_output(["git", "show", f"{UPSTREAM_REF}:{relative}"], cwd=repo, stderr=subprocess.DEVNULL)
            upstream_sources[relative] = hashlib.sha256(historical_source).hexdigest()
        except (subprocess.CalledProcessError, FileNotFoundError):
            upstream_sources[relative] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    result = {
        "analysis_kind": "legacy_saved_rollout_reanalysis",
        "fresh_training_runs": 0,
        "fresh_simulation_runs": 0,
        "repository_head_at_analysis": commit,
        "upstream_reference_for_historical_code_audit": UPSTREAM_REF,
        "upstream_source_code_sha256": upstream_sources,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "bootstrap": {"method": "whole-trajectory percentile bootstrap", "seed": args.seed, "resamples": args.bootstrap_resamples, "confidence_level": 0.95, "training_seed_resampling": False, "pairing_verified_by_ids": False},
        "cautions": CAUTIONS,
        "source_code_sha256": sources,
        "artifacts": artifacts,
        "comparisons": comparisons,
        "post_hoc_observed_pareto_fronts": {
            dataset: {metric: pareto_ids([r for r in artifacts if r["dataset"] == dataset], metric) for metric in ("mse_at_200", "rollout_mean_mse")}
            for dataset in paired_indices
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, report)
    print(f"Audited {len(artifacts)} artifacts; {len(comparisons)} selected-policy comparisons.")
    print(f"Machine-readable results: {output}")
    print(f"Audit report: {report}")
    for c in comparisons:
        if c["dataset"] == "WaterDrop" and c["baseline_kind"] == "fixed_mse_base":
            print("WaterDrop MSE@200 adaptive-minus-baseline:", json.dumps(c["metrics"]["mse_at_200"]))


if __name__ == "__main__":
    main()
