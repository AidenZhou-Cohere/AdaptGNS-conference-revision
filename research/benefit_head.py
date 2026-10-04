"""Exploratory training-only ridge prediction of frozen-model graph benefit.

Targets are base minus dense per-particle vector squared residuals. They are
not single-edge action values. The 25% budget remains mandatory even when
predictions are negative. Neither validation nor test labels enter fitting.
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from torch.torch_version import TorchVersion

from research import budget_graph, pilot
from research.risk_benefit import correlation, sha256, write_json


POLICIES = ("base", "random25", "speed25", "current_risk25", "benefit25", "dense")
DIAGNOSTICS = ("predicted_benefit_spearman", "risk_benefit_spearman",
               "predicted_benefit_mse", "fraction_predicted_positive",
               "mean_true_benefit", "mean_predicted_benefit")


def fit_ridge(features, targets, regularization=1e-3):
    """Minimize mean((y-b-Zw)^2) + regularization*||w||² in float64.

    Z uses population means/standard deviations from training particles only.
    Centering permits an unpenalized intercept b=mean(y). Constant features
    are scaled by one. The loss averages all pooled particle/frame samples.
    """
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("Expected nonempty [samples, features] and [samples] arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or regularization <= 0:
        raise ValueError("Finite data and strictly positive regularization required")
    center, scale = x.mean(0), x.std(0)
    constant = scale < 1e-12
    scale[constant] = 1.0
    z, intercept = (x - center) / scale, float(y.mean())
    covariance = z.T @ z / len(z)
    system = covariance + regularization * np.eye(x.shape[1])
    rhs = z.T @ (y - intercept) / len(z)
    weights = np.linalg.solve(system, rhs)
    fit = {"feature_mean": center, "feature_scale": scale, "weights": weights,
           "intercept": np.array(intercept), "regularization": np.array(regularization)}
    prediction = predict_ridge(features, fit)
    mse = float(np.mean((prediction-y)**2))
    diagnostics = {"n_samples": len(y), "n_features": x.shape[1],
                   "constant_features": int(constant.sum()),
                   "target_mean": intercept, "target_sd": float(y.std()),
                   "training_mse": mse,
                   "training_regularized_objective": mse + regularization*float(weights@weights),
                   "system_condition_number": float(np.linalg.cond(system)),
                   "normal_equation_residual": float(np.linalg.norm(system@weights-rhs))}
    return fit, diagnostics


def predict_ridge(features, fit):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1:] != fit["feature_mean"].shape:
        raise ValueError("Feature dimensions do not match fitted coefficients")
    return (x-fit["feature_mean"])/fit["feature_scale"] @ fit["weights"] + float(fit["intercept"])


@torch.no_grad()
def base_with_embedding(model, frame, faithful):
    """Capture the exact detached input to the existing mean head; no core edit."""
    captured = []
    def capture(_module, args):
        captured.append(args[0].detach().cpu().numpy().copy())
    handle = model.mean.register_forward_pre_hook(capture)
    try:
        mean, risk = pilot.run_forward(model, frame, frame["graph"].base, faithful)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one mean-head input per simulator pass")
    return mean, risk, captured[0]


def vector_error(mean, target):
    return (mean-target).square().sum(-1).numpy().astype(np.float64)


def mean_metrics(records):
    metrics = {}
    for policy in POLICIES:
        metrics[policy] = {name: float(np.mean([record["policies"][policy][name] for record in records]))
                           for name in ("mse_normalized_acceleration", "directed_edges")}
    diagnostics = {}
    for name in DIAGNOSTICS:
        values = [record["diagnostics"][name] for record in records if record["diagnostics"][name] is not None]
        diagnostics[name] = float(np.mean(values)) if values else None
    return {"policies": metrics, "diagnostics": diagnostics}


def aggregate_frames(records):
    trajectories = {name: mean_metrics([row for row in records if row["trajectory"] == name])
                    for name in sorted({row["trajectory"] for row in records})}
    return {"trajectories": trajectories, "equal_trajectory_mean": mean_metrics(list(trajectories.values()))}


@torch.no_grad()
def fit_and_evaluate(checkpoint, frames, expected_protocol, coefficient_path, raw_path, regularization):
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != expected_protocol:
        raise ValueError(f"Checkpoint protocol differs: {checkpoint}")
    arch = expected_protocol["architecture"]
    model = pilot.PilotGNS(width=arch["latent_width"], depth=arch["processor_depth"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    seed, kind = int(payload["seed"]), payload["objective"]
    faithful = kind == "faithful"
    started = time.perf_counter()
    embeddings, targets, training_records = [], [], []
    for frame in frames["train"]:
        base_mean, _, embedding = base_with_embedding(model, frame, faithful)
        graph = frame["graph"]
        dense_mean, _ = pilot.run_forward(model, frame, np.concatenate((graph.base, graph.extra)), faithful)
        target = vector_error(base_mean, frame["target"]) - vector_error(dense_mean, frame["target"])
        embeddings.append(embedding)
        targets.append(target)
        training_records.append({"id": frame["id"], "trajectory": frame["trajectory"],
                                 "n_particles": len(target), "mean_signed_benefit": float(target.mean())})
    fit, fit_diagnostics = fit_ridge(np.concatenate(embeddings), np.concatenate(targets), regularization)
    fit_seconds = time.perf_counter() - started
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coefficient_path, **fit)
    del embeddings, targets
    # The saved pilot controls use the same independent RNG reset per split.
    original = json.loads(checkpoint.with_suffix(".json").read_text())
    splits, raw = {}, {}
    for split in ("valid", "test"):
        rng = np.random.default_rng(91300+seed)
        control_records = original["validation" if split == "valid" else "test"]
        if [row["id"] for row in control_records] != [frame["id"] for frame in frames[split]]:
            raise ValueError("Evaluation frames differ from original pilot controls")
        records = []
        for index, (frame, control) in enumerate(zip(frames[split], control_records)):
            graph = frame["graph"]
            budget = math.floor(.25*len(graph.extra))
            base_mean, risk, embedding = base_with_embedding(model, frame, faithful)
            gain = predict_ridge(embedding, fit)
            pairs_by_policy = {
                "base": graph.base,
                "random25": budget_graph.random_pairs(graph, budget, rng),
                "speed25": budget_graph.select_pairs(graph, frame["speed"], budget),
                "current_risk25": budget_graph.select_pairs(graph, risk.numpy(), budget),
                "benefit25": budget_graph.select_pairs(graph, gain, budget),
                "dense": np.concatenate((graph.base, graph.extra)),
            }
            errors, measurements = {}, {}
            for policy, pairs in pairs_by_policy.items():
                if policy.endswith("25") and len(pairs) != len(graph.base)+budget:
                    raise RuntimeError("Equal incremental edge budget violated")
                mean = base_mean if policy == "base" else pilot.run_forward(model, frame, pairs, faithful)[0]
                error = vector_error(mean, frame["target"])
                errors[policy] = error
                mse = float(error.mean()/mean.shape[-1])
                measurements[policy] = {"mse_normalized_acceleration": mse, "directed_edges": 2*len(pairs)}
                if policy != "benefit25":
                    old = control["policies"][policy]
                    if old["directed_edges"] != 2*len(pairs) or not np.isclose(mse, old["mse_normalized_acceleration"], rtol=2e-5, atol=1e-7):
                        raise ValueError(f"Control replay differs: {checkpoint.name} {split} {frame['id']} {policy}")
            benefit = errors["base"]-errors["dense"]
            diagnostics = {"predicted_benefit_spearman": correlation(gain, benefit),
                           "risk_benefit_spearman": correlation(risk.numpy(), benefit),
                           "predicted_benefit_mse": float(np.mean((gain-benefit)**2)),
                           "fraction_predicted_positive": float((gain>0).mean()),
                           "mean_true_benefit": float(benefit.mean()),
                           "mean_predicted_benefit": float(gain.mean())}
            prefix = f"{split}_{index:03d}"
            raw.update({f"{prefix}_predicted_benefit": gain, f"{prefix}_true_benefit": benefit,
                        f"{prefix}_risk": risk.numpy()})
            for policy, error in errors.items():
                raw[f"{prefix}_{policy}_vector_se"] = error
            records.append({"id": frame["id"], "trajectory": frame["trajectory"], "step": frame["step"],
                            "n_particles": len(gain), "extra_pair_budget": budget,
                            "raw_prefix": prefix, "policies": measurements, "diagnostics": diagnostics})
        splits[split] = {"frames": records, **aggregate_frames(records)}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw)
    return {"objective": kind, "seed": seed, "checkpoint_sha256": sha256(checkpoint),
            "original_controls_sha256": sha256(checkpoint.with_suffix(".json")),
            "coefficient_file": {"path": str(coefficient_path.resolve()), "sha256": sha256(coefficient_path)},
            "raw_file": {"path": str(raw_path.resolve()), "sha256": sha256(raw_path)},
            "training": {"frames": training_records, "diagnostics": fit_diagnostics,
                         "label_generation_and_fit_seconds": fit_seconds},
            "total_seconds": time.perf_counter()-started, "splits": splits}


def stat(values):
    values = [value for value in values if value is not None]
    return {"mean": float(np.mean(values)) if values else None,
            "seed_sd": float(np.std(values, ddof=1)) if len(values)>1 else None,
            "n_defined": len(values)}


def summarize(runs, objectives):
    result = {}
    for kind in objectives:
        selected = sorted([run for run in runs.values() if run["objective"] == kind], key=lambda run: run["seed"])
        for split in ("valid", "test"):
            rows = [run["splits"][split]["equal_trajectory_mean"] for run in selected]
            policies = {policy: stat([row["policies"][policy]["mse_normalized_acceleration"] for row in rows])
                        for policy in POLICIES}
            deltas = {}
            for comparator in ("base", "random25", "speed25", "current_risk25"):
                difference = [row["policies"]["benefit25"]["mse_normalized_acceleration"]
                              - row["policies"][comparator]["mse_normalized_acceleration"] for row in rows]
                deltas[comparator] = {**stat(difference), "seeds_with_lower_mse": int(np.sum(np.asarray(difference)<0))}
            result[f"{kind}/{split}"] = {"objective": kind, "split": split, "seeds": [run["seed"] for run in selected],
                                        "policies": policies, "benefit_minus_comparator": deltas,
                                        "diagnostics": {name: stat([row["diagnostics"][name] for row in rows]) for name in DIAGNOSTICS}}
    return result


def fmt(value, digits=4):
    if value["mean"] is None:
        return "undefined"
    suffix = "" if value["seed_sd"] is None else f" ± {value['seed_sd']:.{digits}f}"
    return f"{value['mean']:.{digits}f}" + suffix


def write_report(path, output, result):
    rows = ["# Exploratory prediction of graph-expansion benefit", "",
            f"**{len(result['runs'])}/{len(result['expected_runs'])} frozen pilot checkpoints evaluated.** "
            "This follow-up was designed after inspecting the initial pilot. It is exploratory, not an untouched-test confirmation or a new-algorithm novelty claim.", "",
            "For each frozen model, a ridge head predicts signed base-minus-dense vector squared residual from its detached 48-dimensional base-graph mean-head input. "
            "Only the pilot's 192 preselected training frames enter fitting. Neither validation nor test targets enter standardization, fitting, or hyperparameter selection. "
            "All fifteen models are retained; the simulator weights are unchanged.", "",
            "If X contains pooled training particle embeddings, let Z=(X−mean(X))/std(X), y be signed benefit, and n the number of particle/frame samples. "
            "The fixed objective is mean((y−b−Zw)²)+0.001||w||². Its solution is b=mean(y) and "
            "w=(ZᵀZ/n+0.001I)⁻¹Zᵀ(y−mean(y))/n. The intercept is unpenalized; constant feature scales are one. "
            "Training weights particle/frame samples equally, so larger trajectories contribute more particles. Coefficients and preprocessing statistics are saved per checkpoint.", "",
            "At evaluation, the pair priority is max(predicted benefit at either endpoint). The head adds exactly floor(0.25 × annulus pairs), preserving every base pair, "
            "with both directions and no self-loops. Negative predictions are allowed, but the fixed budget still forces the same number of additions. "
            "This candidate does not implement a do-nothing action or estimate each edge's marginal value.", "",
            "The table gives observed-history normalized acceleration coordinate MSE, averaged equally across frames then three trajectories, "
            "and reported as mean ± sample SD over five simulator-training seeds. Each row replays the original random, speed and current-risk controls, "
            "with matching random seeds and checks against saved control errors and edge counts.", "",
            "| Objective | Split | Base | Random25 | Speed25 | Current risk25 | Benefit25 | Dense |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for summary in result["summaries"].values():
        rows.append(f"| {summary['objective']} | {summary['split']} | " + " | ".join(fmt(summary['policies'][policy]) for policy in POLICIES) + " |")
    rows += ["", "Paired seed-level MSE differences below are benefit25 minus its comparator; negative is better. "
             "The ± values are sample SDs of paired differences, not confidence intervals or significance claims.", "",
             "| Objective | Split | Δ versus random25 | Δ versus speed25 | Δ versus current risk25 | Benefit–label rank ρ | Risk–label rank ρ |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for summary in result["summaries"].values():
        delta, diag = summary["benefit_minus_comparator"], summary["diagnostics"]
        rows.append(f"| {summary['objective']} | {summary['split']} | "
                    + " | ".join(fmt(delta[c]) for c in ("random25", "speed25", "current_risk25"))
                    + f" | {fmt(diag['predicted_benefit_spearman'],3)} | {fmt(diag['risk_benefit_spearman'],3)} |")
    rows += ["", "Dense-intervention labels are model-specific changes after adding the entire annulus, not single-edge utilities or physical causal effects. "
             "The max-endpoint pair rule is a heuristic and tie breaking depends on particle IDs. The benefit head and current-risk control both need a base scoring pass "
             "before the selected-graph prediction. There is no autonomous-rollout or end-to-end speed result for the benefit head.", "",
             "The convenience subset contains eight training and three validation/test trajectories per split and uses a compact, zero-noise model. "
             "Training-label dependence, limited state coverage, graph-intervention mismatch and possible failure under predicted histories restrict generalization. "
             "Reusing this already inspected test subset means the experiment can motivate a preregistered new evaluation but cannot serve as independent confirmation.", "",
             f"Summed label-generation and ridge-fit time: {sum(run['training']['label_generation_and_fit_seconds'] for run in result['runs'].values()):.1f} seconds. "
             f"Summed complete checkpoint evaluation time: {sum(run['total_seconds'] for run in result['runs'].values()):.1f} seconds. "
             "These CPU measurements include concurrent work and are not optimized simulator latency benchmarks.", "",
             f"[Code]({Path(__file__).resolve()}) · [Complete numerical results]({output.resolve()})", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("research/results/waterdrop_pilot"))
    parser.add_argument("--output", type=Path, default=Path("research/results/benefit_head.json"))
    parser.add_argument("--report", type=Path, default=Path("../benefit_head_results.md"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    protocol_path = args.checkpoint_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if sha256(pilot.__file__) != protocol["code_sha256"]:
        raise ValueError("Pilot source hash differs from the training protocol")
    metadata_path = args.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    frames = {split: pilot.load_frames(args.data_dir / f"{split}-pilot.npz", metadata, count, rng)
              for split, count, rng in (("train",24,810), ("valid",12,811), ("test",12,812))}
    for split, split_frames in frames.items():
        if [frame["id"] for frame in split_frames] != protocol["frames"][split]:
            raise ValueError(f"{split} frame IDs differ from pilot protocol")
    source_paths = [Path(__file__), Path(pilot.__file__), Path(budget_graph.__file__)]
    # Hash the imported diagnostic helpers too: a source change invalidates cache.
    source_paths.append(Path(__file__).with_name("risk_benefit.py"))
    provenance = {"source_sha256": {path.name: sha256(path) for path in source_paths},
                  "data_sha256": {path.name: sha256(path) for path in [metadata_path]+[args.data_dir/f"{split}-pilot.npz" for split in frames]},
                  "pilot_protocol_sha256": sha256(protocol_path),
                  "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                               "torch": str(torch.__version__), "threads": args.threads}}
    expected = [f"{kind}_seed{seed}" for seed in protocol["seeds"] for kind in protocol["objectives"]]
    regularization = 1e-3
    cache = json.loads(args.output.read_text()) if args.output.exists() else {}
    runs = cache.get("runs", {}) if cache.get("provenance") == provenance else {}
    for seed in protocol["seeds"]:
        for kind in protocol["objectives"]:
            name = f"{kind}_seed{seed}"
            checkpoint = args.checkpoint_dir/f"{name}.pt"
            if not checkpoint.exists() or not checkpoint.with_suffix(".json").exists():
                raise FileNotFoundError(f"All checkpoints required: missing {name}")
            previous = runs.get(name)
            if previous and previous["checkpoint_sha256"] == sha256(checkpoint) and previous["original_controls_sha256"] == sha256(checkpoint.with_suffix(".json")):
                if all(Path(previous[key]["path"]).exists() and sha256(previous[key]["path"]) == previous[key]["sha256"] for key in ("coefficient_file","raw_file")):
                    continue
            coefficient_path = args.output.parent/"benefit_head_coefficients"/f"{name}.npz"
            raw_path = args.output.parent/"benefit_head_samples"/f"{name}.npz"
            runs[name] = fit_and_evaluate(checkpoint, frames, protocol, coefficient_path, raw_path, regularization)
            if runs[name]["objective"] != kind or runs[name]["seed"] != seed:
                raise ValueError("Checkpoint identity does not match filename")
            partial = {"provenance": provenance, "runs": runs, "complete": False}
            write_json(args.output, partial)
            print("fitted and evaluated", name, f"{runs[name]['total_seconds']:.1f}s", flush=True)
    result = {"schema_version": 1, "scope": "Post-pilot exploratory training-only ridge benefit head; observed-history evaluation",
              "regularization": regularization, "fit_split": "train", "prediction_feature": "detached base mean-head input",
              "label": "base minus dense normalized acceleration vector squared residual",
              "budget": "exact floor(0.25*annulus_pairs), max-endpoint predicted gain; signed scores, mandatory budget",
              "training_weighting": "equal pooled particle/frame samples", "evaluation_weighting": "equal frames per trajectory, equal trajectories, equal seeds",
              "hyperparameters_tuned": False, "test_previously_inspected": True,
              "provenance": provenance, "expected_runs": expected, "complete": all(name in runs for name in expected),
              "runs": runs, "summaries": summarize(runs, protocol["objectives"])}
    write_json(args.output, result)
    write_report(args.report, args.output, result)
    print(json.dumps({"complete": result["complete"], "runs": len(runs)}), flush=True)


if __name__ == "__main__":
    main()
