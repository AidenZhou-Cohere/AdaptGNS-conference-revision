"""Exploratory frozen-model risk/error/benefit diagnostic on pilot frames.

No training, checkpoint selection, policy fitting, or physical causal claim.
Positive signed benefit means base vector squared error minus dense error > 0.
Adding every annulus pair is a model intervention, not a per-node action value.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import spearmanr
import torch
from torch.torch_version import TorchVersion

from research import budget_graph, pilot


METRICS = (
    "risk_error_spearman", "risk_benefit_spearman", "error_benefit_spearman",
    "risk_error_minus_risk_benefit_spearman",
    "base_vector_squared_error", "dense_vector_squared_error",
    "mean_signed_benefit", "fraction_harmful", "fraction_beneficial",
    "fraction_unchanged", "base_directed_edges", "dense_directed_edges",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rho = float(spearmanr(x, y).statistic)
    return rho if np.isfinite(rho) else None


def frame_metrics(risk, base_error, dense_error):
    """Use float64 for differences/aggregation of float32 simulator residuals."""
    risk, base_error, dense_error = (np.asarray(x, dtype=np.float64)
                                    for x in (risk, base_error, dense_error))
    if risk.ndim != 1 or risk.shape != base_error.shape or risk.shape != dense_error.shape:
        raise ValueError("Risk and both squared residuals must have matching 1D shapes")
    if not len(risk) or not all(np.isfinite(x).all() for x in (risk, base_error, dense_error)):
        raise ValueError("Diagnostic requires finite, nonempty particle arrays")
    benefit = base_error - dense_error
    rho_error, rho_benefit = correlation(risk, base_error), correlation(risk, benefit)
    metrics = {
        "risk_error_spearman": rho_error,
        "risk_benefit_spearman": rho_benefit,
        "error_benefit_spearman": correlation(base_error, benefit),
        "risk_error_minus_risk_benefit_spearman": (
            None if rho_error is None or rho_benefit is None else rho_error - rho_benefit),
        "base_vector_squared_error": float(base_error.mean()),
        "dense_vector_squared_error": float(dense_error.mean()),
        "mean_signed_benefit": float(benefit.mean()),
        "fraction_harmful": float((benefit < 0).mean()),
        "fraction_beneficial": float((benefit > 0).mean()),
        "fraction_unchanged": float((benefit == 0).mean()),
    }
    return metrics, benefit


def average_metrics(rows):
    """Equal-weight averages; explicitly count undefined correlation entries."""
    result, counts = {}, {}
    for name in METRICS:
        values = [row[name] for row in rows if row.get(name) is not None]
        result[name] = float(np.mean(values)) if values else None
        counts[name] = len(values)
    return {"metrics": result, "defined_counts": counts, "n_rows": len(rows)}


def aggregate_frames(records):
    trajectories = {}
    for name in sorted({record["trajectory"] for record in records}):
        rows = [record["metrics"] for record in records if record["trajectory"] == name]
        trajectories[name] = average_metrics(rows)
    aggregate = average_metrics([row["metrics"] for row in trajectories.values()])
    return {"trajectories": trajectories, "equal_trajectory_mean": aggregate}


@torch.no_grad()
def evaluate_checkpoint(checkpoint, frames, expected_protocol, raw_path):
    # Our generated checkpoint stores torch.__version__ as a TorchVersion object.
    # Keep restricted loading and allowlist only that known metadata class.
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    protocol = payload["protocol"]
    if protocol != expected_protocol:
        raise ValueError(f"Checkpoint protocol differs from run protocol: {checkpoint}")
    architecture = protocol["architecture"]
    model = pilot.PilotGNS(width=architecture["latent_width"],
                           depth=architecture["processor_depth"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    objective = payload["objective"]
    raw_arrays, splits = {}, {}
    for split, split_frames in frames.items():
        records = []
        for index, frame in enumerate(split_frames):
            graph = frame["graph"]
            dense = np.concatenate((graph.base, graph.extra), axis=0)
            base_mean, risk = pilot.run_forward(model, frame, graph.base, objective == "faithful")
            dense_mean, _ = pilot.run_forward(model, frame, dense, objective == "faithful")
            base_error = (base_mean - frame["target"]).square().sum(-1).numpy()
            dense_error = (dense_mean - frame["target"]).square().sum(-1).numpy()
            scores = risk.numpy()
            metrics, benefit = frame_metrics(scores, base_error, dense_error)
            metrics["base_directed_edges"] = 2 * len(graph.base)
            metrics["dense_directed_edges"] = 2 * len(dense)
            prefix = f"{split}_{index:03d}"
            raw_arrays.update({f"{prefix}_risk": scores,
                               f"{prefix}_base_vector_se": base_error,
                               f"{prefix}_dense_vector_se": dense_error,
                               f"{prefix}_signed_benefit": benefit})
            records.append({"id": frame["id"], "trajectory": frame["trajectory"],
                            "step": frame["step"], "n_particles": len(scores),
                            "raw_prefix": prefix, "metrics": metrics})
        splits[split] = {"frames": records, **aggregate_frames(records)}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw_arrays)
    return {"objective": objective, "seed": int(payload["seed"]),
            "checkpoint_sha256": sha256(checkpoint),
            "raw_arrays": {"path": str(raw_path.resolve()), "sha256": sha256(raw_path)},
            "splits": splits}


def summarize_seeds(runs, objectives):
    summaries = {}
    for objective in objectives:
        selected = sorted((run for run in runs.values() if run["objective"] == objective),
                          key=lambda run: run["seed"])
        for split in ("valid", "test"):
            statistics = {}
            for name in METRICS:
                values = [run["splits"][split]["equal_trajectory_mean"]["metrics"][name]
                          for run in selected]
                values = [value for value in values if value is not None]
                statistics[name] = {
                    "mean": float(np.mean(values)) if values else None,
                    "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "n_seeds_defined": len(values),
                }
            summaries[f"{objective}/{split}"] = {
                "objective": objective, "split": split,
                "seeds": [run["seed"] for run in selected], "statistics": statistics}
    return summaries


def format_stat(stat, percentage=False):
    if stat["mean"] is None:
        return "undefined"
    scale = 100 if percentage else 1
    suffix = "%" if percentage else ""
    mean = stat["mean"] * scale
    sd = stat["seed_sd"]
    return (f"{mean:.3f}{suffix}" if sd is None else
            f"{mean:.3f} ± {sd * scale:.3f}{suffix}")


def write_report(path, output_path, result):
    completed, expected = len(result["runs"]), len(result["expected_runs"])
    rows = ["# Does predicted risk rank the benefit of more edges?", "",
            f"**Exploratory diagnostic: {completed}/{expected} predeclared pilot checkpoints available.** "
            "No additional training, model selection, or policy fitting was performed.", "",
            "Each frozen model is evaluated on the same 36 preselected validation frames and 36 test frames "
            "(12 frames from each of three trajectories per split). We compare its base graph with the "
            "same graph plus every candidate pair within 1.267 times the base radius. Predicted risk "
            "comes from the base pass. For particle i, signed benefit is its base squared vector residual "
            "minus its dense-graph squared vector residual; positive values indicate improvement.", "",
            "Spearman correlations are computed across particles within each frame, then averaged equally "
            "over frames within each trajectory, trajectories within each seed, and finally seeds. "
            "Values after ± are sample standard deviations across training seeds, not confidence intervals. "
            "Undefined correlations are omitted with counts retained in JSON. Harmful means strictly "
            "negative signed benefit; exact zeros are stored separately.", "",
            "| Objective | Split | Seeds | Risk–error ρ | Risk–benefit ρ | Harmful particles | Mean signed benefit |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for summary in result["summaries"].values():
        stats = summary["statistics"]
        rows.append(f"| {summary['objective']} | {summary['split']} | {len(summary['seeds'])} | "
                    f"{format_stat(stats['risk_error_spearman'])} | "
                    f"{format_stat(stats['risk_benefit_spearman'])} | "
                    f"{format_stat(stats['fraction_harmful'], True)} | "
                    f"{format_stat(stats['mean_signed_benefit'])} |")
    rows += ["", "Signed benefits use squared normalized-acceleration vector units (sum over two coordinates). "
             "Divide vector squared errors by two to match the pilot’s coordinate-mean MSE. "
             "The JSON retains per-frame, per-trajectory, and per-seed measurements; compressed NPZ files "
             "retain every particle’s score, base/dense residual, and signed benefit.", "",
             "This tests whether a residual score tracks one particular graph intervention. It does not "
             "measure the marginal value of expanding only the scored particle, does not match the "
             "25% extra-pair policy budget, and does not establish a causal effect in the physical system. "
             "The benefit includes the base error algebraically, so correlation with benefit alone is not "
             "proof of useful allocation. Weak or negative correlations challenge the proposed proxy for "
             "this intervention; they do not prove every uncertainty-guided graph policy fails. "
             "All results remain conditional on this small convenience sample, architecture, training "
             "schedule, candidate radius, and ground-truth histories, with no autonomous rollout or speed claim.", "",
             f"[Machine-readable results]({output_path.resolve()}) · "
             f"[Diagnostic code]({Path(__file__).resolve()})", ""]
    if result["pending"]:
        rows += ["Pending checkpoints: " + ", ".join(result["pending"]) + ".", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("research/results/waterdrop_pilot"))
    parser.add_argument("--output", type=Path, default=Path("research/results/risk_benefit.json"))
    parser.add_argument("--report", type=Path, default=Path("../risk_benefit.md"))
    parser.add_argument("--seeds", type=int, nargs="+", help="Optional initial validation subset")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    protocol_path = args.checkpoint_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source_paths = [Path(__file__), Path(pilot.__file__), Path(budget_graph.__file__)]
    sources = {path.name: sha256(path) for path in source_paths}
    if protocol["code_sha256"] != sources["pilot.py"]:
        raise ValueError("Current pilot.py differs from the training protocol's source hash")
    meta_path = args.data_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    files = [meta_path] + [args.data_dir / f"{split}-pilot.npz" for split in ("valid", "test")]
    data_hashes = {path.name: sha256(path) for path in files}
    frames = {split: pilot.load_frames(args.data_dir / f"{split}-pilot.npz", metadata, 12, rng)
              for split, rng in (("valid", 811), ("test", 812))}
    for split, split_frames in frames.items():
        if [frame["id"] for frame in split_frames] != protocol["frames"][split]:
            raise ValueError(f"{split} frames differ from the predeclared pilot protocol")
    seeds = protocol["seeds"] if args.seeds is None else args.seeds
    if not set(seeds) <= set(protocol["seeds"]):
        raise ValueError("Requested seeds must belong to the predeclared pilot protocol")
    expected = [f"{objective}_seed{seed}" for seed in protocol["seeds"]
                for objective in protocol["objectives"]]
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "torch": str(torch.__version__),
                "threads": args.threads}
    provenance = {"source_sha256": sources, "data_sha256": data_hashes,
                  "pilot_protocol_sha256": sha256(protocol_path), "versions": versions}
    cache = json.loads(args.output.read_text()) if args.output.exists() else {}
    runs = cache.get("runs", {}) if cache.get("provenance") == provenance else {}
    for seed in seeds:
        for objective in protocol["objectives"]:
            name = f"{objective}_seed{seed}"
            checkpoint = args.checkpoint_dir / f"{name}.pt"
            if not checkpoint.exists() or not checkpoint.with_suffix(".json").exists():
                continue
            if name in runs and runs[name]["checkpoint_sha256"] == sha256(checkpoint):
                raw = Path(runs[name]["raw_arrays"]["path"])
                if raw.exists() and sha256(raw) == runs[name]["raw_arrays"]["sha256"]:
                    continue
            raw_path = args.output.parent / "risk_benefit_samples" / f"{name}.npz"
            runs[name] = evaluate_checkpoint(checkpoint, frames, protocol, raw_path)
            if runs[name]["objective"] != objective or runs[name]["seed"] != seed:
                raise ValueError("Checkpoint identity does not match its filename")
            print("diagnosed", name, flush=True)
    result = {
        "schema_version": 1,
        "scope": "Exploratory same-state frozen-model diagnostic; no training, tuning or new policy",
        "benefit_definition": "base vector squared residual minus dense vector squared residual",
        "graph_intervention": "Add all candidate annulus pairs, R/r=1.267; retain every base edge",
        "aggregation": "Particles within frame; equal frames per trajectory; equal trajectories per seed; equal seeds",
        "harmful_definition": "signed benefit < 0; exact zero recorded separately; no tolerance threshold",
        "provenance": provenance, "expected_runs": expected,
        "pending": [name for name in expected if name not in runs],
        "complete": all(name in runs for name in expected),
        "runs": runs, "summaries": summarize_seeds(runs, protocol["objectives"]),
    }
    write_json(args.output, result)
    write_report(args.report, args.output, result)
    print(json.dumps({"completed": len(runs), "expected": len(expected),
                      "complete": result["complete"], "pending": result["pending"]}), flush=True)


if __name__ == "__main__":
    main()
