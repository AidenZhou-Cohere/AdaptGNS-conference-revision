"""CPU-only associations between cached pilot risk and observed kinematics.

No checkpoint loading, model inference, fitting a predictor, or reserved-test
access. The accompanying protocol fixes the estimands and missing-value rules.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from research.physical_complexity import local_flow_diagnostics
from research.rank_diagnostics import spearman_record

OBJECTIVES = ("faithful", "nll", "beta_nll")
SEEDS = tuple(range(5))
SPLITS = ("valid", "test")
METRICS = ("strain", "vorticity", "strain_partial", "vorticity_partial",
           "divergence", "velocity_dispersion", "error_on_fit", "benefit_on_fit",
           "neighbor_count_all", "speed_all", "wall_clearance_all",
           "observed_acceleration_all", "error_all", "benefit_all")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_record(values):
    finite = [float(value) for value in values if value is not None]
    if any(not np.isfinite(value) for value in finite):
        raise ValueError("Nonfinite coefficient in aggregation")
    return {"required": len(values), "defined": len(finite),
            "mean": float(np.mean(finite)) if finite and len(finite) == len(values) else None,
            "defined_only_mean": float(np.mean(finite)) if finite else None}


def aggregate_frames(rows):
    if len(rows) != 36 or len({row["id"] for row in rows}) != 36:
        raise ValueError("Expected all 36 unique frames")
    names = sorted({row["trajectory"] for row in rows})
    if len(names) != 3:
        raise ValueError("Expected exactly three trajectories")
    trajectories = {}
    for name in names:
        selected = [row for row in rows if row["trajectory"] == name]
        if len(selected) != 12:
            raise ValueError("Expected twelve frames in every trajectory")
        trajectories[name] = {key: mean_record([row["correlations"][key]["value"] for row in selected])
                              for key in METRICS}
    metrics = {}
    for key in METRICS:
        required = mean_record([trajectories[name][key]["mean"] for name in names])
        conditional = mean_record([trajectories[name][key]["defined_only_mean"] for name in names])
        metrics[key] = {"unconditional_equal_trajectory_mean": required["mean"],
                        "defined_frame_equal_trajectory_mean": conditional["mean"],
                        "required_frames": 36,
                        "defined_frames": sum(trajectories[name][key]["defined"] for name in names),
                        "trajectories_with_any_defined_frame": conditional["defined"]}
    return {"trajectories": trajectories, "metrics": metrics}


def seed_summary(runs):
    result = {}
    for objective in OBJECTIVES:
        for split in SPLITS:
            groups = [runs[f"{objective}_seed{seed}"]["splits"][split]["aggregate"]["metrics"] for seed in SEEDS]
            metrics = {}
            for key in METRICS:
                values = [group[key]["unconditional_equal_trajectory_mean"] for group in groups]
                conditional = [group[key]["defined_frame_equal_trajectory_mean"] for group in groups]
                required = mean_record(values)
                defined = mean_record(conditional)
                metrics[key] = {"seed_values": values, "mean": required["mean"],
                    "sample_seed_sd": float(np.std(values, ddof=1)) if required["defined"] == 5 else None,
                    "conditional_seed_values": conditional, "defined_frame_mean": defined["mean"],
                    "defined_frame_sample_seed_sd": float(np.std(conditional, ddof=1))
                        if defined["defined"] == 5 else None,
                    "conditional_seeds_defined": defined["defined"],
                    "defined_frames_per_seed": [group[key]["defined_frames"] for group in groups]}
            result[f"{objective}/{split}"] = metrics
    return result


def descriptors(history, bounds):
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[0] != 3 or history.shape[-1] != 2:
        raise ValueError("Three observed frames with shape (3,N,2) required")
    if not np.isfinite(history).all():
        raise ValueError("All three observed frames must be finite")
    x = history[-1]
    flow = local_flow_diagnostics(x, x - history[-2])
    flow["observed_acceleration"] = np.linalg.norm(x - 2 * history[-2] + history[-3], axis=1)
    # Signed clearance from the nearest box plane, not a free-surface estimate.
    flow["wall_clearance"] = np.minimum(x - bounds[:, 0], bounds[:, 1] - x).min(axis=1)
    return flow


def frame_correlations(flow, risk, base_error, benefit):
    risk, base_error, benefit = [np.asarray(value, dtype=np.float64) for value in (risk, base_error, benefit)]
    n = len(flow["fit_valid"])
    if any(value.shape != (n,) or not np.isfinite(value).all() for value in (risk, base_error, benefit)):
        raise ValueError("Cached arrays have wrong shape or nonfinite values")
    if np.any(risk <= 0) or np.any(base_error < 0):
        raise ValueError("Risk must be positive and squared error nonnegative")
    mask = flow["fit_valid"]
    controls = np.column_stack([flow[key][mask] for key in ("neighbor_count", "speed", "wall_clearance")])
    correlations = {}
    for key, target in (("strain", flow["strain_frobenius"]), ("vorticity", flow["abs_vorticity"]),
                        ("divergence", flow["abs_divergence"]), ("velocity_dispersion", flow["velocity_dispersion"]),
                        ("error_on_fit", base_error), ("benefit_on_fit", benefit)):
        correlations[key] = spearman_record(risk[mask], target[mask])
    for key, target in (("strain_partial", "strain_frobenius"), ("vorticity_partial", "abs_vorticity")):
        correlations[key] = spearman_record(risk[mask], flow[target][mask], controls)
    for key, target in (("neighbor_count_all", flow["neighbor_count"]), ("speed_all", flow["speed"]),
                        ("wall_clearance_all", flow["wall_clearance"]),
                        ("observed_acceleration_all", flow["observed_acceleration"]),
                        ("error_all", base_error), ("benefit_all", benefit)):
        correlations[key] = spearman_record(risk, target)
    risk_groups = {}
    for name, selected in (("fit_included", mask), ("fit_excluded", ~mask)):
        values = risk[selected]
        risk_groups[name] = {"n": len(values), "mean": float(values.mean()) if len(values) else None,
                             "median": float(np.median(values)) if len(values) else None}
    return correlations, risk_groups


def run(args):
    started = time.perf_counter()
    identity = json.loads(args.input_identity.read_text())
    if sha(args.risk_summary) != identity["risk_summary_sha256"]:
        raise ValueError("Saved pilot summary differs from pre-analysis input identity")
    reference = json.loads(args.risk_summary.read_text())
    names = {f"{objective}_seed{seed}" for objective in OBJECTIVES for seed in SEEDS}
    if set(reference["runs"]) != names or set(reference["expected_runs"]) != names or reference["pending"]:
        raise ValueError("Require every one of the fifteen original pilot runs")
    source_hashes = reference["provenance"]["data_sha256"]
    data_paths = [args.data_dir / name for name in ("metadata.json", "valid-pilot.npz", "test-pilot.npz")]
    for path in data_paths:
        if sha(path) != source_hashes[path.name]:
            raise ValueError(f"Cached geometry source hash differs: {path.name}")
    metadata = json.loads(data_paths[0].read_text())
    bounds = np.asarray(metadata["bounds"], dtype=np.float64)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Expected finite two-dimensional bounds")
    canonical = reference["runs"]["faithful_seed0"]["splits"]
    frame_fields = ("id", "trajectory", "step", "n_particles", "raw_prefix")
    cache, geometry, raw = {}, {}, {}
    for split in SPLITS:
        frames = canonical[split]["frames"]
        if len(frames) != 36 or len({frame["id"] for frame in frames}) != 36:
            raise ValueError("Expected 36 unique canonical frames")
        geometry[split] = []
        with np.load(args.data_dir / f"{split}-pilot.npz", allow_pickle=False) as data:
            positions = {name: data[name] for name in {frame["trajectory"] for frame in frames}}
            for index, frame in enumerate(frames):
                name, t, n = frame["trajectory"], frame["step"], frame["n_particles"]
                values = positions[name]
                if type(t) is not int or not 7 <= t < len(values) or values.shape[1:] != (n, 2):
                    raise ValueError("Frame identity/geometry shape mismatch")
                types = data[name.replace("position_", "type_")]
                if np.any(types == 3):
                    raise ValueError("The compact pilot supports dynamic particles only")
                if frame["id"] != f"{name}:{t}" or frame["raw_prefix"] != f"{split}_{index:03d}":
                    raise ValueError("Frame join does not match original generator")
                flow = descriptors(values[t-3:t], bounds)
                cache[split, frame["id"]] = flow
                prefix = frame["raw_prefix"]
                raw.update({f"{prefix}_{key}": value for key, value in flow.items()})
                valid = int(flow["fit_valid"].sum())
                geometry[split].append({**{key: frame[key] for key in frame_fields},
                    "fit_valid_particles": valid, "fit_invalid_particles": n-valid, "fit_valid_fraction": valid/n,
                    "outside_box_particles": int(np.sum(flow["wall_clearance"] < 0)),
                    "no_neighbor_particles": int(np.sum(flow["neighbor_count"] == 0))})
    runs = {}
    inputs = {path.name: sha(path) for path in data_paths}
    inputs["risk_benefit_summary"] = sha(args.risk_summary)
    inputs["input_identity"] = sha(args.input_identity)
    for name in sorted(names):
        old = reference["runs"][name]
        if name != f"{old['objective']}_seed{old['seed']}":
            raise ValueError("Model name/identity mismatch")
        path = args.array_dir / f"{name}.npz"
        if sha(path) != old["raw_arrays"]["sha256"]:
            raise ValueError(f"Saved risk/error/benefit arrays changed: {name}")
        inputs[f"risk_arrays/{name}.npz"] = sha(path)
        item = {"objective": old["objective"], "seed": old["seed"],
                "inherited_checkpoint_sha256": old["checkpoint_sha256"], "splits": {}}
        with np.load(path, allow_pickle=False) as arrays:
            expected_keys = {f"{frame['raw_prefix']}_{key}" for split in SPLITS for frame in canonical[split]["frames"]
                             for key in ("risk", "base_vector_se", "dense_vector_se", "signed_benefit")}
            if set(arrays.files) != expected_keys:
                raise ValueError("Unexpected/missing raw array keys")
            for split in SPLITS:
                frames = old["splits"][split]["frames"]
                if [{key: f[key] for key in frame_fields} for f in frames] != [
                        {key: f[key] for key in frame_fields} for f in canonical[split]["frames"]]:
                    raise ValueError("Models do not share exact frame identities and order")
                rows = []
                for frame in frames:
                    prefix = frame["raw_prefix"]
                    risk, base, dense, benefit = [arrays[f"{prefix}_{key}"].astype(np.float64) for key in
                                                 ("risk", "base_vector_se", "dense_vector_se", "signed_benefit")]
                    if not np.isfinite(dense).all() or np.any(dense < 0) or not np.array_equal(benefit, base-dense):
                        raise ValueError("Saved signed benefit does not equal base minus dense")
                    correlations, risk_groups = frame_correlations(cache[split, frame["id"]], risk, base, benefit)
                    rows.append({**{key: frame[key] for key in frame_fields},
                                 "correlations": correlations, "risk_by_fit_eligibility": risk_groups})
                item["splits"][split] = {"frames": rows, "aggregate": aggregate_frames(rows)}
        runs[name] = item
    raw_path = args.output_prefix.with_suffix(".npz")
    with raw_path.open("wb") as stream:
        np.savez_compressed(stream, **raw)
    repo = Path(__file__).resolve().parents[1]
    return {"schema": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exploratory cached compact-pilot associations; no model inference or reserved full-test access",
        "protocol_sha256": sha(args.protocol),
        "source_sha256": {name: sha(repo / name) for name in (
            "research/physical_complexity.py", "research/rank_diagnostics.py", "research/analyze_physical_complexity.py")},
        "inputs_sha256": inputs, "metadata_dt": metadata.get("dt"),
        "units": "coordinates per stored frame; gradients per frame; correlations dimensionless",
        "geometry": geometry, "runs": runs, "summaries": seed_summary(runs),
        "geometry_arrays": {"path": raw_path.name, "sha256": sha(raw_path)},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "postprocessing_seconds": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--risk-summary", type=Path, default=Path("research/results/risk_benefit.json"))
    parser.add_argument("--array-dir", type=Path, default=Path("research/results/risk_benefit_samples"))
    parser.add_argument("--protocol", type=Path, default=Path("research/protocols/physical_complexity_pilot.md"))
    parser.add_argument("--input-identity", type=Path, default=Path("research/protocols/physical_complexity_inputs.json"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if any(args.output_prefix.with_suffix(suffix).exists() for suffix in (".json", ".npz", ".failure.json")):
        parser.error("Fresh output prefix required; preserve previous outcomes")
    try:
        result = run(args)
        args.output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    except Exception as error:
        args.output_prefix.with_suffix(".failure.json").write_text(json.dumps({
            "error": f"{type(error).__name__}: {error}", "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha(args.protocol)}, indent=2)+"\n")
        raise
    print(json.dumps({"models": len(result["runs"]), "unique_observed_frames": sum(map(len, result["geometry"].values())),
                      "postprocessing_seconds": result["postprocessing_seconds"]}))


if __name__ == "__main__":
    main()
