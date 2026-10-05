"""Exploratory saved-array actual-action benefit analysis; no model inference.

See protocols/full_action_benefit_20261005.md. The source protocol is fixed
before new outcome extraction. Whole-frame hindsight choices are not policies.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.stats import rankdata

from research import summarize_full_same_state as strict

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/protocols/full_action_benefit_20261005.md"
POLICIES = strict.POLICIES
ACTIONS = POLICIES[1:]
RISKS = ("current_base_q", "previous_observed_base_q")
UNITS = ("position", "normalized")
PORTFOLIOS = {"all_five": POLICIES, "budgeted_with_base": (POLICIES[0], *POLICIES[2:])}
BENEFIT_METRICS = ("mean_vector_benefit", "mean_coordinate_benefit", "helpful_fraction", "harmful_fraction",
                   "zero_fraction", "mean_alignment_term", "mean_perturbation_cost",
                   "frame_helpful_indicator", "frame_harmful_indicator", "frame_zero_indicator")
SIGNS = ("harmful", "zero", "helpful")
DEGREE_METRICS = ("mean", "maximum", "second_moment", "nonzero_fraction", "gini", "herfindahl", "maximum_share")
ORACLE_METRICS = ("coordinate_mse", "base_minus_oracle", "previous_risk_minus_oracle", "best_non_base_coordinate_mse",
                  "benefit_from_allowing_base", "base_strictly_best", "base_co_best", "best_tie_count")
sha256, read_json, require = strict.sha256, strict.read_json, strict.require


def metric_names():
    result = []
    for unit in UNITS:
        result += [f"error/{unit}/{policy}/coordinate_mse" for policy in POLICIES]
        result += [f"paired_error/{unit}/{left}_minus_{right}" for left, right in itertools.combinations(POLICIES, 2)]
        for action in ACTIONS:
            result += [f"benefit/{unit}/{action}/{name}" for name in BENEFIT_METRICS]
            result += [f"dense_proxy/{unit}/{action}/{name}" for name in ("sign_disagreement_fraction", "opposite_nonzero_fraction")]
            result += [f"dense_proxy/{unit}/{action}/dense_{left}_actual_{right}_fraction" for left in SIGNS for right in SIGNS]
            result += [f"correlation_difference/{unit}/{action}/current_minus_previous"]
            for risk in RISKS:
                result += [f"correlation/{unit}/{action}/{risk}"]
                result += [f"correlation_difference/{unit}/{action}/{risk}/actual_minus_dense"]
                for quartile in range(4):
                    result += [f"risk_profile/{unit}/{action}/{risk}/q{quartile + 1}/{name}"
                               for name in ("particle_fraction", "mean_vector_benefit", "harmful_fraction")]
        for portfolio, policies in PORTFOLIOS.items():
            result += [f"oracle/{unit}/{portfolio}/{name}" for name in ORACLE_METRICS]
            result += [f"oracle/{unit}/{portfolio}/choice/{policy}" for policy in policies]
    for action in ACTIONS:
        result += [f"optional_degree/{action}/{name}" for name in DEGREE_METRICS]
        result += [f"degree_correlation/{action}/{risk}" for risk in RISKS]
    result += [f"optional_jaccard/{left}__{right}" for left, right in itertools.combinations(POLICIES, 2)]
    result += ["risk_agreement/current_vs_previous_spearman"]
    return tuple(result)


METRICS = metric_names()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def real_array(value, name, shape):
    array = np.asarray(value)
    require(array.dtype.kind in "iuf" and array.shape == shape and np.isfinite(array).all(),
            f"Invalid finite numeric {name} shape/dtype")
    return array.astype(np.float64)


def require_close(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-9, atol=1e-15),
            f"Saved arithmetic differs: {name}")


def correlation(x, y):
    require(x.shape == y.shape and x.ndim == 1 and np.isfinite(x).all() and np.isfinite(y).all(),
            "Correlation requires matching finite vectors")
    if len(x) < 2:
        return None, "fewer_than_two_particles"
    x, y = rankdata(x, method="average"), rankdata(y, method="average")
    x -= x.mean(); y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return None, "constant_rank_vector"
    return float(np.clip(x.dot(y) / denominator, -1., 1.)), None


def risk_quartiles(risk):
    """Average ranks keep every exact-score tie together, even across a cut."""
    require(risk.ndim == 1 and len(risk) > 0 and np.isfinite(risk).all(), "Finite nonempty risk required")
    percentile = (rankdata(risk, method="average") - .5) / len(risk)
    return np.minimum(np.floor(percentile * 4).astype(np.int64), 3)


def pair_set(array, n, name):
    array = np.asarray(array)
    require(array.ndim == 2 and array.shape[1] == 2 and array.dtype.kind in "iu", f"Invalid pair array {name}")
    require(np.all(array[:, 0] < array[:, 1]) and np.all(array >= 0) and np.all(array < n),
            f"Non-simple/out-of-range pairs {name}")
    result = set(map(tuple, array.tolist()))
    require(len(result) == len(array), f"Duplicate pairs {name}")
    return result


def degree_statistics(degree):
    degree = np.asarray(degree, dtype=np.float64)
    require(degree.ndim == 1 and len(degree) and np.isfinite(degree).all() and np.all(degree >= 0),
            "Nonempty nonnegative degree vector required")
    total, n = float(degree.sum()), len(degree)
    gini = (2 * np.dot(np.arange(1, n + 1), np.sort(degree)) / (n * total) - (n + 1) / n) if total else 0.
    return {"mean": float(degree.mean()), "maximum": float(degree.max()), "second_moment": float((degree ** 2).mean()),
            "nonzero_fraction": float(np.mean(degree > 0)), "gini": max(float(gini), 0.),
            "herfindahl": float(np.sum((degree / total) ** 2)) if total else None,
            "maximum_share": float(degree.max() / total) if total else None}


def whole_frame_oracle(errors, policies):
    """One global action per observed frame; never a particlewise mixture."""
    values = np.asarray([float(np.mean(errors[policy]) / 2) for policy in policies])
    require(np.isfinite(values).all() and np.all(values >= 0) and policies[0] == "base", "Invalid oracle errors")
    best = float(values.min())
    chosen = int(np.argmin(values))
    forced = float(values[1:].min())
    result = {"coordinate_mse": best, "base_minus_oracle": float(values[0] - best),
              "previous_risk_minus_oracle": float(np.mean(errors[POLICIES[-1]]) / 2 - best),
              "best_non_base_coordinate_mse": forced, "benefit_from_allowing_base": forced - best,
              "base_strictly_best": float(values[0] < forced), "base_co_best": float(values[0] == best),
              "best_tie_count": float(np.sum(values == best))}
    result.update({"choice/" + policy: float(index == chosen) for index, policy in enumerate(policies)})
    return result


def blank_frame(row, status=None):
    status = row["status"] if status is None else status
    return {key: row.get(key) for key in ("source_index", "target_frame", "trajectory_id", "n_particles")} | {
        "status": status, "failure": row.get("failure"), "metrics": {name: None for name in METRICS},
        "undefined_reasons": {name: status + "_frame" for name in METRICS}}


def analyze_frame(row, arrays):
    if row["status"] != "complete":
        return blank_frame(row), {}
    n = row["n_particles"]
    require(type(n) is int and n > 0, "Positive particle count required")
    target = real_array(arrays["target_position"], "target", (n, 2))
    require(array_hash(arrays["target_position"]) == row["target_sha256"], "Target hash differs")
    for key, digest in (("current_observed_history", "observed_history_sha256"),
                        ("previous_observed_history", "previous_observed_history_sha256")):
        real_array(arrays[key], key, (6, n, 2))
        require(str(arrays[key].dtype) == "float32" and array_hash(arrays[key]) == row[digest], "History bytes/dtype differ")
    types = np.asarray(arrays["particle_types"])
    require(types.shape == (n,) and types.dtype.kind in "iu" and not np.any(types == 3), "Invalid WaterDrop types")
    scale = real_array(row["saved_acceleration_normalization"]["std"], "normalization std", (2,))
    require(np.all(scale > 0), "Normalization must be positive")
    risks = {name: real_array(arrays[name], name, (n,)) for name in RISKS}
    require(all(np.all(value > 0) for value in risks.values()), "Risk must be positive")
    predictions = {policy: real_array(arrays["prediction_" + policy], policy, (n, 2)) for policy in POLICIES}
    record = blank_frame(row)
    record["undefined_reasons"] = {}
    metrics, undefined, derived = record["metrics"], record["undefined_reasons"], {}
    record["dense_proxy_sign_counts"] = {}
    def put(name, value, reason=None):
        require(name in metrics and (value is None or math.isfinite(value)), f"Unexpected/nonfinite metric {name}")
        metrics[name] = value
        if value is None:
            require(isinstance(reason, str), "Undefined metric requires a reason")
            undefined[name] = reason
    for unit in UNITS:
        unit_scale = scale if unit == "normalized" else np.ones(2)
        errors = {policy: np.sum(((prediction - target) / unit_scale) ** 2, axis=1)
                  for policy, prediction in predictions.items()}
        for policy in POLICIES:
            require_close(arrays[f"{unit}_vector_se_{policy}"], errors[policy], f"{unit} {policy} vector errors")
            require_close(row["accuracy"][policy][unit + "_coordinate_mse"], errors[policy].mean() / 2,
                          f"{unit} {policy} coordinate MSE")
            put(f"error/{unit}/{policy}/coordinate_mse", float(errors[policy].mean() / 2))
            derived[f"error_{unit}_{policy}"] = errors[policy]
        for left, right in itertools.combinations(POLICIES, 2):
            put(f"paired_error/{unit}/{left}_minus_{right}", float(np.mean(errors[left] - errors[right]) / 2))
        dense_benefit = errors["base"] - errors["dense"]
        dense_sign = np.sign(dense_benefit).astype(np.int64)
        record["dense_proxy_sign_counts"][unit] = {}
        for action in ACTIONS:
            benefit = errors["base"] - errors[action]
            residual = (target - predictions["base"]) / unit_scale
            delta = (predictions[action] - predictions["base"]) / unit_scale
            alignment = 2 * np.sum(residual * delta, axis=1)
            cost = np.sum(delta ** 2, axis=1)
            # Difference-of-squares and the independently formed decomposition
            # incur cancellation when the action is almost identical to base.
            envelope = 32 * np.finfo(np.float64).eps * np.maximum(errors["base"] + errors[action] + cost, 1e-300)
            require(np.all(np.abs(benefit - (alignment - cost)) <= envelope), "Benefit decomposition identity failed")
            if action == "dense":
                require_close(arrays["signed_dense_benefit_" + unit], benefit, f"{unit} dense benefit")
            for label, value in (("mean_vector_benefit", benefit.mean()), ("mean_coordinate_benefit", benefit.mean() / 2),
                    ("helpful_fraction", np.mean(benefit > 0)), ("harmful_fraction", np.mean(benefit < 0)),
                    ("zero_fraction", np.mean(benefit == 0)), ("mean_alignment_term", alignment.mean()),
                    ("mean_perturbation_cost", cost.mean()), ("frame_helpful_indicator", float(benefit.mean() > 0)),
                    ("frame_harmful_indicator", float(benefit.mean() < 0)), ("frame_zero_indicator", float(benefit.mean() == 0))):
                put(f"benefit/{unit}/{action}/{label}", float(value))
            actual_sign = np.sign(benefit).astype(np.int64)
            cross_tab = np.zeros((3, 3), dtype=np.int64)
            np.add.at(cross_tab, (dense_sign + 1, actual_sign + 1), 1)
            record["dense_proxy_sign_counts"][unit][action] = {"rows_dense_columns_actual": list(SIGNS), "counts": cross_tab.tolist()}
            prefix = f"dense_proxy/{unit}/{action}/"
            put(prefix + "sign_disagreement_fraction", float(np.mean(dense_sign != actual_sign)))
            put(prefix + "opposite_nonzero_fraction", float(np.mean(dense_sign * actual_sign < 0)))
            for i, left in enumerate(SIGNS):
                for j, right in enumerate(SIGNS):
                    put(prefix + f"dense_{left}_actual_{right}_fraction", float(cross_tab[i, j] / n))
            for label, value in (("benefit", benefit), ("alignment", alignment), ("perturbation_cost", cost)):
                derived[f"{label}_{unit}_{action}"] = value
            for risk, values in risks.items():
                value, reason = correlation(values, benefit)
                put(f"correlation/{unit}/{action}/{risk}", value, reason)
                dense_value, dense_reason = correlation(values, dense_benefit)
                put(f"correlation_difference/{unit}/{action}/{risk}/actual_minus_dense",
                    value - dense_value if value is not None and dense_value is not None else None,
                    reason or dense_reason)
                bins = risk_quartiles(values)
                derived["quartile_" + risk] = bins
                for quartile in range(4):
                    selected = bins == quartile
                    prefix = f"risk_profile/{unit}/{action}/{risk}/q{quartile + 1}/"
                    put(prefix + "particle_fraction", float(selected.mean()))
                    put(prefix + "mean_vector_benefit", float(benefit[selected].mean()) if selected.any() else None, "empty_risk_quartile")
                    put(prefix + "harmful_fraction", float(np.mean(benefit[selected] < 0)) if selected.any() else None, "empty_risk_quartile")
            current, previous = [metrics[f"correlation/{unit}/{action}/{risk}"] for risk in RISKS]
            put(f"correlation_difference/{unit}/{action}/current_minus_previous",
                current - previous if current is not None and previous is not None else None, "required_correlation_undefined")
        for portfolio, policies in PORTFOLIOS.items():
            for key, value in whole_frame_oracle(errors, policies).items():
                put(f"oracle/{unit}/{portfolio}/{key}", value)
    base = pair_set(arrays["candidate_base_pairs_float64"], n, "base")
    annulus = pair_set(arrays["candidate_annulus_pairs_float64"], n, "annulus")
    require(not base & annulus, "Base and annulus overlap")
    optional = {}
    audit = row["graph_audit"]
    require(audit["base_pairs"] == len(base) and audit["annulus_pairs"] == len(annulus)
            and audit["candidate_pairs"] == len(base | annulus) and audit["optional_budget"] == len(annulus) // 4,
            "Graph audit candidate counts differ")
    for policy in POLICIES:
        pairs = arrays["pairs_" + policy]
        selected = pair_set(pairs, n, policy)
        require(base <= selected <= base | annulus, "Selected graph drops mandatory or adds unavailable pairs")
        optional[policy] = selected - base
        expected = 0 if policy == "base" else len(annulus) if policy == "dense" else len(annulus) // 4
        require(len(optional[policy]) == expected, "Selected graph budget differs")
        counts = audit["policies"][policy]
        require(counts["retained_pairs"] == len(selected) and counts["retained_optional_pairs"] == expected
                and counts["directed_edges"] == 2 * len(selected), "Recorded selected counts differ")
        first = next(call for call in row["timed_calls"] if call["method"] == policy)
        require(array_hash(pairs) == first["pair_sha256"], "Selected pairs differ from first timed call")
        if policy == "base":
            continue
        degree = np.zeros(n, dtype=np.int64)
        for i, j in optional[policy]:
            degree[i] += 1; degree[j] += 1
        derived["optional_degree_" + policy] = degree
        for key, value in degree_statistics(degree).items():
            put(f"optional_degree/{policy}/{key}", value, "no_optional_pairs")
        for risk, values in risks.items():
            value, reason = correlation(values, degree.astype(np.float64))
            put(f"degree_correlation/{policy}/{risk}", value, reason)
    for left, right in itertools.combinations(POLICIES, 2):
        union, intersection = optional[left] | optional[right], optional[left] & optional[right]
        jaccard = len(intersection) / len(union) if union else 1.
        saved = audit["overlaps"][left + "__" + right]
        require(saved["optional_intersection"] == len(intersection) and saved["optional_union"] == len(union), "Overlap counts differ")
        require_close(saved["optional_jaccard"], jaccard, "optional Jaccard")
        put(f"optional_jaccard/{left}__{right}", float(jaccard))
    value, reason = correlation(risks[RISKS[0]], risks[RISKS[1]])
    put("risk_agreement/current_vs_previous_spearman", value, reason)
    require(set(undefined) == {key for key, value in metrics.items() if value is None}, "Unaccounted undefined metric")
    record["risk_quartile_counts"] = {risk: np.bincount(derived["quartile_" + risk], minlength=4).tolist() for risk in RISKS}
    return record, derived


def average_required(values):
    require(all(value is None or math.isfinite(value) for value in values), "Nonfinite aggregate input")
    return float(np.mean(values)) if values and all(value is not None for value in values) else None


def aggregate_frames(frames, source_indices=strict.SOURCE_INDICES, target_frames=strict.TARGET_FRAMES):
    expected = set(itertools.product(source_indices, target_frames))
    keys = [(row["source_index"], row["target_frame"]) for row in frames]
    require(len(keys) == len(set(keys)) and set(keys) <= expected, "Unexpected or duplicate analysis frame")
    indexed = dict(zip(keys, frames))
    complete = [indexed.get(key, blank_frame({"source_index": key[0], "target_frame": key[1]}, "missing")) for key in sorted(expected)]
    trajectories = {str(source): {name: average_required([row["metrics"][name] for row in complete if row["source_index"] == source])
                                 for name in METRICS} for source in source_indices}
    metrics = {}
    for name in METRICS:
        values = [trajectories[str(source)][name] for source in source_indices]
        metrics[name] = {"equal_trajectory_mean": average_required(values), "required_frames": len(expected),
            "defined_frames": sum(row["metrics"][name] is not None for row in complete), "required_trajectories": len(source_indices),
            "defined_trajectories": sum(value is not None for value in values),
            "undefined_reason_counts": dict(Counter(row["undefined_reasons"][name] for row in complete if row["metrics"][name] is None))}
    trajectory_outcomes = {}
    for unit in UNITS:
        for action in ACTIONS:
            values = [trajectories[str(source)][f"benefit/{unit}/{action}/mean_vector_benefit"] for source in source_indices]
            counts = {"helpful": sum(value is not None and value > 0 for value in values),
                      "harmful": sum(value is not None and value < 0 for value in values),
                      "zero": sum(value is not None and value == 0 for value in values),
                      "undefined": sum(value is None for value in values)}
            trajectory_outcomes[f"{unit}/{action}"] = {"required_trajectories": len(values), "counts": counts,
                "fractions": {label: count / len(values) if counts["undefined"] == 0 else None
                              for label, count in counts.items() if label != "undefined"}}
    return {"required_frames": len(expected), "observed_frames": len(frames),
            "frame_status_counts": dict(Counter(row["status"] for row in complete)), "trajectories": trajectories,
            "trajectory_action_outcomes": trajectory_outcomes, "metrics": metrics}


def across_seeds(values):
    require(len(values) == 3, "Exactly three paired training seeds required")
    mean = average_required(values)
    return {"seed_values": values, "mean": mean, "sample_seed_sd": float(np.std(values, ddof=1)) if mean is not None else None,
            "required_seeds": 3, "defined_seeds": sum(value is not None for value in values)}


def inventory(root):
    """Enumerate only canonical evidence paths, before reading outcome values."""
    found = {}
    for objective in strict.OBJECTIVES:
        for seed in strict.SEEDS:
            directory = root / f"{objective}_seed{seed}"
            paths = [directory / name for name in ("protocol.json", "result.json", "status.json")]
            paths += [directory / f"trajectory_{source:06d}_target_{target:04d}.{suffix}"
                      for source, target in strict.EXPECTED_KEYS for suffix in ("json", "npz")]
            for path in paths:
                if path.exists():
                    require(path.is_file() and not path.is_symlink(), "Expected regular nonsymlink input file")
                    found[str(path.relative_to(root))] = sha256(path)
    return found


def verify_inventory(root, before):
    after = inventory(root)
    require(before == after, "Input snapshot changed during analysis")


def render_report(result):
    def value(record):
        return "undefined" if record["mean"] is None else f"{record['mean']:.7g} ± {record['sample_seed_sd']:.4g}"
    lines = ["# Exploratory actual-action benefit analysis", "", "Post-inspection saved-array analysis; no inference, retraining or new test records.", "",
             "Means and sample SDs use exactly three seed means. All per-particle benefits, frame/trajectory values, undefined counts, input hashes and failures are retained.", "",
             "| Objective | Action | Mean normalized vector benefit | Harmful particle fraction | Previous-risk / benefit Spearman |",
             "|---|---|---:|---:|---:|"]
    for objective in strict.OBJECTIVES:
        metrics = result["objectives"][objective]
        for action in ACTIONS:
            keys = (f"benefit/normalized/{action}/mean_vector_benefit", f"benefit/normalized/{action}/harmful_fraction",
                    f"correlation/normalized/{action}/previous_observed_base_q")
            lines.append(f"| {objective} | {action} | " + " | ".join(value(metrics[key]) for key in keys) + " |")
    lines += ["", "Positive benefit means base error minus actual-action error; negative values are retained. Benefits are whole-graph interventions, share base residual algebraically, and are not marginal edge utility. Position-space results and current-risk profiles remain separate in JSON.", "",
              "| Objective | Hindsight portfolio | Normalized coordinate MSE | Benefit from allowing base | Base strictly best fraction |",
              "|---|---|---:|---:|---:|"]
    for objective in strict.OBJECTIVES:
        metrics = result["objectives"][objective]
        for portfolio in PORTFOLIOS:
            keys = [f"oracle/normalized/{portfolio}/{name}" for name in ("coordinate_mse", "benefit_from_allowing_base", "base_strictly_best")]
            lines.append(f"| {objective} | {portfolio} | " + " | ".join(value(metrics[key]) for key in keys) + " |")
    lines += ["", "The oracle uses one complete prediction per observed frame and target information in hindsight. It is not deployable, not a particlewise mixture, and not an optimum over arbitrary edge subsets. The all-five portfolio includes the more expensive dense graph. No speedup, significance or independent confirmation is claimed."]
    return "\n".join(lines) + "\n"


def run(args):
    root, output = args.evaluation_root.resolve(), args.output_dir.resolve()
    require(not output.exists(), "Existing analysis directory must be preserved")
    require(root != output and root not in output.parents, "Analysis output must be outside frozen evidence root")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = [Path(__file__).resolve(), Path(strict.__file__).resolve(), Path(strict.shared.__file__).resolve(),
               ROOT / "research/full_same_state.py", args.protocol.resolve(), args.training_protocol.resolve(), args.companion_protocol.resolve()]
    source_hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path) for path in sources}
    source_paths = dict(zip(source_hashes, sources))
    identity = {"schema": 1, "scope": "post_inspection_exploratory_saved_arrays", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": source_hashes, "input_root": str(root), "software": {"python": platform.python_version(),
                    "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
                "protocol_sha256": sha256(args.protocol)}
    atomic_json(output / "status.json", {"state": "running", **identity})
    try:
        before = inventory(root)
        atomic_json(output / "input_identity.json", {**identity, "input_files_sha256": before})
        checked = strict.summarize(root, sha256(args.training_protocol), sha256(args.companion_protocol),
                                   sha256(ROOT / "research/full_same_state.py"))
        validation_errors = checked["consistency_errors"] + [f"{run['objective']} seed {run['seed']}: {error}"
                            for run in checked["runs"] for error in run["errors"]]
        require(checked["state"] != "invalid" and not checked["consistency_errors"],
                "Strict input validation failed: " + str(validation_errors))
        runs = {}
        for run_info in checked["runs"]:
            label = f"{run_info['objective']}_seed{run_info['seed']}"
            directory = root / label
            output_frames = output / "frames" / label
            output_frames.mkdir(parents=True)
            rows, index = [], []
            for compact in run_info["frames"]:
                path = directory / compact["record_file"]
                row, record_sha = read_json(path)
                require(record_sha == compact["record_sha256"], "Record changed after strict validation")
                array_path = directory / compact["array_file"]
                require(sha256(array_path) == compact["array_sha256"], "Array changed after strict validation")
                if row["status"] == "complete":
                    with np.load(array_path, allow_pickle=False) as archive:
                        record, arrays = analyze_frame(row, archive)
                else:
                    record, arrays = analyze_frame(row, {})
                record["input_record_sha256"] = record_sha
                record["input_array_sha256"] = compact["array_sha256"]
                stem = path.stem
                if arrays:
                    derived_path = output_frames / (stem + ".npz")
                    with derived_path.open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    record["derived_array_file"] = str(derived_path.relative_to(output))
                    record["derived_array_sha256"] = sha256(derived_path)
                record_path = output_frames / (stem + ".json")
                atomic_json(record_path, record)
                index.append({"source_index": row["source_index"], "target_frame": row["target_frame"], "status": row["status"],
                              "record_file": str(record_path.relative_to(output)), "record_sha256": sha256(record_path)})
                rows.append(record)
            aggregate = aggregate_frames(rows)
            # Missing/uncommitted evaluator status is not promoted to a valid result,
            # even if all expected per-frame artifacts happen to be on disk.
            if not run_info["eligible"]:
                for metric in aggregate["metrics"].values():
                    metric["equal_trajectory_mean"] = None
                    metric["ineligible_run_reason"] = run_info["state"]
                for outcome in aggregate["trajectory_action_outcomes"].values():
                    outcome["fractions"] = dict.fromkeys(outcome["fractions"], None)
                    outcome["ineligible_run_reason"] = run_info["state"]
            runs[label] = {key: run_info.get(key) for key in ("objective", "seed", "state", "eligible", "errors", "checkpoint_sha256", "protocol_sha256", "result_sha256")}
            runs[label].update(frames=index, aggregate=aggregate)
        objectives = {objective: {name: across_seeds([runs[f"{objective}_seed{seed}"]["aggregate"]["metrics"][name]["equal_trajectory_mean"]
                         for seed in strict.SEEDS]) for name in METRICS} for objective in strict.OBJECTIVES}
        trajectory_objectives = {objective: {f"{unit}/{action}/{label}": across_seeds([
                runs[f"{objective}_seed{seed}"]["aggregate"]["trajectory_action_outcomes"][f"{unit}/{action}"]["fractions"][label]
                for seed in strict.SEEDS]) for unit in UNITS for action in ACTIONS for label in ("helpful", "harmful", "zero")}
                for objective in strict.OBJECTIVES}
        objective_differences = {}
        for name in METRICS:
            left, right = objectives["nll"][name]["seed_values"], objectives["faithful"][name]["seed_values"]
            objective_differences[name] = across_seeds([a - b if a is not None and b is not None else None for a, b in zip(left, right)])
        verify_inventory(root, before)
        require(all(sha256(source_paths[name]) == digest for name, digest in source_hashes.items()), "Analysis source changed during execution")
        result = {**identity, "state": checked["state"], "required_models": 6, "required_frames_per_model": 297,
                  "input_snapshot_sha256": strict.shared.canonical_hash(before), "input_files_verified_after": len(before),
                  "input_identity_file": "input_identity.json", "input_identity_sha256": sha256(output / "input_identity.json"),
                  "interpretation": {"scope": "exploratory, after inspection; existing observed-history teacher-forced actions only",
                      "aggregation": "particles within frame; eleven frames within trajectory; 27 trajectories within seed; exactly three seeds with sample SD",
                      "missing": "all required observations; null for any missing/failed/undefined required metric; no available-case means",
                      "oracle": "hindsight choice of one whole-frame action in fixed portfolio; not deployable and not arbitrary-edge optimality",
                      "benefit": "signed complete-graph-action residual reduction; shared base residual; no causal/marginal-edge inference"},
                  "runs": runs, "objectives": objectives, "trajectory_action_outcomes": trajectory_objectives,
                  "paired_nll_minus_faithful": objective_differences,
                  "analysis_wall_seconds": time.perf_counter() - started}
        atomic_json(output / "results.json", result)
        (output / "report.md").write_text(render_report(result))
        atomic_json(output / "status.json", {"state": "complete" if checked["state"] == "complete" else "incomplete",
                    "results_sha256": sha256(output / "results.json"), "report_sha256": sha256(output / "report.md"),
                    "completed_utc": datetime.now(timezone.utc).isoformat(), "input_files_verified_after": len(before)})
        return result
    except BaseException as error:
        atomic_json(output / "status.json", {"state": "error", "error_type": type(error).__name__, "error": str(error),
                    "generated_utc": datetime.now(timezone.utc).isoformat(), "partial_outputs_preserved": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--training-protocol", type=Path, default=ROOT / "research/protocols/full_waterdrop_100k.md")
    parser.add_argument("--companion-protocol", type=Path, default=ROOT / "research/protocols/full_same_state_diagnostic.md")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"state": result["state"], "output_dir": str(args.output_dir),
                      "analysis_wall_seconds": result["analysis_wall_seconds"]}, allow_nan=False))


if __name__ == "__main__":
    main()
