"""Exact exploratory optional-exposure decomposition of saved predictions only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from research import analyze_full_action_benefit as action

strict = action.strict
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/protocols/optional_exposure_decomposition_20261006.md"
RISK = "previous-observed-base-risk25"
POLICIES = (RISK, "random25", "speed25")
PAIRS = {"risk_minus_random": (RISK, "random25"), "risk_minus_speed": (RISK, "speed25"),
         "speed_minus_random": ("speed25", "random25")}
PRIMARY_GROUPS = ("neither", "left_only", "right_only", "both")
BOTH_GROUPS = ("both_less", "both_equal", "both_more")
GROUPS = PRIMARY_GROUPS + BOTH_GROUPS
QUANTITIES = ("error", "alignment", "cost", "degree")
WHOLE = ("left_error", "right_error", "error_difference", "alignment_difference", "cost_difference", "degree_difference", "arithmetic_scale")
GROUP_METRICS = ("particle_fraction",) + tuple(f"{q}_contribution" for q in QUANTITIES) + tuple(f"conditional_mean_{q}_difference" for q in QUANTITIES)
METRICS = tuple(f"{pair}/whole/{metric}" for pair in PAIRS for metric in WHOLE) + tuple(
    f"{pair}/{group}/{metric}" for pair in PAIRS for group in GROUPS for metric in GROUP_METRICS)
ACTION_IDENTITIES = {
    "results.json": "c9a608d3f17cfc84001020b27048ef448958daa2705d135be5f4492e8bbdc008",
    "input_identity.json": "647c94d886a9d8ee6f2788615c6b5fb95758ff97ca0f94d8ec69a10f9ee41718",
    "status.json": "04a221181efc1fdb27bc6b2c24712bde280429c3bc257cc060be402b7b2d262a"}
DEPENDENCIES = {
    "research/analyze_full_action_benefit.py": "e8f167646986e08b650e58b4388cd56e5277ec200708eb3d9788eaedf39c34b3",
    "research/summarize_full_same_state.py": "2d8de802b4109c9dc84cdbb5c53a72b28942cb6fdbacfd8437ea884bb7487065",
    "research/summarize_full_rollouts.py": "fee246c0abe0a289374dcc7b383be8827579fdeb711f966ae9d2a3480048070b",
    "research/full_same_state.py": "ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592",
    "research/protocols/full_waterdrop_100k.md": "5566a586f93014f1c25a488f748f3ee1d1da61773f4865a03a12d8a7ed208926",
    "research/protocols/full_same_state_diagnostic.md": "e59a351b14753802d43bbfaec69d9d45e009c895330d599a2848bc268783d381",
    "research/protocols/full_action_benefit_20261005.md": "f9e779324fc54ad7d8404f5402acfc4e8386e80b5c306db0b425513995d5f0c5"}
require, sha256, read_json, atomic_json = action.require, action.sha256, action.read_json, action.atomic_json


def identity_check(actual, expected, magnitude, terms, name):
    """Predetermined eps64 cancellation envelope, never fitted to outcomes."""
    a, b, scale = np.asarray(actual), np.asarray(expected), np.asarray(magnitude)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.isfinite(scale).all() and np.all(scale >= 0), f"Nonfinite identity inputs: {name}")
    bound = 256 * np.finfo(np.float64).eps * max(int(terms), 1) * np.maximum(scale, 1e-300)
    difference = np.abs(a - b)
    require(np.all(difference <= bound), f"Additive identity failed: {name}")
    return {"name": name, "max_abs_residual": float(np.max(difference)), "max_absolute_bound": float(np.max(bound)), "terms": int(terms)}


def blank_frame(row, status=None, reason=None):
    status = status or row.get("status", "missing")
    reason = reason or f"{status}_frame"
    return {key: row.get(key) for key in ("source_index", "target_frame", "trajectory_id", "n_particles")} | {
        "status": status, "failure": row.get("failure"), "metrics": dict.fromkeys(METRICS),
        "undefined_reasons": dict.fromkeys(METRICS, reason), "pairs": {}, "identity_checks": []}


def exposure_masks(left, right):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.ndim == 1 and len(left) > 0
            and left.dtype.kind in "iu" and right.dtype.kind in "iu"
            and np.all(left >= 0) and np.all(right >= 0), "Invalid optional incident degree vectors")
    a, b = left > 0, right > 0
    masks = {"neither": ~a & ~b, "left_only": a & ~b, "right_only": ~a & b, "both": a & b,
        "both_less": a & b & (left < right), "both_equal": a & b & (left == right), "both_more": a & b & (left > right)}
    require(np.all(sum(masks[g].astype(int) for g in PRIMARY_GROUPS) == 1), "Primary masks must partition every particle")
    require(np.array_equal(sum(masks[g].astype(int) for g in BOTH_GROUPS), masks["both"].astype(int)), "Both subgroups must partition both")
    return masks


def check_partition_metrics(metrics, terms=1):
    checks = []
    for pair in PAIRS:
        if metrics[f"{pair}/whole/error_difference"] is None:
            continue
        scale = metrics[f"{pair}/whole/arithmetic_scale"]
        for quantity in QUANTITIES:
            mag = scale if quantity != "degree" else max(1., abs(metrics[f"{pair}/whole/degree_difference"]))
            checks.append(identity_check(sum(metrics[f"{pair}/{g}/{quantity}_contribution"] for g in PRIMARY_GROUPS),
                metrics[f"{pair}/whole/{quantity}_difference"], mag, terms, f"{pair}/{quantity}/four_groups"))
            checks.append(identity_check(sum(metrics[f"{pair}/{g}/{quantity}_contribution"] for g in BOTH_GROUPS),
                metrics[f"{pair}/both/{quantity}_contribution"], mag, terms, f"{pair}/{quantity}/both_subgroups"))
        checks.append(identity_check(sum(metrics[f"{pair}/{g}/particle_fraction"] for g in PRIMARY_GROUPS), 1., 1., terms, f"{pair}/fractions"))
        checks.append(identity_check(sum(metrics[f"{pair}/{g}/particle_fraction"] for g in BOTH_GROUPS),
            metrics[f"{pair}/both/particle_fraction"], 1., terms, f"{pair}/both_fractions"))
        for group in GROUPS:
            checks.append(identity_check(metrics[f"{pair}/{group}/error_contribution"],
                metrics[f"{pair}/{group}/cost_contribution"] - metrics[f"{pair}/{group}/alignment_contribution"],
                scale, terms, f"{pair}/{group}/cost_minus_alignment"))
        checks.append(identity_check(metrics[f"{pair}/whole/error_difference"],
            metrics[f"{pair}/whole/cost_difference"] - metrics[f"{pair}/whole/alignment_difference"],
            scale, terms, f"{pair}/whole/cost_minus_alignment"))
    return checks


def decompose_predictions(row, target, base, predictions, scale, degrees):
    """Pure synthetic-testable core; caller enforces immutable input provenance."""
    n = row["n_particles"]
    require(type(n) is int and n > 0, "Positive particle count required")
    target = action.real_array(target, "target", (n, 2))
    base = action.real_array(base, "base", (n, 2))
    scale = action.real_array(scale, "std", (2,))
    require(np.all(scale > 0), "Positive acceleration scale required")
    record = blank_frame(row, "complete")
    residual = (target - base) / scale
    base_error = np.sum(residual ** 2, axis=1) / 2
    derived = {"normalized_base_residual": residual, "normalized_coordinate_error_base": base_error}
    available = {}
    for policy in POLICIES:
        if policy not in predictions or policy not in degrees:
            continue
        prediction = action.real_array(predictions[policy], policy, (n, 2))
        degree = np.asarray(degrees[policy])
        require(degree.shape == (n,) and degree.dtype.kind in "iu" and np.all(degree >= 0), "Invalid optional degrees")
        delta = (prediction - base) / scale
        error = np.sum(((prediction - target) / scale) ** 2, axis=1) / 2
        alignment = 2 * np.sum(residual * delta, axis=1) / 2
        cost = np.sum(delta ** 2, axis=1) / 2
        available[policy] = {"error": error, "alignment": alignment, "cost": cost, "degree": degree.astype(np.int64)}
        derived.update({f"normalized_prediction_change_{policy}": delta, f"normalized_coordinate_error_{policy}": error,
            f"normalized_coordinate_alignment_{policy}": alignment, f"normalized_coordinate_cost_{policy}": cost,
            f"optional_degree_{policy}": degree})
    for pair, (left, right) in PAIRS.items():
        absent = [p for p in (left, right) if p not in available]
        if absent:
            record["pairs"][pair] = {"status": "missing_action", "actions": absent}
            for key in METRICS:
                if key.startswith(pair + "/"):
                    record["undefined_reasons"][key] = "missing_action:" + ",".join(absent)
            continue
        a, b = available[left], available[right]
        require(int(a["degree"].sum()) == int(b["degree"].sum()), "Sparse comparisons must have equal optional budgets")
        masks = exposure_masks(a["degree"], b["degree"])
        differences = {q: a[q] - b[q] for q in QUANTITIES}
        magnitude = base_error + a["error"] + b["error"] + a["cost"] + b["cost"] + np.abs(a["alignment"]) + np.abs(b["alignment"])
        record["identity_checks"].append(identity_check(differences["error"], differences["cost"] - differences["alignment"], magnitude, 2, pair + "/particle_identity"))
        whole = {"left_error": float(a["error"].mean()), "right_error": float(b["error"].mean()),
            "arithmetic_scale": float(magnitude.mean()), **{q + "_difference": float(v.mean()) for q, v in differences.items()}}
        require(whole["degree_difference"] == 0., "Equal budgets require zero whole-frame optional-degree difference")
        for key, value in whole.items():
            name = f"{pair}/whole/{key}"; record["metrics"][name] = value; record["undefined_reasons"].pop(name)
        record["pairs"][pair] = {"status": "complete", "left": left, "right": right, "groups": {}}
        primary_ids = np.full(n, -1, dtype=np.int8); both_ids = np.full(n, -1, dtype=np.int8)
        for group, mask in masks.items():
            count = int(mask.sum())
            stats = {"particle_fraction": count / n}
            for q, difference in differences.items():
                stats[f"{q}_contribution"] = float(np.sum(difference[mask]) / n)
                stats[f"conditional_mean_{q}_difference"] = float(np.mean(difference[mask])) if count else None
            record["pairs"][pair]["groups"][group] = {"particles": count, **stats}
            for key, value in stats.items():
                name = f"{pair}/{group}/{key}"; record["metrics"][name] = value
                if value is None:
                    record["undefined_reasons"][name] = "empty_group"
                else:
                    record["undefined_reasons"].pop(name)
            if group in PRIMARY_GROUPS:
                primary_ids[mask] = PRIMARY_GROUPS.index(group)
            else:
                both_ids[mask] = BOTH_GROUPS.index(group)
        for q, values in differences.items():
            derived[f"{pair}__{q}_difference"] = values
        derived[f"{pair}__primary_group_id"] = primary_ids
        derived[f"{pair}__both_degree_subgroup_id"] = both_ids
    if any(p["status"] != "complete" for p in record["pairs"].values()):
        record["status"] = "partial"
    record["identity_checks"] += check_partition_metrics(record["metrics"], n)
    require(set(record["undefined_reasons"]) == {key for key, value in record["metrics"].items() if value is None}, "Unaccounted undefined metrics")
    return record, derived


def decompose_frame(row, arrays, saved_action_record, saved_action_arrays):
    if row["status"] != "complete":
        return blank_frame(row), {}
    replay_record, replay_arrays = action.analyze_frame(row, arrays)
    # All previous action analysis outcomes are checked, not just favorable arms.
    for key, value in replay_record.items():
        require(saved_action_record.get(key) == value, f"Saved action record replay differs: {key}")
    require(set(replay_arrays) == set(saved_action_arrays), "Saved action derived-array keys differ")
    for key, value in replay_arrays.items():
        require(np.asarray(saved_action_arrays[key]).dtype == value.dtype, f"Saved action dtype differs: {key}")
        action.require_close(saved_action_arrays[key], value, f"action replay {key}")
    predictions = {policy: arrays["prediction_" + policy] for policy in POLICIES}
    degrees = {policy: replay_arrays["optional_degree_" + policy] for policy in POLICIES}
    record, derived = decompose_predictions(row, arrays["target_position"], arrays["prediction_base"],
        predictions, row["saved_acceleration_normalization"]["std"], degrees)
    for policy in POLICIES:
        for ours, previous in (("error", "error"), ("alignment", "alignment"), ("cost", "perturbation_cost")):
            action.require_close(derived[f"normalized_coordinate_{ours}_{policy}"],
                replay_arrays[f"{previous}_normalized_{policy}"] / 2, f"coordinate factor2 {policy}/{ours}")
    record["action_replay"] = {"passed": True, "record_keys": len(replay_record), "derived_arrays": len(replay_arrays)}
    return record, derived


def weighted_conditionals(metrics):
    result = {}
    for pair in PAIRS:
        for group in GROUPS:
            fraction = metrics[f"{pair}/{group}/particle_fraction"]
            for q in QUANTITIES:
                contribution = metrics[f"{pair}/{group}/{q}_contribution"]
                reason = "missing_required_input" if fraction is None or contribution is None else "zero_weighted_group_fraction" if fraction == 0 else None
                result[f"{pair}/{group}/{q}"] = {"value": contribution / fraction if reason is None else None,
                    "reason": reason, "weighted_particle_fraction": fraction, "weighted_contribution": contribution}
    return result


def aggregate_frames(frames, source_indices=strict.SOURCE_INDICES, target_frames=strict.TARGET_FRAMES):
    expected = set(itertools.product(source_indices, target_frames))
    keys = [(r["source_index"], r["target_frame"]) for r in frames]
    require(len(set(keys)) == len(keys) and set(keys) <= expected, "Duplicate or unexpected decomposition frames")
    indexed = dict(zip(keys, frames))
    complete = [indexed.get(key, blank_frame({"source_index": key[0], "target_frame": key[1]}, "missing")) for key in sorted(expected)]
    trajectories = {}
    for source in source_indices:
        rows = [r for r in complete if r["source_index"] == source]
        metrics = {name: action.average_required([r["metrics"][name] for r in rows]) for name in METRICS}
        trajectories[str(source)] = {"metrics": metrics, "weighted_conditionals": weighted_conditionals(metrics),
            "identity_checks": check_partition_metrics(metrics, len(rows)), "required_frames": len(target_frames),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "undefined_reason_counts": {name: dict(Counter(r["undefined_reasons"][name] for r in rows if r["metrics"][name] is None)) for name in METRICS}}
    metrics = {name: action.average_required([trajectories[str(source)]["metrics"][name] for source in source_indices]) for name in METRICS}
    return {"required_frames": len(expected), "observed_frames": len(frames), "required_trajectories": len(source_indices),
        "frame_status_counts": dict(Counter(r["status"] for r in complete)), "metrics": metrics,
        "metric_coverage": {name: {"defined_frames": sum(r["metrics"][name] is not None for r in complete),
            "defined_trajectories": sum(t["metrics"][name] is not None for t in trajectories.values()),
            "undefined_reason_counts": dict(Counter(r["undefined_reasons"][name] for r in complete if r["metrics"][name] is None))} for name in METRICS},
        "trajectories": trajectories, "weighted_conditionals": weighted_conditionals(metrics),
        "identity_checks": check_partition_metrics(metrics, len(expected))}


def summarize_objectives(runs):
    result = {}
    for objective in strict.OBJECTIVES:
        models = [runs[f"{objective}_seed{seed}"]["aggregate"] for seed in strict.SEEDS]
        metrics = {name: action.across_seeds([m["metrics"][name] for m in models]) for name in METRICS}
        result[objective] = {"metrics": metrics, "weighted_conditionals": {
            name: action.across_seeds([m["weighted_conditionals"][name]["value"] for m in models]) for name in models[0]["weighted_conditionals"]},
            "mean_identity_checks": check_partition_metrics({k: v["mean"] for k, v in metrics.items()}, 3)}
    return result


def checked_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Input path escapes evidence root")
    require(path.is_file() and not (root / relative).is_symlink(), "Required regular nonsymlink input missing")
    return path


def render_report(result):
    def fmt(value):
        return "undefined" if value["mean"] is None else f"{value['mean']:.8g} ± {value['sample_seed_sd']:.5g}"
    lines = ["# Exploratory optional-exposure error decomposition", "", "Exact decomposition of previously inspected, original no-loop observed-history predictions. No new inference or permutation null.", "",
        "All entries are normalized coordinate squared-error differences, left minus right; positive error favors the right action. Three equal-trajectory seed means and sample SDs are descriptive.", "",
        "| Objective | Pair | Whole error gap | Neither | Left only | Right only | Both |", "|---|---|---:|---:|---:|---:|---:|"]
    for objective in strict.OBJECTIVES:
        metrics = result["objectives"][objective]["metrics"]
        for pair in PAIRS:
            keys = [f"{pair}/whole/error_difference"] + [f"{pair}/{g}/error_contribution" for g in PRIMARY_GROUPS]
            lines.append(f"| {objective} | {pair} | " + " | ".join(fmt(metrics[k]) for k in keys) + " |")
    lines += ["", "The four unconditional contributions sum to the whole error gap. Within each group, error contribution = cost-difference contribution minus alignment-difference contribution; all quantities divide vector terms by 2. The supplementary less/equal/more-degree subgroups partition only the both-covered group.", "",
        "Conditional means remain null for empty groups and are never averaged over available frames only. Separately labeled weighted conditionals divide the fixed-weight contribution by its fixed-weight group fraction. Missing required scientific inputs propagate to the relevant full-population means.", "",
        "Exposure is selected by policy and is not a pretreatment confounder. Multiple message-passing blocks can affect particles without directly incident optional edges. This partition does not identify marginal edge value, establish a causal concentration mechanism, prove autonomous stability or measure inference speed."]
    return "\n".join(lines) + "\n"


def run(args):
    evaluation, previous, output = args.evaluation_root.resolve(), args.action_root.resolve(), args.output_dir.resolve()
    require(not output.exists(), "Preserve existing attempt/output directory")
    require(all(output != source and source not in output.parents for source in (evaluation, previous)), "Output must be outside immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    snapshots, failures = {}, []
    source_paths = {**{name: ROOT / name for name in DEPENDENCIES}, str(Path(__file__).resolve()): Path(__file__).resolve(), str(PROTOCOL): PROTOCOL}
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    identity = {"schema": 1, "scope": "post_inspection_original_100k_optional_exposure_decomposition", "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes, "protocol_sha256": sha256(PROTOCOL), "evaluation_root": str(evaluation), "action_root": str(previous),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()}}
    atomic_json(output / "status.json", {"state": "running", **identity})
    try:
        require(all(source_hashes[name] == digest for name, digest in DEPENDENCIES.items()), "Frozen dependency source/protocol changed")
        for name, digest in ACTION_IDENTITIES.items():
            path = checked_path(previous, name)
            require(sha256(path) == digest, f"Pinned previous analysis differs: {name}")
            snapshots[str(path)] = digest
        anchor = read_json(previous / "input_identity.json")[0]
        old_result = read_json(previous / "results.json")[0]
        old_status = read_json(previous / "status.json")[0]
        require(old_status["state"] == "complete" and old_status["results_sha256"] == ACTION_IDENTITIES["results.json"], "Previous analysis incomplete")
        expected = anchor["input_files_sha256"]
        before = action.inventory(evaluation)
        changed = {name for name in set(before) | set(expected) if before.get(name) != expected.get(name)}
        snapshots.update({str(evaluation / name): digest for name, digest in before.items()})
        atomic_json(output / "input_identity.json", {**identity, "pinned_action_identity": ACTION_IDENTITIES,
            "evaluation_expected_files_sha256": expected, "evaluation_observed_files_sha256": before,
            "evaluation_missing_or_changed": sorted(changed)})
        checked = strict.summarize(evaluation, DEPENDENCIES["research/protocols/full_waterdrop_100k.md"],
            DEPENDENCIES["research/protocols/full_same_state_diagnostic.md"], DEPENDENCIES["research/full_same_state.py"])
        runs = {}
        for run_info in checked["runs"]:
            label = f"{run_info['objective']}_seed{run_info['seed']}"
            folder = output / "frames" / label; folder.mkdir(parents=True)
            original_frames = {(r["source_index"], r["target_frame"]): r for r in run_info["frames"]}
            previous_frames = {(r["source_index"], r["target_frame"]): r for r in old_result["runs"][label]["frames"]}
            require(len(previous_frames) == len(strict.EXPECTED_KEYS) and set(previous_frames) == set(strict.EXPECTED_KEYS), "Previous analysis does not preserve all frames")
            run_issues = list(run_info["errors"]) + list(checked["consistency_errors"])
            if any(name.startswith(label + "/") for name in changed):
                run_issues.append("Pinned original input missing or changed")
            if not run_info["eligible"]:
                run_issues.append("Original model result/provenance is ineligible:" + run_info["state"])
            records, index = [], []
            for source, target in strict.EXPECTED_KEYS:
                key = (source, target)
                item = {"source_index": source, "target_frame": target, "trajectory_id": f"test:{source:06d}"}
                hashes, derived = {}, {}
                if run_issues:
                    record = blank_frame(item, "invalid", "; ".join(run_issues))
                elif key not in original_frames:
                    record = blank_frame(item, "missing")
                else:
                    try:
                        compact = original_frames[key]
                        row_path = checked_path(evaluation / label, compact["record_file"])
                        row, digest = read_json(row_path)
                        require(digest == compact["record_sha256"], "Original record changed after validation")
                        array_path = checked_path(evaluation / label, compact["array_file"])
                        require(sha256(array_path) == compact["array_sha256"], "Original array changed after validation")
                        hashes.update(input_record_sha256=digest, input_array_sha256=compact["array_sha256"])
                        if row["status"] != "complete":
                            record = blank_frame(row)
                        else:
                            prior_index = previous_frames[key]
                            hashes.update(expected_action_record_file=prior_index["record_file"],
                                expected_action_record_sha256=prior_index["record_sha256"])
                            prior_path = checked_path(previous, prior_index["record_file"])
                            # Preserve even invalid observed bytes before parsing or
                            # comparing them to their expected committed identity.
                            prior_hash = sha256(prior_path)
                            snapshots[str(prior_path)] = prior_hash
                            hashes.update(observed_action_record_file=str(prior_path), observed_action_record_sha256=prior_hash)
                            require(prior_hash == prior_index["record_sha256"], "Saved action record differs")
                            prior, parsed_hash = read_json(prior_path)
                            require(parsed_hash == prior_hash, "Saved action record changed during parse")
                            require(prior["input_record_sha256"] == digest and prior["input_array_sha256"] == compact["array_sha256"], "Action/original input identity differs")
                            hashes.update(expected_action_array_file=prior["derived_array_file"],
                                expected_action_array_sha256=prior["derived_array_sha256"])
                            prior_array = checked_path(previous, prior["derived_array_file"])
                            observed_array_hash = sha256(prior_array)
                            snapshots[str(prior_array)] = observed_array_hash
                            hashes.update(observed_action_array_file=str(prior_array), observed_action_array_sha256=observed_array_hash)
                            require(observed_array_hash == prior["derived_array_sha256"], "Saved action array differs")
                            hashes.update(action_record_sha256=prior_hash, action_array_sha256=prior["derived_array_sha256"])
                            with np.load(array_path, allow_pickle=False) as original_arrays, np.load(prior_array, allow_pickle=False) as old_arrays:
                                record, derived = decompose_frame(row, original_arrays, prior, old_arrays)
                    except (ValueError, KeyError, TypeError, OSError, OverflowError, IndexError, AttributeError) as error:
                        record = blank_frame(item, "invalid", f"{type(error).__name__}: {error}")
                        record["failure"] = {"category": "input_or_arithmetic_validation", "error": str(error)}
                        derived = {}
                record.update(hashes)
                stem = f"trajectory_{source:06d}_target_{target:04d}"
                if derived:
                    derived_path = folder / (stem + ".npz")
                    with derived_path.open("xb") as stream:
                        np.savez_compressed(stream, **derived)
                    record.update(derived_array_file=str(derived_path.relative_to(output)), derived_array_sha256=sha256(derived_path))
                row_path = folder / (stem + ".json"); atomic_json(row_path, record)
                index.append({"source_index": source, "target_frame": target, "status": record["status"],
                    "record_file": str(row_path.relative_to(output)), "record_sha256": sha256(row_path)})
                records.append(record)
                if record["status"] != "complete":
                    failures.append({"model": label, "source_index": source, "target_frame": target,
                        "status": record["status"], "failure": record["failure"], "undefined_reasons": sorted(set(record["undefined_reasons"].values()))})
            runs[label] = {key: run_info.get(key) for key in ("objective", "seed", "state", "eligible", "checkpoint_sha256", "protocol_sha256", "result_sha256")}
            runs[label].update(issues=run_issues, frames=index, aggregate=aggregate_frames(records))
        require(action.inventory(evaluation) == before, "Original inventory changed during analysis")
        require(all(Path(path).is_file() and sha256(Path(path)) == digest for path, digest in snapshots.items()), "Inputs changed during analysis")
        require(all(sha256(source_paths[name]) == digest for name, digest in source_hashes.items()), "Analysis source changed during execution")
        atomic_json(output / "all_inputs_sha256.json", snapshots)
        result = {**identity, "state": "complete" if not failures else "incomplete", "required_models": 6,
            "required_frames_per_model": 297, "required_total_frames": 1782, "comparison_order": PAIRS,
            "primary_groups": PRIMARY_GROUPS, "supplementary_both_subgroups": BOTH_GROUPS, "runs": runs,
            "objectives": summarize_objectives(runs), "failures": failures, "consistency_errors": checked["consistency_errors"],
            "inputs_verified_after": len(snapshots), "all_inputs_sha256_file": "all_inputs_sha256.json",
            "all_inputs_sha256": sha256(output / "all_inputs_sha256.json"), "analysis_wall_seconds": time.perf_counter() - started,
            "interpretation": {"units": "normalized acceleration coordinate squared error; vector error/alignment/cost divided by2",
                "group_contribution": "sum(mask * paired quantity)/N; additive over four primary groups; both subgroups partition only both",
                "conditioning": "policy-defined optional exposure, descriptive and not pretreatment causal adjustment",
                "weighted_conditional": "fixed frame/trajectory-weighted contribution divided by fixed-weight fraction, never an available-frame mean",
                "scope": "all previously inspected original no-loop same-state arrays; no new test population, inference, graph permutations or autonomous rollout"}}
        atomic_json(output / "results.json", result)
        (output / "report.md").write_text(render_report(result))
        atomic_json(output / "status.json", {"state": result["state"], "results_sha256": sha256(output / "results.json"),
            "report_sha256": sha256(output / "report.md"), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "failed_or_missing_or_invalid_frames": len(failures), "required_total_frames": 1782})
        return result
    except BaseException as error:
        atomic_json(output / "status.json", {"state": "error", "error_type": type(error).__name__, "error": str(error),
            "generated_utc": datetime.now(timezone.utc).isoformat(), "partial_outputs_preserved": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print({"state": result["state"], "output_dir": str(args.output_dir), "analysis_wall_seconds": result["analysis_wall_seconds"]})


if __name__ == "__main__":
    main()
