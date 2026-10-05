"""Independent scalar arithmetic audit; never invokes an evaluator or summarizer."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import math
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/full-evaluation"
OUT = Path(__file__).with_suffix(".json")
MD = Path(__file__).with_suffix(".md")
if OUT.exists() or MD.exists():
    raise RuntimeError("Refusing to overwrite any previous audit attempt")
OBJECTIVES = ("faithful", "nll")
SEEDS = (0, 1, 2)
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
INDICES = tuple(range(3, 30))
BOUNDARY = tuple(f"{source}_boundary_{metric}" for source in ("predicted", "ground_truth")
                 for metric in ("mean_fraction_outside_gt_1e_minus6", "mean_step_maximum_excursion",
                                "trajectory_maximum_excursion"))
METRICS = ("mean_rollout_mse", "mse_at_200", "mse_at_995", "failure_fraction", "mean_directed_edges",
           "mean_rollout_wall_seconds", "mean_total_network_passes") + BOUNDARY
AUDITS = {
    "faithful_seed0": "evaluation_monitor_20261005T1751_faithful0_audit.json",
    "nll_seed0": "evaluation_monitor_20261005T1821_nll0_audit.json",
    "faithful_seed1": "evaluation_monitor_20261005T1851_faithful1_audit.json",
    "nll_seed1": "evaluation_monitor_20261005T1921_nll1_audit.json",
    "faithful_seed2": "evaluation_monitor_20261005T1951_faithful2_audit.json",
    "nll_seed2": "evaluation_monitor_20261005T2022_nll2_audit.json",
}
report = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
          "scope": "Independent arithmetic from all saved scalar trajectory records, including per-step JSON MSE, edges and boundary diagnostics. No model/checkpoint/data-array access, inference, process action, or production summarizer invocation.",
          "anomalies": []}
hashes = {}
checks = 0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    payload = path.read_bytes()
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        report["anomalies"].append(label)


def compare(actual, expected, label):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), label + ": keys")
        if isinstance(actual, dict):
            for key in set(actual).intersection(expected):
                compare(actual[key], expected[key], label + "/" + str(key))
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), label + ": list length")
        if isinstance(actual, list):
            for index, (a, b) in enumerate(zip(actual, expected)):
                compare(a, b, label + "/" + str(index))
    elif type(expected) in (int, float):
        check(type(actual) in (int, float) and math.isfinite(actual)
              and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
              label + f": {actual!r} differs from independently computed {expected!r}")
    else:
        check(actual == expected and (expected is not None or actual is None),
              label + f": {actual!r} differs from {expected!r}")


def mean(values):
    return math.fsum(values) / len(values)


def full_mean(values):
    return mean(values) if len(values) == 27 and all(v is not None for v in values) else None


def stats(values):
    available = sum(v is not None for v in values)
    avg = mean(values) if available == 3 else None
    sd = math.sqrt(math.fsum((v - avg) ** 2 for v in values) / 2) if available == 3 else None
    return {"seed_values": [{"seed": seed, "value": value} for seed, value in zip(SEEDS, values)],
            "defined_seed_count": available, "mean": avg, "sample_sd": sd}


def boundary(raw):
    values = {}
    for source in ("predicted", "ground_truth"):
        items = raw[source + "_boundary_per_step"]
        check(len(items) == raw["completed_steps"], "boundary length " + raw["record_file"]
              if "record_file" in raw else "boundary length")
        values[source + "_boundary_mean_fraction_outside_gt_1e_minus6"] = (
            mean([v["fraction_particles_outside_by_more_than_1e-6"] for v in items]) if items else None)
        values[source + "_boundary_mean_step_maximum_excursion"] = (
            mean([v["maximum_coordinate_excursion"] for v in items]) if items else None)
        values[source + "_boundary_trajectory_maximum_excursion"] = (
            max(v["maximum_coordinate_excursion"] for v in items) if items else None)
    if raw["status"] == "complete":
        return {"full_horizon": values, "accepted_prefix_if_failed": None}
    return {"full_horizon": {k: None for k in BOUNDARY}, "accepted_prefix_if_failed": {
        "completed_steps": raw["completed_steps"],
        "scope": "accepted steps only; excludes the failed attempt; never used in full-horizon or across-seed means",
        "metrics": values}}


def metric(row, name):
    if name in BOUNDARY:
        return row["boundary_diagnostics"]["full_horizon"][name]
    if name == "mse_at_200":
        return row["mse_at_steps"]["200"]
    if name == "mse_at_995":
        return row["mse_at_steps"]["995"]
    if name == "failure_fraction":
        return float(row["status"] == "failed")
    return row[{"mean_rollout_wall_seconds": "total_wall_seconds",
                "mean_total_network_passes": "total_network_passes"}.get(name, name)]


def seed_policy(rows, seed):
    failed = [r for r in rows if r["status"] == "failed"]
    calculated = {key: full_mean([metric(row, key) for row in rows]) for key in METRICS}
    return {"seed": seed, "run_state": "complete", "required_trajectories": 27,
            "observed_trajectories": len(rows), "completed_trajectories": len(rows) - len(failed),
            "failed_trajectories": len(failed), "missing_source_indices": [],
            "all_sample_full_horizon_error_defined": calculated["mean_rollout_mse"] is not None,
            "all_sample_full_horizon_boundary_defined": all(calculated[k] is not None for k in BOUNDARY),
            "metrics": calculated,
            "failures": [{key: row[key] for key in ("trajectory_id", "source_index", "completed_steps", "failure")}
                         for row in failed],
            "boundary_prefix_diagnostics_if_failed": [
                {"trajectory_id": row["trajectory_id"], "source_index": row["source_index"],
                 **row["boundary_diagnostics"]["accepted_prefix_if_failed"]} for row in failed]}


def paired(left_objective, right_objective, left_policy, right_policy, rows_by_key):
    per_seed = []
    for seed in SEEDS:
        pairs = []
        for index in INDICES:
            left = rows_by_key[left_objective, seed, left_policy, index]
            right = rows_by_key[right_objective, seed, right_policy, index]
            check(left["trajectory_id"] == right["trajectory_id"], "paired trajectory identity")
            differences = {}
            for key in METRICS:
                a, b = metric(left, key), metric(right, key)
                differences[key] = a - b if a is not None and b is not None else None
            pairs.append({"source_index": index, "trajectory_id": left["trajectory_id"], "deltas": differences})
        per_seed.append({"seed": seed, "eligible": True, "paired_trajectories": 27,
                         "metrics": {k: full_mean([p["deltas"][k] for p in pairs]) for k in METRICS},
                         "trajectory_pairs": pairs})
    return {"direction": "left minus right; negative error/failure differences favor left",
            "left_policy": left_policy, "right_policy": right_policy, "per_seed": per_seed,
            "metrics": {k: stats([row["metrics"][k] for row in per_seed]) for k in METRICS}}


resource_maxima = {}
rejected_attempts = []


def retain_maxima(scope, item, location):
    values = {key: item.get(key) for key in ("candidate_pairs", "retained_optional_pairs",
                                            "available_annulus_pairs", "directed_edges")}
    values["retained_pairs"] = item["directed_edges"] // 2 if "directed_edges" in item else item.get("retained_pairs")
    values["maximum_coordinate_excursion"] = item.get("predicted_boundary", {}).get("maximum_coordinate_excursion")
    for key, value in values.items():
        if value is None:
            continue
        for group in (scope, f"{scope}:{location['objective']}:{location['policy']}"):
            previous = resource_maxima.setdefault(group, {}).get(key)
            if previous is None or value > previous["maximum"]:
                resource_maxima[group][key] = {"maximum": value, "location": location}


try:
    hashes[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    summary = read(BASE / "reports/full_rollouts.json")
    compare({k: summary[k] for k in ("state", "complete_models", "aggregation_eligible_models",
                                    "required_policy_trajectories", "consistency_errors")},
            {"state": "complete", "complete_models": 6, "aggregation_eligible_models": 6,
             "required_policy_trajectories": 810, "consistency_errors": []}, "summary completion")
    compare(summary["required_source_indices"], list(INDICES), "required indices")
    compare(summary["required_policies"], list(POLICIES), "required policies")
    rows_by_key, failures, totals, complete_run_hashes = {}, [], [], {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            name = f"{objective}_seed{seed}"
            directory = BASE / "rollout" / name
            prior = read(ROOT / "work" / AUDITS[name])
            check(prior["passed"] and not prior["anomalies"], "previous strict audit: " + name)
            saved = read(directory / "result.json")
            status = read(directory / "status.json")
            protocol = read(directory / "protocol.json")
            result_hash = hashes[str((directory / "result.json").relative_to(ROOT))]
            complete_run_hashes[name] = result_hash
            check(result_hash == prior["result_sha256"] == status["result_sha256"], "stable result: " + name)
            check(protocol["checkpoint_sha256"] == prior["checkpoint_metadata_sha256"], "checkpoint identity: " + name)
            check(len(saved["records"]) == 135, "record count: " + name)
            run = next(r for r in summary["runs"] if (r["objective"], r["seed"]) == (objective, seed))
            check(run["result_sha256"] == result_hash and run["eligible_for_aggregation"], "summary run identity: " + name)
            check(len(run["records"]) == 135, "summary record count: " + name)
            archived = {(r["source_index"], r["policy"]): r for r in run["records"]}
            for row in saved["records"]:
                raw = read(directory / row["record_file"])
                raw_hash = hashes[str((directory / row["record_file"]).relative_to(ROOT))]
                check(raw_hash == row["record_sha256"] == prior["input_sha256"][
                    str((directory / row["record_file"]).relative_to(ROOT))], "stable raw record " + name + "/" + row["record_file"])
                compare({key: raw[key] for key in row if key not in ("record_file", "record_sha256")},
                        {key: row[key] for key in row if key not in ("record_file", "record_sha256")},
                        "compact/raw " + name + "/" + row["record_file"])
                complete = row["status"] == "complete"
                check((complete and row["completed_steps"] == 995) or (not complete and row["completed_steps"] < 995),
                      "completion length " + name + "/" + row["record_file"])
                compare(row["mean_rollout_mse"], mean(raw["mse_per_step"]) if complete else None, "independent MSE")
                compare(row["mean_directed_edges"], mean(raw["directed_edges_per_step"]) if complete else None, "independent edges")
                compare(row["mse_at_final_horizon"], raw["mse_per_step"][-1] if complete else None, "final MSE")
                location = {"objective": objective, "seed": seed, "policy": row["policy"],
                            "source_index": row["source_index"], "trajectory_id": row["trajectory_id"]}
                for attempt in raw["attempts"]:
                    scope = "accepted_forecast" if attempt["accepted"] else "rejected_forecast"
                    retain_maxima(scope, attempt, {**location, "forecast_step": attempt["forecast_step"]})
                    if not attempt["accepted"]:
                        rejected_attempts.append({**location, "attempt": attempt})
                if raw["warmup"].get("passes"):
                    retain_maxima("initial_base_scoring_warmup", raw["warmup"], {**location, "forecast_step": 0})
                row = {**row, "boundary_diagnostics": boundary(raw)}
                compare(archived[row["source_index"], row["policy"]], row, "summary trajectory " + name + "/" + row["record_file"])
                key = (objective, seed, row["policy"], row["source_index"])
                check(key not in rows_by_key, "unique trajectory/policy")
                rows_by_key[key] = row
                if not complete:
                    failures.append({"objective": objective, "seed": seed, **row})
            compare(saved["total_recorded_rollout_wall_seconds"], mean([r["total_wall_seconds"] for r in saved["records"]]) * 135,
                    "recorded wall sum " + name)
            totals.append({"name": name, "invocation_wall_seconds": saved["invocation_wall_seconds"],
                           "recorded_rollout_wall_seconds": saved["total_recorded_rollout_wall_seconds"],
                           "completed_steps_including_failed_prefixes": sum(r["completed_steps"] for r in saved["records"]),
                           "total_network_passes": sum(r["total_network_passes"] for r in saved["records"])})
    check(len(rows_by_key) == 810, "all 810 distinct outcomes")
    policies = {}
    for objective in OBJECTIVES:
        policies[objective] = {}
        for policy in POLICIES:
            per_seed = [seed_policy([rows_by_key[objective, seed, policy, i] for i in INDICES], seed) for seed in SEEDS]
            policies[objective][policy] = {"per_seed": per_seed,
                "metrics": {k: stats([r["metrics"][k] for r in per_seed]) for k in METRICS}}
    compare(summary["policies"], policies, "all policy aggregations")
    comparisons = {}
    pairs = [(p, "base") for p in POLICIES[1:]] + [("laggedrisk25", p) for p in ("dense", "random25", "speed25")]
    for objective in OBJECTIVES:
        for left, right in pairs:
            comparisons[f"{objective}:{left}-minus-{right}"] = paired(objective, objective, left, right, rows_by_key)
    for policy in POLICIES:
        comparisons[f"nll-minus-faithful:{policy}"] = paired("nll", "faithful", policy, policy, rows_by_key)
    compare(summary["paired_comparisons"], comparisons, "all paired trajectory/seed/group differences")
    faithful_risk = {}
    for reference in ("base", "random25", "speed25", "dense"):
        comparison = comparisons[f"faithful:laggedrisk25-minus-{reference}"]
        result = {}
        for key in ("mean_rollout_mse", "mse_at_200", "mse_at_995"):
            differences = [r["metrics"][key] for r in comparison["per_seed"]]
            denominators = [r["metrics"][key] for r in policies["faithful"][reference]["per_seed"]]
            percentages = [100 * d / b for d, b in zip(differences, denominators)]
            result[key] = {"absolute_difference": stats(differences), "within_seed_percentage_change": stats(percentages),
                           "seed_signs": ["lower" if d < 0 else "higher" if d > 0 else "equal" for d in differences],
                           "percentage_definition": "100 * (laggedrisk seed mean - reference seed mean) / reference seed mean; then mean and sample SD across three seeds; not a ratio of group means"}
        result["boundary_absolute_differences"] = {k: comparison["metrics"][k] for k in BOUNDARY}
        faithful_risk[reference] = result
    boundary_table = {obj: {pol: {k: policies[obj][pol]["metrics"][k] for k in BOUNDARY}
                            for pol in POLICIES} for obj in OBJECTIVES}
    # Complete trajectories can have large excursions without triggering the computational guard.
    completed_worst = {}
    for objective in OBJECTIVES:
        for policy in POLICIES:
            records = [r for (obj, _, pol, _), r in rows_by_key.items()
                       if obj == objective and pol == policy and r["status"] == "complete"]
            maximum = max(r["boundary_diagnostics"]["full_horizon"]["predicted_boundary_trajectory_maximum_excursion"] for r in records)
            completed_worst[f"{objective}:{policy}"] = {"maximum": maximum,
                "scope": "Descriptive worst completed trajectory only; not an all-sample boundary aggregate or physical-validity claim"}
    report.update(
        strict_run_result_sha256=complete_run_hashes,
        outcomes=810, completed_outcomes=810 - len(failures), failed_outcomes=len(failures),
        failures=failures,
        failure_counts_by_objective_seed_policy=dict(Counter(f"{r['objective']}_seed{r['seed']}:{r['policy']}" for r in failures)),
        failure_categories=dict(Counter(r["failure"]["category"] for r in failures)),
        verified_policy_aggregates=policies,
        verified_paired_comparison_metrics={name: data["metrics"] for name, data in comparisons.items()},
        faithful_laggedrisk_comparisons=faithful_risk,
        boundary_aggregates=boundary_table,
        descriptive_worst_completed_trajectory_excursion=completed_worst,
        resource_and_excursion_maxima={
            "scopes": resource_maxima,
            "retained_pairs_definition": "One half of symmetric directed edges; optional retained pairs are recorded separately.",
            "candidate_definition": "All radius-factor 1.267 undirected pairs before policy selection, including mandatory base pairs.",
            "interpretation": "Descriptive maxima over recorded operations; accepted forecasts, rejected forecasts and the initial base scoring warmup are distinct. Different policies evolve different states.",
            "rejected_attempts": rejected_attempts,
            "peak_memory": None,
            "peak_memory_status": "Not recorded; no inference from candidate or edge counts."},
        autonomous_timing={"runs": totals, "sum_invocation_wall_seconds": math.fsum(r["invocation_wall_seconds"] for r in totals),
                           "sum_recorded_rollout_wall_seconds": math.fsum(r["recorded_rollout_wall_seconds"] for r in totals),
                           "policy_statistics": {obj: {pol: policies[obj][pol]["metrics"]["mean_rollout_wall_seconds"]
                                                        for pol in POLICIES} for obj in OBJECTIVES},
                           "interpretation": "Measured autonomous rollout cost includes failure prefixes and lagged-risk initialization. Fixed policy order, evolving policy-dependent geometry/edge counts, differing failed lengths and warmup scope prevent a controlled speedup conclusion. Use separately measured same-state repeated timings for placement/runtime claims."},
        limits=["Exactly three training seeds; sample SD is not a confidence interval or a significance result.",
                "First three official test trajectories and historical aggregate results were inspected previously; no pristine independent confirmation claim.",
                "NLL seed2 failures make all-sample full-horizon error, edges and boundary means undefined for base/dense/random25/laggedrisk25, including their three-seed aggregates.",
                "All failures occurred after step 200; MSE@200 remains defined over all 27 trajectories per seed.",
                "Boundary diagnostics are geometric, and the coordinate guard is a computational limit rather than a physical-validity test."])
    for name, expected in list(hashes.items()):
        check(digest(ROOT / name) == expected, "input changed while auditing: " + name)
    report["passed"] = not report["anomalies"]
except Exception as error:
    report["anomalies"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
report.update(checks_total=checks, input_sha256=hashes, completed_utc=datetime.now(timezone.utc).isoformat())
with OUT.open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")


def show(value, digits=7):
    return "undefined" if value["mean"] is None else f"{value['mean']:.{digits}g} ± {value['sample_sd']:.{digits}g}"


lines = ["# Independent full-rollout aggregation audit", "",
         f"Passed: **{report['passed']}**. Checks: {checks:,}; input files: {len(hashes)}.", "",
         "Every saved compact record, raw scalar JSON record and aggregate remains preserved. No evaluator, production summarizer, model or dataset array was loaded.", ""]
if report.get("faithful_laggedrisk_comparisons"):
    lines += ["All six jobs cover exactly 27 official trajectories (indices 3–29), five policies and 995 planned steps: 810 outcomes, 802 complete and eight failed. All eight failures belong to NLL seed2 and exceed the absolute-coordinate computational guard of 10. Their full-horizon errors, edges and boundary means remain undefined. The other five models complete all 135 outcomes each.", "",
              "The faithful objective provides no consistent rollout benefit from lagged-risk allocation. Negative paired changes favor lagged risk. The primary error comparison is:", "",
              "| Reference | Lagged risk minus reference MSE | Within-seed percentage change | Signs at seeds 0,1,2 |",
              "|---|---:|---:|---|"]
    for ref, values in faithful_risk.items():
        v = values["mean_rollout_mse"]
        lines.append(f"| {ref} | {show(v['absolute_difference'])} | {show(v['within_seed_percentage_change'])}% | {', '.join(v['seed_signs'])} |")
    lines += ["", "Percentages are computed from each seed's equal-trajectory mean error, then averaged across the three seed pairs; they are not ratios of group means. Errors and percentages use sample SD across three training seeds, with no significance claim. All secondary-horizon comparisons are retained in the JSON.", "",
              "| Objective | Policy | Mean rollout MSE | MSE@200 | MSE@995 |",
              "|---|---|---:|---:|---:|"]
    for obj in OBJECTIVES:
        for pol in POLICIES:
            v = policies[obj][pol]["metrics"]
            lines.append(f"| {obj} | {pol} | {show(v['mean_rollout_mse'])} | {show(v['mse_at_200'])} | {show(v['mse_at_995'])} |")
    lines += ["", "NLL full-horizon policy comparisons involving any failed policy remain undefined. Speed25 alone has all 81 NLL trajectories complete. All failures occur after step 200, so the separately reported 200-step metric uses all required trajectories.", "",
              "| Failed model | Source index | Policy | Accepted steps | Rejected step | Maximum absolute coordinate |",
              "|---|---:|---|---:|---:|---:|"]
    for row in failures:
        lines.append(f"| {row['objective']} seed{row['seed']} | {row['source_index']} | {row['policy']} | {row['completed_steps']} | {row['failure']['forecast_step']} | {row['failure']['maximum_absolute_coordinate']:.12g} |")
    lines += ["", "Boundary diagnostics also give mixed evidence. The faithful lagged-risk outside fraction can be smaller while its excursion magnitudes are larger; completing 995 numerical steps does not establish physical plausibility. The reference truth itself has nonzero boundary excursions. Fractions use a 1e-6 coordinate threshold; each trajectory averages its steps equally, and each seed averages its 27 trajectories equally.", "",
              "| Objective | Policy | Predicted outside fraction | Truth outside fraction | Predicted mean step maximum excursion | Truth mean step maximum excursion |",
              "|---|---|---:|---:|---:|---:|"]
    for obj in OBJECTIVES:
        for pol in POLICIES:
            values = boundary_table[obj][pol]
            keys = ("predicted_boundary_mean_fraction_outside_gt_1e_minus6", "ground_truth_boundary_mean_fraction_outside_gt_1e_minus6",
                    "predicted_boundary_mean_step_maximum_excursion", "ground_truth_boundary_mean_step_maximum_excursion")
            lines.append(f"| {obj} | {pol} | " + " | ".join(show(values[k]) for k in keys) + " |")
    lines += ["", "Full-horizon boundary comparisons and the distinct accepted prefixes of all eight failures are retained in the JSON. No survivors-only mean is substituted.", "",
              "| Faithful reference | Lagged-risk minus reference outside fraction | Mean step maximum excursion difference | Mean trajectory maximum excursion difference |",
              "|---|---:|---:|---:|"]
    for ref, values in faithful_risk.items():
        v = values["boundary_absolute_differences"]
        lines.append(f"| {ref} | " + " | ".join(show(v[k]) for k in (
            "predicted_boundary_mean_fraction_outside_gt_1e_minus6", "predicted_boundary_mean_step_maximum_excursion",
            "predicted_boundary_trajectory_maximum_excursion")) + " |")
    timing = report["autonomous_timing"]
    lines += ["", f"Recorded per-rollout wall counters sum to {timing['sum_recorded_rollout_wall_seconds']:.6f} s; six invocation counters sum to {timing['sum_invocation_wall_seconds']:.6f} s. These are different timing scopes.", "",
              timing["interpretation"], "",
              "Candidate/retained counts and worst excursions are recorded separately for accepted forecasts, rejected forecasts and initial base-scoring passes in the JSON. A retained pair is one undirected pair (two directed edges). Peak memory was not recorded and is not inferred from graph size.", "",
              "This is a prospective full-architecture extension after inspection of historical aggregates and three official test trajectories. It is not pristine independent confirmation. The evidence does not establish conference readiness, a general allocation advantage, or a speedup."]
if report["anomalies"]:
    lines += ["", "Audit anomalies (retained):", "", "```json", json.dumps(report["anomalies"], indent=2), "```"]
with MD.open("x") as stream:
    stream.write("\n".join(lines) + "\n")
print(json.dumps({"passed": report["passed"], "checks_total": checks, "input_files": len(hashes),
                  "failed_outcomes": report.get("failed_outcomes"), "output": str(OUT),
                  "anomalies": report["anomalies"]}, indent=2))
