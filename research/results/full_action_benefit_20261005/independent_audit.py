"""Independent full action-benefit audit; imports no research production code."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "outputs/AdaptGNS"
ANALYSIS = WORKSPACE / "work/full-action-benefit-20261005"
POLICIES = ("base", "dense", "random25", "speed25", "previous-observed-base-risk25")
ACTIONS, RISKS = POLICIES[1:], ("current_base_q", "previous_observed_base_q")
UNITS, SIGNS = ("position", "normalized"), ("harmful", "zero", "helpful")
PORTFOLIOS = {"all_five": POLICIES, "budgeted_with_base": ("base", *POLICIES[2:])}
SOURCES, TARGETS = tuple(range(3, 30)), (7, 106, 205, 304, 403, 502, 601, 700, 799, 898, 1000)
CHECKS = 0
MAX_DIFF = 0.


def check(condition, context):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(context)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def same(actual, expected, context):
    global CHECKS, MAX_DIFF
    if actual is None or expected is None:
        check(actual is None and expected is None, context + ": undefined mismatch")
        return
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape and a.dtype.kind in "iufb" and b.dtype.kind in "iufb", context + ": shape/type")
    check(bool(np.isfinite(a).all() and np.isfinite(b).all()), context + ": nonfinite")
    diff = np.abs(a.astype(float) - b.astype(float))
    if diff.size:
        MAX_DIFF = max(MAX_DIFF, float(diff.max()))
    CHECKS += int(a.size)
    if not np.allclose(a, b, rtol=1e-11, atol=1e-15):
        raise AssertionError(context + ": arithmetic differs, max diff=" + str(float(diff.max())))


def ranks(values):
    """Independent average ranks from a stable sorting/group scan."""
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + 1 + j) / 2
        i = j
    return result


def corr(x, y):
    if len(x) < 2:
        return None, "fewer_than_two_particles"
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, "constant_rank_vector"
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1]), None


def required_mean(values):
    return float(np.mean(values)) if values and all(x is not None for x in values) else None


def frame_expected(source, arrays, derived, result, context):
    n = source["n_particles"]
    check(source["status"] == result["status"] == "complete", context + ": complete observed frame")
    check(source["failure"] is None and result["failure"] is None, context + ": failure null")
    for name in ("source_index", "target_frame", "trajectory_id", "n_particles"):
        check(source[name] == result[name], context + ": " + name)
    target = arrays["target_position"]
    check(target.shape == (n, 2) and ahash(target) == source["target_sha256"], context + ": target hash")
    for name, key in (("current_observed_history", "observed_history_sha256"), ("previous_observed_history", "previous_observed_history_sha256")):
        check(arrays[name].shape == (6, n, 2) and ahash(arrays[name]) == source[key], context + ": history hash")
    scores = {key: arrays[key].astype(float) for key in RISKS}
    check(all(q.shape == (n,) and np.isfinite(q).all() and np.all(q > 0) for q in scores.values()), context + ": risk shape/positivity")
    predictions = {key: arrays["prediction_" + key].astype(float) for key in POLICIES}
    values, reasons = {}, {}
    def put(key, value, reason=None):
        values[key] = value
        if value is None:
            reasons[key] = reason
    expected_derived = set()
    for unit in UNITS:
        divisor = np.asarray(source["saved_acceleration_normalization"]["std"]) if unit == "normalized" else np.ones(2)
        errors = {}
        for policy in POLICIES:
            residual = (predictions[policy] - target) / divisor
            errors[policy] = np.einsum("ij,ij->i", residual, residual)
            same(arrays[f"{unit}_vector_se_{policy}"], errors[policy], context + ": source vector error")
            same(derived[f"error_{unit}_{policy}"], errors[policy], context + ": derived vector error")
            expected_derived.add(f"error_{unit}_{policy}")
            put(f"error/{unit}/{policy}/coordinate_mse", float(errors[policy].mean() / 2))
        for left, right in itertools.combinations(POLICIES, 2):
            put(f"paired_error/{unit}/{left}_minus_{right}", float(np.mean(errors[left] - errors[right]) / 2))
        dense_b = errors["base"] - errors["dense"]
        same(arrays["signed_dense_benefit_" + unit], dense_b, context + ": original dense benefit")
        for policy in ACTIONS:
            b = errors["base"] - errors[policy]
            residual = (target - predictions["base"]) / divisor
            delta = (predictions[policy] - predictions["base"]) / divisor
            alignment = np.einsum("ij,ij->i", residual, delta) * 2
            perturbation = np.einsum("ij,ij->i", delta, delta)
            same(b, alignment - perturbation, context + ": signed residual identity")
            for kind, array in (("benefit", b), ("alignment", alignment), ("perturbation_cost", perturbation)):
                name = f"{kind}_{unit}_{policy}"
                expected_derived.add(name)
                same(derived[name], array, context + ": " + name)
            metrics = {"mean_vector_benefit": float(b.mean()), "mean_coordinate_benefit": float(b.mean() / 2),
                       "helpful_fraction": float(np.count_nonzero(b > 0) / n), "harmful_fraction": float(np.count_nonzero(b < 0) / n),
                       "zero_fraction": float(np.count_nonzero(b == 0) / n), "mean_alignment_term": float(alignment.mean()),
                       "mean_perturbation_cost": float(perturbation.mean()), "frame_helpful_indicator": float(b.mean() > 0),
                       "frame_harmful_indicator": float(b.mean() < 0), "frame_zero_indicator": float(b.mean() == 0)}
            for key, value in metrics.items():
                put(f"benefit/{unit}/{policy}/{key}", value)
            dense_masks, actual_masks = (dense_b < 0, dense_b == 0, dense_b > 0), (b < 0, b == 0, b > 0)
            counts = [[int(np.count_nonzero(a & c)) for c in actual_masks] for a in dense_masks]
            check(result["dense_proxy_sign_counts"][unit][policy] == {"rows_dense_columns_actual": list(SIGNS), "counts": counts}, context + ": dense sign count table")
            put(f"dense_proxy/{unit}/{policy}/sign_disagreement_fraction", float((n - np.trace(counts)) / n))
            put(f"dense_proxy/{unit}/{policy}/opposite_nonzero_fraction", float((counts[0][2] + counts[2][0]) / n))
            for i, dense_label in enumerate(SIGNS):
                for j, actual_label in enumerate(SIGNS):
                    put(f"dense_proxy/{unit}/{policy}/dense_{dense_label}_actual_{actual_label}_fraction", float(counts[i][j] / n))
            actual_corr = {}
            for risk, q in scores.items():
                coefficient, reason = corr(q, b)
                dense_coefficient, dense_reason = corr(q, dense_b)
                actual_corr[risk] = coefficient
                put(f"correlation/{unit}/{policy}/{risk}", coefficient, reason)
                put(f"correlation_difference/{unit}/{policy}/{risk}/actual_minus_dense",
                    coefficient - dense_coefficient if coefficient is not None and dense_coefficient is not None else None, reason or dense_reason)
                quartiles = np.searchsorted([.25, .5, .75], (ranks(q) - .5) / n, side="right")
                same(derived["quartile_" + risk], quartiles, context + ": tie-aware bins")
                expected_derived.add("quartile_" + risk)
                check(result["risk_quartile_counts"][risk] == [int(np.count_nonzero(quartiles == k)) for k in range(4)], context + ": risk bin counts")
                for k in range(4):
                    selected = quartiles == k
                    prefix = f"risk_profile/{unit}/{policy}/{risk}/q{k+1}/"
                    put(prefix + "particle_fraction", float(np.count_nonzero(selected) / n))
                    put(prefix + "mean_vector_benefit", float(b[selected].mean()) if selected.any() else None, "empty_risk_quartile")
                    put(prefix + "harmful_fraction", float(np.count_nonzero(b[selected] < 0) / selected.sum()) if selected.any() else None, "empty_risk_quartile")
            current, previous = (actual_corr[key] for key in RISKS)
            put(f"correlation_difference/{unit}/{policy}/current_minus_previous",
                current - previous if current is not None and previous is not None else None, "required_correlation_undefined")
        for portfolio, options in PORTFOLIOS.items():
            mse = {policy: float(errors[policy].mean() / 2) for policy in options}
            best = min(mse.values())
            # Whole-frame reduction precedes policy choice. No per-node mixing.
            ties = [policy for policy in options if mse[policy] == best]
            forced = min(mse[policy] for policy in options if policy != "base")
            oracle = {"coordinate_mse": best, "base_minus_oracle": mse["base"] - best,
                      "previous_risk_minus_oracle": mse[POLICIES[-1]] - best,
                      "best_non_base_coordinate_mse": forced, "benefit_from_allowing_base": forced - best,
                      "base_strictly_best": float(mse["base"] < forced), "base_co_best": float("base" in ties),
                      "best_tie_count": float(len(ties))}
            oracle.update({"choice/" + policy: float(policy == ties[0]) for policy in options})
            for key, value in oracle.items():
                put(f"oracle/{unit}/{portfolio}/{key}", value)
    base = {tuple(pair) for pair in arrays["candidate_base_pairs_float64"].tolist()}
    annulus = {tuple(pair) for pair in arrays["candidate_annulus_pairs_float64"].tolist()}
    check(not base & annulus, context + ": candidate partition")
    selected_optional = {}
    for policy in POLICIES:
        pairs = arrays["pairs_" + policy]
        pair_set = {tuple(pair) for pair in pairs.tolist()}
        check(len(pair_set) == len(pairs) and all(0 <= i < j < n for i, j in pair_set), context + ": simple pairs")
        check(base <= pair_set <= base | annulus, context + ": mandatory preservation")
        selected_optional[policy] = pair_set - base
        count = 0 if policy == "base" else len(annulus) if policy == "dense" else len(annulus) // 4
        check(len(selected_optional[policy]) == count, context + ": exact budget")
        first_hash = next(call["pair_sha256"] for call in source["timed_calls"] if call["method"] == policy)
        check(ahash(pairs) == first_hash, context + ": first timed pair hash")
        if policy == "base":
            continue
        incidence = np.array(sorted(selected_optional[policy]), dtype=int).reshape(-1)
        degree = np.bincount(incidence, minlength=n)
        same(derived["optional_degree_" + policy], degree, context + ": endpoint-incidence degree")
        expected_derived.add("optional_degree_" + policy)
        total = float(degree.sum())
        ordered = np.sort(degree)
        # Sum all unordered pair degree differences via a cumulative sum.
        separations = np.arange(n) * ordered - np.r_[0, np.cumsum(ordered)[:-1]]
        gini = float(separations.sum() / (n * total)) if total else 0.
        stats = {"mean": float(total / n), "maximum": float(degree.max()), "second_moment": float(np.dot(degree, degree) / n),
                 "nonzero_fraction": float(np.count_nonzero(degree) / n), "gini": gini,
                 "herfindahl": float(np.dot(degree, degree) / total ** 2) if total else None,
                 "maximum_share": float(degree.max() / total) if total else None}
        for key, value in stats.items():
            put(f"optional_degree/{policy}/{key}", value, "no_optional_pairs")
        for risk, q in scores.items():
            coefficient, reason = corr(q, degree)
            put(f"degree_correlation/{policy}/{risk}", coefficient, reason)
    for left, right in itertools.combinations(POLICIES, 2):
        a, b = selected_optional[left], selected_optional[right]
        put(f"optional_jaccard/{left}__{right}", len(a & b) / len(a | b) if a | b else 1.)
    coefficient, reason = corr(scores[RISKS[0]], scores[RISKS[1]])
    put("risk_agreement/current_vs_previous_spearman", coefficient, reason)
    check(set(derived.files) == expected_derived, context + ": derived key set")
    check(set(values) == set(result["metrics"]), context + ": metric set")
    check(reasons == result["undefined_reasons"], context + ": exact undefined reasons")
    for key, expected in values.items():
        same(result["metrics"][key], expected, context + ": " + key)
    return values, reasons


def audit():
    started = time.perf_counter()
    state = read(ANALYSIS / "status.json")
    check(state["state"] == "complete", "Analysis must be committed complete before audit")
    analysis_inventory = {str(path.relative_to(ANALYSIS)): sha(path) for path in sorted(ANALYSIS.rglob("*")) if path.is_file()}
    check(sha(ANALYSIS / "results.json") == state["results_sha256"], "Committed result checksum")
    check(sha(ANALYSIS / "report.md") == state["report_sha256"], "Committed report checksum")
    result = read(ANALYSIS / "results.json")
    identity = read(ANALYSIS / "input_identity.json")
    check(sha(ANALYSIS / "input_identity.json") == result["input_identity_sha256"], "Input identity checksum")
    original_root = Path(identity["input_root"])
    check(identity["input_files_sha256"] and len(identity["input_files_sha256"]) == 3582, "Complete original input inventory count")
    for path, digest in identity["input_files_sha256"].items():
        check(sha(original_root / path) == digest, "Original input unchanged before audit: " + path)
    for path, digest in identity["source_sha256"].items():
        check(sha(REPO / path) == digest, "Analysis/frozen source unchanged: " + path)
    check(identity["source_sha256"]["research/analyze_full_action_benefit.py"] == "e8f167646986e08b650e58b4388cd56e5277ec200708eb3d9788eaedf39c34b3", "Reviewed frozen analyzer")
    check(identity["protocol_sha256"] == "f9e779324fc54ad7d8404f5402acfc4e8386e80b5c306db0b425513995d5f0c5", "Reviewed frozen exploratory protocol")
    run_metrics, run_outcomes, records = {}, {}, 0
    for objective in ("faithful", "nll"):
        for seed in (0, 1, 2):
            label = f"{objective}_seed{seed}"
            run = result["runs"][label]
            check(run["objective"] == objective and run["seed"] == seed and run["state"] == "complete" and run["eligible"], label + ": run identity/state")
            protocol = read(original_root / label / "protocol.json")
            check(run["checkpoint_sha256"] == protocol["checkpoint_sha256"], label + ": checkpoint SHA retained")
            check(run["protocol_sha256"] == sha(original_root / label / "protocol.json"), label + ": source protocol retained")
            check(run["result_sha256"] == sha(original_root / label / "result.json"), label + ": source result retained")
            indexed = {(row["source_index"], row["target_frame"]): row for row in run["frames"]}
            check(len(run["frames"]) == 297 and set(indexed) == set(itertools.product(SOURCES, TARGETS)), label + ": exact frame coverage")
            metrics, reasons = {}, {}
            for source, target in itertools.product(SOURCES, TARGETS):
                compact = indexed[source, target]
                path = ANALYSIS / compact["record_file"]
                check(sha(path) == compact["record_sha256"], label + ": derived record hash")
                row = read(path)
                stem = f"trajectory_{source:06d}_target_{target:04d}"
                source_json, source_array = original_root / label / (stem + ".json"), original_root / label / (stem + ".npz")
                check(sha(source_json) == row["input_record_sha256"], label + ": source row SHA")
                check(sha(source_array) == row["input_array_sha256"], label + ": source array SHA")
                check(sha(ANALYSIS / row["derived_array_file"]) == row["derived_array_sha256"], label + ": derived array SHA")
                source_row = read(source_json)
                with np.load(source_array, allow_pickle=False) as original, np.load(ANALYSIS / row["derived_array_file"], allow_pickle=False) as derived:
                    expected, undefined = frame_expected(source_row, original, derived, row, f"{label}/{source}/{target}")
                metrics[source, target], reasons[source, target] = expected, undefined
                records += 1
            names = tuple(expected)
            per_trajectory = {str(source): {name: required_mean([metrics[source, target][name] for target in TARGETS]) for name in names} for source in SOURCES}
            run_metrics[label] = {name: required_mean([per_trajectory[str(source)][name] for source in SOURCES]) for name in names}
            aggregate = run["aggregate"]
            check(aggregate["required_frames"] == aggregate["observed_frames"] == 297, label + ": aggregation frame count")
            check(aggregate["frame_status_counts"] == {"complete": 297}, label + ": frame status counts")
            for source in SOURCES:
                for name in names:
                    same(aggregate["trajectories"][str(source)][name], per_trajectory[str(source)][name], label + ": trajectory " + name)
            for name in names:
                entry = aggregate["metrics"][name]
                same(entry["equal_trajectory_mean"], run_metrics[label][name], label + ": seed " + name)
                check(entry["required_frames"] == 297 and entry["required_trajectories"] == 27, label + ": required counts")
                check(entry["defined_frames"] == sum(values[name] is not None for values in metrics.values()), label + ": defined frame counts")
                check(entry["defined_trajectories"] == sum(per_trajectory[str(source)][name] is not None for source in SOURCES), label + ": defined trajectory counts")
                counter = Counter(reason[name] for reason in reasons.values() if name in reason)
                check(entry["undefined_reason_counts"] == dict(counter), label + ": undefined count reasons")
            run_outcomes[label] = {}
            for unit in UNITS:
                for action in ACTIONS:
                    vector = [per_trajectory[str(source)][f"benefit/{unit}/{action}/mean_vector_benefit"] for source in SOURCES]
                    counts = {"helpful": sum(x is not None and x > 0 for x in vector), "harmful": sum(x is not None and x < 0 for x in vector),
                              "zero": sum(x is not None and x == 0 for x in vector), "undefined": sum(x is None for x in vector)}
                    fractions = {key: counts[key] / 27 if not counts["undefined"] else None for key in ("helpful", "harmful", "zero")}
                    saved = aggregate["trajectory_action_outcomes"][f"{unit}/{action}"]
                    check(saved == {"required_trajectories": 27, "counts": counts, "fractions": fractions}, label + ": trajectory sign frequencies")
                    run_outcomes[label].update({f"{unit}/{action}/{key}": value for key, value in fractions.items()})
    for objective in ("faithful", "nll"):
        for name in names:
            values = [run_metrics[f"{objective}_seed{seed}"][name] for seed in (0, 1, 2)]
            check_summary(result["objectives"][objective][name], values, objective + ": group " + name)
        for name in run_outcomes["faithful_seed0"]:
            values = [run_outcomes[f"{objective}_seed{seed}"][name] for seed in (0, 1, 2)]
            check_summary(result["trajectory_action_outcomes"][objective][name], values, objective + ": trajectory signs")
    for name in names:
        values = []
        for seed in (0, 1, 2):
            nll, faithful = run_metrics[f"nll_seed{seed}"][name], run_metrics[f"faithful_seed{seed}"][name]
            values.append(nll - faithful if nll is not None and faithful is not None else None)
        check_summary(result["paired_nll_minus_faithful"][name], values, "paired objective difference " + name)
    for path, digest in identity["input_files_sha256"].items():
        check(sha(original_root / path) == digest, "Original input unchanged after audit: " + path)
    after = {str(path.relative_to(ANALYSIS)): sha(path) for path in sorted(ANALYSIS.rglob("*")) if path.is_file()}
    check(after == analysis_inventory, "All analysis artifacts unchanged during independent audit")
    check(records == 1782, "All 6x297 frames audited")
    return {"passed": True, "checks": CHECKS, "audited_frames": records, "original_input_files": len(identity["input_files_sha256"]),
            "analysis_files": len(analysis_inventory), "analysis_result_sha256": sha(ANALYSIS / "results.json"),
            "analysis_snapshot_sha256": hashlib.sha256(json.dumps(analysis_inventory, sort_keys=True).encode()).hexdigest(),
            "maximum_comparison_absolute_difference": MAX_DIFF, "source_sha256": sha(__file__),
            "implementation": "Independent NumPy dot/einsum formulas, sort-scan average ranks, corrcoef, sign mask cross-tabs, whole-frame portfolio reductions, bincount endpoint incidence, cumulative pairwise Gini, and independent hierarchy; no production analysis import",
            "checks_scope": "Each scalar/array comparison element is counted; provenance and structure checks counted individually. Audit counts are software verification, not statistical sample size.",
            "wall_seconds": time.perf_counter() - started}


def check_summary(saved, values, context):
    for a, b in zip(saved["seed_values"], values):
        same(a, b, context + ": paired seed value")
    check(len(saved["seed_values"]) == 3 and saved["required_seeds"] == 3, context + ": exactly three seeds")
    check(saved["defined_seeds"] == sum(x is not None for x in values), context + ": defined seeds")
    mean = required_mean(values)
    same(saved["mean"], mean, context + ": mean")
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / 2) if mean is not None else None
    same(saved["sample_seed_sd"], sd, context + ": sample SD")


if __name__ == "__main__":
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(__file__).parent / ("attempt_" + stamp + ".json")
    try:
        report = audit()
    except BaseException as error:
        report = {"passed": False, "checks_before_failure": CHECKS, "error_type": type(error).__name__, "error": str(error),
                  "source_sha256": sha(__file__), "generated_utc": datetime.now(timezone.utc).isoformat()}
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    report["generated_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit_report": str(output), **report}, allow_nan=False))
