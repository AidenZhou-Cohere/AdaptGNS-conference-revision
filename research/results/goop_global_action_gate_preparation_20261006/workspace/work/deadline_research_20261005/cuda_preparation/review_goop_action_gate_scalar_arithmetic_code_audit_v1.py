"""Independent rational-number oracle and fixed-grid missingness probes."""
import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_independent_scalar_arithmetic", HERE / "test_goop_action_gate_scalar_core_v1.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
S, P = T.S, T.P
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(HERE / "goop_action_gate_scalar_core_v1.py") == "a0c0098e72c073b3468a2cc9a0d7a23c723250148f1dc1a550f35e02425081b1"
cases = []
stages = T.synthetic_stages(); exact = {}
for seed in range(3):
    for source in range(30):
        for ordinal, policy in enumerate(S.POLICIES):
            value = Fraction(1000 + (seed + 1) * (source + 3) ** 2 + (ordinal + 1) * (source % 7 - 3) * 8
                             + (seed + 2) * (ordinal + 1), 64)
            exact[seed, source, policy] = value
            stages[seed]["rows"][source * 4 + ordinal]["validated"]["metrics"]["mean_rollout_mse"] = float(value)
summary = S.summarize(stages, P)
means = {(seed, policy): sum((exact[seed, i, policy] for i in range(30)), Fraction()) / 30
         for seed in range(3) for policy in S.POLICIES}
checked = 0


def compare(actual, expected):
    global checked
    assert math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-12), (actual, expected)
    checked += 1


def verify_seed_summary(actual, expected):
    average = sum(expected, Fraction()) / 3
    for seed in range(3): compare(actual["seed_values"][str(seed)], expected[seed])
    compare(actual["mean"], average)
    compare(actual["sample_sd"], math.sqrt(float(sum(((v - average) ** 2 for v in expected), Fraction()) / 2)))


for policy in S.POLICIES:
    verify_seed_summary(summary["absolute"]["mean_rollout_mse"][policy], [means[s, policy] for s in range(3)])
for control in S.REFERENCES:
    verify_seed_summary(summary["primary_full_H395_contrasts"]["learned_global_gate_minus_" + control],
                        [means[s, "learned_global_gate"] - means[s, control] for s in range(3)])
cases.append({"case": "independent_fraction_oracle_all30sources_three_seeds_absolute_and_paired", "passed": True,
              "numeric_comparisons": checked, "sample_sd_denominator": 2})

stages = T.synthetic_stages(); T.mutate_cell(stages, 2, 29, "base", "committed_guard_failed", completed=300)
value = S.summarize(stages, P)
assert value["absolute"]["mean_rollout_mse"]["base"]["mean"] is None
assert value["absolute"]["mse_forecast200"]["base"]["defined_seed_pairs"] == 3
assert value["primary_full_H395_contrasts"]["learned_global_gate_minus_random25"]["mean"] is not None
cases.append({"case": "last_source_failure_nulls_full_H395_without_dropping_source_or_unrelated_control", "passed": True})

stages = T.synthetic_stages(); stages[1]["scientific_verification_passed"] = False
value = S.summarize(stages, P)
assert value["all_required_outcomes_numerically_complete"] and not value["all_required_outcomes_complete"]
assert value["unverified_stage_rows_retained_as_diagnostics"] == 120
for metric in S.METRICS + ("end_to_end_case_wall_seconds",):
    assert all(v["seed_values"]["1"] is None and v["mean"] is None for v in value["absolute"][metric].values())
assert sum(v["consumed_committed_work_totals"]["deployed_forward_attempts"]
           for stage in value["runtime"] for v in stage["per_policy"].values()) == 142200
cases.append({"case": "unverified_model_stage_nulls_all_scientific_metrics_preserves142200_consumed_forwards", "passed": True})

stages = T.synthetic_stages(); stages[1]["rows"][0]["case_wall_seconds"] = None
value = S.summarize(stages, P)
assert value["absolute"]["end_to_end_case_wall_seconds"]["base"]["mean"] is None
assert value["absolute"]["processing_plus_publication_seconds"]["base"]["mean"] is not None
observed = value["runtime"][1]["per_policy"]["base"]
assert observed["case_wall_seconds_observed_total"] == 58 and observed["committed_rows_missing_case_wall_seconds"] == 1
cases.append({"case": "missing_end_to_end_case_time_not_imputed_from_partial_components", "passed": True})

stages = T.synthetic_stages()
for stage in stages:
    for record in stage["rows"]:
        if record["policy"] == "learned_global_gate": record["validated"]["realized_cost"]["requested_expansions"] = 1
value = S.summarize(stages, P)
assert value["always_base_requested_action_collapse"] is False and value["learned_no_effective_expansion"] is True
assert value["constructive_success_automatically_assigned"] is False
cases.append({"case": "requested_expansion_distinguished_from_zero_effective_expansion", "passed": True})

receipt = {"schema": "adaptgns_goop_action_gate_scalar_arithmetic_independent_code_audit_v1", "status": "passed",
           "source_sha256": sha(HERE / "goop_action_gate_scalar_core_v1.py"), "probe_source_sha256": sha(Path(__file__)),
           "synthetic_only": True, "passed": len(cases), "cases": cases}
target = HERE / "goop_action_gate_scalar_arithmetic_independent_code_audit_v1.json"
target.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"path": str(target), "sha256": sha(target), "passed": len(cases)}, indent=2))
