"""Extract reviewed scalar evidence and write separate paper candidates; no scientific rerun."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = ROOT / "work/deadline_research_20261005"
OUT = TOP / "waterdrop110k_integration_candidate_code_audit_v1"
WD = ROOT / "work/continuation-summary-20261006-v1/result.json"
AUDIT = ROOT / "work/continuation-summary-audit-20261006-v1/audit.json"
GP = ROOT / "outputs/AdaptGNS/research/results/goop2d_graph_exposure_100k_20261006"
GOOP = GP / "paired_scalar_summary.json.gz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def save(name, value):
    path = OUT / name
    with path.open("x") as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")


def read_key(obj, keys):
    for k in keys:
        obj = obj[k]
    return obj


def fmt(value, precision=6):
    return "null" if value is None else f"{value:.{precision}g}"


def texstat(stat, scale=1, precision=3, signed=False):
    if stat["mean"] is None:
        return "---"
    sign = "+" if signed else ""
    return f"${stat['mean'] * scale:{sign}.{precision}f}\\pm{stat['sample_sd'] * scale:.{precision}f}$"


def seeds(stat):
    s = stat["seed_values"]
    return [s[str(i)] for i in range(3)] if isinstance(s, dict) else s


assert sha(WD) == "082bdc9c33614fea6445e3373e94eeafcfdfc516d4a27d04096f380ad7781e64"
assert sha(AUDIT) == "caebc314c5188b7bed6776bb1adc1b060c602fc1d77bd16f93c8960302e3cbbe"
assert sha(GOOP) == "bb3c3f54cdeaf2b1198f49b8437dc711d80ce5e8347ab70363558ab515852978"
manifest = json.loads((GP / "manifest.json").read_text())
assert sha(GOOP) == manifest[GOOP.name]["sha256"]
assert GOOP.stat().st_size == manifest[GOOP.name]["bytes"]
w = json.loads(WD.read_text())
a = json.loads(AUDIT.read_text())
g = json.load(gzip.open(GOOP, "rt"))
assert w["state"] == "complete" and w["source_and_input_reverified_after_analysis"]
assert a["passed"] is True and a["checks"] == 12101908
assert w["coverage"] == {"endpoints": 6, "jobs": 12, "observed_frames": 2550,
                         "observed_policy_slots": 12750, "autonomous_outcomes": 810}
assert all(j["failed_records"] == 0 for j in w["jobs"])
OUT.mkdir(exist_ok=False)

source_files = [WD, AUDIT, GOOP, GP / "manifest.json",
    ROOT / "work/conference_experiments_main.tex", ROOT / "work/manuscript_body.tex",
    ROOT / "outputs/revised_manuscript.tex", ROOT / "outputs/abstract_revision_for_author_review.md",
    ROOT / "outputs/AdaptGNS/research/protocols/faithful_graph_support_10k_20261005.md",
    ROOT / "outputs/AdaptGNS/research/protocols/continuation_summary_20261006.md",
    TOP / "waterdrop110k_interpretation_review_saved_array_v1.md"]
snapshot = {rel(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in source_files}
save("source_snapshot.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "files": snapshot})

# Every WaterDrop statistic is preserved at its exact source location, including nulls.
rows = []
details = {}
for pop, pv in w["populations"].items():
    details[pop] = {}
    for metric, mv in pv["metrics"].items():
        details[pop][metric] = {k: mv[k] for k in ("path", "sha256", "uncompressed_json_sha256")}
        for arm, policies in mv["absolute"].items():
            for policy, stat in policies.items():
                rows.append({"population": pop, "metric": metric, "kind": "absolute",
                    "arm": arm, "policy": policy,
                    "source_key": ["populations", pop, "metrics", metric, "absolute", arm, policy],
                    "statistics": stat})
        for contrast, stat in mv["paired"].items():
            rows.append({"population": pop, "metric": metric, "kind": "paired", "contrast": contrast,
                "source_key": ["populations", pop, "metrics", metric, "paired", contrast],
                "statistics": stat})
assert len(rows) == 2002 and sum(len(v) for v in details.values()) == 91
assert all(read_key(w, r["source_key"]) == r["statistics"] for r in rows)
save("waterdrop_all_scalar_mappings.json", {"source_path": rel(WD), "source_sha256": sha(WD),
    "required_seed_order": [0, 1, 2], "metric_families": 91, "statistics_count": len(rows),
    "metric_details": details,
    "null_scope": "Nulls and defined/required seed counts are preserved. Unit and trajectory null reasons remain in the referenced immutable metric detail files. No survivor averaging or raw prediction reconstruction is performed.",
    "rows": rows})

claims = {}
def claim(name, source, obj, key, description):
    value = read_key(obj, key)
    claims[name] = {"description": description, "source_path": rel(source),
                   "source_sha256": sha(source), "key": key, "value": value}
    return value

def wc(name, pop, metric, kind, *keys, description=""):
    return claim(name, WD, w, ["populations", pop, "metrics", metric, kind, *keys], description)

def gc(name, key, description=""):
    return claim(name, GOOP, g, key, description)

for pop in ("observed_valid", "observed_test"):
    for metric in ("position_coordinate_mse", "normalized_coordinate_mse"):
        for contrast in ("mix_minus_base__base", "mix_minus_base__random25", "base__risk_minus_random",
                         "mix__risk_minus_random", "risk_minus_random_interaction"):
            wc(f"waterdrop_{pop}_{metric}_{contrast}", pop, metric, "paired", contrast)
for contrast in w["populations"]["autonomous_test"]["metrics"]["mean_rollout_mse"]["paired"]:
    wc("waterdrop_h995_" + contrast, "autonomous_test", "mean_rollout_mse", "paired", contrast)
for arm in ("base", "mix"):
    for p in w["populations"]["autonomous_test"]["policies"]:
        for m in ("mean_rollout_mse", "failure_fraction", "predicted_boundary_mean_fraction_outside_gt_1e_minus6",
                  "ground_truth_boundary_mean_fraction_outside_gt_1e_minus6", "predicted_boundary_trajectory_maximum_excursion",
                  "ground_truth_boundary_trajectory_maximum_excursion", "mean_rollout_wall_seconds", "mean_directed_edges"):
            wc(f"waterdrop_h995_{arm}_{p}_{m}", "autonomous_test", m, "absolute", arm, p)
for arm in ("base", "mix"):
    gc("goop_observed_risk_minus_random_" + arm,
       ["diagnostics", "same_state_test", "previous_observed_risk_minus_random_position_mse", arm])
gc("goop_observed_interaction", ["diagnostics", "same_state_test", "risk_minus_random_mix_minus_base_interaction"])
for policy in ("base", "random25"):
    gc("goop_observed_mix_minus_base_" + policy,
       ["diagnostics", "same_state_test", "mix_minus_base", f"accuracy/{policy}/position_coordinate_mse"])
for arm in ("base", "mix"):
    for policy in ("random25", "speed25", "laggedrisk25", "relative-velocity-RMS25"):
        gc(f"goop_h395_{arm}_{policy}", ["full_rollout", "absolute", "mean_rollout_mse", arm, policy])
    gc("goop_h395_risk_minus_random_" + arm,
       ["full_rollout", "within_arm_policy_contrasts", "mean_rollout_mse", arm, "laggedrisk25_minus_random25"])
gc("goop_h395_exposure_random25", ["full_rollout", "mix_minus_base", "mean_rollout_mse", "random25"])
gc("goop_h395_interaction", ["full_rollout", "risk_minus_random_mix_minus_base_interaction", "mean_rollout_mse"])
gc("goop_full_horizon_coverage", ["full_rollout", "coverage"])
gc("goop_full_horizon_coverage_by_model", ["full_rollout", "coverage_by_model"])
claim("waterdrop_coverage", WD, w, ["coverage"], "Six endpoints; twelve jobs; all planned observed and autonomous units.")
claim("waterdrop_job_failure_and_invocation_records", WD, w, ["jobs"], "Zero failed scientific records; invocation timing scope and nulls retained.")
save("claim_source_map.json", claims)

main = r"""A separate WaterDrop follow-up pairs 10k base-only and mixed-graph continuations from three faithful 100k parents, inheriting optimizer state and matched frame/noise schedules. Its exposure effect is conditional on those parents, distinct from Goop training from initialization. Across five evaluation policies, mixed training changes the observed-test risk-minus-random position-MSE gap from $(+1.62\pm1.07)\times10^{-10}$ to $(-0.99\pm0.67)\times10^{-10}$, with every seed changing sign (Table~\ref{tab:goop-exposure-placement}). Validation's relative gap also improves, but mixed risk remains worse than random on average. All 810 H995 autonomous outcomes complete. Under random25, mix-minus-base MSE is $-.0060\pm.0056$, improving in every seed; dense, speed and cached risk also improve in each seed. Yet the autonomous risk-minus-random interaction is $+.00284\pm.00466$, with mixed seed signs. Exposure can therefore improve observed-state placement without a consistent autonomous placement gain. Substantial box excursions and increased recorded rollout times remain (Appendix~\ref{sec:waterdrop-110k-exposure}).
"""
save("candidate_main_waterdrop.tex", main)

table = [r"\begin{table}[t]", r"\centering\small",
    r"\caption{Observed-test position-MSE contrasts, $\times10^{-10}$, within each dataset. Goop trains from initialization to 100k; WaterDrop continues three 100k parents to 110k. Values are means and sample SDs of three paired seeds; negative values favor the first term. The interaction changes the risk-minus-random gap after exposure. These scales do not compare effect size across materials.}",
    r"\label{tab:goop-exposure-placement}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrr}", r"\toprule",
    r"Contrast & Goop & WaterDrop\\", r"\midrule"]
for label, gkey, wkey in (
    ("Mix $-$ base, base policy", "goop_observed_mix_minus_base_base", "mix_minus_base__base"),
    ("Mix $-$ base, random25", "goop_observed_mix_minus_base_random25", "mix_minus_base__random25"),
    ("Risk $-$ random, base train.", "goop_observed_risk_minus_random_base", "base__risk_minus_random"),
    ("Risk $-$ random, mixed train.", "goop_observed_risk_minus_random_mix", "mix__risk_minus_random"),
    ("Training interaction", "goop_observed_interaction", "risk_minus_random_interaction")):
    ws = w["populations"]["observed_test"]["metrics"]["position_coordinate_mse"]["paired"][wkey]
    table.append(f"{label} & {texstat(claims[gkey]['value'], 1e10, 2, True)} & {texstat(ws, 1e10, 2, True)}\\\\")
table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
save("candidate_observed_contrast_table.tex", "\n".join(table))

compression = r"""The earlier WaterDrop controls isolate residual ranking from action value: at 100k updates, faithful/NLL risk correlates with base error ($.357\pm.060$ / $.411\pm.023$), but weakly with actual sparse benefit ($-.021\pm.049$ / $-.052\pm.013$); random has lower one-step error in every seed. Restoring trained self-messages reduces error without reversing that validation/test ordering. Risk's larger alignment gain is outweighed by its squared prediction change in every full-model seed. Autonomous signs vary, with failures and box excursions under both graph conventions (Appendices~\ref{sec:compact-study}, \ref{sec:actual-action-benefit}, \ref{sec:graph-bridge}, \ref{sec:optional-exposure}, \ref{sec:full-results} and~\ref{sec:native-rollout-followup}). These findings motivate testing exposure separately from placement.
"""
save("candidate_compressed_existing_waterdrop.tex", compression)

report = ["# WaterDrop 110k: complete scalar evidence and interpretation", "",
    "This report extracts completed saved scalar summaries; it does not rerun scientific analysis. The standalone machine-readable companion `waterdrop_all_scalar_mappings.json` preserves all 91 metric families and 2,002 absolute/paired statistic objects, with exact JSON keys, three seed values, sample SDs, required/defined counts, and metric-detail hashes. Values below use Python round-trip decimal representations.", "",
    "## Design and scope", "",
    "Six faithful endpoints continue three fixed 100k parents for 10k updates, one base-only and one mixed-graph arm per seed. Both inherit Adam moments/counters and matched frame/noise schedules; LR is 1e-5. Mix independently expands each example with probability 0.5, selecting exactly floor(0.25 × optional-annulus pairs) uniformly and appending both directions after the native capped/self base prefix. This estimates continuation effects conditional on those parents. It is exploratory after inspected tests; it is not fresh-training, NLL, or independent-confirmation evidence.", "",
    "Observed validation has 128 histories per endpoint, observed test 297; all 2,550 histories and 12,750 policy slots are retained. Autonomous test includes sources 3–29, five policies and all 995 forecasts: 810 required outcomes. All scientific failure fractions are zero, all outcomes complete. Training arm `base` differs from inference policy `base`. Observed risk is `previous-observed-base-risk25`; autonomous risk is `laggedrisk25`.", "",
    "Coordinates/particles are averaged within frame, then frames equally within trajectory, trajectories equally within seed. Seed order is 0,1,2; all three enter mean and sample SD (ddof=1). Differences form at matched frames/trajectories first. Required undefined values propagate to null; no survivor average. Base-action gain is exactly zero, and its gain rank correlation is undefined. Full null reasons remain in the immutable metric detail files referenced in the machine companion.", "",
    "## Scientific comparison with Goop", "",
    "Goop trains six faithful models from initialization for 100k updates, evaluates six policies on all 30 test sources through H395, and retains 1,077 completed plus three guard-failed outcomes. Its mixed risk H395 mean and risk-minus-random interaction remain null. WaterDrop is a five-policy 100k-parent +10k continuation through H995. Their absolute scales and training interventions differ.", "",
    "In both studies graph exposure improves autonomous error at fixed random25 policy in every paired seed. Goop observed risk stays worse than random in every mixed-model seed despite a smaller gap. WaterDrop observed-test risk reverses from worse to better in all three seeds; on validation its relative gap improves but the mixed risk-minus-random mean remains positive. WaterDrop's autonomous risk advantage changes sign across seeds, and its mean training interaction is positive, so the observed-state benefit does not establish a consistent autonomous placement gain.", "",
    "Dense, random, speed and risk WaterDrop policies all improve H995 MSE under mix in every seed. Base inference has one unfavorable seed. Dense remains worse than base in every seed of both arms; speed has the lowest across-seed H995 MSE in both arms. These are descriptive paired findings with three parents, not significance tests.", "",
    "All outcomes complete but geometric deviations remain: mixed random/speed/risk outside fractions exceed 16%, against 2.68% in matching truth. All five recorded autonomous policy times and edge counts increase under mix. These fixed-order timings follow different trajectories and shared computation, so neither time ratios nor edge counts estimate a causal speedup. Metadata-box excursions do not test conservation or certify physical validity.", ""]

def mdmetric(pop, metric, paired=True):
    mv = w["populations"][pop]["metrics"][metric]
    report.extend([f"### {pop}: `{metric}`", "", f"Source key: `populations.{pop}.metrics.{metric}` in `{rel(WD)}`.", "",
        "| Arm / policy | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | Defined / required |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for arm, ps in mv["absolute"].items():
        for p, st in ps.items():
            vals = ["null" if v is None else repr(v) for v in (*seeds(st), st["mean"], st["sample_sd"])]
            report.append(f"| {arm} / {p} | " + " | ".join(vals) + f" | {st['defined_seeds']}/{st['required_seeds']} |")
    report.append("")
    if paired:
        report.extend(["| Paired contrast | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | Defined / required |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for c, st in mv["paired"].items():
            vals = ["null" if v is None else repr(v) for v in (*seeds(st), st["mean"], st["sample_sd"])]
            report.append(f"| {c} | " + " | ".join(vals) + f" | {st['defined_seeds']}/{st['required_seeds']} |")
        report.append("")

report.extend(["## Every policy and paired primary effect", ""])
for pop in ("observed_valid", "observed_test"):
    for metric in ("position_coordinate_mse", "normalized_coordinate_mse"):
        mdmetric(pop, metric)
mdmetric("autonomous_test", "mean_rollout_mse")
report.extend(["## Horizon, failure, physical and measured-cost details", ""])
for pop in ("observed_valid", "observed_test"):
    for metric in ("failure_fraction", "previous_risk_base_residual_spearman", "previous_risk_gain_spearman",
                   "harmful_particle_fraction", "graph_build_and_selection_seconds", "forward_operational_seconds",
                   "scoring_seconds", "network_passes_for_standalone_policy"):
        mdmetric(pop, metric, False)
for metric in ("mse_at_1", "mse_at_10", "mse_at_50", "mse_at_200", "mse_at_500", "mse_at_995", "failure_fraction",
               "predicted_boundary_mean_fraction_outside_gt_1e_minus6", "ground_truth_boundary_mean_fraction_outside_gt_1e_minus6",
               "predicted_boundary_mean_step_maximum_excursion", "predicted_boundary_trajectory_maximum_excursion",
               "ground_truth_boundary_trajectory_maximum_excursion", "mean_rollout_wall_seconds",
               "mean_wall_seconds_including_native_parity", "mean_native_parity_seconds", "mean_graph_operational_seconds",
               "mean_forward_component_seconds", "mean_directed_edges", "mean_retained_optional_pairs",
               "mean_total_network_passes", "completed_steps", "forecast_network_passes"):
    mdmetric("autonomous_test", metric, False)
report.extend(["## Audit and failure history", "",
    "The strict summary completed once and reverified sources/inputs after analysis; the separate audit completed once with 12,101,908 checks. The audit independently checks observed-array and autonomous-scalar arithmetic, paired hierarchy and trace hashes. It does not independently admit checkpoint/data sources, replay graph guards, or regenerate unsaved predictions; strict summary provides complementary checks.", "",
    "All original evaluation, summary and audit processes were observed absent after exit. An ancillary metadata-print command wrote the completed-analysis gate and then raised TypeError because it called len() on the integer audit check count. The gate and bound evidence were reread and rehashed; no scientific stage was rerun and no result overwritten. The child also could not poll the parent's session registry; the original session exit was subsequently supplied by root. Both operational/reporting failures remain in the completion evidence manifest.", "",
    "The report is a compact scalar publication, not a standalone raw reproduction bundle. Immutable original checkpoints, source-data files, arrays, traces and detailed per-unit null reasons remain at the bound source paths. No omission is represented as a new successful measurement.", ""])
save("waterdrop110k_full_report.md", "\n".join(report))

appendix = [r"\section{WaterDrop: paired graph-exposure continuation}", r"\label{sec:waterdrop-110k-exposure}",
    r"This exploratory follow-up continues each of three faithful 100k parents for 10k additional updates under paired base-only and mixed-graph arms. It inherits Adam moments/counters, pairs frame/noise schedules and fixes learning rate $10^{-5}$. Each mixed example independently receives a uniformly drawn quarter of the optional annulus with probability $1/2$, appended bidirectionally after the native capped/self prefix. The six 110k endpoints are fixed; there is no checkpoint selection. This estimates continuation effects conditional on the original parents, distinct from Goop training from initialization and from the earlier NLL comparison. Test histories had previously been inspected.", "",
    r"All 128 validation and 297 test histories per endpoint are retained: 2,550 histories and 12,750 policy slots. All 810 autonomous outcomes complete through H995 (27 official test sources 3--29, five policies, six endpoints); every scientific failure count is zero. Validation/test are separate. Observed risk recomputes its score on the preceding observed base history; cached autonomous risk uses its own preceding selected graph. Frames average equally within trajectory and trajectories equally within seed; matched-unit differences precede aggregation. Means and sample SDs use all three seeds. The machine-readable evidence preserves all 91 metric families, every seed/contrast, required/defined counts and detail-file references; the complete report additionally displays horizon, physical, correlation and timing values.", "",
    r"\begin{table}[p]\centering\scriptsize",
    r"\caption{WaterDrop continuation, observed coordinate position MSE ($\times10^{-9}$) and normalized-acceleration coordinate MSE ($\times10^{-3}$). Risk25 is the preceding-observed-base controller. Every row has zero failed required histories.}",
    r"\label{tab:waterdrop110k-observed}", r"\begin{tabular}{llrrrr}\toprule",
    r"Training & Policy & Valid. position & Test position & Valid. normalized & Test normalized\\\midrule"]
for arm in ("base", "mix"):
    for p in w["populations"]["observed_test"]["policies"]:
        label = "risk25" if "risk" in p else p
        stats = [texstat(w["populations"][pop]["metrics"][m]["absolute"][arm][p], scale)
                 for m, scale in (("position_coordinate_mse", 1e9), ("normalized_coordinate_mse", 1e3))
                 for pop in ("observed_valid", "observed_test")]
        appendix.append(f"{arm} & {label} & " + " & ".join(stats) + r"\\")
appendix.extend([r"\bottomrule\end{tabular}\end{table}", "",
    r"\begin{table}[p]\centering\scriptsize",
    r"\caption{WaterDrop autonomous position MSE. All rows contain 81 completed outcomes with zero failures. H995 averages every forecast; @200 and @995 are pointwise. Seed columns give H995 seed means.}",
    r"\label{tab:waterdrop110k-rollout}", r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{llrrrrrr}\toprule", r"Training & Policy & H995 & @200 & @995 & Seed 0 & Seed 1 & Seed 2\\\midrule"])
am = w["populations"]["autonomous_test"]["metrics"]
for arm in ("base", "mix"):
    for p in w["populations"]["autonomous_test"]["policies"]:
        ss = am["mean_rollout_mse"]["absolute"][arm][p]
        vals = [texstat(am[m]["absolute"][arm][p]) for m in ("mean_rollout_mse", "mse_at_200", "mse_at_995")]
        vals += [f"{v:.4f}" for v in seeds(ss)]
        appendix.append(f"{arm} & {'cached risk25' if 'risk' in p else p} & " + " & ".join(vals) + r"\\")
appendix.extend([r"\bottomrule\end{tabular}\end{table}", "",
    r"\begin{table}[p]\centering\scriptsize",
    r"\caption{WaterDrop full-H995 boundary and descriptive measured cost. Outside counts particles beyond the metadata box by more than $10^{-6}$; excursion is trajectory-maximum coordinate excursion, then averaged. Matching truth is $2.678\%$ outside and $.00782$ excursion for all rows. Times are per trajectory: recorded policy wall time, then duration including native parity. All values are three-seed means and sample SDs. Different evolving geometries and fixed policy order prevent causal speedup claims.}",
    r"\label{tab:waterdrop110k-cost}", r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{llrrrrr}\toprule", r"Training & Policy & Outside (\%) & Excursion & Seconds & With parity & Edges/step\\\midrule"])
for arm in ("base", "mix"):
    for p in w["populations"]["autonomous_test"]["policies"]:
        vals = [texstat(am[m]["absolute"][arm][p], scale, prec) for m, scale, prec in (
            ("predicted_boundary_mean_fraction_outside_gt_1e_minus6",100,2),
            ("predicted_boundary_trajectory_maximum_excursion",1,3),
            ("mean_rollout_wall_seconds",1,2), ("mean_wall_seconds_including_native_parity",1,2),
            ("mean_directed_edges",1,0))]
        appendix.append(f"{arm} & {'cached risk25' if 'risk' in p else p} & " + " & ".join(vals) + r"\\")
appendix.extend([r"\bottomrule\end{tabular}\end{table}", "",
    r"At observed test histories, risk-minus-random position-MSE reverses from $(+1.616\pm1.070)\times10^{-10}$ to $(-.993\pm.666)\times10^{-10}$, changing sign in all three seeds; the interaction is $(-2.609\pm.741)\times10^{-10}$. Validation also has a negative interaction in every seed, but mixed risk-minus-random remains $(+.210\pm1.362)\times10^{-10}$, with seed signs $-/+/-$. Test base-policy error worsens in every seed under mix, and random-policy error worsens in two; this differs from the autonomous exposure effect. Actual risk-action-gain Spearman correlations remain small (base/mix: $-.0024\pm.0190$ / $.0166\pm.0174$), compared with base-residual correlations $.2856\pm.0623$ / $.2825\pm.0624$.", "",
    r"All four augmented policies improve H995 error in every paired seed under mix. Dense nevertheless remains worse than base within every seed and arm, while speed has the lowest mean error in each arm. The autonomous risk-minus-random interaction is $+.00284\pm.00466$, with seed effects $-.00161,+.00769,+.00246$. Thus the observed test reversal is a conditional positive placement result, not a consistent autonomous placement improvement. All outcomes complete, but substantial box excursions persist. Recorded duration and edge counts increase under mix for all five policies. Base, dense, random and speed make 995 forecast passes per trajectory; cached risk adds one initialization scoring pass (996 total). Previous-observed risk instead requires a separate scoring forward for every evaluated history; overlapping verification/preprocessing means its exact standalone latency cannot be reconstructed by summing stored components.", "",
    r"The completed strict summary rechecks sources and inputs; a complementary independent audit verifies raw observed-array/autonomous-scalar arithmetic, paired hierarchy and trace hashes in 12,101,908 checks. It does not independently replay source admission, guards or unsaved predictions. An ancillary metadata-print failure and a session-registry access limitation are retained separately and did not change the scientific outcomes. Neither scientific analysis stage was rerun.", ""])
save("candidate_waterdrop_appendix.tex", "\n".join(appendix))

abstract = """Residual-guided computation is attractive when prediction difficulty varies, but difficulty alone does not identify useful interventions. We study this distinction in particle graph simulators using an exact budget for additional neighbor pairs and cached residual-scale scores. A conditional allocation bound separates score error, temporal drift and mismatch with action benefit. Paired faithful-regression studies compare base-only and mixed-graph exposure across three seeds. Goop models train from initialization for 100,000 updates; WaterDrop models continue three fixed 100,000-update parents for 10,000 more. Under a fixed random allocation policy, mixed exposure lowers full-rollout error in every seed of both studies. Placement behaves differently: residual risk remains worse than random on matched Goop histories, while WaterDrop's observed-test ordering reverses in every seed after exposure. That reversal does not establish an autonomous placement gain: WaterDrop's rollout interaction has mixed signs, and three Goop guard failures leave its full-horizon interaction undefined. All 810 WaterDrop rollouts complete, but substantial boundary excursions persist in both studies. These results separate learning to use additional interactions from selecting them effectively and motivate evaluating observed action benefit, autonomous feedback, failures and measured cost together.
"""
save("candidate_abstract_dated_20261006.md", "## Additional candidate from completed paired exposure studies — October 6, 2026\n\n"
     + "Candidate only; the earlier proposal and current manuscript abstract remain preserved. This wording uses completed and verified Goop2D and WaterDrop 110k evidence. Pending Sand and D3 results are excluded.\n\n" + abstract
     + "\nSources: Goop `paired_scalar_summary.json.gz` SHA256 `" + sha(GOOP)
     + "`; WaterDrop `result.json` SHA256 `" + sha(WD) + "`; complementary WaterDrop audit SHA256 `" + sha(AUDIT)
     + "`. Exact claim mappings are in `work/deadline_research_20261005/waterdrop110k_integration_candidate_code_audit_v1/claim_source_map.json`. This proposal does not alter the manuscript title or abstract.\n")

notes = """# Integration candidate and compression notes

No manuscript file has been edited by this extraction. Root owns integration and compilation.

- Replace the existing `tab:goop-exposure-placement` table with `candidate_observed_contrast_table.tex`; it adds a clearly separated WaterDrop continuation column and the base-training risk-minus-random row so the observed reversal is visible.
- Insert `candidate_main_waterdrop.tex` after the Goop observed-state paragraph, or use it as the brief continuation subsection. It is approximately 150 words. Keep the two training lineages and H395/H995 distinct.
- Compress the two paragraphs under “Why can residual ranking misallocate?” in `work/conference_experiments_main.tex` into `candidate_compressed_existing_waterdrop.tex`. This preserves the faithful/NLL residual and actual-action correlations, every-seed random advantage at 100k, self-message control, alignment-versus-squared-change result, autonomous sign variation, failures, physical caveat and all six appendix references. The earlier diagnostic numbers are retained from current manuscript text, not newly reaudited by this extraction.
- The prior last sentence “The new Goop result further distinguishes improvement from graph exposure from improvement due to risk placement” becomes redundant once the explicit cross-study comparison is present.
- Use `candidate_waterdrop_appendix.tex` for supporting tables. Full precision and all three seed values, every policy, failure, boundary and cost metric are in `waterdrop110k_full_report.md` and all 91 metric families/12 paired contrasts in `waterdrop_all_scalar_mappings.json`. The scalar package is not a standalone raw reproduction bundle.
- The figure can remain Goop-specific. Its caption correctly states the Goop result and should not be generalized to WaterDrop. A new main figure is unnecessary given the page budget.

Narrow introduction/conclusion updates recommended (candidate text only):

Introduction study paragraph: “A separate WaterDrop study pairs 10k graph-exposure continuations from three fixed faithful 100k parents, alongside the earlier objective, action-benefit and graph-convention controls.”

Introduction results paragraph, following the existing Goop result: “WaterDrop continuation instead reverses the observed-test risk-versus-random ordering in every seed, while its autonomous placement interaction remains mixed. Both studies improve a fixed random policy after graph exposure. Together they show that exposure, observed-state placement and autonomous allocation require separate evidence.”

Conclusion, after the existing Goop sentence: “WaterDrop continuation provides a conditional positive result: graph exposure reverses risk's observed-test deficit in every seed. Yet its autonomous placement interaction changes sign across seeds, despite improved random-policy rollout error. Graph exposure can therefore improve a simulator and sometimes its observed placement without establishing a general autonomous allocation advantage.”

Limitations should explicitly add that WaterDrop effects are conditional on three inspected 100k parents and do not establish fresh-training generality. Do not call the paired studies independent confirmation. Actual title/abstract are untouched; the separate dated abstract candidate is optional author-review material.
"""
save("integration_notes.md", notes)

# Extraction validation: exact source objects and successful formatting, never a scientific rerun.
for row in rows:
    assert read_key(w, row["source_key"]) == row["statistics"]
for item in claims.values():
    obj = w if item["source_path"] == rel(WD) else g
    assert read_key(obj, item["key"]) == item["value"]
assert all(sha(p) == snapshot[rel(p)]["sha256"] for p in source_files)
save("extraction_verification.json", {"passed": True, "source_hashes_rechecked": True,
    "source_manuscripts_unchanged": True, "waterdrop_statistic_objects_exactly_mapped": len(rows),
    "waterdrop_metric_families": 91, "claim_objects_exactly_mapped": len(claims),
    "main_fragment_whitespace_word_count": len(main.split()), "abstract_word_count": len(abstract.split()),
    "no_scientific_execution": True, "independent_interpretation_note_sha256": sha(source_files[-1])})
print(json.dumps({"output": str(OUT), "statistic_objects": len(rows), "claim_objects": len(claims),
                  "main_words": len(main.split()), "abstract_words": len(abstract.split())}))
