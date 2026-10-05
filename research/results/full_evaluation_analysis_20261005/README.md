# Fixed full-model WaterDrop findings

## October 5: locked full-architecture evaluation complete

All six fixed 100,000-update models and all twelve evaluation jobs completed. The final queue status is **October 5, 20:22:35 UTC**. The experiment covers **810 autonomous trajectory-policy outcomes** (27 test trajectories × five policies × six models) and **1,782 same-state frames**. Official test indices are **3–29**; prior inspection of indices 0–2 and historical aggregates prevents a claim of pristine independent confirmation.

**The results do not support general cached-risk accuracy–runtime superiority.** All 405 faithful outcomes completed. Eight NLL seed-2 outcomes hit the coordinate guard: base 1, dense 2, random 4, cached risk 1, speed 0. Affected all-sample full-horizon MSE, edge and boundary means remain undefined. All failures and accepted-prefix diagnostics are retained without retries.

Faithful cached risk versus base, random, speed and dense has paired full-rollout MSE differences **+0.001555 ± 0.002630**, **+0.000827 ± 0.002250**, **+0.003008 ± 0.003151**, and **−0.010031 ± 0.004290**, respectively. Within-seed percentage changes average **+12.19 ± 12.92%**, **+8.76 ± 12.22%**, **+10.09 ± 5.12%**, and **−28.99 ± 17.16%**. Differences against base/random change sign; speed wins and dense loses in all three faithful seeds. These are descriptive three-seed means and sample SDs, not significance claims.

At identical observed histories, previous-base risk has mean frame correlations of **0.357 ± 0.060 / 0.411 ± 0.023** with base residuals (faithful/NLL), versus **−0.022 ± 0.045 / −0.048 ± 0.012** with signed dense benefit. Random allocation beats previous-observed risk in all six seed-model means. This teacher-forced diagnostic is distinct from autonomous cached risk. The natural-base costs are **10.740 / 10.496 ms**, random costs **11.801 / 11.555 ms**, and previous-observed-risk costs **22.382 / 21.881 ms**, including its extra scoring pass. Every policy executes the risk head; fixed-order autonomous durations and different geometries do not establish speedup.

Computational completion does not certify physical validity. Faithful cached risk has **13.11 ± 6.10%** outside particles versus **2.68%** for same-frame truth. Its mean trajectory-maximum excursion is **0.3016**, compared with **0.2393** for base and **0.00782** for truth. Lower outside fractions can coexist with worse excursions. Conservation, peak memory and optimized mean-only performance remain unmeasured.

Independent audits pass **92,129 rollout arithmetic/aggregation checks**, **1,068,869 same-state array/aggregation checks**, and **20,928 provenance/integrity checks**. All 761 previously hashed row files and 41 earlier checkpoint-verification records/timestamps are preserved. One NLL seed-2 observed-history case (source 6, target 106) produced two risk-selected pair hashes over six timed calls; maximum recorded prediction difference **2.45e−5**, exact budgets intact. All repetitions and the initial audit-only warmup-null handling failure remain recorded.

The manuscript includes the full results and passes native compilation with the eight-page main-text assertion. The evidence report includes all endpoints, failures, boundary references and common-state costs. Author verification, registered-abstract comparison, exported manuscript PDF inspection and anonymous submission/asset-rights checks remain required. No submission or external message has been sent.

[Full rollout analysis](rollout_analysis.md) · [Same-state analysis](same_state_findings.md) · [Compact original evidence](../full_evaluation_20261005/README.md)


## Reproduction and scope

The independent audit scripts preserve their original local paths and input hashes. Full raw-record/array audits require the original inventoried outputs; the compact public package does not contain those arrays. The production `research.full_rollout`, `research.full_same_state` and strict summarizers reproduce measurements from the fixed final checkpoints and official data under the recorded protocols. Checkpoints and data remain local; retraining or separately supplied weights is required. No historical checkpoint has been recovered.

To regenerate the two new manuscript table/prose inserts using only the public summaries, run from the repository root:

```sh
python3 -m research.render_full_evaluation_tables --output-dir /tmp/full-evaluation-tables
```

The renderer requires the original summary hashes because its interpretation is specific to these frozen results. It produces `full_evaluation_main.tex`, `full_evaluation_appendix.tex` and a generation manifest. It does not rerun inference or alter the original summaries. The original audit-only warmup-null handling failure is preserved as `rollout_analysis_attempt1.*`; the corrected audit does not modify scientific inputs.

This research-fork package is nonanonymous and is not a submission supplement. Dataset redistribution rights, author verification, venue eligibility and anonymous packaging remain author tasks.
