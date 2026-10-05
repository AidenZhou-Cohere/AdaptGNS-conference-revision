# Full same-state diagnostic results

**Diagnostic result set: complete.**

Completed evaluator runs: 6/6; eligible for aggregation: 6/6. Each model requires 297 fixed observed frames (27 trajectories × 11 targets).

Reported means and sample SDs use exactly three training-seed means. The six timing repetitions are averaged within each frame. Missing runs, failed frames and undefined required correlations remain explicit; no successful subset is substituted.

| Model | State | Recorded frames |
|---|---|---:|
| faithful seed 0 | complete | 297/297 |
| faithful seed 1 | complete | 297/297 |
| faithful seed 2 | complete | 297/297 |
| nll seed 0 | complete | 297/297 |
| nll seed 1 | complete | 297/297 |
| nll seed 2 | complete | 297/297 |

| Objective | Policy | Position MSE | Normalized acceleration MSE |
|---|---|---:|---:|
| faithful | base | 4.35611e-09 ± 6.29e-11 | 0.00959136 ± 0.000138 |
| faithful | dense | 5.41183e-09 ± 3.86e-10 | 0.0119159 ± 0.000849 |
| faithful | random25 | 4.08243e-09 ± 1.2e-10 | 0.00898859 ± 0.000265 |
| faithful | speed25 | 4.28355e-09 ± 6.55e-11 | 0.00943158 ± 0.000144 |
| faithful | previous-observed-base-risk25 | 4.3031e-09 ± 1.17e-10 | 0.00947464 ± 0.000256 |
| nll | base | 4.81928e-09 ± 2.3e-10 | 0.01061 ± 0.000506 |
| nll | dense | 5.61637e-09 ± 8.77e-11 | 0.0123657 ± 0.000195 |
| nll | random25 | 4.46882e-09 ± 6.24e-11 | 0.0098385 ± 0.000138 |
| nll | speed25 | 4.804e-09 ± 9.03e-11 | 0.0105771 ± 0.000199 |
| nll | previous-observed-base-risk25 | 4.85908e-09 ± 3.26e-11 | 0.0106985 ± 7.23e-05 |

| Objective | Risk correlation with current outcome | Mean frame Spearman |
|---|---|---:|
| faithful | previous risk vs base error | 0.35706 ± 0.0598 |
| faithful | previous risk vs dense benefit | -0.0218746 ± 0.0448 |
| faithful | current base risk vs base error | 0.357189 ± 0.0592 |
| faithful | current base risk vs dense benefit | -0.0222031 ± 0.0447 |
| nll | previous risk vs base error | 0.411323 ± 0.0226 |
| nll | previous risk vs dense benefit | -0.0481923 ± 0.0116 |
| nll | current base risk vs base error | 0.4123 ± 0.0241 |
| nll | current base risk vs dense benefit | -0.0474381 ± 0.0113 |

| Objective | Signed dense benefit, normalized vector SE | Natural-base seconds | Shared-superset-base seconds | Previous-risk score-generation seconds |
|---|---:|---:|---:|---:|
| faithful | -0.00464907 ± 0.00189 | 0.0107397 ± 0.000135 | 0.011067 ± 0.000147 | 0.0105964 ± 0.000119 |
| nll | -0.00351137 ± 0.000642 | 0.0104956 ± 2.67e-05 | 0.0108147 ± 3.23e-05 | 0.0103467 ± 1.84e-05 |

The JSON includes signed paired policy/objective differences, failure and undefined-correlation counts, explicit missing frame IDs, and a separately labeled particle-weighted error/benefit alternative. Positive/negative/zero fractions refer to normalized-vector benefit; unequal coordinate scales can give a different sign from position-space benefit. The previous-observed-base risk diagnostic is distinct from autonomous cached risk. Timing is a reference-pipeline measurement; no cached amortization, significance or general speedup claim is made.
