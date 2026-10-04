# Full same-state diagnostic results

**Results pending. No full diagnostic outcomes are present; no experimental results are reported.**

Completed evaluator runs: 0/6; eligible for aggregation: 0/6. Each model requires 297 fixed observed frames (27 trajectories × 11 targets).

Reported means and sample SDs use exactly three training-seed means. The six timing repetitions are averaged within each frame. Missing runs, failed frames and undefined required correlations remain explicit; no successful subset is substituted.

| Model | State | Recorded frames |
|---|---|---:|
| faithful seed 0 | missing | 0/297 |
| faithful seed 1 | missing | 0/297 |
| faithful seed 2 | missing | 0/297 |
| nll seed 0 | missing | 0/297 |
| nll seed 1 | missing | 0/297 |
| nll seed 2 | missing | 0/297 |

| Objective | Policy | Position MSE | Normalized acceleration MSE |
|---|---|---:|---:|
| faithful | base | — | — |
| faithful | dense | — | — |
| faithful | random25 | — | — |
| faithful | speed25 | — | — |
| faithful | previous-observed-base-risk25 | — | — |
| nll | base | — | — |
| nll | dense | — | — |
| nll | random25 | — | — |
| nll | speed25 | — | — |
| nll | previous-observed-base-risk25 | — | — |

| Objective | Risk correlation with current outcome | Mean frame Spearman |
|---|---|---:|
| faithful | previous risk vs base error | — |
| faithful | previous risk vs dense benefit | — |
| faithful | current base risk vs base error | — |
| faithful | current base risk vs dense benefit | — |
| nll | previous risk vs base error | — |
| nll | previous risk vs dense benefit | — |
| nll | current base risk vs base error | — |
| nll | current base risk vs dense benefit | — |

| Objective | Signed dense benefit, normalized vector SE | Natural-base seconds | Shared-superset-base seconds | Previous-risk score-generation seconds |
|---|---:|---:|---:|---:|
| faithful | — | — | — | — |
| nll | — | — | — | — |

The JSON includes signed paired policy/objective differences, failure and undefined-correlation counts, explicit missing frame IDs, and a separately labeled particle-weighted error/benefit alternative. Positive/negative/zero fractions refer to normalized-vector benefit; unequal coordinate scales can give a different sign from position-space benefit. The previous-observed-base risk diagnostic is distinct from autonomous cached risk. Timing is a reference-pipeline measurement; no cached amortization, significance or general speedup claim is made.
