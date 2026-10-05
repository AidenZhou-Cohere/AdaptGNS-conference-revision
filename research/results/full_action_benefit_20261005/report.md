# Exploratory actual-action benefit analysis

Post-inspection saved-array analysis; no inference, retraining or new test records.

Means and sample SDs use exactly three seed means. All per-particle benefits, frame/trajectory values, undefined counts, input hashes and failures are retained.

| Objective | Action | Mean normalized vector benefit | Harmful particle fraction | Previous-risk / benefit Spearman |
|---|---|---:|---:|---:|
| faithful | dense | -0.004649071 ± 0.001893 | 0.6032028 ± 0.01657 | -0.02187459 ± 0.04484 |
| faithful | random25 | 0.001205529 ± 0.0007164 | 0.506922 ± 0.01609 | 0.01324021 ± 0.03243 |
| faithful | speed25 | 0.0003195576 ± 0.0005229 | 0.4224828 ± 0.009112 | -0.03480659 ± 0.02127 |
| faithful | previous-observed-base-risk25 | 0.0002334311 ± 0.0007593 | 0.5139788 ± 0.007801 | -0.02121411 ± 0.0487 |
| nll | dense | -0.00351137 ± 0.0006416 | 0.614058 ± 0.01241 | -0.0481923 ± 0.0116 |
| nll | random25 | 0.001543038 ± 0.0007446 | 0.5023249 ± 0.01236 | 0.007575112 ± 0.008046 |
| nll | speed25 | 6.576685e-05 ± 0.0006998 | 0.4208792 ± 0.009094 | -0.04202242 ± 0.009671 |
| nll | previous-observed-base-risk25 | -0.0001770636 ± 0.0009436 | 0.5013187 ± 0.02057 | -0.05178438 ± 0.01307 |

Positive benefit means base error minus actual-action error; negative values are retained. Benefits are whole-graph interventions, share base residual algebraically, and are not marginal edge utility. Position-space results and current-risk profiles remain separate in JSON.

| Objective | Hindsight portfolio | Normalized coordinate MSE | Benefit from allowing base | Base strictly best fraction |
|---|---|---:|---:|---:|
| faithful | all_five | 0.008083915 ± 8.808e-05 | 0.0002146998 ± 9.058e-05 | 0.3535354 ± 0.1218 |
| faithful | budgeted_with_base | 0.008100736 ± 8.039e-05 | 0.0002151579 ± 9.037e-05 | 0.359147 ± 0.1196 |
| nll | all_five | 0.008971435 ± 0.0002602 | 0.0002176519 ± 7.802e-05 | 0.3580247 ± 0.06257 |
| nll | budgeted_with_base | 0.008992706 ± 0.0002589 | 0.0002176519 ± 7.802e-05 | 0.3580247 ± 0.06257 |

The oracle uses one complete prediction per observed frame and target information in hindsight. It is not deployable, not a particlewise mixture, and not an optimum over arbitrary edge subsets. The all-five portfolio includes the more expensive dense graph. No speedup, significance or independent confirmation is claimed.
