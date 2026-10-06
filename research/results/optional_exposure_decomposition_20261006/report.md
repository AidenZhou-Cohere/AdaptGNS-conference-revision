# Exploratory optional-exposure error decomposition

Exact decomposition of previously inspected, original no-loop observed-history predictions. No new inference or permutation null.

All entries are normalized coordinate squared-error differences, left minus right; positive error favors the right action. Three equal-trajectory seed means and sample SDs are descriptive.

| Objective | Pair | Whole error gap | Neither | Left only | Right only | Both |
|---|---|---:|---:|---:|---:|---:|
| faithful | risk_minus_random | 0.00048604875 ± 0.00011128 | 0.0001049155 ± 1.6151e-05 | -7.7491866e-05 ± 4.6145e-05 | 9.358349e-05 ± 1.3843e-05 | 0.00036504163 ± 9.7102e-05 |
| faithful | risk_minus_speed | 4.3063281e-05 ± 0.00011831 | 7.3305382e-05 ± 2.7947e-05 | -5.2672099e-05 ± 5.45e-05 | -7.1031244e-05 ± 4.2781e-05 | 9.3461242e-05 ± 5.2596e-05 |
| faithful | speed_minus_random | 0.00044298547 ± 0.00012927 | 7.3302205e-05 ± 2.444e-05 | -6.1575285e-05 ± 5.3887e-05 | 0.00011224823 ± 3.672e-05 | 0.00031901031 ± 6.2033e-05 |
| nll | risk_minus_random | 0.00086005057 ± 0.00010188 | 0.00014814386 ± 4.4481e-05 | 8.4427098e-05 ± 7.3341e-05 | 0.0001194449 ± 5.257e-05 | 0.00050803471 ± 2.0087e-05 |
| nll | risk_minus_speed | 0.0001214152 ± 0.00013919 | 8.3062199e-05 ± 2.6331e-05 | 8.2835426e-06 ± 4.1125e-05 | -7.4066649e-05 ± 1.2221e-05 | 0.00010413611 ± 7.3289e-05 |
| nll | speed_minus_random | 0.00073863537 ± 8.1541e-05 | 0.00011395244 ± 3.3406e-05 | 8.7300635e-05 ± 5.1746e-05 | 0.0001342685 ± 5.9185e-05 | 0.0004031138 ± 3.8948e-05 |

The four unconditional contributions sum to the whole error gap. Within each group, error contribution = cost-difference contribution minus alignment-difference contribution; all quantities divide vector terms by 2. The supplementary less/equal/more-degree subgroups partition only the both-covered group.

Conditional means remain null for empty groups and are never averaged over available frames only. Separately labeled weighted conditionals divide the fixed-weight contribution by its fixed-weight group fraction. Missing required scientific inputs propagate to the relevant full-population means.

Exposure is selected by policy and is not a pretreatment confounder. Multiple message-passing blocks can affect particles without directly incident optional edges. This partition does not identify marginal edge value, establish a causal concentration mechanism, prove autonomous stability or measure inference speed.
