# Independent full-rollout aggregation audit

Passed: **True**. Checks: 92,129; input files: 837.

Every saved compact record, raw scalar JSON record and aggregate remains preserved. No evaluator, production summarizer, model or dataset array was loaded.

All six jobs cover exactly 27 official trajectories (indices 3–29), five policies and 995 planned steps: 810 outcomes, 802 complete and eight failed. All eight failures belong to NLL seed2 and exceed the absolute-coordinate computational guard of 10. Their full-horizon errors, edges and boundary means remain undefined. The other five models complete all 135 outcomes each.

The faithful objective provides no consistent rollout benefit from lagged-risk allocation. Negative paired changes favor lagged risk. The primary error comparison is:

| Reference | Lagged risk minus reference MSE | Within-seed percentage change | Signs at seeds 0,1,2 |
|---|---:|---:|---|
| base | 0.00155489 ± 0.002630394 | 12.18609 ± 12.9152% | higher, higher, lower |
| random25 | 0.000826677 ± 0.002250333 | 8.757602 ± 12.21693% | higher, higher, lower |
| speed25 | 0.003008492 ± 0.003151335 | 10.0906 ± 5.123832% | higher, higher, higher |
| dense | -0.01003057 ± 0.004290466 | -28.99187 ± 17.16283% | lower, lower, lower |

Percentages are computed from each seed's equal-trajectory mean error, then averaged across the three seed pairs; they are not ratios of group means. Errors and percentages use sample SD across three training seeds, with no significance claim. All secondary-horizon comparisons are retained in the JSON.

| Objective | Policy | Mean rollout MSE | MSE@200 | MSE@995 |
|---|---|---:|---:|---:|
| faithful | base | 0.02664612 ± 0.02015343 | 0.01289084 ± 0.008576386 | 0.03822668 ± 0.02558685 |
| faithful | dense | 0.03823158 ± 0.01879609 | 0.02233968 ± 0.02030936 | 0.05677427 ± 0.01888736 |
| faithful | random25 | 0.02737433 ± 0.02009854 | 0.01459265 ± 0.01242569 | 0.03911481 ± 0.02473863 |
| faithful | speed25 | 0.02519252 ± 0.01477317 | 0.01278473 ± 0.00777339 | 0.03652089 ± 0.01877135 |
| faithful | laggedrisk25 | 0.02820101 ± 0.01786212 | 0.01402393 ± 0.01093972 | 0.04135817 ± 0.02294187 |
| nll | base | undefined | 0.02770471 ± 0.02460146 | undefined |
| nll | dense | undefined | 0.02712491 ± 0.02221797 | undefined |
| nll | random25 | undefined | 0.02662946 ± 0.02360219 | undefined |
| nll | speed25 | 0.0320984 ± 0.01408378 | 0.02726063 ± 0.02313562 | 0.04269211 ± 0.01529203 |
| nll | laggedrisk25 | undefined | 0.02527981 ± 0.02419027 | undefined |

NLL full-horizon policy comparisons involving any failed policy remain undefined. Speed25 alone has all 81 NLL trajectories complete. All failures occur after step 200, so the separately reported 200-step metric uses all required trajectories.

| Failed model | Source index | Policy | Accepted steps | Rejected step | Maximum absolute coordinate |
|---|---:|---|---:|---:|---:|
| nll seed2 | 6 | random25 | 898 | 899 | 10.0103216171 |
| nll seed2 | 16 | dense | 901 | 902 | 10.0091485977 |
| nll seed2 | 17 | base | 918 | 919 | 10.0100908279 |
| nll seed2 | 17 | dense | 912 | 913 | 10.0111122131 |
| nll seed2 | 20 | random25 | 918 | 919 | 10.0050868988 |
| nll seed2 | 26 | random25 | 876 | 877 | 10.0039644241 |
| nll seed2 | 29 | random25 | 893 | 894 | 10.0061340332 |
| nll seed2 | 29 | laggedrisk25 | 891 | 892 | 10.0082979202 |

Boundary diagnostics also give mixed evidence. The faithful lagged-risk outside fraction can be smaller while its excursion magnitudes are larger; completing 995 numerical steps does not establish physical plausibility. The reference truth itself has nonzero boundary excursions. Fractions use a 1e-6 coordinate threshold; each trajectory averages its steps equally, and each seed averages its 27 trajectories equally.

| Objective | Policy | Predicted outside fraction | Truth outside fraction | Predicted mean step maximum excursion | Truth mean step maximum excursion |
|---|---|---:|---:|---:|---:|
| faithful | base | 0.1346418 ± 0.06004986 | 0.02677722 ± 0 | 0.141842 ± 0.09591686 | 0.005671049 ± 0 |
| faithful | dense | 0.1487802 ± 0.05287402 | 0.02677722 ± 0 | 0.2575867 ± 0.08510605 | 0.005671049 ± 0 |
| faithful | random25 | 0.1360525 ± 0.06251167 | 0.02677722 ± 0 | 0.1633152 ± 0.0969045 | 0.005671049 ± 0 |
| faithful | speed25 | 0.1322044 ± 0.05995362 | 0.02677722 ± 0 | 0.1656407 ± 0.08404541 | 0.005671049 ± 0 |
| faithful | laggedrisk25 | 0.1310547 ± 0.06095894 | 0.02677722 ± 0 | 0.1924956 ± 0.1063386 | 0.005671049 ± 0 |
| nll | base | undefined | undefined | undefined | undefined |
| nll | dense | undefined | undefined | undefined | undefined |
| nll | random25 | undefined | undefined | undefined | undefined |
| nll | speed25 | 0.1173857 ± 0.07769509 | 0.02677722 ± 0 | 0.3630286 ± 0.2730775 | 0.005671049 ± 0 |
| nll | laggedrisk25 | undefined | undefined | undefined | undefined |

Full-horizon boundary comparisons and the distinct accepted prefixes of all eight failures are retained in the JSON. No survivors-only mean is substituted.

| Faithful reference | Lagged-risk minus reference outside fraction | Mean step maximum excursion difference | Mean trajectory maximum excursion difference |
|---|---:|---:|---:|
| base | -0.00358709 ± 0.01163967 | 0.05065354 ± 0.01239303 | 0.06232662 ± 0.01140381 |
| random25 | -0.004997788 ± 0.006110895 | 0.02918036 ± 0.01060464 | 0.03343588 ± 0.02171129 |
| speed25 | -0.001149734 ± 0.001959123 | 0.02685492 ± 0.0281794 | 0.02875331 ± 0.03625457 |
| dense | -0.01772554 ± 0.008139702 | -0.06509116 ± 0.02644059 | -0.0851825 ± 0.03754602 |

Recorded per-rollout wall counters sum to 10138.507160 s; six invocation counters sum to 10173.892703 s. These are different timing scopes.

Measured autonomous rollout cost includes failure prefixes and lagged-risk initialization. Fixed policy order, evolving policy-dependent geometry/edge counts, differing failed lengths and warmup scope prevent a controlled speedup conclusion. Use separately measured same-state repeated timings for placement/runtime claims.

Candidate/retained counts and worst excursions are recorded separately for accepted forecasts, rejected forecasts and initial base-scoring passes in the JSON. A retained pair is one undirected pair (two directed edges). Peak memory was not recorded and is not inferred from graph size.

This is a prospective full-architecture extension after inspection of historical aggregates and three official test trajectories. It is not pristine independent confirmation. The evidence does not establish conference readiness, a general allocation advantage, or a speedup.
