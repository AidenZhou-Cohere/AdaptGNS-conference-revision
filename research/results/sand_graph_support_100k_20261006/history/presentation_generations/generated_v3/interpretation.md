# Sand: complete admitted interpretation

The principal new finding is adverse for cached-risk allocation. All 1,080 autonomous outcomes and all 3,648 mixed-stage cells complete, but graph exposure worsens cached-risk H314 error and its risk-minus-random gap in every paired seed. This does not contradict a conditional observed-history interaction; it demonstrates that the interaction does not transfer automatically through autonomous feedback.

## Exposure and absolute allocation

| Evaluation policy | Observed-test mix−base (×10⁻⁹) | Seed signs | H314 mix−base | Seed signs |
|---|---:|:---:|---:|:---:|
| base | -0.8437067 ± 2.9134 | + / + / - | -3.776074e-05 ± 0.002628252 | - / - / + |
| dense | -14.65839 ± 7.86219 | - / - / - | -0.003319072 ± 0.002959939 | - / - / - |
| random25 | -3.853536 ± 3.66441 | - / - / - | -0.01057904 ± 0.01861049 | - / - / + |
| speed25 | -5.53874 ± 5.645589 | - / - / - | +0.009857673 ± 0.01898406 | - / + / + |
| laggedrisk25 | -6.744583 ± 6.46724 | - / - / - | +0.009167157 ± 0.01201005 | + / + / + |
| relative-velocity-RMS25 | -7.351095 ± 5.33382 | - / - / - | -0.002746434 ± 0.004975468 | - / - / + |

Every expanded policy improves observed-test error in all three seeds. Base-policy observed error worsens in seeds 0 and 1. Under autonomous feedback, only dense exposure improves all three seeds; random, base and RMS improve two, and speed improves one. Cached risk improves none. Dense itself remains worse than base and random in all seeds and both arms. Native base beats random in all seeds of both arms. A training improvement under a fixed expanded policy therefore cannot be presented as evidence that expansion outperforms the native graph.

## Risk placement at each endpoint

| Endpoint | Base risk−random | Mixed risk−random | Interaction | Interaction signs |
|---|---:|---:|---:|:---:|
| Observed validation (×10⁻⁹) | +11.08709 ± 2.125031 | +7.014335 ± 0.6672391 | -4.072755 ± 1.807203 | - / - / - |
| Observed test (×10⁻⁹) | +4.759377 ± 3.804163 | +1.86833 ± 1.037691 | -2.891047 ± 2.809052 | - / - / - |
| Autonomous H314 | +0.01416149 ± 0.02120679 | +0.03390769 ± 0.05182419 | +0.0197462 ± 0.03061955 | + / + / + |

All observed and autonomous within-arm risk-minus-random values are positive in every seed. The observed interaction is negative in every seed; the autonomous interaction is positive in every seed. The complete companion retains the exact ordered seed values, including the small positive H314 interaction in seed 2 (3.9882560103739195e−5). Three training seeds do not support treating trajectories or frames as independent replicates.

## Physical and computational qualifications

Across all six policies, mixed training lowers the across-seed mean outside-box fraction but increases trajectory-maximum excursion in every paired seed. Mixed random25 has 10.4701% outside (truth 0.0618251%) and mean trajectory-maximum excursion 0.324406 (truth 0.000795367). Thus a lower fraction outside is not a uniform physical improvement. All six mean committed-call times increase under mix; their seed signs are mixed. Synchronized calls include native parity and exclude separate setup/publication/recovery. Shared hosts, fixed policy order and changed rollout geometries preclude a causal speedup claim.

Observed-test residual-risk/base-error correlations remain positive (base 0.251146±0.086003; mix 0.271585±0.025858), while risk/own-sparse-benefit correlations are negative (base −0.118949±0.044055; mix −0.062382±0.036330), with all per-model benefit correlations negative. Clean-validation MSE improves only in seed 2; NLL improves in seeds 1 and 2. These mixed residual-fit diagnostics do not establish action-value calibration.

## Cross-material reading and provenance

The fixed tables retain Goop, WaterDrop and Sand in their predeclared order. Goop and WaterDrop improve random-policy autonomous error in every paired seed, while Sand improves only two. Goop retains three guard failures and undefined affected full-horizon statistics. WaterDrop is a 100k-parent-to-110k continuation with five policies, so RMS is not predeclared. Goop and Sand are fresh 100k training studies with six policies; their horizons are395 and314, versus WaterDrop995. These differences and Sand’s distinct CUDA lineage prevent a controlled material-effect interpretation. No title/abstract update or favorable-policy selection is made.

The compact companion contains all4,216 three-seed statistic objects:1,107 Goop,2,002 WaterDrop and1,107 Sand, together with exact input pins and full Sand accounting/runtime provenance. Primary appendix tables retain every policy/arm and all primary test/full-horizon seed means. The retained paired audit contains25,606 checks. Saved-array/scalar-series support does not replay unsaved trajectories, source normalization, graph construction or model execution. Earlier failed source variants, numerical gates and transport attempts remain preserved.
