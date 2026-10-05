# Exploratory graph-convention bridge

Post-inspection follow-up; frozen original results remain separate. Means and sample SDs use three paired seed means. Validation/test and all failures remain separate. Operational durations are not a speed comparison.

## valid: normalized coordinate MSE

| Case | Faithful | NLL |
|---|---:|---:|
| base_cap128_loops0 | 0.010553652 ± 0.000434029 | 0.011425601 ± 0.000316256 |
| base_cap128_loops1 | 0.0086427241 ± 0.000479509 | 0.00948654536 ± 0.000528114 |
| base_uncapped_loops0 | 0.010553652 ± 0.000434029 | 0.011425601 ± 0.000316256 |
| base_uncapped_loops1 | 0.0086427241 ± 0.000479509 | 0.00948654536 ± 0.000528114 |
| dense_cap128_loops0 | 0.0131163194 ± 0.000869252 | 0.0144579483 ± 0.001101 |
| dense_cap128_loops1 | 0.010978327 ± 0.000950987 | 0.0127418182 ± 0.00107216 |
| dense_uncapped_loops0 | 0.0131163194 ± 0.000869252 | 0.0144579483 ± 0.001101 |
| random25_uncapped_loops0 | 0.00999740299 ± 0.000196424 | 0.010927796 ± 0.000535936 |
| speed25_uncapped_loops0 | 0.0107958651 ± 0.000354902 | 0.011890778 ± 0.000520689 |
| previous-observed-base-risk25_uncapped_loops0 | 0.0107870026 ± 0.000440876 | 0.0125257835 ± 0.00103984 |
| current-base-risk25_uncapped_loops0 | 0.0108148924 ± 0.000404973 | 0.0125506872 ± 0.00105297 |
| dense_uncapped_loops1 | 0.010978327 ± 0.000950987 | 0.0127418182 ± 0.00107216 |
| random25_uncapped_loops1 | 0.00822275101 ± 0.000437961 | 0.00931075471 ± 0.000682544 |
| speed25_uncapped_loops1 | 0.00911317694 ± 0.00046861 | 0.0100985887 ± 0.000576613 |
| previous-observed-base-risk25_uncapped_loops1 | 0.00910033338 ± 0.000616144 | 0.0108918721 ± 0.00105389 |
| current-base-risk25_uncapped_loops1 | 0.00910138842 ± 0.000622287 | 0.0109239128 ± 0.00108078 |

| Paired interaction (loop1 minus loop0) | Faithful | NLL |
|---|---:|---:|
| loop_interaction__previous-observed-base-risk25_minus_random25 | 8.79828086e-05 ± 0.000138643 | -1.68701536e-05 ± 0.000142195 |
| loop_interaction__dense_minus_base | -0.000227064478 ± 9.12618e-05 | 0.000222925526 ± 0.000261356 |
| loop_interaction__current-base-risk25_minus_random25 | 6.11480461e-05 ± 8.87163e-05 | -9.73313339e-06 ± 0.000141194 |

## test: normalized coordinate MSE

| Case | Faithful | NLL |
|---|---:|---:|
| base_cap128_loops0 | 0.00959135856 ± 0.000137845 | 0.0106100148 ± 0.000505693 |
| base_cap128_loops1 | 0.00787928785 ± 0.000164056 | 0.00877493202 ± 0.00022189 |
| base_uncapped_loops0 | 0.00959135856 ± 0.000137845 | 0.0106100148 ± 0.000505693 |
| base_uncapped_loops1 | 0.00787928785 ± 0.000164056 | 0.00877493202 ± 0.00022189 |
| dense_cap128_loops0 | 0.0119158936 ± 0.000849317 | 0.0123656992 ± 0.000194509 |
| dense_cap128_loops1 | 0.00992164712 ± 0.000797516 | 0.0107826172 ± 0.000153556 |
| dense_uncapped_loops0 | 0.0119158936 ± 0.000849317 | 0.0123656992 ± 0.000194509 |
| random25_uncapped_loops0 | 0.00899730206 ± 0.000237211 | 0.00988678969 ± 0.000442476 |
| speed25_uncapped_loops0 | 0.00943157929 ± 0.000143994 | 0.0105771312 ± 0.000199238 |
| previous-observed-base-risk25_uncapped_loops0 | 0.00947464212 ± 0.000256444 | 0.0106985463 ± 7.22841e-05 |
| current-base-risk25_uncapped_loops0 | 0.00951327037 ± 0.000284557 | 0.0106518906 ± 2.96927e-05 |
| dense_uncapped_loops1 | 0.00992164712 ± 0.000797516 | 0.0107826172 ± 0.000153556 |
| random25_uncapped_loops1 | 0.00736064354 ± 0.000174783 | 0.00830986175 ± 0.000256978 |
| speed25_uncapped_loops1 | 0.00788995534 ± 0.000159019 | 0.00891996354 ± 7.69527e-05 |
| previous-observed-base-risk25_uncapped_loops1 | 0.00786214929 ± 0.000273277 | 0.00907385692 ± 8.19283e-05 |
| current-base-risk25_uncapped_loops1 | 0.00790133087 ± 0.000295528 | 0.00907646335 ± 8.97228e-05 |

| Paired interaction (loop1 minus loop0) | Faithful | NLL |
|---|---:|---:|
| loop_interaction__previous-observed-base-risk25_minus_random25 | 2.41656911e-05 ± 4.95664e-05 | -4.77614288e-05 ± 0.000212574 |
| loop_interaction__dense_minus_base | -0.00028217581 ± 0.000159528 | 0.000252000848 ± 0.000339435 |
| loop_interaction__current-base-risk25_minus_random25 | 2.47190263e-05 ± 1.55479e-05 | 1.50065854e-06 ± 0.000231542 |

Risk loop contrasts include changed scoring and selection; base/dense/random/speed have fixed selected nonself pairs between loop arms. Same-state effects do not establish autonomous stability or resolve lack of expansion training support.

Arithmetic, coverage and integrity audit checks: 851400.
