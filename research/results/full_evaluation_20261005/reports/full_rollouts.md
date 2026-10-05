# Full WaterDrop rollout results

All six evaluator runs are complete. Individual trajectories may still have failed; those failures remain in the results.

Completed evaluator runs: 6/6; eligible for aggregation: 6/6. Required: faithful and NLL, seeds 0–2, five policies, official test source indices 3–29, 995 forecast steps.

Errors are averaged equally over 27 trajectories within each seed. Tables show the mean ± sample SD of exactly three seed means. Missing values are undefined or pending; failed trajectories are never dropped.

| Model | State | Recorded policy trajectories |
|---|---|---:|
| faithful seed 0 | complete | 135/135 |
| faithful seed 1 | complete | 135/135 |
| faithful seed 2 | complete | 135/135 |
| nll seed 0 | complete | 135/135 |
| nll seed 1 | complete | 135/135 |
| nll seed 2 | complete | 135/135 |

| Objective | Policy | Mean rollout MSE | MSE@200 | MSE@995 | Failure fraction |
|---|---|---:|---:|---:|---:|
| faithful | base | 0.0266461 ± 0.0202 | 0.0128908 ± 0.00858 | 0.0382267 ± 0.0256 | 0 ± 0 |
| faithful | dense | 0.0382316 ± 0.0188 | 0.0223397 ± 0.0203 | 0.0567743 ± 0.0189 | 0 ± 0 |
| faithful | random25 | 0.0273743 ± 0.0201 | 0.0145926 ± 0.0124 | 0.0391148 ± 0.0247 | 0 ± 0 |
| faithful | speed25 | 0.0251925 ± 0.0148 | 0.0127847 ± 0.00777 | 0.0365209 ± 0.0188 | 0 ± 0 |
| faithful | laggedrisk25 | 0.028201 ± 0.0179 | 0.0140239 ± 0.0109 | 0.0413582 ± 0.0229 | 0 ± 0 |
| nll | base | — | 0.0277047 ± 0.0246 | — | 0.0123457 ± 0.0214 |
| nll | dense | — | 0.0271249 ± 0.0222 | — | 0.0246914 ± 0.0428 |
| nll | random25 | — | 0.0266295 ± 0.0236 | — | 0.0493827 ± 0.0855 |
| nll | speed25 | 0.0320984 ± 0.0141 | 0.0272606 ± 0.0231 | 0.0426921 ± 0.0153 | 0 ± 0 |
| nll | laggedrisk25 | — | 0.0252798 ± 0.0242 | — | 0.0123457 ± 0.0214 |

Boundary diagnostics use the fraction of particles outside the metadata bounds by more than 1e-6. Fractions and per-step maxima are averaged equally over forecast steps within each trajectory, then equally over trajectories within each seed. The trajectory maximum is the worst excursion in one trajectory; its table column averages those maxima across trajectories, not the single worst particle from the entire experiment. Values are mean ± sample SD across all three seeds. Ground-truth references use the same forecast frames. These are geometric diagnostics, not a physical-validity certificate.

| Objective | Policy | Predicted outside fraction | Truth outside fraction | Predicted mean step max excursion | Truth mean step max excursion | Predicted mean trajectory max excursion | Truth mean trajectory max excursion |
|---|---|---:|---:|---:|---:|---:|---:|
| faithful | base | 0.134642 ± 0.06 | 0.0267772 ± 0 | 0.141842 ± 0.0959 | 0.00567105 ± 0 | 0.239254 ± 0.108 | 0.00782047 ± 0 |
| faithful | dense | 0.14878 ± 0.0529 | 0.0267772 ± 0 | 0.257587 ± 0.0851 | 0.00567105 ± 0 | 0.386763 ± 0.0807 | 0.00782047 ± 0 |
| faithful | random25 | 0.136052 ± 0.0625 | 0.0267772 ± 0 | 0.163315 ± 0.0969 | 0.00567105 ± 0 | 0.268144 ± 0.0964 | 0.00782047 ± 0 |
| faithful | speed25 | 0.132204 ± 0.06 | 0.0267772 ± 0 | 0.165641 ± 0.084 | 0.00567105 ± 0 | 0.272827 ± 0.0852 | 0.00782047 ± 0 |
| faithful | laggedrisk25 | 0.131055 ± 0.061 | 0.0267772 ± 0 | 0.192496 ± 0.106 | 0.00567105 ± 0 | 0.30158 ± 0.112 | 0.00782047 ± 0 |
| nll | base | — | — | — | — | — | — |
| nll | dense | — | — | — | — | — | — |
| nll | random25 | — | — | — | — | — | — |
| nll | speed25 | 0.117386 ± 0.0777 | 0.0267772 ± 0 | 0.363029 ± 0.273 | 0.00567105 ± 0 | 0.56502 ± 0.367 | 0.00782047 ± 0 |
| nll | laggedrisk25 | — | — | — | — | — | — |

Any failed or missing trajectory makes the corresponding all-sample full-horizon boundary means undefined, including the truth reference. The JSON retains accepted-prefix diagnostics for failed trajectories separately; these prefixes exclude the rejected attempt and are never substituted into the table or pooled across surviving trajectories.

The JSON contains every recorded trajectory outcome, explicit missing runs/trajectories, and paired policy and objective differences. Negative paired error or failure differences favor the left-hand method. No significance or speedup claim follows from this summary.
