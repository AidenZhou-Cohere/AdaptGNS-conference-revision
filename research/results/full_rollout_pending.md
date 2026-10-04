# Full WaterDrop rollout results

**Results pending. No rollout outcomes are present; no experimental outcomes are reported.**

Completed evaluator runs: 0/6; eligible for aggregation: 0/6. Required: faithful and NLL, seeds 0–2, five policies, official test source indices 3–29, 995 forecast steps.

Errors are averaged equally over 27 trajectories within each seed. Tables show the mean ± sample SD of exactly three seed means. Missing values are undefined or pending; failed trajectories are never dropped.

| Model | State | Recorded policy trajectories |
|---|---|---:|
| faithful seed 0 | missing | 0/135 |
| faithful seed 1 | missing | 0/135 |
| faithful seed 2 | missing | 0/135 |
| nll seed 0 | missing | 0/135 |
| nll seed 1 | missing | 0/135 |
| nll seed 2 | missing | 0/135 |

| Objective | Policy | Mean rollout MSE | MSE@200 | MSE@995 | Failure fraction |
|---|---|---:|---:|---:|---:|
| faithful | base | — | — | — | — |
| faithful | dense | — | — | — | — |
| faithful | random25 | — | — | — | — |
| faithful | speed25 | — | — | — | — |
| faithful | laggedrisk25 | — | — | — | — |
| nll | base | — | — | — | — |
| nll | dense | — | — | — | — |
| nll | random25 | — | — | — | — |
| nll | speed25 | — | — | — | — |
| nll | laggedrisk25 | — | — | — | — |

Boundary diagnostics use the fraction of particles outside the metadata bounds by more than 1e-6. Fractions and per-step maxima are averaged equally over forecast steps within each trajectory, then equally over trajectories within each seed. The trajectory maximum is the worst excursion in one trajectory; its table column averages those maxima across trajectories, not the single worst particle from the entire experiment. Values are mean ± sample SD across all three seeds. Ground-truth references use the same forecast frames. These are geometric diagnostics, not a physical-validity certificate.

| Objective | Policy | Predicted outside fraction | Truth outside fraction | Predicted mean step max excursion | Truth mean step max excursion | Predicted mean trajectory max excursion | Truth mean trajectory max excursion |
|---|---|---:|---:|---:|---:|---:|---:|
| faithful | base | — | — | — | — | — | — |
| faithful | dense | — | — | — | — | — | — |
| faithful | random25 | — | — | — | — | — | — |
| faithful | speed25 | — | — | — | — | — | — |
| faithful | laggedrisk25 | — | — | — | — | — | — |
| nll | base | — | — | — | — | — | — |
| nll | dense | — | — | — | — | — | — |
| nll | random25 | — | — | — | — | — | — |
| nll | speed25 | — | — | — | — | — | — |
| nll | laggedrisk25 | — | — | — | — | — | — |

Any failed or missing trajectory makes the corresponding all-sample full-horizon boundary means undefined, including the truth reference. The JSON retains accepted-prefix diagnostics for failed trajectories separately; these prefixes exclude the rejected attempt and are never substituted into the table or pooled across surviving trajectories.

The JSON contains every recorded trajectory outcome, explicit missing runs/trajectories, and paired policy and objective differences. Negative paired error or failure differences favor the left-hand method. No significance or speedup claim follows from this summary.
