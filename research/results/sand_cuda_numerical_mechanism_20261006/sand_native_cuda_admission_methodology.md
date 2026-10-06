# Methodological assessment after strict CUDA gradient failures

Prepared October 6, 2026, after the deterministic v2 numerical report and before scientific Sand training. `sand_cuda_admission_scalar_review.json` records the source-report SHA256 and exact JSON paths behind the quantities below.

**Recommendation: assess a separately declared native-CUDA study after the bounded mechanism review; do not relabel the strict parity checks as passed.** Both failed validation reports remain scientific/engineering evidence. The Sand scientific protocol is still a draft, no scientific training has started, and raw Sand test has not been accessed. A documented protocol revision now can be prospective for the efficacy experiment, while being explicitly informed by backend diagnostics. It cannot be described as an unchanged preregistration or a successful CPU reproduction.

## What the present scalar evidence shows

This review read saved JSON only; it loaded no trajectory arrays, checkpoint or model. The deterministic v2 report still has status `validation_failed`. It compares 189 parameter tensors containing 1,624,979 gradient entries per case/objective. Two faithful tensors fail the unchanged elementwise criterion `abs(error) <= 5e-5 + 5e-4*abs(reference)`:

| Fixed training case | Tensor | Failed entries | Maximum absolute gradient difference | Tensor-relative L2 difference | Maximum tolerance ratio |
|---|---|---:|---:|---:|---:|
| Median | block 4, edge MLP NN-1 weight | 1 | 5.7948e-5 | 9.6203e-5 | 1.1293 |
| Large | block 0, node MLP NN-0 weight | 12 | 1.1638e-4 | 7.2337e-5 | 1.7991 |

All six case/objective mean-prediction relative L2 differences lie between 6.07e-7 and 7.01e-7, with maximum absolute prediction difference 1.1921e-6 in normalized-acceleration units. The largest loss difference is 2.3842e-7. Ordered graph checks, feature checks at their declared tolerance, forward/loss checks, all NLL gradient checks, deterministic within-CUDA faithful/MSE controls and the bounded checkpoint/state/RNG replay checks pass. Cross-device feature tensors need not be bitwise equal; their tolerance checks passed.

This pattern supports a localized numerical investigation. It does not itself prove harmlessness, a ReLU mechanism, exact long-run replay, or equality of 100,000-update optimization trajectories. The fraction of failed coordinates is descriptive, not a probability or significance calculation. It is especially insufficient to dismiss the issue solely by global gradient norm: Adam rescales gradients per coordinate, so a small absolute gradient disagreement can alter an update when the reference gradient is near zero.

## Useful bounded mechanism evidence

Keep the same admitted training cases, model initialization, noise, objective definitions and original tolerances. The planned ReLU sign/gradient control should identify the first differing activation masks and the magnitude/distance from zero at those locations, then compare original versus common-mask gradient discrepancies. Preserve each natural-gradient result; common-mask backpropagation is a diagnostic intervention, never a replacement training algorithm or a new passing score for the original check.

If inexpensive, also report the actual first-Adam parameter-update differences, gradient/update sign changes and prediction/loss differences after that update, using the same optimizer recipe and initialization. Give per-tensor maxima and relative norms plus affected-coordinate counts, including near-zero cases. These quantities connect the engineering discrepancy to the learning operation. A short finite/replay check on native CUDA and an end-to-end timing measurement serve different purposes and must remain separately labeled.

Do not select only a favorable case, hide a differing activation, average away a bad component or increase the old tolerance until the status turns green. A finite tiny forward difference alone cannot establish long-horizon accuracy or physical validity. Conversely, failure of a deliberately strict cross-device elementwise gradient comparison is not, by itself, a proof that native CUDA computes an invalid learned simulator.

## Decision after mechanism review

A new native-CUDA experiment is methodologically defensible if the review finds no source/data/graph/loss/gradient-detachment defect; the remaining discrepancy is localized and consistent with the measured finite-precision/nondifferentiable behavior; actual update/output diagnostics show no unexplained instability; and within-CUDA objective semantics, strict deterministic operation and bounded state/RNG/optimizer replay are supported. The evidence supports only those tested scopes. If unexplained broad differences, materially unstable updates, or failures of the within-CUDA semantics/replay remain, defer the whole cohort and preserve the evidence.

Record a **new explicit native-study admission decision** with the original cross-device gradient gate still false, the two failed report hashes, mechanism-report hash, exact selected backend/trainer/evaluator/runtime, supporting scopes, exceptions and scientific rationale. The authoring agent should state that the original proposed engineering gate was not met and explain why the new study asks a valid scientific question under a different admission basis. Freeze this decision and the complete six-model/test protocol before starting scientific training; no threshold change or automatic waiver is implied by this memo.

All six models and all policy comparisons must share the admitted deterministic CUDA recipe, final 100k endpoint and paired host schedules. Within-study objective/policy contrasts can then address Sand allocation behavior conditional on that implementation. They do not establish CPU/Metal reproduction, a material-only causal effect, universal backend equivalence or conference readiness. Historical Sand aggregates were already inspected, so the study remains an exploratory extension even though raw test stays reserved until protocol and models are fixed. The full-cohort time forecast and final author review remain independent admission requirements.
