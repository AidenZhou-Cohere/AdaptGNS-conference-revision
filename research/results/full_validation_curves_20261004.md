# Fixed-model validation audit

The [saved curves](full_validation_curves.png) and [numeric summary](full_validation_curves.json) retain every recorded scheduled diagnostic. At the October 4, 2026, 22:24 UTC snapshot, faithful seed 0 has completed all 100,000 updates and all 21 scheduled validation records. NLL seed 0 has initialization and 5k validation, with 7,000 recorded training updates. Four other models remain unstarted. These are clean-validation diagnostics; the six-model study and every full-model test evaluation remain incomplete.

| Model | Update | Coordinate MSE | Binned vector-risk gap |
|---|---:|---:|---:|
| faithful seed 0 | 0 | 0.02112561 | 1.37612962 |
| faithful seed 0 | 5,000 | 0.03245334 | 0.04667656 |
| faithful seed 0 | 10,000 | 0.02882344 | 0.02104889 |
| faithful seed 0 | 15,000 | 0.02638288 | 0.01207657 |
| faithful seed 0 | 20,000 | 0.01660929 | 0.02637350 |
| faithful seed 0 | 25,000 | 0.01896221 | 0.01839164 |
| faithful seed 0 | 30,000 | 0.01484493 | 0.01491904 |
| faithful seed 0 | 35,000 | 0.01580970 | 0.01320417 |
| faithful seed 0 | 40,000 | 0.01542683 | 0.01110890 |
| faithful seed 0 | 45,000 | 0.01379131 | 0.01093056 |
| faithful seed 0 | 50,000 | 0.01158499 | 0.01146908 |
| faithful seed 0 | 55,000 | 0.01150961 | 0.01582046 |
| faithful seed 0 | 60,000 | 0.01038780 | 0.01284647 |
| faithful seed 0 | 65,000 | 0.01025963 | 0.01380523 |
| faithful seed 0 | 70,000 | 0.01054690 | 0.00997069 |
| faithful seed 0 | 75,000 | 0.01077674 | 0.00834635 |
| faithful seed 0 | 80,000 | 0.00972925 | 0.00724032 |
| faithful seed 0 | 85,000 | 0.00899191 | 0.01059303 |
| faithful seed 0 | 90,000 | 0.00891186 | 0.01155909 |
| faithful seed 0 | 95,000 | 0.00879904 | 0.00712952 |
| faithful seed 0 | 100,000 | 0.00811862 | 0.01228642 |
| nll seed 0 | 0 | 0.02112561 | 1.37612962 |
| nll seed 0 | 5,000 | 0.03379660 | 0.03363214 |

Faithful seed 0 ends with coordinate MSE 0.00811862, 7.73% below 95k; the binned risk gap instead rises 72.33% to 0.01228642 and Gaussian NLL worsens by 0.03782486. NLL seed 0 at 5k has MSE 0.03379660, 59.98% above initialization. Earlier adverse faithful error intervals at 5k, 25k, 35k, 70k and 75k, and the risk-gap increases at 55k, 85k and 90k, remain in the complete records. These different diagnostic directions do not establish calibration, convergence, rollout accuracy or useful allocation. The final faithful checkpoint is the predeclared 100k model; no earlier diagnostic selected it.

The 20k validation predates a system restart. Its saved bytes were preserved when the reviewed recovery restored the 20k model/optimizer/RNG state; resumed training replayed at least 1,600 unsaved updates before the 25k validation. [Recovery evidence](training_recovery_20261004.json) retains the interrupted logs' identities and differing replay losses. Restoring state does not establish bitwise-identical Metal continuation.

The read-only summarizer verifies the declared protocol, fixed configuration, complete frozen-source file set and its current hashes. It recomputes equal-frame trajectory means and equal-trajectory means from saved records, checks coordinate/vector units, and checks the arithmetic consistency of calibration-bin counts, masses, risk totals and gap. It does not load models or data, recompute predictions, or recover particle-level bin membership from frame means. Missing protocol files cannot hide existing results as unstarted models.

The latest independent audit of faithful 95k/100k and NLL 0/5k passed 6,818 checks with no experimental anomaly. It also verified all 21 final faithful validation filenames/hashes, the shared 128-frame/30-trajectory configuration, and 51 corresponding logged minibatches through 5k. Paired configurations differ only by objective; frame IDs, learning rates and particle counts match. This does not certify bitwise-identical optimization. An initial audit expectation omitted the intentional step-1 log; that audit version is preserved and its expectation corrected from frozen source. Earlier 65k–75k and 80k–95k independent audits passed 6,619 and 8,265 checks and remain archived. These checks neither reconstruct raw predictions nor recover particle-level calibration-bin membership.

Every seed remains visible. Objective-level means and sample SDs require all three seed measurements at the same scheduled update; available seeds are never substituted for the requested group. Failed-run histories are retained. All scheduled records are required for a run declared complete. The plotted units use the stored noise-adjusted training normalization and must not be confused with position-rollout MSE or the compact pilot's differently normalized acceleration metrics.

Verification: **30 synthetic tests passed**. They include unequal frame and particle weights, missing seeds, corrupt identities/metrics/bins, incomplete validation histories, missing provenance and incorrect source-file lists. An independent read-only agent review found two provenance gaps; both were reproduced, fixed and added to the tests. The plot was visually inspected. These checks do not replace human author verification.

From the repository root:

```sh
python -m research.summarize_training_validation \
  --output-prefix /path/outside/active/training/directories/full_validation_curves --plot
```

[Fixed protocol](full_experiment_protocol.md) · [Live continuation state](continuation_state.md)
