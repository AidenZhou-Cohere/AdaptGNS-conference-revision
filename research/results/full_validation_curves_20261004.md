# Fixed-model validation audit

The [saved curves](full_validation_curves.png) and [numeric summary](full_validation_curves.json) retain every recorded scheduled diagnostic. At the October 4, 2026, 20:24 UTC snapshot, faithful seed 0 has initialization and all nine scheduled measurements through 45,000 updates; the other five models are unstarted. These are interim clean-validation diagnostics, not final model or test results.

| Update | Coordinate MSE | Binned vector-risk gap |
|---|---:|---:|
| 0 | 0.02112561 | 1.37612962 |
| 5,000 | 0.03245334 | 0.04667656 |
| 10,000 | 0.02882344 | 0.02104889 |
| 15,000 | 0.02638288 | 0.01207657 |
| 20,000 | 0.01660929 | 0.02637350 |
| 25,000 | 0.01896221 | 0.01839164 |
| 30,000 | 0.01484493 | 0.01491904 |
| 35,000 | 0.01580970 | 0.01320417 |
| 40,000 | 0.01542683 | 0.01110890 |
| 45,000 | 0.01379131 | 0.01093056 |

Error at 40k falls 2.4% from 35k and at 45k falls another 10.6%, to 0.01379131 (34.7% below initialization). The corresponding binned-gap changes are -15.9% and -1.6%. Gaussian NLL nevertheless worsens by 0.06707 at 40k before improving at 45k; every metric and earlier adverse point is retained in the scalar records. Mean accuracy, binned scale agreement and Gaussian score are separate diagnostics. They do not establish ranking usefulness, rollout accuracy or full conditional calibration, and they do not select a different final checkpoint.

The 20k validation predates a system restart. Its saved bytes were preserved when the reviewed recovery restored the 20k model/optimizer/RNG state; resumed training replayed at least 1,600 unsaved updates before the 25k validation. [Recovery evidence](training_recovery_20261004.json) retains the interrupted logs' identities and differing replay losses. Restoring state does not establish bitwise-identical Metal continuation.

The read-only summarizer verifies the declared protocol, fixed configuration, complete frozen-source file set and its current hashes. It recomputes equal-frame trajectory means and equal-trajectory means from saved records, checks coordinate/vector units, and checks the arithmetic consistency of calibration-bin counts, masses, risk totals and gap. It does not load models or data, recompute predictions, or recover particle-level bin membership from frame means. Missing protocol files cannot hide existing results as unstarted models.

An independent check of the saved 35k, 40k and 45k records passed 4,940 finite-value, identity and arithmetic checks, with no anomalies. All 128 frames, 30 trajectories and particle counts are identical across these records and match the fixed configuration. Earlier audits remain preserved. This record-level check does not independently recover predictions or calibration-bin membership.

Every seed remains visible. Objective-level means and sample SDs require all three seed measurements at the same scheduled update; available seeds are never substituted for the requested group. Failed-run histories are retained. All scheduled records are required for a run declared complete. The plotted units use the stored noise-adjusted training normalization and must not be confused with position-rollout MSE or the compact pilot's differently normalized acceleration metrics.

Verification: **30 synthetic tests passed**. They include unequal frame and particle weights, missing seeds, corrupt identities/metrics/bins, incomplete validation histories, missing provenance and incorrect source-file lists. An independent read-only agent review found two provenance gaps; both were reproduced, fixed and added to the tests. The plot was visually inspected. These checks do not replace human author verification.

From the repository root:

```sh
python -m research.summarize_training_validation \
  --output-prefix /path/outside/active/training/directories/full_validation_curves --plot
```

[Fixed protocol](full_experiment_protocol.md) · [Live continuation state](continuation_state.md)
