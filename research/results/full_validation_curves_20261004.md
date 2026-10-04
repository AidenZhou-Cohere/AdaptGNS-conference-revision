# Fixed-model validation audit

The [saved curves](full_validation_curves.png) and [numeric summary](full_validation_curves.json) retain every recorded scheduled diagnostic. The snapshot is October 4, 2026, 19:53 UTC: faithful seed 0 has initialization and all seven scheduled measurements through 35,000 updates; the other five models have no validation records. These are interim clean-validation diagnostics, not final model or test results.

The first model's normalized acceleration coordinate MSE is 0.02112561 at initialization, 0.03245334 at 5k, 0.02882344 at 10k, 0.02638288 at 15k, 0.01660929 at 20k, 0.01896221 at 25k, 0.01484493 at 30k and 0.01580970 at 35k. The 30k error falls 21.7% from 25k; 35k rises 6.5% from 30k while remaining 25.2% below initialization. Every point, including the adverse 5k and 25k observations, is retained. The binned vector-risk gap is 0.01491904 at 30k and 0.01320417 at 35k, falling 11.5% in the last interval while mean error rises. Mean accuracy and binned scale agreement are separate diagnostics; neither establishes ranking usefulness, rollout accuracy or full conditional calibration. These observations do not select a checkpoint or establish the final 100,000-update outcome.

The 20k validation predates a system restart. Its saved bytes were preserved when the reviewed recovery restored the 20k model/optimizer/RNG state; resumed training replayed at least 1,600 unsaved updates before the 25k validation. [Recovery evidence](training_recovery_20261004.json) retains the interrupted logs' identities and differing replay losses. Restoring state does not establish bitwise-identical Metal continuation.

The read-only summarizer verifies the declared protocol, fixed configuration, complete frozen-source file set and its current hashes. It recomputes equal-frame trajectory means and equal-trajectory means from saved records, checks coordinate/vector units, and checks the arithmetic consistency of calibration-bin counts, masses, risk totals and gap. It does not load models or data, recompute predictions, or recover particle-level bin membership from frame means. Missing protocol files cannot hide existing results as unstarted models.

An independent check of the saved 25k, 30k and 35k records passed 4,940 finite-value, identity and arithmetic checks, with no anomalies. All 128 frames, 30 trajectories and particle counts are identical across these records and match the fixed configuration. Earlier audits remain preserved. This record-level check does not independently recover predictions or calibration-bin membership.

Every seed remains visible. Objective-level means and sample SDs require all three seed measurements at the same scheduled update; available seeds are never substituted for the requested group. Failed-run histories are retained. All scheduled records are required for a run declared complete. The plotted units use the stored noise-adjusted training normalization and must not be confused with position-rollout MSE or the compact pilot's differently normalized acceleration metrics.

Verification: **30 synthetic tests passed**. They include unequal frame and particle weights, missing seeds, corrupt identities/metrics/bins, incomplete validation histories, missing provenance and incorrect source-file lists. An independent read-only agent review found two provenance gaps; both were reproduced, fixed and added to the tests. The plot was visually inspected. These checks do not replace human author verification.

From the repository root:

```sh
python -m research.summarize_training_validation \
  --output-prefix /path/outside/active/training/directories/full_validation_curves --plot
```

[Fixed protocol](full_experiment_protocol.md) · [Live continuation state](continuation_state.md)
