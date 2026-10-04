# Fixed-model validation audit

The [saved curves](full_validation_curves_20261004.png) and [numeric summary](full_validation_curves_20261004.json) retain every recorded scheduled diagnostic. The current snapshot is October 4, 2026, 18:56 UTC: faithful seed 0 has initialization and all four scheduled measurements through 20,000 updates; the other five models have no validation records. These are interim clean-validation diagnostics, not final model or test results.

The first model's normalized acceleration coordinate MSE changes from 0.02112561 at initialization to 0.03245334 at 5,000 updates, 0.02882344 at 10,000, 0.02638288 at 15,000 and 0.01660929 at 20,000. The latest error is 37.0% below 15,000 updates and 21.4% below initialization. The binned vector-risk gap changes from 1.37613 to 0.04668, 0.02105, 0.01208 and 0.02637: it worsens at 20k while mean error improves. Mean accuracy and binned scale agreement are separate diagnostics; neither establishes ranking usefulness, rollout accuracy or full conditional calibration. These observations do not select a checkpoint or establish the final 100,000-update outcome.

The 20k validation predates a system restart. Its saved bytes were preserved when the reviewed recovery restored the 20k model/optimizer/RNG state; later training replays at least 1,600 unsaved updates. [Recovery evidence](full_training_recovery_20261004T1853.json) retains the interrupted logs' identities and differing replay losses. Restoring state does not establish bitwise-identical Metal continuation.

The read-only summarizer verifies the declared protocol, fixed configuration, complete frozen-source file set and its current hashes. It recomputes equal-frame trajectory means and equal-trajectory means from saved records, checks coordinate/vector units, and checks the arithmetic consistency of calibration-bin counts, masses, risk totals and gap. It does not load models or data, recompute predictions, or recover particle-level bin membership from frame means. Missing protocol files cannot hide existing results as unstarted models.

Every seed remains visible. Objective-level means and sample SDs require all three seed measurements at the same scheduled update; available seeds are never substituted for the requested group. Failed-run histories are retained. All scheduled records are required for a run declared complete. The plotted units use the stored noise-adjusted training normalization and must not be confused with position-rollout MSE or the compact pilot's differently normalized acceleration metrics.

Verification: **30 synthetic tests passed**. They include unequal frame and particle weights, missing seeds, corrupt identities/metrics/bins, incomplete validation histories, missing provenance and incorrect source-file lists. An independent read-only agent review found two provenance gaps; both were reproduced, fixed and added to the tests. The plot was visually inspected. These checks do not replace human author verification.

From the repository root:

```sh
python -m research.summarize_training_validation \
  --output-prefix /path/outside/active/training/directories/full_validation_curves --plot
```

[Fixed protocol](../protocols/full_waterdrop_100k.md) · [Dated progress](full_training_status_20261004.json)
