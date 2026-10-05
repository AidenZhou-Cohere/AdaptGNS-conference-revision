# Fixed-model validation audit

[Saved curves](full_validation_curves_20261005.png) · [Numeric summary](full_validation_curves_20261005.json)

At the October 5, 01:53 UTC snapshot, faithful and NLL seed 0 have each completed the fixed 100,000 updates. Faithful seed 1 is running at 5,000 updates; three models are unstarted. The existing supervisor advanced automatically, with no duplicate training. Both final checkpoint byte hashes match their saved pointers. All 44 scheduled validation records are retained (21 per completed model, two for faithful seed 1). The six-model study and all full-model test evaluations remain incomplete.

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
| faithful seed 1 | 0 | 0.09258327 | 1.10612698 |
| faithful seed 1 | 5,000 | 0.03634267 | 0.04648564 |
| nll seed 0 | 0 | 0.02112561 | 1.37612962 |
| nll seed 0 | 5,000 | 0.03379660 | 0.03363214 |
| nll seed 0 | 10,000 | 0.03238986 | 0.00671458 |
| nll seed 0 | 15,000 | 0.02117157 | 0.01721261 |
| nll seed 0 | 20,000 | 0.01976448 | 0.01464342 |
| nll seed 0 | 25,000 | 0.01738344 | 0.01177771 |
| nll seed 0 | 30,000 | 0.01704335 | 0.00950243 |
| nll seed 0 | 35,000 | 0.01382302 | 0.01076486 |
| nll seed 0 | 40,000 | 0.01323541 | 0.00651977 |
| nll seed 0 | 45,000 | 0.01492226 | 0.00662464 |
| nll seed 0 | 50,000 | 0.01437832 | 0.00913025 |
| nll seed 0 | 55,000 | 0.01199984 | 0.00990784 |
| nll seed 0 | 60,000 | 0.01173967 | 0.00942189 |
| nll seed 0 | 65,000 | 0.01062644 | 0.00723387 |
| nll seed 0 | 70,000 | 0.01051538 | 0.00668471 |
| nll seed 0 | 75,000 | 0.01078618 | 0.00611084 |
| nll seed 0 | 80,000 | 0.00996955 | 0.00586473 |
| nll seed 0 | 85,000 | 0.01003005 | 0.00695782 |
| nll seed 0 | 90,000 | 0.00932503 | 0.00774489 |
| nll seed 0 | 95,000 | 0.00945481 | 0.00590052 |
| nll seed 0 | 100,000 | 0.00945431 | 0.00663060 |

At the fixed seed-0 endpoints, NLL coordinate MSE is **0.00945431** versus faithful **0.00811862**: NLL is **16.45% worse**. Its binned vector-risk gap is **0.00663060** versus **0.01228642**, **46.03% smaller**; constant-free Gaussian NLL is **−4.48935** versus **−4.28672**. This is one paired seed's clean normalized-acceleration validation comparison, not a three-seed estimate, conditional-calibration guarantee, policy comparison or rollout result.

Every scheduled adverse interval remains. Faithful final MSE improves 7.73% from 95k while its gap worsens 72.33% and Gaussian NLL worsens by 0.03782486. NLL final MSE changes by less than 0.01% from 95k while its gap worsens 12.37% and Gaussian NLL worsens by 0.00831541; final MSE is 1.39% above 90k. NLL's earlier 5k error of 0.03379660 remains 60.0% above its initialization. Faithful seed 1 has 5k MSE 0.03634267 and gap 0.04648564. These observations do not select an earlier checkpoint.

The 20k validation predates a system restart. Its saved bytes were preserved when the reviewed recovery restored the 20k model/optimizer/RNG state; resumed training replayed at least 1,600 unsaved updates before the 25k validation. [Recovery evidence](full_training_recovery_20261004T1853.json) retains the interrupted logs' identities and differing replay losses. Restoring state does not establish bitwise-identical Metal continuation.

The read-only summarizer verifies the declared protocol, fixed configuration, complete frozen-source file set and its current hashes. It recomputes equal-frame trajectory means and equal-trajectory means from saved records, checks coordinate/vector units, and checks the arithmetic consistency of calibration-bin counts, masses, risk totals and gap. It does not load models or data, recompute predictions, or recover particle-level bin membership from frame means. Missing protocol files cannot hide existing results as unstarted models.

The latest independent audit of NLL 90k/95k/100k and faithful seed 1 at 0/5k passes 13,165 saved-record checks without anomalies. All frozen source/configuration identities and identical validation frames across training seeds were verified. Prior NLL audits from 10k through 90k are retained, including all unfavorable intervals. The earlier 6,818-check first-completion audit and its corrected step-1 log expectation remain preserved; no experimental record was changed. These checks do not load models, reconstruct raw predictions or recover particle-level calibration-bin membership.

Every seed remains visible. Objective-level means and sample SDs require all three seed measurements at the same scheduled update; available seeds are never substituted for the requested group. Failed-run histories are retained. All scheduled records are required for a run declared complete. The plotted units use the stored noise-adjusted training normalization and must not be confused with position-rollout MSE or the compact pilot's differently normalized acceleration metrics.

Verification: **30 synthetic tests passed**. They include unequal frame and particle weights, missing seeds, corrupt identities/metrics/bins, incomplete validation histories, missing provenance and incorrect source-file lists. An independent read-only agent review found two provenance gaps; both were reproduced, fixed and added to the tests. The plot was visually inspected. These checks do not replace human author verification.

From the repository root:

```sh
python -m research.summarize_training_validation \
  --output-prefix /path/outside/active/training/directories/full_validation_curves --plot
```

[Fixed protocol](../protocols/full_waterdrop_100k.md) · [Dated status](full_training_status_20261005.json)
