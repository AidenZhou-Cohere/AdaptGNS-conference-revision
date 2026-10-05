# Fixed-model validation audit

[Saved curves](full_validation_curves_20261005.png) · [Numeric summary](full_validation_curves_20261005.json)

At the October 5, 16:03 UTC snapshot, all six faithful and corrected-NLL models (seeds 0–2) have completed exactly 100,000 updates each. Training ended at 15:58:10 UTC, with 600,000 requested updates and all 126 scheduled validation records retained. All six final checkpoint byte hashes match their saved pointers. These results complete the fixed training budget and clean-validation curves; full-rollout policy evaluation remains a separate study. The reviewed locked evaluation supervisor started at 16:05 UTC; this training snapshot contains no evaluation outcomes.

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
| faithful seed 1 | 10,000 | 0.02470587 | 0.04123915 |
| faithful seed 1 | 15,000 | 0.02165328 | 0.02471569 |
| faithful seed 1 | 20,000 | 0.02287291 | 0.00853733 |
| faithful seed 1 | 25,000 | 0.01532890 | 0.01532545 |
| faithful seed 1 | 30,000 | 0.01479255 | 0.02868758 |
| faithful seed 1 | 35,000 | 0.01266339 | 0.02466136 |
| faithful seed 1 | 40,000 | 0.01278638 | 0.01342487 |
| faithful seed 1 | 45,000 | 0.01218058 | 0.00854418 |
| faithful seed 1 | 50,000 | 0.01246814 | 0.01258838 |
| faithful seed 1 | 55,000 | 0.01135956 | 0.01256269 |
| faithful seed 1 | 60,000 | 0.01289928 | 0.00666377 |
| faithful seed 1 | 65,000 | 0.01344861 | 0.00661752 |
| faithful seed 1 | 70,000 | 0.01046347 | 0.01609606 |
| faithful seed 1 | 75,000 | 0.01019790 | 0.01317760 |
| faithful seed 1 | 80,000 | 0.00955121 | 0.01138633 |
| faithful seed 1 | 85,000 | 0.00943162 | 0.01262935 |
| faithful seed 1 | 90,000 | 0.00975465 | 0.00611907 |
| faithful seed 1 | 95,000 | 0.00881816 | 0.01087275 |
| faithful seed 1 | 100,000 | 0.00905942 | 0.00831614 |
| faithful seed 2 | 0 | 0.04784364 | 0.99512302 |
| faithful seed 2 | 5,000 | 0.02855401 | 0.03308256 |
| faithful seed 2 | 10,000 | 0.04210602 | 0.01473980 |
| faithful seed 2 | 15,000 | 0.02090587 | 0.02971389 |
| faithful seed 2 | 20,000 | 0.02161414 | 0.02653115 |
| faithful seed 2 | 25,000 | 0.02089469 | 0.01056508 |
| faithful seed 2 | 30,000 | 0.02131099 | 0.00864989 |
| faithful seed 2 | 35,000 | 0.01514568 | 0.00677632 |
| faithful seed 2 | 40,000 | 0.01470791 | 0.01423152 |
| faithful seed 2 | 45,000 | 0.01348364 | 0.00894491 |
| faithful seed 2 | 50,000 | 0.01409105 | 0.01921253 |
| faithful seed 2 | 55,000 | 0.01153141 | 0.01465916 |
| faithful seed 2 | 60,000 | 0.00981675 | 0.01969201 |
| faithful seed 2 | 65,000 | 0.01151657 | 0.01195382 |
| faithful seed 2 | 70,000 | 0.01009408 | 0.01389522 |
| faithful seed 2 | 75,000 | 0.00992735 | 0.01410624 |
| faithful seed 2 | 80,000 | 0.01057413 | 0.00705749 |
| faithful seed 2 | 85,000 | 0.00892347 | 0.00990891 |
| faithful seed 2 | 90,000 | 0.00897560 | 0.01112693 |
| faithful seed 2 | 95,000 | 0.00924206 | 0.00955363 |
| faithful seed 2 | 100,000 | 0.00875011 | 0.01110184 |
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
| nll seed 1 | 0 | 0.09258327 | 1.10612698 |
| nll seed 1 | 5,000 | 0.04016202 | 0.02630095 |
| nll seed 1 | 10,000 | 0.02784090 | 0.02092321 |
| nll seed 1 | 15,000 | 0.02088444 | 0.00799364 |
| nll seed 1 | 20,000 | 0.02058948 | 0.01327490 |
| nll seed 1 | 25,000 | 0.01720840 | 0.00223810 |
| nll seed 1 | 30,000 | 0.01618207 | 0.01790272 |
| nll seed 1 | 35,000 | 0.01691147 | 0.00296157 |
| nll seed 1 | 40,000 | 0.01396956 | 0.01064461 |
| nll seed 1 | 45,000 | 0.01284748 | 0.00445792 |
| nll seed 1 | 50,000 | 0.01204817 | 0.00713491 |
| nll seed 1 | 55,000 | 0.01140425 | 0.00951841 |
| nll seed 1 | 60,000 | 0.01159520 | 0.00517884 |
| nll seed 1 | 65,000 | 0.01332597 | 0.00745664 |
| nll seed 1 | 70,000 | 0.01128645 | 0.00903372 |
| nll seed 1 | 75,000 | 0.01009716 | 0.00949335 |
| nll seed 1 | 80,000 | 0.00986228 | 0.00969668 |
| nll seed 1 | 85,000 | 0.00983250 | 0.00656093 |
| nll seed 1 | 90,000 | 0.01126841 | 0.00592475 |
| nll seed 1 | 95,000 | 0.00940738 | 0.00862123 |
| nll seed 1 | 100,000 | 0.01003003 | 0.00484836 |
| nll seed 2 | 0 | 0.04784364 | 0.99512302 |
| nll seed 2 | 5,000 | 0.03512826 | 0.02018598 |
| nll seed 2 | 10,000 | 0.03757353 | 0.01382827 |
| nll seed 2 | 15,000 | 0.02070987 | 0.02689825 |
| nll seed 2 | 20,000 | 0.02181187 | 0.01820525 |
| nll seed 2 | 25,000 | 0.01577530 | 0.01487797 |
| nll seed 2 | 30,000 | 0.01577741 | 0.00831034 |
| nll seed 2 | 35,000 | 0.01708708 | 0.00247131 |
| nll seed 2 | 40,000 | 0.01433757 | 0.01057499 |
| nll seed 2 | 45,000 | 0.01255455 | 0.01101761 |
| nll seed 2 | 50,000 | 0.01388960 | 0.00808449 |
| nll seed 2 | 55,000 | 0.01170000 | 0.00906911 |
| nll seed 2 | 60,000 | 0.01120080 | 0.00919742 |
| nll seed 2 | 65,000 | 0.01201317 | 0.00392731 |
| nll seed 2 | 70,000 | 0.01001881 | 0.01021635 |
| nll seed 2 | 75,000 | 0.00985941 | 0.00711408 |
| nll seed 2 | 80,000 | 0.00991529 | 0.00547651 |
| nll seed 2 | 85,000 | 0.00968707 | 0.00364607 |
| nll seed 2 | 90,000 | 0.00925696 | 0.00817245 |
| nll seed 2 | 95,000 | 0.00946278 | 0.00648856 |
| nll seed 2 | 100,000 | 0.00897528 | 0.00754457 |

At the fixed seed-0 endpoints, NLL coordinate MSE is **0.00945431** versus faithful **0.00811862** (**16.45% worse**); its binned vector-risk gap is **0.00663060** versus **0.01228642** (**46.03% smaller**), and Gaussian NLL is **−4.48935** versus **−4.28672**. At seed 1, NLL coordinate MSE is **0.01003003** versus faithful **0.00905942** (**10.71% worse**); its gap is **0.00484836** versus **0.00831614** (**41.70% smaller**), and Gaussian NLL is **−4.43652** versus **−4.30024**. The seed-2 pair now shows the same validation tradeoff. These are clean normalized-acceleration diagnostics, not a conditional-calibration guarantee, policy comparison or rollout result.

Every scheduled adverse interval remains. Faithful seed 0 final MSE improves 7.73% from 95k while its gap worsens 72.33% and Gaussian NLL worsens by 0.03782486. NLL seed 0 final MSE changes by less than 0.01% from 95k while its gap worsens 12.37% and Gaussian NLL worsens by 0.00831541; final MSE is 1.39% above 90k. NLL seed 0's earlier 5k error of 0.03379660 remains 60.0% above its initialization. Faithful and NLL seed 1 final MSE are **2.74% and 6.62% worse than 95k**, respectively, despite smaller gaps and better likelihood. Their gaps increased **77.69% and 45.51% at 95k**, respectively. Earlier adverse intervals and both seed-0 endpoint reversals remain recorded. These diagnostics do not select an earlier checkpoint.

The 20k validation predates a system restart. Its saved bytes were preserved when the reviewed recovery restored the 20k model/optimizer/RNG state; resumed training replayed at least 1,600 unsaved updates before the 25k validation. [Recovery evidence](full_training_recovery_20261004T1853.json) retains the interrupted logs' identities and differing replay losses. Restoring state does not establish bitwise-identical Metal continuation.

The read-only summarizer verifies the declared protocol, fixed configuration, complete frozen-source file set and its current hashes. It recomputes equal-frame trajectory means and equal-trajectory means from saved records, checks coordinate/vector units, and checks the arithmetic consistency of calibration-bin counts, masses, risk totals and gap. It does not load models or data, recompute predictions, or recover particle-level bin membership from frame means. Missing protocol files cannot hide existing results as unstarted models.

The prior independent audit of faithful seed 2 at 80k/85k/90k/95k/100k and NLL seed 2 initialization passed 15,780 saved-record checks. Its record and all subsequent NLL-seed-2 audits remain preserved. The current unchanged read-only summarizer checks all 126 records. The independent completion audit passes 330,370 checks over all 126 records and 17 frozen files; its initial failed assumption of exactly equal paired initialization metrics is preserved, with observed differences of about 1e−9 reported in the corrected audit. No scientific data, training source or numerical verification tolerances changed. Curves were generated at 2026-10-05T16:05:58.148487+00:00; training status was separately recorded at 2026-10-05T16:03:27.535178+00:00. These checks do not load models, reconstruct predictions or recover particle-level calibration-bin membership.

All 21 scheduled updates now have exactly three seeds for both objectives. At the fixed final endpoint, faithful coordinate MSE is **0.00864272 ± 0.00047951**, binned vector-risk gap **0.01056813 ± 0.00203824**, and constant-free Gaussian NLL **−4.269077 ± 0.042800**. Corrected-NLL values are **0.00948654 ± 0.00052811**, **0.00634118 ± 0.00137120**, and **−4.490928 ± 0.055212**, respectively (three-seed mean ± sample SD). These are clean normalized-acceleration validation diagnostics; they do not establish conditional calibration, ranking usefulness, convergence, a policy effect or a rollout advantage.

The three matched seed pairs show the same validation tradeoff. Corrected-NLL coordinate MSE is **16.45% / 10.71% / 2.57% higher** than faithful for seeds 0/1/2; binned gaps are **46.03% / 41.70% / 32.04% smaller**, and all three Gaussian NLL scores are lower. These are descriptive paired differences over three training seeds, not independent trajectory-level treatment estimates. No validation checkpoint is selected. Faithful seed 2 completed at 11:46:32 UTC: final coordinate MSE **0.00875011**, binned vector-risk gap **0.01110184**, and Gaussian NLL **−4.220276**. Its final gap worsens **16.21%** from 95k despite improving MSE and likelihood. Its 85k gap increased 40.40%; at 90k MSE increased 0.58%, gap 12.29% and NLL worsened 0.039170; at 95k MSE increased 2.97% and NLL worsened 0.044306. All earlier adverse observations are retained. Every seed remains visible. Objective-level means and sample SDs require all three seed measurements at the same scheduled update; available seeds are never substituted for the requested group. Failed-run histories are retained. All scheduled records are required for a run declared complete. The plotted units use the stored noise-adjusted training normalization and must not be confused with position-rollout MSE or the compact pilot's differently normalized acceleration metrics.


NLL seed 2 ends at coordinate MSE **0.00897528**, gap **0.00754457**, and Gaussian NLL **-4.546913**. Its final gap worsens **16.27%** from 95k despite improving MSE and likelihood. At 90k the gap increases **124.14%**; at 95k MSE increases **2.22%** and Gaussian NLL worsens by **0.069203**. All earlier adverse intervals remain in the full curves and records.

Every NLL-seed-2 interval with an increase in one of the three reported diagnostics is listed below; lower values are preferred for each.

| Previous → current update | Metric that increased | Previous | Current |
|---|---|---:|---:|
| 5,000 → 10,000 | Coordinate MSE | 0.03512826 | 0.03757353 |
| 10,000 → 15,000 | Binned vector-risk gap | 0.01382827 | 0.02689825 |
| 15,000 → 20,000 | Coordinate MSE | 0.02070987 | 0.02181187 |
| 25,000 → 30,000 | Coordinate MSE | 0.01577530 | 0.01577741 |
| 30,000 → 35,000 | Coordinate MSE | 0.01577741 | 0.01708708 |
| 30,000 → 35,000 | Gaussian NLL | -3.86734741 | -3.85764213 |
| 35,000 → 40,000 | Binned vector-risk gap | 0.00247131 | 0.01057499 |
| 40,000 → 45,000 | Binned vector-risk gap | 0.01057499 | 0.01101761 |
| 45,000 → 50,000 | Coordinate MSE | 0.01255455 | 0.01388960 |
| 45,000 → 50,000 | Gaussian NLL | -4.00905279 | -4.00536002 |
| 50,000 → 55,000 | Binned vector-risk gap | 0.00808449 | 0.00906911 |
| 55,000 → 60,000 | Binned vector-risk gap | 0.00906911 | 0.00919742 |
| 60,000 → 65,000 | Coordinate MSE | 0.01120080 | 0.01201317 |
| 60,000 → 65,000 | Gaussian NLL | -4.17670644 | -4.14119063 |
| 65,000 → 70,000 | Binned vector-risk gap | 0.00392731 | 0.01021635 |
| 75,000 → 80,000 | Coordinate MSE | 0.00985941 | 0.00991529 |
| 85,000 → 90,000 | Binned vector-risk gap | 0.00364607 | 0.00817245 |
| 90,000 → 95,000 | Coordinate MSE | 0.00925696 | 0.00946278 |
| 90,000 → 95,000 | Gaussian NLL | -4.50970535 | -4.44050255 |
| 95,000 → 100,000 | Binned vector-risk gap | 0.00648856 | 0.00754457 |

Verification: **30 synthetic tests passed**. They include unequal frame and particle weights, missing seeds, corrupt identities/metrics/bins, incomplete validation histories, missing provenance and incorrect source-file lists. An independent read-only agent review found two provenance gaps; both were reproduced, fixed and added to the tests. Earlier versions of the plot were visually inspected; the regenerated completion plot has a separate current visual check. These checks do not replace human author verification.

From the repository root:

```sh
python -m research.summarize_training_validation \
  --output-prefix /path/outside/active/training/directories/full_validation_curves --plot
```

[Fixed protocol](../protocols/full_waterdrop_100k.md) · [Dated status](full_training_status_20261005.json)
