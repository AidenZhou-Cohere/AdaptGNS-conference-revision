# Adaptive Interaction Graphs: correctness and controlled experiments


Current conference revision (October 6): the [manuscript](research/results/sand_graph_support_100k_20261006/presentation/revised_manuscript.tex) studies how training exposure to additional interactions differs from placing those interactions using residual risk. Complete paired comparisons now cover [Goop2D100k](research/results/goop2d_graph_exposure_100k_20261006/README.md), [WaterDrop110k continuation](research/results/waterdrop_continuation_110k_20261006/README.md), and [Sand100k](research/results/sand_graph_support_100k_20261006/README.md). Fixed-random autonomous error improves in every paired Goop and WaterDrop seed, and in two of three Sand seeds. Sand's native base policy still beats random expansion in every seed under both training arms.

Graph exposure and risk placement remain distinct findings. On Sand, exposure narrows the observed-history risk deficit but worsens autonomous cached-risk error in every seed. All 1,080 Sand autonomous outcomes and all six policies are retained, alongside adverse physical drift and descriptive cost increases. These complete controls support a conditional benefit from graph exposure without establishing general adaptive accuracy or efficiency superiority.

The current paper includes the fixed cross-material comparison and compiles within its eight-page main-text assertion. The [21-page evidence report](research/results/sand_report_addendum_20261006/revision_report_candidate.pdf) adds a concise Sand summary while preserving all 20 earlier pages. Title/abstract comparison and exported-PDF inspection remain outstanding. Goop3D's original analysis window ended incomplete; a separate, fully scoped observed-history analysis is running with all original outcome states retained and no autonomous numerical aggregation. It supplies no admitted accuracy result yet. Earlier milestones below are historical.

## Earlier milestones (historical)

The [DesignSafe Sand arrays](research/results/sand_data_admission_20261006/README.md) are now admitted for data compatibility:1,000training/30validation trajectories,T320,type6,with lossless numeric repackaging and full source/member checks. No test data was fetched. The [two actual-data CUDA comparisons](research/results/sand_cuda_validation_20261006/README.md) retain their **failed strict gradient-parity outcomes**; deterministic execution fixes repeated-CUDA gradient discrepancies but a localized CPU/CUDA difference still needs mechanism review. No scientific Sand model or timing benchmark has launched. These preparation records remain separate from all completed experiments.

## October 5: exploratory graph-convention follow-up

The [six-model graph bridge](research/results/graph_convention_bridge_20261005/README.md) completed40,800 cases across2,550 observed histories, with no failures. Restoring trained self-messages reduces baseline test error by17.85±0.75% (faithful) and17.21±3.01% (NLL), but risk still loses to random in every seed on validation and test. These are paired within-seed reductions, with sample SD across three seeds. Cap128 is inactive on the observed histories; all native parity gates pass. The [figure and tables](research/results/graph_convention_bridge_analysis_20261005/README.md) reproduce from the hash-pinned compact publication.

The earlier [saved-action analysis](research/results/full_action_benefit_20261005/README.md) also retains weak risk/actual-action rank association, nonmonotone high-risk structure and a nondeployable whole-frame abstention reference. These separate exploratory findings refine the mechanism question without replacing the original outcomes below. Native autonomous controls are running; an equal-update paired faithful graph-exposure continuation is prepared. All failed/negative evidence is retained.

## October 5: locked full-architecture evaluation complete

All six fixed 100,000-update models and all twelve evaluation jobs completed. The final queue status is **October 5, 20:22:35 UTC**. The experiment covers **810 autonomous trajectory-policy outcomes** (27 test trajectories × five policies × six models) and **1,782 same-state frames**. Official test indices are **3–29**; prior inspection of indices 0–2 and historical aggregates prevents a claim of pristine independent confirmation.

**The results do not support general cached-risk accuracy–runtime superiority.** All 405 faithful outcomes completed. Eight NLL seed-2 outcomes hit the coordinate guard: base 1, dense 2, random 4, cached risk 1, speed 0. Affected all-sample full-horizon MSE, edge and boundary means remain undefined. All failures and accepted-prefix diagnostics are retained without retries.

Faithful cached risk versus base, random, speed and dense has paired full-rollout MSE differences **+0.001555 ± 0.002630**, **+0.000827 ± 0.002250**, **+0.003008 ± 0.003151**, and **−0.010031 ± 0.004290**, respectively. Within-seed percentage changes average **+12.19 ± 12.92%**, **+8.76 ± 12.22%**, **+10.09 ± 5.12%**, and **−28.99 ± 17.16%**. Differences against base/random change sign; speed wins and dense loses in all three faithful seeds. These are descriptive three-seed means and sample SDs, not significance claims.

At identical observed histories, previous-base risk has mean frame correlations of **0.357 ± 0.060 / 0.411 ± 0.023** with base residuals (faithful/NLL), versus **−0.022 ± 0.045 / −0.048 ± 0.012** with signed dense benefit. Random allocation beats previous-observed risk in all six seed-model means. This teacher-forced diagnostic is distinct from autonomous cached risk. The natural-base costs are **10.740 / 10.496 ms**, random costs **11.801 / 11.555 ms**, and previous-observed-risk costs **22.382 / 21.881 ms**, including its extra scoring pass. Every policy executes the risk head; fixed-order autonomous durations and different geometries do not establish speedup.

Computational completion does not certify physical validity. Faithful cached risk has **13.11 ± 6.10%** outside particles versus **2.68%** for same-frame truth. Its mean trajectory-maximum excursion is **0.3016**, compared with **0.2393** for base and **0.00782** for truth. Lower outside fractions can coexist with worse excursions. Conservation, peak memory and optimized mean-only performance remain unmeasured.

The final evaluation/summary operational suite passes **143 tests**. Public table generation reproduces both manuscript inserts byte-for-byte without inference.

Independent audits pass **92,129 rollout arithmetic/aggregation checks**, **1,068,869 same-state array/aggregation checks**, and **20,928 provenance/integrity checks**. All 761 previously hashed row files and 41 earlier checkpoint-verification records/timestamps are preserved. One NLL seed-2 observed-history case (source 6, target 106) produced two risk-selected pair hashes over six timed calls; maximum recorded prediction difference **2.45e−5**, exact budgets intact. All repetitions and the initial audit-only warmup-null handling failure remain recorded.

The local manuscript includes the full results and passes native compilation with the eight-page main-text assertion. The evidence report includes all endpoints, failures, boundary references and common-state costs. Author verification, registered-abstract comparison, exported manuscript PDF inspection and anonymous submission/asset-rights checks remain required. No submission or external message has been sent.

[Compact original records and provenance](research/results/full_evaluation_20261005/README.md) · [Independent analysis](research/results/full_evaluation_analysis_20261005/README.md)


This research fork repairs material inconsistencies in the released AdaptGNS implementation and adds reproducible tests of edge allocation. **The original strict Pareto and long-horizon-superiority claims are not established by the released arrays.** See the statistical audit and the clearly separated new WaterDrop pilot. Corrected code does not retroactively validate old checkpoints or numbers.

Based on `aidenzhou8/AdaptGNS` commit `5cf0d7cb9d927f4277c4538917ae570d8dc9d7b6`, itself derived from [geoelements/gns](https://github.com/geoelements/gns). The original release description is preserved in `README_legacy_release.md` as historical context; its commands and claims should not be treated as the corrected protocol.

## What changed

- Correct d-dimensional Gaussian NLL with explicit variance semantics; `legacy_nll` remains available for forensic reproduction.
- `mse` and established faithful heteroscedastic regression objectives, with tests verifying that the latter preserves mean-network MSE gradients. Faithful regression is prior work (Stirn et al., AISTATS 2023), not a new loss.
- Checkpoints retain architecture, trained normalization, seed/configuration, uncertainty interpretation, and graph backend. Validation uses its own split without silently falling back to test.
- Batched adaptive selection is trajectory-local. Calibration binning, dynamic-particle masking, checkpoint compatibility, and saved update counts are tested.
- Explicit SciPy CPU radius backend for portable execution. Its capped-neighbor selection can differ from PyG; CPU results are not GPU performance measurements.
- Exact optional-pair budgets with random/speed/risk controls, historical whole-trajectory bootstraps, and local pilot experiments with immutable frame selection and five training seeds.

## Environment

The local work used Python 3.12, PyTorch with native Apple Metal support, SciPy and PyG. Compact pilot experiments used CPU. `research/requirements-local-lock.txt` records the exact installed packages. For a fresh CPU environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy torch torch-geometric absl-py matplotlib pytest tfrecord crc32c protobuf
export PYTHONPATH="$PWD:$PWD/adaptive-gns"
python -m pytest tests research/tests -q
```

CUDA needs a matching Torch build and an explicitly chosen graph backend. The `scipy_host` backend performs CPU neighbor search and transfers edges on both Metal and CUDA; it does not require a compiled PyG radius operator. A separate [GB200 execution validation](research/results/cuda_execution_validation_20261006/README.md) now passes bounded full-model CPU/CUDA graph, loss, gradient and optimizer/RNG checks, with all four GPUs passing tiny environment smokes. It records its own ARM64/CUDA package lock and an image driver-library correction. CUDA replay is numerically close rather than bitwise exact; real-data CUDA training and autonomous evaluation remain to be validated. Distributed training and the inherited full upstream test suite were not validated by these checks. Do not treat the macOS lock file as a portable CUDA installation recipe.

## Corrected full model

The original NPZ data format is documented in `adaptive-gns/README.md`. Supply separate `train.npz`, `valid.npz`, and `test.npz` with `metadata.json`. No original full-model checkpoints or complete benchmark datasets are bundled.

```bash
python -m gns.train --mode=train --data_path=WaterDrop/dataset/ \
  --model_path=WaterDrop/models/faithful_seed0/ \
  --loss=faithful --seed=0 --radius_backend=scipy \
  --ntraining_steps=500000 --nsave_steps=5000 --lr_decay_steps=500000

python adaptive-gns/scripts/evaluate_rollout_mse.py \
  --data_path=WaterDrop/dataset/ --model_path=WaterDrop/models/faithful_seed0/ \
  --model_file=latest --split=valid

python adaptive-gns/scripts/evaluate_adaptive_rollout.py \
  --data_path=WaterDrop/dataset/ --model_path=WaterDrop/models/faithful_seed0/ \
  --model_file=latest --split=valid --sigma_percentile=70 --radius_factor=1.267

python adaptive-gns/scripts/evaluate_budget_policies.py \
  --data_path=WaterDrop/dataset/ \
  --checkpoint=WaterDrop/models/faithful_seed0/model-500000.pt \
  --split=valid --extra_fraction=.25 --max_steps=50 \
  --output=WaterDrop/models/faithful_seed0/budget_valid.json
```

The last evaluator is a same-observed-state diagnostic, not an autonomous rollout. Its uncapped symmetric pair construction differs from the capped original graph. Old checkpoints require their original architecture and normalization settings; defaults are documented assumptions, not recovered facts. `--normalization_noise_std=0` reproduces the old normalization mismatch and is not the recommended trained-model evaluation.

The 500,000-step command is a **future full experiment**, not a run completed by this revision. A separate two-update, original-architecture CLI smoke test verified real training, checkpoint restoration, validation, fixed/adaptive rollouts, exact-budget evaluation, and calibration. Its numerical losses are not performance evidence.

## Reproduce the new small WaterDrop pilot

This is a 48-wide, three-block CPU model with no training noise. It uses 8 training, 3 validation and 3 test trajectories, 3,000 updates, and 5 seeds per objective. It does not replace the full benchmark. Data are fetched from the [official GNS release](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate).

```bash
mkdir -p data-pilot
curl -L --fail --range 0-67108863 \
  https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterDrop/train.tfrecord \
  -o data-pilot/train-prefix.tfrecord
curl -L --fail --range 0-16777215 \
  https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterDrop/valid.tfrecord \
  -o data-pilot/valid-prefix.tfrecord
curl -L --fail --range 0-16777215 \
  https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterDrop/test.tfrecord \
  -o data-pilot/test-prefix.tfrecord
curl -L --fail \
  https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterDrop/metadata.json \
  -o data-pilot/metadata.json
python research/prepare_waterdrop.py --data-dir data-pilot \
  --manifest research/results/data_manifest.json
python -m research.pilot --data-dir data-pilot \
  --output-dir research/results/waterdrop_pilot --steps 3000 --seeds 0 1 2 3 4
python -m research.summarize_pilot --data-dir data-pilot
python -m research.risk_benefit --data-dir data-pilot
```

The parser verifies TFRecord CRCs, takes complete records only, and stores numeric arrays without pickle. The training protocol fixes frames, graph schedules, budgets and final-checkpoint selection before the full run. Risk and random selectors have exactly equal retained pair counts at each observed state. The previous-base score diagnostic is distinct from the actual cached-score controller in an autonomous rollout.

## Reanalyze historical results and graph cost

```bash
python research/reanalyze.py
python research/plot_reanalysis.py
python research/benchmark_graph.py --data data-pilot/test-pilot.npz
```

Historical bootstrap intervals resample whole trajectories, condition on single saved models and assume aligned row ordering. They are not multi-seed confidence intervals. The CPU graph benchmark includes tree construction and selection on identical states; it excludes any neural model and cannot establish end-to-end speedup. Its synthetic scoring function is not learned risk.

## Scientific scope

Residual magnitude, calibrated variance, and benefit from adding edges are different targets. The theory gives conditional bookkeeping and perturbation bounds, not universal stability guarantees. Mandatory base edges may grow with density, so an optional-edge budget is not a hard total-compute bound. A deterministic ID tie-break is not permutation equivariant at ties.

Keep historical artifacts, new small-model measurements, and software smoke tests separate. Full corrected multi-seed Sand/WaterDrop experiments and broader regimes remain necessary for a general conference-level performance claim.

## Autonomous pilot and benefit-head extensions

```bash
python -m research.benefit_head --data-dir data-pilot
python -m research.pilot_rollout --help
python -m research.summarize_rollouts \
  --result-root research/results --output research/results/rollout_summary.json \
  --report pilot_rollout_results.md
python -m research.plot_pilot_rollouts
```

All 15 compact models were evaluated under five policies on three test trajectories at both 200 and 995 forecast steps. Every 200-step run completed; 66/225 long runs crossed the predeclared coordinate guard. Full-horizon accuracy is undefined for any group containing a failure. The signed-benefit ridge extension uses training labels only, but was designed after inspecting the pilot test results and remains exploratory. Its complete coefficients and measurements are saved under `research/results/`.

## Original architecture on local Metal

Training and the evaluation scripts accept `--device=mps --radius_backend=scipy_host`. Set `PYTORCH_ENABLE_MPS_FALLBACK=0` before Python starts. The host backend preserves its deterministic strict-radius/nearest-neighbor-cap rules and reports graph-transfer provenance; it is not an optimized native GPU neighbor search. The original CLI completed a 100-update CPU/Metal smoke check with checkpoint restoration and short validation rollouts. Forward parity is close, but strict gradient and longer optimization-path identity are not certified.

`research/benchmark_devices.py` records the hardware feasibility measurements; `research/smoke_metal_cli.py` reproduces the original-CLI check. A sandbox may hide the local Metal device even when it is available in a normal local terminal.

## Bounded full-data training extension

The [fixed protocol](research/protocols/full_waterdrop_100k.md) uses the original 128-wide, ten-block architecture, all 1,000 training trajectories, three paired seeds and two objectives. Its 100,000-update target is shorter than the historical 500,000-update runs. The completed compact pilot must not be presented as this larger experiment.

At the October 5, 16:03 UTC snapshot, all six faithful/NLL models at seeds 0–2 have completed exactly 100,000 updates: 600,000 total. All six final checkpoint hashes and all 126 scheduled validation records are verified. Native process checks confirmed training had exited. The reviewed supervisor launched the locked test evaluation at 16:05 UTC after model/provenance and test checksum checks. The final rollout and same-state results are now complete; see the completion section above. The [complete validation curves](research/results/full_validation_curves_20261005.md) preserve every observation. At the fixed seed-0 endpoints, NLL coordinate MSE is **0.00945431** versus faithful **0.00811862** (**16.45% worse**); its binned vector-risk gap is **0.00663060** versus **0.01228642** (**46.03% smaller**), and Gaussian NLL is **−4.48935** versus **−4.28672**. At seed 1, NLL coordinate MSE is **0.01003003** versus faithful **0.00905942** (**10.71% worse**); its gap is **0.00484836** versus **0.00831614** (**41.70% smaller**), and Gaussian NLL is **−4.43652** versus **−4.30024**. The first two seed pairs show the same validation tradeoff as seed 2. These are clean normalized-acceleration diagnostics, a paired three-seed validation comparison, not a conditional-calibration guarantee, policy comparison or rollout result. Faithful and NLL seed 1 final MSE are **2.74% and 6.62% worse than 95k**, respectively, despite smaller gaps and better likelihood. Their gaps increased **77.69% and 45.51% at 95k**, respectively. Earlier adverse intervals and both seed-0 endpoint reversals remain recorded. These diagnostics do not select an earlier checkpoint. The [second-completion audit](research/results/second_model_completion_audit_20261005.json) and [verification ledger](research/results/second_model_verification_20261005.json) document this limited scope, full logged schedule pairing and distinct time bookkeeping. The [third-completion audit](research/results/third_model_completion_audit_20261005.json) and [third-completion ledger](research/results/third_model_verification_20261005.json) extend the evidence. The [fourth-completion audit](research/results/fourth_model_completion_audit_20261005.json) and [fourth-completion ledger](research/results/fourth_model_verification_20261005.json) document the second paired endpoint. Both objectives now have three-seed summaries at every scheduled validation update. At 100k, faithful MSE is **0.00864272 ± 0.00047951**, gap **0.01056813 ± 0.00203824**, and Gaussian NLL **−4.269077 ± 0.042800**. NLL values are **0.00948654 ± 0.00052811**, **0.00634118 ± 0.00137120**, and **−4.490928 ± 0.055212**, respectively (mean ± sample SD across three seeds). Paired NLL-minus-faithful differences are **+0.00084382 ± 0.00056601** in MSE, **−0.00422696 ± 0.00123824** in gap, and **−0.221851 ± 0.096620** in NLL. NLL has higher MSE and smaller gaps in all three pairs. Within-seed percentages average **+9.91 ± 6.97%** in MSE and **−39.92 ± 7.16%** in gap; these are not ratios of group means or significance claims. Seed 2 NLL ends at MSE **0.00897528**, gap **0.00754457**, NLL **−4.546913**: MSE is **2.57% higher** and gap **32.04% smaller** than faithful. Its final gap worsens **16.27%** from 95k despite improving MSE/NLL; its 90k gap increased **124.14%**, and 95k MSE increased **2.22%** with NLL worsening **0.069203**. All earlier adverse intervals remain retained. Faithful seed 2 completed at 11:46:32 UTC: final coordinate MSE **0.00875011**, binned vector-risk gap **0.01110184**, and Gaussian NLL **−4.220276**. Its final gap worsens **16.21%** from 95k despite improving MSE and likelihood. Its 85k gap increased 40.40%; at 90k MSE increased 0.58%, gap 12.29% and NLL worsened 0.039170; at 95k MSE increased 2.97% and NLL worsened 0.044306. All earlier adverse observations are retained. The [fifth-completion audit](research/results/fifth_model_completion_audit_20261005.json) and [fifth-completion ledger](research/results/fifth_model_verification_20261005.json) document the first complete objective-level three-seed validation endpoint. The earlier faithful recovery and all historical evidence remain preserved.

Download complete official `train.tfrecord`, `valid.tfrecord`, and `metadata.json` to a data directory. The conversion streams and verifies every CRC, retains trajectory IDs, verifies content uniqueness across splits, and publishes a numeric memory-mapped manifest only when complete. It defaults to converting train and validation; test evaluation remains a separate locked step.

```bash
python -m research.prepare_full_waterdrop \
  --input-dir data-full/raw --output-dir data-full/converted \
  --metadata data-full/raw/metadata.json --expected-train-bytes 4541246980

PYTORCH_ENABLE_MPS_FALLBACK=0 python -m research.full_training \
  --train-manifest data-full/converted/train.json \
  --valid-manifest data-full/converted/valid.json \
  --metadata data-full/converted/metadata.json \
  --protocol research/protocols/full_waterdrop_100k.md \
  --output-dir research/results/full_waterdrop_100k/faithful_seed0 \
  --objective faithful --seed 0 --device mps
```

`full_training` derives a paired frame/noise schedule from seed and update index, uses clean fixed validation frames with stored training normalization, records source/data/software hashes, and atomically saves model and Adam state. `--resume` requires identical configuration and hashes. `--clear-stale-lock` verifies a previous PID is dead before taking ownership. `--stop-after` pauses at an optimizer boundary without changing the target budget; nonstandard `--steps` and other protocol overrides are explicitly labeled.

`research/run_full_queue.py` executes the six fixed jobs sequentially, stops on a numerical failure, verifies completed checkpoint hashes, and enforces an explicit UTC deadline. Its default deadline is specific to this October 2026 revision; set `--deadline-utc` for later reproduction. It never evaluates the test split. Live long-run output is kept local and gitignored until curated final results are published.

`research/monitor_training.py` reads progress without changing the run. It verifies the saved configuration and frozen source hashes, separates PID presence from command-identity verification, and reports conditional throughput projections. A projection is not a promised completion time or permission to change the fixed budget.

```bash
python -m research.monitor_training --output training_progress.json
python -m research.summarize_full_rollouts \
  --evaluation-root research/results/full_waterdrop_rollouts \
  --output-prefix research/results/full_rollout_summary
```

The rollout summarizer reads only evaluator artifacts. It verifies protocol, data/checkpoint provenance and record/trace hashes, recomputes metrics from per-step records, and gives equal-trajectory seed means followed by the three-seed mean and sample SD. Boundary excursions include ground-truth references; failed prefixes are separate from undefined full-horizon means. Missing models, interrupted evaluations and failed trajectories remain explicit. `full_rollout_pending.json` is a pending-only schema demonstration, not an experiment result.

`research/full_same_state.py` implements the [separate companion diagnostic](research/protocols/full_same_state_diagnostic.md): identical observed histories, exact pair budgets, signed dense-intervention benefit, tie/overlap audits and repeated synchronized runtime. Its previous-observed-base score is a teacher-forced lag diagnostic, not the autonomous cached-own-graph controller. Natural base-radius timing is separated from the expanded-superset reference overhead. No new full-model test outcomes have been collected.

```bash
python -m research.full_same_state --help
python -m research.summarize_full_same_state \
  --evaluation-root research/results/full_waterdrop_same_state \
  --output-prefix research/results/full_same_state_summary
python -m research.noise_target_toy
```

The [noise-target derivation](research/notes/noise_augmentation_analysis.md) and [tie-symmetry analysis](research/notes/tie_symmetry_analysis.md) state their assumptions and limits. The Gaussian toy uses synthetic draws only; it does not estimate WaterDrop's augmentation effect.

The same-state summarizer validates all six final-model identities and 297 prescribed frames per model, keeps natural/shared-base costs and current/previous risk correlations separate, and requires exactly three seed means. Its `full_same_state_pending` artifacts contain no experimental outcomes. Latest verification: 161 tests passed with one sandbox-only MPS skip in the combined suite; the subsequent standalone same-state summarizer passed 29 additional focused tests. Native Metal integration was tested separately before the fixed training queue launched.

`research/summarize_training_validation.py` reads every saved scheduled clean-validation point without loading models or data. It checks configuration and current frozen-source hashes, recomputes trajectory-weighted metrics, and checks calibration-bin arithmetic. Bin membership cannot be recovered from saved frame means. Group curves require all three seeds; failed histories and missing measurements remain explicit. Its 30 synthetic tests include corrupt provenance, unequal frame/particle weighting and missing seeds.

```sh
python -m research.summarize_training_validation \
  --output-prefix research/results/full_validation_curves --plot
python -m research.nonadditive_allocation_toy
```

The [actual-set allocation note](research/notes/nonadditive_allocation_analysis.md) proves the additional uniform-approximation term needed to transfer an additive-score regret bound to nonlinear set benefit. A monotone exact-budget counterexample shows that perfect singleton gains alone are insufficient. The synthetic enumeration does not establish an interaction bound for the trained simulator.

`research/run_evaluation_queue.py` defaults to read-only preflight. It refuses checkpoint loading while training is active and never reads reserved test data in check-only mode. Execution requires all six fixed final models, every scheduled validation record, exact source/configuration hashes, inactive process checks and exclusive training/model/evaluation locks. It then converts the pinned test source and runs the twelve locked evaluations plus two summaries sequentially. Numerical/protocol errors stop for review; completed guard failures remain results. The hard cutoff is October 7, 2026, 08:00 UTC. SIGTERM, SIGINT, orphan children and deadline recovery have synthetic coverage.

```sh
python -m research.run_evaluation_queue --help
# Supply --training-dir, --data-dir, --raw-test and a separate local --output-dir.
# --check-only is the default; --run retains the six-model and inactivity gates.
```

The supervisor and genuine child parsers passed 65 synthetic tests. Its real read-only check during active training returned `not_ready`, preserved both lock hashes and created no evaluation directory; no checkpoint or test data was loaded. Raw final evaluation artifacts belong in a separate local work directory, with only reviewed compact results curated for publication.

The [manuscript evidence audit](research/results/manuscript_evidence_audit_20261004.json) maps 97 entries and reports 16,491 saved-scalar, rounding, aggregation, identity and provenance checks with no numerical mismatches. It does not rerun simulations, raw ranks or historical bootstraps. The compact rollout report now states the base-only cached-risk first forecast and the risk-head cost included in base timings. Only report text changed; the original numerical summary and producer hash remain preserved.


## Exploratory physical descriptors from cached pilot predictions

[The complete report](research/results/physical_complexity_pilot.md) and [figure](research/results/physical_complexity_pilot.png) preserve all 15 models, both inspected splits, fit coverage, every descriptor and all seed values. The protocol and pinned summary identity were committed at `4a10565` before computing outcomes. Risk correlates moderately with local strain; the association is smaller after speed/count/wall adjustment, and adjusted vorticity is weak. This is descriptive compact-pilot evidence and does not establish useful allocation or independent confirmation.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m research.analyze_physical_complexity \
  --data-dir data-pilot --output-prefix work/physical_complexity/analysis
python -m research.report_physical_complexity \
  --input work/physical_complexity/analysis.json --output-prefix research/results/physical_complexity_pilot
python -m pytest research/tests/test_physical_complexity.py \
  research/tests/test_rank_diagnostics.py research/tests/test_analyze_physical_complexity.py -q
```

The analysis loads no checkpoints and runs no model inference. Exact cached-source hashes are required; fresh output prefixes preserve earlier outcomes. All 122 focused synthetic tests passed. Independent scipy/QR and aggregation checks found no mismatch across 33,070 saved-array checks. Numerical sources/protocols remain at their pre-analysis hashes. Raw descriptor arrays remain local; the scalar artifact records their hash. The larger training/evaluation protocols are unchanged.


## Cheap physical allocation on the fixed compact pilot

The [complete exploratory comparison](research/results/physical_allocation_pilot.md), [scalar artifact](research/results/physical_allocation_pilot.json) and [figure](research/results/physical_allocation_pilot.png) retain two new exact-budget controls and all six original policies. Negative mandatory-base degree and observed velocity RMS use only the original observed graph/history. All 15 fixed checkpoints, both previously inspected splits and all particles are included. The protocol and input identity were frozen at `3d1320b` before inference, after inspecting the earlier correlation results.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m research.physical_allocation_pilot \
  --data-dir data-pilot --output-dir work/physical-allocation-attempt
python -m research.report_physical_allocation \
  --input work/physical-allocation-attempt/summary.json \
  --output-prefix research/results/physical_allocation_pilot
python -m pytest research/tests/test_physical_allocation.py \
  research/tests/test_physical_allocation_pilot.py -q
```

The runner requires the pinned compact checkpoints, original controls and data hashes; fresh output directories preserve failed or partial attempts. One CPU thread completed the attempt in 37.737 seconds. All 6,480 original-policy replay comparisons and exact edge counts passed. The 54 focused tests passed before inference; the independent artifact audit passed 84,612 checks, including all 1,080 raw-frame hashes and a predetermined 90-frame numerical sample. Its JSON states what was not independently recomputed. The accompanying audit script is archived verbatim, with its original local `work/physical_allocation_pilot/attempt_20261004` layout; it is not the portable experiment entry point. Raw arrays and model files remain local, with their hashes retained.

Velocity RMS improves NLL test error over random/current/previous risk, but faithful test means favor risk policies and beta-NLL reverses its random-control comparison between validation and test. Reported percentage changes are computed separately within seed, then averaged. These exploratory one-step outcomes establish neither autonomous stability nor a full-architecture or runtime advantage. The locked full-model evaluation still has its original five policies.

The [six-model completion ledger](research/results/six_model_verification_20261005.json) retains all prior validation/checkpoint objects and links the [330,370-check validation audit](research/results/six_model_validation_audit_20261005.json), [timing audit](research/results/six_model_timing_audit_20261005.json), and [evaluation launch gate](research/results/evaluation_launch_20261005.json). All six 100k models are complete; the locked test-policy and same-state results remain incomplete. Initial unsuccessful audit expectations are preserved separately.
