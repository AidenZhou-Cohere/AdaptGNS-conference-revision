# Adaptive Interaction Graphs: correctness and controlled experiments

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

CUDA use needs a matching Torch build and the appropriate PyG radius backend (`torch_cluster`/supported compiled dependencies). Native Metal uses the explicit `scipy_host` backend for CPU neighbor search and edge transfer; unsupported-kernel fallback must stay disabled. CUDA, distributed training, and the inherited full upstream test suite were not validated in this local revision. Do not assume the macOS lock file is a portable CUDA installation recipe.

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
