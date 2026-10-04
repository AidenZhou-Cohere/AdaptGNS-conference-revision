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

The local work used Python 3.12, CPU Torch, SciPy and PyG. `research/requirements-local-lock.txt` records the exact installed packages. For a fresh CPU environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy torch torch-geometric absl-py matplotlib pytest tfrecord crc32c protobuf
export PYTHONPATH="$PWD:$PWD/adaptive-gns"
python -m pytest tests research/tests -q
```

GPU use needs a Torch build matching the machine and the appropriate PyG radius backend (`torch_cluster`/supported compiled dependencies). CUDA, distributed training, and the inherited full upstream test suite were not validated in this local revision. Do not assume the macOS lock file is a portable CUDA installation recipe.

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
