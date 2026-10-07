# Simulation commands

Run these commands from the distribution root. Data and result directories should be outside `code/` and separate from each other. All output locations must be fresh unless a trainer is explicitly resuming its own exact configuration. Source-only checks and `--help` do not import a model or read simulation arrays.

## Goop and Sand training

Use the complete ordered 1000-trajectory train manifest and adjacent `metadata.json` described in [DATA.md](DATA.md). Goop uses T 401/type 7 trajectories; Sand T 320/type 6. The original metadata hashes, native graph prefix, pair-budget schedule, faithful objective, noise, Adam state and update schedule remain checked.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python code/portable/train_portable.py \
  --dataset goop --frozen-dir code/studies --repo code \
  --train-manifest /data/Goop/train.json --metadata /data/Goop/metadata.json \
  --arm base --seed 0 --cuda-index 0 --output-dir /runs/goop/base_seed0 --execute

CUBLAS_WORKSPACE_CONFIG=:4096:8 python code/portable/train_portable.py \
  --dataset sand --frozen-dir code/studies --repo code \
  --train-manifest /data/Sand/train.json --metadata /data/Sand/metadata.json \
  --arm mix --seed 0 --cuda-index 0 --output-dir /runs/sand/mix_seed0 --execute
```

Repeat for both arms and seeds 0,1,2 in distinct directories. Each job uses exactly 100,000 updates, batch 2, LR 1e-4→1e-5, noise 6.7e-4, width 128 and 10 message-passing blocks. The source-equivalence command verifies the whole optimizer-loop AST before execution. It also checks the byte identities of the packaged reference sources. New runs carry a portable schema and their own input hashes.

The CUDA profile remains explicitly GB200/PyTorch 2.13.0+cu129. This adapter makes data and orchestration paths portable; it does not assert equivalence across GPU generations or software stacks. An unavailable device fails rather than selecting another backend.

## Native graph evaluation

The evaluation entry supports Goop H 395, Sand H 314 and WaterDrop continuation H 995. The complete 30-trajectory test manifest remains present for every material. WaterDrop selects sources 3–29; Goop and Sand select0–29. No policy subset is selected. Goop/Sand include RMS25; WaterDrop retains its five declared policies.

First obtain the exact SHA256s of your selected manifest, metadata and checkpoint. Substitute those64-character hexadecimal values in the command; they bind the specific inputs you intend to evaluate. Omitting `--execute` performs metadata/configuration checks only.

```sh
python code/evaluate.py --study goop --mode rollout --split test \
  --arm base --seed 0 --manifest /data/Goop/test.json \
  --metadata /data/Goop/metadata.json --checkpoint /runs/goop/base_seed0/checkpoint-000100000.pt \
  --manifest-sha256 MANIFEST_SHA256 --metadata-sha256 METADATA_SHA256 \
  --checkpoint-sha256 CHECKPOINT_SHA256 --device cpu --output-dir /results/goop/base_seed0
```

Add `--execute` to run. For the original CUDA arithmetic profile use `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `--device cuda:0`. Use `--study sand` or `waterdrop` with the corresponding complete manifest/checkpoint. WaterDrop requires its faithful 110k continuation schema. CPU evaluation uses the same graph and metric functions but need not reproduce CUDA/MPS floating-point trajectories.

The portable WaterDrop command currently supports autonomous evaluation only; its observed-history execution path is not packaged as a portable command.

For Goop/Sand common-history comparisons use `--mode observed --split valid` or `--split test`; for the 128 fixed clean-validation frames use `--mode clean-validation --split valid`. Observed Risk25 scores the preceding observed base graph; autonomous Risk25 caches its own selected graph. Failure records and numeric prefix traces are written without retries. A complete-grid record counts returned cases, including guards; it is not a claim that every trajectory succeeded. Full-horizon means require all required cases to succeed.

## WaterDrop and Goop3D scope

The 100k faithful WaterDrop parent uses the explicit-path trainer below. Repeat for seeds 0, 1 and 2, with a distinct output directory for each seed.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=code:code/adaptive-gns python -m research.full_training \
  --train-manifest /data/WaterDrop/train.json --valid-manifest /data/WaterDrop/valid.json \
  --metadata /data/WaterDrop/metadata.json --protocol code/research/protocols/full_waterdrop_100k.md \
  --objective faithful --seed 0 --device mps --steps 100000 --threads 2 \
  --output-dir /models/waterdrop/faithful_seed0
```
 The portable continuation entry restores the complete parent model, Adam and RNG state, retains absolute schedule steps 100000–109999 and the fixed LR 1e-5, and binds the parent hash explicitly:

```sh
python code/portable/continue_waterdrop.py --repo code --check-source-equivalence
PYTORCH_ENABLE_MPS_FALLBACK=0 python code/portable/continue_waterdrop.py --repo code \
  --parent-checkpoint /models/waterdrop/faithful_seed0/checkpoint-100000.pt \
  --parent-sha256 PARENT_SHA256 --train-manifest /data/WaterDrop/train.json \
  --valid-manifest /data/WaterDrop/valid.json --metadata /data/WaterDrop/metadata.json \
  --arm mix --seed 0 --output-dir /runs/waterdrop/mix_seed0 --execute
```

Repeat each parent at both arms and seeds 0–2. Parents freshly trained with the packaged source and protocol are supported. The complete parent configuration, data identity, normalization and validation schedule are checked. Archived original 100k parents also bind their original source/protocol hashes; an explicit parent hash alone does not make them compatible with this cleaned source tree. New continuations carry a separate portable schema. Supply the same explicit `--parent-sha256` to `code/evaluate.py` when evaluating them. Do not reset Adam or call fresh 110k training this paired continuation experiment. The MPS/two-thread/no-fallback profile remains required. The adapter has source/configuration tests; it has not been run on real parent checkpoints in this distribution.

The Goop3D scalar results, split-specific source grid, checkpoint/config hashes and selected data hashes are in `results/goop3d/` and `protocols/goop3d_inputs.json`. Its radius.025 and2M-pair/5M-edge guards differ from the 2D settings. The source modules under `studies/` preserve its numerical definitions. The three-material portable evaluation command above does not substitute a 2D protocol for the separate 3D comparison.
