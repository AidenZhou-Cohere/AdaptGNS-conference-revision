# Prepared CUDA Sand training entry

`train_sand_cuda.py` is preparation only. No real data, repository model, GPU,
network or remote process was executed while writing or testing it. No endpoint,
cohort, source admission or launch is selected by the entry. The default command
only prints its contract and does not import Torch or repository code.

The source is the human-selected, already acquired DesignSafe Sand NPZ family,
repackaged losslessly into the existing numeric manifest schema. The trainer
never opens a legacy NPZ or invokes pickle reconstruction. It reads only a root-
admitted `train.json`, its numeric arrays and unchanged Sand metadata. The actual
trajectory length comes from the observed source admission; neither 320 nor 321
is assumed from metadata. Six observed frames and two coordinates are fixed.

## Reuse and fixed recipe

The entry verifies SHA256 for the frozen `research/full_training.py` and six GNS
modules before import, and verifies the imported modules' paths. It reuses
`sample_indices`, `host_noise`, `learning_rate`, `unpack_batch`, `forward_batch`,
the model/loss builders, numeric manifest loader and tensor-copy/finite helpers.
It does not reuse the WaterDrop runner, MPS RNG capture, checkpoint schema,
validation schedule or standard-run classification.

The fixed model is width 128, ten message-passing blocks, two MLP layers, batch
two, history six, original noise 6.7e-4 and original noise-adjusted normalization.
Only faithful or corrected NLL is accepted. Training uses base-only native
`scipy_host` radius .015, cap 128 and self candidates, with no radius augmentation.
Adam uses betas (.9, .999), epsilon 1e-8, no weight decay/clipping,
`foreach=False`, `fused=False`, and the original exponential 1e-4 to 1e-5 schedule
over 100000 updates. The endpoint is a required argument and does not rescale
that schedule. A prospective protocol can explicitly supply 100000 updates.

Only the already validated Torch 2.13.0+cu129 / CUDA 12.9 / GB200 environment is
accepted. Execution is float32 with TF32 disabled, and no AMP, compile, DDP,
device fallback or tuning. CPU execution of this entry is not offered. CPU toy
models are used only by the separate test file.

## Admission JSON and converter contract

Root creates admission only after reviewing the actual structural report. The
converter's `complete_structural_only` status cannot admit training. Required
admission fields are:

```json
{
  "schema": "adaptgns_sand_training_admission_v1",
  "status": "admitted",
  "dataset": "Sand",
  "manifest_sha256": "REPLACE_WITH_TRAIN_MANIFEST_SHA256",
  "metadata_sha256": "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0",
  "converter_sha256": "REPLACE_WITH_REVIEWED_CONVERTER_SHA256",
  "structural_report_sha256": "REPLACE_WITH_STRUCTURAL_REPORT_SHA256",
  "record_count": "REPLACE_WITH_OBSERVED_INTEGER",
  "frames_per_trajectory": "REPLACE_WITH_OBSERVED_SINGLE_INTEGER",
  "particle_type_ids": "REPLACE_WITH_OBSERVED_SORTED_INTEGER_LIST",
  "position_dtype": "<f4"
}
```

This is an intentionally invalid template, not an issued admission. The actual
record count and frame length must be positive integers, with at least seven
frames. Particle types must be the exact observed sorted list in 0..8, excluding
3. Every source position array must be little-endian float32; float64 or another
source precision stops for review rather than silently accepting the frozen
loader's float32 cast. Particle types may be scalar or length N, int32/int64.
Unknown top-level, source, record or array fields are rejected. Material or
auxiliary fields and type 3 require a separately reviewed contract.

The agreed numeric manifest top-level fields are `format`, `version`, `split`,
`dataset`, `source`, `metadata`, `metadata_sha256`, `record_count`, `records` and
`converter_sha256`. Source has exactly `family`, `dataset`, `file`, `size_bytes`,
`sha256`, `ZIP_CRC_verified`, `member_count` and `acquisition_report_sha256`.
Records retain ordinal `id`, `source_index`, `source_member`, numeric
`positions`/`particle_types`, and both file-composite and logical content hashes;
an optional `source_inner_index` is permitted. Arrays have path, shape, dtype,
size_bytes and SHA256. The loader verifies all numeric array/metadata hashes;
the entry additionally checks finite positions, exact T, observed types, array
byte lengths, duplicate hashes and source order. Cross-split duplicate and
source-to-output equality evidence belongs in the reviewed structural report.

## Explicit invocation after freeze

No concrete launch command or endpoint is supplied here. The interface is:

```text
python train_sand_cuda.py --execute
  --repo REPOSITORY
  --train-manifest ADMITTED_NUMERIC_ROOT/train.json
  --admission ROOT_ISSUED_ADMISSION.json
  --structural-report REVIEWED_STRUCTURAL_REPORT.json
  --protocol FROZEN_RESEARCH_PROTOCOL.md
  --output-dir NEW_RUN_DIRECTORY
  --objective faithful|nll --seed EXPLICIT_SEED --updates EXPLICIT_ENDPOINT
  --cuda-index DEVICE_INDEX --threads CPU_THREADS
  --checkpoint-every EXPLICIT_INTERVAL --log-every EXPLICIT_INTERVAL
```

The optional `--stop-after` stops at a committed optimizer boundary, labels the
run `planned_stop_incomplete`, and preserves the requested endpoint/configuration.
Resume adds `--resume` with exactly the same source/data/protocol/software/device
identity and scientific settings. GPU UUID and index are bound to the config;
device migration is not an implicit exact-resume operation. A stale `run.lock`
can be removed only with `--resume --clear-stale-lock`, after a same-host PID is
verified dead. Live, reused, malformed, foreign-host or unverifiable PIDs are
not cleared; a leftover recovery lock requires manual inspection.

Each atomic checkpoint is standard model format v2 plus the distinct
`adaptgns_sand_cuda_training_v1` lineage, exact configuration, optimizer state,
completed update count, history, CPU RNG and selected-device CUDA RNG. Resume
accepts only the checksum-bound `latest.json`; unsafe pointer paths, over-endpoint
counts, source/config drift and the old MPS schema are rejected. Missing pointers
require manual root inspection; the entry never silently starts over. Moment
coverage, exact CPU step counters, Adam hyperparameters/device/dtype/shape and
finiteness are checked. Second moments must be nonnegative. Restore compares the
model/optimizer bytes and both RNG states exactly. This does not claim bitwise
CUDA kernel trajectories across runs or devices.

Loss, all predictions/variances/targets, every gradient, parameters and Adam
moments are guarded. A failed update never advances the successful-update count
or committed pointer. Device moment predicates are aggregated before a single
host boolean synchronization. `status.json` separates successfully guarded
`completed_steps` from `committed_steps`; interruption may require replay from
the last committed checkpoint. The state is `complete` only at the exact
requested endpoint.

## Minimum validation still needed before scientific launch

1. Freeze and independently review the final converter, trainer, admission and
   prospective research protocol; verify actual NPZ structure, precision, types,
   full counts, unchanged metadata and train/validation duplicate checks.
2. Run a bounded actual-training-data CPU/CUDA forward/loss/gradient comparison
   and a same-CUDA checkpoint/CPU+CUDA-RNG/Adam replay under the selected source
   contract. The existing synthetic GB200 validator does not admit source data.
3. Measure sustained actual-data guarded update time and checkpoint I/O under
   intended concurrency. `guarded_update_seconds` spans synchronized host
   sampling/noise, transfers, host graph construction, forward/backward, guards
   and Adam. It excludes checkpoint/log/status I/O. Logged rows are samples when
   `--log-every` exceeds one; do not treat their average as all-update throughput.
   Use the complete externally timed run interval for end-to-end planning.
4. Admit a complete cohort only after separately budgeting validation, full
   rollouts, diagnostics, audits and manuscript work. This entry contains no
   validation/test evaluation and performs no checkpoint selection. Those need
   a separately frozen Sand contract; the old 995-step WaterDrop evaluator is
   not sufficient.

Local verification: `test_train_sand_cuda.py` covers default no-import behavior,
explicit endpoint/seed requirements, admission/schema rejection, source hashes,
locks/pointers, exact toy CPU/mock-CUDA RNG and Adam replay, nonfinite guards,
and planned stop/resume versus uninterrupted execution to the exact endpoint.
These checks do not establish real-data compatibility, real CUDA replay or
scientific quality.
