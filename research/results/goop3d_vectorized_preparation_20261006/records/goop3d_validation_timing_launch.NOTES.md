# Root-owned six-call Goop-3D validation timing

This preparation writes contracts and supplies a separate supervisor around
the unchanged reviewed evaluator. It does not launch a job, acquire any data,
read a test source or select a scientific endpoint. The existing capacity,
trainer, evaluator and their frozen protocols remain unchanged.

## Prepared files

| File | SHA-256 |
| --- | --- |
| `run_goop3d_validation_timing_v1.py` | `104e7887a2e9fd9760e2cdacbcab6e850856120a9b6e941a8908b764ccccbb29` |
| `test_run_goop3d_validation_timing_v1.py` | `15b2d6d9422bc7337cbdaa1463899d76753b98e6f9dbd058aac6d09988944a53` |
| `prepare_goop3d_validation_timing_release_v1.py` | `136c9cf30c2f7422514e8fc48f79edab5e17e511f1f639fb9ff6688a5ceaa4b7` |
| `test_prepare_goop3d_validation_timing_release_v1.py` | `fa3f0e2a004621b000f34e3f214ec1307a767dad32f2b944c97c6d3d7c9415d3` |
| `goop3d_valid_timing_split_admission.template.json` | `858347060fdd69a39ed147ec714474d4f4771f09601016e3c3ac02f38e3e2d08` |

Twenty-two supervisor tests and eight contract-generation tests passed. They are
scalar fixtures/process mocks, including wait4 accounting, duplicate/ancestor
identity, timeout cleanup, partial launch, contamination, missing work and
retained guard outcomes. No actual GPU or official numeric-array execution
was performed. Initial fixture failures are preserved/documented in
`goop3d_validation_timing_authoring_record_v1.json`. Independent review is a
separate root prerequisite; use its exact final hashes if revisions are made.

## Required state before preparation

All six512-update capacity probes must finish successfully. Preserve and
inspect the reviewed capacity supervisor's complete `summary.json`, status,
both waves and all checkpoint/pairing audits. Root must verify the capacity
supervisor and every worker are stopped/reaped or independently absent and
that no competing research or GPU process remains. Do not merely infer this
from a512-step checkpoint on disk.

Create a new receipt from `goop3d_capacity_stopped_receipt.template.json`:
set issued_by=root, status=`supervisor_and_all_six_workers_stopped`, bind the
actual capacity summary SHA, and set the absence/reaping Boolean true only
after the process check. Keep PID/start/command evidence and the check time in
the receipt's additional fields. The generator and supervisor require this
receipt; neither can make the root's factual process assertion for it.

Stage the evaluator, native/metric adapters, worksheet, timing supervisor and
generator, new timing protocol and updated inert valid admission template into
the third VM's existing `cuda_preparation` directory. Retain all unchanged
common core, auxiliary and official context-source dependencies. The actual
data path is `goop3d_numeric_train_valid_20261006_v1/valid.json`; it references
the full100-record numeric tree and adjacent metadata. A local scalar-only
manifest copy is not a runnable replacement.

## Generate exact releases from one root spec

Start from `goop3d_validation_timing_root_spec.template.json`, whose paths,
hostname and GPU0/GPU1 UUIDs come from the third VM's recorded capacity launch.
Recheck these against the current host. Write a new spec (do not overwrite the
inert template), and set:

- status=`approved_for_contract_generation_and_six_bounded_validation_calls`
- issued_by=`root`
- root_approved_valid_source_admission=`true`
- process_identity_checked_utc to a fresh timezone-aware root check
- reviewed source/template hashes to the independently reviewed exact versions
- capacity_stop_receipt to the verified new receipt

Keep every scientific/test permission flag false. No final cohort is needed.
The spec grants narrow authority to derive these six concrete validation-only
releases; generating them does not start execution. Root selected limits of
full-rollout3600 inner/3900 outer seconds and same-state/clean-validation600
inner/900 outer seconds. If root chooses different limits prospectively,
the child max_seconds and parent bounds are generated consistently.

Use the same required deterministic environment as capacity:
`CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`,
and no `CUDA_VISIBLE_DEVICES`. From `/root/repos/AdaptGNS-cuda-20261006`:

```text
.venv/bin/python -B cuda_preparation/prepare_goop3d_validation_timing_release_v1.py
  --write-contracts
  --spec cuda_preparation/goop3d_validation_timing_root_spec_v1.json
  --contracts-dir goop3d_timing_contracts_20261006_v1
```

The generator reads scalar reports and hashes bytes; it does not deserialize
checkpoints, load trajectory arrays, inspect processes, call nvidia-smi or
launch inference. It derives the two selected checkpoint paths and hashes
from the complete verified capacity summary, binds all absolute evidence and
source paths, creates the original-valid source admission and all six child
releases, and runs the unchanged supervisor's scalar preflight. Only then does
it publish `supervisor_release.json` and `preparation_report.json`.

The generated valid admission includes source SHA
`5242810fa2057fbcf8e2d4da203823855f6b71e77dcfa848a618fb6871f6778b`.
An earlier inert template omitted this required field; it was corrected and
preserved before any admission or execution. All actual numeric and auxiliary
bytes are still verified by the evaluator before CUDA inference.

Inspect the preparation report and its exact six commands. The parent release
binds all generated child-release hashes, the stopped receipt, both checkpoint
hashes, interpreter bytes, all common inputs and the reviewed sources. The
generator retains the venv interpreter path for execution while hashing its
resolved executable, avoiding loss of venv activation through symlink
resolution. The timing output and contracts directory must both be fresh and
separate from each other, source, data and capacity trees.

If preparation fails, keep the failed directory and `failed_preparation.json`;
do not launch any of its children separately. Fix the reviewed cause and
prepare a new distinctly named directory. The active parent release is not
published until preflight passes. Do not refresh a stale timestamp in an
already bound release by hand; generate a new reviewed spec and fresh bundle.

## Launch once, after inspecting the concrete bundle

The process check must remain fresh: the parent accepts a root timestamp at
most300 seconds old. If review takes longer, preserve that bundle and generate
a new one with fresh process evidence. The supervisor also checks for existing
research/GPU processes before each wave and creates an exclusive output lock.

With the proposed limits, root can use this single outer invocation:

```text
timeout --signal=TERM --kill-after=30s 6300
  .venv/bin/python -B -u cuda_preparation/run_goop3d_validation_timing_v1.py
  --execute
  --release goop3d_timing_contracts_20261006_v1/supervisor_release.json
  --output-dir goop3d_validation_timing_20261006_v1
```

Keep the shell invocation's PID/start/command identity and logs. Do not invoke
individual evaluator commands concurrently with this supervisor. The wave
order is full-rollout, same-state, clean-validation; each wave has base0 on
GPU0 and mix0 on GPU1. All children from one wave are reaped before the next.
The fixed timing source indices are37,51,64. Root can monitor `status.json`,
child evaluator status, raw external receipts and `gpu_observations.json`.

Every child has its reviewed evaluator inner budget plus an independent
whole-child wall-clock budget from before process creation to wait4 exit.
The supervisor cutoff is2026-10-07 01:00UTC. Its owned cleanup sends
SIGINT/SIGTERM/SIGKILL after0/5/10 seconds and stops waiting after15 seconds.
It signals only children whose PID/start/command identity matches. If a child
cannot be reaped, the lock remains and the inventory records its exact
identity; root must resolve that before any new work. The external30-second
grace leaves time for the supervisor's15-second cleanup.

## Outcomes and deadline worksheet

`timing_inventory.json` always has all six planned identities. A stopped child
has raw elapsed time, PID/command identity, exit code, wait4 CPU/RSS statistics,
signals and stdout/stderr paths. Its entire output tree is byte-hashed only
after reaping. Unattempted and unreaped work is explicitly labeled, without a
final artifact-hash claim for a live child. The receipt includes observer lag;
these are measured whole-process durations, not model-only kernel time.

A guard-complete child may exit zero and allow the remaining diagnostic waves
to run. It remains a failed full-H timing outcome. A nonzero exit,
implementation/parity error, changed input, timeout or GPU contamination
stops later work and preserves all attempted/missing records. No retry occurs.

Feed the complete, root-reviewed inventory to the existing scalar worksheet
only after checking the supervisor's final state and lock removal. That
worksheet requires every full-H295 outcome and diagnostic frame; a guard,
missing policy or unattempted mode yields forecast_unavailable. It scales
measured per-policy/frame maxima serially and receives no concurrency credit.
Neither successful timing nor a favorable worksheet chooses a scientific
endpoint or promotes a capacity checkpoint. Preserve the original third-host
capacity_summary, planning_config and timing_inventory at their bound absolute
paths; the scientific release will consume those original bindings, not remapped
local copies. Failed-H hashes and an unavailable worksheet also remain intact
if root prospectively chooses a separate quota-based scientific endpoint.
