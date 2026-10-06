# Sand scientific supervisor preparation

This is a new supervisor source for a root-released scientific cohort. It is
description-only without `--execute`. Preparation and synthetic tests do not
admit or launch training. Root owns release, host execution, cohort admission,
evaluation and any later recovery decision.

## Fixed work and lineage

Run fresh faithful base/mix models for seeds 0, 1 and 2, exactly 100000 updates.
Host role A owns base0/mix0/base1/mix1 on GPU indices 0/1/2/3. Host role B owns
base2/mix2 on GPU indices 0/1. Both hosts must report four distinct physical GPU
UUIDs in physical-index order. All assigned models use two CPU threads,
checkpoint cadence 10000 and scalar-log cadence 100. There is no resume,
stop-after, seed substitution, automatic retry, network control or probe promotion.

Pinned trainer: `train_sand_graph_support_cuda.py`, SHA256
`fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124`.
Pinned capacity adapter: `measure_sand_graph_support_capacity.py`, SHA256
`ca9b5ed45455b575179464d35c42d0d03e87edb7508f4deea2451f9f3fb50878`.
Pinned lifecycle: `measure_sand_cuda_capacity_v2.py`, SHA256
`c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd`.

The adapter is imported into a private module. Owned-process identity,
initial-pointer observation, wait4 reaping, signal escalation and atomic JSON
are unchanged pinned lifecycle helpers. The capacity run-wave loop is not used.
The new loop polls children every 0.2 seconds and records GPU/status observations
every 30 seconds. A child is registered immediately after Popen, before its
identity check, so a failed identity check cannot orphan an unrecorded child.

Reaping performs only clean-exit and complete-100k-status checks while any peers
may be running. Full CPU checkpoint/history checks begin after every child has
been reaped. The per-child outcome's `verification=passed` means that cheap exit
check; the wave explicitly records this scope. Only the eventual worker summary
can claim `verified_host_scientific_endpoints`.

## Complete workload and deadline release

The scientific release must bind exact SHA256 bytes for this supervisor, the
trainer, capacity and lifecycle sources, frozen train manifest/admission/
structural report, prospective scientific protocol, complete six-job capacity
forecast, complete cost estimate, current host process check, root-reviewed host
clock check and the Python executable. The declared Python bytes must also match
the running supervisor interpreter. The numerical repository source pins are
checked separately. Copied input/release bytes and all live source/input/release
bytes are rechecked before launch and after endpoint/pairing inspection.

The six-policy workload is fixed: 1080 H314 autonomous outcomes plus 124704
diagnostic forwards. The cost contract rejects missing or nonpositive rollout,
diagnostic or terminal-ledger reserves. The distinct graph-capacity forecast
must contain all six verified 512-update timing probes, exact embedded cost
bytes and a fitting full workload. These probes stay infrastructure only.

Let q6 be the maximum measured steady seconds/update and r6 the maximum
nonnegative external-minus-guarded residual across those six concurrent jobs.
The supervisor recomputes:

```
training = 1.35 * (100000*q6 + 12*r6) + ledger_total_reserve_seconds
remaining = full_rollout_seconds + diagnostics_execution_seconds + 3600
total = training + remaining
compute_analysis_deadline = 2026-10-07T01:00:00+00:00
training_stop <= deadline - remaining
cleanup_trigger = training_stop - clock_error_bound_seconds - 15 seconds
latest_start <= min(deadline - total, cleanup_trigger - training)
current_host_time + clock_error_bound_seconds <= latest_start
```

The extra 15 seconds uses the pinned bounded cleanup allowance. Root may choose
earlier starts/stops. The launch fit is checked after setup and before each
child. Running children are interrupted at cleanup_trigger; only owned children
are signaled, with SIGINT/TERM/KILL escalation and bounded reaping. Any failure
preserves outcomes and leaves whole-cohort admission false. These are forecast
and monitoring bounds, not a guarantee against OS/driver hangs or scheduling
stalls; root must review any unreaped-child report.

Root must use a final prospective protocol that includes the sixth physical
baseline, fixes its complete cost, names reviewed implementation sources and
records the same-device numerical implementation evidence. The earlier draft
protocol remains unmodified and unadmitted. Initial states are saved before
training; their bytewise pairing audit is permitted after updates start and is
required before cohort admission. Do not claim a before-step1 audit barrier.

## Supporting evidence schemas

`adaptgns_sand_scientific_process_check_v1` requires `checked_utc`, exact `host`,
physical-index-ordered `gpu_uuids`, `matching_training_processes: []` and
`gpu_processes: []`. Root obtains these from fresh actual host evidence. The
supervisor also performs a current nvidia-smi idle check before spawning.

`adaptgns_sand_scientific_clock_check_v1` requires `issued_by: root`, `checked_utc`,
exact `host`, `root_host_samples_reviewed: true` and a finite
`clock_error_bound_seconds` between 0 and 5. Both clock/process checks must be at
most 300 seconds old and no more than 5 seconds in the future when validated.
Root must actually compare host/root clock samples before asserting this bound.

The three `*_TEMPLATE.json` files are deliberately incomplete and unadmitted.
They are not launch authorizations. Fill the release only after all evidence,
prospective protocol and total-workload fit are reviewed; hash the completed
supporting evidence files, not these templates. All paths below are explicit:

```
PYTHON supervise_sand_graph_support_science.py --execute --host-role A \
  --release ROOT_RELEASE --repo NUMERICAL_REPOSITORY \
  --trainer train_sand_graph_support_cuda.py \
  --capacity-source measure_sand_graph_support_capacity.py \
  --lifecycle-source measure_sand_cuda_capacity_v2.py \
  --train-manifest TRAIN_MANIFEST --admission TRAIN_ADMISSION \
  --structural-report STRUCTURAL_REPORT --protocol FINAL_PROTOCOL \
  --forecast COMPLETE_FORECAST --costs COMPLETE_COSTS \
  --process-check HOST_PROCESS_CHECK --clock-check HOST_CLOCK_CHECK \
  --python PYTHON --output-dir FRESH_HOST_OUTPUT
```

Host B uses `--host-role B`, its own actual host evidence/release and a separate
fresh output directory. Neither invocation manages the other host.

## Endpoint and pairing proof

Only a clean one-attempt endpoint with requested/completed/committed 100000 can
pass. Inspect all 100000 graph/schedule rows, all 1001 scalar rows, exact saved
stdout/history/status correspondence, 11 checkpoint filenames, initial/final
file hashes and pointers, complete config/runtime/source/data lineage, finite
float32 model tensors, complete finite Adam moments at step100000, empty initial
Adam and serialized CPU/CUDA RNG vectors. Checkpoints are loaded on CPU only.
Intermediate checkpoints have required names; this verifier does not load all
nine intermediate payloads.

Simulator architecture/bounds/settings are exact; saved normalization means and
scales must be finite two-coordinate float32 with positive scales, and initial
and final simulator configs must match exactly. Normalization construction is
bound to the pinned CUDA trainer/core source; this CPU inspector does not
recompute CUDA square roots. State-dict values map in registration order to
optimizer parameters: the pinned simulator/network contain no registered
buffers, and scalar synthetic tests do not independently instantiate the full
model to reprove that source property.

Within each seed, initial model, simulator config, empty Adam and serialized
CPU/CUDA RNG must be exactly equal. Every one of 100000 frame/noise/native-graph/
graph-RNG records must match across arms, with arm-specific exposure budgets
validated by the frozen trainer. All 1001 saved learning-rate/frame/particle rows
must match exactly. A one-float64-ULP tolerance applies only to local scalar LR
recomputation and records any discrepancy; it never relaxes paired saved LR.
Unlogged learning rates follow the common frozen deterministic schedule source.

Each successful host summary retains `whole_six_model_cohort_admitted: false`.
Root must verify both host summaries, identical cohort/release/protocol lineage,
all six final endpoint hashes, pairing and every required audit before issuing a
distinct final-cohort/evaluation admission. There is no combine or evaluation
mode here.

## Preparation tests

```
work/venv/bin/python -B -m pytest -q -p no:cacheprovider \
  work/deadline_research_20261005/cuda_preparation/test_supervise_sand_graph_support_science.py
```

Thirty tests use scalar dictionaries, tiny individual CPU tensors, temporary
synthetic files and mocked child processes. They include launch-fit refusal,
cleanup registration, deferred endpoint inspection, byte-change rejection,
complete/partial/nonfinite payload checks and valid zero timing residual.
No actual model, trajectory/checkpoint, CUDA, remote host or scientific process
is executed by these preparation tests.
