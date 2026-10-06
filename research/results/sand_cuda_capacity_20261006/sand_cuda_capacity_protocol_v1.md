# Deterministic native-CUDA Sand capacity measurement, version 1

This is a prospective infrastructure timing protocol. It does not admit a
scientific cohort. Freeze this file, supervisor, trainer, core sources, numeric
training manifest, admission, structural report, root numerical assessment and
Python executable by SHA256 in a root-issued release before execution.

The earlier `sand_cuda_feasibility_schedule.md` remains preserved as the initial
proposal. Its exact 4+2 scheduling and 512-update probe plan are retained. Its
references to `train_sand_cuda.py` and passing the strict CPU/CUDA gate are
superseded only for this separately authorized timing measurement. Both strict
CPU/CUDA gradient reports remain FAILED. Root's
`sand_native_cuda_numerical_assessment.json` permits timing the natural,
deterministic CUDA backend; it does not assert CPU training-trajectory
reproduction or scientific admission. No diagnostic backward masks enter this
trainer. The residual small/NLL Adam-update discrepancy is retained alongside
all original failures.

## Fixed jobs and isolation

| Wave | CUDA index | Objective | Seed |
|---|---:|---|---:|
| A | 0 | faithful | 0 |
| A | 1 | nll | 0 |
| A | 2 | faithful | 1 |
| A | 3 | nll | 1 |
| B | 0 | faithful | 2 |
| B | 1 | nll | 2 |

The four jobs in A launch together. Every A outcome must verify before B starts.
B launches two jobs, leaving CUDA indices 2 and 3 idle. All GPUs must be free of
compute processes before either wave. Any foreign GPU process or incorrect
UUID assignment during observation fails the measurement. All outcomes,
including partial runs and failed attempts, remain in their fresh infrastructure
directory. No automatic retry, resume, overwrite, seed selection or checkpoint
promotion is allowed. A later scientific cohort must start from scratch.

The trainer is `train_sand_cuda_deterministic.py`, SHA256
`9dd376f6d4b6881290b134c8f410b89933bcdb026c9b52de601ba0ec7bb05a9b`.
It retains width 128, 10 message-passing blocks, 2 MLP layers, batch 2, history 6,
noise 6.7e-4 and the pinned graph/Adam recipes. Each command includes exactly:

```text
--execute --updates 100000 --stop-after 512 --threads 2
--checkpoint-every 10000 --log-every 1
```

Use Torch 2.13.0+cu129, CUDA 12.9, validated GB200 hardware, float32 and strict
deterministic algorithms with warn_only false. TF32, AMP, compile and DDP stay
off. Required environment:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

Do not set CUDA_VISIBLE_DEVICES. Use explicit physical-index UUID mapping from a
fresh root process/GPU identity check, at most 300 seconds before launch. A
release with status `admitted_for_timing`, issued_by `root`, explicit rationale,
exact source/interpreter/data/protocol/review hashes and
scientific_training_admitted=false is mandatory. The supplied template is
intentionally not admitted.

## Verification and preserved measurements

Success requires process exit 0, removed run lock, no temporary artifacts,
`planned_stop_incomplete`, completed_steps=committed_steps=512 and requested
steps 100000. Verify exact source/config/data/runtime fields, canonical run
configuration hash, initial step-0 and final step-512 pointers and checkpoint
byte hashes, final status/history consistency, and all 512 finite ordered rows.
Both checkpoints remain infrastructure artifacts. Their numerical tensors are
not loaded by this supervisor; trainer-side finite-state/replay guards are the
numerical protection inherited from the reviewed trainer.

Persist stdout, stderr, process argv/PID/start-ticks/executable identity, initial
pointer immediately after observation, external start/end/elapsed time, exit
code and Linux peak host RSS (ru_maxrss KiB converted to bytes), per-job outcomes,
GPU UUID/process observations and wave timings. The supervisor uses owned-child
identity checks before interrupt/terminate/kill, then attempts bounded reaping.
Unreaped children are explicitly recorded and prohibit any unreviewed restart.
A wave can run at most 3600 seconds; cleanup uses 15 additional seconds.

The external timer includes import/admission/startup, model creation, checkpoint
and logging work and exit-observer lag. The nominal polling interval is 0.1
seconds; verification and GPU inventory calls can increase observer lag. Preserve
this overhead instead of subtracting it. Record wave launch skew and its total
span minus the slowest external job duration.

Require at least one simultaneous observation of every wave job on its assigned
GPU with trainer state running and update count64..511. Preserve every observation.
This proves actual overlap; it does not prove every point of every steady window
has the full wave concurrency. Report q4/q2 as the observed nominal four/two-job
launch-wave rates. The fixed contingency below addresses uncertainty only as a
planning allowance, not a statistical bound. There is no measured six-parallel
or eight-GPU scaling estimate in this protocol.

Pair faithful and NLL by seed and verify exact frame IDs, total particle counts
and learning rates for every row. Frame/noise schedules depend on seed/update
through the pinned helpers and have no objective input. Validate IDs against the
admitted training source. Never use loss or probe performance to choose seeds,
endpoints or a favorable subset.

## Frozen forecast

Updates1..64 are warmup. Keep all512 raw guarded and elapsed timings; report
mean/median/linearly interpolated p90/maximum guarded time for updates65..512,
warmup guarded total and excess over64 times the steady guarded mean.

For each job i:

```text
q_i = (elapsed_seconds[512] - elapsed_seconds[64]) /448
r_raw_i = external_elapsed_seconds_i - sum(all512 guarded_update_seconds_i)
r_i = max(0, r_raw_i)
q4 = max(q_i in A); q2 = max(q_i in B)
r4 = max(r_i in A); r2 = max(r_i in B)
training_forecast_seconds = 1.35*(100000*(q4+q2) +12*(r4+r2))
```

q is elapsed wall time, not the mean guarded-update time. Keep raw and clipped
residuals. The 1.35 contingency and factor12 are fixed engineering allowances,
not probabilistic confidence limits. Wall-q and residual partly double-count
logging; retain this deliberately conservative overlap. Logging every update
also exceeds the later scientific logging frequency. Do not adapt the formula
after viewing timings.

A training-only forecast is available after all six probes verify. A complete
deadline-fit decision stays null until there are separate, preserved timing
estimates for (a) all900 full-H314 rollout outcomes: six models ×30 trajectories
×five policies, and (b) required diagnostics execution. The current declared
diagnostics workload has95400 same-state and2304 clean forwards, total97704;
its estimate must cover97704 times the maximum measured whole-call time plus
explicit graph/artifact overhead. A proposed different workload needs a new
prospective protocol and source review, not silent replacement in this record.

```text
total_compute_analysis_seconds = training_forecast_seconds
    + complete_evaluation_seconds + diagnostics_execution_seconds +3600
```

The final3600 seconds remain an additional diagnostics/verification allowance;
they do not substitute for unmeasured diagnostic execution. Required predicted
completion is by2026-10-07 01:00 UTC, preserving seven hours for writing before
the08:00 UTC no-new-work cutoff. All cost estimates and each forecast snapshot
are hashed and preserved. Even a favorable forecast does not automatically
admit scientific training: root must freeze a separate scientific protocol and
explicit release. Reserved Sand test payloads remain untouched during capacity
and validation timing.
