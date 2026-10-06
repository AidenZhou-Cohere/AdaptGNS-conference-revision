# Bounded Goop3D vectorized capacity protocol

This is infrastructure preparation with no selected scientific training endpoint.
The capacity payload, release, output and summary have separate schemas. The
fixed learning-rate horizon `probe_lr_horizon=100000` is a probe parameter only.
Every future scientific cohort requires a separate prospective endpoint, protocol
and root release, and starts from fresh paired initialization.

## Prerequisites and fixed jobs

Use the hash-pinned `train_goop3d_graph_support_cuda_v2.py` numerical routines,
`goop3d_graph_support_vectorized_v1.py` graph, and the unchanged process lifecycle
in `measure_sand_cuda_capacity_v2.py`. The capacity harness privately replaces
that lifecycle's schedule, worker command, worker audit and pairing audit. No
scientific trainer command, resume path or promotion option is exposed.

Execution requires the complete pinned train manifest, structural/acquisition/
auxiliary reports, official context-semantics evidence, source-only data review,
CPU equivalence proof, and an independently reviewed passing same-CUDA numerical
report. The latter contains all six small/median/large × base/mix cases, two
original/candidate updates per case, both checkpoint replays, all 26 optimizer
calls, exact graph/network/output/gradient/model/Adam/RNG checks, and nonzero
optional-edge exposure. It must preserve prospective batches before CUDA and
reverify source/data inputs. Numerical-validator timings are not capacity rates.

Root must issue a fresh `admitted_for_capacity_only` release binding every
source, protocol, report, data-evidence and interpreter hash. Its source-only
`data_contract` has no endpoint fields; `scientific_training_admitted` and
`scientific_endpoint_selected` remain false. The supplied template is inert.

| Wave | CUDA index | Arm | Seed |
|---|---:|---|---:|
| A | 0 | base | 0 |
| A | 1 | mix | 0 |
| A | 2 | base | 1 |
| A | 3 | mix | 1 |
| B | 0 | base | 2 |
| B | 1 | mix | 2 |

All objectives are faithful. Wave A must fully pass before B starts. Each job
uses exactly 512 updates, batch two, history six, width128, ten message-passing
blocks, two MLP layers, radius0.025, noise6.7e-4 and the reviewed graph mixture.
Adam and the deterministic host sample/noise schedules are unchanged. The probe
learning rate is the reviewed exponential schedule evaluated with horizon100000,
not a512-step decay to the final learning rate. Updates1–64 are warmup.

The environment is Torch2.13.0+cu129, CUDA12.9, GB200, two CPU threads per worker,
float32 matmul precision highest, strict deterministic algorithms with warn-only
false, and TF32/AMP/compile/DDP false. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`.
Do not remap `CUDA_VISIBLE_DEVICES`. The release binds hostname and four distinct
physical GPU UUIDs in the raw `nvidia-smi` `GPU-...` spelling. Root's process
inventory must be at most300seconds old at supervisor launch.

## Isolation, evidence and verification

Use one four-GPU host, separate from the existing Goop2D science hosts. All its
GPUs must have no compute processes before either wave. Foreign process/device
observations fail the capacity run. The inherited lifecycle records argv, PID,
parent/start ticks, initial checkpoint pointer, stdout/stderr, external elapsed
time, Linux peak host RSS, per-job outcome and device observations. It requires
an observation of every wave job running on the assigned device at updates64–511.
That proves observed overlap, not continuous full-wave concurrency at every
point. Keep launch skew and all observer overhead in the raw record.

Each wave has the inherited3600second timeout plus at most15seconds of cleanup.
The worker and supervisor refuse to continue at2026-10-07 01:00UTC. Cleanup
signals only captured owned children with matching identities, progresses through
SIGINT/SIGTERM/SIGKILL, and records unreaped children. Failures never trigger an
automatic retry or release of the next wave. Root retains ownership of the outer
host/process timeout and must review any unreaped children.

Outputs must be fresh and separate from data and numerical-core trees. Checkpoints
are written only at0 and512, with purpose `bounded_capacity_only_never_promote`,
the distinct capacity schema, full model/Adam/RNG state, graph/scalar history,
capacity configuration and source/data lineage. They contain no `training_config`
or `run_config` and are rejected by the scientific trainer's restore routine.
CPU verification loads both saved checkpoints, checks exact dtype/value restoration,
finite parameters/moments and every Adam step, and validates RNG payloads. It is
an artifact audit, not a CUDA resume test; CUDA replay is covered by the separate
numerical prerequisite.

All512 finite scalar rows must exactly match stdout and graph records; sample IDs
are recomputed from the fixed seed/update schedule. Paired arms must share initial
model/RNG/empty Adam and every sample, noise hash, learning rate, particle count,
native graph hash/count, noisy-current hash, exposure coin, seed material and
annulus budget. Optional selection and expanded graphs may differ as prescribed.
Worker configuration must equal the supervisor's full released input mapping.
All source/core/input copies, observed wave records and checkpoint/artifact hashes
are checked again before a successful summary.

On worker failure preserve history, initial/last committed pointer, error/context,
current model, Adam, gradients and RNG in `unsuccessful_state.pt`; a secondary
preservation failure is itself recorded without deleting prior output. Supervisor
failure preserves wave outcomes and marks whether wave B was ever released. Do
not relabel an interrupted or partially verified run as measured capacity.

## Capacity quantities and interpretation

For job i, use elapsed wall time and all raw guarded timings:

```text
q_i = (elapsed_seconds[512] - elapsed_seconds[64]) / 448
r_i = external_elapsed_seconds_i - sum(guarded_update_seconds[1..512])
q4 = max(q_i in A); q2 = max(q_i in B)
r4 = max(max(0,r_i) in A); r2 = max(max(0,r_i) in B)
```

Reject residuals below−1e-6; clip only roundoff within that tolerance. The external
timer includes startup, checkpoint/log work and exit-observer lag; supervisor
verification can increase that lag. Wall-q includes per-step status/log overhead.
These engineering rates and residuals do not establish full-study runtime.

The summary reports `all_six_verified_capacity_only`, `q4_q2`, `r4_r2` and all six
audited jobs. Both `scientific_endpoint_updates` and
`full_study_runtime_forecast_seconds` remain null. Any later feasibility worksheet
must separately account for the proposed scientific endpoint, full evaluation,
diagnostics and verification before a prospective endpoint can be selected.
No loss, probe accuracy, partial run or favorable seed may select that endpoint.
