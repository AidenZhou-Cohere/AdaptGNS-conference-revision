# Prospective Sand graph-support capacity plan

**DRAFT v1, not released or executed.** This plan is separate from the frozen
faithful/NLL 4+2-wave capacity protocol. Preserve that protocol, formula and all
outcomes unchanged. It estimates its own two-wave infrastructure recipe only;
never use it as measured graph-mixture or six-parallel capacity.

The intended scientific workload is six fresh faithful models:base/mix×seeds0/1/2,
100000 updates each. Measure the exact new graph-support trainer after its own
review and numerical admission. The pending second four-GPU VM must independently
match the admitted runtime/driver/GB200 source/data/checkpoint contracts. Until
that endpoint is identified and checked, this is a hardware proposal only.

## Proposed simultaneous launch allocation

| Host | CUDA index | Arm | Seed |
|---|---:|---|---:|
| A, existing four-GPU host | 0 | base | 0 |
| A | 1 | mix | 0 |
| A | 2 | base | 1 |
| A | 3 | mix | 1 |
| B, second four-GPU host | 0 | base | 2 |
| B | 1 | mix | 2 |

The six jobs run concurrently; hostB indices2/3 stay idle. Each pair shares a
host, and graph/data/noise schedules omit device/host/arm. Record launch skew and
both host clocks/process identities/GPU UUIDs. This is six-job capacity using
up to eight available GPUs, not a measured eight-job scaling result. The schedule
must be frozen and actually measured; if root chooses a different mapping or
six jobs on another topology, issue a separate prospective schedule first.

Every probe uses fresh outputs, updates100000, stop_after512, checkpoint_every
10000, log_every1 and2threads. Retain step0/512 checkpoints and full512-row
training/graph histories. Verify initial tensor identity within seed, exact
frame/noise/LR schedules, exposure coins and optional budgets, nativebase prefix,
finite state, source/config/data hashes and strict planned_stop_incomplete512
status. All six must pass. No checkpoint promotion, resume into science,
favorable-arm/seed selection or automatic numerical retry.

Require explicit root timing-only release and process/GPU checks on both hosts.
A coordinated launch/observer must detect foreign work, retain all failures and
show all six GPU jobs overlapped within updates64..511. Its claim remains
nominal full-launch concurrency, with preserved time series; one observation
cannot establish constant concurrency throughout every steady window.

## Prospective formula, subject to review before any measurement

Keep warmup1..64 and steady65..512. For jobi define:

```
q_i = (elapsed_seconds[512] − elapsed_seconds[64])/448
r_i = max(0, external_elapsed_i − sum(all512 guarded_update_seconds_i))
q6 = max(q_i over all6)
r6 = max(r_i over all6)
```

Unlike the staged original probe, simultaneous six-job training uses a maximum,
notq4+q2. Root must freeze this distinct formula before viewing graphmix timing:

```
training_seconds = 1.35*(100000*q6 + 12*r6) + ledger_total_reserve_seconds
total_compute_analysis_seconds = training_seconds
  + full_rollout_seconds + diagnostics_execution_seconds +3600
```

The preferred deadline-aware ledger allowance is an explicit conservative
`ledger_total_reserve_seconds`, frozen by root with supporting serialized-size,
CPU validation and storage-throughput evidence. It may be a substantial fixed
reserve; a large new synthetic measurement suite is not required. It must cover
all11 checkpoint publications (step0 and10k..100k), final history publication,
validation of accumulated ledger rows, CPU-tree copies, serialization/fsync and
shared-storage/concurrency effects. Root must state why the chosen reserve bounds
this work, rather than assuming that a round number is sufficient. A minimal
byte-size/storage check and a generous reserve are preferable to engineering a
new framework under this deadline.

If terminal checkpoint timec_terminal is measured or explicitly bounded, a
sufficient proposed reserve is at least1.35×11×c_terminal, wherec_terminal includes
final-history publication and is the slowest per-model terminal-size path under
intended storage/concurrency. Applying the terminal cost to every checkpoint
conservatively covers growth. This may double-count work already inr6; retain it
prospectively rather than adjusting after results. Contingency1.35 and factor12
are planning allowances, not statistical confidence bounds.

A512-update ledger is not representative of the100000-record scientific ledger.
Every graph-support checkpoint validates accumulated history and serializes it
with model/Adam/RNG state. If a small separate I/O check is needed, it must be
reviewed infrastructure with clearly synthetic100000-record metadata and fixed
probe tensor bytes; never create an artifact that masquerades as a trained100k
checkpoint. Retain its source/input hashes, scope, failures and peak host memory.
If neither measurement nor a justified conservative bound is available, leave
scientific deadline-fit undefined. Do not infer terminal-ledger costs fromr6.

Full-rollout cost must cover900 completeH314 outcomes for the five core policies,
or1080 if root admits the optional relative-velocity-RMS25 baseline before
scientific release. Cover all declared policies and
models, with every warmup/graph/selection/transfer/forward guard accounted for.
A guarded short rollout is not a complete-horizon estimate. Diagnostics cost
must cover95400common-state+2304clean forwards, or122400+2304 if the optional
physical baseline is included, plus graph/artifact overhead, with
validation/test reported separately later. If times are measured serially then
parallel evaluation is proposed, do not assume linear scaling. Every missing
cost keeps deadline fit undefined.

Forecast completion by2026-10-07 01:00UTC, leaving seven hours until08:00UTC
no-new-work cutoff for writing/review/compilation. Root must review the complete
conservative estimate and independently freeze the scientific source/protocol/
schedule before launch. Even a timing pass does not authorize science. Capacity
failure leaves all evidence intact and the whole scientific cohort deferred.
