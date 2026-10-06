# Sand graph-support capacity measurement protocol v1

**Infrastructure only; execution requires a separate root release.** This protocol is separate from the frozen
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

For the prepared six-policy evaluation scope, full-rollout cost must cover1080
completeH314 outcomes, including relative-velocity-RMS25. Cover all declared policies and
models, with every warmup/graph/selection/transfer/forward guard accounted for.
A guarded short rollout is not a complete-horizon estimate. Diagnostics cost
must cover122400common-state+2304clean forwards, including the physical baseline,
plus graph/artifact overhead, with
validation/test reported separately later. If times are measured serially then
parallel evaluation is proposed, do not assume linear scaling. Every missing
cost keeps deadline fit undefined.

Forecast completion by2026-10-07 01:00UTC, leaving seven hours until08:00UTC
no-new-work cutoff for writing/review/compilation. Root must review the complete
conservative estimate and independently freeze the scientific source/protocol/
schedule before launch. Even a timing pass does not authorize science. Capacity
failure leaves all evidence intact and the whole scientific cohort deferred.

## Per-host worker and combined verification

Freeze `measure_sand_graph_support_capacity.py`, its tests and this protocol
before graph-support capacity outcomes. The worker privately imports the exact
unchanged `measure_sand_cuda_capacity_v2.py` lifecycle source atSHA256
`c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd`.
It reuses its launch/observe/owned-process cleanup/reap machinery. It substitutes
only this worker's private schedule, graph-run verifier, paired-ledger verifier
and fixed command with `--arm`. No shared file/module used by another process is
modified. The trainer definition source is pinned at
`fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124`.
No model or CUDA module is imported by the supervisor.

Host roleA executes its four-job wave once; host roleB executes its two-job wave
once. Root launches them close together using the approved transport. The worker
contains no SSH/network credentials or transport logic. A failure on one host
cannot automatically launch/retry/cancel work on another host. Root reviews both
process inventories before further action. All attempts and artifacts persist.

Each host needs a distinct root release with schema
`adaptgns_sand_graph_support_capacity_release_v1`, status`admitted_for_timing`,
issued_by`root`, scientific_training_admitted=false, shared nonemptycohort_id,
fixedhost_role A/B, complete6-job schedule, its four explicit GPU UUIDs, exact
source/interpreter/data/protocol/numerical-gate hashes, required environment and
review rationale. The root process identity check must be at most300seconds old.
The release includes a root-attested UTC clock-error bound between0 and5seconds,
backed by contemporaneous host/root clock checks. Retain those checks.

The separate graph-support numerical gate must have schema
`adaptgns_sand_graph_support_cuda_numerical_admission_v1`,
status`admitted_for_timing_only`, issued_by`root`, the exacttrainer_sha256 and
scientific_training_admitted=false. The previous base-only Sand assessment does
not satisfy this gate. Numerical reports/failures and graph-support same-CUDA
base semantics/replay evidence remain separately preserved.

Both successful workers require the fixed512checkpoint/status/history plus a
single matching attempt, no unsuccessful artifacts,512 graph ledger rows with
correct arm/seed/coin/RNG/budget/noise identities and exact scalar/graph particle
and frame agreement. Base/mix pairs require identical frame/noise/native-graph/
coin/LR evidence. Initial checkpoints remain available for root tensor-identity
auditing before scientific admission; the scalar worker never deserializes them.
Raw NVIDIA/PyTorch UUID strings remain recorded; strict full UUID normalization
permits the observed optional literal`GPU-` prefix only.

Combine complete copied worker roots using this same pinned supervisor and
trainer definitions. Recheck every input snapshot/release, worker final-status/
summary hash, wave membership/process/UUID observations, checkpoint bytes and
all512 rows. Require distinct hostnames and eight distinct inventory UUIDs; both
roles must sharecohort_id, protocol, data, core/trainer/supervisor/numerical gate
hashes and declared numerical runtime versions. Within each host require at least
two all-job GPU observations in updates64..511. Shrink each observed steady
interval inward by its root-attested clock-error bound, then require the two
intervals overlap. This establishes actual cross-host overlap under the stated
clock bound; it does not assert constant six-way concurrency through every
update of every steady window. Preserve launch skew and all observations.

Without a complete cost JSON the combined summary reports only
`1.35*(100000*q6+12*r6)` and leaves training-with-ledger, total cost and deadline
fit null. A cost estimate with schema
`adaptgns_sand_graph_support_cost_estimate_v1` must declare complete_workload_estimated,
policy_count6,1080outcomes,H314,124704diagnostic forwards, positive
full_rollout_seconds/diagnostics_execution_seconds/ledger_total_reserve_seconds,
a concreteledger_bound_rationale and preservedtiming_evidence_sha256 values.
This is root-reviewed planning evidence, not automatic scientific authorization.

CLI (placeholders are not a launch authorization):

```text
python measure_sand_graph_support_capacity.py --execute --host-role A
  --release ROOT_HOST_A_RELEASE.json --repo FROZEN_REPOSITORY
  --trainer train_sand_graph_support_cuda.py
  --lifecycle-source measure_sand_cuda_capacity_v2.py
  --train-manifest TRAIN.json --admission ADMISSION.json
  --structural-report STRUCTURAL.json --protocol THIS_PROTOCOL.md
  --numerical-gate ROOT_GRAPH_NUMERICAL_GATE.json
  --python ABSOLUTE_HOST_PYTHON --output-dir FRESH_HOST_OUTPUT

python measure_sand_graph_support_capacity.py --combine COPIED_A_ROOT COPIED_B_ROOT
  --trainer train_sand_graph_support_cuda.py
  --lifecycle-source measure_sand_cuda_capacity_v2.py
  --summary-output FRESH_COMBINED_SUMMARY.json
```

HostB uses the same fields with its own release/output and`--host-role B`.
Optionally add`--costs FROZEN_COMPLETE_COSTS.json` to combination. Keep every
forecast snapshot in a fresh file and preserve all adverse evidence.

The earlier scientific draft's five-policy fallback remains a preserved proposal.
The currently prepared graph-support evaluation adapters require six policies, so
this complete-cost gate rejects old five-policy900-outcome/97704-call estimates.
Any future fallback would need an explicit prospectively reviewed source/protocol
version before scientific launch; it cannot be inferred from missing costs.

A scalar verification portability correction is also prospective here. In the
completed original ARM/Linux capacity summary, learning-rate values atsteps16and274
recompute on the Mac at exactly onefloat64ULP difference; all1536saved paired rows
remain exactly equal. Preserve that independent-audit discrepancy. This graph
supervisor therefore permits at most onefloat64ULP when recomputing the expected
learning-rate formula on another platform, records every discrepancy and keeps
raw history/stdout/paired-arm LR comparisons exact. Only a temporary verifier
copy normalizes the expected LR before reuse of the frozen scalar-timing helper.
No training rate, checkpoint, optimizer tolerance or scientific source is changed.
