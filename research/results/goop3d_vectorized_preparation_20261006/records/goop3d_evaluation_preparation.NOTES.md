# Separate Goop-3D evaluation and deadline preparation

Prepared 2026-10-06. This is implementation preparation, not an execution or
scientific endpoint admission. Root owns actual data access, device/process
control, bounded timing releases, endpoint selection, final cohort review and
publication. No actual training arrays, reserved test, CUDA device or network
source was accessed while preparing these files. The official validation scalar
manifest was read to derive the prospective indices below.

## Reviewed evaluator

| File | SHA-256 |
| --- | --- |
| `evaluate_goop3d_graph_support_v1.py` | `9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de` |
| `goop3d_native_evaluation_v1.py` | `a742123093aff433f3a4e302929a52bae4f5fee9610195d86df726852575b1d5` |
| `goop3d_diagnostic_metrics_v1.py` | `5ef3de96e470eea495dd56c1f60396bbd166b9ea5c5bc340864715cb9e2c173b` |
| `test_evaluate_goop3d_graph_support_v1.py` | `bdb8dec03146bb572596b38ac60a6a1f547ffa18dd8c312092d17dd530c7b79d` |
| `goop3d_evaluation_independent_review_methods.json` | `d4874cb6443609c8c233d54194de56601f50dec12faafef9394223e6f508de4c` |

All 21 targeted tests passed locally in 3.74 seconds and independently in 3.69
seconds. Tests use tiny synthetic CPU models and mocked driver publication;
they do not assert actual CUDA or official-data success. The two review findings
were corrected before these hashes: all release, common/numerical source,
context/auxiliary and data bindings are carried through final verification;
CUDA index and physical UUID must match the root release and observed runtime.
Mutation fixtures preserve and reject outcomes after any required source changes.

The separate D3 native and diagnostic modules explicitly copy the needed frozen
implementations and use isolated helper namespaces. No frozen 2D training,
rollout, graph, protocol or diagnostic file was modified. Radius is 0.025, radius
factor 1.267, dimension three, time step 0.0025, and the official bounds are
[0.2, 0.8] in all coordinates. The graph retains the native directed cap128/self
base prefix, with an uncapped symmetric optional suffix. Guards remain 2 million
unordered candidate pairs per example, 5 million directed edges and absolute
state magnitude 10. A physical boundary excursion is a diagnostic; it is not
silently relabeled as that numerical state guard.

## Prospective evaluation schedules

The complete source is always verified before selecting evaluation records.
For a source with N records, use all records when N <= 30; otherwise use
`floor(j*(N-1)/29)` for j=0,...,29. This deterministic source-order grid is not a
probability sample or a claim of representativeness.

The actual validation source has 100 records. Its fixed final grid is:

```
0,3,6,10,13,17,20,23,27,30,34,37,40,44,47,51,54,58,61,64,68,71,75,78,81,85,88,92,95,99
```

Each trajectory contains 301 positions. Six observed frames leave H=295
autonomous predictions, whose targets are positions 6 through 300. Do not
substitute the H=294 convenience-path horizon. All six final policies are:
base, dense, random25, speed25, laggedrisk25 and relative-velocity-RMS25.
The quarter-budget policies select exactly the declared optional-pair budget.
Full rollout carries each policy's own predicted history. Lagged risk caches
the previous autonomous output, with its initial base warmup retained in timing.

Final same-state diagnostics use targets 7,80,153,226,300 on each selected
source. Both the current and previous histories are observed. The risk policy
is named `previous-observed-base-risk25` here to distinguish it from autonomous
lagged risk. Its standalone timing includes the previous-base scoring pass.
One warmup plus seven cyclic orders cover the six policies and a separate
natural-base timing reference. Repeated graph/output consistency, signed
sparse/dense benefits, negative-benefit fractions and risk-vs-benefit
correlations are retained. Future targets are absent from policy inputs.

Clean validation uses 128 evenly spaced histories in the flattened selected
validation grid (`floor(k*(295*Nselected-1)/127)`, k=0,...,127), followed by equal
frame means within trajectory and equal trajectory means. For normalized
vector squared error e and coordinate variance q, coordinate MSE is mean(e)/3,
predicted vector squared error is 3*mean(q), and constant-free Gaussian NLL is
mean(0.5*e/q + 1.5*log(q)). These three-dimensional metrics must not reuse the 2D
coefficient or denominator.

Full-rollout traces include forecast steps 1,10,50,200,295 when completed. A
defined pointwise forecast200 prefix after later failure remains a prefix;
it cannot substitute for a complete H295 mean. All guard/execution failures,
boundary arrays and incomplete prefixes remain in their original artifacts.
Missing seed/trajectory/policy cells must produce undefined paired aggregates,
not a smaller success-only sample.

## Cheap bounded timing before scientific endpoint selection

Use the reviewed six-model 512-update capacity checkpoints, never a final-model
schema. Prospectively time the base seed0 and mix seed0 capacity checkpoints.
For each arm, execute three separate validation invocations:

1. `full-rollout`: all six policies at H295 on validation indices 37,51,64.
2. `same-state`: target7 on validation indices 37,51,64.
3. `clean-validation`: 128 evenly spaced histories within those three sources.

The three timing indices are the smallest, lower-median and largest particle
counts **within the already fixed final validation grid**, restored to source
order: index37 has 2,588 particles, index51 has 12,699 and index64 has 7,004.
The timing choice uses scalar source counts, not errors or rollout behavior.
Full timing yields 18 complete full-H outcomes per arm. A guard failure means
the full-horizon forecast is unavailable; do not extrapolate the surviving
prefix or silently replace the source/policy.

The checkpoint schema must be
`adaptgns_goop3d_vectorized_capacity_checkpoint_v1`, with purpose
`bounded_capacity_only_never_promote`, scientific training false, completed
steps512, probe LR horizon100000, exact arm/seed/trainer/graph/train-manifest
bindings, and no scientific training/run configuration. The probe LR horizon
does not select the final scientific endpoint.

## Root invocation contract

The entry is description-only unless `--execute` is passed. Each actual
invocation requires these arguments, in addition to fresh `--output-dir`:

```
--purpose capacity_timing|final_evaluation
--mode full-rollout|same-state|clean-validation
--split valid|test --arm base|mix --seed 0|1|2
--checkpoint PATH --checkpoint-sha256 SHA --checkpoint-updates UPDATES
--cuda-index INDEX --gpu-uuid PHYSICAL_UUID --threads 2 --max-seconds LIMIT
--release ROOT_RELEASE --repo ADAPTGNS_REPO
--manifest COMPLETE_SPLIT_MANIFEST --split-admission ROOT_SPLIT_ADMISSION
--structural-report REPORT --acquisition-report REPORT
--context-semantics REVIEW --auxiliary-report CENSUS
--trainer-source train_goop3d_graph_support_cuda_v2.py --protocol PROTOCOL
```

Final evaluation also requires `--cohort` and `--cohort-audit`. Test additionally
requires `--cross-split-audit`. Clean validation accepts only the valid split.
The root must wrap the **whole invocation**, including input verification,
model loading, publication and final checks, in an externally enforced timeout.
The inner alarm covers individual inference calls and is not a process-level
substitute. The entry refuses remapped `CUDA_VISIBLE_DEVICES`, requires the
reviewed deterministic CUDA environment, and stops at the 2026-10-07 01:00 UTC
compute/analysis cutoff.

The release schema is `adaptgns_goop3d_evaluation_release_v1`, status
`admitted_for_execution`, issued by root. It binds purpose/mode/split/arm/seed,
checkpoint hash and updates, evaluator/graph hashes, CUDA index/UUID,
max_seconds and `whole_invocation_outer_timeout_required: true`.
`files_sha256` maps absolute paths to their exact bytes for every supplied
data/evidence/checkpoint/protocol/trainer file, plus the native, metric and graph
adapters, cohort/audit and test overlap audit when applicable.
`numerical_source_sha256` must bind the four original pinned files below; the
entry also binds all trainer common-core dependencies and all official context
source and auxiliary-checker files through completion.

| Original dependency | SHA-256 |
| --- | --- |
| `research/graph_convention_bridge.py` | `2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d` |
| `research/full_rollout.py` | `b0a37ee47619e699298b86865649e63c402cc1cb5dc2fa3dfd4055f7fa966eb8` |
| `research/full_same_state.py` | `ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592` |
| `research/budget_graph.py` | `951f0d13863672f1dd248febf8cc960ba500103ca95361e06c9b0577913a8187` |

Capacity timing additionally sets scientific training false. Final evaluation
requires a separate root-selected prospective endpoint in the release and a
six-checkpoint D3v2 cohort. The cohort schema is
`adaptgns_goop3d_graph_support_final_cohort_v2`, status
`frozen_for_final_evaluation`; the audit schema is
`adaptgns_goop3d_graph_support_complete_cohort_audit_v2`, status
`all_six_endpoints_and_pairing_verified`. Every base/mix seed0/1/2 endpoint,
all optimizer steps, full graph history, source/data/protocol hashes, initial
paired tensors/RNG and all sample/noise/LR/graph schedules must be audited.
Neither old 2D schemas nor capacity checkpoints pass this gate. It precedes
even hashing the test manifest.

The split admission schema is
`adaptgns_goop3d_evaluation_split_admission_v1`, status `admitted`, issued by
root. It binds the complete manifest, acquisition/structural/auxiliary reports,
metadata, context review, converter/source hashes, actual record count/type7,
T301 and D3. Test needs future complete source verification plus
`adaptgns_goop3d_all_split_integrity_audit_v1`, all three manifest hashes and no
duplicate pairs. Its actual record count must be discovered honestly. The
final source-order grid rule applies after whole-source verification.

## Scalar deadline worksheet

| File | SHA-256 |
| --- | --- |
| `goop3d_deadline_worksheet_v1.py` | `8a4946e813a6b9ef3d0cb86c5649d614d2bc89fd08b9962bf3ec877e94a20db6` |
| `test_goop3d_deadline_worksheet_v1.py` | `034010e0121f029c076402577aad818685e3e6ff620833730caeb2dfa875bb4f` |
| `goop3d_deadline_worksheet_independent_review_methods.json` | `a4a20cd32b4b522eec8e80bfe33994a8fcad2bbd87d85aa4cd9476088185e21d` |

All 11 scalar/synthetic fixture tests passed locally in 0.36 seconds and in the
independent methods review's 59-test capacity-plus-worksheet run in 3.36
seconds. This review does not repeat the capacity supervisor's checkpoint
tensor/Adam audit; the worksheet trusts the root-bound verified capacity
summary. Late timing-artifact mutation is tested to leave no successful
forecast. Every timing artifact and the evaluator source is rehashed at the
end of scalar collection.

`goop3d_deadline_worksheet_v1.py` uses only scalar JSON and byte hashes. It
deserializes no tensors or numeric arrays and reads no test files. It requires
the verified complete six-model capacity summary and the six stopped timing
invocations above. The root timing inventory schema is
`adaptgns_goop3d_capacity_timing_inventory_v1`, status
`all_required_processes_stopped`, issued by root. Each entry identifies arm,
mode, directory and a `files_sha256` map covering protocol/status, every row
JSON and every NPZ. Its `external` receipt contains pid, exit_code0,
stopped_and_reaped true and measured whole-process elapsed_seconds.

The scalar planning configuration schema is
`adaptgns_goop3d_deadline_planning_config_v1`. It contains a candidate_endpoints
integer list, selected_endpoint null, evaluation_overlap_credit false,
checkpoint_every, runtime_multiplier >=1,
transfer_review_analysis_reserve_seconds, and planning_start_utc. The worksheet
release schema is `adaptgns_goop3d_deadline_worksheet_release_v1`, status
`admitted_for_scalar_planning`, issued by root, scientific_training_admitted
false, with exact absolute-path hashes of all inputs and worksheet source.

For each candidate U, training core time is U*(q4+q2). Each checkpoint save is
conservatively charged the entire observed capacity non-kernel residual in
each wave; this deliberately overcharges startup. Evaluation uses the maximum
observed complete trajectory duration for each full-rollout policy and maximum
per-frame diagnostic duration, all including publication. It serially budgets
all six models, 30 validation plus at most30 future test sources, five
same-state targets/source and128 clean validation frames/model. Whole-process
setup/residual cost is included for each final invocation. No speedup is
credited for unmeasured evaluation concurrency. The explicit multiplier and
transfer/review/analysis reserve are then added.

These are engineering estimates, not guarantees or confidence bounds. A
512-update seed0 checkpoint, final checkpoint, other seed and future test
geometry can differ in runtime. The worksheet never chooses an endpoint or
admits training; root must prospectively select one based on time/source
evidence and launch all six final models from scratch. Any missing, failed or
incomplete timing input leaves the forecast unavailable with its reason.

## Analysis and manuscript boundaries

Treat the training seed as the replicate for paired arm/policy effects. Report
all three seed contrasts and their mean/sample SD; do not use particle or time
step counts as independent sample size. A key graph-exposure interaction is
`(risk-random)_mix - (risk-random)_base`, with negative values favorable for an
error metric. Aggregate each complete source equally within a seed; keep
validation and test separate. Retain undefined contrasts when a required
cell fails. Mechanism diagnostics concern common observed states and do not
prove autonomous rollout gains.

Do not merge this D3 study with historical arrays, compact pilots, WaterDrop,
Sand, Goop2D, infrastructure512 checkpoints or unsuccessful capacity attempts.
This deadline study is a prospectively fixed extension after prior inspection
of other evidence, not pristine independent confirmation. Conference readiness
still requires actual evidence, full failure accounting, manuscript checks and
author verification.
