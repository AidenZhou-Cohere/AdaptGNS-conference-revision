# D3 stopped-evaluation scalar interface

This specifies a new analysis interface for
`summarize_goop3d_graph_support_v1.py`. It changes no frozen training, evaluation,
diagnostic, capacity or D2 analysis source. A future evaluation supervisor may
produce this ledger after root reviews its implementation; this document and its
inert templates do not launch work or admit any endpoint, cohort or test source.
Methods and statistics agreed on a supervisor-candidate ledger followed by
separate root collection and analysis releases.

The module uses only standard Python libraries and the hash-pinned scalar
`schedules` and `summarize` helpers. It does not import a numerical model, execute
inference or deserialize arrays. Collection hashes NPZ files as opaque bytes and
checks saved scalar arithmetic. It deliberately cannot establish independent
tensor correctness, unrecorded full-rollout positions, conservation or speedup
from graph counts.

## Fixed study identity

The dataset is `Goop-3D`, with 301 frames, H295, D3, native radius .025, dt .0025,
type7 and bounds [.2,.8] in each coordinate. Exact manifest and metadata bytes
remain authoritative. The final cohort is the six fresh faithful models: base
and mix, each at seeds0,1,2, at one **already root-selected** prospective endpoint.
The collector checks the frozen D3 evaluator's complete-cohort gate. It accepts
neither a D2 cohort nor capacity checkpoint promotion. This study remains
distinct from Goop2D100k and WaterDrop100k plus10k continuation.

All six models require each stage below, including stages never started:

| Stage | Mode | Split |
|---|---|---|
| `full_rollout_valid` | `full-rollout` | `valid` |
| `full_rollout_test` | `full-rollout` | `test` |
| `same_state_valid` | `same-state` | `valid` |
| `same_state_test` | `same-state` | `test` |
| `clean_validation` | `clean-validation` | `valid` |

Every expected cell comes from the frozen D3 evaluator's
`schedules(complete_manifest['records'], mode, 'final_evaluation')`. Full-rollout
cells expand every schedule item in this exact policy order:
`base,dense,random25,speed25,laggedrisk25,relative-velocity-RMS25`.
The source grid is allN ifN<=30, otherwise floor(j*(N-1)/29),j0..29. Same-state
targets are7,80,153,226,300. Clean validation selects128 distinct flattened
histories from the selected sources and295 histories/source. Do not assume the
future test population has100records or assume a universal1080outcome count.

Both complete split manifests are required to establish these denominators. If
test acquisition/conversion never establishes its complete source count, retain
that infrastructure failure and report analysis unavailable. Do not manufacture
an expected test grid from a guess or report a full-study summary from validation
alone. Given established source manifests, all failed/missing evaluation cells
remain represented.

## Ledger envelope

`schema` is `adaptgns_goop3d_final_evaluation_ledger_v1`;
`producer` is `supervisor_candidate`, not `root`. Required envelope fields are:

- `dataset:"Goop-3D"`, `state:"stopped_all_owned_processes_reaped"`,
  `unreaped_owned_children:[]`, `all_pinned_inputs_reverified:true`.
- Exact `cohort_sha256`, `cohort_audit_sha256`, `protocol_sha256` and
  `source_manifest_sha256:{"valid":SHA,"test":SHA}`.
- `endpoint_updates`: the already admitted common integer endpoint.
- `stages`: all30 unique arm/seed/stage entries, in any order. The collector
  independently imposes its canonical model/stage order.

Additional envelope fields may preserve root release provenance, physical GPU
UUIDs, execution source hashes, global elapsed clocks, errors or worker receipts.
They remain intact in the collected ledger. A summary does not sum overlapping
child clocks into a claimed global wall time.

Each stage entry contains:

- `arm`, `seed`, `stage`, `mode`, `split`, `endpoint_updates`,
  `checkpoint_sha256`, all matching the final cohort and table above.
- `directory`: one distinct original absolute execution-host output directory.
  It may be absent for never-started work. Symlinks, remapped local directory
  copies and unexpected committed row files are rejected.
- `command`: exact evaluator argv, including interpreter/source and `--execute`,
  `--purpose final_evaluation`, all model/mode/split/endpoint options, `--release`,
  `--output-dir`, original `--cohort`, `--cohort-audit`, `--protocol`, `--manifest`,
  `--trainer-source` and `--checkpoint` paths. Duplicate options are rejected.
- `release_file` and `release_sha256`: the original root child evaluation
  release, prepared even for a stage that ultimately never starts. The release
  must be final-evaluation scope and include `scientific_endpoint_updates`,
  endpoint/checkpoint/model fields, frozen evaluator/graph pins and exact bound
  cohort/audit/protocol/manifest/trainer/checkpoint paths. The collector checks
  the same release in the command and saved evaluation protocol.
- `outcome`: the whole-invocation process receipt described below.
- `cells`: the **entire** exact ordered schedule for the stage, with the
  coverage fields below. Missing cells cannot simply be omitted.

## Process and cell accounting

A never-started outcome has `started:false`, `state:"never_started"`, `pid:null`
and a nonempty `termination_reason`. Its cells must all be `never_started`, and
no saved protocol may claim it ran.

A started outcome has `started:true`, `stopped_and_reaped:true`, positive integer
`pid` and `start_ticks`, nonempty `hostname`, exact `command`, integer `exit_code`,
nonnegative finite `elapsed_seconds`, positive finite `outer_timeout_seconds`,
`absolute_stop_utc`, a `signals` list and nonempty `termination_reason`.
Preserve raw start/end times, raw physical GPU identity, CPU/RSS and observer lag
as extra fields. The future supervisor owns actual identity matching, signal
delivery and reaping; the scalar collector does not inspect or signal processes.
Whole-child time must cover the summed committed synchronized calls and
publication durations. Failed/interrupted attempts and quota termination remain
distinct in these receipts, even when no numeric row was committed.

Each cell copies its full expected schedule item (including source index,
trajectory ID, particles/size-group for rollout, or schedule index/target frame
for diagnostics), plus `policy` for full rollouts. Its `state` is exactly one of:

| State | Required additional fields |
|---|---|
| `completed_required_outcome` | `row_file`, `row_sha256`, `artifact_sha256`, `failure:null` |
| `recorded_failed_outcome` | Same file/hash fields; `failure` exactly equals the saved row's failure object |
| `timed_out_current` | Nonempty `reason`; no committed row |
| `not_completed_before_invocation_end` | Nonempty `reason`; no committed row |
| `never_started` | Nonempty `reason`; no committed row |

The row basename is `trajectory_IIIIII_POLICY.json` for a full rollout or
`trajectory_IIIIII_target_TTT.json` for a diagnostic. Its NPZ has the same stem.
D3 always uses `artifact_file`, `artifact_sha256`, `publication_seconds`;
do not substitute D2's `trace_*` fields. Rows bind their exact saved protocol.
Every output file, including a failed attempt, orphan NPZ or temporary artifact,
is retained and byte-hashed. An orphan artifact does not count as a committed
outcome. Saved `result.json` / `summary.json` may be stale only when they exactly
match a prefix of the committed rows; this is recorded explicitly.

## Root collection release and command

The separate root release has schema
`adaptgns_goop3d_scalar_collection_release_v1`, `issued_by:"root"`,
`status:"approved_stopped_scalar_collection"`, `collector_sha256` equal to the
reviewed new module bytes, and `files_sha256` mapping **exactly** these six
original absolute paths to their hashes: ledger, final cohort, complete cohort
audit, complete validation manifest, complete test manifest and scientific
protocol. The template is intentionally unadmitted and incomplete.

Use this form on the host retaining those original paths after stopped-output
review; capitalized paths are placeholders, not an executable release:

```text
PYTHON summarize_goop3d_graph_support_v1.py --execute --mode collect
  --ledger ORIGINAL_LEDGER --cohort ORIGINAL_COHORT
  --cohort-audit ORIGINAL_COHORT_AUDIT --valid-manifest ORIGINAL_VALID
  --test-manifest ORIGINAL_TEST --protocol ORIGINAL_SCIENTIFIC_PROTOCOL
  --root-release ROOT_COLLECTION_RELEASE --output FRESH_COLLECTION_JSON
```

The collector preserves all stage/cell/process records, raw scalar rows,
diagnostic summaries, exact source schedules, physical metadata/truth references,
original input bindings, child release hashes and every output-file hash. It
verifies all fixed source pins before and after collection. No checkpoint tensor
is deserialized; checkpoint identity is bound through the reviewed cohort,
complete checkpoint audit and child releases. This is explicitly narrower than
an independent tensor/source audit.

## Root scalar analysis release and outputs

Schema is `adaptgns_goop3d_scalar_analysis_release_v1`; use `issued_by:"root"`,
`status:"approved_fixed_scalar_aggregation"`, exact `summarizer_sha256`,
`collection_sha256`, `cohort_sha256` and `endpoint_updates`.

```text
PYTHON summarize_goop3d_graph_support_v1.py --execute --mode summarize
  --collection FROZEN_COLLECTION_JSON --root-release ROOT_ANALYSIS_RELEASE
  --output FRESH_SUMMARY_JSON
```

This second command may run on a local scalar copy because it binds that complete
collection's bytes. It does not remap the recorded original provenance paths or
read test arrays. Every output uses exclusive creation; retain all unsuccessful
attempt logs and previous outputs rather than overwrite them.

Full-rollout means require all declared sources for that seed/policy; three-seed
means/sample SD require all three seed values. Missing/guarded cells remain null.
Pointwise forecast200 can remain defined for an accepted prefix whose later
H295 failed. Physical boundary and graph/cap means obey the full-H completeness
rule; failed prefixes remain separately labelled. Exact truth-boundary scalars
must agree across model/policy records for each source and step.

Diagnostic aggregation reuses the frozen D3 function: within frame, equal frames
within source, equal sources, then three ordered training seeds. Undefined
correlations and their reasons remain null; they are not averaged away. The new
summary adds same-state graph/cap/boundary aggregates and checks paired
model-independent graph selections across arms. Previous-observed-base risk is
distinct from autonomous `laggedrisk25`.

For both autonomous and same-state accuracy, report each arm's policy contrasts
against base, random, speed and relative-velocity RMS; all overall mix-minus-base
policy effects; and the explicit interaction
`(mix-trained risk - mix-trained random) - (base-trained risk - base-trained random)`.
An improved interaction or improved graph-exposure model alone does not establish
that residual-risk ranking beats those controls. Retain every seed sign.

Whole-invocation, committed-call/publication and per-policy clocks remain
separate. Same-state repeated timing includes its extra score-generation pass,
and all raw warmup/parity/repeated-call records remain in the collection. Neither
parallel clock sums, fixed-order autonomous durations nor edge counts establish
speedup. The deterministic source-order grid is not a probability sample; these
outputs make no significance, pristine-confirmation, conservation or conference-
readiness claim.
