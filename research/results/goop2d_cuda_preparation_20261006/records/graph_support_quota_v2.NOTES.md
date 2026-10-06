# Separate quota execution basis for Sand and Goop

These two entry points describe their contracts by default. Root owns actual
capacity review, releases, process/clock checks, all model/data/CUDA launches,
the six-model endpoint audit, and final scientific admission. Nothing here
changes the frozen trainers, the 100000-update endpoint, the v1 timing evidence,
or the manuscript. Sand v1's incomplete full-horizon timing forecast remains
undefined/refused. A quota is permission to spend bounded time, not a throughput
estimate or a promise of complete evaluation.

## Training

`supervise_graph_support_science_quota_v2.py` requires `--execute`, `--dataset`
(`Sand` or `Goop`), `--host-role`, and every path shown by `--help`. In addition
to the v1 scientific arguments, pass `--v1-source` pointing to the unchanged
`supervise_sand_graph_support_science.py`. Goop also requires
`--acquisition-report`, `--context-semantics`, and `--auxiliary-report`.
The Goop context source directory is resolved by the pinned Goop capacity helper.

The `--forecast` file is the actual complete `all_six_verified` capacity summary
from the dataset's pinned capacity combiner. Its existing evaluation forecast
may remain null. The wrapper checks all six measured jobs and their q6/r6.
The new `--costs` allocation has schema
`adaptgns_{sand|goop}_graph_support_execution_allocation_v2`. See the nonadmitted
templates; root must supply the measured training-ledger reserve, its rationale
and evidence hashes and issue the allocation prospectively.

The training budget is

`1.35 * (100000 * q6 + 12 * r6) + ledger_total_reserve_seconds`.

The subsequent budget is `11760 + 2700 + 3600` seconds: one concurrent
evaluation stream allocation, cohort/preflight verification, and analysis.
Training cleanup begins before the released training stop by the reviewed
clock error plus the unchanged 15-second cleanup allowance. The exact latest
start must fit the training budget before that cleanup trigger. The second
launch-fit check occurs after input copying and before any child.

The scientific release schema is
`adaptgns_{sand|goop}_graph_support_scientific_release_quota_v2`, with status
`admitted_for_scientific_training`. It additionally declares `dataset`, the exact
`cost_basis` string in the templates, `all_required_evaluation_outcomes_promised:
false`, and `v1_timing_gate_reinterpreted: false`. All other root host, clock,
process, schedule, cohort and byte bindings retain the v1 meaning. In particular,
`files_sha256` contains distinct `supervisor` (new wrapper) and `v1_source`
(unchanged original) entries. Goop includes all three additional evidence files
and the four official context source files. The allocation hash also binds the
reviewed evaluation quota supervisor hash.

The v1 training process loop, cleanup, initial pairing and whole-history pairing
are privately imported unchanged. Endpoint checks are dataset-specific: Goop
uses frame range 6..400, type 7, Goop payload/run-config keys, and complete
official source/context provenance. Sand specialization retains the v1 endpoint
and history logic; shared numerical checkpoint/Adam/RNG checks are identical.
Full endpoint checks happen after all local children are reaped. The initial
pairing audit remains after training; no before-step-1 barrier is claimed.

## Evaluation

`supervise_graph_support_evaluation_quota_v2.py --execute` requires only
`--release`, `--lifecycle-source`, and a fresh `--output-dir`. The release supplies
argv arrays for the pinned final evaluators, not shell strings. Each argv begins
with the absolute Python and evaluator paths followed by `--execute`; do not
insert `-u` in this evaluator argv. Every file argument and both executables
must be pinned by absolute path in the release's `files_sha256` map. Repository
and output directories are directories, not file-hash entries. Evaluator input
contracts continue to verify source arrays and all scientific admissions.

The schema is `adaptgns_graph_support_evaluation_quota_release_v2`, status
`admitted_for_execution_allocation`, issuer `root`, and dataset `Sand` or `Goop`.
Use the exact `environment`, quotas, A/B mapping, and flags checked by the source.
Fresh root process/clock evidence supplies the actual hostname, all four GPU
UUIDs, `clock_error_bound_seconds <= 5`, and `process_clock_checked_utc`.

Each model stream performs these distinct stages once, in this order:

| Stage key | Evaluator mode / split | Whole-process quota |
|---|---|---:|
| `full_rollout_test` | `full-rollout` / `test` | 7200 s |
| `same_state_valid` | `same-state` / `valid` | 1800 s |
| `same_state_test` | `same-state` / `test` | 1800 s |
| `clean_validation` | `clean-validation` / `valid` | 900 s |

Each child has at most 15 seconds of cleanup allocation after its quota.
The per-stream total is 11760 seconds. Host A runs base0/mix0/base1/mix1 on
GPUs 0/1/2/3; host B runs base2/mix2 on GPUs 0/1. Each stage's output must be
`OUTPUT/jobs/ARM_seedSEED/STAGE`. All stages share their frozen checkpoint and
all streams share one frozen cohort/protocol. Goop test commands additionally
require the cross-split audit.

`outer_processing_reserve_seconds` means work still required outside child
invocations at queue launch. Root must explicitly account for it. Ordinarily,
after the separately reserved 2700-second cohort/preflight work has finished,
3600 seconds of analysis remain. Do not charge an already spent cohort reserve
a second time. Queue setup must still finish before the released latest start.
The queue checks remaining quotas plus cleanup before every distinct stage;
the active global cutoff also preserves this outer reserve. Compute/analysis
deadline remains 2026-10-07 01:00 UTC.

The quota covers the complete evaluator lifetime, including setup, array/model
loading, inference, artifact I/O and child verification. Process ownership is
checked before signaling. SIGINT/TERM/KILL escalation occurs at quota, +5 and
+10 seconds; an unreaped child at +15 stops admission of further stages.
Ordinary OS scheduling and polling resolution affect observation times, so
reported invocation duration is an upper bound. Read-only GPU observations run
asynchronously so their subprocess timeout cannot block the child monitor.

A verified quota stop may be followed by the next distinct planned stage.
Nonzero exits without an initiated quota stop, explicit implementation/parity
errors, changed inputs, foreign GPU work or a lost reserve require root review.
No retry occurs. Fixed stage/cell order can select which cells finish under the
quota; completed subsets must not become survivor-only headline estimates.

`process_outcomes.json` retains actual launches, signal history, worker failures
and unreaped children. `coverage_ledger.json` distinguishes committed complete
rows, committed failed rows with original failure details, timeout-current,
missing and never-started cells. Missing work is not relabeled a model guard
failure. If an owned child is unreaped, coverage auditing is deferred. Malformed
rows or changed inputs retain an explicit failed final status. Queue coverage
checks JSON identity/status/horizon only; it computes no MSE and does not replace
root's scientific, referenced-array or complete-cohort audit.

## Preparation verification

The evaluation supervisor has 27 passing mocked/scalar tests and an independent
statistics review. The training adapter has 56 passing tests covering both
datasets, quota arithmetic/refusals, Goop lineage, tiny CPU payloads, frame
bounds, private v1 binding and AST preservation. These are preparation checks;
they are not model training, capacity measurements or evaluation results.
