# Proposed D3 observed-history analysis

Status: source-only proposal for root review. No phase, clock, release, or
allowance is issued. Root can admit a separate bounded analysis under the
human's standing scope after Sand priorities and fresh feasibility checks.
The original D3 analysis hour remains expired and immutable.

## Scientific target and fixed comparison grid

Assess whether graph-mixture training improves one-step behavior on the
original observed histories, and whether it changes the benefit of risk-based
selection relative to equal-budget random selection. This could add a useful
conditional mechanism result while preserving the incomplete autonomous study.
It does not answer whether either system is stable when rolled out autonomously.

Fix the numeric scope to all six original 25,000-update models: arms `base` and
`mix`, each at training seeds 0, 1, 2. Include both original `same_state_valid`
and `same_state_test` stages, and the six `clean_validation` stages as a
separately labelled normalization/calibration diagnostic. Do not add a clean
test stage or any new data, model, graph policy, seed, or target frame.

The source schedule requires 30 fixed source trajectories per split, with
same-state target frames 7, 80, 153, 226, 300. Therefore the same-state grid is
`6 models × 2 splits × 30 sources × 5 frames = 1,800` rows. Clean validation has
`6 × 128 = 768` predetermined histories, for 2,568 observed-history cells over
18 stages. The declared full grid has another 2,160 autonomous cells, for the
accepted total of 4,728. These counts follow the frozen schedule and accepted
full denominator, not any new inspection of results. The new driver must
derive and verify these exact schedules from the SHA-bound collection before
numeric reads; any disagreement stops the invocation.

Require all seven diagnostic call cases in each complete same-state row:
`base`, `dense`, `random25`, `speed25`, `relative-velocity-RMS25`,
`previous-observed-base-risk25`, and `natural_base_reference`, including the
frozen warmup/repetition/parity obligations. Only the first six are policy
contrasts; the natural-base case is the parity reference.

Completeness is checked for the entire predeclared family and all three paired
seeds. Existing missing or failed cells remain in the accounting. A family
with an incomplete required grid receives a null full-family comparison; do
not replace it with the intersection of surviving cells. No policy or seed
selection may depend on the observed error values. Validation, test, and clean
validation remain separately reported, with prior test exposure disclosed.

## Exact retained evidence

The sole data-selection authority is the previously accepted stopped collection:

```text
/root/repos/AdaptGNS-cuda-20261006/goop3d_final_analysis_20261006_v1/stopped_collection.json
SHA256 616c612724f62d21bcb521c6163e83d9038b34fdd8012980d6ed55f4b05bed5f
```

Its existing root review is SHA256
`ff50a7d4a53bae8913c9fe4f12c66364904906fc3f7cfb0e3ae5d8963eb47401`;
independent collection review is
`afc6e4c96f6aec0bf5e322d04d62bfae0817953e210d7d04f4302cabcd7d9f65`.
The new receipt must bind both, the six-model cohort
`645343fc2a1c6ef0a82e212e351b03b1a4d081702d3c8e2187b244a62c6b9bef`,
and the frozen validation/test manifest hashes already in the collection.
No source manifest, original trajectory array, checkpoint, or model needs to
be opened again for this scoped audit; their prior admission remains inherited
and explicitly limited.

Under the original queue
`/root/repos/AdaptGNS-cuda-20261006/goop3d_final_evaluation_20261006_v1`,
permit only these data paths derived from the verified collection:

```text
jobs/{base,mix}_seed{0,1,2}/{same_state_valid,same_state_test,clean_validation}/protocol.json
jobs/{model}/{stage}/trajectory_{source_index:06d}_target_{target_frame:03d}.json
jobs/{model}/{stage}/trajectory_{source_index:06d}_target_{target_frame:03d}.npz
```

The second and third patterns are not open-ended glob authority: enumerate
only committed row basenames and their exact sibling artifact names from the
fixed 2,568-cell grid and accepted `files_sha256` inventory. Validate all 30
stage identities, all 4,728 original cell states/reasons, invocation outcomes,
and declared tree membership from the collection. Re-read protocol and row
bytes only for the 18 observed stages, and numeric NPZ bytes only for their
committed rows. No autonomous result file, autonomous NPZ, uncommitted leftover,
child log, original audit output, or unexplained path may be opened. A missing
required retained observed file is an integrity failure, distinct from a cell
that was already missing in the original collection.

For every permitted file, require its exact accepted length and SHA256,
no-follow path traversal, ordinary regular-file status, and stable metadata
through the read. Bind row-to-cell-to-NPZ hashes and protocol/schedule/model
identity. Stream file hashing; process and release one archive at a time.
Inspect ZIP member names, duplicate entries and uncompressed sizes before
loading with `allow_pickle=False`; extract no files. Apply explicit caps in the
new frozen source and preserve any cap failure. A proposed implementation cap
is 512 MiB for the collection, 16 MiB per JSON, 8 GiB compressed/decoded total
per NPZ, and a 32 GiB address-space limit for the two-thread CPU worker. These
are rejection ceilings, not evidence that the actual inputs fit. Changing
them after a failure would require a new proposal, not a same-invocation retry.

Before successful publication, rehash the collection, source files and all
permitted data files actually used. Fresh integrity is claimed only for this
listed subset; unchanged bytes across the entire autonomous queue are not
recertified. Original full-queue provenance remains the accepted collection.

## Arithmetic and output claims

Reuse the reviewed NumPy-only diagnostic arithmetic, preserving tolerances and
failure handling: observed histories, targets and particle types; saved
prediction/risk/raw-risk triplets; variance floor; repeated-call/parity
arithmetic; residuals; one-step error; signed benefits; correlations; geometric
boundary diagnostics; timing aggregation; and clean-normalized error/NLL.
Keep one-step position-coordinate MSE as the primary error. Normalized errors
remain conditional on the saved normalization statistics, whose truth is not
independently re-established without the checkpoint.

Also preserve the existing pure graph/action and pairing checks: supplied
edges, exact optional budget, no duplicates, canonical pairs, bidirectional
optional edges, saved ranking/selection evidence, and matching histories and
targets. Preserve the same-state float64 target and clean float32 target raw
hashes separately; any shared float32 target identity requires the original
lossless roundtrip. Model-independent policies must use the same graph and
draw across arms. The risk policy depends on each model, so do not require its
selected graph to agree across arms or describe its contrast as a common-graph
effect.

For each split and policy p, compute equal-frame then equal-trajectory means
`L[a,s,p]` only with the full declared denominator. The fixed contrasts are:

```text
training change: L[mix,s,p] - L[base,s,p]
within-arm allocation change: L[a,s,risk] - L[a,s,random25]
interaction: (L[mix,s,risk] - L[mix,s,random25])
             - (L[base,s,risk] - L[base,s,random25])
```

Retain base, dense, speed and relative-velocity controls and all three seed
values. Compute mean and sample SD across the three seed contrasts only when
all are defined. Use the frozen equal-trajectory aggregation for the uneven
128-history clean grid. Report benefits and timing as secondary diagnostics
with their full denominators. Do not bootstrap particles, treat frames as
independent training runs, calculate new p-values, tune selection rules, or
claim end-to-end speedup from the saved diagnostic timers.

The result must include the complete original 4,728-cell accounting and the
accepted 2,899 completed / 1,817 not completed / 12 timed-current totals. Label
the 2,160 autonomous cells as outside this numeric re-audit, retaining their
original states. The full autonomous cohort did not complete; any unavailable
full-cohort rollout mean stays unavailable. Operational censoring and missing
audit output do not establish numerical divergence or a termination cause.

## Separately versioned sources and products

Prepare a new inert package, for example
`goop3d_observed_history_analysis_UNADMITTED_transition_v1`, containing:

1. `audit_goop3d_observed_histories_v1.py`: a new exact-input driver and new
   `goop3d_observed_history_saved_audit_v1` schema. It must never invoke the
   original `audit_collection` or main function. No existing source is edited.
2. `goop3d_observed_history_arithmetic_v1.py`: the needed pure functions copied
   from the reviewed saved auditor with an AST-equality record; preserve the
   unchanged `goop3d_saved_diagnostic_audit_v1.py` as a byte-pinned dependency.
3. `summarize_goop3d_observed_histories_v1.py` and an independently written
   `check_goop3d_observed_history_summary_v1.py`: operate only on the new audit
   receipt, reconstruct all fixed denominators and contrasts, and reject an
   original full-audit schema substituted for this limited result.
4. New root controller/contract files for one specifically named observed
   operation, plus the exact native owner/bootstrap sources or independently
   reviewed derivatives needed to recognize that new operation. The existing
   D3 operator cannot be reused by changing its expired `analysis_phase.json`.
   Preserve the qualified hard monotonic timer, timeout, parent-death behavior,
   registered identities and cleanup logic; any contract-map change needs
   independent source review and focused fixtures.
   If the alternate route is still needed, include a separately versioned
   D3 phase-bound proxy and child-environment wrapper. The existing metadata
   proxy accepts only 5–60 seconds and cannot serve a long numeric operation.
   A reviewed Sand phase-route design can inform the implementation, but its
   Sand phase or pins cannot authorize D3. Bind the exact original TLS authority
   and verified upstream, opaque CONNECT, the newly admitted D3 phase and
   absolute stop, bounded connections, and complete proxy exit/closure. Do not
   extend the existing metadata proxy in place or restart proxies to extend an
   expired phase. This is a prospective requirement, not a claimed explanation
   for the missing original audit output.
5. Exact input/source/dependency pins, scope and output allowlists, focused
   fixture receipt, independent review, handoff, and final manifest.

Use a new output root such as
`/root/repos/AdaptGNS-cuda-20261006/goop3d_observed_history_analysis_20261006_v1`.
Keep `audit.json`, `summary.json`, `arithmetic_check.json`, an append-only progress
ledger, and failure/owner/transport metadata separate. Partial progress cannot
be consumed as a successful audit. Never write `saved_array_audit.json`, use
the original full-audit success schema, or fabricate a receipt accepted by the
old full-stage paired checker. Original outputs and failures remain untouched.

Focused tests should cover exact/partial family accounting, every seed retained,
same-source pairing and target representation, model-dependent risk graphs,
extra/missing rows or archive members, object arrays, path/symlink refusal,
integrity changes, failed repeats, deadline interruption, and publication only
after required checks. Use synthetic data; no scientific fixtures run during
source preparation. Independent reviewers inspect source and fixtures before
root freezes the package. Do not duplicate the full existing test suite.

## Proposed fixed allocation and feasibility

Propose exactly one new 3,600-second analysis allocation, with one absolute end
`T0 + 3600 seconds <= 2026-10-07 01:00:00 UTC`. Therefore root must start it no
later than 00:00 UTC, after the package and contracts are reviewed and frozen.
No current clock was queried for this proposal; the recorded 22:26 observation
cannot establish present fit. Root must obtain a fresh coherent root/host UTC
and monotonic mapping, exact boot, and fresh full native closure before issuing
anything. A smaller remaining window is not silently substituted for the hour.

Suggested fixed subdeadlines, all relative to the single T0:

| Interval | Work and hard outcome |
| --- | --- |
| 0–120 s | Verify collection/source pins, all cell accounting, exact observed allowlist, per-file size limits and family completeness. Stop on integrity disagreement. |
| 120–2700 s | One bounded observed-data audit including final observed-file/source rehash. Stop admitting new work at 2700 s; an incomplete audit gets no successful scientific receipt. |
| 2700–2940 s | Complete fixed aggregation and independent arithmetic check from a passing audit receipt only. |
| 2940–3480 s | Collect bounded outputs, obtain actual tool exits and complete owner/worker closure, and perform independent root product review. |
| 3480–3540 s | Final input/source and native-state review needed for acceptance, with no additional numeric work. |
| 3540–3600 s | Reserved cleanup and final publication; never extend the allocation to obtain a favorable or complete result. |

This is a ceiling, not a runtime prediction. No completed observed-only
throughput measurement exists, and the prior missing audit output cannot
provide one. The first phase can report total permitted bytes, maximum archive
size, cell coverage and the remaining budget without exposing error values.
If the bounded source/grid/size gates do not pass or the fixed audit deadline
is reached, preserve that exact failure/incompleteness and stop. Do not launch
a pilot inference, rerun the original audit, reset T0, resume a partial archive
pass, extend the allocation, or drop expensive policies.

All remote workers remain CPU-only with two threads. Every child receives a
literal absolute monotonic hard stop derived from this new admitted allocation;
fresh root/host time disagreement prevents launch. Per-operation shutdown and
publication reserves must fit their fixed subdeadlines, and transport/proxy
lifetimes must end within the outer allocation. Native guards remain required
even if a local clock stalls or jumps. Original tool exits, registered native
closure, and the proxy's actual lifetime are separate acceptance evidence.

If root cannot prepare, review, and start this exact scope by 00:00 UTC, or
chooses to preserve attention for Sand, retain the accepted coverage and
limitations result. No D3 allocation is warranted merely because some wall
time may remain. This proposal does not claim feasibility or conference-ready
scientific evidence before those gates are met.
