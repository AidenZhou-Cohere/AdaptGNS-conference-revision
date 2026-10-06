# Goop saved-array audit candidate revision3 — independent review pending

This fresh snapshot repairs a supporting-auditor representation-key defect
identified by the first real revision2 A audit. It does not change any frozen
trainer, evaluator, selector, collector, scalar aggregator, scientific output,
cohort or formula. The v1 and v2 snapshots and the real failed receipt are
preserved. No real audit has run on revision3; synthetic tests do not admit
scientific results. Independent source review and parent release remain required.

`source_sha256.json` seals the five sources/tests. `synthetic_verification.json`
records the sealed-copy command and passing **92 tests and 78 subtests**.
`development_failure_history.json` retains earlier development defects;
`actual_v2_failure_history.json` separately records the real audit failure and
its evidence. The copied failure-evidence JSON contains the parsed remote
receipt and its recorded raw-byte SHA, not a new raw-byte receipt copy.

## What is independently checked

- Exact scoped-v3 collection SHA, frozen source/protocol identity, fixed model,
  stage and cell grids, stopped-input reverification receipt, JSON/NPZ byte
  bindings, numeric array descriptors, and unchanged read inputs/source through
  publication. Original paths remain in the cell ledger; a byte-identical local
  queue tree is allowed. Missing and failed cells stay explicit.
- For rollouts, MSE at the saved accepted forecasts **1, 10, 50, 200, 395**,
  saved truth boundaries, prediction boundaries recoverable from forecasts and
  sparse histories, overlaps, state hashes, graph prefix/count/hash consistency,
  retained risk, parity feature/output arithmetic, and saved warmup/rejected
  numerical guards. Full-H and failed-prefix means are recomputed from the
  *recorded scalar series*, not from unsaved trajectory positions. Forecast200
  may remain defined for a later failed rollout; full-H statistics remain null.
- For diagnostics, saved residuals, position and normalized-coordinate MSE,
  vector-error benefits, sign fractions, dense/sparse disagreements, average-rank
  Spearman correlations, constant-rank nulls, clean variance/NLL arithmetic,
  parity/repeat/status gates, and timing mean/median/sample-SD arithmetic.
- Where diagnostic candidate arrays are retained, optional actions are
  recomputed over that exact saved annulus: dense, random PCG64/SeedSequence,
  speed, previous-observed risk, and physical RMS. Physical RMS uses incoming
  nonself native support, float64 displacement divided by **dt=.0025**, and zero
  for isolated particles. Diagnostic error normalization remains discrete
  acceleration units; no physical dt-squared factor enters residual metrics.
- Common observed states, available targets and model-independent graphs are
  compared across paired models, with seed-specific random subsets. Targets
  shared with saved rollout steps are additionally compared as float32 bytes.
- Equal-frame-within-source then equal-source diagnostic means and complete
  rollout source means; the companion independently checks three ordered seed
  values, sample SDs, paired differences, interaction signs, coverage and null
  propagation against the published scoped-v3 summary. There are no survivor
  means or replacement seeds.

## Unsupported evidence and operational limits

The audit does **not** reload official trajectory source files, reconstruct
checkpoint normalization, run a model, reproduce geometric search completeness,
replay unsaved positions/graphs, reconstruct generic exceptions, or verify
hardware clocks/isolation. Scalar-series checks cannot establish that unsaved
step MSE agrees with unsaved model predictions. Clean target normalization is
not reconstructible from its saved arrays. Unsaved repeat graphs and physical
scores are not freshly recomputed. Measured runtime is operational data; no
dedicated-host or causal speedup claim follows.

Collections are parsed into memory. The parser streams its SHA passes and
avoids an extra retained binary copy, but a near-1-GB JSON collection can still
need several GB of RAM. NPZs are processed one row at a time; each row's arrays
are materialized together. Run A and B **sequentially** on a machine with
adequate free memory; preserve an OOM/timeout rather than silently narrowing
scope. No remote compute, GPU or network is invoked by these programs.

## Invocation after independent review

Run the snapshot scripts together, retaining exact hashes and fresh outputs
outside the retained queue tree. Default invocation only describes the scope.
Supply reviewed SHA256 values; example placeholders are not runnable commands.

```text
python audit_goop_saved_arrays_v1.py --execute \
  --collection /path/to/role_A_collection.json \
  --collection-sha256 <exact_role_A_collection_sha256> \
  --queue-root /path/to/byte_identical_A_queue \
  --output /fresh/path/role_A_array_audit.json

# Repeat sequentially for role B, then check the published paired summary.
python audit_goop_paired_arrays_v1.py --execute \
  --audit-a /path/to/role_A_array_audit.json --audit-a-sha256 <exact_sha256> \
  --audit-b /path/to/role_B_array_audit.json --audit-b-sha256 <exact_sha256> \
  --summary /path/to/paired_scalar_summary.json --summary-sha256 <exact_sha256> \
  --output /fresh/path/paired_array_audit.json
```

`passed_supported_checks` means only the scope above passed. A mismatch emits a
failed receipt without modifying originals. Process interruption/OOM can occur
before a receipt exists; preserve its exit/stdout/stderr separately. Keep every
failed receipt and all affected source hashes before reviewing a retry.

## Revision2 repairs after independent review

The v1 snapshot remains unchanged and must not be used for actual admission.
Independent review found that a collection cell could disagree with its bound
row failure, and the paired companion omitted per-model rollout coverage and
failed-prefix listings. Revision2 requires exact cell/row failure identity,
emits independently computed failed-prefix boundary aggregates, and compares
all per-model counts, categories and exact failed-prefix listings with the
published summary. Nonzero interaction-sign regression tests were added.
The sealed revision2 output schemas retained their v1 interface names and
emitted `audit_revision: 2`; its companion required that revision and source
hashes. Revision2 passed independent review, then its first actual A audit
failed on the cross-mode target namespace defect described below.


## Revision3: exact target identity across frozen representations

The actual v2 failure occurred at base/seed0/clean_validation/(4,105): v2 put
raw `target_position` hashes for both same-state and clean-validation under
`valid/target_position/(4, 105)`. Frozen evaluation source deliberately promotes
the same-state source target to float64 (line355), while clean validation keeps
float32 (line432). The retained narrow evidence establishes equal shape
[1398,2], different raw hashes/dtypes, exact numeric equality, and zero maximum
absolute difference. This is an auditor comparison defect, not evidence of
changed data or a numerical scientific failure.

Revision3 records raw target hashes under a **mode-specific** key, preserving
exact within-mode identity. It retains the shared split/source/frame
`truth_float32` key across modes and available rollout truth. Before that
canonical comparison, every present target must have the frozen mode dtype,
shape (particles,2), finite values, and an **exact lossless float32 roundtrip**.
No tolerance or lossy rounding is accepted. Legitimate early failed rows may
lack a target and contribute no invented target hash. Validation/test keys
remain separate. All other shared arrays and original metric arithmetic are
unchanged; the diagnostic helper is byte-identical to revision2.

Synthetic regressions cover equal values in the two frozen representations,
a true changed canonical target, a same-mode raw target mismatch, a float64
nextafter perturbation that float32 rounding would hide, wrong dtype/shape,
nonfinite targets, absent targets on failed rows, and distinct validation/test
targets. Collection regressions use authentic fixed stage/cell/protocol grids
and SHA-bound synthetic NPZs; no real numerical result informed their values.
Main and paired receipts now emit `audit_revision: 3`; the companion requires
revision3 and the exact current main/helper hashes. Invoke only after independent
review and parent release, using fresh outputs and preserving every failed receipt.
