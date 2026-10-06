# Goop saved-array audit candidate — independent review pending

This directory contains a byte-identical snapshot of five new supporting
sources. None modifies or imports the frozen trainer, evaluator, selector,
collector, or scalar aggregator. No real scientific outcomes were used to
develop the formulas or tests. Do not treat a synthetic pass as a real-data
audit. Independent review is required before the first real application.

`source_sha256.json` seals the sources. `synthetic_verification.json` records
the command, source hashes and passing result: **75 tests and 78 subtests**.
`development_failure_history.json` retains development defects repaired before
sealing; these were not experimental outcomes.

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
