# Cache-only observed finalization v2

This small successor repairs only the independent checker’s floating sample-SD reduction. All 2,568 original row-array audits passed; the old process then failed during its final checker because three identical large graph-count means produced a false nonzero SD when centered at a rounded mean. The old runner, original checker, source/protocol pins, row cache and failed check remain unchanged.

The new checker computes the same unbiased three-seed sample variance from exact rational pairwise differences divided by six, followed by a 60-digit Decimal square root and final float conversion. The original mean, denominators, seed values, null rules, family-completion logic, contrasts and equality tolerance remain unchanged. The pure verification body is retained. Revision and correction metadata distinguish its result from the failed v1 checker.

`source_manifest.json` pins both new source files and every original source/cache/failure identity. `run_arguments.json` is the exact proposed argv for root to run on teal, using:

- Original cache: `/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007/observed_results_v1`
- Original runner: `/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007/observed_v1`
- Frozen arithmetic: `/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop3d_observed_history_analysis_v1`
- Fresh output: `/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007/observed_finalized_v2`

Root owns staging, actual process-closure verification, launch and subsequent product collection. The finisher acquires the original cache’s existing kernel lock in read-only file mode. It never creates/deletes cache files, changes its index or starts workers. The output must be fresh and separate. If finalization is interrupted, preserve that output and choose a fresh successor output name after root verifies closure; the original row cache remains reusable.

The finisher imports the exact original runner’s `prepare` and `merge` functions plus the unchanged summarizer. Preparation reads original scalar metadata and reconstructs the complete expected 2,568-task grid and exact source/input task hashes. A read-only cache adapter verifies every cached record SHA, identity and task hash. Missing, failed, altered or extra cache rows refuse finalization. No `audit_cell`, process pool, scientific model execution or NPZ decoding is called.

After merge and corrected arithmetic verification, original NPZ inputs are byte-hashed only. This retains the old final input check without repeating any array-derived scientific calculation. Original collection, source and cache bytes are also rechecked. The previous failure is copied byte-identically into the fresh output.

Products are `audit.json`, `summary.json`, `arithmetic_check.json`, `retained_original_failure.json`, and a final `completion.json` that binds all product and cache hashes and explicitly records zero row-array audit calls. The audit keeps every original 4,728 cell-accounting record, including the unfinished autonomous cells from the original invocation; this observed finalizer does not replace the separate new autonomous completion analysis.

Five synthetic tests pass: exact-variance regression including the real-case scalar values, unchanged verifier/tolerance AST, corrupted accuracy rejection, no scientific/process/decoder calls in the finisher, and read-only cache identity/hash/completeness guards. The tests do not run the finisher on real research arrays or regenerate the 2,568 row audits. Independent source review and actual product review remain root/reviewer gates before scientific admission.
