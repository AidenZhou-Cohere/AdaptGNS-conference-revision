# Goop3D completion: focused source advice

This is a source-only review of successor orchestration opportunities under the human's renewed authorization. It does not execute a model, inspect numerical result arrays, admit accuracy, or modify an original source/outcome. The original unsuccessful attempts remain historical facts; successors should have their own output paths and metadata.

## Observed-history auditing

The original driver `cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1/audit_goop3d_observed_histories_v1.py` is serial across 18 model/stages and every row (lines 114–159). A process pool over independent complete row identities can invoke unchanged `diagnostic.audit_row`, graph/action checks, and pairing-hash extraction. Merge receipts in the original stage/cell order, then run the unchanged equal-trajectory aggregation and all cross-model shared-identity comparisons. Final acceptance must still cover all 2,568 observed cells and all 4,728 original accounting cells.

Read/hash/parse the 659,091,312-byte accepted collection once in the parent, perform complete accounting once, and dispatch bounded row descriptors rather than serializing the full collection to each task. Each worker should load/decompress one archive at a time. The original archive cap is 8 GiB decoded (`observed_common.py:31`); worker count must respect that upper bound plus Python graph-set and array working memory. Root reports roughly 1 TiB RAM and 144 CPUs per host, making the proposed 16 workers plausible; this is not a measured memory guarantee.

Strong source-level bottleneck candidates are full NPZ decompression (`numeric_archive`, lines 14–25), Python tuple/set graph checks (`goop3d_observed_history_arithmetic_v1.py:118–155`), repeated rank loops (`goop3d_saved_diagnostic_audit_v1.py:92–100`) and many saved timing-round outputs. Their actual contributions are unprofiled. Process parallelism keeps these arithmetic/check implementations unchanged. Do not remove repeated-output checks, inspect only the first timing round, relax tolerances or drop graph/action/RNG verification to save time.

The original byte-bound reads and final byte recheck are distinct checks, not redundant permission machinery. Retain both; parallelize independent final file rehashes if useful. Avoid rehashing the full collection or every frozen source once per row. Per-cell receipts may checkpoint successful work, but must bind the original row/archive hashes and exact audit source hashes, include computed metrics and shared-identity hashes, and only be reused when those identities still match. The original `audit.progress.jsonl` records labels only (driver line 152), so it cannot serve as a completed numeric-audit cache.

## Autonomous completion

The original evaluator invokes each rollout with seed `93000 + 1000 * training_seed + source_index`, horizon 295 and trace steps `(1,10,50,200,295)` (`evaluate_goop3d_graph_support_v1.py:363`). `native.rollout` constructs a local NumPy generator at line 292, a local six-frame history at line 268, and local risk cache at line 294. Thus whole `(arm, seed, split, source_index, policy)` cells are independent dispatch units. Preserve the exact seed; do not add arm, split, policy, host or worker identifiers.

One process per GPU can keep a frozen model loaded while consuming cells for that model. Keep sequential forecast updates inside each cell: candidate/selection order, float32 predictions, cached-risk warmup, physical RMS arithmetic and every guard remain unchanged. Parallelism changes descriptive shared-host timing, not the defined accuracy computation. Record the new runtime lineage instead of making an isolated-speedup claim.

Only the initial six frames are full starting state. Old sparse traces do not contain a complete per-step rollout checkpoint with history, cached scores and RNG state. A timed-out cell can be restarted from its original initial state into a new successor directory; it cannot be presented as exact checkpoint continuation. Preserve its original partial/timed record. Reuse already complete original cells, and merge old/new cells by a frozen status-only rule. Retain any original scientific guard outcome and its fixed-denominator role; do not resample based on error or choose among repeated results by accuracy.

## Sources read

Paths above are relative to `work/deadline_research_20261005/` unless they already start with `cuda_preparation/`; all are under that preparation directory.

| Source basename | SHA256 |
| --- | --- |
| `audit_goop3d_observed_histories_v1.py` | `36f6b18b903e1ce7f89c138155a1147fcfab581246f11e7f0bbc0517d9010aff` |
| `goop3d_observed_history_arithmetic_v1.py` | `0f03e7876765fe56d9c9e33404b6e66ae014c81c5022b9c59b04d77d6c884161` |
| `goop3d_saved_diagnostic_audit_v1.py` | `d772313e7a475897cc2dc7d5a893d10c89a8cad67fee9c9b5fe2fa3683006c2a` |
| `goop3d_native_evaluation_v1.py` | `a742123093aff433f3a4e302929a52bae4f5fee9610195d86df726852575b1d5` |
| `evaluate_goop3d_graph_support_v1.py` | `9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de` |

The successor drivers had not yet appeared when this early advice was written. Their source/RNG/checkpoint/restart semantics still require focused review. No nested release or clock framework is proposed.
