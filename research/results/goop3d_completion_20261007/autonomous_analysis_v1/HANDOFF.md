# Autonomous completion analysis handoff

This new driver collects exactly the 331 immutable original autonomous outcomes plus up to 1,829 new final worker commits. All 2,160 source/model/policy cells remain in every snapshot. It preserves original invocation outcomes and missing-cell history separately. It neither loads a model nor opens official source trajectory arrays. The numerical work is CPU arithmetic on saved JSON/NPZ only.

The reviewed frozen arithmetic is reused without modification: `audit_rollout`, `validate_rollout`, `full_summary`, and the independently implemented `paired_metrics`. Trace reconstruction is limited to saved forecasts 1/10/50/200/295 and overlapping saved histories. Full trajectory metrics check recorded scalar series; unsaved errors are not array-recomputed. Boundary excursions do not establish physical validity. Recorded time arithmetic cannot establish hardware isolation or causal speedup.

## Inputs and deployment

`source_manifest.json` pins the analysis driver, four existing arithmetic sources, original ledger, official metadata, original completion plan and running worker source. Root owns staging, process identity, all remote operations and launches. The driver is not itself a network or process-control tool.

Use teal for analysis. The exact original ledger is:

`/root/repos/AdaptGNS-cuda-20261006/goop3d_final_evaluation_20261006_v1/final_evaluation_ledger.json`

SHA256 `ef0673b69180c0c648d7220a300ac526c115482b0aa26ddab39a2ddd9146939b`, 6,940,087 bytes.

Use the unchanged metadata in teal’s original repository (runtime_v1 was staged only on yellow/aquamarine):

`/root/repos/AdaptGNS-cuda-20261006/goop3d_numeric_train_valid_20261006_v1/metadata.json`

SHA256 `727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55`, 471 bytes.

The 659MB original stopped scalar collection is not required. The exact plan and ledger supply the complete grid and all original hash bindings. Original 331 JSON/NPZ outputs are read directly from teal's original queue.

`run_arguments.json` provides an explicit suggested argv for a first run. `resume_arguments.json` adds `--resume`. Root may choose the fresh source/output directory names before the first run, retaining explicit absolute roots. The input/output/source paths are identity-bound after the first run. Use the exact frozen driver thereafter. Sixteen CPU audit workers is the initial suggestion and can be reduced if local GPU jobs experience resource contention; per-cell results do not depend on worker count. No arbitrary elapsed limit is built into this driver.

Workers 00–03 are read from teal's `autonomous_results_v1`. Workers 04–11 are read from teal's `collected_autonomous_v1` after immutable outputs are copied from yellow/aquamarine. Missing destination workers and uncommitted cell directories are permitted during incremental snapshots.

## Incremental transfer of stable commits

For each source worker, stage only these immutable inputs. Never overwrite destination bytes with a different SHA256; retain/report conflicts.

1. Copy `worker_identity.json` and `protocol.json` once, verify their SHA256 and plan/source identity, and publish each via a temporary file and atomic rename.
2. Enumerate `cells/*/commit.json`. If the destination already has the same marker hash, skip that cell. A different existing marker is a conflict.
3. For a newly observed marker, resolve its relative `row_file` and `artifact_file` beneath that cell directory. Copy only those JSON/NPZ bytes to temporary destination files. Verify the marker's declared SHA256 before publishing them atomically.
4. Publish the verified `commit.json` last. The analyzer admits a cell only through that final marker, so a transfer interrupted before it remains uncommitted.
5. Preserve immutable `attempts/attempt_NNNNNN/{owner,previous_owner,identity,resume_verification,runtime,failed_attempt,outcome}.json` files when present. These retain prior infrastructure errors and attempts; they do not affect the fixed scientific cell denominator.
6. Do not copy live root `owner.json` or `status.json` as frozen evidence. After root verifies the worker's exact process closure and actual exit code, capture its final status and native process receipt separately. Do not read/copy growing logs or uncommitted cell artifacts to make an analysis snapshot.

This does not require whole-tree recopy, repeated training, or completed-cell reruns. Data-transfer destinations should keep `worker_XX/cells/<cell_id>/...` layout. The suggested argv names every worker root explicitly.

## Resume and interpretation

The driver holds a kernel lock and refuses concurrent use of the same audit output. Each attempt writes a fresh collection/audit/summary/receipt under `attempts/NNNNNN`; `latest.json` only points to the most recent successful snapshot. Per-cell audit checkpoints bind the exact JSON/NPZ task hashes and source/runtime identity. Explicit `--resume` reuses successful checks and audits only newly arrived commits. Earlier observed inputs/cells cannot disappear or change. Every snapshot rehashes all selected artifacts, including cached rows, before publishing products.

Failed saved-array checks are retained and stop final success; they are not retried unless root explicitly passes `--retry-failed` after review. This repeats an audit calculation only, never inference. Numerical guard outcomes that pass their saved-evidence checks are permanent scientific failed outcomes, never retry candidates.

Any missing cell propagates through its complete-source means and three-seed statistics; survivor means are not computed. A later numerical failure may still retain the predeclared pointwise H200 value, while H295 and complete trajectory means remain null. All original invocation failures remain separate from cell outcomes. Per-cell synchronized runtime is preserved, but different host/time/order conditions prevent a causal speedup conclusion.

A receipt saying all2,160 rows are audited is only an arithmetic completion milestone. Scientific admission additionally requires root to verify each of the 12 workers' actual exit0, closed PID/start identity, successful final input reverification, final copied artifact hashes, and independent result review. This driver explicitly does not verify remote process exits or declare conference readiness.

## Verification status

The driver is separately versioned; frozen training, model, protocol and numerical evaluator files remain untouched. Tests use invented two-particle arrays and a metadata-only full-grid fixture. Development fixture failures and all test logs remain in this directory. The independent reader reviews the exact source pin before remote execution.
