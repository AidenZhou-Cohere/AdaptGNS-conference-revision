# Resumable Goop3D observed-history verification

The user explicitly asked to continue unfinished work and remove the arbitrary cutoff. This successor audits the existing saved observed-history files using the exact frozen scientific functions. It performs no training, model inference, new graph generation, or autonomous-rollout evaluation. Earlier attempts and their files remain unchanged.

## Runtime files and command

Stage these three files together in one new source directory: `run_observed_resumable.py`, `source_pins.json`, and `protocol.json`. The runner finds both JSON files beside itself. The existing frozen source directory is `/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop3d_observed_history_analysis_v1`. Root selected the fresh output directory `/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007/observed_results_v1`.

From the staged source directory, the complete command is:

```sh
env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /root/repos/AdaptGNS-cuda-20261006/.venv/bin/python -B -u ./run_observed_resumable.py --execute --workers 16 --output /root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007/observed_results_v1 --source-pins-sha256 365c8071abe5016a9092813d57177705386f6568efe03ce1893b640d4b8e7c92 --protocol-sha256 d2c2815cfee9b1f0eb4aff854eb4acb6c00b0ec0b6f232e308bde9487078f167
```

Root owns staging and launch. No proxy, per-action release, elapsed cutoff, hourly allowance or clock-agreement guard is used. The runner is NumPy-only; root's native-driver environment is inherited for consistency. Sixteen processes have at most32 rows in flight, with one BLAS/OMP thread each.

## Resume and failure behavior

Every successful row produces an fsynced JSON record with audited values, exact pairing hashes and scientific-check count. The record's SHA enters a separately fsynced index only after the record is durable. A single output lock prevents concurrent writers. Restart with the same command and output to reuse committed rows. Source/protocol, accepted collection, task-input pins and runtime identity must agree. Cache corruption fails explicitly. An unindexed row from an interrupted commit is recomputed and compared with the preserved bytes before commitment.

A failed row is preserved under its attempt directory and prevents final products. Other independent rows still finish and checkpoint. A later invocation skips previously failed rows unless `--retry-failed` is explicitly supplied; original failure records remain. Index-write failures abort the attempt rather than being mislabeled scientific failures. An already completed output verifies its saved product/index hashes and returns without auditing again.

The old audit's progress log contains1112 completed row labels out of2568, with no audited numerical cache. Those labels cannot safely substitute for numerical results. This successor recomputes those rows once. The exact original worklist contains5154 files totaling45,775,548,716 bytes, including18 protocols. New progress records report durable completed/failed counts and elapsed time; they are informative rather than stopping conditions.

## Scientific equivalence and outputs

The original accounting and archive parser are imported unchanged. The per-row body preserves the original diagnostic, all timing/parity, graph/action and pairing checks. Completion order does not change arithmetic: shared-identity comparisons and stage aggregates are rebuilt in the original fixed cell order. All2568 observed cells and4728 original accounting states remain. Missing cells retain their denominator; incomplete families retain null comparisons. Every selected original input, the collection and bound science sources are rehashed before final publication.

The original summary function and independent arithmetic checker run directly after the merge. Successful output contains `audit.json`, `summary.json`, `arithmetic_check.json` and `completion.json`; the latter binds the products and committed index. The legacy `phase_sha256` is null because this successor has no timed phase; successor protocol/source hashes and the prior failed phase identify provenance explicitly. Completed arithmetic is evidence for independent scientific interpretation, with the original observed-history limitations retained.

## Verification

Twelve focused synthetic tests passed, including real spawned workers matching serial frozen results, a clean-row hand oracle, semantic corruption after repinning, cache corruption, interrupted index commitment/orphan recovery, failure-history preservation, cross-row identity disagreement, exact full-population summary equivalence, missing-cell/null-family behavior, and an already-completed fast exit. No real collection, model, trajectory or saved numerical result was read by those tests.

Independent source review: `../review_v1/observed_source_review.json`, SHA `f8c99c3bd0c28cfb2b6b565651e9a7a4136db6980518733db9585cbd64ccb6ef`; independent12-test tool53100d exited0. Runner SHA `8bcdb32f7d36a9a4c6f3bcfdb96eecbe36c5b419a855b35473448627e7e7f76b`. Runtime source/protocol bytes were frozen before root staging.
