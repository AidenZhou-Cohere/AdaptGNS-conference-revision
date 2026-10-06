# Cost-study failures and Sand completion preparation

This package preserves two failed Goop2D validation cost pilots and separately reviewed Sand completion preparation. **It contains no admitted Goop2D cost result and no completed Sand cohort or evaluation result.** Existing scientific Goop/WaterDrop accuracy results are outside this package.

## Goop2D status and interpretation

The initial v1 cost source was not launched. Hardware qualification distinguished CUDA `mem_get_info` capacity (197,897,617,408 bytes) from the `nvidia-smi` framebuffer total (189,471 MiB); the APIs are pinned separately. The earlier property-based probe is preserved and is not presented as a `mem_get_info` measurement.

The v2 pilot failed its graph-audit equality check before any complete state was committed. The physical-policy graph descriptor includes a measured duration that should not be required to repeat exactly. The differing historical field values were not recorded, so this source defect and its synthetic reproduction are distinguished from directly observed failure values.

A single explicitly amended v3 attempt excluded only the physical policy's `physical_score_seconds` from equality, required valid retained durations on both sides, and kept every other field exact. It retained the original clock beginning **2026-10-06 15:42:19.395876 UTC**, the original **16:42:19.395876 UTC** stop, and all time already spent; its cumulative preflight allowance was explicitly changed from 300 to 900 seconds.

The v3 pilot failed with `Live child identity changed`. Three first-arm children each published 15 diagnostic scalar states; all 45 preserved scalar states pass the repetition/memory receipt checks. All 45 physical timing pairs retain distinct valid durations. Nevertheless, the owner verified **zero states and zero models**, one child exited after SIGINT, and the three counterpart arms never launched. Root declined the full study and any further repair. No subset timing, speedup, memory comparison, or full-workload extrapolation is computed or supported. A process-exit/inspection race is supported as a source-level possibility; the offending live native row/PID was not saved, so it is not established as the historical cause.

## Files

- `goop2d_sources_and_reviews.tar.gz`: exact v1/v2/v3 sources, reused reporters, source tests and failed test history, hardware qualification distinctions, amendments, independent reviews, and the final decline decision.
- `goop2d_candidates_and_failures.tar.gz`: compact exact candidate/staging metadata and both original failure text trees. This includes all 45 v3 state JSON files, every child log and receipt, the pinned validation manifest, original external exits, and native closure evidence.
- `sand_completion_bridge_preparation.tar.gz`: a separate source/preparation family containing the reviewed recovered-cohort bridge, reserved-test adapter, unresolved handoff templates, tests, and relevant frozen source dependencies. The bridge preserves original 50k plus recovered 50k provenance and makes no fresh-start or cross-device bitwise-equivalence claim. Its templates remain unadmitted.
- `archive_manifest.json`: every archive and member's exact SHA-256 and byte count, plus explicit omission policy. Member paths are relative to the original workspace.
- `verify_package.py`, `replay_failure_evidence.py`, and `run_source_tests.py`: standard-library verification and CPU synthetic/source-test replay.
- `build_package_original_workspace.py`: construction source retained for provenance. It is intended for the original full workspace; archive verification and test replay below are the portable interfaces.

Historical releases, launchers and test fixtures are preserved as evidence. They are not current authorization to execute experiments. Do not run the historical launchers.

## Uniform numerical omission

No model checkpoint, trajectory array, or NPZ reference payload is included. Both scalar failure trees are preserved without selecting favorable policies or states. V2 has no NPZ artifacts. All 45 v3 NPZ artifacts (43,800,152 member bytes) are uniformly omitted; their exact opaque hashes and sizes remain in `collection_inventory.json`. Root also preserved those original bytes locally as `original_numeric_artifacts.tar` (43,888,640 bytes; SHA-256 `7a171591e4ee81722281ce69ef3e2a9fd481391f3c41ddd1c69573c67102119d`), with the original remote files retained. The included `numeric_archive_receipt.json` records that preservation. No array was deserialized for this package.

The artifact checks below establish correspondence among collected text, committed scalar hashes, opaque NPZ hash pointers, and the source guards. They do not replace a numerical replay or convert a failed pilot into scientific evidence.

## Reproduction

Python 3.10 or later is required. Verification and the selected tests use only the Python standard library; CUDA, PyTorch, NumPy, network access, model files and data arrays are unnecessary.

From this directory, verify all archive bytes and exact member sets:

```sh
python3 -B verify_package.py
```

Replay the selected CPU/source tests and failure-accounting checks in a fresh temporary extraction. Choose a new output directory:

```sh
python3 -B run_source_tests.py --output /tmp/adaptgns-cost-package-tests
```

The test runner executes synthetic contract/regression tests for all three source versions, reporting integration tests using synthetic complete workloads, and the Sand completion bridge tests. Synthetic reporting tests do not summarize the real failed pilot. `source_tests.log` and `test_receipt.json` retain the full test result.

To inspect exact source and evidence files manually, choose an unused extraction directory:

```sh
python3 -B verify_package.py --extract-to /tmp/adaptgns-cost-package
python3 -B replay_failure_evidence.py /tmp/adaptgns-cost-package
```

The reconstructed final package passed **136 CPU/source tests** across 11 modules. `verification/source_tests.log`, `verification/test_receipt.json`, and `verification/package_test_execution.json` retain the result. `preparation_history` preserves superseded package manifests, helper sources, review findings and the earlier passing test attempt. The final package-level manifest records every supplied file before root's separate publication step.
