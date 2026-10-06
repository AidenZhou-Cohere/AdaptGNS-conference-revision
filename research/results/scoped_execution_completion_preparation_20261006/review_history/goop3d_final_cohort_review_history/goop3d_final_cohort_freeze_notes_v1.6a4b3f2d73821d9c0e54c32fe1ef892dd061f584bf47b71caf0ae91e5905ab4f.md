# D3 final-cohort byte freeze adapter

This is a separate preparation module. It does not change the frozen scientific supervisor, trainer, protocol, evaluator or numerical implementation. Default invocation describes scope only; `--execute --mode inventory|freeze` is required to act.

Root must first verify that the full six scientific jobs are finished, all worker/supervisor/outer identities are stopped or reaped, and the completed endpoint and paired-initial-state audit is acceptable. The adapter does not inspect live processes or deserialize checkpoint tensors. Its tensor/Adam conclusions are explicitly inherited from the pinned completed supervisor audit. Capacity endpoints and incomplete or unsuccessful science trees cannot be admitted.

1. Inventory the immutable completed science tree as opaque bytes with `--execute --mode inventory --science-dir SCIENCE --output-file FRESH_INVENTORY_OUTSIDE_SCIENCE`.
2. Independently review that complete inventory and scalar supervisor receipts. Root supplies a fresh stopped-process receipt with every supervisor/worker PID (additional outer PIDs may be recorded), the completed summary hash and empty matching process/GPU lists. The adapter trusts this explicit root process evidence; it does not manufacture absence checks.
3. Root writes a mode-specific freeze release binding this adapter, pinned supervisor/evaluator, complete inventory, stopped receipt, absolute science/fresh output paths and the prospectively selected endpoint. Templates are intentionally unadmitted.
4. Run `--execute --mode freeze --science-dir SCIENCE --inventory INVENTORY --stopped-receipt STOPPED --root-release RELEASE --output-dir FRESH_COHORT_DIR`.

The adapter independently checks the exact source/configuration/data snapshot closure, all six cell identities and endpoint pointers/checkpoint bytes, initial/final audit conclusions, all five pairing flags, A/B reaped-wave chronology and candidate-to-summary correspondence. It runs the unchanged evaluator's scalar cohort gate for every model. It rehashes source/evidence and the entire science tree before publishing `cohort.json` last. The stopped receipt must still be no more than 300 seconds old and time must precede 2026-10-07 01:00 UTC. A failure retains any output with `failed_freeze.json`, never active cohort admission. Fresh paths are required; do not overwrite failed attempts.

No test headers, test files, official arrays, inference, GPU or process control are accessed by this adapter. The independent reviewer must verify the source and synthetic-test hashes before a concrete root release is issued.

Development attempts are retained: attempts1/2 each passed13 tests; attempt3 reached19passed/1failed because the synthetic test expected capitalized `Wave` while the correctly rejected receipt reported lowercase `wave`; the test-only pattern was corrected and attempt4 passed20 tests. Rebound negative fixtures exercise semantic refusal after updating synthetic inventory hashes; these are not real scientific receipts or admissions.
