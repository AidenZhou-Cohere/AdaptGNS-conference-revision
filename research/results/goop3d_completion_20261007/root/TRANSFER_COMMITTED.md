# Incremental committed-cell transfer

Root executes the helper; its author performed synthetic local tests only.

```sh
work/venv/bin/python -B work/goop3d_completion_20261007/root/transfer_committed.py --host yellow-worm-77 --label yellow_snapshot_001
work/venv/bin/python -B work/goop3d_completion_20261007/root/transfer_committed.py --host aquamarine-toad-75 --label aqua_snapshot_001
```

Use a fresh label on each call. A label collision fails locally before network access. The existing SSH configuration supplies transport; no DNS/proxy setup is changed. Yellow maps only workers 04–07, aqua only 08–11. The source is `/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007/autonomous_results_v1`; teal receives the unchanged worker-relative layout in `collected_autonomous_v1` alongside the source root. Every remote inventory and streaming action verifies the exact hostname and boot ID from the saved `initial_host_observations.json`.

The destination inventory hashes existing committed row/NPZ bytes locally, and the source inventory reads only published JSON plus NPZ file metadata. Matching destination markers and file descriptors remove complete cells from the transfer manifest, avoiding repeated large network transfers. Only cells with a source `commit.json` are eligible. Scientific failure rows remain eligible and are preserved without interpreting their metrics.

The source streams a bound manifest and exact files through local SSH pipes into a teal receiver. Neither NPZ arrays nor model code are loaded. The receiver verifies SHA and size before no-clobber publication; each marker is published after its two referenced files and immutable worker metadata are verified. Existing identical bytes are retained; conflicting bytes are rejected. Concurrent transfers for the same source use a receiver-owned nonblocking lock, separate from worker locks. Distinct hosts have disjoint worker ranges and may transfer concurrently.

Top-level live `owner.json`, `status.json`, locks, logs and temporary/uncommitted cell files are excluded. Published immutable files under `attempts/attempt_NNNNNN/` are preserved: `owner.json`, `previous_owner.json`, `identity.json`, `resume_verification.json`, `runtime.json`, `outcome.json`, and `failed_attempt.json`. This allowlist matches the current worker source's once-published names. A final collection of closure/status evidence remains a separate root action after actual worker closure.

Local receipts live in `root/transfer_committed_<label>/`. Remote receipts and staging live in `collected_autonomous_v1/_transfer_attempts/<source-host>_<label>/`. Successful staged files are hardlinked into the canonical layout, so retention needs no second physical copy. Interrupted staging, prior attempts and per-file receipts are never overwritten or deleted. Restart with a fresh label; complete matching cells are skipped, while uncommitted destination partials are verified before any identical-byte reuse. A completed transfer is a snapshot transfer result, **not** a worker/cohort completion claim. No task-duration cutoff is imposed; SSH connection/liveness checks remain ordinary transport checks.

Run the synthetic checks with:

```sh
work/venv/bin/python -B work/goop3d_completion_20261007/root/test_transfer_committed.py
```

The tests cover marker-last publication, same-hash incremental skip, failed-outcome retention, complete attempt metadata, ignored live/uncommitted files, SHA corruption, truncation/recovery, conflicting destination bytes, unsafe paths/symlinks, changed commit binding, marker order, duplicate JSON, and host/boot mismatch. They do not execute SSH, inspect live hosts, or evaluate science.
