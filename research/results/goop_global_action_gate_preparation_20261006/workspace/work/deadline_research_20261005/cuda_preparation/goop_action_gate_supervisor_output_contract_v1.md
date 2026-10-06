# Scoped gate output contract

Schema: `adaptgns_goop_action_gate_scoped_supervisor_v1`. No output is scientific/test admission by itself. The root collector verifies stopped processes, pinned source/protocol/cohort/head identity and actual arrays separately.

`process_outcomes.json` is written before final ledger checks. It contains `jobs` (exactlyseed0/1/2, each with `entry` and `outcome`), `owned_registry` for every launched child, `unreaped_owned_children`, and `abort_reason`. It remains available when `phase_ledger.json` is absent. A missing ledger does not by itself prove children stopped. Root must inspect actual PID/start identities, all registry return codes, empty unreaped list, and fresh completed-process evidence before accessing changing arrays.

Each entry includes `stream=mix_seedS`, `stage` (driver mode), integerseed, arm=mix, physicalgpu/uuid, exact command, output directory, whole-invocation quota and all `expected_cells`. For final test, each entry has all120 declared source/policy identities; allthree entries preserve360 even if a seed never launches. It is a model/phase ledger, not a survivor metric table.

Ordinary reaped outcome fields inherited from the reviewed D3 process observer:

```json
{
  "started": true,
  "state": "stopped_and_reaped",
  "stopped_and_reaped": true,
  "pid": 123,
  "start_ticks": 456,
  "hostname": "synthetic-example-only",
  "command": ["/absolute/lexical/venv/python", "/absolute/driver.py"],
  "exit_code": 0,
  "elapsed_seconds": 12.3,
  "outer_timeout_seconds": 60,
  "started_utc": "timezone-aware timestamp",
  "exit_observed_utc": "timezone-aware timestamp",
  "signals": [],
  "quota_expired_at_observation": false,
  "quota_stop_initiated": false,
  "inner_timeout_reported": false,
  "current_before_stop": null,
  "worker_failed_attempt": null,
  "worker_failed_attempt_parse_error": null,
  "initial_process_identity": {"pid": 123, "ppid": 122, "start_ticks": 456},
  "observer_lag_included": true
}
```

This illustrative JSON is not an executable release or real process evidence. Actual records also retain resource usage, actual absolute stop, GPU index/UUID and full captured argv identity. A `complete_collection` is attached only after all owned children are reaped and the complete committed scalar grid is checked: `{path,sha256,status,required_rows,committed_rows}`. Large collection parsing and hashes are deferred until then; active polling uses only small status/metadata checks.

`never_started` has started=false,pid=null and retains command/quota/reason. A launched child whose outcome accounting failed is **never** recast as never-started: it has started=true and state `stopped_and_reaped_unclassified` or `unreaped_owned_child`, actual PID/identity/returncode/signals/current, null unverified elapsed/timeout details and an explicit accounting-failure reason. `owned_registry` independently preserves `{entry,pid,identity,returncode,started_utc,signals,cleanup_errors}` for every launch.

Quota expiration at observation is conservative: a late poll does not prove the process ran until that instant. `quota_stop_initiated` identifies active supervisor deadline intervention. `current_before_stop` is the original driver's current cell when available. Nested `failed_attempt.failure.category=execution_budget` is explicitly represented as an inner budget event; original failure JSON is retained. Never treat either quota case as numerical instability. If only some rows committed, the collector assigns timeout/current and remaining missing states from this evidence and original driver records without inventing a full-horizon scalar.

All labels/capacity require complete outcomes; any numerical/infrastructure/budget failure prevents the next seed. Final test permits `complete_with_guard_failures` only when all120 cells committed for that model, the driver returned0 and source/model checks passed; other sources/policies/seeds still run. The collector retains each guarded trajectory and leaves its full395-step metric undefined.

`phase_ledger.json` is published last after stopped source/scalar evidence rechecks. Fields include mode, `state=complete_fixed_phase|stopped_requires_review`, jobs, abort_reason, elapsed_seconds, all_pinned_inputs_reverified, input_sha256 and supervisor_files. `numerical_arrays_audited_by_supervisor=false`, automatic_retry=false and test_admitted_by_this_ledger=false are explicit. `supervisor_files` covers parent files/logs; it excludes `jobs` arrays. Completed child collection hashes are bound separately in their outcomes. The full driver collection inventories and every available numeric NPZ remain the collector's responsibility, including partial arrays and rows in unsuccessful directories.

Per-child `logs/mix_seedS.outcome.json` retains measured timing for the capacity calculator. `process_outcomes.json` and the final ledger carry any later collection verification and nested-budget classification. Preserve all versions and failures; do not promote the timing file alone to complete scientific evidence.
