# Goop3D scalar deadline decision preparation

This is inert preparation for a later root decision. It reads no timing outcomes,
calculates no forecast, selects no endpoint and admits no process. The timing
supervisor already owns the one released six-call validation attempt. Do not
launch another evaluator while it runs. Scientific results, the earlier Sand,
Goop2D and WaterDrop studies, and the infrastructure timing remain distinct.

## Frozen inputs and exact execution-host paths

Use the existing third host and its original paths:

- Interpreter: `/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python`.
- Capacity summary: `/root/repos/AdaptGNS-cuda-20261006/goop3d_vectorized_capacity_20261006_v1/summary.json`, SHA256 `39f048a7257428b0ffe342e52ed2f956e90cd5dec4463835b40a9d75855a47e1`.
- Validation manifest: `/root/repos/AdaptGNS-cuda-20261006/goop3d_numeric_train_valid_20261006_v1/valid.json`, SHA256 `f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef`.
- Timing inventory, only after root verifies process completion/reaping and retained artifacts: `/root/repos/AdaptGNS-cuda-20261006/goop3d_validation_timing_20261006_v1/timing_inventory.json`.
- Worksheet: `/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop3d_deadline_worksheet_v1.py`, SHA256 `8a4946e813a6b9ef3d0cb86c5649d614d2bc89fd08b9962bf3ec877e94a20db6`.

The original `capacity_summary` and new `planning_config` absolute paths become
worksheet input bindings consumed by `supervise_goop3d_science_v1.py`. Local
copies are archival copies, not replacements for those bindings. Preserve all
six timing identities and failed or missing rows even if no forecast is possible.
The worksheet hashes numeric artifact files as opaque bytes; it does not
deserialize arrays or calculate performance outcomes.

## Root configuration and invocation

Copy `goop3d_deadline_planning_config.template.json` into a new root file named
`goop3d_deadline_planning_root_v1.json` on the execution host. Set
`planning_start_utc` to a freshly observed UTC decision time after the stopped
timing review. Never backdate it to the timing or capacity launch. Keep candidates
`[10000,25000,50000,100000]`, checkpoint cadence 2500, log cadence 100, multiplier
1.35, no evaluation overlap credit, and at least 3600 seconds for transfer,
review and analysis. The compute/analysis target is October 7 01:00 UTC. The
October 7 08:00 UTC no-new-work cutoff does not extend that planning target.

Keep `selected_endpoint:null`; this configuration does not make the scientific
decision. Preserve this template and freeze the root derivative once hashed.
If the decision becomes stale, use a separately named fresh configuration,
release and worksheet; retain the earlier attempt.

Populate `goop3d_deadline_worksheet_release.template.json` from actual host bytes,
including the finalized inventory and fresh planning configuration. Root changes
the derivative's status to `admitted_for_scalar_planning` and issuer to `root`.
Recheck the three prefilled digests rather than trusting the template. Then, once
separately authorized by root, the concrete scalar-only invocation is:

```sh
/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python \
  /root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop3d_deadline_worksheet_v1.py \
  --execute \
  --release /root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop3d_deadline_worksheet_root_release_v1.json \
  --capacity-summary /root/repos/AdaptGNS-cuda-20261006/goop3d_vectorized_capacity_20261006_v1/summary.json \
  --valid-manifest /root/repos/AdaptGNS-cuda-20261006/goop3d_numeric_train_valid_20261006_v1/valid.json \
  --timing-inventory /root/repos/AdaptGNS-cuda-20261006/goop3d_validation_timing_20261006_v1/timing_inventory.json \
  --planning-config /root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop3d_deadline_planning_root_v1.json \
  --output-file /root/repos/AdaptGNS-cuda-20261006/cuda_preparation/goop3d_deadline_worksheet_root_v1.json
```

This command is documentation only; no command has been executed by preparing
this guide. The output file must not already exist. Retain stdout, stderr,
exit status and bytes. A successful worksheet has status
`engineering_worksheet_complete_not_execution_admission`. An unavailable
forecast has `forecast_unavailable`, an empty forecasts list and recorded reasons,
normally with exit 2. Failure before output publication is also retained, and is
not evidence of a valid quota branch by itself.

## Preferred decision using complete full-horizon timing

The frozen worksheet requires both distinct seed-0 capacity models, full H295
for all fixed timing sources/policies, same-state and clean diagnostics, exact
artifact hashes and stopped/reaped processes. It budgets all six scientific
models, up to 30 validation plus 30 test sources, all six full-rollout policies,
five same-state targets per source and 128 clean validation histories per model.
It gives no concurrency credit for evaluation. Runtime depends on final model
states, seeds, geometry, test size and contention; the result is an engineering
estimate, not a measured full-study runtime or statistical bound.

Root may select one of the four candidates only after reviewing the complete
worksheet, fresh clock, remaining resource availability and all reserves. Do not
use validation/test accuracy or probe loss. In the separate
`goop3d_scientific_endpoint_plan.template.json` derivative use:

- `status:prospectively_selected`, `issued_by:root`, the one chosen endpoint,
  `checkpoint_every:2500`, `log_every:100`, and an explicit prospective rationale.
- `selection_basis:prospective_compute_budget_no_validation_or_test_selection`
  and `selection_used_validation_or_test_accuracy:false`.
- `cost_basis:measured_full_horizon_engineering_estimate`, multiplier at least
  1.35, and `evaluation_and_analysis_reserve_seconds` at least the selected
  worksheet's `evaluation_serial_seconds` times that multiplier plus the planning
  transfer/review/analysis reserve.
- Positive fixed audit reserves for waves A and B. A proposed 600 seconds per
  wave is an operational allowance for root review, not a measured audit bound;
  increase it prospectively if needed. Account for these allowances in the final
  release windows in addition to the worksheet total.

The supervisor independently recomputes the worksheet arithmetic. A fitting
worksheet row is necessary, but final wave windows can be tighter after audit,
clock and cleanup allowances. The endpoint must fit both validations. If no
candidate fits, the full-timing branch cannot admit any endpoint from this grid.
Do not turn an unfavorable fit into a failed-timing quota case.

## Conditional fallback when full-H timing actually fails

This fallback applies only if complete full-H timing was attempted and is
unavailable, with the failed evidence preserved and reviewed. A worksheet error
caused solely by a path, hash, release or template mistake does not establish
that full-H timing failed. Preserve and separately correct such preparation
errors without changing the frozen numerical/evaluation sources or retrying
failed scientific outcomes.

If root considers a time-bounded study worthwhile despite failed full-H timing,
make a new explicit prospective operational decision. No endpoint is selected
here. A concrete quota proposal for that decision is:

1. Retain the candidate grid and the same 1.35 multiplier, checkpoint2500/log100.
   Allow **21600 seconds (6 hours)** for final evaluation activity, followed by
   **at least 3600 seconds (1 hour)** for transfer, review, analysis and writing.
   The resulting `evaluation_and_analysis_reserve_seconds` proposal is **25200**.
   These are allocated clock quotas, not predicted runtimes. Root may select
   different explicit quotas prospectively if justified, never after favorable
   partial outcomes.
2. Propose **600 seconds per wave** for stopped endpoint/source/pairing review.
   For each wave, use its already measured capacity rate `q[w]` and residual
   `r[w]` to budget `1.35 * (U*q[w] + (ceil(U/2500)+1)*r[w]) + 600` seconds.
   This guide does not evaluate the expression or choose `U`.
3. Root selects a candidate only if both serial training windows, the declared
   wave review reserves, fresh clock error, 15-second owned-child cleanup per
   wave and the entire fixed evaluation/analysis reserve fit before October 7
   01:00 UTC. If none fits, leave the study unadmitted and report that limitation.
4. The endpoint plan must use
   `cost_basis:fixed_operational_quota_after_failed_full_horizon_timing`,
   `all_required_evaluation_outcomes_promised:false`,
   `full_study_measured_runtime_claim:false`,
   `complete_full_horizon_timing_attempt_failed:true`, and a nonempty
   `preserved_failed_timing_sha256` list. Bind the unavailable worksheet,
   finalized inventory and the specific failure/guard/timeout or incomplete-H
   evidence in the decision's provenance. Preserve identities and reasons for
   diagnostics that were not launched.
5. The quota branch does not shrink the fixed scientific evaluation grid,
   relabel successful prefixes as H295, choose favorable seeds/policies or
   promise all evaluation cells finish. A later root-reviewed evaluation release
   must bind the fixed schedule, exact absolute stop clock, failure/missing-cell
   accounting and owned-process termination checks. Until that release is ready,
   no evaluation launch follows from this plan. All required cells remain in
   denominators; affected complete-cohort means stay undefined.

Record an actual-completion runtime only for the completed process or work it
measures. Never extrapolate a failed prefix to full H295 or describe the quota as
a measured runtime for the complete study. No automatic retry or checkpoint
promotion is admitted.

## Handoff to the scientific release

After the root endpoint decision, fill the separate scientific training admission
and `goop3d_scientific_release.template.json`. Both bind the capacity summary,
worksheet and endpoint-plan hashes. Freeze exact source/protocol/input/interpreter
bytes; retain original source and planning paths. Use one fresh output directory,
four exact GPU UUIDs, hostname, process/clock receipts no older than five minutes,
and matching CLI update/checkpoint/log values. The release explicitly binds
`output_dir`, the two sequential wave windows, cost basis and 01:00 UTC deadline.

The reviewed science supervisor is
`supervise_goop3d_science_v1.py`, SHA256
`7bf35efb06aa92bf84d3e30730412297aae9be62114a692f12e6ee80b55d107e`.
Its `validate_plan` and `validate_release` functions, and the corresponding
independent review, are authoritative for exact schema and window checks.
Every scientific run starts fresh with empty Adam; none of the capacity
checkpoints can be promoted or resumed. The supervisor produces cohort
candidates, not final test or evaluation admission. Root verifies all six
endpoints and freezes the final cohort before reserved test acquisition.
