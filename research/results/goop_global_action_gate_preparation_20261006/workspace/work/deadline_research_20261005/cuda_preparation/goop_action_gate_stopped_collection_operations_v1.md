# Goop action-gate stopped numerical collection

This is preparation for root review. It does not authorize a test phase or operate any process. Use the final independent core/collector receipts before actual collection. The frozen trainers, driver, numerical policies, scientific protocol and existing results remain unchanged.

`summarize_goop_action_gate_v1.py` combines methods-owned process/input/byte checks with statistics-owned `goop_action_gate_scalar_core_v1.py`. It runs on the original host where the released queue, admitted source arrays, selected heads and original checkpoints remain accessible. It does not deserialize checkpoints, import a simulator, signal a process, fit a gate or launch inference. The result is a scalar JSON suitable for transfer after the original numeric bytes have been audited. All source arrays, NPZ files, unsuccessful rows, opaque partial artifacts and original logs remain in place.

## Process closure before reading arrays

Root first verifies the original supervisor has exited and been reaped, inspects `process_outcomes.json`, and independently verifies every entry in `owned_registry`. A missing `phase_ledger.json` is allowed; it is not evidence of process exit. The complete registry, every reaped return code, empty `unreaped_owned_children`, and current PID/start identities remain mandatory.

Complete `goop_action_gate_process_closure.template.json` using actual observations. The status must become `all_gate_processes_stopped_and_reaped`, the issuer `root`, and the timestamp timezone-aware. Bind the exact queue release and process-outcomes byte digests. The supervisor identity must retain its original argv, including exact lexical Python, `--execute`, the original `--release` path and the queue `--output-dir`. Supply one child proof per actual registry entry, in registry order:

```json
{
  "seed": 0,
  "identity": {
    "pid": 0,
    "ppid": 0,
    "start_ticks": 0,
    "argv": [],
    "executable": "illustrative placeholder, never evidence"
  },
  "reaped": false,
  "exit_code": null
}
```

Do not fill placeholders with guessed identity or return codes. Root evidence must match the actual original capture and parent PID. For a launched child whose supervisor accounting failed, a separate root identity/reaping observation is necessary; it remains a started unsuccessful child. Never-started jobs have no child proof, no registry entry and no stream directory. The collector rejects live original identities and any currently executing gate driver/supervisor, including a different phase. A reused PID with a different start tick is recorded without control. The root receipt must be at most 30 minutes old when collection starts; the collector independently checks process closure again immediately before publication.

## Root collection release

After closure, snapshot the entire original queue with `tree_snapshot` from the reviewed collector and compute `inventory_sha256` over its canonical JSON. The inventory includes every directory and file, including empty directories and partial artifacts. No queue contents may change after this snapshot.

Complete `goop_action_gate_stopped_collection_release.template.json`. The status is `approved_for_stopped_numeric_collection`, the issuer `root`, and `hostname` must name the original host. `queue_root` is the exact original absolute queue path. `output` is a fresh scalar JSON outside the queue, with an existing ordinary parent directory. `collector_sha256` and `source_sha256` bind the independently reviewed collector and its exact `SOURCE_PINS` mapping. `queue_release_sha256` binds `release_snapshot.json`; `process_outcomes_sha256` binds the recorded registry/outcomes. `phase_ledger_sha256` is the actual ledger digest when present, otherwise explicitly null. `process_closure` binds the new independent root receipt.

`files_sha256` must include the complete original parent/child input closure, original parent release path, original child release paths, the closure receipt, all driver protocol inputs, frozen source files, exact venv/binary pins, cohort and complete cohort audit, original training admission/protocol, capacity receipt, all selected heads and common selection, test manifest/metadata/structural/acquisition/auxiliary/cross-split evidence, all 30 original position/type/auxiliary arrays and immutable context sources. Checkpoints are read only as opaque bytes for SHA verification. Preserve the exact lexical venv interpreter path; resolving it can discard the intended environment.

The collector reuses the frozen driver scalar admission at the original recorded admission timestamp, plus the original cohort, reserved split and context gates. It does not call the execution admission `release_gate` or bypass its fresh-output rule. The original gate phase remains closed; there is no new execution admission.

## Collection and output interpretation

Run the reviewed collector with the original lexical interpreter, `-B`, `--collect`, `--queue`, `--release` and `--output`. Use the admitted deterministic environment already required by the original cohort gate. Redirect stdout/stderr to a new retained log outside the queue. No source edits, retries, checkpoint loading or GPU commands are involved.

Every planned model/source/policy is represented: three mix100k seeds, 30 official test sources, four policies, 395 forecasts, 360 cells. Committed rows require exact JSON/NPZ bytes, original source/head/model/protocol identity and the core's numeric verification. Invalid or unreadable records retain a `collector_validation_error`, their file evidence and a missing metric; they are not fabricated into guard outcomes. Missing cells distinguish a known current timeout, an uncompleted started invocation and a never-started invocation. An execution-budget row retains its numeric prefix and is classified `timed_out_current`. Scientific guard rows retain the failure, accepted-prefix diagnostics and any defined pointwise H200; incomplete H395 metrics stay null.

Each stage also has explicit `scientific_verification_passed`. Its original driver model-state and input reverification must both have passed; collection, final status, row/tree evidence and reaped invocation must agree. A completed stage additionally needs the exact original supervisor `complete_collection` commitment. A newly computed root inventory never replaces that earlier commitment. A partial nonzero-exit stage can retain scientific verification when its complete retained terminal evidence explicitly verifies the unchanged model and inputs. Otherwise its numerical rows and consumed costs remain diagnostic evidence, while its seed's admitted scientific means and paired effects are null, including H200. The committed model's configuration digest must match the original cohort configuration.

Whole-case wall time is read only from a valid, exactly matched final collection `case_timings` receipt. The row's processing/publication components do not include every serialization operation; absent case receipts produce null end-to-end means. Parent whole-invocation timing remains separate and includes setup, finalization and interrupted work. Fixed policy order and shared-host timing do not establish a causal speedup.

`status=stopped_numeric_collection_complete` means the stopped collection audit completed; the nested summary separately states whether all 360 outcomes were recorded or completed. `stopped_collection_contains_invalid_evidence` is an audit failure requiring review even if other rows validate. All partial file hashes and each invalid record remain in the scalar output. The summary never uses survivor means, automatically declares constructive success or assigns conference readiness. Validation oracle/benefit diagnostics are copied only from the frozen selection and explicitly remain validation-only, nondeployable evidence.

Source bytes, full queue inventory, root receipt and scalar serialization are rechecked before exclusive publication. If any terminal check fails, no summary is published; retain the command's failure log and every original artifact. Root must investigate the original failure before any separately authorized attempt. The collector never performs an automatic retry.
