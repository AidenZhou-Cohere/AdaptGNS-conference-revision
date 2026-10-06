# Original Sand summary capture recovery

The original summary worker session **19303 exited 0**. Its first capture never
connected to the local proxy: the recorded sandbox error is `proxyconnect tcp:
dial tcp 127.0.0.1:56069: connect: operation not permitted`. Original transport
exit255, empty stdout, full stderr/command/external, original root tool exit1,
and original summary worker exit0/attestation are preserved and byte-pinned.
Root's 23:09:50 native observation is additionally pinned: original successor
PID/group48605 remains the exact recorded process and failed local capture
PID/group49548 is absent. That record is an observation, not perpetual liveness.

This wrapper performs exactly one action: call the unchanged frozen
`close_product(state, 'sand_summarize')`, remapping only its transport tag from
`sand_summarize.capture` to **sand_summarize.capture_recovery1**. It never runs
the summary worker, changes a scientific source, resets a clock, fabricates an
old receipt, or replaces the current proxy. Frozen full owner/native/input and
semantic checks publish the previously absent original summary review; a
separate recovery receipt binds that review to the actual new capture tag.

Use workspace Python with explicit `-B`, this source, and these arguments:

```
--root-execute --package-sha256 MANIFEST
--source-review ABSOLUTE_REVIEW --source-review-sha256 REVIEW_SHA
--route-ready ABSOLUTE_EXISTING_SUCCESSOR_READY --route-ready-sha256 READY_SHA
```

The actual network invocation must use root's already-authorized tool with
`sandbox_permissions=require_escalated`; running again in the default network
sandbox would reproduce the local restriction. This is an execution permission
selection, not a source change or a new user approval requirement.

The independent source receipt must have schema
`sand_summary_capture_recovery_independent_review_v1`, status
`passed_source_and_synthetic_review`, and the exact `source_manifest_sha256`.
The existing reviewed base recovery package71c50f55 and its independent
e8e4085f receipt are rehashed first, retaining original phase0ac7a368/anchor
06cee5e7 and original stop **2026-10-06T23:33:01.226132+00:00**. The new route
ready schema and limits are validated by that reviewed base. There is no
action selector, operation override or arbitrary command forwarding.

After root observes this recovery's original tool exit and the actual summary
review is independently checked, the reviewed `continue_original.py` may run
the original paired worker once, then record its actual exit, close its product
and complete the phase. Use explicit network permission for its network actions.
