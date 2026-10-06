# Prepared final Goop 2D cohort and scalar analysis adapters

These are separate preparation files. Frozen training/evaluation code and the
scientific protocol remain unchanged. No checkpoint, GPU, validation/test source
or actual scientific output was accessed by the preparation author. Tiny CPU
fixtures are the only executed inputs. Root owns release, actual execution and
remaining scientific array/source audit.

## Completed-host cohort interface

`prepare_goop_final_cohort_quota_v2.py` supports description-only default,
`--execute --mode inventory` and `--execute --mode build`. Inventory reads an
entire completed host training tree, hashes every ordinary file including all
11 checkpoints per model, verifies completed supervisor/child/config/pointer
receipts and binds the exact reviewed scientific quota-v2 source. The output
must be fresh and outside the training tree. It does not remove failures,
partial files or locks; these prevent promotion and remain retained.

Root must first verify stopped process identity and author a JSON process receipt:

- `schema`: `adaptgns_goop_completed_host_process_check_v1`
- `issued_by`: `root`
- `matching_training_processes`: `[]`
- `supervisor_exited`: `true`; `all_children_reaped`: `true`
- `host_role`: `A` or `B`
- `hostname`, `supervisor_pid`, `cohort_id`: exactly the completed launch receipt
- `checked_utc`: timezone-aware timestamp after terminal receipt, no more than
  1800 seconds before inventory starts.

Command arguments: `--host-root`, `--process-check`, `--output`.
After both inventories pass, root authors the cohort mapping release:

- `schema`: `adaptgns_goop_final_cohort_release_quota_v2`
- `status`: `approved_for_cohort_interface_mapping`; `issued_by`: `root`
- `adapter_sha256`: actual reviewed adapter SHA
- `cohort_id`: exactly the common training cohort ID
- `review_rationale`: nonempty explicit root review description
- `inventory_sha256`: object with exact `A` and `B` inventory file hashes
- `final_source_sha256`: exact adapter `FINAL_PINS` mapping, keys
  `protocol`, `train_admission`, `trainer_source`, `benchmark_helper`,
  `data_contract`, `diagnostic_source`.

Build arguments: `--inventory-a`, `--inventory-b`, `--root-release`, fresh
`--output-dir` and one path for every `FINAL_PINS` key, with underscores in flags
replaced by hyphens. Outputs are `cohort_audit.json`, `cohort.json` and the exact
root release snapshot. The cohort/audit retain the frozen evaluator's v1
interface, which accepts quota-v2's unchanged training schema and actual protocol
hash. The test suite exercises the actual frozen `check_cohort` function against
a generated six-model synthetic cohort.

The tensor/Adam/initialization/full100k pairing conclusion is inherited from the
reviewed quota-v2 supervisor's completed audit and is bound to fresh file hashes.
This is **not a second tensor audit**. The nine intermediate checkpoint payloads
are not deserialized by the supervisor or adapter; all eleven checkpoint byte
hashes are retained. Each final evaluator still hashes and validates its selected
checkpoint payload before any inference. Root must explicitly accept this audit
scope; no output claims otherwise.

## Stopped evaluation collection and paired scalar analysis

`summarize_goop_graph_support_quota_v2.py` also describes by default. After each
host's quota queue is stopped and all owned children are reaped, use
`--execute --mode collect --queue-root ... --cohort ... --output ...`.
The collector verifies the saved root release, cohort/selected model/protocol
identity, complete expected coverage grid, individual JSON row hashes and
referenced numeric artifact bytes. It recomputes scalar MSE arithmetic and the
frozen evaluator's clean/same-state aggregation. Failed/missing work is retained;
a quota timeout is not counted as a model guard failure. Existing temporary and
unsuccessful files are hashed and preserved. Collection does not recover or
restart anything.

After both collection receipts exist, root authors:

- `schema`: `adaptgns_goop_paired_scalar_analysis_release_quota_v2`
- `status`: `approved_for_fixed_scalar_aggregation`; `issued_by`: `root`
- `summarizer_sha256`: actual reviewed adapter SHA
- `collection_sha256`: exact `A` and `B` collection hashes
- `cohort_sha256`: exact common frozen cohort hash.

Then use `--execute --mode summarize --collection-a ... --collection-b ...
--root-release ... --output ...`. The summary requires all24 model-stage
receipts, including explicit never-started stages. It reports all1080 planned
full-rollout outcomes, failure categories, prefix boundary diagnostics and
parent whole-invocation durations. Required cell failure/missingness makes the
corresponding all-sample estimand null; unrelated dense failures do not erase a
fully observed risk-versus-random contrast. No survivor means are substituted.

For each metric, average all30 sources equally within each seed, then show all
three ordered paired-seed effects, their mean and sample SD. Output includes
absolute arm/policy means, policy-minus-base/random within each arm, every
policy's mix-minus-base change and the predeclared risk-minus-random interaction.
A negative interaction alone is not risk superiority. Forecast200 and395 are
pointwise errors; a committed accepted200 prefix may support the former even
when the later full horizon failed. Full-horizon means and boundary means still
remain null for that row. The same-state risk diagnostic uses previous observed
native-base risk, separately named from autonomous cached-own-graph risk.

The collector verifies numeric artifact **bytes** and saved scalar arithmetic;
it does **not** independently recompute every graph/benefit/trace metric from
arrays or official source data. That scientific artifact audit remains separate
and must be completed or explicitly reported as a limitation before claims are
made. Fixed-order autonomous call timings describe the executed workload and
do not alone establish a causal policy speedup. Every output says this is an
exploratory Goop extension informed by prior evidence, not pristine confirmation.

## Synthetic validation and preparation outcomes

- Cohort adapter:15 CPU tests passed, including actual frozen evaluator interface.
- Scalar adapter:17 CPU tests passed, including adverse absolute effects despite
  negative interaction, missing required cells, unrelated dense failure,
  pointwise forecast200/395, later failed200 prefix, exact seed SD, duplicate
  rows, changed truth/row/array bytes and actual frozen diagnostic aggregation.
- The first attempted `python3 -m pytest` used system Python3.14, which has no
  pytest. No tests ran. The existing `work/venv/bin/python` ran the tests above.
  This environment-only unsuccessful invocation changed no scientific files.
- Frozen scientific sources and protocol were not edited.
