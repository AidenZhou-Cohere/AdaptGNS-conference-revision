# Goop scoped-v3 scalar collection and fixed paired analysis

This separate adapter bridges the approved `adaptgns_goop_evaluation_gpu_scoped_v3` operational receipts to the unchanged quota-v2 scientific scalar functions. It does not edit or admit old quota-v2 queues under a new label. Frozen v2 source, its prior collections, all failed attempts and all observed failures remain separate.

## Inputs and fixed source identities

The adapter is `summarize_goop_graph_support_scoped_v3.py`. Its `SOURCE_PINS` are exact source byte identities, not filenames inferred from a directory:

| Source | SHA256 |
|---|---|
| Frozen scientific scalar functions, `summarize_goop_graph_support_quota_v2.py` | `da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c` |
| Approved supervisor, `supervise_goop_evaluation_gpu_scoped_v3.py` | `a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21` |
| Frozen evaluator, `evaluate_goop_graph_support_final.py` | `cd970e04012930d896b31944271ccaf7da2b0881f22fe2f903533a8d5911dc6d` |

The required operational amendment is `goop_gpu_scoped_evaluation_amendment_v3.md`, SHA256 `0ecb8d3c6b5dd74150d6253cc84be2548827b235490eb35da50afcc0df6e28f4`. A preserved local copy may be supplied; its hash must equal the amendment digest in the queue release and in the separate collection release. The original remote amendment path remains recorded in `queue_release`.

## Separate root releases

Collection release fields:

- `schema`: `adaptgns_goop_scalar_collection_release_scoped_v3`
- `issued_by`: `root`
- `status`: `approved_for_stopped_scalar_collection`
- `collector_sha256`: exact reviewed adapter source hash
- `source_sha256`: the exact three-entry `SOURCE_PINS` mapping
- `cohort_sha256`: the frozen six-model exact100k cohort JSON byte hash
- `queue_release_sha256`: the queue's original `release_snapshot.json` byte hash
- `operational_amendment_sha256`: the fixed amendment hash above
- `original_queue_root`: the absolute original queue path recorded by its released commands
- `local_queue_inventory_sha256`: the exact relative file/directory inventory digest of the queue tree being collected; compute with `inventory_sha256(tree_snapshot(local_queue_root))` using the reviewed adapter

Analysis release fields:

- `schema`: `adaptgns_goop_paired_scalar_analysis_release_scoped_v3`
- `issued_by`: `root`
- `status`: `approved_for_fixed_scalar_aggregation`
- `collector_sha256`: exact reviewed adapter source hash (same source executes both modes)
- `source_sha256`: the exact three-entry `SOURCE_PINS` mapping
- `cohort_sha256`: the same frozen cohort JSON byte hash
- `collection_sha256`: exactly `A` and `B`, with hashes of the two separately approved collected receipts

Passing a Python dictionary is not a substitute for either release file: file bytes are captured before processing and checked again after output serialization.

## Invocation

Description-only mode is the default. Collection needs explicit `--execute --mode collect --queue-root ABS --cohort ABS --operational-amendment ABS --root-release ABS --output ABS`. The output must be new and outside the queue tree. Analysis needs explicit `--execute --mode summarize --collection-a ABS --collection-b ABS --root-release ABS --output ABS`.

The collector supports either the original stopped host tree or a complete preserved copy. Released evaluator `--output-dir` and committed coverage paths must match `original_queue_root/jobs/<stream>/<stage>`, while local scalar/array reads join only the checked relative paths under `--queue-root`. Original command, protocol and ledger paths remain untouched in the receipt. The root release separately binds the exact local relative tree inventory, so a relocated copy cannot silently replace its origin. Numeric artifacts must be local basenames; escapes are rejected.

The full local-to-collector NPZ byte audit is mandatory in either location. A scalar-only transfer with remote array hash records cannot be passed to collect as a complete queue. For efficiency, root may execute this scalar-only CPU collector on the original VM’s stopped tree, retaining verified arrays there, and transfer the resulting collection JSON and inventory to the Mac. Later public artifact accounting must distinguish remotely retained verified arrays from local JSON copies. No new receipt-only binary-verification mode exists.

## Receipt meaning and snapshot guarantees

The collector requires stopped/reaped owned processes and all eight B or sixteen A stage receipts. It reads JSON from the exact initial byte/hash snapshot; coverage arithmetic uses those snapshots. It verifies referenced NPZ bytes without deserializing arrays, preserves every ordinary partial/temporary file and empty directory, and rejects symlinks or unsupported entries. Full queue entry/hash/size state and the adapter, frozen helpers, cohort, amendment and root-release bytes are checked after aggregation and again after serialization, immediately before exclusive publication.

The collector retains raw GPU inventory, identity annotations, native captured process identities, the process report's inventory hash, the operational amendment, original queue release, status, failure categories and all missing work. A pending observation is preserved as pending, including an inventory rejected by the supervisor; it is never relabeled as successfully validated. Matching/exited owned process observations must agree with captured child identities and released devices. Reused native PIDs on unassigned devices remain observed and uncontrolled. No live process queries or global-idleness assumptions are introduced by collection.

An initial GPU query may fail before any inventory or child exists. The adapter accepts this only with an explicit pre-child abort, null inventory digest, no children or started outcomes, and preserves `gpu_observation_file_state=absent_before_any_child`. This is an absence of observation, not an observed idle host. Missing inventory after any child blocks collection.

The analysis rechecks both collection byte hashes and its original source/root-release bytes after aggregation and serialization. It can consume transferred collection receipts without rereading the original hosts' arrays. This guarantee covers the approved collection bytes, not a fresh audit of remote outputs after collection publication.

## Scientific quantities are unchanged

All 24 stage receipts and 1,080 full-rollout cells are retained. Full-H395 values require complete relevant source coverage; no survivor averaging is introduced. Pointwise H200 can remain defined for an accepted prefix that later fails H395. Full-horizon boundary values remain distinct from failed-prefix diagnostics. Equal-source model values, three paired training-seed values, sample standard deviations, absolute values, within-arm contrasts, mix-minus-base contrasts, and the risk-minus-random mix-minus-base interaction use the exact frozen v2 pure functions. Same-state aggregation remains equal-frame within source, then equal-source, then three paired seeds. Undefined required values remain null.

Runtime is labeled `shared_host_operational_measurement`. Unassigned GPU activity and fixed policy order prevent interpreting these numbers as dedicated-host or causal speedups. This is an exploratory extension informed by prior inspected results; no pristine independent confirmation or conference readiness is established by scalar collection.

The adapter never executes models, opens numerical arrays, accesses reserved sources, launches children, retunes policies, retries failures, changes endpoints, or recomputes historical outcomes. Root's execution admission and independent scientific/array audit remain separate.
