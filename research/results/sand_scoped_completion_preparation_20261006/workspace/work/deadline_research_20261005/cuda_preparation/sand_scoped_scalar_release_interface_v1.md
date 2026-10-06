# Sand scoped scalar collection and paired analysis v1

This separate collector preserves the original Sand scientific estimands while adapting to the new scoped operational queue. It never edits frozen sources or runs models. It follows the independently reviewed Goop scoped-v3 collector's snapshot, origin, root-release and native-identity checks. It does not relabel a Goop or earlier Sand queue as a new study.

## Exact scientific and operational bridge

- T320, fullH314, pointwise forecast200/314; two-dimensional boundary diagnostics.
- Six faithful100k models, base/mix×seeds0/1/2. A: base1/mix1/base2/mix2 on GPUs0/1/2/3; B: base0/mix0 on GPUs2/3.
- Four stages/model: full_rollout_test, same_state_valid, same_state_test, clean_validation. All24 receipts, including missing work, are mandatory for paired analysis.
- Full rollout: all30 sources×six policies×six models=1080 required cells.
- Same-state: targets7,85,163,241,319 on every source,150 frames/model/split.
- Clean validation:128 flattened indices floor(i*(9420-1)/127), mapped to source n//314 and target n%314+6.
- Optional policies, strict graph conventions, physical guards, failed-prefix treatment, equal-source means, three paired seed effects, sampleSD and all original contrasts remain unchanged.

The seed/mean/contrast/two-dimensional-boundary primitives and diagnostic paired aggregation are privately imported by exact hash from `summarize_goop_graph_support_quota_v2.py` (`da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c`). Sand has explicit `expected`, `validate_rollout` and `full_summary` functions: it does not monkeypatch Goop's H395 globals or pass Sand rows through the Goop horizon validator. Frozen Sand `evaluate_sand_graph_support_final.summarize` supplies within-frame/equal-frame/equal-source diagnostic summaries.

The four exact `SOURCE_PINS` entries are the frozen scientific primitive source, the new `supervise_sand_final_evaluation_scoped_v1.py` adapter, its underlying immutable `supervise_goop_evaluation_gpu_scoped_v3.py` engine (`a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21`), and frozen Sand final evaluator (`952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58`). Use the final independently reviewed values from the collector's source; do not freeze an intermediate supervisor candidate.

Other required identities: original scientific protocol `e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d`; trainer `fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124`; benchmark `8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13`; scoped operational amendment `sand_scoped_operational_amendment_v1.md`, `411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738`.

## Root release contracts

Collection release:

- schema `adaptgns_sand_scalar_collection_release_scoped_v1`
- issued_by `root`; status `approved_for_stopped_scalar_collection`
- collector_sha256: exact reviewed Sand scalar source
- source_sha256: exact four-entry SOURCE_PINS mapping
- cohort_sha256: complete original frozen six100k cohort JSON
- queue_release_sha256: original queue release_snapshot.json
- operational_amendment_sha256: fixed amendment above
- original_queue_root: original absolute queue path
- local_queue_inventory_sha256: `inventory_sha256(tree_snapshot(queue_root))` computed by this reviewed collector on the complete tree being collected

Analysis release:

- schema `adaptgns_sand_paired_scalar_analysis_release_scoped_v1`
- issued_by `root`; status `approved_for_fixed_scalar_aggregation`
- collector_sha256: exact reviewed Sand scalar source (same source executes both modes)
- source_sha256: exact four-entry SOURCE_PINS mapping
- cohort_sha256: same complete frozen cohort JSON
- collection_sha256: exactly A and B, each bound to its separately approved collected JSON

Queue/status/coverage/process schema must be `adaptgns_sand_evaluation_gpu_scoped_v1`; queue release schema must be `adaptgns_sand_evaluation_gpu_scoped_release_v1`. Cohort schema/status remain `adaptgns_sand_graph_support_final_cohort_v1` / `frozen_for_final_evaluation`. Every source/protocol/model hash in stage receipts must agree with its frozen cohort and release. A different source/schema or incomplete model grid is rejected, not coerced.

Use explicit `--execute --mode collect --queue-root ABS --cohort ABS --operational-amendment ABS --root-release ABS --output ABS`. Summary uses `--execute --mode summarize --collection-a ABS --collection-b ABS --root-release ABS --output ABS`. Without --execute, source is description-only. New output is exclusive and outside the queue tree. Each release is an actual file whose bytes remain bound through serialization; an unbound dictionary is insufficient.

## Scope and preservation

Every referenced NPZ must exist locally to the collector and its bytes must match its committed row. No array is deserialized. Root may execute this CPU collector on the original VM's stopped output tree and transfer the resulting collection JSON and relative inventory to the Mac; public accounting must distinguish the verified arrays retained on the VM from the local JSON copy. A complete relocated copy is also supported, using the explicit original_queue_root plus relative local inventory binding. A scalar-only transfer cannot bypass array-byte checks.

Original remote command, protocol, outcome and coverage paths remain unchanged. Released output/cell paths must agree with the original queue/stage; array reads use local basenames only, rejecting escapes. Initial full file/directory/hash/size inventory precedes row parsing. JSON parsing verifies exactly those bytes. Original source, cohort, amendment and release hashes plus the full stopped tree are rechecked after aggregation and again after serialization immediately before exclusive publication. Analysis similarly rechecks the approved A/B collection bytes and its source/release bindings; it does not imply a later fresh read of remote arrays.

All owned Sand children must be reaped. B's unassigned GPUs0/1 can contain unrelated observed activity; the collector neither controls it nor assumes a globally idle host. Preserve every raw GPU observation and its native-identity annotations, including rejected/pending inventories. Passed owned identities must match captured launch identities and released devices. Reused native PIDs on unassigned devices remain observed and uncontrolled. Initial GPU query failure can produce explicit absent_before_any_child inventory only when no child/started outcome exists and an abort is recorded; absence is not an observation of idleness. Missing inventory after a child blocks collection.

Preserve all ordinary temporary files, empty directories, guard failures, unsuccessful prefixes, timeout/current work and never-started cells. A complete H314 mean is null if any required relevant source is failed/missing; no survivor means are introduced. H200 is pointwise and can remain defined in a prefix that fails later. Full-H boundaries remain distinct from failed-prefix geometric diagnostics. Runtime is shared_host_operational_measurement; fixed policy order and unrelated contention do not establish dedicated-host or causal speedup. An unverified source-input integrity receipt is preservable but cannot enter scientific aggregation.

Root's scoped scheduling amendment retains the04:00UTC compute/analysis deadline,03:00 evaluation cutoff and3600-second analysis reserve. Collection creates no new time allowance, scientific execution, recovery permission or readiness claim. Sand, Goop, Goop-3D, WaterDrop, capacity probes and historical arrays remain separate evidence families.
