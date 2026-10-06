# Action-gate scalar numerical core interface v1

`goop_action_gate_scalar_core_v1.py` is pure NumPy/Python: no filesystem, process, simulator, fitting or execution entry point. The separately reviewed operational collector provides exact-byte-verified inputs and injects two immutable modules: `core=goop_global_action_gate_core_v1.py` SHA d3986c2ac3786e9fc0d36d76339bfeded5448497f7dad9b0559efaaaa53a43ca and `pure=summarize_goop_graph_support_quota_v2.py` SHA da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c. Only `pure.boundary`, `mean_complete`, `contrast`, and `seed_summary` are reused; its old scientific model/policy grid is not used.

## validate_row

`validate_row(row, arrays, expected_identity, positions, particle_types, head, core, pure)` returns schema `adaptgns_goop_global_action_gate_validated_scalar_row_v1`.

- `row`: parsed exact committed driver row JSON; `arrays`: dictionary loaded from its exact referenced NPZ with `allow_pickle=False`. Core requires numeric arrays only and exact key/dtype/shape/value SHA descriptors for every array, including rejected/nonfinite arrays.
- `expected_identity`: exactly nine keys: `model_seed`, `source_index`, `policy`, `checkpoint_sha256`, `head_sha256`, `selection_sha256`, `source_manifest_sha256`, `source_trajectory_content_sha256`, `protocol_sha256`. Last is the driver output `protocol.json` byte hash. Constants enforce mix,100k,H395 and the separate gate protocol.
- `positions`: original admitted source float32[401,N,2], all finite. `particle_types`: original int64[N], all7. Scalar type normalization is not permitted; the frozen Goop manifest/driver require vectors. Wrapper checks original manifest array file bytes/dtype/shape/provenance.
- `head`: exact selected fitted head for this model. Wrapper checks its file hash and exact inclusion in the common selection receipt. Core checks head/model checkpoint and pure prediction contract.

Recomputations include accepted-prefix coordinate MSE and geometric boundaries, exact initial/source truth, autonomous learned feature vectors and decisions, keyed gate/pair RNG material, every graph's native-prefix/optional-budget counters, and retained trace graph hashes/pair orientations. No new graph or forecast is generated. Complete full-H summaries remain distinct from failed-prefix diagnostics and pointwiseH200.

Result fields: schema, model_seed, source_index, policy, status, complete, completed_steps, failure, metrics (exact module `METRICS` keys), accepted_prefix_boundary, mse_per_step, timing, realized_cost, all_numeric_descriptors_verified, accepted_errors_boundaries_gates_recomputed. Invalid inputs raise.

## summarize

`summarize(stages,pure)` consumes exactly three stages with integer seeds0/1/2. Each stage has:

```
{seed, scientific_verification_passed:<literal bool>, outcome, runtime,
 rows:[{source_index,policy,row_file,row_sha256,artifact_file,artifact_sha256,
        case_wall_seconds,validated:<validate_row return>}],
 cells:[{source_index,policy,state,failure}], evidence:<wrapper-owned>}
```

Cells are all120 source-major/policy-minor combinations per seed in fixed policy order base/random25/learned_global_gate/validation_rate_random_gate. States are `committed_complete`, `committed_guard_failed`, `committed_execution_failed`, `timed_out_current`, `uncompleted_after_started_invocation`, `never_started`. A committed execution_budget row is `timed_out_current` and still retains its row/NPZ. Missing timed-out-current cells need not have a row. Every other committed state requires a byte-validated row with exact matching failure.

`case_wall_seconds` is optional/None and is supplied only from an exact, ordered, one-to-one match to the committed collection's `case_timings`. It must include its row's reported processing/publication components. The core emits separate `end_to_end_case_wall_seconds` equal-source/paired means, with missing timings remaining null. Driver row publication_seconds omits final rowJSON serialization/write and some descriptor work; these partial component sums are retained and explicitly labeled. Parent whole-invocation/outcome evidence covers setup/finalization/interrupted work.

The result has all360 coverage, absolute policy metrics, three learned-minus-control contrasts for every metric, primary full-H395 contrasts, seed values/mean/sampleSD without undefined-value dropping, failed prefixes, physical diagnostics, realized cost totals, and always-base/no-effective-expansion flags. It never automatically assigns constructive success. Fixed-order/shared-host timings do not establish causal speedup; boundaries do not establish conservation. Validation-only selection diagnostics are the wrapper's responsibility to preserve from the selected receipt.

Each stage requires explicit `scientific_verification_passed`. The operational wrapper sets it only after final model/input/status/collection and reaped-process checks. False retains raw validated rows, failed prefixes and consumed-cost totals as diagnostics, but nulls every admitted seed mean and paired effect, including H200 and case wall time. `all_required_outcomes_complete` requires360 numerically complete rows and all three verified model stages. Separate `all_required_outcomes_numerically_complete`, `scientific_verification_by_seed`, `all_model_stages_scientifically_verified` and `unverified_stage_rows_retained_as_diagnostics` preserve the distinction. A scientifically verified partial final collection can still yield H200 only when all required30 source values for that policy exist.

Failure validation binds phase, forecast index, accepted-prefix length, parity completion and deployed attempt count. Every post-initial phase requires passed native parity/two calls. Numerical current-forward exceptions require an attempted next forward. A quota timer may fire between phase assignment and counter increment; that boundary stays allowed. Post-append scoring failure can retain395 accepted forecasts with failure index396 while its full-horizon mean remains undefined. Native trace self counts reflect cap128 ordering; source-ID ties among coincident particles can cap out some originally added self loops.
