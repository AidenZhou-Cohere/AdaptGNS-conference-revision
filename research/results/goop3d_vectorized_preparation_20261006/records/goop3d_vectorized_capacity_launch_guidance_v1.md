# Root-owned capacity launch guidance

This note supplies bindings and argument structure. It is not a release and does
not execute anything. The reviewed source/protocol/template remain unchanged.

## Template and exact release changes

Start from `goop3d_vectorized_capacity_release.template.json`, SHA256
`209faa4d8410de128325ba8e0f2a323d80782fad35de3075b217506988c28d26`.
Write a new root release file, leaving that template unchanged. Root must set:

| Field | Required value |
|---|---|
| `status` | `admitted_for_capacity_only` |
| `issued_by` | `root` |
| `files_sha256.numerical_report` | `c5367c9e1c3dbd11a9ef280bd8dac81f7bd8e8b35674ab94dcc2e7d4c105f552` |
| `files_sha256.python` | SHA256 of the resolved executable supplied by `--python` on the target host |
| `numerical_report_independently_reviewed` | `true`, after the actual numerical review passes |
| `review_rationale` | Nonempty root rationale linking the exact numerical and capacity source reviews |
| `hostname` | Target host's exact `socket.gethostname()` |
| `gpu_uuids` | Four raw `nvidia-smi` `GPU-...` UUID strings in CUDA-index order0,1,2,3 |
| `process_identity_checked_utc` | Fresh timezone-aware root inventory timestamp; age at supervisor validation≤300seconds |
| `data_contract` | Exact contract from the reviewed actual numerical release, described below |

Retain every other template field, including schema, purpose, source/data/protocol
hashes, fixed4+2 schedule, environment,512updates,64warmup and100000probe LR
horizon. Both scientific flags remain false. The horizon is a capacity parameter,
not a selected endpoint. Do not add endpoint selection or scientific admission.

The actual numerical report is `goop3d_vectorized_numerical_v1_report.json` with
the hash above. Scalar inspection shows `implementation_passed`, all six
small/median/large × base/mix cases, two completed comparisons per case,
5/5/4/4/4/4 graph gates,26optimizer calls, both exact checkpoint replays and
`all_inputs_reverified=true`. It reports the required GB200, Torch2.13.0+cu129,
CUDA12.9, two-thread deterministic runtime. Its131.27second validation duration
is not a capacity measurement.

## Source-only data contract

The `data_contract` in `goop3d_vectorized_numerical_release_v1.json` already has
the exact required source-only fields. Its bytes are bound by the numerical
report's `release_sha256=ce2fd036b49c0e2310a0e639131c3ef1eb62cbf7564cffd1bf04a420128117c7`.
Root may copy that nested object unchanged after verifying the release hash.
It equals the capacity template's object except that its `status` is `admitted`
and `issued_by` is `root`.

The nested schema is `adaptgns_goop3d_training_admission_v2`, used by the pinned
complete-source validator. Its scope is `bounded_implementation_data_only` and
`scientific_training_admitted=false`. It binds the actual1000train records,
T301/D3,295000six-frame histories, type7, `<f4` positions and `<i8` particle types,
and every source/converter/reader/context/auxiliary hash. It explicitly preserves
and excludes `step_context` according to the released-parser contract. It must
not contain `prospective_endpoint_updates`, `endpoint_selection_basis`, or
`selection_evidence_sha256`. The nested admitted status describes source access
only; the outer release describes capacity only.

## CLI inputs and file mapping

These local scalar evidence copies all match the release's fixed hashes. On the
target host supply the corresponding exact-byte files. In particular,
`--train-manifest` must point to the real numeric-data tree containing the
manifest's referenced arrays and adjacent metadata, not merely its local scalar
copy in `goop3d_structural_metadata_v1`.

| CLI argument | Local evidence/reference file |
|---|---|
| `--train-manifest` | `goop3d_structural_metadata_v1/train.json` |
| `--structural-report` | `goop3d_structural_metadata_v1/structural_report.json` |
| `--acquisition-report` | `goop3d_acquisition_v1_report.json` |
| `--context-semantics` | `goop3d_context_semantics_review.json` |
| `--auxiliary-report` | `goop3d_auxiliary_v1_report.json` |
| `--data-review` | `goop3d_data_compatibility_root_review_v1.json` |
| `--cpu-proof` | `goop3d_vectorized_cpu_comparison_v1_report.json` |
| `--protocol` | `goop3d_vectorized_capacity_protocol_v1.md` |
| `--numerical-report` | `goop3d_vectorized_numerical_v1_report.json` |

The source entry is `measure_goop3d_vectorized_capacity_v1.py`, SHA256
`dc72264af052ac32e13c0ef1af4746b2cf5c7a25fb85117d89cb03c553871af5`.
Its sibling pinned dependencies are trainer v2, vectorized graph, numerical
validator and Sand lifecycle v2. The validator's loader additionally needs the
original `goop3d_graph_support.py`; keep it at its original reviewed hash. All
trainer common-core and official context-source files must remain available at
their expected relative paths. `--repo` is the staged AdaptGNS repository root,
containing `research/` and `adaptive-gns/`.

The argument vector is exactly the following structure, with absolute target
paths substituted by root. The supervisor creates the six fixed worker commands;
do not invoke the workers separately or pass `--job` to the supervisor.

```text
PYTHON -u PREP/measure_goop3d_vectorized_capacity_v1.py
  --execute --mode supervise
  --repo ADAPTGNS_REPO --python PYTHON
  --release ROOT_CAPACITY_RELEASE --output-dir FRESH_CAPACITY_OUTPUT
  --train-manifest ACTUAL_DATA/train.json
  --structural-report STRUCTURAL_REPORT
  --acquisition-report ACQUISITION_REPORT
  --context-semantics CONTEXT_REVIEW
  --auxiliary-report AUXILIARY_REPORT
  --data-review DATA_REVIEW --cpu-proof CPU_PROOF
  --protocol PREP/goop3d_vectorized_capacity_protocol_v1.md
  --numerical-report NUMERICAL_REPORT
```

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`
before starting the supervisor. Leave `CUDA_VISIBLE_DEVICES` absent. Use the
dedicated four-device host after every device is free of compute processes.
The output must not exist and must be separate from the source/core and data
trees. The inherited lifecycle enforces one3600second limit per wave, up to
15seconds cleanup, and the2026-10-07 01:00UTC compute cutoff. Root retains the
whole-invocation timeout and device/process ownership.

Success is the complete supervisor `status.json` plus `summary.json` with
`all_six_verified_capacity_only`, both observed waves, all checkpoint/scalar
audits and final source/artifact rechecks. Preserve output even on failure.
The summary's selected endpoint and full-study runtime forecast remain null.
