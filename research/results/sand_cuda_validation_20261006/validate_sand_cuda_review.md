# Actual-data CUDA validator implementation review

Reviewed source: `validate_sand_cuda.py`, SHA256
`60083ba485bf0e99e9fef34b9c76d81a0ad109d517974d5ef25657203fb1081c`.
This is an implementation gate, not an actual-data/CUDA result. Review and tests
used source text, fabricated CPU tensors, tiny linear models and mocked CUDA RNG.
No repository model, source trajectory array, CUDA device, network or remote
execution was used.

The initial wrapper was reviewed independently and the following issues fixed:

- Require all three named cases, both objectives in every case and both replay
  results explicitly; empty or partially populated sections cannot pass.
- Reject frame lengths without distinct eligible first/middle target frames.
  Source-order tie breaking and the `T-6` flattened dataset offset are tested;
  both 320- and 321-frame sources map correctly. Actual admitted T=320 is valid.
- Replay uses original host noise at step zero and step one. Separate CPU and
  CUDA random probes actively advance both RNG streams, without changing any
  physical training input. Both probes and final RNG state must replay exactly.
- Require a parameter change after the first update, exact serialization and
  restoration at one completed update, and numerical resumed outputs/model/
  optimizer agreement under the unchanged pinned replay tolerances.
- Publish replay progress before each stage. A restore or later failure retains
  completed parity results, serialized checkpoint identity and the precise
  failed phase. Numerical failures remain failed results and do not erase other
  completed cases.
- Compare encoder node/edge feature tensors with the fixed forward tolerance;
  ordered edge identities retain the delegated exact comparison.
- Verify structural report and metadata identity again at completion alongside
  trainer, manifest and admission hashes. Source array hashes are verified by
  the admitted loader at startup; this requires immutable numeric inputs during
  execution and is not a continuous source-file monitor.
- Reject invalid CUDA indices, thread counts and trainer hash arguments before
  execution. Existing output directories are never reused.

Root also identified a trainer manifest-count gap. The prepared trainer now
requires `source.member_count == manifest.record_count == admitted record_count`,
with a regression test. Its new SHA256 is
`7a9dec9ac2ae1b49856655441ab52fa881c5bb93d8850108dc52939fdfe6ef3a`.

Focused local verification: 18 wrapper CPU/mock cases and 45 trainer CPU/mock
cases passed together: **63 passed in 1.84 seconds**. Command:

```text
work/venv/bin/python -m pytest -q work/deadline_research_20261005/cuda_preparation/test_validate_sand_cuda.py work/deadline_research_20261005/cuda_preparation/test_train_sand_cuda.py
```

An additional independent read-only agent review closed with no material
finding on the exact final validator hash above. It checked schedule/index
mapping, pinned delegation, finite/gradient/Adam guards, serialization,
CPU/CUDA RNG probe replay, completion and failure paths. It ran no tests,
models or data. Local toy Torch is 2.14.1; the actual target is the previously
validated 2.13.0+cu129 GB200 stack and still requires root-owned execution.

Root owns the final source/admission freeze, actual-data execution and result
classification. A passing implementation check does not establish scientific
accuracy, training completion, real-data throughput or a feasible full cohort.
