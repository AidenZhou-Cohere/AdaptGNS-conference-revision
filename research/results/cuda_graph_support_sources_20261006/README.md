# Native CUDA graph-support preparation sources

This is a source snapshot supporting the sibling Goop 2D, Goop 3D and Sand
preparation evidence families. It contains the exact reviewed programs, tests,
metadata fixtures and pinned numerical dependencies. `provenance.json` maps
every copied source to its original workspace path, SHA256 and byte count.
The nested `workspace/work/...` and `workspace/outputs/AdaptGNS/...` layout
preserves relative paths used by the unchanged tests and source hash checks.

The Goop 2D scientific runs were active when this snapshot was curated. Their
fixed launch declarations are preserved in the evidence family, but no mutable
training metrics, completed scientific result, or efficacy claim is published
here. The separate Goop 3D trainer is a reviewed candidate with an explicit
prospective endpoint requirement; no endpoint or training admission is supplied.
The later graph-bookkeeping optimization is outside this snapshot.

The reserved Goop 2D test-preparation adapter is included as reviewed source. It
requires the complete six-model 100k cohort and root release before any official
test metadata request or byte read. Its tests use generated records and mocked
HTTP. No official test source is included here.

From the repository root, with NumPy, SciPy, PyTorch, PyTorch Geometric, pytest,
CRC32C, TFRecord/protobuf and requests available:

```sh
python research/results/cuda_graph_support_sources_20261006/verify_publication.py
python research/results/cuda_graph_support_sources_20261006/verify_publication.py --run-tests
```

The first command checks all four family manifests and every compressed evidence
member without extracting it. The second additionally copies the source tree
to a temporary directory and runs each test file in its own process, preserving
the tests that require no GNS module to have been imported. Tests use tiny CPU
tensors, generated data and mocked execution receipts; they do not access real
datasets/checkpoints, CUDA, or remote machines. Historical release tests contain
the declared October 7, 2026 01:00 UTC cutoff; their positive-gate fixtures are
time-bound. `requirements-cuda-lock.txt` records the research CUDA environment,
not a requirement to install that GPU build for these CPU checks.

`packaging_verification.json` records the actual local replay. Initial packaging
attempts exposed missing scalar fixtures/dependencies and then cross-test import
contamination in a combined pytest invocation. The fixtures were copied with
exact hashes and the runner now isolates files; scientific source/tests were not
edited. The available failed packaging logs are retained in `verification_history`.

Evidence families:

- [Goop 2D preparation and frozen launch](../goop2d_cuda_preparation_20261006/README.md)
- [Goop 3D source census and first-record timing](../goop3d_cuda_preparation_20261006/README.md)
- [Sand graph-support capacity and failed complete-horizon timing gate](../sand_graph_support_cuda_preparation_20261006/README.md)
