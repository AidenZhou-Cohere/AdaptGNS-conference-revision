# Goop 3D vectorized implementation and bounded preparation

This frozen package preserves the reviewed vectorized graph implementation,
trainer v2, numerical validator, capacity harness, evaluator, timing preparation,
and scientific supervisor. It includes completed bounded numerical and capacity
evidence. It supplies **no selected scientific endpoint, admitted scientific
training run, efficacy result, or test-set result**. The timing run was active
when this selection was frozen; only its fixed launch declarations are included.

The numerical report records 26 optimizer calls across three prospective size
strata and two arms, with 12 exact original/candidate saved-step pairs and two
checkpoint replay pairs. Its independent scalar/source audit passed 721 checks.
Root separately reread all 32 referenced numerical payloads; the public package
contains their hashes and sizes, not model, gradient, RNG or input-array bytes.
These are bounded implementation checks and do not establish simulation quality.

All six capacity jobs completed 512 updates: base/mix, seeds 0–2, using a wave of
four jobs followed by a wave of two. The independent audit passed 47,096 checks
on 3,072 update rows, 3,072 graph rows, 6,144 examples, 74 scalar/source artifacts,
and 12 omitted checkpoint references. Wave maxima were:

| Engineering quantity | Four-job wave A | Two-job wave B |
|---|---:|---:|
| Steady wall seconds per update, q | 0.5898935053 | 0.5723552509 |
| External lifetime minus all guarded update seconds, r | 83.5739182197 | 68.2740334235 |

These measurements include CPU audit contention and process-exit observer lag.
They are not ideal GPU throughput, a full-study forecast, or evidence for a
scientific endpoint. The exact q/r values and all per-job rows remain in the
original scalar snapshot. The first local capacity audit required exact
Mac/Linux exponentiation equality and failed at two of 512 learning-rate rows
by one float64 ULP in every job. The final audit explicitly permits one ULP in
that cross-platform formula recomputation; saved paired-arm learning rates
remain exactly equal. Both audit sources and attempt logs are retained, and the
frozen training sources and results were unchanged.

The timing launch declarations specify base seed 0 and mix seed 0 on GPUs 0/1,
with full-rollout, same-state and clean-validation waves. The full-rollout
inner/outer limits are 3,600/3,900 seconds; the other modes use 600/900 seconds.
The launch notes record an external 6,300-second watchdog plus 30 seconds of
grace. These are prospective limits, not observed timings. Mutable timing
stdout/stderr, progress and result directories are excluded. Scientific and
deadline templates are inert: endpoint and decision-time inputs remain unset;
no forecast has been run from these templates. Capacity checkpoints are not
scientific models.

`workspace/` preserves the original `work/...` and `outputs/AdaptGNS/...` layout
needed by unchanged source pins and tests. Its support closure copies all 110
files from the earlier [source package](../cuda_graph_support_sources_20261006/README.md)
with exact hashes; those include reused Goop 2D/Sand helpers and old test
fixtures. Only the ten newly listed test files are replayed by this package.
The predecessor families and their manifests remain unchanged.

`records/` contains exact frozen reports, reviews and development histories.
The earlier first-record v1 failure and v2 correction remain in the predecessor
[Goop 3D evidence family](../goop3d_cuda_preparation_20261006/README.md).
Here, `third_vm_pre_capacity_inventory_v1.json` is intentionally a zero-byte,
invalid-JSON artifact from the missing lifecycle-helper preflight. Its adjacent
failure record is a labelled transcription of the observed tool result, not a
claimed raw stderr capture. The corrected v2 preflight is separate. Failed
synthetic fixture/source versions and their final passing reviews are retained
without relabelling the earlier outcomes.

`archives/` preserves six original archives unchanged: the completed capacity
scalar snapshot and five small source-staging archives. The latter intentionally
retain staged source bytes even where readable copies also exist. They contain
no checkpoint or official numeric array payloads. `archive_catalogs.json`
indexes every regular member by hash/size and preserves directory entries. The
8.75 MB capacity summary appears only inside its original archive at
`goop3d_vectorized_capacity_20261006_v1/summary.json`; `omissions.json` records its
exact location and the 32 numerical plus 12 capacity payload omissions.

Historical absolute paths, experiment hostnames and GPU UUIDs are preserved as
nonsecret provenance. They identify the original execution environment; they are
not portable launch instructions. The reviewed `audit_goop3d_*` sources are
provenance records that refer to original staging-relative files and may write
reviews at import time. The package verifier parses them without importing or
executing them. Replaying those historical audits requires deliberately
reconstructing their original scalar input layout and avoiding existing output
paths; no automatic audit replay or scientific launch is supplied here. No SSH,
Coder authentication/session, credential or unrelated personal file is included.

From the repository root, using a Python environment with NumPy, SciPy, PyTorch,
PyTorch Geometric, pytest, CRC32C, TFRecord/protobuf and requests installed:

```sh
python research/results/goop3d_vectorized_preparation_20261006/verify_publication.py
python research/results/goop3d_vectorized_preparation_20261006/verify_publication.py --run-tests
```

The first command checks every file, archive member, provenance hash, omission,
and selected scope condition without executing research code. The second copies
`workspace/` to a temporary directory and runs ten separate CPU test processes
(expected total: 260 tests) using tiny synthetic tensors, generated records,
scalar fixtures and mocked execution. It does not access official numeric data,
real checkpoints, remote machines or CUDA. Historical positive release gates
expire on October 7, 2026 at 01:00 UTC; tests are not rewritten to bypass that
cutoff. `requirements-cuda-lock.txt` records the research environment, not a
requirement to install a GPU build for these CPU checks.

`packaging_verification.json` records the actual packaged test replay and
source hashes. `verification_history/` retains any packaging failures and
repairs. `provenance.json` maps copies to their original workspace-relative
sources; `manifest.json` binds all published bytes except itself. Optional
`--check-originals ORIGINAL_WORKSPACE` and `--check-predecessors` additionally
recheck available original files and predecessor records.
