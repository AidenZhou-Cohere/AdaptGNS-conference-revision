# Sand native-CUDA capacity: failed V1 and completed V2

This family records infrastructure timing for the original **native-base graph,
faithful/NLL objectives, seeds 0/1/2, in sequential four-job and two-job waves**.
Every successful probe stopped at 512 of the requested 100,000 updates. These
checkpoints are incomplete infrastructure artifacts. This is separate from the
later faithful base/mix graph-support study and its proposed six-job concurrency.

The V2 training-only planning estimate is **46,841.893 seconds (13.012 hours)**:

| Quantity | Observed value |
|---|---:|
| Four-job wave maximum steady wall seconds/update, q4 | 0.139080583 |
| Two-job wave maximum steady wall seconds/update, q2 | 0.198057743 |
| Four-job maximum external residual seconds, r4 | 32.944958436 |
| Two-job maximum external residual seconds, r2 | 49.043859380 |

The frozen formula is `1.35 * (100000 * (q4 + q2) + 12 * (r4 + r2))`.
The factors are engineering allowances, not statistical bounds. Wave assignments,
seeds and execution time differ; the q2/q4 ratio does not estimate GPU scaling.
**Full-study completion time and deadline fit remain null** because full rollout
and diagnostic execution costs were not measured. Neither a scientific cohort nor
CPU/CUDA equivalence is admitted. The earlier strict CPU/CUDA failures remain
failures; the separate natural-CUDA timing assessment is included unchanged.

## Preserved failures and scope

V1 failed because its supervisor compared Torch's bare GPU UUID with NVIDIA's
literal `GPU-`-prefixed spelling. The physical device identities agree. NLL seed 0
finished 512 updates, but its supervisor verification failed. Cleanup interrupted
faithful seed 0 at 500 updates, faithful seed 1 at 482, and NLL seed 1 at 477;
all three had committed only their initial checkpoint. No second wave started,
and no owned child remained unreaped. V1 remains a failed attempt.

V2 made the separately reviewed UUID canonicalization change and used a fresh
release/output directory. All six probes verified. The compact report contains
all 3,072 update rows and 1,536 exact faithful/NLL paired comparisons. The original
supervisor performed checkpoint-byte, source/configuration, process and wave
checks on the research machine. The compact public report does not reproduce
those remote checks by itself.

An initial local scalar audit failed when macOS recomputed the Linux/ARM learning
rate one float64 ULP differently. The exact failed verifier and failure review are
preserved. The revised scalar-only audit allows at most one ULP for this local
recomputation and retains exact saved learning-rate equality between paired arms.
Its 20,056 checks passed. All 12 observed local differences remain recorded:
updates 16 and 274 in each of six jobs. Saved training values and the original
training/supervisor sources were not changed.

## Contents and reproducibility

`manifest.json` hashes every other published file. `curation_provenance.json`
records exact copied source bytes and two explicit public projections. Private
execution-host labels and ephemeral run-lock tokens were removed only from the
V1 runtime/progress projections; every remaining decoded JSON value is preserved.
The private Coder control-process dump is omitted, with its original hash retained.
No credentials or connection files are included.

The family includes both supervisors, the exact trainer and its parent/delta
needed by the runtime tests, protocols/releases/preflights, process checks,
independent reviews, the complete scalar summary, both scalar audit sources,
the training manifest and structural report. These last two are metadata and
hash inventories, not trajectory arrays.

From a checkout with Python and pytest available:

```sh
python research/results/sand_cuda_capacity_20261006/verify_publication.py
python research/results/sand_cuda_capacity_20261006/verify_publication.py --run-cpu-checks
```

The optional checks run the included synthetic supervisor/runtime tests and
scalar audits in a temporary copy. They do not import a real Torch model, open
trajectory arrays/checkpoints, launch a capacity process, use CUDA or contact a
remote host. They never overwrite the archived scalar audit. Local `pow`
recomputation differences may depend on the CPU/libm platform.

`omitted_raw_inventory.json` records 17 known checkpoint references and their
hashes, the raw data/metadata identities, and missing local proof. In particular,
V2's original launch/status/wave-observation files and full per-job logs were not
available in this local curation. Per-job protocol/status/history/pointer hashes
are retained in the summary, but those raw files were not copied. Raw checkpoint
and trajectory bytes remain on the research machine; no new retrieval or model
execution was performed to build this family.

See the sibling `sand_cuda_numerical_mechanism_20261006` and
`sand_cuda_validation_20261006` families for the unchanged numerical validation
history, and `sand_data_admission_20261006` for data admission provenance.
