# Prospective four-GPU Sand feasibility measurement

This is a root-owned infrastructure proposal, not a scientific launch or a
measured forecast. Use only after the admitted-source CUDA parity/replay gate
passes. The known source contract is 1000 training trajectories, 320 frames,
float32 positions and int64 type-6 particles. Validation/test trajectories are
not used by these training measurements.

Use two waves to measure the intended four-job stage and two-job tail of a
six-run workload. Start all jobs in each wave together; begin B only after A
has completed successfully. Every job uses a new directory beneath a visibly
separate infrastructure root. No output from these runs may be resumed,
renamed or promoted into a scientific run.

| Wave | GPU index | Objective | Seed | Required updates | Planned stop |
|---|---:|---|---:|---:|---:|
| A, four concurrent jobs | 0 | faithful | 0 | 100000 | 512 |
| A, four concurrent jobs | 1 | nll | 0 | 100000 | 512 |
| A, four concurrent jobs | 2 | faithful | 1 | 100000 | 512 |
| A, four concurrent jobs | 3 | nll | 1 | 100000 | 512 |
| B, two concurrent jobs | 0 | faithful | 2 | 100000 | 512 |
| B, two concurrent jobs | 1 | nll | 2 | 100000 | 512 |

Pairing is by seed, so objectives receive the same host frame/noise schedule.
Keep GPU indices 2 and 3 idle in B to measure the genuine two-job tail. This
schedule is for capacity admission; it does not estimate an objective effect
on runtime independently of physical GPU assignment. If root instead intends
another scientific scheduling policy, measure that policy before extrapolating.

All jobs use these fixed arguments, filling only the table's objective, seed,
GPU and a unique infrastructure output directory:

```text
python train_sand_cuda.py --execute
  --repo /root/repos/AdaptGNS-cuda-20261006
  --train-manifest /root/repos/AdaptGNS-cuda-20261006/sand_numeric_train_valid_20261006_v1/train.json
  --admission FROZEN_ROOT_ADMISSION.json
  --structural-report FROZEN_STRUCTURAL_REPORT.json
  --protocol FROZEN_INFRASTRUCTURE_MEASUREMENT_PROTOCOL.md
  --output-dir NEW_INFRASTRUCTURE_JOB_DIRECTORY
  --objective TABLE_OBJECTIVE --seed TABLE_SEED --cuda-index TABLE_GPU
  --updates 100000 --stop-after 512
  --threads 2 --checkpoint-every 10000 --log-every 1
```

The endpoint remains 100000 in the configuration, but the only acceptable
success status is `planned_stop_incomplete`, with both `completed_steps` and
`committed_steps` equal to 512. This labels the checkpoint correctly and keeps
the original 100000-update learning-rate schedule. Step-zero and step-512
checkpoints are written. Any missing/nonfinite/failed run blocks the complete
measurement; do not replace it with a favorable subset.

Preserve the frozen command/configuration/hash inventory, stdout/stderr, status,
history, both checkpoint pointers/hashes and external timing for every process.
Use a root-side observer or GNU `/usr/bin/time -v` to capture process start/end,
exit code, total elapsed time and peak host memory. Also record wave start/end
and the observed concurrent PIDs/GPU UUIDs. The job's full process time includes
imports, CUDA startup, complete array-hash validation, model creation,
checkpoints and logging; the trainer's guarded update timer does not.

Record all 512 per-update timings. Predeclare updates 1–64 as warmup and use
65–512 for steady summaries; retain warmup and all failures in the raw record.
For each job report mean, median, p90 and maximum `guarded_update_seconds` in
that fixed window, plus total external elapsed time. Also compute the steady
wall time per update from `(elapsed_seconds[512] - elapsed_seconds[64]) / 448`,
which includes the intervening logging/status work. Logging every update adds
overhead relative to a scientific log interval of 100; keep this distinction
visible in the forecast.

Check that every job has exactly one row for each update 1–512 and that paired
objectives have identical frame IDs, particle counts and learning rates at
each step. The noise schedule is the hash-pinned seed/update host helper; no
objective input enters it. Do not judge feasibility using early training loss
or choose seeds from these results.

For a transparent planning calculation, let `q4` be the slowest job's measured
steady wall seconds/update in wave A, and `q2` the corresponding quantity in B.
The core training projection for this strict two-wave schedule is
`100000 * (q4 + q2)`. Add measured startup and periodic checkpoint/teardown
allowances, a predeclared runtime contingency, and the separately measured
costs of complete validation, all policies/full rollouts, diagnostics, audits
and manuscript work. Retain the full external-minus-guarded-time residual
rather than silently treating it as zero; measure checkpoint I/O separately if
it cannot be isolated reliably. Do not simply multiply a synthetic validator
median or a single 512-step whole-process average by the endpoint.

Root must freeze the forecast formula, contingency and remaining-time cutoff
before looking at feasibility outcomes. A 512-update observation is an
engineering estimate, not a guaranteed upper bound on 100000 updates. Full
scientific runs, if admitted, start afresh in separate directories with their
own frozen scientific protocol and declared logging/checkpoint intervals.
