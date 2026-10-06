# Sand graph-support evaluation capacity, version 1

Prospective infrastructure measurement only. Freeze this source before observing
evaluation timings. Use every one of the six complete, verified 512-update
graph-support capacity checkpoints: base/mix by seeds 0, 1, 2. These are not
scientific endpoints, and their accuracy is not a basis for changing the study.
Preserve failures and all earlier capacity and CPU/CUDA validation evidence.

Use the same A4/B2 host/GPU assignment as the graph-support capacity protocol.
On each assigned GPU execute full-rollout, same-state, then clean-validation
timing modes, without overlap on that GPU. Start all six streams together after
both capacity supervisors have ended and their identities/status are checked.
Keep the remaining two GPUs idle. Record cross-host launch and GPU observations;
do not claim six-way measured concurrency without actual overlap. Do not run
other GPU workloads during these measurements.

All inputs are the unchanged admitted Sand validation split. The benchmark
chooses source-small/lower-median/large cases by (particle count, source index),
with no outcome-based selection. All six policies are mandatory: base, dense,
random25, speed25, laggedrisk25, relative-velocity-RMS25. Full rollouts cover every
H=314 step. Each checkpoint therefore has 18 complete trajectory-policy cases.
The diagnostic modes use the same three sources at targets 7, 85, 163, 241, 319.
The benchmark includes the reviewed scoring, parity, warmup and timing work,
and separately records setup/publication and total invocation elapsed time.
Each mode has a 1800-second bound; any incomplete required case leaves a
complete-horizon forecast undefined. No automatic retry or guarded-prefix
extrapolation is allowed.

Frozen source SHA256 values:

- benchmark_sand_graph_support_rollout.py:
  8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13
- sand_graph_support_policy.py:
  4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a
- evaluate_sand_graph_support_final.py:
  952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58
- train_sand_graph_support_cuda.py:
  fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124

Each invocation requires its own root release binding checkpoint bytes, arm,
seed, mode, data admission, training protocol, this protocol, source hashes and
the complete six-policy list. Source and checkpoint bytes remain immutable.
The numeric results are retained as infrastructure evidence, not promoted to
scientific estimates. No reserved test data may be fetched for these checks.

## Prospective whole-study planning calculation

Forecast the fixed scientific scope: 1080 full rollouts (180 per checkpoint),
300 common-state frames per checkpoint (150 validation plus 150 test), and 128
clean-validation frames per checkpoint. This is 124704 diagnostic forward calls
across the complete cohort. The benchmark covers 15 frames per diagnostic mode.

For each checkpoint let R be the slowest of its 18 complete full-rollout case
durations, E be total invocation time, and S be the sum of those case durations.
Define overhead O=max(0,E-S). Use the following conservative operational forecast
under the measured six-stream assignment:

```
full_rollout_seconds = 1.5 * max_over_models(180 * R + 100 * O)
diagnostics_execution_seconds = 1.5 * max_over_models(
    200 * same_state_total_invocation_seconds
    + (128 / 15)**2 * clean_validation_total_invocation_seconds)
    + 2700
```

The slowest full case is applied to every trajectory and policy, rather than
averaging favorable smaller cases. The evaluator rewrites an accumulating JSON
result after each case/frame. Applying quadratic ratios to the full overhead
(18 to 180 cases), and to the entire diagnostic invocation (15 to 150 frames
on each of two splits, and 15 to 128 clean frames), conservatively covers that
growth under similar record size. This intentionally overcounts linear forward
and setup costs. An additional 2700-second planning reserve covers larger final
100k checkpoint loading/ledger validation and final cohort integrity processing:
45 minutes across the concurrent six-model evaluation path, subject to review
against the measured terminal ledger sizes and load/validation costs. This is a
declared engineering allowance, not an empirical worst-case bound. Six-stream
observations support this assignment only; no ideal linear
GPU speedup or eight-job scaling follows. The factor 1.5 is a planning margin,
not a statistical bound. Future 100k model geometry and uninspected test states
can exceed validation timings; monitor actual execution and preserve overruns.

Combine these quantities with the separately measured training rate, explicit
terminal-ledger reserve, and 3600 seconds for aggregation/verification. Require
forecast completion before October 7, 01:00 UTC. Scientific training still needs
a separate full source/protocol/release review; this document authorizes none.
