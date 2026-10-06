# Sand: paired graph-support study from initialization, version 1

**Prospective scientific design, prepared for root review and freeze. This file
does not admit training or evaluation.** Root must bind its final byte hash to
separate complete-cost and scientific releases before launch. All six policies
below are mandatory. There is no five-policy fallback, reduced seed set, shorter
endpoint, or promotion of a capacity checkpoint.

Preserve `sand_graph_support_100k_protocol_draft_v1.md` unchanged at SHA256
`dd89d02b7e4e669870659fd112561f17f229aa7004ac7e151465d541b5f4903a`.
Also preserve the older faithful/NLL proposal, capacity probes, failed numerical
checks and all WaterDrop results as separate evidence families. This version
finalizes the draft's six-policy design and actual reviewed APIs without
inspecting new timing, scientific, or reserved-test outcomes to choose it.

## Scientific question and scope

Does exposing faithful training to random optional interaction pairs change the
relative performance of risk-directed versus random allocation in Sand? The
primary interaction for each training seed is

`I_seed = (MSE_risk - MSE_random)_mix - (MSE_risk - MSE_random)_base`.

A negative interaction means graph exposure improves the risk-versus-random
contrast. It does not establish that risk beats random in the mix arm, that
absolute risk-policy error improves, or that the contrast did not change because
random worsened. Report all arm-policy means, both within-arm contrasts and
each policy's mix-minus-base change. Improvements shared by all policies suggest
broad robustness to changed graphs, not a specific benefit from risk allocation.
The residual-risk head is not an action-value estimator. Random sparse training
support does not guarantee support for score-selected or dense graphs.

This is an exploratory study informed by historical Sand aggregates, WaterDrop
full-model outcomes, graph-convention screens and support-continuation evidence.
Reserved Sand test payloads are not a design input. This does not make the
project pristine independent confirmation. Cross-study WaterDrop/Sand contrasts
also differ in source format, horizon, backend and training intervention; they
do not isolate a material-only effect.

## Data and source lineage

Use the admitted DesignSafe PRJ-3702 version 1 Sand data,
DOI `10.17603/ds2-0phb-dg64`. Preserve original archives and acquisition receipts.
The received train NPZ SHA256 is
`e0b7f68b50702f1af3edfa828b6097e3b28158ddb57631491e60a29658318f7d`;
the received validation NPZ SHA256 is
`f9c29861107b481ad5eb4a340f4f1016fcaf4050cd6cfc8ae3056f148ab6d903`.
These are received-byte hashes; no publisher NPZ hash was available. Retain ZIP
CRC checks, exact numeric dtype/shape/value hashes and original archive order.

The admitted source contains 1000 train and 30 validation trajectories, each
with T=320, float32 positions shaped [T,N,2], int64 particle types with observed
ID 6, no type 3 and no auxiliary fields. The forecast horizon is H=314. Train
particle counts min/median/max are 107/1253/1976; validation is 218/1355/1929.
Train/validation exact-content duplicate checks use the preserved dtype/shape/
value definition. Unsupported source fields, precision, types or shapes stop
admission rather than being dropped or relabeled.

| Input | SHA256 |
|---|---|
| Numeric training manifest | `f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f` |
| Numeric validation manifest | `7c61b7b4633287fe18c74c94fa56f84aa9ce06e9f23002c853eea3a12d8671f4` |
| Training admission | `fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73` |
| Structural report | `bc63fabfc07a663ab866f454c07e984dd57b8d4b32aa06a760838c4f84da72e6` |
| Metadata | `cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0` |
| Converter | `37c135a9eaa484a5fdbed2176780913b71cd3be4df602983fcd3b644c1a57ff9` |

Acquisition, conversion, admission and source records remain separately preserved.

## Reviewed implementation and numerical limits

| File | SHA256 |
|---|---|
| `train_sand_graph_support_cuda.py` | `fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124` |
| `supervise_sand_graph_support_science.py` | `a22a59c46231c328e4939a0f021b3bb2fab9e2f93c0848bb1abd39aad2d5d305` |
| `benchmark_sand_graph_support_rollout.py` | `8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13` |
| `evaluate_sand_graph_support_final.py` | `952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58` |
| `sand_graph_support_policy.py` | `4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a` |
| `measure_sand_graph_support_capacity.py` | `ca9b5ed45455b575179464d35c42d0d03e87edb7508f4deea2451f9f3fb50878` |
| `measure_sand_cuda_capacity_v2.py` | `c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd` |

Every transitive numerical helper is bound by the exact source maps in the
pinned trainer and benchmark. The graph-support schema is
`adaptgns_sand_graph_support_cuda_training_v1`; its checkpoint field is
`cuda_sand_graph_support_schema`, and its training metadata field is
`cuda_sand_graph_support_run`. Older Sand and WaterDrop checkpoint schemas are
rejected. Keep old evaluators and frozen training sources unchanged.

This is a distinct deterministic native-CUDA lineage. Both earlier strict
CPU/CUDA gradient gates remain **FAILED**. ReLU backward-mask counterfactuals
explain observed threshold differences but leave a small/NLL Adam-update
discrepancy; no altered mask or backward rule enters training. Neither those
diagnostics nor this design establish matching CPU/CUDA optimization paths.

Previously reviewed same-CUDA graph-support checks establish exact old-native
versus new-base first-update tensors on the three selected source-size batches
and same-device checkpoint replay under the checked inputs. The mix replay's
post-resume update is unexpanded: it restores an earlier expanded state but does
not directly test an actively expanded post-resume update. These checks do not
prove 100k convergence, every graph geometry, or cross-device bitwise identity.
Retain their full reports and failed predecessor evidence. Timing-only numerical
admission is not scientific release. All scientific arm/policy comparisons use
the declared CUDA variant; no CPU-equivalence claim is permitted.

## Six fixed training models and host allocation

Train faithful-only models for arms {base,mix} and seeds {0,1,2}, each from
initialization for exactly 100000 updates. The full model uses width 128,
10 message-passing blocks, two MLP layers, nine-type/16-wide embedding, and the
original scalar variance head. Use all eligible six-frame training histories,
batch size 2, uniform sampling with replacement, noise 6.7e-4 and the original
noise-adjusted normalization. Faithful mean MSE and detached variance-feature/
residual-target semantics are unchanged; variance floor is 1e-6.

| Host role | Physical CUDA index | Arm | Seed |
|---|---:|---|---:|
| A | 0 | base | 0 |
| A | 1 | mix | 0 |
| A | 2 | base | 1 |
| A | 3 | mix | 1 |
| B | 0 | base | 2 |
| B | 1 | mix | 2 |

Run these six models concurrently. Host B GPUs 2/3 remain idle. The frozen
supervisor queries all host GPUs: before launch the entire host must be idle,
and any unrelated GPU process on B indices 2/3 is rejected during monitoring.
Do not schedule Goop, timing, or other GPU work on either host concurrently with
this scientific worker. Root records actual host identities, all four UUIDs per
host, launch/process/clock evidence, observed concurrency and the fixed mapping.
This is six-job execution on two four-GPU hosts, not eight-job scaling evidence.

Use one process per assigned GB200, two CPU threads, Torch 2.13.0+cu129 and CUDA
12.9, float32 and strict deterministic algorithms with `warn_only=False`.
TF32, AMP, compilation and DDP are off. Require
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`;
leave `CUDA_VISIBLE_DEVICES` unset. Freeze executable/package/environment and
source/configuration/data/protocol bytes in each root release.

Pair initialization, sampled frame IDs, particle counts, host-noise bytes,
learning rates and optimizer hyperparameters by seed. Data and noise schedules
use the original pinned helpers at completed steps 0..99999; arm, process order,
host and GPU index do not enter their seed material. Adam decays exponentially
from 1e-4 to 1e-5 over 100000 updates, with betas (.9,.999), eps 1e-8, no weight
decay or gradient clipping, `foreach=False` and `fused=False`.

Each model saves step-zero state before its first update, then checkpoints at
10000,...,100000, giving 11 checkpoint files. The scientific supervisor audits
initial model, simulator configuration, empty Adam and CPU/CUDA RNG identity
after all its children finish. There is **no pre-step-1 pairing-audit barrier**.
The audit is mandatory before root admits the complete cohort; a mismatch is a
failed pairing, not permission to select favorable endpoints.

Log scalar training rows at step 1 and every 100 updates (1001 rows). Save all
100000 per-update graph/schedule records plus model, Adam, CPU/CUDA RNG and
configuration. Check every scalar row against stdout/status and its graph
record. Paired saved LR equality remains exact. At most one float64 ULP is
permitted only when recomputing the logged analytical LR locally, with every
discrepancy retained; the training schedule is not changed.

## Training graph intervention

Build each example's native graph on its noisy observed current positions.
Retain the exact native directed mandatory prefix: radius r=.015, strict native
norm/query semantics, receiver cap 128, self candidates, and native receiver/
distance/source order. Preserve cap asymmetry and possible self omissions under
pathological equal-distance ties. The optional universe is the uncapped strict
geometric radius R=1.267r unordered pair set minus the uncapped strict geometric
radius-r set. Short pairs omitted by the native cap are not annulus candidates.

For example slot j and completed update u, draw one uniform from
`SeedSequence([20261005,seed,u,j,4409])`; the exposure coin is draw<.5.
`SeedSequence([20261005,seed,u,j,5501])` supplies the independent pair permutation.
These streams omit arm and do not advance Torch, frame or noise RNG.

Base adds no optional pairs. When its coin succeeds, mix selects exactly
floor(.25*annulus_count) pairs uniformly without replacement; otherwise it adds
zero. Canonicalize selected unordered pairs, emit both orientations in the
pinned bridge order and append after the unchanged native prefix. Do not recap,
symmetrize the base, add optional self edges, connect examples or inspect future
targets. Zero budget is exactly native-base even when the coin succeeds.

Preserve frame/noise IDs; each example's coin and RNG material; particle, native,
self and cap counts; annulus count/budget; selected count; ordered base/optional
hashes and noisy-state hash. Audit all committed updates and paired native
inputs. Report realized exposure and zero-budget frequency without selecting
models on those quantities.

## Cost gate, release and failure handling

Before launch, root must review the distinct six-model graph-support capacity,
full-horizon evaluation timing and terminal-ledger evidence. The old staged
faithful/NLL rate cannot serve as measured base/mix capacity. Timing probes and
synthetic terminal envelopes remain infrastructure artifacts, never endpoints.

Use `sand_graph_support_evaluation_timing_protocol_v1.md`, SHA256
`b22b74d6829fdf8385f534b668d8b3bbbadbfe4bde37dfc8e64d58ba7b7c47c2`.
It fixes all six streams, all six 512-update probes, full H314 cases and both
diagnostic modes. Its prospective whole-study cost is

```
full_rollout_seconds = 1.5 * max_models(180*slowest_complete_case + 100*invocation_overhead)
diagnostics_execution_seconds = 1.5 * max_models(
    200*same_state_15_frame_invocation + (128/15)^2*clean_15_frame_invocation) + 2700
training_seconds = 1.35*(100000*q6 + 12*r6) + ledger_total_reserve_seconds
remaining_seconds = full_rollout_seconds + diagnostics_execution_seconds + 3600
total_compute_analysis_seconds = training_seconds + remaining_seconds
```

Here q6 and r6 are the largest measured steady update time and nonnegative
external-minus-guarded residual across all six graph-support probes. Invocation
overhead is total elapsed minus the sum of full-case times. The 100-fold and
200-fold factors conservatively charge accumulated JSON rewrite growth under
comparable row size; whole diagnostic scaling intentionally overcounts forward
and setup work. The extra 2700 seconds covers larger final100k loading, ledger
validation and cohort integrity, subject to review against measured terminal
size/work. These margins are planning allowances, not confidence bounds or
empirical worst-case guarantees. Any required incomplete timing case leaves
full deadline fit undefined. Later model/test geometry can exceed the forecast.

The separate terminal-ledger reserve must cover all 11 publications and final
history under intended concurrency/storage. The reviewed synthetic worker's
path includes full-ledger validation, D2H copies, serialization/fsync, binary
hash, synthetic pointer/two-status publication and final history JSON. It omits
live model finite/Adam predicates, live RNG capture and remaining payload
construction; root explicitly reserves these. Multiplying the synthetic path
by 1.35*11 does not by itself establish coverage of omitted work.

Require compute/analysis completion by **2026-10-07 01:00 UTC**, leaving seven
hours for writing/review before the **08:00 UTC** no-new-work cutoff. The
scientific supervisor requires

```
training_stop <= 01:00 UTC - remaining_seconds
cleanup_trigger = training_stop - clock_error_bound_seconds - 15 seconds
latest_start <= min(01:00 UTC - total_compute_analysis_seconds,
                    cleanup_trigger - training_seconds)
current_host_time + clock_error_bound_seconds <= latest_start
```

Root binds a complete six-job forecast, source/data hashes, process evidence
and reviewed clock error in [0,5] seconds to each scientific release. Process
and clock checks must be no older than 300 seconds. The worker rechecks launch
fit, copied/live bytes and current GPU inventory before spawning. It records
GPU/status observations every 30 seconds, polls/reaps children at 0.2 seconds,
and reserves bounded owned-child cleanup before evaluation time. Operational
observations are not proof of uninterrupted six-way concurrency.

There is no automated retry, resume, seed replacement or early endpoint
substitution in this supervisor. Keep unsuccessful state/context/history,
temporary files, logs and every failed attempt. A future operational recovery
requires a separate root decision using exact native PID/parent/argv/start,
dead-lock and checkpoint/config/source checks; do not run duplicate trainers or
retry a numerical failure without review. Current one-attempt host admission
does not silently accept such a recovery. If the full declared study does not
fit or cannot complete, retain and report it as incomplete without shrinking
the scientific comparison after outcomes.

## Complete-cohort and reserved-test admission

Only all six final100000-update checkpoints can define this cohort. Verify clean
endpoint/status/pointers, all graph/history rows, finite model and Adam moments,
step100000 counters, complete source/data/protocol lineage, initial pairing and
saved schedules. Check the exact simulator architecture/bounds, finite float32
two-coordinate normalization means/scales, positive scales and initial/final
configuration equality. Normalization construction follows pinned CUDA code;
the CPU inspector does not recompute CUDA square roots.

The supervisor checks initial/final checkpoint hashes/payloads and required
intermediate filenames; it does not deserialize all nine intermediate payloads.
Preserve and hash the retained intermediate artifacts for reproducibility.
Cheap per-child `verification=passed` during reaping means only clean-exit/
complete100k status. Only `verified_host_scientific_endpoints` certifies that
host's full endpoint/pairing inspection. Root must check both host summaries
and all six identities before issuing the separate final cohort and audit.

The final evaluator requires
`adaptgns_sand_graph_support_final_cohort_v1` and a byte-pinned
`adaptgns_sand_graph_support_complete_cohort_audit_v1` with all six endpoints
and three pairings verified. Root retains underlying evidence; an unsupported
declaration is insufficient. Every invocation verifies its selected checkpoint
and pinned source/admission bytes; the complete audit covers other-host models.
The final entry does not deserialize all six checkpoints on every call.

After the complete cohort and sources are frozen, root may acquire/decode/admit
the declared 85825802-byte Sand test NPZ. Preserve receipt/hash, ZIP CRC,
conversion/array/manifest hashes, source order and complete duplicate/accounting
checks across train/validation/test. Require all 30 test trajectories, T320,
type6 and the same precision/features. A discrepancy stops admission; no
convenient subset is substituted. Final split admission binds the cohort,
scientific protocol and exact source/manifest/report hashes. No evaluator
acquires data. Do not label these results pristine independent confirmation.

## Six policies and autonomous full rollouts

For every final model, run `--mode full-rollout --split test` on all 30 source
trajectories and all 314 forecasts after the first six observed frames. The
ordered API policy list is

`base, dense, random25, speed25, laggedrisk25, relative-velocity-RMS25`.

This yields 180 complete required outcomes per model and **1080** across the
cohort. Base retains the native prefix; dense adds the entire annulus; each of
the four budgeted policies appends exactly floor(.25*available_annulus_pairs).
Scored policies rank optional pairs by maximum endpoint score with existing
deterministic lexicographic ties. Speed uses the last-step current-history
speed. Random seed is `93000+1000*training_seed+source_index`, omitting arm;
different autonomous geometry may produce different later draws.

`laggedrisk25` is the autonomous cached-risk policy. It pays an initial native-
base scoring pass on observed history, uses its full budget from forecast 1,
then caches risk from its own preceding selected graph and predicted history.
Every policy executes the same checkpoint's risk head. Retain synchronized
end-to-end graph/selection/transfer/scoring/forward time, initial parity/warmup,
trace/artifact I/O, final verification and total invocation time. Full rollout
policy order is fixed as above; no balanced autonomous timing order is claimed.

For `relative-velocity-RMS25`, cast the last two current positions to float64
before subtraction and divide by physical dt=.0025. For particle i, use the
retained incoming native-base nonself senders N_i and

`score_i = sqrt(mean_{j in N_i} ||v_i-v_j||^2)`, with score 0 for empty N_i.

Accumulate float64 squared two-coordinate differences in native edge order via
`numpy.add.at`, divide by integer neighbor counts and take the square root.
Rank/select the same exact annulus budget and symmetric suffix. Keep cap/order/
asymmetry; do not replace support with an uncapped geometric neighborhood. This
is local relative motion, not uncertainty or action value. Retain physical
score/support-count hashes and common-state score arrays; charge all score,
graph and selection work. The Euclidean score is rotation-invariant conditional
on support, while cap/tie behavior can still affect the full policy.

Use the frozen guards: nonfinite state/output/risk, nonpositive variance,
absolute coordinate exceeding 10 or more than 100000 candidate unordered pairs.
Retain failure phase/forecast, rejected arrays when available and accepted
prefixes. A wrapper timeout may not return an internal native prefix; record
that absence explicitly. Implementation/parity failures stop further inference
for review. Numerical/resource failures remain recorded outcomes. These are
computational guards, not physical-validity criteria.

Record predicted and same-frame truth boundary fractions and mean/maximum
excursions outside metadata bounds [.1,.9]^2, plus initial observed boundary
behavior. Boundary diagnostics neither assert conservation nor become the
failure threshold. Keep finite-but-physically-poor rollouts visible.

## Clean validation and common-state diagnostics

`--mode clean-validation --split valid` evaluates 128 distinct source-order
flattened history indices `floor(i*(9420-1)/127)`, i=0..127. Each flattened
index maps to source=index//314, target_frame=index%314+6, with observed input
`positions[target_frame-6:target_frame]`. Use zero noise, saved training
normalization and the native-base graph. Retain raw head, floored variance,
normalized prediction/target and vector squared residual arrays. Metrics are
normalized acceleration coordinate MSE, realized vector squared error,
predicted vector squared error (2*variance), and constant-free two-dimensional
isotropic Gaussian NLL `0.5*vector_squared_error/variance + log(variance)`.
These final-only diagnostics do not select checkpoints.

`--mode same-state --split valid` and `--split test` each use all 30 trajectories
at zero-based targets 7,85,163,241,319: 150 frames per split/model. Current
observed history is `positions[t-6:t]`; previous is `positions[t-7:t-1]`.
Selection, prediction and timing complete before the target is accessed.
Same-state risk is `previous-observed-base-risk25`, computed from the previous
observed native-base history, and is distinct from autonomous `laggedrisk25`.

The ordered diagnostic policies are
`base, dense, random25, speed25, relative-velocity-RMS25, previous-observed-base-risk25`,
plus `natural_base_reference` solely as a timing/construction reference.
Random material is `[20261006,93000,training_seed,split_index,source_index,target_frame]`,
with split index 0 for valid and 1 for test. It omits arm and resets for each
repeated call. Use one warmup round and seven cyclic timed rotations; every
policy/reference occupies each slot once. Each risk call pays its own previous-
base scoring pass. Current/previous native-versus-supplied parity contributes
four forwards; warmup adds eight and timed rounds add 56: **68 per frame**.
Six models times two splits times 150 frames therefore require 122400 forwards;
clean validation adds 2304 (three per frame), totaling **124704**.

Retain native/supplied parity outputs, exact edges/prefix/budgets, all six policy
predictions/risks, repeated predictions, physical scores, residual arrays and
signed actual base-minus-selected-policy squared-error reductions. Compare risk
with base error and with each signed action utility using per-frame Spearman;
retain undefined reasons and coverage. Report dense/sparse benefit-sign
disagreement and dense-positive/sparse-nonpositive fractions. Dense benefit is
not a substitute for realized sparse-policy benefit. Check natural/shared base
edges exactly and outputs within the frozen parity tolerances: prediction
absolute 2e-7, risk absolute 1e-6 and relative 1e-5. No tolerance tuning follows
scientific outcomes.

Aggregate particles within frame, frames equally within each trajectory, then
trajectories equally, and training seeds equally. Keep validation/test separate.
Missing or undefined required values leave that complete metric null. Preserve
per-policy values and failure coverage even when another policy makes the frame
status failed. Report synchronized timing phases and total pipeline wall time;
edge counts, fixed device assignment and divergent geometry alone do not imply
causal speedup.

## Estimands and reporting

Full-horizon coordinate MSE averages particles/coordinates within forecast,
all 314 forecasts within trajectory and all 30 trajectories equally within
seed. Compute risk-minus-random within matched seed/trajectory, then mix-minus-
base. Report all three seed interactions, their mean and sample standard
deviation, with the underlying absolute arm-policy means. Negative MSE
differences favor the first term. There are **three independent training-seed
pairs**, not particles, frames or trajectories as independent training repeats.

Report full-horizon failure counts and paired failure indicators beside MSE.
A failed/missing rollout required by a complete-sample contrast makes that
contrast null; do not impute a finite MSE or compare only survivors. Accepted-
prefix and completed-case summaries remain separate with denominators. Recorded
computational failures define failure indicators; missing executions stay
missing. Any incomplete model leaves the complete six-model study incomplete.

Secondary contrasts are mix-minus-base for each policy; risk-minus-speed,
risk-minus-base, risk-minus-dense and dense-minus-base within each arm;
risk-minus-relative-velocity and relative-velocity-minus-random within each
arm; and the relative-velocity-versus-random mix-minus-base interaction. Report
forecasts 200 and 314 as fixed secondary endpoints, all seed signs, null reasons,
failures and boundary effects. Trace forecasts are 1,10,50,200,314. Relative
changes require explicit positive denominators. Do not select favorable
horizons/trajectories, pool frames for significance, or search for p-values;
n=3 supports descriptive consistency, not strong significance claims.

Final complete-cohort paired analysis and an independent saved-result audit are
separate from the per-model evaluator. Root must preserve raw arrays/scalars,
source/configuration/data/checkpoint hashes and failed outcomes, verify the
analysis against this fixed estimand, and distinguish historical arrays,
compact pilots, smoke/capacity tests, native-CUDA science and earlier WaterDrop
families. Native LaTeX compilation, reproducible released code/results, evidence
and checklist updates, exported PDF/anonymity/rights review and author
verification remain required. This protocol makes no conference-readiness or
submission claim.
