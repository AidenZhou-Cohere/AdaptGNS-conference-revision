# Sand: paired graph-support study from initialization

**DRAFT v1 — no scientific release, launch or endpoint admission.** Root chose
this question before Sand capacity outcomes or reserved test access. Preserve
`sand_full_100k_protocol_draft.md` unchanged: that older proposal contrasts
faithful and NLL objectives. Preserve its six faithful/NLL capacity probes as
infrastructure. Neither their checkpoints nor their two-wave runtime forecast
become scientific evidence for this distinct base/mix study.

## Scientific question and interpretation

Does exposing faithful training to random optional interaction pairs change the
relative performance of risk-directed versus random allocation in Sand? The
primary interaction is

`I_seed = (MSE_risk − MSE_random)_mix − (MSE_risk − MSE_random)_base`.

A negative interaction means that random graph exposure makes risk allocation
relatively less harmful or more beneficial compared with random allocation. It
does not by itself show risk is better than random in the mix arm: report both
within-arm differences and all absolute errors. Improvements shared equally by
all policies indicate a broad graph-robustness effect, not a specific allocation
benefit. A learned residual-risk head is not an action-value estimator. The
mixture supplies random sparse support; it does not guarantee support for
score-selected graphs or the complete dense annulus.

This study is exploratory. Historical Sand aggregates, WaterDrop full-model
results, graph-convention screens and support-continuation evidence informed
its design. Sand reserved test payloads have not been inspected. Cross-study
WaterDrop/Sand comparisons also differ in source format, horizon, backend and
training intervention; they do not identify a material-only causal effect.

## Source and numerical lineage

Use the admitted DesignSafe PRJ-3702 version1 Sand data, DOI10.17603/ds2-0phb-dg64.
Retain the original archives and source receipts: train NPZ SHA256
`e0b7f68b50702f1af3edfa828b6097e3b28158ddb57631491e60a29658318f7d`,
valid NPZ SHA256
`f9c29861107b481ad5eb4a340f4f1016fcaf4050cd6cfc8ae3056f148ab6d903`.
These are received-byte hashes; no publisher NPZ hash was available. Preserve
ZIP CRC checks, exact dtype/shape/value hashes and lossless archive ordering.

Admission establishes1000 train and30 valid trajectories, everyT=320, positions
`<f4` of shape[T,N,2], particle types`<i8` with observed ID6, no type3 or auxiliary
fields. Eligible forecast horizon isH=314. Train N min/median/max is107/1253/1976;
valid is218/1355/1929. Train/valid exact content-duplicate checks passed under
the preserved dtype/shape/value definition. Numeric train/valid manifest SHA256:
`f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f` /
`7c61b7b4633287fe18c74c94fa56f84aa9ce06e9f23002c853eea3a12d8671f4`.
Structural report SHA256:
`bc63fabfc07a663ab866f454c07e984dd57b8d4b32aa06a760838c4f84da72e6`.
Metadata SHA256:
`cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0`.
Converter SHA256:
`37c135a9eaa484a5fdbed2176780913b71cd3be4df602983fcd3b644c1a57ff9`.
Unsupported fields, shapes, types, precision or split properties stop admission
rather than being dropped, relabeled or truncated.

Use a distinct deterministic native-CUDA lineage. Both prior strict CPU/CUDA
gradient gates remain FAILED. The ReLU backward-mask counterfactual explains the
observed threshold failures, while a residual small/NLL Adam-update discrepancy
remains. It does not establish matching cross-device optimization trajectories.
The root numerical assessment currently admits timing only. No diagnostic mask
or altered backward rule may enter scientific training. All scientific arm and
policy comparisons must share the declared deterministic CUDA backend.

The candidate `train_sand_graph_support_cuda.py` needs its own source review,
synthetic budget/faithful-isolation tests, actual-data base-path and within-CUDA
replay checks, capacity measurement, checkpoint/evaluator lineage support and
root scientific release. Final source hashes are deliberately pending. Existing
faithful/NLL evaluators must reject this new schema until separate wrappers are
reviewed; never edit their frozen contracts to accept another lineage silently.

## Six fixed training models

Train faithful-only models for arms{base,mix} × seeds{0,1,2}, exactly100000 updates
from initialization. Every arm uses the same width128,10-message-passing-block,
2-MLP-layer architecture, nine-type/16-wide embedding and scalar variance head.
Use all eligible six-frame training histories, batch2, uniform sampling with
replacement, original noise6.7e-4 and original noise-adjusted normalization.
Faithful mean MSE and detached variance-feature/residual-target semantics are
unchanged; required variance floor is1e-6. The base arm supplies a matched
100k-training control, not an imported existing checkpoint.

Pair initialization, sampled frame IDs, particle counts, host-noise bytes,
learning rates and optimizer hyperparameters by seed. The original pinned
`sample_indices` and `host_noise` helpers use completed update0..99999. Different
arm, process order, GPU index or host must not alter these schedules. Initialization
weights must have equal tensor hashes within each seed before any update.
Adam LR decays exponentially from1e-4 to1e-5 over100000 updates; betas(.9,.999),
eps1e-8, no weight decay or gradient clipping, foreach=False,fused=False.
Checkpoints every10000 updates; training scalar logs every100. Save full model,
Adam, CPU+CUDA RNG, configuration and every per-update graph/schedule ledger.

No validation/test checkpoint selection, favorable seed replacement, early
endpoint substitution or mixing of infrastructure and science outputs. Only all
six final100000-update model hashes define the planned cohort. Numerical failure
requires review and preservation of failed state; any incomplete required model
leaves complete-cohort scientific metrics undefined. Operational recovery uses
fresh native PID/parent/argv/start checks, dead-lock guards, exact input/config/
checkpoint bytes and reviewed restoration; never start duplicate trainers.

## Training graph intervention

Build each example's native graph on its noisy observed current positions.
Retain the exact native directed mandatory prefix: radiusr=.015, strict native
float32 norm/query semantics, receiver cap128, self candidates and native
receiver/distance/source ordering. Preserve possible asymmetric cap effects and
self omissions in pathological equal-distance ties.

The optional universe is the uncapped strict geometric radiusR=1.267r unordered
pair set minus the uncapped strict geometric radiusr set. Short pairs omitted by
the native cap are not optional annulus pairs. Never connect examples or use a
future target. For each example slotj and completed updateu, draw one uniform
from `SeedSequence([20261005,seed,u,j,4409])`; exposure coin isdraw<.5. A separate
`SeedSequence([20261005,seed,u,j,5501])` supplies a pair permutation. These streams
omit arm and touch neither Torch nor data/noise RNG.

Base never adds pairs. Mix, on a successful coin, selects exactly
B=floor(.25×annulus_count) pairs uniformly without replacement, otherwise0.
Canonicalize selected unordered pairs, emit both directed orientations in the
pinned bridge order, then append them after the unchanged mandatory prefix.
Do not recap, symmetrize the base or add optional self edges. Directed edge
count is native_count+2B when exposed. B=0 is exactly nativebase even if the
coin succeeds. A batch may have0,1,2 exposed examples.

Retain each update's frame IDs, noise hash, per-example coin/RNG identity,
particle count, native/self/cap counts, annulus size, selected budget, ordered
base/optional hashes and noisy-state hash. Audit exact budgets and pairing across
all committed updates. Report realized exposure and zero-budget frequency; never
select models by their realized exposure.

## Backend, capacity and final release

One process per assigned GB200; Torch2.13.0+cu129/CUDA12.9,float32, strict
`torch.use_deterministic_algorithms(True,warn_only=False)`, TF32/AMP/compile/DDP
off,2CPU threads, CUBLAS_WORKSPACE_CONFIG=:4096:8 and the reviewed driver library
path. Freeze exact executable/environment, host/GPU UUIDs, complete source/core/
protocol/configuration/data and numerical admission hashes before launch.

Use `sand_graph_support_capacity_proposal_v1.md` for the prospective six-job
concurrency measurement. Hardware availability is not a speed measurement.
Graph-mix work and100k-sized ledger validation/checkpoint costs must be measured
or conservatively bounded; the older faithful/NLL capacity rate is contextual
only. Include all final evaluation/diagnostics/verification work. Require
predicted compute/analysis completion by2026-10-07 01:00UTC, preserving seven
hours for writing before08:00UTC no-new-work cutoff. If the complete cohort and
required evaluation do not fit, defer the study as a whole rather than reducing
seeds/endpoints/controls. A favorable forecast is necessary, not automatic
scientific admission.

## Final validation and reserved test

Only after all six final model hashes and scientific/evaluation sources are
frozen may root acquire/decode/admit the listed85825802-byte Sand test NPZ.
Retain its received hash, CRC, conversion/array/manifest hashes, original order
and complete duplicate/accounting checks across splits. Require30 test records,
T=320, types/features/precision matching the declared contract. Any discrepancy
stops evaluation for review; do not select a convenient subset.

Final clean validation uses128 distinct indices
`floor(i*(9420−1)/127)`, i=0..127, in source-order flattened30×314 eligible
histories. Use zero noise and the nativebase graph. Save per-frame normalized
coordinate MSE, realized/predicted vector squared error and constant-free2D
isotropic Gaussian NLL, plus raw head and floored variance. Average frames within
trajectory then trajectories equally; all128 identities remain available.
These final-only diagnostics do not select endpoints.

For autonomous test evaluation use all30 trajectories, allH=314 forecasts from
the first six observed frames, all six models and all five policies:900 outcomes.
Policies are nativebase, dense(full annulus), random25, speed25 and cachedrisk25.
Use precisely the same mandatory/optional semantics as training. All budgeted
policies appendfloor(.25×available annulus pairs), dense appendsall. Rank pairs by
max endpoint score with deterministic lexicographic ties. Speed is current-history
last-step speed. Random rollout seed is93000+1000×training_seed+source_index and
omits arm; different autonomous geometries may produce different later draws.

Every policy executes the same checkpoint's risk head. Cached risk pays one
nativebase scoring pass on the initial observed history, uses its exact budget
from forecast1, then caches scores from its own previous selected graph and
predicted history. Record synchronized end-to-end graph/selection/transfer/
score/forward/runtime including this warmup; show I/O/verification separately.
Balance policy timing order prospectively where supported. Autonomous geometry,
fixed device assignment or edge count alone cannot establish causal speedup.

Guards: nonfinite state/prediction/risk, nonpositive required variance, absolute
coordinate>10, or>100000 candidate undirected pairs. Preserve first failure
phase/forecast, rejected state and accepted prefix; do not silently discard
failed policies. These are computational guards. Physical validity is separately
reported as predicted and same-frame truth boundary fractions and mean/maximum
excursions from metadata bounds[.1,.9]^2.

## Prospective optional sixth physical baseline

Before scientific release, root may include relative-velocity-RMS25 only if its
reviewed implementation and fully charged measured cost fit the same deadline.
The decision must use implementation/timing evidence, never probe accuracy,
scientific model outcomes or reserved test results. Record inclusion or deferral
and its reason in the final protocol; do not add/drop it after scientific launch.
The five core policies and primary risk-versus-random interaction remain fixed.

For particlei letv_i be its last observed position difference divided by the
metadata physical timestep, and letN_i contain the retained incoming nativebase
nonself senders. Define

`s_i = sqrt(mean_{j in N_i} ||v_i − v_j||²)`, with `s_i=0` ifN_i is empty.

Use the same current history, mandatory cap128/self/order/asymmetry and exact
annulus universe. Rank optional pairs bymax(s_i,s_j), break ties by the existing
lexicographic rule, and append exactlyfloor(.25×annulus_count) symmetric pairs.
Derive scores from native message support; do not replace it with an uncapped
geometric neighborhood. Fix arithmetic/reduction dtype and order in source
before release; a global timestep factor does not affect ranking. This score
measures local relative motion, not uncertainty or error. Its Euclidean norm is
rotation-invariant, while neighborhood cap effects remain. Charge construction,
velocity/reduction, ranking and forward work to the policy's measured cost.

If included, all six models receive all six policies on all30 test trajectories:
1080 fullH314 outcomes. Add the baseline to every common-state frame and balanced
timing rotation: seven policy/reference slots, eight forwards per round including
risk's extra scoring call,4parity+8warmup+56timed=68forwards/frame. The workload
becomes122400same-state+2304clean=124704forwards. Both extra180rollouts and27000
common-state forwards must be forecast. Preserve all core five-policy outcomes
under the final declared timing order; do not silently reuse timing from the
five-policy implementation. Additional secondary contrasts are risk−relative-
velocity within each arm, relative-velocity−random, and its mix−base interaction.

## Estimands, pairing and failure accounting

Full-horizon coordinate MSE averages particles/coordinates within step, all314
steps within trajectory and all30 trajectories equally within seed. Compute
risk−random within each matched seed/trajectory, then the mix−base interaction;
this equals a difference of complete equal-weight seed means. Show all three
seed effects, their mean and sampleSD, with underlying arm-policy means. Negative
MSE differences favor the first term. The independent training replication
count isthree paired seeds, not particles, frames or trajectories.

Full-horizon failure counts and paired failure indicators accompany the primary
MSE interaction. A failed/missing required rollout makes the corresponding
complete-sample MSE contrast null. Do not impute failure as a finite MSE, compare
survivors only or condition away a harmed policy. Report accepted-prefix and
completed-case summaries separately with coverage. Failure indicators remain
defined for recorded computational failures; missing executions remain missing.

Predeclared secondary contrasts: mix−base for each policy, risk−speed within each
arm, dense−base within each arm, plus risk−nativebase and risk−dense. Report
forecast200 and314 coordinate errors as fixed secondary endpoints, all policies,
all seed signs, undefined reasons, failures and boundary effects. Relative changes
are supplemental only with explicit positive denominators. No pooled-frame
significance, favorable horizon/trajectory selection or p-value search; n=3
supports descriptive consistency, not strong significance claims.

## Common-state allocation and measured costs

Evaluate all30 validation and all30 test trajectories separately at zero-based
targets7,85,163,241,319. Current observed history ispositions[t−6:t], previous
observed history ispositions[t−7:t−1]. Complete graph selection, prediction and
timing before reading the target. Same-state risk is previous-observed-nativebase
risk, distinct from autonomous cached-own-graph risk. Random material is
[20261006,93000,training_seed,split_index,source_index,target_frame], split index0
valid/1test, omitting arm and reconstructed for each repeated call.

Retain base/dense/all three sparse predictions, residual/risk arrays, signed
actual base-minus-selected-action error reductions, pair identities, exact budgets,
cap/self/prefix checks, and dense/sparse benefit-sign disagreement. Compare risk
with residuals and with signed action utility using per-frame Spearman and
explicit undefined coverage. Do not substitute dense benefit for sparse-action
benefit. Aggregate particles within frame, frames equally within trajectory,
then trajectories equally and seeds equally. Keep splits separate.

Retain the prepared six cyclic timing rotations, one untimed warmup and native/
supplied parity checks, including previous-base scoring on every risk call and a
natural-base construction reference. Without the optional physical baseline, this is53 forwards per common-state frame:
4parity+7warmup+42timed. Six models×two splits×150frames require95400 calls;
clean validation adds2304, total97704 before failures. If the optional physical
baseline is admitted, use the124704-call workload declared above instead. Include graph construction,
ledger/artifact I/O and all failed attempts in runtime accounting. A reduced
cost design requires a new prospective reviewed version before outcomes.

## Unresolved gates

Before freeze, complete the graph-support trainer/base-path/replay review, exact
full-concurrency and terminal-ledger capacity checks, distinct arm-aware final
cohort/evaluator/summarizer wrappers, full-horizon validation timing, and a
conservative complete-work forecast. Then root may issue a separate scientific
release. No current document or template certifies conference readiness; evidence
integration, source/results reproduction, native LaTeX compilation, exported
PDF/anonymity/rights checks and author verification remain required.
