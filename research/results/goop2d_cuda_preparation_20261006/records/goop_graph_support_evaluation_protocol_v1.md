# Goop graph-support evaluation: prospective version 1

**Prepared workload and estimands only; no timing, training, test-access or
scientific release.** Complete adapter/source review and exact source hashes are
pending. Root must freeze this design and implementation before measurement or
scientific outcomes. Do not modify frozen Sand files or promote Sand/Goop
capacity checkpoints. The machine-readable workload is
`goop_graph_support_evaluation_workload_v1.json`; this document is not an
executable admission or cost estimate.

## Scientific scope and admitted lineage

Evaluate six fresh faithful models: base/mix arms crossed with seeds 0,1,2,
each at exactly 100000 updates. The primary seed-level estimand is
`(MSE_risk-MSE_random)_mix - (MSE_risk-MSE_random)_base` under autonomous full
rollouts. Report all arm-policy means, within-arm risk-minus-random differences
and each policy's mix-minus-base change. A negative interaction does not imply
risk beats random, absolute risk error improves, or random did not worsen.
Random sparse training exposure does not make the residual head an action-value
estimator or guarantee support for dense/score-selected graphs.

The candidate trainer is `train_goop_graph_support_cuda.py`, SHA256
`dd9ef01a16116f04a578bbfe9ed01774e66fdc08d60b913072a33ad211b319c1`.
The checkpoint schema is `adaptgns_goop_graph_support_cuda_training_v1`, field
`cuda_goop_graph_support_schema`, with training key
`cuda_goop_graph_support_run`. Source maps bind the unchanged numerical core.
The candidate adapters are `benchmark_goop_graph_support_rollout.py`,
`evaluate_goop_graph_support_final.py` and `goop_evaluation_contract.py`.
They share unchanged `sand_graph_support_policy.py` at SHA256
`4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a`.
The three new adapter hashes must be fixed in the root cohort and execution
release after independent review. Shared numerical policy/diagnostic bodies
must retain reviewed arithmetic while the separate Goop wrappers bind its
source lineage, T401/type7, context contract and schedules.

Use the complete official GNS Goop TFRecord source with original ordering,
generation/length/CRC/acquisition receipts and received hashes. Exact metadata
SHA256 is `565d6e13be91be6a6b0fbc31a9aa19ed70a5505228411d0928288fcf874ca3dd`.
Require T=401, float32 [T,N,2] positions and preserved int64 [N] type-7 vectors;
30 validation and 30 reserved-test trajectories. Root binds exact conversion,
manifest, reader, structural/admission and duplicate/accounting evidence.
Unknown shapes, types, context semantics or precision stop admission.

Preserve every stored `step_context` array, including nonfinite values, without
dropping, repairing or using it as model input. Its omission requires the
separate reviewed official parser/metadata contract: the released parser uses
context only when metadata declares context normalization, which the pinned
Goop metadata does not. The auxiliary-byte census is integrity evidence, not
the semantic reason for exclusion; do not call NaNs placeholders without source
support. Reverify all preserved arrays and context bytes before/after evaluation.

This study uses all 401 stored frames: six observed and **395 forecasts**. The
official TensorFlow convenience evaluation drops the final stored frame and
uses 394 forecasts. H395 is the declared complete stored-trajectory protocol,
not an exact reproduction of that TF evaluation path.

All comparisons use the separately admitted deterministic native-CUDA variant,
Goop normalization and type-7 inputs. Earlier CPU/CUDA equivalence failures stay
failed; Sand implementation/timing results do not establish Goop numerical
admission, cost or quality. No diagnostic gradient masks or changed backward
rules enter training/evaluation. This is an exploratory extension informed by
earlier particle-simulation work, not pristine independent confirmation or a
material-only causal comparison with Sand/WaterDrop.

## Six fixed policies and complete autonomous evaluation

For each final model use all 30 test trajectories, all H395 forecasts and the
ordered API policies
`base, dense, random25, speed25, laggedrisk25, relative-velocity-RMS25`.
This is 180 trajectory-policy outcomes/model and **1080** across the cohort.
The first six observed frames initialize every rollout; later histories are
the policy's own predictions. No policy, seed, trajectory or horizon selection.

Preserve native strict radius .015, receiver cap128, self candidates, directed
base ordering/asymmetry and unchanged mandatory prefix. Optional pairs are the
uncapped strict radius1.267r annulus minus the uncapped strict-r pair set; native
cap-omitted short pairs are not optional. Dense appends all optional pairs;
each sparse policy appends exactly floor(.25*annulus_count) unordered pairs in
both orientations, with no recapping or target access. Scored selection uses
max endpoint score and the existing lexicographic ties.

Speed uses current-history last-step speed. Random seed is
`93000+1000*training_seed+source_index`, omitting arm. Autonomous `laggedrisk25`
pays an initial observed native-base score pass, uses its exact budget from
forecast1 and then caches risk from its own preceding selected graph/history.
Every policy executes the same model head. Retain full end-to-end policy cost,
parity/warmup, graph/selection/transfer/forward components and total invocation
wall time; report artifact I/O separately. Fixed autonomous order is operational
timing, not balanced causal speed comparison.

Relative-velocity RMS uses float64 last-position subtraction divided by Goop
metadata dt=.0025. For each receiver use retained incoming native-base nonself
senders, accumulate squared 2D velocity differences in native order using
`numpy.add.at`, divide by neighbor count, and take square root; empty support
scores zero. Score optional pairs by maximum endpoint RMS with the same budget
and ties. This is local motion heterogeneity, not uncertainty or action value.
Charge graph/score/reduction/ranking work and save scores/support hashes.

Guards remain nonfinite state/output/risk, nonpositive required variance,
absolute coordinate>10, or more than100000 candidate unordered pairs. Retain
first failure phase/forecast, rejected values where returned and accepted
prefixes. A timeout may leave no returned internal prefix; record that absence.
Parity/implementation failures stop inference for review; no automatic retry.
Report predicted/truth boundary fractions and excursions outside metadata
[.1,.9]^2 independently of computational guards. Do not hide finite but poor
physical behavior or imply a conservation guarantee.

## Observed-state and clean diagnostics

Same-state evaluation uses all30 validation and all30 test trajectories
separately at zero-based targets **7,105,203,301,400**. For target t, current
history is `positions[t-6:t]` and previous history `positions[t-7:t-1]`.
Complete every selection/inference/timing call before reading the target.
Same-state risk is `previous-observed-base-risk25`, computed on the previous
observed native-base history; it is not autonomous cached-own-graph risk.

Diagnostic policy order is
`base,dense,random25,speed25,relative-velocity-RMS25,previous-observed-base-risk25`,
plus `natural_base_reference` for construction/timing only. Random material is
`[20261006,93000,training_seed,split_index,source_index,target_frame]`, with
valid=0/test=1, arm omitted and RNG reset per repeated call. Use one warmup and
seven cyclic timed rotations, each slot occupied once per policy/reference.
Charge the extra previous-base scoring pass on every risk call. Four parity,
eight warmup and56 timed forwards give **68 forwards/frame**.

Clean validation selects128 distinct indices
`floor(i*(11850-1)/127)`, i=0..127, from source-order30*395 histories. Map index
to source=index//395 and target=index%395+6. Use zero noise, saved Goop training
normalization and native-base graph, with three forwards/history including
parity. Retain normalized acceleration coordinate MSE, realized vector squared
error, predicted vector squared error2*variance, constant-free2D Gaussian NLL
`0.5*vector_squared_error/variance+log(variance)`, raw head/floored variance and
prediction/target arrays. No endpoint selection follows these diagnostics.

Per model the two common-state splits total300 frames; the cohort has1800
frames*68=122400 forwards. Clean validation adds6*128*3=2304, yielding
**124704 diagnostic forwards**. These counts exclude autonomous rollout calls,
which are separately charged as1080 completeH395 outcomes.

Retain exact graphs/budgets/native prefix, repeated outputs, physical scores,
residual/risk arrays, signed actual base-minus-selected-action error reductions,
dense/sparse benefit-sign disagreement, and per-frame Spearman with undefined
reasons. Dense benefit does not substitute for sparse-action benefit. Native/
supplied and repeated-output checks use the reviewed fixed tolerances; no
outcome-driven relaxation. Aggregate particles within frame, equal frames
within trajectory, equal trajectories, then equal seeds; keep splits separate.

## Paired analysis and failure accounting

Full-horizon coordinate MSE averages particles/coordinates within forecast,
all395 forecasts within trajectory and all30 trajectories equally per seed.
Form matched seed/trajectory risk-minus-random then the mix-minus-base
interaction; show all three seed effects, mean and sampleSD. Three paired
training seeds are the replication unit, not pooled particles/frames.

Failed or missing required outcomes make the corresponding complete-sample
MSE contrast null. Retain failure indicators, first-failure times, accepted-prefix
and completed-case summaries with explicit coverage; no finite-MSE imputation
or survivor-only contrast. Missing executions stay missing. Secondary contrasts
are policy-specific mix-minus-base; within-arm risk-minus-speed/base/dense/RMS,
dense-minus-base and RMS-minus-random; and the RMS-versus-random interaction.
Fixed secondary forecasts are200 and395, with trace forecasts1,10,50,200,395.
Report all seed signs, boundary effects and null reasons; no p-value search or
favorable horizon/trajectory selection. Relative changes require explicit
positive denominators. Initial pairing is audited before cohort admission;
do not claim a pre-step1 barrier if the supervisor audits saved states later.

## Timing, scheduling and final release gates

Timing uses separately admitted complete512 infrastructure checkpoints from all
six arm/seed cells and validation source-small/lower-median/large cases selected
by (particle count,source index), without accuracy-based selection. Each model
requires18 completeH395 rollout cases and15 observed frames in each diagnostic
timing mode (the same three sources and fixed five targets). A failed/guarded
prefix cannot support a complete-horizon cost. Preserve all timings/failures.

Before any timing outcomes, root must freeze the intended actual host/GPU
mapping, concurrency, maximum timing budgets, conservative extrapolation and
complete-cost formula. Do not copy Sand's q6, horizon, source costs, final model
loading reserve or six-stream concurrency claim as if measured for Goop. Charge
all training/terminal ledger,1080 fullH395 outcomes,124704 diagnostic forwards,
setup/publication, accumulated JSON rewrite growth, complete cohort verification
and writing reserve. If linear case/frame scaling is used, separately reserve
quadratic accumulated JSON output and larger final100k checkpoint loads.

The frozen Sand scientific monitor requires its whole host GPU inventory to
contain only its owned children; it rejects unrelated work even on host B's
otherwise unused indices2/3. Therefore no concurrent Goop GPU work may run there
while Sand supervision is active. Goop feasibility begins only after applicable
Sand workers have ended, processes/locks/GPU identity are rechecked and root
confirms the whole Goop workload can still fit. No new host purchase, source
mutation or implicit spare-GPU execution is authorized by this file.

Require compute/analysis completion by2026-10-07 01:00 UTC and preserve the
08:00 UTC no-new-work cutoff. A separate root release must bind all reviewed
Goop sources, numerical/data/context admission, prospective training/evaluation
protocol, exact complete-cost evidence, current process/clock checks and final
six-model plan. If the complete plan does not fit, retain Goop as preparation/
infrastructure and defer science as a whole; no reduced outcome-selected study.

Final test acquisition/admission follows all-six endpoint/optimizer/pairing
verification and frozen scientific/evaluation sources. The final evaluator must
check the complete cohort/audit before opening the test manifest and verify its
selected checkpoint bytes/payload. Keep Goop distinct from Sand, WaterDrop,
historical arrays, smoke tests and timing infrastructure. Root owns every real
execution and release; this protocol claims neither completion nor readiness.
