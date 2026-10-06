# Goop 2D: paired graph-support science, version 2

Prospective scientific design for a separate root release. This file does not
launch or admit a run. Goop 2D has priority; Sand remains prepared, and Goop-3D
is a separate possible extension subject to its own complete remaining cost.
Do not alter frozen Sand/Goop trainers or promote any capacity checkpoint.

## Scope and preserved earlier plans

Train six faithful Goop models from initialization: base/mix by seeds0,1,2,
each exactly100000 updates. Evaluate all six policies on all30 reserved test
trajectories for all395 forecasts after six observed frames. The primary paired
seed interaction is
`(MSE_laggedrisk25-MSE_random25)_mix - (MSE_laggedrisk25-MSE_random25)_base`.
Report all absolute arm/policy means, both within-arm contrasts and every
policy's mix-minus-base change. Negative interaction alone does not establish
risk beats random, lower absolute risk error, or that random did not worsen.

The scientific policy, horizon, diagnostics, estimands and failure definitions
in `goop_graph_support_evaluation_protocol_v1.md`, SHA256
`8b1b84686e5efba17a8b3c9178b8a3157ebaad4fcb62020aca34fd569e4a57a1`, are
incorporated unchanged. Its workload JSON is SHA256
`1fe0fbb743bf56754d9f8fb3f4ab91d305fdcbb56c4ee037adaa76c32096342e`.
**This v2 replaces that document's evaluation timing/extrapolation prerequisite
with the prospective whole-invocation allocation below.** It also uses the
training-only q6/r6 portion of the separate Goop capacity protocol; it does not
claim its older complete-cost formula passed without its required evidence.
Preserve both earlier plans and undefined forecasts unchanged. No additional
512-update autonomous rollouts are required to make this quota allocation.
Sand's failed v1 full-horizon timing gate remains failed/undefined: its63/108
complete and45guarded infrastructure outcomes are not Goop evidence.

## Exact data and implementation

Use all1000 official GNS Goop training records, all30 validation records,
T401 float32 positions[T,N,2], full int64 type7 vectors, radius.015, bounds[.1,.9]^2
and dt=.0025. All395000 six-frame training histories are eligible. Preserve
source TFRecords, generation/length/CRC/SHA receipts, original order, conversion
hashes and all auxiliary arrays. The reviewed official parser excludes context
when metadata lacks context_mean; this is the reason for omitting preserved
step_context from model inputs, independently of its zero/NaN census. Recheck
context bytes and the four exact official source files before/after training.
T401/H395 is the complete stored-trajectory protocol; the released TensorFlow
convenience evaluation drops a final stored frame and uses H394.

| Input/source | SHA256 |
|---|---|
| Goop metadata | `565d6e13be91be6a6b0fbc31a9aa19ed70a5505228411d0928288fcf874ca3dd` |
| Training manifest | `5ef43daf9bac961a69bd460a539891624b79c8a13f80ca851e85802197982256` |
| Validation manifest | `3227e03c4c7fcee9f99c4104010774cabc74c34bfda667d76c5894590dcfc415` |
| Training/data admission | `c2a12ef0c55b47f4c9493027450f8b51f52dbbd6043915bb09c1648b6cb9edeb` |
| Structural report | `6d68af35c5685b4d248c523b8dd05884ee569e100bce06b12b8eee4d1c0a1cdd` |
| Acquisition report | `9e4b106324b1e7c5add4404565eca888eaa0ebd3c233622308c113e88c5e6a8d` |
| Context semantics | `c81ae2f1565e61542bcc406c4a9d71b67135620857ad62292eaef0e29b407de9` |
| Auxiliary census | `3278055e119c482c5620ff66b2c4e4e6c09361240155cc77a7ad98dbaa13d8d1` |
| Goop trainer | `dd9ef01a16116f04a578bbfe9ed01774e66fdc08d60b913072a33ad211b319c1` |
| Goop capacity wrapper | `7b75518697fe9f496dc1ea8e6a64406fffdb3b7022471b36548038c1202c2d03` |
| Goop evaluation contract | `4ee1e33dbe666804d1827b88565620a435d04836a170671914a71acb818be090` |
| Goop benchmark/helper | `2e0ef0b84635c8430102cd797648d24cec14b457e1df900eb784d5e167966f8f` |
| Goop final evaluator | `cd970e04012930d896b31944271ccaf7da2b0881f22fe2f903533a8d5911dc6d` |
| Shared physical policy | `4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a` |

All transitive numerical source pins remain required by these entries. Root
binds the final independently reviewed Goop scientific supervisor adapter and
whole-invocation quota wrapper byte hashes in the same scientific release.
The training entry is `supervise_graph_support_science_quota_v2.py`, using
`adaptgns_goop_graph_support_scientific_supervisor_quota_v2` and root release
`adaptgns_goop_graph_support_scientific_release_quota_v2`. Evaluation uses
`supervise_graph_support_evaluation_quota_v2.py`. The adapter must call
Goop admission/context/manifest and payload verification,
use `adaptgns_goop_graph_support_cuda_training_v1`,
`cuda_goop_graph_support_schema` and `cuda_goop_graph_support_run`, accept target
frames6..400, and retain Goop model normalization. Substituting a filename while
still invoking Sand load paths, source contracts or payload checks is invalid.
No training launch occurs until that small adapter is reviewed and byte-pinned.

## Fixed training recipe and pairing

Width128, ten message-passing blocks, two-layer MLPs, nine-type/16-wide embedding,
six observed frames, batch2, uniform history sampling with replacement, noise
6.7e-4 and noise-adjusted normalization. Use the unchanged faithful objective,
detached variance-feature/residual-target semantics and variance floor1e-6.
Adam:1e-4 initial LR, exponential decay to1e-5 at100k, betas(.9,.999),eps1e-8,
no weight decay/clipping, foreach=False/fused=False. No AMP/TF32/compile/DDP.
Use the pinned Torch2.13.0+cu129/CUDA12.9 GB200 stack, two CPU threads and strict
deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, exact root-bound
LD_LIBRARY_PATH, and no CUDA_VISIBLE_DEVICES remapping.

Native base uses strict-r geometry, receiver cap128/self candidates, original
receiver/distance/source order and possible cap asymmetry. The optional annulus
is the strict1.267r pair set minus uncapped strict-r pairs. Cap-omitted short
pairs are not optional. Base adds none. For completed update u and example j,
the independent coin from SeedSequence([20261005,seed,u,j,4409]) selects mix
exposure with probability.5; SeedSequence([20261005,seed,u,j,5501]) selects
exactly floor(.25*annulus_count) uniformly without replacement. Append both
orientations after the exact native directed prefix, without recapping. No
cross-example edges or future-target access. Zero budget equals native base.

Pair initial model/configuration, sampled frames, noise bytes, LR and native
inputs by seed; arm never enters those streams. Record all100000 graph/schedule
rows. Save step0 before update1 and then10000,...,100000 (11checkpoints);
log at step1/every100 (1001scalar rows). Pairing audit uses saved initialization
after children complete; do not claim a pre-step1 audit barrier. Audit all saved
schedule/graph budgets and endpoint model/Adam/RNG/status/pointer identities
before cohort admission. Same-CUDA implementation/replay evidence is required;
earlier CPU/CUDA equivalence failures remain failed, not repaired by this study.

Fixed concurrent mapping: hostA CUDA0/1=base/mix seed0 and CUDA2/3=base/mix seed1;
hostB CUDA0/1=base/mix seed2, with B2/3idle. Hosts remain exclusive to the owned
study. No concurrent Sand/Goop3D work on unused GPUs while the inherited monitor
rejects unrelated whole-host processes. Fresh PID/start/argv/parent, all physical
GPU UUIDs, source/data bytes and bounded clock-error checks remain mandatory.
No automated retry/resume, numerical-failure retry, seed replacement, shortened
endpoint, checkpoint promotion or training from an infrastructure parent.

## Measured training plus enforced evaluation allocation

Use actual complete six-job Goop512 training-capacity reports, the intended
A4/B2 overlap and reviewed initialization pairing. Let q6 be the slowest steady
per-update wall rate after64 warmup updates and r6 the largest nonnegative
external process time minus all512 guarded-update times. Preserve all original
rows and use every model. Do not borrow Sand q6 or a horizon/size scaling rule.

```
training_seconds = 1.35*(100000*q6 + 12*r6) + terminal_ledger_reserve_seconds
evaluation_invocation_seconds = 7200 + 1800 + 1800 + 900 = 11700
evaluation_cleanup_seconds = 4*15 = 60
remaining_seconds = 11700 + 60 + 2700 + 3600
total_compute_analysis_seconds = training_seconds + remaining_seconds
```

Root explicitly justifies all11 terminal publications,100k history size,
live finite/Adam/RNG work, serialization/fsync/hashing and contention. Existing
same-host synthetic IO evidence may inform an allowance only after checking
Goop payload/ledger compatibility; label this as transferred engineering
allowance, not measured Goop terminal runtime.2700seconds separately reserves
final checkpoint/cohort/load work and3600seconds result verification/analysis.
Margins are planning allowances, not confidence bounds or proven worst cases.
The 2700-second cohort/preflight reserve is charged once. At evaluation queue
release, `outer_processing_reserve_seconds` covers only remaining outside-queue
work: ordinarily the 3600-second analysis reserve after cohort/preflight is
complete, plus any still-unspent outside work. Do not reserve a second2700seconds
for cohort work already completed; do not drop work that remains. The allocation
file must bind the reviewed `evaluation_quota_supervisor_sha256`.

For each concurrent model stream, enforce whole-process stage quotas in order:
full-rollout test7200s, same-state valid1800s, same-state test1800s, clean valid900s.
Charge imports, source/checkpoint checks, all inference/graph/scoring/transfer,
compression/fsync and publication from spawn to exit. Outer owned-process
monitoring, exact identity and bounded reap/cleanup must enforce these budgets
independently of the evaluator's inner alarms. Allow15s cleanup per stage. Do
not start the next stage when cleanup is unresolved or remaining fit is lost.
This limits operational expenditure; it does **not** promise complete outcomes.

Require compute and analysis by2026-10-07 01:00UTC. Root release binds exact
measured training evidence, stated reserve basis, both reviewed wrappers,
source/data/numerical evidence and this protocol in one prospective admission.
Recheck `now+clock_error <= latest_start`, where latest_start is no later than
01:00 minus total cost; training stop must leave all remaining_seconds plus its
own clock/cleanup reserve. Preserve the October7 08:00UTC no-new-work cutoff
and writing reserve. No further full-horizon512timing gate or favorable-outcome
screen is added under v2. A failed/incomplete capacity or implementation check
still cannot pass. If the complete planned allocation does not fit, defer this
science as a whole; do not trim seeds/policies/trajectories after outcomes.

## Final admission, workload and accounting

After all six exact100k endpoints and three initialization/schedule pairings
pass, root may acquire and admit the complete reserved official Goop test
source. Verify received generation/size/CRC/SHA, complete offsets/payload CRCs,
conversion/source bytes, T401/type7/context contract, all30records and duplicate
accounting against train/valid. Do not invent a test checksum in advance. The
final evaluator checks the complete cohort/audit before reading a test manifest
and verifies its selected checkpoint; no source discrepancy permits a subset.

Required full workload is1080 trajectory-policy outcomes: base,dense,random25,
speed25,laggedrisk25,relative-velocity-RMS25; allH395. Same-state targets are
7,105,203,301,400 on all30 validation and30 test sources separately. Clean
validation uses128 fixed flattened indices floor(i*(11850-1)/127). Preserve all
policy/scoring/parity/repeated-order diagnostics specified in the incorporated
v1 scientific design:124704diagnostic forwards, physical boundary diagnostics,
signed actual action utility, residual-vs-benefit ranking and measured cost.
Autonomous cached-own-graph risk and previous-observed-base risk remain distinct.

Every planned item is either committed complete, committed guard-failed,
interrupted/uncommitted current work, or never started. A timeout is an execution
budget event; never relabel it as model instability, pair-cap or coordinate
failure. Preserve missing prefix evidence, temporary files, process/signal
records and unsuccessful outcomes. Recorded numerical/resource guards retain
their categories/phase; parity/implementation errors stop inference for review.
All-required MSE/paired contrasts are null when required values fail or are
missing. Fixed source/policy order makes quota survivors selected; do not compare
survivor means as full-cohort effects or count never-started items as model
failures. Report observed guard failures and missing executions separately.

Average coordinates/particles within step, all395steps within source, sources
equally within seed, then three seed pairs equally. Show all seed effects,mean
and sampleSD; frames/particles are not independent training replications.
Secondary endpoints200/395 and all fixed contrasts in v1 remain unchanged.
This is an exploratory native-CUDA extension informed by earlier work, not
pristine independent confirmation or a material-only causal comparison. Keep
Goop2D,Goop3D,Sand,WaterDrop,historical arrays and infrastructure distinct.
Reproducible results, source/hash evidence, manuscript/checklist updates and
author verification remain required; no readiness or submission claim follows.
