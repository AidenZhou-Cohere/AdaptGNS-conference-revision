# Exploratory graph-convention bridge — October 5, 2026

This follow-up is designed **after inspection** of all locked full WaterDrop outcomes. It does not amend, replace, or repair the fixed 100k experiment. The original eight NLL failures, source hashes, checkpoints, and evaluation outputs remain preserved. No outcome in this bridge is independent confirmation. Root reviews and commits the new source, tests, and this protocol before model inference, and launches the six jobs sequentially on local compute only. Work stops starting new jobs at October 7, 08:00 UTC.

## Question and prior knowledge

The fixed trainers used strict-radius, receiver-capped (128), directed graphs with self-messages and no graph expansion augmentation. The locked reference evaluator used uncapped symmetric pairs with no self-messages, with its own float32 squared-distance `<=` classification. We test whether those graph conventions affect base predictions and relative allocation performance. We do not infer that this resolves the distinct absence of expanded-edge training support.

Before designing this bridge, a saved-geometry inspection found maximum nonself base/dense degrees **22/31** in the 297 existing observed test states, and mean self-edge fraction **12.6035%** of the native base directed graph. The cap was therefore already known inactive on those states. This is explicitly prior inspected evidence, not a new discovery of this experiment. Validation cap behavior and predictor effects remain to be measured. Cap inactivity on observed states does not imply inactivity in autonomous rollouts.

## Frozen population and inputs

- The six original final 100,000-update checkpoints: faithful and corrected NLL, seeds 0, 1, 2. Neither checkpoint nor objective is selected by this bridge.
- Each model's exact 128 clean validation histories listed in its immutable training configuration. All 30 official validation trajectories are represented; no resampling.
- The existing 297 observed test histories: source indices 3–29, targets 7, 106, 205, 304, 403, 502, 601, 700, 799, 898, 1000. Existing array/row checksums must match, and histories/types/targets must match the checksum-verified official converted source.
- Total: 425 histories/model, 2,550 histories across six models. Validation and test remain separate. The test history schedule is the same as the completed diagnostic, not a new independent sample.
- Stored training normalization, weights, and variance semantics are unchanged. No noise or training occurs. Official test source SHA256 remains `b7f147c22e96fd3fbb8d595702cb8c85e412bfb449d7b4d761cc03eb76246c31`.

## Native identity gate

For every history, before interpreting policy predictions, run native `predict_positions_with_variance` and a supplied-edge base pass. The supplied graph reconstructs the native float32 Euclidean norm with strict `< r`, SciPy query candidates, source-distance then source-ID sorting within receiver, cap128 including the self-edge candidate, and receiver-major edge order. Exact equality is required for directed edge arrays, node features, and edge features. Native and supplied decoded predictions must agree within absolute tolerance 2e-7; raw risks must pass finite checks, and converted positive risks must agree within tolerance 1e-6 absolute and 1e-5 relative. These tolerances are fixed before outcomes; maximum differences are saved.

A native identity/parity failure saves raw arrays and stops that model job for review. Raw and converted risks both must meet the declared risk agreement tolerances; conversion must not conceal raw-head disagreement. It is not silently retried, loosened, or discarded. Exact byte identity is required for inputs/features, while small tolerances on network outputs allow known local numerical replay variation. Passing this gate does not certify a different device or arbitrary graph equivalence.

## Graph cases and selectors

All new pair universes use the native strict-radius float32 norm with separate radius-r and radius-R queries, R=1.267r. Expanded candidate construction has the same 100,000 unordered-pair resource guard. The locked evaluator's original base/annulus sets are independently reconstructed and their symmetric differences saved; no equality is assumed or forced.

Base and dense graphs are each evaluated under capped128/uncapped × no-self-messages/self-messages. This yields eight factorial cases. For a capped graph the cap applies per receiver and may produce directed asymmetry; capped dense is a convention control and does not claim an exact optional-pair budget.

Under coincident-position ties a capped receiver can even lose its self candidate after distance/ID sorting; this follows the native rule. The exactly-one-self-edge invariant applies to uncapped loop-on policies, not to arbitrary capped cases.

For uncapped loop-off and loop-on separately, compare base, dense, random25, speed25, previous-observed-base-risk25, and current-base-risk25. Base/dense overlap the factorial cases: **16 distinct cases/history** in total. Every budgeted policy preserves every strict base pair, retains floor(0.25 × annulus pair count) optional unordered pairs, and emits both orientations. Loop-on adds exactly one directed self edge per particle after symmetric nonself expansion. No policy applies a cap after selection. No self-edge is passed through pair symmetrization, which would double it.

Risk selection scores come from the same loop convention and the uncapped base graph. Previous-risk obtains its score from the preceding observed six-frame history. Current-risk uses the current observed six-frame base pass. These are teacher-forced diagnostics, not autonomous cached-own-graph policies. Risk pass costs are explicitly included in standalone network-pass counts (two rather than one), although already computed scores/predictions may be reused in this exploratory screen. Speed is the norm of the final observed position difference.

Pair score is max(endpoint scores). Descending stable sorting on lexicographically ordered annulus pairs fixes tie handling. Random seed material is `[20261005,771,training_seed,split_code,source_index,target_frame]`, where valid/test codes are 0/1. The same random set is used under both loop conventions and both objectives at a given training seed/history. Policy construction never reads the target. Exact identical ordered graph IDs within the same history/model may reuse one prediction; the original case is recorded. This is a mathematical equality reuse, not replacement of different outputs after comparison.

Base, dense, random, and speed have the same selected nonself pairs across loop conventions; those contrasts isolate the self-message channel. Risk policies regenerate scores under the tested convention, so their loop contrast includes both the changed channel and changed selection. This study does not identify the self-message effect at a fixed risk-selected pair set. The loop interaction for risk is the total policy-convention interaction, not a mediation analysis.

## Outcomes, failure handling, and saved records

Save paired per-frame decoded predictions, raw and converted risks, selected pairs/directed edges, graph byte hashes/counts/degrees, scores, residual vectors in position and decoder-equivalent normalized acceleration coordinates, normalization, history/target hashes, native/supplied features, parity maxima, cap-binding counts, and explicit failed-case details. Native parity failures also save their unsuccessful raw outputs. Position absolute coordinate >10, nonfinite output/risk, or nonpositive converted risk is a failed outcome. Targets first enter calculations after all policy predictions. Negative gains remain signed.

Output files are new and separate from all locked studies. Immutable run manifests pin protocol, source, checkpoint, original training-config hash, existing same-state result/row/array hashes, official converted manifests and all consumed positions/types bytes, runtime, library versions, and source device. Input hashes are checked again at completion. Atomic rows, arrays, status, result index, and an exclusive reviewed `RunLock` support interruption recovery. Recovery requires exact provenance and each prior row/array checksum. Failed outcomes are reused as failed; native parity failure cannot proceed on resume. Orphan arrays require manual review/preservation rather than overwrite.

## Prespecified analysis

Analyze validation and test separately. Compute particle-mean coordinate MSE within each history, equal-history means within each trajectory, then equal-trajectory means within each training seed; report the three seed values, their mean, and sample SD, with no significance or generalization claim. Validation has 4–5 sampled histories/trajectory, so an unweighted frame mean is a separately labeled sensitivity summary, not the primary aggregate. Require all 16 cases ×425 frames/model. A missing/failed required observation makes its corresponding full-population case metric undefined; preserve the failure count and raw outcomes.

Primary paired contrasts are:

1. Loop-on minus loop-off base and dense MSE, within each cap convention.
2. Capped minus uncapped base and dense MSE, within each loop convention.
3. Previous-risk minus random25 and current-risk minus random25, separately by loop convention.
4. Loop interaction: (risk minus random under loops-on) minus (risk minus random under loops-off), for each risk policy.
5. Dense minus base under each loop convention and its loop interaction.
6. Current-risk minus previous-risk, including exact optional-pair overlap.

The two primary interactions are previous-risk-minus-random loop1 minus loop0, and dense-minus-base loop1 minus loop0. The current-risk counterpart is secondary. Retain all 16 case metrics, loop1-minus-loop0 for each uncapped policy, capped-minus-uncapped base/dense for each loop convention, every sparse-minus-base/random/speed contrast within loop convention, and current-minus-previous risk. Compute paired effects within trajectory and then seed rather than subtracting independently weighted aggregates.

Record native parity counts/maxima, feature/edge equality, current base/dense and previous-base cap activation, original/new radius-classifier graph differences, exact budgets/self-edge counts, and graph-ID aliases. Save current/previous risk selection overlap and within-frame Spearman score-rank agreement in each loop arm. Also summarize correlations of each risk score with its own sparse action's signed base-minus-action normalized vector benefit and with base residual. These actual-action labels supplement rather than replace the already published dense-benefit diagnostic. Retain undefined constant-vector correlations and make required undefined correlation aggregates null, with explicit counts.

All remaining contrasts are descriptive. No seed, trajectory, graph convention, policy, or checkpoint is selected for the headline based on its favorable outcome. Capped graph effects and self-message effects are interpreted separately from graph-expansion support. An unchanged ranking meaningfully limits the convention confound; a changed ranking narrows interpretation of the original locked findings.

## Cost and scope limits

Single-call operational durations and standalone network-pass counts are saved. Candidate graphs and score passes are shared, case order is fixed, duplicate graphs may reuse outputs, and no six-repeat timing benchmark is run. These durations are not comparable policy latency, cached-rollout amortized cost, throughput, or a speedup claim. This screen is intended to take minutes of local inference, with I/O/audit overhead recorded. It establishes one-step observed-history effects only; a later autonomous bridge requires a separate protocol.

## Command contract

From the research repository, with the recorded experiment Python and root-controlled sequential execution:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m research.graph_convention_bridge \
  --checkpoint /absolute/path/to/original/final/model.pt \
  --validation-manifest /absolute/path/to/work/full-data/converted/valid.json \
  --test-manifest /absolute/path/to/work/full-data/converted/test.json \
  --saved-same-state-dir /absolute/path/to/work/full-evaluation/same_state/faithful_seed0 \
  --output-dir /absolute/path/to/work/graph-convention-bridge/faithful_seed0 \
  --device mps --threads 2
```

The paths above are templates; the parent resolves each verified final checkpoint and corresponding same-state directory. `--resume` requires the original output manifest; stale lock clearing is allowed only after native process identity and dead-lock verification. No subprocesses, training, network transfer, or new test selection are performed by this module.
