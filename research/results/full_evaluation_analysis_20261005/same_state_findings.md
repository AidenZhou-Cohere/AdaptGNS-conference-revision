# Final same-state and runtime audit

Independent audit passed **1,068,869 checks**. It reopened all 1,782 numeric frame archives and raw JSON records; verified their checksums and stability; reconstructed graphs and allocations; recomputed residuals, four Spearman coefficients/frame, all repeated-call statistics, weighted alternatives, and every saved within-seed policy/objective comparison. The existing strict loader independently checked recorded protocol/checkpoint/data provenance. No model loading, inference, or result replacement occurred.

The scope is 27 official test trajectories (indices 3–29), each with 11 fixed observed histories, for each of six final 100k checkpoints. Current histories, preceding histories, targets and particle types match the pinned converted source and match across all models. Each policy receives the same current history; previous-observed-base risk uses an additional preceding-history base pass. It is distinct from autonomous own-graph cached risk. Prior inspection of other test records and historical aggregates remains disclosed.

## Error and signed dense benefit

All 1,782 frame outcomes completed, with **zero failed frames and zero undefined values among 7,128 required Spearman coefficients**. Values below are equal-particle means within frame, equal frames within trajectory, equal trajectories within seed, then mean ± sample SD across exactly three training seeds.

| Objective | Policy | Position coordinate MSE | Decoder-equivalent normalized coordinate MSE |
|---|---|---:|---:|
| faithful | base | 4.35611e-09 ± 6.28672e-11 | 0.00959136 ± 0.000137844 |
| faithful | dense | 5.41183e-09 ± 3.86207e-10 | 0.0119159 ± 0.000849317 |
| faithful | random25 | 4.08243e-09 ± 1.2029e-10 | 0.00898859 ± 0.0002647 |
| faithful | speed25 | 4.28355e-09 ± 6.54707e-11 | 0.00943158 ± 0.000143994 |
| faithful | previous-observed-base-risk25 | 4.3031e-09 ± 1.16525e-10 | 0.00947464 ± 0.000256444 |
| nll | base | 4.81928e-09 ± 2.29717e-10 | 0.01061 ± 0.000505694 |
| nll | dense | 5.61637e-09 ± 8.77111e-11 | 0.0123657 ± 0.000194509 |
| nll | random25 | 4.46882e-09 ± 6.23539e-11 | 0.0098385 ± 0.000137688 |
| nll | speed25 | 4.804e-09 ± 9.0257e-11 | 0.0105771 ± 0.000199238 |
| nll | previous-observed-base-risk25 | 4.85908e-09 ± 3.25941e-11 | 0.0106985 ± 7.22836e-05 |

Dense increases seed-average one-step error in every seed of both objectives. Random25 gives the lowest seed-average one-step error among the five policies in all six models. Previous-observed-base risk has higher seed-average error than random25 in all six models, but its difference from base changes sign across seeds. These statements concern observed-history one-step errors, not autonomous rollout rankings.

| Objective | Previous risk minus random, position MSE | Previous risk minus base, position MSE | Dense minus base, position MSE |
|---|---:|---:|---:|
| faithful | 2.20672e-10 ± 5.04396e-11 | -5.30097e-11 ± 1.72646e-10 | 1.05572e-09 ± 4.3045e-10 |
| nll | 3.90253e-10 ± 4.62589e-11 | 3.98006e-11 ± 2.14496e-10 | 7.97091e-10 ± 1.46188e-10 |

| Objective | Previous risk / base error | Previous risk / signed dense benefit | Current-base risk / base error | Current-base risk / signed dense benefit |
|---|---:|---:|---:|---:|
| faithful | 0.357060 ± 0.059820 | -0.021875 ± 0.044836 | 0.357189 ± 0.059210 | -0.022203 ± 0.044698 |
| nll | 0.411323 ± 0.022636 | -0.048192 ± 0.011604 | 0.412300 ± 0.024146 | -0.047438 ± 0.011313 |

These are averages of within-frame rank correlations, not pooled-particle correlations. The positive risk–error association does not supply positive average rank association with this signed dense intervention. Faithful risk–benefit seed means change sign. Neither result estimates marginal single-edge value.

| Objective | Signed normalized-vector benefit | Positive fraction | Negative fraction | Zero fraction |
|---|---:|---:|---:|---:|
| faithful | -0.00464907 ± 0.00189289 | 0.390959 ± 0.0165843 | 0.603203 ± 0.0165732 | 0.00583823 ± 1.11996e-05 |
| nll | -0.00351137 ± 0.000641553 | 0.380102 ± 0.0123933 | 0.614058 ± 0.0124074 | 0.00583987 ± 1.44191e-05 |

Benefit is base-minus-dense vector squared error. Negative values are preserved. The sign fractions above use normalized residuals; unequal coordinate scales can change the sign relative to position space. All saved signed array entries were independently recomputed.

## Exact placement and numerical repeat evidence

Across the 297 unique observed states, the mean mandatory count is 2,113.879 undirected pairs (range 599–5,112), the mean annulus count is 1,233.512 (258–3,129), and the mean expanded candidate count is 3,347.391 (857–8,241). All three budgeted selectors retain floor(0.25 × annulus pairs), mean 308.010 optional pairs (64–782), plus every mandatory pair. Their mean total is 2,421.889 pairs; the mean within-state retained fraction of dense pairs is 0.726295. Directed edge counts are twice the retained pair counts. The candidate search still includes the expanded set.

The audit independently reconstructed the natural float64 base graph, expanded-then-filtered float64 graph, and saved float32 squared-distance classification. **All 297 observed states have identical base/annulus sets under the two classifications**; all six models share these geometries. This empirical equality does not assert equivalence on unobserved or autonomous states.

All exact allocations, random stream resets, maximum-endpoint score rankings, lexicographic cutoff ties and optional Jaccard overlaps were checked from arrays. Speed has a cutoff boundary tie at 201/297 histories in every model; previous-risk boundary ties occur at 189–202 histories depending on model. ID-based tie-breaking therefore matters.

One case had a different selected pair set across timing repeats: **NLL seed 2, source index 6, target 106, previous-observed-base-risk25**. Rounds 0–4 shared the first pair hash; round 5 had a second hash. The maximum recorded prediction difference from the first timed call is 2.4497509e−5. The first-call cutoff contained three tied pairs, of which one was selected. All other recorded repeated prediction differences are at most 5.9604645e−8; natural/shared-base prediction differences are also at most that value. Per-repeat score/prediction arrays were not saved, so the audit verifies the committed hashes/counts and reported difference, but cannot reconstruct each repeat or establish a unique cause. The first timed call remains the declared residual/allocation reference. No repeat or failure was replaced.

## Measured runtime

There are 64,152 timed predictor calls and 10,692 warmup predictor calls. Each frame executes six cases once for warmup and six rotated rounds; the previous-risk case uses two network passes and two candidate builds per call, and every other case uses one. That is 49 network passes/frame, 87,318 total. All recorded ordering, phase inclusion, repeated means/medians/sample SDs, and source/target pairing were checked. Each case occupies every timing-order slot once per frame.

| Objective | Natural base ms | Dense ms | Random25 ms | Speed25 ms | Previous observed risk ms | Shared-superset base ms |
|---|---:|---:|---:|---:|---:|---:|
| faithful | 10.740 ± 0.135 | 13.860 ± 0.155 | 11.801 ± 0.108 | 11.783 ± 0.146 | 22.382 ± 0.253 | 11.067 ± 0.147 |
| nll | 10.496 ± 0.027 | 13.593 ± 0.028 | 11.555 ± 0.010 | 11.519 ± 0.019 | 21.881 ± 0.058 | 10.815 ± 0.032 |

| Objective | Extra previous-risk scoring ms | Paired shared-superset minus natural base ms | Previous-risk / natural-base ratio |
|---|---:|---:|---:|
| faithful | 10.596 ± 0.119 | 0.327 ± 0.015 | 2.08410 ± 0.00263 |
| nll | 10.347 ± 0.018 | 0.319 ± 0.018 | 2.08473 ± 0.00109 |

Timing repetitions first reduce within frame; they are not extra trained-model replicates. The documented native MPS configuration uses SciPy host graphs, two PyTorch threads, CPU fallback disabled and synchronized predictor timing. Times include graph construction/selection, features, transfers, network and decoding, and exclude residual/tie/overlap audits and file writing. Every policy runs the checkpoint variance head. The host is shared.

The previous-observed-base risk measurement pays for its additional full scoring pass at every observed frame. Subtracting this cost would not measure autonomous cached deployment. Natural base versus shared-superset base isolates the imposed expanded-search reference overhead on these states; it is not a production speedup. Autonomous policies create different states and use fixed policy order; their durations cannot isolate placement cost, and failed rollouts are shorter. An optimized mean-only GNS, peak memory, general cached deployment speedup and physical conservation remain unmeasured.

The complete observed-state diagnostic supports an allocation-analysis account with adverse outcomes retained. It does not establish general accuracy–runtime superiority or conference readiness.
