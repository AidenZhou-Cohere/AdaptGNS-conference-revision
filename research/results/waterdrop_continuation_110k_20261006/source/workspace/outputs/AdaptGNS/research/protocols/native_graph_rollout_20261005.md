# Exploratory native-graph autonomous bridge — October 5, 2026

This separate follow-up restores the original capped base graph and self-message convention during full autonomous rollout. It was designed after inspection of the completed locked experiment and before any outcomes from this new rollout family. The graph-convention observed-history bridge was already designed and launched; no favorable one-step outcome selects this rollout population. Original sources, protocols, 100k checkpoints, all eight prior NLL failures, and all locked evaluations remain unchanged and separately named.

Root reviews and commits this protocol, its new source and synthetic tests before launch. Only root launches the six models sequentially on local compute. No new trajectory starts at or after October 7, 08:00 UTC; any partial evaluation is preserved as incomplete. There is no external submission, messaging, purchase, or cloud compute.

## Fixed models and population

Use all six original final 100,000-update checkpoints, faithful/NLL × seeds 0,1,2, without selecting a checkpoint by validation or test results. Use all official test source indices 3–29 and all five existing policies: base, dense, random25, speed25, laggedrisk25. Every planned rollout has 995 forecasts following the first six observed frames: 810 trajectory-policy outcomes total. The official test TFRecord SHA256 remains `b7f147c22e96fd3fbb8d595702cb8c85e412bfb449d7b4d761cc03eb76246c31`. The converted manifest and every consumed positions/types array are checksum verified.

Historical aggregates, test indices 0–2, all locked outcomes, and prior observed-state geometry were inspected. This is exploratory, not independent confirmation. The frozen initial random seed is `93000 + 1000*training_seed + source_index`, matching the locked rollout. Later draws can differ when native graph dynamics change candidate counts; common initial random material does not create identical actions on different states.

## Graph and action

For each forecast, construct the strict geometric radius-r pair universe from the current predicted positions, with r=.015, R=1.267r, native float32 Euclidean norms and separate SciPy queries. Construct the mandatory directed graph by the native rule: receiver-major ordering, neighbor distance then source-ID order, strict `< r`, self-edge candidates included, nearest 128 candidates per receiver. The cap can make directed adjacency asymmetric and, under coincident-position ties, may exclude a self candidate. This is the actual training graph convention, not an idealized guarantee that every capped node retains its self edge.

Optional unordered pairs are the strict geometric annulus, E_R minus E_r. Missing within-r edges excluded by the native receiver cap do not become optional candidates. Preserve the full native directed base **as an ordered prefix**. Append both orientations of selected optional pairs, ordered by receiver, distance, then source ID. Never cap or remove edges after appending. Optional pairs exclude self edges. This retains native base messages even when their receiver cap binds while allowing the explicitly budgeted additional interactions.

The combined array contains an ordered native-base block followed by an ordered optional block. It differs from globally sorting the union by receiver, as the observed-history bridge does. The base-only path remains exactly native. Floating-point message summation can depend on ordering; this distinction is retained rather than describing the two evaluators as byte-identical on expanded graphs.

- Base appends no optional pair.
- Dense appends all annulus pairs.
- Random25, speed25 and laggedrisk25 append floor(.25 × available annulus pairs), exactly.
- Speed ranks max endpoint speed, with speed from the final predicted position difference.
- Cached risk ranks max endpoint positive variance from the previous selected-graph prediction. It first pays for a separate native-base score pass on the initial six observed frames; that mean prediction is discarded. Forecast 1 already uses the exact optional budget.
- Lexicographic optional-pair order and stable descending rank sorting fix ties. Every later history and score is the policy's own predicted state/output. Future targets never enter selection or prediction.

The uncapped additions mean adaptive receiver degrees can exceed128. This is intentional; applying a final cap would violate the base-preserving exact optional-budget intervention. The comparison restores native base semantics and changes self edges, radius classification, cap behavior, and edge ordering together relative to the original locked evaluator. It is a full native-convention bridge, not an isolated causal self-loop experiment. The separately designed observed-state factorial helps interpret these components. Neither experiment solves the original model's lack of expanded-graph training exposure.

## Native parity gate

Before each policy trajectory, run the frozen graph bridge's native-versus-supplied base check on the initial six frames. Require exact ordered directed-edge, node-feature and edge-feature identity. Require decoded-prediction absolute agreement ≤2e-7 and raw/converted risk agreement with absolute1e-6 and relative1e-5 tolerances, with finiteness checked before conversion can conceal a problem. Save both raw outputs and features. A parity failure is committed as an unsuccessful outcome and stops that model job for review; it is not rerun or patched silently.

These two verification passes are separate from policy execution. Cached-risk warmup is an additional explicit policy scoring pass. Save parity duration, policy duration including warmup, total duration including parity, and both network-pass counts. No part of this fixed-order changing-state evaluation establishes a causal latency or speedup comparison.

The reused predictor component timer starts after input tensor construction and transfer. It includes feature construction, network forward, decoding and output transfer; input preparation/transfer is included only in total policy wall time. Graph durations also include count/hash/asymmetry diagnostics. Per-trajectory wall time includes scalar diagnostics but excludes its subsequent compressed-file publication. These scopes are operational accounting, not an optimized benchmark.

## Guards, failure and artifacts

Retain the original computational guards: nonfinite states/predictions/raw or converted risk, nonpositive converted risk, maximum absolute coordinate10, and more than100,000 expanded candidate unordered pairs. Guard failure stops the trajectory at the rejected attempt; every accepted prefix, rejected-step reason and available raw rejected prediction/risk/history/edges is preserved. State and graph guard failures save the failed input history and available cached score even when no forward output exists. No scientific failure is retried. A required failed/missing trajectory leaves its policy's all-sample full-horizon error and edge/boundary aggregates undefined. Finite prefixes are diagnostic only.

For every attempted forecast save coordinate MSE after prediction, native-base directed/self-edge count, cap-active receiver count and removed-edge count, directed asymmetry, graph/pair hashes, candidate/optional count, prediction hash, boundary diagnostics with the same truth reference, graph time and forward time. The native base prefix must remain exact, and total directed edges must equal native base edges plus twice the selected optional pairs.

Save scalar rows and compressed numeric traces at forecasts1,10,50,200,500,995: current six-frame predicted/observed history, forecast position, same-step truth, current and prior score, selected optional pairs and full directed edges. A missing prior score for noncached policies at the first forecast uses a NaN trace sentinel; it never enters inference. Preserve initial native parity arrays and explicit cached-risk warmup outputs. Numeric traces can contain failed nonfinite outputs without pretending they are completed metrics.

Each new run manifest pins source, protocol, checkpoint, original training-config hash, official data, backend/library versions and thread count. Recheck bytes on completion. Exclusive reviewed locks, atomic rows/arrays/status and checksum-verified recovery prevent duplicate writes. A prior failed trajectory is recovered as failed. Orphan traces/temporary files require review and preservation, not overwrite. `--clear-stale-lock` is only allowed after native process identity and dead-lock verification.

## Prespecified analysis

Keep all three seed values for each objective/policy. Report equal-trajectory seed means and mean ± sample SD across the three seeds for full-horizon mean position MSE, MSE@200 and MSE@995, all failure counts, boundary fractions/excursions relative to same-step truth, and native graph/cap/asymmetry diagnostics. Use the same original unit conventions. A required failure makes only its relevant full-population metrics undefined; no survivor mean replaces them. Report failed prefixes separately.

Pair identical source IDs within seeds. Primary policy contrasts are cached risk minus base, random25 and speed25; dense-minus-base and current bridge minus original locked outcome are descriptive convention controls. Report every objective and seed; do not select only a favorable native result. An original/native full-horizon paired comparison is undefined when either required trajectory failed. Failure counts and secondary horizon metrics remain defined where coverage permits. Do not treat shorter failed durations as speed improvements. All outcomes and original failures remain in the manuscript/evidence record regardless of direction.

## Command

From the research repository, using the recorded experiment Python:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m research.native_graph_rollout \
  --checkpoint research/results/full_waterdrop_100k/faithful_seed0/checkpoint-100000.pt \
  --test-manifest ../../work/full-data/converted/test.json \
  --output-dir ../../work/native-graph-rollout-20261005/faithful_seed0 \
  --device mps --threads 2
```

Root substitutes each of the six fixed model identities, one at a time, after review. No new training or model selection occurs in this module.
