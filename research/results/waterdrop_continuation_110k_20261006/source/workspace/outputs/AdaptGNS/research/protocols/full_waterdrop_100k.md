# Larger local WaterDrop experiment: fixed protocol

Protocol set October 4, 2026, after the compact pilot and before training or inspecting any new full-model outcomes. This is a prospective extension, not a claim that the overall research process was preregistered. Historical aggregate results and the first three official test trajectories have already been inspected.

**Purpose:** test whether the corrected original architecture supports useful cached-risk edge allocation, and whether differences depend on the mean-training objective. This experiment uses more data and the original architecture, but its 100,000-update training budget is shorter than the historical 500,000-update runs.

## Training

- Use every complete trajectory and eligible six-frame history in the official WaterDrop training split. Verify every TFRecord checksum and preserve trajectory IDs and file/record hashes. Validation and test records never enter training.
- Architecture: original 128-wide, ten-message-passing-block GNS with its particle embedding and scalar variance head; six input frames. Metadata radius 0.015, original capped graph and self-loop convention, explicit SciPy host graph construction.
- Train on the base graph with the original random-walk noise, final velocity-noise standard deviation 6.7e-4. Store the same noise-adjusted normalization in every checkpoint.
- Objectives: corrected isotropic Gaussian NLL and faithful regression. The faithful mean path uses the squared-error objective with detached risk features and residuals, providing a mean predictor trained by MSE; faithful regression is established prior work.
- Three paired seeds: 0, 1, 2. Use the same initialization and per-update frame/noise schedule within each seed across objectives. Batch size 2. Adam at 1e-4, decaying exponentially to 1e-5 over 100,000 updates. No weight decay or test-driven hyperparameter changes.
- Use the final checkpoint at exactly 100,000 updates for each requested model. Save recoverable model/optimizer state every 10,000 updates. Preserve interrupted/failed runs. Checkpoint selection does not use test or validation error.
- Native local Metal, with CPU fallback disabled. Record software, device, graph backend, source hashes, elapsed time and training progress. CPU/Metal floating-point trajectories need not be identical.

## Validation and feasibility

Run fixed one-step diagnostics every 5,000 updates on 128 validation frames spread across the official validation split. Keep their identities in the protocol. Report acceleration MSE and residual-scale diagnostics separately from training loss. These checks detect divergence and characterize learning; they do not select a different final checkpoint. Finite smoke tests and sustained throughput determine whether all requested jobs fit the deadline. If a job cannot finish, report it as incomplete and preserve the fixed target; do not silently call an earlier checkpoint the specified result.

## Locked evaluation

Evaluate official test trajectories 3 onward only after the full experiment configuration and checkpoints are fixed. These trajectories were not used in the new compact pilot, though historical aggregate benchmark results were already seen. Keep any later exploratory analyses clearly labeled.

For each full-model checkpoint compare base, dense radius 1.267×base, random optional pairs, speed-ranked optional pairs, and cached previous-step risk. Retain floor(0.25 × available annulus pairs) for the three budgeted policies; preserve the common mandatory base graph. The evaluator uses uncapped symmetric pairs with no self-loops for every policy; this differs from the capped, self-loop training graph and is not an exact historical reproduction. Check graph conventions on identical observed states. Cached risk starts with one base-graph warmup on the same six initial observed frames, whose time and extra pass are included in its cost. It then uses the exact optional budget from the first forecast and caches scores from its own selected graph thereafter. This initialization differs from the compact pilot, which made its first forecast on the base graph.

The primary outcomes are (1) failures before 995 forecast steps and (2) mean position MSE over all 995 steps when every required trajectory completes. Report MSE@200 and @995 as secondary metrics. A group with failed trajectories has undefined all-sample full-horizon error; do not replace it with a survivors-only mean. Guards: nonfinite states, predictions or risk; any absolute coordinate above 10; or more than 100,000 candidate undirected pairs. These are computational limits, not physical-validity tests. Also report boundary excursions, since a numerically completed rollout can still be physically implausible.

Compare policies within each seed on identical trajectory IDs. Give trajectory-level outcomes, equal-trajectory seed means, and sample SD across the three training seeds. With only three seeds, avoid strong significance claims. At identical observed histories, separately compare equal-budget placement, signed dense-intervention benefit, candidate/retained edge counts and synchronized runtime. Autonomous geometries differ, so they cannot by themselves isolate direct placement cost. Include score-generation overhead and candidate construction in end-to-end timing.

## Decision rule

A general superiority claim requires favorable, consistent accuracy and failure outcomes across seeds and controls, together with measured cost. A single favorable horizon, one objective, three inspected trajectories, or an edge-count reduction on different geometries is insufficient. If the full experiment remains incomplete or unfavorable, retain the narrower reproducibility and allocation-analysis framing in the manuscript.
