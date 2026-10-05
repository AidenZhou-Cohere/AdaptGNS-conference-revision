# Exploratory optional-exposure error decomposition

This protocol and its separate source/tests are reviewed and frozen before extracting any new decomposition outcomes. The original six-model 100k study and its actual-action analysis have already been inspected. This follow-up is descriptive reuse of those saved arrays, not independent confirmation. It performs no model execution, training, graph resampling or optional permutation null. Root alone authorizes execution after review/freeze, on local CPU before October 7, 2026, 08:00 UTC.

## Fixed population and identity

Use every original faithful/NLL seed 0, 1, 2 checkpoint's same-state result: test source indices 3–29, targets 7, 106, 205, 304, 403, 502, 601, 700, 799, 898, 1000. There are exactly 297 histories per model and 1,782 frames. These are observed histories under the original no-self-loop evaluation convention, separate from the graph-convention bridge, native autonomous evaluations and 110k continuations. No frame, seed, model or favorable subgroup is selected.

The byte-pinned existing actual-action analysis is the provenance anchor. Its original `results.json`, `input_identity.json` and `status.json` SHA256 values are respectively:

- `c9a608d3f17cfc84001020b27048ef448958daa2705d135be5f4492e8bbdc008`
- `647c94d886a9d8ee6f2788615c6b5fb95758ff97ca0f94d8ec69a10f9ee41718`
- `04a221181efc1fdb27bc6b2c24712bde280429c3bc257cc060be402b7b2d262a`

Require the original same-state protocol/result/status and each raw JSON/NPZ checksum to match this anchor. The frozen strict same-state validator verifies the final-100k training recipe, checkpoint identities, common runtime/source/configuration/data fingerprints, fixed population, saved calls/normalization and all record/array identities. Recheck current pinned evaluator/validator/action-analysis sources and protocols. Checkpoints and datasets are not loaded or rehashed by this postprocess; their recorded identities remain those of the previously audited original outputs. Hashes do not substitute for unavailable payloads.

For every complete frame, recompute the existing actual-action analyzer from the original predictions, targets, graph pairs and scales; compare every returned scalar/count and derived array against the pinned saved actual-action record/NPZ. In particular, verify vector error, prediction-change alignment/cost, selected-pair budget and optional incidence. This checks saved action arithmetic before adding new groups. Original repeat variability is retained through its recorded source; the saved first timed prediction/selection is used, without a new draw or a favorable repeat choice.

## Fixed comparisons and groups

Primary: previous-observed-base-risk25 minus random25. Secondary: previous-observed-base-risk25 minus speed25; speed25 minus random25. Positive error differences favor the right-hand policy. All three actions retain the same floor(0.25 × annulus-pair-count) optional unordered pairs and all mandatory pairs.

Reconstruct optional incident degree by subtracting the mandatory base-pair set from each selected simple unordered-pair set. Require all selected pairs to be available, every base pair retained, exact budgets, no duplicate/self/out-of-range pairs, and degree sum equal to twice the optional-pair count. This degree excludes mandatory edges and counts each optional edge once at each endpoint.

For each pair, partition all particles into exactly four groups: `neither`, `left_only`, `right_only`, `both`, defined by optional degree zero/positive under the two actions. Split `both` additionally into `both_less`, `both_equal`, `both_more`, according to the sign of left degree minus right degree. These three disjoint supplementary subgroups partition `both`; they are not three additional groups in the primary four-way sum. No data-chosen degree threshold is used.

## Exact coordinate error and alignment/cost identities

For particle i, coordinate dimension is fixed at 2. Let s be the saved two-coordinate acceleration standard deviation, y the target, b the base prediction and p_a a policy prediction. In float64, form

`r = (y - b) / s`, `delta_a = (p_a - b) / s`,
`error_a = sum(((p_a - y) / s)^2) / 2`,
`alignment_a = 2 * sum(r * delta_a) / 2`,
`cost_a = sum(delta_a^2) / 2`.

Every term is in normalized acceleration **coordinate** squared-error units. The previous analysis's vector errors, alignment and costs are divided by 2 before comparison. Verify, per particle,

`error_left - error_right = (cost_left - cost_right) - (alignment_left - alignment_right)`.

The alignment difference is reported with its algebraic sign and is subtracted from the cost difference. Positive alignment difference favors the left action. Do not reverse this sign or mix vector and coordinate quantities.

For every group, retain its particle count/fraction, conditional mean error difference, conditional mean alignment/cost differences, conditional mean optional-degree difference, and the corresponding **unconditional whole-frame contributions** `sum(group_mask * quantity) / N`. Empty groups have fraction/contributions exactly zero; all conditional means are null with an `empty_group` reason. Report raw group/subgroup labels and per-particle quantities in a new derived NPZ.

Primary group contributions must sum to the whole-frame paired error, alignment, cost and optional-degree differences. The three both-covered subgroup contributions must sum to the both group's contribution. Each group's error contribution must equal cost contribution minus alignment contribution. Equal budgets also require whole-frame mean degree difference zero. All signed adverse values are retained.

For these cancellation-sensitive identities, use an absolute bound `256 * eps_float64 * max(number_of_summed_terms, 1) * max(mean_or_per_particle_absolute_magnitude, 1e-300)`, where magnitude includes the nonnegative left/right/base errors and both policy costs and absolute alignment terms as applicable. Saved-array replay uses the frozen actual-action analyzer's existing tolerances; these tolerances are not changed after outcomes. Exact integer mask/degree/count/budget checks have no tolerance.

## Aggregation and nulls

Average each unconditional contribution/fraction over the fixed eleven frames within each trajectory, then over all 27 trajectories for each seed. Report all three seed values, their arithmetic mean and sample SD (`ddof=1`) separately for faithful/NLL. Preserve all three comparison directions and every group/subgroup. The four-way and both-subgroup additive identities are checked again for trajectory, seed and three-seed means. Standard deviations themselves are not additive.

Conditional frame means have strict all-required-frame aggregation: a single empty required group makes that conditional average null, with exact undefined counts. Never average only nonempty frames. As a separately labeled descriptive summary at trajectory/seed level, also report `weighted_conditional = weighted_unconditional_contribution / weighted_group_fraction`. It is defined only when every required contribution/fraction is defined and the weighted fraction is positive. This ratio is an explicitly group-size-weighted effect under the fixed frame weights, not an average of nonempty-frame conditional means and not a rescue of a missing scientific input. Across-seed summaries of this ratio still require exactly three defined seeds.

Every expected frame gets a new JSON record, including failed/missing/invalid inputs. A missing or invalid committed model-level source/provenance makes all that model's frame metrics null. A missing or invalid raw/action frame leaves its new metrics null and preserves its source hashes/error reason. A failed original frame remains failed. The core decomposition also propagates an unavailable action only to comparisons that require it; this does not bypass the stronger original checksum validation. Any required null propagates to the corresponding trajectory/seed/three-seed mean. No survivor-only headline or available-seed average is produced. Cross-model provenance mismatches make all scientific means ineligible. Earlier failed attempts remain on disk; an existing output directory is refused.

## Interpretation and saved outputs

Optional exposure is produced by the policy. Conditioning on it is a descriptive partition, not adjustment for a pretreatment confounder or a causal effect estimate. Ten message-passing blocks can propagate effects to particles with no directly incident optional edge; `neither` does not mean unaffected. A larger cost or an uncovered-group gap may suggest a follow-up mechanism, but neither identifies a marginal edge value or proves concentration caused the performance gap. Speed is included because concentration is not specific to risk. No particle-level significance tests or policy speedup are claimed.

Save every per-frame/group metric, exact contribution identity residual/bound, derived-array hash, trajectory/seed result, conditional null count, all raw/action input identities, source/software hashes, failed outcomes and attempt status. Rehash inventories and source dependencies before final publication. `analysis_wall_seconds` includes validation, hashing, decomposition and output I/O; it is postprocessing runtime, not policy inference time.

From the repository root after review/freeze:

```sh
python -m research.analyze_optional_exposure_decomposition \
  --evaluation-root /path/to/work/full-evaluation/same_state \
  --action-root /path/to/work/full-action-benefit-20261005 \
  --output-dir /path/to/new-optional-exposure-analysis
```

This primary analysis deliberately defers the proposed 64 graph-only permutation nulls; no null graphs or new predictions are generated here.
