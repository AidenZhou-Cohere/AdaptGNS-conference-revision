# Goop3D 25k observed presentation renderer

`generated_candidate_v1/` is a review candidate, pending independent review of the actual finalizer products. No canonical manuscript source was edited. Its `inputs.json` records this pending status explicitly. Root owns eventual integration and the separate autonomous analysis.

The renderer reads only local hash-bound saved JSON: audit, unchanged summary, exact-variance arithmetic check, cache-finalization completion, retained original failure, and the diagnosis of the 36 false nonzero graph SDs. It requires all 2,568 observed cells and all 4,728 original accounting cells, all 18 observed model/stages, both accuracy metrics, all six policies, every fixed within-arm contrast, and all four clean metrics. It checks the full known diagnostic-key grid and all paired arithmetic against the saved seed means. It never loads NPZ files, models, original arrays or autonomous products.

Run from the workspace, with absolute paths and a fresh output directory:

```sh
work/venv/bin/python -B work/goop3d_completion_20261007/presentation_observed_v1/render_observed.py \
  --inputs /absolute/path/to/inputs.json \
  --inputs-sha256 EXACT_SHA256 \
  --output /absolute/path/to/fresh_candidate_directory
```

The input manifest has schema `goop3d_observed_presentation_inputs_v1`, a `review_state`, and `products` entries for `audit`, `summary`, `arithmetic_check`, `completion`, `retained_original_failure`, and `diagnosis`. Each entry contains only the exact absolute `path` and `sha256`. The actual candidate manifest is `actual_inputs_candidate_v1.json`, SHA `2bdc5118d2219b74df4e8aee9b1a7e188cfede73afb822e0de9458709175a1c6`.

The finalizer audit, summary, check and completion hashes are cross-checked internally; paths and bytes are reverified before publishing the new candidate. The formatter uses exact-rational pairwise sample variance for its reduction check. Identical large binary floating-point inputs therefore have zero SD. Saved statistics are never modified; three required seeds remain three, and undefined diagnostic correlations remain null.

The candidate contains:

- `appendix_goop3d_observed.tex`: a self-contained appendix fragment with 13 tables. Separate component fragments retain four complete accuracy panels, two five-contrast tables, two all-policy training-effect tables, all clean values/effects, descriptive timing, risk/error/benefit correlations, and historical accounting. Both validation and test stay visible. All accuracy/contrast/clean seed values are displayed; every other seed/metric remains in the complete companion.
- `findings_observed.md`: all policy training-effect signs, both within-arm risk gaps and the interaction, derived from exact saved seed values. It preserves heterogeneous exposure effects and validation/test disagreement.
- `claim_source_map.json`: every rendered statistic's source summary hash, JSON path, multiplier and ordered seed signs.
- `source_summary_unchanged.json`: a byte-identical copy of summary `2130ccb1c0c116766a6f2ec6e724102fed48edbd692a279026aaf1dcdaa09109`.
- `complete_observed_companion.json.gz`: all 1,382 statistic objects, the unchanged summary, every original accounting cell, all clean/timing/benefit/correlation/graph/boundary values, input hashes, checked finalization lineage, retained failure and diagnosis. Full per-row audit details remain in the hash-bound audit input.
- `receipt.json`: source/output hashes, counts, the companion's uncompressed hash, and explicit flags confirming no manuscript edits or autonomous/scientific execution.

Position-coordinate MSE entries and contrasts multiply saved coordinate-squared values by **10^9**, explicitly captioned as units **10^-9**. Normalized quantities retain their own units. Six-significant-digit text preserves small nonzero signs; the companion retains full precision. Three-seed sample SDs are not confidence intervals. A narrower risk gap is not a general risk advantage, and fixed-policy exposure effects do not establish that expansion beats native base.

`% GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT` is a future insertion point at the end of the fragment. An independently verified H295 fragment and autonomous companion may follow it. Their cached-risk protocol, full-horizon metrics, per-policy failures and denominators must remain separate from observed statistics. Historical full-rollout rows in the accounting table describe the original stopped attempt only.

The source is mapped by `../editorial_integration_plan.md`: replace the attempted-Goop3D appendix paragraph only after review, add at most one scoped main-text reference, preserve the two main three-material tables and existing adverse results, then perform native compilation and visual QA. No layout certification is claimed by this renderer.

Synthetic verification:

```sh
work/venv/bin/python -B work/goop3d_completion_20261007/presentation_observed_v1/test_render_observed.py
```

Eight tests cover the complete fixed grid, output/source immutability, large constant graph means, tiny signs, missing seeds/survivor means, missing policies, wrong coverage, bad hashes/linkage, mixed observed/autonomous schemas, diagnosis scope, unsafe paths and duplicate JSON.
