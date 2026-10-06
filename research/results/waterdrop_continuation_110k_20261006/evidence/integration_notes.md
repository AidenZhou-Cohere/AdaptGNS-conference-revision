# Integration candidate and compression notes

No manuscript file has been edited by this extraction. Root owns integration and compilation.

- Replace the existing `tab:goop-exposure-placement` table with `candidate_observed_contrast_table.tex`; it adds a clearly separated WaterDrop continuation column and the base-training risk-minus-random row so the observed reversal is visible.
- Insert `candidate_main_waterdrop.tex` after the Goop observed-state paragraph, or use it as the brief continuation subsection. It is approximately 150 words. Keep the two training lineages and H395/H995 distinct.
- Compress the two paragraphs under “Why can residual ranking misallocate?” in `work/conference_experiments_main.tex` into `candidate_compressed_existing_waterdrop.tex`. This preserves the faithful/NLL residual and actual-action correlations, every-seed random advantage at 100k, self-message control, alignment-versus-squared-change result, autonomous sign variation, failures, physical caveat and all six appendix references. The earlier diagnostic numbers are retained from current manuscript text, not newly reaudited by this extraction.
- The prior last sentence “The new Goop result further distinguishes improvement from graph exposure from improvement due to risk placement” becomes redundant once the explicit cross-study comparison is present.
- Use `candidate_waterdrop_appendix.tex` for supporting tables. Full precision and all three seed values, every policy, failure, boundary and cost metric are in `waterdrop110k_full_report.md` and all 91 metric families/12 paired contrasts in `waterdrop_all_scalar_mappings.json`. The scalar package is not a standalone raw reproduction bundle.
- The figure can remain Goop-specific. Its caption correctly states the Goop result and should not be generalized to WaterDrop. A new main figure is unnecessary given the page budget.

Narrow introduction/conclusion updates recommended (candidate text only):

Introduction study paragraph: “A separate WaterDrop study pairs 10k graph-exposure continuations from three fixed faithful 100k parents, alongside the earlier objective, action-benefit and graph-convention controls.”

Introduction results paragraph, following the existing Goop result: “WaterDrop continuation instead reverses the observed-test risk-versus-random ordering in every seed, while its autonomous placement interaction remains mixed. Both studies improve a fixed random policy after graph exposure. Together they show that exposure, observed-state placement and autonomous allocation require separate evidence.”

Conclusion, after the existing Goop sentence: “WaterDrop continuation provides a conditional positive result: graph exposure reverses risk's observed-test deficit in every seed. Yet its autonomous placement interaction changes sign across seeds, despite improved random-policy rollout error. Graph exposure can therefore improve a simulator and sometimes its observed placement without establishing a general autonomous allocation advantage.”

Limitations should explicitly add that WaterDrop effects are conditional on three inspected 100k parents and do not establish fresh-training generality. Do not call the paired studies independent confirmation. Actual title/abstract are untouched; the separate dated abstract candidate is optional author-review material.
