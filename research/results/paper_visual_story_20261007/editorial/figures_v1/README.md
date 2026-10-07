# Main visual story

Two self-contained native LaTeX figures replace the earlier single-material slope chart, small method schematic, and two main contrast tables. Root retains ownership of canonical integration and compilation. No canonical file was edited here.

## Opening: teach the intervention, then separate the questions

Use `constructive_overview.tex` near the introduction (`fig:constructive-overview`, 470 × 238 pt). The top row shows the actual constructive operation: four native pairs remain, six local pairs are candidates, and one selected pair is appended. The bottom row separates learning to use extra messages, choosing their placement on fixed observed histories, and sustaining placement gains in autonomous feedback. Each card shows all three seed signs for all three materials, including Goop's undefined full-horizon interaction. These are three distinct contrasts, named within the cards; they are not three estimates of one effect.

The graph is an explanatory drawing, not an experimental particle snapshot. The colored triplets use existing exact scalar records. Do not interpret a negative training effect as evidence that expansion beats the native graph. Keep the main-text statement that native base beats random expansion in every Sand seed of both arms.

## Evidence: replace both repetitive main tables

Use `cross_material_evidence.tex` in Experiments (`fig:cross-material-evidence`, 470 × 330 pt). This is taller than the initial 290 pt target to keep 12 rows and their labels readable. It retains every row/column of both main tables, adding the existing autonomous risk-gap interaction as the final highlighted row. Goop, WaterDrop, and Sand remain separate columns, each with its explicit endpoint and scale.

The upper five rows are the original observed-test-history contrasts. The lower seven rows contain all six policy training effects and the risk-gap interaction. A diamond with a horizontal interval shows the unchanged mean ± sample seed SD; three small points preserve individual seeds. Colors indicate negative or positive contrast. No confidence interval or new fit is computed. Four Goop three-seed means stay undefined: base, dense, cached risk, and the risk-gap interaction. Their two defined seed points remain visible, with a dagger rather than a survivor mean. WaterDrop RMS remains NP. The caption preserves the observed-base versus autonomous-own-graph risk distinction; root retains full-cohort counts in the main text.

Update old main references rather than aliasing them blindly: `fig:goop-exposure-placement` formerly referred to A/B Goop panels; `tab:goop-exposure-placement` and `tab:goop-rollouts` now point to the observed/full bands of `fig:cross-material-evidence`. The earlier `fig:interaction-schematic` can be removed because the construction is in the opening figure. Preserve full scalar tables in the appendices.

Root's existing fixed-source Goop qualitative strip is a complementary third main figure. It should be labeled as physical mismatch in a fixed example, not a paired training-effect comparison.

## Files and checks

Both figures use native `picture`, `xcolor`, and standard fonts; no image inclusion, TikZ/PGF, new LaTeX package, alternate PDF build, or native UI operation is required. Each `*_picture.tex` contains just the grouped picture. PNG/SVG previews come from the same primitives and draw order. Root still needs native compilation; final native pixel/page inspection awaits author export.

Rebuild locally:

```sh
work/venv/bin/python -B work/paper_visual_story_20261007/figures_v1/build_figures.py
```

`fidelity.json` binds the original admitted cross-material presentation maps and copies every displayed record without recomputing means or SDs. It records 27 opening seed outcomes, 36 evidence slots, 101 plotted seed effects, four undefined means, and one NP slot. Every data/interval coordinate is inside its declared axis; preview text boxes stay on canvas. `*_primitives.json` binds both native and preview products. Earlier source files remain unchanged. The first preview and its corrected mean-marker stacking issue are retained under `history_first_preview/`.

The latest label/caption correction explicitly identifies the observed test split, fixes the Train quotation, and shortens both captions. `test_label_caption_delta.json` verifies that all data, coordinates, axes, and non-text primitives are unchanged. The complete preceding version is retained under `history_before_test_labels/`; `source_visual_review.json` describes that preceding geometry review and the delta supplies the current product hashes.
