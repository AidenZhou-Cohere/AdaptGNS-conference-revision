# Admitted fixed Goop2D qualitative figure

The export is ready for root to integrate into the existing manuscript and
compile in its existing editor. No manuscript, canonical source, frozen
exporter, evaluator, training source, or selection was changed by this step.
No model inference or extra test selection was performed.

## Immutable figure and interpretation

- Figure: `goop_fixed_source12_inline_v1.tex`, SHA256
  `2a5b34584dda3196f3070c5ba4eb072140e1999943edfc10ae611bd8a03ba6b4`.
- Export receipt: `goop_fixed_source12_inline_v1.json`.
- Adjacent prose: `integration_note_v1.tex`. Keep this outside the immutable
  generated fragment. It records the visible elevated-particle mismatch in
  both mixed forecasts, the changed training/policy factors, and the different
  boundary-count thresholds.
- Figure label: `fig:qualitative-goop2d`.
- Independent numerical check: `selected_inline_verification_v1.json`.
- Direct PNG preview: `goop_fixed_source12_inline_direct_preview_v1.png`,
  SHA256 `7d49af42634a6bbf2408507ea0b316eeae0101c9dcab2e21d946f048d2c1c517`.
- Visual-review record: `selected_inline_visual_review_v1.json`.

The fixed source is seed 0, source 12, `test:000012`, 1,083 particles: the lower
median (zero-based rank 14) of the 30-source census sorted by initial particle
count and original source index. All 1/200/395 forecast panels are retained.
All three selected trajectories complete 395 forecasts. Counts of strict
box crossings at forecasts 1/200/395 are truth 0/0/0, base/base 0/553/465,
mix/random25 0/0/11, and mix/laggedrisk25 0/0/14. They describe only this example.

## Canonical insertion recipe (root only)

The current canonical sources are `work/manuscript_body.tex` and
`work/conference_experiments_main.tex`; the existing builder is
`work/build_manuscript.py`. The builder embeds their contents in the single
`outputs/revised_manuscript.tex` used by the native editor.

1. In `work/manuscript_body.tex`, keep the figure in the existing
   `sec:goop-graph-exposure` appendix, after `tab:goop-all-cost` and the existing
   paired-seed `fig:goop-exposure-placement`, before any later appendix or
   `\end{document}`. Append the exact contents of `integration_note_v1.tex`,
   followed by the exact figure fragment. This retains the quantitative
   attribution before its qualitative illustration. Paste/embed contents;
   do not introduce an external `\input` or image dependency.
2. Optionally append this one sentence to the Goop physical-diagnostics
   paragraph in `work/conference_experiments_main.tex`:
   `A fixed metadata-selected rollout illustrates the remaining physical
   mismatch in Figure~\ref{fig:qualitative-goop2d}.`
3. Run the existing canonical builder from the workspace root:
   `work/venv/bin/python work/build_manuscript.py`.
4. Verify the generated standalone manuscript includes the figure exactly
   once and retains the complete 12-panel grid; then compile and inspect only
   the existing manuscript in its existing editor. Check figure/caption
   placement, float ordering, glyph visibility, main-page limit, and references.

This is a 480-by-427 point full-width `figure*` with 12,996 glyphs and a long
caption. It belongs in the appendix. Native integrated rendering remains for
root to verify; the direct PNG is not a TeX-compiled proof.

## Reproduction without inference

The root fetch receipt preserves all eight fetched result files and hashes.
The manifest and request fix source, seed, policies and frames. To reproduce
the existing export from those admitted files, use a new output prefix if
preserving all existing artifacts:

```sh
MPLCONFIGDIR=/tmp/adaptgns-goop-qualitative-mpl-v1 work/venv/bin/python work/conference_presentation_20261006/export_qualitative_picture_v1.py --request work/conference_presentation_20261006/goop_qualitative_request_v1/render_request_v1.json --output-prefix /tmp/goop_fixed_source12_inline_recheck
```

The independent checker is
`verify_and_preview_selected_inline_v1.py`. It rechecks fetched hashes,
source/seed/truth/history identities, all 12,996 glyph coordinates and classes,
panel counts and fragment structure without importing model, evaluator or
renderer. Its direct PNG uses approximate fonts. It creates no PDF and opens
no tab.
