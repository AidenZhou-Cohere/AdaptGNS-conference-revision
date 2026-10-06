# Inline qualitative figure preparation

This separate backend emits plain LaTeX picture fragments for the existing
standalone manuscript, with no external images or additional LaTeX packages.
It preserves the already-fixed source, seed, times, every particle, explicit
failed/missing panels, and visible out-of-box counts. A defined orthographic
projection makes the 3D view consistent across methods. The original reviewed
renderer remains unchanged.

**All figures and checks here are synthetic. No model, empirical outcome or
scientific manuscript was changed. Native LaTeX compilation and final figure
layout remain unverified.** The fragments are not standalone documents and must
not be presented as research evidence. Real use requires complete-study admission.

`source/qualitative_picture_exporter_v1.md` describes integration and limitations;
`source/qualitative_picture_authoring_record_v1.json` binds the final source and
synthetic fragments. Earlier caption variants are retained. Root repeated the
invariant/stress checks with fresh synthetic inputs in a separate source copy,
and both resulting fragments matched the final original bytes exactly. The
90000-glyph stress case serialized3.27MB; native TeX time and memory are unmeasured.

To reproduce from this directory with Python, NumPy and Matplotlib, copy these
five files from `source/` into a fresh directory: `export_qualitative_picture_v1.py`,
`render_qualitative_v1.py`, `qualitative_selection_plan_v2.md`,
`qualitative_synthetic_checks_v1.py`, and `check_qualitative_picture_v1.py`.
From that fresh directory run:

```sh
python3 qualitative_synthetic_checks_v1.py --make-fixtures qualitative_synthetic_v3
python3 check_qualitative_picture_v1.py
python3 export_qualitative_picture_v1.py --request qualitative_synthetic_v3/Goop3D/request.json --output-prefix inline_reproduced/goop3d
```

The commands create synthetic-only artifacts and do no inference or compilation.
Their destinations must be fresh. Original receipts preserve authoring paths;
regenerating fixtures avoids depending on those paths. The full original raster
fixtures and visual reviews remain in the adjacent
`qualitative_figure_preparation_20261006` family (manifest
`f4a9bdbf688e853e171ea4843878bd50a684a6875838e33644c52542e584a9e1`).

`original_copies.json` binds17 unaltered source/artifact files. `manifest.json`
binds this complete preparation family. Root must still integrate actual admitted
figures into the same manuscript, compile natively and inspect their final scale.
