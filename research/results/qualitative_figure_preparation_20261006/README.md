# Qualitative trajectory figure preparation

A fixed renderer for prospective Goop2D, Goop3D and Sand trajectory figures. The
example source is selected by particle-count metadata, with seed0 and saved
forecasts1,200,H. Common physical axes, complete particle sets, explicit failed
and missing panels, and out-of-box markers prevent a favorable-example display.

**Every plotted example here is synthetic. This family contains no trained model,
real dataset, new experimental finding or manuscript figure.** Real rendering
requires the completed study and its full failure accounting to be admitted.
The renderer is not a scientific-admission tool.

The source and17 synthetic checks are in `source/`; details and exact future
request fields are in `source/qualitative_renderer_v1.md`. The final inspected
examples are `source/qualitative_synthetic_v3`. Earlier layouts and source snapshots
are retained in v1/v2, including the repaired3D heading overlap and clipped axis.
Recorded requests and receipts retain their original authoring paths; regenerate
synthetic fixtures to use this package in another checkout.

From this directory, with Python, NumPy and Matplotlib available:

```sh
python3 source/qualitative_synthetic_checks_v1.py
python3 source/qualitative_synthetic_checks_v1.py --make-fixtures source/reproduced_synthetic
```

The second destination must not already exist. It creates synthetic PDF figures
and companion JSON receipts. Poppler can render these PDFs to PNG for inspection.
No simulator or GPU computation is involved. All17 checks also pass from a separate
copy of this package; root inspected the final2D/3D fixture layout.

The optional global action-gate column and WaterDrop continuation are not supported
by this small renderer. Native standalone LaTeX image integration is also pending;
these artifacts do not change or replace the open manuscript. A final figure must
still be checked at its actual manuscript scale. `original_copies.json` binds the
unaltered originals; `manifest.json` binds this package, including verification.
