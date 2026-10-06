# Writing refinement after optional-exposure analysis

This v4 reporting snapshot sharpens the interpretation and notation of the unchanged [audited scientific evidence](../optional_exposure_decomposition_20261006/README.md). The [v3 snapshot](../optional_exposure_analysis_20261006/README.md), all its reviews and its source remain preserved. No model, numerical analysis, experimental protocol, result value, plot coordinate or PNG changed.

The manuscript now foregrounds the controlled empirical contribution, explicitly includes graph-input transfer in the cache-error assumption, and distinguishes the decomposition identity from the additional positive-alignment observation. Scalar coordinate error uses `ell` rather than the Method's vector residual symbol; normalization is explicitly elementwise on the stored acceleration scales. The quadratic prediction-change term is distinct from computational cost. All 22 preceding tables, title/abstract and 29 displayed mathematical environments are unchanged. Native compilation passes the eight-page main-text assertion; the evidence report is unchanged at 16 pages. The native compiler does not export the submission manuscript PDF. Author review and submission checks remain outstanding.

`scientific_narrative_review.md` preserves the proposals; `narrative_refinement_review.json` checks the implemented subset. The optional equal-degree sentence proposed for the main text was not added. `conference_argument_map.md` records current evidence limits and interpretation of future paired-continuation outcomes before those outcomes exist. It changes no numerical protocol.

Reproduce the revised inserts from this directory with NumPy and Matplotlib, choosing a new output path:

```sh
python renderer_source.py \
  --results ../optional_exposure_decomposition_20261006/results.json.gz \
  --audit ../optional_exposure_decomposition_20261006/independent_audit.json \
  --output-dir /path/to/new-render
```

Both LaTeX inserts and the numerical JSON artifacts reproduce byte-for-byte; PDF metadata can differ across renders. The independent renderer review checks 92 mean/SD cells and 90 seed coordinates. Eleven existing renderer tests pass. Tests rely on the repository's frozen synthetic fixture. `MANIFEST.json` pins every included artifact. This package contains result inserts and document verification, not the complete manuscript or an anonymous submission archive.
