# Inline qualitative picture exporter v1

`export_qualitative_picture_v1.py` is a separate presentation backend for the
sealed `render_qualitative_v1.py` (SHA256
`38937abd2474a212d0fecf515e00a00d778d56c4f477eb5e3d0a5a4f11d103eb`).
It checks those source bytes before import and delegates all request parsing,
fixed source/seed/time selection, input hashes, and study identity checks to
that unchanged renderer. No frozen experiment source was changed.

The output is a **fragment for pasting into the existing standalone manuscript**,
not a second LaTeX document. It contains one `figure*`, a 480-by-427-point plain
`picture`, caption, and label. It has no document class, preamble, external image,
input file, package, macro definition, or compiler call. The manuscript's current
preamble does not load a color package, so this exporter uses filled black
circles and distinct crossed markers with explicit out-of-box counts. This
monochrome convention is stated in the caption and legend.

All particles are emitted in their existing ID order. There is no subsampling,
occlusion culling, or inference. A 2D coordinate uses the physical x-y identity
projection. A 3D coordinate uses an orthographic view at elevation 25 degrees,
azimuth -55 degrees, identical for all panels. For azimuth a and elevation e,
the two projection rows are `[-sin(a), cos(a), 0]` and
`[-sin(e)cos(a), -sin(e)sin(a), cos(e)]`. Coordinates are centered on the
metadata-box center and scaled equally in both display dimensions, using the
projected metadata corners. The receipt records the exact matrix and bounds.
This matches the declared camera direction while explicitly defining the
projection independently of Matplotlib's rendering internals.

Points outside the physical metadata box are clamped to its physical boundary
before projection and shown as crosses, with no silent removal. Each populated
panel shows the exact count. Missing panels show the recorded state and unknown
prefix; committed failures retain their known prefix, including zero. Failed
trajectories' saved earlier forecasts stay visible with their failure annotation.
The exact 1/200/H indices and source-frame mapping are retained. Truth still
comes only from agreeing saved targets; no raw-data fallback was introduced.

## Invocation and integration

Use the same admitted request schema described in `qualitative_renderer_v1.md`.
The output prefix must be fresh. For a synthetic example:

```sh
env MPLCONFIGDIR=work/conference_presentation_20261006/.mplconfig \
  work/venv/bin/python work/conference_presentation_20261006/export_qualitative_picture_v1.py \
  --request work/conference_presentation_20261006/qualitative_synthetic_v3/Goop3D/request.json \
  --output-prefix work/conference_presentation_20261006/qualitative_inline_synthetic_v2/goop3d
```

That exact prefix is already generated; use a new versioned prefix when rerunning.
For admitted research results, change only the request path and fresh output
prefix. Do not remove the SYNTHETIC CHECK title from synthetic fragments or
present them as research results. Paste the generated `.tex` fragment's contents
into the existing manuscript; do not add an `input` or `includegraphics` link.
Root owns manuscript integration and native compilation. No new packages should
be required. Keep its caption with the plot; the caption includes numeric bounds,
selection, cached-risk distinction, missingness, and interpretation limits.

## Checks and size

`check_qualitative_picture_v1.py` checks synthetic fixture identities, all-particle
glyph counts, outside counts, missing/failure prefixes, forecast/source-frame
mapping, the orthonormal projection basis, bounded projected metadata corners,
escaped TeX text, balanced braces, and absence of file/package/compiler commands.
`qualitative_picture_checks_v1.json` preserves the check result. The two fixture
fragments contain 231 and 297 particle glyphs, respectively, with two crosses
each. A repeated-coordinate synthetic stress case used 10000 particles in each
of nine populated panels: all 90000 glyphs were emitted, producing 3270486 bytes
in about 0.047 seconds on the local interpreter. The stress source was measured
in memory; it is not a research result and is not inserted into the manuscript.

Source size grows linearly with the number of displayed particles. Native TeX
compile time, memory use, and final layout are **unverified**; no compiler was
called for this task. The large stress result identifies a concrete possible
integration cost, not a measured TeX failure. Do not silently reduce particle
counts if the native compiler cannot handle a real figure. Preserve that outcome
and review the integration choice. Root must inspect the compiled final figure.

An independent source/schema review found no actionable issue in the projection,
particle preservation, failure states, or plain-picture structure. The final
numeric-bounds caption refinement did not change the geometry. All work remained
synthetic/source-only; no real outcomes, datasets, models, remote calls, installs,
or manuscript edits were used.

The initial `qualitative_inline_synthetic_v1` fragments remain preserved. Final
`qualitative_inline_synthetic_v2` adds numeric physical bounds to the caption.
`qualitative_picture_authoring_record_v1.json` records the exact final file hashes.
