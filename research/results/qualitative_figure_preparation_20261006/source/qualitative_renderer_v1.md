# Fixed qualitative trajectory renderer v1

This is a source-only presentation tool, prepared and checked with inert
synthetic arrays. No research outcomes, data arrays, checkpoints, or inference
were accessed during authoring. It follows `qualitative_selection_plan_v2.md`.

`render_qualitative_v1.py` consumes a hashed test manifest's **scalar metadata**,
three committed row/protocol pairs (or explicit missing cells), and only their
saved numeric trace NPZs. It imports neither evaluators nor model code. Root
must first admit the complete study and its failure/missing accounting; this
renderer checks artifact identity and consistency, not scientific admission.
Its required `root_admission_note` for real artifacts records that distinction.

Supported study names are `Goop2D`, `Goop3D`, and `Sand`. They bind respectively
to manifest dataset names `Goop`, `Goop-3D`, and `Sand`, forecast horizons
395/295/314 and training budgets 100000/25000/100000. All are faithful models
at seed 0. Each request renders one study separately. The full metadata census
determines the lower median of `(initial particle count, original source index)`;
the caller cannot choose another source or display time.

The three predictor columns are fixed: `base/base`, `mix/random25`, and
`mix/laggedrisk25`. The two mix policies must bind the same checkpoint hash.
`laggedrisk25` is cached rollout risk, not the separate
`previous-observed-base-risk25` same-state diagnostic. The optional separate
Goop global action gate is not supported by this small renderer: its artifact
schema differs. Append it only with a separately reviewed extension after its
entire declared comparison completes; never substitute it for a fixed column.

## Input request

Every file reference is `{"path": "...", "sha256": "64 lower-case hex digits"}`.
Paths are absolute or relative to the request. Row NPZ filenames are resolved
relative to the row file, using `trace_file/trace_sha256` for Goop2D and Sand,
or `artifact_file/artifact_sha256` for Goop3D. JSON and NPZ are parsed from the
same bytes that were hashed. `allow_pickle=False` is used for NPZ.

The request has this structure; replace every illustrative path/hash/note with
the exact admitted values before executing:

```json
{
  "schema": "qualitative-render-request-v1",
  "study": "Goop2D",
  "synthetic": false,
  "root_admission_note": "Identify the completed study and root admission/accounting receipt here.",
  "selection_plan": {"path": "qualitative_selection_plan_v2.md", "sha256": "<plan SHA256>"},
  "manifest": {"path": "<admitted test manifest>", "sha256": "<manifest SHA256>"},
  "cells": [
    {"arm": "base", "policy": "base", "row": {"path": "<selected base row>", "sha256": "<row SHA256>"}, "protocol": {"path": "<base protocol>", "sha256": "<protocol SHA256>"}},
    {"arm": "mix", "policy": "random25", "row": {"path": "<selected random row>", "sha256": "<row SHA256>"}, "protocol": {"path": "<mix protocol>", "sha256": "<protocol SHA256>"}},
    {"arm": "mix", "policy": "laggedrisk25", "row": {"path": "<selected cached-risk row>", "sha256": "<row SHA256>"}, "protocol": {"path": "<mix protocol>", "sha256": "<protocol SHA256>"}}
  ]
}
```

For an unreturned cell, replace its `row` and `protocol` with
`"coverage_state": "timed_out_current", "reason": "Exact accounting reason"`.
Other permitted missing states are `not_completed_before_invocation_end`,
`never_started`, and `outside_evaluated_source_grid`. The last is valid only
when the chosen Goop3D median is outside its declared full-source-order grid:
all N for N<=30, else `floor(j*(N-1)/29)` for j=0..29. It is never valid for
Goop2D or Sand. A missing cell has unknown accepted prefix, not zero. Committed
failures retain the recorded prefix, including zero, and all accepted traces.

Requested forecast IDs are exactly 1, 200 and H; lookup uses the saved
`forecast_steps` values. Original source-frame indices are 6, 205 and H+5.
Missing saved times are labelled, never backfilled. Rejected predictions are
never loaded. Ground truth is assembled from agreeing saved accepted targets
across the three policies; if no trace saves a target, truth is explicitly
missing too. This tool deliberately has no raw-data fallback.

## Future execution

After root has created the admitted request at the path below, run from the
repository root. The output prefix must be fresh and inside the presentation
work directory. Existing output files are never overwritten.

```sh
env MPLCONFIGDIR=work/conference_presentation_20261006/.mplconfig \
  work/venv/bin/python work/conference_presentation_20261006/render_qualitative_v1.py \
  --request work/conference_presentation_20261006/goop2d_request.json \
  --output-prefix work/conference_presentation_20261006/figures/goop2d_seed0_median
```

This writes a PDF and matching JSON receipt with input/source/PDF hashes, each
panel's exact forecast/source-frame index, state, accepted prefix, out-of-box
particle count, view, and a suggested companion caption. All particle IDs are
shown; there is no subsampling. Physical axes are fixed at metadata bounds for
all methods/times. Red crosses place every out-of-box particle at the boundary;
the exact count remains visible. The 3D view is orthographic with elevation 25
and azimuth -55 degrees, with no depth color scale. Caption text carries the
selection, schema, missingness and interpretation caveats.

Inspect the PDF-derived PNG at the final manuscript scale before integration:

```sh
/Users/aiden.zhou/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm \
  -r 120 -png -singlefile \
  work/conference_presentation_20261006/figures/goop2d_seed0_median.pdf \
  work/conference_presentation_20261006/figures/goop2d_seed0_median
```

The native standalone LaTeX compiler does not support additional project files,
so these PDFs/PNGs cannot simply be added as external image dependencies in the
same native standalone manuscript. No second LaTeX/vector backend was added.
Keep the image and companion caption as reviewable presentation artifacts until
root chooses an integration path. Do not claim native image integration or
conference readiness from successful renderer checks.

## Synthetic verification and retained predecessors

`qualitative_synthetic_checks_v1.py` provides 17 cheap contract checks, including
metadata median selection, all three study endpoints, saved-time lookup,
unknown versus zero prefixes, missing truth, row/checkpoint/objective identity,
checksum failures, and rejecting a false outside-grid claim. Re-run with:

```sh
env MPLCONFIGDIR=work/conference_presentation_20261006/.mplconfig \
  work/venv/bin/python work/conference_presentation_20261006/qualitative_synthetic_checks_v1.py
```

Final `qualitative_synthetic_v3` fixtures illustrate 2D partial failure plus
unreturned missing outcome, and 3D zero-prefix failure, alongside two out-of-box
particles. Their conspicuous SYNTHETIC CHECK titles must remain. Both final
PDF-derived PNGs were inspected; headings, axes, failure labels and legend are
legible. Root independently inspected v3 Goop3D and confirmed layout repairs.
`qualitative_synthetic_v1` and `v2` are retained: v1 had overlapping 3D header
text and clipped z labels; v2 repaired these but retained explanatory footer
text that root requested moving to the companion caption. Their source snapshots
are retained alongside the superseded fixtures. They are not research evidence.

`qualitative_renderer_authoring_record_v1.json` is the exact final file/hash
inventory and verification record. `qualitative_renderer_checks_v1.log` records
the final synthetic test run. No installs or remote calls were used.
