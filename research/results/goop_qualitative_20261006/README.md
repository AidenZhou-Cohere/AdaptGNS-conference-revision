# Fixed Goop2D qualitative rollout

This package adds one metadata-selected physical illustration to the completed
Goop2D study in `../goop2d_graph_exposure_100k_20261006` at commit
`26645fd180cbc54f1d008e13fc733a1643c003c9`. The parent package is unchanged.
It is not a new experiment or a selection by error. Seed 0, source 12
(`test:000012`, 1,083 particles) is the lower median of the complete 30-source
census sorted by initial particle count and source index. The full 1/200/395
forecast grid and all particles are retained.

## Interpretation and limits

Ground truth settles near the lower boundary while both mixed-training
forecasts retain elevated groups. The example illustrates substantial
remaining physical mismatch. Its base-trained/base and mixed-policy columns
change both training and evaluation graph; training-effect inference rests on
the quantitative paired comparison under the same random25 policy. It does
not estimate population performance or support a speedup claim.

Counts of strict metadata-box crossings at forecasts 1/200/395 are truth
0/0/0, base/base 0/553/465, mix/random25 0/0/11, and mix/cached-risk25 0/0/14.
These counts use any crossing, whereas the quantitative table uses excursions
greater than 1e-6. Crosses are clamped to the box and do not encode excursion
magnitude. Cached risk here is autonomous laggedrisk25, distinct from the
previous-observed same-state diagnostic. All three selected outcomes complete;
the parent package and manuscript preserve every study failure.

## Contents and validation

- `presentation/goop_fixed_source12_inline_v1.tex`: unchanged inline figure,
  SHA256 `2a5b34584dda3196f3070c5ba4eb072140e1999943edfc10ae611bd8a03ba6b4`.
- `presentation/goop_fixed_source12_inline_direct_preview_v1.png`: direct
  drawing of emitted glyphs with approximate fonts; it is not a native TeX proof.
- `presentation/integration_note_v1.tex`: exact surrounding interpretation.
- `presentation/revised_manuscript.tex`: exact integrated standalone source,
  SHA256 `81aa5098e9ddca1bb435e7980aa76bcad444ebd5424ff7f66190be7705df0a8d`.
- `presentation/*diagnostics.tex` and `*before_qualitative.tex`: exact source
  excerpts with offsets in `copy_provenance.json`.
- `canonical/`: exact original body before insertion, current canonical
  snippets and all builder inputs. The current body is reconstructed from
  the preserved body, exact note and exact figure without storing another
  duplicate of the large particle glyph block.
- `inputs/`, `verification/`, `source/`: exact selection, fetch, numerical,
  visual, integration and source receipts plus the unchanged exporter/renderer.

Every one of 12,996 glyphs was independently checked against the selected saved
arrays, including coordinates, rounding, inside/outside class and counts.
Source verification establishes exact insertion after the Goop paired-seed
figure, 66 unique labels, all 65 original labels retained, unchanged title and
abstract, and byte-identical canonical reconstruction. Root's copied native
integration receipt records compilation success; final native PDF visual
inspection was still pending when this package was made. No new native
compilation, model inference or remote action occurred during curation.

## Verify and reproduce

From the research-fork root:

```sh
python research/results/goop_qualitative_20261006/source/verify_curated_package_v1.py research/results/goop_qualitative_20261006
```

This checks the exact manifest file set, all hashes, copied-source provenance,
source excerpts and byte-identical canonical build entirely in memory. It
does not alter files or require particle arrays, NumPy, Matplotlib or LaTeX.

To re-export particle glyphs, first restore the exact omitted arrays and
per-step row JSONs listed in `source_pointers.json`, using the original
relative locations recorded in `copy_provenance.json` and the fetch manifest.
Restore the unchanged exporter, renderer, selection plan, request and input
protocols to their recorded relative locations as well. Verify every SHA256;
then run the command in `notes/INTEGRATION_README_v1.md` with a new output prefix.
The exporter requires NumPy; direct-PNG verification also requires Matplotlib.
Local versions are recorded in `runtime_versions.json`. Do not substitute
another source, seed, policy, checkpoint or error-selected frame.

Raw NPZs, large per-step row JSONs and the original tar are deliberately
omitted; their exact hashes and source pointers remain. Credentials and
process inventories are excluded. The package preserves paper evidence;
it does not imply conference readiness or author verification.
