# Optional-exposure reporting artifacts

This family contains the final v3 figure, manuscript inserts and independent reporting reviews derived from the adjacent `optional_exposure_decomposition_20261006` scientific package. It adds no model inference, training, new test population or scientific result. The underlying study uses the six original 100k checkpoints and the already inspected, no-self-loop observed-history predictions on 27 trajectories × 11 frames each. Native self-loop and 110k continuation results belong to separate studies.

The three paired comparisons are previous-observed-base-risk25 minus random25, that risk policy minus speed25, and speed25 minus random25. Positive values favor the right policy. Coordinate MSE differences/contributions and cost/alignment terms are displayed ×1,000; fractions are displayed as percentages. These are normalized acceleration coordinate quantities, with the original vector terms divided by two. Error difference equals cost difference minus alignment difference.

All four primary groups, all three both-covered degree subgroups, every comparison direction and all signed values remain in the scientific package and appendix. The primary groups partition particles by optional-edge incidence. The three degree subgroups partition only the both-covered group and must not be added again to the primary partition. Contributions use sum(mask × paired quantity)/N. Frames receive equal weight within a trajectory, trajectories equal weight within a seed, then exactly three seeds are summarized by mean and sample SD. Sample SD is not a confidence interval; the 90 figure coordinates are repeated model/condition measurements, not 90 independent samples. Empty-group conditional means remain undefined; no available-frame or available-seed substitution is introduced.

Exposure is defined by each policy. This descriptive partition does not identify a causal concentration effect, marginal edge value, independent confirmation, autonomous stability or inference speed. Message passing can affect particles with no directly incident optional edge. Adverse and null outcomes are retained in the adjacent exact scientific evidence.

## Included reporting and review evidence

- The seven original v3 files are preserved byte for byte: `optional_exposure_main.tex`, `optional_exposure_appendix.tex`, `optional_exposure_decomposition.png`, `optional_exposure_decomposition.pdf`, `plot_coordinates.json`, `mechanism_findings.json`, and `render_manifest.json`.
- `renderer_source.py` and `test_render_optional_exposure_findings.py` preserve final renderer/test source. `renderer_review_v3.json` and its independent audit script verify 92 mean±SD cells, 90 seed coordinates and six mechanism rows. The figure was visually inspected as recorded by the original review.
- `report_review_v2.json` and its audit script check 14 mean±SD cells and 24 seed sign/ordering claims on page 16 of the separate evidence-report PDF. `manuscript_preservation_review.json` and its audit script establish that both title fields, the abstract and 19 prior tables are unchanged, with only two intended insertions and three new tables.
- `document_verification.json` records successful native LaTeX compilation of manuscript SHA256 `745ea370d8a13826cf431b294514acee611d07b96be1e799502cf7184175aeef`, the eight-page main-text assertion, 47 unique labels and 12 resolved citation keys. This is distinct from the 16-page evidence-report PDF and its numeric/visual checks. The native compiler did not export a manuscript PDF; author visual inspection and registration, rights and anonymity verification remain outstanding. This package does not claim submission readiness.
- `compact_publication_verification.json` links the scientific package's byte/coverage check. `superseded_reviews/` retains earlier review records. `render_attempt_inventory.json` hashes all three original renderer attempts; v1/v2 outputs remain unchanged locally and v3 alone supplies the final presentation.
- `REPLAY_VERIFICATION.json` and `replay_renderer_audit.json` record a fresh replay from the adjacent exact gzip result. Both TeX inserts, all plot coordinates and the mechanism rows matched byte for byte. The replay independently passed the same 92-cell/90-coordinate audit. Its complete temporary output directory is preserved and inventoried. Original figures remain the reviewed v3 bytes; regenerated PNG/PDF metadata can differ.

## Reproduce the reporting

From this directory, with Python, NumPy and Matplotlib installed, choose an output directory that does not exist:

```sh
python renderer_source.py \
  --results ../optional_exposure_decomposition_20261006/results.json.gz \
  --audit ../optional_exposure_decomposition_20261006/independent_audit.json \
  --output-dir /path/to/new-reporting-directory
```

The renderer checks the independently audited full-result hash and exactly three required seed values. It refuses to overwrite an existing output directory. Both TeX inserts and numerical JSON artifacts should reproduce exactly. The original manifest identifies the plain input result; replay from gzip has the same decompressed result hash and a different input-file hash. PNG/PDF bytes may differ because of output metadata or library versions.

Run `python3 verify_reporting.py` here to verify every packaged byte and adjacent scientific-result linkage without inference. `MANIFEST.json` is written last and lists every included file, original copied-source hash and rechecked manuscript/report identities. Report/manuscript PDFs and raw particle arrays are not embedded. Review scripts preserve their original CLI contracts and require the corresponding external report/manuscript or uncompressed exact result; renderer synthetic tests run in the repository with its frozen synthetic fixtures.
