# WaterDrop 110k applied-text review

Date: 2026-10-06  
Reviewer: `/root/code_audit/saved_array_contract_review`  
Decision: **PASS. No unresolved applied-text, numerical-transcription, label, or preservation findings in the reviewed source versions.**

## Scope

This is an independent, bounded comparison of the applied manuscript against the frozen reviewed WaterDrop candidates and the pre-integration snapshots. It also reviews the changed introduction, conclusion, and limitations for scientific scope and checks whether the recorded native compilation success is bound to the current source. It does not rerun experiments, scientific summaries, audits, tests, the manuscript builder, or the compiler; access raw models/arrays; edit manuscript/scientific files; perform Git operations; or inspect an exported PDF.

The scalar and claim review is recorded separately in `work/deadline_research_20261005/waterdrop110k_integration_review_saved_array_v1.md`, SHA256 `4389c5ee4ad06d5190706ec2a56a34bd9350121acb9b62f69d8f956838b6f921`. Its verified candidates are the comparison baseline here. The earlier 100k diagnostics and theoretical bound remain inherited material, not newly scientifically revalidated by this applied-text review.

## Applied candidates and generated-source consistency

The reviewed observed-contrast table, compressed earlier-WaterDrop paragraph, and complete new WaterDrop appendix each occur **byte-for-byte exactly once** in their intended current source and exactly once in `outputs/revised_manuscript.tex`. Consequently, the previously reviewed 290 numerical table entries are carried into the generated manuscript unchanged, including signs, units, scales, sample SDs, and seed columns.

The new WaterDrop main paragraph is exact after one documented editorial replacement: `Across five evaluation policies, mixed training` becomes `Mixed training`. Removing that modifier avoids suggesting that the risk-minus-random contrast averages five policies. No numbers or scientific qualifications change.

The complete generated document body equals an independently reconstructed, in-memory expansion of the current body and its source insertion markers. The document boundary was matched as the actual line `\begin{document}`, rather than occurrences quoted inside the embedded style's diagnostic strings. This comparison did not execute or write with the builder. The builder and the generated preamble are unchanged from their snapshots.

All **25 structural/text assertions passed**, covering candidate insertion, the editorial replacement, source expansion, preamble preservation, title/abstract preservation, appendix preservation, labels/references, and compile-receipt binding.

## Preservation and labels

- All **66 pre-existing labels** retain their exact multiplicities in the generated source. The current **70 labels are unique**. The four additions are `sec:waterdrop-110k-exposure`, `tab:waterdrop110k-observed`, `tab:waterdrop110k-rollout`, and `tab:waterdrop110k-cost`.
- The body source similarly changes from 25 to 29 unique labels, with the same four additions and every old label preserved.
- All **29 distinct literal reference targets** found through `ref`, `eqref`, `pageref`, and `autoref` resolve to labels in the current generated source.
- The old generated appendix, from `\appendix` through the bytes immediately before `\end{document}`, is preserved as an exact prefix of the new appendix: **592,619 bytes**, SHA256 `c2758407df074529f422b91fb7004759dc43e5f9fed9782645383755f329a8a9`.
- The corresponding old body-source appendix is also preserved exactly: **541,418 bytes**, SHA256 `9b40891760a73c4f4f65d1ddd7e62e0007615ed211c4647a70cf0910236b0f54`.
- The full prefix before the introduction, including title and abstract, is unchanged in both the body source and generated manuscript. The dated abstract candidate remains separate optional author-review material; it was not silently substituted into the manuscript.

## Scientific scope of the changed prose

The introduction identifies fresh Goop models and a separate WaterDrop study continuing three faithful 100k parents for 10k additional updates. Its Goop placement claim is restricted to matched **test** histories, and the WaterDrop reversal is explicitly **observed-test**. It does not extend the every-seed mixed-arm placement advantage to validation.

The experiments paragraph states that WaterDrop exposure effects are conditional on the parents and distinct from Goop training from initialization. It retains the positive mixed-arm validation risk-minus-random average, all 810 completed H995 outcomes, the favorable random-policy exposure effect, the positive mean autonomous interaction with mixed seed signs, substantial box excursions, and increased recorded rollout times. The appendix retains the validation seed signs, matched-unit aggregation, previous-observed versus cached-autonomous controller distinction, inherited optimizer and schedule details, inspected-test qualification, and fixed 110k endpoints.

The updated limitations explicitly describe inspected WaterDrop parents and distinct training budgets, retain the three-seed limitation, and deny generalization across architectures, budgets, or materials. The main continuation paragraph and appendix supply the explicit conditional-lineage restriction. Goop remains exploratory after the earlier WaterDrop analyses. Neither study is presented as independent confirmation or as proving fresh-training WaterDrop generality.

The conclusion accurately separates three claims: both studies improve the fixed random policy's full rollouts in every paired seed; Goop risk remains worse than random on observed test histories with its autonomous interaction undefined after required failures; and WaterDrop's observed-test reversal coexists with an autonomous placement interaction that changes sign across seeds. It states that these conditional gains do not establish a general allocation or efficiency advantage.

Failure, physical, and cost caveats remain visible in the main text and appendix. Goop's required failures and undefined full-horizon means/interactions are retained rather than replaced with survivor means. WaterDrop's complete outcomes do not erase its boundary excursions. The prose distinguishes measured descriptive cost from causal speedup, keeps the separate scoring-pass conventions, and retains the limitations concerning conservation, peak memory, and optimized runtime. No unsupported positive claim was introduced by the integration.

## Native compilation and page-cap evidence

The recorded native compilation receipt reports `kind: success`, identifies `mcp__codex_app__compile_latex_document`, names the exact current output path, and binds the result to SHA256 `4cd942d9ba1f054b513d57e652eade88f13df63fdb760d81d1dfb25a91daee98`. That digest equals the source reviewed here. The receipt also records that the eight-page main-text assertion passed.

The generated source retains the unchanged assertion immediately after flushing main text with `\clearpage`: if the next page counter exceeds 9, it raises `\PackageError` for exceeding the eight-page main-text limit. Thus the success receipt and unchanged assertion support the reported **within-eight-page-cap** result. This review does not independently compile, measure an exact page count, or certify rendered layout. The receipt itself records `exported_pdf_visual_inspection: pending`; visual verification remains outstanding and is not covered by this PASS.

## Reviewed identities

Current source identities match the parent's handoff; no source-hash drift was observed.

| Current file | Bytes | SHA256 |
|---|---:|---|
| `outputs/revised_manuscript.tex` | 657129 | `4cd942d9ba1f054b513d57e652eade88f13df63fdb760d81d1dfb25a91daee98` |
| `work/manuscript_body.tex` | 578692 | `17201631acea97246d6ef37dc168f45bb835201dcd980452a4db79540ada0b0e` |
| `work/conference_experiments_main.tex` | 7990 | `05cc45b758c27627c9e2adc651c1cea4fd327758d852f09f7a55d0e246919fb4` |
| `work/build_manuscript.py` | 2884 | `aa023359da96c95bfebc41ef638e1b76bd7f7c49ae3ff280d581c76cda658134` |

Snapshots and receipts below are in `work/conference_presentation_20261006/waterdrop110k_integration_20261006`.

| File | SHA256 |
|---|---|
| `revised_manuscript.tex.before` | `5c5518c6cc7fa927f9b056002332853c66da0fcfbab71f2acb985f5cd9d1b93e` |
| `manuscript_body.tex.before` | `2d98e7ef62d57cf17b8a7c62ef30d041396d300296640929a37c21566097ae9b` |
| `conference_experiments_main.tex.before` | `830aa19f3307c42ef8a892f7c66936d03638685351c736690f6367b8053a09a5` |
| `build_manuscript.py.before` | `aa023359da96c95bfebc41ef638e1b76bd7f7c49ae3ff280d581c76cda658134` |
| `applied_source_v1.json` | `62fcf819978b24405d3e1079581a61e696acea6cbaef1a469af6fd5a31eb9066` |
| `applied_structural_check_v1.json` | `11b5e9d3adb8e96dd907d9ce6fda435b474e68cb6e596d85366d49463f51991b` |
| `native_compile_v1.json` | `56695402e760079fda8be912352efe629e96da31899a32a0312501651b979545` |

Frozen candidates below are in `work/deadline_research_20261005/waterdrop110k_integration_candidate_code_audit_v1`.

| File | SHA256 |
|---|---|
| `candidate_main_waterdrop.tex` | `220a2465f339b546319cf57453c3463ea924543d3325f542b8885a98d8b3c7b3` |
| `candidate_observed_contrast_table.tex` | `de5af5c765c3ed4502b532575c3dd661ce81059545ec2053e5ccf2b870e82a4e` |
| `candidate_compressed_existing_waterdrop.tex` | `bee907ef5bb8292667ec00c679dc86397d5f9f17478736faeb38debf509fcfc5` |
| `candidate_waterdrop_appendix.tex` | `957a47512ea8c5810f648b9058ef86e90078dd80412b38715291e29e244bebe0` |
| `candidate_abstract_dated_20261006.md` | `89271df7ceaa80f94af7413727462db61a73e9b28ec1d776e702e14735af2992` |

No corrections are requested within this bounded review. PDF visual inspection remains a separate required step before layout or submission-readiness claims.
