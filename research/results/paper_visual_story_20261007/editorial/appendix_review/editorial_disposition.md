# Curated appendix handoff

The source candidate is `curated_appendix.tex` (SHA-256 `47422fd72b7e05c0a107af87c0b77de238db2bbbf28ee0761d748a50edecadd9`). It is an appendix fragment, beginning with `\appendix\onecolumn`, and has no document terminator or external TeX inputs. It is intended to replace the old appendix after the official backmatter in the existing manuscript. Root owns canonical integration and compilation in the same open native source. No alternative PDF/tab was produced.

The appendix now answers five reader questions: what was run, what every policy achieved, whether completed forecasts were physically plausible and what they cost, what connects residual prediction to useful actions, and what the shorter 3D observed extension adds. It has 5 sections and 10 tables, with approximately 3,886 whitespace tokens including table cells and mathematics. The 6–10 appendix-page target is unverified until root compiles the integrated source; no page-count or visual-approval claim is made.

## What stays in the paper

| Source block | Disposition | Reason / exact destination |
|---|---|---|
| Main directed base graph, optional annulus, exact quarter budget, training mixture and cache cycle | KEEP MAIN | These define the contribution and its actionable intervention. Keep the concise version from root/sibling. |
| Main residual-versus-action distinction and simple fixed-history benefit | KEEP MAIN | This is the organizing scientific question. The formal conditional definition moves to `sec:benefit-derivations`. |
| Main NLL and faithful equation blocks (`eq:nll`, `eq:faithful`) | KEEP APPENDIX | Both align environments are copied byte-for-byte; notation, gradient separation, implementation floor/scale qualification and conditional clipping detail remain in `sec:implementation-details`. |
| Main exact edge-count decomposition (`eq:decomposition`) | KEEP APPENDIX | Byte-for-byte align environment preserved under `sec:physical-cost`; explains why changed geometry cannot establish same-state placement savings. |
| Main whole-policy outcome tables / repeated effects | KEEP MAIN as evidence atlas; KEEP APPENDIX absolute baselines | Main figure carries effect directions; appendix retains every policy and both arms without duplicating all contrast tables. |
| Existing selected Goop particle strip | KEEP MAIN | Root moves it to main Figure 3. The large appendix copy is not duplicated. Retain fixed-source, common-bounds, cross-column confounding and count-threshold caveats in the main caption/prose. |
| Goop absolute observed/H395 tables and every guard | KEEP APPENDIX | One combined absolute table retains all 12 rows; separate failure table retains all 3 guards exactly. Null means are not repaired by survivors or @200. |
| WaterDrop 110k observed/H995 tables | KEEP APPENDIX | One combined absolute table retains all 10 rows, validation/test separation and the conditional-parent training interpretation. No declared RMS policy is invented. |
| Sand observed/H314 tables | KEEP APPENDIX | One combined absolute table retains all 12 rows; prose preserves observed/autonomous divergence and native-base superiority over random. |
| Goop/WaterDrop/Sand physical-cost tables | KEEP APPENDIX | All 34 rows retained with outside fraction, excursion and one clearly scoped measured-time column. Different timing definitions and failed-group nulls remain explicit. |
| `Additional derivations`: conditional top-budget regret and residual optimum | KEEP APPENDIX | Retains `sec:allocation-score-conditions`, `sec:residual-moment`, the short proof, lag/estimation/mismatch requirements and boundary optimum qualification. |
| `% NONADDITIVE_ALLOCATION_INSERT` | KEEP APPENDIX in short form | Uniform set approximation and `2Bδ+2ε_B` proof remain. Extended mixed-difference theory, complementarity examples and abstention development move to repository. No unsupported rollout or abstention guarantee. |
| `% NOISE_AUGMENTATION_INSERT` | KEEP APPENDIX in short form | Retains adjusted-target identity and clean-calibration caveat, plus physical-unit conversion. Full Gaussian example and extended decomposition move to repository. |
| `% TIE_SYMMETRY_INSERT` | KEEP APPENDIX in short form | Deterministic ID ties and lack of permutation equivariance at ties remain; stabilizer-group argument and tie alternatives move to repository. |
| `% FULL_ACTION_APPENDIX_INSERT` | KEEP APPENDIX in short form | Retains signed-action identity, original observed-history scope, exact error/benefit correlations, adverse risk/random ordering and whole-action interpretation. Full tables/portfolios/strata move to repository. |
| `% GRAPH_BRIDGE_APPENDIX_INSERT` | KEEP APPENDIX in short form | Restored-self-message result and its scope remain under `sec:graph-bridge`; full factorial, interaction and correlation tables move to repository. |
| Earlier full WaterDrop NLL failures | KEEP APPENDIX in short form | All 8 seed-2 coordinate-resource-guard failures explicitly counted: 1 base, 2 dense, 4 random, 1 cached risk; 81 required per policy, undefined affected full-H995 means, no reruns. Full objective benchmark remains separate in repository. |
| `% GOOP3D_OBSERVED_APPENDIX_INSERT` | REPLACE with modest KEEP APPENDIX summary | Exactly 18 already reviewed scalar objects become 2 compact tables. Retains all 2,568 observed cells, 25k scope, mixed seed signs and risk still losing on average. Does not imply autonomous performance. |
| `% GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT` | ONE FUTURE MARKER | Exactly one marker remains after the D3 observed narrative. No old whole-appendix insertion or prospective result prose is included. Root may fill it only with the reviewed separate final renderer. |

## What moves out of the paper

MOVE TO REPOSITORY means preserve the exact evidence and source, not delete, hide or recompute it. CUT refers only to redundant paper prose, operational roadmaps or repeated displays.

| Original section / marker | Disposition | Reason |
|---|---|---|
| `Historical results and implementation audit`, old body lines 521–571 | MOVE TO REPOSITORY | Historical arrays and code discrepancy audit are supporting provenance, not a fourth principal experiment. Current paper no longer needs a walkthrough of historical reconstruction. |
| `Additional derivations`, old lines 572–653: time-correlation/Lipschitz derivation, calibration-bin and percentile-cost calculations | MOVE TO REPOSITORY | Valid explanatory material, but not needed once the short conditional risk–benefit proof and exact directed budget carry the argument. |
| `Compact WaterDrop study and exploratory follow-ups`, old lines 654–688, `% PILOT_RESULTS_INSERT`, `% ROLLOUT_RESULTS_INSERT` | MOVE TO REPOSITORY | Keep compact architecture, tiny inspected split, exploratory ridge head and partial rollout controls distinct from the full architecture. They distract from the three complete comparisons. |
| `Fixed original-architecture extension`, old lines 710–824, `% FULL_EVALUATION_APPENDIX_INSERT` | MOVE TO REPOSITORY, short failure/mechanism statements retained | Training timeline, 126 validation records, old full result tables and model-recovery audit are important evidence, but are not the central exposure intervention. No adverse results are discarded. |
| `% OPTIONAL_EXPOSURE_APPENDIX_INSERT` | MOVE TO REPOSITORY | Exhaustive action-decomposition and selection-support controls are subsumed by the short signed-benefit identity and new complete exposure comparisons. |
| `% NATIVE_FOLLOWUP_APPENDIX_INSERT` | MOVE TO REPOSITORY | Its full native rollout table is not needed to support the current three-study conclusions. Preserve it separately from the newer paired 110k WaterDrop endpoints. |
| `Exploratory risk--kinematics diagnostic`, old lines 825–906 | MOVE TO REPOSITORY | Strain/vorticity correlations do not resolve action allocation. Their full controls add pages without strengthening the central result. |
| `Exploratory cheap physical allocation controls`, old lines 907–1007 | MOVE TO REPOSITORY | Compact-pilot physical descriptors and controls are not the same as full-model Speed25/RMS25, whose exact definitions and complete results are retained. |
| Sand `tab:sand-all-training-effects`, `tab:sand-all-placement`, `tab:sand-clean`; Goop/WaterDrop per-seed columns and normalized duplicates | MOVE TO REPOSITORY | Main atlas and absolute appendix tables already expose the outcomes; exact per-seed/contrast/normalized values remain machine-readable. Avoid table-by-table repetition. |
| Old large inline Goop appendix particle figure (~1121–14224) | CUT duplicate from appendix; KEEP original source and main selected strip | One well-explained visual in the main text is more useful than repeated particle material. |
| Whole-graph gate capacity, timestamps, 36 admission outcomes and ridge-fit operational account (~14227–14250) | MOVE TO REPOSITORY; short no-held-out-performance caveat retained | The gate did not reach held-out evaluation. A long account would imply centrality or successful validation that the results do not support. |
| Launch times, check counts, process identities, queue state, transfer/recovery histories and future-work roadmaps inside scientific sections | CUT paper narration; preserve records in repository | These document execution integrity but do not belong in the scientific narrative. |

Exact original section/label/marker boundaries and whole-source hashes are in `source_inventory.json`; line numbers above refer to body hash `d51a03faa6b06194718b942a43f73a4b1ad1e9aafc9cd3d05229a28c322fa267`, not the integrated successor. The extraction keeps the original 10 table blocks and 3 moved equations in `source_table_snapshot.json`, so the generator remains reproducible after canonical integration.

## Cross-reference and integration contract

Preserved section labels: `sec:implementation-details`, `sec:noise-augmentation`, `sec:tie-symmetry`, `sec:allocation-score-conditions`, `sec:residual-moment`, `sec:nonadditive-allocation`, `sec:goop-graph-exposure`, `sec:waterdrop-110k-exposure`, `sec:sand-graph-exposure`, `sec:goop3d-25k-exposure`, `sec:actual-action-benefit`, `sec:graph-bridge`. Added `sec:physical-cost` and `sec:benefit-derivations`. Existing primary absolute/cost/failure table labels remain on their combined tables.

Remove or redirect any main/backmatter references to old labels: `sec:historical-audit`, `sec:compact-study`, `sec:physical-rank`, `sec:physical-allocation`, `sec:full-protocol`, `sec:full-results`, `sec:optional-exposure`, `sec:native-rollout-followup`, and removed detailed table labels. In particular, old D3 absolute/normalized/clean/timing tables are not reinstated by this fragment. The exact active D3 labels are `tab:goop3d25k-observed-position-training` and `tab:goop3d25k-observed-position-contrasts`.

Do not rerun the old builder marker registry against a retained old appendix. The curated fragment uses no old insertion marker except the one authorized future D3 autonomous marker. Root's simplified builder should consume only the retained main fragments and that separate future insertion. Official checklist and AI statement remain root-owned and outside this fragment.

## Verification and limits

`source_integrity_check.json` is PASS: 204 selected cells from 102 source table rows checked in the corresponding candidate arm/policy row; 34 absolute-policy rows and 34 physical/cost rows retained; 18 prepared D3 cells retained; the three moved align environments match byte-for-byte; braces, environments, math-delimiter parity and local labels/references pass. Failure tabular is copied verbatim from the stored source block. Numeric aggregates were not recomputed. Source generation is CPU text processing only.

`hardware_line_sources.json` binds the exact existing runtime/protocol evidence used for the new computing-infrastructure line. Hardware is descriptive context, not a claim of isolated or comparable benchmark timing. The clipping discussion is conditional because the inspected full-model protocols record no gradient clipping; the faithful gradient-separation principle is retained without claiming shared or separate clipping was applied to those runs.

The candidate has not been compiled or rendered independently, in accordance with the same-open-source restriction. Root should verify pagination, float placement, figure/table references, bibliography and visual quality in the integrated manuscript before treating it as final.

Reader correction: the Goop scope row now states 150 validation /150 test histories, matching the fixed protocol. The prior candidate and receipts are retained in `history_before_goop_scope_correction/`; `goop_scope_correction.json` binds the source and edit. This correction changes no outcome or aggregate.
