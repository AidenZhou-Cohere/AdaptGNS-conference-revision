# Goop3D validation split admission template

This template is inert: `status=not_admitted`, `issued_by=root_required`. Root must independently verify the current complete converted validation source and change those fields to `admitted` / `root` in a separate release file. It does not admit test access or scientific training.

The exact field names follow `evaluate_goop3d_graph_support_v1.py` (`check_split`, reviewed SHA9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de):

| Field | Source / requirement |
|---|---|
| schema | `adaptgns_goop3d_evaluation_split_admission_v1` |
| split / dataset | `valid` / `Goop-3D` |
| manifest_sha256 | Complete original valid manifest, SHA f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef |
| source_sha256 | Original complete valid TFRecord SHA5242810fa2057fbcf8e2d4da203823855f6b71e77dcfa848a618fb6871f6778b, matching the manifest and received acquisition record |
| structural_report_sha256 | Complete train/valid structural report |
| metadata_sha256 | Original metadata bytes, same as manifest descriptor |
| context_semantics_sha256 | Reviewed official parser/model context semantics |
| acquisition_report_sha256 | Complete official train/valid acquisition report |
| auxiliary_report_sha256 | Preserved auxiliary census report |
| frames / dimension | 301 / 3 |
| record_count |100, from the complete pinned valid manifest |
| particle_type_ids |[7], from structural and auxiliary/data evidence |
| converter_sha256 | Original manifest converter binding |

The execution release separately binds this admission file, all report/source bytes, exact capacity512 checkpoint, seed0 arm, physical GPU and timing purpose/mode. The evaluator itself verifies all numeric position/type arrays, auxiliary arrays, record offsets, original metadata and context-source hashes before inference and again checks input bindings at completion. This template review reads scalar manifests/reports only; it has not re-read validation numeric arrays.

No cohort/test/cross-split fields are inserted: those require a future distinct final cohort and reserved-test integrity admission. Capacity timing is infrastructure only; no checkpoint promotion or efficacy-based endpoint selection.

Review correction: the initial inert template omitted `source_sha256`; the corrected field is required by the evaluator acquisition/source equality gate. The initial template remains preserved as `goop3d_valid_timing_split_admission.template.initial_missing_source_sha.json`; neither version was admitted or executed.
