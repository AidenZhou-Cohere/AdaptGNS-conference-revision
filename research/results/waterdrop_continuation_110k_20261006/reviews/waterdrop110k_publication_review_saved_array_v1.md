# WaterDrop 110k independent publication-copy review

Date: 2026-10-06  
Reviewer: `/root/code_audit/saved_array_contract_review`  
Decision: **PASS for the reviewed staged package and planned publication-copy operation. No unresolved inventory, copy-integrity, scope, or source-binding findings.**

## Reviewed stage and limits

Stage: `work/deadline_research_20261005/waterdrop110k_publication_staging_code_audit_v1`  
Manifest SHA256: `76b03c5b726b0bbba7c7845845454a5e987f4ffa4e7b4f6637cf2c58adfa4cb9`  
Revised plan SHA256: `56b5af45664030376707bb4990ed71b09910a120673c5e9abac421bf05b8a5e7`

This review independently checks copies, hashes, inventory, retained scalar/coverage structure, publication qualifications, source/compile binding, and the curator's publication source. It did not execute the curator, a scientific stage, tests, a builder, or a compiler; inspect raw checkpoints, datasets, prediction arrays, or traces; decompress or reconstruct the combined archive; edit staged/manuscript/scientific files; inspect native processes; perform Git operations; or upload anything. No archive was decompressed during this copy review. The only new file written is this review note.

This PASS applies to the stated stage before the review is added and publication metadata is updated. It is not evidence that promotion, commit, push, external upload, or PDF visual inspection has occurred. The original scalar, candidate, and applied-text reviews remain the scientific and manuscript comparison baseline; this copy review does not expand their scope.

## Exact inventory and copies

The stage has **162 manifest-bound files totaling 19,796,372 bytes**, excluding `manifest.json` itself. The filesystem inventory is exactly the manifest keys plus that manifest, with no missing files, unexpected files, or symlinks. Every manifest size and SHA256 matches the actual staged file.

All **153 fixed copies totaling 19,583,426 bytes** match the revised plan in destination, size, and SHA256, and every corresponding source original independently matches those same size/hash requirements. Destinations are unique. The copied revised plan, historical plan, and curator source match their originals. The remaining entries are exactly the planned generated metadata. `copy_provenance.json` binds the correct revised plan and reproduces its complete fixed-file mapping exactly.

Compared with the preserved first plan, the revised plan removes only `summary/audited_records.json.gz` from the scientific payload, adds the completed `reviews/waterdrop110k_applied_text_review_saved_array_v1.md`, and changes none of the other fixed-copy entries. The README and plan identify the first plan as historical and superseded. No LFS pointer, split archive, release-upload substitute, or hidden copy of the omitted combined archive is present.

## Scientific evidence retention

`summary/` contains exactly the original `result.json`, original `status.json`, and all **91** metric detail archives referenced by the result. Each compressed metric archive's hash matches both its original and the unchanged result reference. No metric archive was recompressed or regenerated.

The result retains 30 observed-validation metric families, 30 observed-test families, and 31 autonomous-test families. Each retains both training arms, all five controller identities, all three seed values, 10 absolute statistic objects, and 12 paired statistic objects: **2,002 statistic objects** in total. The 10 undefined aggregate objects remain present with null values; publication does not replace them with survivor means. Preserving each original metric archive also preserves its detailed unit, trajectory, seed, and null-reason records. Inspection of the frozen publishing source confirms that these detail archives contain per-metric policy values and null reasons, distinct from the combined derived-record payload.

Coverage and job identities remain complete:

- Six endpoints and 12 unique jobs: both arms, seeds 0–2, and observed/autonomous modes.
- 128 validation histories and 297 test histories per endpoint; 2,550 observed histories and 12,750 policy slots overall.
- 27 autonomous test units per endpoint, five policies, and 810 H995 outcomes overall.
- Every recorded scientific job failure count is zero.

The status is complete and binds the exact packaged result hash. The original complementary independent audit is preserved with `passed: true` and **12,101,908 checks**. Its source-summary hash matches the packaged result, and its audit-source hash matches the copied independent-audit program. This records the earlier audit result; no audit was rerun here.

The complete report, all-scalar mappings, claim-source mappings, cohort/protocol identities, manuscript candidates, and three earlier independent reviews are exact copies. The report's comparative Goop source is explicitly external in the neighboring published result directory; its compressed SHA256 was independently rechecked as `bb3c3f54cdeaf2b1198f49b8437dc711d80ce5e8347ab70363558ab515852978`.

## Omitted combined archive and reproduction limits

The local combined archive remains a regular file, with its original compressed size and SHA256 independently verified:

| Property | Verified or recorded value |
|---|---|
| Original project-relative path | `work/continuation-summary-20261006-v1/audited_records.json.gz` |
| Compressed bytes | `368626721` |
| Compressed SHA256, independently rehashed here | `8acc8933e0b4a6ef98fc98f16b81739895968fcbd0df3cf796118c982fcea539` |
| Uncompressed JSON SHA256, preserved from the completed strict summary | `8f9bb7ec9b0483529d9d5c44f7a66c583044a44a61927f38e47408bd06df90da` |
| Missing package-relative reference | `summary/audited_records.json.gz` |
| Reference owner | `summary/result.json::audited_records` |

Both hashes in `external_artifacts.json` match the unchanged result reference, and its source path/size match the revised plan. The uncompressed hash was **not newly recomputed**: it remains explicitly identified as recorded strict-summary evidence. The archive was not decompressed, reconstructed, moved, rewritten, or copied into the stage.

The omission is uniform over the combined derived records for every cohort, policy, and seed. The README, omissions metadata, and external-artifact manifest explicitly deny a standalone raw reproduction bundle. They distinguish restoring the existing exact archive to the recorded summary location from reconstructing it using the separately required frozen sources, dependency environment, checkpoints/data, and evaluation inputs. All **10,736 original input-path/hash references** remain unchanged in the result, including references to external raw inputs. The package does not pretend those referenced payloads are present.

## Source, manuscript, and compilation binding

The 34 copied scientific source/protocol files under `source/workspace/` all match the frozen identities in `summary/result.json::input_files_sha256`. The additional independent audit and extraction source are also plan-bound exact copies. Their presence is explicitly qualified: workspace-relative or absolute dependencies remain, and the source directory is not advertised as a self-contained executable raw experiment.

The packaged standalone manuscript, body, experiments source, builder, pre-integration snapshots, and integration receipts match the reviewed versions. The native receipt records success for the original output path and SHA256 `4cd942d9ba1f054b513d57e652eade88f13df63fdb760d81d1dfb25a91daee98`; both the packaged standalone source and the current original have that digest. The receipt retains the eight-page main-text assertion result. The applied-text review's preservation of title, abstract, 66 prior labels, four new labels, and old appendix bytes therefore remains source-bound. Exported PDF visual inspection remains outside these checks and is not claimed complete.

## README/report interpretation and payload restrictions

The README and exact reviewed report retain the necessary distinctions: continuation conditional on three inspected 100k parents versus fresh Goop training; observed test versus validation; previous-observed versus autonomous cached risk; beneficial fixed-random exposure versus inconsistent autonomous placement interaction; complete computational outcomes versus physical validity; and descriptive recorded cost versus causal speedup. The WaterDrop observed-test sign reversal is not generalized to validation or to consistent autonomous allocation. The positive mean autonomous interaction, mixed seed signs, box excursions, increased recorded durations/edge counts, and required null behavior remain visible. The report preserves Goop's guard failures and undefined interaction.

The staged inventory contains no checkpoint, dataset, raw prediction/trace-array, credential-directory, SSH-configuration, or native-process-inventory payload. All non-gzip files decode as UTF-8. A focused text scan found no private-key block or high-confidence AWS/GitHub/OpenAI credential signature, and recursive JSON-key inspection found no native PID, process command-line, environment-variable, or credential-value fields. Path/hash references to omitted inputs remain scientific provenance and are not their payloads. The semantic operational failure history is retained without native-process snapshots. These checks support the curated payload restrictions; they are not a general-purpose secrets audit of the entire project.

## Curator publication source

Curator: `work/deadline_research_20261005/curate_waterdrop110k_publication_code_audit_v1.py`  
SHA256: `4414bb53cbada13cb13bb69ef50f45ce4b3309110db778b57ce9818f3bb2a85e`

For the reviewed stage, its publication phase checks that target package/report paths do not already exist, checks the supplied review hash, verifies the existing stage manifest, revalidates package coverage and source bindings, and rehashes all original fixed sources plus the retained combined archive. It then adds the supplied review and preserves exact pre-review copies of both the manifest and publication-verification metadata under `curation/` before updating the live metadata. It copies the exact reviewed report, renames this stage into the output package, and verifies the resulting manifest and external report hash. It does not alter scientific contents or source originals. No scientific import/execution, shell/Git operation, network upload, LFS operation, archive split, or archive reconstruction occurs in this phase.

The expected post-review metadata changes and stage promotion do not invalidate this review's fixed-payload decision; the parent must retain the initial manifest and bind this note's returned hash in the final manifest, as the reviewed publication code does. This review does not itself claim that those future operations succeeded.

## Key reviewed hashes

Paths below are relative to the reviewed stage unless identified otherwise.

| File | SHA256 |
|---|---|
| Revised copy plan (original and `curation/` copy) | `56b5af45664030376707bb4990ed71b09910a120673c5e9abac421bf05b8a5e7` |
| Historical first copy plan (original and `curation/` copy) | `bde73c684a608ab7c5c2cb6baa7947d9e9fc81813fb1be427a59c6171b7f217a` |
| `manifest.json` | `76b03c5b726b0bbba7c7845845454a5e987f4ffa4e7b4f6637cf2c58adfa4cb9` |
| `copy_provenance.json` | `91b8c4870bef10143125d907b859fe8d0710158897d7dd549dbf0b74bd0b8145` |
| `publication_verification.json` | `c202d6ac1c7e0d177c96ef9a5ea62df62b66a9f5451e0f7eb3d57678c2053b1d` |
| `external_artifacts.json` | `e7fa8bfab7983d357923eee4df65889db4a58b062bc6360541cfbde4a5baa4f2` |
| `omissions.json` | `82096fc9a89dff748cc0307b1526b710d23eaea7f9aa3a3fe23bd9ee2290ed2b` |
| `operational_failure_history.json` | `e000cc68a952192979c848cf7003458ee8fd02a51b67ec6943b8c3ec8f6797a5` |
| `README.md` | `fa21dcfc68903ea339ea20ac7a0b0a0a6cb485fe533aea6af59cf129c1aafc2e` |
| `report.md` | `ab5706b7dd15fe63666daf751e7b0a8f4cbd20fb2b56948c20ae5ac0a514984d` |
| `summary/result.json` | `082bdc9c33614fea6445e3373e94eeafcfdfc516d4a27d04096f380ad7781e64` |
| `summary/status.json` | `ba3f93254a492fe65685cf69f87f327652810d1e99b000f0a346133bc1edf2fb` |
| `audit/audit.json` | `caebc314c5188b7bed6776bb1adc1b060c602fc1d77bd16f93c8960302e3cbbe` |
| `presentation/revised_manuscript.tex` | `4cd942d9ba1f054b513d57e652eade88f13df63fdb760d81d1dfb25a91daee98` |
| `presentation/integration/native_compile_v1.json` | `56695402e760079fda8be912352efe629e96da31899a32a0312501651b979545` |

No correction is required within this bounded copy/publication review.
