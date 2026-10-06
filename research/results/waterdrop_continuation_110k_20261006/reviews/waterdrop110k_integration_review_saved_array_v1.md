# Independent WaterDrop 110k integration review

Date: 2026-10-06  
Reviewer: `/root/code_audit/saved_array_contract_review`  
Decision: **PASS for the requested source, text, and scalar review. No unresolved numerical or scientific findings within that scope.**

## Scope and limits

This review covers the WaterDrop integration candidates in `work/deadline_research_20261005/waterdrop110k_integration_candidate_code_audit_v1`, including the main text, observed-contrast table, appendix, dated abstract, integration notes, extraction verification, claim-source map, and supporting full report and scalar mappings. It checks their claims against the previously produced, checksum-verified scientific summaries and frozen protocol copies.

This was a source/text/scalar review. It did not rerun experiments or the scientific summary/audit programs, access model checkpoints or raw arrays, edit manuscript sources, or perform Git operations. The existing audit's pass and check count below are recorded evidence, not a claim that this reviewer executed that audit. The earlier 100k compression numbers and the existing theoretical bound are inherited material and were not newly scientifically reaudited here. Root must still integrate the candidates, compile the manuscript, and inspect the resulting layout. This decision does not certify submission readiness.

## Verified sources

| Source | SHA256 | Role |
|---|---|---|
| `work/continuation-summary-20261006-v1/result.json` | `082bdc9c33614fea6445e3373e94eeafcfdfc516d4a27d04096f380ad7781e64` | WaterDrop scalar results |
| `work/continuation-summary-audit-20261006-v1/audit.json` | `caebc314c5188b7bed6776bb1adc1b060c602fc1d77bd16f93c8960302e3cbbe` | Existing independent audit: passed, 12,101,908 checks |
| `outputs/AdaptGNS/research/results/goop2d_graph_exposure_100k_20261006/paired_scalar_summary.json.gz` | `bb3c3f54cdeaf2b1198f49b8437dc711d80ce5e8347ab70363558ab515852978` | Goop compressed scalar source; compressed checksum and decompressed JSON verified |
| Frozen faithful continuation protocol | `90f896adefedf6e63503835db2479ac7e4f357043c01918b3334a846a8ffbba3` | Training lineage and exposure definition |
| Frozen strict summary protocol | `ea582e278a9bcfcd8a6eeb98f994e450dc2036352debfeb74a3c512602338454` | Cohorts, estimands, aggregation, and failure/null rules |

The two numbered protocol copies in `cohort_and_protocol_identity.json` have contiguous line numbering. Reconstructing each text with its terminal newline reproduces its original SHA256 exactly.

## Numerical and extraction verification

- All **133 claim objects** exactly match their referenced source JSON paths.
- All **2,002 statistic objects** exactly match the WaterDrop result, with complete and unique coverage. These comprise **91 metric families**, each with 10 absolute objects and 12 paired objects.
- All detail-file path and hash references match the source result.
- All **290 numerical table entries across 35 rows** were checked, including units/scales, rounding, means, sample standard deviations, seed columns, and ordering: **zero mismatches**.
- Prose numerical roundings and seed-sign statements agree with the verified sources.
- Coverage, scientific failure semantics, required null propagation, and conditional training lineage are represented consistently.

The candidate distinguishes same-policy exposure `D_p = E_mix,p - E_base,p`, within-arm placement `A_a = E_a,risk - E_a,random`, and their interaction `I = A_mix - A_base = D_risk - D_random`. Differences are formed on matched units before equal-trajectory and equal-seed aggregation. Three-seed dispersion is sample SD (`ddof=1`), not uncertainty from treating all histories as independent replicates. Required nulls propagate; survivor means are not substituted.

## Scientific interpretation checks

The WaterDrop study is correctly described as six 110k endpoints obtained from three faithful 100k parents by paired base/mix 10k continuations, with inherited Adam moments/counters, matched frame/noise schedules, and learning rate `1e-5`. Mixed exposure independently selects per example with probability 0.5; on selected examples it samples exactly `floor(0.25 * annulus_count)` unordered pairs uniformly and appends both directions after the unchanged native capped/self prefix. This is distinct from Goop's training from initialization for 100k.

The reported WaterDrop coverage is internally consistent: 12 jobs; 128 validation and 297 test observed histories per endpoint; 2,550 histories and 12,750 policy slots; and 810 autonomous H995 outcomes from 27 test sources (3–29), five policies, and six endpoints. All WaterDrop outcomes are complete, with all scientific failure counts zero. The candidates retain the exploratory, post-test-inspection qualification and do not treat these continuations as independent confirmation, evidence of fresh-training generality, or NLL evidence.

The principal interpretation agrees with the scalars:

- WaterDrop observed-test risk-minus-random position MSE reverses from positive in each base-arm seed to negative in each mix-arm seed. The across-seed means are `+1.6158928377245146e-10` and `-9.926710627323604e-11`; the interaction is `-2.608563900456876e-10` with sample SD `7.410205477816127e-11`.
- WaterDrop validation does not establish the same every-seed mixed-arm advantage: its mix-arm risk-minus-random signs are negative/positive/negative. The candidate's observed-test qualification is therefore material.
- The autonomous random-policy exposure mean is `-0.0059993461496299924`, with all three seeds negative. Dense, speed, and cached-risk exposure also improve in every seed. The autonomous placement interaction has mean `+0.002843511449577538` and mixed seed signs; it does not establish a general autonomous risk-placement benefit.
- Dense remains worse than sparse base within every WaterDrop seed and arm, while speed has the lowest across-seed autonomous mean in both arms. Observed preceding-base risk and autonomous cached risk are distinct controllers.
- Risk score correlation with base residual is positive, but correlation with actual risk-action gain is close to zero. A base-action gain rank correlation is null by construction because its gain is identically zero.
- Boundary excursions remain substantially worse than truth, and all recorded autonomous runtimes and edge counts increase under mix in each paired seed. The candidates do not support a conservation, physical-validity, standalone-latency, or causal-speedup claim. Cached risk uses 995 forecast passes plus one initialization pass, excluding the separate native parity verification; observed risk requires a separate scoring pass for each history.
- The Goop comparison preserves its three candidate-pair resource guards and required nulls: mixed cached-risk seed 2 has null full-horizon MSE and the corresponding interaction is null. Its random H395 exposure improves in every seed, but this does not imply a complete cached-risk comparison.

## Resolved wording issue

An earlier supporting report said Goop observed risk was worse than random in “every mixed-model seed” without restricting the claim to test histories. That statement holds on observed test histories, while Goop mixed validation seed 0 has negative risk-minus-random error.

The final dated abstract now says **“matched Goop test histories”** and the final full report says **“Goop observed-test risk.”** Both phrases were checked in the final files, and their hashes match the versions below. The issue is **resolved**, with no remaining requested correction. The original versions and `observed_test_qualifier_receipt.json` remain available in the candidate directory.

The final main text explicitly says “placement on observed test histories.” The three appendix tables were changed to `table*` with default scriptsize spacing, and the final operational/audit-count appendix paragraph was removed while retaining that evidence in the supporting full report. These editorial changes introduced no numerical changes. Their rendered layout remains for root to inspect.

## Final reviewed file identities

All 11 SHA256 values below were reconfirmed against the final files when closing this review. File paths in this table are relative to `work/deadline_research_20261005/waterdrop110k_integration_candidate_code_audit_v1`.

| File | SHA256 |
|---|---|
| `candidate_main_waterdrop.tex` | `220a2465f339b546319cf57453c3463ea924543d3325f542b8885a98d8b3c7b3` |
| `candidate_observed_contrast_table.tex` | `de5af5c765c3ed4502b532575c3dd661ce81059545ec2053e5ccf2b870e82a4e` |
| `candidate_waterdrop_appendix.tex` | `957a47512ea8c5810f648b9058ef86e90078dd80412b38715291e29e244bebe0` |
| `candidate_abstract_dated_20261006.md` | `89271df7ceaa80f94af7413727462db61a73e9b28ec1d776e702e14735af2992` |
| `integration_notes.md` | `a2746249d560e0592b60c3909ed292fc3dafec6c91c03ac4c3a650ca08cf9203` |
| `extraction_verification.json` | `df6b0752a6e9477747b4165ab9314aecc57ad5f2d8b0461c9fe546994851534d` |
| `claim_source_map.json` | `f9101fb07ca9ecb3ec99343c51265518d4ad174cc59f990684f34cac77dee5b4` |
| `cohort_and_protocol_identity.json` | `ac506ebf0f850ca5d36b4e9d4e37c4517e9d32923caf8ec568dc874b61b29afe` |
| `waterdrop_all_scalar_mappings.json` | `e19e1641eb91b65b9d413fb460f970543d9c36a686d4e8788ae115327343e9e6` |
| `waterdrop110k_full_report.md` | `ab5706b7dd15fe63666daf751e7b0a8f4cbd20fb2b56948c20ae5ac0a514984d` |
| `candidate_compressed_existing_waterdrop.tex` | `bee907ef5bb8292667ec00c679dc86397d5f9f17478736faeb38debf509fcfc5` |

No unresolved numerical or scientific findings remain in the reviewed candidates. The remaining integration, compilation, and visual inspection work belongs to root and is outside this review's pass decision.
