# Reserved Goop 2D test preparation: root execution sequence

`prepare_goop_reserved_test_v1.py` is a separate adapter. It never imports a
model or runs inference. Preparation testing used generated TFRecords and mock
HTTP only; no official test HEAD, GET, payload, checksum or example was accessed.
The unknown test generation, length and CRC are deliberately **not invented**.
The adapter reuses exact reviewed `prepare_goop_official.py` conversion functions
and the original pinned TFRecord reader, without changing either source.

Freeze this adapter only after independent review. Every actual mode requires
`--execute`, `--mode`, `--cohort`, `--cohort-audit`, `--root-release`. The gate
checks all six distinct faithful base/mix seed0/1/2 endpoints at100k, complete
model/Adam/schedule verification, original cohort/audit hashes and protocol-v2
hash before reading any test-side file or making even a HEAD request. Root must
have completed the fresh host byte inventories and complete-cohort review first.
The adapter's gate closes at the compute/analysis cutoff, October7 01:00 UTC.

## 1. Obtain and freeze official source metadata after cohort freeze

Root creates a fresh JSON release with:

- `schema`: `adaptgns_goop_reserved_test_preparation_release_v1`
- `status`: `approved_for_source_metadata`; `issued_by`: `root`
- `issued_utc`: aware UTC time after `cohort.created_utc`
- `cohort_sha256`, `cohort_audit_sha256`: exact completed-cohort files
- `preparation_source_sha256`: independently reviewed adapter hash
- `source_url`: `https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop/test.tfrecord`
- `all_six_checkpoint_hashes_verified`: `true`
- `test_converter_independently_reviewed`: `true`
- `review_rationale`: root's explicit review conclusion.

Run mode `inspect` with those common flags and fresh `--output` for the header
receipt. Only HEAD is sent, with identity encoding and redirects refused. The
receipt preserves publisher headers, numeric generation, content length and
CRC32C. An error stays in the saved receipt. No SHA256 is claimed before bytes
are received.

Root reviews the header receipt and writes a **new** release preserving the
same bindings but setting `status: approved_for_acquisition_preparation` and
`source_metadata_sha256` to that receipt's exact hash. All later modes also take
`--source-metadata` with that receipt. Thus GET is bound to a specific publisher
generation, length and CRC before download. Keep both releases and the header
receipt.

## 2. Acquire source bytes

Mode `acquire` additionally takes `--metadata` (the already frozen362-byte Goop
metadata.json) and fresh `--output-dir`. It copies metadata with an explicit
local-copy label and GETs only test.tfrecord at the pinned generation. Response
generation/length/CRC must match the admitted header receipt. It computes full
received SHA256 and CRC32C, fsyncs before publication, and retains partial bytes
plus their actual hashes on an interrupted/failed stream. No resume/retry or
source-generation substitution occurs automatically. Root must inspect any
failure; the existing files remain untouched.

Outputs: test.tfrecord, metadata.json, acquisition_report.json. Preserve all
source bytes and the receipt, including unsuccessful outcomes.

## 3. Convert all records without changing the source contract

Mode `convert` additionally takes acquired `--input-dir`, its
`--acquisition-report`, original pinned `--reader`, and a separate fresh
`--output-dir`. Reader path is the original research `prepare_full_waterdrop.py`
with SHA ab077f11…d2f33. The pinned Goop converter's `convert_one` performs whole
object CRC/SHA checks, every TFRecord length/payload CRC, full source offsets,
exact array element-byte comparison, duplicate detection and preserved staging.

All30 test records must haveT401, type7, source keys0..29 and context shape401x1,
as required by the already frozen final evaluator. Any actual discrepancy is
saved as a failed structural report with staging/source retained. It is never
fixed, silently converted, subsetted or used to tune the protocol. Successful
outputs are test.json, test/arrays, metadata.json and structural_report.json.
The manifest converter SHA is this separate adapter; the structural report also
records its unchanged helper SHA. Its schema deliberately remains
`official_goop_numeric_preparation_v1`, which the frozen test contract accepts.

## 4. Census and complete train/valid/test overlap audit

Mode `census` additionally takes `--numeric-root` from conversion,
`--train-manifest` and `--valid-manifest` at their actual array roots,
`--acquisition-report`, and a new evidence `--output-dir`.
The training/validation manifest hashes must match the frozen original pins.
It verifies position/type file hashes and actual dtype/shape/element bytes for
all1000/30/30 records. Duplicate identity is independent of NPY headers and
source IDs. All within/across-split duplicates are retained in the report and
prevent admission. It also hashes and describes every preserved test auxiliary,
including exact float32 bit patterns. The frozen evaluator expects400positive
zeros and one canonicalNaN per401x1 array; deviations are retained for review.
Omission semantics still come from the official parser's absent context_mean,
not those values.

Outputs: auxiliary_report.json, cross_split_audit.json,
preparation_status.json and, only on all checks passing,
final_test_admission.candidate.json. The candidate is explicitly
`prepared_requires_root_review`; it cannot pass the final evaluator. The
adapter temporarily checks its structural fields in memory against the exact
frozen split validator but never publishes an admitted file or starts inference.

## 5. Root admission and pure final preflight

Root reviews all source/conversion/census/overlap receipts and creates a separate
final test admission from the candidate, changing status to
`admitted_for_final_evaluation` only after verification. Keep the candidate.
The hashes already bind the cohort, protocol, training admission and every test
artifact. The cohort/audit and both valid/test final admissions must be identical
across evaluation hosts.

Mode `preflight` additionally takes `--numeric-root`, root `--admission`,
`--acquisition-report`, original `--context-semantics` (with the sibling
`goop_context_semantics_sources` directory), `--auxiliary-report`,
`--cross-split-audit`, and fresh `--output`. It invokes the unchanged frozen
`goop_evaluation_contract.validate_split` and `verify_evidence`, including
auxiliary bytes, four official parser/model sources and complete overlap pins.
It does no inference. The locked evaluator independently repeats its own cohort,
checkpoint and split checks at each invocation.

Prepare the corresponding final validation admission using the already reviewed
original validation/source/census pins plus this completed cohort hash and the
protocol-v2 hash. No validation reconversion or policy changes are needed.
Then root can release the fixed quota-v2 four-stage evaluation queue. All1080
planned rollout outcomes and all missing/guarded results retain their declared
accounting, with no claim of pristine independent confirmation.

## Preparation tests

17 generated-data/mock-HTTP CPU tests passed in4.63s. These include strict
pre-access gates, HEAD-only metadata, generation-pinned GET, partial-byte/hash
preservation, changed-header rejection, full30 generated-record conversion,
noncanonicalNaN preservation, changed numeric bytes, no partial-source promotion,
all1000/30/30 synthetic overlap accounting and the actual frozen split/evidence
preflight against synthetic test arrays. No actual publisher request ran.
