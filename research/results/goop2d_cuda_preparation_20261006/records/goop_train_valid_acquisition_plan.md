# Official Goop 2D train/validation preparation

Status: preparation only. The methods agent has opened no Goop TFRecord payload and has launched no scientific model or acquisition. The root agent owns acquisition, observed-data admission, the scientific protocol and any later training. This plan does not authorize test access or reuse the Sand NPZ admission contract.

The official source base is `https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop/`. Read-only HEAD responses and the 362-byte metadata are saved in `../extension_feasibility_20261006/public_metadata_inventory.json` and `../extension_feasibility_20261006/Goop_metadata_json.json`. Preserve that evidence.

| File | Complete bytes | GCS generation | CRC32C, base64 |
| --- | ---: | --- | --- |
| metadata.json | 362 | 1599153763201175 | 9pwq+g== |
| train.tfrecord | 3358535518 | 1599154403717337 | OrpMIQ== |
| valid.tfrecord | 93682221 | 1599153784596039 | EqHuFw== |

Metadata SHA256 is `565d6e13be91be6a6b0fbc31a9aa19ed70a5505228411d0928288fcf874ca3dd`. It declares 2D positions, sequence length 400, radius 0.015, dt 0.0025 and bounds 0.1–0.9. The official parser expects sequence length plus one frames; 401 is an expectation to check against the complete payload, not an observed fact. No trajectory count is assumed.

## Root-owned acquisition

1. Use a fresh acquisition directory and a unique attempt record. Fetch only the three exact names above. Pin each GET to the observed GCS generation, verify response identity and retain response headers. Do not use the generic original dataset downloader: it also fetches test data.
2. Stream to fresh partial files, count complete received bytes, compute SHA256 and CRC32C, then compare the generation, complete length and CRC32C with the table. Retain partial bytes, response/error details and failed checksum attempts. Do not silently resume into or replace a previous unsuccessful attempt.
3. Publish a completed acquisition receipt only after every selected object has passed. The receipt uses schema `official_goop_train_valid_acquisition_v1`, status `complete`, dataset `Goop`, source_family `official_gns_tfrecord`, and `files` records with these exact fields: `name`, `saved_name`, `url`, `status`, `received_bytes`, `generation`, `crc32c_base64`, `crc32c_verified`, `sha256`. Both names are the exact basename, the URL is the base above plus the name, status is `complete`, and CRC verification is boolean true. Each SHA256 must be computed from the complete acquired object. No executable sample receipt or placeholder hashes are supplied here.

The train and validation files total 3,452,217,739 bytes before metadata. Numeric arrays and retained failed attempts require additional disk space; check capacity before fetching. Generation/CRC mismatch is an admission failure requiring review, not a reason to update the pins silently.

## Conversion and structural review

`prepare_goop_official.py` is description-only by default. For a root-approved conversion, pass `--execute --input-dir <complete acquisition directory> --output-dir <fresh separate output> --acquisition-report <completed receipt> --reader <unchanged outputs/AdaptGNS/research/prepare_full_waterdrop.py> --splits train valid`.

The wrapper pins the reader to SHA256 `ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33` and reuses its `VerifiedTFRecords` and `decode_record` functions without changing them. It does not call the WaterDrop-specific `convert_split`: that routine imposes other dataset assumptions and removes failed staging. The wrapper recomputes complete source SHA256/CRC32C, checks each record's length and payload CRC32C through EOF, preserves record order and keys, and compares saved element bytes with the original protobuf byte strings.

Positions remain float32 and particle types remain int64. Optional `step_context` arrays are retained byte-for-byte, including NaN payload bits and negative zero; they are explicitly unreviewed for model use. Unknown fields, malformed shapes, nonfinite positions and fewer than seven frames fail with preserved staging. Exact duplicate position/type trajectories within or across the selected splits fail; auxiliary values are retained but excluded from the duplicate definition. Actual record count, frame lengths, horizons after six frames, particle types, auxiliary presence and official-parser agreement are reported for admission. Both selected splits must finish conversion and integrity checks before the first manifest is published. A publication interruption may leave a partial manifest set; the structural report must be complete and both manifest hashes must match before admission.

Do not infer training admission from successful conversion. The final report says `complete_structural_only` and `scientific_training_admission: false`. The root must review actual frame lengths, all complete records, exact particle type IDs, auxiliary semantics, source family, duplicate evidence, bounds and training-loader compatibility. A later Goop trainer must accept the separately reviewed `official_gns_tfrecord` manifest; the Sand trainer deliberately accepts a different family. Keep Goop source, manifests, runs and results separate from WaterDrop and Sand, including unsuccessful outcomes. Test acquisition and any reserved test evaluation need their own frozen protocol and root gate.

## Preparation checks

Fifteen tiny synthetic tests pass, including source-array bytes, auxiliary NaN bits, differing actual frame counts, source order, whole-object checksums across multiple 1 MiB blocks, record CRC failures, receipt identity, duplicate refusal, unknown fields, nonfinite positions, fresh-output refusal and publication after both splits finish. These checks establish wrapper behavior only; they are not observations about Goop or training performance. See `prepare_goop_official_synthetic_test.log` and `prepare_goop_official_preparation.json` for exact hashes.
