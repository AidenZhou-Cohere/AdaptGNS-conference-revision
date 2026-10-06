"""Pure Goop split/evidence contract; no source acquisition or numerical imports."""
import hashlib
import json
from pathlib import Path
import re

METADATA_SHA = "565d6e13be91be6a6b0fbc31a9aa19ed70a5505228411d0928288fcf874ca3dd"
TRAIN_MANIFEST_SHA = "5ef43daf9bac961a69bd460a539891624b79c8a13f80ca851e85802197982256"
VALID_MANIFEST_SHA = "3227e03c4c7fcee9f99c4104010774cabc74c34bfda667d76c5894590dcfc415"
CONVERTER_SHA = "fa4b7d883d7c360500fc3c603c9cd14538c916f1ddd7435aedfc8ec2420f2985"
READER_SHA = "ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33"
ACQUISITION_SHA = "9e4b106324b1e7c5add4404565eca888eaa0ebd3c233622308c113e88c5e6a8d"
CONTEXT_SHA = "c81ae2f1565e61542bcc406c4a9d71b67135620857ad62292eaef0e29b407de9"
CENSUS_SHA = "3278055e119c482c5620ff66b2c4e4e6c09361240155cc77a7ad98dbaa13d8d1"
CONTEXT_SOURCES = {"README.md": "9db737244fccd3ea37d525ba244b7ea2bddd300f420a346bbfdd2893c419a69a",
    "reading_utils.py": "5868022f76eaf15b2626125bdaa3c973ccaf2dfc0b0ec50d9e05d0c3d973e149",
    "train.py": "44b0d7759b3af37cb9c0dc442c13440302823413091e0f1b572bc7f4b95f6d60",
    "learned_simulator.py": "bda3d60fcaf8a7a1a963fe6e87790e38955c38602da1eeee5d214ef2f4518da9"}
OMISSION = {"name": "step_context", "handling": "preserved_and_excluded_from_model_inputs",
    "reason": "unused_by_released_goop_parser_model_contract_without_context_mean",
    "official_commit": "f5de0ede8430809180254ee957abf36ed62579ef"}
AUXILIARY_USE = "Preserved source auxiliary data; semantics unreviewed and model admission required"
SOURCES = {
    "train": (1000, 3358535518, "8a0b0cfe3ef56f533abf14d963d5cf632235aa3570cf61a4b42ab655a15cae59", "1599154403717337", "OrpMIQ==", TRAIN_MANIFEST_SHA),
    "valid": (30, 93682221, "0e7b261d9ce59491ed79756a1d60bb57324c597a042e257f21b53b52b3c4f79c", "1599153784596039", "EqHuFw==", VALID_MANIFEST_SHA),
}
MANIFEST_KEYS = {"format", "version", "split", "dataset", "source", "metadata", "metadata_sha256", "record_count", "records", "converter_sha256", "reader_sha256"}
SOURCE_KEYS = {"family", "dataset", "file", "size_bytes", "sha256", "generation", "crc32c_base64", "CRC_verified", "record_count", "acquisition_report_sha256"}
RECORD_KEYS = {"id", "source_index", "source_key", "source_offset_bytes", "record_payload_bytes", "record_payload_sha256", "positions", "particle_types", "step_context", "trajectory_content_sha256"}
ARRAY_KEYS = {"path", "shape", "dtype", "size_bytes", "sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_split(manifest, admission, structural, split, final=False):
    require(split in (("valid", "test") if final else ("train", "valid")), "Split is outside declared timing/final scope")
    schema = "adaptgns_goop_graph_support_final_evaluation_admission_v1" if final else "adaptgns_goop_rollout_admission_v1"
    status = "admitted_for_final_evaluation" if final else "admitted_for_feasibility"
    require(admission.get("schema") == schema and admission.get("status") == status and admission.get("issued_by") == "root"
            and admission.get("dataset") == "Goop" and admission.get("split") == split, "Separate root-issued Goop split admission required")
    for key in ("manifest_sha256", "structural_report_sha256", "source_sha256", "converter_sha256", "acquisition_report_sha256", "auxiliary_report_sha256"):
        require(digest(admission.get(key)), "Exact split evidence SHA256 required: " + key)
    count = 1000 if split == "train" else 30
    require(admission.get("frames_per_trajectory") == 401 and admission.get("record_count") == count
            and admission.get("particle_type_ids") == [7] and admission.get("position_dtype") == "<f4"
            and admission.get("particle_type_dtype") == "<i8" and admission.get("auxiliary_policy") == OMISSION
            and admission.get("context_semantics_sha256") == CONTEXT_SHA, "Goop frame/type/auxiliary source contract differs")
    require(set(manifest) == MANIFEST_KEYS and manifest.get("format") == "gns-trajectory-manifest" and manifest.get("version") == 1
            and manifest.get("dataset") == "Goop" and manifest.get("split") == split, "Exact official Goop numeric manifest required")
    source = manifest["source"]
    require(set(source) == SOURCE_KEYS and source.get("family") == "official_gns_tfrecord" and source.get("dataset") == "Goop"
            and source.get("file") == split + ".tfrecord" and source.get("record_count") == count and source.get("CRC_verified") is True
            and integer(source.get("size_bytes"), 1) and digest(source.get("sha256")) and source["sha256"] == admission["source_sha256"]
            and isinstance(source.get("generation"), str) and source["generation"].isdigit()
            and isinstance(source.get("crc32c_base64"), str) and len(source["crc32c_base64"]) == 8
            and source.get("acquisition_report_sha256") == admission["acquisition_report_sha256"], "Official source generation/CRC/identity differs")
    if split in SOURCES:
        n, size, source_sha, generation, crc, manifest_sha = SOURCES[split]
        require((count, source["size_bytes"], source["sha256"], source["generation"], source["crc32c_base64"], admission["manifest_sha256"])
                == (n, size, source_sha, generation, crc, manifest_sha)
                and admission["converter_sha256"] == CONVERTER_SHA and admission["acquisition_report_sha256"] == ACQUISITION_SHA
                and admission["auxiliary_report_sha256"] == CENSUS_SHA, "Acquired train/valid pins differ")
    else:
        require(final and admission.get("reserved_test_acquired_after_cohort_freeze") is True
                and admission.get("test_converter_independently_reviewed") is True
                and digest(admission.get("cross_split_audit_sha256")), "Reserved test requires later complete source/converter/overlap admission")
    require(manifest.get("metadata_sha256") == admission.get("metadata_sha256") == METADATA_SHA
            and manifest.get("reader_sha256") == admission.get("reader_sha256") == READER_SHA
            and manifest.get("converter_sha256") == admission["converter_sha256"], "Metadata/reader/converter identity differs")
    metadata = manifest["metadata"]
    require(metadata.get("dim") == 2 and metadata.get("sequence_length") == 400 and metadata.get("dt") == .0025
            and metadata.get("bounds") == [[.1, .9], [.1, .9]] and metadata.get("default_connectivity_radius") == .015
            and "context_mean" not in metadata and "context_std" not in metadata, "Official Goop metadata/input contract differs")
    require(structural.get("schema") == "official_goop_numeric_preparation_v1" and structural.get("status") == "complete_structural_only"
            and structural.get("test_accessed") is (split == "test")
            and structural.get("dataset") == "Goop" and structural.get("source_family") == "official_gns_tfrecord"
            and structural.get("metadata_sha256") == METADATA_SHA and structural.get("reader_sha256") == READER_SHA
            and structural.get("wrapper_sha256") == admission["converter_sha256"]
            and structural.get("acquisition_report_sha256") == admission["acquisition_report_sha256"], "Structural provenance differs")
    detail = structural.get("splits", {}).get(split, {})
    require(detail.get("manifest_sha256") == admission["manifest_sha256"] and detail.get("record_count") == count
            and detail.get("frame_lengths") == [401] and detail.get("forecast_horizons_after_six_frames") == [395]
            and detail.get("particle_type_ids") == [7] and all(detail.get(k) is True for k in
                ("source_EOF_SHA256_verified", "whole_object_crc32c_verified", "TFRecord_length_and_payload_CRC32C_verified",
                 "all_match_official_parser_frame_count", "all_source_array_bytes_preserved_exact")), "Complete structural preservation evidence required")
    records = manifest.get("records", [])
    require(len(records) == manifest.get("record_count") == len(detail.get("records", [])) == count, "All split records and preservation evidence required")
    contents, payloads, offset = set(), set(), 0
    for index, (record, evidence) in enumerate(zip(records, detail["records"])):
        require(set(record) == RECORD_KEYS and record.get("id") == f"{split}:{index:06d}"
                and integer(record.get("source_index")) and record["source_index"] == index and record.get("source_key") == [index]
                and record.get("source_offset_bytes") == offset and integer(record.get("record_payload_bytes"), 1)
                and digest(record.get("record_payload_sha256")) and record["record_payload_sha256"] not in payloads, "Ordered source records/offsets/keys differ")
        offset += record["record_payload_bytes"] + 16
        payloads.add(record["record_payload_sha256"])
        p, t, aux = (record[k] for k in ("positions", "particle_types", "step_context"))
        for name, descriptor, prefix in (("positions", p, "position"), ("particle_types", t, "type"), ("step_context", aux, "step_context")):
            require(set(descriptor) == ARRAY_KEYS | ({"use"} if name == "step_context" else set())
                    and descriptor.get("path") == f"{split}/{prefix}_{index:06d}.npy" and digest(descriptor.get("sha256"))
                    and integer(descriptor.get("size_bytes"), 1), "Exact preserved numeric descriptors required")
        require(isinstance(p.get("shape"), list) and len(p["shape"]) == 3 and p["shape"][0] == 401
                and integer(p["shape"][1], 1) and p["shape"][2] == 2 and p["dtype"] == "<f4"
                and t["shape"] == [p["shape"][1]] and t["dtype"] == "<i8"
                and aux["shape"] == [401, 1] and aux["dtype"] == "<f4" and aux["use"] == AUXILIARY_USE, "Preserved array frame/type/context shape differs")
        require(evidence.get("source_index") == index and evidence.get("source_offset_bytes") == record["source_offset_bytes"]
                and evidence.get("frames") == 401 and evidence.get("particles") == p["shape"][1]
                and evidence.get("position_dtype") == "<f4" and evidence.get("particle_type_dtype") == "<i8"
                and evidence.get("particle_type_ids") == [7] and evidence.get("all_source_array_bytes_preserved_exact") is True,
                "Per-record preserved source evidence differs")
        content = hashlib.sha256((p["sha256"] + ":" + t["sha256"]).encode()).hexdigest()
        require(record["trajectory_content_sha256"] == content and content not in contents, "Duplicate/inconsistent split trajectory")
        contents.add(content)
    require(offset == source["size_bytes"], "Record offsets do not cover complete official source")
    return 401


def verify_evidence(args, manifest, admission, helpers):
    """Called after root split/cohort gates. Preserve and hash excluded auxiliaries."""
    files = {str(Path(getattr(args, key)).resolve()): admission[pin] for key, pin in
             (("acquisition_report", "acquisition_report_sha256"), ("context_semantics", "context_semantics_sha256"),
              ("auxiliary_report", "auxiliary_report_sha256"))}
    context_root = args.context_semantics.resolve().parent / "goop_context_semantics_sources"
    files.update({str(context_root / name): pin for name, pin in CONTEXT_SOURCES.items()})
    require(all(sha(path) == pin for path, pin in files.items()), "Acquisition/context/census evidence bytes differ")
    receipt = json.loads(args.acquisition_report.read_text())
    source = manifest["source"]
    matches = [r for r in receipt.get("files", []) if r.get("name") == source["file"]]
    require(receipt.get("status") == "complete" and receipt.get("dataset") == "Goop" and receipt.get("source_family") == "official_gns_tfrecord"
            and len(matches) == 1, "Complete acquired split receipt required")
    row = matches[0]
    require(row.get("status") == "complete" and row.get("saved_name") == source["file"]
            and row.get("url") == "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop/" + source["file"]
            and row.get("received_bytes") == source["size_bytes"] and row.get("sha256") == source["sha256"]
            and row.get("generation") == source["generation"] and row.get("crc32c_base64") == source["crc32c_base64"]
            and row.get("crc32c_verified") is True, "Received official split identity differs")
    census = json.loads(args.auxiliary_report.read_text())
    observed = census.get("splits", {}).get(args.split, {})
    require(census.get("schema") == "adaptgns_goop_auxiliary_census_v1" and census.get("status") == "all_preserved_auxiliary_bytes_verified"
            and census.get("test_accessed") is (args.split == "test")
            and observed.get("manifest_sha256") == admission["manifest_sha256"] and observed.get("record_count") == len(manifest["records"])
            and all(observed.get(k) is True for k in ("context_mean_absent", "context_std_absent", "all_context_descriptors_and_bytes_verified"))
            and len(observed.get("records", [])) == len(manifest["records"]), "Complete matching auxiliary census required")
    for record, detail in zip(manifest["records"], observed["records"]):
        descriptor = record["step_context"]
        require(detail.get("id") == record["id"] and detail.get("source_index") == record["source_index"]
                and detail.get("sha256") == descriptor["sha256"] and detail.get("shape") == [401, 1] and detail.get("dtype") == "<f4"
                and detail.get("elements") == 401 and detail.get("nan_count") == 1 and detail.get("finite_count") == 400
                and detail.get("positive_infinity_count") == detail.get("negative_infinity_count") == 0
                and detail.get("unique_float32_bits_hex") == ["00000000", "7fc00000"], "Auxiliary census values differ; source review required")
        path = helpers.data_loader._manifest_array_path(args.manifest.resolve().parent, descriptor)
        require(path.stat().st_size == descriptor["size_bytes"] and sha(path) == descriptor["sha256"], "Preserved auxiliary bytes differ")
        values = helpers.np.load(path, allow_pickle=False)
        require(values.dtype.str == "<f4" and list(values.shape) == [401, 1] and int(helpers.np.isnan(values).sum()) == 1
                and int(helpers.np.isfinite(values).sum()) == 400
                and [f"{int(v):08x}" for v in helpers.np.unique(values.view("<u4"))] == ["00000000", "7fc00000"], "Preserved auxiliary representation changed")
        files[str(path)] = descriptor["sha256"]
    if args.split == "test":
        audit_path = args.cross_split_audit
        require(audit_path is not None and sha(audit_path) == admission["cross_split_audit_sha256"], "Root full cross-split audit required")
        audit = json.loads(audit_path.read_text())
        require(audit.get("schema") == "adaptgns_goop_all_split_integrity_audit_v1" and audit.get("issued_by") == "root"
                and audit.get("status") == "all_required_splits_verified" and audit.get("duplicate_pairs") == []
                and audit.get("manifest_sha256") == {"train": TRAIN_MANIFEST_SHA, "valid": VALID_MANIFEST_SHA, "test": admission["manifest_sha256"]}
                and audit.get("record_counts") == {"train": 1000, "valid": 30, "test": 30}
                and audit.get("definition") == "exact stored position/type dtype,shape,bytes; auxiliaries preserved separately",
                "Complete exact train/valid/test overlap accounting required")
        files[str(audit_path.resolve())] = admission["cross_split_audit_sha256"]
    return files
