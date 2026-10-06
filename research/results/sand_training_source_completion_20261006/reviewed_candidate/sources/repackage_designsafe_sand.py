#!/usr/bin/env python3
"""Lossless numeric repackaging of the exact acquired DesignSafe Sand NPZ family.

No TFRecord reconstruction, test split, network, model or general pickle loader.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import re
import stat
import struct
import sys
import time
import zipfile
import zlib

sys.dont_write_bytecode = True
SCHEMA = "designsafe_sand_numeric_repackage_v1"
METADATA_SHA = "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0"
PUBLISHED_DIRECTORY = "/published-data/PRJ-3702/Project--graph-network-simulator-datasets/data/Sand/dataset/"
SOURCE_BYTES = {"train": 2676940383, "valid": 82712898}
SOURCE_SHA = {"train": "e0b7f68b50702f1af3edfa828b6097e3b28158ddb57631491e60a29658318f7d",
              "valid": "f9c29861107b481ad5eb4a340f4f1016fcaf4050cd6cfc8ae3056f148ab6d903"}
COUNTS = {"train": 1000, "valid": 30}
GLOBALS = {("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct")}
OPCODES = {"PROTO", "GLOBAL", "BINPUT", "BININT1", "TUPLE1", "SHORT_BINBYTES", "TUPLE3",
           "REDUCE", "MARK", "BINUNICODE", "NEWFALSE", "NEWTRUE", "NONE", "BININT", "TUPLE",
           "BUILD", "EMPTY_LIST", "BINGET", "BININT2", "BINBYTES", "APPENDS", "STOP"}
MAX_MEMBER_BYTES = 128 * 1024 * 1024


class AdmissionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def object_payload(raw):
    require(raw[:8] == b"\x93NUMPY\x01\x00" and len(raw) >= 10, "Unsupported NPY version/header")
    length = struct.unpack("<H", raw[8:10])[0]
    require(0 < length <= 65536 and 10 + length < len(raw), "Invalid NPY header length")
    header = ast.literal_eval(raw[10:10 + length].decode("latin1"))
    require(header == {"descr": "|O", "fortran_order": False, "shape": (2,)},
            "Outer NPY is not the reviewed two-array object envelope; auxiliaries are not discarded")
    return raw[10 + length:]


def inspect_pickle(payload):
    names, counts, stop = set(), {}, None
    for opcode, argument, position in pickletools.genops(payload):
        require(opcode.name in OPCODES, f"Pickle opcode {opcode.name} outside reviewed protocol3 profile")
        counts[opcode.name] = counts.get(opcode.name, 0) + 1
        if opcode.name == "PROTO":
            require(position == 0 and argument == 3, "Only the reviewed pickle protocol3 is supported")
        if opcode.name == "GLOBAL":
            module, name = argument.split(" ", 1)
            require((module, name) in GLOBALS, "Pickle global outside the three NumPy reconstruction primitives")
            names.add((module, name))
        if opcode.name == "STOP":
            stop = position
    require(counts.get("PROTO") == 1 and names == GLOBALS and stop == len(payload) - 1,
            "Pickle protocol/globals/terminal EOF differs from reviewed profile")
    return {"protocol": 3, "globals": [list(value) for value in sorted(names)], "opcode_counts": counts}


def restricted_array(np, payload):
    inspected = inspect_pickle(payload)
    multiarray = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
    serialized_dtypes = []
    def captured_dtype(*args):
        dtype = np.dtype(*args)
        serialized_dtypes.append(dtype)
        return dtype
    permitted = {("numpy", "dtype"): captured_dtype, ("numpy", "ndarray"): np.ndarray,
                 ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct}
    class ArrayUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            require((module, name) in permitted, "Unpickler refused an unreviewed global")
            return permitted[(module, name)]
        def persistent_load(self, pid):
            raise AdmissionError("Persistent pickle references are unsupported")
    stream = io.BytesIO(payload)
    value = ArrayUnpickler(stream, fix_imports=False, encoding="ASCII", errors="strict").load()
    require(stream.tell() == len(payload), "Unpickler did not finish at exact payload EOF")
    # NumPy's ndarray __setstate__ can normalize nonnative endian data. Refuse
    # that case instead of falsely claiming byte/dtype preservation afterward.
    require(all(dtype.isnative and dtype.fields is None and dtype.subdtype is None
                and dtype.kind in "Ofiu" for dtype in serialized_dtypes),
            "Serialized dtype would require endian normalization or an unreviewed dtype reconstruction")
    inspected["serialized_dtypes"] = [dtype.str for dtype in serialized_dtypes]
    require(type(value) is np.ndarray and value.dtype == np.dtype("O") and value.shape == (2,),
            "Decoded outer object differs from reviewed two-array envelope")
    return value, inspected


def value_hash(array):
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_arrays(np, outer, metadata):
    positions, types = outer[0], outer[1]
    require(type(positions) is np.ndarray and type(types) is np.ndarray, "Both envelope entries must be plain NumPy arrays")
    require(not positions.dtype.hasobject and positions.dtype.kind == "f" and positions.ndim == 3
            and positions.shape[0] >= 7 and positions.shape[1] > 0 and positions.shape[2] == metadata["dim"],
            "Positions require finite floating[T>=7,N,dim]; no temporal rebuilding or dtype cast is allowed")
    require(not types.dtype.hasobject and types.dtype.kind in "iu" and types.ndim in (0, 1)
            and (types.ndim == 0 or types.shape == (positions.shape[1],)), "Particle types require an integer scalar or lengthN array")
    require(bool(np.isfinite(positions).all()), "Nonfinite source positions; source and staging retained")
    require(bool(((types >= 0) & (types <= 8)).all()), "Particle type outside the supported0..8 schema")
    bounds = np.asarray(metadata["bounds"], dtype=np.float64)
    require(bounds.shape == (metadata["dim"], 2) and bool(np.isfinite(bounds).all())
            and bool((bounds[:, 1] > bounds[:, 0]).all()), "Invalid metadata boundaries")
    expanded = np.broadcast_to(types, (positions.shape[1],)) if types.ndim == 0 else types
    ids, frequencies = np.unique(expanded, return_counts=True)
    outside = ((positions < bounds[:, 0]) | (positions > bounds[:, 1])).any(axis=-1)
    details = {"frames": int(positions.shape[0]), "particles": int(positions.shape[1]),
               "position_dtype": positions.dtype.str, "particle_type_dtype": types.dtype.str,
               "particle_type_shape": list(types.shape), "particle_type_counts": {str(int(i)): int(n) for i, n in zip(ids, frequencies)},
               "kinematic_type3_particles": int((expanded == 3).sum()),
               "position_min": positions.min(axis=(0, 1)).tolist(), "position_max": positions.max(axis=(0, 1)).tolist(),
               "outside_boundary_particle_frames": int(outside.sum()), "boundary_policy": "reported, never clipped or rejected for excursion alone",
               "source_position_value_sha256": value_hash(positions), "source_type_value_sha256": value_hash(types)}
    return positions, types, details


def save_array(np, path, split, array):
    np.save(path, array, allow_pickle=False)
    restored = np.load(path, mmap_mode="r", allow_pickle=False)
    require(restored.dtype.str == array.dtype.str and restored.shape == array.shape and value_hash(restored) == value_hash(array),
            "Numeric repackaging changed dtype, shape or element bytes")
    del restored
    return {"path": f"{split}/{path.name}", "shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": sha(path), "size_bytes": path.stat().st_size}


def check_archive_eof(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65557))
        tail = stream.read(65557)
    end = tail.rfind(b"PK\x05\x06")
    require(end >= 0 and end + 22 <= len(tail), "Missing ZIP end record")
    fields = struct.unpack("<4s4H2IH", tail[end:end + 22])
    require(end + 22 + fields[-1] == len(tail), "Bytes remain after ZIP end/comment")


def repackage_split(np, archive_path, split, row, staging, metadata, seen, expected_count):
    require(not archive_path.is_symlink() and archive_path.is_file(), "Source archive must be a regular non-symlink file")
    initial = archive_path.stat()
    require(initial.st_size == row["received_bytes"] and sha(archive_path) == row["received_sha256"], "Archive differs from complete acquisition receipt")
    check_archive_eof(archive_path)
    staging.mkdir(mode=0o700)
    records, detail_rows = [], []
    current = {"source_index": None, "source_member": None}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members, listed = archive.infolist(), row["zip"]["members"]
            require(len(members) == len(listed) == expected_count == row["zip"]["member_count"], "Archive record count differs from observed acquisition inventory")
            require(len({item.filename for item in members}) == len(members), "Duplicate ZIP source identifiers")
            require(all(first.header_offset < second.header_offset for first, second in zip(members, members[1:])),
                    "ZIP inventory order differs from physical member order")
            for index, (member, earlier) in enumerate(zip(members, listed)):
                current = {"source_index": index, "source_member": member.filename}
                require(re.fullmatch(r"simulation_trajectory_[0-9]+\.npy", member.filename) is not None
                        and not member.is_dir() and stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG), "Unsupported ZIP member identity/type")
                require(not member.flag_bits & 1 and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and 0 < member.file_size <= MAX_MEMBER_BYTES, "Unsupported encrypted/compressed/oversized member")
                require(earlier["source_member_index"] == index and earlier["name"] == member.filename
                        and earlier["uncompressed_bytes"] == member.file_size and earlier["compressed_bytes"] == member.compress_size
                        and earlier["listed_crc32"] == f"{member.CRC:08x}", "ZIP order/identity differs from acquisition report")
                with archive.open(member) as stream:
                    raw = stream.read(member.file_size + 1)
                    require(len(raw) == member.file_size and stream.read(1) == b"", "ZIP member did not finish at exact EOF")
                require(zlib.crc32(raw) & 0xffffffff == member.CRC, "Full member CRC differs")
                outer, pickle_info = restricted_array(np, object_payload(raw))
                positions, types, details = validate_arrays(np, outer, metadata)
                logical = hashlib.sha256((details["source_position_value_sha256"] + ":" + details["source_type_value_sha256"]).encode()).hexdigest()
                require(logical not in seen, "Duplicate trajectory values within/across selected splits")
                seen.add(logical)
                position_info = save_array(np, staging / f"position_{index:06d}.npy", split, positions)
                type_info = save_array(np, staging / f"type_{index:06d}.npy", split, types)
                record = {"id": f"{split}:{index:06d}", "source_index": index, "source_member": member.filename,
                          "positions": position_info, "particle_types": type_info,
                          "trajectory_content_sha256": hashlib.sha256((position_info["sha256"] + ":" + type_info["sha256"]).encode()).hexdigest(),
                          "logical_content_sha256": logical}
                records.append(record)
                detail_rows.append({**current, **details, "full_member_crc32": f"{member.CRC:08x}",
                                    "member_sha256": hashlib.sha256(raw).hexdigest(), "pickle": pickle_info,
                                    "numeric_dtype_shape_values_verified_exact": True})
                if (index + 1) % 50 == 0:
                    atomic_json(staging / "progress.json", {"state": "in_progress", "completed_members": len(records), "last_source_member": member.filename})
        final = archive_path.stat()
        require((initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns)
                and sha(archive_path) == row["received_sha256"], "Archive changed during repackaging")
        summary = {"record_count": len(records), "frame_lengths": sorted({value["frames"] for value in detail_rows}),
                   "position_dtypes": sorted({value["position_dtype"] for value in detail_rows}),
                   "particle_type_dtypes": sorted({value["particle_type_dtype"] for value in detail_rows}),
                   "particle_type_ids": sorted({int(key) for value in detail_rows for key in value["particle_type_counts"]}),
                   "kinematic_type3_particles": sum(value["kinematic_type3_particles"] for value in detail_rows),
                   "outside_boundary_particle_frames": sum(value["outside_boundary_particle_frames"] for value in detail_rows),
                   "ZIP_CRC_verified": True, "source_eof_and_hash_verified": True, "all_numeric_values_preserved_exact": True,
                   "records": detail_rows}
        return records, summary
    except BaseException as error:
        atomic_json(staging / "failure.json", {"state": "failed", **current, "completed_members": len(records),
            "error_type": type(error).__name__, "reason": str(error),
            "source_and_staging_retained": True})
        atomic_json(staging / "completed_member_details.json", detail_rows)
        raise


def validate_receipt(receipt, splits):
    require(receipt.get("schema") == "sand_public_acquisition_v2" and receipt.get("status") == "complete_with_object_dtypes"
            and receipt.get("version") == 1 and receipt.get("published_directory") == PUBLISHED_DIRECTORY
            and receipt.get("dataset") == "Sand" and receipt.get("project_id") == "PRJ-3702"
            and receipt.get("doi") == "10.17603/ds2-0phb-dg64" and receipt.get("metadata_reference_sha256") == METADATA_SHA,
            "Expected completed exact DesignSafe Sand acquisition receipt")
    rows = {row["name"]: row for row in receipt["files"]}
    require(len(rows) == len(receipt["files"]), "Duplicate acquisition file identity")
    metadata_row = rows["metadata.json"]
    require(metadata_row["status"] == "downloaded_and_inspected" and metadata_row["stage"] == "complete"
            and metadata_row["saved_name"] == "metadata.json" and metadata_row["received_sha256"] == METADATA_SHA
            and metadata_row["received_bytes"] == metadata_row["expected_bytes"] == 363,
            "Metadata acquisition identity differs")
    for split in splits:
        row = rows[split + ".npz"]
        require(row["status"] == "downloaded_and_inspected" and row["stage"] == "complete" and row["saved_name"] == split + ".npz"
                and row["received_bytes"] == row["expected_bytes"] == SOURCE_BYTES[split]
                and row["received_sha256"] == SOURCE_SHA[split] and row["zip"]["member_count"] == COUNTS[split],
                "Source archive differs from the exact acquired family")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--acquisition-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory; parent must exist")
    parser.add_argument("--splits", nargs="+", choices=("train", "valid"), default=["train", "valid"])
    args = parser.parse_args(argv)
    if len(args.splits) != len(set(args.splits)):
        parser.error("Duplicate split names are not allowed")
    args.splits = [split for split in ("train", "valid") if split in args.splits]
    return args


def main(argv=None):
    args = parse_args(argv)
    receipt_bytes = args.acquisition_report.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    receipt = json.loads(receipt_bytes)
    rows = validate_receipt(receipt, args.splits)
    metadata_path = args.input_dir / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    require(hashlib.sha256(metadata_bytes).hexdigest() == METADATA_SHA, "Input metadata bytes differ from exact source")
    metadata = json.loads(metadata_bytes)
    require(metadata["dim"] == 2, "This repackaging contract covers2D Sand only")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    report_path = args.output_dir / "structural_report.json"
    converter_sha = sha(__file__)
    report = {"schema": SCHEMA, "status": "in_progress", "dataset": "Sand", "source_family": "designsafe_published_npz",
              "metadata_sha256": METADATA_SHA, "converter_sha256": converter_sha, "acquisition_report_sha256": receipt_sha,
              "numpy_pickle_globals": [list(value) for value in sorted(GLOBALS)], "pickle_protocol": 3,
              "test_split_accessed": False, "scientific_training_admission": False, "splits": {}}
    atomic_json(report_path, report)
    started = time.perf_counter()
    try:
        import numpy as np
        report["numpy_version"] = np.__version__
        manifests, seen = {}, set()
        for split in args.splits:
            records, summary = repackage_split(np, args.input_dir / (split + ".npz"), split, rows[split + ".npz"],
                args.output_dir / ("." + split + ".staging"), metadata, seen, COUNTS[split])
            row = rows[split + ".npz"]
            report["splits"][split] = summary
            manifests[split] = {"format": "gns-trajectory-manifest", "version": 1, "split": split, "dataset": "Sand",
                "source": {"family": "designsafe_published_npz", "dataset": "Sand", "file": split + ".npz",
                           "size_bytes": row["received_bytes"], "sha256": row["received_sha256"], "ZIP_CRC_verified": True,
                           "member_count": len(records), "acquisition_report_sha256": receipt_sha},
                "metadata": metadata, "metadata_sha256": METADATA_SHA, "record_count": len(records), "records": records,
                "converter_sha256": converter_sha}
            atomic_json(report_path, report)
        require(sha(args.acquisition_report) == receipt_sha and sha(metadata_path) == METADATA_SHA and sha(__file__) == converter_sha,
                "Receipt, metadata or repackager changed during execution")
        (args.output_dir / "metadata.json").write_bytes(metadata_bytes)
        for split, manifest in manifests.items():
            (args.output_dir / ("." + split + ".staging")).replace(args.output_dir / split)
            manifest_path = args.output_dir / (split + ".json")
            atomic_json(manifest_path, manifest)
            report["splits"][split]["manifest_sha256"] = sha(manifest_path)
        report.update(status="complete_structural_only", duplicate_trajectories_within_and_across_selected_splits=False,
                      duplicate_definition="identical decoded dtype, shape and C-order element bytes for both arrays; selected splits only",
                      all_manifest_outputs_published=True, elapsed_seconds=time.perf_counter() - started)
        atomic_json(report_path, report)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, reason=str(error),
                      staging_retained=True, elapsed_seconds=time.perf_counter() - started)
        atomic_json(report_path, report)
        raise
    print(json.dumps({"status": report["status"], "splits": args.splits, "report": "structural_report.json", "scientific_training_admission": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
