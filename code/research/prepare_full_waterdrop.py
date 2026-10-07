"""Particle-simulation methods and data utilities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time

import crc32c
import numpy as np
from tfrecord import example_pb2


FORMAT = "gns-trajectory-manifest"
DEFAULT_COUNTS = {"train": 1000, "valid": 30, "test": 30}


class TFRecordIntegrityError(ValueError):
    pass


def masked_crc(data):
    value = crc32c.crc32c(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class VerifiedTFRecords:
    """Sequential reader. Whole-file hash is available only after exact EOF."""
    def __init__(self, path, max_record_bytes=128 * 1024 * 1024):
        self.path = Path(path)
        self.max_record_bytes = max_record_bytes
        self.sha256 = None
        self.size_bytes = None
        self.record_count = 0

    def __iter__(self):
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            initial = os.fstat(source.fileno())
            while True:
                offset = source.tell()
                header = source.read(8)
                if not header:
                    break
                if len(header) != 8:
                    raise TFRecordIntegrityError(f"Truncated TFRecord length at byte {offset}")
                length_crc = source.read(4)
                if len(length_crc) != 4:
                    raise TFRecordIntegrityError(f"Truncated TFRecord length CRC at byte {offset}")
                if struct.unpack("<I", length_crc)[0] != masked_crc(header):
                    raise TFRecordIntegrityError(f"TFRecord length CRC mismatch at byte {offset}")
                length = struct.unpack("<Q", header)[0]
                if length > self.max_record_bytes:
                    raise TFRecordIntegrityError(f"Record of {length} bytes exceeds configured size guard")
                payload = source.read(length)
                if len(payload) != length:
                    raise TFRecordIntegrityError(f"Truncated TFRecord payload at byte {offset}")
                payload_crc = source.read(4)
                if len(payload_crc) != 4:
                    raise TFRecordIntegrityError(f"Truncated TFRecord payload CRC at byte {offset}")
                if struct.unpack("<I", payload_crc)[0] != masked_crc(payload):
                    raise TFRecordIntegrityError(f"TFRecord payload CRC mismatch at byte {offset}")
                for block in (header, length_crc, payload, payload_crc):
                    digest.update(block)
                index = self.record_count
                self.record_count += 1
                yield index, offset, payload
            final = os.fstat(source.fileno())
            if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                raise TFRecordIntegrityError("Source changed during conversion; use a completed download")
            if source.tell() != initial.st_size:
                raise TFRecordIntegrityError("Did not consume the complete source file")
            self.sha256 = digest.hexdigest()
            self.size_bytes = initial.st_size


def one_bytes(feature, label):
    if feature.WhichOneof("kind") != "bytes_list" or len(feature.bytes_list.value) != 1:
        raise TFRecordIntegrityError(f"Expected one byte string for {label}")
    return feature.bytes_list.value[0]


def array_description(path, split, array):
    return {"path": f"{split}/{path.name}", "shape": list(array.shape),
            "dtype": array.dtype.str, "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def decode_record(payload, index, offset, split, staging, dimension, expected_frames):
    example = example_pb2.SequenceExample()
    try:
        example.ParseFromString(payload)
    except Exception as error:
        raise TFRecordIntegrityError(f"Invalid protobuf in record {index}") from error
    context = example.context.feature
    sequences = example.feature_lists.feature_list
    if set(context) != {"key", "particle_type"}:
        raise TFRecordIntegrityError(f"Unexpected context fields in record {index}: {list(context)}")
    if "position" not in sequences or set(sequences) - {"position", "step_context"}:
        raise TFRecordIntegrityError(f"Unexpected sequence fields in record {index}: {list(sequences)}")
    if context["key"].WhichOneof("kind") != "int64_list" or not context["key"].int64_list.value:
        raise TFRecordIntegrityError(f"Missing integer source key in record {index}")
    raw_types = one_bytes(context["particle_type"], "particle_type")
    if not raw_types or len(raw_types) % 8:
        raise TFRecordIntegrityError(f"Invalid particle-type byte length in record {index}")
    particle_types = np.frombuffer(raw_types, dtype="<i8")
    if np.any((particle_types < 0) | (particle_types > 8)):
        raise TFRecordIntegrityError(f"Particle IDs outside the GNS range 0..8 in record {index}")
    frames = sequences["position"].feature
    if not frames or (expected_frames is not None and len(frames) != expected_frames):
        raise TFRecordIntegrityError(f"Record {index} has {len(frames)} frames; expected {expected_frames}")
    shape = (len(frames), len(particle_types), dimension)
    position_path = staging / f"position_{index:06d}.npy"
    positions = np.lib.format.open_memmap(position_path, mode="w+", dtype="<f4", shape=shape)
    for frame_index, frame in enumerate(frames):
        raw = one_bytes(frame, "position")
        if len(raw) != len(particle_types) * dimension * 4:
            raise TFRecordIntegrityError(f"Position size mismatch in record {index}, frame {frame_index}")
        values = np.frombuffer(raw, dtype="<f4").reshape(len(particle_types), dimension)
        if not np.isfinite(values).all():
            raise TFRecordIntegrityError(f"Nonfinite position in record {index}, frame {frame_index}")
        positions[frame_index] = values
    positions.flush()
    position_info = array_description(position_path, split, positions)
    del positions
    type_path = staging / f"type_{index:06d}.npy"
    np.save(type_path, particle_types, allow_pickle=False)
    type_info = array_description(type_path, split, particle_types)
    record = {"id": f"{split}:{index:06d}", "source_index": index,
              "source_key": list(context["key"].int64_list.value), "source_offset_bytes": offset,
              "record_payload_bytes": len(payload), "record_payload_sha256": hashlib.sha256(payload).hexdigest(),
              "positions": position_info, "particle_types": type_info,
              "trajectory_content_sha256": hashlib.sha256(
                  (position_info["sha256"] + ":" + type_info["sha256"]).encode()).hexdigest()}
    if "step_context" in sequences:
        auxiliary = sequences["step_context"].feature
        if len(auxiliary) != len(frames):
            raise TFRecordIntegrityError(f"step_context frame count mismatch in record {index}")
        width_bytes = len(one_bytes(auxiliary[0], "step_context"))
        if not width_bytes or width_bytes % 4:
            raise TFRecordIntegrityError(f"Invalid step_context shape in record {index}")
        auxiliary_path = staging / f"step_context_{index:06d}.npy"
        array = np.lib.format.open_memmap(auxiliary_path, mode="w+", dtype="<f4",
                                         shape=(len(frames), width_bytes // 4))
        for frame_index, frame in enumerate(auxiliary):
            raw = one_bytes(frame, "step_context")
            if len(raw) != width_bytes:
                raise TFRecordIntegrityError(f"step_context width changes in record {index}")
            array[frame_index] = np.frombuffer(raw, dtype="<f4")
        array.flush()
        record["step_context"] = array_description(auxiliary_path, split, array)
        record["step_context"]["use"] = "preserved auxiliary data; NaN placeholders allowed; not a simulator input"
        del array
    return record


def convert_split(source_path, output_dir, metadata_path, split, expected_records=None,
                  expected_source_bytes=None, expected_source_sha256=None,
                  max_record_bytes=128 * 1024 * 1024):
    source_path, output_dir, metadata_path = map(Path, (source_path, output_dir, metadata_path))
    if split not in DEFAULT_COUNTS:
        raise ValueError("split must be train, valid, or test")
    metadata = json.loads(metadata_path.read_text())
    dimension = int(metadata["dim"])
    if dimension != 2:
        raise ValueError("This converter targets the two-dimensional WaterDrop schema")
    expected_frames = int(metadata["sequence_length"]) + 1 if metadata.get("sequence_length") is not None else None
    expected_records = DEFAULT_COUNTS[split] if expected_records is None else expected_records
    if expected_source_bytes is not None and source_path.stat().st_size != expected_source_bytes:
        raise TFRecordIntegrityError("Source size differs from expected complete download size")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, destination = output_dir / f"{split}.json", output_dir / split
    if manifest_path.exists() or destination.exists():
        raise FileExistsError(f"Split output already exists: {destination}; choose a new output directory")
    output_metadata = output_dir / "metadata.json"
    if output_metadata.exists() and json.loads(output_metadata.read_text()) != metadata:
        raise ValueError("Output metadata differs from requested metadata")
    other_contents = set()
    for other in DEFAULT_COUNTS:
        path = output_dir / f"{other}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("format") != FORMAT:
                raise ValueError(f"Unrecognized existing split manifest: {path}")
            other_contents.update(row["trajectory_content_sha256"] for row in previous["records"])
    staging = Path(tempfile.mkdtemp(prefix=f".{split}-", suffix=".staging", dir=output_dir))
    records, seen = [], set()
    started = time.perf_counter()
    reader = VerifiedTFRecords(source_path, max_record_bytes)
    try:
        for index, offset, payload in reader:
            record = decode_record(payload, index, offset, split, staging, dimension, expected_frames)
            content_hash = record["trajectory_content_sha256"]
            if content_hash in other_contents:
                raise TFRecordIntegrityError(f"Duplicate trajectory content across splits: {record['id']}")
            if content_hash in seen:
                raise TFRecordIntegrityError(f"Duplicate trajectory content within split: {record['id']}")
            seen.add(content_hash)
            records.append(record)
            if len(records) % 50 == 0:
                print(split, "verified trajectories", len(records), flush=True)
        if len(records) != expected_records:
            raise TFRecordIntegrityError(f"Found {len(records)} records; expected {expected_records}")
        if expected_source_sha256 is not None and reader.sha256 != expected_source_sha256.lower():
            raise TFRecordIntegrityError("Source SHA256 differs from expected value")
        if not output_metadata.exists():
            temporary = output_metadata.with_suffix(".json.tmp")
            temporary.write_bytes(metadata_path.read_bytes())
            temporary.replace(output_metadata)
        manifest = {"format": FORMAT, "version": 1, "split": split,
                    "source": {"file": source_path.name, "size_bytes": reader.size_bytes,
                               "sha256": reader.sha256, "CRC_verified": True,
                               "record_count": reader.record_count},
                    "metadata": metadata, "metadata_sha256": file_sha256(output_metadata),
                    "record_count": len(records), "records": records,
                    "position_units": "original dataset coordinates, no rescaling",
                    "memory_contract": "one protobuf record plus a single mapped output trajectory; no all-trajectory stack",
                    "publication": "manifest appears only after complete source and all arrays are verified",
                    "conversion_seconds": time.perf_counter() - started,
                    "converter_sha256": file_sha256(__file__)}
        if split == "test":
            manifest["evaluation_reservation"] = {
                "previously_inspected_pilot_source_indices": [0, 1, 2],
                "reserved_new_model_holdout_source_indices": list(range(3, len(records))),
                "instruction": "Do not evaluate new models on reserved trajectories before freezing the validation-selected protocol"}
        # The record directory is complete before a reader can see its manifest.
        staging.replace(destination)
        atomic_json(manifest_path, manifest)
        print(split, "complete", len(records), "records", reader.size_bytes, "source bytes", flush=True)
        return manifest
    except Exception:
        # Only this invocation's private staging files are removed; source data
        # and any previously published splits remain intact.
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=DEFAULT_COUNTS, default=["train", "valid"])
    parser.add_argument("--expected-train-bytes", type=int, default=None)
    parser.add_argument("--max-record-bytes", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args()
    for split in args.splits:
        convert_split(args.input_dir / f"{split}.tfrecord", args.output_dir, args.metadata, split,
                      expected_source_bytes=args.expected_train_bytes if split == "train" else None,
                      max_record_bytes=args.max_record_bytes)


if __name__ == "__main__":
    main()
