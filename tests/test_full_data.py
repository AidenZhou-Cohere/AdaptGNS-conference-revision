import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
from tfrecord import example_pb2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.prepare_full_waterdrop import (TFRecordIntegrityError, VerifiedTFRecords,
                                             convert_split, masked_crc)
from gns import data_loader


def record_bytes(positions, particle_types, key=42):
    example = example_pb2.SequenceExample()
    example.context.feature["key"].int64_list.value.append(key)
    example.context.feature["particle_type"].bytes_list.value.append(
        np.asarray(particle_types, dtype="<i8").tobytes())
    for position in positions:
        example.feature_lists.feature_list["position"].feature.add().bytes_list.value.append(
            np.asarray(position, dtype="<f4").tobytes())
        example.feature_lists.feature_list["step_context"].feature.add().bytes_list.value.append(
            np.array([np.nan], dtype="<f4").tobytes())
    payload = example.SerializeToString()
    header = struct.pack("<Q", len(payload))
    return header + struct.pack("<I", masked_crc(header)) + payload + struct.pack("<I", masked_crc(payload))


def fixture_data(tmp_path):
    positions = np.arange(8 * 2 * 2, dtype=np.float32).reshape(8, 2, 2) / 100
    types = np.array([0, 5], dtype=np.int64)
    source = tmp_path / "train.tfrecord"
    source.write_bytes(record_bytes(positions, types))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"dim": 2, "sequence_length": 7}))
    output = tmp_path / "converted"
    return positions, types, source, metadata, output


def test_numeric_and_manifest_samples_match_legacy_data(tmp_path):
    positions, types, source, metadata, output = fixture_data(tmp_path)
    manifest = convert_split(source, output, metadata, "train", expected_records=1)
    np.savez(tmp_path / "numeric.npz", position_0=positions, type_0=types)
    legacy = np.empty(1, dtype=object)
    legacy[0] = (positions, types)
    np.savez(tmp_path / "legacy.npz", gns_data=legacy)
    datasets = [data_loader.SamplesDataset(path, 6, verify_hashes=True) for path in
                (output / "train.json", tmp_path / "numeric.npz", tmp_path / "legacy.npz")]
    assert [len(dataset) for dataset in datasets] == [2, 2, 2]
    for index in range(2):
        reference = datasets[0][index]
        for dataset in datasets[1:]:
            sample = dataset[index]
            for actual, expected in zip(sample[0], reference[0]):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(sample[1], reference[1])
    assert isinstance(datasets[0]._data, data_loader.ManifestTrajectories)
    assert isinstance(datasets[0]._data[0][0], np.memmap)
    assert not datasets[0]._data[0][0].flags.writeable
    assert len(data_loader.SamplesDataset(output / "train.npz", 6)) == 2
    assert manifest["records"][0]["source_key"] == [42]
    auxiliary = np.load(output / manifest["records"][0]["step_context"]["path"], allow_pickle=False)
    assert np.isnan(auxiliary).all()


@pytest.mark.parametrize("remove_bytes", [1, 4, 5, 20])
def test_truncated_sources_never_publish_a_manifest(tmp_path, remove_bytes):
    _, _, source, metadata, output = fixture_data(tmp_path)
    source.write_bytes(source.read_bytes()[:-remove_bytes])
    with pytest.raises(TFRecordIntegrityError, match="Truncated"):
        convert_split(source, output, metadata, "train", expected_records=1)
    assert not (output / "train.json").exists()
    assert not list(output.glob("*.staging"))


def test_payload_crc_corruption_is_detected_before_decoding(tmp_path):
    _, _, source, _, _ = fixture_data(tmp_path)
    raw = bytearray(source.read_bytes())
    raw[-5] ^= 1
    source.write_bytes(raw)
    with pytest.raises(TFRecordIntegrityError, match="payload CRC"):
        list(VerifiedTFRecords(source))


def test_length_crc_and_excessive_record_size_are_rejected(tmp_path):
    _, _, source, _, _ = fixture_data(tmp_path)
    raw = bytearray(source.read_bytes())
    raw[0] ^= 1
    source.write_bytes(raw)
    with pytest.raises(TFRecordIntegrityError, match="length CRC"):
        list(VerifiedTFRecords(source))
    header = struct.pack("<Q", 1024 * 1024)
    source.write_bytes(header + struct.pack("<I", masked_crc(header)))
    with pytest.raises(TFRecordIntegrityError, match="size guard"):
        list(VerifiedTFRecords(source, max_record_bytes=100))


def test_duplicate_trajectory_with_different_key_is_rejected_across_splits(tmp_path):
    positions, types, source, metadata, output = fixture_data(tmp_path)
    convert_split(source, output, metadata, "train", expected_records=1)
    valid = tmp_path / "valid.tfrecord"
    valid.write_bytes(record_bytes(positions, types, key=99))
    with pytest.raises(TFRecordIntegrityError, match="across splits"):
        convert_split(valid, output, metadata, "valid", expected_records=1)
    assert (output / "train.json").exists()
    assert not (output / "valid.json").exists()


def test_hash_checks_catch_modified_numeric_arrays(tmp_path):
    _, _, source, metadata, output = fixture_data(tmp_path)
    manifest = convert_split(source, output, metadata, "train", expected_records=1)
    array_path = output / manifest["records"][0]["positions"]["path"]
    modified = np.load(array_path, mmap_mode="r+")
    modified[0, 0, 0] = 99
    modified.flush()
    del modified
    with pytest.raises(ValueError, match="Array SHA256 mismatch"):
        data_loader.SamplesDataset(output / "train.json", 6, verify_hashes=True)


def test_manifest_cannot_escape_dataset_directory(tmp_path):
    _, _, source, metadata, output = fixture_data(tmp_path)
    manifest = convert_split(source, output, metadata, "train", expected_records=1)
    manifest["records"][0]["positions"]["path"] = "../outside.npy"
    (output / "train.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes dataset root"):
        data_loader.SamplesDataset(output / "train.json", 6)


def test_nonfinite_positions_do_not_publish_dataset(tmp_path):
    positions, types, source, metadata, output = fixture_data(tmp_path)
    positions[1, 0, 0] = np.nan
    source.write_bytes(record_bytes(positions, types))
    with pytest.raises(TFRecordIntegrityError, match="Nonfinite position"):
        convert_split(source, output, metadata, "train", expected_records=1)
    assert not (output / "train.json").exists()
