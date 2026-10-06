"""Offline tiny synthetic tests; no published NPZ payload is opened."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import pickle
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

import repackage_designsafe_sand as rep

META = {"dim": 2, "bounds": [[0.1, 0.9], [0.1, 0.9]], "sequence_length": 320}
MANIFEST_KEYS = {"format", "version", "split", "dataset", "source", "metadata", "metadata_sha256", "record_count", "records", "converter_sha256"}
RECORD_KEYS = {"id", "source_index", "source_member", "positions", "particle_types", "trajectory_content_sha256", "logical_content_sha256"}


def envelope(positions=None, types=None, frames=9):
    if positions is None:
        positions = np.linspace(0.2, 0.8, frames * 6, dtype=np.float32).reshape(frames, 3, 2)
    if types is None:
        types = np.array([0, 1, 6], dtype=np.int64)
    value = np.empty(2, dtype=object)
    value[0], value[1] = positions, types
    return value


def payload(value, protocol=3):
    # NumPy 2 uses a new internal module spelling. The acquired archive uses
    # the legacy NumPy spelling; change only the synthetic fixture's GLOBAL.
    return pickle.dumps(value, protocol=protocol).replace(
        b"cnumpy._core.multiarray\n_reconstruct\n", b"cnumpy.core.multiarray\n_reconstruct\n")


def member(value=None, raw_payload=None, header=None):
    output = io.BytesIO()
    np.lib.format.write_array_header_1_0(output, header or {"descr": "|O", "fortran_order": False, "shape": (2,)})
    output.write(payload(envelope() if value is None else value) if raw_payload is None else raw_payload)
    return output.getvalue()


def make_zip(path, members, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, raw in members:
            archive.writestr(name, raw)
    with zipfile.ZipFile(path) as archive:
        inventory = [{"source_member_index": index, "name": item.filename, "compressed_bytes": item.compress_size,
                      "uncompressed_bytes": item.file_size, "listed_crc32": f"{item.CRC:08x}"}
                     for index, item in enumerate(archive.infolist())]
    return {"name": path.name, "saved_name": path.name, "received_bytes": path.stat().st_size,
            "expected_bytes": path.stat().st_size, "received_sha256": rep.sha(path),
            "status": "downloaded_and_inspected", "stage": "complete",
            "zip": {"member_count": len(inventory), "members": inventory}}


class RepackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sand_repackager_synthetic_")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_split(self, values, split="train", seen=None, names=None):
        names = names or [f"simulation_trajectory_{index}.npy" for index in range(len(values))]
        archive = self.root / f"{split}.npz"
        row = make_zip(archive, [(name, member(value)) for name, value in zip(names, values)])
        staging = self.root / f".{split}.staging"
        records, summary = rep.repackage_split(np, archive, split, row, staging, META,
                                                set() if seen is None else seen, len(values))
        return archive, staging, records, summary

    def test_float32_roundtrip_and_manifest_contract(self):
        original = envelope()
        archive, staging, records, summary = self.run_split([original])
        self.assertEqual(set(records[0]), RECORD_KEYS)
        self.assertEqual(summary["frame_lengths"], [9])
        self.assertTrue(summary["all_numeric_values_preserved_exact"])
        self.assertTrue(summary["ZIP_CRC_verified"])
        self.assertTrue(summary["source_eof_and_hash_verified"])
        for name, index in (("positions", 0), ("particle_types", 1)):
            desc = records[0][name]
            saved = staging / Path(desc["path"]).name
            restored = np.load(saved, allow_pickle=False)
            self.assertEqual(restored.dtype.str, original[index].dtype.str)
            self.assertEqual(restored.shape, original[index].shape)
            self.assertEqual(restored.tobytes(order="C"), original[index].tobytes(order="C"))
            self.assertEqual(rep.sha(saved), desc["sha256"])
            self.assertEqual(saved.stat().st_size, desc["size_bytes"])
        self.assertTrue(archive.exists())

    def test_nondefault_frame_length_float64_and_fortran_positions(self):
        positions = np.asfortranarray(np.linspace(0.2, 0.8, 331 * 6).reshape(331, 3, 2))
        original = envelope(positions, np.array([6, 6, 6], dtype=np.int64))
        _, staging, _, summary = self.run_split([original])
        restored = np.load(staging / "position_000000.npy", allow_pickle=False)
        types = np.load(staging / "type_000000.npy", allow_pickle=False)
        self.assertEqual(summary["frame_lengths"], [331])
        self.assertEqual(summary["position_dtypes"], [positions.dtype.str])
        self.assertEqual(restored.tobytes(order="C"), positions.tobytes(order="C"))
        self.assertEqual(types.shape, (3,))
        self.assertEqual(types.dtype.str, original[1].dtype.str)
        self.assertEqual(types.tobytes(), original[1].tobytes())

    def test_nonnative_serialized_dtype_refused_before_false_lossless_claim(self):
        for index in (0, 1):
            original = envelope()
            original[index] = original[index].astype(original[index].dtype.newbyteorder("S"))
            with self.subTest(index=index), self.assertRaisesRegex(rep.AdmissionError, "endian normalization"):
                rep.restricted_array(np, payload(original))

    def test_preserves_archive_order_not_lexical_names(self):
        values = [envelope(), envelope(positions=envelope()[0] + np.float32(0.001))]
        _, _, records, _ = self.run_split(values, names=["simulation_trajectory_9.npy", "simulation_trajectory_2.npy"])
        self.assertEqual([record["source_member"] for record in records], ["simulation_trajectory_9.npy", "simulation_trajectory_2.npy"])
        self.assertEqual([record["source_index"] for record in records], [0, 1])

    def test_type3_and_boundary_excursions_are_reported_without_clipping(self):
        value = envelope(types=np.array([3, 3, 6], dtype=np.int32))
        value[0][0, 0] = [-0.5, 1.4]
        _, staging, _, summary = self.run_split([value])
        self.assertEqual(summary["kinematic_type3_particles"], 2)
        self.assertEqual(summary["outside_boundary_particle_frames"], 1)
        self.assertEqual(summary["particle_type_ids"], [3, 6])
        np.testing.assert_array_equal(np.load(staging / "position_000000.npy", allow_pickle=False), value[0])

    def test_duplicate_inside_split_preserves_first_output_and_failure(self):
        archive = self.root / "train.npz"
        row = make_zip(archive, [(f"simulation_trajectory_{i}.npy", member()) for i in range(2)])
        staging = self.root / ".train.staging"
        with self.assertRaisesRegex(rep.AdmissionError, "Duplicate trajectory"):
            rep.repackage_split(np, archive, "train", row, staging, META, set(), 2)
        failure = json.loads((staging / "failure.json").read_text())
        self.assertEqual(failure["completed_members"], 1)
        self.assertEqual(failure["source_index"], 1)
        self.assertTrue((staging / "position_000000.npy").exists())
        self.assertEqual(len(json.loads((staging / "completed_member_details.json").read_text())), 1)
        self.assertEqual(rep.sha(archive), row["received_sha256"])

    def test_duplicate_across_splits_rejected(self):
        seen = set()
        self.run_split([envelope()], seen=seen)
        with self.assertRaisesRegex(rep.AdmissionError, "Duplicate trajectory"):
            self.run_split([envelope()], split="valid", seen=seen)
        self.assertEqual(json.loads((self.root / ".valid.staging" / "failure.json").read_text())["completed_members"], 0)

    def test_duplicate_definition_is_representation_sensitive(self):
        scalar = envelope(types=np.array(6, dtype=np.int64))
        vector = envelope(types=np.array([6, 6, 6], dtype=np.int64))
        self.assertNotEqual(rep.value_hash(scalar[1]), rep.value_hash(vector[1]))

    def test_scalar_validation_and_numeric_save_preserve_shape_and_dtype(self):
        value = envelope(types=np.array(6, dtype=">i8"))
        _, types, details = rep.validate_arrays(np, value, META)
        self.assertEqual(details["particle_type_counts"], {"6": 3})
        description = rep.save_array(np, self.root / "scalar.npy", "train", types)
        self.assertEqual(description["shape"], [])
        self.assertEqual(description["dtype"], ">i8")
        self.assertEqual(np.load(self.root / "scalar.npy", allow_pickle=False).tobytes(), types.tobytes())

    def test_unobserved_scalar_pickle_opcode_is_not_silently_admitted(self):
        # A NumPy scalar ndarray needs EMPTY_TUPLE, absent from the reviewed
        # first source-member profile. The decoder remains intentionally narrow.
        with self.assertRaisesRegex(rep.AdmissionError, "EMPTY_TUPLE"):
            rep.restricted_array(np, payload(envelope(types=np.array(6, dtype=np.int64))))

    def test_full_crc_failure_retained(self):
        archive = self.root / "train.npz"
        row = make_zip(archive, [("simulation_trajectory_0.npy", member())])
        contents = bytearray(archive.read_bytes())
        with zipfile.ZipFile(archive) as opened:
            info = opened.infolist()[0]
            start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        contents[start + 15] ^= 1
        archive.write_bytes(contents)
        # Let the source hash bind the corrupted synthetic bytes while preserving
        # the ZIP's original member CRC; this isolates the full CRC check.
        row["received_sha256"] = rep.sha(archive)
        staging = self.root / ".train.staging"
        with self.assertRaises(zipfile.BadZipFile):
            rep.repackage_split(np, archive, "train", row, staging, META, set(), 1)
        failure = json.loads((staging / "failure.json").read_text())
        self.assertEqual(failure["error_type"], "BadZipFile")
        self.assertIn("CRC", failure["reason"])
        self.assertEqual(failure["completed_members"], 0)

    def test_archive_hash_difference_refused(self):
        archive = self.root / "train.npz"
        row = make_zip(archive, [("simulation_trajectory_0.npy", member())])
        row["received_sha256"] = "0" * 64
        with self.assertRaisesRegex(rep.AdmissionError, "differs from complete acquisition"):
            rep.repackage_split(np, archive, "train", row, self.root / ".train.staging", META, set(), 1)
        self.assertFalse((self.root / ".train.staging").exists())

    def test_trailing_zip_bytes_refused(self):
        archive = self.root / "train.npz"
        row = make_zip(archive, [("simulation_trajectory_0.npy", member())])
        with archive.open("ab") as stream:
            stream.write(b"unexpected")
        row.update(received_bytes=archive.stat().st_size, received_sha256=rep.sha(archive))
        with self.assertRaisesRegex(rep.AdmissionError, "Bytes remain after ZIP"):
            rep.repackage_split(np, archive, "train", row, self.root / ".train.staging", META, set(), 1)

    def test_inventory_name_count_crc_and_order_mismatch_refused(self):
        for field, value in (("name", "simulation_trajectory_99.npy"), ("source_member_index", 1), ("listed_crc32", "00000000"), ("uncompressed_bytes", 9)):
            with self.subTest(field=field):
                archive = self.root / f"{field}.npz"
                row = make_zip(archive, [("simulation_trajectory_0.npy", member())])
                row["zip"]["members"][0][field] = value
                with self.assertRaisesRegex(rep.AdmissionError, "order/identity"):
                    rep.repackage_split(np, archive, "train", row, self.root / f".{field}.staging", META, set(), 1)

    def test_fresh_staging_required(self):
        _, staging, _, _ = self.run_split([envelope()])
        sentinel = staging / "sentinel"
        sentinel.write_text("preserved")
        with self.assertRaises(FileExistsError):
            self.run_split([envelope()])
        self.assertEqual(sentinel.read_text(), "preserved")

    def test_symlink_archive_refused(self):
        archive = self.root / "source.npz"
        row = make_zip(archive, [("simulation_trajectory_0.npy", member())])
        link = self.root / "train.npz"
        link.symlink_to(archive)
        with self.assertRaisesRegex(rep.AdmissionError, "regular non-symlink"):
            rep.repackage_split(np, link, "train", row, self.root / ".train.staging", META, set(), 1)

    def test_forbidden_global_never_executes(self):
        forbidden = b"\x80\x03cos\nsystem\nX\x04\x00\x00\x00true\x85R."
        with self.assertRaisesRegex(rep.AdmissionError, "global outside"):
            rep.restricted_array(np, forbidden)

    def test_protocol2_protocol4_and_trailing_pickle_bytes_refused(self):
        for raw in (payload(envelope(), protocol=2), payload(envelope(), protocol=4), payload(envelope()) + b"unused"):
            with self.subTest(raw_prefix=raw[:3]), self.assertRaises(rep.AdmissionError):
                rep.restricted_array(np, raw)

    def test_unreviewed_global_alias_and_opcode_refused(self):
        raw = payload(envelope())
        for changed in (raw.replace(b"cnumpy.core.multiarray", b"cnumpy._core.multiarray"), raw[:2] + b"0" + raw[2:]):
            with self.assertRaises(rep.AdmissionError):
                rep.inspect_pickle(changed)

    def test_invalid_or_auxiliary_outer_envelope_refused(self):
        for raw in (b"not-npy", member(header={"descr": "|O", "fortran_order": False, "shape": (3,)}), member(header={"descr": "<f4", "fortran_order": False, "shape": (2,)})):
            with self.assertRaises(rep.AdmissionError):
                rep.object_payload(raw)
        outer = np.empty(3, dtype=object)
        outer[:] = [np.ones((9, 3, 2)), np.array([6, 6, 6]), np.array([1])]
        with self.assertRaisesRegex(rep.AdmissionError, "Decoded outer"):
            rep.restricted_array(np, payload(outer))

    def test_invalid_arrays_fail_without_cast_or_temporal_rebuild(self):
        values = []
        for positions in (np.zeros((6, 3, 2)), np.zeros((9, 3, 3)), np.zeros((9, 3, 2), dtype=np.int64), np.full((9, 3, 2), np.nan)):
            values.append(envelope(positions=positions))
        for types in (np.array([6, 6]), np.array([6.0, 6.0, 6.0]), np.array([0, 3, 9]), np.array([-1, 2, 6]), np.zeros((1, 3), dtype=np.int64)):
            values.append(envelope(types=types))
        value = envelope()
        value[1] = [0, 1, 6]
        values.append(value)
        for value in values:
            with self.subTest(position_shape=value[0].shape, types=repr(value[1])), self.assertRaises(rep.AdmissionError):
                rep.validate_arrays(np, value, META)

    def test_roundtrip_verification_detects_changed_saved_values(self):
        original_save = np.save
        def corrupt_save(path, array, **kwargs):
            return original_save(path, array + 1, **kwargs)
        with mock.patch.object(np, "save", side_effect=corrupt_save), self.assertRaisesRegex(rep.AdmissionError, "changed dtype"):
            rep.save_array(np, self.root / "array.npy", "train", np.array([1.0], dtype=np.float64))

    def test_test_cli_and_duplicate_split_refused(self):
        base = ["--input-dir", "missing", "--acquisition-report", "missing", "--output-dir", "missing"]
        for splits in (["test"], ["train", "train"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                rep.parse_args(base + ["--splits"] + splits)
            self.assertEqual(raised.exception.code, 2)
        self.assertEqual(rep.parse_args(base + ["--splits", "valid", "train"]).splits, ["train", "valid"])

    def main_fixture(self, duplicate=False):
        input_dir = self.root / "input"
        input_dir.mkdir()
        metadata_bytes = json.dumps(META).encode().ljust(363, b" ")
        (input_dir / "metadata.json").write_bytes(metadata_bytes)
        metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
        metadata_row = {"name": "metadata.json", "saved_name": "metadata.json", "received_bytes": 363, "expected_bytes": 363,
                        "received_sha256": metadata_sha, "status": "downloaded_and_inspected", "stage": "complete"}
        values = {"train": envelope(), "valid": envelope(positions=envelope()[0] + np.float32(0 if duplicate else 0.01))}
        rows = {split: make_zip(input_dir / f"{split}.npz", [("simulation_trajectory_0.npy", member(value))]) for split, value in values.items()}
        receipt = {"schema": "sand_public_acquisition_v2", "status": "complete_with_object_dtypes", "version": 1,
                   "published_directory": rep.PUBLISHED_DIRECTORY, "dataset": "Sand", "project_id": "PRJ-3702", "doi": "10.17603/ds2-0phb-dg64",
                   "metadata_reference_sha256": metadata_sha, "files": [metadata_row, rows["train"], rows["valid"]]}
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        changes = {"METADATA_SHA": metadata_sha, "SOURCE_BYTES": {s: r["received_bytes"] for s, r in rows.items()},
                   "SOURCE_SHA": {s: r["received_sha256"] for s, r in rows.items()}, "COUNTS": {"train": 1, "valid": 1}}
        argv = ["--input-dir", str(input_dir), "--acquisition-report", str(receipt_path), "--output-dir", str(self.root / "output")]
        return argv, changes, receipt, receipt_path

    def test_main_publishes_exact_schema_after_both_splits_pass(self):
        argv, changes, _, _ = self.main_fixture()
        with mock.patch.multiple(rep, **changes), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rep.main(argv), 0)
        output = self.root / "output"
        report = json.loads((output / "structural_report.json").read_text())
        self.assertEqual(report["status"], "complete_structural_only")
        self.assertFalse(report["scientific_training_admission"])
        self.assertFalse(report["test_split_accessed"])
        self.assertIn("dtype, shape", report["duplicate_definition"])
        for split in ("train", "valid"):
            path = output / f"{split}.json"
            manifest = json.loads(path.read_text())
            self.assertEqual(set(manifest), MANIFEST_KEYS)
            self.assertEqual(set(manifest["records"][0]), RECORD_KEYS)
            self.assertEqual(report["splits"][split]["manifest_sha256"], rep.sha(path))
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual(manifest["records"][0]["positions"]["shape"][0], 9)
            self.assertTrue((output / manifest["records"][0]["positions"]["path"]).exists())
            self.assertFalse((output / f".{split}.staging").exists())

    def test_main_failure_keeps_staging_and_publishes_no_manifest(self):
        argv, changes, _, _ = self.main_fixture(duplicate=True)
        with mock.patch.multiple(rep, **changes), self.assertRaisesRegex(rep.AdmissionError, "Duplicate trajectory"):
            rep.main(argv)
        output = self.root / "output"
        report = json.loads((output / "structural_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["scientific_training_admission"])
        self.assertTrue((output / ".train.staging" / "position_000000.npy").exists())
        self.assertTrue((output / ".valid.staging" / "failure.json").exists())
        self.assertFalse((output / "train.json").exists())
        self.assertFalse((output / "valid.json").exists())

    def test_main_existing_output_refused_without_overwriting(self):
        argv, changes, _, _ = self.main_fixture()
        output = self.root / "output"
        output.mkdir()
        (output / "sentinel").write_text("keep")
        with mock.patch.multiple(rep, **changes), self.assertRaises(FileExistsError):
            rep.main(argv)
        self.assertEqual((output / "sentinel").read_text(), "keep")
        self.assertEqual(list(output.iterdir()), [output / "sentinel"])

    def test_receipt_family_identity_and_completion_enforced(self):
        _, changes, receipt, _ = self.main_fixture()
        variants = []
        for key, value in (("schema", "other"), ("version", 2), ("published_directory", "/other/"), ("status", "partial"), ("dataset", "WaterDrop"), ("project_id", "other"), ("doi", "other")):
            changed = copy.deepcopy(receipt)
            changed[key] = value
            variants.append(changed)
        for row_index in (0, 1):
            for key, value in (("status", "failed"), ("stage", "partial"), ("saved_name", "other"), ("received_bytes", 0), ("received_sha256", "0" * 64)):
                changed = copy.deepcopy(receipt)
                changed["files"][row_index][key] = value
                variants.append(changed)
        with mock.patch.multiple(rep, **changes):
            rep.validate_receipt(receipt, ["train", "valid"])
            for changed in variants:
                with self.subTest(receipt=changed), self.assertRaises(rep.AdmissionError):
                    rep.validate_receipt(changed, ["train", "valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
