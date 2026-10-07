"""Synthetic filesystem/stream tests only; no SSH or numerical array loading."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transfer_committed", HERE / "transfer_committed.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class Transfers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.src = self.root / "source"
        self.dst = self.root / "destination"
        self.src.mkdir()
        self.dst.mkdir()
        self.ns = {"__name__": "transfer_synthetic"}
        exec(compile(helper.REMOTE, "<transfer_remote>", "exec"), self.ns)
        self.host = {"hostname": "source-host", "boot_id": "source-boot"}
        self.binding = {"workers": [4], "label": "synthetic", "source_identity": self.host}
        self.request = {"binding": self.binding}
        self.worker = self.src / "worker_04"
        self.worker.mkdir()
        self.cell = "base_seed1__test__000000__random25"
        self.partial_cell = "base_seed1__test__000001__random25"
        self.identity = {"schema": self.ns["SCHEMA"] + "_identity", "plan_sha256": self.ns["PLAN_SHA"],
                         "worker_index": 4, "hostname": self.host["hostname"], "worker_source_sha256": "a" * 64,
                         "cell_ids": [self.cell, self.partial_cell]}
        self.identity_sha = self.save(self.worker / "worker_identity.json", self.identity)
        protocol = {"schema": self.ns["SCHEMA"], "plan_sha256": self.ns["PLAN_SHA"], "worker_index": 4,
                    "hostname": self.host["hostname"], "worker_source_sha256": "a" * 64,
                    "cell_ids": self.identity["cell_ids"], "identity_sha256": self.identity_sha}
        self.protocol_sha = self.save(self.worker / "protocol.json", protocol)
        attempt = self.worker / "attempts" / "attempt_000001"
        for name in self.ns["ATTEMPT_FILES"]:
            self.save(attempt / name, {"name": name, "immutable": True})
        self.save(self.worker / "owner.json", {"live": True})
        self.save(self.worker / "status.json", {"live": True})
        self.save(attempt / "status.json.tmp.growing", {"excluded": True})
        self.artifact = self.worker / "cells" / self.cell / "attempt_000001" / "trace.npz"
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(b"synthetic opaque NPZ transport bytes\0" * 4096)
        artifact_sha = self.ns["file_hash"](self.artifact)
        row = {"completion_cell_id": self.cell, "protocol_sha256": self.protocol_sha,
               "artifact_sha256": artifact_sha, "artifact_file": "trace.npz", "status": "failed",
               "failure": {"category": "synthetic_guard"}}
        row_sha = self.save(self.artifact.parent / "row.json", row)
        marker = {"schema": self.ns["SCHEMA"] + "_cell_commit", "cell_id": self.cell,
                  "plan_sha256": self.ns["PLAN_SHA"], "identity_sha256": self.identity_sha,
                  "protocol_sha256": self.protocol_sha, "attempt": "attempt_000001",
                  "row_file": "attempt_000001/row.json", "row_sha256": row_sha,
                  "artifact_file": "attempt_000001/trace.npz", "artifact_sha256": artifact_sha}
        self.marker_path = self.artifact.parent.parent / "commit.json"
        self.save(self.marker_path, marker)
        partial = self.worker / "cells" / self.partial_cell / "attempt_000001"
        partial.mkdir(parents=True)
        (partial / "trace.npz.tmp.growing").write_bytes(b"do not read")
        self.partial = partial

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = helper.encode(value)
        path.write_bytes(raw)
        return self.ns["digest"](raw)

    def inventory(self, root, verify=False):
        return self.ns["inventory"](root, [4], self.host, verify)

    def manifest(self):
        return helper.choose_manifest(self.inventory(self.src), self.inventory(self.dst, True), self.binding)

    def stream(self, manifest):
        output = io.BytesIO()
        self.ns["pack"](manifest, self.src, output, self.request)
        output.seek(0)
        self.assertEqual(json.loads(output.readline()), manifest)
        return output

    def receive(self, manifest, data=None, label="attempt1"):
        attempt = self.dst / "_transfer_attempts" / label
        attempt.mkdir(parents=True)
        return self.ns["receive"](manifest, self.dst, data or self.stream(manifest), attempt, self.request)

    def test_source_inventory_reads_only_committed_and_published_metadata(self):
        original = self.ns["file_hash"]
        read_paths = []

        def tracked(path):
            read_paths.append(Path(path))
            self.assertNotIn(self.partial, Path(path).parents)
            self.assertNotEqual(Path(path), self.artifact)  # Stat only until transfer.
            return original(path)

        self.ns["file_hash"] = tracked
        inventory = self.inventory(self.src)
        self.assertEqual(list(inventory["workers"][0]["cells"]), [self.cell])
        names = [e["path"] for e in inventory["workers"][0]["metadata"]]
        self.assertEqual(len(names), 2 + len(self.ns["ATTEMPT_FILES"]))
        self.assertNotIn("worker_04/owner.json", names)
        self.assertNotIn("worker_04/status.json", names)
        self.assertFalse(any("tmp" in n for n in names))

    def test_roundtrip_marker_last_and_incremental_skip(self):
        manifest = self.manifest()
        original = self.ns["os"].link
        publications = []

        def link(src, dst):
            publications.append(str(Path(dst).relative_to(self.dst)))
            if Path(dst).name == "commit.json":
                self.assertTrue((Path(dst).parent / "attempt_000001" / "trace.npz").is_file())
                self.assertTrue((Path(dst).parent / "attempt_000001" / "row.json").is_file())
            return original(src, dst)

        with mock.patch.object(self.ns["os"], "link", side_effect=link):
            records = self.receive(manifest)
        self.assertEqual(publications[-1], str(self.marker_path.relative_to(self.src)))
        self.assertEqual(len(records), len(manifest["files"]))
        self.assertEqual(self.inventory(self.src), {**self.inventory(self.dst, True), "root": str(self.src)})
        second = self.manifest()
        self.assertEqual(second["files"], [])
        self.assertEqual(second["already_verified_cells_skipped"], 1)
        # Failed scientific outcome transfers just like a successful one.
        row = json.loads((self.dst / self.artifact.parent.relative_to(self.src) / "row.json").read_bytes())
        self.assertEqual(row["status"], "failed")

    def test_destination_corruption_is_rejected_before_retransfer(self):
        self.receive(self.manifest())
        (self.dst / self.artifact.relative_to(self.src)).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            self.inventory(self.dst, True)

    def test_path_and_marker_reference_rejection(self):
        for value in ("../outside", "/absolute", "a/../b", "a//b", "a/./b", "a\\b", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.ns["relative"](value)
        marker = json.loads(self.marker_path.read_bytes())
        marker["artifact_file"] = "../../outside.npz"
        self.save(self.marker_path, marker)
        with self.assertRaisesRegex(ValueError, "marker reference"):
            self.inventory(self.src)

    def test_symlink_reference_rejected(self):
        original = self.root / "outside.npz"
        self.artifact.rename(original)
        self.artifact.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.inventory(self.src)

    def test_wrong_identity_or_commit_binding_rejected(self):
        marker = json.loads(self.marker_path.read_bytes())
        marker["identity_sha256"] = "b" * 64
        self.save(self.marker_path, marker)
        with self.assertRaisesRegex(ValueError, "commit identity"):
            self.inventory(self.src)

    def test_received_bad_sha_leaves_no_marker_and_retains_partial(self):
        manifest = self.manifest()
        data = self.stream(manifest)
        raw = data.getvalue()
        offset = raw.index(b"synthetic opaque NPZ transport bytes")
        corrupted = raw[:offset] + b"X" + raw[offset + 1:]
        incoming = io.BytesIO(corrupted)
        incoming.readline()
        with self.assertRaisesRegex(ValueError, "received file SHA"):
            self.receive(manifest, incoming)
        target_marker = self.dst / self.marker_path.relative_to(self.src)
        self.assertFalse(target_marker.exists())
        staged = list((self.dst / "_transfer_attempts" / "attempt1").rglob("*.part"))
        self.assertTrue(any(p.name == "trace.npz.part" for p in staged))
        self.receive(self.manifest(), label="attempt2")
        self.assertTrue(target_marker.exists())
        self.assertTrue(all(p.exists() for p in staged))

    def test_truncated_stream_preserves_staging_then_recovers(self):
        manifest = self.manifest()
        data = self.stream(manifest)
        raw = data.getvalue()
        start = raw.index(b"synthetic opaque NPZ transport bytes")
        incoming = io.BytesIO(raw[:start + 1200])
        incoming.readline()
        with self.assertRaises((ValueError, self.ns["tarfile"].ReadError)):
            self.receive(manifest, incoming)
        self.assertFalse((self.dst / self.marker_path.relative_to(self.src)).exists())
        self.assertTrue(list((self.dst / "_transfer_attempts" / "attempt1").rglob("*.part")))
        self.receive(self.manifest(), label="attempt2")
        self.assertTrue((self.dst / self.marker_path.relative_to(self.src)).exists())

    def test_existing_uncommitted_conflict_is_not_overwritten(self):
        manifest = self.manifest()
        target = self.dst / self.artifact.relative_to(self.src)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing differing partial")
        with self.assertRaisesRegex(ValueError, "different destination bytes"):
            self.receive(manifest)
        self.assertEqual(target.read_bytes(), b"existing differing partial")
        self.assertFalse((self.dst / self.marker_path.relative_to(self.src)).exists())

    def test_marker_first_and_extra_paths_rejected(self):
        manifest = self.manifest()
        marker = manifest["files"].pop()
        manifest["files"].insert(0, marker)
        with self.assertRaisesRegex(ValueError, "marker must follow"):
            self.ns["validate_manifest"](manifest, self.request)
        manifest = self.manifest()
        manifest["files"][0]["path"] = "worker_04/status.json"
        with self.assertRaisesRegex(ValueError, "unsafe metadata"):
            self.ns["validate_manifest"](manifest, self.request)

    def test_host_boot_mismatch_rejected(self):
        self.ns["machine_identity"] = lambda: self.host
        self.assertEqual(self.ns["verify_host"](self.host), self.host)
        with self.assertRaisesRegex(ValueError, "host/boot"):
            self.ns["verify_host"]({**self.host, "boot_id": "wrong"})

    def test_duplicate_json_and_changed_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.ns["strict"](b'{"a": 1, "a": 2}')
        self.receive(self.manifest())
        original = self.inventory(self.src)
        changed = copy.deepcopy(self.inventory(self.dst, True))
        changed["workers"][0]["metadata"][0]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different destination metadata"):
            helper.choose_manifest(original, changed, self.binding)


if __name__ == "__main__":
    unittest.main(verbosity=2)
