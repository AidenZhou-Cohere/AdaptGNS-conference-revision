"""Synthetic, stdlib-only tests; never import scientific modules or model data."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("portable_waterdrop", HERE / "continue_waterdrop.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ContinuationBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.adapter = HERE / "continue_waterdrop.py"
        self.identity = entry.parent_identity(1, "a" * 64, entry.sha(self.adapter))

    def valid_payload(self):
        return {"graph_support_schema": entry.PORTABLE_SCHEMA,
                "continuation_config": {"schema": entry.PORTABLE_SCHEMA,
                    "scope": entry.PORTABLE_SCOPE, "seed": 1,
                    "parent_checkpoint_sha256": "a" * 64,
                    "portable_reproduction": copy.deepcopy(self.identity),
                    "input_files_sha256": {"adapter.py": self.identity["adapter_sha256"]}}}

    def test_identity_retains_continuation_and_schedule(self):
        self.assertEqual(self.identity["absolute_schedule_steps"], [100000, 109999])
        self.assertEqual(self.identity["learning_rate"], 1e-5)
        self.assertFalse(self.identity["historical_study_receipt"])

    def test_invalid_hash_and_seed(self):
        for bad in (None, "A" * 64, "a" * 63, "g" * 64):
            with self.assertRaises(ValueError):
                entry.parent_identity(1, bad, "c" * 64)
        for bad in (True, -1, 3, 1.):
            with self.assertRaises(ValueError):
                entry.parent_identity(bad, "a" * 64, "c" * 64)

    def test_portable_endpoint_identity(self):
        self.assertEqual(entry.validate_portable_identity(self.valid_payload(), 1, "a" * 64), self.identity)

    def test_rejects_wrong_parent_or_adapter(self):
        payload = self.valid_payload()
        with self.assertRaises(ValueError):
            entry.validate_portable_identity(payload, 1, "b" * 64)
        with self.assertRaises(ValueError):
            entry.validate_portable_identity(payload, 1, "a" * 64, "b" * 64)

    def test_rejects_historical_schema(self):
        payload = self.valid_payload()
        payload["graph_support_schema"] = 1
        with self.assertRaises(ValueError):
            entry.validate_portable_identity(payload, 1, "a" * 64)

    def test_rejects_offset_or_adapter_pin_change(self):
        for change in ("schedule", "pin"):
            payload = self.valid_payload()
            if change == "schedule":
                payload["continuation_config"]["portable_reproduction"]["absolute_schedule_steps"] = [0, 9999]
            else:
                payload["continuation_config"]["input_files_sha256"] = {}
            with self.assertRaises(ValueError):
                entry.validate_portable_identity(payload, 1, "a" * 64)

    def test_cli_has_fixed_original_runtime_and_no_stop_override(self):
        args = entry.parse_args(["--repo", ".", "--parent-checkpoint", "parent.pt",
            "--parent-sha256", "a" * 64, "--train-manifest", "train.json",
            "--valid-manifest", "valid.json", "--metadata", "metadata.json",
            "--output-dir", "new-run", "--seed", "1", "--arm", "mix", "--resume"])
        argv = entry.original_argv(args)
        self.assertIn("--resume", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "mps")
        self.assertEqual(argv[argv.index("--threads") + 1], "2")
        self.assertNotIn("--stop-after", argv)
        self.assertNotIn("--clear-stale-lock", argv)

    def test_source_only_does_not_need_parent(self):
        args = entry.parse_args(["--repo", ".", "--check-source-equivalence"])
        self.assertTrue(args.check_source_equivalence)
        self.assertIsNone(args.parent_checkpoint)

    def test_boundary_keeps_numerical_function_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic_source.py"
            source.write_text("# synthetic source pin\n")
            old_source_hash = entry.SOURCE_SHA256
            try:
                entry.SOURCE_SHA256 = entry.sha(source)
                support = types.SimpleNamespace(__file__=str(source), SCHEMA=1,
                                                PARENT_HASHES={0: "d" * 64, 1: "e" * 64, 2: "f" * 64})
                for name in entry.CORE_FUNCTIONS:
                    setattr(support, name, object())
                original_objects = {name: getattr(support, name) for name in entry.CORE_FUNCTIONS}
                def configure():
                    return {"schema": support.SCHEMA, "scope": "original",
                            "seed": 1, "parent_checkpoint_sha256": "a" * 64,
                            "input_files_sha256": {"data": "b" * 64},
                            "numerical_field": {"lr": 1e-5, "batch_size": 2}}
                support.configure = configure
                entry.install_training_boundary(support, self.identity, self.adapter)
                self.assertTrue(all(getattr(support, name) is value for name, value in original_objects.items()))
                self.assertEqual(support.PARENT_HASHES, {1: "a" * 64})
                changed = support.configure()
                self.assertEqual(changed["numerical_field"], {"lr": 1e-5, "batch_size": 2})
                self.assertEqual(changed["scope"], entry.PORTABLE_SCOPE)
                self.assertEqual(changed["portable_reproduction"], self.identity)
                self.assertEqual(changed["input_files_sha256"]["data"], "b" * 64)
                with self.assertRaises(ValueError):
                    entry.install_training_boundary(support, self.identity, self.adapter)
            finally:
                entry.SOURCE_SHA256 = old_source_hash

    def test_import_path_has_no_scientific_import(self):
        tree = ast.parse(self.adapter.read_text())
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = []
        for node in top_imports:
            names += [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
        self.assertFalse(set(names) & {"torch", "numpy", "research", "gns"})


if __name__ == "__main__":
    unittest.main()
