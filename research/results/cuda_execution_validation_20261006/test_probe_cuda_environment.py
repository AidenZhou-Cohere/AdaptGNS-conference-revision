"""Small CPU-only CLI/output checks; no real dependency imports or GPU calls."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("cuda_probe", Path(__file__).with_name("probe_cuda_environment.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeChecks(unittest.TestCase):
    def test_default_and_explicit_metadata_do_not_request_smoke(self):
        for argv in ([], ["--metadata-only"]):
            self.assertFalse(probe.parse_args(argv).synthetic_smoke)

    def test_smoke_is_explicit_and_flags_are_exclusive(self):
        self.assertTrue(probe.parse_args(["--synthetic-smoke"]).synthetic_smoke)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args(["--metadata-only", "--synthetic-smoke"])

    def test_unknown_or_abbreviated_arguments_are_refused(self):
        for argv in (["--network"], ["--synthetic"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parse_args(argv)

    def test_missing_torch_has_no_fallback(self):
        cuda = probe.cuda_metadata(None)
        self.assertFalse(cuda["available"])
        self.assertEqual(probe.synthetic_cuda_smoke(None, cuda)["status"], "not_run")

    def test_incompatible_torch_stays_a_structured_query_failure(self):
        cuda = probe.cuda_metadata(object())
        self.assertEqual(cuda["status"], "query_failed")
        self.assertFalse(cuda["available"])
        self.assertEqual(cuda["error_type"], "AttributeError")

    def test_metadata_does_not_call_smoke_and_prints_json(self):
        with patch.object(probe, "import_dependencies", return_value=([], {})), \
             patch.object(probe, "host_metadata", return_value={"os": "test"}), \
             patch.object(probe, "synthetic_cuda_smoke", side_effect=AssertionError("must not run")), \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            code = probe.main([])
        report = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["mode"], "metadata-only")
        self.assertEqual(report["synthetic_smoke"]["status"], "not_requested")
        self.assertEqual(set(report), {"schema", "mode", "host", "dependencies", "cuda", "synthetic_smoke", "scope", "missing_core_imports"})

    def test_smoke_status_controls_exit(self):
        for status, expected in (("passed", 0), ("not_run", 2), ("failed", 1)):
            self.assertEqual(probe.exit_code({"mode": "synthetic-smoke", "synthetic_smoke": {"status": status}}), expected)


if __name__ == "__main__":
    unittest.main()
