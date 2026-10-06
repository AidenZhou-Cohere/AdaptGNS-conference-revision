"""CPU/stdlib checks only. Never import Torch or run a simulator."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("cuda_execution_validation", Path(__file__).with_name("validate_cuda_execution.py"))
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ValidationControlChecks(unittest.TestCase):
    def test_adam_requires_complete_nonempty_parameter_state(self):
        class Tensor:
            def __init__(self, step=None, device="cuda:0"):
                self.device = device if step is None else types.SimpleNamespace(type="cpu")
                self.shape, self.dtype, self.step = (2,), "float32", step
            def __int__(self):
                return self.step
            def __float__(self):
                return float(self.step)
            def numel(self):
                return 1
        parameters = [Tensor(), Tensor()]
        state = {parameter: {"step": Tensor(step=2), "exp_avg": Tensor(), "exp_avg_sq": Tensor()} for parameter in parameters}
        self.assertTrue(validator.adam_state_complete(parameters, state, 2))
        self.assertFalse(validator.adam_state_complete([], {}, 2))
        self.assertFalse(validator.adam_state_complete(parameters, {}, 2))
        self.assertFalse(validator.adam_state_complete(parameters, {parameters[0]: state[parameters[0]]}, 2))
        self.assertFalse(validator.adam_state_complete(parameters, state, 1))
        state[parameters[0]]["step"].step = 2.5
        self.assertFalse(validator.adam_state_complete(parameters, state, 2))
        state[parameters[0]]["step"].step = 2
        state[parameters[0]]["step"].numel = lambda: 2
        self.assertFalse(validator.adam_state_complete(parameters, state, 2))
        state[parameters[0]]["step"].numel = lambda: 1
        state[parameters[0]]["exp_avg"].device = "cpu"
        self.assertFalse(validator.adam_state_complete(parameters, state, 2))

    def test_description_default_does_not_execute(self):
        with patch.object(validator, "execute", side_effect=AssertionError("execution forbidden")), contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(validator.main([]), 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["status"], "description_only")
        self.assertFalse(report["scientific_readiness"])
        self.assertEqual(report["architecture"], {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2})

    def test_execution_requires_explicit_flag_and_repo(self):
        self.assertFalse(validator.parse_args(["--repo", "/unused"]).execute)
        for argv in (["--execute"], ["--execute", "--describe", "--repo", "/unused"], ["--exec"], ["--cuda-index", "-1"], ["--threads", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validator.parse_args(argv)
        self.assertTrue(validator.parse_args(["--execute", "--repo", "/unused"]).execute)

    def test_predeclared_tolerances_cannot_be_set_by_cli(self):
        self.assertEqual(validator.TOLERANCES["forward"], {"atol": 2e-5, "rtol": 2e-4})
        self.assertEqual(validator.TOLERANCES["gradient"], {"atol": 5e-5, "rtol": 5e-4})
        self.assertEqual(validator.TOLERANCES["replay"], {"atol": 5e-6, "rtol": 5e-5})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validator.parse_args(["--atol", "1"])

    def test_numeric_gate_distinguishes_description_from_acceptance(self):
        self.assertTrue(validator.numeric_gate(True, 1.0))
        for value in (math.nextafter(1.0, math.inf), math.inf, math.nan, None):
            self.assertFalse(validator.numeric_gate(True, value))
        self.assertFalse(validator.numeric_gate(False, 0.0))
        self.assertFalse(validator.numeric_gate(True, 0.0, exact=True, bitwise_equal=False))
        self.assertTrue(validator.numeric_gate(True, None, exact=True, bitwise_equal=True))

    def test_pass_requires_all_exactly_named_sections(self):
        sections = {name: {"passed": True} for name in validator.REQUIRED}
        self.assertEqual(validator.complete_status(sections), "validation_passed")
        for name in validator.REQUIRED:
            self.assertEqual(validator.complete_status({key: value for key, value in sections.items() if key != name}), "validation_failed")
            self.assertEqual(validator.complete_status({**sections, name: {"passed": False}}), "validation_failed")
        self.assertEqual(validator.complete_status({**sections, "extra": {"passed": True}}), "validation_failed")

    def test_source_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "adaptive-gns" / "gns"
            root.mkdir(parents=True)
            path = root / "sample.py"
            path.write_bytes(b"original\n")
            expected = {"sample.py": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(validator.verify_core(directory, expected)[1], expected)
            path.write_bytes(b"changed\n")
            with self.assertRaises(ValueError):
                validator.verify_core(directory, expected)

    def test_source_error_precedes_dependency_import(self):
        args = types.SimpleNamespace(repo=Path("/unused"))
        with patch.object(validator, "verify_core", side_effect=ValueError("private path must not appear")):
            report = validator.execute(args)
        self.assertEqual(report["failure"], {"stage": "source_verification", "error_type": "ValueError"})
        self.assertNotIn("private path", json.dumps(report))
        self.assertEqual(report["sections"], {})

    def test_wrong_rng_backend_refused_before_state_mutation(self):
        fake = types.SimpleNamespace(set_rng_state=Mock(), cuda=types.SimpleNamespace(set_rng_state=Mock()))
        with self.assertRaises(ValueError):
            validator.rng_restore(fake, {"cpu": b"state"}, "cuda:0")
        fake.set_rng_state.assert_not_called()
        fake.cuda.set_rng_state.assert_not_called()
        validator.rng_restore(fake, {"cpu": b"cpu", "cuda": b"cuda"}, "cuda:2")
        fake.set_rng_state.assert_called_once_with(b"cpu")
        fake.cuda.set_rng_state.assert_called_once_with(b"cuda", "cuda:2")

    def test_exit_codes_distinguish_failure_unavailable_and_pass(self):
        for status, code in (("validation_passed", 0), ("validation_failed", 1), ("not_run", 2)):
            with patch.object(validator, "execute", return_value={"status": status}), contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(validator.main(["--execute", "--repo", "/unused"]), code)
            self.assertEqual(json.loads(stream.getvalue())["status"], status)


if __name__ == "__main__":
    unittest.main()
