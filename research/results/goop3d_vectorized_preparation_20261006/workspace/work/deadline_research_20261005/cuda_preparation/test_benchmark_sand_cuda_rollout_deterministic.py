"""Synthetic contract/orchestration tests only; no models or actual Sand data."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sand_benchmark", HERE / "benchmark_sand_cuda_rollout_deterministic.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture():
    records, details = [], []
    for index, particles in enumerate((8, 3, 12)):
        arrays = {"positions": {"path": f"valid/p{index}.npy", "shape": [9, particles, 2], "dtype": "<f4", "size_bytes": 200, "sha256": h(f"p{index}")},
                  "particle_types": {"path": f"valid/t{index}.npy", "shape": [], "dtype": "<i8", "size_bytes": 136, "sha256": h(f"t{index}")}}
        logical = hashlib.sha256((h(f"pv{index}") + ":" + h(f"tv{index}")).encode()).hexdigest()
        records.append({"id": f"valid:{index:06d}", "source_index": index, "source_member": f"simulation_trajectory_{index}.npy",
                        **arrays, "trajectory_content_sha256": hashlib.sha256((arrays["positions"]["sha256"] + ":" + arrays["particle_types"]["sha256"]).encode()).hexdigest(),
                        "logical_content_sha256": logical})
        details.append({"source_index": index, "source_member": records[-1]["source_member"], "frames": 9, "particles": particles,
                        "position_dtype": "<f4", "particle_type_dtype": "<i8", "particle_type_shape": [],
                        "numeric_dtype_shape_values_verified_exact": True, "source_position_value_sha256": h(f"pv{index}"),
                        "source_type_value_sha256": h(f"tv{index}")})
    manifest = {"format": "gns-trajectory-manifest", "version": 1, "split": "valid", "dataset": "Sand",
        "source": {"family": "designsafe_published_npz", "dataset": "Sand", "file": "valid.npz", "size_bytes": 100,
                   "sha256": h("source"), "ZIP_CRC_verified": True, "member_count": 3, "acquisition_report_sha256": h("receipt")},
        "metadata": {"dim": 2}, "metadata_sha256": bench.METADATA_SHA, "record_count": 3, "records": records, "converter_sha256": h("converter")}
    admission = {"schema": bench.ADMISSION_SCHEMA, "status": "admitted_for_feasibility", "dataset": "Sand", "split": "valid",
        "manifest_sha256": h("manifest"), "structural_report_sha256": h("structural"), "metadata_sha256": bench.METADATA_SHA,
        "converter_sha256": h("converter"), "frames_per_trajectory": 9, "record_count": 3, "particle_type_ids": [6], "position_dtype": "<f4"}
    report = {"schema": "designsafe_sand_numeric_repackage_v1", "status": "complete_structural_only", "dataset": "Sand",
        "source_family": "designsafe_published_npz", "metadata_sha256": bench.METADATA_SHA, "converter_sha256": h("converter"),
        "acquisition_report_sha256": h("receipt"), "splits": {"valid": {"manifest_sha256": h("manifest"), "record_count": 3,
            "frame_lengths": [9], "position_dtypes": ["<f4"], "particle_type_ids": [6], "kinematic_type3_particles": 0,
            "ZIP_CRC_verified": True, "all_numeric_values_preserved_exact": True, "records": details}}}
    return manifest, admission, report


class ContractTests(unittest.TestCase):
    def test_default_is_description_only_and_imports_no_helpers(self):
        with mock.patch.object(bench, "load_helpers", side_effect=AssertionError("must not import numerical helpers")), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(bench.main([]), 0)
        self.assertFalse(json.loads(out.getvalue())["execution"])

    def test_execution_requires_explicit_model_and_all_paths(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(["--execute"])

    def test_test_mode_and_test_split_refused_before_any_read(self):
        for args in (["--execute", "--mode", "locked-test"], ["--split", "test"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.parse_args(args)

    def test_checkpoint_mode_requires_preselected_identity(self):
        base = ["--execute", "--repo", "/repo", "--manifest", "/manifest", "--admission", "/admission", "--structural-report", "/report",
                "--protocol", "/protocol", "--output-dir", "/output", "--split", "valid", "--model-kind", "checkpoint", "--objective", "faithful", "--seed", "0"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(base)
        args = bench.parse_args(base + ["--checkpoint", "/checkpoint", "--checkpoint-sha256", h("checkpoint"), "--checkpoint-updates", "0", "--trainer-source", "/trainer"])
        self.assertEqual(args.checkpoint_updates, 0)

    def test_initialization_refuses_checkpoint_selection(self):
        args = ["--execute", "--repo", "/repo", "--manifest", "/manifest", "--admission", "/admission", "--structural-report", "/report",
                "--protocol", "/protocol", "--output-dir", "/output", "--split", "valid", "--model-kind", "initialized", "--objective", "nll", "--seed", "2", "--checkpoint", "/checkpoint"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(args)

    def test_arbitrary_source_t_and_size_selection(self):
        manifest, admission, report = fixture()
        with mock.patch.dict(bench.SOURCES, valid=(3, 100, h("source"))):
            self.assertEqual(bench.validate_contract(manifest, admission, report, "valid"), 9)
        self.assertEqual([row["source_index"] for row in bench.select_cases(manifest["records"])], [1, 0, 2])

    def test_size_ties_are_source_order_and_lower_median(self):
        manifest, _, _ = fixture()
        records = [copy.deepcopy(manifest["records"][0]) for _ in range(4)]
        for i, record in enumerate(records):
            record.update(source_index=i, id=f"valid:{i:06d}")
        self.assertEqual([row["source_index"] for row in bench.select_cases(records[::-1])], [0, 1, 3])

    def test_incomplete_or_changed_source_contract_refused(self):
        mutations = [lambda m,a,r: m["source"].update(family="different"),
                     lambda m,a,r: m["source"].update(ZIP_CRC_verified=False),
                     lambda m,a,r: a.update(status="admitted"),
                     lambda m,a,r: a.update(frames_per_trajectory=321),
                     lambda m,a,r: a.update(particle_type_ids=[3]),
                     lambda m,a,r: a.update(particle_type_ids=[True]),
                     lambda m,a,r: m["records"][0]["positions"].update(dtype="<f8"),
                     lambda m,a,r: m["records"][0].update(material_property={}),
                     lambda m,a,r: m["records"][0].update(source_index=True),
                     lambda m,a,r: m["records"][0].update(source_member="simulation_trajectory_2.npy"),
                     lambda m,a,r: r.update(status="failed"),
                     lambda m,a,r: r["splits"]["valid"]["records"][0].update(particles=77),
                     lambda m,a,r: r["splits"]["valid"]["records"][0].update(source_position_value_sha256=h("wrong")),
                     lambda m,a,r: m.update(split="test")]
        with mock.patch.dict(bench.SOURCES, valid=(3, 100, h("source"))):
            for i, mutate in enumerate(mutations):
                m, a, r = fixture()
                mutate(m, a, r)
                with self.subTest(mutation=i), self.assertRaises(ValueError):
                    bench.validate_contract(m, a, r, "valid")

    def test_duplicate_numeric_array_and_logical_content_refused(self):
        with mock.patch.dict(bench.SOURCES, valid=(3, 100, h("source"))):
            for field in ("path", "logical"):
                m, a, r = fixture()
                if field == "path":
                    m["records"][1]["positions"]["path"] = m["records"][0]["positions"]["path"]
                else:
                    m["records"][1]["logical_content_sha256"] = m["records"][0]["logical_content_sha256"]
                with self.subTest(field=field), self.assertRaises(ValueError):
                    bench.validate_contract(m, a, r, "valid")

    def test_source_hash_change_fails_before_import(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(bench, "SOURCE_PINS", {"module.py": h("expected")}):
            (Path(directory) / "module.py").write_text("print('must not execute')")
            with mock.patch.object(bench.importlib, "import_module", side_effect=AssertionError("must not import")), self.assertRaisesRegex(ValueError, "source mismatch"):
                bench.load_helpers(directory)

    def test_atomic_json_preserves_existing_uncommitted_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.with_suffix(".json.tmp").write_text("failed write")
            with self.assertRaises(FileExistsError):
                bench.atomic_json(path, {"new": True})
            self.assertEqual(path.with_suffix(".json.tmp").read_text(), "failed write")

    def test_existing_output_refused_without_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (output / "preserve").write_text("earlier attempt")
            args = ["--execute", "--repo", str(root / "repo"), "--manifest", str(root / "numeric/valid.json"),
                    "--admission", str(root / "admission.json"), "--structural-report", str(root / "report.json"),
                    "--protocol", str(root / "protocol.md"), "--output-dir", str(output), "--split", "valid",
                    "--model-kind", "initialized", "--objective", "faithful", "--seed", "0"]
            with mock.patch.object(bench, "DEADLINE", datetime.now(timezone.utc) + timedelta(days=1)), mock.patch.object(bench, "sha", side_effect=AssertionError("must not read")):
                with self.assertRaises(FileExistsError):
                    bench.main(args)
            self.assertEqual([p.name for p in output.iterdir()], ["preserve"])


class CheckpointTests(unittest.TestCase):
    def model_fixture(self):
        _, admission, _ = fixture()
        args = SimpleNamespace(checkpoint_updates=1000, objective="faithful", seed=0)
        config = {"schema": bench.TRAINING_SCHEMA, "dataset": "Sand", "objective": "faithful", "seed": 0, "updates": 100000,
            "runtime": {"deterministic_algorithms": True, "deterministic_warn_only": False, "cublas_workspace_config": ":4096:8",
                        "tf32": False, "amp": False, "compile": False, "ddp": False},
            "data": {"metadata_sha256": bench.METADATA_SHA, "frames_per_trajectory": admission["frames_per_trajectory"],
                     "particle_type_ids": admission["particle_type_ids"], "converter_sha256": admission["converter_sha256"],
                     "structural_report_sha256": admission["structural_report_sha256"], "source": {"sha256": bench.SOURCES["train"][2]}},
            "source_sha256": {**bench.SOURCE_PINS, "train_sand_cuda_deterministic.py": h("trainer")},
            "batch_size": 2, "history": 6, "noise_std": 6.7e-4,
            "architecture": {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
            "graph": {"radius": .015, "backend": "scipy_host", "cap": 128, "self_candidates": True, "augmentation_probability": 0.0},
            "optimizer": {"name": "Adam", "initial_lr": 1e-4, "final_lr": 1e-5, "decay_updates": 100000,
                          "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0, "foreach": False, "fused": False, "gradient_clipping": None}}
        payload = {"format_version": 2, "cuda_sand_training_schema": bench.TRAINING_SCHEMA, "run_config": config,
                   "run_config_sha256": bench.canonical_hash(config), "completed_steps": 1000,
                   "training_config": {"loss": "faithful", "cuda_sand_run": config, "completed_optimizer_updates": 1000}}
        return payload, args, admission

    def test_exact_preselected_checkpoint_contract(self):
        payload, args, admission = self.model_fixture()
        self.assertEqual(bench.check_checkpoint_contract(payload, args, admission, h("trainer")), payload["run_config"])

    def test_changed_checkpoint_family_recipe_or_endpoint_refused(self):
        mutations = [lambda p: p.update(cuda_sand_training_schema="waterdrop"),
                     lambda p: p.update(completed_steps=999),
                     lambda p: p["run_config"].update(seed=False),
                     lambda p: p["run_config"].update(objective="nll"),
                     lambda p: p["run_config"].update(updates=10000),
                     lambda p: p["run_config"]["data"].update(frames_per_trajectory=321),
                     lambda p: p["run_config"]["data"]["source"].update(sha256=h("different source")),
                     lambda p: p["run_config"]["source_sha256"].update({"train_sand_cuda_deterministic.py": h("different trainer")}),
                     lambda p: p["run_config"]["source_sha256"].update({"research/full_training.py": h("different source")}),
                     lambda p: p["run_config"]["graph"].update(cap=64),
                     lambda p: p["run_config"]["optimizer"].update(gradient_clipping=1.0)]
        for i, mutation in enumerate(mutations):
            payload, args, admission = self.model_fixture()
            mutation(payload)
            payload["run_config_sha256"] = bench.canonical_hash(payload["run_config"])
            with self.subTest(mutation=i), self.assertRaises(ValueError):
                bench.check_checkpoint_contract(payload, args, admission, h("trainer"))

    def test_run_configuration_hash_is_enforced(self):
        payload, args, admission = self.model_fixture()
        payload["run_config_sha256"] = h("different")
        with self.assertRaises(ValueError):
            bench.check_checkpoint_contract(payload, args, admission, h("trainer"))

    def test_nondeterministic_or_warn_only_checkpoint_refused(self):
        for change in ({"deterministic_algorithms": False}, {"deterministic_warn_only": True},
                       {"cublas_workspace_config": ":16:8"}, {"tf32": True}, {"amp": True}):
            payload, args, admission = self.model_fixture()
            payload["run_config"]["runtime"].update(change)
            payload["run_config_sha256"] = bench.canonical_hash(payload["run_config"])
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "deterministic CUDA variant"):
                bench.check_checkpoint_contract(payload, args, admission, h("trainer"))

    def test_v1_checkpoint_schema_never_admitted(self):
        payload, args, admission = self.model_fixture()
        payload["cuda_sand_training_schema"] = "adaptgns_sand_cuda_training_v1"
        payload["run_config"]["schema"] = "adaptgns_sand_cuda_training_v1"
        payload["run_config_sha256"] = bench.canonical_hash(payload["run_config"])
        with self.assertRaises(ValueError):
            bench.check_checkpoint_contract(payload, args, admission, h("trainer"))

    def test_missing_workspace_environment_refused_before_cuda(self):
        for environment in ({}, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            with self.subTest(environment=environment), mock.patch.dict(bench.os.environ, environment, clear=True), self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
                bench.configure_cuda(SimpleNamespace(), SimpleNamespace())

    def test_configuration_requests_strict_determinism_and_records_it(self):
        deterministic = mock.Mock()
        torch = SimpleNamespace(__version__="2.13.0+cu129", version=SimpleNamespace(cuda="12.9"),
            use_deterministic_algorithms=deterministic, device=lambda kind,index: f"{kind}:{index}",
            set_num_threads=mock.Mock(), set_float32_matmul_precision=mock.Mock(),
            are_deterministic_algorithms_enabled=lambda: True, is_deterministic_algorithms_warn_only_enabled=lambda: False,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)), cudnn=SimpleNamespace(allow_tf32=True, benchmark=True)),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, set_device=mock.Mock(),
                get_device_properties=lambda device: SimpleNamespace(name="synthetic GB200", uuid="synthetic")))
        helpers = SimpleNamespace(torch=torch, np=SimpleNamespace(__version__="synthetic"), scipy=SimpleNamespace(__version__="synthetic"))
        with mock.patch.dict(bench.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            device, runtime = bench.configure_cuda(helpers, SimpleNamespace(cuda_index=0, threads=2))
        deterministic.assert_called_once_with(True, warn_only=False)
        self.assertEqual(device, "cuda:0")
        self.assertTrue(runtime["deterministic_algorithms"])
        self.assertFalse(runtime["deterministic_warn_only"])
        self.assertEqual(runtime["cublas_workspace_config"], ":4096:8")


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.manifest, _, _ = fixture()
        self.cases = bench.select_cases(self.manifest["records"])
        self.args = SimpleNamespace(output_dir=self.output, max_seconds=60, seed=1)
        self.trajectories = [(np.zeros((9,n,2), dtype=np.float32), np.array(6, dtype=np.int64)) for n in (8,3,12)]
        self.calls = []
        self.failure = None
        self.raise_error = None
        def rollout(model, positions, types, metadata, policy, horizon, seed, device, trace_steps):
            self.calls.append((policy, horizon, seed, trace_steps, len(positions)))
            if self.raise_error:
                raise self.raise_error
            failed = self.failure is not None and len(self.calls) == 1
            steps = 1 if failed else horizon
            return {"horizon": horizon, "policy": policy, "status": "failed" if failed else "complete", "completed_steps": steps,
                    "failure": {"category": self.failure, "forecast_step": 2} if failed else None,
                    "mean_rollout_mse": None if failed else 0.25, "mse_per_step": [0.25] * steps}, {"prediction": np.zeros((steps, len(types.reshape(-1)), 2), dtype=np.float32)}
        self.native = SimpleNamespace(np=np, full=SimpleNamespace(synchronize=lambda device: None), rollout=rollout)
        self.deadline = mock.patch.object(bench, "DEADLINE", datetime.now(timezone.utc) + timedelta(days=1))
        self.deadline.start()
        self.addCleanup(self.deadline.stop)

    def run_cases(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.run_cases(self.native, object(), self.trajectories, self.manifest, self.cases, self.args, "fake", h("protocol"), time.perf_counter())

    def test_all_policies_full_source_horizon_and_saved_artifacts(self):
        rows, state = self.run_cases()
        self.assertEqual((len(rows), state), (15, "complete"))
        self.assertEqual({call[1] for call in self.calls}, {3})
        self.assertEqual({call[3] for call in self.calls}, {(1, 3)})
        self.assertEqual([call[0] for call in self.calls[:5]], list(bench.POLICIES))
        self.assertEqual({call[2] for call in self.calls[:5]}, {94001})
        result = json.loads((self.output / "result.json").read_text())
        for policy in bench.POLICIES:
            self.assertTrue(result["summary"][policy]["all_cases_full_horizon"])
            self.assertFalse(result["summary"][policy]["full_horizon_timing_forecast_admitted"])
        for row in rows:
            self.assertEqual(bench.sha(self.output / row["trace_file"]), row["trace_sha256"])
            self.assertEqual(row["mse_at_declared_trace_steps"], {"1": .25, "3": .25})

    def test_guard_failure_retained_and_no_survivor_replacement(self):
        self.failure = "coordinate_resource_guard"
        rows, state = self.run_cases()
        self.assertEqual(state, "complete_with_guard_failures")
        self.assertEqual(len(self.calls), 15)
        result = json.loads((self.output / "result.json").read_text())
        base = result["summary"]["base"]
        self.assertEqual((base["complete_cases"], base["failed_cases"]), (2, 1))
        self.assertIsNone(base["equal_case_mean_rollout_mse"])
        self.assertIsNone(base["max_full_rollout_wall_seconds_including_parity"])
        self.assertEqual(rows[0]["failure"]["category"], self.failure)

    def test_parity_failure_stops_after_preserving_raw_case(self):
        self.failure = "native_parity_failure"
        with self.assertRaisesRegex(RuntimeError, "parity failure"):
            self.run_cases()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(list(self.output.glob("trajectory_*.npz")))
        self.assertFalse(json.loads((self.output / "failed_attempt.json").read_text())["retry_performed"])

    def test_unhandled_error_preserves_identity_and_missing_outcomes(self):
        self.raise_error = RuntimeError("synthetic backend failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic backend"):
            self.run_cases()
        error = json.loads((self.output / "failed_attempt.json").read_text())
        self.assertEqual(error["case"]["policy"], "base")
        self.assertEqual(error["case"]["source_index"], 1)
        self.assertEqual(error["error_type"], "RuntimeError")
        result = json.loads((self.output / "result.json").read_text())
        self.assertEqual(result["summary"]["base"]["missing_cases"], 3)

    def test_elapsed_budget_stops_before_any_native_case(self):
        sync = mock.Mock(side_effect=AssertionError("Expired budget must not touch CUDA"))
        with mock.patch.object(self.native.full, "synchronize", sync), self.assertRaises(TimeoutError):
            with contextlib.redirect_stdout(io.StringIO()):
                bench.run_cases(self.native, object(), self.trajectories, self.manifest, self.cases, self.args, "fake", h("protocol"), time.perf_counter() - 61)
        sync.assert_not_called()
        self.assertFalse(self.calls)
        self.assertTrue((self.output / "failed_attempt.json").exists())

    def test_initial_sync_occurs_inside_same_alarm_as_rollout(self):
        active = []
        @contextlib.contextmanager
        def alarm(seconds):
            self.assertGreater(seconds, 0)
            active.append(True)
            try:
                yield
            finally:
                active.pop()
        sync = mock.Mock(side_effect=lambda device: self.assertEqual(active, [True]))
        with mock.patch.object(bench, "deadline_alarm", alarm), mock.patch.object(self.native.full, "synchronize", sync):
            self.run_cases()
        self.assertEqual(sync.call_count, 30)

    def test_duplicate_and_unexpected_summary_rows_refused(self):
        row = {"source_index": 1, "policy": "base", "status": "complete", "completed_steps": 3,
               "mean_rollout_mse": .25, "synchronized_call_seconds": .1}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bench.summarize_cases([row, row], self.cases, 3)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            bench.summarize_cases([{**row, "source_index": 123}], self.cases, 3)


if __name__ == "__main__":
    unittest.main()
