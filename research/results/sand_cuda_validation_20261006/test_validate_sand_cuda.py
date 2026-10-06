"""Actual-data wrapper logic tested with fabricated tensors and mocked CUDA only."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wrapper = load("_tested_actual_sand_validator", ROOT / "validate_sand_cuda.py")


def records(counts=(8, 2, 5, 2, 7, 4, 6, 3), frames=320):
    return [{"id": f"train:{i:06d}", "positions": {"shape": [frames, n, 2]}}
            for i, n in enumerate(counts)]


def test_default_description_imports_no_runtime():
    code = """import builtins, runpy, sys
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'scipy', 'gns'}:
        raise AssertionError('default imported runtime: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "validate_sand_cuda.py")],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "description_only"


@pytest.mark.parametrize("frames", [14, 320, 321])
def test_schedule_uses_source_order_ties_and_actual_middle_target(frames):
    source = records(frames=frames)
    result = wrapper.case_schedule(source)
    assert [row["trajectory_indices"] for row in result] == [[1, 3], [2, 6], [4, 0]]
    assert [row["case"] for row in result] == ["small", "median", "large"]
    for row in result:
        assert row["target_frames"] == [6, frames // 2]
        for source_index, target, flat_index in zip(row["trajectory_indices"], row["target_frames"], row["dataset_indices"]):
            trajectory, local = divmod(flat_index, frames - 6)
            assert (trajectory, local + 6) == (source_index, target)
        assert row["trajectory_ids"] == [source[i]["id"] for i in row["trajectory_indices"]]


@pytest.mark.parametrize("source", [records()[:5], records(frames=11), records(frames=12), records(counts=(1, 2, 0, 4, 5, 6))])
def test_schedule_rejects_ineligible_or_degenerate_source(source):
    with pytest.raises(ValueError):
        wrapper.case_schedule(source)


def test_schedule_rejects_mixed_frame_lengths():
    source = records()
    source[0]["positions"]["shape"][0] = 321
    with pytest.raises(ValueError):
        wrapper.case_schedule(source)


def complete_report():
    return {"cases": [{"case": name, "objectives": {objective: {"passed": True} for objective in wrapper.OBJECTIVES}}
                      for name in wrapper.CASE_NAMES],
            "replay": {objective: {"passed": True} for objective in wrapper.OBJECTIVES}}


def test_complete_gate_rejects_vacuous_partial_false_and_duplicate_results():
    assert wrapper.complete_gate(complete_report())
    assert not wrapper.complete_gate({})
    mutations = [lambda report: report["cases"].pop(),
                 lambda report: report["cases"][0]["objectives"].pop("nll"),
                 lambda report: report["replay"].pop("nll"),
                 lambda report: report["cases"][0].update(case="large"),
                 lambda report: report["cases"][0]["objectives"]["faithful"].update(passed=False),
                 lambda report: report["replay"]["faithful"].update(passed=1)]
    for mutate in mutations:
        report = complete_report()
        mutate(report)
        assert not wrapper.complete_gate(report)


def test_bad_helper_hash_never_imports(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="identity mismatch"):
        wrapper.import_file("_unexecuted_bad_helper", path, "0" * 64)


def test_cli_rejects_missing_arguments_or_invalid_runtime_counts():
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--execute"])
    for values in (["--cuda-index", "-1"], ["--threads", "0"], ["--threads", "17"]):
        with pytest.raises(SystemExit):
            wrapper.parse_args(values)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """CPU linear models; CUDA RNG is an independent CPU Generator in this mock."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    trainer = load("_toy_sand_trainer", ROOT / "train_sand_cuda.py")
    compare = load("_toy_sand_compare", ROOT / "validate_cuda_execution.py")
    cuda_rng = torch.Generator(device="cpu").manual_seed(123)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: cuda_rng.manual_seed(seed))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: cuda_rng.get_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: cuda_rng.set_state(value))
    original_rand = torch.rand
    probe_calls = []

    def alternating_probe(*args, **kwargs):
        # advance() calls CPU then CUDA probes; both tensor devices are CPU in this mock.
        stream = "cpu" if len(probe_calls) % 2 == 0 else "cuda"
        probe_calls.append(stream)
        if stream == "cuda":
            kwargs["generator"] = cuda_rng
        return original_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", alternating_probe)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Linear(2, 1)
            self.variance = torch.nn.Linear(2, 1)
            self._checkpoint_config = {"toy": True, "normalization": torch.tensor([0.2, 0.3])}

        def _encoder_preprocessor(self, positions, counts, types):
            indices = torch.arange(len(positions))
            return positions[:, -1], torch.stack([indices, indices]), torch.ones(len(positions), 3)

    source = records(frames=20)
    eligible = 14
    dataset = list(range(len(source) * eligible))

    def unpack(items):
        counts = [source[index // eligible]["positions"]["shape"][1] for index in items]
        positions = torch.cat([torch.full((n, 6, 2), (index + 1) * .001) for index, n in zip(items, counts)])
        return positions, torch.full((sum(counts),), 5), torch.tensor(counts), torch.full((sum(counts), 1), .05)

    noise_calls = []

    def host_noise(shape, types, seed, step):
        noise_calls.append({"seed": seed, "step": step})
        return torch.full(shape, step * 1e-4)

    def cpu_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu_tree(item) for item in value)
        return value

    def finite(values):
        values = list(values)
        return bool(values) and all(bool(torch.isfinite(value).all()) for value in values)

    def finite_gradients(model):
        if not finite(p.grad for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Nonfinite toy gradient")

    def forward(model, raw, noise, device):
        inputs = raw[0][:, -1] + noise[:, -1]
        return model.mean(inputs), model.variance(inputs).exp(), raw[3]

    def loss(pred, target, mask, pred_variance, **kwargs):
        return ((pred[mask] - target[mask]).square() / pred_variance[mask] + pred_variance[mask].log()).mean()

    h = SimpleNamespace(torch=torch, np=np, NOISE=6.7e-4, KINEMATIC=3,
                        build_simulator=lambda *a, **k: ToyModel(), unpack_batch=unpack,
                        host_noise=host_noise, cpu_tree=cpu_tree, forward_batch=forward,
                        tensors_are_finite=finite, assert_finite_gradients=finite_gradients,
                        acceleration_loss=loss, learning_rate=lambda step: 1e-4 * .1 ** (min(step / 99999, 1.0)),
                        synchronize=lambda device: None)
    parity = Mock(return_value={"passed": True, "actual_batch_ordered_edges": {"passed": True},
                               "scope": "mock wrapper delegation only"})
    compare.objective_checks = parity
    monkeypatch.setattr(trainer, "load_helpers", Mock(return_value=h))
    monkeypatch.setattr(trainer, "configure_cuda", Mock(return_value=(torch.device("cpu"), {"scope": "mocked CUDA"})))
    monkeypatch.setattr(trainer, "load_admitted_dataset", Mock(return_value=(dataset, {"dim": 2}, {"mock": True})))
    monkeypatch.setattr(wrapper, "import_file", lambda name, path, expected: trainer if name == "_sand_trainer_validation" else compare)
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"records": source}))
    (tmp_path / "metadata.json").write_text('{"dim":2}')
    admission, structural = tmp_path / "admission.json", tmp_path / "structural.json"
    admission.write_text('{"scope":"mock admission"}')
    structural.write_text('{"scope":"mock structural report"}')
    args = SimpleNamespace(output_dir=tmp_path / "validation", trainer=ROOT / "train_sand_cuda.py",
                           trainer_sha256=wrapper.sha(ROOT / "train_sand_cuda.py"), repo=tmp_path / "not_loaded",
                           train_manifest=manifest, admission=admission, structural_report=structural,
                           cuda_index=0, threads=1)
    return SimpleNamespace(torch=torch, trainer=trainer, compare=compare, h=h, args=args,
                           dataset=dataset, records=source, parity=parity, probe_calls=probe_calls, noise_calls=noise_calls)


def test_wrapper_exercises_all_actual_schedule_cases_and_preserves_batch_artifacts(toy):
    assert wrapper.execute(toy.args) == 0
    report = json.loads((toy.args.output_dir / "report.json").read_text())
    assert report["status"] == "validation_passed" and report["scientific_training"] is False
    assert wrapper.complete_gate(report)
    assert toy.parity.call_count == 6
    assert [call.args[5] for call in toy.parity.call_args_list] == ["faithful", "nll"] * 3
    for row in report["cases"]:
        assert wrapper.sha(toy.args.output_dir / row["batch_file"]) == row["batch_sha256"]
        assert row["objectives"]["faithful"]["encoder_features"]["tensors"].keys() == {"/node_features", "/edge_features"}
    assert toy.trainer.load_admitted_dataset.call_count == 1
    for result in report["replay"].values():
        assert result["restore_completed_steps"] == 1 and result["first_update_changed_parameters"]
        assert result["rng_streams_advanced"] == {"cpu": True, "cuda": True}
        assert result["rng_probes_exact"]["passed"] and result["resumed_rng_exact"]
    assert len(toy.probe_calls) == 12  # Two streams, three update calls, two objectives.
    assert sum(row == {"seed": 0, "step": 1} for row in toy.noise_calls) == 5


def test_failed_parity_is_a_failed_gate_not_an_exception(toy):
    toy.parity.return_value = {"passed": False, "actual_batch_ordered_edges": {"passed": False}}
    assert wrapper.execute(toy.args) == 2
    report = json.loads((toy.args.output_dir / "report.json").read_text())
    assert report["status"] == "validation_failed" and not wrapper.complete_gate(report)
    assert len(report["cases"]) == 3 and len(report["replay"]) == 2


def test_replay_restore_failure_keeps_completed_parity_and_specific_phase(toy, monkeypatch):
    monkeypatch.setattr(toy.trainer, "restore_payload", Mock(side_effect=ValueError("synthetic restore rejection")))
    with pytest.raises(ValueError, match="restore rejection"):
        wrapper.execute(toy.args)
    report = json.loads((toy.args.output_dir / "report.json").read_text())
    assert report["status"] == "validation_error" and report["failed_attempt_preserved"]
    assert report["cases"][0]["objectives"]["faithful"]["passed"]
    failed = report["replay"]["faithful"]
    assert failed["phase"] == "checkpoint_restore" and failed["passed"] is False
    assert failed["status"] == "validation_error" and failed["serialized_exact"]
    assert failed["error"] == "synthetic restore rejection"
    assert wrapper.sha(toy.args.output_dir / failed["checkpoint_file"]) == failed["checkpoint_sha256"]
    assert not wrapper.complete_gate(report)


def test_numeric_replay_failure_reports_failed_result_and_continues_cases(toy, monkeypatch):
    actual_diff = toy.compare.tensor_map_diff

    def altered_diff(torch, reference, candidate, tolerance):
        result = actual_diff(torch, reference, candidate, tolerance)
        if tolerance == "replay":
            result["passed"] = False
        return result

    monkeypatch.setattr(toy.compare, "tensor_map_diff", altered_diff)
    assert wrapper.execute(toy.args) == 2
    report = json.loads((toy.args.output_dir / "report.json").read_text())
    assert report["status"] == "validation_failed" and len(report["cases"]) == 3
    assert all(row["passed"] is False and row["phase"] == "complete" for row in report["replay"].values())


def test_structural_input_change_cannot_finish_as_passed(toy):
    def parity(*args):
        if toy.parity.call_count == 6:
            toy.args.structural_report.write_text('{"changed":true}')
        return {"passed": True}

    toy.parity.side_effect = parity
    with pytest.raises(ValueError, match="inputs changed"):
        wrapper.execute(toy.args)
    report = json.loads((toy.args.output_dir / "report.json").read_text())
    assert report["status"] == "validation_error"


def test_existing_output_is_never_overwritten(toy):
    toy.args.output_dir.mkdir()
    sentinel = toy.args.output_dir / "report.json"
    sentinel.write_text('{"old":true}')
    with pytest.raises(FileExistsError):
        wrapper.execute(toy.args)
    assert sentinel.read_text() == '{"old":true}'
