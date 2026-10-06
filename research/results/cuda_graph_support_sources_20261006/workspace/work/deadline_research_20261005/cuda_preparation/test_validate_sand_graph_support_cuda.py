"""Only width8, one-block synthetic CPU networks and mocked selected-CUDA RNG."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_train_sand_graph_support_cuda import synthetic

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "validate_sand_graph_support_cuda.py"
spec = importlib.util.spec_from_file_location("_graph_cuda_validator_test", PATH)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
trainer = entry.import_pinned(ROOT, "train_sand_graph_support_cuda.py")
old = entry.import_pinned(ROOT, "train_sand_cuda_deterministic.py")
compare = entry.import_pinned(ROOT, "validate_cuda_execution.py")


def batch_for(s):
    return dict(zip(("position_sequence", "particle_types", "nparticles_per_example", "next_positions"), s.batch),
                position_sequence_noise=s.helpers.host_noise(s.batch[0].shape, s.batch[1], 0, 0))


def test_description_imports_no_scientific_code():
    code = """import builtins, runpy, sys
old = builtins.__import__
def limited(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy', 'scipy', 'gns', 'research'):
        raise AssertionError(name)
    return old(name, *args, **kwargs)
builtins.__import__ = limited
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(PATH)], capture_output=True, text=True, check=True)
    description = json.loads(result.stdout)
    assert description["seed"] == 0 and description["max_optimizer_updates"] == 15
    assert description["scientific_training_admitted"] is False
    assert description["prior_cpu_cuda_validation_status"] == "validation_failed"


def test_cli_is_fixed_and_requires_saved_inputs():
    assert entry.parse_args([]).execute is False
    for args in (["--execute"], ["--seed", "1"], ["--objective", "nll"], ["--threads", "0"], ["--cuda-index", "-1"]):
        with pytest.raises(SystemExit):
            entry.parse_args(args)


def test_numeric_and_runtime_helpers_are_identical_and_pins_reject_change(tmp_path):
    assert entry.shared_function_check(ROOT)
    for name, digest in entry.PINS.items():
        assert entry.sha(ROOT / name) == digest
    path = tmp_path / "train_sand_graph_support_cuda.py"
    path.write_text("raise AssertionError('must not import')")
    with pytest.raises(ValueError, match="Pinned helper"):
        entry.import_pinned(tmp_path, path.name)


@pytest.mark.parametrize("graph_step", [0, 1, 2])
def test_native_base_exact_and_mix_graph_schedule_is_independent_of_adam_step(synthetic, graph_step):
    s, results = synthetic, {}
    batch = batch_for(s)
    initial_model, _ = s.make()
    initial = s.helpers.cpu_tree(initial_model.state_dict())
    initial_rng = trainer.capture_rng(s.torch, "mock-cuda")
    for arm in entry.BRANCHES:
        model, opt = s.make()
        model.load_state_dict(initial)
        trainer.restore_rng(s.torch, initial_rng, "mock-cuda")
        result, check, ledger = entry.advance(s.helpers, old if arm == "old_native" else trainer,
            s.support, compare, model, opt, batch, batch["position_sequence_noise"], "cpu", 0, arm, graph_step=graph_step)
        assert check["passed"]
        results[arm] = result
        if ledger is not None:
            assert all(row["coin_seed_material"][2] == graph_step for row in ledger)
    assert entry.branch_comparison(s.helpers, trainer, compare, results["old_native"], results["base"])["passed"]
    assert compare.diff(s.torch, results["base"]["before"]["target"], results["mix"]["before"]["target"], "exact")["passed"]


@pytest.mark.parametrize("arm", ["base", "mix"])
def test_new_schema_serialization_and_resumed_uninterrupted_replay_exact(synthetic, monkeypatch, tmp_path, arm):
    s = synthetic
    cuda = s.torch.Generator(device="cpu").manual_seed(0)
    monkeypatch.setattr(s.torch.cuda, "get_rng_state", lambda device: cuda.get_state())
    monkeypatch.setattr(s.torch.cuda, "set_rng_state", lambda state, device: cuda.set_state(state))
    def probe(h, device):
        return {"cpu": h.torch.rand(17), "cuda": h.torch.rand(17, generator=cuda)}
    monkeypatch.setattr(entry, "rng_probe", probe)
    initial_model, _ = s.make()
    initial = s.helpers.cpu_tree(initial_model.state_dict())
    initial_rng = trainer.capture_rng(s.torch, "cpu")
    def make(state):
        model, _ = s.make()
        model.load_state_dict(state)
        return model
    case = {"trajectory_ids": ["synthetic:0", "synthetic:1"], "target_frames": [6, 6]}
    result = entry.replay(s.helpers, trainer, s.support, compare, make, initial, initial_rng,
                         batch_for(s), case, "cpu", arm, tmp_path)
    assert result["passed"] and result["final_payload_exact"] and result["graph_ledger_exact"]
    assert result["checkpoint_snapshot_history_length"] == 1
    checkpoint = s.torch.load(tmp_path / result["checkpoint"]["file"], weights_only=True)
    assert checkpoint["cuda_sand_graph_support_schema"] == trainer.SCHEMA
    assert checkpoint["run_config"]["seed"] == 0 and checkpoint["completed_steps"] == 1
    assert set(checkpoint["rng_states"]) == {"cpu", "cuda"}
    assert len(checkpoint["history"]["graph_updates"]) == 1


def test_bytewise_comparison_rejects_signed_zero_and_non_tensor_adam_metadata(synthetic):
    s = synthetic
    assert not entry.exact(s.helpers, compare, {"x": s.torch.tensor([0.])}, {"x": s.torch.tensor([-0.])})["passed"]
    model, opt = s.make()
    batch = batch_for(s)
    result, _, _ = entry.advance(s.helpers, trainer, s.support, compare, model, opt, batch,
                                batch["position_sequence_noise"], "cpu", 0, "base")
    changed = copy.deepcopy(result)
    changed["optimizer"]["param_groups"][0]["eps"] = 1e-7
    check = entry.branch_comparison(s.helpers, trainer, compare, result, changed)
    assert not check["passed"] and not check["optimizer_metadata_exact"]


def test_graph_audit_rejects_tampered_ledger_and_edge_order(synthetic):
    s = synthetic
    model, _ = s.make()
    batch = batch_for(s)
    noisy = batch["position_sequence"] + batch["position_sequence_noise"]
    counts = batch["nparticles_per_example"]
    _, native, _ = model._encoder_preprocessor(noisy, counts, batch["particle_types"], None)
    actual, ledger = s.support.append_optional_edges(noisy, counts, native, .015, 0, 0, "mix")
    broken = copy.deepcopy(ledger)
    broken[0]["pair_seed_material"][-1] = 12
    assert not entry.graph_audit(s.helpers, trainer, s.support, compare, noisy, counts,
                                native, actual, broken, 0, "mix", "cpu")["passed"]
    assert not entry.graph_audit(s.helpers, trainer, s.support, compare, noisy, counts,
                                native, actual.flip(1), ledger, 0, "mix", "cpu")["passed"]


def good_report():
    return {"shared_trainer_functions_exact": True,
            "cases": [{"case": case, "branches": {arm: {"passed": True, "graph": {"examples": [
                {"selected_optional_pairs": 1 if arm == "mix" else 0}]}} for arm in entry.BRANCHES},
                "old_native_vs_base": {"passed": True}, "mix_target_exact": {"passed": True}} for case in entry.CASES],
            "replay": {arm: {"passed": True} for arm in entry.ARMS}}


@pytest.mark.parametrize("mutation", [
    lambda r: r["cases"].pop(),
    lambda r: r["replay"].pop("mix"),
    lambda r: r["cases"][0]["branches"].pop("old_native"),
    lambda r: r["cases"][0]["old_native_vs_base"].update(passed=False),
    lambda r: r["cases"][0]["mix_target_exact"].update(passed=False),
    lambda r: r["replay"]["mix"].update(passed=False),
    lambda r: r.update(shared_trainer_functions_exact=False),
    lambda r: [row["branches"]["mix"]["graph"]["examples"][0].update(selected_optional_pairs=0) for row in r["cases"]],
])
def test_completion_gate_rejects_partial_or_failed_evidence(mutation):
    report = good_report()
    assert entry.completion_gate(report)
    mutation(report)
    assert not entry.completion_gate(report)


def test_existing_evidence_is_never_overwritten(synthetic, tmp_path):
    path = tmp_path / "proof.pt"
    path.write_bytes(b"prior failure evidence")
    with pytest.raises(FileExistsError):
        entry.save_evidence(synthetic.torch, path, {})
    assert path.read_bytes() == b"prior failure evidence"


def test_guarded_failure_retains_current_numeric_evidence(synthetic, monkeypatch):
    s = synthetic
    model, opt = s.make()
    batch = batch_for(s)
    def fail(*args):
        raise FloatingPointError("synthetic guard failure")
    monkeypatch.setattr(trainer, "guarded_update", fail)
    with pytest.raises(FloatingPointError) as caught:
        entry.advance(s.helpers, trainer, s.support, compare, model, opt, batch,
                      batch["position_sequence_noise"], "cpu", 0, "mix")
    proof = caught.value.graph_validation_evidence
    assert proof["arm"] == "mix" and proof["step"] == 0
    assert set(proof["before"]) == {"prediction", "variance", "target"}
    assert proof["graph_check"]["passed"] and proof["model_state_at_failure"]
