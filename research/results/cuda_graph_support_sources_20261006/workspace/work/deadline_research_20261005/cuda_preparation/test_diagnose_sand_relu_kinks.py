"""Tiny CPU network tests for the causal diagnostic; no source data/GNS/CUDA."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = load("_kink_test_module", ROOT / "diagnose_sand_relu_kinks.py")


def test_default_does_not_import_torch_or_repository():
    code = """import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','numpy','scipy','gns','research'}: raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "diagnose_sand_relu_kinks.py")],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["training_admitted"] is False


def test_forced_mask_changes_only_derivative_at_crossed_signs():
    import torch
    function = diagnostic.make_mask_function(torch)
    cpu = torch.tensor([1e-8, -1e-8, 0.0, 2.0])
    gpu_like = torch.tensor([-1e-8, 1e-8, 0.0, 2.0], requires_grad=True)
    forced = function.apply(gpu_like, cpu > 0)
    assert torch.equal(forced.view(torch.uint8), torch.relu(gpu_like).view(torch.uint8))
    forced.backward(torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(gpu_like.grad, torch.tensor([2.0, 0.0, 0.0, 5.0]))


def test_natural_mask_sham_matches_ordinary_relu_gradient():
    import torch
    function = diagnostic.make_mask_function(torch)
    original = torch.tensor([-1e-30, 0.0, 1e-30, 2.0], requires_grad=True)
    sham = original.detach().clone().requires_grad_()
    weights = torch.tensor([-2.0, 3.0, -4.0, 5.0])
    (torch.relu(original) * weights).sum().backward()
    (function.apply(sham, sham.detach() > 0) * weights).sum().backward()
    assert torch.equal(original.grad, sham.grad)


def test_shared_relu_calls_are_keyed_by_invocation_and_restored_after_error():
    import torch
    class Reused(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()
        def forward(self, x):
            return self.relu(x) + self.relu(-x)
    model = Reused()
    x = torch.tensor([-1.0, 1.0], requires_grad=True)
    with diagnostic.capture_relu_inputs(torch, model) as events:
        model(x)
    assert [(row["name"], row["call"]) for row in events] == [("relu", 0), ("relu", 1)]
    original = model.relu.forward
    with pytest.raises(RuntimeError, match="test interruption"):
        with diagnostic.force_relu_backward(torch, model, events):
            model(x)
            raise RuntimeError("test interruption")
    assert model.relu.forward == original
    assert torch.equal(model(x), x.abs())


def test_capture_budget_removes_hooks_and_inplace_is_rejected():
    import torch
    model = torch.nn.Sequential(torch.nn.ReLU())
    with pytest.raises(ValueError, match="element budget"):
        with diagnostic.capture_relu_inputs(torch, model, max_elements=1):
            model(torch.ones(2))
    assert not model[0]._forward_pre_hooks
    with pytest.raises(ValueError, match="In-place"):
        with diagnostic.capture_relu_inputs(torch, torch.nn.Sequential(torch.nn.ReLU(inplace=True))):
            pass


def test_mismatched_forced_call_shape_is_rejected_and_restored():
    import torch
    model = torch.nn.Sequential(torch.nn.ReLU())
    events = [{"name": "0", "call": 0, "value": torch.ones(3)}]
    original = model[0].forward
    with pytest.raises(ValueError, match="call/shape"):
        with diagnostic.force_relu_backward(torch, model, events):
            model(torch.ones(2))
    assert model[0].forward == original


def test_sign_localization_retains_zero_and_crossing_magnitudes():
    import torch
    cpu = [{"name": "relu", "call": 0, "value": torch.tensor([-1e-8, 0.0, 1e-8, 2.0])}]
    gpu = [{"name": "relu", "call": 0, "value": torch.tensor([1e-8, 1e-9, -1e-8, 2.0])}]
    summary, arrays = diagnostic.relu_disagreements(torch, cpu, gpu)
    assert summary["sign_disagreements"] == 3 and summary["layers"][0]["cpu_exact_zeros"] == 1
    assert arrays["relu_000_flat_indices"].tolist() == [0, 1, 2]
    assert arrays["relu_000_cpu_values"].tolist() == cpu[0]["value"][:3].tolist()


def test_gradient_sign_counts_do_not_underflow():
    import torch
    compare = load("_kink_sign_compare", ROOT / "validate_cuda_execution.py")
    a = {"weight": torch.tensor([1e-30, 0.0, .1])}
    b = {"weight": torch.tensor([-1e-30, 1e-30, -.1])}
    result = diagnostic.signed_comparison(torch, compare, a, b, "gradient")["tensors"]["weight"]
    assert result["strict_opposite_signs"] == 2
    assert result["sign_disagreements"] == 3
    assert result["outside_tolerance_flat_indices"] == [2]


def test_complete_diagnostic_requires_all_cases_and_branches_but_not_semantic_pass():
    report = {"cases": [{"case": case, "objective": objective, "status": "complete", "phase": "complete",
                         "phases": {branch: {"status": "complete"} for branch in diagnostic.BRANCHES},
                         "comparisons": {branch: {"passed": False} for branch in diagnostic.BRANCHES[1:]},
                         "relu_localization": {"sign_disagreements": 0}}
                        for case in ("small", "median", "large") for objective in ("faithful", "nll")]}
    assert diagnostic.completion_gate(report)
    partial = copy.deepcopy(report)
    partial["cases"][0]["phases"].pop("cuda_cpu_mask_control")
    assert not diagnostic.completion_gate(partial)
    partial = copy.deepcopy(report)
    partial["cases"].pop()
    assert not diagnostic.completion_gate(partial)


def test_tiny_network_own_mask_reproduces_first_adam_and_postupdate_outputs():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    compare = load("_kink_toy_compare", ROOT / "validate_cuda_execution.py")
    trainer = load("_kink_toy_trainer", ROOT / "train_sand_cuda_deterministic.py")
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(2, 4)
            self.relu = torch.nn.ReLU()
            self.last = torch.nn.Linear(4, 3)
        def forward(self, position_sequence, next_positions, **kwargs):
            out = self.last(self.relu(self.first(position_sequence[:, -1])))
            return out[:, :2], out[:, 2].exp(), next_positions
    initial = compare.cpu_tree(torch, Tiny().state_dict())
    def make_model(h, metadata, objective, device, state):
        model = Tiny().to(device)
        model.load_state_dict(state)
        return model
    def finite(values):
        values = list(values)
        return bool(values) and all(bool(torch.isfinite(value).all()) for value in values)
    def gradient_guard(model):
        if not finite(p.grad for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Nonfinite gradient")
    def loss(pred, target, mask, pred_variance, **kwargs):
        return ((pred[mask] - target[mask]).square().sum(-1) / pred_variance[mask] + pred_variance[mask].log()).mean()
    h = SimpleNamespace(torch=torch, cpu_tree=lambda v: compare.cpu_tree(torch, v), assert_finite_gradients=gradient_guard,
                        tensors_are_finite=finite, synchronize=lambda device: None, acceleration_loss=loss)
    batch = {"position_sequence": torch.randn(5, 6, 2), "next_positions": torch.randn(5, 2),
             "position_sequence_noise": torch.zeros(5, 6, 2), "particle_types": torch.full((5,), 6),
             "nparticles_per_example": torch.tensor([5])}
    actual = SimpleNamespace(make_model=make_model)
    baseline, events = diagnostic.run_branch(h, trainer, compare, actual, {}, "faithful", initial, batch, "cpu")
    sham, _ = diagnostic.run_branch(h, trainer, compare, actual, {}, "faithful", initial, batch, "cpu", events)
    for field in ("before", "gradients", "after_state", "parameter_update", "after_output", "optimizer"):
        assert trainer.tree_equal(torch, baseline[field], sham[field])
