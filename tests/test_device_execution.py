import copy
from pathlib import Path

import pytest
import torch

from gns.device_utils import resolve_device, resolve_radius_backend, synchronize
from gns.learned_simulator import LearnedSimulator
from gns.losses import acceleration_loss
from gns.model_io import load_for_evaluation


def test_auto_device_priority_and_explicit_unavailable_failures(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="MPS was requested"):
        resolve_device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    assert resolve_device("auto").type == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    assert resolve_device("cpu").type == "cpu"


def test_metal_rejects_silent_cpu_kernel_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="FALLBACK=0"):
        resolve_device("mps")


def test_backend_selection_keeps_cpu_cuda_defaults_and_names_host_transfer():
    assert resolve_radius_backend("auto", "cpu") == "pyg"
    assert resolve_radius_backend("auto", "cuda") == "pyg"
    assert resolve_radius_backend("auto", "mps") == "scipy_host"
    assert resolve_radius_backend("scipy", "cpu") == "scipy"
    with pytest.raises(ValueError, match="no native MPS"):
        resolve_radius_backend("pyg", "mps")
    with pytest.raises(ValueError, match="CPU-only"):
        resolve_radius_backend("scipy", "mps")


def test_host_graph_preserves_strict_radius_caps_and_example_boundaries(simulator):
    points = torch.tensor([[0., 0.], [.4, 0.], [.8, 0.], [1., 0.], [0., 0.]])
    simulator._max_num_neighbors = 2
    simulator._radius_backend = "scipy"
    ordinary = simulator._compute_graph_connectivity(points, [4, 1], 1.)
    simulator._radius_backend = "scipy_host"
    host = simulator._compute_graph_connectivity(points, [4, 1], 1.)
    assert torch.equal(torch.stack(ordinary), torch.stack(host))
    pairs = set(map(tuple, torch.stack(host).T.tolist()))
    assert (0, 3) not in pairs  # strict radius excludes distance exactly1
    assert (0, 4) not in pairs and (4, 4) in pairs
    assert max(torch.bincount(host[1]).tolist()) <= 2


def test_runtime_backend_override_preserves_checkpoint_provenance(simulator, tmp_path):
    checkpoint = tmp_path / "model.pt"
    simulator._training_config = {"loss": "nll"}
    simulator.save(checkpoint)
    restored, provenance = load_for_evaluation(checkpoint, {}, "cpu", radius_backend="scipy_host")
    assert provenance["checkpoint_radius_backend"] == "pyg"
    assert provenance["runtime"]["graph_execution"] == "cpu_scipy_with_device_transfer"
    assert restored._radius_backend == "scipy_host"
    assert restored._checkpoint_config["radius_backend"] == "scipy_host"
    assert restored._checkpoint_config["latent_dim"] == simulator._checkpoint_config["latent_dim"]


def test_serial_metal_checkpoint_save_does_not_require_ddp_wrapper(tmp_path):
    from gns.train import save_model_and_train_state
    class SerialModel(torch.nn.Linear):
        def save(self, path):
            torch.save(self.state_dict(), path)
    model = SerialModel(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    save_model_and_train_state(None, torch.device("mps"), model,
                               {"model_path": str(tmp_path) + "/"}, 7, 0,
                               optimizer, 1., None, [], [])
    assert (tmp_path / "model-7.pt").is_file()
    state = torch.load(tmp_path / "train_state-7.pt", weights_only=True)
    assert state["global_train_state"]["step"] == 7


def test_synchronize_routes_metal_explicitly(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    synchronize("cpu")
    synchronize("mps")
    assert calls == ["mps"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires native Metal access")
def test_native_metal_host_graph_forward_backward_adaptive_and_checkpoint(simulator, tmp_path):
    config = copy.deepcopy(simulator._checkpoint_config)
    config["radius_backend"] = "scipy_host"
    config["normalization_stats"] = {name: {key: value.to("mps") for key, value in stats.items()}
                                     for name, stats in config["normalization_stats"].items()}
    metal = LearnedSimulator(**config, device="mps").to("mps")
    metal.load_state_dict(simulator.state_dict())
    history = torch.tensor([[.2, .2], [.8, .2], [1.3, .2]])[:, None].repeat(1, 6, 1)
    kinds = torch.zeros(3, dtype=torch.long)
    metal_history, metal_kinds = history.to("mps"), kinds.to("mps")
    simulator._radius_backend = "scipy_host"
    expected_graph = simulator._compute_graph_connectivity(history[:, -1], [2, 1], 1.)
    actual_graph = metal._compute_graph_connectivity(metal_history[:, -1], [2, 1], 1.)
    assert all(t.device.type == "mps" for t in actual_graph)
    assert torch.equal(torch.stack(expected_graph), torch.stack(actual_graph).cpu())
    expected = simulator.predict_positions(history, [3], kinds)
    actual = metal.predict_positions(metal_history, [3], metal_kinds)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
    pred, head, target = metal(next_positions=metal_history[:, -1] + .01,
                               position_sequence_noise=torch.zeros_like(metal_history),
                               position_sequence=metal_history,
                               nparticles_per_example=[3], particle_types=metal_kinds)
    loss = acceleration_loss(pred, target, torch.ones(3, dtype=torch.bool, device="mps"), head)
    loss.backward()
    assert all(torch.isfinite(p.grad).all().item() for p in metal.parameters() if p.grad is not None)
    with torch.no_grad():
        adaptive, _ = metal.predict_positions_adaptive(metal_history, [3], metal_kinds,
                                                       torch.tensor([0., 1., 2.], device="mps"),
                                                       sigma_percentile=50, radius_factor=1.267)
    assert torch.isfinite(adaptive).all().item()
    path = tmp_path / "metal.pt"
    metal._training_config = {"loss": "nll", "runtime": {"device": "mps"}}
    metal.save(path)
    restored, provenance = load_for_evaluation(path, {}, "cpu")
    assert provenance["training_config"]["runtime"]["device"] == "mps"
    with torch.no_grad():
        torch.testing.assert_close(restored.predict_positions(history, [3], kinds), expected)
