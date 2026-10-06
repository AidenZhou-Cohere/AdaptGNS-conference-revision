"""Focused pure-runtime mocks; no Torch model, data or CUDA execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parent
PATH = ROOT / "train_sand_cuda_deterministic.py"
spec = importlib.util.spec_from_file_location("_deterministic_trainer_test", PATH)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def runtime_mock():
    flags = {"enabled": False, "warn_only": True}
    def strict(value, *, warn_only):
        flags.update(enabled=value, warn_only=warn_only)
    cuda = SimpleNamespace(is_available=Mock(return_value=True), device_count=lambda: 4,
        set_device=Mock(), get_device_properties=lambda device: SimpleNamespace(name="NVIDIA GB200", uuid="mock-uuid"),
        get_device_capability=lambda device: (10, 0))
    torch = SimpleNamespace(__version__="2.13.0+cu129", version=SimpleNamespace(cuda="12.9"), cuda=cuda,
        use_deterministic_algorithms=Mock(side_effect=strict), device=lambda kind, index: f"{kind}:{index}",
        set_num_threads=Mock(), set_float32_matmul_precision=Mock(), get_float32_matmul_precision=lambda: "highest",
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                 cudnn=SimpleNamespace(allow_tf32=True, benchmark=True)),
        are_deterministic_algorithms_enabled=lambda: flags["enabled"],
        is_deterministic_algorithms_warn_only_enabled=lambda: flags["warn_only"])
    return SimpleNamespace(torch=torch, np=SimpleNamespace(__version__="mock"), scipy=SimpleNamespace(__version__="mock"))


def test_source_is_exact_declared_minimal_variant():
    inventory = json.loads((ROOT / "sand_deterministic_variant_changes.json").read_text())
    source = (ROOT / "train_sand_cuda.py").read_text()
    assert trainer.sha(ROOT / "train_sand_cuda.py") == inventory["parent_sha256"]
    for before, after in inventory["changes"]:
        assert source.count(before) == 1
        source = source.replace(before, after, 1)
    assert source == PATH.read_text()
    assert trainer.sha(PATH) == inventory["new_sha256"]
    assert trainer.SCHEMA == "adaptgns_sand_cuda_training_deterministic_v2"
    assert trainer.description()["endpoint_selected"] is False


@pytest.mark.parametrize("value", [None, "", ":16:8", ":4096:2"])
def test_workspace_configuration_is_required_before_any_runtime_change(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    helpers = runtime_mock()
    with pytest.raises(RuntimeError, match="before process start"):
        trainer.configure_cuda(helpers, SimpleNamespace(cuda_index=0, threads=2))
    helpers.torch.use_deterministic_algorithms.assert_not_called()
    helpers.torch.cuda.is_available.assert_not_called()


def test_strict_runtime_gate_preserves_precision_and_reports_configuration(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    helpers = runtime_mock()
    device, report = trainer.configure_cuda(helpers, SimpleNamespace(cuda_index=2, threads=2))
    helpers.torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)
    assert device == "cuda:2"
    assert report["deterministic_algorithms"] is True
    assert report["deterministic_warn_only"] is False
    assert report["cublas_workspace_config"] == ":4096:8"
    assert report["tf32"] is False and report["amp"] is False
    assert helpers.torch.backends.cuda.matmul.allow_tf32 is False
    assert helpers.torch.backends.cudnn.allow_tf32 is False
    assert helpers.torch.backends.cudnn.benchmark is False


def test_deterministic_runtime_failure_is_not_downgraded_to_warning(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    helpers = runtime_mock()
    helpers.torch.use_deterministic_algorithms.side_effect = RuntimeError("synthetic unsupported deterministic operation")
    with pytest.raises(RuntimeError, match="unsupported deterministic"):
        trainer.configure_cuda(helpers, SimpleNamespace(cuda_index=0, threads=2))
    helpers.torch.cuda.is_available.assert_not_called()
