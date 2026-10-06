"""CPU toy/mock checks only: no repository model, source dataset or CUDA execution."""
import copy
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PATH = Path(__file__).with_name("train_sand_cuda.py")
spec = importlib.util.spec_from_file_location("prepared_sand_trainer", PATH)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def contract(frames=320):
    position_sha, type_sha = "1" * 64, "2" * 64
    description = {"path": "train/position_000000.npy", "shape": [frames, 4, 2],
                   "dtype": "<f4", "size_bytes": 100, "sha256": position_sha}
    record = {"id": "train:000000", "source_index": 0, "source_member": "trajectory_0.npy",
              "positions": description, "particle_types": {"path": "train/type_000000.npy",
                  "shape": [4], "dtype": "<i8", "size_bytes": 100, "sha256": type_sha},
              "trajectory_content_sha256": hashlib.sha256((position_sha + ":" + type_sha).encode()).hexdigest(),
              "logical_content_sha256": "3" * 64}
    manifest = {"format": "gns-trajectory-manifest", "version": 1, "split": "train", "dataset": "Sand",
                "source": {"family": "designsafe_published_npz", "dataset": "Sand", "file": "train.npz",
                    "size_bytes": 200, "sha256": "4" * 64, "ZIP_CRC_verified": True,
                    "member_count": 1, "acquisition_report_sha256": "5" * 64},
                "metadata": {"dim": 2}, "metadata_sha256": trainer.METADATA_SHA,
                "record_count": 1, "records": [record], "converter_sha256": "6" * 64}
    admission = {"schema": trainer.ADMISSION, "status": "admitted", "dataset": "Sand",
                 "manifest_sha256": "7" * 64, "metadata_sha256": trainer.METADATA_SHA,
                 "converter_sha256": "6" * 64, "structural_report_sha256": "8" * 64,
                 "record_count": 1, "frames_per_trajectory": frames,
                 "particle_type_ids": [5], "position_dtype": "<f4"}
    return manifest, admission


def test_default_description_imports_no_runtime():
    code = """import builtins, runpy, sys
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'scipy', 'gns'}:
        raise AssertionError('description imported runtime: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(PATH)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["endpoint_selected"] is False


def test_execution_requires_explicit_seed_endpoint_and_admission():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--execute"])
    args = ["--execute", "--repo", "r", "--train-manifest", "t", "--admission", "a",
            "--structural-report", "s", "--protocol", "p", "--output-dir", "o",
            "--objective", "faithful", "--seed", "0", "--updates", "17"]
    assert trainer.parse_args(args).updates == 17
    for extra in (["--stop-after", "18"], ["--seed", "-1"], ["--updates", "0"], ["--clear-stale-lock"]):
        with pytest.raises(SystemExit):
            trainer.parse_args(args + extra)


@pytest.mark.parametrize("frames", [320, 321])
def test_actual_frame_count_is_admitted_not_assumed(frames):
    manifest, admission = contract(frames)
    trainer.validate_manifest_contract(manifest, admission)


@pytest.mark.parametrize("change", [
    lambda m, a: a.update(status="complete_structural_only"),
    lambda m, a: a.update(particle_type_ids=[3, 5]),
    lambda m, a: a.update(frames_per_trajectory=321),
    lambda m, a: a.update(record_count=True),
    lambda m, a: m.update(split="test"),
    lambda m, a: m.update(material_property=1.0),
    lambda m, a: m["records"][0].update(material_property=1.0),
    lambda m, a: m["records"][0].update(source_index=2),
    lambda m, a: m["records"][0]["positions"].update(dtype="<f8"),
    lambda m, a: m["records"][0]["positions"].update(shape=[320, 4, 3]),
    lambda m, a: m["records"][0]["particle_types"].update(shape=[3]),
    lambda m, a: m["source"].update(family="original_tfrecord"),
    lambda m, a: m["source"].update(ZIP_CRC_verified=False),
    lambda m, a: m["source"].update(member_count=2),
    lambda m, a: m["records"][0].update(trajectory_content_sha256="9" * 64),
])
def test_reject_unadmitted_or_changed_structure(change):
    manifest, admission = contract()
    change(manifest, admission)
    with pytest.raises(ValueError):
        trainer.validate_manifest_contract(manifest, admission)


def test_duplicate_content_is_rejected():
    manifest, admission = contract()
    second = copy.deepcopy(manifest["records"][0])
    second.update(id="train:000001", source_index=1)
    manifest["records"].append(second)
    manifest["record_count"] = admission["record_count"] = 2
    manifest["source"]["member_count"] = 2
    with pytest.raises(ValueError, match="Duplicate"):
        trainer.validate_manifest_contract(manifest, admission)


def test_hash_mismatch_stops_before_runtime_import(tmp_path):
    for relative in trainer.SOURCE_PINS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('must not be imported')")
    with pytest.raises(ValueError, match="source mismatch"):
        trainer.load_helpers(tmp_path)


def test_lock_requires_explicit_same_host_dead_pid_recovery(tmp_path, monkeypatch):
    with trainer.RunLock(tmp_path) as first:
        with pytest.raises(FileExistsError):
            with trainer.RunLock(tmp_path):
                pass
        monkeypatch.setattr(trainer, "pid_alive", lambda pid: True)
        with pytest.raises(RuntimeError):
            with trainer.RunLock(tmp_path, clear_stale=True):
                pass
        assert json.loads((tmp_path / "run.lock").read_text()) == first
    old = {"pid": 99999, "host": trainer.socket.gethostname(), "token": "old"}
    (tmp_path / "run.lock").write_text(json.dumps(old))
    monkeypatch.setattr(trainer, "pid_alive", lambda pid: False)
    with trainer.RunLock(tmp_path, clear_stale=True) as recovered:
        assert recovered["token"] != "old"
    old["host"] = "another-machine"
    (tmp_path / "run.lock").write_text(json.dumps(old))
    with pytest.raises(RuntimeError):
        with trainer.RunLock(tmp_path, clear_stale=True):
            pass
    assert (tmp_path / "run.lock").exists()


@pytest.mark.parametrize("pid", [None, 0, -1, True, "123"])
def test_malformed_lock_pid_is_not_deletable(pid):
    with pytest.raises(ValueError):
        trainer.pid_alive(pid)


def test_pointer_path_and_endpoint_are_checked(tmp_path):
    config = {"updates": 2}
    path = tmp_path / "checkpoint-000000002.pt"
    path.write_bytes(b"synthetic nonloaded checkpoint")
    pointer = {"path": path.name, "sha256": trainer.sha(path), "completed_steps": 2,
               "run_config_sha256": trainer.config_hash(config)}
    (tmp_path / "latest.json").write_text(json.dumps(pointer))
    assert trainer.read_pointer(tmp_path, config)["completed_steps"] == 2
    for change in ({"path": "../checkpoint-000000002.pt"}, {"completed_steps": 3}, {"sha256": "0" * 64}):
        (tmp_path / "latest.json").write_text(json.dumps({**pointer, **change}))
        with pytest.raises(ValueError):
            trainer.read_pointer(tmp_path, config)


@pytest.fixture
def toy(monkeypatch):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(4001)
    cuda_generator = torch.Generator(device="cpu").manual_seed(5001)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: cuda_generator.get_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: cuda_generator.set_state(value))

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
            raise FloatingPointError("Nonfinite gradient")

    def loss(pred, target, mask, pred_variance, **_):
        return ((pred[mask] - target[mask]).square() / pred_variance[mask] + pred_variance[mask].log()).mean()

    helpers = SimpleNamespace(torch=torch, cpu_tree=cpu_tree, tensors_are_finite=finite,
                              assert_finite_gradients=finite_gradients, acceleration_loss=loss)

    def make():
        model = torch.nn.Linear(2, 2)
        model._checkpoint_config = {"toy": True, "normalization": torch.tensor([0.2, 0.3])}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False, fused=False)
        return model, optimizer

    def update(model, optimizer, step):
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = 1e-4 * .1 ** ((step - 1) / 99999)
        x = torch.randn(3, 2) + torch.randn(3, 2, generator=cuda_generator)
        output = model(x)
        pred, head = output[:, :1], output[:, 1:].exp()
        trainer.guarded_update(helpers, model, optimizer, pred, head, torch.ones_like(pred),
                               torch.ones(3, dtype=torch.bool), "nll", step)

    return SimpleNamespace(torch=torch, helpers=helpers, make=make, update=update, cuda_generator=cuda_generator)


def test_cpu_toy_checkpoint_replays_two_rng_streams_and_adam(toy):
    torch = toy.torch
    config = {"objective": "nll", "updates": 2}
    model, optimizer = toy.make()
    toy.update(model, optimizer, 1)
    payload = trainer.checkpoint_payload(toy.helpers, model, optimizer, config, 1,
                                         {"training": [], "elapsed_seconds": 0.0}, "mock-cuda")
    toy.update(model, optimizer, 2)
    expected = {"model": toy.helpers.cpu_tree(model.state_dict()),
                "optimizer": toy.helpers.cpu_tree(optimizer.state_dict()),
                "rng": trainer.capture_rng(torch, "mock-cuda")}
    restored, restored_optimizer = toy.make()
    completed, _ = trainer.restore_payload(toy.helpers, payload, restored, restored_optimizer, config, "mock-cuda")
    assert completed == 1
    toy.update(restored, restored_optimizer, 2)
    actual = {"model": toy.helpers.cpu_tree(restored.state_dict()),
              "optimizer": toy.helpers.cpu_tree(restored_optimizer.state_dict()),
              "rng": trainer.capture_rng(torch, "mock-cuda")}
    assert trainer.tree_equal(torch, expected, actual)


@pytest.mark.parametrize("mutation", ["empty", "partial", "fractional", "wrongstep", "multistep", "nan", "betas", "lr"])
def test_adam_rejects_incomplete_or_drifted_state(toy, mutation):
    model, optimizer = toy.make()
    toy.update(model, optimizer, 1)
    parameter = next(model.parameters())
    state = optimizer.state[parameter]
    if mutation == "empty":
        optimizer.state.clear()
    elif mutation == "partial":
        del optimizer.state[parameter]
    elif mutation == "fractional":
        state["step"] = toy.torch.tensor(1.5)
    elif mutation == "wrongstep":
        state["step"] = toy.torch.tensor(2.0)
    elif mutation == "multistep":
        state["step"] = toy.torch.tensor([1.0, 1.0])
    elif mutation == "nan":
        state["exp_avg"].fill_(float("nan"))
    elif mutation == "betas":
        optimizer.param_groups[0]["betas"] = (.5, .999)
    elif mutation == "lr":
        optimizer.param_groups[0]["lr"] = .1
    with pytest.raises((ValueError, FloatingPointError)):
        trainer.assert_adam(toy.torch, model, optimizer, 1)


def test_cross_lineage_and_over_endpoint_checkpoint_rejected(toy):
    config = {"objective": "nll", "updates": 2}
    model, optimizer = toy.make()
    payload = trainer.checkpoint_payload(toy.helpers, model, optimizer, config, 0,
                                         {"training": [], "elapsed_seconds": 0.0}, "mock-cuda")
    for change in ({"cuda_sand_training_schema": "full_training"}, {"completed_steps": 3}, {"run_config_sha256": "0" * 64}):
        with pytest.raises(ValueError, match="lineage"):
            trainer.restore_payload(toy.helpers, {**payload, **change}, model, optimizer, config, "mock-cuda")


@pytest.mark.parametrize("failure", ["prediction", "variance", "target", "loss", "gradient"])
def test_nonfinite_guard_prevents_optimizer_step(toy, failure):
    torch = toy.torch
    model, optimizer = toy.make()
    optimizer.step = Mock(wraps=optimizer.step)
    value = model(torch.ones(3, 2))
    pred, head, target = value[:, :1], value[:, 1:].exp(), torch.ones(3, 1)
    if failure == "prediction":
        pred = pred * float("nan")
    elif failure == "variance":
        head = head * float("nan")
    elif failure == "target":
        target.fill_(float("nan"))
    elif failure == "loss":
        toy.helpers.acceleration_loss = lambda *a, **k: pred.sum() * float("nan")
    elif failure == "gradient":
        next(model.parameters()).register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(FloatingPointError):
        trainer.guarded_update(toy.helpers, model, optimizer, pred, head, target,
                               torch.ones(3, dtype=torch.bool), "nll", 1)
    optimizer.step.assert_not_called()


def test_literal_bitwise_comparison_distinguishes_signed_zero(toy):
    assert not trainer.tree_equal(toy.torch, toy.torch.tensor([0.0]), toy.torch.tensor([-0.0]))


def test_toy_training_stops_and_resumes_at_exact_endpoint(toy, tmp_path, monkeypatch):
    """Exercise the entry's loop with a tiny linear model and fabricated tensors."""
    torch, helpers = toy.torch, toy.helpers
    helpers.NOISE, helpers.KINEMATIC = 6.7e-4, 3
    helpers.synchronize = lambda device: None
    helpers.sample_indices = lambda seed, step, length, count: [step % length, (step + 1) % length]
    helpers.unpack_batch = lambda examples: (torch.ones(4, 6, 2), torch.full((4,), 5),
                                             torch.tensor([2, 2]), torch.ones(4, 1))
    helpers.host_noise = lambda shape, types, seed, step: torch.zeros(shape)
    helpers.learning_rate = lambda step: 1e-4 * .1 ** (min(step / 99999, 1.0))
    helpers.frame_identity = lambda dataset, ids, index: f"synthetic:{index}"
    helpers.build_simulator = lambda *args, **kwargs: toy.make()[0]

    def forward(model, batch, noise, device):
        output = model(torch.randn(4, 2) + torch.randn(4, 2, generator=toy.cuda_generator))
        return output[:, :1], output[:, 1:].exp(), batch[3]

    helpers.forward_batch = forward
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: toy.cuda_generator.manual_seed(seed))
    monkeypatch.setattr(trainer, "load_admitted_dataset", lambda args, h: (
        [0, 1, 2], {"dim": 2}, {"trajectory_ids": ["synthetic"], "manifest_sha256": "0" * 64}))
    protocol = tmp_path / "synthetic-protocol.md"
    protocol.write_text("Synthetic loop test only; no scientific protocol.")
    args = SimpleNamespace(objective="nll", seed=2, updates=3, checkpoint_every=2, log_every=1,
                           protocol=protocol, output_dir=tmp_path / "resumed", resume=False, stop_after=2)

    def run(args):
        with trainer.RunLock(args.output_dir) as process:
            trainer.run_training(args, helpers, process, torch.device("cpu"), {"scope": "CPU toy with mocked CUDA RNG"})

    run(args)
    status = json.loads((args.output_dir / "status.json").read_text())
    assert (status["state"], status["completed_steps"], status["committed_steps"]) == ("planned_stop_incomplete", 2, 2)
    args.resume, args.stop_after = True, None
    run(args)
    status = json.loads((args.output_dir / "status.json").read_text())
    assert (status["state"], status["completed_steps"], status["committed_steps"]) == ("complete", 3, 3)
    history = json.loads((args.output_dir / "history.json").read_text())
    assert [row["completed_steps"] for row in history["training"]] == [1, 2, 3]
    resumed = torch.load(args.output_dir / "checkpoint-000000003.pt", map_location="cpu", weights_only=True)
    args.output_dir, args.resume = tmp_path / "uninterrupted", False
    run(args)
    uninterrupted = torch.load(args.output_dir / "checkpoint-000000003.pt", map_location="cpu", weights_only=True)
    for key in ("state_dict", "optimizer_state", "rng_states"):
        assert trainer.tree_equal(torch, resumed[key], uninterrupted[key])

    # A checkpoint file without its published pointer is not committed.
    real_atomic_json = trainer.atomic_json

    def pointer_failure(path, value):
        if Path(path).name == "latest.json" and value["completed_steps"] == 2:
            raise OSError("synthetic pointer publication failure")
        real_atomic_json(path, value)

    monkeypatch.setattr(trainer, "atomic_json", pointer_failure)
    args.output_dir = tmp_path / "pointer-failure"
    with pytest.raises(OSError, match="pointer publication"):
        run(args)
    status = json.loads((args.output_dir / "status.json").read_text())
    assert (status["state"], status["completed_steps"], status["committed_steps"]) == ("failed", 2, 0)
    assert json.loads((args.output_dir / "latest.json").read_text())["completed_steps"] == 0
