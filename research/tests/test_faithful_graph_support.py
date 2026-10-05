"""Tiny CPU semantics/checkpoint tests; never load real models or datasets."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research import faithful_graph_support as support
from research.tests.test_graph_convention_bridge import tiny_model, fixture
from gns.learned_simulator import LearnedSimulator
from gns.model_io import load_for_evaluation


def model():
    initial = tiny_model()
    config = copy.deepcopy(initial._checkpoint_config)
    config["detach_variance_features"] = True
    result = LearnedSimulator(**config, device="cpu")
    result.load_state_dict(initial.state_dict())
    return result


def batch():
    current, _, types, target = fixture()
    position = torch.tensor(current.transpose(1, 0, 2).copy())
    # Deliberately identical coordinates across examples test partition safety.
    return (torch.cat((position, position)), torch.tensor(np.r_[types, types]),
            torch.tensor([len(types), len(types)]), torch.tensor(np.concatenate((target, target))))


def config(arm="base"):
    return {"objective": "faithful", "arm": arm, "seed": 0,
        "parent_checkpoint_sha256": support.PARENT_HASHES[0], "parent_training_config": {"loss": "faithful"},
        "parent_simulator_config_sha256": support.simulator_config_sha256(model()._checkpoint_config),
        "parent_completed_updates": 100000, "additional_updates": 10000, "lr": 1e-5, "batch_size": 2,
        "input_files_sha256": {}, "validation_frames": [], "train": {"trajectory_ids": ["synthetic:0"]}}


def history(completed):
    return {"training": [], "validation": [], "graph_updates": [
        {"absolute_schedule_step": 100000 + index, "completed_before": index, "completed_additional_updates": index + 1}
        for index in range(completed)], "elapsed_seconds": 0.}


def optimizer(m):
    return torch.optim.Adam(m.parameters(), lr=support.LR, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


def test_base_normalized_output_target_loss_gradients_and_adam_update_exact():
    original_model, new_model = model(), model()
    inputs = batch()
    noise = support.original.host_noise(inputs[0].shape, inputs[1], 0, 100000)
    original_output = support.original.forward_batch(original_model, inputs, noise, "cpu")
    new_output = support.forward_batch(new_model, inputs, noise, "cpu", 0, 100000, "base")
    for left, right in zip(original_output, new_output[:3]):
        assert torch.equal(left, right)
    mask = inputs[1] != 3
    losses = [support.original.acceleration_loss(out[0], out[2], mask, pred_variance=out[1], loss_type="faithful", variance_floor=1e-6)
              for out in (original_output, new_output)]
    assert torch.equal(*losses)
    opts = [optimizer(original_model), optimizer(new_model)]
    for loss in losses:
        loss.backward()
    for left, right in zip(original_model.parameters(), new_model.parameters()):
        assert (left.grad is None and right.grad is None) or (
            left.grad is not None and right.grad is not None and torch.equal(left.grad, right.grad))
    for opt in opts:
        opt.step()
    assert_tree_equal(original_model.state_dict(), new_model.state_dict())
    assert_tree_equal(opts[0].state_dict(), opts[1].state_dict())
    assert all(row["selected_optional_pairs"] == 0 for row in new_output[3])


def test_faithful_mean_gradients_do_not_depend_on_risk_loss():
    left, right = model(), model()
    inputs = batch(); noise = torch.zeros_like(inputs[0])
    p, q, y, _ = support.forward_batch(left, inputs, noise, "cpu", 0, 100000, "base")
    mean_only = ((p-y).square().sum(-1)).mean()
    mean_only.backward()
    p, q, y, _ = support.forward_batch(right, inputs, noise, "cpu", 0, 100000, "base")
    loss = support.original.acceleration_loss(p, y, inputs[1] != 3, pred_variance=q, loss_type="faithful", variance_floor=1e-6)
    loss.backward()
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        if "_variance_head" not in name:
            assert (a.grad is None and b.grad is None) or (
                a.grad is not None and b.grad is not None and torch.equal(a.grad, b.grad)), name


@pytest.mark.parametrize("absolute", [100000, 100001, 102499, 102500, 109999])
def test_graph_coin_pair_rng_is_independent_reproducible_and_per_example(absolute):
    torch.manual_seed(22); before = torch.get_rng_state().clone()
    sample_before = support.original.sample_indices(0, absolute, 2000, 2)
    coin, rng, cm, pm = support.graph_rng(0, absolute, 0)
    first = rng.permutation(100)
    same_coin, same_rng, _, _ = support.graph_rng(0, absolute, 0)
    assert coin == same_coin and np.array_equal(first, same_rng.permutation(100))
    assert cm == [20261005, 0, absolute, 0, 4409] and pm == [20261005, 0, absolute, 0, 5501]
    assert support.graph_rng(0, absolute, 1)[2] != cm
    assert torch.equal(torch.get_rng_state(), before)
    assert support.original.sample_indices(0, absolute, 2000, 2) == sample_before


def test_mixture_preserves_native_prefix_exact_budget_and_batch_partition():
    m, inputs = model(), batch()
    absolute = next(step for step in range(100000, 100100) if any(support.graph_rng(0, step, j)[0] for j in range(2)))
    sequence = inputs[0]
    nodes, native, features = m._encoder_preprocessor(sequence, inputs[2], inputs[1], None)
    edge, ledger = support.append_optional_edges(sequence, inputs[2], native, .015, 0, absolute, "mix")
    assert torch.equal(edge[:, :native.shape[1]], native)
    assert edge.shape[1] == native.shape[1] + 2 * sum(row["selected_optional_pairs"] for row in ledger)
    assert any(row["expanded"] and row["selected_optional_pairs"] > 0 for row in ledger)
    membership = torch.repeat_interleave(torch.arange(2), inputs[2])
    assert torch.equal(membership[edge[0]], membership[edge[1]])
    assert int((edge[0] == edge[1]).sum()) == int((native[0] == native[1]).sum())
    for row in ledger:
        assert row["selected_optional_pairs"] == (row["optional_budget_if_exposed"] if row["expanded"] else 0)


def test_cap_omissions_never_become_optional_short_range_pairs():
    m = model()
    points = torch.zeros((132, 6, 2), dtype=torch.float32)
    types, counts = torch.ones(132, dtype=torch.long), torch.tensor([132])
    _, native, _ = m._encoder_preprocessor(points, counts, types, None)
    edge, ledger = support.append_optional_edges(points, counts, native, .015, 0, 100000, "mix")
    assert ledger[0]["receivers_above_native_cap"] == 132
    assert ledger[0]["annulus_pairs"] == 0 and ledger[0]["selected_optional_pairs"] == 0
    assert edge is native
    assert int((edge[0] == edge[1]).sum()) < 132


def test_graphs_and_predictions_do_not_depend_on_target():
    m, inputs = model(), batch()
    changed = (*inputs[:3], inputs[3] + 50)
    noise = torch.zeros_like(inputs[0])
    a = support.forward_batch(m, inputs, noise, "cpu", 0, 100000, "mix")
    b = support.forward_batch(m, changed, noise, "cpu", 0, 100000, "mix")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[3] == b[3]
    assert not torch.equal(a[2], b[2])


def test_parent_optimizer_state_is_preserved_and_not_aliased():
    parent_model, inputs = model(), batch(); parent_opt = optimizer(parent_model)
    support.update_once(parent_model, parent_opt, inputs, torch.zeros_like(inputs[0]), "cpu", 0, 99999, "base")
    for state in parent_opt.state.values():
        state["step"].fill_(100000)
    payload = {"state_dict": copy.deepcopy(parent_model.state_dict()), "optimizer_state": copy.deepcopy(parent_opt.state_dict()),
               "rng_states": support.original.capture_rng("cpu")}
    before = copy.deepcopy(payload)
    child = model(); opt = support.parent_optimizer(child, payload, "cpu")
    assert_tree_equal(opt.state_dict(), payload["optimizer_state"])
    support.update_once(child, opt, inputs, torch.zeros_like(inputs[0]), "cpu", 0, 100000, "base")
    assert_tree_equal(payload, before)
    assert all(float(state["step"]) == 100001 for state in opt.state.values())


def test_save_resume_matches_next_update_and_exposes_evaluable_lineage(tmp_path):
    m, inputs = model(), batch(); opt = optimizer(m); cfg = config("mix")
    noise = support.original.host_noise(inputs[0].shape, inputs[1], 0, 100000)
    support.update_once(m, opt, inputs, noise, "cpu", 0, 100000, "mix")
    path = tmp_path / "checkpoint-extra-00001.pt"
    pointer = support.save_checkpoint(path, m, opt, cfg, 1, history(1), "cpu")
    original_bytes = path.read_bytes()
    restored = model(); restored_opt = optimizer(restored)
    completed, restored_history = support.restore_checkpoint(path, pointer["sha256"], restored, restored_opt, cfg, "cpu")
    assert completed == 1 and restored_history == history(1)
    assert_tree_equal(m.state_dict(), restored.state_dict())
    assert_tree_equal(opt.state_dict(), restored_opt.state_dict())
    next_noise = support.original.host_noise(inputs[0].shape, inputs[1], 0, 100001)
    support.update_once(m, opt, inputs, next_noise, "cpu", 0, 100001, "mix")
    support.update_once(restored, restored_opt, inputs, next_noise, "cpu", 0, 100001, "mix")
    assert_tree_equal(m.state_dict(), restored.state_dict())
    assert_tree_equal(opt.state_dict(), restored_opt.state_dict())
    payload = torch.load(path, weights_only=True)
    assert "full_training_schema" not in payload
    with pytest.raises(ValueError, match="fixed 10k"):
        support.validate_continuation_payload(payload)
    generic, provenance = load_for_evaluation(path, {}, "cpu", radius_backend="scipy_host")
    assert provenance["normalization_source"] == "checkpoint"
    assert_tree_equal(generic.state_dict(), payload["state_dict"])
    assert path.read_bytes() == original_bytes


def test_restore_rng_and_reject_hash_config_ledger_changes(tmp_path):
    m = model(); opt = optimizer(m); cfg = config(); path = tmp_path / "one.pt"
    torch.manual_seed(781)
    pointer = support.save_checkpoint(path, m, opt, cfg, 0, history(0), "cpu")
    expected_rng = torch.rand(5)
    torch.manual_seed(999)
    support.restore_checkpoint(path, pointer["sha256"], m, opt, cfg, "cpu")
    assert torch.equal(torch.rand(5), expected_rng)
    with pytest.raises(ValueError, match="byte hash"):
        support.restore_checkpoint(path, "0" * 64, m, opt, cfg, "cpu")
    with pytest.raises(ValueError, match="configuration"):
        support.restore_checkpoint(path, pointer["sha256"], m, opt, config("mix"), "cpu")
    with pytest.raises(ValueError, match="replace"):
        support.save_checkpoint(path, m, opt, cfg, 0, history(0), "cpu")
    (tmp_path / "orphan.pt.tmp").write_bytes(b"preserve partial checkpoint")
    with pytest.raises(FileExistsError):
        support.save_checkpoint(tmp_path / "orphan.pt", m, opt, cfg, 0, history(0), "cpu")
    assert (tmp_path / "orphan.pt.tmp").read_bytes() == b"preserve partial checkpoint"


def test_endpoint_rejects_old_schema_nested_config_and_simulator_metadata():
    m = model(); cfg = config()
    payload = support.checkpoint_payload(m, optimizer(m), cfg, 10000, history(10000), "cpu")
    assert support.validate_continuation_payload(payload) == (cfg, 10000)
    mutations = [
        (lambda p: p.update(full_training_schema=1), "Old full-training"),
        (lambda p: p["training_config"].update(graph_support_continuation={}), "Nested training"),
        (lambda p: p["training_config"].update(parent_training_config={"loss": "nll"}), "Nested training"),
        (lambda p: p["simulator_config"].update(latent_dim=999), "Simulator architecture"),
        (lambda p: p["simulator_config"]["normalization_stats"]["acceleration"]["std"].add_(.01), "Simulator architecture"),
        (lambda p: p["history"]["graph_updates"].pop(), "ledger/update count"),
        (lambda p: p["history"]["graph_updates"][5000].update(absolute_schedule_step=5), "absolute-step schedule"),
        (lambda p: p["history"]["graph_updates"][5000].update(completed_before=5001), "absolute-step schedule"),
        (lambda p: p["history"]["graph_updates"][5000].update(completed_additional_updates=5000), "absolute-step schedule"),
    ]
    for mutate, error in mutations:
        changed = copy.deepcopy(payload); mutate(changed)
        with pytest.raises(ValueError, match=error):
            support.validate_continuation_payload(changed)


@pytest.mark.parametrize("key,value", [("detach_variance_features", False),
    ("uncertainty_parameterization", "legacy_std"), ("variance_floor", 1e-5)])
def test_faithful_simulator_semantics_required_even_if_metadata_hash_is_replaced(key, value):
    m = model(); payload = support.checkpoint_payload(m, optimizer(m), config(), 0, history(0), "cpu")
    payload["simulator_config"][key] = value
    payload["continuation_config"]["parent_simulator_config_sha256"] = support.simulator_config_sha256(payload["simulator_config"])
    payload["continuation_config_sha256"] = support.original.config_hash(payload["continuation_config"])
    with pytest.raises(ValueError, match="Faithful detached"):
        support.validate_continuation_payload(payload, require_complete=False)


@pytest.mark.parametrize("relative", ["status.json.tmp", "latest.json.tmp", "attempts/prior/status.json.tmp"])
def test_stray_temporary_files_block_resume_before_any_write(tmp_path, relative):
    path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preserve interrupted atomic write")
    (tmp_path / "protocol.json").write_text("unread preflight fixture")
    args = SimpleNamespace(output_dir=tmp_path, resume=True)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Stray temporary"):
        support.run(args, None, None, {}, None, None, "cpu")
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def test_nonfinite_loss_never_updates_parameters(monkeypatch):
    m, inputs = model(), batch(); opt = optimizer(m); before = copy.deepcopy(m.state_dict())
    monkeypatch.setattr(support.original, "acceleration_loss", lambda *a, **kw: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(FloatingPointError, match="not executed"):
        support.update_once(m, opt, inputs, torch.zeros_like(inputs[0]), "cpu", 0, 100000, "base")
    assert_tree_equal(before, m.state_dict()); assert not opt.state


def test_pinned_input_mutation_rejected(tmp_path):
    source = tmp_path / "source.py"; source.write_text("immutable")
    cfg = {"input_files_sha256": {str(source): support.original.sha256(source)}}
    support.verify_pins(cfg)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Pinned"):
        support.verify_pins(cfg)


def test_attempt_failure_and_resume_keep_unsuccessful_state_and_absolute_schedule(tmp_path, monkeypatch):
    class Dataset:
        def __len__(self): return 100
        def __getitem__(self, index): return index
    inputs, dataset = batch(), Dataset()
    monkeypatch.setattr(support.original, "unpack_batch", lambda _: inputs)
    monkeypatch.setattr(support.original, "frame_identity", lambda _d, _ids, index: f"synthetic:{index}")
    monkeypatch.setattr(support.original, "evaluate_validation", lambda *a: {"equal_trajectory_mean": {"coordinate_mse": 1.}, "binned_vector_se_calibration_gap": 1.})
    args = SimpleNamespace(output_dir=tmp_path, seed=0, arm="base", resume=False, stop_after=1)
    m = model(); opt = optimizer(m); cfg = config()
    real_update = support.update_once
    monkeypatch.setattr(support, "update_once", lambda *a: (_ for _ in ()).throw(FloatingPointError("synthetic failure")))
    with pytest.raises(FloatingPointError, match="synthetic"):
        support.run(args, m, opt, cfg, dataset, dataset, "cpu")
    failed = json.loads((tmp_path / "status.json").read_text())
    assert failed["state"] == "failed" and failed["completed_additional_updates"] == 0
    attempt = tmp_path / "attempts" / failed["attempt_id"]
    snapshot = (attempt / "unsuccessful_state.pt").read_bytes()
    assert json.loads((attempt / "current_update.json").read_text())["absolute_schedule_step"] == 100000
    parent_checkpoint = (tmp_path / "checkpoint-extra-00000.pt").read_bytes()
    monkeypatch.setattr(support, "update_once", real_update)
    args.resume = True
    support.run(args, m, opt, cfg, dataset, dataset, "cpu")
    resumed = json.loads((tmp_path / "status.json").read_text())
    assert resumed["state"] == "planned_stop_incomplete" and resumed["completed_additional_updates"] == 1
    assert (attempt / "unsuccessful_state.pt").read_bytes() == snapshot
    assert (tmp_path / "checkpoint-extra-00000.pt").read_bytes() == parent_checkpoint
    assert json.loads((tmp_path / "history.json").read_text())["graph_updates"][0]["absolute_schedule_step"] == 100000
    assert len(list((tmp_path / "attempts").iterdir())) == 2
