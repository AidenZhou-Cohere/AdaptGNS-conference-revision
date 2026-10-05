"""Synthetic native-graph rollout contracts, without real model/data access."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research import native_graph_rollout as native
from research.tests.test_graph_convention_bridge import fixture, tiny_model


def rollout_fixture(horizon=3):
    history, _, types, target = fixture()
    positions = np.concatenate((history, np.broadcast_to(target, (horizon, len(types), 2))), axis=0).copy()
    metadata = {"bounds": [[.1, .9], [.1, .9]], "default_connectivity_radius": .015}
    return positions, types, metadata


@pytest.mark.parametrize("policy", native.POLICIES)
def test_native_graph_preserves_ordered_base_and_exact_optional_pairs(policy):
    history, _, types, _ = fixture()
    score = np.arange(len(types), dtype=float)
    graph, edge, pairs, audit = native.native_graph(history, policy, score, np.random.default_rng(8))
    model = tiny_model()
    sender, receiver = model._compute_graph_connectivity(torch.from_numpy(history[-1]), [len(types)], .015)
    base = torch.stack((sender, receiver)).numpy()
    np.testing.assert_array_equal(edge[:, :base.shape[1]], base)
    expected = 0 if policy == "base" else len(graph.extra) if policy == "dense" else int(.25 * len(graph.extra))
    assert len(pairs) == expected
    assert edge.shape[1] == base.shape[1] + 2 * expected
    assert audit["native_base_prefix_preserved"]
    assert np.sum(edge[0] == edge[1]) == len(types)
    assert np.all(pairs[:, 0] < pairs[:, 1])


def test_native_cap_and_asymmetry_are_preserved_not_symmetrized():
    points = np.stack((np.arange(130, dtype=np.float32) * .00001 + .2, np.full(130, .2, dtype=np.float32)), axis=1)
    history = np.broadcast_to(points, (6, 130, 2)).copy()
    graph, edge, optional, audit = native.native_graph(history, "base", None, np.random.default_rng(0))
    assert audit["native_base_receivers_above_cap_before_capping"] == 130
    assert audit["native_base_max_receiver_degree"] == 128
    assert audit["native_base_edges_removed_by_cap"] == 130 * 2
    assert audit["native_base_asymmetric_directed_edges"] > 0
    assert edge.shape[1] == 130 * 128 and len(optional) == 0


def test_random_selection_reuses_locked_initial_rng_material():
    history, _, _, _ = fixture()
    rng_seed = 93000 + 1000 * 2 + 17
    graph, _, optional, _ = native.native_graph(history, "random25", None, np.random.default_rng(rng_seed))
    original_selected = native.full.choose_pairs(graph, "random25", history, None, np.random.default_rng(rng_seed))
    expected = native.bridge.same.canonical_pairs(original_selected[len(graph.base):])
    np.testing.assert_array_equal(optional, expected)


@pytest.mark.parametrize("policy", native.POLICIES)
def test_short_rollout_retains_parity_boundary_counts_and_native_trace_graphs(policy):
    positions, types, metadata = rollout_fixture()
    row, arrays = native.rollout(tiny_model(), positions, types, metadata, policy, 3, 93000, trace_steps=(1, 2, 3))
    assert row["native_parity"]["passed"] and row["native_parity_network_passes"] == 2
    assert row["status"] == "complete" and row["completed_steps"] == 3
    assert row["forecast_network_passes"] == 3
    assert row["total_network_passes"] == 3 + (policy == "laggedrisk25")
    assert row["total_wall_seconds_including_native_parity"] >= row["total_wall_seconds"]
    assert len(row["predicted_boundary_per_step"]) == len(row["ground_truth_boundary_per_step"]) == 3
    assert arrays["observed_or_predicted_histories"].shape == (3, 6, len(types), 2)
    for step, attempt in enumerate(row["attempts"], start=1):
        edge = arrays[f"edges_forecast_{step:04d}"]
        assert native.full.state_hash(edge) == attempt["directed_edge_sha256"]
        assert attempt["directed_edges"] == attempt["native_base_directed_edges"] + 2 * attempt["retained_optional_pairs"]
    if policy == "laggedrisk25":
        assert row["warmup"]["passes"] == 1
        assert np.isfinite(arrays["cached_risk_before_forecast"]).all()


def test_future_truth_does_not_change_predictions_or_native_graphs():
    positions, types, metadata = rollout_fixture()
    changed = positions.copy(); changed[6:] += .05
    original, first = native.rollout(tiny_model(), positions, types, metadata, "laggedrisk25", 3, 2, trace_steps=(1, 2, 3))
    altered, second = native.rollout(tiny_model(), changed, types, metadata, "laggedrisk25", 3, 2, trace_steps=(1, 2, 3))
    np.testing.assert_array_equal(first["predicted_positions"], second["predicted_positions"])
    assert [row["directed_edge_sha256"] for row in original["attempts"]] == [row["directed_edge_sha256"] for row in altered["attempts"]]
    assert original["mse_per_step"] != altered["mse_per_step"]


def test_coordinate_failure_preserves_rejected_raw_output_and_no_full_mean(monkeypatch):
    positions, types, metadata = rollout_fixture()
    original = native.bridge.supplied
    count = [0]
    def failed_second_forecast(*args, **kwargs):
        output = original(*args, **kwargs)
        count[0] += 1
        if count[0] == 3:  # supplied parity, forecast1, forecast2
            output["prediction"][0, 0] = 11.
        return output
    monkeypatch.setattr(native.bridge, "supplied", failed_second_forecast)
    row, arrays = native.rollout(tiny_model(), positions, types, metadata, "base", 3, 0)
    assert row["status"] == "failed" and row["completed_steps"] == 1
    assert row["failure"]["forecast_step"] == 2
    assert row["mean_rollout_mse"] is None and row["mse_at_final_horizon"] is None
    assert row["prefix_mean_mse_if_failed"] is not None
    assert arrays["rejected_prediction"][0, 0] == 11.
    assert arrays["rejected_history"].shape == (6, len(types), 2)


def test_native_parity_failure_retains_both_comparison_outputs():
    positions, types, metadata = rollout_fixture()
    model = tiny_model()
    original = model.predict_positions_with_variance
    def altered(*args, **kwargs):
        prediction, risk = original(*args, **kwargs)
        return prediction + .01, risk
    model.predict_positions_with_variance = altered
    row, arrays = native.rollout(model, positions, types, metadata, "base", 3, 0)
    assert row["status"] == "failed" and row["failure"]["category"] == "native_parity_failure"
    assert row["completed_steps"] == 0 and row["total_network_passes"] == 0
    assert "initial_parity_native_prediction" in arrays and "initial_parity_supplied_prediction" in arrays
    assert not np.array_equal(arrays["initial_parity_native_prediction"], arrays["initial_parity_supplied_prediction"])


def test_native_prefix_can_have_fewer_self_edges_when_cap_ties_match_native():
    history = np.full((6, 130, 2), .2, dtype=np.float32)
    _, edge, _, audit = native.native_graph(history, "base", None, np.random.default_rng(0))
    assert audit["native_base_self_edges"] == 128
    assert np.sum(edge[0] == edge[1]) == 128


def test_graph_guard_retains_failed_input_history_without_forward_output(monkeypatch):
    positions, types, metadata = rollout_fixture()
    original = native.native_graph
    count = [0]
    def graph_guard(*args, **kwargs):
        count[0] += 1
        if count[0] == 2:
            raise native.full.RolloutGuard("candidate_pair_resource_guard", candidate_pairs=100001, limit=100000)
        return original(*args, **kwargs)
    monkeypatch.setattr(native, "native_graph", graph_guard)
    row, arrays = native.rollout(tiny_model(), positions, types, metadata, "base", 3, 0)
    assert row["failure"]["phase"] == "graph" and row["completed_steps"] == 1
    assert arrays["failed_input_history"].shape == (6, len(types), 2)
    assert arrays["failed_input_cached_risk"].shape == (len(types),)
    assert "rejected_prediction" not in arrays


def fake_rollout(failed_policy=None, failure_category="coordinate_guard"):
    calls = []
    def run(model, positions, types, metadata, policy, horizon, rng_seed, device):
        calls.append(policy)
        failed = policy == failed_policy
        completed = 0 if failed and failure_category == "native_parity_failure" else 2 if failed else horizon
        value = .01
        row = {"status": "failed" if failed else "complete", "failure": {"category": failure_category, "forecast_step": completed + 1} if failed else None,
               "policy": policy, "horizon": horizon, "completed_steps": completed, "rng_seed": rng_seed,
               "mse_at_steps": {str(step): value if completed >= step else None for step in native.full.TRACE_STEPS},
               "mse_at_final_horizon": None if failed else value, "mean_rollout_mse": None if failed else value,
               "mean_directed_edges": None if failed else 20., "total_wall_seconds": .1,
               "total_network_passes": completed, "forecast_network_passes": completed,
               "mse_per_step": [value] * completed}
        return row, {"retained_data": np.asarray([completed], dtype=np.int64)}
    return run, calls


def publication_fixture(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    args = SimpleNamespace(output_dir=directory, resume=False, device="cpu")
    protocol = {"objective": "faithful", "seed": 0, "checkpoint_sha256": "checkpoint",
                "scope": "synthetic test", "trajectory_ids": ["test:000003"], "input_files_sha256": {}}
    manifest = {"records": [{"id": f"test:{i:06d}"} for i in range(4)], "metadata": {}}
    trajectories = [None] * 3 + [(np.zeros((1001, 2, 2), dtype=np.float32), np.ones(2, dtype=np.int64))]
    return args, protocol, manifest, trajectories


def test_atomic_publication_hashes_and_scientific_failure_reuse(tmp_path, monkeypatch):
    args, protocol, manifest, trajectories = publication_fixture(tmp_path)
    fake, calls = fake_rollout("random25")
    monkeypatch.setattr(native, "rollout", fake)
    with native.full.RunLock(args.output_dir):
        native.run(args, protocol, None, trajectories, manifest, [3])
    result_path, status_path = args.output_dir / "result.json", args.output_dir / "status.json"
    result, status = json.loads(result_path.read_text()), json.loads(status_path.read_text())
    assert result["state"] == status["state"] == "complete"
    assert status["result_sha256"] == native.full.sha256(result_path)
    assert result["summary"]["random25"]["mean_rollout_mse"] is None
    assert result["summary"]["random25"]["failed_trajectories"] == 1
    before = {path.name: path.read_bytes() for path in args.output_dir.glob("trajectory_*")}
    args.resume = True
    with native.full.RunLock(args.output_dir):
        native.run(args, protocol, None, trajectories, manifest, [3])
    assert calls == list(native.POLICIES)
    assert before == {path.name: path.read_bytes() for path in args.output_dir.glob("trajectory_*")}


def test_existing_protocol_mismatch_and_tampered_trace_refuse_resume(tmp_path, monkeypatch):
    args, protocol, manifest, trajectories = publication_fixture(tmp_path)
    fake, _ = fake_rollout()
    monkeypatch.setattr(native, "rollout", fake)
    native.run(args, protocol, None, trajectories, manifest, [3])
    with pytest.raises(ValueError, match="resume"):
        native.run(args, protocol, None, trajectories, manifest, [3])
    args.resume = True
    with pytest.raises(ValueError, match="matching provenance"):
        native.run(args, {**protocol, "seed": 1}, None, trajectories, manifest, [3])
    (args.output_dir / "trajectory_000003_base.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="trace is missing or its SHA256 differs"):
        native.run(args, protocol, None, trajectories, manifest, [3])


def test_native_parity_failure_is_committed_then_refused_on_resume(tmp_path, monkeypatch):
    args, protocol, manifest, trajectories = publication_fixture(tmp_path)
    fake, calls = fake_rollout("base", "native_parity_failure")
    monkeypatch.setattr(native, "rollout", fake)
    with pytest.raises(RuntimeError, match="Native parity failed"):
        native.run(args, protocol, None, trajectories, manifest, [3])
    assert json.loads((args.output_dir / "trajectory_000003_base.json").read_text())["failure"]["category"] == "native_parity_failure"
    assert json.loads((args.output_dir / "status.json").read_text())["state"] == "error"
    args.resume = True
    with pytest.raises(RuntimeError, match="Retained native parity failure"):
        native.run(args, protocol, None, trajectories, manifest, [3])
    assert calls == ["base"]


def test_exclusive_run_lock_refuses_duplicate_writer(tmp_path):
    directory = tmp_path / "locked"
    with native.full.RunLock(directory):
        with pytest.raises(RuntimeError, match="already locked"):
            with native.full.RunLock(directory):
                raise AssertionError("Duplicate writer acquired lock")


def test_initial_state_guard_is_a_retained_failed_outcome():
    positions, types, metadata = rollout_fixture()
    positions[0, 0, 0] = np.nan
    row, arrays = native.rollout(tiny_model(), positions, types, metadata, "base", 3, 0)
    assert row["status"] == "failed" and row["completed_steps"] == 0
    assert row["failure"]["phase"] == "initial_state"
    assert row["native_parity_network_passes"] == row["total_network_passes"] == 0
    assert np.isnan(arrays["rejected_history"][0, 0, 0])
    assert row["initial_observed_boundary"] is None


def test_initial_graph_guard_is_saved_before_parity(monkeypatch):
    positions, types, metadata = rollout_fixture()
    def guard(*args, **kwargs):
        raise native.full.RolloutGuard("candidate_pair_resource_guard", candidate_pairs=100001, limit=100000)
    monkeypatch.setattr(native.bridge, "strict_pairs", guard)
    row, arrays = native.rollout(tiny_model(), positions, types, metadata, "base", 3, 0)
    assert row["status"] == "failed" and row["failure"]["phase"] == "initial_graph"
    assert row["native_parity_network_passes"] == 0
    np.testing.assert_array_equal(arrays["rejected_history"], positions[:6])


@pytest.mark.parametrize("name", ["protocol.json.tmp", "status.json.tmp", "trajectory_000003_base.npz.tmp"])
def test_existing_temporary_artifacts_are_never_overwritten(tmp_path, name):
    args, protocol, manifest, trajectories = publication_fixture(tmp_path)
    temporary = args.output_dir / name
    temporary.write_bytes(b"preserve interrupted attempt")
    with pytest.raises(ValueError, match="temporary artifact requires"):
        native.run(args, protocol, None, trajectories, manifest, [3])
    assert temporary.read_bytes() == b"preserve interrupted attempt"
