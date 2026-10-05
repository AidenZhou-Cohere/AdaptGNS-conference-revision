"""Synthetic CPU tests only: no real checkpoint or dataset access."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research import graph_convention_bridge as bridge
from gns.learned_simulator import LearnedSimulator


def tiny_model(cap=128):
    torch.manual_seed(735)
    stats = {name: {"mean": torch.tensor([.01, -.02]), "std": torch.tensor([.3, .4])}
             for name in ("velocity", "acceleration")}
    return LearnedSimulator(particle_dimensions=2, nnode_in=14, nedge_in=3,
        latent_dim=8, nmessage_passing_steps=1, nmlp_layers=1, mlp_hidden_dim=8,
        connectivity_radius=.015, boundaries=[[.1, .9], [.1, .9]], normalization_stats=stats,
        nparticle_types=1, particle_type_embedding_size=0, max_num_neighbors=cap,
        device="cpu", uncertainty_parameterization="variance", radius_backend="scipy_host").eval()


def fixture():
    points = np.asarray([[.2 + .013*x, .2 + .013*y] for x in range(3) for y in range(3)], dtype=np.float32)
    positions = np.broadcast_to(points, (8, len(points), 2)).copy()
    positions[:, :, 0] += np.arange(8, dtype=np.float32)[:, None] * .0001
    return positions[1:7], positions[:6], np.ones(len(points), dtype=np.int64), positions[7]


def identity(split="test"):
    return {"split": split, "source_index": 3, "target_frame": 7, "trajectory_id": f"{split}:000003"}


def evaluate(model=None, target_change=0.):
    current, previous, types, target = fixture()
    return bridge.evaluate_frame(model or tiny_model(), current, previous, types,
                                 target + target_change, identity(), 0, "cpu")


@pytest.mark.parametrize("loops", [False, True])
@pytest.mark.parametrize("cap", [None, 128])
def test_ordered_edges_equal_native_strict_graph_including_cap_and_ties(loops, cap):
    # Radius is binary-exact. Contains a point exactly on the boundary and
    # enough coincident points for the cap to bind with many distance ties.
    radius = .125
    points = np.zeros((132, 2), dtype=np.float32)
    points[-2, 0] = radius
    points[-1, 0] = np.nextafter(np.float32(radius), np.float32(0))
    model = tiny_model(cap=128 if cap else 10000)
    senders, receivers = model._compute_graph_connectivity(torch.from_numpy(points), [len(points)], radius, loops)
    graph = bridge.strict_pairs(points, radius)
    expected = torch.stack((senders, receivers)).numpy()
    actual = bridge.ordered_edges(points, graph.base, loops, cap)
    np.testing.assert_array_equal(actual, expected)
    # Coincident ties can remove a capped node's self candidate, exactly as
    # native code does. Only the uncapped policy guarantees one self edge.
    if loops and cap is None:
        assert np.sum(actual[0] == actual[1]) == len(points)
    if loops and cap == 128:
        assert np.sum(actual[0] == actual[1]) < len(points)


def test_strict_base_query_does_not_admit_expanded_query_only_boundary_pair():
    points = np.asarray([[0., 0.], [.125, 0.], [0., .124]], dtype=np.float32)
    graph = bridge.strict_pairs(points, .125)
    assert (0, 1) not in bridge.same.pair_set(graph.base)
    assert (0, 1) in bridge.same.pair_set(graph.extra)


def test_empty_graph_shapes_and_invalid_pair_inputs():
    points = np.asarray([[0., 0.], [1., 1.]], dtype=np.float32)
    graph = bridge.strict_pairs(points, .01)
    assert graph.base.shape == graph.extra.shape == (0, 2)
    assert bridge.ordered_edges(points, graph.base, False).shape == (2, 0)
    assert bridge.ordered_edges(points, graph.base, True).shape == (2, 2)
    for bad in (np.array([[0, 0]]), np.array([[1, 0]]), np.array([[0, 1], [0, 1]])):
        with pytest.raises(ValueError, match="canonical"):
            bridge.ordered_edges(points, bad, True)


def test_tiny_native_supplied_feature_rawrisk_and_prediction_parity():
    row, arrays = evaluate()
    assert row["native_parity"]["passed"] and row["status"] == "complete"
    assert row["native_parity"]["native_supplied_feature_identity"] == [True, True, True]
    assert row["native_parity"]["raw_risk_max_abs_difference"] == 0
    np.testing.assert_array_equal(arrays["native_node_features"], arrays["supplied_node_features"])
    np.testing.assert_array_equal(arrays["native_edge_features"], arrays["supplied_edge_features"])
    assert set(row["cases"]) == set(bridge.CASE_NAMES)
    assert len(row["cases"]) == 16


def test_loop_and_exact_budget_invariants_for_all_uncapped_policies():
    row, arrays = evaluate()
    n, k = row["n_particles"], row["optional_budget"]
    base = bridge.same.pair_set(arrays["strict_base_pairs"])
    extra = bridge.same.pair_set(arrays["strict_annulus_pairs"])
    assert k > 0
    for loop in (0, 1):
        for policy in bridge.POLICIES:
            name = f"{policy}_uncapped_loops{loop}"
            edge = arrays[f"edges__{name}"]
            self_count = np.sum(edge[0] == edge[1])
            assert self_count == n * loop
            pair_set = {tuple(sorted(pair)) for pair in edge.T if pair[0] != pair[1]}
            assert base <= pair_set <= base | extra
            optional = 0 if policy == "base" else len(extra) if policy == "dense" else k
            assert len(pair_set - base) == optional
            assert edge.shape[1] == 2 * (len(base) + optional) + n * loop
            case = row["cases"][name]
            assert case["network_passes_for_standalone_policy"] == (2 if "risk" in policy else 1)
    for policy in ("random25", "speed25"):
        np.testing.assert_array_equal(arrays[f"pairs__{policy}_uncapped_loops0"], arrays[f"pairs__{policy}_uncapped_loops1"])


def test_same_graph_prediction_reuse_is_exact_and_recorded():
    row, arrays = evaluate()
    for loops in (0, 1):
        left, right = f"base_cap128_loops{loops}", f"base_uncapped_loops{loops}"
        assert row["cases"][right]["reused_from_case"] == left
        np.testing.assert_array_equal(arrays[f"prediction__{left}"], arrays[f"prediction__{right}"])
        assert row["base_graph_audit"][f"loops{loops}"]["capped_equals_uncapped"]
        assert row["base_graph_audit"][f"dense_loops{loops}"]["capped_equals_uncapped"]


def test_targets_do_not_affect_inference_graphs_scores_or_history():
    original, arrays = evaluate()
    changed, shifted = evaluate(target_change=.1)
    for key, value in arrays.items():
        if key.startswith(("edges__", "pairs__", "prediction__", "risk__", "selection_score__")):
            np.testing.assert_array_equal(value, shifted[key])
    assert original["target_sha256"] != changed["target_sha256"]
    assert original["cases"]["base_uncapped_loops0"]["metrics"] != changed["cases"]["base_uncapped_loops0"]["metrics"]


def test_residuals_keep_signed_coordinate_information_and_saved_scale():
    row, arrays = evaluate()
    for name in bridge.CASE_NAMES:
        residual = arrays[f"prediction__{name}"].astype(np.float64) - arrays["target_position"]
        np.testing.assert_array_equal(residual, arrays[f"position_residual__{name}"])
        normalized = residual / arrays["acceleration_std"]
        np.testing.assert_array_equal(normalized, arrays[f"normalized_residual__{name}"])
        assert row["cases"][name]["metrics"]["position_coordinate_mse"] == np.mean(residual ** 2)
        assert row["cases"][name]["metrics"]["normalized_coordinate_mse"] == np.mean(normalized ** 2)


def test_parity_failure_is_returned_with_raw_unsuccessful_outputs():
    model = tiny_model()
    native = model.predict_positions_with_variance
    def disagree(*args, **kwargs):
        prediction, raw = native(*args, **kwargs)
        return prediction + .01, raw
    model.predict_positions_with_variance = disagree
    row, arrays = evaluate(model)
    assert row["status"] == "failed" and row["failure"]["category"] == "native_parity_failure"
    assert not row["cases"]
    assert "native_prediction" in arrays and "supplied_parity_prediction" in arrays
    assert row["native_parity"]["prediction_max_abs_difference"] > .009


def test_rawrisk_disagreement_cannot_hide_behind_variance_conversion():
    model = tiny_model()
    native = model.predict_positions_with_variance
    def disagree(*args, **kwargs):
        prediction, raw = native(*args, **kwargs)
        return prediction, raw + .01
    model.predict_positions_with_variance = disagree
    model.head_to_variance = lambda raw: torch.ones_like(raw)
    row, _ = evaluate(model)
    assert row["status"] == "failed"
    assert row["native_parity"]["risk_max_abs_difference"] == 0
    assert row["native_parity"]["raw_risk_max_abs_difference"] > .009


def test_nonfinite_raw_output_is_retained_and_not_masked():
    output = {"prediction": np.zeros((2, 2)), "raw_risk": np.array([np.nan, 1.]), "risk": np.ones(2)}
    with pytest.raises(bridge.full.RolloutGuard, match="nonfinite_risk"):
        bridge.validate_output(output, 2)


def test_failed_record_recovery_and_tamper_refusal(tmp_path):
    item = identity()
    array = tmp_path / (bridge.stem(item) + ".npz")
    np.savez(array, failed_prediction=np.array([np.nan]))
    path = tmp_path / (bridge.stem(item) + ".json")
    row = {**item, "status": "failed", "failure": {"category": "native_parity_failure"},
           "protocol_sha256": "protocol", "array_file": array.name, "array_sha256": bridge.full.sha256(array)}
    bridge.full.atomic_json(path, row)
    recovered = bridge.recover(tmp_path, item, "protocol")
    assert recovered["status"] == "failed"
    np.savez(array, changed=np.array([0.]))
    with pytest.raises(ValueError, match="array/hash"):
        bridge.recover(tmp_path, item, "protocol")


def test_orphan_array_is_never_overwritten(tmp_path):
    item = identity()
    path = tmp_path / (bridge.stem(item) + ".npz")
    np.savez(path, prior=np.array([1.]))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Orphan"):
        bridge.recover(tmp_path, item, "x")
    assert path.read_bytes() == original


def test_candidate_guard_prevents_materialization(monkeypatch):
    class GuardTree:
        def __init__(self, _):
            pass
        def count_neighbors(self, *args):
            return 2 * (bridge.MAX_PAIRS + 1) + 2
        def query_pairs(self, *args, **kwargs):
            raise AssertionError("Guard must precede allocation")
    monkeypatch.setattr(bridge, "cKDTree", GuardTree)
    with pytest.raises(bridge.full.RolloutGuard, match="resource_guard"):
        bridge.strict_pairs(np.zeros((2, 2)), .1)
