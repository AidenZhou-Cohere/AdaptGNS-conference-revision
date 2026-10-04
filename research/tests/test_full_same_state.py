"""Tiny synthetic CPU fixtures; no checkpoints or reserved dataset access."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import spearmanr
import torch

from research import full_same_state as diagnostic


class TinyModel(torch.nn.Module):
    _connectivity_radius = .015
    _normalization_stats = {"acceleration": {"mean": torch.tensor([.01, -.02]), "std": torch.tensor([2., 4.])}}

    def __init__(self, bad_risk=False):
        super().__init__()
        self.calls = []
        self.bad_risk = bad_risk

    def _forward_with_edge_index(self, sequence, counts, types, senders, receivers, material):
        assert sequence.dtype == torch.float32
        self.calls.append(sequence.detach().cpu().numpy().copy())
        degree = torch.bincount(receivers, minlength=len(types)).float()
        # More edges make a larger positive prediction. Static truth makes the
        # dense intervention worse, testing signed rather than clipped benefit.
        shift = torch.stack((degree * .001, degree * .002), axis=-1)
        prediction = sequence[:, -1] + shift
        head = sequence[:, -1, 0] + 1.
        if self.bad_risk:
            head[0] = float("nan")
        return prediction, head

    def head_to_variance(self, head):
        return head.square()


def fixture():
    points = np.asarray([[.2 + .013*x, .2 + .013*y] for x in range(4) for y in range(4)], dtype=np.float32)
    positions = np.broadcast_to(points, (16, len(points), 2)).copy()
    # Distinguish histories without changing pair distances; future is static
    # at the most recent observed state used for target 7.
    positions[:6, :, 0] += np.arange(6, dtype=np.float32)[:, None] * .0001
    positions[6:, :, 0] += .0006
    return positions, np.full(len(points), 5), {"default_connectivity_radius": .015, "bounds": [[.1, .9], [.1, .9]]}


def frame(model=None, repeats=6):
    positions, types, metadata = fixture()
    return diagnostic.evaluate_frame(model or TinyModel(), positions, types, metadata, 7, 3, 0, 0, repeats=repeats)


def test_observed_lag_indexing_variance_conversion_and_no_target_input():
    model = TinyModel()
    row, arrays = frame(model, repeats=1)
    positions, _, _ = fixture()
    np.testing.assert_array_equal(arrays["current_observed_history"], positions[1:7])
    np.testing.assert_array_equal(arrays["previous_observed_history"], positions[:6])
    np.testing.assert_allclose(arrays["previous_observed_base_q"], (positions[5, :, 0] + 1.) ** 2)
    np.testing.assert_allclose(arrays["current_base_q"], (positions[6, :, 0] + 1.) ** 2)
    allowed = [positions[1:7].transpose(1, 0, 2), positions[:6].transpose(1, 0, 2)]
    assert all(any(np.array_equal(call, permitted) for permitted in allowed) for call in model.calls)
    assert sum(np.array_equal(call, allowed[1]) for call in model.calls) == 2  # warmup + one timed risk pass
    assert row["model_history_dtype"] == "float32"
    assert row["radius_classification_dtype"] == "float64"


def test_future_target_changes_only_residual_diagnostics_and_not_selected_pairs():
    positions, types, metadata = fixture()
    original, arrays = diagnostic.evaluate_frame(TinyModel(), positions, types, metadata, 7, 3, 0, 0, repeats=1)
    altered = positions.copy(); altered[7:] += .5
    changed, changed_arrays = diagnostic.evaluate_frame(TinyModel(), altered, types, metadata, 7, 3, 0, 0, repeats=1)
    assert original["observed_history_sha256"] == changed["observed_history_sha256"]
    assert original["previous_observed_history_sha256"] == changed["previous_observed_history_sha256"]
    for method in diagnostic.POLICIES:
        np.testing.assert_array_equal(arrays["pairs_" + method], changed_arrays["pairs_" + method])
        np.testing.assert_array_equal(arrays["prediction_" + method], changed_arrays["prediction_" + method])
    assert changed["accuracy"]["base"]["position_coordinate_mse"] > original["accuracy"]["base"]["position_coordinate_mse"]


def test_balanced_six_case_order_score_cost_and_fixed_random_repetitions():
    model = TinyModel()
    row, arrays = frame(model)
    assert row["status"] == "complete"
    assert len(row["warmup_calls"]) == 6 and len(row["timed_calls"]) == 36
    assert row["completed_network_passes"] == 49 and len(model.calls) == 49
    for method in diagnostic.TIMING_CASES:
        calls = [call for call in row["timed_calls"] if call["method"] == method]
        assert sorted(call["slot"] for call in calls) == list(range(6))
        assert len({call["pair_sha256"] for call in calls}) == 1
        assert row["repeat_consistency"][method]["maximum_absolute_prediction_difference"] == 0.
        assert row["timing_summary"][method]["end_to_end_seconds"]["observations"] == 6
        expected_passes = 2 if method == "previous-observed-base-risk25" else 1
        assert all(call["network_passes"] == expected_passes for call in calls)
        if expected_passes == 2:
            assert all(call["score_generation_seconds"] > 0 and call["score_graph_seconds"] > 0 for call in calls)
        else:
            assert all(call["score_generation_seconds"] == 0 for call in calls)
    assert row["natural_shared_base_maximum_prediction_difference"] == 0
    np.testing.assert_array_equal(arrays["pairs_base"], arrays["pairs_base_shared_superset"])


def test_exact_budget_mandatory_pairs_overlap_and_no_self_loops():
    row, arrays = frame(repeats=1)
    audit = row["graph_audit"]
    base = diagnostic.pair_set(arrays["candidate_base_pairs_float64"])
    extra = diagnostic.pair_set(arrays["candidate_annulus_pairs_float64"])
    k = int(.25 * len(extra))
    assert k > 0 and audit["optional_budget"] == k
    for method in diagnostic.POLICIES:
        chosen = diagnostic.pair_set(arrays["pairs_" + method])
        assert base <= chosen <= base | extra
        assert all(left < right for left, right in chosen)
        expected = 0 if method == "base" else len(extra) if method == "dense" else k
        assert len(chosen - base) == expected
        assert audit["policies"][method]["directed_edges"] == 2 * len(chosen)
    assert audit["overlaps"]["base__dense"]["optional_intersection"] == 0
    assert audit["natural_shared_float64_base_identical"]


def test_signed_benefit_and_saved_normalization_units_are_exact():
    row, arrays = frame(repeats=1)
    target = arrays["target_position"]
    scale = np.asarray(row["saved_acceleration_normalization"]["std"])
    for method in diagnostic.POLICIES:
        error = arrays["prediction_" + method] - target
        np.testing.assert_allclose(arrays["position_vector_se_" + method], (error ** 2).sum(-1))
        np.testing.assert_allclose(arrays["normalized_vector_se_" + method], ((error / scale) ** 2).sum(-1))
    expected = arrays["normalized_vector_se_base"] - arrays["normalized_vector_se_dense"]
    np.testing.assert_array_equal(arrays["signed_dense_benefit_normalized"], expected)
    assert row["benefit"]["mean_normalized_vector_benefit"] < 0
    assert row["benefit"]["negative_fraction"] == 1.


def test_average_rank_spearman_and_undefined_constant_vectors():
    x, y = [1, 1, 2, 3], [3, 2, 2, 1]
    assert diagnostic.spearman(x, y)["value"] == pytest.approx(spearmanr(x, y).statistic)
    assert diagnostic.spearman([1, 1, 1], [3, 2, 1])["value"] is None
    assert diagnostic.spearman([1], [2])["reason"] == "fewer_than_two_particles"
    with pytest.raises(ValueError, match="finite"):
        diagnostic.spearman([1, np.nan], [1, 2])


def test_optional_cutoff_ties_count_selected_and_rejected_boundary_pairs():
    positions, _, _ = fixture()
    graph = diagnostic.natural_candidates(positions[6], .015, True)
    k = int(.25 * len(graph.extra))
    scores = np.ones(graph.n_nodes)
    result = diagnostic.cutoff_ties(graph, scores, k)
    assert result["tied_pairs"] == len(graph.extra)
    assert result["selected_tied_pairs"] == k
    assert result["rejected_tied_pairs"] == len(graph.extra) - k
    assert result["boundary_tie"]
    assert diagnostic.cutoff_ties(graph, scores, 0)["cutoff_score"] is None


def test_candidate_guard_prevents_pair_allocation(monkeypatch):
    class CountOnlyTree:
        def __init__(self, positions):
            pass
        def count_neighbors(self, other, radius):
            return 100
        def query_pairs(self, *args, **kwargs):
            raise AssertionError("Must stop before allocating pairs")
    monkeypatch.setattr(diagnostic, "cKDTree", CountOnlyTree)
    with pytest.raises(diagnostic.full.RolloutGuard) as caught:
        diagnostic.natural_candidates(np.zeros((4, 2)), .015, True, max_pairs=1)
    assert caught.value.details["category"] == "candidate_pair_resource_guard"


def test_natural_base_search_uses_r_and_shared_reference_uses_expanded_search(monkeypatch):
    positions, types, _ = fixture()
    natural_calls = []
    original = diagnostic.natural_candidates
    def tracked(position, radius, expanded, max_pairs):
        natural_calls.append(expanded)
        return original(position, radius, expanded, max_pairs)
    monkeypatch.setattr(diagnostic, "natural_candidates", tracked)
    current, previous = positions[1:7], positions[:6]
    model = TinyModel()
    base = diagnostic.run_policy(model, current, previous, types, "base", .015, [93000, 0, 3, 7])
    assert natural_calls == [False]
    shared = diagnostic.run_policy(model, current, previous, types, diagnostic.REFERENCE, .015, [93000, 0, 3, 7])
    assert natural_calls == [False]  # shared case calls frozen expanded helper directly
    np.testing.assert_array_equal(base["pairs"], shared["pairs"])


def test_float32_boundary_difference_is_recorded_without_changing_companion_pairs():
    # This stored float32 point lies just outside radius .015 in float64,
    # while float32 squared-distance classification rounds onto the boundary.
    points = np.asarray([[0., 0.], [.014999995008111, 1.2252209671714809e-5]], dtype=np.float32)
    positions = np.broadcast_to(points, (8, 2, 2)).copy()
    row, arrays = diagnostic.evaluate_frame(TinyModel(), positions, np.full(2, 5),
        {"default_connectivity_radius": .015}, 7, 3, 0, 0, repeats=1)
    assert row["status"] == "complete"
    assert row["graph_audit"]["natural_shared_float64_base_identical"]
    assert row["graph_audit"]["frozen_float32_comparison"]["base_symmetric_difference"] == 1
    assert len(arrays["pairs_base"]) == 0
    assert len(arrays["frozen_float32_base_pairs"]) == 1


def test_failure_preserves_call_context_and_never_creates_accuracy():
    row, arrays = frame(TinyModel(bad_risk=True), repeats=1)
    assert row["status"] == "failed" and row["accuracy"] is None
    assert row["failure"]["category"] == "nonfinite_risk"
    assert row["failure"]["warmup"] is True and row["failure"]["round"] == -1
    assert row["failure"]["phase"] in ("current_forward", "previous_score_forward")
    assert len(row["warmup_calls"]) == 1
    assert row["failed_call_attempted_network_passes"] == 1


def test_later_timed_failure_keeps_warmups_and_earlier_calls_without_partial_accuracy():
    class LaterFailure(TinyModel):
        def _forward_with_edge_index(self, *args):
            prediction, risk = super()._forward_with_edge_index(*args)
            if len(self.calls) == 10:
                risk[0] = float("nan")
            return prediction, risk
    model = LaterFailure()
    row, _ = frame(model, repeats=2)
    assert row["status"] == "failed"
    assert len(row["warmup_calls"]) == 6 and all(call["status"] == "complete" for call in row["warmup_calls"])
    assert row["timed_calls"][0]["status"] == "complete"
    assert row["timed_calls"][-1]["status"] == "failed"
    assert row["failure"]["warmup"] is False and row["failure"]["round"] == 0
    assert row["accuracy"] is None and row["timing_summary"] is None
    assert row["completed_network_passes"] + row["failed_call_attempted_network_passes"] == len(model.calls) == 10


def test_summary_distinguishes_particle_weighting_and_preserves_failures_and_undefined_correlations():
    original, _ = frame(repeats=1)
    rows, expected = [], []
    for source, count, value in [(3, 1, 1.), (4, 9, 3.)]:
        row = copy.deepcopy(original)
        row.update(source_index=source, trajectory_id=f"test:{source}", n_particles=count)
        for method in diagnostic.POLICIES:
            row["accuracy"][method]["position_coordinate_mse"] = value
        rows.append(row)
        expected.append({"source_index": source, "target_frame": 7, "trajectory_id": f"test:{source}"})
    result = diagnostic.summarize_frames(rows, expected)
    metric = result["accuracy"]["base"]["position_coordinate_mse"]
    assert metric["equal_trajectory_mean"] == 2. and metric["particle_weighted_mean"] == 2.8
    name = diagnostic.CORRELATIONS[0]
    rows[0]["correlations"][name] = {"value": None, "reason": "constant_rank_vector", "particles": 1}
    result = diagnostic.summarize_frames(rows, expected)
    assert result["correlations"][name]["undefined_completed_frames"] == 1
    assert result["correlations"][name]["equal_trajectory_mean_of_frame_spearman"] is None
    rows[0].update(status="failed", failure={"category": "nonfinite_risk"})
    result = diagnostic.summarize_frames(rows, expected)
    assert not result["accuracy_and_runtime_aggregates_defined"]
    assert result["failed_frames"] == 1 and len(result["failures"]) == 1
    assert result["accuracy"]["base"]["position_coordinate_mse"]["equal_trajectory_mean"] is None
    assert result["timing"]["base"]["end_to_end_seconds"]["equal_trajectory_mean"] is None
    missing = diagnostic.summarize_frames(rows[1:], expected)
    assert len(missing["missing_frames"]) == 1
    assert missing["accuracy"]["base"]["position_coordinate_mse"]["equal_trajectory_mean"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostic.summarize_frames([rows[1], rows[1]], expected)


@pytest.mark.parametrize("bad_risk", [False, True])
def test_atomic_resume_reuses_complete_and_guard_failed_frames_without_new_model_calls(tmp_path, bad_risk):
    positions, types, metadata = fixture()
    model = TinyModel(bad_risk=bad_risk)
    args = SimpleNamespace(output_dir=tmp_path, resume=False, target_frames=[7], repeats=1,
                           max_candidate_pairs=100000, max_abs_coordinate=10., scope="pilot_validation")
    protocol = {"checkpoint_sha256": "synthetic_checkpoint", "objective": "faithful", "schema": 1}
    manifest = {"records": [{"id": "valid:0"}]}
    diagnostic.evaluate_to_directory(args, protocol, model, [(positions, types)], manifest, [0], metadata, 0, "cpu")
    original_calls = len(model.calls)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["state"] == "complete"
    assert saved["records"][0]["status"] == ("failed" if bad_risk else "complete")
    args.resume = True
    diagnostic.evaluate_to_directory(args, protocol, model, [(positions, types)], manifest, [0], metadata, 0, "cpu")
    assert len(model.calls) == original_calls
    assert json.loads((tmp_path / "result.json").read_text())["reused_frames"] == 1
    with pytest.raises(ValueError, match="exactly matching"):
        diagnostic.evaluate_to_directory(args, dict(protocol, changed=True), model, [(positions, types)], manifest, [0], metadata, 0, "cpu")
    path = tmp_path / "trajectory_000000_target_0007.npz"
    path.write_bytes(b"corrupted fixture")
    with pytest.raises(ValueError, match="array file/hash"):
        diagnostic.evaluate_to_directory(args, protocol, model, [(positions, types)], manifest, [0], metadata, 0, "cpu")


def test_resume_recovers_committed_frame_orphan_and_rejects_index_tampering(tmp_path):
    row, arrays = frame(repeats=1)
    stem = "trajectory_000003_target_0007"
    np.savez_compressed(tmp_path / (stem + ".npz"), **arrays)
    row.update(trajectory_id="test:3", protocol_sha256="synthetic_protocol", array_file=stem + ".npz",
               array_sha256=diagnostic.full.sha256(tmp_path / (stem + ".npz")))
    path = tmp_path / (stem + ".json")
    diagnostic.full.atomic_json(path, row)
    item = {"source_index": 3, "target_frame": 7, "trajectory_id": "test:3"}
    recovered = diagnostic.recover_frame(tmp_path, item, "synthetic_protocol")
    assert recovered == diagnostic.compact_frame(row, path)
    bad_index = dict(recovered, record_sha256="wrong")
    with pytest.raises(ValueError, match="hash/index"):
        diagnostic.recover_frame(tmp_path, item, "synthetic_protocol", bad_index)
