"""Synthetic saved-result audit tests; no real outputs or models are read."""
import copy

import numpy as np
import pytest
import torch

from research import summarize_native_graph_rollouts as summary
from research.tests.test_graph_convention_bridge import tiny_model
from research.tests.test_native_graph_rollout import rollout_fixture


def short_result(policy="base", horizon=3):
    positions, types, metadata = rollout_fixture(horizon)
    row, traces = summary.native.rollout(tiny_model(), positions, types, metadata, policy, horizon, 93003)
    row.update(n_particles=len(types), source_index=3, trajectory_id="test:000003")
    return row, traces, positions, types


@pytest.mark.parametrize("policy", summary.native.POLICIES)
def test_native_saved_trajectory_recomputes_geometry_truth_timing_and_passes(policy):
    row, traces, positions, types = short_result(policy)
    audit = summary.Audit()
    metrics, prefix = summary.validate_trajectory(row, traces, positions, types, audit)
    assert audit.checks > 75 and prefix is None
    assert metrics["cap_active_step_fraction"] == 0
    assert metrics["mean_native_base_self_edges"] == len(types)
    assert metrics["mean_native_parity_passes"] == 2
    assert metrics["mean_wall_seconds_including_native_parity"] >= row["total_wall_seconds"]


@pytest.mark.parametrize("policy", ["random25", "laggedrisk25"])
def test_trace_gap_replays_intervening_random_draws_and_prior_risk(policy):
    positions, types, metadata = rollout_fixture(horizon=10)
    model = tiny_model()
    # Keep this synthetic autonomous trajectory inside the coordinate guard
    # while retaining nonempty optional budgets throughout the unstored gap.
    model._normalization_stats["acceleration"] = {
        "mean": torch.zeros(2), "std": torch.full((2,), .001)}
    row, traces = summary.native.rollout(model, positions, types, metadata,
        policy, 10, 93003, trace_steps=(1, 10))
    row.update(n_particles=len(types), source_index=3, trajectory_id="test:000003")
    assert row["status"] == "complete" and row["completed_steps"] == 10
    assert all(value > 0 for value in row["retained_optional_pairs_per_step"])
    np.testing.assert_array_equal(traces["forecast_steps"], [1, 10])
    audit = summary.Audit()
    _, prefix = summary.validate_trajectory(row, traces, positions, types, audit)
    assert prefix is None and audit.checks > 330


def test_native_parity_raw_maximum_tampering_rejected():
    row, traces, positions, types = short_result()
    row["native_parity"]["raw_risk_max_abs_difference"] = .01
    with pytest.raises(ValueError, match="parity maximum"):
        summary.validate_trajectory(row, traces, positions, types, summary.Audit())


def test_trace_graph_order_tampering_rejected_even_with_same_pair_set():
    row, traces, positions, types = short_result("dense")
    traces["edges_forecast_0001"] = traces["edges_forecast_0001"].copy()
    traces["edges_forecast_0001"][:, [0, 1]] = traces["edges_forecast_0001"][:, [1, 0]]
    with pytest.raises(ValueError, match="directed graph differs"):
        summary.validate_trajectory(row, traces, positions, types, summary.Audit())


def test_per_step_cap_removal_arithmetic_is_enforced():
    row, traces, positions, types = short_result()
    row["attempts"][1]["native_base_edges_removed_by_cap"] += 1
    with pytest.raises(ValueError, match="Cap removal arithmetic"):
        summary.validate_trajectory(row, traces, positions, types, summary.Audit())


def test_truth_boundary_must_match_actual_source():
    row, traces, positions, types = short_result()
    positions = positions.copy(); positions[7, 0, 0] = 1.1
    with pytest.raises(ValueError, match="Truth boundary differs"):
        summary.validate_trajectory(row, traces, positions, types, summary.Audit())


def test_raw_coordinate_failure_reconstructed_and_prefix_never_full_metric(monkeypatch):
    positions, types, metadata = rollout_fixture()
    original = summary.BRIDGE.supplied
    calls = [0]
    def reject(*args, **kwargs):
        output = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 3:
            output["prediction"][0, 0] = 11.
        return output
    monkeypatch.setattr(summary.BRIDGE, "supplied", reject)
    row, traces = summary.native.rollout(tiny_model(), positions, types, metadata, "base", 3, 93003)
    row.update(n_particles=len(types), source_index=3, trajectory_id="test:000003")
    metrics, prefix = summary.validate_trajectory(row, traces, positions, types, summary.Audit())
    assert metrics["mean_native_base_directed_edges"] is None
    assert prefix["mean_native_base_directed_edges"] is not None
    assert metrics["mean_wall_seconds_including_native_parity"] > 0
    row["failure"]["maximum_absolute_coordinate"] = 10.5
    with pytest.raises(ValueError, match="failure reason differs"):
        summary.validate_trajectory(row, traces, positions, types, summary.Audit())


def metric_rows(value=1., failed_index=None):
    rows = []
    for index in range(3, 30):
        failed = index == failed_index
        rows.append({"source_index": index, "trajectory_id": f"test:{index:06d}", "policy": "base",
            "status": "failed" if failed else "complete", "mean_rollout_mse": None if failed else value,
            "mse_at_steps": {"200": value, "995": None if failed else value}, "mean_directed_edges": None if failed else 10.,
            "total_wall_seconds": .1, "total_network_passes": 900 if failed else 995,
            "boundary_diagnostics": {"full_horizon": {name: None if failed else .01 for name in summary.original.BOUNDARY_METRICS}},
            "native_metrics": {name: None if failed else value for name in summary.EXTRA_METRICS}})
    return rows


def test_one_required_failure_nulls_full_mean_but_retains_all_sample_failure_fraction():
    result = summary.summarize_seed(metric_rows(failed_index=29), "base")
    assert result["metrics"]["mean_rollout_mse"] is None
    assert result["metrics"]["failure_fraction"] == pytest.approx(1/27)
    assert result["metrics"]["mse_at_200"] == 1.
    assert result["failed_trajectories"] == 1


def test_paired_comparisons_propagate_failures_and_preserve_seed_direction():
    left = [{"records": metric_rows(value=float(seed+2))} for seed in range(3)]
    right = [{"records": metric_rows(value=float(seed+1))} for seed in range(3)]
    result = summary.paired(left, right, "base", "base", ("mean_rollout_mse", "failure_fraction"))
    assert result["metrics"]["mean_rollout_mse"]["mean"] == 1.
    assert result["metrics"]["mean_rollout_mse"]["sample_sd"] == 0.
    left[2] = {"records": metric_rows(value=4., failed_index=8)}
    result = summary.paired(left, right, "base", "base", ("mean_rollout_mse", "failure_fraction"))
    assert result["metrics"]["mean_rollout_mse"]["mean"] is None
    assert result["metrics"]["failure_fraction"]["mean"] == pytest.approx(1/81)
    assert len(result["per_seed"][2]["trajectory_pairs"]) == 27


def test_paired_comparison_rejects_misaligned_source_identity():
    left = [{"records": metric_rows()} for _ in range(3)]
    right = copy.deepcopy(left)
    right[1]["records"][3]["trajectory_id"] = "wrong"
    with pytest.raises(ValueError, match="trajectory IDs differ"):
        summary.paired(left, right, "base", "base", ("mean_rollout_mse",))


def test_missing_required_trajectory_is_not_a_smaller_population():
    with pytest.raises(ValueError, match="Exactly 27"):
        summary.summarize_seed(metric_rows()[:-1], "base")


def test_initial_nan_guard_arrays_are_preserved_and_validated_as_failed_only():
    positions, types, metadata = rollout_fixture()
    positions[0, 0, 0] = np.nan
    row, traces = summary.native.rollout(tiny_model(), positions, types, metadata, "base", 3, 93003)
    row.update(n_particles=len(types), source_index=3, trajectory_id="test:000003")
    metrics, prefix = summary.validate_trajectory(row, traces, positions, types, summary.Audit())
    assert metrics["mean_native_parity_passes"] == 0
    assert metrics["mean_native_base_directed_edges"] is None
    assert prefix["mean_native_base_directed_edges"] is None


@pytest.mark.parametrize("key", ["retained_optional_pairs_per_step", "candidate_pairs_per_step", "base_pairs_per_step", "mean_normalized_acceleration_variance_per_step"])
def test_native_per_step_arrays_cannot_have_unchecked_extra_values(key):
    row, traces, positions, types = short_result()
    row[key] = list(row[key]) + [1]
    with pytest.raises(ValueError, match="array length differs"):
        summary.validate_trajectory(row, traces, positions, types, summary.Audit())


def test_current_trace_risk_must_be_finite_positive():
    row, traces, positions, types = short_result()
    traces["current_risk"][0, 0] = -1.
    with pytest.raises(ValueError, match="Invalid current trace risk"):
        summary.validate_trajectory(row, traces, positions, types, summary.Audit())


def test_native_protocol_self_hash_does_not_override_fixed_contract():
    protocol = copy.deepcopy(summary.NATIVE_CONTRACT)
    protocol.update(threads=2, runtime={"radius_backend": "scipy_host", "device": "mps"}, software={"numpy": "synthetic"})
    valid = summary.validate_native_contract(protocol, summary.Audit())
    assert len(valid) == 64
    for key in ("schema", "scope", "graph", "risk", "random_seed"):
        altered = copy.deepcopy(protocol)
        altered[key] = "unapproved alternative"
        with pytest.raises(ValueError, match="Native protocol contract differs"):
            summary.validate_native_contract(altered, summary.Audit())
