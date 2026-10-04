import numpy as np
import torch

from research.pilot_rollout import POLICIES, integrate, rollout, summarize


class ZeroAcceleration(torch.nn.Module):
    def forward(self, features, edge_index, edge_features, faithful=False):
        return torch.zeros(len(features), 2), torch.ones(len(features))


def example():
    points = np.array([[.2, .2], [.213, .2], [.218, .2]], dtype=np.float32)
    positions = np.broadcast_to(points, (14, 3, 2)).copy()
    metadata = {"vel_mean": [0, 0], "vel_std": [1, 1], "acc_mean": [0, 0],
                "acc_std": [1, 1], "dt": .0025}
    return positions, metadata


def test_integrator_inverts_standardization_without_an_extra_dt():
    positions, metadata = example()
    metadata.update(acc_mean=[.01, -.02], acc_std=[.1, .2])
    normalized_acc = np.broadcast_to(np.array([1., 2.], dtype=np.float32), (3, 2))
    expected = 2 * positions[5] - positions[4] + np.array([.11, .38], dtype=np.float32)
    np.testing.assert_allclose(integrate(positions[:6], normalized_acc, metadata), expected)


def test_zero_acceleration_preserves_static_states():
    positions, metadata = example()
    result = rollout(ZeroAcceleration(), positions, metadata, "faithful", "base", 8, 0)
    assert result["status"] == "complete"
    assert result["completed_steps"] == 8
    assert result["mse_at_final_horizon"] == 0.


def test_future_ground_truth_never_feeds_back_into_predictions():
    positions, metadata = example()
    reference = rollout(ZeroAcceleration(), positions, metadata, "faithful", "laggedrisk25", 8, 0)
    changed = positions.copy()
    changed[6:] += 1
    counterfactual = rollout(ZeroAcceleration(), changed, metadata, "faithful", "laggedrisk25", 8, 0)
    assert reference["final_predicted_state_sha256"] == counterfactual["final_predicted_state_sha256"]
    assert reference["directed_edges_per_step"] == counterfactual["directed_edges_per_step"]
    assert counterfactual["mse_at_final_horizon"] > .99


def test_first_step_uses_base_only_for_lagged_risk_and_full_for_dense():
    positions, metadata = example()
    lagged = rollout(ZeroAcceleration(), positions, metadata, "faithful", "laggedrisk25", 8, 0)
    dense = rollout(ZeroAcceleration(), positions, metadata, "faithful", "dense", 8, 0)
    assert lagged["directed_edges_per_step"][0] == 4
    assert dense["directed_edges_per_step"][0] == 6


def test_candidate_resource_guard_retains_failure_without_final_accuracy():
    positions, metadata = example()
    result = rollout(ZeroAcceleration(), positions, metadata, "faithful", "base", 8, 0,
                     max_candidate_pairs=1)
    assert result["status"] == "failed"
    assert result["failure"]["category"] == "candidate_pair_resource_guard"
    assert result["completed_steps"] == 0
    assert result["mse_at_final_horizon"] is None
    assert result["mean_rollout_mse"] is None


def test_group_summary_does_not_average_only_surviving_trajectories():
    positions, metadata = example()
    rows = []
    for policy in POLICIES:
        for budget in (100000, 1):
            result = rollout(ZeroAcceleration(), positions, metadata, "faithful", policy, 8, 0,
                             max_candidate_pairs=budget)
            result["policy"] = policy
            rows.append(result)
    summary = summarize(rows, expected_trajectories=2)
    for policy in POLICIES:
        assert summary[policy]["completed_trajectories"] == 1
        assert summary[policy]["failed_trajectories"] == 1
        assert summary[policy]["mse_at_final_horizon"] is None
        assert summary[policy]["mean_rollout_mse"] is None
