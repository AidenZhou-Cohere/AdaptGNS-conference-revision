import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from research import full_rollout as full
from research.budget_graph import candidates, directed
from gns.model_io import build_simulator


class StaticSimulator(torch.nn.Module):
    _connectivity_radius = .015

    def __init__(self, displacement=0., bad_risk=False):
        super().__init__()
        self.calls = []
        self.displacement = displacement
        self.bad_risk = bad_risk

    def _forward_with_edge_index(self, sequence, counts, types, senders, receivers, material):
        self.calls.append((sequence.detach().cpu().numpy().copy(), senders.clone(), receivers.clone()))
        raw = torch.arange(1, len(types) + 1, dtype=torch.float32, device=sequence.device)
        if self.bad_risk:
            raw[0] = float("nan")
        return sequence[:, -1] + self.displacement, raw

    def head_to_variance(self, raw):
        return raw.square()  # Ensures the evaluator calls the semantic converter.


def example():
    points = np.asarray([[.2 + .013 * x, .2 + .013 * y] for x in range(4) for y in range(4)], dtype=np.float32)
    positions = np.broadcast_to(points, (14, len(points), 2)).copy()
    metadata = {"bounds": [[.1, .9], [.1, .9]], "default_connectivity_radius": .015,
                "dim": 2, "vel_mean": [0, 0], "vel_std": [1, 1],
                "acc_mean": [0, 0], "acc_std": [1, 1]}
    return positions, np.full(len(points), 5), metadata


def test_original_fixed_feature_and_decoder_agree_on_identical_supplied_graph():
    positions, types, metadata = example()
    # Exercise nonconstant velocity history, different embeddings, unclipped
    # boundary features and nontrivial edge displacements on the same graph.
    node_scale = np.linspace(.5, 1.5, len(types), dtype=np.float32)
    for frame in range(6):
        positions[frame, :, 0] += frame * .00005 * node_scale
        positions[frame, :, 1] -= frame ** 2 * .00001 * node_scale
    types = np.arange(len(types)) % 9
    metadata["bounds"] = [[.19, .25], [.195, .26]]
    metadata["vel_mean"], metadata["vel_std"] = [.00001, -.00002], [.003, .004]
    torch.manual_seed(3)
    model = build_simulator(metadata, 6.7e-4, 6.7e-4, torch.device("cpu"), radius_backend="scipy_host").eval()
    graph = candidates(positions[5], .015, 1.267)
    pairs = np.concatenate((graph.base, graph.extra[:4]))
    edge = torch.as_tensor(directed(pairs))
    sequence = torch.as_tensor(np.ascontiguousarray(positions[:6].transpose(1, 0, 2)))
    particle_types = torch.as_tensor(types)
    counts = torch.tensor([len(types)])
    # Pin the fixed interface to exactly the evaluator's edges. This checks
    # duplicated original feature construction and decoder, not graph equality.
    model._compute_graph_connectivity = lambda *args: (edge[0], edge[1])
    with torch.no_grad():
        expected_position, expected_head = model.predict_positions_with_variance(sequence, counts, particle_types)
        actual_position, actual_variance = full.predict_supplied_graph(model, positions[:6], types, pairs, "cpu")
    np.testing.assert_array_equal(actual_position, expected_position.numpy())
    np.testing.assert_array_equal(actual_variance, model.head_to_variance(expected_head).numpy())


def test_every_budgeted_forecast_has_exact_pairs_and_risk_has_one_warmup():
    positions, types, metadata = example()
    graph = candidates(positions[5], .015, 1.267)
    budget = int(.25 * len(graph.extra))
    assert budget > 0
    for policy in full.POLICIES:
        model = StaticSimulator()
        row, trace = full.rollout(model, positions, types, metadata, policy, 8, 0)
        assert row["status"] == "complete"
        assert row["mean_rollout_mse"] == 0.
        assert row["total_network_passes"] == 8 + (policy == "laggedrisk25")
        assert row["forecast_network_passes"] == 8
        assert len(model.calls) == row["total_network_passes"]
        if policy in full.POLICIES[2:]:
            assert row["retained_optional_pairs_per_step"] == [budget] * 8
        if policy == "laggedrisk25":
            assert row["warmup"]["directed_edges"] == 2 * len(graph.base)
            assert row["warmup"]["total_wall_seconds"] > 0
            assert len(model.calls[1][1]) == 2 * (len(graph.base) + budget)
            np.testing.assert_array_equal(model.calls[0][0], model.calls[1][0])
        else:
            assert row["warmup"] is None
        assert row["mean_normalized_acceleration_variance_per_step"] == [93.5] * 8
        assert trace["forecast_steps"].tolist() == [1]
        for _, senders, receivers in model.calls:
            assert not bool((senders == receivers).any())
            edge_set = set(zip(senders.tolist(), receivers.tolist()))
            assert all((r, s) in edge_set for s, r in edge_set)


def test_future_ground_truth_does_not_affect_any_policy_prediction_or_allocation():
    positions, types, metadata = example()
    shifted = positions.copy()
    shifted[6:] += 1.
    for policy in full.POLICIES:
        actual, trace = full.rollout(StaticSimulator(), positions, types, metadata, policy, 8, 53)
        changed, changed_trace = full.rollout(StaticSimulator(), shifted, types, metadata, policy, 8, 53)
        assert actual["final_predicted_state_sha256"] == changed["final_predicted_state_sha256"]
        assert [x["predicted_state_sha256"] for x in actual["attempts"]] == [x["predicted_state_sha256"] for x in changed["attempts"]]
        assert actual["directed_edges_per_step"] == changed["directed_edges_per_step"]
        assert actual["mean_normalized_acceleration_variance_per_step"] == changed["mean_normalized_acceleration_variance_per_step"]
        np.testing.assert_array_equal(trace["predicted_positions"], changed_trace["predicted_positions"])
        assert changed["mean_rollout_mse"] > .99
        assert actual["ground_truth_boundary_per_step"][0]["fraction_particles_outside"] == 0
        assert changed["ground_truth_boundary_per_step"][0]["fraction_particles_outside"] == 1


def test_candidate_guard_runs_before_pair_allocation(monkeypatch):
    positions, types, metadata = example()
    def forbidden(*args):
        raise AssertionError("pair query must not allocate beyond the guard")
    monkeypatch.setattr(full, "candidates", forbidden)
    row, _ = full.rollout(StaticSimulator(), positions, types, metadata, "base", 8, 0, max_candidate_pairs=1)
    assert row["failure"]["category"] == "candidate_pair_resource_guard"
    assert row["completed_steps"] == 0
    assert row["total_network_passes"] == 0
    assert row["mean_rollout_mse"] is None


@pytest.mark.parametrize("policy", ["base", "laggedrisk25"])
def test_raw_nonfinite_risk_fails_even_when_mean_prediction_is_finite(policy):
    positions, types, metadata = example()
    row, _ = full.rollout(StaticSimulator(bad_risk=True), positions, types, metadata, policy, 8, 0)
    assert row["failure"]["category"] == "nonfinite_risk"
    assert row["completed_steps"] == 0
    assert row["mse_at_final_horizon"] is None


def test_coordinate_guard_preserves_failure_and_boundary_diagnostic():
    positions, types, metadata = example()
    row, _ = full.rollout(StaticSimulator(displacement=20), positions, types, metadata, "base", 8, 0)
    assert row["failure"]["category"] == "coordinate_resource_guard"
    assert row["attempts"][0]["predicted_boundary"]["fraction_particles_outside"] == 1
    assert row["completed_steps"] == 0
    assert row["mse_at_final_horizon"] is None


def test_boundary_reference_retains_tiny_dataset_excursions():
    positions, types, metadata = example()
    positions[:, 0, 0] = np.float32(.1 - 5e-7)
    row, _ = full.rollout(StaticSimulator(), positions, types, metadata, "base", 8, 0)
    pred = row["predicted_boundary_per_step"][0]
    truth = row["ground_truth_boundary_per_step"][0]
    assert pred == truth
    assert truth["fraction_particles_outside"] == 1 / len(types)
    assert truth["fraction_particles_outside_by_more_than_1e-6"] == 0


def test_summary_never_averages_survivors_or_duplicate_ids():
    positions, types, metadata = example()
    good, _ = full.rollout(StaticSimulator(), positions, types, metadata, "base", 8, 0)
    bad, _ = full.rollout(StaticSimulator(), positions, types, metadata, "base", 8, 0, max_candidate_pairs=1)
    good["trajectory_id"], bad["trajectory_id"] = "valid:0", "valid:1"
    summary = full.summarize([good, bad], ["valid:0", "valid:1"], ["base"])["base"]
    assert summary["failed_trajectories"] == 1
    assert not summary["all_sample_full_horizon_error_defined"]
    assert summary["mean_rollout_mse"] is None
    assert summary["mse_at_steps"]["1"] is None
    incomplete = full.summarize([good], ["valid:0", "valid:1"], ["base"])["base"]
    assert incomplete["mean_rollout_mse"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        full.summarize([good, good], ["valid:0", "valid:1"], ["base"])


def test_locked_test_ids_cannot_include_first_three_or_a_selected_subset():
    manifest = {"split": "test", "record_count": 30, "source": {"record_count": 30, "CRC_verified": True,
                "sha256": full.OFFICIAL_TEST_SHA256},
                "records": [{"id": f"test:{i}", "source_index": i} for i in range(30)]}
    assert full.select_records(manifest, "locked_test") == list(range(3, 30))
    for kwargs in [{"start_index": 0}, {"start_index": 4}, {"trajectory_ids": ["test:4"]}, {"max_trajectories": 1}]:
        with pytest.raises(ValueError):
            full.select_records(manifest, "locked_test", **kwargs)
    with pytest.raises(ValueError, match="valid split"):
        full.select_records(manifest, "pilot_validation")
    shortened = copy.deepcopy(manifest); shortened["records"] = shortened["records"][:7]
    with pytest.raises(ValueError, match="complete CRC-verified 30"):
        full.select_records(shortened, "locked_test")
    altered = copy.deepcopy(manifest); altered["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned official"):
        full.select_records(altered, "locked_test")


def test_locked_checkpoint_requires_actual_complete_budget_and_protocol(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Fixed protocol\n")
    run = {"scope": "bounded_full_data_100k", "steps": 100000, "seed": 0, "objective": "faithful",
           "metadata_sha256": "1" * 64,
           "research_protocol_sha256": full.sha256(protocol)}
    digest = hashlib.sha256(json.dumps(run, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = {"simulator_config": {"particle_dimensions": 2, "latent_dim": 128,
               "nmessage_passing_steps": 10, "nmlp_layers": 2, "mlp_hidden_dim": 128,
               "uncertainty_parameterization": "variance", "detach_variance_features": True},
               "training_config": {"loss": "faithful", "completed_optimizer_updates": 100000},
               "completed_steps": 100000, "full_training_schema": 1,
               "run_config": run, "run_config_sha256": digest}
    full.check_checkpoint(payload, "locked_test", protocol, "1" * 64)
    with pytest.raises(ValueError, match="metadata SHA256"):
        full.check_checkpoint(payload, "locked_test", protocol, "2" * 64)
    short = copy.deepcopy(payload); short["completed_steps"] = 100
    with pytest.raises(ValueError, match="complete fixed 100k"):
        full.check_checkpoint(short, "locked_test", protocol, "1" * 64)
    protocol.write_text("Changed protocol\n")
    with pytest.raises(ValueError, match="protocol hash"):
        full.check_checkpoint(payload, "locked_test", protocol, "1" * 64)


def test_discarded_warmup_mean_does_not_trigger_coordinate_resource_guard():
    positions, types, metadata = example()
    class LargeDiscardedMean(StaticSimulator):
        def _forward_with_edge_index(self, *args):
            prediction, risk = super()._forward_with_edge_index(*args)
            return (prediction + 20 if len(self.calls) == 1 else prediction), risk
    row, _ = full.rollout(LargeDiscardedMean(), positions, types, metadata, "laggedrisk25", 8, 0)
    assert row["status"] == "complete"
    assert row["mean_rollout_mse"] == 0.
    assert row["total_network_passes"] == 9


def test_resume_reuses_only_verified_finished_records_and_detects_tampering(tmp_path):
    positions, types, metadata = example()
    row, trace = full.rollout(StaticSimulator(), positions, types, metadata, "base", 8, 0)
    trace_path = tmp_path / "trajectory_000000_base.npz"
    np.savez_compressed(trace_path, **trace)
    row.update(trajectory_id="valid:0", source_index=0, n_particles=len(types), trace_file=trace_path.name,
               trace_sha256=full.sha256(trace_path), protocol_sha256="pinned_protocol")
    row_path = tmp_path / "trajectory_000000_base.json"
    full.atomic_json(row_path, row)
    compact = full.compact_record(row, row_path)
    assert "attempts" not in compact and "mse_per_step" not in compact
    # Recovery also supports an atomic row committed just before an index crash.
    assert full.recover_record(tmp_path, 0, "base", "valid:0", "pinned_protocol", 8) == compact
    assert full.recover_record(tmp_path, 0, "base", "valid:0", "pinned_protocol", 8, compact) == compact
    with pytest.raises(ValueError, match="identity/protocol"):
        full.recover_record(tmp_path, 0, "base", "valid:0", "new_protocol", 8, compact)
    changed = dict(row, total_wall_seconds=0.)
    full.atomic_json(row_path, changed)
    with pytest.raises(ValueError, match="SHA256 or compact"):
        full.recover_record(tmp_path, 0, "base", "valid:0", "pinned_protocol", 8, compact)
    full.atomic_json(row_path, row)
    trace_path.write_bytes(b"bad trace")
    with pytest.raises(ValueError, match="trace is missing or its SHA256"):
        full.recover_record(tmp_path, 0, "base", "valid:0", "pinned_protocol", 8, compact)


def test_kinematic_future_replacement_is_not_supported():
    positions, types, metadata = example()
    types[0] = 3
    with pytest.raises(ValueError, match="kinematic"):
        full.rollout(StaticSimulator(), positions, types, metadata, "base", 8, 0)
