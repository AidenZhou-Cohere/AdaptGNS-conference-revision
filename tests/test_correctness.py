import math
import json
import copy

import numpy as np
import pytest
import torch

from gns.calibration import regression_calibration
from gns.losses import acceleration_loss
from gns.model_io import build_simulator, load_for_evaluation
from gns.learned_simulator import LearnedSimulator
from gns.graph_network import VarianceHead
from evaluate_rollout_mse import rollout_with_step_mse
from evaluate_adaptive_rollout import adaptive_rollout_mse


def test_nll_matches_multivariate_gaussian_and_variance_optimum():
    predicted = torch.tensor([[3., 4.], [1., -2.]], dtype=torch.float64)
    target = torch.zeros_like(predicted)
    variance = torch.tensor([12.5, 2.5], dtype=torch.float64, requires_grad=True)
    actual = acceleration_loss(predicted, target, torch.ones(2, dtype=torch.bool), variance)
    distribution = torch.distributions.MultivariateNormal(
        predicted, covariance_matrix=variance[:, None, None] * torch.eye(2))
    expected = -distribution.log_prob(target).mean() - math.log(2 * math.pi)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(variance.grad, torch.zeros_like(variance))


def test_legacy_objective_is_explicitly_preserved():
    predicted, target = torch.tensor([[3., 4.]]), torch.zeros(1, 2)
    head = torch.tensor([2.])
    variance = head.square() + 1e-6
    expected = (25 / (2 * variance) + 0.5 * variance.log()).mean()
    actual = acceleration_loss(predicted, target, torch.tensor([True]), head,
                               loss_type="legacy_nll")
    torch.testing.assert_close(actual, expected)


def test_kinematic_nans_are_excluded_before_loss_arithmetic():
    predicted = torch.tensor([[2., 3.], [float("nan"), float("nan")]], requires_grad=True)
    loss = acceleration_loss(predicted, torch.zeros_like(predicted),
                             torch.tensor([True, False]), loss_type="mse")
    loss.backward()
    assert loss.item() == 13
    torch.testing.assert_close(predicted.grad[1], torch.zeros(2))
    with pytest.raises(ValueError, match="dynamic particle"):
        acceleration_loss(predicted, predicted, torch.tensor([False, False]))


def test_quantile_is_trajectory_local_and_ties_do_not_expand():
    scores = torch.tensor([0., 1., 2., 10., 11., 12.])
    mask = LearnedSimulator._select_high_uncertainty(scores, [3, 3], 50)
    assert mask.tolist() == [False, False, True, False, False, True]
    assert not LearnedSimulator._select_high_uncertainty(torch.ones(3), [3], 70).any()
    with pytest.raises(ValueError):
        LearnedSimulator._select_high_uncertainty(scores, [6], 101)


def test_graph_union_preserves_base_and_batches(simulator):
    positions = torch.tensor([[0., 0.], [.8, 0.], [1.5, 0.], [0., 0.]])
    base = simulator._compute_graph_connectivity(positions, [3, 1], 1.)
    adaptive = simulator._build_adaptive_edge_index(
        positions, [3, 1], torch.tensor([True, False, False, False]), 1.6)
    base_pairs = set(map(tuple, torch.stack(base).T.tolist()))
    adaptive_pairs = set(map(tuple, torch.stack(adaptive).T.tolist()))
    assert base_pairs <= adaptive_pairs
    assert (0, 2) in adaptive_pairs and (2, 0) in adaptive_pairs
    assert (0, 3) not in adaptive_pairs and (3, 3) in adaptive_pairs
    assert len(adaptive_pairs) == len(adaptive[0])


def test_feature_path_and_rollout_agree_on_base_graph(simulator):
    history = torch.tensor([[0., 0.], [.8, 0.], [1.5, 0.]])[:, None].repeat(1, 6, 1)
    particle_types = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        direct = simulator.predict_positions(history, [3], particle_types)
        edges = simulator._compute_graph_connectivity(history[:, -1], [3], 1.)
        via_graph, _ = simulator._forward_with_edge_index(history, [3], particle_types, *edges)
    torch.testing.assert_close(direct, via_graph)


def test_one_neural_pass_per_rollout_step_and_partial_horizons(simulator):
    positions = torch.tensor([[0., 0.], [.8, 0.]])[:, None].repeat(1, 10, 1)
    particle_types = torch.tensor([0, 3])
    calls = []
    hook = simulator._encode_process_decode.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        mse, edges = adaptive_rollout_mse(simulator, positions, particle_types, None,
                                         2, 2, "cpu", 70, 1.267)
        assert len(calls) == 2 and mse.shape == edges.shape == (2,)
        fixed_mse, _ = rollout_with_step_mse(simulator, positions, particle_types,
                                            None, 2, 2, "cpu")
        assert len(calls) == 4
        np.testing.assert_allclose(fixed_mse[0], mse[0])
    hook.remove()
    with pytest.raises(ValueError, match="nsteps"):
        adaptive_rollout_mse(simulator, positions, particle_types, None,
                             2, 0, "cpu", 70, 1.267)


def test_checkpoint_preserves_architecture_and_normalization(simulator, tmp_path):
    path = tmp_path / "model.pt"
    simulator._training_config = {"loss": "nll", "seed": 42}
    simulator.save(path)
    restored, provenance = load_for_evaluation(path, {"unused": True}, "cpu")
    assert provenance["normalization_source"] == "checkpoint"
    assert provenance["training_config"]["seed"] == 42
    assert restored._checkpoint_config["latent_dim"] == 8
    torch.testing.assert_close(restored._normalization_stats["acceleration"]["std"],
                               simulator._normalization_stats["acceleration"]["std"])
    with pytest.raises(ValueError, match="already store"):
        load_for_evaluation(path, {}, "cpu", normalization_noise_std=0.)
    simulator._training_config["loss"] = "mse"
    simulator.save(path)
    with pytest.raises(ValueError, match="untrained variance head"):
        load_for_evaluation(path, {}, "cpu")
    load_for_evaluation(path, {}, "cpu", allow_missing_variance_head=True)


def test_legacy_upstream_checkpoint_only_allows_missing_variance(simulator, tmp_path):
    path = tmp_path / "legacy.pt"
    state = {k: v for k, v in simulator.state_dict().items() if "_variance_head." not in k}
    torch.save(state, path)
    with pytest.warns(UserWarning), pytest.raises(RuntimeError):
        simulator.load(path)
    with pytest.warns(UserWarning):
        simulator.load(path, allow_missing_variance_head=True)
    del state["_particle_type_embedding.weight"]
    torch.save(state, path)
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="Unexpected checkpoint"):
        simulator.load(path, allow_missing_variance_head=True)


def test_noise_is_part_of_normalization_even_without_input_perturbations():
    metadata = {"dim": 2, "bounds": [[0, 1], [0, 1]], "default_connectivity_radius": .015,
                "acc_mean": [0, 0], "acc_std": [6.545267e-5, 7.965495e-5],
                "vel_mean": [0, 0], "vel_std": [.0014, .0013]}
    simulator = build_simulator(metadata, 6.7e-4, 6.7e-4, "cpu", nmessage_passing_steps=0)
    scale = simulator._normalization_stats["acceleration"]["std"]
    assert (scale / torch.tensor(metadata["acc_std"]) > 8).all()


def test_calibration_keeps_ties_and_includes_maximum():
    ece, bins = regression_calibration(np.array([2., 2., 8., 8.]),
                                       np.array([1., 1., 4., 4.]), 2, n_bins=10)
    assert ece == 0 and sum(b[0] for b in bins) == 4
    constant_ece, constant_bins = regression_calibration(np.full(3, 4.), np.full(3, 2.), 2)
    assert constant_ece == 0 and len(constant_bins) == 1


def test_variance_head_parameter_count_matches_released_architecture():
    assert sum(p.numel() for p in VarianceHead(128).parameters()) == 33153


def test_training_step_saves_completed_update_and_loss_config(exact_radius, tmp_path):
    from gns import train
    data_path, model_path = tmp_path / "data", tmp_path / "models"
    data_path.mkdir()
    model_path.mkdir()
    metadata = {"dim": 2, "bounds": [[0, 1], [0, 1]], "default_connectivity_radius": .2,
                "acc_mean": [0, 0], "acc_std": [.01, .01],
                "vel_mean": [0, 0], "vel_std": [.01, .01]}
    (data_path / "metadata.json").write_text(json.dumps(metadata))
    data = np.empty(1, dtype=object)
    data[0] = (np.broadcast_to(np.array([[.2, .2], [.3, .2]], dtype=np.float32),
                               (7, 2, 2)).copy(), 0)
    np.savez(data_path / "train.npz", gns_data=data)
    settings = {name: train.FLAGS[name].value for name in train.FLAGS}
    settings.update(data_path=str(data_path) + "/", model_path=str(model_path) + "/",
                    ntraining_steps=1, nsave_steps=1, nmessage_passing_steps=1,
                    batch_size=1, validation_interval=None)
    train.train(None, settings, 1, torch.device("cpu"))
    assert (model_path / "model-1.pt").exists()
    assert not (model_path / "model-0.pt").exists()
    checkpoint = torch.load(model_path / "model-1.pt", weights_only=True)
    assert checkpoint["training_config"]["loss"] == "nll"
    state = torch.load(model_path / "train_state-1.pt", weights_only=True)
    assert state["global_train_state"]["step"] == 1


def test_faithful_mean_gradients_match_mse_and_risk_is_trainable(simulator):
    faithful = copy.deepcopy(simulator)
    faithful._encode_process_decode.detach_variance_features = True
    history = torch.tensor([[.2, .2], [.3, .2]])[:, None].repeat(1, 6, 1)
    kwargs = dict(next_positions=history[:, -1] + .1,
                  position_sequence_noise=torch.zeros_like(history),
                  position_sequence=history, nparticles_per_example=[2],
                  particle_types=torch.zeros(2, dtype=torch.long))
    for model, kind in [(simulator, "mse"), (faithful, "faithful")]:
        pred, var, target = model(**kwargs)
        acceleration_loss(pred, target, torch.ones(2, dtype=torch.bool), var,
                           loss_type=kind).backward()
    for (name, mse_param), (_, faithful_param) in zip(simulator.named_parameters(),
                                                     faithful.named_parameters()):
        if "_variance_head." in name:
            assert mse_param.grad is None
            assert faithful_param.grad is not None
        else:
            torch.testing.assert_close(mse_param.grad, faithful_param.grad)


def test_real_scipy_backend_matches_tiny_radius_oracle(simulator):
    points = torch.tensor([[0., 0.], [.8, 0.], [1.5, 0.], [0., 0.]])
    expected = simulator._compute_graph_connectivity(points, [3, 1], 1.)
    simulator._radius_backend = "scipy"
    actual = simulator._compute_graph_connectivity(points, [3, 1], 1.)
    assert set(map(tuple, torch.stack(expected).T.tolist())) == set(map(tuple, torch.stack(actual).T.tolist()))


def test_full_checkpoint_budget_policies_match_edge_counts(simulator):
    from evaluate_budget_policies import evaluate_state
    history = torch.tensor([[0., 0.], [.8, 0.], [1.5, 0.]])[:, None].repeat(1, 6, 1)
    row, risk = evaluate_state(simulator, history, history[:, -1],
                               torch.zeros(3, dtype=torch.long), None,
                               np.array([0., 1., 2.]), np.random.default_rng(0),
                               radius_factor=2., extra_budget=1)
    controls = [row["policies"][p] for p in ("random", "speed", "lagged_risk")]
    assert {r["directed_edges_including_self"] for r in controls} == {9}
    assert {r["incremental_undirected_pairs"] for r in controls} == {1}
    assert risk.shape == (3,)
