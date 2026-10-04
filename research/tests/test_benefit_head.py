import numpy as np
import torch

from research import pilot
from research.benefit_head import base_with_embedding, fit_ridge, predict_ridge
from research.budget_graph import candidates, select_pairs


def test_ridge_matches_independent_augmented_least_squares_with_constant_column():
    rng = np.random.default_rng(412)
    x = rng.normal(size=(70, 5))
    x[:, 4] = 2.0
    y = 3*x[:, 0]-2*x[:, 1]+0.7+rng.normal(size=len(x))*.2
    fit, diagnostics = fit_ridge(x, y, .03)
    z = (x-fit["feature_mean"])/fit["feature_scale"]
    design = np.column_stack((np.ones(len(x)), z))
    penalty = np.column_stack((np.zeros(5), np.eye(5))) * np.sqrt(len(x)*.03)
    coefficients = np.linalg.lstsq(np.vstack((design, penalty)), np.r_[y, np.zeros(5)], rcond=None)[0]
    np.testing.assert_allclose(predict_ridge(x, fit), design@coefficients, atol=1e-11)
    assert diagnostics["constant_features"] == 1
    assert diagnostics["normal_equation_residual"] < 1e-12


def test_embedding_capture_preserves_forward_and_model_parameters():
    torch.manual_seed(32)
    model = pilot.PilotGNS(width=8, depth=2).eval()
    pos = np.array([[.2,.2],[.21,.2],[.22,.2]], dtype=np.float32)
    frame = {"position": pos, "features": torch.randn(3,14), "graph": candidates(pos,.015)}
    before = {key: value.clone() for key, value in model.state_dict().items()}
    mean, risk = pilot.run_forward(model, frame, frame["graph"].base, True)
    captured_mean, captured_risk, embedding = base_with_embedding(model, frame, True)
    torch.testing.assert_close(mean, captured_mean, rtol=0, atol=0)
    torch.testing.assert_close(risk, captured_risk, rtol=0, atol=0)
    torch.testing.assert_close(model.mean(torch.from_numpy(embedding)), mean)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not model.mean._forward_pre_hooks


def test_signed_prediction_retains_exact_budget_and_training_fit_independence():
    training_x = np.array([[1.,0.],[2.,1.],[3.,-1.]])
    fit, _ = fit_ridge(training_x, np.array([-3.,-2.,-1.]))
    original = {key: value.copy() for key,value in fit.items()}
    scores = predict_ridge(np.array([[1.,0.],[2.,1.],[3.,-1.],[1.5,.5]]), fit)
    assert (scores < 0).all()
    graph = candidates(np.array([[0.,0.],[.9,0.],[1.8,0.],[2.7,0.]]), 1., 2.)
    selected = select_pairs(graph, scores, 1)
    assert len(selected) == len(graph.base)+1
    np.testing.assert_array_equal(selected[:len(graph.base)], graph.base)
    for key in fit:
        np.testing.assert_array_equal(fit[key], original[key])
