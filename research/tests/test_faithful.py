import torch
from research.pilot import PilotGNS, objective, clip_gradients


def test_faithful_preserves_every_mean_path_gradient():
    torch.manual_seed(1)
    model = PilotGNS(width=8, depth=1)
    feature = torch.randn(5, 14)
    edges = torch.tensor([[0,1,2,3,4], [1,2,3,4,0]])
    edge_features = torch.randn(5,3)
    target = torch.randn(5,2)
    mean, var = model(feature, edges, edge_features, faithful=True)
    objective(mean, var, target, "faithful").backward()
    actual = {n:p.grad.clone() for n,p in model.named_parameters() if not n.startswith("risk.")}
    model.zero_grad()
    mean, _ = model(feature, edges, edge_features, faithful=True)
    (.5 * (mean-target).square().sum(-1).mean()).backward()
    for name, parameter in model.named_parameters():
        if name in actual:
            torch.testing.assert_close(parameter.grad, actual[name], rtol=0, atol=0)


def test_nll_variance_stationary_at_squared_error_per_dimension():
    mean = torch.tensor([[2., 4.]])
    var = torch.tensor([10.], requires_grad=True)
    loss = objective(mean, var, torch.zeros_like(mean), "nll")
    loss.backward()
    torch.testing.assert_close(var.grad, torch.zeros_like(var), atol=1e-7, rtol=0)


def test_separate_clipping_preserves_mean_update_even_with_large_risk_gradients():
    torch.manual_seed(5)
    model = PilotGNS(width=8, depth=1)
    import copy
    mse_model = copy.deepcopy(model)
    x, edge_features = torch.randn(5,14), torch.randn(5,3)
    edges = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]])
    target = torch.ones(5,2) * 100
    for current, faithful in [(model, True), (mse_model, False)]:
        mean, var = current(x, edges, edge_features, faithful=True)
        loss = objective(mean,var,target,"faithful") if faithful else .5*(mean-target).square().sum(-1).mean()
        loss.backward()
        clip_gradients(current, "faithful")
        torch.optim.Adam(current.parameters(), lr=.001).step()
    for (name, actual), (_, expected) in zip(model.named_parameters(), mse_model.named_parameters()):
        if not name.startswith("risk."):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
