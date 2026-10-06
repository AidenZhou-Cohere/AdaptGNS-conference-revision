"""Synthetic arrays and mocked outputs only; no model construction or dataset."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load("sand_graph_support_policy")


@pytest.fixture
def native():
    benchmark = load("benchmark_sand_graph_support_rollout")
    adapter, _ = benchmark.load_helpers(ROOT.parents[2] / "outputs/AdaptGNS")
    return adapter


def test_incoming_nonself_rms_uses_native_support_and_zero_for_isolated():
    velocities = np.asarray([[0., 0.], [3., 0.], [3., 4.], [8., 8.]])
    history = np.stack((np.zeros_like(velocities), velocities))
    edges = np.asarray([[0, 2, 1, 2], [1, 1, 2, 2]], dtype=np.int64)
    scores, counts = policy.relative_velocity_rms(np, history, edges, 1.)
    assert np.array_equal(counts, [0, 2, 1, 0])
    assert np.allclose(scores, [0, np.sqrt(12.5), 4, 0], rtol=0, atol=1e-15)
    assert scores.dtype == np.float64
    shifted = history.copy(); shifted[-1] += [9., -2.]
    assert np.array_equal(policy.relative_velocity_rms(np, shifted, edges, 1.)[0], scores)
    rotated = history @ np.array([[0., 1.], [-1., 0.]])
    assert np.array_equal(policy.relative_velocity_rms(np, rotated, edges, 1.)[0], scores)
    assert np.array_equal(policy.relative_velocity_rms(np, history, edges, .5)[0], 2*scores)


@pytest.mark.parametrize("dt", [0., -1., float("nan")])
def test_invalid_dt_rejected(dt):
    with pytest.raises(ValueError):
        policy.relative_velocity_rms(np, np.zeros((2, 3, 2)), np.empty((2, 0), dtype=np.int64), dt)


def test_exact_annulus_budget_native_prefix_and_lexicographic_ties(native):
    points = np.asarray([[.2+x*.016, .2+y*.016] for y in range(3) for x in range(3)], dtype=np.float32)
    history = np.broadcast_to(points, (6, 9, 2)).copy()
    graph, edges, optional, audit, scores = native.physical_graph(history, np.random.default_rng(0))
    _, base, _, _ = native.native_graph(history, "base", None, np.random.default_rng(0))
    budget = len(graph.extra)//4
    assert budget > 0 and len(optional) == budget
    assert np.array_equal(edges[:, :base.shape[1]], base)
    assert edges.shape[1] == base.shape[1] + 2*budget
    assert np.array_equal(optional, graph.extra[:budget])
    assert np.array_equal(scores, np.zeros(9))
    assert audit["native_base_prefix_preserved"] and audit["physical_score_dtype"] == "float64"
    assert np.array_equal(native.physical_graph(history,np.random.default_rng(42))[2],optional)


def test_cap_active_zero_annulus_keeps_exact_base(native):
    history = np.full((6, 132, 2), .2, dtype=np.float32)
    _, edges, optional, audit, _ = native.physical_graph(history, np.random.default_rng(0))
    _, original, _, _ = native.native_graph(history,"base",None,np.random.default_rng(0))
    assert np.array_equal(edges, original) and len(optional) == 0
    assert audit["native_base_receivers_above_cap_before_capping"] == 132


def test_core_five_rollouts_delegate_unchanged(native,monkeypatch):
    calls=[]
    def sentinel(*args,**kwargs):calls.append((args,kwargs));return "same", "objects"
    monkeypatch.setattr(native.original,"rollout",sentinel)
    for method in policy.CORE_POLICIES:
        assert native.rollout(None,None,None,{"dt":.0025},method,314,93000,"cpu",trace_steps=(1,314)) == ("same","objects")
        assert calls[-1][0][4] == method and calls[-1][1]["trace_steps"] == (1,314)


def test_physical_full_horizon_mock_rollout_and_guard_preservation(native,monkeypatch):
    points=np.asarray([[.2+x*.016,.2+y*.016] for y in range(3) for x in range(3)],dtype=np.float32)
    positions=np.broadcast_to(points,(9,9,2)).copy()
    types=np.full(9,6,dtype=np.int64)
    metadata={"dt":.0025,"bounds":[[.1,.9],[.1,.9]],"default_connectivity_radius":.015}
    model=SimpleNamespace(_connectivity_radius=.015,_boundaries=metadata["bounds"],eval=lambda:None)
    calls=[]
    def supplied(model,history,types,edges,device):
        calls.append(edges.copy())
        risk=np.ones(len(types),dtype=np.float32)
        return {"prediction":history[-1].copy(),"risk":risk,"raw_risk":risk.copy(),"operational_seconds":0.}
    monkeypatch.setattr(native.bridge,"supplied",supplied)
    monkeypatch.setattr(native.bridge,"native_parity",lambda m,h,t,e,d:({"passed":True},{"native_edges":e.copy()},supplied(m,h,t,e,d)))
    row,traces=native.rollout(model,positions,types,metadata,policy.PHYSICAL_POLICY,3,93000,"cpu",trace_steps=(1,3))
    assert row["status"] == "complete" and row["completed_steps"] == 3 and row["warmup"] is None
    assert row["forecast_network_passes"] == 3 and row["mean_rollout_mse"] == 0
    assert all(a["retained_optional_pairs"] == a["optional_pair_budget"] for a in row["attempts"])
    assert row["attempts"][0]["physical_score_dtype"] == "float64"
    def bad(*args,**kwargs):
        value=supplied(*args,**kwargs);value["prediction"][:]=np.nan;return value
    monkeypatch.setattr(native.bridge,"supplied",bad)
    row,traces=native.rollout(model,positions,types,metadata,policy.PHYSICAL_POLICY,3,93000,"cpu")
    assert row["status"] == "failed" and row["completed_steps"] == 0 and row["mean_rollout_mse"] is None
    assert "rejected_prediction" in traces and "rejected_edges" in traces
