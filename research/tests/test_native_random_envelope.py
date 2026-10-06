"""Synthetic CPU envelope tests; no real checkpoint/data/inference runs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from research import native_random_envelope as evaluate
from research.tests.test_graph_convention_bridge import tiny_model, fixture, identity


def observed(model=None, target_change=0.):
    current, previous, types, target = fixture()
    return evaluate.evaluate_frame(model or tiny_model(), current, previous, types, target+target_change, identity(), 0, "cpu")


def test_all13_native_prefix_exact_draws_and_risk_history():
    row, arrays = observed()
    assert row["status"] == "complete" and row["native_parity"]["passed"]
    assert tuple(row["cases"]) == evaluate.POLICIES and len(row["cases"]) == 13
    current, previous, _, _ = fixture()
    base = arrays["edges__base"]
    for policy in evaluate.POLICIES:
        actual, score = evaluate.selection_spec(policy, {"risk": arrays["previous_base_risk"]}, {"risk": arrays["risk__base"]})
        material = row["random_seed_materials"].get(policy, [20261006,881,0])
        graph, edge, optional, audit = evaluate.native.native_graph(current, actual, score, np.random.default_rng(np.random.SeedSequence(material)))
        np.testing.assert_array_equal(edge, arrays[f"edges__{policy}"])
        np.testing.assert_array_equal(edge[:, :base.shape[1]], base)
        np.testing.assert_array_equal(optional, arrays[f"optional_pairs__{policy}"])
        assert audit == row["cases"][policy]["graph"]
        assert len(optional) == (0 if policy == "base" else len(graph.extra) if policy == "dense" else row["optional_budget"])
    np.testing.assert_array_equal(arrays[f"selection_score__{evaluate.RISKS[0]}"], arrays["previous_base_risk"])
    np.testing.assert_array_equal(arrays[f"selection_score__{evaluate.RISKS[1]}"], arrays["risk__base"])
    assert row["cases"][evaluate.RISKS[1]]["network_passes_for_standalone_policy"] == 2


def test_targets_objective_do_not_change_graphs_outputs_or_rng():
    current, previous, types, target = fixture()
    row, a = evaluate.evaluate_frame(tiny_model(), current, previous, types, target, {**identity(), "objective":"faithful"},2,"cpu")
    changed, b = evaluate.evaluate_frame(tiny_model(), current, previous, types, target+.01,{**identity(), "objective":"nll"},2,"cpu")
    assert row["random_seed_materials"] == changed["random_seed_materials"]
    for draw,name in enumerate(evaluate.RANDOM_CASES):
        assert row["random_seed_materials"][name] == [20261006,881,2,1,3,7,draw]
    for key in a:
        if key.startswith(("edges__", "optional_pairs__", "prediction__", "risk__", "selection_score__")):
            np.testing.assert_array_equal(a[key], b[key])
    assert row["target_sha256"] != changed["target_sha256"]


def test_envelope_counts_ties_nulls_and_exact_eight():
    out=evaluate.envelope([1.,2.,3.,4.,5.,6.,7.,8.], 4.)
    assert (out["random_strictly_better"],out["random_tied"],out["random_strictly_worse"]) == (3,1,4)
    assert out["reference_minus_random_mean"] == -.5 and out["mean"] == 4.5
    assert evaluate.envelope([1.]*8,1.)["random_tied"] == 8
    null=evaluate.envelope([1.]*7+[None],1.)
    assert null["defined_draws"]==7 and all(null[k] is None for k in ("mean","minimum","maximum","random_tied"))
    no_ref=evaluate.envelope([1.]*8)
    assert no_ref["mean"]==1. and no_ref["comparison_null_reason"]
    for values in ([1.]*7, [1.]*9, [float("nan")]*8):
        with pytest.raises(ValueError): evaluate.envelope(values,1.)


def test_raw_metrics_gains_envelopes_and_boundaries_reconstruct():
    row,a=observed()
    for policy in evaluate.POLICIES:
        residual=a[f"prediction__{policy}"].astype(np.float64)-a["target_position"]
        norm=residual/a["acceleration_std"]
        np.testing.assert_array_equal(residual,a[f"position_residual__{policy}"])
        np.testing.assert_array_equal(norm,a[f"normalized_residual__{policy}"])
        gain=np.mean(a["normalized_residual__base"]**2,axis=-1)-np.mean(norm**2,axis=-1)
        np.testing.assert_array_equal(gain,a[f"signed_normalized_coordinate_gain__{policy}"])
        assert row["benefits"][policy]["mean_signed_normalized_coordinate_gain"]==float(gain.mean())
        assert row["cases"][policy]["predicted_boundary"]==evaluate.full.boundary_metrics(a[f"prediction__{policy}"],a["bounds"])
    assert row["random_envelope"]==evaluate.frame_envelopes(row["cases"])


def test_previous_failure_preserves_raw_values_and_current_risk(monkeypatch):
    original=evaluate.bridge.supplied; calls=[0]
    def bad(*args,**kwargs):
        out=original(*args,**kwargs); calls[0]+=1
        if calls[0]==2: out["raw_risk"][0]=np.nan; out["risk"][0]=np.nan
        return out
    monkeypatch.setattr(evaluate.bridge,"supplied",bad)
    row,a=observed()
    assert np.isnan(a["previous_base_risk"][0])
    assert row["cases"][evaluate.RISKS[0]]["status"]=="failed"
    assert row["cases"][evaluate.RISKS[1]]["status"]=="complete"
    assert row["random_envelope"][evaluate.METRICS[0]][evaluate.RISKS[0]]["mean"] is not None
    assert row["random_envelope"][evaluate.METRICS[0]][evaluate.RISKS[0]]["random_tied"] is None


def test_random_failure_keeps_rejected_output_and_null_full_envelope(monkeypatch):
    original=evaluate.bridge.supplied; calls=[0]
    def bad(*args,**kwargs):
        out=original(*args,**kwargs); calls[0]+=1
        if calls[0]==7: out["prediction"][0,0]=11. # parity, previous, dense,speed,previous,current,draw0
        return out
    monkeypatch.setattr(evaluate.bridge,"supplied",bad)
    row,a=observed()
    assert row["cases"]["random25_draw0"]["status"]=="failed"
    assert a["prediction__random25_draw0"][0,0]==11.
    assert row["cases"]["random25_draw0"]["predicted_boundary"]
    assert row["random_envelope"][evaluate.METRICS[0]]["base"]["defined_draws"]==7
    assert row["random_envelope"][evaluate.METRICS[0]]["base"]["mean"] is None
    assert row["cases"]["random25_draw7"]["status"]=="complete"


def test_parity_failure_retains_comparison_and_all13_failed():
    m=tiny_model(); original=m.predict_positions_with_variance
    def disagree(*a,**kw):
        pred,risk=original(*a,**kw);return pred+.01,risk
    m.predict_positions_with_variance=disagree
    row,a=observed(m)
    assert row["failure"]["category"]=="native_parity_failure"
    assert len(row["cases"])==13 and "native_prediction" in a
    assert all(c["status"]=="failed" for c in row["cases"].values())


def test_initial_guard_has13_null_outcomes_and_preserves_history():
    cur,prev,types,target=fixture();cur[0,0,0]=np.nan
    row,a=evaluate.evaluate_frame(tiny_model(),cur,prev,types,target,identity(),0,"cpu")
    assert np.isnan(a["current_history"][0,0,0]) and len(row["cases"])==13
    assert row["random_envelope"][evaluate.METRICS[0]]["base"]["mean"] is None


def test_zero_budget_cap_ties_keep_native_asymmetry_and_all8_ties():
    cur=np.full((6,130,2),.2,dtype=np.float32);types=np.ones(130,dtype=np.int64)
    row,a=evaluate.evaluate_frame(tiny_model(),cur,cur.copy(),types,cur[-1],identity(),0,"cpu")
    assert row["optional_budget"]==0 and row["cases"]["base"]["graph"]["native_base_self_edges"]==128
    assert row["cases"]["base"]["graph"]["native_base_asymmetric_directed_edges"]>0
    for policy in evaluate.POLICIES:
        np.testing.assert_array_equal(a[f"edges__{policy}"],a["edges__base"])
    assert row["random_envelope"][evaluate.METRICS[0]]["base"]["random_tied"]==8


def test_strict_radius_and_score_ties_follow_frozen_helpers():
    points=np.array([[0.,0.],[.125,0.],[0.,.124]],dtype=np.float32)
    graph=evaluate.bridge.strict_pairs(points,.125)
    assert (0,1) not in evaluate.bridge.same.pair_set(graph.base)
    assert (0,1) in evaluate.bridge.same.pair_set(graph.extra)
    cur,*_=fixture();graph=evaluate.bridge.strict_pairs(cur[-1],.015)
    selected=evaluate.bridge.select_pairs(graph,np.ones(graph.n_nodes),int(.25*len(graph.extra)))
    np.testing.assert_array_equal(selected[len(graph.base):],graph.extra[:int(.25*len(graph.extra))])


def test_output_refusal_pins_duplicate_json_and_stream_identities(tmp_path):
    old=tmp_path/"old";old.mkdir();(old/"raw").write_text("retained")
    with pytest.raises(ValueError,match="fresh"):evaluate.preflight_output(old)
    with pytest.raises(ValueError,match="separate"):evaluate.preflight_output(old/"new",[old])
    f=tmp_path/"pin";f.write_text("a");pins={str(f):evaluate.full.sha256(f)}
    evaluate.verify_files(pins);f.write_text("b")
    with pytest.raises(ValueError,match="byte identity"):evaluate.verify_files(pins)
    for text in ('{"a":1,"a":2}', '{"a":NaN}'):
        f.write_text(text)
        with pytest.raises(ValueError):evaluate.strict_json(f)
    for seed,draw in ((True,0),(0,True),(3,0),(0,8)):
        with pytest.raises(ValueError):evaluate.random_material(seed,identity(),draw)
