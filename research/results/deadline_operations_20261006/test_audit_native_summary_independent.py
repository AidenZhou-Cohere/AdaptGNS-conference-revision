"""Synthetic arithmetic tests; imports no simulator or numerical producer."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("independent_native", Path(__file__).with_name("audit_native_summary_independent.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def row(steps=995, native=False):
    complete = steps == 995
    boundary = {"fraction_particles_outside_by_more_than_1e-6": .2, "maximum_coordinate_excursion": .3}
    r = {"status": "complete" if complete else "failed", "failure": None if complete else {"category": "coordinate_guard"},
         "horizon": 995, "completed_steps": steps, "mse_per_step": [1.] * steps, "directed_edges_per_step": [8] * steps,
         "mean_rollout_mse": 1. if complete else None, "mean_directed_edges": 8. if complete else None,
         "mse_at_steps": {str(s): 1. if steps >= s else None for s in (1, 10, 50, 200, 500, 995)},
         "mse_at_final_horizon": 1. if complete else None, "total_wall_seconds": 20., "total_network_passes": steps,
         "predicted_boundary_per_step": [boundary] * steps, "ground_truth_boundary_per_step": [boundary] * steps}
    if native:
        r.update(attempts=[{"forecast_step": s+1, "accepted": True, "features_forward_decode_and_transfer_seconds": .01,
                           "graph_build_and_selection_seconds": .002, **{k: 2 for k in a.COUNTS}}
                          for s in range(steps)], warmup=None, forecast_network_passes=steps,
                 graph_build_and_selection_seconds_including_warmup=steps*.002,
                 features_forward_decode_and_transfer_seconds_including_warmup=steps*.01,
                 total_wall_seconds_including_native_parity=21., native_parity_operational_seconds=1., native_parity_network_passes=2)
    return r


def test_complete_metrics_and_boundary_maximum():
    r = row()
    r["predicted_boundary_per_step"] = [dict(b) for b in r["predicted_boundary_per_step"]]
    r["predicted_boundary_per_step"][0]["maximum_coordinate_excursion"] = .9
    m = a.row_metrics(r, False, a.Audit())
    assert m["mean_rollout_mse"] == 1. and m["failure_fraction"] == 0.
    assert m["predicted_boundary_trajectory_maximum_excursion"] == .9
    assert m["predicted_boundary_mean_step_maximum_excursion"] < .31


def test_failure_preserves_earlier_horizon_and_elapsed_cost():
    m = a.row_metrics(row(300), False, a.Audit())
    assert m["mse_at_200"] == 1. and m["mse_at_995"] is None
    assert m["mean_rollout_mse"] is None and m["mean_directed_edges"] is None
    assert all(m[k] is None for k in a.BOUNDARIES)
    assert m["failure_fraction"] == 1. and m["mean_rollout_wall_seconds"] == 20.


@pytest.mark.parametrize("key,value", [("mean_rollout_mse", 2.), ("mean_directed_edges", 9.), ("mse_at_final_horizon", 3.), ("completed_steps", 994)])
def test_inconsistent_scalar_rejected(key, value):
    r = row(); r[key] = value
    with pytest.raises(ValueError):
        a.row_metrics(r, False, a.Audit())


def test_incomplete_population_and_missing_seed_are_not_averaged():
    with pytest.raises(ValueError): a.mean([1.] * 26, 27)
    assert a.mean([1.] * 26 + [None], 27) is None
    s = a.summary_stat([1., None, 3.])
    assert s["mean"] is None and s["sample_sd"] is None and s["defined_seed_count"] == 2
    assert a.summary_stat([1., 2., 3.])["sample_sd"] == 1.


def test_native_timing_cap_and_failed_prefix():
    r = row(native=True); m = a.row_metrics(r, True, a.Audit())
    assert m["cap_active_step_fraction"] == 1. and m["mean_native_parity_passes"] == 2
    r = row(3, native=True); m = a.row_metrics(r, True, a.Audit())
    assert m["mean_native_base_directed_edges"] is None
    assert m["mean_native_parity_seconds"] == 1. and m["mean_rollout_wall_seconds"] == 20.


def test_failed_forward_and_warmup_passes_count_in_timing():
    r = row(3, native=True)
    r["attempts"].append({"forecast_step": 4, "accepted": False, "features_forward_decode_and_transfer_seconds": .03})
    r["warmup"] = {"passes": 1, "features_forward_decode_and_transfer_seconds": .04}
    r.update(forecast_network_passes=4, total_network_passes=5, features_forward_decode_and_transfer_seconds_including_warmup=.1)
    m = a.row_metrics(r, True, a.Audit())
    assert m["mean_total_network_passes"] == 5
    assert m["mean_forward_component_seconds"] == .1
    r["total_network_passes"] = 4
    with pytest.raises(ValueError): a.row_metrics(r, True, a.Audit())


def test_malformed_prefix_and_nonfinite_json_rejected(tmp_path):
    r = row(3, native=True); r["attempts"][1]["accepted"] = False
    with pytest.raises(ValueError): a.row_metrics(r, True, a.Audit())
    p = tmp_path / "bad.json"; p.write_text('{"x": NaN}')
    with pytest.raises(ValueError): a.Audit().read(p)


@pytest.mark.parametrize("defect", ["failed_full_horizon", "nondict_failure", "missing_horizon", "extra_horizon", "numeric_passed"])
def test_reviewed_fail_closed_status_horizons_and_boolean_types(defect):
    r = row(300)
    if defect == "failed_full_horizon":
        r = row(); r.update(status="failed", failure={"category": "coordinate_guard"})
    elif defect == "nondict_failure": r["failure"] = "coordinate_guard"
    elif defect == "missing_horizon": del r["mse_at_steps"]["50"]
    elif defect == "extra_horizon": r["mse_at_steps"]["2"] = 1.
    else:
        with pytest.raises(ValueError): a.Audit().equal(1, True)
        return
    with pytest.raises(ValueError): a.row_metrics(r, False, a.Audit())


def test_paired_differences_retain_null_unit_before_aggregation():
    left, right = {}, {}
    group = {"per_seed": [], "metrics": {}}
    seed_values = []
    for seed in range(3):
        left["faithful", seed] = {"rows": {}}
        right["faithful", seed] = {"rows": {}}
        units = []
        for index in a.INDICES:
            missing = seed == 1 and index == 6
            x = None if missing else float(seed+index)
            y = float(index)
            left["faithful", seed]["rows"][index, "base"] = {"trajectory_id": str(index), "metrics": {"m": x}}
            right["faithful", seed]["rows"][index, "base"] = {"trajectory_id": str(index), "metrics": {"m": y}}
            units.append({"source_index": index, "trajectory_id": str(index), "deltas": {"m": None if missing else float(seed)}})
        value = None if seed == 1 else float(seed)
        seed_values.append(value)
        group["per_seed"].append({"seed": seed, "trajectory_pairs": units, "metrics": {"m": value}})
    group["metrics"] = {"m": a.summary_stat(seed_values)}
    a.compare_group(group, left, right, "faithful", "base", "base", ("m",), a.Audit())
    wrong = copy.deepcopy(group); wrong["per_seed"][1]["metrics"]["m"] = 1.
    with pytest.raises(ValueError): a.compare_group(wrong, left, right, "faithful", "base", "base", ("m",), a.Audit())
