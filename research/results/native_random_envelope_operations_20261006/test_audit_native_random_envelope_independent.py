import copy
import importlib.util
from pathlib import Path
import math

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("audit_random", Path(__file__).with_name("audit_native_random_envelope_independent.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def test_hand_computed_eight_draw_moments_ranks_and_se():
    e = a.envelope(list(range(8)), 5)
    assert e["mean"] == 3.5
    assert e["sample_sd"] == math.sqrt(6)
    assert e["monte_carlo_standard_error_of_mean"] == pytest.approx(math.sqrt(.75), rel=0, abs=2e-16)
    assert e["reference_minus_random_mean"] == 1.5
    assert [e[k] for k in ("random_strictly_better", "random_tied", "random_strictly_worse")] == [5, 1, 2]


def test_missing_draw_does_not_become_survivor_average():
    e = a.envelope([None] + list(range(7)), 3)
    assert e["defined_draws"] == 7
    assert all(e[k] is None for k in ("mean", "sample_sd", "monte_carlo_standard_error_of_mean", "minimum", "maximum", "random_tied"))
    assert e["random_mean_null_reason"]


def test_failed_reference_preserves_full_random_distribution():
    e = a.envelope(list(range(8)), None)
    assert e["mean"] == 3.5 and e["comparison_null_reason"] and e["random_mean_null_reason"] is None
    assert e["reference_minus_random_mean"] is None


def frame(tid, number, risk):
    values = {m: {c: 0. for c in a.CASES} for m in a.METRICS}
    for m in a.METRICS:
        values[m]["previous-observed-base-risk25"] = risk
    return dict(split="valid", source_index=number, target_frame=10, trajectory_id=tid, values=values)


def test_unequal_trajectory_frame_counts_and_rank_noncommutation():
    frames = [frame("A", 0, 0.), frame("A", 1, 2.), frame("B", 2, 9.)]
    expected = [{k: row[k] for k in a.IDENTITY} for row in frames]
    result = a.aggregate(frames, expected)
    key = "previous-observed-base-risk25"
    assert result["values"][a.METRICS[0]][key] == 5.
    e = result["equal_trajectory_mean_of_frame_envelope_measures"][a.METRICS[0]][key]
    assert e["reference_minus_random_mean"] == 5.
    assert e["random_strictly_better"] == 6.
    assert e["random_tied"] == 2.
    assert result["envelope_of_equal_trajectory_seed_mean_errors"][a.METRICS[0]][key]["random_strictly_better"] == 8


def test_one_failed_unit_nulls_required_seed_contrast():
    frames = [frame("A", 0, None), frame("A", 1, 2.), frame("B", 2, 9.)]
    result = a.aggregate(frames, [{k: row[k] for k in a.IDENTITY} for row in frames])
    assert result["values"][a.METRICS[0]]["previous-observed-base-risk25"] is None
    assert result["values"][a.METRICS[0]]["random25_draw0"] == 0.
    assert result["trajectories"]["A"]["defined_frames"][a.METRICS[0]]["previous-observed-base-risk25"] == 1


def test_three_seed_sample_sd_and_missing_seed():
    assert a.stats([-1., 0., 1.])["sample_sd"] == 1.
    s = a.stats([-1., None, 1.])
    assert s["mean"] is None and s["sample_sd"] is None and s["defined_seeds"] == 2


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
def test_bad_scalar_rejected(value):
    with pytest.raises(ValueError):
        a.envelope([value] + [0.] * 7, 0.)
    with pytest.raises(ValueError):
        a.Audit().equal(value, 1.)


@pytest.mark.parametrize("length", [0, 7, 9])
def test_draw_population_fixed(length):
    with pytest.raises(ValueError):
        a.envelope([0.] * length, 0.)


def test_duplicate_or_missing_frame_refused():
    row = frame("A", 0, 1.)
    expected = [{k: row[k] for k in a.IDENTITY}]
    with pytest.raises(ValueError):
        a.aggregate([row, row], expected)
    with pytest.raises(ValueError):
        a.aggregate([], expected)


def test_independent_raw_coordinate_error_gain_and_recorded_ties():
    arrays = dict(target_position=np.zeros((2, 2)), acceleration_std=np.array([1., 2.]))
    row = {"cases": {}, "benefits": {}}
    values = {m: {} for m in a.METRICS}
    for i, case in enumerate(a.CASES):
        v = float(i + 1)
        arrays["prediction__" + case] = np.array([[v, 2*v], [v, 2*v]])
        arrays["normalized_coordinate_se__" + case] = np.array([v*v, v*v])
        arrays["signed_normalized_coordinate_gain__" + case] = np.array([1-v*v, 1-v*v])
        metrics = dict(position_coordinate_mse=2.5*v*v, normalized_coordinate_mse=v*v)
        row["cases"][case] = dict(status="complete", failure=None, metrics=metrics)
        row["benefits"][case] = dict(mean_signed_normalized_coordinate_gain=1-v*v, harmful_particle_fraction=float(v>1))
        for m in a.METRICS:
            values[m][case] = metrics[m]
    row["random_envelope"] = a.envelopes(values)
    assert a.raw_values(row, arrays, a.Audit()) == values
    row["cases"]["dense"]["metrics"]["normalized_coordinate_mse"] += .1
    with pytest.raises(ValueError):
        a.raw_values(row, arrays, a.Audit())


def test_early_failure_needs_no_missing_target_and_preserves_all_nulls():
    row = {"cases": {c: dict(status="failed", failure={"category": "guard"}, metrics=None) for c in a.CASES}}
    expected = {m: {c: None for c in a.CASES} for m in a.METRICS}
    row["random_envelope"] = a.envelopes(expected)
    assert a.raw_values(row, {}, a.Audit()) == expected


def test_input_changes_and_duplicate_json_refused(tmp_path):
    p = tmp_path / "input.json"
    p.write_text('{"a":1}')
    check = a.Audit()
    check.pin(p)
    p.write_text('{"a":2}')
    with pytest.raises(ValueError):
        check.pin(p)
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        a.read(p)
