"""Synthetic wrapper checks; no released arrays, models or datasets are read."""

from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import spearmanr

from research import analyze_physical_complexity as analysis


def frame_rows(values=(-0.6, 0.0, 0.6)):
    return [{"id": f"trajectory_{trajectory}:{frame}",
             "trajectory": f"trajectory_{trajectory}",
             "correlations": {key: {"value": values[trajectory]} for key in analysis.METRICS}}
            for trajectory in range(3) for frame in range(12)]


def test_complete_frames_use_three_equal_trajectory_means():
    rows = frame_rows((-0.6, 0.0, 0.3))
    result = analysis.aggregate_frames(rows)
    assert len(result["trajectories"]) == 3
    for metric in result["metrics"].values():
        assert metric["unconditional_equal_trajectory_mean"] == pytest.approx(-0.1)
        assert metric["defined_frame_equal_trajectory_mean"] == pytest.approx(-0.1)
        assert metric["required_frames"] == metric["defined_frames"] == 36
        assert metric["trajectories_with_any_defined_frame"] == 3


def test_missing_frames_null_strict_mean_but_conditional_mean_keeps_trajectory_weights():
    rows = frame_rows((0.9, -0.6, 0.3))
    for row in rows[1:12]:
        row["correlations"]["strain"]["value"] = None
    result = analysis.aggregate_frames(rows)
    metric = result["metrics"]["strain"]
    assert metric["unconditional_equal_trajectory_mean"] is None
    assert metric["defined_frame_equal_trajectory_mean"] == pytest.approx(0.2)
    assert metric["defined_frames"] == 25
    assert metric["trajectories_with_any_defined_frame"] == 3
    assert result["trajectories"]["trajectory_0"]["strain"] == {
        "required": 12, "defined": 1, "mean": None, "defined_only_mean": 0.9}
    # Equal trajectory weight differs sharply from pooling the 25 defined frames.
    assert metric["defined_frame_equal_trajectory_mean"] != pytest.approx(
        (0.9 + 12 * -0.6 + 12 * 0.3) / 25)
    assert result["metrics"]["vorticity"]["unconditional_equal_trajectory_mean"] == pytest.approx(0.2)


@pytest.mark.parametrize("missing_trajectories", [1, 3])
def test_conditional_mean_does_not_drop_an_entire_required_trajectory(missing_trajectories):
    rows = frame_rows()
    for row in rows[:12 * missing_trajectories]:
        row["correlations"]["strain"]["value"] = None
    metric = analysis.aggregate_frames(rows)["metrics"]["strain"]
    assert metric["unconditional_equal_trajectory_mean"] is None
    assert metric["defined_frame_equal_trajectory_mean"] is None
    assert metric["defined_frames"] == 36 - 12 * missing_trajectories
    assert metric["trajectories_with_any_defined_frame"] == 3 - missing_trajectories


@pytest.mark.parametrize("defect", ["missing", "duplicate", "two_trajectories", "uneven"])
def test_frame_aggregation_rejects_incomplete_or_misaligned_groups(defect):
    rows = frame_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1]["id"] = rows[0]["id"]
    elif defect == "two_trajectories":
        for row in rows[24:]:
            row["trajectory"] = "trajectory_1"
    else:
        rows[0]["trajectory"] = "trajectory_1"
    with pytest.raises(ValueError):
        analysis.aggregate_frames(rows)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_saved_coefficients_are_rejected_instead_of_skipped(bad):
    rows = frame_rows()
    rows[0]["correlations"]["strain"]["value"] = bad
    with pytest.raises(ValueError, match="Nonfinite"):
        analysis.aggregate_frames(rows)


def seed_runs():
    return {f"{objective}_seed{seed}": {"splits": {
        split: {"aggregate": analysis.aggregate_frames(frame_rows((value,) * 3))}
        for split in analysis.SPLITS}}
        for objective in analysis.OBJECTIVES
        for seed, value in enumerate((-0.4, -0.2, 0.0, 0.2, 0.4))}


def test_seed_summary_preserves_all_five_values_and_uses_sample_sd():
    result = analysis.seed_summary(seed_runs())
    assert set(result) == {f"{objective}/{split}" for objective in analysis.OBJECTIVES
                           for split in analysis.SPLITS}
    metric = result["faithful/valid"]["strain"]
    assert metric["seed_values"] == pytest.approx([-0.4, -0.2, 0.0, 0.2, 0.4])
    assert metric["mean"] == pytest.approx(0.0, abs=1e-15)
    assert metric["sample_seed_sd"] == pytest.approx(np.sqrt(0.1))
    assert metric["defined_frame_sample_seed_sd"] == pytest.approx(np.sqrt(0.1))
    assert metric["conditional_seeds_defined"] == 5
    assert metric["defined_frames_per_seed"] == [36] * 5


def test_partial_frames_can_have_five_seed_conditional_summary_but_no_strict_summary():
    runs = seed_runs()
    rows = frame_rows((0.0, 0.0, 0.0))
    rows[0]["correlations"]["strain"]["value"] = None
    runs["faithful_seed2"]["splits"]["valid"]["aggregate"] = analysis.aggregate_frames(rows)
    metric = analysis.seed_summary(runs)["faithful/valid"]["strain"]
    assert metric["seed_values"][2] is None
    assert metric["mean"] is metric["sample_seed_sd"] is None
    assert metric["defined_frame_mean"] == pytest.approx(0.0, abs=1e-15)
    assert metric["defined_frame_sample_seed_sd"] == pytest.approx(np.sqrt(0.1))
    assert metric["conditional_seeds_defined"] == 5
    assert metric["defined_frames_per_seed"] == [36, 36, 35, 36, 36]


def test_conditional_seed_summary_does_not_drop_a_seed_without_defined_trajectory():
    runs = seed_runs()
    rows = frame_rows()
    for row in rows[:12]:
        row["correlations"]["strain"]["value"] = None
    runs["faithful_seed2"]["splits"]["valid"]["aggregate"] = analysis.aggregate_frames(rows)
    metric = analysis.seed_summary(runs)["faithful/valid"]["strain"]
    assert len(metric["conditional_seed_values"]) == 5
    assert metric["conditional_seed_values"][2] is None
    assert metric["defined_frame_mean"] is metric["defined_frame_sample_seed_sd"] is None
    assert metric["mean"] is metric["sample_seed_sd"] is None
    assert metric["conditional_seeds_defined"] == 4
    assert metric["defined_frames_per_seed"] == [36, 36, 24, 36, 36]


def test_missing_seed_is_rejected_not_substituted_by_four_seed_average():
    runs = seed_runs()
    del runs["nll_seed4"]
    with pytest.raises((KeyError, ValueError)):
        analysis.seed_summary(runs)


def synthetic_flow(mask=None):
    if mask is None:
        mask = np.array([True] * 8 + [False] * 2)
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    values = np.arange(1, n + 1, dtype=float)
    flow = {"fit_valid": mask,
            "neighbor_count": np.resize([3, 4, 3, 5, 4, 6, 7, 4, 1, 0], n),
            "speed": np.resize([2, 1, 4, 2, 6, 3, 8, 5, 9, 7], n),
            "wall_clearance": np.resize([0.1, 0.4, 0.2, 0.8, 0.3, 0.7, 0.9, 0.6, -0.1, 0.0], n),
            "observed_acceleration": values**2}
    for key, target in (("strain_frobenius", values), ("abs_vorticity", values[::-1]),
                        ("abs_divergence", values**2), ("velocity_dispersion", values / 2)):
        flow[key] = np.where(mask, target, np.nan)
    return flow


def test_correlations_use_geometry_mask_and_retain_excluded_risk_coverage():
    flow = synthetic_flow()
    risk = np.array([1, 2, 3, 4, 5, 6, 7, 8, 20, 30], dtype=float)
    correlations, groups = analysis.frame_correlations(flow, risk, risk**2, -risk)
    assert set(correlations) == set(analysis.METRICS)
    for key, record in correlations.items():
        assert record["n"] == (10 if key.endswith("_all") else 8)
    assert correlations["strain"]["value"] == pytest.approx(1.0)
    assert correlations["vorticity"]["value"] == pytest.approx(-1.0)
    assert correlations["benefit_on_fit"]["value"] == pytest.approx(-1.0)
    assert correlations["benefit_all"]["value"] == pytest.approx(-1.0)
    assert correlations["speed_all"]["value"] == pytest.approx(spearmanr(risk, flow["speed"]).statistic)
    assert groups == {"fit_included": {"n": 8, "mean": 4.5, "median": 4.5},
                      "fit_excluded": {"n": 2, "mean": 25.0, "median": 25.0}}


@pytest.mark.parametrize("included", [False, True])
def test_empty_coverage_group_stays_explicit_and_all_particle_metrics_still_run(included):
    flow = synthetic_flow(np.full(10, included))
    risk = np.arange(1, 11, dtype=float)
    correlations, groups = analysis.frame_correlations(flow, risk, risk**2, -risk)
    empty = groups["fit_excluded" if included else "fit_included"]
    assert empty == {"n": 0, "mean": None, "median": None}
    assert correlations["benefit_all"]["n"] == 10
    if not included:
        assert correlations["strain"]["value"] is None
        assert correlations["strain_partial"]["value"] is None
        assert correlations["strain_partial"]["n"] == 0


@pytest.mark.parametrize("which,value", [("risk", 0.0), ("risk", -1.0), ("risk", np.nan),
                                         ("base_error", -1.0), ("benefit", np.inf)])
def test_invalid_cached_scalars_are_rejected(which, value):
    arrays = {"risk": np.arange(1, 11, dtype=float), "base_error": np.ones(10), "benefit": -np.ones(10)}
    arrays[which][0] = value
    with pytest.raises(ValueError):
        analysis.frame_correlations(synthetic_flow(), **arrays)


def test_descriptors_use_recent_displacement_past_acceleration_and_signed_box_clearance(monkeypatch):
    last = np.array([[-0.25, 0.5], [0.2, 0.9], [0.7, 1.3]])
    velocity = np.array([[0.01, 0.0], [0.0, 0.02], [-0.01, 0.03]])
    previous_velocity = np.array([[0.0, 0.0], [0.01, 0.02], [0.03, 0.01]])
    history = np.stack((last - velocity - previous_velocity, last - velocity, last))
    observed_calls = []

    def record_flow(positions, velocities):
        observed_calls.append((positions.copy(), velocities.copy()))
        return {"speed": np.linalg.norm(velocities, axis=1)}

    monkeypatch.setattr(analysis, "local_flow_diagnostics", record_flow)
    original = history.copy()
    flow = analysis.descriptors(history, np.array([[0.0, 1.0], [0.0, 1.0]]))
    assert len(observed_calls) == 1
    np.testing.assert_array_equal(observed_calls[0][0], last)
    np.testing.assert_allclose(observed_calls[0][1], velocity)
    np.testing.assert_allclose(flow["observed_acceleration"], np.linalg.norm(velocity - previous_velocity, axis=1))
    np.testing.assert_allclose(flow["wall_clearance"], [-0.25, 0.1, -0.3])
    np.testing.assert_array_equal(history, original)
    with pytest.raises(ValueError, match="Three observed frames"):
        analysis.descriptors(np.concatenate((history, last[None])), np.array([[0.0, 1.0], [0.0, 1.0]]))


def test_nonfinite_oldest_observation_is_rejected_before_returning_a_descriptor():
    history = np.zeros((3, 4, 2))
    history[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        analysis.descriptors(history, np.array([[0.0, 1.0], [0.0, 1.0]]))


def npz_bytes(values):
    stream = BytesIO()
    np.savez(stream, **values)
    return stream.getvalue()


@pytest.fixture
def cached_fixture(tmp_path):
    data_dir, array_dir = tmp_path / "data", tmp_path / "arrays"
    data_dir.mkdir()
    array_dir.mkdir()
    args = SimpleNamespace(data_dir=data_dir, array_dir=array_dir,
                           risk_summary=tmp_path / "risk_summary.json",
                           input_identity=tmp_path / "input_identity.json",
                           protocol=tmp_path / "protocol.md", output_prefix=tmp_path / "result")
    args.protocol.write_text("Synthetic wrapper fixture; no scientific outcomes.\n")
    (data_dir / "metadata.json").write_text(json.dumps({"bounds": [[0, 1], [0, 1]], "dt": 0.01}))
    grid = np.array([[0.4 + 0.001 * i, 0.4 + 0.001 * j] for i in range(4) for j in range(2)])
    positions = {f"position_{i}": np.stack([grid + t * 1e-5 + i * 1e-4 for t in range(20)])
                 for i in range(3)}
    geometry = {**positions, **{f"type_{i}": np.zeros(8, dtype=int) for i in range(3)}}
    for split in analysis.SPLITS:
        (data_dir / f"{split}-pilot.npz").write_bytes(npz_bytes(geometry))
    split_frames = {split: [{"id": f"position_{trajectory}:{step}", "trajectory": f"position_{trajectory}",
                            "step": step, "n_particles": 8, "raw_prefix": f"{split}_{trajectory * 12 + step - 7:03d}"}
                           for trajectory in range(3) for step in range(7, 19)] for split in analysis.SPLITS}
    risk, base, dense = np.arange(1, 9, dtype=float), np.arange(8, dtype=float) / 16, np.ones(8) / 32
    raw = {f"{frame['raw_prefix']}_{key}": value for frames in split_frames.values() for frame in frames
           for key, value in (("risk", risk), ("base_vector_se", base),
                              ("dense_vector_se", dense), ("signed_benefit", base - dense))}
    payload = npz_bytes(raw)
    runs = {}
    for objective in analysis.OBJECTIVES:
        for seed in analysis.SEEDS:
            name = f"{objective}_seed{seed}"
            path = array_dir / f"{name}.npz"
            path.write_bytes(payload)
            runs[name] = {"objective": objective, "seed": seed, "checkpoint_sha256": "a" * 64,
                          "raw_arrays": {"sha256": analysis.sha(path)},
                          "splits": {split: {"frames": deepcopy(frames)} for split, frames in split_frames.items()}}
    reference = {"runs": runs, "expected_runs": list(runs), "pending": [],
                 "provenance": {"data_sha256": {path.name: analysis.sha(path) for path in data_dir.iterdir()}}}

    def save_reference():
        args.risk_summary.write_text(json.dumps(reference))
        args.input_identity.write_text(json.dumps({"risk_summary_sha256": analysis.sha(args.risk_summary)}))

    save_reference()
    return SimpleNamespace(args=args, reference=reference, positions=positions, raw=raw,
                           frames=split_frames, save_reference=save_reference)


def stub_frame_correlations(monkeypatch):
    # Rank/coverage semantics are tested above; this isolates cache orchestration.
    monkeypatch.setattr(analysis, "frame_correlations", lambda flow, risk, base, benefit:
        ({key: {"value": 0.125, "reason": None, "n": len(risk)} for key in analysis.METRICS}, {}))


def test_synthetic_join_computes_geometry_once_from_exact_past_windows(cached_fixture, monkeypatch):
    fixture = cached_fixture
    stub_frame_correlations(monkeypatch)
    original_descriptors, histories = analysis.descriptors, []

    def spy(history, bounds):
        histories.append(history.copy())
        return original_descriptors(history, bounds)

    monkeypatch.setattr(analysis, "descriptors", spy)
    result = analysis.run(fixture.args)
    assert len(histories) == 72
    expected = [fixture.positions[frame["trajectory"]][frame["step"] - 3:frame["step"]]
                for split in analysis.SPLITS for frame in fixture.frames[split]]
    for actual, past in zip(histories, expected):
        np.testing.assert_array_equal(actual, past)
    assert len(result["runs"]) == 15
    assert all(len(result["geometry"][split]) == 36 for split in analysis.SPLITS)
    assert result["summaries"]["faithful/valid"]["strain"]["mean"] == 0.125
    assert result["geometry_arrays"]["sha256"] == analysis.sha(fixture.args.output_prefix.with_suffix(".npz"))
    assert result["inputs_sha256"]["risk_benefit_summary"] == analysis.sha(fixture.args.risk_summary)


def test_summary_identity_rejects_changed_bytes_before_parsing(cached_fixture):
    cached_fixture.args.risk_summary.write_text("Not JSON: must fail the hash gate first")
    with pytest.raises(ValueError, match="pre-analysis input identity"):
        analysis.run(cached_fixture.args)


def test_cached_geometry_hash_change_is_rejected(cached_fixture):
    path = cached_fixture.args.data_dir / "metadata.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="geometry source hash differs"):
        analysis.run(cached_fixture.args)


def test_fifteen_run_gate_rejects_missing_models(cached_fixture):
    del cached_fixture.reference["runs"]["nll_seed4"]
    cached_fixture.save_reference()
    with pytest.raises(ValueError, match="fifteen original pilot runs"):
        analysis.run(cached_fixture.args)


def test_raw_risk_hash_change_is_rejected(cached_fixture):
    path = cached_fixture.args.array_dir / "beta_nll_seed0.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="risk/error/benefit arrays changed"):
        analysis.run(cached_fixture.args)


def test_each_model_must_share_canonical_frame_order(cached_fixture, monkeypatch):
    stub_frame_correlations(monkeypatch)
    frames = cached_fixture.reference["runs"]["nll_seed0"]["splits"]["valid"]["frames"]
    frames[0], frames[1] = frames[1], frames[0]
    cached_fixture.save_reference()
    with pytest.raises(ValueError, match="exact frame identities and order"):
        analysis.run(cached_fixture.args)


@pytest.mark.parametrize("defect", ["prefix", "particle_count", "target_step"])
def test_canonical_frame_join_rejects_invalid_identity(cached_fixture, defect):
    frame = cached_fixture.reference["runs"]["faithful_seed0"]["splits"]["valid"]["frames"][0]
    if defect == "prefix":
        frame["raw_prefix"] = "valid_999"
    elif defect == "particle_count":
        frame["n_particles"] += 1
    else:
        frame["step"] = 6
    cached_fixture.save_reference()
    with pytest.raises(ValueError, match="Frame (join|identity)"):
        analysis.run(cached_fixture.args)


@pytest.mark.parametrize("defect", ["missing_key", "signed_benefit"])
def test_rehashed_raw_arrays_still_require_complete_keys_and_signed_benefit_identity(cached_fixture, defect):
    raw = {key: value.copy() for key, value in cached_fixture.raw.items()}
    if defect == "missing_key":
        del raw["valid_000_risk"]
    else:
        raw["valid_000_signed_benefit"][0] += 0.25
    path = cached_fixture.args.array_dir / "beta_nll_seed0.npz"
    path.write_bytes(npz_bytes(raw))
    cached_fixture.reference["runs"]["beta_nll_seed0"]["raw_arrays"]["sha256"] = analysis.sha(path)
    cached_fixture.save_reference()
    with pytest.raises(ValueError, match="raw array keys|signed benefit"):
        analysis.run(cached_fixture.args)
