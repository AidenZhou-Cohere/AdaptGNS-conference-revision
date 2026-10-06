"""Independent synthetic hierarchy and strict saved-array audit checks."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research import native_random_envelope as evaluate
from research import summarize_native_random_envelope as summary
from research.tests.test_graph_convention_bridge import fixture, identity, tiny_model
from research.tests.test_native_random_envelope import observed


def unit(trajectory, frame, reference, draws):
    values = {metric: {**{name: float(reference) for name in evaluate.CONTROLS},
                       **{name: None if value is None else float(value)
                          for name, value in zip(evaluate.RANDOM_CASES, draws)}}
              for metric in evaluate.METRICS}
    return {"split": "valid", "source_index": {"a": 0, "b": 1}[trajectory],
            "target_frame": frame, "trajectory_id": trajectory, "values": values}


def hierarchy_rows():
    return [unit("a", 7, 1, range(8)), unit("a", 8, 50, [100-d for d in range(8)]),
            unit("b", 7, 20, [d*d for d in range(8)])]


def schedule(rows):
    return [{key: row[key] for key in summary.IDENTITY} for row in rows]


def test_independent_hierarchy_draw_means_cancel_without_cancelling_frame_dispersion():
    rows = hierarchy_rows()
    result = summary.summarize_rows(rows, schedule(rows))
    for metric in evaluate.METRICS:
        # Trajectory a's opposite draw effects cancel exactly across its two
        # frames. The single-frame b still gets half the seed-level weight.
        traj = result["trajectories"]["a"]["envelope_of_trajectory_mean_errors"][metric]["base"]
        assert traj["draw_values"] == [50.]*8
        assert traj["minimum"] == traj["maximum"] == traj["mean"] == 50.
        assert traj["sample_sd"] == 0.
        assert traj["monte_carlo_standard_error_of_mean"] == 0.
        assert result["values"][metric]["base"] == 22.75
        env = result["envelope_of_equal_trajectory_seed_mean_errors"][metric]["base"]
        assert env["draw_values"] == [25., 25.5, 27., 29.5, 33., 37.5, 43., 49.5]
        assert env["mean"] == 33.75
        assert env["reference_minus_random_mean"] == -11.
        assert np.isclose(env["sample_sd"], np.sqrt(79.5))
        assert np.isclose(env["monte_carlo_standard_error_of_mean"], np.sqrt(9.9375))
        assert env["random_strictly_better"] == 0 and env["random_strictly_worse"] == 8
        frame = result["equal_trajectory_mean_of_frame_envelope_measures"][metric]["base"]
        assert frame["reference_minus_random_mean"] == -11.
        assert frame["random_strictly_better"] == 2.75
        assert frame["random_tied"] == .25
        assert frame["random_strictly_worse"] == 5.
        assert frame["sample_sd"] > 0.
        assert result["required_frames"] == 3 and result["required_trajectories"] == 2


def test_independent_complete_hierarchy_invariant_to_input_schedule_permutation():
    rows = hierarchy_rows()
    original = summary.summarize_rows(rows, schedule(rows))
    reverse = summary.summarize_rows(rows[::-1], schedule(rows)[::-1])
    for key in ("values", "envelope_of_equal_trajectory_seed_mean_errors",
                "equal_trajectory_mean_of_frame_envelope_measures", "trajectories"):
        assert original[key] == reverse[key]


@pytest.mark.parametrize("missing", ["draw", "reference", "frame"])
def test_independent_failure_nulls_exact_required_population_not_successful_controls(missing):
    rows = hierarchy_rows()
    expected = schedule(rows)
    if missing == "draw":
        for metric in evaluate.METRICS:
            rows[1]["values"][metric]["random25_draw7"] = None
    elif missing == "reference":
        for metric in evaluate.METRICS:
            rows[1]["values"][metric][evaluate.RISKS[0]] = None
    else:
        rows.pop(1)
    result = summary.summarize_rows(rows, expected)
    for metric in evaluate.METRICS:
        env = result["envelope_of_equal_trajectory_seed_mean_errors"][metric][evaluate.RISKS[0]]
        if missing == "reference":
            assert env["mean"] == 33.75 and env["defined_draws"] == 8
            assert env["reference_minus_random_mean"] is None
            assert result["values"][metric]["base"] == 22.75
            assert result["values"][metric][evaluate.RISKS[0]] is None
        elif missing == "draw":
            assert env["mean"] is None and env["defined_draws"] == 7
            assert env["monte_carlo_standard_error_of_mean"] is None
            assert result["values"][metric]["random25_draw0"] == 25.
            assert result["values"][metric]["base"] == 22.75
            assert env["comparison_null_reason"]
        else:
            assert env["mean"] is None and env["defined_draws"] == 0
            assert all(value is None for value in result["values"][metric].values())
            assert result["missing_frames"] == 1
        assert result["equal_trajectory_mean_of_frame_envelope_measures"][metric][evaluate.RISKS[0]]["reference_minus_random_mean"] is None


def paired_cohort():
    runs = []
    for objective, seed in sorted(evaluate.COHORT):
        rows = hierarchy_rows()
        # Only risk references differ across seeds; random streams remain a
        # within-model Monte Carlo quantity, not extra training replicates.
        for row in rows:
            for metric in evaluate.METRICS:
                for risk in evaluate.RISKS:
                    row["values"][metric][risk] += 2.*seed
        splits = {split: summary.summarize_rows(rows, schedule(rows)) for split in ("valid", "test")}
        runs.append({"objective": objective, "seed": seed, "splits": splits})
    return runs


def test_independent_three_seed_contrasts_keep_signs_and_do_not_pool_draws():
    result = summary.summarize_cohort(paired_cohort()[::-1])
    for split in ("valid", "test"):
        for objective in ("faithful", "nll"):
            for metric in evaluate.METRICS:
                for risk in evaluate.RISKS:
                    stats = result[split][objective]["control_minus_random_mean_statistics"][metric][risk]
                    assert stats["required_seeds"] == stats["defined_seeds"] == 3
                    assert stats["seed_values"] == [-11., -9., -7.]
                    assert stats["mean"] == -9. and stats["sample_sd"] == 2.


def test_independent_one_seed_undefined_cannot_produce_two_seed_primary_mean():
    runs = paired_cohort()
    target = next(row for row in runs if (row["objective"], row["seed"]) == ("faithful", 1))
    for metric in evaluate.METRICS:
        target["splits"]["test"]["equal_trajectory_mean_of_frame_envelope_measures"][metric][evaluate.RISKS[0]]["reference_minus_random_mean"] = None
    result = summary.summarize_cohort(runs)
    for metric in evaluate.METRICS:
        stats = result["test"]["faithful"]["control_minus_random_mean_statistics"][metric][evaluate.RISKS[0]]
        assert stats["seed_values"] == [-11., None, -7.]
        assert stats["defined_seeds"] == 2 and stats["mean"] is stats["sample_sd"] is None
        assert stats["null_reason"]
        assert result["valid"]["faithful"]["control_minus_random_mean_statistics"][metric][evaluate.RISKS[0]]["mean"] == -9.
        assert result["test"]["nll"]["control_minus_random_mean_statistics"][metric][evaluate.RISKS[0]]["mean"] == -9.


@pytest.mark.parametrize("mutation", ["status", "case_cause", "negative_time", "boolean_time"])
def test_independent_guard_record_rejects_invalid_status_cause_or_timing(mutation):
    current, previous, types, target = fixture()
    current = current.copy()
    current[0, 0, 0] = np.nan
    row, arrays = evaluate.evaluate_frame(tiny_model(), current, previous, types, target, identity(), 0, "cpu")
    row["seed"] = 0
    if mutation == "status":
        row["status"] = "not-a-published-state"
    elif mutation == "case_cause":
        row["cases"]["base"]["failure"] = {"category": "fabricated"}
    elif mutation == "negative_time":
        row["operational_seconds"] = -1.
    else:
        row["operational_seconds"] = True
    with pytest.raises(ValueError):
        summary.audit_frame(row, arrays, summary.Audit())


def test_independent_numerical_failure_cannot_be_hidden_under_unknown_case_status(monkeypatch):
    original = evaluate.bridge.supplied
    calls = [0]
    def failed_dense(*args, **kwargs):
        output = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 3:
            output["prediction"][0, 0] = 11.
        return output
    monkeypatch.setattr(evaluate.bridge, "supplied", failed_dense)
    row, arrays = observed()
    row["seed"] = 0
    assert row["cases"]["dense"]["status"] == "failed"
    row["cases"]["dense"]["status"] = "not-a-published-state"
    row["status"], row["failure"] = "complete", None
    with pytest.raises(ValueError):
        summary.audit_frame(row, arrays, summary.Audit())


@pytest.mark.parametrize("value", [-1., True, float("inf")])
def test_independent_previous_score_measurement_requires_nonnegative_finite_number(value):
    row, arrays = observed()
    row["seed"] = 0
    row["previous_score"]["operational_seconds"] = value
    with pytest.raises(ValueError):
        summary.audit_frame(row, arrays, summary.Audit())


def test_independent_summary_preserves_raw_physical_and_signed_gain_records():
    row, arrays = observed()
    row["seed"] = 0
    result = summary.audit_frame(row, arrays, summary.Audit())
    assert result["benefits"] == row["benefits"]
    assert result["previous_risk_correlations"] == row["previous_risk_correlations"]
    assert result["ground_truth_boundary"] == row["ground_truth_boundary"]
    assert result["current_observed_boundary"] == row["current_observed_boundary"]
    assert set(result["case_boundaries"]) == set(evaluate.POLICIES)
    assert set(result["case_timings"]) == set(evaluate.POLICIES)


def pin_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "synthetic_repo"
    repo.mkdir()
    sources = [repo / "evaluator.py", repo / "method.md"]
    monkeypatch.setattr(evaluate, "source_paths", lambda: sources)
    monkeypatch.setattr(summary.full, "REPO", repo)
    protocol = {"checkpoint": str(tmp_path / "model" / "checkpoint-100000.pt"),
                "validation_manifest": str(tmp_path / "valid" / "manifest.json"),
                "test_manifest": str(tmp_path / "test" / "manifest.json"),
                "saved_same_state_dir": str(tmp_path / "saved")}
    run = {"source_sha256": {"training.py": "synthetic frozen digest"}}
    valid = {"records": [{"positions": {"path": "positions.npy"}, "particle_types": {"path": "types.npy"}}]}
    test = {"records": [{"positions": {"path": "test_positions.npy"}, "particle_types": {"path": "test_types.npy"}}]}
    saved = {(3, 7): {"record_file": "frame.json", "array_file": "frame.npz"}}
    # List the expected dependency categories explicitly, independently of the
    # function under review and without writing in the real source tree.
    expected = {str(path) for path in sources} | {str(repo / "training.py")}
    relative = ["model/checkpoint-100000.pt", "model/protocol.json", "model/latest.json", "model/status.json",
                "valid/manifest.json", "valid/metadata.json", "valid/positions.npy", "valid/types.npy",
                "test/manifest.json", "test/test_positions.npy", "test/test_types.npy",
                "saved/protocol.json", "saved/result.json", "saved/frame.json", "saved/frame.npz"]
    expected |= {str(tmp_path / path) for path in relative}
    for name in expected:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic bytes: " + path.name)
    protocol["input_files_sha256"] = {name: evaluate.full.sha256(name) for name in expected}
    return protocol, run, valid, test, saved, expected


def test_independent_required_input_pin_population_is_exact(tmp_path, monkeypatch):
    protocol, run, valid, test, saved, expected = pin_fixture(tmp_path, monkeypatch)
    assert summary.required_input_paths(protocol, run, valid, test, saved) == expected
    summary.verify_required_input_pins(protocol, run, valid, test, saved, summary.Audit())


@pytest.mark.parametrize("relative", [
    "synthetic_repo/evaluator.py", "synthetic_repo/training.py", "model/checkpoint-100000.pt",
    "model/protocol.json", "model/latest.json", "model/status.json", "valid/metadata.json",
    "valid/manifest.json", "test/manifest.json", "valid/positions.npy", "valid/types.npy",
    "test/test_positions.npy", "test/test_types.npy", "saved/protocol.json", "saved/result.json",
    "saved/frame.json", "saved/frame.npz",
])
def test_independent_omitted_input_pin_is_refused_even_if_other_hashes_match(tmp_path, monkeypatch, relative):
    protocol, run, valid, test, saved, expected = pin_fixture(tmp_path, monkeypatch)
    del protocol["input_files_sha256"][str(tmp_path / relative)]
    with pytest.raises(ValueError, match="pin key coverage"):
        summary.verify_required_input_pins(protocol, run, valid, test, saved, summary.Audit())


def test_independent_extra_alias_or_mutated_required_pin_is_refused(tmp_path, monkeypatch):
    protocol, run, valid, test, saved, expected = pin_fixture(tmp_path, monkeypatch)
    original = copy.deepcopy(protocol)
    protocol["input_files_sha256"][str(tmp_path / "unrelated")] = "unrelated hash"
    with pytest.raises(ValueError, match="pin key coverage"):
        summary.verify_required_input_pins(protocol, run, valid, test, saved, summary.Audit())
    path = Path(original["checkpoint"])
    path.write_text("changed after earlier model audit")
    with pytest.raises(ValueError, match="byte hash"):
        summary.verify_required_input_pins(original, run, valid, test, saved, summary.Audit())


@pytest.mark.parametrize("field,value", [("schema", True), ("random_draw_ids", [False, 1, 2, 3, 4, 5, 6, 7]),
                                         ("random_draw_ids", [0, True, 2, 3, 4, 5, 6, 7])])
def test_independent_run_header_rejects_boolean_schema_or_draw_identity_first(tmp_path, field, value):
    protocol = {"scope": evaluate.SCOPE, "schema": 1, "cases": list(evaluate.POLICIES),
                "random_draw_ids": list(range(8)), "objective": "faithful", "seed": 0,
                "checkpoint_sha256": "synthetic", "input_files_sha256": {}}
    protocol[field] = value
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    result = {"state": "complete", "scope": evaluate.SCOPE, "operational_seconds": 0.,
              "protocol_sha256": evaluate.full.sha256(tmp_path / "protocol.json"),
              "objective": "faithful", "seed": 0, "checkpoint_sha256": "synthetic"}
    (tmp_path / "result.json").write_text(json.dumps(result))
    status = {"state": "complete", "result_sha256": evaluate.full.sha256(tmp_path / "result.json"),
              "objective": "faithful", "seed": 0, "operational_seconds": 0.}
    (tmp_path / "status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="schema|draw"):
        summary.load_run(tmp_path, summary.Audit(), {})


@pytest.mark.parametrize("case", ["base", "random25_draw3", *evaluate.RISKS])
def test_independent_shared_scoring_time_must_match_its_source(case):
    row, arrays = observed()
    row["seed"] = 0
    row["cases"][case]["scoring_seconds"] += .1
    with pytest.raises(ValueError):
        summary.audit_frame(row, arrays, summary.Audit())
