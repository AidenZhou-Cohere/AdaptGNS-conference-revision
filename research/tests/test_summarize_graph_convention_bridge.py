"""Synthetic tests of pairing, failure nulls and saved-array arithmetic."""
import copy
import json
import sys

import numpy as np
import pytest

from research import summarize_graph_convention_bridge as summary
from research.tests.test_graph_convention_bridge import evaluate


def test_equal_trajectory_weighting_differs_from_unbalanced_frame_weight():
    rows = [{"trajectory_id": "A", "measures": {"x": 2.}},
            {"trajectory_id": "B", "measures": {"x": 10.}},
            {"trajectory_id": "B", "measures": {"x": 10.}}]
    result = summary.summary_measure(rows, "x")
    assert result["equal_trajectory_mean"] == 6.
    assert result["equal_frame_mean_sensitivity"] == pytest.approx(22/3)
    assert result["trajectories"]["A"]["required_frames"] == 1
    assert result["trajectories"]["B"]["required_frames"] == 2


def test_one_undefined_required_observation_nulls_group_without_survivor_mean():
    rows = [{"trajectory_id": "A", "measures": {"x": 1.}},
            {"trajectory_id": "A", "measures": {"x": None}},
            {"trajectory_id": "B", "measures": {"x": 3.}}]
    result = summary.summary_measure(rows, "x")
    assert result["equal_trajectory_mean"] is None
    assert result["equal_frame_mean_sensitivity"] is None
    assert result["trajectories"]["B"]["value"] == 3.
    assert result["undefined_frames"] == 1


def test_three_seed_sample_sd_and_incomplete_seed_null():
    result = summary.seed_statistics([1., 3., 5.])
    assert result["mean"] == 3. and result["sample_sd"] == 2.
    failed = summary.seed_statistics([1., None, 5.])
    assert failed["mean"] is None and failed["sample_sd"] is None
    assert failed["seed_values"] == [1., None, 5.]


def test_frame_arrays_recompute_all_cases_and_fixed_pair_contrasts():
    row, arrays = evaluate()
    audit = summary.Audit()
    result = summary.analyze_frame(row, arrays, audit)
    assert audit.checks > 200
    metric = "normalized_coordinate_mse"
    for name in summary.bridge.CASE_NAMES:
        assert result["measures"][f"case__{name}__{metric}"] == row["cases"][name]["metrics"][metric]
    name = "loop_interaction__previous-observed-base-risk25_minus_random25"
    weights = summary.CONTRASTS[name]
    expected = sum(weight * row["cases"][case]["metrics"][metric] for case, weight in weights.items())
    assert result["measures"][f"contrast__{name}__{metric}"] == expected
    assert result["measures"]["current_previous__loops1__optional_jaccard"] is not None


def test_tampered_saved_prediction_is_rejected_even_if_finite():
    row, arrays = evaluate()
    arrays["prediction__base_uncapped_loops0"] = arrays["prediction__base_uncapped_loops0"].copy()
    arrays["prediction__base_uncapped_loops0"][0, 0] += .01
    with pytest.raises(ValueError, match="residual differs"):
        summary.analyze_frame(row, arrays, summary.Audit())


def test_failed_case_nulls_only_affected_contrasts_and_preserves_other_metrics():
    row, arrays = evaluate()
    name = "previous-observed-base-risk25_uncapped_loops1"
    row["cases"][name].update(status="failed", failure={"category": "nonfinite_risk"}, metrics=None)
    row.update(status="failed", failure={"category": "case_failures"})
    result = summary.analyze_frame(row, arrays, summary.Audit())
    metric = "normalized_coordinate_mse"
    assert result["measures"][f"case__{name}__{metric}"] is None
    assert result["measures"][f"contrast__loop_interaction__previous-observed-base-risk25_minus_random25__{metric}"] is None
    assert result["measures"][f"contrast__loop_interaction__dense_minus_base__{metric}"] is not None


def test_actual_sparse_benefit_correlation_uses_signed_action_label():
    row, arrays = evaluate()
    result = summary.analyze_frame(row, arrays, summary.Audit())
    policy = "current-base-risk25"
    base, action = "base_uncapped_loops0", policy + "_uncapped_loops0"
    error_base = np.sum(arrays[f"normalized_residual__{base}"] ** 2, axis=1)
    error_action = np.sum(arrays[f"normalized_residual__{action}"] ** 2, axis=1)
    expected = summary.bridge.same.spearman(arrays[f"selection_score__{action}"], error_base - error_action)["value"]
    assert result["measures"][f"risk_correlation__{policy}__loops0__own_sparse_benefit"] == expected


def test_graph_summary_distinguishes_cap_and_native_checks():
    row, arrays = evaluate()
    result = summary.analyze_frame(row, arrays, summary.Audit())
    report = summary.graph_diagnostics([result])
    assert report["native_parity_passed_frames"] == 1
    assert report["native_parity_maxima"]["raw_risk_max_abs_difference"] == 0
    assert all(value["cap_active_frames"] == 0 for value in report["cap_counts"].values())
    assert report["exact_graph_reuse_cases"] >= 4
    assert all(value == 0 for value in report["failed_cases"].values())


def test_signed_interaction_convention_is_unambiguous():
    weights = summary.CONTRASTS["loop_interaction__dense_minus_base"]
    assert weights == {"dense_uncapped_loops1": 1, "base_uncapped_loops1": -1,
                       "dense_uncapped_loops0": -1, "base_uncapped_loops0": 1}


def test_saved_parity_flags_and_maxima_are_recomputed_from_raw_arrays():
    row, arrays = evaluate()
    row["native_parity"]["raw_risk_max_abs_difference"] = .001
    with pytest.raises(ValueError, match="maximum differs"):
        summary.analyze_frame(row, arrays, summary.Audit())


def test_factorial_edge_order_tamper_with_updated_hash_is_rejected():
    row, arrays = evaluate()
    name = "dense_cap128_loops0"
    edge = arrays[f"edges__{name}"].copy()
    edge[:, [0, 1]] = edge[:, [1, 0]]
    arrays[f"edges__{name}"] = edge
    row["cases"][name]["graph"]["edge_sha256"] = summary.bridge.full.state_hash(edge)
    with pytest.raises(ValueError, match="factorial edge semantics"):
        summary.analyze_frame(row, arrays, summary.Audit())


def test_recorded_cap_activity_is_recomputed():
    row, arrays = evaluate()
    row["base_graph_audit"]["dense_loops1"]["receivers_above_cap"] = 1
    with pytest.raises(ValueError, match="Cap activation"):
        summary.analyze_frame(row, arrays, summary.Audit())


def minimal_runs():
    frames = [{"split": split, "source_index": 0, "target_frame": i + 7,
               "trajectory_id": split + ":0", "pairing_hashes": {"current_history": "current", "previous_history": "previous",
                    "target_position": "target", "particle_types": "types", "acceleration_std": "std"}}
              for split, count in (("valid", 128), ("test", 297)) for i in range(count)]
    schedule = [{key: value for key, value in row.items() if key != "pairing_hashes"} for row in frames]
    return [{"objective": objective, "seed": seed, "checkpoint_sha256": objective + str(seed),
             "source_protocol": {"expected_frames": copy.deepcopy(schedule)}, "frames": copy.deepcopy(frames),
             "splits": {split: {"measures": {"metric": {"equal_trajectory_mean": float(seed)}}, "failed_frames": 0}
                        for split in ("valid", "test")}} for objective in ("faithful", "nll") for seed in (0, 1, 2)]


@pytest.mark.parametrize("field", ["current_history", "previous_history", "target_position", "particle_types", "acceleration_std"])
def test_cross_model_pairing_compares_raw_input_and_normalization_hashes(field):
    runs = minimal_runs()
    assert summary.summarize(runs, summary.Audit())["faithful"]["valid"]["measures"]["metric"]["mean"] == 1.
    runs[-1]["frames"][0]["pairing_hashes"][field] = "changed"
    with pytest.raises(ValueError, match="Cross-model history/target/normalization"):
        summary.summarize(runs, summary.Audit())


def test_six_distinct_checkpoints_are_required():
    runs = minimal_runs()
    runs[-1]["checkpoint_sha256"] = runs[0]["checkpoint_sha256"]
    with pytest.raises(ValueError, match="distinct checkpoint"):
        summary.summarize(runs, summary.Audit())


def test_existing_analysis_failure_directory_is_never_replaced(tmp_path, monkeypatch):
    directory = tmp_path / "prior"
    directory.mkdir()
    failure = directory / "failed_analysis.json"
    failure.write_text('{"reason":"preserve prior attempt"}')
    original = failure.read_bytes()
    monkeypatch.setattr(sys, "argv", ["summary", "--runs", *(["unused"] * 6), "--output-dir", str(directory)])
    with pytest.raises(SystemExit):
        summary.main()
    assert failure.read_bytes() == original


def test_summary_source_is_pinned_before_execution_and_change_preserves_failure(tmp_path, monkeypatch):
    output = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["summary", "--runs", *(["unused"] * 6), "--output-dir", str(output)])
    monkeypatch.setattr(summary, "load_run", lambda *args: {"source_protocol": {"input_files_sha256": {}}, "input_files_sha256": {}})
    monkeypatch.setattr(summary, "summarize", lambda *args: {})
    calls = [0]
    def source_hash(path):
        calls[0] += 1
        return "original-source" if calls[0] == 1 else "changed-source"
    monkeypatch.setattr(summary.bridge.full, "sha256", source_hash)
    with pytest.raises(ValueError, match="Summary source changed"):
        summary.main()
    pinned = json.loads((output / "analysis_source.json").read_text())
    failure = json.loads((output / "failed_analysis.json").read_text())
    assert pinned["summary_source_sha256"] == "original-source"
    assert failure["summary_source_sha256"] == "original-source"
    assert failure["summary_source_sha256_at_failure"] == "changed-source"


def test_actual_source_frame_audit_rejects_consistent_but_wrong_saved_target():
    row, arrays = evaluate()
    positions = np.concatenate((arrays["previous_history"][:1], arrays["current_history"], arrays["target_position"][None]), axis=0)
    record = {"id": row["trajectory_id"]}
    summary.audit_official_frame(row, arrays, positions, arrays["particle_types"], record, summary.Audit())
    arrays["target_position"] = arrays["target_position"] + .01
    with pytest.raises(ValueError, match="target differs from actual official source"):
        summary.audit_official_frame(row, arrays, positions, arrays["particle_types"], record, summary.Audit())
