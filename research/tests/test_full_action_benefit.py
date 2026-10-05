"""Synthetic postprocess checks only; no real evaluation arrays or models."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from research import analyze_full_action_benefit as analysis


def fixture():
    n = 5
    target = np.zeros((n, 2), dtype=np.float64)
    base_prediction = np.column_stack((np.arange(1, n + 1) / 10, np.arange(n, 0, -1) / 20))
    predictions = {"base": base_prediction,
        "dense": base_prediction * np.array([1.4, .4]),
        "random25": base_prediction * .7,
        "speed25": base_prediction * np.array([.5, 1.5]),
        "previous-observed-base-risk25": base_prediction * np.array([.9, 1.2])}
    current = np.arange(6 * n * 2, dtype=np.float32).reshape(6, n, 2) / 100
    previous = current - np.float32(.01)
    std = np.array([.1, .3])
    arrays = {"current_observed_history": current, "previous_observed_history": previous,
              "target_position": target, "particle_types": np.zeros(n, dtype=np.int64),
              "current_base_q": np.arange(1, n + 1, dtype=float),
              "previous_observed_base_q": np.arange(n, 0, -1, dtype=float)}
    row = {"source_index": 3, "target_frame": 7, "trajectory_id": "synthetic:3", "n_particles": n,
           "status": "complete", "failure": None, "target_sha256": analysis.array_hash(target),
           "observed_history_sha256": analysis.array_hash(current), "previous_observed_history_sha256": analysis.array_hash(previous),
           "saved_acceleration_normalization": {"mean": [0, 0], "std": std.tolist()}, "accuracy": {}, "timed_calls": []}
    for policy, prediction in predictions.items():
        arrays["prediction_" + policy] = prediction
        row["accuracy"][policy] = {}
        for unit, scale in (("position", np.ones(2)), ("normalized", std)):
            se = np.sum(((prediction - target) / scale) ** 2, axis=1)
            arrays[f"{unit}_vector_se_{policy}"] = se
            row["accuracy"][policy][unit + "_coordinate_mse"] = float(se.mean() / 2)
    for unit in analysis.UNITS:
        arrays["signed_dense_benefit_" + unit] = arrays[f"{unit}_vector_se_base"] - arrays[f"{unit}_vector_se_dense"]
    base = {(0, 1), (3, 4)}
    annulus = set(itertools.combinations(range(n), 2)) - base
    selected = {"base": base, "dense": base | annulus, "random25": base | {(0, 2), (2, 4)},
                "speed25": base | {(0, 3), (0, 4)}, "previous-observed-base-risk25": base | {(1, 2), (1, 3)}}
    arrays["candidate_base_pairs_float64"] = np.array(sorted(base), dtype=np.int64)
    arrays["candidate_annulus_pairs_float64"] = np.array(sorted(annulus), dtype=np.int64)
    audit = {"base_pairs": len(base), "annulus_pairs": len(annulus), "candidate_pairs": len(base | annulus),
             "optional_budget": len(annulus) // 4, "policies": {}, "overlaps": {}}
    for policy in analysis.POLICIES:
        pairs = np.array(sorted(selected[policy]), dtype=np.int64).reshape(-1, 2)
        arrays["pairs_" + policy] = pairs
        row["timed_calls"].append({"method": policy, "pair_sha256": analysis.array_hash(pairs)})
        audit["policies"][policy] = {"retained_pairs": len(pairs), "retained_optional_pairs": len(selected[policy] - base),
                                    "directed_edges": 2 * len(pairs)}
    for left, right in itertools.combinations(analysis.POLICIES, 2):
        a, b = selected[left] - base, selected[right] - base
        intersection, union = len(a & b), len(a | b)
        audit["overlaps"][left + "__" + right] = {"optional_intersection": intersection, "optional_union": union,
                                                  "optional_jaccard": intersection / union if union else 1.}
    row["graph_audit"] = audit
    return row, arrays


def scalar_frame(source, target, value, status="complete"):
    result = analysis.blank_frame({"source_index": source, "target_frame": target, "n_particles": 10, "status": status})
    if status == "complete":
        result["metrics"] = dict.fromkeys(analysis.METRICS, value)
        result["undefined_reasons"] = {}
    return result


def test_actual_benefits_and_independent_decomposition():
    row, arrays = fixture()
    record, derived = analysis.analyze_frame(row, arrays)
    assert record["status"] == "complete"
    assert set(record["metrics"]) == set(analysis.METRICS)
    for unit in analysis.UNITS:
        for action in analysis.ACTIONS:
            expected = arrays[f"{unit}_vector_se_base"] - arrays[f"{unit}_vector_se_{action}"]
            np.testing.assert_allclose(derived[f"benefit_{unit}_{action}"], expected)
            np.testing.assert_allclose(derived[f"alignment_{unit}_{action}"] - derived[f"perturbation_cost_{unit}_{action}"], expected)
            assert record["metrics"][f"benefit/{unit}/{action}/mean_coordinate_benefit"] == pytest.approx(expected.mean() / 2)
            fractions = [record["metrics"][f"benefit/{unit}/{action}/{kind}_fraction"] for kind in ("helpful", "harmful", "zero")]
            assert sum(fractions) == pytest.approx(1.)


def test_position_normalization_may_reverse_benefit_sign():
    row, arrays = fixture()
    _, derived = analysis.analyze_frame(row, arrays)
    assert np.any(np.sign(derived["benefit_position_dense"]) != np.sign(derived["benefit_normalized_dense"]))


def test_negative_results_and_no_input_mutation():
    row, arrays = fixture()
    original_row = copy.deepcopy(row)
    originals = {key: value.copy() for key, value in arrays.items()}
    record, derived = analysis.analyze_frame(row, arrays)
    assert np.any(derived["benefit_normalized_dense"] < 0)
    assert record["metrics"]["benefit/normalized/dense/mean_vector_benefit"] < 0
    assert row == original_row
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, originals[key])


def test_dense_proxy_signs_counts_rank_agreement_and_paired_contrasts():
    row, arrays = fixture()
    record, derived = analysis.analyze_frame(row, arrays)
    assert record["metrics"]["risk_agreement/current_vs_previous_spearman"] == pytest.approx(-1.)
    for unit in analysis.UNITS:
        dense = derived[f"benefit_{unit}_dense"]
        for action in analysis.ACTIONS:
            benefit = derived[f"benefit_{unit}_{action}"]
            counts = np.array(record["dense_proxy_sign_counts"][unit][action]["counts"])
            assert counts.sum() == row["n_particles"]
            assert record["metrics"][f"dense_proxy/{unit}/{action}/sign_disagreement_fraction"] == np.mean(np.sign(dense) != np.sign(benefit))
            if action == "dense":
                assert np.trace(counts) == row["n_particles"]
                assert record["metrics"][f"dense_proxy/{unit}/{action}/opposite_nonzero_fraction"] == 0
                for risk in analysis.RISKS:
                    assert record["metrics"][f"correlation_difference/{unit}/{action}/{risk}/actual_minus_dense"] == 0
            for kind, condition in (("helpful", benefit.mean() > 0), ("harmful", benefit.mean() < 0), ("zero", benefit.mean() == 0)):
                assert record["metrics"][f"benefit/{unit}/{action}/frame_{kind}_indicator"] == float(condition)
        for left, right in itertools.combinations(analysis.POLICIES, 2):
            expected = np.mean(derived[f"error_{unit}_{left}"] - derived[f"error_{unit}_{right}"]) / 2
            assert record["metrics"][f"paired_error/{unit}/{left}_minus_{right}"] == expected


def test_oracle_is_whole_frame_not_particle_mixture():
    errors = {policy: np.array([4., 4.]) for policy in analysis.POLICIES}
    errors["random25"] = np.array([0., 10.])
    errors["speed25"] = np.array([10., 0.])
    oracle = analysis.whole_frame_oracle(errors, analysis.PORTFOLIOS["all_five"])
    assert oracle["coordinate_mse"] == 2.
    assert oracle["choice/base"] == 1.
    # A particlewise mixer would report zero; that is not this oracle.
    assert oracle["coordinate_mse"] > np.min(np.stack(list(errors.values())), axis=0).mean() / 2


def test_oracle_dense_and_budget_portfolios_are_distinct():
    errors = {policy: np.array([4., 4.]) for policy in analysis.POLICIES}
    errors["dense"] = np.array([0., 0.])
    full = analysis.whole_frame_oracle(errors, analysis.PORTFOLIOS["all_five"])
    budget = analysis.whole_frame_oracle(errors, analysis.PORTFOLIOS["budgeted_with_base"])
    assert full["coordinate_mse"] == 0 and budget["coordinate_mse"] == 2
    assert full["choice/dense"] == 1 and "choice/dense" not in budget


def test_base_abstention_is_positive_only_when_base_beats_every_action():
    errors = {policy: np.array([4., 4.]) for policy in analysis.POLICIES}
    errors["base"] = np.array([1., 1.])
    oracle = analysis.whole_frame_oracle(errors, analysis.PORTFOLIOS["all_five"])
    assert oracle["benefit_from_allowing_base"] == 1.5
    assert oracle["base_strictly_best"] == 1 and oracle["base_co_best"] == 1
    assert oracle["best_tie_count"] == 1


def test_oracle_exact_tie_favors_base_but_retains_tie_count():
    errors = {policy: np.ones(3) for policy in analysis.POLICIES}
    oracle = analysis.whole_frame_oracle(errors, analysis.PORTFOLIOS["all_five"])
    assert oracle["choice/base"] == 1 and oracle["best_tie_count"] == 5
    assert oracle["base_strictly_best"] == 0 and oracle["base_co_best"] == 1
    assert oracle["benefit_from_allowing_base"] == 0


def test_average_rank_quartiles_do_not_split_ties():
    q = np.array([1., 1., 1., 1., 2., 3., 4., 5.])
    bins = analysis.risk_quartiles(q)
    assert len(set(bins[:4])) == 1
    assert set(bins) <= {0, 1, 2, 3}
    # All constant scores receive percentile .5 and belong to q3.
    assert np.array_equal(analysis.risk_quartiles(np.ones(9)), np.full(9, 2))


def test_empty_quartiles_and_constant_correlation_explicit():
    row, arrays = fixture()
    arrays["current_base_q"][:] = 1.
    record, _ = analysis.analyze_frame(row, arrays)
    assert record["risk_quartile_counts"]["current_base_q"] == [0, 0, 5, 0]
    name = "risk_profile/normalized/random25/current_base_q/q1/mean_vector_benefit"
    assert record["metrics"][name] is None
    assert record["undefined_reasons"][name] == "empty_risk_quartile"
    name = "correlation/normalized/random25/current_base_q"
    assert record["metrics"][name] is None and record["undefined_reasons"][name] == "constant_rank_vector"


@pytest.mark.parametrize("x,y,expected,reason", [
    ([1., 2.], [2., 1.], -1., None),
    ([1.], [1.], None, "fewer_than_two_particles"),
    ([1., 1.], [2., 3.], None, "constant_rank_vector"),
    ([1., 2., 2., 4.], [4., 2., 2., 1.], -1., None)])
def test_correlations_with_ties_and_undefined(x, y, expected, reason):
    value, actual_reason = analysis.correlation(np.array(x), np.array(y))
    assert value == pytest.approx(expected) if expected is not None else value is None
    assert actual_reason == reason


def test_degree_concentration_for_star_uniform_and_empty():
    star = analysis.degree_statistics(np.array([4, 1, 1, 1, 1]))
    uniform = analysis.degree_statistics(np.ones(5))
    empty = analysis.degree_statistics(np.zeros(5))
    assert star["gini"] == pytest.approx(.3)
    assert star["herfindahl"] == pytest.approx(.3125)
    assert star["maximum_share"] == .5
    assert uniform["gini"] == 0 and uniform["herfindahl"] == pytest.approx(.2)
    assert empty["gini"] == 0 and empty["nonzero_fraction"] == 0
    assert empty["herfindahl"] is None and empty["maximum_share"] is None


def test_pair_incidence_and_overlap_from_saved_sets():
    row, arrays = fixture()
    record, derived = analysis.analyze_frame(row, arrays)
    assert np.array_equal(derived["optional_degree_speed25"], [2, 0, 0, 1, 1])
    assert record["metrics"]["optional_jaccard/random25__speed25"] == 0
    assert record["metrics"]["optional_jaccard/dense__random25"] == .25
    assert record["metrics"]["optional_degree/random25/mean"] == .8


@pytest.mark.parametrize("mutation,match", [
    (lambda row, a: a["position_vector_se_random25"].__setitem__(0, 12.), "Saved arithmetic differs"),
    (lambda row, a: a["signed_dense_benefit_normalized"].__setitem__(0, 12.), "Saved arithmetic differs"),
    (lambda row, a: a["current_base_q"].__setitem__(0, 0.), "Risk must be positive"),
    (lambda row, a: a["prediction_dense"].__setitem__((0, 0), np.nan), "Invalid finite numeric"),
    (lambda row, a: row.__setitem__("target_sha256", "0" * 64), "Target hash differs"),
    (lambda row, a: a["current_observed_history"].__setitem__((0, 0, 0), 100.), "History bytes/dtype differ"),
    (lambda row, a: a["pairs_random25"].__setitem__(0, [0, 0]), "Non-simple/out-of-range"),
    (lambda row, a: row["timed_calls"][0].__setitem__("pair_sha256", "0" * 64), "Selected pairs differ"),
    (lambda row, a: row["graph_audit"]["overlaps"]["base__dense"].__setitem__("optional_union", 999), "Overlap counts differ"),
    (lambda row, a: row["saved_acceleration_normalization"].__setitem__("std", [0., 1.]), "Normalization must be positive")])
def test_invalid_input_never_becomes_a_scientific_result(mutation, match):
    row, arrays = fixture()
    mutation(row, arrays)
    with pytest.raises(ValueError, match=match):
        analysis.analyze_frame(row, arrays)


def test_failed_frame_retains_reason_without_requiring_prediction_arrays():
    row = {"source_index": 3, "target_frame": 7, "trajectory_id": "synthetic", "status": "failed",
           "n_particles": 5, "failure": {"category": "synthetic_nonfinite"}}
    result, arrays = analysis.analyze_frame(row, {})
    assert result["failure"] == row["failure"] and arrays == {}
    assert all(value is None for value in result["metrics"].values())
    assert set(result["undefined_reasons"].values()) == {"failed_frame"}


def test_aggregation_equal_frames_then_trajectories_not_particles():
    rows = [scalar_frame(3, 7, 2), scalar_frame(3, 9, 4), scalar_frame(4, 7, 8), scalar_frame(4, 9, 10)]
    rows[0]["n_particles"] = 100000
    result = analysis.aggregate_frames(rows, (3, 4), (7, 9))
    for name in analysis.METRICS:
        assert result["trajectories"]["3"][name] == 3
        assert result["trajectories"]["4"][name] == 9
        assert result["metrics"][name]["equal_trajectory_mean"] == 6


def test_missing_frame_and_undefined_metric_never_available_case_averaged():
    rows = [scalar_frame(3, 7, 2), scalar_frame(3, 9, 4), scalar_frame(4, 7, 8)]
    result = analysis.aggregate_frames(rows, (3, 4), (7, 9))
    assert result["frame_status_counts"] == {"complete": 3, "missing": 1}
    assert all(value["equal_trajectory_mean"] is None for value in result["metrics"].values())
    assert result["metrics"][analysis.METRICS[0]]["defined_frames"] == 3
    assert result["metrics"][analysis.METRICS[0]]["undefined_reason_counts"] == {"missing_frame": 1}
    rows.append(scalar_frame(4, 9, 10))
    name = analysis.METRICS[0]
    rows[0]["metrics"][name] = None
    rows[0]["undefined_reasons"][name] = "empty_risk_quartile"
    result = analysis.aggregate_frames(rows, (3, 4), (7, 9))
    assert result["metrics"][name]["equal_trajectory_mean"] is None
    assert result["metrics"][analysis.METRICS[1]]["equal_trajectory_mean"] == 6


def test_trajectory_action_signs_use_trajectory_mean_not_particle_or_frame_majority():
    rows = [scalar_frame(3, 7, 1.), scalar_frame(3, 9, -3.), scalar_frame(4, 7, 3.), scalar_frame(4, 9, 3.)]
    result = analysis.aggregate_frames(rows, (3, 4), (7, 9))
    for outcome in result["trajectory_action_outcomes"].values():
        assert outcome["counts"] == {"helpful": 1, "harmful": 1, "zero": 0, "undefined": 0}
        assert outcome["fractions"] == {"helpful": .5, "harmful": .5, "zero": 0}
    result = analysis.aggregate_frames(rows[:-1], (3, 4), (7, 9))
    for outcome in result["trajectory_action_outcomes"].values():
        assert outcome["counts"]["undefined"] == 1
        assert all(value is None for value in outcome["fractions"].values())


@pytest.mark.parametrize("rows", [[scalar_frame(3, 7, 2)] * 2, [scalar_frame(99, 7, 2)]])
def test_duplicate_and_unknown_frame_identity_rejected(rows):
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        analysis.aggregate_frames(rows, (3,), (7,))


def test_three_seed_sample_sd_and_missing_seed_propagation():
    result = analysis.across_seeds([1., 2., 6.])
    assert result["mean"] == 3 and result["sample_seed_sd"] == pytest.approx(np.std([1, 2, 6], ddof=1))
    assert result["seed_values"] == [1., 2., 6.]
    partial = analysis.across_seeds([1., None, 6.])
    assert partial["mean"] is None and partial["sample_seed_sd"] is None and partial["defined_seeds"] == 2
    with pytest.raises(ValueError, match="Exactly three"):
        analysis.across_seeds([1., 2.])


def test_inventory_detects_mutation_addition_and_missing_file(tmp_path):
    directory = tmp_path / "faithful_seed0"
    directory.mkdir()
    file = directory / "protocol.json"
    file.write_text('{"synthetic":1}')
    before = analysis.inventory(tmp_path)
    analysis.verify_inventory(tmp_path, before)
    file.write_text('{"synthetic":2}')
    with pytest.raises(ValueError, match="snapshot changed"):
        analysis.verify_inventory(tmp_path, before)
    file.write_text('{"synthetic":1}')
    (directory / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot changed"):
        analysis.verify_inventory(tmp_path, before)
    (directory / "result.json").unlink()
    file.unlink()
    with pytest.raises(ValueError, match="snapshot changed"):
        analysis.verify_inventory(tmp_path, before)


def test_input_symlink_rejected(tmp_path):
    directory = tmp_path / "faithful_seed0"
    directory.mkdir()
    source = tmp_path / "elsewhere.json"
    source.write_text("{}")
    (directory / "protocol.json").symlink_to(source)
    with pytest.raises(ValueError, match="nonsymlink"):
        analysis.inventory(tmp_path)


def test_existing_output_refused_without_modification(tmp_path):
    from argparse import Namespace
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("prior adverse output")
    with pytest.raises(ValueError, match="Existing analysis"):
        analysis.run(Namespace(evaluation_root=tmp_path / "inputs", output_dir=output))
    assert marker.read_text() == "prior adverse output"


def test_outputs_cannot_be_inside_frozen_inputs(tmp_path):
    from argparse import Namespace
    root = tmp_path / "inputs"
    with pytest.raises(ValueError, match="outside frozen evidence"):
        analysis.run(Namespace(evaluation_root=root, output_dir=root / "new"))
    assert not root.exists()


def test_json_is_finite_and_roundtrips(tmp_path):
    row, arrays = fixture()
    record, _ = analysis.analyze_frame(row, arrays)
    output = tmp_path / "record.json"
    analysis.atomic_json(output, record)
    assert json.loads(output.read_text()) == record
    assert not output.with_suffix(".json.tmp").exists()


def test_import_does_not_load_torch_or_model_modules():
    code = "import sys; from research import analyze_full_action_benefit; assert 'torch' not in sys.modules; assert not any(x.startswith('gns.') for x in sys.modules)"
    completed = subprocess.run([sys.executable, "-c", code], cwd=analysis.ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("inject_failure", [False, True])
def test_synthetic_run_writes_derived_provenance_and_preserves_failure(tmp_path, monkeypatch, inject_failure):
    """The existing strict schema loader has its own tests; mock its boundary.

    This exercises our complete disk path using a partial, synthetic snapshot,
    never a real dataset/array or model.
    """
    from argparse import Namespace
    root = tmp_path / "synthetic_inputs"
    directory = root / "faithful_seed0"
    directory.mkdir(parents=True)
    row, arrays = fixture()
    stem = "trajectory_000003_target_0007"
    record_path, array_path = directory / (stem + ".json"), directory / (stem + ".npz")
    with array_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    analysis.atomic_json(record_path, row)
    compact = {"record_file": record_path.name, "record_sha256": analysis.sha256(record_path),
               "array_file": array_path.name, "array_sha256": analysis.sha256(array_path)}
    runs = [{"objective": objective, "seed": seed, "state": "incomplete", "errors": [], "eligible": False,
             "frames": [compact] if objective == "faithful" and seed == 0 else []}
            for objective in analysis.strict.OBJECTIVES for seed in analysis.strict.SEEDS]
    # The strict loader uses this error to denote a valid, uncommitted snapshot;
    # it must remain incomplete/null rather than become an integrity failure.
    runs[0]["errors"] = ["Completion is not committed in status.json"]
    monkeypatch.setattr(analysis.strict, "summarize", lambda *args: {"state": "incomplete", "consistency_errors": [], "runs": runs})
    if inject_failure:
        def fail(*args):
            raise ValueError("synthetic integrity failure")
        monkeypatch.setattr(analysis, "analyze_frame", fail)
    output = tmp_path / "new_analysis"
    args = Namespace(evaluation_root=root, output_dir=output, protocol=analysis.PROTOCOL,
                     training_protocol=analysis.ROOT / "research/protocols/full_waterdrop_100k.md",
                     companion_protocol=analysis.ROOT / "research/protocols/full_same_state_diagnostic.md")
    before = analysis.inventory(root)
    if inject_failure:
        with pytest.raises(ValueError, match="synthetic integrity failure"):
            analysis.run(args)
        status = json.loads((output / "status.json").read_text())
        assert status["state"] == "error" and status["partial_outputs_preserved"] is True
        assert (output / "input_identity.json").exists()
    else:
        result = analysis.run(args)
        assert result["state"] == "incomplete" and result["input_files_verified_after"] == 2
        assert json.loads((output / "status.json").read_text())["state"] == "incomplete"
        derived = output / "frames/faithful_seed0" / (stem + ".npz")
        record = json.loads((derived.with_suffix(".json")).read_text())
        assert record["derived_array_sha256"] == analysis.sha256(derived)
        assert record["input_record_sha256"] == compact["record_sha256"]
        assert all(value["mean"] is None for value in result["objectives"]["faithful"].values())
        assert all(value["mean"] is None for value in result["paired_nll_minus_faithful"].values())
    analysis.verify_inventory(root, before)
