"""Synthetic-only identities and failure/weighting checks; no real outputs."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research import analyze_optional_exposure_decomposition as analysis
from research.tests.test_full_action_benefit import fixture


def saved_fixture():
    row, arrays = fixture()
    old_row, old_arrays = analysis.action.analyze_frame(row, arrays)
    return row, arrays, old_row, old_arrays


def core_frame(source=3, target=7, n=7, degree=None, multiplier=1.):
    row = {"source_index": source, "target_frame": target, "trajectory_id": f"test:{source:06d}", "n_particles": n, "status": "complete"}
    truth = np.zeros((n, 2))
    base = np.column_stack((np.linspace(.1, .8, n), np.linspace(.4, -.3, n)))
    predictions = {analysis.RISK: base * np.array([.8, 1.3]) * multiplier,
                   "random25": base * np.array([1.5, .4]) * multiplier,
                   "speed25": base * np.array([1.1, .7]) * multiplier}
    degrees = degree or {analysis.RISK: np.array([0, 2, 0, 1, 2, 3, 0]),
                        "random25": np.array([0, 0, 2, 3, 2, 1, 0]),
                        "speed25": np.array([1, 1, 2, 0, 2, 2, 0])}
    return row, truth, base, predictions, np.array([.1, .3]), degrees


def test_complete_original_action_replay_and_coordinate_factor_two():
    row, arrays, old_row, old_arrays = saved_fixture()
    result, derived = analysis.decompose_frame(row, arrays, old_row, old_arrays)
    assert result["status"] == "complete" and result["action_replay"]["passed"]
    for policy in analysis.POLICIES:
        for ours, previous in (("error", "error"), ("alignment", "alignment"), ("cost", "perturbation_cost")):
            np.testing.assert_allclose(derived[f"normalized_coordinate_{ours}_{policy}"], old_arrays[f"{previous}_normalized_{policy}"] / 2)
    for pair, (left, right) in analysis.PAIRS.items():
        direct = ((old_arrays[f"error_normalized_{left}"] - old_arrays[f"error_normalized_{right}"]) / 2).mean()
        assert result["metrics"][f"{pair}/whole/error_difference"] == pytest.approx(direct)


def test_primary_and_both_subgroups_exactly_partition_every_particle():
    result, arrays = analysis.decompose_predictions(*core_frame())
    groups = result["pairs"]["risk_minus_random"]["groups"]
    assert [groups[g]["particles"] for g in analysis.PRIMARY_GROUPS] == [2, 1, 1, 3]
    assert [groups[g]["particles"] for g in analysis.BOTH_GROUPS] == [1, 1, 1]
    np.testing.assert_array_equal(arrays["risk_minus_random__primary_group_id"], [0, 1, 2, 3, 3, 3, 0])
    np.testing.assert_array_equal(arrays["risk_minus_random__both_degree_subgroup_id"], [-1, -1, -1, 0, 1, 2, -1])
    for pair in analysis.PAIRS:
        for quantity in analysis.QUANTITIES:
            m = result["metrics"]
            assert sum(m[f"{pair}/{g}/{quantity}_contribution"] for g in analysis.PRIMARY_GROUPS) == pytest.approx(m[f"{pair}/whole/{quantity}_difference"])
            assert sum(m[f"{pair}/{g}/{quantity}_contribution"] for g in analysis.BOTH_GROUPS) == pytest.approx(m[f"{pair}/both/{quantity}_contribution"])


def test_cost_alignment_sign_and_coordinate_units_independently():
    args = core_frame(); result, derived = analysis.decompose_predictions(*args)
    row, truth, base, predictions, std, _ = args
    residual = (truth - base) / std
    for pair, (left, right) in analysis.PAIRS.items():
        dl, dr = (predictions[left] - base) / std, (predictions[right] - base) / std
        alignment_difference = np.sum(residual * (dl - dr), axis=1)
        cost_difference = (np.sum(dl ** 2, axis=1) - np.sum(dr ** 2, axis=1)) / 2
        direct_difference = (np.sum(((predictions[left] - truth) / std) ** 2, axis=1) - np.sum(((predictions[right] - truth) / std) ** 2, axis=1)) / 2
        np.testing.assert_allclose(derived[f"{pair}__error_difference"], direct_difference)
        np.testing.assert_allclose(derived[f"{pair}__alignment_difference"], alignment_difference)
        np.testing.assert_allclose(derived[f"{pair}__cost_difference"], cost_difference)
        np.testing.assert_allclose(cost_difference - alignment_difference, direct_difference)
    assert np.any(derived["risk_minus_random__error_difference"] < 0)


def test_neither_group_can_have_nonzero_effect_through_message_passing():
    result, _ = analysis.decompose_predictions(*core_frame())
    assert result["metrics"]["risk_minus_random/neither/particle_fraction"] > 0
    assert result["metrics"]["risk_minus_random/neither/error_contribution"] != 0


def test_empty_groups_have_zero_contributions_but_null_conditional_means():
    args = list(core_frame())
    args[-1] = {p: np.zeros(7, dtype=np.int64) for p in analysis.POLICIES}
    result, _ = analysis.decompose_predictions(*args)
    for pair in analysis.PAIRS:
        assert result["metrics"][f"{pair}/neither/particle_fraction"] == 1
        for group in analysis.GROUPS[1:]:
            for quantity in analysis.QUANTITIES:
                assert result["metrics"][f"{pair}/{group}/{quantity}_contribution"] == 0
                name = f"{pair}/{group}/conditional_mean_{quantity}_difference"
                assert result["metrics"][name] is None and result["undefined_reasons"][name] == "empty_group"
    assert result["status"] == "complete"


def test_zero_error_zero_action_and_empty_degree_identity_envelope():
    args = list(core_frame())
    args[1] = args[2] = np.zeros((7, 2))
    args[3] = {p: np.zeros((7, 2)) for p in analysis.POLICIES}
    args[-1] = {p: np.zeros(7, dtype=np.int64) for p in analysis.POLICIES}
    result, _ = analysis.decompose_predictions(*args)
    assert all(check["max_abs_residual"] == 0 for check in result["identity_checks"])


def test_missing_action_nulls_only_required_pairs_in_core_without_salvaging_raw_inputs():
    args = list(core_frame())
    args[3] = dict(args[3]); del args[3]["random25"]
    result, _ = analysis.decompose_predictions(*args)
    assert result["status"] == "partial"
    for pair in ("risk_minus_random", "speed_minus_random"):
        assert result["metrics"][f"{pair}/whole/error_difference"] is None
        assert result["undefined_reasons"][f"{pair}/whole/error_difference"] == "missing_action:random25"
    assert result["metrics"]["risk_minus_speed/whole/error_difference"] is not None
    # Actual raw-frame replay is stricter: an absent saved action is corruption.
    row, arrays, old, previous = saved_fixture()
    del arrays["prediction_random25"]
    with pytest.raises(KeyError):
        analysis.decompose_frame(row, arrays, old, previous)


def test_failed_original_frame_is_all_null_and_preserves_guard_reason():
    row = {"source_index": 3, "target_frame": 7, "status": "failed", "failure": {"category": "coordinate_guard"}}
    record, arrays = analysis.decompose_frame(row, {}, {}, {})
    assert record["status"] == "failed" and record["failure"] == row["failure"]
    assert all(v is None for v in record["metrics"].values()) and arrays == {}


@pytest.mark.parametrize("mutation,match", [
    (lambda r,a,old,derived: derived["error_normalized_random25"].__setitem__(0, 999.), "action replay"),
    (lambda r,a,old,derived: old["metrics"].__setitem__("optional_degree/random25/mean", 999.), "Saved action record replay"),
    (lambda r,a,old,derived: a["normalized_vector_se_random25"].__setitem__(0, 999.), "Saved arithmetic differs"),
    (lambda r,a,old,derived: derived.pop("optional_degree_random25"), "derived-array keys differ"),
    (lambda r,a,old,derived: a["pairs_random25"].__setitem__(0, [0,0]), "Non-simple"),
])
def test_saved_action_or_raw_arithmetic_corruption_never_becomes_scientific_result(mutation, match):
    row, arrays, old, previous = saved_fixture()
    mutation(row, arrays, old, previous)
    with pytest.raises(ValueError, match=match):
        analysis.decompose_frame(row, arrays, old, previous)


def test_unequal_optional_budgets_rejected():
    args = list(core_frame()); args[-1]["random25"][0] = 1
    with pytest.raises(ValueError, match="equal optional budgets"):
        analysis.decompose_predictions(*args)


def test_decomposition_never_mutates_raw_inputs_or_prior_results():
    args = saved_fixture(); before = copy.deepcopy(args)
    analysis.decompose_frame(*args)
    for actual, previous in zip(args, before):
        if actual is args[0] or actual is args[2]:
            assert actual == previous
        else:
            for key in actual:
                np.testing.assert_array_equal(actual[key], previous[key])


def test_identity_guard_rejects_factor_two_and_wrong_sign():
    with pytest.raises(ValueError, match="Additive identity failed"):
        analysis.identity_check(1., 2., 4., 2, "coordinate_factor")
    result, _ = analysis.decompose_predictions(*core_frame())
    m = dict(result["metrics"])
    m["risk_minus_random/both/error_contribution"] += .001
    with pytest.raises(ValueError, match="Additive identity failed"):
        analysis.check_partition_metrics(m, 7)


def test_equal_frame_then_trajectory_weights_and_group_size_weighted_conditionals():
    frames = []
    for source, target, multiplier in ((3,7,1.), (3,106,2.), (4,7,3.), (4,106,4.)):
        args = list(core_frame(source,target,multiplier=multiplier))
        if (source,target) == (3,7):
            # A legitimate empty both group contributes zero, not a missing frame.
            args[-1] = {p: np.zeros(7,dtype=np.int64) for p in analysis.POLICIES}
        elif (source,target) == (3,106):
            # One shared-exposure particle here versus three on other frames.
            args[-1]["random25"] = np.array([1,0,2,1,0,0,4])
        frames.append(analysis.decompose_predictions(*args)[0])
    result = analysis.aggregate_frames(frames, (3,4), (7,106))
    for name in ("risk_minus_random/whole/error_difference", "risk_minus_random/both/error_contribution", "risk_minus_random/both/particle_fraction"):
        expected = np.mean([np.mean([f["metrics"][name] for f in frames if f["source_index"]==s]) for s in (3,4)])
        assert result["metrics"][name] == expected
    conditional = "risk_minus_random/both/conditional_mean_error_difference"
    assert result["metrics"][conditional] is None
    assert result["metric_coverage"][conditional]["undefined_reason_counts"] == {"empty_group":1}
    c = result["weighted_conditionals"]["risk_minus_random/both/error"]
    assert c["value"] == result["metrics"]["risk_minus_random/both/error_contribution"] / result["metrics"]["risk_minus_random/both/particle_fraction"]
    assert c["value"] is not None
    available_conditionals = [f["metrics"][conditional] for f in frames if f["metrics"][conditional] is not None]
    assert c["value"] != pytest.approx(np.mean(available_conditionals))
    assert result["trajectories"]["3"]["metrics"][conditional] is None


def test_different_particle_counts_do_not_change_equal_frame_trajectory_weights():
    first = analysis.decompose_predictions(*core_frame(3,7,multiplier=1.))[0]
    args = list(core_frame(4,7,multiplier=2.))
    args[0]["n_particles"] = 14
    args[1] = np.tile(args[1],(2,1)); args[2] = np.tile(args[2],(2,1))
    args[3] = {p:np.tile(a,(2,1)) for p,a in args[3].items()}
    args[5] = {p:np.tile(d,2) for p,d in args[5].items()}
    second = analysis.decompose_predictions(*args)[0]
    result = analysis.aggregate_frames([first,second],(3,4),(7,))
    key = "risk_minus_random/whole/error_difference"
    a,b = first["metrics"][key],second["metrics"][key]
    assert result["metrics"][key] == pytest.approx((a+b)/2)
    assert result["metrics"][key] != pytest.approx((7*a+14*b)/21)


def test_missing_frame_nulls_contributions_no_available_frame_rescue():
    frames = [analysis.decompose_predictions(*core_frame(3,7))[0]]
    result = analysis.aggregate_frames(frames, (3,), (7,106))
    assert result["required_frames"] == 2 and result["observed_frames"] == 1
    assert result["frame_status_counts"] == {"complete":1, "missing":1}
    assert all(value is None for value in result["metrics"].values())
    assert result["weighted_conditionals"]["risk_minus_random/both/error"]["reason"] == "missing_required_input"


def test_aggregate_rejects_duplicate_or_unexpected_frame():
    record, _ = analysis.decompose_predictions(*core_frame())
    with pytest.raises(ValueError, match="Duplicate or unexpected"):
        analysis.aggregate_frames([record,record], (3,), (7,))
    with pytest.raises(ValueError, match="Duplicate or unexpected"):
        analysis.aggregate_frames([record], (4,), (7,))


def test_three_seed_sample_sd_and_missing_seed_null_propagation():
    runs = {}
    for obj in analysis.strict.OBJECTIVES:
        for seed in analysis.strict.SEEDS:
            record, _ = analysis.decompose_predictions(*core_frame(multiplier=seed+1))
            runs[f"{obj}_seed{seed}"] = {"aggregate":analysis.aggregate_frames([record],(3,),(7,))}
    result = analysis.summarize_objectives(runs)
    key="risk_minus_random/whole/error_difference"
    values=[runs[f"faithful_seed{s}"]["aggregate"]["metrics"][key] for s in range(3)]
    assert result["faithful"]["metrics"][key]["mean"] == np.mean(values)
    assert result["faithful"]["metrics"][key]["sample_seed_sd"] == np.std(values,ddof=1)
    runs["faithful_seed2"]={"aggregate":analysis.aggregate_frames([], (3,), (7,))}
    result=analysis.summarize_objectives(runs)
    assert result["faithful"]["metrics"][key]["mean"] is None
    assert result["faithful"]["metrics"][key]["defined_seeds"] == 2
    assert result["nll"]["metrics"][key]["mean"] is not None


def test_existing_output_attempt_is_never_overwritten(tmp_path):
    output=tmp_path/"attempt";output.mkdir();(output/"unsuccessful.json").write_text("preserve")
    args=SimpleNamespace(evaluation_root=tmp_path/"source",action_root=tmp_path/"action",output_dir=output)
    with pytest.raises(ValueError,match="Preserve existing"):
        analysis.run(args)
    assert (output/"unsuccessful.json").read_text()=="preserve"


def test_path_traversal_and_symlink_inputs_rejected(tmp_path):
    inside=tmp_path/"input";inside.mkdir();outside=tmp_path/"outside";outside.write_text("x")
    with pytest.raises(ValueError,match="escapes"):
        analysis.checked_path(inside,"../outside")
    (inside/"link").symlink_to(outside)
    with pytest.raises(ValueError):
        analysis.checked_path(inside,"link")


def test_frozen_dependency_hashes_match_without_loading_models_or_outputs():
    assert all(analysis.sha256(analysis.ROOT/path)==digest for path,digest in analysis.DEPENDENCIES.items())


def synthetic_io_fixture(tmp_path, monkeypatch):
    """Six synthetic models, one frame each; mock only the prior cohort gate."""
    monkeypatch.setattr(analysis.strict, "SOURCE_INDICES", (3,))
    monkeypatch.setattr(analysis.strict, "TARGET_FRAMES", (7,))
    monkeypatch.setattr(analysis.strict, "EXPECTED_KEYS", ((3,7),))
    # Defaults were captured at function definition; inject the tiny schedule
    # only at this test seam, preserving production's 27x11 population.
    aggregate = analysis.aggregate_frames
    monkeypatch.setattr(analysis, "aggregate_frames", lambda records: aggregate(records, (3,), (7,)))
    evaluation, previous = tmp_path/"evaluation", tmp_path/"action"
    evaluation.mkdir();previous.mkdir()
    checked_runs=[];old_result={"runs":{}}
    for obj in analysis.strict.OBJECTIVES:
        for seed in analysis.strict.SEEDS:
            label=f"{obj}_seed{seed}"
            folder=evaluation/label;folder.mkdir()
            prior_folder=previous/"frames"/label;prior_folder.mkdir(parents=True)
            row,arrays,old_record,old_arrays=saved_fixture()
            stem="trajectory_000003_target_0007"
            raw_array=folder/(stem+".npz");np.savez_compressed(raw_array,**arrays)
            raw_row=folder/(stem+".json");analysis.atomic_json(raw_row,row)
            raw_hash,array_hash=analysis.sha256(raw_row),analysis.sha256(raw_array)
            prior_array=prior_folder/(stem+".npz");np.savez_compressed(prior_array,**old_arrays)
            old_record.update(input_record_sha256=raw_hash,input_array_sha256=array_hash,
                derived_array_file=str(prior_array.relative_to(previous)),derived_array_sha256=analysis.sha256(prior_array))
            prior_row=prior_folder/(stem+".json");analysis.atomic_json(prior_row,old_record)
            old_result["runs"][label]={"frames":[{"source_index":3,"target_frame":7,
                "record_file":str(prior_row.relative_to(previous)),"record_sha256":analysis.sha256(prior_row)}]}
            for name in ("protocol.json","result.json","status.json"):
                analysis.atomic_json(folder/name,{"synthetic":True})
            checked_runs.append({"objective":obj,"seed":seed,"state":"complete","eligible":True,"errors":[],
                "frames":[{"source_index":3,"target_frame":7,"record_file":raw_row.name,"record_sha256":raw_hash,
                    "array_file":raw_array.name,"array_sha256":array_hash}]})
    anchor={"input_files_sha256":analysis.action.inventory(evaluation)}
    analysis.atomic_json(previous/"input_identity.json",anchor)
    analysis.atomic_json(previous/"results.json",old_result)
    analysis.atomic_json(previous/"status.json",{"state":"complete","results_sha256":analysis.sha256(previous/"results.json")})
    monkeypatch.setattr(analysis,"ACTION_IDENTITIES",{name:analysis.sha256(previous/name) for name in analysis.ACTION_IDENTITIES})
    monkeypatch.setattr(analysis.strict,"summarize",lambda *args:{"runs":checked_runs,"consistency_errors":[]})
    return SimpleNamespace(evaluation_root=evaluation,action_root=previous,output_dir=tmp_path/"result")


def test_reduced_synthetic_io_preserves_every_expected_frame_and_finishes(tmp_path,monkeypatch):
    args=synthetic_io_fixture(tmp_path,monkeypatch)
    result=analysis.run(args)
    assert result["state"]=="complete" and result["failures"]==[]
    assert len(result["runs"])==6
    assert all(len(run["frames"])==1 for run in result["runs"].values())
    assert json.loads((args.output_dir/"status.json").read_text())["state"]=="complete"


@pytest.mark.parametrize("extension",["json","npz"])
def test_invalid_observed_prior_bytes_are_preserved_hashed_and_null_not_salvaged(tmp_path,monkeypatch,extension):
    args=synthetic_io_fixture(tmp_path,monkeypatch)
    prior=args.action_root/"frames/faithful_seed0"/f"trajectory_000003_target_0007.{extension}"
    prior.write_bytes(b"deliberately invalid synthetic prior evidence")
    actual_hash=analysis.sha256(prior)
    result=analysis.run(args)
    assert result["state"]=="incomplete" and len(result["failures"])==1
    record=json.loads((args.output_dir/"frames/faithful_seed0/trajectory_000003_target_0007.json").read_text())
    kind="record" if extension=="json" else "array"
    assert record[f"observed_action_{kind}_sha256"]==actual_hash
    assert record[f"expected_action_{kind}_sha256"]!=actual_hash
    assert record[f"observed_action_{kind}_file"]==str(prior)
    assert record["status"]=="invalid" and all(v is None for v in record["metrics"].values())
    snapshots=json.loads((args.output_dir/"all_inputs_sha256.json").read_text())
    assert snapshots[str(prior)]==actual_hash and analysis.sha256(prior)==actual_hash
    assert result["objectives"]["faithful"]["metrics"]["risk_minus_random/whole/error_difference"]["mean"] is None
    assert result["objectives"]["nll"]["metrics"]["risk_minus_random/whole/error_difference"]["mean"] is not None
