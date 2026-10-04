"""Independent schema-1 synthetic artifact fixtures; no models or source data."""
import json
from pathlib import Path
import shutil
import statistics

import pytest

from research import summarize_full_same_state as summary

H = summary.shared.canonical_hash
TRAINING_HASH, COMPANION_HASH, EVALUATOR_HASH = map(H, ("synthetic training protocol", "synthetic companion protocol", "synthetic evaluator source"))


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")))


def read(path):
    return json.loads(path.read_text())


def protocol(objective, seed):
    metadata_hash = H("synthetic metadata")
    shared_source = {"adaptive-gns/gns/example.py": H("synthetic shared source")}
    training = {"objective": objective, "seed": seed, "scope": "bounded_full_data_100k", "steps": 100000,
                "batch_size": 2, "history": 6, "noise_std": 6.7e-4,
                "architecture": {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
                "metadata_sha256": metadata_hash, "research_protocol_sha256": TRAINING_HASH,
                "source_sha256": shared_source}
    for split, count, digest in (("train", 1000, summary.shared.OFFICIAL_TRAIN_SHA256), ("valid", 30, summary.shared.OFFICIAL_VALID_SHA256)):
        training[split] = {"n_trajectories": count, "source": {"sha256": digest}, "manifest_sha256": H(["synthetic", split, "manifest"])}
    records = [{"source_index": source, "id": f"test:{source:06d}",
                "positions": {"shape": [1001, source + 1, 2], "sha256": H([source, "synthetic positions"])},
                "particle_types": {"sha256": H([source, "synthetic types"])}, "trajectory_content_sha256": H([source, "synthetic content"])}
               for source in summary.SOURCE_INDICES]
    return {"schema": 1, "scope": "locked_test", "split": "test", "objective": objective, "seed": seed,
            "training_protocol_sha256": TRAINING_HASH, "companion_protocol_sha256": COMPANION_HASH,
            "checkpoint_sha256": H(["synthetic checkpoint", objective, seed]), "metadata_sha256": metadata_hash,
            "manifest_sha256": H("synthetic test manifest"), "source_tfrecord": {"sha256": summary.shared.OFFICIAL_TEST_SHA256,
                "record_count": 30, "CRC_verified": True}, "trajectory_records": records,
            "target_frames": list(summary.TARGET_FRAMES), "policies": list(summary.POLICIES), "timing_cases": list(summary.CASES),
            "timed_rounds": 6, "warmup_rounds": 1, "guards": {"max_candidate_pairs": 100000, "max_absolute_coordinate": 10.},
            "code_sha256": {**shared_source, "research/full_same_state.py": EVALUATOR_HASH},
            "checkpoint_provenance": {"checkpoint_format": 2, "normalization_source": "checkpoint",
                "uncertainty_parameterization": "variance", "connectivity_radius": .015, "nmessage_passing_steps": 10,
                "training_config": {"loss": objective, "completed_optimizer_updates": 100000, "full_run": training}},
            "runtime": {"device": "synthetic CPU fixture"}, "threads": 1}


def frame(source, target, objective, seed, protocol_hash):
    n = source + 1
    warmups, timed = [], []
    for round_index in range(-1, 6):
        for slot, case in enumerate(summary.expected_order(source, target, round_index)):
            index = summary.CASES.index(case)
            risk = case == summary.POLICIES[-1]
            forward = .002 + seed * .0001 + (.0002 if objective == "nll" else 0)
            graph = .001 + index * .0001
            score = .003 + seed * .0001 if risk else 0.
            call = {"round": round_index, "slot": slot, "method": case, "warmup": round_index == -1, "status": "complete",
                    "network_passes": 1 + int(risk), "candidate_builds": 1 + int(risk),
                    "pair_sha256": H([source, target, "synthetic pairs", "base" if case == summary.REFERENCE else case]),
                    "current_graph_and_selection_seconds": graph, "current_forward_seconds": forward,
                    "score_generation_seconds": score, "score_graph_seconds": .001 if risk else 0.,
                    "score_forward_seconds": score - .001 if risk else 0.,
                    "end_to_end_seconds": graph + forward + score + .0005 + max(round_index, 0) * .00001}
            (warmups if round_index == -1 else timed).append(call)
    times = {}
    for case in summary.CASES:
        times[case] = {}
        for name in summary.TIMING_FIELDS:
            values = [call[name] for call in timed if call["method"] == case]
            times[case][name] = {"observations": 6, "expected": 6, "mean": statistics.fmean(values),
                                 "median": statistics.median(values), "sample_sd": statistics.stdev(values)}
    base = source + target * .001 + seed * 10 + (5 if objective == "nll" else 0)
    offsets = dict(zip(summary.POLICIES, [0, 1, 2, 3, -2]))
    accuracy = {case: {"position_coordinate_mse": base + offsets[case], "normalized_coordinate_mse": (base + offsets[case]) / 4}
                for case in summary.POLICIES}
    correlations = {name: {"value": value + seed * .01 + (.02 if objective == "nll" else 0), "reason": None, "particles": n}
                    for name, value in zip(summary.CORRELATIONS, [.2, -.1, .4, .1])}
    graph_policies = {}
    for case in summary.CASES:
        optional = 0 if case in ("base", summary.REFERENCE) else 8 if case == "dense" else 2
        graph_policies[case] = {"retained_pairs": 20 + optional, "retained_optional_pairs": optional, "directed_edges": 2 * (20 + optional)}
    return {"source_index": source, "target_frame": target, "trajectory_id": f"test:{source:06d}", "n_particles": n,
            "status": "complete", "failure": None, "protocol_sha256": protocol_hash,
            "random_seed_material": [93000, seed, source, target], "model_history_dtype": "float32", "radius_classification_dtype": "float64",
            "saved_acceleration_normalization": {"mean": [0., 0.], "std": [2., 2.]},
            "observed_history_sha256": H([source, target, "synthetic observed"]),
            "previous_observed_history_sha256": H([source, target, "synthetic previous"]), "target_sha256": H([source, target, "synthetic target"]),
            "warmup_calls": warmups, "timed_calls": timed, "completed_network_passes": 49, "failed_call_attempted_network_passes": 0,
            "accuracy": accuracy, "benefit": {"mean_normalized_vector_benefit": -.5, "mean_position_vector_benefit": -2.,
                "positive_fraction": 0., "negative_fraction": 1., "zero_fraction": 0.}, "correlations": correlations,
            "timing_summary": times, "graph_audit": {"base_pairs": 20, "annulus_pairs": 8, "candidate_pairs": 28, "optional_budget": 2,
                "policies": graph_policies, "natural_shared_float64_base_identical": True,
                "frozen_float32_comparison": {"base_symmetric_difference": 0, "annulus_symmetric_difference": 0}},
            "repeat_consistency": {case: {"distinct_pair_hashes": 1, "maximum_absolute_prediction_difference": 0.} for case in summary.CASES},
            "natural_shared_base_maximum_prediction_difference": 0.}


def compact(row, path):
    return {**{key: row[key] for key in summary.COMPACT_FIELDS}, "record_file": path.name, "record_sha256": summary.sha256(path)}


def commit(directory, saved):
    write(directory / "result.json", saved)
    write(directory / "status.json", {"state": "complete" if saved["state"] == "complete" else "running",
                                       "result_sha256": summary.sha256(directory / "result.json")})


def build_run(directory, objective, seed):
    directory.mkdir(parents=True)
    provenance = protocol(objective, seed)
    write(directory / "protocol.json", provenance)
    protocol_hash = summary.sha256(directory / "protocol.json")
    records = []
    for source, target in summary.EXPECTED_KEYS:
        row = frame(source, target, objective, seed, protocol_hash)
        stem = f"trajectory_{source:06d}_target_{target:04d}"
        arrays = directory / (stem + ".npz")
        arrays.write_bytes(b"synthetic checksum fixture; not a real particle array")
        row.update(array_file=arrays.name, array_sha256=summary.sha256(arrays))
        path = directory / (stem + ".json")
        write(path, row)
        records.append(compact(row, path))
    commit(directory, {"state": "complete", "scope": "locked_test", "protocol_sha256": protocol_hash,
                       "checkpoint_sha256": provenance["checkpoint_sha256"], "objective": objective, "seed": seed,
                       "records": records, "summary": {"ignored": "must recompute aggregation from records"}})


@pytest.fixture(scope="module")
def complete_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("same_state_six_model_synthetic")
    for objective in summary.OBJECTIVES:
        for seed in summary.SEEDS:
            build_run(root / f"{objective}_seed{seed}", objective, seed)
    return root


@pytest.fixture
def copied_fixture(tmp_path, complete_fixture):
    destination = tmp_path / "copy"
    destination.mkdir()
    # Only faithful seed 0 is mutated. Other runs are immutable shared fixture
    # directories, avoiding thousands of redundant copies per negative test.
    for directory in complete_fixture.iterdir():
        target = destination / directory.name
        if directory.name == "faithful_seed0":
            shutil.copytree(directory, target)
        else:
            target.symlink_to(directory, target_is_directory=True)
    return destination


def collect(root):
    return summary.summarize(root, TRAINING_HASH, COMPANION_HASH, EVALUATOR_HASH)


def change_frame(directory, transform, source=3, target=7):
    path = directory / f"trajectory_{source:06d}_target_{target:04d}.json"
    row = read(path); transform(row); write(path, row)
    saved = read(directory / "result.json")
    saved["records"] = [compact(row, path) if (item["source_index"], item["target_frame"]) == (source, target) else item
                         for item in saved["records"]]
    commit(directory, saved)


def test_empty_prepared_or_missing_runs_are_pending_without_fabricated_statistics(tmp_path):
    (tmp_path / "faithful_seed0").mkdir()
    result = collect(tmp_path)
    assert result["state"] == "pending" and result["complete_models"] == 0
    assert len(result["runs"]) == 6
    for objective in summary.OBJECTIVES:
        group = result["objectives"][objective]
        assert all(value["mean"] is None and value["sample_sd"] is None for value in group["metrics"].values())
        assert all(len(seed["missing_frames"]) == 297 for seed in group["per_seed"])
    assert "Results pending" in summary.render_report(result)
    assert "no experimental results" in summary.render_report(result)


def test_complete_fixture_uses_equal_trajectory_seed_means_and_three_model_sample_sd(complete_fixture):
    result = collect(complete_fixture)
    assert result["state"] == "complete", [(run["state"], run["errors"]) for run in result["runs"]]
    expected0 = 16 + statistics.fmean(summary.TARGET_FRAMES) * .001
    metric = result["objectives"]["faithful"]["metrics"]["accuracy/base/position_coordinate_mse"]
    assert [item["value"] for item in metric["seed_values"]] == pytest.approx([expected0, expected0 + 10, expected0 + 20])
    assert metric["mean"] == pytest.approx(expected0 + 10)
    assert metric["sample_sd"] == pytest.approx(10.)
    weighted = result["objectives"]["faithful"]["particle_weighted_error_and_benefit"]["accuracy/base/position_coordinate_mse"]
    assert weighted["mean"] != metric["mean"]
    paired = result["paired_comparisons"]["faithful:previous-observed-base-risk25-minus-base"]
    assert paired["metrics"]["accuracy/position_coordinate_mse"]["mean"] == pytest.approx(-2.)
    assert result["paired_comparisons"]["nll-minus-faithful"]["metrics"]["accuracy/base/position_coordinate_mse"]["mean"] == pytest.approx(5.)
    # Natural and reference base timing remain separate; repeat SD is not the
    # reported between-trained-seed SD.
    natural = result["objectives"]["faithful"]["metrics"]["timing/base/end_to_end_seconds"]
    reference = result["objectives"]["faithful"]["metrics"]["timing/base_shared_superset/end_to_end_seconds"]
    assert reference["mean"] - natural["mean"] == pytest.approx(.0005)
    assert natural["sample_sd"] == pytest.approx(.0001)
    assert natural["sample_sd"] != pytest.approx(statistics.stdev([.00001 * i for i in range(6)]))
    score = result["objectives"]["faithful"]["metrics"][f"timing/{summary.POLICIES[-1]}/score_generation_seconds"]
    assert score["mean"] == pytest.approx(.0031)
    assert result["objectives"]["faithful"]["metrics"]["benefit/mean_normalized_vector_benefit"]["mean"] == -.5


def test_missing_seed_prevents_available_seed_average(copied_fixture):
    (copied_fixture / "faithful_seed2").unlink()
    result = collect(copied_fixture)
    assert result["state"] == "incomplete" and result["complete_models"] == 5
    metric = result["objectives"]["faithful"]["metrics"]["accuracy/base/position_coordinate_mse"]
    assert metric["defined_seed_count"] == 2 and metric["mean"] is None and metric["sample_sd"] is None
    assert result["paired_comparisons"]["nll-minus-faithful"]["metrics"]["accuracy/base/position_coordinate_mse"]["mean"] is None


def test_partial_result_with_all_frames_is_not_promoted_to_complete(copied_fixture):
    directory = copied_fixture / "faithful_seed0"
    saved = read(directory / "result.json"); saved["state"] = "partial"; commit(directory, saved)
    result = collect(copied_fixture)
    assert result["state"] == "incomplete" and result["complete_models"] == 5
    seed = result["objectives"]["faithful"]["per_seed"][0]
    assert seed["observed_frames"] == 297
    assert all(value is None for value in seed["equal_trajectory_metrics"].values())


def test_failed_required_frame_makes_accuracy_benefit_and_timing_null_without_dropping_it(copied_fixture):
    def fail(row):
        row.update(status="failed", failure={"category": "nonfinite_risk", "round": 0, "slot": 2},
                   accuracy=None, benefit=None, correlations=None, timing_summary=None, graph_audit=None)
    change_frame(copied_fixture / "faithful_seed0", fail)
    result = collect(copied_fixture)
    assert result["state"] == "complete"  # Completed evaluation, with a failed frame.
    group = result["objectives"]["faithful"]
    assert group["per_seed"][0]["failed_frames"] == 1 and len(group["per_seed"][0]["failures"]) == 1
    for name in ("accuracy/base/position_coordinate_mse", "benefit/mean_normalized_vector_benefit", "timing/base/end_to_end_seconds"):
        assert group["metrics"][name]["mean"] is None
    assert group["metrics"]["failure_fraction"]["mean"] == pytest.approx(1 / (297 * 3))
    assert result["paired_comparisons"]["faithful:previous-observed-base-risk25-minus-base"]["metrics"]["accuracy/position_coordinate_mse"]["mean"] is None


def test_undefined_current_correlation_does_not_replace_it_with_previous_or_defined_frame_mean(copied_fixture):
    field = "current_base_risk_vs_dense_benefit"
    def undefine(row):
        row["correlations"][field].update(value=None, reason="constant_rank_vector")
    change_frame(copied_fixture / "faithful_seed0", undefine)
    result = collect(copied_fixture)
    group = result["objectives"]["faithful"]
    assert result["state"] == "complete"
    assert group["per_seed"][0]["correlation_counts"][field] == {"defined_completed_frames": 296, "undefined_completed_frames": 1}
    assert group["metrics"]["correlation/" + field]["mean"] is None
    assert group["metrics"]["correlation/previous_risk_vs_dense_benefit"]["mean"] == pytest.approx(-.09)
    assert result["paired_comparisons"]["faithful:current-minus-previous-risk-correlation"]["metrics"]["dense_benefit"]["mean"] is None


@pytest.mark.parametrize("change", ["missing", "duplicate", "array_hash", "raw_hash", "timing_order", "timing_mean", "signed_benefit", "status_hash"])
def test_invalid_record_or_publication_is_rejected(copied_fixture, change):
    directory = copied_fixture / "faithful_seed0"
    if change in ("missing", "duplicate"):
        saved = read(directory / "result.json")
        if change == "missing": saved["records"].pop()
        else: saved["records"].append(saved["records"][0])
        commit(directory, saved)
    elif change == "array_hash":
        (directory / "trajectory_000003_target_0007.npz").write_bytes(b"changed fixture")
    elif change == "raw_hash":
        path = directory / "trajectory_000003_target_0007.json"
        row = read(path); row["accuracy"]["base"]["position_coordinate_mse"] += 1; write(path, row)
    elif change == "timing_order":
        change_frame(directory, lambda row: row["timed_calls"][0].update(slot=5))
    elif change == "timing_mean":
        change_frame(directory, lambda row: row["timing_summary"]["base"]["end_to_end_seconds"].update(mean=999.))
    elif change == "signed_benefit":
        change_frame(directory, lambda row: row["benefit"].update(mean_normalized_vector_benefit=.5))
    else:
        write(directory / "status.json", {"state": "complete", "result_sha256": "0" * 64})
    run = summary.load_run(directory, "faithful", 0, TRAINING_HASH, COMPANION_HASH, EVALUATOR_HASH)
    assert run["state"] == "invalid"
    assert run["eligible"] is False


def test_uncommitted_final_status_stays_incomplete(copied_fixture):
    write(copied_fixture / "faithful_seed0/status.json", {"state": "running"})
    result = collect(copied_fixture)
    assert result["state"] == "incomplete" and result["complete_models"] == 5


@pytest.mark.parametrize("change", ["scope", "steps", "seed", "source", "schedule", "policies", "metadata", "evaluator", "protocol"])
def test_wrong_scope_checkpoint_or_convention_is_rejected(change):
    value = protocol("faithful", 0)
    if change == "scope": value["scope"] = "pilot_validation"
    elif change == "steps": value["checkpoint_provenance"]["training_config"]["completed_optimizer_updates"] = 100
    elif change == "seed": value["checkpoint_provenance"]["training_config"]["full_run"]["seed"] = 1
    elif change == "source": value["source_tfrecord"]["record_count"] = 3
    elif change == "schedule": value["target_frames"].pop()
    elif change == "policies": value["policies"][-1] = "laggedrisk25"
    elif change == "metadata": value["metadata_sha256"] = H("other metadata")
    elif change == "evaluator": value["code_sha256"]["research/full_same_state.py"] = H("other source")
    else: value["companion_protocol_sha256"] = H("other protocol")
    with pytest.raises(ValueError):
        summary.validate_protocol(value, "faithful", 0, TRAINING_HASH, COMPANION_HASH, EVALUATOR_HASH)


def test_mismatched_paired_observed_history_disables_all_aggregation(copied_fixture):
    change_frame(copied_fixture / "faithful_seed0", lambda row: row.update(observed_history_sha256=H("other observed geometry")))
    result = collect(copied_fixture)
    assert result["state"] == "invalid"
    assert any("histories" in message for message in result["consistency_errors"])
    assert result["aggregation_eligible_models"] == 0
    assert result["objectives"]["nll"]["metrics"]["accuracy/base/position_coordinate_mse"]["mean"] is None


def test_cross_run_common_provenance_and_checkpoint_reuse_fail_closed(monkeypatch):
    def fake_loader(directory, objective, seed, *args):
        return {"directory": str(directory), "objective": objective, "seed": seed, "state": "complete", "eligible": True,
                "errors": [], "frames": [], "common_identity_sha256": "different" if (objective, seed) == ("nll", 2) else "same",
                "checkpoint_sha256": "duplicate"}
    monkeypatch.setattr(summary, "load_run", fake_loader)
    result = collect(Path("synthetic"))
    assert result["state"] == "invalid" and len(result["consistency_errors"]) == 2
    assert result["aggregation_eligible_models"] == 0


def test_required_297_frame_mean_does_not_accept_a_small_fixture_as_full_result():
    frames = [{"source_index": 3}] * 11
    assert summary.equal_trajectory_mean(frames, [1.] * 11) is None


def test_near_cancelling_benefit_allows_reduction_roundoff_but_rejects_inconsistent_gain():
    base_vector_se = [1e8 + 1e-5, 1e8 - 1e-5]
    dense_vector_se = [1e8, 1e8 + 1e-7]
    benefit = statistics.fmean(a - b for a, b in zip(base_vector_se, dense_vector_se))
    base = statistics.fmean(base_vector_se) / 2
    dense = statistics.fmean(dense_vector_se) / 2
    assert not summary.close(benefit, 2 * (base - dense))
    assert summary.benefit_consistent(benefit, base, dense, 2)
    assert not summary.benefit_consistent(benefit + .01, base, dense, 2)
    # The numerical exception is scoped to the cancellation identity.
    assert not summary.close(.001, .002)


def test_score_generation_cannot_discount_its_recorded_graph_and_forward_components():
    row = frame(3, 7, "faithful", 0, "synthetic_protocol")
    call = next(item for item in row["warmup_calls"] if item["method"] == summary.POLICIES[-1])
    call["score_generation_seconds"] -= .002
    call["end_to_end_seconds"] -= .002
    with pytest.raises(ValueError, match="Score generation timing omits"):
        summary.validate_calls(row)
