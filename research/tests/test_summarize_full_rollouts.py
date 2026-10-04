"""Synthetic artifact fixtures only; never load models or reserved test data."""
import json
import os
from pathlib import Path
import shutil
import statistics

import pytest

from research import summarize_full_rollouts as summary


PROTOCOL_HASH = summary.canonical_hash("synthetic scientific protocol")


def write_json(path, value):
    # Copies of the large per-step schema fixtures share immutable hardlinks.
    # Replace files atomically so mutations cannot change the common fixture.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")))
    temporary.replace(path)


def boundary_entry(n_particles, outside_count=0, excursion=0.):
    fraction = outside_count / n_particles
    return {"fraction_particles_outside": fraction,
            "fraction_particles_outside_by_more_than_1e-6": fraction if excursion > 1e-6 else 0.,
            "maximum_coordinate_excursion": excursion,
            "mean_particle_maximum_excursion": fraction * excursion,
            "coordinate_minimum": [0, 0], "coordinate_maximum": [1 + excursion, 1]}


def protocol(objective, seed):
    descriptors = [{"id": f"test:{index:06d}", "source_index": index,
                    "positions": {"shape": [1001, index + 1, 2], "sha256": summary.canonical_hash([index, "positions"])},
                    "particle_types": {"sha256": summary.canonical_hash([index, "types"])},
                    "trajectory_content_sha256": summary.canonical_hash([index, "content"])}
                   for index in summary.SOURCE_INDICES]
    common_source = {"gns/example.py": summary.canonical_hash("synthetic shared code")}
    run = {"scope": "bounded_full_data_100k", "steps": 100000, "batch_size": 2, "history": 6,
           "noise_std": 6.7e-4, "objective": objective, "seed": seed,
           "architecture": {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
           "metadata_sha256": summary.canonical_hash("synthetic metadata"), "research_protocol_sha256": PROTOCOL_HASH,
           "source_sha256": common_source,
           "train": {"n_trajectories": 1000, "source": {"sha256": summary.OFFICIAL_TRAIN_SHA256},
                     "manifest_sha256": summary.canonical_hash("synthetic train manifest")},
           "valid": {"n_trajectories": 30, "source": {"sha256": summary.OFFICIAL_VALID_SHA256},
                     "manifest_sha256": summary.canonical_hash("synthetic valid manifest")}}
    return {"schema": 2, "scope": "locked_test", "split": "test", "objective": objective, "seed": seed,
            "horizon": 995, "context_frames": 6, "policies": list(summary.POLICIES),
            "source_tfrecord": {"sha256": summary.OFFICIAL_TEST_SHA256, "CRC_verified": True, "record_count": 30},
            "pinned_test_source_sha256": summary.OFFICIAL_TEST_SHA256, "trajectory_records": descriptors,
            "trajectory_ids": [row["id"] for row in descriptors],
            "manifest_sha256": summary.canonical_hash("synthetic test manifest"),
            "metadata_sha256": run["metadata_sha256"], "research_protocol_sha256": PROTOCOL_HASH,
            "checkpoint_sha256": summary.canonical_hash(["synthetic checkpoint", objective, seed]),
            "graph": {"base_radius": .015, "radius_factor": 1.267, "optional_fraction": .25},
            "guards": {"max_candidate_pairs": 100000, "max_absolute_coordinate": 10.},
            "random_seed": {"base": 93000}, "code_sha256": common_source,
            "checkpoint_provenance": {"checkpoint_format": 2, "normalization_source": "checkpoint",
                "uncertainty_parameterization": "variance", "nmessage_passing_steps": 10,
                "connectivity_radius": .015, "training_config": {"loss": objective,
                    "completed_optimizer_updates": 100000, "full_run": run}}}


def commit_result(directory, value):
    write_json(directory / "result.json", value)
    write_json(directory / "status.json", {"state": "complete" if value["state"] == "complete" else "running",
                                          "result_sha256": summary.sha256(directory / "result.json")})


def compact(row, path):
    return {**{key: row[key] for key in summary.COMPACT_FIELDS},
            "record_file": path.name, "record_sha256": summary.sha256(path)}


def build_run(directory, objective, seed):
    directory.mkdir(parents=True)
    provenance = protocol(objective, seed)
    write_json(directory / "protocol.json", provenance)
    protocol_hash = summary.sha256(directory / "protocol.json")
    records = []
    for index in summary.SOURCE_INDICES:
        for policy in summary.POLICIES:
            policy_offset = {"base": 0, "dense": 1, "random25": 2, "speed25": 3, "laggedrisk25": -2}[policy]
            value = float(index + 10 * seed + (5 if objective == "nll" else 0) + policy_offset)
            trace_path = directory / f"trajectory_{index:06d}_{policy}.npz"
            # A synthetic checksum payload, never presented as a real model trace.
            trace_path.write_bytes(b"synthetic fixture trace")
            row = {"trajectory_id": f"test:{index:06d}", "source_index": index, "policy": policy,
                   "status": "complete", "failure": None, "horizon": 995, "n_particles": index + 1,
                   "rng_seed": 93000 + 1000 * seed + index, "completed_steps": 995,
                   "mse_at_steps": {str(step): value for step in summary.TRACE_STEPS},
                   "mse_at_final_horizon": value, "mean_rollout_mse": value,
                   "mean_directed_edges": float(index * 2), "total_wall_seconds": float(index),
                   "total_network_passes": 995 + int(policy == "laggedrisk25"), "forecast_network_passes": 995,
                   "trace_file": trace_path.name, "trace_sha256": summary.sha256(trace_path),
                   "protocol_sha256": protocol_hash, "mse_per_step": [value] * 995,
                   "directed_edges_per_step": [index * 2] * 995,
                   "predicted_boundary_per_step": [boundary_entry(index + 1, seed + 1, .001 * (index + seed + 1))]
                                                   + [boundary_entry(index + 1)] * 994,
                   "ground_truth_boundary_per_step": [boundary_entry(index + 1, 1, 2e-6)]
                                                      + [boundary_entry(index + 1)] * 994}
            row_path = directory / f"trajectory_{index:06d}_{policy}.json"
            write_json(row_path, row)
            records.append(compact(row, row_path))
    result = {"state": "complete", "protocol_sha256": protocol_hash, "objective": objective, "seed": seed,
              "checkpoint_sha256": provenance["checkpoint_sha256"], "records": records,
              "summary": {"deliberately_ignored": "aggregator must recompute its own statistics"}}
    commit_result(directory, result)


@pytest.fixture(scope="module")
def complete_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("six_model_synthetic_fixture")
    for objective in summary.OBJECTIVES:
        for seed in summary.SEEDS:
            build_run(root / f"{objective}_seed{seed}", objective, seed)
    return root


@pytest.fixture
def copied_fixture(tmp_path, complete_fixture):
    destination = tmp_path / "copy"
    shutil.copytree(complete_fixture, destination, copy_function=os.link)
    return destination


def read(path):
    return json.loads(path.read_text())


def change_row(directory, index, policy, transform):
    path = directory / f"trajectory_{index:06d}_{policy}.json"
    row = read(path)
    transform(row)
    write_json(path, row)
    result = read(directory / "result.json")
    result["records"] = [compact(row, path) if (item["source_index"], item["policy"]) == (index, policy) else item
                         for item in result["records"]]
    commit_result(directory, result)


def test_empty_input_is_explicitly_pending_and_contains_no_fabricated_means(tmp_path):
    result = summary.summarize(tmp_path / "no_evaluation", PROTOCOL_HASH)
    assert result["state"] == "pending" and result["complete_models"] == 0
    assert len(result["runs"]) == 6
    assert all(run["state"] == "missing" for run in result["runs"])
    for objective in summary.OBJECTIVES:
        for policy in summary.POLICIES:
            group = result["policies"][objective][policy]
            assert all(metric["mean"] is None and metric["sample_sd"] is None for metric in group["metrics"].values())
            assert all(seed["missing_source_indices"] == list(summary.SOURCE_INDICES) for seed in group["per_seed"])
    assert "Results pending" in summary.render_report(result)
    assert "no experimental outcomes" in summary.render_report(result)


def test_empty_prepared_directories_remain_pending(tmp_path):
    (tmp_path / "faithful_seed0").mkdir()
    result = summary.summarize(tmp_path, PROTOCOL_HASH)
    assert result["state"] == "pending"
    assert result["runs"][0]["state"] == "pending_evaluation"


def test_equal_trajectory_then_three_seed_mean_and_sample_sd(complete_fixture):
    result = summary.summarize(complete_fixture, PROTOCOL_HASH)
    assert result["state"] == "complete", result["runs"]
    assert result["complete_models"] == 6
    metric = result["policies"]["faithful"]["base"]["metrics"]["mean_rollout_mse"]
    assert [row["value"] for row in metric["seed_values"]] == [16., 26., 36.]
    assert metric["mean"] == 26. and metric["sample_sd"] == 10.
    # Particle counts vary with source index: a particle-weighted mean differs.
    weighted = sum(i * (i + 1) for i in range(3, 30)) / sum(i + 1 for i in range(3, 30))
    assert metric["seed_values"][0]["value"] != weighted
    paired = result["paired_comparisons"]["faithful:laggedrisk25-minus-base"]
    assert paired["metrics"]["mean_rollout_mse"]["mean"] == -2.
    assert paired["metrics"]["mean_rollout_mse"]["sample_sd"] == 0.
    assert all(row["paired_trajectories"] == 27 for row in paired["per_seed"])
    assert result["paired_comparisons"]["nll-minus-faithful:base"]["metrics"]["mean_rollout_mse"]["mean"] == 5.

    boundaries = result["policies"]["faithful"]["base"]
    mean_one_particle_fraction = statistics.fmean(1 / (i + 1) for i in summary.SOURCE_INDICES) / 995
    predicted = boundaries["metrics"]["predicted_boundary_mean_fraction_outside_gt_1e_minus6"]
    assert [row["value"] for row in predicted["seed_values"]] == pytest.approx(
        [mean_one_particle_fraction * (seed + 1) for seed in summary.SEEDS])
    assert predicted["mean"] == pytest.approx(2 * mean_one_particle_fraction)
    assert predicted["sample_sd"] == pytest.approx(mean_one_particle_fraction)
    # Time means must not become trajectory maxima, or particle-weighted means.
    step_max = boundaries["metrics"]["predicted_boundary_mean_step_maximum_excursion"]
    trajectory_max = boundaries["metrics"]["predicted_boundary_trajectory_maximum_excursion"]
    assert step_max["mean"] == pytest.approx(.018 / 995)
    assert trajectory_max["mean"] == pytest.approx(.018)
    assert trajectory_max["sample_sd"] == pytest.approx(.001)
    assert predicted["seed_values"][0]["value"] != pytest.approx(27 / sum(i + 1 for i in summary.SOURCE_INDICES) / 995)
    truth = boundaries["metrics"]["ground_truth_boundary_mean_fraction_outside_gt_1e_minus6"]
    assert truth["mean"] == pytest.approx(mean_one_particle_fraction) and truth["sample_sd"] == 0
    assert boundaries["metrics"]["ground_truth_boundary_trajectory_maximum_excursion"]["mean"] == pytest.approx(2e-6)
    assert all(seed["all_sample_full_horizon_boundary_defined"] for seed in boundaries["per_seed"])
    assert "Boundary diagnostics" in summary.render_report(result)


def test_missing_seed_never_becomes_two_seed_mean(copied_fixture):
    shutil.rmtree(copied_fixture / "faithful_seed2")
    result = summary.summarize(copied_fixture, PROTOCOL_HASH)
    assert result["state"] == "incomplete" and result["complete_models"] == 5
    metric = result["policies"]["faithful"]["base"]["metrics"]["mean_rollout_mse"]
    assert metric["defined_seed_count"] == 2 and metric["mean"] is None and metric["sample_sd"] is None
    for name in summary.BOUNDARY_METRICS:
        metric = result["policies"]["faithful"]["base"]["metrics"][name]
        assert metric["defined_seed_count"] == 2 and metric["mean"] is None and metric["sample_sd"] is None
    assert result["paired_comparisons"]["faithful:laggedrisk25-minus-base"]["metrics"]["mean_rollout_mse"]["mean"] is None


def test_partial_even_with_all_rows_never_relabels_complete(copied_fixture):
    directory = copied_fixture / "faithful_seed0"
    saved = read(directory / "result.json")
    saved["state"] = "partial"
    commit_result(directory, saved)
    result = summary.summarize(copied_fixture, PROTOCOL_HASH)
    assert result["state"] == "incomplete" and result["complete_models"] == 5
    seed = result["policies"]["faithful"]["base"]["per_seed"][0]
    assert seed["observed_trajectories"] == 27 and seed["metrics"]["mean_rollout_mse"] is None
    assert not seed["all_sample_full_horizon_boundary_defined"]
    assert all(seed["metrics"][name] is None for name in summary.BOUNDARY_METRICS)


def test_failed_trajectory_is_retained_without_survivor_mean(copied_fixture):
    def fail(row):
        row.update(status="failed", failure={"category": "coordinate_resource_guard", "forecast_step": 251},
                   completed_steps=250, mean_rollout_mse=None, mse_at_final_horizon=None, mean_directed_edges=None,
                   forecast_network_passes=251, total_network_passes=251)
        row["mse_per_step"] = row["mse_per_step"][:250]
        row["directed_edges_per_step"] = row["directed_edges_per_step"][:250]
        row["predicted_boundary_per_step"] = row["predicted_boundary_per_step"][:250]
        row["ground_truth_boundary_per_step"] = row["ground_truth_boundary_per_step"][:250]
        row["mse_at_steps"] = {str(step): row["mse_per_step"][step-1] if step <= 250 else None for step in summary.TRACE_STEPS}
    change_row(copied_fixture / "faithful_seed0", 3, "base", fail)
    result = summary.summarize(copied_fixture, PROTOCOL_HASH)
    assert result["state"] == "complete"  # Evaluation finished; trajectory failed.
    group = result["policies"]["faithful"]["base"]
    assert group["per_seed"][0]["failed_trajectories"] == 1
    assert len(group["per_seed"][0]["failures"]) == 1
    assert group["metrics"]["mean_rollout_mse"]["mean"] is None
    assert group["metrics"]["mse_at_995"]["mean"] is None
    assert group["metrics"]["mse_at_200"]["mean"] == 26.
    assert group["metrics"]["failure_fraction"]["mean"] == pytest.approx(1 / 81)
    assert not group["per_seed"][0]["all_sample_full_horizon_boundary_defined"]
    for name in summary.BOUNDARY_METRICS:
        assert group["metrics"][name]["mean"] is None
        assert group["metrics"][name]["sample_sd"] is None
        assert group["metrics"][name]["defined_seed_count"] == 2
    prefix = group["per_seed"][0]["boundary_prefix_diagnostics_if_failed"]
    assert len(prefix) == 1 and prefix[0]["source_index"] == 3 and prefix[0]["completed_steps"] == 250
    assert prefix[0]["metrics"]["predicted_boundary_mean_fraction_outside_gt_1e_minus6"] == pytest.approx(.25 / 250)
    assert prefix[0]["metrics"]["ground_truth_boundary_trajectory_maximum_excursion"] == pytest.approx(2e-6)
    assert "never used" in prefix[0]["scope"]
    paired = result["paired_comparisons"]["faithful:laggedrisk25-minus-base"]
    assert paired["metrics"]["mean_rollout_mse"]["mean"] is None
    assert paired["metrics"]["failure_fraction"]["mean"] == pytest.approx(-1 / 81)
    assert all(paired["metrics"][name]["mean"] is None for name in summary.BOUNDARY_METRICS)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_id", "fabricated_mean", "raw_tamper", "trace_tamper", "boundary_length"])
def test_invalid_records_cannot_enter_a_final_aggregate(copied_fixture, change):
    directory = copied_fixture / "faithful_seed0"
    saved = read(directory / "result.json")
    if change == "missing":
        saved["records"].pop()
        commit_result(directory, saved)
    elif change == "duplicate":
        saved["records"].append(saved["records"][0])
        commit_result(directory, saved)
    elif change == "wrong_id":
        change_row(directory, 3, "base", lambda row: row.update(trajectory_id="test:000002"))
    elif change == "fabricated_mean":
        change_row(directory, 3, "base", lambda row: row.update(mean_rollout_mse=123.))
    elif change == "raw_tamper":
        path = directory / "trajectory_000003_base.json"
        row = read(path); row["mean_rollout_mse"] = 123.; write_json(path, row)
    elif change == "boundary_length":
        change_row(directory, 3, "base", lambda row: row["ground_truth_boundary_per_step"].pop())
    else:
        trace = directory / "trajectory_000003_base.npz"
        trace.unlink()
        trace.write_bytes(b"changed fixture")
    result = summary.summarize(copied_fixture, PROTOCOL_HASH)
    assert result["state"] == "invalid"
    assert result["runs"][0]["state"] == "invalid"
    assert result["policies"]["faithful"]["base"]["metrics"]["mean_rollout_mse"]["mean"] is None


@pytest.mark.parametrize("change", ["scope", "budget", "metadata", "test_source", "objective", "seed", "holdout"])
def test_protocol_rejects_smoke_wrong_provenance_and_holdout(change):
    value = protocol("faithful", 0)
    if change == "scope":
        value["scope"] = "pilot_validation"
    elif change == "budget":
        value["checkpoint_provenance"]["training_config"]["completed_optimizer_updates"] = 100
    elif change == "metadata":
        value["metadata_sha256"] = summary.canonical_hash("other metadata")
    elif change == "test_source":
        value["source_tfrecord"]["sha256"] = summary.canonical_hash("other source")
    elif change == "objective":
        value["checkpoint_provenance"]["training_config"]["loss"] = "nll"
    elif change == "seed":
        value["checkpoint_provenance"]["training_config"]["full_run"]["seed"] = 1
    else:
        value["trajectory_records"][0]["source_index"] = 2
    with pytest.raises(ValueError):
        summary.validate_protocol(value, "faithful", 0, PROTOCOL_HASH)


def test_uncommitted_completion_and_status_hash_mismatch_are_not_complete(copied_fixture):
    directory = copied_fixture / "faithful_seed0"
    write_json(directory / "status.json", {"state": "running"})
    result = summary.summarize(copied_fixture, PROTOCOL_HASH)
    assert result["state"] == "incomplete" and result["complete_models"] == 5
    write_json(directory / "status.json", {"state": "complete", "result_sha256": "0" * 64})
    result = summary.summarize(copied_fixture, PROTOCOL_HASH)
    assert result["state"] == "invalid"


def test_cross_run_common_identity_and_checkpoint_reuse_fail_closed(monkeypatch):
    def fake_loader(directory, objective, seed, expected):
        return {"objective": objective, "seed": seed, "directory": str(directory), "state": "complete",
                "records": [], "errors": [], "eligible_for_aggregation": True,
                "common_identity_sha256": "different" if (objective, seed) == ("nll", 2) else "same",
                "checkpoint_sha256": "reused", "trajectory_ids": []}
    monkeypatch.setattr(summary, "load_run", fake_loader)
    result = summary.summarize(Path("synthetic"), PROTOCOL_HASH)
    assert result["state"] == "invalid" and len(result["consistency_errors"]) == 2
    assert all(not run["eligible_for_aggregation"] for run in result["runs"])
    assert result["policies"]["faithful"]["base"]["metrics"]["failure_fraction"]["mean"] is None


def test_seed_summary_uses_sample_sd_and_requires_all_three_slots():
    result = summary.across_seeds([1., 2., 6.])
    assert result["sample_sd"] == statistics.stdev([1., 2., 6.])
    assert result["sample_sd"] != statistics.pstdev([1., 2., 6.])
    assert summary.across_seeds([1., None, 6.])["mean"] is None
    with pytest.raises(ValueError):
        summary.across_seeds([1., 2.])


@pytest.mark.parametrize("source", ["predicted", "ground_truth"])
@pytest.mark.parametrize("change", ["missing_field", "fraction_negative", "fraction_over_one", "threshold_exceeds_exact",
                                   "nonfinite_fraction", "negative_excursion", "mean_exceeds_maximum", "inconsistent_threshold",
                                   "coordinate_shape", "coordinate_nonfinite", "coordinate_reversed", "boolean_scalar"])
def test_boundary_schema_rejects_invalid_ranges_and_nonfinite_values(source, change):
    item = boundary_entry(4, 1, .001)
    if change == "missing_field":
        item.pop("coordinate_minimum")
    elif change == "fraction_negative":
        item["fraction_particles_outside"] = -.1
    elif change == "fraction_over_one":
        item["fraction_particles_outside"] = 1.1
    elif change == "threshold_exceeds_exact":
        item["fraction_particles_outside_by_more_than_1e-6"] = .5
    elif change == "nonfinite_fraction":
        item["fraction_particles_outside"] = float("nan")
    elif change == "negative_excursion":
        item["maximum_coordinate_excursion"] = -.001
    elif change == "mean_exceeds_maximum":
        item["mean_particle_maximum_excursion"] = .002
    elif change == "inconsistent_threshold":
        item["fraction_particles_outside_by_more_than_1e-6"] = 0.
    elif change == "coordinate_shape":
        item["coordinate_minimum"] = [0]
    elif change == "coordinate_nonfinite":
        item["coordinate_maximum"] = [float("inf"), 1]
    elif change == "coordinate_reversed":
        item["coordinate_minimum"] = [2, 0]
    else:
        item["mean_particle_maximum_excursion"] = True
    with pytest.raises(ValueError, match="boundary"):
        summary.summarize_boundary_series([item], 1, source, 10.)


def test_boundary_step_lengths_strict_threshold_and_prediction_guard():
    item = boundary_entry(4, 1, 1e-6)
    values = summary.summarize_boundary_series([item], 1, "predicted", 10.)
    assert values["predicted_boundary_mean_fraction_outside_gt_1e_minus6"] == 0.
    assert values["predicted_boundary_trajectory_maximum_excursion"] == 1e-6
    for series in ([], [item, item], {"0": item}):
        with pytest.raises(ValueError, match="array length"):
            summary.summarize_boundary_series(series, 1, "predicted", 10.)
    item["coordinate_maximum"] = [11, 1]
    with pytest.raises(ValueError, match="resource guard"):
        summary.summarize_boundary_series([item], 1, "predicted", 10.)
    # The prediction resource guard must not be invented as a truth-data bound.
    assert summary.summarize_boundary_series([item], 1, "ground_truth", 10.)[
        "ground_truth_boundary_mean_fraction_outside_gt_1e_minus6"] == 0.


def test_zero_step_failure_has_no_boundary_mean_or_invented_prefix_value():
    result = summary.boundary_diagnostics({"status": "failed", "completed_steps": 0,
        "predicted_boundary_per_step": [], "ground_truth_boundary_per_step": []}, 10.)
    assert all(value is None for value in result["full_horizon"].values())
    assert result["accepted_prefix_if_failed"]["completed_steps"] == 0
    assert all(value is None for value in result["accepted_prefix_if_failed"]["metrics"].values())
