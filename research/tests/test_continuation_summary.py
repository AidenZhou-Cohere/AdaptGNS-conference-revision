"""Synthetic-only saved-result admission, numerical audit and estimand tests."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research import summarize_continuation_evaluation as summary
from research.tests import test_continuation_evaluation as fixture


def test_unequal_frame_counts_equal_weight_trajectories_and_preserve_all_differences():
    records, units = fixture.paired_fixture()
    result = summary.aggregate_metric(records, units)
    effect = result["paired"]["mix_minus_base__base"]
    assert effect["seed_values"] == [5., 10., 15.]
    assert effect["mean"] == 10 and effect["sample_sd"] == 5
    assert effect["seeds"][0]["trajectories"][0]["value"] == 1
    assert len(effect["seeds"][0]["trajectories"][0]["units"]) == 2
    interaction = result["paired"]["risk_minus_random_interaction"]
    assert interaction["seed_values"] == [0, 1, 2]
    assert interaction["mean"] == interaction["sample_sd"] == 1
    assert len(result["paired"]) == 12
    assert set(result["absolute"]) == {"base", "mix"}


def test_failure_nulls_propagate_without_erasing_valid_other_policies_or_seeds():
    records, units = fixture.paired_fixture()
    records[0]["policy_values"]["base"] = None
    records[0]["policy_null_reasons"] = {"base": "guard failed"}
    result = summary.aggregate_metric(records, units)
    absolute = result["absolute"]["base"]["base"]
    assert absolute["mean"] is absolute["sample_sd"] is None
    assert absolute["defined_seeds"] == 2
    assert absolute["seeds"][0]["null_units"] == 1
    assert absolute["seeds"][0]["trajectories"][0]["units"][0]["null_reason"] == "base/base: guard failed"
    assert result["absolute"]["base"]["random25"]["mean"] is not None
    assert result["paired"]["risk_minus_random_interaction"]["mean"] == 1
    assert result["paired"]["mix_minus_base__base"]["mean"] is None


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(arm="mixed"),
    lambda rows: rows[0].update(seed=True),
    lambda rows: rows[0].update(unit_id="unknown"),
    lambda rows: rows[0]["policy_values"].update(base=True),
    lambda rows: rows[0]["policy_values"].update(base=float("nan")),
    lambda rows: rows[0]["policy_values"].pop("base"),
])
def test_bad_paired_id_value_or_policy_rejected(mutation):
    records, units = fixture.paired_fixture(); mutation(records)
    with pytest.raises(ValueError):
        summary.aggregate_metric(records, units)


def test_missing_unit_preserved_as_null_in_pure_hierarchy():
    records, units = fixture.paired_fixture(); records.pop()
    result = summary.aggregate_metric(records, units)["paired"]["risk_minus_random_interaction"]
    assert result["mean"] is None and result["defined_seeds"] == 2
    assert result["seeds"][2]["defined_units"] == 2 and result["seeds"][2]["required_units"] == 3
    assert "missing committed unit" in result["seeds"][2]["trajectories"][1]["units"][0]["null_reason"]


def test_observed_arrays_recompute_all_metrics_and_constant_base_correlation():
    row, arrays = fixture.observed()
    audit = summary.Audit()
    result = summary.analyze_observed_frame(row, arrays, audit)
    assert audit.checks > 100
    for policy in summary.evaluation.POLICIES:
        residual = arrays["prediction__" + policy].astype(np.float64) - arrays["target_position"]
        assert result["metrics"][policy]["position_coordinate_mse"] == np.mean(residual ** 2)
        assert result["metrics"][policy]["normalized_coordinate_mse"] == np.mean((residual / arrays["acceleration_std"]) ** 2)
        assert result["metrics"][policy]["failure_fraction"] == 0
    assert result["metrics"]["base"]["mean_signed_normalized_coordinate_gain"] == 0
    assert result["metrics"]["base"]["previous_risk_gain_spearman"] is None
    assert result["null_reasons"]["base"]["previous_risk_gain_spearman"] == "constant_rank_vector"


@pytest.mark.parametrize("kind", ["mse", "gain", "risk_correlation", "boundary", "random_mask", "prefix", "target_hash", "timing", "reuse"])
def test_observed_saved_scalar_graph_or_semantic_tamper_refused(kind):
    row, arrays = fixture.observed()
    if kind == "mse": row["cases"]["dense"]["metrics"]["position_coordinate_mse"] += .001
    elif kind == "gain": arrays["signed_normalized_coordinate_gain__dense"][0] += .1
    elif kind == "risk_correlation": row["previous_risk_base_residual_correlation"]["value"] = .333
    elif kind == "boundary": row["cases"]["dense"]["predicted_boundary"]["maximum_coordinate_excursion"] += 1
    elif kind == "random_mask": arrays["optional_pairs__random25"] = np.array([[100, 101]])
    elif kind == "prefix": arrays["edges__base"] = arrays["edges__base"][:, ::-1].copy()
    elif kind == "target_hash": row["target_sha256"] = "0" * 64
    elif kind == "timing": row["cases"]["dense"]["forward_operational_seconds"] = -1
    elif kind == "reuse": row["cases"]["base"]["reused_native_parity_output"] = False
    with pytest.raises(ValueError):
        summary.analyze_observed_frame(row, arrays, summary.Audit())


def bad_scoring(monkeypatch):
    original = fixture.evaluate.bridge.supplied
    calls = [0]
    def replaced(*args, **kwargs):
        output = original(*args, **kwargs); calls[0] += 1
        if calls[0] == 2:
            output["raw_risk"][0] = output["risk"][0] = np.nan
        return output
    monkeypatch.setattr(fixture.evaluate.bridge, "supplied", replaced)
    return fixture.observed()


def test_isolated_previous_scoring_failure_preserves_other_policy_results(monkeypatch):
    row, arrays = bad_scoring(monkeypatch)
    result = summary.analyze_observed_frame(row, arrays, summary.Audit())
    risk = summary.evaluation.POLICIES[-1]
    assert result["metrics"][risk]["failure_fraction"] == 1
    assert result["metrics"][risk]["position_coordinate_mse"] is None
    assert result["metrics"][risk]["network_passes_for_standalone_policy"] == 1
    for policy in summary.evaluation.POLICIES[:-1]:
        assert result["metrics"][policy]["position_coordinate_mse"] is not None
        assert result["metrics"][policy]["previous_risk_gain_spearman"] is None
    assert np.isnan(result["diagnostics"]["previous_score"]["failure"].get("unused", np.nan))
    arrays["previous_base_raw_risk"][0] = arrays["previous_base_risk"][0] = 1
    with pytest.raises(ValueError, match="not reproduced"):
        summary.analyze_observed_frame(row, arrays, summary.Audit())


def test_previous_history_guard_has_no_scoring_output_and_keeps_other_policies():
    current, previous, types, target = fixture.fixture()
    previous[0, 0, 0] = np.nan
    row, arrays = fixture.evaluate.evaluate_frame(fixture.model(), current, previous, types, target, fixture.identity(), 0, "cpu")
    assert row["previous_score"]["failure"]["phase"] == "previous_history"
    assert "previous_base_prediction" not in arrays
    result = summary.analyze_observed_frame(row, arrays, summary.Audit())
    risk = summary.evaluation.POLICIES[-1]
    assert result["metrics"][risk]["failure_fraction"] == 1
    assert result["metrics"][risk]["network_passes_for_standalone_policy"] == 0
    for policy in summary.evaluation.POLICIES[:-1]:
        assert result["metrics"][policy]["failure_fraction"] == 0
        assert result["metrics"][policy]["position_coordinate_mse"] is not None
        assert result["metrics"][policy]["previous_risk_gain_spearman"] is None


def test_failed_action_retains_boundary_and_measured_duration(monkeypatch):
    original = fixture.evaluate.bridge.supplied
    calls = [0]
    def replaced(*args, **kwargs):
        output = original(*args, **kwargs); calls[0] += 1
        if calls[0] == 3: output["prediction"][0, 0] = 11
        return output
    monkeypatch.setattr(fixture.evaluate.bridge, "supplied", replaced)
    row, arrays = fixture.observed()
    result = summary.analyze_observed_frame(row, arrays, summary.Audit())
    metrics = result["metrics"]["dense"]
    assert metrics["position_coordinate_mse"] is None
    assert metrics["failure_fraction"] == 1
    assert metrics["predicted_boundary_maximum_coordinate_excursion"] > 10
    assert metrics["forward_operational_seconds"] > 0


def test_early_current_history_and_parity_failures_keep_no_fabricated_outputs():
    current, previous, types, target = fixture.fixture()
    current[0, 0, 0] = np.nan
    row, arrays = fixture.evaluate.evaluate_frame(fixture.model(), current, previous, types, target, fixture.identity(), 0, "cpu")
    result = summary.analyze_observed_frame(row, arrays, summary.Audit())
    assert "target_position" not in arrays
    assert all(result["metrics"][p]["failure_fraction"] == 1 for p in summary.evaluation.POLICIES)
    current, previous, types, target = fixture.fixture()
    model = fixture.model(); original = model.predict_positions_with_variance
    def changed(*args, **kwargs):
        prediction, risk = original(*args, **kwargs)
        return prediction + .01, risk
    model.predict_positions_with_variance = changed
    row, arrays = fixture.evaluate.evaluate_frame(model, current, previous, types, target, fixture.identity(), 0, "cpu")
    result = summary.analyze_observed_frame(row, arrays, summary.Audit())
    assert result["failure"]["category"] == "native_parity_failure"
    assert result["metrics"]["base"]["position_coordinate_mse"] is None


def test_pairing_graph_change_refused_but_learned_risk_change_allowed():
    row, arrays = fixture.observed()
    left = summary.analyze_observed_frame(row, arrays, summary.Audit())
    left.update(arm="base", seed=0)
    right = copy.deepcopy(left); right["arm"] = "mix"
    right["policy_graph_hashes"][summary.evaluation.POLICIES[-1]] = "different learned score graph"
    result = summary.audit_pairing([copy.deepcopy(left), copy.deepcopy(right)], [], summary.Audit())
    assert len(result) == 1
    right["policy_graph_hashes"]["random25"] = "wrong shared random mask"
    with pytest.raises(ValueError, match="graph differs"):
        summary.audit_pairing([left, right], [], summary.Audit())


def test_autonomous_early_horizon_and_failure_inclusive_timing_preserved():
    row = {"status": "failed", "mse_at_steps": {str(s): .5 if s <= 200 else None for s in summary.scalar_audit.TRACE_STEPS},
        "completed_steps": 240, "forecast_network_passes": 241, "mean_rollout_mse": None, "mean_directed_edges": None,
        "total_wall_seconds": 12.5, "total_network_passes": 242,
        "boundary_diagnostics": {"full_horizon": {name: None for name in summary.scalar_audit.BOUNDARY_METRICS}},
        "native_metrics": {name: None for name in summary.native_audit.EXTRA_METRICS}}
    row["native_metrics"]["mean_wall_seconds_including_native_parity"] = 12.7
    values = summary.autonomous_values(row)
    assert values["mean_rollout_mse"] is values["mse_at_995"] is None
    assert values["mse_at_200"] == .5 and values["mse_at_1"] == .5
    assert values["failure_fraction"] == 1
    assert values["mean_rollout_wall_seconds"] == 12.5
    assert values["mean_wall_seconds_including_native_parity"] == 12.7
    assert values["forecast_network_passes"] == 241


def test_empty_overlap_remains_undefined_with_reason():
    empty = np.empty((0, 2), dtype=np.int64)
    assert summary.overlap(empty, empty) == {"intersection_pairs": 0, "union_pairs": 0, "jaccard": None, "reason": "empty_optional_union"}
    assert summary.overlap(None, empty)["reason"] == "required selection unavailable"


@pytest.mark.parametrize("text", ['{"state":"complete","state":"running"}', '{"value":NaN}', '{"value":Infinity}'])
def test_noncanonical_json_refused(tmp_path, text):
    path = tmp_path / "bad.json"; path.write_text(text)
    with pytest.raises(ValueError): summary.read_json(path)


def test_exact_inventory_and_archived_attempts_preserved(tmp_path):
    (tmp_path / "result.json").write_text('{}')
    prior = tmp_path / "attempt_history" / "old"; prior.mkdir(parents=True)
    (prior / "status.json").write_text('{"state":"interrupted"}')
    (prior / "archive.json").write_text(json.dumps({"source": "status.json", "sha256": summary.full.sha256(prior / "status.json")}))
    pins = summary.inventory(tmp_path, {"result.json"})
    assert len(pins) == 3
    (tmp_path / "orphan.npz").write_bytes(b"keep")
    with pytest.raises(ValueError, match="orphan"):
        summary.inventory(tmp_path, {"result.json"})
    assert (tmp_path / "orphan.npz").read_bytes() == b"keep"


@pytest.mark.parametrize("name", ["run.lock", "frame.npz.tmp", "nested/file.json", "attempt_history/old/unknown.json"])
def test_unknown_live_temporary_or_nested_artifacts_refused(tmp_path, name):
    path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"preserve")
    with pytest.raises(ValueError): summary.inventory(tmp_path, set())
    assert path.read_bytes() == b"preserve"


def test_discovery_uses_protocol_identity_not_directory_name_and_rejects_duplicates(tmp_path):
    for index, (arm, seed) in enumerate(sorted(summary.evaluation.COHORT)):
        job = tmp_path / f"uninformative{index}"; job.mkdir()
        (job / "protocol.json").write_text(json.dumps({"mode": "observed", "arm": arm, "seed": seed}))
    jobs = summary.discover_jobs(tmp_path, "observed")
    assert set(jobs) == summary.evaluation.COHORT
    extra = tmp_path / "extra"; extra.mkdir()
    (extra / "protocol.json").write_text(json.dumps({"mode": "observed", "arm": "base", "seed": 0}))
    with pytest.raises(ValueError, match="duplicate"):
        summary.discover_jobs(tmp_path, "observed")


def test_committed_job_requires_exact_arm_bound_lineage_and_result_hash(tmp_path):
    protocol = {"scope": summary.EVALUATION_SCOPE, "objective": "faithful", "arm": "mix", "seed": 1,
        "original_seed": 1, "checkpoint_sha256": "endpoint", "parent_checkpoint_sha256": "parent",
        "continuation_config_sha256": "config", "completed_total_updates": 110000, "completed_additional_updates": 10000}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    result = {**protocol, "state": "complete", "protocol_sha256": summary.full.sha256(tmp_path / "protocol.json"), "invocation_wall_seconds": 17}
    (tmp_path / "result.json").write_text(json.dumps(result))
    status = {"state": "complete", "result_sha256": summary.full.sha256(tmp_path / "result.json")}
    (tmp_path / "status.json").write_text(json.dumps(status))
    identity = {key: protocol[key] for key in summary.IDENTITY_FIELDS}
    (tmp_path / "lineage_identity.json").write_text(json.dumps(identity))
    summary.committed_job(tmp_path, "autonomous", summary.Audit())
    identity["arm"] = "base"; (tmp_path / "lineage_identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="lineage identity"):
        summary.committed_job(tmp_path, "autonomous", summary.Audit())
    identity["arm"] = "mix"; (tmp_path / "lineage_identity.json").write_text(json.dumps(identity))
    status["state"] = "running"; (tmp_path / "status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="uncommitted"):
        summary.committed_job(tmp_path, "autonomous", summary.Audit())
    status["state"] = "complete"; (tmp_path / "status.json").write_text(json.dumps(status))
    (tmp_path / "result.json").write_text(json.dumps(result) + " ")
    with pytest.raises(ValueError, match="result hash"):
        summary.committed_job(tmp_path, "autonomous", summary.Audit())


def test_lossless_detail_publication_is_fresh_and_byte_pinned(tmp_path):
    value = {"failed": {"value": None, "reason": "nonfinite model output"}, "negative_effect": -1.23}
    path = tmp_path / "details.json.gz"
    descriptor = summary.write_compressed(path, value)
    assert json.loads(gzip.decompress(path.read_bytes())) == value
    assert summary.full.sha256(path) == descriptor["sha256"]
    with pytest.raises(FileExistsError): summary.write_compressed(path, value)


def test_frozen_endpoint_gate_retains_rejection_of_relabelled_parent_and_wrong_schedule(tmp_path, monkeypatch):
    # These fixture checkpoints are tiny CPU-generated synthetic models only.
    manifest, document, configs = fixture.endpoint_cohort(tmp_path, monkeypatch)
    entries, loaded, pins = summary.evaluation.load_verified_cohort(manifest, "synthetic-metadata")
    assert loaded == configs and len(entries) == 6
    entry = document["endpoints"][0]
    path = Path(entry["checkpoint"])
    payload = summary.torch.load(path, weights_only=True)
    payload["history"]["graph_updates"][0]["noise_sha256"] = "unpaired-noise"
    summary.torch.save(payload, path)
    entry["sha256"] = summary.full.sha256(path)
    latest = summary.read_json(path.parent / "latest.json"); latest["sha256"] = entry["sha256"]
    status = summary.read_json(path.parent / "status.json"); status["latest_checkpoint"] = latest
    (path.parent / "latest.json").write_text(json.dumps(latest)); (path.parent / "status.json").write_text(json.dumps(status))
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Paired frame/noise"):
        summary.evaluation.load_verified_cohort(manifest, "synthetic-metadata")


def protocol_fixture(tmp_path):
    entries = {(arm, seed): {"arm": arm, "seed": seed, "checkpoint": str(tmp_path / f"{arm}{seed}.pt"),
                           "sha256": f"{i+1:064x}"} for i, (arm, seed) in enumerate(sorted(summary.evaluation.COHORT))}
    entry = entries["mix", 1]
    config = {"parent_checkpoint_sha256": "p" * 64, "parent_training_config": {"loss": "faithful"}}
    manifest = {"records": [{"id": f"traj{i}"} for i in range(30)], "source": {"sha256": "synthetic-source"}}
    paths = [Path(summary.evaluation.__file__), summary.evaluation.PROTOCOL, Path(summary.native.__file__),
        Path(summary.bridge.__file__), Path(summary.bridge.same.__file__), Path(summary.full.__file__),
        summary.native.PROTOCOL, summary.bridge.PROTOCOL, summary.evaluation.TRAINING_PROTOCOL,
        summary.full.REPO / "research/budget_graph.py"]
    runtime = {"device": "mps", "radius_backend": "scipy_host", "graph_execution": "cpu_scipy_with_device_transfer",
               "mps_fallback_environment": "0", "torch_version": "synthetic-torch"}
    protocol = {"schema": 1, "scope": summary.EVALUATION_SCOPE, "mode": "autonomous", "arm": "mix", "seed": 1, "original_seed": 1,
        "objective": "faithful", "checkpoint_sha256": entry["sha256"], "parent_checkpoint_sha256": config["parent_checkpoint_sha256"],
        "continuation_config_sha256": summary.evaluation.support.original.config_hash(config), "cohort": list(entries.values()),
        "completed_total_updates": 110000, "completed_additional_updates": 10000, "expected_frames": None,
        "source_indices": list(range(3, 30)), "trajectory_ids": [r["id"] for r in manifest["records"][3:]],
        "source_tfrecord": manifest["source"], "horizon": 995, "policies": list(summary.native.POLICIES),
        "graph": "unchanged strict-r native capped128+self directed prefix; uncapped symmetric optional annulus suffix; r=.015,R=1.267r",
        "observed_risk": "previous observed six-frame native-base scoring; fresh every observed frame; two standalone network passes",
        "autonomous_risk": "one initial native-base warmup; forecast1 exact budget; cache from previous own selected graph and predicted history",
        "random_observed": "[20261005,771,original_seed,split_code,source_index,target_frame]; no arm identity",
        "random_autonomous": "93000+1000*original_seed+source_index; no arm identity; later geometry may differ",
        "parent_archive_usage": "only byte-verified histories, target and types; never endpoint output attribution",
        "native_parity": "two separate verification calls per frame/trajectory; raw outputs retained; parity failure halts model job",
        "time_scope": "operational synchronized calls with fixed order/shared preprocessing; not policy-speed or speedup evidence",
        "guards": {"max_candidate_pairs": 100000, "max_abs_coordinate": 10.}, "deadline_utc": summary.native.DEADLINE,
        "threads": 2, "runtime": runtime, "software": {"torch": "synthetic-torch"}, "saved_parent_test_input_archive": None,
        "input_files_sha256": {str(path.resolve()): summary.full.sha256(path) for path in paths}}
    protocol["checkpoint_provenance"] = {"checkpoint_format": 2, "normalization_source": "checkpoint", "checkpoint_radius_backend": "scipy_host",
        "connectivity_radius": .015, "nmessage_passing_steps": 10, "uncertainty_parameterization": "variance", "variance_floor": 1e-6,
        "max_num_neighbors": 128, "radius_backend": "scipy_host", "edge_convention": "directed_with_self_loops", "runtime": runtime,
        "training_config": {"loss": "faithful", "graph_support_continuation": config,
            "parent_training_config": config["parent_training_config"], "completed_optimizer_updates": 110000}}
    return protocol, entry, config, entries, manifest


def test_110k_protocol_admits_explicit_lineage_without_old100k_validation(tmp_path):
    protocol, entry, config, entries, manifest = protocol_fixture(tmp_path)
    contract = summary.validate_job_protocol(protocol, "autonomous", entry, config, entries, [], manifest, {}, summary.Audit())
    assert len(contract) == 64


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(arm="base"), lambda p: p.update(original_seed=True),
    lambda p: p.update(completed_total_updates=100000), lambda p: p.update(completed_total_updates=110000.),
    lambda p: p.update(completed_additional_updates=9999), lambda p: p.update(checkpoint_sha256="parent"),
    lambda p: p.update(parent_checkpoint_sha256="other-parent"), lambda p: p.update(continuation_config_sha256="wrong-config"),
    lambda p: p["cohort"][0].update(arm="mix"), lambda p: p.update(graph="uncapped no-loop original100k"),
    lambda p: p["runtime"].update(mps_fallback_environment="1"),
    lambda p: p["input_files_sha256"].update({str(Path(summary.evaluation.__file__).resolve()): "0" * 64}),
])
def test_wrong_endpoint_parent_arm_source_or_runtime_protocol_is_refused(tmp_path, mutation):
    protocol, entry, config, entries, manifest = protocol_fixture(tmp_path)
    protocol = copy.deepcopy(protocol); mutation(protocol)
    with pytest.raises(ValueError):
        summary.validate_job_protocol(protocol, "autonomous", entry, config, entries, [], manifest, {}, summary.Audit())


def test_native_loader_uses_only_structural_scalar_view_and_frozen_low_level_checks(tmp_path, monkeypatch):
    # Dispatch test only. Numerical native/scalar validators have their own
    # existing synthetic suites; a fake100k admission call would fail here.
    manifest = {"records": [{"id": f"t{i}"} for i in range(30)]}
    protocol = {"seed": 2, "arm": "mix", "guards": {"max_abs_coordinate": 10.}}
    positions = np.zeros((1001, 2, 2), np.float32)
    types = np.ones(2, np.int64)
    bounds = np.asarray([[0, 1], [0, 1]], dtype=np.float64)
    for name in ("protocol.json", "result.json", "status.json", "lineage_identity.json"):
        (tmp_path / name).write_text('{}')
    compacts = []
    for index in range(3, 30):
        for policy in summary.native.POLICIES:
            stem = f"trajectory_{index:06d}_{policy}"
            row = {"source_index": index, "policy": policy, "status": "complete", "native_parity": {"passed": True}}
            (tmp_path / (stem + ".json")).write_text(json.dumps(row))
            np.savez(tmp_path / (stem + ".npz"), bounds=bounds, initial_observed_positions=positions[:6])
            compacts.append({**row, "record_file": stem + ".json", "trace_file": stem + ".npz",
                "record_sha256": summary.full.sha256(tmp_path / (stem + ".json")),
                "trace_sha256": summary.full.sha256(tmp_path / (stem + ".npz"))})
    calls = []
    def scalar(directory, compact, view, digest):
        assert view == {"trajectory_records": manifest["records"][3:], "seed": 2, "guards": {"max_absolute_coordinate": 10.}}
        assert digest == "new110k-protocol"
        calls.append(1)
        return dict(compact)
    def forbidden(*args, **kwargs): raise AssertionError("Original100k admission invoked")
    monkeypatch.setattr(summary.scalar_audit, "validate_record", scalar)
    monkeypatch.setattr(summary.scalar_audit, "validate_protocol", forbidden)
    monkeypatch.setattr(summary.native_audit, "load_native", forbidden)
    monkeypatch.setattr(summary.native_audit, "checkpoint_validation_view", forbidden)
    monkeypatch.setattr(summary.native_audit, "validate_trajectory", lambda *a: ({}, None))
    monkeypatch.setattr(summary, "autonomous_values", lambda row: {"failure_fraction": 0.})
    rows, pins = summary.load_autonomous(tmp_path, protocol, {"records": compacts}, "new110k-protocol",
        {i: (positions, types) for i in range(3, 30)}, manifest, {"bounds": bounds}, summary.Audit())
    assert len(calls) == len(rows) == 135 and len(pins) == 274


def test_rejected_endpoint_output_isolation_creates_no_directory_or_status(tmp_path, monkeypatch):
    manifest = tmp_path / "control" / "cohort.json"
    manifest.parent.mkdir()
    entries = [{"arm": arm, "seed": seed, "checkpoint": str(tmp_path / "endpoints" / f"{arm}{seed}" / "checkpoint-extra-10000.pt"),
                "sha256": f"{i+1:064x}"} for i, (arm, seed) in enumerate(sorted(summary.evaluation.COHORT))]
    manifest.write_text(json.dumps({"schema": 1, "scope": "faithful_graph_support_110k_endpoints", "endpoints": entries}))
    endpoint = Path(entries[0]["checkpoint"]).parent; endpoint.mkdir(parents=True)
    output = endpoint / "forbidden-analysis"
    args = SimpleNamespace(output_dir=output, cohort_manifest=manifest, observed_root=tmp_path / "observed",
        autonomous_root=tmp_path / "autonomous", validation_manifest=tmp_path / "data" / "valid.json", test_manifest=tmp_path / "data" / "test.json")
    def forbidden(*args): raise AssertionError("Checkpoint payload gate should not run before destination rejection")
    monkeypatch.setattr(summary.evaluation, "load_verified_cohort", forbidden)
    with pytest.raises(ValueError, match="endpoint checkpoint tree"):
        summary.analyze(args)
    assert not output.exists()
    assert list(endpoint.iterdir()) == []
