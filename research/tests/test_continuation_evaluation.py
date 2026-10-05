"""Synthetic CPU endpoint gates, matched policies, failure and pairing tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research import continuation_evaluation as evaluate
from research.tests.test_faithful_graph_support import model, batch, optimizer, history
from research.tests.test_graph_convention_bridge import fixture, identity
from research.tests.test_native_graph_rollout import rollout_fixture


def observed():
    current, previous, types, target = fixture()
    return evaluate.evaluate_frame(model(), current, previous, types, target, identity(), 0, "cpu")


def test_observed_policies_match_native_prefix_suffix_and_exact_budgets():
    row, arrays = observed()
    assert row["status"] == "complete" and row["native_parity"]["passed"]
    assert tuple(row["cases"]) == evaluate.POLICIES
    assert row["previous_score"]["status"] == "complete"
    current, previous, _, _ = fixture()
    base = arrays["edges__base"]
    for policy in evaluate.POLICIES:
        actual = "laggedrisk25" if policy == "previous-observed-base-risk25" else policy
        score = arrays["previous_base_risk"] if actual == "laggedrisk25" else None
        graph, edge, optional, audit = evaluate.native.native_graph(current, actual, score,
            np.random.default_rng(np.random.SeedSequence(row["random_seed_material"])))
        np.testing.assert_array_equal(edge, arrays[f"edges__{policy}"])
        np.testing.assert_array_equal(edge[:, :base.shape[1]], base)
        np.testing.assert_array_equal(optional, arrays[f"optional_pairs__{policy}"])
        expected = 0 if policy == "base" else len(graph.extra) if policy == "dense" else row["optional_budget"]
        assert len(optional) == expected and edge.shape[1] == base.shape[1] + 2 * expected
        assert row["cases"][policy]["graph"] == audit
    _, old, _, _ = evaluate.native.native_graph(previous, "base", None, np.random.default_rng(0))
    np.testing.assert_array_equal(old, arrays["previous_base_edges"])
    assert row["cases"]["previous-observed-base-risk25"]["network_passes_for_standalone_policy"] == 2


def test_future_targets_and_arm_labels_cannot_change_graphs_predictions_or_rng():
    current, previous, types, target = fixture()
    a, first = evaluate.evaluate_frame(model(), current, previous, types, target, {**identity(), "arm": "base"}, 2, "cpu")
    b, second = evaluate.evaluate_frame(model(), current, previous, types, target + .05, {**identity(), "arm": "mix"}, 2, "cpu")
    assert a["random_seed_material"] == b["random_seed_material"] == [20261005, 771, 2, 1, 3, 7]
    for key in first:
        if key.startswith(("edges__", "optional_pairs__", "prediction__", "risk__", "selection_score__")):
            np.testing.assert_array_equal(first[key], second[key])
    assert a["target_sha256"] != b["target_sha256"]


def test_signed_actual_action_gains_and_boundary_truth_reconstruct_from_arrays():
    row, arrays = observed()
    for policy in evaluate.POLICIES:
        residual = arrays[f"prediction__{policy}"].astype(np.float64) - arrays["target_position"]
        normalized = residual / arrays["acceleration_std"]
        np.testing.assert_array_equal(residual, arrays[f"position_residual__{policy}"])
        np.testing.assert_array_equal(normalized, arrays[f"normalized_residual__{policy}"])
        gain = np.mean(arrays["normalized_residual__base"] ** 2, axis=-1) - np.mean(normalized ** 2, axis=-1)
        np.testing.assert_array_equal(gain, arrays[f"signed_normalized_coordinate_gain__{policy}"])
        assert row["benefits"][policy]["mean_signed_normalized_coordinate_gain"] == float(gain.mean())
        assert row["previous_risk_correlations"][policy] == evaluate.bridge.same.spearman(arrays["previous_base_risk"], gain)
        assert row["cases"][policy]["predicted_boundary"] == evaluate.full.boundary_metrics(arrays[f"prediction__{policy}"], arrays["bounds"])
    assert row["ground_truth_boundary"] == evaluate.full.boundary_metrics(arrays["target_position"], arrays["bounds"])


def test_previous_scoring_failure_retains_raw_score_and_other_policy_results(monkeypatch):
    original = evaluate.bridge.supplied
    calls = [0]
    def bad_score(*args, **kwargs):
        output = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:  # current native parity supplied, then previous scoring
            output["raw_risk"][0] = np.nan
            output["risk"][0] = np.nan
        return output
    monkeypatch.setattr(evaluate.bridge, "supplied", bad_score)
    row, arrays = observed()
    assert row["status"] == "failed" and row["previous_score"]["status"] == "failed"
    assert np.isnan(arrays["previous_base_raw_risk"][0])
    assert row["cases"]["previous-observed-base-risk25"]["failure"]["category"] == "scoring_pass_failed"
    assert row["benefits"]["previous-observed-base-risk25"]["mean_signed_normalized_coordinate_gain"] is None
    assert all(row["cases"][policy]["status"] == "complete" for policy in evaluate.POLICIES[:-1])
    assert "prediction__previous-observed-base-risk25" not in arrays


def test_failed_action_keeps_rejected_prediction_and_no_favorable_mean(monkeypatch):
    original = evaluate.bridge.supplied
    calls = [0]
    def bad_dense(*args, **kwargs):
        output = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 3:
            output["prediction"][0, 0] = 11.
        return output
    monkeypatch.setattr(evaluate.bridge, "supplied", bad_dense)
    row, arrays = observed()
    assert row["cases"]["dense"]["status"] == "failed" and arrays["prediction__dense"][0, 0] == 11.
    assert row["cases"]["dense"]["metrics"] is None
    assert row["benefits"]["dense"]["mean_signed_normalized_coordinate_gain"] is None
    assert row["cases"]["dense"]["predicted_boundary"] == evaluate.full.boundary_metrics(arrays["prediction__dense"], arrays["bounds"])
    assert arrays["prediction__dense"][0, 0] - arrays["bounds"][0, 1] > 10.
    assert row["cases"]["random25"]["status"] == "complete"


def test_native_parity_failure_keeps_raw_comparison_and_all_five_failed_cases():
    current, previous, types, target = fixture()
    m = model(); original = m.predict_positions_with_variance
    def altered(*args, **kwargs):
        prediction, risk = original(*args, **kwargs)
        return prediction + .01, risk
    m.predict_positions_with_variance = altered
    row, arrays = evaluate.evaluate_frame(m, current, previous, types, target, identity(), 0, "cpu")
    assert row["status"] == "failed" and row["failure"]["category"] == "native_parity_failure"
    assert len(row["cases"]) == 5 and all(case["status"] == "failed" for case in row["cases"].values())
    assert "native_prediction" in arrays and "supplied_parity_prediction" in arrays
    assert not np.array_equal(arrays["native_prediction"], arrays["supplied_parity_prediction"])


def test_initial_current_guard_preserves_history_and_five_failure_records():
    current, previous, types, target = fixture(); current[0, 0, 0] = np.nan
    row, arrays = evaluate.evaluate_frame(model(), current, previous, types, target, identity(), 0, "cpu")
    assert row["failure"]["phase"] == "current_history" and row["native_parity"] is None
    assert np.isnan(arrays["current_history"][0, 0, 0]) and len(row["cases"]) == 5


def test_cap_active_and_empty_optional_keep_native_asymmetry_and_zero_gain():
    current = np.full((6, 130, 2), .2, dtype=np.float32)
    types = np.ones(130, dtype=np.int64)
    row, arrays = evaluate.evaluate_frame(model(), current, current.copy(), types, current[-1], identity(), 0, "cpu")
    assert row["status"] == "complete" and row["optional_budget"] == 0
    assert row["cases"]["base"]["graph"]["native_base_receivers_above_cap_before_capping"] == 130
    assert row["cases"]["base"]["graph"]["native_base_self_edges"] == 128
    for policy in evaluate.POLICIES:
        np.testing.assert_array_equal(arrays[f"edges__{policy}"], arrays["edges__base"])
        assert row["benefits"][policy]["mean_signed_normalized_coordinate_gain"] == 0.


def trained_tiny():
    m, inputs = model(), batch(); opt = optimizer(m)
    evaluate.support.update_once(m, opt, inputs, torch.zeros_like(inputs[0]), "cpu", 0, 100000, "base")
    return m, opt


def test_adam_lineage_rejects_relabelled100k_parent_and_mutated_groups():
    m, opt = trained_tiny()
    for state in opt.state.values(): state["step"].fill_(100000)
    parent = {"optimizer_state": copy.deepcopy(opt.state_dict()), "state_dict": copy.deepcopy(m.state_dict())}
    before = evaluate.optimizer_signature(parent)
    child = copy.deepcopy(parent)
    for state in child["optimizer_state"]["state"].values(): state["step"].add_(10000)
    after = evaluate.optimizer_signature(child)
    evaluate.validate_optimizer_lineage(before, after)
    with pytest.raises(ValueError, match="Adam counters"):
        evaluate.validate_optimizer_lineage(before, before)
    after["groups"][0]["betas"] = (.8, .999)
    with pytest.raises(ValueError, match="Adam group"):
        evaluate.validate_optimizer_lineage(before, after)
    next(iter(child["optimizer_state"]["state"].values()))["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite endpoint Adam"):
        evaluate.optimizer_signature(child)


def endpoint_cohort(tmp_path, monkeypatch):
    m, opt = trained_tiny()
    simulator = copy.deepcopy(m._checkpoint_config)
    simulator.update(latent_dim=128, nmessage_passing_steps=10, nmlp_layers=2, mlp_hidden_dim=128)
    old_hashes, parents, configs, entries = {}, {}, {}, []
    for seed in (0, 1, 2):
        run = {"seed": seed, "objective": "faithful", "steps": 100000, "scope": "bounded_full_data_100k",
            "research_protocol_sha256": evaluate.full.sha256(evaluate.TRAINING_PROTOCOL), "metadata_sha256": "synthetic-metadata",
            "train": {"manifest_sha256": "train"}, "valid": {"manifest_sha256": "valid"},
            "validation_frames": [], "source_sha256": {}}
        parent = {"format_version": 2, "full_training_schema": 1, "completed_steps": 100000,
            "simulator_config": simulator, "state_dict": m.state_dict(), "optimizer_state": copy.deepcopy(opt.state_dict()),
            "training_config": {"loss": "faithful", "completed_optimizer_updates": 100000}, "run_config": run,
            "run_config_sha256": evaluate.support.original.config_hash(run)}
        for state in parent["optimizer_state"]["state"].values(): state["step"].fill_(100000)
        path = tmp_path / f"parent{seed}.pt"; torch.save(parent, path)
        old_hashes[seed], parents[seed] = evaluate.full.sha256(path), parent
    monkeypatch.setattr(evaluate.support, "PARENT_HASHES", old_hashes)
    for arm, seed in sorted(evaluate.COHORT):
        parent = parents[seed]
        cfg = {"objective": "faithful", "arm": arm, "seed": seed, "parent_checkpoint_sha256": old_hashes[seed],
            "parent_run_config_sha256": parent["run_config_sha256"], "parent_training_config": parent["training_config"],
            "parent_simulator_config_sha256": evaluate.support.simulator_config_sha256(simulator),
            "parent_completed_updates": 100000, "additional_updates": 10000, "lr": 1e-5, "batch_size": 2, "noise_std": 6.7e-4,
            "train": parent["run_config"]["train"], "valid": parent["run_config"]["valid"], "validation_frames": [], "graph": {},
            "input_files_sha256": {str(tmp_path / f"parent{seed}.pt"): old_hashes[seed],
                str(Path(evaluate.support.__file__).resolve()): evaluate.TRAINER_SHA256,
                str(evaluate.support.PROTOCOL.resolve()): evaluate.TRAINER_PROTOCOL_SHA256}}
        hist = history(10000)
        for row in hist["graph_updates"]:
            row.update(frame_ids=[f"synthetic:{row['absolute_schedule_step']}"], noise_sha256="synthetic-noise")
        payload = evaluate.support.checkpoint_payload(m, opt, cfg, 10000, hist, "cpu")
        payload["simulator_config"] = copy.deepcopy(simulator)
        for state in payload["optimizer_state"]["state"].values(): state["step"].fill_(110000)
        directory = tmp_path / f"{arm}_{seed}"; directory.mkdir()
        path = directory / "checkpoint-extra-10000.pt"; torch.save(payload, path)
        entry = {"arm": arm, "seed": seed, "checkpoint": str(path), "sha256": evaluate.full.sha256(path)}
        entries.append(entry); configs[arm, seed] = cfg
        latest = {"path": path.name, "sha256": entry["sha256"], "completed_additional_updates": 10000, "completed_total_updates": 110000}
        for name, value in (("latest.json", latest), ("protocol.json", cfg), ("status.json", {"state": "complete",
            "latest_checkpoint": latest, "completed_additional_updates": 10000, "completed_total_updates": 110000,
            "config_sha256": evaluate.support.original.config_hash(cfg)})):
            (directory / name).write_text(json.dumps(value))
    document = {"schema": 1, "scope": "faithful_graph_support_110k_endpoints", "endpoints": entries}
    manifest = tmp_path / "cohort.json"; manifest.write_text(json.dumps(document))
    return manifest, document, configs


def test_complete_synthetic_six_endpoint_cohort_validates_parent_only_old_gate(tmp_path, monkeypatch):
    manifest, document, configs = endpoint_cohort(tmp_path, monkeypatch)
    original = evaluate.full.check_checkpoint; checked = []
    def spy(payload, *args, **kwargs):
        assert "full_training_schema" in payload and "graph_support_schema" not in payload
        checked.append(payload["completed_steps"])
        return original(payload, *args, **kwargs)
    monkeypatch.setattr(evaluate.full, "check_checkpoint", spy)
    entries, loaded, pins = evaluate.load_verified_cohort(manifest, "synthetic-metadata")
    assert set(entries) == evaluate.COHORT and loaded == configs and checked == [100000] * 3
    assert str(manifest.resolve()) in pins and len(pins) >= 27


@pytest.mark.parametrize("mutation,match", [
    (lambda doc: doc["endpoints"].pop(), "Exactly base/mix"),
    (lambda doc: doc["endpoints"][0].update(arm="third"), "Exactly base/mix"),
    (lambda doc: doc["endpoints"][0].update(checkpoint=doc["endpoints"][1]["checkpoint"]), "distinct checkpoint"),
    (lambda doc: doc["endpoints"][0].update(sha256=doc["endpoints"][1]["sha256"]), "distinct checkpoint"),
])
def test_malformed_or_incomplete_cohort_refused(tmp_path, monkeypatch, mutation, match):
    _, document, _ = endpoint_cohort(tmp_path, monkeypatch); mutation(document)
    with pytest.raises(ValueError, match=match):
        evaluate.validate_cohort_manifest(document, tmp_path)


def test_endpoint_old_schema_incomplete_ledger_and_uncommitted_publication_refused(tmp_path, monkeypatch):
    _, document, configs = endpoint_cohort(tmp_path, monkeypatch)
    entry = document["endpoints"][0]; cfg = configs[entry["arm"], entry["seed"]]
    payload = torch.load(entry["checkpoint"], weights_only=True)
    evaluate.check_endpoint(payload, entry)
    changed = copy.deepcopy(payload); changed["full_training_schema"] = 1
    with pytest.raises(ValueError, match="Old full-training"):
        evaluate.check_endpoint(changed, entry)
    changed = copy.deepcopy(payload); changed["history"]["graph_updates"][-1]["absolute_schedule_step"] -= 1
    with pytest.raises(ValueError, match="absolute-step"):
        evaluate.check_endpoint(changed, entry)
    status_path = Path(entry["checkpoint"]).parent / "status.json"
    status = json.loads(status_path.read_text()); status["state"] = "running"; status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="publication must be complete"):
        evaluate.check_committed_endpoint(entry, cfg)


def test_cohort_endpoint_hash_mismatch_stops_before_any_parent_gate(tmp_path, monkeypatch):
    manifest, document, _ = endpoint_cohort(tmp_path, monkeypatch)
    Path(document["endpoints"][0]["checkpoint"]).write_bytes(b"modified endpoint")
    monkeypatch.setattr(evaluate, "validate_parent_identity", lambda *a: (_ for _ in ()).throw(AssertionError("parent checked too early")))
    with pytest.raises(ValueError, match="Endpoint checkpoint byte"):
        evaluate.load_verified_cohort(manifest, "synthetic-metadata")


@pytest.mark.parametrize("defect,match", [("relabelled_adam", "Adam counters"), ("changed_paired_noise", "Paired frame/noise")])
def test_rehashed_complete_fake_endpoint_still_fails_lineage(tmp_path, monkeypatch, defect, match):
    manifest, document, _ = endpoint_cohort(tmp_path, monkeypatch)
    entry = document["endpoints"][0]
    payload = torch.load(entry["checkpoint"], weights_only=True)
    if defect == "relabelled_adam":
        for state in payload["optimizer_state"]["state"].values(): state["step"].fill_(100000)
    else:
        payload["history"]["graph_updates"][123]["noise_sha256"] = "a different recorded noise realization"
    torch.save(payload, entry["checkpoint"])
    entry["sha256"] = evaluate.full.sha256(entry["checkpoint"])
    directory = Path(entry["checkpoint"]).parent
    latest = json.loads((directory / "latest.json").read_text()); latest["sha256"] = entry["sha256"]
    status = json.loads((directory / "status.json").read_text()); status["latest_checkpoint"] = latest
    (directory / "latest.json").write_text(json.dumps(latest)); (directory / "status.json").write_text(json.dumps(status))
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        evaluate.load_verified_cohort(manifest, "synthetic-metadata")


def publication(tmp_path):
    directory = tmp_path / "observed"; directory.mkdir()
    item = identity()
    args = SimpleNamespace(output_dir=directory, resume=False, device="cpu", saved_same_state_dir=tmp_path / "parent-input")
    protocol = {"arm": "mix", "seed": 1, "checkpoint_sha256": "new-endpoint", "parent_checkpoint_sha256": "old-parent",
        "scope": "synthetic", "expected_frames": [item], "input_files_sha256": {}}
    return args, protocol, item


def test_observed_publication_recovery_keeps_failed_arrays_and_rejects_hostile_orphan(tmp_path, monkeypatch):
    args, protocol, item = publication(tmp_path)
    monkeypatch.setattr(evaluate.bridge, "frame_input", lambda *a: fixture())
    real = evaluate.evaluate_frame; calls = []
    def fail_one(*arguments):
        row, arrays = real(*arguments); calls.append(1)
        row["status"], row["failure"] = "failed", {"category": "synthetic_case_failure"}
        return row, arrays
    monkeypatch.setattr(evaluate, "evaluate_frame", fail_one)
    evaluate.run_observed(args, protocol, model(), {}, [item], {})
    path = args.output_dir / (evaluate.bridge.stem(item) + ".json")
    before = path.read_bytes(); prior_status = (args.output_dir / "status.json").read_bytes(); args.resume = True
    evaluate.run_observed(args, protocol, model(), {}, [item], {})
    assert calls == [1] and path.read_bytes() == before
    archives = list((args.output_dir / "attempt_history").glob("*/status.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == prior_status
    result = json.loads((args.output_dir / "result.json").read_text())
    assert result["failed_frames"] == 1 and result["arm"] == "mix"
    (args.output_dir / "result.json").unlink()  # hostile orphan must still fail lineage gate
    row = json.loads(path.read_text()); row["arm"] = "base"; path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Recovered observed row lineage"):
        evaluate.run_observed(args, protocol, model(), {}, [item], {})
    assert calls == [1]


@pytest.mark.parametrize("name", ["latest.json.tmp", "prior/status.json.tmp", "protocol.json.tmp"])
def test_output_temp_preservation_before_any_observed_writes(tmp_path, name):
    args, protocol, item = publication(tmp_path)
    path = args.output_dir / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="Temporary output"):
        evaluate.run_observed(args, protocol, None, {}, [item], {})
    assert path.read_bytes() == b"preserve" and not (args.output_dir / "protocol.json").exists()


def test_native_autonomous_passthrough_has_original_seed_and_no_old_gate(tmp_path, monkeypatch):
    args = SimpleNamespace(output_dir=tmp_path, resume=False, device="cpu")
    protocol = {"arm": "mix", "seed": 2, "original_seed": 2, "checkpoint_sha256": "endpoint", "parent_checkpoint_sha256": "parent",
        "continuation_config_sha256": "config", "completed_total_updates": 110000, "completed_additional_updates": 10000,
        "source_indices": list(range(3, 30)), "horizon": 995, "policies": list(evaluate.native.POLICIES)}
    sent = []
    monkeypatch.setattr(evaluate.native, "run", lambda *a: sent.append(a))
    monkeypatch.setattr(evaluate.full, "check_checkpoint", lambda *a: (_ for _ in ()).throw(AssertionError("old gate called")))
    m, trajectories, manifest = object(), object(), object()
    evaluate.run_autonomous(args, protocol, m, trajectories, manifest, list(range(3, 30)))
    assert sent[0] == (args, protocol, m, trajectories, manifest, list(range(3, 30)))
    assert json.loads((tmp_path / "lineage_identity.json").read_text())["arm"] == "mix"
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "status.json").write_bytes(b"prior completed native status")
    args.resume = True
    evaluate.run_autonomous(args, protocol, m, trajectories, manifest, list(range(3, 30)))
    assert next((tmp_path / "attempt_history").glob("*/status.json")).read_bytes() == b"prior completed native status"
    with pytest.raises(ValueError, match="Every official source"):
        evaluate.run_autonomous(args, protocol, m, trajectories, manifest, [3])


def paired_fixture():
    units = [{"trajectory_id": "a", "unit_id": str(index)} for index in (1, 2)] + [{"trajectory_id": "b", "unit_id": "1"}]
    records = []
    for arm, seed in sorted(evaluate.COHORT):
        for unit in units:
            offset = (1 + seed) * (1 if unit["trajectory_id"] == "a" else 9) if arm == "mix" else 0
            values = {policy: 10. + offset + index for index, policy in enumerate(evaluate.POLICIES)}
            values["previous-observed-base-risk25"] += seed if arm == "mix" else 0
            records.append({"arm": arm, "seed": seed, **unit, "policy_values": values})
    return records, units


def test_paired_arm_contrasts_difference_before_equal_trajectory_seed_aggregation():
    records, units = paired_fixture()
    result = evaluate.paired_contrasts(records, units)
    base = result["mix_minus_base__base"]
    assert [row["difference"] for row in base["seeds"]] == [5., 10., 15.]  # not frame-weighted11/3
    assert base["mean"] == 10. and base["sample_sd"] == 5.
    interaction = result["risk_minus_random_interaction"]
    assert [row["difference"] for row in interaction["seeds"]] == [0., 1., 2.]
    assert interaction["mean"] == interaction["sample_sd"] == 1.


def test_missing_failed_or_duplicate_paired_units_cannot_make_favorable_subset_mean():
    records, units = paired_fixture()
    records[0]["policy_values"]["base"] = None
    result = evaluate.paired_contrasts(records, units)["mix_minus_base__base"]
    assert result["mean"] is None and result["sample_sd"] is None and result["complete_seed_count"] == 2
    assert result["seeds"][0]["missing_or_failed_units"] == 1
    records, units = paired_fixture(); records.pop()
    assert evaluate.paired_contrasts(records, units)["risk_minus_random_interaction"]["mean"] is None
    records, units = paired_fixture(); records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError, match="Unknown/duplicate"):
        evaluate.paired_contrasts(records, units)
