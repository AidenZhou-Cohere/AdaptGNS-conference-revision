"""Synthetic arithmetic, preservation and control-flow checks; no trained models."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research import physical_allocation_pilot as runner


def required_ids():
    return [f"position_{trajectory}:{step}" for trajectory in range(3) for step in range(7, 19)]


def rows(values=(1.0, 2.0, 6.0)):
    return [{"id": name, "trajectory": name.split(":")[0], "n_particles": 1 if index < 12 else 1000,
             "policies": {policy: {"mse_normalized_acceleration": values[index // 12], "directed_edges": 20}
                          for policy in runner.POLICIES}}
            for index, name in enumerate(required_ids())]


def test_equal_frame_then_trajectory_means_and_missing_frame_nulls():
    records = rows()
    complete = runner.aggregate_frames(records, required_ids())
    assert complete["equal_trajectory_mean"]["policies"]["base"]["mse_normalized_acceleration"] == 3.0
    missing = runner.aggregate_frames(records[:-1], required_ids())
    assert missing["completed_frames"] == 35
    assert missing["trajectories"]["position_0"]["policies"]["base"]["mse_normalized_acceleration"] == 1.0
    assert missing["equal_trajectory_mean"]["policies"]["base"]["mse_normalized_acceleration"] is None
    assert missing["trajectories"]["position_2"]["completed_frames"] == 11


@pytest.mark.parametrize("defect", ["duplicate", "wrong_trajectory", "missing_policy", "nonfinite"])
def test_corrupt_completed_frames_are_not_treated_as_missing(defect):
    records = rows()
    if defect == "duplicate":
        records[-1]["id"] = records[0]["id"]
    elif defect == "wrong_trajectory":
        records[0]["trajectory"] = "position_1"
    elif defect == "missing_policy":
        del records[0]["policies"]["dense"]
    else:
        records[0]["policies"]["base"]["mse_normalized_acceleration"] = np.nan
    with pytest.raises(ValueError):
        runner.aggregate_frames(records, required_ids())


def complete_runs():
    result = {}
    for objective in runner.OBJECTIVES:
        for seed in runner.SEEDS:
            records = rows((1.0 + seed,) * 3)
            for row in records:
                row["policies"]["inverse_count25"]["mse_normalized_acceleration"] *= 0.9
            result[f"{objective}_seed{seed}"] = {"splits": {split: runner.aggregate_frames(records, required_ids())
                                                             for split in runner.SPLITS}}
    return result


def test_paired_seed_differences_percentages_and_sample_sd_require_five_seeds():
    runs = complete_runs()
    complete = runner.summarize(runs)["faithful/valid"]
    difference = complete["physical_minus_control"]["inverse_count25"]["base"]
    assert difference["difference"]["seed_values"] == pytest.approx([-0.1, -0.2, -0.3, -0.4, -0.5])
    assert difference["difference"]["mean"] == pytest.approx(-0.3)
    assert difference["difference"]["sample_seed_sd"] == pytest.approx(np.sqrt(0.025))
    assert difference["percent_difference"]["seed_values"] == pytest.approx([-10] * 5)
    assert difference["seeds_with_lower_mse"] == 5
    del runs["faithful_seed4"]
    missing = runner.summarize(runs)["faithful/valid"]["physical_minus_control"]["inverse_count25"]["base"]
    assert missing["difference"]["seed_values"][-1] is None
    assert missing["difference"]["mean"] is missing["difference"]["sample_seed_sd"] is None
    assert missing["percent_difference"]["mean"] is None
    assert missing["seeds_with_lower_mse"] is None


def test_zero_control_error_keeps_difference_but_undefined_percent():
    runs = complete_runs()
    for seed in runner.SEEDS:
        for split in runner.SPLITS:
            runs[f"faithful_seed{seed}"]["splits"][split]["equal_trajectory_mean"]["policies"]["base"]["mse_normalized_acceleration"] = 0.0
    value = runner.summarize(runs)["faithful/valid"]["physical_minus_control"]["inverse_count25"]["base"]
    assert value["difference"]["mean"] is not None
    assert value["percent_difference"]["seed_values"] == [None] * 5
    assert value["percent_difference"]["mean"] is None


@pytest.fixture
def synthetic_frame(monkeypatch):
    n = 6
    graph = runner.budget_graph.Candidates(np.array([[0, 1], [1, 2], [3, 4]]),
        np.array([[0, 2], [0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3]]), n)
    previous = runner.budget_graph.Candidates(np.array([[0, 5], [2, 4]]), np.empty((0, 2), dtype=int), n)
    position = np.arange(12, dtype=np.float32).reshape(6, 2) / 100
    prev_position = position - np.array([[0, 0], [.001, 0], [0, .002], [.003, .003], [0, 0], [.001, -.001]], dtype=np.float32)
    frame = {"id": "position_0:7", "trajectory": "position_0", "step": 7, "graph": graph,
        "prev_graph": previous, "position": position, "prev_position": prev_position,
        "speed": np.linalg.norm(position - prev_position, axis=1), "target": torch.zeros(n, 2),
        "features": torch.zeros(n, 14), "prev_features": torch.ones(n, 14)}
    calls = []

    def forward(_model, selected_frame, pairs, faithful, previous=False):
        calls.append({"previous": previous, "pairs": pairs.copy()})
        degree = np.bincount(pairs.reshape(-1), minlength=n)
        mean = torch.tensor(np.column_stack((degree / 10, np.arange(n) / 10)), dtype=torch.float32)
        risk = torch.tensor(np.arange(n, 0, -1) if previous else np.arange(1, n + 1), dtype=torch.float32)
        return mean, risk

    monkeypatch.setattr(runner.pilot, "run_forward", forward)
    # Reuse the original control evaluator with a deterministic synthetic forward.
    model = SimpleNamespace(eval=lambda: None)
    control = runner.pilot.evaluate(model, [frame], "faithful", 0)[0]
    calls.clear()
    return SimpleNamespace(frame=frame, control=control, model=model, calls=calls, forward=forward)


def test_synthetic_eight_policy_replay_exact_budgets_base_reuse_and_previous_graph(synthetic_frame):
    fixture = synthetic_frame
    row, raw = runner.evaluate_frame(fixture.model, fixture.frame, fixture.control, "faithful", np.random.default_rng(91300))
    assert set(row["policies"]) == set(runner.POLICIES)
    assert set(row["control_replay"]) == set(runner.CONTROLS)
    assert row["extra_pair_budget"] == 2
    assert row["isolated_particles"] == 1
    assert len(fixture.calls) == 9
    assert fixture.calls[1]["previous"]
    np.testing.assert_array_equal(fixture.calls[1]["pairs"], fixture.frame["prev_graph"].base)
    assert sum(not call["previous"] and np.array_equal(call["pairs"], fixture.frame["graph"].base)
               for call in fixture.calls) == 1
    for policy in runner.POLICIES:
        assert raw[f"{policy}_vector_se"].shape == (6,)
        assert row["policies"][policy]["mse_normalized_acceleration"] == pytest.approx(raw[f"{policy}_vector_se"].mean() / 2)
        assert len(raw[f"{policy}_selected_extra_pairs"]) == (0 if policy == "base" else 8 if policy == "dense" else 2)
    assert row["physical_control_overlap"]["inverse_count25"]["dense"]["fraction_of_physical"] == 1.0
    assert row["physical_control_overlap"]["inverse_count25"]["base"]["shared_optional_pairs"] == 0


def test_allocation_scores_and_selected_pairs_do_not_look_at_target(synthetic_frame):
    fixture = synthetic_frame
    _, original = runner.evaluate_frame(fixture.model, fixture.frame, fixture.control, "faithful", np.random.default_rng(91300))
    changed = dict(fixture.frame, target=fixture.frame["target"] + 100.0)
    changed_control = runner.pilot.evaluate(fixture.model, [changed], "faithful", 0)[0]
    _, altered = runner.evaluate_frame(fixture.model, changed, changed_control, "faithful", np.random.default_rng(91300))
    for key in ("negative_neighbor_count", "velocity_dispersion", "current_risk", "previous_base_risk"):
        np.testing.assert_array_equal(original[key], altered[key])
    for policy in runner.POLICIES:
        np.testing.assert_array_equal(original[f"{policy}_selected_extra_pairs"], altered[f"{policy}_selected_extra_pairs"])
        assert not np.array_equal(original[f"{policy}_vector_se"], altered[f"{policy}_vector_se"])


def test_cutoff_ties_retain_all_tied_pairs_and_selected_slots(synthetic_frame):
    graph = synthetic_frame.frame["graph"]
    selected = runner.budget_graph.select_pairs(graph, np.zeros(graph.n_nodes), 2)[len(graph.base):]
    audit, all_tied, selected_tied = runner.tie_audit(graph, np.zeros(graph.n_nodes), selected, 2)
    assert audit == {"cutoff": 0.0, "strictly_higher_pairs": 0, "all_tied_pairs": 8,
                     "selected_tied_slots": 2, "split_cutoff_tie": True}
    np.testing.assert_array_equal(all_tied, graph.extra)
    np.testing.assert_array_equal(selected_tied, graph.extra[:2])
    empty, tied, chosen = runner.tie_audit(graph, np.zeros(graph.n_nodes), np.empty((0, 2), dtype=int), 0)
    assert empty["cutoff"] is None
    assert tied.shape == chosen.shape == (0, 2)


@pytest.mark.parametrize("defect", ["mse", "edges"])
def test_replay_failure_preserves_all_completed_policy_arrays_outside_accepted_aggregation(synthetic_frame, tmp_path, defect):
    fixture = synthetic_frame
    control = deepcopy(fixture.control)
    if defect == "mse":
        control["policies"]["lagged_base_risk25"]["mse_normalized_acceleration"] += 0.01
    else:
        control["policies"]["lagged_base_risk25"]["directed_edges"] += 2
    with pytest.raises(runner.ReplayError) as error:
        runner.evaluate_frame(fixture.model, fixture.frame, control, "faithful", np.random.default_rng(91300))
    partial = error.value.partial_frame
    assert set(partial["row"]["policies"]) == set(runner.POLICIES)
    assert all(f"{policy}_vector_se" in partial["raw"] for policy in runner.POLICIES)
    path = runner.save_failed_frame(tmp_path, {"model": "faithful_seed0", "split": "valid", "index": 0}, partial)
    saved = json.loads((tmp_path / path).read_text())
    assert saved["accepted_for_aggregation"] is False
    assert saved["state"] == "failed"
    with np.load(tmp_path / saved["raw_file"]["path"], allow_pickle=False) as arrays:
        assert set(arrays.files) == set(partial["raw"])
    aggregate = runner.aggregate_frames([], required_ids())
    assert aggregate["equal_trajectory_mean"]["policies"]["inverse_count25"]["mse_normalized_acceleration"] is None


def test_late_nonfinite_prediction_retains_prior_measurements_and_failed_prediction(synthetic_frame, monkeypatch):
    fixture = synthetic_frame
    count = 0

    def failing_forward(*args, **kwargs):
        nonlocal count
        count += 1
        mean, risk = fixture.forward(*args, **kwargs)
        if count == 9:
            mean[0, 0] = torch.nan
        return mean, risk

    monkeypatch.setattr(runner.pilot, "run_forward", failing_forward)
    with pytest.raises(ValueError, match="Finite CPU") as error:
        runner.evaluate_frame(fixture.model, fixture.frame, fixture.control, "faithful", np.random.default_rng(91300))
    partial = error.value.partial_frame
    assert "inverse_count25" in partial["row"]["policies"]
    assert "velocity_rms25" not in partial["row"]["policies"]
    assert np.isnan(partial["raw"]["last_attempted_mean"][0, 0])


def test_forward_exception_names_attempted_policy_not_previous_completed_policy(synthetic_frame, monkeypatch):
    fixture = synthetic_frame
    count = 0

    def raising_forward(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 9:
            raise RuntimeError("synthetic forward failure")
        return fixture.forward(*args, **kwargs)

    monkeypatch.setattr(runner.pilot, "run_forward", raising_forward)
    with pytest.raises(RuntimeError, match="synthetic forward") as error:
        runner.evaluate_frame(fixture.model, fixture.frame, fixture.control, "faithful", np.random.default_rng(91300))
    assert error.value.partial_frame["row"]["last_attempted_policy"] == "velocity_rms25"
    assert "inverse_count25" in error.value.partial_frame["row"]["policies"]


def test_immutable_raw_frames_survive_an_interrupted_later_write(tmp_path, monkeypatch):
    first = tmp_path / "valid_000.npz"
    pointer = runner.write_frame_raw(first, {"errors": np.arange(4.0)})
    original = first.read_bytes()

    def interrupted(stream, **arrays):
        stream.write(b"incomplete ZIP data")
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(runner.np, "savez_compressed", interrupted)
    second = tmp_path / "valid_001.npz"
    with pytest.raises(OSError, match="interrupted"):
        runner.write_frame_raw(second, {"errors": np.arange(3.0)})
    assert first.read_bytes() == original
    assert runner.sha256(first) == pointer["sha256"]
    assert not second.exists()
    assert second.with_suffix(".npz.pending").read_bytes() == b"incomplete ZIP data"
    with pytest.raises(FileExistsError):
        runner.write_frame_raw(first, {"errors": np.zeros(4)})


def test_existing_output_directory_is_rejected_before_input_or_model_access(tmp_path, monkeypatch):
    existing = tmp_path / "attempt"
    existing.mkdir()
    monkeypatch.setattr(runner, "verify_inputs", lambda *_: pytest.fail("must not inspect inputs"))
    with pytest.raises(FileExistsError):
        runner.run(SimpleNamespace(output_dir=existing))


def test_failed_input_attempt_keeps_failure_and_null_required_groups(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic protocol")
    identity = tmp_path / "identity.json"
    identity.write_text("{}")
    monkeypatch.setattr(runner.torch, "load", lambda *_args, **_kwargs: pytest.fail("no model load before provenance"))
    output = tmp_path / "failed-attempt"
    with pytest.raises(KeyError):
        runner.run(SimpleNamespace(output_dir=output, protocol=protocol, input_identity=identity, data_dir=tmp_path))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["state"] == "failed"
    assert summary["summaries_validated"] is False
    assert (output / "failure.json").exists()
    assert summary["summaries"]["faithful/valid"]["policies"]["base"]["seed_values"] == [None] * 5


def test_operational_cap_stops_before_starting_more_frame_work(monkeypatch):
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 301.0)
    with pytest.raises(TimeoutError, match="300-second"):
        runner.enforce_cap(0.0)


def test_frame_sampling_seeds_match_the_original_pilot():
    assert runner.FRAME_SEEDS == {"valid": 811, "test": 812}


@pytest.fixture
def synthetic_identity(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    data.mkdir()
    for name in runner.REQUIRED_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Synthetic bytes; never deserialize as a checkpoint")
    protocol = {"steps": 3000, "seeds": list(runner.SEEDS), "objectives": list(runner.OBJECTIVES),
        "architecture": {"latent_width": 48, "processor_depth": 3, "history": 6, "noise": 0},
        "frames": {split: required_ids() for split in runner.SPLITS},
        "code_sha256": runner.sha256(runner.pilot.__file__)}
    (repo / runner.TRAINING_PROTOCOL).write_text(json.dumps(protocol))
    for name in ("metadata.json", "valid-pilot.npz", "test-pilot.npz"):
        (data / name).write_bytes(b"Synthetic bytes; never parse as numeric data")
    identity = {"files": {name: runner.sha256(repo / name) for name in sorted(runner.REQUIRED_FILES)},
        "data_sha256": {path.name: runner.sha256(path) for path in data.iterdir()},
        "existing_numeric_source_sha256": {name: runner.sha256(module.__file__) for name, module in
            (("pilot.py", runner.pilot), ("budget_graph.py", runner.budget_graph), ("risk_benefit.py", runner.risk_benefit))}}
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(identity))
    return SimpleNamespace(repo=repo, data=data, identity=path, protocol=protocol)


def test_all_manifest_hashes_are_verified_without_deserializing_models_or_numeric_data(synthetic_identity, monkeypatch):
    fixture = synthetic_identity
    monkeypatch.setattr(runner.torch, "load", lambda *_args, **_kwargs: pytest.fail("provenance check must not load weights"))
    monkeypatch.setattr(runner.np, "load", lambda *_args, **_kwargs: pytest.fail("provenance check must not load numeric data"))
    _, protocol = runner.verify_inputs(fixture.data, fixture.identity, fixture.repo)
    assert protocol == fixture.protocol


@pytest.mark.parametrize("source", ["checkpoint", "compact_data"])
def test_corrupt_pinned_bytes_fail_before_model_or_data_deserialization(synthetic_identity, monkeypatch, source):
    fixture = synthetic_identity
    path = (fixture.repo / runner.PILOT_DIRECTORY / "faithful_seed0.pt" if source == "checkpoint"
            else fixture.data / "test-pilot.npz")
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(runner.torch, "load", lambda *_args, **_kwargs: pytest.fail("must fail before weight loading"))
    with pytest.raises(ValueError, match="Pinned .*hash differs"):
        runner.verify_inputs(fixture.data, fixture.identity, fixture.repo)


def test_manifest_cannot_add_training_or_reserved_data_sources(synthetic_identity):
    fixture = synthetic_identity
    value = json.loads(fixture.identity.read_text())
    value["data_sha256"]["train-pilot.npz"] = "0" * 64
    fixture.identity.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Only metadata"):
        runner.verify_inputs(fixture.data, fixture.identity, fixture.repo)
