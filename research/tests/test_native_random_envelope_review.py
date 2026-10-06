"""Independent adversarial evaluator checks; synthetic data/models only."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research import native_random_envelope as study
from research.tests.test_graph_convention_bridge import fixture, identity, tiny_model


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def original_payload(metadata_hash="synthetic-metadata"):
    run = {"seed": 0, "objective": "faithful", "steps": 100000,
           "scope": "bounded_full_data_100k", "metadata_sha256": metadata_hash,
           "research_protocol_sha256": study.full.sha256(study.TRAINING_PROTOCOL)}
    return {"full_training_schema": 1, "completed_steps": 100000,
            "simulator_config": {"particle_dimensions": 2, "latent_dim": 128,
                "nmessage_passing_steps": 10, "nmlp_layers": 2, "mlp_hidden_dim": 128,
                "uncertainty_parameterization": "variance", "detach_variance_features": True},
            "training_config": {"loss": "faithful", "completed_optimizer_updates": 100000},
            "run_config": run, "run_config_sha256": config_hash(run)}


def publish_original(directory, payload):
    checkpoint = directory / "checkpoint-100000.pt"
    torch.save(payload, checkpoint)
    latest = {"path": checkpoint.name, "sha256": study.full.sha256(checkpoint), "completed_steps": 100000}
    status = {"state": "complete", "completed_steps": 100000, "latest_checkpoint": latest,
              "objective": payload["run_config"]["objective"], "seed": payload["run_config"]["seed"],
              "run_config_sha256": payload["run_config_sha256"]}
    for name, value in (("latest.json", latest), ("status.json", status), ("protocol.json", payload["run_config"])):
        (directory / name).write_text(json.dumps(value))
    return checkpoint


def test_independent_complete_original_publication_accepted(tmp_path):
    payload = original_payload()
    checkpoint = publish_original(tmp_path, payload)
    study.full.check_checkpoint(payload, "locked_test", study.TRAINING_PROTOCOL, "synthetic-metadata")
    assert study.committed_checkpoint(checkpoint, payload) == study.full.sha256(checkpoint)


@pytest.mark.parametrize("filename,key,value", [
    ("latest.json", "completed_steps", 99999),
    ("latest.json", "sha256", "not-the-checkpoint"),
    ("latest.json", "path", "checkpoint-110000.pt"),
    ("status.json", "state", "running"),
    ("status.json", "completed_steps", 110000),
    ("status.json", "seed", 1),
    ("status.json", "objective", "nll"),
    ("status.json", "run_config_sha256", "changed"),
    ("protocol.json", "seed", 1),
])
def test_independent_uncommitted_or_relabelled_publication_refused(tmp_path, filename, key, value):
    payload = original_payload()
    checkpoint = publish_original(tmp_path, payload)
    document = study.strict_json(tmp_path / filename)
    document[key] = value
    (tmp_path / filename).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Committed|publication"):
        study.committed_checkpoint(checkpoint, payload)


@pytest.mark.parametrize("defect", ["110k", "partial_updates", "head", "detachment", "run_hash", "objective", "boolean_seed"])
def test_independent_bad_checkpoint_stops_before_any_model_load(tmp_path, monkeypatch, defect):
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")
    payload = original_payload(study.full.sha256(metadata))
    if defect == "110k":
        payload["completed_steps"] = 110000
    elif defect == "partial_updates":
        payload["training_config"]["completed_optimizer_updates"] = 99999
    elif defect == "head":
        payload["simulator_config"]["uncertainty_parameterization"] = "std"
    elif defect == "detachment":
        payload["simulator_config"]["detach_variance_features"] = False
    elif defect == "run_hash":
        payload["run_config_sha256"] = "incorrect"
    elif defect == "objective":
        payload["run_config"]["objective"] = "nll"
        payload["run_config_sha256"] = config_hash(payload["run_config"])
    else:
        payload["run_config"]["seed"] = True
        payload["run_config_sha256"] = config_hash(payload["run_config"])
    checkpoint = publish_original(tmp_path, payload)
    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed checkpoint reached model/data loading")
    monkeypatch.setattr(study.full, "load_for_evaluation", forbidden)
    monkeypatch.setattr(study.full, "load_manifest_data", forbidden)
    args = SimpleNamespace(checkpoint=checkpoint, validation_manifest=tmp_path / "valid.json")
    with pytest.raises(ValueError):
        study.prepare_inputs(args, "cpu")


def test_independent_previous_guard_counts_zero_attempted_passes(tmp_path):
    current, previous, types, target = fixture()
    previous = previous.copy()
    previous[0, 0, 0] = np.nan
    row, arrays = study.evaluate_frame(tiny_model(), current, previous, types, target, identity(), 0, "cpu")
    failed = row["cases"][study.RISKS[0]]
    assert failed["status"] == "failed"
    assert failed["network_passes_for_standalone_policy"] == 0
    assert row["previous_score"]["failure"]["phase"] == "previous_history"
    assert "previous_base_prediction" not in arrays
    assert np.isnan(arrays["previous_history"][0, 0, 0])
    assert row["cases"][study.RISKS[1]]["status"] == "complete"
    assert all(row["cases"][name]["status"] == "complete" for name in study.RANDOM_CASES)
    summary = row["random_envelope"][study.METRICS[0]][study.RISKS[0]]
    assert summary["defined_draws"] == 8 and summary["mean"] is not None
    assert summary["reference_minus_random_mean"] is None


def test_independent_current_base_dependency_failure_does_not_skip_other_draws(monkeypatch):
    original = study.bridge.native_parity
    def bad_current(*args, **kwargs):
        parity, arrays, output = original(*args, **kwargs)
        output["risk"][0] = 0.
        return parity, arrays, output
    monkeypatch.setattr(study.bridge, "native_parity", bad_current)
    row, arrays = study.evaluate_frame(tiny_model(), *fixture(), identity(), 0, "cpu")
    assert row["cases"]["base"]["status"] == "failed"
    assert row["cases"][study.RISKS[1]]["failure"]["category"] == "scoring_pass_failed"
    assert row["cases"][study.RISKS[1]]["network_passes_for_standalone_policy"] == 1
    assert arrays["risk__base"][0] == 0.
    assert row["cases"][study.RISKS[0]]["status"] == "complete"
    assert all(row["cases"][name]["status"] == "complete" for name in study.RANDOM_CASES)
    assert row["random_envelope"][study.METRICS[0]]["base"]["defined_draws"] == 8


def run_fixture(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    pin = tmp_path / "input"
    pin.write_text("original bytes")
    expected = [identity(), {**identity(), "target_frame": 8}]
    protocol = {"objective": "faithful", "seed": 0, "checkpoint_sha256": "synthetic",
                "cases": list(study.POLICIES), "expected_frames": expected,
                "input_files_sha256": {str(pin): study.full.sha256(pin)}}
    args = SimpleNamespace(output_dir=output, saved_same_state_dir=tmp_path / "saved", device="cpu")
    monkeypatch.setattr(study.bridge, "frame_input", lambda *args: tuple(value.copy() for value in fixture()))
    return args, protocol, expected, pin


def test_independent_run_commits_parity_failure_then_stops_without_retry(tmp_path, monkeypatch):
    args, protocol, expected, pin = run_fixture(tmp_path, monkeypatch)
    model = tiny_model()
    original = model.predict_positions_with_variance
    def disagree(*args, **kwargs):
        prediction, risk = original(*args, **kwargs)
        return prediction + .01, risk
    model.predict_positions_with_variance = disagree
    with pytest.raises(RuntimeError, match="Native parity failure retained"):
        study.run(args, protocol, model, {}, expected, {})
    result = study.strict_json(args.output_dir / "result.json")
    assert result["state"] == "partial" and len(result["records"]) == 1
    record = result["records"][0]
    assert record["status"] == "failed" and record["failure"]["category"] == "native_parity_failure"
    row = study.strict_json(args.output_dir / record["record_file"])
    assert len(row["cases"]) == 13 and all(case["status"] == "failed" for case in row["cases"].values())
    assert study.full.sha256(args.output_dir / record["array_file"]) == record["array_sha256"]
    assert study.full.sha256(args.output_dir / record["record_file"]) == record["record_sha256"]
    assert study.strict_json(args.output_dir / "status.json")["state"] == "error"
    before = {path.name: study.full.sha256(path) for path in args.output_dir.iterdir()}
    with pytest.raises(ValueError, match="Fresh owned"):
        study.run(args, protocol, model, {}, expected, {})
    assert before == {path.name: study.full.sha256(path) for path in args.output_dir.iterdir()}


def test_independent_run_retains_failed_case_and_complete_other_population(tmp_path, monkeypatch):
    args, protocol, expected, pin = run_fixture(tmp_path, monkeypatch)
    original = study.evaluate_frame
    def failed_previous(model, current, previous, *rest):
        previous[0, 0, 0] = np.nan
        return original(model, current, previous, *rest)
    monkeypatch.setattr(study, "evaluate_frame", failed_previous)
    study.run(args, protocol, tiny_model(), {}, expected, {})
    result = study.strict_json(args.output_dir / "result.json")
    assert result["state"] == "complete" and result["failed_frames"] == 2 and result["complete_frames"] == 0
    assert len(result["records"]) == 2
    status = study.strict_json(args.output_dir / "status.json")
    assert status["state"] == "complete" and status["result_sha256"] == study.full.sha256(args.output_dir / "result.json")


def test_independent_final_pin_mutation_prevents_complete_publication(tmp_path, monkeypatch):
    args, protocol, expected, pin = run_fixture(tmp_path, monkeypatch)
    original = study.evaluate_frame
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        pin.write_text("changed after admission")
        return result
    monkeypatch.setattr(study, "evaluate_frame", mutate)
    with pytest.raises(ValueError, match="byte identity"):
        study.run(args, protocol, tiny_model(), {}, expected, {})
    result = study.strict_json(args.output_dir / "result.json")
    assert result["state"] == "partial" and len(result["records"]) == 2
    assert study.strict_json(args.output_dir / "status.json")["state"] == "error"


def test_independent_cutoff_precedes_next_frame_and_records_stop(tmp_path, monkeypatch):
    args, protocol, expected, pin = run_fixture(tmp_path, monkeypatch)
    class AtCutoff(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(study.native.DEADLINE)
    monkeypatch.setattr(study, "datetime", AtCutoff)
    def forbidden(*args, **kwargs):
        raise AssertionError("Cutoff reached real/synthetic predictor")
    monkeypatch.setattr(study, "evaluate_frame", forbidden)
    with pytest.raises(TimeoutError, match="Research cutoff"):
        study.run(args, protocol, None, {}, expected, {})
    status = study.strict_json(args.output_dir / "status.json")
    assert status["state"] == "deadline_stopped" and status["committed_frames"] == 0
    assert not (args.output_dir / "result.json").exists()
    assert not list(args.output_dir.glob("*.npz"))
