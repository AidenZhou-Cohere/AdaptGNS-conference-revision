"""Real child-parser contract checks with synthetic paths and no execution.

Every child main is stopped immediately after argparse accepts its arguments.
No converter, model, checkpoint, dataset, device, lock, or subprocess is used.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib
from pathlib import Path

import pytest

from research import run_evaluation_queue as queue


class ParsedWithoutExecution(BaseException):
    def __init__(self, namespace):
        self.namespace = namespace


def parse_child(command, monkeypatch):
    """Run the genuine parser, stopping before child operational validation."""
    assert command[1] == "-m"
    module = importlib.import_module(command[2])
    genuine_parse = argparse.ArgumentParser.parse_args
    calls = []

    def stop_after_parse(parser, args=None, namespace=None):
        # Pass the actual generated argv directly, without invoking a shell.
        assert args is None
        value = genuine_parse(parser, command[3:], namespace)
        calls.append(value)
        raise ParsedWithoutExecution(value)

    with monkeypatch.context() as child_patch:
        child_patch.setattr(argparse.ArgumentParser, "parse_args", stop_after_parse)
        with pytest.raises(ParsedWithoutExecution) as stopped:
            module.main()
    assert calls == [stopped.value.namespace]
    return stopped.value.namespace


@pytest.fixture
def synthetic_args(tmp_path):
    root = tmp_path / "synthetic paths with spaces"
    return queue.parse_args([
        "--training-dir", str(root / "training"),
        "--data-dir", str(root / "converted"),
        "--raw-test", str(root / "raw" / "test.tfrecord"),
        "--output-dir", str(root / "evaluation"),
        "--protocol", str(root / "training protocol.md"),
        "--companion-protocol", str(root / "same state protocol.md"),
        "--check-only",
    ])


@pytest.fixture
def synthetic_models(synthetic_args):
    return [{**model, "checkpoint": str(synthetic_args.training_dir / model["name"] / "checkpoint-100000.pt"),
             "checkpoint_sha256": f"{index + 1:064x}", "metadata_sha256": "a" * 64}
            for index, model in enumerate(queue.MODELS)]


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("job_index", range(12))
def test_generated_evaluator_argv_is_accepted_with_exact_locked_settings(
        synthetic_args, synthetic_models, monkeypatch, job_index, resume):
    jobs = queue.evaluation_jobs(synthetic_args, synthetic_models)
    assert len(jobs) == 12
    assert [job["name"] for job in jobs] == [f"{kind}/{model['name']}"
        for kind in ("rollout", "same_state") for model in synthetic_models]
    job = jobs[job_index]
    parsed = parse_child(job["command"] + (["--resume"] if resume else []), monkeypatch)
    assert parsed.manifest == synthetic_args.data_dir / "test.json"
    assert parsed.checkpoint == Path(job["model"]["checkpoint"])
    assert parsed.output_dir == job["output"]
    assert parsed.scope == "locked_test" and parsed.device == "mps"
    assert parsed.threads == 2 and parsed.clear_stale_lock is True and parsed.resume is resume
    assert parsed.max_trajectories is None
    assert parsed.max_candidate_pairs == 100000 and parsed.max_abs_coordinate == 10.
    if job["kind"] == "rollout":
        assert parsed.protocol == synthetic_args.protocol
        assert parsed.horizon == 995 and parsed.start_index == 3
        assert parsed.policies == list(queue.rollout_summary.POLICIES)
        assert parsed.rng_seed == 93000
        assert parsed.trajectory_ids is None
        assert parsed.expected_test_source_sha256 == queue.SOURCE_SHA256
    else:
        assert parsed.protocol == synthetic_args.companion_protocol
        assert parsed.training_protocol == synthetic_args.protocol
        assert parsed.target_frames == list(queue.diagnostic_summary.TARGET_FRAMES)
        assert parsed.repeats == 6
    assert not synthetic_args.raw_test.exists()
    assert not synthetic_args.training_dir.exists() and not synthetic_args.data_dir.exists()
    assert not synthetic_args.output_dir.exists()


def test_generated_converter_and_summary_commands_reach_their_real_parsers(
        synthetic_args, synthetic_models, monkeypatch):
    """Capture command construction through the queue, with all I/O gates fake."""
    before_cutoff = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(queue, "now", lambda: before_cutoff)
    monkeypatch.setattr(queue, "preflight", lambda *args, **kwargs: {
        "state": "ready", "models": synthetic_models, "blocking_reasons": [],
        "validation_evidence": {"scope": "synthetic parser-only gate"}})
    monkeypatch.setattr(queue.reviewed, "acquire_lock", lambda path: "synthetic-lock-token")
    monkeypatch.setattr(queue, "RunLock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(queue, "source_snapshot", lambda: {"synthetic_source.py": "b" * 64})
    monkeypatch.setattr(queue, "sha256", lambda path: queue.SOURCE_SHA256)
    monkeypatch.setattr(queue, "verify_test_manifest", lambda *args: {"metadata_sha256": "a" * 64})
    monkeypatch.setattr(queue, "read_json", lambda path: {"state": "complete"})
    completed_outputs = set()
    monkeypatch.setattr(queue, "validate_evaluation", lambda job, *args:
                        "complete" if job["output"] in completed_outputs else "new")
    commands = []

    def capture(command, *args, **kwargs):
        parsed = parse_child(command, monkeypatch)
        commands.append((command[2], parsed))
        if command[2] in {"research.full_rollout", "research.full_same_state"}:
            completed_outputs.add(parsed.output_dir)
        return {"state": "complete", "started": False}

    monkeypatch.setattr(queue, "supervise", capture)
    result = queue.run_queue(synthetic_args)
    assert result["state"] == "complete"
    assert len(commands) == 15  # converter, twelve evaluator commands, two summaries
    assert commands[0][0] == "research.prepare_full_waterdrop"
    converter = commands[0][1]
    assert converter.input_dir == synthetic_args.raw_test.parent
    assert converter.output_dir == synthetic_args.data_dir
    assert converter.metadata == synthetic_args.data_dir / "metadata.json"
    assert converter.splits == ["test"]
    assert converter.expected_train_bytes is None
    assert [name for name, _ in commands[-2:]] == ["research.summarize_full_rollouts", "research.summarize_full_same_state"]
    rollout, same_state = (item[1] for item in commands[-2:])
    assert rollout.evaluation_root == synthetic_args.output_dir / "rollout"
    assert rollout.protocol == synthetic_args.protocol
    assert rollout.output_prefix == synthetic_args.output_dir / "reports/full_rollouts"
    assert same_state.evaluation_root == synthetic_args.output_dir / "same_state"
    assert same_state.training_protocol == synthetic_args.protocol
    assert same_state.companion_protocol == synthetic_args.companion_protocol
    assert same_state.output_prefix == synthetic_args.output_dir / "reports/full_same_state"
    # Only synthetic supervisor metadata was written under its tmp output.
    assert not synthetic_args.raw_test.exists()
    assert not synthetic_args.data_dir.exists() and not synthetic_args.training_dir.exists()
    assert not (synthetic_args.output_dir / "rollout").exists()
    assert not (synthetic_args.output_dir / "same_state").exists()
