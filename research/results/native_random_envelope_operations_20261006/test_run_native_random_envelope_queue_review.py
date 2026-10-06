"""Independent queue review: all paths, clocks and subprocesses are synthetic."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from test_run_native_random_envelope_queue import (
    q, isolated, launch_fixture, release_fixture, write,
)


def test_independent_advancing_clock_can_launch_initial_exact_budget(monkeypatch):
    prepared, processes, calls = launch_fixture(monkeypatch)
    baseline = datetime(2026, 10, 6, 7, tzinfo=timezone.utc)
    ticks = [0]
    def advancing():
        ticks[0] += 1
        return baseline + timedelta(milliseconds=ticks[0])
    monkeypatch.setattr(q, "now", advancing)
    assert q.launch(prepared) == 0
    assert len(processes) == 7 and len(calls) == 6
    status = q.read(q.OUTPUT / "queue_status.json")
    assert status["state"] == "complete"
    assert status["scientific_summary_and_audit_pending"] is True
    assert all(job["state"] == "complete" for job in status["jobs"])


def test_independent_final_child_overrun_preserves_inference_but_stops_new_analysis(monkeypatch):
    prepared, processes, calls = launch_fixture(monkeypatch)
    baseline = datetime(2026, 10, 6, 7, tzinfo=timezone.utc)
    elapsed = [0.]
    monkeypatch.setattr(q, "now", lambda: baseline + timedelta(seconds=elapsed[0]))
    monkeypatch.setattr(q.time, "perf_counter", lambda: elapsed[0])
    original_constructor = q.subprocess.Popen
    class TimedProcess(original_constructor):
        def wait(self, timeout=None):
            if self.command[0] != "caffeinate":
                elapsed[0] += 1400. if len(processes) == 7 else 100.
            return super().wait(timeout)
    monkeypatch.setattr(q.subprocess, "Popen", TimedProcess)
    assert q.launch(prepared) == 2
    assert elapsed[0] == 1900. and elapsed[0] + q.SUMMARY_RESERVE > q.TOTAL_BUDGET
    status = q.read(q.OUTPUT / "queue_status.json")
    assert status["state"] == "gate_stopped" and status["inference_complete"] is True
    assert status["scientific_summary_and_audit_pending"] is True
    assert all(job["state"] == "complete" for job in status["jobs"])
    assert len(calls) == 6 and len(processes) == 7
    assert processes[0].terminated and not any(child.terminated for child in processes[1:])


def test_independent_new_heavy_process_stops_before_second_child(monkeypatch):
    prepared, processes, calls = launch_fixture(monkeypatch)
    checks = [0]
    def busy_after_first():
        checks[0] += 1
        if checks[0] == 3:
            raise RuntimeError("Synthetic competing trainer appeared")
    monkeypatch.setattr(q, "assert_idle", busy_after_first)
    assert q.launch(prepared) == 2
    assert len(processes) == 2 and len(calls) == 1
    status = q.read(q.OUTPUT / "queue_status.json")
    assert status["state"] == "gate_stopped"
    assert status["jobs"][0]["state"] == "complete"
    assert all(job["state"] == "unstarted" for job in status["jobs"][1:])


def test_independent_postchild_immutable_mutation_prevents_new_jobs(monkeypatch):
    prepared, processes, calls = launch_fixture(monkeypatch)
    checks = [0]
    def mutated_after_first(*args):
        checks[0] += 1
        if checks[0] == 3:
            raise RuntimeError("Synthetic immutable source mutation")
    monkeypatch.setattr(q, "verify_pins", mutated_after_first)
    assert q.launch(prepared) == 1
    status = q.read(q.OUTPUT / "queue_status.json")
    assert status["state"] == "error" and status["jobs"][0]["state"] == "error"
    assert len(processes) == 2 and len(calls) == 1
    assert all(job["state"] == "unstarted" for job in status["jobs"][1:])


@pytest.mark.parametrize("command", [
    "/synthetic/python3.12 -X dev -W error -m research.native_random_envelope --launch",
    "/synthetic/Python -B -mresearch.native_graph_rollout",
    "/synthetic/python -- /x/run_continuation_evaluation_queue.py --launch",
    "/synthetic/python -u /x/native_random_envelope.py --launch",
    "/synthetic/run_graph_support_queue.py --launch",
])
def test_independent_heavy_process_argument_forms_detected(command):
    assert q.heavy_processes("100 " + command, 999) == [{"pid": 100, "command": command}]
    assert q.heavy_processes("100 " + command, 100) == []


@pytest.mark.parametrize("field", ["recorded_utc", "root_process_check_utc"])
def test_independent_root_release_exact15minute_boundary_and_utc(field, monkeypatch):
    release, native = release_fixture(monkeypatch)
    release[field] = (q.now() - timedelta(seconds=900)).isoformat()
    assert q.verify_release(release, native)
    release[field] = (q.now() - timedelta(seconds=900, microseconds=1)).isoformat()
    with pytest.raises(RuntimeError, match="stale|future"):
        q.verify_release(release, native)
    release[field] = q.now().replace(tzinfo=None).isoformat()
    with pytest.raises(RuntimeError, match="stale|future"):
        q.verify_release(release, native)


@pytest.mark.parametrize("mutation", ["protocol", "result", "checkpoint", "duplicate_seed"])
def test_independent_rehashed_passing_summary_must_bind_exact_prior_native_jobs(mutation, monkeypatch):
    release, native = release_fixture(monkeypatch)
    summary_path = Path(release["strict_native_summary_path"])
    report = q.read(summary_path)
    row = report["native_runs"][0]
    if mutation == "protocol":
        row["protocol_sha256"] = "d" * 64
    elif mutation == "result":
        result_path = str(q.NATIVE / q.NAMES[0] / "result.json")
        row["input_files_sha256"][result_path] = "d" * 64
    elif mutation == "checkpoint":
        row["checkpoint_sha256"] = "d" * 64
    else:
        report["native_runs"][1] = report["native_runs"][0]
    write(summary_path, report)
    release["strict_native_summary_sha256"] = q.sha(summary_path)
    audit_path = Path(release["native_scalar_audit_path"])
    audit = q.read(audit_path)
    audit["source_summary_sha256"] = release["strict_native_summary_sha256"]
    write(audit_path, audit)
    release["native_scalar_audit_sha256"] = q.sha(audit_path)
    with pytest.raises(RuntimeError, match="Native|native|summary"):
        q.verify_release(release, native)


def test_independent_native_summary_uses_existing_frozen_schema_without_directory(monkeypatch):
    release, native = release_fixture(monkeypatch)
    summary_path = Path(release["strict_native_summary_path"])
    report = q.read(summary_path)
    # Frozen summarize_native_graph_rollouts.load_native records location via
    # absolute input-file hash keys; it does not emit a directory field.
    assert all("directory" not in run for run in report["native_runs"])
    pins = q.verify_release(release, native)
    assert pins[str(summary_path)] == q.sha(summary_path)
    assert all(str(q.NATIVE / name / "protocol.json") in run["input_files_sha256"]
               for name, run in zip(q.NAMES, report["native_runs"]))
