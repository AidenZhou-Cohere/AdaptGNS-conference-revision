"""Synthetic scalar reports and mocked process lifecycle only; no model/CUDA."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "supervise_graph_support_evaluation_quota_v2.py"
spec = importlib.util.spec_from_file_location("quota_test_subject", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
NOW = datetime(2026, 10, 6, 5, tzinfo=timezone.utc)


def atomic(path, value):
    Path(path).write_text(json.dumps(value))


def release_fixture(tmp_path, monkeypatch, dataset="Sand", role="B"):
    args = SimpleNamespace(output_dir=tmp_path / "out", lifecycle_source=Path("/lifecycle.py"))
    helper = SimpleNamespace(canonical_gpu_uuid=lambda x: x)
    monkeypatch.setattr(m, "load_lifecycle", lambda path: helper)
    monkeypatch.setattr(m, "sha", lambda path: "a" * 64)
    pins = {str(SOURCE): "a" * 64, "/lifecycle.py": m.LIFECYCLE_SHA,
            "/python": "a" * 64, "/final.py": m.EVALUATORS[dataset]}
    streams = []
    for arm, seed, gpu in m.SCHEDULE[role]:
        stream = dict(id=f"{arm}_seed{seed}", arm=arm, seed=seed, gpu=gpu, commands=[])
        for name, mode, split, quota in m.STAGES:
            options = {"--" + name: "/" + name for name in ("repo", "benchmark-helper", "cohort", "manifest", "admission",
                "structural-report", "train-admission", "trainer-source", "protocol", "cohort-audit")}
            options.update({"--checkpoint": f"/{arm}{seed}.pt", "--checkpoint-sha256": "c" * 64,
                "--benchmark-sha256": "b" * 64, "--output-dir": str(args.output_dir / "jobs" / stream["id"] / name),
                "--mode": mode, "--split": split, "--objective": "faithful", "--arm": arm, "--seed": str(seed),
                "--cuda-index": str(gpu), "--threads": "2", "--max-seconds": str(quota)})
            if dataset == "Goop":
                options.update({"--" + key: "/" + key for key in ("acquisition-report", "context-semantics", "auxiliary-report")})
                if split == "test":
                    options["--cross-split-audit"] = "/cross-split-audit"
            pins.update({v: "a" * 64 for v in options.values() if v.startswith("/")})
            pins[options["--checkpoint"]] = "c" * 64
            pins[options["--benchmark-helper"]] = "b" * 64
            stream["commands"].append(["/python", "/final.py", "--execute", *[x for pair in options.items() for x in pair]])
        streams.append(stream)
    release = dict(schema=m.RELEASE_SCHEMA, status="admitted_for_execution_allocation", issued_by="root", cost_basis=m.COST_BASIS,
        dataset=dataset, host_role=role, host=m.socket.gethostname(), environment=m.ENVIRONMENT,
        stage_quotas_seconds={name: quota for name, _, _, quota in m.STAGES}, cleanup_seconds_per_invocation=15,
        outer_processing_reserve_seconds=6300, clock_error_bound_seconds=1, process_clock_checked_utc=NOW.isoformat(),
        latest_start_utc=(NOW + timedelta(hours=1)).isoformat(), compute_analysis_deadline_utc=m.DEADLINE.isoformat(),
        gpu_uuids=["gpu0", "gpu1", "gpu2", "gpu3"], files_sha256=pins, streams=streams)
    return args, release, helper


@pytest.mark.parametrize("dataset,role", [("Sand", "A"), ("Sand", "B"), ("Goop", "A"), ("Goop", "B")])
def test_valid_exact_release(tmp_path, monkeypatch, dataset, role):
    args, release, helper = release_fixture(tmp_path, monkeypatch, dataset, role)
    assert m.validate_release(release, args, NOW) is helper
    assert m.STREAM_SECONDS == 11760


@pytest.mark.parametrize("change", [
    lambda r: r.update(cost_basis="measured_complete_horizon"),
    lambda r: r.update(outer_processing_reserve_seconds=0),
    lambda r: r.update(clock_error_bound_seconds=6),
    lambda r: r.update(process_clock_checked_utc=(NOW - timedelta(minutes=6)).isoformat()),
    lambda r: r.update(latest_start_utc=(m.DEADLINE - timedelta(seconds=11760)).isoformat()),
    lambda r: r.update(gpu_uuids=["same"] * 4),
    lambda r: r["streams"][0].update(gpu=3),
    lambda r: r["streams"][0]["commands"][0].extend(["--resume", "/old"]),
    lambda r: r["streams"][0]["commands"][0].extend(["--threads", "2"]),
    lambda r: r["files_sha256"].pop("/cohort-audit"),
    lambda r: r["streams"].pop(),
])
def test_refuses_release_changes(tmp_path, monkeypatch, change):
    args, release, _ = release_fixture(tmp_path, monkeypatch)
    change(release)
    with pytest.raises(ValueError):
        m.validate_release(release, args, NOW)


def test_coverage_uses_exact_fixed_workload():
    assert len(m.expected_cells("Sand", "full_rollout_test")) == 180
    assert m.expected_cells("Sand", "same_state_test")[:5] == [(0, x) for x in (7, 85, 163, 241, 319)]
    assert m.expected_cells("Goop", "same_state_valid")[:5] == [(0, x) for x in (7, 105, 203, 301, 400)]
    for dataset, last in (("Sand", 319), ("Goop", 400)):
        cells = m.expected_cells(dataset, "clean_validation")
        assert len(cells) == len(set(cells)) == 128 and cells[0] == (0, 6) and cells[-1] == (29, last)


def test_coverage_preserves_failures_and_separates_timeout_missing(tmp_path):
    atomic(tmp_path / "trajectory_0_base.json", dict(source_index=0, policy="base", status="complete", completed_steps=314, horizon=314))
    failure = dict(category="coordinate_guard", threshold=100)
    atomic(tmp_path / "trajectory_0_dense.json", dict(source_index=0, policy="dense", status="failed", failure=failure))
    out = m.account_stage("Sand", "full_rollout_test", tmp_path, dict(quota_expired=True, current_before_stop=dict(source_index=0, policy="random25")))
    assert out["recorded_cells"] == 2 and out["missing_cells"] == 178
    assert [c["state"] for c in out["cells"][:4]] == ["completed_required_outcome", "recorded_failed_outcome", "timed_out_current", "not_completed_before_invocation_end"]
    assert out["cells"][1]["failure"] == failure and out["no_survivor_mean_computed"]


def test_never_started_and_invalid_completed_horizon(tmp_path):
    empty = m.account_stage("Goop", "clean_validation", tmp_path, dict(state="never_started"))
    assert all(c["state"] == "never_started" for c in empty["cells"])
    atomic(tmp_path / "trajectory_0_base.json", dict(source_index=0, policy="base", status="complete", completed_steps=314, horizon=314))
    with pytest.raises(ValueError, match="full horizon"):
        m.account_stage("Goop", "full_rollout_test", tmp_path, {})


class ImmediatePool:
    def __init__(self, **kwargs): pass
    def submit(self, fn):
        value = fn()
        return SimpleNamespace(done=lambda: True, result=lambda: value)
    def shutdown(self, **kwargs): pass


def mocked_queue(tmp_path, monkeypatch, behavior="complete", bad_identity=False, bad_row=False, changed_input=False):
    args = SimpleNamespace(output_dir=tmp_path / "queue")
    now = [0.0]
    monkeypatch.setattr(m, "STAGES", tuple((name, mode, split, 1) for name, mode, split, _ in m.STAGES))
    monkeypatch.setattr(m.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(m.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(m, "datetime", type("Clock", (datetime,), {"now": staticmethod(lambda zone: NOW)}))
    monkeypatch.setattr(m, "ThreadPoolExecutor", ImmediatePool)
    checks = []
    def recheck(release):
        checks.append(True)
        if changed_input and len(checks) > 1:
            raise ValueError("changed synthetic input")
    monkeypatch.setattr(m, "recheck_files", recheck)
    processes = []
    commands = [["/python", "/final", "--execute", "--output-dir", str(args.output_dir / "jobs/base_seed2" / name)] for name, _, _, _ in m.STAGES]
    release = dict(dataset="Sand", streams=[dict(id="base_seed2", gpu=0, commands=commands)], clock_error_bound_seconds=1,
                   latest_start_utc=(NOW + timedelta(hours=1)).isoformat(), gpu_uuids=["gpu0"] * 4, outer_processing_reserve_seconds=6300)
    def popen(command, **kwargs):
        p = SimpleNamespace(pid=100 + len(processes), returncode=None, command=command, born=now[0], signaled=None)
        assert kwargs["start_new_session"] is True
        processes.append(p)
        directory = Path(m.flags(command)["--output-dir"])
        directory.mkdir()
        atomic(directory / "status.json", {"case": dict(source_index=0, policy="base"), "current": dict(source_index=0, target_frame=7)})
        if bad_row and len(processes) == 1:
            (directory / "trajectory_0_base.json").write_text("malformed")
        return p
    monkeypatch.setattr(m.subprocess, "Popen", popen)
    def wait4(pid, options):
        p = next(p for p in processes if p.pid == pid)
        complete = behavior == "complete" and now[0] >= p.born + .4
        failed = behavior in ("error", "late_error") and now[0] >= p.born + .4
        killed = p.signaled is not None and behavior != "unreaped" and (bad_identity or p.signaled == signal.SIGKILL)
        return (pid, 256 if failed else 0, None) if complete or failed or killed else (0, 0, None)
    monkeypatch.setattr(m.os, "wait4", wait4)
    def identity(pid):
        p = next(p for p in processes if p.pid == pid)
        if behavior == "late_error":
            now[0] += 1.1
        return dict(argv=["wrong"] if bad_identity else p.command, ppid=m.os.getpid())
    def stop(children, sig):
        for child in children:
            assert child["process"] in processes
            child["process"].signaled = sig
            child["signals"].append(dict(signal=signal.Signals(sig).name))
    helper = SimpleNamespace(atomic_json=atomic, gpu_processes=lambda: [], process_identity=identity, stop_owned=stop,
                             canonical_gpu_uuid=lambda value: value)
    result = m.execute_queue(args, release, helper)
    return result, m.read(args.output_dir / "queue_status.json"), m.read(args.output_dir / "coverage_ledger.json"), processes, now[0]


def test_mock_normal_four_distinct_stages(tmp_path, monkeypatch):
    result, status, ledger, processes, _ = mocked_queue(tmp_path, monkeypatch)
    assert result == 0 and status["state"] == "allocation_finished" and len(processes) == 4
    assert len(ledger["stages"]) == 4 and ledger["whole_cohort_metrics_admitted"] is False


def test_mock_quota_escalation_and_no_retry(tmp_path, monkeypatch):
    result, status, ledger, processes, elapsed = mocked_queue(tmp_path, monkeypatch, behavior="timeout")
    assert result == 0 and len(processes) == 4 and elapsed < 4 * 16
    for entry in ledger["stages"]:
        assert entry["outcome"]["quota_expired"]
        assert [s["signal"] for s in entry["outcome"]["signals"]] == ["SIGINT", "SIGTERM", "SIGKILL"]
    assert ledger["stages"][0]["cells"][0]["state"] == "timed_out_current"


def test_mock_nonquota_error_stops_later_stages(tmp_path, monkeypatch):
    result, status, ledger, processes, _ = mocked_queue(tmp_path, monkeypatch, behavior="error")
    assert result == 1 and len(processes) == 1
    assert all(e["outcome"]["state"] == "never_started" for e in ledger["stages"][1:])


def test_late_observed_error_cannot_be_reclassified_as_authorized_quota_stop(tmp_path, monkeypatch):
    result, status, ledger, processes, _ = mocked_queue(tmp_path, monkeypatch, behavior="late_error")
    assert result == 1 and len(processes) == 1
    assert ledger["stages"][0]["outcome"]["quota_expired"] is True
    assert ledger["stages"][0]["outcome"]["quota_stop_initiated"] is False


def test_mock_identity_failure_registered_and_cleanup_reaped_not_never_started(tmp_path, monkeypatch):
    result, status, ledger, processes, elapsed = mocked_queue(tmp_path, monkeypatch, behavior="timeout", bad_identity=True)
    assert result == 1 and len(processes) == 1 and elapsed < 15
    assert ledger["stages"][0]["outcome"]["state"] == "aborted_by_supervisor"
    assert not status["unreaped_owned_children"]


def test_mock_unreaped_cleanup_is_bounded_and_coverage_deferred(tmp_path, monkeypatch):
    result, status, ledger, processes, elapsed = mocked_queue(tmp_path, monkeypatch, behavior="unreaped")
    assert result == 1 and len(processes) == 1 and elapsed < 17
    assert len(status["unreaped_owned_children"]) == 1
    assert all(e["coverage_audit_state"] == "deferred_while_owned_child_unreaped" and e["cells"] is None for e in ledger["stages"])


@pytest.mark.parametrize("kwargs", [dict(bad_row=True), dict(changed_input=True)])
def test_mock_final_audit_failure_preserves_status(tmp_path, monkeypatch, kwargs):
    result, status, ledger, processes, _ = mocked_queue(tmp_path, monkeypatch, **kwargs)
    assert result == 1 and status["state"] == "stopped_requires_review" and status["audit_errors"]
    assert len(processes) == 4


def test_default_launches_nothing(monkeypatch, capsys):
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: pytest.fail("No child allowed"))
    assert m.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "description_only"
