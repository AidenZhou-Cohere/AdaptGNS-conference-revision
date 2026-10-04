"""Synthetic artifacts/processes only: no experimental checkpoint, test source or GPU."""
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from research import run_evaluation_queue as queue
from research.tests.test_summarize_training_validation import config, make_point

EARLY = queue.HARD_DEADLINE - timedelta(days=1)
DEAD_PID = 99999999


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")))


def args_for(tmp_path, *extra):
    return queue.parse_args(["--training-dir", str(tmp_path / "training"), "--data-dir", str(tmp_path / "data"),
                             "--raw-test", str(tmp_path / "raw/test.tfrecord"), "--output-dir", str(tmp_path / "evaluation"), *extra])


def check(args, **kwargs):
    return queue.preflight(args, scan=lambda: [], alive=lambda pid: False,
                           header_loader=lambda path: queue.read_json(path), clock=lambda: EARLY, **kwargs)


@pytest.fixture
def completed(tmp_path, config):
    """Six JSON-only fake checkpoint headers plus real 21-point validation fixtures."""
    args = args_for(tmp_path)
    metadata = {"bounds": [[0., 1.], [0., 1.]], "acc_mean": [0., 0.], "acc_std": [.1, .2],
                "vel_mean": [0., 0.], "vel_std": [.3, .4]}
    write(args.data_dir / "metadata.json", metadata)
    for split in ("train", "valid"):
        write(args.data_dir / (split + ".json"), {"records": []})
    sources = [queue.REPO / "research/full_training.py", *sorted((queue.REPO / "adaptive-gns/gns").glob("*.py"))]
    base = {**config, "research_protocol_sha256": queue.sha256(args.protocol), "metadata_sha256": queue.sha256(args.data_dir / "metadata.json"),
            "source_sha256": {str(path.relative_to(queue.REPO)): queue.sha256(path) for path in sources}}
    for split, count, digest in (("train", 1000, queue.rollout_summary.OFFICIAL_TRAIN_SHA256),
                                 ("valid", 30, queue.rollout_summary.OFFICIAL_VALID_SHA256)):
        base[split] = {"n_trajectories": count, "source": {"sha256": digest}, "manifest_sha256": queue.sha256(args.data_dir / (split + ".json"))}
    for model in queue.MODELS:
        folder = args.training_dir / model["name"]
        cfg = {**base, "objective": model["objective"], "seed": model["seed"]}
        digest = queue.reviewed.config_hash(cfg)
        write(folder / "protocol.json", cfg)
        write(folder / "status.json", {"state": "complete", "completed_steps": 100000, "requested_steps": 100000,
              "objective": model["objective"], "seed": model["seed"], "run_config_sha256": digest, "pid": DEAD_PID})
        simulator = {"particle_dimensions": 2, "latent_dim": 128, "nmessage_passing_steps": 10, "nmlp_layers": 2,
                     "mlp_hidden_dim": 128, "uncertainty_parameterization": "variance", "connectivity_radius": .015,
                     "variance_floor": 1e-6, "max_num_neighbors": 128, "radius_backend": "scipy_host",
                     "detach_variance_features": model["objective"] == "faithful", "boundaries": metadata["bounds"],
                     "normalization_stats": {name: {"mean": metadata[prefix + "_mean"],
                          "std": [math.sqrt(value**2 + (6.7e-4)**2) for value in metadata[prefix + "_std"]]}
                          for name, prefix in (("acceleration", "acc"), ("velocity", "vel"))}}
        checkpoint = folder / "checkpoint-100000.pt"
        write(checkpoint, {"format_version": 2, "full_training_schema": 1, "completed_steps": 100000,
                          "run_config": cfg, "run_config_sha256": digest, "simulator_config": simulator,
                          "training_config": {"loss": model["objective"], "completed_optimizer_updates": 100000, "full_run": cfg}})
        write(folder / "latest.json", {"path": checkpoint.name, "completed_steps": 100000, "sha256": queue.sha256(checkpoint)})
        for step in range(0, 100001, 5000):
            write(folder / "validation" / f"step-{step:06d}.json", make_point(cfg, step))
    return args


def no_access(*args, **kwargs):
    raise AssertionError("Protected operation must not be reached")


def test_default_check_is_read_only_and_all_six_models_are_required(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    write(args.training_dir / "faithful_seed0/status.json", {"state": "complete", "completed_steps": 100000,
          "requested_steps": 100000, "pid": DEAD_PID})
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(queue, "checkpoint_header", no_access)
    monkeypatch.setattr(queue.validation_summary, "summarize", no_access)
    monkeypatch.setattr(queue, "verify_test_manifest", no_access)
    result = check(args)
    assert result["state"] == "not_ready" and len(result["blocking_reasons"]) == 5
    assert not args.run and not result["test_source_accessed"] and not result["test_manifest_accessed"]
    assert not args.output_dir.exists()
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.parametrize("evidence", ["process", "unknown_inventory", "live_training_lock", "live_model_status", "live_output_child", "invalid_pid"])
def test_activity_blocks_before_any_checkpoint_or_test_access(tmp_path, monkeypatch, evidence):
    args = args_for(tmp_path)
    for model in queue.MODELS:
        write(args.training_dir / model["name"] / "status.json", {"state": "complete", "completed_steps": 100000,
              "requested_steps": 100000, "pid": DEAD_PID})
    scan = lambda: []
    if evidence == "process": scan = lambda: [{"pid": 41, "command": "python -m research.prepare_full_waterdrop"}]
    elif evidence == "unknown_inventory":
        def scan(): raise PermissionError("synthetic ps permission failure")
    elif evidence == "live_training_lock": write(args.training_dir / "queue.lock", {"pid": 41})
    elif evidence == "live_model_status":
        path = args.training_dir / "faithful_seed0/status.json"
        write(path, {**queue.read_json(path), "pid": 41})
    elif evidence == "live_output_child": write(args.output_dir / "queue_status.json", {"pid": DEAD_PID, "child_pid": 41})
    elif evidence == "invalid_pid": write(args.output_dir / "queue.lock", {"pid": None})
    monkeypatch.setattr(queue.validation_summary, "summarize", no_access)
    monkeypatch.setattr(queue, "checkpoint_header", no_access)
    monkeypatch.setattr(queue, "verify_test_manifest", no_access)
    result = queue.preflight(args, scan=scan, alive=lambda pid: pid == 41, clock=lambda: EARLY)
    assert result["state"] == "not_ready" and result["blocking_reasons"]


def test_complete_fixture_passes_full_validation_gate_without_test_access(completed, monkeypatch):
    monkeypatch.setattr(queue, "verify_test_manifest", no_access)
    result = check(completed)
    assert result["state"] == "ready", result["blocking_reasons"]
    assert len(result["models"]) == 6
    assert all(len(row["points"]) == 21 for row in result["validation_evidence"]["runs"])
    assert "recorded_utc" not in result["validation_evidence"]
    assert result["validation_evidence"] == check(completed)["validation_evidence"]
    assert not completed.output_dir.exists() and not completed.raw_test.exists()


@pytest.mark.parametrize("change", ["missing_point", "source_map", "source_hash", "checksum", "variance_floor", "radius_backend", "normalization"])
def test_six_statuses_are_insufficient_without_valid_frozen_evidence(completed, monkeypatch, change):
    folder = completed.training_dir / "faithful_seed0"
    if change == "missing_point":
        (folder / "validation/step-005000.json").unlink()
    elif change in ("source_map", "source_hash"):
        cfg = queue.read_json(folder / "protocol.json")
        if change == "source_map": cfg["source_sha256"].pop(next(iter(cfg["source_sha256"])))
        else: cfg["source_sha256"]["research/full_training.py"] = "0" * 64
        write(folder / "protocol.json", cfg)
    else:
        checkpoint = folder / "checkpoint-100000.pt"
        header = queue.read_json(checkpoint)
        if change == "checksum": header["completed_steps"] = 1
        elif change == "normalization": header["simulator_config"]["normalization_stats"]["velocity"]["std"][0] *= 2
        elif change == "variance_floor": header["simulator_config"]["variance_floor"] = 1e-4
        elif change == "radius_backend": header["simulator_config"]["radius_backend"] = "other"
        write(checkpoint, header)
        if change != "checksum":
            pointer = queue.read_json(folder / "latest.json")
            write(folder / "latest.json", {**pointer, "sha256": queue.sha256(checkpoint)})
    if change in ("missing_point", "source_map", "source_hash"):
        monkeypatch.setattr(queue, "verify_model", no_access)
    result = check(completed)
    assert result["state"] == "invalid" and result["blocking_reasons"]
    assert not completed.output_dir.exists() and not completed.raw_test.exists()


@pytest.mark.parametrize("root", ["training", "data"])
@pytest.mark.parametrize("relationship", ["equal", "descendant", "ancestor"])
def test_output_directory_must_be_fully_disjoint(tmp_path, root, relationship):
    output = tmp_path / root
    if relationship == "descendant": output /= "nested"
    if relationship == "ancestor": output = tmp_path
    with pytest.raises(ValueError, match="disjoint"):
        args_for(tmp_path, "--output-dir", str(output))


def test_deadline_cannot_be_extended_or_naive(tmp_path):
    for deadline in ("2026-10-07T08:00:01+00:00", "2026-10-07T07:00:00"):
        with pytest.raises(SystemExit): args_for(tmp_path, "--deadline-utc", deadline)
    args = args_for(tmp_path)
    result = queue.preflight(args, scan=no_access, header_loader=no_access, clock=lambda: queue.HARD_DEADLINE)
    assert result["state"] == "deadline_reached_incomplete" and not args.output_dir.exists()


def test_all_supervised_children_are_detected_in_process_inventory(monkeypatch):
    modules = sorted(queue.RELEVANT_MODULES)
    lines = [f"{index + 40} python -m {module}" for index, module in enumerate(modules)]
    lines += [f"{index + 100} python /synthetic/{module.rsplit('.', 1)[-1]}.py" for index, module in enumerate(modules)]
    lines += ["900 python -m research.prepare_full_waterdrop --input-dir 'unclosed", "901 unrelated-other-command"]
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "\n".join(lines), ""))
    found = queue.relevant_processes()
    assert len(found) == 2 * len(modules) + 1 and found[-1]["pid"] == 900


@pytest.mark.parametrize("state", ["error", "failed"])
def test_orphan_error_is_not_retried_as_new(tmp_path, state):
    args = args_for(tmp_path)
    job = queue.evaluation_jobs(args, [{**queue.MODELS[0], "checkpoint": "synthetic"}])[0]
    write(job["output"] / "status.json", {"state": state})
    with pytest.raises(ValueError, match="requiring review"):
        queue.validate_evaluation(job, args, {}, {})


def test_orphan_artifacts_or_status_require_review_but_log_only_can_restart(tmp_path):
    args = args_for(tmp_path)
    job = queue.evaluation_jobs(args, [{**queue.MODELS[0], "checkpoint": "synthetic"}])[0]
    write(job["output"] / "supervisor.log", {"synthetic": True})
    assert queue.validate_evaluation(job, args, {}, {}) == "new"
    write(job["output"] / "status.json", {"state": "interrupted"})
    with pytest.raises(ValueError, match="Orphan evaluation status"):
        queue.validate_evaluation(job, args, {}, {})
    (job["output"] / "status.json").unlink()
    write(job["output"] / "trajectory_000003_base.json", {"synthetic": True})
    with pytest.raises(ValueError, match="Orphan"):
        queue.validate_evaluation(job, args, {}, {})


def test_no_child_starts_when_deadline_passes_during_prelaunch(tmp_path):
    times = iter([EARLY, queue.HARD_DEADLINE])
    result = queue.supervise(["synthetic"], tmp_path / "log", queue.HARD_DEADLINE,
                             clock=lambda: next(times), scan=lambda: [], popen=no_access)
    assert result == {"state": "deadline_reached_incomplete", "started": False}


def test_real_inert_child_is_reaped_at_deadline_and_partial_artifact_survives(tmp_path):
    marker = tmp_path / "partial.json"
    captured = []
    command = [sys.executable, "-c", "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('partial'); time.sleep(60)", str(marker)]
    def launch(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        captured.append(child)
        until = time.monotonic() + 10
        while not marker.exists() and child.poll() is None and time.monotonic() < until: time.sleep(.01)
        assert marker.exists()
        return child
    result = queue.supervise(command, tmp_path / "log", queue.HARD_DEADLINE,
                             clock=lambda: queue.HARD_DEADLINE if captured else EARLY,
                             scan=lambda: [], popen=launch, on_start=lambda pid: write(tmp_path / "pid.json", {"pid": pid}))
    assert result["state"] == "deadline_reached_incomplete" and result["started"]
    assert captured[0].poll() is not None and marker.read_text() == "partial"
    assert queue.read_json(tmp_path / "pid.json")["pid"] == captured[0].pid


def test_concurrent_training_reaps_own_child_without_signalling_other_process(tmp_path, monkeypatch):
    calls = []
    class Child:
        pid, returncode = 21, None
        def poll(self): return self.returncode
    child = Child()
    scans = iter([[], [{"pid": 21, "command": "own child"}, {"pid": 41, "command": "other trainer"}]])
    def stop(owned):
        assert owned is child
        calls.append(owned.pid)
        owned.returncode = -2
    monkeypatch.setattr(queue.reviewed, "stop_child", stop)
    with pytest.raises(ValueError, match="Concurrent"):
        queue.supervise(["synthetic"], tmp_path / "log", queue.HARD_DEADLINE,
                        clock=lambda: EARLY, scan=lambda: next(scans), popen=lambda *a, **k: child)
    assert calls == [21]


def test_pid_publication_failure_still_reaps_child(tmp_path, monkeypatch):
    class Child:
        pid, returncode = 21, None
        def poll(self): return self.returncode
    child = Child()
    monkeypatch.setattr(queue.reviewed, "stop_child", lambda owned: setattr(owned, "returncode", -2))
    with pytest.raises(AssertionError, match="Protected"):
        queue.supervise(["synthetic"], tmp_path / "log", queue.HARD_DEADLINE,
                        clock=lambda: EARLY, scan=lambda: [], popen=lambda *a, **k: child, on_start=no_access)
    assert child.poll() == -2


@pytest.mark.parametrize("change", [None, "embedded_metadata", "source", "duplicate", "escape", "array_hash", "dtype"])
def test_numeric_manifest_gate_uses_only_safe_synthetic_arrays(tmp_path, change):
    import numpy as np
    args = args_for(tmp_path)
    metadata = {"bounds": [[0, 1], [0, 1]], "dim": 2}
    write(args.data_dir / "metadata.json", metadata)
    for split in ("train", "valid"):
        write(args.data_dir / (split + ".json"), {"records": []})
    records = []
    for index in range(30):
        descriptors = {}
        for key, array in (("positions", np.full((1001, 1, 2), index, dtype=np.float32)),
                           ("particle_types", np.array(5, dtype=np.int64))):
            path = args.data_dir / "test" / f"{index:06d}_{key}.npy"
            path.parent.mkdir(exist_ok=True)
            np.save(path, array, allow_pickle=False)
            descriptors[key] = {"path": str(path.relative_to(args.data_dir)), "shape": list(array.shape),
                                "dtype": str(array.dtype), "sha256": queue.sha256(path)}
        records.append({"id": f"test:{index:06d}", "source_index": index, **descriptors,
                        "trajectory_content_sha256": queue.reviewed.config_hash(["synthetic", index])})
    digest = queue.sha256(args.data_dir / "metadata.json")
    manifest = {"format": "gns-trajectory-manifest", "version": 1, "split": "test", "record_count": 30,
                "source": {"sha256": queue.SOURCE_SHA256, "record_count": 30, "CRC_verified": True},
                "records": records, "metadata": metadata, "metadata_sha256": digest,
                "converter_sha256": queue.sha256(queue.REPO / "research/prepare_full_waterdrop.py")}
    if change == "embedded_metadata": manifest["metadata"] = {"dim": 3}
    elif change == "source": manifest["source"]["sha256"] = "0" * 64
    elif change == "duplicate": records[1]["trajectory_content_sha256"] = records[0]["trajectory_content_sha256"]
    elif change == "escape": records[0]["positions"]["path"] = "../outside.npy"
    elif change == "array_hash": records[0]["positions"]["sha256"] = "0" * 64
    elif change == "dtype": records[0]["positions"]["dtype"] = "float64"
    write(args.data_dir / "test.json", manifest)
    if change is None:
        identity = queue.verify_test_manifest(args, digest)
        assert identity["record_count"] == 30 and identity["evaluated_source_indices"] == list(range(3, 30))
        assert identity["manifest_sha256"] == queue.sha256(args.data_dir / "test.json")
    else:
        with pytest.raises(ValueError): queue.verify_test_manifest(args, digest)
    assert not args.raw_test.exists()


def test_synthetic_queue_interruption_resume_and_fixed_order_hold_all_locks(completed, monkeypatch):
    args = completed
    args.raw_test.parent.mkdir()
    args.raw_test.write_bytes(b"synthetic source; never a real dataset")
    write(args.data_dir / "test.json", {"synthetic": True})
    original_sha = queue.sha256
    monkeypatch.setattr(queue, "sha256", lambda path: queue.SOURCE_SHA256 if Path(path) == args.raw_test else original_sha(path))
    monkeypatch.setattr(queue, "now", lambda: EARLY)
    monkeypatch.setattr(queue, "checkpoint_header", queue.read_json)
    monkeypatch.setattr(queue, "relevant_processes", lambda: [])
    monkeypatch.setattr(queue.reviewed, "process_alive", lambda pid: False)
    sources = queue.source_snapshot()
    launches = []
    interrupted = False
    def locks_owned():
        assert queue.read_json(args.output_dir / "queue.lock")["pid"] == os.getpid()
        assert queue.read_json(args.training_dir / "queue.lock")["pid"] == os.getpid()
        assert all(queue.read_json(args.training_dir / row["name"] / "run.lock")["pid"] == os.getpid() for row in queue.MODELS)
    def manifest(arg, metadata):
        locks_owned()
        return {"manifest_sha256": "fixture-manifest", "metadata_sha256": metadata, "source_sha256": queue.SOURCE_SHA256}
    monkeypatch.setattr(queue, "verify_test_manifest", manifest)
    def validate(job, *unused):
        marker = job["output"] / "synthetic-result.json"
        return queue.read_json(marker)["state"] if marker.exists() else "new"
    monkeypatch.setattr(queue, "validate_evaluation", validate)
    def supervise(command, log, deadline, on_start):
        nonlocal interrupted
        locks_owned()
        on_start(12345678)
        assert queue.read_json(args.output_dir / "queue_status.json")["child_pid"] == 12345678
        launches.append(command)
        module = command[2]
        if module in ("research.full_rollout", "research.full_same_state"):
            output = Path(command[command.index("--output-dir") + 1])
            if len(launches) == 3 and not interrupted:
                interrupted = True
                write(output / "synthetic-result.json", {"state": "resume", "preserved": True})
                return {"state": "deadline_reached_incomplete", "started": True, "child_pid": 12345678}
            write(output / "synthetic-result.json", {"state": "complete", "scientific_guard_failure": True})
        else:
            prefix = Path(command[command.index("--output-prefix") + 1])
            write(prefix.with_suffix(".json"), {"state": "complete"})
        return {"state": "complete", "started": True, "child_pid": 12345678}
    monkeypatch.setattr(queue, "supervise", supervise)
    first = queue.run_queue(args)
    assert first["state"] == "deadline_reached_incomplete" and len(launches) == 3
    before = original_sha(args.output_dir / "rollout/faithful_seed0/synthetic-result.json")
    second = queue.run_queue(args)
    assert second["state"] == "complete"
    assert original_sha(args.output_dir / "rollout/faithful_seed0/synthetic-result.json") == before
    assert len(launches) == 15  # One interrupted launch + twelve evaluations + two summaries.
    assert [cmd[2] for cmd in launches] == ["research.full_rollout"] * 7 + ["research.full_same_state"] * 6 + [
        "research.summarize_full_rollouts", "research.summarize_full_same_state"]
    assert "--resume" in launches[3] and all("--resume" not in cmd for i, cmd in enumerate(launches) if i != 3)
    assert queue.read_json(args.output_dir / "queue_protocol.json")["source_sha256"] == sources
    assert not (args.output_dir / "queue.lock").exists() and not (args.training_dir / "queue.lock").exists()
    assert all(not (args.training_dir / row["name"] / "run.lock").exists() for row in queue.MODELS)
    # Completed scientific guard failures are reused; only pure summaries repeat.
    before_count = len(launches)
    assert queue.run_queue(args)["state"] == "complete"
    assert [cmd[2] for cmd in launches[before_count:]] == ["research.summarize_full_rollouts", "research.summarize_full_same_state"]
    write(args.output_dir / "queue_status.json", {"state": "failed", "pid": DEAD_PID})
    with pytest.raises(ValueError, match="requires review"):
        queue.run_queue(args)
    assert len(launches) == before_count + 2


def test_strict_loader_retains_completed_guard_failure_as_reusable_result(tmp_path, monkeypatch):
    from research.tests import test_summarize_full_rollouts as fixture
    args = args_for(tmp_path)
    old_protocol = fixture.protocol
    def protocol(objective, seed):
        return {**old_protocol(objective, seed), "threads": 2, "runtime": {"device": "mps", "mps_fallback_environment": "0"}}
    monkeypatch.setattr(fixture, "protocol", protocol)
    folder = args.output_dir / "rollout/faithful_seed0"
    fixture.build_run(folder, "faithful", 0)
    def fail(row):
        row.update(status="failed", failure={"category": "coordinate_resource_guard", "forecast_step": 251},
                   completed_steps=250, mean_rollout_mse=None, mse_at_final_horizon=None, mean_directed_edges=None,
                   forecast_network_passes=251, total_network_passes=251)
        for name in ("mse_per_step", "directed_edges_per_step", "predicted_boundary_per_step", "ground_truth_boundary_per_step"):
            row[name] = row[name][:250]
        row["mse_at_steps"] = {str(step): row["mse_per_step"][step - 1] if step <= 250 else None for step in queue.rollout_summary.TRACE_STEPS}
    fixture.change_row(folder, 3, "base", fail)
    original_sha = queue.sha256
    monkeypatch.setattr(queue, "sha256", lambda path: fixture.PROTOCOL_HASH if path == args.protocol else original_sha(path))
    provenance = queue.read_json(folder / "protocol.json")
    model = {**queue.MODELS[0], "checkpoint": "synthetic", "checkpoint_sha256": provenance["checkpoint_sha256"]}
    job = queue.evaluation_jobs(args, [model])[0]
    identity = {key: provenance[key] for key in ("manifest_sha256", "metadata_sha256")}
    before = {p.name: original_sha(p) for p in folder.iterdir() if p.is_file()}
    assert queue.validate_evaluation(job, args, identity, provenance["code_sha256"]) == "complete"
    assert before == {p.name: original_sha(p) for p in folder.iterdir() if p.is_file()}
    saved = fixture.read(folder / "result.json")
    saved["state"] = "partial"
    fixture.commit_result(folder, saved)
    assert queue.validate_evaluation(job, args, identity, provenance["code_sha256"]) == "resume"


def test_sigterm_reaps_inert_child_before_releasing_temporary_lock(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text('''from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
import os, sys
from research import run_evaluation_queue as q
root = Path(sys.argv[1])
lock = root / "owned.lock"
token = q.reviewed.acquire_lock(lock)
try:
    with q.interruption_handlers():
        with ExitStack() as stack:
            stack.callback(q.release_lock, lock, token)
            command = [sys.executable, "-c", "import time; time.sleep(60)"]
            q.supervise(command, root / "child.log", q.now() + timedelta(seconds=30), scan=lambda: [],
                        on_start=lambda pid: q.atomic_json(root / "ready.json", {"pid": pid}))
except KeyboardInterrupt:
    q.atomic_json(root / "cleaned.json", {"lock_removed": not lock.exists()})
''')
    env = dict(os.environ, PYTHONPATH=str(queue.REPO) + os.pathsep + str(queue.REPO / "adaptive-gns"))
    worker = subprocess.Popen([sys.executable, str(script), str(tmp_path)], cwd=queue.REPO, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid = None
    try:
        until = time.monotonic() + 15
        while not (tmp_path / "ready.json").exists() and worker.poll() is None and time.monotonic() < until: time.sleep(.02)
        assert (tmp_path / "ready.json").exists(), worker.communicate(timeout=2)
        child_pid = queue.read_json(tmp_path / "ready.json")["pid"]
        assert (tmp_path / "owned.lock").exists()
        worker.send_signal(signal.SIGTERM)
        stdout, stderr = worker.communicate(timeout=15)
        assert worker.returncode == 0, (stdout, stderr)
        assert queue.read_json(tmp_path / "cleaned.json")["lock_removed"]
        assert not queue.reviewed.process_alive(child_pid)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=5)
        if child_pid is not None and queue.reviewed.process_alive(child_pid): os.kill(child_pid, signal.SIGKILL)
