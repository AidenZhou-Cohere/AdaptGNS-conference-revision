"""Read-only/synthetic operational review; never launches a real process."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pytest


LAUNCHER = Path(__file__).with_name("run_graph_support_queue.py")
spec = importlib.util.spec_from_file_location("reviewed_support_queue", LAUNCHER)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.fixture(autouse=True)
def isolated_queue_paths(tmp_path, monkeypatch):
    repo = tmp_path / "outputs/AdaptGNS"
    launcher = tmp_path / "work/deadline_research_20261005/run_graph_support_queue.py"
    launcher.parent.mkdir(parents=True, exist_ok=True); launcher.write_bytes(b"synthetic launcher source")
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "REPO", repo)
    monkeypatch.setattr(queue, "OUTPUT", tmp_path / "new-output")
    monkeypatch.setattr(queue, "PRIOR", tmp_path / "prior")
    monkeypatch.setattr(queue, "FREEZE", launcher.with_name("freeze.json"))
    monkeypatch.setattr(queue, "__file__", str(launcher))
    monkeypatch.setattr(queue, "OWNED_OUTPUT", False)
    for name in queue.NAMES:
        path = repo / f"research/results/full_waterdrop_100k/{name}/checkpoint-100000.pt"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(("synthetic checkpoint " + name).encode())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prior_fixture(tmp_path):
    directory = tmp_path / "prior"
    jobs = []
    policies = ("base", "dense", "random25", "speed25", "laggedrisk25")
    for name in queue.NAMES:
        objective, seed_text = name.split("_seed"); seed = int(seed_text)
        base = directory / name; base.mkdir(parents=True)
        checkpoint = queue.REPO / f"research/results/full_waterdrop_100k/{name}/checkpoint-100000.pt"
        protocol = {"objective": objective, "seed": seed, "checkpoint_sha256": queue.sha256(checkpoint),
            "source_indices": list(range(3, 30)), "trajectory_ids": [f"test:{index:06d}" for index in range(3, 30)],
            "policies": list(policies), "horizon": 995}
        write(base / "protocol.json", protocol)
        digest = queue.sha256(base / "protocol.json")
        rows = []
        for index in range(3, 30):
            for policy in policies:
                stem = f"trajectory_{index:06d}_{policy}"
                trace = base / (stem + ".npz"); trace.write_bytes(b"synthetic immutable trace bytes")
                row = {"trajectory_id": f"test:{index:06d}", "source_index": index, "policy": policy,
                    "status": "complete", "failure": None, "horizon": 995, "n_particles": 2,
                    "rng_seed": 93000 + 1000 * seed + index, "completed_steps": 995,
                    "mse_at_steps": {str(step): 1. for step in (1, 10, 100, 250, 500, 995)},
                    "mse_at_final_horizon": 1., "mean_rollout_mse": 1., "mean_directed_edges": 4.,
                    "total_wall_seconds": .01, "total_network_passes": 995 + (policy == "laggedrisk25"), "forecast_network_passes": 995,
                    "trace_file": trace.name, "trace_sha256": queue.sha256(trace), "protocol_sha256": digest,
                    "mse_per_step": [1.] * 995}
                path = base / (stem + ".json"); write(path, row)
                compact = {key: value for key, value in row.items() if key not in ("mse_per_step", "protocol_sha256")}
                rows.append({**compact, "record_file": path.name, "record_sha256": queue.sha256(path)})
        result = {"state": "complete", "objective": objective, "seed": seed,
            "checkpoint_sha256": protocol["checkpoint_sha256"], "protocol_sha256": digest, "records": rows}
        write(base / "result.json", result)
        write(base / "status.json", {"state": "complete", "result_sha256": queue.sha256(base / "result.json")})
        jobs.append({"name": name, "state": "complete", "returncode": 0})
    write(directory / "queue_status.json", {"state": "complete", "jobs": jobs})
    return directory


def reseal_child(base, result):
    write(base / "result.json", result)
    write(base / "status.json", {"state": "complete", "result_sha256": queue.sha256(base / "result.json")})


def test_prior_complete_accepts_exact_six_model_publication(tmp_path):
    directory = prior_fixture(tmp_path)
    assert queue.prior_complete(directory)["state"] == "complete"


@pytest.mark.parametrize("defect", ["queue_lock", "child_lock", "missing_job", "duplicate_job", "running_job", "nonzero_job", "bad_result_hash"])
def test_native_prerequisite_lock_cohort_process_and_hash_refusals(tmp_path, defect):
    directory = prior_fixture(tmp_path)
    status_path = directory / "queue_status.json"; status = json.loads(status_path.read_text())
    if defect == "queue_lock":
        (directory / "run.lock").write_text("owned")
    elif defect == "child_lock":
        (directory / queue.NAMES[0] / "run.lock").write_text("owned")
    elif defect == "missing_job":
        status["jobs"].pop()
    elif defect == "duplicate_job":
        status["jobs"][0] = status["jobs"][1]
    elif defect == "running_job":
        status["jobs"][0]["state"] = "running"
    elif defect == "nonzero_job":
        status["jobs"][0]["returncode"] = 1
    else:
        (directory / queue.NAMES[0] / "result.json").write_text("tampered")
    write(status_path, status)
    with pytest.raises(RuntimeError):
        queue.prior_complete(directory)


@pytest.mark.parametrize("defect", ["empty_records", "missing_outcome", "duplicate_outcome", "wrong_seed", "wrong_protocol", "partial_result"])
def test_checksum_consistent_hostile_native_results_are_not_completion(tmp_path, defect):
    directory = prior_fixture(tmp_path); base = directory / queue.NAMES[0]
    result = json.loads((base / "result.json").read_text())
    if defect == "empty_records": result["records"] = []
    elif defect == "missing_outcome": result["records"].pop()
    elif defect == "duplicate_outcome": result["records"][0] = result["records"][1]
    elif defect == "wrong_seed": result["seed"] = 99
    elif defect == "wrong_protocol": result["protocol_sha256"] = "wrong"
    else: result["state"] = "partial"
    reseal_child(base, result)
    with pytest.raises(RuntimeError):
        queue.prior_complete(directory)


@pytest.mark.parametrize("defect", ["wrong_horizon", "running_record", "changed_row_identity"])
def test_self_consistent_invalid_outcome_payloads_refused(tmp_path, defect):
    directory = prior_fixture(tmp_path); base = directory / queue.NAMES[0]
    result = json.loads((base / "result.json").read_text()); compact = result["records"][0]
    path = base / compact["record_file"]; row = json.loads(path.read_text())
    if defect == "wrong_horizon": row["horizon"] = compact["horizon"] = 10
    elif defect == "running_record": row["status"] = compact["status"] = "running"
    else: row["trajectory_id"] = "another source"
    write(path, row); compact["record_sha256"] = queue.sha256(path)
    reseal_child(base, result)
    with pytest.raises((RuntimeError, ValueError)):
        queue.prior_complete(directory)


def test_committed_guard_failure_counts_as_completed_scientific_outcome(tmp_path):
    directory = prior_fixture(tmp_path); base = directory / queue.NAMES[0]
    result = json.loads((base / "result.json").read_text()); compact = result["records"][0]
    path = base / compact["record_file"]; row = json.loads(path.read_text())
    fields = {"status": "failed", "failure": {"category": "coordinate_guard", "forecast_step": 3}, "completed_steps": 2,
              "mean_rollout_mse": None, "mean_directed_edges": None, "mse_at_final_horizon": None}
    row.update(fields); compact.update(fields); row["mse_per_step"] = [1., 1.]
    write(path, row); compact["record_sha256"] = queue.sha256(path); reseal_child(base, result)
    assert queue.prior_complete(directory)["state"] == "complete"


def test_heavy_module_matching_is_exact_and_excludes_only_own_pid():
    process_text = "\n".join([
        "10 /venv/python -u -m research.full_training --seed 0",
        "11 /venv/python -m research.faithful_graph_support --arm mix",
        "12 /venv/python -m research.continuation_evaluation --mode observed",
        "13 /venv/python -m research.full_training_extra",
        "14 /bin/echo research.full_training",
        "15 /venv/python -m research.native_graph_rollout",
    ])
    assert [int(row.split()[0]) for row in queue.heavy_processes(process_text, 15)] == [10, 11, 12]


def test_between_job_supervisors_are_heavy_even_without_a_child():
    processes = "\n".join([
        "51 /venv/python -m research.run_evaluation_queue",
        "52 /venv/python /work/run_graph_bridge_queue.py",
        "53 /venv/python /work/run_native_rollout_queue.py",
        "54 /venv/python /work/run_graph_support_queue.py --launch",
        "55 /venv/python /work/run_graph_support_queue.py --launch",
    ])
    assert [int(row.split()[0]) for row in queue.heavy_processes(processes, 55)] == [51, 52, 53, 54]


def test_six_commands_pair_parent_data_schedule_and_mps_settings():
    commands = [queue.command(seed, arm, executable="/synthetic/python") for seed in range(3) for arm in ("base", "mix")]
    assert len(commands) == 6
    for seed in range(3):
        left, right = commands[2 * seed:2 * seed + 2]
        assert left[:4] == right[:4] == ["/synthetic/python", "-u", "-m", "research.faithful_graph_support"]
        parsed = [{cmd[index]: cmd[index + 1] for index in range(4, len(cmd), 2)} for cmd in (left, right)]
        assert parsed[0]["--arm"] == "base" and parsed[1]["--arm"] == "mix"
        assert parsed[0]["--seed"] == parsed[1]["--seed"] == str(seed)
        for key in ("--parent-checkpoint", "--train-manifest", "--valid-manifest", "--metadata", "--device", "--threads"):
            assert parsed[0][key] == parsed[1][key]
        assert parsed[0]["--device"] == "mps" and parsed[0]["--threads"] == "2"
        assert "--resume" not in left and "--clear-stale-lock" not in left


def test_check_refusal_has_no_directory_or_process_side_effects(tmp_path, monkeypatch):
    output = tmp_path / "new-output"
    monkeypatch.setattr(queue, "OUTPUT", output)
    monkeypatch.setattr(queue, "PRIOR", tmp_path / "missing-prior")
    monkeypatch.setattr(queue, "LATEST_START", datetime(2100, 1, 1, tzinfo=timezone.utc))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("launched process during check")))
    monkeypatch.setattr(sys, "argv", [str(LAUNCHER), "--check"])
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        queue.main()
    assert not output.exists()
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


def test_empty_freeze_inventory_cannot_pass_read_only_check(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze.json"; write(freeze, {"training_commit": queue.TRAINING_COMMIT, "files_sha256": {}})
    monkeypatch.setattr(queue, "FREEZE", freeze)
    monkeypatch.setattr(queue, "OUTPUT", tmp_path / "uncreated")
    monkeypatch.setattr(queue, "LATEST_START", datetime(2100, 1, 1, tzinfo=timezone.utc))
    monkeypatch.setattr(queue, "prior_complete", lambda path: {"state": "complete"})
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: "")
    with pytest.raises(RuntimeError):
        queue.preflight()
    assert not queue.OUTPUT.exists()


def test_exact_freeze_inventory_detects_missing_extra_and_changed_source(tmp_path):
    for relative in queue.required_frozen_files():
        path = queue.ROOT / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"synthetic frozen bytes")
    frozen = {"training_commit": queue.TRAINING_COMMIT,
              "files_sha256": {relative: queue.sha256(queue.ROOT / relative) for relative in queue.required_frozen_files()}}
    queue.verify_freeze(frozen)
    missing = {**frozen, "files_sha256": dict(frozen["files_sha256"])}; missing["files_sha256"].pop(next(iter(missing["files_sha256"])))
    with pytest.raises(RuntimeError, match="inventory"):
        queue.verify_freeze(missing)
    extra = {**frozen, "files_sha256": {**frozen["files_sha256"], "unexpected.py": "wrong"}}
    with pytest.raises(RuntimeError, match="inventory"):
        queue.verify_freeze(extra)
    (queue.ROOT / next(iter(frozen["files_sha256"]))).write_bytes(b"changed bytes")
    with pytest.raises(RuntimeError, match="changed"):
        queue.verify_freeze(frozen)
