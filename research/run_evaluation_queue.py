"""Supervise the fixed post-training evaluations; default is read-only preflight.

No test-source/manifest access occurs in check-only mode or before all six final
models pass preflight. --run holds the reviewed training queue/model locks as an
exclusion gate, then runs conversion, twelve evaluations and two summaries in a
fixed order. Numerical/protocol errors require review; guard-failed scientific
outcomes are completed results and are never rerun for a preferable outcome.
"""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from research import run_full_queue as reviewed
from research.full_training import RunLock, atomic_json
from research import summarize_full_rollouts as rollout_summary
from research import summarize_full_same_state as diagnostic_summary
from research import summarize_training_validation as validation_summary

REPO = Path(__file__).resolve().parents[1]
HARD_DEADLINE = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)
MODELS = tuple({"objective": objective, "seed": seed, "name": f"{objective}_seed{seed}"}
               for seed in (0, 1, 2) for objective in ("faithful", "nll"))
SOURCE_SHA256 = rollout_summary.OFFICIAL_TEST_SHA256
RELEVANT_MODULES = {"research.full_training", "research.run_full_queue", "research.full_rollout", "research.full_same_state",
                    "research.prepare_full_waterdrop", "research.summarize_full_rollouts", "research.summarize_full_same_state"}
sha256, require = reviewed.sha256, rollout_summary.require


def now():
    return datetime.now(timezone.utc)


def read_json(path):
    return rollout_summary.read_json(path)[0]


def checked_pid(value):
    require(type(value) is int and value > 0, "Unknown/invalid PID; inactivity cannot be established")
    return value


def relevant_processes():
    """Read process arguments; a failed inventory is an unknown, never idle."""
    result = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid, command = int(fields[0]), fields[1]
        try:
            words = shlex.split(command)
        except ValueError:
            words = command.split()  # Truncated/unmatched quotes must not hide a live child.
        modules = [words[index + 1] for index, word in enumerate(words[:-1]) if word == "-m"]
        scripts = {Path(word).name for word in words if word.endswith(".py")}
        if set(modules) & RELEVANT_MODULES or scripts & {name.rsplit(".", 1)[-1] + ".py" for name in RELEVANT_MODULES}:
            found.append({"pid": pid, "command": command})
    return found


def activity(training_dir, ignore_pids=(), scan=None, alive=None, output_dir=None):
    scan = relevant_processes if scan is None else scan
    alive = reviewed.process_alive if alive is None else alive
    ignored = set(ignore_pids)
    reasons = []
    try:
        reasons.extend(f"Active training/evaluation process {row['pid']}: {row['command']}"
                       for row in scan() if row["pid"] not in ignored)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        reasons.append(f"Process inventory unknown: {type(error).__name__}: {error}")
    paths = [(training_dir / "queue.lock", ("pid",)), (training_dir / "queue_status.json", ("pid", "child_pid"))]
    if output_dir is not None:
        paths += [(output_dir / "queue.lock", ("pid",)), (output_dir / "queue_status.json", ("pid", "child_pid"))]
    paths += [(training_dir / model["name"] / name, ("pid",)) for model in MODELS for name in ("run.lock", "status.json")]
    for path, fields in paths:
        if not path.exists():
            continue
        try:
            value = read_json(path)
            for field in fields:
                if field not in value and field == "child_pid":
                    continue
                pid = checked_pid(value.get(field))
                if pid not in ignored and alive(pid):
                    reasons.append(f"Live or permission-unknown PID {pid} recorded by {path.name} ({field})")
        except (ValueError, KeyError, TypeError, OSError) as error:
            reasons.append(f"Cannot establish inactivity for {path}: {error}")
    return reasons


def checkpoint_header(path):
    """CPU-only header validation; reached only after six complete statuses."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {name: payload.get(name) for name in ("format_version", "full_training_schema", "completed_steps",
            "training_config", "run_config", "run_config_sha256", "simulator_config")}


def numeric_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def verify_model(args, model, status, header_loader=None):
    output = args.training_dir / model["name"]
    pointer = read_json(output / "latest.json")
    require(pointer.get("path") == "checkpoint-100000.pt", "Only the exact final 100k checkpoint is eligible")
    # Reuse the reviewed final-budget, config, data, source and checksum checks.
    reviewed.verify_completed(output, model, args, status)
    config = read_json(output / "protocol.json")
    require(config.get("steps") == 100000 and config.get("batch_size") == 2 and config.get("history") == 6
            and config.get("noise_std") == 6.7e-4 and config.get("architecture") == {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
            "Training recipe differs from fixed full experiment")
    require(config["runtime"].get("device") == "mps" and config["runtime"].get("mps_fallback_environment") == "0",
            "Training was not the prescribed native MPS run with fallback disabled")
    for split, count, digest in (("train", 1000, rollout_summary.OFFICIAL_TRAIN_SHA256), ("valid", 30, rollout_summary.OFFICIAL_VALID_SHA256)):
        require(config[split].get("n_trajectories") == count and config[split]["source"].get("sha256") == digest,
                f"Training {split} source provenance differs")
    path = output / pointer["path"]
    header = (checkpoint_header if header_loader is None else header_loader)(path)
    require(header.get("format_version") == 2 and header.get("full_training_schema") == 1
            and header.get("completed_steps") == 100000 and header.get("run_config") == config
            and header.get("run_config_sha256") == reviewed.config_hash(config), "Checkpoint header/config/budget differs")
    training = header["training_config"]
    require(training.get("loss") == model["objective"] and training.get("completed_optimizer_updates") == 100000
            and training.get("full_run") == config, "Checkpoint objective or full-run provenance differs")
    simulator = header["simulator_config"]
    required = {"particle_dimensions": 2, "latent_dim": 128, "nmessage_passing_steps": 10, "nmlp_layers": 2,
                "mlp_hidden_dim": 128, "uncertainty_parameterization": "variance", "connectivity_radius": .015,
                "variance_floor": 1e-6, "max_num_neighbors": 128, "radius_backend": "scipy_host",
                "detach_variance_features": model["objective"] == "faithful"}
    require(all(simulator.get(key) == value for key, value in required.items()), "Checkpoint simulator architecture/variance semantics differ")
    metadata = read_json(args.data_dir / "metadata.json")
    require(simulator.get("boundaries") == metadata["bounds"], "Checkpoint boundaries differ from metadata")
    for name, prefix in (("acceleration", "acc"), ("velocity", "vel")):
        stats = simulator["normalization_stats"][name]
        actual_mean, actual_std = numeric_list(stats["mean"]), numeric_list(stats["std"])
        expected_std = [math.sqrt(value * value + (6.7e-4) ** 2) for value in metadata[prefix + "_std"]]
        require(len(actual_mean) == len(actual_std) == 2
                and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12) for a, b in zip(actual_mean, metadata[prefix + "_mean"]))
                and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12) for a, b in zip(actual_std, expected_std)),
                "Saved training normalization differs from original noise-adjusted scales")
    return {**model, "checkpoint": str(path), "checkpoint_sha256": pointer["sha256"],
            "run_config_sha256": reviewed.config_hash(config), "metadata_sha256": config["metadata_sha256"],
            "common_training_sha256": reviewed.config_hash({key: value for key, value in config.items() if key not in ("objective", "seed")})}


def preflight(args, ignore_pids=(), scan=None, alive=None, header_loader=None, clock=None):
    clock = now if clock is None else clock
    result = {"schema": 1, "state": "not_ready", "test_source_accessed": False, "test_manifest_accessed": False,
              "deadline_utc": args.deadline.isoformat(), "models": [], "blocking_reasons": []}
    if clock() >= args.deadline:
        result.update(state="deadline_reached_incomplete", blocking_reasons=["Hard cutoff reached; no new jobs may start"])
        return result
    reasons = activity(args.training_dir, ignore_pids, scan, alive, args.output_dir)
    statuses = []
    for model in MODELS:
        path = args.training_dir / model["name"] / "status.json"
        try:
            value = read_json(path) if path.exists() else {}
            ready = value.get("state") == "complete" and value.get("completed_steps") == 100000 and value.get("requested_steps") == 100000
            result["models"].append({**model, "state": value.get("state", "missing"), "completed_steps": value.get("completed_steps"), "final_status_ready": ready})
            statuses.append(value)
        except (ValueError, TypeError, OSError) as error:
            reasons.append(f"Unreadable {model['name']} status: {error}")
            result["models"].append({**model, "state": "invalid", "final_status_ready": False})
            statuses.append({})
    if reasons or not all(model["final_status_ready"] for model in result["models"]):
        result["blocking_reasons"] = reasons + [f"{model['name']} is not a completed 100k model" for model in result["models"] if not model["final_status_ready"]]
        return result  # No checkpoint loading and no test access on this path.
    verified = []
    try:
        validation = validation_summary.summarize(args.training_dir, sha256(args.protocol), REPO)
        require(validation.get("current_frozen_source_files_verified") is True
                and {(row["objective"], row["seed"]) for row in validation["runs"]}
                    == {(row["objective"], row["seed"]) for row in MODELS}
                and len(validation["runs"]) == 6, "Validation history does not verify the six frozen models")
        for row in validation["runs"]:
            require(row["state"] == "complete" and not row["missing_scheduled_updates"]
                    and [point["completed_steps"] for point in row["points"]] == list(range(0, 100001, 5000)),
                    "All 21 scheduled validation points are required for every final model")
        # Persist only stable evidence. The report's generation timestamp changes
        # on every read and must not invalidate an otherwise identical resume.
        result["validation_evidence"] = {"source_sha256": validation["source_sha256"], "runs": [
            {key: row[key] for key in ("objective", "seed", "run_config_sha256", "protocol_file_sha256", "status_file_sha256", "points")}
            for row in validation["runs"]]}
        for model, status in zip(MODELS, statuses):
            verified.append(verify_model(args, model, status, header_loader))
        require(len({model["checkpoint_sha256"] for model in verified}) == 6, "Checkpoint checksum reused across different models")
        require(len({model["common_training_sha256"] for model in verified}) == 1, "Common training/data/source/runtime provenance differs")
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as error:
        result.update(state="invalid", blocking_reasons=[f"Final-model verification failed: {error}"])
        return result
    if clock() >= args.deadline:
        result.update(state="deadline_reached_incomplete", blocking_reasons=["Cutoff reached during final-model verification"])
        return result
    result.update(state="ready", models=verified)
    return result


def source_snapshot():
    names = ("run_evaluation_queue.py", "run_full_queue.py", "full_training.py", "prepare_full_waterdrop.py",
             "full_rollout.py", "full_same_state.py", "budget_graph.py", "summarize_full_rollouts.py", "summarize_full_same_state.py",
             "summarize_training_validation.py")
    paths = [REPO / "research" / name for name in names] + sorted((REPO / "adaptive-gns/gns").glob("*.py"))
    return {str(path.relative_to(REPO)): sha256(path) for path in paths}


def evaluation_jobs(args, models):
    result = []
    for kind in ("rollout", "same_state"):
        for model in models:
            output = args.output_dir / kind / model["name"]
            module = "research.full_rollout" if kind == "rollout" else "research.full_same_state"
            command = [sys.executable, "-m", module, "--manifest", str(args.data_dir / "test.json"),
                       "--checkpoint", model["checkpoint"], "--output-dir", str(output), "--scope", "locked_test",
                       "--device", "mps", "--threads", "2", "--clear-stale-lock"]
            if kind == "rollout":
                command += ["--protocol", str(args.protocol), "--horizon", "995", "--policies", *rollout_summary.POLICIES,
                            "--expected-test-source-sha256", SOURCE_SHA256,
                            "--start-index", "3", "--rng-seed", "93000", "--max-candidate-pairs", "100000", "--max-abs-coordinate", "10"]
            else:
                command += ["--protocol", str(args.companion_protocol), "--training-protocol", str(args.protocol),
                            "--target-frames", *map(str, diagnostic_summary.TARGET_FRAMES), "--repeats", "6",
                            "--max-candidate-pairs", "100000", "--max-abs-coordinate", "10"]
            result.append({"name": kind + "/" + model["name"], "kind": kind, "model": model, "output": output, "command": command})
    return result


def verify_test_manifest(args, expected_metadata_hash):
    """Reached only inside the six-model exclusion gate, never check-only."""
    path = args.data_dir / "test.json"
    manifest = read_json(path)
    require(manifest.get("format") == "gns-trajectory-manifest" and manifest.get("version") == 1 and manifest.get("split") == "test"
            and manifest.get("record_count") == 30 and len(manifest.get("records", [])) == 30,
            "Test manifest is not complete numeric format-v1 with 30 records")
    require(manifest["source"].get("sha256") == SOURCE_SHA256 and manifest["source"].get("record_count") == 30
            and manifest["source"].get("CRC_verified") is True and manifest.get("metadata_sha256") == expected_metadata_hash,
            "Test source/CRC/metadata provenance differs")
    require(manifest.get("metadata") == read_json(args.data_dir / "metadata.json"), "Embedded test metadata differs")
    require(manifest.get("converter_sha256") == sha256(REPO / "research/prepare_full_waterdrop.py"), "Test converter source differs")
    require([row.get("source_index") for row in manifest["records"]] == list(range(30))
            and len({row["id"] for row in manifest["records"]}) == 30, "Test IDs/source order differ")
    existing = set()
    for split in ("train", "valid"):
        existing.update(row["trajectory_content_sha256"] for row in read_json(args.data_dir / (split + ".json"))["records"])
    contents = [row["trajectory_content_sha256"] for row in manifest["records"]]
    require(len(set(contents)) == 30 and not set(contents) & existing, "Duplicate test trajectory within/across splits")
    for row in manifest["records"]:
        shape = row["positions"]["shape"]
        require(len(shape) == 3 and shape[0] == 1001 and shape[-1] == 2 and shape[1] > 0, "Test trajectory shape differs")
    from gns.data_loader import load_manifest_data
    load_manifest_data(path, verify_hashes=True)  # Safe numeric mmap header/dtype/path plus every array checksum.
    return {"manifest_sha256": sha256(path), "metadata_sha256": expected_metadata_hash, "source_sha256": SOURCE_SHA256,
            "record_count": 30, "evaluated_source_indices": list(range(3, 30))}


def validate_evaluation(job, args, test_identity, sources):
    output, model = job["output"], job["model"]
    status = read_json(output / "status.json") if (output / "status.json").exists() else {}
    require(status.get("state") not in ("error", "failed"), f"{job['name']} has an infrastructure/numerical error requiring review")
    if not (output / "protocol.json").exists() and not (output / "result.json").exists():
        require(not (output / "status.json").exists(), "Orphan evaluation status lacks the original protocol; preserve for review")
        require(not any(output.glob("trajectory_*")), "Orphan evaluation artifacts lack the original protocol; preserve for review")
        return "new"
    if job["kind"] == "rollout":
        result = rollout_summary.load_run(output, model["objective"], model["seed"], sha256(args.protocol))
    else:
        result = diagnostic_summary.load_run(output, model["objective"], model["seed"], sha256(args.protocol),
                    sha256(args.companion_protocol), sources["research/full_same_state.py"])
    require(result["state"] not in ("invalid", "missing"), f"Invalid saved {job['name']}: {result.get('errors')}")
    protocol = read_json(output / "protocol.json")
    require(protocol.get("checkpoint_sha256") == model["checkpoint_sha256"]
            and protocol.get("manifest_sha256") == test_identity["manifest_sha256"]
            and protocol.get("metadata_sha256") == test_identity["metadata_sha256"]
            and protocol.get("threads") == 2 and protocol["runtime"].get("device") == "mps"
            and protocol["runtime"].get("mps_fallback_environment") == "0", "Saved evaluator checkpoint/data/device configuration differs")
    require(all(path in sources and sources[path] == value for path, value in protocol["code_sha256"].items()),
            "Saved evaluator source hashes differ from queue snapshot")
    # Complete results may contain guard-failed trajectories/frames. The strict
    # loaders preserve these outcomes and do not require finite full-horizon MSE.
    return "complete" if result["state"] == "complete" else "resume"


def supervise(command, log_path, deadline, *, clock=None, sleeper=None, popen=None, scan=None, on_start=None):
    clock, sleeper = (now if clock is None else clock), (time.sleep if sleeper is None else sleeper)
    popen, scan = (subprocess.Popen if popen is None else popen), (relevant_processes if scan is None else scan)
    if clock() >= deadline:
        return {"state": "deadline_reached_incomplete", "started": False}
    require(not scan(), "Training/evaluation activity appeared before child launch")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child = None
    with log_path.open("a") as stream:
        try:
            # Check again after prelaunch work; never start a child after cutoff.
            if clock() >= deadline:
                return {"state": "deadline_reached_incomplete", "started": False}
            child = popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=REPO,
                          env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0"))
            if on_start is not None:
                on_start(child.pid)  # Recovery checks this even if the parent later dies.
            while child.poll() is None:
                if clock() >= deadline:
                    reviewed.stop_child(child)
                    return {"state": "deadline_reached_incomplete", "started": True, "child_pid": child.pid}
                others = [process for process in scan() if process["pid"] != child.pid]
                require(not others, f"Concurrent training/evaluation appeared; partial timing requires review: {others}")
                sleeper(min(5., max(0., (deadline - clock()).total_seconds())))
            require(child.returncode == 0, f"Child exited {child.returncode}; inspect {log_path}")
            return {"state": "complete", "started": True, "child_pid": child.pid}
        finally:
            if child is not None and child.poll() is None:
                reviewed.stop_child(child)  # Escalate and reap before any lock can be released.


def release_lock(path, token):
    if path.exists() and read_json(path).get("token") == token:
        path.unlink()


@contextmanager
def interruption_handlers():
    """Turn SIGTERM/SIGINT into cleanup, ignoring repeats until children are reaped."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupt(signum, frame):
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt(f"Received signal {signum}")
    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_queue(args):
    initial = preflight(args)
    if initial["state"] != "ready":
        return initial
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / "queue.lock"
    token = reviewed.acquire_lock(lock)
    try:
        with ExitStack() as stack:
            # Holding these original reviewed locks blocks both a restarted
            # training queue and direct training invocations for any final model.
            training_lock = args.training_dir / "queue.lock"
            training_token = reviewed.acquire_lock(training_lock)
            stack.callback(release_lock, training_lock, training_token)
            for model in MODELS:
                stack.enter_context(RunLock(args.training_dir / model["name"], clear_stale=True))
            gated = preflight(args, ignore_pids=(os.getpid(),))
            require(gated["state"] == "ready", f"Six-model gate changed: {gated['blocking_reasons']}")
            sources = source_snapshot()
            config = {"schema": 1, "deadline_utc": args.deadline.isoformat(), "models": gated["models"],
                      "validation_evidence": gated["validation_evidence"],
                      "training_dir": str(args.training_dir), "data_dir": str(args.data_dir), "raw_test": str(args.raw_test),
                      "training_protocol_sha256": sha256(args.protocol), "companion_protocol_sha256": sha256(args.companion_protocol),
                      "raw_test_sha256": SOURCE_SHA256, "source_sha256": sources,
                      "job_order": [job["name"] for job in evaluation_jobs(args, gated["models"])],
                      "runtime": "sequential native MPS, threads=2, fallback=0; no training overlap"}
            path = args.output_dir / "queue_protocol.json"
            if path.exists():
                require(read_json(path) == config, "Evaluation queue configuration/model/source hashes changed; use a new explicit output root")
            else:
                atomic_json(path, config)
            state_path = args.output_dir / "queue_status.json"
            if state_path.exists():
                require(read_json(state_path).get("state") != "failed", "Earlier evaluation queue failure requires review; automatic numerical/protocol-error retry is disabled")
            def status(state, **details):
                value = {"state": state, "pid": os.getpid(), "updated_utc": now().isoformat(), "deadline_utc": args.deadline.isoformat(),
                         "queue_protocol_sha256": sha256(path), **details}
                atomic_json(state_path, value)
                return value
            try:
                if now() >= args.deadline:
                    return status("deadline_reached_incomplete")
                status("running", stage="verify_test_source")
                require(args.raw_test.name == "test.tfrecord" and sha256(args.raw_test) == SOURCE_SHA256, "Raw test source SHA256 differs")
                if now() >= args.deadline:
                    return status("deadline_reached_incomplete", stage="verify_test_source")
                if not (args.data_dir / "test.json").exists():
                    # Preserve an interrupted converter's published-but-unindexed
                    # directory; never delete partial arrays to obtain a fresh run.
                    orphan = args.data_dir / "test"
                    if orphan.exists():
                        require(orphan.is_dir() and not orphan.is_symlink(), "Unexpected unpublished test path")
                        orphan.rename(args.data_dir / (".test-unpublished-" + uuid.uuid4().hex))
                    command = [sys.executable, "-m", "research.prepare_full_waterdrop", "--input-dir", str(args.raw_test.parent),
                               "--output-dir", str(args.data_dir), "--metadata", str(args.data_dir / "metadata.json"), "--splits", "test"]
                    status("running", stage="convert_test")
                    outcome = supervise(command, args.output_dir / "conversion.log", args.deadline,
                                        on_start=lambda pid: status("running", stage="convert_test", command=command, child_pid=pid))
                    if outcome["state"] != "complete":
                        return status(outcome["state"], stage="convert_test", **{key: value for key, value in outcome.items() if key != "state"})
                test_identity = verify_test_manifest(args, gated["models"][0]["metadata_sha256"])
                identity_path = args.output_dir / "test_identity.json"
                if identity_path.exists():
                    require(read_json(identity_path) == test_identity, "Numeric test manifest changed after queue publication")
                else:
                    atomic_json(identity_path, test_identity)
                for job in evaluation_jobs(args, gated["models"]):
                    if now() >= args.deadline:
                        return status("deadline_reached_incomplete", current_job=job["name"])
                    mode = validate_evaluation(job, args, test_identity, sources)
                    if mode == "complete":
                        status("running", completed_job=job["name"], reused=True)
                        continue
                    command = job["command"] + (["--resume"] if mode == "resume" else [])
                    status("running", current_job=job["name"], command=command)
                    outcome = supervise(command, job["output"] / "supervisor.log", args.deadline,
                                        on_start=lambda pid: status("running", current_job=job["name"], command=command, child_pid=pid))
                    if outcome["state"] != "complete":
                        return status(outcome["state"], current_job=job["name"], **{key: value for key, value in outcome.items() if key != "state"})
                    require(validate_evaluation(job, args, test_identity, sources) == "complete", "Child did not commit a complete verified evaluation")
                summary_commands = [
                    ("rollout", [sys.executable, "-m", "research.summarize_full_rollouts", "--evaluation-root", str(args.output_dir / "rollout"),
                                 "--protocol", str(args.protocol), "--output-prefix", str(args.output_dir / "reports/full_rollouts")]),
                    ("same_state", [sys.executable, "-m", "research.summarize_full_same_state", "--evaluation-root", str(args.output_dir / "same_state"),
                                    "--training-protocol", str(args.protocol), "--companion-protocol", str(args.companion_protocol),
                                    "--output-prefix", str(args.output_dir / "reports/full_same_state")])]
                for kind, command in summary_commands:
                    status("running", stage="summarize_" + kind, command=command)
                    outcome = supervise(command, args.output_dir / ("summary_" + kind + ".log"), args.deadline,
                                        on_start=lambda pid: status("running", stage="summarize_" + kind, command=command, child_pid=pid))
                    if outcome["state"] != "complete":
                        return status(outcome["state"], stage="summarize_" + kind)
                    name = "full_rollouts.json" if kind == "rollout" else "full_same_state.json"
                    require(read_json(args.output_dir / "reports" / name).get("state") == "complete", "Six-model aggregation is incomplete or inconsistent")
                return status("complete")
            except BaseException as error:
                status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
                raise
    finally:
        release_lock(lock, token)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Local work/ directory: raw validation/test arrays must not be published by this tool")
    parser.add_argument("--protocol", type=Path, default=REPO / "research/protocols/full_waterdrop_100k.md")
    parser.add_argument("--companion-protocol", type=Path, default=REPO / "research/protocols/full_same_state_diagnostic.md")
    parser.add_argument("--deadline-utc", default=HARD_DEADLINE.isoformat())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Execute only after all six models pass the exclusion gate")
    mode.add_argument("--check-only", action="store_true", help="Default: no lock mutation, checkpoint loading while active, or test access")
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline_utc)
    if args.deadline.tzinfo is None or args.deadline > HARD_DEADLINE:
        parser.error("Deadline must include a timezone and cannot exceed October 7, 2026 08:00 UTC")
    for name in ("training_dir", "data_dir", "raw_test", "output_dir", "protocol", "companion_protocol"):
        setattr(args, name, getattr(args, name).resolve())
    require(all(not args.output_dir.is_relative_to(root) and not root.is_relative_to(args.output_dir)
                for root in (args.training_dir, args.data_dir)), "Evaluation output must be disjoint from training and dataset directories")
    return args


def main():
    args = parse_args()
    with interruption_handlers():
        result = run_queue(args) if args.run else preflight(args)
    if not args.run:
        result["mode"] = "check_only"
        if result["state"] == "ready":
            result["planned_jobs"] = [job["name"] for job in evaluation_jobs(args, result["models"])]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
