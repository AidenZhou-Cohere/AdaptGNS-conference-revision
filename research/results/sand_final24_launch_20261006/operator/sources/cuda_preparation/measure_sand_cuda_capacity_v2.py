#!/usr/bin/env python3
"""Prepare/measure the fixed 4+2-wave, 512-update Sand CUDA capacity probe.

Description only by default. Execution requires an explicit root timing release.
No scientific training, test evaluation, resume or promotion of probe checkpoints.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_cuda_capacity_v2"
RELEASE_SCHEMA = "adaptgns_sand_cuda_capacity_release_v2"
TRAINING_SCHEMA = "adaptgns_sand_cuda_training_deterministic_v2"
STOP = 512
WARMUP = 64
UPDATES = 100000
WAVE_TIMEOUT = 3600
CLEANUP_SECONDS = 15
COMPUTE_DEADLINE = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "LD_LIBRARY_PATH": "/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"}
PINS = {
    "trainer": "9dd376f6d4b6881290b134c8f410b89933bcdb026c9b52de601ba0ec7bb05a9b",
    "train_manifest": "f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f",
    "admission": "fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73",
    "structural_report": "bc63fabfc07a663ab866f454c07e984dd57b8d4b32aa06a760838c4f84da72e6",
}
CORE_PINS = {
    "research/full_training.py": "c6bec98cb8200ef74c3d5d30309dba858c0dbef49f3380a0bc8762683c04e60d",
    "adaptive-gns/gns/data_loader.py": "287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef",
    "adaptive-gns/gns/device_utils.py": "324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326",
    "adaptive-gns/gns/graph_network.py": "af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91",
    "adaptive-gns/gns/learned_simulator.py": "216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf",
    "adaptive-gns/gns/losses.py": "94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5",
    "adaptive-gns/gns/model_io.py": "06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e",
}
SCHEDULE = [
    {"id": "faithful_seed0", "wave": "A", "gpu": 0, "objective": "faithful", "seed": 0},
    {"id": "nll_seed0", "wave": "A", "gpu": 1, "objective": "nll", "seed": 0},
    {"id": "faithful_seed1", "wave": "A", "gpu": 2, "objective": "faithful", "seed": 1},
    {"id": "nll_seed1", "wave": "A", "gpu": 3, "objective": "nll", "seed": 1},
    {"id": "faithful_seed2", "wave": "B", "gpu": 0, "objective": "faithful", "seed": 2},
    {"id": "nll_seed2", "wave": "B", "gpu": 1, "objective": "nll", "seed": 2},
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    return ordered[lower] + (index-lower) * (ordered[min(lower+1, len(ordered)-1)]-ordered[lower])


def fixed_command(args, job):
    return [str(args.python), "-u", str(args.trainer), "--execute", "--repo", str(args.repo),
            "--train-manifest", str(args.train_manifest), "--admission", str(args.admission),
            "--structural-report", str(args.structural_report), "--protocol", str(args.protocol),
            "--output-dir", str(args.output_dir / "jobs" / job["id"]),
            "--objective", job["objective"], "--seed", str(job["seed"]), "--cuda-index", str(job["gpu"]),
            "--updates", str(UPDATES), "--stop-after", str(STOP), "--threads", "2",
            "--checkpoint-every", "10000", "--log-every", "1"]


def validate_rows(rows, particle_counts):
    require(len(rows) == STOP and [row.get("completed_steps") for row in rows] == list(range(1, STOP+1)),
            "Every update1..512 must occur exactly once in order")
    previous = 0.0
    for step, row in enumerate(rows, 1):
        require(set(row) == {"completed_steps", "loss", "lr", "frame_ids", "particles", "guarded_update_seconds", "elapsed_seconds"},
                "Unexpected/missing training-history fields")
        require(type(row["completed_steps"]) is int and finite(row["guarded_update_seconds"], True)
                and finite(row["elapsed_seconds"], True) and row["elapsed_seconds"] > previous
                and type(row["loss"]) in (int, float) and math.isfinite(row["loss"]), "Nonfinite or nonmonotone training record")
        require(row["lr"] == 1e-4 * (1e-5 / 1e-4) ** ((step-1)/99999), "Learning-rate schedule differs")
        ids = row["frame_ids"]
        require(isinstance(ids, list) and len(ids) == 2 and all(isinstance(value, str) for value in ids), "Two paired frame identities required")
        particles = 0
        for identity in ids:
            trajectory, target = identity.rsplit(":", 1)
            require(trajectory in particle_counts and target.isdigit() and 6 <= int(target) < 320, "Training frame identity outside admitted source")
            particles += particle_counts[trajectory]
        require(type(row["particles"]) is int and row["particles"] == particles, "Recorded particle count differs from frame identities")
        previous = row["elapsed_seconds"]
    guards = [row["guarded_update_seconds"] for row in rows]
    steady = guards[WARMUP:]
    wall = (rows[-1]["elapsed_seconds"] - rows[WARMUP-1]["elapsed_seconds"]) / (STOP-WARMUP)
    require(finite(wall, True) and wall*(STOP-WARMUP)+1e-6 >= sum(steady), "Steady wall cannot exclude guarded update work")
    return {"steady_wall_seconds_per_update": wall,
            "steady_guarded_seconds": {"mean": statistics.mean(steady), "median": statistics.median(steady),
                                       "p90_linear": percentile(steady, .9), "maximum": max(steady)},
            "sum_all512_guarded_seconds": sum(guards), "sum_warmup64_guarded_seconds": sum(guards[:WARMUP]),
            "warmup_excess_over_steady_guarded_mean_seconds": sum(guards[:WARMUP]) - WARMUP*statistics.mean(steady),
            "all512_timings": [{key: row[key] for key in ("completed_steps", "guarded_update_seconds", "elapsed_seconds")} for row in rows]}


def canonical_gpu_uuid(value):
    """Accept exact NVIDIA/PyTorch UUID spellings; retain raw values in evidence."""
    require(isinstance(value, str) and re.fullmatch(r"(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value) is not None,
            "Malformed full GPU UUID")
    return value.removeprefix("GPU-").lower()


def verify_job(directory, job, external, manifest, protocol_sha, gpu_uuid):
    directory = Path(directory)
    require(external.get("exit_code") == 0 and finite(external.get("elapsed_seconds"), True), "Process did not finish successfully")
    require(not (directory / "run.lock").exists() and not list(directory.glob("*.tmp")), "Live/stale lock or partial checkpoint remains")
    config, status, history, pointer = [read_json(directory / name) for name in ("protocol.json", "status.json", "history.json", "latest.json")]
    config_sha = canonical_hash(config)
    require(config.get("schema") == TRAINING_SCHEMA and config.get("dataset") == "Sand"
            and config.get("objective") == job["objective"] and type(config.get("seed")) is int and config["seed"] == job["seed"]
            and config.get("updates") == UPDATES and config.get("batch_size") == 2 and config.get("history") == 6
            and config.get("checkpoint_every") == 10000 and config.get("log_every") == 1
            and config.get("research_protocol_sha256") == protocol_sha, "Training configuration differs from timing contract")
    require(config.get("source_sha256") == {**CORE_PINS, "train_sand_cuda_deterministic.py": PINS["trainer"]}, "Trainer/core source lineage differs")
    require(config.get("architecture") == {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2}
            and config.get("noise_std") == 6.7e-4
            and config.get("graph") == {"radius": .015, "backend": "scipy_host", "cap": 128, "self_candidates": True, "augmentation_probability": 0.0}
            and config.get("optimizer") == {"name": "Adam", "initial_lr": 1e-4, "final_lr": 1e-5, "decay_updates": 100000,
                "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0, "foreach": False, "fused": False, "gradient_clipping": None},
            "Architecture/noise/graph/optimizer differs")
    data = config.get("data", {})
    require(data.get("manifest_sha256") == PINS["train_manifest"] and data.get("admission_sha256") == PINS["admission"]
            and data.get("structural_report_sha256") == PINS["structural_report"]
            and data.get("frames_per_trajectory") == 320 and data.get("particle_type_ids") == [6], "Training data lineage differs")
    runtime = config.get("runtime", {})
    require(runtime.get("device") == f"cuda:{job['gpu']}" and canonical_gpu_uuid(runtime.get("uuid")) == canonical_gpu_uuid(gpu_uuid)
            and runtime.get("torch") == "2.13.0+cu129" and runtime.get("cuda") == "12.9" and runtime.get("threads") == 2
            and runtime.get("deterministic_algorithms") is True and runtime.get("deterministic_warn_only") is False
            and runtime.get("cublas_workspace_config") == ENVIRONMENT["CUBLAS_WORKSPACE_CONFIG"]
            and all(runtime.get(key) is False for key in ("tf32", "amp", "compile", "ddp")), "CUDA runtime/GPU mapping differs")
    require(status.get("schema") == TRAINING_SCHEMA and status.get("state") == "planned_stop_incomplete"
            and type(status.get("completed_steps")) is int and status["completed_steps"] == STOP
            and type(status.get("committed_steps")) is int and status["committed_steps"] == STOP
            and status.get("requested_steps") == UPDATES and status.get("error") is None
            and status.get("objective") == job["objective"] and status.get("seed") == job["seed"]
            and status.get("run_config_sha256") == config_sha and status.get("process", {}).get("pid") == external["pid"],
            "Only the expected committed planned_stop_incomplete512 outcome is successful")
    expected_pointer = {"path": f"checkpoint-{STOP:09d}.pt", "completed_steps": STOP, "run_config_sha256": config_sha,
                        "sha256": sha(directory / f"checkpoint-{STOP:09d}.pt")}
    require(pointer == expected_pointer and status.get("latest_checkpoint") == pointer, "Final checkpoint pointer/hash/status disagree")
    initial = external.get("initial_pointer")
    require(initial == {"path": "checkpoint-000000000.pt", "completed_steps": 0, "run_config_sha256": config_sha,
                        "sha256": sha(directory / "checkpoint-000000000.pt")}, "Observed step-zero pointer/hash is missing or inconsistent")
    rows = history.get("training", [])
    counts = {row["id"]: row["positions"]["shape"][1] for row in manifest["records"]}
    stats = validate_rows(rows, counts)
    stdout_rows = [json.loads(line) for line in Path(external["stdout_file"]).read_text().splitlines() if line.strip()]
    require(stdout_rows == rows and status.get("last_training") == rows[-1], "Stdout/history/last status training rows differ")
    require(finite(history.get("elapsed_seconds"), True) and history["elapsed_seconds"] >= rows[-1]["elapsed_seconds"], "Final history time is invalid")
    residual = external["elapsed_seconds"] - stats["sum_all512_guarded_seconds"]
    require(residual >= -1e-6, "External timing cannot be shorter than guarded work")
    return {**job, "status": "verified_512_timing_probe", "config_sha256": config_sha, "gpu_uuid": gpu_uuid,
            "external": external, "initial_pointer": initial, "final_pointer": pointer,
            "artifact_sha256": {name: sha(directory/name) for name in ("protocol.json", "status.json", "history.json", "latest.json")},
            **stats, "external_minus_all_guarded_seconds": residual,
            "nonnegative_external_minus_all_guarded_seconds": max(0., residual), "pairing_rows": rows}


def verify_pairing(jobs):
    by_id = {job["id"]: job for job in jobs}
    require(len(by_id) == len(jobs), "Duplicate job identity")
    checked = []
    for seed in sorted({job["seed"] for job in jobs}):
        a, b = by_id.get(f"faithful_seed{seed}"), by_id.get(f"nll_seed{seed}")
        require(a is not None and b is not None, "Both objectives required for each measured seed")
        require(len(a["pairing_rows"]) == len(b["pairing_rows"]) == STOP, "Complete paired512 rows required")
        for left, right in zip(a["pairing_rows"], b["pairing_rows"]):
            require(all(left[key] == right[key] for key in ("completed_steps", "frame_ids", "particles", "lr")), "Paired host frame/particle/lr schedule differs")
        checked.append({"seed": seed, "paired_rows": STOP, "frame_ids_particles_lr_exact": True})
    return checked


def forecast(jobs, evaluation_seconds=None, diagnostics_seconds=None, at=None):
    require({job["id"] for job in jobs} == {job["id"] for job in SCHEDULE} and len(jobs) == 6,
            "All six complete timing models required for any forecast")
    q = {wave: max(job["steady_wall_seconds_per_update"] for job in jobs if job["wave"] == wave) for wave in ("A", "B")}
    r = {wave: max(job["nonnegative_external_minus_all_guarded_seconds"] for job in jobs if job["wave"] == wave) for wave in ("A", "B")}
    require(all(finite(value, True) for value in q.values()) and all(finite(value) for value in r.values()), "Invalid timing quantities")
    training = 1.35 * (100000 * (q["A"] + q["B"]) + 12 * (r["A"] + r["B"]))
    at = now() if at is None else at
    require(at.tzinfo is not None, "Forecast origin must have an explicit timezone")
    require(evaluation_seconds is None or finite(evaluation_seconds, True), "Positive complete-evaluation estimate required")
    require(diagnostics_seconds is None or finite(diagnostics_seconds, True), "Positive diagnostics execution estimate required")
    total = None if evaluation_seconds is None or diagnostics_seconds is None else training + evaluation_seconds + diagnostics_seconds + 3600
    finish = None if total is None else at + timedelta(seconds=total)
    return {"formula": "1.35*(100000*(q4+q2)+12*(r4+r2))", "q4": q["A"], "q2": q["B"], "r4": r["A"], "r2": r["B"],
            "training_seconds": training, "complete_evaluation_seconds": evaluation_seconds,
            "diagnostics_execution_seconds": diagnostics_seconds, "diagnostics_verification_allowance_seconds": 3600,
            "total_compute_analysis_seconds": total, "forecast_origin_utc": at.isoformat(),
            "forecast_finish_utc": finish.isoformat() if finish else None, "compute_analysis_deadline_utc": COMPUTE_DEADLINE.isoformat(),
            "fits_before_seven_hour_writing_reserve": None if finish is None else finish <= COMPUTE_DEADLINE,
            "interpretation": "Planning contingency, not a probabilistic bound. Wall-q and residual partly double-count logging. No scientific admission is automatic."}


def process_identity(pid):
    root = Path("/proc") / str(pid)
    stat_fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "state": stat_fields[0], "ppid": int(stat_fields[1]), "start_ticks": int(stat_fields[19]),
            "argv": [value.decode() for value in (root / "cmdline").read_bytes().split(b"\0") if value],
            "executable": os.readlink(root / "exe")}


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if row:
            require(len(row) == 2 and row[0].strip().isdigit(), "Unexpected GPU process inventory")
            rows.append({"pid": int(row[0]), "gpu_uuid": row[1].strip()})
    return rows


def stop_owned(children, signum=signal.SIGINT):
    """Signal only an unreaped child with the captured PID/start/command identity."""
    for child in children:
        proc = child["process"]
        if proc.returncode is not None:
            continue
        event = {"signal": signal.Signals(signum).name, "at_utc": now().isoformat()}
        try:
            identity = process_identity(proc.pid)
            if identity["state"] == "Z":
                continue  # wait4 will reap it; zombies have an empty cmdline.
            prior = child.get("identity")
            require(identity["ppid"] == os.getpid() and identity["argv"] == child["command"]
                    and (prior is None or identity["start_ticks"] == prior["start_ticks"]),
                    "Refusing to signal changed process identity")
            if prior is None:
                child["identity"] = identity
            os.killpg(proc.pid, signum)
            event["result"] = "sent_to_owned_group"
        except (ProcessLookupError, FileNotFoundError):
            event["result"] = "already_exited"
        except Exception as error:
            event.update(result="refused_or_failed", error_type=type(error).__name__, error=str(error))
        child["signals"].append(event)


def capture_initial(args, child):
    path = args.output_dir / "jobs" / child["job"]["id"] / "latest.json"
    if child["initial_pointer"] is None and path.exists():
        pointer = read_json(path)
        if pointer.get("completed_steps") == 0:
            child["initial_pointer"] = pointer
            atomic_json(args.output_dir / "logs" / (child["job"]["id"] + ".initial_pointer.json"), pointer)


def reap_child(args, child, record, verified, launch, manifest):
    proc = child["process"]
    if proc.returncode is not None:
        return None
    pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
    if not pid:
        return None
    proc.returncode = os.waitstatus_to_exitcode(status)
    for handle in child["handles"]:
        handle.close()
    external = {"pid": pid, "identity": child["identity"], "command": child["command"], "started_utc": child["started_utc"],
        "ended_utc": now().isoformat(), "elapsed_seconds": time.perf_counter()-child["started"], "exit_code": proc.returncode,
        "peak_host_rss_bytes": usage.ru_maxrss*1024, "peak_host_rss_source_units": "Linux ru_maxrss KiB, multiplied by1024",
        "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
        "stdout_file": child["stdout_file"], "stderr_file": child["stderr_file"], "initial_pointer": child["initial_pointer"],
        "signals": child["signals"], "nominal_poll_interval_seconds": .1, "elapsed_includes_exit_observer_lag": True}
    outcome = {**child["job"], "external": external}
    failure = None
    try:
        value = verify_job(args.output_dir/"jobs"/child["job"]["id"], child["job"], external, manifest,
                           launch["files_sha256"]["protocol"], launch["gpu_uuids"][child["job"]["gpu"]])
        verified.append(value)
        outcome["verification"] = "passed"
    except Exception as error:
        outcome.update(verification="failed", error_type=type(error).__name__, error=str(error))
        failure = f"{child['job']['id']}: {error}"
    record["jobs"].append(outcome)  # Retain the outcome even if a subsequent write fails.
    atomic_json(args.output_dir/"logs"/(child["job"]["id"]+".outcome.json"), outcome)
    atomic_json(args.output_dir/f"wave_{record['wave']}.json", record)
    return failure


def cleanup_owned(args, children, record, verified, launch, manifest):
    started = time.perf_counter()
    sent = set()
    while any(c["process"].returncode is None for c in children):
        elapsed = time.perf_counter()-started
        for threshold, signum in ((0, signal.SIGINT), (5, signal.SIGTERM), (10, signal.SIGKILL)):
            if elapsed >= threshold and signum not in sent:
                stop_owned(children, signum)
                sent.add(signum)
        for child in children:
            if child["process"].returncode is None:
                try:
                    capture_initial(args, child)
                except Exception as error:
                    record.setdefault("cleanup_errors", []).append({"id": child["job"]["id"], "error": str(error)})
                try:
                    reap_child(args, child, record, verified, launch, manifest)
                except Exception as error:
                    record.setdefault("cleanup_errors", []).append({"id": child["job"]["id"], "error": str(error)})
        if elapsed >= CLEANUP_SECONDS:
            break
        time.sleep(.1)
    record["unreaped_owned_children"] = [
        {"id": c["job"]["id"], "pid": c["process"].pid, "identity": c["identity"], "command": c["command"], "signals": c["signals"]}
        for c in children if c["process"].returncode is None]
    for child in children:
        for handle in child["handles"]:
            handle.close()


def run_wave(args, wave, launch, manifest):
    jobs = [job for job in SCHEDULE if job["wave"] == wave]
    require(not gpu_processes(), "GPU compute processes exist before wave; root must review identities")
    children, verified = [], []
    started = time.perf_counter()
    record = {"wave": wave, "state": "running", "started_utc": now().isoformat(), "jobs": [], "observations": [], "full_steady_overlap_observed": False}
    atomic_json(args.output_dir/f"wave_{wave}.json", record)
    try:
        for job in jobs:
            command = fixed_command(args, job)
            logs = args.output_dir / "logs"
            stdout_path, stderr_path = logs / (job["id"] + ".stdout.jsonl"), logs / (job["id"] + ".stderr.txt")
            stdout, stderr = stdout_path.open("xb"), stderr_path.open("xb")
            external_started = time.perf_counter()
            try:
                proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env={**os.environ, **ENVIRONMENT}, start_new_session=True)
            except BaseException:
                stdout.close()
                stderr.close()
                raise
            child = {"job": job, "process": proc, "identity": None, "command": command, "handles": (stdout, stderr), "signals": [],
                     "started": external_started, "started_utc": now().isoformat(), "stdout_file": str(stdout_path), "stderr_file": str(stderr_path), "initial_pointer": None}
            children.append(child)  # Register immediately, even if identity inspection fails.
            identity = process_identity(proc.pid)
            require(identity["argv"] == command and identity["ppid"] == os.getpid(), "Launched trainer identity differs")
            child["identity"] = identity
            atomic_json(logs/(job["id"]+".launch.json"), {"job": job, "pid": proc.pid, "identity": identity, "command": command, "started_utc": child["started_utc"]})
        record["launch_skew_seconds"] = max(c["started"] for c in children)-min(c["started"] for c in children)
        last_observation = -math.inf
        while any(child["process"].returncode is None for child in children):
            elapsed = time.perf_counter()-started
            require(elapsed <= WAVE_TIMEOUT and now() < COMPUTE_DEADLINE, "Wave timing budget/deadline reached")
            for child in children:
                if child["process"].returncode is not None:
                    continue
                capture_initial(args, child)
                failure = reap_child(args, child, record, verified, launch, manifest)
                require(failure is None, failure or "job failed")
            if elapsed-last_observation >= 1:
                last_observation = elapsed
                known = {c["process"].pid: c for c in children}
                running = {pid: c for pid, c in known.items() if c["process"].returncode is None}
                processes = gpu_processes()
                require(all(p["pid"] in known and p["gpu_uuid"] == launch["gpu_uuids"][known[p["pid"]]["job"]["gpu"]] for p in processes),
                        "External process or unexpected GPU assignment contaminated the capacity probe")
                states = {}
                for pid, child in running.items():
                    path = args.output_dir / "jobs" / child["job"]["id"] / "status.json"
                    if path.exists():
                        value = read_json(path)
                        states[str(pid)] = {"state": value.get("state"), "completed_steps": value.get("completed_steps")}
                overlap = (len(running) == len(jobs) and {p["pid"] for p in processes} == set(running)
                           and len(states) == len(jobs) and all(s["state"] == "running" and type(s["completed_steps"]) is int and WARMUP <= s["completed_steps"] < STOP for s in states.values()))
                record["full_steady_overlap_observed"] |= overlap
                record["observations"].append({"elapsed_seconds": elapsed, "utc": now().isoformat(), "gpu_processes": processes, "trainer_states": states})
                atomic_json(args.output_dir/f"wave_{wave}.json", record)
            time.sleep(.1)
        require(len(verified) == len(jobs) and record["full_steady_overlap_observed"], "Complete steady overlap and all wave outcomes are required")
        verify_pairing(verified)
        record["state"] = "verified"
        return verified, record
    except BaseException as error:
        record.update(state="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        cleanup_owned(args, children, record, verified, launch, manifest)
        raise
    finally:
        record["ended_utc"] = now().isoformat()
        record["wave_elapsed_seconds"] = time.perf_counter()-started
        if children:
            record["observer_and_launch_span_minus_slowest_process_seconds"] = record["wave_elapsed_seconds"] - max((j["external"]["elapsed_seconds"] for j in record["jobs"]), default=0.)
        atomic_json(args.output_dir/f"wave_{wave}.json", record)


def run_waves(run_one):
    completed = []
    for wave in ("A", "B"):
        jobs, record = run_one(wave)
        require(record.get("state") == "verified", "Failed/incomplete wave blocks the next wave")
        completed.extend(jobs)
    verify_pairing(completed)
    return completed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--summarize", type=Path)
    for name in ("release", "repo", "trainer", "train-manifest", "admission", "structural-report", "protocol", "mechanism-review", "output-dir", "python", "evaluation-estimate", "diagnostics-estimate", "summary-output"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args(argv)
    if args.execute and any(getattr(args, key) is None for key in ("release", "repo", "trainer", "train_manifest", "admission", "structural_report", "protocol", "mechanism_review", "output_dir", "python")):
        parser.error("Execution requires explicit release/source/data/protocol/review/interpreter/fresh-output paths")
    return args


def evaluation_seconds(path):
    if path is None:
        return None
    value = read_json(path)
    require(value.get("schema") == "adaptgns_sand_full_rollout_cost_estimate_v1" and value.get("status") == "complete_workload_estimated"
            and value.get("workload") == {"models": 6, "trajectories_per_model": 30, "policies_per_trajectory": 5, "horizon": 314, "outcomes": 900}
            and value.get("all_reference_rollouts_complete") is True and finite(value.get("total_seconds"), True)
            and isinstance(value.get("timing_evidence_sha256"), list) and bool(value["timing_evidence_sha256"]),
            "A separately verified complete900-outcome full-horizon evaluation estimate is required")
    return value["total_seconds"]


def diagnostics_seconds(path):
    if path is None:
        return None
    value = read_json(path)
    require(value.get("schema") == "adaptgns_sand_diagnostics_cost_estimate_v1"
            and value.get("status") == "complete_workload_estimated"
            and value.get("workload") == {"same_state_forward_calls": 95400, "clean_forward_calls": 2304, "total_forward_calls": 97704}
            and finite(value.get("maximum_measured_whole_call_seconds"), True)
            and finite(value.get("graph_and_artifact_overhead_seconds"))
            and value.get("total_seconds") >= 97704*value["maximum_measured_whole_call_seconds"]+value["graph_and_artifact_overhead_seconds"]
            and isinstance(value.get("timing_evidence_sha256"), list) and bool(value["timing_evidence_sha256"]),
            "Separately measured complete diagnostics cost required;3600seconds is an additional verification allowance")
    return value["total_seconds"]


def validate_release(release, hashes):
    require(release.get("schema") == RELEASE_SCHEMA and release.get("status") == "admitted_for_timing"
            and release.get("issued_by") == "root" and release.get("scientific_training_admitted") is False
            and release.get("files_sha256") == hashes and release.get("environment") == ENVIRONMENT
            and release.get("schedule") == SCHEDULE and isinstance(release.get("review_rationale"), str) and release["review_rationale"].strip(),
            "Explicit root timing-only release with exact input hashes/schedule/environment is required")
    uuids = release.get("gpu_uuids")
    require(isinstance(uuids, list) and len(uuids) == len(set(uuids)) == 4
            and all(isinstance(x, str) and x for x in uuids), "Four explicit CUDA-index UUIDs required")
    require(len({canonical_gpu_uuid(value) for value in uuids}) == 4, "Four distinct physical GPU UUIDs required")
    return uuids


def verify_snapshot(root):
    launch = read_json(root/"launch.json")
    require(launch.get("schema") == SCHEMA and launch.get("schedule") == SCHEDULE
            and launch.get("environment") == ENVIRONMENT, "Measurement launch contract differs")
    hashes = launch.get("files_sha256", {})
    expected = {"trainer", "train_manifest", "admission", "structural_report", "protocol", "mechanism_review", "python", "supervisor"}
    require(set(hashes) == expected and set(launch.get("input_snapshots", {})) == expected-{"python"}, "Complete copied input inventory required")
    for key, relative in launch["input_snapshots"].items():
        path = (root/relative).resolve()
        require(path.parent == (root/"inputs").resolve() and sha(path) == hashes[key], "Copied input hash/path differs: "+key)
    require(all(hashes[key] == expected for key, expected in PINS.items()) and hashes["supervisor"] == sha(__file__), "Frozen capacity source/data differs")
    require(sha(root/"inputs/release.json") == launch.get("release_sha256"), "Root release snapshot differs")
    release = read_json(root/"inputs/release.json")
    require(validate_release(release, hashes) == launch.get("gpu_uuids"), "Launch/release GPU UUIDs differ")
    return launch


def verify_wave_record(root, wave, launch):
    record = read_json(root/f"wave_{wave}.json")
    jobs = [job for job in SCHEDULE if job["wave"] == wave]
    require(record.get("wave") == wave and record.get("state") == "verified" and record.get("full_steady_overlap_observed") is True
            and not record.get("unreaped_owned_children") and len(record.get("jobs", [])) == len(jobs), "Complete successful wave required")
    outcomes = record["jobs"]
    require({o.get("id") for o in outcomes} == {j["id"] for j in jobs}, "Exact wave membership required")
    pids = {}
    for outcome in outcomes:
        job = next(j for j in jobs if j["id"] == outcome["id"])
        require(all(outcome.get(key) == value for key, value in job.items()) and outcome.get("verification") == "passed", "Wave job fields differ")
        external = outcome["external"]
        require(external["command"] == launch["commands"][job["id"]] and not external.get("signals"), "Captured process command/signals differ")
        require(read_json(root/"logs"/(job["id"]+".outcome.json")) == outcome, "Standalone outcome differs")
        observed = read_json(root/"logs"/(job["id"]+".launch.json"))
        require(observed["job"] == job and observed["pid"] == external["pid"] and observed["identity"] == external["identity"]
                and observed["command"] == external["command"], "Process launch identity differs")
        require(read_json(root/"logs"/(job["id"]+".initial_pointer.json")) == external["initial_pointer"], "Initial pointer observation differs")
        require(external["pid"] not in pids, "Unique process identity required")
        pids[external["pid"]] = launch["gpu_uuids"][job["gpu"]]
    overlap = False
    for observation in record.get("observations", []):
        processes = observation["gpu_processes"]
        require(all(p["pid"] in pids and p["gpu_uuid"] == pids[p["pid"]] for p in processes), "GPU observation contaminated")
        states = observation["trainer_states"]
        overlap |= ({p["pid"] for p in processes} == set(pids) and set(states) == {str(p) for p in pids}
                    and all(s["state"] == "running" and type(s["completed_steps"]) is int and WARMUP <= s["completed_steps"] < STOP for s in states.values()))
    require(overlap, "No simultaneous steady-window GPU/process evidence")
    return outcomes


def main(argv=None):
    args = parse_args(argv)
    if not args.execute and args.summarize is None:
        print(json.dumps({"schema": SCHEMA, "execute": False, "schedule": SCHEDULE, "updates": UPDATES, "stop_after": STOP,
                          "requires": "root admitted_for_timing release; no scientific admission", "formula": "1.35*(100000*(q4+q2)+12*(r4+r2))"}, indent=2))
        return 0
    if args.summarize is not None:
        root = args.summarize.resolve()
        launch = verify_snapshot(root)
        manifest = read_json(root / launch["input_snapshots"]["train_manifest"])
        jobs = []
        for wave in ("A", "B"):
            for outcome in verify_wave_record(root, wave, launch):
                job = next(j for j in SCHEDULE if j["id"] == outcome["id"])
                external = dict(outcome["external"])
                external["stdout_file"] = str(root / "logs" / (job["id"] + ".stdout.jsonl"))
                jobs.append(verify_job(root/"jobs"/job["id"], job, external, manifest, launch["files_sha256"]["protocol"], launch["gpu_uuids"][job["gpu"]]))
        summary = {"schema": SCHEMA, "status": "all_six_verified", "pairing": verify_pairing(jobs), "jobs": jobs,
                   "forecast": forecast(jobs, evaluation_seconds(args.evaluation_estimate), diagnostics_seconds(args.diagnostics_estimate)),
                   "cost_estimates_sha256": {key: sha(path) for key, path in (("evaluation", args.evaluation_estimate), ("diagnostics", args.diagnostics_estimate)) if path is not None},
                   "scientific_training_admitted": False}
        if args.summary_output:
            require(not args.summary_output.exists(), "Summary output must be fresh; preserve earlier forecast snapshots")
            atomic_json(args.summary_output, summary)
            print(json.dumps({"summary": str(args.summary_output), "sha256": sha(args.summary_output), "forecast": summary["forecast"]}, indent=2))
        else:
            print(json.dumps(summary, indent=2))
        return 0
    require(sys.platform.startswith("linux") and hasattr(os, "wait4"), "Execution requires the Linux CUDA host")
    require(now() < COMPUTE_DEADLINE, "Capacity measurement cannot begin after the compute/analysis deadline")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "CUDA_VISIBLE_DEVICES remapping is not admitted")
    inputs = {key: getattr(args, key).resolve() for key in ("trainer", "train_manifest", "admission", "structural_report", "protocol", "mechanism_review", "python")}
    inputs["supervisor"] = Path(__file__).resolve()
    hashes = {key: sha(path) for key, path in inputs.items()}
    require(all(hashes[key] == value for key, value in PINS.items()), "Frozen trainer/data/admission source changed")
    require(all(sha(args.repo/path) == expected for path, expected in CORE_PINS.items()), "Frozen numerical core changed")
    release = read_json(args.release)
    uuids = validate_release(release, hashes)
    checked = datetime.fromisoformat(release["process_identity_checked_utc"])
    require(checked.tzinfo is not None and -60 <= (now()-checked).total_seconds() <= 300, "Fresh root process-identity check required")
    args.output_dir = args.output_dir.resolve()
    require(all(args.output_dir != p and p not in args.output_dir.parents for p in (args.train_manifest.resolve().parent, args.repo.resolve()/"adaptive-gns")), "Infrastructure output must be separate from input trees")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    for name in ("inputs", "jobs", "logs"):
        (args.output_dir/name).mkdir()
    snapshots = {}
    for key, path in inputs.items():
        if key != "python":
            snapshots[key] = "inputs/"+key+path.suffix
            (args.output_dir/snapshots[key]).write_bytes(path.read_bytes())
    (args.output_dir/"inputs/release.json").write_bytes(args.release.read_bytes())
    launch = {"schema": SCHEMA, "files_sha256": hashes, "gpu_uuids": uuids, "schedule": SCHEDULE,
              "input_snapshots": snapshots, "commands": {job["id"]: fixed_command(args, job) for job in SCHEDULE},
              "release_sha256": sha(args.release), "environment": ENVIRONMENT, "pid": os.getpid(), "hostname": socket.gethostname(), "started_utc": now().isoformat()}
    atomic_json(args.output_dir/"launch.json", launch)
    try:
        manifest = read_json(args.train_manifest)
        jobs = run_waves(lambda wave: run_wave(args, wave, launch, manifest))
        require(all(sha(path) == hashes[key] for key, path in inputs.items()), "Frozen inputs changed during measurement")
        require(all(sha(args.repo/path) == expected for path, expected in CORE_PINS.items()), "Frozen numerical core changed during measurement")
        verify_snapshot(args.output_dir)
        for wave in ("A", "B"):
            verify_wave_record(args.output_dir, wave, launch)
        summary = {"schema": SCHEMA, "status": "all_six_verified", "jobs": jobs, "pairing": verify_pairing(jobs),
                   "forecast": forecast(jobs), "scientific_training_admitted": False}
        atomic_json(args.output_dir/"capacity_summary.json", summary)
        atomic_json(args.output_dir/"status.json", {"state": "complete_capacity_measurement", "ended_utc": now().isoformat(),
                    "summary_sha256": sha(args.output_dir/"capacity_summary.json"), "scientific_training_admitted": False})
        return 0
    except BaseException as error:
        atomic_json(args.output_dir/"status.json", {"state": "failed_capacity_measurement", "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc(), "at_utc": now().isoformat(),
                    "wave_B_released": (args.output_dir/"wave_B.json").exists(), "all_outcomes_retained": True, "scientific_training_admitted": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
