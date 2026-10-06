#!/usr/bin/env python3
"""Separate root-released scientific Sand supervisor; description by default.

One host role, fresh faithful base/mix models, exactly100000 updates. No network,
cross-host process control, automatic retry/recovery, checkpoint promotion or
evaluation. Reuses pinned owned-child identity/capture/reap/cleanup helpers.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_graph_support_scientific_supervisor_v1"
RELEASE_SCHEMA = "adaptgns_sand_graph_support_scientific_release_v1"
CAPACITY_SHA = "ca9b5ed45455b575179464d35c42d0d03e87edb7508f4deea2451f9f3fb50878"
TRAINER_SHA = "fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124"
LIFECYCLE_SHA = "c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd"
DEADLINE = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
UPDATES = 100000
OBSERVE_SECONDS = 30
C = B = T = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Timezone-aware UTC evidence required")
    return result.astimezone(timezone.utc)


def configure(args):
    global C, B, T
    require(sha(args.capacity_source) == CAPACITY_SHA, "Pinned capacity adapter differs")
    spec = importlib.util.spec_from_file_location("_scientific_capacity_helpers", args.capacity_source)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    C.configure(args.lifecycle_source, args.trainer)
    B, T = C.B, C.T
    B.verify_job = record_reaped_job
    B.fixed_command = fixed_command
    # The original run_wave is not used: its 3600-second bound and one-second
    # full observation rewrites were designed for512-update timing only.


def fixed_command(args, job):
    return [str(args.python), "-u", str(args.trainer), "--execute", "--repo", str(args.repo),
            "--train-manifest", str(args.train_manifest), "--admission", str(args.admission),
            "--structural-report", str(args.structural_report), "--protocol", str(args.protocol),
            "--output-dir", str(args.output_dir / "jobs" / job["id"]), "--objective", "faithful",
            "--arm", job["arm"], "--seed", str(job["seed"]), "--cuda-index", str(job["gpu"]),
            "--updates", "100000", "--threads", "2", "--checkpoint-every", "10000", "--log-every", "100"]


def forecast_contract(forecast, costs, costs_sha):
    require(forecast.get("schema") == C.SCHEMA and forecast.get("status") == "all_six_verified"
            and forecast.get("scientific_training_admitted") is False
            and forecast.get("cost_estimate_sha256") == costs_sha, "Complete distinct graph-support capacity forecast required")
    jobs = forecast.get("jobs", [])
    require(len(jobs) == 6 and {j.get("id") for j in jobs} == {j["id"] for j in C.SCHEDULE},
            "All six measured graph-support jobs required")
    for measured in jobs:
        planned = next(j for j in C.SCHEDULE if j["id"] == measured["id"])
        require(all(measured.get(k) == v for k, v in planned.items())
                and measured.get("status") == "verified_512_graph_timing_probe", "Capacity job lineage differs")
    f = forecast["forecast"]
    require(f.get("costs") == costs and f.get("verification_allowance_seconds") == 3600
            and f.get("fits_before_writing_reserve") is True and stamp(f["deadline_utc"]) == DEADLINE,
            "Complete measured rollout/diagnostics/ledger forecast and deadline fit required")
    q = max(j["steady_wall_seconds_per_update"] for j in jobs)
    r = max(j["nonnegative_external_minus_all_guarded_seconds"] for j in jobs)
    require(B.finite(q, True) and B.finite(r), "Invalid six-job timing")
    training = 1.35 * (UPDATES * q + 12 * r) + costs["ledger_total_reserve_seconds"]
    remaining = costs["full_rollout_seconds"] + costs["diagnostics_execution_seconds"] + 3600
    total = training + remaining
    for key, expected in (("q6", q), ("r6", r), ("training_including_terminal_ledger_seconds", training),
                          ("total_compute_analysis_seconds", total)):
        require(B.finite(f.get(key), key != "r6") and math.isclose(f[key], expected, rel_tol=1e-12, abs_tol=1e-9),
                "Complete forecast arithmetic differs: " + key)
    require(stamp(f["forecast_finish_utc"]) <= DEADLINE, "Forecast was not admitted before compute cutoff")
    return {"training_seconds": training, "remaining_evaluation_verification_seconds": remaining,
            "total_seconds": total, "latest_start_utc": DEADLINE - timedelta(seconds=total),
            "training_stop_utc": DEADLINE - timedelta(seconds=remaining)}


def validate_release(release, hashes, args, process, clock, forecast, costs, now=None):
    now = B.now() if now is None else now
    budget = forecast_contract(forecast, costs, hashes["costs"])
    require(release.get("schema") == RELEASE_SCHEMA and release.get("status") == "admitted_for_scientific_training"
            and release.get("issued_by") == "root" and release.get("scientific_training_admitted") is True
            and release.get("files_sha256") == hashes and release.get("schedule") == C.SCHEDULE
            and release.get("environment") == C.ENVIRONMENT and release.get("host_role") == args.host_role
            and release.get("host") == socket.gethostname()
            and isinstance(release.get("cohort_id"), str) and bool(release["cohort_id"].strip())
            and isinstance(release.get("review_rationale"), str) and bool(release["review_rationale"].strip()),
            "Exact root scientific release required")
    uuids = release.get("gpu_uuids")
    require(isinstance(uuids, list) and len(uuids) == 4 and len({B.canonical_gpu_uuid(u) for u in uuids}) == 4,
            "Four unique physical GPU UUIDs required")
    require(process.get("schema") == "adaptgns_sand_scientific_process_check_v1"
            and process.get("host") == release["host"] and process.get("gpu_uuids") == uuids
            and process.get("matching_training_processes") == [] and process.get("gpu_processes") == [],
            "Root process check must identify this idle host and GPU inventory")
    require(clock.get("schema") == "adaptgns_sand_scientific_clock_check_v1"
            and clock.get("issued_by") == "root" and clock.get("host") == release["host"]
            and clock.get("root_host_samples_reviewed") is True
            and B.finite(clock.get("clock_error_bound_seconds")) and clock["clock_error_bound_seconds"] <= 5,
            "Root-reviewed host clock evidence required")
    for evidence in (process, clock):
        require(-5 <= (now - stamp(evidence["checked_utc"])).total_seconds() <= 300, "Fresh process/clock evidence required")
    require(release.get("process_identity_checked_utc") == process["checked_utc"]
            and release.get("clock_checked_utc") == clock["checked_utc"]
            and release.get("clock_error_bound_seconds") == clock["clock_error_bound_seconds"],
            "Release clock/process evidence differs")
    latest = stamp(release["latest_start_utc"])
    stop = stamp(release["training_stop_utc"])
    cleanup_trigger = stop - timedelta(seconds=clock["clock_error_bound_seconds"] + B.CLEANUP_SECONDS)
    require(stamp(release["compute_analysis_deadline_utc"]) == DEADLINE
            and latest <= budget["latest_start_utc"] and stop <= budget["training_stop_utc"]
            and latest <= cleanup_trigger - timedelta(seconds=budget["training_seconds"])
            and now + timedelta(seconds=clock["clock_error_bound_seconds"]) <= latest < stop <= DEADLINE,
            "Launch no longer fits complete workload or training/evaluation reserve")
    return {**budget, "latest_start_utc": latest, "training_stop_utc": stop,
            "cleanup_trigger_utc": cleanup_trigger, "gpu_uuids": uuids}


def record_reaped_job(directory, job, external, manifest, protocol_sha, gpu_uuid):
    """Cheap exit checks only while peers may still be training."""
    external["nominal_poll_interval_seconds"] = .2
    require(external.get("exit_code") == 0 and not external.get("signals"), "Scientific child did not exit cleanly")
    status = read(Path(directory) / "status.json")
    require(status.get("state") == "complete" and status.get("error") is None
            and status.get("completed_steps") == status.get("committed_steps") == status.get("requested_steps") == UPDATES,
            "Clean child must report a complete100k endpoint before peer monitoring continues")
    return {**job, "status": "reaped_pending_endpoint_verification", "directory": str(directory), "external": external}


def verify_bound_inputs(args, paths, hashes, snapshots, launch, release):
    require(all(sha(path) == hashes[key] for key, path in paths.items())
            and all(sha(args.repo / name) == value for name, value in T.SOURCE_PINS.items())
            and all(sha(args.output_dir / name) == hashes[key] for key, name in snapshots.items())
            and sha(args.release) == sha(args.output_dir / "inputs/release.json") == launch["release_sha256"]
            and read(args.release) == release, "Live or copied scientific input/release/source changed")


def check_history(history, job, manifest):
    T.validate_graph_history(history, UPDATES, job["arm"], job["seed"])
    rows = history["training"]
    expected_steps = [1, *range(100, UPDATES + 1, 100)]
    require([r.get("completed_steps") for r in rows] == expected_steps, "Exact scientific logging cadence required")
    counts = {r["id"]: r["positions"]["shape"][1] for r in manifest["records"]}
    previous, lr_differences = 0., []
    for row in rows:
        step = row["completed_steps"]
        require(set(row) == {"completed_steps", "loss", "lr", "frame_ids", "particles", "guarded_update_seconds", "elapsed_seconds"}
                and math.isfinite(row["loss"]) and B.finite(row["guarded_update_seconds"], True)
                and B.finite(row["elapsed_seconds"], True) and row["elapsed_seconds"] > previous,
                "Malformed/nonfinite scientific training row")
        previous = row["elapsed_seconds"]
        rate = 1e-4 * (1e-5 / 1e-4) ** ((step - 1) / 99999)
        require(abs(row["lr"] - rate) <= math.ulp(rate), "Logged LR differs beyond one local float64 ULP")
        if row["lr"] != rate:
            lr_differences.append({"completed_steps": step, "saved_lr": row["lr"], "local_recomputed_lr": rate})
        record = history["graph_updates"][step - 1]
        require(row["frame_ids"] == record["frame_ids"] and row["particles"] == sum(e["n_particles"] for e in record["examples"]),
                "Scalar and graph history differ")
    for record in history["graph_updates"]:
        for slot, identity in enumerate(record["frame_ids"]):
            trajectory, target = identity.rsplit(":", 1)
            require(trajectory in counts and target.isdigit() and 6 <= int(target) < 320
                    and record["examples"][slot]["n_particles"] == counts[trajectory], "Graph ledger frame/particle identity differs")
    return {"graph_updates": UPDATES, "logged_rows": len(rows), "local_lr_recomputation_discrepancies": lr_differences}


def check_payload(torch, payload, config, completed, history):
    require(payload.get("format_version") == 2 and payload.get("cuda_sand_graph_support_schema") == T.SCHEMA
            and not any(k in payload for k in ("cuda_sand_training_schema", "full_training_schema", "graph_support_schema"))
            and type(payload.get("completed_steps")) is int and payload["completed_steps"] == completed
            and payload.get("run_config") == config and payload.get("run_config_sha256") == B.canonical_hash(config)
            and payload.get("training_config") == {"loss": "faithful", "cuda_sand_graph_support_run": config,
                "completed_optimizer_updates": completed} and payload.get("history") == history,
            "Checkpoint lineage, endpoint, history or configuration differs")
    require(len(history.get("graph_updates", [])) == completed and B.finite(history.get("elapsed_seconds")),
            "Checkpoint graph history count or elapsed time differs")
    simulator = payload.get("simulator_config", {})
    expected = {"particle_dimensions": 2, "nnode_in": 30, "nedge_in": 3, "latent_dim": 128, "nmessage_passing_steps": 10,
                "nmlp_layers": 2, "mlp_hidden_dim": 128, "connectivity_radius": .015, "nparticle_types": 9,
                "particle_type_embedding_size": 16, "uncertainty_parameterization": "variance", "variance_floor": 1e-6,
                "max_num_neighbors": 128, "detach_variance_features": True, "radius_backend": "scipy_host",
                "boundaries": [[.1, .9], [.1, .9]], "boundary_clamp_limit": 1.}
    require(set(simulator) == {*expected, "normalization_stats"} and all(simulator.get(k) == v for k, v in expected.items()),
            "Final simulator recipe differs")
    normalization = simulator["normalization_stats"]
    require(isinstance(normalization, dict) and set(normalization) == {"acceleration", "velocity"}, "Normalization families differ")
    for values in normalization.values():
        require(isinstance(values, dict) and set(values) == {"mean", "std"} and all(isinstance(t, torch.Tensor)
                and t.dtype == torch.float32 and t.device.type == "cpu" and t.shape == (2,)
                and bool(torch.isfinite(t).all()) for t in values.values()) and bool((values["std"] > 0).all()),
                "Saved normalization must be finite positive-scale two-dimensional float32")
    state, optimizer = payload.get("state_dict"), payload.get("optimizer_state", {})
    require(isinstance(state, dict) and bool(state) and all(isinstance(t, torch.Tensor)
            and t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in state.values()), "Model tensors must all be finite float32")
    groups, moments = optimizer.get("param_groups", []), optimizer.get("state", {})
    require(len(groups) == 1 and len(groups[0].get("params", [])) == len(state)
            and len(set(groups[0]["params"])) == len(state), "Adam parameter coverage differs")
    group = groups[0]
    settings = {"betas": (.9, .999), "eps": 1e-8, "weight_decay": 0., "amsgrad": False, "maximize": False,
                "foreach": False, "capturable": False, "differentiable": False, "fused": False}
    require(all(group.get(k) == v for k, v in settings.items()), "Adam settings differ")
    expected_lr = 1e-4 * (.1 ** (max(completed - 1, 0) / 99999))
    require(group.get("lr") == expected_lr, "Checkpoint Adam learning rate differs")
    if completed == 0:
        require(not moments, "Fresh initialization requires empty Adam state")
    else:
        require(set(moments) == set(group["params"]), "Every parameter requires Adam moments")
        for identity, parameter in zip(group["params"], state.values()):
            entry = moments[identity]
            require(set(entry) == {"step", "exp_avg", "exp_avg_sq"} and isinstance(entry["step"], torch.Tensor)
                    and entry["step"].device.type == "cpu" and entry["step"].numel() == 1
                    and float(entry["step"]) == completed, "Exact Adam endpoint required")
            for key in ("exp_avg", "exp_avg_sq"):
                value = entry[key]
                require(isinstance(value, torch.Tensor) and value.dtype == parameter.dtype and value.shape == parameter.shape
                        and bool(torch.isfinite(value).all()) and (key != "exp_avg_sq" or bool((value >= 0).all())),
                        "Invalid Adam moment shape, precision or finiteness")
    rng = payload.get("rng_states", {})
    require(set(rng) == {"cpu", "cuda"} and all(isinstance(t, torch.Tensor) and t.dtype == torch.uint8
            and t.device.type == "cpu" and t.ndim == 1 and t.numel() > 0 for t in rng.values()), "CPU/CUDA RNG serialization missing")
    return {"completed_steps": completed, "finite_model_tensors": len(state), "adam_parameter_states": len(moments),
            "cpu_cuda_rng_serialized": True, "checkpoint_schema_verified": True}


def verify_job(directory, job, external, manifest, protocol_sha, gpu_uuid):
    import torch  # CPU checkpoint inspection only; no model construction or CUDA call.
    directory = Path(directory)
    require(external.get("exit_code") == 0 and not external.get("signals"), "Scientific child did not exit cleanly")
    require(not (directory / "run.lock").exists() and not list(directory.rglob("*.tmp"))
            and not list(directory.rglob("unsuccessful*")) and not list(directory.rglob("state_preservation_error*")),
            "Failed or partial scientific attempt cannot pass")
    config, status, history, pointer = [read(directory / name) for name in ("protocol.json", "status.json", "history.json", "latest.json")]
    require(config.get("schema") == T.SCHEMA and config.get("dataset") == "Sand" and config.get("objective") == "faithful"
            and config.get("arm") == job["arm"] and config.get("seed") == job["seed"] and config.get("updates") == UPDATES
            and config.get("checkpoint_every") == 10000 and config.get("log_every") == 100
            and config.get("research_protocol_sha256") == protocol_sha
            and config.get("initialization") == "from scratch; paired seed across arms; empty Adam; no parent checkpoint"
            and config.get("graph_exposure") == T.graph_exposure_config(job["arm"])
            and config.get("source_sha256") == {**T.SOURCE_PINS, "train_sand_graph_support_cuda.py": TRAINER_SHA},
            "Scientific trainer lineage/recipe differs")
    data, runtime = config.get("data", {}), config.get("runtime", {})
    require(config.get("batch_size") == 2 and config.get("history") == 6 and config.get("noise_std") == 6.7e-4
            and config.get("architecture") == {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2}
            and config.get("graph") == {"radius": .015, "backend": "scipy_host", "cap": 128, "self_candidates": True, "augmentation_probability": 0.}
            and config.get("optimizer") == {"name": "Adam", "initial_lr": 1e-4, "final_lr": 1e-5, "decay_updates": 100000,
                "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0., "foreach": False, "fused": False, "gradient_clipping": None},
            "Scientific architecture/noise/optimizer metadata differs")
    require(all(data.get(k) == v for k, v in (("manifest_sha256", C.DATA_PINS["train_manifest"]),
        ("admission_sha256", C.DATA_PINS["admission"]), ("structural_report_sha256", C.DATA_PINS["structural_report"])))
        and data.get("frames_per_trajectory") == 320 and data.get("particle_type_ids") == [6], "Scientific data lineage differs")
    require(runtime.get("device") == f"cuda:{job['gpu']}" and B.canonical_gpu_uuid(runtime.get("uuid")) == B.canonical_gpu_uuid(gpu_uuid)
            and runtime.get("torch") == "2.13.0+cu129" and runtime.get("cuda") == "12.9" and runtime.get("threads") == 2
            and runtime.get("deterministic_algorithms") is True and runtime.get("deterministic_warn_only") is False
            and runtime.get("cublas_workspace_config") == C.ENVIRONMENT["CUBLAS_WORKSPACE_CONFIG"]
            and all(runtime.get(k) is False for k in ("tf32", "amp", "compile", "ddp")), "Scientific CUDA provenance differs")
    config_sha = B.canonical_hash(config)
    require(status.get("schema") == T.SCHEMA and status.get("state") == "complete" and status.get("error") is None
            and status.get("completed_steps") == status.get("committed_steps") == status.get("requested_steps") == UPDATES
            and status.get("run_config_sha256") == config_sha and status.get("arm") == job["arm"] and status.get("seed") == job["seed"]
            and status.get("objective") == "faithful" and status.get("process", {}).get("pid") == external["pid"], "Only complete scientific100k endpoints pass")
    attempts = list((directory / "attempts").iterdir())
    require(len(attempts) == 1 and attempts[0].name == status.get("attempt_id")
            and read(attempts[0] / "status.json") == status, "Exactly one fresh consistent attempt required")
    details = check_history(history, job, manifest)
    stdout = [json.loads(line) for line in Path(external["stdout_file"]).read_text().splitlines() if line.strip()]
    require(stdout == history["training"] and status.get("last_training") == stdout[-1], "Scientific stdout/status/history differ")
    expected_files = {f"checkpoint-{step:09d}.pt" for step in range(0, UPDATES + 1, 10000)}
    require({p.name for p in directory.glob("checkpoint-*.pt")} == expected_files, "Exact initial and10k checkpoint cadence required")
    checkpoint = directory / f"checkpoint-{UPDATES:09d}.pt"
    expected = {"path": checkpoint.name, "completed_steps": UPDATES, "run_config_sha256": config_sha, "sha256": sha(checkpoint)}
    require(pointer == expected and status.get("latest_checkpoint") == pointer, "Final checkpoint byte/pointer identity differs")
    initial = external.get("initial_pointer")
    require(initial == {"path": "checkpoint-000000000.pt", "completed_steps": 0, "run_config_sha256": config_sha,
                        "sha256": sha(directory / "checkpoint-000000000.pt")}, "Fresh initial checkpoint observation required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    endpoint = check_payload(torch, payload, config, UPDATES, history)
    final_simulator = payload["simulator_config"]
    del payload
    initial_payload = torch.load(directory / initial["path"], map_location="cpu", weights_only=True)
    require(initial_payload["history"] == {"training": [], "graph_updates": [], "elapsed_seconds": initial_payload["history"]["elapsed_seconds"]},
            "Initial checkpoint must precede every update")
    initial_check = check_payload(torch, initial_payload, config, 0, initial_payload["history"])
    require(T.tree_equal(torch, initial_payload["simulator_config"], final_simulator),
            "Initial/final saved simulator normalization/configuration changed")
    require(sha(checkpoint) == expected["sha256"] and sha(directory / initial["path"]) == initial["sha256"], "Checkpoint changed during inspection")
    return {**job, "status": "verified_scientific100k_endpoint", "directory": str(directory), "config_sha256": config_sha,
            "runtime": runtime, "external": external, "initial_pointer": initial, "final_pointer": pointer,
            "history": details, "endpoint": endpoint, "initial_checkpoint": initial_check,
            "artifact_sha256": {name: sha(directory / name) for name in ("protocol.json", "status.json", "history.json", "latest.json")}}


def verify_pairing(jobs):
    import torch
    by_id = {j["id"]: j for j in jobs}
    require(len(by_id) == len(jobs), "Duplicate scientific job")
    results = []
    for seed in sorted({j["seed"] for j in jobs}):
        a, b = by_id.get(f"base_seed{seed}"), by_id.get(f"mix_seed{seed}")
        require(a is not None and b is not None, "Both scientific arms required per seed")
        left, right = [torch.load(Path(j["directory"]) / j["initial_pointer"]["path"], map_location="cpu", weights_only=True) for j in (a, b)]
        require(all(T.tree_equal(torch, left[key], right[key]) for key in ("state_dict", "simulator_config", "optimizer_state", "rng_states")),
                "Initial paired model/Adam/CPU-CUDA RNG tensors differ")
        del left, right
        histories = [read(Path(j["directory"]) / "history.json") for j in (a, b)]
        for lrow, rrow in zip(histories[0]["training"], histories[1]["training"]):
            require(all(lrow[k] == rrow[k] for k in ("completed_steps", "frame_ids", "particles", "lr")), "Paired saved LR/frame schedule differs")
        for lrow, rrow in zip(histories[0]["graph_updates"], histories[1]["graph_updates"]):
            require(all(lrow[k] == rrow[k] for k in ("completed_steps", "absolute_schedule_step", "frame_ids", "noise_sha256")),
                    "Paired full100k frame/noise schedule differs")
            for x, y in zip(lrow["examples"], rrow["examples"]):
                require(all(x[k] == y[k] for k in ("example_slot", "n_particles", "exposure_coin", "coin_seed_material", "pair_seed_material",
                    "native_directed_edges", "native_self_edges", "receivers_above_native_cap", "annulus_pairs", "optional_budget_if_exposed",
                    "native_edge_sha256", "noisy_current_sha256")), "Paired native graph/RNG evidence differs")
        require(all(len(h["graph_updates"]) == UPDATES and len(h["training"]) == 1001 for h in histories), "Full paired histories required")
        results.append({"seed": seed, "initial_model_adam_cpu_cuda_rng_exact": True, "all100k_frame_noise_native_graph_rng_exact": True,
                        "all_logged_saved_lr_exact": True, "initial_audit_ordering": "verified after training; no before-step1 barrier claimed"})
    return results


def run_host(args, launch, manifest, stop):
    jobs = [j for j in C.SCHEDULE if j["wave"] == args.host_role]
    require(not B.gpu_processes(), "GPU work exists; root must review before launch")
    children, reaped = [], []
    cleanup_trigger = stop - timedelta(seconds=launch["clock_error_bound_seconds"] + B.CLEANUP_SECONDS)
    started = time.perf_counter()
    record = {"wave": args.host_role, "schema": SCHEMA, "state": "running", "jobs": [], "observations": [],
              "started_utc": B.now().isoformat(), "observation_interval_seconds": OBSERVE_SECONDS,
              "cleanup_trigger_utc": cleanup_trigger.isoformat(),
              "child_outcome_verification_scope": "clean100k status only; full endpoint audits after every child is reaped"}
    path = args.output_dir / f"wave_{args.host_role}.json"
    B.atomic_json(path, record)
    try:
        for job in jobs:
            require(B.now() + timedelta(seconds=launch["clock_error_bound_seconds"]) <= stamp(launch["latest_start_utc"])
                    and B.now() < cleanup_trigger, "Agreed launch-fit window closed before next child")
            command = fixed_command(args, job)
            stdout_path = args.output_dir / "logs" / (job["id"] + ".stdout.jsonl")
            stderr_path = args.output_dir / "logs" / (job["id"] + ".stderr.txt")
            stdout, stderr = stdout_path.open("xb"), stderr_path.open("xb")
            external_started = time.perf_counter()
            try:
                proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env={**os.environ, **C.ENVIRONMENT}, start_new_session=True)
            except BaseException:
                stdout.close(); stderr.close()
                raise
            child = {"job": job, "process": proc, "identity": None, "command": command, "handles": (stdout, stderr), "signals": [],
                     "started": external_started, "started_utc": B.now().isoformat(), "stdout_file": str(stdout_path),
                     "stderr_file": str(stderr_path), "initial_pointer": None}
            children.append(child)
            identity = B.process_identity(proc.pid)
            require(identity["argv"] == command and identity["ppid"] == os.getpid(), "Launched scientific child identity differs")
            child["identity"] = identity
            B.atomic_json(args.output_dir / "logs" / (job["id"] + ".launch.json"),
                          {"job": job, "pid": proc.pid, "identity": identity, "command": command, "started_utc": child["started_utc"]})
        record["launch_skew_seconds"] = max(c["started"] for c in children) - min(c["started"] for c in children)
        last_observation = -math.inf
        while any(c["process"].returncode is None for c in children):
            require(B.now() < cleanup_trigger, "Scientific training cleanup cutoff reached; evaluation reserve retained")
            for child in children:
                if child["process"].returncode is None:
                    B.capture_initial(args, child)
                    failure = B.reap_child(args, child, record, reaped, launch, manifest)
                    require(failure is None, failure or "Scientific child failed")
            elapsed = time.perf_counter() - started
            if elapsed - last_observation >= OBSERVE_SECONDS:
                last_observation = elapsed
                known = {c["process"].pid: c for c in children}
                processes = B.gpu_processes()
                require(all(p["pid"] in known and B.canonical_gpu_uuid(p["gpu_uuid"]) ==
                    B.canonical_gpu_uuid(launch["gpu_uuids"][known[p["pid"]]["job"]["gpu"]]) for p in processes),
                    "Foreign GPU work or changed GPU assignment detected")
                states = {}
                for child in children:
                    status = args.output_dir / "jobs" / child["job"]["id"] / "status.json"
                    if status.exists():
                        value = read(status)
                        states[str(child["process"].pid)] = {k: value.get(k) for k in ("state", "completed_steps", "committed_steps")}
                        require(value.get("state") not in ("failed", "interrupted"), "Scientific child reported failure")
                record["observations"].append({"utc": B.now().isoformat(), "elapsed_seconds": elapsed,
                                               "gpu_processes": processes, "trainer_states": states})
                B.atomic_json(path, record)
            time.sleep(.2)
        require(len(reaped) == len(jobs), "Every assigned scientific child must be reaped")
        record.update(state="verifying_endpoints", all_children_reaped_utc=B.now().isoformat())
        B.atomic_json(path, record)
        verified = [verify_job(item["directory"], next(j for j in jobs if j["id"] == item["id"]), item["external"], manifest,
                               launch["files_sha256"]["protocol"], launch["gpu_uuids"][item["gpu"]]) for item in reaped]
        pairing = verify_pairing(verified)
        record.update(state="verified", pairing=pairing)
        return verified, pairing
    except BaseException as error:
        record.update(state="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        B.cleanup_owned(args, children, record, reaped, launch, manifest)
        raise
    finally:
        record.update(ended_utc=B.now().isoformat(), wave_elapsed_seconds=time.perf_counter() - started)
        B.atomic_json(path, record)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host-role", choices=("A", "B"))
    fields = ("release", "repo", "trainer", "capacity-source", "lifecycle-source", "train-manifest", "admission",
              "structural-report", "protocol", "forecast", "costs", "process-check", "clock-check", "python", "output-dir")
    for name in fields:
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.execute and (args.host_role is None or any(getattr(args, name.replace("-", "_")) is None for name in fields)):
        parser.error("Scientific execution requires role and every explicit release/source/data/forecast/process/clock/output argument")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": SCHEMA, "status": "description_only", "updates": UPDATES, "host_roles": "A:base0/mix0/base1/mix1; B:base2/mix2",
            "trainer_sha256": TRAINER_SHA, "capacity_source_sha256": CAPACITY_SHA, "lifecycle_source_sha256": LIFECYCLE_SHA,
            "checkpoint_every": 10000, "log_every": 100, "automatic_retry_resume_or_promotion": False,
            "whole_six_model_cohort_admission": "root-owned after both host endpoint/pairing reports"}, indent=2))
        return 0
    require(sys.platform.startswith("linux") and hasattr(os, "wait4"), "Scientific execution requires Linux")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Device remapping is not admitted")
    configure(args)
    keys = ("trainer", "capacity_source", "lifecycle_source", "train_manifest", "admission", "structural_report", "protocol",
            "forecast", "costs", "process_check", "clock_check", "python")
    paths = {key: getattr(args, key).resolve() for key in keys}
    paths["supervisor"] = Path(__file__).resolve()
    hashes = {key: sha(path) for key, path in paths.items()}
    require(hashes["python"] == sha(Path(sys.executable)), "Supervisor and declared trainer Python bytes must match")
    require(all(hashes[key] == value for key, value in C.DATA_PINS.items()), "Frozen scientific data admission differs")
    require(all(sha(args.repo / name) == value for name, value in T.SOURCE_PINS.items()), "Frozen numerical core differs")
    costs = C.cost_contract(args.costs)
    release = read(args.release)
    budget = validate_release(release, hashes, args, read(args.process_check), read(args.clock_check), read(args.forecast), costs)
    args.output_dir = args.output_dir.resolve()
    protected = (args.train_manifest.resolve().parent, (args.repo / "adaptive-gns").resolve())
    require(all(args.output_dir != p and p not in args.output_dir.parents and args.output_dir not in p.parents for p in protected),
            "Fresh scientific output must be separate from protected inputs")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    for name in ("inputs", "jobs", "logs"):
        (args.output_dir / name).mkdir()
    snapshots = {}
    for key, path in paths.items():
        if key != "python":
            snapshots[key] = "inputs/" + key + path.suffix
            (args.output_dir / snapshots[key]).write_bytes(path.read_bytes())
    (args.output_dir / "inputs/release.json").write_bytes(args.release.read_bytes())
    launch = {"schema": SCHEMA, "host_role": args.host_role, "cohort_id": release["cohort_id"], "hostname": socket.gethostname(),
              "files_sha256": hashes, "input_snapshots": snapshots, "release_sha256": sha(args.release),
              "gpu_uuids": budget["gpu_uuids"], "schedule": C.SCHEDULE, "environment": C.ENVIRONMENT,
              "started_utc": B.now().isoformat(), "pid": os.getpid(), "training_stop_utc": budget["training_stop_utc"].isoformat(),
              "cleanup_trigger_utc": budget["cleanup_trigger_utc"].isoformat(),
              "latest_start_utc": budget["latest_start_utc"].isoformat(), "clock_error_bound_seconds": release["clock_error_bound_seconds"],
              "commands": {j["id"]: fixed_command(args, j) for j in C.SCHEDULE if j["wave"] == args.host_role}}
    B.atomic_json(args.output_dir / "launch.json", launch)
    B.atomic_json(args.output_dir / "status.json", {"state": "running_host_scientific_training", "host_role": args.host_role,
        "cohort_id": release["cohort_id"], "started_utc": launch["started_utc"], "whole_six_model_cohort_admitted": False})
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Scientific supervisor received " + signal.Signals(signum).name)
    try:
        for sig in previous_handlers:
            signal.signal(sig, interrupted)
        # Recheck launch fit and copied bytes after setup, before any child exists.
        validate_release(release, hashes, args, read(args.process_check), read(args.clock_check), read(args.forecast), costs)
        verify_bound_inputs(args, paths, hashes, snapshots, launch, release)
        jobs, pairing = run_host(args, launch, read(args.train_manifest), budget["training_stop_utc"])
        verify_bound_inputs(args, paths, hashes, snapshots, launch, release)
        summary = {"schema": SCHEMA, "status": "verified_host_scientific_endpoints", "host_role": args.host_role,
                   "cohort_id": release["cohort_id"], "jobs": jobs, "pairing": pairing, "all_inputs_reverified": True,
                   "whole_six_model_cohort_admitted": False, "requires": "Root verifies both roles and all six endpoints before evaluation admission"}
        B.atomic_json(args.output_dir / "worker_summary.json", summary)
        B.atomic_json(args.output_dir / "status.json", {"state": "complete_host_scientific_training", "host_role": args.host_role,
            "summary_sha256": sha(args.output_dir / "worker_summary.json"), "ended_utc": B.now().isoformat(),
            "whole_six_model_cohort_admitted": False})
        return 0
    except BaseException as error:
        B.atomic_json(args.output_dir / "status.json", {"state": "failed_host_scientific_training", "host_role": args.host_role,
            "error_type": type(error).__name__, "error": str(error), "ended_utc": B.now().isoformat(), "all_outcomes_retained": True,
            "automatic_retry_or_recovery": False, "whole_six_model_cohort_admitted": False})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
