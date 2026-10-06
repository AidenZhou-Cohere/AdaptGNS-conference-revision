#!/usr/bin/env python3
"""Separate Sand/Goop quota-basis scientific release; description by default.

Reuses pinned v1 owned training loop and pairing checks in a private module.
Only fixed evaluation allocation replaces the empirical evaluation-cost gate.
Goop uses explicit source/data/context/history/payload lineage checks. No GPU
launch, model/data load or scientific import occurs without --execute.
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
import sys

sys.dont_write_bytecode = True
V1_SHA = "a22a59c46231c328e4939a0f021b3bb2fab9e2f93c0848bb1abd39aad2d5d305"
PINS = {
    "Sand": {"capacity": "ca9b5ed45455b575179464d35c42d0d03e87edb7508f4deea2451f9f3fb50878", "trainer": "fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124", "frames": 320, "type": 6},
    "Goop": {"capacity": "7b75518697fe9f496dc1ea8e6a64406fffdb3b7022471b36548038c1202c2d03", "trainer": "dd9ef01a16116f04a578bbfe9ed01774e66fdc08d60b913072a33ad211b319c1", "frames": 401, "type": 7},
}
LIFECYCLE_SHA = "c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd"
COST_BASIS = "fixed_execution_allocation_not_empirical_full_horizon_forecast"
STAGE_QUOTAS = {"full_rollout_test": 7200, "same_state_valid": 1800, "same_state_test": 1800, "clean_validation": 900}
EVALUATION_ALLOCATION = 11760
EVALUATION_QUOTA_SHA = "0bd714b0bece0c349a9b18eae14221f6602ffc61ea736bad4fa3b5dc087157d9"
DEADLINE = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
UPDATES = 100000
DATASET = SCHEMA = RELEASE_SCHEMA = TRAINER_SHA = FRAMES = PARTICLE_TYPE = None
V = C = B = T = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Timezone-aware UTC evidence required")
    return result.astimezone(timezone.utc)


def private_import(path, expected, name):
    require(sha(path) == expected, "Pinned helper differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(args):
    global DATASET, SCHEMA, RELEASE_SCHEMA, TRAINER_SHA, FRAMES, PARTICLE_TYPE, V, C, B, T
    DATASET = args.dataset
    spec = PINS[DATASET]
    SCHEMA = f"adaptgns_{DATASET.lower()}_graph_support_scientific_supervisor_quota_v2"
    RELEASE_SCHEMA = f"adaptgns_{DATASET.lower()}_graph_support_scientific_release_quota_v2"
    TRAINER_SHA, FRAMES, PARTICLE_TYPE = spec["trainer"], spec["frames"], spec["type"]
    V = private_import(args.v1_source, V1_SHA, "_private_scientific_v1_lifecycle")
    C = private_import(args.capacity_source, spec["capacity"], "_private_quota_scientific_capacity")
    C.configure(args.lifecycle_source, args.trainer)
    B, T = C.B, C.T
    V.C, V.B, V.T, V.SCHEMA, V.TRAINER_SHA = C, B, T, SCHEMA, TRAINER_SHA
    V.verify_job = verify_job
    V.fixed_command = fixed_command
    B.verify_job = V.record_reaped_job
    B.fixed_command = fixed_command


def fixed_command(args, job):
    command = [str(args.python), "-u", str(args.trainer), "--execute", "--repo", str(args.repo),
            "--train-manifest", str(args.train_manifest), "--admission", str(args.admission),
            "--structural-report", str(args.structural_report), "--protocol", str(args.protocol),
            "--output-dir", str(args.output_dir / "jobs" / job["id"]), "--objective", "faithful",
            "--arm", job["arm"], "--seed", str(job["seed"]), "--cuda-index", str(job["gpu"]),
            "--updates", "100000", "--threads", "2", "--checkpoint-every", "10000", "--log-every", "100"]
    if DATASET == "Goop":
        for name in ("acquisition_report", "context_semantics", "auxiliary_report"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return command


def cost_contract(costs):
    require(costs.get("schema") == f"adaptgns_{DATASET.lower()}_graph_support_execution_allocation_v2"
            and costs.get("status") == "fixed_allocation_declared" and costs.get("issued_by") == "root"
            and costs.get("cost_basis") == COST_BASIS and costs.get("dataset") == DATASET
            and costs.get("evaluation_quota_supervisor_sha256") == EVALUATION_QUOTA_SHA
            and costs.get("stage_quotas_seconds") == STAGE_QUOTAS and costs.get("cleanup_seconds_per_invocation") == 15
            and costs.get("evaluation_stream_allocation_seconds") == EVALUATION_ALLOCATION
            and costs.get("cohort_verification_reserve_seconds") == 2700 and costs.get("analysis_reserve_seconds") == 3600
            and costs.get("all_required_evaluation_outcomes_promised") is False
            and costs.get("v1_timing_gate_reinterpreted") is False
            and B.finite(costs.get("ledger_total_reserve_seconds"), True)
            and isinstance(costs.get("ledger_bound_rationale"), str) and bool(costs["ledger_bound_rationale"].strip())
            and isinstance(costs.get("ledger_evidence_sha256"), list) and bool(costs["ledger_evidence_sha256"])
            and all(T.digest_string(x) for x in costs["ledger_evidence_sha256"]),
            "Explicit quota allocation and independently justified training-ledger reserve required")
    return costs


def forecast_contract(forecast, costs, costs_sha):
    cost_contract(costs)
    require(forecast.get("schema") == C.SCHEMA and forecast.get("status") == "all_six_verified"
            and forecast.get("scientific_training_admitted") is False, "Distinct complete six-job capacity evidence required")
    jobs = forecast.get("jobs", [])
    require(len(jobs) == 6 and {j.get("id") for j in jobs} == {j["id"] for j in C.SCHEDULE}, "All six measured graph-support jobs required")
    for measured in jobs:
        planned = next(j for j in C.SCHEDULE if j["id"] == measured["id"])
        require(all(measured.get(k) == v for k, v in planned.items())
                and measured.get("status") == "verified_512_graph_timing_probe"
                and B.finite(measured.get("steady_wall_seconds_per_update"), True)
                and B.finite(measured.get("nonnegative_external_minus_all_guarded_seconds")), "Capacity job lineage or timing differs")
    q = max(j["steady_wall_seconds_per_update"] for j in jobs)
    r = max(j["nonnegative_external_minus_all_guarded_seconds"] for j in jobs)
    require(forecast.get("forecast", {}).get("q6") == q and forecast["forecast"].get("r6") == r,
            "Original measured capacity q6/r6 differ")
    training = 1.35 * (UPDATES * q + 12 * r) + costs["ledger_total_reserve_seconds"]
    remaining = EVALUATION_ALLOCATION + costs["cohort_verification_reserve_seconds"] + costs["analysis_reserve_seconds"]
    total = training + remaining
    return {"cost_basis": COST_BASIS, "allocation_sha256": costs_sha, "q6": q, "r6": r,
            "training_seconds": training, "remaining_evaluation_verification_seconds": remaining,
            "total_seconds": total, "latest_start_utc": DEADLINE - timedelta(seconds=total),
            "training_stop_utc": DEADLINE - timedelta(seconds=remaining),
            "all_required_evaluation_outcomes_promised": False, "v1_timing_gate_reinterpreted": False}


def validate_release(release, hashes, args, process, clock, forecast, costs, now=None):
    now = B.now() if now is None else now
    budget = forecast_contract(forecast, costs, hashes["costs"])
    require(release.get("schema") == RELEASE_SCHEMA and release.get("status") == "admitted_for_scientific_training"
            and release.get("issued_by") == "root" and release.get("scientific_training_admitted") is True
            and release.get("cost_basis") == COST_BASIS and release.get("dataset") == DATASET
            and release.get("all_required_evaluation_outcomes_promised") is False
            and release.get("v1_timing_gate_reinterpreted") is False
            and release.get("files_sha256") == hashes and release.get("schedule") == C.SCHEDULE
            and release.get("environment") == C.ENVIRONMENT and release.get("host_role") == args.host_role
            and release.get("host") == socket.gethostname()
            and isinstance(release.get("cohort_id"), str) and bool(release["cohort_id"].strip())
            and isinstance(release.get("review_rationale"), str) and bool(release["review_rationale"].strip()),
            "Exact root scientific release required")
    uuids = release.get("gpu_uuids")
    require(isinstance(uuids, list) and len(uuids) == 4 and len({B.canonical_gpu_uuid(u) for u in uuids}) == 4,
            "Four unique physical GPU UUIDs required")
    require(process.get("schema") == f"adaptgns_{DATASET.lower()}_scientific_process_check_v1"
            and process.get("host") == release["host"] and process.get("gpu_uuids") == uuids
            and process.get("matching_training_processes") == [] and process.get("gpu_processes") == [],
            "Root process check must identify this idle host and GPU inventory")
    require(clock.get("schema") == f"adaptgns_{DATASET.lower()}_scientific_clock_check_v1"
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
            require(trajectory in counts and target.isdigit() and 6 <= int(target) < FRAMES
                    and record["examples"][slot]["n_particles"] == counts[trajectory], "Graph ledger frame/particle identity differs")
    return {"graph_updates": UPDATES, "logged_rows": len(rows), "local_lr_recomputation_discrepancies": lr_differences}


def check_payload(torch, payload, config, completed, history):
    require(payload.get("format_version") == 2 and payload.get(f"cuda_{DATASET.lower()}_graph_support_schema") == T.SCHEMA
            and not any(k in payload for k in ("cuda_sand_training_schema", "cuda_goop_training_schema", "full_training_schema", "graph_support_schema",
                "cuda_goop_graph_support_schema" if DATASET == "Sand" else "cuda_sand_graph_support_schema"))
            and type(payload.get("completed_steps")) is int and payload["completed_steps"] == completed
            and payload.get("run_config") == config and payload.get("run_config_sha256") == B.canonical_hash(config)
            and payload.get("training_config") == {"loss": "faithful", f"cuda_{DATASET.lower()}_graph_support_run": config,
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
    require(config.get("schema") == T.SCHEMA and config.get("dataset") == DATASET and config.get("objective") == "faithful"
            and config.get("arm") == job["arm"] and config.get("seed") == job["seed"] and config.get("updates") == UPDATES
            and config.get("checkpoint_every") == 10000 and config.get("log_every") == 100
            and config.get("research_protocol_sha256") == protocol_sha
            and config.get("initialization") == "from scratch; paired seed across arms; empty Adam; no parent checkpoint"
            and config.get("graph_exposure") == T.graph_exposure_config(job["arm"])
            and config.get("source_sha256") == {**T.SOURCE_PINS, f"train_{DATASET.lower()}_graph_support_cuda.py": TRAINER_SHA},
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
        and data.get("frames_per_trajectory") == FRAMES and data.get("particle_type_ids") == [PARTICLE_TYPE], "Scientific data lineage differs")
    if DATASET == "Goop":
        require(data.get("source") == T.OFFICIAL_TRAIN_SOURCE and data.get("n_trajectories") == 1000
                and data.get("eligible_frames") == 395000 and data.get("forecast_horizon_after_six_frames") == 395
                and all(data.get(key) == value for key, value in {
                    "metadata_sha256": T.METADATA_SHA, "reader_sha256": T.READER_SHA, "converter_sha256": T.CONVERTER_SHA,
                    "acquisition_report_sha256": T.ACQUISITION_SHA, "context_semantics_sha256": T.CONTEXT_SEMANTICS_SHA,
                    "auxiliary_report_sha256": T.AUXILIARY_REPORT_SHA, "context_source_sha256": T.CONTEXT_SOURCE_PINS}.items())
                and data.get("auxiliary_policy") == {"name": "step_context", "handling": "preserved_and_excluded_from_model_inputs",
                    "reason": T.OMISSION_REASON, "official_commit": T.OFFICIAL_COMMIT}, "Scientific Goop context/source lineage differs")
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dataset", choices=("Sand", "Goop"), default="Sand")
    parser.add_argument("--host-role", choices=("A", "B"))
    fields = ("release", "repo", "trainer", "capacity-source", "lifecycle-source", "v1-source", "train-manifest", "admission",
              "structural-report", "protocol", "forecast", "costs", "process-check", "clock-check", "python", "output-dir")
    extra = ("acquisition-report", "context-semantics", "auxiliary-report")
    for name in fields + extra:
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    required = fields + (extra if args.dataset == "Goop" else ())
    if args.execute and (args.host_role is None or any(getattr(args, name.replace("-", "_")) is None for name in required)):
        parser.error("Scientific execution requires every explicit dataset/release/source/admission/capacity/clock/output argument")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": f"adaptgns_{args.dataset.lower()}_graph_support_scientific_supervisor_quota_v2", "status": "description_only", "updates": UPDATES, "host_roles": "A:base0/mix0/base1/mix1; B:base2/mix2",
            "cost_basis": COST_BASIS, "dataset": args.dataset, "pinned_sources": PINS, "v1_source_sha256": V1_SHA,
            "evaluation_stream_allocation_seconds": EVALUATION_ALLOCATION, "all_required_evaluation_outcomes_promised": False,
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
    paths["v1_source"] = args.v1_source.resolve()
    if DATASET == "Goop":
        paths.update({name: getattr(args, name).resolve() for name in ("acquisition_report", "context_semantics", "auxiliary_report")})
        paths.update(C.context_source_paths(args.context_semantics))
    hashes = {key: sha(path) for key, path in paths.items()}
    require(hashes["python"] == sha(Path(sys.executable)), "Supervisor and declared trainer Python bytes must match")
    require(all(hashes[key] == value for key, value in C.DATA_PINS.items()), "Frozen scientific data admission differs")
    require(all(sha(args.repo / name) == value for name, value in T.SOURCE_PINS.items()), "Frozen numerical core differs")
    T.validate_manifest_contract(read(args.train_manifest), read(args.admission))
    if DATASET == "Goop":
        T.verify_goop_evidence(args, read(args.train_manifest), read(args.admission))
    costs = cost_contract(read(args.costs))
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
              "cost_basis": COST_BASIS, "budget": {k: v.isoformat() if isinstance(v, datetime) else v for k, v in budget.items()},
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
        V.verify_bound_inputs(args, paths, hashes, snapshots, launch, release)
        jobs, pairing = V.run_host(args, launch, read(args.train_manifest), budget["training_stop_utc"])
        V.verify_bound_inputs(args, paths, hashes, snapshots, launch, release)
        summary = {"schema": SCHEMA, "status": "verified_host_scientific_endpoints", "host_role": args.host_role,
                   "cohort_id": release["cohort_id"], "cost_basis": COST_BASIS, "v1_timing_gate_reinterpreted": False,
                   "all_required_evaluation_outcomes_promised": False, "jobs": jobs, "pairing": pairing, "all_inputs_reverified": True,
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
