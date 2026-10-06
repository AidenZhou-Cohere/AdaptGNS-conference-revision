#!/usr/bin/env python3
"""Root-released whole-invocation evaluation quotas; description only by default.

No training, acquisition, cross-host control, recovery or retries. Scientific
guards retain their meaning; an exhausted wall-time allocation leaves missing
work. Root supplies reviewed final-evaluator argv, never a shell command.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_graph_support_evaluation_quota_v2"
RELEASE_SCHEMA = "adaptgns_graph_support_evaluation_quota_release_v2"
COST_BASIS = "fixed_execution_allocation_not_empirical_full_horizon_forecast"
DEADLINE = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
LIFECYCLE_SHA = "c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd"
EVALUATORS = {"Sand": "952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58",
              "Goop": "cd970e04012930d896b31944271ccaf7da2b0881f22fe2f903533a8d5911dc6d"}
STAGES = (("full_rollout_test", "full-rollout", "test", 7200),
          ("same_state_valid", "same-state", "valid", 1800),
          ("same_state_test", "same-state", "test", 1800),
          ("clean_validation", "clean-validation", "valid", 900))
CLEANUP = 15
STREAM_SECONDS = sum(stage[3] + CLEANUP for stage in STAGES)
ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "LD_LIBRARY_PATH": "/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"}
SCHEDULE = {"A": [("base", 0, 0), ("mix", 0, 1), ("base", 1, 2), ("mix", 1, 3)],
            "B": [("base", 2, 0), ("mix", 2, 1)]}
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25", "relative-velocity-RMS25")


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


def utc(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Timezone-aware clock required")
    return result.astimezone(timezone.utc)


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def load_lifecycle(path):
    require(sha(path) == LIFECYCLE_SHA, "Pinned owned-process helpers differ")
    spec = importlib.util.spec_from_file_location("_quota_owned_process_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flags(command):
    require(isinstance(command, list) and all(isinstance(v, str) and v for v in command)
            and len(command) > 3 and command[2] == "--execute", "Explicit evaluator argv required")
    rest = command[3:]
    require(len(rest) % 2 == 0 and all(rest[i].startswith("--") for i in range(0, len(rest), 2)), "Named evaluator options required")
    result = dict(zip(rest[::2], rest[1::2]))
    require(len(result) * 2 == len(rest), "Duplicate evaluator option")
    return result


def validate_release(release, args, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    require(release.get("schema") == RELEASE_SCHEMA and release.get("status") == "admitted_for_execution_allocation"
            and release.get("issued_by") == "root" and release.get("cost_basis") == COST_BASIS
            and release.get("dataset") in EVALUATORS and release.get("host_role") in SCHEDULE
            and release.get("host") == socket.gethostname() and release.get("environment") == ENVIRONMENT,
            "Exact root quota release required")
    require(utc(release["compute_analysis_deadline_utc"]) == DEADLINE
            and release.get("stage_quotas_seconds") == {name: quota for name, _, _, quota in STAGES}
            and release.get("cleanup_seconds_per_invocation") == CLEANUP
            and finite(release.get("outer_processing_reserve_seconds"), True), "Fixed quotas and separate outer-work reserve required")
    error = release.get("clock_error_bound_seconds")
    require(finite(error) and error <= 5 and -5 <= (now - utc(release["process_clock_checked_utc"])).total_seconds() <= 300,
            "Fresh root process/clock assessment with error<=5seconds required")
    latest = utc(release["latest_start_utc"])
    require(now + timedelta(seconds=error) <= latest <= DEADLINE - timedelta(seconds=STREAM_SECONDS + release["outer_processing_reserve_seconds"] + error),
            "Complete six-stream allocation and outer reserve no longer fit")
    pins = release.get("files_sha256", {})
    require(isinstance(pins, dict) and pins.get(str(Path(__file__).resolve())) == sha(__file__)
            and pins.get(str(args.lifecycle_source.resolve())) == LIFECYCLE_SHA, "Quota/lifecycle source pins required")
    uuids = release.get("gpu_uuids")
    helper = load_lifecycle(args.lifecycle_source)
    require(isinstance(uuids, list) and len(uuids) == 4 and len({helper.canonical_gpu_uuid(x) for x in uuids}) == 4, "Four unique physical GPU UUIDs required")
    streams = release.get("streams", [])
    expected = SCHEDULE[release["host_role"]]
    require(len(streams) == len(expected) and [(s.get("arm"), s.get("seed"), s.get("gpu")) for s in streams] == expected,
            "Fixed host-role stream mapping required")
    shared = None
    for stream in streams:
        require(stream.get("id") == f"{stream['arm']}_seed{stream['seed']}" and len(stream.get("commands", [])) == 4, "Four fixed stages per model required")
        model_common = None
        for command, (name, mode, split, quota) in zip(stream["commands"], STAGES):
            options = flags(command)
            needed = {"--repo", "--benchmark-helper", "--benchmark-sha256", "--cohort", "--manifest", "--admission", "--structural-report",
                      "--train-admission", "--trainer-source", "--protocol", "--checkpoint", "--checkpoint-sha256", "--cohort-audit",
                      "--output-dir", "--mode", "--split", "--objective", "--arm", "--seed", "--cuda-index", "--threads", "--max-seconds"}
            if release["dataset"] == "Goop":
                needed |= {"--acquisition-report", "--context-semantics", "--auxiliary-report"}
                if split == "test":
                    needed.add("--cross-split-audit")
            require(set(options) == needed and options["--mode"] == mode and options["--split"] == split
                    and options["--objective"] == "faithful" and options["--arm"] == stream["arm"]
                    and options["--seed"] == str(stream["seed"]) and options["--cuda-index"] == str(stream["gpu"])
                    and options["--threads"] == "2" and options["--max-seconds"] == str(quota), "Fixed evaluator stage/model settings differ")
            require(Path(options["--repo"]).is_absolute(), "Absolute numerical repository required")
            require(Path(command[0]).is_absolute() and Path(command[1]).is_absolute()
                    and pins.get(command[0]) == sha(sys.executable) and pins.get(command[1]) == EVALUATORS[release["dataset"]], "Pinned Python and final evaluator required")
            file_options = needed - {"--repo", "--benchmark-sha256", "--checkpoint-sha256", "--output-dir", "--mode", "--split", "--objective", "--arm", "--seed", "--cuda-index", "--threads", "--max-seconds"}
            require(all(Path(options[k]).is_absolute() and options[k] in pins for k in file_options), "Every evaluator input file must be pinned")
            require(pins[options["--checkpoint"]] == options["--checkpoint-sha256"]
                    and pins[options["--benchmark-helper"]] == options["--benchmark-sha256"], "Command checkpoint/benchmark hashes differ")
            expected_output = args.output_dir.resolve() / "jobs" / stream["id"] / name
            require(Path(options["--output-dir"]).resolve() == expected_output, "Stage output must be new and owned by this queue")
            common = {k: options[k] for k in ("--checkpoint", "--checkpoint-sha256", "--cohort", "--protocol", "--train-admission", "--trainer-source", "--cohort-audit", "--repo", "--benchmark-helper")}
            require(model_common in (None, common), "Every stage must use the same frozen model/cohort")
            model_common = common
        cohort_common = {k: v for k, v in model_common.items() if k not in ("--checkpoint", "--checkpoint-sha256")}
        require(shared in (None, cohort_common), "All model streams must share one frozen cohort/protocol")
        shared = cohort_common
    return helper


def recheck_files(release):
    require(all(Path(path).is_absolute() and sha(path) == digest for path, digest in release["files_sha256"].items()), "Released file bytes changed")


def expected_cells(dataset, stage):
    if stage == "full_rollout_test":
        return [(i, policy) for i in range(30) for policy in POLICIES]
    frames = 320 if dataset == "Sand" else 401
    if stage == "clean_validation":
        return [(n // (frames - 6), n % (frames - 6) + 6) for n in (i * (30 * (frames - 6) - 1) // 127 for i in range(128))]
    return [(i, 7 + j * (frames - 8) // 4) for i in range(30) for j in range(5)]


def account_stage(dataset, stage, directory, outcome):
    """After all children exit: preserve rows; create coverage only, no MSE mean."""
    directory = Path(directory)
    expected = expected_cells(dataset, stage)
    rows, current = {}, None
    key = (lambda row: (row["source_index"], row["policy"])) if stage == "full_rollout_test" else (lambda row: (row["source_index"], row["target_frame"]))
    if directory.exists():
        if stage == "full_rollout_test":
            paths = list(directory.glob("trajectory_*_*.json"))
        else:
            paths = list(directory.glob("trajectory_*_target_*.json"))
        for path in paths:
            row = read(path)
            identity = key(row)
            require(identity in expected and identity not in rows and row.get("status") in ("complete", "failed"), "Unexpected/duplicate saved evaluation row")
            if stage == "full_rollout_test" and row["status"] == "complete":
                require(row.get("completed_steps") == row.get("horizon") == (314 if dataset == "Sand" else 395), "Completed rollout must cover declared full horizon")
            rows[identity] = {"state": "completed_required_outcome" if row["status"] == "complete" else "recorded_failed_outcome",
                              "path": str(path), "sha256": sha(path), "worker_status": row["status"], "failure": row.get("failure")}
        status_path = directory / "status.json"
        if status_path.exists():
            status = read(status_path)
            candidate = status.get("case") if stage == "full_rollout_test" else status.get("current")
            if candidate is not None:
                current = key(candidate)
    candidate = outcome.get("current_before_stop")
    if candidate is not None:
        current = key(candidate)
    cells = []
    for identity in expected:
        value = rows.get(identity)
        if value is None:
            state = "timed_out_current" if outcome.get("quota_expired") and identity == current else "not_completed_before_invocation_end"
            if outcome.get("state") == "never_started":
                state = "never_started"
            value = {"state": state, "missing": True}
        cells.append({"source_index": identity[0], "policy" if stage == "full_rollout_test" else "target_frame": identity[1], **value})
    return {"stage": stage, "required_cells": len(expected), "recorded_cells": len(rows), "missing_cells": len(expected) - len(rows),
            "cells": cells, "complete_sample_metrics_admitted": False, "no_survivor_mean_computed": True,
            "audit_scope": "Committed JSON row identities/status and rollout horizon only; referenced arrays and scientific metrics require root audit"}


def execute_queue(args, release, helper):
    output = args.output_dir.resolve()
    output.mkdir(mode=0o700, exist_ok=False)
    (output / "jobs").mkdir()
    (output / "logs").mkdir()
    helper.atomic_json(output / "release_snapshot.json", release)
    states = [{"stream": stream, "next_stage": 0, "child": None, "outcomes": []} for stream in release["streams"]]
    all_children = []
    started = time.perf_counter()
    abort = None
    evaluation_deadline = DEADLINE - timedelta(seconds=release["outer_processing_reserve_seconds"])
    gpu_pool, gpu_future = ThreadPoolExecutor(max_workers=1), None
    def publish():
        helper.atomic_json(output / "queue_status.json", {"schema": SCHEMA, "state": "aborting" if abort else "running",
            "elapsed_seconds": time.perf_counter() - started, "streams": [{"id": s["stream"]["id"], "next_stage": s["next_stage"],
                "active_pid": s["child"]["process"].pid if s["child"] else None, "outcomes": s["outcomes"]} for s in states], "abort_reason": abort})
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, frame):
        nonlocal abort
        abort = "Supervisor received " + signal.Signals(signum).name
    for sig in previous:
        signal.signal(sig, interrupted)
    try:
        require(not helper.gpu_processes(), "Foreign GPU work exists before evaluation queue")
        recheck_files(release)
        last_publish, last_gpu = -math.inf, -math.inf
        while any(s["child"] or (not abort and s["next_stage"] < 4) for s in states):
            now = time.perf_counter()
            if datetime.now(timezone.utc) + timedelta(seconds=release["clock_error_bound_seconds"] + CLEANUP) >= evaluation_deadline:
                abort = abort or "Reserved evaluation cutoff reached; preserving outer-work reserve"
            for state in states:
                child = state["child"]
                if child is None and not abort and state["next_stage"] < 4:
                    index = state["next_stage"]
                    remaining_allocation = sum(stage[3] + CLEANUP for stage in STAGES[index:])
                    require(datetime.now(timezone.utc) + timedelta(seconds=release["clock_error_bound_seconds"] + remaining_allocation) <= evaluation_deadline,
                            "Remaining distinct stages no longer fit before outer-work reserve")
                    if index == 0:
                        require(datetime.now(timezone.utc) + timedelta(seconds=release["clock_error_bound_seconds"]) <= utc(release["latest_start_utc"]),
                                "Initial stream launch window closed during setup")
                    name, _, _, quota = STAGES[index]
                    command = state["stream"]["commands"][index]
                    (output / "jobs" / state["stream"]["id"]).mkdir(exist_ok=True)
                    logbase = output / "logs" / (state["stream"]["id"] + "_" + name)
                    handles = (logbase.with_suffix(".stdout.txt").open("xb"), logbase.with_suffix(".stderr.txt").open("xb"))
                    begin = time.perf_counter()
                    try:
                        proc = subprocess.Popen(command, stdout=handles[0], stderr=handles[1], env={**os.environ, **ENVIRONMENT}, start_new_session=True)
                    except BaseException:
                        for handle in handles:
                            handle.close()
                        raise
                    child = {"process": proc, "identity": None, "command": command, "signals": [], "handles": handles,
                             "started": begin, "deadline": begin + quota, "stop_started": None, "quota_expired": False,
                             "stage": name, "stream_id": state["stream"]["id"], "gpu": state["stream"]["gpu"],
                             "current_before_stop": None, "output_dir": flags(command)["--output-dir"]}
                    state["child"] = child
                    all_children.append(child)  # Registered before any fallible identity check.
                    identity = helper.process_identity(proc.pid)
                    require(identity["argv"] == command and identity["ppid"] == os.getpid(), "Owned evaluator process identity differs")
                    child["identity"] = identity
                    state["next_stage"] += 1
                if child is None:
                    continue
                pid, wait_status, usage = os.wait4(child["process"].pid, os.WNOHANG)
                if pid:
                    # Exit is only observed at polling resolution. Conservatively
                    # count a late observation as quota exhausted, without claiming
                    # the child was running until this exact observation time.
                    child["quota_expired"] |= time.perf_counter() >= child["deadline"]
                    child["process"].returncode = os.waitstatus_to_exitcode(wait_status)
                    for handle in child["handles"]:
                        handle.close()
                    outcome = {"stage": child["stage"], "state": "quota_expired" if child["quota_expired"] else "exited",
                               "exit_code": child["process"].returncode, "quota_expired": child["quota_expired"],
                               "quota_stop_initiated": child["quota_expired"] and child["stop_started"] is not None,
                               "whole_invocation_seconds_upper_bound": time.perf_counter() - child["started"], "pid": pid,
                               "output_dir": child["output_dir"], "signals": child["signals"], "current_before_stop": child["current_before_stop"]}
                    state["outcomes"].append(outcome)
                    state["child"] = None
                    failure_path = Path(child["output_dir"]) / "failed_attempt.json"
                    failure = read(failure_path) if failure_path.exists() else None
                    outcome["worker_failed_attempt"] = failure
                    explicit_error = failure is not None and failure.get("error_type") not in ("KeyboardInterrupt", "TimeoutError")
                    if explicit_error or (outcome["exit_code"] != 0 and not outcome["quota_stop_initiated"]):
                        abort = abort or "Non-quota or unclassified evaluator error; root review required"
                    continue
                now = time.perf_counter()
                if (abort or now >= child["deadline"]) and child["stop_started"] is None:
                    child["stop_started"] = min(now, child["deadline"]) if not abort else now
                    child["quota_expired"] = not abort and now >= child["deadline"]
                    status_path = Path(child["output_dir"]) / "status.json"
                    if status_path.exists():
                        status = read(status_path)
                        child["current_before_stop"] = status.get("case") if child["stage"] == "full_rollout_test" else status.get("current")
                if child["stop_started"] is not None:
                    elapsed = now - child["stop_started"]
                    sent = {event["signal"] for event in child["signals"]}
                    for threshold, sig in ((0, signal.SIGINT), (5, signal.SIGTERM), (10, signal.SIGKILL)):
                        if elapsed >= threshold and signal.Signals(sig).name not in sent:
                            helper.stop_owned([child], sig)
                    require(elapsed < CLEANUP, "Owned evaluator remains unreaped after15seconds; no subsequent stage admitted")
            elapsed = time.perf_counter() - started
            if gpu_future is not None and gpu_future.done():
                known = {c["process"].pid: c for c in all_children if c["process"].returncode is None}
                # A child may exit while the read-only query is in flight.
                known.update({c["process"].pid: c for c in all_children})
                require(all(p["pid"] in known and helper.canonical_gpu_uuid(p["gpu_uuid"]) == helper.canonical_gpu_uuid(release["gpu_uuids"][known[p["pid"]]["gpu"]])
                            for p in gpu_future.result()), "Foreign GPU work or unexpected assignment detected")
                gpu_future = None
            if not abort and gpu_future is None and elapsed - last_gpu >= 30:
                gpu_future = gpu_pool.submit(helper.gpu_processes)
                last_gpu = elapsed
            if elapsed - last_publish >= 5:
                publish()
                last_publish = elapsed
            time.sleep(.2)
    except BaseException as error:
        abort = abort or f"{type(error).__name__}: {error}"
        # Best-effort bounded cleanup, with the same ownership helper and no retries.
        for child in all_children:
            if child["process"].returncode is None and child["stop_started"] is None:
                child["stop_started"] = time.perf_counter()
        while any(c["process"].returncode is None and time.perf_counter() - c["stop_started"] < CLEANUP for c in all_children):
            for child in all_children:
                if child["process"].returncode is None:
                    elapsed = time.perf_counter() - child["stop_started"]
                    sent = {event["signal"] for event in child["signals"]}
                    for threshold, sig in ((0, signal.SIGINT), (5, signal.SIGTERM), (10, signal.SIGKILL)):
                        if elapsed >= threshold and signal.Signals(sig).name not in sent:
                            helper.stop_owned([child], sig)
                    pid, status, _ = os.wait4(child["process"].pid, os.WNOHANG)
                    if pid:
                        child["process"].returncode = os.waitstatus_to_exitcode(status)
            time.sleep(.1)
    finally:
        gpu_pool.shutdown(wait=False, cancel_futures=True)
        for child in all_children:
            for handle in child["handles"]:
                handle.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    unreaped = [{"pid": c["process"].pid, "command": c["command"], "signals": c["signals"]} for c in all_children if c["process"].returncode is None]
    for state in states:
        for child in all_children:
            if child["stream_id"] == state["stream"]["id"] and not any(o["stage"] == child["stage"] for o in state["outcomes"]):
                state["outcomes"].append({"stage": child["stage"], "state": "aborted_by_supervisor",
                    "exit_code": child["process"].returncode, "quota_expired": child["quota_expired"],
                    "whole_invocation_seconds_upper_bound": time.perf_counter() - child["started"], "pid": child["process"].pid,
                    "output_dir": child["output_dir"], "signals": child["signals"], "current_before_stop": child["current_before_stop"]})
    helper.atomic_json(output / "process_outcomes.json", {"schema": SCHEMA, "streams": [{"id": s["stream"]["id"], "outcomes": s["outcomes"]} for s in states],
        "all_children": [{k: c[k] for k in ("identity", "command", "signals", "stage", "quota_expired", "output_dir")} for c in all_children],
        "unreaped_owned_children": unreaped, "abort_reason": abort})
    ledgers, audit_errors = [], []
    for state in states:
        for index, (name, _, _, _) in enumerate(STAGES):
            outcome = next((o for o in state["outcomes"] if o["stage"] == name), {"stage": name, "state": "never_started"})
            directory = flags(state["stream"]["commands"][index])["--output-dir"]
            entry = {"stream": state["stream"]["id"], "outcome": outcome, "stage": name}
            if unreaped:
                entry.update(coverage_audit_state="deferred_while_owned_child_unreaped", cells=None)
            else:
                try:
                    entry.update(account_stage(release["dataset"], name, directory, outcome), coverage_audit_state="row_coverage_checked")
                except Exception as error:
                    entry.update(coverage_audit_state="failed_requires_root_review", error=f"{type(error).__name__}: {error}")
                    audit_errors.append({"stream": entry["stream"], "stage": name, "error": entry["error"]})
            ledgers.append(entry)
    inputs_verified = False
    try:
        recheck_files(release)
        inputs_verified = True
    except Exception as error:
        audit_errors.append({"input_reverification_error": f"{type(error).__name__}: {error}"})
    helper.atomic_json(output / "coverage_ledger.json", {"schema": SCHEMA, "dataset": release["dataset"], "cost_basis": COST_BASIS,
        "stages": ledgers, "abort_reason": abort, "audit_errors": audit_errors, "unreaped_owned_children": unreaped, "whole_cohort_metrics_admitted": False,
        "all_required_outcomes_promised": False, "automatic_retry": False, "v1_timing_gate_reinterpreted": False})
    helper.atomic_json(output / "queue_status.json", {"schema": SCHEMA, "state": "stopped_requires_review" if abort or unreaped or audit_errors else "allocation_finished",
        "coverage_ledger_sha256": sha(output / "coverage_ledger.json"), "elapsed_seconds": time.perf_counter() - started,
        "unreaped_owned_children": unreaped, "audit_errors": audit_errors, "all_pinned_inputs_reverified": inputs_verified})
    return 1 if abort or unreaped or audit_errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    for name in ("release", "lifecycle-source", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": SCHEMA, "status": "description_only", "cost_basis": COST_BASIS,
            "stage_quotas_seconds": {name: q for name, _, _, q in STAGES}, "cleanup_seconds_per_stage": CLEANUP,
            "maximum_stream_allocation_seconds": STREAM_SECONDS, "all_required_outcomes_promised": False}, indent=2))
        return 0
    require(all((args.release, args.lifecycle_source, args.output_dir)) and sys.platform.startswith("linux")
            and hasattr(os, "wait4") and os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Explicit Linux execution arguments without device remapping required")
    release = read(args.release)
    helper = validate_release(release, args)
    return execute_queue(args, release, helper)


if __name__ == "__main__":
    raise SystemExit(main())
