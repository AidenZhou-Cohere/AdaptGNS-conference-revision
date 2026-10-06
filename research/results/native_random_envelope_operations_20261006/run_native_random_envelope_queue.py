"""Reviewed-release-only, fresh optional native random-envelope supervisor.

No retry, recovery, automatic shortening, nested supervisor or live-child kill.
--check performs CPU admission and byte checks only and creates no output tree.
Ordinary scientific failures in complete valid publications remain outcomes and
do not stop the fixed cohort. Nonzero exit or integrity failure stops new jobs.
No scientific/helper hashes are guessed here: root supplies a final reviewed
freeze and a fresh release bound to completed native analysis and process checks.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/AdaptGNS"
OUTPUT = ROOT / "work/native-random-envelope-20261006"
NATIVE = ROOT / "work/native-graph-rollout-20261005"
TRAINING = ROOT / "work/faithful-graph-support-10k-20261006"
ENDPOINT = ROOT / "work/continuation-evaluation-20261006"
SAVED = ROOT / "work/full-evaluation/same_state"
DATA = ROOT / "work/full-data/converted"
FREEZE = Path(__file__).with_name("native_random_envelope_queue_freeze.json")
RELEASE = Path(__file__).with_name("native_random_envelope_launch_release.json")
JOBS = tuple((objective, seed) for objective in ("faithful", "nll") for seed in (0, 1, 2))
NAMES = tuple(f"{objective}_seed{seed}" for objective, seed in JOBS)
NATIVE_POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
CASES = ("base", "dense", "speed25", "previous-observed-base-risk25", "current-base-risk25") + tuple(f"random25_draw{i}" for i in range(8))
LATEST_START = datetime(2026, 10, 6, 8, tzinfo=timezone.utc)
FINISH_TARGET = datetime(2026, 10, 6, 8, 45, tzinfo=timezone.utc)
HARD_CUTOFF = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)
INITIAL_ALLOWANCE = 300
SUMMARY_RESERVE = 900
TOTAL_BUDGET = 2700
RELEASE_MAX_AGE = 900
SCOPE = "reviewed_native_random_envelope_pretraining_v1"
MODULES = {"research." + name for name in ("full_training", "run_full_queue", "run_evaluation_queue", "full_rollout", "full_same_state",
    "native_graph_rollout", "faithful_graph_support", "continuation_evaluation", "graph_convention_bridge", "native_random_envelope",
    "summarize_native_graph_rollouts", "summarize_continuation_evaluation", "summarize_native_random_envelope")}
SUPERVISORS = {f"run_{name}_queue.py" for name in ("graph_bridge", "native_rollout", "graph_support", "continuation_evaluation", "native_random_envelope")}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(), "Regular immutable file required: " + str(path))
    return path


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(token):
        raise RuntimeError("Nonfinite JSON token: " + token)
    return json.loads(regular(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def same(actual, expected):
    if type(actual) is bool or type(expected) is bool:
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) or isinstance(expected, dict):
        return isinstance(actual, dict) and isinstance(expected, dict) and set(actual) == set(expected) and all(same(actual[k], expected[k]) for k in actual)
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        return isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)) and len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def verify_pins(pins):
    for path, digest in pins.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) and sha(regular(path)) == digest, "Pinned bytes changed: " + str(path))


def merge_pins(target, extra):
    for path, digest in extra.items():
        require(path not in target or target[path] == digest, "Conflicting immutable hash: " + str(path))
        target[path] = digest


def no_locks(*roots):
    for root in roots:
        root = Path(root)
        require(not root.is_symlink(), "Linked input root")
        if root.exists():
            for path in root.rglob("*"):
                require(not path.is_symlink() and not (path.name.endswith(".lock") or ".lock." in path.name or path.name.endswith(".tmp")),
                        "Lock/recovery/temp/linked artifact requires review: " + str(path))


def pretraining_only():
    no_locks(TRAINING, ENDPOINT)
    require(not TRAINING.exists() and not ENDPOINT.exists(), "Existing training or endpoint tree blocks optional pretraining launch")


def heavy_processes(process_text, own_pid):
    found = []
    for line in process_text.splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        require(len(parts) == 2 and parts[0].isdigit(), "Unparseable process inventory")
        pid, command = int(parts[0]), parts[1]
        if pid == own_pid:
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            words = command.split()
        if not words:
            continue
        program = Path(words[0]).name
        target, module = (program if program in SUPERVISORS else None), None
        if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", program, re.IGNORECASE):
            index = 1
            while index < len(words):
                word = words[index]
                if word == "-c":
                    break
                if word == "-m":
                    module = words[index+1] if index+1 < len(words) else None; break
                if word.startswith("-m") and len(word) > 2:
                    module = word[2:]; break
                if word in ("-W", "-X"):
                    index += 2; continue
                if word == "--":
                    target = Path(words[index+1]).name if index+1 < len(words) else None; break
                if not word.startswith("-"):
                    target = Path(word).name; break
                index += 1
        if module in MODULES or target in SUPERVISORS or target in {m.split(".")[-1]+".py" for m in MODULES}:
            found.append({"pid": pid, "command": command})
    return found


def assert_idle():
    require(not heavy_processes(subprocess.check_output(["ps", "-axo", "pid=,command="], text=True), os.getpid()), "Heavy evaluator/trainer/supervisor is live")


def science():
    sys.path.insert(0, str(REPO))
    from research import native_random_envelope
    return native_random_envelope


def required_frozen_files():
    paths = {Path(__file__).resolve(), Path(__file__).with_name("test_run_native_random_envelope_queue.py").resolve()}
    paths.add(Path(__file__).with_name("test_run_native_random_envelope_queue_review.py").resolve())
    names = ("native_random_envelope.py", "summarize_native_random_envelope.py", "native_graph_rollout.py", "graph_convention_bridge.py",
        "full_same_state.py", "full_rollout.py", "full_training.py", "budget_graph.py", "protocols/native_random_envelope_20261006.md",
        "protocols/native_graph_rollout_20261005.md", "protocols/graph_convention_bridge_20261005.md", "protocols/full_waterdrop_100k.md")
    paths.update(REPO / "research" / name for name in names)
    paths.update((REPO / "research/tests").glob("test_*native_random_envelope*.py"))
    paths.update((REPO / "adaptive-gns/gns").glob("*.py"))
    paths.update(DATA / name for name in ("train.json", "valid.json", "test.json", "metadata.json"))
    for name in NAMES:
        paths.update(REPO / "research/results/full_waterdrop_100k" / name / file for file in ("checkpoint-100000.pt", "protocol.json", "latest.json", "status.json"))
        paths.update(SAVED / name / file for file in ("protocol.json", "result.json", "status.json"))
    return {str(path.relative_to(ROOT)) for path in paths}


def verify_freeze(frozen):
    require(frozen.get("schema") == 1 and frozen.get("scope") == SCOPE and frozen.get("scientific_review_complete") is True, "Final reviewed source freeze required")
    expected = {"latest_launch_utc": LATEST_START.isoformat(), "finish_target_utc": FINISH_TARGET.isoformat(),
        "initial_job_allowance_seconds": INITIAL_ALLOWANCE, "summary_audit_reserve_seconds": SUMMARY_RESERVE, "total_budget_seconds": TOTAL_BUDGET}
    require(all(same(frozen.get(k), value) for k, value in expected.items()), "Frozen forecast contract differs")
    require(set(frozen.get("files_sha256", {})) == required_frozen_files(), "Exact final scientific/helper/input freeze inventory required")
    verify_pins({str(ROOT / key): digest for key, digest in frozen["files_sha256"].items()})


def forecast(remaining, allowance, at=None, started=None, initial=False):
    at = now() if at is None else at
    require(at.tzinfo is not None and at.utcoffset() == timedelta(0), "UTC-aware clock required")
    require(type(allowance) is int and allowance >= INITIAL_ALLOWANCE, "Job allowance cannot decrease below initial300s")
    require(at < HARD_CUTOFF, "Research cutoff reached")
    if initial:
        require(at < LATEST_START, "Optional pretraining initial launch gate passed")
    seconds = len(remaining) * allowance + SUMMARY_RESERVE
    elapsed = 0. if started is None else (at-started).total_seconds()
    require(elapsed >= 0, "Planning clock moved backwards")
    result = {"checked_utc": at.isoformat(), "remaining_jobs": len(remaining), "job_allowance_seconds": allowance,
        "summary_audit_reserve_seconds": SUMMARY_RESERVE, "remaining_allowance_seconds": seconds, "elapsed_execution_budget_seconds": elapsed,
        "forecast_execution_and_analysis_seconds": elapsed+seconds, "forecast_finish_utc": (at+timedelta(seconds=seconds)).isoformat(),
        "scope": "Conservative forecast, not a guarantee.2700s covers execution and remaining summary/audit from first evaluator launch, including subsequent supervisor overhead. Admission/setup is measured separately and excluded; absolute08:45 finish still applies. No automatic child interruption or shortened scientific population."}
    require(seconds <= TOTAL_BUDGET and elapsed+seconds <= TOTAL_BUDGET and at+timedelta(seconds=seconds) <= FINISH_TARGET,
            "Execution/analysis forecast cannot preserve45minute budget and08:45 finish including900s summary/audit reserve")
    return result


def verify_release(release, native_identity, initial=True):
    require(release.get("schema") == 1 and release.get("scope") == SCOPE and release.get("root_native_identity_verified") is True,
            "Explicit root native identity/release required")
    require(release.get("freeze_sha256") == sha(FREEZE), "Release does not bind final freeze")
    require(release.get("native_queue_sha256") == native_identity["native_queue_sha256"]
            and same(release.get("native_jobs"), native_identity["native_jobs"]), "Release native identity differs")
    if initial:
        for name in ("recorded_utc", "root_process_check_utc"):
            value = datetime.fromisoformat(release[name])
            require(value.tzinfo is not None and value.utcoffset() == timedelta(0) and 0 <= (now()-value).total_seconds() <= RELEASE_MAX_AGE,
                    "Root release/process check is stale or future dated")
    pins = {}
    for prefix in ("strict_native_summary", "native_scalar_audit"):
        path = Path(release[prefix+"_path"])
        require(path.is_absolute(), "Absolute reviewed analysis path required")
        digest = release[prefix+"_sha256"]
        require(sha(regular(path)) == digest, "Released native analysis bytes differ")
        result = read(path)
        if prefix == "strict_native_summary":
            require(result.get("audit", {}).get("passed") is True and len(result.get("native_runs", [])) == 6 and len(result.get("original_runs", [])) == 6,
                    "Strict whole-cohort native summary not passing")
            require(result.get("scope") == "post-inspection native convention follow-up on original six100k models", "Released native summary scope differs")
            expected = {row["name"]:row for row in native_identity["native_jobs"]}
            require(set(expected) == set(NAMES) and len(native_identity["native_jobs"]) == 6, "Released native identity cohort differs")
            seen = set()
            for run in result["native_runs"]:
                objective, seed = run.get("objective"), run.get("seed")
                require(type(seed) is int and (objective,seed) in JOBS, "Native summary original model identity differs")
                name = f"{objective}_seed{seed}"
                require(name not in seen, "Duplicate native summary model");seen.add(name)
                directory = NATIVE/name; checkpoint = REPO/"research/results/full_waterdrop_100k"/name/"checkpoint-100000.pt"
                # Frozen native summary records have no directory field. The
                # exact absolute protocol/result pin keys below bind location.
                require(run.get("state") == "complete" and run.get("eligible_for_aggregation") is True,
                        "Native summary job completion differs")
                require(run.get("checkpoint_sha256") == sha(regular(checkpoint)), "Native summary checkpoint differs")
                require(run.get("protocol_sha256") == expected[name]["protocol_sha256"], "Native summary protocol differs")
                inputs = run.get("input_files_sha256",{})
                require(inputs.get(str((directory/"result.json").resolve())) == expected[name]["result_sha256"]
                    and inputs.get(str((directory/"protocol.json").resolve())) == expected[name]["protocol_sha256"], "Native summary raw publication binding differs")
                pins[str(checkpoint)] = sha(checkpoint)
        else:
            require(result.get("passed") is True and result.get("native_outcomes") == result.get("original_outcomes") == 810,
                    "Whole-cohort native scalar audit not passing")
            require(result.get("source_summary_sha256") == release["strict_native_summary_sha256"], "Native audit does not bind released summary")
        pins[str(path)] = digest
    return pins


def prior_complete(ev):
    no_locks(NATIVE)
    queue = read(NATIVE / "queue_status.json")
    jobs = queue.get("jobs", [])
    require(queue.get("state") == "complete" and len(jobs) == 6 and {j.get("name") for j in jobs} == set(NAMES), "All six native jobs must be complete")
    pins = {str(NATIVE / "queue_status.json"): sha(NATIVE / "queue_status.json")}; identity = []
    for name in NAMES:
        entry = next(j for j in jobs if j["name"] == name)
        require(entry.get("state") == "complete" and type(entry.get("returncode")) is int and entry["returncode"] == 0, "Native child not successful")
        directory = NATIVE / name; objective, seed_text = name.split("_seed"); seed = int(seed_text)
        protocol, result, status = [read(directory / file) for file in ("protocol.json", "result.json", "status.json")]
        checkpoint = REPO / "research/results/full_waterdrop_100k" / name / "checkpoint-100000.pt"
        require(result.get("state") == status.get("state") == "complete" and status.get("result_sha256") == sha(directory/"result.json")
            and result.get("protocol_sha256") == sha(directory/"protocol.json"), "Native committed hashes differ")
        for doc in (protocol, result):
            require(doc.get("objective") == objective and type(doc.get("seed")) is int and doc["seed"] == seed
                and doc.get("checkpoint_sha256") == sha(checkpoint), "Native original checkpoint identity differs")
        require(protocol.get("source_indices") == list(range(3, 30)) and len(protocol.get("trajectory_ids", [])) == 27
            and len(set(protocol["trajectory_ids"])) == 27, "Native trajectory population differs")
        records = result.get("records", [])
        require(len(records) == 135 and {(r["source_index"], r["policy"]) for r in records} == {(i,p) for i in range(3,30) for p in NATIVE_POLICIES}, "Native135 outcomes required")
        for row in records:
            require(type(row["source_index"]) is int, "Native integer source identity required")
            recovered = ev.full.recover_record(directory, row["source_index"], row["policy"], protocol["trajectory_ids"][row["source_index"]-3], result["protocol_sha256"], 995, row)
            require(same(recovered, row), "Native recovered/indexed record differs")
            for key, hashkey in (("record_file", "record_sha256"), ("trace_file", "trace_sha256")):
                path = directory / row[key]
                require(path.parent == directory and Path(row[key]).name == row[key], "Native unsafe artifact path")
                require(sha(regular(path)) == row[hashkey], "Native raw record/trace hash differs")
                pins[str(path)] = row[hashkey]
            require(not row.get("failure") or row["failure"].get("category") != "native_parity_failure", "Native parity failure requires review")
        for file in ("protocol.json", "result.json", "status.json"):
            pins[str(directory/file)] = sha(directory/file)
        identity.append({"name": name, "protocol_sha256": sha(directory/"protocol.json"), "result_sha256": sha(directory/"result.json")})
    return {"native_queue_sha256": pins[str(NATIVE/"queue_status.json")], "native_jobs": identity}, pins


def prepare_model_inputs(ev, objective, seed):
    """CPU checkpoint admission and frozen schedule/archive verification only."""
    name = f"{objective}_seed{seed}"; checkpoint = REPO/"research/results/full_waterdrop_100k"/name/"checkpoint-100000.pt"
    payload = ev.torch.load(checkpoint, map_location="cpu", weights_only=True)
    ev.full.check_checkpoint(payload, "locked_test", ev.TRAINING_PROTOCOL, sha(DATA/"metadata.json"))
    config = payload["run_config"]
    require(config["objective"] == objective and type(config["seed"]) is int and config["seed"] == seed, "Original model identity differs")
    checkpoint_hash = ev.committed_checkpoint(checkpoint, payload)
    paths = list(ev.source_paths()) + [checkpoint, DATA/"valid.json", DATA/"test.json", DATA/"metadata.json"]
    paths += [checkpoint.parent/file for file in ("protocol.json", "latest.json", "status.json")]
    for relative, digest in config["source_sha256"].items():
        path = regular(REPO/relative); require(sha(path) == digest, "Original frozen source changed"); paths.append(path)
    expected, valid, test = ev.bridge.expected_frames(config, DATA/"valid.json", DATA/"test.json")
    require(len(expected) == 425 and Counter(item["split"] for item in expected) == {"valid":128,"test":297}, "Complete observed schedule required")
    require(valid["metadata"] == test["metadata"] == read(DATA/"metadata.json"), "Split metadata differs")
    for manifest in (valid, test):
        for item in manifest["records"]:
            for key in ("positions", "particle_types"):
                relative = Path(item[key]["path"])
                require(not relative.is_absolute() and ".." not in relative.parts, "Data path escaped source tree")
                path = regular(DATA/relative); require(sha(path) == item[key]["sha256"], "Converted data hash differs"); paths.append(path)
    saved = SAVED/name; lookup, saved_identity = ev.bridge.verify_saved_test(saved, checkpoint_hash, expected)
    status = read(saved/"status.json")
    require(status.get("state") == "complete" and status.get("result_sha256") == sha(saved/"result.json"), "Saved original input archive incomplete")
    paths += [saved/"protocol.json", saved/"result.json"]
    for row in lookup.values():
        paths += [saved/row[key] for key in ("array_file", "record_file")]
    pins = {str(regular(path).resolve()): sha(path) for path in paths}
    del payload
    return {"checkpoint_sha256": checkpoint_hash, "expected": expected, "test_source": test["source"],
        "saved": saved_identity, "input_pins": pins}


def preflight():
    admission_started = now()
    forecast(JOBS, INITIAL_ALLOWANCE, initial=True); pretraining_only(); assert_idle()
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "Fresh output required; no retry")
    no_locks(NATIVE, SAVED, REPO/"research/results/full_waterdrop_100k")
    frozen = read(FREEZE); verify_freeze(frozen)
    ev = science(); native_identity, native_pins = prior_complete(ev)
    release = read(RELEASE); analysis_pins = verify_release(release, native_identity)
    models = {job: prepare_model_inputs(ev, *job) for job in JOBS}
    immutable = {str(ROOT/path): digest for path,digest in frozen["files_sha256"].items()}
    for pins in (native_pins, analysis_pins, {str(FREEZE):sha(FREEZE), str(RELEASE):sha(RELEASE)}, *[m["input_pins"] for m in models.values()]):
        merge_pins(immutable, pins)
    verify_pins(immutable); assert_idle(); pretraining_only(); verify_release(release, native_identity)
    return {"models":models,"immutable":immutable,"native_identity":native_identity,"release":release,
            "admission_started_utc":admission_started.isoformat(),"admission_seconds":(now()-admission_started).total_seconds(),
            "planning":forecast(JOBS,INITIAL_ALLOWANCE,initial=True)}


def command(objective, seed, executable=None):
    name = f"{objective}_seed{seed}"
    require((objective,seed) in JOBS and type(seed) is int, "Exact original six jobs required")
    return [str(executable or sys.executable), "-u", "-m", "research.native_random_envelope",
        "--checkpoint", str(REPO/"research/results/full_waterdrop_100k"/name/"checkpoint-100000.pt"),
        "--validation-manifest", str(DATA/"valid.json"), "--test-manifest", str(DATA/"test.json"),
        "--saved-same-state-dir", str(SAVED/name), "--output-dir", str(OUTPUT/name), "--device", "mps", "--launch"]


def verify_job(directory, objective, seed, prepared):
    ev = science(); directory = Path(directory); no_locks(directory)
    model = prepared["models"][objective,seed]
    protocol, result, status = [read(directory/file) for file in ("protocol.json","result.json","status.json")]
    digest = sha(directory/"protocol.json")
    for row in (protocol,result,status):
        require(row.get("objective") == objective and type(row.get("seed")) is int and row["seed"] == seed, "Postjob model identity differs")
    require(result.get("state") == status.get("state") == "complete" and status.get("result_sha256") == sha(directory/"result.json")
        and result.get("protocol_sha256") == digest, "Postjob publication incomplete/hash mismatch")
    require(protocol.get("scope") == result.get("scope") == ev.SCOPE and protocol.get("checkpoint_sha256") == result.get("checkpoint_sha256") == model["checkpoint_sha256"], "Postjob scope/checkpoint differs")
    require(protocol.get("cases") == list(CASES) and protocol.get("random_draw_ids") == list(range(8))
        and protocol.get("expected_frames") == model["expected"] and protocol.get("saved_test_inputs") == model["saved"]
        and protocol.get("source_tfrecord") == model["test_source"], "Postjob fixed scientific population differs")
    require(protocol.get("input_files_sha256") == model["input_pins"] and protocol.get("envelope_protocol_sha256") == sha(ev.PROTOCOL), "Postjob exact source/input inventory differs")
    runtime = protocol.get("runtime", {})
    require(type(protocol.get("threads")) is int and protocol["threads"] == 2 and runtime.get("device") == "mps"
        and runtime.get("radius_backend") == "scipy_host" and runtime.get("mps_fallback_environment") == "0", "Postjob execution contract differs")
    require(type(status.get("committed_frames")) is int and status["committed_frames"] == result.get("required_frames") == 425, "Postjob425 committed frames required")
    records = result.get("records", []); index = {row["record_file"]:row for row in records}
    require(len(records) == len(index) == 425, "Postjob unique425 records required")
    allowed = {"protocol.json","result.json","status.json"}; pins = {str(directory/file):sha(directory/file) for file in allowed}; failures=[]
    for item in model["expected"]:
        name = ev.bridge.stem(item)+".json"
        recovered = ev.bridge.recover(directory,item,digest)
        require(recovered is not None and same(recovered,index.get(name)), "Recovered/indexed optional frame differs")
        row = read(directory/name)
        require(row.get("objective") == objective and type(row.get("seed")) is int and row["seed"] == seed and row.get("checkpoint_sha256") == model["checkpoint_sha256"], "Raw optional model identity differs")
        require(all(type(row[key]) is int for key in ("source_index","target_frame")), "Raw frame identity integer required")
        require(set(row.get("cases",{})) == set(CASES), "Exact13 case slots required")
        for case_name,case in row["cases"].items():
            require(case.get("status") in ("complete","failed") and (case["status"] == "complete") == (case.get("failure") is None)
                and (case.get("metrics") is not None) == (case["status"] == "complete"), "Scientific failure/status/metrics inconsistent")
            if case["status"] == "failed":
                require(isinstance(case["failure"],dict) and case["failure"], "Empty scientific failure")
                failures.append({**item,"case":case_name,"failure":case["failure"]})
        require((row["status"] == "complete") == all(case["status"] == "complete" for case in row["cases"].values()), "Frame/case completion differs")
        require(not row.get("failure") or row["failure"].get("category") != "native_parity_failure", "Parity failure requires review")
        for file,hashkey in ((name,"record_sha256"),(recovered["array_file"],"array_sha256")):
            require(Path(file).name == file, "Unsafe optional artifact path")
            require(sha(regular(directory/file)) == recovered[hashkey], "Optional raw hash differs")
            allowed.add(file);pins[str(directory/file)]=recovered[hashkey]
    require(result.get("complete_frames") == sum(row["status"] == "complete" for row in records)
        and result.get("failed_frames") == sum(row["status"] == "failed" for row in records), "Optional failure counts differ")
    require({path.name for path in directory.iterdir()} == allowed, "Orphan/unexpected optional artifacts require review")
    verify_pins(model["input_pins"]);verify_pins(pins)
    return {"passed":True,"records":425,"case_slots":5525,"scientific_failures":failures,"failed_case_count":len(failures),"files_sha256":pins}


def capture(directory):
    return {str(path.relative_to(directory)): {"sha256":sha(path),"size_bytes":path.stat().st_size}
        for path in sorted(Path(directory).rglob("*")) if path.is_file() and not path.is_symlink()} if Path(directory).exists() else {}


def launch(prepared):
    initialization_started = now()
    ev = science(); atomic = ev.full.atomic_json
    forecast(JOBS,INITIAL_ALLOWANCE,initial=True);assert_idle();pretraining_only();verify_pins(prepared["immutable"])
    verify_release(prepared["release"],prepared["native_identity"])
    forecast(JOBS,INITIAL_ALLOWANCE,initial=True)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "Never overwrite/retry existing output")
    OUTPUT.mkdir(parents=True,exist_ok=False)
    status={"state":"starting","pid":os.getpid(),"started_utc":initialization_started.isoformat(),
        "initialization_started_utc":initialization_started.isoformat(),"admission_started_utc":prepared.get("admission_started_utc"),
        "admission_seconds":prepared.get("admission_seconds"),"execution_budget_started_utc":None,
        "budget_scope":"2700s execution plus remaining summary/audit starts immediately before first evaluator; admission and initial setup excluded and separately measured. Later supervisor overhead included. Absolute08:45 finish enforced.","jobs":[{
        "name":f"{o}_seed{s}","objective":o,"seed":s,"state":"unstarted","command":command(o,s)} for o,s in JOBS],
        "scope":SCOPE,"freeze_sha256":sha(FREEZE),"release_sha256":sha(RELEASE),"summary_audit_reserve_seconds":SUMMARY_RESERVE,
        "no_automatic_recovery":True,"scientific_summary_and_audit_pending":True,"inference_complete":False}
    owned=False;keepawake=None;child=None;started=None;allowance=INITIAL_ALLOWANCE
    try:
        with ev.full.RunLock(OUTPUT):
            owned=True
            atomic(OUTPUT/"input_identity.json",{"immutable_files_sha256":prepared["immutable"],"native_identity":prepared["native_identity"]})
            atomic(OUTPUT/"source_freeze.json",read(FREEZE));atomic(OUTPUT/"launch_release.json",prepared["release"])
            keepawake=subprocess.Popen(["caffeinate","-i","-w",str(os.getpid())])
            status.update(state="running",keepawake_pid=keepawake.pid);atomic(OUTPUT/"launch.json",status);atomic(OUTPUT/"queue_status.json",status)
            for index,(objective,seed) in enumerate(JOBS):
                try:
                    forecast(JOBS[index:],allowance,started=started,initial=started is None)
                    assert_idle();pretraining_only();no_locks(NATIVE,SAVED)
                    verify_pins(prepared["immutable"])
                    if started is None:
                        verify_release(prepared["release"],prepared["native_identity"])
                    status["planning"]=forecast(JOBS[index:],allowance,started=started,initial=started is None)
                except RuntimeError as error:
                    status.update(state="gate_stopped",stop_reason=str(error),updated_utc=now().isoformat(),job_allowance_seconds=allowance)
                    atomic(OUTPUT/"queue_status.json",status);return 2
                entry=status["jobs"][index];directory=OUTPUT/entry["name"]
                require(not directory.exists() and not directory.is_symlink(), "Existing child output requires review, no retry")
                entry.update(state="starting",started_utc=now().isoformat());atomic(OUTPUT/"queue_status.json",status)
                clock=time.perf_counter();log_path=OUTPUT/(entry["name"]+".log")
                with log_path.open("x") as log:
                    if started is None:
                        # The initial6*300+900 allowance exactly fills2700s.
                        # Start it at admission to the first evaluator, after
                        # measured setup, and pass the same timestamp so a real
                        # advancing clock cannot consume nonexistent headroom.
                        started=now()
                        try:
                            status["planning"]=forecast(JOBS,allowance,at=started,started=started,initial=True)
                        except RuntimeError as error:
                            status.update(state="gate_stopped",stop_reason=str(error),updated_utc=now().isoformat())
                            atomic(OUTPUT/"queue_status.json",status);return 2
                        status.update(execution_budget_started_utc=started.isoformat(),
                            initialization_seconds_before_first_evaluator=(started-initialization_started).total_seconds())
                    child=subprocess.Popen(entry["command"],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0"})
                    entry.update(state="running",pid=child.pid);atomic(OUTPUT/"queue_status.json",status)
                    code=child.wait()
                elapsed=time.perf_counter()-clock;allowance=max(allowance,math.ceil(elapsed))
                entry.update(returncode=code,elapsed_seconds=elapsed,completed_utc=now().isoformat(),log_sha256=sha(log_path),state="verifying")
                status["job_allowance_seconds"]=allowance;atomic(OUTPUT/"queue_status.json",status)
                try:
                    entry["verification"]=verify_job(directory,objective,seed,prepared)
                    require(code == 0,"Child exit nonzero; no retry")
                    verify_pins(prepared["immutable"])
                except Exception as error:
                    entry.update(state="error",error={"type":type(error).__name__,"message":str(error)},preserved_artifacts=capture(directory))
                    status.update(state="error");atomic(OUTPUT/"queue_status.json",status);return code or 1
                entry["state"]="complete";atomic(OUTPUT/"queue_status.json",status)
            status.update(inference_complete=True,inference_completed_utc=now().isoformat(),
                supervisor_seconds_since_initialization=(now()-initialization_started).total_seconds())
            final_checked=now()
            status["final_analysis_reserve_check"]={"checked_utc":final_checked.isoformat(),
                "elapsed_execution_budget_seconds":(final_checked-started).total_seconds(),
                "summary_audit_reserve_seconds":SUMMARY_RESERVE,
                "execution_plus_reserve_seconds":(final_checked-started).total_seconds()+SUMMARY_RESERVE,
                "total_execution_analysis_budget_seconds":TOTAL_BUDGET,
                "forecast_analysis_finish_utc":(final_checked+timedelta(seconds=SUMMARY_RESERVE)).isoformat(),
                "absolute_finish_target_utc":FINISH_TARGET.isoformat()}
            try:
                status["planning"]=forecast((),allowance,at=final_checked,started=started)
            except RuntimeError as error:
                status.update(state="gate_stopped",stop_reason=str(error),analysis_reserve_available=False,updated_utc=now().isoformat())
                atomic(OUTPUT/"queue_status.json",status);return 2
            status.update(state="complete",completed_utc=now().isoformat(),finished_after_forecast_target=now()>FINISH_TARGET,
                analysis_reserve_available=True)
            atomic(OUTPUT/"queue_status.json",status)
        return 0
    except BaseException as error:
        if owned:
            status.update(state="interrupted" if isinstance(error,KeyboardInterrupt) else "error",error={"type":type(error).__name__,"message":str(error)},
                live_child_pid=child.pid if child is not None and child.poll() is None else None)
            atomic(OUTPUT/"queue_status.json",status)
        raise
    finally:
        # Only the keep-awake child created and owned by this invocation may
        # be stopped; never signal an evaluator or another supervisor.
        if owned and keepawake is not None and (child is None or child.poll() is not None) and keepawake.poll() is None:
            keepawake.terminate()
            try:keepawake.wait(timeout=5)
            except subprocess.TimeoutExpired:pass


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument("--check",action="store_true");mode.add_argument("--launch",action="store_true")
    args=parser.parse_args(argv);prepared=preflight()
    if args.check:
        print(json.dumps({"ready":True,"outputtree_written":False,"planning":prepared["planning"],"commands":[command(*job) for job in JOBS]},indent=2));return 0
    return launch(prepared)


if __name__ == "__main__":
    raise SystemExit(main())
