"""Explicit, no-retry supervisor for all six fixed faithful110k endpoints.

Operational code only; frozen evaluator/trainer/protocol files are never changed.
--check verifies provenance with an ephemeral manifest and never writes OUTPUT.
The remaining-time gate is a conservative forecast, not a runtime guarantee.
Root must review/freeze this helper and its inventory before --check or --launch.
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
import tempfile
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/AdaptGNS"
OUTPUT = ROOT / "work/continuation-evaluation-20261006"
TRAINING = ROOT / "work/faithful-graph-support-10k-20261006"
NATIVE = ROOT / "work/native-graph-rollout-20261005"
SAVED = ROOT / "work/full-evaluation/same_state"
DATA = ROOT / "work/full-data/converted"
FREEZE = Path(__file__).with_name("continuation_evaluation_queue_freeze.json")
LATEST_START = datetime(2026, 10, 6, 15, tzinfo=timezone.utc)
FINISH_TARGET = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
HARD_CUTOFF = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)
INITIAL_ALLOWANCES = {"observed": 300, "autonomous": 5700}
EVALUATOR_COMMIT = "c97c0a7"
COHORT = tuple((arm, seed) for seed in range(3) for arm in ("base", "mix"))
JOBS = tuple((mode, arm, seed) for mode in ("observed", "autonomous") for arm, seed in COHORT)
OBSERVED_POLICIES = ("base", "dense", "random25", "speed25", "previous-observed-base-risk25")
AUTONOMOUS_POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
MODULES = {"research." + name for name in ("full_training", "run_full_queue", "run_evaluation_queue", "full_rollout",
    "full_same_state", "native_graph_rollout", "faithful_graph_support", "continuation_evaluation", "graph_convention_bridge")}
SUPERVISORS = {f"run_{name}_queue.py" for name in ("graph_bridge", "native_rollout", "graph_support", "continuation_evaluation")}
PINNED = {
    "research/continuation_evaluation.py": "ec2ee1ba6e2a0cb78c55bd2720031af5d4fb8535d17c679aa6d319b74378a528",
    "research/protocols/continuation_evaluation_20261005.md": "ce0ae3028155775c39c56cf51860cda07a54bd912f24af60e5b44110f826e7b4",
    "research/tests/test_continuation_evaluation.py": "e3a06b2ab6602c211a0379805c3e004a31f7ca1d97b34c18b98e056df3b230aa",
    "research/faithful_graph_support.py": "71c2b3740aab635665d28b37addbd0b89360cccb3136d275e54b864d3b2a7eaa",
    "research/protocols/faithful_graph_support_10k_20261005.md": "90f896adefedf6e63503835db2479ac7e4f357043c01918b3334a846a8ffbba3",
    "research/native_graph_rollout.py": "b4bbca6660dc449c81958010e30fda8be2bc0aaf58f677e0946d26e600e6daf5",
    "research/protocols/native_graph_rollout_20261005.md": "7a28da92d9f7b133da8cea7a018400c424f553f9ff4d5f41c515142a078ff1ce",
    "research/graph_convention_bridge.py": "2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d",
    "research/protocols/graph_convention_bridge_20261005.md": "12994ff13d16841890398e9ed11f40dfed844ff6238073adbd3e2e63913c9e45",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key requires review: " + key)
            result[key] = value
        return result
    def invalid(token):
        raise RuntimeError("Nonfinite JSON token requires review: " + token)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def evaluator():
    sys.path.insert(0, str(REPO))
    from research import continuation_evaluation
    return continuation_evaluation


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(), "Missing or linked immutable file: " + str(path))
    return path


def verify_pins(pins):
    for path, expected in pins.items():
        require(sha(regular(path)) == expected, "Pinned bytes changed: " + str(path))


def required_frozen_files():
    paths = {Path(__file__).resolve(), Path(__file__).with_name("test_run_continuation_evaluation_queue.py").resolve()}
    paths.update(REPO / name for name in PINNED)
    paths.update(REPO / "research" / name for name in ("full_training.py", "full_rollout.py", "full_same_state.py", "budget_graph.py",
        "protocols/full_waterdrop_100k.md", "protocols/full_same_state_diagnostic.md"))
    paths.update((REPO / "adaptive-gns/gns").glob("*.py"))
    paths.update(Path(__file__).with_name(name).resolve() for name in ("run_graph_support_queue.py", "run_native_rollout_queue.py",
        "graph_support_queue_freeze.json", "native_source_freeze.json", "continuation_evaluation_independent_review_final.json"))
    paths.update(DATA / name for name in ("train.json", "valid.json", "test.json", "metadata.json"))
    for seed in range(3):
        paths.add(REPO / f"research/results/full_waterdrop_100k/faithful_seed{seed}/checkpoint-100000.pt")
        paths.update(SAVED / f"faithful_seed{seed}" / name for name in ("result.json", "protocol.json", "status.json"))
    return {str(path.relative_to(ROOT)) for path in paths}


def verify_freeze(frozen):
    require(frozen.get("schema") == 1 and frozen.get("scope") == "reviewed_continuation_evaluation_queue_v1"
            and frozen.get("evaluation_commit") == EVALUATOR_COMMIT, "Unexpected queue freeze identity")
    require(frozen.get("latest_launch_utc") == LATEST_START.isoformat() and frozen.get("finish_target_utc") == FINISH_TARGET.isoformat()
            and frozen.get("initial_job_allowances_seconds") == INITIAL_ALLOWANCES, "Operational forecast differs from reviewed freeze")
    require(set(frozen.get("files_sha256", {})) == required_frozen_files(), "Incomplete/extra source, parent or manifest freeze inventory")
    verify_pins({str(ROOT / path): value for path, value in frozen["files_sha256"].items()})
    require(all(frozen["files_sha256"][str((REPO / path).relative_to(ROOT))] == value for path, value in PINNED.items()),
            "Freeze attempts to change reviewed evaluator/trainer sources")


def no_locks(*roots):
    for root in roots:
        root = Path(root)
        if root.exists():
            for path in root.rglob("*"):
                require(not path.is_symlink(), "Linked evidence path requires review: " + str(path))
                require(not (path.name.endswith(".lock") or ".lock." in path.name or path.name.endswith(".tmp")),
                        "Existing lock/recovery/temp artifact requires manual review: " + str(path))


def heavy_processes(text, own_pid):
    found = []
    for line in text.splitlines():
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
            words = command.split()  # A truncated quote must not hide an exact module/script.
        if not words:
            continue
        program = Path(words[0]).name
        target = program if program in SUPERVISORS else None
        module = None
        if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", program, re.IGNORECASE):
            index = 1
            while index < len(words):
                word = words[index]
                if word == "-c":  # Shell/code text that mentions a module is not a running evaluator.
                    break
                if word == "-m":
                    module = words[index + 1] if index + 1 < len(words) else None
                    break
                if word.startswith("-m") and len(word) > 2:
                    module = word[2:]
                    break
                if word in ("-W", "-X"):
                    index += 2
                    continue
                if word == "--":
                    target = Path(words[index + 1]).name if index + 1 < len(words) else None
                    break
                if not word.startswith("-"):
                    target = Path(word).name
                    break
                index += 1
        if module in MODULES or target in SUPERVISORS or target in {name.split(".")[-1] + ".py" for name in MODULES}:
            found.append({"pid": pid, "command": command})
    return found


def assert_idle():
    text = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    require(not heavy_processes(text, os.getpid()), "Another heavy evaluator/trainer/supervisor is live")


def forecast_record(remaining, allowances, at=None):
    at = now() if at is None else at
    seconds = sum(allowances[mode] for mode, _, _ in remaining)
    return {"checked_utc": at.isoformat(), "remaining_jobs": len(remaining), "remaining_allowance_seconds": seconds,
        "forecast_finish_utc": (at + timedelta(seconds=seconds)).isoformat(), "finish_target_utc": FINISH_TARGET.isoformat(),
        "same_mode_job_allowances_seconds": dict(allowances), "scope": "Conservative operational forecast, not a guaranteed duration or automatic child interruption"}


def forecast(remaining, allowances, at=None, initial=False):
    at = now() if at is None else at
    require(at.tzinfo is not None, "UTC-aware planning clock required")
    require(at < HARD_CUTOFF, "Hard research cutoff reached")
    if initial:
        require(at < LATEST_START, "Initial latest launch time passed")
    record = forecast_record(remaining, allowances, at)
    require(at < FINISH_TARGET and datetime.fromisoformat(record["forecast_finish_utc"]) <= FINISH_TARGET,
            "Remaining-work forecast no longer fits review/writing reserve")
    return record

def completed_training_manifest():
    no_locks(TRAINING)
    queue = read(regular(TRAINING / "queue_status.json"))
    jobs = queue.get("jobs", [])
    require(queue.get("state") == "complete" and len(jobs) == 6
            and {(j.get("arm"), j.get("seed")) for j in jobs} == set(COHORT), "All six training jobs must be complete")
    require(all(j.get("name") == f"{j['arm']}_seed{j['seed']}" and j.get("state") == "complete" and j.get("returncode") == 0
                and type(j["seed"]) is int for j in jobs), "Training job completion/name/return code differs")
    entries, pins = [], {str(TRAINING / "queue_status.json"): sha(TRAINING / "queue_status.json")}
    for arm, seed in COHORT:
        directory = TRAINING / f"{arm}_seed{seed}"
        status, latest, config = (read(regular(directory / name)) for name in ("status.json", "latest.json", "protocol.json"))
        require(status.get("state") == "complete" and status.get("completed_additional_updates") == 10000
                and status.get("completed_total_updates") == 110000 and status.get("latest_checkpoint") == latest,
                "Training status not committed at fixed110k endpoint")
        require(latest.get("path") == "checkpoint-extra-10000.pt" and latest.get("completed_additional_updates") == 10000
                and latest.get("completed_total_updates") == 110000, "Latest pointer must select fixed10000-additional checkpoint")
        require(config.get("arm") == arm and config.get("seed") == seed and config.get("objective") == "faithful", "Training protocol identity differs")
        checkpoint = regular(directory / "checkpoint-extra-10000.pt")
        require(sha(checkpoint) == latest.get("sha256"), "Committed endpoint hash differs")
        entries.append({"arm": arm, "seed": seed, "checkpoint": str(checkpoint), "sha256": latest["sha256"]})
        for name in ("status.json", "latest.json", "protocol.json", "checkpoint-extra-10000.pt"):
            pins[str(directory / name)] = sha(directory / name)
    require(len({row["sha256"] for row in entries}) == 6, "Six distinct endpoint byte identities required")
    return {"schema": 1, "scope": "faithful_graph_support_110k_endpoints", "endpoints": entries}, pins


def verify_cohort_document(document, metadata_hash):
    """The frozen loader alone admits endpoint schema, Adam lineage and paired schedules."""
    ev = evaluator()
    with tempfile.TemporaryDirectory(prefix="continuation-cohort-check-") as temp:
        path = Path(temp) / "cohort_manifest.json"
        path.write_bytes(json_bytes(document))
        entries, configs, pins = ev.load_verified_cohort(path, metadata_hash)
        require(pins.pop(str(path.resolve())) == sha(path), "Temporary cohort manifest not verified")
    return entries, configs, pins


def manifest_array_pins(manifest, directory):
    pins = {}
    for record in manifest["records"]:
        for key in ("positions", "particle_types"):
            relative = Path(record[key]["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "Data array escapes manifest directory")
            path = regular(directory / relative)
            require(sha(path) == record[key]["sha256"], "Data array byte hash differs")
            pins[str(path)] = record[key]["sha256"]
    return pins


def evaluation_source_paths():
    names = ["continuation_evaluation.py", "native_graph_rollout.py", "graph_convention_bridge.py", "full_same_state.py",
        "full_rollout.py", "faithful_graph_support.py", "full_training.py", "budget_graph.py",
        "protocols/continuation_evaluation_20261005.md", "protocols/native_graph_rollout_20261005.md",
        "protocols/graph_convention_bridge_20261005.md", "protocols/faithful_graph_support_10k_20261005.md",
        "protocols/full_waterdrop_100k.md"]
    return [REPO / "research" / name for name in names] + sorted((REPO / "adaptive-gns/gns").glob("*.py"))


def preflight():
    timing = forecast(JOBS, INITIAL_ALLOWANCES, initial=True)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "Existing outputtree needs separately reviewed recovery; never overwrite")
    assert_idle()
    no_locks(TRAINING, NATIVE, SAVED, REPO / "research/results/full_waterdrop_100k")
    frozen = read(regular(FREEZE)); verify_freeze(frozen)
    manifest, training_pins = completed_training_manifest()
    ev = evaluator()
    entries, configs, cohort_pins = verify_cohort_document(manifest, sha(DATA / "metadata.json"))
    expected, valid, test = ev.bridge.expected_frames(configs["base", 0], DATA / "valid.json", DATA / "test.json")
    require(len(expected) == 425 and Counter(row["split"] for row in expected) == {"valid": 128, "test": 297}, "Full observed schedule required")
    require(valid["metadata"] == test["metadata"] == read(DATA / "metadata.json"), "Metadata identity differs")
    require(ev.full.select_records(test, "locked_test") == list(range(3, 30)), "Official locked test population differs")
    base_pins = dict(cohort_pins)
    ev.merge_pins(base_pins, {str(path): sha(path) for path in evaluation_source_paths() + [DATA / name for name in ("valid.json", "test.json", "metadata.json")]})
    ev.merge_pins(base_pins, manifest_array_pins(valid, DATA)); ev.merge_pins(base_pins, manifest_array_pins(test, DATA))
    saved, saved_pins = {}, {}
    for seed in range(3):
        directory = SAVED / f"faithful_seed{seed}"
        lookup, identity = ev.bridge.verify_saved_test(directory, configs["base", seed]["parent_checkpoint_sha256"], expected)
        status = read(regular(directory / "status.json"))
        require(status.get("state") == "complete" and status.get("result_sha256") == sha(directory / "result.json"), "Parent same-state status incomplete")
        saved[seed] = identity
        paths = [directory / name for name in ("result.json", "protocol.json")]
        paths += [directory / row[key] for row in lookup.values() for key in ("array_file", "record_file")]
        saved_pins[seed] = {str(regular(path)): sha(path) for path in paths}
    immutable = {str(ROOT / key): value for key, value in frozen["files_sha256"].items()}
    ev.merge_pins(immutable, training_pins); ev.merge_pins(immutable, base_pins)
    for pins in saved_pins.values(): ev.merge_pins(immutable, pins)
    immutable[str(FREEZE)] = sha(FREEZE)
    verify_pins(immutable)
    assert_idle(); no_locks(TRAINING, NATIVE, SAVED)
    timing = forecast(JOBS, INITIAL_ALLOWANCES, initial=True)
    return {"frozen": frozen, "manifest": manifest, "entries": entries, "configs": configs,
        "expected": expected, "test": test, "base_pins": base_pins, "saved": saved, "saved_pins": saved_pins,
        "immutable": immutable, "timing": timing}


def command(mode, arm, seed, executable=None):
    result = [str(executable or sys.executable), "-u", "-m", "research.continuation_evaluation", "--mode", mode,
        "--cohort-manifest", str(OUTPUT / "cohort_manifest.json"), "--arm", arm, "--seed", str(seed),
        "--validation-manifest", str(DATA / "valid.json"), "--test-manifest", str(DATA / "test.json"),
        "--protocol", str(REPO / "research/protocols/continuation_evaluation_20261005.md"),
        "--output-dir", str(OUTPUT / mode / f"{arm}_seed{seed}"), "--device", "mps", "--threads", "2"]
    if mode == "observed": result += ["--saved-same-state-dir", str(SAVED / f"faithful_seed{seed}")]
    return result


def job_input_pins(prepared, mode, seed):
    result = {**prepared["base_pins"], str(OUTPUT / "cohort_manifest.json"): sha(OUTPUT / "cohort_manifest.json")}
    if mode == "observed": result.update(prepared["saved_pins"][seed])
    return result


def verify_job(directory, mode, arm, seed, prepared):
    """Read-only committed-record recovery; guard failures remain valid outcomes."""
    ev = evaluator(); directory = Path(directory)
    no_locks(directory)
    protocol, result, status = (read(regular(directory / name)) for name in ("protocol.json", "result.json", "status.json"))
    config = prepared["configs"][arm, seed]; endpoint = prepared["entries"][arm, seed]
    lineage = {"arm": arm, "seed": seed, "original_seed": seed, "checkpoint_sha256": endpoint["sha256"],
        "parent_checkpoint_sha256": config["parent_checkpoint_sha256"], "continuation_config_sha256": ev.support.original.config_hash(config),
        "completed_total_updates": 110000, "completed_additional_updates": 10000}
    require(all(protocol.get(key) == value for key, value in lineage.items()) and protocol.get("mode") == mode
            and protocol.get("objective") == "faithful", "Postjob endpoint/protocol lineage differs")
    require(protocol.get("cohort") == prepared["manifest"]["endpoints"], "Postjob complete cohort differs")
    require(protocol.get("source_indices") == list(range(3, 30)) and protocol.get("trajectory_ids") == [prepared["test"]["records"][i]["id"] for i in range(3, 30)]
            and protocol.get("source_tfrecord") == prepared["test"]["source"], "Postjob test population differs")
    policies = OBSERVED_POLICIES if mode == "observed" else AUTONOMOUS_POLICIES
    require(protocol.get("policies") == list(policies) and protocol.get("horizon") == (1 if mode == "observed" else 995), "Postjob policy/horizon differs")
    require(protocol.get("guards") == {"max_candidate_pairs": ev.bridge.MAX_PAIRS, "max_abs_coordinate": ev.bridge.MAX_ABS}, "Postjob guard contract differs")
    runtime = protocol.get("runtime", {})
    require(protocol.get("threads") == 2 and runtime.get("device") == "mps" and runtime.get("radius_backend") == "scipy_host"
            and runtime.get("mps_fallback_environment") == "0" and protocol.get("deadline_utc") == HARD_CUTOFF.isoformat(), "Postjob runtime/deadline differs")
    require(protocol.get("input_files_sha256") == job_input_pins(prepared, mode, seed), "Postjob exact input/source inventory differs")
    verify_pins(protocol["input_files_sha256"])
    digest = sha(directory / "protocol.json")
    require(status.get("state") == result.get("state") == "complete" and status.get("result_sha256") == sha(directory / "result.json")
            and result.get("protocol_sha256") == digest, "Postjob result/status/protocol not committed complete")
    require(result.get("objective") == "faithful" and result.get("seed") == seed and result.get("checkpoint_sha256") == endpoint["sha256"], "Postjob compact result identity differs")
    allowed = {"protocol.json", "status.json", "result.json"}; failures = []
    records = result.get("records", [])
    if mode == "observed":
        expected = prepared["expected"]
        require(protocol.get("expected_frames") == expected and protocol.get("saved_parent_test_input_archive") == prepared["saved"][seed], "Postjob observed schedule/archive differs")
        require(result.get("arm") == arm and result.get("parent_checkpoint_sha256") == config["parent_checkpoint_sha256"]
                and result.get("required_frames") == 425 and status.get("committed_frames") == 425
                and status.get("arm") == arm and status.get("seed") == seed, "Postjob observed coverage/lineage differs")
        index = {row["record_file"]: row for row in records}
        require(len(records) == len(index) == 425, "Every observed frame must be indexed exactly once")
        for item in expected:
            name = ev.bridge.stem(item) + ".json"
            recovered = ev.bridge.recover(directory, item, digest)
            require(recovered is not None and recovered == index.get(name), "Observed recovered/indexed frame differs")
            row = read(directory / name)
            require(all(row.get(key) == lineage[key] for key in ("arm", "original_seed", "checkpoint_sha256", "parent_checkpoint_sha256")), "Observed row lineage differs")
            require(row.get("random_seed_material") == ev.random_material(seed, item), "Observed paired random schedule differs")
            require(set(row.get("cases", {})) == set(policies), "Observed policy outcome family differs")
            for policy, case in row["cases"].items():
                require(case.get("status") in ("complete", "failed") and (case["status"] == "complete") == (case.get("failure") is None), "Observed policy failure/status inconsistent")
                require((case.get("metrics") is not None) == (case["status"] == "complete"), "Observed failed policy metrics must remain undefined")
                if case["status"] == "failed": failures.append({**item, "policy": policy, "failure": case["failure"]})
            require((row["status"] == "complete") == all(c["status"] == "complete" for c in row["cases"].values()), "Observed frame/case status differs")
            require(not row.get("failure") or row["failure"].get("category") != "native_parity_failure", "Native parity failure must halt for review")
            allowed.update((name, recovered["array_file"]))
        require(result.get("complete_frames") == sum(r["status"] == "complete" for r in records)
                and result.get("failed_frames") == sum(r["status"] == "failed" for r in records), "Observed failure counts differ")
    else:
        require(protocol.get("expected_frames") is None and protocol.get("saved_parent_test_input_archive") is None, "Autonomous job contains observed attribution")
        require(read(regular(directory / "lineage_identity.json")) == lineage, "Autonomous immutable lineage identity differs")
        allowed.add("lineage_identity.json")
        index = {(row["source_index"], row["policy"]): row for row in records}
        require(len(records) == len(index) == 135 and set(index) == {(i, p) for i in range(3, 30) for p in policies}, "Every autonomous source/policy required exactly once")
        for (source, policy), row in index.items():
            recovered = ev.full.recover_record(directory, source, policy, prepared["test"]["records"][source]["id"], digest, 995, row)
            require(recovered == row and row.get("rng_seed") == 93000 + 1000 * seed + source, "Autonomous recovered/indexed/random identity differs")
            require(not row.get("failure") or row["failure"].get("category") != "native_parity_failure", "Native parity failure must halt for review")
            allowed.update((row["record_file"], row["trace_file"]))
            if row["status"] == "failed": failures.append({"source_index": source, "policy": policy, "failure": row["failure"], "completed_steps": row["completed_steps"]})
        require(result.get("summary") == ev.full.summarize(records, protocol["trajectory_ids"]), "Autonomous failure-aware summary differs")
    require({p.name for p in directory.iterdir() if p.is_file()} == allowed and not any(p.is_dir() for p in directory.iterdir()), "Unexpected/orphan job artifacts require review")
    return {"passed": True, "records": len(records), "scientific_failed_policy_outcomes": len(failures), "failures": failures,
        "protocol_sha256": digest, "result_sha256": sha(directory / "result.json"), "status_sha256": sha(directory / "status.json")}


def capture_job(directory):
    """Record partial unsuccessful bytes without mutating or recovering them."""
    directory = Path(directory)
    return {str(p.relative_to(directory)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
        for p in sorted(directory.rglob("*")) if p.is_file() and not p.is_symlink()} if directory.exists() else {}


def launch(prepared):
    ev = evaluator(); atomic = ev.full.atomic_json
    forecast(JOBS, INITIAL_ALLOWANCES, initial=True); assert_idle(); verify_pins(prepared["immutable"])
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "Never overwrite/retry an existing outputtree")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    status = {"state": "starting", "pid": os.getpid(), "started_utc": now().isoformat(),
        "launcher_sha256": sha(Path(__file__)), "freeze_sha256": sha(FREEZE), "jobs": [
            {"mode": mode, "arm": arm, "seed": seed, "name": f"{mode}_{arm}_seed{seed}", "state": "unstarted",
             "command": command(mode, arm, seed)} for mode, arm, seed in JOBS],
        "planning": prepared["timing"], "duration_assumptions": "Initial observed300s/autonomous5700s per job; same-mode allowances only increase to observed elapsed maxima. Forecasts are not guarantees. No automatic child interruption.",
        "remaining_work_is_incomplete": True}
    owned = False
    try:
        with ev.full.RunLock(OUTPUT):
            owned = True
            (OUTPUT / "cohort_manifest.json").write_bytes(json_bytes(prepared["manifest"]))
            (OUTPUT / "source_freeze.json").write_bytes(FREEZE.read_bytes())
            atomic(OUTPUT / "input_identity.json", {"immutable_files_sha256": prepared["immutable"], "cohort_manifest_sha256": sha(OUTPUT / "cohort_manifest.json"),
                "training_queue_sha256": sha(TRAINING / "queue_status.json"), "expected_observed_frames": prepared["expected"]})
            keepawake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            status.update(state="running", keepawake_pid=keepawake.pid)
            atomic(OUTPUT / "launch.json", status)
            atomic(OUTPUT / "queue_status.json", status)
            allowances = dict(INITIAL_ALLOWANCES)
            for index, (mode, arm, seed) in enumerate(JOBS):
                try:
                    status["planning"] = forecast(JOBS[index:], allowances)
                except RuntimeError as error:
                    status.update(state="forecast_stopped", stop_reason=str(error), updated_utc=now().isoformat(),
                        planning=forecast_record(JOBS[index:], allowances))
                    atomic(OUTPUT / "queue_status.json", status)
                    return 2
                assert_idle(); no_locks(TRAINING, NATIVE, SAVED)
                verify_pins(prepared["immutable"])
                try:
                    status["planning"] = forecast(JOBS[index:], allowances)  # Recheck after potentially slow immutable hashing.
                except RuntimeError as error:
                    status.update(state="forecast_stopped", stop_reason=str(error), updated_utc=now().isoformat(),
                        planning=forecast_record(JOBS[index:], allowances))
                    atomic(OUTPUT / "queue_status.json", status)
                    return 2
                directory = OUTPUT / mode / f"{arm}_seed{seed}"
                require(not directory.exists() and not directory.is_symlink(), "Child output already exists; never retry or overwrite")
                entry = status["jobs"][index]
                entry.update(state="starting", started_utc=now().isoformat())
                atomic(OUTPUT / "queue_status.json", status)
                started = time.perf_counter()
                log_path = OUTPUT / (entry["name"] + ".log")
                with log_path.open("x") as log:
                    child = subprocess.Popen(entry["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
                    entry.update(state="running", pid=child.pid)
                    atomic(OUTPUT / "queue_status.json", status)
                    code = child.wait()
                elapsed = time.perf_counter() - started
                entry.update(returncode=code, elapsed_seconds=elapsed, completed_utc=now().isoformat(), log_sha256=sha(log_path), state="verifying")
                allowances[mode] = max(allowances[mode], math.ceil(elapsed))
                status["same_mode_job_allowances_seconds"] = dict(allowances)
                atomic(OUTPUT / "queue_status.json", status)
                try:
                    entry["verification"] = verify_job(directory, mode, arm, seed, prepared)
                    require(code == 0, "Child process failed despite completed-looking artifacts")
                    verify_pins(prepared["immutable"])
                except Exception as error:
                    entry.update(state="error", verification_error={"type": type(error).__name__, "message": str(error)}, preserved_artifacts=capture_job(directory))
                    status.update(state="error", updated_utc=now().isoformat())
                    atomic(OUTPUT / "queue_status.json", status)
                    return code or 1
                entry.update(state="complete")
                atomic(OUTPUT / "queue_status.json", status)
            status.update(state="complete", completed_utc=now().isoformat(), remaining_work_is_incomplete=False,
                finished_after_planning_target=now() > FINISH_TARGET)
            atomic(OUTPUT / "queue_status.json", status)
        return 0
    except BaseException as error:
        if not owned:
            raise  # A competing lock owner is never modified by a refused acquisition.
        # A still-running child is never killed/retried here. Its PID/log and all bytes remain for root review.
        failure = {"type": type(error).__name__, "message": str(error), "utc": now().isoformat()}
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "error", failure=failure)
        with (OUTPUT / "queue_error.json").open("x") as stream:
            json.dump(failure, stream, indent=2); stream.write("\n")
        atomic(OUTPUT / "queue_status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    prepared = preflight()
    if args.check:
        print(json.dumps({"ready": True, "checked_utc": now().isoformat(), "cohort_manifest_sha256": hashlib.sha256(json_bytes(prepared["manifest"])).hexdigest(),
            "planned_jobs": [command(*job) for job in JOBS], "planning": prepared["timing"], "outputtree_written": False}, indent=2))
        return 0
    return launch(prepared)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"state": "refused_or_error", "type": type(error).__name__, "reason": str(error)}), file=sys.stderr)
        raise
