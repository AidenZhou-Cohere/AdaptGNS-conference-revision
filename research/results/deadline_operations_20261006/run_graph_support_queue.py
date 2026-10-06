"""Reviewed local queue helper; explicit --launch and completed native queue required.

This helper never resumes/retries an existing output tree. The scientific trainer
and protocol stay frozen; operational launch estimates are separate from them.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/AdaptGNS"
OUTPUT = ROOT / "work/faithful-graph-support-10k-20261006"
PRIOR = ROOT / "work/native-graph-rollout-20261005"
FREEZE = Path(__file__).with_name("graph_support_queue_freeze.json")
CUTOFF = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)
LATEST_START = datetime(2026, 10, 6, 9, tzinfo=timezone.utc)
sys.path.insert(0, str(REPO))
from research.full_training import RunLock
from research.full_rollout import atomic_json, recover_record, sha256

TRAINING_COMMIT = "89b01f9ee6127cc928bd659dbc080ef2ab916227"
OWNED_OUTPUT = False
NAMES = tuple(f"{obj}_seed{seed}" for obj in ("faithful", "nll") for seed in range(3))
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
HEAVY = re.compile(r"(?:^|\s)-m\s+research\.(?:full_training|run_full_queue|run_evaluation_queue|full_rollout|full_same_state|native_graph_rollout|faithful_graph_support|continuation_evaluation|graph_convention_bridge)(?:\s|$)|(?:^|[/\s])run_(?:graph_bridge|native_rollout|graph_support)_queue\.py(?:\s|$)")


def required_frozen_files():
    paths = [Path(__file__).resolve()]
    paths += [REPO / "research" / name for name in (
        "faithful_graph_support.py", "graph_convention_bridge.py", "full_training.py",
        "full_rollout.py", "full_same_state.py", "budget_graph.py",
        "protocols/faithful_graph_support_10k_20261005.md", "protocols/full_waterdrop_100k.md")]
    paths += sorted((REPO / "adaptive-gns/gns").glob("*.py"))
    paths += [REPO / f"research/results/full_waterdrop_100k/faithful_seed{s}/checkpoint-100000.pt" for s in range(3)]
    paths += [ROOT / "work/full-data/converted" / name for name in ("train.json", "valid.json", "metadata.json")]
    return {str(p.relative_to(ROOT)) for p in paths}


def verify_freeze(frozen):
    require(frozen.get("training_commit") == TRAINING_COMMIT, "Unexpected frozen training commit")
    require(set(frozen.get("files_sha256", {})) == required_frozen_files(), "Incomplete/unexpected source/parent/manifest freeze inventory")
    for path, digest in frozen["files_sha256"].items():
        require(sha256(ROOT / path) == digest, "Frozen source/parent/manifest changed: " + path)


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def prior_complete(directory):
    require(not (directory / "run.lock").exists(), "Prior native queue still owns a lock")
    status = json.loads((directory / "queue_status.json").read_text())
    require(status.get("state") == "complete", "Native queue must finish before continuation training")
    jobs = status.get("jobs", [])
    require(len(jobs) == 6 and {j["name"] for j in jobs} == set(NAMES), "Incomplete native prerequisite cohort")
    for job in jobs:
        require(job.get("state") == "complete" and job.get("returncode") == 0, "Native job not complete")
        base = directory / job["name"]
        require(not (base / "run.lock").exists(), "Native child lock still present")
        child = json.loads((base / "status.json").read_text())
        require(child.get("state") == "complete" and child["result_sha256"] == sha256(base / "result.json"),
                "Native completed result checksum mismatch")
        result = json.loads((base / "result.json").read_text())
        protocol = json.loads((base / "protocol.json").read_text())
        objective, seed_text = job["name"].split("_seed")
        checkpoint = REPO / f"research/results/full_waterdrop_100k/{job['name']}/checkpoint-100000.pt"
        require(result.get("state") == "complete" and result.get("objective") == objective and result.get("seed") == int(seed_text), "Native result model identity differs")
        require(protocol.get("objective") == objective and protocol.get("seed") == int(seed_text), "Native protocol model identity differs")
        require(result.get("protocol_sha256") == sha256(base / "protocol.json"), "Native protocol hash differs")
        require(result.get("checkpoint_sha256") == protocol.get("checkpoint_sha256") == sha256(checkpoint), "Native checkpoint identity differs")
        require(protocol.get("source_indices") == list(range(3,30)) and len(protocol.get("trajectory_ids", [])) == 27 and len(set(protocol["trajectory_ids"])) == 27, "Native protocol trajectory population differs")
        records = result.get("records", [])
        require(len(records) == 135 and {(r["source_index"], r["policy"]) for r in records} == {(i,p) for i in range(3,30) for p in POLICIES}, "Native trajectory/policy coverage incomplete")
        for row in records:
            for file_key, hash_key in (("record_file","record_sha256"),("trace_file","trace_sha256")):
                path = base / row[file_key]
                require(path.parent == base and path.is_file() and sha256(path) == row[hash_key], "Native row/trace payload hash differs")
            recovered = recover_record(base,row["source_index"],row["policy"],
                protocol["trajectory_ids"][row["source_index"]-3],result["protocol_sha256"],995,row)
            require(recovered == row, "Native recovered/indexed row identity differs")
    return status


def heavy_processes(process_text, own_pid):
    rows = []
    for line in process_text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != own_pid and HEAVY.search(parts[1]):
            rows.append(line.strip())
    return rows


def command(seed, arm, executable=sys.executable):
    data = ROOT / "work/full-data/converted"
    return [executable, "-u", "-m", "research.faithful_graph_support",
            "--parent-checkpoint", str(REPO / f"research/results/full_waterdrop_100k/faithful_seed{seed}/checkpoint-100000.pt"),
            "--train-manifest", str(data / "train.json"), "--valid-manifest", str(data / "valid.json"),
            "--metadata", str(data / "metadata.json"), "--seed", str(seed), "--arm", arm,
            "--output-dir", str(OUTPUT / f"{arm}_seed{seed}"), "--device", "mps", "--threads", "2"]


def preflight():
    require(datetime.now(timezone.utc) < LATEST_START, "Conservative latest launch time passed; reassess feasibility before changing this operational gate")
    require(not OUTPUT.exists(), "Existing continuation tree requires manual reviewed recovery; no automatic retry")
    prior_complete(PRIOR)
    frozen = json.loads(FREEZE.read_text())
    verify_freeze(frozen)
    processes = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    require(not heavy_processes(processes, os.getpid()), "Conflicting heavy research process exists")
    return frozen


def main():
    global OWNED_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    frozen = preflight()
    if args.check:
        print(json.dumps({"ready": True, "checked_utc": now(), "planned_jobs": [command(s,a) for s in range(3) for a in ("base","mix")]}))
        return 0
    OUTPUT.mkdir()
    with RunLock(OUTPUT):
        OWNED_OUTPUT = True
        keepawake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        launch = {"state":"running", "pid":os.getpid(), "keepawake_pid":keepawake.pid,
                  "started_utc":now(), "launcher_sha256":sha256(Path(__file__)),
                  "freeze_sha256":sha256(FREEZE), "frozen_training_commit":frozen["training_commit"], "jobs":[]}
        atomic_json(OUTPUT / "queue_status.json", launch)
        for seed in range(3):
            for arm in ("base", "mix"):
                require(datetime.now(timezone.utc) < CUTOFF, "Work cutoff reached; no next training job")
                for path, digest in frozen["files_sha256"].items():
                    require(sha256(ROOT / path) == digest, "Frozen bytes changed before next job: " + path)
                entry = {"seed":seed,"arm":arm,"name":f"{arm}_seed{seed}","command":command(seed,arm),
                         "started_utc":now(),"state":"starting"}
                launch["jobs"].append(entry)
                atomic_json(OUTPUT / "queue_status.json", launch)
                started = time.perf_counter()
                with (OUTPUT / (entry["name"] + ".log")).open("x") as log:
                    child = subprocess.Popen(entry["command"],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                        env={**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0"})
                    entry.update(pid=child.pid,state="running")
                    atomic_json(OUTPUT / "queue_status.json", launch)
                    code = child.wait()
                status_path = OUTPUT / entry["name"] / "status.json"
                child_status = json.loads(status_path.read_text()) if status_path.exists() else {}
                complete = code == 0 and child_status.get("state") == "complete" and child_status.get("completed_additional_updates") == 10000
                entry.update(returncode=code,state="complete" if complete else "error",completed_utc=now(),
                             wall_seconds=time.perf_counter()-started,log_sha256=sha256(OUTPUT/(entry["name"]+".log")))
                if not complete:
                    launch.update(state="error",updated_utc=now())
                    atomic_json(OUTPUT / "queue_status.json", launch)
                    return code or 1
                atomic_json(OUTPUT / "queue_status.json", launch)
        launch.update(state="complete",completed_utc=now())
        atomic_json(OUTPUT / "queue_status.json", launch)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Any started queue remains in place for manual identity/failure review.
        if OWNED_OUTPUT:
            failure = {"state":"error","pid":os.getpid(),"updated_utc":now(),
                       "type":type(error).__name__,"error":str(error)}
            with (OUTPUT / "queue_error.json").open("x") as stream:
                json.dump(failure,stream,indent=2)
            status_path = OUTPUT / "queue_status.json"
            prior_status = json.loads(status_path.read_text()) if status_path.exists() else {}
            atomic_json(status_path,{**prior_status,**failure})
        print(json.dumps({"state":"refused_or_error","utc":now(),"type":type(error).__name__,"reason":str(error)}),file=sys.stderr)
        raise
