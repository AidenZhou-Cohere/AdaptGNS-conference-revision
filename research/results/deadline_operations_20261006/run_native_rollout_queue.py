"""Sequential, no-retry local launcher for the reviewed native autonomous bridge."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/AdaptGNS"
OUTPUT = ROOT / "work/native-graph-rollout-20261005"
CUTOFF = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)
sys.path.insert(0, str(REPO))
from research.full_training import RunLock
from research.full_rollout import atomic_json, sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    freeze_path = Path(__file__).with_name("native_source_freeze.json")
    frozen = json.loads(freeze_path.read_text())
    for path, digest in frozen["source_sha256"].items():
        if sha256(REPO / path) != digest:
            raise RuntimeError("Reviewed source changed: " + path)
    if OUTPUT.exists():
        raise RuntimeError("Existing queue output requires manual reviewed recovery; no automatic retry")
    OUTPUT.mkdir()
    with RunLock(OUTPUT):
        launch = {"started_utc": now(), "pid": os.getpid(), "source_freeze_sha256": sha256(freeze_path),
                  "launcher_sha256": sha256(Path(__file__)), "frozen_commit": frozen["git_commit"], "jobs": []}
        atomic_json(OUTPUT / "launch.json", launch)
        for objective in ("faithful", "nll"):
            for seed in (0, 1, 2):
                if datetime.now(timezone.utc) >= CUTOFF:
                    atomic_json(OUTPUT / "queue_status.json", {"state": "cutoff", "updated_utc": now(), "jobs": launch["jobs"]})
                    return 2
                for path, digest in frozen["source_sha256"].items():
                    if sha256(REPO / path) != digest:
                        raise RuntimeError("Reviewed source changed before job: " + path)
                name = f"{objective}_seed{seed}"
                command = [sys.executable, "-u", "-m", "research.native_graph_rollout",
                           "--checkpoint", str(REPO / f"research/results/full_waterdrop_100k/{name}/checkpoint-100000.pt"),
                           "--test-manifest", str(ROOT / "work/full-data/converted/test.json"),
                           "--output-dir", str(OUTPUT / name), "--device", "mps", "--threads", "2"]
                entry = {"name": name, "command": command, "started_utc": now(), "state": "starting"}
                launch["jobs"].append(entry)
                atomic_json(OUTPUT / "launch.json", launch)
                started = time.perf_counter()
                with (OUTPUT / (name + ".log")).open("x") as log:
                    child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                             env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
                    entry.update(pid=child.pid, state="running")
                    atomic_json(OUTPUT / "queue_status.json", {"state": "running", "pid": os.getpid(), "updated_utc": now(), "jobs": launch["jobs"]})
                    code = child.wait()
                entry.update(returncode=code, elapsed_seconds=time.perf_counter()-started, completed_utc=now(),
                             state="complete" if code == 0 else "error", log_sha256=sha256(OUTPUT / (name + ".log")))
                atomic_json(OUTPUT / "launch.json", launch)
                atomic_json(OUTPUT / "queue_status.json", {"state": "running" if code == 0 else "error", "pid": os.getpid(), "updated_utc": now(), "jobs": launch["jobs"]})
                print(json.dumps(entry), flush=True)
                if code:
                    return code
        atomic_json(OUTPUT / "queue_status.json", {"state": "complete", "completed_utc": now(), "jobs": launch["jobs"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
