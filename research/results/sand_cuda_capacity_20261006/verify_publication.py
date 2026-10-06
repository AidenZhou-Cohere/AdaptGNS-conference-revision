#!/usr/bin/env python3
"""Verify this compact public capacity family; optional scalar/mock CPU checks.

No real model, numeric trajectory array, checkpoint, CUDA or network execution.
Optional checks run in a temporary copy and preserve every archived result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TESTS = ("test_measure_sand_cuda_capacity.py", "test_measure_sand_cuda_capacity_v2.py",
         "test_train_sand_cuda_deterministic.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(name):
    return json.loads((HERE / name).read_text())


def verify():
    manifest = read("manifest.json")
    files = manifest["files"]
    expected = {row["path"] for row in files}
    observed = {p.name for p in HERE.iterdir() if p.is_file() and p.name != "manifest.json"}
    require(len(expected) == len(files) and observed == expected, "Manifest membership differs")
    for row in files:
        path = HERE / row["path"]
        require(path.parent == HERE and not path.is_symlink(), "Unexpected manifest path")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "File bytes differ: " + row["path"])
    provenance = read("curation_provenance.json")
    for row in provenance["source_files"]:
        if row["transformation"] == "exact byte copy":
            require(sha(HERE / row["public_file"]) == row["source_sha256"], "Original copied bytes differ")
    for version in (1, 2):
        release = read(f"sand_cuda_capacity_release_20261006_v{version}.json")
        require(release["status"] == "admitted_for_timing" and release["scientific_training_admitted"] is False,
                "Root release scope differs")
        bound = {"trainer": "train_sand_cuda_deterministic.py", "train_manifest": "train.json",
                 "admission": "sand_train_admission.json", "structural_report": "sand_numeric_structural_report.json",
                 "protocol": f"sand_cuda_capacity_protocol_v{version}.md",
                 "mechanism_review": "sand_native_cuda_numerical_assessment.json",
                 "supervisor": "measure_sand_cuda_capacity.py" if version == 1 else "measure_sand_cuda_capacity_v2.py"}
        for field, name in bound.items():
            require(release["files_sha256"][field] == sha(HERE / name), "Root release byte binding differs: " + field)
        require(release["preflight_sha256"] == sha(HERE / f"capacity_native_preflight_20261006_v{version}.json"),
                "Root preflight hash differs")
    summary = read("sand_capacity_v2_summary.json")
    require(summary["status"] == "all_six_verified" and summary["scientific_training_admitted"] is False,
            "V2 scope/status differs")
    require(len(summary["jobs"]) == 6 and sum(len(j["pairing_rows"]) for j in summary["jobs"]) == 3072,
            "All six complete update sequences required")
    for field in ("complete_evaluation_seconds", "diagnostics_execution_seconds", "total_compute_analysis_seconds",
                  "forecast_finish_utc", "fits_before_seven_hour_writing_reserve"):
        require(summary["forecast"][field] is None, "Unmeasured full-study cost/fit was filled: " + field)
    failure = read("sand_capacity_v1_failure_review.json")
    require(failure["wave_B_never_started"] is True and failure["original_failed_status_preserved"] is True
            and failure["unreaped_owned_children"] == [], "V1 failure provenance differs")
    runtime = read("sand_capacity_v1_runtime_failure.public.json")
    wanted = {"faithful_seed0": ("interrupted", 500, 0), "faithful_seed1": ("interrupted", 482, 0),
              "nll_seed0": ("planned_stop_incomplete", 512, 512), "nll_seed1": ("interrupted", 477, 0)}
    require(set(runtime) == set(wanted), "V1 jobs differ")
    for job, values in wanted.items():
        status = runtime[job]["status"]
        require(tuple(status[key] for key in ("state", "completed_steps", "committed_steps")) == values,
                "V1 stopped/committed updates differ")
        require("host" not in status["process"] and "token" not in status["process"], "Private process metadata retained")
    wave = read("sand_capacity_v1_wave_A.json")
    require(wave["state"] == "failed" and len(wave["jobs"]) == 4 and len(wave["observations"]) == 101
            and wave["unreaped_owned_children"] == [], "V1 wave evidence differs")
    require(sorted(j["external"]["exit_code"] for j in wave["jobs"]) == [-2, -2, -2, 0], "V1 interruptions differ")
    audit = read("sand_capacity_v2_independent_scalar_audit.json")
    portability = read("sand_capacity_v2_local_recomputation_review.json")
    require(portability["preserved_audit_sha256"] == sha(HERE / "audit_sand_base_capacity_scalars_first_attempt.py")
            and portability["summary_sha256"] == sha(HERE / "sand_capacity_v2_summary.json")
            and audit["audit_source_sha256"] == sha(HERE / "audit_sand_base_capacity_scalars.py"),
            "Preserved failed/revised scalar audit sources differ")
    require(audit["status"] == "passed" and audit["scalar_checks"] == 20056
            and audit["paired_rows_checked"] == 1536 and audit["training_rows_checked"] == 3072,
            "Recorded independent audit counts differ")
    differences = audit["local_LR_recomputation_discrepancies"]
    require(len(differences) == 12 and {r["step"] for r in differences} == {16, 274}
            and all(r["ulps"] == 1.0 for r in differences)
            and len({(r["id"], r["step"]) for r in differences}) == 12,
            "Preserved local one-ULP evidence differs")
    omitted = read("omitted_raw_inventory.json")
    require(len(omitted["checkpoint_references"]) == 17 and omitted["raw_checkpoint_bytes_included"] is False
            and omitted["raw_trajectory_arrays_included"] is False, "Raw-proof inventory differs")
    return {"status": "passed", "manifest_file_count": len(files),
            "manifest_payload_bytes": sum(row["bytes"] for row in files),
            "exact_copied_files": sum(row["transformation"] == "exact byte copy" for row in provenance["source_files"]),
            "public_projections": len(provenance["public_projections"]), "capacity_training_rows": 3072,
            "paired_rows": 1536, "preserved_local_lr_differences": 12, "checkpoint_hash_references": 17,
            "scientific_training_admitted": False, "complete_deadline_fit": None}


def run_cpu_checks():
    archived = read("sand_capacity_v2_independent_scalar_audit.json")
    archived_hash = sha(HERE / "sand_capacity_v2_independent_scalar_audit.json")
    with tempfile.TemporaryDirectory(prefix="sand-capacity-public-") as directory:
        copied = Path(directory)
        for path in HERE.iterdir():
            if path.is_file():
                shutil.copyfile(path, copied / path.name)
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        tests = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
            cwd=copied, env=environment, text=True, capture_output=True, timeout=60)
        require(tests.returncode == 0, "Copied synthetic/runtime tests failed:\n" + tests.stdout + tests.stderr)
        first = subprocess.run([sys.executable, "-B", "audit_sand_base_capacity_scalars_first_attempt.py"],
            cwd=copied, env=environment, text=True, capture_output=True, timeout=60)
        require(first.returncode == 0 or "Learning-rate schedule" in first.stderr,
                "Preserved exact-recomputation audit failed for an unexpected reason")
        revised = subprocess.run([sys.executable, "-B", "audit_sand_base_capacity_scalars.py"],
            cwd=copied, env=environment, text=True, capture_output=True, timeout=60)
        require(revised.returncode == 0, "Revised scalar audit failed:\n" + revised.stdout + revised.stderr)
        observed = json.loads((copied / "sand_capacity_v2_independent_scalar_audit.json").read_text())
        require(observed["status"] == "passed" and observed["scalar_checks"] == 20056
                and observed["paired_rows_checked"] == 1536 and observed["training_rows_checked"] == 3072,
                "Reproduced scalar audit is incomplete")
        require(observed["source_sha256"] == archived["source_sha256"]
                and observed["audit_source_sha256"] == archived["audit_source_sha256"]
                and observed["complete_deadline_fit"] is None and observed["scientific_training_admitted"] is False,
                "Reproduced audit source/scope differs")
        return {"status": "passed", "tests_stdout": tests.stdout.replace(directory, "<TEMP_FAMILY>"),
                "tests_stderr": tests.stderr.replace(directory, "<TEMP_FAMILY>"),
                "strict_first_audit_returncode": first.returncode,
                "strict_first_audit_reason": "Learning-rate schedule recomputation" if first.returncode else "No exact-LR discrepancy on this platform",
                "revised_scalar_checks": observed["scalar_checks"], "paired_rows": observed["paired_rows_checked"],
                "local_lr_recomputation_discrepancies": len(observed["local_LR_recomputation_discrepancies"]),
                "reproduced_audit_matches_archived_json": observed == archived,
                "public_archived_audit_unchanged": sha(HERE / "sand_capacity_v2_independent_scalar_audit.json") == archived_hash,
                "real_model_data_checkpoint_cuda_or_network_execution": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-cpu-checks", action="store_true")
    args = parser.parse_args()
    result = verify()
    if args.run_cpu_checks:
        result["cpu_checks"] = run_cpu_checks()
        verify()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
