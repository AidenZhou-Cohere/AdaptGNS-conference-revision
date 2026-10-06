"""Verify the compact exposure package with the Python standard library only."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

MODELS = tuple(f"{objective}_seed{seed}" for objective in ("faithful", "nll") for seed in (0, 1, 2))
TARGETS = (7, 106, 205, 304, 403, 502, 601, 700, 799, 898, 1000)
EXPECTED = {(source, target) for source in range(3, 30) for target in TARGETS}
PAIR_MAP = {"risk_minus_random": ["previous-observed-base-risk25", "random25"],
            "risk_minus_speed": ["previous-observed-base-risk25", "speed25"],
            "speed_minus_random": ["speed25", "random25"]}
PAIR_NAMES = set(PAIR_MAP)
PRIMARY = ("neither", "left_only", "right_only", "both")
BOTH = ("both_less", "both_equal", "both_more")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_name(name):
    require(isinstance(name, str) and bool(name), "Path must be a nonempty string")
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name
            and all(part not in ("", ".", "..") for part in path.parts), "Unsafe archive/package path")
    return name


def compact_projection(result):
    compact = {key: value for key, value in result.items() if key != "runs"}
    compact["publication"] = {"exact_result": "results.json.gz", "frame_records": "frame_records.tar.gz",
        "projection": "Original top-level values and every model aggregate except per-trajectory records; exact result retains all trajectories and indices. Every original per-frame JSON is retained in the archive."}
    compact["runs"] = {}
    for name, run in result["runs"].items():
        compact["runs"][name] = {key: value for key, value in run.items() if key not in ("aggregate", "frames")}
        compact["runs"][name]["aggregate"] = {key: value for key, value in run["aggregate"].items() if key != "trajectories"}
        compact["runs"][name]["original_frame_count"] = len(run["frames"])
        compact["runs"][name]["exact_result_run_key"] = name
    return compact


def validate_result(result):
    require(result["state"] == "complete" and result["required_models"] == 6
            and result["required_frames_per_model"] == 297 and result["required_total_frames"] == 1782,
            "Only the completed fixed exposure population is publishable")
    require(set(result["runs"]) == set(MODELS) and result["comparison_order"] == PAIR_MAP,
            "Model/comparison population differs")
    require(tuple(result["primary_groups"]) == PRIMARY and tuple(result["supplementary_both_subgroups"]) == BOTH,
            "Group definitions differ")
    require(result["scope"] == "post_inspection_original_100k_optional_exposure_decomposition",
            "Wrong experiment family")
    require(not result["consistency_errors"] and not result["failures"], "Unfinished or invalid analysis")
    require(set(result["objectives"]) == {"faithful", "nll"}, "Objective population differs")
    records = {}
    for name, run in result["runs"].items():
        require(run["eligible"] and not run["issues"], "Ineligible original model")
        index = {(row["source_index"], row["target_frame"]): row for row in run["frames"]}
        require(len(run["frames"]) == len(index) == 297 and set(index) == EXPECTED, "Wrong frame coverage")
        require(set(run["aggregate"]["trajectories"]) == {str(source) for source in range(3, 30)},
                "Missing trajectory aggregates")
        for (source, target), row in index.items():
            relative = f"frames/{name}/trajectory_{source:06d}_target_{target:04d}.json"
            require(row["record_file"] == relative and row["status"] == "complete", "Wrong frame identity/state")
            records[relative] = row
    return records


def iter_archived_records(path):
    seen = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = safe_name(member.name)
            require(member.isfile() and name not in seen and member.uid == member.gid == member.mtime == 0,
                    "Unexpected, duplicate, or noncanonical archive member")
            seen.add(name)
            stream = archive.extractfile(member)
            require(stream is not None, "Missing archive member contents")
            data = stream.read()
            require(len(data) == member.size, "Truncated archived record")
            yield name, data


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text())
    audit = json.loads((root / "PUBLICATION_AUDIT.json").read_text())
    require(audit["passed"] and audit["manifest_sha256"] == sha(root / "PUBLICATION_MANIFEST.json"), "Publication manifest differs")
    require(not root.is_symlink() and not any(path.is_symlink() for path in root.rglob("*")), "Package symlinks are not supported")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == set(manifest["files"]) | {"PUBLICATION_MANIFEST.json", "PUBLICATION_AUDIT.json"},
            "Unexpected or missing package files")
    for relative, identity in manifest["files"].items():
        safe_name(relative)
        path = root / relative
        require(not path.is_symlink() and path.stat().st_size == identity["size_bytes"] and sha(path) == identity["sha256"],
                "Packaged bytes differ: " + relative)
        if identity.get("encoding") == "original_bytes":
            require(identity["source_sha256"] == identity["sha256"], "Copied original identity differs")
    full_bytes = gzip.decompress((root / "results.json.gz").read_bytes())
    identity = manifest["files"]["results.json.gz"]
    require(digest(full_bytes) == identity["source_sha256"] and len(full_bytes) == identity["source_size_bytes"],
            "Compressed exact result differs")
    result = json.loads(full_bytes)
    expected_records = validate_result(result)
    require(json.loads((root / "results.json").read_text()) == compact_projection(result), "Compact projection differs")
    status = json.loads((root / "status.json").read_text())
    require(status["state"] == "complete" and status["results_sha256"] == digest(full_bytes)
            and status["report_sha256"] == sha(root / "report.md"), "Original completion status differs")
    require(result["all_inputs_sha256"] == sha(root / "all_inputs_sha256.json"), "Original input inventory differs")
    provenance = json.loads((root / "provenance.json").read_text())
    require(provenance["result_sha256"] == digest(full_bytes), "Publication provenance differs")
    independent = json.loads((root / "independent_audit.json").read_text())
    require(independent.get("passed") is True and independent["audited_frames"] == 1782, "Independent audit not passed/complete")
    audit_field = provenance["independent_audit_result_hash_field"]
    require(independent[audit_field] == digest(full_bytes), "Independent audit applies to different results")
    require(sha(root / "independent_audit.py") == provenance["independent_audit_source_sha256"] == independent["source_sha256"], "Audit source differs")
    require(sha(root / "independent_audit.json") == provenance["independent_audit_sha256"], "Audit record differs")
    require(sha(root / "analysis_source.py") == provenance["analysis_source_sha256"] == independent["analysis_source_sha256"], "Analysis source differs")
    require(sha(root / "protocol.md") == provenance["analysis_protocol_sha256"] == independent["analysis_protocol_sha256"] == result["protocol_sha256"], "Analysis protocol differs")
    inventory = json.loads((root / "raw_inventory.json").read_text())
    rows = {row["path"]: row for row in inventory["files"]}
    require(len(rows) == inventory["file_count"] and sum(row["size_bytes"] for row in rows.values()) == inventory["size_bytes"],
            "Raw inventory duplicates or total mismatch")
    require(rows["results.json"]["sha256"] == digest(full_bytes) and rows["results.json"]["size_bytes"] == len(full_bytes), "Raw full-result identity differs")
    frame_index = json.loads((root / "frame_archive_index.json").read_text())
    require(set(frame_index["files"]) == set(expected_records), "Archived frame index coverage differs")
    count = nulls = 0
    derived_names = set()
    for name, data in iter_archived_records(root / "frame_records.tar.gz"):
        require(name in expected_records and name in rows, "Unexpected archived frame")
        require(digest(data) == expected_records[name]["record_sha256"] == rows[name]["sha256"]
                == frame_index["files"][name]["sha256"], "Original archived frame bytes differ")
        require(len(data) == rows[name]["size_bytes"] == frame_index["files"][name]["size_bytes"], "Archived frame size differs")
        row = json.loads(data)
        original_index = expected_records[name]
        require(row["status"] == "complete" and row["source_index"] == original_index["source_index"]
                and row["target_frame"] == original_index["target_frame"] and set(row["pairs"]) == PAIR_NAMES,
                "Archived frame identity/policy outcomes differ")
        require(set(row["undefined_reasons"]) == {key for key, value in row["metrics"].items() if value is None},
                "Archived null reasons do not match null metrics")
        for pair in row["pairs"].values():
            require(pair["status"] == "complete" and set(pair["groups"]) == set(PRIMARY + BOTH), "Archived group population differs")
        if "derived_array_file" in row:
            safe_name(row["derived_array_file"])
            derived_names.add(row["derived_array_file"])
            derived = rows[row["derived_array_file"]]
            require(derived["sha256"] == row["derived_array_sha256"], "Omitted derived-array identity differs")
        require(frame_index["files"][name]["null_metric_count"] == len(row["undefined_reasons"]), "Frame null count differs")
        require(rows[name]["publication"] == {"file": "frame_records.tar.gz", "member": name, "encoding": "tar_member_original_bytes"}, "Archive raw identity location differs")
        nulls += len(row["undefined_reasons"])
        count += 1
    require(count == len(expected_records) == 1782, "Incomplete archived frame population")
    require(len(derived_names) == 1782, "Derived-array identity population differs")
    for relative, row in rows.items():
        safe_name(relative)
        location = row["publication"]
        if location is None:
            require(bool(row["omitted_reason"]), "Omission not documented")
        if location and location["encoding"] == "original_bytes":
            require(manifest["files"][location["file"]]["source_sha256"] == row["sha256"], "Raw copied identity differs")
    require(count == frame_index["record_count"] and nulls == frame_index["null_metric_count"], "Frame/null counts differ")
    if (root / "rendered/render_manifest.json").exists():
        rendered = json.loads((root / "rendered/render_manifest.json").read_text())
        require(rendered["results_sha256"] == digest(full_bytes) and rendered["independent_audit_sha256"] == sha(root / "independent_audit.json"), "Rendered source identity differs")
        require(rendered["renderer_sha256"] == sha(root / "renderer_source.py"), "Renderer source identity differs")
        for name, expected in rendered["outputs"].items():
            safe_name(name)
            require(sha(root / "rendered" / name) == expected, "Rendered artifact differs")
    value = {"passed": True, "package_files": len(manifest["files"]), "raw_files": len(rows),
             "original_frame_records": count, "frame_null_metrics_preserved": nulls, "result_sha256": digest(full_bytes)}
    print(json.dumps(value, indent=2))
    return value


if __name__ == "__main__":
    verify(Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent)
