"""Verify compact graph-bridge publication using Python's standard library."""
import gzip
import hashlib
import json
from pathlib import Path
import sys


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text())
    audit = json.loads((root / "PUBLICATION_AUDIT.json").read_text())
    require(audit["passed"] and audit["manifest_sha256"] == sha(root / "PUBLICATION_MANIFEST.json"), "Manifest identity differs")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected = set(manifest["files"]) | {"PUBLICATION_MANIFEST.json", "PUBLICATION_AUDIT.json"}
    require(actual == expected, "Missing or unexpected package files")
    for path, identity in manifest["files"].items():
        p = root / path
        require(p.stat().st_size == identity["size_bytes"] and sha(p) == identity["sha256"], f"Packaged bytes differ: {path}")
        if identity.get("encoding") == "original bytes":
            require(identity["source_sha256"] == identity["sha256"], f"Copied-source identity differs: {path}")
    zipped = root / "graph_convention_bridge.json.gz"
    full_bytes = gzip.decompress(zipped.read_bytes())
    full_hash = hashlib.sha256(full_bytes).hexdigest()
    require(full_hash == manifest["files"][zipped.name]["source_sha256"], "Original decompressed summary differs")
    require(len(full_bytes) == manifest["files"][zipped.name]["source_size_bytes"], "Original decompressed size differs")
    full = json.loads(full_bytes)
    compact = json.loads((root / "results.json").read_text())
    require(compact["publication"]["original_summary_sha256"] == full_hash, "Compact projection points at different source")
    for k, v in full.items():
        if k != "runs":
            require(compact[k] == v, f"Aggregate projection differs: {k}")
    require(len(compact["runs"]) == len(full["runs"]) == 6, "Incomplete models")
    for index, (short, run) in enumerate(zip(compact["runs"], full["runs"])):
        for key, value in run.items():
            if key not in ("frames", "input_files_sha256", "source_protocol"):
                require(short[key] == value, f"Per-run projection differs: {index}/{key}")
        require(short["full_summary_run_index"] == index, "Run index differs")
    independent = json.loads((root / "graph_bridge_aggregate_audit.json").read_text())
    require(independent["passed"] and independent["summary_sha256"] == full_hash, "Independent audit identity differs")
    inventory = json.loads((root / "raw_inventory.json").read_text())
    files = {row["path"]: row for row in inventory["files"]}
    require(len(files) == inventory["file_count"] == 5129, "Raw inventory duplicate/missing paths")
    require(sum(row["size_bytes"] for row in files.values()) == inventory["size_bytes"], "Raw inventory byte count differs")
    for row in files.values():
        packaged = row["publication"]
        if packaged is not None:
            identity = manifest["files"][packaged["file"]]
            require(row["sha256"] == identity["source_sha256"], "Inventoried copied identity differs")
    coverage = json.loads((root / "coverage.json").read_text())
    totals = {"total_frames": 0, "failed_frames": 0, "total_case_outcomes": 0, "failed_case_outcomes": 0, "native_parity_passed_frames": 0}
    for run in full["runs"]:
        name = f"{run['objective']}_seed{run['seed']}"
        protocol = json.loads((root / "runs" / name / "protocol.json").read_text())
        result = json.loads((root / "runs" / name / "result.json").read_text())
        status = json.loads((root / "runs" / name / "status.json").read_text())
        require(status["state"] == result["state"] == "complete", f"Job incomplete: {name}")
        require(status["result_sha256"] == sha(root / "runs" / name / "result.json"), f"Result identity differs: {name}")
        require(result["protocol_sha256"] == sha(root / "runs" / name / "protocol.json") == run["protocol_sha256"], f"Protocol identity differs: {name}")
        require(len(result["records"]) == len(run["frames"]) == 425, f"Incomplete run: {name}")
        for record in result["records"]:
            for filekey, hashkey in (("record_file", "record_sha256"), ("array_file", "array_sha256")):
                require(files[f"{name}/{record[filekey]}"]["sha256"] == record[hashkey], "Raw result index differs")
        for split in ("valid", "test"):
            block = coverage["models"][name]["splits"][split]
            scientific = run["splits"][split]
            frames = [f for f in run["frames"] if f["split"] == split]
            require(len(frames) == block["observed_frames"] == block["required_frames"] == scientific["required_frames"], "Split count differs")
            require(sum(f["status"] != "complete" for f in frames) == block["failed_frames"] == scientific["failed_frames"], "Split failures differ")
            require(block["native_parity_passed_frames"] == scientific["graph_diagnostics"]["native_parity_passed_frames"], "Parity counts differ")
            require(set(block["cases"]) == set(protocol["cases"]) and len(block["cases"]) == 16, "Case population differs")
            totals["total_frames"] += block["observed_frames"]
            totals["failed_frames"] += block["failed_frames"]
            totals["native_parity_passed_frames"] += block["native_parity_passed_frames"]
            for case, counts in block["cases"].items():
                require(counts["observed"] == counts["complete"] + counts["failed"] == block["required_frames"] - counts["missing"], "Case accounting differs")
                require(counts["failed"] == scientific["graph_diagnostics"]["failed_cases"][case], "Case failures differ")
                totals["total_case_outcomes"] += counts["observed"]
                totals["failed_case_outcomes"] += counts["failed"]
    require(all(coverage[k] == value for k, value in totals.items()), "Full case totals differ")
    require(totals["total_frames"] == 2550 and totals["total_case_outcomes"] == 40800, "Wrong full population")
    print(json.dumps({"passed": True, "package_files_checked": len(manifest["files"]), "raw_inventory_entries": len(files),
        "exact_full_summary_sha256": full_hash, **totals}, indent=2))


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent)
