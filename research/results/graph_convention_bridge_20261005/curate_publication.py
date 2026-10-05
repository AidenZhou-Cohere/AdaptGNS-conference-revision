"""Publish compact bridge artifacts without inference or changing raw outputs."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "work/graph-convention-bridge-20261005"
REPO = ROOT / "outputs/AdaptGNS"
DEST = REPO / "research/results/graph_convention_bridge_20261005"
AUDIT = ROOT / "work/deadline_research_20261005/graph_bridge_aggregate_audit.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main():
    if DEST.exists():
        raise FileExistsError(f"Preserve existing publication: {DEST}")
    summary_file = RAW / "reports/graph_convention_bridge.json"
    summary = json.loads(summary_file.read_text())
    independent = json.loads(AUDIT.read_text())
    assert independent["passed"] and independent["summary_sha256"] == sha(summary_file)
    assert summary["audit"]["passed"]
    assert sha(REPO / "research/summarize_graph_convention_bridge.py") == summary["summary_source_sha256"]
    source_files = sorted(p for p in RAW.rglob("*") if p.is_file())
    assert not any(p.is_symlink() for p in source_files)
    original = {p.relative_to(RAW).as_posix(): {"size_bytes": p.stat().st_size, "sha256": sha(p)} for p in source_files}
    assert len(original) == 5129
    DEST.mkdir(parents=True)
    copied = {}

    def copy(source, relative):
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert source.read_bytes() == target.read_bytes()
        copied[relative] = {"source_path": str(source), "source_sha256": sha(source), "encoding": "original bytes"}

    for run in summary["runs"]:
        name = f"{run['objective']}_seed{run['seed']}"
        for filename in ("protocol.json", "result.json", "status.json"):
            copy(RAW / name / filename, f"runs/{name}/{filename}")
    for source, target in [
        (RAW / "reports/graph_convention_bridge.md", "report.md"),
        (RAW / "reports/analysis_source.json", "analysis_source.json"),
        (RAW / "launch.json", "queue/launch.json"),
        (RAW / "queue_status.json", "queue/status.json"),
        (AUDIT, "graph_bridge_aggregate_audit.json"),
        (AUDIT.with_name("audit_graph_bridge_aggregates.py"), "independent_audit.py"),
        (REPO / "research/protocols/graph_convention_bridge_20261005.md", "protocol.md"),
        (AUDIT.with_name("source_freeze.json"), "source_freeze.json"),
    ]:
        copy(source, target)
    compressed = DEST / "graph_convention_bridge.json.gz"
    with compressed.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as zipped:
            with summary_file.open("rb") as source:
                shutil.copyfileobj(source, zipped)
    assert gzip.decompress(compressed.read_bytes()) == summary_file.read_bytes()
    copied[compressed.name] = {"source_path": str(summary_file), "source_sha256": sha(summary_file),
        "source_size_bytes": summary_file.stat().st_size, "encoding": "deterministic gzip; decompressed original bytes"}

    compact = {k: v for k, v in summary.items() if k != "runs"}
    compact["publication"] = {"scope": "Exact aggregate values and all seed/trajectory summaries from original strict analysis; no recomputation of scientific outputs",
        "original_full_summary": "graph_convention_bridge.json.gz", "original_summary_sha256": sha(summary_file),
        "per_frame_summary_records": "All 2550 are preserved without alteration in graph_convention_bridge.json.gz",
        "input_hashes_and_protocols": "Original runs/*/protocol.json; original full summary also preserves its complete input inventory"}
    compact["runs"] = [{k: v for k, v in run.items() if k not in ("frames", "input_files_sha256", "source_protocol")} |
        {"full_summary_run_index": index, "original_protocol": f"runs/{run['objective']}_seed{run['seed']}/protocol.json"}
        for index, run in enumerate(summary["runs"])]
    write(DEST / "results.json", compact)

    coverage = {"required_models": 6, "required_frames_per_model": {"valid": 128, "test": 297},
        "required_cases_per_frame": 16, "models": {}, "total_frames": 0, "failed_frames": 0,
        "total_case_outcomes": 0, "failed_case_outcomes": 0, "native_parity_passed_frames": 0}
    provenance = {"scope": "Identities copied from the audited run protocols; no training or prediction performed by publication",
        "original_raw_root": str(RAW), "models": {}, "source_files_rechecked": {}, "converted_manifest_identities": {},
        "summary_sha256": sha(summary_file), "summary_source_sha256": summary["summary_source_sha256"]}
    for run in summary["runs"]:
        name = f"{run['objective']}_seed{run['seed']}"
        protocol = json.loads((RAW / name / "protocol.json").read_text())
        result = json.loads((RAW / name / "result.json").read_text())
        status = json.loads((RAW / name / "status.json").read_text())
        assert status["state"] == result["state"] == "complete"
        assert status["result_sha256"] == original[f"{name}/result.json"]["sha256"]
        assert result["protocol_sha256"] == original[f"{name}/protocol.json"]["sha256"] == run["protocol_sha256"]
        assert len(result["records"]) == 425
        assert set(protocol["cases"]) == set(run["splits"]["test"]["graph_diagnostics"]["failed_cases"])
        cases = protocol["cases"]
        records_by_id = {(r["split"], r["source_index"], r["target_frame"]): r for r in run["frames"]}
        assert len(records_by_id) == 425
        model = {"objective": run["objective"], "seed": run["seed"], "splits": {}}
        for split in ("valid", "test"):
            model["splits"][split] = {"required_frames": run["splits"][split]["required_frames"],
                "observed_frames": 0, "failed_frames": 0, "native_parity_passed_frames": 0,
                "cases": {c: {"observed": 0, "complete": 0, "failed": 0, "missing": 0} for c in cases}}
        for record in result["records"]:
            for filename, hashname in (("record_file", "record_sha256"), ("array_file", "array_sha256")):
                assert original[f"{name}/{record[filename]}"]["sha256"] == record[hashname]
            row = json.loads((RAW / name / record["record_file"]).read_text())
            key = (row["split"], row["source_index"], row["target_frame"])
            assert records_by_id[key]["status"] == row["status"] == record["status"]
            assert set(row["cases"]) == set(cases)
            split = model["splits"][row["split"]]
            split["observed_frames"] += 1
            split["failed_frames"] += row["status"] != "complete"
            split["native_parity_passed_frames"] += bool(row["native_parity"]["passed"])
            for case, values in row["cases"].items():
                assert values["status"] in ("complete", "failed")
                split["cases"][case]["observed"] += 1
                split["cases"][case][values["status"]] += 1
        for split_name, split in model["splits"].items():
            assert split["observed_frames"] == split["required_frames"]
            assert split["failed_frames"] == run["splits"][split_name]["failed_frames"]
            for case, counts in split["cases"].items():
                counts["missing"] = split["required_frames"] - counts["observed"]
                assert counts["failed"] == run["splits"][split_name]["graph_diagnostics"]["failed_cases"][case]
            coverage["total_frames"] += split["observed_frames"]
            coverage["failed_frames"] += split["failed_frames"]
            coverage["native_parity_passed_frames"] += split["native_parity_passed_frames"]
            coverage["total_case_outcomes"] += sum(c["observed"] for c in split["cases"].values())
            coverage["failed_case_outcomes"] += sum(c["failed"] for c in split["cases"].values())
        coverage["models"][name] = model
        provenance["models"][name] = {k: protocol[k] for k in ("seed", "objective", "checkpoint_sha256", "run_config_sha256", "bridge_protocol_sha256", "runtime", "software", "threads")}
        provenance["models"][name]["original_protocol"] = f"runs/{name}/protocol.json"
        provenance["models"][name]["original_protocol_sha256"] = run["protocol_sha256"]
        for source_path, expected in protocol["input_files_sha256"].items():
            p = Path(source_path)
            if str(p).startswith(str(REPO)) and p.suffix in (".py", ".md"):
                assert sha(p) == expected
                provenance["source_files_rechecked"][str(p.relative_to(REPO))] = expected
            if p.name in ("valid.json", "test.json", "metadata.json"):
                assert sha(p) == expected
                provenance["converted_manifest_identities"][p.name] = {"path": str(p), "sha256": expected}
                if p.name != "metadata.json":
                    provenance["converted_manifest_identities"][p.name]["official_source"] = json.loads(p.read_text())["source"]
    assert coverage["total_frames"] == 2550 and coverage["total_case_outcomes"] == 40800
    write(DEST / "coverage.json", coverage)
    write(DEST / "provenance.json", provenance)

    encoded_by_source = {v["source_path"]: {"file": k, "encoding": v["encoding"]} for k, v in copied.items()}
    inventory = {"scope": "Every file under the completed bridge raw output root; external training/data/same-state inputs are separately identified by run protocols",
        "source_root": str(RAW), "file_count": len(original), "size_bytes": sum(x["size_bytes"] for x in original.values()), "files": []}
    omitted_kinds, omitted_bytes = Counter(), Counter()
    for path, identity in original.items():
        location = encoded_by_source.get(str(RAW / path))
        kind = "particle_arrays" if path.endswith(".npz") else "raw_frame_json" if Path(path).name.startswith(("valid_", "test_")) else "process_log" if path.endswith(".log") else "compact_artifact"
        inventory["files"].append({"path": path, **identity, "kind": kind, "publication": location,
            "omitted_reason": None if location else "Retained locally; particle arrays require the original run, and raw frame JSON/logs are outside this compact package"})
        if not location:
            omitted_kinds[kind] += 1; omitted_bytes[kind] += identity["size_bytes"]
    inventory["omitted_counts"] = dict(omitted_kinds)
    inventory["omitted_bytes"] = dict(omitted_bytes)
    write(DEST / "raw_inventory.json", inventory)

    def fmt(item, digits=8):
        return f"{item['mean']:.{digits}g} ± {item['sample_sd']:.{digits}g}"
    rows = []
    for split in ("valid", "test"):
        for objective in ("faithful", "nll"):
            m = summary["groups"][objective][split]["measures"]
            f = independent["findings"][f"{objective}/{split}"]
            rows.append(f"| {split} | {objective} | {fmt(m['case__base_uncapped_loops0__normalized_coordinate_mse'])} | {fmt(m['case__base_uncapped_loops1__normalized_coordinate_mse'])} | {fmt(f['base_mse_relative_reduction_percent'], 5)}% | {fmt(f['loops1_risk_minus_random'])} |")
    README = """# Exploratory graph-convention bridge

Restoring native self-messages lowers observed-history base prediction error, but previous-observed-risk allocation still has higher mean error than random allocation in every objective/seed on both splits. The six original final 100k checkpoints are unchanged. This follow-up was designed after inspection of the completed fixed study; it does not replace or repair its original results or eight NLL rollout failures.

Each model evaluates the same 128 validation histories (30 trajectories) and 297 existing test histories (27 trajectories, indices 3–29) under all 16 declared graph cases. **2,550 frames and 40,800 case outcomes completed, with zero failed/missing cases; all 2,550 native identity/parity gates passed.** These are paired repeated observations from six models, not 40,800 independent statistical samples. `coverage.json` provides every model/split/case count, including zeros.

The table reports decoder-equivalent normalized-acceleration coordinate MSE, with equal-history averages within each trajectory, equal-trajectory averages within each seed, then mean ± sample SD across seeds 0–2. Positive risk-minus-random differences favor random. Base reduction is computed as 100×(loop-off−loop-on)/loop-off **inside each seed**, followed by the seed mean and sample SD; it is not a ratio of group means. All16 cases, all seed/trajectory summaries, 37 paired contrasts, signs, null/coverage accounting and graph diagnostics are retained in `results.json`; the exact original per-frame summary is retained in `graph_convention_bridge.json.gz`.

| Split | Objective | Base, loops off | Base, loops on | Within-seed base reduction | Previous risk minus random, loops on |
|---|---|---:|---:|---:|---:|
""" + "\n".join(rows) + """

With loops on, dense-minus-base test error is **0.00204236 ± 0.000881155** for faithful and **0.00200769 ± 0.000164968** for NLL; dense is worse in all three seeds of each objective. Current-risk minus previous-risk test differences are **0.0000391816 ± 0.0000567699** and **0.00000260643 ± 0.0000192493**, respectively. Fresh scoring does not establish a reliable improvement in this three-seed observed-history diagnostic. The complete validation counterparts and all loop-off contrasts remain in the original report and JSON.

The cap never binds in the measured current base/dense graphs or previous-base scoring graphs; the new and old radius classifiers select the same measured pair universes. Cap inactivity on the 297 test states was already inspected before this protocol (maximum nonself base/dense degrees 22/31), so it is not an independent discovery. Validation confirms the measured cap behavior. These observations do not establish cap inactivity during autonomous rollouts.

Base/dense/random/speed keep their selected nonself pair sets fixed between loop arms. Risk scores and selections change with the loop convention, so risk loop contrasts include both changes. The bridge's random seed material is `[20261005,771,training_seed,split_code,source_index,target_frame]`, shared between loop arms and objectives; this is **a different draw from the original locked random policy**. Cross-study differences cannot be attributed solely to loops. Previous risk uses the preceding observed history; current risk uses the current observed base graph. Neither is the autonomous policy's cached risk from its own preceding selected graph. The bridge globally orders directed edges; the separate native autonomous study preserves native base edges as a prefix before appending optional edges.

All results are exploratory. The exact same already inspected test histories and previously used validation schedule are reused; the first three official test trajectories and historical aggregates were inspected earlier. This is not pristine independent confirmation, a new test population, a significance claim, an autonomous-stability result, or evidence that expanded-edge training support is adequate. Single-call times share work and use fixed case order; they do not establish policy latency or speedup.

## Included and omitted evidence

- `report.md`, `analysis_source.json`, the six `runs/*/{protocol,result,status}.json` triplets and queue records preserve original bytes.
- `graph_convention_bridge.json.gz` decompresses to the exact 53,075,022-byte strict-summary JSON (SHA256 `1623cab0517dd4a20ceb668e322aa98ff064a6dc4e54b62b63d46d7c05ff27d5`). It retains every per-frame derived scalar and all original input-file identities. `results.json` is a lossless projection of its aggregate/seed/trajectory sections; it removes only duplicated per-frame records, input hash dictionaries and full protocol copies, which remain available in the compressed original and run protocols.
- `independent_audit.py` and `graph_bridge_aggregate_audit.json` preserve the separate 5777-check aggregate audit; the strict summary records 851400 geometry, arithmetic, coverage and integrity checks. These are verification counts, not scientific samples. `protocol.md`, `source_freeze.json` and `provenance.json` pin source/configuration/checkpoint/data identities. This curation rechecked current pinned source files and converted manifest bytes; it did not rerun inference or rehash the original dataset/checkpoint payloads.
- `raw_inventory.json` records relative paths, sizes and SHA256 for **all 5129 raw files**, including every omitted output. The omitted 2550 particle NPZ archives, 2550 per-frame raw JSON records and six process logs remain unmodified locally. NPZ holds particle predictions, risks, histories, features and graph arrays. The per-frame JSON contains original detailed case timing and graph records beyond the derived scalar summary. Logs contain process output. No training/validation/test datasets or checkpoints are embedded. Their hashes identify assets; they are not download links or substitutes for missing files. Complete source/geometry reauditing requires the inventoried raw arrays and external data.

`PUBLICATION_MANIFEST.json` and `PUBLICATION_AUDIT.json` check package identity and document exact copied-source relationships. Run `python3 verify_publication.py` from this directory to verify packaged hashes, compressed-source identity, exact aggregate projection and all-case coverage without original data or model inference. Absolute paths in unchanged originals are provenance strings, not portable paths. This is nonanonymous research-fork evidence; anonymity, asset rights and author verification remain separate submission checks.

## Reproduction

The frozen evaluator is `research/graph_convention_bridge.py`; `protocol.md` gives its checkpoint/data/runtime command contract and native identity gate. With all six original raw output directories available, run from the repository root:

```sh
python -m research.summarize_graph_convention_bridge --runs \\
  /path/to/bridge/faithful_seed0 /path/to/bridge/faithful_seed1 /path/to/bridge/faithful_seed2 \\
  /path/to/bridge/nll_seed0 /path/to/bridge/nll_seed1 /path/to/bridge/nll_seed2 \\
  --output-dir /path/to/new-report-directory
```

The strict summary requires a new output directory and all original referenced files; it performs no inference and verifies hashes/coverage. Preserved originals use the recorded local absolute paths, so reproducing these exact identity checks requires that layout or a fresh protocol-governed rerun from available inputs. The separate aggregate-audit script retains its original workspace paths and can be run in the recorded layout. This publication is not a standalone rerun bundle.
"""
    (DEST / "README.md").write_text(README)
    # Preserve the exact curation procedure along with scientific artifacts.
    copy(Path(__file__).resolve(), "curate_publication.py")
    verifier = Path(__file__).with_name("verify_graph_bridge_publication.py")
    copy(verifier, "verify_publication.py")
    checks = 0
    assert set(original) == {p.relative_to(RAW).as_posix() for p in RAW.rglob("*") if p.is_file()}
    for path, expected in original.items():
        assert (RAW / path).stat().st_size == expected["size_bytes"] and sha(RAW / path) == expected["sha256"]
        checks += 1
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for p in sorted(DEST.rglob("*")):
        if p.is_file():
            relative = p.relative_to(DEST).as_posix()
            manifest["files"][relative] = {"size_bytes": p.stat().st_size, "sha256": sha(p), **copied.get(relative, {})}
            checks += 1
    write(DEST / "PUBLICATION_MANIFEST.json", manifest)
    write(DEST / "PUBLICATION_AUDIT.json", {"passed": True, "scope": "Package byte identity, all raw output hashes stable, result-index correspondence and complete all-case coverage; no model execution",
        "created_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha(DEST / "PUBLICATION_MANIFEST.json"),
        "raw_files_checked_before_and_after": len(original), "raw_bytes_preserved": inventory["size_bytes"],
        "copied_original_artifacts": len(copied), "package_files_excluding_manifest_and_audit": len(manifest["files"]),
        "package_bytes_excluding_manifest_and_audit": sum(x["size_bytes"] for x in manifest["files"].values()),
        "hash_and_manifest_checks": checks, "all_case_outcomes": coverage["total_case_outcomes"],
        "failed_case_outcomes": coverage["failed_case_outcomes"], "omitted_counts": dict(omitted_kinds), "omitted_bytes": dict(omitted_bytes)})
    print(json.dumps(json.loads((DEST / "PUBLICATION_AUDIT.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
