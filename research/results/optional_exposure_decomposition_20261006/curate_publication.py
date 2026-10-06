"""Curate completed, independently audited saved-array exposure evidence; no inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import shutil
import tarfile
import sys

sys.dont_write_bytecode = True

try:
    from verify_publication import compact_projection, digest, require, safe_name, sha, validate_result, verify
except ImportError:
    from verify_optional_exposure_publication import compact_projection, digest, require, safe_name, sha, validate_result, verify

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/AdaptGNS"
SOURCE_HASH = "c7a915d008f237551cc7d4efdcd63bc86bef4e555c9f53d6d25b51b1b1cf7e80"
PROTOCOL_HASH = "050393b2270c0d69ae94dd3644b46e336496e30d556aa6aa1e7025225dc2fe83"
TEST_HASH = "45f2c663830afe91ffa5c5901e75c4871b9de3f64746cabeec5b858f8ff3ad9e"
SOURCE_COMMIT = "0f3589f"
VERIFIER_PATH = Path(verify.__code__.co_filename).resolve()
TEST_PATH = Path(__file__).with_name("test_optional_exposure_publication.py")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(root):
    require(root.is_dir() and not root.is_symlink(), "Missing/nonsupported raw directory")
    items = sorted(root.rglob("*"))
    require(not any(p.is_symlink() for p in items), "Do not follow raw evidence symlinks")
    return {p.relative_to(root).as_posix(): {"sha256": sha(p), "size_bytes": p.stat().st_size}
            for p in items if p.is_file()}


def exact_gzip(source, target):
    with target.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as zipped:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, zipped)
    require(gzip.decompress(target.read_bytes()) == source.read_bytes(), "Exact gzip changed source bytes")


def exact_archive(raw, names, destination):
    """Archive exact JSON bytes with stable order, gzip header, and tar metadata."""
    require(len(names) == len(set(names)), "Duplicate requested archive name")
    index = {}
    nulls = 0
    with destination.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted(names):
                    safe_name(name)
                    source = raw / name
                    require(source.is_file() and not source.is_symlink(), "Missing archived JSON")
                    data = source.read_bytes()
                    row = json.loads(data)
                    header = tarfile.TarInfo(name)
                    header.size = len(data)
                    header.mode = 0o644
                    header.uid = header.gid = header.mtime = 0
                    header.uname = header.gname = ""
                    archive.addfile(header, io.BytesIO(data))
                    null_count = len(row["undefined_reasons"])
                    index[name] = {"sha256": digest(data), "size_bytes": len(data), "null_metric_count": null_count}
                    nulls += null_count
    return {"encoding": "deterministic tar.gz containing exact original per-frame JSON bytes", "record_count": len(index),
            "null_metric_count": nulls, "files": index}


def curator(args):
    raw, destination, repo = args.raw.resolve(), args.destination.resolve(), args.repo.resolve()
    require(not destination.exists(), "Preserve existing publication directory")
    require(raw != destination and raw not in destination.parents and destination not in raw.parents,
            "Publication must be separate from raw evidence")
    result_path = raw / "results.json"
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    records = validate_result(result)
    status = json.loads((raw / "status.json").read_text())
    require(status["state"] == "complete" and status["results_sha256"] == digest(result_bytes)
            and status["report_sha256"] == sha(raw / "report.md"), "Original completion identity differs")
    audit = json.loads(args.audit.read_text())
    require(audit.get("passed") is True and audit["results_sha256"] == digest(result_bytes)
            and audit["audited_frames"] == 1782, "Independent full-array audit must pass first")
    require(audit["source_sha256"] == sha(args.audit_source), "Independent audit source differs")
    require(audit["analysis_source_sha256"] == SOURCE_HASH and audit["analysis_protocol_sha256"] == PROTOCOL_HASH,
            "Independent audit covers another analysis")
    source = repo / "research/analyze_optional_exposure_decomposition.py"
    protocol = repo / "research/protocols/optional_exposure_decomposition_20261006.md"
    tests = repo / "research/tests/test_optional_exposure_decomposition.py"
    require(sha(source) == SOURCE_HASH and sha(protocol) == PROTOCOL_HASH and sha(tests) == TEST_HASH,
            "Frozen analysis source, protocol or synthetic tests differ")
    require(result["source_sha256"][str(source)] == SOURCE_HASH and result["protocol_sha256"] == PROTOCOL_HASH,
            "Original result is from another source")
    review = json.loads(args.review.read_text())
    require(review["state"] == "pass" and review["source_sha256"]["research/analyze_optional_exposure_decomposition.py"] == SOURCE_HASH
            and review["source_sha256"]["research/protocols/optional_exposure_decomposition_20261006.md"] == PROTOCOL_HASH
            and review["source_sha256"]["research/tests/test_optional_exposure_decomposition.py"] == TEST_HASH,
            "Pre-execution review identity differs")
    raw_before = snapshot(raw)
    require(raw_before["results.json"]["sha256"] == digest(result_bytes), "Results changed while opening")
    external = {source, protocol, tests, args.audit, args.audit_source, args.audit_tests, args.review, Path(__file__).resolve(),
                VERIFIER_PATH, TEST_PATH}
    claim_checks = getattr(args, "claim_checks", None)
    if claim_checks is not None:
        claim = json.loads(claim_checks.read_text())
        require(claim["passed"] and claim["results_sha256"] == digest(result_bytes) and claim["audit_record_sha256"] == sha(args.audit), "Claim check identity differs")
        external.add(claim_checks)
    audit_attempts = sorted(args.audit.parent.glob("attempt_*.json"))
    external.update(audit_attempts)
    render_files = []
    if args.render_dir is not None:
        render = json.loads((args.render_dir / "render_manifest.json").read_text())
        require(render["results_sha256"] == digest(result_bytes) and render["independent_audit_sha256"] == sha(args.audit),
                "Renderer uses another result/audit")
        renderer_source = repo / "research/render_optional_exposure_findings.py"
        require(render["renderer_sha256"] == sha(renderer_source), "Renderer source differs")
        actual = {p.name for p in args.render_dir.iterdir() if p.is_file()}
        require(actual == set(render["outputs"]) | {"render_manifest.json"}, "Unexpected renderer output set")
        for relative, expected in render["outputs"].items():
            safe_name(relative)
            p = args.render_dir / relative
            require(p.is_file() and not p.is_symlink() and sha(p) == expected, "Renderer output hash differs")
        render_files = [args.render_dir / name for name in sorted(actual)] + [renderer_source]
        external.update(render_files)
    external_before = {str(path): {"sha256": sha(path), "size_bytes": path.stat().st_size} for path in external}
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}

    def copy(source, relative):
        safe_name(relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "Duplicate publication target")
        shutil.copyfile(source, target)
        require(sha(source) == sha(target), "Copy changed bytes")
        copied[relative] = {"source_path": str(source.resolve()), "source_sha256": sha(source), "encoding": "original_bytes"}

    for name in ("status.json", "report.md", "input_identity.json", "all_inputs_sha256.json"):
        copy(raw / name, name)
    for source_file, relative in [(source, "analysis_source.py"), (protocol, "protocol.md"), (tests, "analysis_tests.py"),
            (args.audit, "independent_audit.json"), (args.audit_source, "independent_audit.py"), (args.audit_tests, "independent_audit_tests.py"),
            (args.review, "pre_execution_review.json"), (Path(__file__).resolve(), "curate_publication.py"),
            (VERIFIER_PATH, "verify_publication.py"), (TEST_PATH, "test_optional_exposure_publication.py"),
            (args.audit_source, "audit_optional_exposure_decomposition.py")]:
        copy(source_file, relative)
    if claim_checks is not None:
        copy(claim_checks, "claim_checks.json")
    for attempt in audit_attempts:
        copy(attempt, "audit_attempts/" + attempt.name)
    for path in render_files:
        copy(path, "rendered/" + path.name if path.parent == args.render_dir else "renderer_source.py")
    exact_gzip(result_path, destination / "results.json.gz")
    copied["results.json.gz"] = {"source_path": str(result_path), "source_sha256": digest(result_bytes),
        "source_size_bytes": len(result_bytes), "encoding": "deterministic_gzip_original_bytes"}
    write(destination / "results.json", compact_projection(result))
    archive_index = exact_archive(raw, list(records), destination / "frame_records.tar.gz")
    for name, row in archive_index["files"].items():
        require(row["sha256"] == raw_before[name]["sha256"] == records[name]["record_sha256"], "Archived record identity changed")
    write(destination / "frame_archive_index.json", archive_index)
    provenance = {"scope": "Compact publication of independently audited post-inspection saved-array decomposition; no inference or scientific recomputation",
        "raw_root": str(raw), "result_sha256": digest(result_bytes), "source_freeze_commit": SOURCE_COMMIT,
        "analysis_source_sha256": SOURCE_HASH, "analysis_protocol_sha256": PROTOCOL_HASH,
        "independent_audit_result_hash_field": "results_sha256", "independent_audit_source_sha256": sha(args.audit_source),
        "independent_audit_sha256": sha(args.audit), "pre_execution_review_sha256": sha(args.review),
        "operational_timing": {"cpu_saved_array_analysis_wall_seconds": result["analysis_wall_seconds"], "independent_audit_wall_seconds": audit["wall_seconds"], "interpretation": "Postprocessing/audit operational times; not inference latency or speedup"},
        "source_or_audit_artifacts_rechecked": external_before,
        "asset_scope": "Original checkpoints/datasets are identified by unchanged input provenance, not embedded or rehashed by curation"}
    write(destination / "provenance.json", provenance)
    copied_locations = {v["source_path"]: {"file": name, "encoding": v["encoding"]} for name, v in copied.items()}
    inventory = {"scope": "Every raw output under the completed optional-exposure directory, including omitted derived particle arrays and logs",
                 "source_root": str(raw), "file_count": len(raw_before), "size_bytes": sum(r["size_bytes"] for r in raw_before.values()), "files": []}
    omitted_count, omitted_bytes = Counter(), Counter()
    for name, identity in raw_before.items():
        location = copied_locations.get(str(raw / name))
        if name in records:
            location = {"file": "frame_records.tar.gz", "member": name, "encoding": "tar_member_original_bytes"}
        kind = "derived_particle_arrays" if name.endswith(".npz") else "frame_record" if name in records else "other_output"
        inventory["files"].append({"path": name, **identity, "kind": kind, "publication": location,
            "omitted_reason": None if location else "Retained unchanged locally; not embedded in this compact package"})
        if location is None:
            omitted_count[kind] += 1
            omitted_bytes[kind] += identity["size_bytes"]
    inventory.update(omitted_counts=dict(omitted_count), omitted_bytes=dict(omitted_bytes))
    write(destination / "raw_inventory.json", inventory)
    (destination / "README.md").write_text(readme(archive_index, raw_before, inventory, args.render_dir is not None, result["analysis_wall_seconds"], audit["wall_seconds"]))
    require(snapshot(raw) == raw_before, "Raw evidence changed during publication; retain partial package for review")
    require(all(path.is_file() and sha(path) == external_before[str(path)]["sha256"] for path in external),
            "Source/review/audit/renderer changed during publication")
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            name = path.relative_to(destination).as_posix()
            manifest["files"][name] = {"size_bytes": path.stat().st_size, "sha256": sha(path), **copied.get(name, {})}
    write(destination / "PUBLICATION_MANIFEST.json", manifest)
    write(destination / "PUBLICATION_AUDIT.json", {"passed": True, "scope": "Original byte identities, stable raw inventory and all-frame archive correspondence; standalone verifier validates package. Separate independent audit owns particle arithmetic.",
        "manifest_sha256": sha(destination / "PUBLICATION_MANIFEST.json"), "raw_files_checked_before_and_after": len(raw_before),
        "raw_bytes_checked_before_and_after": inventory["size_bytes"], "frame_records": archive_index["record_count"],
        "frame_null_metrics_preserved": archive_index["null_metric_count"], "omitted_counts": dict(omitted_count)})
    try:
        return verify(destination)
    except Exception as error:
        failed = json.loads((destination / "PUBLICATION_AUDIT.json").read_text())
        failed.update(passed=False, verifier_error=repr(error), partial_package_preserved=True)
        write(destination / "PUBLICATION_AUDIT.json", failed)
        raise


def readme(index, raw_files, inventory, rendered, analysis_seconds, audit_seconds):
    return f'''# Exploratory optional-exposure decomposition

This post-inspection study partitions paired errors from the six original 100k models' saved, no-loop, observed-history predictions. It performs no new model inference. All six models, all 27 test trajectories (indices 3–29), all 11 fixed target frames, all three comparisons and all seven group records are retained. The test histories and previous action-benefit results were inspected before this protocol; these results are not independent confirmation.

The CPU saved-array analysis took {analysis_seconds:.4f} s; the separate independent audit took {audit_seconds:.4f} s. These operational timings are separate from model inference latency.

`report.md` is the exact original scientific report, with every sign and reported outcome unchanged. `results.json` preserves the original top-level values, all objective summaries and all model aggregates except repeated per-trajectory records and frame indices. `results.json.gz` decompresses to the exact original {raw_files['results.json']['size_bytes']:,}-byte result (SHA256 `{raw_files['results.json']['sha256']}`), including all trajectory aggregates, frame indices, null coverage, checkpoint/configuration identities, sample SDs and all three seed values.

## Design and interpretation

The contrasts are risk minus random, risk minus speed, and speed minus random. Risk is the original `previous-observed-base-risk25` policy. Positive error differences favor the right policy. Error, alignment and perturbation cost use normalized acceleration coordinate units, dividing the previous vector quantities by two. Error difference equals cost difference minus alignment difference.

The four primary groups (`neither`, `left_only`, `right_only`, `both`) partition particles by direct optional-edge incidence. Each contribution is sum(mask × paired quantity)/N; the four contributions sum to the whole-frame difference. `both_less`, `both_equal`, and `both_more` partition only the `both` group. The primary and supplementary groups must not be added together. A particle without a directly incident optional edge can still be affected through message passing.

Frames are averaged equally within each trajectory, trajectories equally within each seed, and the three seed means are summarized by mean and sample SD. The histories and particles are repeated observations, not independent replications. Empty groups have zero fractions/contributions and null conditional means. Strict conditional means require every frame in the fixed population; separately named weighted conditionals divide the weighted contribution by the weighted group fraction. No available-frame averaging, particle pooling, significance claim or multiplicity-adjusted inference is introduced.

This is a descriptive partition by policy-defined exposure. It does not adjust for pretreatment confounding, identify marginal edge value or a causal concentration mechanism, establish autonomous stability, or measure inference speed. It does not replace the original fixed experiment, its eight NLL rollout failures, the separate native self-loop study, or the 110k continuation study.

## Exact evidence and omission scope

- `frame_records.tar.gz` contains all **{index['record_count']:,} exact original per-frame JSON files**, with all three comparisons, primary and supplementary group outcomes, input/derived-array hashes and **{index['null_metric_count']:,} null metric entries**. `frame_archive_index.json` records each archive member's size, hash and null count. Empty-group null outcomes are retained.
- `status.json`, `report.md`, `input_identity.json`, `all_inputs_sha256.json`, `independent_audit.json`, `independent_audit.py`, `independent_audit_tests.py`, `pre_execution_review.json`, `protocol.md`, analysis source and tests preserve exact bytes. Every existing `attempt_*.json` in the audit directory is copied to `audit_attempts/`, including unsuccessful audit attempts if present. `provenance.json` links the source freeze `{SOURCE_COMMIT}` and separate audit.
- `raw_inventory.json` lists **all {len(raw_files):,} raw files** and {inventory['size_bytes']:,} bytes, with SHA256 and explicit included/omitted locations. Derived particle NPZ arrays remain unchanged locally; every NPZ identity is linked from its archived frame record. Other nonembedded outputs, if present, are also inventoried. This package retains all scalar/group/null outcomes without duplicating particle vectors.
- No checkpoints, datasets or previous raw same-state/action-benefit inputs are embedded. Their preserved hashes identify required assets but are not download links. Full particle arithmetic reauditing requires those external assets and omitted arrays. The curation verifies stable output bytes; it does not rerun the separate full-array audit or rehash dataset/checkpoint payloads.
''' + ('''
`rendered/` preserves verified TeX inserts, PDF/PNG figure, exact plot coordinates and the renderer manifest. The exact renderer source is `renderer_source.py`.
''' if rendered else '') + '''
Run `python3 verify_publication.py` in this directory for the standard-library package check. It verifies every packaged hash, exact decompressed result/projection, original status/audit linkage, all 1,782 exact JSON records, null accounting and omitted-array identities without extracting the archive or loading models. `PUBLICATION_MANIFEST.json` and `PUBLICATION_AUDIT.json` record package identity and the raw byte-stability check.

## Reproduction

The frozen implementation is `research/analyze_optional_exposure_decomposition.py`; `protocol.md` gives the complete contract. With the original saved inputs available, run from the repository root:

```sh
python -m research.analyze_optional_exposure_decomposition \\
  --evaluation-root /path/to/original/full-same-state \\
  --action-root /path/to/original/action-benefit \\
  --output-dir /path/to/new-exposure-output
```

The command requires a fresh output directory, all original fixed-population inputs and pinned hashes. Absolute paths in exact originals are provenance strings and may require the recorded layout for exact identity replay. The separate audit script accepts explicit analysis, repository and output paths plus expected source/protocol hashes; it also requires the original saved inputs. This compact evidence package is not a standalone rerun bundle. Research-fork artifacts are nonanonymous; author verification, anonymity and asset rights remain submission checks.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-source", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--audit-tests", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--claim-checks", type=Path)
    args = parser.parse_args()
    for name in ("raw", "destination", "audit", "audit_source", "audit_tests", "review", "repo", "render_dir", "claim_checks"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    curator(args)


if __name__ == "__main__":
    main()
