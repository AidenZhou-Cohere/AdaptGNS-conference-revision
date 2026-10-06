"""Copy final derived reporting evidence and replay renderer; never perform inference."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from verify_optional_exposure_reporting import need, read, sha, verify

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/AdaptGNS"
WORK = ROOT / "work/deadline_research_20261005"
SCIENCE = REPO / "research/results/optional_exposure_decomposition_20261006"
RENDER = WORK / "optional_exposure_render_v3"
DEST = REPO / "research/results/optional_exposure_analysis_20261006"
PYTHON = ROOT / "work/venv/bin/python"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inventory(root):
    need(not any(p.is_symlink() for p in root.rglob("*")), "Evidence symlinks unsupported")
    return {p.relative_to(root).as_posix(): {"sha256": sha(p), "size_bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    need(not DEST.exists(), "Preserve previous reporting package")
    science_before = inventory(SCIENCE)
    render_manifest = read(RENDER / "render_manifest.json")
    need(sha(RENDER / "render_manifest.json") == "9515730c519cb9da5c8e86d2fb5f2d54c3352bc4824391e2ab6b35a0b07c9fff", "Final v3 render manifest differs")
    need(sha(WORK / "optional_exposure_renderer_review_v3.json") == "abaf4940003b41b1a5d787dd4646497793a61f25c2245584a28aa70e557725a9", "Final renderer review differs")
    need(sha(WORK / "optional_exposure_report_review_v2.json") == "423ae0c816631557746b54891a5751efd424843dd0c4bd30ca026dc6b769365b", "Final report review differs")
    need(sha(WORK / "optional_exposure_document_preservation_review.json") == "a5881a5fdb46a9d16e4145c382431dbb53e3b74beb93d74304f3db3c635e3212", "Final preservation review differs")
    sources = {name: RENDER / name for name in render_manifest["outputs"]}
    sources.update({
        "render_manifest.json": RENDER / "render_manifest.json",
        "renderer_source.py": REPO / "research/render_optional_exposure_findings.py",
        "test_render_optional_exposure_findings.py": REPO / "research/tests/test_render_optional_exposure_findings.py",
        "renderer_review_v3.json": WORK / "optional_exposure_renderer_review_v3.json",
        "audit_optional_exposure_renderer.py": WORK / "audit_optional_exposure_renderer.py",
        "report_review_v2.json": WORK / "optional_exposure_report_review_v2.json",
        "audit_optional_exposure_report.py": WORK / "audit_optional_exposure_report.py",
        "compact_publication_verification.json": WORK / "optional_exposure_publication_verification.json",
        "manuscript_preservation_review.json": WORK / "optional_exposure_document_preservation_review.json",
        "audit_optional_document_preservation.py": WORK / "audit_optional_document_preservation.py",
        "document_verification.json": WORK / "optional_document_verification.json",
        "curate_reporting.py": Path(__file__).resolve(),
        "verify_reporting.py": WORK / "verify_optional_exposure_reporting.py",
    })
    for name in ("optional_exposure_renderer_review_v2.json", "optional_exposure_report_review_v1.json"):
        sources["superseded_reviews/" + name] = WORK / name
    before = {str(path): {"sha256": sha(path), "size_bytes": path.stat().st_size} for path in sources.values()}
    attempts = {}
    for name in ("optional_exposure_render", "optional_exposure_render_v2", "optional_exposure_render_v3"):
        attempts[name] = {"source_root": str(WORK / name), "state": "final" if name.endswith("v3") else "superseded_layout_or_wording",
                          "files": inventory(WORK / name), "publication": "root package files" if name.endswith("v3") else "Retained unchanged locally; identities inventoried here"}
    doc = read(WORK / "optional_document_verification.json")
    manuscript, report = ROOT / "outputs/revised_manuscript.tex", ROOT / "outputs/revision_report.pdf"
    need(sha(manuscript) == doc["manuscript_sha256"] and sha(report) == doc["report_sha256"], "Final document bytes differ")
    document_hashes = {str(p): sha(p) for p in (manuscript, report)}
    DEST.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        target = DEST / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        need(sha(target) == before[str(source)]["sha256"], "Copied artifact bytes differ")
    write(DEST / "render_attempt_inventory.json", {"scope": "All three preserved renderer output attempts; only v3 is final", "attempts": attempts})
    temp_root = Path(tempfile.mkdtemp(prefix="optional_exposure_reporting_replay_", dir=WORK))
    render_output = temp_root / "rendered"
    command = [str(PYTHON), str(DEST / "renderer_source.py"), "--results", str(SCIENCE / "results.json.gz"),
               "--audit", str(SCIENCE / "independent_audit.json"), "--output-dir", str(render_output)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    started = datetime.now(timezone.utc).isoformat()
    execution = subprocess.run(command, cwd=DEST, env=env, text=True, capture_output=True)
    write(temp_root / "execution.json", {"command": command, "cwd": str(DEST), "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "returncode": execution.returncode,
        "stdout": execution.stdout, "stderr": execution.stderr})
    need(execution.returncode == 0, "Replay failed; attempt preserved")
    exact_names = ("optional_exposure_main.tex", "optional_exposure_appendix.tex", "plot_coordinates.json", "mechanism_findings.json")
    exact = {name: {"reference_sha256": sha(DEST / name), "replayed_sha256": sha(render_output / name),
                   "byte_identical": (DEST / name).read_bytes() == (render_output / name).read_bytes()} for name in exact_names}
    need(all(row["byte_identical"] for row in exact.values()), "Replay changed TeX/data bytes; attempt preserved")
    result_bytes = gzip.decompress((SCIENCE / "results.json.gz").read_bytes())
    (temp_root / "exact_results.json").write_bytes(result_bytes)
    audit_command = [str(PYTHON), str(DEST / "audit_optional_exposure_renderer.py"), "--render-root", str(render_output),
        "--result", str(temp_root / "exact_results.json"), "--audit", str(SCIENCE / "independent_audit.json"),
        "--output", str(temp_root / "renderer_audit.json")]
    checked = subprocess.run(audit_command, cwd=DEST, env=env, text=True, capture_output=True)
    write(temp_root / "audit_execution.json", {"command": audit_command, "returncode": checked.returncode,
        "stdout": checked.stdout, "stderr": checked.stderr})
    need(checked.returncode == 0, "Independent replay audit failed; attempt preserved")
    shutil.copyfile(temp_root / "renderer_audit.json", DEST / "replay_renderer_audit.json")
    replay = {"passed": True, "scope": "Pure render replay from adjacent exact compressed result; no inference or new scientific analysis",
        "results_sha256": hashlib.sha256(result_bytes).hexdigest(), "results_gzip_sha256": sha(SCIENCE / "results.json.gz"),
        "renderer_sha256": sha(DEST / "renderer_source.py"), "preserved_attempt_root": str(temp_root),
        "command": command, "exact_artifacts": exact,
        "figure_bytes": {name: {"reference_sha256": sha(DEST / name), "replayed_sha256": sha(render_output / name),
            "byte_identical": sha(DEST / name) == sha(render_output / name)} for name in ("optional_exposure_decomposition.png", "optional_exposure_decomposition.pdf")},
        "figure_byte_note": "TeX, plot coordinates and mechanism records must be exact; regenerated image/PDF metadata may differ. Preserved publication figures remain original v3 bytes.",
        "replay_audit_sha256": sha(DEST / "replay_renderer_audit.json"),
        "renderer_synthetic_tests": {"passed": 11, "seconds": 10.14, "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=adaptive-gns:adaptive-gns/scripts ../../work/venv/bin/python -m pytest -p no:cacheprovider research/tests/test_render_optional_exposure_findings.py -q", "cwd": str(REPO)},
        "attempt_inventory": inventory(temp_root)}
    write(DEST / "REPLAY_VERIFICATION.json", replay)
    (DEST / "README.md").write_text(README)
    need(inventory(SCIENCE) == science_before, "Compact scientific package changed during reporting curation")
    need(all(sha(Path(path)) == value["sha256"] for path, value in before.items()), "Reporting source changed during curation")
    need(all(sha(Path(path)) == expected for path, expected in document_hashes.items()), "Manuscript/report changed during curation")
    for name, item in attempts.items():
        need(inventory(WORK / name) == item["files"], "Prior renderer attempt changed")
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Derived reporting of final optional-exposure evidence, separate from frozen scientific package",
        "source_scientific_package": {"relative_path": "../optional_exposure_decomposition_20261006", "manifest_sha256": sha(SCIENCE / "PUBLICATION_MANIFEST.json"),
            "compressed_result_sha256": sha(SCIENCE / "results.json.gz"), "result_sha256": replay["results_sha256"]},
        "source_files_rechecked": before, "document_files_rechecked": document_hashes,
        "files": {}}
    for path in sorted(DEST.rglob("*")):
        if path.is_file():
            name = path.relative_to(DEST).as_posix()
            item = {"sha256": sha(path), "size_bytes": path.stat().st_size}
            if name in sources: item.update(source_path=str(sources[name]), source_sha256=before[str(sources[name])]["sha256"], encoding="original_bytes")
            manifest["files"][name] = item
    write(DEST / "MANIFEST.json", manifest)  # Always the final package write.
    verify(DEST)


README = '''# Optional-exposure reporting artifacts

This family contains the final v3 figure, manuscript inserts and independent reporting reviews derived from the adjacent `optional_exposure_decomposition_20261006` scientific package. It adds no model inference, training, new test population or scientific result. The underlying study uses the six original 100k checkpoints and the already inspected, no-self-loop observed-history predictions on 27 trajectories × 11 frames each. Native self-loop and 110k continuation results belong to separate studies.

The three paired comparisons are previous-observed-base-risk25 minus random25, that risk policy minus speed25, and speed25 minus random25. Positive values favor the right policy. Coordinate MSE differences/contributions and cost/alignment terms are displayed ×1,000; fractions are displayed as percentages. These are normalized acceleration coordinate quantities, with the original vector terms divided by two. Error difference equals cost difference minus alignment difference.

All four primary groups, all three both-covered degree subgroups, every comparison direction and all signed values remain in the scientific package and appendix. The primary groups partition particles by optional-edge incidence. The three degree subgroups partition only the both-covered group and must not be added again to the primary partition. Contributions use sum(mask × paired quantity)/N. Frames receive equal weight within a trajectory, trajectories equal weight within a seed, then exactly three seeds are summarized by mean and sample SD. Sample SD is not a confidence interval; the 90 figure coordinates are repeated model/condition measurements, not 90 independent samples. Empty-group conditional means remain undefined; no available-frame or available-seed substitution is introduced.

Exposure is defined by each policy. This descriptive partition does not identify a causal concentration effect, marginal edge value, independent confirmation, autonomous stability or inference speed. Message passing can affect particles with no directly incident optional edge. Adverse and null outcomes are retained in the adjacent exact scientific evidence.

## Included reporting and review evidence

- The seven original v3 files are preserved byte for byte: `optional_exposure_main.tex`, `optional_exposure_appendix.tex`, `optional_exposure_decomposition.png`, `optional_exposure_decomposition.pdf`, `plot_coordinates.json`, `mechanism_findings.json`, and `render_manifest.json`.
- `renderer_source.py` and `test_render_optional_exposure_findings.py` preserve final renderer/test source. `renderer_review_v3.json` and its independent audit script verify 92 mean±SD cells, 90 seed coordinates and six mechanism rows. The figure was visually inspected as recorded by the original review.
- `report_review_v2.json` and its audit script check 14 mean±SD cells and 24 seed sign/ordering claims on page 16 of the separate evidence-report PDF. `manuscript_preservation_review.json` and its audit script establish that both title fields, the abstract and 19 prior tables are unchanged, with only two intended insertions and three new tables.
- `document_verification.json` records successful native LaTeX compilation of manuscript SHA256 `745ea370d8a13826cf431b294514acee611d07b96be1e799502cf7184175aeef`, the eight-page main-text assertion, 47 unique labels and 12 resolved citation keys. This is distinct from the 16-page evidence-report PDF and its numeric/visual checks. The native compiler did not export a manuscript PDF; author visual inspection and registration, rights and anonymity verification remain outstanding. This package does not claim submission readiness.
- `compact_publication_verification.json` links the scientific package's byte/coverage check. `superseded_reviews/` retains earlier review records. `render_attempt_inventory.json` hashes all three original renderer attempts; v1/v2 outputs remain unchanged locally and v3 alone supplies the final presentation.
- `REPLAY_VERIFICATION.json` and `replay_renderer_audit.json` record a fresh replay from the adjacent exact gzip result. Both TeX inserts, all plot coordinates and the mechanism rows matched byte for byte. The replay independently passed the same 92-cell/90-coordinate audit. Its complete temporary output directory is preserved and inventoried. Original figures remain the reviewed v3 bytes; regenerated PNG/PDF metadata can differ.

## Reproduce the reporting

From this directory, with Python, NumPy and Matplotlib installed, choose an output directory that does not exist:

```sh
python renderer_source.py \\
  --results ../optional_exposure_decomposition_20261006/results.json.gz \\
  --audit ../optional_exposure_decomposition_20261006/independent_audit.json \\
  --output-dir /path/to/new-reporting-directory
```

The renderer checks the independently audited full-result hash and exactly three required seed values. It refuses to overwrite an existing output directory. Both TeX inserts and numerical JSON artifacts should reproduce exactly. The original manifest identifies the plain input result; replay from gzip has the same decompressed result hash and a different input-file hash. PNG/PDF bytes may differ because of output metadata or library versions.

Run `python3 verify_reporting.py` here to verify every packaged byte and adjacent scientific-result linkage without inference. `MANIFEST.json` is written last and lists every included file, original copied-source hash and rechecked manuscript/report identities. Report/manuscript PDFs and raw particle arrays are not embedded. Review scripts preserve their original CLI contracts and require the corresponding external report/manuscript or uncompressed exact result; renderer synthetic tests run in the repository with its frozen synthetic fixtures.
'''


if __name__ == "__main__":
    main()
