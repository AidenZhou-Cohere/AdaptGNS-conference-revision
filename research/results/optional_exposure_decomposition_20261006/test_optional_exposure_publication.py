"""Small synthetic byte-preservation and fail-closed publication tests; no inference."""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

try:
    import curate_publication as c
    import verify_publication as v
except ImportError:
    import curate_optional_exposure_publication as c
    import verify_optional_exposure_publication as v


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    repo = tmp_path / "repo"
    sources = [("research/analyze_optional_exposure_decomposition.py", "SOURCE_HASH"),
               ("research/protocols/optional_exposure_decomposition_20261006.md", "PROTOCOL_HASH"),
               ("research/tests/test_optional_exposure_decomposition.py", "TEST_HASH")]
    for relative, key in sources:
        path = repo / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic " + relative + "\n")
        monkeypatch.setattr(c, key, v.sha(path))
    write(raw / "all_inputs_sha256.json", {"synthetic_input": "hash"})
    write(raw / "input_identity.json", {"scope": "synthetic"})
    (raw / "report.md").write_text("Synthetic signed adverse outcome and undefined conditional.\n")
    runs = {}
    for model in v.MODELS:
        frames = []
        for source, target in sorted(v.EXPECTED):
            relative = f"frames/{model}/trajectory_{source:06d}_target_{target:04d}.json"
            derived = relative.removesuffix(".json") + ".npz"
            path = raw / derived; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic derived bytes; no vectors")
            record = {"status": "complete", "source_index": source, "target_frame": target,
                "metrics": {"signed_error": -1.2, "empty_conditional": None},
                "undefined_reasons": {"empty_conditional": "empty_group"},
                "pairs": {pair: {"status": "complete", "groups": {group: {"count": 0, "contribution": 0, "conditional": None}
                    for group in v.PRIMARY + v.BOTH}} for pair in v.PAIR_NAMES},
                "derived_array_file": derived, "derived_array_sha256": v.sha(path)}
            write(raw / relative, record)
            frames.append({"source_index": source, "target_frame": target, "status": "complete",
                "record_file": relative, "record_sha256": v.sha(raw / relative)})
        runs[model] = {"eligible": True, "issues": [], "frames": frames,
                      "aggregate": {"trajectories": {str(i): {"empty": None, "signed": -1.2} for i in range(3,30)}, "metric": None}}
    result = {"state": "complete", "required_models": 6, "required_frames_per_model": 297, "required_total_frames": 1782,
        "scope": "post_inspection_original_100k_optional_exposure_decomposition", "comparison_order": v.PAIR_MAP,
        "primary_groups": v.PRIMARY, "supplementary_both_subgroups": v.BOTH, "runs": runs,
        "objectives": {"faithful": {"signed": -1.2, "empty": None}, "nll": {"signed": 2.0}},
        "consistency_errors": [], "failures": [], "all_inputs_sha256": v.sha(raw / "all_inputs_sha256.json"),
        "source_sha256": {str(repo / sources[0][0]): c.SOURCE_HASH}, "protocol_sha256": c.PROTOCOL_HASH, "analysis_wall_seconds": 0.25}
    write(raw / "results.json", result)
    write(raw / "status.json", {"state": "complete", "results_sha256": v.sha(raw / "results.json"), "report_sha256": v.sha(raw / "report.md")})
    audit_dir = tmp_path / "audit"; audit_dir.mkdir()
    audit_source = tmp_path / "audit.py"; audit_source.write_text("# synthetic standalone audit\n")
    audit_tests = tmp_path / "audit_tests.py"; audit_tests.write_text("# synthetic audit selftests\n")
    audit = audit_dir / "attempt_pass.json"
    write(audit, {"passed": True, "results_sha256": v.sha(raw / "results.json"), "audited_frames": 1782,
        "source_sha256": v.sha(audit_source), "analysis_source_sha256": c.SOURCE_HASH, "analysis_protocol_sha256": c.PROTOCOL_HASH, "wall_seconds": 0.1})
    write(audit_dir / "attempt_failed.json", {"passed": False, "error": "Preserved synthetic unsuccessful audit"})
    review = tmp_path / "review.json"
    write(review, {"state": "pass", "source_sha256": {relative: getattr(c, key) for relative, key in sources}})
    return argparse.Namespace(raw=raw, destination=tmp_path / "package", repo=repo, audit=audit,
        audit_source=audit_source, audit_tests=audit_tests, review=review, render_dir=None)


def test_complete_package_preserves_nulls_signed_outcomes_and_failed_audit(synthetic):
    result = c.curator(synthetic)
    assert result["passed"] and result["original_frame_records"] == result["frame_null_metrics_preserved"] == 1782
    assert gzip.decompress((synthetic.destination / "results.json.gz").read_bytes()) == (synthetic.raw / "results.json").read_bytes()
    assert (synthetic.destination / "audit_attempts/attempt_failed.json").read_bytes() == (synthetic.audit.parent / "attempt_failed.json").read_bytes()
    compact = json.loads((synthetic.destination / "results.json").read_text())
    assert compact["objectives"]["faithful"] == {"signed": -1.2, "empty": None}
    assert all("trajectories" not in run["aggregate"] for run in compact["runs"].values())
    with pytest.raises(ValueError, match="Preserve existing"):
        c.curator(synthetic)
    path = synthetic.destination / "independent_audit_tests.py"
    path.write_text("bad")
    with pytest.raises(ValueError, match="Packaged bytes differ"):
        v.verify(synthetic.destination)


def test_incomplete_or_mismatched_audit_refuses_before_destination(synthetic):
    audit = json.loads(synthetic.audit.read_text()); audit["passed"] = False
    write(synthetic.audit, audit)
    with pytest.raises(ValueError, match="audit must pass"):
        c.curator(synthetic)
    assert not synthetic.destination.exists()


def test_missing_frame_is_not_available_case_publication(synthetic):
    p = synthetic.raw / "results.json"; result = json.loads(p.read_text())
    result["runs"][v.MODELS[0]]["frames"].pop()
    with pytest.raises(ValueError, match="Wrong frame coverage"):
        v.validate_result(result)


def test_deterministic_compression_and_archive(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    for name, value in [("a.json", {"undefined_reasons": {"metric": "empty"}}), ("b.json", {"undefined_reasons": {}})]:
        write(raw / name, value)
    a = tmp_path / "a.gz"; b = tmp_path / "b.gz"
    c.exact_gzip(raw / "a.json", a); c.exact_gzip(raw / "a.json", b)
    assert a.read_bytes() == b.read_bytes()
    a = tmp_path / "a.tar.gz"; b = tmp_path / "b.tar.gz"
    index = c.exact_archive(raw, ["b.json", "a.json"], a)
    c.exact_archive(raw, ["a.json", "b.json"], b)
    assert a.read_bytes() == b.read_bytes() and index["null_metric_count"] == 1
    assert dict(v.iter_archived_records(a)) == {name: (raw/name).read_bytes() for name in ("a.json", "b.json")}


@pytest.mark.parametrize("name", ["../bad.json", "/absolute.json", "a/../bad", "a//b", "./bad", "", None])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError):
        v.safe_name(name)


@pytest.mark.parametrize("name,duplicate", [("../escape.json", False), ("a.json", True)])
def test_archive_path_or_duplicate_rejected(tmp_path, name, duplicate):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        for i in range(2 if duplicate else 1):
            header = tarfile.TarInfo(name); header.size = 2
            stream.addfile(header, io.BytesIO(b"{}"))
    with pytest.raises(ValueError):
        list(v.iter_archived_records(path))


def test_snapshot_does_not_follow_symlinks(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "outside").symlink_to(tmp_path / "another")
    with pytest.raises(ValueError, match="symlinks"):
        c.snapshot(raw)
