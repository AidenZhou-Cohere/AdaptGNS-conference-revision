"""Verify reporting package and its adjacent exact scientific evidence, without inference."""
import gzip
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(value, message):
    if not value:
        raise ValueError(message)


def verify(root):
    root = Path(root).resolve()
    manifest = read(root / "MANIFEST.json")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    need(not any(p.is_symlink() for p in root.rglob("*")), "Unexpected symlink")
    need(files == set(manifest["files"]) | {"MANIFEST.json"}, "Reporting file population differs")
    for name, identity in manifest["files"].items():
        need(not name.startswith("/") and ".." not in Path(name).parts, "Unsafe file name")
        path = root / name
        need(path.stat().st_size == identity["size_bytes"] and sha(path) == identity["sha256"], "Packaged bytes differ: " + name)
        if "source_sha256" in identity:
            need(identity["source_sha256"] == identity["sha256"], "Copied source relationship differs")
    scientific = root.parent / "optional_exposure_decomposition_20261006"
    source = manifest["source_scientific_package"]
    need(sha(scientific / "PUBLICATION_MANIFEST.json") == source["manifest_sha256"], "Adjacent scientific package differs")
    need(sha(scientific / "results.json.gz") == source["compressed_result_sha256"], "Exact result gzip differs")
    exact_result = gzip.decompress((scientific / "results.json.gz").read_bytes())
    result_sha = hashlib.sha256(exact_result).hexdigest()
    need(result_sha == source["result_sha256"], "Decompressed scientific result differs")
    independent = read(scientific / "independent_audit.json")
    need(independent["passed"] and independent["audited_frames"] == 1782 and independent["results_sha256"] == result_sha, "Scientific audit differs")
    render = read(root / "render_manifest.json")
    need(render["results_sha256"] == result_sha and render["independent_audit_sha256"] == sha(scientific / "independent_audit.json"), "Render source differs")
    need(render["renderer_sha256"] == sha(root / "renderer_source.py"), "Renderer source differs")
    for name, expected in render["outputs"].items():
        need(sha(root / name) == expected, "Original rendered bytes differ")
    renderer_review = read(root / "renderer_review_v3.json")
    need(renderer_review["passed"] and renderer_review["results_sha256"] == result_sha and renderer_review["render_manifest_sha256"] == sha(root / "render_manifest.json"), "Renderer review differs")
    need(renderer_review["review_source_sha256"] == sha(root / "audit_optional_exposure_renderer.py"), "Renderer audit source differs")
    need(sum(renderer_review["cells"].values()) == 92 and renderer_review["seed_coordinates"] == 90, "Renderer review coverage differs")
    report = read(root / "report_review_v2.json")
    need(report["passed"] and report["results_sha256"] == result_sha and report["mean_sd_cells"] == 14, "Report review differs")
    need(report["review_source_sha256"] == sha(root / "audit_optional_exposure_report.py"), "Report audit source differs")
    preservation = read(root / "manuscript_preservation_review.json")
    need(preservation["passed"] and preservation["final_v3_inserts_present_exactly"] and preservation["only_insertions"], "Preservation review differs")
    need(preservation["review_source_sha256"] == sha(root / "audit_optional_document_preservation.py"), "Preservation audit source differs")
    need(preservation["title_fields_unchanged"] == 2 and preservation["abstract_unchanged"] and preservation["preexisting_tables_unchanged"] == 19, "Preservation coverage differs")
    document = read(root / "document_verification.json")
    need(document["results_sha256"] == result_sha and document["report_sha256"] == report["pdf_sha256"], "Document/result linkage differs")
    need(document["preservation_review_sha256"] == sha(root / "manuscript_preservation_review.json")
         and document["report_review_v2_sha256"] == sha(root / "report_review_v2.json")
         and document["renderer_review_v3_sha256"] == sha(root / "renderer_review_v3.json"), "Document review linkage differs")
    need(document["publication_verification_sha256"] == sha(root / "compact_publication_verification.json"), "Publication review linkage differs")
    compact = read(root / "compact_publication_verification.json")
    need(compact["publication_manifest_sha256"] == source["manifest_sha256"] and compact["source_result_sha256"] == result_sha, "Compact package review differs")
    replay = read(root / "REPLAY_VERIFICATION.json")
    need(replay["passed"] and replay["results_sha256"] == result_sha and replay["renderer_sha256"] == render["renderer_sha256"], "Replay identity differs")
    need(set(replay["exact_artifacts"]) == {"optional_exposure_main.tex", "optional_exposure_appendix.tex", "plot_coordinates.json", "mechanism_findings.json"}, "Replay artifact set differs")
    for name, record in replay["exact_artifacts"].items():
        need(record["byte_identical"] and record["reference_sha256"] == record["replayed_sha256"] == sha(root / name), "Replay comparison differs")
    replay_audit = read(root / "replay_renderer_audit.json")
    need(replay_audit["passed"] and replay_audit["results_sha256"] == result_sha and sum(replay_audit["cells"].values()) == 92 and replay_audit["seed_coordinates"] == 90, "Replay numeric audit differs")
    need(sha(root / "replay_renderer_audit.json") == replay["replay_audit_sha256"], "Replay audit identity differs")
    value = {"passed": True, "manifest_sha256": sha(root / "MANIFEST.json"), "packaged_files": len(manifest["files"]),
             "result_sha256": result_sha, "exact_replayed_artifacts": len(replay["exact_artifacts"]),
             "verified_renderer_cells": 92, "verified_seed_coordinates": 90, "report_mean_sd_cells": 14,
             "native_compilation_record": document["native_latex"]}
    print(json.dumps(value, indent=2))
    return value


if __name__ == "__main__":
    verify(Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent)
