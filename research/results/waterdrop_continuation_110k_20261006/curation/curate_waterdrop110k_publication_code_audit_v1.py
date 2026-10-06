"""Copy reviewed WaterDrop evidence. This never runs scientific code or changes inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = ROOT / "work/deadline_research_20261005"
PLAN = TOP / "waterdrop110k_publication_copy_plan_code_audit_v2.json"
PLAN_SHA = "56b5af45664030376707bb4990ed71b09910a120673c5e9abac421bf05b8a5e7"
STAGE = TOP / "waterdrop110k_publication_staging_code_audit_v1"
OUTPUT = ROOT / "outputs/AdaptGNS/research/results/waterdrop_continuation_110k_20261006"
REPORT = ROOT / "outputs/waterdrop_continuation_110k_20261006.md"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, indent=2, allow_nan=False)
            f.write("\n")


def copy_checked(src, dst, expected):
    assert src.is_file() and not src.is_symlink(), src
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with src.open("rb") as incoming, dst.open("xb") as outgoing:
        while block := incoming.read(8 * 1024 * 1024):
            h.update(block)
            outgoing.write(block)
    assert h.hexdigest() == expected, (src, "source changed or wrong hash")
    assert sha(dst) == expected, (dst, "copy mismatch")


def manifest(directory):
    return {str(p.relative_to(directory)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p != directory / "manifest.json"}


def verify_sources(plan):
    for x in plan["required_fixed_files"]:
        p = ROOT / x["source"]
        assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size == x["bytes"] and sha(p) == x["expected_sha256"], p
    external = plan["external_artifact"]
    p = ROOT / external["source"]
    assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size == external["bytes"] and sha(p) == external["compressed_sha256"]


def validate_package(stage, plan):
    for x in plan["required_fixed_files"]:
        p = stage / x["destination"]
        assert p.stat().st_size == x["bytes"] and sha(p) == x["expected_sha256"]
    summary = json.loads((stage / "summary/result.json").read_text())
    assert summary["state"] == "complete" and summary["source_and_input_reverified_after_analysis"]
    metric_files = [v for pop in summary["populations"].values() for v in pop["metrics"].values()]
    assert len(metric_files) == 91
    for v in metric_files:
        assert sha(stage / "summary" / v["path"]) == v["sha256"]
    assert len(list((stage / "summary").iterdir())) == 93
    assert not (stage / "summary/audited_records.json.gz").exists()
    assert summary["coverage"] == {"endpoints": 6, "jobs": 12, "observed_frames": 2550,
                                    "observed_policy_slots": 12750, "autonomous_outcomes": 810}
    assert all(j["failed_records"] == 0 for j in summary["jobs"])
    audit = json.loads((stage / "audit/audit.json").read_text())
    assert audit["passed"] and audit["checks"] == 12101908
    native = json.loads((stage / "presentation/integration/native_compile_v1.json").read_text())
    assert native["kind"] == "success"
    assert native["source_sha256"] == sha(stage / "presentation/revised_manuscript.tex")
    assert native["main_text_eight_page_limit_assertion_passed"]
    # These restrictions concern included payloads, not unchanged scientific path references.
    for p in stage.rglob("*"):
        if p.is_file():
            assert p.suffix not in {".pt", ".pth", ".npz", ".npy", ".pkl", ".tfrecord"}, p
            assert all(part not in {".aws", ".ssh", "credentials"} for part in p.parts), p
    return {"coverage": summary["coverage"], "metric_detail_files": len(metric_files),
            "independent_audit_checks": audit["checks"],
            "native_compile_source_sha256": native["source_sha256"]}


def stage(plan):
    assert not OUTPUT.exists() and not REPORT.exists()
    verify_sources(plan)
    STAGE.mkdir(exist_ok=False)
    for x in plan["required_fixed_files"]:
        copy_checked(ROOT / x["source"], STAGE / x["destination"], x["expected_sha256"])
    for p in (PLAN, TOP / "waterdrop110k_publication_copy_plan_code_audit_v1.json", Path(__file__)):
        copy_checked(p, STAGE / "curation" / p.name, sha(p))
    write(STAGE / "copy_provenance.json", {
        "plan_path": str(PLAN.relative_to(ROOT)), "plan_sha256": PLAN_SHA,
        "copy_method": "Byte-for-byte streaming copies; original compression unchanged. No scientific code run.",
        "files": plan["required_fixed_files"]})
    external = dict(plan["external_artifact"])
    external.update({
        "missing_package_reference": "summary/audited_records.json.gz",
        "reference_owner": "summary/result.json::audited_records",
        "compressed_hash_reverified_at_curation": True,
        "uncompressed_hash_scope": "Recorded hash from the completed strict summary; curation does not decompress or regenerate this archive.",
        "input_scope": "Combined derived records over every planned observed and autonomous endpoint/policy/unit. Underlying checkpoint, dataset, raw observed arrays and autonomous trace sources remain external, with exact original identities in summary/result.json::input_files_sha256.",
        "reconstruction_scope": "Restoring the exact existing local archive at its recorded relative location satisfies the unchanged summary reference. Reconstructing from raw sources additionally requires all frozen scientific source, protocol, checkpoint/data and evaluation inputs bound by the original summary; the Git package alone is insufficient. Curation performs no reconstruction or rerun."})
    write(STAGE / "external_artifacts.json", {"schema": "waterdrop110k_external_artifacts_v1", "artifacts": [external],
        "uniform_omission": "One combined record payload is omitted for all cohorts and policies. No seed, outcome, failure, metric family or contrast is selectively removed."})
    summary = json.loads((STAGE / "summary/result.json").read_text())
    counts = {}
    for path in summary["input_files_sha256"]:
        suffix = Path(path).suffix or "<none>"
        counts[suffix] = counts.get(suffix, 0) + 1
    write(STAGE / "omissions.json", {"standalone_raw_reproduction_bundle": False,
        "included": "Original result/status, all91 per-metric detail archives, original independent audit, every required cohort/policy/seed/failure statistic, focused frozen code/protocols, report and reviewed manuscript.",
        "external_combined_records": "external_artifacts.json",
        "raw_inputs": {"copied": False, "hash_reference": "summary/result.json::input_files_sha256",
                       "original_hash_reference_count": len(summary["input_files_sha256"]),
                       "extension_counts_of_original_references": counts,
                       "scope": "Original models, source datasets, per-job prediction arrays, trace arrays and auxiliary committed input artifacts are not duplicated."},
        "native_process_inventories_credentials_ssh_configuration": "Not packaged.",
        "goop_comparison_source": {"path": "../goop2d_graph_exposure_100k_20261006/paired_scalar_summary.json.gz",
            "sha256": "bb3c3f54cdeaf2b1198f49b8437dc711d80ce5e8347ab70363558ab515852978",
            "scope": "Neighboring published source for comparative Goop values in report/claim mappings; not duplicated."},
        "manuscript_builder_scope": "Current standalone revised_manuscript.tex is included. The original builder and numerical tools retain workspace-relative/absolute dependencies; their copied presence does not imply a self-contained runnable raw experiment."})
    write(STAGE / "operational_failure_history.json", {"scientific_retries_or_reruns": 0,
        "events": [
            {"scope": "session observation", "failure": "Child could not poll the root's original session registry.",
             "resolution": "Root supplied the original external exit receipt; no duplicate evaluation or process signal."},
            {"scope": "ancillary summary metadata print", "failure": "TypeError after successful completed-analysis gate serialization: len() was applied to integer audit check count.",
             "resolution": "All gate assertions and bound hashes were reread successfully; no scientific rerun."},
            {"scope": "ancillary abstract-append metadata check", "failure": "AssertionError from assuming one aistatstitle invocation rather than main and supplementary titles.",
             "resolution": "Both title macros were recorded; unchanged whole manuscript hashes and abstract were checked. No manuscript writes."}],
        "resolved_editorial_issue": "The Goop every-seed risk disadvantage was explicitly qualified as observed test; validation is not included in that claim.",
        "native_process_inventory_policy": "Only semantic failure history is copied; native PIDs, commands, snapshots and credentials are not included."})
    write(STAGE / "README.md", """# WaterDrop: faithful paired 110k graph-exposure continuation

Three fixed faithful 100k parents each receive paired 10k base-only and mixed-graph continuations, with inherited optimizer state and matched frame/noise schedules. The six 110k endpoints evaluate five policies. This is exploratory after test inspection, conditional on those parents; it is not training from initialization, NLL evidence, or independent confirmation.

All 2,550 observed histories (128 validation +297 test per endpoint), 12,750 observed policy slots, and 810 autonomous H995 outcomes are retained. Every scientific failure count is zero. All policies and three seed values remain visible; statistics use the frozen equal-trajectory hierarchy, matched-unit differences, and sample SD across three seeds. Required undefined values propagate to null rather than survivor averages.

Mixed exposure reverses the observed-test risk-minus-random ordering in all three seeds. Validation improves the relative gap but does not establish a mixed-arm average advantage. Under autonomous random25, mix-minus-base H995 MSE is -0.00599935 ±0.00556525, improving in every seed. The autonomous risk-minus-random interaction is +0.00284351 ±0.00466257, with mixed seed signs. Dense, speed and cached risk also improve under mix in each seed, but dense remains worse than base within each arm and speed has the lowest mean error. Computational completion does not imply physical validity: large box excursions remain, and recorded duration/edge counts increase under mix. Fixed policy order and differing evolved geometries do not establish causal speedup.

## Contents and reproducibility scope

`summary/` preserves the exact original `result.json`, `status.json`, and all **91** metric detail archives. Each metric retains every required unit/trajectory/seed statistic and null reason. `audit/audit.json` is the original completed complementary independent audit, with 12,101,908 checks. It checks observed-array/autonomous-scalar arithmetic, paired hierarchy and trace hashes; it does not independently admit checkpoint/source data, replay all guards/graphs, or regenerate unsaved predictions. The strict summary provides complementary validation.

**This is not a standalone raw reproduction bundle.** The one 368,626,721-byte combined derived archive, `audited_records.json.gz`, remains unchanged at `work/continuation-summary-20261006-v1/audited_records.json.gz` relative to the original project root. `external_artifacts.json` supplies exact compressed/uncompressed hashes, size, unchanged summary-reference mapping, and reconstruction/input scope. Its omission is uniform across all cohorts and policies; no failed or unfavorable result is dropped. All 91 per-metric details and complete summary statistics remain here. Original result references are preserved rather than rewritten.

Raw checkpoints, datasets, per-job prediction arrays and trace arrays remain external at the immutable paths/hashes bound by `summary/result.json::input_files_sha256`. Restoring the existing combined archive satisfies its stored reference; reproducing the original raw validation additionally requires those inputs and the frozen dependency environment. `omissions.json` records these limits. Credentials, SSH configuration and native process inventories are excluded. `operational_failure_history.json` preserves the semantic reporting/session failures without those inventories. No scientific stage was rerun for publication.

`report.md` is an exact copy of the complete reviewed scientific report; the same bytes are also published as `outputs/waterdrop_continuation_110k_20261006.md`. `evidence/` contains exact source mappings for all2,002 statistic objects and133 focused claims, lineage/protocol identities and candidate fragments. Comparative Goop claims point to the neighboring Goop result package; its distinct from-initialization H395 lineage and guard failures remain separate.

`presentation/` snapshots the integrated standalone manuscript, body, main experiment source and original builder, with pre-integration sources and applied/native-compile receipts. The native compile succeeded for the recorded source and its eight-page main-text assertion passed. The title, abstract, 66 prior labels and old appendix bytes are preserved. Exported PDF visual inspection remains outside these source checks. The builder retains its original workspace dependencies; the copied standalone manuscript is the compilation artifact. `reviews/` preserves independent interpretation, candidate, applied-text and copy-review scopes. These reviews do not certify submission readiness.

`source/` copies the focused frozen scientific/protocol dependency closure and independent audit. `curation/` preserves both copy plans and the workspace-specific byte-copy helper. `copy_provenance.json` maps every original input to its packaged byte identity, and `manifest.json` binds all package files except itself. The initial full-archive copy plan remains historical; the revised compact plan supersedes it. No LFS, release upload, archive splitting, commit or push is performed here.
""")
    checked = validate_package(STAGE, plan)
    verify_sources(plan)
    write(STAGE / "publication_verification.json", {"schema": "waterdrop110k_publication_verification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "byte_verified_pending_independent_copy_review",
        "fixed_files_copied": len(plan["required_fixed_files"]), "fixed_copy_bytes": plan["fixed_copy_bytes"],
        "all_source_hashes_reverified_after_copy": True, "external_combined_archive_retained_and_hash_verified": True,
        "scientific_stages_rerun": False, "checks": checked})
    write(STAGE / "manifest.json", manifest(STAGE))
    print(json.dumps({"stage": str(STAGE), "manifest_sha256": sha(STAGE / "manifest.json"),
                      "files": len(manifest(STAGE)), "bytes": sum(x["bytes"] for x in manifest(STAGE).values())}))


def publish(plan, review_path, review_hash):
    assert not OUTPUT.exists() and not REPORT.exists()
    assert sha(review_path) == review_hash
    # The reviewer writes a separate immutable note; caller supplies its returned hash.
    assert "PASS" in review_path.read_text()
    original_manifest = json.loads((STAGE / "manifest.json").read_text())
    assert original_manifest == manifest(STAGE)
    validate_package(STAGE, plan)
    verify_sources(plan)
    copy_checked(review_path, STAGE / "reviews" / review_path.name, review_hash)
    for name in ("manifest.json", "publication_verification.json"):
        copy_checked(STAGE / name, STAGE / "curation" / ("before_copy_review_" + name), sha(STAGE / name))
    verification = json.loads((STAGE / "publication_verification.json").read_text())
    verification.update({"status": "published_byte_verified_and_independently_copy_reviewed",
        "publication_utc": datetime.now(timezone.utc).isoformat(),
        "independent_copy_review": {"path": "reviews/" + review_path.name, "sha256": review_hash},
        "all_source_hashes_reverified_before_publication": True})
    (STAGE / "publication_verification.json").unlink()
    write(STAGE / "publication_verification.json", verification)
    (STAGE / "manifest.json").unlink()
    write(STAGE / "manifest.json", manifest(STAGE))
    assert json.loads((STAGE / "manifest.json").read_text()) == manifest(STAGE)
    expected_report = next(x["expected_sha256"] for x in plan["required_fixed_files"] if x["destination"] == "report.md")
    copy_checked(STAGE / "report.md", REPORT, expected_report)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    STAGE.rename(OUTPUT)
    assert json.loads((OUTPUT / "manifest.json").read_text()) == manifest(OUTPUT)
    assert sha(REPORT) == expected_report
    write(TOP / "waterdrop110k_publication_receipt_code_audit_v1.json", {
        "package": str(OUTPUT.relative_to(ROOT)), "manifest_sha256": sha(OUTPUT / "manifest.json"),
        "external_report": str(REPORT.relative_to(ROOT)), "report_sha256": expected_report,
        "package_files_excluding_manifest": len(manifest(OUTPUT)),
        "package_bytes_excluding_manifest": sum(x["bytes"] for x in manifest(OUTPUT).values()),
        "external_archive_retained_and_reverified": plan["external_artifact"],
        "independent_copy_review_sha256": review_hash, "scientific_reruns": 0,
        "no_git_operations_or_external_uploads": True})
    print(json.dumps({"published": str(OUTPUT), "manifest_sha256": sha(OUTPUT / "manifest.json"),
                      "report": str(REPORT), "report_sha256": expected_report}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["stage", "publish"])
    parser.add_argument("--review", type=Path)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    assert sha(PLAN) == PLAN_SHA
    plan = json.loads(PLAN.read_text())
    if args.phase == "stage":
        stage(plan)
    else:
        assert args.review and args.review_sha256
        publish(plan, args.review, args.review_sha256)
