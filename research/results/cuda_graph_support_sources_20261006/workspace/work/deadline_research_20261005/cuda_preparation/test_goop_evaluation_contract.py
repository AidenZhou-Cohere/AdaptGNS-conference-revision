"""Tiny synthetic auxiliary files and scalar evidence; no acquired arrays/model."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("_contract_test_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence_fixture(tmp_path):
    fixture = load("test_evaluate_goop_graph_support_final")
    bench, manifest, admission, structural, args, _ = fixture.split_fixture(tmp_path, "test")
    contract = bench.load_data_contract()
    for name in ("acquisition_report", "context_semantics", "auxiliary_report", "cross_split_audit"):
        setattr(args, name, tmp_path / (name + ".json"))
    args.context_semantics.write_bytes((ROOT / "goop_context_semantics_review.json").read_bytes())
    source_dir = tmp_path / "goop_context_semantics_sources"
    source_dir.mkdir()
    for name in contract.CONTEXT_SOURCES:
        (source_dir / name).write_bytes((ROOT / "goop_context_semantics_sources" / name).read_bytes())
    (tmp_path / "test").mkdir()
    rows = []
    for record in manifest["records"]:
        descriptor = record["step_context"]
        path = tmp_path / descriptor["path"]
        values = np.zeros((401, 1), dtype="<f4")
        values[0] = np.nan
        np.save(path, values)
        descriptor.update(size_bytes=path.stat().st_size, sha256=contract.sha(path))
        rows.append({"id": record["id"], "source_index": record["source_index"], "sha256": descriptor["sha256"],
            "shape": [401, 1], "dtype": "<f4", "elements": 401, "finite_count": 400, "nan_count": 1,
            "positive_infinity_count": 0, "negative_infinity_count": 0, "unique_float32_bits_hex": ["00000000", "7fc00000"]})
    args.manifest.write_text(json.dumps(manifest))
    admission["manifest_sha256"] = contract.sha(args.manifest)
    source = manifest["source"]
    receipt = {"status": "complete", "dataset": "Goop", "source_family": "official_gns_tfrecord", "files": [{
        "name": "test.tfrecord", "saved_name": "test.tfrecord", "status": "complete",
        "url": "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop/test.tfrecord",
        "received_bytes": source["size_bytes"], "sha256": source["sha256"], "generation": source["generation"],
        "crc32c_base64": source["crc32c_base64"], "crc32c_verified": True}]}
    args.acquisition_report.write_text(json.dumps(receipt))
    admission["acquisition_report_sha256"] = contract.sha(args.acquisition_report)
    census = {"schema": "adaptgns_goop_auxiliary_census_v1", "status": "all_preserved_auxiliary_bytes_verified", "test_accessed": True,
        "splits": {"test": {"manifest_sha256": admission["manifest_sha256"], "record_count": 30, "context_mean_absent": True,
            "context_std_absent": True, "all_context_descriptors_and_bytes_verified": True, "records": rows}}}
    args.auxiliary_report.write_text(json.dumps(census))
    admission["auxiliary_report_sha256"] = contract.sha(args.auxiliary_report)
    audit = {"schema": "adaptgns_goop_all_split_integrity_audit_v1", "issued_by": "root", "status": "all_required_splits_verified",
        "duplicate_pairs": [], "manifest_sha256": {"train": contract.TRAIN_MANIFEST_SHA, "valid": contract.VALID_MANIFEST_SHA,
            "test": admission["manifest_sha256"]}, "record_counts": {"train": 1000, "valid": 30, "test": 30},
        "definition": "exact stored position/type dtype,shape,bytes; auxiliaries preserved separately"}
    args.cross_split_audit.write_text(json.dumps(audit))
    admission["cross_split_audit_sha256"] = contract.sha(args.cross_split_audit)
    helpers = SimpleNamespace(np=np, data_loader=SimpleNamespace(_manifest_array_path=lambda root, d: root / d["path"]))
    return contract, args, manifest, admission, helpers, census, audit


def test_every_excluded_auxiliary_and_root_overlap_evidence_is_bound(tmp_path):
    c, args, manifest, admission, helpers, _, _ = evidence_fixture(tmp_path)
    pins = c.verify_evidence(args, manifest, admission, helpers)
    assert len(pins) == 38  # 30 auxiliaries, 3 reports, 4 context sources, 1 split audit.
    assert all(c.sha(path) == pin for path, pin in pins.items())
    path = tmp_path / manifest["records"][0]["step_context"]["path"]
    values = np.load(path, allow_pickle=False)
    values.view("<u4")[0, 0] = 0x7fc00001
    np.save(path, values)
    with pytest.raises(ValueError, match="bytes differ"):
        c.verify_evidence(args, manifest, admission, helpers)


@pytest.mark.parametrize("case", ["missing_census", "wrong_nan_bits", "wrong_source", "partial_overlap", "duplicate_overlap", "not_root"])
def test_wrong_evidence_cannot_pass_even_when_its_file_hash_matches(tmp_path, case):
    c, args, manifest, admission, helpers, census, audit = evidence_fixture(tmp_path)
    if case == "missing_census":
        census["splits"]["test"]["records"].pop()
    elif case == "wrong_nan_bits":
        census["splits"]["test"]["records"][0]["unique_float32_bits_hex"][-1] = "7fc00001"
    elif case == "wrong_source":
        receipt = json.loads(args.acquisition_report.read_text())
        receipt["files"][0]["url"] = "https://example.invalid/test.tfrecord"
        args.acquisition_report.write_text(json.dumps(receipt))
        admission["acquisition_report_sha256"] = c.sha(args.acquisition_report)
    elif case == "partial_overlap":
        audit["manifest_sha256"].pop("train")
    elif case == "duplicate_overlap":
        audit["duplicate_pairs"] = [["train:0", "test:0"]]
    else:
        audit["issued_by"] = "unreviewed"
    args.auxiliary_report.write_text(json.dumps(census))
    admission["auxiliary_report_sha256"] = c.sha(args.auxiliary_report)
    args.cross_split_audit.write_text(json.dumps(audit))
    admission["cross_split_audit_sha256"] = c.sha(args.cross_split_audit)
    with pytest.raises(ValueError):
        c.verify_evidence(args, manifest, admission, helpers)


def test_core_numerical_functions_are_unchanged_from_reviewed_sand_sources():
    c = load("goop_evaluation_contract")
    pairs = [("benchmark", "benchmark_sand_graph_support_rollout.py", "benchmark_goop_graph_support_rollout.py",
              "8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13",
              ("select_cases", "configure_cuda", "tree_equal", "prepare_model", "summarize_cases", "run_cases")),
             ("diagnostic", "evaluate_sand_graph_support_final.py", "evaluate_goop_graph_support_final.py",
              "952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58",
              ("timing_order", "natural_base_edges", "run_policy", "same_state", "clean_validation", "aggregate", "summarize"))]
    for _, old_name, new_name, pin, names in pairs:
        assert c.sha(ROOT / old_name) == pin
        old = (ROOT / old_name).read_text().replace("Sand", "Goop").replace("np.all(types == 6)", "np.all(types == 7)")
        new = (ROOT / new_name).read_text()
        funcs = lambda source: {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        a, b = funcs(old), funcs(new)
        for name in names:
            assert a[name] == b[name], name
    assert c.sha(ROOT / "sand_graph_support_policy.py") == "4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a"


def test_full_workload_and_observed_schedule_counts():
    final = load("evaluate_goop_graph_support_final")
    records = [{"id": f"valid:{i:06d}", "positions": {"shape": [401, 2, 2]}} for i in range(30)]
    assert [r["target_frame"] for r in final.schedules(records, "same-state")[:5]] == [7, 105, 203, 301, 400]
    assert len(final.schedules(records, "clean-validation")) == 128
    assert 6 * 6 * 30 == 1080
    assert 6 * 2 * 30 * 5 * (4 + 8 + 7 * 8) + 6 * 128 * 3 == 124704
