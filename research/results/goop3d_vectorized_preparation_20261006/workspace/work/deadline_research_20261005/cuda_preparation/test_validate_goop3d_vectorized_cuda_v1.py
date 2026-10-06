"""Synthetic tinyCPU and mocked CUDA control only; no actual arrays/device work."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from test_train_goop3d_graph_support_cuda_v1 import synthetic
import validate_goop3d_vectorized_cuda_v1 as V

ROOT = Path(__file__).resolve().parent


def test_default_has_no_scientific_imports_or_data_execution():
    code = """import builtins,runpy,sys
old=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','numpy','scipy','gns','research'):raise AssertionError(name)
    return old(name,*args,**kwargs)
builtins.__import__=checked
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, V.__file__], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["max_optimizer_updates"] == 26 and report["scientific_endpoint_selected"] is False


def manifest():
    records = [{"id": f"train:{i:06d}", "positions": {"shape": [301, 1000 + i % 17, 3]}} for i in range(1000)]
    return {"dataset": "Goop-3D", "split": "train", "record_count": 1000, "records": records}


def test_schedule_depends_only_on_count_source_order_and_fixed_frames():
    m = manifest()
    a = V.expected_schedule(m)
    for record in m["records"]:
        record["unrelated_outcome"] = 123
    assert a == V.expected_schedule(m)
    order = sorted(range(1000), key=lambda i: (m["records"][i]["positions"]["shape"][1], i))
    for row, start in zip(a, (0, 500, 998)):
        assert row["trajectory_indices"] == order[start:start + 2]
        assert row["target_frames"] == [6, 150]
        assert row["dataset_indices"] == [order[start] * 295, order[start + 1] * 295 + 144]
    m["split"] = "test"
    with pytest.raises(ValueError):
        V.expected_schedule(m)


def release(args, schedule):
    return {"schema": V.RELEASE_SCHEMA, "status": "admitted_for_bounded_validation", "issued_by": "root",
        "scientific_training_admitted": False, "source_sha256": V.sha(V.__file__), "trainer_sha256": V.TRAINER_SHA,
        "original_graph_sha256": V.ORIGINAL_SHA, "candidate_graph_sha256": V.CANDIDATE_SHA,
        "data_files_sha256": V.DATA_PINS, "schedule": schedule, "cuda_index": args.cuda_index, "gpu_uuid": args.gpu_uuid,
        "max_optimizer_updates": 26, "process_identity_checked_utc": datetime.now(timezone.utc).isoformat()}


def test_bounded_release_rejects_endpoint_scope_device_source_and_stale_inventory():
    args = SimpleNamespace(cuda_index=2, gpu_uuid="gpu-test")
    schedule = V.expected_schedule(manifest())
    source = release(args, schedule)
    V.validate_release(args, source, schedule)
    for key, value in (("scientific_training_admitted", True), ("source_sha256", "0" * 64),
        ("cuda_index", 1), ("max_optimizer_updates", 27),
        ("process_identity_checked_utc", (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())):
        changed = {**source, key: value}
        with pytest.raises(ValueError):
            V.validate_release(args, changed, schedule)


@pytest.mark.parametrize("arm", ["base", "mix"])
def test_graph_gates_and_two_ordinary_updates_are_bytewise_exact(synthetic, arm):
    s = synthetic
    trainer, h, original, candidate = V.load_modules(s.repo)
    left, lo = s.make()
    right, ro = s.make()
    for step in range(2):
        noise = h.host_noise(s.batch[0].shape, s.batch[1], 0, step)
        check = V.graph_gate(h, trainer, original, candidate, left, s.batch, noise, "cpu", step, arm)
        assert check["passed"] and check["rng_unchanged"]
        a = V.advance(h, trainer, original, left, lo, s.batch, noise, "cpu", step, step, arm)
        b = V.advance(h, trainer, candidate, right, ro, s.batch, noise, "cpu", step, step, arm)
        assert V.exact_comparison(h, trainer, a, b)["passed"]
        changed = copy.deepcopy(b)
        changed["before"]["prediction"][0, 0] += 1
        assert V.exact_comparison(h, trainer, a, changed)["passed"] is False


@pytest.mark.parametrize("failure_at", [None, 0, 4])
def test_actual_mock_orchestration_runs26_updates_and_preserves_failures(synthetic, tmp_path, monkeypatch, failure_at):
    s = synthetic
    trainer, h, original, candidate = V.load_modules(s.repo)
    args = SimpleNamespace(repo=s.repo, train_manifest=tmp_path / "data" / "train.json",
        release=tmp_path / "release.json", output_dir=tmp_path / "output", cuda_index=0, gpu_uuid="test-uuid", threads=1)
    args.train_manifest.parent.mkdir()
    args.train_manifest.write_text(json.dumps(manifest()))
    args.release.write_text("{}")
    schedule = [{"case": name, "trajectory_ids": ["train:000000", "train:000001"], "target_frames": [6, 150],
        "trajectory_indices": [0, 1], "particle_counts": [8, 8], "dataset_indices": [0, 1]} for name in V.CASES]
    monkeypatch.setattr(V, "expected_schedule", lambda m: schedule)
    monkeypatch.setattr(V, "validate_release", lambda *args: None)
    original_sha = V.sha
    monkeypatch.setattr(V, "sha", lambda path: V.DATA_PINS["train_manifest"] if Path(path) == args.train_manifest else original_sha(path))
    # This isolates orchestration from separately reviewed complete-source admission.
    monkeypatch.setattr(V, "load_data", lambda *args: ([None, None], {}, {"synthetic": True}))
    monkeypatch.setattr(V, "verify_data_evidence", lambda *args: None)
    monkeypatch.setattr(V, "load_modules", lambda repo: (trainer, h, original, candidate))
    monkeypatch.setattr(trainer, "unpack_batch", lambda *args: s.batch)
    def configure(h, args):
        assert len(V.read(args.output_dir / "report.json")["saved_batches"]) == 3
        return "cpu", {"uuid": args.gpu_uuid}
    monkeypatch.setattr(trainer, "configure_cuda", configure)
    def make(h, metadata, device, initial=None):
        model, optimizer = s.make()
        if initial is not None:
            model.load_state_dict(initial)
        return model, optimizer
    monkeypatch.setattr(V, "make_model", make)
    monkeypatch.setattr(h.data_loader, "load_manifest_data", lambda *args, **kwargs: None)
    if failure_at is not None:
        original_advance, counter = V.advance, [0]
        def fail(*args, **kwargs):
            if counter[0] == failure_at:
                raise FloatingPointError("Synthetic retained numerical failure")
            counter[0] += 1
            return original_advance(*args, **kwargs)
        monkeypatch.setattr(V, "advance", fail)
        with pytest.raises(FloatingPointError):
            V.execute(args)
        report = V.read(args.output_dir / "report.json")
        assert report["status"] == "implementation_failed" and report["completed_optimizer_calls"] == failure_at
        assert report["unsuccessful_outcome_retained"] is True
        assert (args.output_dir / report["unsuccessful_state"]["file"]).is_file()
        assert report["attempted_update"] is not None
        return
    assert V.execute(args) == 0
    report = V.read(args.output_dir / "report.json")
    assert report["status"] == "implementation_passed" and report["completed_optimizer_calls"] == 26
    assert all(report["replay"][arm]["final_payload_exact"] for arm in V.ARMS)
    assert len(list(args.output_dir.glob("*_prospective_batch.npz"))) == 3
    with pytest.raises(FileExistsError):
        V.execute(args)


def test_source_only_data_scope_refuses_scientific_endpoint_admission(monkeypatch):
    args = SimpleNamespace(**{key: Path(key) for key in V.DATA_PINS})
    monkeypatch.setattr(V, "sha", lambda path: V.DATA_PINS[str(path)])
    source_only = {"scope": "bounded_implementation_data_only", "scientific_training_admitted": False}
    for extra in ({"scientific_training_admitted": True}, {"prospective_endpoint_updates": 1000},
                  {"endpoint_selection_basis": "validation_selected"}, {"selection_evidence_sha256": ["0" * 64]}):
        with pytest.raises(ValueError, match="Source-only"):
            V.verify_data_evidence(args, None, None, {"data_contract": {**source_only, **extra}})
