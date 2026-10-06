"""Independent collector composition checks with the real driver scalar gate.

All files are synthetic. Only the separately frozen cohort/split/context helper
bodies are replaced with recording sentinels; driver_context, validate_release,
head/selection/capacity checks and actual byte binding remain real.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("_independent_context_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load("test_run_goop_action_gate_v1")
M = load("summarize_goop_action_gate_v1")


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture(tmp_path, monkeypatch, change=None):
    args, release = T.release_fixture(tmp_path, monkeypatch, "test-rollout")
    args.repo.unlink(); args.repo.mkdir()
    source = args.repo / "synthetic.py"; source.write_text("synthetic source bytes; never imported\n")
    position = tmp_path / "position.npy"; position.write_bytes(b"synthetic opaque array; never loaded")
    types = tmp_path / "types.npy"; types.write_bytes(b"synthetic opaque types; never loaded")
    context_array = tmp_path / "context.npy"; context_array.write_bytes(b"synthetic opaque context; never loaded")
    metadata = {"bounds": [[.1, .9], [.1, .9]]}
    metadata_path = tmp_path / "metadata.json"; write(metadata_path, metadata)
    manifest = {"metadata": metadata, "records": [{name: {"path": p.name, "sha256": M.sha(p)}
                 for name, p in (("positions", position), ("particle_types", types), ("step_context", context_array))}]}
    model = {"arm": "mix", "seed": 0, "checkpoint_sha256": args.checkpoint_sha256, "config_sha256": "c" * 64}
    cohort = {"models": [model]}
    for path, value in ((args.manifest, manifest), (args.cohort, cohort), (args.train_admission, {"train": True}),
                        (args.admission, {"split": "test"}), (args.structural_report, {"structural": True})):
        write(path, value)
    head = json.loads(args.head.read_bytes())
    head["training_input_hashes"] = {"train_collection_sha256": "1" * 64, "checkpoint_sha256": args.checkpoint_sha256,
                                    "cohort_sha256": M.sha(args.cohort), "fit_release_sha256": "2" * 64}
    if change == "head_cohort": head["training_input_hashes"]["cohort_sha256"] = "f" * 64
    if change == "head_training_keys": head["training_input_hashes"].pop("fit_release_sha256")
    write(args.head, head)
    selection = json.loads(args.selection.read_bytes()); selection["heads"]["0"] = head; write(args.selection, selection)
    capacity_path = Path(release["complete_capacity_report"]["path"])
    capacity = json.loads(capacity_path.read_bytes())
    capacity.update(cohort_sha256=M.sha(args.cohort), selection_sha256=M.sha(args.selection)); write(capacity_path, capacity)
    release["complete_capacity_report"]["sha256"] = M.sha(capacity_path)
    release["files_sha256"] = {p: M.sha(p) for p in release["files_sha256"]}
    write(args.release, release)
    bindings = {**release["files_sha256"], str(args.release): M.sha(args.release), str(metadata_path): M.sha(metadata_path),
                str(source): M.sha(source), **{str(p): M.sha(p) for p in (position, types, context_array)}}
    command = [sys.executable, str(HERE / "run_goop_action_gate_v1.py"), "--execute", "--mode", args.mode,
               "--release", str(args.release), "--seed", "0", "--cuda-index", "0", "--threads", "2",
               "--max-seconds", "100", "--checkpoint-sha256", args.checkpoint_sha256]
    for name in T.D.PATH_ARGS:
        value = getattr(args, name.replace("-", "_"))
        if value is not None: command += ["--" + name, str(value)]
    entry = {"command": command, "seed": 0, "gpu": 0, "directory": str(args.output_dir), "outer_timeout_seconds": 100}
    phase = {"process_clock_checked_utc": T.NOW.isoformat(), "complete_capacity_report": release["complete_capacity_report"],
             "absolute_stop_utc": release["absolute_stop_utc"]}
    events = []

    def check_cohort(value, parsed, train):
        assert value == cohort and train == {"train": True}
        assert (parsed.split, parsed.arm, parsed.objective) == ("test", "mix", "faithful")
        events.append("frozen_cohort_helper_composed")

    def check_split(bench, value, admission, structural, parsed, cohort_sha):
        assert value == manifest and admission == {"split": "test"} and structural == {"structural": True}
        assert cohort_sha == M.sha(args.cohort)
        events.append("frozen_split_helper_composed")

    def evidence(parsed, value, admission, helpers):
        assert helpers.data_loader._manifest_array_path is M.array_path
        events.append("frozen_context_helper_composed")
        return {str(args.acquisition_report): M.sha(args.acquisition_report)}

    numerical = SimpleNamespace(final=SimpleNamespace(check_cohort=check_cohort, check_split=check_split,
                                                     METADATA_SHA=M.sha(metadata_path)),
                                bench=SimpleNamespace(SOURCE_PINS={"synthetic.py": M.sha(source)}, load_split_evidence=evidence))
    monkeypatch.setattr(T.D, "modules", lambda: numerical)
    if change == "parent_capacity": phase["complete_capacity_report"] = {"path": str(capacity_path), "sha256": "0" * 64}
    elif change == "parent_stop": phase["absolute_stop_utc"] = "2026-10-07T02:59:00+00:00"
    elif change == "output": entry["directory"] += "_different"
    elif change == "metadata": metadata_path.write_text("{}")
    elif change == "source_binding": bindings.pop(str(source))
    elif change == "context_binding": bindings.pop(str(args.acquisition_report))
    elif change == "array_binding": bindings.pop(str(position))
    elif change == "release_binding": bindings[str(args.release)] = "0" * 64
    return entry, phase, bindings, SimpleNamespace(D=T.D), events


def test_actual_driver_scalar_gate_composes_without_model_or_array_loading(tmp_path, monkeypatch):
    entry, phase, bindings, mods, events = fixture(tmp_path, monkeypatch)
    context = M.driver_context(entry, phase, bindings, mods)
    assert context.model["config_sha256"] == "c" * 64
    assert events == ["frozen_cohort_helper_composed", "frozen_split_helper_composed", "frozen_context_helper_composed"]


@pytest.mark.parametrize("change", ["parent_capacity", "parent_stop", "output", "metadata", "source_binding",
                                   "context_binding", "array_binding", "release_binding", "head_cohort", "head_training_keys"])
def test_original_lineage_composition_rejects_drift(tmp_path, monkeypatch, change):
    entry, phase, bindings, mods, _ = fixture(tmp_path, monkeypatch, change)
    with pytest.raises((ValueError, KeyError)):
        M.driver_context(entry, phase, bindings, mods)
