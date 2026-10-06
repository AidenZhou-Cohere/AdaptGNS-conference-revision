"""Synthetic arrays, mocked predictions and tiny CPU tensors only; no real model/data."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("_test_final_diagnostics", ROOT / "evaluate_sand_final_diagnostics.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def manifest_records():
    return [{"id": f"valid:{i:06d}", "positions": {"shape": [320, 9, 2]}} for i in range(30)]


def test_clean_and_common_state_schedules_are_fixed_and_cover_all_trajectories():
    clean = entry.schedules(manifest_records(), "clean-validation")
    assert len(clean) == 128 and len({(x["source_index"], x["target_frame"]) for x in clean}) == 128
    assert {x["source_index"] for x in clean} == set(range(30))
    assert (clean[0]["source_index"], clean[0]["target_frame"]) == (0, 6)
    assert (clean[-1]["source_index"], clean[-1]["target_frame"]) == (29, 319)
    common = entry.schedules(manifest_records(), "same-state")
    assert len(common) == 150
    assert [x["target_frame"] for x in common[:5]] == [7, 85, 163, 241, 319]


def test_six_repetitions_balance_every_timing_slot():
    orders = [entry.timing_order(7, 3, repeat) for repeat in range(6)]
    for method in entry.TIMING_CASES:
        assert sorted(order.index(method) for order in orders) == list(range(6))


def test_missing_or_undefined_values_do_not_become_survivor_means():
    expected = [{"source_index": source, "target_frame": frame} for source in (0, 1) for frame in (7, 85)]
    rows = [{**item, "value": 1.0} for item in expected]
    assert entry.aggregate(expected, rows, lambda row: row["value"])["equal_trajectory_mean"] == 1.0
    rows[0]["value"] = None
    report = entry.aggregate(expected, rows, lambda row: row["value"])
    assert report["equal_trajectory_mean"] is None and report["defined_frames"] == 3 and report["defined_trajectories"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        entry.aggregate(expected, rows + [rows[0]], lambda row: row["value"])


def cohort_fixture(tmp_path):
    files = {}
    for name in ("protocol", "train_admission", "trainer_source", "benchmark_helper"):
        files[name] = tmp_path / (name + ".txt")
        files[name].write_text(name)
    args = SimpleNamespace(**files, benchmark_sha256=entry.sha(files["benchmark_helper"]), objective="faithful", seed=0,
                           checkpoint_sha256="1" * 64)
    cohort = {"schema": entry.COHORT_SCHEMA, "status": "frozen_for_final_evaluation", "dataset": "Sand", "updates": 100000,
              "protocol_sha256": entry.sha(files["protocol"]), "training_admission_sha256": entry.sha(files["train_admission"]),
              "trainer_source_sha256": entry.sha(files["trainer_source"]), "benchmark_helper_sha256": args.benchmark_sha256,
              "diagnostic_source_sha256": entry.sha(ROOT / "evaluate_sand_final_diagnostics.py"), "deterministic_algorithms": True,
              "cublas_workspace_config": ":4096:8", "models": [{"objective": objective, "seed": seed, "completed_steps": 100000,
                  "checkpoint_sha256": str(index + 1) * 64} for index, (objective, seed) in enumerate(
                      [(o, s) for o in ("faithful", "nll") for s in (0, 1, 2)])]}
    training = {"schema": "adaptgns_sand_training_admission_v1", "status": "admitted", "dataset": "Sand",
                "frames_per_trajectory": 320, "particle_type_ids": [6]}
    return args, cohort, training


def test_complete_cohort_gate_precedes_any_test_manifest(tmp_path, monkeypatch):
    args, cohort, training = cohort_fixture(tmp_path)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert entry.check_cohort(cohort, args, training)["checkpoint_sha256"] == args.checkpoint_sha256
    for mutation in (lambda c: c["models"].pop(), lambda c: c["models"][0].update(completed_steps=512),
                     lambda c: c.update(diagnostic_source_sha256="0" * 64)):
        changed = copy.deepcopy(cohort)
        mutation(changed)
        with pytest.raises(ValueError):
            entry.check_cohort(changed, args, training)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    with pytest.raises(ValueError, match="environment"):
        entry.check_cohort(cohort, args, training)


def test_default_description_imports_no_scientific_packages():
    path = str(ROOT / "evaluate_sand_final_diagnostics.py")
    program = f"""import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy', 'scipy', 'research', 'gns'):
        raise AssertionError('Default mode attempted scientific import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [{path!r}]
runpy.run_path({path!r}, run_name='__main__')
"""
    process = subprocess.run([sys.executable, "-I", "-"], input=program, text=True, capture_output=True, check=True)
    report = json.loads(process.stdout)
    assert report["status"] == "description_only" and report["execution"] is False


def test_main_rejects_incomplete_cohort_before_reading_test_manifest(tmp_path, monkeypatch):
    args, cohort, training = cohort_fixture(tmp_path)
    args.execute, args.cohort, args.manifest = True, tmp_path / "cohort.json", tmp_path / "reserved_test.json"
    args.train_admission.write_text(json.dumps(training))
    cohort["training_admission_sha256"] = entry.sha(args.train_admission)
    cohort["models"].pop()
    args.cohort.write_text(json.dumps(cohort))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(entry, "parse_args", lambda argv: args)
    reads, original = [], Path.read_text
    def monitored(path, *args, **kwargs):
        reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", monitored)
    with pytest.raises(ValueError, match="All six"):
        entry.main([])
    assert args.manifest not in reads


def split_fixture(tmp_path, split):
    spec = importlib.util.spec_from_file_location("_final_split_benchmark", ROOT / "benchmark_sand_cuda_rollout_deterministic.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args = SimpleNamespace(split=split)
    for name in ("train_admission", "protocol", "manifest", "structural_report"):
        setattr(args, name, tmp_path / (name + ".json"))
        getattr(args, name).write_text(name)
    source_sha = bench.SOURCES["valid"][2] if split == "valid" else "9" * 64
    acquisition_sha, converter_sha, cohort_sha = "a" * 64, "b" * 64, "c" * 64
    source = {"family": "designsafe_published_npz", "dataset": "Sand", "file": split + ".npz",
              "size_bytes": 82712898 if split == "valid" else 85825802, "sha256": source_sha,
              "ZIP_CRC_verified": True, "member_count": 30, "acquisition_report_sha256": acquisition_sha}
    manifest = {"format": "gns-trajectory-manifest", "version": 1, "split": split, "dataset": "Sand", "source": source,
                "metadata": {}, "metadata_sha256": entry.METADATA_SHA, "record_count": 30,
                "records": [], "converter_sha256": converter_sha}
    evidence_rows = []
    for index in range(30):
        value_hash = lambda label: entry.hashlib.sha256(f"{index}:{label}".encode()).hexdigest()
        positions = {"path": f"p{index}.npy", "shape": [320, 9, 2], "dtype": "<f4", "size_bytes": 100,
                     "sha256": value_hash("p")}
        types = {"path": f"t{index}.npy", "shape": [], "dtype": "<i8", "size_bytes": 100, "sha256": value_hash("t")}
        sp, st = value_hash("source_position"), value_hash("source_type")
        combine = lambda a, b: entry.hashlib.sha256((a + ":" + b).encode()).hexdigest()
        manifest["records"].append({"id": f"{split}:{index:06d}", "source_index": index, "source_member": f"{index}.npy",
            "positions": positions, "particle_types": types, "trajectory_content_sha256": combine(positions["sha256"], types["sha256"]),
            "logical_content_sha256": combine(sp, st)})
        evidence_rows.append({"source_index": index, "source_member": f"{index}.npy", "frames": 320, "particles": 9,
            "position_dtype": "<f4", "particle_type_dtype": "<i8", "particle_type_shape": [],
            "numeric_dtype_shape_values_verified_exact": True, "source_position_value_sha256": sp, "source_type_value_sha256": st})
    args.manifest.write_text(json.dumps(manifest))
    structural = {"schema": "designsafe_sand_numeric_repackage_v1", "status": "complete_structural_only", "dataset": "Sand",
        "source_family": "designsafe_published_npz", "metadata_sha256": entry.METADATA_SHA, "converter_sha256": converter_sha,
        "acquisition_report_sha256": acquisition_sha, "splits": {split: {"manifest_sha256": entry.sha(args.manifest),
            "record_count": 30, "frame_lengths": [320], "position_dtypes": ["<f4"], "particle_type_ids": [6],
            "kinematic_type3_particles": 0, "ZIP_CRC_verified": True, "all_numeric_values_preserved_exact": True, "records": evidence_rows}}}
    args.structural_report.write_text(json.dumps(structural))
    admission = {"schema": entry.ADMISSION_SCHEMA, "status": "admitted_for_final_evaluation", "dataset": "Sand", "split": split,
        "cohort_manifest_sha256": cohort_sha, "training_admission_sha256": entry.sha(args.train_admission),
        "protocol_sha256": entry.sha(args.protocol), "manifest_sha256": entry.sha(args.manifest),
        "structural_report_sha256": entry.sha(args.structural_report), "frames_per_trajectory": 320, "record_count": 30,
        "particle_type_ids": [6], "position_dtype": "<f4", "source_sha256": source_sha,
        "metadata_sha256": entry.METADATA_SHA, "converter_sha256": converter_sha}
    return bench, manifest, admission, structural, args, cohort_sha


@pytest.mark.parametrize("split", ["valid", "test"])
def test_separate_final_split_admission_requires_matching_provenance(tmp_path, split):
    values = split_fixture(tmp_path, split)
    entry.check_split(*values)
    for key in ("cohort_manifest_sha256", "training_admission_sha256", "protocol_sha256", "manifest_sha256",
                "structural_report_sha256", "source_sha256", "converter_sha256"):
        changed = copy.deepcopy(values[2])
        changed[key] = "0" * 64
        with pytest.raises(ValueError):
            entry.check_split(values[0], values[1], changed, *values[3:])
    changed = copy.deepcopy(values[1])
    changed["records"][1]["source_member"] = changed["records"][0]["source_member"]
    with pytest.raises(ValueError, match="record identity"):
        entry.check_split(values[0], changed, *values[2:])


@pytest.fixture
def synthetic(monkeypatch):
    # Import pinned numerical definitions; instantiate no GNS model.
    repo = ROOT.parents[2] / "outputs" / "AdaptGNS"
    spec = importlib.util.spec_from_file_location("_final_test_benchmark", ROOT / "benchmark_sand_cuda_rollout_deterministic.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    native, h = bench.load_helpers(repo)
    np, torch = native.np, h.torch
    torch.set_num_threads(1)
    base = np.asarray([[.2 + x * .016, .2 + y * .016] for y in range(3) for x in range(3)], dtype=np.float32)
    positions = np.stack([base + np.float32(frame * 1e-6) for frame in range(320)])
    model = SimpleNamespace(_normalization_stats={"acceleration": {"std": np.array([.1, .2]), "mean": np.zeros(2)}},
                            head_to_variance=lambda head: head.clamp_min(1e-6))
    def supplied(model, history, types, edges, device):
        prediction = history[-1] + (history[-1] - history[-2]) + np.float32(edges.shape[1] * 1e-7)
        risk = np.linspace(1.0, 2.0, len(types), dtype=np.float32)
        return {"prediction": prediction, "risk": risk, "raw_risk": risk.copy(), "operational_seconds": 0.0}
    monkeypatch.setattr(native.bridge, "supplied", supplied)
    monkeypatch.setattr(native.bridge, "native_parity", lambda model, history, types, edges, device:
        ({"passed": True}, {"native_edges": edges.copy()}, supplied(model, history, types, edges, device)))
    item = {"source_index": 0, "trajectory_id": "valid:000000", "schedule_index": 0, "target_frame": 85}
    return SimpleNamespace(native=native, h=h, model=model, positions=positions, types=np.full(9, 6, dtype=np.int64),
                           item=item, metadata={"bounds": [[.1, .9], [.1, .9]]}, supplied=supplied)


def test_natural_base_preserves_native_strict_cap_and_self_prefix(synthetic):
    n, np = synthetic.native, synthetic.native.np
    for history in (synthetic.positions[:6], np.full((6, 132, 2), .2, dtype=np.float32)):
        edges, _ = entry.natural_base_edges(n, history)
        _, native_edges, _, _ = n.native_graph(history, "base", None, np.random.default_rng(0))
        assert np.array_equal(edges, native_edges)


def test_same_state_charges_previous_base_pass_and_has_exact_budgets(synthetic):
    s = synthetic
    row, arrays = entry.same_state(s.native, s.model, s.positions, s.types, s.metadata, s.item, "valid", 0, "cpu")
    assert row["status"] == "complete", row["failure"]
    assert len(row["timed_calls"]) == 36 and len(row["warmup_calls"]) == 6
    for call in row["timed_calls"]:
        assert call["timing"]["network_passes"] == (2 if call["method"] == entry.POLICIES[-1] else 1)
    for method in entry.POLICIES[2:]:
        graph = row["policies"][method]["graph"]
        assert graph["retained_optional_pairs"] == graph["optional_pair_budget"] == 3
        assert graph["native_base_prefix_preserved"]
    assert row["natural_shared_base"]["ordered_edges_exact"]
    assert "signed_normalized_benefit__previous-observed-base-risk25" in arrays
    assert "dense_sparse_sign_disagreement_fraction" in row["benefit"][entry.POLICIES[-1]]
    assert s.native.np.array_equal(arrays["current_history"], s.positions[79:85])
    assert s.native.np.array_equal(arrays["previous_history"], s.positions[78:84])
    for repetition in range(7):
        assert f"r{repetition}_previous-observed-base-risk25__previous_base_risk" in arrays


def test_future_target_cannot_change_selection_or_predictions(synthetic):
    s = synthetic
    original, a = entry.same_state(s.native, s.model, s.positions, s.types, s.metadata, s.item, "valid", 0, "cpu")
    changed = s.positions.copy()
    changed[85] += .01
    modified, b = entry.same_state(s.native, s.model, changed, s.types, s.metadata, s.item, "valid", 0, "cpu")
    for key in a:
        if key.endswith("__prediction") or key.endswith("__edges") or key.endswith("__selected_optional_pairs"):
            assert s.native.np.array_equal(a[key], b[key])
    assert original["policies"]["base"]["metrics"] != modified["policies"]["base"]["metrics"]


def test_failed_dense_outputs_preserve_other_policies_and_null_coverage(synthetic, monkeypatch):
    s = synthetic
    def failed_dense(model, history, types, edges, device):
        output = s.supplied(model, history, types, edges, device)
        if edges.shape[1] == 33:
            output["prediction"][:] = s.native.np.nan
        return output
    monkeypatch.setattr(s.native.bridge, "supplied", failed_dense)
    row, arrays = entry.same_state(s.native, s.model, s.positions, s.types, s.metadata, s.item, "valid", 0, "cpu")
    assert row["status"] == "failed" and row["policies"]["dense"]["metrics"] is None
    assert row["policies"]["base"]["status"] == "complete"
    assert any(s.native.np.isnan(value).any() for key, value in arrays.items() if "dense" in key and key.endswith("prediction"))
    summary = entry.summarize([s.item], [row], "same-state")
    assert summary["accuracy"]["dense"]["position_coordinate_mse"]["equal_trajectory_mean"] is None
    assert summary["accuracy"]["base"]["position_coordinate_mse"]["equal_trajectory_mean"] is not None


@pytest.mark.parametrize("raw_head", [0.0, 1e-9, 3.0])
def test_clean_validation_uses_interpreted_variance_floor_and_saved_raw_head(synthetic, monkeypatch, raw_head):
    s, torch = synthetic, synthetic.h.torch
    monkeypatch.setattr(s.h, "forward_batch", lambda model, batch, noise, device:
                        (torch.full((9, 2), 2.0), torch.full((9,), raw_head), torch.zeros(9, 2)))
    row, arrays = entry.clean_validation(s.native, s.h, s.model, s.positions, s.types, s.item, "cpu")
    assert row["status"] == "complete", row["failure"]
    assert row["metrics"]["normalized_acceleration_coordinate_mse"] == 4.0
    assert row["metrics"]["realized_normalized_vector_se"] == 8.0
    assert row["metrics"]["predicted_normalized_vector_se"] == pytest.approx(2 * max(raw_head, 1e-6))
    assert (arrays["raw_head"] == raw_head).all()
    assert (arrays["predicted_variance"] > 0).all()


def test_native_parity_failure_returns_saved_evidence(synthetic, monkeypatch):
    s = synthetic
    monkeypatch.setattr(s.native.bridge, "native_parity", lambda model, history, types, edges, device:
                        ({"passed": False}, {"native_edges": edges.copy()}, s.supplied(model, history, types, edges, device)))
    for call in (lambda: entry.clean_validation(s.native, s.h, s.model, s.positions, s.types, s.item, "cpu"),
                 lambda: entry.same_state(s.native, s.model, s.positions, s.types, s.metadata, s.item, "valid", 0, "cpu")):
        row, arrays = call()
        assert row["failure"]["category"] == "native_parity_failure"
        assert any("parity" in key and "edges" in key for key in arrays)


def test_natural_shared_output_disagreement_preserves_evidence_and_stops(synthetic, monkeypatch):
    s, original = synthetic, entry.run_policy
    def altered(*args, **kwargs):
        row, arrays = original(*args, **kwargs)
        if args[5] == entry.NATURAL_BASE:
            arrays["prediction"] += .001
        return row, arrays
    monkeypatch.setattr(entry, "run_policy", altered)
    row, arrays = entry.same_state(s.native, s.model, s.positions, s.types, s.metadata, s.item, "valid", 0, "cpu")
    assert row["failure"]["category"] == "execution_error"
    assert row["natural_shared_base"]["ordered_edges_exact"]
    assert not row["natural_shared_base"]["outputs_within_fixed_parity_tolerance"]
    assert "target_position" not in arrays
    assert any(entry.NATURAL_BASE in key and key.endswith("prediction") for key in arrays)


def test_execution_exception_preserves_current_graph_and_halts_before_next_inference(synthetic, monkeypatch):
    s = synthetic
    calls = []
    def broken(model, history, types, edges, device):
        calls.append(edges.shape[1])
        if edges.shape[1] == 33:
            raise RuntimeError("synthetic dense execution failure")
        return s.supplied(model, history, types, edges, device)
    monkeypatch.setattr(s.native.bridge, "supplied", broken)
    row, arrays = entry.same_state(s.native, s.model, s.positions, s.types, s.metadata, s.item, "valid", 0, "cpu")
    assert row["failure"]["category"] == "execution_error" and row["failure"]["method"] == "dense"
    assert calls == [9, 33] and len(row["warmup_calls"]) == 2 and not row["timed_calls"]
    assert arrays["r0_dense__edges"].shape == (2, 33) and "target_position" not in arrays


def test_changed_previous_risk_cannot_hide_behind_same_selected_graph(synthetic, monkeypatch):
    s, original, count = synthetic, entry.run_policy, 0
    def altered(*args, **kwargs):
        nonlocal count
        row, arrays = original(*args, **kwargs)
        if args[5] == entry.POLICIES[-1]:
            count += 1
            arrays["previous_base_risk"] += count * .1
        return row, arrays
    monkeypatch.setattr(entry, "run_policy", altered)
    row, arrays = entry.same_state(s.native, s.model, s.positions, s.types, s.metadata, s.item, "valid", 0, "cpu")
    assert row["policies"][entry.POLICIES[-1]]["status"] == "failed"
    assert row["policies"][entry.POLICIES[-1]]["metrics"] is None
    calls = [call for call in row["timed_calls"] if call["method"] == entry.POLICIES[-1]]
    assert calls[-1]["repeat_consistency"]["ordered_edges_exact"]
    assert not calls[-1]["repeat_consistency"]["previous_score_outputs_within_fixed_parity_tolerance"]
