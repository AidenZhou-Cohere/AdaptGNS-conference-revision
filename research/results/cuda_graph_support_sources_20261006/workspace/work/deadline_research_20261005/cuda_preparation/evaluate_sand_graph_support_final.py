#!/usr/bin/env python3
"""Prepared final Sand clean-validation and native common-state diagnostics.

Default is description only. Execution requires a frozen complete six-model
100k cohort, separately admitted split, original training admission and pinned
benchmark helper. No acquisition, training, checkpoint selection or tuning.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_graph_support_final_evaluation_v1"
COHORT_SCHEMA = "adaptgns_sand_graph_support_final_cohort_v1"
ADMISSION_SCHEMA = "adaptgns_sand_graph_support_final_evaluation_admission_v1"
PHYSICAL_POLICY = "relative-velocity-RMS25"
POLICIES = ("base", "dense", "random25", "speed25", PHYSICAL_POLICY, "previous-observed-base-risk25")
NATURAL_BASE = "natural_base_reference"
TIMING_CASES = POLICIES + (NATURAL_BASE,)
REPEATS = 7
FRAMES = 320
COUNT = 30
METADATA_SHA = "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def description():
    return {"schema": SCHEMA, "status": "description_only", "execution": False,
            "models": "all six frozen faithful base/mix x seeds0/1/2 at exactly100000 updates; selected member only",
            "clean_validation": "128 distinct floor(i*(eligible-1)/127) validation-history indices; clean noise0; no model selection",
            "full_rollout": "all30 admitted trajectories; full314 forecasts; six prospectively included policies; one frozen arm/seed per invocation",
            "same_state": "all30 admitted trajectories in each separately reported split; targets7,85,163,241,319 for T320",
            "policies": POLICIES, "timing_reference": NATURAL_BASE, "warmup_rounds": 1, "timed_rounds": REPEATS,
            "timing": "seven cyclic orders; every case occupies every slot once; previous-base score pass charged",
            "risk_source": "previous observed history on native base, distinct from autonomous cached-own-graph risk",
            "test_gate": "root all-six endpoint/optimizer/pairing audit pin and selected checkpoint bytes verified before opening any evaluation manifest"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("clean-validation", "same-state", "full-rollout"))
    parser.add_argument("--split", choices=("valid", "test"))
    for name in ("repo", "benchmark-helper", "cohort", "manifest", "admission", "structural-report",
                 "train-admission", "trainer-source", "protocol", "checkpoint", "output-dir", "cohort-audit"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--benchmark-sha256")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--objective", choices=("faithful",), default="faithful")
    parser.add_argument("--arm", choices=("base", "mix"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.execute:
        required = ("mode", "split", "repo", "benchmark_helper", "benchmark_sha256", "cohort", "manifest",
                    "admission", "structural_report", "train_admission", "trainer_source", "protocol", "checkpoint",
                    "checkpoint_sha256", "objective", "arm", "seed", "output_dir", "cohort_audit")
        require(all(getattr(args, name) is not None for name in required), "Every execution/source/cohort argument is required")
        require(args.seed in (0, 1, 2) and digest(args.benchmark_sha256) and digest(args.checkpoint_sha256), "Invalid fixed seed or SHA256")
        require(args.mode != "clean-validation" or args.split == "valid", "Clean validation cannot access test")
    require(args.cuda_index >= 0 and 1 <= args.threads <= 16 and 1 <= args.max_seconds <= 7200, "Invalid runtime bound")
    return args


def check_cohort(cohort, args, train_admission):
    """Root audits all six endpoint bytes once; every call verifies the audit pin.

    Other cohort members may live on the second host. The selected checkpoint
    is byte-verified here and its payload is validated before any inference.
    """
    require(cohort.get("schema") == COHORT_SCHEMA and cohort.get("status") == "frozen_for_final_evaluation"
            and cohort.get("dataset") == "Sand" and cohort.get("updates") == 100000,
            "Complete frozen graph-support final cohort required")
    for field, path in (("protocol_sha256", args.protocol), ("training_admission_sha256", args.train_admission),
                        ("trainer_source_sha256", args.trainer_source), ("cohort_audit_sha256", args.cohort_audit)):
        require(cohort.get(field) == sha(path), "Cohort input identity differs: " + field)
    require(cohort.get("benchmark_helper_sha256") == args.benchmark_sha256 == sha(args.benchmark_helper),
            "Final benchmark helper identity differs")
    require(cohort.get("diagnostic_source_sha256") == sha(__file__), "Final evaluation source is not frozen")
    require(train_admission.get("schema") == "adaptgns_sand_training_admission_v1"
            and train_admission.get("status") == "admitted" and train_admission.get("dataset") == "Sand"
            and train_admission.get("frames_per_trajectory") == FRAMES and train_admission.get("particle_type_ids") == [6],
            "Original admitted training lineage required")
    expected = {(arm, seed) for arm in ("base", "mix") for seed in (0, 1, 2)}
    models = cohort.get("models", [])
    require(len(models) == 6 and {(row.get("arm"), row.get("seed")) for row in models} == expected
            and all(type(row.get("seed")) is int and row.get("objective") == "faithful"
                    and row.get("completed_steps") == 100000 and digest(row.get("checkpoint_sha256")) for row in models)
            and len({row["checkpoint_sha256"] for row in models}) == 6, "All six distinct final base/mix checkpoints required")
    audit = json.loads(args.cohort_audit.read_text())
    require(audit.get("schema") == "adaptgns_sand_graph_support_complete_cohort_audit_v1"
            and audit.get("status") == "all_six_endpoints_and_pairing_verified"
            and audit.get("training_schema") == "adaptgns_sand_graph_support_cuda_training_v1",
            "Root complete endpoint/optimizer/pairing audit required")
    for field in ("protocol_sha256", "training_admission_sha256", "trainer_source_sha256"):
        require(audit.get(field) == cohort[field], "Cohort audit input identity differs")
    checked = audit.get("models", [])
    require(len(checked) == 6 and {(r.get("arm"), r.get("seed")) for r in checked} == expected,
            "Audit must cover every arm/seed")
    for row in models:
        evidence = next(r for r in checked if (r["arm"], r["seed"]) == (row["arm"], row["seed"]))
        require(evidence.get("checkpoint_sha256") == row["checkpoint_sha256"]
                and evidence.get("completed_steps") == evidence.get("graph_history_updates") == 100000
                and evidence.get("checkpoint_every") == 10000 and evidence.get("log_every") == 100
                and evidence.get("all_optimizer_steps_equal_100000") is True
                and evidence.get("all_state_and_moments_finite") is True
                and evidence.get("source_data_protocol_verified") is True
                and evidence.get("checkpoint_bytes_verified") is True,
                "Root endpoint audit is incomplete or differs from frozen cohort")
    pairs = audit.get("paired_seeds", [])
    require(len(pairs) == 3 and {r.get("seed") for r in pairs} == {0, 1, 2}
            and all(type(r.get("seed")) is int and r.get("initial_model_tensor_identity") is True
                    and r.get("initial_cpu_cuda_rng_identity") is True
                    and r.get("all_frame_noise_lr_schedules_equal") is True
                    and r.get("all_graph_budgets_and_rng_material_verified") is True for r in pairs),
            "Root audit must verify initialization and every paired schedule")
    require(cohort.get("policies") == ["base", "dense", "random25", "speed25", "laggedrisk25", PHYSICAL_POLICY],
            "All six prospectively included rollout policies required")
    selected = next(row for row in models if (row["arm"], row["seed"]) == (args.arm, args.seed))
    require(selected["checkpoint_sha256"] == args.checkpoint_sha256 == sha(args.checkpoint),
            "Selected checkpoint bytes differ from complete cohort")
    require(cohort.get("deterministic_algorithms") is True and cohort.get("cublas_workspace_config") == ":4096:8"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Frozen deterministic environment required before process start")
    return selected


def check_split(bench, manifest, admission, structural, args, cohort_hash):
    require(admission.get("schema") == ADMISSION_SCHEMA and admission.get("status") == "admitted_for_final_evaluation"
            and admission.get("dataset") == "Sand" and admission.get("split") == args.split
            and admission.get("cohort_manifest_sha256") == cohort_hash
            and admission.get("training_admission_sha256") == sha(args.train_admission)
            and admission.get("protocol_sha256") == sha(args.protocol), "Separate final split admission must bind this cohort and protocol")
    require(admission.get("manifest_sha256") == sha(args.manifest)
            and admission.get("structural_report_sha256") == sha(args.structural_report), "Admitted split bytes differ")
    require(admission.get("frames_per_trajectory") == FRAMES and admission.get("record_count") == COUNT
            and admission.get("particle_type_ids") == [6] and admission.get("position_dtype") == "<f4", "Final Sand source contract differs")
    require(set(manifest) == bench.MANIFEST_KEYS and manifest.get("format") == "gns-trajectory-manifest"
            and manifest.get("version") == 1 and manifest.get("dataset") == "Sand" and manifest.get("split") == args.split,
            "Exact numeric split manifest required")
    source = manifest["source"]
    expected_bytes = 82712898 if args.split == "valid" else 85825802
    require(set(source) == bench.SOURCE_KEYS and source.get("family") == "designsafe_published_npz"
            and source.get("dataset") == "Sand" and source.get("file") == args.split + ".npz"
            and source.get("size_bytes") == expected_bytes and source.get("member_count") == COUNT
            and source.get("ZIP_CRC_verified") is True and digest(source.get("sha256"))
            and source["sha256"] == admission.get("source_sha256")
            and digest(source.get("acquisition_report_sha256")), "Exact acquired final source identity differs")
    if args.split == "valid":
        require(source["sha256"] == bench.SOURCES["valid"][2], "Validation source changed")
    require(manifest.get("metadata_sha256") == admission.get("metadata_sha256") == METADATA_SHA
            and digest(manifest.get("converter_sha256")) and manifest["converter_sha256"] == admission.get("converter_sha256"),
            "Final metadata/converter identity differs")
    require(structural.get("schema") == "designsafe_sand_numeric_repackage_v1"
            and structural.get("status") == "complete_structural_only" and structural.get("dataset") == "Sand"
            and structural.get("source_family") == "designsafe_published_npz"
            and structural.get("metadata_sha256") == METADATA_SHA
            and structural.get("converter_sha256") == admission["converter_sha256"]
            and structural.get("acquisition_report_sha256") == source["acquisition_report_sha256"], "Structural provenance differs")
    detail = structural.get("splits", {}).get(args.split, {})
    require(detail.get("manifest_sha256") == admission["manifest_sha256"] and detail.get("record_count") == COUNT
            and detail.get("frame_lengths") == [FRAMES] and detail.get("position_dtypes") == ["<f4"]
            and detail.get("particle_type_ids") == [6] and detail.get("kinematic_type3_particles") == 0
            and detail.get("ZIP_CRC_verified") is True and detail.get("all_numeric_values_preserved_exact") is True,
            "Complete structural preservation evidence required")
    require(manifest.get("record_count") == len(manifest.get("records", [])) == len(detail.get("records", [])) == COUNT, "Every final split record is required")
    contents, logical, paths, members = set(), set(), set(), set()
    for index, (record, evidence) in enumerate(zip(manifest["records"], detail["records"])):
        require(not (set(record) - bench.RECORD_KEYS) and (bench.RECORD_KEYS - {"source_inner_index"}) <= set(record)
                and type(record["source_index"]) is int and record["source_index"] == index
                and record["id"] == f"{args.split}:{index:06d}" and isinstance(record["source_member"], str)
                and record["source_member"] not in members, "Ordered record identity/fields differ")
        members.add(record["source_member"])
        p, t = record["positions"], record["particle_types"]
        for array in (p, t):
            require(set(array) == bench.ARRAY_KEYS and digest(array["sha256"])
                    and type(array["size_bytes"]) is int and array["size_bytes"] > 0
                    and isinstance(array["path"], str) and array["path"] not in paths, "Array identity/fields differ")
            paths.add(array["path"])
        require(len(p["shape"]) == 3 and p["shape"][0] == FRAMES and p["shape"][2] == 2
                and type(p["shape"][1]) is int and p["shape"][1] > 0 and p["dtype"] == "<f4"
                and t["shape"] in ([], [p["shape"][1]]) and t["dtype"] == "<i8", "Source shape/dtype differs")
        require(evidence.get("source_index") == index and evidence.get("source_member") == record["source_member"]
                and evidence.get("frames") == FRAMES and evidence.get("particles") == p["shape"][1]
                and evidence.get("position_dtype") == p["dtype"] and evidence.get("particle_type_dtype") == t["dtype"]
                and evidence.get("particle_type_shape") == t["shape"] and evidence.get("numeric_dtype_shape_values_verified_exact") is True,
                "Record and structural evidence differ")
        content = hashlib.sha256((p["sha256"] + ":" + t["sha256"]).encode()).hexdigest()
        require(content == record["trajectory_content_sha256"] and content not in contents
                and digest(record["logical_content_sha256"]) and record["logical_content_sha256"] not in logical,
                "Duplicate/inconsistent record content")
        for key in ("source_position_value_sha256", "source_type_value_sha256"):
            require(digest(evidence.get(key)), "Missing exact source-value hash")
        require(record["logical_content_sha256"] == hashlib.sha256((evidence["source_position_value_sha256"] + ":" + evidence["source_type_value_sha256"]).encode()).hexdigest(),
                "Source logical-value identity differs")
        contents.add(content)
        logical.add(record["logical_content_sha256"])


def schedules(records, mode):
    require(len(records) == COUNT and all(row["positions"]["shape"][0] == FRAMES for row in records), "Complete T320 split required")
    if mode == "clean-validation":
        eligible = COUNT * (FRAMES - 6)
        indices = [i * (eligible - 1) // 127 for i in range(128)]
        require(len(set(indices)) == 128, "128 distinct clean validation histories required")
        pairs = [(index // (FRAMES - 6), index % (FRAMES - 6) + 6) for index in indices]
    else:
        targets = sorted({7 + j * (FRAMES - 8) // 4 for j in range(5)})
        pairs = [(index, target) for index in range(COUNT) for target in targets]
    return [{"schedule_index": k, "source_index": index, "trajectory_id": records[index]["id"], "target_frame": target}
            for k, (index, target) in enumerate(pairs)]


def timing_order(source_index, schedule_index, round_index):
    offset = (source_index * 5 + schedule_index + round_index) % len(TIMING_CASES)
    return TIMING_CASES[offset:] + TIMING_CASES[:offset]


def natural_base_edges(native, history):
    """Strict float32 base-only query; same pinned cap/self/ordering helper."""
    bridge, np = native.bridge, native.np
    points = np.asarray(history[-1], dtype=np.float32)
    tree = bridge.cKDTree(points)
    count = int((tree.count_neighbors(tree, .015) - len(points)) // 2)
    if count > bridge.MAX_PAIRS:
        raise native.full.RolloutGuard("candidate_pair_resource_guard", candidate_pairs=count, limit=bridge.MAX_PAIRS)
    pairs = bridge.same.canonical_pairs(tree.query_pairs(.015, output_type="ndarray"))
    distances = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    pairs = pairs[distances < .015]
    return bridge.ordered_edges(points, pairs, True, 128), pairs


def run_policy(native, model, current, previous, types, method, seed_material, device):
    """No target argument; standalone cost includes the previous-base risk pass."""
    np, bridge, full = native.np, native.bridge, native.full
    full.synchronize(device)
    started = time.perf_counter()
    arrays, score, passes, phase = {}, None, 0, "observed_state"
    timing = {"score_graph_seconds": 0.0, "score_forward_seconds": 0.0, "score_generation_seconds": 0.0}
    try:
        full.check_state(current, "observed_state", bridge.MAX_ABS)
        full.check_state(previous, "previous_observed_state", bridge.MAX_ABS)
        if method == POLICIES[-1]:
            score_started = time.perf_counter()
            phase = "previous_score_graph"
            _, old_edges, _, old_audit = native.native_graph(previous, "base", None, np.random.default_rng(0))
            timing["score_graph_seconds"] = time.perf_counter() - score_started
            arrays["previous_base_edges"] = old_edges
            phase = "previous_score_forward"
            forward_started = time.perf_counter()
            passes += 1
            old = bridge.supplied(model, previous, types, old_edges, device)
            arrays.update({"previous_base_" + key: old[key] for key in ("prediction", "risk", "raw_risk")})
            bridge.validate_output(old, len(types))
            score = old["risk"]
            timing["score_forward_seconds"] = time.perf_counter() - forward_started
            timing["score_generation_seconds"] = time.perf_counter() - score_started
        phase = "current_graph"
        graph_started = time.perf_counter()
        if method == NATURAL_BASE:
            edges, pairs = natural_base_edges(native, current)
            optional = np.empty((0, 2), dtype=np.int64)
            audit = {"directed_edge_sha256": full.state_hash(edges), "retained_optional_pairs": 0,
                     "natural_base_pairs": len(pairs), "native_base_prefix_preserved": True}
            arrays["base_pairs"] = pairs
        else:
            native_policy = "laggedrisk25" if method == POLICIES[-1] else method
            rng = np.random.default_rng(np.random.SeedSequence(seed_material))
            if method == PHYSICAL_POLICY:
                graph, edges, optional, audit, physical_scores = native.physical_graph(current, rng)
                arrays["physical_selection_scores"] = physical_scores
            else:
                graph, edges, optional, audit = native.native_graph(current, native_policy, score, rng)
            arrays.update(base_pairs=graph.base, annulus_pairs=graph.extra)
        timing["current_graph_and_selection_seconds"] = time.perf_counter() - graph_started
        arrays.update(edges=edges, selected_optional_pairs=optional)
        phase = "current_forward"
        forward_started = time.perf_counter()
        passes += 1
        output = bridge.supplied(model, current, types, edges, device)
        arrays.update({key: output[key] for key in ("prediction", "risk", "raw_risk")})
        bridge.validate_output(output, len(types))
        timing["current_forward_seconds"] = time.perf_counter() - forward_started
        full.synchronize(device)
        timing.update(end_to_end_seconds=time.perf_counter() - started, network_passes=passes)
        return {"status": "complete", "failure": None, "graph": audit, "timing": timing}, arrays
    except full.RolloutGuard as error:
        full.synchronize(device)
        return {"status": "failed", "failure": {**error.details, "phase": phase},
                "timing": {**timing, "failed_attempt_seconds": time.perf_counter() - started, "attempted_network_passes": passes}}, arrays
    except Exception as error:
        # Preserve the current graph and any returned outputs before stopping.
        return {"status": "failed", "failure": {"category": "execution_error", "phase": phase,
                    "error_type": type(error).__name__, "error": str(error)},
                "timing": {**timing, "failed_attempt_seconds": time.perf_counter() - started, "attempted_network_passes": passes}}, arrays


def same_state(native, model, positions, particle_types, metadata, item, split, seed, device):
    np, bridge, full = native.np, native.bridge, native.full
    target_frame = item["target_frame"]
    current = np.array(positions[target_frame - 6:target_frame], dtype=np.float32, copy=True)
    previous = np.array(positions[target_frame - 7:target_frame - 1], dtype=np.float32, copy=True)
    types = np.asarray(particle_types, dtype=np.int64)
    if types.ndim == 0:
        types = np.full(current.shape[1], types, dtype=np.int64)
    material = [20261006, 93000, seed, 0 if split == "valid" else 1, item["source_index"], target_frame]
    row = {**item, "status": "failed", "failure": None, "random_seed_material": material,
           "risk_source": "previous observed native-base history", "warmup_calls": [], "timed_calls": [],
           "policies": {}, "native_parity": {}, "benefit": {}, "correlations": {}}
    arrays = {"current_history": current, "previous_history": previous, "particle_types": types}
    started = time.perf_counter()
    try:
        full.check_state(current, "current_observed_history", bridge.MAX_ABS)
        full.check_state(previous, "previous_observed_history", bridge.MAX_ABS)
        require(current.shape == previous.shape and current.shape[0] == 6 and current.shape[2] == 2
                and types.shape == (current.shape[1],) and np.all(types == 6), "Observed history/type contract differs")
        for label, history in (("current", current), ("previous", previous)):
            _, edges, _, _ = native.native_graph(history, "base", None, np.random.default_rng(0))
            parity, saved, output = bridge.native_parity(model, history, types, edges, device)
            row["native_parity"][label] = parity
            arrays.update({label + "_parity_" + key: value for key, value in saved.items()})
            arrays.update({label + "_parity_supplied_" + key: output[key] for key in ("prediction", "risk", "raw_risk")})
            if not parity["passed"]:
                row["failure"] = {"category": "native_parity_failure", "history": label}
                row["total_wall_seconds_including_warmup_parity_audits"] = time.perf_counter() - started
                return row, arrays
        row["parity_and_setup_seconds"] = time.perf_counter() - started
        reference, stable = {}, {method: True for method in TIMING_CASES}
        for repetition in range(-1, REPEATS):
            calls = row["warmup_calls"] if repetition < 0 else row["timed_calls"]
            for slot, method in enumerate(timing_order(item["source_index"], item["schedule_index"], max(0, repetition))):
                result, saved = run_policy(native, model, current, previous, types, method, material, device)
                call = {"method": method, "round": repetition, "slot": slot, **result}
                if "prediction" in saved:
                    call["prediction_sha256"] = full.state_hash(saved["prediction"])
                calls.append(call)
                prefix = f"r{repetition + 1}_{method}__"
                # Keep all repeated predictions/risks, first graph and any failed graph.
                for key, value in saved.items():
                    if repetition == 0 or result["status"] != "complete" or (repetition >= 0 and method not in reference) or key.endswith(("prediction", "risk")):
                        arrays[prefix + key] = value
                if (result.get("failure") or {}).get("category") == "execution_error":
                    row["failure"] = {**result["failure"], "method": method, "round": repetition, "slot": slot}
                    row["total_wall_seconds_including_warmup_parity_audits"] = time.perf_counter() - started
                    return row, arrays
                if repetition < 0:
                    if result["status"] != "complete":
                        stable[method] = False
                    continue
                if result["status"] != "complete":
                    stable[method] = False
                    continue
                if method not in reference:
                    reference[method] = (result, saved)
                else:
                    old = reference[method][1]
                    identical_edges = np.array_equal(old["edges"], saved["edges"])
                    numerical = np.allclose(old["prediction"], saved["prediction"], rtol=0, atol=bridge.PARITY_PRED_ATOL)
                    numerical = numerical and all(np.allclose(old[key], saved[key], rtol=bridge.PARITY_RISK_RTOL,
                        atol=bridge.PARITY_RISK_ATOL) for key in ("risk", "raw_risk"))
                    call["repeat_consistency"] = {"ordered_edges_exact": bool(identical_edges), "outputs_within_fixed_parity_tolerance": bool(numerical)}
                    if method == POLICIES[-1]:
                        score_identical_edges = np.array_equal(old["previous_base_edges"], saved["previous_base_edges"])
                        score_numerical = np.allclose(old["previous_base_prediction"], saved["previous_base_prediction"],
                            rtol=0, atol=bridge.PARITY_PRED_ATOL) and all(np.allclose(old[key], saved[key],
                                rtol=bridge.PARITY_RISK_RTOL, atol=bridge.PARITY_RISK_ATOL)
                                for key in ("previous_base_risk", "previous_base_raw_risk"))
                        call["repeat_consistency"].update(previous_score_ordered_edges_exact=bool(score_identical_edges),
                            previous_score_outputs_within_fixed_parity_tolerance=bool(score_numerical))
                        numerical = numerical and score_identical_edges and score_numerical
                    if method == PHYSICAL_POLICY:
                        exact_scores = np.array_equal(old["physical_selection_scores"], saved["physical_selection_scores"])
                        call["repeat_consistency"]["physical_scores_exact"] = bool(exact_scores)
                        numerical = numerical and exact_scores
                    stable[method] &= bool(identical_edges and numerical)
        if "base" in reference and NATURAL_BASE in reference:
            shared, natural = reference["base"][1], reference[NATURAL_BASE][1]
            row["natural_shared_base"] = {"ordered_edges_exact": bool(np.array_equal(shared["edges"], natural["edges"])),
                "prediction_max_abs_difference": float(np.max(np.abs(shared["prediction"] - natural["prediction"]))),
                "risk_max_abs_difference": float(np.max(np.abs(shared["risk"] - natural["risk"]))),
                "raw_risk_max_abs_difference": float(np.max(np.abs(shared["raw_risk"] - natural["raw_risk"]))),
                "outputs_within_fixed_parity_tolerance": bool(np.allclose(shared["prediction"], natural["prediction"],
                    rtol=0, atol=bridge.PARITY_PRED_ATOL) and all(np.allclose(shared[key], natural[key],
                        rtol=bridge.PARITY_RISK_RTOL, atol=bridge.PARITY_RISK_ATOL) for key in ("risk", "raw_risk")))}
            require(row["natural_shared_base"]["ordered_edges_exact"]
                    and row["natural_shared_base"]["outputs_within_fixed_parity_tolerance"],
                    "Natural/shared native base graph or output differs")
        # Targets enter only after every selection/inference/timing call.
        target = np.asarray(positions[target_frame], dtype=np.float64)
        require(target.shape == current[-1].shape and np.isfinite(target).all(), "Invalid target")
        std = bridge.same.normalization(model)["std"]
        arrays.update(target_position=target, acceleration_std=std)
        errors = {}
        for method in TIMING_CASES:
            calls = [call for call in row["timed_calls"] if call["method"] == method]
            complete = method in reference and stable[method] and len(calls) == REPEATS and all(call["status"] == "complete" for call in calls)
            record = {"status": "complete" if complete else "failed", "repeat_consistent": stable[method],
                      "completed_repetitions": sum(call["status"] == "complete" for call in calls), "expected_repetitions": REPEATS,
                      "metrics": None, "timing": {key: bridge.same.timing_stats(
                          [call["timing"][key] for call in calls if call["status"] == "complete"], REPEATS)
                          for key in ("end_to_end_seconds", "score_generation_seconds", "score_graph_seconds", "score_forward_seconds",
                                      "current_graph_and_selection_seconds", "current_forward_seconds")}}
            if complete:
                output = reference[method][1]
                metrics, residual, normalized = bridge.residual_record(output, target, std)
                record.update(metrics=metrics, graph=reference[method][0]["graph"],
                              prediction_boundary=full.boundary_metrics(output["prediction"], metadata["bounds"]))
                arrays.update({"position_residual__" + method: residual, "normalized_residual__" + method: normalized})
                errors[method] = (residual ** 2).sum(-1), (normalized ** 2).sum(-1)
            row["policies"][method] = record
        row["truth_boundary"] = full.boundary_metrics(target, metadata["bounds"])
        risk_ready = POLICIES[-1] in errors
        if risk_ready:
            previous_risk = reference[POLICIES[-1]][1]["previous_base_risk"]
            arrays["previous_observed_base_risk"] = previous_risk
            if "base" in errors:
                row["correlations"]["previous_risk_vs_base_error"] = bridge.same.spearman(previous_risk, errors["base"][1])
        if "base" in errors:
            for method in POLICIES[1:]:
                if method not in errors:
                    continue
                position_benefit = errors["base"][0] - errors[method][0]
                benefit = errors["base"][1] - errors[method][1]
                arrays["signed_position_benefit__" + method] = position_benefit
                arrays["signed_normalized_benefit__" + method] = benefit
                row["benefit"][method] = {"mean_position_vector_benefit": float(position_benefit.mean()),
                    "mean_normalized_vector_benefit": float(benefit.mean()), "positive_fraction": float(np.mean(benefit > 0)),
                    "negative_fraction": float(np.mean(benefit < 0)), "zero_fraction": float(np.mean(benefit == 0))}
                if risk_ready:
                    row["correlations"]["previous_risk_vs_" + method + "_benefit"] = bridge.same.spearman(previous_risk, benefit)
                if method != "dense" and "dense" in errors:
                    dense_benefit = errors["base"][1] - errors["dense"][1]
                    row["benefit"][method]["dense_sparse_sign_disagreement_fraction"] = float(np.mean(np.sign(dense_benefit) != np.sign(benefit)))
                    row["benefit"][method]["dense_positive_sparse_nonpositive_fraction"] = float(np.mean((dense_benefit > 0) & (benefit <= 0)))
                    row["correlations"]["dense_vs_" + method + "_benefit"] = bridge.same.spearman(dense_benefit, benefit)
        row["status"] = "complete" if all(value["status"] == "complete" for value in row["policies"].values()) else "failed"
        if row["status"] != "complete":
            row["failure"] = {"category": "policy_guard_or_repeat_failure"}
    except full.RolloutGuard as error:
        row["failure"] = dict(error.details)
    except Exception as error:
        row["failure"] = {"category": "execution_error", "error_type": type(error).__name__, "error": str(error)}
    row["total_wall_seconds_including_warmup_parity_audits"] = time.perf_counter() - started
    return row, arrays


def clean_validation(native, helpers, model, positions, particle_types, item, device):
    np, torch, bridge = native.np, helpers.torch, native.bridge
    frame = item["target_frame"]
    history = np.array(positions[frame - 6:frame], dtype=np.float32, copy=True)
    types = np.asarray(particle_types, dtype=np.int64)
    if types.ndim == 0:
        types = np.full(history.shape[1], types, dtype=np.int64)
    row, arrays = {**item, "status": "failed", "failure": None, "metrics": None}, {"observed_history": history, "particle_types": types}
    try:
        native.full.check_state(history, "clean_history", bridge.MAX_ABS)
        _, edges, _, audit = native.native_graph(history, "base", None, np.random.default_rng(0))
        parity, saved, output = bridge.native_parity(model, history, types, edges, device)
        row["native_parity"] = parity
        arrays.update({"parity_" + key: value for key, value in saved.items()})
        arrays.update({"parity_supplied_" + key: output[key] for key in ("prediction", "risk", "raw_risk")})
        if not parity["passed"]:
            row["failure"] = {"category": "native_parity_failure"}
            return row, arrays
        bridge.validate_output(output, len(types))
        target = np.array(positions[frame], dtype=np.float32, copy=True)
        batch = helpers.unpack_batch([((history.transpose(1, 0, 2), types, len(types)), target)])
        with torch.no_grad():
            pred, head, target_acc = helpers.forward_batch(model, batch, torch.zeros_like(batch[0]), device)
            variance = model.head_to_variance(head)
        p, q, y = (value.detach().cpu().double().numpy() for value in (pred, variance, target_acc))
        arrays.update(target_position=target, normalized_prediction=p, raw_head=head.detach().cpu().numpy(),
                      predicted_variance=q, normalized_target=y, native_edges=edges)
        if not helpers.tensors_are_finite((pred, head, variance, target_acc)) or not bool((variance > 0).all()):
            raise native.full.RolloutGuard("nonfinite_or_nonpositive_clean_prediction_variance_target")
        se = ((p - y) ** 2).sum(-1)
        arrays["normalized_vector_se"] = se
        row.update(status="complete", graph=audit, metrics={"normalized_acceleration_coordinate_mse": float(se.mean() / 2),
            "realized_normalized_vector_se": float(se.mean()), "predicted_normalized_vector_se": float(2 * q.mean()),
            "constant_free_gaussian_nll": float((.5 * se / q + np.log(q)).mean())})
    except native.full.RolloutGuard as error:
        row["failure"] = dict(error.details)
    except Exception as error:
        row["failure"] = {"category": "execution_error", "error_type": type(error).__name__, "error": str(error)}
    return row, arrays


def aggregate(expected, rows, value):
    keys = {(item["source_index"], item["target_frame"]) for item in expected}
    actual = {(row["source_index"], row["target_frame"]) for row in rows}
    require(len(actual) == len(rows) and actual <= keys, "Duplicate/unexpected result identity")
    values = {(row["source_index"], row["target_frame"]): value(row) for row in rows}
    trajectories = {}
    for source in sorted({key[0] for key in keys}):
        wanted = sorted(key for key in keys if key[0] == source)
        available = [values.get(key) for key in wanted]
        defined = [item for item in available if item is not None and math.isfinite(item)]
        trajectories[str(source)] = {"expected": len(wanted), "defined": len(defined),
                                    "mean": sum(defined) / len(defined) if len(defined) == len(wanted) else None}
    means = [row["mean"] for row in trajectories.values()]
    return {"expected_frames": len(expected), "defined_frames": sum(row["defined"] for row in trajectories.values()),
            "expected_trajectories": len(trajectories), "defined_trajectories": sum(item is not None for item in means),
            "equal_trajectory_mean": sum(means) / len(means) if all(item is not None for item in means) else None,
            "trajectories": trajectories}


def summarize(expected, rows, mode):
    summary = {"expected_frames": len(expected), "returned_frames": len(rows),
               "complete_frames": sum(row["status"] == "complete" for row in rows),
               "failed_frames": sum(row["status"] != "complete" for row in rows),
               "aggregation": "within frame then equal frames within trajectory, then equal trajectories; missing/undefined values remain null"}
    if mode == "clean-validation":
        names = ("normalized_acceleration_coordinate_mse", "realized_normalized_vector_se", "predicted_normalized_vector_se", "constant_free_gaussian_nll")
        summary["metrics"] = {name: aggregate(expected, rows, lambda row, key=name: (row.get("metrics") or {}).get(key)) for name in names}
    else:
        summary["accuracy"] = {method: {name: aggregate(expected, rows,
            lambda row, p=method, key=name: (row.get("policies", {}).get(p, {}).get("metrics") or {}).get(key))
            for name in ("position_coordinate_mse", "normalized_coordinate_mse")} for method in POLICIES}
        summary["benefit"] = {method: {name: aggregate(expected, rows,
            lambda row, p=method, key=name: row.get("benefit", {}).get(p, {}).get(key))
            for name in (("mean_position_vector_benefit", "mean_normalized_vector_benefit", "positive_fraction", "negative_fraction", "zero_fraction")
                         + (("dense_sparse_sign_disagreement_fraction", "dense_positive_sparse_nonpositive_fraction") if method != "dense" else ()))}
            for method in POLICIES[1:]}
        summary["timing"] = {method: {name: aggregate(expected, rows,
            lambda row, p=method, key=name: row.get("policies", {}).get(p, {}).get("timing", {}).get(key, {}).get("mean")
                if row.get("policies", {}).get(p, {}).get("status") == "complete" else None)
            for name in ("end_to_end_seconds", "score_generation_seconds", "score_graph_seconds", "score_forward_seconds",
                         "current_graph_and_selection_seconds", "current_forward_seconds")} for method in TIMING_CASES}
        correlations = ("previous_risk_vs_base_error",) + tuple("previous_risk_vs_" + name + "_benefit" for name in POLICIES[1:]) + tuple("dense_vs_" + name + "_benefit" for name in POLICIES[2:])
        summary["correlations"] = {}
        for name in correlations:
            entry = aggregate(expected, rows, lambda row, key=name: row.get("correlations", {}).get(key, {}).get("value"))
            reasons = {}
            for row in rows:
                result = row.get("correlations", {}).get(name)
                if result is None or result.get("value") is None:
                    reason = "required_policy_failed_or_missing" if result is None else result["reason"]
                    reasons[reason] = reasons.get(reason, 0) + 1
            entry["undefined_returned_frame_reasons"] = reasons
            summary["correlations"][name] = entry
    return summary


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps(description(), indent=2))
        return 0
    # Cohort authorization precedes reading any evaluation manifest or array.
    cohort = json.loads(args.cohort.read_text())
    train_admission = json.loads(args.train_admission.read_text())
    check_cohort(cohort, args, train_admission)
    spec = importlib.util.spec_from_file_location("_sand_final_benchmark_helper", args.benchmark_helper)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    require(datetime.now(timezone.utc) < bench.DEADLINE, "No new work after research cutoff")
    require(cohort.get("training_schema") == bench.TRAINING_SCHEMA, "Final helper/training lineage mismatch")
    manifest, admission, structural = [json.loads(path.read_text()) for path in (args.manifest, args.admission, args.structural_report)]
    check_split(bench, manifest, admission, structural, args, sha(args.cohort))
    metadata_path = args.manifest.resolve().parent / "metadata.json"
    require(sha(metadata_path) == METADATA_SHA and json.loads(metadata_path.read_text()) == manifest["metadata"], "Actual metadata differs")
    protected = (args.manifest.resolve().parent, args.checkpoint.resolve().parent, (args.repo / "adaptive-gns").resolve())
    output = args.output_dir.resolve()
    require(all(output != path and output not in path.parents and path not in output.parents for path in protected), "Fresh output must be separate from source/data/model trees")
    output.mkdir(mode=0o700, exist_ok=False)
    started = time.perf_counter()
    rows, current = [], None
    try:
        files = (args.cohort, args.train_admission, args.manifest, args.admission, args.structural_report, args.protocol,
                 args.checkpoint, args.trainer_source, args.benchmark_helper, args.cohort_audit, metadata_path, Path(__file__),
                 args.benchmark_helper.with_name("sand_graph_support_policy.py"))
        inputs = {str(path.resolve()): sha(path) for path in files}
        native, helpers = bench.load_helpers(args.repo)
        inputs.update({str((args.repo / path).resolve()): expected for path, expected in bench.SOURCE_PINS.items()})
        trajectories = helpers.data_loader.load_manifest_data(args.manifest, verify_hashes=True)
        require(len(trajectories) == COUNT, "Missing final trajectories")
        for record, (positions, types) in zip(manifest["records"], trajectories):
            require(positions.shape == tuple(record["positions"]["shape"]) and helpers.np.isfinite(positions).all()
                    and helpers.np.all(types == 6), "Actual final source values/types differ")
            for field in ("positions", "particle_types"):
                path = helpers.data_loader._manifest_array_path(args.manifest.resolve().parent, record[field])
                require(path.stat().st_size == record[field]["size_bytes"], "Actual array byte length differs")
                inputs[str(path)] = record[field]["sha256"]
        helpers.torch.use_deterministic_algorithms(cohort["deterministic_algorithms"], warn_only=False)
        device, runtime = bench.configure_cuda(helpers, args)
        require(helpers.torch.are_deterministic_algorithms_enabled() == cohort["deterministic_algorithms"]
                and not helpers.torch.is_deterministic_algorithms_warn_only_enabled(), "Actual CUDA reduction mode differs")
        runtime.update(deterministic_algorithms=cohort["deterministic_algorithms"], deterministic_warn_only=False,
                       cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        model_args = SimpleNamespace(**vars(args), model_kind="checkpoint", checkpoint_updates=100000, training_protocol=args.protocol)
        model, identity = bench.prepare_model(helpers, model_args, manifest["metadata"], train_admission, device)
        # The new benchmark contract already verifies final100k cadence, exact
        # training protocol and deterministic provenance while loading this file.
        # Avoid deserializing the full100k graph ledger a second time.
        expected = ([{"size_group": "complete_final_split", "source_index": i, "trajectory_id": row["id"],
                      "particles": row["positions"]["shape"][1]} for i, row in enumerate(manifest["records"]) ]
                    if args.mode == "full-rollout" else schedules(manifest["records"], args.mode))
        protocol = {"schema": SCHEMA, "mode": args.mode, "split": args.split, "model": identity, "schedule": expected,
                    "policies": bench.POLICIES if args.mode == "full-rollout" else POLICIES, "timing_reference": NATURAL_BASE, "timed_repetitions": REPEATS,
                    "timing_order": "seven cyclic rotations; one untimed warmup round; standalone previous-base score pass charged",
                    "clean_validation_noise": 0.0, "runtime": runtime, "input_files_sha256": inputs,
                    "setup_seconds": time.perf_counter() - started, "source_frame_count": FRAMES,
                    "prediction_parity_atol": native.bridge.PARITY_PRED_ATOL,
                    "risk_parity_atol": native.bridge.PARITY_RISK_ATOL, "risk_parity_rtol": native.bridge.PARITY_RISK_RTOL}
        bench.atomic_json(output / "protocol.json", protocol)
        protocol_hash = sha(output / "protocol.json")
        if args.mode == "full-rollout":
            args.result_schema = SCHEMA
            rows, state = bench.run_cases(native, model, trajectories, manifest, expected, args, device, protocol_hash, started)
            for path, expected_hash in inputs.items():
                require(sha(path) == expected_hash, "Input/source changed during final rollout: " + path)
            bench.atomic_json(output / "status.json", {"state": state, "committed_outcomes": len(rows),
                "expected_outcomes": COUNT * len(bench.POLICIES), "all_inputs_reverified": True,
                "total_execution_seconds": time.perf_counter() - started, "checkpoint_selection_performed": False})
            return 0
        with helpers.torch.no_grad():
            for current in expected:
                remaining = min(args.max_seconds - (time.perf_counter() - started), (bench.DEADLINE - datetime.now(timezone.utc)).total_seconds())
                bench.atomic_json(output / "status.json", {"state": "running", "current": current, "committed_frames": len(rows)})
                with bench.deadline_alarm(remaining):
                    positions, types = trajectories[current["source_index"]]
                    if args.mode == "same-state":
                        row, arrays = same_state(native, model, positions, types, manifest["metadata"], current, args.split, args.seed, device)
                    else:
                        row, arrays = clean_validation(native, helpers, model, positions, types, current, device)
                require(all(isinstance(value, helpers.np.ndarray) and not value.dtype.hasobject for value in arrays.values()), "Numeric-only artifacts required")
                name = f"trajectory_{current['source_index']:06d}_target_{current['target_frame']:03d}"
                path = output / (name + ".npz")
                with path.with_suffix(".npz.tmp").open("xb") as stream:
                    helpers.np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                path.with_suffix(".npz.tmp").replace(path)
                row.update(artifact_file=path.name, artifact_sha256=sha(path), protocol_sha256=protocol_hash,
                           numeric_arrays={key: {"shape": list(value.shape), "dtype": value.dtype.str,
                               "value_sha256": native.full.state_hash(value)} for key, value in arrays.items()})
                bench.atomic_json(path.with_suffix(".json"), row)
                rows.append(row)
                bench.atomic_json(output / "summary.json", summarize(expected, rows, args.mode))
                print(json.dumps({**current, "status": row["status"]}), flush=True)
                if row.get("failure") and row["failure"].get("category") in ("native_parity_failure", "execution_error"):
                    raise RuntimeError("Final diagnostic implementation/parity failure retained; further inference stopped")
        for path, expected_hash in inputs.items():
            require(sha(path) == expected_hash, "Input/source changed during final diagnostics: " + path)
        state = "complete" if all(row["status"] == "complete" for row in rows) else "complete_with_guard_failures"
        bench.atomic_json(output / "status.json", {"state": state, "committed_frames": len(rows), "expected_frames": len(expected),
            "all_inputs_reverified": True, "total_execution_seconds": time.perf_counter() - started, "checkpoint_selection_performed": False})
        return 0
    except BaseException as error:
        bench.atomic_json(output / "failed_attempt.json", {"current": current, "committed_frames": len(rows),
            "error_type": type(error).__name__, "error": str(error), "all_existing_outputs_retained": True})
        bench.atomic_json(output / "status.json", {"state": "error", "committed_frames": len(rows), "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
