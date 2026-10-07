"""Particle-simulation methods and data utilities."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import torch

from research import full_rollout as full
from research.budget_graph import Candidates, random_pairs, select_pairs

POLICIES = ("base", "dense", "random25", "speed25", "previous-observed-base-risk25")
REFERENCE = "base_shared_superset"
TIMING_CASES = POLICIES + (REFERENCE,)
TARGET_FRAMES = (7, 106, 205, 304, 403, 502, 601, 700, 799, 898, 1000)
REPEATS = 6
PROTOCOL_PATH = Path(__file__).parent / "protocols/full_same_state_diagnostic.md"
TRAINING_PROTOCOL_PATH = Path(__file__).parent / "protocols/full_waterdrop_100k.md"
CORRELATIONS = ("previous_risk_vs_base_error", "previous_risk_vs_dense_benefit",
                "current_base_risk_vs_base_error", "current_base_risk_vs_dense_benefit")


def canonical_pairs(pairs):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))] if len(pairs) else pairs


def pair_set(pairs):
    return set(map(tuple, np.asarray(pairs, dtype=np.int64).reshape(-1, 2).tolist()))


def natural_candidates(position, radius, expanded, max_pairs=100000):
    """One counted tree/query; float64 classification for direct-r equivalence."""
    position = np.asarray(position, dtype=np.float64)
    query_radius = radius * full.RADIUS_FACTOR if expanded else radius
    tree = cKDTree(position)
    count = int((tree.count_neighbors(tree, query_radius) - len(position)) // 2)
    if count > max_pairs:
        raise full.RolloutGuard("candidate_pair_resource_guard", candidate_pairs=count, limit=max_pairs)
    pairs = canonical_pairs(tree.query_pairs(query_radius, output_type="ndarray"))
    if len(pairs) != count:
        raise RuntimeError("Candidate count/query disagree")
    if not expanded:
        return Candidates(pairs, np.empty((0, 2), dtype=np.int64), len(position))
    d2 = ((position[pairs[:, 0]] - position[pairs[:, 1]]) ** 2).sum(1)
    keep = d2 <= radius ** 2
    return Candidates(pairs[keep], pairs[~keep], len(position))


def spearman(x, y):
    """Average ranks, with explicit undefined rather than dropping constants."""
    x, y = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman requires matching finite vectors")
    if len(x) < 2:
        return {"value": None, "reason": "fewer_than_two_particles", "particles": len(x)}
    a, b = rankdata(x, method="average"), rankdata(y, method="average")
    a -= a.mean(); b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return {"value": None, "reason": "constant_rank_vector", "particles": len(x)}
    return {"value": float(np.clip(a.dot(b) / denominator, -1., 1.)), "reason": None, "particles": len(x)}


def cutoff_ties(graph, scores, budget):
    if scores is None or budget == 0 or len(graph.extra) == 0:
        return {"applicable": scores is not None, "cutoff_score": None,
                "tied_pairs": 0, "selected_tied_pairs": 0, "rejected_tied_pairs": 0, "boundary_tie": False}
    values = np.maximum(scores[graph.extra[:, 0]], scores[graph.extra[:, 1]])
    cutoff = float(np.sort(values)[-budget])
    tied = int(np.sum(values == cutoff))
    selected = int(budget - np.sum(values > cutoff))
    return {"applicable": True, "cutoff_score": cutoff, "tied_pairs": tied,
            "selected_tied_pairs": selected, "rejected_tied_pairs": tied - selected,
            "boundary_tie": 0 < selected < tied}


def timing_stats(values, expected):
    complete = len(values) == expected and all(math.isfinite(value) for value in values)
    return {"observations": len(values), "expected": expected,
            "mean": float(np.mean(values)) if complete else None,
            "median": float(np.median(values)) if complete else None,
            "sample_sd": float(np.std(values, ddof=1)) if complete and expected > 1 else None}


def policy_order(source_index, schedule_index, round_index):
    start = (source_index * len(TARGET_FRAMES) + schedule_index + round_index) % len(TIMING_CASES)
    return TIMING_CASES[start:] + TIMING_CASES[:start]


def normalization(model):
    stats = model._normalization_stats["acceleration"]
    result = {name: value.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(value)
              else np.asarray(value, dtype=np.float64) for name, value in stats.items()}
    if result["std"].shape != (2,) or not np.isfinite(result["std"]).all() or np.any(result["std"] <= 0):
        raise ValueError("Saved positive two-coordinate acceleration standard deviations required")
    return result


@torch.no_grad()
def run_policy(model, current, previous, particle_types, method, radius, random_seed_material,
               device="cpu", max_pairs=100000, max_abs=10.):
    """Timed predictor only; no future target or residual calculation enters."""
    if method not in TIMING_CASES:
        raise ValueError("Unknown timing case")
    full.synchronize(device)
    started = time.perf_counter()
    passes, phase, score = 0, "observed_state", None
    timing = {"score_generation_seconds": 0., "score_graph_seconds": 0., "score_forward_seconds": 0.}
    try:
        full.check_state(current, "observed_state", max_abs)
        if method == "previous-observed-base-risk25":
            full.check_state(previous, "previous_observed_state", max_abs)
            phase = "previous_score_graph"
            score_started = time.perf_counter()
            graph_started = time.perf_counter()
            old_graph = natural_candidates(previous[-1], radius, False, max_pairs)
            timing["score_graph_seconds"] = time.perf_counter() - graph_started
            phase = "previous_score_forward"
            forward_started = time.perf_counter(); passes += 1
            discarded, score = full.predict_supplied_graph(model, previous, particle_types, old_graph.base, device)
            timing["score_forward_seconds"] = time.perf_counter() - forward_started
            if not np.isfinite(discarded).all() or not np.isfinite(score).all():
                raise full.RolloutGuard("nonfinite_previous_score")
            timing["score_generation_seconds"] = time.perf_counter() - score_started
        phase = "current_graph"
        graph_started = time.perf_counter()
        if method == REFERENCE:
            graph = full.checked_candidates(np.asarray(current[-1], dtype=np.float64), radius, max_pairs)
        else:
            graph = natural_candidates(current[-1], radius, method != "base", max_pairs)
        budget = math.floor(full.FRACTION * len(graph.extra))
        if method in ("base", REFERENCE):
            pairs = graph.base
        elif method == "dense":
            pairs = np.concatenate((graph.base, graph.extra))
        elif method == "random25":
            rng = np.random.default_rng(np.random.SeedSequence(random_seed_material))
            pairs = random_pairs(graph, budget, rng)
        else:
            if method == "speed25":
                score = np.linalg.norm(current[-1] - current[-2], axis=-1)
            pairs = select_pairs(graph, score, budget)
        pairs = canonical_pairs(pairs)
        if method in POLICIES[2:] and len(pairs) - len(graph.base) != budget:
            raise RuntimeError("Optional pair budget differs from exact K")
        timing["current_graph_and_selection_seconds"] = time.perf_counter() - graph_started
        phase = "current_forward"
        forward_started = time.perf_counter(); passes += 1
        prediction, current_q = full.predict_supplied_graph(model, current, particle_types, pairs, device)
        timing["current_forward_seconds"] = time.perf_counter() - forward_started
        full.check_state(prediction, "prediction", max_abs)
        if not np.isfinite(current_q).all():
            raise full.RolloutGuard("nonfinite_current_risk")
        if prediction.shape != current[-1].shape or current_q.shape != (len(particle_types),):
            raise RuntimeError("Model output shape differs")
        full.synchronize(device)
        timing.update(end_to_end_seconds=time.perf_counter() - started, network_passes=passes,
                      candidate_builds=2 if method == "previous-observed-base-risk25" else 1)
        return {"prediction": prediction, "current_q": current_q, "selection_score": score,
                "pairs": pairs, "graph": graph, "timing": timing}
    except full.RolloutGuard as error:
        full.synchronize(device)
        error.details.update(method=method, phase=phase, attempted_network_passes=passes,
                             elapsed_seconds=time.perf_counter() - started)
        raise


def graph_audit(reference, current, radius, max_pairs):
    """Untimed verification; save both companion and frozen conventions."""
    expanded = natural_candidates(current[-1], radius, True, max_pairs)
    natural = reference["base"]["pairs"]
    shared = reference[REFERENCE]["pairs"]
    if not np.array_equal(natural, canonical_pairs(expanded.base)) or not np.array_equal(natural, shared):
        raise RuntimeError("Natural/shared float64 base pair identities differ")
    frozen = full.checked_candidates(np.asarray(current[-1], dtype=np.float32), radius, max_pairs)
    base_set, extra_set = pair_set(expanded.base), pair_set(expanded.extra)
    audits, optional_sets = {}, {}
    arrays = {"candidate_base_pairs_float64": expanded.base, "candidate_annulus_pairs_float64": expanded.extra,
              "frozen_float32_base_pairs": frozen.base, "frozen_float32_annulus_pairs": frozen.extra}
    budget = math.floor(full.FRACTION * len(expanded.extra))
    for method in TIMING_CASES:
        selected = pair_set(reference[method]["pairs"])
        if not base_set <= selected or not selected <= base_set | extra_set:
            raise RuntimeError("A policy removed mandatory pairs or admitted an out-of-range pair")
        optional_sets[method] = selected - base_set
        expected = 0 if method in ("base", REFERENCE) else len(extra_set) if method == "dense" else budget
        if len(optional_sets[method]) != expected:
            raise RuntimeError("Saved selected optional pair count differs")
        audits[method] = {"retained_pairs": len(selected), "directed_edges": 2 * len(selected),
                          "retained_optional_pairs": len(optional_sets[method]),
                          "cutoff_ties": cutoff_ties(expanded, reference[method]["selection_score"], budget)}
        arrays["pairs_" + method] = reference[method]["pairs"]
    overlaps = {}
    for left, right in itertools.combinations(POLICIES, 2):
        a, b = optional_sets[left], optional_sets[right]
        union = a | b
        overlaps[left + "__" + right] = {"optional_intersection": len(a & b), "optional_union": len(union),
                                          "optional_jaccard": len(a & b) / len(union) if union else 1.}
    return {"base_pairs": len(base_set), "annulus_pairs": len(extra_set), "candidate_pairs": len(base_set | extra_set),
            "optional_budget": budget, "policies": audits, "overlaps": overlaps,
            "natural_shared_float64_base_identical": True,
            "frozen_float32_comparison": {"base_symmetric_difference": len(base_set ^ pair_set(frozen.base)),
                                           "annulus_symmetric_difference": len(extra_set ^ pair_set(frozen.extra)),
                                           "frozen_base_pairs": len(frozen.base), "frozen_annulus_pairs": len(frozen.extra)}}, arrays


@torch.no_grad()
def evaluate_frame(model, positions, particle_types, metadata, target_frame, source_index, schedule_index,
                   seed, device="cpu", repeats=REPEATS, max_pairs=100000, max_abs=10.):
    if target_frame < 7 or target_frame >= len(positions) or repeats < 1:
        raise ValueError("Target requires both six-frame observed histories and a positive repeat count")
    current = np.array(positions[target_frame - 6:target_frame], dtype=np.float32, copy=True)
    previous = np.array(positions[target_frame - 7:target_frame - 1], dtype=np.float32, copy=True)
    types = np.array(particle_types, dtype=np.int64, copy=True)
    if types.ndim == 0:
        types = np.full(current.shape[1], types, dtype=np.int64)
    if current.ndim != 3 or current.shape[0] != 6 or current.shape[-1] != 2 or types.shape != (current.shape[1],) or np.any(types == 3):
        raise ValueError("Six-frame 2D WATERDROP history with no kinematic particles required")
    radius = float(metadata["default_connectivity_radius"])
    if not math.isclose(radius, float(model._connectivity_radius), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Checkpoint and metadata radii differ")
    scales = normalization(model)
    material = [93000, int(seed), int(source_index), int(target_frame)]
    row = {"target_frame": target_frame, "source_index": source_index, "n_particles": len(types),
           "status": "failed", "failure": None, "random_seed_material": material,
           "observed_history_sha256": full.state_hash(current), "previous_observed_history_sha256": full.state_hash(previous),
           "model_history_dtype": str(current.dtype), "radius_classification_dtype": "float64",
           "saved_acceleration_normalization": {key: value.tolist() for key, value in scales.items()},
           "warmup_calls": [], "timed_calls": [], "accuracy": None, "benefit": None, "correlations": None,
           "graph_audit": None, "timing_summary": None, "repeat_consistency": None}
    arrays = {"current_observed_history": current, "previous_observed_history": previous, "particle_types": types}
    reference, hashes, max_differences = {}, {case: [] for case in TIMING_CASES}, {case: 0. for case in TIMING_CASES}
    started = time.perf_counter()
    model.eval()
    try:
        for round_index in range(-1, repeats):
            warmup = round_index == -1
            for slot, method in enumerate(policy_order(source_index, schedule_index, max(round_index, 0))):
                call = {"method": method, "round": round_index, "slot": slot, "warmup": warmup, "status": "failed"}
                container = row["warmup_calls"] if warmup else row["timed_calls"]
                try:
                    output = run_policy(model, current, previous, types, method, radius, material, device, max_pairs, max_abs)
                except full.RolloutGuard as error:
                    error.details.update(round=round_index, slot=slot, warmup=warmup)
                    call["failure"] = error.details
                    container.append(call)
                    raise
                call.update(status="complete", **output["timing"], pair_sha256=full.state_hash(output["pairs"]))
                container.append(call)
                if warmup:
                    continue
                hashes[method].append(call["pair_sha256"])
                if method not in reference:
                    reference[method] = output
                else:
                    max_differences[method] = max(max_differences[method], float(np.max(np.abs(
                        output["prediction"].astype(np.float64) - reference[method]["prediction"].astype(np.float64)))))
        # The target first enters the computation after all predictor calls.
        audit_started = time.perf_counter()
        target = np.asarray(positions[target_frame], dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("Nonfinite target coordinates")
        row["target_sha256"] = full.state_hash(target)
        arrays["target_position"] = target
        row["graph_audit"], pair_arrays = graph_audit(reference, current, radius, max_pairs)
        arrays.update(pair_arrays)
        accuracy, position_se, normalized_se = {}, {}, {}
        for method in POLICIES:
            prediction = reference[method]["prediction"].astype(np.float64)
            residual = prediction - target
            position_se[method] = np.sum(residual ** 2, axis=-1)
            normalized_se[method] = np.sum((residual / scales["std"]) ** 2, axis=-1)
            accuracy[method] = {"position_coordinate_mse": float(position_se[method].mean() / 2),
                                "normalized_coordinate_mse": float(normalized_se[method].mean() / 2)}
            arrays["prediction_" + method] = prediction
            arrays["position_vector_se_" + method] = position_se[method]
            arrays["normalized_vector_se_" + method] = normalized_se[method]
        q_previous = reference["previous-observed-base-risk25"]["selection_score"]
        q_current = reference["base"]["current_q"]
        benefit = normalized_se["base"] - normalized_se["dense"]
        position_benefit = position_se["base"] - position_se["dense"]
        arrays.update(previous_observed_base_q=q_previous, current_base_q=q_current,
                      signed_dense_benefit_normalized=benefit, signed_dense_benefit_position=position_benefit)
        row["accuracy"] = accuracy
        row["benefit"] = {"mean_normalized_vector_benefit": float(benefit.mean()),
                          "mean_position_vector_benefit": float(position_benefit.mean()),
                          "positive_fraction": float(np.mean(benefit > 0)), "negative_fraction": float(np.mean(benefit < 0)),
                          "zero_fraction": float(np.mean(benefit == 0))}
        row["correlations"] = dict(zip(CORRELATIONS,
            [spearman(q_previous, normalized_se["base"]), spearman(q_previous, benefit),
             spearman(q_current, normalized_se["base"]), spearman(q_current, benefit)]))
        row["timing_summary"] = {method: {name: timing_stats([call[name] for call in row["timed_calls"] if call["method"] == method], repeats)
            for name in ("end_to_end_seconds", "score_generation_seconds", "score_graph_seconds", "score_forward_seconds",
                         "current_graph_and_selection_seconds", "current_forward_seconds")}
            for method in TIMING_CASES}
        row["repeat_consistency"] = {method: {"distinct_pair_hashes": len(set(hashes[method])),
                                              "maximum_absolute_prediction_difference": max_differences[method]}
                                     for method in TIMING_CASES}
        row["natural_shared_base_maximum_prediction_difference"] = float(np.max(np.abs(
            reference["base"]["prediction"].astype(np.float64) - reference[REFERENCE]["prediction"].astype(np.float64))))
        row.update(status="complete", residual_and_graph_audit_seconds=time.perf_counter() - audit_started)
    except full.RolloutGuard as error:
        row["failure"] = dict(error.details)
    row["total_wall_seconds_including_warmups_and_audits"] = time.perf_counter() - started
    row["completed_network_passes"] = sum(call["network_passes"] for call in row["warmup_calls"] + row["timed_calls"] if call["status"] == "complete")
    row["failed_call_attempted_network_passes"] = row["failure"].get("attempted_network_passes", 0) if row["failure"] else 0
    return row, arrays


def summarize_frames(rows, expected):
    expected_keys = {(item["source_index"], item["target_frame"]): item["trajectory_id"] for item in expected}
    if not expected_keys or len(expected_keys) != len(expected):
        raise ValueError("Expected frame identities must be nonempty and unique")
    keys = [(row["source_index"], row["target_frame"]) for row in rows]
    if len(keys) != len(set(keys)) or any(key not in expected_keys for key in keys):
        raise ValueError("Duplicate or unexpected diagnostic frame")
    if any(row["trajectory_id"] != expected_keys[key] for row, key in zip(rows, keys)):
        raise ValueError("Diagnostic trajectory identity differs")
    completed = [row for row in rows if row["status"] == "complete"]
    all_complete = len(completed) == len(expected)
    sources = sorted({key[0] for key in expected_keys})
    def aggregate(values):
        valid = all_complete and len(values) == len(expected) and all(value is not None for value in values)
        if not valid:
            return {"equal_frame_mean": None, "equal_trajectory_mean": None, "particle_weighted_mean": None}
        per_trajectory = [float(np.mean([value for row, value in zip(rows, values) if row["source_index"] == source])) for source in sources]
        return {"equal_frame_mean": float(np.mean(values)), "equal_trajectory_mean": float(np.mean(per_trajectory)),
                "particle_weighted_mean": float(np.average(values, weights=[row["n_particles"] for row in rows]))}
    accuracy = {method: {metric: aggregate([row["accuracy"][method][metric] if row["status"] == "complete" else None for row in rows])
                for metric in ("position_coordinate_mse", "normalized_coordinate_mse")} for method in POLICIES}
    benefits = {name: aggregate([row["benefit"][name] if row["status"] == "complete" else None for row in rows])
                for name in ("mean_normalized_vector_benefit", "mean_position_vector_benefit", "positive_fraction", "negative_fraction", "zero_fraction")}
    correlations = {}
    for name in CORRELATIONS:
        values = [row["correlations"][name]["value"] if row["status"] == "complete" else None for row in rows]
        means = aggregate(values)
        correlations[name] = {"defined_completed_frames": sum(value is not None for value in values),
                              "undefined_completed_frames": sum(row["correlations"][name]["value"] is None for row in completed),
                              "equal_trajectory_mean_of_frame_spearman": means["equal_trajectory_mean"],
                              "pooled_particle_spearman": None, "pooled_particle_spearman_computed": False}
    timing = {method: {name: aggregate([row["timing_summary"][method][name]["mean"] if row["status"] == "complete" else None for row in rows])
               for name in ("end_to_end_seconds", "score_generation_seconds", "current_graph_and_selection_seconds", "current_forward_seconds")}
               for method in TIMING_CASES}
    return {"required_frames": len(expected), "attempted_frames": len(rows), "completed_frames": len(completed),
            "failed_frames": len(rows) - len(completed), "missing_frames": [item for item in expected if (item["source_index"], item["target_frame"]) not in keys],
            "all_required_frames_completed": all_complete, "accuracy_and_runtime_aggregates_defined": all_complete,
            "aggregation": "particles within frame; equal frames within trajectory then equal trajectories; particle-weighted alternative explicitly separate",
            "accuracy": accuracy, "signed_dense_benefit": benefits, "correlations": correlations, "timing": timing,
            "failures": [{key: row[key] for key in ("source_index", "target_frame", "trajectory_id", "failure")} for row in rows if row["status"] == "failed"]}


COMPACT_FIELDS = ("source_index", "target_frame", "trajectory_id", "n_particles", "status", "failure", "accuracy", "benefit",
                  "correlations", "timing_summary", "graph_audit", "array_file", "array_sha256")


def compact_frame(row, path):
    return {**{key: row[key] for key in COMPACT_FIELDS}, "record_file": path.name, "record_sha256": full.sha256(path)}


def recover_frame(directory, item, protocol_hash, indexed=None):
    stem = f"trajectory_{item['source_index']:06d}_target_{item['target_frame']:04d}"
    path, array_path = directory / (stem + ".json"), directory / (stem + ".npz")
    if not path.exists():
        if indexed is not None:
            raise ValueError("Indexed diagnostic record is missing")
        return None
    row = json.loads(path.read_text())
    if any(row.get(key) != item[key] for key in ("source_index", "target_frame", "trajectory_id")) or row.get("protocol_sha256") != protocol_hash:
        raise ValueError("Diagnostic record identity/protocol differs")
    if row.get("status") not in ("complete", "failed") or (row["status"] == "complete") != (row.get("failure") is None):
        raise ValueError("Diagnostic completion/failure record is inconsistent")
    if row.get("array_file") != array_path.name or not array_path.exists() or full.sha256(array_path) != row.get("array_sha256"):
        raise ValueError("Diagnostic array file/hash differs")
    compact = compact_frame(row, path)
    if indexed is not None and compact != indexed:
        raise ValueError("Diagnostic record hash/index differs")
    return compact


def evaluate_to_directory(args, protocol, model, trajectories, manifest, selected, metadata, seed, device):
    """Caller owns RunLock. A committed failed frame is reused on resume."""
    directory = args.output_dir
    protocol_path, result_path = directory / "protocol.json", directory / "result.json"
    indexed = {}
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Resume requires exactly matching diagnostic protocol/source/data/checkpoint/runtime hashes")
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved.get("protocol_sha256") != full.sha256(protocol_path):
                raise ValueError("Saved diagnostic result protocol hash differs")
            for row in saved["records"]:
                key = (row["source_index"], row["target_frame"])
                if key in indexed:
                    raise ValueError("Duplicate saved diagnostic frame")
                indexed[key] = row
    elif args.resume or result_path.exists():
        raise ValueError("Original diagnostic protocol is required; existing artifacts must be preserved")
    else:
        full.atomic_json(protocol_path, protocol)
    digest = full.sha256(protocol_path)
    expected = [{"source_index": index, "trajectory_id": manifest["records"][index]["id"], "target_frame": target}
                for index in selected for target in args.target_frames]
    expected_keys = {(item["source_index"], item["target_frame"]) for item in expected}
    if not set(indexed) <= expected_keys:
        raise ValueError("Unexpected saved diagnostic frame")
    recovered = {}
    for item in expected:
        key = (item["source_index"], item["target_frame"])
        row = recover_frame(directory, item, digest, indexed.get(key))
        if row is not None:
            if not args.resume:
                raise ValueError("Existing diagnostic frame requires --resume")
            recovered[key] = row
    rows = list(recovered.values())
    full.atomic_json(directory / "status.json", {"state": "running", "reused_frames": len(recovered), "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        for index in selected:
            if all((index, target) in recovered for target in args.target_frames):
                continue
            positions, types = trajectories[index]
            for schedule_index, target in enumerate(args.target_frames):
                if (index, target) in recovered:
                    continue
                row, arrays = evaluate_frame(model, positions, types, metadata, target, index, schedule_index, seed, device,
                                             args.repeats, args.max_candidate_pairs, args.max_abs_coordinate)
                stem = f"trajectory_{index:06d}_target_{target:04d}"
                array_path = directory / (stem + ".npz")
                with array_path.with_suffix(".npz.tmp").open("wb") as stream:
                    np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
                array_path.with_suffix(".npz.tmp").replace(array_path)
                row.update(trajectory_id=manifest["records"][index]["id"], protocol_sha256=digest,
                           array_file=array_path.name, array_sha256=full.sha256(array_path))
                path = directory / (stem + ".json")
                full.atomic_json(path, row)
                rows.append(compact_frame(row, path))
                full.atomic_json(result_path, {"state": "partial", "protocol_sha256": digest, "records": rows,
                                              "summary": summarize_frames(rows, expected)})
                print(json.dumps({"source_index": index, "target_frame": target, "status": row["status"]}), flush=True)
        rows.sort(key=lambda row: (row["source_index"], row["target_frame"]))
        full.atomic_json(result_path, {"state": "complete", "protocol_sha256": digest, "checkpoint_sha256": protocol["checkpoint_sha256"],
                                      "scope": args.scope, "seed": seed, "objective": protocol["objective"], "reused_frames": len(recovered),
                                      "records": rows, "summary": summarize_frames(rows, expected)})
        full.atomic_json(directory / "status.json", {"state": "complete", "result_sha256": full.sha256(result_path), "reused_frames": len(recovered)})
    except BaseException as error:
        full.atomic_json(directory / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                         "error_type": type(error).__name__, "error": str(error), "committed_frames": len(rows)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot_validation", "locked_test"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--training-protocol", type=Path, default=TRAINING_PROTOCOL_PATH)
    parser.add_argument("--target-frames", type=int, nargs="+", default=list(TARGET_FRAMES))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--max-candidate-pairs", type=int, default=100000)
    parser.add_argument("--max-abs-coordinate", type=float, default=10.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    args = parser.parse_args()
    if (args.threads < 1 or args.repeats < 1 or args.max_candidate_pairs < 1 or args.max_abs_coordinate <= 0
            or not math.isfinite(args.max_abs_coordinate) or min(args.target_frames) < 7
            or args.target_frames != sorted(set(args.target_frames))):
        parser.error("Positive threads/repeats and sorted unique target frames >=7 required")
    if args.scope == "locked_test" and (tuple(args.target_frames) != TARGET_FRAMES or args.repeats != REPEATS
            or args.max_trajectories is not None or args.max_candidate_pairs != 100000 or args.max_abs_coordinate != 10.):
        parser.error("Locked diagnostic requires every fixed frame/trajectory and the fixed timing/guard settings")
    torch.set_num_threads(args.threads)
    device = full.resolve_device(args.device)
    manifest = json.loads(args.manifest.read_text())
    selected = full.select_records(manifest, args.scope, max_trajectories=args.max_trajectories)
    trajectories = full.load_manifest_data(args.manifest, verify_hashes=True)
    metadata_path = args.manifest.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if manifest["metadata"] != metadata:
        raise ValueError("Manifest metadata differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    full.check_checkpoint(payload, args.scope, args.training_protocol, full.sha256(metadata_path))
    model, provenance = full.load_for_evaluation(args.checkpoint, metadata, device, radius_backend="scipy_host")
    if np.asarray(model._boundaries).tolist() != metadata["bounds"]:
        raise ValueError("Checkpoint boundaries differ from metadata")
    seed = payload.get("run_config", {}).get("seed", payload.get("training_config", {}).get("seed"))
    if type(seed) is not int or seed < 0:
        raise ValueError("Explicit nonnegative training seed required")
    source_paths = [Path(__file__), Path(full.__file__), full.REPO / "research/budget_graph.py", full.REPO / "research/full_training.py"]
    source_paths += sorted((full.REPO / "adaptive-gns/gns").glob("*.py"))
    protocol = {"schema": 1, "scope": args.scope, "split": manifest["split"], "seed": seed,
                "objective": payload["training_config"]["loss"], "companion_protocol_sha256": full.sha256(args.protocol),
                "training_protocol_sha256": full.sha256(args.training_protocol), "checkpoint_sha256": full.sha256(args.checkpoint),
                "checkpoint_provenance": provenance, "manifest_sha256": full.sha256(args.manifest),
                "metadata_sha256": full.sha256(metadata_path), "source_tfrecord": manifest["source"],
                "trajectory_records": [manifest["records"][index] for index in selected],
                "target_frames": args.target_frames, "policies": list(POLICIES), "timing_cases": list(TIMING_CASES),
                "timed_rounds": args.repeats, "warmup_rounds": 1, "timing_order": "cyclic balanced rotation; six rounds in locked scope",
                "score_history": "previous observed six-frame base graph, one timestep older; full score pass included every call",
                "random_subset": "same subset every repeat; SeedSequence([93000,training_seed,source_index,target_frame])",
                "graph_convention": "float64 radius classification, lexicographic pair order, uncapped symmetric no self loops; model histories float32",
                "base_timing": "natural radius-r candidate search; separate base_shared_superset computational reference; no cached amortization claim",
                "guards": {"max_candidate_pairs": args.max_candidate_pairs, "max_absolute_coordinate": args.max_abs_coordinate},
                "code_sha256": {str(path.relative_to(full.REPO)): full.sha256(path) for path in source_paths},
                "runtime": full.runtime_provenance(device, "scipy_host"), "software": {"numpy": np.__version__, "scipy": full.scipy.__version__,
                "python": full.platform.python_version(), "platform": full.platform.platform()}, "threads": args.threads}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with full.RunLock(args.output_dir, clear_stale=args.clear_stale_lock):
        evaluate_to_directory(args, protocol, model, trajectories, manifest, selected, metadata, seed, device)


if __name__ == "__main__":
    main()
