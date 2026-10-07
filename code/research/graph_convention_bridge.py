"""Particle-simulation methods and data utilities."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from research import full_rollout as full
from research import full_same_state as same
from research.budget_graph import Candidates, random_pairs, select_pairs

PROTOCOL = Path(__file__).parent / "protocols/graph_convention_bridge_20261005.md"
POLICIES = ("base", "dense", "random25", "speed25",
            "previous-observed-base-risk25", "current-base-risk25")
FACTORIAL = ("base_cap128_loops0", "base_cap128_loops1",
             "base_uncapped_loops0", "base_uncapped_loops1",
             "dense_cap128_loops0", "dense_cap128_loops1")
CASE_NAMES = FACTORIAL + tuple(f"{p}_uncapped_loops{loop}" for loop in (0, 1) for p in POLICIES[1:])
MAX_PAIRS = 100000
MAX_ABS = 10.
PARITY_PRED_ATOL = 2e-7
PARITY_RISK_ATOL = 1e-6
PARITY_RISK_RTOL = 1e-5


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def strict_pairs(position, radius):
    """Native float32 norm and strict radius, symmetric pair universe."""
    points = np.asarray(position, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or radius <= 0:
        raise ValueError("Finite 2D points and positive radius required")
    tree = cKDTree(points)
    count = int((tree.count_neighbors(tree, radius * full.RADIUS_FACTOR) - len(points)) // 2)
    if count > MAX_PAIRS:
        raise full.RolloutGuard("candidate_pair_resource_guard", candidate_pairs=count, limit=MAX_PAIRS)
    # Query matches native cKDTree candidates; its subsequent norm is float32.
    def query(r):
        found = same.canonical_pairs(tree.query_pairs(r, output_type="ndarray"))
        distance = np.linalg.norm(points[found[:, 0]] - points[found[:, 1]], axis=1)
        return found[distance < r]
    base, expanded = query(radius), query(radius * full.RADIUS_FACTOR)
    base_set = same.pair_set(base)
    extra = np.asarray([pair for pair in expanded if tuple(pair) not in base_set], dtype=np.int64).reshape(-1, 2)
    if not base_set <= same.pair_set(expanded):
        raise RuntimeError("Strict radius pair sets are unexpectedly nonnested")
    return Candidates(base, extra, len(points))


def ordered_edges(position, pairs, loops, cap=None):
    """Receiver-major, distance then source-ID order; exactly one self edge.

    Cap is only used for base/dense factorial cases. It includes the self edge,
    exactly as native training does, and may create asymmetric directed edges.
    No policy applies a post-selection cap.
    """
    points = np.asarray(position, dtype=np.float32)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    n = len(points)
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("Cap must be a positive integer or None")
    if len(pairs) and (np.any(pairs[:, 0] >= pairs[:, 1]) or np.any(pairs < 0) or np.any(pairs >= n)
                       or len(np.unique(pairs, axis=0)) != len(pairs)):
        raise ValueError("Unique canonical non-self pairs required")
    edges = np.concatenate((pairs.T, pairs[:, ::-1].T), axis=1)
    if loops:
        ids = np.arange(n, dtype=np.int64)
        edges = np.concatenate((edges, np.stack((ids, ids))), axis=1)
    distances = np.linalg.norm(points[edges[0]] - points[edges[1]], axis=1)
    order = np.lexsort((edges[0], distances, edges[1]))
    edges = edges[:, order]
    if cap is not None and edges.shape[1]:
        _, first = np.unique(edges[1], return_index=True)
        within_receiver = np.arange(edges.shape[1]) - np.repeat(first, np.diff(np.r_[first, edges.shape[1]]))
        edges = edges[:, within_receiver < cap]
    return np.ascontiguousarray(edges)


def tensors(history, types, device):
    sequence = torch.as_tensor(np.ascontiguousarray(history.transpose(1, 0, 2)), dtype=torch.float32, device=device)
    particle_types = torch.as_tensor(types, dtype=torch.long, device=device)
    counts = torch.tensor([len(types)], dtype=torch.long, device=device)
    return sequence, counts, particle_types


def output_arrays(model, prediction, raw_risk):
    # Preserve values before guards, including nonfinite failed outputs.
    prediction = prediction.detach().cpu().numpy().copy()
    raw = raw_risk.detach().cpu().numpy().reshape(-1).copy()
    risk = model.head_to_variance(raw_risk).detach().cpu().numpy().reshape(-1).copy()
    return {"prediction": prediction, "raw_risk": raw, "risk": risk}


def validate_output(output, n):
    if output["prediction"].shape != (n, 2) or output["risk"].shape != (n,):
        raise RuntimeError("Predictor shape differs from history")
    full.check_state(output["prediction"], "prediction", MAX_ABS)
    if not np.isfinite(output["raw_risk"]).all() or not np.isfinite(output["risk"]).all():
        raise full.RolloutGuard("nonfinite_risk")
    if np.any(output["risk"] <= 0):
        raise full.RolloutGuard("nonpositive_risk")


@torch.no_grad()
def supplied(model, history, types, edges, device, capture=False):
    sequence, counts, particle_types = tensors(history, types, device)
    edge = torch.as_tensor(edges, dtype=torch.long, device=device)
    features = []
    handle = None
    if capture:
        def save_inputs(_module, args):
            features.extend(value.detach().cpu().numpy().copy() for value in args)
        handle = model._encode_process_decode.register_forward_pre_hook(save_inputs)
    full.synchronize(device)
    started = time.perf_counter()
    try:
        prediction, raw = model._forward_with_edge_index(sequence, counts, particle_types, edge[0], edge[1], None)
        full.synchronize(device)
        result = output_arrays(model, prediction, raw)
    finally:
        if handle is not None:
            handle.remove()
    result["operational_seconds"] = time.perf_counter() - started
    if capture:
        if len(features) != 3:
            raise RuntimeError("Expected exactly one three-input network call")
        result["features"] = features
    return result


@torch.no_grad()
def native_parity(model, history, types, expected_edges, device):
    """Native graph/feature identity and forward agreement before policies."""
    sequence, counts, particle_types = tensors(history, types, device)
    native_features = []
    def save_inputs(_module, args):
        native_features.extend(value.detach().cpu().numpy().copy() for value in args)
    handle = model._encode_process_decode.register_forward_pre_hook(save_inputs)
    full.synchronize(device)
    started = time.perf_counter()
    try:
        pred, raw = model.predict_positions_with_variance(sequence, counts, particle_types)
        full.synchronize(device)
        native = output_arrays(model, pred, raw)
    finally:
        handle.remove()
    native["operational_seconds"] = time.perf_counter() - started
    if len(native_features) != 3:
        raise RuntimeError("Expected exactly one native three-input network call")
    graph_equal = np.array_equal(native_features[1], expected_edges)
    supplied_output = supplied(model, history, types, expected_edges, device, capture=True)
    feature_equal = [np.array_equal(a, b) for a, b in zip(native_features, supplied_output["features"])]
    finite = all(np.isfinite(native[key]).all() and np.isfinite(supplied_output[key]).all()
                 for key in ("prediction", "risk", "raw_risk"))
    agreement = finite and np.allclose(native["prediction"], supplied_output["prediction"], rtol=0, atol=PARITY_PRED_ATOL)
    agreement = bool(agreement and np.allclose(native["risk"], supplied_output["risk"],
                                             rtol=PARITY_RISK_RTOL, atol=PARITY_RISK_ATOL))
    agreement = bool(agreement and np.allclose(native["raw_risk"], supplied_output["raw_risk"],
                                             rtol=PARITY_RISK_RTOL, atol=PARITY_RISK_ATOL))
    difference = lambda key: float(np.max(np.abs(native[key] - supplied_output[key]))) if finite else None
    audit = {"native_edge_identity": graph_equal, "native_supplied_feature_identity": feature_equal,
             "prediction_max_abs_difference": difference("prediction"), "risk_max_abs_difference": difference("risk"),
             "raw_risk_max_abs_difference": difference("raw_risk"),
             "prediction_atol": PARITY_PRED_ATOL, "risk_atol": PARITY_RISK_ATOL, "risk_rtol": PARITY_RISK_RTOL,
             "finite": finite, "prediction_risk_agree": agreement,
             "passed": bool(graph_equal and all(feature_equal) and agreement)}
    arrays = {"native_prediction": native["prediction"], "native_raw_risk": native["raw_risk"],
              "native_risk": native["risk"], "native_edges": native_features[1],
              "native_node_features": native_features[0], "native_edge_features": native_features[2],
              "supplied_node_features": supplied_output["features"][0],
              "supplied_edge_features": supplied_output["features"][2]}
    supplied_output.pop("features")
    return audit, arrays, supplied_output


def graph_record(edges, n, base_pairs, optional_pairs):
    degree = np.bincount(edges[1], minlength=n)
    return {"directed_edges": int(edges.shape[1]), "edge_sha256": full.state_hash(edges),
            "self_edges": int(np.sum(edges[0] == edges[1])), "mandatory_unordered_pairs": int(base_pairs),
            "selected_optional_pairs": int(optional_pairs), "max_receiver_degree": int(degree.max()),
            "receiver_degree_mean": float(degree.mean())}


def residual_record(output, target, std):
    residual = output["prediction"].astype(np.float64) - target
    normalized = residual / std
    return {"position_coordinate_mse": float(np.mean(residual ** 2)),
            "normalized_coordinate_mse": float(np.mean(normalized ** 2))}, residual, normalized


@torch.no_grad()
def evaluate_frame(model, current, previous, types, target, identity, seed, device):
    """Targets are consulted only after every graph and predictor call."""
    started = time.perf_counter()
    current, previous = np.asarray(current, dtype=np.float32), np.asarray(previous, dtype=np.float32)
    types = np.asarray(types, dtype=np.int64)
    if current.shape != previous.shape or current.ndim != 3 or current.shape[0] != 6 or current.shape[2] != 2:
        raise ValueError("Two six-frame 2D observed histories required")
    if types.shape != (current.shape[1],) or np.any(types == 3):
        raise ValueError("One non-kinematic particle type per node required")
    full.check_state(current, "current_history", MAX_ABS)
    full.check_state(previous, "previous_history", MAX_ABS)
    n, radius = len(types), float(model._connectivity_radius)
    graph = strict_pairs(current[-1], radius)
    old_graph = strict_pairs(previous[-1], radius)
    budget = math.floor(full.FRACTION * len(graph.extra))
    material = [20261005, 771, seed, 0 if identity["split"] == "valid" else 1,
                identity["source_index"], identity["target_frame"]]
    std = same.normalization(model)["std"]
    arrays = {"current_history": current, "previous_history": previous, "particle_types": types,
              "strict_base_pairs": graph.base, "strict_annulus_pairs": graph.extra,
              "previous_strict_base_pairs": old_graph.base, "acceleration_std": std}
    row = {**identity, "status": "running", "failure": None, "n_particles": n,
           "current_history_sha256": full.state_hash(current), "previous_history_sha256": full.state_hash(previous),
           "random_seed_material": material, "optional_budget": budget, "cases": {}, "native_parity": None}
    outputs, cached_edges = {}, {}
    native_edges = ordered_edges(current[-1], graph.base, True, 128)
    parity, parity_arrays, native_supplied = native_parity(model, current, types, native_edges, device)
    arrays.update(parity_arrays)
    row["native_parity"] = parity
    # Save raw parity outputs and stop the model run if native identity fails.
    if not parity["passed"]:
        row.update(status="failed", failure={"category": "native_parity_failure"})
        arrays.update(supplied_parity_prediction=native_supplied["prediction"],
                      supplied_parity_risk=native_supplied["risk"], supplied_parity_raw_risk=native_supplied["raw_risk"])
        row["operational_seconds"] = time.perf_counter() - started
        return row, arrays

    def current_prediction(name, edges):
        digest = full.state_hash(edges)
        if digest in cached_edges:
            original, output = cached_edges[digest]
            return {**output, "reused_from_case": original}
        output = native_supplied if name == "base_cap128_loops1" else supplied(model, current, types, edges, device)
        cached_edges[digest] = (name, output)
        return output

    def save_case(name, output, edges, optional, passes=1, score_seconds=0.):
        arrays[f"edges__{name}"] = edges
        for key in ("prediction", "raw_risk", "risk"):
            arrays[f"{key}__{name}"] = output[key]
        case = {"status": "complete", "failure": None,
                "graph": graph_record(edges, n, len(graph.base), optional),
                "network_passes_for_standalone_policy": passes, "scoring_seconds": score_seconds,
                "forward_operational_seconds": output["operational_seconds"],
                "reused_from_case": output.get("reused_from_case"), "metrics": None}
        try:
            validate_output(output, n)
        except full.RolloutGuard as error:
            case.update(status="failed", failure=error.details)
        row["cases"][name] = case
        outputs[name] = output
        return case

    for cap_name, cap in (("cap128", 128), ("uncapped", None)):
        for loops in (0, 1):
            name = f"base_{cap_name}_loops{loops}"
            edges = ordered_edges(current[-1], graph.base, bool(loops), cap)
            output = current_prediction(name, edges)
            save_case(name, output, edges, 0)
    all_pairs = np.concatenate((graph.base, graph.extra))
    for loops in (0, 1):
        name = f"dense_cap128_loops{loops}"
        edges = ordered_edges(current[-1], all_pairs, bool(loops), 128)
        output = current_prediction(name, edges)
        # A directed cap can truncate optional pairs asymmetrically. Do not
        # claim an exact undirected retained count for this factorial control.
        save_case(name, output, edges, len(graph.extra))
        row["cases"][name]["graph"]["selected_optional_pairs"] = None
        row["cases"][name]["graph"]["available_optional_pairs_before_cap"] = len(graph.extra)
    for name in FACTORIAL:
        if "cap128" in name:
            audit = row["cases"][name]["graph"]
            audit["base_unordered_pairs_before_cap"] = audit.pop("mandatory_unordered_pairs")
    # Cap diagnostics are independent of errors and keep directed asymmetry visible.
    row["base_graph_audit"] = {}
    for loops in (0, 1):
        uncapped = arrays[f"edges__base_uncapped_loops{loops}"]
        capped = arrays[f"edges__base_cap128_loops{loops}"]
        degree = np.bincount(uncapped[1], minlength=n)
        row["base_graph_audit"][f"loops{loops}"] = {
            "receivers_above_cap": int(np.sum(degree > 128)),
            "edges_removed_by_cap": int(uncapped.shape[1] - capped.shape[1]),
            "capped_equals_uncapped": bool(np.array_equal(capped, uncapped))}
        dense_edges = ordered_edges(current[-1], all_pairs, bool(loops))
        dense_capped = arrays[f"edges__dense_cap128_loops{loops}"]
        dense_degree = np.bincount(dense_edges[1], minlength=n)
        row["base_graph_audit"][f"dense_loops{loops}"] = {
            "receivers_above_cap": int(np.sum(dense_degree > 128)),
            "edges_removed_by_cap": int(dense_edges.shape[1] - dense_capped.shape[1]),
            "capped_equals_uncapped": bool(np.array_equal(dense_edges, dense_capped))}
    frozen = full.checked_candidates(current[-1], radius, MAX_PAIRS)
    row["locked_graph_comparison"] = {
        "base_pair_symmetric_difference": len(same.pair_set(graph.base) ^ same.pair_set(frozen.base)),
        "annulus_pair_symmetric_difference": len(same.pair_set(graph.extra) ^ same.pair_set(frozen.extra))}
    arrays.update(locked_base_pairs=frozen.base, locked_annulus_pairs=frozen.extra)
    for loops in (0, 1):
        old_edges = ordered_edges(previous[-1], old_graph.base, bool(loops))
        old_output = supplied(model, previous, types, old_edges, device)
        arrays[f"previous_base_edges_loops{loops}"] = old_edges
        for key in ("prediction", "raw_risk", "risk"):
            arrays[f"previous_base_{key}_loops{loops}"] = old_output[key]
        old_failure = None
        try:
            validate_output(old_output, n)
        except full.RolloutGuard as error:
            old_failure = error.details
        row[f"previous_score_loops{loops}"] = {"status": "failed" if old_failure else "complete",
            "failure": old_failure, "operational_seconds": old_output["operational_seconds"],
            "max_receiver_degree": int(np.bincount(old_edges[1], minlength=n).max()),
            "receivers_above_cap": int(np.sum(np.bincount(old_edges[1], minlength=n) > 128))}
        base_name = f"base_uncapped_loops{loops}"
        for policy in POLICIES[1:]:
            name = f"{policy}_uncapped_loops{loops}"
            scoring_failure = old_failure if policy == "previous-observed-base-risk25" else (
                row["cases"][base_name]["failure"] if policy == "current-base-risk25" else None)
            if scoring_failure:
                row["cases"][name] = {"status": "failed", "failure": {"category": "scoring_pass_failed", "cause": scoring_failure}, "metrics": None}
                continue
            score, passes, score_seconds = None, 1, 0.
            if policy == "dense":
                pairs = np.concatenate((graph.base, graph.extra))
            elif policy == "random25":
                pairs = random_pairs(graph, budget, np.random.default_rng(np.random.SeedSequence(material)))
            else:
                if policy == "speed25":
                    score = np.linalg.norm(current[-1] - current[-2], axis=1)
                elif policy == "previous-observed-base-risk25":
                    score, passes, score_seconds = old_output["risk"], 2, old_output["operational_seconds"]
                else:
                    score, passes, score_seconds = outputs[base_name]["risk"], 2, outputs[base_name]["operational_seconds"]
                pairs = select_pairs(graph, score, budget)
            optional = len(pairs) - len(graph.base)
            if optional != (len(graph.extra) if policy == "dense" else budget):
                raise RuntimeError("Exact optional pair budget violated")
            if not same.pair_set(graph.base) <= same.pair_set(pairs) <= same.pair_set(np.concatenate((graph.base, graph.extra))):
                raise RuntimeError("Policy changed mandatory pairs or left the candidate universe")
            edges = ordered_edges(current[-1], pairs, bool(loops))
            if edges.shape[1] != 2 * len(pairs) + loops * n:
                raise RuntimeError("Directed edge or self-edge count differs")
            output = current_prediction(name, edges)
            save_case(name, output, edges, optional, passes, score_seconds)
            arrays[f"pairs__{name}"] = same.canonical_pairs(pairs)
            if score is not None:
                arrays[f"selection_score__{name}"] = score
    # All selection and inference precede target access.
    target = np.asarray(target, dtype=np.float64)
    if target.shape != current[-1].shape or not np.isfinite(target).all():
        raise ValueError("Finite matching target required")
    arrays["target_position"] = target
    row["target_sha256"] = full.state_hash(target)
    for name, output in outputs.items():
        if row["cases"][name]["status"] == "complete":
            metric, residual, normalized = residual_record(output, target, std)
            row["cases"][name]["metrics"] = metric
            arrays[f"position_residual__{name}"] = residual
            arrays[f"normalized_residual__{name}"] = normalized
    failures = [{"case": name, "failure": value["failure"]} for name, value in row["cases"].items() if value["status"] == "failed"]
    row.update(status="failed" if failures else "complete", failure={"category": "case_failures", "cases": failures} if failures else None,
               operational_seconds=time.perf_counter() - started)
    return row, arrays


def expected_frames(run_config, validation_manifest, test_manifest):
    if full.sha256(validation_manifest) != run_config["valid"]["manifest_sha256"]:
        raise ValueError("Validation manifest differs from training")
    valid = json.loads(Path(validation_manifest).read_text())
    test = json.loads(Path(test_manifest).read_text())
    selected = full.select_records(test, "locked_test")
    ids = [record["id"] for record in valid["records"]]
    schedule = run_config["validation_frames"]
    if len(schedule) != 128 or len({item["id"] for item in schedule}) != 128:
        raise ValueError("Exactly 128 unique fixed validation frames required")
    rows = []
    for item in schedule:
        index, target = ids.index(item["trajectory"]), item["target_frame"]
        if target < 7 or item["id"] != f"{ids[index]}:{target}":
            raise ValueError("Fixed validation identity/history differs")
        rows.append({"split": "valid", "source_index": index, "target_frame": target, "trajectory_id": ids[index]})
    rows += [{"split": "test", "source_index": index, "target_frame": target,
              "trajectory_id": test["records"][index]["id"]} for index in selected for target in same.TARGET_FRAMES]
    if len(rows) != 425:
        raise ValueError("Exactly 425 fixed observed histories required")
    return rows, valid, test


def verify_saved_test(directory, checkpoint_hash, expected):
    directory = Path(directory)
    result_path, protocol_path = directory / "result.json", directory / "protocol.json"
    result, protocol = json.loads(result_path.read_text()), json.loads(protocol_path.read_text())
    if (result.get("state") != "complete" or result.get("checkpoint_sha256") != checkpoint_hash
            or result.get("protocol_sha256") != full.sha256(protocol_path) or protocol.get("checkpoint_sha256") != checkpoint_hash):
        raise ValueError("Saved same-state completion/checkpoint/protocol differs")
    lookup = {(row["source_index"], row["target_frame"]): row for row in result["records"]}
    keys = {(row["source_index"], row["target_frame"]) for row in expected if row["split"] == "test"}
    if len(lookup) != len(result["records"]) or set(lookup) != keys:
        raise ValueError("Saved same-state test coverage differs")
    for key, row in lookup.items():
        for file_key, hash_key in (("array_file", "array_sha256"), ("record_file", "record_sha256")):
            name = row[file_key]
            if Path(name).name != name or full.sha256(directory / name) != row[hash_key]:
                raise ValueError("Saved test file/hash differs")
    return lookup, {"result_sha256": full.sha256(result_path), "protocol_sha256": full.sha256(protocol_path),
                    "records": [{key: row[key] for key in ("source_index", "target_frame", "trajectory_id", "array_file", "array_sha256", "record_file", "record_sha256")} for row in result["records"]]}


def frame_input(item, trajectories, saved_directory, saved_lookup):
    positions, types = trajectories[item["split"]][item["source_index"]]
    t = item["target_frame"]
    current, previous = np.array(positions[t-6:t], dtype=np.float32), np.array(positions[t-7:t-1], dtype=np.float32)
    target, types = np.asarray(positions[t], dtype=np.float64), np.asarray(types, dtype=np.int64)
    if types.ndim == 0:
        types = np.full(current.shape[1], types, dtype=np.int64)
    if item["split"] == "test":
        saved = saved_lookup[(item["source_index"], t)]
        if saved["trajectory_id"] != item["trajectory_id"]:
            raise ValueError("Saved/source trajectory identity differs")
        with np.load(Path(saved_directory) / saved["array_file"], allow_pickle=False) as arrays:
            for name, value in (("current_observed_history", current), ("previous_observed_history", previous),
                                ("target_position", target), ("particle_types", types)):
                if not np.array_equal(arrays[name], value):
                    raise ValueError("Saved same-state history/target differs from verified source")
            current, previous = arrays["current_observed_history"].copy(), arrays["previous_observed_history"].copy()
    return current, previous, types, target


def stem(item):
    return f"{item['split']}_{item['source_index']:06d}_{item['target_frame']:04d}"


def record_index(row, path):
    return {**{key: row[key] for key in ("split", "source_index", "target_frame", "trajectory_id", "status", "failure", "array_file", "array_sha256")},
            "record_file": path.name, "record_sha256": full.sha256(path)}


def recover(directory, item, protocol_hash):
    path = directory / (stem(item) + ".json")
    if not path.exists():
        # An array without an atomic row may be from an interrupted forward.
        # Preserve it rather than silently overwrite an unsuccessful attempt.
        if (directory / (stem(item) + ".npz")).exists():
            raise ValueError("Orphan array requires manual preservation/review")
        return None
    row = json.loads(path.read_text())
    if any(row.get(key) != value for key, value in item.items()) or row.get("protocol_sha256") != protocol_hash:
        raise ValueError("Recovered record identity/protocol differs")
    array = directory / (stem(item) + ".npz")
    if row.get("array_file") != array.name or full.sha256(array) != row.get("array_sha256"):
        raise ValueError("Recovered record array/hash differs")
    if row.get("status") not in ("complete", "failed") or (row["status"] == "complete") != (row.get("failure") is None):
        raise ValueError("Recovered status/failure inconsistent")
    return record_index(row, path)


def run(args, protocol, model, trajectories, expected, saved_lookup):
    directory = args.output_dir
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing run requires --resume and exact protocol/provenance")
    elif args.resume or any(directory.glob("*.json")) or any(directory.glob("*.npz")):
        raise ValueError("Missing original protocol or existing output requires review")
    else:
        full.atomic_json(protocol_path, protocol)
    digest = full.sha256(protocol_path)
    indexed = {}
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("protocol_sha256") != digest:
            raise ValueError("Existing result protocol hash differs")
        indexed = {row["record_file"]: row for row in result["records"]}
        if len(indexed) != len(result["records"]):
            raise ValueError("Duplicate existing result rows")
    records = []
    recovered = {}
    for item in expected:
        row = recover(directory, item, digest)
        if row:
            if not args.resume or (row["record_file"] in indexed and indexed[row["record_file"]] != row):
                raise ValueError("Recovered index/hash differs or resume missing")
            recovered[stem(item)] = row
            records.append(row)
    if set(indexed) - {row["record_file"] for row in records}:
        raise ValueError("Missing/unexpected indexed frame")
    full.atomic_json(directory / "status.json", {"state": "running", "pid": os.getpid(), "started_utc": utc_now(), "reused_frames": len(recovered)})
    try:
        for item in expected:
            if stem(item) in recovered:
                if recovered[stem(item)]["failure"] and recovered[stem(item)]["failure"].get("category") == "native_parity_failure":
                    raise RuntimeError("Retained native parity failure requires review; no retry")
                continue
            current, previous, types, target = frame_input(item, trajectories, args.saved_same_state_dir, saved_lookup)
            row, arrays = evaluate_frame(model, current, previous, types, target, item, protocol["seed"], args.device)
            array = directory / (stem(item) + ".npz")
            temporary = array.with_suffix(".npz.tmp")
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush(); os.fsync(stream.fileno())
            temporary.replace(array)
            row.update(protocol_sha256=digest, array_file=array.name, array_sha256=full.sha256(array))
            path = directory / (stem(item) + ".json")
            full.atomic_json(path, row)
            records.append(record_index(row, path))
            full.atomic_json(result_path, {"state": "partial", "protocol_sha256": digest, "records": records})
            full.atomic_json(directory / "status.json", {"state": "running", "pid": os.getpid(), "committed_frames": len(records), "last_frame": item, "last_status": row["status"], "updated_utc": utc_now()})
            print(json.dumps({**item, "status": row["status"], "committed_frames": len(records)}), flush=True)
            if row["failure"] and row["failure"].get("category") == "native_parity_failure":
                raise RuntimeError("Native parity failed; artifacts retained, stop for review")
        # Recheck input bytes after predictions; no silent mid-run input changes.
        for path, expected_hash in protocol["input_files_sha256"].items():
            if full.sha256(path) != expected_hash:
                raise RuntimeError("Pinned input bytes changed during evaluation")
        full.atomic_json(result_path, {"state": "complete", "scope": protocol["scope"], "protocol_sha256": digest,
            "seed": protocol["seed"], "objective": protocol["objective"], "required_frames": len(expected),
            "records": records, "complete_frames": sum(row["status"] == "complete" for row in records),
            "failed_frames": sum(row["status"] == "failed" for row in records)})
        full.atomic_json(directory / "status.json", {"state": "complete", "completed_utc": utc_now(), "result_sha256": full.sha256(result_path), "committed_frames": len(records)})
    except BaseException as error:
        full.atomic_json(directory / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
            "error_type": type(error).__name__, "error": str(error), "committed_frames": len(records), "updated_utc": utc_now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--saved-same-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--training-protocol", type=Path, default=full.REPO / "research/protocols/full_waterdrop_100k.md")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    args.output_dir = args.output_dir.resolve()
    protected = (args.checkpoint.resolve().parent.parent, args.saved_same_state_dir.resolve().parent,
                 args.validation_manifest.resolve().parent, args.test_manifest.resolve().parent)
    if any(args.output_dir == directory or directory in args.output_dir.parents or args.output_dir in directory.parents
           for directory in protected):
        parser.error("A separate exploratory output directory outside original input/result trees is required")
    torch.set_num_threads(args.threads)
    device = full.resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with full.RunLock(args.output_dir, clear_stale=args.clear_stale_lock):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        metadata_path = args.validation_manifest.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        full.check_checkpoint(payload, "locked_test", args.training_protocol, full.sha256(metadata_path))
        config = payload["run_config"]
        for relative, digest in config["source_sha256"].items():
            if full.sha256(full.REPO / relative) != digest:
                raise ValueError("Frozen training source changed")
        expected, valid, test = expected_frames(config, args.validation_manifest, args.test_manifest)
        if valid["metadata"] != metadata or test["metadata"] != metadata:
            raise ValueError("Split metadata differs")
        trajectories = {"valid": full.load_manifest_data(args.validation_manifest, verify_hashes=True),
                        "test": full.load_manifest_data(args.test_manifest, verify_hashes=True)}
        checkpoint_hash = full.sha256(args.checkpoint)
        lookup, saved = verify_saved_test(args.saved_same_state_dir, checkpoint_hash, expected)
        model, provenance = full.load_for_evaluation(args.checkpoint, metadata, device, radius_backend="scipy_host")
        if model._max_num_neighbors != 128 or np.asarray(model._boundaries).tolist() != metadata["bounds"]:
            raise ValueError("Checkpoint graph/boundaries differ")
        paths = [Path(__file__), Path(full.__file__), Path(same.__file__), full.REPO / "research/budget_graph.py",
                 full.REPO / "research/full_training.py", args.protocol, args.training_protocol,
                 args.checkpoint, args.validation_manifest, args.test_manifest, metadata_path,
                 args.saved_same_state_dir / "protocol.json", args.saved_same_state_dir / "result.json"]
        paths += sorted((full.REPO / "adaptive-gns/gns").glob("*.py"))
        for row in lookup.values():
            paths += [args.saved_same_state_dir / row["array_file"], args.saved_same_state_dir / row["record_file"]]
        for manifest, directory in ((valid, args.validation_manifest.parent), (test, args.test_manifest.parent)):
            for record in manifest["records"]:
                paths += [directory / record[key]["path"] for key in ("positions", "particle_types")]
        protocol = {"schema": 1, "scope": "post-inspection exploratory graph-convention bridge; not locked confirmation",
            "seed": config["seed"], "objective": config["objective"], "checkpoint_sha256": checkpoint_hash,
            "checkpoint_provenance": provenance, "run_config_sha256": payload["run_config_sha256"],
            "bridge_protocol_sha256": full.sha256(args.protocol), "expected_frames": expected,
            "cases": list(CASE_NAMES), "saved_test_inputs": saved,
            "graph_semantics": "float32 native strict norms; receiver-major distance/ID directed ordering; capped factorial includes self edges in cap; policies uncapped; loop toggle adds exactly one self edge",
            "risk_scoring": "same loop convention and base graph; current and previous score passes included in standalone pass count; reused here for operational efficiency",
            "time_scope": "single calls, shared precomputed graphs; operational durations only; not policy latency or speedup",
            "guards": {"max_candidate_pairs": MAX_PAIRS, "max_abs_coordinate": MAX_ABS},
            "runtime": full.runtime_provenance(device, "scipy_host"), "threads": args.threads,
            "software": {"numpy": np.__version__, "scipy": full.scipy.__version__, "python": full.platform.python_version(), "platform": full.platform.platform()},
            "input_files_sha256": {str(path.resolve()): full.sha256(path) for path in paths}}
        run(args, protocol, model, trajectories, expected, lookup)


if __name__ == "__main__":
    main()
