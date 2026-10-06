"""Separate D3 evaluation primitives; no source/model/data selection or launch.

Copied metric/rollout structure from pinned native_graph_rollout b4bbca66...
and graph_convention_bridge 2c589c3c..., with explicit D3/.025/2M/5M guards.
Fast graph bookkeeping is the separately reviewed ca5898... source. Existing
modules are never mutated; shared pure numerical/statistical helpers stay pinned
by the execution entry. Autonomous cache and observed-state risk stay distinct.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from research import full_rollout as full
from research import full_same_state as _same

GRAPH_SHA = "ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50"
CORE_POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
PHYSICAL_POLICY = "relative-velocity-RMS25"
POLICIES = CORE_POLICIES + (PHYSICAL_POLICY,)
MAX_PAIRS, MAX_EDGES, MAX_ABS = 2000000, 5000000, 10.
RADIUS, DT = .025, .0025
PARITY_PRED_ATOL, PARITY_RISK_ATOL, PARITY_RISK_RTOL = 2e-7, 1e-6, 1e-5
path = Path(__file__).with_name("goop3d_graph_support_vectorized_v1.py")
if hashlib.sha256(path.read_bytes()).hexdigest() != GRAPH_SHA:
    raise ValueError("Reviewed3D graph source differs")
spec = importlib.util.spec_from_file_location("_goop3d_eval_fast_graph", path)
graph3d = importlib.util.module_from_spec(spec); sys.modules[spec.name] = graph3d; spec.loader.exec_module(graph3d)


def require(value, message):
    if not value: raise ValueError(message)


def strict_pairs(position, radius):
    points = np.asarray(position, dtype=np.float32)
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all() and radius == RADIUS, "Finite3D/.025 graph required")
    tree = cKDTree(points)
    count = int((tree.count_neighbors(tree, radius * 1.267) - len(points)) // 2)
    if count > MAX_PAIRS:
        raise full.RolloutGuard("candidate_pair_resource_guard", candidate_pairs=count, limit=MAX_PAIRS)
    return graph3d.strict_pairs(points, radius)


def ordered_edges(position, pairs, loops, cap=None):
    edges = graph3d.ordered_edges(position, pairs, loops, cap)
    if edges.shape[1] > MAX_EDGES:
        raise full.RolloutGuard("directed_edge_resource_guard", directed_edges=edges.shape[1], limit=MAX_EDGES)
    return edges


def normalization(model):
    stats = model._normalization_stats["acceleration"]
    values = {name: v.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(v) else np.asarray(v,dtype=np.float64) for name,v in stats.items()}
    require(values["std"].shape == (3,) and np.isfinite(values["std"]).all() and np.all(values["std"] > 0), "PositiveD3 acceleration normalization required")
    return values

same = SimpleNamespace(**vars(_same)); same.normalization = normalization


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
    if output["prediction"].shape != (n, 3) or output["risk"].shape != (n,):
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


def relative_velocity_rms(np, history, native_edges, dt):
    """Float64 arithmetic; sequential native-order add.at; incoming nonself only."""
    history = np.asarray(history)
    edges = np.asarray(native_edges)
    if (history.ndim != 3 or history.shape[0] < 2 or history.shape[2] != 3
            or not np.isfinite(history).all() or not math.isfinite(dt) or dt <= 0):
        raise ValueError("Finite observed 3D history and positive physical timestep required")
    n = history.shape[1]
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu" or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("Native directed integer edge list required")
    velocity = (history[-1].astype(np.float64) - history[-2].astype(np.float64)) / dt
    senders, receivers = edges[:, edges[0] != edges[1]]
    relative = velocity[receivers] - velocity[senders]
    squared = np.sum(relative * relative, axis=1, dtype=np.float64)
    sums, counts = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.int64)
    np.add.at(sums, receivers, squared)
    np.add.at(counts, receivers, 1)
    scores = np.sqrt(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite relative-velocity RMS score")
    return scores, counts


def native_graph(history, policy, cached_risk, rng):
    """Preserve native directed base as an ordered prefix; append annulus edges.

    The native receiver cap applies only to the mandatory base. Optional pairs
    are geometric annulus pairs, not within-r neighbors excluded by the cap.
    Their two orientations are uncapped and never remove a native edge.
    """
    graph = bridge.strict_pairs(history[-1], .025)
    native_base = bridge.ordered_edges(history[-1], graph.base, True, 128)
    uncapped_base = bridge.ordered_edges(history[-1], graph.base, True)
    scores = None
    if policy == PHYSICAL_POLICY:
        scores, _ = relative_velocity_rms(np, history, native_base, DT)
    selected = full.choose_pairs(graph, "laggedrisk25" if policy == PHYSICAL_POLICY else policy, history, scores if scores is not None else cached_risk, rng)
    if not np.array_equal(selected[:len(graph.base)], graph.base):
        raise RuntimeError("Selector changed the geometric base prefix")
    optional = bridge.same.canonical_pairs(selected[len(graph.base):])
    expected = 0 if policy == "base" else len(graph.extra) if policy == "dense" else int(.25 * len(graph.extra))
    if len(optional) != expected or not bridge.same.pair_set(optional) <= bridge.same.pair_set(graph.extra):
        raise RuntimeError("Exact optional annulus budget differs")
    extra_edges = bridge.ordered_edges(history[-1], optional, False)
    edges = np.concatenate((native_base, extra_edges), axis=1)
    if edges.shape[1] != native_base.shape[1] + 2 * expected:
        raise RuntimeError("Optional directed edge count differs")
    if edges.shape[1] > MAX_EDGES:
        raise full.RolloutGuard("directed_edge_resource_guard", directed_edges=edges.shape[1], limit=MAX_EDGES)
    # Unique directed indices; int64 keys preserve exactly the old set count.
    require(graph.n_nodes < 2**31, "D3 node identity exceeds guarded index arithmetic")
    nonself = native_base[:, native_base[0] != native_base[1]]
    keys = np.sort(nonself[0] * graph.n_nodes + nonself[1])
    reverse = nonself[1] * graph.n_nodes + nonself[0]
    index = np.searchsorted(keys, reverse)
    present = index < len(keys)
    present[present] &= keys[index[present]] == reverse[present]
    asymmetric = int((~present).sum())
    degree = np.bincount(uncapped_base[1], minlength=graph.n_nodes)
    native_degree = np.bincount(native_base[1], minlength=graph.n_nodes)
    audit = {"candidate_pairs": len(graph.base) + len(graph.extra), "geometric_base_pairs": len(graph.base),
        "available_annulus_pairs": len(graph.extra), "optional_pair_budget": int(.25 * len(graph.extra)),
        "retained_optional_pairs": len(optional), "directed_edges": edges.shape[1],
        "native_base_directed_edges": native_base.shape[1], "native_base_self_edges": int(np.sum(native_base[0] == native_base[1])),
        "native_base_receivers_above_cap_before_capping": int(np.sum(degree > 128)),
        "native_base_edges_removed_by_cap": uncapped_base.shape[1] - native_base.shape[1],
        "native_base_max_receiver_degree": int(native_degree.max()),
        "native_base_asymmetric_directed_edges": asymmetric,
        "native_base_sha256": full.state_hash(native_base), "selected_optional_pair_sha256": full.state_hash(optional),
        "directed_edge_sha256": full.state_hash(edges), "native_base_prefix_preserved": bool(np.array_equal(edges[:, :native_base.shape[1]], native_base))}
    return graph, edges, optional, audit


@torch.no_grad()
def rollout(model, positions, particle_types, metadata, policy, horizon, rng_seed, device="cpu", trace_steps=full.TRACE_STEPS):
    if policy not in POLICIES or horizon < 1 or len(positions) < horizon + 6:
        raise ValueError("Known policy and full input/target horizon required")
    if positions.ndim != 3 or positions.shape[2] != 3 or float(model._connectivity_radius) != .025:
        raise ValueError("Native bridge requires the original 3D radius .025 model")
    types = np.asarray(particle_types, dtype=np.int64)
    if types.ndim == 0:
        types = np.full(positions.shape[1], types, dtype=np.int64)
    if types.shape != (positions.shape[1],) or np.any(types == 3):
        raise ValueError("One nonkinematic type per Goop-3D particle required")
    bounds = np.asarray(metadata["bounds"], dtype=np.float64)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Finite ordered 3D bounds required")
    if float(metadata["default_connectivity_radius"]) != .025 or not np.array_equal(np.asarray(model._boundaries), bounds):
        raise ValueError("Model and metadata graph/boundaries differ")
    model.eval()
    history = np.array(positions[:6], dtype=np.float32, copy=True)
    initial_hash = full.state_hash(history)
    all_started = time.perf_counter()
    traces = {"initial_observed_positions": history.copy(), "particle_types": types.copy(), "bounds": bounds}
    initial_failure, initial_phase, parity_passes = None, "initial_state", 0
    parity = {"passed": False, "not_attempted_reason": None}
    try:
        full.check_state(history, "initial_state", bridge.MAX_ABS)
        initial_phase = "initial_graph"
        initial_graph = bridge.strict_pairs(history[-1], .025)
        base_edges = bridge.ordered_edges(history[-1], initial_graph.base, True, 128)
        initial_phase = "native_parity"
        parity, parity_arrays, parity_supplied = bridge.native_parity(model, history, types, base_edges, device)
        parity_passes = 2
        traces.update({f"initial_parity_{key}": value for key, value in parity_arrays.items()})
        traces.update({f"initial_parity_supplied_{key}": parity_supplied[key] for key in ("prediction", "risk", "raw_risk")})
        traces["initial_parity_supplied_edges"] = base_edges
    except full.RolloutGuard as error:
        initial_failure = {**error.details, "forecast_step": 0, "phase": initial_phase}
        parity["not_attempted_reason"] = initial_failure
        traces["failed_input_history"] = history.copy()
        traces["rejected_history"] = history.copy()
    parity_seconds = time.perf_counter() - all_started
    started = time.perf_counter()
    rng = np.random.default_rng(rng_seed)
    failure = initial_failure or (None if parity["passed"] else {"category": "native_parity_failure", "forecast_step": 0, "phase": "native_parity"})
    cached_risk, warmup = None, None
    attempts, mse, edges_per_step, optional_per_step, candidates, geometric_base = [], [], [], [], [], []
    risk_means, predicted_boundary, truth_boundary = [], [], []
    selected_trace, trace_prediction, trace_truth, trace_history = [], [], [], []
    trace_current_risk, trace_cached_risk = [], []
    total_passes = forecast_passes = 0
    graph_seconds = forward_seconds = 0.
    if policy == "laggedrisk25" and failure is None:
        warmup_started = time.perf_counter()
        warmup = {"passes": 0, "mean_prediction_discarded": True, "history_sha256": initial_hash}
        try:
            graph_started = time.perf_counter()
            _, warmup_edges, _, graph_audit = native_graph(history, "base", None, rng)
            warmup["graph_build_and_selection_seconds"] = time.perf_counter() - graph_started
            warmup.update(graph_audit)
            total_passes += 1; warmup["passes"] = 1
            output = bridge.supplied(model, history, types, warmup_edges, device)
            for key in ("prediction", "raw_risk", "risk"):
                traces[f"warmup_{key}"] = output[key]
            warmup["features_forward_decode_and_transfer_seconds"] = output["operational_seconds"]
            bridge.validate_output(output, len(types))
            cached_risk = output["risk"]
        except full.RolloutGuard as error:
            failure = {**error.details, "forecast_step": 0, "phase": "warmup"}
            traces["failed_input_history"] = history.copy()
            traces["rejected_history"] = history.copy()
        warmup["total_wall_seconds"] = time.perf_counter() - warmup_started
    for step in range(1, horizon + 1):
        if failure is not None:
            break
        attempt = {"forecast_step": step, "accepted": False}
        phase = "state"
        try:
            full.check_state(history, "state", bridge.MAX_ABS)
            phase = "graph"
            graph_started = time.perf_counter()
            _, edge, optional, graph_audit = native_graph(history, policy, cached_risk, rng)
            duration = time.perf_counter() - graph_started
            attempt.update(graph_audit, graph_build_and_selection_seconds=duration)
            graph_seconds += duration
            phase = "forward"
            total_passes += 1; forecast_passes += 1
            output = bridge.supplied(model, history, types, edge, device)
            attempt["features_forward_decode_and_transfer_seconds"] = output["operational_seconds"]
            forward_seconds += output["operational_seconds"]
            prediction, risk = output["prediction"], output["risk"]
            if np.isfinite(prediction).all():
                attempt["predicted_boundary"] = full.boundary_metrics(prediction, bounds)
            try:
                bridge.validate_output(output, len(types))
            except full.RolloutGuard:
                for key in ("prediction", "raw_risk", "risk"):
                    traces[f"rejected_{key}"] = output[key]
                traces["rejected_history"] = history.copy()
                traces["rejected_edges"] = edge.copy()
                traces["rejected_optional_pairs"] = optional.copy()
                raise
            # Future truth is read only after the autonomous prediction.
            truth = np.asarray(positions[6 + step - 1], dtype=np.float64)
            if not np.isfinite(truth).all():
                raise ValueError("Ground-truth coordinates are nonfinite")
            error = float(np.mean((prediction.astype(np.float64) - truth) ** 2))
            mse.append(error); edges_per_step.append(edge.shape[1]); optional_per_step.append(len(optional))
            candidates.append(graph_audit["candidate_pairs"]); geometric_base.append(graph_audit["geometric_base_pairs"])
            risk_means.append(float(risk.mean()))
            predicted_boundary.append(attempt["predicted_boundary"])
            truth_boundary.append(full.boundary_metrics(truth, bounds))
            attempt.update(accepted=True, coordinate_mse=error, ground_truth_boundary=truth_boundary[-1],
                           predicted_state_sha256=full.state_hash(prediction))
            if step in trace_steps:
                selected_trace.append(step); trace_prediction.append(prediction.copy()); trace_truth.append(truth.astype(np.float32))
                trace_history.append(history.copy()); trace_current_risk.append(risk.copy())
                trace_cached_risk.append(np.full(len(types), np.nan, dtype=np.float32) if cached_risk is None else cached_risk.copy())
                traces[f"edges_forecast_{step:04d}"] = edge.copy()
                traces[f"optional_pairs_forecast_{step:04d}"] = optional.copy()
            cached_risk = risk
            history = np.concatenate((history[1:], prediction[None]), axis=0)
        except full.RolloutGuard as error:
            failure = {**error.details, "forecast_step": step, "phase": phase}
            attempt["failure"] = failure
            traces["failed_input_history"] = history.copy()
            traces.setdefault("rejected_history", history.copy())
            if cached_risk is not None:
                traces["failed_input_cached_risk"] = cached_risk.copy()
        attempts.append(attempt)
    full.synchronize(device)
    complete = failure is None and len(mse) == horizon
    if warmup:
        graph_seconds += warmup.get("graph_build_and_selection_seconds", 0.)
        forward_seconds += warmup.get("features_forward_decode_and_transfer_seconds", 0.)
    row = {"status": "complete" if complete else "failed", "failure": failure, "policy": policy,
        "horizon": horizon, "completed_steps": len(mse), "rng_seed": int(rng_seed), "warmup": warmup,
        "native_parity": parity, "native_parity_network_passes": parity_passes, "native_parity_operational_seconds": parity_seconds,
        "total_network_passes": total_passes, "forecast_network_passes": forecast_passes,
        "mse_at_steps": {str(s): mse[s-1] if len(mse) >= s else None for s in full.TRACE_STEPS if s <= horizon},
        "mse_at_final_horizon": mse[-1] if complete else None,
        "mean_rollout_mse": float(np.mean(mse)) if complete else None,
        "mean_directed_edges": float(np.mean(edges_per_step)) if complete else None,
        "prefix_mean_mse_if_failed": float(np.mean(mse)) if mse and not complete else None,
        "total_wall_seconds": time.perf_counter() - started,
        "total_wall_seconds_including_native_parity": time.perf_counter() - all_started,
        "graph_build_and_selection_seconds_including_warmup": graph_seconds,
        "features_forward_decode_and_transfer_seconds_including_warmup": forward_seconds,
        "initial_observed_state_sha256": initial_hash, "final_predicted_state_sha256": full.state_hash(history[-1]),
        "mse_per_step": mse, "directed_edges_per_step": edges_per_step,
        "retained_optional_pairs_per_step": optional_per_step, "candidate_pairs_per_step": candidates,
        "base_pairs_per_step": geometric_base, "mean_normalized_acceleration_variance_per_step": risk_means,
        "initial_observed_boundary": [full.boundary_metrics(frame, bounds) for frame in positions[:6]] if np.isfinite(positions[:6]).all() else None,
        "predicted_boundary_per_step": predicted_boundary, "ground_truth_boundary_per_step": truth_boundary,
        "attempts": attempts}
    traces.update(forecast_steps=np.asarray(selected_trace, dtype=np.int64),
        predicted_positions=np.stack(trace_prediction) if trace_prediction else np.empty((0, len(types), 3), dtype=np.float32),
        ground_truth_positions=np.stack(trace_truth) if trace_truth else np.empty((0, len(types), 3), dtype=np.float32),
        observed_or_predicted_histories=np.stack(trace_history) if trace_history else np.empty((0, 6, len(types), 3), dtype=np.float32),
        current_risk=np.stack(trace_current_risk) if trace_current_risk else np.empty((0, len(types)), dtype=np.float32),
        cached_risk_before_forecast=np.stack(trace_cached_risk) if trace_cached_risk else np.empty((0, len(types)), dtype=np.float32))
    return row, traces


def physical_graph(history, rng):
    graph, edges, optional, audit = native_graph(history, PHYSICAL_POLICY, None, rng)
    base = ordered_edges(history[-1], graph.base, True, 128)
    scores, _ = relative_velocity_rms(np, history, base, DT)
    return graph, edges, optional, audit, scores

bridge = SimpleNamespace(np=np, torch=torch, full=full, same=same, cKDTree=cKDTree,
    strict_pairs=strict_pairs, ordered_edges=ordered_edges, supplied=supplied, native_parity=native_parity,
    validate_output=validate_output, graph_record=graph_record, residual_record=residual_record,
    MAX_PAIRS=MAX_PAIRS, MAX_ABS=MAX_ABS, PARITY_PRED_ATOL=PARITY_PRED_ATOL,
    PARITY_RISK_ATOL=PARITY_RISK_ATOL, PARITY_RISK_RTOL=PARITY_RISK_RTOL)
