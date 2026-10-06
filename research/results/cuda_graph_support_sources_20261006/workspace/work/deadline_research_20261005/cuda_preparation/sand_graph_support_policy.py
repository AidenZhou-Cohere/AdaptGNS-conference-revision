"""Separate Sand physical policy adapter; no imports of model packages at import.

Five core policies call the pinned native implementation unchanged. The sixth
uses deterministic float64 relative-velocity RMS over native incoming support.
The physical rollout loop is an explicit copy with a changed graph callback;
no frozen module globals, training sources or existing evaluator are modified.
"""
import math
import time

CORE_POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
PHYSICAL_POLICY = "relative-velocity-RMS25"
POLICIES = CORE_POLICIES + (PHYSICAL_POLICY,)


def relative_velocity_rms(np, history, native_edges, dt):
    """Float64 arithmetic; sequential native-order add.at; incoming nonself only."""
    history = np.asarray(history)
    edges = np.asarray(native_edges)
    if (history.ndim != 3 or history.shape[0] < 2 or history.shape[2] != 2
            or not np.isfinite(history).all() or not math.isfinite(dt) or dt <= 0):
        raise ValueError("Finite observed 2D history and positive physical timestep required")
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


def physical_graph(native, history, rng, dt):
    """Preserve native directed base as an ordered prefix; append annulus edges.

    The native receiver cap applies only to the mandatory base. Optional pairs
    are geometric annulus pairs, not within-r neighbors excluded by the cap.
    Their two orientations are uncapped and never remove a native edge.
    """
    np, bridge, full = native.np, native.bridge, native.full
    policy = PHYSICAL_POLICY
    graph = bridge.strict_pairs(history[-1], .015)
    native_base = bridge.ordered_edges(history[-1], graph.base, True, 128)
    uncapped_base = bridge.ordered_edges(history[-1], graph.base, True)
    score_started = time.perf_counter()
    scores, counts = relative_velocity_rms(np, history, native_base, dt)
    score_seconds = time.perf_counter() - score_started
    selected = full.choose_pairs(graph, "laggedrisk25", history, scores, rng)
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
    native_set = set(map(tuple, native_base.T.tolist()))
    nonself = {(a, b) for a, b in native_set if a != b}
    asymmetric = sum((b, a) not in nonself for a, b in nonself)
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
    audit.update(physical_score=PHYSICAL_POLICY, physical_score_dtype="float64", physical_score_sha256=full.state_hash(scores),
        physical_neighbor_counts_sha256=full.state_hash(counts), physical_isolated_particles=int(np.sum(counts == 0)),
        physical_score_seconds=score_seconds, physical_timestep=dt)
    return graph, edges, optional, audit, scores

def _physical_rollout(native, model, positions, particle_types, metadata, policy, horizon, rng_seed, device="cpu", trace_steps=None):
    np, bridge, full = native.np, native.bridge, native.full
    trace_steps = full.TRACE_STEPS if trace_steps is None else trace_steps
    def native_graph(history, method, cached, rng):
        if method == PHYSICAL_POLICY:
            return physical_graph(native, history, rng, float(metadata["dt"]))[:4]
        return native.native_graph(history, method, cached, rng)
    if policy not in POLICIES or horizon < 1 or len(positions) < horizon + 6:
        raise ValueError("Known policy and full input/target horizon required")
    if positions.ndim != 3 or positions.shape[2] != 2 or float(model._connectivity_radius) != .015:
        raise ValueError("Native bridge requires the original 2D radius .015 model")
    types = np.asarray(particle_types, dtype=np.int64)
    if types.ndim == 0:
        types = np.full(positions.shape[1], types, dtype=np.int64)
    if types.shape != (positions.shape[1],) or np.any(types == 3):
        raise ValueError("One nonkinematic type per WaterDrop particle required")
    bounds = np.asarray(metadata["bounds"], dtype=np.float64)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Finite ordered 2D bounds required")
    if float(metadata["default_connectivity_radius"]) != .015 or not np.array_equal(np.asarray(model._boundaries), bounds):
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
        initial_graph = bridge.strict_pairs(history[-1], .015)
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
        predicted_positions=np.stack(trace_prediction) if trace_prediction else np.empty((0, len(types), 2), dtype=np.float32),
        ground_truth_positions=np.stack(trace_truth) if trace_truth else np.empty((0, len(types), 2), dtype=np.float32),
        observed_or_predicted_histories=np.stack(trace_history) if trace_history else np.empty((0, 6, len(types), 2), dtype=np.float32),
        current_risk=np.stack(trace_current_risk) if trace_current_risk else np.empty((0, len(types)), dtype=np.float32),
        cached_risk_before_forecast=np.stack(trace_cached_risk) if trace_cached_risk else np.empty((0, len(types)), dtype=np.float32))
    return row, traces

class NativeAdapter:
    """Read-only facade; the original native module is never patched."""
    def __init__(self, native, dt=.0025):
        if tuple(native.POLICIES) != CORE_POLICIES:
            raise ValueError("Pinned core policy order differs")
        self.original = native
        self.np, self.torch, self.bridge, self.full = native.np, native.torch, native.bridge, native.full
        self.dt = dt

    def native_graph(self, history, policy, cached_risk, rng):
        if policy == PHYSICAL_POLICY:
            return physical_graph(self.original, history, rng, self.dt)[:4]
        return self.original.native_graph(history, policy, cached_risk, rng)

    def physical_graph(self, history, rng):
        return physical_graph(self.original, history, rng, self.dt)

    def rollout(self, model, positions, types, metadata, policy, horizon, rng_seed, device="cpu", trace_steps=None):
        if float(metadata["dt"]) != self.dt:
            raise ValueError("Physical-policy timestep differs from metadata")
        trace_steps = self.full.TRACE_STEPS if trace_steps is None else trace_steps
        if policy in CORE_POLICIES:
            return self.original.rollout(model, positions, types, metadata, policy, horizon, rng_seed, device, trace_steps=trace_steps)
        if policy != PHYSICAL_POLICY:
            raise ValueError("Unknown declared policy")
        with self.torch.no_grad():
            return _physical_rollout(self.original, model, positions, types, metadata, policy, horizon, rng_seed, device, trace_steps)
