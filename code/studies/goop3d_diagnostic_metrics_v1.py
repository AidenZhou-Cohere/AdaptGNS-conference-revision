"""Numerical particle-simulation helpers. Use code/evaluate.py for evaluation."""
import math
import time
PHYSICAL_POLICY="relative-velocity-RMS25"
POLICIES=("base","dense","random25","speed25",PHYSICAL_POLICY,"previous-observed-base-risk25")
NATURAL_BASE="natural_base_reference"
TIMING_CASES=POLICIES+(NATURAL_BASE,)
REPEATS=7

def require(value,message):
    if not value: raise ValueError(message)


def timing_order(source_index, schedule_index, round_index):
    offset = (source_index * 5 + schedule_index + round_index) % len(TIMING_CASES)
    return TIMING_CASES[offset:] + TIMING_CASES[:offset]


def natural_base_edges(native, history):
    """Strict float32 base-only query; same pinned cap/self/ordering helper."""
    bridge, np = native.bridge, native.np
    points = np.asarray(history[-1], dtype=np.float32)
    tree = bridge.cKDTree(points)
    count = int((tree.count_neighbors(tree, .025) - len(points)) // 2)
    if count > bridge.MAX_PAIRS:
        raise native.full.RolloutGuard("candidate_pair_resource_guard", candidate_pairs=count, limit=bridge.MAX_PAIRS)
    pairs = bridge.same.canonical_pairs(tree.query_pairs(.025, output_type="ndarray"))
    distances = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    pairs = pairs[distances < .025]
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
        require(current.shape == previous.shape and current.shape[0] == 6 and current.shape[2] == 3
                and types.shape == (current.shape[1],) and np.all(types == 7), "Observed history/type contract differs")
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
        row.update(status="complete", graph=audit, metrics={"normalized_acceleration_coordinate_mse": float(se.mean() / 3),
            "realized_normalized_vector_se": float(se.mean()), "predicted_normalized_vector_se": float(3 * q.mean()),
            "constant_free_gaussian_nll": float((.5 * se / q + 1.5 * np.log(q)).mean())})
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
