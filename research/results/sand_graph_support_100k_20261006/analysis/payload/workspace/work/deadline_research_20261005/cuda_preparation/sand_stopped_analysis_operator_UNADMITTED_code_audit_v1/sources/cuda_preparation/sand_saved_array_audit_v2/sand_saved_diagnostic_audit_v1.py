"""NumPy-only arithmetic audit of saved Sand final diagnostic rows.

No model, evaluator, source trajectory, or graph builder is imported. The caller
owns file/collection/model admission. This checks the saved numeric descriptors,
available parity/repeated outputs, residuals, benefits, correlations, boundaries,
timing arithmetic and clean-normalized metrics. It does not reconstruct missing
graphs, learned features/outputs, clean normalization from checkpoint statistics,
source truth, GPU determinism, elapsed time, or generic execution exceptions.

Early failures may lack targets, policy summaries, supplied edges, or downstream
outputs. Available evidence is checked; absent evidence is never manufactured.
The returned dict uses the evaluator's flattened equal-trajectory metric paths.
"""
import hashlib

import numpy as np

POLICIES = ("base", "dense", "random25", "speed25", "relative-velocity-RMS25",
            "previous-observed-base-risk25")
TIMING_CASES = POLICIES + ("natural_base_reference",)
TIMES = ("end_to_end_seconds", "score_generation_seconds", "score_graph_seconds",
         "score_forward_seconds", "current_graph_and_selection_seconds", "current_forward_seconds")
PRED_ATOL, RISK_ATOL, RISK_RTOL = 2e-7, 1e-6, 1e-5


def expected_metric_keys(mode):
    """The fixed evaluator leaves, including leaves undefined for every row."""
    if mode == "clean-validation":
        return tuple("metrics/" + key for key in ("normalized_acceleration_coordinate_mse",
                     "realized_normalized_vector_se", "predicted_normalized_vector_se", "constant_free_gaussian_nll"))
    if mode != "same-state":
        raise ValueError("Unknown diagnostic mode")
    keys = ["accuracy/" + method + "/" + key for method in POLICIES
            for key in ("position_coordinate_mse", "normalized_coordinate_mse")]
    for method in POLICIES[1:]:
        names = ("mean_position_vector_benefit", "mean_normalized_vector_benefit",
                 "positive_fraction", "negative_fraction", "zero_fraction")
        if method != "dense":
            names += ("dense_sparse_sign_disagreement_fraction", "dense_positive_sparse_nonpositive_fraction")
        keys.extend("benefit/" + method + "/" + key for key in names)
    keys.extend("timing/" + method + "/" + key for method in TIMING_CASES for key in TIMES)
    keys.append("correlations/previous_risk_vs_base_error")
    keys.extend("correlations/previous_risk_vs_" + method + "_benefit" for method in POLICIES[1:])
    keys.extend("correlations/dense_vs_" + method + "_benefit" for method in POLICIES[2:])
    return tuple(keys)


def _require(checks, condition, label):
    checks.equal(bool(condition), True, label)


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _descriptors(row, arrays, checks):
    descriptors = row.get("numeric_arrays", {})
    checks.equal(set(descriptors), set(arrays), "complete numeric descriptor inventory")
    for key, value in arrays.items():
        _require(checks, isinstance(value, np.ndarray) and not value.dtype.hasobject,
                 key + ": numeric ndarray")
        checks.equal(descriptors[key], {"shape": list(value.shape), "dtype": value.dtype.str,
                                      "value_sha256": _hash(value)}, key + ": numeric descriptor")


def _boundary(points, bounds):
    points = np.asarray(points, dtype=np.float64)
    excursion = np.maximum(np.maximum(bounds[:, 0] - points, points - bounds[:, 1]), 0.)
    particle = excursion.max(axis=-1)
    return {"fraction_particles_outside": float(np.mean(particle > 0)),
            "fraction_particles_outside_by_more_than_1e-6": float(np.mean(particle > 1e-6)),
            "maximum_coordinate_excursion": float(excursion.max()),
            "mean_particle_maximum_excursion": float(particle.mean()),
            "coordinate_minimum": points.min(axis=0).tolist(),
            "coordinate_maximum": points.max(axis=0).tolist()}


def _check_numbers(actual, expected, checks, label):
    checks.equal(set(actual), set(expected), label + ": keys")
    for key, value in expected.items():
        if isinstance(value, list):
            checks.array(np.asarray(actual[key]), np.asarray(value), label + "/" + key)
        elif value is None or isinstance(value, (str, bool)):
            checks.equal(actual[key], value, label + "/" + key)
        else:
            checks.close(actual[key], value, label + "/" + key)


def _ranks(values):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start + 1 + stop) / 2.
    return ranks


def _spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 2:
        return {"value": None, "reason": "fewer_than_two_particles", "particles": len(x)}
    a, b = _ranks(x), _ranks(y)
    a -= a.mean(); b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return {"value": None, "reason": "constant_rank_vector", "particles": len(x)}
    return {"value": float(np.clip(a.dot(b) / denominator, -1., 1.)),
            "reason": None, "particles": len(x)}


def _outputs(arrays, prefix, n, checks, complete=False):
    present = {key: arrays[prefix + key] for key in ("prediction", "risk", "raw_risk")
               if prefix + key in arrays}
    if complete:
        checks.equal(set(present), {"prediction", "risk", "raw_risk"}, prefix + ": complete outputs")
    for key, value in present.items():
        checks.equal(value.shape, (n, 2) if key == "prediction" else (n,), prefix + key + ": shape")
        checks.equal(value.dtype.str, np.dtype(np.float32).str, prefix + key + ": float32 output")
        if complete:
            _require(checks, np.isfinite(value).all(), prefix + key + ": finite accepted output")
    if "raw_risk" in present and "risk" in present:
        wanted = np.maximum(present["raw_risk"], np.float32(1e-6))
        _require(checks, np.array_equal(present["risk"], wanted, equal_nan=True), prefix + ": variance floor")
    if complete:
        _require(checks, np.max(np.abs(present["prediction"])) <= 10. and np.all(present["risk"] > 0),
                 prefix + ": accepted prediction/risk guards")
    return present


def _parity(record, arrays, prefix, supplied_edges, n, checks):
    if record is None:
        return
    native = _outputs(arrays, prefix + "native_", n, checks)
    supplied = _outputs(arrays, prefix + "supplied_", n, checks)
    # A returned native_parity record always has both outputs, even on failure.
    checks.equal(set(native), {"prediction", "risk", "raw_risk"}, prefix + ": native outputs")
    checks.equal(set(supplied), set(native), prefix + ": supplied outputs")
    finite = all(np.isfinite(value).all() for output in (native, supplied) for value in output.values())
    agreement = bool(finite and np.allclose(native["prediction"], supplied["prediction"], rtol=0, atol=PRED_ATOL)
                     and all(np.allclose(native[key], supplied[key], rtol=RISK_RTOL, atol=RISK_ATOL)
                             for key in ("risk", "raw_risk")))
    checks.equal(record["finite"], finite, prefix + ": finite")
    checks.equal(record["prediction_risk_agree"], agreement, prefix + ": output agreement")
    for key, value in (("prediction_atol", PRED_ATOL), ("risk_atol", RISK_ATOL), ("risk_rtol", RISK_RTOL)):
        checks.equal(record[key], value, prefix + ": fixed " + key)
    for key in native:
        difference = float(np.max(np.abs(native[key] - supplied[key]))) if finite else None
        checks.close(record[key + "_max_abs_difference"], difference, prefix + key + ": difference")
    flags = record["native_supplied_feature_identity"]
    checks.equal(len(flags), 3, prefix + ": three feature identity flags")
    for index, key in ((0, "node_features"), (2, "edge_features")):
        left, right = arrays[prefix + "native_" + key], arrays[prefix + "supplied_" + key]
        checks.equal(left.shape, (n, 30) if key == "node_features" else (arrays[prefix + "native_edges"].shape[1], 3),
                     prefix + key + ": native shape")
        checks.equal(flags[index], bool(np.array_equal(left, right)), prefix + key + ": identity")
    checks.equal(flags[1], record["native_edge_identity"], prefix + ": edge identity flag agreement")
    if supplied_edges is not None:
        checks.equal(record["native_edge_identity"], bool(np.array_equal(arrays[prefix + "native_edges"], supplied_edges)),
                     prefix + ": available supplied edge identity")
    checks.equal(record["passed"], bool(record["native_edge_identity"] and all(flags) and agreement), prefix + ": parity conjunction")


def _timing(calls, method, record, checks):
    result = {}
    for key in TIMES:
        values = [c["timing"][key] for c in calls if c["status"] == "complete"]
        _require(checks, all(np.isfinite(v) and v >= 0 for v in values), method + "/" + key + ": measured values")
        complete = len(values) == 7
        stats = {"observations": len(values), "expected": 7,
                 "mean": float(np.mean(values)) if complete else None,
                 "median": float(np.median(values)) if complete else None,
                 "sample_sd": float(np.std(values, ddof=1)) if complete else None}
        _check_numbers(record["timing"][key], stats, checks, method + "/" + key)
        result["timing/" + method + "/" + key] = stats["mean"] if record["status"] == "complete" else None
    return result


def _same_state(row, arrays, bounds, checks):
    current, previous, types = arrays["current_history"], arrays["previous_history"], arrays["particle_types"]
    n = len(types)
    checks.equal(current.shape, (6, n, 2), "current history shape")
    checks.equal(previous.shape, current.shape, "previous history shape")
    checks.equal(current.dtype.str, np.dtype(np.float32).str, "current history float32")
    checks.equal(previous.dtype.str, np.dtype(np.float32).str, "previous history float32")
    _require(checks, np.array_equal(current[:-1], previous[1:], equal_nan=True), "overlapping observed histories")
    for family, rounds in (("warmup_calls", (-1,)), ("timed_calls", range(7))):
        expected = []
        for repetition in rounds:
            offset = (row["source_index"] * 5 + row["schedule_index"] + max(0, repetition)) % len(TIMING_CASES)
            order = TIMING_CASES[offset:] + TIMING_CASES[:offset]
            expected.extend((repetition, slot, method) for slot, method in enumerate(order))
        actual = [(call["round"], call["slot"], call["method"]) for call in row.get(family, [])]
        checks.equal(actual, expected[:len(actual)], family + ": fixed ordered call prefix")
        _require(checks, len(actual) <= len(expected), family + ": no extra calls")
    if row.get("timed_calls"):
        checks.equal(len(row.get("warmup_calls", [])), 7, "timed calls follow full warmup round")
    reference = {}
    for call in row.get("warmup_calls", []) + row.get("timed_calls", []):
        method, repetition = call["method"], call["round"]
        _require(checks, method in TIMING_CASES and type(repetition) is int and -1 <= repetition <= 6, "call method/round")
        _require(checks, call["status"] in ("complete", "failed"), "recorded call status")
        prefix = f"r{repetition + 1}_{method}__"
        outputs = _outputs(arrays, prefix, n, checks, call["status"] == "complete")
        if "prediction" in outputs:
            checks.equal(call.get("prediction_sha256"), _hash(outputs["prediction"]), prefix + ": prediction hash")
        old = _outputs(arrays, prefix + "previous_base_", n, checks,
                       call["status"] == "complete" and method == POLICIES[-1])
        if call["status"] != "complete" or repetition < 0:
            continue
        if method not in reference:
            reference[method] = (call, outputs, old, prefix)
        else:
            first = reference[method]
            numerical = bool(np.allclose(first[1]["prediction"], outputs["prediction"], rtol=0, atol=PRED_ATOL)
                             and all(np.allclose(first[1][key], outputs[key], rtol=RISK_RTOL, atol=RISK_ATOL)
                                     for key in ("risk", "raw_risk")))
            recorded = call["repeat_consistency"]
            checks.equal(recorded["outputs_within_fixed_parity_tolerance"], numerical, prefix + ": repeated outputs")
            if method == POLICIES[-1]:
                score_equal = bool(np.allclose(first[2]["prediction"], old["prediction"], rtol=0, atol=PRED_ATOL)
                                   and all(np.allclose(first[2][key], old[key], rtol=RISK_RTOL, atol=RISK_ATOL)
                                           for key in ("risk", "raw_risk")))
                checks.equal(recorded["previous_score_outputs_within_fixed_parity_tolerance"], score_equal,
                             prefix + ": repeated previous scores")
    current_edges = arrays.get("r1_base__edges")
    previous_edges = arrays.get("r1_previous-observed-base-risk25__previous_base_edges")
    _parity(row.get("native_parity", {}).get("current"), arrays, "current_parity_", current_edges, n, checks)
    _parity(row.get("native_parity", {}).get("previous"), arrays, "previous_parity_", previous_edges, n, checks)
    if "natural_shared_base" in row:
        base, natural = reference["base"], reference["natural_base_reference"]
        saved = row["natural_shared_base"]
        edges_equal = bool(np.array_equal(arrays[base[3] + "edges"], arrays[natural[3] + "edges"]))
        checks.equal(saved["ordered_edges_exact"], edges_equal, "natural/shared available edges")
        agree = True
        for key in ("prediction", "risk", "raw_risk"):
            checks.close(saved[key + "_max_abs_difference"], float(np.max(np.abs(base[1][key] - natural[1][key]))),
                         "natural/shared " + key)
            agree &= bool(np.allclose(base[1][key], natural[1][key], rtol=0 if key == "prediction" else RISK_RTOL,
                                      atol=PRED_ATOL if key == "prediction" else RISK_ATOL))
        checks.equal(saved["outputs_within_fixed_parity_tolerance"], agree, "natural/shared output agreement")
    flat, errors = {}, {}
    policies = row.get("policies", {})
    finalized = row["status"] == "complete" or (row.get("failure") or {}).get("category") == "policy_guard_or_repeat_failure"
    # This guard precedes every policy summary in the evaluator. A rejected
    # comparison can be retained on an early failure, but cannot coexist with
    # finalized or partially written downstream policy summaries.
    if (finalized or policies) and "base" in reference and "natural_base_reference" in reference:
        natural_gate = row.get("natural_shared_base")
        _require(checks, isinstance(natural_gate, dict) and natural_gate.get("ordered_edges_exact") is True
                 and natural_gate.get("outputs_within_fixed_parity_tolerance") is True,
                 "natural/shared base gate before policy summaries")
    if finalized:
        checks.equal(set(policies), set(TIMING_CASES), "complete policy summary grid")
        checks.equal(len(row["warmup_calls"]), 7, "finalized warmup call count")
        checks.equal(len(row["timed_calls"]), 49, "finalized timed call count")
        checks.equal(set(row["native_parity"]), {"current", "previous"}, "finalized parity coverage")
        _require(checks, all(row["native_parity"][key]["passed"] for key in ("current", "previous")), "finalized parity gates")
        checks.equal(row["status"] == "complete", all(p["status"] == "complete" for p in policies.values()),
                     "finalized frame/policy completion")
    for method, record in policies.items():
        _require(checks, method in TIMING_CASES, "known policy summary")
        calls = [c for c in row["timed_calls"] if c["method"] == method]
        warmups = [c for c in row["warmup_calls"] if c["method"] == method]
        # Reproduce stable[method], including a rejected warmup. A failed
        # summary is subject to the same predicate as a complete summary;
        # otherwise dropping a successful arm could manufacture a failure.
        stable = bool(all(c["status"] == "complete" for c in warmups + calls)
                      and all(all(c.get("repeat_consistency", {}).values()) for c in calls))
        checks.equal(record["repeat_consistent"], stable, method + ": repeat consistency status")
        complete = bool(method in reference and stable and len(calls) == 7
                        and all(c["status"] == "complete" for c in calls))
        checks.equal(record["status"], "complete" if complete else "failed", method + ": policy completion status")
        checks.equal(record["completed_repetitions"], sum(c["status"] == "complete" for c in calls), method + ": complete repeats")
        checks.equal(record["expected_repetitions"], 7, method + ": fixed repeats")
        flat.update(_timing(calls, method, record, checks))
        if record["status"] != "complete":
            checks.equal(record.get("metrics"), None, method + ": failed accuracy remains undefined")
            continue
        _require(checks, len(warmups) == 1 and warmups[0]["status"] == "complete", method + ": complete warmup")
        _require(checks, method in reference and len(calls) == 7 and all(c["status"] == "complete" for c in calls)
                 and record["repeat_consistent"] is True, method + ": complete repetition evidence")
        _require(checks, all(all(c.get("repeat_consistency", {}).values()) for c in calls),
                 method + ": completion cannot hide a recorded repeat mismatch")
        target, std = arrays["target_position"], arrays["acceleration_std"]
        _require(checks, target.shape == (n, 2) and std.shape == (2,) and np.isfinite(target).all()
                 and np.isfinite(std).all() and np.all(std > 0), "target and acceleration scales")
        checks.equal(target.dtype.str, np.dtype(np.float64).str, "same-state target float64")
        checks.equal(std.dtype.str, np.dtype(np.float64).str, "saved acceleration scales float64")
        prediction = reference[method][1]["prediction"]
        residual = prediction.astype(np.float64) - target
        normalized = residual / std
        checks.array(arrays["position_residual__" + method], residual, method + ": position residual")
        checks.array(arrays["normalized_residual__" + method], normalized, method + ": normalized residual")
        metrics = {"position_coordinate_mse": float(np.mean(residual ** 2)),
                   "normalized_coordinate_mse": float(np.mean(normalized ** 2))}
        _check_numbers(record["metrics"], metrics, checks, method + ": accuracy")
        _check_numbers(record["prediction_boundary"], _boundary(prediction, bounds), checks, method + ": boundary")
        errors[method] = (residual ** 2).sum(-1), (normalized ** 2).sum(-1)
        if method in POLICIES:
            flat.update({"accuracy/" + method + "/" + key: value for key, value in metrics.items()})
    if "truth_boundary" in row:
        _check_numbers(row["truth_boundary"], _boundary(arrays["target_position"], bounds), checks, "target boundary")
    expected_benefit, expected_corr = {}, {}
    risk_ready = POLICIES[-1] in errors
    if risk_ready:
        risk = reference[POLICIES[-1]][2]["risk"]
        checks.array(arrays["previous_observed_base_risk"], risk, "previous observed risk alias")
        if "base" in errors:
            expected_corr["previous_risk_vs_base_error"] = _spearman(risk, errors["base"][1])
    if "base" in errors:
        for method in POLICIES[1:]:
            if method not in errors:
                continue
            position = errors["base"][0] - errors[method][0]
            normalized = errors["base"][1] - errors[method][1]
            expected_benefit[method] = {"mean_position_vector_benefit": float(position.mean()),
                "mean_normalized_vector_benefit": float(normalized.mean()), "positive_fraction": float(np.mean(normalized > 0)),
                "negative_fraction": float(np.mean(normalized < 0)), "zero_fraction": float(np.mean(normalized == 0))}
            if finalized or method in row.get("benefit", {}):
                checks.array(arrays["signed_position_benefit__" + method], position, method + ": signed position benefit")
                checks.array(arrays["signed_normalized_benefit__" + method], normalized, method + ": signed normalized benefit")
            if risk_ready:
                expected_corr["previous_risk_vs_" + method + "_benefit"] = _spearman(risk, normalized)
            if method != "dense" and "dense" in errors:
                dense = errors["base"][1] - errors["dense"][1]
                expected_benefit[method].update(dense_sparse_sign_disagreement_fraction=float(np.mean(np.sign(dense) != np.sign(normalized))),
                    dense_positive_sparse_nonpositive_fraction=float(np.mean((dense > 0) & (normalized <= 0))))
                expected_corr["dense_vs_" + method + "_benefit"] = _spearman(dense, normalized)
    for family, expected in (("benefit", expected_benefit), ("correlations", expected_corr)):
        actual = row.get(family, {})
        if finalized:
            checks.equal(set(actual), set(expected), family + ": available complete dependencies")
        for key, value in actual.items():
            _require(checks, key in expected, family + ": reconstructible saved metric")
            _check_numbers(value, expected[key], checks, family + "/" + key)
            if family == "benefit":
                flat.update({family + "/" + key + "/" + name: number for name, number in expected[key].items()})
            else:
                flat[family + "/" + key] = expected[key]["value"]
    return flat


def _clean(row, arrays, bounds, checks):
    types, history = arrays["particle_types"], arrays["observed_history"]
    n = len(types)
    checks.equal(history.shape, (6, n, 2), "clean history shape")
    checks.equal(history.dtype.str, np.dtype(np.float32).str, "clean history float32")
    _parity(row.get("native_parity"), arrays, "parity_", arrays.get("native_edges"), n, checks)
    if row["status"] != "complete":
        checks.equal(row.get("metrics"), None, "failed clean metrics remain undefined")
        if (row.get("failure") or {}).get("category") == "nonfinite_or_nonpositive_clean_prediction_variance_target":
            values = [arrays[key] for key in ("normalized_prediction", "raw_head", "predicted_variance", "normalized_target")]
            _require(checks, not all(np.isfinite(value).all() for value in values)
                     or not np.all(arrays["predicted_variance"] > 0), "reproduce saved clean numerical guard")
        return {}
    _require(checks, isinstance(row.get("native_parity"), dict) and row["native_parity"]["passed"] is True,
             "complete clean parity gate")
    p, q, y, head = (arrays[key] for key in ("normalized_prediction", "predicted_variance", "normalized_target", "raw_head"))
    checks.equal(p.shape, (n, 2), "clean prediction shape")
    checks.equal(y.shape, p.shape, "clean target shape")
    checks.equal(q.shape, (n,), "clean variance shape")
    checks.equal(head.shape, q.shape, "clean head shape")
    for key in ("normalized_prediction", "predicted_variance", "normalized_target"):
        checks.equal(arrays[key].dtype.str, np.dtype(np.float64).str, "clean stored double " + key)
    checks.equal(head.dtype.str, np.dtype(np.float32).str, "clean head float32")
    checks.equal(arrays["target_position"].shape, (n, 2), "clean source target shape")
    checks.equal(arrays["target_position"].dtype.str, np.dtype(np.float32).str, "clean source target float32")
    _require(checks, all(np.isfinite(value).all() for value in (p, q, y, head)) and np.all(q > 0), "accepted clean finite positive arrays")
    checks.array(q, np.maximum(head, np.float32(1e-6)).astype(np.float64), "clean scalar variance floor")
    se = ((p - y) ** 2).sum(-1)
    checks.array(arrays["normalized_vector_se"], se, "clean normalized vector squared error")
    metrics = {"normalized_acceleration_coordinate_mse": float(se.mean() / 2),
               "realized_normalized_vector_se": float(se.mean()), "predicted_normalized_vector_se": float(2 * q.mean()),
               "constant_free_gaussian_nll": float((.5 * se / q + np.log(q)).mean())}
    _check_numbers(row["metrics"], metrics, checks, "clean metrics")
    return {"metrics/" + key: value for key, value in metrics.items()}


def audit_row(row, arrays, bounds, checks, *, mode):
    """Return reconstructed flat metrics; checks exposes equal/close/array.

    mode is explicit: ``same-state`` or ``clean-validation``. ``arrays`` must be
    an already hash-admitted mapping of all NPZ keys to numeric ndarrays. No IO.
    Checks.close must accept two None values; arrays containing rejected NaNs
    are only byte/descriptor-checked or compared with explicit equal_nan logic.
    """
    _require(checks, mode in ("same-state", "clean-validation"), "known diagnostic mode")
    _require(checks, row.get("status") in ("complete", "failed"), "committed diagnostic status")
    checks.equal(row.get("failure") is None, row["status"] == "complete", "diagnostic failure/status")
    bounds = np.asarray(bounds, dtype=np.float64)
    _require(checks, bounds.shape == (2, 2) and np.isfinite(bounds).all() and np.all(bounds[:, 1] > bounds[:, 0]), "physical bounds")
    _descriptors(row, arrays, checks)
    types = arrays["particle_types"]
    _require(checks, types.ndim == 1 and len(types) > 0 and types.dtype.kind in "iu" and np.all(types == 6), "Sand particle types")
    return _same_state(row, arrays, bounds, checks) if mode == "same-state" else _clean(row, arrays, bounds, checks)
