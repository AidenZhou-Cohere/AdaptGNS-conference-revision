"""Synthetic scientific checks for the NumPy-only saved diagnostic auditor.

All arrays below are generated in memory. No evaluator, graph constructor,
checkpoint, trajectory, collection, or result artifact is imported or opened.
The main same-state oracle uses explicit hand-calculated residual/error/benefit
tables and tied-rank correlations. Mutations normally refresh byte descriptors
so semantic rejection cannot be credited merely to an obsolete hash.

Run: work/venv/bin/python -B <this file>
"""
import copy
import hashlib
import math
import unittest

import numpy as np

import goop3d_saved_diagnostic_audit_v1 as diagnostic
from goop3d_observed_history_arithmetic_v1 import Checks


POLICIES = ("base", "dense", "random25", "speed25",
            "relative-velocity-RMS25", "previous-observed-base-risk25")
CASES = POLICIES + ("natural_base_reference",)
TIMES = ("end_to_end_seconds", "score_generation_seconds", "score_graph_seconds",
         "score_forward_seconds", "current_graph_and_selection_seconds", "current_forward_seconds")
BOUNDS = np.array([[-1., 1.], [-1., 1.], [-1., 1.]])
PREVIOUS_RISK = np.array([1., 2., 2., 4.], dtype=np.float32)
HEAD = np.array([-2., .25, 1., 4.], dtype=np.float32)
FLOOR = float(np.float32(1e-6))

# Target is zero and acceleration standard deviations are (2, 1). Explicit
# normalized residuals make an accidental dt**2 factor or coordinate/vector
# factor of two visible without reproducing the auditor's vector arithmetic.
PREDICTIONS = {
    "base": ((2, 0), (0, 2), (2, 2), (0, 0)),
    "dense": ((0, 0), (0, 3), (2, 0), (0, 0)),
    "random25": ((2, 0), (0, 0), (0, 3), (0, 0)),
    "speed25": ((0, 2), (2, 2), (0, 0), (0, 0)),
    "relative-velocity-RMS25": ((0, 1), (0, 2), (2, 2), (0, 1)),
    "previous-observed-base-risk25": ((0, 1), (2, 0), (0, 2), (0, 0)),
}
PREDICTIONS[CASES[-1]] = PREDICTIONS["base"]
NORMALIZED = {
    "base": ((1, 0), (0, 2), (1, 2), (0, 0)),
    "dense": ((0, 0), (0, 3), (1, 0), (0, 0)),
    "random25": ((1, 0), (0, 0), (0, 3), (0, 0)),
    "speed25": ((0, 2), (1, 2), (0, 0), (0, 0)),
    "relative-velocity-RMS25": ((0, 1), (0, 2), (1, 2), (0, 1)),
    "previous-observed-base-risk25": ((0, 1), (1, 0), (0, 2), (0, 0)),
}
NORMALIZED[CASES[-1]] = NORMALIZED["base"]
MSES = {"base": (2, 5/4), "dense": (13/8, 5/4), "random25": (13/8, 5/4),
        "speed25": (3/2, 9/8), "relative-velocity-RMS25": (7/4, 11/8),
        "previous-observed-base-risk25": (9/8, 3/4), CASES[-1]: (2, 5/4)}
POSITION_BENEFITS = {"dense": (4, -5, 4, 0), "random25": (0, 4, -1, 0),
                     "speed25": (0, -4, 8, 0), "relative-velocity-RMS25": (3, 0, 0, -1),
                     "previous-observed-base-risk25": (3, 0, 4, 0)}
NORMALIZED_BENEFITS = {"dense": (1, -5, 4, 0), "random25": (0, 4, -4, 0),
                       "speed25": (-3, -1, 5, 0), "relative-velocity-RMS25": (0, 0, 0, -1),
                       "previous-observed-base-risk25": (0, 3, 1, 0)}
# position mean, normalized mean, positive, negative, zero, disagreement,
# dense-positive/sparse-nonpositive; dense has no final two leaves.
BENEFIT_SCALARS = {"dense": (3/4, 0, 1/2, 1/4, 1/4),
                   "random25": (3/4, 0, 1/4, 1/4, 1/2, 3/4, 1/2),
                   "speed25": (1, 1/4, 1/4, 1/2, 1/4, 1/4, 1/4),
                   "relative-velocity-RMS25": (1/2, -1/4, 0, 1/4, 3/4, 1, 1/2),
                   "previous-observed-base-risk25": (7/4, 1, 1/2, 0, 1/2, 1/2, 1/4)}
BENEFIT_NAMES = ("mean_position_vector_benefit", "mean_normalized_vector_benefit",
                 "positive_fraction", "negative_fraction", "zero_fraction",
                 "dense_sparse_sign_disagreement_fraction", "dense_positive_sparse_nonpositive_fraction")
# Previous-risk ranks are (1, 2.5, 2.5, 4); centered squared norm is 4.5.
CORRELATIONS = {"previous_risk_vs_base_error": -1/math.sqrt(10),
                "previous_risk_vs_dense_benefit": -1/math.sqrt(10),
                "previous_risk_vs_random25_benefit": 0,
                "previous_risk_vs_speed25_benefit": 2/math.sqrt(10),
                "previous_risk_vs_relative-velocity-RMS25_benefit": -math.sqrt(2/3),
                "previous_risk_vs_previous-observed-base-risk25_benefit": 0,
                "dense_vs_random25_benefit": -3/math.sqrt(10),
                "dense_vs_speed25_benefit": 2/5,
                "dense_vs_relative-velocity-RMS25_benefit": 1/math.sqrt(15),
                "dense_vs_previous-observed-base-risk25_benefit": -1/math.sqrt(10)}


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def seal(row, arrays):
    row["numeric_arrays"] = {
        key: {"shape": list(value.shape), "dtype": value.dtype.str, "value_sha256": sha(value)}
        for key, value in arrays.items()}


def boundary_record(outside, maximum, mean, low, high):
    return {"fraction_particles_outside": outside,
            "fraction_particles_outside_by_more_than_1e-6": outside,
            "maximum_coordinate_excursion": maximum, "mean_particle_maximum_excursion": mean,
            "coordinate_minimum": list(low), "coordinate_maximum": list(high)}


BOUNDARIES = {"base": boundary_record(3/4, 1, 3/4, (0, 0), (2, 2)),
              "dense": boundary_record(1/2, 2, 3/4, (0, 0), (2, 3)),
              "random25": boundary_record(1/2, 2, 3/4, (0, 0), (2, 3)),
              "speed25": boundary_record(1/2, 1, 1/2, (0, 0), (2, 2)),
              "relative-velocity-RMS25": boundary_record(1/2, 1, 1/2, (0, 1), (2, 2)),
              "previous-observed-base-risk25": boundary_record(1/2, 1, 1/2, (0, 0), (2, 2))}
BOUNDARIES[CASES[-1]] = BOUNDARIES["base"]


PREDICTIONS = {k: tuple((*v, 0) for v in rows) for k, rows in PREDICTIONS.items()}
NORMALIZED = {k: tuple((*v, 0) for v in rows) for k, rows in NORMALIZED.items()}
MSES = {k: tuple(value*2/3 for value in values) for k, values in MSES.items()}
for value in BOUNDARIES.values():
    if len(value['coordinate_minimum']) == 2:
        value['coordinate_minimum'].append(0)
        value['coordinate_maximum'].append(0)


def add_outputs(arrays, prefix, prediction, head=HEAD):
    arrays[prefix + "prediction"] = np.array(prediction, dtype=np.float32)
    arrays[prefix + "raw_risk"] = head.copy()
    # Both floor branches are exercised; this is an explicit oracle, not a call
    # to the implementation's head conversion or a frozen model helper.
    arrays[prefix + "risk"] = np.array([max(float(x), FLOOR) for x in head], dtype=np.float32)


def add_parity(arrays, prefix, prediction, head=HEAD):
    edges = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    for name in ("native_", "supplied_"):
        add_outputs(arrays, prefix + name, prediction, head)
        arrays[prefix + name + "node_features"] = np.arange(148, dtype=np.float32).reshape(4, 37)/32
        arrays[prefix + name + "edge_features"] = np.zeros((4, 4), dtype=np.float32)
    arrays[prefix + "native_edges"] = edges
    return {"finite": True, "prediction_risk_agree": True,
            "prediction_atol": 2e-7, "risk_atol": 1e-6, "risk_rtol": 1e-5,
            "prediction_max_abs_difference": 0., "risk_max_abs_difference": 0.,
            "raw_risk_max_abs_difference": 0., "native_supplied_feature_identity": [True, True, True],
            "native_edge_identity": True, "passed": True}


def same_state_fixture():
    # Consecutive frame numbers make an off-by-one history slice detectable.
    arrays = {"current_history": np.broadcast_to(np.arange(1, 7, dtype=np.float32)[:, None, None], (6, 4, 3)).copy(),
              "previous_history": np.broadcast_to(np.arange(6, dtype=np.float32)[:, None, None], (6, 4, 3)).copy(),
              "particle_types": np.full(4, 7, dtype=np.int64),
              "target_position": np.zeros((4, 3), dtype=np.float64),
              "acceleration_std": np.array([2., 1., 1.], dtype=np.float64)}
    row = {"source_index": 2, "schedule_index": 3, "target_frame": 226,
           "status": "complete", "failure": None, "warmup_calls": [], "timed_calls": [],
           "policies": {}, "benefit": {}, "correlations": {}}
    # Explicit order for source=2, schedule=3: round 0 starts with natural;
    # round 1 starts with base; subsequent starts advance one method each.
    orders = (CASES[-1:] + CASES[:-1],) + tuple(CASES[i:] + CASES[:i] for i in range(6))
    for repetition in range(-1, 7):
        family = "warmup_calls" if repetition == -1 else "timed_calls"
        order = orders[0] if repetition == -1 else orders[repetition]
        for slot, method in enumerate(order):
            prefix = f"r{repetition + 1}_{method}__"
            add_outputs(arrays, prefix, PREDICTIONS[method])
            call = {"round": repetition, "slot": slot, "method": method, "status": "complete",
                    "prediction_sha256": sha(arrays[prefix + "prediction"]),
                    "timing": {name: (1000 if repetition < 0 else repetition + 1)*(CASES.index(method)+1)*(k+1)/8
                               for k, name in enumerate(TIMES)}}
            if method == POLICIES[-1]:
                add_outputs(arrays, prefix + "previous_base_", PREDICTIONS["base"], PREVIOUS_RISK)
            if repetition == 0:
                arrays[prefix + "edges"] = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
                if method == POLICIES[-1]:
                    arrays[prefix + "previous_base_edges"] = arrays[prefix + "edges"].copy()
            if repetition > 0:
                call["repeat_consistency"] = {"ordered_edges_exact": True, "outputs_within_fixed_parity_tolerance": True}
                if method == POLICIES[-1]:
                    call["repeat_consistency"].update(previous_score_ordered_edges_exact=True,
                                                     previous_score_outputs_within_fixed_parity_tolerance=True)
                if method == POLICIES[-2]:
                    call["repeat_consistency"]["physical_scores_exact"] = True
            row[family].append(call)
    for i, method in enumerate(CASES):
        row["policies"][method] = {
            "status": "complete", "repeat_consistent": True, "completed_repetitions": 7, "expected_repetitions": 7,
            "metrics": dict(zip(("position_coordinate_mse", "normalized_coordinate_mse"), MSES[method])),
            "prediction_boundary": copy.deepcopy(BOUNDARIES[method]),
            "timing": {name: {"observations": 7, "expected": 7, "mean": 4*(i+1)*(k+1)/8,
                              "median": 4*(i+1)*(k+1)/8, "sample_sd": math.sqrt(14/3)*(i+1)*(k+1)/8}
                       for k, name in enumerate(TIMES)}}
        arrays["position_residual__" + method] = np.array(PREDICTIONS[method], dtype=np.float64)
        arrays["normalized_residual__" + method] = np.array(NORMALIZED[method], dtype=np.float64)
    for method in POLICIES[1:]:
        row["benefit"][method] = dict(zip(BENEFIT_NAMES, BENEFIT_SCALARS[method]))
        arrays["signed_position_benefit__" + method] = np.array(POSITION_BENEFITS[method], dtype=np.float64)
        arrays["signed_normalized_benefit__" + method] = np.array(NORMALIZED_BENEFITS[method], dtype=np.float64)
    row["correlations"] = {name: {"value": value, "reason": None, "particles": 4} for name, value in CORRELATIONS.items()}
    row["native_parity"] = {"current": add_parity(arrays, "current_parity_", PREDICTIONS["base"]),
                            "previous": add_parity(arrays, "previous_parity_", PREDICTIONS["base"], PREVIOUS_RISK)}
    row["natural_shared_base"] = {"ordered_edges_exact": True, "prediction_max_abs_difference": 0.,
                                   "risk_max_abs_difference": 0., "raw_risk_max_abs_difference": 0.,
                                   "outputs_within_fixed_parity_tolerance": True}
    row["truth_boundary"] = boundary_record(0., 0., 0., (0, 0, 0), (0, 0, 0))
    arrays["previous_observed_base_risk"] = PREVIOUS_RISK.copy()
    seal(row, arrays)
    return row, arrays


def clean_fixture(*, floor=False):
    arrays = {"particle_types": np.full(4, 7, dtype=np.int64),
              "observed_history": np.zeros((6, 4, 3), dtype=np.float32),
              "target_position": np.zeros((4, 3), dtype=np.float32),
              "normalized_prediction": np.array([[1, 2, 1], [0, 0, 0], [2, -1, 2], [1, 1, 0]], dtype=np.float64),
              "normalized_target": np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float64),
              "raw_head": np.array([1, 2, 4, 8], dtype=np.float32),
              "predicted_variance": np.array([1, 2, 4, 8], dtype=np.float64),
              "normalized_vector_se": np.array([6, 1, 9, 2], dtype=np.float64),
              "native_edges": np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)}
    metrics = dict(zip(("normalized_acceleration_coordinate_mse", "realized_normalized_vector_se",
                        "predicted_normalized_vector_se", "constant_free_gaussian_nll"),
                       (3/2, 9/2, 45/4, 9/8 + 9/4*math.log(2))))
    if floor:
        arrays["raw_head"][:2] = [-1, 0]
        arrays["predicted_variance"][:2] = FLOOR
        arrays["normalized_prediction"][:2] = arrays["normalized_target"][:2]
        arrays["normalized_vector_se"][:2] = 0
        metrics = dict(zip(metrics, (11/12, 11/4, 9 + 1.5*FLOOR, 5/16 + 3/4*math.log(FLOOR) + 15/8*math.log(2))))
    row = {"status": "complete", "failure": None, "metrics": metrics,
           "native_parity": add_parity(arrays, "parity_", PREDICTIONS["base"])}
    seal(row, arrays)
    return row, arrays


def discard_failed_policy_metrics(row, arrays, method):
    """Model the evaluator's finalized omission of a failed policy's leaves."""
    row.update(status="failed", failure={"category": "policy_guard_or_repeat_failure"})
    record = row["policies"][method]
    record.update(status="failed", repeat_consistent=False, metrics=None)
    record.pop("prediction_boundary")
    for prefix in ("position_residual__", "normalized_residual__", "signed_position_benefit__", "signed_normalized_benefit__"):
        arrays.pop(prefix + method)
    row["benefit"].pop(method)
    row["correlations"].pop("previous_risk_vs_" + method + "_benefit")
    row["correlations"].pop("dense_vs_" + method + "_benefit")


