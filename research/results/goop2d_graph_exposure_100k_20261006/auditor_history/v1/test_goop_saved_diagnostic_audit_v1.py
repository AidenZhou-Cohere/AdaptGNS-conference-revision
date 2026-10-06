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

import goop_saved_diagnostic_audit_v1 as diagnostic
from audit_goop_saved_arrays_v1 import Checks


POLICIES = ("base", "dense", "random25", "speed25",
            "relative-velocity-RMS25", "previous-observed-base-risk25")
CASES = POLICIES + ("natural_base_reference",)
TIMES = ("end_to_end_seconds", "score_generation_seconds", "score_graph_seconds",
         "score_forward_seconds", "current_graph_and_selection_seconds", "current_forward_seconds")
BOUNDS = np.array([[-1., 1.], [-1., 1.]])
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
        arrays[prefix + name + "node_features"] = np.arange(120, dtype=np.float32).reshape(4, 30)/32
        arrays[prefix + name + "edge_features"] = np.zeros((4, 3), dtype=np.float32)
    arrays[prefix + "native_edges"] = edges
    return {"finite": True, "prediction_risk_agree": True,
            "prediction_atol": 2e-7, "risk_atol": 1e-6, "risk_rtol": 1e-5,
            "prediction_max_abs_difference": 0., "risk_max_abs_difference": 0.,
            "raw_risk_max_abs_difference": 0., "native_supplied_feature_identity": [True, True, True],
            "native_edge_identity": True, "passed": True}


def same_state_fixture():
    # Consecutive frame numbers make an off-by-one history slice detectable.
    arrays = {"current_history": np.broadcast_to(np.arange(1, 7, dtype=np.float32)[:, None, None], (6, 4, 2)).copy(),
              "previous_history": np.broadcast_to(np.arange(6, dtype=np.float32)[:, None, None], (6, 4, 2)).copy(),
              "particle_types": np.full(4, 7, dtype=np.int64),
              "target_position": np.zeros((4, 2), dtype=np.float64),
              "acceleration_std": np.array([2., 1.], dtype=np.float64)}
    row = {"source_index": 2, "schedule_index": 3, "target_frame": 203,
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
    row["truth_boundary"] = boundary_record(0., 0., 0., (0, 0), (0, 0))
    arrays["previous_observed_base_risk"] = PREVIOUS_RISK.copy()
    seal(row, arrays)
    return row, arrays


def clean_fixture(*, floor=False):
    arrays = {"particle_types": np.full(4, 7, dtype=np.int64),
              "observed_history": np.zeros((6, 4, 2), dtype=np.float32),
              "target_position": np.zeros((4, 2), dtype=np.float32),
              "normalized_prediction": np.array([[1, 2], [0, 0], [2, -1], [1, 1]], dtype=np.float64),
              "normalized_target": np.array([[0, 0], [0, 1], [1, 1], [2, 0]], dtype=np.float64),
              "raw_head": np.array([1, 2, 4, 8], dtype=np.float32),
              "predicted_variance": np.array([1, 2, 4, 8], dtype=np.float64),
              "normalized_vector_se": np.array([5, 1, 5, 2], dtype=np.float64),
              "native_edges": np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)}
    metrics = dict(zip(("normalized_acceleration_coordinate_mse", "realized_normalized_vector_se",
                        "predicted_normalized_vector_se", "constant_free_gaussian_nll"),
                       (13/8, 13/4, 15/2, 7/8 + 3/2*math.log(2))))
    if floor:
        arrays["raw_head"][:2] = [-1, 0]
        arrays["predicted_variance"][:2] = FLOOR
        arrays["normalized_prediction"][:2] = arrays["normalized_target"][:2]
        arrays["normalized_vector_se"][:2] = 0
        metrics = dict(zip(metrics, (7/8, 7/4, 6 + FLOOR, 3/16 + math.log(FLOOR)/2 + 5/4*math.log(2))))
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


class DiagnosticAuditTests(unittest.TestCase):
    def audit(self, fixture, mode="same-state"):
        checks = Checks()
        values = diagnostic.audit_row(*fixture, BOUNDS, checks, mode=mode)
        self.assertGreater(checks.count, 0)
        return values

    def rejects(self, fixture, label, mode="same-state", *, reseal=True):
        if reseal:
            seal(*fixture)
        with self.assertRaisesRegex(ValueError, label):
            self.audit(fixture, mode)

    def test_complete_same_state_hand_oracle_and_fixed_metric_grid(self):
        values = self.audit(same_state_fixture())
        self.assertEqual(len(values), 97)
        self.assertEqual(set(values), set(diagnostic.expected_metric_keys("same-state")))
        self.assertEqual(len(set(diagnostic.expected_metric_keys("same-state"))), 97)
        for method in POLICIES:
            self.assertEqual(values[f"accuracy/{method}/position_coordinate_mse"], MSES[method][0])
            self.assertEqual(values[f"accuracy/{method}/normalized_coordinate_mse"], MSES[method][1])
        for method, scalars in BENEFIT_SCALARS.items():
            for key, expected in zip(BENEFIT_NAMES, scalars):
                self.assertEqual(values[f"benefit/{method}/{key}"], expected)
        for key, expected in CORRELATIONS.items():
            self.assertAlmostEqual(values["correlations/" + key], expected, places=14)
        self.assertEqual(values["timing/base/end_to_end_seconds"], .5)
        self.assertEqual(values["timing/natural_base_reference/current_forward_seconds"], 21)
        self.assertNotIn("accuracy/natural_base_reference/position_coordinate_mse", values)

    def test_clean_hand_oracle_and_fixed_metric_grid(self):
        row, arrays = clean_fixture()
        values = self.audit((row, arrays), "clean-validation")
        self.assertEqual(set(values), set(diagnostic.expected_metric_keys("clean-validation")))
        self.assertEqual(len(values), 4)
        self.assertEqual(values["metrics/normalized_acceleration_coordinate_mse"], 13/8)
        self.assertEqual(values["metrics/realized_normalized_vector_se"], 13/4)
        self.assertEqual(values["metrics/predicted_normalized_vector_se"], 15/2)
        self.assertAlmostEqual(values["metrics/constant_free_gaussian_nll"], 7/8 + 1.5*math.log(2), places=14)

    def test_clean_zero_and_negative_heads_use_float32_floor(self):
        values = self.audit(clean_fixture(floor=True), "clean-validation")
        self.assertEqual(values["metrics/predicted_normalized_vector_se"], 6 + FLOOR)
        self.assertLess(values["metrics/constant_free_gaussian_nll"], 0)

    def test_numeric_descriptors_catch_inventory_dtype_shape_and_bytes(self):
        mutations = {
            "inventory": lambda r, a: r["numeric_arrays"].pop("target_position"),
            "dtype": lambda r, a: r["numeric_arrays"]["target_position"].update(dtype="<f8"),
            "shape": lambda r, a: r["numeric_arrays"]["target_position"].update(shape=[8]),
            "bytes": lambda r, a: a["target_position"].__setitem__((0, 0), .5),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                fixture = clean_fixture()
                mutate(*fixture)
                self.rejects(fixture, "numeric descriptor", "clean-validation", reseal=False)

    def test_object_array_rejected_before_any_scientific_use(self):
        row, arrays = clean_fixture()
        arrays["untrusted_extra"] = np.array([object()], dtype=object)
        self.rejects((row, arrays), "numeric ndarray", "clean-validation")

    def test_goop_types_and_physical_bounds_are_required(self):
        for invalid in (np.full(4, 6), np.full(4, 7, dtype=np.float32), np.full((2, 2), 7)):
            with self.subTest(shape=invalid.shape, dtype=str(invalid.dtype)):
                row, arrays = clean_fixture()
                arrays["particle_types"] = invalid
                self.rejects((row, arrays), "Goop particle types", "clean-validation")
        for bounds in (np.zeros((2, 2)), [[0, 1], [0, np.nan]], np.zeros((3, 2))):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, "physical bounds"):
                    diagnostic.audit_row(*clean_fixture(), bounds, Checks(), mode="clean-validation")

    def test_unknown_mode_and_status_failure_contradiction_rejected(self):
        with self.assertRaisesRegex(ValueError, "known diagnostic mode"):
            self.audit(clean_fixture(), "unreviewed")
        with self.assertRaises(ValueError):
            diagnostic.expected_metric_keys("unreviewed")
        row, arrays = clean_fixture()
        row["failure"] = {"category": "execution_error"}
        self.rejects((row, arrays), "failure/status", "clean-validation")

    def test_same_state_history_overlap_and_dtypes(self):
        row, arrays = same_state_fixture()
        arrays["previous_history"][1, 0, 0] += 1
        self.rejects((row, arrays), "overlapping observed histories")
        row, arrays = same_state_fixture()
        arrays["current_history"] = arrays["current_history"].astype(np.float64)
        self.rejects((row, arrays), "current history float32")

    def test_fixed_call_order_and_warmup_prefix_are_checked(self):
        for family in ("warmup_calls", "timed_calls"):
            with self.subTest(family=family):
                row, arrays = same_state_fixture()
                row[family][0], row[family][1] = row[family][1], row[family][0]
                self.rejects((row, arrays), "fixed ordered call prefix")
        row, arrays = same_state_fixture()
        row["warmup_calls"] = row["warmup_calls"][:6]
        self.rejects((row, arrays), "timed calls follow full warmup")

    def test_successful_output_shape_dtype_guards_and_floor(self):
        mutations = (
            ("shape", lambda a: a.__setitem__("r1_base__prediction", np.zeros((4, 3), np.float32))),
            ("float32 output", lambda a: a.__setitem__("r1_base__prediction", a["r1_base__prediction"].astype(np.float64))),
            ("finite accepted output", lambda a: a["r1_base__prediction"].__setitem__((0, 0), np.nan)),
            ("accepted prediction/risk guards", lambda a: a["r1_base__prediction"].__setitem__((0, 0), 11)),
            ("variance floor", lambda a: a["r1_base__risk"].__setitem__(0, 1)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                row, arrays = same_state_fixture()
                mutate(arrays)
                self.rejects((row, arrays), label)

    def test_prediction_hash_and_repeat_output_flags_independently_checked(self):
        row, arrays = same_state_fixture()
        next(c for c in row["timed_calls"] if c["method"] == "base")["prediction_sha256"] = "0"*64
        self.rejects((row, arrays), "prediction hash")
        row, arrays = same_state_fixture()
        arrays["r2_random25__prediction"][0, 0] += .125
        next(c for c in row["timed_calls"] if c["method"] == "random25" and c["round"] == 1)["prediction_sha256"] = sha(arrays["r2_random25__prediction"])
        self.rejects((row, arrays), "repeated outputs")

    def test_previous_score_repeats_and_saved_alias_independently_checked(self):
        row, arrays = same_state_fixture()
        arrays["r2_previous-observed-base-risk25__previous_base_prediction"][0, 0] += .125
        self.rejects((row, arrays), "repeated previous scores")
        row, arrays = same_state_fixture()
        arrays["previous_observed_base_risk"][0] = 2
        self.rejects((row, arrays), "previous observed risk alias")

    def test_completion_cannot_hide_recorded_graph_or_physical_repeat_failure(self):
        for method, flag in (("random25", "ordered_edges_exact"), (POLICIES[-2], "physical_scores_exact")):
            with self.subTest(method=method):
                row, arrays = same_state_fixture()
                next(c for c in row["timed_calls"] if c["method"] == method and c["round"] == 1)["repeat_consistency"][flag] = False
                self.rejects((row, arrays), "completion cannot hide a recorded repeat mismatch|repeat consistency status")

    def test_residual_normalization_and_vector_benefit_are_recomputed(self):
        for key, label in (("position_residual__dense", "position residual"),
                           ("normalized_residual__dense", "normalized residual"),
                           ("signed_position_benefit__dense", "signed position benefit"),
                           ("signed_normalized_benefit__dense", "signed normalized benefit")):
            with self.subTest(key=key):
                row, arrays = same_state_fixture()
                arrays[key] *= 2
                self.rejects((row, arrays), label)

    def test_coordinate_vector_and_dense_sparse_sign_conventions_rejected_if_changed(self):
        row, arrays = same_state_fixture()
        row["policies"]["base"]["metrics"]["position_coordinate_mse"] *= 2
        self.rejects((row, arrays), "accuracy/position_coordinate_mse")
        row, arrays = same_state_fixture()
        row["benefit"]["dense"]["mean_position_vector_benefit"] /= 2
        self.rejects((row, arrays), "mean_position_vector_benefit")
        row, arrays = same_state_fixture()
        row["benefit"]["random25"]["dense_sparse_sign_disagreement_fraction"] = .5
        self.rejects((row, arrays), "dense_sparse_sign_disagreement_fraction")

    def test_tied_rank_correlation_is_not_pearson_or_ordinal_rank(self):
        row, arrays = same_state_fixture()
        row["correlations"]["previous_risk_vs_base_error"]["value"] = -.2
        self.rejects((row, arrays), "previous_risk_vs_base_error/value")
        self.assertEqual(diagnostic._spearman([4, 4, 1, 9], [4, 4, 1, 9]),
                         {"value": 1., "reason": None, "particles": 4})

    def test_constant_previous_risk_retains_explicit_null_correlations(self):
        row, arrays = same_state_fixture()
        for key in arrays:
            if key == "previous_observed_base_risk" or "previous_base_risk" in key or "previous_base_raw_risk" in key:
                arrays[key][:] = 2
            if key.startswith("previous_parity_") and (key.endswith("_risk") or key.endswith("_raw_risk")):
                arrays[key][:] = 2
        for key in row["correlations"]:
            if key.startswith("previous_risk_"):
                row["correlations"][key] = {"value": None, "reason": "constant_rank_vector", "particles": 4}
        seal(row, arrays)
        values = self.audit((row, arrays))
        self.assertEqual(sum(v is None for v in values.values()), 6)
        self.assertIsNone(values["correlations/previous_risk_vs_base_error"])
        self.assertEqual(diagnostic._spearman([3], [8]),
                         {"value": None, "reason": "fewer_than_two_particles", "particles": 1})
        self.assertEqual(diagnostic._spearman([1, 2, 3], [9, 9, 9]),
                         {"value": None, "reason": "constant_rank_vector", "particles": 3})

    def test_null_correlation_reason_cannot_be_relabelled(self):
        row, arrays = same_state_fixture()
        row["correlations"]["previous_risk_vs_base_error"] = {"value": None, "reason": "constant_rank_vector", "particles": 4}
        self.rejects((row, arrays), "previous_risk_vs_base_error/value")

    def test_boundary_particle_fraction_and_coordinate_maximum_recomputed(self):
        for owner, key in (("base", "fraction_particles_outside"), ("dense", "maximum_coordinate_excursion"),
                           ("relative-velocity-RMS25", "coordinate_minimum")):
            with self.subTest(owner=owner, key=key):
                row, arrays = same_state_fixture()
                row["policies"][owner]["prediction_boundary"][key] = [0, 0] if key == "coordinate_minimum" else .123
                self.rejects((row, arrays), "boundary/" + key)
        row, arrays = same_state_fixture()
        row["truth_boundary"]["fraction_particles_outside"] = .5
        self.rejects((row, arrays), "target boundary/fraction_particles_outside")

    def test_all_timing_components_use_seven_calls_and_sample_sd(self):
        for component in TIMES:
            for statistic in ("mean", "median", "sample_sd", "observations"):
                with self.subTest(component=component, statistic=statistic):
                    row, arrays = same_state_fixture()
                    row["policies"]["base"]["timing"][component][statistic] += .25
                    self.rejects((row, arrays), component + "/" + statistic)

    def test_negative_or_nonfinite_measured_times_are_rejected(self):
        for value in (-.001, float("nan"), float("inf")):
            with self.subTest(value=value):
                row, arrays = same_state_fixture()
                row["timed_calls"][0]["timing"][TIMES[0]] = value
                self.rejects((row, arrays), "measured values")

    def test_failed_numerical_call_keeps_metrics_and_runtime_undefined(self):
        row, arrays = same_state_fixture()
        method = "random25"
        call = next(c for c in row["timed_calls"] if c["method"] == method and c["round"] == 6)
        call.update(status="failed", failure={"category": "nonfinite_prediction"})
        call.pop("repeat_consistency")
        arrays["r7_random25__prediction"][0, 0] = np.nan
        call["prediction_sha256"] = sha(arrays["r7_random25__prediction"])
        discard_failed_policy_metrics(row, arrays, method)
        row["policies"][method]["completed_repetitions"] = 6
        for record in row["policies"][method]["timing"].values():
            record.update(observations=6, mean=None, median=None, sample_sd=None)
        seal(row, arrays)
        values = self.audit((row, arrays))
        self.assertNotIn("accuracy/random25/position_coordinate_mse", values)
        self.assertNotIn("benefit/random25/positive_fraction", values)
        self.assertNotIn("correlations/previous_risk_vs_random25_benefit", values)
        for component in TIMES:
            self.assertIsNone(values["timing/random25/" + component])
        row["policies"][method]["timing"][TIMES[0]]["mean"] = .5
        self.rejects((row, arrays), "end_to_end_seconds/mean")

    def test_failed_repeat_with_seven_calls_cannot_supply_mean_runtime(self):
        row, arrays = same_state_fixture()
        call = next(c for c in row["timed_calls"] if c["method"] == "random25" and c["round"] == 1)
        arrays["r2_random25__prediction"][0, 0] += .125
        call["prediction_sha256"] = sha(arrays["r2_random25__prediction"])
        call["repeat_consistency"]["outputs_within_fixed_parity_tolerance"] = False
        discard_failed_policy_metrics(row, arrays, "random25")
        seal(row, arrays)
        values = self.audit((row, arrays))
        self.assertEqual(row["policies"]["random25"]["timing"][TIMES[0]]["observations"], 7)
        self.assertIsNone(values["timing/random25/" + TIMES[0]])
        row["policies"]["random25"]["metrics"] = {"position_coordinate_mse": 0}
        self.rejects((row, arrays), "failed accuracy remains undefined")

    def test_failed_warmup_keeps_policy_failed_despite_seven_successful_timed_calls(self):
        row, arrays = same_state_fixture()
        call = next(c for c in row["warmup_calls"] if c["method"] == "random25")
        call.update(status="failed", failure={"category": "nonfinite_prediction"})
        arrays["r0_random25__prediction"][0, 0] = np.nan
        call["prediction_sha256"] = sha(arrays["r0_random25__prediction"])
        discard_failed_policy_metrics(row, arrays, "random25")
        seal(row, arrays)
        values = self.audit((row, arrays))
        self.assertEqual(row["policies"]["random25"]["completed_repetitions"], 7)
        self.assertIsNone(values["timing/random25/" + TIMES[0]])

    def test_finalized_policy_status_matches_complete_repeat_evidence(self):
        for consistency in (True, False):
            with self.subTest(consistency=consistency):
                row, arrays = same_state_fixture()
                discard_failed_policy_metrics(row, arrays, "random25")
                row["policies"]["random25"]["repeat_consistent"] = consistency
                self.rejects((row, arrays), "repeat consistency status|policy completion status")

    def test_early_generic_failures_do_not_invent_downstream_values(self):
        row, arrays = same_state_fixture()
        arrays = {key: arrays[key] for key in ("current_history", "previous_history", "particle_types")}
        row = {"source_index": 2, "schedule_index": 3, "status": "failed", "failure": {"category": "execution_error"},
               "warmup_calls": [], "timed_calls": []}
        seal(row, arrays)
        self.assertEqual(self.audit((row, arrays)), {})
        # A returned failed first warmup need not contain a prediction array.
        row["warmup_calls"] = [{"round": -1, "slot": 0, "method": CASES[-1], "status": "failed"}]
        self.assertEqual(self.audit((row, arrays)), {})
        row, arrays = clean_fixture()
        arrays = {key: arrays[key] for key in ("observed_history", "particle_types")}
        row = {"status": "failed", "failure": {"category": "execution_error"}, "metrics": None}
        seal(row, arrays)
        self.assertEqual(self.audit((row, arrays), "clean-validation"), {})

    def test_parity_feature_edge_and_conjunction_flags_are_checked(self):
        for suffix, label in (("supplied_node_features", "node_features: identity"),
                              ("supplied_edge_features", "edge_features: identity"),
                              ("native_edges", "available supplied edge identity")):
            with self.subTest(suffix=suffix):
                row, arrays = clean_fixture()
                arrays["parity_" + suffix].flat[0] += 1
                self.rejects((row, arrays), label, "clean-validation")
        row, arrays = clean_fixture()
        row["native_parity"]["passed"] = False
        self.rejects((row, arrays), "parity conjunction", "clean-validation")

    def test_parity_fixed_tolerances_and_measured_difference_checked(self):
        row, arrays = clean_fixture()
        row["native_parity"]["prediction_atol"] = 1e-3
        self.rejects((row, arrays), "fixed prediction_atol", "clean-validation")
        row, arrays = clean_fixture()
        arrays["parity_supplied_prediction"][0, 1] = np.float32(1e-7)
        row["native_parity"]["prediction_max_abs_difference"] = float(np.float32(1e-7))
        seal(row, arrays)
        self.audit((row, arrays), "clean-validation")
        row["native_parity"]["prediction_max_abs_difference"] = 0
        self.rejects((row, arrays), "prediction: difference", "clean-validation")

    def test_failed_nonfinite_parity_preserves_null_differences(self):
        row, arrays = clean_fixture()
        row.update(status="failed", failure={"category": "native_parity_failure"}, metrics=None)
        arrays["parity_supplied_prediction"][0, 0] = np.nan
        row["native_parity"].update(finite=False, prediction_risk_agree=False, passed=False,
                                     prediction_max_abs_difference=None, risk_max_abs_difference=None,
                                     raw_risk_max_abs_difference=None)
        seal(row, arrays)
        self.assertEqual(self.audit((row, arrays), "clean-validation"), {})
        row["native_parity"]["risk_max_abs_difference"] = 0
        self.rejects((row, arrays), "risk: difference", "clean-validation")

    def test_natural_shared_base_saved_edges_and_outputs_are_checked(self):
        row, arrays = same_state_fixture()
        arrays["r1_natural_base_reference__edges"][0, 0] = 1
        self.rejects((row, arrays), "natural/shared available edges")
        row, arrays = same_state_fixture()
        row["natural_shared_base"]["prediction_max_abs_difference"] = 1
        self.rejects((row, arrays), "natural/shared prediction")

    def test_finalized_requires_natural_shared_base_gate(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                row, arrays = same_state_fixture()
                if missing:
                    row.pop("natural_shared_base")
                else:
                    arrays["r1_natural_base_reference__edges"][0, 0] = 1
                    row["natural_shared_base"]["ordered_edges_exact"] = False
                self.rejects((row, arrays), "natural/shared base gate")

    def test_recorded_natural_base_gate_failure_retains_no_downstream_metrics(self):
        row, arrays = same_state_fixture()
        arrays["r1_natural_base_reference__edges"][0, 0] = 1
        row["natural_shared_base"]["ordered_edges_exact"] = False
        row.update(status="failed", failure={"category": "execution_error"}, policies={}, benefit={}, correlations={})
        row.pop("truth_boundary")
        arrays = {k: v for k, v in arrays.items() if k not in ("target_position", "acceleration_std", "previous_observed_base_risk")
                  and not k.startswith(("position_residual__", "normalized_residual__", "signed_"))}
        seal(row, arrays)
        self.assertEqual(self.audit((row, arrays)), {})

    def test_clean_normalized_error_mse_variance_and_nll_are_checked(self):
        for key in ("normalized_acceleration_coordinate_mse", "realized_normalized_vector_se",
                    "predicted_normalized_vector_se", "constant_free_gaussian_nll"):
            with self.subTest(key=key):
                row, arrays = clean_fixture()
                row["metrics"][key] += .5
                self.rejects((row, arrays), "clean metrics/" + key, "clean-validation")
        row, arrays = clean_fixture()
        arrays["normalized_vector_se"][0] = 10
        self.rejects((row, arrays), "normalized vector squared error", "clean-validation")
        row, arrays = clean_fixture()
        arrays["predicted_variance"][0] = 2
        self.rejects((row, arrays), "scalar variance floor", "clean-validation")

    def test_clean_required_saved_dtypes_and_shapes(self):
        for key, dtype, label in (("normalized_prediction", np.float32, "stored double"),
                                   ("predicted_variance", np.float32, "stored double"),
                                   ("normalized_target", np.float32, "stored double"),
                                   ("raw_head", np.float64, "head float32"),
                                   ("target_position", np.float64, "source target float32")):
            with self.subTest(key=key):
                row, arrays = clean_fixture()
                arrays[key] = arrays[key].astype(dtype)
                self.rejects((row, arrays), label, "clean-validation")
        row, arrays = clean_fixture()
        arrays["normalized_prediction"] = np.zeros((4, 3), dtype=np.float64)
        self.rejects((row, arrays), "clean prediction shape", "clean-validation")

    def test_clean_rejected_numerical_guard_is_reproduced_for_each_cause(self):
        for key, value in (("normalized_prediction", np.nan), ("normalized_target", np.inf),
                            ("raw_head", np.nan), ("predicted_variance", 0.), ("predicted_variance", -1.)):
            with self.subTest(key=key, value=value):
                row, arrays = clean_fixture()
                row.update(status="failed", failure={"category": "nonfinite_or_nonpositive_clean_prediction_variance_target"}, metrics=None)
                arrays[key].flat[0] = value
                arrays.pop("normalized_vector_se")
                seal(row, arrays)
                self.assertEqual(self.audit((row, arrays), "clean-validation"), {})

    def test_clean_fabricated_numerical_failure_or_failed_metrics_rejected(self):
        row, arrays = clean_fixture()
        row.update(status="failed", failure={"category": "nonfinite_or_nonpositive_clean_prediction_variance_target"}, metrics=None)
        self.rejects((row, arrays), "reproduce saved clean numerical guard", "clean-validation")
        row, arrays = clean_fixture()
        row.update(status="failed", failure={"category": "execution_error"})
        self.rejects((row, arrays), "failed clean metrics remain undefined", "clean-validation")

    def test_clean_complete_numerically_invalid_prediction_rejected(self):
        for key, value in (("normalized_prediction", np.nan), ("normalized_target", np.inf),
                            ("raw_head", np.nan), ("predicted_variance", 0.)):
            with self.subTest(key=key):
                row, arrays = clean_fixture()
                arrays[key].flat[0] = value
                self.rejects((row, arrays), "accepted clean finite positive arrays", "clean-validation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
