"""Synthetic-only independent arithmetic/coverage tests; no research imports."""
import ast
import copy
import gzip
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import pytest

PATH = Path(__file__).with_name("audit_continuation_summary_independent.py")
SPEC = importlib.util.spec_from_file_location("independent_continuation_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def boundary(position):
    # Independent scalar fixture for a [-.5,.5]^2 box.
    points = np.asarray(position).tolist()
    excursions = [max(0., *(abs(coordinate) - .5 for coordinate in point)) for point in points]
    return {"fraction_particles_outside": sum(value > 0 for value in excursions) / len(points),
        "fraction_particles_outside_by_more_than_1e-6": sum(value > 1e-6 for value in excursions) / len(points),
        "maximum_coordinate_excursion": max(excursions), "mean_particle_maximum_excursion": sum(excursions) / len(points),
        "coordinate_minimum": [min(point[axis] for point in points) for axis in range(2)],
        "coordinate_maximum": [max(point[axis] for point in points) for axis in range(2)]}


def observed():
    arrays = {"current_history": np.zeros((6, 3, 2), dtype=np.float32), "target_position": np.zeros((3, 2), dtype=np.float32),
        "acceleration_std": np.array([1., 2.]), "bounds": np.array([[-.5, .5], [-.5, .5]]),
        "previous_base_risk": np.array([1., 2., 3.]), "previous_base_prediction": np.zeros((3, 2), dtype=np.float32)}
    row = {"n_particles": 3, "status": "complete", "failure": None, "operational_seconds": 7.,
        "ground_truth_boundary": boundary(arrays["target_position"]), "current_observed_boundary": boundary(arrays["current_history"][-1]),
        "previous_score": {"status": "complete", "failure": None, "operational_seconds": .3},
        "cases": {}, "benefits": {}, "previous_risk_correlations": {},
        "previous_risk_base_residual_correlation": {"value": 1., "reason": None, "particles": 3}}
    magnitudes = ([1, 2, 3], [0, 3, 3], [2, 2, 2], [3, 2, 1], [0, 0, 0])
    for policy, y, correlation in zip(audit.OBS_POLICIES, magnitudes, (None, -.5, 1., 1., 1.)):
        prediction = np.array([[v, 2*v] for v in y], dtype=np.float32)
        se = np.array([v*v for v in y], dtype=np.float64)
        gain = np.array([1, 4, 9], dtype=np.float64) - se
        arrays["prediction__" + policy] = prediction
        arrays["position_residual__" + policy] = prediction.astype(np.float64)
        arrays["normalized_residual__" + policy] = np.array([[v, v] for v in y], dtype=np.float64)
        arrays["normalized_coordinate_se__" + policy] = se
        arrays["signed_normalized_coordinate_gain__" + policy] = gain
        row["cases"][policy] = {"status": "complete", "failure": None,
            "graph": {**{name: 2 for name in audit.COUNTS}, "directed_edges": 6},
            "graph_build_and_selection_seconds": .1, "forward_operational_seconds": .2,
            "scoring_seconds": .3 if policy == audit.OBS_POLICIES[-1] else 0.,
            "network_passes_for_standalone_policy": 2 if policy == audit.OBS_POLICIES[-1] else 1,
            "reused_native_parity_output": policy == "base", "predicted_boundary": boundary(prediction),
            "metrics": {"position_coordinate_mse": 5 * sum(v*v for v in y) / 6,
                        "normalized_coordinate_mse": sum(v*v for v in y) / 3}}
        row["benefits"][policy] = {"mean_signed_normalized_coordinate_gain": float(sum(gain) / 3),
            "harmful_particle_fraction": float(sum(gain < 0) / 3)}
        row["previous_risk_correlations"][policy] = {"value": correlation, "reason": "constant_rank_vector" if correlation is None else None, "particles": 3}
    return row, arrays


def early_observed():
    failure = {"category": "nonfinite_state", "phase": "current_history"}
    return {"n_particles": 3, "status": "failed", "failure": failure, "operational_seconds": .2,
        "cases": {policy: {"status": "failed", "failure": failure, "metrics": None} for policy in audit.OBS_POLICIES}}, {}


def autonomous(steps=995, policy="base", rejected=False):
    done = steps == 995
    mse = [float((i % 5) + 1) for i in range(steps)]
    zero = boundary([[0., 0.], [.1, -.1]])
    outside = boundary([[2., 1.], [0., 0.]])
    attempts = [{"forecast_step": step + 1, "accepted": True, "graph_build_and_selection_seconds": .1,
        "features_forward_decode_and_transfer_seconds": .2,
        **{name: 2 for name in audit.COUNTS}} for step in range(steps)]
    if rejected:
        attempts.append({"forecast_step": steps + 1, "accepted": False, "graph_build_and_selection_seconds": .4,
                         "features_forward_decode_and_transfer_seconds": .5})
    warmup = {"passes": 1, "graph_build_and_selection_seconds": .6,
              "features_forward_decode_and_transfer_seconds": .7} if policy == "laggedrisk25" else None
    return {"status": "complete" if done else "failed", "failure": None if done else {"category": "numerical", "phase": "forward"},
        "policy": policy, "completed_steps": steps, "horizon": 995, "mse_per_step": mse,
        "directed_edges_per_step": [4] * steps, "mse_at_steps": {str(h): mse[h-1] if steps >= h else None for h in audit.HORIZONS},
        "mean_rollout_mse": 3. if done else None, "mean_directed_edges": 4. if done else None,
        "mse_at_final_horizon": 5. if done else None, "predicted_boundary_per_step": [copy.deepcopy(outside) for _ in range(steps)],
        "ground_truth_boundary_per_step": [copy.deepcopy(zero) for _ in range(steps)], "attempts": attempts, "warmup": warmup,
        "forecast_network_passes": steps + int(rejected), "total_network_passes": steps + int(rejected) + int(warmup is not None),
        "total_wall_seconds": 100., "total_wall_seconds_including_native_parity": 101.,
        "native_parity_operational_seconds": 1., "native_parity_network_passes": 2,
        "graph_build_and_selection_seconds_including_warmup": .1*steps + .4*int(rejected) + .6*int(warmup is not None),
        "features_forward_decode_and_transfer_seconds_including_warmup": .2*steps + .5*int(rejected) + .7*int(warmup is not None)}


def population():
    units = [{"trajectory_id": "A", "unit_id": str(i)} for i in (1, 2, 3)] + [{"trajectory_id": "B", "unit_id": "1"}]
    lookup = {}
    for arm in audit.ARMS:
        for seed in audit.SEEDS:
            for index, unit in enumerate(units):
                value = 10 + seed + [1, 3, 5, 9][index]
                for policy, offset in zip(audit.OBS_POLICIES, (0, -1, 2, 3, 4)):
                    v = value + offset
                    if arm == "mix":
                        v += [0, 2, 4, 10][index] * (seed + 1)
                        if policy == audit.OBS_POLICIES[-1]:
                            v += [-3, 0, 3, 4*(seed-1)][index]
                    lookup[arm, seed, unit["trajectory_id"], unit["unit_id"], policy] = {"loss": v}, {"loss": None}
    return lookup, units


def summary_fixture(lookup, units, terms, paired=False):
    # Deliberately separate scalar fixture builder. Its expected numerical
    # values are asserted by hand in tests, not merely round-tripped.
    seeds = []
    for seed in range(3):
        trajectories = []
        for tid in sorted({u["trajectory_id"] for u in units}):
            rows = []
            for unit in sorted((u for u in units if u["trajectory_id"] == tid), key=lambda u: u["unit_id"]):
                values = [lookup[arm, seed, tid, unit["unit_id"], policy][0]["loss"] for arm, policy, sign in terms]
                reasons = [f"{arm}/{policy}: {lookup[arm, seed, tid, unit['unit_id'], policy][1]['loss']}"
                    for (arm, policy, sign), value in zip(terms, values) if value is None]
                rows.append({**unit, "value": None if reasons else sum(v * sign for v, (_, _, sign) in zip(values, terms)),
                             "null_reason": "; ".join(reasons) if reasons else None})
            valid = [row["value"] for row in rows if row["value"] is not None]
            trajectories.append({"trajectory_id": tid, "value": sum(valid)/len(valid) if len(valid) == len(rows) else None,
                "required_units": len(rows), "defined_units": len(valid), "units": rows})
        valid = [row["value"] for row in trajectories if row["value"] is not None]
        defined = sum(row["defined_units"] for row in trajectories)
        seeds.append({"seed": seed, "value": sum(valid)/len(valid) if len(valid) == len(trajectories) else None,
            "required_units": len(units), "defined_units": defined, "null_units": len(units)-defined, "trajectories": trajectories})
    valid = [row["value"] for row in seeds if row["value"] is not None]
    result = {"seeds": seeds, "seed_values": [row["value"] for row in seeds], "mean": sum(valid)/3 if len(valid) == 3 else None,
        "sample_sd": statistics.stdev(valid) if len(valid) == 3 else None, "required_seeds": 3, "defined_seeds": len(valid)}
    if paired:
        result["terms"] = terms
    return result, {key: value for key, value in result.items() if key != "seeds"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))
    return audit.sha(path)


def test_audit_imports_are_bounded_and_no_dynamic_inference_entrypoint():
    tree = ast.parse(PATH.read_text())
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert modules <= {"argparse", "datetime", "gzip", "hashlib", "json", "math", "pathlib", "statistics", "time", "numpy"}
    assert "torch" not in PATH.read_text()


def test_hand_derived_observed_errors_adverse_gains_correlations_and_boundaries():
    row, arrays = observed()
    values, nulls = audit.observed_metrics(row, arrays, audit.Audit())
    assert len(values["base"]) == 30
    assert values["base"]["position_coordinate_mse"] == pytest.approx(35/3)
    assert values["base"]["normalized_coordinate_mse"] == pytest.approx(14/3)
    assert values["dense"]["mean_signed_normalized_coordinate_gain"] == pytest.approx(-4/3)
    assert values["dense"]["harmful_particle_fraction"] == pytest.approx(1/3)
    assert values["dense"]["previous_risk_gain_spearman"] == pytest.approx(-.5)
    assert values["base"]["previous_risk_gain_spearman"] is None
    assert nulls["base"]["previous_risk_gain_spearman"] == "constant_rank_vector"
    assert values["base"]["predicted_boundary_maximum_coordinate_excursion"] == 5.5
    assert values["base"]["predicted_boundary_mean_particle_maximum_excursion"] == 3.5
    assert values[audit.OBS_POLICIES[-1]]["scoring_seconds"] == .3


@pytest.mark.parametrize("mutation", ["mse", "gain", "harm", "correlation", "boundary", "saved_residual", "negative_time", "boolean_pass", "reuse", "graph_count"])
def test_observed_scalar_tampering_rejected(mutation):
    row, arrays = observed()
    if mutation == "mse": row["cases"]["dense"]["metrics"]["position_coordinate_mse"] += .1
    elif mutation == "gain": row["benefits"]["dense"]["mean_signed_normalized_coordinate_gain"] += .1
    elif mutation == "harm": row["benefits"]["dense"]["harmful_particle_fraction"] = 0
    elif mutation == "correlation": row["previous_risk_correlations"]["dense"]["value"] = .5
    elif mutation == "boundary": row["cases"]["dense"]["predicted_boundary"]["maximum_coordinate_excursion"] += 1
    elif mutation == "saved_residual": arrays["normalized_residual__dense"][0, 0] += .1
    elif mutation == "negative_time": row["cases"]["dense"]["forward_operational_seconds"] = -.1
    elif mutation == "boolean_pass": row["cases"]["dense"]["network_passes_for_standalone_policy"] = True
    elif mutation == "reuse": row["cases"]["base"]["reused_native_parity_output"] = 1
    else: row["cases"]["dense"]["graph"]["directed_edges"] = -1
    with pytest.raises(ValueError):
        audit.observed_metrics(row, arrays, audit.Audit())


def test_isolated_previous_score_failure_retains_other_policies_and_physical_references():
    row, arrays = observed()
    risk = audit.OBS_POLICIES[-1]
    failure = {"category": "nonfinite_risk", "phase": "previous_forward"}
    row["previous_score"].update(status="failed", failure=failure)
    case_failure = {"category": "scoring_pass_failed", "cause": failure}
    row["cases"][risk] = {"status": "failed", "failure": case_failure, "metrics": None, "network_passes_for_standalone_policy": 1}
    row.update(status="failed", failure={"category": "case_failures", "cases": [{"case": risk, "failure": case_failure}]})
    row["benefits"][risk] = {"mean_signed_normalized_coordinate_gain": None, "reason": "required base or action output failed"}
    row["previous_risk_correlations"] = {}
    row["previous_risk_base_residual_correlation"] = {"value": None, "reason": "required scoring or base output failed"}
    del arrays["prediction__" + risk]
    values, reasons = audit.observed_metrics(row, arrays, audit.Audit())
    assert values[risk]["failure_fraction"] == 1 and values[risk]["position_coordinate_mse"] is None
    assert values[risk]["network_passes_for_standalone_policy"] == 1
    assert values[risk]["ground_truth_boundary_maximum_coordinate_excursion"] == 0
    assert values["dense"]["position_coordinate_mse"] == 15
    assert reasons["dense"]["previous_risk_gain_spearman"] == audit.UNAVAILABLE
    assert reasons["dense"]["previous_risk_base_residual_spearman"] == "required scoring or base output failed"


def test_failed_action_retains_finite_rejected_boundary_but_not_error_or_gain():
    row, arrays = observed()
    failure = {"category": "invalid_risk", "phase": "forward"}
    row["cases"]["dense"].update(status="failed", failure=failure, metrics=None)
    row.update(status="failed", failure={"category": "case_failures", "cases": [{"case": "dense", "failure": failure}]})
    row["benefits"]["dense"] = {"mean_signed_normalized_coordinate_gain": None, "reason": "required base or action output failed"}
    del row["previous_risk_correlations"]["dense"]
    values, _ = audit.observed_metrics(row, arrays, audit.Audit())
    assert values["dense"]["failure_fraction"] == 1
    assert values["dense"]["position_coordinate_mse"] is None
    assert values["dense"]["mean_signed_normalized_coordinate_gain"] is None
    assert values["dense"]["predicted_boundary_maximum_coordinate_excursion"] == 5.5
    assert values["random25"]["position_coordinate_mse"] == 10


def test_early_failure_has_only_failure_fraction():
    row, arrays = early_observed()
    values, reasons = audit.observed_metrics(row, arrays, audit.Audit())
    assert all([value for name, value in metrics.items() if name != "failure_fraction"] == [None]*29 for metrics in values.values())
    arrays["target_position"] = np.zeros((3, 2))
    with pytest.raises(ValueError): audit.observed_metrics(row, arrays, audit.Audit())


@pytest.mark.parametrize("x,y,expected,reason", [([1,1,2,3],[1,2,3,4], 3/math.sqrt(10), None),
    ([1,2,3],[3,2,1], -1., None), ([1,1],[2,3], None, "constant_rank_vector"), ([1],[2], None, "fewer_than_two_particles")])
def test_own_tied_rank_correlation(x, y, expected, reason):
    result = audit.spearman(x, y)
    assert result["reason"] == reason
    assert result["value"] is None if expected is None else result["value"] == pytest.approx(expected)


def test_boundary_threshold_includes_strict_tolerance_and_anisotropic_box():
    values = audit.boundaries(np.array([[.5, 1.], [.5000005, 1.], [-.500002, -1.]]), [[-.5, .5], [-1, 1]])
    assert values["fraction_particles_outside"] == pytest.approx(2/3)
    assert values["fraction_particles_outside_by_more_than_1e-6"] == pytest.approx(1/3)
    assert values["maximum_coordinate_excursion"] == pytest.approx(2e-6)


def test_autonomous_all_horizons_counts_boundary_and_time_recomputed():
    values, reasons = audit.autonomous_metrics(autonomous(policy="laggedrisk25"), audit.Audit())
    assert len(values) == 31
    assert values["mean_rollout_mse"] == 3
    assert [values[f"mse_at_{h}"] for h in audit.HORIZONS] == [1,5,5,5,5,5]
    assert values["mean_total_network_passes"] == 996
    assert values["mean_graph_operational_seconds"] == pytest.approx(100.1)
    assert values["mean_forward_component_seconds"] == pytest.approx(199.7)
    assert values["predicted_boundary_mean_fraction_outside_gt_1e_minus6"] == .5
    assert values["predicted_boundary_trajectory_maximum_excursion"] == 1.5
    assert values["ground_truth_boundary_trajectory_maximum_excursion"] == 0
    assert values["cap_active_step_fraction"] == 1
    assert set(reasons.values()) == {None}


def test_autonomous_failure_keeps_early_horizons_and_rejected_attempt_time():
    values, reasons = audit.autonomous_metrics(autonomous(steps=201, policy="laggedrisk25", rejected=True), audit.Audit())
    assert values["failure_fraction"] == 1 and values["completed_steps"] == 201
    assert values["mse_at_200"] == 5 and values["mse_at_500"] is None
    assert values["mean_rollout_mse"] is values["mean_directed_edges"] is None
    assert values["predicted_boundary_trajectory_maximum_excursion"] is None
    assert values["mean_total_network_passes"] == 203
    assert values["mean_graph_operational_seconds"] == pytest.approx(21.1)
    assert values["mean_forward_component_seconds"] == pytest.approx(41.4)
    assert values["mean_rollout_wall_seconds"] == 100
    assert reasons["mse_at_500"] == audit.AUTO_NULL


@pytest.mark.parametrize("mutation", ["missing_horizon", "mean", "early_horizon", "fraction", "outside_order", "max_mean", "negative_time", "time_total", "boolean_accepted", "wrong_attempt", "missing_pass", "survivor", "empty_failure", "float_steps"])
def test_autonomous_adverse_or_malformed_scalar_not_erased(mutation):
    row = autonomous(201, rejected=True)
    if mutation == "missing_horizon": del row["mse_at_steps"]["10"]
    elif mutation == "mean": row["mean_rollout_mse"] = 2
    elif mutation == "early_horizon": row["mse_at_steps"]["200"] = 1
    elif mutation == "fraction": row["predicted_boundary_per_step"][0]["fraction_particles_outside_by_more_than_1e-6"] = 0
    elif mutation == "outside_order": row["predicted_boundary_per_step"][0]["fraction_particles_outside_by_more_than_1e-6"] = .8
    elif mutation == "max_mean": row["predicted_boundary_per_step"][0]["mean_particle_maximum_excursion"] = 4
    elif mutation == "negative_time": row["attempts"][-1]["features_forward_decode_and_transfer_seconds"] = -.1
    elif mutation == "time_total": row["total_wall_seconds_including_native_parity"] = 99
    elif mutation == "boolean_accepted": row["attempts"][0]["accepted"] = 1
    elif mutation == "wrong_attempt": row["attempts"][-1]["forecast_step"] = 300
    elif mutation == "missing_pass": row["forecast_network_passes"] -= 1
    elif mutation == "survivor": row["status"] = "complete"; row["failure"] = None
    elif mutation == "empty_failure": row["failure"] = {}
    else: row["completed_steps"] = 201.
    with pytest.raises(ValueError): audit.autonomous_metrics(row, audit.Audit())


def test_hand_derived_equal_trajectory_weighting_and_interaction():
    lookup, units = population()
    definitions = audit.contrasts(audit.OBS_POLICIES)
    assert len(definitions) == 12
    terms = definitions["mix_minus_base__base"]
    detail, compact = summary_fixture(lookup, units, terms, True)
    assert compact["seed_values"] == [6, 12, 18]
    assert compact["mean"] == 12 and compact["sample_sd"] == 6
    audit.check_hierarchy(detail, compact, lookup, units, terms, "loss", audit.Audit())
    terms = definitions["risk_minus_random_interaction"]
    detail, compact = summary_fixture(lookup, units, terms, True)
    assert compact["seed_values"] == [-2,0,2] and compact["sample_sd"] == 2
    assert detail["seeds"][0]["trajectories"][0]["value"] == 0
    audit.check_hierarchy(detail, compact, lookup, units, terms, "loss", audit.Audit())


def test_null_propagates_to_trajectory_seed_mean_sd_with_exact_reason():
    lookup, units = population()
    risk = audit.OBS_POLICIES[-1]
    lookup["base", 1, "A", "2", risk] = {"loss": None}, {"loss": "scientific failure"}
    terms = audit.contrasts(audit.OBS_POLICIES)["risk_minus_random_interaction"]
    detail, compact = summary_fixture(lookup, units, terms, True)
    assert compact["seed_values"] == [-2,None,2]
    assert compact["mean"] is compact["sample_sd"] is None
    assert detail["seeds"][1]["null_units"] == 1
    assert detail["seeds"][1]["trajectories"][0]["units"][1]["null_reason"] == "base/previous-observed-base-risk25: scientific failure"
    audit.check_hierarchy(detail, compact, lookup, units, terms, "loss", audit.Audit())


@pytest.mark.parametrize("mutation", ["weighting", "swap_seed", "duplicate_unit", "unit_value", "compact", "null_reason", "survivor", "boolean_seed", "boolean_value", "nan_value", "missing_raw", "terms", "unknown_trajectory"])
def test_hierarchy_tampering_rejected(mutation):
    lookup, units = population()
    terms = audit.contrasts(audit.OBS_POLICIES)["mix_minus_base__base"]
    detail, compact = summary_fixture(lookup, units, terms, True)
    detail, compact = copy.deepcopy(detail), copy.deepcopy(compact)
    if mutation == "weighting": detail["seeds"][0]["value"] = 4
    elif mutation == "swap_seed": detail["seeds"][0]["seed"] = 1
    elif mutation == "duplicate_unit": detail["seeds"][0]["trajectories"][0]["units"][1] = detail["seeds"][0]["trajectories"][0]["units"][0]
    elif mutation == "unit_value": detail["seeds"][0]["trajectories"][0]["units"][0]["value"] = 1
    elif mutation == "compact": compact["mean"] += 1
    elif mutation == "null_reason": detail["seeds"][0]["trajectories"][0]["units"][0]["null_reason"] = "invented failure"
    elif mutation == "survivor": lookup["base",0,"A","1","base"] = {"loss": None}, {"loss": "failure"}
    elif mutation == "boolean_seed": detail["seeds"][0]["seed"] = False
    elif mutation == "boolean_value": lookup["base",0,"A","1","base"] = {"loss": True}, {"loss": None}
    elif mutation == "nan_value": lookup["base",0,"A","1","base"] = {"loss": float("nan")}, {"loss": None}
    elif mutation == "missing_raw": del lookup["base",0,"A","1","base"]
    elif mutation == "terms": detail["terms"][0] = ("mix", "base", -1)
    else: detail["seeds"][0]["trajectories"][0]["trajectory_id"] = "C"
    with pytest.raises(ValueError): audit.check_hierarchy(detail, compact, lookup, units, terms, "loss", audit.Audit())


def test_lossless_population_all_ten_absolute_and_twelve_paired_entries(tmp_path):
    lookup, units = population()
    detail, index = {"absolute": {}, "paired": {}}, {"absolute": {}, "paired": {}}
    for arm in audit.ARMS:
        detail["absolute"][arm], index["absolute"][arm] = {}, {}
        for policy in audit.OBS_POLICIES:
            d, c = summary_fixture(lookup, units, [(arm,policy,1)])
            detail["absolute"][arm][policy], index["absolute"][arm][policy] = d, c
    for name, terms in audit.contrasts(audit.OBS_POLICIES).items():
        detail["paired"][name], index["paired"][name] = summary_fixture(lookup, units, terms, True)
    data = json.dumps(detail).encode()
    path = tmp_path / "observed_valid__loss.json.gz"
    path.write_bytes(gzip.compress(data))
    index.update(path=path.name, sha256=audit.sha(path), uncompressed_json_sha256=audit.hashlib.sha256(data).hexdigest())
    population_index = {"required_units_per_endpoint": 4, "policies": list(audit.OBS_POLICIES), "metrics": {"loss": index}}
    check = audit.Audit()
    audit.check_population(tmp_path, "observed_valid", population_index, lookup, units, audit.OBS_POLICIES, ("loss",), check)
    assert check.checks > 3000
    index["paired"]["risk_minus_random_interaction"]["mean"] = 9
    with pytest.raises(ValueError): audit.check_population(tmp_path, "observed_valid", population_index, lookup, units, audit.OBS_POLICIES, ("loss",), audit.Audit())


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'])
def test_duplicate_nonfinite_json_rejected(text):
    with pytest.raises(ValueError): audit.decode(text)


def test_input_hash_and_local_path_checks(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"x":1}')
    check = audit.Audit(); check.read(path)
    path.write_text('{"x":2}')
    with pytest.raises(ValueError, match="hash|changed"): check.read(path)
    for name in ("../input.json", str(path), ".", "..", ""):
        with pytest.raises(ValueError): audit.local_path(tmp_path, name)
    (tmp_path / "link.json").symlink_to(path)
    with pytest.raises(ValueError): audit.local_path(tmp_path, "link.json")


def protocol(arm, seed, mode, expected):
    return {"arm": arm, "seed": seed, "original_seed": seed, "mode": mode, "scope": audit.EVAL_SCOPE,
        "objective": "faithful", "completed_total_updates": 110000, "completed_additional_updates": 10000,
        "policies": list(audit.OBS_POLICIES if mode == "observed" else audit.AUTO_POLICIES), "source_indices": list(audit.INDICES),
        "checkpoint_sha256": str(seed + (1 if arm == "base" else 4))*64,
        "parent_checkpoint_sha256": str(seed+7)*64, "continuation_config_sha256": "c"*64,
        "expected_frames": expected if mode == "observed" else None, "trajectory_ids": ["test"+str(i) for i in audit.INDICES]}


def scheduled():
    return [{"split": "valid", "source_index": i % 30, "target_frame": 10 + i//30, "trajectory_id": "valid"+str(i%30)} for i in range(128)] + [
        {"split": "test", "source_index": i, "target_frame": 10+j, "trajectory_id": "test"+str(i)} for i in audit.INDICES for j in range(11)]


def job_fixture(directory, arm="base", seed=0, mode="observed", actual_records=False):
    p = protocol(arm, seed, mode, scheduled())
    ph = write_json(directory / "protocol.json", p)
    records = []
    count = 425 if mode == "observed" else 135
    if actual_records and mode == "observed":
        for unit in scheduled():
            row, arrays = early_observed()
            stem = f"{unit['split']}_{unit['source_index']:06d}_{unit['target_frame']:04d}"
            np.savez_compressed(directory / (stem + ".npz"), **arrays)
            row.update(**unit, arm=arm, original_seed=seed, checkpoint_sha256=p["checkpoint_sha256"], parent_checkpoint_sha256=p["parent_checkpoint_sha256"],
                protocol_sha256=ph, array_file=stem+".npz", array_sha256=audit.sha(directory / (stem+".npz")))
            record_sha = write_json(directory / (stem+".json"), row)
            records.append({key: row[key] for key in ("split", "source_index", "target_frame", "trajectory_id", "status", "failure", "array_file", "array_sha256")} | {
                "record_file": stem+".json", "record_sha256": record_sha})
    elif actual_records:
        for i in audit.INDICES:
            for policy in audit.AUTO_POLICIES:
                row = autonomous(0, policy)
                stem = f"trajectory_{i:06d}_{policy}"
                (directory / (stem+".npz")).write_bytes(b"synthetic trace hash only")
                row.update(source_index=i, trajectory_id="test"+str(i), rng_seed=93000+1000*seed+i, protocol_sha256=ph,
                    trace_file=stem+".npz", trace_sha256=audit.sha(directory / (stem+".npz")))
                rh = write_json(directory / (stem+".json"), row)
                records.append({key: row[key] for key in ("source_index", "trajectory_id", "policy", "status", "failure", "completed_steps", "trace_file", "trace_sha256")} | {
                    "record_file": stem+".json", "record_sha256": rh})
    else:
        records = [{"status": "failed"} for _ in range(count)]
    result = {"state": "complete", "scope": audit.EVAL_SCOPE, "protocol_sha256": ph, "arm": arm, "seed": seed,
        "objective": "faithful", "checkpoint_sha256": p["checkpoint_sha256"], "parent_checkpoint_sha256": p["parent_checkpoint_sha256"],
        "required_frames": 425, "complete_frames": 0, "failed_frames": 425, "records": records}
    if mode == "autonomous":
        result["invocation_wall_seconds"] = 10.
        write_json(directory / "lineage_identity.json", {key: p[key] for key in audit.IDENTITY_FIELDS})
    rh = write_json(directory / "result.json", result)
    write_json(directory / "status.json", {"state": "complete", "arm": arm, "seed": seed, "committed_frames": 425, "result_sha256": rh})
    return directory, p, result


def test_all_425_committed_observed_frames_scalar_loader_and_no_survivor_drop(tmp_path):
    job = job_fixture(tmp_path / "job", actual_records=True)
    check = audit.Audit()
    units = audit.observed_units(job[1], check)
    rows = audit.load_observed(job, units, check)
    assert len(rows["observed_valid"]) == 640 and len(rows["observed_test"]) == 1485
    assert all(value[0]["failure_fraction"] == 1 for pop in rows.values() for value in pop.values())
    assert len(check.pins) == 850
    job[2]["records"][0]["source_index"] = True
    with pytest.raises(ValueError): audit.load_observed(job, units, audit.Audit())


def test_all_135_committed_autonomous_outcomes_preserve_failures(tmp_path):
    job = job_fixture(tmp_path / "job", mode="autonomous", actual_records=True)
    units = [{"source_index": i, "trajectory_id": "test"+str(i), "unit_id": str(i)} for i in audit.INDICES]
    check = audit.Audit(); rows = audit.load_autonomous(job, units, check)
    assert len(rows) == 135 and len(check.pins) == 270
    assert all(value[0]["failure_fraction"] == 1 for value in rows.values())
    assert all(value[0]["mean_rollout_mse"] is None for value in rows.values())
    assert all(value[0]["mean_rollout_wall_seconds"] == 100 for value in rows.values())
    job[2]["records"][1] = copy.deepcopy(job[2]["records"][0])
    with pytest.raises(ValueError): audit.load_autonomous(job, units, audit.Audit())


@pytest.mark.parametrize("mode", ["observed", "autonomous"])
def test_six_explicit_job_identities_not_directory_names(tmp_path, mode):
    for index, (arm, seed) in enumerate(sorted(audit.COHORT)):
        job_fixture(tmp_path / ("arbitrary" + str(index)), arm, seed, mode)
    assert set(audit.jobs(tmp_path, mode, audit.Audit())) == audit.COHORT


@pytest.mark.parametrize("mutation", ["partial", "lock", "duplicate_arm", "boolean_seed", "wrong_updates", "wrong_hash", "lineage", "missing_job", "temporary"])
def test_job_admission_does_not_accept_partial_or_swapped_cohort(tmp_path, mutation):
    for index, (arm, seed) in enumerate(sorted(audit.COHORT)):
        job_fixture(tmp_path / ("job" + str(index)), arm, seed, "autonomous")
    directory = tmp_path / "job0"
    if mutation == "lock": (directory / "run.lock").write_text("owned")
    elif mutation == "temporary": (directory / "partial.tmp").write_text("partial")
    elif mutation == "missing_job": (directory / "status.json").unlink()
    else:
        name = "status.json" if mutation in ("partial", "wrong_hash") else "lineage_identity.json" if mutation == "lineage" else "protocol.json"
        data = json.loads((directory / name).read_text())
        if mutation == "partial": data["state"] = "running"
        elif mutation == "wrong_hash": data["result_sha256"] = "0"*64
        elif mutation == "duplicate_arm": data["arm"] = "mix"
        elif mutation == "boolean_seed": data["seed"] = False
        elif mutation == "wrong_updates": data["completed_total_updates"] = 100000
        elif mutation == "lineage": data["parent_checkpoint_sha256"] = "0"*64
        write_json(directory / name, data)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.jobs(tmp_path, "autonomous", audit.Audit())


def test_output_isolation_precedes_any_directory_creation(tmp_path):
    summary = tmp_path / "summary" / "result.json"
    endpoint = tmp_path / "endpoint" / "checkpoint.pt"
    write_json(summary, {"cohort": [{"checkpoint": str(endpoint)}], "input_files_sha256": {}})
    for output in (summary.parent / "audit", endpoint.parent / "audit", tmp_path, tmp_path / "observed" / "audit"):
        with pytest.raises(ValueError): audit.isolated_output(output, summary, tmp_path/"observed", tmp_path/"autonomous")
    assert not endpoint.parent.exists() and not (summary.parent/"audit").exists()
    assert audit.isolated_output(tmp_path/"new-audit", summary, tmp_path/"observed", tmp_path/"autonomous") == tmp_path/"new-audit"


def test_failed_audit_attempt_is_preserved(tmp_path, monkeypatch):
    summary = tmp_path / "summary" / "result.json"
    write_json(summary, {"cohort": [{"checkpoint": str(tmp_path / "endpoint" / "checkpoint.pt")}], "input_files_sha256": {}})
    out = tmp_path / "audit"
    def fail(*args): raise ValueError("synthetic incomplete cohort")
    monkeypatch.setattr(audit, "execute", fail)
    monkeypatch.setattr(sys, "argv", ["audit", "--summary", str(summary), "--observed-root", str(tmp_path/"observed"),
        "--autonomous-root", str(tmp_path/"autonomous"), "--output-dir", str(out)])
    with pytest.raises(ValueError, match="incomplete cohort"): audit.main()
    record = json.loads((out/"audit.json").read_text())
    assert record["passed"] is False and record["error_type"] == "ValueError"
    assert record["error"] == "synthetic incomplete cohort" and record["audit_source_sha256"] == audit.sha(PATH)
    with pytest.raises(ValueError, match="Fresh"): audit.main()


def test_execute_complete_synthetic_twelve_job_coverage_and_summary_bindings(tmp_path, monkeypatch):
    """Real scalar loaders/identity checks, with detail arithmetic tested above.

    Only the publication-detail reader is replaced here to bound redundant
    synthetic gzip generation. No real experiment path or inference is used.
    """
    roots = {mode: tmp_path / mode for mode in ("observed", "autonomous")}
    metadata, cohort = [], []
    for mode, root in roots.items():
        for arm, seed in sorted(audit.COHORT):
            directory, protocol, result = job_fixture(root/f"{arm}{seed}", arm, seed, mode, actual_records=True)
            metadata.append({"mode": mode, "arm": arm, "seed": seed, "directory": str(directory.resolve()),
                "protocol_sha256": result["protocol_sha256"], "records": len(result["records"]),
                "failed_records": len(result["records"]), "result_sha256": audit.sha(directory/"result.json"),
                "invocation_wall_seconds": result.get("invocation_wall_seconds")})
            if mode == "observed":
                cohort.append({"arm": arm, "seed": seed, "sha256": protocol["checkpoint_sha256"],
                               "checkpoint": str(tmp_path/"endpoints"/f"{arm}{seed}"/"checkpoint.pt")})
    fake_source = tmp_path/"synthetic_sources"/"summarize_continuation_evaluation.py"
    fake_source.parent.mkdir()
    fake_source.write_text("# Synthetic identity only; never imported.\n")
    monkeypatch.setattr(audit, "SUMMARY_SHA256", audit.sha(fake_source))
    pins = {str(path.resolve()): audit.sha(path) for root in roots.values() for path in root.rglob("*") if path.is_file()}
    pins[str(fake_source.resolve())] = audit.sha(fake_source)
    directory = tmp_path/"summary"
    directory.mkdir()
    archive = directory/"audited_records.json.gz"
    archive.write_bytes(gzip.compress(b'{"synthetic":true}'))
    report = {"state": "complete", "scope": audit.SCOPE, "source_and_input_reverified_after_analysis": True,
        "completed_total_updates": 110000, "completed_additional_updates": 10000,
        "coverage": {"endpoints": 6, "jobs": 12, "observed_frames": 2550, "observed_policy_slots": 12750, "autonomous_outcomes": 810},
        "cohort": cohort, "jobs": metadata, "input_files_sha256": pins,
        "populations": {name: {} for name in ("observed_valid", "observed_test", "autonomous_test")},
        "audited_records": {"path": archive.name, "sha256": audit.sha(archive)}}
    summary_path = directory/"result.json"
    def commit():
        digest = write_json(summary_path, report)
        write_json(directory/"status.json", {"state": "complete", "result_sha256": digest})
    commit()
    seen = []
    def check_publication(directory, name, index, lookup, units, policies, metrics, check):
        seen.append(name)
        assert len(lookup) == {"observed_valid": 3840, "observed_test": 8910, "autonomous_test": 810}[name]
        assert len(units) == {"observed_valid": 128, "observed_test": 297, "autonomous_test": 27}[name]
        assert len(metrics) == (31 if name == "autonomous_test" else 30)
        assert all(value[0]["failure_fraction"] == 1 for value in lookup.values())
    monkeypatch.setattr(audit, "check_population", check_publication)
    check = audit.Audit()
    result = audit.execute(summary_path, roots["observed"], roots["autonomous"], check)
    assert result["observed_frames"] == 2550 and result["autonomous_outcomes"] == 810
    assert result["paired_contrasts_per_metric"] == 12 and set(seen) == set(report["populations"])
    assert len(check.pins) == 6766
    report["cohort"][0]["sha256"] = "f"*64
    commit()
    with pytest.raises(ValueError, match="value/null differs"):
        audit.execute(summary_path, roots["observed"], roots["autonomous"], audit.Audit())


def test_hierarchy_refuses_float_identity_and_float_count():
    lookup, units = population()
    for unit in units:
        unit["source_index"] = 1
    terms = [("base", "base", 1)]
    detail, compact = summary_fixture(lookup, units, terms)
    detail["seeds"][0]["trajectories"][0]["units"][0]["source_index"] = 1.
    with pytest.raises(ValueError, match="Integer"):
        audit.check_hierarchy(detail, compact, lookup, units, terms, "loss", audit.Audit())
    detail, compact = summary_fixture(lookup, units, terms)
    detail["seeds"][0]["required_units"] = 4.
    with pytest.raises(ValueError, match="Integer"):
        audit.check_hierarchy(detail, compact, lookup, units, terms, "loss", audit.Audit())


@pytest.mark.parametrize("kind", ["original100k", "converted_data", "source_code", "configuration"])
def test_pinned_input_directories_are_protected_before_any_creation(tmp_path, kind, monkeypatch):
    summary = tmp_path/"summary"/"result.json"
    immutable = tmp_path/"immutable"/kind
    source = immutable/{"original100k": "checkpoint.pt", "converted_data": "positions.npy",
                        "source_code": "trainer.py", "configuration": "config.json"}[kind]
    write_json(summary, {"cohort": [{"checkpoint": str(tmp_path/"endpoints"/"checkpoint.pt")}],
                        "input_files_sha256": {str(source): "a"*64}})
    out = immutable/"new-audit"
    monkeypatch.setattr(sys, "argv", ["audit", "--summary", str(summary), "--observed-root", str(tmp_path/"observed"),
        "--autonomous-root", str(tmp_path/"autonomous"), "--output-dir", str(out)])
    with pytest.raises(ValueError, match="overlaps immutable input tree"):
        audit.main()
    assert not immutable.exists() and not out.exists()
    # A sibling work analysis root remains valid, with metadata reads only.
    valid = tmp_path/"work"/"independent-audit"
    assert audit.isolated_output(valid, summary, tmp_path/"observed", tmp_path/"autonomous") == valid
