"""Independent saved-scalar and paired-hierarchy audit of faithful110k results.

Only stdlib and NumPy; no research imports, checkpoint loads or inference.
The frozen strict summarizer supplies complementary source/data/model admission,
scientific guard reproduction, and graph/trace reconstruction. This audit reads
every raw observed NPZ/JSON and autonomous JSON, hashes (but does not reconstruct)
autonomous traces, and independently checks every published scalar hierarchy.
Never run on a partial cohort. A new isolated output directory preserves failures.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

ARMS = ("base", "mix")
SEEDS = (0, 1, 2)
COHORT = {(arm, seed) for arm in ARMS for seed in SEEDS}
OBS_POLICIES = ("base", "dense", "random25", "speed25", "previous-observed-base-risk25")
AUTO_POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
INDICES = tuple(range(3, 30))
HORIZONS = (1, 10, 50, 200, 500, 995)
BOUNDARY_FIELDS = ("fraction_particles_outside", "fraction_particles_outside_by_more_than_1e-6",
                   "maximum_coordinate_excursion", "mean_particle_maximum_excursion")
COUNTS = ("native_base_directed_edges", "native_base_self_edges", "native_base_receivers_above_cap_before_capping",
          "native_base_edges_removed_by_cap", "native_base_asymmetric_directed_edges", "retained_optional_pairs")
OBS_METRICS = ("position_coordinate_mse", "normalized_coordinate_mse", "failure_fraction",
    "mean_signed_normalized_coordinate_gain", "harmful_particle_fraction", "previous_risk_gain_spearman",
    "previous_risk_base_residual_spearman", "graph_build_and_selection_seconds", "forward_operational_seconds",
    "scoring_seconds", "network_passes_for_standalone_policy", "directed_edges", "retained_optional_pairs") + COUNTS[:-1] + tuple(
    f"{source}_boundary_{field}" for source in ("predicted", "ground_truth", "current_observed") for field in BOUNDARY_FIELDS)
AUTO_BOUNDARIES = tuple(f"{source}_boundary_{metric}" for source in ("predicted", "ground_truth")
    for metric in ("mean_fraction_outside_gt_1e_minus6", "mean_step_maximum_excursion", "trajectory_maximum_excursion"))
AUTO_METRICS = ("mean_rollout_mse", "mse_at_200", "mse_at_995", "failure_fraction", "mean_directed_edges",
    "mean_rollout_wall_seconds", "mean_total_network_passes") + AUTO_BOUNDARIES + tuple("mean_" + name for name in COUNTS) + (
    "cap_active_step_fraction", "mean_wall_seconds_including_native_parity", "mean_native_parity_seconds",
    "mean_native_parity_passes", "mean_graph_operational_seconds", "mean_forward_component_seconds") + tuple(
    f"mse_at_{step}" for step in HORIZONS if step not in (200, 995)) + ("completed_steps", "forecast_network_passes")
IDENTITY_FIELDS = ("arm", "seed", "original_seed", "checkpoint_sha256", "parent_checkpoint_sha256",
                   "continuation_config_sha256", "completed_total_updates", "completed_additional_updates")
SUMMARY_SHA256 = "f15bf6796d11b9a95713bd92603307041af80d7b3f2a2624e537862e723887dd"
SCOPE = "post-inspection exploratory paired faithful110k continuation analysis"
EVAL_SCOPE = "post-inspection exploratory faithful110k graph-support endpoint evaluation"
UNAVAILABLE = "required scientific output unavailable"
AUTO_NULL = "required full horizon or fixed horizon not completed"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def nonnegative(value):
    require(number(value) and value >= 0, "Nonnegative finite nonboolean scalar required")
    return value


def integer(value, low=0):
    require(type(value) is int and value >= low, "Integer count/identity required")
    return value


def strict_mean(values):
    require(values and all(value is None or number(value) for value in values), "Finite fixed population required")
    return math.fsum(values) / len(values) if all(value is not None for value in values) else None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


class Audit:
    def __init__(self):
        self.checks, self.pins = 0, {}

    def equal(self, actual, expected, context="value"):
        self.checks += 1
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), context + ": dictionary keys differ")
            for key, value in expected.items():
                self.equal(actual[key], value, context + "/" + str(key))
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), context + ": sequence length differs")
            for index, (a, b) in enumerate(zip(actual, expected)):
                self.equal(a, b, context + "/" + str(index))
        elif type(expected) is bool:
            require(type(actual) is bool and actual == expected, context + ": boolean differs")
        elif type(expected) in (int, float):
            require(number(actual) and number(expected) and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-13), context + ": numeric differs")
        else:
            require(actual == expected, context + ": value/null differs")

    def array(self, actual, expected, context):
        self.checks += 1
        require(actual.shape == expected.shape and np.allclose(actual, expected, rtol=2e-12, atol=1e-14, equal_nan=False), context + ": array differs")

    def pin(self, path, expected=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), "Regular nonsymlink input required: " + str(path))
        resolved = str(path.resolve())
        digest = sha(path)
        if expected is not None:
            self.equal(digest, expected, resolved + " hash")
        if resolved in self.pins:
            self.equal(digest, self.pins[resolved], "Input changed during audit")
        self.pins[resolved] = digest
        return digest

    def read(self, path, expected=None):
        self.pin(path, expected)
        return decode(Path(path).read_bytes())

    def compressed(self, directory, index):
        path = local_path(directory, index["path"])
        self.pin(path, index["sha256"])
        data = gzip.decompress(path.read_bytes())
        self.equal(hashlib.sha256(data).hexdigest(), index["uncompressed_json_sha256"], "Decompressed JSON hash")
        return decode(data)


def local_path(directory, name):
    require(isinstance(name, str) and Path(name).name == name and name not in ("", ".", ".."), "Local artifact filename required")
    path = Path(directory) / name
    require(not path.is_symlink() and path.resolve().parent == Path(directory).resolve(), "Artifact escaped job directory")
    return path


def outcome(row):
    require(row["status"] in ("complete", "failed"), "Scientific status required")
    complete = row["status"] == "complete"
    require((complete and row["failure"] is None) or (not complete and isinstance(row["failure"], dict)
            and isinstance(row["failure"].get("category"), str) and row["failure"]["category"]), "Failure reason/status mismatch")
    return complete


def boundaries(position, bounds):
    position, bounds = np.asarray(position, dtype=np.float64), np.asarray(bounds, dtype=np.float64)
    require(position.ndim == 2 and position.shape[0] > 0 and position.shape[1] == 2 and bounds.shape == (2, 2)
            and np.isfinite(position).all() and np.isfinite(bounds).all() and np.all(bounds[:, 0] < bounds[:, 1]), "Finite 2D physical state required")
    distance = np.maximum(position - bounds[:, 1], bounds[:, 0] - position).clip(min=0)
    particle = distance.max(axis=1)
    return {"fraction_particles_outside": float(np.count_nonzero(particle > 0) / len(particle)),
        "fraction_particles_outside_by_more_than_1e-6": float(np.count_nonzero(particle > 1e-6) / len(particle)),
        "maximum_coordinate_excursion": float(particle.max()), "mean_particle_maximum_excursion": float(particle.mean()),
        "coordinate_minimum": position.min(axis=0).tolist(), "coordinate_maximum": position.max(axis=0).tolist()}


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), "Matching finite correlation vectors required")
    if len(x) < 2:
        return {"value": None, "reason": "fewer_than_two_particles", "particles": len(x)}
    def ranks(values):
        # Average zero-based ranks; centered correlations are invariant to +1.
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = np.empty(len(values), dtype=np.float64)
        first = 0
        while first < len(values):
            last = first + 1
            while last < len(values) and values[order[last]] == values[order[first]]:
                last += 1
            result[order[first:last]] = (first + last - 1) / 2
            first = last
        return result - result.mean()
    a, b = ranks(x), ranks(y)
    aa, bb = float(a @ a), float(b @ b)
    if aa == 0 or bb == 0:
        return {"value": None, "reason": "constant_rank_vector", "particles": len(x)}
    return {"value": min(1., max(-1., float(a @ b) / math.sqrt(aa * bb))), "reason": None, "particles": len(x)}


def observed_metrics(row, arrays, audit):
    """Saved predictions/targets only; graph scalar counts are not reconstructed."""
    n = integer(row["n_particles"], 1)
    complete = outcome(row)
    nonnegative(row["operational_seconds"])
    audit.equal(set(row["cases"]), set(OBS_POLICIES), "Observed policies")
    metrics = {policy: {metric: None for metric in OBS_METRICS} for policy in OBS_POLICIES}
    reasons = {policy: {metric: UNAVAILABLE for metric in OBS_METRICS} for policy in OBS_POLICIES}
    def put(policy, name, value, reason=None):
        require(value is None or number(value), "Finite observed metric required")
        require(value is not None or isinstance(reason, str) and reason, "Null reason required")
        metrics[policy][name], reasons[policy][name] = value, reason if value is None else None
    for policy, case in row["cases"].items():
        put(policy, "failure_fraction", float(not outcome(case)))
    failure = row["failure"] or {}
    if failure.get("phase") in ("current_history", "current_graph") or failure.get("category") == "native_parity_failure":
        require(not complete and all(case["status"] == "failed" and case["metrics"] is None for case in row["cases"].values()), "Early failure has successful output")
        require("target_position" not in arrays and not any(key.startswith("prediction__") for key in arrays), "Early failure has downstream outputs")
        return metrics, reasons
    target, scale = arrays["target_position"].astype(np.float64), arrays["acceleration_std"].astype(np.float64)
    require(target.shape == (n, 2) and scale.shape == (2,) and np.isfinite(target).all() and np.isfinite(scale).all() and np.all(scale > 0), "Target/normalization differs")
    for source, position in (("ground_truth", target), ("current_observed", arrays["current_history"][-1])):
        values = boundaries(position, arrays["bounds"])
        audit.equal(row[source + "_boundary"], values, source + " boundaries")
        for policy in OBS_POLICIES:
            for field in BOUNDARY_FIELDS:
                put(policy, source + "_boundary_" + field, values[field])
    prior_complete = outcome(row["previous_score"])
    good = {}
    for policy in OBS_POLICIES:
        case = row["cases"][policy]
        if policy == OBS_POLICIES[-1] and not prior_complete:
            audit.equal(case["failure"], {"category": "scoring_pass_failed", "cause": row["previous_score"]["failure"]}, "Risk score failure")
            audit.equal(case["metrics"], None)
            passes = int("previous_base_prediction" in arrays)
            audit.equal(case["network_passes_for_standalone_policy"], passes, "Failed-score passes")
            put(policy, "network_passes_for_standalone_policy", passes)
            require("prediction__" + policy not in arrays, "Failed risk scoring has action prediction")
            continue
        prediction = arrays["prediction__" + policy].astype(np.float64)
        require(prediction.shape == (n, 2), "Prediction shape differs")
        for name in ("graph_build_and_selection_seconds", "forward_operational_seconds", "scoring_seconds"):
            put(policy, name, nonnegative(case[name]))
        passes = 2 if policy == OBS_POLICIES[-1] else 1
        integer(case["network_passes_for_standalone_policy"])
        audit.equal(case["network_passes_for_standalone_policy"], passes)
        audit.equal(case["scoring_seconds"], row["previous_score"]["operational_seconds"] if passes == 2 else 0.)
        audit.equal(case["reused_native_parity_output"], policy == "base")
        put(policy, "network_passes_for_standalone_policy", passes)
        for name in ("directed_edges",) + COUNTS:
            put(policy, name, integer(case["graph"][name]))
        if np.isfinite(prediction).all():
            values = boundaries(prediction, arrays["bounds"])
            audit.equal(case["predicted_boundary"], values, "Predicted boundaries")
            for field in BOUNDARY_FIELDS:
                put(policy, "predicted_boundary_" + field, values[field])
        if not outcome(case):
            audit.equal(case["metrics"], None)
            continue
        require(np.isfinite(prediction).all(), "Successful nonfinite prediction")
        residual = prediction - target
        normalized = residual / scale
        errors = np.sum(normalized * normalized, axis=1) / 2
        for key, expected in (("position_residual", residual), ("normalized_residual", normalized), ("normalized_coordinate_se", errors)):
            audit.array(arrays[key + "__" + policy], expected, key)
        for name, value in (("position_coordinate_mse", float(np.sum(residual * residual) / (2 * n))),
                            ("normalized_coordinate_mse", float(np.sum(errors) / n))):
            audit.equal(case["metrics"][name], value, name)
            put(policy, name, value)
        good[policy] = errors
    failures = [{"case": policy, "failure": row["cases"][policy]["failure"]} for policy in OBS_POLICIES if row["cases"][policy]["status"] == "failed"]
    audit.equal(row["failure"], {"category": "case_failures", "cases": failures} if failures else None)
    correlation_records = {}
    for policy in OBS_POLICIES:
        if "base" not in good or policy not in good:
            audit.equal(row["benefits"][policy], {"mean_signed_normalized_coordinate_gain": None, "reason": "required base or action output failed"})
            continue
        gains = good["base"] - good[policy]
        audit.array(arrays["signed_normalized_coordinate_gain__" + policy], gains, "Signed gain")
        for name, value in (("mean_signed_normalized_coordinate_gain", float(np.sum(gains) / n)),
                            ("harmful_particle_fraction", float(np.count_nonzero(gains < 0) / n))):
            audit.equal(row["benefits"][policy][name], value, name)
            put(policy, name, value)
        if prior_complete:
            coefficient = spearman(arrays["previous_base_risk"], gains)
            correlation_records[policy] = coefficient
            put(policy, "previous_risk_gain_spearman", coefficient["value"], coefficient["reason"])
    audit.equal(row["previous_risk_correlations"], correlation_records, "Risk/gain correlations")
    coefficient = spearman(arrays["previous_base_risk"], good["base"]) if prior_complete and "base" in good else {
        "value": None, "reason": "required scoring or base output failed"}
    audit.equal(row["previous_risk_base_residual_correlation"], coefficient)
    for policy in OBS_POLICIES:
        put(policy, "previous_risk_base_residual_spearman", coefficient["value"], coefficient["reason"])
    return metrics, reasons


def autonomous_metrics(row, audit):
    """Every accepted scalar and attempted timing, including unsuccessful runs."""
    done, steps = outcome(row), integer(row["completed_steps"])
    require(steps <= 995 and done == (steps == 995), "Full horizon/failure prefix differs")
    audit.equal(row["horizon"], 995)
    mse, edges, attempts = row["mse_per_step"], row["directed_edges_per_step"], row["attempts"]
    audit.equal(len(mse), steps); audit.equal(len(edges), steps)
    for value in mse:
        nonnegative(value)
    for value in edges:
        integer(value)
    values = {"mean_rollout_mse": strict_mean(mse) if done else None, "failure_fraction": float(not done),
        "mean_directed_edges": strict_mean(edges) if done else None, "mean_rollout_wall_seconds": nonnegative(row["total_wall_seconds"]),
        "mean_total_network_passes": integer(row["total_network_passes"]), "completed_steps": steps,
        "forecast_network_passes": integer(row["forecast_network_passes"])}
    audit.equal(set(row["mse_at_steps"]), {str(step) for step in HORIZONS})
    for horizon in HORIZONS:
        values[f"mse_at_{horizon}"] = mse[horizon - 1] if steps >= horizon else None
        audit.equal(row["mse_at_steps"][str(horizon)], values[f"mse_at_{horizon}"], "Fixed horizon")
    for key in ("mean_rollout_mse", "mean_directed_edges"):
        audit.equal(row[key], values[key], key)
    audit.equal(row["mse_at_final_horizon"], values["mse_at_995"])
    for source in ("predicted", "ground_truth"):
        series = row[source + "_boundary_per_step"]
        audit.equal(len(series), steps)
        fractions, maxima = [], []
        for entry in series:
            audit.equal(set(entry), set(BOUNDARY_FIELDS) | {"coordinate_minimum", "coordinate_maximum"})
            exact, threshold, maximum, mean = [nonnegative(entry[field]) for field in BOUNDARY_FIELDS]
            require(0 <= threshold <= exact <= 1 and mean <= maximum + 1e-14, "Boundary scalar order differs")
            require((maximum > 0) == (exact > 0) == (mean > 0) and (maximum > 1e-6) == (threshold > 0), "Boundary fractions/excursions inconsistent")
            low, high = entry["coordinate_minimum"], entry["coordinate_maximum"]
            require(len(low) == len(high) == 2 and all(number(v) for v in low + high) and all(a <= b for a, b in zip(low, high)), "Boundary coordinate range invalid")
            fractions.append(threshold); maxima.append(maximum)
        values[source + "_boundary_mean_fraction_outside_gt_1e_minus6"] = strict_mean(fractions) if done else None
        values[source + "_boundary_mean_step_maximum_excursion"] = strict_mean(maxima) if done else None
        values[source + "_boundary_trajectory_maximum_excursion"] = max(maxima) if done else None
    require(len(attempts) in (steps, steps + 1), "Attempt count differs")
    audit.equal([integer(entry["forecast_step"], 1) for entry in attempts], list(range(1, len(attempts) + 1)))
    audit.equal([entry["accepted"] for entry in attempts], [True] * steps + [False] * (len(attempts) - steps))
    accepted = attempts[:steps]
    for name in COUNTS:
        scalars = [integer(entry[name]) for entry in accepted]
        values["mean_" + name] = strict_mean(scalars) if done else None
    values["cap_active_step_fraction"] = strict_mean([float(entry["native_base_receivers_above_cap_before_capping"] > 0) for entry in accepted]) if done else None
    warmup = row["warmup"] or {}
    passes = sum("features_forward_decode_and_transfer_seconds" in entry for entry in attempts)
    audit.equal(row["forecast_network_passes"], passes)
    warmup_passes = integer(warmup.get("passes", 0))
    require(warmup_passes in ((0, 1) if row["policy"] == AUTO_POLICIES[-1] else (0,)), "Policy warmup passes differ")
    require(steps <= passes <= steps + 1 and (not done or passes == 995 and warmup_passes == int(row["policy"] == AUTO_POLICIES[-1])), "Accepted/attempted network passes differ")
    audit.equal(row["total_network_passes"], passes + warmup_passes)
    for metric, key in (("mean_graph_operational_seconds", "graph_build_and_selection_seconds"),
                        ("mean_forward_component_seconds", "features_forward_decode_and_transfer_seconds")):
        total = math.fsum(nonnegative(entry.get(key, 0.)) for entry in attempts) + nonnegative(warmup.get(key, 0.))
        audit.equal(row[key + "_including_warmup"], total)
        values[metric] = total
    values.update(mean_wall_seconds_including_native_parity=nonnegative(row["total_wall_seconds_including_native_parity"]),
        mean_native_parity_seconds=nonnegative(row["native_parity_operational_seconds"]), mean_native_parity_passes=integer(row["native_parity_network_passes"]))
    require(values["mean_wall_seconds_including_native_parity"] >= values["mean_rollout_wall_seconds"], "Timing scopes reversed")
    audit.equal(set(values), set(AUTO_METRICS))
    return values, {name: AUTO_NULL if value is None else None for name, value in values.items()}


def contrasts(policies):
    require(policies in (OBS_POLICIES, AUTO_POLICIES), "Fixed ordered policies required")
    risk = policies[4]
    result = {"mix_minus_base__" + policy: [("mix", policy, 1), ("base", policy, -1)] for policy in policies}
    result["risk_minus_random_interaction"] = [("mix", risk, 1), ("mix", "random25", -1), ("base", risk, -1), ("base", "random25", 1)]
    for arm in ARMS:
        for label, left, right in (("risk_minus_random", risk, "random25"), ("risk_minus_speed", risk, "speed25"), ("dense_minus_base", "dense", "base")):
            result[arm + "__" + label] = [(arm, left, 1), (arm, right, -1)]
    return result


def check_hierarchy(detail, compact, lookup, units, terms, metric, audit):
    """Walk the publication, recomputing from independent unit scalars.

    Every declared unit must appear once in sorted trajectory/unit order. This
    does not invoke or reproduce the production aggregate function's lookup.
    """
    grouped = {}
    for unit in units:
        grouped.setdefault(unit["trajectory_id"], []).append(unit)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["unit_id"])
    require(len({(u["trajectory_id"], u["unit_id"]) for u in units}) == len(units) and units, "Declared units must be unique/nonempty")
    actual_seeds = detail["seeds"]
    audit.equal([integer(row["seed"]) for row in actual_seeds], list(SEEDS))
    seed_values = []
    for seed, saved in enumerate(actual_seeds):
        audit.equal([row["trajectory_id"] for row in saved["trajectories"]], sorted(grouped), "Trajectory population/order")
        trajectory_values, defined_total = [], 0
        for trajectory in saved["trajectories"]:
            tid = trajectory["trajectory_id"]
            expected_units, unit_values = grouped[tid], []
            audit.equal(len(trajectory["units"]), len(expected_units))
            for actual, unit in zip(trajectory["units"], expected_units):
                for field, expected in unit.items():
                    if type(expected) is int:
                        integer(actual[field])
                values, reasons = [], []
                for arm, policy, coefficient in terms:
                    key = arm, seed, tid, unit["unit_id"], policy
                    require(key in lookup, "Missing committed raw unit/policy")
                    metrics, nulls = lookup[key]
                    value = metrics[metric]
                    require(value is None or number(value), "Finite nonboolean metric required")
                    reason = nulls[metric]
                    require((value is None and isinstance(reason, str) and reason) or value is not None and reason is None, "Raw null/reason differs")
                    if value is None:
                        reasons.append(arm + "/" + policy + ": " + reason)
                    else:
                        values.append(coefficient * value)
                value = None if reasons else sum(values)
                audit.equal(actual, {**unit, "value": value, "null_reason": "; ".join(reasons) if reasons else None}, "Unit contrast")
                unit_values.append(value)
            trajectory_value = strict_mean(unit_values)
            defined = sum(value is not None for value in unit_values)
            integer(trajectory["required_units"]); integer(trajectory["defined_units"])
            audit.equal({key: value for key, value in trajectory.items() if key != "units"}, {
                "trajectory_id": tid, "value": trajectory_value, "required_units": len(expected_units), "defined_units": defined})
            trajectory_values.append(trajectory_value); defined_total += defined
        value = strict_mean(trajectory_values)
        for field in ("required_units", "defined_units", "null_units"):
            integer(saved[field])
        audit.equal({key: value for key, value in saved.items() if key != "trajectories"}, {"seed": seed,
            "value": value, "required_units": len(units), "defined_units": defined_total, "null_units": len(units) - defined_total})
        seed_values.append(value)
    expected = {"seed_values": seed_values, "mean": strict_mean(seed_values),
        "sample_sd": statistics.stdev(seed_values) if all(value is not None for value in seed_values) else None,
        "required_seeds": 3, "defined_seeds": sum(value is not None for value in seed_values)}
    if "terms" in detail:
        expected["terms"] = terms
    for item in (detail, compact):
        integer(item["required_seeds"]); integer(item["defined_seeds"])
    audit.equal({key: value for key, value in detail.items() if key != "seeds"}, expected, "Detailed seed summary")
    audit.equal(compact, expected, "Compact seed summary")


def check_population(directory, name, population, lookup, units, policies, metrics, audit):
    audit.equal(population["required_units_per_endpoint"], len(units))
    audit.equal(population["policies"], policies)
    audit.equal(set(population["metrics"]), set(metrics))
    expected_keys = {(arm, seed, unit["trajectory_id"], unit["unit_id"], policy) for arm, seed in COHORT for unit in units for policy in policies}
    audit.equal(set(lookup), expected_keys, "Raw population coverage")
    definitions = contrasts(policies)
    for metric in metrics:
        index = population["metrics"][metric]
        audit.equal(index["path"], name + "__" + metric + ".json.gz")
        detail = audit.compressed(directory, index)
        audit.equal(set(detail), {"absolute", "paired"})
        audit.equal(set(detail["absolute"]), set(ARMS)); audit.equal(set(index["absolute"]), set(ARMS))
        audit.equal(set(detail["paired"]), set(definitions)); audit.equal(set(index["paired"]), set(definitions))
        for arm in ARMS:
            audit.equal(set(detail["absolute"][arm]), set(policies)); audit.equal(set(index["absolute"][arm]), set(policies))
            for policy in policies:
                check_hierarchy(detail["absolute"][arm][policy], index["absolute"][arm][policy], lookup, units,
                                [(arm, policy, 1)], metric, audit)
        for label, terms in definitions.items():
            require("terms" in detail["paired"][label], "Paired terms missing")
            check_hierarchy(detail["paired"][label], index["paired"][label], lookup, units, terms, metric, audit)


def jobs(root, mode, audit):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and not (root / "run.lock").exists(), "Inactive regular family root required")
    result = {}
    for directory in sorted(root.iterdir()):
        require(directory.is_dir() and not directory.is_symlink() and not (directory / "run.lock").exists(), "Unknown or active job")
        require(not any(path.name.endswith(".tmp") or path.is_symlink() for path in directory.rglob("*")), "Temporary/symlink job artifact")
        protocol, status = audit.read(directory / "protocol.json"), audit.read(directory / "status.json")
        identity = protocol["arm"], integer(protocol["seed"])
        require(identity in COHORT and identity not in result, "Unknown/duplicate arm/original seed")
        for name, expected in (("scope", EVAL_SCOPE), ("mode", mode), ("objective", "faithful"),
            ("original_seed", identity[1]), ("completed_total_updates", 110000), ("completed_additional_updates", 10000),
            ("policies", OBS_POLICIES if mode == "observed" else AUTO_POLICIES), ("source_indices", INDICES)):
            audit.equal(protocol[name], expected, "Protocol " + name)
        for name in ("original_seed", "completed_total_updates", "completed_additional_updates"):
            integer(protocol[name])
        audit.equal(status["state"], "complete")
        publication = audit.read(directory / "result.json", status["result_sha256"])
        audit.equal(publication["state"], "complete")
        audit.equal(publication["protocol_sha256"], audit.pins[str((directory / "protocol.json").resolve())])
        for name in ("scope", "seed", "objective", "checkpoint_sha256"):
            audit.equal(publication[name], protocol[name])
        integer(publication["seed"])
        if mode == "observed":
            for name in ("arm", "parent_checkpoint_sha256"):
                audit.equal(publication[name], protocol[name])
            audit.equal(status["arm"], identity[0]); audit.equal(status["seed"], identity[1]); integer(status["seed"])
            audit.equal(status["committed_frames"], 425); audit.equal(publication["required_frames"], 425)
            audit.equal(len(publication["records"]), 425)
        else:
            audit.equal(audit.read(directory / "lineage_identity.json"), {key: protocol[key] for key in IDENTITY_FIELDS})
            nonnegative(publication["invocation_wall_seconds"])
            audit.equal(len(publication["records"]), 135)
        result[identity] = directory, protocol, publication
    audit.equal(set(result), COHORT)
    return result


def observed_units(protocol, audit):
    expected = protocol["expected_frames"]
    audit.equal(len(expected), 425)
    units = {"observed_valid": [], "observed_test": []}
    for item in expected:
        require(item["split"] in ("valid", "test") and isinstance(item["trajectory_id"], str) and item["trajectory_id"], "Observed unit identity invalid")
        index, target = integer(item["source_index"]), integer(item["target_frame"], 1)
        units["observed_" + item["split"]].append({**item, "unit_id": f"{index}:{target}"})
    audit.equal(len(units["observed_valid"]), 128); audit.equal(len(units["observed_test"]), 297)
    require({u["source_index"] for u in units["observed_test"]} == set(INDICES), "Exact test source population required")
    for values in units.values():
        require(len({(u["trajectory_id"], u["unit_id"]) for u in values}) == len(values), "Duplicate scheduled unit")
    return units


def load_observed(job, units, audit):
    directory, protocol, publication = job
    expected = {(unit["split"], unit["source_index"], unit["target_frame"]): unit for values in units.values() for unit in values}
    require(len(expected) == 425, "Observed schedule identities duplicate")
    rows, seen, failed = {"observed_valid": {}, "observed_test": {}}, set(), 0
    for compact in publication["records"]:
        identity = compact["split"], integer(compact["source_index"]), integer(compact["target_frame"], 1)
        require(identity in expected and identity not in seen, "Unexpected/duplicate observed record")
        seen.add(identity); unit = expected[identity]
        stem = f"{identity[0]}_{identity[1]:06d}_{identity[2]:04d}"
        audit.equal(compact["record_file"], stem + ".json"); audit.equal(compact["array_file"], stem + ".npz")
        row = audit.read(local_path(directory, compact["record_file"]), compact["record_sha256"])
        for name, value in compact.items():
            if name not in ("record_file", "record_sha256"):
                audit.equal(row[name], value, "Observed compact/raw")
        for name, value in unit.items():
            if name != "unit_id":
                audit.equal(row[name], value)
        for name in ("arm", "original_seed", "checkpoint_sha256", "parent_checkpoint_sha256"):
            audit.equal(row[name], protocol[name])
        integer(row["original_seed"]); integer(row["source_index"]); integer(row["target_frame"], 1)
        audit.equal(row["protocol_sha256"], publication["protocol_sha256"])
        path = local_path(directory, compact["array_file"])
        audit.pin(path, row["array_sha256"])
        with np.load(path, allow_pickle=False) as arrays:
            values, reasons = observed_metrics(row, arrays, audit)
        failed += int(row["status"] == "failed")
        population = rows["observed_" + identity[0]]
        for policy in OBS_POLICIES:
            population[protocol["arm"], protocol["seed"], unit["trajectory_id"], unit["unit_id"], policy] = values[policy], reasons[policy]
    audit.equal(seen, set(expected)); audit.equal(publication["failed_frames"], failed); audit.equal(publication["complete_frames"], 425 - failed)
    return rows


def load_autonomous(job, units, audit):
    directory, protocol, publication = job
    expected = {(unit["source_index"], policy): unit for unit in units for policy in AUTO_POLICIES}
    rows, seen = {}, set()
    for compact in publication["records"]:
        identity = integer(compact["source_index"]), compact["policy"]
        require(identity in expected and identity not in seen, "Unexpected/duplicate autonomous record")
        seen.add(identity); unit = expected[identity]
        stem = f"trajectory_{identity[0]:06d}_{identity[1]}"
        audit.equal(compact["record_file"], stem + ".json"); audit.equal(compact["trace_file"], stem + ".npz")
        row = audit.read(local_path(directory, compact["record_file"]), compact["record_sha256"])
        for name, value in compact.items():
            if name not in ("record_file", "record_sha256"):
                audit.equal(row[name], value, "Autonomous compact/raw")
        integer(row["source_index"])
        audit.equal(row["trajectory_id"], unit["trajectory_id"])
        audit.equal(row["protocol_sha256"], publication["protocol_sha256"])
        audit.equal(row["rng_seed"], 93000 + 1000 * protocol["seed"] + identity[0])
        audit.pin(local_path(directory, compact["trace_file"]), compact["trace_sha256"])
        rows[protocol["arm"], protocol["seed"], unit["trajectory_id"], unit["unit_id"], identity[1]] = autonomous_metrics(row, audit)
    audit.equal(seen, set(expected))
    return rows


def execute(summary_path, observed_root, autonomous_root, audit):
    summary_path = Path(summary_path)
    status = audit.read(summary_path.parent / "status.json")
    audit.equal(status["state"], "complete")
    report = audit.read(summary_path, status["result_sha256"])
    audit.equal(report["state"], "complete"); audit.equal(report["scope"], SCOPE)
    audit.equal(report["source_and_input_reverified_after_analysis"], True)
    audit.equal(report["completed_total_updates"], 110000); audit.equal(report["completed_additional_updates"], 10000)
    integer(report["completed_total_updates"]); integer(report["completed_additional_updates"])
    audit.equal(report["coverage"], {"endpoints": 6, "jobs": 12, "observed_frames": 2550, "observed_policy_slots": 12750, "autonomous_outcomes": 810})
    for value in report["coverage"].values():
        integer(value)
    summary_sources = [(path, digest) for path, digest in report["input_files_sha256"].items() if Path(path).name == "summarize_continuation_evaluation.py"]
    require(len(summary_sources) == 1 and summary_sources[0][1] == SUMMARY_SHA256, "Frozen summary source pin differs")
    audit.pin(summary_sources[0][0], SUMMARY_SHA256)
    families = {"observed": jobs(observed_root, "observed", audit), "autonomous": jobs(autonomous_root, "autonomous", audit)}
    cohort = {}
    for entry in report["cohort"]:
        key = entry["arm"], integer(entry["seed"])
        require(key in COHORT and key not in cohort, "Summary cohort identity duplicate/unknown")
        cohort[key] = entry
    audit.equal(set(cohort), COHORT)
    require(len({entry["sha256"] for entry in cohort.values()}) == 6, "Six distinct endpoint hashes required")
    units = observed_units(families["observed"]["base", 0][1], audit)
    ids = families["autonomous"]["base", 0][1]["trajectory_ids"]
    require(len(ids) == 27 and len(set(ids)) == 27 and all(isinstance(value, str) and value for value in ids), "27 explicit trajectory identities required")
    units["autonomous_test"] = [{"source_index": index, "trajectory_id": tid, "unit_id": str(index)} for index, tid in zip(INDICES, ids)]
    for unit in units["observed_test"]:
        audit.equal(unit["trajectory_id"], ids[unit["source_index"] - 3])
    summary_jobs = {}
    for job in report["jobs"]:
        key = job["mode"], job["arm"], integer(job["seed"])
        require(key not in summary_jobs, "Duplicate summary job")
        summary_jobs[key] = job
    audit.equal(set(summary_jobs), {(mode, arm, seed) for mode in families for arm, seed in COHORT})
    populations = {name: {} for name in units}
    parents = {}
    for identity in sorted(COHORT):
        left = families["observed"][identity][1]
        right = families["autonomous"][identity][1]
        for field in IDENTITY_FIELDS:
            audit.equal(left[field], right[field], "Observed/autonomous endpoint identity")
        audit.equal(left["checkpoint_sha256"], cohort[identity]["sha256"])
        seed = identity[1]
        if seed in parents:
            audit.equal(left["parent_checkpoint_sha256"], parents[seed], "Paired original parent")
        parents[seed] = left["parent_checkpoint_sha256"]
        audit.equal(left["expected_frames"], families["observed"]["base", 0][1]["expected_frames"])
        audit.equal(right["trajectory_ids"], ids)
        for mode in families:
            directory, protocol, publication = families[mode][identity]
            row = summary_jobs[mode, *identity]
            audit.equal(row["directory"], str(directory.resolve()))
            audit.equal(row["protocol_sha256"], publication["protocol_sha256"])
            audit.equal(row["result_sha256"], audit.pins[str((directory / "result.json").resolve())])
            audit.equal(row["records"], 425 if mode == "observed" else 135)
            audit.equal(row["failed_records"], sum(item["status"] == "failed" for item in publication["records"]))
            integer(row["records"]); integer(row["failed_records"])
            audit.equal(row["invocation_wall_seconds"], publication.get("invocation_wall_seconds"))
        observed = load_observed(families["observed"][identity], {name: value for name, value in units.items() if name.startswith("observed")}, audit)
        for name, values in observed.items():
            populations[name].update(values)
        populations["autonomous_test"].update(load_autonomous(families["autonomous"][identity], units["autonomous_test"], audit))
    audit.equal(set(report["populations"]), set(units))
    for path, digest in audit.pins.items():
        if any(Path(root).resolve() in Path(path).parents for root in (observed_root, autonomous_root)):
            audit.equal(report["input_files_sha256"].get(path), digest, "Summary/raw input hash binding")
    for name in units:
        observed = name.startswith("observed")
        check_population(summary_path.parent, name, report["populations"][name], populations[name], units[name],
                         OBS_POLICIES if observed else AUTO_POLICIES, OBS_METRICS if observed else AUTO_METRICS, audit)
    # The lossless raw archive is byte-verified; arithmetic comes independently
    # from the original raw jobs, not from this summary-produced intermediate.
    audit.pin(local_path(summary_path.parent, report["audited_records"]["path"]), report["audited_records"]["sha256"])
    for path, digest in audit.pins.items():
        audit.equal(sha(path), digest, "Input changed before audit completion")
    return {"observed_frames": 2550, "observed_policy_slots": 12750, "autonomous_outcomes": 810,
        "populations": 3, "metrics_per_population": {"observed_valid": len(OBS_METRICS), "observed_test": len(OBS_METRICS), "autonomous_test": len(AUTO_METRICS)},
        "paired_contrasts_per_metric": 12, "source_summary_sha256": audit.pins[str(summary_path.resolve())]}


def isolated_output(output, summary, observed, autonomous):
    output = Path(output).resolve()
    require(not output.exists(), "Fresh audit output required")
    protected = [Path(summary).resolve().parent, Path(observed).resolve(), Path(autonomous).resolve()]
    # Metadata-only read; never load a checkpoint or create a directory first.
    report = decode(Path(summary).read_bytes())
    protected.extend(Path(entry["checkpoint"]).resolve().parent for entry in report["cohort"])
    # The strict summary pins original100k parents, converted data, manifests,
    # scientific sources and configuration files as well as110k endpoints.
    # Protect every pinned file's directory without opening any such file.
    pins = report["input_files_sha256"]
    require(isinstance(pins, dict), "Summary input identity map required before output creation")
    for name in pins:
        require(isinstance(name, str) and Path(name).is_absolute(), "Absolute immutable input path required")
        protected.append(Path(name).resolve().parent)
    require(not any(output == path or path in output.parents or output in path.parents for path in protected), "Audit output overlaps immutable input tree")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "observed-root", "autonomous-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    directory = isolated_output(args.output_dir, args.summary, args.observed_root, args.autonomous_root)
    directory.mkdir(parents=True, exist_ok=False)
    audit, started = Audit(), time.perf_counter()
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": audit.pin(__file__),
        "scope": "Independent raw observed-array/autonomous-scalar arithmetic and complete paired hierarchy; no checkpoint/model/data-source admission, scientific guard or graph/autonomous-trace reconstruction, or inference. Frozen strict summarizer provides those complementary checks."}
    try:
        record.update(execute(args.summary, args.observed_root, args.autonomous_root, audit), passed=True)
    except BaseException as error:
        record.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(checks=audit.checks, wall_seconds=time.perf_counter() - started, input_files_sha256=audit.pins)
        (directory / "audit.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
