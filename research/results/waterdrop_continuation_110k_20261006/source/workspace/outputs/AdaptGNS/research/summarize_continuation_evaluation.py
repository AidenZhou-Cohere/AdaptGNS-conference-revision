"""Strict saved-array analysis of the separate paired faithful 110k study.

CPU checkpoint deserialization admits the six endpoint/parent/Adam lineages.
No simulator is constructed and no model forward is called. All twelve jobs
must be committed before complete-population summaries can be published.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from research import continuation_evaluation as evaluation
from research import summarize_graph_convention_bridge as observed_audit
from research import summarize_native_graph_rollouts as native_audit
from research import summarize_full_rollouts as scalar_audit

full, bridge, native = evaluation.full, evaluation.bridge, evaluation.native
Audit = observed_audit.Audit
PROTOCOL = Path(__file__).parent / "protocols/continuation_summary_20261006.md"
SCOPE = "post-inspection exploratory paired faithful110k continuation analysis"
EVALUATION_SCOPE = "post-inspection exploratory faithful110k graph-support endpoint evaluation"
FROZEN = {
    "research/continuation_evaluation.py": "ec2ee1ba6e2a0cb78c55bd2720031af5d4fb8535d17c679aa6d319b74378a528",
    "research/protocols/continuation_evaluation_20261005.md": "ce0ae3028155775c39c56cf51860cda07a54bd912f24af60e5b44110f826e7b4",
    "research/summarize_native_graph_rollouts.py": "c7999ae9062e19ab10033537749d725145768aa08f7a548f092bcb1d5697c815",
    "research/summarize_full_rollouts.py": "fee246c0abe0a289374dcc7b383be8827579fdeb711f966ae9d2a3480048070b",
    "research/summarize_graph_convention_bridge.py": "67558f70cc3774eca5134b8d900f7312b2f2091b01e1870ab9954915b56855c9",
}
IDENTITY_FIELDS = ("arm", "seed", "original_seed", "checkpoint_sha256", "parent_checkpoint_sha256",
                   "continuation_config_sha256", "completed_total_updates", "completed_additional_updates")
BOUNDARIES = ("fraction_particles_outside", "fraction_particles_outside_by_more_than_1e-6",
              "maximum_coordinate_excursion", "mean_particle_maximum_excursion")
OBSERVED_METRICS = ("position_coordinate_mse", "normalized_coordinate_mse", "failure_fraction",
    "mean_signed_normalized_coordinate_gain", "harmful_particle_fraction", "previous_risk_gain_spearman",
    "previous_risk_base_residual_spearman", "graph_build_and_selection_seconds", "forward_operational_seconds",
    "scoring_seconds", "network_passes_for_standalone_policy", "directed_edges", "retained_optional_pairs",
    "native_base_directed_edges", "native_base_self_edges", "native_base_receivers_above_cap_before_capping",
    "native_base_edges_removed_by_cap", "native_base_asymmetric_directed_edges") + tuple(
    f"{source}_boundary_{field}" for source in ("predicted", "ground_truth", "current_observed") for field in BOUNDARIES)
AUTONOMOUS_METRICS = native_audit.METRICS + tuple(
    f"mse_at_{step}" for step in scalar_audit.TRACE_STEPS if f"mse_at_{step}" not in native_audit.METRICS) + (
    "completed_steps", "forecast_network_passes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    """Reject duplicate keys and nonfinite JSON before semantic validation."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON token: {value}")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def finite(value):
    return type(value) in (int, float) and np.isfinite(value)


def strict_mean(values):
    require(all(value is None or finite(value) for value in values), "Expected finite number or explicit null")
    return float(np.mean(values)) if values and all(value is not None for value in values) else None


def unit_key(row):
    require(isinstance(row.get("trajectory_id"), str) and row["trajectory_id"], "Trajectory identifier required")
    require(isinstance(row.get("unit_id"), str) and row["unit_id"], "Unit identifier required")
    return row["trajectory_id"], row["unit_id"]


def hierarchical(values, expected_units):
    """Retain every unit/null reason, equal histories then equal trajectories."""
    expected = {unit_key(row): row for row in expected_units}
    require(expected and len(expected) == len(expected_units), "Unique nonempty declared units required")
    require(set(values) == set(expected), "Unknown or missing hierarchy units")
    grouped = {}
    for key, item in sorted(expected.items()):
        value, reason = values[key]
        require(value is None or finite(value), "Finite nonboolean metric required")
        require((value is None and isinstance(reason, str) and reason) or (value is not None and reason is None),
                "Metric/null reason mismatch")
        grouped.setdefault(key[0], []).append({**item, "value": value, "null_reason": reason})
    trajectories = []
    for name, units in grouped.items():
        trajectories.append({"trajectory_id": name, "value": strict_mean([row["value"] for row in units]),
            "required_units": len(units), "defined_units": sum(row["value"] is not None for row in units), "units": units})
    defined = sum(row["value"] is not None for rows in grouped.values() for row in rows)
    return {"value": strict_mean([row["value"] for row in trajectories]), "required_units": len(expected),
            "defined_units": defined, "null_units": len(expected) - defined, "trajectories": trajectories}


def three_seeds(seeds):
    require([row["seed"] for row in seeds] == [0, 1, 2], "Exactly three ordered original seeds required")
    values = [row["value"] for row in seeds]
    complete = all(value is not None for value in values)
    return {"seeds": seeds, "seed_values": values, "mean": strict_mean(values),
            "sample_sd": float(np.std(values, ddof=1)) if complete else None,
            "required_seeds": 3, "defined_seeds": sum(value is not None for value in values)}


def contrast_definitions(policies):
    require(tuple(policies) in (evaluation.POLICIES, native.POLICIES), "Exact ordered five-policy family required")
    risk = policies[-1]
    definitions = {f"mix_minus_base__{policy}": [("mix", policy, 1), ("base", policy, -1)] for policy in policies}
    definitions["risk_minus_random_interaction"] = [("mix", risk, 1), ("mix", "random25", -1),
                                                     ("base", risk, -1), ("base", "random25", 1)]
    for arm in ("base", "mix"):
        for label, a, b in (("risk_minus_random", risk, "random25"), ("risk_minus_speed", risk, "speed25"),
                             ("dense_minus_base", "dense", "base")):
            definitions[f"{arm}__{label}"] = [(arm, a, 1), (arm, b, -1)]
    return definitions


def aggregate_metric(records, expected_units, policies=evaluation.POLICIES):
    """Pure absolute and paired hierarchy; partial values propagate null.

    A row has arm, seed, trajectory_id, unit_id, policy_values and optional
    policy_null_reasons. Missing rows become explicit nulls, never survivors.
    The artifact loader separately requires all committed units to exist.
    """
    definitions = contrast_definitions(policies)
    expected = {unit_key(row) for row in expected_units}
    require(expected and len(expected) == len(expected_units), "Unique nonempty declared units required")
    lookup = {}
    for row in records:
        unit = unit_key(row)
        require(type(row.get("seed")) is int and (row.get("arm"), row["seed"]) in evaluation.COHORT,
                "Unknown/noninteger arm or original seed")
        key = row["arm"], row["seed"], *unit
        require(unit in expected and key not in lookup, "Unknown/duplicate paired unit")
        require(set(row["policy_values"]) == set(policies), "Exact policy values required")
        require(all(value is None or finite(value) for value in row["policy_values"].values()), "Finite nonboolean value required")
        lookup[key] = row
    def terms_for(seed, terms):
        values = {}
        for unit in expected:
            terms_values, reasons = [], []
            for arm, policy, coefficient in terms:
                row = lookup.get((arm, seed, *unit))
                value = row["policy_values"][policy] if row else None
                if value is None:
                    reason = row.get("policy_null_reasons", {}).get(policy) if row else "missing committed unit"
                    reasons.append(f"{arm}/{policy}: {reason or 'required metric undefined'}")
                else:
                    terms_values.append(coefficient * value)
            values[unit] = (None, "; ".join(reasons)) if reasons else (float(sum(terms_values)), None)
        return {"seed": seed, **hierarchical(values, expected_units)}
    absolute = {arm: {policy: three_seeds([terms_for(seed, [(arm, policy, 1)]) for seed in (0, 1, 2)])
                     for policy in policies} for arm in ("base", "mix")}
    paired = {name: {"terms": terms, **three_seeds([terms_for(seed, terms) for seed in (0, 1, 2)])}
              for name, terms in definitions.items()}
    reference = evaluation.paired_contrasts(records, expected_units, policies)
    for name, result in paired.items():
        require(result["seed_values"] == [row["difference"] for row in reference[name]["seeds"]]
                and result["mean"] == reference[name]["mean"] and result["sample_sd"] == reference[name]["sample_sd"],
                "Paired hierarchy differs from frozen comparison helper")
    return {"absolute": absolute, "paired": paired}


def inventory(directory, expected_names):
    """Strict family inventory; archived statuses are preserved and verified."""
    directory = Path(directory)
    files, actual = {}, set()
    require(directory.is_dir() and not directory.is_symlink(), "Regular job directory required")
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Symlink artifact rejected")
        relative = path.relative_to(directory).as_posix()
        require(not relative.endswith(".tmp"), "Temporary artifact requires review")
        if path.is_dir():
            require(relative == "attempt_history" or (relative.startswith("attempt_history/") and len(path.relative_to(directory).parts) == 2),
                    "Unknown nested artifact directory")
            continue
        require(path.is_file(), "Nonregular artifact rejected")
        files[str(path.resolve())] = full.sha256(path)
        if relative.startswith("attempt_history/"):
            parts = Path(relative).parts
            require(len(parts) == 3 and parts[-1] in ("status.json", "archive.json"), "Unknown attempt-history artifact")
        else:
            actual.add(relative)
    require(actual == set(expected_names), "Missing/orphan/unknown job artifacts")
    history = directory / "attempt_history"
    if history.exists():
        for attempt in history.iterdir():
            require(attempt.is_dir() and {p.name for p in attempt.iterdir()} == {"status.json", "archive.json"},
                    "Incomplete attempt-history archive")
            archived = read_json(attempt / "archive.json")
            require(archived.get("source") == "status.json" and archived.get("sha256") == full.sha256(attempt / "status.json"),
                    "Archived status identity differs")
    return files


def discover_jobs(root, mode):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Regular family root required")
    jobs = {}
    for directory in sorted(root.iterdir()):
        require(directory.is_dir() and not directory.is_symlink(), "Unknown family artifact or job directory")
        protocol = read_json(directory / "protocol.json")
        identity = protocol.get("arm"), protocol.get("seed")
        require(type(protocol.get("seed")) is int and identity in evaluation.COHORT and identity not in jobs,
                "Unknown/duplicate job arm or seed")
        require(protocol.get("mode") == mode, "Mislabeled evaluation job family")
        jobs[identity] = directory
    require(set(jobs) == evaluation.COHORT, "Exactly six committed arm/seed jobs required")
    return jobs


def validate_job_protocol(protocol, mode, entry, config, entries, expected, manifest, required_pins, audit):
    constants = {"schema": 1, "scope": EVALUATION_SCOPE, "mode": mode, "arm": entry["arm"],
        "seed": entry["seed"], "original_seed": entry["seed"], "objective": "faithful",
        "checkpoint_sha256": entry["sha256"], "parent_checkpoint_sha256": config["parent_checkpoint_sha256"],
        "continuation_config_sha256": evaluation.support.original.config_hash(config),
        "completed_total_updates": 110000, "completed_additional_updates": 10000,
        "expected_frames": expected if mode == "observed" else None,
        "source_indices": list(range(3, 30)), "trajectory_ids": [row["id"] for row in manifest["records"][3:]],
        "source_tfrecord": manifest["source"], "horizon": 1 if mode == "observed" else 995,
        "policies": list(evaluation.POLICIES if mode == "observed" else native.POLICIES),
        "graph": "unchanged strict-r native capped128+self directed prefix; uncapped symmetric optional annulus suffix; r=.015,R=1.267r",
        "observed_risk": "previous observed six-frame native-base scoring; fresh every observed frame; two standalone network passes",
        "autonomous_risk": "one initial native-base warmup; forecast1 exact budget; cache from previous own selected graph and predicted history",
        "random_observed": "[20261005,771,original_seed,split_code,source_index,target_frame]; no arm identity",
        "random_autonomous": "93000+1000*original_seed+source_index; no arm identity; later geometry may differ",
        "parent_archive_usage": "only byte-verified histories, target and types; never endpoint output attribution",
        "native_parity": "two separate verification calls per frame/trajectory; raw outputs retained; parity failure halts model job",
        "time_scope": "operational synchronized calls with fixed order/shared preprocessing; not policy-speed or speedup evidence",
        "guards": {"max_candidate_pairs": 100000, "max_abs_coordinate": 10.}, "deadline_utc": native.DEADLINE, "threads": 2}
    for key, value in constants.items():
        audit.equal(protocol.get(key), value, "Endpoint evaluation protocol differs: " + key)
    require(all(type(protocol[key]) is int for key in ("schema", "seed", "original_seed", "completed_total_updates",
                                                     "completed_additional_updates", "threads", "horizon")),
            "Integer schema/seed/update/runtime counters required")
    cohort = evaluation.validate_cohort_manifest({"schema": 1, "scope": "faithful_graph_support_110k_endpoints",
                                               "endpoints": protocol["cohort"]}, full.REPO)
    audit.equal(cohort, entries, "Evaluation cohort identity differs")
    pins = protocol["input_files_sha256"]
    require(all(Path(path).is_absolute() and str(Path(path).resolve()) == path and scalar_audit.valid_hash(digest)
                for path, digest in pins.items()), "Canonical absolute file pins required")
    for path, digest in required_pins.items():
        audit.equal(pins.get(path), digest, "Evaluation missing/different required admission input: " + path)
    for relative, digest in FROZEN.items():
        if relative.startswith("research/summarize_"):
            continue  # Analysis source did not exist among inference inputs.
        audit.equal(pins.get(str((full.REPO / relative).resolve())), digest, "Frozen evaluator pin differs")
    for path in (Path(native.__file__), Path(bridge.__file__), Path(bridge.same.__file__), Path(full.__file__),
                 native.PROTOCOL, bridge.PROTOCOL, evaluation.TRAINING_PROTOCOL, full.REPO / "research/budget_graph.py"):
        audit.equal(pins.get(str(path.resolve())), full.sha256(path), "Required numerical dependency pin differs")
    runtime = protocol["runtime"]
    audit.equal(runtime.get("device"), "mps", "Real evaluation device differs")
    audit.equal(runtime.get("radius_backend"), "scipy_host", "Graph backend differs")
    audit.equal(runtime.get("mps_fallback_environment"), "0", "MPS fallback must be disabled")
    audit.equal(runtime.get("graph_execution"), "cpu_scipy_with_device_transfer", "Graph transfer scope differs")
    audit.equal(runtime.get("torch_version"), protocol["software"]["torch"], "Runtime/software torch differs")
    provenance = protocol["checkpoint_provenance"]
    for key, value in {"checkpoint_format": 2, "normalization_source": "checkpoint", "checkpoint_radius_backend": "scipy_host",
        "connectivity_radius": .015, "nmessage_passing_steps": 10, "uncertainty_parameterization": "variance",
        "max_num_neighbors": 128, "radius_backend": "scipy_host", "edge_convention": "directed_with_self_loops",
        "runtime": runtime, "training_config": {"loss": "faithful", "graph_support_continuation": config,
            "parent_training_config": config["parent_training_config"], "completed_optimizer_updates": 110000}}.items():
        audit.equal(provenance.get(key), value, "Loaded checkpoint provenance differs: " + key)
    saved = protocol.get("saved_parent_test_input_archive")
    if mode == "autonomous":
        audit.equal(saved, None, "Autonomous job must not use an observed parent output archive")
    else:
        require(isinstance(saved, dict), "Observed parent input archive identity required")
        archive_rows = saved["records"]
        expected_test = {(row["source_index"], row["target_frame"], row["trajectory_id"]) for row in expected if row["split"] == "test"}
        actual = {(row["source_index"], row["target_frame"], row["trajectory_id"]) for row in archive_rows}
        audit.equal(len(archive_rows), 297, "Parent input archive count differs")
        audit.equal(actual, expected_test, "Parent input archive frame identities differ")
        for name, digest in (("result.json", saved["result_sha256"]), ("protocol.json", saved["protocol_sha256"])):
            audit.equal(sum(Path(path).name == name and value == digest for path, value in pins.items()), 1,
                        "Parent archive publication hash not uniquely pinned")
        for row in archive_rows:
            for key in ("record", "array"):
                name, digest = row[key + "_file"], row[key + "_sha256"]
                require(Path(name).name == name, "Unsafe parent input archive path")
                audit.equal(sum(Path(path).name == name and value == digest for path, value in pins.items()), 1,
                            "Parent archive raw artifact hash not uniquely pinned")
    return scalar_audit.canonical_hash({key: protocol[key] for key in ("graph", "guards", "runtime", "threads", "software",
        "observed_risk", "autonomous_risk", "random_observed", "random_autonomous", "time_scope")})


def reproduce_guard(saved, action, audit):
    require(isinstance(saved, dict) and isinstance(saved.get("category"), str), "Scientific guard reason required")
    try:
        action()
    except full.RolloutGuard as error:
        for key, value in error.details.items():
            # The evaluator overwrites the numerical guard's local phase
            # (e.g. prediction) with its pipeline phase (e.g. forward).
            if key != "phase":
                audit.equal(saved.get(key), value, "Saved scientific failure differs from retained arrays")
    else:
        raise ValueError("Saved failure is not reproduced by retained arrays")


def output(arrays, prefix, n):
    result = {key: arrays[prefix.format(key=key)] for key in ("prediction", "raw_risk", "risk")}
    require(result["prediction"].shape == (n, 2) and result["raw_risk"].shape == result["risk"].shape == (n,),
            "Saved prediction/risk shape differs")
    require(all(value.dtype == np.float32 for value in result.values()), "Native float32 outputs required")
    require(np.array_equal(result["risk"], np.maximum(result["raw_risk"], np.float32(1e-6)), equal_nan=True),
            "Saved variance differs from frozen faithful head conversion")
    return result


def overlap(left, right):
    if left is None or right is None:
        return {"intersection_pairs": None, "union_pairs": None, "jaccard": None, "reason": "required selection unavailable"}
    a, b = bridge.same.pair_set(left), bridge.same.pair_set(right)
    return {"intersection_pairs": len(a & b), "union_pairs": len(a | b),
            "jaccard": len(a & b) / len(a | b) if a | b else None,
            "reason": None if a | b else "empty_optional_union"}


def analyze_observed_frame(row, arrays, audit):
    """Audit saved arrays only; failures are phase-specific and policy-local."""
    n = row["n_particles"]
    require(type(n) is int and n > 0, "Positive integer particle count required")
    current, previous, scale, bounds = (arrays[key] for key in ("current_history", "previous_history", "acceleration_std", "bounds"))
    require(current.shape == previous.shape == (6, n, 2) and arrays["particle_types"].shape == (n,), "History/type shapes differ")
    require(current.dtype == previous.dtype == np.float32 and arrays["particle_types"].dtype == np.int64, "History/type dtype differs")
    require(scale.shape == (2,) and np.isfinite(scale).all() and np.all(scale > 0), "Positive two-coordinate normalization required")
    audit.equal(full.state_hash(current), row["current_history_sha256"], "Current history hash differs")
    audit.equal(full.state_hash(previous), row["previous_history_sha256"], "Previous history hash differs")
    audit.equal(set(row["cases"]), set(evaluation.POLICIES), "All five policy slots required")
    audit.check(row["status"] in ("complete", "failed") and (row["status"] == "complete") == (row["failure"] is None), "Frame status/failure differs")
    audit.check(finite(row["operational_seconds"]) and row["operational_seconds"] >= 0, "Invalid measured frame duration")
    metrics = {p: {name: None for name in OBSERVED_METRICS} for p in evaluation.POLICIES}
    reasons = {p: {name: "required scientific output unavailable" for name in OBSERVED_METRICS} for p in evaluation.POLICIES}
    def put(policy, name, value, reason=None):
        require(value is None or finite(value), "Finite metric or scientific null required")
        metrics[policy][name], reasons[policy][name] = value, reason if value is None else None
    for policy in evaluation.POLICIES:
        case = row["cases"][policy]
        require(case["status"] in ("complete", "failed") and (case["status"] == "complete") == (case["failure"] is None), "Case status/failure differs")
        put(policy, "failure_fraction", float(case["status"] == "failed"))
    pairing = {key: full.state_hash(arrays[key]) if key in arrays else None for key in (
        "current_history", "previous_history", "particle_types", "acceleration_std", "bounds", "target_position",
        "strict_base_pairs", "strict_annulus_pairs")}
    result = {**{key: row[key] for key in ("split", "source_index", "target_frame", "trajectory_id", "status", "failure")},
        "unit_id": f"{row['source_index']}:{row['target_frame']}", "metrics": metrics, "null_reasons": reasons,
        "diagnostics": row, "pairing_hashes": pairing, "policy_graph_hashes": {}, "_optional": {}}
    failure = row["failure"] or {}
    early_phase = failure.get("phase")
    if early_phase in ("current_history", "current_graph"):
        action = (lambda: full.check_state(current, "current_history", bridge.MAX_ABS)) if early_phase == "current_history" else (
            lambda: native.native_graph(current, "base", None, np.random.default_rng(0)))
        reproduce_guard(failure, action, audit)
        audit.equal(row["native_parity"], None, "Early failure unexpectedly passed parity")
        for case in row["cases"].values():
            audit.equal(case, {"status": "failed", "failure": failure, "metrics": None}, "Early failure policy slot differs")
        audit.check("target_position" not in arrays and not any(key.startswith("prediction__") for key in arrays), "Early failure contains downstream outputs")
        return result
    # Existing bridge parity validator is numerical only. Supply its alias
    # without changing the continuation row or inventing any 100k identity.
    parity_arrays = dict(arrays)
    for key in ("prediction", "risk", "raw_risk"):
        parity_arrays[f"{key}__base_cap128_loops1"] = arrays[f"supplied_parity_{key}"]
    observed_audit.audit_native_parity(row, parity_arrays, audit)
    if failure.get("category") == "native_parity_failure":
        audit.check(not row["native_parity"]["passed"], "Failed parity marked passing")
        audit.check(all(case["status"] == "failed" and case["metrics"] is None for case in row["cases"].values()), "Parity failure has successful policies")
        audit.check("target_position" not in arrays, "Parity failure contains downstream target")
        return result
    audit.check(row["native_parity"]["passed"], "Policies require passing parity")
    rng = lambda: np.random.default_rng(np.random.SeedSequence(row["random_seed_material"]))
    graph, base_edges, _, _ = native.native_graph(current, "base", None, rng())
    audit.array(arrays["strict_base_pairs"], graph.base, "Strict base candidates differ")
    audit.array(arrays["strict_annulus_pairs"], graph.extra, "Strict optional candidates differ")
    audit.equal(row["optional_budget"], int(.25 * len(graph.extra)), "Observed exact optional budget differs")
    prior = row["previous_score"]
    require(prior["status"] in ("complete", "failed") and (prior["status"] == "complete") == (prior["failure"] is None), "Previous-score status differs")
    score_output = None
    if prior["status"] == "failed":
        phase = prior["failure"]["phase"]
        if phase == "previous_history":
            reproduce_guard(prior["failure"], lambda: full.check_state(previous, "previous_history", bridge.MAX_ABS), audit)
        elif phase == "previous_graph":
            reproduce_guard(prior["failure"], lambda: native.native_graph(previous, "base", None, rng()), audit)
        elif phase == "previous_forward":
            score_output = output(arrays, "previous_base_{key}", n)
            reproduce_guard(prior["failure"], lambda: bridge.validate_output(score_output, n), audit)
        else:
            raise ValueError("Unknown previous-score failure phase")
    else:
        full.check_state(previous, "previous_history", bridge.MAX_ABS)
        _, old_edges, _, old_audit = native.native_graph(previous, "base", None, rng())
        audit.array(arrays["previous_base_edges"], old_edges, "Previous native graph differs")
        audit.equal(prior["graph"], old_audit, "Previous native graph audit differs")
        score_output = output(arrays, "previous_base_{key}", n)
        bridge.validate_output(score_output, n)
    if prior["status"] == "failed" and prior["failure"]["phase"] == "previous_forward":
        _, old_edges, _, _ = native.native_graph(previous, "base", None, rng())
        audit.array(arrays["previous_base_edges"], old_edges, "Failed previous-score native graph differs")
    if score_output is not None:
        audit.check(finite(prior["operational_seconds"]) and prior["operational_seconds"] >= 0, "Invalid previous scoring duration")
    target = arrays["target_position"]
    require(target.shape == (n, 2) and np.isfinite(target).all(), "Finite matching target required")
    audit.equal(full.state_hash(target), row["target_sha256"], "Target hash differs")
    for source, position in (("ground_truth", target), ("current_observed", current[-1])):
        calculated = full.boundary_metrics(position, bounds)
        audit.equal(row[source + "_boundary"], calculated, "Reference boundary differs")
        for policy in evaluation.POLICIES:
            for field in BOUNDARIES:
                put(policy, source + "_boundary_" + field, calculated[field])
    good = {}
    for policy in evaluation.POLICIES:
        case = row["cases"][policy]
        if policy == evaluation.POLICIES[-1] and prior["status"] == "failed":
            audit.equal(case["failure"], {"category": "scoring_pass_failed", "cause": prior["failure"]}, "Risk scoring failure differs")
            audit.equal(case["metrics"], None, "Failed scoring has metrics")
            audit.equal(case["network_passes_for_standalone_policy"], 1 if score_output is not None else 0, "Failed scoring pass count differs")
            put(policy, "network_passes_for_standalone_policy", case["network_passes_for_standalone_policy"])
            audit.check(f"prediction__{policy}" not in arrays, "Failed scoring unexpectedly produced risk-policy output")
            continue
        actual = "laggedrisk25" if policy == evaluation.POLICIES[-1] else policy
        score = score_output["risk"] if actual == "laggedrisk25" else None
        _, edges, optional, graph_audit = native.native_graph(current, actual, score, rng())
        audit.array(arrays[f"edges__{policy}"], edges, "Selected native graph or random mask differs")
        audit.array(arrays[f"optional_pairs__{policy}"], optional, "Selected optional pairs differ")
        audit.array(edges[:, :base_edges.shape[1]], base_edges, "Native mandatory ordered prefix differs")
        audit.equal(case["graph"], graph_audit, "Selected graph audit differs")
        if policy in ("speed25", evaluation.POLICIES[-1]):
            expected_score = score if score is not None else np.linalg.norm(current[-1] - current[-2], axis=-1)
            audit.array(arrays[f"selection_score__{policy}"], expected_score, "Selection score differs")
        result["policy_graph_hashes"][policy] = graph_audit["directed_edge_sha256"]
        result["_optional"][policy] = optional.copy()
        predicted = output(arrays, "{key}__" + policy, n)
        if policy == "base":
            for key in predicted:
                audit.array(predicted[key], arrays[f"supplied_parity_{key}"], "Base parity-output reuse differs")
        audit.equal(case["reused_native_parity_output"], policy == "base", "Parity reuse flag differs")
        for name in ("graph_build_and_selection_seconds", "forward_operational_seconds", "scoring_seconds"):
            audit.check(finite(case[name]) and case[name] >= 0, "Invalid measured observed component time")
            put(policy, name, case[name])
        audit.equal(case["network_passes_for_standalone_policy"], 2 if score is not None else 1, "Observed policy pass count differs")
        audit.equal(case["scoring_seconds"], prior["operational_seconds"] if score is not None else 0., "Scoring time reuse differs")
        put(policy, "network_passes_for_standalone_policy", case["network_passes_for_standalone_policy"])
        for name in OBSERVED_METRICS:
            if name in graph_audit:
                put(policy, name, graph_audit[name])
        if np.isfinite(predicted["prediction"]).all():
            boundary = full.boundary_metrics(predicted["prediction"], bounds)
            audit.equal(case["predicted_boundary"], boundary, "Predicted boundary differs")
            for field in BOUNDARIES:
                put(policy, "predicted_boundary_" + field, boundary[field])
        if case["status"] == "failed":
            audit.equal(case["failure"]["phase"], "forward", "Unexpected action failure phase")
            reproduce_guard(case["failure"], lambda: bridge.validate_output(predicted, n), audit)
            audit.equal(case["metrics"], None, "Failed action has favorable metric")
            continue
        bridge.validate_output(predicted, n)
        residual = predicted["prediction"].astype(np.float64) - target
        normalized = residual / scale
        se = np.mean(normalized ** 2, axis=-1)
        for key, expected in (("position_residual", residual), ("normalized_residual", normalized), ("normalized_coordinate_se", se)):
            audit.array(arrays[f"{key}__{policy}"], expected, "Saved residual/particle error differs")
        for name, value in (("position_coordinate_mse", float(np.mean(residual ** 2))),
                            ("normalized_coordinate_mse", float(np.mean(normalized ** 2)))):
            audit.scalar(case["metrics"][name], value, "Saved frame MSE differs")
            put(policy, name, value)
        good[policy] = se
    failures = [{"case": p, "failure": row["cases"][p]["failure"]} for p in evaluation.POLICIES if row["cases"][p]["status"] == "failed"]
    audit.equal(row["failure"], {"category": "case_failures", "cases": failures} if failures else None, "Frame failure ledger differs")
    expected_correlations = {}
    for policy in evaluation.POLICIES:
        if "base" not in good or policy not in good:
            audit.equal(row["benefits"][policy], {"mean_signed_normalized_coordinate_gain": None,
                        "reason": "required base or action output failed"}, "Failed-action gain is not null")
            continue
        gain = good["base"] - good[policy]
        audit.array(arrays[f"signed_normalized_coordinate_gain__{policy}"], gain, "Actual-action signed gain differs")
        for name, value in (("mean_signed_normalized_coordinate_gain", float(gain.mean())), ("harmful_particle_fraction", float(np.mean(gain < 0)))):
            audit.scalar(row["benefits"][policy][name], value, "Gain/harm scalar differs")
            put(policy, name, value)
        if prior["status"] == "complete":
            correlation = bridge.same.spearman(score_output["risk"], gain)
            expected_correlations[policy] = correlation
            put(policy, "previous_risk_gain_spearman", correlation["value"], correlation["reason"])
    audit.equal(row["previous_risk_correlations"], expected_correlations, "Risk/action correlations differ")
    correlation = bridge.same.spearman(score_output["risk"], good["base"]) if prior["status"] == "complete" and "base" in good else {
        "value": None, "reason": "required scoring or base output failed"}
    audit.equal(row["previous_risk_base_residual_correlation"], correlation, "Risk/residual correlation differs")
    for policy in evaluation.POLICIES:
        put(policy, "previous_risk_base_residual_spearman", correlation["value"], correlation["reason"])
    result["risk_random_overlap"] = overlap(result["_optional"].get(evaluation.POLICIES[-1]), result["_optional"].get("random25"))
    return result


def committed_job(directory, mode, audit):
    protocol, result, status = [read_json(Path(directory) / name) for name in ("protocol.json", "result.json", "status.json")]
    digest = full.sha256(Path(directory) / "protocol.json")
    audit.equal(status.get("state"), "complete", "Job completion is uncommitted")
    audit.equal(result.get("state"), "complete", "Job result is incomplete")
    audit.equal(status.get("result_sha256"), full.sha256(Path(directory) / "result.json"), "Committed result hash differs")
    audit.equal(result.get("protocol_sha256"), digest, "Result protocol hash differs")
    for key in ("scope", "objective", "seed", "checkpoint_sha256"):
        audit.equal(result.get(key), protocol[key], "Result identity differs: " + key)
    if mode == "observed":
        for key in ("arm", "parent_checkpoint_sha256"):
            audit.equal(result.get(key), protocol[key], "Observed result lineage differs")
        for key in ("arm", "seed"):
            audit.equal(status.get(key), protocol[key], "Observed committed status lineage differs")
        audit.equal(status.get("committed_frames"), 425, "Observed committed frame count differs")
    else:
        identity = read_json(Path(directory) / "lineage_identity.json")
        audit.equal(identity, {key: protocol[key] for key in IDENTITY_FIELDS}, "Autonomous lineage identity differs")
        audit.check(finite(result["invocation_wall_seconds"]) and result["invocation_wall_seconds"] >= 0,
                    "Invalid autonomous invocation duration")
    return protocol, result, digest


def load_observed(directory, protocol, result, protocol_hash, expected, sources, manifests, normalization, audit):
    audit.equal(len(expected), 425, "Exactly 425 observed histories required")
    audit.equal(sum(item["split"] == "valid" for item in expected), 128, "Exactly 128 validation histories required")
    audit.equal(sum(item["split"] == "test" for item in expected), 297, "Exactly 297 test histories required")
    verified_pins = {str((Path(directory) / name).resolve()): full.sha256(Path(directory) / name)
                     for name in ("protocol.json", "result.json", "status.json")}
    expected_lookup = {bridge.stem(item): item for item in expected}
    actual = {bridge.stem(row): row for row in result["records"]}
    audit.equal(len(actual), len(result["records"]), "Duplicate observed records")
    audit.equal(set(actual), set(expected_lookup), "Missing/unknown observed records")
    audit.equal(result["required_frames"], 425, "Observed required population differs")
    files = {"protocol.json", "result.json", "status.json"}
    rows = []
    for key, item in expected_lookup.items():
        compact = actual[key]
        audit.check(type(compact["source_index"]) is int and type(compact["target_frame"]) is int, "Integer observed identity required")
        audit.equal(compact["record_file"], key + ".json", "Unsafe/renamed observed record")
        audit.equal(compact["array_file"], key + ".npz", "Unsafe/renamed observed array")
        path, array_path = Path(directory) / compact["record_file"], Path(directory) / compact["array_file"]
        audit.equal(full.sha256(path), compact["record_sha256"], "Observed raw record hash differs")
        row = read_json(path)
        require(all(type(row[key]) is int for key in ("source_index", "target_frame", "original_seed")),
                "Integer raw observed identity required")
        for name, value in item.items():
            audit.equal(row[name], value, "Observed identity differs")
        audit.equal(row["protocol_sha256"], protocol_hash, "Observed raw protocol differs")
        for name, expected_value in (("arm", protocol["arm"]), ("original_seed", protocol["seed"]),
            ("checkpoint_sha256", protocol["checkpoint_sha256"]), ("parent_checkpoint_sha256", protocol["parent_checkpoint_sha256"])):
            audit.equal(row.get(name), expected_value, "Observed row lineage differs")
        audit.equal(row["random_seed_material"], evaluation.random_material(protocol["seed"], item), "Observed random material differs")
        audit.equal(bridge.record_index(row, path), compact, "Observed compact/raw index differs")
        audit.equal(full.sha256(array_path), row["array_sha256"], "Observed array hash differs")
        verified_pins[str(path.resolve())] = compact["record_sha256"]
        verified_pins[str(array_path.resolve())] = row["array_sha256"]
        with np.load(array_path, allow_pickle=False) as arrays:
            positions, types = sources[item["split"]][item["source_index"]]
            observed_audit.audit_official_frame(item, arrays, positions, types, manifests[item["split"]]["records"][item["source_index"]], audit)
            audit.array(arrays["acceleration_std"], normalization["std"], "Endpoint normalization differs")
            audit.array(arrays["bounds"], normalization["bounds"], "Endpoint bounds differ")
            frame = analyze_observed_frame(row, arrays, audit)
        rows.append({**frame, "arm": protocol["arm"], "seed": protocol["seed"], "record_file": str(path.resolve()),
                     "record_sha256": compact["record_sha256"], "array_sha256": row["array_sha256"]})
        files.update((path.name, array_path.name))
    audit.equal(result["complete_frames"], sum(row["status"] == "complete" for row in rows), "Complete-frame count differs")
    audit.equal(result["failed_frames"], sum(row["status"] == "failed" for row in rows), "Failed-frame count differs")
    actual_pins = inventory(directory, files)
    audit.check(all(actual_pins[path] == digest for path, digest in verified_pins.items()), "Observed bytes changed during audit")
    return rows, actual_pins


def autonomous_values(row):
    values = {name: native_audit.metric(row, name) for name in native_audit.METRICS}
    values.update({f"mse_at_{step}": row["mse_at_steps"][str(step)] for step in scalar_audit.TRACE_STEPS})
    values.update(completed_steps=row["completed_steps"], forecast_network_passes=row["forecast_network_passes"])
    require(set(values) == set(AUTONOMOUS_METRICS), "Fixed autonomous metric coverage differs")
    require(all(value is None or finite(value) for value in values.values()), "Finite autonomous metric or scientific null required")
    return values


def load_autonomous(directory, protocol, result, protocol_hash, trajectories, manifest, normalization, audit):
    verified_pins = {str((Path(directory) / name).resolve()): full.sha256(Path(directory) / name)
                     for name in ("protocol.json", "result.json", "status.json", "lineage_identity.json")}
    expected = {(index, policy) for index in range(3, 30) for policy in native.POLICIES}
    actual = {(row["source_index"], row["policy"]) for row in result["records"]}
    audit.equal(actual, expected, "Missing/unknown autonomous units")
    audit.equal(len(result["records"]), 135, "Duplicate autonomous units")
    # Structural scalar validator input only: it contains no experiment scope,
    # training schema, original checkpoint identity, or 100k relabeling.
    structural = {"trajectory_records": manifest["records"][3:], "seed": protocol["seed"],
                  "guards": {"max_absolute_coordinate": protocol["guards"]["max_abs_coordinate"]}}
    files, rows = {"protocol.json", "result.json", "status.json", "lineage_identity.json"}, []
    for compact in result["records"]:
        require(type(compact["source_index"]) is int, "Integer autonomous identity required")
        verified = scalar_audit.validate_record(Path(directory), compact, structural, protocol_hash)
        path, trace_path = Path(directory) / compact["record_file"], Path(directory) / compact["trace_file"]
        row = read_json(path)
        require(type(row["source_index"]) is int, "Integer raw autonomous source index required")
        verified_pins[str(path.resolve())] = compact["record_sha256"]
        verified_pins[str(trace_path.resolve())] = compact["trace_sha256"]
        positions, types = trajectories[row["source_index"]]
        types = np.asarray(types, dtype=np.int64)
        if types.ndim == 0:
            types = np.full(positions.shape[1], types, dtype=np.int64)
        with np.load(trace_path, allow_pickle=False) as traces:
            audit.array(traces["bounds"], normalization["bounds"], "Autonomous endpoint bounds differ")
            metrics, prefix = native_audit.validate_trajectory(row, traces, positions, types, audit)
            initial_hash = full.state_hash(traces["initial_observed_positions"])
        verified.update(native_metrics=metrics)
        values = autonomous_values(verified)
        rows.append({**verified, "arm": protocol["arm"], "seed": protocol["seed"],
            "unit_id": str(row["source_index"]), "metrics": values,
            "null_reasons": {key: "required full horizon or fixed horizon not completed" if value is None else None for key, value in values.items()},
            "native_prefix_diagnostics_if_failed": prefix, "initial_observed_state_sha256": initial_hash,
            "native_parity": row["native_parity"], "record_file": str(path.resolve()), "raw_record": row})
        files.update((path.name, trace_path.name))
    actual_pins = inventory(directory, files)
    audit.check(all(actual_pins[path] == digest for path, digest in verified_pins.items()), "Autonomous bytes changed during audit")
    return rows, actual_pins


def audit_pairing(observed_rows, autonomous_rows, audit):
    lookup = {(row["arm"], row["seed"], row["split"], row["unit_id"]): row for row in observed_rows}
    pairs = []
    for key, left in lookup.items():
        arm, seed, split, unit = key
        if arm != "base":
            continue
        right = lookup["mix", seed, split, unit]
        audit.equal(left["trajectory_id"], right["trajectory_id"], "Paired observed trajectory differs")
        for field in ("current_history", "previous_history", "particle_types", "acceleration_std", "bounds"):
            audit.equal(left["pairing_hashes"][field], right["pairing_hashes"][field], "Paired immutable observed inputs differ")
        for field in ("target_position", "strict_base_pairs", "strict_annulus_pairs"):
            a, b = left["pairing_hashes"][field], right["pairing_hashes"][field]
            if a is not None and b is not None:
                audit.equal(a, b, "Paired observed geometry/target differs")
        for policy in evaluation.POLICIES[:-1]:
            a, b = left["policy_graph_hashes"].get(policy), right["policy_graph_hashes"].get(policy)
            if a is not None and b is not None:
                audit.equal(a, b, "Paired base/dense/random/speed graph differs")
        pairs.append({"seed": seed, "split": split, "trajectory_id": left["trajectory_id"], "unit_id": unit,
            "risk_overlap_between_arms": overlap(left["_optional"].get(evaluation.POLICIES[-1]), right["_optional"].get(evaluation.POLICIES[-1])),
            "base_arm_risk_random_overlap": left.get("risk_random_overlap"), "mix_arm_risk_random_overlap": right.get("risk_random_overlap")})
    for row in observed_rows:
        row.pop("_optional")
    initial = {}
    for row in autonomous_rows:
        key = row["seed"], row["source_index"]
        if key in initial:
            audit.equal(row["initial_observed_state_sha256"], initial[key], "Autonomous paired initial observed states differ")
        initial[key] = row["initial_observed_state_sha256"]
    return pairs


def write_compressed(path, value):
    """Fresh deterministic gzip; all raw/failure rows remain losslessly stored."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with Path(path).open("xb") as stream:
        stream.write(gzip.compress(data, mtime=0))
    return {"path": Path(path).name, "sha256": full.sha256(path), "uncompressed_json_sha256": scalar_audit.digest_bytes(data)}


def publish_population(directory, name, rows, expected_units, policies, metric_names):
    index = {}
    for metric in metric_names:
        if name.startswith("observed"):
            records = [{"arm": row["arm"], "seed": row["seed"], "trajectory_id": row["trajectory_id"], "unit_id": row["unit_id"],
                "policy_values": {p: row["metrics"][p][metric] for p in policies},
                "policy_null_reasons": {p: row["null_reasons"][p][metric] for p in policies}} for row in rows]
        else:
            grouped = {}
            for row in rows:
                key = row["arm"], row["seed"], row["trajectory_id"], row["unit_id"]
                unit = grouped.setdefault(key, {"arm": key[0], "seed": key[1], "trajectory_id": key[2], "unit_id": key[3],
                    "policy_values": {}, "policy_null_reasons": {}})
                require(row["policy"] not in unit["policy_values"], "Duplicate autonomous policy slot")
                unit["policy_values"][row["policy"]] = row["metrics"][metric]
                unit["policy_null_reasons"][row["policy"]] = row["null_reasons"][metric]
            records = list(grouped.values())
        result = aggregate_metric(records, expected_units, policies)
        record = write_compressed(Path(directory) / f"{name}__{metric}.json.gz", result)
        compact = lambda summary: {key: value for key, value in summary.items() if key != "seeds"}
        record.update(absolute={arm: {p: compact(result["absolute"][arm][p]) for p in policies} for arm in ("base", "mix")},
                      paired={key: compact(value) for key, value in result["paired"].items()})
        index[metric] = record
    return {"required_units_per_endpoint": len(expected_units), "policies": list(policies), "metrics": index}


def source_pins():
    result = {str((full.REPO / relative).resolve()): digest for relative, digest in FROZEN.items()}
    evaluation.verify_files(result)
    for path in (Path(__file__), PROTOCOL):
        result[str(path.resolve())] = full.sha256(path)
    for path in sorted((full.REPO / "research/tests").glob("test_*continuation*summary*.py")):
        result[str(path.resolve())] = full.sha256(path)
    return result


def analyze(args):
    started = time.perf_counter()
    directory = Path(args.output_dir).resolve()
    require(not directory.exists(), "Fresh analysis directory required; preserve prior attempts")
    protected = [Path(value).resolve() for value in (args.observed_root, args.autonomous_root)]
    protected += [Path(value).resolve().parent for value in (args.cohort_manifest, args.validation_manifest, args.test_manifest)]
    require(not any(directory == path or path in directory.parents or directory in path.parents for path in protected),
            "Separate analysis output outside all input trees required")
    # Read only the explicit manifest path metadata before creating output.
    # A rejected destination must never create even a failure-status file
    # inside an immutable endpoint tree. Full CPU lineage admission follows.
    preliminary_entries = evaluation.validate_cohort_manifest(read_json(args.cohort_manifest), Path(args.cohort_manifest).parent)
    require(not any(directory == Path(row["checkpoint"]).parent or Path(row["checkpoint"]).parent in directory.parents
                    or directory in Path(row["checkpoint"]).parent.parents for row in preliminary_entries.values()),
            "Analysis output must be separate from every endpoint checkpoint tree")
    require(datetime.now(timezone.utc) < datetime.fromisoformat(native.DEADLINE), "Research cutoff reached; no new analysis")
    directory.mkdir(parents=True)
    audit, pins = Audit(), {}
    try:
        evaluation.merge_pins(pins, source_pins())
        metadata_path = Path(args.validation_manifest).parent / "metadata.json"
        metadata = read_json(metadata_path)
        # The frozen gate loads checkpoint tensors onto CPU and does not build
        # a simulator. It verifies all parent/Adam/config/schedule lineages.
        entries, configs, admitted = evaluation.load_verified_cohort(args.cohort_manifest, full.sha256(metadata_path))
        audit.equal(entries, preliminary_entries, "Cohort manifest changed after output isolation check")
        require(all(type(config["seed"]) is int and type(config["parent_completed_updates"]) is int
                    and type(config["additional_updates"]) is int for config in configs.values()),
                "Integer continuation configuration identity/update counters required")
        require(not any(directory == Path(row["checkpoint"]).parent or Path(row["checkpoint"]).parent in directory.parents
                        or directory in Path(row["checkpoint"]).parent.parents for row in entries.values()),
                "Analysis output must be separate from every endpoint checkpoint tree")
        evaluation.merge_pins(pins, admitted)
        expected, valid, test = bridge.expected_frames(configs["base", 0], args.validation_manifest, args.test_manifest)
        audit.equal(valid["metadata"], metadata, "Validation metadata differs")
        audit.equal(test["metadata"], metadata, "Test metadata differs")
        audit.equal(valid["source"]["sha256"], scalar_audit.OFFICIAL_VALID_SHA256, "Official validation source differs")
        audit.equal(test["source"]["sha256"], scalar_audit.OFFICIAL_TEST_SHA256, "Official test source differs")
        data_pins = {}
        for manifest_path, manifest in ((Path(args.validation_manifest), valid), (Path(args.test_manifest), test)):
            for path in (manifest_path, manifest_path.parent / "metadata.json"):
                data_pins[str(path.resolve())] = full.sha256(path)
            for record in manifest["records"]:
                for key in ("positions", "particle_types"):
                    path = manifest_path.parent / record[key]["path"]
                    require(path.resolve().is_relative_to(manifest_path.parent.resolve()), "Manifest data path escaped source tree")
                    data_pins[str(path.resolve())] = record[key]["sha256"]
        evaluation.merge_pins(pins, data_pins)
        evaluation.verify_files(data_pins)
        sources = {"valid": full.load_manifest_data(args.validation_manifest, verify_hashes=True),
                   "test": full.load_manifest_data(args.test_manifest, verify_hashes=True)}
        jobs = {mode: discover_jobs(root, mode) for mode, root in (("observed", args.observed_root), ("autonomous", args.autonomous_root))}
        all_observed, all_autonomous, job_rows, common_hash = [], [], [], None
        required = {**admitted, **data_pins}
        # The frozen evaluator explicitly pins the validation metadata file;
        # its numeric test loader also verifies a potentially separate test
        # metadata file. Pin that file here, without inventing an inference pin.
        test_metadata = (Path(args.test_manifest).parent / "metadata.json").resolve()
        if test_metadata != metadata_path.resolve():
            required.pop(str(test_metadata), None)
        prepared = {}
        for identity in sorted(evaluation.COHORT):
            for mode in ("observed", "autonomous"):
                job = jobs[mode][identity]
                protocol, result, protocol_hash = committed_job(job, mode, audit)
                contract = validate_job_protocol(protocol, mode, entries[identity], configs[identity], entries, expected, test, required, audit)
                if common_hash is not None:
                    audit.equal(contract, common_hash, "Common graph/data/software/runtime contract differs across twelve jobs")
                common_hash = contract
                evaluation.merge_pins(pins, protocol["input_files_sha256"])
                for name in ("protocol.json", "result.json", "status.json"):
                    evaluation.merge_pins(pins, {str((job / name).resolve()): full.sha256(job / name)})
                prepared[mode, identity] = protocol, result, protocol_hash
        # Verify the union once before raw-array auditing rather than hashing
        # the large common training corpus twelve times.
        evaluation.verify_files(pins)
        for identity in sorted(evaluation.COHORT):
            entry, config = entries[identity], configs[identity]
            payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
            simulator_config = payload["simulator_config"]
            audit.equal(evaluation.support.simulator_config_sha256(simulator_config), config["parent_simulator_config_sha256"],
                        "Endpoint simulator/normalization hash differs")
            normalization = {"std": simulator_config["normalization_stats"]["acceleration"]["std"].numpy().astype(np.float64),
                             "bounds": np.asarray(simulator_config["boundaries"], dtype=np.float64)}
            del payload
            for mode in ("observed", "autonomous"):
                job = jobs[mode][identity]
                protocol, result, protocol_hash = prepared[mode, identity]
                audit.equal(protocol["checkpoint_provenance"]["variance_floor"], simulator_config["variance_floor"], "Endpoint variance floor differs")
                if mode == "observed":
                    rows, raw_pins = load_observed(job, protocol, result, protocol_hash, expected, sources,
                                                  {"valid": valid, "test": test}, normalization, audit)
                    all_observed.extend(rows)
                else:
                    rows, raw_pins = load_autonomous(job, protocol, result, protocol_hash, sources["test"], test, normalization, audit)
                    all_autonomous.extend(rows)
                evaluation.merge_pins(pins, raw_pins)
                job_rows.append({"mode": mode, "arm": identity[0], "seed": identity[1], "directory": str(job.resolve()),
                    "protocol_sha256": protocol_hash, "records": len(rows), "failed_records": sum(row["status"] == "failed" for row in rows),
                    "result_sha256": full.sha256(job / "result.json"), "invocation_wall_seconds": result.get("invocation_wall_seconds"),
                    "note": "Invocation duration covers only final invocation when prior statuses are archived; no inferred total."})
        audit.equal(len(all_observed), 2550, "Complete observed population required")
        audit.equal(len(all_autonomous), 810, "Complete autonomous population required")
        pairs = audit_pairing(all_observed, all_autonomous, audit)
        evaluation.verify_files(pins)
        populations = {}
        for split in ("valid", "test"):
            units = [{**item, "unit_id": f"{item['source_index']}:{item['target_frame']}"} for item in expected if item["split"] == split]
            rows = [row for row in all_observed if row["split"] == split]
            populations["observed_" + split] = publish_population(directory, "observed_" + split, rows, units, evaluation.POLICIES, OBSERVED_METRICS)
        units = [{"source_index": i, "trajectory_id": test["records"][i]["id"], "unit_id": str(i)} for i in range(3, 30)]
        populations["autonomous_test"] = publish_population(directory, "autonomous_test", all_autonomous, units, native.POLICIES, AUTONOMOUS_METRICS)
        detail = write_compressed(directory / "audited_records.json.gz", {"observed": all_observed, "autonomous": all_autonomous, "pairing": pairs})
        evaluation.verify_files(pins)
        report = {"schema": 1, "state": "complete", "scope": SCOPE, "created_utc": datetime.now(timezone.utc).isoformat(),
            "completed_total_updates": 110000, "completed_additional_updates": 10000, "cohort": list(entries.values()),
            "coverage": {"endpoints": 6, "jobs": 12, "observed_frames": 2550, "observed_policy_slots": 12750, "autonomous_outcomes": 810},
            "jobs": job_rows, "populations": populations, "audited_records": detail, "input_files_sha256": pins,
            "source_and_input_reverified_after_analysis": True, "audit_checks": audit.checks,
            "analysis_wall_seconds": time.perf_counter() - started, "common_contract_sha256": common_hash,
            "interpretation": ["Exploratory conditional on three faithful100k parents and fixed paired continuation schedules; no fresh-training or NLL generality.",
                "Observed histories and autonomous own-prediction geometries are distinct populations. Tests were inspected before this follow-up.",
                "Arm means and paired contrasts use equal trajectories within each seed; all three seeds required; any required null propagates.",
                "Negative error/failure arm effects favor mix; negative risk-random interaction favors relative risk allocation after mix exposure.",
                "Positive signed gain favors the action; correlation direction is associative. Edge savings and operational timing are not causal speedups.",
                "Prediction, truth and observed boundary diagnostics are distinct; numerical completion does not establish physical validity.",
                "Raw rejected attempts and all unsuccessful outcomes remain in input artifacts and the lossless audited record archive.",
                "Observed base forward reuses parity; previous graph is untimed. No exact standalone policy latency is reconstructed."]}
        full.atomic_json(directory / "result.json", report)
        full.atomic_json(directory / "status.json", {"state": "complete", "result_sha256": full.sha256(directory / "result.json")})
        return report
    except BaseException as error:
        full.atomic_json(directory / "status.json", {"state": "invalid_or_incomplete", "scope": SCOPE,
            "error_type": type(error).__name__, "error": str(error), "audit_checks": audit.checks,
            "input_files_sha256_before_failure": pins, "analysis_wall_seconds": time.perf_counter() - started,
            "instruction": "Preserve this attempt and all input outcomes; no partial means are eligible."})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort-manifest", "validation-manifest", "test-manifest", "observed-root", "autonomous-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = analyze(args)
    print(json.dumps({"state": report["state"], "coverage": report["coverage"], "audit_checks": report["audit_checks"]}))


if __name__ == "__main__":
    main()
