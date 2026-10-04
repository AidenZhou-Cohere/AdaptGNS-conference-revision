"""Strict six-model aggregation of the prospective same-state diagnostic.

Reads JSON and hashes arrays only. Does not load models, parse particle arrays,
or access source datasets. Runtime repetitions remain within-frame measurements;
the independent model axis is exactly the three requested training seeds.
"""
import argparse
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import statistics
import sys

from research import summarize_full_rollouts as shared

OBJECTIVES, SEEDS = shared.OBJECTIVES, shared.SEEDS
SOURCE_INDICES = tuple(range(3, 30))
TARGET_FRAMES = (7, 106, 205, 304, 403, 502, 601, 700, 799, 898, 1000)
POLICIES = ("base", "dense", "random25", "speed25", "previous-observed-base-risk25")
REFERENCE = "base_shared_superset"
CASES = POLICIES + (REFERENCE,)
CORRELATIONS = ("previous_risk_vs_base_error", "previous_risk_vs_dense_benefit",
                "current_base_risk_vs_base_error", "current_base_risk_vs_dense_benefit")
TIMING_FIELDS = ("end_to_end_seconds", "score_generation_seconds", "score_graph_seconds", "score_forward_seconds",
                 "current_graph_and_selection_seconds", "current_forward_seconds")
BENEFIT_FIELDS = ("mean_normalized_vector_benefit", "mean_position_vector_benefit", "positive_fraction", "negative_fraction", "zero_fraction")
ACCURACY_FIELDS = ("position_coordinate_mse", "normalized_coordinate_mse")
COMPACT_FIELDS = ("source_index", "target_frame", "trajectory_id", "n_particles", "status", "failure", "accuracy", "benefit",
                  "correlations", "timing_summary", "graph_audit", "array_file", "array_sha256")
EXPECTED_KEYS = tuple(itertools.product(SOURCE_INDICES, TARGET_FRAMES))
WEIGHTED_METRICS = tuple(f"accuracy/{policy}/{field}" for policy in POLICIES for field in ACCURACY_FIELDS) + tuple(
    f"benefit/{field}" for field in BENEFIT_FIELDS)
METRICS = WEIGHTED_METRICS + tuple(f"correlation/{field}" for field in CORRELATIONS) + tuple(
    f"timing/{case}/{field}" for case in CASES for field in TIMING_FIELDS) + ("warmup/total_seconds", "failure_fraction")
POLICY_COMPARISONS = tuple((policy, "base") for policy in POLICIES[1:]) + tuple(
    (POLICIES[-1], policy) for policy in ("dense", "random25", "speed25"))
require, sha256, read_json = shared.require, shared.sha256, shared.read_json


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected):
    return finite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-15)


def benefit_consistent(actual, base_coordinate_mse, dense_coordinate_mse, particles):
    """Account for cancellation between two large nonnegative vector errors.

    The evaluator averages per-particle differences; the check subtracts two
    independently reduced means. Four eps64*n times their summed vector-error
    magnitudes is a conservative linear-summation roundoff envelope. It only
    validates this identity; stored benefits and other tolerances are unchanged.
    """
    expected = 2 * (base_coordinate_mse - dense_coordinate_mse)
    magnitude = 2 * (abs(base_coordinate_mse) + abs(dense_coordinate_mse))
    roundoff = 4 * sys.float_info.epsilon * max(particles, 1) * magnitude
    return finite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=max(1e-15, roundoff))


def validate_protocol(protocol, objective, seed, training_hash, companion_hash, evaluator_hash):
    require(protocol.get("schema") == 1 and protocol.get("scope") == "locked_test" and protocol.get("split") == "test",
            "Expected schema-1 locked same-state diagnostic")
    require(protocol.get("objective") == objective and type(protocol.get("seed")) is int and protocol["seed"] == seed,
            "Diagnostic objective/seed differs from expected run identity")
    require(protocol.get("target_frames") == list(TARGET_FRAMES) and protocol.get("policies") == list(POLICIES)
            and protocol.get("timing_cases") == list(CASES) and protocol.get("timed_rounds") == 6
            and protocol.get("warmup_rounds") == 1, "Diagnostic schedule/policies/timing rounds differ")
    require(protocol.get("training_protocol_sha256") == training_hash and protocol.get("companion_protocol_sha256") == companion_hash,
            "Training or companion protocol SHA256 differs")
    source = protocol["source_tfrecord"]
    require(source.get("sha256") == shared.OFFICIAL_TEST_SHA256 and source.get("record_count") == 30
            and source.get("CRC_verified") is True, "Official complete 30-record test source identity differs")
    records = protocol["trajectory_records"]
    require([row.get("source_index") for row in records] == list(SOURCE_INDICES)
            and len({row["id"] for row in records}) == 27, "Required trajectory identities are source indices 3 through 29")
    for row in records:
        shape = row["positions"]["shape"]
        require(len(shape) == 3 and shape[0] == 1001 and type(shape[1]) is int and shape[1] > 0 and shape[2] == 2,
                "Expected 1001-frame two-dimensional trajectory")
        require(all(shared.valid_hash(row[name]["sha256"]) for name in ("positions", "particle_types"))
                and shared.valid_hash(row["trajectory_content_sha256"]), "Missing trajectory content hashes")
    require(all(shared.valid_hash(protocol.get(name)) for name in ("checkpoint_sha256", "manifest_sha256", "metadata_sha256")),
            "Missing checkpoint/data hashes")
    require(protocol["guards"] == {"max_candidate_pairs": 100000, "max_absolute_coordinate": 10.}, "Diagnostic guards differ")
    provenance = protocol["checkpoint_provenance"]
    training = provenance["training_config"]
    run = training["full_run"]
    require(provenance.get("checkpoint_format") == 2 and provenance.get("normalization_source") == "checkpoint"
            and provenance.get("uncertainty_parameterization") == "variance" and provenance.get("connectivity_radius") == .015
            and provenance.get("nmessage_passing_steps") == 10, "Checkpoint architecture/normalization provenance differs")
    require(training.get("loss") == objective and training.get("completed_optimizer_updates") == 100000
            and run.get("objective") == objective and type(run.get("seed")) is int and run["seed"] == seed
            and run.get("scope") == "bounded_full_data_100k" and run.get("steps") == 100000
            and run.get("batch_size") == 2 and run.get("history") == 6 and run.get("noise_std") == 6.7e-4
            and run.get("architecture") == {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
            "Checkpoint is not the requested final 100k model")
    require(run.get("metadata_sha256") == protocol["metadata_sha256"] and run.get("research_protocol_sha256") == training_hash,
            "Checkpoint metadata/training protocol differs from diagnostic")
    for split, count, digest in (("train", 1000, shared.OFFICIAL_TRAIN_SHA256), ("valid", 30, shared.OFFICIAL_VALID_SHA256)):
        require(run[split].get("n_trajectories") == count and run[split]["source"].get("sha256") == digest
                and shared.valid_hash(run[split].get("manifest_sha256")), f"Checkpoint {split} provenance differs")
    sources, training_sources = protocol["code_sha256"], run["source_sha256"]
    require(sources.get("research/full_same_state.py") == evaluator_hash and bool(training_sources)
            and all(shared.valid_hash(value) for value in itertools.chain(sources.values(), training_sources.values())),
            "Diagnostic source differs from the reviewed evaluator or source hashes are missing")
    require(all(sources[name] == value for name, value in training_sources.items() if name in sources),
            "Shared training/diagnostic code changed")
    common = {"diagnostic": {key: value for key, value in protocol.items()
                              if key not in ("objective", "seed", "checkpoint_sha256", "checkpoint_provenance")},
              "checkpoint": {key: value for key, value in provenance.items() if key != "training_config"},
              "training": {key: value for key, value in run.items() if key not in ("objective", "seed")},
              "training_metadata": {key: value for key, value in training.items() if key not in ("loss", "full_run")}}
    return shared.canonical_hash(common)


def expected_order(source_index, target, round_index):
    start = (source_index * 11 + TARGET_FRAMES.index(target) + max(round_index, 0)) % 6
    return CASES[start:] + CASES[:start]


def validate_calls(row):
    """Recompute six-repeat descriptive means; never use repeats as seeds."""
    require(len(row["warmup_calls"]) == 6 and len(row["timed_calls"]) == 36, "Complete frame is missing required calls")
    for warmup, calls in ((True, row["warmup_calls"]), (False, row["timed_calls"])):
        expected = [(round_index, slot, case) for round_index in ([-1] if warmup else range(6))
                    for slot, case in enumerate(expected_order(row["source_index"], row["target_frame"], round_index))]
        actual = [(call["round"], call["slot"], call["method"]) for call in calls]
        require(actual == expected, "Recorded timing order is not the fixed balanced rotation")
        for call in calls:
            risk = call["method"] == POLICIES[-1]
            require(call.get("status") == "complete" and call.get("warmup") is warmup
                    and call.get("network_passes") == 1 + int(risk) and call.get("candidate_builds") == 1 + int(risk),
                    "Call status/score-generation accounting differs")
            require(shared.valid_hash(call.get("pair_sha256")) and all(shared.finite_nonnegative(call.get(name)) for name in TIMING_FIELDS),
                    "Invalid timing value or pair hash")
            require(call["end_to_end_seconds"] + 1e-12 >= call["score_generation_seconds"] + call["current_graph_and_selection_seconds"] + call["current_forward_seconds"],
                    "End-to-end timing omits a required phase")
            if risk:
                require(call["score_generation_seconds"] + 1e-12 >= call["score_graph_seconds"] + call["score_forward_seconds"],
                        "Score generation timing omits graph construction or the previous forward pass")
            if not risk:
                require(all(call[name] == 0 for name in ("score_generation_seconds", "score_graph_seconds", "score_forward_seconds")),
                        "Unexpected score-generation pass for this case")
    require(row.get("completed_network_passes") == 49 and row.get("failed_call_attempted_network_passes") == 0,
            "Complete frame network-pass total differs")
    means = {}
    for case in CASES:
        calls = [call for call in row["timed_calls"] if call["method"] == case]
        require(row["repeat_consistency"][case]["distinct_pair_hashes"] == len({call["pair_sha256"] for call in calls})
                and shared.finite_nonnegative(row["repeat_consistency"][case]["maximum_absolute_prediction_difference"]),
                "Repeated allocation/prediction audit differs")
        for name in TIMING_FIELDS:
            values = [call[name] for call in calls]
            stats = row["timing_summary"][case][name]
            require(stats.get("observations") == 6 and stats.get("expected") == 6
                    and close(stats.get("mean"), statistics.fmean(values)) and close(stats.get("median"), statistics.median(values))
                    and close(stats.get("sample_sd"), statistics.stdev(values)), "Saved repeated-call timing statistics differ")
            means[f"timing/{case}/{name}"] = statistics.fmean(values)
    means["warmup/total_seconds"] = sum(call["end_to_end_seconds"] for call in row["warmup_calls"])
    return means


def validate_graph(row):
    audit = row["graph_audit"]
    base, annulus, budget = (audit[key] for key in ("base_pairs", "annulus_pairs", "optional_budget"))
    require(all(type(value) is int and value >= 0 for value in (base, annulus, budget))
            and audit["candidate_pairs"] == base + annulus and budget == math.floor(.25 * annulus)
            and audit.get("natural_shared_float64_base_identical") is True,
            "Candidate/base/optional budget audit differs")
    require(set(audit["policies"]) == set(CASES), "Missing allocation audit case")
    for case in CASES:
        selected = audit["policies"][case]
        expected = 0 if case in ("base", REFERENCE) else annulus if case == "dense" else budget
        require(selected["retained_optional_pairs"] == expected and selected["retained_pairs"] == base + expected
                and selected["directed_edges"] == 2 * (base + expected), "Retained pair count differs from exact policy budget")
    require(shared.finite_nonnegative(row["natural_shared_base_maximum_prediction_difference"]),
            "Invalid natural/shared prediction audit")
    return {"base_pairs": base, "annulus_pairs": annulus, "optional_budget": budget,
            "frozen_float32_comparison": audit["frozen_float32_comparison"]}


def validate_frame(directory, compact, protocol, protocol_hash):
    source, target = compact["source_index"], compact["target_frame"]
    require(type(source) is int and type(target) is int and (source, target) in EXPECTED_KEYS, "Unexpected diagnostic frame")
    stem = f"trajectory_{source:06d}_target_{target:04d}"
    require(compact.get("record_file") == stem + ".json" and compact.get("array_file") == stem + ".npz",
            "Unexpected diagnostic artifact path")
    row, digest = read_json(directory / (stem + ".json"))
    require(digest == compact.get("record_sha256") and set(compact) == set(COMPACT_FIELDS) | {"record_file", "record_sha256"}
            and all(row.get(key) == compact[key] for key in COMPACT_FIELDS), "Raw diagnostic record/hash differs from index")
    require(sha256(directory / (stem + ".npz")) == row["array_sha256"], "Diagnostic array SHA256 differs")
    description = protocol["trajectory_records"][source - 3]
    require(row.get("protocol_sha256") == protocol_hash and row["trajectory_id"] == description["id"]
            and row["n_particles"] == description["positions"]["shape"][1]
            and row.get("random_seed_material") == [93000, protocol["seed"], source, target]
            and row.get("model_history_dtype") == "float32" and row.get("radius_classification_dtype") == "float64",
            "Frame protocol, identity, random stream or numeric convention differs")
    scales = row["saved_acceleration_normalization"]
    require(set(scales) == {"mean", "std"} and len(scales["mean"]) == len(scales["std"]) == 2
            and all(finite(value) for value in scales["mean"]) and all(finite(value) and value > 0 for value in scales["std"]),
            "Invalid saved acceleration normalization")
    require(all(shared.valid_hash(row.get(key)) for key in ("observed_history_sha256", "previous_observed_history_sha256")),
            "Missing observed-history identity hashes")
    require((row["status"] == "complete" and row["failure"] is None)
            or (row["status"] == "failed" and isinstance(row["failure"], dict) and isinstance(row["failure"].get("category"), str)),
            "Frame completion/failure record is inconsistent")
    metrics = {name: None for name in METRICS}
    metrics["failure_fraction"] = float(row["status"] == "failed")
    common_input = {key: row[key] for key in ("trajectory_id", "n_particles", "observed_history_sha256", "previous_observed_history_sha256")}
    if row["status"] == "complete":
        require(shared.valid_hash(row.get("target_sha256")), "Missing target hash")
        common_input["target_sha256"] = row["target_sha256"]
        common_input["graph"] = validate_graph(row)
        require(set(row["accuracy"]) == set(POLICIES), "Missing policy accuracy")
        for policy in POLICIES:
            for field in ACCURACY_FIELDS:
                value = row["accuracy"][policy][field]
                require(shared.finite_nonnegative(value), "Nonfinite or negative error metric")
                metrics[f"accuracy/{policy}/{field}"] = value
        for field in BENEFIT_FIELDS:
            value = row["benefit"][field]
            require(finite(value) and (not field.endswith("fraction") or 0 <= value <= 1), "Invalid signed benefit metric")
            metrics[f"benefit/{field}"] = value
        require(close(sum(row["benefit"][name] for name in BENEFIT_FIELDS[2:]), 1.), "Benefit sign fractions do not sum to one")
        require(benefit_consistent(row["benefit"]["mean_position_vector_benefit"], row["accuracy"]["base"]["position_coordinate_mse"],
                                  row["accuracy"]["dense"]["position_coordinate_mse"], row["n_particles"])
                and benefit_consistent(row["benefit"]["mean_normalized_vector_benefit"], row["accuracy"]["base"]["normalized_coordinate_mse"],
                                       row["accuracy"]["dense"]["normalized_coordinate_mse"], row["n_particles"]),
                "Signed benefit differs from base-minus-dense vector squared errors")
        require(set(row["correlations"]) == set(CORRELATIONS), "Missing current/previous risk correlation")
        for field in CORRELATIONS:
            coefficient = row["correlations"][field]
            value = coefficient["value"]
            require(coefficient.get("particles") == row["n_particles"]
                    and ((value is None and coefficient.get("reason") in ("constant_rank_vector", "fewer_than_two_particles"))
                         or (finite(value) and -1 <= value <= 1 and coefficient.get("reason") is None)),
                    "Correlation is neither a valid coefficient nor an explicit undefined result")
            metrics[f"correlation/{field}"] = value
        metrics.update(validate_calls(row))
    else:
        require(all(row.get(key) is None for key in ("accuracy", "benefit", "correlations", "timing_summary")),
                "Failed frame supplies partial accuracy/correlation/timing aggregates")
    return {"source_index": source, "target_frame": target, "trajectory_id": row["trajectory_id"], "n_particles": row["n_particles"],
            "status": row["status"], "failure": row["failure"], "metrics": metrics, "common_input": common_input,
            "normalization_sha256": shared.canonical_hash(scales), "record_file": compact["record_file"], "record_sha256": digest,
            "array_file": compact["array_file"], "array_sha256": row["array_sha256"]}


def load_run(directory, objective, seed, training_hash, companion_hash, evaluator_hash):
    directory = Path(directory)
    output = {"objective": objective, "seed": seed, "directory": str(directory), "state": "missing", "errors": [],
              "frames": [], "eligible": False, "common_identity_sha256": None}
    if not directory.exists():
        return output
    try:
        require(directory.is_dir(), "Expected diagnostic directory")
        if not (directory / "protocol.json").exists() and not (directory / "result.json").exists():
            require(not any(directory.glob("trajectory_*")), "Frame artifacts exist without their protocol")
            output["state"] = "pending_evaluation"
            return output
        protocol, protocol_hash = read_json(directory / "protocol.json")
        identity = validate_protocol(protocol, objective, seed, training_hash, companion_hash, evaluator_hash)
        output.update(common_identity_sha256=identity, protocol_sha256=protocol_hash, checkpoint_sha256=protocol["checkpoint_sha256"])
        if not (directory / "result.json").exists():
            output["state"] = "pending_evaluation"
            return output
        saved, result_hash = read_json(directory / "result.json")
        require(saved.get("protocol_sha256") == protocol_hash and saved.get("state") in ("partial", "complete"), "Result protocol/state differs")
        frames, keys = [], set()
        for compact in saved["records"]:
            frame = validate_frame(directory, compact, protocol, protocol_hash)
            key = frame["source_index"], frame["target_frame"]
            require(key not in keys, "Duplicate diagnostic frame")
            frames.append(frame); keys.add(key)
        require(len({frame["normalization_sha256"] for frame in frames}) <= 1, "Saved normalization changed within a run")
        output.update(frames=frames, result_sha256=result_hash, evaluator_state=saved["state"],
                      missing_frames=[{"source_index": index, "target_frame": target} for index, target in EXPECTED_KEYS if (index, target) not in keys])
        status = read_json(directory / "status.json")[0] if (directory / "status.json").exists() else {}
        output["evaluator_status"] = status.get("state", "missing")
        if saved["state"] == "complete":
            require(keys == set(EXPECTED_KEYS), "Complete diagnostic result is missing required frames")
            require(saved.get("scope") == "locked_test" and saved.get("seed") == seed and saved.get("objective") == objective
                    and saved.get("checkpoint_sha256") == protocol["checkpoint_sha256"], "Final result identity differs")
            if status.get("state") != "complete":
                output.update(state="incomplete", errors=["Completion is not committed in status.json"])
                return output
            require(status.get("result_sha256") == result_hash, "Status/result SHA256 differs; snapshot may be changing")
            output.update(state="complete", eligible=True)
        else:
            output["state"] = "incomplete"
    except (ValueError, KeyError, TypeError, OSError, OverflowError, IndexError, AttributeError) as error:
        output.update(state="invalid", eligible=False, errors=[f"{type(error).__name__}: {error}"])
    return output


def equal_trajectory_mean(frames, values):
    if len(frames) != 297 or len(values) != 297 or any(value is None for value in values):
        return None
    by_source = {source: [] for source in SOURCE_INDICES}
    for frame, value in zip(frames, values):
        by_source[frame["source_index"]].append(value)
    if any(len(values) != 11 for values in by_source.values()):
        return None
    return statistics.fmean(statistics.fmean(values) for values in by_source.values())


def seed_summary(run):
    frames = sorted(run["frames"], key=lambda row: (row["source_index"], row["target_frame"]))
    present = {(row["source_index"], row["target_frame"]) for row in frames}
    metrics = {name: equal_trajectory_mean(frames, [row["metrics"][name] for row in frames]) if run["eligible"] else None for name in METRICS}
    weighted = {}
    for name in WEIGHTED_METRICS:
        weighted[name] = (sum(row["metrics"][name] * row["n_particles"] for row in frames) / sum(row["n_particles"] for row in frames)
                          if metrics[name] is not None else None)
    correlations = {name: {"defined_completed_frames": sum(row["status"] == "complete" and row["metrics"][f"correlation/{name}"] is not None for row in frames),
                            "undefined_completed_frames": sum(row["status"] == "complete" and row["metrics"][f"correlation/{name}"] is None for row in frames)}
                    for name in CORRELATIONS}
    return {"seed": run["seed"], "run_state": run["state"], "required_frames": 297, "observed_frames": len(frames),
            "failed_frames": sum(row["status"] == "failed" for row in frames),
            "missing_frames": [{"source_index": source, "target_frame": target} for source, target in EXPECTED_KEYS if (source, target) not in present],
            "equal_trajectory_metrics": metrics, "particle_weighted_error_and_benefit": weighted,
            "correlation_counts": correlations,
            "failures": [{key: row[key] for key in ("source_index", "target_frame", "trajectory_id", "failure")} for row in frames if row["status"] == "failed"]}


def paired_metric(left, right, left_name, right_name):
    a = {(row["source_index"], row["target_frame"]): row for row in left["frames"]}
    b = {(row["source_index"], row["target_frame"]): row for row in right["frames"]}
    if not left["eligible"] or not right["eligible"] or set(a) != set(EXPECTED_KEYS) or set(b) != set(EXPECTED_KEYS):
        return None
    frames, differences = [], []
    for key in EXPECTED_KEYS:
        frames.append(a[key])
        x, y = a[key]["metrics"][left_name], b[key]["metrics"][right_name]
        differences.append(x - y if x is not None and y is not None else None)
    return equal_trajectory_mean(frames, differences)


def paired_summary(left_runs, right_runs, fields):
    result = {}
    for label, left_name, right_name in fields:
        values = [paired_metric(left, right, left_name, right_name) for left, right in zip(left_runs, right_runs)]
        result[label] = shared.across_seeds(values)
    return {"direction": "left minus right on identical source/target frames within seed, then exactly three-seed mean and sample SD",
            "required_paired_frames_per_seed": 297, "metrics": result}


def summarize(root, training_hash, companion_hash, evaluator_hash):
    root = Path(root)
    runs = [load_run(root / f"{objective}_seed{seed}", objective, seed, training_hash, companion_hash, evaluator_hash)
            for objective in OBJECTIVES for seed in SEEDS]
    errors = []
    identities = {run["common_identity_sha256"] for run in runs if run["common_identity_sha256"] is not None}
    hashes = [run["checkpoint_sha256"] for run in runs if "checkpoint_sha256" in run]
    if len(identities) > 1:
        errors.append("Common diagnostic/training/data/source/runtime conventions differ across runs")
    if len(hashes) != len(set(hashes)):
        errors.append("Checkpoint SHA256 reused across distinct objective/seed identities")
    normalizers, frame_inputs = set(), {}
    for run in runs:
        for row in run["frames"]:
            normalizers.add(row["normalization_sha256"])
            key = row["source_index"], row["target_frame"]
            # Failed frames may not have read target or completed graph audits.
            observed = {name: value for name, value in row["common_input"].items() if name not in ("target_sha256", "graph")}
            frame_inputs.setdefault((key, "observed"), set()).add(shared.canonical_hash(observed))
            if row["status"] == "complete":
                frame_inputs.setdefault((key, "complete"), set()).add(shared.canonical_hash(row["common_input"]))
    if len(normalizers) > 1:
        errors.append("Saved acceleration normalization differs across models")
    if any(len(values) > 1 for values in frame_inputs.values()):
        errors.append("Paired observed/target histories, particle counts or geometry audits differ across models")
    if errors:
        for run in runs:
            run["eligible"] = False
    groups = {objective: [run for run in runs if run["objective"] == objective] for objective in OBJECTIVES}
    summaries = {}
    for objective, objective_runs in groups.items():
        seeds = [seed_summary(run) for run in objective_runs]
        summaries[objective] = {"per_seed": seeds,
            "metrics": {name: shared.across_seeds([row["equal_trajectory_metrics"][name] for row in seeds]) for name in METRICS},
            "particle_weighted_error_and_benefit": {name: shared.across_seeds([row["particle_weighted_error_and_benefit"][name] for row in seeds]) for name in WEIGHTED_METRICS}}
    comparisons = {}
    for objective, objective_runs in groups.items():
        for left, right in POLICY_COMPARISONS + ((REFERENCE, "base"),):
            fields = [("timing/" + name, f"timing/{left}/{name}", f"timing/{right}/{name}") for name in TIMING_FIELDS]
            if left in POLICIES:
                fields += [("accuracy/" + name, f"accuracy/{left}/{name}", f"accuracy/{right}/{name}") for name in ACCURACY_FIELDS]
            comparisons[f"{objective}:{left}-minus-{right}"] = paired_summary(objective_runs, objective_runs, fields)
        fields = [("base_error", "correlation/current_base_risk_vs_base_error", "correlation/previous_risk_vs_base_error"),
                  ("dense_benefit", "correlation/current_base_risk_vs_dense_benefit", "correlation/previous_risk_vs_dense_benefit")]
        comparisons[f"{objective}:current-minus-previous-risk-correlation"] = paired_summary(objective_runs, objective_runs, fields)
    comparisons["nll-minus-faithful"] = paired_summary(groups["nll"], groups["faithful"], [(name, name, name) for name in METRICS])
    complete = sum(run["state"] == "complete" for run in runs)
    state = ("invalid" if errors or any(run["state"] == "invalid" for run in runs) else "complete" if complete == 6
             else "pending" if all(run["state"] in ("missing", "pending_evaluation") for run in runs) else "incomplete")
    return {"schema": 1, "state": state, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "required_models": 6, "complete_models": complete, "aggregation_eligible_models": sum(run["eligible"] for run in runs),
            "required_frames_per_model": 297, "required_source_indices": list(SOURCE_INDICES), "required_target_frames": list(TARGET_FRAMES),
            "required_objectives": list(OBJECTIVES), "required_seeds": list(SEEDS), "policies": list(POLICIES), "timing_cases": list(CASES),
            "training_protocol_sha256": training_hash, "companion_protocol_sha256": companion_hash, "evaluator_source_sha256": evaluator_hash,
            "consistency_errors": errors,
            "interpretation": {"scope": "Teacher-forced same-observed-state diagnostics, not autonomous cached-own-graph rollout performance",
                               "primary_aggregation": "eleven equal frames per trajectory, then 27 equal trajectories, then exactly three seed means and sample SD (ddof=1)",
                               "repetitions": "six balanced runtime calls are reduced within each frame; never independent trained-model replicates",
                               "missing": "missing/partial/uncommitted runs, failed required frames or undefined required correlations give null applicable means; no available-seed or survivor average",
                               "timing": "natural base and base_shared_superset remain separate; previous-observed-base score generation is included; no amortized cached deployment or general speedup claim",
                               "signed_benefit": "base-minus-dense vector squared error; negative values preserved; sign fractions use normalized-vector benefit, whose sign can differ from position-space benefit under unequal coordinate scales; dense intervention is not marginal edge utility",
                               "correlations": "current-base and previous-observed-base correlations remain distinct; mean of frame Spearman coefficients, not pooled particle correlation",
                               "verification_limit": "validates recorded checkpoint provenance and JSON/array checksums; does not reopen checkpoints or parse particle arrays"},
            "runs": runs, "objectives": summaries, "paired_comparisons": comparisons}


def render_report(result):
    lines = ["# Full same-state diagnostic results", ""]
    if result["state"] == "pending":
        lines += ["**Results pending. No full diagnostic outcomes are present; no experimental results are reported.**", ""]
    else:
        lines += [f"**Diagnostic result set: {result['state']}.**", ""]
    lines += [f"Completed evaluator runs: {result['complete_models']}/6; eligible for aggregation: {result['aggregation_eligible_models']}/6. Each model requires 297 fixed observed frames (27 trajectories × 11 targets).", "",
              "Reported means and sample SDs use exactly three training-seed means. The six timing repetitions are averaged within each frame. Missing runs, failed frames and undefined required correlations remain explicit; no successful subset is substituted.", "",
              "| Model | State | Recorded frames |", "|---|---|---:|"]
    lines += [f"| {run['objective']} seed {run['seed']} | {run['state']} | {len(run['frames'])}/297 |" for run in result["runs"]]
    def value(metric):
        return "—" if metric["mean"] is None else f"{metric['mean']:.6g} ± {metric['sample_sd']:.3g}"
    lines += ["", "| Objective | Policy | Position MSE | Normalized acceleration MSE |", "|---|---|---:|---:|"]
    for objective in OBJECTIVES:
        for policy in POLICIES:
            metrics = result["objectives"][objective]["metrics"]
            lines.append(f"| {objective} | {policy} | {value(metrics[f'accuracy/{policy}/position_coordinate_mse'])} | {value(metrics[f'accuracy/{policy}/normalized_coordinate_mse'])} |")
    lines += ["", "| Objective | Risk correlation with current outcome | Mean frame Spearman |", "|---|---|---:|"]
    for objective in OBJECTIVES:
        for field in CORRELATIONS:
            lines.append(f"| {objective} | {field.replace('_', ' ')} | {value(result['objectives'][objective]['metrics']['correlation/' + field])} |")
    lines += ["", "| Objective | Signed dense benefit, normalized vector SE | Natural-base seconds | Shared-superset-base seconds | Previous-risk score-generation seconds |",
              "|---|---:|---:|---:|---:|"]
    for objective in OBJECTIVES:
        metrics = result["objectives"][objective]["metrics"]
        names = ("benefit/mean_normalized_vector_benefit", "timing/base/end_to_end_seconds", "timing/base_shared_superset/end_to_end_seconds",
                 f"timing/{POLICIES[-1]}/score_generation_seconds")
        lines.append(f"| {objective} | " + " | ".join(value(metrics[name]) for name in names) + " |")
    lines += ["", "The JSON includes signed paired policy/objective differences, failure and undefined-correlation counts, explicit missing frame IDs, and a separately labeled particle-weighted error/benefit alternative. Positive/negative/zero fractions refer to normalized-vector benefit; unequal coordinate scales can give a different sign from position-space benefit. The previous-observed-base risk diagnostic is distinct from autonomous cached risk. Timing is a reference-pipeline measurement; no cached amortization, significance or general speedup claim is made."]
    errors = result["consistency_errors"] + [f"{run['objective']} seed {run['seed']}: {error}" for run in result["runs"] for error in run["errors"]]
    if errors:
        lines += ["", "Validation issues:", ""] + [f"- {error}" for error in errors]
    return "\n".join(lines) + "\n"


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True, help="Six faithful_seed0..2 / nll_seed0..2 diagnostic directories")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, default=Path(__file__).parent / "protocols/full_waterdrop_100k.md")
    parser.add_argument("--companion-protocol", type=Path, default=Path(__file__).parent / "protocols/full_same_state_diagnostic.md")
    args = parser.parse_args()
    result = summarize(args.evaluation_root, sha256(args.training_protocol), sha256(args.companion_protocol), sha256(Path(__file__).parent / "full_same_state.py"))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output_prefix.with_suffix(".md").write_text(render_report(result))
    print(json.dumps({"state": result["state"], "complete_models": result["complete_models"], "output_prefix": str(args.output_prefix)}))


if __name__ == "__main__":
    main()
