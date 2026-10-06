"""Validate and summarize the fixed six-model WaterDrop rollout experiment.

Reads evaluator artifacts only: no checkpoint loading, model execution, or test
dataset access. Missing/partial runs remain incomplete. A completed evaluation
can contain failed trajectories; their all-sample full-horizon errors are null.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


OBJECTIVES = ("faithful", "nll")
SEEDS = (0, 1, 2)
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
SOURCE_INDICES = tuple(range(3, 30))
TRACE_STEPS = (1, 10, 50, 200, 500, 995)
HORIZON = 995
OFFICIAL_TEST_SHA256 = "b7f147c22e96fd3fbb8d595702cb8c85e412bfb449d7b4d761cc03eb76246c31"
OFFICIAL_TRAIN_SHA256 = "3fc97233eecff0216751ed36810f7e6f95f0224b2e93c86f991fab98e73f04eb"
OFFICIAL_VALID_SHA256 = "805fd3e03b15aa45e2f79e2ebe53f35473de10f9fb42c5d7cc963d663d44d364"
BOUNDARY_METRICS = tuple(f"{source}_boundary_{metric}" for source in ("predicted", "ground_truth")
                        for metric in ("mean_fraction_outside_gt_1e_minus6", "mean_step_maximum_excursion",
                                       "trajectory_maximum_excursion"))
BOUNDARY_FIELDS = {"fraction_particles_outside", "fraction_particles_outside_by_more_than_1e-6",
                   "maximum_coordinate_excursion", "mean_particle_maximum_excursion",
                   "coordinate_minimum", "coordinate_maximum"}
METRICS = ("mean_rollout_mse", "mse_at_200", "mse_at_995", "failure_fraction",
           "mean_directed_edges", "mean_rollout_wall_seconds", "mean_total_network_passes") + BOUNDARY_METRICS
COMPACT_FIELDS = ("trajectory_id", "source_index", "policy", "status", "failure", "horizon", "n_particles",
                  "rng_seed", "completed_steps", "mse_at_steps", "mse_at_final_horizon", "mean_rollout_mse",
                  "mean_directed_edges", "total_wall_seconds", "total_network_passes", "forecast_network_passes",
                  "trace_file", "trace_sha256")
POLICY_COMPARISONS = tuple((policy, "base") for policy in POLICIES[1:]) + tuple(
    ("laggedrisk25", policy) for policy in ("dense", "random25", "speed25"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    data = Path(path).read_bytes()
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(data, parse_constant=nonfinite), digest_bytes(data)


def canonical_hash(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def same_number(actual, expected):
    if expected is None:
        return actual is None
    return finite_nonnegative(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-15)


def summarize_boundary_series(series, completed, source, coordinate_guard):
    """Validate every accepted forecast step, then give time-balanced diagnostics.

    These geometric quantities do not establish physical validity. The failing
    attempted state is deliberately outside the accepted-step boundary arrays.
    """
    require(isinstance(series, list) and len(series) == completed,
            f"Invalid {source} boundary array length")
    fractions, maxima = [], []
    for step, item in enumerate(series, 1):
        label = f"{source} boundary step {step}"
        require(isinstance(item, dict) and set(item) == BOUNDARY_FIELDS, f"Invalid {label} fields")
        exact = item["fraction_particles_outside"]
        threshold = item["fraction_particles_outside_by_more_than_1e-6"]
        maximum = item["maximum_coordinate_excursion"]
        mean = item["mean_particle_maximum_excursion"]
        require(finite_nonnegative(exact) and finite_nonnegative(threshold)
                and 0 <= threshold <= exact <= 1, f"Invalid {label} outside fractions")
        require(finite_nonnegative(maximum) and finite_nonnegative(mean)
                and (mean <= maximum or math.isclose(mean, maximum, rel_tol=1e-12, abs_tol=0)),
                f"Invalid {label} excursion magnitudes")
        require((maximum > 0) == (exact > 0) and (maximum > 1e-6) == (threshold > 0)
                and (maximum > 0) == (mean > 0), f"Inconsistent {label} fractions/excursions")
        low, high = item["coordinate_minimum"], item["coordinate_maximum"]
        require(isinstance(low, list) and isinstance(high, list) and len(low) == len(high) == 2
                and all(type(value) in (int, float) and math.isfinite(value) for value in low + high)
                and all(a <= b for a, b in zip(low, high)), f"Invalid {label} coordinate ranges")
        if source == "predicted":
            require(all(abs(value) <= coordinate_guard for value in low + high),
                    f"Accepted {label} exceeds the coordinate resource guard")
        fractions.append(threshold)
        maxima.append(maximum)
    return {f"{source}_boundary_mean_fraction_outside_gt_1e_minus6": statistics.fmean(fractions) if completed else None,
            f"{source}_boundary_mean_step_maximum_excursion": statistics.fmean(maxima) if completed else None,
            f"{source}_boundary_trajectory_maximum_excursion": max(maxima) if completed else None}


def boundary_diagnostics(row, coordinate_guard):
    metrics = {}
    for source in ("predicted", "ground_truth"):
        metrics.update(summarize_boundary_series(row[f"{source}_boundary_per_step"], row["completed_steps"],
                                                 source, coordinate_guard))
    complete = row["status"] == "complete"
    return {"full_horizon": metrics if complete else {name: None for name in BOUNDARY_METRICS},
            "accepted_prefix_if_failed": None if complete else {
                "completed_steps": row["completed_steps"],
                "scope": "accepted steps only; excludes the failed attempt; never used in full-horizon or across-seed means",
                "metrics": metrics}}


def validate_protocol(protocol, objective, seed, research_protocol_sha256):
    """Return a common fingerprint; only the trained objective/seed may differ."""
    require(protocol.get("schema") == 2 and protocol.get("scope") == "locked_test"
            and protocol.get("split") == "test", "Expected schema-2 locked test evaluation")
    require(protocol.get("objective") == objective and type(protocol.get("seed")) is int and protocol.get("seed") == seed,
            "Evaluation objective/seed differs from expected directory identity")
    require(protocol.get("horizon") == HORIZON and protocol.get("context_frames") == 6
            and protocol.get("policies") == list(POLICIES), "Horizon/context/policies differ from fixed protocol")
    source = protocol["source_tfrecord"]
    require(source.get("sha256") == OFFICIAL_TEST_SHA256 and source.get("record_count") == 30
            and source.get("CRC_verified") is True
            and protocol.get("pinned_test_source_sha256") == OFFICIAL_TEST_SHA256,
            "Official complete 30-record test source identity differs")
    descriptions = protocol["trajectory_records"]
    require([row.get("source_index") for row in descriptions] == list(SOURCE_INDICES),
            "Required test source indices are exactly 3 through 29")
    ids = [row["id"] for row in descriptions]
    require(len(set(ids)) == 27 and protocol.get("trajectory_ids") == ids,
            "Trajectory IDs are duplicated or inconsistent")
    for row in descriptions:
        shape = row["positions"]["shape"]
        require(len(shape) == 3 and shape[0] == 1001 and type(shape[1]) is int and shape[1] > 0
                and shape[2] == 2, "Expected 1001-frame 2D test trajectory")
        require(valid_hash(row["positions"]["sha256"]) and valid_hash(row["particle_types"]["sha256"])
                and valid_hash(row["trajectory_content_sha256"]), "Missing trajectory content hashes")
    for name in ("manifest_sha256", "metadata_sha256", "checkpoint_sha256", "research_protocol_sha256"):
        require(valid_hash(protocol.get(name)), f"Invalid {name}")
    require(protocol["research_protocol_sha256"] == research_protocol_sha256,
            "Scientific protocol SHA256 differs from the requested fixed protocol")
    graph, guards = protocol["graph"], protocol["guards"]
    require(graph.get("base_radius") == .015 and graph.get("radius_factor") == 1.267
            and graph.get("optional_fraction") == .25, "Graph budget differs from fixed protocol")
    require(guards.get("max_candidate_pairs") == 100000 and guards.get("max_absolute_coordinate") == 10.
            and protocol["random_seed"].get("base") == 93000, "Resource guards/random seed differ")
    provenance = protocol["checkpoint_provenance"]
    training = provenance["training_config"]
    run = training["full_run"]
    require(provenance.get("checkpoint_format") == 2 and provenance.get("normalization_source") == "checkpoint"
            and provenance.get("uncertainty_parameterization") == "variance"
            and provenance.get("nmessage_passing_steps") == 10
            and provenance.get("connectivity_radius") == .015, "Checkpoint architecture/normalization provenance differs")
    require(training.get("loss") == objective and training.get("completed_optimizer_updates") == 100000
            and run.get("objective") == objective and run.get("seed") == seed
            and run.get("scope") == "bounded_full_data_100k" and run.get("steps") == 100000
            and run.get("batch_size") == 2 and run.get("history") == 6 and run.get("noise_std") == 6.7e-4,
            "Checkpoint is not the requested completed 100k objective/seed recipe")
    require(run.get("architecture") == {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
            "Checkpoint training architecture differs from the original 128x10 model")
    require(run.get("metadata_sha256") == protocol["metadata_sha256"]
            and run.get("research_protocol_sha256") == research_protocol_sha256,
            "Checkpoint metadata/scientific protocol differs from evaluation")
    for split, count, expected in (("train", 1000, OFFICIAL_TRAIN_SHA256), ("valid", 30, OFFICIAL_VALID_SHA256)):
        info = run[split]
        require(info.get("n_trajectories") == count and info["source"].get("sha256") == expected
                and valid_hash(info.get("manifest_sha256")), f"Checkpoint {split} data identity differs")
    sources, eval_sources = run["source_sha256"], protocol["code_sha256"]
    require(bool(sources) and bool(eval_sources)
            and all(valid_hash(value) for value in itertools.chain(sources.values(), eval_sources.values())),
            "Missing frozen source hashes")
    require(all(value == eval_sources[name] for name, value in sources.items() if name in eval_sources),
            "Training/evaluation source hashes differ on shared code")
    # The model checksum, objective, seed and their corresponding training
    # fields necessarily differ. Everything else must describe one experiment.
    common_run = {key: value for key, value in run.items() if key not in ("objective", "seed")}
    common_provenance = {key: value for key, value in provenance.items() if key != "training_config"}
    common_protocol = {key: value for key, value in protocol.items()
                       if key not in ("seed", "objective", "checkpoint_sha256", "checkpoint_provenance")}
    return canonical_hash({"evaluation": common_protocol, "training": common_run,
                           "checkpoint_provenance": common_provenance})


def validate_record(directory, compact, protocol, protocol_hash):
    index, policy = compact["source_index"], compact["policy"]
    require(type(index) is int and index in SOURCE_INDICES and policy in POLICIES,
            "Unexpected source index or policy")
    record_name = f"trajectory_{index:06d}_{policy}.json"
    trace_name = f"trajectory_{index:06d}_{policy}.npz"
    require(compact.get("record_file") == record_name and compact.get("trace_file") == trace_name,
            "Unexpected record/trace path")
    row, row_hash = read_json(directory / record_name)
    require(row_hash == compact.get("record_sha256"), "Raw record SHA256 differs from compact index")
    require(set(compact) == set(COMPACT_FIELDS) | {"record_file", "record_sha256"}
            and all(row.get(key) == compact[key] for key in COMPACT_FIELDS), "Raw record and compact index differ")
    require(sha256(directory / trace_name) == row["trace_sha256"], "Trace SHA256 differs")
    description = protocol["trajectory_records"][index - 3]
    require(row.get("protocol_sha256") == protocol_hash and row["trajectory_id"] == description["id"]
            and row["n_particles"] == description["positions"]["shape"][1]
            and row["rng_seed"] == 93000 + 1000 * protocol["seed"] + index and row["horizon"] == HORIZON,
            "Raw record identity/seed/protocol differs")
    completed = row["completed_steps"]
    require(type(completed) is int and 0 <= completed <= HORIZON, "Invalid completed-step count")
    require((row["status"] == "complete" and completed == HORIZON and row["failure"] is None)
            or (row["status"] == "failed" and completed < HORIZON and isinstance(row["failure"], dict)
                and isinstance(row["failure"].get("category"), str)), "Inconsistent completion/failure outcome")
    mse, edges = row["mse_per_step"], row["directed_edges_per_step"]
    require(len(mse) == completed and all(finite_nonnegative(value) for value in mse)
            and len(edges) == completed and all(type(value) is int and value >= 0 for value in edges),
            "Invalid per-step errors/edge counts")
    require(set(row["mse_at_steps"]) == {str(step) for step in TRACE_STEPS}, "Trace-step metric keys differ")
    for step in TRACE_STEPS:
        require(same_number(row["mse_at_steps"][str(step)], mse[step - 1] if completed >= step else None),
                "Reported horizon error differs from raw per-step errors")
    complete = row["status"] == "complete"
    for field, expected in (("mean_rollout_mse", statistics.fmean(mse) if complete else None),
                            ("mse_at_final_horizon", mse[-1] if complete else None),
                            ("mean_directed_edges", statistics.fmean(edges) if complete else None)):
        require(same_number(row[field], expected), f"{field} differs from raw steps or suppresses a failure")
    require(finite_nonnegative(row["total_wall_seconds"])
            and type(row["total_network_passes"]) is int and type(row["forecast_network_passes"]) is int
            and completed <= row["forecast_network_passes"] <= completed + 1,
            "Invalid timing/network-pass count")
    warmup_passes = row["total_network_passes"] - row["forecast_network_passes"]
    require(warmup_passes in ((0, 1) if policy == "laggedrisk25" else (0,)), "Invalid warmup pass count")
    if complete:
        require(row["forecast_network_passes"] == HORIZON and warmup_passes == int(policy == "laggedrisk25"),
                "Completed rollout network-pass count differs from the fixed policy")
    return {**compact, "boundary_diagnostics": boundary_diagnostics(row, protocol["guards"]["max_absolute_coordinate"])}


def load_run(directory, objective, seed, research_protocol_sha256):
    directory = Path(directory)
    result = {"objective": objective, "seed": seed, "directory": str(directory), "state": "missing",
              "errors": [], "records": [], "eligible_for_aggregation": False, "common_identity_sha256": None}
    if not directory.exists():
        return result
    try:
        require(directory.is_dir(), "Expected an evaluator directory")
        if not (directory / "protocol.json").exists() and not (directory / "result.json").exists():
            require(not any(directory.glob("trajectory_*")), "Per-trajectory artifacts exist without protocol.json")
            result["state"] = "pending_evaluation"
            return result
        protocol, protocol_hash = read_json(directory / "protocol.json")
        result["common_identity_sha256"] = validate_protocol(protocol, objective, seed, research_protocol_sha256)
        result.update(protocol_sha256=protocol_hash, checkpoint_sha256=protocol["checkpoint_sha256"],
                      trajectory_ids=protocol["trajectory_ids"])
        if not (directory / "result.json").exists():
            result["state"] = "pending_evaluation"
            return result
        saved, result_hash = read_json(directory / "result.json")
        require(saved.get("protocol_sha256") == protocol_hash and saved.get("state") in ("partial", "complete"),
                "Result protocol/state differs")
        rows, keys = [], set()
        for compact in saved["records"]:
            row = validate_record(directory, compact, protocol, protocol_hash)
            key = (row["source_index"], row["policy"])
            require(key not in keys, "Duplicate trajectory/policy record")
            keys.add(key)
            rows.append(row)
        result.update(records=rows, result_sha256=result_hash, evaluator_state=saved["state"],
                      missing_policy_trajectories=[{"source_index": index, "policy": policy}
                          for index in SOURCE_INDICES for policy in POLICIES if (index, policy) not in keys])
        status = read_json(directory / "status.json")[0] if (directory / "status.json").exists() else {}
        result["evaluator_status"] = status.get("state", "missing")
        if saved["state"] == "complete":
            require(len(keys) == 27 * 5, "Complete result is missing required trajectory/policy outcomes")
            require(saved.get("seed") == seed and saved.get("objective") == objective
                    and saved.get("checkpoint_sha256") == protocol["checkpoint_sha256"],
                    "Final result checkpoint/objective/seed differs")
            if status.get("state") != "complete":
                result.update(state="incomplete", errors=["Final result exists but evaluator completion is not committed in status.json"])
                return result
            require(status.get("result_sha256") == result_hash,
                    "Completion status/result SHA256 differs; snapshot may be changing")
            result.update(state="complete", eligible_for_aggregation=True)
        else:
            result["state"] = "incomplete"
    except (ValueError, KeyError, TypeError, OSError, OverflowError, IndexError, AttributeError) as error:
        result.update(state="invalid", eligible_for_aggregation=False,
                      errors=[f"{type(error).__name__}: {error}"])
    return result


def record_metric(row, metric):
    if metric in BOUNDARY_METRICS:
        return row["boundary_diagnostics"]["full_horizon"][metric]
    if metric == "mse_at_200":
        return row["mse_at_steps"]["200"]
    if metric == "mse_at_995":
        return row["mse_at_steps"]["995"]
    if metric == "failure_fraction":
        return float(row["status"] == "failed")
    if metric == "mean_rollout_wall_seconds":
        return row["total_wall_seconds"]
    if metric == "mean_total_network_passes":
        return row["total_network_passes"]
    return row[metric]


def complete_mean(values):
    return statistics.fmean(values) if len(values) == 27 and all(value is not None for value in values) else None


def seed_policy_summary(run, policy):
    rows = sorted((row for row in run["records"] if row["policy"] == policy), key=lambda row: row["source_index"])
    present = {row["source_index"] for row in rows}
    eligible = run["eligible_for_aggregation"] and present == set(SOURCE_INDICES)
    metrics = {name: complete_mean([record_metric(row, name) for row in rows]) if eligible else None for name in METRICS}
    return {"seed": run["seed"], "run_state": run["state"], "required_trajectories": 27,
            "observed_trajectories": len(rows), "completed_trajectories": sum(row["status"] == "complete" for row in rows),
            "failed_trajectories": sum(row["status"] == "failed" for row in rows),
            "missing_source_indices": [index for index in SOURCE_INDICES if index not in present],
            "all_sample_full_horizon_error_defined": metrics["mean_rollout_mse"] is not None,
            "all_sample_full_horizon_boundary_defined": all(metrics[name] is not None for name in BOUNDARY_METRICS),
            "metrics": metrics,
            "failures": [{key: row[key] for key in ("trajectory_id", "source_index", "completed_steps", "failure")}
                         for row in rows if row["status"] == "failed"],
            "boundary_prefix_diagnostics_if_failed": [
                {"trajectory_id": row["trajectory_id"], "source_index": row["source_index"],
                 **row["boundary_diagnostics"]["accepted_prefix_if_failed"]}
                for row in rows if row["status"] == "failed"]}


def across_seeds(values):
    require(len(values) == 3, "Three explicit seed slots required")
    available = [value for value in values if value is not None]
    complete = len(available) == 3
    return {"seed_values": [{"seed": seed, "value": value} for seed, value in zip(SEEDS, values)],
            "defined_seed_count": len(available), "mean": statistics.fmean(available) if complete else None,
            "sample_sd": statistics.stdev(available) if complete else None}


def paired_comparison(left_runs, right_runs, left_policy, right_policy):
    seed_rows = []
    for seed, left, right in zip(SEEDS, left_runs, right_runs):
        left_map = {row["source_index"]: row for row in left["records"] if row["policy"] == left_policy}
        right_map = {row["source_index"]: row for row in right["records"] if row["policy"] == right_policy}
        eligible = left["eligible_for_aggregation"] and right["eligible_for_aggregation"]
        pairs = []
        for index in SOURCE_INDICES:
            if index not in left_map or index not in right_map:
                continue
            a, b = left_map[index], right_map[index]
            if a["trajectory_id"] != b["trajectory_id"]:
                require(not eligible, "Paired trajectory IDs differ")
                continue
            delta = {}
            for metric in METRICS:
                x, y = record_metric(a, metric), record_metric(b, metric)
                delta[metric] = x - y if eligible and x is not None and y is not None else None
            pairs.append({"source_index": index, "trajectory_id": a["trajectory_id"], "deltas": delta})
        seed_rows.append({"seed": seed, "eligible": eligible, "paired_trajectories": len(pairs),
                          "metrics": {name: complete_mean([row["deltas"][name] for row in pairs])
                                      if eligible else None for name in METRICS}, "trajectory_pairs": pairs})
    return {"direction": "left minus right; negative error/failure differences favor left",
            "left_policy": left_policy, "right_policy": right_policy, "per_seed": seed_rows,
            "metrics": {name: across_seeds([row["metrics"][name] for row in seed_rows]) for name in METRICS}}


def summarize(evaluation_root, research_protocol_sha256):
    root = Path(evaluation_root)
    runs = [load_run(root / f"{objective}_seed{seed}", objective, seed, research_protocol_sha256)
            for objective in OBJECTIVES for seed in SEEDS]
    identities = {run["common_identity_sha256"] for run in runs if run["common_identity_sha256"] is not None}
    checkpoints = [run["checkpoint_sha256"] for run in runs if "checkpoint_sha256" in run]
    consistency_errors = []
    if len(identities) > 1:
        consistency_errors.append("Evaluation/training/data/source/software provenance differs across the six-model experiment")
    if len(checkpoints) != len(set(checkpoints)):
        consistency_errors.append("A checkpoint SHA256 is reused across different objective/seed combinations")
    if consistency_errors:
        for run in runs:
            run["eligible_for_aggregation"] = False
    grouped = {objective: [run for run in runs if run["objective"] == objective] for objective in OBJECTIVES}
    policy_results = {}
    for objective, objective_runs in grouped.items():
        policy_results[objective] = {}
        for policy in POLICIES:
            per_seed = [seed_policy_summary(run, policy) for run in objective_runs]
            policy_results[objective][policy] = {"per_seed": per_seed,
                "metrics": {name: across_seeds([row["metrics"][name] for row in per_seed]) for name in METRICS}}
    comparisons = {}
    for objective, objective_runs in grouped.items():
        for left, right in POLICY_COMPARISONS:
            comparisons[f"{objective}:{left}-minus-{right}"] = paired_comparison(objective_runs, objective_runs, left, right)
    for policy in POLICIES:
        comparisons[f"nll-minus-faithful:{policy}"] = paired_comparison(grouped["nll"], grouped["faithful"], policy, policy)
    complete_runs = sum(run["state"] == "complete" for run in runs)
    state = ("invalid" if consistency_errors or any(run["state"] == "invalid" for run in runs)
             else "complete" if complete_runs == 6
             else "pending" if all(run["state"] in ("missing", "pending_evaluation") for run in runs) else "incomplete")
    return {"schema": 1, "state": state, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Fixed six-model locked WaterDrop evaluation; completed evaluation is distinct from successful trajectories",
            "research_protocol_sha256": research_protocol_sha256, "official_test_sha256": OFFICIAL_TEST_SHA256,
            "required_objectives": list(OBJECTIVES), "required_seeds": list(SEEDS), "required_policies": list(POLICIES),
            "required_source_indices": list(SOURCE_INDICES), "required_models": 6, "complete_models": complete_runs,
            "aggregation_eligible_models": sum(run["eligible_for_aggregation"] for run in runs),
            "required_policy_trajectories": 810, "consistency_errors": consistency_errors,
            "aggregation": {"within_seed": "equal trajectories, with coordinate MSE already averaged over particles",
                            "across_seeds": "mean and sample SD (ddof=1) of exactly three seed means; no available-seed average",
                            "paired": "left-minus-right differences on identical trajectory IDs within each seed, then three-seed mean/sample SD",
                            "failures": "full-horizon MSE and edges undefined after any failed/missing trajectory; failure counts retained",
                            "boundary": "outside fraction uses >1e-6 coordinate excursion; average steps equally within a trajectory, trajectories equally within a seed, and report exactly three seed means/sample SD; predicted and same-frame ground-truth references are separate",
                            "boundary_maxima": "mean_step_maximum_excursion averages the per-step maximum over steps; trajectory_maximum_excursion takes the largest excursion across all steps, then reports the equal-trajectory mean of these trajectory maxima",
                            "boundary_failures": "every full-horizon boundary metric, including the ground-truth reference, is undefined for a failed/missing trajectory; accepted-prefix diagnostics are retained separately and never averaged across survivors or seeds",
                            "partial": "partial or uncommitted evaluator runs never enter aggregate means",
                            "timing": "includes failures and warmup; different completion lengths and fixed policy order prevent a speedup inference"},
            "runs": runs, "policies": policy_results, "paired_comparisons": comparisons}


def render_report(summary):
    lines = ["# Full WaterDrop rollout results", ""]
    if summary["state"] == "pending":
        lines += ["**Results pending. No rollout outcomes are present; no experimental outcomes are reported.**", ""]
    elif summary["state"] != "complete":
        lines += [f"**Results {summary['state']}. The required six-model evaluation is not complete.**", ""]
    else:
        lines += ["All six evaluator runs are complete. Individual trajectories may still have failed; those failures remain in the results.", ""]
    lines += [f"Completed evaluator runs: {summary['complete_models']}/6; eligible for aggregation: {summary['aggregation_eligible_models']}/6. Required: faithful and NLL, seeds 0–2, five policies, official test source indices 3–29, 995 forecast steps.", "",
              "Errors are averaged equally over 27 trajectories within each seed. Tables show the mean ± sample SD of exactly three seed means. Missing values are undefined or pending; failed trajectories are never dropped.", "",
              "| Model | State | Recorded policy trajectories |", "|---|---|---:|"]
    lines += [f"| {run['objective']} seed {run['seed']} | {run['state']} | {len(run['records'])}/135 |" for run in summary["runs"]]
    lines += ["", "| Objective | Policy | Mean rollout MSE | MSE@200 | MSE@995 | Failure fraction |",
              "|---|---|---:|---:|---:|---:|"]
    def value(metric):
        return "—" if metric["mean"] is None else f"{metric['mean']:.6g} ± {metric['sample_sd']:.3g}"
    for objective in OBJECTIVES:
        for policy in POLICIES:
            metrics = summary["policies"][objective][policy]["metrics"]
            lines.append(f"| {objective} | {policy} | " + " | ".join(value(metrics[name]) for name in
                         ("mean_rollout_mse", "mse_at_200", "mse_at_995", "failure_fraction")) + " |")
    lines += ["", "Boundary diagnostics use the fraction of particles outside the metadata bounds by more than 1e-6. Fractions and per-step maxima are averaged equally over forecast steps within each trajectory, then equally over trajectories within each seed. The trajectory maximum is the worst excursion in one trajectory; its table column averages those maxima across trajectories, not the single worst particle from the entire experiment. Values are mean ± sample SD across all three seeds. Ground-truth references use the same forecast frames. These are geometric diagnostics, not a physical-validity certificate.", "",
              "| Objective | Policy | Predicted outside fraction | Truth outside fraction | Predicted mean step max excursion | Truth mean step max excursion | Predicted mean trajectory max excursion | Truth mean trajectory max excursion |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    boundary_order = tuple(f"{source}_boundary_{metric}" for metric in
                           ("mean_fraction_outside_gt_1e_minus6", "mean_step_maximum_excursion", "trajectory_maximum_excursion")
                           for source in ("predicted", "ground_truth"))
    for objective in OBJECTIVES:
        for policy in POLICIES:
            metrics = summary["policies"][objective][policy]["metrics"]
            lines.append(f"| {objective} | {policy} | " + " | ".join(value(metrics[name]) for name in boundary_order) + " |")
    lines += ["", "Any failed or missing trajectory makes the corresponding all-sample full-horizon boundary means undefined, including the truth reference. The JSON retains accepted-prefix diagnostics for failed trajectories separately; these prefixes exclude the rejected attempt and are never substituted into the table or pooled across surviving trajectories."]
    lines += ["", "The JSON contains every recorded trajectory outcome, explicit missing runs/trajectories, and paired policy and objective differences. Negative paired error or failure differences favor the left-hand method. No significance or speedup claim follows from this summary."]
    errors = summary["consistency_errors"] + [f"{run['objective']} seed {run['seed']}: {error}"
             for run in summary["runs"] for error in run["errors"]]
    if errors:
        lines += ["", "Validation issues:", ""] + [f"- {error}" for error in errors]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True,
                        help="Contains faithful_seed0..2 and nll_seed0..2 evaluator directories")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).parent / "protocols/full_waterdrop_100k.md")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Write PREFIX.json and PREFIX.md")
    args = parser.parse_args()
    result = summarize(args.evaluation_root, sha256(args.protocol))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output_prefix.with_suffix(".md").write_text(render_report(result))
    print(json.dumps({"state": result["state"], "complete_models": result["complete_models"],
                      "json": str(args.output_prefix.with_suffix('.json')), "report": str(args.output_prefix.with_suffix('.md'))}))


if __name__ == "__main__":
    main()
