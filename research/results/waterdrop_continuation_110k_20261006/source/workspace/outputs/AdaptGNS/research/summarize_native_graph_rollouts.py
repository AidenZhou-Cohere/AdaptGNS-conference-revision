"""Strict, fixed-population saved-result analysis of native graph rollouts.

No checkpoint loading or model execution. Original and native-convention
outcomes remain distinct, with paired comparisons only for identical models
and source trajectories. This module accepts only the original six 100k models.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from research import native_graph_rollout as native
from research import summarize_full_rollouts as original
from research.summarize_graph_convention_bridge import Audit

BASE = native.full
BRIDGE = native.bridge
FULL_NATIVE_METRICS = ("native_base_directed_edges", "native_base_self_edges",
    "native_base_receivers_above_cap_before_capping", "native_base_edges_removed_by_cap",
    "native_base_asymmetric_directed_edges", "retained_optional_pairs")
EXTRA_METRICS = tuple("mean_" + name for name in FULL_NATIVE_METRICS) + (
    "cap_active_step_fraction", "mean_wall_seconds_including_native_parity", "mean_native_parity_seconds",
    "mean_native_parity_passes", "mean_graph_operational_seconds", "mean_forward_component_seconds")
METRICS = original.METRICS + EXTRA_METRICS
NATIVE_CONTRACT = {
    "schema": 1, "scope": "post-inspection exploratory native-graph autonomous bridge; not locked confirmation",
    "graph": "strict native float32 radius; native capped128 directed base+self-edge candidates; preserve base ordered prefix; append uncapped symmetric selected geometric annulus pairs; R=1.267r",
    "random_seed": "93000+1000*training_seed+source_index; same initial material as locked rollout; geometry-dependent subsequent draws differ",
    "risk": "one explicit native-base warmup, forecast1 exact budget, later cached risk from previous own selected graph and predicted history",
    "native_parity": "two extra verification passes per trajectory before policy timer; raw arrays saved; failure stops model job",
    "time_scope": "operational synchronized durations; fixed policy order and changing geometry; no causal speed comparison; native verification separately reported",
    "guards": {"max_candidate_pairs": 100000, "max_abs_coordinate": 10.}, "deadline_utc": native.DEADLINE,
    "trace_steps": list(original.TRACE_STEPS), "policies": list(native.POLICIES), "horizon": 995,
    "source_indices": list(range(3, 30))}


def validate_native_contract(protocol, audit):
    for key, expected in NATIVE_CONTRACT.items():
        audit.equal(protocol.get(key), expected, "Native protocol contract differs: " + key)
    audit.check(type(protocol["threads"]) is int and protocol["threads"] > 0, "Native thread count invalid")
    audit.equal(protocol["runtime"]["radius_backend"], "scipy_host", "Native graph backend differs")
    return original.canonical_hash({key: protocol[key] for key in tuple(NATIVE_CONTRACT) + ("runtime", "threads", "software")})


def metric(row, name):
    return original.record_metric(row, name) if name in original.METRICS else row.get("native_metrics", {}).get(name)


def summarize_seed(records, policy):
    rows = sorted((row for row in records if row["policy"] == policy), key=lambda row: row["source_index"])
    if [row["source_index"] for row in rows] != list(range(3, 30)):
        raise ValueError("Exactly 27 source trajectories are required for each policy")
    return {"policy": policy, "required_trajectories": 27,
        "completed_trajectories": sum(row["status"] == "complete" for row in rows),
        "failed_trajectories": sum(row["status"] == "failed" for row in rows),
        "metrics": {name: original.complete_mean([metric(row, name) for row in rows]) for name in METRICS},
        "trajectories": rows}


def paired(left_runs, right_runs, left_policy, right_policy, names=METRICS):
    seeds = []
    for seed, left, right in zip((0, 1, 2), left_runs, right_runs):
        a = {row["source_index"]: row for row in left["records"] if row["policy"] == left_policy}
        b = {row["source_index"]: row for row in right["records"] if row["policy"] == right_policy}
        if set(a) != set(range(3, 30)) or set(b) != set(range(3, 30)):
            raise ValueError("Paired comparison requires all 27 source trajectories")
        rows = []
        for index in range(3, 30):
            if a[index]["trajectory_id"] != b[index]["trajectory_id"]:
                raise ValueError("Paired trajectory IDs differ")
            values = {}
            for name in names:
                x, y = metric(a[index], name), metric(b[index], name)
                values[name] = x - y if x is not None and y is not None else None
            rows.append({"source_index": index, "trajectory_id": a[index]["trajectory_id"], "deltas": values})
        seeds.append({"seed": seed, "trajectory_pairs": rows,
                      "metrics": {name: original.complete_mean([row["deltas"][name] for row in rows]) for name in names}})
    return {"direction": "left minus right; negative error/failure favors left", "left_policy": left_policy,
            "right_policy": right_policy, "per_seed": seeds,
            "metrics": {name: original.across_seeds([row["metrics"][name] for row in seeds]) for name in names}}


def checkpoint_validation_view(protocol, manifest, manifest_hash, metadata_hash, research_hash):
    """Compatibility input for unchanged original numerical validators only.

    The returned schema is never published as this experiment's protocol and
    never changes the native run's hash or exploratory scope.
    """
    sources = {}
    for name, digest in protocol["input_files_sha256"].items():
        path = Path(name)
        if path.suffix == ".py" and path.is_relative_to(BASE.REPO):
            sources[str(path.relative_to(BASE.REPO))] = digest
    return {"schema": 2, "scope": "locked_test", "split": "test", "objective": protocol["objective"],
        "seed": protocol["seed"], "horizon": 995, "context_frames": 6, "policies": list(native.POLICIES),
        "source_tfrecord": manifest["source"], "pinned_test_source_sha256": BASE.OFFICIAL_TEST_SHA256,
        "trajectory_records": manifest["records"][3:], "trajectory_ids": protocol["trajectory_ids"],
        "manifest_sha256": manifest_hash, "metadata_sha256": metadata_hash,
        "checkpoint_sha256": protocol["checkpoint_sha256"], "checkpoint_provenance": protocol["checkpoint_provenance"],
        "research_protocol_sha256": research_hash, "graph": {"base_radius": .015, "radius_factor": 1.267, "optional_fraction": .25},
        "guards": {"max_candidate_pairs": 100000, "max_absolute_coordinate": 10.}, "random_seed": {"base": 93000},
        "code_sha256": sources}


def validate_native_parity(row, traces, audit):
    parity = row["native_parity"]
    if row["native_parity_network_passes"] == 0:
        audit.check(row["status"] == "failed" and row["failure"]["phase"] in ("initial_state", "initial_graph"), "Unexecuted parity is not an initial guard failure")
        audit.check(np.array_equal(traces["rejected_history"], traces["initial_observed_positions"], equal_nan=True), "Initial failed history differs")
        return
    audit.equal(row["native_parity_network_passes"], 2, "Native verification pass count differs")
    graph = BRIDGE.strict_pairs(traces["initial_observed_positions"][-1], .015)
    expected = BRIDGE.ordered_edges(traces["initial_observed_positions"][-1], graph.base, True, 128)
    audit.array(traces["initial_parity_supplied_edges"], expected, "Initial native base reconstruction differs")
    equal = [np.array_equal(traces["initial_parity_native_node_features"], traces["initial_parity_supplied_node_features"]),
             np.array_equal(traces["initial_parity_native_edges"], expected),
             np.array_equal(traces["initial_parity_native_edge_features"], traces["initial_parity_supplied_edge_features"])]
    arrays = {key: (traces[f"initial_parity_native_{key}"], traces[f"initial_parity_supplied_{key}"])
              for key in ("prediction", "risk", "raw_risk")}
    finite = all(np.isfinite(a).all() and np.isfinite(b).all() for a, b in arrays.values())
    agree = bool(finite and np.allclose(*arrays["prediction"], rtol=0, atol=BRIDGE.PARITY_PRED_ATOL)
                 and all(np.allclose(*arrays[key], rtol=BRIDGE.PARITY_RISK_RTOL, atol=BRIDGE.PARITY_RISK_ATOL) for key in ("risk", "raw_risk")))
    for key, value in (("native_edge_identity", equal[1]), ("native_supplied_feature_identity", equal), ("finite", finite),
                       ("prediction_risk_agree", agree), ("passed", bool(all(equal) and agree)),
                       ("prediction_atol", BRIDGE.PARITY_PRED_ATOL), ("risk_atol", BRIDGE.PARITY_RISK_ATOL), ("risk_rtol", BRIDGE.PARITY_RISK_RTOL)):
        audit.equal(parity[key], value, "Native parity flag/tolerance differs")
    for key, (a, b) in arrays.items():
        audit.equal(parity[key + "_max_abs_difference"], float(np.max(np.abs(a-b))) if finite else None, "Native parity maximum differs")


def validate_graph_counts(attempt, policy, n, audit):
    candidate, base, annulus = [attempt[key] for key in ("candidate_pairs", "geometric_base_pairs", "available_annulus_pairs")]
    audit.check(all(type(value) is int and value >= 0 for value in (candidate, base, annulus)), "Invalid graph pair counts")
    audit.equal(candidate, base + annulus, "Candidate decomposition differs")
    audit.check(candidate <= 100000, "Accepted graph exceeds candidate guard")
    budget = int(.25 * annulus)
    audit.equal(attempt["optional_pair_budget"], budget, "Budget differs")
    optional = 0 if policy == "base" else annulus if policy == "dense" else budget
    audit.equal(attempt["retained_optional_pairs"], optional, "Retained optional budget differs")
    native_edges, self_edges = attempt["native_base_directed_edges"], attempt["native_base_self_edges"]
    audit.check(type(native_edges) is int and 0 <= native_edges <= 128*n, "Native directed count exceeds receiver cap")
    audit.check(type(self_edges) is int and 0 <= self_edges <= n, "Native self-edge count differs")
    audit.equal(attempt["native_base_edges_removed_by_cap"], 2*base+n-native_edges, "Cap removal arithmetic differs")
    audit.equal(attempt["directed_edges"], native_edges+2*optional, "Native plus optional directed count differs")
    audit.check(attempt["native_base_prefix_preserved"] is True, "Native ordered prefix not preserved")
    audit.check(0 <= attempt["native_base_receivers_above_cap_before_capping"] <= n, "Invalid capped receiver count")
    audit.check(0 <= attempt["native_base_asymmetric_directed_edges"] <= native_edges-self_edges, "Invalid directed asymmetry count")
    audit.check(0 <= attempt["native_base_max_receiver_degree"] <= 128, "Native receiver maximum exceeds cap")
    for key in ("native_base_sha256", "selected_optional_pair_sha256", "directed_edge_sha256"):
        audit.check(original.valid_hash(attempt[key]), "Invalid graph SHA256")


def audit_trace_graph(history, cache, rng, attempt, policy, edge, optional, audit):
    _, rebuilt_edge, rebuilt_optional, rebuilt = native.native_graph(history, policy, cache, rng)
    audit.array(edge, rebuilt_edge, "Trace native directed graph differs")
    audit.array(optional, rebuilt_optional, "Trace optional pairs differ")
    for key, value in rebuilt.items():
        audit.equal(attempt[key], value, "Trace graph/cap/asymmetry metadata differs")


def validate_failure(row, traces, audit):
    failure = row["failure"]
    if failure is None:
        return
    phase = failure["phase"]
    expected_step = 0 if phase in ("initial_state", "initial_graph", "native_parity", "warmup") else row["completed_steps"]+1
    audit.equal(failure["forecast_step"], expected_step, "Failed attempt index differs from accepted prefix")
    if phase == "native_parity":
        audit.check(not row["native_parity"]["passed"], "Native parity failure has passing gate")
        return
    history = traces["failed_input_history"]
    audit.check(np.array_equal(history, traces["rejected_history"], equal_nan=True), "Failed input history copies differ")
    try:
        if phase in ("initial_state", "state"):
            BASE.check_state(history, "initial_state" if phase == "initial_state" else "state", 10.)
        elif phase in ("initial_graph", "graph"):
            BRIDGE.strict_pairs(history[-1], .015)
        elif phase in ("forward", "warmup"):
            prefix = "warmup" if phase == "warmup" else "rejected"
            output = {key: traces[f"{prefix}_{key}"] for key in ("prediction", "raw_risk", "risk")}
            BRIDGE.validate_output(output, row["n_particles"])
        else:
            raise ValueError("Unknown native failure phase")
    except BASE.RolloutGuard as error:
        for key, value in error.details.items():
            if key != "phase":
                audit.equal(failure[key], value, "Saved failure reason differs from raw failed input/output")
    else:
        raise ValueError("Saved scientific failure cannot be reproduced from retained arrays")


def validate_trajectory(row, traces, positions, types, audit):
    completed, n, policy = row["completed_steps"], row["n_particles"], row["policy"]
    failed_initial = row["failure"] is not None and row["failure"]["phase"] == "initial_state"
    audit.check(np.array_equal(traces["initial_observed_positions"], positions[:6], equal_nan=failed_initial), "Initial observations differ from official source")
    audit.array(traces["particle_types"], types, "Types differ from official source")
    audit.equal(BASE.state_hash(traces["initial_observed_positions"]), row["initial_observed_state_sha256"], "Initial history hash differs")
    validate_native_parity(row, traces, audit)
    initial_boundary = [BASE.boundary_metrics(frame, traces["bounds"]) for frame in positions[:6]] if np.isfinite(positions[:6]).all() else None
    audit.equal(row["initial_observed_boundary"], initial_boundary, "Initial observed boundary differs")
    for key in ("retained_optional_pairs_per_step", "candidate_pairs_per_step", "base_pairs_per_step", "mean_normalized_acceleration_variance_per_step"):
        audit.equal(len(row[key]), completed, "Native per-step array length differs")
    audit.check(all(original.finite_nonnegative(value) and value > 0 for value in row["mean_normalized_acceleration_variance_per_step"]), "Invalid per-step positive risk mean")
    attempts = row["attempts"]
    audit.equal([item["forecast_step"] for item in attempts], list(range(1, len(attempts)+1)), "Attempt sequence differs")
    accepted = [item for item in attempts if item["accepted"]]
    audit.equal(len(accepted), completed, "Accepted prefix length differs")
    audit.check(all(item["accepted"] for item in attempts[:completed]) and all(not item["accepted"] for item in attempts[completed:]), "Accepted steps are not a prefix")
    audit.check(len(attempts) in (completed, completed+1), "Too many rejected attempts")
    expected_trace_steps = [step for step in original.TRACE_STEPS if step <= completed]
    audit.equal(traces["forecast_steps"].tolist(), expected_trace_steps, "Trace forecast schedule differs")
    trace_slots = {step: slot for slot, step in enumerate(expected_trace_steps)}
    rng = np.random.default_rng(row["rng_seed"])
    forward_calls = 0
    for step, attempt in enumerate(attempts, 1):
        if "directed_edges" in attempt:
            validate_graph_counts(attempt, policy, n, audit)
        if "features_forward_decode_and_transfer_seconds" in attempt:
            forward_calls += 1
        if step <= completed:
            audit.equal(attempt["coordinate_mse"], row["mse_per_step"][step-1], "Attempt/prefix MSE differs")
            for target, key in (("directed_edges_per_step", "directed_edges"), ("retained_optional_pairs_per_step", "retained_optional_pairs"),
                                ("candidate_pairs_per_step", "candidate_pairs"), ("base_pairs_per_step", "geometric_base_pairs")):
                audit.equal(row[target][step-1], attempt[key], "Attempt/prefix graph count differs")
            truth_boundary = BASE.boundary_metrics(positions[5+step], traces["bounds"])
            audit.equal(row["ground_truth_boundary_per_step"][step-1], truth_boundary, "Truth boundary differs from actual source")
            audit.equal(attempt["ground_truth_boundary"], truth_boundary, "Attempt truth boundary differs")
            audit.equal(attempt["predicted_boundary"], row["predicted_boundary_per_step"][step-1], "Attempt/prefix predicted boundary differs")
        if step in trace_slots:
            slot = trace_slots[step]
            history = traces["observed_or_predicted_histories"][slot]
            prediction, truth = traces["predicted_positions"][slot], np.asarray(positions[5+step], dtype=np.float64)
            audit.array(traces["ground_truth_positions"][slot], truth.astype(np.float32), "Trace truth differs from source")
            audit.equal(BASE.state_hash(prediction), attempt["predicted_state_sha256"], "Trace prediction hash differs")
            audit.scalar(row["mse_per_step"][step-1], float(np.mean((prediction.astype(np.float64)-truth)**2)), "Trace MSE arithmetic differs")
            audit.equal(BASE.boundary_metrics(prediction, traces["bounds"]), attempt["predicted_boundary"], "Trace predicted boundary differs")
            for history_slot in range(6):
                source_frame = step-1+history_slot
                if source_frame < 6:
                    audit.array(history[history_slot], positions[source_frame], "Trace initial history differs")
                else:
                    prior_forecast = source_frame-5
                    audit.equal(BASE.state_hash(history[history_slot]), attempts[prior_forecast-1]["predicted_state_sha256"], "Trace history not linked to accepted predicted state")
            cache = traces["cached_risk_before_forecast"][slot]
            current_risk = traces["current_risk"][slot]
            audit.check(np.isfinite(current_risk).all() and np.all(current_risk > 0), "Invalid current trace risk")
            if step > 1:
                audit.check(np.isfinite(cache).all() and np.all(cache > 0), "Invalid noninitial cached trace risk")
                audit.scalar(float(cache.mean()), row["mean_normalized_acceleration_variance_per_step"][step-2], "Trace cache does not match preceding risk mean")
            elif policy != "laggedrisk25":
                audit.check(np.isnan(cache).all(), "Initial unused cache sentinel differs")
            if policy == "laggedrisk25":
                audit.check(np.isfinite(cache).all() and np.all(cache > 0), "Cached policy has invalid trace score")
                if step == 1:
                    audit.array(cache, traces["warmup_risk"], "Initial cache differs from explicit warmup")
            audit_trace_graph(history, cache, rng, attempt, policy, traces[f"edges_forecast_{step:04d}"], traces[f"optional_pairs_forecast_{step:04d}"], audit)
            audit.scalar(row["mean_normalized_acceleration_variance_per_step"][step-1], float(traces["current_risk"][slot].mean()), "Trace risk mean differs")
        elif not attempt["accepted"] and "rejected_edges" in traces:
            cache = traces["failed_input_cached_risk"] if "failed_input_cached_risk" in traces else None
            audit_trace_graph(traces["rejected_history"], cache, rng, attempt, policy, traces["rejected_edges"], traces["rejected_optional_pairs"], audit)
        elif policy == "random25" and "available_annulus_pairs" in attempt:
            rng.permutation(attempt["available_annulus_pairs"])
    audit.equal(row["forecast_network_passes"], forward_calls, "Forward pass count differs from attempts")
    warmup = row["warmup"]
    warmup_passes = warmup.get("passes", 0) if warmup else 0
    audit.equal(row["total_network_passes"], forward_calls+warmup_passes, "Total policy pass count differs")
    if warmup:
        audit.equal(policy, "laggedrisk25", "Unexpected score warmup")
        audit.equal(warmup["history_sha256"], row["initial_observed_state_sha256"], "Warmup history differs")
        audit.check(warmup["mean_prediction_discarded"] is True, "Warmup mean was not discarded")
    graph_seconds = sum(item.get("graph_build_and_selection_seconds", 0.) for item in attempts) + (warmup.get("graph_build_and_selection_seconds", 0.) if warmup else 0.)
    forward_seconds = sum(item.get("features_forward_decode_and_transfer_seconds", 0.) for item in attempts) + (warmup.get("features_forward_decode_and_transfer_seconds", 0.) if warmup else 0.)
    audit.scalar(row["graph_build_and_selection_seconds_including_warmup"], graph_seconds, "Graph component total differs")
    audit.scalar(row["features_forward_decode_and_transfer_seconds_including_warmup"], forward_seconds, "Forward component total differs")
    audit.check(row["total_wall_seconds_including_native_parity"] >= row["total_wall_seconds"] >= 0, "Wall time scopes inconsistent")
    audit.check(row["native_parity_operational_seconds"] >= 0, "Negative native verification duration")
    if completed in trace_slots:
        final_state = traces["predicted_positions"][trace_slots[completed]]
        audit.equal(BASE.state_hash(final_state), row["final_predicted_state_sha256"], "Final predicted state hash differs")
    elif row["status"] == "failed":
        final_state = traces["failed_input_history"][-1] if "failed_input_history" in traces else traces["initial_observed_positions"][-1]
        audit.equal(BASE.state_hash(final_state), row["final_predicted_state_sha256"], "Failed-prefix final state hash differs")
    validate_failure(row, traces, audit)
    native_prefix = {"mean_"+name: float(np.mean([item[name] for item in accepted])) if completed else None for name in FULL_NATIVE_METRICS}
    native_prefix["cap_active_step_fraction"] = float(np.mean([item["native_base_receivers_above_cap_before_capping"] > 0 for item in accepted])) if completed else None
    full_metrics = native_prefix if row["status"] == "complete" else {key: None for key in native_prefix}
    full_metrics.update(mean_wall_seconds_including_native_parity=row["total_wall_seconds_including_native_parity"],
        mean_native_parity_seconds=row["native_parity_operational_seconds"], mean_native_parity_passes=row["native_parity_network_passes"],
        mean_graph_operational_seconds=graph_seconds, mean_forward_component_seconds=forward_seconds)
    return full_metrics, native_prefix if row["status"] == "failed" else None


def load_native(directory, objective, seed, research_hash, audit):
    directory = Path(directory)
    protocol, protocol_hash = original.read_json(directory / "protocol.json")
    result, result_hash = original.read_json(directory / "result.json")
    status, status_hash = original.read_json(directory / "status.json")
    audit.equal(result["state"], "complete", "Native evaluation incomplete")
    audit.equal(status["state"], "complete", "Native completion uncommitted")
    audit.equal(status["result_sha256"], result_hash, "Native result hash differs")
    audit.equal(result["protocol_sha256"], protocol_hash, "Native protocol hash differs")
    native_contract_hash = validate_native_contract(protocol, audit)
    audit.equal(protocol["native_bridge_protocol_sha256"], BASE.sha256(native.PROTOCOL), "Native scientific protocol differs")
    audit.equal(protocol["source_indices"], list(range(3, 30)), "Native source coverage differs")
    audit.equal(protocol["policies"], list(native.POLICIES), "Native policy coverage differs")
    audit.equal(protocol["horizon"], 995, "Native full horizon differs")
    audit.equal(protocol["guards"], {"max_candidate_pairs": 100000, "max_abs_coordinate": 10.}, "Native resource guards differ")
    audit.equal(protocol["trace_steps"], list(original.TRACE_STEPS), "Native trace protocol differs")
    for key, value in (("objective", objective), ("seed", seed)):
        audit.equal(protocol[key], value, "Native model identity differs")
        audit.equal(result[key], value, "Native result identity differs")
    audit.equal(result["checkpoint_sha256"], protocol["checkpoint_sha256"], "Native checkpoint identity differs")
    manifest_paths = [Path(path) for path in protocol["input_files_sha256"] if Path(path).name == "test.json"]
    audit.equal(len(manifest_paths), 1, "One pinned official test manifest required")
    manifest_path = manifest_paths[0]
    manifest = json.loads(manifest_path.read_text())
    trajectories = BASE.load_manifest_data(manifest_path, verify_hashes=True)
    metadata_path = manifest_path.parent / "metadata.json"
    view = checkpoint_validation_view(protocol, manifest, BASE.sha256(manifest_path), BASE.sha256(metadata_path), research_hash)
    checkpoint_common_hash = original.validate_protocol(view, objective, seed, research_hash)
    common_identity_hash = original.canonical_hash({"native_contract": native_contract_hash, "original_checkpoint_contract": checkpoint_common_hash})
    audit.equal(original.canonical_hash(protocol["checkpoint_provenance"]["training_config"]["full_run"]), protocol["run_config_sha256"], "Original training config hash differs")
    expected = {(index, policy) for index in range(3, 30) for policy in native.POLICIES}
    actual = {(row["source_index"], row["policy"]) for row in result["records"]}
    audit.equal(actual, expected, "Native outcome coverage differs")
    audit.equal(len(result["records"]), 135, "Duplicate native outcomes")
    rows, hashes = [], {str((directory/name).resolve()): digest for name, digest in (("protocol.json", protocol_hash), ("result.json", result_hash), ("status.json", status_hash))}
    for compact in result["records"]:
        verified = original.validate_record(directory, compact, view, protocol_hash)
        audit.check(True, "Original strict scalar/boundary validator passed")
        row_path, trace_path = directory/compact["record_file"], directory/compact["trace_file"]
        row = original.read_json(row_path)[0]
        positions, types = trajectories[row["source_index"]]
        if np.asarray(types).ndim == 0:
            types = np.full(positions.shape[1], types, dtype=np.int64)
        with np.load(trace_path, allow_pickle=False) as traces:
            audit.array(traces["bounds"], np.asarray(manifest["metadata"]["bounds"], dtype=np.float64), "Trace bounds differ from official metadata")
            native_metrics, native_prefix = validate_trajectory(row, traces, positions, types, audit)
            initial_hash = BASE.state_hash(traces["initial_observed_positions"])
        hashes[str(row_path.resolve())], hashes[str(trace_path.resolve())] = compact["record_sha256"], compact["trace_sha256"]
        rows.append({**verified, "native_metrics": native_metrics, "native_prefix_diagnostics_if_failed": native_prefix,
            "native_parity": row["native_parity"], "initial_observed_state_sha256": initial_hash,
            "operational_time_scope": "Policy total includes input transfers and diagnostics, excludes file publication/native parity; native parity separately included in total-with-parity. Forward component excludes input preparation/transfer; graph component includes audits."})
    return {"objective": objective, "seed": seed, "scope": protocol["scope"], "eligible_for_aggregation": True,
        "state": "complete", "checkpoint_sha256": protocol["checkpoint_sha256"], "records": rows,
        "common_identity_sha256": common_identity_hash,
        "protocol_sha256": protocol_hash, "input_files_sha256": hashes, "source_protocol": protocol,
        "per_policy": {policy: summarize_seed(rows, policy) for policy in native.POLICIES}}


def summarize(native_runs, original_runs, audit):
    new = {(run["objective"], run["seed"]): run for run in native_runs}
    old = {(run["objective"], run["seed"]): run for run in original_runs}
    expected = {(objective, seed) for objective in ("faithful", "nll") for seed in (0, 1, 2)}
    audit.equal(set(new), expected, "Native six-model coverage differs")
    audit.equal(set(old), expected, "Original six-model coverage differs")
    audit.equal(len(native_runs), 6, "Duplicate native models")
    audit.equal(len(original_runs), 6, "Duplicate original models")
    audit.equal(len({run["checkpoint_sha256"] for run in native_runs}), 6, "Six distinct original checkpoints required")
    audit.equal(len({run["common_identity_sha256"] for run in native_runs}), 1, "Native common protocol identity differs across models")
    audit.equal(len({run["common_identity_sha256"] for run in original_runs}), 1, "Original common protocol identity differs across models")
    for key in expected:
        audit.check(old[key]["eligible_for_aggregation"], "Original locked outcome invalid/incomplete")
        audit.equal(new[key]["checkpoint_sha256"], old[key]["checkpoint_sha256"], "Convention comparison changed checkpoint")
    groups = {}
    for objective in ("faithful", "nll"):
        runs, previous = [new[objective, seed] for seed in (0, 1, 2)], [old[objective, seed] for seed in (0, 1, 2)]
        groups[objective] = {"policies": {policy: {"per_seed": [run["per_policy"][policy] for run in runs],
            "metrics": {name: original.across_seeds([run["per_policy"][policy]["metrics"][name] for run in runs]) for name in METRICS}}
            for policy in native.POLICIES},
            "native_policy_comparisons": {left+"_minus_"+right: paired(runs, runs, left, right) for left, right in original.POLICY_COMPARISONS},
            "native_minus_original": {policy: paired(runs, previous, policy, policy, original.METRICS) for policy in native.POLICIES}}
    return groups


def render(report):
    lines = ["# Exploratory native-graph autonomous bridge", "", "All original six 100k models and 27 trajectories/policy are retained. This is a post-inspection native-convention comparison, not an isolated self-loop effect or independent confirmation. Any required failure makes the corresponding full-horizon mean undefined.", ""]
    def fmt(value):
        return "undefined" if value["mean"] is None else f"{value['mean']:.8g} ± {value['sample_sd']:.5g}"
    lines += ["| Policy | Faithful full-horizon MSE | NLL full-horizon MSE |", "|---|---:|---:|"]
    for policy in native.POLICIES:
        values = [report["groups"][objective]["policies"][policy]["metrics"]["mean_rollout_mse"] for objective in ("faithful", "nll")]
        lines.append(f"| {policy} | {fmt(values[0])} | {fmt(values[1])} |")
    lines += ["", "Native base edges retain capped directed order and self-edge candidates; optional annulus messages are appended uncapped. This jointly changes graph semantics relative to the original experiment. Measured durations include diagnostics and do not establish speedup. Every paired seed/trajectory comparison, boundary reference, failed prefix, cap/asymmetry diagnostic and timing scope is saved in JSON.", "", f"Saved arithmetic/identity checks passed: {report['audit']['checks']}. Original strict scalar/boundary validators also passed for every native row."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Preserve all prior analysis attempts; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_hash = BASE.sha256(__file__)
    source_files = [Path(__file__), Path(original.__file__), Path(__file__).parent/"summarize_graph_convention_bridge.py"]
    analysis_hashes = {str(path.resolve()): BASE.sha256(path) for path in source_files}
    BASE.atomic_json(args.output_dir/"analysis_source.json", {"summary_source_sha256": source_hash, "analysis_source_files_sha256": analysis_hashes,
                     "started_utc": datetime.now(timezone.utc).isoformat()})
    audit = Audit()
    try:
        research_hash = BASE.sha256(BASE.REPO/"research/protocols/full_waterdrop_100k.md")
        native_runs, old_runs = [], []
        for objective in ("faithful", "nll"):
            for seed in (0, 1, 2):
                name = f"{objective}_seed{seed}"
                native_runs.append(load_native(args.native_root/name, objective, seed, research_hash, audit))
                old_run = original.load_run(args.original_root/name, objective, seed, research_hash)
                audit.check(old_run["eligible_for_aggregation"], "Original run invalid or incomplete")
                old_status, old_status_hash = original.read_json(args.original_root/name/"status.json")
                audit.equal(old_status.get("state"), "complete", "Original completion status changed")
                audit.equal(old_status.get("result_sha256"), old_run["result_sha256"], "Original result completion evidence changed")
                old_run["status_sha256"] = old_status_hash
                old_runs.append(old_run)
        groups = summarize(native_runs, old_runs, audit)
        pinned = {}
        for run in native_runs:
            for path, digest in {**run["source_protocol"]["input_files_sha256"], **run["input_files_sha256"]}.items():
                if path in pinned:
                    audit.equal(pinned[path], digest, "Shared input hashes disagree")
                pinned[path] = digest
        for run in old_runs:
            directory = Path(run["directory"])
            pinned[str((directory/"protocol.json").resolve())] = run["protocol_sha256"]
            pinned[str((directory/"result.json").resolve())] = run["result_sha256"]
            pinned[str((directory/"status.json").resolve())] = run["status_sha256"]
            for row in run["records"]:
                pinned[str((directory/row["record_file"]).resolve())] = row["record_sha256"]
                pinned[str((directory/row["trace_file"]).resolve())] = row["trace_sha256"]
        pinned.update(analysis_hashes)
        for path, digest in pinned.items():
            audit.equal(BASE.sha256(path), digest, "Pinned input/source bytes changed")
        audit.equal(BASE.sha256(__file__), source_hash, "Summary source changed during execution")
        report = {"schema": 1, "scope": "post-inspection native convention follow-up on original six100k models",
            "summary_source_sha256": source_hash, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_source_files_sha256": analysis_hashes,
            "groups": groups, "native_runs": native_runs, "original_runs": old_runs,
            "audit": {"passed": True, "checks": audit.checks},
            "limits": "Predicted per-step scalar records are fully checked for arithmetic/domain consistency; saved trace predictions/graphs are reconstructed and linked to state hashes. Unstored intermediate model outputs cannot be independently regenerated without inference. No new inference occurs here."}
        BASE.atomic_json(args.output_dir/"native_graph_rollouts.json", report)
        (args.output_dir/"native_graph_rollouts.md").write_text(render(report))
    except BaseException as error:
        BASE.atomic_json(args.output_dir/"failed_analysis.json", {"passed": False, "checks_before_failure": audit.checks,
            "summary_source_sha256": source_hash, "summary_source_sha256_at_failure": BASE.sha256(__file__),
            "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
