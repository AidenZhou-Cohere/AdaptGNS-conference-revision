"""Separate endpoint evaluation of the six faithful graph-support continuations.

The original 100k validators are used only for immutable parent checkpoints.
Completed 110k endpoints use the new lineage gate. No inference at import.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from research import faithful_graph_support as support
from research import graph_convention_bridge as bridge
from research import native_graph_rollout as native

full = native.full
PROTOCOL = Path(__file__).parent / "protocols/continuation_evaluation_20261005.md"
TRAINING_PROTOCOL = full.REPO / "research/protocols/full_waterdrop_100k.md"
TRAINER_SHA256 = "71c2b3740aab635665d28b37addbd0b89360cccb3136d275e54b864d3b2a7eaa"
TRAINER_PROTOCOL_SHA256 = "90f896adefedf6e63503835db2479ac7e4f357043c01918b3334a846a8ffbba3"
POLICIES = ("base", "dense", "random25", "speed25", "previous-observed-base-risk25")
COHORT = {(arm, seed) for arm in ("base", "mix") for seed in (0, 1, 2)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(pins):
    for path, digest in pins.items():
        require(full.sha256(path) == digest, f"Pinned input/source byte identity differs: {path}")


def merge_pins(target, pins):
    for name, digest in pins.items():
        path = str(Path(name).resolve())
        require(path not in target or target[path] == digest, "Conflicting immutable input hashes")
        target[path] = digest


def validate_cohort_manifest(document, directory):
    require(document.get("schema") == 1 and document.get("scope") == "faithful_graph_support_110k_endpoints",
            "A separate complete110k cohort manifest is required")
    entries = document.get("endpoints", [])
    require(len(entries) == 6 and {(row.get("arm"), row.get("seed")) for row in entries} == COHORT,
            "Exactly base/mix at all three original seeds are required")
    result = {}
    for row in entries:
        require(type(row["seed"]) is int and isinstance(row.get("sha256"), str)
                and len(row["sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["sha256"]),
                "Canonical endpoint hash and integer seed required")
        path = Path(row["checkpoint"])
        path = (Path(directory) / path).resolve() if not path.is_absolute() else path.resolve()
        result[row["arm"], row["seed"]] = {**row, "checkpoint": str(path)}
    require(len({row["checkpoint"] for row in result.values()}) == 6
            and len({row["sha256"] for row in result.values()}) == 6,
            "Six distinct checkpoint files and byte identities required")
    return result


def check_endpoint(payload, entry):
    config, completed = support.validate_continuation_payload(payload, require_complete=True)
    require(config["arm"] == entry["arm"] and config["seed"] == entry["seed"] and completed == 10000,
            "Endpoint arm/original seed differs from cohort")
    required = {"particle_dimensions": 2, "latent_dim": 128, "nmessage_passing_steps": 10,
                "nmlp_layers": 2, "mlp_hidden_dim": 128, "max_num_neighbors": 128,
                "connectivity_radius": .015, "radius_backend": "scipy_host"}
    require(all(payload["simulator_config"].get(key) == value for key, value in required.items()),
            "Full native128x10 endpoint architecture required")
    pins = config["input_files_sha256"]
    for path, digest in ((Path(support.__file__), TRAINER_SHA256), (support.PROTOCOL, TRAINER_PROTOCOL_SHA256)):
        require(pins.get(str(path.resolve())) == digest and full.sha256(path) == digest,
                "Frozen continuation trainer/protocol identity differs")
    return config


def check_committed_endpoint(entry, config):
    directory = Path(entry["checkpoint"]).parent
    paths = [directory / name for name in ("latest.json", "status.json", "protocol.json")]
    latest, status, saved_config = (json.loads(path.read_text()) for path in paths)
    require(Path(entry["checkpoint"]).name == "checkpoint-extra-10000.pt"
            and latest.get("path") == Path(entry["checkpoint"]).name and latest.get("sha256") == entry["sha256"]
            and latest.get("completed_additional_updates") == 10000 and latest.get("completed_total_updates") == 110000,
            "Endpoint must be the committed latest fixed10000-additional checkpoint")
    require(saved_config == config and status.get("state") == "complete" and status.get("latest_checkpoint") == latest
            and status.get("completed_additional_updates") == 10000 and status.get("completed_total_updates") == 110000
            and status.get("config_sha256") == support.original.config_hash(config),
            "Endpoint run publication must be complete with matching configuration and pointer")
    return {str(path.resolve()): full.sha256(path) for path in paths}


def optimizer_signature(payload):
    optimizer = payload["optimizer_state"]
    groups = copy.deepcopy(optimizer["param_groups"])
    states = {}
    for key, state in optimizer["state"].items():
        step = state.get("step")
        require(torch.is_tensor(step) and step.ndim == 0 and bool(torch.isfinite(step))
                and float(step) == int(float(step)), "Finite integer Adam update counter required")
        require({"step", "exp_avg", "exp_avg_sq"} <= set(state), "Complete inherited Adam moments required")
        require(all(not torch.is_tensor(value) or bool(torch.isfinite(value).all()) for value in state.values()),
                "Nonfinite endpoint Adam state")
        states[key] = {"step": int(float(step)), "tensor_shapes": {
            name: (list(value.shape), str(value.dtype)) for name, value in state.items() if torch.is_tensor(value)}}
    require(states, "Trained Adam state is required")
    weights = payload["state_dict"]
    require(all(torch.is_tensor(value) and bool(torch.isfinite(value).all()) for value in weights.values()),
            "Finite model state tensors required")
    return {"groups": groups, "states": states,
            "weights": {name: (list(value.shape), str(value.dtype)) for name, value in weights.items()}}


def validate_optimizer_lineage(parent, child):
    require(parent["weights"] == child["weights"] and set(parent["states"]) == set(child["states"]),
            "Endpoint model/Adam parameter identity differs from parent")
    require(len(parent["groups"]) == len(child["groups"]), "Adam group count differs")
    for before, after in zip(parent["groups"], child["groups"]):
        require(after["lr"] == support.LR and {key: value for key, value in before.items() if key != "lr"}
                == {key: value for key, value in after.items() if key != "lr"}, "Inherited Adam group configuration differs")
    for key, before in parent["states"].items():
        after = child["states"][key]
        require(before["step"] == 100000 and after["step"] == before["step"] + 10000
                and after["tensor_shapes"] == before["tensor_shapes"],
                "Adam counters/moment shapes do not establish100k-to110k lineage")


def validate_parent_identity(parent, config, metadata_sha256):
    # Deliberately applies the original validator only to the original parent.
    full.check_checkpoint(parent, "locked_test", TRAINING_PROTOCOL, metadata_sha256)
    run = parent["run_config"]
    require(run["objective"] == "faithful" and run["seed"] == config["seed"]
            and parent["run_config_sha256"] == config["parent_run_config_sha256"]
            and parent["training_config"] == config["parent_training_config"], "Immutable parent training identity differs")
    require(support.simulator_config_sha256(parent["simulator_config"]) == config["parent_simulator_config_sha256"],
            "Parent simulator/normalization hash differs")
    for key in ("train", "valid", "validation_frames"):
        require(run[key] == config[key], f"Original parent {key} identity differs")
    for relative, digest in run["source_sha256"].items():
        path = str((full.REPO / relative).resolve())
        require(config["input_files_sha256"].get(path) == digest and full.sha256(path) == digest,
                "Original frozen training source identity differs")


def load_verified_cohort(path, metadata_sha256):
    """Read all six endpoints and three parents, release each payload promptly."""
    path = Path(path)
    entries = validate_cohort_manifest(json.loads(path.read_text()), path.parent)
    configs, signatures, schedule_hashes, pins = {}, {}, {}, {str(path.resolve()): full.sha256(path)}
    for identity in sorted(entries):
        entry = entries[identity]
        require(full.sha256(entry["checkpoint"]) == entry["sha256"], "Endpoint checkpoint byte hash differs")
        payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
        config = check_endpoint(payload, entry)
        configs[identity] = copy.deepcopy(config)
        signatures[identity] = optimizer_signature(payload)
        schedule_hashes[identity] = support.original.config_hash([
            {key: row[key] for key in ("absolute_schedule_step", "frame_ids", "noise_sha256")}
            for row in payload["history"]["graph_updates"]])
        merge_pins(pins, config["input_files_sha256"])
        merge_pins(pins, check_committed_endpoint(entry, config))
        merge_pins(pins, {entry["checkpoint"]: entry["sha256"]})
        del payload
    verify_files(pins)
    for seed in (0, 1, 2):
        left, right = configs["base", seed], configs["mix", seed]
        require({key: value for key, value in left.items() if key != "arm"}
                == {key: value for key, value in right.items() if key != "arm"},
                "Paired arms differ beyond the declared graph exposure")
        require(schedule_hashes["base", seed] == schedule_hashes["mix", seed], "Paired frame/noise histories differ")
        parent_paths = [name for name, digest in left["input_files_sha256"].items()
                        if digest == support.PARENT_HASHES[seed]]
        require(len(parent_paths) == 1, "One byte-pinned immutable parent required per seed")
        parent = torch.load(parent_paths[0], map_location="cpu", weights_only=True)
        validate_parent_identity(parent, left, metadata_sha256)
        before = optimizer_signature(parent)
        for arm in ("base", "mix"):
            validate_optimizer_lineage(before, signatures[arm, seed])
        del parent
    common = ("train", "valid", "validation_frames", "graph", "additional_updates", "lr", "batch_size", "noise_std")
    for config in configs.values():
        require(all(config[key] == configs["base", 0][key] for key in common),
                "Cohort common data/schedule/graph/optimization contract differs")
    return entries, configs, pins


def random_material(seed, identity):
    require(seed in (0, 1, 2) and identity["split"] in ("valid", "test"), "Known original seed/split required")
    return [20261005, 771, seed, 0 if identity["split"] == "valid" else 1,
            identity["source_index"], identity["target_frame"]]


@torch.no_grad()
def evaluate_frame(model, current, previous, types, target, identity, seed, device):
    """Five matching native-prefix policies; future target enters after inference."""
    started = time.perf_counter()
    current, previous = np.asarray(current, dtype=np.float32), np.asarray(previous, dtype=np.float32)
    types = np.asarray(types, dtype=np.int64)
    require(current.shape == previous.shape and current.ndim == 3 and current.shape[0] == 6
            and current.shape[2] == 2, "Two matching six-frame 2D histories required")
    require(types.shape == (current.shape[1],) and not np.any(types == 3), "One nonkinematic type per particle required")
    n = len(types)
    material = random_material(seed, identity)
    rng = lambda: np.random.default_rng(np.random.SeedSequence(material))
    std = bridge.same.normalization(model)["std"]
    arrays = {"current_history": current.copy(), "previous_history": previous.copy(), "particle_types": types.copy(),
              "acceleration_std": std, "bounds": np.asarray(model._boundaries, dtype=np.float64)}
    row = {**identity, "status": "running", "failure": None, "n_particles": n, "cases": {},
           "random_seed_material": material, "current_history_sha256": full.state_hash(current),
           "previous_history_sha256": full.state_hash(previous), "native_parity": None, "previous_score": None}
    def failed_all(failure):
        row.update(status="failed", failure=failure, operational_seconds=time.perf_counter() - started)
        row["cases"] = {name: {"status": "failed", "failure": failure, "metrics": None} for name in POLICIES}
        return row, arrays
    phase = "current_history"
    try:
        full.check_state(current, phase, bridge.MAX_ABS)
        phase = "current_graph"
        graph, base_edges, _, base_audit = native.native_graph(current, "base", None, rng())
        arrays.update(strict_base_pairs=graph.base, strict_annulus_pairs=graph.extra)
        row["optional_budget"] = int(.25 * len(graph.extra))
        phase = "native_parity"
        parity, parity_arrays, base_output = bridge.native_parity(model, current, types, base_edges, device)
        row["native_parity"] = parity
        arrays.update(parity_arrays)
        for key in ("prediction", "raw_risk", "risk"):
            arrays[f"supplied_parity_{key}"] = base_output[key]
        if not parity["passed"]:
            return failed_all({"category": "native_parity_failure", "phase": phase})
    except full.RolloutGuard as error:
        return failed_all({**error.details, "phase": phase})

    previous_output, score_failure = None, None
    phase = "previous_history"
    try:
        full.check_state(previous, phase, bridge.MAX_ABS)
        phase = "previous_graph"
        _, old_edges, _, old_audit = native.native_graph(previous, "base", None, rng())
        arrays["previous_base_edges"] = old_edges
        phase = "previous_forward"
        previous_output = bridge.supplied(model, previous, types, old_edges, device)
        for key in ("prediction", "raw_risk", "risk"):
            arrays[f"previous_base_{key}"] = previous_output[key]
        bridge.validate_output(previous_output, n)
        row["previous_score"] = {"status": "complete", "failure": None, "graph": old_audit,
            "operational_seconds": previous_output["operational_seconds"]}
    except full.RolloutGuard as error:
        score_failure = {**error.details, "phase": phase}
        row["previous_score"] = {"status": "failed", "failure": score_failure,
            "operational_seconds": previous_output["operational_seconds"] if previous_output else None}

    outputs = {}
    for policy in POLICIES:
        if policy == "previous-observed-base-risk25" and score_failure:
            row["cases"][policy] = {"status": "failed", "failure": {"category": "scoring_pass_failed", "cause": score_failure},
                "metrics": None, "network_passes_for_standalone_policy": 1 if previous_output else 0}
            continue
        phase, graph_seconds = "graph", None
        case = {"status": "running", "failure": None, "metrics": None}
        try:
            graph_start = time.perf_counter()
            actual = "laggedrisk25" if policy == "previous-observed-base-risk25" else policy
            score = previous_output["risk"] if actual == "laggedrisk25" else None
            _, edges, optional, audit = native.native_graph(current, actual, score, rng())
            graph_seconds = time.perf_counter() - graph_start
            require(np.array_equal(edges[:, :base_edges.shape[1]], base_edges), "Policy changed native mandatory prefix")
            arrays[f"edges__{policy}"], arrays[f"optional_pairs__{policy}"] = edges, optional
            if score is not None:
                arrays[f"selection_score__{policy}"] = score
            elif policy == "speed25":
                arrays[f"selection_score__{policy}"] = np.linalg.norm(current[-1] - current[-2], axis=-1)
            phase = "forward"
            output = base_output if policy == "base" else bridge.supplied(model, current, types, edges, device)
            outputs[policy] = output
            for key in ("prediction", "raw_risk", "risk"):
                arrays[f"{key}__{policy}"] = output[key]
            case.update(graph=audit, graph_build_and_selection_seconds=graph_seconds,
                forward_operational_seconds=output["operational_seconds"],
                network_passes_for_standalone_policy=2 if score is not None else 1,
                scoring_seconds=previous_output["operational_seconds"] if score is not None else 0.,
                reused_native_parity_output=policy == "base")
            bridge.validate_output(output, n)
            case["status"] = "complete"
        except full.RolloutGuard as error:
            case.update(status="failed", failure={**error.details, "phase": phase})
        row["cases"][policy] = case

    # Future labels and all residuals are deliberately downstream of every call.
    target = np.asarray(target, dtype=np.float64)
    require(target.shape == current[-1].shape and np.isfinite(target).all(), "Finite matching target required")
    arrays["target_position"] = target
    row["target_sha256"] = full.state_hash(target)
    row["ground_truth_boundary"] = full.boundary_metrics(target, arrays["bounds"])
    row["current_observed_boundary"] = full.boundary_metrics(current[-1], arrays["bounds"])
    for policy, output in outputs.items():
        if np.isfinite(output["prediction"]).all():
            row["cases"][policy]["predicted_boundary"] = full.boundary_metrics(output["prediction"], arrays["bounds"])
        if row["cases"][policy]["status"] == "complete":
            metric, residual, normalized = bridge.residual_record(output, target, std)
            row["cases"][policy]["metrics"] = metric
            arrays[f"position_residual__{policy}"], arrays[f"normalized_residual__{policy}"] = residual, normalized
            arrays[f"normalized_coordinate_se__{policy}"] = np.mean(normalized ** 2, axis=-1)
    row["benefits"], row["previous_risk_correlations"] = {}, {}
    for policy in POLICIES:
        case = row["cases"][policy]
        if row["cases"]["base"]["status"] != "complete" or case["status"] != "complete":
            row["benefits"][policy] = {"mean_signed_normalized_coordinate_gain": None,
                "reason": "required base or action output failed"}
            continue
        gain = arrays["normalized_coordinate_se__base"] - arrays[f"normalized_coordinate_se__{policy}"]
        arrays[f"signed_normalized_coordinate_gain__{policy}"] = gain
        row["benefits"][policy] = {"mean_signed_normalized_coordinate_gain": float(gain.mean()),
            "harmful_particle_fraction": float(np.mean(gain < 0))}
        if not score_failure:
            row["previous_risk_correlations"][policy] = bridge.same.spearman(previous_output["risk"], gain)
    if not score_failure and row["cases"]["base"]["status"] == "complete":
        row["previous_risk_base_residual_correlation"] = bridge.same.spearman(
            previous_output["risk"], arrays["normalized_coordinate_se__base"])
    else:
        row["previous_risk_base_residual_correlation"] = {"value": None, "reason": "required scoring or base output failed"}
    failures = [{"case": name, "failure": case["failure"]} for name, case in row["cases"].items() if case["status"] == "failed"]
    row.update(status="failed" if failures else "complete", failure={"category": "case_failures", "cases": failures} if failures else None,
               operational_seconds=time.perf_counter() - started)
    return row, arrays


def preflight_output(directory):
    require(not list(Path(directory).rglob("*.tmp")), "Temporary output requires preservation/review before any write")


def archive_resume_status(args):
    """Retain exact preceding status bytes before a resumed writer replaces it."""
    status = args.output_dir / "status.json"
    if args.resume and status.exists():
        directory = args.output_dir / "attempt_history" / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "status.json").open("xb") as stream:
            stream.write(status.read_bytes()); stream.flush(); os.fsync(stream.fileno())
        full.atomic_json(directory / "archive.json", {"captured_utc": bridge.utc_now(),
            "source": "status.json", "sha256": full.sha256(directory / "status.json")})


def run_observed(args, protocol, model, trajectories, expected, saved_lookup):
    directory = args.output_dir
    preflight_output(directory)
    require(protocol["expected_frames"] == expected, "Observed schedule differs from hashed protocol")
    protocol_path, result_path = directory / "protocol.json", directory / "result.json"
    if protocol_path.exists():
        require(args.resume and json.loads(protocol_path.read_text()) == protocol, "Resume requires exact original protocol")
    else:
        require(not args.resume and not any(path.name != "run.lock" for path in directory.iterdir()),
                "Existing unowned output or missing resume protocol requires review")
        full.atomic_json(protocol_path, protocol)
    digest = full.sha256(protocol_path)
    indexed = {}
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        require(previous["protocol_sha256"] == digest, "Existing observed result protocol differs")
        indexed = {row["record_file"]: row for row in previous["records"]}
        require(len(indexed) == len(previous["records"]), "Duplicate observed index rows")
    records, recovered = [], {}
    for item in expected:
        record = bridge.recover(directory, item, digest)
        if record:
            saved_row = json.loads((directory / record["record_file"]).read_text())
            require(saved_row.get("arm") == protocol["arm"] and saved_row.get("original_seed") == protocol["seed"]
                    and saved_row.get("checkpoint_sha256") == protocol["checkpoint_sha256"]
                    and saved_row.get("parent_checkpoint_sha256") == protocol["parent_checkpoint_sha256"],
                    "Recovered observed row lineage differs")
            require(args.resume and (record["record_file"] not in indexed or indexed[record["record_file"]] == record),
                    "Recovered observed row/index differs or explicit resume missing")
            recovered[bridge.stem(item)] = record
            records.append(record)
    require(set(indexed) <= {row["record_file"] for row in records}, "Missing/unexpected indexed observed frame")
    archive_resume_status(args)
    def publish_state(state, **extra):
        full.atomic_json(directory / "status.json", {"state": state, "pid": os.getpid(),
            "arm": protocol["arm"], "seed": protocol["seed"], "committed_frames": len(records),
            "updated_utc": bridge.utc_now(), **extra})
    try:
        publish_state("running")
        for item in expected:
            old = recovered.get(bridge.stem(item))
            if old:
                if old["failure"] and old["failure"].get("category") == "native_parity_failure":
                    raise RuntimeError("Retained native parity failure requires review; no retry")
                continue
            if datetime.now(timezone.utc) >= datetime.fromisoformat(native.DEADLINE):
                raise TimeoutError("Research cutoff reached before next observed frame")
            current, previous, types, target = bridge.frame_input(item, trajectories, args.saved_same_state_dir, saved_lookup)
            row, arrays = evaluate_frame(model, current, previous, types, target, item, protocol["seed"], args.device)
            array = directory / (bridge.stem(item) + ".npz")
            with array.with_suffix(".npz.tmp").open("xb") as stream:
                np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
            array.with_suffix(".npz.tmp").replace(array)
            row.update(arm=protocol["arm"], original_seed=protocol["seed"], checkpoint_sha256=protocol["checkpoint_sha256"],
                parent_checkpoint_sha256=protocol["parent_checkpoint_sha256"], protocol_sha256=digest,
                array_file=array.name, array_sha256=full.sha256(array))
            path = array.with_suffix(".json")
            full.atomic_json(path, row)
            records.append(bridge.record_index(row, path))
            full.atomic_json(result_path, {"state": "partial", "protocol_sha256": digest,
                "arm": protocol["arm"], "seed": protocol["seed"], "records": records})
            publish_state("running", last_frame=item, last_status=row["status"])
            print(json.dumps({**item, "arm": protocol["arm"], "seed": protocol["seed"], "status": row["status"]}), flush=True)
            if row["failure"] and row["failure"].get("category") == "native_parity_failure":
                raise RuntimeError("Native parity failed; unsuccessful frame committed for review")
        verify_files(protocol["input_files_sha256"])
        full.atomic_json(result_path, {"state": "complete", "scope": protocol["scope"], "protocol_sha256": digest,
            "arm": protocol["arm"], "seed": protocol["seed"], "objective": "faithful",
            "checkpoint_sha256": protocol["checkpoint_sha256"], "parent_checkpoint_sha256": protocol["parent_checkpoint_sha256"],
            "required_frames": len(expected), "records": records,
            "complete_frames": sum(row["status"] == "complete" for row in records),
            "failed_frames": sum(row["status"] == "failed" for row in records)})
        publish_state("complete", result_sha256=full.sha256(result_path))
    except BaseException as error:
        publish_state("deadline_stopped" if isinstance(error, TimeoutError) else "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                      error_type=type(error).__name__, error=str(error))
        raise


def run_autonomous(args, protocol, model, trajectories, manifest, selected):
    preflight_output(args.output_dir)
    require(selected == list(range(3, 30)) and protocol["source_indices"] == selected
            and protocol["horizon"] == 995 and protocol["policies"] == list(native.POLICIES),
            "Every official source3..29, five policies and995 forecasts are required")
    require(protocol["arm"] in ("base", "mix") and protocol["seed"] in (0, 1, 2), "Explicit arm/original seed required")
    identity_path = args.output_dir / "lineage_identity.json"
    identity = {key: protocol[key] for key in ("arm", "seed", "original_seed", "checkpoint_sha256",
                "parent_checkpoint_sha256", "continuation_config_sha256", "completed_total_updates", "completed_additional_updates")}
    if identity_path.exists():
        require(args.resume and json.loads(identity_path.read_text()) == identity, "Saved autonomous lineage identity differs")
    if args.resume:
        protocol_path = args.output_dir / "protocol.json"
        require(protocol_path.exists() and json.loads(protocol_path.read_text()) == protocol,
                "Native resume requires original exact protocol before status archival")
    archive_resume_status(args)
    # Frozen native.run owns scientific failures, traces, cutoff and recovery.
    # Its protocol hash binds the arm even though its compact index uses seed.
    native.run(args, protocol, model, trajectories, manifest, selected)
    if not identity_path.exists():
        full.atomic_json(identity_path, identity)


def paired_contrasts(records, expected_units, policies=POLICIES):
    """Differences first, equal trajectory weighting, then three-seed summaries.

    Each record has arm/seed/trajectory_id/unit_id and policy_values. Call
    separately for validation and test. Missing/failed required units remain
    undefined; no survival-only or favorable subset mean is returned.
    """
    expected = {(row["trajectory_id"], row["unit_id"]) for row in expected_units}
    require(len(expected) == len(expected_units) and expected, "Unique nonempty declared units required")
    require(set(policies) in (set(POLICIES), set(native.POLICIES)), "Exactly one predeclared five-policy family required")
    lookup = {}
    for row in records:
        key = row["arm"], row["seed"], row["trajectory_id"], row["unit_id"]
        require((row["arm"], row["seed"]) in COHORT and (row["trajectory_id"], row["unit_id"]) in expected
                and key not in lookup and set(row["policy_values"]) == set(policies), "Unknown/duplicate paired unit or policy")
        require(all(value is None or (isinstance(value, (int, float)) and np.isfinite(value))
                    for value in row["policy_values"].values()), "Finite value or explicit failed/missing null required")
        lookup[key] = row["policy_values"]
    risk = "previous-observed-base-risk25" if "previous-observed-base-risk25" in policies else "laggedrisk25"
    definitions = {f"mix_minus_base__{policy}": [("mix", policy, 1), ("base", policy, -1)] for policy in policies}
    definitions["risk_minus_random_interaction"] = [("mix", risk, 1), ("mix", "random25", -1),
                                                    ("base", risk, -1), ("base", "random25", 1)]
    for arm in ("base", "mix"):
        for label, left, right in (("risk_minus_random", risk, "random25"), ("risk_minus_speed", risk, "speed25"),
                                   ("dense_minus_base", "dense", "base")):
            definitions[f"{arm}__{label}"] = [(arm, left, 1), (arm, right, -1)]
    summaries = {}
    for name, terms in definitions.items():
        seeds = []
        for seed in (0, 1, 2):
            by_trajectory, missing = {}, 0
            for trajectory, unit in sorted(expected):
                values = [lookup.get((arm, seed, trajectory, unit), {}).get(policy) for arm, policy, _ in terms]
                if any(value is None for value in values):
                    missing += 1
                    continue
                difference = sum(value * term[2] for value, term in zip(values, terms))
                by_trajectory.setdefault(trajectory, []).append(difference)
            effect = float(np.mean([np.mean(values) for values in by_trajectory.values()])) if missing == 0 else None
            seeds.append({"seed": seed, "difference": effect, "required_units": len(expected), "missing_or_failed_units": missing})
        values = [row["difference"] for row in seeds]
        complete = all(value is not None for value in values)
        summaries[name] = {"seeds": seeds, "mean": float(np.mean(values)) if complete else None,
            "sample_sd": float(np.std(values, ddof=1)) if complete else None,
            "complete_seed_count": sum(value is not None for value in values)}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("observed", "autonomous"), required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "mix"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--saved-same-state-dir", type=Path, help="Original faithful100k saved test input archive; required for observed mode")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    args = parser.parse_args()
    require(args.device == "mps" and args.threads == 2 and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
            "Real endpoint evaluation requires two-thread MPS with explicit disabled fallback")
    require(args.mode != "observed" or args.saved_same_state_dir is not None, "Observed mode requires parent input archive")
    require(datetime.now(timezone.utc) < datetime.fromisoformat(native.DEADLINE), "Research cutoff reached; no new evaluation")
    metadata_path = args.validation_manifest.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    entries, configs, pins = load_verified_cohort(args.cohort_manifest, full.sha256(metadata_path))
    entry, config = entries[args.arm, args.seed], configs[args.arm, args.seed]
    expected, valid, test = bridge.expected_frames(config, args.validation_manifest, args.test_manifest)
    require(valid["metadata"] == test["metadata"] == metadata, "Original and evaluation metadata differ")
    selected = full.select_records(test, "locked_test")
    args.output_dir = args.output_dir.resolve()
    protected = [Path(row["checkpoint"]).parent.parent for row in entries.values()]
    protected += [args.validation_manifest.resolve().parent, args.test_manifest.resolve().parent]
    if args.saved_same_state_dir is not None:
        protected.append(args.saved_same_state_dir.resolve().parent)
    require(not any(args.output_dir == path or path in args.output_dir.parents or args.output_dir in path.parents for path in protected),
            "Separate endpoint evaluation tree outside checkpoint/data/parent result trees required")
    torch.set_num_threads(args.threads)
    device = full.resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with full.RunLock(args.output_dir, clear_stale=args.clear_stale_lock):
        preflight_output(args.output_dir)
        lookup, saved = {}, None
        if args.mode == "observed":
            # Only parent archive histories/targets/types are reused, never its predictions.
            lookup, saved = bridge.verify_saved_test(args.saved_same_state_dir, config["parent_checkpoint_sha256"], expected)
        trajectories = {"test": full.load_manifest_data(args.test_manifest, verify_hashes=True)}
        if args.mode == "observed":
            trajectories["valid"] = full.load_manifest_data(args.validation_manifest, verify_hashes=True)
        model, provenance = full.load_for_evaluation(entry["checkpoint"], metadata, device, radius_backend="scipy_host")
        require(support.simulator_config_sha256(model._checkpoint_config) == config["parent_simulator_config_sha256"],
                "Loaded model simulator/normalization differs from immutable parent")
        paths = [Path(__file__), Path(native.__file__), Path(bridge.__file__), Path(bridge.same.__file__),
            Path(full.__file__), Path(support.__file__), Path(support.original.__file__),
            full.REPO / "research/budget_graph.py", args.protocol, native.PROTOCOL, bridge.PROTOCOL,
            support.PROTOCOL, TRAINING_PROTOCOL, args.validation_manifest, args.test_manifest, metadata_path]
        paths += sorted((full.REPO / "adaptive-gns/gns").glob("*.py"))
        for manifest, directory in ((valid, args.validation_manifest.parent), (test, args.test_manifest.parent)):
            for record in manifest["records"]:
                paths += [directory / record[key]["path"] for key in ("positions", "particle_types")]
        if saved is not None:
            paths += [args.saved_same_state_dir / "result.json", args.saved_same_state_dir / "protocol.json"]
            for row in lookup.values():
                paths += [args.saved_same_state_dir / row["array_file"], args.saved_same_state_dir / row["record_file"]]
        merge_pins(pins, {str(path.resolve()): full.sha256(path) for path in paths})
        protocol = {"schema": 1, "scope": "post-inspection exploratory faithful110k graph-support endpoint evaluation",
            "mode": args.mode, "arm": args.arm, "seed": args.seed, "original_seed": args.seed, "objective": "faithful",
            "checkpoint_sha256": entry["sha256"], "checkpoint_provenance": provenance,
            "parent_checkpoint_sha256": config["parent_checkpoint_sha256"],
            "continuation_config_sha256": support.original.config_hash(config), "cohort": list(entries.values()),
            "completed_total_updates": 110000, "completed_additional_updates": 10000,
            "expected_frames": expected if args.mode == "observed" else None,
            "trajectory_ids": [test["records"][index]["id"] for index in selected], "source_indices": selected,
            "source_tfrecord": test["source"], "horizon": 995 if args.mode == "autonomous" else 1,
            "policies": list(POLICIES if args.mode == "observed" else native.POLICIES),
            "graph": "unchanged strict-r native capped128+self directed prefix; uncapped symmetric optional annulus suffix; r=.015,R=1.267r",
            "observed_risk": "previous observed six-frame native-base scoring; fresh every observed frame; two standalone network passes",
            "autonomous_risk": "one initial native-base warmup; forecast1 exact budget; cache from previous own selected graph and predicted history",
            "random_observed": "[20261005,771,original_seed,split_code,source_index,target_frame]; no arm identity",
            "random_autonomous": "93000+1000*original_seed+source_index; no arm identity; later geometry may differ",
            "saved_parent_test_input_archive": saved, "parent_archive_usage": "only byte-verified histories, target and types; never endpoint output attribution",
            "native_parity": "two separate verification calls per frame/trajectory; raw outputs retained; parity failure halts model job",
            "time_scope": "operational synchronized calls with fixed order/shared preprocessing; not policy-speed or speedup evidence",
            "guards": {"max_candidate_pairs": bridge.MAX_PAIRS, "max_abs_coordinate": bridge.MAX_ABS},
            "deadline_utc": native.DEADLINE, "runtime": full.runtime_provenance(device, "scipy_host"), "threads": args.threads,
            "software": {"numpy": np.__version__, "scipy": full.scipy.__version__, "python": full.platform.python_version(), "torch": str(torch.__version__)},
            "input_files_sha256": pins}
        verify_files(pins)
        if args.mode == "observed":
            run_observed(args, protocol, model, trajectories, expected, lookup)
        else:
            run_autonomous(args, protocol, model, trajectories["test"], test, selected)


if __name__ == "__main__":
    main()
