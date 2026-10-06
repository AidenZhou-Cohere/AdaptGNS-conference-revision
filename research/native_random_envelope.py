"""Exploratory exact-budget random-action envelope on native observed histories.

Separate original100k outcome family. No model or data loading at import.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from research import graph_convention_bridge as bridge
from research import native_graph_rollout as native

full = native.full
PROTOCOL = Path(__file__).parent / "protocols/native_random_envelope_20261006.md"
TRAINING_PROTOCOL = full.REPO / "research/protocols/full_waterdrop_100k.md"
RISKS = ("previous-observed-base-risk25", "current-base-risk25")
CONTROLS = ("base", "dense", "speed25") + RISKS
RANDOM_CASES = tuple(f"random25_draw{draw}" for draw in range(8))
POLICIES = CONTROLS + RANDOM_CASES
METRICS = ("position_coordinate_mse", "normalized_coordinate_mse")
SCOPE = "post-inspection exploratory native random-action envelope on original100k; not independent confirmation"
COHORT = {(objective, seed) for objective in ("faithful", "nll") for seed in (0, 1, 2)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def verify_files(pins):
    for path, digest in pins.items():
        require(full.sha256(path) == digest, f"Pinned input/source byte identity differs: {path}")


def random_material(seed, identity, draw):
    require(type(seed) is int and seed in (0, 1, 2) and type(draw) is int and draw in range(8),
            "Known integer original seed and draw0..7 required")
    require(identity["split"] in ("valid", "test") and type(identity["source_index"]) is int
            and identity["source_index"] >= 0 and type(identity["target_frame"]) is int
            and identity["target_frame"] >= 7, "Known split and integer observed source/frame required")
    return [20261006, 881, seed, 0 if identity["split"] == "valid" else 1,
            identity["source_index"], identity["target_frame"], draw]


def selection_spec(policy, previous_output, base_output):
    require(policy in POLICIES, "Unknown envelope case")
    if policy in RANDOM_CASES:
        return "random25", None
    if policy in RISKS:
        return "laggedrisk25", (previous_output if policy == RISKS[0] else base_output)["risk"]
    return policy, None


def envelope(draw_values, reference=None):
    require(len(draw_values) == 8 and all(value is None or (type(value) in (int, float)
            and np.isfinite(value)) for value in draw_values), "Exactly eight finite/null draw values required")
    require(reference is None or (type(reference) in (int, float) and np.isfinite(reference)), "Finite/null reference required")
    defined = sum(value is not None for value in draw_values)
    result = {"draw_values": draw_values, "required_draws": 8, "defined_draws": defined,
        "mean": None, "minimum": None, "maximum": None, "sample_sd": None,
        "monte_carlo_standard_error_of_mean": None,
        "reference_minus_random_mean": None, "random_strictly_better": None,
        "random_tied": None, "random_strictly_worse": None,
        "random_mean_null_reason": "required random draw failed/missing" if defined != 8 else None,
        "comparison_null_reason": "required reference or random draw failed/missing" if defined != 8 or reference is None else None}
    if defined == 8:
        result.update(mean=float(np.mean(draw_values)), minimum=min(draw_values), maximum=max(draw_values),
                      sample_sd=float(np.std(draw_values, ddof=1)),
                      monte_carlo_standard_error_of_mean=float(np.std(draw_values, ddof=1)/np.sqrt(8)))
        if reference is not None:
            result.update(reference_minus_random_mean=float(reference-result["mean"]),
                random_strictly_better=sum(value < reference for value in draw_values),
                random_tied=sum(value == reference for value in draw_values),
                random_strictly_worse=sum(value > reference for value in draw_values))
    return result


def frame_envelopes(cases):
    require(set(cases) == set(POLICIES), "Exact envelope case coverage required")
    def metric(name, key):
        case = cases[name]
        return case["metrics"][key] if case["status"] == "complete" else None
    return {key: {control: envelope([metric(name, key) for name in RANDOM_CASES], metric(control, key))
                  for control in CONTROLS} for key in METRICS}


@torch.no_grad()
def evaluate_frame(model, current, previous, types, target, identity, seed, device):
    """Thirteen matching native-prefix cases; future target enters after inference."""
    started = time.perf_counter()
    current, previous = np.asarray(current, dtype=np.float32), np.asarray(previous, dtype=np.float32)
    types = np.asarray(types, dtype=np.int64)
    require(current.shape == previous.shape and current.ndim == 3 and current.shape[0] == 6
            and current.shape[2] == 2, "Two matching six-frame 2D histories required")
    require(types.shape == (current.shape[1],) and not np.any(types == 3), "One nonkinematic type per particle required")
    n = len(types)
    materials = {name: random_material(seed, identity, draw) for draw, name in enumerate(RANDOM_CASES)}
    rng = lambda name="base": np.random.default_rng(np.random.SeedSequence(materials.get(name, [20261006, 881, 0])))
    std = bridge.same.normalization(model)["std"]
    arrays = {"current_history": current.copy(), "previous_history": previous.copy(), "particle_types": types.copy(),
              "acceleration_std": std, "bounds": np.asarray(model._boundaries, dtype=np.float64)}
    row = {**identity, "status": "running", "failure": None, "n_particles": n, "cases": {},
           "random_seed_materials": materials, "current_history_sha256": full.state_hash(current),
           "previous_history_sha256": full.state_hash(previous), "native_parity": None, "previous_score": None}
    def failed_all(failure):
        row.update(status="failed", failure=failure, operational_seconds=time.perf_counter() - started)
        row["cases"] = {name: {"status": "failed", "failure": failure, "metrics": None} for name in POLICIES}
        row["random_envelope"] = frame_envelopes(row["cases"])
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

    base_failure = None
    try:
        bridge.validate_output(base_output, n)
    except full.RolloutGuard as error:
        base_failure = error.details
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
        scoring_failure = score_failure if policy == RISKS[0] else base_failure if policy == RISKS[1] else None
        if scoring_failure:
            row["cases"][policy] = {"status": "failed", "failure": {"category": "scoring_pass_failed", "cause": scoring_failure},
                "metrics": None, "network_passes_for_standalone_policy":
                    (1 if previous_output is not None else 0) if policy == RISKS[0] else 1}
            continue
        phase, graph_seconds = "graph", None
        case = {"status": "running", "failure": None, "metrics": None}
        try:
            graph_start = time.perf_counter()
            actual, score = selection_spec(policy, previous_output, base_output)
            _, edges, optional, audit = native.native_graph(current, actual, score, rng(policy))
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
                scoring_seconds=(previous_output if policy == RISKS[0] else base_output)["operational_seconds"] if score is not None else 0.,
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
    row["random_envelope"] = frame_envelopes(row["cases"])
    return row, arrays


def preflight_output(directory, protected=()):
    directory = Path(directory).resolve()
    require(not directory.exists(), "A fresh nonexistent output directory is required; preserve every prior attempt")
    for path in protected:
        path = Path(path).resolve()
        require(directory != path and path not in directory.parents and directory not in path.parents,
                "A separate output tree outside protected inputs/results is required")
    return directory


def committed_checkpoint(checkpoint, payload):
    checkpoint = Path(checkpoint)
    latest = strict_json(checkpoint.parent / "latest.json")
    status = strict_json(checkpoint.parent / "status.json")
    protocol = strict_json(checkpoint.parent / "protocol.json")
    digest = full.sha256(checkpoint)
    require(checkpoint.name == "checkpoint-100000.pt" and latest == {
        "path": checkpoint.name, "sha256": digest, "completed_steps": 100000},
        "Committed original fixed100k checkpoint required")
    cfg = payload["run_config"]
    require(protocol == cfg and status.get("state") == "complete" and status.get("completed_steps") == 100000
            and status.get("latest_checkpoint") == latest and status.get("objective") == cfg["objective"]
            and status.get("seed") == cfg["seed"] and status.get("run_config_sha256") == payload["run_config_sha256"],
            "Original checkpoint publication identity/completion differs")
    return digest


def source_paths():
    return [Path(__file__), PROTOCOL, TRAINING_PROTOCOL, Path(native.__file__), native.PROTOCOL,
        Path(bridge.__file__), bridge.PROTOCOL, Path(full.__file__), Path(bridge.same.__file__),
        full.REPO / "research/budget_graph.py", full.REPO / "research/full_training.py",
        *sorted((full.REPO / "adaptive-gns/gns").glob("*.py"))]


def prepare_inputs(args, device):
    """Admit only committed original100k checkpoints and the fixed observed population."""
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata_path = args.validation_manifest.parent / "metadata.json"
    metadata = strict_json(metadata_path)
    full.check_checkpoint(payload, "locked_test", TRAINING_PROTOCOL, full.sha256(metadata_path))
    config = payload["run_config"]
    require(type(config["seed"]) is int and (config["objective"], config["seed"]) in COHORT,
            "Original faithful/NLL seeds0,1,2 required")
    checkpoint_hash = committed_checkpoint(args.checkpoint, payload)
    for relative, digest in config["source_sha256"].items():
        require(full.sha256(full.REPO / relative) == digest, "Frozen training source changed")
    strict_json(args.validation_manifest); strict_json(args.test_manifest)
    expected, valid, test = bridge.expected_frames(config, args.validation_manifest, args.test_manifest)
    require(valid["metadata"] == metadata and test["metadata"] == metadata, "Split metadata differs")
    trajectories = {"valid": full.load_manifest_data(args.validation_manifest, verify_hashes=True),
                    "test": full.load_manifest_data(args.test_manifest, verify_hashes=True)}
    lookup, saved = bridge.verify_saved_test(args.saved_same_state_dir, checkpoint_hash, expected)
    paths = source_paths() + [args.checkpoint, args.validation_manifest, args.test_manifest, metadata_path,
        args.saved_same_state_dir / "protocol.json", args.saved_same_state_dir / "result.json"]
    paths += [args.checkpoint.parent / name for name in ("protocol.json", "latest.json", "status.json")]
    paths += [full.REPO / relative for relative in config["source_sha256"]]
    for row in lookup.values():
        paths += [args.saved_same_state_dir / row[key] for key in ("array_file", "record_file")]
    for manifest, directory in ((valid, args.validation_manifest.parent), (test, args.test_manifest.parent)):
        for record in manifest["records"]:
            paths += [directory / record[key]["path"] for key in ("positions", "particle_types")]
    pins = {str(path.resolve()): full.sha256(path) for path in paths}
    verify_files(pins)
    model, provenance = full.load_for_evaluation(args.checkpoint, metadata, device, radius_backend="scipy_host")
    require(model._max_num_neighbors == 128 and float(model._connectivity_radius) == .015
            and np.asarray(model._boundaries).tolist() == metadata["bounds"], "Native model graph/boundaries differ")
    protocol = {"schema": 1, "scope": SCOPE, "objective": config["objective"], "seed": config["seed"],
        "checkpoint_sha256": checkpoint_hash, "checkpoint_provenance": provenance,
        "run_config_sha256": payload["run_config_sha256"], "envelope_protocol_sha256": full.sha256(PROTOCOL),
        "expected_frames": expected, "cases": list(POLICIES), "random_draw_ids": list(range(8)),
        "saved_test_inputs": saved, "source_tfrecord": test["source"],
        "validation_manifest": str(args.validation_manifest.resolve()), "test_manifest": str(args.test_manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "saved_same_state_dir": str(args.saved_same_state_dir.resolve()),
        "graph_semantics": "native strict-r capped128/self base prefix; separately ordered uncapped symmetric optional annulus suffix; no global resort",
        "random_seed_material": "[20261006,881,original_seed,split_code,source_index,target_frame,draw_id]; objective omitted",
        "time_scope": "shared work/fixed order single operational calls, including parity; not standalone policy latency",
        "guards": {"max_candidate_pairs": bridge.MAX_PAIRS, "max_abs_coordinate": bridge.MAX_ABS},
        "runtime": full.runtime_provenance(device, "scipy_host"), "threads": 2,
        "software": {"numpy": np.__version__, "scipy": full.scipy.__version__, "python": full.platform.python_version(),
                     "platform": full.platform.platform()}, "input_files_sha256": pins}
    return protocol, model, trajectories, expected, lookup


def run(args, protocol, model, trajectories, expected, saved_lookup):
    """Fresh only: atomic committed frames, no retries or implicit recovery."""
    directory = Path(args.output_dir)
    require(not any(path.name != "run.lock" for path in directory.iterdir()), "Fresh owned output required")
    require(protocol["expected_frames"] == expected and protocol["cases"] == list(POLICIES), "Fixed schedule/cases differ")
    verify_files(protocol["input_files_sha256"])
    full.atomic_json(directory / "protocol.json", protocol)
    protocol_hash = full.sha256(directory / "protocol.json")
    started = time.perf_counter()
    records = []
    def status(state, **extra):
        full.atomic_json(directory / "status.json", {"state": state, "pid": os.getpid(),
            "objective": protocol["objective"], "seed": protocol["seed"], "committed_frames": len(records),
            "updated_utc": bridge.utc_now(), "operational_seconds": time.perf_counter()-started, **extra})
    try:
        status("running")
        for item in expected:
            if datetime.now(timezone.utc) >= datetime.fromisoformat(native.DEADLINE):
                raise TimeoutError("Research cutoff reached before next observed frame")
            inputs = bridge.frame_input(item, trajectories, args.saved_same_state_dir, saved_lookup)
            row, arrays = evaluate_frame(model, *inputs, item, protocol["seed"], args.device)
            path = directory / (bridge.stem(item) + ".npz")
            with path.with_suffix(".npz.tmp").open("xb") as stream:
                np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
            path.with_suffix(".npz.tmp").replace(path)
            row.update(objective=protocol["objective"], seed=protocol["seed"], checkpoint_sha256=protocol["checkpoint_sha256"],
                protocol_sha256=protocol_hash, array_file=path.name, array_sha256=full.sha256(path))
            full.atomic_json(path.with_suffix(".json"), row)
            records.append(bridge.record_index(row, path.with_suffix(".json")))
            full.atomic_json(directory / "result.json", {"state": "partial", "protocol_sha256": protocol_hash, "records": records})
            status("running", last_frame=item, last_status=row["status"])
            print(json.dumps({**item, "status": row["status"], "committed_frames": len(records)}), flush=True)
            if row["failure"] and row["failure"].get("category") == "native_parity_failure":
                raise RuntimeError("Native parity failure retained; stop for independent review, no retry")
        verify_files(protocol["input_files_sha256"])
        full.atomic_json(directory / "result.json", {"state": "complete", "scope": SCOPE,
            "protocol_sha256": protocol_hash, "objective": protocol["objective"], "seed": protocol["seed"],
            "checkpoint_sha256": protocol["checkpoint_sha256"], "required_frames": len(expected), "records": records,
            "complete_frames": sum(row["status"] == "complete" for row in records),
            "failed_frames": sum(row["status"] == "failed" for row in records),
            "operational_seconds": time.perf_counter()-started})
        status("complete", result_sha256=full.sha256(directory / "result.json"))
    except BaseException as error:
        status("deadline_stopped" if isinstance(error, TimeoutError) else "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
               error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "validation-manifest", "test-manifest", "saved-same-state-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    require(args.launch, "Explicit --launch required after root process/deadline checks and source freeze")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0", "Explicit MPS CPU fallback disable required")
    args.output_dir = preflight_output(args.output_dir, (args.checkpoint.resolve().parent.parent,
        args.saved_same_state_dir.resolve().parent, args.validation_manifest.resolve().parent, args.test_manifest.resolve().parent))
    torch.set_num_threads(2)
    device = full.resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with full.RunLock(args.output_dir):
        # Admission failures are retained too; no overwrite of another process's directory/lock.
        try:
            prepared = prepare_inputs(args, device)
            run(args, *prepared)
        except BaseException as error:
            if not (args.output_dir / "status.json").exists():
                full.atomic_json(args.output_dir / "status.json", {"state": "admission_error", "pid": os.getpid(),
                    "error_type": type(error).__name__, "error": str(error), "updated_utc": bridge.utc_now()})
            raise


if __name__ == "__main__":
    main()
