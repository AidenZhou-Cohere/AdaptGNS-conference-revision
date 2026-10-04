"""Fixed, exploratory physical allocation controls on the existing CPU pilot.

Launch as a fresh ``python -m`` process so thread limits precede BLAS imports.
No training, GPU execution, reserved full-test access, retries or resume.
"""
import os
import sys

NUMERICAL_PACKAGES_PRELOADED = [name for name in ("numpy", "scipy", "torch") if name in sys.modules]
THREAD_ENVIRONMENT = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
os.environ.update(THREAD_ENVIRONMENT)

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import torch
from torch.torch_version import TorchVersion

from research import budget_graph, physical_allocation, pilot, risk_benefit
from research.risk_benefit import sha256, write_json

OBJECTIVES = ("faithful", "nll", "beta_nll")
SEEDS = tuple(range(5))
SPLITS = ("valid", "test")
FRAME_SEEDS = {"valid": 811, "test": 812}
CONTROLS = ("base", "dense", "random25", "speed25", "current_risk25", "lagged_base_risk25")
PHYSICAL_POLICIES = ("inverse_count25", "velocity_rms25")
POLICIES = CONTROLS + PHYSICAL_POLICIES
MAX_SECONDS = 300.0
PILOT_DIRECTORY = Path("research/results/waterdrop_pilot")
TRAINING_PROTOCOL = PILOT_DIRECTORY / "protocol.json"
REQUIRED_FILES = {str(TRAINING_PROTOCOL), "research/results/risk_benefit.json",
    "research/results/benefit_head.json"} | {
    str(PILOT_DIRECTORY / f"{objective}_seed{seed}{suffix}")
    for objective in OBJECTIVES for seed in SEEDS for suffix in (".pt", ".json")}
TIMING_SCOPE = ("Operational CPU measurements during concurrent training; shared candidate construction, "
    "base reuse, scoring passes, raw-output IO and original fixed policy order prevent a fair policy-latency "
    "comparison or speedup claim. No mean-only optimized baseline is measured.")


class ReplayError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__(f"Original control replay differs: {details}")


def expected_trajectories(frame_ids):
    if len(frame_ids) != 36 or len(set(frame_ids)) != 36:
        raise ValueError("Exactly 36 unique required frames must be declared")
    groups = Counter(name.rsplit(":", 1)[0] for name in frame_ids)
    if len(groups) != 3 or set(groups.values()) != {12}:
        raise ValueError("Exactly three trajectories with twelve frames each required")
    return sorted(groups)


def aggregate_frames(records, expected_ids):
    """Strict 12-frame, then three-trajectory means; missing frames stay null."""
    names = expected_trajectories(expected_ids)
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records) or not set(by_id).issubset(expected_ids):
        raise ValueError("Duplicate or unexpected completed frame")
    for row in records:
        if row["trajectory"] != row["id"].rsplit(":", 1)[0] or set(row["policies"]) != set(POLICIES):
            raise ValueError("Completed frame identity or policy set differs")
        for measurement in row["policies"].values():
            if any(not np.isfinite(measurement[key]) or measurement[key] < 0
                   for key in ("mse_normalized_acceleration", "directed_edges")):
                raise ValueError("Invalid completed frame measurement")
    trajectories = {}
    for name in names:
        selected = [row for row in records if row["trajectory"] == name]
        trajectories[name] = {"required_frames": 12, "completed_frames": len(selected),
            "policies": {policy: {metric: float(np.mean([row["policies"][policy][metric] for row in selected]))
                if len(selected) == 12 else None
                for metric in ("mse_normalized_acceleration", "directed_edges")} for policy in POLICIES}}
    complete = len(records) == 36
    policies = {policy: {metric: float(np.mean([trajectories[name]["policies"][policy][metric] for name in names]))
        if complete else None for metric in ("mse_normalized_acceleration", "directed_edges")} for policy in POLICIES}
    return {"required_frames": 36, "completed_frames": len(records), "trajectories": trajectories,
            "equal_trajectory_mean": {"policies": policies}}


def five_seed_stat(values):
    if len(values) != 5 or any(value is not None and not np.isfinite(value) for value in values):
        raise ValueError("Five finite-or-null seed values required")
    complete = all(value is not None for value in values)
    return {"seed_values": values, "required_seeds": 5,
            "defined_seeds": sum(value is not None for value in values),
            "mean": float(np.mean(values)) if complete else None,
            "sample_seed_sd": float(np.std(values, ddof=1)) if complete else None}


def summarize(runs):
    summaries = {}
    for objective in OBJECTIVES:
        for split in SPLITS:
            per_policy = {policy: [] for policy in POLICIES}
            for seed in SEEDS:
                run = runs.get(f"{objective}_seed{seed}")
                group = run["splits"][split]["equal_trajectory_mean"]["policies"] if run else None
                for policy in POLICIES:
                    per_policy[policy].append(group[policy]["mse_normalized_acceleration"] if group else None)
            differences = {}
            for policy in PHYSICAL_POLICIES:
                differences[policy] = {}
                for control in CONTROLS:
                    pairs = list(zip(per_policy[policy], per_policy[control]))
                    delta = [a - b if a is not None and b is not None else None for a, b in pairs]
                    percent = [100.0 * (a - b) / b if a is not None and b is not None and b != 0 else None
                               for a, b in pairs]
                    differences[policy][control] = {"difference": five_seed_stat(delta),
                        "percent_difference": five_seed_stat(percent),
                        "seeds_with_lower_mse": sum(value < 0 for value in delta)
                            if all(value is not None for value in delta) else None}
            summaries[f"{objective}/{split}"] = {"objective": objective, "split": split,
                "seeds": list(SEEDS), "policies": {key: five_seed_stat(value) for key, value in per_policy.items()},
                "physical_minus_control": differences}
    return summaries


def verify_inputs(data_dir, input_identity, repo):
    identity = json.loads(Path(input_identity).read_text())
    if set(identity["files"]) != REQUIRED_FILES:
        raise ValueError("Input manifest must pin both summaries, protocol and all fifteen checkpoint/control pairs")
    if set(identity["data_sha256"]) != {"metadata.json", "valid-pilot.npz", "test-pilot.npz"}:
        raise ValueError("Only metadata and the two compact evaluation sources are allowed")
    for name, expected in identity["files"].items():
        if sha256(repo / name) != expected:
            raise ValueError(f"Pinned input hash differs: {name}")
    for name, expected in identity["data_sha256"].items():
        if sha256(Path(data_dir) / name) != expected:
            raise ValueError(f"Pinned compact-data hash differs: {name}")
    old_modules = {"pilot.py": pilot, "budget_graph.py": budget_graph, "risk_benefit.py": risk_benefit}
    if set(identity["existing_numeric_source_sha256"]) != set(old_modules):
        raise ValueError("Original numerical source identities are incomplete")
    for name, module in old_modules.items():
        if sha256(module.__file__) != identity["existing_numeric_source_sha256"][name]:
            raise ValueError(f"Original numerical source changed: {name}")
    protocol = json.loads((repo / TRAINING_PROTOCOL).read_text())
    if protocol["code_sha256"] != sha256(pilot.__file__):
        raise ValueError("Pilot source differs from original training protocol")
    architecture = protocol["architecture"]
    if (protocol["steps"] != 3000 or protocol["seeds"] != list(SEEDS)
            or protocol["objectives"] != list(OBJECTIVES)
            or any(architecture[key] != value for key, value in
                   (("latent_width", 48), ("processor_depth", 3), ("history", 6), ("noise", 0)))):
        raise ValueError("Fixed compact training population or architecture differs")
    for split in SPLITS:
        expected_trajectories(protocol["frames"][split])
    return identity, protocol


def replay_controls(measurements, control):
    if set(control["policies"]) != set(CONTROLS):
        raise ValueError("All six original control policies required")
    differences = {}
    for policy in CONTROLS:
        new, old = measurements[policy], control["policies"][policy]
        actual, expected = new["mse_normalized_acceleration"], old["mse_normalized_acceleration"]
        if not np.isfinite(expected) or expected < 0:
            raise ValueError("Invalid original control measurement")
        difference = {"mse_difference": actual - expected, "actual_mse": actual, "original_mse": expected,
                      "actual_directed_edges": new["directed_edges"], "original_directed_edges": old["directed_edges"]}
        differences[policy] = difference
        if (new["directed_edges"] != old["directed_edges"]
                or not np.isclose(actual, expected, rtol=2e-5, atol=1e-7)):
            raise ReplayError({"frame": control["id"], "policy": policy, **difference,
                               "completed_replay_comparisons": differences})
    return differences


def checked_prediction(mean, risk, n):
    if (mean.device.type != "cpu" or risk.device.type != "cpu" or mean.shape != (n, 2) or risk.shape != (n,)
            or not torch.isfinite(mean).all() or not torch.isfinite(risk).all() or not torch.all(risk > 0)):
        raise ValueError("Finite CPU predictions and positive finite per-particle risk required")


def selected_extra(graph, pairs, count):
    pairs = np.asarray(pairs)
    if len(pairs) != len(graph.base) + count or not np.array_equal(pairs[:len(graph.base)], graph.base):
        raise ValueError("Mandatory graph retention or exact optional budget violated")
    extra = pairs[len(graph.base):]
    codes = extra[:, 0] * graph.n_nodes + extra[:, 1]
    available = graph.extra[:, 0] * graph.n_nodes + graph.extra[:, 1]
    if len(np.unique(codes)) != count or not np.isin(codes, available).all():
        raise ValueError("Selected optional pairs are duplicated or unavailable")
    return extra


def tie_audit(graph, scores, chosen, budget):
    """Store all optional pairs tied at the cutoff and chosen tied slots."""
    if budget == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return {"cutoff": None, "strictly_higher_pairs": 0, "all_tied_pairs": 0,
                "selected_tied_slots": 0, "split_cutoff_tie": False}, empty, empty.copy()
    priority = np.maximum(scores[graph.extra[:, 0]], scores[graph.extra[:, 1]])
    cutoff = float(np.sort(priority)[-budget])
    tied = graph.extra[priority == cutoff]
    selected_priority = np.maximum(scores[chosen[:, 0]], scores[chosen[:, 1]])
    selected_tied = chosen[selected_priority == cutoff]
    higher = int(np.sum(priority > cutoff))
    if len(selected_tied) != budget - higher:
        raise ValueError("Selected cutoff tie does not fill the declared budget")
    return {"cutoff": cutoff, "strictly_higher_pairs": higher, "all_tied_pairs": len(tied),
            "selected_tied_slots": len(selected_tied), "split_cutoff_tie": len(tied) > len(selected_tied)}, tied, selected_tied


def _evaluate_frame(model, frame, control, objective, rng, progress):
    started = time.perf_counter()
    graph, n = frame["graph"], frame["graph"].n_nodes
    budget = math.floor(0.25 * len(graph.extra))
    progress["row"].update({"n_particles": n, "base_pairs": len(graph.base),
                            "available_extra_pairs": len(graph.extra), "extra_pair_budget": budget})
    raw = progress["raw"]
    raw.update({"base_pairs": graph.base, "available_extra_pairs": graph.extra})
    if (control["id"] != frame["id"] or control["trajectory"] != frame["trajectory"]
            or control["n_particles"] != n or control["extra_pair_budget"] != budget):
        raise ValueError("Original control/frame identity or budget differs")
    faithful = objective == "faithful"
    stamp = time.perf_counter()
    base_mean, current_risk = pilot.run_forward(model, frame, graph.base, faithful)
    if base_mean.device.type == "cpu" and current_risk.device.type == "cpu":
        raw.update({"base_scoring_mean": base_mean.numpy(), "current_risk": current_risk.numpy()})
    checked_prediction(base_mean, current_risk, n)
    base_seconds = time.perf_counter() - stamp
    stamp = time.perf_counter()
    previous_mean, previous_risk = pilot.run_forward(model, frame, frame["prev_graph"].base, faithful, previous=True)
    if previous_mean.device.type == "cpu" and previous_risk.device.type == "cpu":
        raw.update({"previous_base_scoring_mean": previous_mean.numpy(), "previous_base_risk": previous_risk.numpy()})
    checked_prediction(previous_mean, previous_risk, n)
    previous_seconds = time.perf_counter() - stamp
    stamp = time.perf_counter()
    velocity = np.asarray(frame["position"], dtype=np.float64) - np.asarray(frame["prev_position"], dtype=np.float64)
    physical = physical_allocation.physical_scores(velocity, graph.base)
    physical_seconds = time.perf_counter() - stamp
    scores = {"speed25": np.asarray(frame["speed"]), "current_risk25": current_risk.numpy(),
              "lagged_base_risk25": previous_risk.numpy(),
              "inverse_count25": physical["negative_neighbor_count"], "velocity_rms25": physical["velocity_dispersion"]}
    stamp = time.perf_counter()
    pairs = {"base": graph.base, "dense": np.concatenate((graph.base, graph.extra)),
             "random25": budget_graph.random_pairs(graph, budget, rng)}
    pairs.update({policy: budget_graph.select_pairs(graph, score, budget) for policy, score in scores.items()})
    selection_seconds = time.perf_counter() - stamp
    extras = {policy: selected_extra(graph, pairs[policy], 0 if policy == "base" else
              len(graph.extra) if policy == "dense" else budget) for policy in POLICIES}
    measurements = progress["row"]["policies"]
    raw.update({"base_pairs": graph.base, "available_extra_pairs": graph.extra,
        "observed_velocity": velocity, "current_risk": current_risk.numpy(), "previous_base_risk": previous_risk.numpy(),
        "speed": np.asarray(frame["speed"]), **physical})
    for policy in POLICIES:
        stamp = time.perf_counter()
        progress["row"]["last_attempted_policy"] = policy
        mean, variance = (base_mean, current_risk) if policy == "base" else pilot.run_forward(model, frame, pairs[policy], faithful)
        if mean.device.type == "cpu" and variance.device.type == "cpu":
            # Keep a failed/nonfinite prediction if validation raises next.
            raw["last_attempted_mean"], raw["last_attempted_variance"] = mean.numpy(), variance.numpy()
        checked_prediction(mean, variance, n)
        seconds = time.perf_counter() - stamp
        error = (mean - frame["target"]).square().sum(-1).numpy().astype(np.float64)
        if error.shape != (n,) or not np.isfinite(error).all() or np.any(error < 0):
            raise ValueError("Invalid per-particle vector squared error")
        measurements[policy] = {"mse_normalized_acceleration": float(error.mean() / 2),
            "directed_edges": 2 * len(pairs[policy]), "forward_seconds": seconds, "reused_base_prediction": policy == "base"}
        raw[f"{policy}_vector_se"] = error
        raw[f"{policy}_selected_extra_pairs"] = extras[policy]
    replay = replay_controls(measurements, control)
    ties = {}
    for policy, score in scores.items():
        audit, all_tied, selected_tied = tie_audit(graph, score, extras[policy], budget)
        ties[policy] = audit
        raw[f"{policy}_all_cutoff_tied_pairs"] = all_tied
        raw[f"{policy}_selected_cutoff_tied_pairs"] = selected_tied
    overlap = {}
    for policy in PHYSICAL_POLICIES:
        overlap[policy] = {}
        a = extras[policy][:, 0] * n + extras[policy][:, 1]
        for comparator in CONTROLS:
            b = extras[comparator][:, 0] * n + extras[comparator][:, 1]
            shared = int(len(np.intersect1d(a, b)))
            union = len(a) + len(b) - shared
            overlap[policy][comparator] = {"shared_optional_pairs": shared,
                "physical_optional_pairs": len(a), "control_optional_pairs": len(b),
                "fraction_of_physical": shared / len(a) if len(a) else None,
                "jaccard": shared / union if union else 1.0}
    row = {"id": frame["id"], "trajectory": frame["trajectory"], "step": frame["step"], "n_particles": n,
        "base_pairs": len(graph.base), "available_extra_pairs": len(graph.extra), "extra_pair_budget": budget,
        "isolated_particles": int(physical["isolated"].sum()), "policies": measurements,
        "control_replay": replay, "cutoff_ties": ties, "physical_control_overlap": overlap,
        "timing": {"base_scoring_seconds": base_seconds, "previous_base_scoring_seconds": previous_seconds,
            "physical_scores_seconds": physical_seconds, "selection_seconds": selection_seconds,
            "frame_seconds_excluding_output_io": time.perf_counter() - started}}
    return row, raw


@torch.no_grad()
def evaluate_frame(model, frame, control, objective, rng):
    progress = {"row": {"id": frame["id"], "trajectory": frame["trajectory"],
                         "step": frame["step"], "policies": {}}, "raw": {}}
    try:
        return _evaluate_frame(model, frame, control, objective, rng, progress)
    except BaseException as error:
        error.partial_frame = progress
        raise


def write_frame_raw(path, arrays):
    """Immutable per-frame archive; incomplete temporary files are retained."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite a completed raw frame")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return {"path": str(path), "sha256": sha256(path)}


def save_failed_frame(output, context, progress):
    prefix = f"failed_{context.get('split', 'unknown')}_{context.get('index', 0):03d}"
    directory = Path(output) / context["model"]
    record = {**progress["row"], "state": "failed", "accepted_for_aggregation": False}
    if progress["raw"]:
        pointer = write_frame_raw(directory / f"{prefix}.npz", progress["raw"])
        pointer["path"] = str(Path(pointer["path"]).relative_to(output))
        record["raw_file"] = pointer
    write_json(directory / f"{prefix}.json", record)
    return str((directory / f"{prefix}.json").relative_to(output))


def enforce_cap(started):
    if time.perf_counter() - started >= MAX_SECONDS:
        raise TimeoutError("Predeclared 300-second operational cap reached; required groups remain incomplete")


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    repo = Path(__file__).resolve().parents[1]
    sources = {"physical_allocation_pilot.py": Path(__file__), "physical_allocation.py": Path(physical_allocation.__file__),
        "pilot.py": Path(pilot.__file__), "budget_graph.py": Path(budget_graph.__file__), "risk_benefit.py": Path(risk_benefit.__file__)}
    result = {"schema": 1, "scope": "Exploratory physical allocation on fixed compact models; no independent test confirmation",
        "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running", "expected_runs": [
            f"{objective}_seed{seed}" for seed in SEEDS for objective in OBJECTIVES], "runs": {},
        "timing_scope": TIMING_SCOPE, "operational_cap_seconds": MAX_SECONDS,
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "thread_environment": THREAD_ENVIRONMENT, "numerical_packages_preloaded": NUMERICAL_PACKAGES_PRELOADED,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": str(torch.__version__), "device": "cpu", "torch_threads": 1}}
    write_json(output / "summary.json", {**result, "summaries": summarize({})})
    context = {"stage": "input_verification"}
    active = None
    try:
        result["protocol_sha256"] = sha256(args.protocol)
        result["input_identity_sha256"] = sha256(args.input_identity)
        identity, training_protocol = verify_inputs(args.data_dir, args.input_identity, repo)
        result["input_identity"] = identity
        torch.set_num_threads(1)
        if torch.get_num_threads() != 1:
            raise RuntimeError("Exactly one Torch CPU thread required")
        metadata = json.loads((Path(args.data_dir) / "metadata.json").read_text())
        context["stage"] = "frame_loading"
        stamp = time.perf_counter()
        frames = {split: pilot.load_frames(Path(args.data_dir) / f"{split}-pilot.npz", metadata, 12, FRAME_SEEDS[split])
                  for split in SPLITS}
        result["frame_loading_and_candidates_seconds"] = time.perf_counter() - stamp
        for split in SPLITS:
            if [frame["id"] for frame in frames[split]] != training_protocol["frames"][split]:
                raise ValueError("Original protocol frame identities differ from compact data")
        for seed in SEEDS:
            for objective in OBJECTIVES:
                enforce_cap(started)
                name = f"{objective}_seed{seed}"
                context = {"stage": "checkpoint_loading", "model": name}
                active = {"objective": objective, "seed": seed, "state": "running", "splits": {
                    split: {"frames": [], **aggregate_frames([], training_protocol["frames"][split])} for split in SPLITS}}
                result["runs"][name] = active
                model_json = output / f"{name}.json"
                write_json(model_json, active)
                checkpoint = repo / PILOT_DIRECTORY / f"{name}.pt"
                controls_path = checkpoint.with_suffix(".json")
                for path in (checkpoint, controls_path):
                    if sha256(path) != identity["files"][str(path.relative_to(repo))]:
                        raise ValueError("Pinned checkpoint/control changed after input verification")
                controls = json.loads(controls_path.read_text())
                if controls["objective"] != objective or type(controls["seed"]) is not int or controls["seed"] != seed:
                    raise ValueError("Original control objective/seed differs")
                with torch.serialization.safe_globals([TorchVersion]):
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if (payload["protocol"] != training_protocol or payload["objective"] != objective
                        or type(payload["seed"]) is not int or payload["seed"] != seed):
                    raise ValueError("Checkpoint protocol/objective/seed differs")
                model = pilot.PilotGNS(width=48, depth=3)
                model.load_state_dict(payload["state_dict"], strict=True)
                model.eval()
                if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                    raise RuntimeError("Non-CPU checkpoint execution forbidden")
                active.update({"checkpoint_sha256": sha256(checkpoint), "original_controls_sha256": sha256(controls_path)})
                for split in SPLITS:
                    rng = np.random.default_rng(91300 + seed)
                    old_frames = controls["validation" if split == "valid" else "test"]
                    if [row["id"] for row in old_frames] != training_protocol["frames"][split]:
                        raise ValueError("Original control frame identities/order differ")
                    rows = active["splits"][split]["frames"]
                    for index, (frame, control) in enumerate(zip(frames[split], old_frames)):
                        context = {"stage": "frame", "model": name, "split": split, "frame": frame["id"], "index": index}
                        enforce_cap(started)
                        row, arrays = evaluate_frame(model, frame, control, objective, rng)
                        prefix = f"{split}_{index:03d}"
                        row["raw_prefix"] = prefix
                        context["raw_frame_file"] = str(Path(name) / f"{prefix}.npz")
                        row["raw_file"] = write_frame_raw(output / name / f"{prefix}.npz", arrays)
                        row["raw_file"]["path"] = str(Path(row["raw_file"]["path"]).relative_to(output))
                        rows.append(row)
                        active["splits"][split] = {"frames": rows, **aggregate_frames(rows, training_protocol["frames"][split])}
                        write_json(model_json, active)
                active["state"] = "complete"
                active["raw_layout"] = "Immutable per-frame NPZ files with paths/hashes on each accepted frame"
                write_json(model_json, active)
                result["summaries"] = summarize(result["runs"])
                result["elapsed_seconds"] = time.perf_counter() - started
                write_json(output / "summary.json", result)
                active = None
                del model, payload
        context = {"stage": "final_provenance"}
        if any(sha256(path) != result["source_sha256"][name] for name, path in sources.items()):
            raise ValueError("Numerical source changed during experiment")
        if sha256(args.protocol) != result["protocol_sha256"] or sha256(args.input_identity) != result["input_identity_sha256"]:
            raise ValueError("Analysis protocol or input identity changed during experiment")
        result["state"] = "complete"
    except BaseException as error:
        failure = {"error": f"{type(error).__name__}: {error}", "context": context,
                   "recorded_utc": datetime.now(timezone.utc).isoformat(), "details": getattr(error, "details", None)}
        if hasattr(error, "partial_frame"):
            try:
                failure["partial_frame_record"] = save_failed_frame(output, context, error.partial_frame)
            except BaseException as preservation_error:
                failure["partial_frame_preservation_error"] = f"{type(preservation_error).__name__}: {preservation_error}"
        if active is not None:
            active["state"], active["failure"] = "failed", failure
            write_json(model_json, active)
        result["state"], result["failure"] = "failed", failure
        write_json(output / "failure.json", failure)
        raise
    finally:
        result["summaries"] = summarize(result["runs"])
        result["summaries_validated"] = result["state"] == "complete"
        if context.get("stage") == "final_provenance" and result["state"] != "complete":
            result["unvalidated_summaries"] = result["summaries"]
            result["summaries"] = summarize({})
        result["elapsed_seconds"] = time.perf_counter() - started
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-identity", type=Path, default=Path("research/protocols/physical_allocation_inputs.json"))
    parser.add_argument("--protocol", type=Path, default=Path("research/protocols/physical_allocation_pilot.md"))
    args = parser.parse_args()
    if NUMERICAL_PACKAGES_PRELOADED:
        parser.error("Launch a fresh python -m process so one-thread limits precede numerical imports")
    result = run(args)
    print(json.dumps({"state": result["state"], "models": len(result["runs"]), "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
