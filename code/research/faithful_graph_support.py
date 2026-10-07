"""Particle-simulation methods and data utilities."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from research import full_training as original
from research import graph_convention_bridge as bridge
from gns.model_io import load_for_evaluation

REPO = original.REPO
PROTOCOL = REPO / "research/protocols/faithful_graph_support_10k_20261005.md"
SCHEMA = 1
PARENT_UPDATES = 100000
EXTRA_UPDATES = 10000
SAVE_EVERY = 2500
LR = 1e-5
PARENT_HASHES = {
    0: "35eef43a0d50c812f9d440a528649a086e141eec20062e649c1487ef381a48db",
    1: "14dcd33ffd5ce5952ad5c2dfafe18812d044561927cfd00ae1bc6bbcbe2f83bf",
    2: "42c8c157c8a48cffaa58e4377193c14cc2403b4e5119299a1a90eebd018a55d3",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def simulator_config_sha256(config):
    """Canonical metadata and tensor-byte identity, including normalization."""
    def canonical(value):
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            return {"tensor_dtype": str(value.dtype), "shape": list(value.shape),
                    "bytes_sha256": bridge.full.state_hash(array)}
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        return value
    return original.config_hash(canonical(config))


def graph_rng(seed, absolute_step, example_slot):
    require(seed >= 0 and absolute_step >= 0 and example_slot >= 0, "Nonnegative RNG identity required")
    coin_material = [20261005, seed, absolute_step, example_slot, 4409]
    pair_material = [20261005, seed, absolute_step, example_slot, 5501]
    coin = bool(np.random.default_rng(np.random.SeedSequence(coin_material)).random() < .5)
    rng = np.random.default_rng(np.random.SeedSequence(pair_material))
    return coin, rng, coin_material, pair_material


def append_optional_edges(noisy_sequence, counts, native_edges, radius, seed, absolute_step, arm):
    """Native batch graph prefix plus exact symmetric optional suffix.

    Mandatory edges are obtained from the native model, not reconstructed or
    symmetrized. The independent per-example graph RNG never touches torch.
    """
    require(arm in ("base", "mix"), "Unknown continuation arm")
    counts = torch.as_tensor(counts).detach().cpu().numpy().astype(np.int64)
    points = noisy_sequence[:, -1].detach().cpu().numpy().astype(np.float32, copy=False)
    native = native_edges.detach().cpu().numpy()
    require(counts.ndim == 1 and np.all(counts > 0) and int(counts.sum()) == len(points), "Invalid batch partition")
    require(native.shape[0] == 2 and native.dtype.kind in "iu", "Invalid native edges")
    offsets = np.r_[0, np.cumsum(counts)]
    require(np.all(native >= 0) and np.all(native < len(points)), "Native edge index out of range")
    membership = np.repeat(np.arange(len(counts)), counts)
    require(np.all(membership[native[0]] == membership[native[1]]), "Native cross-example edge")
    additions, ledger = [], []
    for slot, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        start, end = int(start), int(end)
        local = points[start:end]
        graph = bridge.strict_pairs(local, radius)
        coin, rng, coin_material, pair_material = graph_rng(seed, absolute_step, slot)
        budget = math.floor(.25 * len(graph.extra))
        active = arm == "mix" and coin
        optional = graph.extra[rng.permutation(len(graph.extra))[:budget]] if active else np.empty((0, 2), dtype=np.int64)
        optional = bridge.same.canonical_pairs(optional)
        extra_edges = bridge.ordered_edges(local, optional, False)
        require(extra_edges.shape[1] == 2 * (budget if active else 0), "Exact 2B optional count violated")
        require(not np.any(extra_edges[0] == extra_edges[1]), "Optional self edge")
        mask = (native[1] >= start) & (native[1] < end)
        local_native = native[:, mask] - start
        expected_native = bridge.ordered_edges(local, graph.base, True, 128)
        require(np.array_equal(local_native, expected_native), "Native graph differs from strict/capped/self convention")
        if extra_edges.size:
            additions.append(extra_edges + start)
        full_degree = np.bincount(bridge.ordered_edges(local, graph.base, True)[1], minlength=len(local))
        ledger.append({"example_slot": slot, "n_particles": len(local), "exposure_coin": coin,
            "expanded": active, "coin_seed_material": coin_material, "pair_seed_material": pair_material,
            "native_directed_edges": int(local_native.shape[1]), "native_self_edges": int(np.sum(local_native[0] == local_native[1])),
            "receivers_above_native_cap": int(np.sum(full_degree > 128)), "annulus_pairs": len(graph.extra),
            "optional_budget_if_exposed": budget, "selected_optional_pairs": len(optional),
            "native_edge_sha256": bridge.full.state_hash(local_native),
            "optional_pair_sha256": bridge.full.state_hash(optional), "noisy_current_sha256": bridge.full.state_hash(local)})
    all_edges = np.concatenate((native, *additions), axis=1) if additions else native
    require(np.array_equal(all_edges[:, :native.shape[1]], native), "Native mandatory prefix changed")
    require(all_edges.shape[1] == native.shape[1] + 2 * sum(row["selected_optional_pairs"] for row in ledger), "Batch optional budget differs")
    require(np.all(membership[all_edges[0]] == membership[all_edges[1]]), "Augmented cross-example edge")
    # Returning the original tensor in the no-addition case also retains its
    # exact edge order and avoids a needless device round trip.
    edges = native_edges if not additions else torch.as_tensor(all_edges, dtype=torch.long, device=native_edges.device)
    return edges, ledger


def forward_batch(model, batch, noise, device, seed, absolute_step, arm):
    """Direct normalized output; never decode positions then invert them."""
    position, types, counts, labels = batch
    position, noise = position.to(device), noise.to(device)
    types, counts, labels = types.to(device), counts.to(device), labels.to(device)
    noisy = position + noise
    nodes, native_edges, native_features = model._encoder_preprocessor(
        noisy, counts, types, None, augment_radius_prob=0., augment_radius_factor=1.267)
    edges, ledger = append_optional_edges(noisy, counts, native_edges, float(model._connectivity_radius), seed, absolute_step, arm)
    if edges is native_edges:
        features = native_features
    else:
        current = noisy[:, -1]
        displacements = (current[edges[0]] - current[edges[1]]) / model._connectivity_radius
        features = torch.cat((displacements, torch.norm(displacements, dim=-1, keepdim=True)), dim=-1)
    prediction, head = model._encode_process_decode(nodes, edges, features)
    target = model._inverse_decoder_postprocessor(labels + noise[:, -1], noisy)
    return prediction, head, target, ledger


def update_once(model, optimizer, batch, noise, device, seed, absolute_step, arm):
    model.train()
    for group in optimizer.param_groups:
        group["lr"] = LR
    optimizer.zero_grad(set_to_none=True)
    prediction, head, target, ledger = forward_batch(model, batch, noise, device, seed, absolute_step, arm)
    mask = (batch[1] != original.KINEMATIC).to(device)
    loss = None
    try:
        loss = original.acceleration_loss(prediction, target, mask, pred_variance=head,
                                          loss_type="faithful", variance_floor=1e-6)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite continuation loss; optimizer update not executed")
        loss.backward()
        original.assert_finite_gradients(model)
        optimizer.step()
        if not original.tensors_are_finite(model.parameters()):
            raise FloatingPointError("Nonfinite continuation parameters after update; last committed checkpoint retained")
    except BaseException as error:
        error.graph_support_context = {"graph_ledger": ledger,
            "loss_text": repr(float(loss.detach().cpu())) if loss is not None else None}
        raise
    return float(loss.detach().cpu()), ledger


def parent_optimizer(model, parent_payload, device):
    """Clone the complete Adam and RNG states; never reset Adam moments."""
    model.load_state_dict(parent_payload["state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    optimizer.load_state_dict(copy.deepcopy(parent_payload["optimizer_state"]))
    original.restore_rng(parent_payload["rng_states"], device)
    for group in optimizer.param_groups:
        require(tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8 and group["weight_decay"] == 0., "Parent Adam hyperparameters differ")
        group["lr"] = LR
    require(model._encode_process_decode.detach_variance_features, "Faithful detached variance features required")
    return optimizer


def checkpoint_payload(model, optimizer, config, completed, history, device):
    require(0 <= completed <= EXTRA_UPDATES, "Continuation step out of range")
    original.synchronize(device)
    return {"format_version": 2, "graph_support_schema": SCHEMA,
        "state_dict": original.cpu_tree(model.state_dict()),
        "simulator_config": original.cpu_tree(model._checkpoint_config),
        "training_config": {"loss": "faithful", "graph_support_continuation": config,
            "parent_training_config": config["parent_training_config"],
            "completed_optimizer_updates": PARENT_UPDATES + completed},
        "continuation_config": config, "continuation_config_sha256": original.config_hash(config),
        "completed_additional_updates": completed, "completed_total_updates": PARENT_UPDATES + completed,
        "optimizer_state": original.cpu_tree(optimizer.state_dict()), "rng_states": original.cpu_tree(original.capture_rng(device)),
        "history": copy.deepcopy(history)}


def save_checkpoint(path, model, optimizer, config, completed, history, device):
    path = Path(path)
    require(not path.exists(), "Refusing to replace a prior checkpoint")
    payload = checkpoint_payload(model, optimizer, config, completed, history, device)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return {"path": path.name, "sha256": original.sha256(path), "completed_additional_updates": completed,
            "completed_total_updates": PARENT_UPDATES + completed}


def validate_continuation_payload(payload, expected_config=None, require_complete=True):
    """Callable by a separate evaluation wrapper; old full-run schema is absent."""
    require(payload.get("format_version") == 2 and payload.get("graph_support_schema") == SCHEMA, "Not a graph-support lineage checkpoint")
    require("full_training_schema" not in payload, "Old full-training schema must be absent from continuation lineage")
    config = payload["continuation_config"]
    require(payload["continuation_config_sha256"] == original.config_hash(config), "Continuation config hash differs")
    if expected_config is not None:
        require(config == expected_config, "Continuation configuration/source/data/parent identity differs")
    completed = payload["completed_additional_updates"]
    require(type(completed) is int and 0 <= completed <= EXTRA_UPDATES, "Invalid continuation updates")
    require(payload["completed_total_updates"] == PARENT_UPDATES + completed, "Total lineage update count differs")
    require(config["objective"] == "faithful" and config["arm"] in ("base", "mix"), "Unknown continuation identity")
    require(config["seed"] in PARENT_HASHES and config["parent_checkpoint_sha256"] == PARENT_HASHES[config["seed"]],
            "Unknown immutable parent checkpoint")
    require(config["lr"] == LR and config["batch_size"] == 2, "Fixed continuation optimizer/batch contract differs")
    require(config["parent_completed_updates"] == PARENT_UPDATES and config["additional_updates"] == EXTRA_UPDATES,
            "Fixed parent/additional budget differs")
    training = payload["training_config"]
    require(training["loss"] == "faithful" and training["completed_optimizer_updates"] == PARENT_UPDATES + completed,
            "Evaluable metadata lineage differs")
    require(training["graph_support_continuation"] == config and
            training["parent_training_config"] == config["parent_training_config"] and
            config["parent_training_config"]["loss"] == "faithful", "Nested training configuration differs")
    simulator = payload["simulator_config"]
    require(simulator_config_sha256(simulator) == config["parent_simulator_config_sha256"],
            "Simulator architecture/normalization differs from pinned parent configuration")
    require(simulator["detach_variance_features"] is True and simulator["uncertainty_parameterization"] == "variance"
            and simulator["variance_floor"] == 1e-6, "Faithful detached variance semantics differ")
    ledger = payload["history"]["graph_updates"]
    require(isinstance(ledger, list) and len(ledger) == completed, "Committed graph ledger/update count differs")
    for index, row in enumerate(ledger):
        require(type(row.get("absolute_schedule_step")) is int and row["absolute_schedule_step"] == PARENT_UPDATES + index
                and type(row.get("completed_before")) is int and row["completed_before"] == index
                and type(row.get("completed_additional_updates")) is int and row["completed_additional_updates"] == index + 1,
                "Committed graph ledger absolute-step schedule differs")
    if require_complete:
        require(completed == EXTRA_UPDATES, "Only the fixed 10k continuation endpoint is evaluable")
    return config, completed


def restore_checkpoint(path, expected_sha, model, optimizer, config, device):
    require(original.sha256(path) == expected_sha, "Resume checkpoint byte hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _, completed = validate_continuation_payload(payload, config, require_complete=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    original.restore_rng(payload["rng_states"], device)
    return completed, payload["history"]


def verify_pins(config):
    for path, digest in config["input_files_sha256"].items():
        require(original.sha256(path) == digest, f"Pinned source/data/parent changed: {path}")


def configure(args, parent, train_info, valid_info, device):
    parent_config = parent["run_config"]
    require(parent_config["seed"] == args.seed and parent_config["objective"] == "faithful", "Wrong parent objective/seed")
    require(train_info == parent_config["train"] and valid_info == parent_config["valid"], "Parent train/validation dataset identity differs")
    paths = [Path(__file__), Path(original.__file__), Path(bridge.__file__), Path(bridge.full.__file__),
             Path(bridge.same.__file__), REPO / "research/budget_graph.py", args.protocol, args.parent_protocol,
             args.parent_checkpoint, args.train_manifest, args.valid_manifest, args.metadata]
    paths += sorted((REPO / "adaptive-gns/gns").glob("*.py"))
    for manifest_path in (args.train_manifest, args.valid_manifest):
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["records"]:
            paths += [manifest_path.parent / row[key]["path"] for key in ("positions", "particle_types")]
    return {"schema": SCHEMA, "scope": "exploratory_faithful_graph_support_100k_to110k",
        "objective": "faithful", "arm": args.arm, "seed": args.seed,
        "parent_checkpoint_sha256": original.sha256(args.parent_checkpoint),
        "parent_run_config_sha256": parent["run_config_sha256"], "parent_training_config": parent["training_config"],
        "parent_simulator_config_sha256": simulator_config_sha256(parent["simulator_config"]),
        "parent_completed_updates": PARENT_UPDATES, "additional_updates": EXTRA_UPDATES,
        "batch_size": 2, "lr": LR, "noise_std": original.NOISE,
        "validation_frames": parent_config["validation_frames"], "validation_interval": SAVE_EVERY,
        "checkpoint_interval": SAVE_EVERY, "train": train_info, "valid": valid_info,
        "graph": {"mandatory": "native strict-r cap128/self candidates; preserve native directed prefix",
            "optional": "strict E_R minus strict E_r; random floor(.25*annulus) pairs; uncapped symmetric suffix",
            "mixture_probability_per_example": .5, "radius": .015, "radius_factor": 1.267,
            "coin_seed_material": "[20261005,seed,absolute_step,example_slot,4409]",
            "pair_seed_material": "[20261005,seed,absolute_step,example_slot,5501]"},
        "runtime": {**original.runtime_provenance(device, "scipy_host"), "threads": args.threads,
            "python": original.platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
        "protocol_sha256": original.sha256(args.protocol),
        "input_files_sha256": {str(path.resolve()): original.sha256(path) for path in paths}}


def run(args, model, optimizer, config, train, valid, device):
    output = args.output_dir
    require(not list(output.rglob("*.tmp")), "Stray temporary output requires preservation and review before any resume write")
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        require(args.resume and json.loads(protocol_path.read_text()) == config, "Existing output requires exact-provenance resume")
    else:
        require(not args.resume and not any(p.name != "run.lock" for p in output.iterdir()), "Missing protocol or existing unowned evidence")
        original.atomic_json(protocol_path, config)
    completed = 0
    history = {"training": [], "validation": [], "graph_updates": [], "elapsed_seconds": 0.}
    latest = None
    pointer = output / "latest.json"
    if args.resume:
        require(pointer.exists(), "Missing committed pointer; no automatic initialization or replay")
        latest = json.loads(pointer.read_text())
        require(Path(latest["path"]).name == latest["path"], "Unsafe checkpoint pointer")
        completed, history = restore_checkpoint(output / latest["path"], latest["sha256"], model, optimizer, config, device)
        require(latest["completed_additional_updates"] == completed and latest["completed_total_updates"] == PARENT_UPDATES + completed,
                "Checkpoint pointer update count differs")
    stop = EXTRA_UPDATES if args.stop_after is None else args.stop_after
    require(completed <= stop <= EXTRA_UPDATES, "Stopping boundary precedes checkpoint or exceeds fixed endpoint")
    attempt_id = uuid.uuid4().hex
    attempt_dir = output / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    start, prior_elapsed = time.perf_counter(), float(history["elapsed_seconds"])
    started_utc = original.utc_now()
    def status(state, error=None):
        value = {"state": state, "attempt_id": attempt_id, "pid": os.getpid(), "started_utc": started_utc,
            "updated_utc": original.utc_now(), "completed_additional_updates": completed,
            "completed_total_updates": PARENT_UPDATES + completed, "requested_additional_updates": EXTRA_UPDATES,
            "config_sha256": original.config_hash(config), "latest_checkpoint": latest,
            "elapsed_seconds": prior_elapsed + time.perf_counter() - start, "error": error}
        original.atomic_json(attempt_dir / "status.json", value)
        original.atomic_json(output / "status.json", value)
    def save_now():
        nonlocal latest
        history["elapsed_seconds"] = prior_elapsed + time.perf_counter() - start
        latest = save_checkpoint(output / f"checkpoint-extra-{completed:05d}.pt", model, optimizer, config, completed, history, device)
        original.atomic_json(pointer, latest)
    def validate_now():
        result = original.evaluate_validation(model, valid, config["validation_frames"], device)
        result.update(completed_additional_updates=completed, completed_total_updates=PARENT_UPDATES + completed,
                      continuation_config_sha256=original.config_hash(config))
        path = attempt_dir / f"validation-extra-{completed:05d}.json"
        original.atomic_json(path, result)
        history["validation"].append({"completed_additional_updates": completed,
            "record_file": str(path.relative_to(output)), "record_sha256": original.sha256(path),
            **result["equal_trajectory_mean"], "binned_vector_se_calibration_gap": result["binned_vector_se_calibration_gap"]})
    try:
        status("running")
        if latest is None:
            validate_now(); save_now()
        while completed < stop:
            absolute = PARENT_UPDATES + completed
            indices = original.sample_indices(args.seed, absolute, len(train), 2)
            batch = original.unpack_batch([train[index] for index in indices])
            noise = original.host_noise(batch[0].shape, batch[1], args.seed, absolute)
            attempted = {"absolute_schedule_step": absolute, "completed_before": completed,
                "frame_ids": [original.frame_identity(train, config["train"]["trajectory_ids"], index) for index in indices],
                "noise_sha256": bridge.full.state_hash(noise.numpy())}
            original.atomic_json(attempt_dir / "current_update.json", attempted)
            loss, graph_ledger = update_once(model, optimizer, batch, noise, device, args.seed, absolute, args.arm)
            completed += 1
            history["graph_updates"].append({**attempted, "completed_additional_updates": completed, "examples": graph_ledger})
            if completed % 100 == 0 or completed in (1, stop):
                row = {"completed_additional_updates": completed, "completed_total_updates": PARENT_UPDATES + completed,
                    "loss": loss, "lr": LR, "frame_ids": attempted["frame_ids"],
                    "elapsed_seconds": prior_elapsed + time.perf_counter() - start}
                history["training"].append(row); status("running"); print(json.dumps(row), flush=True)
            if completed % SAVE_EVERY == 0:
                validate_now(); save_now()
            elif completed == stop:
                save_now()
        verify_pins(config)
        original.atomic_json(output / "history.json", history)
        status("complete" if completed == EXTRA_UPDATES else "planned_stop_incomplete")
    except BaseException as error:
        # Attempt-scoped history and raw numerical state preserve replayed or
        # failed work without changing the last committed good checkpoint.
        original.atomic_json(attempt_dir / "unsuccessful_history.json", history)
        original.atomic_json(attempt_dir / "unsuccessful_context.json", getattr(error, "graph_support_context", {}))
        try:
            with (attempt_dir / "unsuccessful_state.pt").open("xb") as stream:
                torch.save({"state_dict": original.cpu_tree(model.state_dict()),
                    "optimizer_state": original.cpu_tree(optimizer.state_dict()),
                    "completed_additional_updates": completed, "error_type": type(error).__name__}, stream)
        except BaseException as preservation_error:
            original.atomic_json(attempt_dir / "state_preservation_error.json", {"error": str(preservation_error)})
        status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-checkpoint", "train-manifest", "valid-manifest", "metadata", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--arm", choices=("base", "mix"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--parent-protocol", type=Path, default=REPO / "research/protocols/full_waterdrop_100k.md")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Explicit planned incomplete stop; never changes fixed scientific endpoint")
    args = parser.parse_args()
    require(args.threads == 2, "Fixed two-thread runtime required")
    require(args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
            "Real fixed continuations require MPS and explicit disabled CPU fallback; synthetic helpers use CPU")
    if args.stop_after is not None:
        require(0 <= args.stop_after <= EXTRA_UPDATES, "Invalid planned stopping boundary")
    args.output_dir = args.output_dir.resolve()
    protected = (args.parent_checkpoint.resolve().parent.parent, args.train_manifest.resolve().parent, args.valid_manifest.resolve().parent)
    require(not any(args.output_dir == path or path in args.output_dir.parents or args.output_dir in path.parents for path in protected),
            "Continuation output must be separate from parent/data trees")
    torch.set_num_threads(args.threads)
    device = original.resolve_device(args.device)
    require(original.sha256(args.parent_checkpoint) == PARENT_HASHES[args.seed], "Parent checkpoint byte identity differs")
    parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
    metadata = json.loads(args.metadata.read_text())
    bridge.full.check_checkpoint(parent, "locked_test", args.parent_protocol, original.sha256(args.metadata))
    require(parent["completed_steps"] == PARENT_UPDATES and parent["run_config"]["objective"] == "faithful", "Fixed faithful100k parent required")
    for path, digest in parent["run_config"]["source_sha256"].items():
        require(original.sha256(REPO / path) == digest, "Frozen parent training source changed")
    train, _, train_info = original.load_manifest_dataset(args.train_manifest, "train")
    valid, _, valid_info = original.load_manifest_dataset(args.valid_manifest, "valid")
    require(original.validation_schedule(valid, valid_info["trajectory_ids"], 128) == parent["run_config"]["validation_frames"], "Original fixed validation schedule differs")
    config = configure(args, parent, train_info, valid_info, device)
    verify_pins(config)
    model, _ = load_for_evaluation(args.parent_checkpoint, metadata, device, radius_backend="scipy_host")
    optimizer = parent_optimizer(model, parent, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with original.RunLock(args.output_dir, clear_stale=args.clear_stale_lock):
        run(args, model, optimizer, config, train, valid, device)


if __name__ == "__main__":
    main()
