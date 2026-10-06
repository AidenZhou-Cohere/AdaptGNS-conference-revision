#!/usr/bin/env python3
"""Bounded same-device CUDA graph-support semantics and exact checkpoint replay.

Default is description only. Uses only the three already-saved Sand training
batches; no trajectory loading, new data retrieval, CPU/CUDA tolerance change,
ReLU intervention, scientific checkpoint promotion or training admission.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_graph_support_cuda_validation_v1"
SEED = 0
CASES = ("small", "median", "large")
ARMS = ("base", "mix")
BRANCHES = ("old_native", "base", "mix")
PINS = {
    "train_sand_graph_support_cuda.py": "fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124",
    "train_sand_cuda_deterministic.py": "9dd376f6d4b6881290b134c8f410b89933bcdb026c9b52de601ba0ec7bb05a9b",
    "validate_cuda_execution.py": "21b2555e84a2c6221b0651c5bcc592b67a61ae5b7e3dd02da031054dd79e2a68",
    "validate_sand_cuda.py": "60083ba485bf0e99e9fef34b9c76d81a0ad109d517974d5ef25657203fb1081c",
    "diagnose_sand_relu_kinks.py": "dc1748b3e3f9ea32e28d16a8906c1cd0eaa4fff3a872741153e2244595f8555e",
}
REPORT_SHA = "1d46bd0044859a7f6f969275cdfaf91c57f4935d786453074fcecd2ec6e73245"
METADATA_SHA = "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0"
SHARED_FUNCTIONS = ("configure_cuda", "guarded_update", "assert_adam", "capture_rng", "restore_rng", "tree_equal")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_pinned(directory, filename):
    path = Path(directory) / filename
    require(sha(path) == PINS[filename], "Pinned helper differs: " + filename)
    spec = importlib.util.spec_from_file_location("_graph_cuda_check_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_function_check(directory):
    functions = []
    for filename in ("train_sand_cuda_deterministic.py", "train_sand_graph_support_cuda.py"):
        nodes = ast.parse((Path(directory) / filename).read_text()).body
        functions.append({node.name: ast.dump(node, include_attributes=False) for node in nodes
                          if isinstance(node, ast.FunctionDef) and node.name in SHARED_FUNCTIONS})
    return set(functions[0]) == set(SHARED_FUNCTIONS) and functions[0] == functions[1]


def describe():
    return {"schema": SCHEMA, "status": "description_only", "seed": SEED, "objective": "faithful",
            "cases": list(CASES), "branches": list(BRANCHES), "saved_report_sha256": REPORT_SHA,
            "data": "exact three prior saved training batch NPZs only; no trajectory/validation/test loading",
            "semantics": "one deterministic CUDA device; old-native versus new-base bytewise forward/target/loss/all gradients/first Adam; mix exact native prefix, annulus budget, RNG and ledger, finite update",
            "replay": "base and mix: small batch, steps0 and1, new-schema checkpoint after step0; bytewise CPU/CUDA RNG restoration, random probes and resumed/uninterrupted step1",
            "comparison": "pinned comparator exact, including signed zero; no CPU/CUDA numerical comparison",
            "max_optimizer_updates": 15, "scientific_training_admitted": False,
            "prior_cpu_cuda_validation_status": "validation_failed", "source_pins": PINS}


def exact(h, compare, a, b):
    return compare.tensor_map_diff(h.torch, compare.flatten_tensors(h.torch, a),
                                   compare.flatten_tensors(h.torch, b), "exact")


def raw_batch(batch):
    return tuple(batch[key] for key in ("position_sequence", "particle_types", "nparticles_per_example", "next_positions"))


def optimizer(h, model):
    return h.torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                             weight_decay=0., foreach=False, fused=False)


def graph_audit(h, trainer, support, compare, noisy, counts, native, actual, ledger, step, arm, device):
    """Check captured network inputs against the fixed graph definition and RNG."""
    torch, np, bridge = h.torch, h.np, support.bridge
    before_rng = trainer.capture_rng(torch, device)
    repeated, repeated_ledger = support.append_optional_edges(noisy, counts, native, .015, SEED, step, arm)
    rng_unchanged = trainer.tree_equal(torch, before_rng, trainer.capture_rng(torch, device))
    points = noisy[:, -1].detach().cpu().numpy()
    native_np, actual_np = native.detach().cpu().numpy(), actual.detach().cpu().numpy()
    partition = counts.detach().cpu().numpy()
    offsets = np.r_[0, np.cumsum(partition)]
    additions, expected_ledger = [], []
    for slot, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        start, end = int(start), int(end)
        local = points[start:end]
        graph = bridge.strict_pairs(local, .015)
        coin_material, pair_material = [20261005, SEED, step, slot, 4409], [20261005, SEED, step, slot, 5501]
        coin = bool(np.random.default_rng(np.random.SeedSequence(coin_material)).random() < .5)
        active, budget = arm == "mix" and coin, math.floor(.25 * len(graph.extra))
        rng = np.random.default_rng(np.random.SeedSequence(pair_material))
        optional = graph.extra[rng.permutation(len(graph.extra))[:budget]] if active else np.empty((0, 2), dtype=np.int64)
        optional = bridge.same.canonical_pairs(optional)
        local_native = native_np[:, (native_np[1] >= start) & (native_np[1] < end)] - start
        require(np.array_equal(local_native, bridge.ordered_edges(local, graph.base, True, 128)),
                "Native graph differs from strict radius/cap/self convention")
        optional_edges = bridge.ordered_edges(local, optional, False)
        if optional_edges.size:
            additions.append(optional_edges + start)
        degree = np.bincount(bridge.ordered_edges(local, graph.base, True)[1], minlength=len(local))
        expected_ledger.append({"example_slot": slot, "n_particles": len(local), "exposure_coin": coin,
            "expanded": active, "coin_seed_material": coin_material, "pair_seed_material": pair_material,
            "native_directed_edges": int(local_native.shape[1]), "native_self_edges": int(np.sum(local_native[0] == local_native[1])),
            "receivers_above_native_cap": int(np.sum(degree > 128)), "annulus_pairs": len(graph.extra),
            "optional_budget_if_exposed": budget, "selected_optional_pairs": len(optional),
            "native_edge_sha256": bridge.full.state_hash(local_native), "optional_pair_sha256": bridge.full.state_hash(optional),
            "noisy_current_sha256": bridge.full.state_hash(local)})
    expected = np.concatenate((native_np, *additions), axis=1) if additions else native_np
    membership = np.repeat(np.arange(len(partition)), partition)
    checks = {"native_prefix_exact": np.array_equal(actual_np[:, :native_np.shape[1]], native_np),
              "ordered_edges_and_budget_exact": np.array_equal(actual_np, expected),
              "partition_preserved": bool(np.all(membership[actual_np[0]] == membership[actual_np[1]])),
              "ledger_exact": ledger == expected_ledger == repeated_ledger,
              "graph_torch_rng_unchanged": rng_unchanged,
              "zero_addition_tensor_identity": bool(additions) or repeated is native}
    repeat_check = compare.diff(torch, actual, repeated, "exact")
    return {"passed": all(checks.values()) and repeat_check["passed"], **checks,
            "repeat_edges": repeat_check, "examples": ledger,
            "native_directed_edges": int(native_np.shape[1]), "actual_directed_edges": int(actual_np.shape[1])}


def advance(h, trainer, support, compare, model, opt, batch, noise, device, step, arm, graph_step=None):
    """One ordinary faithful trainer update with read-only network-input capture."""
    torch = h.torch
    graph_step = step if graph_step is None else graph_step
    model.train()
    require(model._encode_process_decode.detach_variance_features is True, "Faithful detached variance required")
    trainer.assert_adam(torch, model, opt, step)
    opt.param_groups[0]["lr"] = h.learning_rate(step)
    opt.zero_grad(set_to_none=True)
    initial = h.cpu_tree(model.state_dict())
    raw = raw_batch(batch)
    noisy = raw[0].to(device) + noise.to(device)
    counts, types = raw[2].to(device), raw[1].to(device)
    with torch.no_grad():
        native_nodes, native_edges, native_features = model._encoder_preprocessor(noisy, counts, types, None,
            augment_radius_prob=0., augment_radius_factor=1.267)
    captured = []
    def capture(module, arguments):
        require(len(arguments) == 3, "Expected nodes, edges and edge features")
        captured.append(h.cpu_tree(dict(zip(("nodes", "edges", "features"), arguments))))
    handle = model._encode_process_decode.register_forward_pre_hook(capture)
    rng_before = trainer.capture_rng(torch, device)
    try:
        if arm == "old_native":
            pred, head, target = h.forward_batch(model, raw, noise, device)
            ledger = None
        else:
            pred, head, target, ledger = support.forward_batch(model, raw, noise, device, SEED, graph_step, arm)
    finally:
        handle.remove()
    require(len(captured) == 1, "Exactly one network invocation required")
    forward_rng_unchanged = trainer.tree_equal(torch, rng_before, trainer.capture_rng(torch, device))
    graph = captured[0]
    if arm == "old_native":
        graph_check = exact(h, compare, {"nodes": native_nodes, "edges": native_edges, "features": native_features}, graph)
    else:
        graph_check = graph_audit(h, trainer, support, compare, noisy, counts, native_edges,
                                 graph["edges"], ledger, graph_step, arm, device)
        graph_check["native_nodes_exact"] = compare.diff(torch, native_nodes, graph["nodes"], "exact")
        graph_check["native_features_prefix_exact"] = compare.diff(torch, native_features,
            graph["features"][:native_features.shape[0]], "exact")
        graph_check["passed"] &= graph_check["native_nodes_exact"]["passed"] and graph_check["native_features_prefix_exact"]["passed"]
    try:
        loss = trainer.guarded_update(h, model, opt, pred, head, target, types != h.KINEMATIC, "faithful", step + 1)
    except BaseException as error:
        error.graph_validation_evidence = {"arm": arm, "step": step, "graph_step": graph_step,
            "before": h.cpu_tree({"prediction": pred, "variance": head, "target": target}),
            "graph": graph, "graph_check": graph_check, "gradients": compare.grads(torch, model),
            "model_state_at_failure": h.cpu_tree(model.state_dict()), "optimizer_at_failure": h.cpu_tree(opt.state_dict())}
        raise
    result = {"before": h.cpu_tree({"prediction": pred, "variance": head, "target": target, "loss": loss}),
              "graph": graph, "gradients": compare.grads(torch, model), "after_state": h.cpu_tree(model.state_dict()),
              "optimizer": h.cpu_tree(opt.state_dict()), "rng_after": trainer.capture_rng(torch, device)}
    result["parameter_update"] = {name: result["after_state"][name] - initial[name] for name, _ in model.named_parameters()}
    require(all(value is not None for value in result["gradients"].values()), "Every parameter requires a gradient")
    finite = all(bool(torch.isfinite(value).all()) for value in compare.flatten_tensors(torch, result).values())
    summary = {"passed": finite and graph_check["passed"] and forward_rng_unchanged, "finite": finite,
               "forward_rng_unchanged": forward_rng_unchanged, "graph": graph_check, "completed_steps": step + 1}
    return result, summary, ledger


def save_evidence(torch, path, payload):
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": Path(path).name, "sha256": sha(path)}


def branch_comparison(h, trainer, compare, native, base):
    checks = {name: exact(h, compare, native[name], base[name]) for name in
              ("before", "graph", "gradients", "after_state", "optimizer", "parameter_update", "rng_after")}
    full_optimizer = trainer.tree_equal(h.torch, native["optimizer"], base["optimizer"])
    return {"passed": all(row["passed"] for row in checks.values()) and full_optimizer,
            "checks": checks, "optimizer_metadata_exact": full_optimizer}


def frame_ids(case):
    return [f"{identity}:{target}" for identity, target in zip(case["trajectory_ids"], case["target_frames"])]


def history_record(support, case, noise, step, ledger):
    return {"completed_steps": step + 1, "absolute_schedule_step": step, "frame_ids": frame_ids(case),
            "noise_sha256": support.bridge.full.state_hash(noise.numpy()), "examples": ledger}


def rng_probe(h, device):
    return {"cpu": h.torch.rand(17, device="cpu"), "cuda": h.torch.rand(17, device=device).cpu()}


def replay(h, trainer, support, compare, make_model, initial, initial_rng, batch, case, device, arm, directory):
    torch = h.torch
    model = make_model(initial)
    opt = optimizer(h, model)
    trainer.restore_rng(torch, initial_rng, device)
    noise0 = batch["position_sequence_noise"]
    first, first_check, ledger = advance(h, trainer, support, compare, model, opt, batch, noise0, device, 0, arm)
    history = {"training": [], "graph_updates": [history_record(support, case, noise0, 0, ledger)], "elapsed_seconds": 0.0}
    config = {"objective": "faithful", "arm": arm, "seed": SEED, "updates": 100000,
              "graph_exposure": trainer.graph_exposure_config(arm), "purpose": "bounded_validation_only"}
    # checkpoint_payload references history; detach the snapshot before any next step.
    payload = copy.deepcopy(trainer.checkpoint_payload(h, model, opt, config, 1, history, device))
    checkpoint = Path(directory) / (arm + "_replay_checkpoint_step1.pt")
    checkpoint_artifact = save_evidence(torch, checkpoint, payload)
    noise1 = h.host_noise(batch["position_sequence"].shape, batch["particle_types"], SEED, 1)
    probe_expected = rng_probe(h, device)
    expected, expected_check, expected_ledger = advance(h, trainer, support, compare, model, opt, batch, noise1, device, 1, arm)
    expected_history = copy.deepcopy(history)
    expected_history["graph_updates"].append(history_record(support, case, noise1, 1, expected_ledger))
    trainer.validate_graph_history(expected_history, 2, arm, SEED)
    expected_payload = copy.deepcopy(trainer.checkpoint_payload(h, model, opt, config, 2, expected_history, device))
    del model, opt
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    serialized_exact = trainer.tree_equal(torch, payload, loaded)
    restored = make_model(initial)
    restored_opt = optimizer(h, restored)
    completed, restored_history = trainer.restore_payload(h, loaded, restored, restored_opt, config, device)
    restoration = {"completed_step": completed == 1,
        "model": trainer.tree_equal(torch, payload["state_dict"], h.cpu_tree(restored.state_dict())),
        "optimizer": trainer.tree_equal(torch, payload["optimizer_state"], h.cpu_tree(restored_opt.state_dict())),
        "rng": trainer.tree_equal(torch, payload["rng_states"], trainer.capture_rng(torch, device)),
        "history": restored_history == payload["history"]}
    probe_actual = rng_probe(h, device)
    actual, actual_check, actual_ledger = advance(h, trainer, support, compare, restored, restored_opt, batch, noise1, device, 1, arm)
    restored_history["graph_updates"].append(history_record(support, case, noise1, 1, actual_ledger))
    actual_payload = copy.deepcopy(trainer.checkpoint_payload(h, restored, restored_opt, config, 2, restored_history, device))
    replay_check = branch_comparison(h, trainer, compare, expected, actual)
    probes = exact(h, compare, probe_expected, probe_actual)
    final_payload_exact = trainer.tree_equal(torch, expected_payload, actual_payload)
    artifact = save_evidence(torch, Path(directory) / (arm + "_replay_evidence.pt"),
        {"first": first, "expected": expected, "resumed": actual, "probe_expected": probe_expected,
         "probe_resumed": probe_actual, "expected_payload": expected_payload, "resumed_payload": actual_payload})
    passed = (all(row["passed"] for row in (first_check, expected_check, actual_check, replay_check, probes))
              and serialized_exact and all(restoration.values()) and final_payload_exact)
    return {"passed": passed, "checkpoint": checkpoint_artifact, "evidence": artifact,
            "serialized_payload_exact": serialized_exact, "restoration": restoration,
            "steps": {"first": first_check, "uninterrupted": expected_check, "resumed": actual_check},
            "rng_probes": probes, "resumed_update": replay_check, "final_payload_exact": final_payload_exact,
            "graph_ledger_exact": expected_ledger == actual_ledger,
            "checkpoint_snapshot_history_length": len(payload["history"]["graph_updates"])}


def completion_gate(report):
    cases = report.get("cases", [])
    return (report.get("shared_trainer_functions_exact") is True
            and [row.get("case") for row in cases] == list(CASES)
            and all(set(row.get("branches", {})) == set(BRANCHES)
                    and all(row["branches"][arm].get("passed") is True for arm in BRANCHES)
                    and row.get("old_native_vs_base", {}).get("passed") is True
                    and row.get("mix_target_exact", {}).get("passed") is True for row in cases)
            and any(example.get("selected_optional_pairs", 0) > 0 for row in cases
                    for example in row.get("branches", {}).get("mix", {}).get("graph", {}).get("examples", []))
            and set(report.get("replay", {})) == set(ARMS)
            and all(report["replay"][arm].get("passed") is True for arm in ARMS))


def execute(args):
    require(sha(args.validation_report) == REPORT_SHA, "Exact prior failed report required")
    require(sha(args.metadata) == METADATA_SHA, "Exact prior metadata required")
    previous = json.loads(args.validation_report.read_text())
    require(previous.get("status") == "validation_failed" and [r["case"] for r in previous["cases"]] == list(CASES),
            "Prior failed three-case validation is not to be changed")
    directory = Path(__file__).resolve().parent
    modules = {name: import_pinned(directory, name) for name in PINS}
    trainer, old = modules["train_sand_graph_support_cuda.py"], modules["train_sand_cuda_deterministic.py"]
    compare, actual, saved = modules["validate_cuda_execution.py"], modules["validate_sand_cuda.py"], modules["diagnose_sand_relu_kinks.py"]
    require(shared_function_check(directory), "Old/new trainer numeric or CUDA helpers differ")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    started = time.perf_counter()
    report = {**describe(), "status": "in_progress", "source_sha256": sha(__file__), "cases": [], "replay": {},
              "shared_trainer_functions_exact": True, "all_inputs_reverified": False}
    def save():
        trainer.atomic_json(args.output_dir / "report.json", report)
    save()
    try:
        h, support = trainer.load_helpers(args.repo)
        device, runtime = trainer.configure_cuda(h, args)
        report["runtime"] = runtime
        metadata = json.loads(args.metadata.read_text())
        h.torch.manual_seed(SEED)
        with h.torch.cuda.device(device):
            h.torch.cuda.manual_seed(SEED)
        initial_model = actual.make_model(h, metadata, "faithful", device)
        initial = h.cpu_tree(initial_model.state_dict())
        initial_rng = trainer.capture_rng(h.torch, device)
        report["initial_state"] = save_evidence(h.torch, args.output_dir / "initial_state_seed0.pt", initial)
        del initial_model
        make_model = lambda state: actual.make_model(h, metadata, "faithful", device, state)
        inputs = {args.validation_report: REPORT_SHA, args.metadata: METADATA_SHA,
                  Path(__file__): report["source_sha256"], **{directory / name: value for name, value in PINS.items()},
                  **{args.repo / name: value for name, value in trainer.SOURCE_PINS.items()}}
        batches = []
        for index, case in enumerate(previous["cases"]):
            batch, path = saved.check_saved_batch(h, case, args.batch_dir)
            inputs[path] = case["batch_sha256"]
            batches.append((batch, case))
            entry = {"case": case["case"], "batch_file": path.name, "batch_sha256": sha(path),
                     "graph_schedule_step": index, "particle_counts": case["particle_counts"], "branches": {}}
            report["cases"].append(entry)
            save()
            results = {}
            for arm in BRANCHES:
                model = make_model(initial)
                opt = optimizer(h, model)
                trainer.restore_rng(h.torch, initial_rng, device)
                # Every semantics branch is a first Adam step. The graph schedule
                # uses the saved-case index; Adam's update schedule remains zero.
                # Graph schedule and optimizer update are independent in this check.
                result, check, _ = advance(h, old if arm == "old_native" else trainer, support, compare,
                    model, opt, batch, batch["position_sequence_noise"], device, 0, arm, graph_step=index)
                results[arm] = result
                check["evidence"] = save_evidence(h.torch, args.output_dir / f"{case['case']}_{arm}.pt", result)
                entry["branches"][arm] = check
                del model, opt
                save()
            entry["old_native_vs_base"] = branch_comparison(h, trainer, compare, results["old_native"], results["base"])
            entry["mix_target_exact"] = compare.diff(h.torch, results["base"]["before"]["target"], results["mix"]["before"]["target"], "exact")
            del results
            save()
        for arm in ARMS:
            report["replay"][arm] = replay(h, trainer, support, compare, make_model, initial, initial_rng,
                batches[0][0], batches[0][1], device, arm, args.output_dir)
            save()
        for path, expected in inputs.items():
            require(sha(path) == expected, "Source or saved input changed: " + str(path))
        passed = completion_gate(report)
        report.update(status="implementation_passed" if passed else "implementation_failed",
                      all_inputs_reverified=True, elapsed_seconds=time.perf_counter() - started,
                      scientific_training_admitted=False)
        save()
        return 0 if passed else 2
    except BaseException as error:
        if hasattr(error, "graph_validation_evidence"):
            report["failed_update_evidence"] = save_evidence(h.torch, args.output_dir / "failed_update_evidence.pt",
                                                             error.graph_validation_evidence)
        report.update(status="implementation_error", error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter() - started, failed_attempt_preserved=True)
        save()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    for name in ("repo", "validation-report", "batch-dir", "metadata", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.execute and any(getattr(args, name) is None for name in ("repo", "validation_report", "batch_dir", "metadata", "output_dir")):
        parser.error("Execution requires all five source/input/output paths")
    if args.cuda_index < 0 or not 1 <= args.threads <= 16:
        parser.error("CUDA index must be nonnegative; threads must be 1..16")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.execute:
        raise SystemExit(execute(args))
    print(json.dumps(describe(), indent=2))
