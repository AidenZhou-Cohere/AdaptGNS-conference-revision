#!/usr/bin/env python3
"""Opt-in synthetic validation of a hash-pinned128x10 simulator on CPU/CUDA.

Default: describe the fixed checks without importing Torch or repository code.
No datasets, supplied checkpoints, installers, network or training jobs.
"""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_cuda_synthetic_validation_v1"
CORE = {
    "device_utils.py": "324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326",
    "graph_network.py": "af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91",
    "learned_simulator.py": "216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf",
    "losses.py": "94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5",
    "model_io.py": "06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e",
}
TOLERANCES = {
    "forward": {"atol": 2e-5, "rtol": 2e-4},
    "loss": {"atol": 1e-5, "rtol": 2e-4},
    "gradient": {"atol": 5e-5, "rtol": 5e-4},
    "replay": {"atol": 5e-6, "rtol": 5e-5},
    "exact": {"atol": 0.0, "rtol": 0.0},
}
METADATA = {"dim": 2, "bounds": [[0.0, 1.0], [0.0, 1.0]],
            "default_connectivity_radius": 0.015, "vel_mean": [0.0, 0.0],
            "vel_std": [0.01, 0.012], "acc_mean": [0.0, 0.0], "acc_std": [0.001, 0.0012]}
REQUIRED = ("graph_parity", "objective_parity", "adam_rng_replay", "timing")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def description():
    return {"schema": SCHEMA, "status": "description_only", "scientific_readiness": False,
            "core_reference_sha256": CORE, "tolerances": TOLERANCES,
            "precision": "float32; TF32/AMP disabled; no torch.compile; Adam foreach=False/fused=False",
            "architecture": {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
            "graph": "explicit scipy_host, strict radius0.015, cap128 including self candidates",
            "parity_batch": {"counts": [16, 16], "history": 6, "dimension": 2, "objectives": ["faithful", "nll"]},
            "timing_batch": {"counts": [576, 576], "warmup_updates": 1, "measured_updates": 3,
                             "scope": "synthetic grid capacity measurement, not real-data throughput or a cohort forecast"},
            "replay": "two synthetic Adam updates per branch; in-memory serialization; CPU and selected CUDA backend",
            "tolerance_basis": "Conservative float32 engineering thresholds fixed before execution; allow ten-block reduction-order accumulation, not scientific equivalence. Exact checks cover ordered edges, serialized state restoration and RNG. Near-zero relative error is descriptive; the gate uses atol+rtol*abs(reference).",
            "unsupported": ["MPS", "AMP/TF32", "PyG radius backend", "DDP/multiple devices in one run",
                            "real data/checkpoints", "material features", "autonomous rollouts", "cross-device bitwise replay"],
            "required_sections": list(REQUIRED)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--describe", action="store_true")
    group.add_argument("--execute", action="store_true", help="Explicitly run bounded synthetic CPU/CUDA checks")
    parser.add_argument("--repo", type=Path, help="Research repository containing adaptive-gns/gns; required for execution")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.cuda_index < 0 or not 1 <= args.threads <= 16:
        parser.error("CUDA index must be nonnegative; CPU threads must be1..16")
    if args.execute and args.repo is None:
        parser.error("--execute requires --repo")
    return args


def verify_core(repo, expected=None):
    expected = CORE if expected is None else expected
    root = Path(repo).resolve() / "adaptive-gns" / "gns"
    actual = {name: sha(root / name) for name in expected}
    if actual != expected:
        raise ValueError("Core source identity differs; no execution allowed")
    return root, actual


def complete_status(sections):
    return "validation_passed" if set(sections) == set(REQUIRED) and all(
        sections[name].get("passed") is True for name in REQUIRED) else "validation_failed"


def numeric_gate(finite, maximum_tolerance_ratio, exact=False, bitwise_equal=False):
    if not finite:
        return False
    return bool(bitwise_equal) if exact else bool(
        maximum_tolerance_ratio is not None and math.isfinite(maximum_tolerance_ratio)
        and maximum_tolerance_ratio <= 1.0)


def diff(torch, reference, candidate, tolerance):
    rule = TOLERANCES[tolerance]
    row = {"tolerance": tolerance, "atol": rule["atol"], "rtol": rule["rtol"], "passed": False}
    if reference is None or candidate is None:
        return {**row, "reason": "missing tensor"}
    a, b = reference.detach().cpu(), candidate.detach().cpu()
    row.update(reference_shape=list(a.shape), candidate_shape=list(b.shape))
    if a.shape != b.shape or a.dtype != b.dtype:
        return {**row, "reason": "shape or dtype mismatch"}
    # Compare element bytes: ordinary torch.equal treats +0 and -0 as equal.
    row["bitwise_equal"] = bool(torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                           b.contiguous().reshape(-1).view(torch.uint8)))
    a, b = a.to(torch.float64), b.to(torch.float64)
    row["finite"] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not row["finite"]:
        return {**row, "reason": "nonfinite tensor", "nonfinite_reference": int((~torch.isfinite(a)).sum()),
                "nonfinite_candidate": int((~torch.isfinite(b)).sum())}
    delta = (a - b).abs()
    row["max_absolute_error"] = float(delta.max()) if a.numel() else 0.0
    row["max_relative_error_floor1e-12"] = float((delta / a.abs().clamp_min(1e-12)).max()) if a.numel() else 0.0
    row["relative_l2_error_floor1e-12"] = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a).clamp_min(1e-12))
    scale = rule["atol"] + rule["rtol"] * a.abs()
    ratio = torch.where(scale > 0, delta / scale.clamp_min(1e-300), torch.where(delta == 0, 0.0, float("inf")))
    maximum = float(ratio.max()) if a.numel() else 0.0
    row["maximum_tolerance_ratio"] = maximum if math.isfinite(maximum) else None
    row["outside_tolerance_elements"] = int((delta > scale).sum())
    row["passed"] = numeric_gate(row["finite"], row["maximum_tolerance_ratio"], tolerance == "exact", row["bitwise_equal"])
    return row


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(torch, item) for item in value)
    return value


def flatten_tensors(torch, value, prefix=""):
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten_tensors(torch, item, f"{prefix}/{key}"))
        return result
    if isinstance(value, (tuple, list)):
        return {key: tensor for index, item in enumerate(value)
                for key, tensor in flatten_tensors(torch, item, f"{prefix}/{index}").items()}
    return {}


def tensor_map_diff(torch, reference, candidate, tolerance):
    names_equal = set(reference) == set(candidate)
    rows = {name: diff(torch, reference.get(name), candidate.get(name), tolerance)
            for name in sorted(set(reference) | set(candidate))}
    return {"passed": names_equal and bool(rows) and all(row["passed"] for row in rows.values()),
            "names_equal": names_equal, "tensor_count": len(rows), "tensors": rows}


def fixture(torch, side=4):
    # Two independent regular grids; deterministic host generation, no source data.
    grid = torch.arange(side * side, dtype=torch.float32)
    base = torch.stack(((grid % side) * 0.006 + 0.2, (grid // side) * 0.006 + 0.2), dim=1)
    base = torch.cat((base, base + torch.tensor([0.3, 0.1])), dim=0)
    velocity, acceleration = torch.tensor([0.0007, -0.0003]), torch.tensor([1e-5, -2e-5])
    frames = torch.stack([base + step * velocity + 0.5 * step * step * acceleration for step in range(7)], dim=1)
    types = torch.full((len(base),), 5, dtype=torch.long)
    types[::7] = 3  # Exercise dynamic-particle masking; this is not a Sand fixture.
    generator = torch.Generator(device="cpu").manual_seed(9107 + side)
    noise = torch.randn((len(base), 6, 2), generator=generator) * 2e-5
    noise[types == 3] = 0
    return {"position_sequence": frames[:, :6].contiguous(), "next_positions": frames[:, 6].contiguous(),
            "position_sequence_noise": noise, "particle_types": types,
            "nparticles_per_example": torch.tensor([side * side, side * side], dtype=torch.long)}


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def make_model(torch, core, objective, device, state=None):
    model = core.build_simulator(METADATA, 6.7e-4, 6.7e-4, device,
                                connectivity_radius=0.015, nmessage_passing_steps=10,
                                uncertainty_parameterization="variance", variance_floor=1e-6,
                                detach_variance_features=objective == "faithful", radius_backend="scipy_host").to(device)
    if state is not None:
        model.load_state_dict(state, strict=True)
    config = model._checkpoint_config
    if not (config["latent_dim"] == 128 and config["nmessage_passing_steps"] == 10 and config["nmlp_layers"] == 2
            and model._radius_backend == "scipy_host" and model._max_num_neighbors == 128):
        raise ValueError("Actual model architecture/graph differs from validation contract")
    return model


def backward(torch, losses, model, batch, objective):
    model.train()
    model.zero_grad(set_to_none=True)
    prediction, variance, target = model(**batch, material_property=None, augment_radius_prob=0.0)
    loss = losses.acceleration_loss(prediction, target, batch["particle_types"] != 3,
                                    pred_variance=variance, loss_type=objective, variance_floor=1e-6)
    loss.backward()
    return {"prediction": prediction.detach(), "variance": variance.detach(), "target": target.detach(), "loss": loss.detach()}


def grads(torch, model):
    return {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()}


def graph_checks(torch, np, cpu_model, cuda_model, device):
    radius = np.float32(0.015)
    fixtures = {
        "strict_threshold": (np.asarray([[0, 0], [radius, 0], [np.nextafter(radius, np.float32(0)), 0],
                                           [np.nextafter(radius, np.float32(1)), 0], [0.2, 0.2]], dtype=np.float32), [5]),
        "coincident_cap_ties": (np.zeros((132, 2), dtype=np.float32), [132]),
        "batch_separation": (np.zeros((8, 2), dtype=np.float32), [4, 4]),
    }
    rows = {}
    for name, (points, counts) in fixtures.items():
        expected, offset = [[], []], 0
        for count in counts:
            local = points[offset:offset + count]
            for receiver in range(count):
                distance = np.linalg.norm(local - local[receiver], axis=1)
                neighbors = np.flatnonzero(distance < 0.015)
                neighbors = neighbors[np.lexsort((neighbors, distance[neighbors]))[:128]]
                expected[0].extend((neighbors + offset).tolist())
                expected[1].extend([receiver + offset] * len(neighbors))
            offset += count
        expected = torch.tensor(expected, dtype=torch.long)
        edges = []
        for model, backend in ((cpu_model, "cpu"), (cuda_model, device)):
            senders, receivers = model._compute_graph_connectivity(torch.tensor(points, device=backend),
                torch.tensor(counts, device=backend), 0.015, add_self_edges=True)
            edges.append(torch.stack((senders, receivers)).cpu())
        rows[name] = {"cpu_cuda": diff(torch, edges[0], edges[1], "exact"),
                      "independent_order": diff(torch, expected, edges[0], "exact"),
                      "directed_edges": int(expected.shape[1]),
                      "self_edges": int((expected[0] == expected[1]).sum())}
    return {"passed": all(value["cpu_cuda"]["passed"] and value["independent_order"]["passed"] for value in rows.values()),
            "fixtures": rows, "note": "Self candidates can be excluded by cap when132coincident nodes tie; expected nearest/index ordering is tested."}


def objective_checks(torch, losses, cpu_model, cuda_model, batch, objective, device):
    outputs, gradient_maps, actual_edges = [], [], []
    for model, backend in ((cpu_model, "cpu"), (cuda_model, device)):
        moved = move_batch(batch, backend)
        noisy = moved["position_sequence"] + moved["position_sequence_noise"]
        edges = model._encoder_preprocessor(noisy, moved["nparticles_per_example"], moved["particle_types"])[1]
        actual_edges.append(edges.detach().cpu())
        outputs.append(cpu_tree(torch, backward(torch, losses, model, moved, objective)))
        gradient_maps.append(grads(torch, model))
    metrics = {name: diff(torch, outputs[0][name], outputs[1][name], "loss" if name == "loss" else "forward")
               for name in outputs[0]}
    gradient_check = tensor_map_diff(torch, gradient_maps[0], gradient_maps[1], "gradient")
    normalization = tensor_map_diff(torch, flatten_tensors(torch, cpu_model._normalization_stats),
                                   flatten_tensors(torch, cuda_model._normalization_stats), "forward")
    edges = diff(torch, actual_edges[0], actual_edges[1], "exact")
    faithful = {"applicable": objective == "faithful", "passed": True}
    if objective == "faithful":
        faithful["backends"] = {}
        for index, (model, backend) in enumerate(((cpu_model, "cpu"), (cuda_model, device))):
            backward(torch, losses, model, move_batch(batch, backend), "mse")
            mse = grads(torch, model)
            names = [name for name in mse if "_variance_head" not in name]
            mean_check = tensor_map_diff(torch, {name: gradient_maps[index][name] for name in names},
                                        {name: mse[name] for name in names}, "gradient")
            head_names = [name for name in mse if "_variance_head" in name]
            head_detached = bool(head_names) and all(mse[name] is None for name in head_names)
            head_learns = any(gradient_maps[index][name] is not None and bool(torch.count_nonzero(gradient_maps[index][name])) for name in head_names)
            faithful["backends"][backend] = {"mean_matches_mse": mean_check, "mse_head_gradients_absent": head_detached,
                                            "faithful_head_has_nonzero_gradient": head_learns}
            faithful["passed"] &= mean_check["passed"] and head_detached and head_learns
    return {"passed": all(row["passed"] for row in metrics.values()) and gradient_check["passed"] and normalization["passed"] and edges["passed"] and faithful["passed"],
            "outputs": metrics, "per_parameter_gradients": gradient_check, "actual_batch_ordered_edges": edges,
            "normalization": normalization, "parameter_count": sum(parameter.numel() for parameter in cpu_model.parameters()),
            "faithful_gradient_control": faithful}


def rng_capture(torch, device):
    states = {"cpu": torch.get_rng_state().clone()}
    if str(device).startswith("cuda"):
        states["cuda"] = torch.cuda.get_rng_state(device).clone()
    return states


def rng_restore(torch, states, device):
    expected = {"cpu", "cuda"} if str(device).startswith("cuda") else {"cpu"}
    if set(states) != expected:
        raise ValueError("RNG state/backend identity differs")
    torch.set_rng_state(states["cpu"])
    if "cuda" in states:
        torch.cuda.set_rng_state(states["cuda"], device)


def optimizer(torch, model):
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0, foreach=False, fused=False)


def adam_state_complete(parameters, state, expected_step):
    parameters = list(parameters)
    if not parameters or set(state) != set(parameters):
        return False
    for parameter in parameters:
        entry = state[parameter]
        if not {"step", "exp_avg", "exp_avg_sq"} <= set(entry):
            return False
        if (entry["step"].device.type != "cpu" or entry["step"].numel() != 1
                or float(entry["step"]) != expected_step):
            return False
        for name in ("exp_avg", "exp_avg_sq"):
            moment = entry[name]
            if moment.device != parameter.device or moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                return False
    return True


def replay_check(torch, core, losses, objective, initial, batch, device):
    def advance(model, optim):
        moved = move_batch(batch, device)
        noise = torch.randn(batch["position_sequence_noise"].shape, device="cpu") * 1e-5
        moved["position_sequence_noise"] = moved["position_sequence_noise"] + noise.to(device)
        if str(device).startswith("cuda"):
            moved["position_sequence_noise"] += torch.randn(noise.shape, device=device) * 1e-5
        moved["position_sequence_noise"] *= (moved["particle_types"] != 3).view(-1, 1, 1)
        output = backward(torch, losses, model, moved, objective)
        if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite/missing synthetic replay gradient")
        optim.step()
        return cpu_tree(torch, output)

    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(8821).get_state())
    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(7731).get_state(), device)
    model = make_model(torch, core, objective, device, initial)
    optim = optimizer(torch, model)
    advance(model, optim)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    payload = {"model": cpu_tree(torch, model.state_dict()), "optimizer": cpu_tree(torch, optim.state_dict()),
               "rng": rng_capture(torch, device), "backend": str(device), "objective": objective, "completed_updates": 1}
    first_update_changed_parameters = any(not torch.equal(payload["model"][name], initial[name]) for name in initial)
    first_state_complete = adam_state_complete(model.parameters(), optim.state, 1)
    stream = io.BytesIO()
    torch.save(payload, stream)
    checkpoint_hash = hashlib.sha256(stream.getvalue()).hexdigest()
    uninterrupted = advance(model, optim)
    expected = {"model": cpu_tree(torch, model.state_dict()), "optimizer": cpu_tree(torch, optim.state_dict())}
    expected_rng = rng_capture(torch, device)
    restored = make_model(torch, core, objective, device, initial)
    restored_optim = optimizer(torch, restored)
    stream.seek(0)
    loaded = torch.load(stream, map_location="cpu", weights_only=True)
    if loaded["backend"] != str(device) or loaded["objective"] != objective or loaded["completed_updates"] != 1:
        raise ValueError("Synthetic in-memory checkpoint identity differs")
    restored.load_state_dict(loaded["model"], strict=True)
    restored_optim.load_state_dict(loaded["optimizer"])
    before = tensor_map_diff(torch, flatten_tensors(torch, payload), flatten_tensors(torch, loaded), "exact")
    rng_restore(torch, loaded["rng"], device)
    restored_before = {"model": cpu_tree(torch, restored.state_dict()), "optimizer": cpu_tree(torch, restored_optim.state_dict()),
                       "rng": rng_capture(torch, device)}
    restoration = tensor_map_diff(torch, flatten_tensors(torch, payload), flatten_tensors(torch, restored_before), "exact")
    hyperparameters = (payload["optimizer"]["param_groups"] == loaded["optimizer"]["param_groups"]
                       == restored_optim.state_dict()["param_groups"])
    resumed = advance(restored, restored_optim)
    actual = {"model": cpu_tree(torch, restored.state_dict()), "optimizer": cpu_tree(torch, restored_optim.state_dict())}
    state_check = tensor_map_diff(torch, flatten_tensors(torch, expected), flatten_tensors(torch, actual), "replay")
    rng_check = tensor_map_diff(torch, expected_rng, rng_capture(torch, device), "exact")
    output_check = tensor_map_diff(torch, uninterrupted, resumed, "replay")
    placements = adam_state_complete(restored.parameters(), restored_optim.state, 2)
    return {"passed": first_update_changed_parameters and first_state_complete and before["passed"] and restoration["passed"] and hyperparameters and state_check["passed"] and rng_check["passed"] and output_check["passed"] and placements,
            "backend": str(device), "objective": objective, "in_memory_checkpoint_sha256": checkpoint_hash,
            "serialized_tensors_exact": before, "loaded_model_optimizer_rng_exact": restoration,
            "optimizer_hyperparameters_exact": hyperparameters, "two_update_state_replay": state_check, "rng_replay_exact": rng_check,
            "first_update_changed_parameters": first_update_changed_parameters, "adam_state_complete_after_first_update": first_state_complete,
            "outputs": output_check, "adam_step2_and_moment_placement": placements,
            "note": "Bitwise flags are descriptive; numeric replay uses tighter float32 tolerance. No external checkpoint or disk file."}


def timing_check(torch, core, losses, initial, device, progress=None):
    progress = {"passed": False, "rows": []} if progress is None else progress
    model = make_model(torch, core, "faithful", device, initial)
    optim = optimizer(torch, model)
    batch, rows = fixture(torch, 24), progress["rows"]
    for index in range(4):
        row = {"index": index, "warmup": index == 0, "status": "incomplete"}
        rows.append(row)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        moved = move_batch(batch, device)
        torch.cuda.synchronize(device)
        transferred = time.perf_counter()
        output = backward(torch, losses, model, moved, "faithful")
        torch.cuda.synchronize(device)
        differentiated = time.perf_counter()
        if not bool(torch.isfinite(output["loss"])) or not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite synthetic timing update")
        optim.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite parameter after synthetic timing update")
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        graph_start = time.perf_counter()
        noisy = moved["position_sequence"] + moved["position_sequence_noise"]
        edges = model._compute_graph_connectivity(noisy[:, -1], moved["nparticles_per_example"], 0.015)
        torch.cuda.synchronize(device)
        graph_done = time.perf_counter()
        row.update({"status": "complete", "input_transfer_seconds": transferred - started,
                     "forward_backward_including_internal_graph_seconds": differentiated - transferred,
                     "gradient_guard_and_adam_seconds": finished - differentiated,
                     "full_guarded_update_seconds": finished - started,
                     "separately_timed_graph_call_seconds": graph_done - graph_start,
                     "directed_edges": int(edges[0].numel())})
    measured = [row["full_guarded_update_seconds"] for row in rows if not row["warmup"]]
    return {"passed": len(measured) == 3 and all(math.isfinite(value) and value > 0 for value in measured),
            "backend": str(device), "counts": [576, 576], "rows": rows,
            "median_guarded_update_seconds": statistics.median(measured),
            "scope": "Synthetic1152-node batch only; isolated graph call is additional and excluded from update time. Phase synchronization and finite guards are measured; not real-data throughput, transfer-of-dataset cost or a finish forecast."}


def execute(args):
    report = description()
    report.update(status="validation_failed", sections={})
    stage = "source_verification"
    try:
        root, actual = verify_core(args.repo)
        report["verified_core_sha256"] = actual
        stage = "dependency_import"
        import torch
        import numpy as np
        import scipy
        import torch_geometric
        if not torch.cuda.is_available() or args.cuda_index >= torch.cuda.device_count():
            report.update(status="not_run", reason="Requested CUDA device unavailable; no CPU/MPS fallback")
            return report
        device = f"cuda:{args.cuda_index}"
        sys.path.insert(0, str(root.parent))
        core = importlib.import_module("gns.model_io")
        losses = importlib.import_module("gns.losses")
        for filename in CORE:
            module = importlib.import_module(f"gns.{filename[:-3]}")
            if Path(module.__file__).resolve() != (root / filename).resolve():
                raise ValueError("Imported core module path differs from verified source")
        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.get_float32_matmul_precision() != "highest":
            raise ValueError("Requested float32/TF32 precision settings were not applied")
        properties = torch.cuda.get_device_properties(args.cuda_index)
        report["runtime"] = {"python": platform.python_version(), "os": platform.system(), "architecture": platform.machine(),
            "torch": str(torch.__version__), "cuda_build": torch.version.cuda, "numpy": np.__version__, "scipy": scipy.__version__,
            "torch_geometric": torch_geometric.__version__, "device": device, "gpu_name": properties.name,
            "capability": [properties.major, properties.minor], "gpu_memory_bytes": properties.total_memory,
            "compiled_architectures": torch.cuda.get_arch_list(), "threads": args.threads,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "precision": "float32 highest; TF32 false; no AMP; Adam foreach/fused false"}
        with torch.random.fork_rng(devices=[args.cuda_index]), torch.cuda.device(args.cuda_index):
            models, states = {}, {}
            for objective in ("faithful", "nll"):
                torch.set_rng_state(torch.Generator(device="cpu").manual_seed(20261006).get_state())
                cpu_model = make_model(torch, core, objective, "cpu")
                states[objective] = cpu_tree(torch, cpu_model.state_dict())
                cuda_model = make_model(torch, core, objective, device, states[objective])
                models[objective] = cpu_model, cuda_model
            stage = "graph_parity"
            report["sections"][stage] = graph_checks(torch, np, *models["faithful"], device)
            stage = "objective_parity"
            cases = {}
            report["sections"][stage] = {"passed": False, "objectives": cases}
            for objective in ("faithful", "nll"):
                stage = f"objective_parity/{objective}"
                cases[objective] = objective_checks(torch, losses, *models[objective], fixture(torch), objective, device)
            report["sections"]["objective_parity"]["passed"] = all(row["passed"] for row in cases.values())
            stage = "adam_rng_replay"
            cases = {}
            report["sections"][stage] = {"passed": False, "cases": cases}
            for objective in ("faithful", "nll"):
                for backend in ("cpu", device):
                    stage = f"adam_rng_replay/{objective}/{backend}"
                    cases[f"{objective}/{backend}"] = replay_check(torch, core, losses, objective, states[objective], fixture(torch), backend)
            report["sections"]["adam_rng_replay"]["passed"] = all(row["passed"] for row in cases.values())
            stage = "timing"
            if all(row["passed"] for row in report["sections"].values()):
                report["sections"][stage] = {"passed": False, "rows": []}
                report["sections"][stage] = timing_check(torch, core, losses, states["faithful"], device, report["sections"][stage])
            else:
                report["sections"][stage] = {"passed": False, "status": "not_run_after_validation_failure"}
        verify_core(args.repo)
        report["status"] = complete_status(report["sections"])
    except Exception as error:
        report["failure"] = {"stage": stage, "error_type": type(error).__name__}
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.execute:
        # Python-level diagnostics only; native extensions may write to OS streams.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            report = execute(args)
    else:
        report = description()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] in {"description_only", "validation_passed"} else 2 if report["status"] == "not_run" else 1


if __name__ == "__main__":
    raise SystemExit(main())
