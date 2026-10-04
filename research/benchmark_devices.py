#!/usr/bin/env python3
"""Benchmark the full original-size GNS on CPU and Apple Metal, without core edits.

The CPU SciPy graph is passed into an instance-local callback so that the same
core feature construction, 128-wide / 10-layer GNN, likelihood, and optimizer
can execute on both devices. This is not native MPS neighbor construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import types
from pathlib import Path

# Do not silently send unsupported Metal kernels back to CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import scipy
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "adaptive-gns"))
from gns.losses import acceleration_loss
from gns.model_io import build_simulator
from gns.noise_utils import get_random_walk_noise_for_position_sequence


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def describe(samples: list[float]) -> dict:
    q25, median, q75 = np.quantile(samples, [0.25, 0.5, 0.75])
    return {"median_ms": float(median), "q25_ms": float(q25), "q75_ms": float(q75), "raw_ms": samples, "n": len(samples)}


def fixed_graph_callback(self, node_features, nparticles_per_example, radius, add_self_edges=True):
    # Benchmark-only adapter. The caller must supply the graph for this exact
    # noisy batch. It replaces neighbor construction, not feature/GNN kernels.
    return self._benchmark_senders, self._benchmark_receivers


def host_batch(data: dict, frame: int, batch_examples: int, noise_std: float, graph_fn, seed: int) -> dict:
    positions, targets, particle_types, counts = [], [], [], []
    for index in range(batch_examples):
        trajectory = data[f"position_{index}"]
        positions.append(torch.from_numpy(np.ascontiguousarray(trajectory[frame - 5:frame + 1].transpose(1, 0, 2))))
        targets.append(torch.from_numpy(np.ascontiguousarray(trajectory[frame + 1])))
        particle_types.append(torch.from_numpy(data[f"type_{index}"].astype(np.int64, copy=False)))
        counts.append(trajectory.shape[1])
    position = torch.cat(positions).float()
    target = torch.cat(targets).float()
    kinds = torch.cat(particle_types).long()
    count = torch.tensor(counts, dtype=torch.long)
    # Identical reproducible host-side noise feeds CPU and MPS.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        noise = get_random_walk_noise_for_position_sequence(position, noise_std)
    noise *= (kinds != 3).view(-1, 1, 1)
    senders, receivers = graph_fn((position + noise)[:, -1], count, 0.015)
    return {"position": position, "target": target, "particle_types": kinds, "counts": count, "noise": noise, "senders": senders, "receivers": receivers}


def on_device(batch: dict, device: str) -> dict:
    return {name: value.to(device) for name, value in batch.items()}


def forward(model, batch: dict):
    model._benchmark_senders = batch["senders"]
    model._benchmark_receivers = batch["receivers"]
    return model.predict_accelerations(
        next_positions=batch["target"], position_sequence_noise=batch["noise"],
        position_sequence=batch["position"], nparticles_per_example=batch["counts"],
        particle_types=batch["particle_types"])


def training_step(model, optimizer, batch: dict) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    pred, head, target = forward(model, batch)
    loss = acceleration_loss(pred, target, batch["particle_types"] != 3,
                             pred_variance=head, loss_type="nll")
    loss.backward()
    optimizer.step()
    return loss.detach()


def compare_arrays(cpu: np.ndarray, mps: np.ndarray, atol: float, rtol: float) -> dict:
    difference = mps.astype(np.float64) - cpu.astype(np.float64)
    denominator = float(np.linalg.norm(cpu.astype(np.float64).ravel()))
    return {"finite_cpu": bool(np.isfinite(cpu).all()), "finite_mps": bool(np.isfinite(mps).all()), "max_abs_difference": float(np.max(np.abs(difference))), "relative_l2_difference": float(np.linalg.norm(difference.ravel()) / max(denominator, 1e-30)), "atol": atol, "rtol": rtol, "allclose": bool(np.allclose(cpu, mps, atol=atol, rtol=rtol))}


def parity(models: dict, initial_state: dict, batches: dict) -> dict:
    values = {}
    for device, model in models.items():
        model.load_state_dict(initial_state)
        model.train()
        model.zero_grad(set_to_none=True)
        pred, head, target = forward(model, batches[device])
        loss = acceleration_loss(pred, target, batches[device]["particle_types"] != 3,
                                 pred_variance=head, loss_type="nll")
        loss.backward()
        synchronize(device)
        gradients = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
        values[device] = {"prediction": pred.detach().cpu().numpy(), "variance_head": head.detach().cpu().numpy(), "normalized_target": target.detach().cpu().numpy(), "loss": np.asarray([loss.detach().cpu().item()]), "parameter_gradients": gradients.cpu().numpy()}
        model.zero_grad(set_to_none=True)
        del pred, head, target, loss, gradients
    result = {}
    for name in values["cpu"]:
        atol, rtol = (2e-5, 2e-3) if name == "parameter_gradients" else (2e-5, 2e-4)
        result[name] = compare_arrays(values["cpu"][name], values["mps"][name], atol, rtol)
    result["all_checks_pass"] = all(v["allclose"] and v["finite_cpu"] and v["finite_mps"] for v in result.values())
    result["cpu_loss"] = float(values["cpu"]["loss"][0])
    result["mps_loss"] = float(values["mps"]["loss"][0])
    return result


def detailed_parity(models: dict, initial_state: dict, batches: dict) -> dict:
    """Inspect gradient discrepancies and ReLU branch flips without timing."""
    saved, cpu_relu, relu_differences = {}, {}, []
    for device, model in models.items():
        model.load_state_dict(initial_state)
        model.train()
        model.zero_grad(set_to_none=True)
        hooks = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.ReLU):
                continue
            def collect(module, inputs, output, name=name, device=device):
                value = inputs[0].detach().cpu().numpy()
                if device == "cpu":
                    cpu_relu[name] = value.copy()
                else:
                    reference = cpu_relu[name]
                    mismatch = (reference > 0) != (value > 0)
                    relu_differences.append({"module": name, "elements": int(value.size), "sign_flips": int(mismatch.sum()), "max_abs_cpu_input_at_flip": float(np.abs(reference[mismatch]).max()) if mismatch.any() else 0.0, "max_abs_mps_input_at_flip": float(np.abs(value[mismatch]).max()) if mismatch.any() else 0.0})
            hooks.append(module.register_forward_hook(collect))
        pred, head, target = forward(model, batches[device])
        loss = acceleration_loss(pred, target, batches[device]["particle_types"] != 3,
                                 pred_variance=head, loss_type="nll")
        loss.backward()
        synchronize(device)
        gradients = {name: p.grad.detach().cpu().numpy().copy() for name, p in model.named_parameters() if p.grad is not None}
        for hook in hooks:
            hook.remove()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.step()
        synchronize(device)
        updated = {name: p.detach().cpu().numpy().copy() for name, p in model.named_parameters()}
        saved[device] = {"gradients": gradients, "updated": updated, "prediction": pred.detach().cpu().numpy(), "loss": float(loss.detach().cpu()), "gradient_parameter_count": sum(g.size for g in gradients.values())}
        model.zero_grad(set_to_none=True)
        del pred, head, target, loss, optimizer
    names = list(saved["cpu"]["gradients"])
    cg = np.concatenate([saved["cpu"]["gradients"][name].ravel() for name in names])
    mg = np.concatenate([saved["mps"]["gradients"][name].ravel() for name in names])
    absolute = np.abs(mg.astype(np.float64) - cg.astype(np.float64))
    tensor_comparisons = []
    for name in names:
        a, b = saved["cpu"]["gradients"][name], saved["mps"]["gradients"][name]
        tensor_comparisons.append({"name": name, "shape": list(a.shape), "cpu_l2_norm": float(np.linalg.norm(a.astype(np.float64))), "mps_l2_norm": float(np.linalg.norm(b.astype(np.float64))), **compare_arrays(a, b, 2e-5, 2e-3)})
    initial = {name: value.numpy() for name, value in initial_state.items()}
    update_names = list(saved["cpu"]["updated"])
    cpu_updates = np.concatenate([(saved["cpu"]["updated"][name] - initial[name]).ravel() for name in update_names])
    mps_updates = np.concatenate([(saved["mps"]["updated"][name] - initial[name]).ravel() for name in update_names])
    # Repeat MPS's backward from exactly the same weights and inputs to expose
    # device-internal nondeterminism separately from CPU-versus-MPS rounding.
    model = models["mps"]
    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)
    pred, head, target = forward(model, batches["mps"])
    loss = acceleration_loss(pred, target, batches["mps"]["particle_types"] != 3,
                             pred_variance=head, loss_type="nll")
    loss.backward()
    synchronize("mps")
    repeated = np.concatenate([p.grad.detach().cpu().numpy().ravel() for name, p in model.named_parameters() if p.grad is not None])
    result = {
        "prediction": compare_arrays(saved["cpu"]["prediction"], saved["mps"]["prediction"], 2e-5, 2e-4),
        "cpu_loss": saved["cpu"]["loss"], "mps_loss": saved["mps"]["loss"],
        "gradient_strict_original_threshold": compare_arrays(cg, mg, 2e-5, 2e-3),
        "gradient_looser_descriptive_threshold_not_a_certification": compare_arrays(cg, mg, 1e-3, 5e-3),
        "gradient_cpu_l2_norm": float(np.linalg.norm(cg.astype(np.float64))),
        "gradient_mps_l2_norm": float(np.linalg.norm(mg.astype(np.float64))),
        "gradient_difference_l2_norm": float(np.linalg.norm(mg.astype(np.float64) - cg.astype(np.float64))),
        "gradient_elements": int(cg.size),
        "elements_failing_strict_threshold": int(np.count_nonzero(absolute > 2e-5 + 2e-3 * np.abs(cg))),
        "absolute_gradient_difference_quantiles": {str(q): float(np.quantile(absolute, q)) for q in (0.5, 0.95, 0.99, 0.999, 1.0)},
        "largest_gradient_tensors_by_absolute_difference": sorted(tensor_comparisons, key=lambda item: item["max_abs_difference"], reverse=True)[:10],
        "relu_total_elements": sum(item["elements"] for item in relu_differences),
        "relu_sign_flips": sum(item["sign_flips"] for item in relu_differences),
        "relu_modules_with_sign_flips": [item for item in relu_differences if item["sign_flips"]],
        "one_adam_update_comparison": compare_arrays(cpu_updates, mps_updates, 2e-5, 2e-3),
        "mps_backward_repeat_comparison": compare_arrays(mg, repeated, 2e-5, 2e-3),
    }
    model.zero_grad(set_to_none=True)
    return result


def run_diagnostics(args, data, metadata) -> None:
    results = {"diagnostic_kind": "cpu_mps_gradient_and_relu_boundary_audit", "seed": args.seed, "float_precision": "float32", "looser_threshold_interpretation": "Descriptive sensitivity threshold, not a preregistered certification or evidence of equivalent trained models", "experiments": []}
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal access is required for detailed diagnostics")
    for batch_examples in args.batch_examples:
        torch.manual_seed(args.seed)
        cpu = build_simulator(metadata, 6.7e-4, 6.7e-4, torch.device("cpu"), radius_backend="scipy")
        graph_fn = cpu._compute_graph_connectivity
        initial_state = {name: tensor.detach().clone() for name, tensor in cpu.state_dict().items()}
        mps = build_simulator(metadata, 6.7e-4, 6.7e-4, torch.device("mps"), radius_backend="scipy_host").to("mps")
        mps._normalization_stats = {name: {key: value.to("mps") for key, value in stats.items()} for name, stats in cpu._normalization_stats.items()}
        models = {"cpu": cpu, "mps": mps}
        for model in models.values():
            model._compute_graph_connectivity = types.MethodType(fixed_graph_callback, model)
        for index, frame in enumerate((100, 250, 400, 550)):
            host = host_batch(data, frame, batch_examples, 6.7e-4, graph_fn, args.seed + index)
            batches = {device: on_device(host, device) for device in models}
            details = detailed_parity(models, initial_state, batches)
            results["experiments"].append({"batch_examples": batch_examples, "n_particles": len(host["position"]), "frame": frame, **details})
            print(f"Detailed parity batch={batch_examples} frame={frame}: gradient_relL2={details['gradient_strict_original_threshold']['relative_l2_difference']:.6g}, ReLU_flips={details['relu_sign_flips']}, strict_failed_elements={details['elements_failing_strict_threshold']}", flush=True)
        del models, cpu, mps, batches, initial_state
        torch.mps.empty_cache()
    output = args.output.with_name("device_parity_diagnostics.json")
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    lines = ["", "## Detailed follow-up: strict gradient parity and ReLU boundaries", "", "The initial timing study is supplemented by four noisy histories for each batch size, using identical random weights. The original strict elementwise tolerance remains atol=2e-5, rtol=2e-3 for gradients. A looser atol=1e-3, rtol=5e-3 comparison is reported as a descriptive sensitivity check only; it is not retroactive certification. Forward hooks record ReLU input sign changes on the two devices. Detailed per-tensor errors, gradient norms, one-Adam-update differences, and a repeated identical MPS backward pass are saved in `device_parity_diagnostics.json`.", "", "| Examples | Particles | Frame | Gradient relative L2 error | Strict failing elements | ReLU sign flips | Looser check |", "|---:|---:|---:|---:|---:|---:|---|"]
    for item in results["experiments"]:
        lines.append(f"| {item['batch_examples']} | {item['n_particles']} | {item['frame']} | {item['gradient_strict_original_threshold']['relative_l2_difference']:.6g} | {item['elements_failing_strict_threshold']}/{item['gradient_elements']} | {item['relu_sign_flips']}/{item['relu_total_elements']} | {item['gradient_looser_descriptive_threshold_not_a_certification']['allclose']} |")
    lines += ["", "Small forward differences can change derivatives at ReLU inputs near zero. Recorded sign flips are consistent with this sensitivity, but do not prove that every gradient discrepancy is caused by it. Strict failures remain failures under the original criterion. The benchmark establishes that the native Metal kernels execute and provides empirical timing; it does not certify identical optimization paths or conference-benchmark accuracy across devices. Any long local run should first use a short fixed-seed CPU/Metal learning-curve sanity check and retain device provenance.", ""]
    with args.report.open("a") as stream:
        stream.write("\n".join(lines))
    print(f"Saved detailed diagnostics: {output}", flush=True)


def write_report(result: dict, path: Path) -> None:
    lines = [
        "# CPU versus Apple Metal feasibility for the full-size GNS",
        "",
        "This benchmark runs the repository's 128-wide GNS with 10 message-passing layers, two hidden layers per MLP, its particle embedding and variance head, and corrected Gaussian NLL. It uses fresh random initialization and real official WATERDROP histories; it is a device feasibility and timing study, not an accuracy or convergence experiment.",
        "",
        "## Scope and implementation",
        "",
        "- No core model files are modified. The SciPy neighbor backend rejects non-CPU positions, so an instance-local callback supplies exact CPU-built edges to the existing feature construction and GNN. The callback bypasses graph construction only.",
        "- `precomputed_graph` timings use precomputed edge indices and input tensors already on each device. `host_graph_and_transfer` timings include fresh host noise, CPU SciPy graph construction, input/edge transfer, feature computation, forward, NLL, backward, and Adam update. Dataset file reading, a production DataLoader, validation rollouts, and checkpoint saving are excluded.",
        "- All times are synchronized wall-clock durations. CPU uses one PyTorch thread; MPS fallback to CPU is disabled. Device order is randomized within repeats. The host is not isolated from other work.",
        "- Training histories contain six frames. Noise follows the repository's random-walk augmentation with final velocity-noise standard deviation 6.7e-4; graph construction uses the perturbed latest positions, strict radius 0.015, up to 128 nearest neighbors per receiver, and self-loops.",
        "- Both devices receive identical weights, histories, noise, graph indices, and normalization constants for the parity check. The tests compare predicted acceleration, the positive variance head, normalized targets, NLL, and all parameter gradients in float32.",
        f"- PyTorch {result['platform']['torch']}; NumPy {result['platform']['numpy']}; SciPy {result['platform']['scipy']}; {result['platform']['system']} {result['platform']['machine']}. MPS built={result['platform']['mps_built']}, available={result['platform']['mps_available']}. CPU threads={result['platform']['torch_threads']}.",
        "",
    ]
    if result.get("error"):
        lines += ["## Blocking result", "", "```text", result["error"], "```", "", "No successful full-model Metal training throughput is established by this run.", ""]
    for experiment in result.get("experiments", []):
        lines += [f"## Batch with {experiment['batch_examples']} example(s), {experiment['n_particles']} particles", "", f"Model parameters: {experiment['parameter_count']:,}; directed edges across the four noisy histories: {experiment['edge_counts']}. Checkpoint seed: {result['protocol']['seed']}.", "", f"Output/backward parity: **{'passed the stated tolerances' if experiment['parity']['all_checks_pass'] else 'did not pass every stated tolerance'}**.", "", "| Quantity | Maximum absolute difference | Relative L2 difference | Allclose |", "|---|---:|---:|---|"]
        for quantity, values in experiment["parity"].items():
            if isinstance(values, dict):
                lines.append(f"| {quantity} | {values['max_abs_difference']:.6g} | {values['relative_l2_difference']:.6g} | {values['allclose']} |")
        lines += ["", "| Mode | Work measured | CPU median ms [Q25, Q75] | MPS median ms [Q25, Q75] | CPU/MPS median ratio |", "|---|---|---|---|---:|"]
        for mode, timings in experiment["timings"].items():
            for phase, device_values in timings.items():
                cpu, mps = device_values["cpu"], device_values["mps"]
                lines.append(f"| {mode} | {phase} | {cpu['median_ms']:.4f} [{cpu['q25_ms']:.4f}, {cpu['q75_ms']:.4f}] | {mps['median_ms']:.4f} [{mps['q25_ms']:.4f}, {mps['q75_ms']:.4f}] | {cpu['median_ms']/mps['median_ms']:.3f} |")
        lines += ["", "### Arithmetic projections, not confirmed training runs", "", "| Device | Projected hours for 500,000 updates | Projected updates in 72 hours |", "|---|---:|---:|"]
        for device, estimate in experiment["projections"].items():
            lines.append(f"| {device} | {estimate['hours_for_500k_updates']:.2f} | {estimate['updates_in_72_hours']:,.0f} |")
        lines += ["", "The projections multiply the measured median host-graph/transfer-inclusive Adam-step time by the requested update count. They assume the same batch size, particle/edge counts, precision, and sustained throughput. They exclude validation, checkpointing, full-data loading, adaptive expanded graphs, thermal changes, other jobs, and learning failures. They are planning arithmetic rather than evidence that 500,000 updates will finish on schedule.", ""]
    lines += ["## Reproduce", "", "Run the script with local Metal access: `python research/benchmark_devices.py --data ../../work/data/train-pilot.npz --metadata ../../work/data/metadata.json`. The benchmark deliberately fails rather than silently using CPU if MPS is unavailable. Raw timings, source hashes, parity tolerances, and the exact platform are in `research/results/device_benchmark.json`.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "research/results/device_benchmark.json")
    parser.add_argument("--report", type=Path, default=REPO.parent / "device_benchmark.md")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--batch-examples", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--diagnose-only", action="store_true", help="Append detailed parity diagnostics without rerunning timings")
    args = parser.parse_args()
    if args.repeats < 3 or args.warmups < 1 or any(b not in (1, 2) for b in args.batch_examples):
        parser.error("Use >=3 repeats, >=1 warmup, and batch sizes1/2")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(args.metadata.read_text())
    if args.diagnose_only:
        run_diagnostics(args, data, metadata)
        return
    result = {"experiment_type": "full_size_gns_cpu_mps_feasibility", "core_files_modified_by_benchmark": False, "fresh_training_for_accuracy": False, "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(), "mps_cpu_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cpu_isolated": False, "load_average_start": os.getloadavg()}, "protocol": {"width": 128, "message_passing_steps": 10, "hidden_mlp_layers": 2, "history_frames": 6, "recorded_frames_zero_based": [100, 250, 400, 550], "noise_std": 6.7e-4, "loss": "corrected_nll", "optimizer": "Adam", "learning_rate": 1e-4, "repeats": args.repeats, "warmups": args.warmups, "seed": args.seed, "float_precision": "float32"}, "source_sha256": {"data": sha256(args.data), "metadata": sha256(args.metadata), "benchmark_script": sha256(Path(__file__)), **{name: sha256(REPO / "adaptive-gns/gns" / name) for name in ("learned_simulator.py", "graph_network.py", "model_io.py", "losses.py", "noise_utils.py")}}, "experiments": []}
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable in this process; run with local Metal access outside a restrictive sandbox")
        for batch_examples in args.batch_examples:
            torch.manual_seed(args.seed)
            cpu = build_simulator(metadata, 6.7e-4, 6.7e-4, torch.device("cpu"), radius_backend="scipy")
            cpu_graph = cpu._compute_graph_connectivity
            initial_state = {name: tensor.detach().clone() for name, tensor in cpu.state_dict().items()}
            mps = build_simulator(metadata, 6.7e-4, 6.7e-4, torch.device("mps"), radius_backend="scipy_host").to("mps")
            # Preserve precisely the CPU-computed normalization constants.
            mps._normalization_stats = {name: {key: value.to("mps") for key, value in stats.items()} for name, stats in cpu._normalization_stats.items()}
            models = {"cpu": cpu, "mps": mps}
            host_batches = [host_batch(data, frame, batch_examples, 6.7e-4, cpu_graph, args.seed + index) for index, frame in enumerate((100, 250, 400, 550))]
            device_batches = {device: [on_device(batch, device) for batch in host_batches] for device in models}
            for model in models.values():
                model._compute_graph_connectivity = types.MethodType(fixed_graph_callback, model)
            experiment = {"batch_examples": batch_examples, "n_particles": int(host_batches[0]["position"].shape[0]), "edge_counts": [int(batch["senders"].numel()) for batch in host_batches], "parameter_count": sum(p.numel() for p in cpu.parameters()), "parity": parity(models, initial_state, {d: b[0] for d, b in device_batches.items()}), "timings": {}}
            print(f"Batch {batch_examples}: particles={experiment['n_particles']}, edges={experiment['edge_counts']}, parity={experiment['parity']['all_checks_pass']}", flush=True)
            order_rng = np.random.default_rng(args.seed)
            for mode in ("precomputed_graph", "host_graph_and_transfer"):
                experiment["timings"][mode] = {}
                phases = ("forward_no_grad", "forward_backward_adam") if mode == "precomputed_graph" else ("forward_backward_adam",)
                for phase in phases:
                    optimizers = {}
                    for device, model in models.items():
                        model.load_state_dict(initial_state)
                        model.train(phase != "forward_no_grad")
                        optimizers[device] = torch.optim.Adam(model.parameters(), lr=1e-4)
                    durations = {device: [] for device in models}
                    for iteration in range(args.warmups + args.repeats):
                        batch_index = iteration % len(host_batches)
                        frame = (100, 250, 400, 550)[batch_index]
                        for device in order_rng.permutation(["cpu", "mps"]):
                            device = str(device)
                            synchronize(device)
                            start = time.perf_counter_ns()
                            if mode == "precomputed_graph":
                                batch = device_batches[device][batch_index]
                            else:
                                batch = on_device(host_batch(data, frame, batch_examples, 6.7e-4, cpu_graph, args.seed + batch_index), device)
                            if phase == "forward_no_grad":
                                with torch.no_grad():
                                    output = forward(models[device], batch)
                            else:
                                output = training_step(models[device], optimizers[device], batch)
                            synchronize(device)
                            milliseconds = (time.perf_counter_ns() - start) / 1e6
                            if iteration >= args.warmups:
                                durations[device].append(float(milliseconds))
                            # Data inspection excluded from the timing region.
                            if phase == "forward_backward_adam" and not torch.isfinite(output).item():
                                raise RuntimeError(f"Non-finite loss on {device}")
                            del output
                    experiment["timings"][mode][phase] = {device: describe(samples) for device, samples in durations.items()}
                    print(f"{mode}/{phase}: CPU={np.median(durations['cpu']):.2f}ms MPS={np.median(durations['mps']):.2f}ms", flush=True)
            experiment["projections"] = {}
            for device in models:
                milliseconds = experiment["timings"]["host_graph_and_transfer"]["forward_backward_adam"][device]["median_ms"]
                experiment["projections"][device] = {"hours_for_500k_updates": milliseconds * 500000 / 1000 / 3600, "updates_in_72_hours": 72 * 3600 * 1000 / milliseconds, "planning_estimate_only": True}
            experiment["mps_live_allocated_bytes_at_end"] = torch.mps.current_allocated_memory()
            experiment["mps_driver_allocated_bytes_at_end"] = torch.mps.driver_allocated_memory()
            result["experiments"].append(experiment)
            # Save after every batch size so a later failure preserves evidence.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            del cpu, mps, models, optimizers, device_batches, initial_state
            torch.mps.empty_cache()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["error_traceback"] = traceback.format_exc()
        print(result["error"], flush=True)
    result["platform"]["load_average_end"] = os.getloadavg()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, args.report)
    print(f"Saved {args.output} and {args.report}", flush=True)
    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
