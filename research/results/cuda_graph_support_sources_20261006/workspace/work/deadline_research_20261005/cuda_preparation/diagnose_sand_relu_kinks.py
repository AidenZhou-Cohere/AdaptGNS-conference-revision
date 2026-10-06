#!/usr/bin/env python3
"""Bounded ReLU-branch/first-Adam diagnostic; never changes an admission result.

Uses the exact three saved validation batches, original seed 1729 and both
objectives. No source trajectory loading, tuning, scientific training or
threshold relaxation. Default does not import Torch or repository modules.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_relu_kink_diagnostic_v1"
TRAINER_SHA = "9dd376f6d4b6881290b134c8f410b89933bcdb026c9b52de601ba0ec7bb05a9b"
COMPARE_SHA = "21b2555e84a2c6221b0651c5bcc592b67a61ae5b7e3dd02da031054dd79e2a68"
ACTUAL_VALIDATOR_SHA = "60083ba485bf0e99e9fef34b9c76d81a0ad109d517974d5ef25657203fb1081c"
METADATA_SHA = "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0"
MAX_RELU_ELEMENTS = 300_000_000
MAX_DISAGREEMENTS = 1_000_000
BRANCHES = ("cpu_original", "cuda_original", "cuda_native_mask_sham", "cuda_cpu_mask_control")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_file(name, path, expected):
    require(sha(path) == expected, "Pinned diagnostic helper differs: " + Path(path).name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_mask_function(torch):
    class FrozenMaskReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, mask):
            require(mask.dtype == torch.bool and mask.shape == value.shape and mask.device == value.device,
                    "A matching boolean backward mask is required")
            ctx.save_for_backward(mask)
            return torch.relu(value)

        @staticmethod
        def backward(ctx, gradient):
            (mask,) = ctx.saved_tensors
            return torch.where(mask, gradient, torch.zeros_like(gradient)), None
    return FrozenMaskReLU


@contextmanager
def capture_relu_inputs(torch, model, max_elements=MAX_RELU_ELEMENTS):
    events, counts, handles = [], {}, []
    total = 0

    def hook(name):
        def record(module, arguments):
            nonlocal total
            require(len(arguments) == 1 and isinstance(arguments[0], torch.Tensor), "Expected one ReLU input tensor")
            value = arguments[0]
            total += value.numel()
            require(total <= max_elements, "Predeclared ReLU trace element budget exceeded")
            cpu = value.detach().cpu().clone()
            require(bool(torch.isfinite(cpu).all()), "Nonfinite ReLU preactivation")
            occurrence = counts.get(name, 0)
            counts[name] = occurrence + 1
            events.append({"name": name, "call": occurrence, "value": cpu})
        return record

    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                require(not module.inplace, "In-place ReLU is outside the diagnostic contract")
                handles.append(module.register_forward_pre_hook(hook(name)))
        require(bool(handles), "No module ReLUs found; diagnostic cannot cover this model")
        yield events
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def force_relu_backward(torch, model, events):
    """Temporary backward-only intervention; every forward remains torch.relu(x)."""
    function = make_mask_function(torch)
    expected = {(row["name"], row["call"]): row["value"] > 0 for row in events}
    require(len(expected) == len(events) and bool(expected), "Unique recorded ReLU calls required")
    counts, seen, originals = {}, set(), []
    succeeded = False

    def replacement(name):
        def forward(value):
            occurrence = counts.get(name, 0)
            counts[name] = occurrence + 1
            key = (name, occurrence)
            require(key in expected and expected[key].shape == value.shape, "ReLU call/shape differs from recorded masks")
            seen.add(key)
            return function.apply(value, expected[key].to(device=value.device))
        return forward

    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                require(not module.inplace, "In-place ReLU unsupported")
                originals.append((module, module.forward))
                module.forward = replacement(name)
        yield
        succeeded = True
    finally:
        for module, original in originals:
            module.forward = original
        if succeeded:
            require(seen == set(expected), "Not every recorded ReLU call was exercised")


def relu_disagreements(torch, cpu, gpu):
    require([(r["name"], r["call"]) for r in cpu] == [(r["name"], r["call"]) for r in gpu], "CPU/GPU ReLU call order differs")
    rows, arrays, total = [], {}, 0
    for index, (left, right) in enumerate(zip(cpu, gpu)):
        a, b = left["value"], right["value"]
        require(a.shape == b.shape and a.dtype == b.dtype, "CPU/GPU ReLU preactivation shape/dtype differs")
        flat_a, flat_b = a.reshape(-1), b.reshape(-1)
        chosen = torch.nonzero((flat_a > 0) != (flat_b > 0), as_tuple=False).reshape(-1)
        total += chosen.numel()
        require(total <= MAX_DISAGREEMENTS, "Predeclared sign-disagreement output budget exceeded")
        selected_a, selected_b = flat_a[chosen], flat_b[chosen]
        prefix = f"relu_{index:03d}"
        arrays[prefix + "_flat_indices"] = chosen.numpy()
        arrays[prefix + "_cpu_values"] = selected_a.numpy()
        arrays[prefix + "_cuda_values"] = selected_b.numpy()
        rows.append({"name": left["name"], "call": left["call"], "shape": list(a.shape), "elements": a.numel(),
            "disagreements": chosen.numel(), "cpu_positive": int((a > 0).sum()), "cuda_positive": int((b > 0).sum()),
            "cpu_exact_zeros": int((a == 0).sum()), "cuda_exact_zeros": int((b == 0).sum()),
            "maximum_absolute_preactivation_difference": float((a - b).abs().max()),
            "minimum_absolute_cpu_preactivation": float(a.abs().min()), "minimum_absolute_cuda_preactivation": float(b.abs().min()),
            "disagreement_max_abs_cpu": float(selected_a.abs().max()) if chosen.numel() else None,
            "disagreement_max_abs_cuda": float(selected_b.abs().max()) if chosen.numel() else None,
            "arrays_prefix": prefix})
    return {"module_calls": len(rows), "total_elements": sum(row["elements"] for row in rows),
            "sign_disagreements": total, "calls_with_disagreements": sum(row["disagreements"] > 0 for row in rows), "layers": rows}, arrays


def signed_comparison(torch, compare, reference, candidate, tolerance):
    result = compare.tensor_map_diff(torch, reference, candidate, tolerance)
    for name, record in result["tensors"].items():
        a, b = reference.get(name), candidate.get(name)
        if a is None or b is None or a.shape != b.shape:
            continue
        # Match compare.diff's float64 tolerance arithmetic for exact failing indices.
        a, b = a.detach().cpu().reshape(-1).double(), b.detach().cpu().reshape(-1).double()
        sign_changes = torch.sign(a) != torch.sign(b)
        opposite = ((a < 0) & (b > 0)) | ((a > 0) & (b < 0))
        outside = (a - b).abs() > compare.TOLERANCES[tolerance]["atol"] + compare.TOLERANCES[tolerance]["rtol"] * a.abs()
        record.update(sign_disagreements=int(sign_changes.sum()), strict_opposite_signs=int(opposite.sum()),
                      reference_zeros=int((a == 0).sum()), candidate_zeros=int((b == 0).sum()),
                      outside_tolerance_flat_indices=torch.nonzero(outside, as_tuple=False).reshape(-1).tolist())
    return result


def run_branch(h, trainer, compare, actual, metadata, objective, initial, batch, device, forced_events=None):
    torch = h.torch
    model = actual.make_model(h, metadata, objective, device, initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0.0, foreach=False, fused=False)
    moved = compare.move_batch(batch, device)
    if forced_events is None:
        with capture_relu_inputs(torch, model) as events:
            before = compare.cpu_tree(torch, compare.backward(torch, h, model, moved, objective))
    else:
        events = None
        with force_relu_backward(torch, model, forced_events):
            before = compare.cpu_tree(torch, compare.backward(torch, h, model, moved, objective))
    gradients = compare.grads(torch, model)
    require(h.tensors_are_finite(before.values()), "Nonfinite pre-update output")
    require(all(value is not None for value in gradients.values()), "Missing diagnostic gradient")
    h.assert_finite_gradients(model)
    optimizer.step()
    trainer.assert_adam(torch, model, optimizer, 1)
    require(h.tensors_are_finite(model.parameters()), "Nonfinite first-Adam model state")
    h.synchronize(device)
    after_state = h.cpu_tree(model.state_dict())
    update = {name: after_state[name] - initial[name] for name in after_state}
    with torch.no_grad():
        pred, variance, target = model(**moved, material_property=None, augment_radius_prob=0.0)
        loss = h.acceleration_loss(pred, target, moved["particle_types"] != 3,
                                   pred_variance=variance, loss_type=objective, variance_floor=1e-6)
        after = h.cpu_tree({"prediction": pred, "variance": variance, "target": target, "loss": loss})
    require(h.tensors_are_finite(after.values()), "Nonfinite post-update output")
    return {"before": before, "gradients": gradients, "after_state": after_state, "parameter_update": update,
            "after_output": after, "optimizer": h.cpu_tree(optimizer.state_dict())}, events


def check_saved_batch(h, report_case, directory):
    name = report_case["batch_file"]
    require(name == report_case["case"] + "_batch.npz", "Unexpected saved-batch filename")
    path = Path(directory) / name
    require(path.resolve().parent == Path(directory).resolve() and sha(path) == report_case["batch_sha256"], "Saved batch identity differs")
    keys = {"position_sequence", "particle_types", "nparticles_per_example", "next_positions", "position_sequence_noise"}
    with h.np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == keys, "Saved batch fields differ")
        values = {name: archive[name].copy() for name in keys}
    counts = values["nparticles_per_example"]
    require(counts.dtype == h.np.dtype("int64") and counts.tolist() == report_case["particle_counts"], "Saved particle counts differ")
    n = int(counts.sum())
    for name, shape in (("position_sequence", (n, 6, 2)), ("position_sequence_noise", (n, 6, 2)), ("next_positions", (n, 2))):
        require(values[name].shape == shape and values[name].dtype == h.np.dtype("float32") and h.np.isfinite(values[name]).all(), "Saved float batch shape/dtype/finiteness differs")
    require(values["particle_types"].shape == (n,) and values["particle_types"].dtype == h.np.dtype("int64")
            and h.np.all(values["particle_types"] == 6), "Saved Sand particle types differ")
    return {name: h.torch.from_numpy(value) for name, value in values.items()}, path


def completion_gate(report):
    expected = {(case, objective) for case in ("small", "median", "large") for objective in ("faithful", "nll")}
    cases = report.get("cases", [])
    return (len(cases) == len(expected) and {(row.get("case"), row.get("objective")) for row in cases} == expected
            and all(row.get("status") == "complete" and row.get("phase") == "complete"
                    and set(row.get("phases", {})) == set(BRANCHES)
                    and all(row["phases"][branch].get("status") == "complete" for branch in BRANCHES)
                    and set(row.get("comparisons", {})) == set(BRANCHES[1:]) and "relu_localization" in row for row in cases))


def execute(args):
    require(sha(args.validation_report) == args.validation_report_sha256, "Original v2 report SHA differs")
    previous = json.loads(args.validation_report.read_text())
    require(previous.get("status") == "validation_failed" and previous.get("source_sha256") == ACTUAL_VALIDATOR_SHA
            and previous.get("trainer_sha256") == TRAINER_SHA and previous.get("runtime", {}).get("deterministic_algorithms") is True,
            "Exact prior deterministic failed report is required; its status is not revised")
    require([case["case"] for case in previous.get("cases", [])] == ["small", "median", "large"], "All three original cases required")
    require(sha(args.metadata) == METADATA_SHA, "Original Sand metadata differs")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    started = time.perf_counter()
    report = {"schema": SCHEMA, "status": "in_progress", "prior_validation_status": "validation_failed",
              "scientific_training_admitted": False, "source_sha256": sha(__file__), "cases": [],
              "validation_report_sha256": args.validation_report_sha256, "trainer_sha256": TRAINER_SHA,
              "branches": BRANCHES, "initialization_seed": 1729, "max_relu_elements": MAX_RELU_ELEMENTS,
              "max_disagreement_elements": MAX_DISAGREEMENTS,
              "interpretation": "Backward-mask interventions are diagnostic counterfactuals, never a deployable/training model"}

    def save():
        path = args.output_dir / "report.json"
        path.with_suffix(".json.tmp").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        path.with_suffix(".json.tmp").replace(path)

    save()
    try:
        root = Path(__file__).parent
        trainer = import_file("_kink_deterministic_trainer", args.trainer, TRAINER_SHA)
        compare = import_file("_kink_original_compare", root / "validate_cuda_execution.py", COMPARE_SHA)
        actual = import_file("_kink_original_actual_validator", root / "validate_sand_cuda.py", ACTUAL_VALIDATOR_SHA)
        h = trainer.load_helpers(args.repo)
        device, runtime = trainer.configure_cuda(h, args)
        require(str(device) == previous["runtime"]["device"] and runtime["uuid"] == previous["runtime"]["uuid"], "Use the same selected GPU as v2")
        metadata = json.loads(args.metadata.read_text())
        report.update(runtime=runtime, tolerances=compare.TOLERANCES)
        inputs = {str(args.validation_report): args.validation_report_sha256, str(args.metadata): METADATA_SHA,
                  str(args.trainer): TRAINER_SHA, str(root / "validate_cuda_execution.py"): COMPARE_SHA,
                  str(root / "validate_sand_cuda.py"): ACTUAL_VALIDATOR_SHA, str(Path(__file__)): report["source_sha256"]}
        inputs.update({str(args.repo / path): expected for path, expected in trainer.SOURCE_PINS.items()})
        for original_case in previous["cases"]:
            batch, batch_path = check_saved_batch(h, original_case, args.batch_dir)
            inputs[str(batch_path)] = original_case["batch_sha256"]
            for objective in ("faithful", "nll"):
                entry = {"case": original_case["case"], "objective": objective, "status": "in_progress",
                         "batch_sha256": original_case["batch_sha256"], "phases": {}}
                report["cases"].append(entry)
                save()
                h.torch.manual_seed(1729)
                seed_model = actual.make_model(h, metadata, objective, "cpu")
                initial = h.cpu_tree(seed_model.state_dict())
                del seed_model
                results, events = {}, {}
                for branch, backend, forced in ((BRANCHES[0], "cpu", None), (BRANCHES[1], device, None),
                                                (BRANCHES[2], device, "cuda_original"), (BRANCHES[3], device, "cpu_original")):
                    entry["phase"] = branch
                    save()
                    result, captured = run_branch(h, trainer, compare, actual, metadata, objective, initial, batch, backend,
                                                  events[forced] if forced else None)
                    results[branch] = result
                    if captured is not None:
                        events[branch] = captured
                    flat = compare.flatten_tensors(h.torch, result)
                    array_index = {f"tensor_{index:04d}": name for index, name in enumerate(flat)}
                    arrays = {key: flat[name].numpy() for key, name in array_index.items()}
                    path = args.output_dir / f"{entry['case']}_{objective}_{branch}.npz"
                    h.np.savez_compressed(path, **arrays)
                    entry["phases"][branch] = {"status": "complete", "artifact_file": path.name, "artifact_sha256": sha(path),
                                               "saved_tensor_count": len(arrays), "array_paths": array_index}
                    save()
                localization, arrays = relu_disagreements(h.torch, events["cpu_original"], events["cuda_original"])
                path = args.output_dir / f"{entry['case']}_{objective}_relu_disagreements.npz"
                h.np.savez_compressed(path, **arrays)
                entry["relu_localization"] = {**localization, "artifact_file": path.name, "artifact_sha256": sha(path)}
                reference, gpu = results["cpu_original"], results["cuda_original"]
                entry["comparisons"] = {}
                for branch in BRANCHES[1:]:
                    result = results[branch]
                    entry["comparisons"][branch] = {
                        "gradients_vs_cpu_original": signed_comparison(h.torch, compare, reference["gradients"], result["gradients"], "gradient"),
                        "gradients_vs_cuda_original": signed_comparison(h.torch, compare, gpu["gradients"], result["gradients"], "gradient"),
                        "preupdate_outputs_vs_cuda_original_exact": compare.tensor_map_diff(h.torch, gpu["before"], result["before"], "exact"),
                        "preupdate_outputs_vs_cpu_original": {name: compare.diff(h.torch, reference["before"][name], result["before"][name], "loss" if name == "loss" else "forward") for name in reference["before"]},
                        "first_adam_state_vs_cpu": compare.tensor_map_diff(h.torch, reference["after_state"], result["after_state"], "replay"),
                        "first_adam_update_vs_cpu": signed_comparison(h.torch, compare, reference["parameter_update"], result["parameter_update"], "replay"),
                        "first_adam_update_vs_cuda_original": signed_comparison(h.torch, compare, gpu["parameter_update"], result["parameter_update"], "replay"),
                        "first_adam_optimizer_vs_cpu": compare.tensor_map_diff(h.torch, compare.flatten_tensors(h.torch, reference["optimizer"]), compare.flatten_tensors(h.torch, result["optimizer"]), "replay"),
                        "postupdate_outputs_vs_cpu": {name: compare.diff(h.torch, reference["after_output"][name], result["after_output"][name], "loss" if name == "loss" else "forward") for name in reference["after_output"]},
                        "postupdate_outputs_vs_cuda_original": {name: compare.diff(h.torch, gpu["after_output"][name], result["after_output"][name], "loss" if name == "loss" else "forward") for name in gpu["after_output"]}}
                entry.update(status="complete", phase="complete")
                save()
                del results, events, initial
        for path, expected in inputs.items():
            require(sha(path) == expected, "Pinned diagnostic input changed")
        require(completion_gate(report), "Diagnostic case/branch completeness differs from the fixed plan")
        report.update(status="diagnostic_complete", elapsed_seconds=time.perf_counter() - started,
                      all_inputs_reverified=True, scientific_training_admitted=False)
        save()
        return 0
    except BaseException as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started)
        save()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    for name in ("repo", "trainer", "validation-report", "batch-dir", "metadata", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--validation-report-sha256")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    require(args.cuda_index >= 0 and 1 <= args.threads <= 16, "Invalid runtime arguments")
    if args.execute:
        require(all(getattr(args, name) is not None for name in ("repo", "trainer", "validation_report", "validation_report_sha256", "batch_dir", "metadata", "output_dir")), "Every execution argument is required")
        require(len(args.validation_report_sha256) == 64 and all(c in "0123456789abcdef" for c in args.validation_report_sha256), "Exact report SHA256 required")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.execute:
        raise SystemExit(execute(args))
    print(json.dumps({"schema": SCHEMA, "status": "description_only", "branches": BRANCHES,
                      "same_saved_batches": ["small", "median", "large"], "objectives": ["faithful", "nll"],
                      "initial_seed": 1729, "training_admitted": False, "tolerances": "original unchanged",
                      "purpose": "localize ReLU sign crossings; backward-only CPU-mask control with GPU-mask sham; first Adam and post-update outputs"}, indent=2))
