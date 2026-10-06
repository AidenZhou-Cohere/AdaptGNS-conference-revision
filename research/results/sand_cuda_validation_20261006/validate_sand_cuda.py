#!/usr/bin/env python3
"""Bounded actual-training-data CPU/CUDA parity and CUDA checkpoint replay.

Preparation default; no scientific model, validation/test access or selection.
The original synthetic validator and original training helpers remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_actual_data_cuda_validation_v1"
VALIDATOR_SHA = "21b2555e84a2c6221b0651c5bcc592b67a61ae5b7e3dd02da031054dd79e2a68"
OBJECTIVES = ("faithful", "nll")
CASE_NAMES = ("small", "median", "large")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_file(name, path, expected):
    if sha(path) != expected:
        raise ValueError("Validation helper identity mismatch: " + Path(path).name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe():
    return {"schema": SCHEMA, "status": "description_only", "scientific_training": False,
            "data": "admitted train only; no validation/test or scientific checkpoints",
            "cases": "rank trajectories by (particle count, archive index); consecutive pairs at ranks0, n//2, n-2; targets6 and T//2",
            "objectives": ["faithful", "nll"], "model_initialization_seed": 1729,
            "noise": "pinned host_noise(seed0, step=case index)",
            "parity": "actual ordered edges, encoder node/edge features, forward/loss/all gradients and faithful MSE control",
            "replay": "each objective: fixed first actual case, host noise for step0 then step1; separate CPU/CUDA RNG probe draws; exact serialized RNG/model/Adam restoration and numerical resumed update",
            "tolerances": "unchanged from pinned synthetic validator; no post-outcome relaxation",
            "synthetic_validator_sha256": VALIDATOR_SHA}


def case_schedule(records):
    if len(records) < 6:
        raise ValueError("At least six complete training trajectories required")
    counts = [r["positions"]["shape"][1] for r in records]
    frames = {r["positions"]["shape"][0] for r in records}
    if (any(type(n) is not int or n <= 0 for n in counts)
            or len(frames) != 1 or any(type(t) is not int for t in frames) or min(frames) < 14):
        raise ValueError("Positive particle counts and common T>=14 required for distinct eligible first/middle targets")
    t = next(iter(frames))
    ordered = sorted(range(len(records)), key=lambda i: (counts[i], i))
    rows = []
    for name, rank in (("small", 0), ("median", len(records)//2), ("large", len(records)-2)):
        ids = ordered[rank:rank+2]
        rows.append({"case": name, "trajectory_indices": ids, "target_frames": [6, t//2],
                     "trajectory_ids": [records[i]["id"] for i in ids],
                     "particle_counts": [counts[i] for i in ids],
                     "dataset_indices": [ids[0]*(t-6), ids[1]*(t-6)+t//2-6]})
    return rows


def complete_gate(report):
    """No empty or partially populated result can pass this implementation gate."""
    cases = report.get("cases", [])
    replay_rows = report.get("replay", {})
    return (len(cases) == len(CASE_NAMES) and [row.get("case") for row in cases] == list(CASE_NAMES)
            and all(set(row.get("objectives", {})) == set(OBJECTIVES)
                    and all(row["objectives"][name].get("passed") is True for name in OBJECTIVES) for row in cases)
            and set(replay_rows) == set(OBJECTIVES)
            and all(replay_rows[name].get("passed") is True for name in OBJECTIVES))


def make_model(h, metadata, objective, device, state=None):
    model = h.build_simulator(metadata, h.NOISE, h.NOISE, device,
        connectivity_radius=.015, nmessage_passing_steps=10,
        uncertainty_parameterization="variance", variance_floor=1e-6,
        detach_variance_features=objective == "faithful", radius_backend="scipy_host").to(device)
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def tensor_batch(h, dataset, schedule, index):
    raw = h.unpack_batch([dataset[i] for i in schedule["dataset_indices"]])
    noise = h.host_noise(raw[0].shape, raw[1], 0, index)
    return raw, noise, {"position_sequence": raw[0], "particle_types": raw[1],
                        "nparticles_per_example": raw[2], "next_positions": raw[3],
                        "position_sequence_noise": noise}


def replay(h, trainer, compare, metadata, objective, initial, raw, noise, device, directory, progress=None):
    torch = h.torch
    result = {"passed": False, "status": "in_progress", "phase": "initialization", "objective": objective,
              "limit": "same-device numerical replay; separate RNG probes do not alter physical inputs; no bitwise trajectory claim"}

    def publish(phase, **fields):
        result.update(phase=phase, **fields)
        if progress is not None:
            progress(result)

    config = {"schema": SCHEMA, "objective": objective, "updates": 2,
              "purpose": "bounded implementation validation; never a scientific checkpoint"}

    def advance(model, optimizer, completed):
        for group in optimizer.param_groups:
            group["lr"] = h.learning_rate(completed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Exercise both RNG streams without perturbing the admitted training inputs.
        probes = {"cpu": torch.rand(8, device="cpu"), "cuda": torch.rand(8, device=device)}
        step_noise = noise if completed == 0 else h.host_noise(raw[0].shape, raw[1], 0, completed)
        pred, head, target = h.forward_batch(model, raw, step_noise, device)
        loss = trainer.guarded_update(h, model, optimizer, pred, head, target,
                                     (raw[1] != h.KINEMATIC).to(device), objective, completed+1)
        h.synchronize(device)
        return h.cpu_tree({"prediction": pred, "variance": head, "target": target, "loss": loss}), h.cpu_tree(probes)

    try:
        publish("initialization")
        torch.manual_seed(8821)
        torch.cuda.manual_seed(7731)
        model = make_model(h, metadata, objective, device, initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False, fused=False)
        publish("first_update")
        advance(model, optimizer, 0)
        payload = trainer.checkpoint_payload(h, model, optimizer, config, 1,
                                             {"training": [], "elapsed_seconds": 0.0}, device)
        changed = not trainer.tree_equal(torch, initial, payload["state_dict"])
        publish("checkpoint_serialization", first_update_changed_parameters=changed)
        checkpoint = directory / (objective + "_validation_step1.pt")
        torch.save(payload, checkpoint)
        publish("uninterrupted_second_update", checkpoint_file=checkpoint.name, checkpoint_sha256=sha(checkpoint))
        expected_output, expected_probes = advance(model, optimizer, 1)
        expected = {"model": h.cpu_tree(model.state_dict()), "optimizer": h.cpu_tree(optimizer.state_dict())}
        expected_rng = trainer.capture_rng(torch, device)
        advanced = {name: not trainer.tree_equal(torch, payload["rng_states"][name], expected_rng[name])
                    for name in ("cpu", "cuda")}
        del optimizer, model
        publish("checkpoint_restore", rng_streams_advanced=advanced)
        restored = make_model(h, metadata, objective, device, initial)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4, foreach=False, fused=False)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        serialized_exact = trainer.tree_equal(torch, payload, loaded)
        publish("checkpoint_restore", serialized_exact=serialized_exact)
        completed, _ = trainer.restore_payload(h, loaded, restored, restored_optimizer, config, device)
        if completed != 1:
            raise ValueError("Replay must restore exactly one completed optimizer update")
        restored_rng_exact = trainer.tree_equal(torch, payload["rng_states"], trainer.capture_rng(torch, device))
        publish("resumed_second_update", restore_completed_steps=completed,
                restore_model_optimizer_exact=True, restore_rng_exact=restored_rng_exact)
        output, probes = advance(restored, restored_optimizer, completed)
        actual = {"model": h.cpu_tree(restored.state_dict()), "optimizer": h.cpu_tree(restored_optimizer.state_dict())}
        state_check = compare.tensor_map_diff(torch, compare.flatten_tensors(torch, expected), compare.flatten_tensors(torch, actual), "replay")
        output_check = compare.tensor_map_diff(torch, expected_output, output, "replay")
        probe_check = compare.tensor_map_diff(torch, expected_probes, probes, "exact")
        rng_exact = trainer.tree_equal(torch, expected_rng, trainer.capture_rng(torch, device))
        passed = (changed and all(advanced.values()) and serialized_exact and restored_rng_exact and rng_exact
                  and state_check["passed"] and output_check["passed"] and probe_check["passed"])
        publish("complete", status="complete", passed=passed, resumed_rng_exact=rng_exact,
                resumed_state=state_check, resumed_output=output_check, rng_probes_exact=probe_check)
        return result
    except BaseException as error:
        publish(result["phase"], status="validation_error", passed=False, error_type=type(error).__name__, error=str(error))
        raise


def execute(args):
    args.output_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = {**describe(), "status": "in_progress", "cases": [], "replay": {},
              "source_sha256": sha(__file__), "trainer_sha256": args.trainer_sha256}
    def save():
        temporary = args.output_dir / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output_dir / "report.json")
    save()
    try:
        trainer = import_file("_sand_trainer_validation", args.trainer, args.trainer_sha256)
        compare = import_file("_sand_compare", Path(__file__).with_name("validate_cuda_execution.py"), VALIDATOR_SHA)
        h = trainer.load_helpers(args.repo)
        device, runtime = trainer.configure_cuda(h, args)
        dataset, metadata, data_info = trainer.load_admitted_dataset(args, h)
        report.update(runtime=runtime, data=data_info, tolerances=compare.TOLERANCES,
                      train_manifest_sha256=sha(args.train_manifest), admission_sha256=sha(args.admission),
                      structural_report_sha256=sha(args.structural_report),
                      metadata_sha256=sha(args.train_manifest.parent / "metadata.json"))
        manifest = json.loads(args.train_manifest.read_text())
        schedule = case_schedule(manifest["records"])
        report["schedule"] = schedule
        save()
        for case_index, case in enumerate(schedule):
            raw, noise, batch = tensor_batch(h, dataset, case, case_index)
            input_file = args.output_dir / (case["case"] + "_batch.npz")
            h.np.savez(input_file, **{key: value.cpu().numpy() for key, value in batch.items()})
            entry = {**case, "batch_file": input_file.name, "batch_sha256": sha(input_file), "objectives": {}}
            report["cases"].append(entry)
            save()
            for objective in OBJECTIVES:
                h.torch.manual_seed(1729)
                cpu = make_model(h, metadata, objective, "cpu")
                initial = h.cpu_tree(cpu.state_dict())
                gpu = make_model(h, metadata, objective, device, initial)
                features = []
                for model, backend in ((cpu, "cpu"), (gpu, device)):
                    moved = compare.move_batch(batch, backend)
                    preprocessed = model._encoder_preprocessor(moved["position_sequence"] + moved["position_sequence_noise"],
                        moved["nparticles_per_example"], moved["particle_types"])
                    features.append(h.cpu_tree({"node_features": preprocessed[0], "edge_features": preprocessed[2]}))
                feature_check = compare.tensor_map_diff(h.torch, compare.flatten_tensors(h.torch, features[0]),
                    compare.flatten_tensors(h.torch, features[1]), "forward")
                parity = compare.objective_checks(h.torch, h, cpu, gpu, batch, objective, str(device))
                entry["objectives"][objective] = {"passed": feature_check["passed"] and parity["passed"],
                                                   "encoder_features": feature_check, "parity": parity}
                del gpu, cpu
                save()
                if case_index == 0:
                    def replay_progress(row):
                        report["replay"][objective] = row.copy()
                        save()
                    replay(h, trainer, compare, metadata, objective, initial, raw, noise, device, args.output_dir,
                           progress=replay_progress)
                    save()
        all_pass = complete_gate(report)
        report.update(status="validation_passed" if all_pass else "validation_failed",
                      elapsed_seconds=time.perf_counter()-started, scientific_training=False)
        if (sha(args.trainer) != args.trainer_sha256 or sha(args.train_manifest) != report["train_manifest_sha256"]
                or sha(args.admission) != report["admission_sha256"]
                or sha(args.structural_report) != report["structural_report_sha256"]
                or sha(args.train_manifest.parent / "metadata.json") != report["metadata_sha256"]):
            raise ValueError("Validation inputs changed during execution")
        save()
        return 0 if all_pass else 2
    except BaseException as error:
        report.update(status="validation_error", error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter()-started, failed_attempt_preserved=True)
        save()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    for name in ("repo", "trainer", "train-manifest", "admission", "structural-report", "output-dir"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--trainer-sha256")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.execute and any(getattr(args, name) is None for name in ("repo", "trainer", "trainer_sha256", "train_manifest", "admission", "structural_report", "output_dir")):
        parser.error("Execution requires every source/data/output argument")
    if args.cuda_index < 0 or not 1 <= args.threads <= 16:
        parser.error("CUDA index must be nonnegative; threads must be 1..16")
    if args.execute and re.fullmatch(r"[0-9a-f]{64}", args.trainer_sha256) is None:
        parser.error("Expected lowercase trainer SHA256")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.execute:
        raise SystemExit(execute(args))
    print(json.dumps(describe(), indent=2))
