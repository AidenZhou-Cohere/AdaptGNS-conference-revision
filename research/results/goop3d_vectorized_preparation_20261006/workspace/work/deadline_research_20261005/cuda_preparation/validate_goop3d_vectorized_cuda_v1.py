#!/usr/bin/env python3
"""Bounded same-CUDA original/vectorized3D check; no scientific endpoint or test.

Save count-ranked train batches before CUDA. For each batch and base/mix arm,
compare two fresh original/candidate updates; replay candidate step2 from its
small-case step1 checkpoint. Root owns device inventory and outer timeout.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_goop3d_vectorized_cuda_validation_v1"
RELEASE_SCHEMA = "adaptgns_goop3d_vectorized_cuda_validation_release_v1"
TRAINER_SHA = "8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc"
ORIGINAL_SHA = "7d43fe7d06ac450b6b9031181902cfc6d3d9754af3a26e17e96fd46c4a1bf64f"
CANDIDATE_SHA = "ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50"
DATA_PINS = {
    "train_manifest": "0f0ce1802e202b9faaf86f0a6ce059c286ed454ceb53273cb926980614b0f864",
    "structural_report": "282a7ffdf55c2f09a92719bff5e02ac5df17be35bd45f0ff59a0fb51b5c646bf",
    "acquisition_report": "e1d651d3987e479f6927c3ab202b14ed6b84ccf4d8b5a5348960143e5bc5ba8d",
    "context_semantics": "5eb6818ae2699c57e62e80f63724248e8573e4536195eb471df0c647122fb1a5",
    "auxiliary_report": "9117a9efa6516ce037a53d277972229e89b52a62849c3921f98536e436e2f174",
    "data_review": "040630e9464e6ade3aab57ef33a28e8ba02e721ba04a5b8630d3d2e834545334",
    "cpu_proof": "01e2262037342bd866565e1bd55b46d4c0e64160542df413c1c40e11cb2d113c",
}
CASES, ARMS = ("small", "median", "large"), ("base", "mix")
SEED, PROBE_LR_HORIZON = 0, 100000


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_tensor(torch, path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": Path(path).name, "sha256": sha(path)}


def private_import(path, digest, name):
    require(sha(path) == digest, "Pinned implementation differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules(repo):
    directory = Path(__file__).resolve().parent
    trainer = private_import(directory / "train_goop3d_graph_support_cuda_v2.py", TRAINER_SHA, "_goop3d_vectorized_check_trainer")
    h, candidate = trainer.load_helpers(repo)
    require(trainer.GRAPH_SHA == CANDIDATE_SHA, "Candidate graph pin differs")
    original = private_import(directory / "goop3d_graph_support.py", ORIGINAL_SHA, "_goop3d_vectorized_check_original")
    return trainer, h, original, candidate


def expected_schedule(manifest):
    records = manifest.get("records", [])
    require(manifest.get("dataset") == "Goop-3D" and manifest.get("split") == "train"
        and manifest.get("record_count") == len(records) == 1000
        and all(r["positions"]["shape"][0] == 301 and r["positions"]["shape"][2] == 3 for r in records),
        "Complete pinned T301/D3 train source required")
    order = sorted(range(len(records)), key=lambda i: (records[i]["positions"]["shape"][1], i))
    schedule = []
    for name, rank in (("small", 0), ("median", 500), ("large", 998)):
        ids = order[rank:rank + 2]
        schedule.append({"case": name, "trajectory_indices": ids, "target_frames": [6, 150],
            "trajectory_ids": [records[i]["id"] for i in ids],
            "particle_counts": [records[i]["positions"]["shape"][1] for i in ids],
            "dataset_indices": [ids[0] * 295, ids[1] * 295 + 144]})
    return schedule


def verify_data_evidence(args, h, trainer, release):
    require(all(sha(getattr(args, key)) == digest for key, digest in DATA_PINS.items()), "Pinned3D data/evidence differs")
    contract = release.get("data_contract", {})
    require(contract.get("scope") == "bounded_implementation_data_only"
        and contract.get("scientific_training_admitted") is False
        and not any(key in contract for key in ("prospective_endpoint_updates", "endpoint_selection_basis", "selection_evidence_sha256")),
        "Source-only root contract required; no scientific endpoint admission")
    manifest = read(args.train_manifest)
    trainer.validate_manifest_contract(manifest, contract)
    rows = trainer.verify_goop3d_evidence(args, manifest, contract)
    trainer.verify_goop3d_auxiliary_arrays(args, h, manifest, rows)
    metadata_path = args.train_manifest.resolve().parent / "metadata.json"
    require(sha(metadata_path) == trainer.METADATA_SHA and read(metadata_path) == manifest["metadata"], "Exact metadata bytes required")
    review, proof = read(args.data_review), read(args.cpu_proof)
    require(review.get("status") == "passed_data_report_consistency_and_source_semantics"
        and review.get("scientific_training_admitted") is False and review.get("endpoint_selected") is False
        and proof.get("status") == "complete_same_state_cpu_graph_equivalence_and_timing_only"
        and proof.get("candidate_sha256") == CANDIDATE_SHA and proof.get("scientific_training_admitted") is False,
        "Complete source-only review and CPU proof required")
    return manifest, contract, manifest["metadata"]


def load_data(args, h, trainer, release):
    manifest, contract, metadata = verify_data_evidence(args, h, trainer, release)
    dataset, _, info = h.load_manifest_dataset(args.train_manifest, "train")
    observed = set()
    for record, (positions, types) in zip(manifest["records"], dataset._data):
        require(positions.shape == tuple(record["positions"]["shape"]) and positions.dtype.str == "<f4"
            and types.shape == (positions.shape[1],) and types.dtype.str == "<i8" and h.np.isfinite(positions).all(),
            "Actual complete training array shape/dtype/finiteness differs")
        observed.update(int(v) for v in h.np.unique(types))
        for field in ("positions", "particle_types"):
            path = h.data_loader._manifest_array_path(args.train_manifest.resolve().parent, record[field])
            require(path.stat().st_size == record[field]["size_bytes"], "Actual numeric byte length differs")
    require(len(dataset._data) == contract["record_count"] and len(dataset) == contract["eligible_six_frame_histories"]
        and sorted(observed) == contract["particle_type_ids"], "Actual complete source count/types/histories differ")
    info.update(data_scope="bounded_implementation_only", scientific_training_admitted=False, source_evidence_sha256=DATA_PINS)
    return dataset, metadata, info


def make_model(h, metadata, device, initial=None):
    model = h.build_simulator(metadata, h.NOISE, h.NOISE, device, connectivity_radius=.025,
        nmessage_passing_steps=10, uncertainty_parameterization="variance", variance_floor=1e-6,
        detach_variance_features=True, radius_backend="scipy_host").to(device).train()
    require(model._checkpoint_config["particle_dimensions"] == 3 and model._checkpoint_config["nnode_in"] == 37
        and model._checkpoint_config["nedge_in"] == 4, "D3 full architecture required")
    if initial is not None:
        model.load_state_dict(initial, strict=True)
    optimizer = h.torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
        weight_decay=0., foreach=False, fused=False)
    return model, optimizer


def graph_gate(h, trainer, original, candidate, model, batch, noise, device, graph_step, arm):
    before = trainer.capture_rng(h.torch, device)
    noisy = batch[0].to(device) + noise.to(device)
    offset = 0
    for count in batch[2].tolist():
        original.check_candidate_bound(noisy[offset:offset + count, -1].detach().cpu().numpy(), .025)
        offset += count
    with h.torch.no_grad():
        _, native, _ = model._encoder_preprocessor(noisy, batch[2].to(device), batch[1].to(device), None,
            augment_radius_prob=0., augment_radius_factor=1.267)
        a, ledger_a = original.append_optional_edges(noisy, batch[2], native, .025, SEED, graph_step, arm)
        b, ledger_b = candidate.append_optional_edges(noisy, batch[2], native, .025, SEED, graph_step, arm)
    require(trainer.tree_equal(h.torch, a, b) and ledger_a == ledger_b, "Original/candidate graph order or ledger differs")
    require(trainer.tree_equal(h.torch, before, trainer.capture_rng(h.torch, device)), "Graph gate changed RNG")
    return {"passed": True, "edges_exact": True, "ledger_exact": True, "rng_unchanged": True,
        "ordered_edge_sha256": candidate.state_hash(a.cpu().numpy()), "graph_ledger": ledger_a}


def advance(h, trainer, graph, model, optimizer, batch, noise, device, step, graph_step, arm):
    trainer.assert_adam(h.torch, model, optimizer, step, PROBE_LR_HORIZON)
    optimizer.param_groups[0]["lr"] = trainer.learning_rate(step, PROBE_LR_HORIZON)
    optimizer.zero_grad(set_to_none=True)
    captured = []
    handle = model._encode_process_decode.register_forward_pre_hook(
        lambda module, values: captured.append(h.cpu_tree(dict(zip(("nodes", "edges", "features"), values)))))
    try:
        prediction, head, target, ledger = graph.forward_batch(model, batch, noise, device, SEED, graph_step, arm)
    finally:
        handle.remove()
    require(len(captured) == 1, "Exactly one graph-network invocation required")
    loss = trainer.guarded_update(h, model, optimizer, prediction, head, target, (batch[1] != 3).to(device),
        "faithful", step + 1, PROBE_LR_HORIZON)
    h.synchronize(device)
    result = {"before": h.cpu_tree({"prediction": prediction, "head": head, "target": target, "loss": loss}),
        "graph": captured[0], "ledger": ledger,
        "gradients": {name: h.cpu_tree(p.grad) for name, p in model.named_parameters()},
        "state_dict": h.cpu_tree(model.state_dict()), "optimizer_state": h.cpu_tree(optimizer.state_dict()),
        "rng_states": trainer.capture_rng(h.torch, device)}
    return result


def exact_comparison(h, trainer, a, b):
    checks = {key: trainer.tree_equal(h.torch, a[key], b[key]) for key in a}
    return {"passed": a.keys() == b.keys() and all(checks.values()), "bytewise_component_checks": checks}


def history_row(case, graph, noise, step, ledger):
    return {"completed_steps": step + 1, "absolute_schedule_step": step,
        "frame_ids": [f"{name}:{target}" for name, target in zip(case["trajectory_ids"], case["target_frames"])],
        "noise_sha256": graph.state_hash(noise.numpy()), "examples": ledger}


def validate_release(args, release, schedule):
    require(release.get("schema") == RELEASE_SCHEMA and release.get("status") == "admitted_for_bounded_validation"
        and release.get("issued_by") == "root" and release.get("scientific_training_admitted") is False
        and release.get("source_sha256") == sha(__file__) and release.get("trainer_sha256") == TRAINER_SHA
        and release.get("original_graph_sha256") == ORIGINAL_SHA and release.get("candidate_graph_sha256") == CANDIDATE_SHA
        and release.get("data_files_sha256") == DATA_PINS and release.get("schedule") == schedule
        and release.get("cuda_index") == args.cuda_index and release.get("gpu_uuid") == args.gpu_uuid
        and release.get("max_optimizer_updates") == 26, "Exact root bounded numerical release required")
    checked = datetime.fromisoformat(release["process_identity_checked_utc"])
    require(checked.tzinfo is not None and -60 <= (datetime.now(timezone.utc) - checked).total_seconds() <= 300,
        "Fresh root device/process identity check required")


def description():
    return {"schema": SCHEMA, "status": "description_only", "scientific_training_admitted": False,
        "test_accessed": False, "schedule": "count/source-index-ranked pairs0/1,500/501,998/999; targets6/150",
        "arms": list(ARMS), "seed": SEED, "steps_per_branch": 2, "max_optimizer_updates": 26,
        "replay": "candidate small-case step2 from step1 checkpoint, base and mix",
        "probe_lr_horizon": PROBE_LR_HORIZON, "scientific_endpoint_selected": False,
        "comparison": "bytewise exact same-CUDA graph/network inputs, outputs, gradients, Adam, RNG; no CPU/CUDA claim"}


def execute(args):
    release = read(args.release)
    require(sha(args.train_manifest) == DATA_PINS["train_manifest"], "Pinned training manifest required")
    schedule = expected_schedule(read(args.train_manifest))
    validate_release(args, release, schedule)
    target = args.output_dir.resolve()
    for protected in (args.train_manifest.resolve().parent, (args.repo / "adaptive-gns").resolve()):
        require(target != protected and protected not in target.parents and target not in protected.parents,
            "Validation output must be separate from source/data trees")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {**description(), "status": "in_progress", "source_sha256": sha(__file__), "release_sha256": sha(args.release),
        "trainer_sha256": TRAINER_SHA, "original_graph_sha256": ORIGINAL_SHA, "candidate_graph_sha256": CANDIDATE_SHA,
        "schedule": schedule, "cases": [], "replay": {}, "attempted_update": None, "completed_optimizer_calls": 0}
    started = time.perf_counter()
    save = lambda: write(args.output_dir / "report.json", report)
    save()
    trainer = h = model = optimizer = None
    try:
        trainer, h, original, candidate = load_modules(args.repo)
        dataset, metadata, info = load_data(args, h, trainer, release)
        report["training_data"] = info
        batches = []
        for index, case in enumerate(schedule):
            batch = trainer.unpack_batch(h, [dataset[i] for i in case["dataset_indices"]])
            path = args.output_dir / (case["case"] + "_prospective_batch.npz")
            with path.open("xb") as stream:
                h.np.savez(stream, position_sequence=batch[0].numpy(), particle_types=batch[1].numpy(),
                    nparticles_per_example=batch[2].numpy(), next_positions=batch[3].numpy())
            batches.append(batch)
            report.setdefault("saved_batches", []).append({**case, "file": path.name, "sha256": sha(path),
                "noise_steps": [2 * index, 2 * index + 1]})
        report["prospective_batches_saved_before_cuda"] = True
        save()
        device, runtime = trainer.configure_cuda(h, args)
        require(str(runtime["uuid"]).lower().removeprefix("gpu-") == args.gpu_uuid.lower().removeprefix("gpu-"),
            "Root-selected physical GPU differs")
        report["runtime"] = runtime
        h.torch.manual_seed(SEED)
        h.torch.cuda.manual_seed(SEED)
        model, optimizer = make_model(h, metadata, device)
        initial = h.cpu_tree(model.state_dict())
        initial_rng = trainer.capture_rng(h.torch, device)
        report["initial_state"] = save_tensor(h.torch, args.output_dir / "initial_seed0.pt", initial)
        del model, optimizer
        model = optimizer = None
        for index, (case, batch) in enumerate(zip(schedule, batches)):
            for arm in ARMS:
                row = {"case": case["case"], "arm": arm, "steps": [], "graph_gates": []}
                report["cases"].append(row)
                stored = {}
                for label, graph in (("original", original), ("candidate", candidate)):
                    gc.collect()
                    h.torch.cuda.empty_cache()
                    model, optimizer = make_model(h, metadata, device, initial)
                    trainer.restore_rng(h.torch, initial_rng, device)
                    history = {"training": [], "graph_updates": [], "elapsed_seconds": 0.0}
                    config = {"objective": "faithful", "arm": arm, "seed": SEED, "updates": PROBE_LR_HORIZON,
                        "graph_exposure": trainer.graph_exposure_config(arm), "purpose": "bounded_numerical_only_never_promote",
                        "scientific_training_admitted": False, "validation_source_sha256": sha(__file__)}
                    for step in range(2):
                        graph_step = 2 * index + step
                        noise = h.host_noise(batch[0].shape, batch[1], SEED, graph_step)
                        report["attempted_update"] = {"case": case["case"], "arm": arm, "branch": label, "step": step,
                            "graph_step": graph_step, "noise_sha256": candidate.state_hash(noise.numpy())}
                        save()
                        row["graph_gates"].append(graph_gate(h, trainer, original, candidate, model, batch, noise, device, graph_step, arm))
                        result = advance(h, trainer, graph, model, optimizer, batch, noise, device, step, graph_step, arm)
                        report["completed_optimizer_calls"] += 1
                        artifact = save_tensor(h.torch, args.output_dir / f"{case['case']}_{arm}_{label}_step{step+1}.pt", result)
                        if label == "original":
                            stored[step] = artifact
                        else:
                            expected = h.torch.load(args.output_dir / stored[step]["file"], map_location="cpu", weights_only=True)
                            comparison = exact_comparison(h, trainer, expected, result)
                            row["steps"].append({"completed_steps": step + 1, "original": stored[step], "candidate": artifact, **comparison})
                            require(comparison["passed"], "Same-CUDA original/candidate numerical mismatch")
                            del expected
                        if label == "candidate" and index == 0:
                            history["graph_updates"].append(history_row(case, candidate, noise, step, result["ledger"]))
                            if step == 0:
                                payload = copy.deepcopy(trainer.checkpoint_payload(h, model, optimizer, config, 1, history, device))
                                checkpoint = save_tensor(h.torch, args.output_dir / f"{arm}_bounded_replay_checkpoint_step1.pt", payload)
                            else:
                                expected_payload = copy.deepcopy(trainer.checkpoint_payload(h, model, optimizer, config, 2, history, device))
                                expected_result = artifact
                        del result
                        save()
                    if label == "candidate" and index == 0:
                        del model, optimizer
                        model, optimizer = make_model(h, metadata, device, initial)
                        loaded = h.torch.load(args.output_dir / checkpoint["file"], map_location="cpu", weights_only=True)
                        require(trainer.tree_equal(h.torch, loaded, payload), "Serialized checkpoint differs")
                        completed, restored_history = trainer.restore_payload(h, loaded, model, optimizer, config, device)
                        require(completed == 1 and trainer.tree_equal(h.torch, trainer.capture_rng(h.torch, device), loaded["rng_states"]),
                            "Replay checkpoint/RNG restoration differs")
                        report["attempted_update"] = {"case": "small", "arm": arm, "branch": "candidate_replay", "step": 1}
                        save()
                        noise = h.host_noise(batch[0].shape, batch[1], SEED, 1)
                        row["graph_gates"].append(graph_gate(h, trainer, original, candidate, model, batch, noise, device, 1, arm))
                        actual = advance(h, trainer, candidate, model, optimizer, batch, noise, device, 1, 1, arm)
                        report["completed_optimizer_calls"] += 1
                        artifact = save_tensor(h.torch, args.output_dir / f"{arm}_bounded_replay_step2.pt", actual)
                        expected = h.torch.load(args.output_dir / expected_result["file"], map_location="cpu", weights_only=True)
                        check = exact_comparison(h, trainer, expected, actual)
                        restored_history["graph_updates"].append(history_row(case, candidate, noise, 1, actual["ledger"]))
                        final_payload = trainer.checkpoint_payload(h, model, optimizer, config, 2, restored_history, device)
                        payload_exact = trainer.tree_equal(h.torch, expected_payload, final_payload)
                        report["replay"][arm] = {**check, "checkpoint": checkpoint, "resumed_artifact": artifact,
                            "checkpoint_serialization_exact": True, "final_payload_exact": payload_exact}
                        require(check["passed"] and payload_exact, "Same-CUDA checkpoint replay differs")
                        del actual, expected, final_payload, expected_payload, payload, loaded
                    del model, optimizer
                    model = optimizer = None
                    save()
        require(len(report["cases"]) == 6 and report["completed_optimizer_calls"] == 26
            and set(report["replay"]) == set(ARMS)
            and all(len(row["steps"]) == 2 and all(step["passed"] for step in row["steps"]) for row in report["cases"])
            and any(example["selected_optional_pairs"] > 0 for row in report["cases"] for gate in row["graph_gates"]
                for example in gate["graph_ledger"]), "Complete nonvacuous26-update graph/numerical/replay proof required")
        verify_data_evidence(args, h, trainer, release)
        h.data_loader.load_manifest_data(args.train_manifest, verify_hashes=True)
        require(sha(args.release) == report["release_sha256"] and sha(__file__) == report["source_sha256"], "Validation source/release changed")
        directory = Path(__file__).resolve().parent
        require(sha(directory / "train_goop3d_graph_support_cuda_v2.py") == TRAINER_SHA
            and sha(directory / "goop3d_graph_support.py") == ORIGINAL_SHA
            and sha(directory / "goop3d_graph_support_vectorized_v1.py") == CANDIDATE_SHA
            and all(sha(args.repo / key) == value for key, value in trainer.SOURCE_PINS.items())
            and all(sha(args.output_dir / row["file"]) == row["sha256"] for row in report["saved_batches"]),
            "Pinned source or saved batches changed")
        report.update(status="implementation_passed", all_inputs_reverified=True, attempted_update=None)
        return 0
    except BaseException as error:
        report.update(status="implementation_failed", error_type=type(error).__name__, error=str(error),
            unsuccessful_outcome_retained=True)
        if h is not None and model is not None and optimizer is not None:
            try:
                report["unsuccessful_state"] = save_tensor(h.torch, args.output_dir / "unsuccessful_state.pt",
                    {"state_dict": h.cpu_tree(model.state_dict()), "optimizer_state": h.cpu_tree(optimizer.state_dict()),
                     "gradients": {name: h.cpu_tree(p.grad) for name, p in model.named_parameters()},
                     "attempted_update": report["attempted_update"]})
            except BaseException as preservation_error:
                report["state_preservation_error"] = str(preservation_error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    for name in ("release", "repo", "train-manifest", "structural-report", "acquisition-report", "context-semantics",
                 "auxiliary-report", "data-review", "cpu-proof", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(description()))
        return 0
    require(all(getattr(args, key) is not None for key in ("release", "repo", *DATA_PINS, "output_dir", "gpu_uuid"))
        and args.cuda_index >= 0 and 1 <= args.threads <= 16, "Explicit bounded inputs/device/fresh output required")
    def interrupted(signum, frame):
        raise InterruptedError("Bounded validation interrupted by signal " + str(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return execute(args)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
