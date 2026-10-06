#!/usr/bin/env python3
"""Separate from-scratch Sand CUDA graph-support trainer. Default is description only.

Execution requires a separately admitted numeric train manifest and an explicit
research protocol, arm, seed and fixed 100000-update endpoint. This faithful-only
base/mix study has a distinct CUDA lineage; no prior checkpoint may initialize it. No data acquisition, tuning,
validation selection, test access, AMP, TF32, compile, DDP or device fallback.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
import uuid

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_graph_support_cuda_training_v1"
ADMISSION = "adaptgns_sand_training_admission_v1"
METADATA_SHA = "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0"
SOURCE_PINS = {
    "research/faithful_graph_support.py": "71c2b3740aab635665d28b37addbd0b89360cccb3136d275e54b864d3b2a7eaa",
    "research/graph_convention_bridge.py": "2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d",
    "research/full_rollout.py": "b0a37ee47619e699298b86865649e63c402cc1cb5dc2fa3dfd4055f7fa966eb8",
    "research/full_same_state.py": "ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592",
    "research/budget_graph.py": "951f0d13863672f1dd248febf8cc960ba500103ca95361e06c9b0577913a8187",
    "research/full_training.py": "c6bec98cb8200ef74c3d5d30309dba858c0dbef49f3380a0bc8762683c04e60d",
    "adaptive-gns/gns/data_loader.py": "287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef",
    "adaptive-gns/gns/device_utils.py": "324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326",
    "adaptive-gns/gns/graph_network.py": "af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91",
    "adaptive-gns/gns/learned_simulator.py": "216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf",
    "adaptive-gns/gns/losses.py": "94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5",
    "adaptive-gns/gns/model_io.py": "06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e",
}
RECORD_KEYS = {"id", "source_index", "source_member", "source_inner_index",
               "positions", "particle_types", "trajectory_content_sha256", "logical_content_sha256"}
ARRAY_KEYS = {"path", "shape", "dtype", "size_bytes", "sha256"}
MANIFEST_KEYS = {"format", "version", "split", "dataset", "source", "metadata",
                 "metadata_sha256", "record_count", "records", "converter_sha256"}
SOURCE_KEYS = {"family", "dataset", "file", "size_bytes", "sha256", "ZIP_CRC_verified",
               "member_count", "acquisition_report_sha256"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def digest_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def pid_alive(pid):
    if not integer(pid, 1):
        raise ValueError("Malformed lock PID; refusing recovery")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class RunLock:
    """Only an explicitly requested same-host, dead-PID lock may be removed."""
    def __init__(self, directory, clear_stale=False):
        self.path = Path(directory) / "run.lock"
        self.clear_stale = clear_stale
        self.payload = {"pid": os.getpid(), "host": socket.gethostname(),
                        "started_utc": utc_now(), "token": uuid.uuid4().hex}
        self.owned = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.clear_stale:
            recovery = self.path.with_name("run.lock.recovery")
            fd = os.open(recovery, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, json.dumps(self.payload).encode())
                os.fsync(fd)
                if self.path.exists():
                    old = json.loads(self.path.read_text())
                    if old.get("host") != self.payload["host"] or pid_alive(old.get("pid")):
                        raise RuntimeError("Cannot clear a live, foreign-host or unverifiable lock")
                    self.path.unlink()
            finally:
                os.close(fd)
                recovery.unlink()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, json.dumps(self.payload).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        self.owned = True
        return self.payload

    def __exit__(self, *_):
        if self.owned and self.path.exists():
            if json.loads(self.path.read_text()).get("token") == self.payload["token"]:
                self.path.unlink()


def description():
    return {"schema": SCHEMA, "status": "preparation_only", "endpoint_selected": False,
            "requires": "explicit --execute, admitted DesignSafe Sand numeric train manifest, protocol, base/mix arm, seed0/1/2 and updates100000",
            "architecture": "128 width, 10 message-passing blocks, 2 MLP layers; batch 2; history 6; 2D",
            "graph": "native strict radius .015/cap128/self prefix; mix independently exposes each example with probability.5 to an exact25% annulus append",
            "recipe": "noise 6.7e-4; original noise-adjusted normalization; faithful only; Adam 1e-4 to 1e-5 over 100000 updates",
            "precision": "float32; strict deterministic algorithms; CUBLAS_WORKSPACE_CONFIG=:4096:8; no AMP/TF32/compile/DDP; Adam foreach=False, fused=False",
            "lineage": "from-scratch faithful base/mix x seeds0/1/2; no old checkpoints; independent schema; no launch admission implied",
            "source_pins": SOURCE_PINS,
            "limits": "T and type IDs require observed-source admission; no material fields or type 3; no validation/test evaluation; no cross-device bitwise guarantee"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    for name in ("repo", "train-manifest", "admission", "structural-report", "protocol", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--objective", choices=("faithful",), default="faithful")
    parser.add_argument("--arm", choices=("base", "mix"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--updates", type=int)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        required = ("repo", "train_manifest", "admission", "structural_report", "protocol",
                    "output_dir", "arm", "objective", "seed", "updates")
        if any(getattr(args, name) is None for name in required):
            parser.error("Execution requires " + ", ".join("--" + name.replace("_", "-") for name in required))
        if args.seed not in (0, 1, 2) or args.updates != 100000:
            parser.error("Fixed seed0/1/2 and exactly100000 updates required")
    if (args.cuda_index < 0 or not 1 <= args.threads <= 16
            or min(args.checkpoint_every, args.log_every) < 1):
        parser.error("Invalid CUDA index, thread count or logging/checkpoint interval")
    if args.stop_after is not None and (args.updates is None or not 0 < args.stop_after <= args.updates):
        parser.error("stop-after must be positive and no greater than the explicit endpoint")
    if args.clear_stale_lock and not args.resume:
        parser.error("Dead-lock recovery requires --resume; never restart over a previous run")
    return args


def load_helpers(repo):
    repo = Path(repo).resolve()
    if {name: sha(repo / name) for name in SOURCE_PINS} != SOURCE_PINS:
        raise ValueError("Frozen graph-support/helper/core source mismatch")
    names = {relative.removeprefix("adaptive-gns/").removesuffix(".py").replace("/", "."): repo / relative
             for relative in SOURCE_PINS}
    for name, path in names.items():
        if name in sys.modules and Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise ValueError("Unrelated cached module: " + name)
    sys.path[:0] = [str(repo), str(repo / "adaptive-gns")]
    support = importlib.import_module("research.faithful_graph_support")
    for name, path in names.items():
        if name not in sys.modules or Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise ValueError("Imported helper is not pinned: " + name)
    return support.original, support


def graph_exposure_config(arm):
    if arm not in ("base", "mix"):
        raise ValueError("Unknown graph-support arm")
    return {"arm": arm, "mixture_probability_per_example": .5 if arm == "mix" else 0.,
            "radius_factor": 1.267, "optional_pair_fraction": .25,
            "mandatory": "unchanged native strict-r directed cap128/self prefix",
            "optional": "strict full uncapped E_R minus strict E_r; symmetric ordered suffix, never recapped",
            "coin_seed_material": "[20261005,seed,completed_step,example_slot,4409]",
            "pair_seed_material": "[20261005,seed,completed_step,example_slot,5501]",
            "selection_inputs": "noisy observed history only; no future target"}


def validate_graph_history(history, completed, arm, seed):
    if not isinstance(history, dict) or set(history) != {"training", "graph_updates", "elapsed_seconds"}:
        raise ValueError("Malformed graph-support checkpoint history")
    if not math.isfinite(history["elapsed_seconds"]) or history["elapsed_seconds"] < 0:
        raise ValueError("Invalid graph-support elapsed time")
    if any(not integer(row.get("completed_steps"), 1) or row["completed_steps"] > completed for row in history["training"]):
        raise ValueError("Invalid graph-support training history index")
    records = history["graph_updates"]
    if not isinstance(records, list) or len(records) != completed:
        raise ValueError("Every completed update requires a graph/schedule record")
    for index, record in enumerate(records):
        if record.get("completed_steps") != index + 1 or record.get("absolute_schedule_step") != index:
            raise ValueError("Graph history update schedule differs")
        if not digest_string(record.get("noise_sha256")) or not isinstance(record.get("frame_ids"), list) or len(record["frame_ids"]) != 2:
            raise ValueError("Paired host schedule evidence is missing")
        examples = record.get("examples", [])
        if len(examples) != 2 or [row.get("example_slot") for row in examples] != [0, 1]:
            raise ValueError("Two ordered graph-exposure examples required")
        for row in examples:
            active = arm == "mix" and row.get("exposure_coin") is True
            if row.get("expanded") is not active or type(row.get("exposure_coin")) is not bool:
                raise ValueError("Arm/exposure history differs")
            if not integer(row.get("annulus_pairs")) or row.get("optional_budget_if_exposed") != math.floor(.25 * row["annulus_pairs"]):
                raise ValueError("Graph-exposure optional budget differs")
            if row.get("selected_optional_pairs") != (row["optional_budget_if_exposed"] if active else 0):
                raise ValueError("Graph-exposure selected count differs")
            if not all(digest_string(row.get(key)) for key in ("native_edge_sha256", "optional_pair_sha256", "noisy_current_sha256")):
                raise ValueError("Graph-exposure identity hash missing")
            if (row.get("coin_seed_material") != [20261005, seed, index, row["example_slot"], 4409]
                    or row.get("pair_seed_material") != [20261005, seed, index, row["example_slot"], 5501]):
                raise ValueError("Graph-exposure RNG schedule differs")


def validate_manifest_contract(manifest, admission):
    if admission.get("schema") != ADMISSION or admission.get("status") != "admitted" or admission.get("dataset") != "Sand":
        raise ValueError("Root-issued Sand admission is required")
    for key in ("manifest_sha256", "metadata_sha256", "converter_sha256", "structural_report_sha256"):
        if not digest_string(admission.get(key)):
            raise ValueError("Admission is missing a SHA256: " + key)
    if (set(manifest) != MANIFEST_KEYS or manifest.get("format") != "gns-trajectory-manifest"
            or manifest.get("version") != 1 or manifest.get("split") != "train" or manifest.get("dataset") != "Sand"):
        raise ValueError("Only a version-1 numeric training manifest is accepted")
    source = manifest.get("source", {})
    if (set(source) != SOURCE_KEYS or source.get("family") != "designsafe_published_npz" or source.get("dataset") != "Sand"
            or source.get("ZIP_CRC_verified") is not True or not digest_string(source.get("sha256"))
            or source.get("file") != "train.npz" or not integer(source.get("size_bytes"), 1) or not integer(source.get("member_count"), 1)
            or not digest_string(source.get("acquisition_report_sha256"))):
        raise ValueError("DesignSafe Sand source identity/integrity is missing")
    if (manifest.get("metadata_sha256") != METADATA_SHA or admission["metadata_sha256"] != METADATA_SHA
            or manifest.get("converter_sha256") != admission["converter_sha256"]):
        raise ValueError("Metadata/converter identity differs from admission")
    frames, count, types = (admission.get(key) for key in ("frames_per_trajectory", "record_count", "particle_type_ids"))
    if not integer(frames, 7) or not integer(count, 1):
        raise ValueError("A single observed frame length and complete record count must be admitted")
    if (not isinstance(types, list) or not types or any(not integer(t) or t > 8 or t == 3 for t in types)
            or types != sorted(set(types)) or admission.get("position_dtype") != "<f4"):
        raise ValueError("Admit exact type IDs without type 3 and float32 positions; other precision needs review")
    records = manifest.get("records", [])
    if len(records) != count or manifest.get("record_count") != count or source["member_count"] != count:
        raise ValueError("Complete record counts differ from admission")
    ids, contents, logical_contents = set(), set(), set()
    for index, record in enumerate(records):
        if set(record) - RECORD_KEYS or not {"id", "source_index", "source_member", "positions", "particle_types", "trajectory_content_sha256"} <= set(record):
            raise ValueError("Unexpected/missing record fields; material and auxiliary inputs require review")
        if (record["id"] != f"train:{index:06d}" or not integer(record["source_index"])
                or record["source_index"] != index or record["id"] in ids
                or not isinstance(record["source_member"], str) or not record["source_member"]
                or ("source_inner_index" in record and not integer(record["source_inner_index"]))):
            raise ValueError("Source ordering or unique ordinal IDs are inconsistent")
        ids.add(record["id"])
        p, t = record["positions"], record["particle_types"]
        for array in (p, t):
            if set(array) != ARRAY_KEYS or not digest_string(array["sha256"]) or not integer(array["size_bytes"], 1):
                raise ValueError("Array descriptions require exact numeric hash/shape/dtype/size keys")
        shape = p["shape"]
        if (not isinstance(shape, list) or len(shape) != 3 or shape[0] != frames or shape[2] != 2
                or not integer(shape[1], 1) or p["dtype"] != "<f4"
                or t["shape"] not in ([], [shape[1]]) or t["dtype"] not in ("<i8", "<i4")):
            raise ValueError("Unexpected trajectory shape/dtype; no silent conversion is permitted")
        content = hashlib.sha256((p["sha256"] + ":" + t["sha256"]).encode()).hexdigest()
        if record["trajectory_content_sha256"] != content or content in contents:
            raise ValueError("Duplicate trajectory or inconsistent content hash")
        contents.add(content)
        logical = record.get("logical_content_sha256")
        if not digest_string(logical) or logical in logical_contents:
            raise ValueError("Missing/duplicate logical trajectory content")
        logical_contents.add(logical)


def load_admitted_dataset(args, helpers):
    admission = json.loads(args.admission.read_text())
    manifest = json.loads(args.train_manifest.read_text())
    validate_manifest_contract(manifest, admission)
    if sha(args.train_manifest) != admission["manifest_sha256"] or sha(args.structural_report) != admission["structural_report_sha256"]:
        raise ValueError("Manifest/structural report does not match root admission")
    metadata_path = args.train_manifest.resolve().parent / "metadata.json"
    if sha(metadata_path) != METADATA_SHA:
        raise ValueError("Actual Sand metadata bytes differ")
    metadata = json.loads(metadata_path.read_text())
    if metadata != manifest.get("metadata") or metadata.get("dim") != 2 or metadata.get("sequence_length") != 320:
        raise ValueError("Manifest metadata mismatch")
    dataset, _, info = helpers.load_manifest_dataset(args.train_manifest, "train")
    for record in manifest["records"]:
        for name in ("positions", "particle_types"):
            path = helpers.data_loader._manifest_array_path(args.train_manifest.resolve().parent, record[name])
            if path.stat().st_size != record[name]["size_bytes"]:
                raise ValueError("Actual numeric array byte length differs from its manifest")
    observed_types = set()
    for positions, particle_types in dataset._data:
        if positions.shape[0] != admission["frames_per_trajectory"] or positions.shape[2] != 2:
            raise ValueError("Loaded frame count/dimension differs")
        if not helpers.np.isfinite(positions).all():
            raise ValueError("Nonfinite source trajectory")
        observed_types.update(int(value) for value in helpers.np.unique(particle_types))
    if sorted(observed_types) != admission["particle_type_ids"] or 3 in observed_types:
        raise ValueError("Observed types differ from admission; type 3 is not supported")
    info.update({"frames_per_trajectory": admission["frames_per_trajectory"], "particle_type_ids": sorted(observed_types),
                 "admission_sha256": sha(args.admission), "structural_report_sha256": sha(args.structural_report),
                 "converter_sha256": admission["converter_sha256"], "metadata_sha256": METADATA_SHA})
    return dataset, metadata, info


def capture_rng(torch, device):
    return {"cpu": torch.get_rng_state().cpu().clone(),
            "cuda": torch.cuda.get_rng_state(device).cpu().clone()}


def restore_rng(torch, states, device):
    if set(states) != {"cpu", "cuda"}:
        raise ValueError("Both CPU and selected CUDA RNG states are required")
    for value in states.values():
        if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel():
            raise ValueError("Malformed RNG state")
    torch.set_rng_state(states["cpu"])
    torch.cuda.set_rng_state(states["cuda"], device)
    if not tree_equal(torch, states, capture_rng(torch, device)):
        raise ValueError("RNG restoration was not exact")


def tree_equal(torch, a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and bool(torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                     b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(torch, a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(torch, x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def assert_adam(torch, model, optimizer, completed):
    parameters = list(model.parameters())
    if len(optimizer.param_groups) != 1:
        raise ValueError("Exactly one Adam parameter group is required")
    group = optimizer.param_groups[0]
    if len(group["params"]) != len(parameters) or any(x is not y for x, y in zip(group["params"], parameters)):
        raise ValueError("Adam parameter order differs from the model")
    expected = {"betas": (.9, .999), "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False,
                "maximize": False, "foreach": False, "capturable": False, "differentiable": False, "fused": False}
    if any(group.get(key) != value for key, value in expected.items()):
        raise ValueError("Adam hyperparameters differ from the fixed recipe")
    expected_lr = 1e-4 * (.1 ** (min(max(completed - 1, 0) / 99999, 1.0)))
    if group["lr"] != expected_lr:
        raise ValueError("Adam learning rate differs from the fixed update schedule")
    if completed == 0:
        if optimizer.state:
            raise ValueError("Step-zero Adam must have no moment state")
        return
    if not parameters or set(optimizer.state) != set(parameters):
        raise ValueError("Adam state does not cover exactly every model parameter")
    moment_checks = []
    for p in parameters:
        state = optimizer.state[p]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unexpected or incomplete Adam moment state")
        counter = state["step"]
        if counter.device.type != "cpu" or counter.numel() != 1 or float(counter) != completed:
            raise ValueError("Noncapturable Adam step must be an exact CPU update counter")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if value.device != p.device or value.shape != p.shape or value.dtype != p.dtype:
                raise ValueError("Adam moment shape/device/dtype differs")
            moment_checks.append(torch.isfinite(value).all())
            if key == "exp_avg_sq":
                moment_checks.append((value >= 0).all())
    # Preserve every predicate, but synchronize only the aggregate device scalar.
    if not bool(torch.stack(moment_checks).all()):
        raise FloatingPointError("Adam moments are nonfinite or second moments are negative")


def guarded_update(helpers, model, optimizer, pred, head, target, mask, objective, next_step):
    torch = helpers.torch
    if not helpers.tensors_are_finite((pred, head, target)) or not bool((head > 0).all()):
        raise FloatingPointError("Nonfinite prediction/variance/target or nonpositive variance")
    loss = helpers.acceleration_loss(pred, target, mask, pred_variance=head,
                                     loss_type=objective, variance_floor=1e-6)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite loss; no optimizer update")
    loss.backward()
    if any(p.grad is None for p in model.parameters()):
        raise FloatingPointError("Missing parameter gradient; no optimizer update")
    helpers.assert_finite_gradients(model)
    optimizer.step()
    if not helpers.tensors_are_finite(model.parameters()):
        raise FloatingPointError("Nonfinite parameter after update; committed checkpoint retained")
    assert_adam(torch, model, optimizer, next_step)
    return loss


def checkpoint_payload(helpers, model, optimizer, config, completed, history, device):
    validate_graph_history(history, completed, config["arm"], config["seed"])
    if not helpers.tensors_are_finite(model.state_dict().values()):
        raise FloatingPointError("Cannot commit nonfinite model state")
    assert_adam(helpers.torch, model, optimizer, completed)
    return {"format_version": 2, "cuda_sand_graph_support_schema": SCHEMA,
            "state_dict": helpers.cpu_tree(model.state_dict()),
            "simulator_config": helpers.cpu_tree(model._checkpoint_config),
            "training_config": {"loss": config["objective"], "cuda_sand_graph_support_run": config,
                                "completed_optimizer_updates": completed},
            "optimizer_state": helpers.cpu_tree(optimizer.state_dict()),
            "run_config": config, "run_config_sha256": config_hash(config),
            "completed_steps": completed, "history": history,
            "rng_states": helpers.cpu_tree(capture_rng(helpers.torch, device))}


def restore_payload(helpers, payload, model, optimizer, config, device):
    torch = helpers.torch
    completed = payload.get("completed_steps")
    if (payload.get("format_version") != 2 or payload.get("cuda_sand_graph_support_schema") != SCHEMA
            or any(key in payload for key in ("cuda_sand_training_schema", "full_training_schema", "graph_support_schema"))
            or payload.get("run_config_sha256") != config_hash(config) or payload.get("run_config") != config
            or not integer(completed) or completed > config["updates"]
            or payload.get("training_config") != {"loss": config["objective"], "cuda_sand_graph_support_run": config,
                                                  "completed_optimizer_updates": completed}):
        raise ValueError("Checkpoint lineage/configuration/endpoint mismatch")
    if not tree_equal(torch, payload.get("simulator_config"), helpers.cpu_tree(model._checkpoint_config)):
        raise ValueError("Simulator configuration differs")
    history = payload["history"]
    validate_graph_history(history, completed, config["arm"], config["seed"])
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if not tree_equal(torch, payload["state_dict"], helpers.cpu_tree(model.state_dict())):
        raise ValueError("Model restoration was not exact")
    if not tree_equal(torch, payload["optimizer_state"], helpers.cpu_tree(optimizer.state_dict())):
        raise ValueError("Optimizer restoration was not exact")
    if not helpers.tensors_are_finite(model.parameters()):
        raise FloatingPointError("Checkpoint model contains nonfinite parameters")
    assert_adam(torch, model, optimizer, completed)
    restore_rng(torch, payload["rng_states"], device)
    return completed, history


def read_pointer(output, config):
    pointer = json.loads((output / "latest.json").read_text())
    step = pointer.get("completed_steps")
    if (set(pointer) != {"path", "sha256", "completed_steps", "run_config_sha256"}
            or not integer(step) or step > config["updates"]
            or pointer["path"] != f"checkpoint-{step:09d}.pt"
            or pointer["run_config_sha256"] != config_hash(config)
            or not digest_string(pointer["sha256"])
            or (output / pointer["path"]).resolve().parent != output.resolve()
            or sha(output / pointer["path"]) != pointer["sha256"]):
        raise ValueError("Invalid committed checkpoint pointer")
    return pointer


def configure_cuda(helpers, args):
    torch = helpers.torch
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic variant requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before process start")
    torch.use_deterministic_algorithms(True, warn_only=False)
    if str(torch.__version__) != "2.13.0+cu129" or torch.version.cuda != "12.9":
        raise RuntimeError("Only the validated Torch 2.13.0+cu129 stack is admitted by this entry")
    if not torch.cuda.is_available() or args.cuda_index >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA device unavailable; no fallback")
    device = torch.device("cuda", args.cuda_index)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    properties = torch.cuda.get_device_properties(device)
    if "GB200" not in properties.name:
        raise RuntimeError("This prepared entry is bounded to the validated GB200 hardware")
    runtime = {"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
               "numpy": str(helpers.np.__version__), "scipy": str(helpers.scipy.__version__),
               "python": platform.python_version(), "platform": platform.platform(),
               "device": str(device), "name": properties.name, "uuid": str(properties.uuid),
               "capability": list(torch.cuda.get_device_capability(device)), "threads": args.threads,
               "tf32": False, "amp": False, "compile": False, "ddp": False,
               "float32_matmul_precision": torch.get_float32_matmul_precision(),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
               "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    return device, runtime


def run_training(args, helpers, support, process, device, runtime):
    torch = helpers.torch
    train, metadata, data_info = load_admitted_dataset(args, helpers)
    config = {"schema": SCHEMA, "dataset": "Sand", "objective": args.objective,
              "seed": args.seed, "arm": args.arm, "updates": args.updates, "batch_size": 2, "history": 6,
              "initialization": "from scratch; paired seed across arms; empty Adam; no parent checkpoint",
              "graph_exposure": graph_exposure_config(args.arm),
              "architecture": {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
              "noise_std": helpers.NOISE, "graph": {"radius": .015, "backend": "scipy_host", "cap": 128,
                                                       "self_candidates": True, "augmentation_probability": 0.0},
              "optimizer": {"name": "Adam", "initial_lr": 1e-4, "final_lr": 1e-5, "decay_updates": 100000,
                            "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0,
                            "foreach": False, "fused": False, "gradient_clipping": None},
              "checkpoint_every": args.checkpoint_every, "log_every": args.log_every,
              "data": data_info, "research_protocol_sha256": sha(args.protocol),
              "source_sha256": {**SOURCE_PINS, "train_sand_graph_support_cuda.py": sha(__file__)}, "runtime": runtime,
              "selection": "fixed final requested update; no validation/test selection"}
    output = args.output_dir
    protected = (args.train_manifest.resolve().parent, (args.repo / "adaptive-gns").resolve())
    resolved_output = output.resolve()
    if any(resolved_output == path or path in resolved_output.parents or resolved_output in path.parents for path in protected):
        raise ValueError("Graph-support output must be separate from source/data trees")
    if list(output.rglob("*.tmp")):
        raise ValueError("Prior partial artifacts require review before any resume write")
    protocol = output / "protocol.json"
    if protocol.exists():
        if not args.resume or json.loads(protocol.read_text()) != config:
            raise ValueError("Existing output requires exact matching --resume configuration")
    elif args.resume or any(path.name != "run.lock" for path in output.iterdir()):
        raise ValueError("Resume requires protocol.json; fresh output must be empty")
    else:
        atomic_json(protocol, config)
    torch.manual_seed(args.seed)
    with torch.cuda.device(device):
        torch.cuda.manual_seed(args.seed)
    model = helpers.build_simulator(metadata, helpers.NOISE, helpers.NOISE, device,
        connectivity_radius=.015, nmessage_passing_steps=10, uncertainty_parameterization="variance",
        variance_floor=1e-6, detach_variance_features=True, radius_backend="scipy_host").to(device)
    model._training_config = {"loss": args.objective, "cuda_sand_graph_support_run": config}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False)
    completed, history, latest = 0, {"training": [], "graph_updates": [], "elapsed_seconds": 0.0}, None
    if args.resume:
        latest = read_pointer(output, config)  # Missing pointers require manual recovery; no silent restart.
        payload = torch.load(output / latest["path"], map_location="cpu", weights_only=True)
        completed, history = restore_payload(helpers, payload, model, optimizer, config, device)
        if completed != latest["completed_steps"]:
            raise ValueError("Pointer and checkpoint update counts disagree")
    stop = args.updates if args.stop_after is None else args.stop_after
    if stop < completed:
        raise ValueError("Requested stop precedes restored updates")
    started, prior_elapsed = time.perf_counter(), history["elapsed_seconds"]
    attempt_id = uuid.uuid4().hex
    attempt_dir = output / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    current_context = {"state": "before_first_update"}

    def status(state, error=None):
        record = {"schema": SCHEMA, "state": state, "attempt_id": attempt_id,
            "completed_steps": completed, "committed_steps": latest["completed_steps"] if latest else None,
            "requested_steps": args.updates, "objective": args.objective, "arm": args.arm, "seed": args.seed,
            "run_config_sha256": config_hash(config), "process": process, "updated_utc": utc_now(),
            "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
            "latest_checkpoint": latest, "last_training": history["training"][-1] if history["training"] else None,
            "error": error}
        atomic_json(attempt_dir / "status.json", record)
        atomic_json(output / "status.json", record)

    def checkpoint_now():
        nonlocal latest
        helpers.synchronize(device)
        history["elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
        payload = checkpoint_payload(helpers, model, optimizer, config, completed, history, device)
        path = output / f"checkpoint-{completed:09d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        if path.exists():
            raise ValueError("Refusing to overwrite an existing graph-support checkpoint")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        pointer = {"path": path.name, "sha256": sha(path), "completed_steps": completed,
                   "run_config_sha256": config_hash(config)}
        atomic_json(output / "latest.json", pointer)
        latest = pointer  # Report committed only after the atomic pointer publication succeeds.

    try:
        status("running")
        if latest is None:
            checkpoint_now()
        while completed < stop:
            helpers.synchronize(device)
            update_started = time.perf_counter()
            indices = helpers.sample_indices(args.seed, completed, len(train), 2)
            batch = helpers.unpack_batch([train[index] for index in indices])
            noise = helpers.host_noise(batch[0].shape, batch[1], args.seed, completed)
            frame_ids = [helpers.frame_identity(train, data_info["trajectory_ids"], index) for index in indices]
            current_context = {"completed_before": completed, "absolute_schedule_step": completed,
                "frame_ids": frame_ids, "noise_sha256": support.bridge.full.state_hash(noise.numpy())}
            rate = helpers.learning_rate(completed)
            for group in optimizer.param_groups:
                group["lr"] = rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred, head, target, graph_ledger = support.forward_batch(model, batch, noise, device, args.seed, completed, args.arm)
            current_context["examples"] = graph_ledger
            mask = (batch[1] != helpers.KINEMATIC).to(device)
            loss = guarded_update(helpers, model, optimizer, pred, head, target, mask, args.objective, completed + 1)
            helpers.synchronize(device)
            update_seconds = time.perf_counter() - update_started
            completed += 1
            history["graph_updates"].append({"completed_steps": completed, "absolute_schedule_step": completed - 1,
                "frame_ids": frame_ids, "noise_sha256": current_context["noise_sha256"], "examples": graph_ledger})
            if completed % args.log_every == 0 or completed in (1, stop):
                row = {"completed_steps": completed, "loss": float(loss.detach().cpu()), "lr": rate,
                       "frame_ids": frame_ids,
                       "particles": len(batch[0]), "guarded_update_seconds": update_seconds,
                       "elapsed_seconds": prior_elapsed + time.perf_counter() - started}
                history["training"].append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
                status("running")
            if completed % args.checkpoint_every == 0 or completed == stop:
                checkpoint_now()
        for relative, expected in config["source_sha256"].items():
            path = Path(__file__) if relative == "train_sand_graph_support_cuda.py" else args.repo / relative
            if sha(path) != expected:
                raise ValueError("Source changed during graph-support training: " + relative)
        for path, expected in ((args.train_manifest, data_info["manifest_sha256"]),
                               (args.admission, data_info["admission_sha256"]),
                               (args.structural_report, data_info["structural_report_sha256"]),
                               (args.protocol, config["research_protocol_sha256"])):
            if sha(path) != expected:
                raise ValueError("Input changed during graph-support training: " + str(path))
        helpers.data_loader.load_manifest_data(args.train_manifest, verify_hashes=True)
        status("complete" if completed == args.updates else "planned_stop_incomplete")
        atomic_json(output / "history.json", history)
    except BaseException as error:
        atomic_json(attempt_dir / "unsuccessful_context.json", current_context)
        atomic_json(attempt_dir / "unsuccessful_history.json", history)
        try:
            with (attempt_dir / "unsuccessful_state.pt").open("xb") as stream:
                torch.save({"state_dict": helpers.cpu_tree(model.state_dict()),
                    "optimizer_state": helpers.cpu_tree(optimizer.state_dict()), "completed_steps": completed,
                    "error_type": type(error).__name__}, stream)
        except BaseException as preservation_error:
            atomic_json(attempt_dir / "state_preservation_error.json", {"error": str(preservation_error)})
        status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", f"{type(error).__name__}: {error}")
        raise


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps(description(), indent=2))
        return
    helpers, support = load_helpers(args.repo)
    device, runtime = configure_cuda(helpers, args)
    with RunLock(args.output_dir, args.clear_stale_lock) as process:
        run_training(args, helpers, support, process, device, runtime)


if __name__ == "__main__":
    main()
