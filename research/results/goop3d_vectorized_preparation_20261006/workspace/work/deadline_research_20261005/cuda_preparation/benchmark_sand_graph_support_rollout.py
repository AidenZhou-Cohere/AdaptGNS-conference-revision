#!/usr/bin/env python3
"""Separate Sand base/mix graph-support CUDA timing wrapper; old studies preserved.

The five core policies delegate to pinned native_graph_rollout.rollout unchanged;
the separate RMS policy uses a reviewed graph callback and retained loop. This
entry only times infrastructure checkpoints; the separate final entry gates test.
No acquisition, training, resume, checkpoint selection or data conversion.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_sand_graph_support_rollout_timing_v1"
ADMISSION_SCHEMA = "adaptgns_sand_rollout_admission_v1"
TRAINING_SCHEMA = "adaptgns_sand_graph_support_cuda_training_v1"
DEADLINE = datetime(2026, 10, 7, 8, tzinfo=timezone.utc)
METADATA_SHA = "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0"
SOURCES = {
    "train": (1000, 2676940383, "e0b7f68b50702f1af3edfa828b6097e3b28158ddb57631491e60a29658318f7d"),
    "valid": (30, 82712898, "f9c29861107b481ad5eb4a340f4f1016fcaf4050cd6cfc8ae3056f148ab6d903"),
}
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25", "relative-velocity-RMS25")
CORE_POLICIES = POLICIES[:-1]
POLICY_SHA = "4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a"
TRAINER_SHA = "fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124"
DIAGNOSTIC_SHA = "952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58"
SOURCE_PINS = {
    "research/native_graph_rollout.py": "b4bbca6660dc449c81958010e30fda8be2bc0aaf58f677e0946d26e600e6daf5",
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
MANIFEST_KEYS = {"format", "version", "split", "dataset", "source", "metadata", "metadata_sha256",
                 "record_count", "records", "converter_sha256"}
SOURCE_KEYS = {"family", "dataset", "file", "size_bytes", "sha256", "ZIP_CRC_verified",
               "member_count", "acquisition_report_sha256"}
RECORD_KEYS = {"id", "source_index", "source_member", "source_inner_index", "positions", "particle_types",
               "trajectory_content_sha256", "logical_content_sha256"}
ARRAY_KEYS = {"path", "shape", "dtype", "size_bytes", "sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("feasibility", "locked-test"), default="feasibility")
    parser.add_argument("--diagnostic-mode", choices=("full-rollout", "same-state", "clean-validation"), default="full-rollout")
    for name in ("repo", "manifest", "admission", "structural-report", "protocol", "output-dir", "checkpoint", "trainer-source", "train-admission", "training-protocol", "timing-release"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--split", choices=("train", "valid"))
    parser.add_argument("--model-kind", choices=("initialized", "checkpoint"))
    parser.add_argument("--objective", choices=("faithful",), default="faithful")
    parser.add_argument("--arm", choices=("base", "mix"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--checkpoint-updates", type=int)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=int, default=900,
                        help="Elapsed execution budget for case admission/alarm; setup and final integrity I/O are also recorded")
    args = parser.parse_args(argv)
    if args.execute:
        if args.mode != "feasibility":
            parser.error("Locked test execution is not implemented/admitted; no test source is opened")
        required = ("repo", "manifest", "admission", "structural_report", "protocol", "output_dir", "split", "model_kind", "objective", "arm", "seed", "train_admission", "training_protocol", "timing_release")
        if any(getattr(args, key) is None for key in required):
            parser.error("Execution requires " + ", ".join("--" + key.replace("_", "-") for key in required))
        if args.seed not in (0, 1, 2):
            parser.error("Fixed seed0/1/2 required")
        if args.model_kind != "checkpoint":
            parser.error("Only separately admitted512-update graph-support infrastructure checkpoints are supported")
        if args.diagnostic_mode != "full-rollout" and args.split != "valid":
            parser.error("Diagnostic timing uses validation only; never reinterpret training as a test split")
        checkpoint_fields = (args.checkpoint, args.checkpoint_sha256, args.checkpoint_updates, args.trainer_source)
        if args.model_kind == "checkpoint":
            if (any(value is None for value in checkpoint_fields) or not digest(args.checkpoint_sha256)
                    or args.checkpoint_updates != 512):
                parser.error("Checkpoint mode requires its preselected path/SHA, exactly512 infrastructure updates and trainer source")
        elif any(value is not None for value in checkpoint_fields):
            parser.error("Initialized mode does not accept checkpoint arguments")
    if args.cuda_index < 0 or not 1 <= args.threads <= 16 or not 1 <= args.max_seconds <= 7200:
        parser.error("Invalid CUDA index, thread count or bounded execution budget (1..7200 seconds)")
    return args


def select_cases(records):
    require(len(records) >= 3, "Three source-size-selected cases require at least three trajectories")
    ranked = sorted(records, key=lambda row: (row["positions"]["shape"][1], row["source_index"]))
    choices = (("small", 0), ("median", (len(ranked) - 1) // 2), ("large", len(ranked) - 1))
    return [{"size_group": label, "size_rank": rank, "source_index": ranked[rank]["source_index"],
             "trajectory_id": ranked[rank]["id"], "particles": ranked[rank]["positions"]["shape"][1]}
            for label, rank in choices]


def validate_contract(manifest, admission, structural, split):
    require(split in SOURCES, "Only training/validation feasibility splits are supported")
    require(admission.get("schema") == ADMISSION_SCHEMA and admission.get("status") == "admitted_for_feasibility"
            and admission.get("dataset") == "Sand" and admission.get("split") == split,
            "An explicit root-issued Sand split admission for feasibility is required")
    for key in ("manifest_sha256", "structural_report_sha256", "metadata_sha256", "converter_sha256"):
        require(digest(admission.get(key)), "Admission lacks exact SHA256: " + key)
    require(set(manifest) == MANIFEST_KEYS and manifest.get("format") == "gns-trajectory-manifest"
            and manifest.get("version") == 1 and manifest.get("split") == split and manifest.get("dataset") == "Sand",
            "Exact version-1 numeric Sand manifest required")
    count, size, source_sha = SOURCES[split]
    source = manifest["source"]
    require(set(source) == SOURCE_KEYS and source.get("family") == "designsafe_published_npz"
            and source.get("dataset") == "Sand" and source.get("file") == split + ".npz"
            and source.get("size_bytes") == size and source.get("sha256") == source_sha
            and source.get("member_count") == count and source.get("ZIP_CRC_verified") is True
            and digest(source.get("acquisition_report_sha256")), "Exact acquired source identity differs")
    frames, type_ids = admission.get("frames_per_trajectory"), admission.get("particle_type_ids")
    require(integer(frames, 8) and admission.get("record_count") == count and admission.get("position_dtype") == "<f4",
            "Admit the actual common frame count T>=8, complete count and float32 source precision")
    require(isinstance(type_ids, list) and type_ids and type_ids == sorted(set(type_ids))
            and all(integer(value) and value <= 8 and value != 3 for value in type_ids), "Unsupported particle type admission")
    require(manifest.get("metadata_sha256") == admission["metadata_sha256"] == METADATA_SHA
            and manifest.get("converter_sha256") == admission["converter_sha256"], "Metadata/converter identity differs")
    require(structural.get("schema") == "designsafe_sand_numeric_repackage_v1"
            and structural.get("status") == "complete_structural_only" and structural.get("dataset") == "Sand"
            and structural.get("source_family") == "designsafe_published_npz"
            and structural.get("metadata_sha256") == METADATA_SHA
            and structural.get("converter_sha256") == admission["converter_sha256"]
            and structural.get("acquisition_report_sha256") == source["acquisition_report_sha256"],
            "Complete structural conversion provenance is required")
    report = structural.get("splits", {}).get(split, {})
    require(report.get("manifest_sha256") == admission["manifest_sha256"] and report.get("record_count") == count
            and report.get("frame_lengths") == [frames] and report.get("position_dtypes") == ["<f4"]
            and report.get("particle_type_ids") == type_ids and report.get("kinematic_type3_particles") == 0
            and report.get("ZIP_CRC_verified") is True and report.get("all_numeric_values_preserved_exact") is True,
            "Structural source properties differ from split admission")
    records = manifest["records"]
    require(len(records) == manifest["record_count"] == count, "Complete source record count differs")
    require(len(report.get("records", [])) == count, "Complete per-record structural evidence required")
    contents, logical, paths, member_names = set(), set(), set(), set()
    for index, record in enumerate(records):
        require(not (set(record) - RECORD_KEYS) and (RECORD_KEYS - {"source_inner_index"}) <= set(record)
                and record["id"] == f"{split}:{index:06d}" and integer(record["source_index"])
                and record["source_index"] == index and isinstance(record["source_member"], str)
                and re.fullmatch(r"simulation_trajectory_[0-9]+\.npy", record["source_member"]) is not None
                and record["source_member"] not in member_names
                and ("source_inner_index" not in record or integer(record["source_inner_index"])),
                "Record fields or ordered source identity differs")
        member_names.add(record["source_member"])
        p, t = record["positions"], record["particle_types"]
        for array in (p, t):
            require(set(array) == ARRAY_KEYS and digest(array["sha256"]) and integer(array["size_bytes"], 1)
                    and isinstance(array["path"], str) and array["path"] not in paths, "Array identity/hash/size differs")
            paths.add(array["path"])
        shape = p["shape"]
        require(isinstance(shape, list) and len(shape) == 3 and shape[0] == frames and integer(shape[1], 1)
                and shape[2] == 2 and p["dtype"] == "<f4" and t["shape"] in ([], [shape[1]])
                and t["dtype"] in ("<i4", "<i8"), "Array shape/dtype differs; no implicit conversion is allowed")
        detail = report["records"][index]
        require(detail.get("source_index") == index and detail.get("source_member") == record["source_member"]
                and detail.get("frames") == frames and detail.get("particles") == shape[1]
                and detail.get("position_dtype") == p["dtype"] and detail.get("particle_type_dtype") == t["dtype"]
                and detail.get("particle_type_shape") == t["shape"]
                and detail.get("numeric_dtype_shape_values_verified_exact") is True,
                "Record identity/shape differs from preserved structural evidence")
        content = hashlib.sha256((p["sha256"] + ":" + t["sha256"]).encode()).hexdigest()
        require(record["trajectory_content_sha256"] == content and content not in contents
                and digest(record["logical_content_sha256"]) and record["logical_content_sha256"] not in logical,
                "Duplicate/inconsistent trajectory content")
        require(digest(detail.get("source_position_value_sha256")) and digest(detail.get("source_type_value_sha256"))
                and record["logical_content_sha256"] == hashlib.sha256(
                    (detail["source_position_value_sha256"] + ":" + detail["source_type_value_sha256"]).encode()).hexdigest(),
                "Logical source values differ from the structural conversion record")
        contents.add(content)
        logical.add(record["logical_content_sha256"])
    return frames


def load_helpers(repo):
    repo = Path(repo).resolve()
    require({name: sha(repo / name) for name in SOURCE_PINS} == SOURCE_PINS, "Frozen numerical helper source mismatch")
    names = {relative.removeprefix("adaptive-gns/").removesuffix(".py").replace("/", "."): repo / relative
             for relative in SOURCE_PINS}
    for name, path in names.items():
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == path.resolve(), "Unrelated cached module: " + name)
    sys.path[:0] = [str(repo), str(repo / "adaptive-gns")]
    native = importlib.import_module("research.native_graph_rollout")
    for name, path in names.items():
        require(name in sys.modules and Path(sys.modules[name].__file__).resolve() == path.resolve(),
                "Imported helper is not pinned: " + name)
    require(tuple(native.POLICIES) == CORE_POLICIES, "Pinned policy contract differs")
    policy_path = Path(__file__).with_name("sand_graph_support_policy.py")
    require(sha(policy_path) == POLICY_SHA, "Physical policy source differs")
    spec = importlib.util.spec_from_file_location("_sand_graph_support_policy", policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy.NativeAdapter(native), importlib.import_module("research.full_training")


def configure_cuda(helpers, args):
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "Deterministic variant requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before process start")
    torch = helpers.torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    require(str(torch.__version__) == "2.13.0+cu129" and torch.version.cuda == "12.9", "Validated CUDA stack required")
    require(torch.cuda.is_available() and args.cuda_index < torch.cuda.device_count(), "Requested CUDA unavailable; no fallback")
    device = torch.device("cuda", args.cuda_index)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    props = torch.cuda.get_device_properties(device)
    require("GB200" in props.name, "This prepared wrapper is bounded to GB200")
    return device, {"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "numpy": helpers.np.__version__,
                    "scipy": helpers.scipy.__version__, "python": platform.python_version(), "platform": platform.platform(),
                    "device": str(device), "name": props.name, "uuid": str(props.uuid), "threads": args.threads,
                    "tf32": False, "amp": False, "compile": False, "ddp": False,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def tree_equal(torch, left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
                and bool(torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                     right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(torch, left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def check_checkpoint_contract(payload, args, admission, trainer_sha):
    require(admission.get("schema") == "adaptgns_sand_training_admission_v1"
            and admission.get("status") == "admitted" and admission.get("dataset") == "Sand",
            "Original root training admission required")
    config = payload.get("run_config", {})
    require(payload.get("format_version") == 2 and payload.get("cuda_sand_graph_support_schema") == TRAINING_SCHEMA
            and not any(key in payload for key in ("cuda_sand_training_schema", "full_training_schema", "graph_support_schema"))
            and config.get("schema") == TRAINING_SCHEMA and config.get("dataset") == "Sand"
            and payload.get("run_config_sha256") == canonical_hash(config)
            and integer(payload.get("completed_steps")) and payload.get("completed_steps") == args.checkpoint_updates
            and config.get("objective") == args.objective == "faithful" and config.get("arm") == args.arm and args.arm in ("base", "mix")
            and args.seed in (0, 1, 2) and integer(config.get("seed"))
            and config.get("seed") == args.seed and config.get("updates") == 100000,
            "Checkpoint is not the preselected Sand CUDA model/configuration/endpoint")
    require(payload.get("training_config") == {"loss": args.objective, "cuda_sand_graph_support_run": config,
                                               "completed_optimizer_updates": args.checkpoint_updates},
            "Checkpoint training metadata differs")
    require(args.checkpoint_updates in (512, 100000), "Only timing512 or final100k endpoints are accepted")
    require(config.get("checkpoint_every") == 10000 and config.get("log_every") == (1 if args.checkpoint_updates == 512 else 100),
            "Infrastructure/final checkpoint cadence differs")
    require(config.get("initialization") == "from scratch; paired seed across arms; empty Adam; no parent checkpoint",
            "From-scratch graph-support lineage required")
    require(config.get("research_protocol_sha256") == sha(args.training_protocol), "Training protocol identity differs")
    require(sha(args.trainer_source) == TRAINER_SHA, "Pinned graph-support training source differs")
    spec = importlib.util.spec_from_file_location("_sand_graph_support_checkpoint_contract", args.trainer_source)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    require(config.get("graph_exposure") == trainer.graph_exposure_config(args.arm), "Graph exposure recipe differs")
    trainer.validate_graph_history(payload.get("history"), args.checkpoint_updates, args.arm, args.seed)
    require(all(config.get("source_sha256", {}).get(path) == expected for path, expected in trainer.SOURCE_PINS.items()),
            "Checkpoint graph-support/core source pins differ")
    data = config.get("data", {})
    require(data.get("admission_sha256") == sha(args.train_admission)
            and data.get("manifest_sha256") == admission.get("manifest_sha256"), "Original training admission/manifest differs")
    require(data.get("metadata_sha256") == METADATA_SHA and data.get("frames_per_trajectory") == admission["frames_per_trajectory"]
            and data.get("particle_type_ids") == admission["particle_type_ids"]
            and data.get("converter_sha256") == admission["converter_sha256"]
            and data.get("structural_report_sha256") == admission["structural_report_sha256"]
            and data.get("source", {}).get("sha256") == SOURCES["train"][2], "Training/evaluation data lineage differs")
    require(config.get("source_sha256", {}).get("train_sand_graph_support_cuda.py") == trainer_sha == TRAINER_SHA, "Selected training entry hash differs")
    for path, expected in SOURCE_PINS.items():
        if path in ("research/full_training.py",) or path.startswith("adaptive-gns/"):
            require(config.get("source_sha256", {}).get(path) == expected, "Checkpoint frozen core source differs")
    require(config.get("batch_size") == 2 and config.get("history") == 6 and config.get("noise_std") == 6.7e-4
            and config.get("architecture") == {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2}
            and config.get("graph") == {"radius": .015, "backend": "scipy_host", "cap": 128, "self_candidates": True,
                                       "augmentation_probability": 0.0}, "Training graph/architecture/noise differs")
    require(config.get("optimizer") == {"name": "Adam", "initial_lr": 1e-4, "final_lr": 1e-5, "decay_updates": 100000,
            "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0, "foreach": False, "fused": False,
            "gradient_clipping": None}, "Training optimizer recipe differs")
    runtime = config.get("runtime", {})
    require(runtime.get("deterministic_algorithms") is True and runtime.get("deterministic_warn_only") is False
            and runtime.get("cublas_workspace_config") == ":4096:8" and runtime.get("tf32") is False
            and runtime.get("amp") is False and runtime.get("compile") is False and runtime.get("ddp") is False,
            "Selected checkpoint is not from the strict deterministic CUDA variant")
    return config


def prepare_model(helpers, args, metadata, admission, device):
    torch = helpers.torch
    torch.manual_seed(args.seed)
    with torch.cuda.device(device):
        torch.cuda.manual_seed(args.seed)
    model = helpers.build_simulator(metadata, 6.7e-4, 6.7e-4, device, connectivity_radius=.015,
        nmessage_passing_steps=10, uncertainty_parameterization="variance", variance_floor=1e-6,
        detach_variance_features=args.objective == "faithful", radius_backend="scipy_host").to(device)
    if args.model_kind == "initialized":
        model._training_config = {"loss": args.objective, "scope": "Sand feasibility only; untrained initialization", "seed": args.seed}
        path = args.output_dir / "initialized_model.pt"
        payload = {"format_version": 2, "state_dict": helpers.cpu_tree(model.state_dict()),
                   "simulator_config": helpers.cpu_tree(model._checkpoint_config), "training_config": model._training_config}
        with path.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        identity = {"kind": "initialized", "completed_updates": 0, "seed": args.seed, "objective": args.objective,
                    "saved_model": path.name, "saved_model_sha256": sha(path)}
    else:
        require(sha(args.checkpoint) == args.checkpoint_sha256, "Preselected checkpoint SHA256 differs before load")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = check_checkpoint_contract(payload, args, admission, sha(args.trainer_source))
        require(tree_equal(torch, payload.get("simulator_config"), helpers.cpu_tree(model._checkpoint_config)),
                "Selected simulator architecture/normalization differs from the admitted recipe")
        model.load_state_dict(payload["state_dict"], strict=True)
        require(tree_equal(torch, payload["state_dict"], helpers.cpu_tree(model.state_dict())), "Model state was not loaded exactly")
        model._training_config = payload["training_config"]
        require(sha(args.checkpoint) == args.checkpoint_sha256, "Checkpoint changed during load")
        identity = {"kind": "preselected_checkpoint", "completed_updates": args.checkpoint_updates, "seed": args.seed,
                    "objective": args.objective, "arm": args.arm, "checkpoint_sha256": args.checkpoint_sha256,
                    "run_config_sha256": canonical_hash(config), "training_source_sha256": sha(args.trainer_source)}
    require(helpers.tensors_are_finite(model.state_dict().values()), "Nonfinite initial/selected model state")
    return model.eval(), identity


@contextmanager
def deadline_alarm(seconds):
    if seconds <= 0:
        raise TimeoutError("Execution budget/cutoff reached before the next case")
    previous = signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Existing alarm would conflict with bounded execution")
    def timeout(_signal, _frame):
        raise TimeoutError("Sand feasibility budget/cutoff reached; partial attempt preserved, no retry")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def summarize_cases(rows, cases, horizon):
    summaries = {}
    expected = {case["source_index"] for case in cases}
    require(all(row["source_index"] in expected and row["policy"] in POLICIES for row in rows), "Unexpected case/policy result")
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        require(len({row["source_index"] for row in selected}) == len(selected), "Duplicate case/policy result")
        complete = [row for row in selected if row["status"] == "complete" and row["completed_steps"] == horizon]
        all_complete = len(selected) == len(complete) == len(cases)
        summaries[policy] = {"required_cases": len(cases), "recorded_cases": len(selected), "complete_cases": len(complete),
            "failed_cases": len(selected) - len(complete), "missing_cases": len(cases) - len(selected),
            "all_cases_full_horizon": all_complete,
            "equal_case_mean_rollout_mse": sum(row["mean_rollout_mse"] for row in complete) / len(cases) if all_complete else None,
            "max_full_rollout_wall_seconds_including_parity": max(row["synchronized_call_seconds"] for row in complete) if all_complete else None,
            "all_case_total_wall_seconds_including_parity": sum(row["synchronized_call_seconds"] for row in selected),
            "completed_case_timing_seconds": [row["synchronized_call_seconds"] for row in complete],
            "full_horizon_timing_forecast_admitted": False}
    return summaries


def run_cases(native, model, trajectories, manifest, cases, args, device, protocol_hash, started):
    rows, current = [], None
    horizon = manifest["records"][0]["positions"]["shape"][0] - 6
    trace_steps = tuple(sorted({step for step in (1, 10, 50, 200, horizon) if step <= horizon}))
    def save_result(state):
        final = getattr(args, "mode", None) == "full-rollout"
        atomic_json(args.output_dir / "result.json", {"schema": getattr(args, "result_schema", SCHEMA), "state": state, "protocol_sha256": protocol_hash,
            "horizon": horizon, "rows": rows, "summary": summarize_cases(rows, cases, horizon),
            "interpretation": ("All30 admitted final trajectories for one frozen arm/seed; complete-cohort paired analysis is separate"
                               if final else "Three source-size-selected feasibility cases; no scientific cohort/test result or automatic cost admission")})
    try:
        for case in cases:
            index = case["source_index"]
            positions, types = trajectories[index]
            for policy in POLICIES:
                current = {**case, "policy": policy, "horizon": horizon, "started_utc": utc_now()}
                atomic_json(args.output_dir / "status.json", {"state": "running", "pid": os.getpid(), "case": current, "committed_outcomes": len(rows)})
                remaining = min(args.max_seconds - (time.perf_counter() - started), (DEADLINE - datetime.now(timezone.utc)).total_seconds())
                with deadline_alarm(remaining):
                    native.full.synchronize(device)
                    call_started = time.perf_counter()
                    row, traces = native.rollout(model, positions, types, manifest["metadata"], policy, horizon,
                                                93000 + 1000 * args.seed + index, device, trace_steps=trace_steps)
                    native.full.synchronize(device)
                call_seconds = time.perf_counter() - call_started
                require(row["horizon"] == horizon and row["policy"] == policy, "Native returned a different horizon/policy")
                require(all(isinstance(value, native.np.ndarray) and not value.dtype.hasobject for value in traces.values()),
                        "Native traces must remain numeric arrays")
                path = args.output_dir / f"trajectory_{index:06d}_{policy}.npz"
                io_started = time.perf_counter()
                with path.with_suffix(".npz.tmp").open("xb") as stream:
                    native.np.savez_compressed(stream, **traces)
                    stream.flush()
                    os.fsync(stream.fileno())
                path.with_suffix(".npz.tmp").replace(path)
                row.update(**case, arm=args.arm, training_seed=args.seed, objective=args.objective,
                           protocol_sha256=protocol_hash, trace_file=path.name, trace_sha256=sha(path),
                           synchronized_call_seconds=call_seconds, trace_publication_seconds=time.perf_counter() - io_started,
                           requested_trace_steps=list(trace_steps),
                           mse_at_declared_trace_steps={str(step): row["mse_per_step"][step-1] if row["completed_steps"] >= step else None
                                                       for step in trace_steps})
                atomic_json(path.with_suffix(".json"), row)
                rows.append(row)
                save_result("partial")
                print(json.dumps({"case": case["size_group"], "source_index": index, "policy": policy,
                                  "status": row["status"], "completed_steps": row["completed_steps"], "horizon": horizon}), flush=True)
                if row.get("failure") and row["failure"].get("category") in ("native_parity_failure", "execution_error"):
                    raise RuntimeError("Native parity/implementation failure retained; review before any further inference")
        state = "complete" if all(row["status"] == "complete" for row in rows) else "complete_with_guard_failures"
        save_result(state)
        return rows, state
    except BaseException as error:
        atomic_json(args.output_dir / "failed_attempt.json", {"case": current, "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "committed_outcomes": len(rows), "at_utc": utc_now(),
            "unreturned_native_prefix_available": False, "retry_performed": False})
        save_result("error")
        raise


def diagnostic_cases(cases):
    return [{"schedule_index":k*5+j,"source_index":case["source_index"],"trajectory_id":case["trajectory_id"],
             "target_frame":target,"size_group":case["size_group"]}
            for k,case in enumerate(cases) for j,target in enumerate((7,85,163,241,319))]


def run_timing_diagnostics(native, helpers, model, trajectories, manifest, cases, args, device, protocol_hash, started):
    """Timing512 only: fixed source-size cases, fixed observed frames, no tuning."""
    path=Path(__file__).with_name("evaluate_sand_graph_support_final.py")
    require(sha(path)==DIAGNOSTIC_SHA,"Pinned clean/common-state diagnostic helper differs")
    spec=importlib.util.spec_from_file_location("_sand_graph_support_timing_diagnostics",path)
    diagnostic=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    require(args.split=="valid" and args.checkpoint_updates==512
            and all(row["positions"]["shape"][0]==320 for row in manifest["records"]),
            "Diagnostic timing requires admitted validation T320 and512 infrastructure endpoint")
    expected,rows,current=diagnostic_cases(cases),[],None
    try:
        for current in expected:
            remaining=min(args.max_seconds-(time.perf_counter()-started),(DEADLINE-datetime.now(timezone.utc)).total_seconds())
            atomic_json(args.output_dir/"status.json",{"state":"running","current":current,"committed_frames":len(rows)})
            with deadline_alarm(remaining),helpers.torch.no_grad():
                native.full.synchronize(device)
                call_started=time.perf_counter()
                positions,types=trajectories[current["source_index"]]
                if args.diagnostic_mode=="same-state":
                    row,arrays=diagnostic.same_state(native,model,positions,types,manifest["metadata"],current,"valid",args.seed,device)
                else:
                    row,arrays=diagnostic.clean_validation(native,helpers,model,positions,types,current,device)
                native.full.synchronize(device)
                call_seconds=time.perf_counter()-call_started
            require(all(isinstance(value,helpers.np.ndarray) and not value.dtype.hasobject for value in arrays.values()),
                    "Numeric-only diagnostic artifacts required")
            artifact=args.output_dir/f"trajectory_{current['source_index']:06d}_target_{current['target_frame']:03d}.npz"
            publication_started=time.perf_counter()
            with artifact.with_suffix(".npz.tmp").open("xb") as stream:
                helpers.np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
            artifact.with_suffix(".npz.tmp").replace(artifact)
            row.update(arm=args.arm,training_seed=args.seed,scope="512-update infrastructure diagnostic timing only",
                       artifact_file=artifact.name,artifact_sha256=sha(artifact),protocol_sha256=protocol_hash,
                       synchronized_diagnostic_call_seconds_including_parity_warmup_audits=call_seconds,
                       numeric_artifact_publication_seconds=time.perf_counter()-publication_started)
            atomic_json(artifact.with_suffix(".json"),row);rows.append(row)
            atomic_json(args.output_dir/"result.json",{"schema":SCHEMA,"scope":"non-test512 infrastructure timing only",
                "diagnostic_mode":args.diagnostic_mode,"protocol_sha256":protocol_hash,"rows":rows,
                "summary":diagnostic.summarize(expected,rows,args.diagnostic_mode),"expected_schedule":expected,
                "timing_forecast_admitted":False})
            if (row.get("failure") or {}).get("category") in ("native_parity_failure","execution_error"):
                raise RuntimeError("Diagnostic parity/implementation failure retained; further inference stopped")
        state="complete" if all(row["status"]=="complete" for row in rows) else "complete_with_guard_failures"
        return rows,state
    except BaseException as error:
        atomic_json(args.output_dir/"failed_attempt.json",{"current":current,"committed_frames":len(rows),
            "error_type":type(error).__name__,"error":str(error),"all_existing_outputs_retained":True,"retry_performed":False})
        raise


def check_timing_release(args):
    release = json.loads(args.timing_release.read_text())
    require(release.get("schema") == "adaptgns_sand_graph_support_timing_release_v1"
            and release.get("status") == "admitted_for_timing_only" and release.get("scientific_training_admission") is False,
            "Separate root timing-only graph-support release required")
    require(release.get("arm") == args.arm and release.get("seed") == args.seed
            and release.get("checkpoint_updates") == 512 and release.get("checkpoint_sha256") == args.checkpoint_sha256,
            "Timing release checkpoint/arm/seed differs")
    for key, path in (("trainer_sha256", args.trainer_source), ("evaluation_source_sha256", Path(__file__)),
                      ("protocol_sha256", args.protocol), ("training_protocol_sha256", args.training_protocol),
                      ("training_admission_sha256", args.train_admission)):
        require(release.get(key) == sha(path), "Timing release source/input differs: " + key)
    require(release.get("policies") == list(POLICIES), "All six timing policies required")
    require(release.get("diagnostic_mode") == args.diagnostic_mode, "Timing release diagnostic mode differs")


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": SCHEMA, "execution": False, "supported": "train/valid feasibility only",
            "model": "root-admitted faithful base/mix seed0/1/2 graph-support checkpoint at512 infrastructure updates only",
            "cases": "small/lower-median/large ranked by source particle count then source_index", "policies": POLICIES,
            "horizon": "full admitted T-6", "backend": "strict deterministic CUDA; CUBLAS_WORKSPACE_CONFIG=:4096:8", "test_execution": "disabled pending reviewed all-six-final gate"}, indent=2))
        return 0
    require(datetime.now(timezone.utc) < DEADLINE, "Research cutoff reached; no new work may start")
    check_timing_release(args)
    args.output_dir = args.output_dir.resolve()
    protected = [args.manifest.resolve().parent, (args.repo / "adaptive-gns").resolve()]
    if args.checkpoint:
        protected.append(args.checkpoint.resolve().parent)
    require(all(args.output_dir != path and path not in args.output_dir.parents and args.output_dir not in path.parents
                for path in protected), "Fresh output must be separate from source, dataset and checkpoint trees")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    started = time.perf_counter()
    process = {"pid": os.getpid(), "hostname": socket.gethostname(), "started_utc": utc_now(), "argv": sys.argv}
    atomic_json(args.output_dir / "status.json", {"state": "admitting", "process": process})
    try:
        inputs = {str(path.resolve()): sha(path) for path in
                  (args.manifest, args.admission, args.structural_report, args.protocol, args.train_admission, args.training_protocol, args.timing_release, Path(__file__), Path(__file__).with_name("sand_graph_support_policy.py"), Path(__file__).with_name("evaluate_sand_graph_support_final.py"))}
        manifest, admission, structural = [json.loads(path.read_text()) for path in (args.manifest, args.admission, args.structural_report)]
        require(sha(args.manifest) == admission["manifest_sha256"] and sha(args.structural_report) == admission["structural_report_sha256"],
                "Manifest/report bytes differ from explicit admission")
        frames = validate_contract(manifest, admission, structural, args.split)
        metadata_path = args.manifest.resolve().parent / "metadata.json"
        require(sha(metadata_path) == METADATA_SHA and json.loads(metadata_path.read_text()) == manifest["metadata"], "Metadata bytes/content differ")
        inputs[str(metadata_path)] = METADATA_SHA
        cases = select_cases(manifest["records"])
        native, helpers = load_helpers(args.repo)
        inputs.update({str((args.repo / path).resolve()): expected for path, expected in SOURCE_PINS.items()})
        trajectories = helpers.data_loader.load_manifest_data(args.manifest, verify_hashes=True)
        observed_types = set()
        for record, (positions, types) in zip(manifest["records"], trajectories):
            require(positions.shape[0] == frames and helpers.np.isfinite(positions).all(), "Nonfinite/mismatched numeric positions")
            observed_types.update(int(value) for value in helpers.np.unique(types))
            for field in ("positions", "particle_types"):
                path = helpers.data_loader._manifest_array_path(args.manifest.resolve().parent, record[field])
                require(path.stat().st_size == record[field]["size_bytes"], "Numeric array byte length differs")
                inputs[str(path)] = record[field]["sha256"]
        require(sorted(observed_types) == admission["particle_type_ids"], "Actual numeric type IDs differ from admission")
        if args.checkpoint:
            inputs[str(args.checkpoint.resolve())] = args.checkpoint_sha256
            inputs[str(args.trainer_source.resolve())] = sha(args.trainer_source)
        device, runtime = configure_cuda(helpers, args)
        training_admission = json.loads(args.train_admission.read_text())
        model, identity = prepare_model(helpers, args, manifest["metadata"], training_admission, device)
        if args.model_kind == "initialized":
            inputs[str(args.output_dir / identity["saved_model"])] = identity["saved_model_sha256"]
        protocol = {"schema": SCHEMA, "scope": "bounded non-test feasibility; not scientific training admission", "diagnostic_mode": args.diagnostic_mode,
            "model": identity, "split": args.split, "frames_per_trajectory": frames, "horizon": frames - 6,
            "cases": cases, "case_selection": "sort all source trajectories by (particles,source_index); ranks0,(n-1)//2,n-1",
            "policies": list(POLICIES), "policy_order": "fixed; operational timing, not causal speed comparison",
            "random_seed": ("93000+1000*model_seed+source_index" if args.diagnostic_mode == "full-rollout"
                            else "[20261006,93000,training_seed,0,source_index,target_frame]; reset each same-state call; arm omitted"
                            if args.diagnostic_mode == "same-state" else "none; clean native-base validation with zero noise"),
            "graph": "pinned native capped128/self base prefix + symmetric uncapped annulus; factor1.267; floor25%",
            "runtime": runtime, "process": process, "input_files_sha256": inputs,
            "setup_seconds": time.perf_counter() - started, "max_seconds": args.max_seconds, "cutoff_utc": DEADLINE.isoformat(),
            "failed_prefix_timing_is_full_rollout_estimate": False, "test_accessed": False}
        atomic_json(args.output_dir / "protocol.json", protocol)
        protocol_hash = sha(args.output_dir / "protocol.json")
        if args.diagnostic_mode == "full-rollout":
            rows, state = run_cases(native, model, trajectories, manifest, cases, args, device, protocol_hash, started)
        else:
            rows, state = run_timing_diagnostics(native, helpers, model, trajectories, manifest, cases, args, device, protocol_hash, started)
        for path, expected in inputs.items():
            require(sha(path) == expected, "Pinned input/source changed during execution: " + path)
        atomic_json(args.output_dir / "status.json", {"state": state, "process": process, "completed_utc": utc_now(),
            "committed_outcomes": len(rows), "all_inputs_reverified": True, "result_sha256": sha(args.output_dir / "result.json"),
            "total_execution_seconds": time.perf_counter() - started, "forecast_admitted": False})
        return 0
    except BaseException as error:
        atomic_json(args.output_dir / "status.json", {"state": "deadline_stopped" if isinstance(error, TimeoutError) else "error",
            "process": process, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "at_utc": utc_now(), "all_outputs_retained": True, "retry_performed": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
