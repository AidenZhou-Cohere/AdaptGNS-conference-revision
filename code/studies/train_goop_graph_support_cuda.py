#!/usr/bin/env python3
"""Numerical particle-simulation helpers. Use code/evaluate.py for evaluation."""
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
SCHEMA = "adaptgns_goop_graph_support_cuda_training_v1"
ADMISSION = "adaptgns_goop_training_admission_v1"
METADATA_SHA = "565d6e13be91be6a6b0fbc31a9aa19ed70a5505228411d0928288fcf874ca3dd"
SOURCE_PINS = {
    "research/faithful_graph_support.py": "67515a25882a86c53e8c1152a8c5e35ac32e1d43667849276cd03068e1845d09",
    "research/graph_convention_bridge.py": "b4a1c2c2ca29a7134f59b94cf85e71c9dcd9235e2c59d59637612f0f62e2d6f7",
    "research/full_rollout.py": "70068dc81aae9190884f2e9888bdec82e9cf80c8f65cef8303011d7fd0c4eee8",
    "research/full_same_state.py": "2efd2cf53394b7214f59b322d256fcb64f2e500b4cfc7ff59ae0cbda7110cfef",
    "research/budget_graph.py": "46df0b5608a4f586a3dbe9653f84c79f6e70e80a2d3bd258e9339406e33a7d59",
    "research/full_training.py": "64c925ef198dc2fd032925abdf2f3b5af87d0ce8d29ca9955d21777f899d3a53",
    "adaptive-gns/gns/data_loader.py": "287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef",
    "adaptive-gns/gns/device_utils.py": "324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326",
    "adaptive-gns/gns/graph_network.py": "af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91",
    "adaptive-gns/gns/learned_simulator.py": "216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf",
    "adaptive-gns/gns/losses.py": "94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5",
    "adaptive-gns/gns/model_io.py": "06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e",
}
MANIFEST_SHA = "5ef43daf9bac961a69bd460a539891624b79c8a13f80ca851e85802197982256"
CONVERTER_SHA = "fa4b7d883d7c360500fc3c603c9cd14538c916f1ddd7435aedfc8ec2420f2985"
READER_SHA = "52822d4373097dd0341f47f58fa8121b443160022540edaee0b9fcb831ba043b"
ACQUISITION_SHA = "9e4b106324b1e7c5add4404565eca888eaa0ebd3c233622308c113e88c5e6a8d"
CONTEXT_SEMANTICS_SHA = "c81ae2f1565e61542bcc406c4a9d71b67135620857ad62292eaef0e29b407de9"
AUXILIARY_REPORT_SHA = "3278055e119c482c5620ff66b2c4e4e6c09361240155cc77a7ad98dbaa13d8d1"
OFFICIAL_COMMIT = "f5de0ede8430809180254ee957abf36ed62579ef"
CONTEXT_SOURCE_PINS = {
    "README.md": "9db737244fccd3ea37d525ba244b7ea2bddd300f420a346bbfdd2893c419a69a",
    "reading_utils.py": "5868022f76eaf15b2626125bdaa3c973ccaf2dfc0b0ec50d9e05d0c3d973e149",
    "train.py": "44b0d7759b3af37cb9c0dc442c13440302823413091e0f1b572bc7f4b95f6d60",
    "learned_simulator.py": "bda3d60fcaf8a7a1a963fe6e87790e38955c38602da1eeee5d214ef2f4518da9",
}
OFFICIAL_TRAIN_SOURCE = {
    "family": "official_gns_tfrecord", "dataset": "Goop", "file": "train.tfrecord",
    "size_bytes": 3358535518, "sha256": "8a0b0cfe3ef56f533abf14d963d5cf632235aa3570cf61a4b42ab655a15cae59",
    "generation": "1599154403717337", "crc32c_base64": "OrpMIQ==", "CRC_verified": True,
    "record_count": 1000, "acquisition_report_sha256": ACQUISITION_SHA,
}
AUXILIARY_USE = "Preserved source auxiliary data; semantics unreviewed and model admission required"
OMISSION_REASON = "unused_by_released_goop_parser_model_contract_without_context_mean"
RECORD_KEYS = {"id", "source_index", "source_key", "source_offset_bytes", "record_payload_bytes",
               "record_payload_sha256", "positions", "particle_types", "step_context", "trajectory_content_sha256"}
ARRAY_KEYS = {"path", "shape", "dtype", "size_bytes", "sha256"}
MANIFEST_KEYS = {"format", "version", "split", "dataset", "source", "metadata",
                 "metadata_sha256", "record_count", "records", "converter_sha256", "reader_sha256"}


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
            "requires": "explicit --execute, admitted official Goop numeric train manifest, protocol, base/mix arm, seed0/1/2 and updates100000",
            "architecture": "128 width, 10 message-passing blocks, 2 MLP layers; batch 2; history 6; 2D",
            "graph": "native strict radius .015/cap128/self prefix; mix independently exposes each example with probability.5 to an exact25% annulus append",
            "recipe": "noise 6.7e-4; original noise-adjusted normalization; faithful only; Adam 1e-4 to 1e-5 over 100000 updates",
            "precision": "float32; strict deterministic algorithms; CUBLAS_WORKSPACE_CONFIG=:4096:8; no AMP/TF32/compile/DDP; Adam foreach=False, fused=False",
            "lineage": "from-scratch faithful base/mix x seeds0/1/2; no old checkpoints; independent schema; no launch admission implied",
            "source_pins": SOURCE_PINS,
            "limits": "exact official train: 1000 trajectories, T401, type7; stored step_context preserved but excluded by official metadata/parser contract; no validation/test evaluation or cross-device bitwise guarantee"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    for name in ("repo", "train-manifest", "admission", "structural-report", "acquisition-report", "context-semantics", "auxiliary-report", "protocol", "output-dir"):
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
        required = ("repo", "train_manifest", "admission", "structural_report", "acquisition_report", "context_semantics", "auxiliary_report", "protocol",
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
    if (admission.get("schema") != ADMISSION or admission.get("status") != "admitted"
            or admission.get("dataset") != "Goop" or admission.get("issued_by") != "root"):
        raise ValueError("Root-issued Goop admission is required")
    expected_pins = {"manifest_sha256": MANIFEST_SHA, "metadata_sha256": METADATA_SHA,
                     "converter_sha256": CONVERTER_SHA, "reader_sha256": READER_SHA,
                     "acquisition_report_sha256": ACQUISITION_SHA, "context_semantics_sha256": CONTEXT_SEMANTICS_SHA,
                     "auxiliary_report_sha256": AUXILIARY_REPORT_SHA}
    if any(admission.get(key) != value for key, value in expected_pins.items()):
        raise ValueError("Root admission differs from the exact reviewed Goop source contract")
    if not all(digest_string(admission.get(key)) for key in ("structural_report_sha256", "auxiliary_report_sha256")):
        raise ValueError("Root admission must bind structural and auxiliary reports")
    policy = admission.get("auxiliary_policy")
    if policy != {"name": "step_context", "handling": "preserved_and_excluded_from_model_inputs",
                  "reason": OMISSION_REASON, "official_commit": OFFICIAL_COMMIT}:
        raise ValueError("Explicit source-supported Goop context omission is required")
    if (admission.get("frames_per_trajectory") != 401 or admission.get("record_count") != 1000
            or admission.get("particle_type_ids") != [7] or admission.get("position_dtype") != "<f4"
            or admission.get("particle_type_dtype") != "<i8"):
        raise ValueError("This entry admits only the exact T401/type7/1000-record Goop training source")
    if (set(manifest) != MANIFEST_KEYS or manifest.get("format") != "gns-trajectory-manifest"
            or manifest.get("version") != 1 or manifest.get("split") != "train" or manifest.get("dataset") != "Goop"):
        raise ValueError("Only the exact official Goop version-1 training manifest is accepted")
    if manifest.get("source") != OFFICIAL_TRAIN_SOURCE:
        raise ValueError("Official Goop source identity, generation, CRC or complete count differs")
    if any(manifest.get(key) != value for key, value in expected_pins.items()
           if key in ("metadata_sha256", "converter_sha256", "reader_sha256")):
        raise ValueError("Metadata/converter/reader identity differs from admission")
    metadata = manifest.get("metadata", {})
    if (metadata.get("dim") != 2 or metadata.get("sequence_length") != 400
            or metadata.get("default_connectivity_radius") != .015 or metadata.get("dt") != .0025
            or metadata.get("bounds") != [[.1, .9], [.1, .9]]
            or "context_mean" in metadata or "context_std" in metadata):
        raise ValueError("Goop geometry/time/context metadata differs")
    records = manifest.get("records", [])
    if len(records) != 1000 or manifest.get("record_count") != 1000:
        raise ValueError("Complete 1000-record Goop training source required")
    contents, payloads, next_offset = set(), set(), 0
    for index, record in enumerate(records):
        if set(record) != RECORD_KEYS:
            raise ValueError("Unexpected/missing record fields; new auxiliary semantics require review")
        if (record["id"] != f"train:{index:06d}" or not integer(record["source_index"])
                or record["source_index"] != index or record["source_key"] != [index]
                or record["source_offset_bytes"] != next_offset or not integer(record["record_payload_bytes"], 1)
                or not digest_string(record["record_payload_sha256"]) or record["record_payload_sha256"] in payloads):
            raise ValueError("Source ordering, keys, offsets or payload identity differs")
        next_offset += record["record_payload_bytes"] + 16
        payloads.add(record["record_payload_sha256"])
        p, t, auxiliary = (record[key] for key in ("positions", "particle_types", "step_context"))
        for name, array, prefix in (("positions", p, "position"), ("particle_types", t, "type"),
                                    ("step_context", auxiliary, "step_context")):
            keys = ARRAY_KEYS | ({"use"} if name == "step_context" else set())
            if (set(array) != keys or not digest_string(array["sha256"]) or not integer(array["size_bytes"], 1)
                    or array["path"] != f"train/{prefix}_{index:06d}.npy"):
                raise ValueError("Exact numeric hash/shape/dtype/size/path descriptors required")
        shape = p["shape"]
        if (not isinstance(shape, list) or len(shape) != 3 or shape[0] != 401 or shape[2] != 2
                or not integer(shape[1], 1) or p["dtype"] != "<f4"
                or t["shape"] != [shape[1]] or t["dtype"] != "<i8"):
            raise ValueError("Preserved float32 positions and full int64 type vector are required")
        if (auxiliary["shape"] != [401, 1] or auxiliary["dtype"] != "<f4" or auxiliary["use"] != AUXILIARY_USE):
            raise ValueError("Preserved Goop step_context descriptor differs")
        content = hashlib.sha256((p["sha256"] + ":" + t["sha256"]).encode()).hexdigest()
        if record["trajectory_content_sha256"] != content or content in contents:
            raise ValueError("Duplicate trajectory or inconsistent content hash")
        contents.add(content)
    if next_offset != OFFICIAL_TRAIN_SOURCE["size_bytes"]:
        raise ValueError("Payload offsets do not cover the complete official TFRecord")


def verify_goop_evidence(args, manifest, admission):
    expected = {"train_manifest": MANIFEST_SHA, "structural_report": admission["structural_report_sha256"],
                "acquisition_report": ACQUISITION_SHA, "context_semantics": CONTEXT_SEMANTICS_SHA,
                "auxiliary_report": admission["auxiliary_report_sha256"]}
    for name, value in expected.items():
        if sha(getattr(args, name)) != value:
            raise ValueError("Goop evidence bytes differ: " + name)
    context_root = args.context_semantics.resolve().parent / "goop_context_semantics_sources"
    for name, value in CONTEXT_SOURCE_PINS.items():
        if sha(context_root / name) != value:
            raise ValueError("Reviewed official context source differs: " + name)
    structural = json.loads(args.structural_report.read_text())
    train_structure = structural.get("splits", {}).get("train", {})
    if (structural.get("schema") != "official_goop_numeric_preparation_v1"
            or structural.get("status") != "complete_structural_only" or structural.get("test_accessed") is not False
            or structural.get("dataset") != "Goop" or structural.get("source_family") != "official_gns_tfrecord"
            or structural.get("metadata_sha256") != METADATA_SHA or structural.get("reader_sha256") != READER_SHA
            or structural.get("wrapper_sha256") != CONVERTER_SHA or structural.get("acquisition_report_sha256") != ACQUISITION_SHA
            or train_structure.get("manifest_sha256") != MANIFEST_SHA or train_structure.get("record_count") != 1000
            or train_structure.get("frame_lengths") != [401] or train_structure.get("particle_type_ids") != [7]
            or train_structure.get("forecast_horizons_after_six_frames") != [395]
            or any(train_structure.get(key) is not True for key in ("source_EOF_SHA256_verified",
                "whole_object_crc32c_verified", "TFRecord_length_and_payload_CRC32C_verified",
                "all_match_official_parser_frame_count", "all_source_array_bytes_preserved_exact"))):
        raise ValueError("Complete reviewed Goop structural evidence required")
    census = json.loads(args.auxiliary_report.read_text())
    split = census.get("splits", {}).get("train", {})
    if (census.get("schema") != "adaptgns_goop_auxiliary_census_v1"
            or census.get("status") != "all_preserved_auxiliary_bytes_verified" or census.get("test_accessed") is not False
            or split.get("manifest_sha256") != MANIFEST_SHA or split.get("record_count") != 1000
            or any(split.get(key) is not True for key in ("context_mean_absent", "context_std_absent",
                "all_context_descriptors_and_bytes_verified")) or split.get("all_context_values_nan") is not False
            or len(split.get("records", [])) != 1000):
        raise ValueError("Complete source-bound Goop auxiliary census required")
    for record, observed in zip(manifest["records"], split["records"]):
        descriptor = record["step_context"]
        if (observed.get("id") != record["id"] or observed.get("source_index") != record["source_index"]
                or observed.get("sha256") != descriptor["sha256"] or observed.get("shape") != [401, 1]
                or observed.get("dtype") != "<f4" or observed.get("elements") != 401 or observed.get("nan_count") != 1
                or observed.get("finite_count") != 400
                or any(observed.get(key) != 0 for key in ("positive_infinity_count", "negative_infinity_count"))
                or observed.get("unique_float32_bits_hex") != ["00000000", "7fc00000"]):
            raise ValueError("Auxiliary census row differs from preserved source descriptor")
    return split["records"]


def verify_goop_auxiliary_arrays(args, helpers, manifest, census_rows):
    for record, observed in zip(manifest["records"], census_rows):
        descriptor = record["step_context"]
        path = helpers.data_loader._manifest_array_path(args.train_manifest.resolve().parent, descriptor)
        if path.stat().st_size != descriptor["size_bytes"] or sha(path) != descriptor["sha256"]:
            raise ValueError("Preserved excluded Goop context bytes differ")
        values = helpers.np.load(path, allow_pickle=False)
        bits = [f"{int(value):08x}" for value in helpers.np.unique(values.view("<u4"))]
        if (values.dtype.str != "<f4" or list(values.shape) != [401, 1]
                or int(helpers.np.isnan(values).sum()) != 1 or int(helpers.np.isfinite(values).sum()) != 400
                or bits != observed["unique_float32_bits_hex"]):
            raise ValueError("Preserved excluded Goop context representation or bits differ")


def load_admitted_dataset(args, helpers):
    admission = json.loads(args.admission.read_text())
    manifest = json.loads(args.train_manifest.read_text())
    validate_manifest_contract(manifest, admission)
    census_rows = verify_goop_evidence(args, manifest, admission)
    metadata_path = args.train_manifest.resolve().parent / "metadata.json"
    if sha(metadata_path) != METADATA_SHA:
        raise ValueError("Actual Goop metadata bytes differ")
    metadata = json.loads(metadata_path.read_text())
    if metadata != manifest.get("metadata"):
        raise ValueError("Manifest metadata mismatch")
    verify_goop_auxiliary_arrays(args, helpers, manifest, census_rows)
    dataset, _, info = helpers.load_manifest_dataset(args.train_manifest, "train")
    for record in manifest["records"]:
        for name in ("positions", "particle_types"):
            path = helpers.data_loader._manifest_array_path(args.train_manifest.resolve().parent, record[name])
            if path.stat().st_size != record[name]["size_bytes"]:
                raise ValueError("Actual numeric array byte length differs from its manifest")
    observed_types = set()
    for positions, particle_types in dataset._data:
        if positions.shape[0] != 401 or positions.shape[2] != 2 or positions.dtype.str != "<f4":
            raise ValueError("Loaded frame count/dimension/dtype differs")
        if particle_types.shape != (positions.shape[1],) or particle_types.dtype.str != "<i8":
            raise ValueError("Exact per-particle Goop type vector required")
        if not helpers.np.isfinite(positions).all():
            raise ValueError("Nonfinite source trajectory")
        observed_types.update(int(value) for value in helpers.np.unique(particle_types))
    if sorted(observed_types) != [7] or len(dataset._data) != 1000 or len(dataset) != 395000:
        raise ValueError("Observed Goop types/counts or full 395000-history coverage differs")
    info.update({"frames_per_trajectory": 401, "particle_type_ids": [7], "forecast_horizon_after_six_frames": 395,
                 "admission_sha256": sha(args.admission), "structural_report_sha256": sha(args.structural_report),
                 "converter_sha256": CONVERTER_SHA, "reader_sha256": READER_SHA, "metadata_sha256": METADATA_SHA,
                 "acquisition_report_sha256": ACQUISITION_SHA, "context_semantics_sha256": CONTEXT_SEMANTICS_SHA,
                 "context_source_sha256": CONTEXT_SOURCE_PINS, "auxiliary_report_sha256": sha(args.auxiliary_report),
                 "auxiliary_policy": admission["auxiliary_policy"]})
    return dataset, metadata, info


def reverify_goop_evidence_and_auxiliaries(args, helpers):
    admission = json.loads(args.admission.read_text())
    manifest = json.loads(args.train_manifest.read_text())
    validate_manifest_contract(manifest, admission)
    census_rows = verify_goop_evidence(args, manifest, admission)
    verify_goop_auxiliary_arrays(args, helpers, manifest, census_rows)


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
    return {"format_version": 2, "cuda_goop_graph_support_schema": SCHEMA,
            "state_dict": helpers.cpu_tree(model.state_dict()),
            "simulator_config": helpers.cpu_tree(model._checkpoint_config),
            "training_config": {"loss": config["objective"], "cuda_goop_graph_support_run": config,
                                "completed_optimizer_updates": completed},
            "optimizer_state": helpers.cpu_tree(optimizer.state_dict()),
            "run_config": config, "run_config_sha256": config_hash(config),
            "completed_steps": completed, "history": history,
            "rng_states": helpers.cpu_tree(capture_rng(helpers.torch, device))}


def restore_payload(helpers, payload, model, optimizer, config, device):
    torch = helpers.torch
    completed = payload.get("completed_steps")
    if (payload.get("format_version") != 2 or payload.get("cuda_goop_graph_support_schema") != SCHEMA
            or any(key in payload for key in ("cuda_goop_training_schema", "cuda_sand_graph_support_schema", "cuda_sand_training_schema", "full_training_schema", "graph_support_schema"))
            or payload.get("run_config_sha256") != config_hash(config) or payload.get("run_config") != config
            or not integer(completed) or completed > config["updates"]
            or payload.get("training_config") != {"loss": config["objective"], "cuda_goop_graph_support_run": config,
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
    config = {"schema": SCHEMA, "dataset": "Goop", "objective": args.objective,
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
              "source_sha256": {**SOURCE_PINS, "train_goop_graph_support_cuda.py": sha(__file__)}, "runtime": runtime,
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
    model._training_config = {"loss": args.objective, "cuda_goop_graph_support_run": config}
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
            path = Path(__file__) if relative == "train_goop_graph_support_cuda.py" else args.repo / relative
            if sha(path) != expected:
                raise ValueError("Source changed during graph-support training: " + relative)
        for path, expected in ((args.train_manifest, data_info["manifest_sha256"]),
                               (args.admission, data_info["admission_sha256"]),
                               (args.structural_report, data_info["structural_report_sha256"]),
                               (args.protocol, config["research_protocol_sha256"])):
            if sha(path) != expected:
                raise ValueError("Input changed during graph-support training: " + str(path))
        helpers.data_loader.load_manifest_data(args.train_manifest, verify_hashes=True)
        reverify_goop_evidence_and_auxiliaries(args, helpers)
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
    raise SystemExit("Use code/evaluate.py or code/portable/ entry points.")
