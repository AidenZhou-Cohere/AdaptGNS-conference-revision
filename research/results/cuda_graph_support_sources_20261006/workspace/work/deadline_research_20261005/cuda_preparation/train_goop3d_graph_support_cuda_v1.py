#!/usr/bin/env python3
"""Separate prospective-endpoint Goop-3D graph-support trainer; describe by default.

No training endpoint, source count/type census or launch is selected here.
Root admission binds the complete converted training source and one endpoint
for paired base/mix x seeds0/1/2 before execution. No probe promotion, tuning,
validation/test access, automatic resume, AMP, TF32, compile or DDP.
Checkpoint/Adam/RNG machinery is derived from the unchanged pinned Goop2D trainer.
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
from types import SimpleNamespace

sys.dont_write_bytecode = True
SCHEMA = "adaptgns_goop3d_graph_support_cuda_training_v1"
ADMISSION = "adaptgns_goop3d_training_admission_v1"
METADATA_SHA = "727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55"
GRAPH_SHA = "7d43fe7d06ac450b6b9031181902cfc6d3d9754af3a26e17e96fd46c4a1bf64f"
BASE_TRAINER_SHA = "dd9ef01a16116f04a578bbfe9ed01774e66fdc08d60b913072a33ad211b319c1"
CONVERTER_SHA = "49dece28bf4494591379b8a667e5366b5bdf1609c4dc0757ceb31664c28f9b74"
READER_SHA = "ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33"
CONTEXT_SEMANTICS_SHA = "5eb6818ae2699c57e62e80f63724248e8573e4536195eb471df0c647122fb1a5"
AUXILIARY_VALIDATOR_SHA = "1b5a8c24fde395b2633ee197557fde3e5bf30ca74908a77f1fb0e011b85e41fb"
FRAMES, DIMENSION, RADIUS = 301, 3, .025
OFFICIAL_TRAIN = {"family": "official_gns_tfrecord", "dataset": "Goop-3D", "file": "train.tfrecord",
                  "size_bytes": 27448425232, "generation": "1599159646798459", "crc32c_base64": "gBeM1w==", "CRC_verified": True}
ENDPOINT_BASIS = "prospective_compute_budget_no_validation_or_test_selection"
PAIRED_SCHEDULE = [{"arm": arm, "seed": seed} for seed in (0, 1, 2) for arm in ("base", "mix")]
SOURCE_PINS = {
    "research/full_training.py": "c6bec98cb8200ef74c3d5d30309dba858c0dbef49f3380a0bc8762683c04e60d",
    "adaptive-gns/gns/data_loader.py": "287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef",
    "adaptive-gns/gns/device_utils.py": "324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326",
    "adaptive-gns/gns/graph_network.py": "af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91",
    "adaptive-gns/gns/learned_simulator.py": "216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf",
    "adaptive-gns/gns/losses.py": "94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5",
    "adaptive-gns/gns/model_io.py": "06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e",
}
OFFICIAL_COMMIT = "f5de0ede8430809180254ee957abf36ed62579ef"
CONTEXT_SOURCE_PINS = {
    "README.md": "9db737244fccd3ea37d525ba244b7ea2bddd300f420a346bbfdd2893c419a69a",
    "reading_utils.py": "5868022f76eaf15b2626125bdaa3c973ccaf2dfc0b0ec50d9e05d0c3d973e149",
    "train.py": "44b0d7759b3af37cb9c0dc442c13440302823413091e0f1b572bc7f4b95f6d60",
    "learned_simulator.py": "bda3d60fcaf8a7a1a963fe6e87790e38955c38602da1eeee5d214ef2f4518da9",
}
AUXILIARY_USE = "Preserved source auxiliary data; semantics unreviewed and model admission required"
OMISSION_REASON = "unused_by_released_parser_model_contract_without_context_mean"
RECORD_KEYS = {"id", "source_index", "source_key", "source_offset_bytes", "record_payload_bytes",
               "record_payload_sha256", "positions", "particle_types", "trajectory_content_sha256"}
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
            "requires": "explicit --execute, complete root Goop-3D admission, prospective --updates, protocol, base/mix arm and seed0/1/2",
            "architecture": "D3;37 node features;4 edge features;128 width;10 message-passing blocks;2-layer MLPs;batch2;history6",
            "graph": "native strict radius.025/cap128/self-candidate prefix; per-example mix probability.5; exact25% symmetric annulus append",
            "recipe": "faithful D3 loss; noise6.7e-4; noise-adjusted metadata normalization; Adam1e-4 to1e-5 over the prospectively admitted endpoint",
            "counts_and_types": "must match actual complete source evidence; no hardcoded1000/30/type7",
            "source_pins": SOURCE_PINS, "graph_adapter_sha256": GRAPH_SHA, "base_trainer_sha256": BASE_TRAINER_SHA,
            "limits": "no scientific/compute admission;2M expanded-radius candidate guard per example and5M batch-edge guard inherited from reviewed3D graph adapter"}


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        required = ("repo", "train_manifest", "admission", "structural_report", "acquisition_report", "context_semantics", "auxiliary_report", "protocol",
                    "output_dir", "arm", "objective", "seed", "updates")
        if any(getattr(args, name) is None for name in required):
            parser.error("Execution requires " + ", ".join("--" + name.replace("_", "-") for name in required))
        if args.seed not in (0, 1, 2) or args.updates < 2:
            parser.error("Seed0/1/2 and an explicit prospective endpoint>=2 updates required")
    if (args.cuda_index < 0 or not 1 <= args.threads <= 16
            or min(args.checkpoint_every, args.log_every) < 1):
        parser.error("Invalid CUDA index, thread count or logging/checkpoint interval")
    if args.clear_stale_lock and not args.resume:
        parser.error("Dead-lock recovery requires --resume; never restart over a previous run")
    return args


def load_helpers(repo):
    repo = Path(repo).resolve()
    if {name: sha(repo / name) for name in SOURCE_PINS} != SOURCE_PINS:
        raise ValueError("Frozen common helper/core source mismatch")
    names = {relative.removeprefix("adaptive-gns/").removesuffix(".py").replace("/", "."): repo / relative for relative in SOURCE_PINS}
    for name, path in names.items():
        if name in sys.modules and Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise ValueError("Unrelated cached module: " + name)
    sys.path[:0] = [str(repo), str(repo / "adaptive-gns")]
    original = importlib.import_module("research.full_training")
    for name, path in names.items():
        if name not in sys.modules or Path(sys.modules[name].__file__).resolve() != path.resolve():
            raise ValueError("Imported common helper is not pinned: " + name)
    graph_path = Path(__file__).with_name("goop3d_graph_support.py")
    if sha(graph_path) != GRAPH_SHA:
        raise ValueError("Reviewed3D graph adapter differs")
    spec = importlib.util.spec_from_file_location("_goop3d_scientific_graph", graph_path)
    support = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = support
    spec.loader.exec_module(support)
    helpers = SimpleNamespace(**vars(original))
    return helpers, support


def unpack_batch(helpers, examples):
    features, labels = helpers.data_loader.collate_fn(examples)
    if len(features) != 3:
        raise ValueError("Only position/type/count model inputs admitted by official context contract")
    position, types, counts = features
    if position.shape[1:] != (6, 3) or labels.shape != (len(position), 3):
        raise ValueError("Expected six-frame3D histories and3D targets")
    return position, types, counts, labels


def learning_rate(step, endpoint):
    if not integer(endpoint, 2) or not integer(step) or step >= endpoint:
        raise ValueError("Prospectively declared endpoint and in-range update required")
    return 1e-4 * (.1 ** (step / (endpoint - 1)))


def validate_prospective_endpoint(args, admission):
    if (admission.get("schema") != ADMISSION or admission.get("status") != "admitted"
            or admission.get("issued_by") != "root" or admission.get("dataset") != "Goop-3D"
            or admission.get("prospective_endpoint_updates") != args.updates
            or not integer(args.updates, 2) or admission.get("endpoint_selection_basis") != ENDPOINT_BASIS
            or admission.get("paired_arm_seed_schedule") != PAIRED_SCHEDULE
            or admission.get("checkpoint_every") != args.checkpoint_every or admission.get("log_every") != args.log_every
            or admission.get("trainer_sha256") != sha(__file__) or admission.get("graph_adapter_sha256") != GRAPH_SHA
            or admission.get("protocol_sha256") != sha(args.protocol)
            or not isinstance(admission.get("selection_evidence_sha256"), list) or not admission["selection_evidence_sha256"]
            or not all(digest_string(v) for v in admission["selection_evidence_sha256"])
            or admission.get("initialization") != "fresh_no_probe_or_parent_checkpoint"):
        raise ValueError("Prospective root endpoint/protocol/paired-cohort admission required")


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
            or admission.get("dataset") != "Goop-3D" or admission.get("issued_by") != "root"):
        raise ValueError("Root-issued complete Goop-3D admission required")
    fixed = {"metadata_sha256": METADATA_SHA, "converter_sha256": CONVERTER_SHA, "reader_sha256": READER_SHA,
             "context_semantics_sha256": CONTEXT_SEMANTICS_SHA, "auxiliary_validator_sha256": AUXILIARY_VALIDATOR_SHA}
    if any(admission.get(k) != v for k, v in fixed.items()) or not all(digest_string(admission.get(k)) for k in
            ("manifest_sha256", "source_sha256", "acquisition_report_sha256", "structural_report_sha256", "auxiliary_report_sha256")):
        raise ValueError("Complete prospective3D source/evidence byte bindings required")
    count, types = admission.get("record_count"), admission.get("particle_type_ids")
    if (not integer(count, 1) or not isinstance(types, list) or not types or types != sorted(set(types))
            or not all(type(v) is int and 0 <= v <= 8 for v in types) or not any(v != 3 for v in types)
            or admission.get("frames_per_trajectory") != FRAMES or admission.get("dimension") != DIMENSION
            or admission.get("eligible_six_frame_histories") != count * (FRAMES - 6)
            or admission.get("position_dtype") != "<f4" or admission.get("particle_type_dtype") != "<i8"):
        raise ValueError("Actual complete3D count/type/history census required")
    if admission.get("auxiliary_policy") != {"name": "step_context", "handling": "preserved_and_excluded_from_model_inputs",
            "reason": OMISSION_REASON, "official_commit": OFFICIAL_COMMIT}:
        raise ValueError("Explicit official-parser-supported auxiliary omission required")
    if (set(manifest) != MANIFEST_KEYS or manifest.get("format") != "gns-trajectory-manifest" or manifest.get("version") != 1
            or manifest.get("split") != "train" or manifest.get("dataset") != "Goop-3D"):
        raise ValueError("Only the exact official3D version1 training manifest is accepted")
    expected_source = {**OFFICIAL_TRAIN, "sha256": admission["source_sha256"], "record_count": count,
                       "acquisition_report_sha256": admission["acquisition_report_sha256"]}
    if manifest.get("source") != expected_source:
        raise ValueError("Complete official3D source identity/generation/CRC/count differs")
    if any(manifest.get(k) != v for k, v in fixed.items() if k in ("metadata_sha256", "converter_sha256", "reader_sha256")):
        raise ValueError("Metadata/converter/reader differs")
    metadata = manifest.get("metadata", {})
    if (metadata.get("dim") != 3 or metadata.get("sequence_length") != 300
            or metadata.get("default_connectivity_radius") != .025
            or "context_mean" in metadata or "context_std" in metadata):
        raise ValueError("Official3D geometry/context metadata differs")
    records = manifest.get("records", [])
    if len(records) != count or manifest.get("record_count") != count:
        raise ValueError("Complete actual record count differs")
    contents, payloads, offset = set(), set(), 0
    for index, record in enumerate(records):
        if set(record) not in (RECORD_KEYS, RECORD_KEYS | {"step_context"}):
            raise ValueError("Unexpected/missing source record fields")
        if (record["id"] != f"train:{index:06d}" or record["source_index"] != index or type(record["source_index"]) is not int
                or not isinstance(record["source_key"], list) or not record["source_key"]
                or any(type(k) is not int for k in record["source_key"]) or record["source_offset_bytes"] != offset
                or not integer(record["record_payload_bytes"], 1) or not digest_string(record["record_payload_sha256"])
                or record["record_payload_sha256"] in payloads):
            raise ValueError("Source ordering/offsets/keys/payload identity differs")
        offset += record["record_payload_bytes"] + 16
        payloads.add(record["record_payload_sha256"])
        descriptors = [("positions", "position"), ("particle_types", "type")]
        if "step_context" in record:
            descriptors.append(("step_context", "step_context"))
        for name, prefix in descriptors:
            array = record[name]
            if (set(array) != ARRAY_KEYS | ({"use"} if name == "step_context" else set())
                    or not digest_string(array["sha256"]) or not integer(array["size_bytes"], 1)
                    or array["path"] != f"train/{prefix}_{index:06d}.npy"):
                raise ValueError("Exact numeric hash/shape/dtype/size/path descriptor required")
        p, t = record["positions"], record["particle_types"]
        shape = p["shape"]
        if (not isinstance(shape, list) or len(shape) != 3 or shape[0] != FRAMES or shape[2] != 3
                or not integer(shape[1], 1) or p["dtype"] != "<f4" or t["shape"] != [shape[1]] or t["dtype"] != "<i8"):
            raise ValueError("Preserved T301/D3 float32 positions and full int64 types required")
        if "step_context" in record:
            aux = record["step_context"]
            if (not isinstance(aux["shape"], list) or len(aux["shape"]) < 2 or aux["shape"][0] != FRAMES
                    or not all(integer(v, 1) for v in aux["shape"][1:]) or aux["dtype"] != "<f4" or aux["use"] != AUXILIARY_USE):
                raise ValueError("Preserved optional3D context descriptor differs")
        content = hashlib.sha256((p["sha256"] + ":" + t["sha256"]).encode()).hexdigest()
        if record["trajectory_content_sha256"] != content or content in contents:
            raise ValueError("Duplicate trajectory or inconsistent content identity")
        contents.add(content)
    if offset != OFFICIAL_TRAIN["size_bytes"]:
        raise ValueError("Payload offsets do not cover complete official3D TFRecord")


def verify_goop3d_evidence(args, manifest, admission):
    paths = {"train_manifest": "manifest_sha256", "structural_report": "structural_report_sha256",
             "acquisition_report": "acquisition_report_sha256", "context_semantics": "context_semantics_sha256",
             "auxiliary_report": "auxiliary_report_sha256"}
    if any(sha(getattr(args, name)) != admission[key] for name, key in paths.items()):
        raise ValueError("Admitted3D evidence bytes differ")
    root = args.context_semantics.resolve().parent / "goop_context_semantics_sources"
    if any(sha(root / name) != digest for name, digest in CONTEXT_SOURCE_PINS.items()):
        raise ValueError("Pinned official parser/model context source differs")
    semantics = json.loads(args.context_semantics.read_text())
    if (semantics.get("schema") != "goop3d_official_context_semantics_review_v1" or semantics.get("official_commit") != OFFICIAL_COMMIT
            or semantics.get("metadata", {}).get("sha256") != METADATA_SHA
            or any(semantics.get("metadata", {}).get(key) is not False for key in ("context_mean_present", "context_std_present"))
            or {name: row.get("sha256") for name, row in semantics.get("sources", {}).items()} != CONTEXT_SOURCE_PINS):
        raise ValueError("Reviewed3D context semantics differs")
    structural = json.loads(args.structural_report.read_text())
    train = structural.get("splits", {}).get("train", {})
    if (structural.get("schema") != "official_goop3d_numeric_preparation_v1" or structural.get("status") != "complete_structural_only"
            or structural.get("test_accessed") is not False or structural.get("dataset") != "Goop-3D"
            or structural.get("source_family") != "official_gns_tfrecord" or structural.get("metadata_sha256") != METADATA_SHA
            or structural.get("reader_sha256") != READER_SHA or structural.get("wrapper_sha256") != CONVERTER_SHA
            or structural.get("acquisition_report_sha256") != admission["acquisition_report_sha256"]
            or train.get("manifest_sha256") != admission["manifest_sha256"] or train.get("record_count") != admission["record_count"]
            or train.get("frame_lengths") != [FRAMES] or train.get("dimension") != 3
            or train.get("particle_type_ids") != admission["particle_type_ids"]
            or train.get("eligible_six_frame_histories") != admission["eligible_six_frame_histories"]
            or train.get("forecast_horizons_after_six_frames") != [FRAMES - 6]
            or any(train.get(key) is not True for key in ("source_EOF_SHA256_verified", "whole_object_crc32c_verified",
                "TFRecord_length_and_payload_CRC32C_verified", "all_match_official_parser_frame_count", "all_source_array_bytes_preserved_exact"))):
        raise ValueError("Complete3D structural/source evidence differs")
    receipt = json.loads(args.acquisition_report.read_text())
    files = {row.get("name"): row for row in receipt.get("files", [])}
    source = files.get("train.tfrecord", {})
    if (receipt.get("schema") != "official_goop3d_train_valid_acquisition_v1" or receipt.get("status") != "complete"
            or receipt.get("dataset") != "Goop-3D" or source.get("status") != "complete"
            or source.get("sha256") != admission["source_sha256"] or source.get("received_bytes") != OFFICIAL_TRAIN["size_bytes"]
            or source.get("generation") != OFFICIAL_TRAIN["generation"] or source.get("crc32c_base64") != OFFICIAL_TRAIN["crc32c_base64"]
            or source.get("crc32c_verified") is not True):
        raise ValueError("Complete official3D acquisition receipt differs")
    census = json.loads(args.auxiliary_report.read_text())
    split = census.get("splits", {}).get("train", {})
    if (census.get("schema") != "adaptgns_goop3d_auxiliary_census_v1" or census.get("status") != "all_preserved_auxiliary_bytes_verified"
            or census.get("test_accessed") is not False or census.get("source_sha256") != AUXILIARY_VALIDATOR_SHA
            or census.get("structural_report_sha256") != admission["structural_report_sha256"] or census.get("metadata_sha256") != METADATA_SHA
            or split.get("manifest_sha256") != admission["manifest_sha256"] or split.get("record_count") != admission["record_count"]
            or len(split.get("records", [])) != admission["record_count"]
            or any(split.get(key) is not True for key in ("context_mean_absent", "context_std_absent", "all_context_descriptors_and_bytes_verified"))):
        raise ValueError("Complete3D auxiliary census differs")
    present = 0
    for record, row in zip(manifest["records"], split["records"]):
        exists = "step_context" in record
        present += exists
        if row.get("id") != record["id"] or row.get("source_index") != record["source_index"] or row.get("present") is not exists:
            raise ValueError("Auxiliary census source identity/presence differs")
        if not exists:
            if set(row) != {"id", "source_index", "present"}:
                raise ValueError("Absent auxiliary cannot contain invented array statistics")
        elif (row.get("sha256") != record["step_context"]["sha256"] or row.get("descriptor_sha256") != config_hash(record["step_context"])
                or row.get("shape") != record["step_context"]["shape"] or row.get("dtype") != "<f4"):
            raise ValueError("Auxiliary descriptor identity differs")
    if split.get("auxiliary_present_count") != present or split.get("auxiliary_absent_count") != len(manifest["records"]) - present:
        raise ValueError("Full auxiliary presence census differs")
    return split["records"]


def verify_goop3d_auxiliary_arrays(args, helpers, manifest, census_rows):
    path = Path(__file__).with_name("audit_goop3d_auxiliary.py")
    if sha(path) != AUXILIARY_VALIDATOR_SHA:
        raise ValueError("Pinned3D auxiliary validator differs")
    spec = importlib.util.spec_from_file_location("_goop3d_auxiliary_verifier", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    for record, expected in zip(manifest["records"], census_rows):
        actual, _ = audit.census_record(args.train_manifest.resolve().parent, "train", record, helpers.np)
        if actual != expected:
            raise ValueError("Actual preserved3D auxiliary bytes/rawbit census differs")


def load_admitted_dataset(args, helpers):
    admission = json.loads(args.admission.read_text())
    validate_prospective_endpoint(args, admission)
    manifest = json.loads(args.train_manifest.read_text())
    validate_manifest_contract(manifest, admission)
    census_rows = verify_goop3d_evidence(args, manifest, admission)
    metadata_path = args.train_manifest.resolve().parent / "metadata.json"
    if sha(metadata_path) != METADATA_SHA:
        raise ValueError("Actual3D metadata bytes differ")
    metadata = json.loads(metadata_path.read_text())
    if metadata != manifest["metadata"]:
        raise ValueError("Manifest metadata differs")
    verify_goop3d_auxiliary_arrays(args, helpers, manifest, census_rows)
    dataset, _, info = helpers.load_manifest_dataset(args.train_manifest, "train")
    observed_types = set()
    for record, (positions, types) in zip(manifest["records"], dataset._data):
        for name in ("positions", "particle_types"):
            path = helpers.data_loader._manifest_array_path(args.train_manifest.resolve().parent, record[name])
            if path.stat().st_size != record[name]["size_bytes"]:
                raise ValueError("Actual3D numeric array byte length differs")
        if (positions.shape != tuple(record["positions"]["shape"]) or positions.dtype.str != "<f4"
                or types.shape != (positions.shape[1],) or types.dtype.str != "<i8" or not helpers.np.isfinite(positions).all()):
            raise ValueError("Actual full3D source shape/dtype/finiteness differs")
        observed_types.update(int(v) for v in helpers.np.unique(types))
    if (sorted(observed_types) != admission["particle_type_ids"] or len(dataset._data) != admission["record_count"]
            or len(dataset) != admission["eligible_six_frame_histories"]):
        raise ValueError("Actual3D type/count/history census differs")
    info.update({"frames_per_trajectory": FRAMES, "dimension": 3, "particle_type_ids": sorted(observed_types),
                 "forecast_horizon_after_six_frames": FRAMES - 6, "admission_sha256": sha(args.admission),
                 "context_source_sha256": CONTEXT_SOURCE_PINS, "auxiliary_policy": admission["auxiliary_policy"]})
    for key in ("structural_report_sha256", "converter_sha256", "reader_sha256", "metadata_sha256",
                "acquisition_report_sha256", "context_semantics_sha256", "auxiliary_report_sha256", "auxiliary_validator_sha256"):
        info[key] = admission[key]
    return dataset, metadata, info


def reverify_goop3d_evidence_and_auxiliaries(args, helpers):
    admission, manifest = json.loads(args.admission.read_text()), json.loads(args.train_manifest.read_text())
    validate_prospective_endpoint(args, admission)
    validate_manifest_contract(manifest, admission)
    census_rows = verify_goop3d_evidence(args, manifest, admission)
    verify_goop3d_auxiliary_arrays(args, helpers, manifest, census_rows)


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


def assert_adam(torch, model, optimizer, completed, endpoint):
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
    expected_lr = learning_rate(max(completed - 1, 0), endpoint)
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


def guarded_update(helpers, model, optimizer, pred, head, target, mask, objective, next_step, endpoint):
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
    assert_adam(torch, model, optimizer, next_step, endpoint)
    return loss


def checkpoint_payload(helpers, model, optimizer, config, completed, history, device):
    validate_graph_history(history, completed, config["arm"], config["seed"])
    if not helpers.tensors_are_finite(model.state_dict().values()):
        raise FloatingPointError("Cannot commit nonfinite model state")
    assert_adam(helpers.torch, model, optimizer, completed, config["updates"])
    return {"format_version": 2, "cuda_goop3d_graph_support_schema": SCHEMA,
            "state_dict": helpers.cpu_tree(model.state_dict()),
            "simulator_config": helpers.cpu_tree(model._checkpoint_config),
            "training_config": {"loss": config["objective"], "cuda_goop3d_graph_support_run": config,
                                "completed_optimizer_updates": completed},
            "optimizer_state": helpers.cpu_tree(optimizer.state_dict()),
            "run_config": config, "run_config_sha256": config_hash(config),
            "completed_steps": completed, "history": history,
            "rng_states": helpers.cpu_tree(capture_rng(helpers.torch, device))}


def restore_payload(helpers, payload, model, optimizer, config, device):
    torch = helpers.torch
    completed = payload.get("completed_steps")
    if (payload.get("format_version") != 2 or payload.get("cuda_goop3d_graph_support_schema") != SCHEMA
            or any(key in payload for key in ("cuda_goop3d_training_schema", "cuda_goop_graph_support_schema", "cuda_goop_training_schema", "cuda_sand_graph_support_schema", "cuda_sand_training_schema", "full_training_schema", "graph_support_schema"))
            or payload.get("run_config_sha256") != config_hash(config) or payload.get("run_config") != config
            or not integer(completed) or completed > config["updates"]
            or payload.get("training_config") != {"loss": config["objective"], "cuda_goop3d_graph_support_run": config,
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
    assert_adam(torch, model, optimizer, completed, config["updates"])
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
    config = {"schema": SCHEMA, "dataset": "Goop-3D", "objective": args.objective,
              "seed": args.seed, "arm": args.arm, "updates": args.updates, "batch_size": 2, "history": 6,
              "initialization": "from scratch; paired seed across arms; empty Adam; no parent checkpoint",
              "graph_exposure": graph_exposure_config(args.arm),
              "architecture": {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
              "noise_std": helpers.NOISE, "graph": {"radius": .025, "backend": "scipy_host", "cap": 128,
                                                       "self_candidates": True, "augmentation_probability": 0.0},
              "optimizer": {"name": "Adam", "initial_lr": 1e-4, "final_lr": 1e-5, "decay_updates": args.updates,
                            "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0,
                            "foreach": False, "fused": False, "gradient_clipping": None},
              "checkpoint_every": args.checkpoint_every, "log_every": args.log_every,
              "data": data_info, "research_protocol_sha256": sha(args.protocol),
              "source_sha256": {**SOURCE_PINS, "train_goop3d_graph_support_cuda_v1.py": sha(__file__), "goop3d_graph_support.py": GRAPH_SHA}, "runtime": runtime,
              "selection": ENDPOINT_BASIS, "prospective_endpoint_updates": args.updates}
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
        connectivity_radius=.025, nmessage_passing_steps=10, uncertainty_parameterization="variance",
        variance_floor=1e-6, detach_variance_features=True, radius_backend="scipy_host").to(device)
    if model._checkpoint_config["particle_dimensions"] != 3 or model._checkpoint_config["nnode_in"] != 37 or model._checkpoint_config["nedge_in"] != 4:
        raise ValueError("Goop-3D simulator dimensions differ")
    model._training_config = {"loss": args.objective, "cuda_goop3d_graph_support_run": config}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False)
    completed, history, latest = 0, {"training": [], "graph_updates": [], "elapsed_seconds": 0.0}, None
    if args.resume:
        latest = read_pointer(output, config)  # Missing pointers require manual recovery; no silent restart.
        payload = torch.load(output / latest["path"], map_location="cpu", weights_only=True)
        completed, history = restore_payload(helpers, payload, model, optimizer, config, device)
        if completed != latest["completed_steps"]:
            raise ValueError("Pointer and checkpoint update counts disagree")
    stop = args.updates
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
            batch = unpack_batch(helpers, [train[index] for index in indices])
            noise = helpers.host_noise(batch[0].shape, batch[1], args.seed, completed)
            frame_ids = [helpers.frame_identity(train, data_info["trajectory_ids"], index) for index in indices]
            current_context = {"completed_before": completed, "absolute_schedule_step": completed,
                "frame_ids": frame_ids, "noise_sha256": support.state_hash(noise.numpy())}
            rate = learning_rate(completed, args.updates)
            for group in optimizer.param_groups:
                group["lr"] = rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred, head, target, graph_ledger = support.forward_batch(model, batch, noise, device, args.seed, completed, args.arm)
            current_context["examples"] = graph_ledger
            mask = (batch[1] != helpers.KINEMATIC).to(device)
            loss = guarded_update(helpers, model, optimizer, pred, head, target, mask, args.objective, completed + 1, args.updates)
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
            path = (Path(__file__) if relative == "train_goop3d_graph_support_cuda_v1.py" else
                    Path(__file__).with_name(relative) if relative == "goop3d_graph_support.py" else args.repo / relative)
            if sha(path) != expected:
                raise ValueError("Source changed during graph-support training: " + relative)
        for path, expected in ((args.train_manifest, data_info["manifest_sha256"]),
                               (args.admission, data_info["admission_sha256"]),
                               (args.structural_report, data_info["structural_report_sha256"]),
                               (args.protocol, config["research_protocol_sha256"])):
            if sha(path) != expected:
                raise ValueError("Input changed during graph-support training: " + str(path))
        helpers.data_loader.load_manifest_data(args.train_manifest, verify_hashes=True)
        reverify_goop3d_evidence_and_auxiliaries(args, helpers)
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
    validate_prospective_endpoint(args, json.loads(args.admission.read_text()))
    helpers, support = load_helpers(args.repo)
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise ValueError("Device remapping is not admitted")
    device, runtime = configure_cuda(helpers, args)
    with RunLock(args.output_dir, args.clear_stale_lock) as process:
        run_training(args, helpers, support, process, device, runtime)


if __name__ == "__main__":
    main()
