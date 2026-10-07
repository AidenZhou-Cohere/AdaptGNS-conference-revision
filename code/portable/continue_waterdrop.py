#!/usr/bin/env python3
"""Continue a complete WaterDrop 100k parent for 10k paired graph-exposure updates.

Supply parent, data and output paths plus the expected parent SHA256. The
MPS/two-thread/no-fallback profile, optimizer state and absolute update schedule
are preserved. Source/configuration checks are separate from --execute."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

SOURCE_SHA256 = "67515a25882a86c53e8c1152a8c5e35ac32e1d43667849276cd03068e1845d09"
PORTABLE_SCHEMA = "adaptgns_waterdrop_graph_support_portable_continuation_v1"
PORTABLE_SCOPE = "portable_faithful_graph_support_100k_to110k"
CORE_FUNCTIONS = (
    "main", "run", "graph_rng", "append_optional_edges", "forward_batch",
    "update_once", "parent_optimizer", "checkpoint_payload", "save_checkpoint",
    "validate_continuation_payload", "restore_checkpoint", "verify_pins",
    "simulator_config_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_hash(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Explicit lowercase SHA256 required")
    return value


def check_source(repo):
    source = Path(repo).resolve() / "research/faithful_graph_support.py"
    require(sha(source) == SOURCE_SHA256, "Frozen WaterDrop continuation source differs")
    return {"source_sha256": SOURCE_SHA256, "packaged_reference_source_matches": True,
            "core_function_names": list(CORE_FUNCTIONS)}


def parent_identity(seed, parent_sha256, adapter_sha256):
    require(type(seed) is int and seed in (0, 1, 2), "Seed must be 0, 1 or 2")
    return {"schema": PORTABLE_SCHEMA, "seed": seed,
            "parent_checkpoint_sha256": valid_hash(parent_sha256),
            "adapter_sha256": valid_hash(adapter_sha256),
            "frozen_continuation_source_sha256": SOURCE_SHA256,
            "parent_updates": 100000, "additional_updates": 10000,
            "absolute_schedule_steps": [100000, 109999], "learning_rate": 1e-5,
            "training_kind": "paired continuation from complete model, Adam and RNG state",
            "historical_study_receipt": False}


def bind_parent_for_validation(support, seed, parent_sha256):
    """Bind an explicit reproduction identity before unchanged endpoint checks.

    Evaluation must require this hash separately from the checkpoint hash. This
    does not validate a parent model or establish historical study provenance;
    the original payload validator still checks architecture/normalization,
    continuation/config hashes, graph-history schedule and the full endpoint.
    """
    require(type(seed) is int and seed in (0, 1, 2), "Seed must be 0, 1 or 2")
    valid_hash(parent_sha256)
    require(sha(support.__file__) == SOURCE_SHA256, "Frozen continuation source differs")
    require(support.SCHEMA in (1, PORTABLE_SCHEMA), "Unexpected continuation schema")
    support.PARENT_HASHES = {seed: parent_sha256}
    support.SCHEMA = PORTABLE_SCHEMA


def validate_portable_identity(payload, seed, parent_sha256, adapter_sha256=None):
    """Additional metadata boundary; call the original validator afterward."""
    config = payload["continuation_config"]
    identity = config.get("portable_reproduction", {})
    require(payload.get("graph_support_schema") == PORTABLE_SCHEMA and
            config.get("schema") == PORTABLE_SCHEMA and config.get("scope") == PORTABLE_SCOPE,
            "Portable continuation schema/scope differs")
    expected = parent_identity(seed, parent_sha256,
                               adapter_sha256 or sha(__file__))
    require(identity == expected, "Portable parent/adapter/schedule identity differs")
    require(config.get("seed") == seed and config.get("parent_checkpoint_sha256") == parent_sha256,
            "Continuation parent/seed differs from explicit expected identity")
    require(expected["adapter_sha256"] in config.get("input_files_sha256", {}).values(),
            "Adapter is missing from the frozen input map")
    return identity


def install_training_boundary(support, identity, adapter_path):
    """Keep every numerical function and original main; replace one identity gate."""
    require(not getattr(support, "_portable_boundary_installed", False),
            "Portable continuation boundary already installed")
    saved_functions = {name: getattr(support, name) for name in CORE_FUNCTIONS}
    original_configure = support.configure
    bind_parent_for_validation(support, identity["seed"], identity["parent_checkpoint_sha256"])
    adapter_path = Path(adapter_path).resolve()
    require(sha(adapter_path) == identity["adapter_sha256"], "Adapter bytes differ")

    def configure(*args, **kwargs):
        config = original_configure(*args, **kwargs)
        require(config["schema"] == PORTABLE_SCHEMA, "Original configure did not retain portable schema")
        require(config["seed"] == identity["seed"] and
                config["parent_checkpoint_sha256"] == identity["parent_checkpoint_sha256"],
                "Original configuration does not match explicit parent")
        config["scope"] = PORTABLE_SCOPE
        config["portable_reproduction"] = dict(identity)
        config["input_files_sha256"][str(adapter_path)] = identity["adapter_sha256"]
        return config

    support.configure = configure
    support._portable_boundary_installed = True
    require(all(getattr(support, name) is function for name, function in saved_functions.items()),
            "A frozen core function was replaced")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--check-source-equivalence", action="store_true")
    for name in ("parent-checkpoint", "train-manifest", "valid-manifest", "metadata", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--parent-sha256")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--arm", choices=("base", "mix"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(not (args.execute and args.check_source_equivalence),
            "Source-only check and execution are separate operations")
    if args.check_source_equivalence:
        return args
    for name in ("parent_checkpoint", "parent_sha256", "train_manifest", "valid_manifest",
                 "metadata", "output_dir", "seed", "arm"):
        require(getattr(args, name) is not None, "Missing --" + name.replace("_", "-"))
    valid_hash(args.parent_sha256)
    return args


def original_argv(args):
    values = ["faithful_graph_support.py"]
    for name in ("parent_checkpoint", "train_manifest", "valid_manifest", "metadata", "output_dir", "seed", "arm"):
        values += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    values += ["--device", "mps", "--threads", "2"]
    for name in ("protocol", "parent_protocol"):
        if getattr(args, name) is not None:
            values += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.resume:
        values += ["--resume"]
    return values


def main(argv=None):
    args = parse_args(argv)
    source_report = check_source(args.repo)
    if args.check_source_equivalence:
        print(json.dumps(source_report, indent=2, sort_keys=True))
        return
    identity = parent_identity(args.seed, args.parent_sha256, sha(__file__))
    require(sha(args.parent_checkpoint) == args.parent_sha256, "Explicit parent checkpoint SHA256 differs")
    if not args.execute:
        print(json.dumps({"configuration_only": True, "identity": identity,
                          "source": source_report, "original_argv": original_argv(args)},
                         indent=2, sort_keys=True))
        return
    # Only this branch imports scientific modules or loads model/data arrays.
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    support = importlib.import_module("research.faithful_graph_support")
    require(Path(support.__file__).resolve() == repo / "research/faithful_graph_support.py",
            "Imported continuation came from a different repository")
    install_training_boundary(support, identity, Path(__file__))
    previous_argv = sys.argv
    try:
        sys.argv = original_argv(args)
        support.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
