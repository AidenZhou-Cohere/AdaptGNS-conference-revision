#!/usr/bin/env python3
"""Train the fixed Goop or Sand 100k recipe from explicit numeric data paths.

The one-GPU GB200/CUDA numerical profile is checked. Source equivalence and
configuration checks do not load simulation arrays or instantiate a model;
--execute enables training and writes a separately identified reproduction."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = {
    "goop": {"name": "Goop", "frames": 401, "type": 7,
             "protocol": "goop_graph_support_100k_protocol_v2.md"},
    "sand": {"name": "Sand", "frames": 320, "type": 6,
             "protocol": "sand_graph_support_100k_protocol_v1.md"},
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def transformed_loop(source, dataset):
    """Five explicit nonnumerical substitutions; all other source is copied.

    Keep this transform identical to the source-only generator. The generated
    loop remains readable, and the equality check runs before any ML import.
    """
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "run_training")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[function.lineno - 1:function.end_lineno])
    initial = "train, metadata, data_info = load_admitted_dataset(args, helpers)"
    require(body.count(initial) == 1, "Original data boundary changed")
    body = body.replace(initial, "train, metadata, data_info = load_portable_dataset(args, helpers)")
    config_end = '              "selection": "fixed final requested update; no validation/test selection"}\n'
    require(body.count(config_end) == 1, "Original config boundary changed")
    body = body.replace(config_end, config_end + '    config["portable_reproduction"] = args.portable_identity\n')
    old_checks = '''        for path, expected in ((args.train_manifest, data_info["manifest_sha256"]),
                               (args.admission, data_info["admission_sha256"]),
                               (args.structural_report, data_info["structural_report_sha256"]),
                               (args.protocol, config["research_protocol_sha256"])):'''
    new_checks = '''        for path, expected in ((args.train_manifest, data_info["manifest_sha256"]),
                               (args.metadata, data_info["metadata_sha256"]),
                               (args.protocol, config["research_protocol_sha256"])):'''
    require(body.count(old_checks) == 1, "Original final input boundary changed")
    body = body.replace(old_checks, new_checks)
    old_arrays = "        helpers.data_loader.load_manifest_data(args.train_manifest, verify_hashes=True)"
    require(body.count(old_arrays) == 1, "Original array recheck boundary changed")
    body = body.replace(old_arrays, "        reverify_portable_inputs(args, helpers, data_info)")
    old_goop = "        reverify_goop_evidence_and_auxiliaries(args, helpers)\n"
    require(body.count(old_goop) == (1 if dataset == "goop" else 0), "Goop auxiliary boundary changed")
    return body.replace(old_goop, "")


def check_source_equivalence(frozen_dir, dataset):
    mapping = json.loads((HERE / "training_source_map.json").read_text())
    row = mapping["datasets"][dataset]
    source_path = Path(frozen_dir) / row["original_basename"]
    loop_path = HERE / row["portable_loop_basename"]
    require(sha(source_path) == row["original_sha256"], "Frozen original trainer bytes differ")
    require(sha(loop_path) == row["portable_loop_sha256"], "Portable loop bytes differ")
    expected = transformed_loop(source_path.read_text(), dataset)
    actual = loop_path.read_text()
    require(actual == expected, "Portable loop has changes beyond the explicit boundary substitutions")
    original_tree = ast.parse(source_path.read_text())
    original_loop = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "run_training")
    portable_loop = ast.parse(actual).body[0]
    # The complete optimizer-update loop, including every scientific call,
    # schedule, graph ledger, finite check and checkpoint trigger, is identical.
    original_while = [n for n in ast.walk(original_loop) if isinstance(n, ast.While)]
    portable_while = [n for n in ast.walk(portable_loop) if isinstance(n, ast.While)]
    require(len(original_while) == len(portable_while) == 1, "Unexpected training-loop structure")
    require(ast.dump(original_while[0], include_attributes=False) ==
            ast.dump(portable_while[0], include_attributes=False), "Numerical update loop changed")
    return {"dataset": dataset, "original_sha256": row["original_sha256"],
            "portable_loop_sha256": row["portable_loop_sha256"],
            "whole_update_loop_ast_identical": True,
            "only_declared_boundary_substitutions": True}


def validate_manifest_header(manifest, metadata, dataset, metadata_sha256, expected_metadata_sha256):
    """Data contract only; no historical admission or execution metadata."""
    spec = DATASETS[dataset]
    require(manifest.get("format") == "gns-trajectory-manifest" and manifest.get("version") == 1,
            "Version-1 numeric trajectory manifest required")
    require(manifest.get("dataset") == spec["name"] and manifest.get("split") == "train",
            "Wrong material or nontraining split")
    require(manifest.get("metadata") == metadata and manifest.get("metadata_sha256") == metadata_sha256
            and metadata_sha256 == expected_metadata_sha256, "Published metadata identity differs")
    require(metadata.get("dim") == 2 and metadata.get("default_connectivity_radius") == .015
            and metadata.get("dt") == .0025, "Fixed geometry/time metadata differs")
    require("context_mean" not in metadata and "context_std" not in metadata,
            "These 2D models do not consume context features")
    records = manifest.get("records", [])
    require(len(records) == manifest.get("record_count") == 1000, "All 1000 training trajectories required")
    content_hashes = set()
    for i, row in enumerate(records):
        require(row.get("id") == f"train:{i:06d}" and row.get("source_index") == i,
                "Source order and exact frame identifiers must be preserved")
        positions, types = row["positions"], row["particle_types"]
        shape = positions.get("shape", [])
        require(len(shape) == 3 and shape[0] == spec["frames"] and shape[2] == 2
                and type(shape[1]) is int and shape[1] > 0 and positions.get("dtype") == "<f4",
                "Fixed float32 trajectory shape required")
        require(types.get("shape") in ([shape[1]], [] if dataset == "sand" else [shape[1]])
                and types.get("dtype") in (("<i8", "<i4") if dataset == "sand" else ("<i8",)),
                "Particle-type descriptor differs")
        digest = hashlib.sha256((positions["sha256"] + ":" + types["sha256"]).encode()).hexdigest()
        require(row.get("trajectory_content_sha256") == digest and digest not in content_hashes,
                "Duplicate or misbound trajectory contents")
        content_hashes.add(digest)
    return spec


def check_auxiliaries(args, helpers, manifest):
    """Preserve Goop's unused context bytes without feeding them to the model."""
    if args.dataset != "goop":
        return {}
    np = helpers.np
    pins = {}
    for row in manifest["records"]:
        descriptor = row["step_context"]
        path = helpers.data_loader._manifest_array_path(args.train_manifest.parent, descriptor)
        require(path.stat().st_size == descriptor["size_bytes"] and sha(path) == descriptor["sha256"],
                "Excluded Goop context bytes differ from the manifest")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        require(values.shape == (401, 1) and values.dtype.str == "<f4"
                and int(np.isnan(values).sum()) == 1 and int(np.isfinite(values).sum()) == 400
                and [f"{int(v):08x}" for v in np.unique(values.view("<u4"))] == ["00000000", "7fc00000"],
                "Excluded Goop context layout/content differs")
        pins[str(path)] = descriptor["sha256"]
    return pins


def load_portable_dataset(args, helpers):
    manifest = json.loads(args.train_manifest.read_text())
    metadata = json.loads(args.metadata.read_text())
    metadata_sha = sha(args.metadata)
    spec = validate_manifest_header(manifest, metadata, args.dataset, metadata_sha, args.expected_metadata_sha256)
    train, _, info = helpers.load_manifest_dataset(args.train_manifest, "train")
    require(len(train._data) == 1000 and len(train) == 1000 * (spec["frames"] - 6),
            "Complete eligible-history grid differs")
    for row, (positions, types) in zip(manifest["records"], train._data):
        require(positions.shape == tuple(row["positions"]["shape"]) and positions.dtype.str == "<f4"
                and helpers.np.isfinite(positions).all(), "Actual training positions differ")
        require(helpers.np.all(types == spec["type"]), "Actual particle type differs")
        for key in ("positions", "particle_types"):
            path = helpers.data_loader._manifest_array_path(args.train_manifest.parent, row[key])
            require(path.stat().st_size == row[key]["size_bytes"], "Actual numeric byte length differs")
    info.update(metadata_sha256=metadata_sha, frames_per_trajectory=spec["frames"],
                particle_type_ids=[spec["type"]], forecast_horizon_after_six_frames=spec["frames"] - 6,
                preserved_unused_context_sha256=check_auxiliaries(args, helpers, manifest))
    return train, metadata, info


def reverify_portable_inputs(args, helpers, data_info):
    helpers.data_loader.load_manifest_data(args.train_manifest, verify_hashes=True)
    require(sha(args.metadata) == data_info["metadata_sha256"], "Metadata changed during training")
    for path, expected in data_info["preserved_unused_context_sha256"].items():
        require(sha(path) == expected, "Excluded context changed during training")
    for path, expected in args.portable_identity["adapter_files_sha256"].items():
        require(sha(path) == expected, "Portable adapter changed during training")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True,
                        help="Directory containing the original trainer and fixed protocol files")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check-source-equivalence", action="store_true")
    for name in ("repo", "train-manifest", "metadata", "output-dir", "protocol"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--arm", choices=("base", "mix"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        for name in ("repo", "train_manifest", "metadata", "output_dir", "arm", "seed"):
            if getattr(args, name) is None:
                parser.error("Execution requires --" + name.replace("_", "-"))
    if args.cuda_index < 0 or (args.clear_stale_lock and not args.resume):
        parser.error("Invalid CUDA index or stale-lock recovery without --resume")
    args.objective, args.updates, args.threads = "faithful", 100000, 2
    args.checkpoint_every, args.log_every, args.stop_after = 10000, 100, None
    return args


def main(argv=None):
    args = parse_args(argv)
    args.frozen_dir = args.frozen_dir.resolve()
    equivalence = check_source_equivalence(args.frozen_dir, args.dataset)
    if args.check_source_equivalence:
        print(json.dumps(equivalence, indent=2))
        return 0
    for name in ("repo", "train_manifest", "metadata", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    require(args.metadata == args.train_manifest.parent / "metadata.json",
            "The manifest loader requires its matching adjacent metadata.json")
    args.protocol = (args.protocol or args.frozen_dir / DATASETS[args.dataset]["protocol"]).resolve()
    original = args.frozen_dir / f"train_{args.dataset}_graph_support_cuda.py"
    spec = importlib.util.spec_from_file_location("_portable_" + args.dataset + "_trainer", original)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)  # Original trainer imports only standard library here.
    args.expected_metadata_sha256 = trainer.METADATA_SHA
    loop_path = HERE / f"{args.dataset}_training_loop.py"
    args.portable_identity = {
        "scope": "new_portable_reproduction_not_original_study_execution",
        "original_training_schema": trainer.SCHEMA,
        "adapter_files_sha256": {str(path): sha(path) for path in
                                 (Path(__file__).resolve(), loop_path, HERE / "training_source_map.json")},
        "source_equivalence": equivalence,
        "dataset_contract": "full ordered 1000-trajectory numeric manifest, original metadata and graph recipe",
    }
    trainer.SCHEMA = f"adaptgns_{args.dataset}_graph_support_portable_training_v1"
    trainer.load_portable_dataset = load_portable_dataset
    trainer.reverify_portable_inputs = reverify_portable_inputs
    exec(compile(loop_path.read_text(), str(loop_path), "exec"), trainer.__dict__)
    # These remain unchanged: exact frozen numerical modules, original CUDA
    # deterministic profile, original local output lock, and all RNG/Adam
    # checkpoint/replay checks. No historical owner/session is contacted.
    helpers, support = trainer.load_helpers(args.repo)
    device, runtime = trainer.configure_cuda(helpers, args)
    with trainer.RunLock(args.output_dir, args.clear_stale_lock) as process:
        trainer.run_training(args, helpers, support, process, device, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
