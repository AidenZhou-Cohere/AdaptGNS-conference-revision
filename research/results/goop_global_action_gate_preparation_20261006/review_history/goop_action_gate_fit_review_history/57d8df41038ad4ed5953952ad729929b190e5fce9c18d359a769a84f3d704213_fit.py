"""Fit three prospective action gates from complete committed label collections.

Description-only by default. No simulator imports, model deserialization,
network, acquisition, subprocesses, or test outcomes. Requires a root release.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

CORE_SHA = "d3986c2ac3786e9fc0d36d76339bfeded5448497f7dad9b0559efaaaa53a43ca"
PROTOCOL_SHA = "bc9235fae89d32ede8d1e7f846bff07bdcdd85f4671d5922d688a9535f7fe024"
SCHEMA = "adaptgns_goop_global_action_gate_fit_v1"
COLLECTION_SCHEMA = "adaptgns_goop_global_action_gate_label_collection_v1"
ROW_SCHEMA = "adaptgns_goop_global_action_gate_label_row_v1"
DEADLINE = datetime(2026, 10, 7, 4, tzinfo=timezone.utc)
ARRAY_KEYS = {"history", "particle_types", "features", "base_edges", "random25_edges",
              "random25_selected_optional_pairs", "base_prediction", "random25_prediction",
              "base_risk", "base_raw_risk", "random25_risk", "random25_raw_risk", "target", "action_errors"}
LINEAGE = ("original_training_protocol_sha256", "core_sha256", "driver_sha256", "cohort_sha256",
           "cohort_audit_sha256", "training_admission_sha256", "metadata_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def snapshot(path, pin):
    raw = Path(path).read_bytes()
    require(digest(pin) and hashlib.sha256(raw).hexdigest() == pin, "JSON bytes changed: " + str(path))
    return json.loads(raw)


def publish(path, value):
    raw = encode(value)
    with Path(path).open("xb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def merge_inputs(target, incoming):
    require(isinstance(incoming, dict) and incoming, "Nonempty original input hash map required")
    for name, pin in incoming.items():
        require(isinstance(name, str) and Path(name).is_absolute() and digest(pin)
                and (name not in target or target[name] == pin), "Conflicting or nonabsolute original input hash")
        target[name] = pin


def verify_inputs(inputs):
    for name, pin in inputs.items():
        require(sha(name) == pin, "Original input changed: " + name)


def inventory(root):
    files, entries = {}, []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if relative == "label_collection.json":
            continue
        require(not path.is_symlink(), "Symlink in label output tree")
        if path.is_dir():
            entries.append(dict(path=relative, kind="directory"))
        else:
            require(path.is_file(), "Unsupported label output entry")
            files[relative] = dict(sha256=sha(path), bytes=path.stat().st_size)
            entries.append(dict(path=relative, kind="file"))
    return files, entries


def basename(name):
    require(isinstance(name, str) and name and Path(name).name == name and name not in (".", ".."),
            "Artifact must be a local basename")
    return name


def array_check(np, core, row, path):
    require(sha(path) == row["artifact_sha256"], "Numeric artifact bytes differ")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == ARRAY_KEYS and len(archive.files) == len(ARRAY_KEYS),
                "Complete numeric action arrays required")
        arrays = {key: archive[key] for key in archive.files}
    require(set(row["numeric_arrays"]) == ARRAY_KEYS, "Complete numeric descriptors required")
    for name, values in arrays.items():
        description = row["numeric_arrays"][name]
        require(values.dtype.kind in "fiu" and np.isfinite(values).all(), "Finite numeric arrays required")
        require(description == dict(shape=list(values.shape), dtype=values.dtype.str,
                    value_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()),
                "Numeric value/shape/dtype descriptor differs")
    h = arrays["history"]
    require(h.dtype == np.dtype("float32") and h.ndim == 3 and h.shape[0] == 6
            and h.shape[1] > 0 and h.shape[2] == 2, "Clean history shape/dtype differs")
    n = h.shape[1]
    require(row["history_sha256"] == hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest(), "History hash differs")
    expected_shapes = {"particle_types": (n,), "features": (11,), "target": (n, 2), "action_errors": (3,)}
    for prefix in ("base", "random25"):
        expected_shapes.update({prefix + "_prediction": (n, 2), prefix + "_risk": (n,), prefix + "_raw_risk": (n,)})
    for key, shape in expected_shapes.items():
        dtype = np.dtype("int64" if key == "particle_types" else "float64" if key in ("features", "action_errors") else "float32")
        require(arrays[key].shape == shape and arrays[key].dtype == dtype, "Action array shape/dtype differs: " + key)
    require((arrays["particle_types"] == 7).all(), "Goop material IDs differ")
    for prefix in ("base", "random25"):
        prediction = arrays[prefix + "_prediction"]
        require(np.max(np.abs(prediction)) <= 10.0 and (arrays[prefix + "_risk"] > 0.0).all(), "Complete action violates numerical guard")
        require(row["prediction_sha256"][prefix] == hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest(),
                "Prediction value hash differs")
    base, added, selected = arrays["base_edges"], arrays["random25_edges"], arrays["random25_selected_optional_pairs"]
    require(all(v.dtype == np.dtype("int64") and v.ndim == 2 for v in (base, added, selected))
            and base.shape[0] == added.shape[0] == 2 and selected.shape[1] == 2,
            "Integer native edges/selected canonical pairs required")
    require(all(((v >= 0) & (v < n)).all() for v in (base, added, selected))
            and (selected[:, 0] < selected[:, 1]).all(), "Graph endpoint indices differ")
    canonical = {tuple(v) for v in selected.tolist()}
    require(len(canonical) == len(selected) and added.shape[1] == base.shape[1] + 2 * len(selected)
            and np.array_equal(added[:, :base.shape[1]], base), "Native prefix/optional pair count differs")
    orientations = canonical | {(j, i) for i, j in canonical}
    require(set(map(tuple, added[:, base.shape[1]:].T.tolist())) == orientations,
            "Appended optional orientations differ")
    recomputed_features = core.features(h)
    require(np.array_equal(recomputed_features, arrays["features"])
            and np.array_equal(recomputed_features, np.asarray(row["features"], dtype=np.float64)), "Saved features differ from current history")
    errors = core.action_errors(arrays["base_prediction"], arrays["random25_prediction"], arrays["target"])
    expected_errors = np.array([errors[key] for key in ("base_mse", "random25_mse", "signed_benefit")])
    require(np.array_equal(expected_errors, arrays["action_errors"])
            and all(row[key] == value for key, value in errors.items()), "Signed action labels differ from predictions/target")


def read_collection(entry, release, np, core, inputs, check_time):
    seed, split = entry["seed"], entry["split"]
    require(type(seed) is int and seed in (0, 1, 2) and split in ("train", "valid"), "Fixed seed/split identity required")
    path = Path(entry["path"])
    require(path.is_absolute() and path.name == "label_collection.json" and not path.is_symlink(), "Original collection path required")
    collection = snapshot(path, entry["sha256"])
    expected = core.row_ids(split); count = len(expected)
    require(collection.get("schema") == COLLECTION_SCHEMA and collection.get("status") == "complete"
            and collection.get("mode") == ("train-labels" if split == "train" else "validation-labels")
            and collection.get("split") == split and collection.get("all_required_labels_complete") is True
            and collection.get("all_inputs_reverified") is True, "Complete scientific label collection required")
    require(all(type(collection.get(k)) is int and collection[k] == count
                for k in ("required_rows", "committed_rows", "complete_rows")), "Full label counts required")
    core.checked_rows(collection["expected_row_ids"], split)
    require(collection["coverage"] == [dict(source_index=s, target_frame=t, state="committed_complete") for s, t in expected],
            "Every scheduled row must be committed complete")
    model = collection["model"]
    require(type(model.get("seed")) is int and model["seed"] == seed and model.get("arm") == "mix"
            and model.get("objective") == "faithful" and model.get("completed_updates") == 100000
            and digest(model.get("checkpoint_sha256")) and digest(model.get("run_config_sha256")), "Fixed mix100k model required")
    require(collection.get("gate_protocol_sha256") == PROTOCOL_SHA and collection.get("core_sha256") == CORE_SHA
            and collection.get("driver_sha256") == release["driver_sha256"]
            and all(digest(collection.get(key)) for key in LINEAGE + ("source_manifest_sha256",)), "Collection lineage differs")
    files, entries = inventory(path.parent)
    require(files == collection["files"] and entries == collection["output_tree_entries"], "Committed label tree differs")
    require("protocol.json" in files, "Per-run protocol receipt required")
    rows = collection["rows"]; require(len(rows) == count, "Missing label rows")
    features, benefits, base_errors, random_errors = [], [], [], []
    for row, (source, frame) in zip(rows, expected):
        check_time()
        require(row.get("schema") == ROW_SCHEMA and type(row.get("model_seed")) is int and row["model_seed"] == seed
                and row.get("arm") == "mix" and row.get("checkpoint_updates") == 100000
                and row.get("checkpoint_sha256") == model["checkpoint_sha256"]
                and type(row.get("source_index")) is int and type(row.get("target_frame")) is int
                and (row["source_index"], row["target_frame"], row.get("split")) == (source, frame, split)
                and row.get("status") == "complete" and row.get("failure") is None
                and type(row.get("network_passes")) is int and row["network_passes"] == 2, "Required paired-action row incomplete or mismatched")
        require(row.get("gate_protocol_sha256") == PROTOCOL_SHA and row.get("protocol_sha256") == files["protocol.json"]["sha256"]
                and row.get("source_manifest_sha256") == collection["source_manifest_sha256"]
                and digest(row.get("source_trajectory_content_sha256"))
                and row.get("pair_rng_material") == core.rng_material(split, seed, source, frame), "Row source/protocol/RNG binding differs")
        row_name, artifact_name = basename(row["row_file"]), basename(row["artifact_file"])
        prefix = f"source_{source:06d}_target_{frame:03d}"
        require(row_name == prefix + ".json" and artifact_name == prefix + ".npz", "Scheduled artifact filename differs")
        require(files[row_name]["sha256"] == row["row_sha256"] and files[artifact_name]["sha256"] == row["artifact_sha256"], "Row/artifact byte inventory differs")
        committed = snapshot(path.parent / row_name, row["row_sha256"])
        require(committed == {key: value for key, value in row.items() if key not in ("row_file", "row_sha256")}, "Collection row differs from committed row JSON")
        array_check(np, core, row, path.parent / artifact_name)
        features.append(row["features"]); benefits.append(row["signed_benefit"])
        base_errors.append(row["base_mse"]); random_errors.append(row["random25_mse"])
    merge_inputs(inputs, collection["input_sha256"])
    merge_inputs(inputs, {str(path): entry["sha256"]})
    return collection, dict(features=np.array(features), benefits=np.array(benefits), row_ids=expected,
                            base_mse=np.array(base_errors), random25_mse=np.array(random_errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    for flag in ("release", "core", "gate-protocol", "output-dir"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(schema=SCHEMA, status="description_only", scientific_execution=False))); return 0
    require(all(getattr(args, key) is not None for key in ("release", "core", "gate_protocol", "output_dir")), "All explicit fit arguments required")
    args.release, args.core, args.gate_protocol, args.output_dir = (p.resolve() for p in (args.release, args.core, args.gate_protocol, args.output_dir))
    release_pin = sha(args.release); release = snapshot(args.release, release_pin)
    require(release.get("schema") == "adaptgns_goop_global_action_gate_fit_release_v1" and release.get("issued_by") == "root"
            and release.get("status") == "approved_complete_labels_for_fitting" and release.get("output_dir") == str(args.output_dir)
            and release.get("core_sha256") == CORE_SHA == sha(args.core)
            and release.get("gate_protocol_sha256") == PROTOCOL_SHA == sha(args.gate_protocol)
            and release.get("fit_source_sha256") == sha(__file__) and digest(release.get("driver_sha256")), "Exact root fitting release required")
    require(type(release.get("max_seconds")) in (int, float) and 0 < release["max_seconds"] < 86400, "Bounded fitting allocation required")
    require(type(release.get("cpu_threads")) is int and release["cpu_threads"] == 2, "Fixed two-thread CPU fit required")
    stop = datetime.fromisoformat(release["absolute_stop_utc"])
    require(stop.tzinfo is not None and stop <= DEADLINE, "Fitting stop must preserve the prospective deadline")
    started = time.monotonic()
    def check_time():
        require(time.monotonic() - started <= release["max_seconds"] and datetime.now(timezone.utc) < stop, "Fitting allocation exhausted")
    check_time()
    entries = release["collections"]
    require(len(entries) == 6 and all(type(e.get("seed")) is int for e in entries)
            and {(e["seed"], e.get("split")) for e in entries} == {(s, split) for s in (0, 1, 2) for split in ("train", "valid")}
            and len({e["path"] for e in entries}) == 6, "All six distinct label collections required")
    inputs = {}; merge_inputs(inputs, release["files_sha256"])
    for path, pin in ((args.release, release_pin), (args.core, CORE_SHA), (args.gate_protocol, PROTOCOL_SHA), (Path(__file__).resolve(), sha(__file__))):
        merge_inputs(inputs, {str(path): pin})
    verify_inputs(inputs)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    spec = importlib.util.spec_from_file_location("_approved_goop_gate_core", args.core)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    import numpy as np
    require(not args.output_dir.exists(), "A new fitting output directory is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    loaded, numeric, candidates, validation = {}, {}, {}, {}
    try:
        for entry in entries:
            key = (entry["seed"], entry["split"])
            loaded[key], numeric[key] = read_collection(entry, release, np, core, inputs, check_time)
        for seed in (0, 1, 2):
            train, valid = loaded[(seed, "train")], loaded[(seed, "valid")]
            require(train["model"] == valid["model"] and all(train[k] == valid[k] == loaded[(0, "train")][k] for k in LINEAGE), "Train/validation model or shared lineage differs")
            require(train["source_manifest_sha256"] != valid["source_manifest_sha256"], "Train/validation manifests must differ")
        require(len({loaded[(seed, "train")]["model"]["checkpoint_sha256"] for seed in (0, 1, 2)}) == 3,
                "Three distinct frozen mix checkpoints required")
        verify_inputs(inputs); check_time()
        for seed in (0, 1, 2):
            train = numeric[(seed, "train")]
            hashes = {"train_collection": sha(next(e["path"] for e in entries if (e["seed"], e["split"]) == (seed, "train"))),
                      "checkpoint": loaded[(seed, "train")]["model"]["checkpoint_sha256"],
                      "cohort": loaded[(seed, "train")]["cohort_sha256"], "fit_release": release_pin}
            candidates[seed] = core.fit_candidates(train["features"], train["benefits"], train["row_ids"], seed, hashes)
            validation[seed] = numeric[(seed, "valid")]
        selection = core.select_common_lambda(candidates, validation)
        outputs = {}
        for seed in (0, 1, 2):
            for penalty in core.LAMBDAS:
                name = f"seed{seed}_lambda{penalty:g}.json"
                outputs[name] = publish(args.output_dir / name, candidates[seed][penalty])
            name = f"selected_seed{seed}.json"; outputs[name] = publish(args.output_dir / name, selection["heads"][seed])
        outputs["selection.json"] = publish(args.output_dir / "selection.json", selection)
        result = dict(schema=SCHEMA, status="all_three_gates_fitted_and_selected", gate_protocol_sha256=PROTOCOL_SHA,
            core_sha256=CORE_SHA, driver_sha256=release["driver_sha256"], fit_release_sha256=release_pin,
            collection_sha256={e["path"]:e["sha256"] for e in entries}, input_sha256=inputs,
            output_sha256=outputs, selected_lambda=selection["selected_lambda"], test_admitted=False,
            runtime=dict(python=sys.version, numpy=np.__version__, cpu_threads=2),
            elapsed_seconds_before_publication=time.monotonic() - started)
        encode(result)
        verify_inputs(inputs)
        for entry in entries:
            original = loaded[(entry["seed"], entry["split"])]; current = inventory(Path(entry["path"]).parent)
            require(current == (original["files"], original["output_tree_entries"]), "Label tree changed during fitting")
        require(all(sha(args.output_dir / name) == pin for name, pin in outputs.items()), "Fitted outputs changed before publication")
        check_time(); publish(args.output_dir / "fit_result.json", result)
    except BaseException as exc:
        publish(args.output_dir / "fit_failure.json", dict(schema=SCHEMA, status="failed_no_test_admission",
            exception_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic() - started))
        raise
    print(json.dumps(dict(status=result["status"], output_dir=str(args.output_dir), test_admitted=False))); return 0


if __name__ == "__main__":
    raise SystemExit(main())
