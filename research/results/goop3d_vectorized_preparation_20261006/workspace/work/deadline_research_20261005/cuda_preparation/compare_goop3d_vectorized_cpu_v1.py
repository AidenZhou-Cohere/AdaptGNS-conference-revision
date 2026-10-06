#!/usr/bin/env python3
"""Pinned first-record CPU graph equality/timing; no model or CUDA calls."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
SCHEMA = "goop3d_first_record_vectorized_cpu_comparison_v1"
PROFILE_SHA = "43a35689e7639166be0da55ce96a12920833ad8f0a99d57a1a6841c906bd74fb"
CANDIDATE_SHA = "ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50"
REFERENCE_SHA = "500a333a397f706f1e0ff9d482cb7389279d0ab5fdb6bdca0c8f612d14e26367"
CPU_REPORT_SHA = "e1ee532ce7d31a9a851736475a22c78a5801bbc1873140cd9880278aeee35bf5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def load_helpers(directory):
    source = directory / "profile_goop3d_first_record_cpu_graph.py"
    candidate_path = directory / "goop3d_graph_support_vectorized_v1.py"
    require(sha(source) == PROFILE_SHA and sha(candidate_path) == CANDIDATE_SHA, "Pinned CPU helper/candidate differs")
    spec = importlib.util.spec_from_file_location("_goop3d_cpu_exact_reference", source)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    bench, original = profile.load_pinned_helpers(directory)
    names = {"Candidates", "require", "state_hash", "canonical_pairs", "pair_set", "check_candidate_bound",
             "strict_pairs", "sorted_pair_difference", "ordered_edges", "graph_rng", "append_optional_edges", "host_noise"}
    tree = ast.parse(candidate_path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    require({node.name for node in selected} == names, "Candidate CPU graph functions missing")
    namespace = {"np": original.np, "torch": original.torch, "cKDTree": original.cKDTree,
        "math": math, "hashlib": hashlib, "dataclass": dataclass, "__name__": __name__,
        "RADIUS_FACTOR": 1.267, "MAX_PAIRS": 2000000, "MAX_BATCH_EDGES": 5000000, "NOISE": 6.7e-4, "KINEMATIC": 3}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(candidate_path), "exec"), namespace)
    return profile, bench, original, SimpleNamespace(**namespace)


def prove_exact(batch, noise, case, profile, original, candidate):
    require(all(value.device.type == "cpu" for value in (*batch, noise)), "CPU tensors required")
    noisy = batch[0] + noise
    points, counts = noisy[:, -1].numpy(), batch[2].tolist()
    native = original.torch.from_numpy(profile.native_graph_cpu(points, counts, original.np, original.cKDTree))
    start = 0
    for count in counts:
        a = original.strict_pairs(points[start:start + count], .025)
        b = candidate.strict_pairs(points[start:start + count], .025)
        for left, right in ((a.base, b.base), (a.extra, b.extra)):
            require(left.dtype == right.dtype and left.shape == right.shape and original.np.array_equal(left, right),
                    "Actual canonical pair arrays differ")
        start += count
    before = original.torch.get_rng_state().clone()
    a, ledger_a = original.append_optional_edges(noisy, batch[2], native, .025, 0, 0, case)
    b, ledger_b = candidate.append_optional_edges(noisy, batch[2], native, .025, 0, 0, case)
    require(a.dtype == b.dtype and a.shape == b.shape and original.torch.equal(a, b)
            and ledger_a == ledger_b, "Actual ordered edges/ledger differ")
    require(original.torch.equal(before, original.torch.get_rng_state()), "Global CPU RNG changed")
    return {"canonical_pairs_exact": True, "all_ordered_edges_exact": True, "all_ledger_fields_exact": True,
        "global_cpu_rng_unchanged": True, "graph_ledger": ledger_a,
        "native_batch_edge_sha256": original.state_hash(native.numpy()),
        "final_batch_edge_sha256": original.state_hash(a.numpy()), "final_directed_edges": int(a.shape[1])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    for name in ("inspection-dir", "metadata", "official-reading-utils", "reference-report", "cpu-report", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": SCHEMA, "execution": False, "cuda_calls": False, "model_or_optimizer": False,
                          "cases": ["base", "expanded25"], "warmup_repetitions": 1, "measured_repetitions": 3}))
        return 0
    require(all(getattr(args, name) is not None for name in
        ("inspection_dir", "metadata", "official_reading_utils", "reference_report", "cpu_report", "output")),
        "Explicit pinned inputs and fresh output required")
    require(not args.output.exists(), "Preserve previous comparison output")
    directory = Path(__file__).resolve().parent
    profile, bench, original, candidate = load_helpers(directory)
    report = {"schema": SCHEMA, "status": "running", "source_sha256": sha(__file__),
        "candidate_sha256": CANDIDATE_SHA, "cpu_profiler_sha256": PROFILE_SHA,
        "cuda_calls": False, "model_or_optimizer": False, "test_accessed": False,
        "scientific_training_admitted": False, "cases": []}
    bench.write(args.output, report)
    started = time.perf_counter()
    try:
        require(sha(args.reference_report) == REFERENCE_SHA and sha(args.cpu_report) == CPU_REPORT_SHA,
                "Completed reference benchmark/CPU report differs")
        reference = json.loads(args.reference_report.read_text())
        cpu_report = json.loads(args.cpu_report.read_text())
        require(reference["status"] == "complete_fixed_state_capacity_only"
                and cpu_report["status"] == "complete_cpu_graph_stage_profile_only", "Completed reference reports required")
        original.torch.set_num_threads(2)
        batch, noise, metadata, info, inputs = bench.load_inputs(args, original)
        for key in ("history_sha256", "labels_sha256", "noise_sha256"):
            require(info[key] == reference["input"][key] == cpu_report["input"][key], "Input/noise identity differs: " + key)
        report.update(input=info, input_files_sha256=inputs, reference_report_sha256=REFERENCE_SHA,
            cpu_report_sha256=CPU_REPORT_SHA, runtime={"python": platform.python_version(), "platform": platform.platform(),
                "torch": str(original.torch.__version__), "numpy": original.np.__version__, "cpu_threads": 2})
        for case in ("base", "expanded25"):
            proof = prove_exact(batch, noise, case, profile, original, candidate)
            expected_ledger = next(row for row in reference["cases"] if row["case"] == case)["updates"][0]["graph_ledger"]
            prior = next(row for row in cpu_report["cases"] if row["case"] == case)["repetitions"][0]
            require(proof["graph_ledger"] == expected_ledger == prior["graph_ledger"], "Frozen graph ledger differs")
            for key in ("native_batch_edge_sha256", "final_batch_edge_sha256", "final_directed_edges"):
                require(proof[key] == prior[key], "Frozen ordered edge identity differs: " + key)
            case_report = {"case": case, "exact_proof": proof, "repetitions": []}
            report["cases"].append(case_report)
            for repetition in range(4):
                order = ("reference", "candidate") if repetition % 2 == 0 else ("candidate", "reference")
                paired = {"repetition": repetition, "warmup": repetition == 0, "measurement_order": list(order)}
                for label in order:
                    graph = original if label == "reference" else candidate
                    row = profile.graph_stages(batch, noise, case, graph)
                    require(row["graph_ledger"] == proof["graph_ledger"], "Repeated ledger differs")
                    for key in ("native_batch_edge_sha256", "final_batch_edge_sha256", "final_directed_edges"):
                        require(row[key] == proof[key], "Repeated edge identity differs")
                    paired[label] = row["stage_seconds"]
                case_report["repetitions"].append(paired)
                bench.write(args.output, report)
            case_report["mean_measured_stage_seconds"] = {label: {stage: statistics.mean(
                row[label][stage] for row in case_report["repetitions"][1:]) for stage in paired[label]}
                for label in ("reference", "candidate")}
        require(all(bench.sha(path) == digest for path, digest in inputs.items())
            and sha(args.reference_report) == REFERENCE_SHA and sha(args.cpu_report) == CPU_REPORT_SHA
            and sha(__file__) == report["source_sha256"]
            and sha(directory / "profile_goop3d_first_record_cpu_graph.py") == PROFILE_SHA
            and sha(directory / "goop3d_graph_support_vectorized_v1.py") == CANDIDATE_SHA
            and sha(directory / "benchmark_goop3d_first_record_v2.py") == profile.BENCHMARK_SHA
            and sha(directory / "goop3d_graph_support.py") == profile.GRAPH_SHA, "Input/source changed")
        report.update(status="complete_same_state_cpu_graph_equivalence_and_timing_only", limitations=[
            "One previously inspected training state; no whole-cohort capacity or scientific efficacy evidence.",
            "Only CPU graph stages measured. No CUDA transfer, GNN, backward or Adam timing/numerical check.",
            "Do not subtract CPU time from the prior GPU benchmark as a measured end-to-end speedup.",
            "Any GPU adoption requires a separate pinned lineage and numerical review before scientific launch."])
        return 0
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        bench.write(args.output, report)


if __name__ == "__main__":
    raise SystemExit(main())
