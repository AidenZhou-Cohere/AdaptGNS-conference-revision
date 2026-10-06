"""Independent scalar/aggregation audit; standard library only, no inference.

The frozen strict native summarizer owns graph/trace and checkpoint admission.
This separate check recomputes its scalar estimands from every native/original
raw row, checks all seed/paired summaries, and rechecks the loaded file bytes.
It never executes on a partial six-model cohort or overwrites an audit attempt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25")
OBJECTIVES = ("faithful", "nll")
INDICES = tuple(range(3, 30))
METRICS = ("mean_rollout_mse", "mse_at_200", "mse_at_995", "failure_fraction",
           "mean_directed_edges", "mean_rollout_wall_seconds", "mean_total_network_passes")
BOUNDARIES = tuple(f"{kind}_boundary_{metric}" for kind in ("predicted", "ground_truth")
                  for metric in ("mean_fraction_outside_gt_1e_minus6", "mean_step_maximum_excursion", "trajectory_maximum_excursion"))
COUNTS = ("native_base_directed_edges", "native_base_self_edges", "native_base_receivers_above_cap_before_capping",
          "native_base_edges_removed_by_cap", "native_base_asymmetric_directed_edges", "retained_optional_pairs")
EXTRAS = tuple("mean_" + name for name in COUNTS) + ("cap_active_step_fraction", "mean_wall_seconds_including_native_parity",
          "mean_native_parity_seconds", "mean_native_parity_passes", "mean_graph_operational_seconds", "mean_forward_component_seconds")
COMPARISONS = tuple((p, "base") for p in POLICIES[1:]) + tuple(("laggedrisk25", p) for p in ("dense", "random25", "speed25"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean(values, count):
    require(len(values) == count, "Fixed population count differs")
    return statistics.fmean(values) if all(v is not None for v in values) else None


class Audit:
    def __init__(self):
        self.checks = 0
        self.pins = {}

    def equal(self, actual, expected):
        self.checks += 1
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), "Dictionary keys differ")
            for key in expected:
                self.equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), "Sequence length differs")
            for a, b in zip(actual, expected):
                self.equal(a, b)
        elif type(expected) is bool:
            require(type(actual) is bool and actual == expected, "Boolean type/value differs")
        elif type(expected) in (int, float):
            require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                    and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-14), "Numeric value differs")
        else:
            require(actual == expected, "Value/null differs")

    def pin(self, path, expected=None):
        path = str(Path(path).resolve())
        h = sha(path)
        if expected is not None:
            self.equal(h, expected)
        if path in self.pins:
            self.equal(h, self.pins[path])
        self.pins[path] = h
        return h

    def read(self, path, expected=None):
        self.pin(path, expected)
        def invalid(value):
            raise ValueError("Nonfinite JSON literal: " + value)
        return json.loads(Path(path).read_text(), parse_constant=invalid)


def row_metrics(row, native, audit):
    steps = row["completed_steps"]
    done = row["status"] == "complete"
    require(type(steps) is int and 0 <= steps <= 995, "Invalid accepted prefix")
    require(row["status"] in ("complete", "failed") and done == (steps == 995 and row["failure"] is None), "Outcome status inconsistent")
    require(done or (steps < 995 and isinstance(row["failure"], dict) and row["failure"]), "Scientific failure prefix/reason invalid")
    audit.equal(row["horizon"], 995)
    mse, edges = row["mse_per_step"], row["directed_edges_per_step"]
    audit.equal(len(mse), steps); audit.equal(len(edges), steps)
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in mse + edges), "Invalid prefix scalar")
    expected = {"mean_rollout_mse": mean(mse, 995) if done else None,
                "mse_at_200": mse[199] if steps >= 200 else None,
                "mse_at_995": mse[994] if done else None,
                "failure_fraction": float(not done), "mean_directed_edges": mean(edges, 995) if done else None,
                "mean_rollout_wall_seconds": row["total_wall_seconds"], "mean_total_network_passes": row["total_network_passes"]}
    for key in ("mean_rollout_mse", "mean_directed_edges"):
        audit.equal(row[key], expected[key])
    audit.equal(set(row["mse_at_steps"]), {"1", "10", "50", "200", "500", "995"})
    for step, value in row["mse_at_steps"].items():
        audit.equal(value, mse[int(step)-1] if steps >= int(step) else None)
    audit.equal(row["mse_at_final_horizon"], expected["mse_at_995"])
    require(expected["mean_rollout_wall_seconds"] >= 0 and expected["mean_total_network_passes"] >= steps, "Negative timing or missing passes")
    for kind in ("predicted", "ground_truth"):
        values = row[kind + "_boundary_per_step"]
        audit.equal(len(values), steps)
        prefix = kind + "_boundary_"
        expected[prefix + "mean_fraction_outside_gt_1e_minus6"] = mean([v["fraction_particles_outside_by_more_than_1e-6"] for v in values], 995) if done else None
        expected[prefix + "mean_step_maximum_excursion"] = mean([v["maximum_coordinate_excursion"] for v in values], 995) if done else None
        expected[prefix + "trajectory_maximum_excursion"] = max(v["maximum_coordinate_excursion"] for v in values) if done else None
    if native:
        attempts = row["attempts"]
        audit.equal([a["forecast_step"] for a in attempts], list(range(1, len(attempts)+1)))
        require(len(attempts) in (steps, steps+1), "Wrong attempted prefix length")
        audit.equal([a["accepted"] for a in attempts], [True]*steps + [False]*(len(attempts)-steps))
        accepted = attempts[:steps]
        for key in COUNTS:
            expected["mean_"+key] = mean([a[key] for a in accepted], 995) if done else None
        expected["cap_active_step_fraction"] = mean([float(a["native_base_receivers_above_cap_before_capping"] > 0) for a in accepted], 995) if done else None
        warmup = row["warmup"] or {}
        passes = sum("features_forward_decode_and_transfer_seconds" in a for a in attempts)
        audit.equal(row["forecast_network_passes"], passes)
        audit.equal(row["total_network_passes"], passes + warmup.get("passes", 0))
        for metric, key in (("mean_graph_operational_seconds", "graph_build_and_selection_seconds"),
                            ("mean_forward_component_seconds", "features_forward_decode_and_transfer_seconds")):
            expected[metric] = sum(a.get(key, 0.) for a in attempts) + warmup.get(key, 0.)
            audit.equal(row[key+"_including_warmup"], expected[metric])
        expected.update(mean_wall_seconds_including_native_parity=row["total_wall_seconds_including_native_parity"],
                        mean_native_parity_seconds=row["native_parity_operational_seconds"], mean_native_parity_passes=row["native_parity_network_passes"])
        require(expected["mean_wall_seconds_including_native_parity"] >= expected["mean_rollout_wall_seconds"], "Timing scopes reversed")
    return expected


def summary_stat(values):
    valid = [v for v in values if v is not None]
    require(len(values) == 3, "Three original seed slots required")
    return {"seed_values": [{"seed": i, "value": v} for i, v in enumerate(values)],
            "defined_seed_count": len(valid), "mean": statistics.fmean(valid) if len(valid) == 3 else None,
            "sample_sd": statistics.stdev(valid) if len(valid) == 3 else None}


def load_family(root, native, audit):
    runs = {}
    for obj in OBJECTIVES:
        for seed in range(3):
            directory = root / f"{obj}_seed{seed}"
            require(not (directory / "run.lock").exists(), "Active child lock")
            status = audit.read(directory / "status.json")
            audit.equal(status["state"], "complete")
            result = audit.read(directory / "result.json", status["result_sha256"])
            audit.equal(result["state"], "complete")
            audit.equal((result["objective"], result["seed"]), (obj, seed))
            protocol = audit.read(directory / "protocol.json", result["protocol_sha256"])
            audit.equal(protocol["checkpoint_sha256"], result["checkpoint_sha256"])
            compact = result["records"]
            require(len(compact) == 135 and {(r["source_index"], r["policy"]) for r in compact} == {(i, p) for i in INDICES for p in POLICIES}, "Fixed outcome population differs")
            rows = {}
            for item in compact:
                for k, h in (("record_file", "record_sha256"), ("trace_file", "trace_sha256")):
                    path = directory / item[k]
                    require(path.parent == directory and path.name == item[k], "Nonlocal record path")
                    audit.pin(path, item[h])
                row = audit.read(directory / item["record_file"], item["record_sha256"])
                for key in ("source_index", "policy", "trajectory_id", "status", "failure", "completed_steps", "trace_sha256"):
                    audit.equal(row[key], item[key])
                audit.equal(row["protocol_sha256"], result["protocol_sha256"])
                rows[item["source_index"], item["policy"]] = {"metrics": row_metrics(row, native, audit), "trajectory_id": row["trajectory_id"]}
            runs[obj, seed] = {"checkpoint_sha256": result["checkpoint_sha256"], "rows": rows}
    return runs


def compare_group(group, left, right, obj, lp, rp, metrics, audit):
    values = {name: [] for name in metrics}
    for seed, row in enumerate(group["per_seed"]):
        audit.equal(row["seed"], seed)
        units = []
        for index in INDICES:
            a, b = left[obj, seed]["rows"][index, lp], right[obj, seed]["rows"][index, rp]
            audit.equal(a["trajectory_id"], b["trajectory_id"])
            delta = {name: a["metrics"][name]-b["metrics"][name] if a["metrics"][name] is not None and b["metrics"][name] is not None else None for name in metrics}
            units.append({"source_index": index, "trajectory_id": a["trajectory_id"], "deltas": delta})
        audit.equal(row["trajectory_pairs"], units)
        expected = {name: mean([v["deltas"][name] for v in units], 27) for name in metrics}
        audit.equal(row["metrics"], expected)
        for name, value in expected.items():
            values[name].append(value)
    audit.equal(group["metrics"], {name: summary_stat(v) for name, v in values.items()})


def execute(summary_path, native_root, original_root, audit):
    report = audit.read(summary_path)
    audit.equal(report["scope"], "post-inspection native convention follow-up on original six100k models")
    audit.equal(report["audit"]["passed"], True)
    require(not (native_root / "run.lock").exists(), "Native queue still active")
    native = load_family(native_root, True, audit)
    original = load_family(original_root, False, audit)
    require(len(report["native_runs"]) == len(report["original_runs"]) == 6, "Summary cohort differs")
    metrics = METRICS + BOUNDARIES + EXTRAS
    audit.equal(set(report["groups"]), set(OBJECTIVES))
    for obj in OBJECTIVES:
        group = report["groups"][obj]
        audit.equal(set(group["policies"]), set(POLICIES))
        audit.equal(set(group["native_minus_original"]), set(POLICIES))
        for seed in range(3):
            audit.equal(native[obj, seed]["checkpoint_sha256"], original[obj, seed]["checkpoint_sha256"])
        for policy in POLICIES:
            values = {name: [] for name in metrics}
            for seed, actual in enumerate(group["policies"][policy]["per_seed"]):
                failed = sum(native[obj, seed]["rows"][i, policy]["metrics"]["failure_fraction"] == 1. for i in INDICES)
                audit.equal(actual["required_trajectories"], 27)
                audit.equal(actual["completed_trajectories"], 27-failed)
                audit.equal(actual["failed_trajectories"], failed)
                expected = {name: mean([native[obj, seed]["rows"][i, policy]["metrics"][name] for i in INDICES], 27) for name in metrics}
                audit.equal(actual["metrics"], expected)
                for name, value in expected.items():
                    values[name].append(value)
            audit.equal(group["policies"][policy]["metrics"], {name: summary_stat(v) for name, v in values.items()})
            compare_group(group["native_minus_original"][policy], native, original, obj, policy, policy, METRICS + BOUNDARIES, audit)
        audit.equal(set(group["native_policy_comparisons"]), {a+"_minus_"+b for a, b in COMPARISONS})
        for left, right in COMPARISONS:
            compare_group(group["native_policy_comparisons"][left+"_minus_"+right], native, native, obj, left, right, metrics, audit)
    for path, digest in audit.pins.items():
        audit.equal(sha(path), digest)
    return {"native_outcomes": 810, "original_outcomes": 810, "source_summary_sha256": audit.pins[str(summary_path.resolve())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit, started = Audit(), time.perf_counter()
    audit.pin(__file__)
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": sha(__file__),
              "scope": "Independent raw scalar and paired aggregation arithmetic; no graph/trace reconstruction, model/data/source admission or inference. Frozen strict summary supplies those complementary checks."}
    try:
        record.update(execute(args.summary, args.native_root, args.original_root, audit), passed=True)
    except BaseException as error:
        record.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(checks=audit.checks, wall_seconds=time.perf_counter()-started, input_files_sha256=audit.pins)
        (args.output_dir / "audit.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
