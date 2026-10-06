"""Complementary independent coordinate-error and random-envelope audit.

Standard library and NumPy only. No model, source-data or graph reconstruction.
The strict summary supplies those checks; this audit verifies raw predictions,
signed gains, exact recorded-scalar ties and every error/envelope aggregation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

CONTROLS = ("base", "dense", "speed25", "previous-observed-base-risk25", "current-base-risk25")
DRAWS = tuple("random25_draw" + str(i) for i in range(8))
CASES = CONTROLS + DRAWS
METRICS = ("position_coordinate_mse", "normalized_coordinate_mse")
FIELDS = ("reference_minus_random_mean", "random_strictly_better", "random_tied",
          "random_strictly_worse", "minimum", "maximum", "sample_sd")
IDENTITY = ("split", "source_index", "target_frame", "trajectory_id")
COHORT = {(o, s) for o in ("faithful", "nll") for s in range(3)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    require(values and all(v is None or finite(v) for v in values), "Finite nonboolean population required")
    return math.fsum(values) / len(values) if all(v is not None for v in values) else None


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=reject)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks, self.pins = 0, {}

    def equal(self, actual, expected):
        self.checks += 1
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), "Dictionary coverage differs")
            for k in expected:
                self.equal(actual[k], expected[k])
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), "Sequence coverage differs")
            for a, b in zip(actual, expected):
                self.equal(a, b)
        elif type(expected) is bool:
            require(type(actual) is bool and actual == expected, "Boolean differs")
        elif finite(expected):
            require(finite(actual) and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-13), "Scalar differs")
        else:
            require(actual == expected, "Identity/null differs")

    def pin(self, path, expected=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), "Regular input file required")
        name, digest = str(path.resolve()), sha(path)
        if expected is not None:
            self.equal(digest, expected)
        if name in self.pins:
            self.equal(digest, self.pins[name])
        self.pins[name] = digest
        return digest


def envelope(values, reference):
    require(len(values) == 8 and all(v is None or finite(v) for v in values), "Exactly eight finite/null draws required")
    require(reference is None or finite(reference), "Finite/null reference required")
    count = sum(v is not None for v in values)
    result = dict(draw_values=values, required_draws=8, defined_draws=count, mean=None,
        minimum=None, maximum=None, sample_sd=None, monte_carlo_standard_error_of_mean=None, reference_minus_random_mean=None,
        random_strictly_better=None, random_tied=None, random_strictly_worse=None,
        random_mean_null_reason="required random draw failed/missing" if count < 8 else None,
        comparison_null_reason="required reference or random draw failed/missing" if count < 8 or reference is None else None)
    if count == 8:
        average = mean(values)
        sd = statistics.stdev(values)
        result.update(mean=average, minimum=min(values), maximum=max(values), sample_sd=sd,
                      monte_carlo_standard_error_of_mean=sd / math.sqrt(8))
        if reference is not None:
            result.update(reference_minus_random_mean=reference-average,
                random_strictly_better=len([v for v in values if v < reference]),
                random_tied=len([v for v in values if v == reference]),
                random_strictly_worse=len([v for v in values if v > reference]))
    return result


def envelopes(values):
    return {m: {c: envelope([values[m][d] for d in DRAWS], values[m][c]) for c in CONTROLS} for m in METRICS}


def stats(values):
    require(len(values) == 3, "Exactly three seeds required")
    average = mean(values)
    return dict(seed_values=values, required_seeds=3, defined_seeds=sum(v is not None for v in values),
        mean=average, sample_sd=statistics.stdev(values) if average is not None else None,
        null_reason=None if average is not None else "required seed/frame/policy failed or missing")


def raw_values(row, arrays, audit):
    audit.equal(set(row["cases"]), set(CASES))
    values = {m: {} for m in METRICS}
    errors = {}
    for case in CASES:
        record = row["cases"][case]
        require(record["status"] in ("complete", "failed"), "Unknown scientific outcome")
        good = record["status"] == "complete"
        require(good == (record["failure"] is None), "Failure status differs")
        if not good:
            require(isinstance(record["failure"], dict) and record["failure"], "Failure reason required")
            audit.equal(record["metrics"], None)
            for m in METRICS:
                values[m][case] = None
            continue
        p, target, scale = [np.asarray(arrays[k], dtype=np.float64) for k in
                           ("prediction__" + case, "target_position", "acceleration_std")]
        require(p.ndim == 2 and p.shape[1] == 2 and p.shape == target.shape and len(p) > 0
                and scale.shape == (2,) and all(np.isfinite(x).all() for x in (p, target, scale))
                and np.all(scale > 0), "Finite 2D prediction/target/normalization required")
        residual = (p - target).reshape(-1)
        normalized = (p - target) / scale
        calculated = dict(position_coordinate_mse=float(residual @ residual / residual.size),
                          normalized_coordinate_mse=float(np.sum(normalized * normalized) / normalized.size))
        for m in METRICS:
            audit.equal(record["metrics"][m], calculated[m])
            # The declared tie rule uses the audited, recorded float64 scalars,
            # not a potentially one-ulp-different independent summation.
            values[m][case] = record["metrics"][m]
        se = np.sum(normalized * normalized, axis=1) / 2
        audit.checks += 1
        require(np.allclose(arrays["normalized_coordinate_se__" + case], se, rtol=2e-12, atol=1e-14), "Particle errors differ")
        errors[case] = se
    for case in CASES:
        if "base" in errors and case in errors:
            gains = errors["base"] - errors[case]
            audit.checks += 1
            require(np.allclose(arrays["signed_normalized_coordinate_gain__" + case], gains, rtol=2e-12, atol=1e-14), "Signed gains differ")
            audit.equal(row["benefits"][case]["mean_signed_normalized_coordinate_gain"], float(np.mean(gains)))
            audit.equal(row["benefits"][case]["harmful_particle_fraction"], float(np.count_nonzero(gains < 0) / len(gains)))
        elif "benefits" in row:
            audit.equal(row["benefits"][case]["mean_signed_normalized_coordinate_gain"], None)
    audit.equal(row["random_envelope"], envelopes(values))
    return values


def aggregate(frames, expected, audit=None, recorded=None):
    wanted = {tuple(item[k] for k in IDENTITY): item for item in expected}
    require(len(wanted) == len(expected) and wanted, "Nonempty unique fixed schedule required")
    found = {tuple(item[k] for k in IDENTITY): item for item in frames}
    require(len(found) == len(frames) and set(found) == set(wanted), "Complete unique fixed frame population required")
    grouped = {}
    for key, item in wanted.items():
        grouped.setdefault(item["trajectory_id"], []).append(found[key])
    cached = {key: envelopes(row["values"]) for key, row in found.items()}
    def average(rows):
        return {m: {c: mean([row["values"][m][c] for row in rows]) for c in CASES} for m in METRICS}
    trajectories = {}
    for tid, items in sorted(grouped.items()):
        values = average(items)
        if recorded is not None:
            require(audit is not None, "Audit required before using recorded aggregate scalars")
            audit.equal(recorded["trajectories"][tid]["values"], values)
            values = recorded["trajectories"][tid]["values"]
        trajectories[tid] = dict(values=values, envelope_of_trajectory_mean_errors=envelopes(values),
            required_frames=len(items), missing_frames=0,
            defined_frames={m: {c: sum(row["values"][m][c] is not None for row in items) for c in CASES} for m in METRICS})
    values = average(list(trajectories.values()))
    if recorded is not None:
        audit.equal(recorded["values"], values)
        # Exact ranks use the recorded float64 aggregate after an independent
        # arithmetic check, as required by the declared equality tie rule.
        values = recorded["values"]
    per_frame = {m: {c: {field: mean([mean([cached[tuple(row[k] for k in IDENTITY)][m][c][field] for row in rows])
        for rows in grouped.values()]) for field in FIELDS} for c in CONTROLS} for m in METRICS}
    return dict(values=values, envelope_of_equal_trajectory_seed_mean_errors=envelopes(values),
        equal_trajectory_mean_of_frame_envelope_measures=per_frame, trajectories=trajectories,
        required_frames=len(expected), present_frames=len(frames), missing_frames=0, required_trajectories=len(grouped))


def execute(path, audit):
    path = Path(path)
    audit.pin(path)
    report = read(path)
    require(report["scope"] == "post-inspection exploratory native random-action envelope on original100k; not independent confirmation", "Wrong evidence family")
    status_path = path.parent / "status.json"
    audit.pin(status_path)
    status = read(status_path)
    audit.equal(status["state"], "complete")
    audit.equal(status["result_sha256"], sha(path))
    audit.equal(report["coverage"], dict(models=6, frames=2550, case_slots=33150, random_slots=20400))
    for name, digest in report["source_files_sha256"].items():
        audit.pin(name, digest)
    runs, published_runs = {}, {}
    for run in report["runs"]:
        require(type(run["seed"]) is int, "Original seed must be integer")
        key = run["objective"], run["seed"]
        require(key in COHORT and key not in runs, "Duplicate or unknown model")
        published_runs[key] = run
        for field in ("raw_files_sha256", "input_files_sha256"):
            for name, digest in run[field].items():
                audit.pin(name, digest)
        dirs = {Path(n).parent for n in run["raw_files_sha256"] if Path(n).name == "protocol.json"}
        require(len(dirs) == 1, "One raw job directory required")
        directory = dirs.pop()
        protocol, raw = read(directory / "protocol.json"), read(directory / "result.json")
        for filename in ("protocol.json", "result.json", "status.json"):
            audit.pin(directory / filename, run["raw_files_sha256"][str(directory / filename)])
        raw_status = read(directory / "status.json")
        audit.equal(raw_status["state"], "complete")
        audit.equal(raw_status["result_sha256"], sha(directory / "result.json"))
        audit.equal(raw["protocol_sha256"], sha(directory / "protocol.json"))
        audit.equal((protocol["objective"], protocol["seed"]), key)
        audit.equal(raw["state"], "complete")
        audit.equal(raw["checkpoint_sha256"], run["checkpoint_sha256"])
        require(len(raw["records"]) == 425 and len(protocol["expected_frames"]) == 425, "All425 frames required")
        frames = []
        failed = 0
        for index in raw["records"]:
            name = index["record_file"]
            require(Path(name).name == name, "Local frame filename required")
            record_path = directory / name
            audit.pin(record_path, index["record_sha256"])
            row = read(record_path)
            require(row["status"] in ("complete", "failed") and (row["status"] == "complete") == (row["failure"] is None), "Frame failure state differs")
            audit.equal(row["status"] == "failed", any(case["status"] == "failed" for case in row["cases"].values()))
            audit.equal(row["protocol_sha256"], sha(directory / "protocol.json"))
            audit.equal((row["objective"], row["seed"]), key)
            audit.equal(row["checkpoint_sha256"], run["checkpoint_sha256"])
            require(Path(row["array_file"]).name == row["array_file"], "Local array filename required")
            array_path = directory / row["array_file"]
            audit.pin(array_path, row["array_sha256"])
            require(type(row["source_index"]) is int and type(row["target_frame"]) is int, "Integer frame identity required")
            with np.load(array_path, allow_pickle=False) as arrays:
                values = raw_values(row, arrays, audit)
            frames.append({**{k: row[k] for k in IDENTITY}, "values": values})
            failed += int(row["status"] == "failed")
        audit.equal(run["failed_frames"], failed)
        audit.equal(run["complete_frames"], 425-failed)
        runs[key] = {}
        for split, count in (("valid", 128), ("test", 297)):
            subset = [row for row in frames if row["split"] == split]
            schedule = [row for row in protocol["expected_frames"] if row["split"] == split]
            require(len(subset) == len(schedule) == count, "Split population differs")
            displayed = run["splits"][split]
            derived = aggregate(subset, schedule, audit, displayed)
            audit.equal({k: displayed[k] for k in derived}, derived)
            published = {tuple(row[k] for k in IDENTITY): row for row in displayed["frames"]}
            require(len(published) == len(subset), "Published frame population differs")
            for row in subset:
                frame_record = published[tuple(row[k] for k in IDENTITY)]
                audit.equal(frame_record["values"], row["values"])
                audit.equal(frame_record["envelope"], envelopes(row["values"]))
            runs[key][split] = derived
    require(set(runs) == COHORT, "Exactly six original models required")
    for split in ("valid", "test"):
        for objective in ("faithful", "nll"):
            seeds = [runs[objective, seed][split] for seed in range(3)]
            displayed = report["summary"][split][objective]
            audit.equal(set(displayed["seeds"]), {"0", "1", "2"})
            for seed in range(3):
                audit.equal(displayed["seeds"][str(seed)], published_runs[objective, seed]["splits"][split])
            expected = {m: {c: stats([row["values"][m][c] for row in seeds]) for c in CASES} for m in METRICS}
            audit.equal(displayed["case_error_statistics"], expected)
            expected = {m: {c: stats([row["equal_trajectory_mean_of_frame_envelope_measures"][m][c]["reference_minus_random_mean"]
                for row in seeds]) for c in CONTROLS} for m in METRICS}
            audit.equal(displayed["control_minus_random_mean_statistics"], expected)
            expected = {m: {c: {f: stats([row["equal_trajectory_mean_of_frame_envelope_measures"][m][c][f]
                for row in seeds]) for f in FIELDS} for c in CONTROLS} for m in METRICS}
            audit.equal(displayed["mean_frame_comparison_statistics"], expected)
    for name, digest in list(audit.pins.items()):
        audit.equal(sha(name), digest)
    return dict(models=6, frames=2550, case_slots=33150, random_slots=20400)


def isolated_output(summary, output):
    report = read(summary)
    output = Path(output).resolve()
    protected = [Path(summary).resolve().parent, Path(__file__).resolve().parent]
    protected += [Path(n).resolve().parent for n in report["source_files_sha256"]]
    for run in report["runs"]:
        protected += [Path(n).resolve().parent for field in ("raw_files_sha256", "input_files_sha256") for n in run[field]]
    require(not output.exists() and not any(output == p or p in output.parents or output in p.parents for p in protected),
            "Fresh output outside immutable input directories required")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = isolated_output(args.summary, args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started, audit = time.perf_counter(), Audit()
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(), audit_source_sha256=sha(__file__),
        scope="Complementary raw coordinate-error/gain and complete random-envelope arithmetic; no graph/model/source-data admission or inference.")
    try:
        result.update(execute(args.summary, audit), passed=True)
    except BaseException as error:
        result.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(checks=audit.checks, wall_seconds=time.perf_counter()-started, input_files_sha256=audit.pins)
        (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
