"""Independent audit of completed physical-allocation artifacts only.

Does not import experiment modules, load checkpoints, read source datasets or
execute models. Every raw-frame hash is checked; ninety frames receive direct
raw-array checks. Original replay values are checked as embedded in the result.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE / "attempt_20261004"
REPO = HERE.parents[1] / "outputs/AdaptGNS"
CONTROLS = ("base", "dense", "random25", "speed25", "current_risk25", "lagged_base_risk25")
PHYSICAL = ("inverse_count25", "velocity_rms25")
POLICIES = CONTROLS + PHYSICAL
SCORED = ("speed25", "current_risk25", "lagged_base_risk25") + PHYSICAL
OBJECTIVES = ("faithful", "nll", "beta_nll")
counts, maxima, mismatches = Counter(), {}, []
alternate_rms_selection = []


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def check(kind, label, actual, expected, exact=False):
    counts[kind] += 1
    if actual is None or expected is None:
        good = actual is None and expected is None
    elif isinstance(expected, str):
        good = actual == expected
    else:
        a, b = np.asarray(actual), np.asarray(expected)
        if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
            good = a.shape == b.shape and np.array_equal(a, b)
        else:
            good = a.shape == b.shape and (np.array_equal(a, b) if exact else np.allclose(a, b, atol=2e-11, rtol=1e-11))
            if a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all():
                err = float(np.max(np.abs(a.astype(float)-b.astype(float))))
                maxima[kind] = max(maxima.get(kind, 0.0), err)
    if not good:
        mismatches.append({"kind": kind, "label": label, "actual": str(actual)[:160], "expected": str(expected)[:160]})


def pairs_set(value):
    return {tuple(map(int, row)) for row in value}


def required_stat(values):
    if any(v is None for v in values):
        return None, None
    mean = float(sum(values)/5)
    sd = float((sum((v-mean)**2 for v in values)/4)**.5)
    return mean, sd


def verify_stat(record, values, label):
    mean, sd = required_stat(values)
    for ix, (a, b) in enumerate(zip(values, record["seed_values"])):
        check("five_seed_statistics", label+f" seed {ix}", a, b)
    check("five_seed_statistics", label+" length", len(record["seed_values"]), 5, exact=True)
    check("five_seed_statistics", label+" required", record["required_seeds"], 5, exact=True)
    check("five_seed_statistics", label+" defined", sum(v is not None for v in values), record["defined_seeds"], exact=True)
    check("five_seed_statistics", label+" mean", mean, record["mean"])
    check("five_seed_statistics", label+" sample SD", sd, record["sample_seed_sd"])


def ranked_pairs(extra, scores, budget):
    # Python tuple ordering makes the score and particle-ID tie rule explicit.
    order = sorted(range(len(extra)), key=lambda j: (-float(max(scores[extra[j, 0]], scores[extra[j, 1]])), int(extra[j, 0]), int(extra[j, 1])))
    return extra[np.asarray(order[:budget], dtype=np.int64)]


def check_raw(row, arrays, random_indices, label):
    n, bcount, ecount, budget = (row[k] for k in ("n_particles", "base_pairs", "available_extra_pairs", "extra_pair_budget"))
    base, extra, velocity = (arrays[k] for k in ("base_pairs", "available_extra_pairs", "observed_velocity"))
    for key, pairs, count in (("base", base, bcount), ("annulus", extra, ecount)):
        check("raw_graph", label+" "+key+" shape", pairs.shape, (count, 2), exact=True)
        check("raw_graph", label+" "+key+" integer", pairs.dtype.kind in "iu", True, exact=True)
        check("raw_graph", label+" "+key+" range", bool(np.all((pairs[:, 0] >= 0) & (pairs[:, 0] < pairs[:, 1]) & (pairs[:, 1] < n))), True, exact=True)
        codes = pairs[:, 0]*n+pairs[:, 1]
        check("raw_graph", label+" "+key+" uniqueness/order", bool(np.all(np.diff(codes) > 0)), True, exact=True)
    check("raw_graph", label+" disjoint", len(pairs_set(base) & pairs_set(extra)), 0, exact=True)
    check("physical_score", label+" velocity shape", velocity.shape, (n, 2), exact=True)
    check("physical_score", label+" finite velocity", bool(np.isfinite(velocity).all()), True, exact=True)
    degree = np.bincount(base.ravel(), minlength=n)
    edge_squared = np.sum((velocity[base[:, 1]]-velocity[base[:, 0]])**2, axis=1)
    totals = np.bincount(base[:, 0], weights=edge_squared, minlength=n) + np.bincount(base[:, 1], weights=edge_squared, minlength=n)
    direct_rms = np.sqrt(np.divide(totals, degree, out=np.zeros(n), where=degree != 0))
    for key, value in (("neighbor_count", degree), ("negative_neighbor_count", -degree.astype(float)), ("isolated", degree == 0)):
        check("physical_score", label+" "+key, value, arrays[key], exact=True)
    check("physical_score", label+" velocity RMS", direct_rms, arrays["velocity_dispersion"])
    check("physical_score", label+" isolated count", int((degree == 0).sum()), row["isolated_particles"], exact=True)
    scores = {"speed25": arrays["speed"], "current_risk25": arrays["current_risk"],
              "lagged_base_risk25": arrays["previous_base_risk"],
              "inverse_count25": arrays["negative_neighbor_count"], "velocity_rms25": arrays["velocity_dispersion"]}
    selected = {policy: arrays[policy+"_selected_extra_pairs"] for policy in POLICIES}
    for policy in POLICIES:
        target_count = 0 if policy == "base" else ecount if policy == "dense" else budget
        actual = selected[policy]
        check("selected_graph", label+" "+policy+" shape", actual.shape, (target_count, 2), exact=True)
        check("selected_graph", label+" "+policy+" unique", len(pairs_set(actual)), target_count, exact=True)
        check("selected_graph", label+" "+policy+" membership", pairs_set(actual).issubset(pairs_set(extra)), True, exact=True)
        if policy == "base":
            expected = np.empty((0, 2), dtype=np.int64)
        elif policy == "dense":
            expected = extra
        elif policy == "random25":
            expected = extra[random_indices]
        else:
            expected = ranked_pairs(extra, scores[policy], budget)
        check("selection_and_ID_ties", label+" "+policy, actual, expected, exact=True)
        errors = arrays[policy+"_vector_se"]
        check("raw_error_mean", label+" "+policy+" shape", errors.shape, (n,), exact=True)
        check("raw_error_mean", label+" "+policy+" finite nonnegative", bool(np.all(np.isfinite(errors) & (errors >= 0))), True, exact=True)
        check("raw_error_mean", label+" "+policy+" coordinate MSE", float(np.sum(errors)/(2*n)), row["policies"][policy]["mse_normalized_acceleration"])
    # Retain whether algebraically equivalent RMS arithmetic would move any
    # pair at a floating-point tie; frozen stored-score selection remains the
    # authoritative executable policy, independently checked above.
    alternative = ranked_pairs(extra, direct_rms, budget)
    change = len(pairs_set(alternative) ^ pairs_set(selected["velocity_rms25"]))
    if change:
        alternate_rms_selection.append({"frame": label, "symmetric_pair_difference": change})
    for policy in SCORED:
        score, chosen = scores[policy], selected[policy]
        priority = np.maximum(score[extra[:, 0]], score[extra[:, 1]])
        record = row["cutoff_ties"][policy]
        if budget:
            cutoff = sorted(priority.tolist(), reverse=True)[budget-1]
            all_tied = extra[priority == cutoff]
            chosen_tied = chosen[np.maximum(score[chosen[:, 0]], score[chosen[:, 1]]) == cutoff]
            higher = int((priority > cutoff).sum())
        else:
            cutoff, higher = None, 0
            all_tied = chosen_tied = np.empty((0, 2), dtype=np.int64)
        for key, value in (("cutoff", cutoff), ("strictly_higher_pairs", higher), ("all_tied_pairs", len(all_tied)), ("selected_tied_slots", len(chosen_tied)), ("split_cutoff_tie", len(all_tied) > len(chosen_tied))):
            check("cutoff_tie_audit", label+" "+policy+" "+key, value, record[key], exact=True)
        check("cutoff_tie_audit", label+" "+policy+" all tied identities", all_tied, arrays[policy+"_all_cutoff_tied_pairs"], exact=True)
        check("cutoff_tie_audit", label+" "+policy+" chosen tied identities", chosen_tied, arrays[policy+"_selected_cutoff_tied_pairs"], exact=True)
    for policy in PHYSICAL:
        a = pairs_set(selected[policy])
        for control in CONTROLS:
            b = pairs_set(selected[control])
            shared, union = len(a & b), len(a | b)
            expected = {"shared_optional_pairs": shared, "physical_optional_pairs": len(a), "control_optional_pairs": len(b),
                        "fraction_of_physical": shared/len(a) if a else None, "jaccard": shared/union if union else 1.0}
            for key, value in expected.items():
                check("pair_overlap", label+" "+policy+" "+control+" "+key, value, row["physical_control_overlap"][policy][control][key])


def audit():
    source = ATTEMPT / "summary.json"
    data = json.loads(source.read_text())
    expected_runs = {f"{objective}_seed{seed}" for objective in OBJECTIVES for seed in range(5)}
    check("completion", "attempt", data["state"], "complete")
    check("completion", "summaries validated", data["summaries_validated"], True, exact=True)
    check("completion", "actual model identities", sorted(data["runs"]), sorted(expected_runs), exact=True)
    check("completion", "declared model identities", sorted(data["expected_runs"]), sorted(expected_runs), exact=True)
    check("completion", "failure key absent", "failure" in data, False, exact=True)
    for name, value in data["source_sha256"].items():
        check("hash", name, digest(REPO / "research" / name), value)
    check("hash", "analysis protocol", digest(REPO / "research/protocols/physical_allocation_pilot.md"), data["protocol_sha256"])
    manifest_path = REPO / "research/protocols/physical_allocation_inputs.json"
    check("hash", "input identity", digest(manifest_path), data["input_identity_sha256"])
    manifest = json.loads(manifest_path.read_text())
    check("input_identity", "embedded manifest", json.dumps(data["input_identity"], sort_keys=True), json.dumps(manifest, sort_keys=True))
    canonical = data["runs"]["faithful_seed0"]
    sample_count, raw_count = 0, 0
    seed_values = {}
    for name, run in sorted(data["runs"].items()):
        check("completion", name, run["state"], "complete")
        check("input_identity", name+" checkpoint digest", run["checkpoint_sha256"], manifest["files"][f"research/results/waterdrop_pilot/{name}.pt"])
        check("input_identity", name+" control digest", run["original_controls_sha256"], manifest["files"][f"research/results/waterdrop_pilot/{name}.json"])
        for split in ("valid", "test"):
            section = run["splits"][split]
            rows = section["frames"]
            check("frame_identity", name+" "+split+" frame count", len(rows), 36, exact=True)
            check("frame_identity", name+" "+split+" unique IDs", len({r["id"] for r in rows}), 36, exact=True)
            geometry_fields = ("id", "trajectory", "step", "n_particles", "base_pairs", "available_extra_pairs", "extra_pair_budget", "isolated_particles")
            random = np.random.default_rng(91300+run["seed"])
            for index, row in enumerate(rows):
                label = name+" "+split+" "+row["id"]
                canonical_row = canonical["splits"][split]["frames"][index]
                for field in geometry_fields:
                    check("frame_identity", label+" "+field, row[field], canonical_row[field], exact=True)
                check("frame_identity", label+" parsed ID", row["id"], f"{row['trajectory']}:{row['step']}")
                check("frame_identity", label+" raw prefix", row["raw_prefix"], f"{split}_{index:03d}")
                check("frame_scalar", label+" budget", row["extra_pair_budget"], row["available_extra_pairs"]//4, exact=True)
                check("frame_scalar", label+" policies", sorted(row["policies"]), sorted(POLICIES), exact=True)
                for policy, m in row["policies"].items():
                    ec = 0 if policy == "base" else row["available_extra_pairs"] if policy == "dense" else row["extra_pair_budget"]
                    check("frame_scalar", label+" "+policy+" edges", m["directed_edges"], 2*(row["base_pairs"]+ec), exact=True)
                    check("frame_scalar", label+" "+policy+" valid MSE", bool(np.isfinite(m["mse_normalized_acceleration"]) and m["mse_normalized_acceleration"] >= 0), True, exact=True)
                check("control_replay", label+" controls", sorted(row["control_replay"]), sorted(CONTROLS), exact=True)
                for policy, replay in row["control_replay"].items():
                    new = row["policies"][policy]
                    check("control_replay", label+" "+policy+" actual MSE", replay["actual_mse"], new["mse_normalized_acceleration"])
                    check("control_replay", label+" "+policy+" difference", replay["actual_mse"]-replay["original_mse"], replay["mse_difference"])
                    check("control_replay", label+" "+policy+" exact edges", replay["actual_directed_edges"], replay["original_directed_edges"], exact=True)
                    check("control_replay", label+" "+policy+" measured edges", replay["actual_directed_edges"], new["directed_edges"], exact=True)
                    # Direct inequality, independent of the producer's isclose.
                    check("control_replay", label+" "+policy+" tolerance", abs(replay["actual_mse"]-replay["original_mse"]) <= 1e-7+2e-5*abs(replay["original_mse"]), True, exact=True)
                raw_path = (ATTEMPT / row["raw_file"]["path"]).resolve()
                if not raw_path.is_relative_to(ATTEMPT.resolve()):
                    raise ValueError("Raw-frame pointer leaves attempt directory")
                check("raw_frame_hash", label, digest(raw_path), row["raw_file"]["sha256"])
                raw_count += 1
                indices = random.permutation(row["available_extra_pairs"])[:row["extra_pair_budget"]]
                if index in (0, 17, 35):
                    with np.load(raw_path, allow_pickle=False) as arrays:
                        check_raw(row, arrays, indices, label)
                    sample_count += 1
            names = sorted({r["trajectory"] for r in rows})
            check("aggregation", name+" "+split+" trajectory count", len(names), 3, exact=True)
            calculated = {}
            for policy in POLICIES:
                trajectory_values = {}
                for trajectory in names:
                    selected = [r for r in rows if r["trajectory"] == trajectory]
                    check("aggregation", name+" "+split+" "+trajectory+" count", len(selected), 12, exact=True)
                    expected = {key: sum(r["policies"][policy][key] for r in selected)/12 for key in ("mse_normalized_acceleration", "directed_edges")}
                    trajectory_values[trajectory] = expected
                    for key, value in expected.items():
                        check("aggregation", name+" "+split+" "+trajectory+" "+policy+" "+key, value, section["trajectories"][trajectory]["policies"][policy][key])
                for key in ("mse_normalized_acceleration", "directed_edges"):
                    expected = sum(trajectory_values[t][key] for t in names)/3
                    check("aggregation", name+" "+split+" "+policy+" "+key, expected, section["equal_trajectory_mean"]["policies"][policy][key])
                    if key == "mse_normalized_acceleration":
                        calculated[policy] = expected
            seed_values[name, split] = calculated
    for group, summary in data["summaries"].items():
        objective, split = group.split("/")
        policy_values = {policy: [seed_values[f"{objective}_seed{seed}", split][policy] for seed in range(5)] for policy in POLICIES}
        for policy in POLICIES:
            verify_stat(summary["policies"][policy], policy_values[policy], group+" "+policy)
        for physical in PHYSICAL:
            for control in CONTROLS:
                difference = [a-b for a, b in zip(policy_values[physical], policy_values[control])]
                percent = [100*(a-b)/b if b != 0 else None for a, b in zip(policy_values[physical], policy_values[control])]
                saved = summary["physical_minus_control"][physical][control]
                verify_stat(saved["difference"], difference, group+" "+physical+" minus "+control)
                verify_stat(saved["percent_difference"], percent, group+" "+physical+" percent "+control)
                check("five_seed_statistics", group+" "+physical+" better than "+control, sum(v < 0 for v in difference), saved["seeds_with_lower_mse"], exact=True)
    check("completion", "hashed frame population", raw_count, 1080, exact=True)
    check("completion", "raw checked sample", sample_count, 90, exact=True)
    return {"summary_sha256": digest(source), "raw_frames_hashed": raw_count, "raw_frames_numerically_checked": sample_count,
            "checkpoint_and_source_dataset_files_opened": 0,
            "elapsed_experiment_seconds": data["elapsed_seconds"]}


result = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
          "scope": "Independent final-artifact audit; no models, checkpoint files or source datasets opened",
          "frozen_experiment_commit": "3d1320b", "audit_source_sha256": digest(__file__),
          "limitations": ["Original replay MSE references are checked as embedded in the final result; original model/control execution is not repeated.",
                          "Graph radii and target residuals are not recomputed from source positions/targets; raw graph integrity and per-particle error means are checked.",
                          "All 1080 raw hashes and scalar hierarchies are verified; raw numerical calculations use predetermined indexes 0,17,35 per model/split (90 frames)."],
          "methods": {"scores": "Direct endpoint bincount and unscaled squared velocity differences, independent of overflow-scaled implementation",
                      "top_k": "Python tuple sorting by negative max endpoint score, then pair IDs; raw frozen scores define floating-point ordering",
                      "ties_overlaps": "Direct pair-set calculations and exact cutoff equality",
                      "aggregation": "Independent sum/12, sum/3, sum/5 and sum squared deviations/4",
                      "replay": "All 6480 embedded original comparisons; exact edge counts and explicit atol+rtol*abs(original) inequality"},
          "tolerances": {"atol": 2e-11, "rtol": 1e-11, "hashes_graphs_ids_counts_ties": "exact"},
          "software": {"python": platform.python_version(), "numpy": np.__version__}}
try:
    result.update(audit())
    result["state"] = "passed" if not mismatches else "mismatches"
except BaseException as error:
    result["state"] = "audit_failed"
    result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
finally:
    result.update({"counts": dict(counts), "total_checks": sum(counts.values()),
                   "maximum_absolute_discrepancy_by_kind": maxima, "mismatch_count": len(mismatches), "mismatches": mismatches,
                   "alternate_arithmetic_rms_selection_changes": alternate_rms_selection})
    (HERE / "independent_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps({key: result[key] for key in ("state", "total_checks", "mismatch_count", "counts", "maximum_absolute_discrepancy_by_kind", "alternate_arithmetic_rms_selection_changes")}, indent=2))
if result["state"] != "passed":
    raise SystemExit(1)
