"""Independent NumPy audit. No production analysis or action arithmetic imports."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np

POLICIES = ("base", "dense", "random25", "speed25", "previous-observed-base-risk25")
SPARSE = POLICIES[2:]
PAIRS = {"risk_minus_random": (POLICIES[-1], "random25"), "risk_minus_speed": (POLICIES[-1], "speed25"), "speed_minus_random": ("speed25", "random25")}
PRIMARY = ("neither", "left_only", "right_only", "both")
BOTH = ("both_less", "both_equal", "both_more")
GROUPS, Q = PRIMARY + BOTH, ("error", "alignment", "cost", "degree")
SOURCES, TARGETS = tuple(range(3, 30)), (7, 106, 205, 304, 403, 502, 601, 700, 799, 898, 1000)
KEYS = tuple(itertools.product(SOURCES, TARGETS))
MODELS = tuple(f"{obj}_seed{s}" for obj in ("faithful", "nll") for s in range(3))
WHOLE = ("left_error", "right_error", "error_difference", "alignment_difference", "cost_difference", "degree_difference", "arithmetic_scale")
STATS = ("particle_fraction",) + tuple(q + "_contribution" for q in Q) + tuple("conditional_mean_" + q + "_difference" for q in Q)
NAMES = tuple(f"{p}/whole/{k}" for p in PAIRS for k in WHOLE) + tuple(f"{p}/{g}/{k}" for p in PAIRS for g in GROUPS for k in STATS)
ANCHOR = {"results.json": "c9a608d3f17cfc84001020b27048ef448958daa2705d135be5f4492e8bbdc008",
 "input_identity.json": "647c94d886a9d8ee6f2788615c6b5fb95758ff97ca0f94d8ec69a10f9ee41718",
 "status.json": "04a221181efc1fdb27bc6b2c24712bde280429c3bc257cc060be402b7b2d262a"}
EPS = np.finfo(np.float64).eps


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None


def magnitude(key, metrics):
    if "fraction" in key or "degree" in key: return max(1., abs(metrics.get(key) or 0.))
    return metrics.get(key.split("/")[0] + "/whole/arithmetic_scale") or max(abs(metrics.get(key) or 0.), 1e-300)


class Audit:
    def __init__(self):
        self.checks, self.max_difference, self.files = 0, 0., {}
    def check(self, ok, context):
        self.checks += 1
        if not ok: raise AssertionError(context)
    def file(self, path, expected=None):
        path = Path(path).resolve()
        self.check(path.is_file(), "Missing file: " + str(path))
        value = sha(path)
        self.check(str(path) not in self.files or self.files[str(path)] == value, "Changed during audit: " + str(path))
        if expected is not None: self.check(value == expected, "File hash differs: " + str(path))
        self.files[str(path)] = value
        return value
    def same(self, a, b, context, scale=1., terms=1, exact=False):
        if a is None or b is None:
            self.check(a is None and b is None, context + ": null mismatch"); return
        a, b = np.asarray(a), np.asarray(b)
        self.check(a.shape == b.shape and a.dtype.kind in "biuf" and b.dtype.kind in "biuf", context + ": shape/type")
        self.check(np.isfinite(a).all() and np.isfinite(b).all(), context + ": nonfinite")
        delta = np.abs(a.astype(float) - b.astype(float)); self.checks += a.size
        if delta.size: self.max_difference = max(self.max_difference, float(delta.max()))
        bound = 256 * EPS * max(terms, 1) * np.maximum(scale, 1e-300)
        self.check(np.array_equal(a, b) if exact else np.all(delta <= bound + 1e-12 * np.abs(b)), context + ": values differ")
    def path(self, root, relative):
        raw = Path(root) / relative; path = raw.resolve()
        self.check(path.is_relative_to(Path(root).resolve()) and not raw.is_symlink(), "Unsafe evidence path")
        return path


def pairset(value, n, audit, context):
    v = np.asarray(value)
    audit.check(v.ndim == 2 and v.shape[1] == 2 and v.dtype.kind in "iu", context + ": pair shape")
    audit.check(np.all(v >= 0) and np.all(v < n) and np.all(v[:, 0] < v[:, 1]), context + ": simple bounded pairs")
    result = set(map(tuple, v.tolist()))
    audit.check(len(result) == len(v), context + ": unique pairs")
    return result


def core(quantities, base_error, derived, audit, context):
    n = len(base_error); metrics, reasons, groups, identities = dict.fromkeys(NAMES), {}, {}, []
    for pair, (left, right) in PAIRS.items():
        absent = [p for p in (left, right) if p not in quantities]
        if absent:
            groups[pair] = {"status": "missing_action", "actions": absent}
            for k in NAMES:
                if k.startswith(pair + "/"): reasons[k] = "missing_action:" + ",".join(absent)
            continue
        a, b = quantities[left], quantities[right]
        difference = {q: a[q] - b[q] for q in Q}
        audit.check(int(a["degree"].sum()) == int(b["degree"].sum()), context + ": equal budgets")
        ids = (a["degree"] > 0).astype(np.int8) + 2 * (b["degree"] > 0).astype(np.int8)
        subids = np.where(ids == 3, np.sign(a["degree"] - b["degree"]) + 1, -1).astype(np.int8)
        masks = {g: ids == j for j, g in enumerate(PRIMARY)}
        masks.update({g: subids == j for j, g in enumerate(BOTH)})
        audit.check(np.all(np.sum([masks[g] for g in PRIMARY], axis=0) == 1), context + ": primary partition")
        audit.check(np.array_equal(np.sum([masks[g] for g in BOTH], axis=0), masks["both"].astype(int)), context + ": both partition")
        scale = base_error + a["error"] + b["error"] + a["cost"] + b["cost"] + np.abs(a["alignment"]) + np.abs(b["alignment"])
        residual = np.abs(difference["error"] - (difference["cost"] - difference["alignment"]))
        bound = 256 * EPS * 2 * np.maximum(scale, 1e-300)
        audit.check(np.all(residual <= bound), context + ": per-particle cost-alignment identity")
        identities.append({"name": pair + "/particle_identity", "terms": 2, "max_abs_residual": float(residual.max()), "max_absolute_bound": float(bound.max())})
        whole = {"left_error": mean(a["error"]), "right_error": mean(b["error"]), "arithmetic_scale": mean(scale), **{q + "_difference": mean(v) for q, v in difference.items()}}
        audit.check(whole["degree_difference"] == 0., context + ": degree zero-sum")
        metrics.update({f"{pair}/whole/{k}": v for k, v in whole.items()})
        groups[pair] = {"status": "complete", "left": left, "right": right, "groups": {}}
        for group, mask in masks.items():
            count = int(np.count_nonzero(mask)); values = {"particles": count, "particle_fraction": count / n}
            for q, v in difference.items():
                values[q + "_contribution"] = math.fsum(v[mask]) / n
                values["conditional_mean_" + q + "_difference"] = mean(v[mask]) if count else None
            groups[pair]["groups"][group] = values
            for k in STATS:
                key = f"{pair}/{group}/{k}"; metrics[key] = values[k]
                if values[k] is None: reasons[key] = "empty_group"
        derived.update({pair + "__" + q + "_difference": v for q, v in difference.items()})
        derived[pair + "__primary_group_id"], derived[pair + "__both_degree_subgroup_id"] = ids, subids
    return {"metrics": metrics, "undefined_reasons": reasons, "pairs": groups, "derived": derived, "particle_checks": identities,
            "status": "partial" if any(g["status"] != "complete" for g in groups.values()) else "complete"}


def reconstruct(raw, arrays, audit, context):
    n = raw["n_particles"]; target = np.asarray(arrays["target_position"], dtype=float)
    std = np.asarray(raw["saved_acceleration_normalization"]["std"], dtype=float)
    audit.check(target.shape == (n, 2) and np.isfinite(target).all() and ahash(arrays["target_position"]) == raw["target_sha256"], context + ": target")
    audit.check(std.shape == (2,) and np.isfinite(std).all() and np.all(std > 0), context + ": positive coordinate scale")
    for key, digest_key in (("current_observed_history", "observed_history_sha256"), ("previous_observed_history", "previous_observed_history_sha256")):
        audit.check(arrays[key].shape == (6, n, 2) and arrays[key].dtype == np.float32 and ahash(arrays[key]) == raw[digest_key], context + ": observed history")
    audit.check(arrays["particle_types"].shape == (n,) and not np.any(arrays["particle_types"] == 3), context + ": types")
    predictions = {p: np.asarray(arrays["prediction_" + p], dtype=float) for p in POLICIES}
    base = predictions["base"]; residual = (target - base) / std
    norm2 = lambda x: np.einsum("ij,ij->i", x, x)
    base_error = norm2(residual) / 2
    derived = {"normalized_base_residual": residual, "normalized_coordinate_error_base": base_error}
    base_pairs = pairset(arrays["candidate_base_pairs_float64"], n, audit, context + "/base")
    extra = pairset(arrays["candidate_annulus_pairs_float64"], n, audit, context + "/annulus")
    audit.check(not base_pairs & extra, context + ": disjoint pair sets")
    graph = raw["graph_audit"]
    audit.check(graph["base_pairs"] == len(base_pairs) and graph["annulus_pairs"] == len(extra) and graph["optional_budget"] == len(extra) // 4 and graph["candidate_pairs"] == len(base_pairs | extra), context + ": candidate counts")
    quantities = {}
    for policy, prediction in predictions.items():
        audit.check(prediction.shape == (n, 2) and np.isfinite(prediction).all(), context + ": finite prediction")
        error = norm2((prediction - target) / std) / 2
        for unit, vector in (("normalized", error * 2), ("position", norm2(prediction - target))):
            audit.same(arrays[f"{unit}_vector_se_{policy}"], vector, context + ": original vector error", scale=np.maximum(vector, 1e-300))
            audit.same(raw["accuracy"][policy][unit + "_coordinate_mse"], mean(vector) / 2, context + ": original coordinate MSE", scale=max(float(vector.max()), 1e-300), terms=n)
        selected = pairset(arrays["pairs_" + policy], n, audit, context + "/" + policy)
        audit.check(base_pairs <= selected <= base_pairs | extra, context + ": mandatory/candidate inclusion")
        optional = selected - base_pairs
        budget = 0 if policy == "base" else len(extra) if policy == "dense" else len(extra) // 4
        audit.check(len(optional) == budget, context + ": exact selected budget")
        counts = graph["policies"][policy]
        audit.check(counts["retained_pairs"] == len(selected) and counts["retained_optional_pairs"] == budget and counts["directed_edges"] == 2 * len(selected), context + ": graph counts")
        first = next(row for row in raw["timed_calls"] if row["method"] == policy)
        audit.check(ahash(arrays["pairs_" + policy]) == first["pair_sha256"], context + ": first timed mask")
        if policy not in SPARSE: continue
        endpoints = np.asarray(sorted(optional), dtype=np.int64).reshape(-1)
        degree = np.bincount(endpoints, minlength=n).astype(np.int64)
        audit.check(int(degree.sum()) == 2 * budget, context + ": degree incidence")
        delta = (prediction - base) / std
        alignment, cost = np.einsum("ij,ij->i", residual, delta), norm2(delta) / 2
        quantities[policy] = {"error": error, "alignment": alignment, "cost": cost, "degree": degree}
        derived.update({"normalized_prediction_change_" + policy: delta, "normalized_coordinate_error_" + policy: error,
            "normalized_coordinate_alignment_" + policy: alignment, "normalized_coordinate_cost_" + policy: cost, "optional_degree_" + policy: degree})
    return core(quantities, base_error, derived, audit, context)


def partition(metrics, terms, audit, context):
    checks = []
    def add(name, a, b, scale):
        bound = 256 * EPS * max(terms, 1) * max(scale, 1e-300); residual = abs(a - b)
        audit.check(residual <= bound, context + ": " + name)
        checks.append({"name": name, "terms": terms, "max_abs_residual": residual, "max_absolute_bound": bound})
    for p in PAIRS:
        if metrics[p + "/whole/error_difference"] is None: continue
        scale = metrics[p + "/whole/arithmetic_scale"]
        for q in Q:
            mag = max(1., abs(metrics[p + "/whole/degree_difference"])) if q == "degree" else scale
            add(f"{p}/{q}/four_groups", math.fsum(metrics[f"{p}/{g}/{q}_contribution"] for g in PRIMARY), metrics[f"{p}/whole/{q}_difference"], mag)
            add(f"{p}/{q}/both_subgroups", math.fsum(metrics[f"{p}/{g}/{q}_contribution"] for g in BOTH), metrics[f"{p}/both/{q}_contribution"], mag)
        add(p + "/fractions", math.fsum(metrics[f"{p}/{g}/particle_fraction"] for g in PRIMARY), 1., 1.)
        add(p + "/both_fractions", math.fsum(metrics[f"{p}/{g}/particle_fraction"] for g in BOTH), metrics[f"{p}/both/particle_fraction"], 1.)
        for g in GROUPS:
            add(f"{p}/{g}/cost_minus_alignment", metrics[f"{p}/{g}/error_contribution"], metrics[f"{p}/{g}/cost_contribution"] - metrics[f"{p}/{g}/alignment_contribution"], scale)
        add(p + "/whole/cost_minus_alignment", metrics[p + "/whole/error_difference"], metrics[p + "/whole/cost_difference"] - metrics[p + "/whole/alignment_difference"], scale)
    return checks


def check_identities(saved, expected, audit, context):
    lookup = {r["name"]: r for r in saved}
    audit.check(len(lookup) == len(saved) == len(expected) and set(lookup) == {r["name"] for r in expected}, context + ": identity inventory")
    for ref in expected:
        row = lookup[ref["name"]]
        audit.check(row["terms"] == ref["terms"] and 0 <= row["max_abs_residual"] <= row["max_absolute_bound"], context + ": declared identity")
        audit.same(row["max_absolute_bound"], ref["max_absolute_bound"], context + ": identity bound", scale=ref["max_absolute_bound"])
        audit.check(abs(row["max_abs_residual"] - ref["max_abs_residual"]) <= ref["max_absolute_bound"], context + ": independent reduction residue")


def check_frame(saved, expected, arrays, audit, context):
    n = saved["n_particles"]
    audit.check(saved["status"] == expected["status"] and set(saved["metrics"]) == set(NAMES), context + ": status/metrics")
    audit.check(saved["undefined_reasons"] == expected["undefined_reasons"], context + ": undefined reasons")
    for k, v in expected["metrics"].items(): audit.same(saved["metrics"][k], v, context + ": " + k, scale=magnitude(k, expected["metrics"]), terms=n)
    audit.check(set(saved["pairs"]) == set(PAIRS), context + ": comparisons")
    for p, ref in expected["pairs"].items():
        row = saved["pairs"][p]
        if ref["status"] != "complete":
            audit.check(row == ref, context + ": missing action"); continue
        audit.check(all(row[k] == ref[k] for k in ("status", "left", "right")) and set(row["groups"]) == set(GROUPS), context + ": comparison/group identities")
        for g in GROUPS:
            audit.check(row["groups"][g]["particles"] == ref["groups"][g]["particles"], context + ": exact group count")
            for k in STATS: audit.same(row["groups"][g][k], ref["groups"][g][k], context + ": nested group", scale=magnitude(f"{p}/{g}/{k}", expected["metrics"]), terms=n)
    audit.check(set(arrays) == set(expected["derived"]), context + ": derived array names")
    for k, v in expected["derived"].items():
        audit.check(arrays[k].dtype == v.dtype, context + ": array dtype")
        audit.same(arrays[k], v, context + ": " + k, scale=np.maximum(np.abs(v), 1e-300), exact=v.dtype.kind in "iu")
    check_identities(saved["identity_checks"], expected["particle_checks"] + partition(expected["metrics"], n, audit, context), audit, context)


def weighted(metrics):
    result = {}
    for p, g, q in itertools.product(PAIRS, GROUPS, Q):
        f, c = metrics[f"{p}/{g}/particle_fraction"], metrics[f"{p}/{g}/{q}_contribution"]
        reason = "missing_required_input" if f is None or c is None else "zero_weighted_group_fraction" if f == 0 else None
        result[f"{p}/{g}/{q}"] = {"value": c / f if reason is None else None, "reason": reason, "weighted_particle_fraction": f, "weighted_contribution": c}
    return result


def compare_weighted(saved, expected, audit, context):
    audit.check(set(saved) == set(expected), context + ": ratio inventory")
    for name, ref in expected.items():
        audit.check(saved[name]["reason"] == ref["reason"], context + ": ratio null reason")
        for k in ("value", "weighted_particle_fraction", "weighted_contribution"):
            audit.same(saved[name][k], ref[k], context + ": weighted " + k, scale=max(abs(ref[k] or 0.), 1e-300), terms=297)


def aggregate(frames, sources=SOURCES, targets=TARGETS):
    trajectories = {}
    for source in sources:
        rows = [frames[source, target] for target in targets]
        metrics = {k: mean(row["metrics"][k] for row in rows) for k in NAMES}
        trajectories[str(source)] = {"metrics": metrics, "weighted_conditionals": weighted(metrics), "required_frames": len(targets),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "undefined_reason_counts": {k: dict(Counter(row["undefined_reasons"][k] for row in rows if row["metrics"][k] is None)) for k in NAMES}}
    metrics = {k: mean(t["metrics"][k] for t in trajectories.values()) for k in NAMES}
    coverage = {k: {"defined_frames": sum(row["metrics"][k] is not None for row in frames.values()),
        "defined_trajectories": sum(t["metrics"][k] is not None for t in trajectories.values()),
        "undefined_reason_counts": dict(Counter(row["undefined_reasons"][k] for row in frames.values() if row["metrics"][k] is None))} for k in NAMES}
    return {"metrics": metrics, "weighted_conditionals": weighted(metrics), "trajectories": trajectories,
            "metric_coverage": coverage, "frame_status_counts": dict(Counter(row["status"] for row in frames.values()))}


def check_aggregate(saved, expected, audit, context):
    audit.check(saved["required_frames"] == saved["observed_frames"] == 297 and saved["required_trajectories"] == 27, context + ": counts")
    audit.check(saved["metric_coverage"] == expected["metric_coverage"] and saved["frame_status_counts"] == expected["frame_status_counts"], context + ": exact missing/empty coverage")
    audit.check(set(saved["trajectories"]) == set(expected["trajectories"]), context + ": trajectories")
    for a, b, terms in [(saved, expected, 297)] + [(saved["trajectories"][s], expected["trajectories"][s], 11) for s in expected["trajectories"]]:
        for k, v in b["metrics"].items(): audit.same(a["metrics"][k], v, context + ": hierarchical " + k, scale=magnitude(k, b["metrics"]), terms=terms)
        compare_weighted(a["weighted_conditionals"], b["weighted_conditionals"], audit, context)
        check_identities(a["identity_checks"], partition(b["metrics"], terms, audit, context), audit, context)
        if terms == 11:
            for k in ("required_frames", "status_counts", "undefined_reason_counts"): audit.check(a[k] == b[k], context + ": trajectory null counts")


def check_seeds(saved, values, audit, context):
    audit.check(saved["required_seeds"] == 3 and saved["defined_seeds"] == sum(v is not None for v in values) and len(saved["seed_values"]) == 3, context + ": all three seeds")
    for a, b in zip(saved["seed_values"], values): audit.same(a, b, context + ": seed value", scale=max(abs(b or 0.), 1e-300), terms=297)
    average = mean(values)
    sd = math.sqrt(math.fsum((v - average) ** 2 for v in values) / 2) if average is not None else None
    audit.same(saved["mean"], average, context + ": mean", scale=max(abs(average or 0.), 1e-300), terms=297)
    audit.same(saved["sample_seed_sd"], sd, context + ": sampleSD", scale=max(abs(sd or 0.), 1e-300), terms=297)


def execute(args, audit):
    root, repo = args.analysis_root.resolve(), args.repo.resolve()
    status, result, identity = read(root / "status.json"), read(root / "results.json"), read(root / "input_identity.json")
    audit.file(root / "results.json", status["results_sha256"]); audit.file(root / "report.md", status["report_sha256"])
    inventory = {str(p.relative_to(root)): audit.file(p) for p in root.rglob("*") if p.is_file()}
    audit.check(status["state"] == result["state"] and result["state"] in ("complete", "incomplete"), "Final publication")
    audit.check(result["required_models"] == 6 and result["required_frames_per_model"] == 297 and result["required_total_frames"] == 1782, "Fixed population")
    audit.check(result["comparison_order"] == {k: list(v) for k, v in PAIRS.items()} and result["primary_groups"] == list(PRIMARY) and result["supplementary_both_subgroups"] == list(BOTH), "Comparison/group definitions")
    audit.file(repo / "research/analyze_optional_exposure_decomposition.py", args.expected_source_sha256)
    audit.file(repo / "research/protocols/optional_exposure_decomposition_20261006.md", args.expected_protocol_sha256)
    audit.check(result["protocol_sha256"] == identity["protocol_sha256"] == args.expected_protocol_sha256, "Protocol freeze")
    for path, digest in result["source_sha256"].items(): audit.file(Path(path) if Path(path).is_absolute() else repo / path, digest)
    audit.check(args.expected_source_sha256 in result["source_sha256"].values(), "Recorded analyzer source")
    original, action = Path(identity["evaluation_root"]), Path(identity["action_root"])
    for name, digest in ANCHOR.items(): audit.file(action / name, digest)
    anchor, old = read(action / "input_identity.json"), read(action / "results.json")
    audit.check(identity["evaluation_expected_files_sha256"] == anchor["input_files_sha256"], "Original byte anchor")
    observed, bad_models = {}, set()
    for relative, digest in anchor["input_files_sha256"].items():
        path = audit.path(original, relative)
        if path.is_file(): observed[relative] = audit.file(path)
        if not path.is_file() or observed[relative] != digest: bad_models.add(relative.split("/")[0])
    audit.check(observed == identity["evaluation_observed_files_sha256"], "Observed original inventory")
    audit.file(root / result["all_inputs_sha256_file"], result["all_inputs_sha256"])
    all_inputs = read(root / result["all_inputs_sha256_file"])
    for path, digest in all_inputs.items(): audit.file(path, digest)
    audit.check(result["inputs_verified_after"] == len(all_inputs), "Post-analysis input verification count")
    audit.check(set(result["runs"]) == set(MODELS), "Six-model identity")
    runs, failures = {}, []
    for label in MODELS:
        run = result["runs"][label]; objective, seed = label.split("_seed")
        audit.check(run["objective"] == objective and run["seed"] == int(seed), label + ": objective/seed")
        index = {(r["source_index"], r["target_frame"]): r for r in run["frames"]}
        old_index = {(r["source_index"], r["target_frame"]): r for r in old["runs"][label]["frames"]}
        audit.check(len(index) == len(run["frames"]) == 297 and set(index) == set(old_index) == set(KEYS), label + ": all fixed frames")
        frames = {}
        for source, target in KEYS:
            context = f"{label}/{source}/{target}"; item = index[source, target]
            path = audit.path(root, item["record_file"]); audit.file(path, item["record_sha256"]); saved = read(path)
            audit.check(saved["source_index"] == source and saved["target_frame"] == target and saved["trajectory_id"] == f"test:{source:06d}", context + ": frame ID")
            stem = f"trajectory_{source:06d}_target_{target:04d}"
            raw_path, raw_array = original / label / (stem + ".json"), original / label / (stem + ".npz")
            prior_item = old_index[source, target]; prior_path = audit.path(action, prior_item["record_file"])
            broken, raw, prior, prior_array = label in bad_models, None, None, None
            if not broken:
                raw = read(raw_path); broken = not prior_path.is_file() or sha(prior_path) != prior_item["record_sha256"]
                if not broken:
                    prior = read(prior_path); prior_array = audit.path(action, prior["derived_array_file"])
                    broken = not prior_array.is_file() or sha(prior_array) != prior["derived_array_sha256"]
            if broken or (raw and raw["status"] != "complete"):
                audit.check(saved["status"] in ("invalid", "missing", "failed") and all(v is None for v in saved["metrics"].values()), context + ": strict invalid nulls")
                audit.check(set(saved["metrics"]) == set(saved["undefined_reasons"]) == set(NAMES) and all(saved["undefined_reasons"].values()), context + ": explicit null reasons")
                audit.check(not saved.get("derived_array_file") and not saved["pairs"] and not saved["identity_checks"], context + ": no invalid numeric output")
                frames[source, target] = {"metrics": dict.fromkeys(NAMES), "undefined_reasons": saved["undefined_reasons"], "status": saved["status"]}
            else:
                audit.file(raw_path, saved["input_record_sha256"]); audit.file(raw_array, saved["input_array_sha256"])
                audit.file(prior_path, saved["action_record_sha256"]); audit.file(prior_array, saved["action_array_sha256"])
                audit.check(prior["input_record_sha256"] == saved["input_record_sha256"] and prior["input_array_sha256"] == saved["input_array_sha256"], context + ": prior/raw identity")
                output = audit.path(root, saved["derived_array_file"]); audit.file(output, saved["derived_array_sha256"])
                with np.load(raw_array, allow_pickle=False) as a, np.load(output, allow_pickle=False) as d, np.load(prior_array, allow_pickle=False) as old_arrays:
                    expected = reconstruct(raw, a, audit, context); check_frame(saved, expected, d, audit, context)
                    for policy in SPARSE:
                        for new, old_name in (("error", "error"), ("alignment", "alignment"), ("cost", "perturbation_cost")):
                            wanted = expected["derived"][f"normalized_coordinate_{new}_{policy}"]
                            audit.same(old_arrays[f"{old_name}_normalized_{policy}"] / 2, wanted, context + ": exact vector/coordinate convention", scale=np.maximum(np.abs(wanted), 1e-300))
                        audit.same(old_arrays["optional_degree_" + policy], expected["derived"]["optional_degree_" + policy], context + ": independent incidence", exact=True)
                audit.check(saved.get("action_replay", {}).get("passed") is True, context + ": action replay")
                frames[source, target] = expected
            if saved["status"] != "complete":
                failures.append({"model": label, "source_index": source, "target_frame": target, "status": saved["status"], "failure": saved["failure"], "undefined_reasons": sorted(set(saved["undefined_reasons"].values()))})
        expected = aggregate(frames); check_aggregate(run["aggregate"], expected, audit, label); runs[label] = expected
    audit.check(result["failures"] == failures and status["failed_or_missing_or_invalid_frames"] == len(failures), "Exact unsuccessful outcome ledger")
    audit.check(result["state"] == ("incomplete" if failures else "complete"), "Completion accounting")
    for objective in ("faithful", "nll"):
        group = result["objectives"][objective]; seed_runs = [runs[f"{objective}_seed{s}"] for s in range(3)]
        for k in NAMES: check_seeds(group["metrics"][k], [r["metrics"][k] for r in seed_runs], audit, objective + ":" + k)
        for k in seed_runs[0]["weighted_conditionals"]:
            check_seeds(group["weighted_conditionals"][k], [r["weighted_conditionals"][k]["value"] for r in seed_runs], audit, objective + ":ratio:" + k)
        metrics = {k: mean(r["metrics"][k] for r in seed_runs) for k in NAMES}
        check_identities(group["mean_identity_checks"], partition(metrics, 3, audit, objective), audit, objective)
    audit.check({str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()} == inventory, "Analysis immutable through audit")
    for path, digest in audit.files.items(): audit.check(sha(path) == digest, "Input/source unchanged after audit")
    return {"passed": True, "audited_frames": 1782, "checks": audit.checks, "maximum_absolute_comparison_difference": audit.max_difference,
        "results_sha256": sha(root / "results.json"), "analysis_source_sha256": args.expected_source_sha256, "analysis_protocol_sha256": args.expected_protocol_sha256,
        "audited_input_files": len(audit.files), "implementation": "Independent NumPy einsum/bincount, integer-coded masks, math.fsum reductions and hierarchy; no production analysis/action arithmetic imports",
        "scope": "Arithmetic/provenance software checks, not statistical sample size; no inference or new predictions"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--analysis-root", type=Path, required=True); p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-source-sha256", required=True); p.add_argument("--expected-protocol-sha256", required=True)
    args = p.parse_args()
    if not args.execute: p.error("Explicit --execute after root authorization is required")
    if args.output_dir.resolve() == args.analysis_root.resolve() or args.analysis_root.resolve() in args.output_dir.resolve().parents:
        p.error("Keep audit attempts outside immutable analysis outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / ("attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex + ".json")
    audit, started = Audit(), time.perf_counter()
    try:
        result = execute(args, audit)
    except BaseException as error:
        result = {"passed": False, "checks_before_failure": audit.checks, "error_type": type(error).__name__, "error": str(error)}
        raise
    finally:
        result.update(source_sha256=sha(__file__), generated_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        with path.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
        print(json.dumps({"audit_attempt": str(path), **result}, allow_nan=False))


if __name__ == "__main__":
    main()
