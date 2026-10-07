"""Particle-simulation methods and data utilities."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from research import graph_convention_bridge as bridge

METRICS = ("position_coordinate_mse", "normalized_coordinate_mse")
RISKS = ("previous-observed-base-risk25", "current-base-risk25")


class Audit:
    def __init__(self):
        self.checks = 0
    def check(self, condition, message):
        self.checks += 1
        if not condition:
            raise ValueError(message)
    def equal(self, actual, expected, message):
        self.check(actual == expected, message)
    def array(self, actual, expected, message):
        self.check(np.array_equal(actual, expected), message)
    def scalar(self, actual, expected, message):
        self.check(actual is not None and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-12, atol=1e-16), message)


def case(policy, loop):
    return f"{policy}_uncapped_loops{loop}"


def contrasts():
    """Linear contrasts apply to paired per-frame metrics before averaging."""
    result = {}
    def add(name, positive, negative):
        result[name] = {positive: 1, negative: -1}
    for policy in bridge.POLICIES:
        add(f"loop1_minus_loop0__{policy}", case(policy, 1), case(policy, 0))
    for policy in ("base", "dense"):
        for loop in (0, 1):
            add(f"cap128_minus_uncapped__{policy}__loops{loop}", f"{policy}_cap128_loops{loop}", case(policy, loop))
    for loop in (0, 1):
        for policy in bridge.POLICIES[2:]:
            for reference in ("base", "random25", "speed25"):
                if policy != reference:
                    add(f"{policy}_minus_{reference}__loops{loop}", case(policy, loop), case(reference, loop))
        add(f"dense_minus_base__loops{loop}", case("dense", loop), case("base", loop))
        add(f"current_minus_previous_risk__loops{loop}", case(RISKS[1], loop), case(RISKS[0], loop))
    for policy, reference in ((RISKS[0], "random25"), (RISKS[1], "random25"), ("dense", "base")):
        result[f"loop_interaction__{policy}_minus_{reference}"] = {
            case(policy, 1): 1, case(reference, 1): -1, case(policy, 0): -1, case(reference, 0): 1}
    return result


CONTRASTS = contrasts()


def complete_mean(values):
    return float(np.mean(values)) if values and all(value is not None and np.isfinite(value) for value in values) else None


def seed_statistics(values):
    valid = len(values) == 3 and all(value is not None and np.isfinite(value) for value in values)
    return {"seed_values": values, "mean": float(np.mean(values)) if valid else None,
            "sample_sd": float(np.std(values, ddof=1)) if valid else None,
            "defined_seeds": sum(value is not None for value in values), "required_seeds": 3}


def summary_measure(rows, name):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["trajectory_id"], []).append(row["measures"].get(name))
    trajectories = {key: {"value": complete_mean(values), "required_frames": len(values),
                           "undefined_frames": sum(value is None for value in values)} for key, values in sorted(grouped.items())}
    return {"equal_trajectory_mean": complete_mean([row["value"] for row in trajectories.values()]),
            "equal_frame_mean_sensitivity": complete_mean([row["measures"].get(name) for row in rows]),
            "required_frames": len(rows), "undefined_frames": sum(row["measures"].get(name) is None for row in rows),
            "trajectories": trajectories}


def optional_set(edges, base):
    return {tuple(sorted(pair)) for pair in edges.T if pair[0] != pair[1]} - base


def audit_native_parity(row, arrays, audit):
    saved = row["native_parity"]
    graph = bridge.strict_pairs(arrays["current_history"][-1], .015)
    expected_edges = bridge.ordered_edges(arrays["current_history"][-1], graph.base, True, 128)
    native_edges_equal = np.array_equal(arrays["native_edges"], expected_edges)
    feature_equal = [np.array_equal(arrays["native_node_features"], arrays["supplied_node_features"]),
                     native_edges_equal,
                     np.array_equal(arrays["native_edge_features"], arrays["supplied_edge_features"])]
    native = {key: arrays[f"native_{key}"] for key in ("prediction", "risk", "raw_risk")}
    parity_failed = row["failure"] is not None and row["failure"].get("category") == "native_parity_failure"
    supplied = {key: arrays[f"supplied_parity_{key}" if parity_failed else f"{key}__base_cap128_loops1"]
                for key in native}
    finite = all(np.isfinite(native[key]).all() and np.isfinite(supplied[key]).all() for key in native)
    agree = bool(finite and np.allclose(native["prediction"], supplied["prediction"], rtol=0, atol=bridge.PARITY_PRED_ATOL)
                 and all(np.allclose(native[key], supplied[key], rtol=bridge.PARITY_RISK_RTOL, atol=bridge.PARITY_RISK_ATOL)
                         for key in ("risk", "raw_risk")))
    for key, value in (("native_edge_identity", native_edges_equal), ("native_supplied_feature_identity", feature_equal),
                       ("finite", finite), ("prediction_risk_agree", agree),
                       ("passed", bool(native_edges_equal and all(feature_equal) and agree)),
                       ("prediction_atol", bridge.PARITY_PRED_ATOL), ("risk_atol", bridge.PARITY_RISK_ATOL),
                       ("risk_rtol", bridge.PARITY_RISK_RTOL)):
        audit.equal(saved[key], value, "Native parity flag/tolerance differs from raw arrays")
    for key in native:
        value = float(np.max(np.abs(native[key] - supplied[key]))) if finite else None
        audit.equal(saved[key + "_max_abs_difference"], value, "Native parity maximum differs from raw arrays")


def analyze_frame(row, arrays, audit):
    identity = {key: row[key] for key in ("split", "source_index", "target_frame", "trajectory_id")}
    measures = {f"case__{name}__{metric}": None for name in bridge.CASE_NAMES for metric in METRICS}
    diagnostics = {"native_parity": row["native_parity"], "failure": row["failure"],
                   "base_graph_audit": row.get("base_graph_audit"), "locked_graph_comparison": row.get("locked_graph_comparison"),
                   "case_status": {name: value["status"] for name, value in row["cases"].items()},
                   "prediction_aliases": {name: value["reused_from_case"] for name, value in row["cases"].items() if value.get("reused_from_case")}}
    n = row["n_particles"]
    audit.array(bridge.full.state_hash(arrays["current_history"]), row["current_history_sha256"], "History hash differs")
    audit.array(bridge.full.state_hash(arrays["previous_history"]), row["previous_history_sha256"], "Previous history hash differs")
    audit.equal(row["status"] == "complete", row["failure"] is None, "Frame failure/status differs")
    audit_native_parity(row, arrays, audit)
    pairing_hashes = {name: bridge.full.state_hash(arrays[name]) if name in arrays else None
                      for name in ("current_history", "previous_history", "target_position", "acceleration_std", "particle_types")}
    parity_failure = row["failure"] is not None and row["failure"].get("category") == "native_parity_failure"
    if parity_failure:
        audit.check(not row["native_parity"]["passed"], "Native failure marked parity pass")
    else:
        audit.equal(set(row["cases"]), set(bridge.CASE_NAMES), "Frame case coverage differs")
        audit.check(row["native_parity"]["passed"], "Nonfailed native gate did not pass")
        audit.array(arrays["native_node_features"], arrays["supplied_node_features"], "Native/supplied node features differ")
        audit.array(arrays["native_edge_features"], arrays["supplied_edge_features"], "Native/supplied edge features differ")
        audit.array(arrays["native_edges"], arrays["edges__base_cap128_loops1"], "Native graph identity differs")
        target, scale = arrays["target_position"], arrays["acceleration_std"]
        audit.check(scale.shape == (2,) and np.isfinite(scale).all() and np.all(scale > 0), "Bad normalization")
        audit.equal(bridge.full.state_hash(target), row["target_sha256"], "Target hash differs")
        base = bridge.same.pair_set(arrays["strict_base_pairs"])
        extra = bridge.same.pair_set(arrays["strict_annulus_pairs"])
        rebuilt = bridge.strict_pairs(arrays["current_history"][-1], .015)
        audit.array(rebuilt.base, arrays["strict_base_pairs"], "Strict native base pairs differ")
        audit.array(rebuilt.extra, arrays["strict_annulus_pairs"], "Strict native annulus pairs differ")
        audit.equal(row["optional_budget"], int(.25 * len(extra)), "Optional budget differs")
        for name, record in row["cases"].items():
            audit.equal(record["status"] == "complete", record["failure"] is None, "Case failure/status differs")
            if record["status"] != "complete":
                audit.check(record["metrics"] is None, "Failed case has complete metric")
                continue
            edge = arrays[f"edges__{name}"]
            audit.equal(edge.shape, (2, record["graph"]["directed_edges"]), "Directed edge shape differs")
            audit.equal(bridge.full.state_hash(edge), record["graph"]["edge_sha256"], "Edge hash differs")
            audit.equal(int(np.sum(edge[0] == edge[1])), record["graph"]["self_edges"], "Self-edge count differs")
            receiver_degree = np.bincount(edge[1], minlength=n)
            audit.equal(record["graph"]["max_receiver_degree"], int(receiver_degree.max()), "Saved maximum receiver degree differs")
            audit.scalar(record["graph"]["receiver_degree_mean"], float(receiver_degree.mean()), "Saved mean receiver degree differs")
            policy, cap_name, loop_name = name.rsplit("_", 2)
            if policy in ("base", "dense"):
                pairs = rebuilt.base if policy == "base" else np.concatenate((rebuilt.base, rebuilt.extra))
                expected_edges = bridge.ordered_edges(arrays["current_history"][-1], pairs,
                                                      bool(int(loop_name[-1])), 128 if cap_name == "cap128" else None)
                audit.array(edge, expected_edges, "Base/dense factorial edge semantics differ")
            if "uncapped" in name:
                loop = int(name[-1]); policy = name.rsplit("_uncapped_loops", 1)[0]
                selected = optional_set(edge, base)
                all_pairs = {tuple(sorted(pair)) for pair in edge.T if pair[0] != pair[1]}
                expected = 0 if policy == "base" else len(extra) if policy == "dense" else row["optional_budget"]
                audit.check(base <= all_pairs <= base | extra, "Mandatory/candidate pair invariant differs")
                audit.equal(len(selected), expected, "Selected optional count differs")
                audit.equal(edge.shape[1], 2 * len(all_pairs) + loop * n, "Symmetric directed/loop count differs")
                audit.equal(record["graph"]["self_edges"], loop * n, "Uncapped loop multiplicity differs")
                if policy in RISKS:
                    score = arrays[f"selection_score__{name}"]
                    expected_score = arrays[f"previous_base_risk_loops{loop}"] if policy == RISKS[0] else arrays[f"risk__{case('base', loop)}"]
                    audit.array(score, expected_score, "Risk score came from wrong history or graph convention")
                    pairs = bridge.select_pairs(rebuilt, score, row["optional_budget"])
                    audit.array(edge, bridge.ordered_edges(arrays["current_history"][-1], pairs, bool(loop)), "Risk selected edges differ")
                elif policy == "random25":
                    rng = np.random.default_rng(np.random.SeedSequence(row["random_seed_material"]))
                    pairs = bridge.random_pairs(rebuilt, row["optional_budget"], rng)
                    audit.array(edge, bridge.ordered_edges(arrays["current_history"][-1], pairs, bool(loop)), "Random selected edges differ")
                elif policy == "speed25":
                    score = np.linalg.norm(arrays["current_history"][-1] - arrays["current_history"][-2], axis=1)
                    pairs = bridge.select_pairs(rebuilt, score, row["optional_budget"])
                    audit.array(edge, bridge.ordered_edges(arrays["current_history"][-1], pairs, bool(loop)), "Speed selected edges differ")
            prediction, risk, raw = arrays[f"prediction__{name}"], arrays[f"risk__{name}"], arrays[f"raw_risk__{name}"]
            audit.check(np.isfinite(prediction).all() and np.isfinite(risk).all() and np.isfinite(raw).all(), "Complete case has nonfinite outputs")
            audit.check(np.max(np.abs(prediction)) <= bridge.MAX_ABS and np.all(risk > 0), "Complete case violates output guard")
            residual = prediction.astype(np.float64) - target
            normalized = residual / scale
            audit.array(residual, arrays[f"position_residual__{name}"], "Position residual differs")
            audit.array(normalized, arrays[f"normalized_residual__{name}"], "Normalized residual differs")
            for metric, expected in zip(METRICS, (float(np.mean(residual ** 2)), float(np.mean(normalized ** 2)))):
                audit.scalar(record["metrics"][metric], expected, "Saved arithmetic differs")
                measures[f"case__{name}__{metric}"] = expected
            if record.get("reused_from_case"):
                original = record["reused_from_case"]
                audit.array(edge, arrays[f"edges__{original}"], "Reused graph not identical")
                for field in ("prediction", "risk", "raw_risk"):
                    audit.array(arrays[f"{field}__{name}"], arrays[f"{field}__{original}"], "Graph alias output differs")
        for loop in (0, 1):
            for policy, audit_name in (("base", f"loops{loop}"), ("dense", f"dense_loops{loop}")):
                uncapped = arrays[f"edges__{policy}_uncapped_loops{loop}"]
                capped = arrays[f"edges__{policy}_cap128_loops{loop}"]
                degrees = np.bincount(uncapped[1], minlength=n)
                expected_cap = {"receivers_above_cap": int(np.sum(degrees > 128)),
                                "edges_removed_by_cap": int(uncapped.shape[1] - capped.shape[1]),
                                "capped_equals_uncapped": bool(np.array_equal(capped, uncapped))}
                audit.equal(row["base_graph_audit"][audit_name], expected_cap, "Cap activation count differs")
            previous_graph = bridge.strict_pairs(arrays["previous_history"][-1], .015)
            previous_edges = bridge.ordered_edges(arrays["previous_history"][-1], previous_graph.base, bool(loop))
            audit.array(arrays[f"previous_base_edges_loops{loop}"], previous_edges, "Previous score graph differs")
            previous_degrees = np.bincount(previous_edges[1], minlength=n)
            audit.equal(row[f"previous_score_loops{loop}"]["max_receiver_degree"], int(previous_degrees.max()), "Previous graph degree differs")
            audit.equal(row[f"previous_score_loops{loop}"]["receivers_above_cap"], int(np.sum(previous_degrees > 128)), "Previous cap activation differs")
        locked = bridge.full.checked_candidates(arrays["current_history"][-1], .015, bridge.MAX_PAIRS)
        audit.array(arrays["locked_base_pairs"], locked.base, "Locked base reconstruction differs")
        audit.array(arrays["locked_annulus_pairs"], locked.extra, "Locked annulus reconstruction differs")
        audit.equal(row["locked_graph_comparison"], {
            "base_pair_symmetric_difference": len(base ^ bridge.same.pair_set(locked.base)),
            "annulus_pair_symmetric_difference": len(extra ^ bridge.same.pair_set(locked.extra))}, "Radius classifier difference count differs")
        for loop in (0, 1):
            previous, current, base_case = case(RISKS[0], loop), case(RISKS[1], loop), case("base", loop)
            score_keys = [f"selection_score__{previous}", f"selection_score__{current}"]
            risk_ok = all(row["cases"][name]["status"] == "complete" for name in (previous, current, base_case))
            for metric in ("optional_jaccard", "optional_intersection", "score_spearman"):
                measures[f"current_previous__loops{loop}__{metric}"] = None
            for policy in RISKS:
                for target_name in ("base_residual", "own_sparse_benefit"):
                    measures[f"risk_correlation__{policy}__loops{loop}__{target_name}"] = None
            if risk_ok:
                sets = [optional_set(arrays[f"edges__{name}"], base) for name in (previous, current)]
                union = sets[0] | sets[1]
                intersection = len(sets[0] & sets[1])
                measures[f"current_previous__loops{loop}__optional_intersection"] = intersection
                measures[f"current_previous__loops{loop}__optional_jaccard"] = intersection / len(union) if union else 1.
                measures[f"current_previous__loops{loop}__score_spearman"] = bridge.same.spearman(arrays[score_keys[0]], arrays[score_keys[1]])["value"]
            for policy in RISKS:
                selected_case = case(policy, loop)
                if row["cases"][selected_case]["status"] == "complete" and row["cases"][base_case]["status"] == "complete":
                    score = arrays[f"selection_score__{selected_case}"]
                    base_error = np.sum(arrays[f"normalized_residual__{base_case}"] ** 2, axis=1)
                    action_error = np.sum(arrays[f"normalized_residual__{selected_case}"] ** 2, axis=1)
                    for target_name, label in (("base_residual", base_error), ("own_sparse_benefit", base_error - action_error)):
                        measures[f"risk_correlation__{policy}__loops{loop}__{target_name}"] = bridge.same.spearman(score, label)["value"]
    for name, weights in CONTRASTS.items():
        for metric in METRICS:
            values = [(weight, measures[f"case__{case_name}__{metric}"]) for case_name, weight in weights.items()]
            measures[f"contrast__{name}__{metric}"] = (float(sum(weight * value for weight, value in values))
                                                       if all(value is not None for _, value in values) else None)
    return {**identity, "status": row["status"], "pairing_hashes": pairing_hashes, "measures": measures, "diagnostics": diagnostics}


def graph_diagnostics(rows):
    parity = [row["diagnostics"]["native_parity"] for row in rows]
    caps = {}
    for name in ("loops0", "loops1", "dense_loops0", "dense_loops1"):
        values = [row["diagnostics"]["base_graph_audit"][name] for row in rows if row["diagnostics"]["base_graph_audit"]]
        caps[name] = {"recorded_frames": len(values), "cap_active_frames": sum(value["receivers_above_cap"] > 0 for value in values),
                      "receivers_above_cap": sum(value["receivers_above_cap"] for value in values),
                      "edges_removed_by_cap": sum(value["edges_removed_by_cap"] for value in values)}
    radius = [row["diagnostics"]["locked_graph_comparison"] for row in rows if row["diagnostics"]["locked_graph_comparison"]]
    differences = {name: {"different_frames": sum(value[name] > 0 for value in radius),
                          "total_pair_symmetric_difference": sum(value[name] for value in radius)}
                   for name in ("base_pair_symmetric_difference", "annulus_pair_symmetric_difference")}
    maxima = {}
    for name in ("prediction_max_abs_difference", "risk_max_abs_difference", "raw_risk_max_abs_difference"):
        values = [value[name] for value in parity if value[name] is not None]
        maxima[name] = max(values) if values else None
    return {"native_parity_passed_frames": sum(value["passed"] for value in parity), "native_parity_required_frames": len(rows),
            "native_parity_maxima": maxima, "cap_counts": caps, "locked_radius_comparison": differences,
            "exact_graph_reuse_cases": sum(len(row["diagnostics"]["prediction_aliases"]) for row in rows),
            "failed_cases": {name: sum(row["diagnostics"]["case_status"].get(name) != "complete" for row in rows) for name in bridge.CASE_NAMES}}


def audit_official_frame(item, arrays, positions, particle_types, source_record, audit):
    audit.equal(source_record["id"], item["trajectory_id"], "Actual source trajectory ID differs")
    target = item["target_frame"]
    audit.array(arrays["current_history"], positions[target - 6:target], "Saved current history differs from actual official source")
    audit.array(arrays["previous_history"], positions[target - 7:target - 1], "Saved previous history differs from actual official source")
    if "target_position" in arrays:
        audit.array(arrays["target_position"], np.asarray(positions[target], dtype=np.float64), "Saved target differs from actual official source")
    source_types = np.asarray(particle_types, dtype=np.int64)
    if source_types.ndim == 0:
        source_types = np.full(positions.shape[1], source_types, dtype=np.int64)
    audit.array(arrays["particle_types"], source_types, "Saved types differ from actual official source")


def load_run(directory, audit):
    directory = Path(directory)
    protocol_path, result_path, status_path = [directory / name for name in ("protocol.json", "result.json", "status.json")]
    protocol, result, status = [json.loads(path.read_text()) for path in (protocol_path, result_path, status_path)]
    audit.equal(status["state"], "complete", "Run status not complete")
    audit.equal(result["state"], "complete", "Run result not complete")
    audit.equal(status["result_sha256"], bridge.full.sha256(result_path), "Result hash differs")
    protocol_hash = bridge.full.sha256(protocol_path)
    audit.equal(result["protocol_sha256"], protocol_hash, "Protocol hash differs")
    audit.equal(protocol["cases"], list(bridge.CASE_NAMES), "Declared case coverage differs")
    audit.equal(protocol["bridge_protocol_sha256"], bridge.full.sha256(bridge.PROTOCOL), "Bridge protocol source differs")
    audit.equal(len(protocol["expected_frames"]), 425, "Declared frame count differs")
    audit.equal(result["required_frames"], 425, "Result required frame count differs")
    audit.check(protocol["objective"] in ("faithful", "nll") and protocol["seed"] in (0, 1, 2), "Unknown model identity")
    for key in ("seed", "objective"):
        audit.equal(result[key], protocol[key], "Result model identity differs")
    expected = {bridge.stem(row): row for row in protocol["expected_frames"]}
    audit.equal(len(expected), 425, "Duplicate expected frames")
    actual = {bridge.stem(row): row for row in result["records"]}
    audit.equal(len(actual), len(result["records"]), "Duplicate result frame")
    audit.equal(set(actual), set(expected), "Missing/unexpected result frame")
    sources, manifests = {}, {}
    for split in ("valid", "test"):
        paths = [Path(path) for path in protocol["input_files_sha256"] if Path(path).name == split + ".json"]
        audit.equal(len(paths), 1, "Exactly one pinned official split manifest required")
        manifest_path = paths[0]
        audit.equal(bridge.full.sha256(manifest_path), protocol["input_files_sha256"][str(manifest_path)], "Pinned official manifest hash differs")
        manifests[split] = json.loads(manifest_path.read_text())
        sources[split] = bridge.full.load_manifest_data(manifest_path, verify_hashes=True)
        audit.equal(manifests[split]["split"], split, "Pinned manifest split differs")
    audit.equal(manifests["test"]["source"]["sha256"], bridge.full.OFFICIAL_TEST_SHA256, "Official test source differs")
    rows, hashes = [], {str(path.resolve()): bridge.full.sha256(path) for path in (protocol_path, result_path, status_path)}
    for key, item in expected.items():
        indexed = actual[key]
        audit.equal(indexed["record_file"], key + ".json", "Unexpected record filename")
        path = directory / indexed["record_file"]
        audit.equal(bridge.full.sha256(path), indexed["record_sha256"], "Frame record hash differs")
        row = json.loads(path.read_text())
        for name, value in item.items():
            audit.equal(row[name], value, "Frame identity differs")
        audit.equal(row["protocol_sha256"], protocol_hash, "Frame protocol differs")
        audit.equal(row["random_seed_material"], [20261005, 771, protocol["seed"],
                    0 if row["split"] == "valid" else 1, row["source_index"], row["target_frame"]], "Random seed material differs")
        audit.equal(bridge.record_index(row, path), indexed, "Compact record index differs")
        array_path = directory / (key + ".npz")
        audit.equal(row["array_file"], array_path.name, "Array filename differs")
        audit.equal(bridge.full.sha256(array_path), row["array_sha256"], "Array hash differs")
        hashes[str(path.resolve())], hashes[str(array_path.resolve())] = indexed["record_sha256"], row["array_sha256"]
        with np.load(array_path, allow_pickle=False) as arrays:
            source_record = manifests[item["split"]]["records"][item["source_index"]]
            positions, particle_types = sources[item["split"]][item["source_index"]]
            audit_official_frame(item, arrays, positions, particle_types, source_record, audit)
            rows.append(analyze_frame(row, arrays, audit))
    audit.equal(sum(row["status"] == "complete" for row in rows), result["complete_frames"], "Complete-frame count differs")
    audit.equal(sum(row["status"] == "failed" for row in rows), result["failed_frames"], "Failed-frame count differs")
    measure_names = sorted({name for row in rows for name in row["measures"]})
    splits = {}
    for split, count in (("valid", 128), ("test", 297)):
        split_rows = [row for row in rows if row["split"] == split]
        audit.equal(len(split_rows), count, "Split frame count differs")
        splits[split] = {"measures": {name: summary_measure(split_rows, name) for name in measure_names},
                         "failed_frames": sum(row["status"] == "failed" for row in split_rows), "required_frames": count,
                         "graph_diagnostics": graph_diagnostics(split_rows)}
    return {"objective": protocol["objective"], "seed": protocol["seed"], "checkpoint_sha256": protocol["checkpoint_sha256"],
            "source_directory": str(directory.resolve()), "protocol_sha256": protocol_hash,
            "input_files_sha256": hashes, "splits": splits, "frames": rows,
            "source_protocol": protocol}


def summarize(runs, audit):
    mapping = {(run["objective"], run["seed"]): run for run in runs}
    audit.equal(set(mapping), {(objective, seed) for objective in ("faithful", "nll") for seed in (0, 1, 2)}, "Six model identities required")
    audit.equal(len(runs), 6, "Exactly six model runs required")
    audit.equal(len({run["checkpoint_sha256"] for run in runs}), 6, "Six distinct checkpoint hashes required")
    populations = [run["source_protocol"]["expected_frames"] for run in runs]
    for population in populations[1:]:
        audit.equal(population, populations[0], "Cross-model history schedule differs")
    for split, count in (("valid", 128), ("test", 297)):
        reference = {(row["source_index"], row["target_frame"]): row for row in runs[0]["frames"] if row["split"] == split}
        audit.equal(len(reference), count, "Reference split coverage differs")
        for run in runs[1:]:
            for row in run["frames"]:
                if row["split"] == split:
                    original = reference[row["source_index"], row["target_frame"]]
                    audit.equal(row["pairing_hashes"], original["pairing_hashes"], "Cross-model history/target/normalization hashes differ")
    groups = {}
    for objective in ("faithful", "nll"):
        groups[objective] = {}
        for split in ("valid", "test"):
            summaries = [mapping[objective, seed]["splits"][split] for seed in (0, 1, 2)]
            names = set(summaries[0]["measures"])
            for summary in summaries[1:]:
                audit.equal(set(summary["measures"]), names, "Cross-seed measure coverage differs")
            groups[objective][split] = {"measures": {name: seed_statistics([summary["measures"][name]["equal_trajectory_mean"] for summary in summaries]) for name in sorted(names)},
                "failed_frames_by_seed": [summary["failed_frames"] for summary in summaries]}
    return groups


def render(report):
    lines = ["# Exploratory graph-convention bridge", "", "Post-inspection follow-up; frozen original results remain separate. Means and sample SDs use three paired seed means. Validation/test and all failures remain separate. Operational durations are not a speed comparison.", ""]
    def fmt(stats):
        return "undefined" if stats["mean"] is None else f"{stats['mean']:.9g} ± {stats['sample_sd']:.6g}"
    for split in ("valid", "test"):
        lines += [f"## {split}: normalized coordinate MSE", "", "| Case | Faithful | NLL |", "|---|---:|---:|"]
        for name in bridge.CASE_NAMES:
            key = f"case__{name}__normalized_coordinate_mse"
            lines.append(f"| {name} | {fmt(report['groups']['faithful'][split]['measures'][key])} | {fmt(report['groups']['nll'][split]['measures'][key])} |")
        lines += ["", "| Paired interaction (loop1 minus loop0) | Faithful | NLL |", "|---|---:|---:|"]
        for name in (f"loop_interaction__{RISKS[0]}_minus_random25", "loop_interaction__dense_minus_base", f"loop_interaction__{RISKS[1]}_minus_random25"):
            key = f"contrast__{name}__normalized_coordinate_mse"
            lines.append(f"| {name} | {fmt(report['groups']['faithful'][split]['measures'][key])} | {fmt(report['groups']['nll'][split]['measures'][key])} |")
        lines += [""]
    lines += ["Risk loop contrasts include changed scoring and selection; base/dense/random/speed have fixed selected nonself pairs between loop arms. Same-state effects do not establish autonomous stability or resolve lack of expansion training support.", "", f"Arithmetic, coverage and integrity audit checks: {report['audit']['checks']}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=6, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Analysis output directory already exists; preserve prior failures and choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "graph_convention_bridge.json"
    source_digest = bridge.full.sha256(__file__)
    bridge.full.atomic_json(args.output_dir / "analysis_source.json", {"summary_source_sha256": source_digest,
                              "started_utc": datetime.now(timezone.utc).isoformat()})
    audit = Audit()
    try:
        runs = [load_run(path, audit) for path in args.runs]
        groups = summarize(runs, audit)
        pinned = {}
        for run in runs:
            for path, digest in run["source_protocol"]["input_files_sha256"].items():
                if path in pinned:
                    audit.equal(pinned[path], digest, "Runs disagree on shared input hash")
                pinned[path] = digest
        for path, digest in pinned.items():
            audit.equal(bridge.full.sha256(path), digest, "Pinned source/checkpoint/data input differs")
        for run in runs:
            for path, digest in run["input_files_sha256"].items():
                audit.equal(bridge.full.sha256(path), digest, "Input changed during analysis")
        audit.equal(bridge.full.sha256(__file__), source_digest, "Summary source changed during analysis")
        report = {"schema": 1, "scope": "post-inspection exploratory graph-convention bridge; no inference",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "summary_source_sha256": source_digest,
            "contrasts": CONTRASTS, "groups": groups, "runs": runs,
            "audit": {"passed": True, "checks": audit.checks}}
        bridge.full.atomic_json(output, report)
        (args.output_dir / "graph_convention_bridge.md").write_text(render(report))
    except BaseException as error:
        bridge.full.atomic_json(args.output_dir / "failed_analysis.json", {"passed": False, "checks_before_failure": audit.checks,
            "error_type": type(error).__name__, "error": str(error), "summary_source_sha256": source_digest,
            "summary_source_sha256_at_failure": bridge.full.sha256(__file__)})
        raise


if __name__ == "__main__":
    main()
