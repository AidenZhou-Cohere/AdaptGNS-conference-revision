"""Summarize complete rollout experiments without survivor-only accuracy."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from research.pilot_rollout import POLICIES, sha256, write_json


def strict_seed_statistics(values):
    """An unconditional seed mean is undefined if any seed is incomplete."""
    defined = all(value is not None for value in values)
    return {"seed_values": values, "n_seeds": len(values),
            "mean": float(np.mean(values)) if defined else None,
            "sample_seed_sd": float(np.std(values, ddof=1)) if defined and len(values) > 1 else None}


def read_experiment(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    runs, sources = {}, []
    for kind in protocol["objectives"]:
        for seed in protocol["seeds"]:
            name = f"{kind}_seed{seed}"
            path = directory / (name + ".json")
            if not path.exists():
                raise ValueError(f"All requested outcomes are required: missing {path}")
            run = json.loads(path.read_text())
            if run["objective"] != kind or run["seed"] != seed:
                raise ValueError(f"Identity mismatch: {path}")
            keys = {(row["policy"], row["trajectory_id"]) for row in run["records"]}
            expected = {(policy, trajectory) for policy in POLICIES for trajectory in protocol["trajectories"]}
            if keys != expected or len(keys) != len(run["records"]):
                raise ValueError(f"Incomplete or duplicated trajectory outcomes: {path}")
            for row in run["records"]:
                if row["status"] == "complete" and row["completed_steps"] != protocol["horizon"]:
                    raise ValueError("Complete outcome has wrong forecast length")
                if row["status"] == "failed" and row["failure"] is None:
                    raise ValueError("Failure outcome requires a stated reason")
            runs[name] = run
            sources.append({"file": str(path.resolve()), "sha256": sha256(path)})
    return protocol, runs, sources


def aggregate(protocol, runs):
    groups, failures = {}, []
    trajectory_count = len(protocol["trajectories"])
    for kind in protocol["objectives"]:
        groups[kind] = {}
        for policy in POLICIES:
            seed_rows = [[row for row in runs[f"{kind}_seed{seed}"]["records"] if row["policy"] == policy]
                         for seed in protocol["seeds"]]
            all_rows = [row for rows in seed_rows for row in rows]
            complete = sum(row["status"] == "complete" for row in all_rows)
            metrics = {}
            for key in ("mse_at_final_horizon", "mean_rollout_mse", "mean_directed_edges"):
                values = [float(np.mean([row[key] for row in rows]))
                          if len(rows) == trajectory_count and all(row["status"] == "complete" for row in rows)
                          else None for rows in seed_rows]
                metrics[key] = strict_seed_statistics(values)
            for step in (50, 200):
                values = [float(np.mean([row["mse_at_steps"][str(step)] for row in rows]))
                          if all(row["mse_at_steps"].get(str(step)) is not None for row in rows)
                          else None for rows in seed_rows]
                metrics[f"mse_at_{step}"] = strict_seed_statistics(values)
            groups[kind][policy] = {
                "attempted": len(all_rows), "completed": complete,
                "failed": len(all_rows) - complete, "metrics": metrics,
                "completed_prediction_steps": sum(row["completed_steps"] for row in all_rows),
                "total_wall_seconds_including_failures": float(sum(row["total_wall_seconds"] for row in all_rows)),
            }
            for seed, rows in zip(protocol["seeds"], seed_rows):
                failures.extend({"objective": kind, "seed": seed, "policy": policy,
                                 "trajectory_id": row["trajectory_id"], **row["failure"]}
                                for row in rows if row["failure"] is not None)
    flat = [group for objective in groups.values() for group in objective.values()]
    return {"horizon": protocol["horizon"], "groups": groups,
            "attempted": sum(group["attempted"] for group in flat),
            "completed": sum(group["completed"] for group in flat),
            "failed": sum(group["failed"] for group in flat),
            "completed_prediction_steps": sum(group["completed_prediction_steps"] for group in flat),
            "total_wall_seconds_including_failures": sum(group["total_wall_seconds_including_failures"] for group in flat),
            "failure_categories": dict(collections.Counter(row["category"] for row in failures)),
            "failure_events": failures}


def paired_deltas(short):
    result = {}
    for kind, groups in short["groups"].items():
        risk = groups["laggedrisk25"]["metrics"]["mse_at_200"]["seed_values"]
        result[kind] = {}
        for comparator in ("base", "dense", "random25", "speed25"):
            control = groups[comparator]["metrics"]["mse_at_200"]["seed_values"]
            delta = [r - c if r is not None and c is not None else None for r, c in zip(risk, control)]
            result[kind][comparator] = {**strict_seed_statistics(delta),
                                       "seeds_favoring_lagged_risk": sum(d < 0 for d in delta if d is not None)}
    return result


def check_prefixes(short_runs, long_runs, horizon):
    count, maximum = 0, 0.
    for name, short in short_runs.items():
        long = long_runs[name]
        if short["checkpoint_sha256"] != long["checkpoint_sha256"]:
            raise ValueError("Horizon experiments use different checkpoints")
        long_rows = {(row["trajectory_id"], row["policy"]): row for row in long["records"]}
        for row in short["records"]:
            other = long_rows[(row["trajectory_id"], row["policy"])]
            if len(other["mse_per_step"]) < len(row["mse_per_step"]):
                raise ValueError("Long run failed before a completed short-run prefix")
            difference = np.abs(np.asarray(row["mse_per_step"]) - other["mse_per_step"][:horizon])
            maximum = max(maximum, float(difference.max(initial=0)))
            if row["directed_edges_per_step"] != other["directed_edges_per_step"][:horizon]:
                raise ValueError("Graph counts differ on common forecast prefix")
            count += 1
    if maximum != 0:
        raise ValueError("Forecast errors differ on common prefix")
    return {"trajectory_policy_prefixes_checked": count, "forecast_steps": horizon,
            "maximum_absolute_mse_difference": maximum, "edge_counts_identical": True}


def number(statistic, digits=6):
    if statistic["mean"] is None:
        return "undefined"
    return f"{statistic['mean']:.{digits}f} ± {statistic['sample_seed_sd']:.{digits}f}"


def write_report(path, result, summary_path):
    short, long = result["horizons"]["200"], result["horizons"]["995"]
    lines = ["# Autonomous rollout results for the compact WaterDrop pilot", "",
             f"**All 15 frozen checkpoints were evaluated under all five policies on all three held-out trajectories: {short['attempted']} rollout attempts at 200 steps and {long['attempted']} at 995 steps.** "
             f"At 200 steps, {short['completed']} completed and {short['failed']} failed. At 995 steps, {long['completed']} completed and {long['failed']} triggered a predeclared guard.", "",
             "This is an exploratory extension of the small pilot, added after inspecting its one-step results. It is not an independent untouched-test confirmation or a reproduction of the original full architecture. "
             "Each checkpoint is a 48-wide, three-message-passing-layer model trained for 3,000 updates on the same small training subset. Five simulator-training seeds and the first three official test trajectories are retained throughout.", "",
             "Only the initial six frames come from ground truth. Subsequent histories, graphs, speed scores and uncertainty scores come from each policy's own predictions. "
             "Integration exactly inverts the normalized discrete second-difference target: x_next=2x_current−x_previous+acc_std·a_normalized+acc_mean; no extra dt factor is applied. "
             "Lagged risk uses a base graph on the first step, then its own previous graph's predicted risk. Other policies apply immediately. "
             "The 995-step reruns reproduce every retained 200-step MSE and graph-count prefix exactly.", "",
             "## The complete 200-step evaluation", "",
             "Position coordinate MSE is averaged over particles, then equally across the three trajectories for each seed. The table shows mean ± sample SD of these five seed means; these are not confidence intervals. "
             "All 200-step outcomes completed, so these comparisons include every requested run.", "",
             "| Objective | Base | Dense | Random25 | Speed25 | Lagged risk25 |",
             "|---|---:|---:|---:|---:|---:|"]
    for kind, groups in short["groups"].items():
        lines.append(f"| {kind} | " + " | ".join(number(groups[p]["metrics"]["mse_at_200"]) for p in POLICIES) + " |")
    lines += ["", "Paired differences are lagged-risk MSE@200 minus its comparator for each training seed, after averaging the same three trajectories. "
              "Negative favors lagged risk. These are descriptive paired differences, with no significance claim.", "",
              "| Objective | Δ versus base | Δ versus random25 | Δ versus speed25 |",
              "|---|---:|---:|---:|"]
    for kind, deltas in result["paired_200_deltas"].items():
        lines.append(f"| {kind} | " + " | ".join(number(deltas[c]) for c in ("base", "random25", "speed25")) + " |")
    lines += ["", "Risk allocation does not produce a consistent rollout improvement across objectives. The direction and magnitude change across training seeds and controls. "
              "The mean MSE@200 of the faithful and beta-NLL lagged-risk policies is worse than their own base-graph policy; NLL benefits on average, with its dense control performing better still.", "",
              "## Full available horizon: failures are outcomes", "",
              "Each cell below is the number of failed trajectory runs out of 15 (five seeds × three trajectories). No failed run is omitted or replaced by its last valid error.", "",
              "| Objective | Base | Dense | Random25 | Speed25 | Lagged risk25 |",
              "|---|---:|---:|---:|---:|---:|"]
    for kind, groups in long["groups"].items():
        lines.append(f"| {kind} | " + " | ".join(f"{groups[p]['failed']}/15" for p in POLICIES) + " |")
    categories = ", ".join(f"{key}: {value}" for key, value in long["failure_categories"].items()) or "none"
    lines += ["", f"Failure categories: {categories}. "
              "The predeclared guards stop on nonfinite states/predictions/risk, any coordinate with absolute value greater than 10, or more than 100,000 candidate undirected pairs. "
              "These are computational guards, not physical-validity criteria: completing 995 steps does not establish that particles remain inside the container or obey conservation laws.", "",
              "Full-horizon accuracy is reported only for objective/policy groups with all 15 trajectories complete. If even one fails, the all-sample final and mean-rollout MSE is undefined in the JSON summary; no survivor-only replacement is reported.", "",
              "| Complete objective/policy group | MSE@995 | Mean rollout MSE | Mean directed edges/step |",
              "|---|---:|---:|---:|"]
    complete_groups = 0
    for kind, groups in long["groups"].items():
        for policy, group in groups.items():
            if group["failed"] == 0:
                complete_groups += 1
                metrics = group["metrics"]
                lines.append(f"| {kind}/{policy} | {number(metrics['mse_at_final_horizon'])} | "
                             f"{number(metrics['mean_rollout_mse'])} | {number(metrics['mean_directed_edges'],1)} |")
    if not complete_groups:
        lines.append("| No group completed all 15 trajectories | undefined | undefined | undefined |")
    lines += ["", "## Compute and interpretation", "",
              f"The 200-step experiment executed {short['completed_prediction_steps']:,} accepted prediction steps, with {short['total_wall_seconds_including_failures']:.1f} summed trajectory wall seconds. "
              f"The 995-step experiment executed {long['completed_prediction_steps']:,} accepted prediction steps, with {long['total_wall_seconds_including_failures']:.1f} summed trajectory wall seconds including failed attempts. "
              "Each run used one CPU thread; other work ran concurrently. Timings include candidate counting/building, selection, feature preparation, neural prediction, integration and error calculation, but exclude checkpoint loading. "
              "The reference builder searches the expanded candidate radius even for the base policy. These are execution measurements, not an optimized GPU or production-speed comparison; early failures also shorten runtime.", "",
              "Random, speed and risk policies select exactly floor(0.25 × available annulus pairs) on their own predicted state, preserve every base pair, emit both directions, and omit self-loops. "
              "Because autonomous policies visit different positions, their total edge counts are not matched. The separate equal-state experiment controls edge budget for attribution. "
              "The raw report includes MSE@50, MSE@200, final MSE, mean rollout MSE, edge counts, per-step traces, complete failure events and checkpoint hashes. "
              "Neither this compact pilot nor these failure counts establish performance of a fully trained original AdaptGNS model.", "",
              f"[Complete numerical summary]({summary_path.resolve()}) · [Rollout code]({Path(__file__).with_name('pilot_rollout.py').resolve()}) · "
              f"[Summary code]({Path(__file__).resolve()})", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    short_protocol, short_runs, short_sources = read_experiment(args.result_root / "waterdrop_rollout200")
    long_protocol, long_runs, long_sources = read_experiment(args.result_root / "waterdrop_rollout995")
    for key in ("seeds", "objectives", "trajectories", "data_sha256", "metadata_sha256", "code_sha256"):
        if short_protocol[key] != long_protocol[key]:
            raise ValueError(f"Experiments differ on {key}")
    short, long = aggregate(short_protocol, short_runs), aggregate(long_protocol, long_runs)
    result = {"scope": "Exploratory compact WaterDrop pilot; all requested outcomes retained",
              "aggregation": "equal trajectories within seed, equal training seeds; SD across five seed means",
              "failure_policy": "group final/mean accuracy undefined if any of its fifteen trajectories fails",
              "horizons": {"200": short, "995": long},
              "paired_200_deltas": paired_deltas(short),
              "prefix_consistency": check_prefixes(short_runs, long_runs, short_protocol["horizon"]),
              "provenance": {"200": short_sources, "995": long_sources,
                             "summarizer_sha256": sha256(__file__)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    write_report(args.report, result, args.output)
    print(json.dumps({h: {k: v[k] for k in ("attempted", "completed", "failed")}
                      for h, v in result["horizons"].items()}))


if __name__ == "__main__":
    main()
