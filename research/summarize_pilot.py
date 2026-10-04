"""Regenerate pilot summary and figures without selecting or tuning policies."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trajectory_mean(records, policy):
    by_trajectory = defaultdict(list)
    for record in records:
        by_trajectory[record["trajectory"]].append(record["policies"][policy]["mse_normalized_acceleration"])
    return {key: float(np.mean(values)) for key, values in by_trajectory.items()}


def summarize(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted(folder.glob("*_seed*.json"))]
    expected = {(o,s) for o in protocol["objectives"] for s in protocol["seeds"]}
    found = {(r["objective"],r["seed"]) for r in runs}
    if found != expected:
        raise ValueError(f"Incomplete pilot: missing {sorted(expected-found)}")
    policies = list(runs[0]["test"][0]["policies"])
    for run in runs:
        if run["curve"][-1]["step"] != protocol["steps"]:
            raise ValueError("Incomplete training trace")
        for split in ["validation", "test"]:
            for row in run[split]:
                counts = {row["policies"][p]["directed_edges"] for p in policies if p.endswith("25")}
                if len(counts) != 1:
                    raise ValueError("Policies are not exactly edge matched")
                if row["policies"]["base"]["directed_edges"] > next(iter(counts)):
                    raise ValueError("Mandatory edge count violated")
    result = {"complete": True, "n_training_runs": len(runs), "protocol": protocol,
                "interpretation": "Five training seeds at fixed 3000 steps; test inference remains conditional on 3 convenience-sampled trajectories. No benchmark or rollout claims.",
                "aggregation": "Each frame first averages over particles and coordinates; frames equally within trajectory, trajectories equally within seed; sample SD over seed means.",
                "sources": {p.name: digest(p) for p in sorted(folder.glob("*.json"))},
                "summaries": {}, "paired_differences": {}}
    for split in ["validation", "test"]:
        result["summaries"][split] = {}
        result["paired_differences"][split] = {}
        for objective in protocol["objectives"]:
            subset = sorted([r for r in runs if r["objective"] == objective], key=lambda r:r["seed"])
            metrics = {}
            for policy in policies:
                seed_values = [float(np.mean(list(trajectory_mean(r[split],policy).values()))) for r in subset]
                metrics[policy] = {"mean": float(np.mean(seed_values)), "sd": float(np.std(seed_values,ddof=1)),
                                   "seed_values": seed_values, "n_seeds": len(seed_values)}
            result["summaries"][split][objective] = metrics
            result["paired_differences"][split][objective] = {}
            for policy in ["current_risk25", "lagged_base_risk25", "speed25"]:
                d = np.array(metrics[policy]["seed_values"])-np.array(metrics["random25"]["seed_values"])
                result["paired_differences"][split][objective][policy] = {
                    "mean_difference": float(d.mean()), "sample_sd": float(d.std(ddof=1)),
                    "seed_differences": list(map(float,d)), "wins": int((d<0).sum()),
                    "inference_caution": "Paired descriptive differences only; n=5, fixed tiny test subset, multiple comparisons."}
    result["total_training_seconds"] = sum(r["training_seconds"] for r in runs)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--folder", type=Path, default=Path(__file__).parent / "results/waterdrop_pilot")
    p.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[2] / "pilot_results.md")
    p.add_argument("--tex-output", type=Path)
    args = p.parse_args()
    result = summarize(args.folder)
    result["additional_provenance"] = {
        "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "data": {p.name:digest(p) for p in sorted(args.data_dir.glob("*-pilot.npz"))},
        "metadata_sha256": digest(args.data_dir/"metadata.json"),
        "code": {p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
    }
    from research.pilot import load_frames
    meta = json.loads((args.data_dir / "metadata.json").read_text())
    reference = {"scope": "Post-hoc descriptive reference: zero normalized acceleration, no fitting or selection", "splits": {}}
    for split, rng_seed in [("valid", 811), ("test", 812)]:
        frames = load_frames(args.data_dir / (split + "-pilot.npz"), meta, 12, rng_seed)
        groups = defaultdict(list)
        for frame in frames:
            groups[frame["trajectory"]].append(float(frame["target"].square().mean()))
        reference["splits"][split] = {"trajectory_means": {k: float(np.mean(v)) for k,v in groups.items()}, "mean": float(np.mean([np.mean(v) for v in groups.values()])), "frame_ids": [f["id"] for f in frames]}
    (args.folder.parent/"constant_acceleration_reference.json").write_text(json.dumps(reference,indent=2)+"\n")
    result["constant_acceleration_reference"] = reference
    (args.folder.parent/"pilot_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    names = {"base":"Base", "dense":"Dense", "random25":"Random", "speed25":"Speed", "current_risk25":"Current risk", "lagged_base_risk25":"Previous-base risk"}
    lines = ["# New WaterDrop pilot: five seeds, controlled edge budgets", "",
        "This is new local training on public WaterDrop data, not a reproduction of the full paper. Fifteen compact models completed 3,000 updates each. The 48-wide, three-block architecture and zero-noise recipe differ from the original 128-wide, ten-block GNS. Eight training, three validation, and three test trajectories were selected by record order before fitting. The test set here is a convenience subset of the original public test split, not a new untouched benchmark test population.", "",
        "Normalized one-step acceleration MSE is averaged equally over the 12 selected frames in each trajectory, then equally over the three trajectories per seed. The table reports means ± sample SD across five independent training seeds. It does not report position rollout MSE, large-dataset generalization, or statistical significance of a deployment improvement.", "",
        "| Objective | Base | Random budget | Speed budget | Current-risk budget | Previous-base-risk budget | Dense |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    texrows = []
    for objective, metrics in result["summaries"]["test"].items():
        ordered = ["base","random25","speed25","current_risk25","lagged_base_risk25","dense"]
        lines.append("| "+objective+" | "+" | ".join(f"{metrics[k]['mean']:.4f} ± {metrics[k]['sd']:.2f}" for k in ordered)+" |")
        texrows.append(objective.replace("_", "-")+" & "+" & ".join(f"${metrics[k]['mean']:.3f}\\pm{metrics[k]['sd']:.3f}$" for k in ordered)+" "+r"\\")
    lines += ["", "All four budgeted policies use exactly floor(0.25 × available annulus pairs) extra undirected pairs on the same observed state. Each pair is emitted in both directions and all base pairs are preserved; no self loops. The max-endpoint risk score uses deterministic particle-ID tie-breaking, which does not preserve permutation equivariance at ties.", "",
              "Current risk needs a base-graph scoring pass. Previous-base risk comes from the previous ground-truth state with a base graph; it does not simulate the original autonomous adaptive controller. These are diagnostic policies. Random selection uses one draw per frame and model seed, so seed variability combines fitting and selection randomness.", "",
              "The recorded forward times include edge-feature/tensor creation and a model call, but exclude candidate search, edge selection, and the scoring prepass. They are not an end-to-end speed benchmark. A separate CPU graph-construction experiment is recorded in graph_benchmark.json.", "",
              f"Summed optimizer-loop time: {result['total_training_seconds']:.1f} seconds; CPU, two Torch threads. This excludes data preparation, process startup, and evaluation. All 15 checkpoints, raw frame metrics, frame IDs, seeds, and data/code hashes are retained locally.", "",
              "No test-dependent checkpoint selection was used. Hyperparameters were fixed before the full run. Results from this pilot must remain distinct from the historical full-architecture arrays and from the two-update original-pipeline smoke test."]
    args.report.write_text("\n".join(lines)+"\n")
    if args.tex_output:
        args.tex_output.write_text(r'''\begin{table*}[t]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{New small WaterDrop pilot: normalized one-step acceleration MSE, five-seed means and sample standard deviations. Lower is better. Four allocation policies share the same exact optional-pair budget on identical observed states. Each seed evaluates only three trajectories; this is not an autonomous rollout benchmark. Full seed statistics appear in the accompanying results.}
\label{tab:pilot}
\begin{tabular}{rrrrrrr}
\toprule
Objective & Base & Random & Speed & Current risk & Previous-base risk & Dense\\
\midrule
'''+"\n".join(texrows)+r'''
\bottomrule
\end{tabular}
\end{table*}
Table~\ref{tab:pilot} reports these new measurements. Their scope is intentionally limited: no result in this table is a full-scale retraining result or a rollout accuracy claim. The accompanying artifact reports all per-seed values and uses the same recorded final checkpoint for every policy.
''')
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(10,4.4),layout="constrained")
    x=np.arange(6)
    for idx,(objective, metrics) in enumerate(result["summaries"]["test"].items()):
        ordered=["base","random25","speed25","current_risk25","lagged_base_risk25","dense"]
        ax.plot(x, [metrics[k]["mean"] for k in ordered], marker="o", label=objective.replace("_", " "))
        for seed in range(5):
            ax.plot(x,[metrics[k]["seed_values"][seed] for k in ordered],alpha=.12,color=f"C{idx}")
    ax.set_xticks(x, [names[k] for k in ordered]); ax.set_ylabel("One-step normalized acceleration MSE")
    ax.set_title("WaterDrop pilot: five seeds, three held-out trajectories")
    ax.legend(loc="best"); fig.savefig(args.folder.parent/"pilot_comparison.png",dpi=180)
    fig.savefig(args.folder.parent/"pilot_comparison.pdf")
    print(json.dumps(result["summaries"]["test"],indent=2))


if __name__=="__main__":
    main()
