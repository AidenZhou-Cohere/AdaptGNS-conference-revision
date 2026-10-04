#!/usr/bin/env python3
"""Render standalone figures from reanalysis.json; does not recompute statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {"adaptive": "#009E73", "fixed_mse_base": "#0072B2", "fixed_mse_dense": "#D55E00", "fixed_nll_base": "#CC79A7", "sweep": "#E69F00"}
LABELS = {"adaptive": "Adaptive (legacy objective)", "fixed_mse_base": "Fixed MSE, r=0.015", "fixed_mse_dense": "Fixed MSE, r=0.019", "fixed_nll_base": "Fixed legacy NLL, r=0.015"}
STYLES = {"adaptive": "-", "fixed_mse_base": "--", "fixed_mse_dense": "-.", "fixed_nll_base": ":"}
KINDS = ("adaptive", "fixed_mse_base", "fixed_mse_dense", "fixed_nll_base")


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "grid.color": "#D5D9DC",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def record(result: dict, dataset: str, kind: str) -> dict:
    matches = [r for r in result["artifacts"] if r["dataset"] == dataset and r["kind"] == kind and (kind != "adaptive" or r["selected_in_paper"])]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one selected record for {dataset}/{kind}")
    return matches[0]


def save(fig: plt.Figure, directory: Path, stem: str, caption: str) -> dict:
    files = {}
    for extension in ("png", "pdf"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches=None, metadata={"Title": stem, "Author": "AdaptGNS research audit", "Subject": caption} if extension == "pdf" else None)
        files[extension] = path.name
    plt.close(fig)
    return {"files": files, "caption": caption}


def rollout_curves(result: dict, directory: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7))
    fig.subplots_adjust(left=0.082, right=0.98, top=0.71, bottom=0.25, wspace=0.29)
    for panel, (ax, dataset) in enumerate(zip(axes, ("WaterDrop", "Sand"))):
        largest = 0.0
        for kind in KINDS:
            artifact = record(result, dataset, kind)
            values = np.asarray(artifact["mse_mean_by_step"])
            steps = np.arange(1, len(values) + 1)
            largest = max(largest, float(values.max()))
            ax.plot(steps, values, color=COLORS[kind], linestyle=STYLES[kind], linewidth=2.1 if kind == "adaptive" else 1.75, label=LABELS[kind])
        horizon = record(result, dataset, "adaptive")["last_predicted_step"]
        ax.axvline(200, color="#92989C", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
        ax.set(xlim=(0, horizon), ylim=(0, largest * 1.09), xlabel="Predicted step after the input history", ylabel="Position MSE")
        ax.set_title(f"({chr(97 + panel)}) {dataset}: {horizon} predicted steps", loc="left", pad=10)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xticks([0, 200, 400, 600, 800, 995] if horizon == 995 else [0, 100, 200, 314])
    handles = [Line2D([0], [0], color=COLORS[k], linestyle=STYLES[k], linewidth=2, label=LABELS[k]) for k in KINDS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.905), ncol=2, frameon=False, handlelength=3.2, columnspacing=2.3)
    fig.suptitle("Saved rollout errors across the complete prediction horizon", x=0.082, ha="left", y=0.985, fontsize=12, fontweight="bold")
    caption = "Means across 30 saved trajectories per method, conditional on one saved checkpoint per method. No training-seed variability is shown. Vertical guides mark step 200. WaterDrop contains 995 and Sand 314 predicted steps after the input history. These are legacy results, not corrected-objective training."
    fig.text(0.082, 0.105, "Means across 30 saved trajectories per method; one saved checkpoint per method. No training-seed variability is shown.\nVertical guides mark step 200. These legacy artifacts contain 995 (WaterDrop) and 314 (Sand) future predictions.", ha="left", va="center", fontsize=8.1, linespacing=1.65, color="#444444")
    return save(fig, directory, "legacy_full_rollout_curves", caption)


def waterdrop_intervals(result: dict, directory: Path) -> dict:
    comparison = next(c for c in result["comparisons"] if c["dataset"] == "WaterDrop" and c["baseline_kind"] == "fixed_mse_base")
    keys = ("mse_at_200", "mse_at_500", "mse_at_995", "rollout_mean_mse")
    row_labels = ("MSE at step 200", "MSE at step 500", "MSE at step 995", "Mean MSE over 995 steps")
    means = np.array([comparison["metrics"][key]["mean"] for key in keys]) * 1000
    intervals = np.array([comparison["metrics"][key]["conditional_trajectory_bootstrap_95ci"] for key in keys]) * 1000
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    fig.subplots_adjust(left=0.26, right=0.965, top=0.80, bottom=0.35)
    y = np.arange(len(keys))[::-1]
    ax.errorbar(means, y, xerr=np.vstack((means - intervals[:, 0], intervals[:, 1] - means)), fmt="o", color=COLORS["fixed_mse_base"], markersize=6.2, elinewidth=1.8, capsize=4.5, capthick=1.3, zorder=3)
    ax.axvline(0, color="#52585C", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(y, labels=row_labels)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(min(-1.2, float(intervals.min()) - 0.25), float(intervals.max()) + 0.45)
    ax.set_xlabel(r"Adaptive minus sparse MSE baseline (position MSE $\times\,10^{-3}$)", labelpad=9)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    fig.suptitle("WaterDrop: conditional trajectory-bootstrap differences", x=0.055, ha="left", y=0.98, fontsize=12, fontweight="bold")
    fig.text(0.26, 0.863, "Mean differences and 95% percentile intervals", ha="left", fontsize=9, color="#444444")
    caption = "100,000 paired bootstrap resamples of 30 whole trajectories, conditional on the saved models. Trajectory pairing is assumed from row order because IDs are absent. Intervals are pointwise and unadjusted for multiple comparisons; they are not uncertainty over training seeds. Negative values favor Adaptive. The rollout mean averages all 995 predicted steps and is not an unnormalized physical-time integral. All numerical statistics are reused from reanalysis.json."
    fig.text(0.055, 0.14, "Negative values favor Adaptive; positive values favor the sparse MSE baseline.\n100,000 resamples of 30 whole trajectories. Pairing by row order is assumed because trajectory IDs are absent.\nIntervals are conditional on the saved models, pointwise and unadjusted; they do not measure training-seed variability.", ha="left", va="center", fontsize=8.1, linespacing=1.65, color="#444444")
    return save(fig, directory, "waterdrop_conditional_delta_intervals", caption)


def sand_tradeoff(result: dict, directory: Path) -> dict:
    sweep = [a for a in result["artifacts"] if a["dataset"] == "Sand" and a["kind"] == "adaptive"]
    selected = record(result, "Sand", "adaptive")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    fig.subplots_adjust(left=0.082, right=0.98, top=0.73, bottom=0.25, wspace=0.29)
    for panel, (ax, metric, title, ymax) in enumerate(zip(axes, ("mse_at_200", "rollout_mean_mse"), ("MSE at step 200", "Mean MSE over 314 steps"), (0.05, 0.036))):
        x = [a["metrics"]["mean_edges_per_step"]["mean"] / 1000 for a in sweep]
        y = [a["metrics"][metric]["mean"] for a in sweep]
        ax.scatter(x, y, s=32, color=COLORS["sweep"], edgecolors="#7B5900", linewidths=0.5, alpha=0.8, zorder=3)
        for kind, marker in (("fixed_mse_base", "o"), ("fixed_mse_dense", "^"), ("fixed_nll_base", "s")):
            a = record(result, "Sand", kind)
            ax.scatter(a["metrics"]["mean_edges_per_step"]["mean"] / 1000, a["metrics"][metric]["mean"], s=55, marker=marker, color=COLORS[kind], edgecolors="white", linewidths=0.6, zorder=4)
        ax.scatter(selected["metrics"]["mean_edges_per_step"]["mean"] / 1000, selected["metrics"][metric]["mean"], s=190, marker="*", color=COLORS["adaptive"], edgecolors="#005A40", linewidths=0.65, zorder=5)
        ax.set(xlim=(0, 32), ylim=(0, ymax), xlabel="Mean realized rollout edges per step (thousands)", ylabel="Position MSE")
        ax.set_title(f"({chr(97 + panel)}) {title}", loc="left", pad=10)
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xticks([0, 10, 20, 30])
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["sweep"], markeredgecolor="#7B5900", markersize=6, label="11 adaptive test configurations"), Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["adaptive"], markeredgecolor="#005A40", markersize=11, label="Configuration chosen in paper")]
    handles += [Line2D([0], [0], marker=marker, color="none", markerfacecolor=COLORS[kind], markeredgecolor="none", markersize=6, label=LABELS[kind]) for kind, marker in (("fixed_mse_base", "o"), ("fixed_mse_dense", "^"), ("fixed_nll_base", "s"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.905), ncol=3, frameon=False, columnspacing=1.35, handletextpad=0.45)
    fig.suptitle("Sand: post-hoc tradeoffs in the saved test artifacts", x=0.082, ha="left", y=0.985, fontsize=12, fontweight="bold")
    caption = "Observed point estimates from the existing Sand test artifacts. The 11 adaptive configurations vary the node percentile and radius factor; the star marks the configuration reported in the original paper. This is a post-hoc sensitivity display, not validation-only tuning or a new selection of hyperparameters. Realized edge counts come from different predicted geometries, not fixed-state compute matching. Both axes include zero. No training-seed uncertainty is available."
    fig.text(0.082, 0.095, "Post-hoc test-artifact display, not validation-only tuning or a new hyperparameter selection. Both axes include zero.\nRealized edge counts use different predicted geometries; they are not equal-state compute budgets. No seed uncertainty is available.", ha="left", va="center", fontsize=8.0, linespacing=1.65, color="#444444")
    return save(fig, directory, "sand_posthoc_tradeoff", caption)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results/reanalysis.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/figures")
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure()
    figures = [rollout_curves(result, args.output_dir), waterdrop_intervals(result, args.output_dir), sand_tradeoff(result, args.output_dir)]
    manifest = {"statistics_recomputed": False, "input_filename": args.input.name, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "matplotlib_version": matplotlib.__version__, "figures": figures}
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "captions.md").write_text("# Figure captions\n\n" + "\n\n".join(f"## {f['files']['pdf']}\n\n{f['caption']}" for f in figures) + "\n")
    print(f"Rendered {len(figures)} standalone PNG/PDF figure pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
