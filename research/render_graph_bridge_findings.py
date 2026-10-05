"""Render hash-pinned exploratory graph-convention findings, without inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPECTED_SUMMARY = "1623cab0517dd4a20ceb668e322aa98ff064a6dc4e54b62b63d46d7c05ff27d5"
OBJECTIVES = ("faithful", "nll")
RISK = "previous-observed-base-risk25"
POLICIES = (("base", "Base"), ("dense", "Dense"), ("random25", "Random25"),
            ("speed25", "Speed25"), (RISK, "Previous risk25"),
            ("current-base-risk25", "Current risk25"))


def read_summary(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    if hashlib.sha256(data).hexdigest() != EXPECTED_SUMMARY:
        raise ValueError("Pinned completed summary changed")
    summary = json.loads(data)
    if not summary["audit"]["passed"] or len(summary["runs"]) != 6:
        raise ValueError("Complete audited cohort required")
    return summary


def measure(summary, objective, split, policy, loops):
    return summary["groups"][objective][split]["measures"][
        f"case__{policy}_uncapped_loops{loops}__normalized_coordinate_mse"]


def pm(value, scale=1., digits=3):
    return f"${value['mean']*scale:.{digits}f}\\pm{value['sample_sd']*scale:.{digits}f}$"


def render(summary, audit, output):
    if not audit["passed"] or audit["summary_sha256"] != EXPECTED_SUMMARY:
        raise ValueError("Independent audit must match summary")
    # Check every headline against all paired seed values before rendering.
    for obj in OBJECTIVES:
        for split in ("valid", "test"):
            base0 = np.array(measure(summary,obj,split,"base",0)["seed_values"])
            base1 = np.array(measure(summary,obj,split,"base",1)["seed_values"])
            expected = (base0-base1)/base0*100
            np.testing.assert_allclose(expected, audit["findings"][f"{obj}/{split}"][
                "base_mse_relative_reduction_percent"]["seed_values"], atol=1e-12, rtol=0)
            for key in ("loops1_risk_minus_random", "loops1_dense_minus_base"):
                if not all(v > 0 for v in audit["findings"][f"{obj}/{split}"][key]["seed_values"]):
                    raise ValueError("All-seed direction claim no longer true")
            if not all(a > b for a,b in zip(
                    measure(summary,obj,split,"current-base-risk25",1)["seed_values"],
                    measure(summary,obj,split,"random25",1)["seed_values"])):
                raise ValueError("Current-risk all-seed direction claim no longer true")
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), constrained_layout=True)
    for obj, color, marker, offset in zip(OBJECTIVES,("#245B8A","#AC4E26"),("o","^"),(-.05,.05)):
        for seed in range(3):
            base = [measure(summary,obj,"test","base",j)["seed_values"][seed]*1000 for j in (0,1)]
            gap = [(measure(summary,obj,"test",RISK,j)["seed_values"][seed]-
                    measure(summary,obj,"test","random25",j)["seed_values"][seed])*1000 for j in (0,1)]
            for ax, values in zip(axes, (base,gap)):
                ax.plot(np.array([0,1])+offset, values, marker=marker, color=color,
                        alpha=.75, lw=1.1, ms=4.5, label=obj.upper() if seed==0 else None)
    for ax in axes:
        ax.set_xticks([0,1], ["Self-loops off", "Self-loops on"])
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=.6)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Base normalized coordinate MSE (×1,000)", fontsize=9)
    axes[0].set_title("Native self-messages improve base accuracy", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0,color="#777777",lw=.8)
    axes[1].set_ylabel("Risk25 minus random25 MSE (×1,000)", fontsize=9)
    axes[1].set_title("Risk placement still loses to random", fontsize=10)
    fig.savefig(output/"graph_convention_bridge.png",dpi=180)
    fig.savefig(output/"graph_convention_bridge.pdf")
    plt.close(fig)
    f = audit["findings"]
    main = r'''\subsection{Does the graph convention explain the deficit?}
The fixed study removed self-loops used during training. We therefore froze a
separate post-inspection bridge on all six checkpoints and 425 observed
histories per model (128 validation, 297 test), crossing self-loops and the
128-neighbor cap for base/dense, with both loop settings for all selectors.
All 2,550 frames completed. Restoring self-loops lowers base test MSE by
'''+pm(f["faithful/test"]["base_mse_relative_reduction_percent"],digits=2)+r'''\% / '''+pm(f["nll/test"]["base_mse_relative_reduction_percent"],digits=2)+r'''\%
(faithful/NLL; paired within-seed changes).
Yet previous-observed risk remains worse than random, and dense worse than base,
in every seed on both splits. Current-base risk also fails to beat random.
The cap is inactive on all observed histories; native graph/feature parity
checks pass. Thus self-messages materially affect absolute accuracy without
explaining the allocation ordering. These controls do not resolve the lack of
expanded-graph training or autonomous feedback; Appendix~\ref{sec:graph-bridge}
gives all policies and the paired loop interactions.
'''
    (output/"graph_bridge_main.tex").write_text(main)
    appendix = r'''\section{Exploratory graph-convention bridge}
\label{sec:graph-bridge}
The six frozen 100,000-update checkpoints are unchanged. Each is evaluated on
the same 128 validation and 297 test histories in a separately frozen 16-case
screen. Separate strict-radius queries reproduce the native float32 distance
filter, receiver ordering, self-edge candidates and cap. The factorial uses
base/dense with cap128 or uncapped and loops off/on; random, speed,
current-base risk and previous-observed-base risk are uncapped in both loop arms.
Every sparse selector uses the exact quarter-annulus budget. The loop contrast
for risk changes its scoring graph and can change selected pairs; random and
speed retain identical nonself selections across loop arms.

\begin{table}[ht]
\centering\small
\caption{Normalized acceleration coordinate MSE, multiplied by $10^3$.
Means and sample SD are across three seed means after equal-trajectory
averaging. Validation has 4--5 histories per trajectory; test has 11.
Cap128 and uncapped base/dense coincide throughout this observed-state screen.}
\label{tab:graph-convention-bridge}
\begin{tabular}{llrrrr}
\toprule
Split & Policy & Faithful, off & Faithful, on & NLL, off & NLL, on\\
\midrule
'''
    for split, label in (("valid","Validation"),("test","Test")):
        for policy, title in POLICIES:
            appendix += label+" & "+title+" & "+" & ".join(
                pm(measure(summary,obj,split,policy,j),1000,3)
                for obj in OBJECTIVES for j in (0,1))+r"\\"+"\n"
        appendix += r"\midrule"+"\n"
    appendix += r'''\end{tabular}
\end{table}

The native cap never binds on any of the 425 geometries; the two radius
classifiers return identical pairs. All native graph/feature parity gates pass,
with maximum prediction discrepancy $5.96\times10^{-8}$ and maximum converted
risk discrepancy $2.09\times10^{-6}$ under the frozen combined tolerance.
The initial cap-inactivity inspection already covered the 297 test geometries;
the bridge is not independent confirmation of that observation. Its random
selector uses a new paired draw, so its random values do not replay the
original fixed-study draw. Prior inspection of test trajectories and aggregates
also applies. No current- or previous-risk selector beats random in any seed
on either split with loops restored.

\begin{table}[ht]
\centering\small
\caption{Paired loop interactions: (policy difference with loops on) minus
(the same difference with loops off), in normalized coordinate MSE $\times10^3$.
Positive risk--random values mean loops increase the risk deficit.}
\label{tab:graph-loop-interactions}
\begin{tabular}{llrr}
\toprule
Split & Interaction & Faithful & NLL\\
\midrule
'''
    for split,label in (("valid","Validation"),("test","Test")):
        for key,title in ((RISK+"_minus_random25","Previous risk -- random"),
                          ("dense_minus_base","Dense -- base"),
                          ("current-base-risk25_minus_random25","Current risk -- random")):
            appendix += label+" & "+title+" & "+" & ".join(
                pm(summary["groups"][obj][split]["measures"][
                    "contrast__loop_interaction__"+key+"__normalized_coordinate_mse"],1000,3)
                for obj in OBJECTIVES)+r"\\"+"\n"
    appendix += r'''\bottomrule
\end{tabular}
\end{table}

These paired contrasts show that absolute accuracy depends on graph inputs,
while the risk--random gap persists under the trained self-loop convention.
They are descriptive three-seed results; no population significance, native
rollout stability or benefit of graph-augmented training follows. All failures,
raw predictions, selection hashes and graph diagnostics are retained.
'''
    (output/"graph_bridge_appendix.tex").write_text(appendix)
    (output/"render_manifest.json").write_text(json.dumps({
        "summary_uncompressed_sha256":EXPECTED_SUMMARY,
        "renderer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"Post-inspection exploratory descriptive graph-convention bridge",
        "outputs":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(output.iterdir())}},indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    render(read_summary(args.summary),json.loads(args.audit.read_text()),args.output_dir)


if __name__=="__main__":
    main()
