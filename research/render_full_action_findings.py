"""Render the separate exploratory action analysis without refitting or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPECTED_ACTION = "c9a608d3f17cfc84001020b27048ef448958daa2705d135be5f4492e8bbdc008"
EXPECTED_SAME = "bef3c5958aa61a48cb3d1d294a6b458a0e8ca0069d4a631a227d356318b080b1"
OBJECTIVES = ("faithful", "nll")
COLORS = ("#245B8A", "#AC4E26")
RISK = "previous-observed-base-risk25"


def read(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Pinned input changed: " + str(path))
    value = json.loads(path.read_text())
    if value["state"] != "complete":
        raise ValueError("Complete six-model input required")
    return value


def pm(value, scale=1., digits=3):
    if value["mean"] is None or value["sample_seed_sd"] is None:
        return "---"
    return f"${value['mean'] * scale:.{digits}f}\\pm{value['sample_seed_sd'] * scale:.{digits}f}$"


def render(action, same, output):
    output.mkdir(parents=True, exist_ok=False)
    groups = action["objectives"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
    labels = ("Base residual", "Dense benefit", "Actual risk25 benefit")
    for obj, color, offset in zip(OBJECTIVES, COLORS, (-.12, .12)):
        g = groups[obj]
        series = [
            [v["value"] for v in same["objectives"][obj]["metrics"]["correlation/previous_risk_vs_base_error"]["seed_values"]],
            g["correlation/normalized/dense/previous_observed_base_q"]["seed_values"],
            g[f"correlation/normalized/{RISK}/previous_observed_base_q"]["seed_values"],
        ]
        for index, values in enumerate(series):
            axes[0].scatter(np.arange(3)*.025 + index + offset - .025, values, s=30, color=color,
                            marker="o" if obj == "faithful" else "^", label=obj.upper() if index == 0 else None)
            axes[0].plot([index+offset-.065,index+offset+.065], [np.mean(values)]*2, color=color, lw=2)
        for index, policy in enumerate(("dense", "random25", "speed25", RISK)):
            values = np.asarray(g[f"benefit/normalized/{policy}/mean_coordinate_benefit"]["seed_values"])*1000
            axes[1].scatter(np.arange(3)*.025 + index + offset - .025, values, s=30, color=color,
                            marker="o" if obj == "faithful" else "^")
            axes[1].plot([index+offset-.065,index+offset+.065], [np.mean(values)]*2, color=color, lw=2)
    for ax in axes:
        ax.axhline(0, color="#777777", lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=.6, zorder=0)
        ax.tick_params(labelsize=9)
    axes[0].set_xticks(range(3), labels, rotation=15)
    axes[0].set_ylabel("Previous-risk Spearman correlation")
    axes[0].set_title("Residual ranking differs from action value", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xticks(range(4), ("Dense", "Random25", "Speed25", "Risk25"))
    axes[1].set_ylabel("Normalized coordinate MSE gain (×1,000)")
    axes[1].set_title("Sparse and dense actions have different effects", fontsize=11)
    fig.savefig(output/"actual_action_mechanism.png", dpi=180)
    fig.savefig(output/"actual_action_mechanism.pdf")
    plt.close(fig)
    main = r'''\subsection{Exploratory actual-action benefit}
After inspecting the fixed study, we reanalyzed all saved predictions to measure
the signed benefit of each actual sparse graph, rather than only full expansion.
Previous-risk correlation with its own 25\% action's normalized vector benefit is
'''+pm(groups["faithful"][f"correlation/normalized/{RISK}/previous_observed_base_q"])+" for faithful and "+pm(groups["nll"][f"correlation/normalized/{RISK}/previous_observed_base_q"])+r''' for NLL.
The corresponding dense and actual-risk benefit signs disagree for
'''+pm(groups["faithful"][f"dense_proxy/normalized/{RISK}/sign_disagreement_fraction"],100,1)+r'''\% and
'''+pm(groups["nll"][f"dense_proxy/normalized/{RISK}/sign_disagreement_fraction"],100,1)+r'''\% of particles, respectively.
Thus dense benefit is an imperfect label for the sparse action, but replacing
it with actual action benefit does not recover strong risk ranking here.
Weak average rank association does not exclude nonmonotone structure: the highest
risk quartile has positive mean risk-action benefit for each objective, although
one NLL seed is negative and random allocation yields larger benefit in that
quartile in all six models.

A hindsight whole-frame choice among base, random25, speed25 and risk25 selects
base as the strictly best action on '''+pm(groups["faithful"]["oracle/normalized/budgeted_with_base/base_strictly_best"],100,1)+r'''\% / '''+pm(groups["nll"]["oracle/normalized/budgeted_with_base/base_strictly_best"],100,1)+r'''\% of frames (faithful/NLL).
This target-using diagnostic motivates studying when to add edges; it is not a
deployable selector or a free-rollout result. All values retain equal-trajectory
weighting and sample SD across three seeds. These post-inspection findings remain
conditional on the original evaluation graph convention.
'''
    # Native LaTeX picture keeps the manuscript standalone; no image asset or
    # extra package is required by the desktop editor's compiler.
    picture = [r"\begin{figure}[t]", r"\centering\setlength{\unitlength}{1pt}",
               r"\begin{picture}(225,167)",
               r"\put(12,155){\makebox(0,0)[l]{\scriptsize Previous-risk Spearman correlation}}",
               r"\put(30,31){\line(1,0){181}}", r"\put(30,31){\line(0,1){114}}"]
    for tick in (-.1,0,.1,.2,.3,.4,.5):
        y=31+(tick+.1)*180
        picture += [r"\put(27,%.2f){\makebox(0,0)[r]{\tiny %.1f}}" % (y,tick),
                    f"\\put(29,{y:.2f}){{\\line(1,0){{3}}}}"]
    picture.append(r"\multiput(32,49)(4,0){45}{\line(1,0){1}}")
    for oi,obj in enumerate(OBJECTIVES):
        g=groups[obj]
        series = [[v["value"] for v in same["objectives"][obj]["metrics"]["correlation/previous_risk_vs_base_error"]["seed_values"]],
                  g["correlation/normalized/dense/previous_observed_base_q"]["seed_values"],
                  g[f"correlation/normalized/{RISK}/previous_observed_base_q"]["seed_values"]]
        for index,values in enumerate(series):
            x=59+61*index+(-7 if oi==0 else 7)
            for seed,value in enumerate(values):
                mark = r"\circle*{3.5}" if oi==0 else r"\makebox(0,0){\tiny$\triangle$}"
                picture.append(f"\\put({x+2*(seed-1):.2f},{31+(value+.1)*180:.2f}){{{mark}}}")
            picture.append(f"\\put({x-4:.2f},{31+(np.mean(values)+.1)*180:.2f}){{\\line(1,0){{8}}}}")
    for x,label in ((59,"Base residual"),(120,"Dense benefit"),(181,"Actual risk25")):
        picture.append(r"\put(%s,21){\makebox(0,0){\tiny %s}}" % (x,label))
    picture += [r"\put(68,7){\circle*{3.5}}", r"\put(74,7){\makebox(0,0)[l]{\tiny Faithful}}",
                r"\put(136,7){\makebox(0,0){\tiny$\triangle$}}",r"\put(143,7){\makebox(0,0)[l]{\tiny NLL}}",
                r"\end{picture}",r"\caption{Residual ranking and actual action value on the saved observed histories. Each symbol is one seed mean; horizontal marks are three-seed means. Actual risk25 benefit uses the whole selected graph. This exploratory comparison retains the original no-self-loop evaluation convention.}",
                r"\label{fig:actual-action-risk}",r"\end{figure}"]
    main += "\n"+"\n".join(picture)+"\n"
    (output/"full_action_main.tex").write_text(main)
    rows = []
    for obj in OBJECTIVES:
        for policy, label in (("dense","Dense"),("random25","Random25"),("speed25","Speed25"),(RISK,"Previous risk25")):
            g = groups[obj]
            keys = (f"benefit/normalized/{policy}/mean_vector_benefit", f"benefit/normalized/{policy}/harmful_fraction", f"correlation/normalized/{policy}/previous_observed_base_q")
            rows.append(obj+" & "+label+" & "+" & ".join(pm(g[key],digits=5 if j==0 else 3) for j,key in enumerate(keys))+r"\\")
    appendix = r'''\section{Benefits of the actual saved graph actions}
\label{sec:actual-action-benefit}
This exploratory analysis uses all 1,782 saved observed-history cases, each with five policy predictions.
For each complete graph action $a$, its signed particle benefit is
$b_{i,a}=\|y_i-p_{i,0}\|^2-\|y_i-p_{i,a}\|^2$.
We compute the same quantity after dividing coordinate residuals by the stored
acceleration scales, and verify $b_{i,a}=2r_i^\top\Delta_{i,a}-\|\Delta_{i,a}\|^2$.
Position and normalized signs are retained separately. This is a whole-graph
intervention, not a marginal edge value. The correlation shares the base residual
algebraically and does not identify a causal role for the risk signal.

\begin{table}[t]
\centering\small
\caption{Actual signed normalized vector benefit, harmful-particle fraction,
and previous-risk/benefit Spearman correlation. Means and sample SD use three
equal-trajectory seed means. Negative benefits and correlations are retained.}
\label{tab:actual-action-benefit}
\begin{tabular}{llrrr}
\toprule
Objective & Action & Benefit & Harm fraction & Correlation\\
\midrule
'''+"\n".join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}

The hindsight portfolios choose a single complete prediction for the entire
observed frame; they never mix particles from incompatible graph actions.
The budgeted portfolio includes base and the three 25\% actions; a separate
five-action portfolio also includes dense and therefore has unequal cost.
Exact ties retain all co-best information and use the fixed base-first order
only to record a representative choice. These target-using diagnostics bound
improvement attainable by choosing from the saved menu, not arbitrary graphs.

Particle benefit signs, all nine dense/actual sign combinations, frame and
trajectory harm frequencies, lag agreement, risk quartiles, optional-degree
concentration, pair overlaps, every seed, and both coordinate systems are saved.
Tied risks remain together in rank quartiles; empty groups remain undefined.
One undefined required value makes its corresponding unconditional aggregate
undefined rather than dropping that observation. Neither these post-inspection
diagnostics nor a hindsight portfolio establish a learned abstention policy.
'''
    (output/"full_action_appendix.tex").write_text(appendix)
    manifest={"input_sha256":{"action":EXPECTED_ACTION,"same_state":EXPECTED_SAME},"generated":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-result", type=Path, required=True)
    parser.add_argument("--same-state-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args=parser.parse_args()
    render(read(args.action_result,EXPECTED_ACTION),read(args.same_state_result,EXPECTED_SAME),args.output_dir)
