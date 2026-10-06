"""Render independently audited optional-exposure results; no new analysis."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OBJECTIVES = ("faithful", "nll")
PAIRS = (("risk_minus_random", "Previous risk -- random", "Risk − random (primary)"),
         ("risk_minus_speed", "Previous risk -- speed", "Risk − speed"),
         ("speed_minus_random", "Speed -- random", "Speed − random"))
PRIMARY = ("neither", "left_only", "right_only", "both")
SECONDARY = ("both_less", "both_equal", "both_more")
GROUPS = (("whole", "Whole gap"), ("neither", "Neither"), ("left_only", "Left only"),
          ("right_only", "Right only"), ("both", "Both"))
SCALE = 1000.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(result, objective, pair, group, quantity="error"):
    suffix = f"whole/{quantity}_difference" if group == "whole" else f"{group}/{quantity}_contribution"
    return result["objectives"][objective]["metrics"][f"{pair}/{suffix}"]


def validate_stat(record):
    values = record["seed_values"]
    require(record["required_seeds"] == 3 and len(values) == 3 and all(v is None or type(v) in (int,float) and math.isfinite(v) for v in values), "Exactly three finite-or-null seed values required")
    require(record["defined_seeds"] == sum(v is not None for v in values), "Defined seed count differs")
    if any(v is None for v in values):
        require(record["mean"] is None and record["sample_seed_sd"] is None, "Missing seed must leave mean and SD undefined")
    else:
        require(np.isclose(record["mean"], np.mean(values), rtol=1e-12, atol=1e-15), "Mean differs from three seeds")
        require(np.isclose(record["sample_seed_sd"], np.std(values,ddof=1), rtol=1e-12, atol=1e-15), "Sample SD differs from three seeds")


def validate_result(result):
    require(result.get("scope") == "post_inspection_original_100k_optional_exposure_decomposition", "Wrong scientific evidence family")
    require(result.get("required_models") == 6 and result.get("required_frames_per_model") == 297 and result.get("required_total_frames") == 1782, "Fixed six-model population required")
    require(result.get("primary_groups") == list(PRIMARY) and result.get("supplementary_both_subgroups") == list(SECONDARY), "Declared partition differs")
    require(set(result["comparison_order"]) == {p[0] for p in PAIRS}, "All three comparison directions required")
    expected = {"risk_minus_random": ["previous-observed-base-risk25","random25"],
        "risk_minus_speed": ["previous-observed-base-risk25","speed25"], "speed_minus_random": ["speed25","random25"]}
    require(result["comparison_order"] == expected, "Comparison direction differs")
    require(set(result["runs"]) == {f"{o}_seed{s}" for o in OBJECTIVES for s in range(3)}, "Missing objective/seed")
    require(set(result["objectives"]) == set(OBJECTIVES), "Both objectives required")
    for obj in OBJECTIVES:
        for key, record in result["objectives"][obj]["metrics"].items():
            validate_stat(record)
            require(record["seed_values"] == [result["runs"][f"{obj}_seed{s}"]["aggregate"]["metrics"][key] for s in range(3)], "Seed values differ from fixed model records")
        for pair,_,_ in PAIRS:
            for seed in range(3):
                row = result["runs"][f"{obj}_seed{seed}"]["aggregate"]["metrics"]
                gap = row[f"{pair}/whole/error_difference"]
                if gap is None:
                    continue
                bound = 256*np.finfo(np.float64).eps*297*max(row[f"{pair}/whole/arithmetic_scale"],1e-300)
                require(abs(sum(row[f"{pair}/{g}/error_contribution"] for g in PRIMARY)-gap) <= bound, "Four group contributions do not sum to whole gap")
                require(abs(sum(row[f"{pair}/{g}/error_contribution"] for g in SECONDARY)-row[f"{pair}/both/error_contribution"]) <= bound, "Both subgroups do not sum to both")
                require(abs(gap - (row[f"{pair}/whole/cost_difference"]-row[f"{pair}/whole/alignment_difference"])) <= bound, "Coordinate cost/alignment sign differs")


def read_verified_result(results_path, audit_path):
    data = results_path.read_bytes()
    file_hash, audit_hash = hashlib.sha256(data).hexdigest(), sha(audit_path)
    if results_path.suffix == ".gz":
        data = gzip.decompress(data)
    results_hash = hashlib.sha256(data).hexdigest()
    audit = json.loads(audit_path.read_text())
    require(audit.get("passed") is True and audit.get("results_sha256") == results_hash
            and audit.get("audited_frames") == 1782, "Independent audit must pass, cover all 1782 frames and match these exact result bytes")
    result = json.loads(data)
    validate_result(result)
    return result, {"results_sha256": results_hash, "results_file_sha256": file_hash, "independent_audit_sha256": audit_hash}


def number(value, digits):
    value = 0. if round(value,digits) == 0 else value
    return f"{value:.{digits}f}"


def pm(record, scale=SCALE, digits=3):
    if record["mean"] is None:
        return "---"
    return "$" + number(record["mean"]*scale,digits) + r"\pm" + number(record["sample_seed_sd"]*scale,digits) + "$"


def mechanism_findings(result):
    rows=[]
    for objective in OBJECTIVES:
        for seed in range(3):
            m=result["runs"][f"{objective}_seed{seed}"]["aggregate"]["metrics"]
            groups={g:m[f"risk_minus_random/{g}/error_contribution"] for g in PRIMARY}
            cost=m["risk_minus_random/whole/cost_difference"]
            alignment=m["risk_minus_random/whole/alignment_difference"]
            defined=all(v is not None for v in (*groups.values(),cost,alignment))
            rows.append({"objective":objective,"seed":seed,"group_error_contributions":groups,
                "cost_difference":cost,"alignment_difference":alignment,
                "both_largest_positive":defined and groups["both"]>0 and all(groups["both"]>groups[g] for g in PRIMARY if g!="both"),
                "cost_exceeds_positive_alignment":defined and cost>alignment>0})
    return {"comparison":"risk_minus_random","per_seed":rows,
        "both_largest_positive_all_six":all(r["both_largest_positive"] for r in rows),
        "cost_exceeds_positive_alignment_all_six":all(r["cost_exceeds_positive_alignment"] for r in rows),
        "faithful_left_only_negative_nll_positive_all_seeds":all(r["group_error_contributions"]["left_only"] is not None and
            (r["group_error_contributions"]["left_only"]<0 if r["objective"]=="faithful" else r["group_error_contributions"]["left_only"]>0) for r in rows)}


def render_figure(result, output):
    # At the report's 6.9-inch width, 11-point labels remain 8.43 points.
    fig,axes=plt.subplots(1,3,figsize=(9.,3.8),sharey=True,layout="constrained")
    plotted=[]
    for panel,(pair,_,title) in enumerate(PAIRS):
        ax=axes[panel]
        for objective,color,marker,offset in zip(OBJECTIVES,("#245B8A","#AC4E26"),("o","^"),(-.14,.14)):
            for group_index,(group,_) in enumerate(GROUPS):
                statistic=measure(result,objective,pair,group)
                values=statistic["seed_values"]
                for seed,value in enumerate(values):
                    if value is not None:
                        x=group_index+offset+(seed-1)*.045
                        ax.scatter([x],[value*SCALE],s=20,marker=marker,color=color,alpha=.75,zorder=3,
                            label=objective.upper()+" seeds" if panel==0 and group_index==0 and seed==0 else None)
                        plotted.append({"panel":panel,"pair":pair,"objective":objective,"seed":seed,"group":group,"x":x,"y":value*SCALE})
                if statistic["mean"] is not None:
                    ax.errorbar(group_index+offset,statistic["mean"]*SCALE,yerr=statistic["sample_seed_sd"]*SCALE,
                        fmt="_",color=color,linewidth=1.2,markersize=10,capsize=3,zorder=4)
        ax.axhline(0,color="#777777",lw=.8,zorder=0)
        ax.set_xticks(range(len(GROUPS)),["Total\ngap" if group=="whole" else label.replace(" ","\n") for group,label in GROUPS])
        ax.set_title(title,fontsize=11)
        ax.tick_params(labelsize=11)
        ax.grid(axis="y",color="#dddddd",lw=.6)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Normalized coordinate MSE × 1,000\nPositive: left policy has higher error",fontsize=11)
    axes[0].legend(frameon=False,fontsize=11)
    fig.suptitle("Optional-exposure decomposition of observed-history policy gaps",fontsize=12)
    fig.supxlabel("Seed dots; bars show mean ± sample SD. Equal trajectory weights.",fontsize=11)
    fig.savefig(output/"optional_exposure_decomposition.png",dpi=180)
    fig.savefig(output/"optional_exposure_decomposition.pdf")
    plt.close(fig)
    return plotted


def render_tex(result):
    gap=[pm(measure(result,obj,"risk_minus_random","whole")) for obj in OBJECTIVES]
    findings=mechanism_findings(result)
    mechanism=""
    if findings["both_largest_positive_all_six"]:
        mechanism += "The both-covered group gives the largest positive unconditional contribution\nin every seed; its particle fraction is reported alongside the contribution.\n"
    if findings["cost_exceeds_positive_alignment_all_six"]:
        mechanism += r"""Relative to random, risk increases both alignment with the base residual
and the squared normalized prediction change. Here $\Delta E$, $\Delta C$
and $\Delta A$ denote risk-minus-random differences in normalized coordinate
error, squared prediction change, and residual alignment, respectively. The identity
$\Delta E=\Delta C-\Delta A$ makes $\Delta C>\Delta A$ equivalent to the
positive gap; the additional observation is $\Delta A>0$ in every seed.
Here $C$ is squared prediction change, not computational cost.
"""
    main=r'''\subsection{Where does the sparse-policy error gap appear?}
Using the original no-self-loop saved predictions, we partition the recorded
risk-minus-random error gap by whether particles
receive optional messages under neither, only the left, only the right, or
both actions. The whole gap is '''+gap[0]+r''' / '''+gap[1]+r'''
in normalized coordinate MSE $\times10^3$ (faithful/NLL; three seed means
and sample SDs). Positive values favor random.
'''+mechanism+r'''The four unconditional contributions sum to this gap; Appendix~\ref{sec:optional-exposure} reports
all groups and the two speed controls. Exposure is defined by the policy,
so the partition is descriptive. With ten message-passing blocks, particles
in the neither group can still be affected through their neighbors.
'''
    appendix=r'''\section{Exploratory optional-exposure decomposition}
\label{sec:optional-exposure}
This post-inspection analysis reuses all 1,782 original observed-history frames
from the six unchanged 100k models, under the original no-self-loop convention.
Previous risk means previous-observed-base risk; it is not autonomous cached
own-graph risk. There is no new inference or graph permutation. The primary
comparison is previous risk minus random; previous risk minus speed and speed
minus random are fixed secondary controls. Positive error differences mean
the left policy has higher error.

Optional incident degree counts only selected unordered annulus pairs; all
three sparse policies retain the same exact optional budget and mandatory
base graph. The four groups partition particles by positive optional degree
under each action. In this appendix, $y_i$, $b_i$ and $p_{ia}$ denote target,
base-predicted and action-predicted positions. Let $s=(s_1,s_2)$ be the saved
acceleration standard deviations, with $\oslash$ denoting elementwise
division. Define scalar normalized coordinate squared error
$\ell_{ia}=\tfrac12\|(p_{ia}-y_i)\oslash s\|^2$.
A group's unconditional contribution is
$N^{-1}\sum_i I_{ig}(\ell_{iL}-\ell_{iR})$.
We average frames within trajectory, then equally average all 27 trajectories
within seed. All three seeds are required for means and sample SDs.

\begin{table}[ht]
\centering\small
\caption{Unconditional normalized coordinate MSE contributions $\times10^3$
and fixed-weight particle fractions (\%). Whole-gap rows give the total paired
error. The four group means sum to the whole gap before rounding; SDs are not
additive. Every entry is a three-seed mean $\pm$ sample SD.}
\label{tab:optional-exposure}
\begin{tabular}{llrrrr}
\toprule
Pair & Group & Faithful MSE & Fraction & NLL MSE & Fraction\\
\midrule
'''
    for pair,label,_ in PAIRS:
        for group,title in GROUPS:
            cells=[]
            for obj in OBJECTIVES:
                cells.append(pm(measure(result,obj,pair,group)))
                cells.append("---" if group=="whole" else pm(result["objectives"][obj]["metrics"][f"{pair}/{group}/particle_fraction"],100,2))
            appendix += label+" & "+title+" & "+" & ".join(cells)+r"\\"+"\n"
        appendix += r"\midrule"+"\n"
    appendix += r'''\end{tabular}
\end{table}

'''
    if findings["faithful_left_only_negative_nll_positive_all_seeds"]:
        appendix += "In the primary risk-minus-random comparison, the risk-only contribution\nis negative in every faithful seed and positive in every NLL seed.\nBoth the signs and the complete speed controls are retained above.\n\n"
    appendix += r'''
For $r_i=(y_i-b_i)\oslash s$ and $\delta_{ia}=(p_{ia}-b_i)\oslash s$,
the alignment and squared-change terms are
$A_{ia}=2r_i^\top\delta_{ia}/2$ and
$C_{ia}=\|\delta_{ia}\|^2/2$.
Coordinate error differences satisfy
$\ell_{iL}-\ell_{iR}=(C_{iL}-C_{iR})-(A_{iL}-A_{iR})$.
The alignment difference is subtracted: positive alignment difference favors
the left action. Every term divides the corresponding vector quantity by two.
The same identity is checked within every group and at each aggregation level.

\begin{table}[ht]
\centering\small
\caption{Whole-frame alignment/cost decomposition, normalized coordinate MSE
$\times10^3$. Error gap equals cost difference minus alignment difference;
all directions are left minus right.}
\label{tab:optional-exposure-alignment}
\begin{tabular}{llrrr}
\toprule
Pair & Objective & Error gap & Cost difference & Alignment difference\\
\midrule
'''
    for pair,label,_ in PAIRS:
        for obj in OBJECTIVES:
            appendix += label+" & "+("NLL" if obj=="nll" else "Faithful")+" & "+" & ".join(pm(measure(result,obj,pair,"whole",q)) for q in ("error","cost","alignment"))+r"\\"+"\n"
    appendix += r'''\bottomrule
\end{tabular}
\end{table}

\begin{table}[ht]
\centering\small
\caption{Supplementary decomposition of the both-covered group by whether
left optional degree is less than, equal to, or greater than right degree.
Entries are unconditional coordinate MSE contributions $\times10^3$;
these three terms sum to the both-covered contribution, not the whole gap.}
\label{tab:optional-exposure-both}
\begin{tabular}{llrr}
\toprule
Pair & Left degree & Faithful & NLL\\
\midrule
'''
    for pair,label,_ in PAIRS:
        for group,title in zip(SECONDARY,("Less","Equal","More")):
            appendix += label+" & "+title+" & "+" & ".join(pm(measure(result,obj,pair,group)) for obj in OBJECTIVES)+r"\\"+"\n"
    appendix += r'''\bottomrule
\end{tabular}
\end{table}

Empty groups contribute zero but have undefined conditional means. Missing
required scientific inputs make corresponding full-population means undefined
(shown as ---); no available-frame or available-seed mean is substituted.
The archived weighted conditional summaries are explicitly ratios of the
fixed-weight contribution to the fixed-weight group fraction, defined only
when all required inputs exist and that fraction is positive. They are
different from averages of nonempty-frame conditional means.

Optional exposure is an outcome of graph selection, not a pretreatment
confounder. This decomposition does not identify marginal edge value or a
causal concentration effect. Ten message-passing blocks transmit effects
beyond directly incident optional edges, so neither-covered particles are
not necessarily unaffected. Test histories were already inspected; these
results provide no independent confirmation, population significance,
autonomous-stability guarantee or policy-runtime advantage. All frame-level
values, failed inputs, source hashes and signed contributions are retained.
'''
    return main,appendix


def render(result, identity, output):
    validate_result(result)
    require(not output.exists(), "Preserve existing rendering directory")
    output.mkdir(parents=True,exist_ok=False)
    plotted=render_figure(result,output)
    main,appendix=render_tex(result)
    (output/"optional_exposure_main.tex").write_text(main)
    (output/"optional_exposure_appendix.tex").write_text(appendix)
    (output/"mechanism_findings.json").write_text(json.dumps(mechanism_findings(result),indent=2)+"\n")
    (output/"plot_coordinates.json").write_text(json.dumps({"scale":SCALE,"positive_direction":"left policy has higher normalized coordinate MSE",
        "points":plotted,"means":"three seeds required; missing means are never replaced by available-seed averages",
        "error_bars":"sample SD across exactly three seed means",
        "figure_inches":[9.,3.8],"minimum_label_point_size":11,
        "minimum_label_points_at_report_width_6_9_inches":11*6.9/9},indent=2)+"\n")
    manifest={**identity,"renderer_sha256":sha(Path(__file__)),"scope":"pure reporting of independently audited original observed-history optional-exposure decomposition",
        "outputs":{p.name:sha(p) for p in sorted(output.iterdir())}}
    (output/"render_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result,identity=read_verified_result(args.results,args.audit)
    manifest=render(result,identity,args.output_dir)
    require(sha(args.results)==identity["results_file_sha256"] and sha(args.audit)==identity["independent_audit_sha256"], "Inputs changed during rendering")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
