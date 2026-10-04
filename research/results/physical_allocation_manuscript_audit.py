"""Audit the physical-allocation appendix and full-source structural references.

Reads only the manuscript, completed scalar summary and analysis protocol.
No model, source dataset, raw prediction artifact or implementation import.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANUSCRIPT = ROOT / "outputs/revised_manuscript.tex"
SUMMARY = HERE / "attempt_20261004/summary.json"
PROTOCOL = ROOT / "outputs/AdaptGNS/research/protocols/physical_allocation_pilot.md"
counts, findings, checks = Counter(), [], []


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def check(kind, label, actual, expected):
    counts[kind] += 1
    ok = actual == expected
    checks.append({"kind": kind, "claim": label, "passed": ok})
    if not ok:
        findings.append({"kind": kind, "claim": label, "actual": actual, "expected": expected})


def without_comments(text):
    output = []
    for line in text.splitlines():
        keep = len(line)
        for i, char in enumerate(line):
            if char != "%":
                continue
            backslashes, j = 0, i-1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                keep = i
                break
        output.append(line[:keep])
    return "\n".join(output)


def check_stat_cell(cell, stat, digits, label):
    match = re.fullmatch(r"\$([-+\d.]+)\\pm([-+\d.]+)\$", cell.strip())
    if match is None:
        raise ValueError("Could not parse numeric table cell: "+cell)
    check("numeric_table_cell", label+" mean", match[1], f"{stat['mean']:.{digits}f}")
    check("numeric_table_cell", label+" sample SD", match[2], f"{stat['sample_seed_sd']:.{digits}f}")


def tabular_rows(block, label):
    table = re.search(r"\\label\{"+re.escape(label)+r"\}(.*?)\\end\{table\}", block, re.S).group(1)
    return [line.strip().removesuffix(r"\\").strip().split("&") for line in table.splitlines() if r"\pm" in line]


raw = MANUSCRIPT.read_bytes()
summary_raw, protocol_raw = SUMMARY.read_bytes(), PROTOCOL.read_bytes()
text, data, protocol = raw.decode(), json.loads(summary_raw), protocol_raw.decode()
block = text.split(r"\section{Exploratory cheap physical allocation controls}", 1)[1].split(r"\end{document}", 1)[0]
plain = " ".join(block.split())
result = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
          "manuscript_sha256": sha_bytes(raw), "summary_sha256": sha_bytes(summary_raw),
          "protocol_sha256": sha_bytes(protocol_raw), "audit_source_sha256": sha_bytes(Path(__file__).read_bytes()),
          "scope": "All physical-allocation appendix numeric tables/claims and definitions; full-source label/citation and environment structure only",
          "limitations": ["Native compilation and visual page layout are not performed by this audit.",
                          "The pre-inference synthetic-test count needs a test-log/verification record, absent from the three permitted inputs.",
                          "Other sections' scientific or numerical claims are not re-audited here."]}
try:
    objectives = {"Faithful": "faithful", "NLL": "nll", r"$\beta$-NLL": "beta_nll"}
    policies = {"Inverse count": "inverse_count25", "Velocity RMS": "velocity_rms25"}
    mse_rows = tabular_rows(block, "tab:physical-allocation-mse")
    check("table_structure", "MSE row count", len(mse_rows), 6)
    for row in mse_rows:
        obj, split = objectives[row[0].strip()], row[1].strip()
        group = data["summaries"][obj+"/"+split]
        for cell, policy in zip(row[2:], ("inverse_count25", "velocity_rms25")):
            check_stat_cell(cell, group["policies"][policy], 4, f"{obj}/{split}/{policy}")
    paired_rows = tabular_rows(block, "tab:physical-allocation-paired")
    check("table_structure", "Paired row count", len(paired_rows), 6)
    for row in paired_rows:
        obj, policy = objectives[row[0].strip()], policies[row[1].strip()]
        group = data["summaries"][obj+"/test"]["physical_minus_control"][policy]
        for cell, control in zip(row[2:], ("random25", "current_risk25", "lagged_base_risk25")):
            check_stat_cell(cell, group[control]["percent_difference"], 3, f"{obj}/test/{policy}/{control} paired %")

    runs = data["runs"]
    frame_count = sum(len(run["splits"][split]["frames"]) for run in runs.values() for split in ("valid", "test"))
    replays = [entry for run in runs.values() for split in ("valid", "test") for row in run["splits"][split]["frames"] for entry in row["control_replay"].values()]
    check("numeric_prose", "15 compact checkpoints", len(runs), 15)
    check("numeric_prose", "36 validation/test frames per model", sorted({len(r["splits"][s]["frames"]) for r in runs.values() for s in ("valid", "test")}), [36])
    check("numeric_prose", "1,080 complete frame/model evaluations", frame_count, 1080)
    check("numeric_prose", "6,480 replay checks", len(replays), 6480)
    check("numeric_prose", "complete attempt", data["state"], "complete")
    check("numeric_prose", "all models complete", all(r["state"] == "complete" for r in runs.values()), True)
    check("numeric_prose", "all original-control edge counts identical", all(r["actual_directed_edges"] == r["original_directed_edges"] for r in replays), True)
    check("numeric_prose", "all MSE checks pass predeclared tolerance", all(abs(r["actual_mse"]-r["original_mse"]) <= 1e-7+2e-5*abs(r["original_mse"]) for r in replays), True)
    max_replay = max(abs(r["mse_difference"]) for r in replays)
    token = re.search(r"absolute MSE difference \$([^$]+)\$", plain)[1]
    mantissa, exponent = f"{max_replay:.2e}".split("e")
    check("numeric_prose", "maximum original-replay difference", token, mantissa+r"\times10^{"+str(int(exponent))+"}")
    timing = re.search(r"attempt took ([\d.]+) seconds", plain)[1]
    check("numeric_prose", "CPU wall time", timing, f"{data['elapsed_seconds']:.3f}")
    check("numeric_prose", "one Torch thread", data["software"]["torch_threads"], 1)
    check("numeric_prose", "CPU device", data["software"]["device"], "cpu")
    check("numeric_prose", "BLAS thread limit", set(data["thread_environment"].values()) == {"1"}, True)

    nll = data["summaries"]["nll/test"]
    for control in ("random25", "speed25", "current_risk25", "lagged_base_risk25"):
        deltas = nll["physical_minus_control"]["velocity_rms25"][control]["difference"]["seed_values"]
        check("directional_prose", "NLL velocity RMS beats "+control+" in all five seeds", len(deltas) == 5 and all(v < 0 for v in deltas), True)
    check("directional_prose", "dense NLL remains more accurate", nll["policies"]["dense"]["mean"] < nll["policies"]["velocity_rms25"]["mean"], True)
    faithful = data["summaries"]["faithful/test"]["policies"]
    for control in ("current_risk25", "lagged_base_risk25"):
        for policy in ("inverse_count25", "velocity_rms25"):
            check("directional_prose", "faithful means favor "+control+" over "+policy, faithful[control]["mean"] < faithful[policy]["mean"], True)
    beta_values = {split: data["summaries"]["beta_nll/"+split]["physical_minus_control"]["velocity_rms25"]["random25"]["percent_difference"]["mean"] for split in ("valid", "test")}
    check("numeric_prose", "beta test paired percentage magnitude token", "$0.432\\%$" in block or "$-0.432\\%$" in block or r"-0.432\%" in block, True)
    check("numeric_prose", "beta test paired percentage magnitude", f"{abs(beta_values['test']):.3f}", "0.432")
    check("numeric_prose", "beta validation paired percentage magnitude token", "$0.883\\%$" in block or "$+0.883\\%$" in block or r"+0.883\%" in block, True)
    check("numeric_prose", "beta validation paired percentage magnitude", f"{beta_values['valid']:.3f}", "0.883")
    check("directional_prose", "beta split disagreement", beta_values["test"] < 0 < beta_values["valid"], True)
    original_wording = "velocity RMS improves the inspected test mean relative to random by"
    check("estimand_wording", "beta percentage prose distinguishes within-seed percentages from group means", original_wording not in plain, True)
    inverse_means = [group["physical_minus_control"]["inverse_count25"][c]["difference"]["mean"] for group in data["summaries"].values() for c in ("random25", "speed25", "current_risk25", "lagged_base_risk25")]
    check("directional_prose", "inverse count mixed results", min(inverse_means) < 0 < max(inverse_means), True)

    canonical = runs["faithful_seed0"]
    expected_ties = {"valid": {"inverse_count25": 35, "velocity_rms25": 27}, "test": {"inverse_count25": 36, "velocity_rms25": 21}}
    for split in ("valid", "test"):
        for policy in ("inverse_count25", "velocity_rms25"):
            for name, run in runs.items():
                value = sum(r["cutoff_ties"][policy]["split_cutoff_tie"] for r in run["splits"][split]["frames"])
                check("numeric_tie_counts", name+" "+split+" "+policy, value, expected_ties[split][policy])
        isolated = sum(r["isolated_particles"] for r in canonical["splits"][split]["frames"])
        check("numeric_prose", split+" isolated particle-frame pairs", isolated, 37 if split == "valid" else 53)
    check("numeric_prose", "tie-frequency prose tokens", all(t in plain for t in ("35/36", "36/36", "27/36", "21/36")), True)
    check("numeric_prose", "isolation prose tokens", "37 validation and 53 test isolated" in plain, True)

    definition_tokens = [r"$-d_i$", r"$u_i=x_i^{t-1}-x_i^{t-2}$", "converted to float64 before subtraction",
        r"d_i^{-1}\sum_{j\in\mathcal N_i}\|u_j-u_i\|^2", r"s_i=0\quad(d_i=0)",
        "no strain-fit eligibility mask is used", "maximum endpoint score", r"\lfloor .25|C_R(x)|\rfloor",
        "mandatory graph and ID tie rule", "float32 radius comparison", "strict-float64", "Neither score is fitted to prediction errors"]
    for token in definition_tokens:
        check("definition_and_scope", token, token in plain, True)
    for token in ("After inspecting those correlations", "separate exploratory follow-up", "not independent confirmation",
                  "sample seed SD", "not confidence intervals or percentages of group means", "previous observed base graph",
                  "not an autonomous cached-own-graph state", "not a general superiority claim", "not measure the prediction effect of relabeling",
                  "neither autonomous-rollout stability nor full-architecture"):
        check("definition_and_scope", token, token in plain, True)
    check("definition_and_scope", "C_R is optional annulus", r"C_R(x)&=E_R(x)\setminus E_r(x)" in text, True)
    check("definition_and_scope", "protocol matches exact quarter budget", "K=floor(0.25 × number of optional pairs)" in protocol, True)
    check("definition_and_scope", "protocol replay tolerances", "rtol=2e-5, atol=1e-7" in protocol, True)

    clean = without_comments(text)
    labels = re.findall(r"\\label\{([^}]+)\}", clean)
    refs = re.findall(r"\\(?:ref|eqref|pageref|autoref)\*?\{([^}]+)\}", clean)
    cites = [key.strip() for group in re.findall(r"\\cite(?:p|t|alp|alt|author|year|yearpar)?\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", clean) for key in group.split(",")]
    bibliography = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", clean)
    check("latex_structure", "duplicate labels", sorted(k for k, n in Counter(labels).items() if n != 1), [])
    check("latex_structure", "undefined references", sorted(set(refs)-set(labels)), [])
    check("latex_structure", "duplicate bibliography keys", sorted(k for k, n in Counter(bibliography).items() if n != 1), [])
    check("latex_structure", "undefined citation keys", sorted(set(cites)-set(bibliography)), [])
    # The embedded conference class contains begin-document tokens inside
    # warning-message and macro definitions. Only the standalone command is
    # the actual document boundary; scan its body for environment nesting.
    document_begin = list(re.finditer(r"(?m)^\s*\\begin\{document\}\s*$", clean))
    document_end = list(re.finditer(r"(?m)^\s*\\end\{document\}\s*$", clean))
    check("latex_structure", "one standalone document begin", len(document_begin), 1)
    check("latex_structure", "one standalone document end", len(document_end), 1)
    body = clean[document_begin[0].end():document_end[-1].start()]
    stack, errors = [], []
    for token in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
        kind, environment = token[1], token[2]
        if kind == "begin":
            stack.append(environment)
        elif not stack or stack[-1] != environment:
            errors.append({"closing": environment, "stack": list(stack)})
        else:
            stack.pop()
    check("latex_structure", "document-body environment nesting", errors, [])
    check("latex_structure", "unclosed document-body environments", stack, [])
    result["structural_inventory"] = {"labels": len(labels), "reference_occurrences": len(refs), "unique_cited_keys": len(set(cites)), "bibliography_keys": len(bibliography), "new_labels": [k for k in labels if "physical-allocation" in k]}
    result["separately_attested_claims"] = [{"claim": "54 synthetic checks passed before inference", "evidence": "Root reports the immediately pre-inference tool invocation returned 54 passed in 1.32 seconds; no independent saved log artifact was accessed.", "independently_audited": False}]
    result["state"] = "passed_with_attested_test_count" if not findings else "findings"
except BaseException as error:
    result["state"] = "audit_failed"
    result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
finally:
    result.update({"counts": dict(counts), "total_checks": sum(counts.values()), "finding_count": len(findings), "findings": findings, "checks": checks})
    (HERE / "manuscript_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps({key: result.get(key) for key in ("state", "total_checks", "finding_count", "findings", "structural_inventory", "failure")}, indent=2))
