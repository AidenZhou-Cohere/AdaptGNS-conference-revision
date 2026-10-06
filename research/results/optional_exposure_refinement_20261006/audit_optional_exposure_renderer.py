"""Bounded independent numeric/text review of the pure reporting artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

PAIRS = {"Previous risk -- random": "risk_minus_random", "Previous risk -- speed": "risk_minus_speed", "Speed -- random": "speed_minus_random"}
GROUPS = {"Whole gap": "whole", "Neither": "neither", "Left only": "left_only", "Right only": "right_only", "Both": "both"}
SUBGROUPS = {"Less": "both_less", "Equal": "both_equal", "More": "both_more"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def numeric(value, digits): return f"{0. if round(value, digits) == 0 else value:.{digits}f}"
def pm(value, scale=1000., digits=3):
    if value["mean"] is None: return "---"
    return "$" + numeric(value["mean"] * scale, digits) + r"\pm" + numeric(value["sample_seed_sd"] * scale, digits) + "$"


def run(root, result_path, audit_path):
    result, audit, manifest = read(result_path), read(audit_path), read(root / "render_manifest.json")
    assert audit["passed"] and audit["results_sha256"] == sha(result_path) == manifest["results_sha256"]
    assert manifest["independent_audit_sha256"] == sha(audit_path)
    for name, digest in manifest["outputs"].items(): assert sha(root / name) == digest
    appendix = (root / "optional_exposure_appendix.tex").read_text()
    main = (root / "optional_exposure_main.tex").read_text()
    counts = {"primary_mse_cells": 0, "fraction_cells": 0, "alignment_cost_cells": 0, "both_subgroup_cells": 0, "main_cells": 0}
    for line in appendix.splitlines():
        cols = [value.strip().rstrip("\\").strip() for value in line.split("&")]
        if cols[0] not in PAIRS: continue
        pair = PAIRS[cols[0]]
        if len(cols) == 6:
            group = GROUPS[cols[1]]
            key = f"{pair}/whole/error_difference" if group == "whole" else f"{pair}/{group}/error_contribution"
            for objective, index in (("faithful", 2), ("nll", 4)):
                assert cols[index] == pm(result["objectives"][objective]["metrics"][key])
                counts["primary_mse_cells"] += 1
                if group == "whole": assert cols[index + 1] == "---"
                else:
                    assert cols[index + 1] == pm(result["objectives"][objective]["metrics"][f"{pair}/{group}/particle_fraction"], 100., 2)
                    counts["fraction_cells"] += 1
        elif len(cols) == 5:
            objective = cols[1].lower()
            for column, quantity in zip(cols[2:], ("error", "cost", "alignment")):
                assert column == pm(result["objectives"][objective]["metrics"][f"{pair}/whole/{quantity}_difference"])
                counts["alignment_cost_cells"] += 1
        elif len(cols) == 4:
            group = SUBGROUPS[cols[1]]
            for objective, column in zip(("faithful", "nll"), cols[2:]):
                assert column == pm(result["objectives"][objective]["metrics"][f"{pair}/{group}/error_contribution"])
                counts["both_subgroup_cells"] += 1
        else: raise AssertionError("Unexpected table shape")
    assert counts["primary_mse_cells"] == 30 and counts["fraction_cells"] == 24 and counts["alignment_cost_cells"] == counts["both_subgroup_cells"] == 18
    for objective in ("faithful", "nll"):
        assert pm(result["objectives"][objective]["metrics"]["risk_minus_random/whole/error_difference"]) in main
        counts["main_cells"] += 1
    points = read(root / "plot_coordinates.json")
    assert points["scale"] == 1000. and len(points["points"]) == 90
    expected = {(pair, obj, seed, group) for pair in PAIRS.values() for obj in ("faithful", "nll") for seed in range(3) for group in GROUPS.values()}
    actual = set()
    for row in points["points"]:
        key = row["pair"], row["objective"], row["seed"], row["group"]; assert key not in actual; actual.add(key)
        group_index = list(GROUPS.values()).index(row["group"]); offset = -.14 if row["objective"] == "faithful" else .14
        assert math.isclose(row["x"], group_index + offset + (row["seed"] - 1) * .045, rel_tol=0, abs_tol=1e-15)
        key = row["pair"] + ("/whole/error_difference" if row["group"] == "whole" else "/" + row["group"] + "/error_contribution")
        expected_y = result["runs"][f"{row['objective']}_seed{row['seed']}"]["aggregate"]["metrics"][key] * 1000.
        assert row["y"] == expected_y
    assert actual == expected
    findings = read(root / "mechanism_findings.json")
    assert findings["comparison"] == "risk_minus_random" and len(findings["per_seed"]) == 6
    for row in findings["per_seed"]:
        values = result["runs"][f"{row['objective']}_seed{row['seed']}"]["aggregate"]["metrics"]
        expected_groups = {group: values[f"risk_minus_random/{group}/error_contribution"] for group in GROUPS.values() if group != "whole"}
        assert row["group_error_contributions"] == expected_groups
        assert row["cost_difference"] == values["risk_minus_random/whole/cost_difference"]
        assert row["alignment_difference"] == values["risk_minus_random/whole/alignment_difference"]
        assert row["both_largest_positive"] and row["cost_exceeds_positive_alignment"]
    for token in ("original no-self-loop convention", "not autonomous cached", "Exposure is defined by the policy", "sample SD", "not a pretreatment", "no independent confirmation"):
        assert token in main + appendix
    return {"passed": True, "results_sha256": sha(result_path), "render_manifest_sha256": sha(root / "render_manifest.json"),
            "renderer_sha256": manifest["renderer_sha256"], "cells": counts, "seed_coordinates": 90,
            "mechanism_seed_rows": 6, "units": "All MSE cells/coordinates independently checked at coordinate-MSE x1000; fractions x100",
            "scope": "Numerical/text/hash review; PNG separately visually inspected. No new inference or numerical analysis.",
            "review_source_sha256": sha(__file__)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--render-root", type=Path, required=True); p.add_argument("--result", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); value = run(a.render_root, a.result, a.audit)
    with a.output.open("x") as stream: json.dump(value, stream, indent=2); stream.write("\n")
    print(json.dumps(value))
