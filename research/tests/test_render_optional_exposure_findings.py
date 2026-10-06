"""Synthetic pure-reporting tests; never read real decomposition outcomes."""
import copy
import gzip
import hashlib
import json

import numpy as np
import pytest

from research import render_optional_exposure_findings as renderer
from research import analyze_optional_exposure_decomposition as analysis
from research.tests.test_optional_exposure_decomposition import core_frame


def result_fixture():
    runs={}
    for objective in renderer.OBJECTIVES:
        for seed in range(3):
            row,_=analysis.decompose_predictions(*core_frame(multiplier=seed+1+(objective=="nll")*.25))
            runs[f"{objective}_seed{seed}"]={"aggregate":analysis.aggregate_frames([row],(3,),(7,))}
    return {"scope":"post_inspection_original_100k_optional_exposure_decomposition","required_models":6,
        "required_frames_per_model":297,"required_total_frames":1782,"primary_groups":list(analysis.PRIMARY_GROUPS),
        "supplementary_both_subgroups":list(analysis.BOTH_GROUPS),"comparison_order":{k:list(v) for k,v in analysis.PAIRS.items()},
        "runs":runs,"objectives":analysis.summarize_objectives(runs)}


def test_all_primary_groups_all_pair_directions_and_table_values():
    result=result_fixture();renderer.validate_result(result)
    main,appendix=renderer.render_tex(result)
    assert "Positive values favor random" in main
    assert "ten message-passing blocks" in main
    assert "not a pretreatment" in appendix
    assert "not necessarily unaffected" in appendix
    assert "ratios of the" in appendix and "available-seed" in appendix
    assert " & Nll & " not in appendix and appendix.count(" & NLL & ") == 3
    rows=[r.split(" & ") for r in appendix.splitlines() if r.startswith(tuple(label+" & " for _,label,_ in renderer.PAIRS))]
    assert len(rows)==30 # 15 primary +6 alignment/cost +9 both subgroup rows.
    for pair_index,(pair,_,_) in enumerate(renderer.PAIRS):
        for group_index,(group,_) in enumerate(renderer.GROUPS):
            row=rows[pair_index*5+group_index]
            for objective_index,obj in enumerate(renderer.OBJECTIVES):
                value=renderer.measure(result,obj,pair,group)
                expected=f"${value['mean']*1000:.3f}\\pm{value['sample_seed_sd']*1000:.3f}$"
                assert row[2+2*objective_index].rstrip("\\")==expected


def test_plot_has_six_seed_dots_per_group_and_exact_times1000_coordinates(tmp_path):
    result=result_fixture(); output=tmp_path/"render"
    manifest=renderer.render(result,{"results_sha256":"synthetic","independent_audit_sha256":"synthetic"},output)
    coordinates=json.loads((output/"plot_coordinates.json").read_text())
    points=coordinates["points"]
    assert coordinates["scale"]==1000 and len(points)==90
    for panel,(pair,_,_) in enumerate(renderer.PAIRS):
        for group_index,(group,_) in enumerate(renderer.GROUPS):
            subset=[p for p in points if p["pair"]==pair and p["group"]==group]
            assert len(subset)==6
            for p in subset:
                expected=renderer.measure(result,p["objective"],pair,group)["seed_values"][p["seed"]]*1000
                assert p["y"]==expected and p["panel"]==panel
                expected_x=group_index+(-.14 if p["objective"]=="faithful" else .14)+(p["seed"]-1)*.045
                assert p["x"]==expected_x
    assert all(renderer.sha(output/name)==digest for name,digest in manifest["outputs"].items())


@pytest.mark.parametrize("field,value",[("mean",999.),("sample_seed_sd",999.),("defined_seeds",2)])
def test_wrong_mean_sd_or_seed_count_refused(field,value):
    result=result_fixture()
    result["objectives"]["faithful"]["metrics"]["risk_minus_random/whole/error_difference"][field]=value
    with pytest.raises(ValueError):renderer.validate_result(result)


def test_missing_seed_is_never_rendered_as_available_seed_mean(tmp_path):
    result=result_fixture()
    result["runs"]["faithful_seed2"]["aggregate"]=analysis.aggregate_frames([],(3,),(7,))
    result["objectives"]=analysis.summarize_objectives(result["runs"])
    renderer.validate_result(result)
    value=renderer.measure(result,"faithful","risk_minus_random","whole")
    assert value["mean"] is None and renderer.pm(value)=="---"
    output=tmp_path/"missing";renderer.render(result,{"results_sha256":"synthetic"},output)
    points=json.loads((output/"plot_coordinates.json").read_text())["points"]
    assert len(points)==75 and all(not (p["objective"]=="faithful" and p["seed"]==2) for p in points)
    assert "---" in (output/"optional_exposure_appendix.tex").read_text()


def test_audit_gate_requires_exact_result_bytes_population_and_pass(tmp_path):
    result=result_fixture();path=tmp_path/"results.json";path.write_text(json.dumps(result))
    audit_path=tmp_path/"audit.json"
    valid={"passed":True,"results_sha256":renderer.sha(path),"audited_frames":1782}
    audit_path.write_text(json.dumps(valid))
    read,identity=renderer.read_verified_result(path,audit_path)
    assert read==result and identity["results_sha256"]==renderer.sha(path)
    zipped=tmp_path/"results.json.gz";zipped.write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    read,identity=renderer.read_verified_result(zipped,audit_path)
    assert read==result and identity["results_file_sha256"]==renderer.sha(zipped)
    for change in ({"passed":False},{"results_sha256":"0"*64},{"audited_frames":1781}):
        audit_path.write_text(json.dumps(valid|change))
        with pytest.raises(ValueError,match="Independent audit must"):
            renderer.read_verified_result(path,audit_path)


def test_wrong_comparison_direction_and_missing_group_rejected():
    result=result_fixture();result["comparison_order"]["risk_minus_random"].reverse()
    with pytest.raises(ValueError,match="direction"):
        renderer.validate_result(result)
    result=result_fixture();result["primary_groups"].pop()
    with pytest.raises(ValueError,match="partition"):
        renderer.validate_result(result)


def test_existing_render_is_preserved(tmp_path):
    output=tmp_path/"old";output.mkdir();(output/"evidence").write_text("keep")
    with pytest.raises(ValueError,match="Preserve existing"):
        renderer.render(result_fixture(),{},output)
    assert (output/"evidence").read_text()=="keep"


def test_tiny_negative_rounding_does_not_display_negative_zero():
    assert renderer.pm({"mean":-1e-8,"sample_seed_sd":1e-8})==r"$0.000\pm0.000$"


def test_substantive_findings_require_every_seed_and_exact_comparison():
    result=result_fixture()
    for objective in renderer.OBJECTIVES:
        for seed in range(3):
            m=result["runs"][f"{objective}_seed{seed}"]["aggregate"]["metrics"]
            for group,value in (("neither",.1),("left_only",-.1 if objective=="faithful" else .1),("right_only",.2),("both",.5)):
                m[f"risk_minus_random/{group}/error_contribution"]=value
            m["risk_minus_random/whole/cost_difference"]=2.
            m["risk_minus_random/whole/alignment_difference"]=1.
    findings=renderer.mechanism_findings(result)
    assert findings["both_largest_positive_all_six"] and findings["cost_exceeds_positive_alignment_all_six"]
    assert findings["faithful_left_only_negative_nll_positive_all_seeds"]
    m=result["runs"]["nll_seed2"]["aggregate"]["metrics"]
    m["risk_minus_random/both/error_contribution"]=.05
    m["risk_minus_random/whole/cost_difference"]=-1.
    findings=renderer.mechanism_findings(result)
    assert not findings["both_largest_positive_all_six"] and not findings["cost_exceeds_positive_alignment_all_six"]
    m["risk_minus_random/left_only/error_contribution"]=None
    assert not renderer.mechanism_findings(result)["faithful_left_only_negative_nll_positive_all_seeds"]
