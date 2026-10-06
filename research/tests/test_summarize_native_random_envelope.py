"""Synthetic saved-array/hierarchy rejection tests; no real data or runs."""
import copy
import numpy as np
import pytest
from research import native_random_envelope as evaluate
from research import summarize_native_random_envelope as summary
from research.tests.test_native_random_envelope import observed
from research.tests.test_graph_convention_bridge import fixture, identity, tiny_model


def audited():
    row,a=observed();row["seed"]=0
    audit=summary.Audit()
    out=summary.audit_frame(row,a,audit)
    return row,a,out,audit


def test_complete_saved_array_reconstruction():
    row,a,out,audit=audited()
    assert audit.checks>200
    assert out["values"]==summary.values_from_cases(row["cases"])
    assert out["envelope"]==row["random_envelope"]


@pytest.mark.parametrize("mutation",["mse","mask","prefix","target","risk_source","draw","parity","boundary","gain","envelope","failed_metric","missing_case"])
def test_tampered_raw_or_derived_values_rejected(mutation):
    row,a=observed();row["seed"]=0
    if mutation=="mse":row["cases"]["base"]["metrics"][evaluate.METRICS[0]]+=1.
    elif mutation=="mask":a["optional_pairs__random25_draw0"][0,0]+=1
    elif mutation=="prefix":a["edges__dense"][0,0]+=1
    elif mutation=="target":a["target_position"][0,0]+=.1
    elif mutation=="risk_source":a[f"selection_score__{evaluate.RISKS[0]}"]=a[f"selection_score__{evaluate.RISKS[0]}"]+1
    elif mutation=="draw":row["random_seed_materials"]["random25_draw0"][-1]=8
    elif mutation=="parity":row["native_parity"]["passed"]=False
    elif mutation=="boundary":row["ground_truth_boundary"]["maximum_coordinate_excursion"]+=1
    elif mutation=="gain":a["signed_normalized_coordinate_gain__dense"][0]+=1
    elif mutation=="envelope":row["random_envelope"][evaluate.METRICS[0]]["base"]["mean"]+=1
    elif mutation=="failed_metric":row["cases"]["dense"]["failure"]={"category":"pretend"}
    else:del row["cases"]["random25_draw7"]
    with pytest.raises(ValueError):summary.audit_frame(row,a,summary.Audit())


def fake_row(trajectory,frame,value,random_values=None):
    values={metric:{name:float(value) for name in evaluate.POLICIES} for metric in evaluate.METRICS}
    if random_values is not None:
        for metric in values:
            for name,v in zip(evaluate.RANDOM_CASES,random_values):values[metric][name]=v
    return {"split":"valid","source_index":0 if trajectory=="a" else 1,"target_frame":frame,"trajectory_id":trajectory,"values":values}


def expected(rows):return [{key:row[key] for key in summary.IDENTITY} for row in rows]


def test_equal_trajectory_weighting_not_pooled_frames():
    rows=[fake_row("a",7,0),fake_row("a",8,0),fake_row("b",7,9)]
    out=summary.summarize_rows(rows,expected(rows))
    assert out["values"][evaluate.METRICS[0]]["base"]==4.5
    assert out["required_trajectories"]==2 and out["required_frames"]==3


def test_frame_rank_average_not_rank_of_mean_and_all_signs_kept():
    rows=[fake_row("a",7,1,[0.]*8),fake_row("a",8,1,[100.]*8)]
    out=summary.summarize_rows(rows,expected(rows))
    metric=evaluate.METRICS[0]
    assert out["equal_trajectory_mean_of_frame_envelope_measures"][metric]["base"]["random_strictly_better"]==4
    assert out["envelope_of_equal_trajectory_seed_mean_errors"][metric]["base"]["random_strictly_better"]==0
    assert out["envelope_of_equal_trajectory_seed_mean_errors"][metric]["base"]["reference_minus_random_mean"]==-49


def test_failed_draw_nulls_all_eight_envelope_but_keeps_other_case_values():
    rows=[fake_row("a",7,1,[0.]*7+[None]),fake_row("b",7,2,[1.]*8)]
    out=summary.summarize_rows(rows,expected(rows));metric=evaluate.METRICS[0]
    assert out["values"][metric]["base"]==1.5
    assert out["values"][metric]["random25_draw0"]==.5
    assert out["values"][metric]["random25_draw7"] is None
    assert out["envelope_of_equal_trajectory_seed_mean_errors"][metric]["base"]["mean"] is None
    assert out["equal_trajectory_mean_of_frame_envelope_measures"][metric]["base"]["random_strictly_better"] is None


def test_missing_required_frame_nulls_all_sample_and_is_not_survivor_mean():
    rows=[fake_row("a",7,1),fake_row("b",7,2)]
    out=summary.summarize_rows(rows[:1],expected(rows))
    assert out["missing_frames"]==1
    assert all(v is None for v in out["values"][evaluate.METRICS[0]].values())
    assert out["trajectories"]["a"]["values"][evaluate.METRICS[0]]["base"]==1
    with pytest.raises(ValueError,match="duplicate"):summary.summarize_rows(rows+rows,expected(rows))
    unknown=copy.deepcopy(rows[0]);unknown["target_frame"]=99
    with pytest.raises(ValueError,match="Unknown"):summary.summarize_rows([unknown],expected(rows))


def test_seed_statistics_and_full_six_model_cohort():
    assert summary.seed_statistics([1.,2.,3.])["sample_sd"]==1.
    assert summary.seed_statistics([1.,None,3.])["mean"] is None
    runs=[]
    for objective,seed in sorted(evaluate.COHORT):
        rows=[fake_row("a",7,seed+1)]
        splits={split:summary.summarize_rows(rows,expected(rows)) for split in ("valid","test")}
        runs.append({"objective":objective,"seed":seed,"splits":splits})
    out=summary.summarize_cohort(runs)
    assert out["test"]["faithful"]["case_error_statistics"][evaluate.METRICS[0]]["base"]["seed_values"]==[1.,2.,3.]
    with pytest.raises(ValueError,match="six"):summary.summarize_cohort(runs[:-1])
    with pytest.raises(ValueError,match="six"):summary.summarize_cohort(runs[:-1]+[runs[0]])


def test_guard_failure_saved_history_and_complete_null_cases_audit():
    cur,prev,types,target=fixture();cur[0,0,0]=np.nan
    row,a=evaluate.evaluate_frame(tiny_model(),cur,prev,types,target,identity(),0,"cpu");row["seed"]=0
    out=summary.audit_frame(row,a,summary.Audit())
    assert all(v is None for v in out["values"][evaluate.METRICS[0]].values())


def test_failed_previous_scoring_and_failed_random_audit(monkeypatch):
    original=evaluate.bridge.supplied;calls=[0]
    def bad(*args,**kwargs):
        out=original(*args,**kwargs);calls[0]+=1
        if calls[0]==2:out["risk"][0]=np.nan;out["raw_risk"][0]=np.nan
        if calls[0]==6:out["prediction"][0,0]=11.
        return out
    monkeypatch.setattr(evaluate.bridge,"supplied",bad)
    row,a=observed();row["seed"]=0
    out=summary.audit_frame(row,a,summary.Audit())
    assert out["values"][evaluate.METRICS[0]][evaluate.RISKS[0]] is None
    assert out["values"][evaluate.METRICS[0]]["random25_draw0"] is None


def test_summary_output_isolation_uses_pinned_metadata_before_any_creation(tmp_path):
    import json
    run=tmp_path/"run";run.mkdir()
    inputs=tmp_path/"inputs";inputs.mkdir()
    sources=tmp_path/"sources";sources.mkdir()
    saved=tmp_path/"saved";saved.mkdir()
    doc={"checkpoint":str(inputs/"checkpoint.pt"),"validation_manifest":str(inputs/"valid.json"),
         "test_manifest":str(inputs/"test.json"),"saved_same_state_dir":str(saved),
         "input_files_sha256":{str(sources/"helper.py"):"x"}}
    (inputs/"protocol.json").write_text(json.dumps({"source_sha256":{}}))
    external=tmp_path/"external";external.mkdir()
    manifest={"records":[{"positions":{"path":"../external/positions.npy"},"particle_types":{"path":"types.npy"}}]}
    for name in ("valid.json","test.json"):(inputs/name).write_text(json.dumps(manifest))
    (saved/"result.json").write_text(json.dumps({"records":[]}))
    (run/"protocol.json").write_text(json.dumps(doc))
    for output in (run/"analysis",inputs/"analysis",sources/"analysis",saved/"analysis",external/"analysis"):
        with pytest.raises(ValueError,match="separate"):summary.preflight_summary_output(output,[run])
        assert not output.exists()
    output=tmp_path/"fresh"
    assert summary.preflight_summary_output(output,[run])==output
    assert not output.exists()


def test_final_cohort_rehash_catches_early_run_mutation(tmp_path):
    raw=tmp_path/"raw.json";raw.write_text("first")
    inp=tmp_path/"input";inp.write_text("input")
    src=tmp_path/"source";src.write_text("source")
    runs=[{"raw_files_sha256":{str(raw):evaluate.full.sha256(raw)},"input_files_sha256":{str(inp):evaluate.full.sha256(inp)}}]
    pins={str(src):evaluate.full.sha256(src)}
    summary.reverify_cohort(runs,pins)
    raw.write_text("changed after earlier run audit")
    with pytest.raises(ValueError,match="byte identity"):summary.reverify_cohort(runs,pins)


@pytest.mark.parametrize("bad,expected_value",[(True,1),(False,0),({"seed":True},{"seed":1}),([False],[0]),({"x":[True]},{"x":[1]})])
def test_bool_never_substitutes_for_numeric_scalar_or_id(bad,expected_value):
    assert not summary.exact_value(bad,expected_value)
    with pytest.raises(ValueError):summary.Audit().equal(bad,expected_value,"bool mismatch")
    with pytest.raises(ValueError):summary.complete_mean([True,1.])
    with pytest.raises(ValueError):summary.seed_statistics([0.,True,2.])
