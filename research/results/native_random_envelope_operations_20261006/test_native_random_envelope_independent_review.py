"""Independent review regressions on fabricated complete scalar evidence only."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("envelope_independent_review",Path(__file__).with_name("audit_native_random_envelope_independent.py"))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def write(path, value):
    path.write_text(json.dumps(value,allow_nan=False))


def synthetic_frame(split, i, values):
    source=(i%30) if split=="valid" else 3+i//11
    target=(7+i//30) if split=="valid" else 7+i%11
    return dict(split=split,source_index=source,target_frame=target,
                trajectory_id=f"{split}:{source:06d}",values=copy.deepcopy(values))


def value_fixture():
    arrays={"target_position":np.zeros((1,2)),"acceleration_std":np.array([1.,2.])}
    row={"cases":{},"benefits":{}}
    values={m:{} for m in a.METRICS}
    se={}
    for i,case in enumerate(a.CASES):
        d=(i+1)*.01
        p=np.array([[d,2*d]])
        arrays["prediction__"+case]=p
        normalized=p/arrays["acceleration_std"]
        se[case]=np.mean(normalized**2,axis=-1)
        arrays["normalized_coordinate_se__"+case]=se[case]
        metrics={"position_coordinate_mse":float(np.mean(p**2)),"normalized_coordinate_mse":float(np.mean(normalized**2))}
        row["cases"][case]={"status":"complete","failure":None,"metrics":metrics}
        for m in a.METRICS:values[m][case]=metrics[m]
    for case in a.CASES:
        gains=se["base"]-se[case]
        arrays["signed_normalized_coordinate_gain__"+case]=gains
        row["benefits"][case]={"mean_signed_normalized_coordinate_gain":float(gains.mean()),"harmful_particle_fraction":float(np.mean(gains<0))}
    row["random_envelope"]=a.envelopes(values)
    return row,arrays,values


@pytest.fixture(scope="module")
def fabricated_complete(tmp_path_factory):
    root=tmp_path_factory.mktemp("fabricated_native_envelope")
    source=root/"strict_summary_source.py";source.write_text("# synthetic source identity only\n")
    template,arrays,values=value_fixture()
    runs=[]
    for objective,seed in sorted(a.COHORT):
        directory=root/f"{objective}_{seed}";directory.mkdir()
        array=directory/"arrays.npz";np.savez_compressed(array,**arrays)
        frames=[synthetic_frame(split,i,values) for split,count in (("valid",128),("test",297)) for i in range(count)]
        expected=[{k:r[k] for k in a.IDENTITY} for r in frames]
        checkpoint=directory/"checkpoint.identity";checkpoint.write_text(f"synthetic {objective} {seed}")
        protocol={"objective":objective,"seed":seed,"expected_frames":expected}
        write(directory/"protocol.json",protocol)
        ph=a.sha(directory/"protocol.json");cp=a.sha(checkpoint)
        compact=[]
        pins={str(array):a.sha(array)}
        for i,frame in enumerate(frames):
            row={**copy.deepcopy(template),**{k:frame[k] for k in a.IDENTITY},"status":"complete","failure":None,
                 "objective":objective,"seed":seed,"checkpoint_sha256":cp,"protocol_sha256":ph,
                 "array_file":array.name,"array_sha256":a.sha(array)}
            path=directory/f"frame_{i:04d}.json";write(path,row)
            compact.append({"record_file":path.name,"record_sha256":a.sha(path)})
            pins[str(path)]=a.sha(path)
        raw={"state":"complete","checkpoint_sha256":cp,"protocol_sha256":ph,"records":compact}
        write(directory/"result.json",raw)
        write(directory/"status.json",{"state":"complete","result_sha256":a.sha(directory/"result.json")})
        for name in ("protocol.json","result.json","status.json"):pins[str(directory/name)]=a.sha(directory/name)
        splits={}
        for split in ("valid","test"):
            subset=[r for r in frames if r["split"]==split]
            sch=[r for r in expected if r["split"]==split]
            splits[split]={**a.aggregate(subset,sch),"frames":[{**r,"envelope":a.envelopes(r["values"])} for r in subset]}
        runs.append({"objective":objective,"seed":seed,"checkpoint_sha256":cp,"raw_files_sha256":pins,
                     "input_files_sha256":{str(checkpoint):cp},"complete_frames":425,"failed_frames":0,"splits":splits})
    groups={}
    lookup={(r["objective"],r["seed"]):r for r in runs}
    for split in ("valid","test"):
        groups[split]={}
        for objective in ("faithful","nll"):
            seeds=[lookup[objective,s]["splits"][split] for s in range(3)]
            groups[split][objective]={"seeds":{str(s):copy.deepcopy(seeds[s]) for s in range(3)},
                "case_error_statistics":{m:{c:a.stats([r["values"][m][c] for r in seeds]) for c in a.CASES} for m in a.METRICS},
                "control_minus_random_mean_statistics":{m:{c:a.stats([r["equal_trajectory_mean_of_frame_envelope_measures"][m][c]["reference_minus_random_mean"] for r in seeds]) for c in a.CONTROLS} for m in a.METRICS},
                "mean_frame_comparison_statistics":{m:{c:{f:a.stats([r["equal_trajectory_mean_of_frame_envelope_measures"][m][c][f] for r in seeds]) for f in a.FIELDS} for c in a.CONTROLS} for m in a.METRICS}}
    report={"scope":"post-inspection exploratory native random-action envelope on original100k; not independent confirmation",
            "coverage":{"models":6,"frames":2550,"case_slots":33150,"random_slots":20400},
            "source_files_sha256":{str(source):a.sha(source)},"runs":runs,"summary":groups}
    return root,report


def save_report(root,report,label):
    directory=root/label;directory.mkdir()
    path=directory/"results.json";write(path,report)
    write(directory/"status.json",{"state":"complete","result_sha256":a.sha(path)})
    return path


def test_fabricated_complete_cohort_passes(fabricated_complete):
    root,report=fabricated_complete
    path=save_report(root,report,"valid_summary")
    assert a.execute(path,a.Audit())==dict(models=6,frames=2550,case_slots=33150,random_slots=20400)


@pytest.mark.parametrize("where",["frame_envelope","copied_seed"])
def test_tampered_published_duplicate_envelope_or_seed_is_rejected(fabricated_complete,where):
    root,original=fabricated_complete;report=copy.deepcopy(original)
    if where=="frame_envelope":
        report["runs"][0]["splits"]["valid"]["frames"][0]["envelope"][a.METRICS[0]]["base"]["mean"]+=1
    else:
        report["summary"]["test"]["nll"]["seeds"]["2"]["values"][a.METRICS[0]]["base"]+=1
    path=save_report(root,report,where)
    with pytest.raises(ValueError):a.execute(path,a.Audit())


def test_independent_sum_one_ulp_difference_can_change_exact_aggregate_ties():
    draw=[1.,1e-16,1e-16]
    reference=float(np.mean(draw))
    assert reference==.3333333333333333
    assert a.mean(draw)==.3333333333333334
    assert a.mean([reference]*3)==reference
    # The protocol defines ties on the audited recorded aggregate scalars.
    recorded=a.envelope([reference]*8,reference)
    assert recorded["random_tied"]==8
    independently_rounded=a.envelope([a.mean(draw)]*8,a.mean([reference]*3))
    assert independently_rounded["random_tied"]==0


def test_recorded_trajectory_and_seed_scalars_govern_exact_ties_after_arithmetic_audit():
    draw=[1.,1e-16,1e-16];ref=float(np.mean(draw))
    frames=[]
    for i,value in enumerate(draw):
        values={m:{c:ref for c in a.CASES} for m in a.METRICS}
        for m in a.METRICS:
            for c in a.DRAWS:values[m][c]=value
        frames.append(dict(split="valid",source_index=0,target_frame=7+i,trajectory_id="A",values=values))
    expected=[{k:r[k] for k in a.IDENTITY} for r in frames]
    recorded=a.aggregate(frames,expected)
    for m in a.METRICS:
        for c in a.CASES:
            recorded["trajectories"]["A"]["values"][m][c]=ref
            recorded["values"][m][c]=ref
    checked=a.aggregate(frames,expected,a.Audit(),recorded)
    for m in a.METRICS:
        assert checked["trajectories"]["A"]["envelope_of_trajectory_mean_errors"][m]["base"]["random_tied"]==8
        assert checked["envelope_of_equal_trajectory_seed_mean_errors"][m]["base"]["random_tied"]==8


def test_output_isolation_includes_source_and_audit_directories(tmp_path):
    sources=tmp_path/"sources";sources.mkdir()
    summaries=tmp_path/"summaries";summaries.mkdir()
    path=summaries/"results.json"
    write(path,{"source_files_sha256":{str(sources/"source.py"):"synthetic"},"runs":[]})
    for output in (sources/"new",Path(a.__file__).parent/"must_not_create_audit_child",summaries/"new"):
        with pytest.raises(ValueError):a.isolated_output(path,output)
        assert not output.exists()
    outside=tmp_path/"clean_audit"
    assert a.isolated_output(path,outside)==outside
    assert not outside.exists()
