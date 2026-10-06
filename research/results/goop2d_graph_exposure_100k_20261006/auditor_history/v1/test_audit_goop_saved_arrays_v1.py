"""Synthetic-only adversarial tests; never loads real models, data or outcomes."""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

import audit_goop_saved_arrays_v1 as audit
import audit_goop_paired_arrays_v1 as paired
import goop_saved_diagnostic_audit_v1 as diagnostic


BOUNDS=np.asarray([[.1,.9],[.1,.9]])


def fixed_boundary(x):
    excess=max(0.,x-.9,.1-x)
    return {"fraction_particles_outside":float(excess>0),"fraction_particles_outside_by_more_than_1e-6":float(excess>1e-6),
            "maximum_coordinate_excursion":excess,"mean_particle_maximum_excursion":excess,
            "coordinate_minimum":[x,.5],"coordinate_maximum":[x,.5]}


def make_rollout(n=395):
    def position(s): return np.asarray([[.5+s/512.,.5]]*2,dtype=np.float32)
    initial=np.stack([position(0)]*6);steps=[s for s in (1,10,50,200,395) if s<=n]
    self_edges=np.asarray([[0,1],[0,1]],dtype=np.int64)
    arr={"initial_observed_positions":initial,"particle_types":np.full(2,7,dtype=np.int64),"bounds":BOUNDS.copy(),
         "forecast_steps":np.asarray(steps,dtype=np.int64),
         "predicted_positions":np.stack([position(s) for s in steps]) if steps else np.empty((0,2,2),np.float32),
         "ground_truth_positions":np.stack([position(0) for s in steps]) if steps else np.empty((0,2,2),np.float32),
         "observed_or_predicted_histories":np.stack([np.stack([position(max(0,k)) for k in range(s-6,s)]) for s in steps]) if steps else np.empty((0,6,2,2),np.float32),
         "current_risk":np.ones((len(steps),2),np.float32),"cached_risk_before_forecast":np.ones((len(steps),2),np.float32)}
    if steps: arr["cached_risk_before_forecast"][0]=np.nan
    parity={"passed":False,"not_attempted_reason":{"category":"candidate_pair_resource_guard"}}
    if n:
        for family in ("native","supplied"):
            for key,val in (("prediction",position(0)),("risk",np.ones(2,np.float32)),("raw_risk",np.ones(2,np.float32)),
                            ("node_features",np.zeros((2,30),np.float32)),("edge_features",np.zeros((2,3),np.float32))):
                arr["initial_parity_"+family+"_"+key]=val.copy()
        arr["initial_parity_native_edges"]=self_edges.copy();arr["initial_parity_supplied_edges"]=self_edges.copy()
        parity={"passed":True,"finite":True,"prediction_risk_agree":True,"native_edge_identity":True,
                "native_supplied_feature_identity":[True,True,True],"prediction_atol":2e-7,"risk_atol":1e-6,"risk_rtol":1e-5,
                "prediction_max_abs_difference":0.,"risk_max_abs_difference":0.,"raw_risk_max_abs_difference":0.}
    errors=[(s/512.)**2/2 for s in range(1,n+1)]
    pred_b=[fixed_boundary(.5+s/512.) for s in range(1,n+1)];truth_b=[fixed_boundary(.5) for _ in range(n)]
    failure=None if n==395 else {"category":"candidate_pair_resource_guard","forecast_step":n+1 if n else 0,"phase":"graph"}
    attempts=[{"forecast_step":s,"accepted":True,"coordinate_mse":errors[s-1],"predicted_boundary":pred_b[s-1],"ground_truth_boundary":truth_b[s-1],
               "predicted_state_sha256":audit.array_hash(position(s)),"directed_edges":2,"retained_optional_pairs":0,
               "native_base_sha256":audit.array_hash(self_edges)} for s in range(1,n+1)]
    if n and failure: attempts.append({"forecast_step":n+1,"accepted":False,"failure":failure})
    row={"status":"complete" if n==395 else "failed","failure":failure,"policy":"base","horizon":395,"completed_steps":n,"particles":2,
         "requested_trace_steps":[1,10,50,200,395],
         "mse_per_step":errors,"mean_rollout_mse":math.fsum(errors)/n if n==395 else None,
         "prefix_mean_mse_if_failed":math.fsum(errors)/n if 0<n<395 else None,"mse_at_final_horizon":errors[-1] if n==395 else None,
         "mse_at_declared_trace_steps":{str(s):errors[s-1] if s<=n else None for s in (1,10,50,200,395)},
         "initial_observed_state_sha256":audit.array_hash(initial),"final_predicted_state_sha256":audit.array_hash(position(n)),
         "initial_observed_boundary":[fixed_boundary(.5) for _ in range(6)],"predicted_boundary_per_step":pred_b,"ground_truth_boundary_per_step":truth_b,
         "directed_edges_per_step":[2]*n,"retained_optional_pairs_per_step":[0]*n,"candidate_pairs_per_step":[0]*n,"base_pairs_per_step":[0]*n,
         "mean_normalized_acceleration_variance_per_step":[1.]*n,"mean_directed_edges":2. if n==395 else None,"attempts":attempts,"native_parity":parity}
    for s in steps:
        arr[f"edges_forecast_{s:04d}"]=self_edges.copy();arr[f"optional_pairs_forecast_{s:04d}"]=np.empty((0,2),np.int64)
    return row,arr


@pytest.mark.parametrize("n",[0,9,10,200,395])
def test_sparse_forecast_and_failure_scope(n):
    row,arr=make_rollout(n)
    metrics,scope=audit.audit_rollout(row,arr,BOUNDS,audit.Checks())
    assert metrics["mean_rollout_mse"] is None if n<395 else metrics["mean_rollout_mse"]==pytest.approx(396*791/(12*512**2))
    assert scope["array_recomputed_mse_forecasts"]==[s for s in (1,10,50,200,395) if s<=n]
    assert "unsaved prediction errors not array-recomputed" in scope["full_and_prefix_MSE_verification"]
    if n>=200: assert metrics["mse_forecast200"]==.5*(200/512.)**2


@pytest.mark.parametrize("corruption",["mean","mse","boundary","step","edge","duplicate","parity","cache","attempt"])
def test_rollout_detects_independent_corruptions(corruption):
    row,arr=make_rollout()
    if corruption=="mean": row["mean_rollout_mse"]+=.1
    if corruption=="mse": arr["ground_truth_positions"][0,0,0]+=.2
    if corruption=="boundary": row["predicted_boundary_per_step"][9]["maximum_coordinate_excursion"]+=.2
    if corruption=="step": arr["forecast_steps"][1]=11
    if corruption=="edge": arr["edges_forecast_0001"][0,0]=2
    if corruption=="duplicate": arr["edges_forecast_0001"][:]=0
    if corruption=="parity": arr["initial_parity_supplied_prediction"][0,0]+=.1
    if corruption=="cache": arr["cached_risk_before_forecast"][1]=np.nan
    if corruption=="attempt": row["attempts"][10]["coordinate_mse"]+=.1
    with pytest.raises(ValueError): audit.audit_rollout(row,arr,BOUNDS,audit.Checks())


def test_retained_nonfinite_rejection_checked_without_retry():
    row,arr=make_rollout(10)
    row["failure"]["category"]="nonfinite_prediction"
    arr["rejected_prediction"]=np.full((2,2),np.nan,np.float32)
    arr["rejected_risk"]=arr["rejected_raw_risk"]=np.ones(2,np.float32)
    _,info=audit.audit_rollout(row,arr,BOUNDS,audit.Checks())
    assert "rejected_output variance/guard arithmetic" in info["saved_guard_checks"]
    row["failure"]["category"]="nonfinite_risk"
    with pytest.raises(ValueError,match="guard category"): audit.audit_rollout(row,arr,BOUNDS,audit.Checks())


def test_equal_source_weight_and_null_propagation():
    expected=[(0,6),(0,7),(1,6)]
    result=audit.aggregate(expected,{(0,6):0.,(0,7):2.,(1,6):9.})
    assert result["equal_trajectory_mean"]==5. # Not pooled frame mean 11/3.
    assert result["defined_frames"]==3
    missing=audit.aggregate(expected,{(0,6):0.,(1,6):9.})
    assert missing["equal_trajectory_mean"] is None
    assert missing["defined_frames"]==2 and missing["defined_trajectories"]==1


@pytest.mark.parametrize("method,chosen",[("base",[]),("dense",[[0,2],[0,3],[1,3],[2,3]]),
    ("random25",[[2,3]]),("speed25",[[0,2]]),("previous-observed-base-risk25",[[0,3]]),("relative-velocity-RMS25",[[0,2]])])
def test_same_state_actions_use_saved_universe_and_scores(method,chosen):
    history=np.zeros((6,4,2),np.float32);history[-1,:,0]=[0,.0025,.0075,0]
    mandatory=np.asarray([[0,1,2,3,0,1,1,2],[0,1,2,3,1,0,2,1]],np.int64)
    optional=np.asarray(chosen,np.int64).reshape(-1,2)
    edges=np.concatenate((mandatory,optional.T,optional[:,::-1].T),axis=1)
    annulus=np.asarray([[0,2],[0,3],[1,3],[2,3]],np.int64)
    v1=float(history[-1,1,0])/.0025;v2=float(history[-1,2,0])/.0025
    physical=np.asarray([v1,math.sqrt((v1*v1+(v2-v1)**2)/2),v2-v1,0.])
    risk=np.asarray([1,2,3,9],np.float32)
    audit.saved_action(method,history,edges,optional,annulus,[20261006,93000,2,1,4,105],risk,physical,audit.Checks())
    if method=="relative-velocity-RMS25":
        physical[1]+=.1
        with pytest.raises(ValueError,match="RMS"): audit.saved_action(method,history,edges,optional,annulus,[],risk,physical,audit.Checks())
    if method=="speed25":
        wrong=np.asarray([[2,3]],np.int64) # Same priority, wrong fixed lexical tie.
        with pytest.raises(ValueError,match="canonical optional action"): audit.saved_action(method,history,edges,wrong,annulus,[],risk,physical,audit.Checks())


def nested_set(tree,key,value):
    parts=key.split("/")
    for p in parts[:-1]: tree=tree.setdefault(p,{})
    tree[parts[-1]]=value


def make_collection(tmp_path):
    queue=tmp_path/"queue";queue.mkdir();files={};stages=[]
    for arm,seed in (("base",2),("mix",2)):
        for stage in audit.STAGES:
            mode="full-rollout" if stage=="full_rollout_test" else "clean-validation" if stage=="clean_validation" else "same-state"
            split="test" if stage in ("full_rollout_test","same_state_test") else "valid"
            cells=[{"source_index":i,"policy" if stage=="full_rollout_test" else "target_frame":p,"state":"never_started"} for i,p in audit.schedule(stage)]
            diag={}
            if mode!="full-rollout":
                for key in diagnostic.expected_metric_keys(mode): nested_set(diag,key,audit.aggregate(audit.schedule(stage),{}))
            stages.append({"arm":arm,"seed":seed,"stage":stage,"mode":mode,"split":split,"cells":cells,"rows":[],"diagnostic_summary":diag,"protocol_sha256":None})
    first=stages[0];directory=Path("jobs/base_seed2/full_rollout_test")
    (queue/directory).mkdir(parents=True)
    def store(name,raw):
        rel=directory/name;(queue/rel).write_bytes(raw);files[str(rel)]={"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
        return files[str(rel)]["sha256"]
    protocol={"schema":audit.EVALUATION_SCHEMA,"mode":"full-rollout","split":"test","source_frame_count":401,
              "model":{"arm":"base","seed":2,"completed_updates":100000,"objective":"faithful"},"schedule":[{"source_index":i} for i in range(30)]}
    psha=store("protocol.json",json.dumps(protocol).encode());first["protocol_sha256"]=psha
    row,arrays=make_rollout(10);row.update(arm="base",training_seed=2,objective="faithful",source_index=0,protocol_sha256=psha,trace_file="trajectory_000000_base.npz")
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);row["trace_sha256"]=store(row["trace_file"],buffer.getvalue())
    name="trajectory_000000_base.json";rsha=store(name,json.dumps(row).encode());first["rows"]=[row]
    first["cells"][0].update(state="recorded_failed_outcome",failure=row["failure"],path=str(Path("/original")/directory/name),sha256=rsha)
    collection={"schema":"adaptgns_goop_evaluation_collection_scoped_v3","status":"stopped_outputs_collected","collector_sha256":audit.COLLECTOR_SHA,
                "source_sha256":audit.SOURCE_PINS,"protocol_sha256":audit.PROTOCOL_SHA,"timing_scope":"shared_host_operational_measurement",
                "queue_status":{"all_pinned_inputs_reverified":True},"host_role":"B","original_queue_root":"/original","files":files,"output_tree_state":{"files":files},
                "stages":stages,"cohort_sha256":"a"*64}
    path=tmp_path/"collection.json";path.write_text(json.dumps(collection))
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),queue,collection


def test_full_fixed_collection_keeps_failed_and_missing(tmp_path):
    path,sha,queue,_=make_collection(tmp_path)
    result=audit.audit_collection(path,sha,queue)
    assert result["status"]=="passed_supported_checks"
    assert len(result["models"])==8
    assert result["models"][0]["coverage"]=={"recorded_failed_outcome":1,"never_started":179}
    assert result["models"][0]["metrics"]["mean_rollout_mse/base"] is None
    assert len(result["row_checks"])==1


@pytest.mark.parametrize("kind",["artifact","row","collection","duplicate_cell","omitted_stage","symlink"])
def test_collection_rejects_unbound_or_incomplete_inputs(tmp_path,kind):
    path,sha,queue,col=make_collection(tmp_path)
    if kind in ("artifact","row"):
        rel=next(k for k in col["files"] if k.endswith(".npz" if kind=="artifact" else "base.json"));f=queue/rel;f.write_bytes(f.read_bytes()+b" ")
    if kind=="collection": path.write_bytes(path.read_bytes()+b" ")
    if kind=="duplicate_cell":
        col["stages"][0]["cells"][1]=col["stages"][0]["cells"][0];path.write_text(json.dumps(col));sha=audit.digest(path.read_bytes())
    if kind=="omitted_stage":
        col["stages"].pop();path.write_text(json.dumps(col));sha=audit.digest(path.read_bytes())
    if kind=="symlink":
        rel=next(k for k in col["files"] if k.endswith(".npz"));f=queue/rel;replacement=tmp_path/"moved.npz";f.rename(replacement);f.symlink_to(replacement)
    with pytest.raises(ValueError): audit.audit_collection(path,sha,queue)


def synthetic_receipts():
    audits={}
    for role,models in audit.MODELS.items():
        out=[]
        for arm,s in models:
            for stage in audit.STAGES:
                if stage=="full_rollout_test":
                    metrics={m+"/"+p:float((1 if arm=="base" else 2)*(s+1)+(pi+1)**2) for m in audit.FULL_METRICS for pi,p in enumerate(audit.POLICIES)}
                else: metrics={k:float((1 if arm=="base" else 2)*(s+1)+ki) for ki,k in enumerate(diagnostic.expected_metric_keys("clean-validation" if stage=="clean_validation" else "same-state"))}
                cells=[{"state":"completed_required_outcome"} for _ in audit.schedule(stage)]
                out.append({"arm":arm,"seed":s,"stage":stage,"metrics":metrics,"cells":cells})
        audits[role]={"host_role":role,"models":out}
    return audits


def test_three_seed_sign_sample_sd_and_complete_pairing():
    result=paired.recompute(synthetic_receipts())
    value=result["full_rollout"]["absolute"]["mean_rollout_mse"]["base"]["base"]
    assert value=={"seed_values":{"0":2.,"1":3.,"2":4.},"required_seed_pairs":3,"defined_seed_pairs":3,"mean":3.,"sample_sd":1.}
    assert result["full_rollout"]["mix_minus_base"]["mean_rollout_mse"]["base"]["mean"]==2.
    assert result["full_rollout"]["within_arm_policy_contrasts"]["mean_rollout_mse"]["base"]["laggedrisk25_minus_random25"]["mean"]==16.
    assert result["full_rollout"]["risk_minus_random_mix_minus_base_interaction"]["mean_rollout_mse"]["mean"]==0.


def test_missing_seed_remains_null_and_cannot_hide_in_summary():
    receipts=synthetic_receipts();receipts["B"]["models"][0]["metrics"]["mean_rollout_mse/base"]=None
    result=paired.recompute(receipts);value=result["full_rollout"]["absolute"]["mean_rollout_mse"]["base"]["base"]
    assert value["mean"] is None and value["sample_sd"] is None and value["defined_seed_pairs"]==2
    tampered=deepcopy(result);tampered["full_rollout"]["absolute"]["mean_rollout_mse"]["base"]["base"]["mean"]=2.5
    with pytest.raises(ValueError): paired.supported_equal(tampered,result,audit.Checks())
    del receipts["B"]["models"][0]
    with pytest.raises(ValueError,match="complete six-model"): paired.recompute(receipts)


@pytest.mark.parametrize("tamper",[None,"cohort","collection","source","cross_host_truth","extra_seed","byte"])
def test_paired_receipt_admission(tmp_path,tamper):
    receipts=synthetic_receipts()
    for role,r in receipts.items():
        r.update(schema="goop_saved_array_audit_v1",status="passed_supported_checks",cohort_sha256="f"*64,
                 collection_sha256=("a" if role=="A" else "b")*64,auditor_sha256=audit.file_hash(audit.__file__),
                 diagnostic_helper_sha256=audit.file_hash(diagnostic.__file__),shared_saved_state_hashes={"test/example":"e"*64})
    summary=paired.recompute(receipts)
    summary.update(schema="adaptgns_goop_graph_support_paired_scalar_summary_scoped_v3",status="fixed_scalar_aggregation_complete",
                   summarizer_sha256=audit.COLLECTOR_SHA,source_sha256=audit.SOURCE_PINS,protocol_sha256=audit.PROTOCOL_SHA,
                   cohort_sha256="f"*64,collection_sha256={"A":"a"*64,"B":"b"*64})
    if tamper=="cohort": summary["cohort_sha256"]="c"*64
    if tamper=="collection": summary["collection_sha256"]["A"]="c"*64
    if tamper=="source": receipts["A"]["auditor_sha256"]="c"*64
    if tamper=="cross_host_truth": receipts["B"]["shared_saved_state_hashes"]["test/example"]="c"*64
    if tamper=="extra_seed": summary["full_rollout"]["absolute"]["mean_rollout_mse"]["base"]["base"]["seed_values"]["3"]=7.
    paths={};hashes={}
    for role,r in receipts.items():
        paths[role]=tmp_path/(role+".json");paths[role].write_text(json.dumps(r));hashes[role]=audit.file_hash(paths[role])
    path=tmp_path/"summary.json";path.write_text(json.dumps(summary));sha=audit.file_hash(path)
    if tamper=="byte": path.write_bytes(path.read_bytes()+b" ")
    if tamper is None:
        result=paired.audit_summary(paths,hashes,path,sha)
        assert result["status"]=="passed_supported_checks" and result["verified_paired_scalars"]["full_rollout"]["required_outcomes"]==1080
    else:
        with pytest.raises(ValueError): paired.audit_summary(paths,hashes,path,sha)
