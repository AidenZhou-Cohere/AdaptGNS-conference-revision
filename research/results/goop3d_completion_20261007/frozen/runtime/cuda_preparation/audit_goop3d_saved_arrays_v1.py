#!/usr/bin/env python3
"""Independent stopped Goop-3D H295 saved-array and scalar-series audit.

Only NumPy and this audit's arithmetic helper are imported. No original
scientific module, model, source trajectory, spatial search or GPU is run.
Sparse saved positions support only their retained forecast steps. Full/prefix
MSE and boundary means are arithmetic checks of recorded scalar series.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import traceback
import numpy as np
import goop3d_saved_diagnostic_audit_v1 as diagnostic
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
COLLECTOR_SHA="d7096bef018ac7e3d28cc429a4711d88209e48a1b812dda0aad995d6a41de5fd"
SOURCE_PINS={
"evaluate_goop3d_graph_support_v1.py":"9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de",
"goop3d_diagnostic_metrics_v1.py":"5ef3de96e470eea495dd56c1f60396bbd166b9ea5c5bc340864715cb9e2c173b",
"goop3d_native_evaluation_v1.py":"a742123093aff433f3a4e302929a52bae4f5fee9610195d86df726852575b1d5",
"train_goop3d_graph_support_cuda_v2.py":"8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc",
"goop3d_graph_support_vectorized_v1.py":"ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50"}
PROTOCOL_SHA="5010f9023a3eee45f85bee80b35fc8e506ca145667daf75027b32c92658faa70"
METADATA_SHA="727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55"
EVALUATION_SCHEMA="adaptgns_goop3d_graph_support_evaluation_v1"
COLLECTION_SCHEMA="adaptgns_goop3d_stopped_scalar_collection_v1"
POLICIES=("base","dense","random25","speed25","laggedrisk25","relative-velocity-RMS25")
MODELS=tuple((a,s) for a in ("base","mix") for s in range(3))
STAGES=(("full_rollout_valid","full-rollout","valid"),("full_rollout_test","full-rollout","test"),
("same_state_valid","same-state","valid"),("same_state_test","same-state","test"),("clean_validation","clean-validation","valid"))
CELL_STATES={"completed_required_outcome","recorded_failed_outcome","timed_out_current","not_completed_before_invocation_end","never_started"}
BOUNDARY_KEYS=("fraction_particles_outside","fraction_particles_outside_by_more_than_1e-6","maximum_coordinate_excursion","mean_particle_maximum_excursion")
G_KEYS=("candidate_pairs","geometric_base_pairs","available_annulus_pairs","optional_pair_budget","retained_optional_pairs","directed_edges","native_base_directed_edges","native_base_self_edges","native_base_receivers_above_cap_before_capping","native_base_edges_removed_by_cap","native_base_max_receiver_degree","native_base_asymmetric_directed_edges")
FULL_METRICS=("mean_rollout_mse","mse_forecast200","mse_forecast295")+tuple(side+"_boundary_"+key for side in ("predicted","ground_truth") for key in tuple("mean_"+k for k in BOUNDARY_KEYS)+("trajectory_maximum_excursion",))+tuple("graph_mean_"+k for k in G_KEYS)+("graph_fraction_steps_cap_active",)

def digest(raw): return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def array_hash(a): return digest(np.ascontiguousarray(a).tobytes())


def finite(x): return type(x) in (float, int) and math.isfinite(x)


def mean(xs): return math.fsum(xs)/len(xs) if xs and all(finite(x) for x in xs) else None


class Checks:
    def __init__(self): self.count=0; self.context=""
    def require(self, condition, label):
        self.count+=1
        if not condition: raise ValueError(self.context+": "+label)
    def equal(self, actual, expected, label): self.require(actual==expected,label)
    def close(self, actual, expected, label):
        if isinstance(expected, dict):
            self.require(isinstance(actual,dict) and set(actual)==set(expected),label+" keys")
            for key,value in expected.items(): self.close(actual[key],value,label+"/"+str(key))
        elif isinstance(expected,(list,tuple)):
            self.require(isinstance(actual,(list,tuple)) and len(actual)==len(expected),label+" length")
            for i,value in enumerate(expected): self.close(actual[i],value,label+"/"+str(i))
        elif expected is None or isinstance(expected,(str,bool)):
            self.equal(actual,expected,label)
        else:
            self.require(finite(actual) and finite(expected) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),label)
    def array(self, actual, expected, label):
        a,b=np.asarray(actual),np.asarray(expected)
        self.require(a.shape==b.shape,label+" shape")
        self.require(np.allclose(a,b,rtol=1e-10,atol=1e-12,equal_nan=True),label+" values")


def boundary(points, bounds):
    points=np.asarray(points,dtype=np.float64)
    excess=np.maximum(np.maximum(bounds[:,0]-points,points-bounds[:,1]),0.)
    worst=excess.max(axis=1)
    return {"fraction_particles_outside":float(np.count_nonzero(worst>0)/len(points)),
            "fraction_particles_outside_by_more_than_1e-6":float(np.count_nonzero(worst>1e-6)/len(points)),
            "maximum_coordinate_excursion":float(np.max(worst)),"mean_particle_maximum_excursion":float(np.mean(worst)),
            "coordinate_minimum":points.min(axis=0).tolist(),"coordinate_maximum":points.max(axis=0).tolist()}


def aggregate(expected, values):
    per_source={}
    for source in sorted({s for s,_ in expected}):
        items=[values.get(key) for key in expected if key[0]==source]
        per_source[str(source)]={"expected":len(items),"defined":sum(finite(v) for v in items),"mean":mean(items)}
    return {"expected_frames":len(expected),"defined_frames":sum(v["defined"] for v in per_source.values()),
            "expected_trajectories":len(per_source),"defined_trajectories":sum(v["mean"] is not None for v in per_source.values()),
            "equal_trajectory_mean":mean([v["mean"] for v in per_source.values()]),"trajectories":per_source}


def tree_leaf(tree,key):
    for part in key.split("/"): tree=tree[part]
    return tree


def diagnostic_shared_hashes(arrays,split,unit,mode,checks):
    """Keep raw representation identity separate from cross-mode source truth.

    Frozen same-state rows save the original float32 source target promoted to
    float64; clean-validation rows save float32. A lossless roundtrip is required
    before using one canonical float32 identity across these representations.
    """
    checks.require(mode in ("same-state","clean-validation"),"known pairing mode")
    hashes={f"{split}/{key}/{unit}":array_hash(arrays[key]) for key in
            ("current_history","previous_history","observed_history","particle_types") if key in arrays}
    if "target_position" in arrays:
        target=arrays["target_position"]
        dtype=np.dtype(np.float32 if mode=="clean-validation" else np.float64)
        checks.equal(target.dtype,dtype,"target pairing frozen representation dtype")
        checks.equal(target.shape,(len(arrays["particle_types"]),3),"target pairing shape")
        checks.require(np.isfinite(target).all(),"finite source target pairing")
        canonical=target.astype(np.float32)
        checks.require(np.isfinite(canonical).all() and np.array_equal(target,canonical.astype(dtype)),
                       "source target lossless float32 roundtrip")
        hashes[f"{split}/{mode}/target_position/{unit}"]=array_hash(target)
        hashes[f"{split}/truth_float32/{unit[0]}/{unit[1]}"]=array_hash(canonical)
    return hashes


def graph_arrays(edges, optional, graph, n, checks, *, base_pairs=None, annulus=None):
    checks.require(edges.ndim==2 and edges.shape[0]==2 and np.issubdtype(edges.dtype,np.integer),"saved edge shape/type")
    checks.require(optional.ndim==2 and optional.shape[1]==2 and np.issubdtype(optional.dtype,np.integer),"optional pair shape/type")
    checks.require(np.all((edges>=0)&(edges<n)) and np.all((optional>=0)&(optional<n)),"saved edge particle IDs")
    checks.require(np.all(optional[:,0]<optional[:,1]) and len(np.unique(optional,axis=0))==len(optional),"unique canonical optional pairs")
    k=len(optional); mandatory_count=edges.shape[1]-2*k
    checks.require(mandatory_count>=0,"nonnegative mandatory edge prefix")
    mandatory=edges[:,:mandatory_count]
    appended=edges[:,mandatory_count:]
    checks.equal(set(map(tuple,appended.T.tolist())),set(map(tuple,optional.tolist())) | set(map(tuple,optional[:,::-1].tolist())),"optional bidirectional edge set")
    checks.equal(len(set(map(tuple,appended.T.tolist()))),2*k,"no duplicate optional directed edges")
    checks.equal(len(set(map(tuple,mandatory.T.tolist()))),mandatory_count,"unique mandatory edges")
    checks.require(not(set(map(tuple,mandatory.T.tolist())) & set(map(tuple,appended.T.tolist()))),"mandatory/optional edges disjoint")
    checks.require(set(mandatory[0,mandatory[0]==mandatory[1]].tolist())<=set(range(n)),"retained mandatory self-edge identities")
    counts={"directed_edges":edges.shape[1],"retained_optional_pairs":k,"native_base_directed_edges":mandatory_count,
            "native_base_self_edges":int(np.sum(mandatory[0]==mandatory[1]))}
    nonself={(int(i),int(j)) for i,j in mandatory.T if i!=j}
    counts["native_base_asymmetric_directed_edges"]=sum((j,i) not in nonself for i,j in nonself)
    degree=np.bincount(mandatory[1],minlength=n)
    counts["native_base_max_receiver_degree"]=int(degree.max())
    checks.require(np.all(degree<=128),"saved mandatory receiver cap")
    if base_pairs is not None and annulus is not None:
        for name,pairs in (("base",base_pairs),("annulus",annulus)):
            checks.require(pairs.ndim==2 and pairs.shape[1]==2 and np.issubdtype(pairs.dtype,np.integer),name+" pair shape")
            checks.require(np.all((pairs>=0)&(pairs<n)) and np.all(pairs[:,0]<pairs[:,1]),name+" pair IDs")
            checks.equal(len(np.unique(pairs,axis=0)),len(pairs),name+" pair uniqueness")
        checks.require(not(set(map(tuple,base_pairs.tolist())) & set(map(tuple,annulus.tolist()))),"base/annulus disjoint")
        checks.require(set(map(tuple,optional.tolist()))<=set(map(tuple,annulus.tolist())),"optional subset of saved annulus")
        checks.require({tuple(sorted((i,j))) for i,j in nonself}<=set(map(tuple,base_pairs.tolist())),"mandatory nonself belongs to saved base pairs")
        uncapped=1+np.bincount(base_pairs.reshape(-1),minlength=n)
        checks.equal(degree.tolist(),np.minimum(uncapped,128).tolist(),"retained perreceiver cap over saved base support")
        counts.update(candidate_pairs=len(base_pairs)+len(annulus),geometric_base_pairs=len(base_pairs),available_annulus_pairs=len(annulus),
                      optional_pair_budget=len(annulus)//4,native_base_edges_removed_by_cap=2*len(base_pairs)+n-mandatory_count,
                      native_base_receivers_above_cap_before_capping=int(np.sum(uncapped>128)))
    for key,value in counts.items():
        if key in graph: checks.equal(graph[key],value,"graph scalar "+key)
    for key,value in (("directed_edge_sha256",array_hash(edges)),("selected_optional_pair_sha256",array_hash(optional)),("native_base_sha256",array_hash(mandatory))):
        if key in graph: checks.equal(graph[key],value,"graph array hash "+key)
    return counts


def saved_action(method,history,edges,optional,annulus,material,old_risk,physical_saved,checks):
    """Recompute actions only over an already saved candidate list, not space."""
    if method in ("base","natural_base_reference"):
        checks.equal(len(optional),0,"base has no optional pairs");return
    if annulus is None: return
    checks.equal(annulus.tolist(),sorted(annulus.tolist()),"saved annulus lexicographic order")
    budget=len(annulus) if method=="dense" else len(annulus)//4
    if method=="dense": chosen=annulus
    elif method=="random25":
        # NumPy PCG64/SeedSequence is the declared random mechanism, not a model.
        rng=np.random.default_rng(np.random.SeedSequence(material))
        chosen=annulus[rng.permutation(len(annulus))[:budget]]
    else:
        if method=="speed25":
            delta=history[-1]-history[-2]
            score=np.sqrt(np.sum(delta*delta,axis=-1))
        elif method=="previous-observed-base-risk25":
            score=old_risk
            checks.require(score is not None,"saved previous-base score exists")
        elif method=="relative-velocity-RMS25":
            mandatory=edges[:,:edges.shape[1]-2*len(optional)]
            senders,receivers=mandatory[:,mandatory[0]!=mandatory[1]]
            velocity=(history[-1].astype(float)-history[-2].astype(float))/.0025
            delta=velocity[receivers]-velocity[senders]
            squares=np.sum(delta*delta,axis=1)
            counts=np.bincount(receivers,minlength=len(history[-1]))
            totals=np.bincount(receivers,weights=squares,minlength=len(history[-1])).astype(np.float64,copy=False)
            score=np.sqrt(np.divide(totals,counts,out=np.zeros_like(totals),where=counts>0))
            checks.require(physical_saved is not None,"retained physical scores exist")
            checks.array(physical_saved,score,"relative-velocity RMS on saved native incoming support")
        else: raise ValueError("Unknown saved diagnostic method")
        checks.require(np.asarray(score).shape==(history.shape[1],) and np.isfinite(score).all(),"finite action score vector")
        priority=np.maximum(score[annulus[:,0]],score[annulus[:,1]])
        chosen=annulus[np.argsort(-priority,kind="stable")[:budget]]
    checks.equal(optional.tolist(),sorted(chosen.tolist()),"canonical optional action over saved candidate universe")


def audit_rollout(row, arrays, bounds, checks):
    n=row["completed_steps"]; horizon=295
    checks.require(type(n) is int and 0<=n<=horizon,"accepted prefix")
    checks.equal(row["horizon"],horizon,"Goop-3D horizon")
    complete=row["status"]=="complete"
    checks.require(row["status"] in ("complete","failed") and complete==(n==horizon),"status/prefix consistency")
    checks.require((row["failure"] is None)==complete,"failure/status consistency")
    mse=row["mse_per_step"]
    checks.require(len(mse)==n and all(finite(v) and v>=0 for v in mse),"complete accepted scalar MSE series")
    checks.close(row["mean_rollout_mse"],mean(mse) if complete else None,"full-H scalar-series mean")
    checks.close(row["prefix_mean_mse_if_failed"],mean(mse) if not complete else None,"failed-prefix scalar-series mean")
    checks.close(row["mse_at_final_horizon"],mse[-1] if complete else None,"final scalar-series MSE")
    for step in (1,10,50,200,295):
        checks.close(row["mse_at_declared_trace_steps"][str(step)],mse[step-1] if step<=n else None,"declared scalar step "+str(step))
    checks.array(arrays["bounds"],bounds,"trace/metadata bounds")
    initial=arrays["initial_observed_positions"]
    checks.require(initial.ndim==3,"initial history rank")
    particles=initial.shape[1]
    checks.require(initial.shape==(6,particles,3) and particles>0,"initial history shape")
    checks.equal(initial.dtype.str,np.dtype(np.float32).str,"initial history float32")
    checks.require(arrays["particle_types"].shape==(particles,) and np.all(arrays["particle_types"]==7),"Goop particle type7")
    checks.equal(row["particles"],particles,"row particle count")
    checks.equal(array_hash(initial),row["initial_observed_state_sha256"],"initial history hash")
    initial_expected=[boundary(frame,bounds) for frame in initial] if np.isfinite(initial).all() else None
    checks.close(row["initial_observed_boundary"],initial_expected,"initial boundary arrays")
    steps=arrays["forecast_steps"]
    checks.require(steps.ndim==1 and np.issubdtype(steps.dtype,np.integer),"saved step type")
    checks.equal(steps.tolist(),[s for s in (1,10,50,200,295) if s<=n],"complete fixed saved-step schedule")
    prediction,truth,histories=arrays["predicted_positions"],arrays["ground_truth_positions"],arrays["observed_or_predicted_histories"]
    checks.require(prediction.shape==truth.shape==(len(steps),particles,3) and histories.shape==(len(steps),6,particles,3),"saved prediction/target/history shape")
    checks.require(all(v.dtype==np.float32 for v in (prediction,truth,histories)),"saved rollout float32 arrays")
    checks.require(np.isfinite(prediction).all() and np.isfinite(truth).all() and np.isfinite(histories).all(),"finite accepted arrays")
    checks.require(np.max(np.abs(prediction),initial=0)<=10,"accepted coordinate guard")
    for side in ("predicted","ground_truth"):
        checks.equal(len(row[side+"_boundary_per_step"]),n,"boundary scalar-series length")
    for key in ("directed_edges_per_step","retained_optional_pairs_per_step","candidate_pairs_per_step","base_pairs_per_step","mean_normalized_acceleration_variance_per_step"):
        checks.equal(len(row[key]),n,"accepted scalar-series length "+key)
    checks.equal(arrays["current_risk"].shape,(len(steps),particles),"saved risk shape")
    checks.equal(arrays["cached_risk_before_forecast"].shape,(len(steps),particles),"saved cached risk shape")
    predicted_by_step={};target_hashes={}
    for index,raw_step in enumerate(steps):
        step=int(raw_step)
        error=float(np.mean((prediction[index].astype(np.float64)-truth[index].astype(np.float64))**2))
        checks.close(mse[step-1],error,"array-recomputed MSE forecast "+str(step))
        checks.close(row["ground_truth_boundary_per_step"][step-1],boundary(truth[index],bounds),"truth boundary forecast "+str(step))
        target_hashes[str(step)]=array_hash(truth[index])
        for k,points in [(step,prediction[index])]+[(step+j-6,histories[index,j]) for j in range(6)]:
            if k<=0:
                checks.array(points,initial[k+5],"saved history observed overlap")
                continue
            hashed=array_hash(points)
            if k in predicted_by_step: checks.equal(predicted_by_step[k],hashed,"saved predicted-history overlap")
            predicted_by_step[k]=hashed
            checks.close(row["predicted_boundary_per_step"][k-1],boundary(points,bounds),"prediction boundary forecast "+str(k))
            if k==n: checks.equal(row["final_predicted_state_sha256"],hashed,"final saved prediction identity")
        edges=arrays[f"edges_forecast_{step:04d}"];optional=arrays[f"optional_pairs_forecast_{step:04d}"]
        attempt=row["attempts"][step-1]
        checks.equal(attempt["predicted_state_sha256"],array_hash(prediction[index]),"saved accepted prediction hash")
        graph_arrays(edges,optional,attempt,particles,checks)
        checks.equal(row["directed_edges_per_step"][step-1],edges.shape[1],"saved directed edge count")
        checks.equal(row["retained_optional_pairs_per_step"][step-1],len(optional),"saved optional pair count")
        risk=arrays["current_risk"][index]
        checks.require(risk.shape==(particles,) and np.isfinite(risk).all() and np.all(risk>0),"accepted saved risk")
        checks.close(row["mean_normalized_acceleration_variance_per_step"][step-1],float(risk.mean()),"saved mean risk")
        cache=arrays["cached_risk_before_forecast"][index]
        if step==1 and row["policy"]!="laggedrisk25":
            checks.require(np.isnan(cache).all(),"initial non-lagged cache is absent")
        else:
            checks.require(np.isfinite(cache).all() and np.all(cache>0),"saved cached risk guard")
    if n==0: checks.equal(row["final_predicted_state_sha256"],array_hash(initial[-1]),"zero-prefix final state")
    checks.close(row["mean_directed_edges"],mean(row["directed_edges_per_step"]) if complete else None,"mean edge scalar series")
    checks.equal(len(row["attempts"]),n+(0 if complete or row["failure"]["forecast_step"]==0 else 1),"accepted plus rejected attempt count")
    metrics={"mean_rollout_mse":mean(mse) if complete else None,"mse_forecast200":mse[199] if n>=200 else None,"mse_forecast295":mse[-1] if complete else None}
    prefix_boundary={}
    for side in ("predicted","ground_truth"):
        series=row[side+"_boundary_per_step"]
        for item in series:
            checks.require(0<=item[BOUNDARY_KEYS[1]]<=item[BOUNDARY_KEYS[0]]<=1,"boundary fraction ordering")
            checks.require(item["mean_particle_maximum_excursion"]<=item["maximum_coordinate_excursion"]+1e-15,"boundary mean/max ordering")
            lo,hi=item["coordinate_minimum"],item["coordinate_maximum"]
            checks.require(isinstance(lo,list) and isinstance(hi,list) and len(lo)==len(hi)==3 and
                           all(finite(a) and finite(b) and a<=b for a,b in zip(lo,hi)),"boundary coordinate ranges")
        for key in BOUNDARY_KEYS:
            checks.require(all(finite(item[key]) and item[key]>=0 for item in series),"finite boundary scalar series")
            value=mean([item[key] for item in series]);name=side+"_boundary_mean_"+key
            metrics[name]=value if complete else None
            if not complete: prefix_boundary[name]=value
        name=side+"_boundary_trajectory_maximum_excursion"
        value=max((item["maximum_coordinate_excursion"] for item in series),default=None)
        metrics[name]=value if complete else None
        if not complete: prefix_boundary[name]=value
    for key, attempt_key in (('directed_edges_per_step','directed_edges'),('retained_optional_pairs_per_step','retained_optional_pairs'),('candidate_pairs_per_step','candidate_pairs'),('base_pairs_per_step','geometric_base_pairs')):
        checks.equal(row[key],[a[attempt_key] for a in row['attempts'][:n]],'full accepted graph series '+key)
    checks.require(all(finite(x) and x>0 for x in row['mean_normalized_acceleration_variance_per_step']),'finite positive accepted scalar risks')
    for i,attempt in enumerate(row["attempts"][:n]):
        checks.equal(attempt["forecast_step"],i+1,"accepted attempt order")
        checks.equal(attempt["accepted"],True,"accepted attempt flag")
        checks.close(attempt["coordinate_mse"],mse[i],"attempt/scalar MSE")
        checks.close(attempt["predicted_boundary"],row["predicted_boundary_per_step"][i],"attempt prediction boundary")
        checks.close(attempt["ground_truth_boundary"],row["ground_truth_boundary_per_step"][i],"attempt truth boundary")
    graphs=[graph_scalar(a,row["policy"],particles,checks) for a in row["attempts"][:n]]
    for key in G_KEYS:
        value=mean([g[key] for g in graphs]); metrics["graph_mean_"+key]=value if complete else None
        if not complete: prefix_boundary["graph_mean_"+key]=value
    metrics["graph_fraction_steps_cap_active"]=mean([int(g["native_base_edges_removed_by_cap"]>0) for g in graphs]) if complete else None
    guard_info=audit_rollout_guards(row,arrays,particles,bounds,checks)
    return metrics,{"completed_steps":n,"accepted_prefix_boundary":prefix_boundary if not complete else None,
                    "array_recomputed_mse_forecasts":steps.tolist(),"array_recomputed_prediction_boundary_forecasts":sorted(predicted_by_step),
                    "full_and_prefix_MSE_verification":"aggregate arithmetic over complete recorded scalar series; unsaved prediction errors not array-recomputed",
                    "initial_history_sha256":array_hash(initial),"truth_hashes":target_hashes,**guard_info}


def audit_rollout_guards(row, arrays, n, bounds, checks):
    import goop3d_saved_diagnostic_audit_v1 as diagnostic
    checked=[]
    parity=row["native_parity"]
    if "finite" in parity:
        diagnostic._parity(parity,arrays,"initial_parity_",arrays["initial_parity_supplied_edges"],n,checks)
        checked.append("initial native/supplied parity arithmetic and saved feature identity")
    else:
        checks.equal(parity["passed"],False,"unattempted parity cannot pass")
    if row["completed_steps"]:
        checks.equal(parity["passed"],True,"accepted forecast follows parity")
    for prefix in ("warmup_","rejected_"):
        present=diagnostic._outputs(arrays,prefix,n,checks)
        if not present: continue
        checks.equal(set(present),{"prediction","risk","raw_risk"},prefix+" complete retained output triplet")
        pred,risk,raw=(present[k] for k in ("prediction","risk","raw_risk"))
        category=("nonfinite_prediction" if not np.isfinite(pred).all() else
                  "coordinate_resource_guard" if np.max(np.abs(pred))>10 else
                  "nonfinite_risk" if not(np.isfinite(raw).all() and np.isfinite(risk).all()) else
                  "nonpositive_risk" if np.any(risk<=0) else None)
        if prefix=="rejected_":
            checks.require(category is not None,"rejected output violates a numeric guard")
            checks.equal(row["failure"]["category"],category,"rejected output guard category")
            if np.isfinite(pred).all():
                checks.close(row["attempts"][-1]["predicted_boundary"],boundary(pred,bounds),"rejected prediction boundary")
        elif category is not None:
            checks.equal(row["failure"]["phase"],"warmup","warmup numeric failure phase")
            checks.equal(row["failure"]["category"],category,"warmup numeric guard category")
        checked.append(prefix+"output variance/guard arithmetic")
    if row["status"]=="failed" and row["failure"]["forecast_step"]>0:
        rejected=row["attempts"][-1]
        checks.equal(rejected["forecast_step"],row["completed_steps"]+1,"rejected forecast position")
        checks.equal(rejected["accepted"],False,"rejected attempt flag")
        checks.equal(rejected["failure"],row["failure"],"rejected attempt failure identity")
    return {"saved_guard_checks":checked,"unsupported_guard_scope":"Missing state/graph evidence and generic execution exceptions are preserved, not reproduced."}


def load_json(path, expected):
    # Hash streams avoid retaining an additional full byte copy of large collections.
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=expected:
            raise ValueError("JSON byte binding differs: "+str(path))
        stream.seek(0)
        value=json.load(io.TextIOWrapper(stream,encoding="utf-8"))
    if file_hash(path)!=expected: raise ValueError("JSON changed during parse: "+str(path))
    return value


def graph_scalar(graph,policy,particles,checks):
    checks.require(all(type(graph.get(k)) is int and graph[k]>=0 for k in G_KEYS),'nonnegative integer graph scalar grid')
    base,extra,native=(graph[k] for k in ('geometric_base_pairs','available_annulus_pairs','native_base_directed_edges'))
    budget=0 if policy=='base' else extra if policy=='dense' else extra//4
    checks.require(graph['candidate_pairs']==base+extra and graph['optional_pair_budget']==extra//4 and graph['retained_optional_pairs']==budget and graph['directed_edges']==native+2*budget,'scalar graph budget relationships')
    checks.require(graph.get('native_base_prefix_preserved') is True and graph['native_base_max_receiver_degree']<=128 and graph['native_base_self_edges']<=particles and native<=128*particles and graph['native_base_receivers_above_cap_before_capping']<=particles and graph['native_base_edges_removed_by_cap']==2*base+particles-native and graph['native_base_asymmetric_directed_edges']<=native-graph['native_base_self_edges'],'scalar native cap and prefix relationships')
    return {k:graph[k] for k in G_KEYS}


def unit(row,mode):
    return row['source_index'],row['policy' if mode=='full-rollout' else 'target_frame']


def schedules_from_collection(collection,checks):
    """Check actual-N source-order arithmetic from the root-bound collection.

    The selected grid includes the original last index. Its maximum identifies
    N. This checks the frozen formula; source-file truth is inherited from the
    separately root-admitted collection and is not independently reread here.
    """
    schedules=collection['source_schedules'];checks.equal(set(schedules),{'valid','test'},'both source schedule splits')
    result={};populations={}
    for split,modes in schedules.items():
        checks.equal(set(modes),{'full-rollout','same-state','clean-validation'},'complete schedule modes')
        full=modes['full-rollout'];checks.require(isinstance(full,list) and full,'nonempty fixed source grid')
        indices=[r['source_index'] for r in full];checks.require(all(type(i) is int and i>=0 for i in indices),'integer source grid')
        count=max(indices)+1;wanted=list(range(count)) if count<=30 else [j*(count-1)//29 for j in range(30)]
        checks.equal(indices,wanted,'actual-N fixed source-order grid')
        checks.require(all(type(r['particles']) is int and r['particles']>0 and isinstance(r['trajectory_id'],str) and r['size_group']=='fixed_source_grid' for r in full),'full source metadata')
        lookup={r['source_index']:r for r in full}
        for mode in ('same-state','clean-validation'):
            pairs=[(i,t) for i in indices for t in (7,80,153,226,300)] if mode=='same-state' else [(indices[n//295],n%295+6) for n in [j*(len(indices)*295-1)//127 for j in range(128)]]
            expected=[{'schedule_index':j,'source_index':i,'target_frame':t,'trajectory_id':lookup[i]['trajectory_id']} for j,(i,t) in enumerate(pairs)]
            checks.equal(modes[mode],expected,'exact D3 '+split+' '+mode+' schedule')
            checks.equal(len(set(pairs)),len(pairs),'distinct diagnostic cells')
        populations[split]=count
        result[split]=modes
    return result,populations


def expected_cells(schedule,mode):
    return [{**r,**({'policy':p} if p else {})} for r in schedule for p in (POLICIES if mode=='full-rollout' else (None,))]


def extra_diagnostic_metrics(row,arrays,bounds,checks):
    flat={}
    particles=len(arrays['particle_types'])
    for method in diagnostic.TIMING_CASES:
        record=row.get('policies',{}).get(method,{})
        if record.get('status')!='complete':continue
        calls=[c for c in row['timed_calls'] if c['method']==method and c['status']=='complete']
        checks.require(bool(calls),'complete diagnostic reference call')
        call=calls[0];prefix=f"r{call['round']+1}_{method}__"
        checks.equal(record['graph'],call['graph'],'policy graph is first complete call')
        edges,optional=arrays[prefix+'edges'],arrays[prefix+'selected_optional_pairs']
        graph_arrays(edges,optional,record['graph'],particles,checks,base_pairs=arrays.get(prefix+'base_pairs'),annulus=arrays.get(prefix+'annulus_pairs'))
        saved_action(method,arrays['current_history'],edges,optional,arrays.get(prefix+'annulus_pairs'),row['random_seed_material'],arrays.get(prefix+'previous_base_risk'),arrays.get(prefix+'physical_selection_scores'),checks)
        for k in BOUNDARY_KEYS:flat['boundary/'+method+'/'+k]=record['prediction_boundary'][k]
        if method!='natural_base_reference':
            graph_scalar(record['graph'],'laggedrisk25' if method==diagnostic.POLICIES[-1] else method,particles,checks)
            for k in G_KEYS:flat['graph/'+method+'/'+k]=record['graph'][k]
    if 'truth_boundary' in row:
        checks.close(row['truth_boundary'],boundary(arrays['target_position'],bounds),'saved target boundary arithmetic')
        for k in BOUNDARY_KEYS:flat['truth_boundary/'+k]=row['truth_boundary'][k]
    return flat


def diagnostic_keys(mode):
    keys=list(diagnostic.expected_metric_keys(mode))
    if mode=='same-state':
        keys += ['boundary/'+p+'/'+k for p in diagnostic.TIMING_CASES for k in BOUNDARY_KEYS]
        keys += ['graph/'+p+'/'+k for p in diagnostic.TIMING_CASES if p!='natural_base_reference' for k in G_KEYS]
        keys += ['truth_boundary/'+k for k in BOUNDARY_KEYS]
    return keys


def audit_collection(collection_path,collection_sha,queue_root):
    checks=Checks();collection=load_json(collection_path,collection_sha)
    checks.equal(collection['schema'],COLLECTION_SCHEMA,'D3 collection schema')
    checks.equal(collection['status'],'stopped_outputs_collected','stopped scalar collection status')
    checks.equal(collection['collector_sha256'],COLLECTOR_SHA,'frozen D3 collector')
    checks.equal(collection['endpoint_updates'],25000,'original selected D3 endpoint')
    ledger=collection['ledger'];checks.equal(ledger['schema'],'adaptgns_goop3d_final_evaluation_ledger_v1','D3 final ledger')
    checks.equal(ledger['state'],'stopped_all_owned_processes_reaped','stopped owned processes')
    checks.equal(ledger['unreaped_owned_children'],[],'no unreaped owned children')
    checks.equal(ledger['all_pinned_inputs_reverified'],True,'stopped input reverification')
    checks.equal(ledger['protocol_sha256'],PROTOCOL_SHA,'original D3 protocol')
    checks.equal(ledger['root_release_sha256'],'4ecf5617b903f7ae0fa662152c8730fdd68b64ffb717e265d6b4a5ca5809b145','original final evaluation release')
    checks.equal(ledger['cohort_sha256'],collection['cohort_sha256'],'same frozen cohort')
    checks.equal(collection['cohort_sha256'],'645343fc2a1c6ef0a82e212e351b03b1a4d081702d3c8e2187b244a62c6b9bef','original completed D3 cohort')
    checks.equal(collection['source_manifest_sha256'],ledger['source_manifest_sha256'],'same source manifests')
    checks.equal(collection['source_manifest_sha256'],{'valid':'f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef','test':'08edf3282267c05a427fc264a1a3aa547e062bc9b5ccb8ab7f982cfb3a9fb7cb'},'original frozen source manifests')
    ref=collection['physical_reference']
    checks.equal((ref['dataset'],ref['frames'],ref['horizon'],ref['dimension'],ref['particle_type'],ref['metadata_sha256']),('Goop-3D',301,295,3,7,METADATA_SHA),'exact D3 physical metadata identity')
    bounds=np.asarray(ref['metadata']['bounds'],dtype=np.float64)
    checks.require(bounds.shape==(3,2) and np.isfinite(bounds).all() and np.all(bounds[:,1]>bounds[:,0]),'valid D3 bounds')
    schedules,populations=schedules_from_collection(collection,checks)
    expected_grid={(a,s,n) for a,s in MODELS for n,_,_ in STAGES}
    stages=collection['stages'];lentries=ledger['stages']
    checks.equal({(s['arm'],s['seed'],s['stage']) for s in stages},expected_grid,'complete six-model five-stage grid')
    checks.equal(len(stages),30,'exactly30 collected stages')
    checks.equal({(s['arm'],s['seed'],s['stage']) for s in lentries},expected_grid,'complete original ledger grid')
    checks.equal(len(lentries),30,'exactly30 ledger stages')
    originals={};queue=Path(queue_root).resolve();checks.require(queue.is_dir() and not Path(queue_root).is_symlink(),'ordinary retained queue root')
    for s in lentries:
        original=Path(s['directory']);checks.require(original.is_absolute() and '..' not in original.parts,'original absolute stage directory')
        # Frozen producer uses queue/jobs/model/stage; relocation changes only queue prefix.
        suffix=Path('jobs')/f"{s['arm']}_seed{s['seed']}"/s['stage']
        checks.equal(tuple(original.parts[-3:]),tuple(suffix.parts),'fixed original stage path suffix')
        originals[original.parents[2]]=True
    checks.equal(len(originals),1,'one original D3 host queue')
    original=next(iter(originals))
    files=collection['files_sha256'];reads={};source_hashes={str(Path(__file__).resolve()):file_hash(__file__),str(Path(diagnostic.__file__).resolve()):file_hash(diagnostic.__file__)}
    source_paths={HERE/'summarize_goop3d_graph_support_v1.py':COLLECTOR_SHA,**{HERE/n:h for n,h in SOURCE_PINS.items()}}
    for p,h in source_paths.items():checks.equal(file_hash(p),h,'frozen producer source bytes');source_hashes[str(p)]=h
    def relocated(p):
        p=Path(p);checks.require(p.is_absolute() and p.is_relative_to(original) and '..' not in p.parts,'retained path remains below original queue')
        result=queue/p.relative_to(original)
        checks.require(all(not q.is_symlink() for q in [result,*result.parents] if q==queue or queue in q.parents),'no retained input symlinks')
        return result
    def bound(p):
        key=str(p);checks.require(key in files,'file in complete collection inventory')
        path=relocated(p);raw=path.read_bytes();entry=files[key]
        checks.equal(len(raw),entry['bytes'],'retained file byte length')
        checks.equal(digest(raw),entry['sha256'],'retained file SHA256')
        reads[str(path)]=entry['sha256'];return raw
    def recheck_layout():
        expected_files=set()
        for directory,state in collection['output_tree_state'].items():
            d=relocated(directory);checks.equal(d.exists(),state['exists'],'stopped stage existence')
            entries=[]
            if d.exists():
                checks.require(d.is_dir(),'ordinary stopped stage directory')
                for p in sorted(d.rglob('*')):
                    checks.require(not p.is_symlink() and (p.is_file() or p.is_dir()),'ordinary complete stopped tree')
                    entries.append([str(p.relative_to(d)),'file' if p.is_file() else 'directory'])
                    if p.is_file():expected_files.add(str(Path(directory)/p.relative_to(d)))
            checks.equal(entries,state['entries'],'exact stopped stage entry set')
        checks.equal(set(collection['output_tree_state']),{s['directory'] for s in lentries},'all30 stopped tree states')
        checks.equal(expected_files,set(files),'complete stopped file inventory')
    recheck_layout()
    # Retain/hash every opaque leftover as well as committed rows; only declared NPZ are decoded.
    for p in files:bound(p)
    models=[];details=[];shared={}
    for stage in stages:
        arm,seed,name=stage['arm'],stage['seed'],stage['stage'];mode,split=next((m,s) for n,m,s in STAGES if n==name)
        checks.context=f'{arm}/seed{seed}/{name}'
        entry=next(s for s in lentries if (s['arm'],s['seed'],s['stage'])==(arm,seed,name))
        directory=Path(entry['directory']);wanted=expected_cells(schedules[split][mode],mode);cells=stage['cells']
        checks.equal((stage['mode'],stage['split']),(mode,split),'stage mode/split')
        checks.equal(stage['cells'],entry['cells'],'original ledger/collection cell identity')
        checks.equal(stage['outcome'],entry['outcome'],'original invocation outcome identity')
        checks.equal(len(cells),len(wanted),'complete declared cell denominator')
        checks.require(all(all(c.get(k)==v for k,v in w.items()) and c.get('state') in CELL_STATES for c,w in zip(cells,wanted)),'fixed cell identities and all outcome states')
        rows={unit(r,mode):r for r in stage['rows']};checks.equal(len(rows),len(stage['rows']),'unique committed rows')
        checks.equal(set(rows),{unit(c,mode) for c in cells if c['state'] in ('completed_required_outcome','recorded_failed_outcome')},'committed versus missing exact grid')
        protocol_path=directory/'protocol.json';protocol=None
        if str(protocol_path) in files:
            protocol=json.loads(bound(protocol_path));checks.equal((protocol['schema'],protocol['purpose'],protocol['mode'],protocol['split'],protocol['arm'],protocol['seed'],protocol['horizon'],protocol['frames'],protocol['checkpoint_updates']),(EVALUATION_SCHEMA,'final_evaluation',mode,split,arm,seed,295,301,25000),'exact D3 saved protocol')
            checks.equal(protocol['schedule'],schedules[split][mode],'protocol/root-bound collection schedule')
            checks.equal(protocol['source_population_count'],populations[split],'actual-N protocol population')
            checks.equal(protocol['prospective_source_grid'],[r['source_index'] for r in schedules[split]['full-rollout']],'prospective full source grid')
            checks.equal(protocol['checkpoint_sha256'],entry['checkpoint_sha256'],'same original model checkpoint')
        if rows:checks.require(protocol is not None,'committed row requires saved protocol')
        checks.equal({Path(p).name for p in files if Path(p).parent==directory and Path(p).match('trajectory_*.json')},{c['row_file'] for c in cells if c['state'] in ('completed_required_outcome','recorded_failed_outcome')},'no unaccounted committed row file')
        if not entry['outcome']['started']:
            checks.require(protocol is None and all(c['state']=='never_started' for c in cells),'never-started stage contains no execution evidence')
        values={};policy_completion={}
        for cell in cells:
            ident=unit(cell,mode)
            if ident not in rows:
                checks.require(isinstance(cell.get('reason'),str) and bool(cell['reason']),'missing/timeout/unexecuted reason retained')
                continue
            row=rows[ident];wanted_cell=next(w for w in wanted if unit(w,mode)==ident)
            checks.require(all(row.get(k)==v for k,v in wanted_cell.items()),'committed row fixed source/schedule identity')
            checks.require(entry['outcome']['started'] is True,'committed row invocation started')
            checks.context=f'{arm}/seed{seed}/{name}/{ident}'
            expected_name=f'trajectory_{ident[0]:06d}_'+(str(ident[1]) if mode=='full-rollout' else f'target_{ident[1]:03d}')+'.json'
            checks.equal(cell['row_file'],expected_name,'exact D3 row basename')
            rowraw=bound(directory/expected_name);checks.equal(digest(rowraw),cell['row_sha256'],'row cell hash');checks.equal(json.loads(rowraw),row,'committed scalar row bytes')
            checks.equal((row['arm'],row['training_seed'],row['objective']),(arm,seed,'faithful'),'row model identity')
            checks.equal(row['status'],'complete' if cell['state']=='completed_required_outcome' else 'failed','row cell status');checks.equal(row.get('failure'),cell.get('failure'),'exact guard or execution failure')
            checks.equal(row['protocol_sha256'],files[str(protocol_path)]['sha256'],'row protocol byte binding')
            artifact=row['artifact_file'];checks.equal(artifact,expected_name[:-5]+'.npz','D3 artifact sibling name')
            raw=bound(directory/artifact);checks.equal(digest(raw),row['artifact_sha256'],'artifact row hash');checks.equal(row['artifact_sha256'],cell['artifact_sha256'],'artifact cell hash')
            with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
                checks.equal(len(archive.files),len(set(archive.files)),'unique numeric archive members');arrays={k:archive[k] for k in archive.files}
            checks.require(all(isinstance(v,np.ndarray) and not v.dtype.hasobject for v in arrays.values()),'numeric-only saved archive')
            source_item=next(r for r in schedules[split]['full-rollout'] if r['source_index']==ident[0])
            checks.equal(arrays['particle_types'].shape,(source_item['particles'],),'saved particle count matches root-bound source grid')
            if mode=='full-rollout':
                metrics,info=audit_rollout(row,arrays,bounds,checks)
                pairings={f'{split}/initial/{ident[0]}':info['initial_history_sha256']}
                pairings.update({f'{split}/truth_float32/{ident[0]}/{int(step)+5}':h for step,h in info['truth_hashes'].items()})
            else:
                metrics=diagnostic.audit_row(row,arrays,bounds,checks,mode=mode)
                if mode=='clean-validation' and row['status']=='complete':
                    graph_scalar(row['graph'],'base',len(arrays['particle_types']),checks)
                    graph_arrays(arrays['native_edges'],np.empty((0,2),dtype=np.int64),row['graph'],len(arrays['particle_types']),checks)
                if mode=='same-state':
                    checks.equal(row['random_seed_material'],[20261006,93000,seed,0 if split=='valid' else 1,ident[0],ident[1]],'fixed diagnostic random seed material')
                    metrics.update(extra_diagnostic_metrics(row,arrays,bounds,checks))
                info={'array_recomputed_diagnostic_metrics':sorted(metrics)}
                if mode=='same-state':policy_completion[str(ident)]=[p for p,r in row.get('policies',{}).items() if r.get('status')=='complete']
                pairings=diagnostic_shared_hashes(arrays,split,ident,mode,checks)
                if mode=='same-state':
                    for method in ('base','dense','random25','speed25','relative-velocity-RMS25','natural_base_reference'):
                        for kind in ('edges','selected_optional_pairs'):
                            key=f'r1_{method}__{kind}'
                            if key in arrays:
                                label=f'{split}/graph/{method}/{kind}/{ident}'+(f'/seed{seed}' if method=='random25' else '')
                                pairings[label]=array_hash(arrays[key])
            for k,h in pairings.items():
                if k in shared:checks.equal(shared[k],h,'same-source/model-independent saved-state identity')
                shared[k]=h
            values[ident]=metrics;details.append({'arm':arm,'seed':seed,'stage':name,'unit':list(ident),'status':row['status'],'failure':row.get('failure'),**info})
        stage_means={};aggregates={}
        if mode=='full-rollout':
            for metric in FULL_METRICS:
                for policy in POLICIES:stage_means[metric+'/'+policy]=mean([values.get((r['source_index'],policy),{}).get(metric) for r in schedules[split][mode]])
        else:
            expected=[unit(r,mode) for r in schedules[split][mode]]
            for metric in diagnostic_keys(mode):
                computed=aggregate(expected,{k:v.get(metric) for k,v in values.items()});aggregates[metric]=computed;stage_means[metric]=computed['equal_trajectory_mean']
                if metric in diagnostic.expected_metric_keys(mode):
                    saved=tree_leaf(stage['diagnostic_summary'],metric)
                    checks.close({k:saved[k] for k in computed},computed,'independent complete diagnostic hierarchy '+metric)
        models.append({'arm':arm,'seed':seed,'stage':name,'mode':mode,'split':split,'metrics':stage_means,'aggregates':aggregates,'cells':cells,'coverage':dict(Counter(c['state'] for c in cells)),'failure_categories':dict(Counter(r['failure']['category'] for r in stage['rows'] if r['status']=='failed')),'policy_completion':policy_completion})
    checks.context='final stopped-byte checks';recheck_layout();checks.equal(file_hash(collection_path),collection_sha,'collection unchanged')
    for p,h in {**reads,**source_hashes}.items():checks.equal(file_hash(p),h,'input/source bytes unchanged')
    return {'schema':'goop3d_saved_array_audit_v1','audit_revision':1,'status':'passed_supported_checks','dataset':'Goop-3D','dimension':3,'horizon':295,'endpoint_updates':25000,'collection_sha256':collection_sha,'cohort_sha256':collection['cohort_sha256'],'physical_reference':ref,'source_manifest_sha256':collection['source_manifest_sha256'],'checks':checks.count,'source_population_counts':populations,'source_schedules':schedules,'models':models,'row_checks':details,'shared_saved_state_hashes':shared,'verified_input_sha256':reads,'audit_source_sha256':source_hashes,'auditor_sha256':source_hashes[str(Path(__file__).resolve())],'diagnostic_helper_sha256':source_hashes[str(Path(diagnostic.__file__).resolve())],'scope':{'array_recomputed':'Only retained forecast1/10/50/200/295 and saved overlapping histories, diagnostic outputs/residuals/benefits/variance/parity arithmetic and available saved graph actions.','scalar_recomputed':'Recorded full/prefix MSE, boundary and graph scalar-series aggregates and fixed diagnostic frame/trajectory hierarchy.','representation':'Same-state target float64 and clean/rollout source target float32 retain separate raw hashes; equality requires a lossless float32 roundtrip.','unsupported':['Unsaved full-trajectory positions/errors or full temporal replay.','Independent official source-position or model normalization truth.','Model/checkpoint/optimizer execution, fresh parity inference, spatial candidate completeness or radius search.','Unsaved graph action reconstruction, runtime clock truth, hardware isolation or causal speedup.'],'all_missing_failed_timeout_unexecuted_cells_retained':True,'survivor_means_computed':False}}


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for name in ('collection','queue-root','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--collection-sha256');a=p.parse_args()
    if not a.execute:print(json.dumps({'status':'description_only','dimension':3,'horizon':295,'models':6,'stages':30}));return 0
    if not all((a.collection,a.collection_sha256,a.queue_root,a.output)):p.error('Exact SHA-bound stopped collection, queue root and fresh output required')
    if a.output.exists() or a.queue_root.resolve()==a.output.resolve() or a.queue_root.resolve() in a.output.resolve().parents:p.error('Fresh output outside retained queue required')
    try:
        result=audit_collection(a.collection,a.collection_sha256,a.queue_root);raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        for path,h in {**result['verified_input_sha256'],**result['audit_source_sha256'],str(a.collection):a.collection_sha256}.items():
            if file_hash(path)!=h:raise ValueError('Input changed before audit publication: '+path)
    except Exception as error:
        with a.output.open('x') as f:json.dump({'schema':'goop3d_saved_array_audit_v1','status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'original_inputs_modified':False},f,indent=2,sort_keys=True)
        raise
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps({'status':result['status'],'checks':result['checks']}));return 0

if __name__=='__main__':raise SystemExit(main())
