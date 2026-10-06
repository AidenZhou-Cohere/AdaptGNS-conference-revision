#!/usr/bin/env python3
"""Independent stopped Sand saved-array/scalar audit. No model or GPU execution.

Sparse forecast arrays validate only their saved steps. Full/prefix rollout
means are independently recalculated from the recorded scalar series, not from
unsaved positions. audit_sand_paired_arrays_v1.py separately checks a published
paired scalar summary against these per-model audit receipts.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
COLLECTOR_SHA = "1e70f1689a75c6f68eb970048aab7837153c19f70caf30ec19143cf66ab5d272"
SOURCE_PINS = {'summarize_goop_graph_support_quota_v2.py': 'da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c', 'supervise_sand_final_evaluation_scoped_v1.py': 'efebe762ea60b1b711925fb9bb3bc91461a16b1440491b2fa554c99171773829', 'supervise_goop_evaluation_gpu_scoped_v3.py': 'a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21', 'evaluate_sand_graph_support_final.py': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58'}
PROTOCOL_SHA = "e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d"
EVALUATION_SCHEMA = "adaptgns_sand_graph_support_final_evaluation_v1"
POLICIES = ("base", "dense", "random25", "speed25", "laggedrisk25", "relative-velocity-RMS25")
STAGES = ("full_rollout_test", "same_state_valid", "same_state_test", "clean_validation")
MODELS = {"A": (("base", 1), ("mix", 1), ("base", 2), ("mix", 2)),
          "B": (("base", 0), ("mix", 0))}
BOUNDARY_KEYS = ("fraction_particles_outside", "fraction_particles_outside_by_more_than_1e-6",
                 "maximum_coordinate_excursion", "mean_particle_maximum_excursion")
FULL_METRICS = ("mean_rollout_mse", "mse_forecast200", "mse_forecast314") + tuple(
    side+"_boundary_"+key for side in ("predicted", "ground_truth")
    for key in tuple("mean_"+key for key in BOUNDARY_KEYS)+( "trajectory_maximum_excursion",))
CELL_STATES = {"completed_required_outcome", "recorded_failed_outcome", "timed_out_current",
               "not_completed_before_invocation_end", "never_started"}


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


def schedule(stage):
    if stage=="full_rollout_test": return [(i,p) for i in range(30) for p in POLICIES]
    if stage=="clean_validation": return [(n//314,n%314+6) for n in (i*9419//127 for i in range(128))]
    return [(i,t) for i in range(30) for t in (7,85,163,241,319)]


def identity(row,stage): return row["source_index"],row["policy" if stage=="full_rollout_test" else "target_frame"]


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
        checks.equal(target.shape,(len(arrays["particle_types"]),2),"target pairing shape")
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
    checks.require(set(mandatory[0,mandatory[0]==mandatory[1]].tolist())<=set(range(n)),"native capped self edges are a particle subset")
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
        checks.array(degree,np.minimum(uncapped,128),"native receiver degree from saved base universe")
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
    n=row["completed_steps"]; horizon=314
    checks.require(type(n) is int and 0<=n<=horizon,"accepted prefix")
    checks.equal(row["horizon"],horizon,"Sand horizon")
    complete=row["status"]=="complete"
    checks.require(row["status"] in ("complete","failed") and complete==(n==horizon),"status/prefix consistency")
    checks.require((row["failure"] is None)==complete,"failure/status consistency")
    mse=row["mse_per_step"]
    checks.require(len(mse)==n and all(finite(v) and v>=0 for v in mse),"complete accepted scalar MSE series")
    checks.close(row["mean_rollout_mse"],mean(mse) if complete else None,"full-H scalar-series mean")
    checks.close(row["prefix_mean_mse_if_failed"],mean(mse) if not complete else None,"failed-prefix scalar-series mean")
    checks.close(row["mse_at_final_horizon"],mse[-1] if complete else None,"final scalar-series MSE")
    for step in (1,10,50,200,314):
        checks.close(row["mse_at_declared_trace_steps"][str(step)],mse[step-1] if step<=n else None,"declared scalar step "+str(step))
    checks.equal(row["requested_trace_steps"],[1,10,50,200,314],"fixed requested traces")
    checks.array(arrays["bounds"],bounds,"trace/metadata bounds")
    initial=arrays["initial_observed_positions"]
    checks.require(initial.ndim==3,"initial history rank")
    particles=initial.shape[1]
    checks.require(initial.shape==(6,particles,2) and particles>0,"initial history shape")
    checks.equal(initial.dtype.str,np.dtype(np.float32).str,"initial history float32")
    checks.require(arrays["particle_types"].shape==(particles,) and np.all(arrays["particle_types"]==6),"Sand particle type6")
    checks.equal(row["particles"],particles,"row particle count")
    checks.equal(array_hash(initial),row["initial_observed_state_sha256"],"initial history hash")
    initial_expected=[boundary(frame,bounds) for frame in initial] if np.isfinite(initial).all() else None
    checks.close(row["initial_observed_boundary"],initial_expected,"initial boundary arrays")
    steps=arrays["forecast_steps"]
    checks.require(steps.ndim==1 and np.issubdtype(steps.dtype,np.integer),"saved step type")
    checks.equal(steps.tolist(),[s for s in (1,10,50,200,314) if s<=n],"complete fixed saved-step schedule")
    prediction,truth,histories=arrays["predicted_positions"],arrays["ground_truth_positions"],arrays["observed_or_predicted_histories"]
    checks.require(prediction.shape==truth.shape==(len(steps),particles,2) and histories.shape==(len(steps),6,particles,2),"saved prediction/target/history shape")
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
    metrics={"mean_rollout_mse":mean(mse) if complete else None,"mse_forecast200":mse[199] if n>=200 else None,"mse_forecast314":mse[-1] if complete else None}
    prefix_boundary={}
    for side in ("predicted","ground_truth"):
        series=row[side+"_boundary_per_step"]
        for item in series:
            checks.require(0<=item[BOUNDARY_KEYS[1]]<=item[BOUNDARY_KEYS[0]]<=1,"boundary fraction ordering")
            checks.require(item["mean_particle_maximum_excursion"]<=item["maximum_coordinate_excursion"]+1e-15,"boundary mean/max ordering")
            lo,hi=item["coordinate_minimum"],item["coordinate_maximum"]
            checks.require(isinstance(lo,list) and isinstance(hi,list) and len(lo)==len(hi)==2 and
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
    for i,attempt in enumerate(row["attempts"][:n]):
        checks.equal(attempt["forecast_step"],i+1,"accepted attempt order")
        checks.equal(attempt["accepted"],True,"accepted attempt flag")
        checks.close(attempt["coordinate_mse"],mse[i],"attempt/scalar MSE")
        checks.close(attempt["predicted_boundary"],row["predicted_boundary_per_step"][i],"attempt prediction boundary")
        checks.close(attempt["ground_truth_boundary"],row["ground_truth_boundary_per_step"][i],"attempt truth boundary")
    guard_info=audit_rollout_guards(row,arrays,particles,bounds,checks)
    return metrics,{"completed_steps":n,"accepted_prefix_boundary":prefix_boundary if not complete else None,
                    "array_recomputed_mse_forecasts":steps.tolist(),"array_recomputed_prediction_boundary_forecasts":sorted(predicted_by_step),
                    "full_and_prefix_MSE_verification":"aggregate arithmetic over complete recorded scalar series; unsaved prediction errors not array-recomputed",
                    "initial_history_sha256":array_hash(initial),"truth_hashes":target_hashes,**guard_info}


def audit_rollout_guards(row, arrays, n, bounds, checks):
    import sand_saved_diagnostic_audit_v1 as diagnostic
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


def audit_collection(collection_path, collection_sha, queue_root):
    import sand_saved_diagnostic_audit_v1 as diagnostic
    source_paths=(Path(__file__).resolve(),Path(diagnostic.__file__).resolve())
    source_hashes={str(p):digest(p.read_bytes()) for p in source_paths}
    checks=Checks();collection=load_json(collection_path,collection_sha)
    checks.equal(collection["schema"],"adaptgns_sand_evaluation_collection_scoped_v1","collection schema")
    checks.equal(collection["status"],"stopped_outputs_collected","collection status")
    checks.equal(collection["collector_sha256"],COLLECTOR_SHA,"reviewed collector identity")
    checks.equal(collection["source_sha256"],SOURCE_PINS,"frozen source pins")
    checks.equal(collection["protocol_sha256"],PROTOCOL_SHA,"frozen scientific protocol")
    checks.equal(collection["timing_scope"],"shared_host_operational_measurement","timing scope")
    checks.equal(collection["queue_status"]["all_pinned_inputs_reverified"],True,"collector input reverification")
    role=collection["host_role"];checks.require(role in MODELS,"host role")
    queue=Path(queue_root).resolve();original=Path(collection["original_queue_root"])
    files=collection["files"];checks.equal(files,collection["output_tree_state"]["files"],"collection inventory views")
    reads={};shared={};models=[];details=[]
    def bound(relative):
        relative=Path(relative)
        checks.require(not relative.is_absolute() and ".." not in relative.parts,"relative input path")
        path=queue/relative
        checks.require(all(not p.is_symlink() for p in (path,*path.parents) if p==queue or queue in p.parents),"ordinary input path below queue")
        raw=path.read_bytes();expected=files[str(relative)]
        checks.equal(len(raw),expected["bytes"],"input byte length "+str(relative))
        checks.equal(digest(raw),expected["sha256"],"input byte hash "+str(relative))
        reads[str(path)]=expected["sha256"]
        return raw
    wanted={(arm,seed,stage) for arm,seed in MODELS[role] for stage in STAGES}
    checks.equal({(s["arm"],s["seed"],s["stage"]) for s in collection["stages"]},wanted,"all model-stage identities")
    checks.equal(len(collection["stages"]),len(wanted),"no duplicate stages")
    for stage in collection["stages"]:
        arm,seed,name=stage["arm"],stage["seed"],stage["stage"]
        checks.context=f"{arm}/seed{seed}/{name}"
        expected=schedule(name);cells=stage["cells"]
        mode="full-rollout" if name=="full_rollout_test" else "clean-validation" if name=="clean_validation" else "same-state"
        split="test" if name in ("full_rollout_test","same_state_test") else "valid"
        checks.equal((stage["mode"],stage["split"]),(mode,split),"stage mode/split")
        checks.equal([identity(c,name) for c in cells],expected,"complete cell grid")
        checks.require(all(c["state"] in CELL_STATES for c in cells),"declared cell states")
        rows={identity(r,name):r for r in stage["rows"]}
        checks.equal(len(rows),len(stage["rows"]),"unique committed rows")
        checks.equal(set(rows),{identity(c,name) for c in cells if c["state"] in ("completed_required_outcome","recorded_failed_outcome")},"committed/missing distinction")
        directory=Path("jobs")/f"{arm}_seed{seed}"/name
        values={};protocol=None;bounds=None
        if rows:
            protocol_raw=bound(directory/"protocol.json");protocol=json.loads(protocol_raw)
            checks.equal(digest(protocol_raw),stage["protocol_sha256"],"protocol byte binding")
            checks.equal(protocol["model"]["arm"],arm,"protocol arm")
            checks.equal(protocol["model"]["seed"],seed,"protocol seed")
            checks.equal(protocol["model"]["completed_updates"],100000,"protocol endpoint")
            checks.equal(protocol["model"]["objective"],"faithful","Sand objective")
            checks.equal((protocol["schema"],protocol["mode"],protocol["split"],protocol["source_frame_count"]),(EVALUATION_SCHEMA,mode,split,320),"protocol schema/split/source frames")
            planned=([r["source_index"] for r in protocol["schedule"]] if name=="full_rollout_test" else [identity(r,name) for r in protocol["schedule"]])
            checks.equal(planned,list(range(30)) if name=="full_rollout_test" else expected,"protocol schedule")
            # Bounds and physical dt were read from exact SHA-bound metadata cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0.
            # Original source positions are not loaded by this auditor.
            bounds=np.asarray([[.1,.9],[.1,.9]],dtype=float)
        for cell in cells:
            unit=identity(cell,name)
            if unit not in rows: continue
            row=rows[unit];checks.context=f"{arm}/seed{seed}/{name}/{unit}"
            row_path=Path(cell["path"])
            checks.equal(row_path.parent,original/directory,"original row directory")
            raw=bound(directory/row_path.name)
            checks.equal(digest(raw),cell["sha256"],"row cell hash")
            checks.equal(json.loads(raw),row,"row/collection scalar identity")
            checks.equal(row["protocol_sha256"],stage["protocol_sha256"],"row protocol")
            scheduled=next(r for r in protocol["schedule"] if (r["source_index"]==unit[0] if name=="full_rollout_test" else identity(r,name)==unit))
            for key,value in scheduled.items(): checks.equal(row[key],value,"row/scheduled "+key)
            checks.equal(row["status"],"complete" if cell["state"]=="completed_required_outcome" else "failed","cell/row status")
            checks.equal(cell.get("failure"),row.get("failure"),"exact cell/row failure identity")
            artifact_key="trace" if name=="full_rollout_test" else "artifact"
            artifact=row[artifact_key+"_file"]
            checks.equal(Path(artifact).name,artifact,"artifact basename")
            raw=bound(directory/artifact);checks.equal(digest(raw),row[artifact_key+"_sha256"],"artifact row hash")
            with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
                arrays={key:archive[key] for key in archive.files}
            checks.require(all(not v.dtype.hasobject for v in arrays.values()),"numeric-only archive")
            if "numeric_arrays" in row:
                checks.equal(set(arrays),set(row["numeric_arrays"]),"complete numeric array descriptors")
                for key,value in arrays.items():
                    checks.equal(row["numeric_arrays"][key],{"shape":list(value.shape),"dtype":value.dtype.str,"value_sha256":array_hash(value)},"array descriptor "+key)
            if name=="full_rollout_test":
                checks.equal((row["arm"],row["training_seed"],row["objective"]),(arm,seed,"faithful"),"rollout identity")
                metrics,info=audit_rollout(row,arrays,bounds,checks)
                pairings={f"test/initial/{unit[0]}":info["initial_history_sha256"]}
                pairings.update({f"test/truth/{unit[0]}/{step}":h for step,h in info["truth_hashes"].items()})
                pairings.update({f"test/truth_float32/{unit[0]}/{int(step)+5}":h for step,h in info["truth_hashes"].items()})
            else:
                metrics=diagnostic.audit_row(row,arrays,bounds,checks,mode="clean-validation" if name=="clean_validation" else "same-state")
                info={"array_recomputed_diagnostic_metrics":sorted(metrics)}
                split="valid" if name in ("same_state_valid","clean_validation") else "test"
                pairings=diagnostic_shared_hashes(arrays,split,unit,mode,checks)
                if name.startswith("same_state"):
                    material=[20261006,93000,seed,0 if split=="valid" else 1,unit[0],unit[1]]
                    checks.equal(row["random_seed_material"],material,"fixed diagnostic random material")
                    key="r1_random25__selected_optional_pairs"
                    if key in arrays: pairings[f"{split}/random_seed{seed}/{unit}"]=array_hash(arrays[key])
                    for method in ("base","dense","speed25","relative-velocity-RMS25","natural_base_reference"):
                        for kind in ("edges","selected_optional_pairs"):
                            key=f"r1_{method}__{kind}"
                            if key in arrays: pairings[f"{split}/common_graph/{method}/{kind}/{unit}"]=array_hash(arrays[key])
                    for call in row.get("warmup_calls",[])+row.get("timed_calls",[]):
                        prefix=f"r{call['round']+1}_{call['method']}__"
                        if prefix+"edges" in arrays and prefix+"selected_optional_pairs" in arrays:
                            graph_arrays(arrays[prefix+"edges"],arrays[prefix+"selected_optional_pairs"],call.get("graph",{}),len(arrays["particle_types"]),checks,
                                         base_pairs=arrays.get(prefix+"base_pairs"),annulus=arrays.get(prefix+"annulus_pairs"))
                            saved_action(call["method"],arrays["current_history"],arrays[prefix+"edges"],arrays[prefix+"selected_optional_pairs"],
                                         arrays.get(prefix+"annulus_pairs"),material,arrays.get(prefix+"previous_base_risk"),arrays.get(prefix+"physical_selection_scores"),checks)
            for key,value in pairings.items():
                if key in shared: checks.equal(shared[key],value,"paired saved state/target/random identity "+key)
                shared[key]=value
            values[unit]=metrics
            details.append({"arm":arm,"seed":seed,"stage":name,"unit":list(unit),"status":row["status"],"failure":row.get("failure"),**info})
        stage_means={}
        if name=="full_rollout_test":
            for metric in FULL_METRICS:
                for policy in POLICIES:
                    stage_means[metric+"/"+policy]=mean([values.get((source,policy),{}).get(metric) for source in range(30)])
        else:
            for metric in diagnostic.expected_metric_keys("clean-validation" if name=="clean_validation" else "same-state"):
                observed={unit:m.get(metric) for unit,m in values.items()}
                computed=aggregate(expected,observed)
                saved=tree_leaf(stage["diagnostic_summary"],metric)
                checks.close({k:saved[k] for k in computed},computed,"independent diagnostic hierarchy "+metric)
                stage_means[metric]=computed["equal_trajectory_mean"]
        models.append({"arm":arm,"seed":seed,"stage":name,"metrics":stage_means,
                       "cells":cells,"coverage":dict(Counter(c["state"] for c in cells))})
    checks.context="final byte reverification"
    checks.equal(file_hash(collection_path),collection_sha,"collection unchanged")
    for path,expected in reads.items(): checks.equal(file_hash(path),expected,"read input unchanged "+path)
    for path,expected in source_hashes.items(): checks.equal(digest(Path(path).read_bytes()),expected,"audit source unchanged "+path)
    return {"schema":"sand_saved_array_audit_v1","audit_revision":2,"status":"passed_supported_checks","host_role":role,
            "collection_sha256":collection_sha,"cohort_sha256":collection["cohort_sha256"],"checks":checks.count,
            "models":models,"row_checks":details,"shared_saved_state_hashes":shared,"verified_input_sha256":reads,
            "auditor_sha256":source_hashes[str(source_paths[0])],"diagnostic_helper_sha256":source_hashes[str(source_paths[1])],"audit_source_sha256":source_hashes,
            "scope":{"array_recomputed":"All saved forecast errors, available prediction/truth boundary states, and supported saved diagnostic scalar formulas.",
                     "scalar_recomputed":"Full/prefix error and full-H boundary aggregates from complete recorded scalar series; equal-frame/equal-source diagnostic hierarchy.",
                     "unsupported":["Unsaved full-trajectory positions/errors and complete temporal replay.","Official source-position/normalization truth independently reloaded from dataset.","Model/checkpoint/optimizer execution or fresh native parity/model verification.","Completeness of geometric candidate sets, spatial radius search, or unsaved graph actions.","Measured clock truth, hardware isolation, or causal speedup."]}}


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--collection",type=Path);parser.add_argument("--collection-sha256")
    parser.add_argument("--queue-root",type=Path);parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({"status":"description_only","scope":__doc__}));return 0
    if not all((args.collection,args.collection_sha256,args.queue_root,args.output)):
        parser.error("collection, exact collection SHA256, queue-root and fresh output required")
    if args.output.exists() or args.queue_root.resolve() in args.output.resolve().parents:
        parser.error("fresh output outside retained queue required")
    try:
        result=audit_collection(args.collection,args.collection_sha256,args.queue_root)
        encoded=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
        for path,expected in {**result["verified_input_sha256"],**result["audit_source_sha256"],str(args.collection):args.collection_sha256}.items():
            if file_hash(path)!=expected: raise ValueError("Input/source changed before publication: "+path)
    except Exception as error:
        result={"schema":"sand_saved_array_audit_v1","status":"failed","error_type":type(error).__name__,"error":str(error),
                "traceback":traceback.format_exc(),"collection":str(args.collection),"expected_collection_sha256":args.collection_sha256,
                "original_inputs_modified":False}
        with args.output.open("x") as stream: json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        raise
    with args.output.open("xb") as stream: stream.write(encoded)
    print(json.dumps({"status":result["status"],"checks":result["checks"],"host_role":result["host_role"]}))
    return 0


if __name__=="__main__":raise SystemExit(main())
