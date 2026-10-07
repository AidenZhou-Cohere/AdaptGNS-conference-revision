"""Exact pure arithmetic declarations from the reviewed saved auditor. No original audit entrypoint."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import goop3d_saved_diagnostic_audit_v1 as diagnostic


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
