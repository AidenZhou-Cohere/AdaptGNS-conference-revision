"""Pure byte-array/scalar validation and fixed360-cell action-gate arithmetic.

No filesystem, processes, simulator imports, fitting or new inference. Callers
must supply byte-verified source arrays, heads and exact identity bindings.
"""
from collections import Counter
import hashlib
import math
import numpy as np

SCHEMA='adaptgns_goop_global_action_gate_validated_scalar_row_v1'
SUMMARY_SCHEMA='adaptgns_goop_global_action_gate_paired_scalar_summary_v1'
PROTOCOL_SHA='bc9235fae89d32ede8d1e7f846bff07bdcdd85f4671d5922d688a9535f7fe024'
POLICIES=('base','random25','learned_global_gate','validation_rate_random_gate')
REFERENCES=('base','random25','validation_rate_random_gate')
TRACE=(1,10,50,200,395)
BOUNDARY_KEYS=('fraction_particles_outside','fraction_particles_outside_by_more_than_1e-6',
    'maximum_coordinate_excursion','mean_particle_maximum_excursion')
STATES={'committed_complete','committed_guard_failed','committed_execution_failed',
    'timed_out_current','uncompleted_after_started_invocation','never_started'}
OPERATIONAL={'execution_error','execution_budget','native_parity_failure'}
COST_METRICS=('requested_expansion_fraction','effective_expansion_fraction','zero_budget_fraction',
    'mean_added_pairs','mean_added_directed_edges','mean_directed_edges','trajectory_processing_seconds',
    'publication_seconds','processing_plus_publication_seconds','feature_and_gate_seconds',
    'graph_seconds','forward_seconds')
METRICS=('mean_rollout_mse','mse_forecast200','mse_forecast395')+COST_METRICS+tuple(
    side+'_boundary_'+key for side in ('predicted','ground_truth')
    for key in tuple('mean_'+name for name in BOUNDARY_KEYS)+('trajectory_maximum_excursion',))


def require(ok,message):
    if not ok:raise ValueError(message)


def finite(value,nonnegative=False):
    return type(value) in (int,float) and math.isfinite(value) and (not nonnegative or value>=0)


def digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def value_hash(values):return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def boundary_metrics(position,bounds):
    """Exact arithmetic of frozen full_rollout.boundary_metrics, no torch import."""
    position=np.asarray(position,dtype=np.float64);bounds=np.asarray(bounds,dtype=np.float64)
    excursions=np.maximum(np.maximum(bounds[:,0]-position,position-bounds[:,1]),0.)
    per_particle=excursions.max(axis=-1)
    return {'fraction_particles_outside':float(np.mean(per_particle>0)),
        'fraction_particles_outside_by_more_than_1e-6':float(np.mean(per_particle>1e-6)),
        'maximum_coordinate_excursion':float(excursions.max()),
        'mean_particle_maximum_excursion':float(per_particle.mean()),
        'coordinate_minimum':position.min(axis=0).tolist(),'coordinate_maximum':position.max(axis=0).tolist()}


def array_contract(row,arrays):
    require(isinstance(arrays,dict) and arrays and all(isinstance(k,str) and isinstance(v,np.ndarray)
        and v.dtype.kind in 'fiub' for k,v in arrays.items()),'Numeric-only loaded artifact arrays required')
    expected={k:{'shape':list(v.shape),'dtype':v.dtype.str,'value_sha256':value_hash(v)} for k,v in arrays.items()}
    require(row.get('numeric_arrays')==expected,'Every numeric descriptor/value byte must match')


def graph_contract(graph,requested,n):
    names=('candidate_pairs','geometric_base_pairs','available_annulus_pairs','optional_pair_budget',
        'retained_optional_pairs','directed_edges','native_base_directed_edges','native_base_self_edges',
        'native_base_receivers_above_cap_before_capping','native_base_edges_removed_by_cap',
        'native_base_max_receiver_degree','native_base_asymmetric_directed_edges')
    require(isinstance(graph,dict) and all(type(graph.get(k)) is int and graph[k]>=0 for k in names),
        'Complete nonnegative integer native graph counters required')
    require(graph['candidate_pairs']==graph['geometric_base_pairs']+graph['available_annulus_pairs']
        and graph['optional_pair_budget']==int(.25*graph['available_annulus_pairs'])
        and graph['retained_optional_pairs']==(graph['optional_pair_budget'] if requested else 0)
        and graph['directed_edges']==graph['native_base_directed_edges']+2*graph['retained_optional_pairs']
        and graph['native_base_directed_edges']>=n and graph['native_base_self_edges']<=n
        and 1<=graph['native_base_max_receiver_degree']<=128 and graph.get('native_base_prefix_preserved') is True,
        'Native prefix/exact optional budget counters differ')
    require(all(digest(graph.get(k)) for k in ('native_base_sha256','selected_optional_pair_sha256','directed_edge_sha256')),
        'Native graph byte hashes required')


def trace_graph_contract(arrays,step,graph,n):
    edges=arrays.get(f'edges_forecast_{step:04d}');pairs=arrays.get(f'optional_pairs_forecast_{step:04d}')
    require(isinstance(edges,np.ndarray) and edges.dtype==np.int64 and edges.shape==(2,graph['directed_edges'])
        and isinstance(pairs,np.ndarray) and pairs.dtype==np.int64 and pairs.shape==(graph['retained_optional_pairs'],2)
        and ((edges>=0)&(edges<n)).all() and ((pairs>=0)&(pairs<n)).all(), 'Trace graph shape/type/endpoints differ')
    require((pairs[:,0]<pairs[:,1]).all() and len(set(map(tuple,pairs.tolist())))==len(pairs),
        'Canonical distinct optional trace pairs required')
    base=edges[:,:graph['native_base_directed_edges']];tail=edges[:,graph['native_base_directed_edges']:]
    degree=np.bincount(base[1],minlength=n)
    # With >128 coincident nodes, source-ID ties may cap out a receiver's own
    # loop. Validate the actual capped prefix, never require N surviving loops.
    require(int(np.sum(base[0]==base[1]))==graph['native_base_self_edges'] and degree.min()>=1
        and int(degree.max())==graph['native_base_max_receiver_degree']
        and len(set(map(tuple,base.T.tolist())))==base.shape[1], 'Capped native trace counts/uniqueness differ')
    orientations=set(map(tuple,pairs.tolist()))|set(map(tuple,pairs[:,::-1].tolist()))
    require(set(map(tuple,tail.T.tolist()))==orientations and value_hash(base)==graph['native_base_sha256']
        and value_hash(pairs)==graph['selected_optional_pair_sha256'] and value_hash(edges)==graph['directed_edge_sha256'],
        'Trace native-prefix/optional orientation/value hashes differ')


def validate_row(row,arrays,expected_identity,positions,particle_types,head,core,pure):
    """Validate one committed record; inputs are already byte-bound by wrapper."""
    fields=('model_seed','source_index','policy','checkpoint_sha256','head_sha256','selection_sha256',
        'source_manifest_sha256','source_trajectory_content_sha256','protocol_sha256')
    require(isinstance(expected_identity,dict) and set(expected_identity)==set(fields)
        and all(row.get(k)==expected_identity[k] for k in fields),'Exact model/source/head/protocol identity required')
    seed,source,policy=(row[k] for k in ('model_seed','source_index','policy'))
    require(type(seed) is int and seed in (0,1,2) and type(source) is int and 0<=source<30 and policy in POLICIES
        and all(digest(row[k]) for k in fields if k.endswith('_sha256'))
        and row.get('schema')=='adaptgns_goop_global_action_gate_rollout_row_v1' and row.get('arm')=='mix'
        and row.get('checkpoint_updates')==100000 and row.get('horizon')==395
        and row.get('gate_protocol_sha256')==PROTOCOL_SHA,'Frozen action-gate test row required')
    require(head.get('model_seed')==seed and type(head.get('model_seed')) is int
        and head.get('training_input_hashes',{}).get('checkpoint_sha256')==row['checkpoint_sha256'],
        'Selected head belongs to a different frozen mean model')
    core.predict(head,np.zeros(11,dtype=np.float64))
    probability=head.get('validation_requested_expansion_fraction')
    require(finite(probability) and 0<=probability<=1,'Frozen validation-request rate required')
    completed=row.get('completed_steps');complete=row.get('status')=='complete'
    require(type(completed) is int and 0<=completed<=395 and row.get('status') in ('complete','failed')
        and (completed==395 and row.get('failure') is None if complete else isinstance(row.get('failure'),dict)
            and isinstance(row['failure'].get('category'),str)),'Complete/failed endpoint inconsistency')
    array_contract(row,arrays)
    require(isinstance(positions,np.ndarray) and positions.dtype==np.float32 and positions.ndim==3
        and positions.shape[0]==401 and positions.shape[1]>0 and positions.shape[2]==2 and np.isfinite(positions).all(),
        'Full admitted Goop source position array required')
    n=positions.shape[1]
    require(isinstance(particle_types,np.ndarray) and particle_types.dtype==np.int64 and particle_types.shape==(n,)
        and (particle_types==7).all(),'Admitted source type array required')
    for name,shape,dtype in [('initial_history',(6,n,2),np.float32),('particle_types',(n,),np.int64),
        ('prediction',(completed,n,2),np.float32),('ground_truth',(completed,n,2),np.float32),
        ('feature_trace',(completed if policy=='learned_global_gate' else 0,11),np.float64)]:
        values=arrays.get(name)
        require(isinstance(values,np.ndarray) and values.shape==shape and values.dtype==dtype and np.isfinite(values).all(),
            'Accepted array shape/type/finiteness differs: '+name)
    require(np.array_equal(arrays['initial_history'],positions[:6])
        and np.array_equal(arrays['particle_types'],particle_types)
        and np.array_equal(arrays['ground_truth'],positions[6:6+completed]),'Initial/accepted truth arrays differ from original source')
    predictions,truth=arrays['prediction'],arrays['ground_truth']
    require(not predictions.size or np.max(np.abs(predictions))<=10.,'Accepted prediction violates original coordinate guard')
    mse=[float(np.mean((predictions[i].astype(np.float64)-truth[i].astype(np.float64))**2)) for i in range(completed)]
    require(row.get('mse_per_step')==mse and all(finite(v,True) for v in mse),'Accepted position-MSE curve differs from arrays')
    require(row.get('mean_rollout_mse')==(float(np.mean(mse)) if complete else None)
        and row.get('mse_at_final_horizon')==(mse[-1] if complete else None), 'Full-H395 mean/final error differs or was promoted after failure')
    require(row.get('mse_at_declared_trace_steps')=={str(s):mse[s-1] if completed>=s else None for s in TRACE},
        'Declared pointwise error differs')
    steps=row.get('steps');require(isinstance(steps,list) and len(steps)==completed,'Accepted step ledger length differs')
    history=arrays['initial_history'].copy();predicted_boundary=[];truth_boundary=[]
    for index,step in enumerate(steps,1):
        require(type(step.get('forecast_step')) is int and step['forecast_step']==index
            and step.get('pair_rng_material')==core.rng_material('test',seed,source,index)
            and step.get('gate_rng_material')==(core.rng_material('test',seed,source,index,gate=True)
                if policy=='validation_rate_random_gate' else None), 'Forecast/order/independent RNG material differs')
        predicted=None
        if policy=='learned_global_gate':
            features=core.features(history);predicted=core.predict(head,features);requested=predicted>0
            require(np.array_equal(features,arrays['feature_trace'][index-1]),'Learned gate features differ from autonomous history')
        elif policy=='validation_rate_random_gate':requested=core.independent_gate(probability,'test',seed,source,index)
        else:requested=policy=='random25'
        require(type(step.get('requested_expand')) is bool and step['requested_expand']==bool(requested)
            and step.get('predicted_benefit')==predicted,'Selected gate decision differs from frozen rule')
        graph=step.get('graph');graph_contract(graph,requested,n)
        require(type(step.get('effective_expansion')) is bool and step['effective_expansion']==(graph['retained_optional_pairs']>0)
            and step.get('coordinate_mse')==mse[index-1]
            and all(finite(step.get(k),True) for k in ('feature_and_gate_seconds','graph_seconds','forward_seconds')),
            'Step expansion/error/timing differs')
        pb=boundary_metrics(predictions[index-1],[[.1,.9],[.1,.9]])
        tb=boundary_metrics(truth[index-1],[[.1,.9],[.1,.9]])
        require(step.get('predicted_boundary')==pb and step.get('ground_truth_boundary')==tb,
            'Boundary scalars differ from retained predicted/truth arrays')
        predicted_boundary.append(pb);truth_boundary.append(tb)
        if index in TRACE:trace_graph_contract(arrays,index,graph,n)
        history=np.concatenate((history[1:],predictions[index-1:index]),axis=0)
    requests=sum(s['requested_expand'] for s in steps);effective=sum(s['effective_expansion'] for s in steps)
    zero=sum(s['graph']['optional_pair_budget']==0 for s in steps)
    for key,total in [('requested_expansion_fraction',requests),('effective_expansion_fraction',effective),('zero_budget_fraction',zero)]:
        require(row.get(key)==(float(total/395) if complete else None),'Incomplete or inconsistent realized-action fraction')
    passes=row.get('network_passes');parity_passes=row.get('native_parity_network_passes')
    require(type(passes) is int and completed<=passes<=min(395,completed+1)
        and (passes==395 and parity_passes==2 and row.get('native_parity',{}).get('passed') is True if complete
            else parity_passes is None or type(parity_passes) is int and parity_passes in (0,2)),
        'Deployed/parity forward accounting differs')
    timing=row.get('timing',{})
    require(all(finite(timing.get(k),True) for k in ('trajectory_processing_seconds','publication_seconds'))
        and (timing.get('native_parity_seconds') is None or finite(timing['native_parity_seconds'],True)),
        'Actual whole-row timing required')
    require(not complete or finite(timing.get('native_parity_seconds'),True),'Complete row parity time required')
    realized={'accepted_forecasts':completed,'deployed_forward_attempts':passes,'parity_forward_calls':parity_passes,
        'requested_expansions':requests,'effective_expansions':effective,'zero_budget_forecasts':zero,
        'added_pairs':sum(s['graph']['retained_optional_pairs'] for s in steps),
        'added_directed_edges':2*sum(s['graph']['retained_optional_pairs'] for s in steps),
        'directed_edges':sum(s['graph']['directed_edges'] for s in steps),
        'trajectory_processing_seconds':timing['trajectory_processing_seconds'],'publication_seconds':timing['publication_seconds'],
        'processing_plus_publication_seconds':timing['trajectory_processing_seconds']+timing['publication_seconds'],
        **{k:sum(s[k] for s in steps) for k in ('feature_and_gate_seconds','graph_seconds','forward_seconds')}}
    require(sum(realized[k] for k in ('feature_and_gate_seconds','graph_seconds','forward_seconds'))
        <=timing['trajectory_processing_seconds']+1e-6,'Accepted component times exceed whole trajectory time')
    metrics={'mean_rollout_mse':row['mean_rollout_mse'],'mse_forecast200':mse[199] if completed>=200 else None,
        'mse_forecast395':row['mse_at_final_horizon']}
    costs={'requested_expansion_fraction':requests/395,'effective_expansion_fraction':effective/395,'zero_budget_fraction':zero/395,
        'mean_added_pairs':realized['added_pairs']/395,'mean_added_directed_edges':realized['added_directed_edges']/395,
        'mean_directed_edges':realized['directed_edges']/395,
        **{k:realized[k] for k in COST_METRICS if k.endswith('_seconds')}}
    metrics.update({k:v if complete else None for k,v in costs.items()})
    prefix={}
    for side,values in [('predicted',predicted_boundary),('ground_truth',truth_boundary)]:
        summary=pure.boundary(values,completed)
        for key,value in summary.items():
            name=side+'_boundary_'+key;metrics[name]=value if complete else None
            if not complete:prefix[name]=value
    require(set(metrics)==set(METRICS),'Complete scalar metric schema required')
    return {'schema':SCHEMA,'model_seed':seed,'source_index':source,'policy':policy,'status':row['status'],
        'complete':complete,'completed_steps':completed,'failure':row['failure'],'metrics':metrics,
        'accepted_prefix_boundary':prefix,'mse_per_step':mse,'timing':dict(timing),'realized_cost':realized,
        'all_numeric_descriptors_verified':True,'accepted_errors_boundaries_gates_recomputed':True}


def summarize(stages,pure):
    require(isinstance(stages,list) and len(stages)==3 and all(type(s.get('seed')) is int for s in stages)
        and {s['seed'] for s in stages}=={0,1,2},'All three fixed model stages required')
    rows={};records={};cells={};coverage=[];failed=[];runtime=[]
    expected=[(i,p) for i in range(30) for p in POLICIES]
    for stage in sorted(stages,key=lambda s:s['seed']):
        seed=stage['seed'];entries=stage.get('cells',[])
        require([(c.get('source_index'),c.get('policy')) for c in entries]==expected
            and all(c.get('state') in STATES for c in entries),'Complete ordered120-cell stage coverage required')
        by_identity={}
        for record in stage.get('rows',[]):
            value=record.get('validated',{});source,policy=value.get('source_index'),value.get('policy')
            require(value.get('schema')==SCHEMA and value.get('model_seed')==seed and type(value.get('model_seed')) is int
                and (source,policy) in expected and (source,policy) not in by_identity
                and (record.get('source_index'),record.get('policy'))==(source,policy)
                and set(value.get('metrics',{}))==set(METRICS)
                and value.get('all_numeric_descriptors_verified') is True
                and value.get('accepted_errors_boundaries_gates_recomputed') is True,'Unique validated row required')
            by_identity[source,policy]=value;rows[seed,source,policy]=value
            wall=record.get('case_wall_seconds')
            require(wall is None or finite(wall,True)
                and wall+1e-6>=value['realized_cost']['processing_plus_publication_seconds'],
                'Actual case wall time cannot omit its reported processing/publication components')
            records[seed,source,policy]=record
            if not value['complete']:
                failed.append({'seed':seed,'source_index':source,'policy':policy,'completed_steps':value['completed_steps'],
                    'failure':value['failure'],'accepted_prefix_boundary':value['accepted_prefix_boundary'],
                    'row_file':record.get('row_file'),'row_sha256':record.get('row_sha256'),
                    'artifact_file':record.get('artifact_file'),'artifact_sha256':record.get('artifact_sha256')})
        for cell in entries:
            key=(cell['source_index'],cell['policy']);value=by_identity.get(key);state=cell['state']
            if value is None:require(state in ('timed_out_current','uncompleted_after_started_invocation','never_started'),
                'Committed coverage cell is missing its byte-validated row')
            else:
                category=(value.get('failure') or {}).get('category')
                wanted='committed_complete' if value['complete'] else 'timed_out_current' if category=='execution_budget' else (
                    'committed_execution_failed' if category in OPERATIONAL else 'committed_guard_failed')
                require(state==wanted and cell.get('failure')==value.get('failure'),'Coverage must retain exact row failure category')
            cells[seed,*key]=cell
        coverage.append({'seed':seed,'counts':dict(Counter(c['state'] for c in entries)),
            'guard_or_execution_failure_categories':dict(Counter(c['failure']['category'] for c in entries if c.get('failure')))})
        totals={}
        for policy in POLICIES:
            present=[v for (i,p),v in by_identity.items() if p==policy]
            keys=('accepted_forecasts','deployed_forward_attempts','requested_expansions','effective_expansions','zero_budget_forecasts',
                'added_pairs','added_directed_edges','directed_edges','trajectory_processing_seconds','publication_seconds',
                'processing_plus_publication_seconds','feature_and_gate_seconds','graph_seconds','forward_seconds')
            totals[policy]={'committed_rows':len(present),'complete_rows':sum(v['complete'] for v in present),
                'consumed_committed_work_totals':{k:sum(v['realized_cost'][k] for v in present) for k in keys},
                'parity_forward_calls_known':sum(v['realized_cost']['parity_forward_calls'] or 0 for v in present),
                'rows_with_unknown_parity_call_count':sum(v['realized_cost']['parity_forward_calls'] is None for v in present)}
            observed=[records[seed,v['source_index'],policy].get('case_wall_seconds') for v in present]
            totals[policy].update(case_wall_seconds_observed_total=sum(v for v in observed if v is not None),
                committed_rows_with_case_wall_seconds=sum(v is not None for v in observed),
                committed_rows_missing_case_wall_seconds=sum(v is None for v in observed))
        runtime.append({'seed':seed,'whole_invocation':stage.get('outcome'),'collection_runtime':stage.get('runtime'),
            'per_policy':totals})
    absolute={};contrasts={}
    for metric in METRICS+('end_to_end_case_wall_seconds',):
        means={}
        for seed in range(3):
            for policy in POLICIES:
                if metric=='end_to_end_case_wall_seconds':
                    values=[records.get((seed,i,policy),{}).get('case_wall_seconds')
                        if rows.get((seed,i,policy),{}).get('complete') is True else None for i in range(30)]
                else:values=[rows.get((seed,i,policy),{}).get('metrics',{}).get(metric) for i in range(30)]
                means[seed,policy]=pure.mean_complete(values)
        absolute[metric]={p:pure.seed_summary([means[s,p] for s in range(3)]) for p in POLICIES}
        contrasts[metric]={'learned_global_gate_minus_'+p:pure.seed_summary([
            pure.contrast(means[s,'learned_global_gate'],means[s,p]) for s in range(3)]) for p in REFERENCES}
    counts=dict(Counter(c['state'] for c in cells.values()));require(sum(counts.values())==360,'All360 outcomes required')
    learned=[rows.get((s,i,'learned_global_gate')) for s in range(3) for i in range(30)]
    learned_complete=all(v is not None and v['complete'] for v in learned)
    return {'schema':SUMMARY_SCHEMA,'protocol_sha256':PROTOCOL_SHA,'required_outcomes':360,'required_forecast_forwards':142200,
        'coverage':counts,'coverage_by_seed':coverage,'all_required_outcomes_recorded':len(rows)==360,
        'all_required_outcomes_complete':counts.get('committed_complete',0)==360,'absolute':absolute,
        'paired_learned_minus_controls':contrasts,'primary_full_H395_contrasts':contrasts['mean_rollout_mse'],
        'failed_accepted_prefixes':failed,'runtime':runtime,
        'always_base_requested_action_collapse':all(v['realized_cost']['requested_expansions']==0 for v in learned) if learned_complete else None,
        'learned_no_effective_expansion':all(v['realized_cost']['effective_expansions']==0 for v in learned) if learned_complete else None,
        'constructive_success_automatically_assigned':False,
        'aggregation':'Coordinate/particle MSE within forecast, all395 forecasts within source,30 sources equally within seed, then three seed values/mean/sampleSD. Required undefined values are never dropped.',
        'endpoint_scope':'H200 is a pointwise secondary endpoint and may survive a later failed prefix; it never substitutes for full-H395 mean error.',
        'runtime_scope':'Whole-invocation parent observations include setup, serialization and interrupted work. End-to-end case wall time comes only from matched collection.case_timings; missing timings remain null. Row processing plus reported artifact-publication timers are partial instrumented components: they omit final rowJSON serialization/write and some descriptor work. Processing includes parity. Fixed order/shared host does not establish causal speedup.',
        'boundary_scope':'Full395 geometric summaries only for complete rows; failed accepted-prefix diagnostics retained separately. No conservation claim.',
        'control_scope':'Random gate matches the frozen validation request rate, not realized test compute. Always-base collapse or only beating always-expanded is not constructive accuracy evidence.'}
