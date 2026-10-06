"""Synthetic NumPy arrays and complete/incomplete360-cell arithmetic only."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location('_scalar_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


S=load('goop_action_gate_scalar_core_v1')
C=load('goop_global_action_gate_core_v1')
P=load('summarize_goop_graph_support_quota_v2')
D=load('run_goop_action_gate_v1')
T=load('test_run_goop_action_gate_v1')


def numeric_fixture(policy='learned_global_gate',failure_step=None,category='synthetic_guard',seed=0):
    native=T.Native(fail_at=failure_step,error=T.Guard(category))
    original=native.native_graph
    def graph(*args):
        value,edges,pairs,audit=original(*args);n=len(args[0][0]);base=edges[:,:n]
        audit.update(candidate_pairs=4,geometric_base_pairs=0,available_annulus_pairs=4,
            native_base_directed_edges=n,native_base_self_edges=n,native_base_receivers_above_cap_before_capping=0,
            native_base_edges_removed_by_cap=0,native_base_max_receiver_degree=1,native_base_asymmetric_directed_edges=0,
            native_base_sha256=S.value_hash(base),selected_optional_pair_sha256=S.value_hash(pairs),directed_edge_sha256=S.value_hash(edges))
        return value,edges,pairs,audit
    native.native_graph=graph;native.full.boundary_metrics=S.boundary_metrics
    positions=T.Positions(native.events).values;types=np.full(4,7,np.int64)
    head=T.head(prediction=1.,p=.47);head['model_seed']=seed
    row,arrays=D.rollout_row(native,C,None,positions,types,{'bounds':[[.1,.9],[.1,.9]]},
        {'source_index':0,'policy':policy},seed,head,'test','cpu')
    identity={'model_seed':seed,'source_index':0,'policy':policy,'checkpoint_sha256':'a'*64,
        'head_sha256':'b'*64,'selection_sha256':'c'*64,'source_manifest_sha256':'d'*64,
        'source_trajectory_content_sha256':'e'*64,'protocol_sha256':'f'*64}
    row.update(identity,gate_protocol_sha256=S.PROTOCOL_SHA)
    row['timing']['publication_seconds']=.01
    row['numeric_arrays']={k:{'shape':list(v.shape),'dtype':v.dtype.str,'value_sha256':S.value_hash(v)} for k,v in arrays.items()}
    return row,arrays,identity,positions,types,head


def validate(fixture):return S.validate_row(*fixture,C,P)


def descriptors(row,arrays):
    row['numeric_arrays']={k:{'shape':list(v.shape),'dtype':v.dtype.str,'value_sha256':S.value_hash(v)} for k,v in arrays.items()}


@pytest.mark.parametrize('policy',S.POLICIES)
def test_actual_driver_synthetic_full_row_validates(policy):
    row,arrays,identity,positions,types,head=numeric_fixture(policy)
    result=S.validate_row(row,arrays,identity,positions,types,head,C,P)
    assert result['complete'] and result['completed_steps']==395
    assert result['metrics']['mean_rollout_mse']==row['mean_rollout_mse']
    assert result['realized_cost']['deployed_forward_attempts']==395
    assert result['accepted_errors_boundaries_gates_recomputed']
    assert result['metrics']['mean_added_directed_edges']==2*result['metrics']['mean_added_pairs']


@pytest.mark.parametrize('step', [1,100,201,301,395])
def test_failed_prefix_keeps_h200_only_if_reached(step):
    fixture=numeric_fixture(failure_step=step);result=validate(fixture)
    assert not result['complete'] and result['completed_steps']==step-1
    assert result['metrics']['mean_rollout_mse'] is result['metrics']['mse_forecast395'] is None
    assert (result['metrics']['mse_forecast200'] is not None)==(step>200)
    assert all(result['metrics'][k] is None for k in S.COST_METRICS)
    assert result['realized_cost']['deployed_forward_attempts']==step
    assert result['accepted_prefix_boundary']


@pytest.mark.parametrize('change',['parity0','parity_unknown','failure_step','initial_phase','forward_attempt'])
def test_failure_phase_parity_and_attempt_cost_must_match_prefix(change):
    fixture=numeric_fixture(failure_step=201);row=fixture[0]
    if change=='parity0':row['native_parity_network_passes']=0
    elif change=='parity_unknown':row['native_parity_network_passes']=None
    elif change=='failure_step':row['failure']['forecast_step']=0
    elif change=='initial_phase':row['failure'].update(phase='initial_state',forecast_step=0)
    elif change=='forward_attempt':row['network_passes']=200
    with pytest.raises(ValueError):validate(fixture)


def test_timeout_between_current_phase_and_forward_increment_remains_valid():
    fixture=numeric_fixture(failure_step=201);row=fixture[0]
    row['failure']['category']='execution_budget';row['network_passes']=200
    assert validate(fixture)['completed_steps']==200


def test_post_append_scoring_failure_at_395_keeps_full_mean_undefined():
    fixture=numeric_fixture();row=fixture[0]
    row.update(status='failed',failure={'category':'execution_error','phase':'scoring','forecast_step':396},
        mean_rollout_mse=None,mse_at_final_horizon=None,requested_expansion_fraction=None,
        effective_expansion_fraction=None,zero_budget_fraction=None)
    result=validate(fixture)
    assert result['completed_steps']==395 and result['metrics']['mean_rollout_mse'] is None
    assert result['metrics']['mse_forecast200'] is not None


@pytest.mark.parametrize('change', ['scalar_mse','scalar_full','future_truth','history','types','feature','decision','pair_rng',
    'gate_rng','boundary','budget','trace_edge','trace_pair','dtype','descriptor','timing','forward_count','parity','head'])
def test_row_drift_rejected(change):
    fixture=numeric_fixture();row,arrays,identity,positions,types,head=fixture
    if change=='scalar_mse':row['mse_per_step'][0]+=1.
    elif change=='scalar_full':row['mean_rollout_mse']+=1.
    elif change=='future_truth':arrays['ground_truth'][0,0,0]+=.001;descriptors(row,arrays)
    elif change=='history':arrays['initial_history'][0,0,0]+=.001;descriptors(row,arrays)
    elif change=='types':arrays['particle_types'][0]=3;descriptors(row,arrays)
    elif change=='feature':arrays['feature_trace'][2,1]+=.001;descriptors(row,arrays)
    elif change=='decision':row['steps'][0]['requested_expand']=False
    elif change=='pair_rng':row['steps'][0]['pair_rng_material'][-1]=2909
    elif change=='gate_rng':row['steps'][0]['gate_rng_material']=[1]
    elif change=='boundary':row['steps'][0]['predicted_boundary']['maximum_coordinate_excursion']=1.
    elif change=='budget':row['steps'][0]['graph']['optional_pair_budget']=2
    elif change=='trace_edge':arrays['edges_forecast_0001'][0,0]=2;descriptors(row,arrays)
    elif change=='trace_pair':arrays['optional_pairs_forecast_0001'][0]=[0,0];descriptors(row,arrays)
    elif change=='dtype':arrays['prediction']=arrays['prediction'].astype(np.float64);descriptors(row,arrays)
    elif change=='descriptor':row['numeric_arrays']['prediction']['value_sha256']='0'*64
    elif change=='timing':row['timing']['trajectory_processing_seconds']=0.
    elif change=='forward_count':row['network_passes']=396
    elif change=='parity':row['native_parity']['passed']=False
    elif change=='head':head['training_input_hashes']['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError):validate(fixture)


def test_rejected_nonfinite_array_preserved_but_not_scored():
    fixture=numeric_fixture(failure_step=2);row,arrays,*_=fixture
    arrays['rejected_prediction']=np.array([[np.nan,np.inf]],dtype=np.float32);descriptors(row,arrays)
    result=validate(fixture)
    assert result['completed_steps']==1 and result['metrics']['mean_rollout_mse'] is None


def test_nonfinite_accepted_array_is_rejected():
    fixture=numeric_fixture();row,arrays,*_=fixture
    arrays['prediction'][0,0,0]=np.nan;descriptors(row,arrays)
    with pytest.raises(ValueError,match='finiteness'):validate(fixture)


def test_dtype_object_array_rejected_before_value_claim():
    fixture=numeric_fixture();row,arrays,*_=fixture
    arrays['unsafe']=np.array([{'x':1}],dtype=object)
    with pytest.raises(ValueError,match='Numeric-only'):validate(fixture)


def test_random_gate_recomputed_from_frozen_validation_rate():
    fixture=numeric_fixture('validation_rate_random_gate');row,arrays,*_=fixture
    assert validate(fixture)['complete']
    row['steps'][5]['requested_expand']=not row['steps'][5]['requested_expand']
    with pytest.raises(ValueError,match='decision'):validate(fixture)


def test_native_cap_can_remove_self_loops_with_130_coincident_nodes():
    n=130
    # Receiver/distance/source-ID ordering from frozen bridge. Every distance
    # is zero, so each receiver keeps sources0..127, including only128 loops.
    base=np.vstack((np.tile(np.arange(128),n),np.repeat(np.arange(n),128))).astype(np.int64)
    pairs=np.empty((0,2),dtype=np.int64)
    graph={'candidate_pairs':n*(n-1)//2,'geometric_base_pairs':n*(n-1)//2,'available_annulus_pairs':0,
        'optional_pair_budget':0,'retained_optional_pairs':0,'directed_edges':base.shape[1],
        'native_base_directed_edges':base.shape[1],'native_base_self_edges':128,
        'native_base_receivers_above_cap_before_capping':n,'native_base_edges_removed_by_cap':n*n-base.shape[1],
        'native_base_max_receiver_degree':128,'native_base_asymmetric_directed_edges':256,'native_base_prefix_preserved':True,
        'native_base_sha256':S.value_hash(base),'selected_optional_pair_sha256':S.value_hash(pairs),'directed_edge_sha256':S.value_hash(base)}
    S.graph_contract(graph,False,n)
    S.trace_graph_contract({'edges_forecast_0001':base,'optional_pairs_forecast_0001':pairs},1,graph,n)
    graph['native_base_self_edges']=n
    with pytest.raises(ValueError,match='trace counts'):S.trace_graph_contract(
        {'edges_forecast_0001':base,'optional_pairs_forecast_0001':pairs},1,graph,n)


def synthetic_stages():
    stages=[]
    for seed in range(3):
        stage={'seed':seed,'scientific_verification_passed':True,'rows':[],'cells':[],'outcome':{'started':True,'stopped_and_reaped':True,'elapsed_seconds':1000.+seed},
            'runtime':{'setup_seconds':1.}}
        for source in range(30):
            base=10.*(seed+1)+source
            for policy in S.POLICIES:
                error=base+{'base':0.,'random25':2.,'learned_global_gate':-(seed+1.),'validation_rate_random_gate':-.5}[policy]
                metrics={k:0. for k in S.METRICS};metrics.update(mean_rollout_mse=error,mse_forecast200=error+1.,mse_forecast395=error+2.)
                requested=0 if policy in ('base','learned_global_gate') else 395
                realized={'accepted_forecasts':395,'deployed_forward_attempts':395,'parity_forward_calls':2,
                    'requested_expansions':requested,'effective_expansions':requested,'zero_budget_forecasts':0,
                    'added_pairs':requested,'added_directed_edges':requested*2,'directed_edges':395+requested*2,
                    'trajectory_processing_seconds':1.,'publication_seconds':.1,'processing_plus_publication_seconds':1.1,
                    'feature_and_gate_seconds':.1,'graph_seconds':.2,'forward_seconds':.3}
                validated={'schema':S.SCHEMA,'model_seed':seed,'source_index':source,'policy':policy,'status':'complete',
                    'complete':True,'completed_steps':395,'failure':None,'metrics':metrics,'accepted_prefix_boundary':{},
                    'mse_per_step':[error]*395,'timing':{},'realized_cost':realized,
                    'all_numeric_descriptors_verified':True,'accepted_errors_boundaries_gates_recomputed':True}
                stage['rows'].append({'source_index':source,'policy':policy,'validated':validated,
                    'case_wall_seconds':2.,
                    'row_file':f'{seed}_{source}_{policy}.json','row_sha256':'a'*64,
                    'artifact_file':f'{seed}_{source}_{policy}.npz','artifact_sha256':'b'*64})
                stage['cells'].append({'source_index':source,'policy':policy,'state':'committed_complete','failure':None})
        stages.append(stage)
    return stages


def mutate_cell(stages,seed,source,policy,state,completed=300,category='coordinate_limit'):
    stage=stages[seed];index=next(i for i,r in enumerate(stage['rows']) if (r['source_index'],r['policy'])==(source,policy))
    cell=next(c for c in stage['cells'] if (c['source_index'],c['policy'])==(source,policy));cell['state']=state
    if state in ('uncompleted_after_started_invocation','never_started'):
        stage['rows'].pop(index);cell['failure']=None;return
    value=stage['rows'][index]['validated'];failure={'category':category,'forecast_step':completed+1}
    value.update(complete=False,status='failed',completed_steps=completed,failure=failure)
    value['mse_per_step']=value['mse_per_step'][:completed]
    for key in S.METRICS:
        if key!='mse_forecast200' or completed<200:value['metrics'][key]=None
    value['accepted_prefix_boundary']={'predicted_boundary_trajectory_maximum_excursion':.01}
    cell['failure']=failure


def test_complete360_all_three_contrasts_and_sample_sd():
    summary=S.summarize(synthetic_stages(),P)
    assert summary['required_outcomes']==360 and summary['coverage']=={'committed_complete':360}
    assert summary['all_required_outcomes_recorded'] and summary['all_required_outcomes_complete']
    expected={'base':([-1.,-2.,-3.],-2.),'random25':([-3.,-4.,-5.],-4.),
        'validation_rate_random_gate':([-.5,-1.5,-2.5],-1.5)}
    for policy,(values,mean) in expected.items():
        result=summary['primary_full_H395_contrasts']['learned_global_gate_minus_'+policy]
        assert list(result['seed_values'].values())==values and result['mean']==mean and result['sample_sd']==1.
    assert summary['absolute']['mean_rollout_mse']['base']['seed_values']=={'0':24.5,'1':34.5,'2':44.5}
    assert summary['always_base_requested_action_collapse'] is True
    assert summary['constructive_success_automatically_assigned'] is False
    assert sum(p['consumed_committed_work_totals']['deployed_forward_attempts']
        for s in summary['runtime'] for p in s['per_policy'].values())==142200


def test_one_failed_source_nulls_required_seed_and_aggregate_not_other_controls():
    stages=synthetic_stages();mutate_cell(stages,1,29,'random25','committed_guard_failed')
    summary=S.summarize(stages,P);primary=summary['primary_full_H395_contrasts']
    assert summary['all_required_outcomes_recorded'] and not summary['all_required_outcomes_complete']
    bad=primary['learned_global_gate_minus_random25']
    assert bad['seed_values']=={'0':-3.,'1':None,'2':-5.} and bad['mean'] is bad['sample_sd'] is None
    assert primary['learned_global_gate_minus_base']['mean']==-2.
    assert summary['paired_learned_minus_controls']['mse_forecast200']['learned_global_gate_minus_random25']['mean']==-4.
    assert summary['failed_accepted_prefixes'][0]['completed_steps']==300


def test_short_prefix_does_not_substitute_for_h200():
    stages=synthetic_stages();mutate_cell(stages,1,29,'random25','committed_guard_failed',completed=100)
    result=S.summarize(stages,P)['paired_learned_minus_controls']['mse_forecast200']['learned_global_gate_minus_random25']
    assert result['mean'] is None and result['defined_seed_pairs']==2


def test_unstarted_seed_retained_no_survivor_mean():
    stages=synthetic_stages();stages[2]['rows']=[];stages[2]['scientific_verification_passed']=False
    stages[2]['outcome']={'started':False,'state':'never_started'}
    for cell in stages[2]['cells']:cell.update(state='never_started',failure=None)
    result=S.summarize(stages,P)
    assert result['coverage']=={'committed_complete':240,'never_started':120}
    assert result['primary_full_H395_contrasts']['learned_global_gate_minus_base']['seed_values']=={'0':-1.,'1':-2.,'2':None}
    assert result['primary_full_H395_contrasts']['learned_global_gate_minus_base']['mean'] is None
    assert result['always_base_requested_action_collapse'] is None


def test_unverified_final_model_state_retains_rows_but_nulls_all_admitted_seed_means():
    stages=synthetic_stages();stages[1]['scientific_verification_passed']=False
    before=copy.deepcopy(stages[1]['rows'])
    result=S.summarize(stages,P)
    assert result['all_required_outcomes_numerically_complete'] and not result['all_required_outcomes_complete']
    assert result['all_required_outcomes_recorded'] and not result['all_model_stages_scientifically_verified']
    assert result['unverified_stage_rows_retained_as_diagnostics']==120 and stages[1]['rows']==before
    for metric,policies in result['absolute'].items():
        assert all(v['seed_values']['1'] is None and v['mean'] is None and v['sample_sd'] is None for v in policies.values())
    assert result['primary_full_H395_contrasts']['learned_global_gate_minus_base']['seed_values']=={'0':-1.,'1':None,'2':-3.}
    assert result['runtime'][1]['per_policy']['base']['consumed_committed_work_totals']['deployed_forward_attempts']==11850
    assert result['always_base_requested_action_collapse'] is None


@pytest.mark.parametrize('value',[None,1,0,'true'])
def test_scientific_stage_verification_is_required_boolean(value):
    stages=synthetic_stages();stages[1]['scientific_verification_passed']=value
    with pytest.raises(ValueError,match='Explicit per-stage'):S.summarize(stages,P)


def test_actual_case_wall_missing_is_not_filled_from_partial_component_timers():
    stages=synthetic_stages();stages[1]['rows'][0]['case_wall_seconds']=None
    result=S.summarize(stages,P)
    value=result['absolute']['end_to_end_case_wall_seconds']['base']
    assert value['seed_values']=={'0':2.,'1':None,'2':2.} and value['mean'] is None
    partial=result['absolute']['processing_plus_publication_seconds']['base']
    assert partial['defined_seed_pairs']==3
    assert result['runtime'][1]['per_policy']['base']['committed_rows_missing_case_wall_seconds']==1


def test_committed_budget_failure_keeps_timedout_distinction():
    stages=synthetic_stages();mutate_cell(stages,0,29,'learned_global_gate','timed_out_current',category='execution_budget')
    result=S.summarize(stages,P)
    assert result['coverage']['timed_out_current']==1 and result['all_required_outcomes_recorded']
    assert result['failed_accepted_prefixes'][0]['failure']['category']=='execution_budget'


@pytest.mark.parametrize('state,category',[('committed_execution_failed','execution_error'),
    ('committed_execution_failed','native_parity_failure'),('committed_guard_failed','nonfinite_risk')])
def test_failure_categories_stay_separate(state,category):
    stages=synthetic_stages();mutate_cell(stages,0,0,'base',state,category=category)
    assert S.summarize(stages,P)['coverage'][state]==1


def test_missing_timedout_current_with_no_committed_row():
    stages=synthetic_stages();mutate_cell(stages,0,29,'learned_global_gate','uncompleted_after_started_invocation')
    next(c for c in stages[0]['cells'] if c['source_index']==29 and c['policy']=='learned_global_gate')['state']='timed_out_current'
    result=S.summarize(stages,P)
    assert result['coverage']['timed_out_current']==1 and not result['all_required_outcomes_recorded']


@pytest.mark.parametrize('change',['missing_seed','duplicate_seed','missing_cell','duplicate_cell','missing_committed_row','wrong_failure_state'])
def test_incomplete_or_inconsistent_coverage_contract_rejected(change):
    stages=synthetic_stages()
    if change=='missing_seed':stages.pop()
    elif change=='duplicate_seed':stages[2]['seed']=1
    elif change=='missing_cell':stages[0]['cells'].pop()
    elif change=='duplicate_cell':stages[0]['cells'][-1]=stages[0]['cells'][0]
    elif change=='missing_committed_row':stages[0]['rows'].pop()
    elif change=='wrong_failure_state':stages[0]['cells'][0]['state']='committed_guard_failed'
    with pytest.raises(ValueError):S.summarize(stages,P)


def test_scalar_core_has_no_filesystem_simulator_or_process_imports():
    import ast
    tree=ast.parse(Path(S.__file__).read_text())
    imports={alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert imports=={'hashlib','math','numpy'}
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('open','eval','exec') for node in ast.walk(tree))
