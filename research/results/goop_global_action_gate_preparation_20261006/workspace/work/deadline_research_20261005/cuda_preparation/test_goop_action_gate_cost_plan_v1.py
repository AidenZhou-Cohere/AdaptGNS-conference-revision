"""Scalar-only synthetic cost tests; no model, arrays, processes or network."""
import copy
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('gate_cost',Path(__file__).with_name('goop_action_gate_cost_plan_v1.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
NOW=datetime(2026,10,6,12,tzinfo=timezone.utc)
def h(s):return hashlib.sha256(str(s).encode()).hexdigest()


def fixture(mode='train-rollout-capacity'):
    manifest={'split':'train','record_count':1000,'records':[{'source_index':i,'positions':{'shape':[401,i+1,2]}} for i in range(1000)]}
    bundle={'mode':mode,'protocol_sha256':M.PROTOCOL_SHA,'driver_sha256':h('driver'),'cohort_sha256':h('cohort'),
        'train_manifest_sha256':h('train'),'selection_sha256':h('selection'),'clock_error_bound_seconds':1}
    expected=M.schedule(mode,manifest);collections={};outcomes={};label=mode=='train-label-capacity'
    for seed in range(3):
        head=h(('head',seed));rows=[{'source_index':i,'target_frame' if label else 'policy':v,'status':'complete',
            'head_sha256':head,'selection_sha256':bundle['selection_sha256']} for i,v in expected]
        timings=[{**r,'wall_seconds':float(seed+1)} for r in rows]
        collections[seed]={'schema':f'adaptgns_goop_global_action_gate_{"label" if label else "rollout"}_collection_v1',
            'status':'complete','mode':mode,'split':'train','gate_protocol_sha256':M.PROTOCOL_SHA,'core_sha256':M.CORE_SHA,
            'original_training_protocol_sha256':M.TRAINING_PROTOCOL_SHA,'driver_sha256':bundle['driver_sha256'],
            'cohort_sha256':bundle['cohort_sha256'],'source_manifest_sha256':bundle['train_manifest_sha256'],
            'all_inputs_reverified':True,'model_state_verified_unchanged':True,'abort_reason':None,'all_required_outcomes_complete':True,
            'model':{'seed':seed,'arm':'mix','objective':'faithful','completed_updates':100000,'checkpoint_sha256':h(('checkpoint',seed))},
            'rows':rows,'required_rows':len(expected),'committed_rows':len(expected),'complete_rows':len(expected),
            'case_timings':timings,'runtime':{'setup_seconds':10},'selection_sha256':bundle['selection_sha256'],'head_sha256':head}
        outcomes[seed]={'started':True,'stopped_and_reaped':True,'exit_code':0,'signals':[],'quota_expired_at_observation':False,
            'inner_timeout_reported':False,'worker_failed_attempt':None,'elapsed_seconds':10+len(rows)*(seed+1)+7,
            'command':['/venv/python','/driver','--seed',str(seed),'--mode',mode,'--cuda-index',str(1 if seed==1 else 0),
                '--checkpoint-sha256',collections[seed]['model']['checkpoint_sha256']]}
    return bundle,collections,outcomes,manifest


def test_test_budget_includes_slowest_policy_margin_setup_parent_overhead_cleanup_and_reserves():
    b,c,o,m=fixture();result=M.build_plan(b,c,o,m,NOW)
    assert [r['test_allocation_seconds'] for r in result['per_seed']]==[179,341,503]
    # Floating arithmetic is conservatively rounded up, never rounded down.
    assert result['test_concurrent_allocation_seconds']==179+503+30
    assert sum(result['remaining_outer_reserves_seconds'].values())==7200
    assert M.stamp(result['latest_test_start_utc'])==M.DEADLINE-timedelta(seconds=712+7200+1)
    assert result['status']=='candidate_requires_root_review' and result['test_admitted'] is False


def test_label_budget_keeps_fulltrain_and_validation_counts():
    b,c,o,m=fixture('train-label-capacity');result=M.build_plan(b,c,o,m,NOW)
    assert len(M.schedule(b['mode'],m))==64 and M.schedule(b['mode'],m)[-1]==(999,400)
    assert result['per_seed'][0]['whole_invocation_allocations_seconds']=={'train_labels':10817,'validation_labels':220}
    assert result['complete_test_cost_remains_unmeasured'] is True


@pytest.mark.parametrize('bad',['missing_seed','guard','missing_case','duplicate_timing','nonfinite','zero_time','missing_overhead',
    'wrong_checkpoint','wrong_gpu','selection','changed_model','quota','wrong_train_count'])
def test_incomplete_or_mismatched_capacity_never_gets_budget(bad):
    b,c,o,m=fixture()
    if bad=='missing_seed':c.pop(2)
    elif bad=='guard':c[0]['rows'][0]['status']='failed'
    elif bad=='missing_case':c[0]['rows'].pop()
    elif bad=='duplicate_timing':c[0]['case_timings'][1]=copy.deepcopy(c[0]['case_timings'][0])
    elif bad=='nonfinite':c[0]['case_timings'][0]['wall_seconds']=float('nan')
    elif bad=='zero_time':c[0]['case_timings'][0]['wall_seconds']=0
    elif bad=='missing_overhead':o[0]['elapsed_seconds']=1
    elif bad=='wrong_checkpoint':o[0]['command'][-1]=h('other')
    elif bad=='wrong_gpu':o[0]['command'][o[0]['command'].index('--cuda-index')+1]='3'
    elif bad=='selection':c[0]['selection_sha256']=h('changed')
    elif bad=='changed_model':c[0]['model_state_verified_unchanged']=False
    elif bad=='quota':o[0]['quota_expired_at_observation']=True
    else:m['records'].pop()
    with pytest.raises(ValueError):M.build_plan(b,c,o,m,NOW)


def test_completed_preflight_recorded_as_spent_without_double_charge():
    b,c,o,m=fixture();b['completed_outer_stages']={'input_preflight':{'path':'/synthetic/completed_preflight.json','sha256':h('completed preflight')}}
    receipt={'schema':'adaptgns_goop_action_gate_preflight_completion_v1','status':'complete','issued_by':'root',
        'protocol_sha256':M.PROTOCOL_SHA,'cohort_sha256':b['cohort_sha256'],'selection_sha256':b['selection_sha256'],
        'actual_elapsed_seconds':120,'completed_utc':NOW.isoformat()}
    with pytest.raises(ValueError,match='Actual completed-stage'):M.build_plan(b,c,o,m,NOW)
    r=M.build_plan(b,c,o,m,NOW,completed_receipts={'input_preflight':(receipt,h('completed preflight'))})
    assert sum(r['remaining_outer_reserves_seconds'].values())==6300
    assert r['completed_outer_stage_evidence']['input_preflight']['actual_elapsed_seconds']==120
    b['completed_outer_stages']['scalar_array_collection']=b['completed_outer_stages']['input_preflight']
    with pytest.raises(ValueError):M.build_plan(b,c,o,m,NOW)


def test_late_capacity_remains_candidate_with_fit_false():
    b,c,o,m=fixture();r=M.build_plan(b,c,o,m,M.DEADLINE-timedelta(seconds=1))
    assert r['complete_remaining_work_fits'] is False and r['all_train_validation_labels_complete'] is False


def test_default_description_only(monkeypatch,capsys):
    monkeypatch.setattr(M,'build_plan',lambda *args:pytest.fail('description only'))
    assert M.main([])==0 and 'description_only' in capsys.readouterr().out
