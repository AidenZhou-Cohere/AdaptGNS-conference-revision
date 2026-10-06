"""Synthetic-only contracts: no model/checkpoint/source trajectory reads."""
import copy
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent


def module(name):
    spec=importlib.util.spec_from_file_location('_test_'+name,HERE/(name+'.py'))
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


D=module('run_goop_action_gate_v1')
C=module('goop_global_action_gate_core_v1')
F=module('fit_goop_action_gate_v1')


class Guard(Exception):
    def __init__(self,category):
        self.details={'category':category};super().__init__(category)


class Positions:
    def __init__(self,events):
        self.events=events
        self.values=np.full((401,4,2),.3,dtype=np.float32)
    def __getitem__(self,index):
        if isinstance(index,int):self.events.append(('target',index))
        return self.values[index]


class Native:
    POLICIES=('base','dense','random25','speed25','laggedrisk25')
    def __init__(self,budget=1,fail_at=None,error=None):
        self.events=[];self.budget=budget;self.fail_at=fail_at;self.error=error
        self.forwards=0;self.draws=[];self.parity_ok=True
        self.full=NS(check_state=self.check,synchronize=lambda _:None,RolloutGuard=Guard,
            state_hash=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            boundary_metrics=lambda p,b:{'outside_fraction':float(np.mean((p<.1)|(p>.9)))})
        self.bridge=NS(MAX_ABS=10.,supplied=self.supplied,validate_output=self.validate,native_parity=self.parity)
    @staticmethod
    def check(values,label,limit):
        if not np.isfinite(values).all():raise Guard('nonfinite_'+label)
        if np.max(np.abs(values))>limit:raise Guard('coordinate_limit_'+label)
    def native_graph(self,history,policy,cached,rng):
        assert policy in ('base','random25') and cached is None
        n=history.shape[1];base=np.vstack((np.arange(n),np.arange(n))).astype(np.int64)
        optional=np.empty((0,2),dtype=np.int64)
        if policy=='random25':
            draw=int(rng.integers(0,n-1));self.draws.append(draw)
            if self.budget:optional=np.array([[draw,draw+1]],dtype=np.int64)
        extra=np.concatenate((optional,optional[:,::-1]),axis=0).T
        edges=np.concatenate((base,extra),axis=1)
        self.events.append(('graph',policy))
        return None,edges,optional,{'retained_optional_pairs':len(optional),'optional_pair_budget':self.budget,
            'native_base_prefix_preserved':True,'directed_edges':edges.shape[1]}
    def supplied(self,model,history,types,edges,device):
        self.forwards+=1;self.events.append(('forward',self.forwards))
        if self.forwards==self.fail_at:raise self.error
        n=len(types);dx=0.00001 if edges.shape[1]>n else 0.
        return {'prediction':np.asarray(history[-1]+dx,dtype=np.float32),
            'risk':np.ones(n,dtype=np.float32),'raw_risk':np.ones(n,dtype=np.float32),'operational_seconds':0.0}
    def validate(self,output,n):
        assert output['prediction'].shape==(n,2) and output['risk'].shape==(n,)
        self.check(output['prediction'],'prediction',10.)
    def parity(self,model,history,types,edges,device):
        self.events.append(('parity',2))
        output={'prediction':history[-1].copy(),'risk':np.ones(len(types),dtype=np.float32),
            'raw_risk':np.ones(len(types),dtype=np.float32)}
        # Exact numeric key family returned by frozen bridge.native_parity.
        arrays={'native_prediction':output['prediction'].copy(),'native_raw_risk':output['raw_risk'].copy(),
            'native_risk':output['risk'].copy(),'native_edges':edges.copy(),
            'native_node_features':history[-1].copy(),'native_edge_features':np.zeros((edges.shape[1],3),np.float32),
            'supplied_node_features':history[-1].copy(),'supplied_edge_features':np.zeros((edges.shape[1],3),np.float32)}
        return {'passed':self.parity_ok},arrays,output


def head(prediction=0.,p=.5):
    return dict(schema=C.SCHEMA,protocol_sha256=C.PROTOCOL_SHA,model_seed=0,
        feature_names=list(C.FEATURES),feature_mean=[0.]*11,feature_std=[1.]*11,feature_scale=[1.]*11,
        label_mean=prediction,label_std=1.,label_scale=1.,coefficients=[0.]*11,ridge_lambda=.1,
        training_states=8000,threshold=0.,training_input_hashes={'checkpoint_sha256':'a'*64},
        validation_requested_expansion_fraction=p)


def rollout(native,policy='base',selected=None):
    positions=Positions(native.events)
    return D.rollout_row(native,C,None,positions,np.full(4,7,np.int64),{'bounds':[[.1,.9],[.1,.9]]},
        {'source_index':0,'policy':policy},0,head() if selected is None else selected,'test','cpu')


def label(native):
    return D.label_row(native,C,None,Positions(native.events),np.full(4,7,np.int64),
        {'source_index':0,'target_frame':6},0,'train','cpu')


def test_label_no_future_access_until_two_actions():
    native=Native();row,arrays=label(native)
    assert row['status']=='complete' and row['network_passes']==2
    assert native.events[-1]==('target',6)
    assert [e[0] for e in native.events]==['graph','forward','graph','forward','target']
    assert set(arrays)==F.ARRAY_KEYS
    assert arrays['history'].shape==(6,4,2) and arrays['features'].dtype==np.float64


@pytest.mark.parametrize('error,category',[(TimeoutError('synthetic budget'),'execution_budget'),
    (RuntimeError('synthetic forward'),'execution_error'),(Guard('nonfinite_risk'),'nonfinite_risk')])
def test_label_failure_keeps_prefix_and_no_target(error,category):
    native=Native(fail_at=2,error=error);row,arrays=label(native)
    assert row['status']=='failed' and row['failure']['category']==category
    assert 'base_prediction' in arrays and 'target' not in arrays
    assert not any(e[0]=='target' for e in native.events)
    assert row['base_mse'] is row['signed_benefit'] is None


def test_failed_label_preserves_nonfinite_output():
    native=Native();old=native.bridge.supplied
    def supplied(*args):
        output=old(*args);output['prediction'][0,0]=np.nan;return output
    native.bridge.supplied=supplied
    row,arrays=label(native)
    assert row['status']=='failed' and np.isnan(arrays['base_prediction'][0,0])
    assert 'target' not in arrays


def test_label_scoring_timeout_category_and_target_preserved(monkeypatch):
    def timeout(*_):raise TimeoutError('synthetic scoring quota')
    monkeypatch.setattr(C,'action_errors',timeout)
    row,arrays=label(Native())
    assert row['failure']['category']=='execution_budget' and 'target' in arrays


@pytest.mark.parametrize('policy',C.POLICIES)
def test_full_rollout_single_deployed_forward_then_truth(policy):
    native=Native();row,arrays=rollout(native,policy,head(prediction=1.))
    assert row['status']=='complete' and row['completed_steps']==395
    assert row['network_passes']==native.forwards==395 and row['native_parity_network_passes']==2
    events=[e[0] for e in native.events if e[0] in ('forward','target')]
    assert events==['forward','target']*395
    assert arrays['prediction'].shape==(395,4,2)
    assert arrays['feature_trace'].shape==(395 if policy=='learned_global_gate' else 0,11)
    assert np.isfinite(arrays['feature_trace']).all()
    assert 'initial_parity_native_prediction' in arrays and 'initial_parity_supplied_raw_risk' in arrays


def test_rng_per_step_independent_of_skipped_expansions():
    native=Native();row,_=rollout(native,'validation_rate_random_gate',head(p=.47))
    requested=[s['forecast_step'] for s in row['steps'] if s['requested_expand']]
    expected=[int(C.pair_rng('test',0,0,s).integers(0,3)) for s in requested]
    assert native.draws==expected and 0<len(requested)<395
    assert all(s['requested_expand']==C.independent_gate(.47,'test',0,0,s['forecast_step']) for s in row['steps'])


def test_zero_budget_exact_base_equivalence():
    base,base_arrays=rollout(Native(0),'base')
    for policy in C.POLICIES[1:]:
        row,arrays=rollout(Native(0),policy,head(prediction=1.,p=1.))
        assert np.array_equal(arrays['prediction'],base_arrays['prediction'])
        assert row['mse_per_step']==base['mse_per_step']
        assert row['zero_budget_fraction']==1. and row['effective_expansion_fraction']==0.


@pytest.mark.parametrize('error,category',[(TimeoutError('synthetic budget'),'execution_budget'),
    (RuntimeError('synthetic forward'),'execution_error'),(Guard('nonfinite_risk'),'nonfinite_risk')])
def test_rollout_failed_prefix_has_null_full_metrics(error,category):
    native=Native(fail_at=4,error=error);row,arrays=rollout(native)
    assert row['failure']['category']==category and row['failure']['forecast_step']==4
    assert row['completed_steps']==3 and arrays['prediction'].shape==(3,4,2)
    assert row['mean_rollout_mse'] is row['mse_at_final_horizon'] is None
    assert row['mse_at_declared_trace_steps']['1']==0. and row['mse_at_declared_trace_steps']['10'] is None
    assert 'failed_input_history' in arrays


@pytest.mark.parametrize('exception',[TimeoutError('synthetic quota'),RuntimeError('parity wrapper')])
def test_parity_exception_does_not_invent_forward_count(exception):
    native=Native()
    def fail(*_):raise exception
    native.bridge.native_parity=fail
    row,arrays=rollout(native)
    assert row['native_parity_network_passes'] is None and row['network_passes']==0
    assert row['failure']['category']==('execution_budget' if isinstance(exception,TimeoutError) else 'native_parity_failure')
    assert row['failure']['forecast_step']==0


def test_initial_graph_failure_is_before_forecasting():
    native=Native()
    def fail(*_):raise RuntimeError('synthetic initial graph')
    native.native_graph=fail
    row,arrays=rollout(native)
    assert row['failure']['phase']=='initial_graph' and row['failure']['forecast_step']==0
    assert row['network_passes']==row['completed_steps']==0


def test_parity_failure_keeps_raw_arrays():
    native=Native();native.parity_ok=False
    row,arrays=rollout(native)
    assert row['failure']['category']=='native_parity_failure'
    assert row['network_passes']==0 and 'initial_parity_native_prediction' in arrays
    assert 'initial_parity_supplied_prediction' in arrays


def test_no_global_policy_mutation():
    native=Native();before=native.POLICIES
    rollout(native,'learned_global_gate',head(prediction=1.))
    assert native.POLICIES is before
    source=Path(D.__file__).read_text()
    assert 'POLICIES =' not in source and '.POLICIES=' not in source and 'setattr(' not in source


def test_fixed_complete_schedules_and_capacity_ties():
    manifest={'records':[{'positions':{'shape':[401,10+i%8,2]}} for i in range(1000)]}
    for mode,split,count in [('train-labels','train',8000),('validation-labels','valid',150)]:
        schedule=D.expected_schedule(C,mode,manifest)
        assert len(schedule)==count and [(s['source_index'],s['target_frame']) for s in schedule]==C.row_ids(split)
    capacity=D.expected_schedule(C,'train-label-capacity',manifest)
    assert len(capacity)==64 and capacity[0]=={'source_index':0,'target_frame':6}
    assert capacity[-1]=={'source_index':999,'target_frame':400}
    ordered=sorted(range(1000),key=lambda i:(manifest['records'][i]['positions']['shape'][1],i))
    assert D.capacity_sources(manifest['records'])==[ordered[0],ordered[499],ordered[999]]
    assert len(D.expected_schedule(C,'train-rollout-capacity',manifest))==12
    assert len(D.expected_schedule(C,'test-rollout',manifest))==120


def test_row_roundtrip_accepted_by_frozen_fitter(tmp_path):
    row,arrays=label(Native());committed=D.publish_row(tmp_path,'synthetic',row,arrays)
    F.array_check(np,C,committed,tmp_path/committed['artifact_file'])
    saved=json.loads((tmp_path/committed['row_file']).read_text())
    assert saved=={k:v for k,v in committed.items() if k not in ('row_file','row_sha256')}


def test_numeric_mutation_during_npz_publication_is_rejected(tmp_path,monkeypatch):
    arrays={'v':np.arange(4,dtype=np.float32)};save=np.savez_compressed
    def mutate(stream,**items):
        save(stream,**items);items['v'][0]=123.
    monkeypatch.setattr(np,'savez_compressed',mutate)
    with pytest.raises(ValueError,match='changed during publication'):
        D.publish_row(tmp_path,'failed',{'timing':{}},arrays)
    assert not (tmp_path/'failed.json').exists() and (tmp_path/'failed.npz.tmp').exists()


def test_serialized_npz_mutation_rejected(tmp_path,monkeypatch):
    save=np.savez_compressed
    def wrong(stream,**items):save(stream,v=np.array([99.],dtype=np.float32))
    monkeypatch.setattr(np,'savez_compressed',wrong)
    with pytest.raises(ValueError,match='Serialized numeric'):
        D.publish_row(tmp_path,'failed',{'timing':{}},{'v':np.arange(4,dtype=np.float32)})


def test_mutation_during_row_json_serialization_rejected(tmp_path,monkeypatch):
    arrays={'v':np.arange(4,dtype=np.float32)};encode=D.encode
    def mutate(row):
        raw=encode(row);arrays['v'][0]=123.;return raw
    monkeypatch.setattr(D,'encode',mutate)
    with pytest.raises(ValueError,match='row serialization'):
        D.publish_row(tmp_path,'failed',{'timing':{}},arrays)
    assert (tmp_path/'failed.npz').exists() and not (tmp_path/'failed.json').exists()


@pytest.mark.parametrize('moment',['before','after'])
def test_scalar_row_mutation_during_serialization_rejected(tmp_path,monkeypatch,moment):
    encode=D.encode
    def mutate(row):
        if moment=='before':row['features'][0]=123.
        raw=encode(row)
        if moment=='after':row['features'][0]=123.
        return raw
    monkeypatch.setattr(D,'encode',mutate)
    with pytest.raises(ValueError,match='Scalar row changed'):
        D.publish_row(tmp_path,'failed',{'features':[1.],'timing':{}},{'v':np.ones(1)})
    assert not (tmp_path/'failed.json').exists()


@pytest.mark.parametrize('moment',['before','after'])
def test_scalar_collection_mutation_during_serialization_rejected(tmp_path,monkeypatch,moment):
    encode=D.encode
    def mutate(row):
        if moment=='before':row['features'][0]=123.
        raw=encode(row)
        if moment=='after':row['features'][0]=123.
        return raw
    monkeypatch.setattr(D,'encode',mutate)
    with pytest.raises(ValueError,match='Scalar collection changed'):
        D.publish_collection(tmp_path,'label_collection.json',{'features':[1.]},{})
    assert not (tmp_path/'label_collection.json').exists()


def test_aggregate_mutation_after_row_commit_rejected(tmp_path):
    row,arrays=label(Native());committed=D.publish_row(tmp_path,'synthetic',row,arrays)
    committed['features'][0]+=1.
    with pytest.raises(ValueError,match='Aggregate scalar row'):
        D.publish_collection(tmp_path,'label_collection.json',{'rows':[committed]},{})


@pytest.mark.parametrize('which',['input','output','directory'])
def test_collection_serialization_mutation_blocks_success(tmp_path,monkeypatch,which):
    source=tmp_path/'source';source.write_text('source');out=tmp_path/'out';out.mkdir();(out/'partial').write_text('partial')
    bindings={str(source):D.sha(source)};encode=D.encode
    def mutate(row):
        raw=encode(row)
        if which=='input':source.write_text('changed')
        elif which=='output':(out/'partial').write_text('changed')
        else:(out/'extra').mkdir()
        return raw
    monkeypatch.setattr(D,'encode',mutate)
    with pytest.raises(ValueError):D.publish_collection(out,'label_collection.json',{'status':'complete'},bindings)
    assert not (out/'label_collection.json').exists()


def test_inventory_order_matches_fitter(tmp_path):
    (tmp_path/'a').mkdir();(tmp_path/'a'/'x').write_text('nested');(tmp_path/'a.json').write_text('sibling')
    (tmp_path/'label_collection.json').write_text('self')
    snapshot=D.tree_snapshot(tmp_path,('label_collection.json',));files,entries=F.inventory(tmp_path)
    assert snapshot=={'files':files,'output_tree_entries':entries}


def test_coverage_retains_failed_never_started():
    schedule=[{'source_index':i,'target_frame':6} for i in range(3)]
    rows=[{**schedule[0],'status':'complete'},{**schedule[1],'status':'failed'}]
    assert [x['state'] for x in D.collection_coverage(schedule,rows)]==['committed_complete','committed_failed','never_started']
    with pytest.raises(ValueError,match='Duplicate'):D.collection_coverage(schedule,rows+rows[:1])


def release_fixture(tmp_path,monkeypatch,mode='train-label-capacity',seed=0):
    args=D.parse_args([]);args.mode=mode;args.seed=seed;args.cuda_index=1 if seed==1 else 0;args.max_seconds=100
    for name in D.PATH_ARGS:
        key=name.replace('-','_')
        if name in ('head','selection','cross-split-audit'):continue
        path=tmp_path/name
        if name!='output-dir':path.write_text('synthetic '+name)
        setattr(args,key,path)
    args.release=tmp_path/'release.json';args.release.write_text('{}')
    args.benchmark_helper=HERE/'benchmark_goop_graph_support_rollout.py'
    args.trainer_source=HERE/'train_goop_graph_support_cuda.py'
    args.gate_protocol=HERE/'goop_global_action_gate_protocol_v1.md'
    args.checkpoint_sha256=D.sha(args.checkpoint)
    monkeypatch.setattr(D,'TRAINING_PROTOCOL_SHA',D.sha(args.protocol))
    release={'schema':D.RELEASE_SCHEMA,'issued_by':'root','status':'approved_for_fixed_gate_driver',
        'host':D.socket.gethostname(),'mode':mode,'seed':seed,'arm':'mix','checkpoint_updates':100000,
        'cuda_index':args.cuda_index,'threads':2,'max_seconds':100,'output_dir':str(args.output_dir),'gpu_uuid':'synthetic-only',
        'source_sha256':{**D.SOURCE_PINS,Path(D.__file__).name:D.sha(D.__file__)},
        'protocol_sha256':D.GATE_PROTOCOL_SHA,'original_training_protocol_sha256':D.TRAINING_PROTOCOL_SHA,
        'absolute_stop_utc':'2026-10-07T03:00:00+00:00','files_sha256':{}}
    if 'rollout' in mode:
        args.head=tmp_path/'head.json';args.selection=tmp_path/'selection.json'
        selected=head();selected['model_seed']=seed;selected['training_input_hashes']['checkpoint_sha256']=args.checkpoint_sha256
        D.write(args.head,selected,exclusive=True)
        D.write(args.selection,{'schema':'adaptgns_goop_global_action_gate_selection_v1','protocol_sha256':D.GATE_PROTOCOL_SHA,
            'heads':{str(s):selected if s==seed else {**selected,'model_seed':s} for s in (0,1,2)}},exclusive=True)
    if mode=='test-rollout':
        args.cross_split_audit=tmp_path/'cross-split-audit';args.cross_split_audit.write_text('synthetic')
        path=tmp_path/'capacity.json'
        D.write(path,{'schema':'adaptgns_goop_global_action_gate_capacity_budget_v1',
            'status':'complete_measured_capacity_passed','protocol_sha256':D.GATE_PROTOCOL_SHA,
            'selection_sha256':D.sha(args.selection),'all_36_rollouts_complete':True,
            'all_train_validation_labels_complete':True,'latest_test_start_utc':'2026-10-07T01:00:00+00:00',
            'driver_sha256':D.sha(D.__file__),'cohort_sha256':D.sha(args.cohort),'mapping':{'GPU0':[0,2],'GPU1':[1]},
            'cleanup_seconds_per_invocation':15,'per_seed':[{'seed':s,'test_allocation_seconds':100,
                'checkpoint_sha256':args.checkpoint_sha256 if s==seed else f'{s+1:064x}'} for s in range(3)]},exclusive=True)
        release['complete_capacity_report']={'path':str(path),'sha256':D.sha(path)}
        release['files_sha256'][str(path)]=D.sha(path)
    for name in D.PATH_ARGS:
        path=getattr(args,name.replace('-','_'))
        if path is not None and name not in ('repo','output-dir'):release['files_sha256'][str(path)]=D.sha(path)
    return args,release


NOW=datetime(2026,10,6,1,tzinfo=timezone.utc)


@pytest.mark.parametrize('mode',D.MODES)
def test_complete_scalar_root_release_accepted(tmp_path,monkeypatch,mode):
    args,release=release_fixture(tmp_path,monkeypatch,mode)
    assert D.validate_release(args,release,NOW)==release['files_sha256']


@pytest.mark.parametrize('change',('seed_bool','source','host','checkpoint','pin_missing','deadline','probability','capacity'))
def test_scalar_release_rejects_drift(tmp_path,monkeypatch,change):
    args,release=release_fixture(tmp_path,monkeypatch,'test-rollout')
    if change=='seed_bool':args.seed=False
    elif change=='source':release['source_sha256'][Path(D.__file__).name]='0'*64
    elif change=='host':release['host']='other'
    elif change=='checkpoint':args.checkpoint.write_text('changed')
    elif change=='pin_missing':del release['files_sha256'][str(args.manifest)]
    elif change=='deadline':release['absolute_stop_utc']='2026-10-07T05:00:00+00:00'
    elif change=='probability':
        value=json.loads(args.head.read_text());value['validation_requested_expansion_fraction']=2.
        D.write(args.head,value);release['files_sha256'][str(args.head)]=D.sha(args.head)
    elif change=='capacity':del release['complete_capacity_report']
    with pytest.raises((ValueError,KeyError)):D.validate_release(args,release,NOW)


def update_capacity(release,change):
    path=Path(release['complete_capacity_report']['path']);capacity=json.loads(path.read_text());change(capacity)
    path.write_text(json.dumps(capacity));pin=D.sha(path)
    release['complete_capacity_report']['sha256']=pin;release['files_sha256'][str(path)]=pin


def test_second_wave_has_its_own_derived_latest_start(tmp_path,monkeypatch):
    args,release=release_fixture(tmp_path,monkeypatch,'test-rollout',seed=2)
    update_capacity(release,lambda c:c.update(latest_test_start_utc=NOW.isoformat()))
    assert D.validate_release(args,release,NOW+timedelta(seconds=114))
    with pytest.raises(ValueError,match='fixed-wave invocation'):
        D.validate_release(args,release,NOW+timedelta(seconds=116))


@pytest.mark.parametrize('seed',[0,1])
def test_first_wave_cannot_use_second_wave_extension(tmp_path,monkeypatch,seed):
    args,release=release_fixture(tmp_path,monkeypatch,'test-rollout',seed=seed)
    update_capacity(release,lambda c:c.update(latest_test_start_utc=NOW.isoformat()))
    with pytest.raises(ValueError,match='fixed-wave invocation'):D.validate_release(args,release,NOW+timedelta(seconds=1))


@pytest.mark.parametrize('bad',[0,-1,True,100.5,float('inf'),float('nan')])
def test_capacity_allocation_requires_positive_integer(tmp_path,monkeypatch,bad):
    args,release=release_fixture(tmp_path,monkeypatch,'test-rollout',seed=2)
    update_capacity(release,lambda c:c['per_seed'][0].update(test_allocation_seconds=bad))
    with pytest.raises(ValueError,match='positive integer'):D.validate_release(args,release,NOW)


@pytest.mark.parametrize('change',['quota','checkpoint','mapping','cleanup','seed_grid','driver','cohort'])
def test_capacity_bound_identity_must_match(tmp_path,monkeypatch,change):
    args,release=release_fixture(tmp_path,monkeypatch,'test-rollout')
    def mutate(c):
        if change=='quota':c['per_seed'][0]['test_allocation_seconds']=101
        elif change=='checkpoint':c['per_seed'][0]['checkpoint_sha256']='f'*64
        elif change=='mapping':c['mapping']={'GPU0':[0],'GPU1':[1,2]}
        elif change=='cleanup':c['cleanup_seconds_per_invocation']=0
        elif change=='seed_grid':c['per_seed'][2]['seed']=1
        elif change=='driver':c['driver_sha256']='f'*64
        elif change=='cohort':c['cohort_sha256']='f'*64
    update_capacity(release,mutate)
    with pytest.raises(ValueError):D.validate_release(args,release,NOW)


def test_own_whole_invocation_must_fit_absolute_stop(tmp_path,monkeypatch):
    args,release=release_fixture(tmp_path,monkeypatch,'test-rollout')
    update_capacity(release,lambda c:c.update(latest_test_start_utc=release['absolute_stop_utc']))
    stop=D.stamp(release['absolute_stop_utc'])
    with pytest.raises(ValueError,match='fixed-wave invocation'):
        D.validate_release(args,release,stop-timedelta(seconds=50))


def execution_fixture(tmp_path,monkeypatch,mode='train-label-capacity'):
    args=D.parse_args([]);args.mode=mode;args.split=D.MODES[mode];args.seed=0;args.arm='mix';args.objective='faithful'
    args.cuda_index=0;args.threads=2;args.max_seconds=3600;args.repo=tmp_path/'repo';args.output_dir=tmp_path/'output'
    bindings={}
    for key in ('manifest','cohort_audit','train_admission','checkpoint','protocol','trainer_source'):
        path=tmp_path/(key+'.json');path.write_text('synthetic '+key);setattr(args,key,path);bindings[str(path)]=D.sha(path)
    args.checkpoint_sha256=bindings[str(args.checkpoint)]
    if 'rollout' in mode:
        for key,value in [('head',head()),('selection',{'synthetic':True})]:
            path=tmp_path/(key+'.json');D.write(path,value,exclusive=True);setattr(args,key,path);bindings[str(path)]=D.sha(path)
    native=Native();positions=Positions([]).values;types=np.full(4,7,np.int64)
    p_path=tmp_path/'positions.npy';t_path=tmp_path/'types.npy';np.save(p_path,positions);np.save(t_path,types)
    descriptor=lambda p,a:{'shape':list(a.shape),'size_bytes':p.stat().st_size,'sha256':D.sha(p),'path':p.name}
    record={'positions':descriptor(p_path,positions),'particle_types':descriptor(t_path,types),'trajectory_content_sha256':'e'*64}
    count=1000 if args.split=='train' else 30
    manifest={'records':[copy.deepcopy(record) for _ in range(count)],'metadata':{'bounds':[[.1,.9],[.1,.9]]}}
    identity={'seed':0,'arm':'mix','objective':'faithful','completed_updates':100000,
        'checkpoint_sha256':args.checkpoint_sha256,'run_config_sha256':'d'*64}
    model=NS(state_dict=lambda:{'synthetic':1})
    helpers=NS(cpu_tree=copy.deepcopy,torch=NS(no_grad=nullcontext),data_loader=NS(
        load_manifest_data=lambda *a,**kw:[(positions,types)]*count,
        _manifest_array_path=lambda root,r:root/r['path']))
    bench=NS(SOURCE_PINS={},deadline_alarm=lambda _:nullcontext(),load_helpers=lambda _:(native,helpers),
        load_split_evidence=lambda *a:{},configure_cuda=lambda *a:('cpu',{'uuid':'synthetic-only'}),
        prepare_model=lambda *a:(model,identity),tree_equal=lambda t,a,b:a==b)
    mods=NS(core=C,bench=bench,final=NS(METADATA_SHA='c'*64))
    context=NS(args=args,mods=mods,bindings=bindings,manifest=manifest,admission={},train_admission={},cohort_sha='b'*64,
        release={'gpu_uuid':'synthetic-only','absolute_stop_utc':'2026-10-07T03:00:00+00:00'})
    expected=([{'source_index':0,'target_frame':6},{'source_index':0,'target_frame':62}] if 'rollout' not in mode
        else [{'source_index':0,'policy':'base'},{'source_index':0,'policy':'random25'}])
    monkeypatch.setattr(D,'expected_schedule',lambda *a:expected)
    return context,native,expected


def test_execute_commits_complete_collection_and_real_bound_arrays(tmp_path,monkeypatch):
    context,native,expected=execution_fixture(tmp_path,monkeypatch)
    assert D.execute(context,time.perf_counter())==0
    output=context.args.output_dir;collection=json.loads((output/'label_collection.json').read_text())
    assert collection['status']=='complete' and collection['required_rows']==collection['complete_rows']==2
    assert collection['all_inputs_reverified'] and collection['model_state_verified_unchanged']
    assert str(tmp_path/'positions.npy') in collection['input_sha256']
    assert 'initial_native_parity.npz' in collection['files']
    assert all(t['wall_seconds']>0 for t in collection['case_timings'])
    assert collection['runtime']['setup_seconds']>0
    for row in collection['rows']:F.array_check(np,C,row,output/row['artifact_file'])


def test_full_150_validation_driver_collection_fitter_roundtrip(tmp_path,monkeypatch):
    original_schedule=D.expected_schedule
    context,native,_=execution_fixture(tmp_path,monkeypatch,'validation-labels')
    monkeypatch.setattr(D,'expected_schedule',original_schedule)
    assert D.execute(context,time.perf_counter())==0
    path=context.args.output_dir/'label_collection.json'
    collection,data=F.read_collection({'seed':0,'split':'valid','path':str(path),'sha256':D.sha(path)},
        {'driver_sha256':D.sha(D.__file__)},np,C,{},lambda:None)
    assert collection['complete_rows']==150 and data['features'].shape==(150,11)
    assert data['row_ids']==C.row_ids('valid')


def test_execute_budget_failure_commits_prefix_and_stops(tmp_path,monkeypatch):
    context,native,expected=execution_fixture(tmp_path,monkeypatch)
    native.fail_at=2;native.error=TimeoutError('synthetic quota')
    assert D.execute(context,time.perf_counter())==1
    collection=json.loads((context.args.output_dir/'label_collection.json').read_text())
    assert collection['status']=='stopped_incomplete_or_failed' and collection['committed_rows']==1
    assert collection['rows'][0]['failure']['category']=='execution_budget'
    assert [c['state'] for c in collection['coverage']]==['committed_failed','never_started']


def test_execute_initial_parity_failure_preserved(tmp_path,monkeypatch):
    context,native,expected=execution_fixture(tmp_path,monkeypatch);native.parity_ok=False
    assert D.execute(context,time.perf_counter())==1
    out=context.args.output_dir
    assert json.loads((out/'initial_native_parity.json').read_text())['status']=='failed'
    assert json.loads((out/'label_collection.json').read_text())['committed_rows']==0
    assert (out/'initial_native_parity.npz').exists()


def test_execute_invalid_head_publishes_failed_setup(tmp_path,monkeypatch):
    context,native,expected=execution_fixture(tmp_path,monkeypatch,'train-rollout-capacity')
    D.write(context.args.head,{'schema':'invalid'})
    context.bindings[str(context.args.head)]=D.sha(context.args.head)
    assert D.execute(context,time.perf_counter())==1
    out=context.args.output_dir
    assert (out/'failed_attempt.json').exists() and (out/'rollout_collection.json').exists()
    assert native.forwards==0


def test_test_rollout_continues_scientific_guards_but_not_runtime_errors(tmp_path,monkeypatch):
    context,native,expected=execution_fixture(tmp_path,monkeypatch,'test-rollout')
    native.fail_at=2;native.error=Guard('synthetic_guard')
    assert D.execute(context,time.perf_counter())==0
    collection=json.loads((context.args.output_dir/'rollout_collection.json').read_text())
    assert collection['status']=='complete_with_guard_failures' and collection['committed_rows']==2
    assert collection['rows'][0]['mean_rollout_mse'] is None and collection['complete_rows']==1


def test_input_mutation_blocks_final_success_collection(tmp_path,monkeypatch):
    context,native,expected=execution_fixture(tmp_path,monkeypatch)
    original=D.publish_row
    def mutate(*args,**kwargs):
        value=original(*args,**kwargs)
        if args[1].startswith('source'):context.args.checkpoint.write_text('changed')
        return value
    monkeypatch.setattr(D,'publish_row',mutate)
    assert D.execute(context,time.perf_counter())==1
    out=context.args.output_dir
    assert not (out/'label_collection.json').exists() and (out/'failed_collection_publication.json').exists()


@pytest.mark.parametrize('which',['artifact','row','protocol','parity'])
def test_previous_commit_mutation_blocks_final_success(tmp_path,monkeypatch,which):
    context,native,_=execution_fixture(tmp_path,monkeypatch)
    original=D.publish_row
    def mutate(*args,**kwargs):
        value=original(*args,**kwargs)
        if args[1].startswith('source'):
            name={'artifact':value['artifact_file'],'row':value['row_file'],
                'protocol':'protocol.json','parity':'initial_native_parity.npz'}[which]
            (context.args.output_dir/name).write_bytes(b'changed committed bytes')
        return value
    monkeypatch.setattr(D,'publish_row',mutate)
    assert D.execute(context,time.perf_counter())==1
    out=context.args.output_dir
    assert not (out/'label_collection.json').exists() and (out/'failed_collection_publication.json').exists()


def test_description_only_does_not_load_modules(monkeypatch,capsys):
    monkeypatch.setattr(D,'modules',lambda:pytest.fail('unexpected scientific import'))
    assert D.main([])==0 and json.loads(capsys.readouterr().out)['no_scientific_execution'] is True


def test_conflicting_binding_rejected(tmp_path):
    path=tmp_path/'input';bindings={str(path):'a'*64}
    with pytest.raises(ValueError,match='Conflicting'):D.bind_inputs(bindings,{str(path):'b'*64})
    with pytest.raises(ValueError):D.bind_inputs({}, {'relative':'a'*64})
