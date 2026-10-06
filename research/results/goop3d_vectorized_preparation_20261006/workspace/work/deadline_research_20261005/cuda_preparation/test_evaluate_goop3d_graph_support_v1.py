"""Tiny synthetic3D models and fake authorization; no datasets/GPU/network."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from test_train_goop3d_graph_support_cuda_v2 import synthetic, entry as trainer

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_eval3d_test',HERE/'evaluate_goop3d_graph_support_v1.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


@pytest.fixture
def native(synthetic):
    n=M.load(HERE/'goop3d_native_evaluation_v1.py',M.sha(HERE/'goop3d_native_evaluation_v1.py'),'_native3d_test')
    return n


def test_description_has_no_numerical_import():
    code="""import builtins,runpy,sys
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','numpy','scipy','gns','research'):raise AssertionError(name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-I','-c',code,str(HERE/'evaluate_goop3d_graph_support_v1.py')],capture_output=True,text=True,check=True)
    assert json.loads(p.stdout)['horizon']==295


def records(count=100):
    return [{'id':f'valid:{i:06d}','source_index':i,'positions':{'shape':[301,300+i,3]}} for i in range(count)]


def test_grid_and_all_schedules_use_complete_source_order():
    assert M.grid_indices(100)==[i*99//29 for i in range(30)]
    assert M.grid_indices(7)==list(range(7))
    final=M.schedules(records(),'full-rollout','final_evaluation')
    assert len(final)==30 and final[0]['source_index']==0 and final[-1]['source_index']==99
    same=M.schedules(records(),'same-state','final_evaluation')
    assert len(same)==150 and [r['target_frame'] for r in same[:5]]==[7,80,153,226,300]
    clean=M.schedules(records(),'clean-validation','final_evaluation')
    assert len(clean)==128 and len({(r['source_index'],r['target_frame']) for r in clean})==128
    assert clean[-1]['source_index']==99 and clean[-1]['target_frame']==300
    timing=M.schedules(records(),'full-rollout','capacity_timing')
    assert len(timing)==3 and [r['source_index'] for r in timing]==[0,47,99]


@pytest.mark.parametrize('policy',M.POLICIES)
def test_three_forecast_3d_rollouts_keep_finite_shape_and_exact_budget(synthetic,native,policy):
    np,torch=synthetic.np,synthetic.torch;model,_=synthetic.make()
    points=synthetic.batch[0][:8,0].numpy()
    positions=np.broadcast_to(points,(9,8,3)).copy();positions[:,:,0]+=np.arange(9,dtype=np.float32)[:,None]*.0001
    row,traces=native.rollout(model,positions,np.full(8,7,dtype=np.int64),{'bounds':[[.1,.9]]*3,'default_connectivity_radius':.025},policy,3,123,'cpu',trace_steps=(1,3))
    assert row['status']=='complete' and row['completed_steps']==3 and len(row['mse_per_step'])==3
    assert traces['predicted_positions'].shape==(2,8,3)
    assert row['native_parity']['passed'] if 'native_parity' in row else row['initial_native_parity']['passed']
    for attempt in row['attempts']:
        if attempt['accepted']:
            assert attempt['native_base_prefix_preserved']
            assert attempt['directed_edges']==attempt['native_base_directed_edges']+2*attempt['retained_optional_pairs']


def test_candidate_and_edge_guards_are_preserved_as_named_failures(synthetic,native,monkeypatch):
    np=synthetic.np;points=np.full((4,3),.2,dtype=np.float32)
    monkeypatch.setattr(native,'MAX_PAIRS',1)
    with pytest.raises(native.full.RolloutGuard) as exc:native.strict_pairs(points,.025)
    assert exc.value.details['category']=='candidate_pair_resource_guard'
    monkeypatch.setattr(native,'MAX_EDGES',1)
    with pytest.raises(native.full.RolloutGuard) as exc:native.ordered_edges(points,np.array([[0,1]],dtype=np.int64),True)
    assert exc.value.details['category']=='directed_edge_resource_guard'


def test_3d_clean_variance_and_nll_use_three_coordinates(synthetic,native):
    np=synthetic.np;model,_=synthetic.make()
    helpers=SimpleNamespace(**vars(synthetic.helpers));helpers.unpack_batch=lambda examples:trainer.unpack_batch(helpers,examples)
    diag=M.load(HERE/'goop3d_diagnostic_metrics_v1.py',M.sha(HERE/'goop3d_diagnostic_metrics_v1.py'),'_diag3d_test')
    points=synthetic.batch[0][:8,0].numpy();positions=np.broadcast_to(points,(8,8,3)).copy()
    item={'source_index':0,'target_frame':6,'schedule_index':0,'trajectory_id':'valid:000000'}
    row,arrays=diag.clean_validation(native,helpers,model,positions,np.full(8,7,dtype=np.int64),item,'cpu')
    assert row['status']=='complete',row
    se=arrays['normalized_vector_se'];q=arrays['predicted_variance']
    assert row['metrics']['normalized_acceleration_coordinate_mse']==pytest.approx(se.mean()/3)
    assert row['metrics']['predicted_normalized_vector_se']==pytest.approx(3*q.mean())
    assert row['metrics']['constant_free_gaussian_nll']==pytest.approx((.5*se/q+1.5*np.log(q)).mean())


def test_same_state_3d_targets_cannot_change_graph_predictions(synthetic,native):
    np=synthetic.np;model,_=synthetic.make()
    diag=M.load(HERE/'goop3d_diagnostic_metrics_v1.py',M.sha(HERE/'goop3d_diagnostic_metrics_v1.py'),'_same3d_test')
    points=synthetic.batch[0][:8,0].numpy();positions=np.broadcast_to(points,(8,8,3)).copy()
    item={'source_index':0,'target_frame':7,'schedule_index':0,'trajectory_id':'valid:000000'}
    first,arrays=diag.same_state(native,model,positions,np.full(8,7,dtype=np.int64),{'bounds':[[.1,.9]]*3},item,'valid',0,'cpu')
    positions[-1]+=.1
    second,changed=diag.same_state(native,model,positions,np.full(8,7,dtype=np.int64),{'bounds':[[.1,.9]]*3},item,'valid',0,'cpu')
    assert first['status']==second['status']=='complete'
    for key in arrays:
        if key.endswith(('prediction','risk','edges','physical_selection_scores')):assert np.array_equal(arrays[key],changed[key]),key
    assert first['policies']['base']['metrics']!=second['policies']['base']['metrics']


def probe_payload():
    return {'schema':M.PROBE_SCHEMA,'purpose':'bounded_capacity_only_never_promote','scientific_training_admitted':False,
        'completed_steps':512,'probe_lr_horizon':100000,'arm':'base','seed':0,'trainer_sha256':M.TRAINER_SHA,
        'graph_sha256':M.GRAPH_SHA,'train_manifest_sha256':M.TRAIN_MANIFEST_SHA}


def test_timing_and_scientific_checkpoint_lineages_are_disjoint():
    args=SimpleNamespace(purpose='capacity_timing',arm='base',seed=0,trainer_source=HERE/'train_goop3d_graph_support_cuda_v2.py')
    payload=probe_payload();assert M.check_payload(payload,args,{},trainer)==100000
    args.purpose='final_evaluation';args.checkpoint_updates=512;args.protocol=HERE/'goop3d_first_record_benchmark_protocol_v1.md'
    with pytest.raises(ValueError):M.check_payload(payload,args,{},trainer)
    args.purpose='capacity_timing'
    for key,value in [('schema','adaptgns_goop_graph_support_cuda_training_v1'),('scientific_training_admitted',True),('completed_steps',100000),('graph_sha256','0'*64)]:
        with pytest.raises(ValueError):M.check_payload({**payload,key:value},args,{},trainer)


def test_cohort_gate_happens_before_any_test_file_hash(monkeypatch,tmp_path):
    # Fail the cohort intentionally; any attempt to hash a test file is a bug.
    cp=tmp_path/'cohort';cp.write_text('{}');audit=tmp_path/'audit';audit.write_text('{}');protocol=tmp_path/'protocol';protocol.write_text('synthetic')
    args=SimpleNamespace(purpose='final_evaluation',mode='full-rollout',split='test',arm='base',seed=0,checkpoint_sha256='a'*64,
        checkpoint_updates=123,max_seconds=900,cuda_index=0,gpu_uuid='synthetic-uuid',trainer_source=HERE/'train_goop3d_graph_support_cuda_v2.py',cohort=cp,cohort_audit=audit,protocol=protocol)
    release={'schema':'adaptgns_goop3d_evaluation_release_v1','status':'admitted_for_execution','issued_by':'root','purpose':args.purpose,
        'mode':args.mode,'split':args.split,'arm':args.arm,'seed':args.seed,'checkpoint_sha256':args.checkpoint_sha256,
        'checkpoint_updates':123,'graph_sha256':M.GRAPH_SHA,'evaluator_sha256':M.sha(M.__file__),
        'whole_invocation_outer_timeout_required':True,'max_seconds':900,'cuda_index':0,'gpu_uuid':args.gpu_uuid,'files_sha256':{str(cp):M.sha(cp),str(audit):M.sha(audit)}}
    path=tmp_path/'release.json';path.write_text(json.dumps(release));args.release=path
    monkeypatch.setattr(M,'cohort_gate',lambda *args:(_ for _ in ()).throw(ValueError('intentional cohort stop')))
    with pytest.raises(ValueError,match='intentional cohort stop'):M.release_gate(args)


def test_runtime_requires_released_physical_cuda_device():
    args=SimpleNamespace(cuda_index=2,gpu_uuid='GPU-synthetic-uuid')
    release={'gpu_uuid':args.gpu_uuid}
    M.runtime_gate(args,release,{'device':'cuda:2','uuid':'synthetic-uuid'})
    for runtime in ({'device':'cuda:0','uuid':'synthetic-uuid'},{'device':'cuda:2','uuid':'other'},{'device':'cpu','uuid':'synthetic-uuid'}):
        with pytest.raises(ValueError,match='physical CUDA'):M.runtime_gate(args,release,runtime)


def test_conflicting_alias_bindings_are_rejected(tmp_path):
    p=tmp_path/'input';p.write_text('before')
    with pytest.raises(ValueError,match='Conflicting'):M.merge_bindings({p:M.sha(p)},{str(p):'0'*64})


@pytest.mark.parametrize('mutation',[None,'numerical','common','context','auxiliary','release'])
def test_mock_driver_retains_failures_when_any_required_source_changes(synthetic,tmp_path,monkeypatch,mutation):
    # Exercise actual driver publication and terminal checks with tiny fake outputs.
    # Source admission/model work is isolated: no real split, checkpoint or CUDA.
    repo=tmp_path/'repo';repo.mkdir();data=tmp_path/'data';data.mkdir();models=tmp_path/'models';models.mkdir()
    inputs=tmp_path/'inputs';inputs.mkdir();output=tmp_path/'output'
    paths={name:inputs/name for name in ('release','split_admission','structural_report','acquisition_report','context_semantics','auxiliary_report','trainer_source','protocol')}
    paths.update(repo=repo,manifest=data/'manifest.json',checkpoint=models/'checkpoint.pt',output_dir=output)
    for key,path in paths.items():
        if key not in ('repo','output_dir'):path.write_text('{}')
    sourcepaths={name:repo/(name+'.py') for name in ('numerical','common','context','auxiliary')}
    for name,path in sourcepaths.items():path.write_text(name)
    numerical={sourcepaths['numerical'].name:M.sha(sourcepaths['numerical'])}
    monkeypatch.setattr(M,'NUMERICAL_SOURCE_PINS',numerical)
    release={'files_sha256':{str(path.resolve()):M.sha(path) for key,path in paths.items() if key not in ('repo','output_dir','release')},
        'evaluator_sha256':M.sha(M.__file__),'numerical_source_sha256':numerical,'gpu_uuid':'synthetic-uuid'}
    for name in ('goop3d_native_evaluation_v1.py','goop3d_diagnostic_metrics_v1.py'):
        release['files_sha256'][str(HERE/name)]=M.sha(HERE/name)
    h=SimpleNamespace(torch=synthetic.torch,np=synthetic.np,synchronize=lambda device:None)
    fake_trainer=SimpleNamespace(SCHEMA=M.TRAIN_SCHEMA,SOURCE_PINS={sourcepaths['common'].name:M.sha(sourcepaths['common'])},
        load_helpers=lambda repo:(h,None),configure_cuda=lambda h,args:(SimpleNamespace(type='cpu'),{'device':'cuda:0','uuid':'synthetic-uuid'}))
    calls=[]
    def rollout(*args,**kwargs):
        if not calls and mutation:
            (paths['release'] if mutation=='release' else sourcepaths[mutation]).write_text('changed after setup')
        calls.append(args)
        return {'status':'complete','completed_steps':295,'mse_per_step':[0.]*295}, {'prediction':synthetic.np.zeros((1,2,3),dtype='float32')}
    fake_native=SimpleNamespace(rollout=rollout)
    def fake_load(path,digest,name):
        return fake_trainer if Path(path)==paths['trainer_source'] else fake_native if Path(path).name=='goop3d_native_evaluation_v1.py' else SimpleNamespace()
    monkeypatch.setattr(M,'load',fake_load)
    monkeypatch.setattr(M,'release_gate',lambda args:release)
    record={'id':'valid:000000','source_index':0,'positions':{'shape':[301,2,3]}}
    extras={str(sourcepaths[k]):M.sha(sourcepaths[k]) for k in ('context','auxiliary')}
    monkeypatch.setattr(M,'check_split',lambda *args:({'records':[record],'metadata':{}},[(None,None)],extras))
    monkeypatch.setattr(M,'prepare_model',lambda *args:object())
    argv=['--execute','--purpose','capacity_timing','--mode','full-rollout','--split','valid','--arm','base','--seed','0',
        '--checkpoint-sha256',M.sha(paths['checkpoint']),'--checkpoint-updates','512','--gpu-uuid','synthetic-uuid']
    for key,path in paths.items():argv += ['--'+key.replace('_','-'),str(path)]
    if mutation:
        with pytest.raises(ValueError,match='Inputs changed'):M.main(argv)
        assert M.read(output/'status.json')['state']=='error'
        assert M.read(output/'failed_attempt.json')['all_outputs_retained'] is True
    else:
        assert M.main(argv)==0
        assert M.read(output/'status.json')['all_inputs_reverified'] is True
    assert len(calls)==6 and len(list(output.glob('trajectory*.npz')))==6
    bindings=M.read(output/'protocol.json')['input_files_sha256']
    assert all(str(path) in bindings for path in (*sourcepaths.values(),paths['release']))
