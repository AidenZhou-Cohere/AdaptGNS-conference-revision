"""Tiny CPU/mocked process tests only; no official arrays, CUDA or remote host."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from test_train_goop3d_graph_support_cuda_v2 import synthetic, entry

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('goop3d_capacity',HERE/'measure_goop3d_vectorized_capacity_v1.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
UUIDS=[f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(4)]


def test_default_inert(monkeypatch):
    monkeypatch.setattr(M,'configure',lambda:(_ for _ in ()).throw(AssertionError('no import')))
    with contextlib.redirect_stdout(io.StringIO()) as out:assert M.main([])==0
    value=json.loads(out.getvalue());assert value['scientific_endpoint_selected'] is False and value['updates_per_job']==512


def test_original_lifecycle_private_and_fixed_4_plus_2():
    M.configure()
    assert M.sha(M.B.__file__)==M.LIFECYCLE_SHA
    assert Path(M.B.run_wave.__code__.co_filename).name=='measure_sand_cuda_capacity_v2.py'
    assert M.B.SCHEDULE==M.SCHEDULE and M.B.verify_job is M.verify_job and M.B.fixed_command is M.fixed_command
    assert [sum(j['wave']==w for j in M.SCHEDULE) for w in ('A','B')]==[4,2]
    assert M.V.DATA_PINS['train_manifest']=='0f0ce1802e202b9faaf86f0a6ce059c286ed454ceb53273cb926980614b0f864'


def test_fixed_worker_command_has_no_scientific_endpoint_or_resume(tmp_path):
    a=SimpleNamespace(**{k:tmp_path/k for k in (*M.DATA_FIELDS,'repo','release','output_dir','python','protocol','numerical_report')})
    c=M.fixed_command(a,M.SCHEDULE[0])
    assert c[c.index('--mode')+1]=='worker' and c[c.index('--job')+1]=='base_seed0'
    assert not any(k in c for k in ('--resume','--updates','--stop-after','--admission'))
    assert 'train_goop3d_graph_support_cuda_v2.py' not in c


def fixture_setup(synthetic,tmp_path,monkeypatch,arm='base'):
    s=synthetic;M.configure();monkeypatch.setattr(M,'T',entry);monkeypatch.setattr(M,'H',s.helpers)
    monkeypatch.setattr(M,'STOP',3);monkeypatch.setattr(M,'WARMUP',1)
    job=next(j for j in M.SCHEDULE if j['arm']==arm and j['seed']==0)
    a=SimpleNamespace(**{k:tmp_path/'data'/k for k in M.DATA_FIELDS},
        **{k:tmp_path/k for k in ('release','protocol','numerical_report','python')},
        repo=s.repo,output_dir=tmp_path/'jobs'/job['id'],job=job['id'],mode='worker')
    for k in (*M.DATA_FIELDS,'release','protocol','numerical_report','python'):
        p=getattr(a,k);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    hashes={k:M.sha(p) for k,p in M.input_paths(a).items()}
    monkeypatch.setattr(M.V,'DATA_PINS',{k:hashes[k] for k in M.DATA_FIELDS})
    info={'trajectory_ids':['train:000000'],'scope':'tiny CPU synthetic'}
    monkeypatch.setattr(M.V,'load_modules',lambda repo:(entry,s.helpers,None,s.support))
    monkeypatch.setattr(M.V,'load_data',lambda *args:([None]*295,{},info))
    monkeypatch.setattr(M.V,'verify_data_evidence',lambda *args:None)
    monkeypatch.setattr(M.V,'make_model',lambda h,metadata,device:s.make(0))
    runtime={'device':f"cuda:{job['gpu']}",'uuid':UUIDS[job['gpu']], 'torch':'2.13.0+cu129','cuda':'12.9','threads':2,
        'name':'synthetic GB200','float32_matmul_precision':'highest',
        'deterministic_algorithms':True,'deterministic_warn_only':False,'cublas_workspace_config':':4096:8',
        'tf32':False,'amp':False,'compile':False,'ddp':False}
    monkeypatch.setattr(entry,'configure_cuda',lambda *args:('cpu',runtime))
    monkeypatch.setattr(entry,'unpack_batch',lambda *args:s.batch)
    monkeypatch.setattr(s.helpers,'sample_indices',lambda *args:[0,1])
    monkeypatch.setattr(s.helpers,'frame_identity',lambda train,ids,i:f'train:000000:{6+i}')
    monkeypatch.setattr(s.helpers.data_loader,'load_manifest_data',lambda *args,**kw:[])
    monkeypatch.setattr(s.torch.cuda,'device',lambda device:contextlib.nullcontext())
    monkeypatch.setattr(s.torch.cuda,'manual_seed',lambda seed:None)
    (tmp_path/'launch.json').write_text(json.dumps({'schema':M.SCHEMA,'purpose':M.PURPOSE,'schedule':M.SCHEDULE,
        'environment':M.ENVIRONMENT,'files_sha256':hashes,'release_sha256':M.sha(a.release),'repo':str(a.repo.resolve())}))
    return s,a,hashes,{'gpu_uuids':UUIDS},job


@pytest.mark.parametrize('arm',['base','mix'])
def test_tiny_worker_serializes_separate_capacity_payload_and_audits(synthetic,tmp_path,monkeypatch,arm):
    s,a,hashes,release,job=fixture_setup(synthetic,tmp_path,monkeypatch,arm)
    with contextlib.redirect_stdout(io.StringIO()) as out:M.worker(a,release,hashes)
    (tmp_path/'stdout.jsonl').write_text(out.getvalue())
    history=M.read(a.output_dir/'history.json');status=M.read(a.output_dir/'status.json');config=M.read(a.output_dir/'protocol.json')
    assert status['state']=='complete_capacity_only' and status['completed_steps']==3
    assert not (a.output_dir/'run.lock').exists()
    final=s.torch.load(a.output_dir/'checkpoint-000000003.pt',map_location='cpu',weights_only=True)
    assert final['schema']==M.CHECKPOINT_SCHEMA and final['probe_lr_horizon']==100000
    assert final['purpose']=='bounded_capacity_only_never_promote' and not final['scientific_training_admitted']
    assert 'run_config' not in final and 'training_config' not in final
    assert final['optimizer_state']['param_groups'][0]['lr']==entry.learning_rate(2,100000)
    assert final['optimizer_state']['param_groups'][0]['lr']!=1e-5
    original=s.torch.load(a.output_dir/'checkpoint-000000000.pt',map_location='cpu',weights_only=True)
    assert original['completed_steps']==0 and not original['optimizer_state']['state']
    pointer={'path':'checkpoint-000000000.pt','sha256':M.sha(a.output_dir/'checkpoint-000000000.pt'),
             'completed_steps':0,'capacity_config_sha256':entry.config_hash(config)}
    external={'exit_code':0,'elapsed_seconds':100.,'pid':status['process']['pid'],'initial_pointer':pointer,'stdout_file':str(tmp_path/'stdout.jsonl')}
    manifest={'metadata':{},'records':[{'id':'train:000000','positions':{'shape':[301,8,3]}}]}
    audit=M.verify_job(a.output_dir,job,external,manifest,hashes['protocol'],UUIDS[job['gpu']])
    assert audit['initial']['all_Adam_steps_exact'] and audit['final']['all_Adam_steps_exact']
    assert len(audit['pairing_rows'])==3
    with pytest.raises(ValueError,match='Checkpoint lineage/configuration/endpoint mismatch'):
        model,opt=s.make();entry.restore_payload(s.helpers,final,model,opt,config,'cpu')


def test_worker_failure_keeps_initial_checkpoint_context_and_stops(synthetic,tmp_path,monkeypatch):
    s,a,hashes,release,job=fixture_setup(synthetic,tmp_path,monkeypatch)
    calls=[];original=s.support.forward_batch
    def fail(*args):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('synthetic graph failure')
        return original(*args)
    monkeypatch.setattr(s.support,'forward_batch',fail)
    with pytest.raises(RuntimeError,match='synthetic graph'):
        with contextlib.redirect_stdout(io.StringIO()):M.worker(a,release,hashes)
    assert len(calls)==2 and M.read(a.output_dir/'status.json')['state']=='failed'
    assert M.read(a.output_dir/'latest.json')['completed_steps']==0
    assert (a.output_dir/'checkpoint-000000000.pt').exists() and not (a.output_dir/'checkpoint-000000003.pt').exists()
    failure=M.read(a.output_dir/'failed_attempt.json')
    assert failure['completed_steps']==1 and failure['current']['completed_before']==1
    preserved=s.torch.load(a.output_dir/'unsuccessful_state.pt',map_location='cpu',weights_only=True)
    assert preserved['completed_steps']==1 and preserved['purpose']=='unsuccessful_capacity_state_never_promote'
    assert preserved['scientific_training_admitted'] is False and preserved['current']==failure['current']
    assert preserved['history']==M.read(a.output_dir/'history.json')
    assert preserved['optimizer_state']['state'] and set(preserved['rng_states'])=={'cpu','cuda'}
    assert set(preserved['gradients'])==set(preserved['state_dict'])
    assert not (a.output_dir/'run.lock').exists()


@pytest.mark.parametrize('change',['schema','purpose','horizon','model_dtype','optimizer_step','history','rng'])
def test_malformed_checkpoint_refused(synthetic,tmp_path,monkeypatch,change):
    s,a,hashes,release,job=fixture_setup(synthetic,tmp_path,monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):M.worker(a,release,hashes)
    path=a.output_dir/'checkpoint-000000003.pt';payload=s.torch.load(path,map_location='cpu',weights_only=True)
    config=M.read(a.output_dir/'protocol.json');history=M.read(a.output_dir/'history.json')
    if change=='schema':payload['schema']='adaptgns_goop3d_graph_support_cuda_training_v2'
    elif change=='purpose':payload['purpose']='scientific'
    elif change=='horizon':payload['probe_lr_horizon']=3
    elif change=='model_dtype':
        key=next(k for k,v in payload['state_dict'].items() if v.is_floating_point());payload['state_dict'][key]=payload['state_dict'][key].double()
    elif change=='optimizer_step':next(iter(payload['optimizer_state']['state'].values()))['step'].fill_(2)
    elif change=='history':payload['history']['graph_updates'].pop()
    else:payload['rng_states']['cuda']=s.torch.ones(3,dtype=s.torch.float32)
    bad=tmp_path/'bad.pt';s.torch.save(payload,bad)
    with pytest.raises((ValueError,FloatingPointError)):
        M.verify_checkpoint(bad,config,3,history,{})


def test_pairing_requires_all_rows_and_same_initial_state():
    M.configure()
    rows=[{'completed_steps':i+1,'frame_ids':['train:000000:6','train:000000:7'],'particles':16,'lr':1e-4} for i in range(512)]
    examples=[{'example_slot':i,'n_particles':8,'exposure_coin':.1,'coin_seed_material':'seed',
        'pair_seed_material':'pair','native_directed_edges':24,'native_self_edges':8,'receivers_above_native_cap':0,
        'annulus_pairs':12,'optional_budget_if_exposed':2,'native_edge_sha256':'c'*64,'noisy_current_sha256':'d'*64} for i in range(2)]
    jobs=[dict(arm=a,seed=0,initial={'model':'same','rng':'same'},pairing_rows=copy.deepcopy(rows),
        graph_rows=[{'noise_sha256':'a'*64,'examples':copy.deepcopy(examples)} for _ in rows]) for a in ('base','mix')]
    assert M.verify_pairing(jobs)[0]['all512_sample_noise_lr_rows_exact']
    for key in ('initial','pairing_rows','graph_rows'):
        bad=copy.deepcopy(jobs)
        if key=='initial':bad[1][key]['model']='other'
        elif key=='pairing_rows':bad[1][key][-1]['lr']=1e-5
        else:bad[1][key][-1]['noise_sha256']='b'*64
        with pytest.raises(ValueError):M.verify_pairing(bad)
    with pytest.raises(ValueError):M.verify_pairing(jobs[:1])
    for key in examples[0]:
        bad=copy.deepcopy(jobs);bad[1]['graph_rows'][-1]['examples'][0][key]='different'
        with pytest.raises(ValueError,match='Paired native graph'):M.verify_pairing(bad)


def test_failed_wave_blocks_second_wave(monkeypatch):
    M.configure();calls=[]
    def bad(wave):calls.append(wave);return [],{'state':'failed'}
    with pytest.raises(ValueError):M.B.run_waves(bad)
    assert calls==['A']


def release_setup(tmp_path,monkeypatch):
    M.configure()
    args=SimpleNamespace(**{k:tmp_path/k for k in (*M.DATA_FIELDS,'release','protocol','numerical_report','python')},
        repo=HERE.parents[2]/'outputs'/'AdaptGNS',output_dir=tmp_path/'capacity',mode='supervise')
    for k in (*M.DATA_FIELDS,'release','protocol','numerical_report','python'):getattr(args,k).write_text('{}')
    manifest={'dataset':'Goop-3D','split':'train','record_count':1000,'records':[
        {'id':f'train:{i:06d}','positions':{'shape':[301,1000+i%17,3]}} for i in range(1000)]}
    args.train_manifest.write_text(json.dumps(manifest))
    monkeypatch.setattr(M.V,'DATA_PINS',{k:M.sha(getattr(args,k)) for k in M.DATA_FIELDS})
    # The frozen complete-source validator is independently tested. Isolate this harness's release gate.
    monkeypatch.setattr(M.T,'validate_manifest_contract',lambda *args:None)
    report={'schema':M.V.SCHEMA,'status':'implementation_passed','scientific_training_admitted':False,
        'scientific_endpoint_selected':False,'test_accessed':False,'prospective_batches_saved_before_cuda':True,
        'source_sha256':M.VALIDATOR_SHA,'trainer_sha256':M.TRAINER_SHA,'candidate_graph_sha256':M.GRAPH_SHA,
        'original_graph_sha256':M.V.ORIGINAL_SHA,'all_inputs_reverified':True,'schedule':M.V.expected_schedule(manifest),
        'probe_lr_horizon':M.PROBE_LR_HORIZON,'attempted_update':None,'completed_optimizer_calls':26,
        'replay':{a:{'passed':True,'final_payload_exact':True,'checkpoint_serialization_exact':True} for a in M.V.ARMS},
        'cases':[{'case':c,'arm':a,'steps':[{'completed_steps':s,'passed':True,'bytewise_component_checks':{'model':True}} for s in (1,2)],
            'graph_gates':[{'passed':True,'edges_exact':True,'ledger_exact':True,'rng_unchanged':True,
                'graph_ledger':[{'selected_optional_pairs':1 if a=='mix' else 0}]} for _ in range(5 if c=='small' else 4)]}
            for c in M.V.CASES for a in M.V.ARMS],
        'runtime':{'torch':'2.13.0+cu129','cuda':'12.9','name':'synthetic GB200','threads':2,
            'deterministic_algorithms':True,'deterministic_warn_only':False,'cublas_workspace_config':':4096:8',
            'float32_matmul_precision':'highest','tf32':False,'amp':False,'compile':False,'ddp':False}}
    args.numerical_report.write_text(json.dumps(report))
    release={'schema':M.RELEASE_SCHEMA,'status':'admitted_for_capacity_only','issued_by':'root','purpose':M.PURPOSE,
        'scientific_training_admitted':False,'scientific_endpoint_selected':False,
        'files_sha256':{k:M.sha(p) for k,p in M.input_paths(args).items()},'schedule':M.SCHEDULE,'environment':M.ENVIRONMENT,
        'updates_per_job':M.STOP,'warmup_updates':M.WARMUP,'probe_lr_horizon':M.PROBE_LR_HORIZON,
        'numerical_report_independently_reviewed':True,'review_rationale':'synthetic gate test',
        'gpu_uuids':UUIDS,'hostname':M.socket.gethostname(),'process_identity_checked_utc':datetime.now(timezone.utc).isoformat(),
        'data_contract':{'scope':'bounded_implementation_data_only','scientific_training_admitted':False}}
    return args,release,report


def test_exact_capacity_release_passes_without_importing_science(tmp_path,monkeypatch):
    args,release,_=release_setup(tmp_path,monkeypatch)
    paths,hashes=M.validate_release(args,release)
    assert hashes==release['files_sha256'] and paths['numerical_report']==args.numerical_report
    assert M.H is None


@pytest.mark.parametrize('change',['schema','scientific','endpoint','updates','horizon','schedule','environment','uuid_prefix',
    'duplicate_uuid','hostname','unreviewed','stale','data_scope','data_endpoint','source_bytes'])
def test_release_refuses_wrong_scope_source_host_schedule_or_stale_inventory(tmp_path,monkeypatch,change):
    args,release,_=release_setup(tmp_path,monkeypatch)
    if change=='schema':release['schema']='scientific'
    elif change=='scientific':release['scientific_training_admitted']=True
    elif change=='endpoint':release['scientific_endpoint_selected']=True
    elif change=='updates':release['updates_per_job']=513
    elif change=='horizon':release['probe_lr_horizon']=512
    elif change=='schedule':release['schedule']=release['schedule'][:-1]
    elif change=='environment':release['environment']={}
    elif change=='uuid_prefix':release['gpu_uuids']=[x.removeprefix('GPU-') for x in UUIDS]
    elif change=='duplicate_uuid':release['gpu_uuids']=UUIDS[:3]+UUIDS[:1]
    elif change=='hostname':release['hostname']='wrong-host'
    elif change=='unreviewed':release['numerical_report_independently_reviewed']=False
    elif change=='stale':release['process_identity_checked_utc']=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
    elif change=='data_scope':release['data_contract']['scope']='scientific'
    elif change=='data_endpoint':release['data_contract']['prospective_endpoint_updates']=100000
    else:args.protocol.write_text('changed bytes')
    with pytest.raises(ValueError):M.validate_release(args,release)


@pytest.mark.parametrize('change',['incomplete','case_missing','case_duplicate','failed_step','failed_component','failed_graph',
    'gate_missing','vacuous','unserialized','replay_failed','schedule','runtime','prospective','test','science'])
def test_report_gate_refuses_partial_vacuous_or_mismatched_cuda_proof(tmp_path,monkeypatch,change):
    args,release,report=release_setup(tmp_path,monkeypatch)
    if change=='incomplete':report['completed_optimizer_calls']=25
    elif change=='case_missing':report['cases'].pop()
    elif change=='case_duplicate':report['cases'][-1]=copy.deepcopy(report['cases'][0])
    elif change=='failed_step':report['cases'][0]['steps'][0]['passed']=False
    elif change=='failed_component':report['cases'][0]['steps'][0]['bytewise_component_checks']['model']=False
    elif change=='failed_graph':report['cases'][0]['graph_gates'][0]['rng_unchanged']=False
    elif change=='gate_missing':report['cases'][0]['graph_gates'].pop()
    elif change=='vacuous':
        for row in report['cases']:
            for gate in row['graph_gates']:
                for e in gate['graph_ledger']:e['selected_optional_pairs']=0
    elif change=='unserialized':report['replay']['mix']['checkpoint_serialization_exact']=False
    elif change=='replay_failed':report['replay']['mix']['final_payload_exact']=False
    elif change=='schedule':report['schedule'][0]['target_frames']=[6,151]
    elif change=='runtime':report['runtime']['tf32']=True
    elif change=='prospective':report['prospective_batches_saved_before_cuda']=False
    elif change=='test':report['test_accessed']=True
    else:report['scientific_training_admitted']=True
    args.numerical_report.write_text(json.dumps(report));release['files_sha256']['numerical_report']=M.sha(args.numerical_report)
    with pytest.raises(ValueError):M.validate_release(args,release)


def test_worker_audit_binds_all_inputs_to_supervisor_release(synthetic,tmp_path,monkeypatch):
    _,a,hashes,release,job=fixture_setup(synthetic,tmp_path,monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):M.worker(a,release,hashes)
    launch=M.read(tmp_path/'launch.json');launch['files_sha256']['numerical_report']='0'*64
    (tmp_path/'launch.json').write_text(json.dumps(launch))
    with pytest.raises(ValueError,match='supervisor release/input binding'):
        M.verify_job(a.output_dir,job,{'exit_code':0,'elapsed_seconds':100.},{},hashes['protocol'],UUIDS[0])


def test_supervisor_preserves_failure_and_does_not_admit_or_advance(tmp_path,monkeypatch):
    args,release,_=release_setup(tmp_path,monkeypatch)
    monkeypatch.setattr(M.sys,'platform','linux')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    args.train_manifest=tmp_path/'data'/'train.json';args.train_manifest.parent.mkdir();args.train_manifest.write_text('{}')
    paths=M.input_paths(args);hashes={k:M.sha(p) for k,p in paths.items()}
    def fail(*args):raise RuntimeError('synthetic worker failure')
    monkeypatch.setattr(M.B,'run_wave',fail)
    with pytest.raises(RuntimeError,match='synthetic worker failure'):M.supervise(args,release,paths,hashes)
    status=M.read(args.output_dir/'status.json')
    assert status['state']=='failed_capacity_only' and status['all_existing_outputs_retained']
    assert not status['wave_B_released'] and not status['scientific_training_admitted'] and not status['scientific_endpoint_selected']
    assert (args.output_dir/'launch.json').exists() and not (args.output_dir/'summary.json').exists()
