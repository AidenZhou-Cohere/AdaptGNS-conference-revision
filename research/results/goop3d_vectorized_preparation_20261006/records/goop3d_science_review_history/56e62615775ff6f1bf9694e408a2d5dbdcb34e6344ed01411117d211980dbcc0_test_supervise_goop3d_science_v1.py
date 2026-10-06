"""Small CPU models, scalar receipts and mocked processes; no actual CUDA/data."""
import contextlib
import copy
from datetime import datetime,timedelta,timezone
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from test_train_goop3d_graph_support_cuda_v2 import synthetic
import supervise_goop3d_science_v1 as M


def args_for(root,updates=37):
    return SimpleNamespace(**{k:root/k for k in M.FIELDS},release=root/'release.json',output_dir=root/'out',
        updates=updates,checkpoint_every=10,log_every=5)


@pytest.fixture(autouse=True)
def configured(tmp_path):M.configure(args_for(tmp_path))


def test_default_has_no_numerical_import_or_process_launch():
    code="""import builtins,runpy,sys
old=builtins.__import__
def restrict(name,*a,**k):
 if name.split('.')[0] in ('torch','numpy','scipy','gns','research'):raise AssertionError(name)
 return old(name,*a,**k)
builtins.__import__=restrict
sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    r=subprocess.run([sys.executable,'-I','-c',code,M.__file__],capture_output=True,text=True,check=True)
    out=json.loads(r.stdout);assert out['endpoint_updates'] is None and not out['scientific_training_admitted']


@pytest.mark.parametrize('endpoint',[2,37,999,100000])
def test_dynamic_endpoint_commands_checkpoint_and_log_cadence(tmp_path,endpoint):
    a=args_for(tmp_path,endpoint)
    for job in M.SCHEDULE:
        c=M.fixed_command(a,job)
        assert c[c.index('--updates')+1]==str(endpoint)
        assert not any(x in c for x in ('--resume','--stop-after','--clear-stale-lock'))
        assert c[c.index('--objective')+1]=='faithful'
    assert M.checkpoint_steps(endpoint,10)[-1]==endpoint and M.checkpoint_steps(endpoint,10)[0]==0
    assert M.logged_steps(endpoint,5)[-1]==endpoint and M.logged_steps(endpoint,5)[0]==1
    assert len(M.checkpoint_steps(endpoint,10))==len(set(M.checkpoint_steps(endpoint,10)))


def planning_fixture(tmp_path):
    a=args_for(tmp_path);M.configure(a)
    for k in M.FIELDS:
        if k!='repo':getattr(a,k).write_text('{}')
    capacity={'schema':'adaptgns_goop3d_vectorized_capacity_v1','status':'all_six_verified_capacity_only',
        'scientific_training_admitted':False,'scientific_endpoint_selected':False,
        'jobs':[{**j,'status':'verified_capacity_only','final_pointer':{'completed_steps':512,'sha256':str(j['seed'])*64},
                 'steady_wall_seconds_per_update':.01,'nonnegative_external_minus_all_guarded_seconds':1.} for j in M.SCHEDULE],
        'pairing':[{'seed':s,'initial_model_rng_Adam_exact':True,'all512_sample_noise_lr_rows_exact':True} for s in range(3)],
        'q4_q2':{'A':.01,'B':.01},'r4_r2':{'A':1.,'B':1.}}
    a.capacity_summary.write_text(json.dumps(capacity))
    planning={'schema':'adaptgns_goop3d_deadline_planning_config_v1','selected_endpoint':None,'evaluation_overlap_credit':False,
        'candidate_endpoints':[37,1000],'checkpoint_every':10,'runtime_multiplier':1.35,
        'transfer_review_analysis_reserve_seconds':600,'planning_start_utc':'2026-10-06T00:00:00+00:00'}
    a.planning_config.write_text(json.dumps(planning))
    timing=[{'arm':arm,'mode':mode,'per_unit_seconds':.01,'setup_and_other_seconds':1.} for arm in ('base','mix') for mode in M.W.MODES]
    worksheet={'schema':M.W.SCHEMA,'status':'engineering_worksheet_complete_not_execution_admission',
        'scientific_endpoint_selected':False,'scientific_training_admitted':False,'test_accessed':False,'timing':timing,
        'input_sha256':{str(getattr(a,k).resolve()):M.sha(getattr(a,k)) for k in ('capacity_summary','planning_config')},
        'forecasts':M.W.forecast(planning,{'A':.01,'B':.01},{'A':1.,'B':1.},timing)}
    a.worksheet.write_text(json.dumps(worksheet))
    plan={'schema':'adaptgns_goop3d_prospective_endpoint_plan_v1','status':'prospectively_selected','issued_by':'root',
        'endpoint_updates':37,'checkpoint_every':10,'log_every':5,'selection_basis':M.T.ENDPOINT_BASIS,
        'selection_used_validation_or_test_accuracy':False,'rationale':'synthetic timing-only test',
        'cost_basis':'measured_full_horizon_engineering_estimate','runtime_multiplier':1.35,
        'evaluation_and_analysis_reserve_seconds':3600,'wave_audit_reserve_seconds':{'A':120,'B':120}}
    a.endpoint_plan.write_text(json.dumps(plan))
    return a,plan,worksheet,planning


def hashes(a):return {k:M.sha(v) for k,v in M.input_paths(a).items() if v.exists() and v.is_file()}


def test_measured_plan_recomputes_complete_forecast_and_wave_cost(tmp_path):
    a,plan,w,p=planning_fixture(tmp_path)
    result,cost=M.validate_plan(a,hashes(a));assert result==plan
    assert cost=={w:1.35*(37*.01+5*1.)+120 for w in ('A','B')}


@pytest.mark.parametrize('change',['wrong_endpoint','accuracy','incomplete','unfit','arithmetic','reserve','multiplier','checkpoint','byte_lineage'])
def test_bad_measured_plan_rejected(tmp_path,change):
    a,plan,w,p=planning_fixture(tmp_path)
    if change=='wrong_endpoint':plan['endpoint_updates']=38
    elif change=='accuracy':plan['selection_used_validation_or_test_accuracy']=True
    elif change=='incomplete':w['timing'].pop()
    elif change=='unfit':w['forecasts'][0]['fits_stated_planning_assumptions']=False
    elif change=='arithmetic':w['forecasts'][0]['projected_total_seconds']=0
    elif change=='reserve':plan['evaluation_and_analysis_reserve_seconds']=1
    elif change=='multiplier':plan['runtime_multiplier']=1
    elif change=='checkpoint':plan['checkpoint_every']=100
    else:w['input_sha256'][str(a.capacity_summary.resolve())]='f'*64
    a.endpoint_plan.write_text(json.dumps(plan));a.worksheet.write_text(json.dumps(w))
    with pytest.raises((ValueError,KeyError)):M.validate_plan(a,hashes(a))


def test_operational_quota_is_explicit_failed_timing_branch(tmp_path):
    a,plan,w,p=planning_fixture(tmp_path)
    plan.update(cost_basis='fixed_operational_quota_after_failed_full_horizon_timing',
        all_required_evaluation_outcomes_promised=False,full_study_measured_runtime_claim=False,
        complete_full_horizon_timing_attempt_failed=True,preserved_failed_timing_sha256=['a'*64])
    w.update(status='forecast_unavailable',forecasts=[],unavailable_reasons=[{'reason':'Synthetic incomplete fullH timing'}])
    a.endpoint_plan.write_text(json.dumps(plan));a.worksheet.write_text(json.dumps(w))
    assert M.validate_plan(a,hashes(a))[0]['full_study_measured_runtime_claim'] is False
    for key in ('complete_full_horizon_timing_attempt_failed','preserved_failed_timing_sha256','full_study_measured_runtime_claim'):
        bad=dict(plan);bad.pop(key);a.endpoint_plan.write_text(json.dumps(bad))
        with pytest.raises(ValueError):M.validate_plan(a,hashes(a))


def release_fixture(tmp_path,monkeypatch):
    a,plan,worksheet,planning=planning_fixture(tmp_path)
    now=datetime(2026,10,6,tzinfo=timezone.utc)
    uuid=[f'GPU-00000000-0000-0000-0000-{i:012x}' for i in range(4)]
    process={'schema':'adaptgns_goop3d_scientific_process_check_v1','issued_by':'root','hostname':M.socket.gethostname(),
        'gpu_uuids':uuid,'matching_training_processes':[],'gpu_processes':[],'checked_utc':now.isoformat()}
    clock={'schema':'adaptgns_goop3d_scientific_clock_check_v1','issued_by':'root','hostname':M.socket.gethostname(),
        'root_host_samples_reviewed':True,'clock_error_bound_seconds':1.,'checked_utc':now.isoformat()}
    a.process_check.write_text(json.dumps(process));a.clock_check.write_text(json.dumps(clock))
    h=hashes(a);a.admission.write_text(json.dumps({'scientific_training_admitted':True,
        'selection_evidence_sha256':[h[k] for k in ('capacity_summary','worksheet','endpoint_plan')]}));h=hashes(a)
    monkeypatch.setattr(M.T,'validate_prospective_endpoint',lambda *a:None)
    monkeypatch.setattr(M.T,'validate_manifest_contract',lambda *a:None)
    release={'schema':M.RELEASE_SCHEMA,'status':'admitted_for_scientific_training','issued_by':'root','scientific_training_admitted':True,
        'files_sha256':h,'schedule':M.SCHEDULE,'environment':M.ENVIRONMENT,'endpoint_updates':37,'hostname':M.socket.gethostname(),
        'cost_basis':plan['cost_basis'],'compute_analysis_deadline_utc':M.DEADLINE.isoformat(),'cohort_id':'synthetic37',
        'independent_source_and_feasibility_reviews_complete':True,'gpu_uuids':uuid,'clock_error_bound_seconds':1.,
        'process_identity_checked_utc':now.isoformat(),'clock_checked_utc':now.isoformat(),
        'waves':{'A':{'latest_start_utc':(now+timedelta(minutes=5)).isoformat(),'stop_utc':(now+timedelta(minutes=10)).isoformat()},
                 'B':{'latest_start_utc':(now+timedelta(minutes=11)).isoformat(),'stop_utc':(now+timedelta(minutes=16)).isoformat()}}}
    return a,h,release,process,clock,now


def test_exact_fresh_two_wave_root_release_passes(tmp_path,monkeypatch):
    a,h,r,p,c,n=release_fixture(tmp_path,monkeypatch)
    assert M.validate_release(a,r,h,now=n)[1]['A']['stop_utc']<M.DEADLINE


@pytest.mark.parametrize('field',['endpoint','host','uuid','scope','stale','foreign','clock','window','reserve','review'])
def test_unfit_or_wrong_process_release_refused(tmp_path,monkeypatch,field):
    a,h,r,p,c,n=release_fixture(tmp_path,monkeypatch)
    if field=='endpoint':r['endpoint_updates']=512
    elif field=='host':r['hostname']='other'
    elif field=='uuid':r['gpu_uuids'][3]=r['gpu_uuids'][0]
    elif field=='scope':r['scientific_training_admitted']=False
    elif field=='stale':p['checked_utc']=(n-timedelta(hours=1)).isoformat()
    elif field=='foreign':p['gpu_processes']=[{'pid':9}]
    elif field=='clock':c['clock_error_bound_seconds']=10
    elif field=='window':r['waves']['A']['stop_utc']=r['waves']['A']['latest_start_utc']
    elif field=='reserve':r['waves']['B']['stop_utc']=M.DEADLINE.isoformat()
    else:r['independent_source_and_feasibility_reviews_complete']=False
    a.process_check.write_text(json.dumps(p));a.clock_check.write_text(json.dumps(c))
    with pytest.raises(ValueError):M.validate_release(a,r,h,now=n)


def tiny_run(s,tmp_path,monkeypatch,arm='base',seed=0):
    a=args_for(tmp_path,3);a.repo=s.repo;a.checkpoint_every=2;a.log_every=2
    a.train_manifest=tmp_path/'data'/'train.json';a.train_manifest.parent.mkdir()
    M.configure(a);monkeypatch.setattr(M,'H',s.helpers)
    for k in M.FIELDS:
        if k not in ('repo','python'):getattr(a,k).write_text('{}')
    a.output_dir.mkdir();(a.output_dir/'jobs').mkdir()
    job=next(j for j in M.SCHEDULE if j['arm']==arm and j['seed']==seed)
    output=a.output_dir/'jobs'/job['id'];output.mkdir()
    h={'train_manifest':M.sha(a.train_manifest),'admission':M.sha(a.admission)}
    (a.output_dir/'launch.json').write_text(json.dumps({'files_sha256':h}))
    a.admission.write_text(json.dumps({'auxiliary_policy':{}}));h['admission']=M.sha(a.admission)
    (a.output_dir/'launch.json').write_text(json.dumps({'files_sha256':h}))
    data={'manifest_sha256':h['train_manifest'],'admission_sha256':h['admission'],'source':{},
        'trajectory_ids':['train:000000'],'n_trajectories':1000,'eligible_frames':295000,'frames_per_trajectory':301,
        'dimension':3,'particle_type_ids':[7],'forecast_horizon_after_six_frames':295,'context_source_sha256':M.T.CONTEXT_SOURCE_PINS,'auxiliary_policy':{}}
    admission={'auxiliary_policy':{}}
    for k in ('structural_report_sha256','converter_sha256','reader_sha256','metadata_sha256','acquisition_report_sha256','context_semantics_sha256','auxiliary_report_sha256','auxiliary_validator_sha256'):
        data[k]=admission[k]='a'*64
    data['structural_report_sha256']=admission['structural_report_sha256']=M.sha(a.structural_report)
    a.admission.write_text(json.dumps(admission));h['admission']=data['admission_sha256']=M.sha(a.admission)
    (a.output_dir/'launch.json').write_text(json.dumps({'files_sha256':h}))
    worker=SimpleNamespace(**{**vars(a),'output_dir':output},arm=arm,seed=seed,objective='faithful',resume=False)
    monkeypatch.setattr(M.T,'load_admitted_dataset',lambda *args:([None]*295,{},data))
    monkeypatch.setattr(M.T,'unpack_batch',lambda *args:s.batch)
    monkeypatch.setattr(M.T,'reverify_goop3d_evidence_and_auxiliaries',lambda *args:None)
    monkeypatch.setattr(s.helpers,'sample_indices',lambda *args:[0,1])
    monkeypatch.setattr(s.helpers,'frame_identity',lambda *args:f'train:000000:{6+args[-1]}')
    monkeypatch.setattr(s.helpers,'build_simulator',lambda *args,**kwargs:s.make(seed)[0])
    monkeypatch.setattr(s.helpers.data_loader,'load_manifest_data',lambda *args,**kwargs:[])
    monkeypatch.setattr(s.torch.cuda,'device',lambda *args:contextlib.nullcontext())
    monkeypatch.setattr(s.torch.cuda,'manual_seed',lambda *args:None)
    monkeypatch.setattr(M.V,'make_model',lambda *args:s.make(seed))
    runtime={'device':f"cuda:{job['gpu']}",'uuid':f'GPU-00000000-0000-0000-0000-{job["gpu"]:012x}',
        'torch':'2.13.0+cu129','cuda':'12.9','name':'synthetic GB200','threads':2,'deterministic_algorithms':True,
        'deterministic_warn_only':False,'float32_matmul_precision':'highest','cublas_workspace_config':':4096:8',
        'tf32':False,'amp':False,'compile':False,'ddp':False}
    with contextlib.redirect_stdout(io.StringIO()) as log:M.T.run_training(worker,s.helpers,s.support,{'pid':123},'cpu',runtime)
    out=tmp_path/'stdout.jsonl';out.write_text(log.getvalue())
    external={'exit_code':0,'signals':[],'pid':123,'elapsed_seconds':100.,'stdout_file':str(out),'initial_pointer':None}
    manifest={'source':{},'metadata':{},'records':[{'id':'train:000000','positions':{'shape':[301,8,3]}}]}
    return a,job,output,external,manifest,runtime


@pytest.mark.parametrize('arm',['base','mix'])
def test_tiny_fresh_scientific_endpoint_and_nonmultiple_checkpoint_audit(synthetic,tmp_path,monkeypatch,arm):
    a,j,out,e,manifest,runtime=tiny_run(synthetic,tmp_path,monkeypatch,arm)
    result=M.verify_job(out,j,e,manifest,M.sha(a.protocol),runtime['uuid'])
    assert result['status']=='verified_scientific_endpoint' and result['endpoint']['completed_steps']==3
    assert set(result['checkpoint_sha256'])=={'checkpoint-000000000.pt','checkpoint-000000002.pt','checkpoint-000000003.pt'}
    assert result['history']['logged_rows']==3 and not result['initial_pointer_observed_while_running']


@pytest.mark.parametrize('mutation',['nonfinite','adam','schema','configuration','rng'])
def test_bad_scientific_checkpoint_refused(synthetic,tmp_path,monkeypatch,mutation):
    a,j,out,e,manifest,runtime=tiny_run(synthetic,tmp_path,monkeypatch)
    path=out/'checkpoint-000000003.pt';p=synthetic.torch.load(path,weights_only=True)
    if mutation=='nonfinite':next(iter(p['state_dict'].values())).fill_(float('nan'))
    elif mutation=='adam':next(iter(p['optimizer_state']['state'].values()))['step'].fill_(2)
    elif mutation=='schema':p['schema']='bounded_capacity_only_never_promote'
    elif mutation=='configuration':p['run_config']['updates']=512
    else:p['rng_states']['cuda']=synthetic.torch.ones(3,dtype=synthetic.torch.float32)
    bad=tmp_path/'bad.pt';synthetic.torch.save(p,bad)
    with pytest.raises((ValueError,FloatingPointError)):
        M.check_checkpoint(bad,M.read(out/'protocol.json'),3,M.read(out/'history.json'),{})


def host_args(tmp_path):
    a=args_for(tmp_path)
    a.host_role='B';a.output_dir=tmp_path/'out';a.output_dir.mkdir()
    for n in ('jobs','logs'):(a.output_dir/n).mkdir()
    M.ACTIVE=a
    return a


def test_failed_identity_retains_registered_child_and_calls_owned_cleanup(tmp_path,monkeypatch):
    a=host_args(tmp_path);now=datetime(2026,10,6,tzinfo=timezone.utc)
    monkeypatch.setattr(M.B,'now',lambda:now);monkeypatch.setattr(M.B,'gpu_processes',lambda:[])
    monkeypatch.setattr(M.subprocess,'Popen',lambda *a,**k:SimpleNamespace(pid=123,returncode=None))
    monkeypatch.setattr(M.B,'process_identity',lambda pid:{'argv':['wrong'],'ppid':M.os.getpid()})
    calls=[]
    def cleanup(args,children,*rest):
        calls.append(len(children))
        for c in children:
            for h in c['handles']:h.close()
    monkeypatch.setattr(M.B,'cleanup_owned',cleanup)
    launch={'latest_start_utc':(now+timedelta(hours=1)).isoformat(),'clock_error_bound_seconds':1.}
    with pytest.raises(ValueError,match='identity'):M.run_wave(a,launch,{},now+timedelta(hours=2))
    assert calls==[1] and M.read(a.output_dir/'wave_B.json')['state']=='failed'


def test_cutoff_prevents_mock_launch(tmp_path,monkeypatch):
    a=host_args(tmp_path);now=datetime(2026,10,6,tzinfo=timezone.utc)
    monkeypatch.setattr(M.B,'now',lambda:now);monkeypatch.setattr(M.B,'gpu_processes',lambda:[])
    monkeypatch.setattr(M.subprocess,'Popen',lambda *a,**k:pytest.fail('launch forbidden'))
    monkeypatch.setattr(M.B,'cleanup_owned',lambda *a:None)
    with pytest.raises(ValueError):M.run_wave(a,{'latest_start_utc':now.isoformat(),'clock_error_bound_seconds':1.},{},now+timedelta(hours=1))


def test_all_children_reaped_before_endpoint_audit(tmp_path,monkeypatch):
    a=host_args(tmp_path);now=datetime(2026,10,6,tzinfo=timezone.utc);children=[];audits=[]
    monkeypatch.setattr(M.B,'now',lambda:now);monkeypatch.setattr(M.B,'gpu_processes',lambda:[])
    def popen(command,**kw):
        child=SimpleNamespace(pid=100+len(children),returncode=None,command=command);children.append(child);return child
    monkeypatch.setattr(M.subprocess,'Popen',popen)
    monkeypatch.setattr(M.B,'process_identity',lambda pid:{'argv':next(c.command for c in children if c.pid==pid),'ppid':M.os.getpid()})
    monkeypatch.setattr(M.B,'capture_initial',lambda *a:None)
    def reap(args,child,record,retained,launch,manifest):
        directory=args.output_dir/'jobs'/child['job']['id'];directory.mkdir()
        (directory/'status.json').write_text(json.dumps({'state':'complete','error':None,'completed_steps':37,'committed_steps':37,'requested_steps':37}))
        external={'exit_code':0,'signals':[]}
        retained.append(M.B.verify_job(directory,child['job'],external,manifest,'p','g'));assert not audits
        child['process'].returncode=0
        for h in child['handles']:h.close()
    monkeypatch.setattr(M.B,'reap_child',reap)
    def audit(directory,job,*args):
        assert all(c.returncode==0 for c in children);audits.append(job['id']);return job
    monkeypatch.setattr(M,'verify_job',audit);monkeypatch.setattr(M,'verify_pairing',lambda *a:[{'seed':2}])
    monkeypatch.setattr(M.time,'sleep',lambda *a:None)
    launch={'latest_start_utc':(now+timedelta(hours=1)).isoformat(),'clock_error_bound_seconds':1.,'files_sha256':{'protocol':'p'},'gpu_uuids':['g']*4}
    result,pairs=M.run_wave(a,launch,{},now+timedelta(hours=2))
    assert len(result)==2 and pairs==[{'seed':2}] and len(audits)==2


def test_failed_wave_cannot_make_candidate_cohort(tmp_path):
    a=args_for(tmp_path)
    with pytest.raises(ValueError,match='Full six-cell'):M.final_candidates(a,{},[],[])
    assert not a.output_dir.exists()


@pytest.mark.parametrize('change',['flag','bounded_scope'])
def test_bounded_admission_cannot_supply_scientific_scope(tmp_path,monkeypatch,change):
    a,h,r,p,c,n=release_fixture(tmp_path,monkeypatch)
    admission=M.read(a.admission)
    if change=='flag':admission['scientific_training_admitted']=False
    else:admission['scope']='bounded_implementation_data_only'
    a.admission.write_text(json.dumps(admission))
    with pytest.raises(ValueError,match='explicitly replace'):M.validate_release(a,r,h,now=n)


def test_inert_admission_template_has_exact_trainer_fields_after_explicit_root_substitution():
    admission=M.read(Path(M.__file__).with_name('goop3d_scientific_training_admission.template.json'))
    assert admission['status']=='not_admitted' and admission['issued_by']=='root_required'
    assert admission['prospective_endpoint_updates'] is None and admission['scientific_training_admitted'] is False
    admission.update(status='admitted',issued_by='root',prospective_endpoint_updates=37,
        scientific_training_admitted=True,checkpoint_every=10,log_every=5,selection_evidence_sha256=['a'*64])
    a=SimpleNamespace(updates=37,checkpoint_every=10,log_every=5,protocol=Path(M.__file__).with_name('goop3d_scientific_protocol_v1.md'))
    M.T.validate_prospective_endpoint(a,admission)
    M.T.validate_manifest_contract(M.read(Path(M.__file__).with_name('goop3d_structural_metadata_v1')/'train.json'),admission)
