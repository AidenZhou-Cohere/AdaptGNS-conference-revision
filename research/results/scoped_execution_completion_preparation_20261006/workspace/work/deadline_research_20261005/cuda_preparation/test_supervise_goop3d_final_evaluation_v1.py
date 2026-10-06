"""Synthetic scalar contracts, opaque bytes and fake processes only; no CUDA/HTTP/model or process launch."""
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import sys
from types import SimpleNamespace
import pytest
import supervise_goop3d_final_evaluation_v1 as M
from test_summarize_goop3d_graph_support_v1 import manifest,rollout

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];REPO=ROOT/'outputs/AdaptGNS'


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(M.encode(value));return path


def fixture(tmp_path,monkeypatch):
    tmp_path=tmp_path.resolve();mods=M.modules();now=M.utc();python=str(Path(sys.executable))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8');monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    metadata=M.read(HERE.parent/'extension_feasibility_20261006/Goop-3D_metadata_json.json')
    valid=manifest(2,'valid');test=manifest(3,'test')
    for split,v in (('valid',valid),('test',test)):v.update(dataset='Goop-3D',split=split,metadata=metadata,metadata_sha256=mods.E.METADATA_SHA)
    a=SimpleNamespace(plan=HERE/'goop3d_stopped_timing_and_decision_v1/cuda_preparation/goop3d_scientific_endpoint_plan_root_v1.json',
        protocol=HERE/'goop3d_scientific_protocol_v1.md',trainer_source=HERE/'train_goop3d_graph_support_cuda_v2.py',
        valid_manifest=put(tmp_path/'data/valid.json',valid),test_manifest=put(tmp_path/'data/test.json',test),
        cohort=tmp_path/'cohort.json',cohort_audit=tmp_path/'audit.json',test_preflight=tmp_path/'test_preflight.json',
        release=tmp_path/'release.json',output_dir=tmp_path/'queue')
    monkeypatch.setattr(M,'VALID_SHA',M.sha(a.valid_manifest));monkeypatch.setattr(mods.C,'VALID_SHA',M.sha(a.valid_manifest))
    U=25000;models=[]
    for arm,seed in mods.C.MODELS:
        checkpoint=tmp_path/'models'/f'{arm}_{seed}.pt';checkpoint.parent.mkdir(exist_ok=True);checkpoint.write_bytes(f'not-a-model-synthetic-{arm}-{seed}'.encode())
        models.append({'arm':arm,'seed':seed,'objective':'faithful','completed_steps':U,'graph_history_updates':U,
            'checkpoint_sha256':M.sha(checkpoint),'checkpoint_path':str(checkpoint),**{k:True for k in
            ('all_optimizer_steps_equal_endpoint','all_state_and_moments_finite','source_data_protocol_verified','checkpoint_bytes_verified')}})
    common={'training_schema':mods.E.TRAIN_SCHEMA,'endpoint_updates':U,'protocol_sha256':M.PROTOCOL_SHA,'trainer_sha256':M.TRAINER_SHA,
        'graph_sha256':mods.E.GRAPH_SHA,'models':models,'issued_by':'root','evaluation_admitted':True,'cohort_id':'explicitly_synthetic_only'}
    audit={**common,'schema':'adaptgns_goop3d_graph_support_complete_cohort_audit_v2','status':'all_six_endpoints_and_pairing_verified',
        'paired_seeds':[{'seed':s,**{k:True for k in ('initial_model_tensor_identity','initial_cpu_cuda_rng_identity','initial_empty_adam_identity',
        'all_frame_noise_lr_schedules_equal','all_graph_budgets_and_rng_material_verified')}} for s in range(3)]}
    put(a.cohort_audit,audit);put(a.cohort,{**common,'schema':'adaptgns_goop3d_graph_support_final_cohort_v2','status':'frozen_for_final_evaluation',
        'cohort_audit_sha256':M.sha(a.cohort_audit),'freeze_adapter_sha256':M.FREEZE_SHA,'created_utc':(now-timedelta(minutes=1)).isoformat()})
    put(a.test_preflight,{'schema':'adaptgns_goop3d_reserved_test_preflight_v1','status':'frozen_split_and_evidence_contract_passed',
        'source_sha256':M.PREPARATION_SHA,'cohort_sha256':M.sha(a.cohort),'manifest_sha256':M.sha(a.test_manifest),'test_evaluation_executed':False,
        'evaluator_sha256':M.EVALUATOR_SHA,'source_order_grid':mods.E.schedules(test['records'],'full-rollout','final_evaluation'),
        'verified_evidence_sha256':{}})
    stubs={s:{k:put(tmp_path/'contracts'/f'{s}_{k}.json',{'synthetic_scalar_placeholder':k}) for k in
        ('split_admission','structural_report','acquisition_report','context_semantics','auxiliary_report','cross_split_audit')} for s in ('valid','test')}
    r={'schema':M.RELEASE_SCHEMA,'status':'approved_for_fixed_whole_invocation_quotas','issued_by':'root','issued_utc':now.isoformat(),
        'dataset':'Goop-3D','source_sha256':M.sha(M.__file__),'collector_sha256':M.COLLECTOR_SHA,'ledger_interface_sha256':M.INTERFACE_SHA,
        'hostname':socket.gethostname(),'environment':M.ENVIRONMENT,'python_environment':M.python_environment(python),'output_dir':str(a.output_dir),
        'whole_supervisor_outer_timeout_required':True,'all_six_models_and_both_splits_independently_admitted':True,'automatic_retry_or_resume':False,
        'cost_basis':M.COST_BASIS,'evaluation_activity_quota_seconds':21600,'analysis_reserve_seconds':3600,
        'all_required_outcomes_promised':False,'full_horizon_runtime_forecast_claim':False,
        'stage_order':['clean_validation','same_state_valid','same_state_test','full_rollout_valid','full_rollout_test'],
        'stage_quotas_seconds':{n:2 for n in M.STAGES},'cleanup_seconds_per_invocation':15,'supervisor_audit_reserve_seconds':60,
        'clock_error_bound_seconds':.1,'clock_checked_utc':now.isoformat(),'reserved_queue_seconds':230,
        'absolute_stop_utc':(now+timedelta(hours=1)).isoformat(),'latest_start_utc':(now+timedelta(minutes=5)).isoformat(),
        'cohort_sha256':M.sha(a.cohort),'cohort_audit_sha256':M.sha(a.cohort_audit),'endpoint_updates':U,
        'gpu_uuids':['GPU-'+f'{i:08d}'+'-1234-5678-9012-000000000000' for i in range(4)],'streams':[]}
    pins={str(HERE/n):d for n,d in M.SOURCE_PINS.items()};pins[str(Path(M.__file__).resolve())]=M.sha(M.__file__)
    pins.update({str(getattr(a,n)):M.sha(getattr(a,n)) for n in M.CONTROL_NAMES});pins.update(M.python_files(r['python_environment']))
    for wave,jobs in M.WAVES.items():
        for arm,seed,gpu in jobs:
            sid=f'{arm}_seed{seed}';model=next(m for m in models if (m['arm'],m['seed'])==(arm,seed));commands=[]
            for name in r['stage_order']:
                mode,split=M.STAGES[name];childfile=tmp_path/'children'/f'{sid}_{name}.json'
                options={'--purpose':'final_evaluation','--mode':mode,'--split':split,'--arm':arm,'--seed':str(seed),
                    '--checkpoint-sha256':model['checkpoint_sha256'],'--checkpoint-updates':str(U),'--cuda-index':str(gpu),
                    '--gpu-uuid':r['gpu_uuids'][gpu],'--threads':'2','--max-seconds':'2','--release':str(childfile),'--repo':str(REPO),
                    '--manifest':str(a.valid_manifest if split=='valid' else a.test_manifest),'--trainer-source':str(a.trainer_source),
                    '--protocol':str(a.protocol),'--checkpoint':model['checkpoint_path'],'--cohort':str(a.cohort),'--cohort-audit':str(a.cohort_audit),
                    '--output-dir':str(a.output_dir/'jobs'/sid/name)}
                options.update({'--'+k.replace('_','-'):str(p) for k,p in stubs[split].items() if k!='cross_split_audit' or split=='test'})
                command=[python,'-B','-u',str(HERE/'evaluate_goop3d_graph_support_v1.py'),'--execute']+[v for k,value in options.items() for v in (k,value)]
                child,_=M.child_namespace(command,mods.C)
                fields=('manifest','split_admission','structural_report','acquisition_report','context_semantics','auxiliary_report','trainer_source','protocol','checkpoint','cohort','cohort_audit')
                cpins={str(getattr(child,k)):M.sha(getattr(child,k)) for k in fields}
                if split=='test':cpins[str(child.cross_split_audit)]=M.sha(child.cross_split_audit)
                cpins.update({str(HERE/n):M.sha(HERE/n) for n in ('goop3d_native_evaluation_v1.py','goop3d_diagnostic_metrics_v1.py','goop3d_graph_support_vectorized_v1.py')})
                cr={'schema':'adaptgns_goop3d_evaluation_release_v1','issued_by':'root','status':'admitted_for_execution','purpose':'final_evaluation',
                    'mode':mode,'split':split,'arm':arm,'seed':seed,'checkpoint_sha256':model['checkpoint_sha256'],'checkpoint_updates':U,
                    'scientific_endpoint_updates':U,'graph_sha256':mods.E.GRAPH_SHA,'evaluator_sha256':M.EVALUATOR_SHA,
                    'whole_invocation_outer_timeout_required':True,'cuda_index':gpu,'gpu_uuid':r['gpu_uuids'][gpu],'max_seconds':2,
                    'files_sha256':cpins,'numerical_source_sha256':mods.E.NUMERICAL_SOURCE_PINS}
                put(childfile,cr);pins.update(cpins);pins[str(childfile)]=M.sha(childfile)
                pins.update({str(REPO/n):d for n,d in {**mods.E.NUMERICAL_SOURCE_PINS,**mods.T.SOURCE_PINS}.items()})
                commands.append(command)
            r['streams'].append({'id':sid,'wave':wave,'arm':arm,'seed':seed,'gpu':gpu,'commands':commands})
    r['files_sha256']=pins;put(a.release,r)
    context=M.validate_release(a,a.release.read_bytes(),mods,now=now)
    return a,context


class FakeRuntime:
    def __init__(self,context,behavior='timeout'):
        self.context=context;self.t=0.;self.start=M.utc();self.behavior=behavior;self.future=None;self.created=[];self.active={};self.closed=False
    def mono(self):return self.t
    def now(self):return self.start+timedelta(seconds=self.t)
    def sleep(self,seconds):self.t+=seconds
    def gpu_now(self):return []
    def gpu_submit(self,owned):self.future=([],dict(owned))
    def gpu_result(self):
        result=self.future;self.future=None;return result
    def launch(self,entry,logroot):
        process=SimpleNamespace(pid=1000+len(self.created),returncode=None);handles=(io.BytesIO(),io.BytesIO())
        self.created.append({'entry':entry,'process':process,'at':self.t,'handles':handles});self.active[process.pid]=self.created[-1]
        root=Path(entry['directory']);root.mkdir(parents=True)
        if self.behavior in ('timeout','malformed','error'):
            value=['malformed'] if self.behavior=='malformed' else {'error_type':'RuntimeError' if self.behavior=='error' else 'TimeoutError','current':None,'synthetic_only':True}
            put(root/'failed_attempt.json',value)
        return process,handles
    def identity(self,pid):
        item=self.active[pid];return {'pid':pid,'ppid':os.getpid(),'start_ticks':pid+100,'argv':item['entry']['command'],'state':'R'}
    def gpu_identity(self,pid):
        if self.active[pid]['process'].returncode is not None:raise FileNotFoundError()
        return self.identity(pid)
    def reap(self,child):
        item=self.active[child['process'].pid]
        if self.behavior in ('hang','refuse'):
            if self.behavior=='refuse' or not any(e['signal']=='SIGKILL' for e in child['signals']):return None
            code=-signal.SIGKILL
        else:
            if self.t-item['at']<.3:return None
            code=1
        child['process'].returncode=code
        return {'exit_code':code,'cpu_user_seconds':.1,'cpu_system_seconds':.01,'linux_peak_rss_kib':1234}
    def stop(self,child,sig):child['signals'].append({'signal':signal.Signals(sig).name,'at_utc':self.now().isoformat(),'result':'refused_or_failed' if self.behavior=='refuse' else 'sent_to_owned_group'})
    def close(self):self.closed=True


def test_default_never_loads_or_launches(monkeypatch,capsys):
    monkeypatch.setattr(M,'modules',lambda:pytest.fail('default must not load'))
    assert M.main([])==0 and json.loads(capsys.readouterr().out)['stage_count']==30


def test_complete_release_composes_all_frozen_scalar_gates(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);assert len(c.entries)==30 and c.release['endpoint_updates']==25000
    assert len(c.mods.C.cell_schedule(c.manifests['test'],'full-rollout'))==18


@pytest.mark.parametrize('bad',['seed','quota','order','window','manifest','source','child','extra_path','cohort'])
def test_release_semantic_refusals(tmp_path,monkeypatch,bad):
    a,c=fixture(tmp_path,monkeypatch);r=M.read(a.release)
    if bad=='seed':r['streams'][0]['seed']=False
    elif bad=='quota':r['stage_quotas_seconds']['clean_validation']=7201
    elif bad=='order':r['stage_order'][0]=r['stage_order'][1]
    elif bad=='window':r['absolute_stop_utc']=(M.DEADLINE-timedelta(seconds=3599)).isoformat()
    elif bad=='manifest':r['files_sha256'][str(a.test_manifest)]='0'*64
    elif bad=='source':r['files_sha256'][str(HERE/'evaluate_goop3d_graph_support_v1.py')]='0'*64
    elif bad=='child':r['streams'][0]['commands'][0][1]='-c'
    elif bad=='extra_path':r['files_sha256'][str(a.output_dir/'unrelated')]='0'*64
    else:r['cohort_sha256']='0'*64
    put(a.release,r)
    with pytest.raises((ValueError,FileNotFoundError)):M.validate_release(a,a.release.read_bytes(),c.mods)


def test_fake_queue_emits_all30_and_strict_collector_accepts(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c)
    monkeypatch.setattr(M.subprocess,'Popen',lambda *a,**k:pytest.fail('no real processes allowed'))
    assert M.execute_queue(a,c,rt)==0
    ledger=M.read(a.output_dir/'final_evaluation_ledger.json');assert len(ledger['stages'])==30 and len(rt.created)==30
    assert all(e['outcome']['inner_timeout_reported'] and all(x['state']=='not_completed_before_invocation_end' for x in e['cells']) for e in ledger['stages'])
    first_b=min(x['at'] for x in rt.created if x['entry']['wave']=='B');last_a=max(x['at'] for x in rt.created if x['entry']['wave']=='A')
    assert first_b>last_a and rt.closed
    paths=[a.output_dir/'final_evaluation_ledger.json',a.cohort,a.cohort_audit,a.valid_manifest,a.test_manifest,a.protocol]
    release={'schema':'adaptgns_goop3d_scalar_collection_release_v1','issued_by':'root','status':'approved_stopped_scalar_collection',
        'collector_sha256':M.COLLECTOR_SHA,'files_sha256':{str(p):M.sha(p) for p in paths}}
    collected=c.mods.C.collect(*paths,release)
    assert len(collected['stages'])==30 and all(len(s['rows'])==0 for s in collected['stages'])


@pytest.mark.parametrize('behavior',['error','malformed'])
def test_infrastructure_failure_reaps_peers_and_preserves_never_started(tmp_path,monkeypatch,behavior):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,behavior)
    assert M.execute_queue(a,c,rt)==1
    ledger=M.read(a.output_dir/'final_evaluation_ledger.json');assert ledger['abort_reason'] and len(rt.created)==4
    assert sum(e['outcome']['started'] for e in ledger['stages'])==4
    assert all(x['process'].returncode is not None for x in rt.created)
    if behavior=='malformed':assert any(e['outcome'].get('worker_failed_attempt_parse_error') for e in ledger['stages'])


def test_whole_invocation_timeout_escalates_and_never_retries(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,'hang')
    assert M.execute_queue(a,c,rt)==0
    ledger=M.read(a.output_dir/'final_evaluation_ledger.json')
    assert len(rt.created)==30 and all(e['outcome']['quota_stop_initiated'] for e in ledger['stages'])
    assert all([x['signal'] for x in e['outcome']['signals']]==['SIGINT','SIGTERM','SIGKILL'] for e in ledger['stages'])


def test_unreaped_identity_refusal_blocks_ledger_and_following_work(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,'refuse')
    assert M.execute_queue(a,c,rt)==1 and len(rt.created)==4
    assert not (a.output_dir/'final_evaluation_ledger.json').exists()
    assert len(M.read(a.output_dir/'process_outcomes.json')['unreaped_owned_children'])==4
    assert M.read(a.output_dir/'failed_publication.json')['unreaped_owned_children']


def test_reaped_gpu_pid_reuse_is_not_owned(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);entry=c.entries[0];rt=FakeRuntime(c)
    process,_=rt.launch(entry,tmp_path);child={'entry':entry,'process':process,'command':entry['command'],'identity':rt.identity(process.pid)}
    process.returncode=0
    monkeypatch.setattr(rt,'gpu_identity',lambda pid:{**child['identity'],'start_ticks':999999})
    with pytest.raises(ValueError,match='Reused live PID'):M.check_gpu_observation([{'pid':process.pid,'gpu_uuid':entry['gpu_uuid']}],{process.pid:child},rt,c.mods.B)
    monkeypatch.setattr(rt,'gpu_identity',lambda pid:(_ for _ in ()).throw(FileNotFoundError()))
    result=M.check_gpu_observation([{'pid':process.pid,'gpu_uuid':entry['gpu_uuid']}],{process.pid:child},rt,c.mods.B)
    assert result[0]['classification']=='stale_sample_of_reaped_absent_owned_child'


def test_late_clock_after_metadata_check_prevents_launch(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c);original=M.verify_control_metadata
    def delayed(context):original(context);rt.t+=400
    monkeypatch.setattr(M,'verify_control_metadata',delayed)
    assert M.execute_queue(a,c,rt)==1 and not rt.created
    ledger=M.read(a.output_dir/'final_evaluation_ledger.json')
    assert all(not s['outcome']['started'] for s in ledger['stages'])
    assert 'window closed after' in ledger['abort_reason']


def test_full_hashes_never_block_active_child_polling(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c);original=M.verify_files
    def stopped_only(files):
        assert not any(x['process'].returncode is None for x in rt.created)
        return original(files)
    monkeypatch.setattr(M,'verify_files',stopped_only)
    assert M.execute_queue(a,c,rt)==0


def test_parent_log_mutation_prevents_final_ledger(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c);original=M.write
    def mutation(path,value):
        result=original(path,value)
        if Path(path).name=='queue_status.json' and value.get('state')=='validated_ledger_before_last_publication':
            p=next((a.output_dir/'logs').glob('*.launch.json'));p.write_bytes(p.read_bytes()+b'mutated')
        return result
    monkeypatch.setattr(M,'write',mutation)
    assert M.execute_queue(a,c,rt)==1 and not (a.output_dir/'final_evaluation_ledger.json').exists()
    assert 'queue tree changed' in M.read(a.output_dir/'failed_publication.json')['error']


def test_final_publication_keeps_root_analysis_reserve(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c);original=M.queue_snapshot;calls=[]
    def late_snapshot(output,C):
        value=original(output,C);calls.append(1)
        if len(calls)==2:rt.t=(M.stamp(c.release['absolute_stop_utc'])-rt.start).total_seconds()+1
        return value
    monkeypatch.setattr(M,'queue_snapshot',late_snapshot)
    assert M.execute_queue(a,c,rt)==1 and not (a.output_dir/'final_evaluation_ledger.json').exists()
    assert 'analysis reserve' in M.read(a.output_dir/'failed_publication.json')['error']


def test_cleanup_continues_after_one_outcome_accounting_error(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,'error');original=M.observe_reaped
    def fault(child,*args):
        if child['process'].pid==1001:raise ValueError('synthetic scalar observation fault')
        return original(child,*args)
    monkeypatch.setattr(M,'observe_reaped',fault)
    assert M.execute_queue(a,c,rt)==1
    assert len(rt.created)==4 and all(x['process'].returncode is not None for x in rt.created)
    report=M.read(a.output_dir/'process_outcomes.json')
    assert report['unreaped_owned_children']==[] and any(c['cleanup_errors'] for c in report['all_children'])


def test_delayed_gpu_query_accepts_owned_successor_after_stage_transition(tmp_path,monkeypatch):
    a,c=fixture(tmp_path,monkeypatch)
    class DelayedRuntime(FakeRuntime):
        def gpu_submit(self,owned):self.future=dict(owned)
        def gpu_result(self):
            if self.future is None or len(self.created)<8:return None
            captured=self.future;self.future=None
            rows=[{'pid':x['process'].pid,'gpu_uuid':x['entry']['gpu_uuid']} for x in self.created if x['process'].returncode is None]
            return rows,captured
    rt=DelayedRuntime(c)
    assert M.execute_queue(a,c,rt)==0 and len(rt.created)==30
    classifications=[v['classification'] for observation in c.gpu_observations for v in observation['classification']]
    assert 'owned_launched_during_query_identity_verified' in classifications


def test_symlink_interpreter_preserves_venv_argv_and_binds_target_and_config(tmp_path,monkeypatch):
    root=tmp_path.resolve()/'synthetic_venv';(root/'bin').mkdir(parents=True)
    python=root/'bin/python';python.symlink_to(Path(sys.executable).resolve());(root/'pyvenv.cfg').write_text('synthetic fixture only\n')
    monkeypatch.setattr(sys,'executable',str(python));monkeypatch.setattr(sys,'prefix',str(root))
    a,c=fixture(tmp_path,monkeypatch)
    assert all(e['command'][0]==str(python) for e in c.entries)
    assert c.bindings[str(python)]==c.bindings[str(python.resolve())]==M.sha(python)
    assert c.bindings[str(root/'pyvenv.cfg')]==M.sha(root/'pyvenv.cfg')
    assert c.release['python_environment']['sys_prefix']==str(root)
    rt=FakeRuntime(c)
    assert M.execute_queue(a,c,rt)==0 and all(e['entry']['command'][0]==str(python) for e in rt.created)
    (root/'pyvenv.cfg').write_text('changed fixture config\n')
    with pytest.raises(ValueError):M.verify_files(c.bindings)
