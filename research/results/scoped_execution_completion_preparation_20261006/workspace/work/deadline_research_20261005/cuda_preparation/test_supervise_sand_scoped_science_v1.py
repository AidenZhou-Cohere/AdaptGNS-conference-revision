"""Synthetic scalar receipts and injected fake runtime only; never GPU/train/test."""
from datetime import datetime,timedelta,timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sand_scoped_test',HERE/'supervise_sand_scoped_science_v1.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def fixture(tmp_path,monkeypatch,role='B'):
    args=SimpleNamespace(host_role=role,output_dir=tmp_path/'queue',repo=tmp_path/'repo',python=Path(sys.executable),
        capacity_source=HERE/'measure_sand_graph_support_capacity.py',lifecycle_source=HERE/'measure_sand_cuda_capacity_v2.py',
        verifier_source=HERE/'supervise_sand_graph_support_science.py',trainer=HERE/'train_sand_graph_support_cuda.py',
        train_manifest=tmp_path/'train.json',admission=tmp_path/'admission.json',structural_report=tmp_path/'structure.json',
        protocol=HERE/'sand_graph_support_100k_protocol_v1.md',schedule_plan=HERE/'sand_scoped_schedule_fixed_spec_v2.json',
        amendment=HERE/'sand_scoped_operational_amendment_v1.md',process_check=tmp_path/'process.json',clock_check=tmp_path/'clock.json')
    mods=M.configure(args);now=datetime(2026,10,6,10,tzinfo=timezone.utc)
    uuids=[f'00000000-0000-0000-0000-{i:012x}' for i in range(4)];jobs=[j for j in M.SCHEDULE if j['wave']==role]
    foreign=[] if role=='A' else [{'pid':9001,'gpu_uuid':uuids[0]}]
    process={'schema':'adaptgns_sand_scoped_process_check_v1','issued_by':'root','host':M.socket.gethostname(),'host_role':role,
        'gpu_uuids':uuids,'checked_utc':now.isoformat(),'matching_sand_training_processes':[],'active_old_whole_host_monitors':[],
        'all_prior_goop_trainers_reaped':True,'all_prior_whole_host_goop_monitors_exited':True,'all_prior_goop_gpu_work_reaped':True,
        'closed_goop_processes':[{'pid':i,'start_ticks':i+1,'argv':['/synthetic/'+n],'role':r,'stopped_and_reaped_or_exited':True}
            for i,n,r in [(111,'trainer.py','trainer'),(222,'monitor.py','whole_host_monitor')]],'gpu_processes':foreign}
    clock={'schema':'adaptgns_sand_scoped_clock_check_v1','issued_by':'root','host':M.socket.gethostname(),
        'checked_utc':now.isoformat(),'root_host_samples_reviewed':True,'clock_error_bound_seconds':1}
    M.write(args.process_check,process);M.write(args.clock_check,clock)
    hashes={'fixture':'synthetic_scalar_only'}
    release={'schema':M.RELEASE_SCHEMA,'status':'admitted_fresh_scoped_training','issued_by':'root','host':M.socket.gethostname(),
        'host_role':role,'dataset':'Sand','schedule':M.SCHEDULE,'environment':M.ENVIRONMENT,'python_environment':M.python_environment(args.python),'files_sha256':hashes,'output_dir':str(args.output_dir),
        'scientific_training_admitted':True,'automatic_retry_resume_or_promotion':False,'live_supervisor_handoff':False,
        'six_model_freeze_before_test':True,'cohort_id':'synthetic-only','gpu_uuids':uuids,'owned_gpu_indices':[j['gpu'] for j in jobs],
        'gpu_policy':'exclusive_owned_devices_observe_all_others_without_control','co_resident_runtime_bound_measured':False,
        'clock_error_bound_seconds':1,'clock_checked_utc':now.isoformat(),'process_checked_utc':now.isoformat(),
        'training_stop_utc':M.TRAINING_STOP.isoformat(),'latest_start_utc':M.LATEST_START.isoformat(),
        'compute_analysis_deadline_utc':M.DEADLINE.isoformat(),'training_allocation_seconds':M.TRAINING_SECONDS,'post_training_allocation_seconds':M.POSTTRAINING_SECONDS}
    c=M.validate_release(args,release,mods,hashes,now);c.bindings={str(args.process_check):M.sha(args.process_check)}
    c.metadata={p:M.file_identity(p) for p in c.bindings}
    args.output_dir.mkdir();(args.output_dir/'jobs').mkdir();(args.output_dir/'logs').mkdir()
    launch={'gpu_uuids':uuids,'files_sha256':{'protocol':M.PROTOCOL_SHA}}
    monkeypatch.setattr(mods.S,'verify_job',lambda directory,job,external,*a:{**job,'directory':directory,'external':external,'synthetic_endpoint_check':True})
    monkeypatch.setattr(mods.S,'verify_pairing',lambda jobs:[{'seed':s,'synthetic_pairing':True} for s in sorted({j['seed'] for j in jobs})])
    return args,c,launch,now


class FakeRuntime:
    def __init__(self,c,start,behavior='complete'):
        self.context=c;self.start=start;self.t=0.;self.future=None;self.created=[];self.behavior=behavior;self.closed=False
    def now(self):return self.start+timedelta(seconds=self.t)
    def mono(self):return self.t
    def sleep(self,s):self.t+=s
    def identity(self,pid):
        for c in self.created:
            if c['process'].pid==pid:return {'pid':pid,'ppid':os.getpid(),'start_ticks':100+pid,'argv':c['command'],'state':'R'}
        raise FileNotFoundError()
    def gpu_identity(self,pid):
        for c in self.created:
            if c['process'].pid==pid and c['process'].returncode is not None:raise FileNotFoundError()
        return self.identity(pid)
    def process_inventory(self):return [self.identity(c['process'].pid) for c in self.created if c['process'].returncode is None]
    def gpu_now(self):return self.context.process['gpu_processes']
    def gpu_submit(self,owned):self.future=(self.gpu_now(),dict(owned))
    def gpu_result(self):r=self.future;self.future=None;return r
    def launch(self,args,job,command):
        p=SimpleNamespace(pid=1000+len(self.created),returncode=None);handles=(io.BytesIO(),io.BytesIO())
        self.created.append({'job':job,'process':p,'command':command,'started':self.t})
        return p,handles,'synthetic.stdout','synthetic.stderr'
    def capture(self,args,c):c['initial_pointer']={'synthetic':True}
    def reap(self,args,c,record,reaped,launch,manifest):
        if self.behavior in ('hang','refuse'):
            if self.behavior=='refuse' or not any(e['signal']=='SIGKILL' for e in c['signals']):return None
            code=-9
        else:
            if self.t-c['started']<.3:return None
            code=1 if self.behavior=='error' else 0
        c['process'].returncode=code
        external={'pid':c['process'].pid,'exit_code':code,'signals':c['signals']}
        record['jobs'].append({**c['job'],'external':external})
        if code:return 'synthetic unsuccessful child'
        reaped.append({**c['job'],'directory':'synthetic','external':external});return None
    def stop(self,c,sig):c['signals'].append({'signal':signal.Signals(sig).name,'result':'synthetic_owned_only'})
    def close(self):self.closed=True


def test_default_never_configures_or_launches(monkeypatch,capsys):
    monkeypatch.setattr(M,'configure',lambda *_:pytest.fail('description must not configure'))
    assert M.main([])==0 and json.loads(capsys.readouterr().out)['updates']==100000


@pytest.mark.parametrize('role,count,seeds',[('A',4,{1,2}),('B',2,{0})])
def test_fixed_scope_and_fresh100k_commands(tmp_path,monkeypatch,role,count,seeds):
    a,c,l,now=fixture(tmp_path,monkeypatch,role)
    assert len(c.jobs)==count and {j['seed'] for j in c.jobs}==seeds
    for j in c.jobs:
        cmd=c.mods.S.fixed_command(a,j)
        assert cmd[cmd.index('--updates')+1]=='100000' and cmd[cmd.index('--cuda-index')+1]==str(j['gpu'])
        assert cmd[cmd.index('--log-every')+1]=='100' and cmd[cmd.index('--checkpoint-every')+1]=='10000'
        assert not set(cmd)&{'--resume','--stop-after','--clear-stale-lock'}
    rt=FakeRuntime(c,now)
    monkeypatch.setattr(M.subprocess,'Popen',lambda *a,**k:pytest.fail('real launch forbidden'))
    verified,pairing=M.run_host(a,c,rt,l,{})
    assert len(verified)==count and {p['seed'] for p in pairing}==seeds and rt.closed
    r=M.read(a.output_dir/f'wave_{role}.json')
    assert r['state']=='verified' and not r['unreaped_owned_children']


@pytest.mark.parametrize('change',['mapping','deadline','handoff','clock','oldmonitor','oldtrainer','ownedforeign'])
def test_release_rejects_unsafe_or_changed_schedule(tmp_path,monkeypatch,change):
    a,c,l,now=fixture(tmp_path,monkeypatch);r=dict(c.release)
    if change=='mapping':r['owned_gpu_indices']=[0,1]
    elif change=='deadline':r['training_stop_utc']=(M.TRAINING_STOP+timedelta(seconds=1)).isoformat()
    elif change=='handoff':r['live_supervisor_handoff']=True
    elif change=='clock':now+=timedelta(seconds=301)
    else:
        p=M.read(a.process_check)
        if change=='oldmonitor':p['all_prior_whole_host_goop_monitors_exited']=False
        elif change=='oldtrainer':p['all_prior_goop_trainers_reaped']=False
        else:p['gpu_processes']=[{'pid':999,'gpu_uuid':r['gpu_uuids'][2]}]
        M.write(a.process_check,p)
    with pytest.raises(ValueError):M.validate_release(a,r,c.mods,c.hashes,now)


def test_native_old_identity_or_duplicate_blocks_launch(tmp_path,monkeypatch):
    a,c,l,now=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,now)
    monkeypatch.setattr(rt,'identity',lambda pid:{'start_ticks':c.process['closed_goop_processes'][0]['start_ticks']})
    with pytest.raises(ValueError,match='still exists'):M.prior_ownership_closed(c,rt)
    monkeypatch.setattr(rt,'identity',lambda pid:(_ for _ in ()).throw(FileNotFoundError()))
    for name in ['train_sand_graph_support_cuda.py','supervise_graph_support_science_quota_v2.py']:
        monkeypatch.setattr(rt,'process_inventory',lambda:[{'pid':999,'argv':['python3.12','-u','-B','/somewhere/'+name]}])
        with pytest.raises(ValueError):M.prior_ownership_closed(c,rt)


def test_own_supervisor_option_values_are_not_executed_sources(tmp_path,monkeypatch):
    a,c,l,now=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,now)
    command=['/venv/bin/python','-B','-u',str(HERE/'supervise_sand_scoped_science_v1.py'),
        '--verifier-source',str(a.verifier_source),'--trainer',str(a.trainer)]
    assert M.executed_python_script(command)=='supervise_sand_scoped_science_v1.py'
    monkeypatch.setattr(rt,'process_inventory',lambda:[{'pid':os.getpid(),'argv':command}])
    assert M.prior_ownership_closed(c,rt)['relevant_process_inventory'][0]['pid']==os.getpid()
    for option in ('-c','-m'):
        assert M.executed_python_script(['python',option,'supervise_sand_graph_support_science.py']) is None


def test_unowned_gpu_never_signalled_and_owned_reuse_rejected(tmp_path,monkeypatch):
    a,c,l,now=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,now)
    rows=c.process['gpu_processes'];result=M.check_gpu(rows,{},c.mods.B,l['gpu_uuids'],[2,3],lambda pid:pytest.fail('unowned identity control'))
    assert result[0]['classification']=='unowned_device_observed_without_control'
    child={'job':c.jobs[0],'process':SimpleNamespace(pid=123,returncode=0),'identity':{'start_ticks':1},'command':['synthetic']}
    with pytest.raises(ValueError,match='assignment changed'):M.check_gpu([{'pid':123,'gpu_uuid':l['gpu_uuids'][0]}],{123:child},c.mods.B,l['gpu_uuids'],[2,3],lambda pid:pytest.fail('wrong GPU must fail first'))
    with pytest.raises(ValueError,match='Reused live'):M.check_gpu([{'pid':123,'gpu_uuid':l['gpu_uuids'][2]}],{123:child},c.mods.B,l['gpu_uuids'],[2,3],lambda pid:{'start_ticks':2})


@pytest.mark.parametrize('behavior',['error','hang','refuse'])
def test_failures_and_cutoff_preserve_children_without_retry(tmp_path,monkeypatch,behavior):
    a,c,l,now=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,now,behavior)
    if behavior in ('hang','refuse'):
        original=rt.sleep
        def reach_cutoff(s):
            original(s)
            if rt.t<2:rt.t=(c.cleanup-now).total_seconds()+1
        monkeypatch.setattr(rt,'sleep',reach_cutoff)
    with pytest.raises(ValueError):M.run_host(a,c,rt,l,{})
    r=M.read(a.output_dir/'wave_B.json')
    assert len(rt.created)==2 and r['state']=='failed' and len(r['all_children'])==2
    assert bool(r['unreaped_owned_children'])==(behavior=='refuse')
    assert all(ch['job']['gpu'] in [2,3] for ch in r['all_children'])


def test_time_recheck_after_process_scan_prevents_any_launch(tmp_path,monkeypatch):
    a,c,l,now=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,now);original=rt.process_inventory
    def delayed():rt.t=(c.latest-now).total_seconds()+1;return original()
    monkeypatch.setattr(rt,'process_inventory',delayed)
    with pytest.raises(ValueError):M.run_host(a,c,rt,l,{})
    assert not rt.created and len(M.read(a.output_dir/'wave_B.json')['never_started'])==2


def test_no_full_hash_during_owned_child_monitoring(tmp_path,monkeypatch):
    a,c,l,now=fixture(tmp_path,monkeypatch);rt=FakeRuntime(c,now);original=M.verify_files
    def verify(bindings):
        assert not any(x['process'].returncode is None for x in rt.created)
        return original(bindings)
    monkeypatch.setattr(M,'verify_files',verify)
    assert len(M.run_host(a,c,rt,l,{})[0])==2


def test_symlink_interpreter_preserves_venv_argv_and_records_environment(tmp_path,monkeypatch):
    root=tmp_path.resolve()/'synthetic_venv';(root/'bin').mkdir(parents=True)
    python=root/'bin/python';python.symlink_to(Path(sys.executable).resolve());(root/'pyvenv.cfg').write_text('synthetic fixture only\n')
    monkeypatch.setattr(sys,'executable',str(python));monkeypatch.setattr(sys,'prefix',str(root))
    a,c,l,now=fixture(tmp_path,monkeypatch);p=c.release['python_environment']
    assert p['lexical_path']==str(python) and p['resolved_binary_path']==str(python.resolve())
    assert p['binary_sha256']==M.sha(python.resolve()) and p['pyvenv_config_sha256']==M.sha(root/'pyvenv.cfg') and p['sys_prefix']==str(root)
    rt=FakeRuntime(c,now)
    assert len(M.run_host(a,c,rt,l,{})[0])==2 and all(ch['command'][0]==str(python) for ch in rt.created)
    with pytest.raises(ValueError,match='lexical'):M.python_environment(python.resolve())
    (root/'pyvenv.cfg').write_text('changed fixture config\n')
    with pytest.raises(ValueError,match='provenance'):M.validate_release(a,c.release,c.mods,c.hashes,now)
