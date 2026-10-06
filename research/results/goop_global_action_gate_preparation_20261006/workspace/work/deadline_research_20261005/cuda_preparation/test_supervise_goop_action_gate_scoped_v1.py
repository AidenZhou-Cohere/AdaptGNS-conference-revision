"""Inert owned-process scheduling probes; no training/data/GPU processes."""
from datetime import datetime,timedelta,timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gate_scoped_ops',HERE/'supervise_goop_action_gate_scoped_v1.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
NOW=datetime(2026,10,6,12,tzinfo=timezone.utc)
def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def fixture(tmp_path,mode='train-label-capacity'):
    mods=SimpleNamespace(R=M.load('supervise_goop3d_final_evaluation_v1.py',M.PINS['supervise_goop3d_final_evaluation_v1.py']),
        S=M.load('supervise_sand_scoped_science_v1.py',M.PINS['supervise_sand_scoped_science_v1.py']),
        B=M.load('measure_sand_cuda_capacity_v2.py',M.PINS['measure_sand_cuda_capacity_v2.py']))
    args=SimpleNamespace(output_dir=tmp_path/'queue');bound=tmp_path/'control.json';put(bound,{'synthetic':True})
    uuids=[f'00000000-0000-0000-0000-{i:012d}' for i in range(4)]
    release={'mode':mode,'hostname':'synthetic','closed_goop_processes':[{'pid':99,'start_ticks':1}],
        'clock_error_bound_seconds':0,'gpu_uuids':uuids,'latest_start_utc':(NOW+timedelta(seconds=20)).isoformat()}
    entries=[]
    for seed in range(3):
        gpu=1 if seed==1 else 0;root=args.output_dir/'jobs'/f'mix_seed{seed}'
        cells=([{'source_index':s,'policy':p} for s in range(30) for p in ('base','random25','learned_global_gate','validation_rate_random_gate')]
            if mode=='test-rollout' else [{'source_index':0,'target_frame':6}])
        entries.append({'stream':f'mix_seed{seed}','stage':mode,'seed':seed,'arm':'mix','gpu':gpu,'gpu_uuid':uuids[gpu],
            'command':['/synthetic/python','/synthetic/run_goop_action_gate_v1.py','--execute','--mode',mode,'--seed',str(seed)],
            'directory':str(root),'outer_timeout_seconds':2,'expected_cells':cells})
    c=SimpleNamespace(release=release,raw=M.encode(release),bindings={str(bound):M.sha(bound)},metadata={str(bound):mods.S.file_identity(bound)},
        entries=entries,mods=mods,stop=NOW+timedelta(seconds=100))
    return args,c


class Runtime:
    def __init__(self,c,behavior='complete'):
        self.c=c;self.behavior=behavior;self.t=0.;self.active={};self.created=[];self.future=None;self.closed=False;self.observations=[]
    def mono(self):return self.t
    def now(self):return NOW+timedelta(seconds=self.t)
    def sleep(self,s):self.t+=s
    def process_inventory(self):return []
    def gpu_now(self):return [{'pid':800,'gpu_uuid':self.c.release['gpu_uuids'][2]}]
    def gpu_result(self):return None
    def gpu_submit(self,owned):pass
    def launch(self,entry,logs):
        process=SimpleNamespace(pid=100+len(self.created),returncode=None);handles=[]
        item={'entry':entry,'process':process,'at':self.t};self.created.append(item);self.active[process.pid]=item
        root=Path(entry['directory']);root.mkdir(parents=True)
        guarded=self.behavior=='guard';state='complete_with_guard_failures' if guarded else 'complete'
        put(root/'status.json',{'state':state,'required_rows':len(entry['expected_cells']),'committed_rows':len(entry['expected_cells']),
            'all_inputs_reverified':True,'model_state_verified_unchanged':True,'abort_reason':None})
        put(root/('rollout_collection.json' if entry['stage']=='test-rollout' else 'label_collection.json'),
            {'status':state,'mode':entry['stage'],'model':{'seed':entry['seed']},'required_rows':len(entry['expected_cells']),
             'committed_rows':len(entry['expected_cells']),'all_inputs_reverified':True,'model_state_verified_unchanged':True,'abort_reason':None,
             'rows':[{**v,'status':'failed' if guarded else 'complete'} for v in entry['expected_cells']]})
        return process,handles
    def identity(self,pid):
        if pid==99:
            if self.behavior=='old_live':return {'pid':99,'start_ticks':1}
            raise FileNotFoundError()
        item=self.active[pid]
        if self.behavior=='identity_error':raise RuntimeError('synthetic identity lookup failed after launch')
        return {'pid':pid,'ppid':os.getpid(),'start_ticks':pid+1,'argv':item['entry']['command']}
    def gpu_identity(self,pid):return self.identity(pid)
    def reap(self,child):
        item=self.active[child['process'].pid]
        if self.behavior in ('hang','refuse'):
            if self.behavior=='refuse' or not any(s['signal']=='SIGKILL' for s in child['signals']):return None
            code=-9
        else:
            if self.t-item['at']<(.9 if item['entry']['seed']==1 else .3):return None
            code=1 if self.behavior=='error' and item['entry']['seed']==0 else 0
        child['process'].returncode=code
        return {'exit_code':code,'cpu_user_seconds':.1,'cpu_system_seconds':.01,'linux_peak_rss_kib':100}
    def stop(self,child,sig):child['signals'].append({'signal':signal.Signals(sig).name,'result':'synthetic_owned_only'})
    def close(self):self.closed=True


def test_default_never_loads(monkeypatch,capsys):
    monkeypatch.setattr(M,'modules',lambda:pytest.fail('description only'))
    assert M.main([])==0 and 'description_only' in capsys.readouterr().out


def test_three_fixed_seeds_share_only0and1_and_successor_can_overlap_seed1(tmp_path):
    a,c=fixture(tmp_path);rt=Runtime(c);assert M.execute_queue(a,c,rt)==0
    assert [v['entry']['seed'] for v in rt.created]==[0,1,2]
    assert rt.created[2]['at']<.9 and [v['entry']['gpu'] for v in rt.created]==[0,1,0] and rt.closed
    ledger=M.snapshot(a.output_dir/'phase_ledger.json')[0]
    assert ledger['state']=='complete_fixed_phase' and len(ledger['jobs'])==3 and not ledger['numerical_arrays_audited_by_supervisor']
    obs=M.snapshot(a.output_dir/'logs/gpu_000000.json')[0]
    assert obs['checked'][0]['classification']=='unowned_device_observed_without_control'


def test_complete_test_guard_grid_continues_all_three_seeds(tmp_path):
    a,c=fixture(tmp_path,'test-rollout');rt=Runtime(c,'guard');assert M.execute_queue(a,c,rt)==0
    ledger=M.snapshot(a.output_dir/'phase_ledger.json')[0]
    assert len(rt.created)==3 and sum(j['outcome']['complete_collection']['committed_rows'] for j in ledger['jobs'])==360


@pytest.mark.parametrize('behavior',['guard','error','old_live','identity_error'])
def test_label_or_infrastructure_failure_never_starts_seed2(tmp_path,behavior):
    a,c=fixture(tmp_path);rt=Runtime(c,behavior);assert M.execute_queue(a,c,rt)==1
    assert not any(v['entry']['seed']==2 for v in rt.created)
    assert all(v['process'].returncode is not None for v in rt.created)
    assert M.snapshot(a.output_dir/'phase_ledger.json')[0]['state']=='stopped_requires_review'


def test_quota_escalates_owned_children_then_retains_missing_seed(tmp_path):
    a,c=fixture(tmp_path);rt=Runtime(c,'hang');assert M.execute_queue(a,c,rt)==1
    ledger=M.snapshot(a.output_dir/'phase_ledger.json')[0]
    assert len(rt.created)==2 and ledger['jobs'][2]['outcome']['state']=='never_started'
    assert all([s['signal'] for s in j['outcome']['signals']]==['SIGINT','SIGTERM','SIGKILL'] for j in ledger['jobs'][:2])


def test_unreaped_refusal_blocks_final_ledger(tmp_path):
    a,c=fixture(tmp_path);rt=Runtime(c,'refuse');assert M.execute_queue(a,c,rt)==1
    assert len(rt.created)==2 and not (a.output_dir/'phase_ledger.json').exists()
    assert M.snapshot(a.output_dir/'process_outcomes.json')[0]['unreaped_owned_children']


def test_late_clock_after_checks_blocks_launch(tmp_path,monkeypatch):
    a,c=fixture(tmp_path);rt=Runtime(c);original=M.verify_metadata
    def late(ctx):original(ctx);rt.t+=200
    monkeypatch.setattr(M,'verify_metadata',late)
    assert M.execute_queue(a,c,rt)==1 and not rt.created


def test_full_hashes_and_collection_parse_only_when_all_children_stopped(tmp_path,monkeypatch):
    a,c=fixture(tmp_path);rt=Runtime(c);verify=M.verify_files;collection=M.collection_complete
    def stopped():assert not any(v['process'].returncode is None for v in rt.created)
    def checked(bindings):stopped();return verify(bindings)
    def audited(entry):stopped();return collection(entry)
    monkeypatch.setattr(M,'verify_files',checked);monkeypatch.setattr(M,'collection_complete',audited)
    assert M.execute_queue(a,c,rt)==0


def test_late_parent_log_mutation_blocks_ready_publication(tmp_path,monkeypatch):
    a,c=fixture(tmp_path);rt=Runtime(c);original=M.encode
    def mutate(value):
        raw=original(value)
        if 'supervisor_files' in value:(a.output_dir/'release_snapshot.json').write_text('changed after serialization')
        return raw
    monkeypatch.setattr(M,'encode',mutate)
    assert M.execute_queue(a,c,rt)==1 and not (a.output_dir/'phase_ledger.json').exists()


def test_foreign_gpu0_work_prevents_any_launch(tmp_path):
    a,c=fixture(tmp_path);rt=Runtime(c);rt.gpu_now=lambda:[{'pid':800,'gpu_uuid':c.release['gpu_uuids'][0]}]
    assert M.execute_queue(a,c,rt)==1 and not rt.created


def release_fixture(tmp_path,monkeypatch,mode='train-label-capacity'):
    """Exercise controller admission around an inert already-reviewed driver gate."""
    a,c=fixture(tmp_path,mode);a.release=tmp_path/'release.json';mods=c.mods;calls=[]
    driver_path=HERE/'run_goop_action_gate_v1.py';driver=M.load(driver_path.name,M.sha(driver_path))
    monkeypatch.setattr(M,'DRIVER_SHA',M.sha(driver_path))
    stop=NOW+timedelta(seconds=600)
    def gate(args,at):
        calls.append(args.seed)
        child_release={'absolute_stop_utc':stop.isoformat()}
        bindings=dict(c.bindings)
        if args.selection is not None:bindings[str(args.selection)]=M.sha(args.selection)
        if mode=='test-rollout':
            child_release['complete_capacity_report']={'path':str(capacity_path),'sha256':M.sha(capacity_path)}
            bindings[str(capacity_path)]=M.sha(capacity_path)
        return SimpleNamespace(release=child_release,cohort_sha='a'*64,bindings=bindings,
            mods=SimpleNamespace(core=None),manifest={'synthetic_scalar_only':True})
    mods.D=SimpleNamespace(MODES=driver.MODES,parse_args=driver.parse_args,release_gate=gate,
        expected_schedule=lambda *args:[{'source_index':0,'target_frame':6}])
    r={**c.release,'schema':M.RELEASE_SCHEMA,'status':'approved_for_one_fixed_gate_phase','issued_by':'root','hostname':M.socket.gethostname(),
        'host_role':'B','output_dir':str(a.output_dir),'environment':mods.S.ENVIRONMENT,'no_retry_or_resume':True,
        'does_not_displace_original_studies':True,'all_six_original_goop_models_frozen':True,
        'python_environment':mods.S.python_environment(sys.executable),'process_clock_checked_utc':NOW.isoformat(),
        'remaining_outer_work_reserve_seconds':3600,'absolute_stop_utc':stop.isoformat(),'latest_start_utc':(NOW+timedelta(seconds=300)).isoformat(),
        'gpu_scope':{'owned_indices':[0,1],'unassigned_devices':'observe_without_control','live_handoff':False},'jobs':[]}
    pins={str(HERE/n):v for n,v in M.PINS.items()};pins.update(c.bindings)
    pins[str(driver_path)]=M.DRIVER_SHA;pins[str(Path(M.__file__).resolve())]=M.sha(M.__file__)
    env=r['python_environment'];pins[env['lexical_path']]=env['binary_sha256'];pins[env['resolved_binary_path']]=env['binary_sha256']
    if env['pyvenv_config_path']:pins[env['pyvenv_config_path']]=env['pyvenv_config_sha256']
    r['files_sha256']=pins
    if 'rollout' in mode:
        selection_path=tmp_path/'selection.json';put(selection_path,{'synthetic_common_selection':True});pins[str(selection_path)]=M.sha(selection_path)
    if mode=='test-rollout':
        capacity_path=tmp_path/'capacity.json';capacity={'schema':'adaptgns_goop_global_action_gate_capacity_budget_v1','status':'complete_measured_capacity_passed',
            'all_36_rollouts_complete':True,'all_train_validation_labels_complete':True,'complete_remaining_work_fits':True,
            'protocol_sha256':M.PINS['goop_global_action_gate_protocol_v1.md'],'driver_sha256':M.DRIVER_SHA,'cohort_sha256':'a'*64,
            'selection_sha256':M.sha(selection_path),'per_seed':[{'seed':s,'test_allocation_seconds':100} for s in range(3)],
            'test_concurrent_allocation_seconds':230,'remaining_outer_reserves_seconds':{'input_preflight':900,'scalar_array_collection':2700,'independent_analysis_manuscript':3600},
            'latest_test_start_utc':r['latest_start_utc']}
        put(capacity_path,capacity);pins[str(capacity_path)]=M.sha(capacity_path)
        r['remaining_outer_work_reserve_seconds']=7200;r['complete_capacity_report']={'path':str(capacity_path),'sha256':M.sha(capacity_path)}
    for seed in range(3):
        gpu=1 if seed==1 else 0
        command=[sys.executable,str(driver_path),'--execute','--mode',r['mode'],'--seed',str(seed),'--cuda-index',str(gpu),
            '--threads','2','--max-seconds','100','--output-dir',str(a.output_dir/'jobs'/f'mix_seed{seed}')]
        if 'rollout' in mode:command+=['--selection',str(selection_path)]
        r['jobs'].append({'seed':seed,'gpu':gpu,'command':command})
    put(a.release,r)
    return a,r,mods,calls


def test_root_admission_composes_all_three_scalar_driver_gates(tmp_path,monkeypatch):
    a,r,mods,calls=release_fixture(tmp_path,monkeypatch)
    context=M.validate_release(a,a.release.read_bytes(),mods,NOW)
    assert calls==[0,1,2] and [v['gpu'] for v in context.entries]==[0,1,0]


@pytest.mark.parametrize('bad',['mapping','duplicate_flag','resolved_python','cohort','source','window','scope','root','child_binding'])
def test_changed_root_release_refused_before_launch(tmp_path,monkeypatch,bad):
    a,r,mods,_=release_fixture(tmp_path,monkeypatch)
    if bad=='mapping':r['jobs'][0]['gpu']=3
    elif bad=='duplicate_flag':r['jobs'][0]['command']+=['--seed','0']
    elif bad=='resolved_python':r['jobs'][0]['command'][0]=str(Path(sys.executable).resolve())
    elif bad=='cohort':r['all_six_original_goop_models_frozen']=False
    elif bad=='source':r['files_sha256'][str(HERE/'run_goop_action_gate_v1.py')]='0'*64
    elif bad=='window':r['latest_start_utc']=(NOW+timedelta(seconds=599)).isoformat()
    elif bad=='scope':r['gpu_scope']['owned_indices']=[2,3]
    elif bad=='root':r['issued_by']='template'
    else:r['files_sha256'].pop(str(tmp_path/'control.json'))
    put(a.release,r)
    with pytest.raises(ValueError):M.validate_release(a,a.release.read_bytes(),mods,NOW)


def test_accounting_failure_never_relabels_launched_child_never_started(tmp_path,monkeypatch):
    a,c=fixture(tmp_path);rt=Runtime(c)
    monkeypatch.setattr(c.mods.R,'observe_reaped',lambda *args:(_ for _ in ()).throw(ValueError('synthetic outcome accounting error')))
    assert M.execute_queue(a,c,rt)==1
    report=M.snapshot(a.output_dir/'process_outcomes.json')[0]
    assert len(report['owned_registry'])==2 and report['unreaped_owned_children']==[]
    assert all(j['outcome']['started'] and j['outcome']['state']=='stopped_and_reaped_unclassified' for j in report['jobs'][:2])
    assert report['jobs'][2]['outcome']['state']=='never_started'


def test_nested_driver_budget_receipt_classification():
    c=SimpleNamespace(mods=SimpleNamespace(R=SimpleNamespace(entry_key=lambda e:(e['arm'],e['seed'],e['stage']))),
        entries=[{'arm':'mix','seed':0,'stage':'test-rollout'}])
    value={'started':True,'state':'stopped_and_reaped','worker_failed_attempt':{'failure':{'category':'execution_budget'}},
        'inner_timeout_reported':False,'quota_stop_initiated':False}
    rows=M.retained_outcomes(c,[],{('mix',0,'test-rollout'):value},None)
    assert rows[0]['outcome']['inner_timeout_reported'] is True and rows[0]['outcome']['termination_reason']=='inner_driver_execution_budget'


def test_all_rollout_children_bind_one_selection_and_capacity(tmp_path,monkeypatch):
    a,r,mods,calls=release_fixture(tmp_path,monkeypatch,'test-rollout')
    assert len(M.validate_release(a,a.release.read_bytes(),mods,NOW).entries)==3 and calls==[0,1,2]
    other=tmp_path/'other_selection.json';put(other,{'synthetic_common_selection':'different'})
    command=r['jobs'][1]['command'];command[command.index('--selection')+1]=str(other);r['files_sha256'][str(other)]=M.sha(other)
    put(a.release,r)
    with pytest.raises(ValueError,match='share one exact common selection'):M.validate_release(a,a.release.read_bytes(),mods,NOW)


@pytest.mark.parametrize('field',['driver_sha256','cohort_sha256','selection_sha256','protocol_sha256'])
def test_parent_capacity_lineage_must_match_children(tmp_path,monkeypatch,field):
    a,r,mods,_=release_fixture(tmp_path,monkeypatch,'test-rollout');path=tmp_path/'capacity.json';value=M.snapshot(path)[0]
    value[field]='0'*64;put(path,value);r['files_sha256'][str(path)]=M.sha(path);r['complete_capacity_report']['sha256']=M.sha(path);put(a.release,r)
    with pytest.raises(ValueError,match='Complete fixed study/capacity'):M.validate_release(a,a.release.read_bytes(),mods,NOW)


def test_different_parent_capacity_path_cannot_replace_child_receipt(tmp_path,monkeypatch):
    a,r,mods,_=release_fixture(tmp_path,monkeypatch,'test-rollout');other=tmp_path/'other_capacity.json';other.write_bytes((tmp_path/'capacity.json').read_bytes())
    r['complete_capacity_report']={'path':str(other),'sha256':M.sha(other)};r['files_sha256'][str(other)]=M.sha(other);put(a.release,r)
    with pytest.raises(ValueError,match='same exact complete capacity'):M.validate_release(a,a.release.read_bytes(),mods,NOW)


@pytest.mark.parametrize('live,device,changed,accepted',[(False,2,True,True),(False,2,False,False),(True,2,True,False),(False,0,True,False)])
def test_retired_pid_reuse_only_unassigned_without_control(tmp_path,live,device,changed,accepted):
    a,c=fixture(tmp_path);entry=c.entries[0];identity={'pid':100,'ppid':os.getpid(),'start_ticks':10,'argv':entry['command']}
    child={'process':SimpleNamespace(pid=100,returncode=None if live else 0),'identity':identity,'command':entry['command'],'job':entry}
    rt=SimpleNamespace(gpu_identity=lambda pid:{**identity,'start_ticks':11 if changed else 10})
    rows=[{'pid':100,'gpu_uuid':c.release['gpu_uuids'][device]}]
    if accepted:
        value=M.scoped_gpu_observation(rows,c,[child],rt)
        assert value['retired_pid_observations'][0]['classification']=='retired_pid_unassigned_device_observed_without_control'
    else:
        with pytest.raises(ValueError):M.scoped_gpu_observation(rows,c,[child],rt)
