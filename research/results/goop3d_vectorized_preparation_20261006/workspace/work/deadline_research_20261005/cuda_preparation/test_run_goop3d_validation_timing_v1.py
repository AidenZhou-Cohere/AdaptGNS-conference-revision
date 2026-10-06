"""Synthetic scalar/process mocks only; no GPU, official arrays or network."""
from datetime import timedelta
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import run_goop3d_validation_timing_v1 as M


def test_default_imports_no_numerical_modules():
    code = """import builtins,runpy,sys
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','numpy','scipy','gns','research'):raise AssertionError(name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    p = subprocess.run([sys.executable, '-I', '-c', code, M.__file__], capture_output=True, text=True, check=True)
    r = json.loads(p.stdout)
    assert len(r['schedule']) == 6 and r['evaluation_overlap_credit'] is False


def test_pinned_dependency_modules_load_without_device_work():
    E, W, B, T = M.modules()
    assert E.TRAIN_SCHEMA == T.SCHEMA and W.EVALUATOR_SHA == M.EVALUATOR_SHA
    uuid='00000000-0000-0000-0000-000000000001'
    assert B.canonical_gpu_uuid('GPU-'+uuid) == B.canonical_gpu_uuid(uuid)


def root_values(tmp_path):
    paths = {k:tmp_path/k for k in M.COMMON}
    paths['repo'].mkdir()
    paths['manifest'] = tmp_path/'data'/'valid.json'; paths['manifest'].parent.mkdir()
    for k, p in paths.items():
        if k != 'repo': p.write_text('{}')
    python = tmp_path/'venv'/'bin'/'python'; python.parent.mkdir(parents=True)
    target = tmp_path/'interpreter'; target.write_text('synthetic executable'); python.symlink_to(target)
    summary = tmp_path/'capacity_summary.json'; summary.write_text('{}')
    stopped = tmp_path/'capacity_stop.json'; stopped.write_text(json.dumps({
        'schema':'adaptgns_goop3d_capacity_stopped_receipt_v1', 'issued_by':'root',
        'status':'supervisor_and_all_six_workers_stopped', 'capacity_summary_sha256':M.sha(summary),
        'all_owned_processes_reaped_or_independently_verified_absent':True}))
    checkpoints = {}
    for arm in ('base','mix'):
        p = tmp_path/arm/'checkpoint.pt'; p.parent.mkdir(); p.write_text(arm)
        checkpoints[arm] = {'path':str(p), 'sha256':M.sha(p)}
    releases = {}
    for job in M.SCHEDULE:
        p = tmp_path/(job['id']+'.json'); p.write_text('{}'); releases[job['id']] = str(p)
    root = {'schema':M.RELEASE_SCHEMA, 'status':'admitted_for_six_validation_timing', 'issued_by':'root',
        'scientific_training_admitted':False, 'scientific_endpoint_selected':False, 'test_access_allowed':False,
        'supervisor_sha256':M.sha(M.__file__), 'schedule':M.SCHEDULE, 'environment':M.ENVIRONMENT,
        'output_dir':str(tmp_path/'output'), 'hostname':M.socket.gethostname(), 'evaluation_overlap_credit':False,
        'process_identity_checked_utc':M.now().isoformat(), 'gpu_uuids':['GPU-first','GPU-second'],
        'common':{k:str(v) for k,v in paths.items()}, 'checkpoints':checkpoints, 'evaluator_releases':releases,
        'inner_seconds':{m:100 for m in M.MODES}, 'outer_seconds':{m:200 for m in M.MODES},
        'cleanup_seconds':15., 'whole_supervisor_outer_timeout_required':True,
        'python':str(python), 'capacity_summary':str(summary), 'capacity_stop_receipt':str(stopped)}
    required = [python.resolve(), summary, stopped] + [v for k,v in paths.items() if k != 'repo']
    required += [Path(v['path']) for v in checkpoints.values()] + [Path(v) for v in releases.values()]
    required += [M.HERE/name for name in ('evaluate_goop3d_graph_support_v1.py','goop3d_native_evaluation_v1.py',
        'goop3d_diagnostic_metrics_v1.py','goop3d_graph_support_vectorized_v1.py','goop3d_deadline_worksheet_v1.py',
        'measure_sand_cuda_capacity_v2.py','audit_goop3d_auxiliary.py')]
    root['files_sha256'] = {str(p):M.sha(p) for p in required}
    release = tmp_path/'root_release.json'; release.write_text(json.dumps(root))
    return root, SimpleNamespace(output_dir=tmp_path/'output', release=release)


def mocks(root):
    E = SimpleNamespace(merge_bindings=lambda *groups:{str(Path(k).resolve()):v for g in groups for k,v in g.items()},
        release_gate=lambda args:{'scientific_training_admitted':False}, execution_bindings=lambda *a:{},
        verify_bindings=lambda bindings:None)
    W = SimpleNamespace(capacity_inputs=lambda report:({}, {}, {(a,0):root['checkpoints'][a]['sha256'] for a in ('base','mix')}))
    B = SimpleNamespace(canonical_gpu_uuid=lambda v:v.lower().removeprefix('gpu-'))
    return E, W, B, SimpleNamespace(SOURCE_PINS={})


def test_exact_six_commands_preserve_venv_path_and_never_request_test(tmp_path):
    root, args = root_values(tmp_path); E,W,B,T = mocks(root)
    specs, bindings = M.preflight(args,root,E,W,B,T)
    assert len(specs) == 6 and str(args.release) in bindings
    for job, child, argv in specs:
        assert argv[0] == root['python'] and argv[0] != str(Path(root['python']).resolve())
        assert child.split == 'valid' and child.checkpoint_updates == 512 and child.seed == 0
        assert argv[argv.index('--purpose')+1] == 'capacity_timing'
        assert argv[argv.index('--gpu-uuid')+1] == root['gpu_uuids'][job['gpu']]


@pytest.mark.parametrize('changed', ['scope','stale','bound','checkpoint','missing_child','same_uuid'])
def test_root_release_rejects_scope_staleness_budget_and_missing_coverage(tmp_path,changed):
    root,args = root_values(tmp_path); E,W,B,T = mocks(root)
    if changed == 'scope': root['test_access_allowed'] = True
    elif changed == 'stale': root['process_identity_checked_utc'] = (M.now()-timedelta(hours=1)).isoformat()
    elif changed == 'bound': root['outer_seconds']['full-rollout'] = 99
    elif changed == 'checkpoint': root['checkpoints']['base']['sha256'] = '0'*64
    elif changed == 'missing_child': root['evaluator_releases'].pop(next(iter(root['evaluator_releases'])))
    else: root['gpu_uuids'] = ['GPU-first','GPU-first']
    with pytest.raises(ValueError): M.preflight(args,root,E,W,B,T)


def test_wait4_receipt_records_raw_lifetime_reaping_and_usage(monkeypatch):
    proc = SimpleNamespace(pid=42,returncode=None); handles=(io.StringIO(),io.StringIO())
    child = dict(process=proc,handles=handles,identity={'start_ticks':123},command=['synthetic'],started_utc='start',started=1.,
        signals=[],stdout_file='stdout',stderr_file='stderr')
    monkeypatch.setattr(M.os,'wait4',lambda pid,options:(pid,0,SimpleNamespace(ru_maxrss=4,ru_utime=.2,ru_stime=.1)))
    monkeypatch.setattr(M.time,'perf_counter',lambda:3.5)
    result = M.reap(child)
    assert result['elapsed_seconds'] == 2.5 and result['stopped_and_reaped'] is True
    assert result['peak_host_rss_bytes'] == 4096 and proc.returncode == 0 and all(h.closed for h in handles)
    assert M.reap(child) is None


def test_duplicate_scan_excludes_only_actual_ancestors(tmp_path,monkeypatch):
    real_path=Path;proc=tmp_path/'proc';proc.mkdir()
    for pid,ppid in ((100,90),(90,1),(1,0),(77,1)):
        directory=proc/str(pid);directory.mkdir()
        (directory/'stat').write_text(f'{pid} (synthetic) S {ppid} 0 0')
        (directory/'cmdline').write_bytes(b'python\0/path/run_goop3d_validation_timing_v1.py\0')
    monkeypatch.setattr(M,'Path',lambda p:proc if str(p)=='/proc' else real_path(p))
    monkeypatch.setattr(M.os,'getpid',lambda:100)
    assert M.ancestor_pids()=={100,90,1}
    assert [r['pid'] for r in M.other_research_processes(M.ancestor_pids())]==[77]


def test_identity_capture_rejects_changed_command(monkeypatch):
    child={'process':SimpleNamespace(pid=42),'command':['expected'],'identity':None}
    B=SimpleNamespace(process_identity=lambda pid:{'ppid':M.os.getpid(),'argv':['wrong'],'start_ticks':1})
    with pytest.raises(ValueError,match='identity'):M.capture_identity(child,B)


def test_cleanup_escalates_only_owned_children_and_reports_unreaped(monkeypatch):
    clock=[0.];sent=[];child={'process':SimpleNamespace(pid=42,returncode=None),'job':{'id':'synthetic'},
        'identity':{'start_ticks':1},'command':['synthetic'],'signals':[]}
    monkeypatch.setattr(M.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(M.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(M,'reap',lambda child:None)
    B=SimpleNamespace(stop_owned=lambda children,signum:sent.append(signum))
    pending=M.cleanup([child],B,lambda *args:None)
    assert sent==[signal.SIGINT,signal.SIGTERM,signal.SIGKILL]
    assert pending[0]['pid']==42 and clock[0]>=15.


@pytest.mark.parametrize('mode', ['complete','guard','nonzero','timeout','second_launch','duplicate','contaminated','late_exit','late_source'])
def test_mock_supervisor_runs_once_and_preserves_missing_failures(tmp_path,monkeypatch,mode):
    root,args = root_values(tmp_path); E,W,B,T = mocks(root)
    specs,bindings = M.preflight(args,root,E,W,B,T)
    clock=[0.]; spawned=[]; signals=[]; inventories=[0];verifications=[0]
    def verify(bindings):
        verifications[0]+=1
        if mode=='late_source' and verifications[0]==5:raise ValueError('synthetic final source mutation')
    E.verify_bindings=verify
    monkeypatch.setattr(M.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(M.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(M,'ancestor_pids',lambda:{1})
    monkeypatch.setattr(M,'other_research_processes',lambda excluded:[{'pid':999}] if mode=='duplicate' else [])
    def gpu():
        inventories[0]+=1
        return [{'pid':999,'gpu_uuid':'GPU-first'}] if mode=='contaminated' and inventories[0]>1 else []
    B.gpu_processes=gpu
    def stop(children,signum):
        signals.append(signum)
        for child in children: child['stop_requested']=True;child['signals'].append({'signal':int(signum)})
    B.stop_owned=stop
    def spawn(output,job,cargs,argv,B):
        if mode=='second_launch' and len(spawned)==1:raise RuntimeError('synthetic launch failure')
        cargs.output_dir.mkdir(parents=True)
        M.write(cargs.output_dir/'status.json',{'state':'complete_with_guard_failures' if mode=='guard' else 'complete','all_inputs_reverified':True})
        proc=SimpleNamespace(pid=100+len(spawned),returncode=None)
        child=dict(job=job,args=cargs,process=proc,identity={'start_ticks':123},command=argv,handles=(),signals=[],started=clock[0])
        spawned.append(child);return child
    monkeypatch.setattr(M,'spawn',spawn);monkeypatch.setattr(M,'capture_identity',lambda *a:None)
    def reap(child):
        if child['process'].returncode is not None:return None
        if mode in ('timeout','contaminated') and not child.get('stop_requested'):return None
        if mode=='late_exit' and child is spawned[0] and not child.get('stop_requested'):clock[0]+=.2
        code=1 if mode=='nonzero' and child is spawned[0] else -2 if child.get('stop_requested') else 0
        child['process'].returncode=code
        return {'pid':child['process'].pid,'exit_code':code,'stopped_and_reaped':True,'elapsed_seconds':clock[0]-child['started'],
            'signals':child['signals']}
    monkeypatch.setattr(M,'reap',reap)
    if mode in ('timeout','late_exit'):root['outer_seconds']={m:.15 for m in M.MODES}
    result=M.supervise(args,root,specs,bindings,E,B)
    inventory=M.read(args.output_dir/'timing_inventory.json')
    assert len(inventory['entries'])==6 and len({c['job']['id'] for c in spawned})==len(spawned)
    assert not (args.output_dir/'run.lock').exists()
    if mode in ('complete','guard'):
        assert result==0 and len(spawned)==6 and not signals
        assert inventory['status']=='all_required_processes_stopped'
        if mode=='guard':assert all(e['evaluator_state']=='complete_with_guard_failures' for e in inventory['entries'])
    else:
        assert result==1 and inventory['failure'] and inventory['status']=='failed_or_incomplete_timing_collection'
        if mode=='late_source':
            assert len(spawned)==6 and inventory['all_planned_processes_stopped'] is True
        else:
            assert len(spawned)<6
            assert any(e['state']=='not_started' and e['unattempted_reason'] for e in inventory['entries'])
        assert (args.output_dir/'failed_attempt.json').exists()
        if mode in ('timeout','second_launch','contaminated'):assert signal.SIGINT in signals
