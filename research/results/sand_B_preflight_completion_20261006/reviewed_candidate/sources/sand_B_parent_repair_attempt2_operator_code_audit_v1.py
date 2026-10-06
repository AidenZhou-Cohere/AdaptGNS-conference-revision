#!/usr/bin/env python3
"""Root-only manual owner-startup repair; never a new phase or scientific retry.

Actions: prepare, setup, issue, stage, then original run/record-original-exit/collect.
The frozen worker, owner, bootstrap and scalar core releases are unchanged.
"""
from pathlib import Path
import base64,datetime,hashlib,importlib.util,json,sys
P=Path(__file__).resolve().parent;C=P/'sand_B_preflight_amendment_released_root_v1';D=C/'attempt2'
R='/root/repos/AdaptGNS-cuda-20261006';Q=R+'/sand_reserved_B_preflight_amendment_20261006_v1';OPS=('preflight_valid','preflight_test')
OLD_OPERATOR='34b003818efbf06ca0a1eb348c204d4d06f7e1104d295bf146e738595b5a8e9b'
PHASE='e22a9b0e23eae4cccb872a7543805fdaccf3ea2d9d6a875fdec17d7c2271ef39'
STOP='2026-10-06T20:00:02.215311+00:00';HOST='aidenzhou-aquamarine-toad-75-6d8b45c98d-mgjlq'
def rd(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def dt(v):return datetime.datetime.fromisoformat(v)
def now():return datetime.datetime.now(datetime.timezone.utc)
SETUP=r'''import pathlib,json,sys,datetime,time,socket,subprocess,hashlib,os
v=json.load(sys.stdin);q=pathlib.Path(v['root']);assert socket.gethostname()==v['hostname'];assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==v['boot_id'];assert datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=15)<datetime.datetime.fromisoformat(v['stop_utc'])
assert q.is_dir() and not q.is_symlink() and str(q.resolve())==str(q)
for path,h in v['inputs'].items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==h
matching=[]
for x in pathlib.Path('/proc').iterdir():
 if not x.name.isdigit():continue
 try:a=[z.decode(errors='surrogateescape') for z in (x/'cmdline').read_bytes().split(b'\0') if z]
 except (FileNotFoundError,ProcessLookupError):continue
 if any(s in a for s in v['forbidden_program_tokens']):matching.append({'pid':int(x.name),'argv':a})
assert matching==[]
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
for x in v['absent_paths']:assert not pathlib.Path(x).exists()
parent=q/'owners';parent.mkdir(mode=0o700,exist_ok=False);assert list(parent.iterdir())==[]
assert datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=5)<datetime.datetime.fromisoformat(v['stop_utc'])
print(json.dumps({'hostname':socket.gethostname(),'created_empty_parent':str(parent),'operation_and_worker_leaves_absent':True,'matching_processes':matching,'clock':{'host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_monotonic_seconds':time.monotonic(),'host_boot_id':v['boot_id']},'same_stop_utc':v['stop_utc']}))'''
STAGE=r'''import pathlib,json,sys,hashlib,base64,datetime
v=json.load(sys.stdin);assert datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=10)<datetime.datetime.fromisoformat(v['stop_utc']);q=pathlib.Path(v['root']);archive=q/'controls'/'attempt1_cpu_controls';archive.mkdir(mode=0o700,exist_ok=False)
rows=[]
for r in v['files']:
 p=pathlib.Path(r['path']);assert str(p.parent)==str(q/'controls');assert hashlib.sha256(p.read_bytes()).hexdigest()==r['old_sha256'];old=archive/p.name;assert not old.exists();p.rename(old);assert hashlib.sha256(old.read_bytes()).hexdigest()==r['old_sha256'];b=base64.b64decode(r['base64']);assert hashlib.sha256(b).hexdigest()==r['sha256']
 with p.open('xb') as f:f.write(b);f.flush()
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];rows.append({'path':str(p),'sha256':r['sha256'],'preserved_attempt1_path':str(old),'preserved_attempt1_sha256':r['old_sha256']})
assert datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=5)<datetime.datetime.fromisoformat(v['stop_utc']);print(json.dumps({'same_stop_utc':v['stop_utc'],'files':rows,'original_phase_core_scientific_files_unchanged':True}))'''
def main():
    action=sys.argv[1] if len(sys.argv)>1 else 'describe'
    if action=='describe':print(json.dumps({'status':'inert_root_only_manual_attempt2','phase_restarted':False,'same_stop_utc':STOP}));return
    assert not sys.flags.optimize and sha(P/'sand_B_preflight_amendment_operator_code_audit_v1.py')==OLD_OPERATOR and sha(C/'amendment_phase.json')==PHASE
    assert rd(C/'amendment_phase.json')['amendment_stop_utc']==STOP and now()<dt(STOP)
    if action=='prepare':
        closure_path=P/'sand_B_failed_start_closure_1953_root_v1.json';closure=rd(closure_path)
        assert closure['hostname']==HOST and closure['matching_processes']==[] and closure['gpu_apps']=='' and all(v['exists'] is False for v in closure['paths'].values())
        failed={}
        for op,session in zip(OPS,(48709,4106)):
            e=rd(C/(op+'.external.json'));original=rd(C/(op+'.root_original_external_exit.json'));err=(C/(op+'.stderr')).read_text()
            assert e['exit_code']==original['original_tool_exit_code']==1 and original['original_tool_session']==session and e['local_transport_reaped'] and not e['local_transport_timeout'] and not e['signals_to_own_local_group']
            assert 'line 635, in execute' in err and 'FileNotFoundError' in err and Q+'/owners/'+op in err and dt(original['observed_utc'])<=dt(closure['checked_utc'])
            failed[op]={suffix:sha(C/(op+suffix)) for suffix in ('.command.json','.external.json','.stderr','.stdout','.root_original_external_exit.json','.cpu_release.json','.mode_release.json','.launch.json')}
        D.mkdir(exist_ok=False)
        for n in ('human_approval.json','amendment_phase.json')+tuple(op+'.mode_release.json' for op in OPS):
            with (D/n).open('xb') as f:f.write((C/n).read_bytes())
        put(D/'manual_attempt2_root_authorization.json',{'status':'root_directed_one_operational_startup_repair','authorization_basis':'Direct human routine operational-adjustment authorization and root request to repair the confirmed missing parent after first owner startups failed before any worker.','same_phase_sha256':PHASE,'same_stop_utc':STOP,'first_failed_attempts':failed,'root_native_closure_path':str(closure_path),'root_native_closure_sha256':sha(closure_path),'original_tool_sessions':[48709,4106],'workers_started_in_attempt1':False,'scientific_rerun_authorized':False,'automatic_retry':False,'maximum_additional_owner_attempts_per_operation':1,'created_utc':now().isoformat(),'source_sha256':sha(__file__)})
        print(json.dumps({'attempt2':str(D),'phase_sha256':PHASE,'same_stop_utc':STOP}));return
    spec=importlib.util.spec_from_file_location('_frozen_sand_B_operator',P/'sand_B_preflight_amendment_operator_code_audit_v1.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o);o.C=D;o.package_check();o.approved();assert sha(D/'amendment_phase.json')==PHASE
    auth=rd(D/'manual_attempt2_root_authorization.json');assert auth['source_sha256']==sha(__file__) and sha(auth['root_native_closure_path'])==auth['root_native_closure_sha256']
    if action=='setup':
        inputs={Q+'/controls/'+n:sha(C/n) for n in ['human_approval.json','amendment_phase.json']+[op+s for op in OPS for s in ('.mode_release.json','.cpu_release.json')]}
        absent=[Q+'/owners']+[Q+'/owners/'+op for op in OPS]+[Q+'/'+op for op in OPS]
        v=o.ssh(SETUP,{'root':Q,'hostname':HOST,'boot_id':rd(auth['root_native_closure_path'])['boot_id'],'stop_utc':STOP,'inputs':inputs,'absent_paths':absent,'forbidden_program_tokens':[R+'/cuda_preparation/supervise_sand_B_preflight_amendment_cpu_v1.py',R+'/cuda_preparation/prepare_sand_reserved_test_recovered_v1.py',R+'/cuda_preparation/launch_sand_B_preflight_amendment_cpu_v1.py']},'setup_parent',45)
        v['clock'].update(source='root_fresh_tool_and_host_clock_evidence',root_reference_utc=now().isoformat(),error_bound_seconds=5);assert abs((dt(v['clock']['host_utc'])-dt(v['clock']['root_reference_utc'])).total_seconds())<=5;put(D/'setup_parent.json',v);print(json.dumps(v));return
    if action=='issue':
        clock=rd(D/'setup_parent.json')['clock'];origin=dt(clock['host_utc']);at=now();assert origin<=at<origin+datetime.timedelta(seconds=300) and at+datetime.timedelta(seconds=30)<dt(STOP)
        for op in OPS:
            r=rd(C/(op+'.cpu_release.json'));r.update(clock_sample=clock,issued_utc=at.isoformat(),invocation_origin_utc=origin.isoformat(),remaining_shared_amendment_seconds=(dt(STOP)-origin).total_seconds(),hard_deadline_monotonic_ns=int((clock['host_monotonic_seconds']+(dt(STOP)-origin).total_seconds()-5)*10**9));put(D/(op+'.cpu_release.json'),r)
            argv=rd(C/(op+'.launch.json'))['argv'];argv[argv.index('--hard-deadline-monotonic-ns')+1]=str(r['hard_deadline_monotonic_ns']);argv[argv.index('--release-sha256')+1]=sha(D/(op+'.cpu_release.json'));put(D/(op+'.launch.json'),{'argv':argv,'command':r['command']})
        print(json.dumps({'phase_sha256':PHASE,'same_stop_utc':STOP,'operational_releases':{op:sha(D/(op+'.cpu_release.json')) for op in OPS}}));return
    if action=='stage':
        files=[{'path':Q+'/controls/'+op+'.cpu_release.json','old_sha256':sha(C/(op+'.cpu_release.json')),'sha256':sha(D/(op+'.cpu_release.json')),'base64':base64.b64encode((D/(op+'.cpu_release.json')).read_bytes()).decode()} for op in OPS]
        v=o.ssh(STAGE,{'root':Q,'stop_utc':STOP,'files':files},'stage_attempt2',45);assert len(v['files'])==2;put(D/'stage.json',v);print(json.dumps(v));return
    assert action in ('run','record-original-exit','collect');o.main()
if __name__=='__main__':main()
