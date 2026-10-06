#!/usr/bin/env python3
"""One root-released gate phase on B0/1; default describes, never executes.

GPU0 runs mixseed0 then mixseed2; GPU1 runs mixseed1. Original Goop ownership
must be closed. Sand2/3 is observed without control. No retry or numeric change.
"""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SCHEMA='adaptgns_goop_action_gate_scoped_supervisor_v1'
RELEASE_SCHEMA='adaptgns_goop_action_gate_scoped_release_v1'
DRIVER_SHA='af51bbe073e52a22d4394918dbce4bc9c34a99645dc7f302b0cd5c48ebba5079'
PINS={'supervise_goop3d_final_evaluation_v1.py':'070d1d5e47cb4ac8650586f72ef434756c47b9b94da9d15016f7a287e945e989',
    'supervise_sand_scoped_science_v1.py':'2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13',
    'measure_sand_cuda_capacity_v2.py':'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd',
    'goop_global_action_gate_protocol_v1.md':'bc9235fae89d32ede8d1e7f846bff07bdcdd85f4671d5922d688a9535f7fe024'}
DEADLINE=datetime(2026,10,7,4,tzinfo=timezone.utc)
CLEANUP=15
SEEDS={0:[0,2],1:[1]}
PRIOR_ENTRIES={'supervise_graph_support_science_quota_v2.py','supervise_graph_support_evaluation_quota_v2.py',
    'train_goop_graph_support_cuda.py','evaluate_goop_graph_support_final.py','supervise_goop_evaluation_gpu_scoped_v3.py'}


def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def snapshot(path):
    raw=Path(path).read_bytes();return json.loads(raw),hashlib.sha256(raw).hexdigest()
def stamp(value):
    t=datetime.fromisoformat(value);require(t.tzinfo is not None,'Timezone-aware clock required');return t.astimezone(timezone.utc)
def now():return datetime.now(timezone.utc)
def load(name,pin):
    p=HERE/name;require(sha(p)==pin,'Reviewed operational/numerical source differs: '+name)
    spec=importlib.util.spec_from_file_location('_gate_scoped_'+name.replace('.','_'),p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def modules():
    return SimpleNamespace(R=load('supervise_goop3d_final_evaluation_v1.py',PINS['supervise_goop3d_final_evaluation_v1.py']),
        S=load('supervise_sand_scoped_science_v1.py',PINS['supervise_sand_scoped_science_v1.py']),
        B=load('measure_sand_cuda_capacity_v2.py',PINS['measure_sand_cuda_capacity_v2.py']),
        D=load('run_goop_action_gate_v1.py',DRIVER_SHA))
def verify_files(bindings):require(all(sha(p)==d for p,d in bindings.items()),'Pinned phase inputs changed')
def verify_metadata(c):require(all(c.mods.S.file_identity(p)==v for p,v in c.metadata.items()),'Pinned control/input metadata changed')


def validate_release(args,raw,mods,at=None):
    at=now() if at is None else at;r=json.loads(raw);R,S,B,D=mods.R,mods.S,mods.B,mods.D
    require(r.get('schema')==RELEASE_SCHEMA and r.get('status')=='approved_for_one_fixed_gate_phase' and r.get('issued_by')=='root'
        and r.get('hostname')==socket.gethostname() and r.get('host_role')=='B' and r.get('mode') in D.MODES
        and r.get('output_dir')==str(args.output_dir) and r.get('environment')==S.ENVIRONMENT
        and r.get('no_retry_or_resume') is True and r.get('does_not_displace_original_studies') is True
        and r.get('all_six_original_goop_models_frozen') is True,'Exact root phase/scientific-resource release required')
    require(r.get('python_environment')==S.python_environment(sys.executable),'Exact lexical venv interpreter required')
    error=r.get('clock_error_bound_seconds');reserve=r.get('remaining_outer_work_reserve_seconds')
    require(S.finite(error) and error<=5 and S.finite(reserve) and reserve>=3600
        and -5<=(at-stamp(r['process_clock_checked_utc'])).total_seconds()<=300,'Fresh root clock/process check and analysis reserve required')
    stop=stamp(r['absolute_stop_utc']);latest=stamp(r['latest_start_utc'])
    require(stop<=DEADLINE-timedelta(seconds=reserve) and at+timedelta(seconds=error)<=latest,'Whole phase must fit before retained analysis reserve')
    require(r.get('gpu_scope')=={'owned_indices':[0,1],'unassigned_devices':'observe_without_control','live_handoff':False}
        and isinstance(r.get('gpu_uuids'),list) and len(r['gpu_uuids'])==4
        and len({B.canonical_gpu_uuid(v) for v in r['gpu_uuids']})==4,'Exact B0/1 physical scope and all four UUIDs required')
    closed=r.get('closed_goop_processes')
    require(isinstance(closed,list) and bool(closed) and all(type(v.get('pid')) is int and v['pid']>0
        and type(v.get('start_ticks')) is int and v['start_ticks']>0 for v in closed),'Original Goop exited identities required')
    pins=r.get('files_sha256',{});require(isinstance(pins,dict),'Complete source/control file closure required')
    fixed={str(HERE/n):p for n,p in PINS.items()};fixed[str(HERE/'run_goop_action_gate_v1.py')]=DRIVER_SHA;fixed[str(Path(__file__).resolve())]=sha(__file__)
    env=r['python_environment'];fixed[env['lexical_path']]=env['binary_sha256'];fixed[env['resolved_binary_path']]=env['binary_sha256']
    if env['pyvenv_config_path']:fixed[env['pyvenv_config_path']]=env['pyvenv_config_sha256']
    require(all(pins.get(p)==d for p,d in fixed.items()),'Pinned reviewed runtime/driver/venv closure required')
    jobs=r.get('jobs',[]);require(len(jobs)==3 and [j.get('seed') for j in jobs]==[0,1,2],'All three fixed seed commands required')
    entries=[];bindings={**pins,str(args.release):hashlib.sha256(raw).hexdigest()};cohort=None
    for job in jobs:
        seed=job['seed'];gpu=1 if seed==1 else 0;command=job.get('command',[])
        require(type(seed) is int and job.get('gpu')==gpu and len(command)>3 and command[:3]==[sys.executable,str(HERE/'run_goop_action_gate_v1.py'),'--execute'],
            'Exact lexical interpreter, frozen driver and fixed seed/device command required')
        flags=command[3:];require(len(flags)%2==0 and len(set(flags[::2]))==len(flags[::2]) and all(k.startswith('--') for k in flags[::2]),'Exact unique-valued driver flags required')
        a=D.parse_args(command[2:]);require(a.seed==seed and a.cuda_index==gpu and a.mode==r['mode'] and a.threads==2
            and type(a.max_seconds) is int and a.max_seconds>0 and a.output_dir==args.output_dir/'jobs'/f'mix_seed{seed}', 'Exact fixed model/mode/quota/output required')
        ctx=D.release_gate(a,at);require(stamp(ctx.release['absolute_stop_utc'])==stop,'Driver and supervisor stop must agree')
        require(cohort in (None,ctx.cohort_sha),'One complete original six-model Goop cohort required');cohort=ctx.cohort_sha
        for p,d in ctx.bindings.items():require(pins.get(p)==d,'Root must bind every child source/control input');bindings[p]=d
        expected=D.expected_schedule(ctx.mods.core,a.mode,ctx.manifest)
        entries.append({'stream':f'mix_seed{seed}','stage':a.mode,'seed':seed,'arm':'mix','gpu':gpu,'gpu_uuid':r['gpu_uuids'][gpu],
            'command':command,'directory':str(a.output_dir),'outer_timeout_seconds':a.max_seconds,'expected_cells':expected})
    allocation=max(sum(entries[s]['outer_timeout_seconds']+CLEANUP for s in seeds) for seeds in SEEDS.values())
    require(latest<=stop-timedelta(seconds=allocation+error),'Both fixed waves and bounded cleanup must fit')
    if r['mode']=='test-rollout':
        cap=r.get('complete_capacity_report');require(isinstance(cap,dict) and pins.get(cap['path'])==cap['sha256'],'Root complete cost receipt required')
        report,report_sha=snapshot(cap['path']);require(report_sha==cap['sha256'] and report.get('schema')=='adaptgns_goop_global_action_gate_capacity_budget_v1'
            and report.get('status')=='complete_measured_capacity_passed' and report.get('all_36_rollouts_complete') is True
            and report.get('all_train_validation_labels_complete') is True and report.get('complete_remaining_work_fits') is True,
            'Complete fixed study/capacity review required before test')
        require([e['outer_timeout_seconds'] for e in entries]==[v['test_allocation_seconds'] for v in report['per_seed']]
            and allocation==report['test_concurrent_allocation_seconds'] and reserve==sum(report['remaining_outer_reserves_seconds'].values())
            and latest<=stamp(report['latest_test_start_utc']),'Exact complete measured test-cost arithmetic required')
    require(not args.output_dir.exists() and all(args.output_dir!=Path(p) and args.output_dir not in Path(p).parents for p in bindings),'Fresh output outside protected inputs required')
    verify_files(bindings)
    return SimpleNamespace(release=r,raw=raw,bindings=bindings,metadata={p:S.file_identity(p) for p in bindings},entries=entries,mods=mods,stop=stop)


def closed_ownership(c,rt,registry):
    for prior in c.release['closed_goop_processes']:
        try:live=rt.identity(prior['pid'])
        except (FileNotFoundError,ProcessLookupError):live=None
        require(live is None or live.get('start_ticks')!=prior['start_ticks'],'Original Goop process identity is still live')
    own={v['process'].pid:v for v in registry}
    for row in rt.process_inventory():
        script=c.mods.S.executed_python_script(row['argv'])
        require(script not in PRIOR_ENTRIES,'Original Goop ownership has not fully closed')
        if script in {'run_goop_action_gate_v1.py',Path(__file__).name} and row['pid']!=os.getpid():
            child=own.get(row['pid']);require(child is not None and child['process'].returncode is None
                and child['identity']['start_ticks']==row['start_ticks'] and child['command']==row['argv'],'Another gate process is live; no duplicate phase launch')


def collection_complete(entry):
    filename='label_collection.json' if 'labels' in entry['stage'] or entry['stage']=='train-label-capacity' else 'rollout_collection.json'
    path=Path(entry['directory'])/filename
    require(path.exists(),'No complete driver collection; preserve partial output')
    value,digest=snapshot(path);allowed={'complete','complete_with_guard_failures'} if entry['stage']=='test-rollout' else {'complete'}
    expected=entry['expected_cells'];key='target_frame' if 'target_frame' in expected[0] else 'policy'
    require(value.get('status') in allowed and value.get('mode')==entry['stage'] and value.get('model',{}).get('seed')==entry['seed']
        and value.get('required_rows')==value.get('committed_rows')==len(expected) and value.get('all_inputs_reverified') is True
        and value.get('model_state_verified_unchanged') is True and value.get('abort_reason') is None
        and [(r.get('source_index'),r.get(key)) for r in value.get('rows',[])]==[(r['source_index'],r[key]) for r in expected],
        'Every fixed driver outcome must be committed with unchanged sources/model')
    return {'path':str(path),'sha256':digest,'status':value['status'],'required_rows':len(expected),'committed_rows':len(value['rows'])}


def terminal_complete(entry):
    value,_=snapshot(Path(entry['directory'])/'status.json')
    allowed={'complete','complete_with_guard_failures'} if entry['stage']=='test-rollout' else {'complete'}
    require(value.get('state') in allowed and value.get('required_rows')==value.get('committed_rows')==len(entry['expected_cells'])
        and value.get('all_inputs_reverified') is True and value.get('model_state_verified_unchanged') is True
        and value.get('abort_reason') is None,'Driver terminal status is incomplete or failed')


def parent_snapshot(output):
    """Only supervisor/scalar evidence; array bytes belong to the root collector."""
    files={}
    for p in sorted(output.iterdir()):
        if p.name in ('jobs','phase_ledger.json'):continue
        for file in ([p] if p.is_file() else sorted(p.rglob('*'))):
            if file.is_dir():continue
            require(file.is_file() and not file.is_symlink(),'Ordinary supervisor evidence files required')
            files[str(file.relative_to(output))]={'sha256':sha(file),'bytes':file.stat().st_size}
    return files


def retained_outcomes(c,registry,outcomes,reason):
    """A failed accounting read must never relabel a launched child never-started."""
    R=c.mods.R;owned={R.entry_key(v['entry']):v for v in registry};rows=[]
    for entry in c.entries:
        key=R.entry_key(entry);value=outcomes.get(key);child=owned.get(key)
        if value is None and child is None:value=R.never_started(entry,reason or 'Not started')
        elif value is None:
            value={'started':True,'state':'unreaped_owned_child' if child['process'].returncode is None else 'stopped_and_reaped_unclassified',
                'stopped_and_reaped':child['process'].returncode is not None,'pid':child['process'].pid,'command':child['command'],
                'initial_process_identity':child['identity'],'exit_code':child['process'].returncode,'signals':child['signals'],
                'termination_reason':'Outcome accounting failed; original process evidence retained','current_before_stop':child['current_before_stop'],
                'quota_stop_initiated':child['quota_stop_initiated'],'quota_expired_at_observation':child['quota_expired'],
                'elapsed_seconds':None,'inner_timeout_reported':None}
        failure=value.get('worker_failed_attempt')
        nested=failure.get('failure',{}) if isinstance(failure,dict) else {}
        if nested.get('category')=='execution_budget' or isinstance(nested.get('failure'),dict) and nested['failure'].get('category')=='execution_budget':
            value['inner_timeout_reported']=True
            if not value.get('quota_stop_initiated'):value['termination_reason']='inner_driver_execution_budget'
        rows.append({'entry':entry,'outcome':value})
    return rows


class Runtime:
    def __init__(self,mods):self.inner=mods.R.Runtime(mods.B);self.mods=mods
    def __getattr__(self,name):return getattr(self.inner,name)
    def process_inventory(self):
        rows=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:row=self.identity(int(p.name))
            except (FileNotFoundError,ProcessLookupError):continue
            script=self.mods.S.executed_python_script(row['argv'])
            if script in PRIOR_ENTRIES|{'run_goop_action_gate_v1.py',Path(__file__).name}:rows.append(row)
        return rows


def execute_queue(args,c,rt):
    R,S=c.mods.R,c.mods.S;output=args.output_dir;output.mkdir(mode=0o700);(output/'jobs').mkdir();(output/'logs').mkdir()
    (output/'release_snapshot.json').write_bytes(c.raw);registry=[];outcomes={};abort=None;started=rt.mono();last_gpu=-30;last_status=-5
    states={gpu:{'pending':list(seeds),'active':None} for gpu,seeds in SEEDS.items()}
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(sig,frame):
        nonlocal abort;abort='Supervisor received '+signal.Signals(sig).name
    for sig in previous:signal.signal(sig,interrupted)
    def observe(rows):
        index=len(list((output/'logs').glob('gpu_*.json')));path=output/'logs'/f'gpu_{index:06d}.json';value={'raw':rows,'utc':rt.now().isoformat()}
        R.write(path,value);value['checked']=S.check_gpu(rows,{v['process'].pid:v for v in registry},c.mods.B,c.release['gpu_uuids'],[0,1],rt.gpu_identity);R.write(path,value)
    try:
        closed_ownership(c,rt,registry);observe(rt.gpu_now());verify_files(c.bindings)
        while any(s['active'] or (s['pending'] and not abort) for s in states.values()):
            if rt.now()+timedelta(seconds=c.release['clock_error_bound_seconds']+CLEANUP)>=c.stop:abort=abort or 'Reserved phase cutoff reached'
            observation=rt.gpu_result()
            if observation is not None:observe(observation[0])
            for gpu,state in states.items():
                child=state['active']
                if child is None and state['pending'] and not abort:
                    remaining=sum(c.entries[s]['outer_timeout_seconds']+CLEANUP for s in state['pending'])
                    verify_metadata(c);closed_ownership(c,rt,registry)
                    require(not abort and rt.now()+timedelta(seconds=remaining+c.release['clock_error_bound_seconds'])<=c.stop,'Complete remaining fixed wave no longer fits')
                    if state['pending'][0] in (0,1):require(rt.now()+timedelta(seconds=c.release['clock_error_bound_seconds'])<=stamp(c.release['latest_start_utc']),'Phase launch window closed')
                    entry=c.entries[state['pending'][0]];begin=rt.mono();at=rt.now();process,handles=rt.launch(entry,output/'logs')
                    child={'entry':entry,'job':entry,'process':process,'handles':handles,'identity':None,'command':entry['command'],'signals':[],
                        'started':begin,'started_utc':at.isoformat(),'deadline':begin+entry['outer_timeout_seconds'],
                        'absolute_stop_utc':(at+timedelta(seconds=entry['outer_timeout_seconds'])).isoformat(),'stop_started':None,'stop_reason':None,
                        'quota_expired':False,'quota_stop_initiated':False,'current_before_stop':None}
                    registry.append(child);state['active']=child;state['pending'].pop(0)
                    child['identity']=rt.identity(process.pid);require(child['identity'].get('ppid')==os.getpid() and child['identity'].get('argv')==entry['command'],
                        'Owned child process identity differs')
                    R.write(output/'logs'/f"mix_seed{entry['seed']}.launch.json",{'entry':entry,'identity':child['identity'],'started_utc':at.isoformat()})
                if child is None:continue
                usage=rt.reap(child)
                if usage is not None:
                    for h in child['handles']:h.close()
                    outcome=R.observe_reaped(child,usage,rt,c.release['hostname']);outcomes[R.entry_key(child['entry'])]=outcome;state['active']=None
                    R.write(output/'logs'/f"mix_seed{child['entry']['seed']}.outcome.json",outcome)
                    if outcome['exit_code']!=0 or outcome['quota_expired_at_observation'] or outcome.get('worker_failed_attempt') or outcome.get('worker_failed_attempt_parse_error'):
                        abort=abort or 'Incomplete, timed-out or failed driver invocation; root review required'
                    else:
                        try:terminal_complete(child['entry'])
                        except Exception as error:abort=abort or str(error)
                    continue
                if abort or rt.mono()>=child['deadline']:
                    abort=abort or 'Whole driver invocation quota expired';child['quota_expired']=rt.mono()>=child['deadline']
                    child['quota_stop_initiated']=child['quota_expired'];child['stop_reason']=abort
            if rt.mono()-last_status>=5 or abort:
                R.write(output/'queue_status.json',{'schema':SCHEMA,'state':'aborting' if abort else 'running','abort_reason':abort,
                    'states':{str(g):{'pending_seeds':s['pending'],'active_pid':s['active']['process'].pid if s['active'] else None} for g,s in states.items()}});last_status=rt.mono()
            if abort:break
            if rt.future is None and rt.mono()-last_gpu>=30:rt.gpu_submit({v['process'].pid:v for v in registry});last_gpu=rt.mono()
            rt.sleep(.2)
    except BaseException as error:abort=abort or f'{type(error).__name__}: {error}'
    finally:
        R.cleanup_owned(registry,outcomes,c,rt,abort or 'Final owned-child cleanup')
        for sig,handler in previous.items():signal.signal(sig,handler)
        rt.close()
        for child in registry:
            for h in child['handles']:h.close()
    unreaped=[v['process'].pid for v in registry if v['process'].returncode is None]
    collection_bindings={}
    if not unreaped:
        for entry in c.entries:
            outcome=outcomes.get(R.entry_key(entry))
            if outcome is None or outcome['exit_code']!=0:continue
            try:
                receipt=collection_complete(entry);outcome['complete_collection']=receipt;collection_bindings[receipt['path']]=receipt['sha256']
            except Exception as error:outcome['completion_validation_error']=str(error);abort=abort or str(error)
    rows=retained_outcomes(c,registry,outcomes,abort)
    owned_registry=[{'entry':v['entry'],'pid':v['process'].pid,'identity':v['identity'],'returncode':v['process'].returncode,
        'started_utc':v['started_utc'],'signals':v['signals'],'cleanup_errors':v.get('cleanup_errors',[])} for v in registry]
    R.write(output/'process_outcomes.json',{'schema':SCHEMA,'jobs':rows,'owned_registry':owned_registry,
        'unreaped_owned_children':unreaped,'abort_reason':abort})
    try:
        require(not unreaped,'Unreaped owned process prevents final phase publication');verify_files(c.bindings)
        tree=parent_snapshot(output);value={'schema':SCHEMA,'mode':c.release['mode'],'state':'stopped_requires_review' if abort else 'complete_fixed_phase',
            'jobs':rows,'abort_reason':abort,'elapsed_seconds':rt.mono()-started,'all_pinned_inputs_reverified':True,'supervisor_files':tree,
            'input_sha256':c.bindings,'numerical_arrays_audited_by_supervisor':False,'automatic_retry':False,'test_admitted_by_this_ledger':False}
        raw=encode(value);verify_files({**c.bindings,**collection_bindings});require(parent_snapshot(output)==tree,'Stopped supervisor output bytes changed')
        require(rt.now()<DEADLINE-timedelta(seconds=3600),'Independent analysis reserve reached before phase publication')
        with (output/'phase_ledger.json').open('xb') as f:f.write(raw)
    except BaseException as error:
        R.write(output/'failed_publication.json',{'schema':SCHEMA,'error':str(error),'all_outputs_retained':True,'unreaped_owned_children':unreaped});return 1
    return 1 if abort else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    p.add_argument('--release',type=Path);p.add_argument('--output-dir',type=Path);args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','driver_reviewed':len(DRIVER_SHA)==64,'mapping':SEEDS,'owns_only_B_GPUs':[0,1]}));return 0
    require(args.release and args.output_dir and sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES') is None,'Explicit unremapped Linux phase required')
    args.release=args.release.resolve();args.output_dir=args.output_dir.resolve();raw=args.release.read_bytes();mods=modules()
    c=validate_release(args,raw,mods);return execute_queue(args,c,Runtime(mods))


if __name__=='__main__':raise SystemExit(main())
