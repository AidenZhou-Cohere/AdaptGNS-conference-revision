#!/usr/bin/env python3
"""Separate scoped Sand100k orchestration; default only describes the contract.

Frozen numerical source and endpoint validators are imported unchanged. Root
must close old Goop ownership before this new supervisor is admitted. No live
handoff, network, test access, retries, resume or endpoint substitution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SCHEMA='adaptgns_sand_scoped_scientific_supervisor_v1'
RELEASE_SCHEMA='adaptgns_sand_scoped_scientific_release_v1'
VERIFIER_SHA='a22a59c46231c328e4939a0f021b3bb2fab9e2f93c0848bb1abd39aad2d5d305'
PLAN_SHA='403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b'
AMENDMENT_SHA='411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738'
PROTOCOL_SHA='e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d'
TRAINER_SHA='fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
CAPACITY_SHA='ca9b5ed45455b575179464d35c42d0d03e87edb7508f4deea2451f9f3fb50878'
LIFECYCLE_SHA='c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd'
DEADLINE=datetime(2026,10,7,4,tzinfo=timezone.utc)
TRAINING_STOP=datetime(2026,10,6,22,44,tzinfo=timezone.utc)
LATEST_START=datetime(2026,10,6,15,5,tzinfo=timezone.utc)
TRAINING_SECONDS=27404.671591931674
POSTTRAINING_SECONDS=18960
CLEANUP=15
SCHEDULE=[dict(id=f'{arm}_seed{seed}',wave=role,gpu=gpu,objective='faithful',arm=arm,seed=seed)
 for role,arm,seed,gpu in [('B','base',0,2),('B','mix',0,3),('A','base',1,0),('A','mix',1,1),('A','base',2,2),('A','mix',2,3)]]
ENVIRONMENT={'CUBLAS_WORKSPACE_CONFIG':':4096:8','LD_LIBRARY_PATH':'/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
# These old orchestration loops treat every GPU PID as exclusive ownership.
OLD_BROAD_SOURCES={'supervise_graph_support_science_quota_v2.py','supervise_graph_support_evaluation_quota_v2.py',
 'supervise_sand_graph_support_science.py','supervise_goop3d_science_v1.py'}
TRAINING_SOURCES={'train_sand_graph_support_cuda.py'}
FIELDS=('release','repo','trainer','capacity_source','lifecycle_source','verifier_source','train_manifest','admission',
 'structural_report','protocol','schedule_plan','amendment','process_check','clock_check','python','output_dir')


def require(v,message):
    if not v:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())
def utc():return datetime.now(timezone.utc)
def stamp(value):
    t=datetime.fromisoformat(value);require(t.tzinfo is not None,'Timezone-aware timestamp required');return t.astimezone(timezone.utc)
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def write(path,value):
    path=Path(path);p=path.with_name(path.name+'.tmp')
    with p.open('xb') as f:f.write(encode(value));f.flush();os.fsync(f.fileno())
    p.replace(path)
def finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
def file_identity(path):
    s=Path(path).stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def python_environment(path):
    p=Path(path)
    require(p.is_absolute() and str(p)==os.path.abspath(p) and str(p)==sys.executable,
        'Preserve the current interpreter lexical absolute path, including its venv symlink')
    cfg=p.parent.parent/'pyvenv.cfg'
    return {'lexical_path':str(p),'resolved_binary_path':str(p.resolve()),'binary_sha256':sha(p),
        'sys_prefix':sys.prefix,'sys_base_prefix':sys.base_prefix,
        'pyvenv_config_path':str(cfg) if cfg.exists() else None,'pyvenv_config_sha256':sha(cfg) if cfg.exists() else None}


def executed_python_script(argv):
    """Identify the direct entry point, never filenames used as option values."""
    if not isinstance(argv,list) or len(argv)<2:return None
    name=Path(argv[0]).name
    if not name.startswith('python') or any(c not in '0123456789.' for c in name[6:]):return None
    index=1
    while index<len(argv) and argv[index] in ('-u','-B'):index+=1
    if index==len(argv) or argv[index].startswith('-'):return None
    return Path(argv[index]).name


def configure(args):
    require(sha(args.verifier_source)==VERIFIER_SHA,'Frozen endpoint verifier differs')
    spec=importlib.util.spec_from_file_location('_sand_scoped_frozen_verifier',args.verifier_source)
    S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);S.configure(args)
    require(sha(args.trainer)==TRAINER_SHA and sha(args.capacity_source)==CAPACITY_SHA
        and sha(args.lifecycle_source)==LIFECYCLE_SHA,'Frozen trainer/helper source differs')
    return SimpleNamespace(S=S,B=S.B,T=S.T,C=S.C)


def schedule_contract(plan):
    require(plan.get('schema')=='adaptgns_sand_scoped_schedule_fixed_spec_v2'
        and plan.get('status')=='fixed_root_selected_plan_not_execution_admission' and plan.get('required_updates_per_model')==100000
        and plan.get('required_models')==6 and plan.get('automatic_retry_or_endpoint_selection') is False,'Fixed prospective scheduling spec required')
    d=plan['deadline_allocation'];training=plan['measured_training_inputs']['per_simultaneous_group_training_allocation_seconds']
    require(training==TRAINING_SECONDS and d['post_training_allocation_seconds']==POSTTRAINING_SECONDS
        and stamp(d['compute_analysis_end_utc'])==DEADLINE and stamp(d['common_training_stop_utc'])==TRAINING_STOP
        and stamp(d['root_selected_yellow_latest_start_utc'])==LATEST_START
        and TRAINING_STOP+timedelta(seconds=POSTTRAINING_SECONDS)==DEADLINE,'Fixed training/posttraining budget differs')
    expected={'aquamarine_pair0':[{'arm':j['arm'],'seed':j['seed'],'gpu':j['gpu']} for j in SCHEDULE if j['wave']=='B'],
        'yellow_pairs1_2':[{'arm':j['arm'],'seed':j['seed'],'gpu':j['gpu']} for j in SCHEDULE if j['wave']=='A']}
    require({g['id']:g['mapping'] for g in plan['groups']}==expected,'Fixed host/seed/GPU mapping differs')


def validate_release(args,release,mods,hashes,now):
    require(release.get('schema')==RELEASE_SCHEMA and release.get('status')=='admitted_fresh_scoped_training'
        and release.get('issued_by')=='root' and release.get('host')==socket.gethostname() and release.get('host_role')==args.host_role
        and release.get('dataset')=='Sand' and release.get('schedule')==SCHEDULE and release.get('environment')==ENVIRONMENT
        and release.get('files_sha256')==hashes and release.get('output_dir')==str(args.output_dir)
        and release.get('scientific_training_admitted') is True and release.get('automatic_retry_resume_or_promotion') is False
        and release.get('live_supervisor_handoff') is False and release.get('six_model_freeze_before_test') is True
        and isinstance(release.get('cohort_id'),str) and bool(release['cohort_id'].strip()),'Exact root scoped scientific release required')
    require(release.get('python_environment')==python_environment(args.python),'Root-bound lexical interpreter/venv provenance differs')
    schedule_contract(read(args.schedule_plan))
    uuids=release.get('gpu_uuids');jobs=[j for j in SCHEDULE if j['wave']==args.host_role]
    require(isinstance(uuids,list) and len(uuids)==4 and len({mods.B.canonical_gpu_uuid(u) for u in uuids})==4
        and release.get('owned_gpu_indices')==[j['gpu'] for j in jobs], 'All physical UUIDs and exact owned scope required')
    require(release.get('gpu_policy')=='exclusive_owned_devices_observe_all_others_without_control'
        and release.get('co_resident_runtime_bound_measured') is False,'Scoped observations and unmeasured co-resident throughput must be explicit')
    process=read(args.process_check);clock=read(args.clock_check)
    require(process.get('schema')=='adaptgns_sand_scoped_process_check_v1' and process.get('issued_by')=='root'
        and process.get('host')==release['host'] and process.get('host_role')==args.host_role and process.get('gpu_uuids')==uuids
        and process.get('matching_sand_training_processes')==[] and process.get('active_old_whole_host_monitors')==[]
        and process.get('all_prior_goop_trainers_reaped') is True and process.get('all_prior_whole_host_goop_monitors_exited') is True
        and (args.host_role!='A' or process.get('all_prior_goop_gpu_work_reaped') is True),'Prior Goop ownership must end before scoped launch')
    prior=process.get('closed_goop_processes')
    require(isinstance(prior,list) and bool(prior)
        and all(type(v.get('pid')) is int and v['pid']>0 and type(v.get('start_ticks')) is int and v['start_ticks']>0
          and isinstance(v.get('argv'),list) and v['argv'] and v.get('stopped_and_reaped_or_exited') is True for v in prior),
        'Concrete closed Goop PID/start/argv receipts required')
    require(any(v.get('role')=='trainer' for v in prior) and any(v.get('role')=='whole_host_monitor' for v in prior),
        'Prior trainer and broad-monitor identities required')
    require(clock.get('schema')=='adaptgns_sand_scoped_clock_check_v1' and clock.get('issued_by')=='root'
        and clock.get('host')==release['host'] and clock.get('root_host_samples_reviewed') is True
        and finite(clock.get('clock_error_bound_seconds')) and clock['clock_error_bound_seconds']<=5,'Reviewed bounded host clock required')
    for evidence in (process,clock):require(-5<=(now-stamp(evidence['checked_utc'])).total_seconds()<=300,'Fresh process/clock evidence required')
    require(release.get('clock_error_bound_seconds')==clock['clock_error_bound_seconds'] and release.get('clock_checked_utc')==clock['checked_utc']
        and release.get('process_checked_utc')==process['checked_utc'],'Root evidence timestamps differ')
    stop=stamp(release['training_stop_utc']);latest=stamp(release['latest_start_utc']);error=clock['clock_error_bound_seconds']
    cleanup=stop-timedelta(seconds=error+CLEANUP)
    require(stop==TRAINING_STOP and stamp(release['compute_analysis_deadline_utc'])==DEADLINE
        and release.get('training_allocation_seconds')==TRAINING_SECONDS and release.get('post_training_allocation_seconds')==POSTTRAINING_SECONDS
        and latest<=LATEST_START and latest+timedelta(seconds=TRAINING_SECONDS)<=cleanup
        and now+timedelta(seconds=error)<=latest,'Complete fixed allocation no longer fits before training cutoff')
    initial_rows=process.get('gpu_processes');require(isinstance(initial_rows,list),'Full root GPU inventory required')
    check_gpu(initial_rows,{},mods.B,uuids,release['owned_gpu_indices'],lambda pid:None)
    return SimpleNamespace(release=release,jobs=jobs,mods=mods,hashes=hashes,process=process,stop=stop,
        latest=latest,cleanup=cleanup,error=error)


def paths_and_hashes(args,mods):
    keys=('trainer','capacity_source','lifecycle_source','verifier_source','train_manifest','admission','structural_report',
        'protocol','schedule_plan','amendment','process_check','clock_check','python')
    paths={k:getattr(args,k) for k in keys};paths['supervisor']=Path(__file__).resolve()
    interpreter=python_environment(args.python);paths['python_target']=Path(interpreter['resolved_binary_path'])
    if interpreter['pyvenv_config_path'] is not None:paths['python_venv_config']=Path(interpreter['pyvenv_config_path'])
    hashes={k:sha(p) for k,p in paths.items()}
    require(hashes['protocol']==PROTOCOL_SHA and hashes['schedule_plan']==PLAN_SHA and hashes['amendment']==AMENDMENT_SHA
        and hashes['python']==sha(sys.executable) and all(hashes[k]==v for k,v in mods.C.DATA_PINS.items()),'Frozen protocol/data/plan/interpreter differs')
    numerical={str(args.repo/name):pin for name,pin in mods.T.SOURCE_PINS.items()}
    require(all(sha(p)==d for p,d in numerical.items()),'Frozen numerical closure differs')
    return paths,hashes,numerical


def check_gpu(rows,owned,B,uuids,indices,identity):
    scope={B.canonical_gpu_uuid(uuids[i]) for i in indices};all_ids={B.canonical_gpu_uuid(v) for v in uuids};details=[]
    for row in rows:
        uid=B.canonical_gpu_uuid(row['gpu_uuid']);require(type(row.get('pid')) is int and row['pid']>0 and uid in all_ids,'Malformed/uninventoried GPU PID or UUID')
        if row['pid'] in owned:
            require(uid==B.canonical_gpu_uuid(uuids[owned[row['pid']]['job']['gpu']]),'Owned GPU assignment changed')
        if uid not in scope:
            details.append({**row,'classification':'unowned_device_observed_without_control'});continue
        require(row['pid'] in owned,'Foreign work on an owned GPU')
        child=owned[row['pid']];require(uid==B.canonical_gpu_uuid(uuids[child['job']['gpu']]),'Owned GPU assignment changed')
        try:current=identity(row['pid'])
        except (FileNotFoundError,ProcessLookupError):
            require(child['process'].returncode is not None,'Unreaped owned GPU identity disappeared')
            details.append({**row,'classification':'stale_reaped_absent_owned_child'});continue
        require(child['process'].returncode is None,'Reused live historical PID is not owned')
        require(current['ppid']==os.getpid() and current['start_ticks']==child['identity']['start_ticks']
            and (current['argv']==child['command'] or current.get('state')=='Z' and current['argv']==[]),'Owned GPU process identity differs')
        details.append({**row,'classification':'owned_identity_verified','identity':current})
    return details


class Runtime:
    def __init__(self,mods):self.mods=mods;self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
    def now(self):return utc()
    def mono(self):return time.perf_counter()
    def sleep(self,s):time.sleep(s)
    def identity(self,pid):return self.mods.B.process_identity(pid)
    def gpu_identity(self,pid):
        try:return self.identity(pid)
        except FileNotFoundError:
            fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
            require(fields[0]=='Z','Unavailable process is not a zombie')
            return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),'state':'Z','argv':[]}
    def process_inventory(self):
        rows=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:value=self.identity(int(path.name))
            except (FileNotFoundError,ProcessLookupError):continue
            name=executed_python_script(value['argv'])
            if name in OLD_BROAD_SOURCES|TRAINING_SOURCES:rows.append(value)
        return rows
    def gpu_now(self):return self.mods.B.gpu_processes()
    def gpu_submit(self,owned):self.query_owned=dict(owned);self.future=self.pool.submit(self.gpu_now)
    def gpu_result(self):
        if self.future is None or not self.future.done():return None
        rows=self.future.result();self.future=None;return rows,self.query_owned
    def launch(self,args,job,command):
        out=args.output_dir/'logs'/(job['id']+'.stdout.jsonl');err=args.output_dir/'logs'/(job['id']+'.stderr.txt')
        handles=(out.open('xb'),err.open('xb'))
        try:p=subprocess.Popen(command,stdout=handles[0],stderr=handles[1],env={**os.environ,**ENVIRONMENT},start_new_session=True)
        except BaseException:
            for h in handles:h.close()
            raise
        return p,handles,str(out),str(err)
    def capture(self,args,child):return self.mods.B.capture_initial(args,child)
    def reap(self,args,child,record,reaped,launch,manifest):return self.mods.B.reap_child(args,child,record,reaped,launch,manifest)
    def stop(self,child,sig):return self.mods.B.stop_owned([child],sig)
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)


def prior_ownership_closed(context,rt,own_children=()):
    original=context.process['closed_goop_processes'];receipts=[]
    for prior in original:
        try:live=rt.identity(prior['pid'])
        except (FileNotFoundError,ProcessLookupError):live=None
        require(live is None or live['start_ticks']!=prior['start_ticks'],'Prior Goop trainer/monitor identity still exists')
        receipts.append({'closed_identity':prior,'current_pid_identity':live})
    own={c['process'].pid:c for c in own_children}
    inventory=rt.process_inventory()
    for row in inventory:
        name=executed_python_script(row['argv'])
        require(name not in OLD_BROAD_SOURCES,'Old whole-host monitor remains active; no live handoff allowed')
        if name in TRAINING_SOURCES:
            child=own.get(row['pid'])
            require(child is not None and child['process'].returncode is None and row['argv']==child['command']
                and row['ppid']==os.getpid() and row['start_ticks']==child['identity']['start_ticks'],'Duplicate/unowned Sand trainer exists')
    return {'closed_process_checks':receipts,'relevant_process_inventory':inventory}


def verify_files(bindings):require(all(sha(p)==d for p,d in bindings.items()),'Pinned input bytes changed')
def verify_metadata(context):require(all(file_identity(p)==v for p,v in context.metadata.items()),'Pinned input metadata changed')


def cleanup(args,context,rt,children,record,reaped,launch,manifest):
    start=rt.mono()
    while any(c['process'].returncode is None for c in children) and rt.mono()-start<CLEANUP:
        elapsed=rt.mono()-start
        for child in children:
            if child['process'].returncode is not None:continue
            for threshold,sig in ((0,signal.SIGINT),(5,signal.SIGTERM),(10,signal.SIGKILL)):
                if elapsed>=threshold and not any(s['signal']==signal.Signals(sig).name for s in child['signals']):
                    try:rt.stop(child,sig)
                    except Exception as e:record.setdefault('cleanup_errors',[]).append({'id':child['job']['id'],'error':str(e)})
            for fn in (lambda:rt.capture(args,child),lambda:rt.reap(args,child,record,reaped,launch,manifest)):
                try:fn()
                except Exception as e:record.setdefault('cleanup_errors',[]).append({'id':child['job']['id'],'error':str(e)})
        rt.sleep(.1)
    record['unreaped_owned_children']=[{'id':c['job']['id'],'pid':c['process'].pid,'identity':c['identity'],
        'command':c['command'],'signals':c['signals']} for c in children if c['process'].returncode is None]


def run_host(args,context,rt,launch,manifest):
    B=context.mods.B;children=[];reaped=[];abort=[None];started=rt.mono();path=args.output_dir/('wave_'+args.host_role+'.json')
    record={'schema':SCHEMA,'wave':args.host_role,'state':'checking_prior_ownership','jobs':[],'observations':[],
        'started_utc':rt.now().isoformat(),'owned_gpu_indices':context.release['owned_gpu_indices'],
        'child_outcome_verification_scope':'clean100k status only; frozen full endpoint checks after all owned children reaped'}
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(sig,frame):abort[0]=abort[0] or 'Supervisor received '+signal.Signals(sig).name
    for s in previous:signal.signal(s,interrupted)
    try:
        verify_files(context.bindings)
        record['prior_ownership']=prior_ownership_closed(context,rt)
        rows=rt.gpu_now();record['initial_gpu_inventory']=rows
        record['initial_gpu_classification']=check_gpu(rows,{},B,launch['gpu_uuids'],context.release['owned_gpu_indices'],rt.gpu_identity)
        write(path,record)
        for job in context.jobs:
            verify_metadata(context);prior_ownership_closed(context,rt,children)
            command=context.mods.S.fixed_command(args,job)
            require(not abort[0] and rt.now()+timedelta(seconds=context.error)<=context.latest
                and rt.now()+timedelta(seconds=TRAINING_SECONDS)<=context.cleanup,'Launch window closed after setup/process checks')
            start_utc=rt.now().isoformat();begin=rt.mono();proc,handles,out,err=rt.launch(args,job,command)
            child={'job':job,'process':proc,'identity':None,'command':command,'handles':handles,'signals':[],
                'started':begin,'started_utc':start_utc,'stdout_file':out,'stderr_file':err,'initial_pointer':None}
            children.append(child)
            identity=rt.identity(proc.pid);child['identity']=identity
            require(identity['argv']==command and identity['ppid']==os.getpid() and type(identity['start_ticks']) is int and identity['start_ticks']>0,
                'Launched child identity differs')
            write(args.output_dir/'logs'/(job['id']+'.launch.json'),{'job':job,'pid':proc.pid,'identity':identity,'command':command,'started_utc':start_utc})
        record['state']='training';record['launch_skew_seconds']=max(c['started'] for c in children)-min(c['started'] for c in children)
        last_query=-math.inf;last_status=-math.inf
        while any(c['process'].returncode is None for c in children):
            require(not abort[0],abort[0] or 'Interrupted')
            require(rt.now()<context.cleanup,'Training cleanup cutoff reached; evaluation/analysis reserve retained')
            for child in children:
                if child['process'].returncode is None:
                    rt.capture(args,child);failure=rt.reap(args,child,record,reaped,launch,manifest)
                    require(failure is None,failure or 'Scientific child failed')
            observation=rt.gpu_result()
            if observation is not None:
                rows,known=observation;entry={'checked_utc':rt.now().isoformat(),'gpu_processes':rows,'classification':'pending'}
                record['observations'].append(entry);write(path,record)
                entry['classification']=check_gpu(rows,known,B,launch['gpu_uuids'],context.release['owned_gpu_indices'],rt.gpu_identity)
                write(path,record)
            now=rt.mono()
            if rt.future is None and now-last_query>=30:
                rt.gpu_submit({c['process'].pid:c for c in children if c['process'].returncode is None});last_query=now
            if now-last_status>=30:
                verify_metadata(context);states={}
                for c in children:
                    p=args.output_dir/'jobs'/c['job']['id']/'status.json'
                    if p.exists():
                        value=read(p);require(isinstance(value,dict),'Malformed child status')
                        states[c['job']['id']]={k:value.get(k) for k in ('state','completed_steps','committed_steps')}
                        require(value.get('state') not in ('failed','interrupted'),'Scientific child reported failure')
                record['latest_trainer_states']=states;record['latest_observed_utc']=rt.now().isoformat();write(path,record);last_status=now
            rt.sleep(.2)
        require(len(reaped)==len(context.jobs),'Every assigned scientific child must be reaped')
    except BaseException as error:abort[0]=abort[0] or type(error).__name__+': '+str(error)
    finally:
        for s in previous:signal.signal(s,signal.SIG_IGN)
        try:cleanup(args,context,rt,children,record,reaped,launch,manifest)
        except BaseException as error:
            abort[0]=abort[0] or 'Cleanup: '+str(error)
            record['unreaped_owned_children']=[{'id':c['job']['id'],'pid':c['process'].pid,'identity':c['identity']} for c in children if c['process'].returncode is None]
        for c in children:
            for h in c['handles']:h.close()
        for s,h in previous.items():signal.signal(s,h)
        try:rt.close()
        except BaseException as error:abort[0]=abort[0] or 'Runtime close: '+str(error)
        try:
            observation=rt.gpu_result()
            if observation is not None:
                rows,known=observation;entry={'checked_utc':rt.now().isoformat(),'gpu_processes':rows,'classification':'pending_after_cleanup'}
                record['observations'].append(entry)
                entry['classification']=check_gpu(rows,known,B,launch['gpu_uuids'],context.release['owned_gpu_indices'],rt.gpu_identity)
        except BaseException as error:abort[0]=abort[0] or 'Final asynchronous GPU inventory: '+str(error)
    record.update(state='stopped_pending_endpoint_audit' if not abort[0] else 'failed',error=abort[0],
        ended_utc=rt.now().isoformat(),elapsed_before_endpoint_audit_seconds=rt.mono()-started,
        all_children=[{'job':c['job'],'pid':c['process'].pid,'identity':c['identity'],'command':c['command'],'exit_code':c['process'].returncode,
            'signals':c['signals']} for c in children],never_started=[j for j in context.jobs if j['id'] not in {c['job']['id'] for c in children}])
    write(path,record)
    try:
        rows=rt.gpu_now();record['final_gpu_inventory']=rows
        record['final_gpu_classification']=check_gpu(rows,{},B,launch['gpu_uuids'],context.release['owned_gpu_indices'],rt.gpu_identity)
        require(not record['unreaped_owned_children'] and abort[0] is None,'Failed/incomplete scoped training retained')
        require(rt.now()+timedelta(seconds=context.error)<context.stop,'Endpoint audit cannot consume posttraining reserve')
        verify_files(context.bindings)
        verified=[]
        for item in reaped:
            require(rt.now()+timedelta(seconds=context.error)<context.stop,'Endpoint audit exceeded training allocation')
            job=next(j for j in context.jobs if j['id']==item['id'])
            verified.append(context.mods.S.verify_job(item['directory'],job,item['external'],manifest,PROTOCOL_SHA,launch['gpu_uuids'][job['gpu']]))
        pairing=context.mods.S.verify_pairing(verified)
        verify_files(context.bindings)
        require(rt.now()+timedelta(seconds=context.error)<=context.stop,'Endpoint verification finished after training stop')
        record.update(state='verified',pairing=pairing,endpoint_audit_finished_utc=rt.now().isoformat());write(path,record)
        return verified,pairing
    except BaseException as error:
        record.update(state='failed',error=record['error'] or type(error).__name__+': '+str(error));write(path,record);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--execute',action='store_true');parser.add_argument('--host-role',choices=['A','B'])
    for name in FIELDS:parser.add_argument('--'+name.replace('_','-'),type=Path)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','schedule':SCHEDULE,'updates':100000,
            'training_stop_utc':TRAINING_STOP.isoformat(),'compute_analysis_deadline_utc':DEADLINE.isoformat(),
            'live_handoff_retry_resume_test_access':False},indent=2));return 0
    require(args.host_role is not None and all(getattr(args,n) is not None for n in FIELDS),'Every explicit input and host role required')
    require(sys.platform.startswith('linux') and hasattr(os,'wait4') and os.environ.get('CUDA_VISIBLE_DEVICES') is None,'Unremapped Linux execution required')
    for n in FIELDS:
        if n!='python':setattr(args,n,getattr(args,n).resolve())
    python_environment(args.python)
    require(not args.output_dir.exists(),'Fresh output required; no duplicate, resume or retry')
    mods=configure(args);paths,hashes,numerical=paths_and_hashes(args,mods)
    release_raw=args.release.read_bytes();release=json.loads(release_raw);release_sha=hashlib.sha256(release_raw).hexdigest()
    context=validate_release(args,release,mods,hashes,utc())
    protected=(args.train_manifest.parent,args.repo/'adaptive-gns')
    require(all(args.output_dir!=p and p not in args.output_dir.parents and args.output_dir not in p.parents for p in protected),'Output must be separate from protected source/data')
    args.output_dir.mkdir(mode=0o700)
    for n in ('inputs','jobs','logs'):(args.output_dir/n).mkdir()
    bindings={str(p):hashes[k] for k,p in paths.items()};bindings.update(numerical);bindings[str(args.release)]=release_sha
    for k,p in paths.items():
        if k in ('python','python_target'):continue
        target=args.output_dir/'inputs'/(k+p.suffix);target.write_bytes(p.read_bytes());bindings[str(target)]=hashes[k]
    target=args.output_dir/'inputs/release.json';target.write_bytes(release_raw);bindings[str(target)]=release_sha
    context.bindings=bindings;context.metadata={p:file_identity(p) for p in bindings}
    launch={'schema':SCHEMA,'host_role':args.host_role,'cohort_id':release['cohort_id'],'hostname':socket.gethostname(),
        'files_sha256':hashes,'release_sha256':bindings[str(args.release)],'gpu_uuids':release['gpu_uuids'],'schedule':SCHEDULE,
        'environment':ENVIRONMENT,'started_utc':utc().isoformat(),'pid':os.getpid(),'training_stop_utc':context.stop.isoformat(),
        'cleanup_trigger_utc':context.cleanup.isoformat(),'latest_start_utc':context.latest.isoformat(),'clock_error_bound_seconds':context.error,
        'commands':{j['id']:mods.S.fixed_command(args,j) for j in context.jobs}}
    write(args.output_dir/'launch.json',launch)
    try:
        validate_release(args,release,mods,hashes,utc())
        jobs,pairing=run_host(args,context,Runtime(mods),launch,read(args.train_manifest))
        verify_files(bindings)
        summary={'schema':SCHEMA,'status':'verified_host_scientific_endpoints','host_role':args.host_role,'cohort_id':release['cohort_id'],
            'jobs':jobs,'pairing':pairing,'all_inputs_reverified':True,'whole_six_model_cohort_admitted':False,
            'scientific_protocol_sha256':PROTOCOL_SHA,'schedule_plan_sha256':PLAN_SHA,'amendment_sha256':hashes['amendment'],
            'endpoint_verifier_sha256':VERIFIER_SHA,'requires':'Separate root complete-six-model cohort/source/test admission'}
        write(args.output_dir/'worker_summary.json',summary)
        write(args.output_dir/'status.json',{'state':'complete_host_scoped_training','summary_sha256':sha(args.output_dir/'worker_summary.json'),
            'whole_six_model_cohort_admitted':False,'ended_utc':utc().isoformat()});return 0
    except BaseException as error:
        write(args.output_dir/'status.json',{'state':'failed_host_scoped_training','error_type':type(error).__name__,'error':str(error),
            'all_outcomes_retained':True,'automatic_retry_or_recovery':False,'whole_six_model_cohort_admitted':False,'ended_utc':utc().isoformat()});return 1


if __name__=='__main__':raise SystemExit(main())
