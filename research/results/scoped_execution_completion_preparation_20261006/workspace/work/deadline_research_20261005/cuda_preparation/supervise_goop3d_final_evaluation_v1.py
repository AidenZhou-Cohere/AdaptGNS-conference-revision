#!/usr/bin/env python3
"""Root-released D3 final evaluation allocation; description only by default.

One dedicated four-GPU host, two fixed model waves, all thirty stage receipts.
No training, acquisition, admission, endpoint choice, retries or result means.
Only explicit CLI execution launches reviewed children; tests inject fake runtime.
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
SCHEMA='adaptgns_goop3d_final_evaluation_supervisor_v1'
RELEASE_SCHEMA='adaptgns_goop3d_final_evaluation_supervisor_release_v1'
LEDGER_SCHEMA='adaptgns_goop3d_final_evaluation_ledger_v1'
COLLECTOR_SHA='d7096bef018ac7e3d28cc429a4711d88209e48a1b812dda0aad995d6a41de5fd'
INTERFACE_SHA='0b27bb725a7616f057047ad1fc104cfdc0144c6c12cbcfe4fbe79b08bbc1f078'
EVALUATOR_SHA='9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de'
TRAINER_SHA='8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc'
LIFECYCLE_SHA='c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd'
PREPARATION_SHA='cdd17f625e0129926f69e81cfe5961c05e31aed7ca875d2550f40e642434523a'
FREEZE_SHA='add53ea8c41ef44a1b3248eaacd4df38b49d396b8ecd374d9f928ba751bc631a'
PLAN_SHA='f38131f2851afdcbf774af506ebe8e02c8eca1be5e143430f68abedb95304c6d'
PROTOCOL_SHA='5010f9023a3eee45f85bee80b35fc8e506ca145667daf75027b32c92658faa70'
VALID_SHA='f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef'
COST_BASIS='fixed_operational_quota_after_failed_full_horizon_timing'
DEADLINE=datetime(2026,10,7,1,tzinfo=timezone.utc)
CLEANUP=15
ENVIRONMENT={'CUBLAS_WORKSPACE_CONFIG':':4096:8',
 'LD_LIBRARY_PATH':'/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
WAVES={'A':(('base',0,0),('mix',0,1),('base',1,2),('mix',1,3)), 'B':(('base',2,0),('mix',2,1))}
STAGES={'full_rollout_valid':('full-rollout','valid'),'full_rollout_test':('full-rollout','test'),
 'same_state_valid':('same-state','valid'),'same_state_test':('same-state','test'),'clean_validation':('clean-validation','valid')}
CONTROL_NAMES=('plan','cohort','cohort_audit','protocol','trainer_source','valid_manifest','test_manifest','test_preflight')
SOURCE_PINS={'summarize_goop3d_graph_support_v1.py':COLLECTOR_SHA,'goop3d_scalar_final_ledger_interface_v1.md':INTERFACE_SHA,
 'evaluate_goop3d_graph_support_v1.py':EVALUATOR_SHA,'train_goop3d_graph_support_cuda_v2.py':TRAINER_SHA,
 'measure_sand_cuda_capacity_v2.py':LIFECYCLE_SHA,'prepare_goop3d_reserved_test_v1.py':PREPARATION_SHA,
 'prepare_goop3d_final_cohort_v1.py':FREEZE_SHA}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def read(path):return json.loads(Path(path).read_bytes())
def stamp(value):
    t=datetime.fromisoformat(value);require(t.tzinfo is not None,'Timezone-aware time required');return t.astimezone(timezone.utc)
def utc():return datetime.now(timezone.utc)
def finite(value,positive=False):return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)
def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def python_environment(path):
    p=Path(path)
    require(p.is_absolute() and str(p)==os.path.abspath(p) and str(p)==sys.executable,
        'Preserve the current interpreter lexical absolute path, including its venv symlink')
    cfg=p.parent.parent/'pyvenv.cfg'
    return {'lexical_path':str(p),'resolved_binary_path':str(p.resolve()),'binary_sha256':sha(p),
        'sys_prefix':sys.prefix,'sys_base_prefix':sys.base_prefix,
        'pyvenv_config_path':str(cfg) if cfg.exists() else None,'pyvenv_config_sha256':sha(cfg) if cfg.exists() else None}


def python_files(value):
    files={value['lexical_path']:value['binary_sha256'],value['resolved_binary_path']:value['binary_sha256']}
    if value['pyvenv_config_path'] is not None:files[value['pyvenv_config_path']]=value['pyvenv_config_sha256']
    return files


def atomic_bytes(path,raw):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path);return hashlib.sha256(raw).hexdigest()


def write(path,value):return atomic_bytes(path,encode(value))


def snapshot(path,expected=None):
    raw=Path(path).read_bytes();d=hashlib.sha256(raw).hexdigest();require(expected is None or d==expected,'JSON byte binding differs: '+str(path));return json.loads(raw),d


def load(name,pin,label):
    path=HERE/name;require(sha(path)==pin,'Reviewed source differs: '+name)
    spec=importlib.util.spec_from_file_location(label,path);module=importlib.util.module_from_spec(spec);sys.modules[label]=module;spec.loader.exec_module(module);return module


def modules():
    return SimpleNamespace(C=load('summarize_goop3d_graph_support_v1.py',COLLECTOR_SHA,'_d3_final_collector'),
        E=load('evaluate_goop3d_graph_support_v1.py',EVALUATOR_SHA,'_d3_final_evaluator'),
        T=load('train_goop3d_graph_support_cuda_v2.py',TRAINER_SHA,'_d3_final_trainer_contract'),
        B=load('measure_sand_cuda_capacity_v2.py',LIFECYCLE_SHA,'_d3_final_lifecycle'))


def merge(*maps):
    result={}
    for values in maps:
        for p,d in values.items():
            require(Path(p).is_absolute() and (str(Path(p).resolve())==p or p==sys.executable) and digest(d) and (p not in result or result[p]==d),'Conflicting/nonabsolute input binding')
            result[p]=d
    return result


def verify_files(files):require(all(sha(p)==d for p,d in files.items()),'Bound source/input bytes changed')


def file_identity(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def verify_control_metadata(context):
    require(all(file_identity(p)==identity for p,identity in context.control_metadata.items()), 'Control-file metadata changed during active scheduling')


def validate_allocation(release,plan,now):
    require(release.get('cost_basis')==COST_BASIS and plan.get('cost_basis')==COST_BASIS
        and plan.get('status')=='prospectively_selected' and plan.get('issued_by')=='root'
        and plan.get('selection_used_validation_or_test_accuracy') is False
        and plan.get('complete_full_horizon_timing_attempt_failed') is True
        and plan.get('evaluation_activity_quota_seconds')==21600 and plan.get('transfer_review_analysis_reserve_seconds')==3600
        and release.get('evaluation_activity_quota_seconds')==21600 and release.get('analysis_reserve_seconds')==3600
        and release.get('all_required_outcomes_promised') is False and release.get('full_horizon_runtime_forecast_claim') is False,
        'Frozen failed-timing operational allocation and analysis reserve required')
    order=release.get('stage_order');quotas=release.get('stage_quotas_seconds',{})
    require(isinstance(order,list) and len(order)==5 and set(order)==set(STAGES) and set(quotas)==set(STAGES)
        and all(type(q) is int and 1<=q<=7200 for q in quotas.values()) and release.get('cleanup_seconds_per_invocation')==CLEANUP,
        'One prospective complete stage order and uniform whole-invocation quotas required')
    audit=release.get('supervisor_audit_reserve_seconds');error=release.get('clock_error_bound_seconds')
    require(type(audit) is int and audit>=60 and finite(error) and error<=5,'Explicit audit reserve and bounded clock error required')
    duration=2*(sum(quotas.values())+5*CLEANUP)+audit
    require(duration<=21600 and release.get('reserved_queue_seconds')==duration,'Two waves plus cleanup/audit must fit six-hour activity allocation')
    stop=stamp(release['absolute_stop_utc']);latest=stamp(release['latest_start_utc'])
    require(stop+timedelta(seconds=3600)<=DEADLINE and -5<=(now-stamp(release['clock_checked_utc'])).total_seconds()<=300
        and now+timedelta(seconds=error)<=latest<=stop-timedelta(seconds=duration+error),
        'Fresh clock/latest-start and complete evaluation/analysis windows required')
    return duration


def child_namespace(command,C):
    options=C.flags(command);required=C.EVALUATOR_OPTIONS-{'--cross-split-audit'}
    require(set(options)==required|({'--cross-split-audit'} if options.get('--split')=='test' else set()),'Every exact child option required')
    result={k[2:].replace('-','_'):v for k,v in options.items()}
    for k in ('seed','checkpoint_updates','cuda_index','threads','max_seconds'):result[k]=int(result[k])
    path_names=('release','repo','manifest','split_admission','structural_report','acquisition_report','context_semantics',
      'auxiliary_report','trainer_source','protocol','checkpoint','cohort','cohort_audit','output_dir','cross_split_audit')
    for k in path_names:
        if k in result:
            require(Path(result[k]).is_absolute() and str(Path(result[k]).resolve())==result[k],'Original absolute child path required')
            result[k]=Path(result[k])
    return SimpleNamespace(**result),options


def validate_release(args,raw,mods,now=None):
    clock=utc if now is None else lambda:now
    now=clock();release=json.loads(raw);root_sha=hashlib.sha256(raw).hexdigest();own_sha=sha(__file__)
    require(release.get('schema')==RELEASE_SCHEMA and release.get('status')=='approved_for_fixed_whole_invocation_quotas'
        and release.get('issued_by')=='root' and release.get('dataset')=='Goop-3D'
        and release.get('source_sha256')==own_sha and release.get('collector_sha256')==COLLECTOR_SHA
        and release.get('ledger_interface_sha256')==INTERFACE_SHA and release.get('hostname')==socket.gethostname()
        and release.get('environment')==ENVIRONMENT and release.get('output_dir')==str(args.output_dir.resolve())
        and release.get('whole_supervisor_outer_timeout_required') is True and release.get('all_six_models_and_both_splits_independently_admitted') is True
        and release.get('automatic_retry_or_resume') is False,'Exact root-reviewed D3 final supervisor release required')
    pins=release.get('files_sha256',{});require(isinstance(pins,dict),'Complete root input bindings required')
    interpreter=python_environment(sys.executable)
    require(release.get('python_environment')==interpreter,'Root-bound lexical interpreter/venv provenance differs')
    # Only non-test authorization/source files are read until complete cohort gate.
    required_initial={str(Path(__file__).resolve()):own_sha,str(args.plan):PLAN_SHA,str(args.protocol):PROTOCOL_SHA,str(args.trainer_source):TRAINER_SHA}
    required_initial.update({str(HERE/name):d for name,d in SOURCE_PINS.items()})
    require(all(pins.get(p)==d for p,d in required_initial.items()),'Pinned plan/protocol/helper closure required')
    plan,plan_sha=snapshot(args.plan,PLAN_SHA);validate_allocation(release,plan,now)
    cohort,ch=snapshot(args.cohort,pins.get(str(args.cohort)));audit,ah=snapshot(args.cohort_audit,pins.get(str(args.cohort_audit)))
    require(pins.get(str(args.cohort))==ch and pins.get(str(args.cohort_audit))==ah
        and release.get('cohort_sha256')==ch and release.get('cohort_audit_sha256')==ah
        and type(release.get('endpoint_updates')) is int and release['endpoint_updates']==plan['endpoint_updates']
        and all(type(x.get('seed')) is int for x in cohort.get('models',[])+audit.get('models',[])+audit.get('paired_seeds',[]))
        and cohort.get('evaluation_admitted') is True and audit.get('evaluation_admitted') is True
        and cohort.get('freeze_adapter_sha256')==FREEZE_SHA and cohort.get('cohort_id')==audit.get('cohort_id'), 'Reviewed frozen six-model D3 identity required')
    require(sha(args.protocol)==PROTOCOL_SHA and sha(args.trainer_source)==TRAINER_SHA,'Frozen protocol/trainer differs')
    for model in cohort.get('models',[]):
        mods.E.cohort_gate(cohort,audit,{'scientific_endpoint_updates':release['endpoint_updates']},SimpleNamespace(protocol=args.protocol,
            trainer_source=args.trainer_source,cohort_audit=args.cohort_audit,arm=model['arm'],seed=model['seed'],checkpoint_sha256=model['checkpoint_sha256']))
    require(len(cohort.get('models',[]))==6 and all(p.get('initial_empty_adam_identity') is True for p in audit['paired_seeds'])
        and stamp(cohort['created_utc'])<=stamp(release['issued_utc'])<=now,'Complete post-freeze root evaluation authorization required')
    # First access to released test manifest/preflight occurs after the cohort gate.
    manifests={};control=dict(required_initial)
    for split,path in (('valid',args.valid_manifest),('test',args.test_manifest)):
        value,d=snapshot(path,pins.get(str(path)));require(pins.get(str(path))==d,'Root manifest bytes required')
        require(value.get('dataset')=='Goop-3D' and value.get('split')==split and type(value.get('record_count')) is int
            and value['record_count']==len(value.get('records',[]))>0 and value.get('metadata_sha256')==mods.E.METADATA_SHA
            and all(type(r.get('source_index')) is int and r['source_index']==i for i,r in enumerate(value['records'])),'Complete ordered D3 source manifest required')
        mods.E.schedules(value['records'],'full-rollout','final_evaluation');manifests[split]=value;control[str(path)]=d
    require(control[str(args.valid_manifest)]==VALID_SHA and manifests['valid']['metadata']==manifests['test']['metadata'],'Original validation/common physical metadata required')
    preflight,ph=snapshot(args.test_preflight,pins.get(str(args.test_preflight)))
    require(pins.get(str(args.test_preflight))==ph and preflight.get('schema')=='adaptgns_goop3d_reserved_test_preflight_v1'
        and preflight.get('status')=='frozen_split_and_evidence_contract_passed' and preflight.get('source_sha256')==PREPARATION_SHA
        and preflight.get('cohort_sha256')==ch and preflight.get('manifest_sha256')==control[str(args.test_manifest)]
        and preflight.get('test_evaluation_executed') is False and preflight.get('evaluator_sha256')==EVALUATOR_SHA
        and preflight.get('source_order_grid')==mods.E.schedules(manifests['test']['records'],'full-rollout','final_evaluation'),
        'Reviewed reserved-test checksum/conversion/admission preflight required')
    control.update({str(args.plan):plan_sha,str(args.cohort):ch,str(args.cohort_audit):ah,str(args.test_preflight):ph,
        str(args.release):root_sha,str(Path(__file__).resolve()):own_sha})
    uuids=release.get('gpu_uuids');require(isinstance(uuids,list) and len(uuids)==4
        and len({mods.B.canonical_gpu_uuid(u) for u in uuids})==4,'Four unique physical GPU identities required')
    streams=release.get('streams',[]);expected=[(w,a,s,g) for w,rows in WAVES.items() for a,s,g in rows]
    require(len(streams)==6 and [(s.get('wave'),s.get('arm'),s.get('seed'),s.get('gpu')) for s in streams]==expected
        and all(type(s.get('seed')) is int and type(s.get('gpu')) is int for s in streams),'Fixed two-wave model/GPU mapping required')
    entries=[];all_bindings=merge(control,preflight.get('verified_evidence_sha256',{}),python_files(interpreter));child_hashes={}
    for stream in streams:
        arm,seed,gpu=stream['arm'],stream['seed'],stream['gpu'];sid=f'{arm}_seed{seed}'
        require(stream.get('id')==sid and len(stream.get('commands',[]))==5,'All five stages for every fixed stream required')
        model=next(m for m in cohort['models'] if (m['arm'],m['seed'])==(arm,seed))
        for name,command in zip(release['stage_order'],stream['commands']):
            mode,split=STAGES[name];child,options=child_namespace(command,mods.C);quota=release['stage_quotas_seconds'][name]
            evaluator_index=next(i for i,v in enumerate(command) if Path(v).name=='evaluate_goop3d_graph_support_v1.py')
            require(command[1:evaluator_index] in ([],['-B'],['-u'],['-B','-u'],['-u','-B'])
                and command[0]==interpreter['lexical_path'] and pins.get(command[0])==interpreter['binary_sha256']
                and Path(command[evaluator_index])==HERE/'evaluate_goop3d_graph_support_v1.py','Exact interpreter/direct evaluator invocation required')
            require(options.get('--execute') is True and child.purpose=='final_evaluation' and child.mode==mode and child.split==split
                and child.arm==arm and options['--seed']==str(seed) and child.threads==2 and options['--threads']=='2'
                and options['--cuda-index']==str(gpu) and child.gpu_uuid==uuids[gpu] and child.max_seconds==quota
                and options['--max-seconds']==str(quota) and options['--checkpoint-updates']==str(release['endpoint_updates'])
                and child.checkpoint_sha256==model['checkpoint_sha256'] and str(child.checkpoint)==model['checkpoint_path']
                and child.output_dir==args.output_dir/'jobs'/sid/name,'Exact frozen model/stage/quota/output identity required')
            require(child.cohort==args.cohort and child.cohort_audit==args.cohort_audit and child.protocol==args.protocol
                and child.trainer_source==args.trainer_source and child.manifest==(args.valid_manifest if split=='valid' else args.test_manifest),
                'One original cohort/protocol and corresponding complete manifest required')
            cr,crh=snapshot(child.release,pins.get(str(child.release)));require(pins.get(str(child.release))==crh,'Root child release bytes required')
            require(type(cr.get('seed')) is int and cr['seed']==seed,'Exact child seed identity required')
            checked=mods.E.release_gate(child);require(checked==cr,'Child release changed during scalar admission')
            cb=mods.E.execution_bindings(child,cr,mods.T,crh)
            require(cr.get('scientific_endpoint_updates')==release['endpoint_updates'],'Child scientific endpoint differs')
            child_hashes[str(child.release)]=crh
            all_bindings=merge(all_bindings,cb)
            entries.append({'arm':arm,'seed':seed,'wave':stream['wave'],'gpu':gpu,'gpu_uuid':uuids[gpu], 'stream':sid,
                'stage':name,'mode':mode,'split':split,'endpoint_updates':release['endpoint_updates'],
                'checkpoint_sha256':model['checkpoint_sha256'],'directory':str(child.output_dir),
                'command':command,'release_file':str(child.release),'release_sha256':crh,'outer_timeout_seconds':quota})
    all_bindings=merge(all_bindings,child_hashes)
    # Root release cannot silently contain an unrelated payload/path; its own hash is captured separately.
    require(pins=={p:d for p,d in all_bindings.items() if p!=str(args.release)},'Exact complete supervisor/child/evidence file closure required')
    protected={str(Path(p).parent) for p in (args.valid_manifest,args.test_manifest)}|{str(Path(m['checkpoint_path']).parent) for m in cohort['models']}
    protected|={str(child_namespace(e['command'],mods.C)[0].repo/'adaptive-gns') for e in entries}
    require(all(args.output_dir!=Path(p) and Path(p) not in args.output_dir.parents and args.output_dir not in Path(p).parents for p in protected), 'Queue output must be separate from data/models/numerical sources')
    verify_files(all_bindings);validate_allocation(release,plan,clock())
    return SimpleNamespace(release=release,root_sha=root_sha,plan=plan,cohort=cohort,manifests=manifests,
        entries=entries,bindings=all_bindings,control_bindings={p:d for p,d in all_bindings.items() if Path(p).suffix!='.npy'},
        control_metadata={p:file_identity(p) for p in all_bindings if Path(p).suffix!='.npy'},mods=mods)


def never_started(entry,reason):
    return {'started':False,'state':'never_started','pid':None,'termination_reason':reason,
        'command':entry['command'],'outer_timeout_seconds':entry['outer_timeout_seconds']}


def status_current(entry):
    root=Path(entry['directory'])
    for name in ('failed_attempt.json','status.json'):
        path=root/name
        if path.exists():
            try:
                value=read(path)
                if value.get('current') is not None:return value['current']
            except Exception:pass  # Raw file remains; strict stopped accounting will inspect it.
    return None


def account_stage(entry,manifest,outcome,C):
    root=Path(entry['directory']);tree=C.tree_snapshot(root);rows={};wanted=C.cell_schedule(manifest,entry['mode'])
    expected={C.filename(item,entry['mode']):item for item in wanted}
    def saved(path):return snapshot(path,tree['files'][str(path)]['sha256'])[0]
    protocol_path=root/'protocol.json';protocol_sha=tree['files'].get(str(protocol_path),{}).get('sha256')
    if not outcome['started']:
        require(not tree['exists'],'A never-started stage must not contain an output tree')
    for path_s,detail in tree['files'].items():
        path=Path(path_s)
        if path.parent!=root or not path.match('trajectory_*.json'):continue
        require(path.name in expected and path.name not in rows,'Unexpected/duplicate committed row')
        row=saved(path);item=expected[path.name]
        require(protocol_sha is not None and row.get('protocol_sha256')==protocol_sha and outcome['started']
            and row.get('status') in ('complete','failed') and type(row.get('training_seed')) is int
            and row['training_seed']==entry['seed'] and row.get('arm')==entry['arm'] and row.get('objective')=='faithful'
            and all(row.get(k)==v for k,v in item.items()),'Committed row identity/status/protocol differs')
        artifact=root/path.with_suffix('.npz').name
        require(row.get('artifact_file')==artifact.name and tree['files'].get(str(artifact),{}).get('sha256')==row.get('artifact_sha256'),
            'Committed row requires matching retained NPZ bytes')
        require(row.get('failure') is None if row['status']=='complete' else isinstance(row.get('failure'),dict),'Explicit scientific failure object required')
        if entry['mode']=='full-rollout':require(row.get('horizon')==295 and (row.get('completed_steps')==295 if row['status']=='complete' else type(row.get('completed_steps')) is int and 0<=row['completed_steps']<295),'Declared full-H completeness differs')
        rows[path.name]={'state':'completed_required_outcome' if row['status']=='complete' else 'recorded_failed_outcome',
            'row_file':path.name,'row_sha256':detail['sha256'],'artifact_sha256':row['artifact_sha256'],'failure':row.get('failure')}
    current=outcome.get('current_before_stop') or status_current(entry)
    current_key=None
    if current is not None:
        try:current_key=C.cell_key(current,entry['mode'])
        except (KeyError,TypeError):pass
    timed=outcome.get('quota_stop_initiated') or outcome.get('inner_timeout_reported')
    cells=[]
    for item in wanted:
        filename=C.filename(item,entry['mode']);cell=rows.get(filename)
        if cell is None:
            state='never_started' if not outcome['started'] else 'timed_out_current' if timed and C.cell_key(item,entry['mode'])==current_key else 'not_completed_before_invocation_end'
            cell={'state':state,'reason':outcome['termination_reason']}
        cells.append({**item,**cell})
    require(C.tree_snapshot(root)==tree,'Stopped stage changed during coverage accounting')
    return {**entry,'outcome':outcome,'cells':cells,'coverage_scope':'Complete declared identities/status/opaque artifact bytes; full scalar validation requires separate root collector release'},tree


class Runtime:
    def __init__(self,helper):self.helper=helper;self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
    def mono(self):return time.perf_counter()
    def now(self):return utc()
    def sleep(self,seconds):time.sleep(seconds)
    def gpu_now(self):return self.helper.gpu_processes()
    def gpu_submit(self,owned):
        self.query_owned=dict(owned);self.future=self.pool.submit(self.helper.gpu_processes)
    def gpu_result(self):
        if self.future is None or not self.future.done():return None
        result=self.future.result();self.future=None;return result,self.query_owned
    def launch(self,entry,logroot):
        base=logroot/(entry['stream']+'_'+entry['stage']);handles=(base.with_suffix('.stdout.txt').open('xb'),base.with_suffix('.stderr.txt').open('xb'))
        try:process=subprocess.Popen(entry['command'],stdout=handles[0],stderr=handles[1],env={**os.environ,**ENVIRONMENT},start_new_session=True)
        except BaseException:
            for h in handles:h.close()
            raise
        return process,handles
    def identity(self,pid):return self.helper.process_identity(pid)
    def gpu_identity(self,pid):
        try:return self.helper.process_identity(pid)
        except FileNotFoundError:
            # An owned zombie can lose /proc/exe before wait4 reaps it.
            fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
            require(fields[0]=='Z','Process identity disappeared without a zombie receipt')
            return {'pid':pid,'state':'Z','ppid':int(fields[1]),'start_ticks':int(fields[19]),'argv':[]}
    def reap(self,child):
        pid,status,usage=os.wait4(child['process'].pid,os.WNOHANG)
        if not pid:return None
        child['process'].returncode=os.waitstatus_to_exitcode(status)
        return {'exit_code':child['process'].returncode,'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime,'linux_peak_rss_kib':usage.ru_maxrss}
    def stop(self,child,sig):self.helper.stop_owned([child],sig)
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)


def observe_reaped(child,usage,rt,hostname):
    now=rt.mono();entry=child['entry'];identity=child.get('identity') or {}
    failure_path=Path(entry['directory'])/'failed_attempt.json';failure=None;failure_error=None
    if failure_path.exists():
        try:
            failure=read(failure_path);require(isinstance(failure,dict), 'Failed-attempt receipt must be an object')
        except Exception as error:failure=None;failure_error=type(error).__name__+': '+str(error)
    inner=bool(failure and failure.get('error_type')=='TimeoutError')
    child['quota_expired'] |= now>=child['deadline']
    reason=child.get('stop_reason') or ('inner_evaluator_timeout' if inner else 'completed_invocation' if usage['exit_code']==0 else 'unclassified_evaluator_failure')
    return {'started':True,'state':'stopped_and_reaped','stopped_and_reaped':True,'pid':child['process'].pid,
        'start_ticks':identity.get('start_ticks'),'hostname':hostname,'command':child['command'],**usage,
        'elapsed_seconds':now-child['started'],'outer_timeout_seconds':entry['outer_timeout_seconds'],
        'absolute_stop_utc':child['absolute_stop_utc'],'started_utc':child['started_utc'],'exit_observed_utc':rt.now().isoformat(),
        'signals':child['signals'],'termination_reason':reason,'quota_expired_at_observation':child['quota_expired'],
        'quota_stop_initiated':child['quota_stop_initiated'],'inner_timeout_reported':inner,
        'current_before_stop':child['current_before_stop'],'worker_failed_attempt':failure,'worker_failed_attempt_parse_error':failure_error,'initial_process_identity':identity,
        'gpu_index':entry['gpu'],'gpu_uuid':entry['gpu_uuid'],'observer_lag_included':True}


def fatal_outcome(outcome):
    failure=outcome.get('worker_failed_attempt')
    if outcome.get('worker_failed_attempt_parse_error'):return 'Malformed failed-attempt receipt requires root review'
    if failure and failure.get('error_type') not in ('TimeoutError','KeyboardInterrupt'):return 'Evaluator implementation/data/infrastructure failure requires root review'
    if outcome['exit_code']!=0 and not outcome.get('quota_stop_initiated') and not outcome.get('inner_timeout_reported'):
        return 'Unclassified nonzero evaluator exit requires root review'
    return None


def check_gpu_observation(rows,owned,rt,helper,current_owned=None):
    current_owned={} if current_owned is None else current_owned
    details=[]
    for row in rows:
        require(row['pid'] in owned or row['pid'] in current_owned,'GPU PID is outside the captured/current owned registry')
        child=current_owned.get(row['pid'],owned.get(row['pid']))
        launched_during_query=owned.get(row['pid']) is not child
        require(helper.canonical_gpu_uuid(row['gpu_uuid'])==helper.canonical_gpu_uuid(child['entry']['gpu_uuid']),'Unexpected physical GPU assignment')
        try:identity=rt.gpu_identity(row['pid'])
        except (FileNotFoundError,ProcessLookupError):
            require(child['process'].returncode is not None,'Unreaped GPU process identity became unavailable')
            details.append({'pid':row['pid'],'classification':'stale_sample_of_reaped_absent_owned_child'});continue
        require(child['process'].returncode is None,'Reused live PID cannot inherit former GPU ownership')
        require(identity.get('ppid')==os.getpid() and identity.get('start_ticks')==child['identity']['start_ticks']
            and (identity.get('argv')==child['command'] or identity.get('state')=='Z' and identity.get('argv')==[]),
            'Current GPU PID/start/command identity differs')
        details.append({'pid':row['pid'],'classification':'owned_launched_during_query_identity_verified' if launched_during_query else 'currently_owned_identity_verified','identity':identity})
    return details


def drive_wave(context,wave,rt,output,registry,outcomes,abort_box):
    r=context.release;order=r['stage_order'];streams=[s for s in r['streams'] if s['wave']==wave]
    states={s['id']:{'stream':s,'next':0,'active':None} for s in streams};last_gpu=-math.inf;last_status=-math.inf
    run_stop=stamp(r['absolute_stop_utc'])-timedelta(seconds=r['supervisor_audit_reserve_seconds'])
    while any(v['active'] is not None or (not abort_box[0] and v['next']<5) for v in states.values()):
        now=rt.mono()
        if rt.now()+timedelta(seconds=r['clock_error_bound_seconds']+CLEANUP)>=run_stop:abort_box[0]=abort_box[0] or 'Reserved parent evaluation cutoff reached'
        observation=rt.gpu_result()
        if observation is not None:
            inventory,owned=observation
            index=len(context.gpu_observations);record={'rows':inventory,'received_utc':rt.now().isoformat(),'owned_at_query_start':list(owned)}
            context.gpu_observations.append(record)
            write(output/'logs'/f'gpu_observation_{index:06d}.json',record)
            current_owned={c['process'].pid:c for c in registry if c['process'].returncode is None}
            record['owned_at_receipt']=list(current_owned)
            record['classification']=check_gpu_observation(inventory,owned,rt,context.mods.B,current_owned)
            write(output/'logs'/f'gpu_observation_{index:06d}.json',record)
        for state in states.values():
            child=state['active']
            if child is None and not abort_box[0] and state['next']<5:
                remaining=sum(r['stage_quotas_seconds'][n]+CLEANUP for n in order[state['next']:])
                if rt.now()+timedelta(seconds=remaining+r['clock_error_bound_seconds'])>run_stop:
                    abort_box[0]='Remaining complete stage allocations no longer fit';continue
                if state['next']==0 and rt.now()+timedelta(seconds=r['clock_error_bound_seconds'])>stamp(r['latest_start_utc']) and wave=='A':
                    abort_box[0]='Initial launch window closed';continue
                verify_control_metadata(context)
                entry=next(e for e in context.entries if e['stream']==state['stream']['id'] and e['stage']==order[state['next']])
                (output/'jobs'/entry['stream']).mkdir(exist_ok=True)
                # Last read-free clock/abort check follows metadata checks and directory setup.
                start_utc=rt.now()
                if abort_box[0] or start_utc+timedelta(seconds=remaining+r['clock_error_bound_seconds'])>run_stop:
                    abort_box[0]=abort_box[0] or 'Launch no longer fits after control checks';continue
                if state['next']==0 and wave=='A' and start_utc+timedelta(seconds=r['clock_error_bound_seconds'])>stamp(r['latest_start_utc']):
                    abort_box[0]='Initial launch window closed after control checks';continue
                begin=rt.mono();process,handles=rt.launch(entry,output/'logs')
                child={'process':process,'handles':handles,'identity':None,'command':entry['command'],'entry':entry,'started':begin,
                    'started_utc':start_utc.isoformat(),'deadline':begin+entry['outer_timeout_seconds'],
                    'absolute_stop_utc':(start_utc+timedelta(seconds=entry['outer_timeout_seconds'])).isoformat(),
                    'signals':[],'stop_started':None,'stop_reason':None,'quota_expired':False,'quota_stop_initiated':False,'current_before_stop':None}
                registry.append(child);state['active']=child;state['next']+=1  # Register before fallible identity reads.
                child['identity']=rt.identity(process.pid)
                require(child['identity'].get('ppid')==os.getpid() and child['identity'].get('argv')==entry['command']
                    and type(child['identity'].get('start_ticks')) is int and child['identity']['start_ticks']>0,'Captured owned child identity differs')
                write(output/'logs'/(entry['stream']+'_'+entry['stage']+'.launch.json'),{'entry':entry,'pid':process.pid,'identity':child['identity'],'started_utc':child['started_utc']})
            if child is None:continue
            usage=rt.reap(child)
            if usage is not None:
                for h in child['handles']:h.close()
                state['active']=None
                try:outcome=observe_reaped(child,usage,rt,r['hostname'])
                except Exception as error:
                    child.setdefault('cleanup_errors',[]).append('Outcome accounting: '+str(error));raise
                outcomes[(entry_key(child['entry']))]=outcome
                write(output/'logs'/(child['entry']['stream']+'_'+child['entry']['stage']+'.outcome.json'),outcome)
                abort_box[0]=abort_box[0] or fatal_outcome(outcome);continue
            now=rt.mono()
            if (abort_box[0] or now>=child['deadline']) and child['stop_started'] is None:
                child['stop_started']=now;child['quota_stop_initiated']=not bool(abort_box[0]);child['quota_expired']=now>=child['deadline']
                child['stop_reason']=abort_box[0] or 'whole_invocation_quota_expired';child['current_before_stop']=status_current(child['entry'])
            if child['stop_started'] is not None:
                elapsed=now-child['stop_started'];sent={x['signal'] for x in child['signals']}
                for threshold,sig in ((0,signal.SIGINT),(5,signal.SIGTERM),(10,signal.SIGKILL)):
                    if elapsed>=threshold and signal.Signals(sig).name not in sent:rt.stop(child,sig)
                require(elapsed<CLEANUP,'Owned evaluator unreaped after bounded cleanup; no subsequent work admitted')
        if not abort_box[0] and rt.future is None and now-last_gpu>=30:
            rt.gpu_submit({c['process'].pid:c for c in registry if c['process'].returncode is None});last_gpu=now
        if now-last_status>=5:
            write(output/'queue_status.json',{'schema':SCHEMA,'state':'aborting' if abort_box[0] else 'running','wave':wave,
                'abort_reason':abort_box[0],'streams':{sid:{'next_stage':s['next'],'active_pid':s['active']['process'].pid if s['active'] else None} for sid,s in states.items()}});last_status=now
        rt.sleep(.2)


def entry_key(entry):return entry['arm'],entry['seed'],entry['stage']


def cleanup_owned(registry,outcomes,context,rt,reason):
    started=rt.mono()
    for child in registry:
        if child['process'].returncode is None:
            child['stop_started']=child['stop_started'] or started;child['stop_reason']=child['stop_reason'] or reason
            child['current_before_stop']=child['current_before_stop'] or status_current(child['entry'])
    while any(c['process'].returncode is None for c in registry) and rt.mono()-started<CLEANUP:
        for child in registry:
            if child['process'].returncode is not None:continue
            elapsed=rt.mono()-child['stop_started'];sent={x['signal'] for x in child['signals']}
            for threshold,sig in ((0,signal.SIGINT),(5,signal.SIGTERM),(10,signal.SIGKILL)):
                if elapsed>=threshold and signal.Signals(sig).name not in sent:
                    try:rt.stop(child,sig)
                    except Exception as error:child.setdefault('cleanup_errors',[]).append('Owned signal: '+str(error))
            try:usage=rt.reap(child)
            except Exception as error:child.setdefault('cleanup_errors',[]).append(str(error));continue
            if usage is not None:
                for h in child['handles']:h.close()
                try:outcomes[entry_key(child['entry'])]=observe_reaped(child,usage,rt,context.release['hostname'])
                except Exception as error:child.setdefault('cleanup_errors',[]).append('Outcome accounting: '+str(error))
        rt.sleep(.1)


def queue_snapshot(output,C):
    tree=C.tree_snapshot(output)
    excluded={'queue_status.json','final_evaluation_ledger.json'}
    tree['entries']=[row for row in tree['entries'] if row[0] not in excluded]
    tree['files']={p:d for p,d in tree['files'].items() if str(Path(p).relative_to(output)) not in excluded}
    return tree


def publish_ledger(args,context,outcomes,registry,abort_reason,started,rt,final_gpu):
    output=args.output_dir;unreaped=[{'pid':c['process'].pid,'identity':c['identity'],'command':c['command'],'signals':c['signals']} for c in registry if c['process'].returncode is None]
    process_report={'schema':SCHEMA,'abort_reason':abort_reason,'final_gpu_inventory':final_gpu,'unreaped_owned_children':unreaped,
        'outcomes':[{'arm':a,'seed':s,'stage':n,'outcome':o} for (a,s,n),o in outcomes.items()],
        'all_children':[{'entry':c['entry'],'pid':c['process'].pid,'identity':c['identity'],'signals':c['signals'],'cleanup_errors':c.get('cleanup_errors',[])} for c in registry]}
    process_sha=write(output/'process_outcomes.json',process_report)
    require(not unreaped,'Unreaped owned children retained; strict final ledger unavailable')
    require(final_gpu.get('status')=='observed' and final_gpu.get('processes')==[], 'Post-cleanup GPU inventory not empty/verified')
    stages=[];trees={}
    for entry in context.entries:
        outcome=outcomes.get(entry_key(entry),never_started(entry,abort_reason or 'Fixed queue allocation ended before this stage started'))
        context.mods.C.validate_outcome(outcome,entry['command'])
        row,tree=account_stage(entry,context.manifests[entry['split']],outcome,context.mods.C);stages.append(row);trees[entry['directory']]=tree
    queue_tree=queue_snapshot(output,context.mods.C)
    ledger={'schema':LEDGER_SCHEMA,'producer':'supervisor_candidate','dataset':'Goop-3D','state':'stopped_all_owned_processes_reaped',
        'unreaped_owned_children':[],'all_pinned_inputs_reverified':True,'cohort_sha256':context.bindings[str(args.cohort)],
        'cohort_audit_sha256':context.bindings[str(args.cohort_audit)],'protocol_sha256':PROTOCOL_SHA,
        'source_manifest_sha256':{s:context.bindings[str(p)] for s,p in (('valid',args.valid_manifest),('test',args.test_manifest))},
        'endpoint_updates':context.release['endpoint_updates'],'stages':stages,'root_release_sha256':context.root_sha,
        'supervisor_sha256':context.bindings[str(Path(__file__).resolve())],'collector_sha256':COLLECTOR_SHA,'ledger_interface_sha256':INTERFACE_SHA,
        'prospective_plan_sha256':PLAN_SHA,'cost_basis':COST_BASIS,'abort_reason':abort_reason,'gpu_uuids':context.release['gpu_uuids'],
        'process_outcomes_sha256':process_sha,'elapsed_before_publication_seconds':rt.mono()-started,'created_utc':rt.now().isoformat(),
        'all_pinned_input_sha256':context.bindings,'stage_tree_snapshots':trees,'queue_tree_snapshot':queue_tree,
        'queue_inventory_excludes':['final_evaluation_ledger.json','queue_status.json'],'full_horizon_runtime_forecast_claim':False,
        'all_required_outcomes_promised':False,'automatic_retry_or_resume':False,'collection_and_analysis_require_separate_root_releases':True}
    raw=encode(ledger)
    status={'schema':SCHEMA,'state':'validated_ledger_before_last_publication','ledger_sha256':hashlib.sha256(raw).hexdigest(),
        'abort_reason':abort_reason,'created_utc':rt.now().isoformat(),'all_thirty_stage_receipts_preserved':True}
    status_sha=write(output/'queue_status.json',status)
    verify_files(context.bindings)
    require(all(context.mods.C.tree_snapshot(Path(d))==tree for d,tree in trees.items()),'Stopped stage tree changed before ledger publication')
    require(queue_snapshot(output,context.mods.C)==queue_tree,'Stopped parent/child queue tree changed before ledger publication')
    require(sha(output/'release_snapshot.json')==hashlib.sha256(encode(context.release)).hexdigest(),'Parent release snapshot differs')
    require(sha(output/'process_outcomes.json')==process_sha and sha(output/'queue_status.json')==status_sha,'Queue evidence changed before ledger publication')
    require(rt.now()+timedelta(seconds=context.release['clock_error_bound_seconds'])<=stamp(context.release['absolute_stop_utc']), 'Root evaluation stop reached; preserve the analysis reserve')
    with (output/'final_evaluation_ledger.json').open('xb') as stream:stream.write(raw)
    return ledger


def execute_queue(args,context,rt):
    output=args.output_dir;output.mkdir(mode=0o700,exist_ok=False);(output/'jobs').mkdir();(output/'logs').mkdir()
    write(output/'release_snapshot.json',context.release)
    registry=[];outcomes={};abort=[None];context.gpu_observations=[];started=rt.mono();previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(signum,frame):abort[0]=abort[0] or 'Supervisor received '+signal.Signals(signum).name
    for sig in previous:signal.signal(sig,interrupted)
    try:
        require(not rt.gpu_now(),'Dedicated GPU host is not idle before queue')
        verify_files(context.bindings);validate_allocation(context.release,context.plan,rt.now())
        for wave in WAVES:
            if abort[0]:break
            begin=rt.now().isoformat();drive_wave(context,wave,rt,output,registry,outcomes,abort)
            require(not any(c['process'].returncode is None for c in registry),'Wave has unreaped owned children')
            verify_files(context.control_bindings)
            write(output/('wave_'+wave+'.json'),{'state':'stopped','started_utc':begin,'ended_utc':rt.now().isoformat(),'abort_reason':abort[0]})
    except BaseException as error:abort[0]=abort[0] or type(error).__name__+': '+str(error)
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        try:cleanup_owned(registry,outcomes,context,rt,abort[0] or 'Final owned-child cleanup')
        except BaseException as error:abort[0]=abort[0] or 'Cleanup exception: '+type(error).__name__+': '+str(error)
        finally:
            for c in registry:
                for h in c['handles']:h.close()
            for sig,handler in previous.items():signal.signal(sig,handler)
        try:rt.close()
        except BaseException as error:abort[0]=abort[0] or 'Runtime close exception: '+type(error).__name__+': '+str(error)
    try:
        try:final_gpu={'status':'observed','processes':rt.gpu_now(),'checked_utc':rt.now().isoformat()}
        except Exception as error:final_gpu={'status':'failed','error':type(error).__name__+': '+str(error)}
        ledger=publish_ledger(args,context,outcomes,registry,abort[0],started,rt,final_gpu)
        return 1 if abort[0] else 0
    except BaseException as error:
        write(output/'failed_publication.json',{'schema':SCHEMA,'state':'strict_ledger_unavailable','error_type':type(error).__name__,
            'error':str(error),'abort_reason':abort[0],'all_outputs_retained':True,
            'unreaped_owned_children':[c['process'].pid for c in registry if c['process'].returncode is None]})
        return 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for name in ('release','output-dir',*(n.replace('_','-') for n in CONTROL_NAMES)):p.add_argument('--'+name,type=Path)
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','training_or_test_access':False,'stage_count':30,
            'stage_order_and_quotas':'root must prospectively declare identical allocations for all six models',
            'evaluation_activity_quota_seconds':21600,'analysis_reserve_seconds':3600,'automatic_retry_or_resume':False},indent=2));return 0
    require(all(getattr(args,n) is not None for n in ('release','output_dir',*CONTROL_NAMES)) and sys.platform.startswith('linux')
        and hasattr(os,'wait4') and os.environ.get('CUDA_VISIBLE_DEVICES') is None,'Explicit dedicated unremapped Linux execution inputs required')
    for k,v in vars(args).copy().items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    require(not args.output_dir.exists(),'Fresh queue directory required; never resume/retry automatically')
    raw=args.release.read_bytes();mods=modules();context=validate_release(args,raw,mods)
    return execute_queue(args,context,Runtime(mods.B))


if __name__=='__main__':raise SystemExit(main())
