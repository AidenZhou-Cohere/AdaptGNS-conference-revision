#!/usr/bin/env python3
"""Root-only Sand stopped analysis in one original3600-second phase.

Help/describe is inert; prepare copies only local bindings. All science remains
in the exact frozen shared owner and six unchanged workers. Failures persist.
"""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,math,os,shlex,signal,subprocess,sys,time
K=Path(__file__).resolve().parent;PREP=K.parent
R='/root/repos/AdaptGNS-cuda-20261006';RP=R+'/cuda_preparation';A=R+'/sand_final_analysis_20261006_v1'
PYTHON=R+'/.venv/bin/python';GLOBAL_STOP='2026-10-07T04:00:00+00:00'
EVAL_STATE=PREP/'sand_final24_released_root_v1'
ALIASES={'A':'yellow-worm-77.coder','B':'aquamarine-toad-75.coder'}
OPS=('sand_collect_A','sand_collect_B','sand_saved_A','sand_saved_B','sand_summarize','sand_paired')
DEPS={'sand_collect_A':(),'sand_collect_B':(),'sand_saved_A':('sand_collect_A',),'sand_saved_B':('sand_collect_B',),'sand_summarize':('sand_collect_A','sand_collect_B'),'sand_paired':('sand_collect_A','sand_collect_B','sand_saved_A','sand_saved_B','sand_summarize')}
SSH_SHA='f04af81d2b926b04f7a8063df5c8535da1d8c9aa637e15cf235c261d6d0725b2'
def need(ok,message):
    if not ok:raise ValueError(message)

def now():return datetime.datetime.now(datetime.timezone.utc)

def dt(x):
    v=datetime.datetime.fromisoformat(x.replace('Z','+00:00'));need(v.tzinfo is not None,'Timezone required');return v

def read(p):return json.loads(Path(p).read_bytes())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(x):return isinstance(x,str) and len(x)==64 and set(x)<=set('0123456789abcdef')

def put(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def merge(*maps):
    out={}
    for mapping in maps:
        for key,value in mapping.items():need(key not in out or out[key]==value,'Conflicting byte pin: '+key);out[key]=value
    return out

def bounded_local(state,argv,tag,stop,mono_stop,stdin=None,limit=None):
    """Reviewed transfer/B-operator transport pattern, with NO prephase fallback.

    Both stops are mandatory. TERM3/KILL3 and publication tail are inside the
    supplied original bound; local SSH closure never asserts remote closure.
    """
    start=now();tick=time.monotonic();remaining=lambda:min((stop-now()).total_seconds(),mono_stop-time.monotonic())
    timeout=remaining()-12
    if limit is not None:timeout=min(timeout,limit)
    need(timeout>0,'No bounded transport interval');put(state/(tag+'.command.json'),{'argv':argv,'timeout_seconds':timeout,'stop_utc':stop.isoformat()})
    proc=subprocess.Popen(argv,stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    timed=False;signals=[];out='';err='';failure=None
    def send(sig):
        try:os.killpg(proc.pid,sig);signals.append(signal.Signals(sig).name)
        except ProcessLookupError:pass
    def tail():return max(0,min(3,remaining()-5))
    try:
        try:out,err=proc.communicate(input=stdin,timeout=timeout)
        except subprocess.TimeoutExpired:
            timed=True;send(signal.SIGTERM)
            try:out,err=proc.communicate(timeout=tail())
            except subprocess.TimeoutExpired:send(signal.SIGKILL);out,err=proc.communicate(timeout=tail())
    except BaseException as e:
        failure=type(e).__name__+': '+str(e)
        if proc.poll() is None:
            send(signal.SIGKILL if timed else signal.SIGTERM)
            try:out,err=proc.communicate(timeout=tail())
            except subprocess.TimeoutExpired:
                send(signal.SIGKILL)
                try:out,err=proc.communicate(timeout=tail())
                except subprocess.TimeoutExpired:pass
        raise
    finally:
        (state/(tag+'.stdout')).write_text(out);(state/(tag+'.stderr')).write_text(err)
        put(state/(tag+'.external.json'),{'started_utc':start.isoformat(),'observed_utc':now().isoformat(),'elapsed_seconds':time.monotonic()-tick,'exit_code':proc.returncode,'failure':failure,'local_transport_timeout':timed,'signals_to_own_local_group':signals,'local_transport_pid':proc.pid,'local_transport_reaped':proc.poll() is not None,'remote_closure_separately_required':True})
    need(not timed and proc.returncode==0 and proc.poll() is not None and not signals,'Transport failed; preserve original attempt and do not retry')
    need(remaining()>5,'Original deadline reached during transport');return out

def stage_rows(files):
    return [{'path':remote,'sha256':sha(local),'base64':base64.b64encode(Path(local).read_bytes()).decode(),'allow_identical_existing':same} for local,remote,same in files]

def save_blobs(state,rows,namespace,allowed):
    paths={}
    for row in rows:
        need(row['path'] in allowed and row['path'] not in paths,'Unexpected/duplicate transport path');raw=base64.b64decode(row['base64']);need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Transport bytes differ')
        dest=state/namespace/row['path'][len(R)+1:];dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(raw)
        paths[row['path']]=dest
    return paths

def exact(left,right):return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)
def bindings():return read(K/'bindings.json')
def spec(op):return read(K/'operations.json')['operations'][op]
def role_of(op):return spec(op)['host_role']
def package_check(pin):
    need(not sys.flags.optimize and digest(pin) and sha(K/'manifest.json')==pin,'Exact independently reviewed operator manifest required')
    for path,h in read(K/'manifest.json')['files_sha256'].items():need(sha(K/path)==h,'Frozen package member changed: '+path)
    need(sha(PREP/'ssh_config')==SSH_SHA,'Reviewed transport changed')
    for path,h in bindings()['original_evidence_sha256'].items():need(sha(path)==h,'Accepted original evidence changed: '+path)
def phase(state):
    value=read(state/'analysis_phase.json');anchor=read(state/'local_phase_anchor.json');start=dt(value['original_analysis_started_utc']);stop=dt(value['original_analysis_stop_utc'])
    need(value['schema']=='adaptgns_sand_original_analysis_phase_v1' and value['issued_by']=='root' and value['status']=='approved_original_analysis_phase' and value['dataset']=='Sand','Original root Sand phase required')
    need(value['original_analysis_seconds']==3600 and value['new_or_restarted_clock_granted'] is False and (stop-start).total_seconds()==3600 and stop<=dt(GLOBAL_STOP) and value['global_analysis_deadline_utc']==GLOBAL_STOP,'Exact full original3600second phase required')
    need(value['original_evaluation_sessions']=={'A':51595,'B':12559} and anchor['phase_sha256']==sha(state/'analysis_phase.json') and anchor['utc']==start.isoformat(),'Original sessions and clock anchor changed')
    elapsed=time.monotonic()-anchor['monotonic_seconds'];need(0<=elapsed<=3600 and abs((now()-start).total_seconds()-elapsed)<=5,'Original UTC/monotonic clock changed')
    for role in ALIASES:
        closure=read(state/(role+'.evaluation_closure.json'));binding=value['original_evaluation_closures'][role]
        need(binding=={'path':A+'/controls/'+role+'.evaluation_closure.json','sha256':sha(state/(role+'.evaluation_closure.json'))} and closure['status']=='original_exit_and_native_absence_verified' and dt(closure['checked_utc'])<=start,'All exact original evaluator closures must precede phase')
    return value,stop,anchor['monotonic_seconds']+3600
def budget(state,tail=0):
    value,stop,mono=phase(state);remaining=min((stop-now()).total_seconds(),mono-time.monotonic())-5-tail
    need(remaining>0,'Original analysis phase exhausted');return value,stop,mono,remaining
def historical_pids(state,role):
    pids={int(pid) for pid,absent in read(state/(role+'.evaluation_closure.json'))['native_absent'].items() if absent is True}
    for op in OPS:
        path=state/(op+'.review.json')
        if role_of(op)==role and path.exists():
            value=read(path);need(value['analysis_phase_sha256']==sha(state/'analysis_phase.json') and all(x is True for x in value['native_absent'].values()),'Prior operation closure changed');pids.update(int(pid) for pid in value['native_absent'])
    return sorted(pids)
def ssh_argv(role,command):return ['ssh','-T','-F',str(PREP/'ssh_config'),ALIASES[role],shlex.join(command)]
def probe(state,role,payload,tag,stop=None,mono_stop=None):
    if stop is None:_,stop,mono_stop,_=budget(state,20)
    cfg=bindings()['roles'][role];payload={**payload,'role':role,'boot_id':cfg['boot_id'],'stop_utc':stop.isoformat()}
    seconds=math.floor(min((stop-now()).total_seconds(),mono_stop-time.monotonic()))-15;need(seconds>0,'No operational probe cleanup allowance')
    command=['/usr/bin/timeout','--signal=TERM','--kill-after=3s',str(seconds)+'s',PYTHON,'-I','-S','-B','-c',(K/'remote_probe.py').read_text()]
    result=json.loads(bounded_local(state,ssh_argv(role,command),tag,stop,mono_stop,json.dumps(payload)))
    need(result['hostname']==cfg['hostname'] and result['host_boot_id']==cfg['boot_id'],'Exact qualified host/boot required');result['root_reference_utc']=now().isoformat();put(state/(tag+'.json'),result);return result
def fresh_clock(state,role,capture):
    _,stop,_,_=budget(state,25);clock=dict(capture['clock']);age=(now()-dt(clock['host_utc'])).total_seconds()
    need(0<=age<=300 and clock['host_boot_id']==bindings()['roles'][role]['boot_id'] and abs((dt(capture['root_reference_utc'])-dt(clock['host_utc'])).total_seconds())<=5,'Fresh original-host/root clock required')
    clock.update(source='root_fresh_tool_and_host_clock_evidence',root_reference_utc=capture['root_reference_utc'],error_bound_seconds=5);return clock,stop
def stage_controls(state,role,files,tag):
    return probe(state,role,{'action':'stage_runtime','stage_files':stage_rows(files),'expected_sha256':bindings()['environment_expected_sha256'],'fresh_paths':[],'native_pids':historical_pids(state,role)},tag)
def runtime_ready(state,role):
    value=read(state/(role+'.runtime_ready.json'));need(value['status']=='accepted_prior_native_qualification_on_fresh_runtime' and value['analysis_phase_sha256']==sha(state/'analysis_phase.json') and value['bindings_sha256']==sha(K/'bindings.json'),'This original phase lacks qualified host readiness')
    need(value['runtime_capture_sha256']==sha(state/(role+'.stage_runtime.json')),'Qualified runtime capture changed');return value
def original_exit(state,op):
    value=read(state/(op+'.root_original_external_exit.json'));external=read(state/(op+'.external.json'))
    need(value['recorded_by']=='root' and type(value['original_tool_session']) is int and value['original_tool_session']>0 and type(value['original_tool_exit_code']) is int and value['original_tool_exit_code']==0,'Actual original analysis tool exit0 required')
    need(sha(value['evidence_path'])==value['evidence_sha256'] and sha(state/(op+'.external.json'))==value['original_transport_sha256'] and type(external['exit_code']) is int and external['exit_code']==0 and external['failure'] is None and external['local_transport_reaped'] is True and external['local_transport_timeout'] is False and external['signals_to_own_local_group']==[],'Original bounded analysis transport failed')
    value_phase,stop,_,_=budget(state);need(dt(value_phase['original_analysis_started_utc'])<=dt(external['started_utc'])<=dt(external['observed_utc'])<=dt(value['observed_utc'])<=now()<stop,'Original analysis exit observation order differs');return value
def accepted(state,op):
    value=read(state/(op+'.review.json'));need(value['status']=='accepted_stopped_product' and value['analysis_phase_sha256']==sha(state/'analysis_phase.json') and value['cpu_release_sha256']==sha(state/(op+'.cpu_release.json')),'Missing accepted original operation/closure')
    original_exit(state,op);need(sha(value['local_product'])==value['product_sha256'] and value['original_tool_exit_sha256']==sha(state/(op+'.root_original_external_exit.json')),'Accepted predecessor bytes changed');return value,read(value['local_product'])
def require_predecessors(state,op):
    for parent in DEPS[op]:
        receipt,_=accepted(state,parent)
        if parent.startswith('sand_collect_'):need(receipt['semantic_review']['eligible_for_aggregation'] is True,'Preserved collection has unresolved original input integrity; downstream science cannot run')
    if op in ('sand_summarize','sand_paired'):
        copied=read(state/'B.scalars_transferred_to_A.json');need(copied['analysis_phase_sha256']==sha(state/'analysis_phase.json'),'Same-phase B scalar transfer required')
        for parent in ('sand_collect_B','sand_saved_B'):
            need(copied['transferred_sha256'][spec(parent)['outputs'][0]]==accepted(state,parent)[0]['product_sha256'],'Transferred B scalar must retain exact accepted bytes')
def begin(state):
    closures={role:read(state/(role+'.evaluation_closure.json')) for role in ALIASES}
    for role,c in closures.items():
        cfg=bindings()['roles'][role];record=read(state/(role+'.evaluation.root_original_external_exit.json'))
        need(c['status']=='original_exit_and_native_absence_verified' and c['issued_by']=='root' and c['dataset']=='Sand' and c['host_role']==role and c['hostname']==cfg['hostname'] and c['original_queue_root']==cfg['queue_root'] and type(c['original_tool_session']) is int and c['original_tool_session']==cfg['session_id'] and type(c['original_exit_code']) is int and c['all_children_reaped'] is True and c['complete_registered_child_ledger_checked'] is True and c['native_absent'] and all(x is True for x in c['native_absent'].values()),'Both original evaluation closures required')
        need(c['original_tool_exit_receipt_sha256']==sha(state/(role+'.evaluation.root_original_external_exit.json')) and record['original_transport_sha256']==sha(EVAL_STATE/(role+'.evaluation.external.json')) and record['evidence_sha256']==sha(record['evidence_path']) and exact(c['original_exit_code'],record['original_tool_exit_code']),'Original evaluation closure/exit bytes changed')
    start=now();tick=time.monotonic();stop=start+datetime.timedelta(seconds=3600);need(stop<=dt(GLOBAL_STOP),'Entire original analysis hour no longer fits')
    value=read(K/'candidates/Sand.analysis_phase.candidate.json');value.update(issued_by='root',status='approved_original_analysis_phase',original_analysis_started_utc=start.isoformat(),original_analysis_stop_utc=stop.isoformat(),original_evaluation_sessions={'A':51595,'B':12559})
    for role,c in closures.items():need(dt(c['checked_utc'])<=start,'Analysis clock predates evaluator closure');value['original_evaluation_closures'][role]['sha256']=sha(state/(role+'.evaluation_closure.json'))
    put(state/'analysis_phase.json',value);put(state/'local_phase_anchor.json',{'utc':start.isoformat(),'monotonic_seconds':tick,'phase_sha256':sha(state/'analysis_phase.json')});print(json.dumps(value))
def stage_runtime(state,role):
    b=bindings();files=[(K/row['package_path'],path,True) for path,row in b['source_files'].items()]
    files += [(state/'analysis_phase.json',A+'/controls/analysis_phase.json',False)]+[(state/(other+'.evaluation_closure.json'),A+'/controls/'+other+'.evaluation_closure.json',False) for other in ALIASES]
    own=[op for op in OPS if role_of(op)==role];fresh=[A+'/owners/'+op for op in own]+[spec(op)['outputs'][0] for op in own]
    if role=='A':fresh += [A+'/B.stopped_collection.json',A+'/B.saved_array_audit.json']
    expected=merge(b['environment_expected_sha256'],{path:row['sha256'] for path,row in b['source_files'].items()})
    capture=probe(state,role,{'action':'stage_runtime','stage_files':stage_rows(files),'expected_sha256':expected,'fresh_paths':fresh,'prepare_empty_owner_parent':True,'native_pids':historical_pids(state,role)},role+'.stage_runtime')
    need(capture['prepared_owner_parent']['path']==A+'/owners' and capture['prepared_owner_parent']['empty'] is True and capture['runtime_sha256']==expected,'Complete source/runtime/parent readiness required');fresh_clock(state,role,capture)
    put(state/(role+'.runtime_ready.json'),{'status':'accepted_prior_native_qualification_on_fresh_runtime','analysis_phase_sha256':sha(state/'analysis_phase.json'),'bindings_sha256':sha(K/'bindings.json'),'runtime_capture_sha256':sha(state/(role+'.stage_runtime.json')),'prior_qualification_review_sha256':b['qualification_reuse']['accepted_review_sha256'],'qualified_host_boot_id':b['roles'][role]['boot_id'],'native_absent':capture['native_absent'],'new_qualification_performed':False})
def snapshot(state,op,pins,tag,with_documents=False):
    role=role_of(op);cfg=bindings()['roles'][role];Q=cfg['queue_root'];required=[];optional=[]
    if with_documents:
        required=[Q+'/'+name for name in ('release_snapshot.json','queue_status.json','coverage_ledger.json','process_outcomes.json')];optional=[Q+'/gpu_observations.json']
        release=read(K/'candidates'/(role+'.original_evaluation_release.json'))
        optional += [command[command.index('--output-dir')+1]+'/protocol.json' for stream in release['streams'] for command in stream['commands']]
    result=probe(state,role,{'action':'snapshot','native_pids':historical_pids(state,role),'inputs_sha256':pins,'tree_roots':spec(op)['protected_tree_roots'],'documents_required':required,'documents_optional':optional},tag)
    if with_documents:
        allowed=set(required+optional);paths=save_blobs(state,result['documents'],tag+'.documents',allowed);need(set(required)<=set(paths),'Complete stopped queue metadata required')
        for path in paths:need(sha(paths[path])==result['inputs_sha256'][path],'Transported stopped scalar bytes differ')
    return result
def unchanged_queue(previous,current,root):
    need(exact(previous['queue_inventories'][root],current['queue_inventories'][root]),'Original stopped queue membership/hash/size changed')
def publication_order(release,terminal,external):
    published=dt(terminal['publication_utc']);error=release['clock_sample']['error_bound_seconds']
    need(type(error) is int and error==5,'Exact original host/root clock tolerance required')
    need(dt(release['invocation_origin_utc'])<=published<dt(release['publication_deadline_utc']),'Original same-host publication interval differs')
    need(published<=dt(external['observed_utc'])+datetime.timedelta(seconds=error),'Original host/root publication observation order differs')
def issue_and_run(state,op):
    role=role_of(op);runtime_ready(state,role);require_predecessors(state,op);s=spec(op);pins=dict(s['inputs_sha256']);core=None
    if op.startswith('sand_collect_'):
        Q=bindings()['roles'][role]['queue_root'];initial=snapshot(state,op,{p:h for p,h in pins.items() if h is not None},op+'.inventory',True)
        core=read(K/'candidates'/(role+'.collection_release.candidate.json'));inventory=initial['queue_inventories'][Q]
        core.update(issued_by='root',status='approved_for_stopped_scalar_collection',local_queue_inventory_sha256=hashlib.sha256((json.dumps(inventory,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest(),queue_release_sha256=inventory['files']['release_snapshot.json']['sha256'])
        put(state/(role+'.collection_release.json'),core);stage_controls(state,role,[(state/(role+'.collection_release.json'),A+'/controls/'+role+'.collection_release.json',False)],op+'.stage_core')
    elif op=='sand_summarize':
        core=read(K/'candidates/summary_release.candidate.json');core.update(issued_by='root',status='approved_for_fixed_scalar_aggregation',collection_sha256={r:accepted(state,'sand_collect_'+r)[0]['product_sha256'] for r in ALIASES})
        put(state/'summary_release.json',core);stage_controls(state,role,[(state/'summary_release.json',A+'/controls/summary_release.json',False)],op+'.stage_core')
    for path,h in list(pins.items()):
        if h is None:
            if path.startswith(A+'/controls/'):pins[path]=sha(state/Path(path).name)
            else:
                matches=[parent for parent in DEPS[op] if spec(parent)['outputs']==[path]];need(len(matches)==1,'Unresolved exact predecessor input');pins[path]=accepted(state,matches[0])[0]['product_sha256']
    capture=snapshot(state,op,pins,op+'.snapshot',op.startswith('sand_collect_'))
    if s['protected_tree_roots']:
        Q=s['protected_tree_roots'][0];previous=read(state/(op+'.inventory.json')) if op.startswith('sand_collect_') else read(state/('sand_collect_'+role+'.snapshot.json'));unchanged_queue(previous,capture,Q)
        for path,h in read(state/(role+'.evaluation_closure.json'))['closure_metadata_sha256'].items():need(capture['inputs_sha256'][path]==h,'Original native-closure metadata changed')
    clock,stop=fresh_clock(state,role,capture);release=read(K/'candidates'/(op+'.cpu_release.candidate.json'))
    release.update(issued_by='root',status='approved_one_bounded_stopped_analysis_invocation',issued_utc=now().isoformat(),clock_sample=clock,invocation_origin_utc=clock['host_utc'],whole_invocation_seconds=(stop-dt(clock['host_utc'])).total_seconds(),work_stop_utc=(stop-datetime.timedelta(seconds=15)).isoformat(),cleanup_deadline_utc=(stop-datetime.timedelta(seconds=5)).isoformat(),publication_deadline_utc=stop.isoformat(),inputs_sha256=capture['inputs_sha256'],protected_tree_entries=capture['protected_tree_entries'])
    release['hard_deadline_monotonic_ns']=int((clock['host_monotonic_seconds']+release['whole_invocation_seconds']-5)*10**9);release['analysis_phase']['sha256']=sha(state/'analysis_phase.json');release['outer_timeout'].update(sha256=bindings()['environment_expected_sha256']['/usr/bin/timeout'],version=read(state/(role+'.stage_runtime.json'))['timeout_version'])
    release['command']=[release['inputs_sha256'][item['sha256_of_input']] if isinstance(item,dict) else item for item in s['argv']]
    put(state/(op+'.cpu_release.json'),release);stage_controls(state,role,[(state/(op+'.cpu_release.json'),A+'/controls/'+op+'.cpu_release.json',False)],op+'.stage_cpu')
    _,stop,mono,_=budget(state,25);argv=[PYTHON,'-I','-S','-B',RP+'/launch_stopped_analysis_cpu_v1.py','--hard-deadline-monotonic-ns',str(release['hard_deadline_monotonic_ns']),'--release',A+'/controls/'+op+'.cpu_release.json','--release-sha256',sha(state/(op+'.cpu_release.json'))]
    put(state/(op+'.launch.json'),{'argv':argv,'command':release['command'],'runtime_ready_sha256':sha(state/(role+'.runtime_ready.json')),'predecessor_review_sha256':{parent:sha(state/(parent+'.review.json')) for parent in DEPS[op]}})
    bounded_local(state,ssh_argv(role,argv),op,stop,mono)

def close_product(state,op):
    from review_products import validate_collection,validate_saved_audit,validate_summary,validate_paired_audit
    role=role_of(op);runtime_ready(state,role);original_exit(state,op);release=read(state/(op+'.cpu_release.json'));s=spec(op)
    capture=probe(state,role,{'action':'capture','directory':release['owner_output_dir'],'outputs':s['outputs'],'historical_pids':historical_pids(state,role),'tree_entries':release['protected_tree_entries']},op+'.capture')
    names={'owner_started.json','child_registered.json','child.stdout','child.stderr','owner_terminal.json'};allowed={release['owner_output_dir']+'/'+name for name in names}|set(s['outputs'])
    paths=save_blobs(state,capture['files'],op+'.collected',allowed);need(set(paths)==allowed,'Complete original owner evidence and scalar product required')
    terminal=read(paths[release['owner_output_dir']+'/owner_terminal.json']);started=read(paths[release['owner_output_dir']+'/owner_started.json']);registered=read(paths[release['owner_output_dir']+'/child_registered.json']);child=terminal['child']
    product_path=paths[s['outputs'][0]];product=read(product_path);product_pin=sha(product_path)
    need(terminal['schema']=='adaptgns_stopped_analysis_cpu_terminal_v1' and terminal['status']=='complete' and terminal['failure'] is None and terminal['release_sha256']==sha(state/(op+'.cpu_release.json')) and terminal['child_native_absent'] is True and terminal['root_original_tool_exit_and_timeout_owner_child_native_closure_required'] is True,'Original operation terminal failed')
    need(child['reaped'] is True and type(child['exit_code']) is int and child['exit_code']==0 and child['signals']==[] and child['cleanup_errors']==[] and exact(child['command'],release['command']),'Original CPU worker failed or differs')
    expected=merge(release['inputs_sha256'],bindings()['environment_expected_sha256'],{A+'/controls/'+op+'.cpu_release.json':sha(state/(op+'.cpu_release.json'))},{A+'/controls/'+r+'.evaluation_closure.json':sha(state/(r+'.evaluation_closure.json')) for r in ALIASES},{release[key]['path']:release[key]['sha256'] for key in ('operation_spec','analysis_phase','owner_source','bootstrap_source')})
    need(exact(terminal['input_sha256'],expected) and terminal['output_sha256']=={s['outputs'][0]:product_pin},'Original owner final full input/product byte pins differ')
    need(exact(started['owner_identity'],terminal['owner_identity']) and exact(started['outer_timeout_identity'],terminal['outer_timeout_identity']) and exact(registered['identity'],child['identity']) and type(registered['pid']) is int and registered['pid']==child['pid'] and exact(registered['owner_identity'],terminal['owner_identity']),'Original outer/owner/registered child identities differ')
    need(exact(started['native_hard_guard'],terminal['native_hard_guard']) and terminal['native_hard_guard']['absolute_deadline_ns']==release['hard_deadline_monotonic_ns'] and terminal['scientific_admission'] is False,'Original absolute native guard/scientific-admission scope differs')
    need(started['release_sha256']==terminal['release_sha256'] and started['scientific_admission'] is False and exact(started['original_clock_sample'],release['clock_sample']) and exact(started['native_outer_timing'],terminal['native_outer_timing']),'Original owner clock/release/native timing differs')
    need(set(terminal['evidence_sha256'])=={release['owner_output_dir']+'/'+name for name in names-{'owner_terminal.json'}} and capture['final_input_and_output_hashes_verified'] is True,'Complete original owner evidence hash set required')
    publication_order(release,terminal,read(state/(op+'.external.json')))
    for field in ('evidence_sha256','output_sha256'):
        for path,h in terminal[field].items():need(sha(paths[path])==h,'Transported original owner/product bytes differ')
    e={'artifact_sha256':product_pin}
    if op.startswith('sand_collect_'):
        snap=read(state/(op+'.snapshot.json'));Q=bindings()['roles'][role]['queue_root'];base=state/(op+'.snapshot.documents')
        need(exact(snap['inputs_sha256'],release['inputs_sha256']) and exact(snap['protected_tree_entries'],release['protected_tree_entries']),'Original collection snapshot/release differs')
        docs={row['path']:row for row in snap['documents']};need(len(docs)==len(snap['documents']),'Duplicate original snapshot scalar')
        def transported(path):
            local=base/path.removeprefix(R+'/');row=docs[path]
            need(sha(local)==row['sha256']==release['inputs_sha256'][path] and local.stat().st_size==row['bytes'],'Original transported scalar bytes changed')
            return read(local)
        protocols={path.removeprefix(Q+'/'):transported(path) for path in docs if path.endswith('/protocol.json')}
        e.update(host_role=role,root_release_sha256=sha(state/(role+'.collection_release.json')),root_release=read(state/(role+'.collection_release.json')),queue_release_sha256=snap['queue_inventories'][Q]['files']['release_snapshot.json']['sha256'],queue_release=transported(Q+'/release_snapshot.json'),inventory=snap['queue_inventories'][Q],ledger=transported(Q+'/coverage_ledger.json'),queue_status=transported(Q+'/queue_status.json'),process_outcomes=transported(Q+'/process_outcomes.json'),gpu_observations=transported(Q+'/gpu_observations.json') if Q+'/gpu_observations.json' in docs else None,protocols=protocols)
        review=validate_collection(product,product_pin,e)
    elif op.startswith('sand_saved_'):
        cr,collection=accepted(state,'sand_collect_'+role);e['collection_sha256']=cr['product_sha256'];review=validate_saved_audit(product,product_pin,e,collection)
    else:
        previous={r:accepted(state,'sand_collect_'+r) for r in ALIASES};collections={r:item[1] for r,item in previous.items()};e['collection_sha256']={r:item[0]['product_sha256'] for r,item in previous.items()}
        if op=='sand_summarize':e['root_release_sha256']=sha(state/'summary_release.json');review=validate_summary(product,product_pin,e,collections)
        else:
            sr,summary=accepted(state,'sand_summarize');audits={r:accepted(state,'sand_saved_'+r) for r in ALIASES};e.update(summary_sha256=sr['product_sha256'],audit_sha256={r:item[0]['product_sha256'] for r,item in audits.items()});review=validate_paired_audit(product,product_pin,e,collections,summary,{r:item[1] for r,item in audits.items()})
    budget(state);need(sha(product_path)==product_pin,'Product changed during review')
    put(state/(op+'.review.json'),{'status':'accepted_stopped_product','operation':op,'checked_utc':now().isoformat(),'analysis_phase_sha256':sha(state/'analysis_phase.json'),'cpu_release_sha256':sha(state/(op+'.cpu_release.json')),'product_sha256':product_pin,'local_product':str(product_path),'native_absent':capture['native_absent'],'original_tool_exit_sha256':sha(state/(op+'.root_original_external_exit.json')),'semantic_review':review,'all_original_failures_timeouts_unexecuted_denominators_preserved':True,'scientific_success_not_implied':True})
def transfer_b(state):
    runtime_ready(state,'A');files=[]
    for op in ('sand_collect_B','sand_saved_B'):
        receipt,_=accepted(state,op);files.append((Path(receipt['local_product']),spec(op)['outputs'][0],False))
    result=probe(state,'A',{'action':'transfer','stage_files':stage_rows(files),'native_pids':historical_pids(state,'A')},'B.scalars_to_A')
    need(result['transferred_sha256']=={remote:sha(local) for local,remote,_ in files},'B scalar transfer differs')
    put(state/'B.scalars_transferred_to_A.json',{'analysis_phase_sha256':sha(state/'analysis_phase.json'),'transferred_sha256':result['transferred_sha256'],'capture_sha256':sha(state/'B.scalars_to_A.json'),'original_B_review_sha256':{op:sha(state/(op+'.review.json')) for op in ('sand_collect_B','sand_saved_B')}})
def close_evaluation(state,role,deadline):
    need(not (state/'analysis_phase.json').exists() and deadline,'Explicit pre-phase short closure bound required');stop=dt(deadline);need(20<(stop-now()).total_seconds()<=60,'Closure observation allowance must be20to60seconds and is not analysis time');mono_stop=time.monotonic()+(stop-now()).total_seconds()
    cfg=bindings()['roles'][role];record=read(state/(role+'.evaluation.root_original_external_exit.json'));extpath=EVAL_STATE/(role+'.evaluation.external.json');external=read(extpath)
    need(record['recorded_by']=='root' and type(record['original_tool_session']) is int and record['original_tool_session']==cfg['session_id'] and type(record['original_tool_exit_code']) is int,'Exact original evaluation exit attestation required')
    need(sha(record['evidence_path'])==record['evidence_sha256'] and sha(extpath)==record['original_transport_sha256'] and external['local_transport_reaped'] is True and dt(external['observed_utc'])<=dt(record['observed_utc'])<=now(),'Original evaluation tool response/transport changed')
    capture=probe(state,role,{'action':'evaluation_closure','original_tool_session':cfg['session_id'],'original_exit_observed':True},role+'.evaluation_closure_capture',stop,mono_stop)
    need(capture['original_release_sha256']==cfg['release_sha256'] and capture['all_children_reaped'] is True and capture['complete_registered_child_ledger_checked'] is True,'Exact original evaluation closure required')
    closure=read(K/'candidates'/('Sand.'+role+'.evaluation_closure.candidate.json'));closure.update(capture);closure.update(issued_by='root',status='original_exit_and_native_absence_verified',original_tool_session=cfg['session_id'],original_exit_code=record['original_tool_exit_code'],original_tool_exit_observed_utc=record['observed_utc'],original_tool_exit_receipt_sha256=sha(state/(role+'.evaluation.root_original_external_exit.json')),original_owner_identity=cfg['owner'])
    put(state/(role+'.evaluation_closure.json'),closure)
def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('action',choices=('describe','prepare','record-evaluation-exit','close-evaluation','begin','stage-runtime','run','record-original-exit','close-product','transfer-B','complete'));parser.add_argument('--root-action',action='store_true');parser.add_argument('--state',type=Path);parser.add_argument('--package-sha256');parser.add_argument('--role',choices=('A','B'));parser.add_argument('--operation',choices=OPS);parser.add_argument('--session-id',type=int);parser.add_argument('--exit-code',type=int);parser.add_argument('--observed-utc');parser.add_argument('--evidence-file',type=Path);parser.add_argument('--observation-deadline-utc');a=parser.parse_args()
    if a.action=='describe':print(json.dumps({'status':'inert_root_only_Sand_analysis_operator','original_sessions':{'A':51595,'B':12559},'original_analysis_seconds':3600,'operations':OPS,'new_native_qualification_required_on_unchanged_hosts':False}));return
    need(a.root_action and a.state and a.state.is_absolute() and a.state.resolve()==a.state,'Explicit root action and canonical state required');package_check(a.package_sha256);state=a.state
    if a.action=='prepare':
        state.mkdir(exist_ok=False);put(state/'operator_binding.json',{'manifest_sha256':a.package_sha256,'bindings_sha256':sha(K/'bindings.json'),'local_preparation_only':True,'clock_issued':False});return
    need(read(state/'operator_binding.json')['manifest_sha256']==a.package_sha256,'Original operator package changed')
    if a.action=='record-evaluation-exit':
        role=a.role;need(role in ALIASES and a.session_id==bindings()['roles'][role]['session_id'] and type(a.exit_code) is int and a.evidence_file and a.observed_utc and dt(a.observed_utc)<=now(),'Actual fixed original evaluation exit required')
        ext=read(EVAL_STATE/(role+'.evaluation.external.json'));need(ext['local_transport_reaped'] is True and dt(ext['observed_utc'])<=dt(a.observed_utc),'Original evaluation transport must have actually ended')
        put(state/(role+'.evaluation.root_original_external_exit.json'),{'recorded_by':'root','original_tool_session':a.session_id,'original_tool_exit_code':a.exit_code,'observed_utc':a.observed_utc,'evidence_path':str(a.evidence_file.resolve()),'evidence_sha256':sha(a.evidence_file),'original_transport_sha256':sha(EVAL_STATE/(role+'.evaluation.external.json'))});return
    if a.action=='close-evaluation':need(a.role in ALIASES,'Role required');close_evaluation(state,a.role,a.observation_deadline_utc);return
    if a.action=='begin':begin(state);return
    budget(state)
    if a.action=='stage-runtime':need(a.role in ALIASES,'Role required');stage_runtime(state,a.role);return
    if a.action=='transfer-B':transfer_b(state);return
    if a.action=='complete':
        for op in OPS:accepted(state,op)
        budget(state)
        put(state/'analysis_completion.json',{'status':'all_six_original_stopped_analysis_operations_reviewed','analysis_phase_sha256':sha(state/'analysis_phase.json'),'completed_utc':now().isoformat(),'operation_review_sha256':{op:sha(state/(op+'.review.json')) for op in OPS},'all_original_failures_and_unexecuted_denominators_preserved':True,'scientific_success_or_speedup_not_implied':True});return
    need(a.operation in OPS,'One exact operation required');op=a.operation
    if a.action=='record-original-exit':
        need(type(a.session_id) is int and a.session_id>0 and a.session_id not in (51595,12559) and type(a.exit_code) is int and a.observed_utc and a.evidence_file,'Actual original operation tool response required')
        need(all(read(p)['original_tool_session']!=a.session_id for p in state.glob('*.root_original_external_exit.json')),'Original tool session cannot be reused')
        ext=read(state/(op+'.external.json'));need(dt(ext['observed_utc'])<=dt(a.observed_utc)<=now(),'Original tool observation order differs')
        put(state/(op+'.root_original_external_exit.json'),{'recorded_by':'root','original_tool_session':a.session_id,'original_tool_exit_code':a.exit_code,'observed_utc':a.observed_utc,'evidence_path':str(a.evidence_file.resolve()),'evidence_sha256':sha(a.evidence_file),'original_transport_sha256':sha(state/(op+'.external.json'))});return
    if a.action=='run':issue_and_run(state,op)
    elif a.action=='close-product':close_product(state,op)
if __name__=='__main__':main()
