#!/usr/bin/env python3
"""Independent completed Sand evidence reader. No producer, validator, clock or network calls.

This binds the already accepted semantic review to original tool/native evidence
and counts original cell identities. It does not repeat numerical statistics or
saved-array arithmetic. Root remains responsible for the original phase clock.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse, base64, datetime as D, hashlib, json, math, shlex
from review_sand_actual_phase_runtime_code_audit_v1 import P,K,S,R,A,PH,AN,PK,sha,read,exact,dt
PY=R+'/.venv/bin/python'; RP=R+'/cuda_preparation'
OPS=('sand_collect_A','sand_collect_B','sand_saved_A','sand_saved_B','sand_summarize','sand_paired')
DEPS={'sand_collect_A':(),'sand_collect_B':(),'sand_saved_A':('sand_collect_A',),'sand_saved_B':('sand_collect_B',),'sand_summarize':('sand_collect_A','sand_collect_B'),'sand_paired':('sand_collect_A','sand_collect_B','sand_saved_A','sand_saved_B','sand_summarize')}
MODELS={'A':(('base',1),('mix',1),('base',2),('mix',2)),'B':(('base',0),('mix',0))}
STAGES={'full_rollout_test':('full-rollout','test'),'same_state_valid':('same-state','valid'),'same_state_test':('same-state','test'),'clean_validation':('clean-validation','valid')}
POLICIES=('base','dense','random25','speed25','laggedrisk25','relative-velocity-RMS25')
STATES={'completed_required_outcome','recorded_failed_outcome','timed_out_current','not_completed_before_invocation_end','never_started'}
TRUST={'sand_actual_phase_runtime_independent_review_code_audit_v1.json':'ea9279300d526689513f300fe923863e02cf4964bb2d14fa94f4111b37610453','sand_stopped_operator_independent_owner_review_v1.json':'4bfe8b78b12af6d8b0d0e60df4abad77d4836f5e438b25b6f66e0987921a01de','sand_stopped_product_validator_independent_transition_v1.json':'9af90d8654895bab50842bf89624559eda1badefda2713a8b3e51603ee2667d5'}
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def audit(op,session,root_sha):
    checks=[]; evidence={}
    def ck(ok,n):
        if not ok:raise ValueError(n)
        checks.append(n)
    def pin(p,h=None):
        p=Path(p);ck(p.is_file() and not p.is_symlink() and p.suffix not in ('.npz','.npy','.pt','.pth'),'Ordinary nonarray evidence '+p.name)
        value=sha(p);evidence[str(p.resolve())]=value
        if h is not None:ck(value==h,'Exact bytes '+p.name)
        return value
    def doc(p,h=None):pin(p,h);return read(p)
    pin(__file__);ck(op in OPS and type(session) is int and session>0 and session not in (51595,12559),'Exact new original operation session')
    mf=doc(K/'manifest.json',PK)
    for p,h in mf['files_sha256'].items():pin(K/p,h)
    for p,h in TRUST.items():pin(P/p,h)
    b=doc(K/'bindings.json');specs=doc(K/'operations.json')['operations'];spec=specs[op];role=spec['host_role'];cfg=b['roles'][role]
    phase=doc(S/'analysis_phase.json',PH);anchor=doc(S/'local_phase_anchor.json',AN);start=dt(phase['original_analysis_started_utc']);stop=dt(phase['original_analysis_stop_utc'])
    ck((stop-start).total_seconds()==3600 and anchor['phase_sha256']==PH and phase['new_or_restarted_clock_granted'] is False,'Original hour retained without reading clock')
    runtime_trust=doc(P/'sand_actual_phase_runtime_independent_review_code_audit_v1.json',TRUST['sand_actual_phase_runtime_independent_review_code_audit_v1.json'])
    runtime=doc(S/(role+'.stage_runtime.json'),runtime_trust['roles'][role]['runtime_capture_sha256']);ready=doc(S/(role+'.runtime_ready.json'),runtime_trust['roles'][role]['runtime_ready_sha256'])
    closure=doc(S/(role+'.evaluation_closure.json'),phase['original_evaluation_closures'][role]['sha256'])
    root=doc(S/(op+'.review.json'),root_sha);root_time=dt(root['checked_utc'])
    ck(root['status']=='accepted_stopped_product' and root['operation']==op and root['analysis_phase_sha256']==PH and root['all_original_failures_timeouts_unexecuted_denominators_preserved'] is True and root['scientific_success_not_implied'] is True and start<=root_time<stop,'Accepted original root semantic review bound without rerun')
    release=doc(S/(op+'.cpu_release.json'),root['cpu_release_sha256']);release_sha=sha(S/(op+'.cpu_release.json'));candidate=doc(K/'candidates'/(op+'.cpu_release.candidate.json'))
    variable={'status','issued_by','issued_utc','clock_sample','invocation_origin_utc','whole_invocation_seconds','work_stop_utc','cleanup_deadline_utc','publication_deadline_utc','inputs_sha256','protected_tree_entries','hard_deadline_monotonic_ns','analysis_phase','outer_timeout','command'}
    ck(set(release)==set(candidate) and all(exact(release[k],v) for k,v in candidate.items() if k not in variable),'Every fixed original CPU release field unchanged')
    ck(release['status']=='approved_one_bounded_stopped_analysis_invocation' and release['issued_by']=='root' and release['analysis_phase']=={'path':A+'/controls/analysis_phase.json','sha256':PH},'CPU invocation bound to original root phase')
    ext=doc(S/(op+'.external.json'));launch=doc(S/(op+'.launch.json'));exit_record=doc(S/(op+'.root_original_external_exit.json'),root['original_tool_exit_sha256'])
    historical=set(closure['native_absent']);predecessors={};prior_products={}
    for parent in OPS:
        path=S/(parent+'.review.json')
        if parent==op or not path.exists():continue
        prior=read(path)
        if dt(prior['checked_utc'])>dt(ext['started_utc']):continue
        doc(path);ck(prior['status']=='accepted_stopped_product' and prior['analysis_phase_sha256']==PH and all(v is True for v in prior['native_absent'].values()),'Earlier original product closure retained '+parent)
        if specs[parent]['host_role']==role:historical.update(prior['native_absent'])
        if parent in DEPS[op]:
            predecessors[parent]=sha(path);prior_products[parent]=doc(prior['local_product'],prior['product_sha256'])
            if parent.startswith('sand_collect_'):ck(prior['semantic_review']['eligible_for_aggregation'] is True,'Original input integrity allows downstream operation '+parent)
    ck(set(predecessors)==set(DEPS[op]) and exact(launch['predecessor_review_sha256'],predecessors),'Exact prerequisite original root reviews')
    def transport(value,label):
        ck(type(value['exit_code']) is int and value['exit_code']==0 and value['failure'] is None and value['local_transport_reaped'] is True and value['local_transport_timeout'] is False and value['signals_to_own_local_group']==[],'Reaped clean original bounded transport '+label)
        ck(start<=dt(value['started_utc'])<=dt(value['observed_utc'])<stop,'Transport stays inside original hour '+label)
    transport(ext,op)
    def probe(tag,action,absence=None):
        value=doc(S/(tag+'.json'));raw=doc(S/(tag+'.stdout'));external=doc(S/(tag+'.external.json'));command=doc(S/(tag+'.command.json'))
        ck(exact({k:v for k,v in value.items() if k!='root_reference_utc'},raw),'Original probe equals raw stdout plus root reference '+tag)
        pin(S/(tag+'.stderr'));ck((S/(tag+'.stderr')).read_bytes()==b'','Empty original probe stderr '+tag);transport(external,tag)
        argv=command['argv'];remote=shlex.split(argv[-1]);alias={'A':'yellow-worm-77.coder','B':'aquamarine-toad-75.coder'}[role]
        ck(len(argv)==6 and argv[:5]==['ssh','-T','-F',str(P/'ssh_config'),alias] and len(remote)==10 and remote[:3]==['/usr/bin/timeout','--signal=TERM','--kill-after=3s'] and remote[3].endswith('s') and remote[3][:-1].isdigit() and int(remote[3][:-1])>0 and remote[4:9]==[PY,'-I','-S','-B','-c'] and remote[9]==(K/'remote_probe.py').read_text() and command['stop_utc']==stop.isoformat(),'Exact frozen bounded helper '+tag)
        ck(value['action']==action and value['role']==role and value['hostname']==cfg['hostname'] and value['host_boot_id']==cfg['boot_id'],'Original host and action '+tag)
        gpu=value['gpu'];ck(gpu['gpu_uuids']==cfg['gpu_uuids'] and gpu['assigned_devices_empty'] is True and all(x['gpu_uuid'] in cfg['gpu_uuids'] and cfg['gpu_uuids'].index(x['gpu_uuid']) not in cfg['owned_gpu_indices'] for x in gpu['observed_unassigned_gpu_processes']),'Assigned devices empty with other devices observed only '+tag)
        if absence is not None:ck(set(value['native_absent'])==absence and all(v is True for v in value['native_absent'].values()),'Exact complete historical native absence '+tag)
        clock=value['clock'];host=dt(clock['host_utc']);reference=dt(value['root_reference_utc'])
        ck(clock['host_boot_id']==cfg['boot_id'] and finite(clock['host_monotonic_seconds']) and value['checked_utc']==clock['host_utc'] and dt(external['started_utc'])-D.timedelta(seconds=5)<=host<=dt(external['observed_utc'])+D.timedelta(seconds=5) and dt(external['observed_utc'])<=reference<=root_time,'Original host clock inside transport and later root observation '+tag)
        return value,external
    snap,snap_ext=probe(op+'.snapshot','snapshot',historical);clock=snap['clock'];sample=dt(clock['host_utc']);m=clock['host_monotonic_seconds']
    expected_clock=dict(clock,root_reference_utc=snap['root_reference_utc'],source='root_fresh_tool_and_host_clock_evidence',error_bound_seconds=5)
    ck(exact(release['clock_sample'],expected_clock) and abs((dt(snap['root_reference_utc'])-sample).total_seconds())<=5 and release['invocation_origin_utc']==clock['host_utc'] and release['whole_invocation_seconds']==(stop-sample).total_seconds() and release['work_stop_utc']==(stop-D.timedelta(seconds=15)).isoformat() and release['cleanup_deadline_utc']==(stop-D.timedelta(seconds=5)).isoformat() and release['publication_deadline_utc']==stop.isoformat() and type(release['hard_deadline_monotonic_ns']) is int and release['hard_deadline_monotonic_ns']==int((m+release['whole_invocation_seconds']-5)*10**9),'Fresh original release clock and strict work cleanup publication deadlines')
    ck(exact(release['outer_timeout'],candidate['outer_timeout']|{'sha256':b['environment_expected_sha256']['/usr/bin/timeout'],'version':runtime['timeout_version']}),'Original qualified outer timeout')
    ck(exact(release['inputs_sha256'],snap['inputs_sha256']) and exact(release['protected_tree_entries'],snap['protected_tree_entries']) and set(release['protected_tree_entries'])==set(spec['protected_tree_roots']),'Original entire stopped snapshot bound to release')
    tree_files={str(PurePosixPath(q)/r[0]) for q,rows in release['protected_tree_entries'].items() for r in rows if r[1]=='file'}
    ck(set(release['inputs_sha256'])==set(spec['inputs_sha256'])|tree_files and all(h is None or release['inputs_sha256'][p]==h for p,h in spec['inputs_sha256'].items()),'Full fixed source and stopped-tree input map')
    command=[release['inputs_sha256'][x['sha256_of_input']] if type(x) is dict else x for x in spec['argv']]
    ck(exact(release['command'],command),'Exact frozen worker argument list')
    argv=[PY,'-I','-S','-B',RP+'/launch_stopped_analysis_cpu_v1.py','--hard-deadline-monotonic-ns',str(release['hard_deadline_monotonic_ns']),'--release',A+'/controls/'+op+'.cpu_release.json','--release-sha256',release_sha]
    ck(exact(launch['argv'],argv) and exact(launch['command'],command) and launch['runtime_ready_sha256']==sha(S/(role+'.runtime_ready.json')),'Original bootstrap launch binds fresh qualified runtime')
    ext_command=doc(S/(op+'.command.json'));ck(ext_command['argv']==['ssh','-T','-F',str(P/'ssh_config'),{'A':'yellow-worm-77.coder','B':'aquamarine-toad-75.coder'}[role],shlex.join(argv)] and ext_command['stop_utc']==stop.isoformat(),'Exact original execution transport')
    stage,stage_ext=probe(op+'.stage_cpu','stage_runtime',historical);ck(exact(stage['runtime_sha256'],b['environment_expected_sha256']),'CPU control staged against original runtime')
    ck(dt(snap_ext['observed_utc'])<=dt(snap['root_reference_utc'])<=dt(release['issued_utc'])<=dt(stage_ext['started_utc'])<=dt(stage_ext['observed_utc'])<=dt(ext['started_utc']) and 0<=(dt(release['issued_utc'])-sample).total_seconds()<=300,'Snapshot release stage launch chronology')
    if op.startswith('sand_collect_') or op=='sand_summarize':
        core_stage,core_ext=probe(op+'.stage_core','stage_runtime',historical);ck(dt(core_ext['observed_utc'])<=dt(snap_ext['started_utc']),'Root scalar control precedes snapshot')
    if op.startswith('sand_collect_'):
        inv,inv_ext=probe(op+'.inventory','snapshot',historical);q=cfg['queue_root'];ck(exact(inv['queue_inventories'][q],snap['queue_inventories'][q]) and dt(inv_ext['observed_utc'])<=dt(core_ext['started_utc']),'Two identical whole stopped-tree inventories bracket fresh collection release')
        core=doc(S/(role+'.collection_release.json'));candidate_core=doc(K/'candidates'/(role+'.collection_release.candidate.json'));dynamic={'issued_by','status','local_queue_inventory_sha256','queue_release_sha256'}
        ck(set(core)==set(candidate_core) and all(exact(core[k],v) for k,v in candidate_core.items() if k not in dynamic) and core['issued_by']=='root' and core['status']=='approved_for_stopped_scalar_collection','Fixed collection contract unchanged')
        inventory=snap['queue_inventories'][q];inventory_sha=hashlib.sha256((json.dumps(inventory,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()
        ck(core['local_queue_inventory_sha256']==inventory_sha and core['queue_release_sha256']==inventory['files']['release_snapshot.json']['sha256'],'Collection release binds exact original inventory and queue release')
    if op.startswith('sand_saved_'):
        collect_snap=doc(S/('sand_collect_'+role+'.snapshot.json'));ck(exact(collect_snap['queue_inventories'][cfg['queue_root']],snap['queue_inventories'][cfg['queue_root']]),'Saved audit uses identical original stopped queue')
    if spec['protected_tree_roots']:ck(all(release['inputs_sha256'][p]==h for p,h in closure['closure_metadata_sha256'].items()),'Original authoritative closure metadata unchanged')
    document_paths={}
    for blob in snap['documents']:
        path=blob['path'];ck(path.startswith(R+'/') and path not in document_paths and (path.endswith('/protocol.json') or PurePosixPath(path).name in ('release_snapshot.json','queue_status.json','coverage_ledger.json','process_outcomes.json','gpu_observations.json')),'Only original transported scalar metadata')
        raw=base64.b64decode(blob['base64'],validate=True);local=S/(op+'.snapshot.documents')/path[len(R)+1:];pin(local,blob['sha256']);ck(type(blob['bytes']) is int and len(raw)==blob['bytes'] and hashlib.sha256(raw).hexdigest()==blob['sha256']==release['inputs_sha256'][path] and local.read_bytes()==raw,'Snapshot scalar blob and local copy bind exact source bytes')
        document_paths[path]=local
    if op.startswith('sand_collect_'):ck({cfg['queue_root']+'/'+n for n in ('release_snapshot.json','queue_status.json','coverage_ledger.json','process_outcomes.json')}<=set(document_paths),'Required original queue metadata transported')
    ck(type(exit_record['original_tool_session']) is int and exit_record['original_tool_session']==session and type(exit_record['original_tool_exit_code']) is int and exit_record['original_tool_exit_code']==0 and exit_record['recorded_by']=='root' and exit_record['original_transport_sha256']==sha(S/(op+'.external.json')),'Original root-attested tool exit0 and transport bytes')
    raw_exit=doc(exit_record['evidence_path'],exit_record['evidence_sha256'])
    if 'tool_response' in raw_exit:
        ck(raw_exit.get('original_session')==session,'Original tool envelope binds session');tool=raw_exit['tool_response']
    elif 'exit' in raw_exit and 'launch' in raw_exit:
        ck(raw_exit['launch'].get('session_id')==session,'Original launch envelope binds session');tool=raw_exit['exit']
    elif 'original_launch' in raw_exit and 'original_completion' in raw_exit:
        ck(raw_exit['original_launch'].get('session_id')==session and dt(raw_exit['root_observed_utc'])<=dt(exit_record['observed_utc']),'Original launch and completion envelope binds session and root observation');tool=raw_exit['original_completion']
    elif 'launch' in raw_exit and 'completion' in raw_exit:
        ck(raw_exit['launch'].get('session_id')==session,'Original launch and completion envelope binds session');tool=raw_exit['completion']
    else:
        tool=raw_exit
        ck(('session_id' in tool and tool['session_id']==session) or ('session_id' not in tool and set(tool)<= {'chunk_id','wall_time_seconds','exit_code','original_token_count','output'} and type(tool.get('chunk_id')) is str and type(tool.get('output')) is str),'Direct completed raw tool envelope bound to root-attested original session')
    ck(type(tool.get('exit_code')) is int and tool['exit_code']==0 and ('session_id' not in tool or tool['session_id']==session),'Genuine completed original tool exit0')
    capture_tag=op+'.capture'
    if op=='sand_saved_A':
        recovery=doc(S/'sand_saved_A.recovery_closure_receipt.json')
        ck(recovery['phase_sha256']==PH and recovery['accepted_root_review_sha256']==root_sha and recovery['actual_capture_tag']=='sand_saved_A.capture_recovery1' and recovery['original_worker_session']==66211 and recovery['original_failed_capture_preserved'] is True and recovery['new_clock_granted'] is False,'Reviewed recovery remaps only original saved_A capture; original worker unchanged')
        source=doc(P/'sand_scalar_recovery_UNADMITTED_code_audit_v1/manifest.json','71c50f558082d127757cc84dff97f7451c7c5e73ff4b5ef48ff41be1af851a79')
        for n,h in source['files_sha256'].items():pin(P/'sand_scalar_recovery_UNADMITTED_code_audit_v1'/n,h)
        pin(P/'sand_scalar_recovery_independent_transition_v1.json','e8e4085ffb2847187928e6c716b3a24bee5bce2c3ea901b304adc72b81efe593')
        pin(P/'sand_recovery_fate_independent_review_code_audit_v1.json','31f759046d1af33cf5123a2771d6b8ce921730ba3e18e1ed096c2cafaf4a8480')
        failures=doc(P/'sand_scalar_recovery_UNADMITTED_code_audit_v1/failure_bindings.json')
        for p,h in failures['files_sha256'].items():pin(p,h)
        failed=doc(S/'sand_saved_A.capture.external.json');ck(failed['exit_code']==255 and failed['local_transport_reaped'] is True,'Original genuine failed capture remains retained as255')
        capture_tag=recovery['actual_capture_tag'];pin(S/(capture_tag+'.json'),recovery['capture_sha256'])
    if op=='sand_summarize':
        recovery=doc(S/'sand_summarize.recovery_closure_receipt.json')
        ck(recovery['phase_sha256']==PH and recovery['accepted_root_review_sha256']==root_sha and recovery['actual_capture_tag']=='sand_summarize.capture_recovery1' and recovery['original_worker_session']==19303 and recovery['original_failed_capture_preserved'] is True and recovery['new_clock_granted'] is False,'Reviewed recovery remaps only original summary capture; original worker unchanged')
        package=P/'sand_summary_capture_recovery_UNADMITTED_code_audit_v1';source=doc(package/'manifest.json','2b8b0b9a8a0a025533d6291f401f1e99fa6be9cb70b8e7b0bb27029f00fc429c')
        for n,h in source['files_sha256'].items():pin(package/n,h)
        pin(P/'sand_summary_capture_recovery_independent_transition_v1.json','6858dc1d3ea1fbc7871cb9f02a761411029506aabf7f653ba0a90bc65935e27a')
        ck(recovery['source_manifest_sha256']==sha(package/'manifest.json') and recovery['source_review_sha256']=='6858dc1d3ea1fbc7871cb9f02a761411029506aabf7f653ba0a90bc65935e27a','Summary recovery exact independently reviewed source bindings')
        failures=doc(package/'failure_bindings.json')
        for p,h in failures['files_sha256'].items():pin(p,h)
        failed=doc(S/'sand_summarize.capture.external.json');ck(failed['exit_code']==255 and failed['local_transport_reaped'] is True and (S/'sand_summarize.capture.stdout').read_bytes()==b'' and 'operation not permitted' in (S/'sand_summarize.capture.stderr').read_text(),'Original sandbox-blocked capture remains genuine255 with unchanged evidence')
        capture_tag=recovery['actual_capture_tag'];pin(S/(capture_tag+'.json'),recovery['capture_sha256'])
    cap,cap_ext=probe(capture_tag,'capture');ck(cap['final_input_and_output_hashes_verified'] is True and exact(cap['native_absent'],root['native_absent']),'Final input/output rehash and original native closure bound by root')
    owner_dir=release['owner_output_dir'];names=('owner_started.json','child_registered.json','child.stdout','child.stderr','owner_terminal.json');allowed={owner_dir+'/'+n for n in names}|set(spec['outputs']);paths={}
    for blob in cap['files']:
        p=blob['path'];ck(p in allowed and p not in paths,'Exact original owner/product blob membership');raw=base64.b64decode(blob['base64'],validate=True);local=S/(op+'.collected')/p[len(R)+1:]
        pin(local,blob['sha256']);ck(type(blob['bytes']) is int and len(raw)==blob['bytes'] and hashlib.sha256(raw).hexdigest()==blob['sha256'] and local.read_bytes()==raw,'Exact original captured blob hash size and local bytes');paths[p]=local
    ck(set(paths)==allowed,'All original owner and scalar product blobs present')
    started=doc(paths[owner_dir+'/owner_started.json']);registered=doc(paths[owner_dir+'/child_registered.json']);terminal=doc(paths[owner_dir+'/owner_terminal.json']);product_path=paths[spec['outputs'][0]];product=doc(product_path,root['product_sha256']);product_sha=sha(product_path)
    ck(Path(root['local_product']).resolve()==product_path.resolve(),'Root semantic review names exact transported product')
    child=terminal['child'];owner=terminal['owner_identity'];outer=terminal['outer_timeout_identity']
    ck(terminal['schema']=='adaptgns_stopped_analysis_cpu_terminal_v1' and terminal['status']=='complete' and terminal['failure'] is None and terminal['release_sha256']==release_sha and terminal['child_native_absent'] is True and terminal['scientific_admission'] is False and terminal['root_original_tool_exit_and_timeout_owner_child_native_closure_required'] is True,'Complete original bounded owner terminal')
    ck(child['reaped'] is True and type(child['exit_code']) is int and child['exit_code']==0 and child['signals']==child['cleanup_errors']==[] and exact(child['command'],command),'Original frozen worker reaped without signals or errors')
    expected=release['inputs_sha256']|b['environment_expected_sha256']|{A+'/controls/'+op+'.cpu_release.json':release_sha}|{A+'/controls/'+r+'.evaluation_closure.json':sha(S/(r+'.evaluation_closure.json')) for r in MODELS}|{release[k]['path']:release[k]['sha256'] for k in ('operation_spec','analysis_phase','owner_source','bootstrap_source')}
    ck(exact(terminal['input_sha256'],expected) and terminal['output_sha256']=={spec['outputs'][0]:product_sha},'Complete original terminal runtime input and output maps')
    ck(set(terminal['evidence_sha256'])=={owner_dir+'/'+n for n in names if n!='owner_terminal.json'},'Exact original terminal evidence map')
    for p,h in terminal['evidence_sha256'].items():pin(paths[p],h)
    ck(exact(started['owner_identity'],owner) and exact(started['outer_timeout_identity'],outer) and started['release_sha256']==release_sha and exact(started['original_clock_sample'],release['clock_sample']) and started['scientific_admission'] is False,'Original owner-start identities release and clock')
    ck(exact(registered['identity'],child['identity']) and exact(registered['owner_identity'],owner) and exact(registered['command'],command) and registered['registered'] is True and registered['reaped'] is False and registered['exit_code'] is None and registered['signals']==registered['cleanup_errors']==[],'Original child registration precedes reap')
    for k in ('pid','started_utc','started_monotonic','registered'):ck(exact(registered[k],child[k]),'Child registration retained '+k)
    for ident in (outer,owner,child['identity']):ck(all(type(ident[k]) is int and ident[k]>0 for k in ('pid','ppid','pgid','sid')) and str(ident['start_id']).isdigit() and int(ident['start_id'])>0,'Full original native identity')
    ck(len({outer['pid'],owner['pid'],child['pid']})==3 and child['pid']==child['identity']['pid'] and owner['ppid']==outer['pid'] and child['identity']['ppid']==owner['pid'] and owner['pgid']==outer['pgid']==child['identity']['pgid'] and owner['sid']==outer['sid']==child['identity']['sid'] and owner['executable']==child['identity']['executable']=='/usr/bin/python3.12' and outer['executable']=='/usr/bin/timeout','Native timeout owner child linkage')
    guard=terminal['native_hard_guard'];timing=terminal['native_outer_timing'];hard=release['hard_deadline_monotonic_ns'];computed=timing['computed_monotonic_ns'];seconds=timing['term_seconds']
    ck(exact(started['native_hard_guard'],guard) and exact(started['native_outer_timing'],timing) and guard['signal']=='SIGKILL' and guard['clock']=='CLOCK_MONOTONIC' and guard['deleted_before_exit'] is False and guard['absolute_deadline_ns']==hard and type(computed) is int and type(seconds) is int and seconds>0 and seconds==(hard-computed)//10**9-20 and 0<=guard['armed_monotonic_ns']-computed<=5*10**9 and timing['owner_guard_armed_monotonic_ns']==guard['armed_monotonic_ns'] and timing['original_absolute_hard_deadline_ns']==hard and timing['bootstrap_allowance_seconds']==5 and timing['earliest_relative_kill_monotonic_ns']==computed+(seconds+15)*10**9,'Original native hard and relative timer bounds')
    outer_argv=['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(seconds)+'s',PY,'-I','-S','-B',release['owner_source']['path'],'--execute','--hard-deadline-monotonic-ns',str(hard),'--outer-term-seconds',str(seconds),'--outer-computed-monotonic-ns',str(computed),'--release',A+'/controls/'+op+'.cpu_release.json','--release-sha256',release_sha]
    ck(exact(outer['argv'],outer_argv) and exact(owner['argv'],outer_argv[4:]) and exact(child['identity']['argv'],command),'Exact native timeout owner and worker argv')
    ck(finite(started['entry_monotonic']) and 0<=started['entry_monotonic']-m<=300 and started['entry_monotonic']<=child['started_monotonic']<=child['reaped_monotonic']<=terminal['publication_monotonic'] and terminal['publication_monotonic']*10**9<min(hard,timing['earliest_relative_kill_monotonic_ns']),'Native worker reap and publication inside original bounds')
    ck(set(cap['native_absent'])==historical|{str(outer['pid']),str(owner['pid']),str(child['pid'])} and all(v is True for v in cap['native_absent'].values()),'Complete original historical and timeout-owner-child native closure')
    published=dt(terminal['publication_utc'])
    ck(start<=dt(release['issued_utc'])<=dt(ext['started_utc'])<=dt(ext['observed_utc'])<=dt(exit_record['observed_utc'])<=dt(cap_ext['started_utc'])<=dt(cap_ext['observed_utc'])<=root_time<stop and sample<=published<stop and dt(ext['started_utc'])-D.timedelta(seconds=5)<=dt(child['started_utc'])<=dt(child['reaped_utc'])<=published<=dt(ext['observed_utc'])+D.timedelta(seconds=5),'Original launch exit capture and publication chronology with only cross-host tolerance')
    def coverage(collection):
        r=collection['host_role'];ck(r in MODELS and collection['schema']=='adaptgns_sand_evaluation_collection_scoped_v1' and collection['status']=='stopped_outputs_collected','Original Sand collection schema')
        rows=collection['stages'];grid={(x['arm'],x['seed'],x['stage']):x for x in rows};wanted={(a,s,n) for a,s in MODELS[r] for n in STAGES};ck(len(rows)==len(grid)==len(wanted) and set(grid)==wanted,'All original role model stages retained')
        counts=Counter()
        for (arm,seed,name),stage in grid.items():
            ck(type(seed) is int and exact([stage['mode'],stage['split']],list(STAGES[name])),'Original model mode split identity')
            units=[(i,p) for i in range(30) for p in POLICIES] if name=='full_rollout_test' else ([(n//314,n%314+6) for n in (i*9419//127 for i in range(128))] if name=='clean_validation' else [(i,t) for i in range(30) for t in (7,85,163,241,319)])
            cells=stage['cells'];ck(len(cells)==len(units),'Full original stage denominator')
            for cell,unit in zip(cells,units):
                ck(type(cell['source_index']) is int and (name=='full_rollout_test' or type(cell['target_frame']) is int) and (cell['source_index'],cell['policy' if name=='full_rollout_test' else 'target_frame'])==unit and cell['state'] in STATES,'Original Sand T320/H314 cell identity and state');counts[cell['state']]+=1
        ck(sum(counts.values())==608*len(MODELS[r]),'Full original role cell count');return counts,len(rows)
    collections={role:product} if op.startswith('sand_collect_') else {p[-1]:v for p,v in prior_products.items() if p.startswith('sand_collect_')}
    counts=Counter();stages_count=0
    for c in collections.values():n,st=coverage(c);counts.update(n);stages_count+=st
    semantic=root['semantic_review'];ck(semantic['status']=='validated_product_metadata' and semantic['schema']==product['schema'] and type(semantic['checks']) is int and semantic['checks']>0 and semantic['required_model_stages']==stages_count and semantic['required_cells']==sum(counts.values()) and exact(semantic['coverage'],dict(counts)) and semantic['all_required_cells_completed'] is (set(counts)=={'completed_required_outcome'}) and semantic['scientific_admission'] is False and semantic['statistics_recomputed'] is False and semantic['arrays_opened'] is False,'Byte-bound root semantic review agrees with independently counted original denominator')
    if op.startswith('sand_collect_'):
        ck(exact(product['output_tree_state'],snap['queue_inventories'][cfg['queue_root']]) and product['collection_release_sha256']==sha(S/(role+'.collection_release.json')) and semantic['eligible_for_aggregation'] is product['queue_status']['all_pinned_inputs_reverified'],'Collection original tree and integrity eligibility retained')
        for name,key in [('release_snapshot.json','queue_release'),('queue_status.json','queue_status'),('process_outcomes.json','process_outcomes')]:ck(exact(product[key],doc(document_paths[cfg['queue_root']+'/'+name])),'Original stopped metadata preserved '+name)
    if op in ('sand_summarize','sand_paired'):
        transfer=doc(S/'B.scalars_transferred_to_A.json');ck(transfer['analysis_phase_sha256']==PH,'B scalar transfer stays in original phase')
        for parent in ('sand_collect_B','sand_saved_B'):
            receipt=doc(S/(parent+'.review.json'));ck(transfer['original_B_review_sha256'][parent]==sha(S/(parent+'.review.json')) and transfer['transferred_sha256'][specs[parent]['outputs'][0]]==receipt['product_sha256'],'Exact accepted B scalar transfer '+parent)
    for p,h in list(evidence.items()):ck(sha(p)==h,'Evidence stable before publication '+Path(p).name)
    return {'schema':'sand_completed_product_independent_code_audit_v1','status':'passed_completed_original_Sand_product_evidence','operation':op,'original_tool_session':session,'analysis_phase_sha256':PH,'accepted_root_review_sha256':root_sha,'root_review_checked_utc':root['checked_utc'],'product_sha256':product_sha,'coverage':dict(counts),'required_model_stages':stages_count,'required_cells':sum(counts.values()),'native_absent_count':len(cap['native_absent']),'checks':checks,'check_count':len(checks),'evidence_sha256':evidence,'semantic_review_rerun':False,'statistics_recomputed':False,'arrays_or_models_opened':False,'clocks_read_or_issued':False,'live_or_remote_action':False,'scientific_admission':False}
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--operation',choices=OPS,required=True);p.add_argument('--original-session',type=int,required=True);p.add_argument('--root-review-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.operation,a.original_session,a.root_review_sha256)
    with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'checks':result['check_count'],'receipt_sha256':sha(a.output),'coverage':result['coverage'],'required_cells':result['required_cells']}))
if __name__=='__main__':main()
