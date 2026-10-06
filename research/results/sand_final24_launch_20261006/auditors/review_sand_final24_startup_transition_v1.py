#!/usr/bin/env python3
"""Independent local startup audit; no remote calls or scientific file reads.

Root invokes only after both original owners are observed and control closes.
The accepted issuance review is byte-bound, never rerun. Synthetic callers
supply checked_at and monotonic; importing this module reads no live clock.
"""
from pathlib import Path
import argparse, datetime as D, hashlib, json, math, re, shlex, time

P=Path(__file__).resolve().parent
PK=P/'sand_final24_operator_UNADMITTED_transition_v1'
PACKAGE_SHA='6898c6d6dcaf0421b6b64c37de9e4d5505e059e8cd2311a81bb203ab897ef942'
PREPARED_SHA='dfd1cb9c210196abd87dafc2d7164487e0aba18586e34eb098d92f85acfdf604'
ISSUER_SHA='c2540e6bdbb202de8c8f4685661fd606ea6401cf426d89bcd82fa668469b904f'
R='/root/repos/AdaptGNS-cuda-20261006'
PYTHON=R+'/.venv/bin/python'
TIMEOUT_SHA='2db30bc57746c940a0643581cd5e2671e505442bea16164143f0ea8ff4e36453'
ENV={'CUBLAS_WORKSPACE_CONFIG':':4096:8','LD_LIBRARY_PATH':'/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
ALIASES={'A':'yellow-worm-77.coder','B':'aquamarine-toad-75.coder'}
GLOBAL_STOP='2026-10-07T04:00:00+00:00'

def sha(path):
    path=Path(path)
    if path.suffix in ('.npy','.npz','.pt','.pth'):
        raise ValueError('Scientific files are outside this local audit')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    def pairs(rows):
        out={}
        for key,value in rows:
            if key in out:raise ValueError('Duplicate JSON key')
            out[key]=value
        return out
    def reject(value):raise ValueError('Nonfinite JSON')
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=reject)

def exact(a,b):
    return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)

def dt(value):
    result=D.datetime.fromisoformat(value)
    if result.tzinfo is None:raise ValueError('Aware timestamp required')
    return result

def now():return D.datetime.now(D.timezone.utc)

def audit(state,issue_review,issue_pin,checked_at=None,monotonic=None):
    checked_at=now() if checked_at is None else checked_at
    monotonic=time.monotonic() if monotonic is None else monotonic
    state=Path(state);checks=[];evidence={}
    def check(value,message):
        if not value:raise ValueError(message)
        checks.append(message)
    def pin(path,expected=None):
        path=Path(path)
        check(path.is_file() and not path.is_symlink(),'Ordinary local evidence: '+str(path))
        result=sha(path);evidence[str(path.resolve())]=result
        if expected is not None:check(result==expected,'Exact evidence bytes: '+str(path))
        return result
    def doc(path,expected=None):pin(path,expected);return read(path)
    package=doc(PK/'manifest.json',PACKAGE_SHA)
    for name,h in package['files_sha256'].items():pin(PK/name,h)
    pin(P/'review_actual_sand_final24_issuance_code_audit_v1.py',ISSUER_SHA)
    prepared=doc(state/'prepared_inputs.json',PREPARED_SHA)
    binding=doc(state/'operator_binding.json')
    check(binding['manifest_sha256']==PACKAGE_SHA,'Original operator binding')
    phase=doc(state/'control_phase.json');phase_pin=sha(state/'control_phase.json')
    anchor=doc(state/'control_anchor.json');start=dt(phase['started_utc']);stop=dt(phase['stop_utc'])
    check(phase['status']=='approved_one_original_final24_control_phase' and phase['issued_by']=='root'
          and phase['clock_restarted'] is False and exact([phase['control_seconds'],phase['evaluation_seconds'],phase['analysis_seconds']],[600,11760,3600])
          and (stop-start).total_seconds()==600 and phase['prepared_inputs_sha256']==PREPARED_SHA,'Same original full600phase and unchanged downstream allocation')
    check(anchor['phase_sha256']==phase_pin and anchor['utc']==start.isoformat()
          and type(anchor['monotonic_seconds']) in (int,float) and math.isfinite(anchor['monotonic_seconds'])
          and monotonic>=anchor['monotonic_seconds'] and abs((checked_at-start).total_seconds()-(monotonic-anchor['monotonic_seconds']))<=5
          and start<=checked_at and checked_at+D.timedelta(seconds=5)<stop and monotonic+5<anchor['monotonic_seconds']+600,'Original UTC/monotonic phase remains open for startup review')
    check(stop+D.timedelta(seconds=11760+3600+15)<=dt(GLOBAL_STOP),'Full original evaluation and analysis fit global stop')
    issue=doc(issue_review,issue_pin)
    check(issue['status']=='passed_actual_final24_issuance' and issue['control_phase_sha256']==phase_pin
          and issue['prepared_inputs_sha256']==PREPARED_SHA and issue['operator_manifest_sha256']==PACKAGE_SHA
          and issue['review_source_sha256']==ISSUER_SHA and set(issue['release_sha256'])=={'A','B'},'Accepted exact both-role issuance review bound without rerunning it')
    check(start<=dt(issue['checked_utc'])<=dt(issue['completed_utc'])<=checked_at,'Accepted issuance review completed in original phase before startup review')
    for path,h in issue['evidence_sha256'].items():pin(path,h)
    completion=doc(state/'control_completion.json')
    check(completion['status']=='both_original_owners_observed_inside_same_control_window'
          and completion['control_phase_sha256']==phase_pin and completion['scientific_completion'] is False
          and start<=dt(completion['checked_utc'])<=checked_at,'Both owner observations completed within original control phase')
    roles={};sessions=set()
    for role,spec in prepared['roles'].items():
        check(role in ('A','B'),'Exact role')
        release=doc(state/(role+'.evaluation_release.json'),issue['release_sha256'][role])
        launch=doc(state/(role+'.launch.json'));staged=doc(state/(role+'.staged.json'))
        check(staged['release_sha256']==issue['release_sha256'][role]
              and staged['independent_issue_review']=={'path':str(Path(issue_review).resolve()),'sha256':issue_pin},role+': staged release uses accepted issuance review')
        stage_capture=doc(state/(role+'.stage_release.json'),staged['capture_sha256'])
        stage_external=doc(state/(role+'.stage_release.external.json'))
        check(stage_capture['action']=='stage_release' and stage_capture['role']==role
              and stage_capture['release_sha256']==issue['release_sha256'][role]
              and stage_capture['same_control_stop_utc']==stop.isoformat()
              and type(stage_external['exit_code']) is int and stage_external['exit_code']==0
              and stage_external['local_transport_reaped'] is True and stage_external['local_transport_timeout'] is False
              and stage_external['signals_to_own_local_group']==[] and stage_external['failure'] is None,role+': exact successful reviewed release staging')
        release_path=spec['owner_argv'][spec['owner_argv'].index('--release')+1]
        shell='test "$(sha256sum -- '+shlex.quote(release_path)+')" = '+shlex.quote(issue['release_sha256'][role]+'  '+release_path)+' && exec '+shlex.join(['/usr/bin/env','-u','CUDA_VISIBLE_DEVICES']+[k+'='+v for k,v in ENV.items()]+spec['owner_argv'])
        check(launch['exact_owner_argv']==spec['owner_argv'] and launch['remote_launch_shell']==shell
              and launch['release_sha256']==issue['release_sha256'][role] and launch['control_phase_sha256']==phase_pin
              and launch['original_tool_session_must_be_recorded'] is True,role+': exact frozen original owner launch and release SHA gate')
        command=doc(state/(role+'.evaluation.command.json'))
        check(command['argv']==['ssh','-T','-F',str(P/'ssh_config'),ALIASES[role],shell]
              and dt(command['stop_utc'])==dt(GLOBAL_STOP)-D.timedelta(seconds=3600)
              and command['remote_closure_separately_required'] is True,role+': exact original evaluation transport and analysis reserve')
        session=doc(state/(role+'.original_session.json'));session_id=session['original_tool_session']
        check(type(session_id) is int and session_id>0 and session_id not in sessions and session['recorded_by']=='root'
              and session['launch_sha256']==sha(state/(role+'.launch.json')),role+': unique original root tool session bound to launch')
        sessions.add(session_id);tool=doc(session['evidence_file'],session['evidence_sha256'])
        check(type(tool.get('session_id')) is int and tool['session_id']==session_id
              and tool.get('exit_code') is None,role+': actual original execution-tool response shows running session')
        capture_path=state/(role+'.observe_owner.json');capture=doc(capture_path)
        observed=doc(state/(role+'.owner_observed.json'),completion['owner_observation_sha256'][role])
        check(observed['status']=='observed_original_final24_owner' and observed['control_phase_sha256']==phase_pin
              and observed['release_sha256']==issue['release_sha256'][role] and type(observed['original_tool_session']) is int
              and observed['original_tool_session']==session_id and observed['native_capture_sha256']==sha(capture_path)
              and observed['scientific_completion'] is False,role+': original session and exact native capture bound by owner receipt')
        external=doc(state/(role+'.observe_owner.external.json'));probe_command=doc(state/(role+'.observe_owner.command.json'))
        raw_capture=doc(state/(role+'.observe_owner.stdout'))
        expected_raw=dict(capture);expected_raw.pop('root_reference_utc')
        check(exact(raw_capture,expected_raw),role+': native capture exactly matches original helper stdout')
        pin(state/(role+'.observe_owner.stderr'))
        check((state/(role+'.observe_owner.stderr')).read_bytes()==b'',role+': observation stderr empty')
        probe_argv=probe_command['argv'];remote_argv=shlex.split(probe_argv[-1])
        check(probe_argv[:5]==['ssh','-T','-F',str(P/'ssh_config'),ALIASES[role]] and len(probe_argv)==6
              and len(remote_argv)==10 and remote_argv[:3]==['/usr/bin/timeout','--signal=TERM','--kill-after=5s']
              and re.fullmatch('[1-9][0-9]*s',remote_argv[3]) is not None and int(remote_argv[3][:-1])<=120
              and remote_argv[4:9]==[PYTHON,'-I','-S','-B','-c'] and remote_argv[9]==(PK/'remote_control.py').read_text(),role+': exact bounded frozen native helper command')
        check(external['exit_code']==0 and type(external['exit_code']) is int and external['local_transport_reaped'] is True
              and external['local_transport_timeout'] is False and external['signals_to_own_local_group']==[] and external['failure'] is None
              and probe_command['stop_utc']==stop.isoformat(),role+': bounded native-observation transport completed successfully')
        check(start<=dt(issue['completed_utc'])<=dt(stage_external['started_utc'])<=dt(stage_external['observed_utc'])
              <=dt(launch['dispatched_utc'])<=dt(session['observed_utc'])<=dt(external['started_utc'])
              <=dt(external['observed_utc'])<=dt(observed['checked_utc'])<=dt(completion['checked_utc'])<=checked_at,role+': original launch/session/native observation/completion chronology')
        check(capture['schema']=='sand_final24_remote_control_observation_v1' and capture['action']=='observe_owner'
              and capture['role']==role and capture['hostname']==spec['hostname'] and capture['same_control_stop_utc']==stop.isoformat()
              and capture['release_sha256']==issue['release_sha256'][role] and exact(capture['input_sha256'],spec['files_sha256']),role+': exact host release and complete immutable input map')
        clock=capture['clock'];sample=dt(clock['host_utc'])
        check(clock['host_boot_id']==spec['expected_boot_id'] and type(clock['host_monotonic_seconds']) in (int,float)
              and math.isfinite(clock['host_monotonic_seconds']) and 0<clock['host_monotonic_seconds']
              and dt(external['started_utc'])-D.timedelta(seconds=5)<=sample<=dt(external['observed_utc'])+D.timedelta(seconds=5)
              and abs((dt(capture['root_reference_utc'])-sample).total_seconds())<=5
              and 0<=(checked_at-sample).total_seconds()<=300 and start-D.timedelta(seconds=5)<=sample<stop,role+': fresh same-boot native/root clock within original600phase')
        pyenv=release['python_environment'];runtime=capture['runtime']
        check(runtime['hostname']==spec['hostname'] and runtime['machine']=='aarch64' and runtime['libc'][0]=='glibc' and bool(runtime['libc'][1])
              and all(exact(runtime['python_environment'][k],v) for k,v in pyenv.items() if k!='sys_prefix')
              and runtime['python_environment']['sys_prefix'] in ('/usr',R+'/.venv') and runtime['isolated'] is True
              and runtime['no_site'] is True and runtime['dont_write_bytecode'] is True and type(runtime['optimize']) is int and runtime['optimize']==0,role+': exact isolated helper runtime and Python identity')
        check(capture['runtime_sha256']=={pyenv['lexical_path']:pyenv['binary_sha256'],pyenv['resolved_binary_path']:pyenv['binary_sha256'],pyenv['pyvenv_config_path']:pyenv['pyvenv_config_sha256'],'/usr/bin/timeout':TIMEOUT_SHA}
              and capture['required_launch_environment']==ENV and capture['required_launch_variables_absent']==['CUDA_VISIBLE_DEVICES'],role+': exact native runtime bytes and source-enforced owner/child environment')
        check(exact(capture['native_absent'],{str(pid):True for pid in spec['historical_pids']}) and exact(capture['historical_absence'],capture['native_absent'])
              and capture['other_matching_science_processes']==[],role+': original owned processes absent and no other matching science')
        owner=capture['owner'];children=capture['children'];permitted=[c for stream in release['streams'] for c in stream['commands']]
        for row in [owner]+children:
            check(set(row)=={'pid','ppid','pgid','sid','start_ticks','executable','argv'}
                  and all(type(row[k]) is int and row[k]>0 for k in ('pid','ppid','pgid','sid','start_ticks'))
                  and row['executable']==pyenv['resolved_binary_path'],role+': exact positive native process identity')
        check(owner['argv']==spec['owner_argv'] and owner['pid'] not in spec['historical_pids'],role+': exact newly started frozen owner argv')
        check(len({row['pid'] for row in [owner]+children})==1+len(children) and len({tuple(row['argv']) for row in children})==len(children)
              and all(row['argv'] in permitted and row['ppid']==owner['pid'] and row['pid']==row['pgid']==row['sid'] for row in children),role+': only unique exact evaluator children with owner and native session identity')
        check(capture['gpu_uuids']==release['gpu_uuids'],role+': all four physical GPU UUIDs preserved')
        child_map={row['pid']:row for row in children};owned=release['gpu_scope']['owned_indices']
        for app in capture['gpu_processes']:
            check(type(app['pid']) is int and app['pid']>0 and app['gpu_uuid'] in capture['gpu_uuids'],role+': exact GPU application identity')
            index=capture['gpu_uuids'].index(app['gpu_uuid'])
            if app['pid'] in child_map:
                argv=child_map[app['pid']]['argv'];check(index==int(argv[argv.index('--cuda-index')+1]) and index in owned,role+': evaluator on its assigned physical GPU')
            else:check(index not in owned,role+': no foreign compute on assigned devices')
        check(capture['scientific_execution_performed'] is False and capture['scientific_admission'] is False,role+': observation makes no scientific completion claim')
        for suffix in ('evaluation.external.json','evaluation.stdout','evaluation.stderr','original_exit.json'):
            check(not (state/(role+'.'+suffix)).exists(),role+': original evaluation transport/exit not yet completed: '+suffix)
        roles[role]={'original_tool_session':session_id,'owner':owner,'children':children,'release_sha256':issue['release_sha256'][role],'native_capture_sha256':sha(capture_path)}
    check(set(roles)=={'A','B'},'Both original owners observed')
    return {'schema':'sand_final24_independent_startup_review_v1','status':'passed_both_original_sessions_and_native_owner_startup','reviewer':'transition_review','checked_utc':checked_at.isoformat(),'operator_manifest_sha256':PACKAGE_SHA,'prepared_inputs_sha256':PREPARED_SHA,'control_phase_sha256':phase_pin,'accepted_issuance_review_sha256':issue_pin,'roles':roles,'checks':checks,'evidence_sha256':evidence,'scientific_completion':False,'all24_stage_completion_asserted':False,'eventual_original_tool_exits_and_stopped_native_closure_still_required':True,'review_source_sha256':sha(__file__)}

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--state',type=Path,required=True);parser.add_argument('--issue-review',type=Path,required=True)
    parser.add_argument('--issue-review-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.state.resolve(),args.issue_review.resolve(),args.issue_review_sha256)
    for path,pin in result['evidence_sha256'].items():
        if sha(path)!=pin:raise ValueError('Evidence changed before receipt publication: '+path)
    phase=read(args.state/'control_phase.json');anchor=read(args.state/'control_anchor.json');finished=now()
    if not(finished+D.timedelta(seconds=5)<dt(phase['stop_utc']) and time.monotonic()+5<anchor['monotonic_seconds']+600):raise ValueError('Original control stop exhausted before publication')
    for role in ('A','B'):
        for suffix in ('evaluation.external.json','evaluation.stdout','evaluation.stderr','original_exit.json'):
            if (args.state/(role+'.'+suffix)).exists():raise ValueError('Original evaluation completed during startup review')
    result['completed_utc']=finished.isoformat()
    with args.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'checks':len(result['checks']),'receipt_sha256':sha(args.output)}))

if __name__=='__main__':main()
