#!/usr/bin/env python3
"""Root-only Sand final24 resolution/staging/launch in one600-second control phase.

Describe/help is inert. Prepare is local and clock-free. Begin-control is the
only control-phase issuer; every original launch is separately observed. Frozen
owner and scientific argv, full cohort, quotas and global stop are unchanged.
"""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,math,os,shlex,signal,subprocess,sys,time
PK=Path(__file__).resolve().parent;P=PK.parent
R='/root/repos/AdaptGNS-cuda-20261006';PYTHON=R+'/.venv/bin/python'
CTRL=R+'/sand_final_evaluation_controls_20261006_v1'
GLOBAL_STOP='2026-10-07T04:00:00+00:00'
ALIASES={'A':'yellow-worm-77.coder','B':'aquamarine-toad-75.coder'}
ENV={'CUBLAS_WORKSPACE_CONFIG':':4096:8','LD_LIBRARY_PATH':'/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
SSH_SHA='f04af81d2b926b04f7a8063df5c8535da1d8c9aa637e15cf235c261d6d0725b2'
def need(x,s):
 if not x:raise ValueError(s)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return type(x) is str and len(x)==64 and set(x)<=set('0123456789abcdef')
def now():return datetime.datetime.now(datetime.timezone.utc)
def dt(s):
 v=datetime.datetime.fromisoformat(s);need(v.tzinfo is not None,'Timezone required');return v
def put(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def package_check(pin):
 need(not sys.flags.optimize and digest(pin) and sha(PK/'manifest.json')==pin,'Exact reviewed operator package required')
 for p,h in read(PK/'manifest.json')['files_sha256'].items():need(sha(PK/p)==h,'Operator payload changed: '+p)
 need(sha(P/'ssh_config')==SSH_SHA,'Reviewed SSH transport changed')
def prepared(state):
 v=read(state/'prepared_inputs.json');need(sha(state/'prepared_inputs.json')==sha(PK/'prepared_inputs.json'),'Prepared inputs changed')
 for p,h in v['original_local_evidence_sha256'].items():need(sha(p)==h,'Original reviewed scalar evidence changed: '+p)
 return v
def phase(state,open_required=True,tail=0):
 p=read(state/'control_phase.json');anchor=read(state/'control_anchor.json')
 need(anchor['phase_sha256']==sha(state/'control_phase.json') and p['status']=='approved_one_original_final24_control_phase' and p['issued_by']=='root','Original control phase required')
 start=dt(p['started_utc']);stop=dt(p['stop_utc']);elapsed=time.monotonic()-anchor['monotonic_seconds']
 need((stop-start).total_seconds()==600 and p['control_seconds']==600 and p['evaluation_seconds']==11760 and p['analysis_seconds']==3600 and p['clock_restarted'] is False,'Fixed600+11760+3600 required')
 need(start.isoformat()==anchor['utc'] and elapsed>=0 and abs((now()-start).total_seconds()-elapsed)<=5,'Original UTC/monotonic control anchor changed')
 need(stop+datetime.timedelta(seconds=11760+3600+15)<=dt(GLOBAL_STOP),'Full unchanged downstream allocation does not fit')
 mono_stop=anchor['monotonic_seconds']+600
 if open_required:need(min((stop-now()).total_seconds(),mono_stop-time.monotonic())>10+tail,'Original control window exhausted')
 return p,stop,mono_stop
def bounded_local(state,argv,tag,stop,mono_stop,stdin=None,limit=None,require_zero=True):
 left=lambda:min((stop-now()).total_seconds(),mono_stop-time.monotonic())
 timeout=left()-12
 if limit is not None:timeout=min(timeout,limit)
 need(timeout>0,'No original bounded transport time remains')
 put(state/(tag+'.command.json'),{'argv':argv,'timeout_seconds':timeout,'stop_utc':stop.isoformat(),'remote_closure_separately_required':True})
 started=now();tick=time.monotonic();proc=subprocess.Popen(argv,stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 out='';err='';signals=[];timed=False;failure=None
 def send(sig):
  try:os.killpg(proc.pid,sig);signals.append(signal.Signals(sig).name)
  except ProcessLookupError:pass
 def tail():return max(0,min(3,left()-5))
 try:
  try:out,err=proc.communicate(input=stdin,timeout=timeout)
  except subprocess.TimeoutExpired:
   timed=True;send(signal.SIGTERM)
   try:out,err=proc.communicate(timeout=tail())
   except subprocess.TimeoutExpired:send(signal.SIGKILL);out,err=proc.communicate(timeout=tail())
 except BaseException as e:
  failure=type(e).__name__+': '+str(e)
  if proc.poll() is None:
   send(signal.SIGTERM)
   try:out,err=proc.communicate(timeout=tail())
   except subprocess.TimeoutExpired:
    send(signal.SIGKILL)
    try:out,err=proc.communicate(timeout=tail())
    except subprocess.TimeoutExpired:pass
  raise
 finally:
  with (state/(tag+'.stdout')).open('x') as f:f.write(out)
  with (state/(tag+'.stderr')).open('x') as f:f.write(err)
  result={'started_utc':started.isoformat(),'observed_utc':now().isoformat(),'elapsed_seconds':time.monotonic()-tick,'exit_code':proc.returncode,'local_transport_timeout':timed,'signals_to_own_local_group':signals,'local_transport_pid':proc.pid,'local_transport_reaped':proc.poll() is not None,'failure':failure,'remote_closure_separately_required':True}
  put(state/(tag+'.external.json'),result)
 need(not timed and proc.poll() is not None and not signals,'Original transport failed: preserve attempt, no automatic retry')
 if require_zero:need(proc.returncode==0 and left()>5,'Original bounded control transport failed')
 return out,result
def ssh(role,command):return ['ssh','-T','-F',str(P/'ssh_config'),ALIASES[role],shlex.join(command)]
def probe(state,role,action,extra=None):
 p=prepared(state);v=p['roles'][role];_,stop,mono_stop=phase(state,tail=25)
 payload={'action':action,'role':role,'stop_utc':stop.isoformat(),'files_sha256':v['files_sha256'],'historical_pids':v['historical_pids'],'forbidden_program_tokens':p['forbidden_program_tokens'],'expected_boot_id':v['expected_boot_id']}
 payload.update(extra or {})
 seconds=min(120,math.floor(min((stop-now()).total_seconds(),mono_stop-time.monotonic()))-20);need(seconds>0,'No remote helper allowance')
 command=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(seconds)+'s',PYTHON,'-I','-S','-B','-c',(PK/'remote_control.py').read_text()]
 out,_=bounded_local(state,ssh(role,command),role+'.'+action,stop,mono_stop,json.dumps(payload),seconds+8)
 result=json.loads(out);need(result['hostname']==v['hostname'],'Unexpected host')
 clock=result['clock'];need(clock['host_boot_id']==v['expected_boot_id'] and abs((now()-dt(clock['host_utc'])).total_seconds())<=5,'Fresh exact host/root clock required')
 result['root_reference_utc']=now().isoformat();put(state/(role+'.'+action+'.json'),result);return result
def accepted_issue_review(state,role,file,pin):
 need(file and digest(pin) and sha(file)==pin,'Actual independent issuance review bytes required');r=read(file)
 need(r['status']=='passed_actual_final24_issuance' and r['control_phase_sha256']==sha(state/'control_phase.json') and r['prepared_inputs_sha256']==sha(state/'prepared_inputs.json') and r['release_sha256'][role]==sha(state/(role+'.evaluation_release.json')),'Independent actual issue review does not bind this release')
 return {'path':str(Path(file).resolve()),'sha256':pin}
def fresh_capture(state,role,action):
 v=read(state/(role+'.'+action+'.json'));p=prepared(state)['roles'][role];age=(now()-dt(v['clock']['host_utc'])).total_seconds()
 need(0<=age<=300 and v['clock']['host_boot_id']==p['expected_boot_id'] and abs((dt(v['root_reference_utc'])-dt(v['clock']['host_utc'])).total_seconds())<=5,'Fresh verified host/root evidence required')
 return v
def resolve_release(candidate,capture,control,prepared_pin,phase_pin,capture_pin):
 r=json.loads(json.dumps(candidate));stop=dt(control['stop_utc'])
 need(r['files_sha256']==capture['input_sha256'],'Fresh complete input bytes differ')
 r.update(status='admitted_for_execution_allocation',issued_by='root',preparation_and_cohort_reserves_already_accounted_before_this_queue=True,clock_error_bound_seconds=5,latest_start_utc=(stop-datetime.timedelta(seconds=5)).isoformat(),process_clock_checked_utc=capture['clock']['host_utc'])
 meta=r['candidate_metadata'];meta.update(actual_release_created=True,fresh_process_closure_attestation={'control_phase_sha256':phase_pin,'capture_sha256':capture_pin},fresh_gpu_inventory_attestation={'capture_sha256':capture_pin},fresh_clock_review_attestation={'capture_sha256':capture_pin,'control_stop_utc':stop.isoformat()},fresh_output_absence_attestation={'capture_sha256':capture_pin},actual_complete_immutable_array_map={p:h for p,h in r['files_sha256'].items() if p.endswith('.npy')})
 meta['prepared_inputs_sha256']=prepared_pin
 return r
def launch_shell(argv,release_sha):
 need(digest(release_sha),'Release SHA required');path=argv[argv.index('--release')+1]
 # The shell checks only the root release bytes, then exec replaces it with env
 # and the exact frozen Python owner argv. No scientific wrapper or extra flag.
 checksum='test "$(sha256sum -- '+shlex.quote(path)+')" = '+shlex.quote(release_sha+'  '+path)
 return checksum+' && exec '+shlex.join(['/usr/bin/env','-u','CUDA_VISIBLE_DEVICES']+[k+'='+v for k,v in ENV.items()]+argv)

def main():
 a=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);a.add_argument('action',choices=['describe','prepare','begin-control','verify-stage','issue','stage','launch','record-session','observe-owner','close-control','record-evaluation-exit']);a.add_argument('--state',type=Path);a.add_argument('--package-sha256');a.add_argument('--root-action',action='store_true');a.add_argument('--role',choices=['A','B']);a.add_argument('--review-file',type=Path);a.add_argument('--review-sha256');a.add_argument('--session-id',type=int);a.add_argument('--exit-code',type=int);a.add_argument('--observed-utc');a.add_argument('--evidence-file',type=Path);args=a.parse_args()
 if args.action=='describe':print(json.dumps({'status':'inert_final24_root_operator','control_seconds':600,'evaluation_seconds':11760,'analysis_seconds':3600,'global_stop_utc':GLOBAL_STOP}));return
 need(args.root_action and args.state and args.state.is_absolute() and args.state.resolve()==args.state,'Explicit root action and canonical state required');package_check(args.package_sha256);s=args.state
 if args.action=='prepare':
  p=read(PK/'prepared_inputs.json')
  for f,h in p['original_local_evidence_sha256'].items():need(sha(f)==h,'Reviewed local evidence changed')
  s.mkdir(exist_ok=False)
  with (s/'prepared_inputs.json').open('xb') as f:f.write((PK/'prepared_inputs.json').read_bytes())
  put(s/'operator_binding.json',{'manifest_sha256':args.package_sha256,'local_preparation_only':True,'clock_read_or_issued':False});print(json.dumps({'status':'local_inputs_prepared_no_clock','prepared_inputs_sha256':sha(s/'prepared_inputs.json')}));return
 need(read(s/'operator_binding.json')['manifest_sha256']==args.package_sha256,'Operator package cannot change mid-phase');p=prepared(s)
 if args.action=='begin-control':
  start=now();tick=time.monotonic();stop=start+datetime.timedelta(seconds=600)
  need(stop+datetime.timedelta(seconds=11760+3600+15)<=dt(GLOBAL_STOP),'Full original control/evaluation/analysis allocation cannot fit')
  put(s/'control_phase.json',{'schema':'sand_final24_original_control_phase_v1','status':'approved_one_original_final24_control_phase','issued_by':'root','started_utc':start.isoformat(),'stop_utc':stop.isoformat(),'control_seconds':600,'evaluation_seconds':11760,'analysis_seconds':3600,'clock_restarted':False,'global_analysis_stop_utc':GLOBAL_STOP,'prepared_inputs_sha256':sha(s/'prepared_inputs.json')})
  put(s/'control_anchor.json',{'utc':start.isoformat(),'monotonic_seconds':tick,'phase_sha256':sha(s/'control_phase.json')});print(json.dumps(read(s/'control_phase.json')));return
 if args.action=='record-evaluation-exit':
  role=args.role;need(role in ALIASES and args.session_id and type(args.exit_code) is int and args.observed_utc and args.evidence_file,'Actual original tool exit fields required')
  session=read(s/(role+'.original_session.json'));ext=read(s/(role+'.evaluation.external.json'))
  need(args.session_id==session['original_tool_session'] and dt(ext['observed_utc'])<=dt(args.observed_utc)<=now(),'Original evaluation observation differs')
  put(s/(role+'.original_exit.json'),{'recorded_by':'root','original_tool_session':args.session_id,'original_tool_exit_code':args.exit_code,'observed_utc':args.observed_utc,'evidence_file':str(args.evidence_file.resolve()),'evidence_sha256':sha(args.evidence_file),'external_sha256':sha(s/(role+'.evaluation.external.json')),'native_closure_still_required':True});return
 control,stop,mono_stop=phase(s)
 if args.action=='close-control':
  receipts={role:read(s/(role+'.owner_observed.json')) for role in ALIASES}
  need(all(v['control_phase_sha256']==sha(s/'control_phase.json') for v in receipts.values()),'Both original owner observations required')
  put(s/'control_completion.json',{'status':'both_original_owners_observed_inside_same_control_window','control_phase_sha256':sha(s/'control_phase.json'),'owner_observation_sha256':{role:sha(s/(role+'.owner_observed.json')) for role in ALIASES},'checked_utc':now().isoformat(),'scientific_completion':False});return
 role=args.role;need(role in ALIASES,'Exact host role required');v=p['roles'][role]
 if args.action=='verify-stage':
  sources={remote:base64.b64encode((PK/row['package_path']).read_bytes()).decode() for remote,row in p['source_files'].items()}
  result=probe(s,role,'verify_stage',{'sources':sources});need(result['input_sha256']==v['files_sha256'],'Full original immutable inputs differ');print(json.dumps({'role':role,'status':'verified_sources_inputs_environment_and_fresh_parents','capture_sha256':sha(s/(role+'.verify_stage.json'))}));return
 if args.action=='issue':
  capture=fresh_capture(s,role,'verify_stage');r=resolve_release(read(PK/v['resolved_candidate']),capture,control,sha(s/'prepared_inputs.json'),sha(s/'control_phase.json'),sha(s/(role+'.verify_stage.json')))
  need(now()+datetime.timedelta(seconds=20)<dt(r['latest_start_utc']),'Insufficient same-control staging/launch allowance');put(s/(role+'.evaluation_release.json'),r);print(json.dumps({'role':role,'release_sha256':sha(s/(role+'.evaluation_release.json')),'prepared_inputs_sha256':sha(s/'prepared_inputs.json'),'control_phase_sha256':sha(s/'control_phase.json')}));return
 if args.action=='stage':
  review=accepted_issue_review(s,role,args.review_file,args.review_sha256);f=s/(role+'.evaluation_release.json')
  result=probe(s,role,'stage_release',{'release':{'path':CTRL+'/'+role+'.evaluation_release.json','sha256':sha(f),'base64':base64.b64encode(f.read_bytes()).decode()}})
  put(s/(role+'.staged.json'),{'release_sha256':sha(f),'capture_sha256':sha(s/(role+'.stage_release.json')),'independent_issue_review':review});print(json.dumps({'role':role,'status':'actual_release_staged','release_sha256':sha(f)}));return
 if args.action=='launch':
  stage=read(s/(role+'.staged.json'));review=stage['independent_issue_review'];accepted_issue_review(s,role,Path(review['path']),review['sha256']);fresh_capture(s,role,'stage_release')
  need(stage['release_sha256']==sha(s/(role+'.evaluation_release.json')),'Issued release changed');need(now()+datetime.timedelta(seconds=15)<dt(read(s/(role+'.evaluation_release.json'))['latest_start_utc']),'Same control launch allowance exhausted')
  argv=v['owner_argv'];command=['ssh','-T','-F',str(P/'ssh_config'),ALIASES[role],launch_shell(argv,stage['release_sha256'])]
  put(s/(role+'.launch.json'),{'exact_owner_argv':argv,'remote_launch_shell':command[-1],'release_sha256':stage['release_sha256'],'control_phase_sha256':sha(s/'control_phase.json'),'dispatched_utc':now().isoformat(),'original_tool_session_must_be_recorded':True})
  eval_stop=dt(GLOBAL_STOP)-datetime.timedelta(seconds=3600);eval_mono=time.monotonic()+(eval_stop-now()).total_seconds()
  _,result=bounded_local(s,command,role+'.evaluation',eval_stop,eval_mono,require_zero=False);code=result['exit_code'];print(json.dumps(result));raise SystemExit(code if code>=0 else 128-code)
 if args.action=='record-session':
  need(args.session_id and args.session_id>0 and args.evidence_file and args.evidence_file.is_file() and (s/(role+'.launch.json')).is_file(),'Actual original launch session evidence required')
  need(all(read(f)['original_tool_session']!=args.session_id for f in s.glob('*.original_session.json')),'Original session cannot be reused')
  put(s/(role+'.original_session.json'),{'recorded_by':'root','original_tool_session':args.session_id,'evidence_file':str(args.evidence_file.resolve()),'evidence_sha256':sha(args.evidence_file),'launch_sha256':sha(s/(role+'.launch.json')),'observed_utc':now().isoformat()});return
 if args.action=='observe-owner':
  original=read(s/(role+'.original_session.json'));need(sha(original['evidence_file'])==original['evidence_sha256'],'Original launch session evidence changed')
  result=probe(s,role,'observe_owner',{'release_sha256':sha(s/(role+'.evaluation_release.json')),'expected_owner_argv':v['owner_argv']})
  put(s/(role+'.owner_observed.json'),{'status':'observed_original_final24_owner','control_phase_sha256':sha(s/'control_phase.json'),'release_sha256':sha(s/(role+'.evaluation_release.json')),'original_tool_session':original['original_tool_session'],'native_capture_sha256':sha(s/(role+'.observe_owner.json')),'checked_utc':now().isoformat(),'scientific_completion':False});print(json.dumps(result));return
 raise ValueError(args.action)
if __name__=='__main__':main()
