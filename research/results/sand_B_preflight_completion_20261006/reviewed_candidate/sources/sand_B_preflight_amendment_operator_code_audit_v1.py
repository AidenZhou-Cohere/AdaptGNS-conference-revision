#!/usr/bin/env python3
"""Narrow root operator for the separately human-approved two-B-preflight amendment.
Description/help is inert. No action may launch without the actual approval record.
"""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,os,shlex,signal,subprocess,sys,time
P=Path(__file__).resolve().parent
PK=P/'sand_B_preflight_amendment_UNADMITTED_transition_v1'
C=P/'sand_B_preflight_amendment_released_root_v1'
R='/root/repos/AdaptGNS-cuda-20261006';Q=R+'/sand_reserved_B_preflight_amendment_20261006_v1'
MANIFEST='72cfe26b2d8c13ea50d24b8d85e450ef228a4bf9ef681fb18a2de525f8640e6d'
SSH_CONFIG='f04af81d2b926b04f7a8063df5c8535da1d8c9aa637e15cf235c261d6d0725b2'
OPS=('preflight_valid','preflight_test')
def now():return datetime.datetime.now(datetime.timezone.utc)
def dt(x):return datetime.datetime.fromisoformat(x)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x') as f:f.write(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def package_check():
 assert not sys.flags.optimize and sha(PK/'manifest.json')==MANIFEST
 for p,h in read(PK/'manifest.json')['files_sha256'].items():assert sha(PK/p)==h,p
 assert sha(P/'ssh_config')==SSH_CONFIG

def approved():
 a=read(C/'human_approval.json');t=read(PK/'human_approval.candidate.json')
 changes={'authorization_source','human_approval_observed_utc','human_message_reference','human_message_text','recorded_by','recorded_utc','reviewed_candidate_manifest_sha256','status'}
 assert {k:v for k,v in a.items() if k not in changes}=={k:v for k,v in t.items() if k not in changes}
 assert a['status']=='explicit_human_approval_received' and a['recorded_by']=='root' and a['authorization_source']=='direct_human_message'
 assert a['reviewed_candidate_manifest_sha256']==MANIFEST and a['human_message_text'].strip() and a['human_message_reference'].strip()
 assert dt('2026-10-06T18:35:55.561287+00:00')<=dt(a['human_approval_observed_utc'])<=dt(a['recorded_utc'])<=now()
 return a

def bounded_local(argv,tag,input=None,limit=45):
 """Reviewed B SSH transport plus the reviewed transfer's interrupt cleanup."""
 approved();phase=C/'amendment_phase.json';start=now();tick=time.monotonic()
 stop=dt(read(phase)['amendment_stop_utc']) if phase.exists() else start+datetime.timedelta(seconds=600)
 mono_stop=tick+(stop-start).total_seconds();timeout=min(limit,(stop-start).total_seconds()-12)
 assert timeout>0,'Shared amendment window exhausted'
 put(C/(tag+'.command.json'),{'argv':argv,'timeout_seconds':timeout})
 proc=subprocess.Popen(argv,stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 timed=False;signals=[];out='';err=''
 try:
  try:out,err=proc.communicate(input=input,timeout=timeout)
  except subprocess.TimeoutExpired:
   timed=True
   try:os.killpg(proc.pid,signal.SIGTERM);signals.append('SIGTERM')
   except ProcessLookupError:pass
   try:out,err=proc.communicate(timeout=3)
   except subprocess.TimeoutExpired:
    try:os.killpg(proc.pid,signal.SIGKILL);signals.append('SIGKILL')
    except ProcessLookupError:pass
    out,err=proc.communicate(timeout=3)
 except BaseException:
  if proc.poll() is None:
   sig=signal.SIGKILL if timed else signal.SIGTERM
   try:os.killpg(proc.pid,sig);signals.append('SIGKILL' if timed else 'SIGTERM')
   except ProcessLookupError:pass
   def tail():return max(0,min(3,(stop-now()).total_seconds()-5,mono_stop-time.monotonic()-5))
   try:out,err=proc.communicate(timeout=tail())
   except subprocess.TimeoutExpired:
    try:os.killpg(proc.pid,signal.SIGKILL);signals.append('SIGKILL')
    except ProcessLookupError:pass
    out,err=proc.communicate(timeout=tail())
  raise
 finally:
  result={'started_utc':start.isoformat(),'observed_utc':now().isoformat(),'elapsed_seconds':time.monotonic()-tick,'exit_code':proc.returncode,'local_transport_timeout':timed,'signals_to_own_local_group':signals,'local_transport_pid':proc.pid,'local_transport_reaped':proc.poll() is not None,'remote_closure_separately_required':True}
  (C/(tag+'.stdout')).write_text(out);(C/(tag+'.stderr')).write_text(err);put(C/(tag+'.external.json'),result)
 assert not timed and proc.returncode==0 and result['local_transport_reaped'],result
 assert min((stop-now()).total_seconds(),mono_stop-time.monotonic())>8
 return out

def ssh(code,payload,tag,limit=45):
 out=bounded_local(['ssh','-T','-F',str(P/'ssh_config'),'aquamarine-toad-75.coder',R+'/.venv/bin/python -I -S -B -c '+shlex.quote(code)],tag,json.dumps(payload),limit)
 return json.loads(out)

PRECHECK=r'''import json,sys,pathlib,hashlib,subprocess,datetime,time,socket
v=json.load(sys.stdin);assert socket.gethostname()==v['hostname'];absence={str(p):not(pathlib.Path('/proc')/str(p)).exists() for p in v['pids']};assert all(absence.values())
for p,h in v['inputs'].items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h
for p in v['fresh']:assert not pathlib.Path(p).exists()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader,nounits'],text=True,timeout=5);assert not apps.strip()
print(json.dumps({'hostname':socket.gethostname(),'clock':{'host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_monotonic_seconds':time.monotonic(),'host_boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()},'native_absent':absence,'gpu_apps':apps,'input_sha256':v['inputs']}))'''
STAGE=r'''import json,sys,pathlib,hashlib,base64,datetime
v=json.load(sys.stdin)
for row in v:
 p=pathlib.Path(row['path']);b=base64.b64decode(row['base64']);assert hashlib.sha256(b).hexdigest()==row['sha256'];assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[{'path':r['path'],'sha256':r['sha256']} for r in v]}))'''
COLLECT=r'''import json,sys,pathlib,hashlib,base64,datetime,socket,subprocess
v=json.load(sys.stdin);q=pathlib.Path(v['owner_output_dir']);t=json.loads((q/'owner_terminal.json').read_bytes());assert socket.gethostname()==v['hostname'];assert t['status']=='complete' and t['failure'] is None and t['release_sha256']==v['_expected_release_hash'];assert t['child']['command']==v['command'] and t['child']['reaped'] and t['child']['exit_code']==0 and t['child']['signals']==[]
pids=[t['outer_timeout_identity']['pid'],t['owner_identity']['pid'],t['child']['pid']];absence={str(p):not(pathlib.Path('/proc')/str(p)).exists() for p in pids};assert all(absence.values())
rows=[]
for p in list(q.rglob('*'))+list(pathlib.Path(v['paths']['--output-dir']).rglob('*')):
 if p.is_dir():continue
 assert p.is_file() and not p.is_symlink();b=p.read_bytes();rows.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'base64':base64.b64encode(b).decode()})
assert all(next(r['sha256'] for r in rows if r['path']==p)==h for p,h in t['output_sha256'].items())
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader,nounits'],text=True,timeout=5);assert not apps.strip()
print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'native_absent':absence,'gpu_apps':apps,'files':rows,'input_sha256':t['input_sha256'],'output_sha256':t['output_sha256']}))'''

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['describe','record-approval','precheck','issue','stage','run','record-original-exit','collect']);a.add_argument('--operation',choices=OPS);a.add_argument('--message-file',type=Path);a.add_argument('--message-reference');a.add_argument('--observed-utc');a.add_argument('--session-id',type=int);a.add_argument('--exit-code',type=int);args=a.parse_args()
 if args.action=='describe':print(json.dumps({'status':'inert_not_authorized','candidate_manifest_sha256':MANIFEST,'operations':OPS,'human_approval_required':True}));return
 package_check()
 if args.action=='record-approval':
  assert args.message_file and args.message_reference and args.observed_utc
  a=read(PK/'human_approval.candidate.json');a.update(authorization_source='direct_human_message',human_approval_observed_utc=args.observed_utc,human_message_reference=args.message_reference,human_message_text=args.message_file.read_text(),recorded_by='root',recorded_utc=now().isoformat(),reviewed_candidate_manifest_sha256=MANIFEST,status='explicit_human_approval_received')
  assert a['human_message_text'].strip() and dt('2026-10-06T18:35:55.561287+00:00')<=dt(args.observed_utc)<=now();C.mkdir();put(C/'human_approval.json',a);approved();return
 approved();commands={r['operation']:r for r in read(PK/'commands.json')['commands']}
 if args.action=='precheck':
  assert not(C/'amendment_phase.json').exists();pins=read(PK/'immutable_original_transfer_bindings.json');pins=pins.get('files_sha256',pins)
  pins.update({R+'/.venv/bin/python':'6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a',R+'/.venv/pyvenv.cfg':'b7d360e62970717794ce0ab4f1e1398a49a8a73a138d34ca920b1eb251fe2f3b','/usr/bin/timeout':'2db30bc57746c940a0643581cd5e2671e505442bea16164143f0ea8ff4e36453'})
  fresh=[Q]+[R+'/sand_reserved_preparation_20261006_v1/preflight_'+s+'_B' for s in ['valid','test']]
  d=ssh(PRECHECK,{'hostname':commands[OPS[0]]['hostname'],'pids':[31391,31394,31395,66270,66272,66273],'inputs':pins,'fresh':fresh},'precheck',60)
  d['clock'].update(root_reference_utc=now().isoformat(),source='root_fresh_tool_and_host_clock_evidence',error_bound_seconds=5);assert abs((dt(d['clock']['host_utc'])-dt(d['clock']['root_reference_utc'])).total_seconds())<=5;put(C/'precheck.json',d);print(json.dumps(d['clock']));return
 if args.action=='issue':
  clock=read(C/'precheck.json')['clock'];origin=dt(clock['host_utc']);start=min(origin,dt(read(C/'precheck.external.json')['started_utc']));end=start+datetime.timedelta(seconds=600);at=now()
  assert dt(approved()['recorded_utc'])<=start<=origin<=at<origin+datetime.timedelta(seconds=300) and at<end-datetime.timedelta(seconds=60)
  assert end+datetime.timedelta(seconds=600+11760+3600+5)<=dt('2026-10-07T04:00:00+00:00')
  phase=read(PK/'amendment_phase.candidate.json');phase.update(issued_by='root',status='approved_new_shared_B_preflight_amendment_phase',issued_utc=at.isoformat(),amendment_started_utc=start.isoformat(),amendment_stop_utc=end.isoformat(),reviewed_candidate_manifest_sha256=MANIFEST);phase['human_approval']['sha256']=sha(C/'human_approval.json');put(C/'amendment_phase.json',phase)
  for op in OPS:
   core=read(PK/(op+'.mode_release.candidate.json'));core.update(issued_by='root',status='approved_for_preflight',issued_utc=now().isoformat(),preparation_started_utc=start.isoformat(),preparation_stop_utc=end.isoformat(),amendment_phase_sha256=sha(C/'amendment_phase.json'));put(C/(op+'.mode_release.json'),core)
   r=read(PK/(op+'.cpu_release.candidate.json'));r.update(issued_by='root',status='approved_one_bounded_B_preflight_amendment_invocation',issued_utc=now().isoformat(),clock_sample=clock,invocation_origin_utc=origin.isoformat(),publication_deadline_utc=end.isoformat(),work_stop_utc=(end-datetime.timedelta(seconds=15)).isoformat(),cleanup_deadline_utc=(end-datetime.timedelta(seconds=5)).isoformat(),remaining_shared_amendment_seconds=(end-origin).total_seconds(),hard_deadline_monotonic_ns=int((clock['host_monotonic_seconds']+(end-origin).total_seconds()-5)*10**9),reviewed_candidate_manifest_sha256=MANIFEST)
   r['amendment_phase']['sha256']=sha(C/'amendment_phase.json');r['human_approval']['sha256']=sha(C/'human_approval.json');r['core_release']['sha256']=sha(C/(op+'.mode_release.json'));r['inputs_sha256']={**core['files_sha256'],r['core_release']['path']:r['core_release']['sha256']};r['outer_timeout']['version']='timeout (GNU coreutils) 9.4';put(C/(op+'.cpu_release.json'),r)
   argv=commands[op]['direct_bootstrap_argv'].copy();argv[argv.index('--hard-deadline-monotonic-ns')+1]=str(r['hard_deadline_monotonic_ns']);argv[argv.index('--release-sha256')+1]=sha(C/(op+'.cpu_release.json'));put(C/(op+'.launch.json'),{'argv':argv,'command':r['command']})
  print(json.dumps({'phase_sha256':sha(C/'amendment_phase.json'),'source_candidate':MANIFEST,'stop':end.isoformat()}));return
 assert now()<dt(read(C/'amendment_phase.json')['amendment_stop_utc'])
 if args.action=='stage':
  files=[(PK/n,R+'/cuda_preparation/'+n) for n in ['supervise_sand_B_preflight_amendment_cpu_v1.py','launch_sand_B_preflight_amendment_cpu_v1.py']]
  files +=[(PK/'history'/Path(v['path']).name,v['path']) for v in read(PK/'amendment_phase.candidate.json')['history'].values()]
  files +=[(C/n,Q+'/controls/'+n) for n in ['human_approval.json','amendment_phase.json']+[op+suffix for op in OPS for suffix in ['.mode_release.json','.cpu_release.json']]]
  rows=[{'path':remote,'sha256':sha(p),'base64':base64.b64encode(p.read_bytes()).decode()} for p,remote in files];d=ssh(STAGE,rows,'stage');assert d['files']==[{k:r[k] for k in ['path','sha256']} for r in rows];put(C/'stage.json',d);print(json.dumps(d));return
 op=args.operation;assert op in OPS
 if args.action=='run':
  assert (C/'stage.json').exists();argv=read(C/(op+'.launch.json'))['argv'];bounded_local(['ssh','-T','-F',str(P/'ssh_config'),'aquamarine-toad-75.coder',shlex.join(argv)],op,limit=600);print(json.dumps(read(C/(op+'.external.json'))));return
 if args.action=='record-original-exit':
  ext=read(C/(op+'.external.json'));assert args.session_id and args.session_id>0 and args.exit_code is not None and args.observed_utc
  assert dt(ext['observed_utc'])<=dt(args.observed_utc)<=now()<dt(read(C/'amendment_phase.json')['amendment_stop_utc'])
  put(C/(op+'.root_original_external_exit.json'),{'original_tool_session':args.session_id,'original_tool_exit_code':args.exit_code,'observed_utc':args.observed_utc,'recorded_by':'root','scope':'Root observation of the original tool session, separate from SSH transport and remote native closure.'});return
 if args.action=='collect':
  original=read(C/(op+'.root_original_external_exit.json'));assert original['original_tool_exit_code']==0
  ext=read(C/(op+'.external.json'));assert ext['exit_code']==0 and ext['local_transport_reaped'] and not ext['local_transport_timeout'];r=read(C/(op+'.cpu_release.json'));r['_expected_release_hash']=sha(C/(op+'.cpu_release.json'));d=ssh(COLLECT,r,op+'.collect',60)
  for row in d['files']:
   b=base64.b64decode(row.pop('base64'));assert sha_bytes(b)==row['sha256'];p=C/'collected'/row['path'].removeprefix(R+'/');p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(b)
  put(C/(op+'.collection.json'),d);print(json.dumps({'operation':op,'checked_utc':d['checked_utc'],'native_absent':d['native_absent'],'files':len(d['files']),'outputs':d['output_sha256']}));return
 raise ValueError(args.action)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':main()
