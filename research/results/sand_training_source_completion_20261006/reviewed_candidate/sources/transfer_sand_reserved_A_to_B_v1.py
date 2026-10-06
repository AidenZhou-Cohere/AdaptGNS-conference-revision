#!/usr/bin/env python3
"""Inert unless --execute receives root release. No scientific execution or data interpretation."""
import argparse,datetime,hashlib,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
R='/root/repos/AdaptGNS-cuda-20261006'
P=Path(__file__).resolve().parent
MAP_SHA='af9d6e1fab219d296f044db00c7d6f484b56d379452d9a4593ab4c216273dbcc'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 with Path(p).open('x') as f:f.write(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc)
def dt(s):
 x=datetime.datetime.fromisoformat(s);assert x.tzinfo is not None;return x

def validate(release,plan,phase):
 assert release['schema']=='sand_reserved_A_to_B_transfer_release_v1'
 assert release['issued_by']=='root' and release['status']=='approved_one_bounded_A_to_B_transfer'
 assert release['automatic_retry'] is False and release['new_or_restarted_clock_granted'] is False
 assert release['original_shared_source_seconds']==900
 assert phase['issued_by']=='root' and phase['schema']=='adaptgns_sand_original_reserved_source_phase_v1'
 assert phase['status']=='approved_original_shared_source_phase'
 assert phase['original_shared_source_seconds']==900 and phase['original_evaluation_seconds']==11760 and phase['original_analysis_seconds']==3600
 assert phase['new_or_restarted_clock_granted'] is False
 assert phase['original_verification_phase_sha256']==plan['original_verification_sha256']
 assert (dt(phase['preparation_stop_utc'])-dt(phase['preparation_started_utc'])).total_seconds()==900
 assert dt(phase['preparation_started_utc'])<=utc()<dt(phase['preparation_stop_utc'])
 assert set(release['files_sha256'])==set(plan['files_sha256'])
 for p,h in release['files_sha256'].items():
  assert isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)
  assert plan['files_sha256'][p] in (None,h)
 assert release['files_sha256'][plan['source_phase_path']]==release['source_phase_sha256']

# Executed by a single remote stdlib process via SSH. Test harness calls this function
# on opaque temporary files; no scientific modules are imported.
def remote_operation(v):
 import datetime,hashlib,json,os,signal,socket,stat,time
 from pathlib import Path
 def now():return datetime.datetime.now(datetime.timezone.utc)
 stop=datetime.datetime.fromisoformat(v['stop_utc'])
 remaining=(stop-now()).total_seconds()-8
 assert remaining>0
 def expired(signum,frame):raise TimeoutError('Original shared source stop')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
 assert socket.gethostname()==v['hostname']
 repo=Path(v['repo']); stage=Path(v['staging']); payload=stage/'payload'
 def guard():assert now()<stop-datetime.timedelta(seconds=8),'Original shared source stop'
 def safe(p,allow_missing=False):
  p=Path(p);assert p.is_absolute() and '..' not in p.parts and p.is_relative_to(repo)
  for a in [p,*p.parents]:
   if a==repo.parent:break
   if a.exists() or a.is_symlink():assert not a.is_symlink()
  if not allow_missing:assert p.is_file()
  return p
 def digest(p):
  guard();p=safe(p);h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):guard();h.update(b)
  return h.hexdigest()
 def tree(p):
  p=safe(p,True);assert p.is_dir();found=set()
  for root,dirs,files in os.walk(p,followlinks=False):
   guard()
   for name in dirs:
    d=Path(root,name);assert not d.is_symlink() and any(d.iterdir()),'Unexpected empty or symbolic directory'
   for name in files:
    f=safe(Path(root,name));found.add(str(f))
  return found
 files=v['files'];numeric={p for p in files if p.startswith(v['numeric_tree']+'/')}
 assert len(numeric)==63
 action=v['action'];result={'action':action,'hostname':socket.gethostname(),'pid':os.getpid(),'checked_utc':now().isoformat()}
 if action=='source':
  assert tree(v['numeric_tree'])==numeric
  assert all(digest(p)==h for p,h in files.items())
  result['files_sha256']=files
 elif action=='prepare':
  safe(stage,True);assert not stage.exists();missing=[]
  for p,h in files.items():
   q=safe(p,True)
   if q.exists():assert q.is_file() and digest(q)==h
   else:assert p not in v['existing_only_paths'];missing.append(p)
  if Path(v['numeric_tree']).exists():assert tree(v['numeric_tree']).issubset(numeric)
  staged=set(missing)
  if staged.intersection(numeric):staged.update(numeric)
  stage.mkdir(mode=0o700);payload.mkdir(mode=0o700)
  # Parent of numeric tree exists; scp -r itself must create the numeric leaf.
  for p in staged:
   target=payload/Path(p).relative_to(repo)
   parent=(payload/Path(v['numeric_tree']).relative_to(repo)).parent if p in numeric else target.parent
   parent.mkdir(mode=0o700,parents=True,exist_ok=True)
  result.update(missing=sorted(missing),staged=sorted(staged))
 elif action=='publish':
  staged=set(v['staged']);assert staged.issubset(files)
  expected={str(payload/Path(p).relative_to(repo)) for p in staged}
  assert tree(payload)==expected
  for p in staged:assert digest(payload/Path(p).relative_to(repo))==files[p]
  published=[];retained=[]
  for p,h in files.items():
   guard();q=safe(p,True)
   if q.exists():assert digest(q)==h;retained.append(p);continue
   assert p in staged and p not in v['existing_only_paths'];q.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
   safe(q,True);source=payload/q.relative_to(repo)
   # link() publishes exclusively; no overwrite and no delete of partial attempts.
   try:os.link(source,q,follow_symlinks=False);published.append(p)
   except FileExistsError:assert digest(q)==h;retained.append(p)
  assert tree(v['numeric_tree'])==numeric
  assert all(digest(p)==h for p,h in files.items())
  result.update(files_sha256=files,published=published,retained=retained,staging_retained=True)
 elif action=='closure':
  absent={str(pid):not Path('/proc',str(pid)).exists() for pid in v['pids']};assert all(absent.values())
  result['prior_remote_helpers_absent']=absent
 else:raise ValueError(action)
 guard();result['finished_utc']=now().isoformat();return result

def remote_code():
 import inspect
 return inspect.getsource(remote_operation)+'\nimport json,sys\nprint(json.dumps(remote_operation(json.load(sys.stdin)),sort_keys=True))\n'

def build_copies(plan,staged,config):
 numeric=plan['numeric_tree'];groups={};commands=[]
 for p in staged:
  if p.startswith(numeric+'/'):continue
  groups.setdefault(str(Path(p).parent),[]).append(p)
 prefix=['scp','-3','-p','-F',str(config)]
 for parent,files in sorted(groups.items()):
  dest=Path(plan['staging_root'])/'payload'/Path(parent).relative_to(R)
  commands.append(prefix+[plan['source_alias']+':'+p for p in sorted(files)]+[plan['target_alias']+':'+str(dest)+'/'])
 if any(p.startswith(numeric+'/') for p in staged):
  dest=Path(plan['staging_root'])/'payload'/Path(numeric).relative_to(R).parent
  commands.append(prefix+['-r',plan['source_alias']+':'+numeric,plan['target_alias']+':'+str(dest)+'/'])
 return commands

class Controller:
 def __init__(self,phase_path,phase_sha,receipt,stop):
  self.phase_path=phase_path;self.phase_sha=phase_sha;self.receipt=receipt;self.stop=stop
  # Map original UTC stop once, then use the earlier UTC/monotonic remaining time.
  self.mono_stop=time.monotonic()+(stop-utc()).total_seconds();self.index=0
 def bounded(self,argv,tag,input=None,limit=60):
  assert sha(self.phase_path)==self.phase_sha
  remaining=min((self.stop-utc()).total_seconds(),self.mono_stop-time.monotonic())
  timeout=min(limit,remaining-12);assert timeout>0,'Original shared source window exhausted'
  self.index+=1;name=f'{self.index:02d}_{tag}';put(self.receipt/(name+'.command.json'),{'argv':argv,'timeout_seconds':timeout})
  start=utc();tick=time.monotonic();proc=subprocess.Popen(argv,stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True);timed=False;signals=[];out='';err=''
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
    immediate=signal.SIGKILL if timed else signal.SIGTERM
    try:os.killpg(proc.pid,immediate);signals.append('SIGKILL' if timed else 'SIGTERM')
    except ProcessLookupError:pass
    def tail_wait():return max(0,min(3,(self.stop-utc()).total_seconds()-5,self.mono_stop-time.monotonic()-5))
    try:out,err=proc.communicate(timeout=tail_wait())
    except subprocess.TimeoutExpired:
     try:os.killpg(proc.pid,signal.SIGKILL);signals.append('SIGKILL')
     except ProcessLookupError:pass
     out,err=proc.communicate(timeout=tail_wait())
   raise
  finally:
   result={'started_utc':start.isoformat(),'observed_utc':utc().isoformat(),'elapsed_seconds':time.monotonic()-tick,'exit_code':proc.returncode,'local_transport_timeout':timed,'signals_to_own_local_group':signals,'local_transport_pid':proc.pid,'local_transport_reaped':proc.poll() is not None,'remote_closure_separately_required':True}
   (self.receipt/(name+'.stdout')).write_text(out);(self.receipt/(name+'.stderr')).write_text(err);put(self.receipt/(name+'.external.json'),result)
  assert not timed and proc.returncode==0 and result['local_transport_reaped'],result
  assert min((self.stop-utc()).total_seconds(),self.mono_stop-time.monotonic())>8
  return out

def main():
 if sys.flags.optimize:raise RuntimeError('Optimized interpreter forbidden: admission assertions must remain active')
 a=argparse.ArgumentParser(allow_abbrev=False);a.add_argument('--execute',action='store_true');a.add_argument('--release',type=Path);a.add_argument('--phase',type=Path);a.add_argument('--receipt-dir',type=Path);args=a.parse_args()
 if not args.execute:print(json.dumps({'status':'description_only','automatic_retry':False}));return
 assert args.release and args.phase and args.receipt_dir
 plan_path=P/'sand_reserved_transfer_UNADMITTED_transition_v1/exact_map.json';assert sha(plan_path)==MAP_SHA
 plan=read(plan_path);release_hash=sha(args.release);release=read(args.release);phase=read(args.phase);validate(release,plan,phase)
 assert release['controller_sha256']==sha(__file__) and release['map_sha256']==MAP_SHA
 config=P/'ssh_config';assert sha(config)==release['ssh_config_sha256']
 assert sha(args.phase)==release['source_phase_sha256']
 args.receipt_dir.mkdir(mode=0o700);put(args.receipt_dir/'release.json',release);put(args.receipt_dir/'phase.json',phase)
 control=Controller(args.phase,release['source_phase_sha256'],args.receipt_dir,dt(phase['preparation_stop_utc']))
 base={'repo':R,'files':release['files_sha256'],'staging':plan['staging_root'],'numeric_tree':plan['numeric_tree'],'existing_only_paths':plan['existing_only_paths'],'stop_utc':phase['preparation_stop_utc']}
 pids={'source':[],'target':[]}
 def call(host,action,**extra):
  payload={**base,'hostname':plan[host+'_host'],'action':action,**extra}
  argv=['ssh','-T','-F',str(config),plan[host+'_alias'],R+'/.venv/bin/python -I -S -B -c '+shlex.quote(remote_code())]
  result=json.loads(control.bounded(argv,host+'_'+action,json.dumps(payload),limit=60));assert result['hostname']==payload['hostname'];pids[host].append(result['pid']);put(args.receipt_dir/(host+'_'+action+'.json'),result);return result
 try:
  call('source','source');prep=call('target','prepare')
  commands=build_copies(plan,prep['staged'],config);put(args.receipt_dir/'scp_commands.json',commands)
  for index,command in enumerate(commands):control.bounded(command,'scp_'+str(index),limit=900)
  # Distinct action receipt filenames; source postcheck has same semantics.
  payload={**base,'hostname':plan['source_host'],'action':'source'}
  out=control.bounded(['ssh','-T','-F',str(config),plan['source_alias'],R+'/.venv/bin/python -I -S -B -c '+shlex.quote(remote_code())],'source_postcheck',json.dumps(payload),limit=60)
  post=json.loads(out);pids['source'].append(post['pid']);put(args.receipt_dir/'source_postcheck.json',post)
  result=call('target','publish',staged=prep['staged']);assert result['files_sha256']==release['files_sha256']
  call('source','closure',pids=pids['source'].copy());call('target','closure',pids=pids['target'].copy())
  assert sha(args.release)==release_hash and sha(args.phase)==release['source_phase_sha256'] and sha(config)==release['ssh_config_sha256'] and sha(__file__)==release['controller_sha256'] and sha(plan_path)==MAP_SHA
  put(args.receipt_dir/'terminal.json',{'status':'complete_exact_copy','finished_utc':utc().isoformat(),'files_sha256':release['files_sha256'],'source_phase_sha256':release['source_phase_sha256'],'staging_retained':True,'all_local_transports_exited_zero_and_reaped':True,'prior_remote_helper_pids_absent':True,'scp_protocol_success_required':True,'root_must_review_original_controller_exit_before_B_preflight':True,'does_not_claim_unobservable_remote_transport_pid_absence':True})
 except BaseException as error:
  put(args.receipt_dir/'failure.json',{'status':'failed_preserve_all_bytes_no_retry','error':repr(error),'finished_utc':utc().isoformat(),'remote_closure_separately_required':True});raise
if __name__=='__main__':main()
