"""Stage exact small sources and detach one uniquely named accounted job."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[3]
REMOTE=r'''
import base64,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
payload=json.loads(sys.stdin.read());spec=payload['spec']
assert socket.gethostname()==spec['hostname']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==spec['boot_id']
base=Path('/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007')
for relative,item in payload['sources'].items():
 p=base/relative;assert p.resolve().is_relative_to(base) and not p.is_symlink()
 data=base64.b64decode(item['base64']);assert hashlib.sha256(data).hexdigest()==item['sha256']
 if p.exists():assert p.read_bytes()==data
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
job=base/'jobs'/spec['job_id'];job.mkdir(parents=True,exist_ok=False)
raw=(json.dumps(spec,sort_keys=True,indent=2)+'\n').encode();pin=hashlib.sha256(raw).hexdigest()
with (job/'spec.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
with (job/'owner.stdout').open('xb') as out,(job/'owner.stderr').open('xb') as err:
 proc=subprocess.Popen(['/usr/bin/python3',str(base/'execution_v1/job_owner.py'),str(job),pin],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
stat=Path(f'/proc/{proc.pid}/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
print(json.dumps({'launched':True,'job':str(job),'owner_pid':proc.pid,'owner_start_ticks':int(fields[19]),'owner_pgid':os.getpgid(proc.pid),'spec_sha256':pin,'hostname':socket.gethostname(),'boot_id':spec['boot_id']}))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--source-prefix',required=True);p.add_argument('--files',nargs='+',required=True);a=p.parse_args()
 spec=json.loads(a.spec.read_text());assert spec['host_alias'] in {'teal-rat-80','yellow-worm-77','aquamarine-toad-75'}
 sources={}
 files=[('execution_v1/job_owner.py',Path(__file__).with_name('job_owner.py'))]+[(a.source_prefix+'/'+name,a.source_dir/name) for name in a.files]
 for relative,path in files:
  data=path.read_bytes();sources[relative]={'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
 payload=json.dumps({'spec':spec,'sources':sources})
 args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(ROOT/'work/deadline_research_20261005/cuda_preparation/ssh_config'),spec['host_alias']+'.coder',shlex.join(['python3','-c',REMOTE])]
 result=subprocess.run(args,input=payload,text=True,capture_output=True)
 destination=a.spec.with_suffix('.launch.json');assert not destination.exists()
 destination.write_text(json.dumps({'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'sources':{k:v['sha256'] for k,v in sources.items()}},indent=2)+'\n')
 print(result.stdout,end='');print(result.stderr,end='')
 raise SystemExit(result.returncode)

if __name__=='__main__':main()
