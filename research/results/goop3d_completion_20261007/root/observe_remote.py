"""Read-only job metadata and bounded progress capture on authorized hosts."""
import argparse,json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
REMOTE=r'''
import json,os,socket,subprocess,sys,time
from pathlib import Path
request=json.loads(sys.stdin.read());base=Path('/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007')
assert socket.gethostname()==request['hostname']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==request['boot_id']
result={'hostname':socket.gethostname(),'boot_id':request['boot_id'],'utc_seconds':time.time(),'jobs':{},'outputs':{}}
for name in request['jobs']:
 job=base/'jobs'/name;item={}
 for filename in ['spec.json','status.json']:
  p=job/filename;item[filename]=json.loads(p.read_text()) if p.exists() else None
 if item['status.json']:
  item['native']={}
  for role in ['owner','child']:
   identity=item['status.json'].get(role)
   if identity:
    p=Path('/proc')/str(identity['pid'])/'stat'
    if p.exists():
     raw=p.read_text();fields=raw[raw.rfind(')')+2:].split()
     item['native'][role]={'pid':identity['pid'],'start_ticks':int(fields[19]),'state':fields[0],'ppid':int(fields[1]),'pgid':int(fields[2]),'argv':(p.parent/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')}
    else:item['native'][role]=None
 for filename in ['worker.stdout','worker.stderr','owner.stderr']:
  p=job/filename
  if p.exists():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));item[filename]=f.read().decode(errors='replace')
 result['jobs'][name]=item
for relative in request.get('outputs',[]):
 p=base/relative;item={'exists':p.exists()}
 if (p/'checkpoint_index.json').exists():
  index=json.loads((p/'checkpoint_index.json').read_text());item['checkpoints']={k:len(index.get(k,{})) for k in ['completed','failed']}
 for name in ['completion.json','status.json','worker_status.json']:
  if (p/name).exists():
   value=json.loads((p/name).read_text())
   if name=='completion.json':value={k:v for k,v in value.items() if k not in ('cache_row_sha256','cache_rows','input_files')}
   item[name]=value
 result['outputs'][relative]=item
result['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(result))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',nargs='+',required=True);p.add_argument('--outputs',nargs='*',default=[]);a=p.parse_args();assert not a.output.exists()
 s=json.loads(a.spec.read_text());request={k:s[k] for k in ['hostname','boot_id']};request.update(jobs=a.jobs,outputs=a.outputs)
 cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(ROOT/'work/deadline_research_20261005/cuda_preparation/ssh_config'),s['host_alias']+'.coder',shlex.join(['python3','-c',REMOTE])]
 r=subprocess.run(cmd,input=json.dumps(request),text=True,capture_output=True)
 a.output.write_text(json.dumps({'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(r.stdout,end='');print(r.stderr,end='');raise SystemExit(r.returncode)
if __name__=='__main__':main()
