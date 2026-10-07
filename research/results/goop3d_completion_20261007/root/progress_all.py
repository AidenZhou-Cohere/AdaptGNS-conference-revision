"""Read-only compact progress across the exact current jobs; save raw captures."""
import argparse,concurrent.futures,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
def one(hi):
 ids=list(range(4*hi,4*hi+4));jobs=[f'autonomous_worker{i:02d}_attempt1' for i in ids];outputs=[f'autonomous_results_v1/worker_{i:02d}' for i in ids]
 if hi==0:jobs.append('observed_16cpu_attempt1');outputs.append('observed_results_v1')
 dest=ROOT/f'{a.label}_host{hi}.json';args=[sys.executable,str(ROOT/'observe_remote.py'),'--spec',str(ROOT/f'autonomous_worker{ids[0]:02d}_attempt1.json'),'--output',str(dest),'--jobs',*jobs,'--outputs',*outputs]
 r=subprocess.run(args,capture_output=True,text=True)
 if r.returncode:return {'host_index':hi,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 d=json.loads(json.loads(dest.read_text())['stdout']);result={'host_index':hi,'captured_utc_seconds':d['utc_seconds'],'gpu_apps':len(d['gpu_processes'].splitlines()),'jobs':{},'progress':{}}
 for name,j in d['jobs'].items():
  st=j['status.json'];result['jobs'][name]={'status':st.get('status'),'returncode':st.get('returncode'),'child':st.get('child'),'native_child_state':(j.get('native',{}).get('child') or {}).get('state')}
  if st.get('returncode'):result['jobs'][name]['error']=j.get('worker.stderr')
 for name,o in d['outputs'].items():
  if 'checkpoints' in o:result['progress'][name]=o['checkpoints'];result['progress'][name]['completion_published']='completion.json' in o
  else:
   st=o.get('status.json',{});result['progress'][name]={'state':st.get('state'),'committed':len(st.get('committed_cell_ids',[])),'current':st.get('current_cell_id')}
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(one,range(3)):print(json.dumps(result))
