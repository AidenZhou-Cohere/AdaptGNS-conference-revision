"""Read-only process/GPU/source inventory for the authorized timing probe."""
from pathlib import Path
import subprocess,json,hashlib,datetime,os
r=Path('/root/repos/AdaptGNS-cuda-20261006'); p=r/'cuda_preparation'; d=r/'sand_numeric_train_valid_20261006_v1'
paths={'trainer':p/'train_sand_cuda_deterministic.py','train_manifest':d/'train.json','admission':p/'sand_train_admission.json','structural_report':d/'structural_report.json','protocol':p/'sand_cuda_capacity_protocol_v2.md','mechanism_review':p/'sand_native_cuda_numerical_assessment.json','python':r/'.venv/bin/python','supervisor':p/'measure_sand_cuda_capacity_v2.py'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
processes=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit(): continue
 try: args=[a.decode() for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
 except (OSError,UnicodeError): continue
 if any(Path(a).name in {'train_sand_cuda.py','train_sand_cuda_deterministic.py','measure_sand_cuda_capacity.py','measure_sand_cuda_capacity_v2.py','train_sand_graph_support_cuda.py','diagnose_sand_relu_kinks.py','validate_sand_cuda.py','benchmark_sand_cuda_rollout.py','benchmark_sand_cuda_rollout_deterministic.py','evaluate_sand_final_diagnostics.py'} for a in args): processes.append({'pid':int(proc.name),'argv':args})
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
active=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':{k:sha(v) for k,v in paths.items()},'gpu_inventory':gpu,'gpu_processes':active,'matching_processes':processes,'output_exists':(r/'sand_capacity_20261006_v2').exists(),'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()}
print(json.dumps(report,indent=2))
assert not active.strip() and not processes and not report['output_exists']
