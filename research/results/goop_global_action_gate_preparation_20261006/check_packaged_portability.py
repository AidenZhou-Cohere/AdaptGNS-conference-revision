"""Reconstruct pinned preparation sources for inert CLIs and four interface checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle',required=True,type=Path)
parser.add_argument('--evidence',required=True,type=Path)
args=parser.parse_args();bundle=args.bundle.resolve();evidence=args.evidence.resolve()
evidence.mkdir(parents=True,exist_ok=False)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
deps=json.loads((bundle/'dependency_paths.json').read_text())
inventory=json.loads((bundle/'source_inventory.json').read_text())
pins=json.loads((bundle/'current_approved_pins.json').read_text())
for family,pin in deps['predecessor_manifests'].items():
 assert sha(bundle.parent/family/'manifest.json')==pin
temp=Path(tempfile.mkdtemp(prefix='goop_gate_packaged_interface_')).resolve()
restored={}
for relative,record in deps['files'].items():
 source=(bundle/record['published_path']).resolve();assert sha(source)==record['sha256']
 target=temp/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 restored[relative]=record['sha256']
for relative,record in inventory.items():
 source=bundle/relative;assert sha(source)==record['sha256']
 if relative.startswith('workspace/'):
  name=relative.removeprefix('workspace/');target=temp/name
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);restored[name]=record['sha256']
runtime=temp/'work/deadline_research_20261005/cuda_preparation'
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(runtime)
commands=[]
for name in [name for name in pins if name.endswith('.py')]:
 compile((runtime/name).read_text(),name,'exec')
 completed=subprocess.run([sys.executable,'-B',str(runtime/name)],cwd=temp,env=env,capture_output=True,text=True,timeout=30)
 (evidence/(name+'.stdout')).write_text(completed.stdout);(evidence/(name+'.stderr')).write_text(completed.stderr)
 commands.append({'kind':'inert_default_or_pure_module_import','entry':name,'exit_code':completed.returncode,
  'stdout_sha256':sha(evidence/(name+'.stdout')),'stderr_sha256':sha(evidence/(name+'.stderr'))})
 assert completed.returncode==0,name+': '+completed.stderr
 if name!='goop_action_gate_scalar_core_v1.py':assert json.loads(completed.stdout)['status']=='description_only'
 else:assert not completed.stdout
selectors=[
 'test_run_goop_action_gate_v1.py::test_row_roundtrip_accepted_by_frozen_fitter',
 'test_supervise_goop_action_gate_scoped_v1.py::test_root_admission_composes_all_three_scalar_driver_gates',
 'test_summarize_goop_action_gate_v1.py::test_original_protocol_closure_validates_without_model_load',
 'test_summarize_goop_action_gate_v1.py::test_synthetic_driver_row_array_core_composition[learned_global_gate]',
]
completed=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*selectors],cwd=runtime,env=env,
 capture_output=True,text=True,timeout=90)
(evidence/'targeted.stdout').write_text(completed.stdout);(evidence/'targeted.stderr').write_text(completed.stderr)
commands.append({'kind':'targeted_packaged_interface_checks','selectors':selectors,'exit_code':completed.returncode,
 'stdout_sha256':sha(evidence/'targeted.stdout'),'stderr_sha256':sha(evidence/'targeted.stderr')})
assert all(sha(temp/name)==pin for name,pin in restored.items())
assert all(sha(bundle/name)==record['sha256'] for name,record in inventory.items())
result={'schema':'adaptgns_goop_gate_packaged_portability_v1','passed':completed.returncode==0,
 'bundle':str(bundle),'temporary_restored_workspace':str(temp),'interpreter':sys.executable,
 'all_package_source_bytes_unchanged':True,'all_restored_source_bytes_unchanged':True,
 'restored_source_files':restored,'commands':commands,
 'scope':'Seven inert entry/module invocations and four selected synthetic interface cases only. No scientific assets, model calls, real job/process operations or network access.'}
(evidence/'checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'passed':result['passed'],'restored_files':len(restored),'inert_entries':7,'targeted_stdout':completed.stdout,
 'evidence':str(evidence)},indent=2))
raise SystemExit(completed.returncode)
