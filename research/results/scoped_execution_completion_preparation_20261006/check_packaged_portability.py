"""Reconstruct only manifest-selected sources and run narrow synthetic checks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle',required=True,type=Path)
parser.add_argument('--evidence',required=True,type=Path)
args=parser.parse_args()
bundle=args.bundle.resolve(); evidence=args.evidence.resolve()
evidence.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
deps=json.loads((bundle/'dependency_paths.json').read_text())
assert sha((bundle/deps['predecessor_manifest']).resolve())==deps['predecessor_manifest_sha256']
sources=json.loads((bundle/'source_inventory.json').read_text())
pins=json.loads((bundle/'current_approved_pins.json').read_text())
temp=Path(tempfile.mkdtemp(prefix='scoped_preparation_portability_')).resolve()
restored=[]
for relative,record in deps['files'].items():
    source=(bundle/record['published_path']).resolve(); target=temp/relative
    assert sha(source)==record['sha256']
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    restored.append({'path':relative,'sha256':record['sha256'],'kind':'predecessor'})
for relative,record in sources.items():
    source=bundle/relative; assert sha(source)==record['sha256']
    if relative.startswith('workspace/'):
        target=temp/relative.removeprefix('workspace/')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        restored.append({'path':relative.removeprefix('workspace/'),'sha256':record['sha256'],'kind':'overlay'})
runtime=temp/'work/deadline_research_20261005/cuda_preparation'
primaries=[name for name in pins if name.endswith('.py')]
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(runtime)
commands=[]
for name in primaries:
    compile((runtime/name).read_text(),name,'exec')
    command=[sys.executable,'-B',str(runtime/name)]
    completed=subprocess.run(command,cwd=temp,env=env,capture_output=True,text=True,timeout=30)
    (evidence/(name+'.stdout')).write_text(completed.stdout)
    (evidence/(name+'.stderr')).write_text(completed.stderr)
    commands.append({'kind':'inert_default_import','entry':name,'exit_code':completed.returncode,
                     'stdout_sha256':sha(evidence/(name+'.stdout')),'stderr_sha256':sha(evidence/(name+'.stderr'))})
    assert completed.returncode==0, name+': '+completed.stderr
selectors=[
 'test_supervise_goop3d_final_evaluation_v1.py::test_default_never_loads_or_launches',
 'test_supervise_goop3d_final_evaluation_v1.py::test_symlink_interpreter_preserves_venv_argv_and_binds_target_and_config',
 'test_supervise_sand_scoped_science_v1.py::test_default_never_configures_or_launches',
 'test_supervise_sand_scoped_science_v1.py::test_symlink_interpreter_preserves_venv_argv_and_records_environment',
 'test_summarize_goop_graph_support_scoped_v3.py::test_relocated_copy_preserves_original_paths_and_bytes',
 'test_summarize_goop_graph_support_scoped_v3.py::test_origin_inventory_and_artifact_escapes_rejected',
]
completed=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*selectors],
    cwd=runtime,env=env,capture_output=True,text=True,timeout=90)
(evidence/'targeted.stdout').write_text(completed.stdout);(evidence/'targeted.stderr').write_text(completed.stderr)
commands.append({'kind':'targeted_packaged_portability','selectors':selectors,'exit_code':completed.returncode,
    'stdout_sha256':sha(evidence/'targeted.stdout'),'stderr_sha256':sha(evidence/'targeted.stderr')})
after={relative:sha(bundle/relative) for relative in sources}
assert all(after[p]==v['sha256'] for p,v in sources.items())
result={'schema':'adaptgns_scoped_preparation_packaged_portability_v1','passed':completed.returncode==0,
 'bundle':str(bundle),'temporary_restored_workspace':str(temp),'interpreter':sys.executable,
 'source_files_unchanged':True,'manifest_selected_copies':restored,'commands':commands,
 'scope':'Source imports/inert default entries and six selected synthetic portability tests (one parameterized); no research processes, models, scientific arrays or remote access.'}
(evidence/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'default_entries':len(primaries),'targeted_stdout':completed.stdout,'evidence':str(evidence)},indent=2))
raise SystemExit(completed.returncode)
