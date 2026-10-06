"""Curate the exact reviewed gate pipeline without reading scientific assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

PINS = {
 'goop_global_action_gate_protocol_v1.md':'bc9235fae89d32ede8d1e7f846bff07bdcdd85f4671d5922d688a9535f7fe024',
 'goop_global_action_gate_core_v1.py':'d3986c2ac3786e9fc0d36d76339bfeded5448497f7dad9b0559efaaaa53a43ca',
 'run_goop_action_gate_v1.py':'af51bbe073e52a22d4394918dbce4bc9c34a99645dc7f302b0cd5c48ebba5079',
 'fit_goop_action_gate_v1.py':'85411dd07972fd555da86210c8bf1825d31801c6d0b168e9f8106f2492ea1ee3',
 'goop_action_gate_cost_plan_v1.py':'b4b7a18da848b7222521eecddf4f612e0a0615dd4d990ea65c35741cf1c8a28b',
 'supervise_goop_action_gate_scoped_v1.py':'4cda31a408c097944e5f15b3502d05bcfab0d3db5b8ee25dcc5fdc1968fd61df',
 'goop_action_gate_scalar_core_v1.py':'a0c0098e72c073b3468a2cc9a0d7a23c723250148f1dc1a550f35e02425081b1',
 'summarize_goop_action_gate_v1.py':'2f1f5fa7085be29d573aadd098eedb764259d06dde79308526052127966eb85f',
}
PREDECESSORS = {
 'goop3d_vectorized_preparation_20261006':'98624ef49322ebcc28ed92561f612d0effbd7719df8e3a160f75c69bdb3b2116',
 'scoped_execution_completion_preparation_20261006':'52f469281184b2bb8c42bb3381bcb7910c3d379689353ecfd04f5893974e3db7',
 'sand_scoped_completion_preparation_20261006':'84922d3b6f70d9f7c12eb96906577d02cc1d18065f362105ee7fe9a26e96ef3c',
}
HISTORY_DIRS = (
 'goop_global_action_gate_core_review_history',
 'goop_action_gate_driver_review_history',
 'goop_action_gate_fit_review_history',
 'goop_action_gate_cost_review_history',
 'goop_action_gate_scalar_core_review_history',
 'goop_action_gate_collector_authoring_history',
)
EXPLICIT_HISTORY = {
 'goop_action_gate_supervisor_reused_pid_draft_fixture_v1/reviewed_source.py':'c7a3f407',
 'goop_action_gate_supervisor_selection_draft_fixture_v1/reviewed_source.py':'6389acc9',
 'goop_action_gate_scalar_collector_draft_probes_code_audit_v1/summarize_goop_action_gate_v1.py':'b53bd70f',
 'goop_action_gate_scalar_collector_draft_probes_code_audit_v1/goop_action_gate_scalar_core_v1.py':'75b4a8be',
 'goop_action_gate_driver_row_snapshot_probe_v1/reviewed_source.py':'082fe7af',
 'goop_action_gate_driver_independent_publication_fixtures_v2/reviewed_source.py':'cbbb2d18',
}
AUTHOR_RECORDS = (
 'goop_action_gate_operational_authoring_record_v1.json',
 'goop_action_gate_driver_authoring_record_v2.json',
 'goop_action_gate_scalar_core_authoring_record_v1.json',
 'goop_action_gate_collector_authoring_record_v1.json',
)
FAMILY = 'goop_global_action_gate_preparation_20261006'
REL = Path('work/deadline_research_20261005/cuda_preparation')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--workspace',required=True,type=Path)
 args=parser.parse_args();root=args.workspace.resolve();src=root/REL
 out=root/'outputs/AdaptGNS/research/results'/FAMILY
 assert not out.exists(),'Refuse to overwrite an existing family'
 for name,pin in PINS.items(): assert sha(src/name)==pin,'Reviewed pin changed: '+name
 # Only explicit final author records bind current versions. Earlier records
 # remain history and may correctly refer to superseded candidate hashes.
 author_pins={}
 for name in AUTHOR_RECORDS:
  record=json.loads((src/name).read_text());author_pins[name]=sha(src/name)
  for key in ('files_sha256','reviews_sha256'):
   for path,pin in record.get(key,{}).items():assert sha(src/path)==pin,path
  for item in record.get('artifacts',[]):
   path=Path(item['path']);assert path.parent==src and sha(path)==item['sha256'],str(path)
 selection={};omitted_logs={};local_trees=[]
 for path in sorted(src.iterdir()):
  if 'action_gate' not in path.name:continue
  if path.is_dir():
   if path.name not in HISTORY_DIRS:
    local_trees.append({'source':str(path.relative_to(root)),
     'reason':'Detailed synthetic fixture tree not copied; compact review/probe records and their source are retained.',
     'fixture_payloads_read_for_curation':False,
     'selected_source_snapshots':[n for n in EXPLICIT_HISTORY if n.startswith(path.name+'/')]})
   continue
  assert path.is_file() and not path.is_symlink(),str(path)
  if path.suffix not in ('.py','.md','.json','.log'):continue
  if path.stat().st_size>200_000:
   assert path.suffix=='.log','Unexpected large selected source'
   omitted_logs[str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':sha(path),'reason':'Oversized raw review log; retained locally'}
   continue
  if path.suffix=='.py' or (path.suffix=='.md' and not path.name.startswith('constructive_')) or path.name.endswith('.template.json') or path.name=='goop_global_action_gate_prospective_freeze_v1.json':
   name='workspace/'+str(REL/path.name)
   kind=('current_approved_pipeline_source_or_protocol' if path.name in PINS else
         'synthetic_test_or_review_source' if path.suffix=='.py' else
         'prospective_interface_template_or_plan')
  else:
   name='reviews/'+path.name;kind='review_authoring_or_preserved_unsuccessful_probe'
  selection[name]=(path,kind)
 for directory in HISTORY_DIRS:
  for path in sorted((src/directory).rglob('*')):
   if path.is_file() and path.suffix in ('.py','.md','.json','.txt') and '__pycache__' not in path.parts:
    selection['review_history/'+str(path.relative_to(src))]=(path,'historical_candidate_not_current_approved_source')
 for name,prefix in EXPLICIT_HISTORY.items():
  path=src/name;assert sha(path).startswith(prefix),name
  selection['review_history/'+name]=(path,'historical_candidate_not_current_approved_source')
 for name,(path,kind) in selection.items():
  assert not path.is_symlink() and path.stat().st_size<200_000,name
  raw=path.read_text()
  assert not re.search(r'-----BEGIN .*PRIVATE KEY-----|\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b',raw),name
  if path.suffix=='.json' and raw:json.loads(raw)
 effective={}
 for family,pin in PREDECESSORS.items():
  prior=out.parent/family;manifest=prior/'manifest.json';assert sha(manifest)==pin
  for name,record in json.loads(manifest.read_text())['files'].items():
   if name.startswith('workspace/'):
    assert sha(prior/name)==record['sha256']
    effective[name.removeprefix('workspace/')]=dict(published_path='../'+family+'/'+name,sha256=record['sha256'],bytes=record['bytes'])
 out.mkdir();inventory={}
 for name,(path,kind) in selection.items():
  target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
  before=sha(path);shutil.copyfile(path,target);assert sha(path)==before==sha(target)
  inventory[name]=dict(source=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=before,kind=kind)
 write(out/'source_inventory.json',inventory)
 write(out/'current_approved_pins.json',PINS)
 write(out/'current_authoring_records.json',author_pins)
 write(out/'dependency_paths.json',{'predecessor_manifests':PREDECESSORS,'predecessor_order':list(PREDECESSORS),
  'files':effective,'order':'Restore exact effective predecessor files, then overlay this family workspace into a new directory.'})
 write(out/'large_local_log_references.json',{'files':omitted_logs,'note':'No selected oversized review logs were copied.'})
 write(out/'omitted_synthetic_fixture_tree_references.json',{'directories':local_trees,
  'scope':'Detailed fixture payloads were not recursively read or copied. Only explicitly listed historical source snapshots were read; packaged primary pytest fixtures generate fresh synthetic inputs.'})
 write(out/'historical_source_limits.json',{'unlocated_candidate_sources':[{
  'name':'summarize_goop_action_gate_v1.py',
  'sha256':'d55aca8943625c6a4afd99d86412ddbce99e1e2ad73a0562dbf2df46e3c0adb1',
  'reference':'reviews/goop_action_gate_stopped_collection_independent_review_code_audit_v1.json',
  'scope':'This intermediate hash is preserved in the review; no matching source snapshot was found in the bounded historical source selection. Current final approved source is included.'}]})
 write(out/'scope.json',{'kind':'reviewed_source_preparation_not_scientific_outcomes',
  'current_primary_python_entries':[n for n in PINS if n.endswith('.py')],
  'science_execution_or_test_admission_granted':False,'frozen_sources_changed':False,
  'excluded':['actual data','model/checkpoint payloads','live output trees','private process inventories','SSH configuration','credentials','detailed synthetic fixture payload trees','oversized raw logs'],
  'historical_approval_rule':'Only current_approved_pins.json names the approved pipeline. Earlier author/review notes and candidate snapshots remain historical, including then-pending work.'})
 shutil.copyfile(Path(__file__),out/'curate_sources.py')
 print(json.dumps({'family':str(out),'copied_files':len(inventory),'copied_bytes':sum(v['bytes'] for v in inventory.values()),
  'effective_predecessor_files':len(effective),'omitted_large_logs':len(omitted_logs),'referenced_fixture_trees':len(local_trees)},indent=2))

if __name__=='__main__':main()
