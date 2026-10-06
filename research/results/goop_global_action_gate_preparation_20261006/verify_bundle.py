"""Read-only gate preparation package and dependency byte verification."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ORDER=('goop3d_vectorized_preparation_20261006','scoped_execution_completion_preparation_20261006',
       'sand_scoped_completion_preparation_20261006')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def safe(root,name):
 rel=PurePosixPath(name)
 require(bool(name) and not rel.is_absolute() and str(rel)==name and all(x not in ('.','..') for x in rel.parts),'Unsafe relative path')
 path=root/rel
 require(path.is_file() and path.resolve().is_relative_to(root.resolve()) and
  all(not q.is_symlink() for q in [path,*path.parents] if q==root or root in q.parents),'Missing or linked selected file')
 return path
def check(path,record):require(path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],'Bytes differ: '+str(path))

def verify(root,original=None):
 manifest=json.loads(safe(root,'manifest.json').read_text())
 require(manifest['schema']=='adaptgns_goop_gate_preparation_manifest_v1','Unexpected manifest schema')
 require(not any(p.is_symlink() for p in root.rglob('*')),'Linked package entry')
 files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
 require(files==set(manifest['files'])|{'manifest.json'},'Unexpected package file set')
 for name,record in manifest['files'].items():check(safe(root,name),record)
 require((root/'FILELIST.txt').read_text().splitlines()==sorted(files),'File list differs')
 inventory=json.loads((root/'source_inventory.json').read_text())
 require(set(inventory)=={n for n in files if n.startswith(('workspace/','reviews/','review_history/'))},'Source inventory differs')
 for name,record in inventory.items():
  check(safe(root,name),record)
  if original:check(safe(original,record['source']),record)
 prefix='workspace/work/deadline_research_20261005/cuda_preparation/'
 pins=json.loads((root/'current_approved_pins.json').read_text())
 for name,pin in pins.items():require(sha(safe(root,prefix+name))==pin,'Approved source differs: '+name)
 for name,pin in json.loads((root/'current_authoring_records.json').read_text()).items():
  require(sha(safe(root,'reviews/'+name))==pin,'Current authoring record differs: '+name)
 scope=json.loads((root/'scope.json').read_text())
 require(scope['science_execution_or_test_admission_granted'] is False,'Source preparation cannot grant admission')
 require(set(scope['current_primary_python_entries'])=={n for n in pins if n.endswith('.py')},'Entry list differs')
 deps=json.loads((root/'dependency_paths.json').read_text());effective={}
 require(deps['predecessor_order']==list(ORDER) and set(deps['predecessor_manifests'])==set(ORDER),'Predecessor order/set differs')
 for family in ORDER:
  prior=root.parent/family;mp=safe(prior,'manifest.json')
  require(sha(mp)==deps['predecessor_manifests'][family],'Predecessor manifest differs')
  for name,record in json.loads(mp.read_text())['files'].items():
   if name.startswith('workspace/'):
    check(safe(prior,name),record)
    effective[name.removeprefix('workspace/')]=dict(published_path='../'+family+'/'+name,sha256=record['sha256'],bytes=record['bytes'])
 require(deps['files']==effective,'Effective predecessor closure differs')
 require(manifest['file_count_excluding_manifest']==len(manifest['files']),'Manifest count differs')
 require(manifest['bytes_excluding_manifest']==sum(v['bytes'] for v in manifest['files'].values()),'Manifest byte count differs')
 return dict(passed=True,manifest_sha256=sha(root/'manifest.json'),package_files=len(files),
  package_bytes=sum((root/n).stat().st_size for n in files),copied_source_files=len(inventory),approved_pins=len(pins),
  effective_predecessor_files=len(effective),original_selected_sources_rechecked=original is not None,research_executed=False)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original-workspace',type=Path)
 args=parser.parse_args();print(json.dumps(verify(Path(__file__).resolve().parent,args.original_workspace),indent=2))
