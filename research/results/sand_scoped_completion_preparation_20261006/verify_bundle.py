"""Read-only verification of packaged Sand preparation and predecessor bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)
def safe(root,name):
    rel=PurePosixPath(name)
    require(not rel.is_absolute() and str(rel)==name and all(x not in ('.','..') for x in rel.parts),'Unsafe package path')
    p=root/rel
    require(p.is_file() and p.resolve().is_relative_to(root.resolve()) and
        all(not q.is_symlink() for q in [p,*p.parents] if q==root or root in q.parents),'Missing or linked package file')
    return p
def check(p,v): require(p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'Bytes differ: '+str(p))

def verify(root,original=None):
    manifest=json.loads((root/'manifest.json').read_text())
    files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(files==set(manifest['files'])|{'manifest.json'},'Unexpected file set')
    for name,v in manifest['files'].items():check(safe(root,name),v)
    require((root/'FILELIST.txt').read_text().splitlines()==sorted(files),'File list differs')
    inventory=json.loads((root/'source_inventory.json').read_text())
    require(set(inventory)=={n for n in files if n.startswith(('workspace/','reviews/','review_history/'))},'Source inventory differs')
    for name,v in inventory.items():
        check(safe(root,name),v)
        if original:check(safe(original,v['source']),v)
    pins=json.loads((root/'current_approved_pins.json').read_text())
    for name,digest in pins.items():require(sha(safe(root,'workspace/work/deadline_research_20261005/cuda_preparation/'+name))==digest,'Approved source differs')
    deps=json.loads((root/'dependency_paths.json').read_text());effective={}
    # The insertion order matters: the later scoped package supersedes old helpers.
    for family in ('goop3d_vectorized_preparation_20261006','scoped_execution_completion_preparation_20261006'):
        require(family in deps['predecessor_manifests'],'Required predecessor omitted')
        prior=root.parent/family;mp=safe(prior,'manifest.json')
        require(sha(mp)==deps['predecessor_manifests'][family],'Predecessor manifest differs')
        for name,v in json.loads(mp.read_text())['files'].items():
            if name.startswith('workspace/'):
                check(safe(prior,name),v)
                effective[name.removeprefix('workspace/')]={'published_path':'../'+family+'/'+name,'sha256':v['sha256'],'bytes':v['bytes']}
    require(deps['files']==effective,'Effective predecessor closure differs')
    return {'status':'passed','package_files':len(files),'original_copies':len(inventory),'effective_predecessor_files':len(effective),'research_executed':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original-workspace',type=Path)
    args=p.parse_args();print(json.dumps(verify(Path(__file__).resolve().parent,args.original_workspace),indent=2))
