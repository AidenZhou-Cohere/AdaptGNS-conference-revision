"""Publish a new source-only Sand completion overlay; never execute research."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args();root=args.workspace.resolve()
    relative=Path('work/deadline_research_20261005/cuda_preparation');src=root/relative
    families=root/'outputs/AdaptGNS/research/results'
    out=families/'sand_scoped_completion_preparation_20261006'
    assert not out.exists(), 'Refuse to replace a published or partial package'
    pins={
        'prepare_sand_final_cohort_scoped_v1.py':'ebad4e30edbdb9f444ed07dabf50b31540ecf7f85b134ee059c150d1ff68ca5f',
        'prepare_sand_reserved_test_scoped_v1.py':'98b41509fa253a86630cf44ebc86bd3f0b969e875713b736867727bdc392b0bd',
        'supervise_sand_final_evaluation_scoped_v1.py':'efebe762ea60b1b711925fb9bb3bc91461a16b1440491b2fa554c99171773829',
        'summarize_sand_graph_support_scoped_v1.py':'1e70f1689a75c6f68eb970048aab7837153c19f70caf30ec19143cf66ab5d272',
    }
    record=json.loads((src/'sand_scoped_final_pipeline_authoring_record_v1.json').read_text())
    copied_dependencies={
        'sand_graph_support_100k_protocol_v1.md':'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d',
        'sand_train_admission.json':'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73',
    }
    current=set(pins)|set(copied_dependencies)|{'sand_scoped_final_pipeline_authoring_record_v1.json',
        'sand_scoped_scalar_release_interface_v1.md','test_summarize_sand_graph_support_scoped_v1.py'}
    for kind in ('sources_sha256','tests_sha256','templates_notes_sha256','independent_reviews_sha256'):
        for name,digest in record[kind].items():
            assert sha(src/name)==digest, name
            current.add(name)
    for name,digest in {**pins,**copied_dependencies}.items(): assert sha(src/name)==digest,name
    selected={'workspace/'+str(relative/name):(src/name,'copied_scientific_dependency' if name in copied_dependencies else 'current_preparation') for name in current}
    prefixes=('sand_final_cohort_','sand_final_evaluation_','sand_reserved_test_','sand_scoped_scalar_')
    for p in sorted(src.iterdir()):
        if p.is_file() and p.name.startswith(prefixes) and p.suffix in ('.json','.log') and p.name not in current:
            selected['reviews/'+p.name]=(p,'review_or_unsuccessful_preparation')
    for name in ('sand_scoped_scalar_review_history','sand_scoped_final_pipeline_review_history','sand_final_pipeline_review_history'):
        for p in sorted((src/name).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.md','.json','.txt','.log') and '__pycache__' not in p.parts:
                selected['review_history/'+str(p.relative_to(src))]=(p,'historical_candidate_not_approved')
    # Include the independent probe sources named by the review records.
    for p in sorted(src.glob('audit_sand_*')):
        if p.suffix=='.py' and ('final_cohort' in p.name or 'reserved_test' in p.name):
            selected['reviews/'+p.name]=(p,'independent_synthetic_probe_source')
    omitted={}
    for destination,(p,kind) in list(selected.items()):
        assert p.is_file() and not p.is_symlink(),p
        if p.stat().st_size>=250000:
            assert p.suffix=='.log' and kind=='review_or_unsuccessful_preparation',p
            omitted[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size,
                'reason':'Oversized raw test log retained in original workspace; compact review and failures remain included.'}
            del selected[destination]
            continue
        assert not any(x in p.name.lower() for x in ('ssh_config','credential','.pt','.npz','.tfrecord')),p
        text=p.read_text()
        assert not re.search(r'-----BEGIN .*PRIVATE KEY-----|\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b',text),p
        if p.suffix=='.json' and text.strip(): json.loads(text)
    ancestors={
        'goop3d_vectorized_preparation_20261006':'98624ef49322ebcc28ed92561f612d0effbd7719df8e3a160f75c69bdb3b2116',
        'scoped_execution_completion_preparation_20261006':'52f469281184b2bb8c42bb3381bcb7910c3d379689353ecfd04f5893974e3db7',
    }
    dependencies={}
    for family,digest in ancestors.items():
        prior=families/family;assert sha(prior/'manifest.json')==digest
        manifest=json.loads((prior/'manifest.json').read_text())
        for name,v in manifest['files'].items():
            if name.startswith('workspace/'):
                assert sha(prior/name)==v['sha256'],name
                dependencies[name.removeprefix('workspace/')]={'published_path':'../'+family+'/'+name,'sha256':v['sha256'],'bytes':v['bytes']}
    out.mkdir();inventory={}
    for destination,(p,kind) in sorted(selected.items()):
        q=out/destination;q.parent.mkdir(parents=True,exist_ok=True)
        before=sha(p);shutil.copyfile(p,q);assert sha(p)==sha(q)==before
        inventory[destination]={'source':str(p.relative_to(root)),'kind':kind,'sha256':before,'bytes':p.stat().st_size}
    write(out/'source_inventory.json',inventory);write(out/'current_approved_pins.json',pins)
    write(out/'large_local_log_references.json',omitted)
    write(out/'dependency_paths.json',{'predecessor_manifests':ancestors,'files':dependencies,
        'order':'Restore these exact effective predecessor workspace paths, then overlay this package workspace into a new directory.'})
    shutil.copyfile(Path(__file__),out/'build_package.py')
    print(json.dumps({'output':str(out),'selected_files':len(inventory),'predecessor_workspace_files':len(dependencies)},indent=2))

if __name__=='__main__':main()
