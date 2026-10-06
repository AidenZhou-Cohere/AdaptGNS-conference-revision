"""Reconstruct only published source; run bounded inert interface checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bundle=a.bundle.resolve();assert not a.output.exists();a.output.mkdir()
    deps=json.loads((bundle/'dependency_paths.json').read_text());record={'scope':'Narrow reconstructed-source portability checks; no real data, model or remote process access','checks':[]}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='sand-source-overlay-') as d:
        root=Path(d)
        for name,v in deps['files'].items():
            source=bundle/v['published_path'];assert sha(source)==v['sha256']
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        for source in (bundle/'workspace').rglob('*'):
            if source.is_file():
                dest=root/source.relative_to(bundle/'workspace');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        original={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
        working=root/'work/deadline_research_20261005/cuda_preparation'
        pins=json.loads((bundle/'current_approved_pins.json').read_text())
        for name in pins:
            result=subprocess.run([sys.executable,str(working/name)],cwd=root,env=env,capture_output=True,text=True,timeout=60)
            (a.output/(name+'.stdout.log')).write_text(result.stdout);(a.output/(name+'.stderr.log')).write_text(result.stderr)
            record['checks'].append({'entry':name,'mode':'default_nonexecuting','exit_code':result.returncode})
            assert result.returncode==0,result.stderr
        cases=[
            'test_prepare_sand_final_cohort_scoped_v1.py::test_frozen_evaluator_contract',
            'test_prepare_sand_reserved_test_scoped_v1.py::test_candidate_root_sign_preflight_uses_frozen_evaluator',
            'test_supervise_sand_final_evaluation_scoped_v1.py::test_frozen_sand_cohort_split_and_scope_contracts',
            'test_summarize_sand_graph_support_scoped_v1.py::test_relocated_copy_preserves_original_paths_and_bytes',
        ]
        command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*cases]
        result=subprocess.run(command,cwd=working,env=env,capture_output=True,text=True,timeout=90)
        (a.output/'interfaces.stdout.log').write_text(result.stdout);(a.output/'interfaces.stderr.log').write_text(result.stderr)
        record['checks'].append({'tests':cases,'exit_code':result.returncode,'summary':result.stdout.strip()})
        assert result.returncode==0,result.stdout+result.stderr
        assert all(sha(root/n)==digest for n,digest in original.items())
        record['original_files_unchanged']=len(original)
    record['status']='passed';(a.output/'packaging_checks.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
