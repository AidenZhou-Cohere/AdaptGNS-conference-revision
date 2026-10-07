"""Inert complementary copy plan; root activates one exclusive result directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(value):
    path = Path(value)
    assert value not in ('', '.') and not path.is_absolute() and '..' not in path.parts
    return path


def read_plain(path, max_bytes=1_000_000):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < max_bytes
    return path.read_bytes()


def check_tree(directory, expected, manifest_name):
    assert directory.is_dir() and not directory.is_symlink()
    paths = list(directory.rglob('*'))
    assert not any(path.is_symlink() for path in paths)
    observed = {str(path.relative_to(directory)) for path in paths if path.is_file()}
    assert observed == set(expected) | {manifest_name}, 'Exact installed tree required'
    for name, row in expected.items():
        raw = read_plain(directory / relative(name))
        assert len(raw) == row['bytes'] and digest(raw) == row['sha256'], name


parser = argparse.ArgumentParser()
parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('publication_plan.json'))
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--copy', action='store_true')
mode.add_argument('--verify-installed', action='store_true')
args = parser.parse_args()
raw_plan = read_plain(args.plan)
plan = json.loads(raw_plan)
assert plan['schema'] == 'd3_final_closure_complementary_inert_publication_plan_v1'
root = Path(plan['workspace']).resolve()
repo = root / 'outputs/AdaptGNS'
parent = repo / 'research/results'
assert parent.is_dir() and not parent.is_symlink() and parent.resolve() == parent
target = repo / relative(plan['destination'])
base = repo / relative(plan['base_requirement']['destination'])
assert target.parent == base.parent == parent and target != base
assert target.name == 'goop3d_observed_history_failure_closure_20261007'
assert base.name == 'goop3d_observed_history_failure_20261007'
entries = plan['entries']
assert len(entries) == plan['file_count'] <= 150
assert sum(row['bytes'] for row in entries) == plan['total_bytes'] < 5_000_000
assert len({row['destination'] for row in entries}) == len(entries)
expected = {row['destination']:{'bytes':row['bytes'],'sha256':row['sha256']} for row in entries}
assert 'copy_provenance.json' not in expected and 'publication_manifest.json' not in expected
expected['copy_provenance.json'] = {'bytes':len(raw_plan),'sha256':digest(raw_plan)}

# Validate the installed predecessor directly; never run its old helper/preflight.
base_raw = read_plain(base / 'publication_manifest.json')
assert digest(base_raw) == plan['base_requirement']['publication_manifest_sha256']
base_manifest = json.loads(base_raw)
assert base_manifest['copy_plan_sha256'] == plan['base_requirement']['copy_plan_sha256']
assert len(base_manifest['files']) + 1 == plan['base_requirement']['files']
assert base_manifest['scientific_result_admission'] is False
check_tree(base, base_manifest['files'], 'publication_manifest.json')

if args.verify_installed:
    actual = json.loads(read_plain(target / 'publication_manifest.json'))
    assert actual['files'] == expected and actual['copy_plan_sha256'] == digest(raw_plan)
    assert actual['base_publication_manifest_sha256'] == digest(base_raw)
    assert actual['scientific_result_admission'] is False
    check_tree(target, expected, 'publication_manifest.json')
    print(json.dumps({'status':'passed_exact_complementary_and_predecessor_trees',
                      'files':len(expected)+1,'repository_changed':False}))
    raise SystemExit(0)

def git(*argv):
    return subprocess.run(['git','-C',str(repo),*argv],check=True,capture_output=True,text=True,timeout=10).stdout

assert git('rev-parse','HEAD').strip() == plan['expected_git_head']
assert git('branch','--show-current').strip() == 'research/conference-revision'
assert git('diff','--name-only','-z') == '' and git('diff','--cached','--name-only','-z') == ''
untracked = set(filter(None,git('ls-files','--others','--exclude-standard','-z').split('\0')))
base_names = {str((base / relative(name)).relative_to(repo)) for name in base_manifest['files']}
base_names.add(str((base / 'publication_manifest.json').relative_to(repo)))
assert untracked <= base_names, 'Only the exact original publication may be untracked'
assert not target.exists(), 'Preserve existing or partial output; use a separately reviewed successor'
sources = []
for row in entries:
    source = root / relative(row['source'])
    assert source.resolve().is_relative_to(root)
    raw = read_plain(source)
    assert len(raw) == row['bytes'] and digest(raw) == row['sha256']
    sources.append((row,source,relative(row['destination'])))
if not args.copy:
    print(json.dumps({'status':'passed_read_only_complementary_gate','files':len(entries),
                      'bytes':plan['total_bytes'],'repository_changed':False}))
    raise SystemExit(0)

# Exact new-directory copies only: no tracked-file overwrite, deletion, staging or push.
target.mkdir()
for row,source,dest in sources:
    raw = read_plain(source)
    assert len(raw) == row['bytes'] and digest(raw) == row['sha256']
    output = target / dest
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw)
    assert digest(read_plain(output)) == row['sha256']
with (target / 'copy_provenance.json').open('xb') as stream:
    stream.write(raw_plan)
manifest = {'schema':'d3_final_closure_complementary_publication_manifest_v1',
            'files':expected,'copy_plan_sha256':digest(raw_plan),
            'base_publication_manifest_sha256':digest(base_raw),
            'scientific_result_admission':False,'manifest_exclusions':['publication_manifest.json']}
with (target / 'publication_manifest.json').open('x') as stream:
    stream.write(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'copied_exact_complementary_closure_status_tree',
                  'files':len(expected)+1,'manifest_sha256':digest(read_plain(target/'publication_manifest.json')),
                  'tracked_file_overwrite_staging_commit_or_push':False}))
