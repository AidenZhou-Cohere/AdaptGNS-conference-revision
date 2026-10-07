"""Read-only by default; --copy exclusively creates one fixed local result tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(value):
    path = Path(value)
    assert not path.is_absolute() and '..' not in path.parts and value not in ('', '.')
    return path


parser = argparse.ArgumentParser()
parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('publication_plan.json'))
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--copy', action='store_true')
mode.add_argument('--verify-installed', action='store_true')
args = parser.parse_args()
raw_plan = args.plan.read_bytes()
plan = json.loads(raw_plan)
assert plan['schema'] == 'd3_failed_observed_scope_inert_publication_plan_v1'
root = Path(plan['workspace']).resolve()
repo = root / 'outputs/AdaptGNS'
target = repo / relative(plan['destination'])
assert target.parent == repo / 'research/results' and target.parent.is_dir() and not target.parent.is_symlink()
entries = plan['entries']
assert len(entries) == plan['file_count'] <= 300
assert sum(item['bytes'] for item in entries) == plan['total_bytes'] <= 5_000_000
assert len({item['destination'] for item in entries}) == len(entries)
expected = {item['destination']: {'bytes': item['bytes'], 'sha256': item['sha256']} for item in entries}
expected['copy_provenance.json'] = {'bytes': len(raw_plan), 'sha256': hashlib.sha256(raw_plan).hexdigest()}

if args.verify_installed:
    assert target.is_dir() and not target.is_symlink()
    manifest = json.loads((target / 'publication_manifest.json').read_text())
    assert manifest['files'] == expected
    assert manifest['copy_plan_sha256'] == hashlib.sha256(raw_plan).hexdigest()
    assert {str(path.relative_to(target)) for path in target.rglob('*') if path.is_file()} == set(expected) | {'publication_manifest.json'}
    assert not any(path.is_symlink() for path in target.rglob('*'))
    for name, record in expected.items():
        path = target / relative(name)
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
    print(json.dumps({'status': 'passed_installed_exact_tree', 'files': len(expected) + 1, 'repository_changed': False}))
    raise SystemExit(0)

for argv, value in ((['rev-parse', 'HEAD'], plan['expected_git_head']),
                    (['branch', '--show-current'], 'research/conference-revision'),
                    (['status', '--porcelain'], '')):
    observed = subprocess.run(['git', '-C', str(repo), *argv], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    assert observed == value, ('repository precondition differs', argv, observed)
assert not target.exists(), 'Preserve an existing or partial destination; use a reviewed successor plan.'
sources = []
for item in entries:
    source = root / relative(item['source'])
    dest = relative(item['destination'])
    assert source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(root)
    assert source.stat().st_size == item['bytes'] < 1_000_000 and sha(source) == item['sha256']
    sources.append((item, source, dest))
if not args.copy:
    print(json.dumps({'status': 'passed_inert_publication_preflight', 'files': len(entries),
                      'bytes': plan['total_bytes'], 'repository_changed': False}))
    raise SystemExit(0)

# No overwrites or deletion: any partial output is retained on failure.
target.mkdir()
for item, source, dest in sources:
    raw = source.read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    output = target / dest
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw)
    assert sha(output) == item['sha256']
with (target / 'copy_provenance.json').open('xb') as stream:
    stream.write(raw_plan)
manifest = {'schema': 'd3_failed_observed_scope_publication_manifest_v1', 'files': expected,
            'copy_plan_sha256': hashlib.sha256(raw_plan).hexdigest(),
            'scientific_result_admission': False, 'original_proxy_closure_separate_addendum': True,
            'manifest_exclusions': ['publication_manifest.json']}
with (target / 'publication_manifest.json').open('x') as stream:
    stream.write(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': 'copied_verified_failed_outcome_source_snapshot',
                  'files': len(entries) + 2, 'manifest_sha256': sha(target / 'publication_manifest.json'),
                  'commit_or_push_performed': False}))
