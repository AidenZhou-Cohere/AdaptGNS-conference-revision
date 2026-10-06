"""Verify a fixed local manifest; --copy exclusively creates its fresh destination."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(value):
    path = Path(value)
    assert not path.is_absolute() and '..' not in path.parts and value not in ('', '.')
    return path


parser = argparse.ArgumentParser()
parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('publication_plan.json'))
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--copy', action='store_true', help='Root activation: exclusively create the fixed repository destination.')
mode.add_argument('--verify-installed', action='store_true')
args = parser.parse_args()
plan_bytes = args.plan.read_bytes()
plan = json.loads(plan_bytes)
assert plan['schema'] == 'sand_inert_publication_copy_plan_v1'
root = Path(plan['workspace']).resolve()
repo = Path(plan['repository']).resolve()
assert repo == root / 'outputs/AdaptGNS'
relative_target = safe_relative(plan['destination'])
assert relative_target.parts[:2] == ('research', 'results')
target = repo / relative_target
assert target.parent.is_dir() and not target.parent.is_symlink()
assert target.resolve().is_relative_to(repo / 'research/results')
entries = plan['entries']
assert len(entries) == plan['file_count'] <= 400
assert sum(entry['bytes'] for entry in entries) == plan['total_bytes'] <= 40_000_000
assert len({entry['destination'] for entry in entries}) == len(entries)
assert all(entry['bytes'] < 12_000_000 for entry in entries)


def expected_sources():
    sources = []
    for entry in entries:
        source = root / safe_relative(entry['source'])
        destination = safe_relative(entry['destination'])
        assert source.is_file() and not source.is_symlink()
        assert source.resolve().is_relative_to(root)
        assert source.stat().st_size == entry['bytes'] and sha(source) == entry['sha256'], str(source)
        sources.append((entry, source, destination))
    return sources


def verify_aliases(installed=False):
    alias_entry = next(entry for entry in entries if entry['destination'] == 'history/prior_generation_byte_aliases.json')
    alias_path = target / alias_entry['destination'] if installed else root / alias_entry['source']
    aliases = json.loads(alias_path.read_text())['aliases']
    assert len(aliases) == plan['prior_generation_alias_count']
    assert sum(item['bytes'] for item in aliases) == plan['prior_generation_alias_bytes_saved']
    by_destination = {entry['destination']: entry for entry in entries}
    for item in aliases:
        entry = by_destination[item['exact_published_bytes']]
        assert (item['bytes'], item['sha256']) == (entry['bytes'], entry['sha256'])
        if not installed:
            source = root / safe_relative(item['source'])
            assert source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(root)
            assert source.stat().st_size == item['bytes'] and sha(source) == item['sha256']


if args.verify_installed:
    assert target.is_dir() and not target.is_symlink()
    manifest = json.loads((target / 'publication_manifest.json').read_text())
    assert manifest['copy_plan_sha256'] == hashlib.sha256(plan_bytes).hexdigest()
    expected = {entry['destination'] for entry in entries} | {'copy_provenance.json', 'publication_manifest.json'}
    actual = {str(path.relative_to(target)) for path in target.rglob('*') if path.is_file()}
    assert actual == expected
    assert not any(path.is_symlink() for path in target.rglob('*'))
    for entry in entries:
        path = target / safe_relative(entry['destination'])
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    assert (target / 'copy_provenance.json').read_bytes() == plan_bytes
    expected_manifest_files = {entry['destination']: {'bytes': entry['bytes'], 'sha256': entry['sha256']} for entry in entries}
    expected_manifest_files['copy_provenance.json'] = {'bytes': len(plan_bytes), 'sha256': hashlib.sha256(plan_bytes).hexdigest()}
    assert manifest['files'] == expected_manifest_files
    verify_aliases(installed=True)
    print(json.dumps({'status': 'passed_installed_exact_file_set_and_hashes', 'files': len(expected), 'repository_changed': False}))
    raise SystemExit(0)

for argv, expected in ((['rev-parse', 'HEAD'], plan['expected_git_head']),
                       (['branch', '--show-current'], plan['expected_git_branch']),
                       (['status', '--porcelain'], '')):
    observed = subprocess.run(['git', '-C', str(repo), *argv], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    assert observed == expected, ('repository precondition', argv, observed)
assert not target.exists(), 'Preserve any previous/partial publication directory; use a reviewed successor plan.'
sources = expected_sources()
verify_aliases()
if not args.copy:
    print(json.dumps({'status': 'passed_inert_copy_preflight', 'files': len(entries),
                      'bytes': plan['total_bytes'], 'destination': str(target), 'repository_changed': False}))
    raise SystemExit(0)

# No overwrite, deletion or cleanup on error. Any partial directory remains
# available for inspection and requires a reviewed successor decision.
target.mkdir()
for entry, source, destination in sources:
    output = target / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
    with output.open('xb') as stream:
        stream.write(raw)
    assert sha(output) == entry['sha256']
with (target / 'copy_provenance.json').open('xb') as stream:
    stream.write(plan_bytes)
manifest_files = {entry['destination']: {'bytes': entry['bytes'], 'sha256': entry['sha256']} for entry in entries}
manifest_files['copy_provenance.json'] = {'bytes': len(plan_bytes), 'sha256': hashlib.sha256(plan_bytes).hexdigest()}
manifest = {'schema': 'sand_exact_publication_manifest_v1', 'files': manifest_files,
            'copy_plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
            'source_snapshot_may_precede_later_addenda': True,
            'no_scientific_reexecution': True, 'manifest_exclusions': ['publication_manifest.json']}
with (target / 'publication_manifest.json').open('x') as stream:
    stream.write(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': 'copied_and_verified_exact_bounded_snapshot', 'files': len(entries) + 2,
                  'destination': str(target), 'manifest_sha256': sha(target / 'publication_manifest.json'),
                  'no_commit_or_push_performed': True}))
