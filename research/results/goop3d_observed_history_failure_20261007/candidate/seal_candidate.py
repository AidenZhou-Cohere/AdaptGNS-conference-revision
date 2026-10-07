"""Seal exact compact failure/source copies and metadata projection only."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


inventory = json.loads((OUT / 'copy_inventory.json').read_text())
for name, record in inventory.items():
    assert (OUT / name).stat().st_size == record['bytes']
    assert sha(OUT / name) == record['sha256'] == sha(ROOT / record['source'])
projection = json.loads((OUT / 'evidence/failure_observation_metadata_projection.json').read_text())
source = ROOT / projection['source']
assert sha(source) == projection['source_sha256']
observation = json.loads(source.read_text())
assert projection['metadata'] == {key: observation[key] for key in projection['selected_keys']}
assert projection['excluded_fields'] == sorted(set(observation) - set(projection['selected_keys']))
assert not any(path.is_symlink() for path in OUT.rglob('*'))
excluded = {'candidate_manifest.json', 'FILELIST.txt', 'sealing_verification.json'}
names = {str(path.relative_to(OUT)) for path in OUT.rglob('*')
         if path.is_file() and str(path.relative_to(OUT)) not in excluded}
names.add('FILELIST.txt')
(OUT / 'FILELIST.txt').write_text(''.join(name + '\n' for name in sorted(names | {'candidate_manifest.json', 'sealing_verification.json'})))
files = {name: {'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in sorted(names)}
write(OUT / 'candidate_manifest.json', {
    'schema': 'd3_observed_failure_final_publication_candidate_manifest_v1',
    'status': 'FAILED_OBSERVED_SCOPE_NO_NUMERIC_ADMISSION_PROXY_CLOSURE_PENDING',
    'files': files, 'source_predecessor_manifest_sha256': 'bf053c343c81609eac2f3f36b70b2226e3b983ae8225d42b2a3f513ed609bd7d',
    'failure_predecessor_manifest_sha256': 'd5ff08879c5ee91c53d68e16a2bd800f76040c5989077ea23f0f295a48490ead',
    'scientific_result_admission': False, 'publication_admission': False,
    'actual_proxy78014_exit_and_closure_addendum_pending': True,
    'manifest_exclusions': ['candidate_manifest.json', 'sealing_verification.json'],
})
result = {'status': 'passed_exact_compact_copy_and_metadata_projection_checks',
          'candidate_manifest_sha256': sha(OUT / 'candidate_manifest.json'),
          'files_including_manifest_and_verification': len(files) + 2,
          'exact_copies': len(inventory),
          'bytes_excluding_verification': sum((OUT / name).stat().st_size for name in files) + (OUT / 'candidate_manifest.json').stat().st_size,
          'predecessors_unchanged': True, 'new_numeric_products_or_arrays_read': False,
          'scientific_tests_clocks_live_probes_or_repo_writes': False,
          'proxy_closure_and_independent_curation_review_pending': True}
write(OUT / 'sealing_verification.json', result)
print(json.dumps(result, indent=2))
