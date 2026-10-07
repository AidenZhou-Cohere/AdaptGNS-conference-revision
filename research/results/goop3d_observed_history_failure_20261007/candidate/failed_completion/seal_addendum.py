"""Verify compact exact failure metadata only and seal this inert addendum."""
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
assert not any(path.is_symlink() for path in OUT.rglob('*'))
pending = json.loads((OUT / 'unresolved_failure_capture_slots.json').read_text())
for key in ('numeric_audit', 'numeric_summary', 'arithmetic_check', 'pipeline_receipt',
            'actual_metadata_only_failure_capture_tools', 'actual_remote_native_closure',
            'actual_phase_proxy78014_exit', 'actual_phase_proxy78014_native_closure'):
    assert pending[key] is None
excluded = {'addendum_manifest.json', 'FILELIST.txt', 'sealing_verification.json'}
names = {str(path.relative_to(OUT)) for path in OUT.rglob('*')
         if path.is_file() and str(path.relative_to(OUT)) not in excluded}
names.add('FILELIST.txt')
(OUT / 'FILELIST.txt').write_text(''.join(name + '\n' for name in sorted(names | {'addendum_manifest.json', 'sealing_verification.json'})))
files = {name: {'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in sorted(names)}
write(OUT / 'addendum_manifest.json', {
    'schema': 'd3_observed_failed_completion_addendum_manifest_v1',
    'status': 'GENUINE_LOCAL_EXIT1_REMOTE_FATE_UNRESOLVED_UNADMITTED',
    'files': files, 'exact_original_copies': len(inventory),
    'scientific_result_admission': False, 'publication_admission': False,
    'no_numeric_products_arrays_or_original_collection_read': True,
    'no_scientific_live_probe_or_repository_actions': True,
    'prior_source_candidate_manifest_sha256': 'bf053c343c81609eac2f3f36b70b2226e3b983ae8225d42b2a3f513ed609bd7d',
    'manifest_exclusions': ['addendum_manifest.json', 'sealing_verification.json'],
})
result = {'status': 'passed_exact_failed_completion_metadata_copy_checks',
          'addendum_manifest_sha256': sha(OUT / 'addendum_manifest.json'),
          'files_including_manifest_and_verification': len(files) + 2,
          'exact_original_copies': len(inventory),
          'bytes_excluding_verification': sum((OUT / name).stat().st_size for name in files) + (OUT / 'addendum_manifest.json').stat().st_size,
          'remote_fate_and_final_closure_pending': True,
          'numeric_products_read': False, 'source_predecessor_manifest_unchanged': True}
write(OUT / 'sealing_verification.json', result)
print(json.dumps(result, indent=2))
