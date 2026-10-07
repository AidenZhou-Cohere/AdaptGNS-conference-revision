"""Check exact compact source/metadata copies only, then seal their manifest."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


inventory = json.loads((OUT / 'source_inventory.json').read_text())
for name, record in inventory.items():
    assert (OUT / name).stat().st_size == record['bytes']
    assert sha(OUT / name) == record['sha256'] == sha(ROOT / record['source'])
assert not any(path.is_symlink() for path in OUT.rglob('*'))
status = json.loads((OUT / 'candidate_status.json').read_text())
assert status['scientific_result_admission'] is False and status['new_numeric_products_or_arrays_read'] is False
slots = json.loads((OUT / 'unresolved_completion_slots.json').read_text())
assert all(value is None for value in slots['four_scientific_products'].values())
assert all(value is None for value in slots['future_bindings'].values())
excluded = {'candidate_manifest.json', 'FILELIST.txt', 'sealing_verification.json'}
names = {str(path.relative_to(OUT)) for path in OUT.rglob('*')
         if path.is_file() and str(path.relative_to(OUT)) not in excluded}
names.add('FILELIST.txt')
(OUT / 'FILELIST.txt').write_text(''.join(name + '\n' for name in sorted(names | {'candidate_manifest.json', 'sealing_verification.json'})))
files = {name: {'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in sorted(names)}
write(OUT / 'candidate_manifest.json', {
    'schema': 'd3_observed_source_metadata_publication_candidate_manifest_v1',
    'status': 'SOURCE_METADATA_ONLY_UNADMITTED', 'files': files,
    'file_count_excluding_manifest_and_verification': len(files),
    'bytes_excluding_manifest_and_verification': sum(item['bytes'] for item in files.values()),
    'scientific_result_admission': False, 'publication_admission': False,
    'large_collection_models_arrays_or_new_numeric_products_not_copied': True,
    'all_actual_final_completion_slots_unresolved': True,
    'historical_intermediate_source_hash_not_located': 'eb0d983fb6b85f8d1ba0fc27a4cd9a82456c8ee9d89469e5ef16d0b6f92cfaa4',
    'manifest_exclusions': ['candidate_manifest.json', 'sealing_verification.json'],
})
result = {
    'status': 'passed_exact_source_metadata_copy_checks_only',
    'candidate_manifest_sha256': sha(OUT / 'candidate_manifest.json'),
    'candidate_files_including_manifest_and_verification': len(files) + 2,
    'exact_original_copies': len(inventory),
    'referenced_originals': len(json.loads((OUT / 'referenced_evidence_inventory.json').read_text())['files']),
    'all_copied_originals_unchanged': True,
    'candidate_bytes_excluding_verification': sum((OUT / name).stat().st_size for name in files) + (OUT / 'candidate_manifest.json').stat().st_size,
    'new_numeric_products_or_arrays_read': False,
    'scientific_programs_tests_clocks_live_probes_or_repo_writes': False,
    'independent_curation_review_and_final_actual_admission_pending': True,
}
write(OUT / 'sealing_verification.json', result)
print(json.dumps(result, indent=2))
