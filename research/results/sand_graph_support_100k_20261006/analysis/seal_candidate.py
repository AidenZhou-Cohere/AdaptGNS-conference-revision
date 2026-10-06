"""Seal compact exact copies and predecessor history without scientific execution."""
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
preservation = json.loads((OUT / 'predecessor_preservation.json').read_text())
previous = ROOT / preservation['predecessor_directory']
assert sha(previous / 'candidate_manifest.json') == preservation['predecessor_manifest_sha256']
previous_manifest = json.loads((previous / 'candidate_manifest.json').read_text())
for name, record in previous_manifest['files'].items():
    assert (previous / name).stat().st_size == record['bytes']
    assert sha(previous / name) == record['sha256']
    if name.startswith('payload/'):
        assert sha(OUT / name) == record['sha256']
for path in previous.iterdir():
    if path.is_file():
        assert sha(path) == sha(OUT / preservation['predecessor_top_level_records_exact_copy'] / path.name)
for item in json.loads((OUT / 'final_product_pins.json').read_text()).values():
    assert sha(OUT / item['payload']) == item['sha256']
status = json.loads((OUT / 'candidate_status.json').read_text())
assert status['scientific_product_interpretation_admitted_by_root']
assert status['publication_admission'] is False
assert not any(path.is_symlink() for path in OUT.rglob('*'))
excluded = {'candidate_manifest.json', 'FILELIST.txt', 'sealing_verification.json'}
names = {str(path.relative_to(OUT)) for path in OUT.rglob('*')
         if path.is_file() and str(path.relative_to(OUT)) not in excluded}
names.add('FILELIST.txt')
(OUT / 'FILELIST.txt').write_text(''.join(name + '\n' for name in sorted(names | {'candidate_manifest.json', 'sealing_verification.json'})))
files = {name: {'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in sorted(names)}
manifest = {
    'schema': 'sand_completed_analysis_compact_publication_candidate_manifest_v2',
    'status': 'ADMITTED_SCIENTIFIC_PRODUCTS_PUBLICATION_REVIEW_PENDING',
    'files': files,
    'file_count_excluding_manifest_and_verification': len(files),
    'bytes_excluding_manifest_and_verification': sum(item['bytes'] for item in files.values()),
    'interpretation_admission_sha256': status['interpretation_admission_sha256'],
    'scientific_product_interpretation_admitted_by_root': True,
    'scientific_accuracy_or_positive_effect_claim': False,
    'statistics_recomputed': False,
    'publication_admission': False,
    'retrieval_proxy_cleanup_pending': status['retrieval_proxy_cleanup_pending'],
    'large_original_collections_audits_models_and_arrays_not_copied': True,
    'manifest_exclusions': ['candidate_manifest.json', 'sealing_verification.json'],
}
write(OUT / 'candidate_manifest.json', manifest)
result = {
    'status': 'passed_compact_exact_copy_and_predecessor_preservation_checks',
    'candidate_manifest_sha256': sha(OUT / 'candidate_manifest.json'),
    'candidate_files_including_manifest_and_verification': len(files) + 2,
    'copied_source_provenance_and_compact_product_originals': len(inventory),
    'referenced_originals': len(json.loads((OUT / 'referenced_evidence_inventory.json').read_text())['files']),
    'all_copied_originals_unchanged': True,
    'predecessor_directory_and_manifest_unchanged': True,
    'predecessor_manifest_sha256': preservation['predecessor_manifest_sha256'],
    'candidate_bytes_excluding_verification': sum((OUT / name).stat().st_size for name in files) + (OUT / 'candidate_manifest.json').stat().st_size,
    'compact_scientific_products_read_only_after_root_admission': True,
    'statistics_recomputed': False,
    'tests_or_scientific_workers_rerun': False,
    'independent_final_curation_review_pending': True,
    'separate_proxy_cleanup_addendum_pending': True,
}
write(OUT / 'sealing_verification.json', result)
print(json.dumps(result, indent=2))
