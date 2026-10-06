"""Package admitted fixed products; never execute scientific or remote programs."""
import hashlib
import json
from pathlib import Path
import re
import shutil

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1] if SCRIPT.parent.name == 'work' else SCRIPT.parents[2]
PREP = ROOT / 'work/deadline_research_20261005/cuda_preparation'
STATE = PREP / 'sand_stopped_analysis_released_root_v1'
PREVIOUS = ROOT / 'work/sand_completed_analysis_publication_candidate_UNADMITTED_20261006_v1'
OUT = ROOT / 'work/sand_completed_analysis_publication_candidate_UNADMITTED_20261006_v2'
PREVIOUS_PIN = 'd547f432548872555b4b4ebbff68bd4539f03ee8939139bd5dd6dd8d33f1e8a8'
ADMISSION_PIN = 'bab36c91925e8368417206f258b8829f0cda78ec9e423b551b769d48332eca64'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


admission_path = PREP / 'sand_complete_cohort_interpretation_admission_root_v2.json'
assert sha(admission_path) == ADMISSION_PIN
admission = json.loads(admission_path.read_text())
assert admission['status'] == 'admitted_fixed_complete_products_for_scientific_interpretation'
assert admission['policies'][-1] == 'relative-velocity-RMS25'
assert sha(PREVIOUS / 'candidate_manifest.json') == PREVIOUS_PIN
previous_manifest = json.loads((PREVIOUS / 'candidate_manifest.json').read_text())
for name, record in previous_manifest['files'].items():
    assert (PREVIOUS / name).stat().st_size == record['bytes']
    assert sha(PREVIOUS / name) == record['sha256']
assert not OUT.exists()

# Bind each compact addition before creating the successor. Large collections
# and side-specific saved-array audits never pass through this copy list.
additions = []
review_records = []
product_records = {}


def add(path, role, pin=None):
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size <= 10_000_000
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert pin is None or pin == actual, str(path)
    assert not re.search(rb'-----BEGIN .*PRIVATE KEY-----|\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b', raw)
    additions.append((path, role, actual, len(raw)))
    return actual


for operation in ('sand_summarize', 'sand_paired'):
    record = admission['products'][operation]
    product = Path(record['path'])
    add(product, 'exact_admitted_complete_scientific_product_no_recomputation', record['sha256'])
    root_review_path = STATE / (operation + '.review.json')
    add(root_review_path, 'accepted_original_product_root_review', record['root_review_sha256'])
    independent_path = PREP / (operation + '_completed_independent_review_code_audit_v1.json')
    add(independent_path, 'completed_product_independent_review', record['independent_review_sha256'])
    root_review = json.loads(root_review_path.read_text())
    independent = json.loads(independent_path.read_text())
    assert root_review['product_sha256'] == independent['product_sha256'] == record['sha256']
    assert independent['accepted_root_review_sha256'] == record['root_review_sha256']
    assert root_review['status'] == 'accepted_stopped_product'
    review_records.append((independent_path, independent))
    product_records[operation] = {
        **record,
        'source': str(product.relative_to(ROOT)),
        'payload': 'payload/workspace/' + str(product.relative_to(ROOT)),
        'bytes': product.stat().st_size,
        'transformation': 'none',
        'hash_recomputed_before_copy': True,
        'statistics_recomputed': False,
    }

add(admission_path, 'root_final_scientific_interpretation_admission_v2', ADMISSION_PIN)
v1 = admission['supersedes_metadata_label_only']
add(Path(v1['path']), 'preserved_prior_admission_with_metadata_policy_label_error', v1['sha256'])
add(STATE / 'analysis_completion.json', 'all_six_original_analysis_operations_completion', admission['analysis_completion_sha256'])
completion_review = PREP / 'sand_original_analysis_completion_independent_review_code_audit_v1.json'
add(completion_review, 'independent_final_original_analysis_completion_review', admission['completion_independent_review_sha256'])
review_records.append((completion_review, json.loads(completion_review.read_text())))

# These exact original tool receipts have empty output and no invocation text.
# Raw SSH commands and failed transport logs remain references, as in v1.
tool_names = (
    'sand_summarize_original_tool_exit_root_v1.json',
    'sand_summarize_capture_recovery_original_tools_root_v1.json',
    'sand_paired_original_tool_exit_root_v1.json',
    'sand_paired_close_original_tools_root_v1.json',
    'sand_analysis_complete_original_tool_root_v1.json',
    'sand_analysis_complete_original_tools_root_v1.json',
)
for name in tool_names:
    path = PREP / name
    raw = path.read_text()
    assert '"cmd"' not in raw and '"command"' not in raw and 'ssh ' not in raw
    add(path, 'exact_original_final_tool_receipt_without_transport_invocation')
for name in ('sand_summarize.root_original_external_exit.json',
             'sand_paired.root_original_external_exit.json',
             'sand_summarize.recovery_closure_receipt.json'):
    add(STATE / name, 'completed_original_worker_or_capture_only_recovery_receipt')

# Preserve the exact independent reader source pinned by the completed reviews.
for name in ('review_completed_sand_product_code_audit_v3.py',
             'review_sand_analysis_completion_code_audit_v1.py'):
    path = PREP / name
    pins = {r['evidence_sha256'][str(path)] for _, r in review_records if str(path) in r.get('evidence_sha256', {})}
    assert len(pins) == 1, name
    add(path, 'exact_completed_independent_reader_source_not_rerun', pins.pop())

shutil.copytree(PREVIOUS, OUT)
history = OUT / 'history/source_provenance_candidate_v1'
history.mkdir(parents=True)
for path in sorted(PREVIOUS.iterdir()):
    if path.is_file():
        shutil.copyfile(path, history / path.name)
shutil.copyfile(Path(__file__), OUT / 'curate_admitted_products_v2.py')
inventory = json.loads((OUT / 'source_inventory.json').read_text())
for path, role, pin, size in additions:
    relative = 'payload/workspace/' + str(path.relative_to(ROOT))
    target = OUT / relative
    assert relative not in inventory and not target.exists(), relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert sha(target) == pin == sha(path)
    inventory[relative] = {'source': str(path.relative_to(ROOT)), 'bytes': size,
                           'sha256': pin, 'role': role, 'transformation': 'none'}

references = json.loads((OUT / 'referenced_evidence_inventory.json').read_text())
for path, review in review_records:
    for original, pin in review.get('evidence_sha256', {}).items():
        source = Path(original)
        assert source.is_relative_to(PREP)
        name = str(source.relative_to(ROOT))
        old = references['files'].get(name)
        if old:
            assert old['sha256'] == pin, name
            continue
        references['files'][name] = {
            'source': name, 'sha256': pin,
            'bytes_from_local_stat': source.stat().st_size if source.is_file() else None,
            'identity_basis': str(path.relative_to(PREP)),
            'identity_rehashed_by_reference_step': False,
            'content_parsed_by_reference_step': False,
            'copied_by_reference_step': False,
            'reason': 'bound_completed_final_evidence_reference_not_bulk_copied',
        }
references['scope'] = ('Frozen source/provenance/failure and admitted final-product evidence. '
                       'Exact compact copies are listed separately; four large products remain reference-only.')
references['schema'] = 'sand_completed_evidence_reference_inventory_v2'

review_pins = json.loads((OUT / 'completed_review_pins.json').read_text())
for path, _ in review_records:
    review_pins[path.name] = sha(path)
write(OUT / 'source_inventory.json', inventory)
write(OUT / 'referenced_evidence_inventory.json', references)
write(OUT / 'completed_review_pins.json', review_pins)
write(OUT / 'final_product_pins.json', product_records)
write(OUT / 'predecessor_preservation.json', {
    'schema': 'sand_curation_successor_v2_history',
    'predecessor_directory': str(PREVIOUS.relative_to(ROOT)),
    'predecessor_manifest_sha256': PREVIOUS_PIN,
    'predecessor_directory_unchanged': True,
    'predecessor_top_level_records_exact_copy': 'history/source_provenance_candidate_v1',
    'unchanged_predecessor_payload_preserved_in_successor': True,
    'admission_v1_retained_with_v2_metadata_label_correction': True,
    'admission_v1_sha256': v1['sha256'],
    'admission_v2_sha256': ADMISSION_PIN,
})
write(OUT / 'candidate_status.json', {
    'status': 'ADMITTED_SCIENTIFIC_PRODUCTS_PUBLICATION_REVIEW_PENDING',
    'interpretation_admission_sha256': ADMISSION_PIN,
    'scientific_product_interpretation_admitted_by_root': True,
    'scientific_accuracy_or_positive_effect_claim': False,
    'publication_admission': False,
    'compact_products_read_and_copied_only_after_root_admission': True,
    'statistics_derived': False,
    'scientific_workers_or_tests_rerun': False,
    'large_products_copied_or_compressed': False,
    'original_sources_changed': False,
    'remote_probes_or_clocks_used': False,
    'pending_root_supplied_items': ['final successor route exit and cleanup evidence'],
    'pending_independent_final_curation_review': True,
    'preserved_original_analysis_stop_utc': '2026-10-06T23:33:01.226132+00:00',
    'retrieval_proxy_cleanup_pending': admission['retrieval_proxy_cleanup_pending'],
})
failure = json.loads((OUT / 'failure_inventory.json').read_text())
failure['schema'] = 'sand_completed_analysis_failure_provenance_inventory_v2'
failure['source'] = 'Preserved completed provenance and failure bindings; admitted final product/review copies remain exact.'
failure['status'] = 'completed_products_admitted_publication_and_proxy_cleanup_pending'
failure['entries'].append({
    'id': 'admission_v1_policy_label_correction',
    'preserve': 'Both exact root admissions are retained. V2 corrects only the abbreviated RMS policy identifier.',
    'scientific_disposition': 'No scientific product, fixed grid, deadline or admission scope changed.',
    'evidence': ['sand_complete_cohort_interpretation_admission_root_v1.json',
                 'sand_complete_cohort_interpretation_admission_root_v2.json'],
})
failure['completed_original_analysis_sessions'] = {
    'summary': {'session': 19303, 'exit_code': 0},
    'summary_capture_only_successor': {'session': 47492, 'exit_code': 0},
    'paired_audit': {'session': 92386, 'exit_code': 0},
    'paired_close': {'session': 46499, 'exit_code': 0},
    'analysis_complete': {'session': 7936, 'exit_code': 0},
}
failure['separate_pending_route_cleanup'] = admission['retrieval_proxy_cleanup_pending']
failure['no_numerical_retry_or_elapsed_cost_reset'] = True
write(OUT / 'failure_inventory.json', failure)

assert sha(PREVIOUS / 'candidate_manifest.json') == PREVIOUS_PIN
print(json.dumps({'successor': str(OUT), 'exact_additions': len(additions),
                  'exact_copied_originals': len(inventory), 'referenced_originals': len(references['files']),
                  'root_interpretation_admission_sha256': ADMISSION_PIN,
                  'pending': ['README/HANDOFF/seal update', 'independent curation review', 'final proxy cleanup']}, indent=2))
