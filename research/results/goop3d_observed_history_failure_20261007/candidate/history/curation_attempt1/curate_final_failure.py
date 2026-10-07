"""Assemble compact source/failure metadata without numerical or live actions."""
import hashlib
import json
from pathlib import Path
import re
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREP = ROOT / 'work/deadline_research_20261005/cuda_preparation'
PREDECESSORS = {
    'source_metadata': ('work/d3_observed_history_source_publication_candidate_UNADMITTED_20261006_v1',
                        'candidate_manifest.json', 'bf053c343c81609eac2f3f36b70b2226e3b983ae8225d42b2a3f513ed609bd7d'),
    'failed_completion': ('work/d3_observed_failed_completion_addendum_UNADMITTED_20261007_v1',
                          'addendum_manifest.json', 'd5ff08879c5ee91c53d68e16a2bd800f76040c5989077ea23f0f295a48490ead'),
}
PACKAGES = {
    'goop3d_final_entry_UNADMITTED_transition_v1': 'fae54495108550246540e0b693bd584e58b65f370cad0cf7fa9368b0426224aa',
    'goop3d_failed_pipeline_capture_UNADMITTED_transition_v1': 'a5fdfa247fa66263125ffb52578328daad4b641b79f768408ee27c4de0eb0201',
}
copies = {}
references = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def copy(source, destination, role, pin=None):
    assert source.is_file() and not source.is_symlink() and source.stat().st_size < 1_000_000
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert pin is None or actual == pin, str(source)
    assert not re.search(rb'-----BEGIN .*PRIVATE KEY-----|\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b', raw)
    target = OUT / destination
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(target) == actual
    copies[destination] = {'source': str(source.relative_to(ROOT)), 'bytes': len(raw),
                           'sha256': actual, 'role': role, 'transformation': 'none'}


for destination, (relative, manifest_name, pin) in PREDECESSORS.items():
    source = ROOT / relative
    assert sha(source / manifest_name) == pin
    manifest = json.loads((source / manifest_name).read_text())
    for name, record in manifest['files'].items():
        assert sha(source / name) == record['sha256'] and (source / name).stat().st_size == record['bytes']
    for path in sorted(source.rglob('*')):
        if path.is_file():
            copy(path, destination + '/' + str(path.relative_to(source)), 'exact_unchanged_predecessor_candidate_file')

for name, pin in PACKAGES.items():
    source = PREP / name
    assert sha(source / 'manifest.json') == pin
    manifest = json.loads((source / 'manifest.json').read_text())
    copy(source / 'manifest.json', 'operational_source/' + name + '/manifest.json', 'exact_frozen_operational_source_manifest', pin)
    for member, member_pin in manifest['files_sha256'].items():
        if Path(member).suffix == '.log':
            references[str((source / member).relative_to(ROOT))] = {'sha256': member_pin,
                'reason': 'retained_original_synthetic_test_log_not_rerun', 'raw_hash_recomputed': False}
        else:
            copy(source / member, 'operational_source/' + name + '/' + member,
                 'exact_frozen_source_test_review_or_failure_contract', member_pin)

files = {
    'goop3d_failed_metadata_original_tools_root_v1.json': '8eeb4d6f717345833001eaddf901d379068b63e00632b8d25948b75c58118802',
    'goop3d_failed_metadata_completed_root_review_v1.json': '3d4c8b75e117f741c146022673c48db0d76f6feb28768ae3626804e98f1f934e',
    'goop3d_failed_pipeline_local_closure_root_v1.json': 'beb3d64b711add4d8dea1ed04a735f77dca393296952799b7e549cf91d051c5e',
    'goop3d_final_entry_independent_root_review_v1.json': '43517650e44e83f539916a039cfa0a12607dbb0c66456e1d01b1e96bc7e1b7fb',
    'goop3d_failed_pipeline_capture_independent_root_review_v1.json': '5f84e07efa075633dc15a368d10f4fc68fbffaa107b9a748d4109efec6b25ad1',
}
for name, pin in files.items():
    copy(PREP / name, 'evidence/' + name, 'exact_original_tool_or_root_source_metadata_review', pin)
copy(PREP / 'goop3d_final_entry_root_tests_v1.json', 'evidence/goop3d_final_entry_root_tests_v1.json',
     'exact_pre_failure_final_entry_source_test_receipt_not_rerun')
copy(ROOT / 'work/d3_observed_failed_completion_addendum_independent_review_statistics_v1.json',
     'reviews/failed_completion_independent_review.json', 'exact134check_predecessor_failure_curation_review',
     '7553ae6d4781be400279036d9742d0a47c92df7aeee38a124f92944c49c8762c')
copy(ROOT / 'work/d3_observed_failed_completion_addendum_independent_review_statistics_v1.md',
     'reviews/failed_completion_independent_review.md', 'exact_predecessor_failure_curation_review_prose')

capture = PREP / 'goop3d_failed_pipeline_metadata_root_v1'
assert sha(capture / 'transport.json') == 'c202135c685f2fbda594bc843e47cab1ad8594c53db9070ba5eebceff87b16bc'
assert sha(capture / 'stdout') == '8966386267205706f972d0007d5a56b9ddb664d91afa4eacfca68e1682964e22'
observation = json.loads((capture / 'stdout').read_text())
assert observation['original_pipeline_session'] == 33822 and observation['original_pipeline_exit_code'] == 1
assert observation['observed_pid_count'] == 45
assert len(observation['native_first']) == len(observation['native_second']) == 45
assert all(item['status'] == 'absent' for item in observation['native_first'].values())
assert all(item['status'] == 'absent' for item in observation['native_second'].values())
assert len(observation['product_sizes_opaque_hashes_only']) == 4
assert all(item == {'status': 'missing'} for item in observation['product_sizes_opaque_hashes_only'].values())
selected = ['schema', 'host', 'boot_id', 'final_boot_id', 'started', 'finished',
            'original_pipeline_session', 'original_pipeline_exit_code', 'phase_sha256', 'release_sha256',
            'all42_plus_actual_new_owner_pids_absent_twice', 'observed_pid_count', 'native_first', 'native_second',
            'new_observer_identity', 'new_observer_timeout_identity', 'product_sizes_opaque_hashes_only',
            'original_success_collector_dispatched', 'collector_absence_basis',
            'registry_identity_coverage_complete', 'registry_bytes_unchanged', 'registry_errors',
            'new_analysis_clock_or_release', 'original_collection_or_arrays_opened',
            'remote_files_written', 'scientific_admission', 'successful_product_numeric_content_parsed',
            'target_processes_signaled', 'termination_cause_established']
projection = {'schema': 'd3_failed_observation_exact_metadata_field_projection_v1',
              'source': str((capture / 'stdout').relative_to(ROOT)),
              'source_sha256': '8966386267205706f972d0007d5a56b9ddb664d91afa4eacfca68e1682964e22',
              'transformation': 'Exact listed top-level metadata fields selected; all field values unchanged.',
              'selected_keys': selected,
              'metadata': {key: observation[key] for key in selected},
              'excluded_fields': sorted(set(observation) - set(selected)),
              'excluded_fields_remain_in_original_capture': True,
              'numeric_product_or_array_content_read': False}
write(OUT / 'evidence/failure_observation_metadata_projection.json', projection)
for name in ('intent.json', 'stdout', 'stderr', 'transport.json'):
    path = capture / name
    references[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'bytes': path.stat().st_size,
        'reason': 'original_completed_metadata_capture_or_transport_envelope_retained_locally',
        'raw_hash_recomputed': True, 'raw_log_or_transport_envelope_copied': False}

write(OUT / 'copy_inventory.json', copies)
write(OUT / 'capture_and_source_references.json', {'schema': 'd3_final_failure_capture_source_reference_inventory_v1',
      'files': references, 'original_large_collection_models_census_and_arrays_not_opened': True})
write(OUT / 'predecessor_and_package_pins.json', {'predecessors': PREDECESSORS, 'operational_source_packages': PACKAGES,
      'predecessors_modified': False, 'no_prior_failed_source_or_result_removed': True})
write(OUT / 'current_failure_disposition.json', {
    'schema': 'd3_observed_failure_current_metadata_disposition_v1',
    'status': 'original_observed_history_analysis_failed_no_numerical_admission',
    'original_pipeline': {'session': 33822, 'exit_code': 1},
    'original_failure_capture': {'session': 51372, 'exit_code': 0},
    'all45_original_old42_plus_new3_remote_identities_absent_twice': True,
    'all4_numerical_products_missing_at_capture': True,
    'recorded_failure_stage': 'audit',
    'recorded_error': 'scoped work deadline or clock disagreement',
    'composite_guard_branch_not_separately_logged': True,
    'model_failure_OOM_or_clock_skew_established': False,
    'local51576_and55854_absence_verified_by_root': True,
    'observer79395_and_timeout79394_native_absence_separately_sampled': False,
    'original_success_collector47PIDclosure_and_finalvalidator': 'unexecuted; prohibited after failed pipeline prerequisite',
    'final_entry_helper': 'tested_source_only_never_executed; preserve as unexecuted preparation',
    'scientific_retry_or_phase_reset': False,
    'all2568_observed_and4728_original_accounting_scope_retained': True,
    'original_expired_hour_and_incomplete_autonomous_outcomes_preserved': True,
    'historical_stage_payload_eb0d983f_source_not_located': True,
    'new_numeric_values_or_arrays_read': False,
    'root_capture_review_sha256': files['goop3d_failed_metadata_completed_root_review_v1.json'],
    'actual_transition_review': None,
    'sole_proxy78014_actual_exit_and_native_closure': None,
    'eventual_proxy_closure_addendum': 'required_exact_later_root_supplied_evidence',
    'scientific_result_admission': False, 'publication_admission': False,
})
print(json.dumps({'exact_copies': len(copies), 'source_and_failure_predecessors_preserved': True,
                  'metadata_projection_fields': len(selected), 'new_numeric_products_read': False,
                  'publication_and_proxy_closure_pending': True}, indent=2))
