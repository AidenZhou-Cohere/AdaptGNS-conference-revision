"""Copy only fixed D3 source/metadata; reference products without opening them."""
import hashlib
import json
from pathlib import Path
import re
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREP = ROOT / 'work/deadline_research_20261005/cuda_preparation'
STATE = PREP / 'goop3d_observed_history_analysis_released_root_v1'
OLD = PREP / 'goop3d_stopped_analysis_released_root_v1'
PACKAGES = {
    'goop3d_observed_history_analysis_UNADMITTED_transition_v1': '1dee2ec7c67639fab665e9fe88ce2cd6fb5862ac9eb9289fafb29446c9165e62',
    'goop3d_metadata_attempt2_UNADMITTED_transition_v1': '1fa7c5a897a98947f02865a07d9680ac54e9ec201a43d61783b3efdedf81bec1',
    'goop3d_post_expiry_metadata_observer_UNADMITTED_transition_v1': 'e9124cbc677e40ab5fed401c9fefb1da04b06c932e86459d2244ddc75e79eb05',
}
PINS = {
    'goop3d_original_postpipeline_command_plan_transition_v1.json': '564b898c97a0e3efb00e49ca112530ff37548ad4bd61c6c66e15cf43002720e6',
    'goop3d_observed_history_final_operational_review_root_v1.json': '4e2d79c578f6da97bceff46cd8655c657e50fe54df91d4edb452c694d4e56753',
    'goop3d_observed_history_source_independent_review_statistics_v1.json': '267388587f5c15cd93beccea05f26727a29bba59d6a11fbeef66aa3602e73028',
    'goop3d_attempt2_actual_admission_staging_independent_transition_review_v1.json': '1497bab47c20314d563ac02d63efa088ca022ccf6041aad54c47d933fdae8230',
    'goop3d_observed_prerequisite_freshness_decline_root_v1.json': 'ec7acfede8ef7d841b7795d8ce5da80ee3a7293214ff1a090cde3ee490593c32',
    'goop3d_metadata_attempt2_independent_root_review_v1.json': '4dc7451b4b43beb4d2f1d99a0b0196ae7990c81b2331ed7eb92d3606810bf22a',
    'goop3d_metadata_attempt2_secondary_transition_review_v1.json': 'cb6af2d588af040f5db8dc05f0df785e5cbfc005df65b5839c4718b165bf2516',
    'goop3d_completed_retained_fate_observation_independent_transition_v1.json': '2c90d522f903746f0151be03932e86c0cd1a36365233bd19f34bb0b816474203',
    'goop3d_collect_completed_independent_review_transition_v1.json': 'afc6e4c96f6aec0bf5e322d04d62bfae0817953e210d7d04f4302cabcd7d9f65',
    'goop3d_actual_frozen_cohort_independent_review_transition_v1.json': 'ade7b5815f50bb387f7b981166115df653a303e649f96fb912b65d236844e106',
    'goop3d_completed_cohort_independent_review_transition_v1.json': '4085aee805718b4ba73c4fd91f53becac25a0530a352e6e4a4b02ed6cc598290',
    'goop3d_census_scientific_product_independent_review_transition_v1.json': 'edb4cc2617451362d56d93234c1814a727de092c915681ea90d0b73efaeaba8a',
}
inventory = {}
references = {}
review_pins = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def reference(path, pin, authority, reason, size=None, rehashed=False):
    path = Path(path)
    name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    assert isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin)
    if name in references:
        assert references[name]['sha256'] == pin, name
        if rehashed:
            references[name]['raw_hash_recomputed_for_reference'] = True
        return
    references[name] = {'source': name, 'sha256': pin, 'bytes_if_recorded': size,
                        'identity_basis': authority, 'reason': reason,
                        'content_parsed_by_reference_step': False,
                        'raw_hash_recomputed_for_reference': rehashed,
                        'copied': False}


def bound_references(value, authority):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            reference(value['path'], value['sha256'], authority, 'bound_original_metadata_or_source_reference', value.get('bytes'))
        for field in ('evidence_sha256', 'bindings'):
            if isinstance(value.get(field), dict):
                for name, record in value[field].items():
                    if not name.startswith('/'):
                        name = str(PREP / name)
                    pin = record.get('sha256') if isinstance(record, dict) else record
                    if isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin):
                        reference(name, pin, authority, 'accepted_source_metadata_or_failure_evidence_reference',
                                  record.get('bytes') if isinstance(record, dict) else None)
        for child in value.values():
            bound_references(child, authority)
    elif isinstance(value, list):
        for child in value:
            bound_references(child, authority)


def copy(path, role, pin=None, parse_metadata=True):
    assert path.is_file() and not path.is_symlink() and path.is_relative_to(PREP)
    assert path.stat().st_size < 1_000_000, str(path)
    relative = 'payload/workspace/' + str(path.relative_to(ROOT))
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert pin is None or actual == pin, str(path)
    assert not re.search(rb'-----BEGIN .*PRIVATE KEY-----|\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b', raw)
    if relative in inventory:
        assert inventory[relative]['sha256'] == actual
        return
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    shutil.copyfile(path, target)
    assert sha(target) == actual
    inventory[relative] = {'source': str(path.relative_to(ROOT)), 'bytes': len(raw),
                           'sha256': actual, 'role': role, 'transformation': 'none'}
    if parse_metadata and path.suffix == '.json':
        bound_references(json.loads(raw), str(path.relative_to(ROOT)))


for name, pin in PACKAGES.items():
    package = PREP / name
    manifest = package / 'manifest.json'
    assert sha(manifest) == pin
    record = json.loads(manifest.read_text())
    copy(manifest, 'exact_frozen_source_package_manifest', pin)
    for member, member_pin in record['files_sha256'].items():
        path = package / member
        if path.suffix in ('.log', '.txt'):
            reference(path, member_pin, str(manifest.relative_to(ROOT)), 'original_source_test_log_retained_without_rerun')
        else:
            copy(path, 'frozen_source_test_contract_or_preserved_source_history', member_pin)

for name, pin in PINS.items():
    copy(PREP / name, 'exact_completed_source_or_metadata_review_not_numeric_result', pin)
    review_pins[name] = pin
for name in ('goop3d_metadata_attempt1_independent_transition_review_v1.json',
             'goop3d_completed_post_expiry_observation_independent_transition_v1.json',
             'goop3d_post_expiry_metadata_observer_independent_source_review_v1.json',
             'goop3d_observed_history_source_independent_review_statistics_v1.md',
             'goop3d_observed_history_source_only_proposal_provenance_transition_v1.json',
             'goop3d_observed_history_analysis_PROPOSAL_transition_v1.md'):
    copy(PREP / name, 'exact_historical_review_or_proposal_with_scope_qualifications')
    review_pins[name] = sha(PREP / name)
for path in sorted((PREP / 'goop3d_observed_history_proposal_history_transition_v1').iterdir()):
    assert path.is_file()
    copy(path, 'preserved_prelaunch_proposal_cap_error_and_history')

for name, pin in (
    ('analysis_phase.json', 'fb5e6c098f83ebf344ce00a9bb7e31a8b135788d9850ad6d713a9283a83358f2'),
    ('local_phase_anchor.json', '3d48900be99d5e062e9c9ec7cb37fcafd2e80fd505740b3beb5075818b983cea'),
    ('pipeline.cpu_release.json', '5c40c1d082a500883c16f0fa4922d4c0ed6578db87583031d1bae4c5af5c4004'),
    ('combined_stage_admission.json', '7a1498005152786cdf596cf408f1d0f454fdda9ce3c0bd120400c1b1adbb07d6'),
    ('fresh_admission.json', '9da74b3cc73c8a2ef16b9bafda524eec1f82e4753b0c52411272eb5d05531ef0'),
    ('prior_fate_review.json', '2c90d522f903746f0151be03932e86c0cd1a36365233bd19f34bb0b816474203'),
    ('root_clock.json', '695637a1e602fa85555684533bf45586c3238cfbb82d2cd4af9f981220f9179b'),
):
    copy(STATE / name, 'actual_released_control_or_completed_metadata_admission', pin)
entry = PREP / 'goop3d_observed_fresh_entry_attempt2_root_v1/goop3d_observed_fresh_entry_root_v1'
for name in ('admission.json', 'local_native_closure.json', 'original_tools.json', 'start_commands.json'):
    copy(entry / name, 'actual_metadata_attempt2_admission_closure_or_source_command_plan')
for directory in ('goop3d_observed_fresh_route_root_v1', 'goop3d_observed_fresh_route_attempt2_root_v1'):
    copy(PREP / directory / 'terminal.json', 'original_completed_metadata_proxy_terminal')
copy(PREP / 'goop3d_observed_phase_route_root_v1/ready.json', 'phase_route_startup_metadata_not_final_closure',
     'a7275b4d8aadd2ac5f5d2315712bb89ca286cb136ba9ac2e028fcd2c5a63a112')
copy(PREP / 'goop3d_observed_pipeline_original_launch_root_v1.json', 'original_pipeline_launch_only_no_completion_inferred',
     'ed3c640b4c9836f324ee59327d017db85c507881f654bb14a6471b6ea3f60239')

for name, pin in (
    ('analysis_phase.json', '46a0e4c5dd7e35e1233735171eaf3df094da1e2f6236a9928c8ee911d21768df'),
    ('d3_collect.review.json', 'ff50a7d4a53bae8913c9fe4f12c66364904906fc3f7cfb0e3ae5d8963eb47401'),
):
    copy(OLD / name, 'original_expired_phase_or_accepted_incomplete_collection_review', pin)
copy(OLD / 'local_phase_anchor.json', 'original_expired_phase_anchor_unchanged')
copy(OLD / 'd3_collect.dns_resume_attempt2.json', 'original_same_phase_DNS_staging_recovery_receipt')

package = PREP / 'goop3d_observed_history_analysis_UNADMITTED_transition_v1'
operations = json.loads((package / 'observed_operations.json').read_text())['operations']['d3_observed_history_pipeline']
for remote, pin in operations['inputs_sha256'].items():
    if pin is None:
        continue
    reference(remote, pin, 'frozen observed_operations.json inputs_sha256', 'remote_exact_source_or_preexisting_collection_identity')
    prefix = '/root/repos/AdaptGNS-cuda-20261006/cuda_preparation/'
    if remote.startswith(prefix):
        member = remote.removeprefix(prefix)
        if '/' not in member:
            copy(PREP / member, 'unchanged_original_scientific_source_or_protocol_dependency', pin)
for name, pin in (
    ('supervise_stopped_analysis_cpu_v1.py', '18b5f6cff4f212c7152810f8ff26e9159416ae322aabd666c37e8a60117d594e'),
    ('coder_exact_route_proxy_root_v1.py', 'e6bfabe7c2dcf668db83d59480c311cde75d03a3824ff3ec2610ff858d55d57e'),
):
    copy(PREP / name, 'unchanged_qualified_owner_or_metadata_route_source_ancestor', pin)

collection = OLD / 'd3_collect.collected/goop3d_final_analysis_20261006_v1/stopped_collection.json'
assert collection.stat().st_size == 659091312
reference(collection, '616c612724f62d21bcb521c6163e83d9038b34fdd8012980d6ed55f4b05bed5f',
          'accepted original collection root and independent reviews; current local stat size only',
          'original_large_collection_not_opened_rehashed_copied_or_compressed', 659091312)
for name, size, pin in (
    ('cohort.json', 5290, '645343fc2a1c6ef0a82e212e351b03b1a4d081702d3c8e2187b244a62c6b9bef'),
    ('cohort_audit.json', 6330, '796d50adf33b20eb040200c829d31650ff5bc97927440220ea3c9489db7d840d'),
    ('freeze_report.json', 1513, '2621ce54ccc4c0bf9f06c878245c9fa7c6651e0c1f8125817145792f9c644d5d'),
):
    reference(PREP / 'goop3d_completed_cohort_1418_root_v1' / name, pin,
              'goop3d_actual_frozen_cohort_independent_review_transition_v1.json',
              'original_six_model_cohort_provenance_reference_only', size)
for name, pin in (
    ('goop3d_root_cross_test_released_1440_v1/final_test_admission.json', '7ea72da086176925d62ffe2bfb5faf211ded37be3babcba696b3e589ff7e3b5b'),
    ('goop3d_root_cross_test_released_1440_v1/cross_split_audit.root.json', '54019964f2923d797f7c314bdaea7d56d37c134ce03f33c94438d5085843a715'),
):
    reference(PREP / name, pin, 'previous root/census admission bindings', 'original_census_and_split_admission_reference_only')

# Failure records contain raw transport or native command text. Hash only
# these known small completed envelopes, keep them local, and do not parse them.
for name in ('goop3d_collect_first_stage_dns_failure_root_v1.json',
             'goop3d_expired_phase_local_interrupt_root_v1.json',
             'goop3d_local_interrupted_transport_closure_root_v1.json',
             'goop3d_observed_metadata_proxy_original_tools_without_terminal_root_v1.json',
             'goop3d_observed_metadata_proxy_original_tools_root_v1.json',
             'goop3d_observed_fresh_original_tools_root_v1.json',
             'goop3d_attempt2_observer_original_tools_root_v1.json',
             'goop3d_attempt2_proxy_original_tools_root_v1.json',
             'goop3d_attempt2_begin_original_tools_root_v1.json'):
    path = PREP / name
    assert path.stat().st_size < 100_000
    reference(path, sha(path), 'exact completed local envelope byte hash, content not parsed by curation script',
              'raw_original_tool_or_transport_evidence_retained_locally', path.stat().st_size, rehashed=True)

write(OUT / 'source_inventory.json', inventory)
write(OUT / 'source_package_pins.json', PACKAGES)
write(OUT / 'review_pins.json', review_pins)
write(OUT / 'referenced_evidence_inventory.json', {'schema': 'd3_observed_source_metadata_reference_inventory_v1',
      'files': references, 'new_scientific_product_or_array_content_read': False,
      'original_collection_content_read_or_rehashed': False})
plan = json.loads((PREP / 'goop3d_original_postpipeline_command_plan_transition_v1.json').read_text())
slots = set()
def find_slots(value):
    if isinstance(value, dict):
        if 'required_future_actual_binding' in value:
            slots.add(value['required_future_actual_binding'])
        for child in value.values(): find_slots(child)
    elif isinstance(value, list):
        for child in value: find_slots(child)
find_slots(plan)
write(OUT / 'unresolved_completion_slots.json', {
    'schema': 'd3_observed_unresolved_completion_slots_v1',
    'original_command_plan_sha256': PINS['goop3d_original_postpipeline_command_plan_transition_v1.json'],
    'original_pipeline_session': 33822, 'original_phase_proxy_session': 78014,
    'actual_original_pipeline_exit': None, 'actual_original_collection_exit': None,
    'actual_original_postcollection_closure_exit': None, 'actual_original_phase_proxy_exit_and_closure': None,
    'future_bindings': {name: None for name in sorted(slots)},
    'four_scientific_products': {name: None for name in operations['outputs']},
    'final_root_numerical_admission': None,
    'no_future_product_path_opened_or_resolved': True,
})
write(OUT / 'candidate_status.json', {
    'schema': 'd3_observed_source_metadata_candidate_status_v1', 'status': 'SOURCE_METADATA_ONLY_UNADMITTED',
    'scope': 'all2568observed_cells_with_all4728original_accounting_states_retained',
    'scientific_result_admission': False, 'publication_admission': False,
    'original_pipeline_launch_recorded_completion_unresolved': 33822,
    'original_phase_proxy_final_closure_unresolved': 78014,
    'original_expired_phase_unchanged': True,
    'new_scope_started_utc': '2026-10-06T23:31:18.235220+00:00',
    'new_scope_stop_utc': '2026-10-07T00:31:18.235220+00:00',
    'final_root_admission_deadline_utc': '2026-10-07T00:30:58.235220+00:00',
    'scientific_programs_tests_clocks_or_live_probes_run': False,
    'new_numeric_products_or_arrays_read': False, 'original_collection_opened_or_rehashed': False,
    'repository_or_manuscript_changed': False,
})
print(json.dumps({'copied_originals': len(inventory), 'copied_bytes': sum(v['bytes'] for v in inventory.values()),
                  'references': len(references), 'future_bindings_left_unresolved': len(slots),
                  'numeric_products_read': False}, indent=2))
