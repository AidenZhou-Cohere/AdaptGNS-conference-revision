"""Preserve genuine failed original exit and unresolved evidence; no live actions."""
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREP = ROOT / 'work/deadline_research_20261005/cuda_preparation'
STATE = PREP / 'goop3d_observed_history_analysis_released_root_v1'
PREVIOUS = ROOT / 'work/d3_observed_history_source_publication_candidate_UNADMITTED_20261006_v1'
PREVIOUS_PIN = 'bf053c343c81609eac2f3f36b70b2226e3b983ae8225d42b2a3f513ed609bd7d'
TOOLS_PIN = 'e55170d5341c4242c90499d1c8f6e9d0c48646759c61c84e6b05baf2d18f4dd1'
TRANSPORT_PIN = 'a9020af53ccbddfdb1c79a7b871c0a0c4acdd1e0df70a742e788edaba979b0bf'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


assert sha(PREVIOUS / 'candidate_manifest.json') == PREVIOUS_PIN
tool_path = PREP / 'goop3d_observed_pipeline_original_tools_root_v1.json'
transport_path = STATE / 'pipeline.transport/transport.json'
assert sha(tool_path) == TOOLS_PIN and sha(transport_path) == TRANSPORT_PIN
tools = json.loads(tool_path.read_text())
transport = json.loads(transport_path.read_text())
assert tools['launch']['session_id'] == 33822
assert tools['completion']['exit_code'] == transport['exit_code'] == 1
assert 'session_id' not in tools['completion']
assert transport['failure'] is None and transport['signals_to_new_local_group_only'] == []
assert transport['remote_native_closure_established'] is False
assert transport['local_transport_descendant_closure_established'] is False
assert transport['finished_utc'] == '2026-10-07T00:16:01.591961+00:00'
assert not any(name in transport for name in ('cmd', 'argv', 'command', 'ssh_argv'))
inventory = {}


def copy(source, destination, pin, role):
    assert source.is_file() and not source.is_symlink() and source.stat().st_size < 200_000
    assert sha(source) == pin
    target = OUT / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    shutil.copyfile(source, target)
    assert sha(target) == pin
    inventory[destination] = {'source': str(source.relative_to(ROOT)), 'sha256': pin,
                              'bytes': source.stat().st_size, 'role': role, 'transformation': 'none'}


copy(tool_path, 'evidence/goop3d_observed_pipeline_original_tools_root_v1.json', TOOLS_PIN,
     'exact_original_launch_and_genuine_failed_completion_with_source_traceback')
copy(transport_path, 'evidence/pipeline.transport/transport.json', TRANSPORT_PIN,
     'exact_completed_local_transport_metadata_not_remote_cause_or_closure')
for name in ('stdout', 'stderr'):
    source = STATE / 'pipeline.transport' / name
    assert source.stat().st_size == 0
    copy(source, 'evidence/pipeline.transport/' + name,
         'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
         'exact_empty_local_transport_stream_no_remote_fate_inference')
copy(PREVIOUS / 'candidate_manifest.json', 'history/source_metadata_candidate_manifest.json', PREVIOUS_PIN,
     'exact_unchanged_prior_source_only_candidate_manifest')
copy(ROOT / 'work/d3_observed_history_source_curation_independent_review_v1.json',
     'history/source_metadata_candidate_independent_review.json',
     'aa0b370c31b0b85e767cfd4da3f16792ad0d9e9bee86b2701e22e0dc0edeaea3',
     'prior_completed1974check_independent_source_only_curation_review')
write(OUT / 'copy_inventory.json', inventory)
write(OUT / 'failed_completion_disposition.json', {
    'schema': 'd3_observed_failed_original_completion_disposition_v1',
    'status': 'genuine_original_local_pipeline_exit1_no_numerical_admission',
    'original_pipeline_session': 33822,
    'original_tool_exit_code': 1,
    'finished_utc_from_original_transport': transport['finished_utc'],
    'original_tools_sha256': TOOLS_PIN,
    'local_transport_sha256': TRANSPORT_PIN,
    'local_transport_pid': 51576,
    'local_caught_failure_field': None,
    'local_signals_recorded': [],
    'local_stdout_bytes': 0, 'local_stderr_bytes': 0,
    'local_transport_descendant_closure_established': False,
    'remote_native_closure_established': False,
    'remote_scientific_worker_exit': None,
    'remote_failure_cause': None,
    'numeric_product_existence_and_validity': 'unobserved_and_unadmitted',
    'failure_null_does_not_mean_success': True,
    'empty_local_streams_do_not_establish_remote_fate': True,
    'original_success_path_disposition': {
        'root_collect.py': 'not_executed_after_prerequisite_failure; original successful collector must not run',
        'root_observe_collection_closure.py': 'not_executed; success-path47PIDobserver must not run',
        'root_validate_products.py': 'not_executed; successful final validator must not run',
    },
    'disposition_authority': 'root explicit task instruction following genuine original33822exit1',
    'metadata_only_failure_capture': 'separate_source_preparation_by_transition_review; actual_capture_pending_root',
    'original_phase_and_stop_unchanged': True,
    'scientific_retry_or_new_phase': False,
    'all_original_4728_accounting_states_and_incomplete_autonomous_outcomes_retained': True,
})
write(OUT / 'reference_pins.json', {
    'schema': 'd3_observed_failed_completion_reference_pins_v1',
    'prior_source_candidate_manifest_read_and_copied': True,
    'other_lineage_reference_files_not_opened_by_addendum': True,
    'lineage_reference_pins': {
        str(PREVIOUS.relative_to(ROOT)): PREVIOUS_PIN,
        'work/deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1/manifest.json': '1dee2ec7c67639fab665e9fe88ce2cd6fb5862ac9eb9289fafb29446c9165e62',
        'work/deadline_research_20261005/cuda_preparation/goop3d_original_postpipeline_command_plan_transition_v1.json': '564b898c97a0e3efb00e49ca112530ff37548ad4bd61c6c66e15cf43002720e6',
        'work/deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_released_root_v1/analysis_phase.json': 'fb5e6c098f83ebf344ce00a9bb7e31a8b135788d9850ad6d713a9283a83358f2',
        'work/deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_released_root_v1/local_phase_anchor.json': '3d48900be99d5e062e9c9ec7cb37fcafd2e80fd505740b3beb5075818b983cea',
        'work/deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_released_root_v1/pipeline.cpu_release.json': '5c40c1d082a500883c16f0fa4922d4c0ed6578db87583031d1bae4c5af5c4004',
        'work/deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_released_root_v1/pipeline.transport/intent.json': '613c4ab59a87afe42da1265c60cb5acddacc4c8d27e52f755ea201849f03ceee',
        'work/deadline_research_20261005/cuda_preparation/goop3d_stopped_analysis_released_root_v1/d3_collect.collected/goop3d_final_analysis_20261006_v1/stopped_collection.json': '616c612724f62d21bcb521c6163e83d9038b34fdd8012980d6ed55f4b05bed5f',
    },
    'original659091312byte_collection_models_census_and_new_numeric_products_not_opened': True,
})
write(OUT / 'unresolved_failure_capture_slots.json', {
    'schema': 'd3_observed_pending_failure_capture_and_closure_v1',
    'actual_metadata_only_failure_capture_tools': None,
    'actual_failure_envelope_capture': None,
    'actual_local_transport_descendant_closure': None,
    'actual_remote_native_closure': None,
    'actual_phase_proxy78014_exit': None,
    'actual_phase_proxy78014_native_closure': None,
    'final_independent_failure_capture_review': None,
    'numeric_audit': None, 'numeric_summary': None, 'arithmetic_check': None, 'pipeline_receipt': None,
    'scientific_result_admission': False,
    'publication_admission': False,
    'no_new_numeric_product_paths_opened_or_checked': True,
})
assert sha(PREVIOUS / 'candidate_manifest.json') == PREVIOUS_PIN
print(json.dumps({'exact_copied_originals': len(inventory),
                  'original_exit_code': 1, 'remote_cause_and_closure': 'unknown',
                  'downstream_success_actions': 'unexecuted_and_prohibited_after_failed_prerequisite',
                  'candidate_predecessor_unchanged': True, 'numeric_products_read': False}, indent=2))
