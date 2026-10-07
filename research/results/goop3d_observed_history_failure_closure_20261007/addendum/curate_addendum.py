"""Build one local metadata/source addendum; never execute copied source."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai')
OUT = ROOT / 'work/d3_observed_failure_closure_addendum_UNADMITTED_20261007_v1'
PREP = 'work/deadline_research_20261005/cuda_preparation/'
BASE = 'work/d3_observed_failure_final_publication_candidate_UNADMITTED_20261007_v2/'
OLDPLAN = 'work/d3_observed_failure_publication_plan_UNADMITTED_20261007_v1/'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(rel, pin=None):
    path = ROOT / rel
    assert path.is_file() and not path.is_symlink()
    assert path.resolve().is_relative_to(ROOT) and path.stat().st_size < 1_000_000
    raw = path.read_bytes()
    if pin is not None:
        assert sha(raw) == pin, (rel, sha(raw), pin)
    return raw

def write(rel, raw):
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)

def dump(rel, doc):
    write(rel, (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode())

copies = []
references = []

def copy(rel, pin=None):
    raw = read(rel, pin)
    dest = 'payload/workspace/' + rel
    assert not any(row['destination'] == dest for row in copies)
    write(dest, raw)
    copies.append({'source': rel, 'destination': dest, 'bytes': len(raw), 'sha256': sha(raw)})
    return raw

def reference(rel, pin, reason):
    raw = read(rel, pin)
    references.append({'path': rel, 'bytes': len(raw), 'sha256': pin, 'reason_not_copied': reason})
    return raw

pins = {
    'goop3d_failed_study_final_proxy_closure_root_v1.json': '5d68cfded1826b00de5564f9401fd9c03b966f70bf6ba3c58808f7fb14e1bdd5',
    'goop3d_failed_phase_proxy_close_root_v1.py': '4b97790958b1ca77413ba391bf9e76004e04d0b4d6b72bca1113afecacb36db3',
    'goop3d_failed_proxy_close_original_tool_root_v1.json': '9e9e82b02c20c9cee0cf9e005c4b1dde9d210a3e6ca16f9c6b525dcb52f8615d',
    'goop3d_failed_proxy_closure_postpublication_root_v1.json': 'c578d58b66d691a3c3dbb0260abde9ef7025bfde1086a84b1d988f493fd8987d',
    'goop3d_observed_phase_route_root_v1/terminal.json': '93cf0eacfb8036054a96b43292b80ecf3e8dd8d9b917916e5e2ee65dca2e047d',
    'goop3d_failed_metadata_close_original_tools_root_v1.json': '8dad78a34e124447ec44e96df4d4479e6501e355d09893d065dd70e8d39182c5',
    'goop3d_failed_metadata_observer_closure_root_v1/stdout': '01d089ad8b3f9b88f2236141d4bafcc732cba73d52cb5f3ff6ab8622a5aa834a',
    'goop3d_failed_metadata_observer_actual_closure_independent_transition_review_v1.json': 'faabdd984b8ca4f3bd4f57ea88363d36d29ac0a8f8eae54e5bc9a0f8ab34381b',
    'goop3d_failed_metadata_observer_closure_source_root_review_v1.json': 'b9c1b4db63f826bf200136c857d9b290b9dee17d567e24a6bd3d564f055714e6',
    'goop3d_failed_pipeline_actual_metadata_independent_transition_review_v1.json': '4dfa42b603f7404d862cdff5ff7cab1e580d1b470076201b4efab4e53382f56f',
    'goop3d_failed_pipeline_actual_metadata_closure_reviewer_addendum_v1.json': '96515aa9b7a44fc0bc26caae4d7859e3d85e9064ab63c681abd31cd9a5e4efb6',
    'goop3d_failed_metadata_completed_root_review_v1.json': '3d4c8b75e117f741c146022673c48db0d76f6feb28768ae3626804e98f1f934e',
    'd3_observed_failure_final_publication_candidate_independent_transition_review_v2.json': '9c9b2e586d7f85f82d2a78f1e953c22db3e1a914246e9aad8bb9d8f4b1cd4a86',
}
for name, pin in pins.items():
    copy(PREP + name, pin)

package = PREP + 'goop3d_failed_metadata_observer_closure_UNADMITTED_transition_v1/'
manifest = json.loads(copy(package + 'manifest.json', 'cfe2578ba81c40c6e48c79f82a97ef72388ced2cb77aff47e5f363151fbf5707'))
for name, pin in manifest['files_sha256'].items():
    copy(package + name, pin)

copy(BASE + 'candidate_manifest.json', 'f781011ebc4806a42bacf3f2abf60a951aca0bdf9bec3990ace2d0b105b355ba')
copy(BASE + 'sealing_verification.json')
copy(BASE + 'current_failure_disposition.json')
oldplanmanifest = json.loads(copy(OLDPLAN + 'plan_manifest.json', '9ec6b869dd5ceca778d6db3b561a266d4e48d8953c7768f79baadac888539304'))
for name, record in oldplanmanifest['files'].items():
    raw = copy(OLDPLAN + name, record['sha256'])
    assert len(raw) == record['bytes']

for dirname in ('work/d3_attempted_study_status_root_v1',
                'work/d3_attempted_study_status_candidate_statistics_v1',
                'work/d3_attempted_study_status_textual_review_statistics_v1',
                'work/d3_status_docs_root_v1'):
    for path in sorted((ROOT / dirname).rglob('*')):
        assert not path.is_symlink()
        if path.is_file():
            assert path.suffix in ('.json', '.md', '.tex', '.txt', '.patch')
            copy(str(path.relative_to(ROOT)))
copy('outputs/revised_manuscript.tex', 'a31587aeaf19f79893aaf297f00f4759516de34d2086f57f2966b3edee08be9e')
copy('work/manuscript_body.tex', '311c7120730c47a86aee08d8f528697b919ad84c96fc887795bb6c4b8266f0e8')
for rel, pin in {
    'outputs/README.md': '447da0e8aede4344a6e58fda859b10bb6600019a68bd4cfce760da037655a62e',
    'outputs/research_findings.md': '175c732c7f06611c016f625a57a1dd29bf4d7bb18bd79934d7f62fa5ba418e6c',
    'outputs/submission_checklist.md': '5d18f3e3a89f74923983aaff06d921e7a21d9e1fc2019250149244206d37fb23',
}.items():
    copy(rel, pin)

proxyrel = PREP + 'goop3d_observed_phase_proxy_original_tools_root_v1.json'
proxyraw = reference(proxyrel, 'f1891549b0089da2cd2842c153bb094c367bbc8cc4e280411c9fb9274c503195', 'Original launch includes route configuration; publish explicitly selected metadata only.')
proxy = json.loads(proxyraw)
launch = json.loads(proxy['launch']['output'])
excluded = ['authority', 'upstream', 'url']
selected = sorted(set(launch) - set(excluded))
assert set(excluded).issubset(launch) and proxy['completion']['exit_code'] == 0
dump('evidence/proxy_original_tools_selected_metadata.json', {
    'schema': 'd3_final_proxy_tools_selected_metadata_projection_v1',
    'source': proxyrel, 'source_sha256': sha(proxyraw),
    'launch_tool_envelope_without_output': {k:v for k,v in proxy['launch'].items() if k != 'output'},
    'launch_output_selected_keys': selected, 'launch_output_excluded_keys': excluded,
    'launch_output_selected_values': {k:launch[k] for k in selected},
    'completion_exact_value': proxy['completion'],
    'is_exact_full_envelope_copy': False,
    'transformation': 'Parse original launch.output JSON; omit exactly three route configuration fields; selected values and completion value remain unchanged.',
})
transportrel = PREP + 'goop3d_failed_metadata_observer_closure_root_v1/transport.json'
traw = reference(transportrel, 'c5afb8f4da8dd808464a8e27126b8627bcee8bfc40700e268e1dc14ba54c23dd', 'Raw local transport envelope remains a local reference; exact observer stdout and independent closure review are copied.')
transport = json.loads(traw)
report = json.loads(read(PREP + 'goop3d_failed_metadata_observer_closure_root_v1/stdout'))
assert transport['remote_report'] == report
assert report['both_exact_prior_metadata_identities_absent_twice']
assert all(value['status'] == 'absent' for key in ('first', 'second') for value in report[key].values())
assert transport['exit_code'] == 0 and transport['failure'] is None
assert transport['signals_to_own_new_local_transport_group'] == []
assert transport['cleanup_errors'] == []
keys = sorted(set(transport) - {'remote_report'})
dump('evidence/observer_transport_selected_metadata.json', {
    'schema': 'd3_final_observer_transport_selected_metadata_projection_v1',
    'source': transportrel, 'source_sha256': sha(traw),
    'selected_keys': keys, 'excluded_keys': ['remote_report'],
    'selected_values': {k:transport[k] for k in keys},
    'remote_report_exact_copy': 'payload/workspace/' + PREP + 'goop3d_failed_metadata_observer_closure_root_v1/stdout',
    'remote_report_equals_parsed_exact_stdout': True,
    'transformation': 'Select listed top-level fields unchanged; the separate copied stdout JSON equals remote_report.',
})

closure = json.loads(read(PREP + 'goop3d_failed_study_final_proxy_closure_root_v1.json'))
post = json.loads(read(PREP + 'goop3d_failed_proxy_closure_postpublication_root_v1.json'))
assert closure['actual_proxy_exit_code'] == 0 and closure['original_proxy_session'] == 78014
assert closure['local_native_closure']['matching_rows'] == []
assert closure['local_native_closure']['observed_pgids'] == [51576, 55854, 56202]
assert closure['terminal']['remaining_worker_threads'] == 0
assert post['original_helper_unmodified'] and post['actual_original_publication_and_tool_exit_verified_before_unchanged_phase_stop']
dump('current_closure_disposition.json', {
    'schema': 'd3_failed_observed_history_final_closure_disposition_v1',
    'status': 'failed_study_closed_without_numerical_admission',
    'predecessor_candidate_manifest_sha256': 'f781011ebc4806a42bacf3f2abf60a951aca0bdf9bec3990ace2d0b105b355ba',
    'predecessor_pending_fields_are_preserved_historical_state': True,
    'closure_resolves_only_exact_named_pending_identities': True,
    'original_pipeline': {'session':33822, 'exit_code':1},
    'recorded_audit_error': 'scoped work deadline or clock disagreement',
    'precise_failed_guard_condition_established': False,
    'numerical_model_failure_OOM_or_clock_skew_established': False,
    'all_four_numerical_products_missing_at_original_capture': True,
    'all45_original_remote_identities_absent_twice_at_capture': True,
    'failure_capture_79394_and79395_absent_twice_by_original_query93447': True,
    'original_proxy78014_exit_code': 0,
    'local_groups51576_55854_56202_and_exact_proxy_absent': True,
    'original_closure_and_postpublication_receipts': {k:v for k,v in pins.items() if 'final_proxy' in k or 'postpublication' in k},
    'later_source_suggestions_did_not_modify_or_rerun_original_helper': True,
    'original_success_collector47PIDobserver_finalvalidator': 'unexecuted_and_prohibited_after_failed_prerequisite',
    'original4728_cell_accounting': {'completed':2899,'not_completed':1817,'timed_current':12,'stages':30},
    'distinct_observed_history_cell_scope': 2568,
    'historical_stage_payload_eb0d983f_not_located': True,
    'scientific_result_admission': False, 'scientific_retry_or_phase_reset': False,
    'current_manuscript_sha256': 'a31587aeaf19f79893aaf297f00f4759516de34d2086f57f2966b3edee08be9e',
    'current_body_sha256': '311c7120730c47a86aee08d8f528697b919ad84c96fc887795bb6c4b8266f0e8',
    'current_status_change': 'one_appendix_status_paragraph; title_abstract_main_and_old_results_unchanged',
    'native_compilation_success_saved_receipt': True,
    'exported_manuscript_PDF_inspection_and_author_verification_pending': True,
    'publication_admission': False,
})
references.append({'path':'outputs/revision_report.pdf','bytes':1004240,'sha256':'bb76664ad0aea0989438af9ad7474f1fd0d109aa6482816e255ef9a741149cbb','reason_not_copied':'Unchanged previously published 21-page report, root confirmed reference-only. Not reopened by this assembler.'})
dump('references.json', {'schema':'d3_final_closure_reference_inventory_v1','references':references,'old_scientific_collections_or_results_reopened':False})
dump('copy_inventory.json', {'schema':'d3_final_closure_exact_copy_inventory_v1','files':copies,'exact_copy_count':len(copies)})
for row in copies:
    assert (OUT / row['destination']).read_bytes() == read(row['source'], row['sha256'])
files = {}
for path in sorted(OUT.rglob('*')):
    assert not path.is_symlink()
    if path.is_file():
        raw = path.read_bytes()
        files[str(path.relative_to(OUT))] = {'bytes':len(raw),'sha256':sha(raw)}
assert len(files) < 150 and sum(x['bytes'] for x in files.values()) < 5_000_000
dump('addendum_manifest.json', {'schema':'d3_final_closure_source_metadata_addendum_manifest_v1','files':files,'scientific_admission':False,'publication_admission':False,'excludes':['addendum_manifest.json','sealing_verification.json']})
dump('sealing_verification.json', {'schema':'d3_final_closure_local_seal_verification_v1','status':'passed_exact_copy_and_selected_metadata_checks','exact_copies':len(copies),'files_including_manifest_and_verification':len(files)+2,'bytes_excluding_manifest_and_verification':sum(x['bytes'] for x in files.values()),'manifest_sha256':sha((OUT/'addendum_manifest.json').read_bytes()),'scientific_tests_compiler_clock_live_probes_or_repository_writes':False,'predecessors_modified':False})
print(json.dumps({'status':'sealed_local_complementary_closure_addendum','manifest_sha256':sha((OUT/'addendum_manifest.json').read_bytes()),'files':len(files)+2,'exact_copies':len(copies),'repository_written':False}))
