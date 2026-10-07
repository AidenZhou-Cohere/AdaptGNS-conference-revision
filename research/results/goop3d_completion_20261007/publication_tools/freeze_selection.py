"""Freeze the parent-specified integration and status sources; no fork writes or science."""
from pathlib import Path
import hashlib
import json

w = Path(__file__).resolve().parents[3]
b = Path('work/goop3d_completion_20261007')
p = w / b / 'publication_plan_v2'

def pin(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

def fresh(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == raw, 'prior preparation output differs: ' + str(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)

prior = json.loads((w / b / 'publication_plan_v1/plan.json').read_text())
existing = {e['destination'] for e in prior['files']} - {'README.md'}
added = []
def add(source, destination, role):
    source = Path(source)
    assert destination not in existing and destination not in {e['destination'] for e in added}, destination
    added.append({'source': str(source), 'destination': destination, 'role': role, **pin(w / source)})

# Parent explicitly assigned the whole sealed integration evidence directory.
for q in sorted((w / b / 'integration_root_v1').iterdir()):
    if q.is_file():
        add(q.relative_to(w), str(q.relative_to(w / b)), 'reviewed_manuscript_integration_or_original_backup')
# Review directory declared stable by independent reviewer at this snapshot.
for q in sorted((w / b / 'review_v1').iterdir()):
    if q.is_file() and str(q.relative_to(w / b)) not in existing:
        add(q.relative_to(w), str(q.relative_to(w / b)), 'independent_review_or_preserved_review_history')
manifest = json.loads((w / b / 'integration_root_v1/integration_manifest.json').read_text())
canonical = {}
for original, pinned in manifest['outputs'].items():
    assert pin(w / original) == pinned, original
    dest = 'manuscript/' + original
    add(original, dest, 'reviewed_canonical_manuscript_source')
    canonical[original] = dest
inputs = ['work/sources/aistats2027.sty', 'work/pilot_results.tex', 'work/rollout_results.tex',
          'work/conference_experiments_main.tex', 'work/goop3d_observed_appendix.tex', 'work/goop3d_autonomous_appendix.tex',
          'work/full_evaluation_main.tex', 'work/full_evaluation_appendix.tex', 'work/full_action_main.tex', 'work/full_action_appendix.tex',
          'work/graph_bridge_main.tex', 'work/graph_bridge_appendix.tex', 'work/native_followup_appendix.tex',
          'work/optional_exposure_main.tex', 'work/optional_exposure_appendix.tex', 'work/noise_augmentation_appendix.tex',
          'work/tie_symmetry_appendix.tex', 'work/nonadditive_allocation_appendix.tex', 'work/manuscript_body.tex']
closure = {'schema': 'goop3d_observed_manuscript_builder_inputs_v1',
           'builder': {'source': 'work/build_manuscript.py', **pin(w / 'work/build_manuscript.py')},
           'generated_manuscript': {'source': 'outputs/revised_manuscript.tex', **pin(w / 'outputs/revised_manuscript.tex')},
           'files': {}, 'absent_optional_inputs': [], 'build_or_scientific_code_executed': False,
           'note': 'Exact source/input closure for the already built and natively compiled observed manuscript. Optional absent inputs stay absent; no autonomous insertion is fabricated.'}
for original in inputs:
    if (w / original).exists():
        closure['files'][original] = pin(w / original)
        if original not in canonical:
            add(original, 'manuscript/' + original, 'unchanged_manuscript_builder_input')
    else:
        closure['absent_optional_inputs'].append(original)
fresh(p / 'manuscript_builder_inputs.json', closure)
add(b / 'publication_plan_v2/manuscript_builder_inputs.json', 'manuscript/builder_inputs.json', 'manuscript_input_closure')
for name in ['plan.json', 'scope_review_note.json', 'real_source_verification_attempt1.log']:
    add(b / 'publication_plan_v1' / name, 'publication_history/plan_v1/' + name, 'superseded_curation_history')
for name in ['prepare_successor.py', 'freeze_selection.py', 'selection_preparation_attempt1_failure.json']:
    add(b / 'publication_plan_v2' / name, 'publication_tools/' + name, 'publication_tool_source_or_preserved_failure')
add(b / 'publication_plan_v2/PACKAGE_README.md', 'README.md', 'package_scope_and_instructions')

status_root = b / 'root/status_observed_integration_0248'
status_receipt = json.loads((w / status_root / 'status_final_receipt.json').read_text())
author_originals = ['outputs/README.md', 'outputs/research_findings.md', 'outputs/submission_checklist.md', 'outputs/author_handoff_20261007.md']
status_documents = []
status_scope = {'schema': 'goop3d_completion_author_status_snapshot_v1', 'author_facing_absolute_path_copies': {},
                'repository_root_documents_staged_separately_by_owner': {},
                'ongoing_live_continuation_or_autonomous_products_copied': False}
for original in author_originals:
    expected = status_receipt[original]
    assert pin(w / original) == expected, 'parent-sealed status document changed: ' + original
    snapshot = p / 'author_status_snapshot' / original
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        assert pin(snapshot) == expected
    else:
        with snapshot.open('xb') as stream:
            stream.write((w / original).read_bytes())
    destination = 'status_documents/author_facing/' + original
    add(snapshot.relative_to(w), destination, 'author_facing_status_snapshot_with_absolute_paths')
    status_documents.append(destination)
    status_scope['author_facing_absolute_path_copies'][original] = {'destination': destination, **expected}
for original in ['outputs/AdaptGNS/README.md', 'outputs/AdaptGNS/AUTHOR_HANDOFF.md']:
    assert pin(w / original) == status_receipt[original], 'parent-sealed repository-root status changed'
    status_scope['repository_root_documents_staged_separately_by_owner'][original] = status_receipt[original]
for q in sorted((w / status_root).rglob('*')):
    if q.is_file():
        add(q.relative_to(w), str(q.relative_to(w / b)), 'status_edit_receipt_or_original_backup')
fresh(p / 'status_document_scope.json', status_scope)
add(b / 'publication_plan_v2/status_document_scope.json', 'status_documents/source_scope.json', 'status_snapshot_scope')
fragment_root = 'presentation_observed_v1/generated_candidate_v4'
receipt = json.loads((w / b / fragment_root / 'receipt.json').read_text())
seal = {'schema': 'goop3d_completion_sealed_publication_successor_v1',
        'prior_plan': {'source': str(b / 'publication_plan_v1/plan.json'), **pin(w / b / 'publication_plan_v1/plan.json')},
        'replace': [{'destination': 'README.md', 'preserve_old_as': 'publication_history/plan_v1/PACKAGE_README.md'}],
        'add': added,
        'final_presentation': {'review': 'review_v1/observed_editorial_v4_review.json',
                              'renderer': 'presentation_observed_v1/render_observed.py', 'receipt': fragment_root + '/receipt.json',
                              'combined_appendix': fragment_root + '/appendix_goop3d_observed.tex',
                              'prior_editorial_review': 'review_v1/observed_editorial_v3_review.json',
                              'original_presentation_review': 'review_v1/observed_presentation_v2_review.json',
                              'published_receipt_members': [n for n in receipt['files'] if not n.endswith('.gz') and n != 'source_summary_unchanged.json']},
        'integration': {'review': 'review_v1/observed_manuscript_integration_review.json',
                        'compile_receipt': 'integration_root_v1/compile_and_exact_delta.json', 'manifest': 'integration_root_v1/integration_manifest.json',
                        'manuscript': 'manuscript/outputs/revised_manuscript.tex', 'canonical_sources': canonical, 'status_documents': status_documents},
        'pending': ['Autonomous completion and independent actual-product review', 'Fresh PDF export and final visual/page inspection', 'Author verification and submission decision'],
        'scope': 'Independently reviewed observed-history completion with final v4 presentation, exact compiled observed manuscript and retained unsuccessful histories. Autonomous numerical products remain reference-only.'}
fresh(p / 'input_seal.json', seal)
print(json.dumps({'status': 'explicit_successor_inputs_sealed', 'seal': pin(p / 'input_seal.json'),
                  'additions': len(added), 'added_bytes': sum(e['bytes'] for e in added), 'builder_inputs': len(closure['files']),
                  'absent_optional_inputs': closure['absent_optional_inputs']}, sort_keys=True))
