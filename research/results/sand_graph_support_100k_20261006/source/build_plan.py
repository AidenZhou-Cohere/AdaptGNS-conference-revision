"""Create an inert bounded copy plan. No repository destination is changed."""
import hashlib
import json
from pathlib import Path

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[1]
REPO = ROOT / 'outputs/AdaptGNS'
PREP = ROOT / 'work/deadline_research_20261005/cuda_preparation'
PRESENTATION = ROOT / 'work/conference_presentation_20261006'
SOURCE = PRESENTATION / 'cross_material_candidate_statistics_v1'
CANDIDATE = ROOT / 'work/sand_completed_analysis_publication_candidate_UNADMITTED_20261006_v2'
DESTINATION = 'research/results/sand_graph_support_100k_20261006'
MANUSCRIPT_PIN = 'd4ed535ce1ec78803d85ba37449c5fe6eec7d32c9e55303e7fff2cdec71c76fe'
CANDIDATE_PIN = '16b9f66a04230fbb83b410cc007b1bce14b9bbc05ce93ef9b98b8780fd749029'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


assert sha(ROOT / 'outputs/revised_manuscript.tex') == MANUSCRIPT_PIN
assert sha(CANDIDATE / 'candidate_manifest.json') == CANDIDATE_PIN
assert not (REPO / DESTINATION).exists()
entries = []
aliases = []
destinations = set()


def add(path, destination, role, pin=None):
    assert path.is_file() and not path.is_symlink() and path.is_relative_to(ROOT), str(path)
    assert path.stat().st_size < 12_000_000, str(path)
    assert destination not in destinations
    actual = sha(path)
    assert pin is None or pin == actual, str(path)
    destinations.add(destination)
    entries.append({'source': str(path.relative_to(ROOT)), 'destination': destination,
                    'bytes': path.stat().st_size, 'sha256': actual,
                    'role': role, 'transformation': 'none'})


for path in sorted(CANDIDATE.rglob('*')):
    if path.is_file():
        add(path, 'analysis/' + str(path.relative_to(CANDIDATE)), 'exact_immutable_admitted_analysis_candidate_v2')
add(ROOT / 'outputs/revised_manuscript.tex', 'presentation/revised_manuscript.tex', 'exact_integrated_standalone_manuscript', MANUSCRIPT_PIN)

canonical = ['build_manuscript.py', 'sources/aistats2027.sty', 'manuscript_body.tex',
             'pilot_results.tex', 'rollout_results.tex', 'conference_experiments_main.tex',
             'full_evaluation_main.tex', 'full_evaluation_appendix.tex', 'full_action_main.tex',
             'full_action_appendix.tex', 'graph_bridge_main.tex', 'graph_bridge_appendix.tex',
             'native_followup_appendix.tex', 'optional_exposure_main.tex', 'optional_exposure_appendix.tex',
             'noise_augmentation_appendix.tex', 'tie_symmetry_appendix.tex', 'nonadditive_allocation_appendix.tex']
for name in canonical:
    add(ROOT / 'work' / name, 'canonical/work/' + name, 'exact_canonical_builder_or_input')

# Preserve current generated v4 completely. Earlier identical bytes are pinned
# aliases; all distinct earlier bytes and every failure/source version are copied.
v4_by_sha = {}
for path in sorted((SOURCE / 'generated_v4').iterdir()):
    assert path.is_file()
    dest = 'presentation_candidate/generated_v4/' + path.name
    add(path, dest, 'preferred_generated_v4_exact_output')
    v4_by_sha[sha(path)] = dest
for dirname in ('generated_v1', 'generated_v2', 'generated_v3'):
    for path in sorted((SOURCE / dirname).iterdir()):
        assert path.is_file()
        digest = sha(path)
        if digest in v4_by_sha:
            aliases.append({'source': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                            'sha256': digest, 'exact_published_bytes': v4_by_sha[digest],
                            'reason': 'Earlier generation bytes identical to preferred v4; no duplicate content required.'})
        else:
            add(path, 'history/presentation_generations/' + dirname + '/' + path.name,
                'exact_distinct_prior_generation_or_failure')
for path in sorted(SOURCE.iterdir()):
    if path.is_file():
        add(path, 'presentation_candidate/' + path.name, 'exact_presentation_source_admission_or_integration_handoff')
for path in sorted((SOURCE / 'history_before_populated_layout').iterdir()):
    assert path.is_file()
    add(path, 'history/presentation_source/' + path.name, 'exact_prior_presentation_source_or_review')
for path in sorted((PRESENTATION / 'root_sand_integration_v1').iterdir()):
    assert path.is_file()
    add(path, 'verification/root_sand_integration_v1/' + path.name, 'exact_root_integration_before_source_or_receipt')
for name in ('cross_material_presentation_plan_before_sand_admission_statistics_v1.json',
             'cross_material_presentation_plan_before_sand_admission_statistics_v1.md'):
    add(PRESENTATION / name, 'presentation_candidate/' + name, 'fixed_pre_admission_cross_material_presentation_plan')
for name, role in (
    ('sand_completed_curation_independent_root_review_v2.json', 'accepted_root_curation_review_1031_checks'),
    ('sand_curation_review_initial_schema_error_root_v1.json', 'preserved_root_review_failure_receipt'),
):
    add(PREP / name, 'reviews/' + name, role)
for name in ('review_sand_curation_root_v1.py', 'review_sand_curation_root_v2.py'):
    add(ROOT / 'work' / name, 'source/' + name, 'exact_root_curation_reviewer_source_history')

write(PLAN / 'prior_generation_byte_aliases.json', {'schema': 'sand_prior_presentation_exact_byte_aliases_v1',
      'aliases': aliases, 'all_prior_distinct_bytes_and_failures_copied': True,
      'regeneration_or_statistics_recomputation': False})
add(PLAN / 'prior_generation_byte_aliases.json', 'history/prior_generation_byte_aliases.json', 'lossless_prior_generation_identity_map')
for name, destination in (('PACKAGE_README.md', 'README.md'),
                          ('publication_copy_verify.py', 'source/publication_copy_verify.py'),
                          ('build_plan.py', 'source/build_plan.py')):
    add(PLAN / name, destination, 'publication_plan_documentation_or_exact_copy_source')

count = len(entries)
total = sum(entry['bytes'] for entry in entries)
assert count <= 400 and total <= 40_000_000
plan = {
    'schema': 'sand_inert_publication_copy_plan_v1',
    'status': 'READY_FOR_ROOT_REVIEW_NOT_EXECUTED',
    'workspace': str(ROOT),
    'repository': str(REPO),
    'expected_git_head': '2fa58526f83d2d0e81df01eb4c4658c5aba0d0f4',
    'expected_git_branch': 'research/conference-revision',
    'destination': DESTINATION,
    'destination_must_be_absent': True,
    'entries': entries,
    'file_count': count,
    'total_bytes': total,
    'maximum_files': 400,
    'maximum_total_bytes': 40_000_000,
    'maximum_single_file_bytes': 12_000_000,
    'manuscript_sha256': MANUSCRIPT_PIN,
    'analysis_candidate_manifest_sha256': CANDIDATE_PIN,
    'root_scientific_admission_sha256': 'bab36c91925e8368417206f258b8829f0cda78ec9e423b551b769d48332eca64',
    'root_curation_review_sha256': 'eb482b0df7fb2732474885445f3d573c200aa7ee1f620cbfa560239fe069776c',
    'prior_generation_alias_count': len(aliases),
    'prior_generation_alias_bytes_saved': sum(alias['bytes'] for alias in aliases),
    'no_scientific_rerun_or_new_compilation': True,
    'no_commit_push_remote_process_or_automation_action': True,
    'excluded': ['four large original Sand collection/side-audit products', 'models/raw arrays/base64 captures',
                 'credentials and raw transport command logs', 'generated Python caches', 'D3 actual transitions'],
    'separate_pending_addenda': ['original Sand proxy62558/native48605 final exit and cleanup',
                                 'additional independent curation/presentation review when supplied'],
    'publication_scope': 'Exact completed scientific/presentation snapshot; conference readiness and final PDF author checks remain unclaimed.',
}
write(PLAN / 'publication_plan.json', plan)
print(json.dumps({'file_count': count, 'bytes': total, 'alias_count': len(aliases),
                  'alias_bytes_saved': plan['prior_generation_alias_bytes_saved'],
                  'plan_sha256': sha(PLAN / 'publication_plan.json'), 'repository_changed': False}, indent=2))
