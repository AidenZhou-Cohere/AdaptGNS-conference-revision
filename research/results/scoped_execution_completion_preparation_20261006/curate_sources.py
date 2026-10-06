"""Explicit, local source/review curation. Never read live job trees or data."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', required=True, type=Path,
                    help='Original research workspace containing work/ and outputs/AdaptGNS/')
ROOT = parser.parse_args().workspace.resolve()
WORK = Path(__file__).resolve().parent
SRC = ROOT/'work/deadline_research_20261005/cuda_preparation'
FAMILY = 'scoped_execution_completion_preparation_20261006'
OUT = ROOT/'outputs/AdaptGNS/research/results'/FAMILY
PREDECESSOR = ROOT/'outputs/AdaptGNS/research/results/goop3d_vectorized_preparation_20261006'
REL = Path('work/deadline_research_20261005/cuda_preparation')

PINS = {
    'prepare_goop3d_final_cohort_v1.py':'add53ea8c41ef44a1b3248eaacd4df38b49d396b8ecd374d9f928ba751bc631a',
    'prepare_goop3d_reserved_test_v1.py':'cdd17f625e0129926f69e81cfe5961c05e31aed7ca875d2550f40e642434523a',
    'summarize_goop3d_graph_support_v1.py':'d7096bef018ac7e3d28cc429a4711d88209e48a1b812dda0aad995d6a41de5fd',
    'supervise_goop3d_final_evaluation_v1.py':'070d1d5e47cb4ac8650586f72ef434756c47b9b94da9d15016f7a287e945e989',
    'supervise_goop_evaluation_gpu_scoped_v3.py':'a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21',
    'summarize_goop_graph_support_scoped_v3.py':'0772f5615a69f513beed721414247ba08ede80b8f3ad6ab7201336655d996c5b',
    'supervise_sand_scoped_science_v1.py':'2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13',
    'goop_gpu_scoped_evaluation_amendment_v3.md':'0ecb8d3c6b5dd74150d6253cc84be2548827b235490eb35da50afcc0df6e28f4',
    'sand_scoped_operational_amendment_v1.md':'411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738',
    'sand_scoped_schedule_fixed_spec_v2.json':'403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b',
    'goop3d_scientific_protocol_v1.md':'5010f9023a3eee45f85bee80b35fc8e506ca145667daf75027b32c92658faa70',
    'goop3d_stopped_timing_and_decision_v1/cuda_preparation/goop3d_scientific_endpoint_plan_root_v1.json':'f38131f2851afdcbf774af506ebe8e02c8eca1be5e143430f68abedb95304c6d',
}
CURRENT = set(PINS) | set('''
test_prepare_goop3d_final_cohort_v1.py test_prepare_goop3d_reserved_test_v1.py
test_summarize_goop3d_graph_support_v1.py test_goop3d_scalar_producer_integration_v1.py
test_supervise_goop3d_final_evaluation_v1.py test_supervise_goop_evaluation_gpu_scoped_v3.py
test_summarize_goop_graph_support_scoped_v3.py test_supervise_sand_scoped_science_v1.py
goop3d_final_cohort_freeze_notes_v1.md goop3d_final_cohort_freeze_release.template.json
goop3d_scientific_stopped_receipt.template.json
goop3d_reserved_test_preparation_notes_v1.md goop3d_reserved_test_preparation_release.template.json
goop3d_scalar_final_ledger_interface_v1.md goop3d_scalar_collection_release.template.json
goop3d_final_evaluation_ledger.template.json goop3d_scalar_analysis_release.template.json
goop3d_final_evaluation_supervisor_notes_v1.md goop3d_final_evaluation_supervisor_release.template.json
goop_scoped_scalar_release_interface_v3.md
sand_scoped_supervisor_operations_v1.md sand_scoped_schedule_fixed_spec_v2.md
sand_scoped_scientific_release.template.json sand_scoped_process_check.template.json sand_scoped_clock_check.template.json
goop3d_scientific_protocol_v1.md
'''.split())

# Include the module imported immediately by the new Goop scalar entry and its
# synthetic fixture. Other immutable runtime helpers are bound to published
# predecessor paths below, rather than silently sourced from the local worktree.
COPIED_DEPENDENCIES = {
    'summarize_goop_graph_support_quota_v2.py':'da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c',
    'test_summarize_goop_graph_support_quota_v2.py':'bea0a6fbb2c28d1a38e4446caac09a4d22c2d3031401cb8f8f7cdba2a992d11a',
}
REVIEW_ROOTS = (
    'goop3d_final_cohort_', 'goop3d_reserved_test_', 'goop3d_scalar_',
    'goop3d_final_supervisor_', 'goop_scoped_scalar_', 'goop_gpu_scoped_v3_',
    'sand_scoped_science_', 'sand_scoped_self_argv_', 'sand_scoped_wrong_gpu_',
    'supervisor_lexical_', 'lexical_interpreter_adapter_', 'lexical_python_independent_',
    'goop3d_lexical_delta_', 'goop3d_final_operations_gap_review_',
)
HISTORY_DIRS = (
    'goop3d_final_supervisor_review_history', 'goop3d_scalar_review_history',
    'goop_scoped_scalar_review_history', 'goop3d_reserved_test_review_history',
    'scoped_supervisor_review_history_code_audit_v1', 'goop3d_final_cohort_review_history',
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


assert not OUT.exists(), 'New family required; do not overwrite a curated bundle'
for name, expected in {**PINS, **COPIED_DEPENDENCIES}.items():
    assert sha(SRC/name) == expected, f'Current approved source changed: {name}'

selection = {}
for name in sorted(CURRENT | set(COPIED_DEPENDENCIES)):
    p = SRC/name
    assert p.is_file() and not p.is_symlink(), name
    role = ('copied_dependency' if name in COPIED_DEPENDENCIES else
            'fixed_prospective_plan_dependency_not_result' if name.endswith('/goop3d_scientific_endpoint_plan_root_v1.json') else
            'current_preparation')
    selection['workspace/'+str(REL/name)] = (p, role)

# Bounded review namespaces contain only author/reviewer source, test logs and
# synthetic probe receipts. Every candidate is enumerated before any copy.
for p in sorted(SRC.iterdir()):
    if p.is_file() and p.name.startswith(REVIEW_ROOTS) and p.suffix in ('.json','.log') and p.name not in CURRENT:
        selection['reviews/'+p.name] = (p, 'review_or_failed_preparation_record')
for dirname in HISTORY_DIRS:
    for p in sorted((SRC/dirname).rglob('*')):
        if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and '__pycache__' not in p.parts:
            selection['review_history/'+str(p.relative_to(SRC))] = (p, 'historical_candidate_not_execution_approved')

for destination, (p, kind) in selection.items():
    assert p.stat().st_size < 200_000, f'Unexpectedly large preparation file: {p}'
    assert not p.is_symlink() and not any(s in p.name.lower() for s in ('ssh_config','credential','.pt','.npz','.tfrecord'))
    if p.suffix == '.json' and p.stat().st_size:
        json.loads(p.read_text())
    raw = p.read_text()
    assert not re.search(r'-----BEGIN .*PRIVATE KEY-----|\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b', raw), str(p)

old_manifest = json.loads((PREDECESSOR/'manifest.json').read_text())
assert sha(PREDECESSOR/'manifest.json') == '98624ef49322ebcc28ed92561f612d0effbd7719df8e3a160f75c69bdb3b2116'
# The full published workspace closure is an exact dependency, not fresh
# execution evidence. Include explicit paths and hashes for its source files.
dependencies = {}
for name, record in old_manifest['files'].items():
    if name.startswith('workspace/'):
        p = PREDECESSOR/name
        assert sha(p) == record['sha256'], 'Published dependency changed: '+name
        dependencies[name.removeprefix('workspace/')] = {
            'published_path':'../goop3d_vectorized_preparation_20261006/'+name,
            'sha256':record['sha256'], 'bytes':record['bytes'],
            'overlay': ('workspace/'+name.removeprefix('workspace/')) in selection,
        }

OUT.mkdir()
files = {}
for destination, (p, kind) in selection.items():
    target = OUT/destination; target.parent.mkdir(parents=True,exist_ok=True)
    before = sha(p); shutil.copyfile(p,target)
    assert sha(p) == before == sha(target), str(p)
    files[destination] = {'sha256':before,'bytes':p.stat().st_size,'source':str(p.relative_to(ROOT)),'kind':kind}

write(OUT/'dependency_paths.json', {
    'predecessor_manifest':'../goop3d_vectorized_preparation_20261006/manifest.json',
    'predecessor_manifest_sha256':sha(PREDECESSOR/'manifest.json'),
    'restoration':'Restore the predecessor workspace files to their listed relative paths, then overlay this family workspace. Never replace active sources or outputs.',
    'files':dependencies,
})
write(OUT/'current_approved_pins.json', PINS)
write(OUT/'source_inventory.json',files)
write(OUT/'scope.json',{
    'kind':'reviewed_operation_preparation_not_scientific_results',
    'current_primary_python_entries':[n for n in PINS if n.endswith('.py')],
    'not_included':['live outputs','datasets','checkpoints','SSH configuration','credentials','unrelated process inventories','unfinished Sand evaluation adapter','stopped D3 timing payloads and decision archive'],
    'historical_approvals_superseded':{
        'supervise_goop3d_final_evaluation_v1.py':'fbe7a03135a1f1ba3482200fd8b2375389c656d2cd81c22c35df848e9b052f62',
        'supervise_sand_scoped_science_v1.py':'8775048fd9ee741b479f8bdd361861a493241cf158d1dc64b3b01b60475478a0',
    },
    'current_lexical_review':'reviews/supervisor_lexical_python_final_independent_review_code_audit_v1.json',
    'science_execution_or_test_admission_granted':False,
})
write(WORK/'selected_files.json',files)
print(json.dumps({'family':str(OUT),'copied_files':len(files),'copied_bytes':sum(v['bytes'] for v in files.values()),'predecessor_dependencies':len(dependencies)},indent=2))
