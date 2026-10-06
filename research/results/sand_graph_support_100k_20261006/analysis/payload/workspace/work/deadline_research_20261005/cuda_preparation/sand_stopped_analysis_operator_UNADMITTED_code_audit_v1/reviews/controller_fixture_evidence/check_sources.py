#!/usr/bin/env python3
"""Local source/receipt hash and AST equivalence checks; no actual science access."""
import ast
import hashlib
import json
from pathlib import Path

PREP = Path(__file__).resolve().parent.parent
K = PREP / 'sand_stopped_analysis_operator_UNADMITTED_code_audit_v1'
OUT = Path(__file__).resolve().parent / 'source_checks.json'
REMOTE = '/root/repos/AdaptGNS-cuda-20261006'
checks = []

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_bytes())
def require(value, name, **details):
    if not value: raise AssertionError(name)
    checks.append(dict(check=name, passed=True, **details))
def same(left, right, label):
    require(sha(left) == sha(right), label, left=str(left), right=str(right), sha256=sha(left))

b = read(K / 'bindings.json')
source_pins = {}
for remote, binding in b['source_files'].items():
    package = K / binding['package_path']
    relative = remote.removeprefix(REMOTE + '/cuda_preparation/')
    if relative.startswith('sand_saved_array_audit_v2/'):
        relative = relative.replace('sand_saved_array_audit_v2/', 'sand_saved_array_audit_UNADMITTED_transition_v2/', 1)
    original = PREP / relative
    require(sha(package) == sha(original) == binding['sha256'], 'Frozen source bytes identical',
            path=str(package), original=str(original), sha256=sha(package))
    source_pins[remote] = binding['sha256']
for name in ['sand_collect_A', 'sand_collect_B', 'sand_saved_A', 'sand_saved_B', 'sand_summarize', 'sand_paired']:
    same(K / 'candidates' / (name + '.cpu_release.candidate.json'),
         PREP / 'stopped_analysis_cpu_UNADMITTED_transition_v1' / (name + '.cpu_release.candidate.json'),
         'Frozen CPU candidate identical: ' + name)
for name in ['Sand.analysis_phase.candidate.json', 'Sand.A.evaluation_closure.candidate.json', 'Sand.B.evaluation_closure.candidate.json']:
    same(K / 'candidates' / name, PREP / 'stopped_analysis_cpu_UNADMITTED_transition_v1' / name, 'Frozen phase/closure candidate identical: ' + name)
for name in ['A.collection_release.candidate.json', 'B.collection_release.candidate.json', 'summary_release.candidate.json']:
    same(K / 'candidates' / name, PREP / 'sand_stopped_analysis_UNADMITTED_transition_v2' / name, 'Frozen scalar core candidate identical: ' + name)
same(K / 'operations.json', PREP / 'stopped_analysis_cpu_UNADMITTED_transition_v1/operations.json', 'Complete ten-operation map identical')
for role in ('A', 'B'):
    path = K / 'candidates' / (role + '.original_evaluation_release.json')
    same(path, PREP / 'sand_final24_released_root_v1' / (role + '.evaluation_release.json'), 'Original evaluation release identical: ' + role)
    require(sha(path) == b['roles'][role]['release_sha256'], 'Original evaluation release bound: ' + role)

def definitions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
old = definitions(K / 'sources/cuda_preparation/supervise_sand_post_completion_cpu_v2.py')
shared = definitions(K / 'sources/cuda_preparation/supervise_stopped_analysis_cpu_v1.py')
for name in b['qualification_reuse']['identical_top_level_definitions']:
    require(old[name] == shared[name], 'Previously qualified AST definition identical: ' + name)
remote_assignments = {}
for node in ast.parse((K / 'remote_probe.py').read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        try: remote_assignments[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError): pass
require(remote_assignments['EVALUATIONS'] == b['roles'], 'Remote original session/owner/startup/GPU/boot/history bindings exact')
require(remote_assignments['SOURCE_PINS'] == source_pins, 'Remote staged-source allowlist equals all frozen source pins')
for name, wanted in b['original_evidence_sha256'].items():
    require(sha(Path(name)) == wanted, 'Original accepted evidence bytes pinned', path=name, sha256=wanted)
require(b['qualification_reuse']['accepted_review_sha256'] ==
        b['original_evidence_sha256'][str(PREP / 'sand_native_qualification_completed_independent_review_code_audit_v1.json')],
        'Qualification reuse accepted review is in enforced original evidence pins')
require(b['environment_expected_sha256'] == {
    REMOTE + '/.venv/bin/python': '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a',
    '/usr/bin/python3.12': '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a',
    REMOTE + '/.venv/pyvenv.cfg': 'b7d360e62970717794ce0ab4f1e1398a49a8a73a138d34ca920b1eb251fe2f3b',
    '/usr/bin/timeout': '2db30bc57746c940a0643581cd5e2671e505442bea16164143f0ea8ff4e36453'},
        'Runtime pins equal previously reviewed qualified runtime')
require(sha(K / 'review_products.py') == 'f96741333ae17eb018a39fe6b69144a2976c13c288e23b95f0d92cd157a6592f',
        'Separately independently reviewed product helper unchanged')
require(sha(PREP / 'sand_stopped_product_validator_independent_transition_v1.json') ==
        '9af90d8654895bab50842bf89624559eda1badefda2713a8b3e51603ee2667d5',
        'Separate product helper independent receipt unchanged')
with OUT.open('x') as f:
    json.dump({'scope': 'local frozen sources/candidates and existing control/qualification metadata hash verification only',
               'check_count': len(checks), 'checks': checks,
               'actual_scientific_artifacts_or_probes_or_phase_clocks_accessed': False}, f, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps({'check_count': len(checks), 'result': 'passed', 'output': str(OUT), 'sha256': sha(OUT)}))
