"""Hash exact local source/metadata copies and write an inert complementary plan."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/aiden.zhou/Documents/Codex/2026-10-04/why-cangmai')
HERE = ROOT / 'work/d3_observed_failure_closure_publication_plan_UNADMITTED_20261007_v1'
CANDIDATE = ROOT / 'work/d3_observed_failure_closure_addendum_UNADMITTED_20261007_v1'
MANIFEST_PIN = '7d8436e9f8d2550dcf0ba420a423d613a421c1c242900cf06f2562597e038db7'
REVIEW = ROOT / 'work/d3_observed_failure_closure_addendum_independent_inventory_review_v1.json'
REVIEW_PIN = '0efe4eb9dc5f5c6e0ad00f2cbf22f2476c2629f628db54a305c4812fe5b73a10'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size < 1_000_000
    return path.read_bytes()

def write(name, doc):
    with (HERE / name).open('x') as stream:
        stream.write(json.dumps(doc, indent=2, sort_keys=True) + '\n')

manifest_raw = read(CANDIDATE / 'addendum_manifest.json')
assert sha(manifest_raw) == MANIFEST_PIN
manifest = json.loads(manifest_raw)
paths = [p for p in sorted(CANDIDATE.rglob('*')) if p.is_file()]
assert len(paths) == 64
assert not any(p.is_symlink() for p in CANDIDATE.rglob('*'))
assert {str(p.relative_to(CANDIDATE)) for p in paths} == set(manifest['files']) | {'addendum_manifest.json','sealing_verification.json'}
for name, row in manifest['files'].items():
    raw = read(CANDIDATE / name)
    assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
assert sha(read(REVIEW)) == REVIEW_PIN
ast.parse(read(HERE / 'copy_verify.py').decode())

entries = []
def add(path, dest):
    raw = read(path)
    entries.append({'source':str(path.relative_to(ROOT)),'destination':dest,'bytes':len(raw),'sha256':sha(raw)})

for path in paths:
    add(path, 'addendum/' + str(path.relative_to(CANDIDATE)))
for source, dest in (
    ('PACKAGE_README.md','README.md'),
    ('copy_verify.py','copy_verify.py'),
    ('assembly_receipt.json','assembly_receipt.json'),
    ('HANDOFF.md','root_handoff.md'),
    ('build_plan.py','publication_plan_builder.py'),
):
    add(HERE / source, dest)
add(REVIEW, 'independent_addendum_review.json')
assert len(entries) == 70 and len({r['destination'] for r in entries}) == len(entries)
assert sum(r['bytes'] for r in entries) < 5_000_000
plan = {
    'schema':'d3_final_closure_complementary_inert_publication_plan_v1',
    'workspace':str(ROOT),
    'destination':'research/results/goop3d_observed_history_failure_closure_20261007',
    'expected_git_head':'b2517a4677dfc4dff9a31043da7f90f783d66943',
    'base_requirement':{
        'destination':'research/results/goop3d_observed_history_failure_20261007',
        'publication_manifest_sha256':'30d8b469aad2310c6abf0e9091d942b3dd9ad422aa24060c33a0d4a800fc4885',
        'copy_plan_sha256':'3c96b82f8abcab42a4bb5fda7ac1a1ecf83f9995250eaa6ed0dcc2466c3f7fc7',
        'files':213,
    },
    'addendum_manifest_sha256':MANIFEST_PIN,
    'independent_addendum_review_sha256':REVIEW_PIN,
    'entries':entries,'file_count':len(entries),'total_bytes':sum(r['bytes'] for r in entries),
    'root_owned_activation':True,'scientific_result_admission':False,
    'old_or_new_publication_preflight_run_during_plan_preparation':False,
    'repository_tracked_files_overwritten':False,
}
write('publication_plan.json', plan)
plan_pin = sha(read(HERE / 'publication_plan.json'))
write('static_packaging_checks.json', {
    'schema':'d3_final_closure_static_copy_plan_checks_v1',
    'status':'passed_exact_local_source_hash_and_helper_syntax_checks',
    'plan_sha256':plan_pin,
    'helper_sha256':sha(read(HERE / 'copy_verify.py')),
    'addendum_manifest_sha256':MANIFEST_PIN,
    'independent_addendum_review_sha256':REVIEW_PIN,
    'source_entries':len(entries),'source_bytes':plan['total_bytes'],
    'helper_AST_parsed_without_execution':True,
    'old_or_new_preflight_or_copy_helper_executed':False,
    'git_commands_science_tests_compiler_clocks_live_probes_or_repo_writes':False,
})
files = {}
for path in sorted(HERE.iterdir()):
    assert path.is_file() and not path.is_symlink()
    raw = read(path)
    files[path.name] = {'bytes':len(raw),'sha256':sha(raw)}
write('plan_manifest.json', {
    'schema':'d3_final_closure_inert_publication_plan_bundle_manifest_v1',
    'files':files,'excludes':['plan_manifest.json'],
    'root_owned_publication':True,'copy_helper_executed':False,
})
print(json.dumps({'status':'sealed_inert_complementary_plan','files_to_copy':len(entries),
                  'bytes_to_copy':plan['total_bytes'],'plan_sha256':plan_pin,
                  'helper_sha256':sha(read(HERE/'copy_verify.py')),
                  'bundle_manifest_sha256':sha(read(HERE/'plan_manifest.json')),
                  'repository_written_or_preflight_run':False}))
