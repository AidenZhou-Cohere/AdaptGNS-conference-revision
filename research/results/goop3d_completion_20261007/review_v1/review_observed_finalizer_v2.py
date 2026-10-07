"""Independent source and synthetic checks; never reads scientific arrays/cache."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PACKAGE = BASE / 'observed_finalize_v2'
FROZEN = BASE.parent / 'deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1'
sys.path[:0] = [str(PACKAGE), str(FROZEN)]
import check_observed_exact_variance_v2 as new
import finish_from_cache as finalizer

checks = []
def require(condition, name):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

pins = {
    'finish_from_cache.py': '8b6f46e24d2cc729fa7c585adce8ad6acae0cae85764dc4f3fc7ceaaaf698a6e',
    'check_observed_exact_variance_v2.py': '9dce53c8f9e87ef1d0a828b163469f34c9249f5b242f26c3a0287f11f489bccb',
    'test_finish.py': 'fd8bbf1c6ead9f87e95a7851e741a938f3d5fc40640878b6c452be068b444134',
}
for name, pin in pins.items():
    require(digest(PACKAGE / name) == pin, 'exact final source bytes: ' + name)
old_tree = ast.parse((FROZEN / 'check_goop3d_observed_history_summary_v1.py').read_text())
new_tree = ast.parse((PACKAGE / 'check_observed_exact_variance_v2.py').read_text())
old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
new_functions = {n.name: n for n in new_tree.body if isinstance(n, ast.FunctionDef)}
for name in ('valid', 'average', 'difference', 'equal'):
    require(ast.dump(old_functions[name]) == ast.dump(new_functions[name]), 'unchanged checker function: ' + name)
require(ast.dump(ast.Module(body=old_functions['verify'].body[:-1], type_ignores=[])) ==
        ast.dump(ast.Module(body=new_functions['verify'].body[:-1], type_ignores=[])),
        'entire scientific verify body before receipt unchanged')
old_return = old_functions['verify'].body[-1].value
new_return = new_functions['verify'].body[-1].value
old_fields = {k.value: ast.dump(v) for k, v in zip(old_return.keys, old_return.values)}
new_fields = {k.value: ast.dump(v) for k, v in zip(new_return.keys, new_return.values)}
require(all(new_fields[k] == v for k, v in old_fields.items()), 'every old result receipt field unchanged')
require(set(new_fields) - set(old_fields) == {'checker_revision', 'sample_sd_method', 'comparison_tolerance_unchanged'},
        'only explicit correction metadata added to checker receipt')
require(new.seed_values([102089.95333333334] * 3)['sample_sd'] == 0, 'reported original large-count failure exact zero SD')
require(new.seed_values([7478.366666666667] * 3)['sample_sd'] == 0, 'second actual repeated count exact zero SD')
require(new.seed_values([52078.3] * 3)['sample_sd'] == 0, 'third actual repeated count exact zero SD')
rng = random.Random(717)
synthetic_cases = [[rng.uniform(-1e9, 1e9) for _ in range(3)] for _ in range(100)]
synthetic_cases += [[1e-300, 2e-300, 3e-300], [-1e150, 0., 1e150], [1e100, 1e100, 1e100]]
for xs in synthetic_cases:
    got = new.seed_values(xs)['sample_sd']
    expected = statistics.stdev(xs)
    require(got == expected or abs(got - expected) <= abs(expected) * 1e-15,
            'independent exact variance synthetic case ' + str(len(checks)))
require(new.seed_values([None, 1., 2.])['sample_sd'] is None, 'missing-seed null rule retained')
require(new.seed_values([float('nan'), 1., 2.])['sample_sd'] is None, 'nonfinite-seed null rule retained')
try:
    new.equal(0., 1e-8)
except ValueError:
    require(True, 'original numerical tolerance rejects material discrepancy')
else:
    raise AssertionError('numerical tolerance weakened')

tree = ast.parse((PACKAGE / 'finish_from_cache.py').read_text())
calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
for forbidden in ('audit_cell', 'initialize_worker', 'ProcessPoolExecutor', 'Popen', 'subprocess.run', 'np.load', 'numpy.load'):
    require(not any(name == forbidden or name.rsplit('.', 1)[-1] == forbidden for name in calls), 'no finisher call: ' + forbidden)
for required in ('runner.prepare', 'runner.merge', 'summarizer.summarize', 'checker.verify', 'fcntl.flock'):
    require(required in calls, 'required exact cache finalization call: ' + required)
source = (PACKAGE / 'finish_from_cache.py').read_text()
for fragment, label in [
    ("with lock_path.open('rb')", 'original lock opened read-only'),
    ('output.mkdir(parents=True, exist_ok=False)', 'fresh output refuses overwrite'),
    ("index['identity'] == identity", 'same source and runtime cache identity'),
    ("len(index['completed']) == 2568", 'all cached rows required'),
    ("not index['failed']", 'failed cached rows rejected'),
    ("set(expected) == set(index['completed'])", 'exact task set required'),
    ("len(store.bindings) == 2569", 'complete cache consumption checked'),
    ("expected_bytes=entry['bytes'], retain=False", 'final scientific file byte checks opaque'),
    ("failure_copy_pin == PRIOR_FAILURE_SHA", 'original failure copied byte-identically'),
    ("'original_arrays_decoded': 0", 'array replay limitation explicit'),
    ("'arbitrary_elapsed_cutoff': False", 'no artificial elapsed phase'),
]:
    require(fragment in source, label)
with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve())) as tmp:
    path = Path(tmp) / 'actual'; path.write_text('synthetic')
    symlink = Path(tmp) / 'link'; symlink.symlink_to(path)
    try:
        finalizer.ordinary(symlink)
    except ValueError:
        require(True, 'synthetic cache symlink refused')
    else:
        raise AssertionError('symlink accepted')

review = {
    'status': 'passed_independent_source_and_synthetic_review',
    'source_sha256': pins,
    'source_checks': len(checks), 'checks': checks,
    'synthetic_seed_vectors': len(synthetic_cases),
    'independently_repeated_author_tests': 5,
    'author_test_original_tool': '7bea15', 'author_test_exit_code': 0,
    'prior_test_interpreter_failure': {
        'original_tool': 'b53592', 'exit_code': 1,
        'cause': 'System python3 lacked numpy before any tests; reran with existing work/venv/bin/python. No production code change.'},
    'old_checker_preserved_sha256': digest(FROZEN / 'check_goop3d_observed_history_summary_v1.py'),
    'scope': 'Source/metadata plus wholly synthetic scalar vectors and temporary cache files only. No original collection, cached result rows, NPZ, model or trajectory execution.',
    'conclusion': 'Separately versioned checker changes only sample-SD reduction to exact pairwise rational variance with high-precision square root. Estimator, seed values, means, null rules, contrasts and comparison tolerances are unchanged. Cache-only finisher preserves original failing attempt, consumes all exact 2568 immutable passing rows with no fallback audit, uses original merge/summary, rehashes original inputs opaquely, and writes distinct successor products.',
    'remaining': 'Root must execute on the exact original runtime after original owner closure; actual zero exit, complete outputs and independent product review precede numerical admission.'
}
output = HERE / 'observed_finalizer_v2_source_review.json'
output.write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps({'review': str(output), 'sha256': digest(output), 'source_checks': len(checks), 'status': review['status']}))
