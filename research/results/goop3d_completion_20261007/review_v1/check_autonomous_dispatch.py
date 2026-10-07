"""Independent AST equivalence of numerical dispatch; no scientific imports."""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
WORKER = BASE/'autonomous_v1/run_worker.py'
PREP = BASE.parent/'deadline_research_20261005/cuda_preparation'
OLD = PREP/'evaluate_goop3d_graph_support_v1.py'
checks = []


def check(ok, label):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calls(tree, text):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == text]


class Canonical(ast.NodeTransformer):
    replacements = {
        "manifest['metadata']": 'metadata',
        "manifests[cell['split']]['metadata']": 'metadata',
        "cell['policy']": 'policy',
        'HORIZON': '295',
        'args.seed': 'seed',
        "worker['seed']": 'seed',
        "item['source_index']": 'source_index',
        "cell['source_index']": 'source_index',
    }

    def visit(self, node):
        text = ast.unparse(node)
        if text in self.replacements:
            return ast.parse(self.replacements[text], mode='eval').body
        return super().visit(node)


old = ast.parse(OLD.read_text())
new = ast.parse(WORKER.read_text())
old_calls, new_calls = calls(old, 'native.rollout'), calls(new, 'native.rollout')
check(len(old_calls) == len(new_calls) == 1, 'one original and successor native rollout call')
check(ast.dump(Canonical().visit(old_calls[0])) == ast.dump(Canonical().visit(new_calls[0])),
      'exact native rollout arguments after variable/path rebinding')
check(len(calls(new, 'E.prepare_model')) == 1, 'one frozen model preparation call')
check(len(calls(new, 'T.configure_cuda')) == 1, 'one frozen CUDA configuration call')
check(len(calls(new, 'E.cohort_gate')) == 1, 'unchanged cohort validator invoked')
check(len(calls(new, 'E.runtime_gate')) == 1, 'physical GPU runtime validator invoked')
check(not calls(new, 'E.main') and not calls(new, 'T.main'), 'old execution CLI/deadline machinery not invoked')
check(not calls(new, 'train') and not calls(new, 'optimizer.step'), 'no training/optimizer dispatch')
check(len(calls(new, 'helpers.data_loader._manifest_array')) == 2,
      'selected position/type arrays use unchanged manifest loader')
check(len(calls(new, 'helpers.torch.no_grad')) == 1, 'rollout retains no_grad context')
check(len(calls(new, 'np.savez_compressed')) == 1, 'unchanged full numeric trace archive serialization')

plan = json.loads((BASE/'autonomous_v1/plan.json').read_text())
for relative, pin in plan['source_pins'].items():
    path = PREP/Path(relative).name if relative.startswith('cuda_preparation/') else Path('outputs/AdaptGNS')/relative
    check(sha(path) == pin, 'exact frozen numerical/source file '+relative)

native_path = PREP/'goop3d_native_evaluation_v1.py'
native = ast.parse(native_path.read_text())
function = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == 'rollout')
source = ast.get_source_segment(native_path.read_text(), function)
check('history = np.array(positions[:6], dtype=np.float32, copy=True)' in source,
      'each native rollout makes fresh float32 initial six-frame history')
check('rng = np.random.default_rng(rng_seed)' in source and 'cached_risk, warmup = None, None' in source,
      'per-cell RNG and cached-risk state are local')
check('for step in range(1, horizon + 1):' in source,
      'forecast steps remain sequential through full declared horizon')
check('MAX_PAIRS, MAX_EDGES, MAX_ABS = 2000000, 5000000, 10.' in native_path.read_text(),
      'all original D3 candidate/edge/state guards remain')

report = {'status': 'passed_source_dispatch_equivalence', 'checks': len(checks),
          'worker_sha256': sha(WORKER), 'original_evaluator_sha256': sha(OLD),
          'plan_sha256': sha(BASE/'autonomous_v1/plan.json'),
          'scientific_source_files_verified': len(plan['source_pins']),
          'checks_descriptions': checks, 'scientific_execution': False,
          'scope': 'Source arguments, unchanged numerical imports and cell-local state only; resume/storage and actual runtime reviewed separately.'}
out = BASE/'review_v1/autonomous_dispatch_review.json'
out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps({'status':report['status'], 'checks':len(checks), 'review_sha256':sha(out)}))
