"""Synthetic-only probe of frozen source reducers; no research output is read."""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / 'deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1'
from proposed_exact_seed_stats import seed_values_exact


def need(ok, message):
    if not ok:
        raise ValueError(message)


def extracted(name, functions):
    path = FROZEN / name
    tree = ast.parse(path.read_text())
    namespace = {'math': math, 'statistics': statistics, 'need': need}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


summarizer = extracted('summarize_goop3d_observed_histories_v1.py', {'finite', 'stats'})
checker = extracted('check_goop3d_observed_history_summary_v1.py', {'valid', 'average', 'seed_values'})
values = [[100000.1] * 3, [1651982.6] * 3, [100000000.1] * 3,
          [100000.1, 100000.10000000002, 100000.10000000004],
          [1., 2., 3.], [-1., 0., 1.], [0., 0., 0.],
          [None, 1., 2.], [1e-250, 2e-250, 3e-250], [1e155, 2e155, 3e155]]
results = []
for xs in values:
    summary = summarizer['stats'](xs)
    try:
        before = checker['seed_values'](xs)
        old_error = None
    except Exception as error:
        before, old_error = None, {'type': type(error).__name__, 'message': str(error)}
    proposed = seed_values_exact(xs)
    need(summary.keys() == proposed.keys(), 'schema unchanged')
    for key in summary:
        if key == 'sample_sd' and summary[key] is not None:
            need(math.isclose(summary[key], proposed[key], rel_tol=1e-10, abs_tol=1e-12), 'same existing tolerance and estimator')
        else:
            need(summary[key] == proposed[key], 'unchanged seed values/counts/mean/nulls')
    old_matches = None if before is None else (before['sample_sd'] is None and summary['sample_sd'] is None or
                  before['sample_sd'] is not None and summary['sample_sd'] is not None and
                  math.isclose(summary['sample_sd'], before['sample_sd'], rel_tol=1e-10, abs_tol=1e-12))
    results.append({'seed_values': xs, 'summarizer': summary, 'frozen_checker': before, 'frozen_checker_error': old_error,
                    'frozen_checker_matches': old_matches, 'proposed': proposed, 'proposed_matches_existing_tolerance': True})
report = {'status': 'source_defect_reproduced_synthetic_only', 'real_failed_leaf_identified': False,
          'tolerance_changed': False, 'variance_formula': 'sum_{i<j}(xi-xj)^2 / (3 * 2)',
          'source_files': {n: hashlib.sha256((FROZEN/n).read_bytes()).hexdigest() for n in (
              'summarize_goop3d_observed_histories_v1.py','check_goop3d_observed_history_summary_v1.py')},
          'cases': results}
path = HERE / 'synthetic_source_probe.json'
path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': report['status'], 'synthetic_cases': len(values),
                  'frozen_mismatches': sum(x['frozen_checker_matches'] is False for x in results),
                  'frozen_exceptions': sum(x['frozen_checker_error'] is not None for x in results),
                  'proposal_matches': len(results), 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
