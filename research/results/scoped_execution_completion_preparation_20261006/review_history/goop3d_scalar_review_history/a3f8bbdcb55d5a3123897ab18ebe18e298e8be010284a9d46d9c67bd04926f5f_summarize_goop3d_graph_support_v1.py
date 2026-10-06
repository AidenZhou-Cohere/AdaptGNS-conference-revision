#!/usr/bin/env python3
"""Strict scalar collection and paired analysis of a separately frozen D3 cohort.

Description only unless --execute. No arrays are deserialized and no research
process is launched. collect hashes stopped output bytes; summarize consumes one
root-bound scalar collection. See goop3d_scalar_final_ledger_interface_v1.md.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop3d_paired_scalar_summary_v1'
COLLECTION_SCHEMA = 'adaptgns_goop3d_stopped_scalar_collection_v1'
LEDGER_SCHEMA = 'adaptgns_goop3d_final_evaluation_ledger_v1'
SOURCE_PINS = {
    'evaluate_goop3d_graph_support_v1.py': '9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de',
    'goop3d_diagnostic_metrics_v1.py': '5ef3de96e470eea495dd56c1f60396bbd166b9ea5c5bc340864715cb9e2c173b',
    'goop3d_native_evaluation_v1.py': 'a742123093aff433f3a4e302929a52bae4f5fee9610195d86df726852575b1d5',
    'train_goop3d_graph_support_cuda_v2.py': '8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc',
    'goop3d_graph_support_vectorized_v1.py': 'ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50',
}
VALID_SHA = 'f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef'
HORIZON, DIMENSION = 295, 3
POLICIES = ('base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25')
DIAG_POLICIES = ('base', 'dense', 'random25', 'speed25', 'relative-velocity-RMS25', 'previous-observed-base-risk25')
TIMING_CASES = DIAG_POLICIES + ('natural_base_reference',)
CORRELATIONS = (('previous_risk_vs_base_error',)
                + tuple('previous_risk_vs_' + p + '_benefit' for p in DIAG_POLICIES[1:])
                + tuple('dense_vs_' + p + '_benefit' for p in DIAG_POLICIES[2:]))
BENEFIT_KEYS = ('mean_position_vector_benefit', 'mean_normalized_vector_benefit',
                'positive_fraction', 'negative_fraction', 'zero_fraction')
DENSE_FRACTIONS = ('dense_sparse_sign_disagreement_fraction', 'dense_positive_sparse_nonpositive_fraction')
MODELS = tuple((a, s) for a in ('base', 'mix') for s in range(3))
STAGES = (('full_rollout_valid', 'full-rollout', 'valid'), ('full_rollout_test', 'full-rollout', 'test'),
          ('same_state_valid', 'same-state', 'valid'), ('same_state_test', 'same-state', 'test'),
          ('clean_validation', 'clean-validation', 'valid'))
CELL_STATES = {'completed_required_outcome', 'recorded_failed_outcome', 'timed_out_current',
               'not_completed_before_invocation_end', 'never_started'}
ROW_STATES = {'completed_required_outcome', 'recorded_failed_outcome'}
B_KEYS = ('fraction_particles_outside', 'fraction_particles_outside_by_more_than_1e-6',
          'maximum_coordinate_excursion', 'mean_particle_maximum_excursion')
G_KEYS = ('candidate_pairs', 'geometric_base_pairs', 'available_annulus_pairs', 'optional_pair_budget',
          'retained_optional_pairs', 'directed_edges', 'native_base_directed_edges', 'native_base_self_edges',
          'native_base_receivers_above_cap_before_capping', 'native_base_edges_removed_by_cap',
          'native_base_max_receiver_degree', 'native_base_asymmetric_directed_edges')
T_KEYS = ('end_to_end_seconds', 'score_generation_seconds', 'score_graph_seconds', 'score_forward_seconds',
          'current_graph_and_selection_seconds', 'current_forward_seconds')
FULL_METRICS = ('mean_rollout_mse', 'mse_forecast200', 'mse_forecast295') + tuple(
    side + '_boundary_' + key for side in ('predicted', 'ground_truth')
    for key in (*('mean_' + k for k in B_KEYS), 'trajectory_maximum_excursion')) + tuple(
    'graph_mean_' + k for k in G_KEYS) + ('graph_fraction_steps_cap_active',)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_snapshot(path, expected_sha=None):
    data = Path(path).read_bytes(); value = hashlib.sha256(data).hexdigest()
    require(expected_sha is None or value == expected_sha, 'Parsed JSON bytes differ from approved snapshot: ' + str(path))
    return json.loads(data), value


def tree_snapshot(directory):
    directory = Path(directory)
    require(not directory.is_symlink(), 'Stage root cannot be a symlink')
    if not directory.exists(): return {'exists': False, 'entries': [], 'files': {}}
    require(directory.is_dir(), 'Stage root must be a directory')
    entries, files = [], {}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Unsupported/symlink output snapshot entry')
        entries.append([str(path.relative_to(directory)), 'file' if path.is_file() else 'directory'])
        if path.is_file(): files[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return {'exists': True, 'entries': entries, 'files': files}


def verify_collection_snapshot(collection):
    require(sha(__file__) == collection['collector_sha256']
            and all(sha(HERE / name) == pin for name, pin in SOURCE_PINS.items()), 'Collector or frozen source changed')
    require(all(sha(path) == value for path, value in collection['input_sha256'].items()), 'Original collection input changed')
    files = {}
    for directory, layout in collection['output_tree_state'].items():
        current = tree_snapshot(directory)
        require({k: current[k] for k in ('exists', 'entries')} == layout, 'Stopped output tree entry set changed')
        files.update(current['files'])
    require(files == collection['files_sha256'], 'Stopped output bytes changed after scalar validation')


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def finite(value, nonnegative=False):
    return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-14)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def mean_complete(values):
    return statistics.fmean(values) if values and all(finite(v) for v in values) else None


def contrast(a, b):
    return a - b if finite(a) and finite(b) else None


def seed_summary(values):
    require(len(values) == 3, 'Exactly three ordered seed pairs required')
    complete = all(finite(v) for v in values)
    return {'seed_values': {str(s): v for s, v in enumerate(values)}, 'required_seed_pairs': 3,
            'defined_seed_pairs': sum(finite(v) for v in values),
            'mean': statistics.fmean(values) if complete else None,
            'sample_sd': statistics.stdev(values) if complete else None}


def load_scalar(name):
    path = HERE / name
    require(sha(path) == SOURCE_PINS[name], 'Frozen D3 scalar helper changed: ' + name)
    spec = importlib.util.spec_from_file_location('_d3_summary_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def expected(manifest, mode):
    require(manifest.get('record_count') == len(manifest.get('records', [])), 'Complete manifest count required')
    return load_scalar('evaluate_goop3d_graph_support_v1.py').schedules(manifest['records'], mode, 'final_evaluation')


def cell_schedule(manifest, mode):
    return [{**item, **({'policy': p} if p else {})}
            for item in expected(manifest, mode) for p in (POLICIES if mode == 'full-rollout' else (None,))]


def cell_key(row, mode):
    return row['source_index'], row['policy' if mode == 'full-rollout' else 'target_frame']


def filename(row, mode):
    suffix = row['policy'] if mode == 'full-rollout' else f'target_{row["target_frame"]:03d}'
    return f'trajectory_{row["source_index"]:06d}_{suffix}.json'


def boundary_record(row):
    require(isinstance(row, dict) and all(finite(row.get(k), True) for k in B_KEYS), 'Invalid boundary scalar')
    require(0 <= row[B_KEYS[1]] <= row[B_KEYS[0]] <= 1
            and row[B_KEYS[3]] <= row[B_KEYS[2]] + 1e-14, 'Boundary fractions/excursions disagree')
    require(all(isinstance(row.get(k), list) and len(row[k]) == DIMENSION and all(finite(v) for v in row[k])
                for k in ('coordinate_minimum', 'coordinate_maximum'))
            and all(a <= b for a, b in zip(row['coordinate_minimum'], row['coordinate_maximum'])), 'D3 boundary coordinate range required')
    return {k: row[k] for k in B_KEYS}


def boundary_series(series, completed):
    require(isinstance(series, list) and len(series) == completed, 'Boundary prefix length differs')
    for row in series: boundary_record(row)
    return {**{'mean_' + k: mean_complete([r[k] for r in series]) for k in B_KEYS},
            'trajectory_maximum_excursion': max((r[B_KEYS[2]] for r in series), default=None)}


def graph_record(row, policy, particles):
    require(all(type(row.get(k)) is int and row[k] >= 0 for k in G_KEYS), 'Nonnegative graph counts required')
    base, extra, native = row['geometric_base_pairs'], row['available_annulus_pairs'], row['native_base_directed_edges']
    budget = 0 if policy == 'base' else extra if policy == 'dense' else extra // 4
    require(row['candidate_pairs'] == base + extra and row['optional_pair_budget'] == extra // 4
            and row['retained_optional_pairs'] == budget and row['directed_edges'] == native + 2 * budget,
            'Exact optional graph budget differs')
    require(row['native_base_prefix_preserved'] is True and row['native_base_max_receiver_degree'] <= 128
            and row['native_base_self_edges'] <= particles and native <= 128 * particles
            and row['native_base_receivers_above_cap_before_capping'] <= particles
            and row['native_base_edges_removed_by_cap'] == 2 * base + particles - native
            and row['native_base_asymmetric_directed_edges'] <= native - row['native_base_self_edges'],
            'Native prefix/cap/count invariants differ')
    require(all(digest(row.get(k)) for k in ('native_base_sha256', 'selected_optional_pair_sha256', 'directed_edge_sha256')),
            'Exact graph hashes required')
    return {k: row[k] for k in G_KEYS}


def validate_rollout(row, arm, seed, item):
    require(row.get('arm') == arm and row.get('training_seed') == seed and type(row.get('training_seed')) is int
            and row.get('objective') == 'faithful' and row.get('horizon') == HORIZON
            and row.get('policy') in POLICIES and all(row.get(k) == v for k, v in item.items()), 'D3 rollout identity differs')
    n = row.get('completed_steps'); mse = row.get('mse_per_step'); complete = row.get('status') == 'complete'
    require(type(n) is int and 0 <= n <= HORIZON and row.get('status') in ('complete', 'failed')
            and isinstance(mse, list) and len(mse) == n and all(finite(v, True) for v in mse), 'Invalid accepted D3 MSE prefix')
    require(complete == (n == HORIZON) and (row.get('failure') is None if complete else isinstance(row.get('failure'), dict)), 'Inconsistent D3 status')
    require((close(row.get('mean_rollout_mse'), statistics.fmean(mse)) and close(row.get('mse_at_final_horizon'), mse[-1]))
            if complete else row.get('mean_rollout_mse') is None and row.get('mse_at_final_horizon') is None
            and isinstance(row['failure'].get('category'), str), 'Full-H means must reflect complete outcomes')
    for step in (1, 10, 50, 200, HORIZON):
        value = row.get('mse_at_declared_trace_steps', {}).get(str(step))
        require(close(value, mse[step - 1]) if n >= step else value is None, 'Pointwise trace error differs')
    require(finite(row.get('synchronized_call_seconds'), True) and row['synchronized_call_seconds'] > 0
            and finite(row.get('publication_seconds'), True), 'Measured row call/publication times required')
    metrics = {'mean_rollout_mse': statistics.fmean(mse) if complete else None,
               'mse_forecast200': mse[199] if n >= 200 else None, 'mse_forecast295': mse[-1] if complete else None}
    prefixes = {}
    for side in ('predicted', 'ground_truth'):
        values = boundary_series(row.get(side + '_boundary_per_step'), n)
        metrics.update({side + '_boundary_' + k: v if complete else None for k, v in values.items()})
        if not complete: prefixes.update({side + '_boundary_' + k: v for k, v in values.items()})
    attempts = row.get('attempts')
    require(isinstance(attempts, list) and len(attempts) in (n, n + 1)
            and [a.get('forecast_step') for a in attempts] == list(range(1, len(attempts) + 1))
            and [a.get('accepted') for a in attempts] == [True] * n + [False] * (len(attempts) - n), 'Accepted/rejected attempt ledger differs')
    for key, attempt_key in (('directed_edges_per_step', 'directed_edges'), ('retained_optional_pairs_per_step', 'retained_optional_pairs'),
                             ('candidate_pairs_per_step', 'candidate_pairs'), ('base_pairs_per_step', 'geometric_base_pairs')):
        require(row.get(key) == [a.get(attempt_key) for a in attempts[:n]], 'Step graph ledger differs: ' + key)
    risks = row.get('mean_normalized_acceleration_variance_per_step')
    require(isinstance(risks, list) and len(risks) == n and all(finite(v) and v > 0 for v in risks), 'Accepted variance prefix differs')
    graphs = []
    for i, attempt in enumerate(attempts[:n]):
        graphs.append(graph_record(attempt, row['policy'], item['particles']))
        require(close(attempt.get('coordinate_mse'), mse[i]) and attempt.get('predicted_boundary') == row['predicted_boundary_per_step'][i]
                and attempt.get('ground_truth_boundary') == row['ground_truth_boundary_per_step'][i], 'Accepted attempt scalar mismatch')
    for k in G_KEYS:
        value = mean_complete([g[k] for g in graphs]); metrics['graph_mean_' + k] = value if complete else None
        if not complete: prefixes['graph_mean_' + k] = value
    metrics['graph_fraction_steps_cap_active'] = mean_complete([int(g['native_base_edges_removed_by_cap'] > 0) for g in graphs]) if complete else None
    require(close(row.get('mean_directed_edges'), mean_complete(row['directed_edges_per_step'])) if complete
            else row.get('mean_directed_edges') is None, 'Mean edge arithmetic differs')
    return metrics, prefixes


def timing_stats(values):
    complete = len(values) == 7 and all(finite(v, True) for v in values)
    return {'observations': len(values), 'expected': 7, 'mean': statistics.fmean(values) if complete else None,
            'median': statistics.median(values) if complete else None, 'sample_sd': statistics.stdev(values) if complete else None}


def validate_diagnostic(row, arm, seed, item, split, mode, particles):
    require(row.get('arm') == arm and row.get('training_seed') == seed and type(row.get('training_seed')) is int and row.get('objective') == 'faithful'
            and all(row.get(k) == v for k, v in item.items()) and row.get('status') in ('complete', 'failed'), 'D3 diagnostic identity differs')
    require(row.get('failure') is None if row['status'] == 'complete' else isinstance(row.get('failure'), dict), 'Diagnostic status/failure differs')
    require(finite(row.get('synchronized_call_seconds'), True) and row['synchronized_call_seconds'] > 0
            and finite(row.get('publication_seconds'), True), 'Diagnostic row clocks required')
    if mode == 'clean-validation':
        if row['status'] == 'complete':
            metric = row.get('metrics', {})
            require(all(finite(metric.get(k)) for k in ('normalized_acceleration_coordinate_mse', 'realized_normalized_vector_se',
                        'predicted_normalized_vector_se', 'constant_free_gaussian_nll'))
                    and metric['normalized_acceleration_coordinate_mse'] >= 0 and metric['predicted_normalized_vector_se'] > 0
                    and close(3 * metric['normalized_acceleration_coordinate_mse'], metric['realized_normalized_vector_se']), 'D3 coordinate/vector clean metrics differ')
            graph_record(row['graph'], 'base', particles)
        else:
            require(row.get('metrics') is None, 'Failed clean metrics must remain null')
        return
    diagnostic = load_scalar('goop3d_diagnostic_metrics_v1.py')
    require(row.get('random_seed_material') == [20261006, 93000, seed, 0 if split == 'valid' else 1, item['source_index'], item['target_frame']]
            and row.get('risk_source') == 'previous observed native-base history', 'Observed-state risk/random material differs')
    base_graph = None
    for group, rounds in (('warmup_calls', (-1,)), ('timed_calls', range(7))):
        calls = row.get(group); require(isinstance(calls, list), 'Diagnostic call ledger required')
        order = [(r, slot, method) for r in rounds for slot, method in enumerate(diagnostic.timing_order(item['source_index'], item['schedule_index'], max(r, 0)))]
        require(len(calls) <= len(order) and [(c.get('round'), c.get('slot'), c.get('method')) for c in calls] == order[:len(calls)], 'Frozen repeated-call ordering differs')
        if row['status'] == 'complete': require(len(calls) == len(order), 'Complete diagnostic omitted calls')
        for call in calls:
            require(call.get('status') in ('complete', 'failed'), 'Call status required')
            if call['status'] != 'complete':
                require(isinstance(call.get('failure'), dict), 'Failed call reason required'); continue
            require(call.get('failure') is None and digest(call.get('prediction_sha256'))
                    and all(finite(call.get('timing', {}).get(k), True) for k in T_KEYS), 'Complete measured call required')
            method = call['method']; timing = call['timing']; graph = call['graph']
            require(timing.get('network_passes') == (2 if method == DIAG_POLICIES[-1] else 1), 'Scoring pass count differs')
            if method == 'natural_base_reference':
                require(graph.get('retained_optional_pairs') == 0 and graph.get('native_base_prefix_preserved') is True
                        and digest(graph.get('directed_edge_sha256')), 'Natural base graph differs')
            else:
                graph_record(graph, 'laggedrisk25' if method == DIAG_POLICIES[-1] else method, particles)
                shared = {k: graph[k] for k in ('candidate_pairs', 'geometric_base_pairs', 'available_annulus_pairs',
                          'native_base_directed_edges', 'native_base_sha256', 'optional_pair_budget')}
                require(base_graph is None or base_graph == shared, 'Same-state mandatory graph/candidates differ')
                base_graph = shared
    records = row.get('policies', {})
    require(set(records) <= set(TIMING_CASES), 'Unexpected diagnostic policy')
    if row['status'] == 'complete':
        require(set(records) == set(TIMING_CASES) and all(r.get('status') == 'complete' for r in records.values())
                and all(c.get('status') == 'complete' for c in row['warmup_calls']), 'Complete diagnostic policy grid missing/failed')
        require(all(row.get('native_parity', {}).get(k, {}).get('passed') is True for k in ('current', 'previous'))
                and row.get('natural_shared_base', {}).get('ordered_edges_exact') is True
                and row['natural_shared_base'].get('outputs_within_fixed_parity_tolerance') is True, 'Complete native/shared parity required')
    for method, record in records.items():
        calls = [c for c in row['timed_calls'] if c['method'] == method]
        good = [c for c in calls if c['status'] == 'complete']
        require(record.get('expected_repetitions') == 7 and record.get('completed_repetitions') == len(good)
                and record.get('status') in ('complete', 'failed'), 'Policy repetition status differs')
        complete = record['status'] == 'complete'
        require(complete == (len(good) == len(calls) == 7 and record.get('repeat_consistent') is True), 'Policy complete/repeat inconsistency')
        if complete:
            for call in calls[1:]:
                keys = ('ordered_edges_exact', 'outputs_within_fixed_parity_tolerance')
                if method == DIAG_POLICIES[-1]: keys += ('previous_score_ordered_edges_exact', 'previous_score_outputs_within_fixed_parity_tolerance')
                if method == 'relative-velocity-RMS25': keys += ('physical_scores_exact',)
                require(all(call.get('repeat_consistency', {}).get(k) is True for k in keys), 'Complete policy repeat ledger differs')
        for k in T_KEYS:
            wanted = timing_stats([c['timing'][k] for c in good]); saved = record.get('timing', {}).get(k, {})
            require(set(saved) == set(wanted) and all(close(saved[v], x) if finite(x) else saved[v] is None
                    for v, x in wanted.items()), 'Repeated timing arithmetic differs')
        if complete:
            metric = record.get('metrics', {}); require(all(finite(metric.get(k), True) for k in ('position_coordinate_mse', 'normalized_coordinate_mse')), 'Complete diagnostic accuracy missing')
            boundary_record(record['prediction_boundary'])
            require(record.get('graph') == good[0].get('graph'), 'Policy graph differs from first successful measured call')
            if method != 'natural_base_reference': graph_record(record['graph'], method, particles)
        else:
            require(all(record.get(k) is None for k in ('metrics', 'graph', 'prediction_boundary')), 'Failed policy accuracy/graph/boundary must remain null')
    benefit_map = row.get('benefit', {}); correlations = row.get('correlations', {})
    require(isinstance(benefit_map, dict) and isinstance(correlations, dict)
            and set(correlations) <= set(CORRELATIONS), 'Diagnostic benefit/correlation grid differs')
    if row['status'] == 'complete':
        require(set(benefit_map) == set(DIAG_POLICIES[1:]) and set(correlations) == set(CORRELATIONS)
                and 'truth_boundary' in row, 'Complete benefit/correlation/truth grid required')
    if 'truth_boundary' in row: boundary_record(row['truth_boundary'])
    for method, benefit in benefit_map.items():
        require(method in DIAG_POLICIES[1:] and all(finite(benefit.get(k)) for k in ('mean_position_vector_benefit', 'mean_normalized_vector_benefit')), 'Signed benefit identity/value differs')
        dense_ready = method != 'dense' and records.get('dense', {}).get('status') == 'complete'
        require(set(benefit) == set(BENEFIT_KEYS + (DENSE_FRACTIONS if dense_ready else ())), 'Signed benefit scalar grid differs')
        if dense_ready:
            require(all(finite(benefit[k]) and 0 <= benefit[k] <= 1 for k in DENSE_FRACTIONS)
                    and benefit[DENSE_FRACTIONS[1]] <= benefit[DENSE_FRACTIONS[0]], 'Dense/sparse sign fractions differ')
        for k in ('positive_fraction', 'negative_fraction', 'zero_fraction'):
            require(finite(benefit.get(k)) and 0 <= benefit[k] <= 1, 'Benefit sign fraction differs')
        require(close(sum(benefit[k] for k in ('positive_fraction', 'negative_fraction', 'zero_fraction')), 1), 'Benefit signs must partition particles')
        for space in ('position', 'normalized'):
            a, b = records.get('base', {}).get('metrics'), records.get(method, {}).get('metrics')
            require(a is not None and b is not None and close(benefit['mean_' + space + '_vector_benefit'],
                    3 * (a[space + '_coordinate_mse'] - b[space + '_coordinate_mse'])), 'Vector benefit disagrees with policy MSE')
    for name, value in correlations.items():
        prerequisites = ('base', DIAG_POLICIES[-1]) if name == 'previous_risk_vs_base_error' else (
            ('base', DIAG_POLICIES[-1], name[len('previous_risk_vs_'):-len('_benefit')])
            if name.startswith('previous_risk_vs_') else ('base', 'dense', name[len('dense_vs_'):-len('_benefit')]))
        require(all(records.get(p, {}).get('status') == 'complete' for p in prerequisites), 'Correlation requires complete contributing policies')
        require(isinstance(value, dict) and (finite(value.get('value')) and -1 <= value['value'] <= 1 and value.get('reason') is None
                or value.get('value') is None and isinstance(value.get('reason'), str)), 'Undefined correlation reason/value differs')


def validate_outcome(outcome, command):
    require(isinstance(outcome, dict) and type(outcome.get('started')) is bool
            and isinstance(outcome.get('termination_reason'), str) and outcome['termination_reason'], 'Explicit process/missing-work outcome required')
    if not outcome['started']:
        require(outcome.get('state') == 'never_started' and outcome.get('pid') is None, 'Never-started process identity differs'); return
    require(outcome.get('stopped_and_reaped') is True and type(outcome.get('pid')) is int and outcome['pid'] > 0
            and type(outcome.get('start_ticks')) is int and outcome['start_ticks'] > 0 and outcome.get('command') == command
            and isinstance(outcome.get('hostname'), str) and outcome['hostname'] and type(outcome.get('exit_code')) is int
            and finite(outcome.get('elapsed_seconds'), True) and isinstance(outcome.get('signals'), list)
            and finite(outcome.get('outer_timeout_seconds'), True) and outcome['outer_timeout_seconds'] > 0
            and isinstance(outcome.get('absolute_stop_utc'), str), 'Stopped whole-invocation process identity/clocks required')


def flags(command):
    require(isinstance(command, list) and len(command) >= 2 and all(isinstance(v, str) for v in command), 'Exact command argv required')
    result = {}; i = 0
    while i < len(command):
        word = command[i]
        if word.startswith('--'):
            require(word not in result, 'Duplicate evaluator option')
            if word == '--execute': result[word] = True; i += 1; continue
            require(i + 1 < len(command) and not command[i + 1].startswith('--'), 'Missing evaluator option value')
            result[word] = command[i + 1]; i += 2
        else: i += 1
    return result


def collect(ledger_path, cohort_path, cohort_audit_path, valid_path, test_path, protocol_path, root_release):
    collector_sha = sha(__file__)
    root_release = json.loads(json.dumps(root_release, allow_nan=False))
    paths = [Path(x).resolve() for x in (ledger_path, cohort_path, cohort_audit_path, valid_path, test_path, protocol_path)]
    snapshots = [json_snapshot(p) for p in paths[:5]]
    inputs = {str(p): h for p, (_, h) in zip(paths[:5], snapshots)}
    inputs[str(paths[5])] = sha(paths[5])
    require(root_release.get('schema') == 'adaptgns_goop3d_scalar_collection_release_v1'
            and root_release.get('issued_by') == 'root' and root_release.get('status') == 'approved_stopped_scalar_collection'
            and root_release.get('collector_sha256') == collector_sha
            and root_release.get('files_sha256') == inputs, 'Exact root stopped-collection release required')
    for name, pin in SOURCE_PINS.items(): require(sha(HERE / name) == pin, 'Frozen D3 source changed')
    ledger, cohort, audit, valid, test = [v for v, _ in snapshots]; manifests = {'valid': valid, 'test': test}
    require(inputs[str(paths[3])] == VALID_SHA, 'Original complete D3 validation manifest required')
    require(test.get('metadata') == valid.get('metadata') and isinstance(valid.get('metadata'), dict), 'Exact common D3 physical metadata required')
    require(ledger.get('schema') == LEDGER_SCHEMA and ledger.get('producer') == 'supervisor_candidate'
            and ledger.get('dataset') == 'Goop-3D' and ledger.get('state') == 'stopped_all_owned_processes_reaped'
            and ledger.get('unreaped_owned_children') == [] and ledger.get('all_pinned_inputs_reverified') is True
            and ledger.get('cohort_sha256') == inputs[str(paths[1])] and ledger.get('cohort_audit_sha256') == inputs[str(paths[2])]
            and ledger.get('protocol_sha256') == inputs[str(paths[5])]
            and ledger.get('source_manifest_sha256') == {'valid': inputs[str(paths[3])], 'test': inputs[str(paths[4])]}, 'Stopped D3 final ledger/input lineage required')
    evaluator = load_scalar('evaluate_goop3d_graph_support_v1.py'); diagnostic = load_scalar('goop3d_diagnostic_metrics_v1.py')
    first = next(m for m in cohort['models'] if (m['arm'], m['seed']) == ('base', 0))
    args = SimpleNamespace(protocol=Path(protocol_path), trainer_source=HERE / 'train_goop3d_graph_support_cuda_v2.py',
                           cohort_audit=Path(cohort_audit_path), arm='base', seed=0, checkpoint_sha256=first['checkpoint_sha256'])
    endpoint = evaluator.cohort_gate(cohort, audit, {'scientific_endpoint_updates': cohort['endpoint_updates']}, args)
    require(ledger.get('endpoint_updates') == endpoint, 'Ledger prospective endpoint differs')
    require(all(type(m.get('seed')) is int for m in cohort['models'] + audit['models'] + audit['paired_seeds'])
            and all(type(s.get('seed')) is int for s in ledger.get('stages', [])), 'Exact integer seed identities required')
    stages = ledger.get('stages', []); grid = {(a, s, name) for a, s in MODELS for name, _, _ in STAGES}
    require(len(stages) == len(grid) and {(s.get('arm'), s.get('seed'), s.get('stage')) for s in stages} == grid, 'All30 D3 model-stage receipts required')
    collected = []; directories = set(); files = {}; trees = {}
    for arm, seed in MODELS:
        model = next(m for m in cohort['models'] if (m['arm'], m['seed']) == (arm, seed))
        for name, mode, split in STAGES:
            entry = next(s for s in stages if (s['arm'], s['seed'], s['stage']) == (arm, seed, name))
            require(entry.get('mode') == mode and entry.get('split') == split and entry.get('checkpoint_sha256') == model['checkpoint_sha256']
                    and entry.get('endpoint_updates') == endpoint, 'Stage/model split or endpoint differs')
            directory = Path(entry['directory']); require(directory.is_absolute() and str(directory.resolve()) == str(directory)
                    and all(directory != p and directory not in p.parents and p not in directory.parents for p in directories), 'Unique disjoint original absolute stage directory required'); directories.add(directory)
            initial_tree = tree_snapshot(directory)
            trees[str(directory)] = {k: initial_tree[k] for k in ('exists', 'entries')}
            files.update(initial_tree['files'])
            def read_output(path):
                require(str(path) in initial_tree['files'], 'Output JSON was absent from initial stopped tree')
                return json_snapshot(path, initial_tree['files'][str(path)]['sha256'])[0]
            command = entry['command']; options = flags(command); validate_outcome(entry['outcome'], command)
            source_args = [Path(v) for v in command if Path(v).name == 'evaluate_goop3d_graph_support_v1.py']
            require(len(source_args) == 1 and source_args[0].is_absolute()
                    and sha(source_args[0]) == SOURCE_PINS['evaluate_goop3d_graph_support_v1.py'], 'Executed evaluator source identity differs')
            inputs[str(source_args[0])] = SOURCE_PINS['evaluate_goop3d_graph_support_v1.py']
            require(options.get('--execute') is True and options.get('--purpose') == 'final_evaluation'
                    and options.get('--mode') == mode and options.get('--split') == split and options.get('--arm') == arm
                    and options.get('--seed') == str(seed) and options.get('--checkpoint-updates') == str(endpoint)
                    and options.get('--checkpoint-sha256') == model['checkpoint_sha256'] and options.get('--output-dir') == str(directory), 'Final evaluator command differs')
            release_path = Path(entry['release_file']); require(release_path.is_absolute(), 'Original absolute child release required')
            release, release_sha = json_snapshot(release_path, entry['release_sha256'])
            require(options.get('--release') == str(release_path)
                    and all(options.get(key) == str(path.resolve()) for key, path in (
                        ('--cohort', Path(cohort_path)), ('--cohort-audit', Path(cohort_audit_path)),
                        ('--protocol', Path(protocol_path)), ('--manifest', Path(valid_path if split == 'valid' else test_path)))),
                    'Original absolute execution input/release paths differ')
            inputs[str(release_path)] = release_sha
            require(release.get('schema') == 'adaptgns_goop3d_evaluation_release_v1' and release.get('issued_by') == 'root'
                    and release.get('status') == 'admitted_for_execution' and release.get('purpose') == 'final_evaluation'
                    and release.get('mode') == mode and release.get('split') == split and release.get('arm') == arm
                    and release.get('seed') == seed and type(release.get('seed')) is int and release.get('checkpoint_updates') == endpoint
                    and release.get('scientific_endpoint_updates') == endpoint
                    and release.get('checkpoint_sha256') == model['checkpoint_sha256']
                    and release.get('evaluator_sha256') == SOURCE_PINS['evaluate_goop3d_graph_support_v1.py']
                    and release.get('graph_sha256') == SOURCE_PINS['goop3d_graph_support_vectorized_v1.py'], 'D3 final child release differs')
            required = {'--cohort': inputs[str(paths[1])], '--cohort-audit': inputs[str(paths[2])], '--protocol': inputs[str(paths[5])],
                        '--manifest': inputs[str(paths[3 if split == 'valid' else 4])], '--trainer-source': SOURCE_PINS['train_goop3d_graph_support_cuda_v2.py'],
                        '--checkpoint': model['checkpoint_sha256']}
            require(all(release.get('files_sha256', {}).get(options.get(k)) == v for k, v in required.items()), 'Released source/model/manifest bindings differ')
            wanted = cell_schedule(manifests[split], mode); cells = entry.get('cells', [])
            require(len(cells) == len(wanted) and all(all(c.get(k) == v for k, v in w.items())
                    and c.get('state') in CELL_STATES for c, w in zip(cells, wanted)), 'Exact expected cell schedule/state required')
            protocol_file = directory / 'protocol.json'
            protocol = read_output(protocol_file) if str(protocol_file) in initial_tree['files'] else None
            protocol_sha = initial_tree['files'].get(str(protocol_file), {}).get('sha256')
            if not entry['outcome']['started']:
                require(protocol is None and all(c['state'] == 'never_started' for c in cells), 'Never-started stage contains execution evidence')
            if protocol is not None:
                require(protocol.get('schema') == evaluator.SCHEMA and protocol.get('purpose') == 'final_evaluation'
                        and protocol.get('mode') == mode and protocol.get('split') == split and protocol.get('arm') == arm
                        and protocol.get('seed') == seed and type(protocol.get('seed')) is int and protocol.get('checkpoint_updates') == endpoint
                        and protocol.get('checkpoint_sha256') == model['checkpoint_sha256'] and protocol.get('frames') == 301
                        and protocol.get('horizon') == HORIZON and protocol.get('policies') == list(POLICIES)
                        and protocol.get('schedule') == expected(manifests[split], mode)
                        and protocol.get('source_population_count') == manifests[split]['record_count']
                        and protocol.get('prospective_source_grid') == evaluator.grid_indices(manifests[split]['record_count'])
                        and protocol.get('selected_source_indices') == sorted({r['source_index'] for r in expected(manifests[split], mode)})
                        and protocol.get('release_sha256') == release_sha
                        and all(protocol.get('input_files_sha256', {}).get(options[k]) == v for k, v in required.items()), 'Saved D3 protocol differs')
            rows = []; row_names = set()
            for cell, item in zip(cells, wanted):
                path = directory / filename(item, mode)
                if cell['state'] not in ROW_STATES:
                    require(not path.exists() and isinstance(cell.get('reason'), str) and cell['reason'], 'Missing cell requires reason and no committed row')
                    continue
                require(entry['outcome']['started'] and protocol is not None and cell.get('row_file') == path.name
                        and path.is_file() and not path.is_symlink()
                        and initial_tree['files'].get(str(path), {}).get('sha256') == cell.get('row_sha256'), 'Committed row identity/bytes differ')
                row = read_output(path); row_names.add(path.name)
                require(row.get('status') == ('complete' if cell['state'] == 'completed_required_outcome' else 'failed')
                        and row.get('protocol_sha256') == protocol_sha
                        and cell.get('failure') == row.get('failure'), 'Row status/protocol/failure differs')
                artifact = directory / row['artifact_file']
                require(artifact.parent == directory and artifact.name == path.with_suffix('.npz').name and artifact.is_file()
                        and not artifact.is_symlink() and initial_tree['files'].get(str(artifact), {}).get('sha256') == row.get('artifact_sha256') == cell.get('artifact_sha256'), 'Numeric artifact bytes differ')
                if mode == 'full-rollout': validate_rollout(row, arm, seed, item)
                else: validate_diagnostic(row, arm, seed, item, split, mode, manifests[split]['records'][item['source_index']]['positions']['shape'][1])
                rows.append(row)
            if entry['outcome']['started']:
                require(entry['outcome']['elapsed_seconds'] + 1e-6 >= sum(r['synchronized_call_seconds'] + r['publication_seconds'] for r in rows), 'Whole-child duration is shorter than committed work')
            require({Path(p).name for p in initial_tree['files'] if Path(p).parent == directory and Path(p).match('trajectory_*.json')} == row_names, 'Unaccounted committed output rows')
            diag = diagnostic.summarize(expected(manifests[split], mode), rows, mode) if mode != 'full-rollout' else None
            snapshot_path = directory / ('summary.json' if diag is not None else 'result.json')
            snapshot = {'state': 'absent', 'recorded_rows': 0, 'committed_rows': len(rows)}
            if str(snapshot_path) in initial_tree['files']:
                saved = read_output(snapshot_path)
                count = saved.get('returned_frames') if diag is not None else len(saved.get('rows', []))
                require(type(count) is int and 0 <= count <= len(rows), 'Aggregate snapshot row count differs')
                require(saved == diagnostic.summarize(expected(manifests[split], mode), rows[:count], mode) if diag is not None
                        else saved.get('schema') == evaluator.SCHEMA and saved.get('protocol_sha256') == protocol_sha
                        and saved.get('rows') == rows[:count] and saved.get('expected_outcomes') == len(wanted), 'Snapshot must match exact committed prefix')
                snapshot.update(state='current' if count == len(rows) else 'stale_valid_prefix', recorded_rows=count)
            collected.append({'arm': arm, 'seed': seed, 'stage': name, 'mode': mode, 'split': split, 'cells': cells,
                              'rows': rows, 'diagnostic_summary': diag, 'outcome': entry['outcome'], 'aggregate_snapshot': snapshot})
    result = {'schema': COLLECTION_SCHEMA, 'status': 'stopped_outputs_collected', 'producer': 'scalar_collector_not_root',
            'collector_sha256': collector_sha, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'input_sha256': inputs, 'cohort_sha256': inputs[str(paths[1])], 'endpoint_updates': endpoint,
            'source_manifest_sha256': ledger['source_manifest_sha256'], 'source_schedules': {split: {mode: expected(m, mode) for mode in ('full-rollout', 'same-state', 'clean-validation')}
                for split, m in manifests.items()}, 'physical_reference': {'dataset': 'Goop-3D', 'frames': 301, 'horizon': HORIZON,
                'dimension': 3, 'particle_type': 7, 'metadata': valid['metadata'], 'metadata_sha256': evaluator.METADATA_SHA,
                'source_manifest_sha256': ledger['source_manifest_sha256'], 'scope': 'exact saved ground-truth boundary references; geometric diagnostics, not conservation tests'},
            'stages': collected, 'files_sha256': files, 'output_tree_state': trees, 'ledger': ledger,
            'numeric_validation_scope': 'opaque NPZ byte hashes and scalar arithmetic only; no independent tensor/source recomputation'}
    verify_collection_snapshot(result)
    return result


def paired_metrics(values, policies, risk):
    metrics = sorted(set(k for v in values.values() for k in v))
    absolute, changes, within, interaction = {}, {}, {}, {}
    refs = ('base', 'random25', 'speed25', 'relative-velocity-RMS25')
    for metric in metrics:
        get = lambda a, s, p: values.get((a, s, p), {}).get(metric)
        absolute[metric] = {a: {p: seed_summary([get(a, s, p) for s in range(3)]) for p in policies} for a in ('base', 'mix')}
        changes[metric] = {p: seed_summary([contrast(get('mix', s, p), get('base', s, p)) for s in range(3)]) for p in policies}
        within[metric] = {a: {p + '_minus_' + ref: seed_summary([contrast(get(a, s, p), get(a, s, ref)) for s in range(3)])
                             for ref in refs for p in policies if p != ref} for a in ('base', 'mix')}
        interaction[metric] = seed_summary([contrast(contrast(get('mix', s, risk), get('mix', s, 'random25')),
                                                      contrast(get('base', s, risk), get('base', s, 'random25'))) for s in range(3)])
    return {'absolute': absolute, 'mix_minus_base_training': changes, 'within_arm_policy_contrasts': within,
            'risk_minus_random_mix_minus_base_interaction': interaction,
            'interaction_formula': '(mix-trained risk - mix-trained random) - (base-trained risk - base-trained random)'}


def runtime(stages):
    return {'invocations': [{'arm': s['arm'], 'seed': s['seed'], 'stage': s['stage'], 'outcome': s['outcome'],
              'committed_rows': len(s['rows']), 'committed_call_seconds': sum(r['synchronized_call_seconds'] for r in s['rows']),
              'committed_publication_seconds': sum(r['publication_seconds'] for r in s['rows']),
              'per_rollout_policy': {p: {'committed_cases': sum(r.get('policy') == p for r in s['rows']),
                  'committed_call_seconds': sum(r['synchronized_call_seconds'] for r in s['rows'] if r.get('policy') == p),
                  'committed_publication_seconds': sum(r['publication_seconds'] for r in s['rows'] if r.get('policy') == p)}
                  for p in POLICIES} if s.get('mode') == 'full-rollout' or s['stage'].startswith('full_rollout_') else None} for s in stages],
            'scope': 'Whole-child lifetimes include setup and interrupted work; child sums are not elapsed wall time. Fixed-order rollout calls do not establish causal speedup. Edge counts are not runtime measurements.'}


def full_summary(stages, schedule):
    expected_items = {(r['source_index'], p): {**r, 'policy': p} for r in schedule for p in POLICIES}
    rows, prefixes, truth = {}, [], {}
    for stage in stages:
        for row in stage['rows']:
            key = (stage['arm'], stage['seed'], row['source_index'], row['policy']); require(key not in rows, 'Duplicate full rollout row')
            metrics, prefix = validate_rollout(row, stage['arm'], stage['seed'], expected_items[row['source_index'], row['policy']]); rows[key] = metrics
            for step, value in enumerate(row['ground_truth_boundary_per_step'], 1):
                identity = (row['source_index'], step); require(identity not in truth or truth[identity] == value, 'Same-source truth boundary differs')
                truth[identity] = value
            if row['status'] == 'failed': prefixes.append({'arm': stage['arm'], 'seed': stage['seed'], 'source_index': row['source_index'],
                'policy': row['policy'], 'completed_steps': row['completed_steps'], 'failure': row['failure'], 'accepted_prefix': prefix})
    metric_names = FULL_METRICS
    values = {(a, s, p): {k: mean_complete([rows.get((a, s, item['source_index'], p), {}).get(k) for item in schedule]) for k in metric_names}
              for a, s in MODELS for p in POLICIES}
    counts = Counter(c['state'] for stage in stages for c in stage['cells']); required = len(schedule) * len(POLICIES) * len(MODELS)
    require(sum(counts.values()) == required, 'Full declared coverage count differs')
    return {'required_outcomes': required, 'coverage': dict(counts),
            'all_required_outcomes_complete': counts['completed_required_outcome'] == required,
            'coverage_by_model': {f'{s["arm"]}_seed{s["seed"]}': dict(Counter(c['state'] for c in s['cells'])) for s in stages},
            'failure_categories_by_model': {f'{s["arm"]}_seed{s["seed"]}': dict(Counter(
                r['failure']['category'] for r in s['rows'] if r['status'] == 'failed')) for s in stages},
            'paired': paired_metrics(values, POLICIES, 'laggedrisk25'), 'failed_accepted_prefixes': prefixes,
            'ground_truth_boundary_by_source': {str(item['source_index']): {
                'defined_forecast_steps': sum((item['source_index'], t) in truth for t in range(1, HORIZON + 1)),
                'required_forecast_steps': HORIZON,
                'full_horizon': boundary_series([truth[item['source_index'], t] for t in range(1, HORIZON + 1)], HORIZON)
                    if all((item['source_index'], t) in truth for t in range(1, HORIZON + 1)) else None} for item in schedule},
            'runtime': runtime(stages), 'scope': 'Equal declared sources per model, then three paired training seeds. H200 is pointwise and can exist before a later failure; H295 and full means require every declared cell. No conservation claim.'}


def leaves(value, path=()):
    if isinstance(value, dict):
        if 'equal_trajectory_mean' in value: return {'/'.join(path): value['equal_trajectory_mean']}
        result = {}
        for k, v in value.items(): result.update(leaves(v, path + (k,)))
        return result
    return {}


def diagnostic_summary(stages, schedule, mode):
    diagnostic = load_scalar('goop3d_diagnostic_metrics_v1.py'); flat = {}; augmented = {}; truth = {}
    for stage in stages:
        require(stage['diagnostic_summary'] == diagnostic.summarize(schedule, stage['rows'], mode), 'Collected diagnostic arithmetic differs')
        summary = stage['diagnostic_summary']; arm, seed = stage['arm'], stage['seed']
        flat[arm, seed] = leaves(summary)
        if mode == 'same-state':
            extra = {'boundary': {}, 'graph': {}}
            for method in TIMING_CASES:
                extra['boundary'][method] = {k: diagnostic.aggregate(schedule, stage['rows'], lambda r, p=method, key=k:
                    (r.get('policies', {}).get(p, {}).get('prediction_boundary') or {}).get(key)) for k in B_KEYS}
                if method != 'natural_base_reference':
                    extra['graph'][method] = {k: diagnostic.aggregate(schedule, stage['rows'], lambda r, p=method, key=k:
                        (r.get('policies', {}).get(p, {}).get('graph') or {}).get(key)) for k in G_KEYS}
            extra['truth_boundary'] = {k: diagnostic.aggregate(schedule, stage['rows'], lambda r, key=k: r.get('truth_boundary', {}).get(key)) for k in B_KEYS}
            for row in stage['rows']:
                if 'truth_boundary' in row:
                    identity = (row['source_index'], row['target_frame']); require(identity not in truth or truth[identity] == row['truth_boundary'], 'Observed truth boundary differs')
                    truth[identity] = row['truth_boundary']
            flat[arm, seed].update(leaves(extra)); augmented[f'{arm}_seed{seed}'] = extra
    keys = sorted(set().union(*(v.keys() for v in flat.values())))
    absolute = {a: {k: seed_summary([flat[a, s].get(k) for s in range(3)]) for k in keys} for a in ('base', 'mix')}
    change = {k: seed_summary([contrast(flat['mix', s].get(k), flat['base', s].get(k)) for s in range(3)]) for k in keys}
    result = {'absolute': absolute, 'mix_minus_base_training': change, 'runtime': runtime(stages), 'augmented_graph_and_boundary': augmented,
              'coverage_by_model': {f'{s["arm"]}_seed{s["seed"]}': dict(Counter(c['state'] for c in s['cells'])) for s in stages},
              'model_summaries': {f'{s["arm"]}_seed{s["seed"]}': s['diagnostic_summary'] for s in stages}}
    if mode == 'same-state':
        values = {(a, s, p): {k: flat[a, s].get(f'accuracy/{p}/{k}') for k in ('position_coordinate_mse', 'normalized_coordinate_mse')}
                  for a, s in MODELS for p in DIAG_POLICIES}
        result['accuracy_policy_contrasts'] = paired_metrics(values, DIAG_POLICIES, 'previous-observed-base-risk25')
        rowmap = {(s['arm'], s['seed'], r['source_index'], r['target_frame']): r for s in stages for r in s['rows']}
        pairing = []
        for seed in range(3):
            for item in schedule:
                a = rowmap.get(('base', seed, item['source_index'], item['target_frame']), {})
                b = rowmap.get(('mix', seed, item['source_index'], item['target_frame']), {})
                for policy in DIAG_POLICIES[:-1]:
                    left = a.get('policies', {}).get(policy, {}); right = b.get('policies', {}).get(policy, {})
                    verifiable = left.get('status') == right.get('status') == 'complete'
                    if verifiable:
                        require(a['random_seed_material'] == b['random_seed_material']
                                and all(left['graph'][k] == right['graph'][k] for k in (*G_KEYS, 'native_base_sha256',
                                    'selected_optional_pair_sha256', 'directed_edge_sha256')), 'Paired observed model-independent graph selection differs')
                    pairing.append({'seed': seed, 'source_index': item['source_index'], 'target_frame': item['target_frame'],
                                    'policy': policy, 'state': 'same_graph_and_draw_verified' if verifiable else 'required_policy_missing_or_failed'})
        result['paired_observed_graph_selection'] = pairing
    return result


def summarize(collection_path, root_release):
    summarizer_sha = sha(__file__)
    root_release = json.loads(json.dumps(root_release, allow_nan=False))
    collection, collection_sha = json_snapshot(collection_path, root_release.get('collection_sha256'))
    require(root_release.get('schema') == 'adaptgns_goop3d_scalar_analysis_release_v1' and root_release.get('issued_by') == 'root'
            and root_release.get('status') == 'approved_fixed_scalar_aggregation' and root_release.get('summarizer_sha256') == summarizer_sha,
            'Exact D3 root analysis release required')
    require(collection.get('schema') == COLLECTION_SCHEMA and collection.get('status') == 'stopped_outputs_collected'
            and collection.get('collector_sha256') == summarizer_sha and collection.get('cohort_sha256') == root_release.get('cohort_sha256')
            and collection.get('endpoint_updates') == root_release.get('endpoint_updates'), 'Frozen D3 scalar collection differs')
    stages = collection['stages']; grid = {(a, s, name) for a, s in MODELS for name, _, _ in STAGES}
    require(len(stages) == 30 and {(s['arm'], s['seed'], s['stage']) for s in stages} == grid, 'Complete model-stage accounting required')
    outputs = {}
    for name, mode, split in STAGES:
        selected = [s for s in stages if s['stage'] == name]; schedule = collection['source_schedules'][split][mode]
        outputs[name] = full_summary(selected, schedule) if mode == 'full-rollout' else diagnostic_summary(selected, schedule, mode)
    require(sha(collection_path) == collection_sha and sha(__file__) == summarizer_sha
            and all(sha(HERE / name) == pin for name, pin in SOURCE_PINS.items()), 'Collection or analysis source changed during aggregation')
    return {'schema': SCHEMA, 'status': 'fixed_scalar_aggregation_complete', 'dataset': 'Goop-3D',
            'endpoint_updates': collection['endpoint_updates'], 'cohort_sha256': collection['cohort_sha256'],
            'collection_sha256': collection_sha, 'summarizer_sha256': summarizer_sha, 'stages': outputs,
            'required_cell_accounting': {f'{s["arm"]}_seed{s["seed"]}_{s["stage"]}': s['cells'] for s in stages},
            'physical_reference': collection['physical_reference'], 'source_manifest_sha256': collection['source_manifest_sha256'],
            'scope': 'Fresh separately selected D3 endpoint; distinct from Goop2D100k and WaterDrop100k plus10k continuation. Fixed source-order grid is not a probability sample. All required missing/failed values remain null, all three seed values retained, no p-values or survivor means.',
            'interpretation': 'Graph-exposure improvement alone does not establish residual-risk ranking advantage. The interaction is reported alongside each arm and overall mix-base effects. Boundaries are geometric diagnostics, not conservation; edge counts are not measured speedup.',
            'numeric_validation_scope': collection['numeric_validation_scope']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true'); p.add_argument('--mode', choices=('collect', 'summarize'))
    for name in ('ledger', 'cohort', 'cohort-audit', 'valid-manifest', 'test-manifest', 'protocol', 'collection', 'root-release', 'output'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema': SCHEMA, 'status': 'description_only', 'horizon': HORIZON, 'dimension': 3,
                          'required_models': 6, 'required_model_stages': 30, 'source_grid': 'allN<=30 otherwise floor(j*(N-1)/29)',
                          'numeric_arrays_deserialized': False, 'scientific_endpoint_selected': False})); return 0
    require(args.output is not None and not args.output.exists() and args.root_release is not None, 'Fresh output and root release required')
    own_source_sha = sha(__file__)
    root_release, root_release_sha = json_snapshot(args.root_release)
    if args.mode == 'collect':
        names = ('ledger', 'cohort', 'cohort_audit', 'valid_manifest', 'test_manifest', 'protocol')
        require(all(getattr(args, name) is not None for name in names), 'All D3 collection inputs required')
        output = args.output.resolve()
        require(all(output != Path(s['directory']).resolve() and Path(s['directory']).resolve() not in output.parents
                    for s in read(args.ledger).get('stages', [])), 'Collection output must be outside stopped stage directories')
        result = collect(*(getattr(args, name) for name in names), root_release)
    elif args.mode == 'summarize':
        require(args.collection is not None, 'Exact scalar collection required'); result = summarize(args.collection, root_release)
    else: p.error('Explicit mode required')
    result['root_release_sha256'] = root_release_sha
    serialized = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    # Serialization precedes terminal checks: no input reads or outcome work
    # follows these checks before exclusive publication.
    if args.mode == 'collect': verify_collection_snapshot(result)
    else:
        require(sha(args.collection) == result['collection_sha256']
                and all(sha(HERE / name) == pin for name, pin in SOURCE_PINS.items()), 'Collection/frozen sources changed before publication')
    require(sha(__file__) == own_source_sha and sha(args.root_release) == root_release_sha, 'Own source/root release changed before publication')
    with args.output.open('xb') as stream: stream.write(serialized)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
