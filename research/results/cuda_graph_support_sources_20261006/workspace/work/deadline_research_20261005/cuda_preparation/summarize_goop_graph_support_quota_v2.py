#!/usr/bin/env python3
"""Post-execution Goop scalar/coverage audit and paired-seed aggregation.

Description only by default. collect reads stopped evaluation outputs and hashes
referenced numeric artifacts; summarize reads two root-bound collection receipts.
No model execution, test source loading, training, selection or statistical tests.
Raw arrays are byte-verified, not independently recomputed; this scope is explicit.
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
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop_graph_support_paired_scalar_summary_quota_v2'
COLLECTION_SCHEMA = 'adaptgns_goop_evaluation_collection_quota_v2'
EVALUATOR_SHA = 'cd970e04012930d896b31944271ccaf7da2b0881f22fe2f903533a8d5911dc6d'
QUOTA_SHA = '0bd714b0bece0c349a9b18eae14221f6602ffc61ea736bad4fa3b5dc087157d9'
PROTOCOL_SHA = 'c8690d0da209c66557e3dbb0cbccd55d660b4cc270683913f74d381516591851'
TRAINER_SHA = 'dd9ef01a16116f04a578bbfe9ed01774e66fdc08d60b913072a33ad211b319c1'
BENCH_SHA = '2e0ef0b84635c8430102cd797648d24cec14b457e1df900eb784d5e167966f8f'
Q_SCHEMA = 'adaptgns_graph_support_evaluation_quota_v2'
E_SCHEMA = 'adaptgns_goop_graph_support_final_evaluation_v1'
POLICIES = ('base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25')
DIAG_POLICIES = ('base', 'dense', 'random25', 'speed25', 'relative-velocity-RMS25', 'previous-observed-base-risk25')
STAGES = (('full_rollout_test', 'full-rollout', 'test'), ('same_state_valid', 'same-state', 'valid'),
          ('same_state_test', 'same-state', 'test'), ('clean_validation', 'clean-validation', 'valid'))
B_KEYS = ('fraction_particles_outside', 'fraction_particles_outside_by_more_than_1e-6',
          'maximum_coordinate_excursion', 'mean_particle_maximum_excursion')
C_STATES = {'completed_required_outcome', 'recorded_failed_outcome', 'timed_out_current',
            'not_completed_before_invocation_end', 'never_started'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f: f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def finite(v, nonnegative=False):
    return type(v) in (int, float) and math.isfinite(v) and (not nonnegative or v >= 0)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def mean_complete(values):
    return statistics.fmean(values) if values and all(finite(v) for v in values) else None


def contrast(a, b):
    return a - b if finite(a) and finite(b) else None


def seed_summary(values):
    require(len(values) == 3, 'Exactly three ordered seed values required')
    complete = all(finite(v) for v in values)
    return {'seed_values': {str(i): v for i, v in enumerate(values)}, 'required_seed_pairs': 3,
            'defined_seed_pairs': sum(finite(v) for v in values), 'mean': statistics.fmean(values) if complete else None,
            'sample_sd': statistics.stdev(values) if complete else None}


def expected(stage):
    if stage == 'full_rollout_test': return [(i, p) for i in range(30) for p in POLICIES]
    if stage == 'clean_validation':
        return [(n // 395, n % 395 + 6) for n in (i * (11850 - 1) // 127 for i in range(128))]
    return [(i, t) for i in range(30) for t in (7, 105, 203, 301, 400)]


def identity(row, stage):
    return row['source_index'], row['policy' if stage == 'full_rollout_test' else 'target_frame']


def import_frozen(name, path, digest):
    require(sha(path) == digest, 'Frozen helper differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def boundary(series, completed):
    require(isinstance(series, list) and len(series) == completed, 'Boundary accepted-prefix length differs')
    for item in series:
        require(all(finite(item.get(k), True) for k in B_KEYS)
                and 0 <= item[B_KEYS[1]] <= item[B_KEYS[0]] <= 1
                and item['mean_particle_maximum_excursion'] <= item['maximum_coordinate_excursion'] + 1e-15,
                'Invalid boundary scalar')
        require(all(isinstance(item.get(k), list) and len(item[k]) == 2 and all(finite(v) for v in item[k])
                    for k in ('coordinate_minimum', 'coordinate_maximum'))
                and all(a <= b for a, b in zip(item['coordinate_minimum'], item['coordinate_maximum'])), 'Invalid boundary coordinate range')
    result = {f'mean_{key}': mean_complete([item[key] for item in series]) for key in B_KEYS}
    result['trajectory_maximum_excursion'] = max((item['maximum_coordinate_excursion'] for item in series), default=None)
    return result


def validate_rollout(row, arm, seed):
    require(row.get('arm') == arm and row.get('training_seed') == seed and type(row.get('training_seed')) is int
            and row.get('objective') == 'faithful' and row.get('horizon') == 395
            and row.get('policy') in POLICIES and type(row.get('source_index')) is int and 0 <= row['source_index'] < 30,
            'Rollout identity/horizon differs')
    completed = row.get('completed_steps')
    require(type(completed) is int and 0 <= completed <= 395 and row.get('status') in ('complete', 'failed'), 'Invalid rollout endpoint/status')
    mse = row.get('mse_per_step')
    require(isinstance(mse, list) and len(mse) == completed and all(finite(v, True) for v in mse), 'Invalid accepted MSE prefix')
    complete = row['status'] == 'complete'
    require(complete == (completed == 395) and (row.get('failure') is None if complete else isinstance(row.get('failure'), dict)),
            'Complete/failed outcome inconsistent')
    if complete:
        require(close(row.get('mean_rollout_mse'), statistics.fmean(mse)) and close(row.get('mse_at_final_horizon'), mse[-1]), 'Saved full-horizon MSE arithmetic differs')
    else:
        require(row.get('mean_rollout_mse') is None and row.get('mse_at_final_horizon') is None
                and isinstance(row['failure'].get('category'), str), 'Failed full-horizon MSE must remain null')
    for step in (1, 10, 50, 200, 395):
        v = row.get('mse_at_declared_trace_steps', {}).get(str(step))
        require(close(v, mse[step - 1]) if completed >= step else v is None, 'Declared trace MSE differs from pointwise forecast')
    require(finite(row.get('synchronized_call_seconds'), True) and finite(row.get('trace_publication_seconds'), True), 'Invalid measured duration')
    metrics = {'mean_rollout_mse': statistics.fmean(mse) if complete else None,
               'mse_forecast200': mse[199] if completed >= 200 else None, 'mse_forecast395': mse[394] if complete else None}
    prefix = {}
    for side in ('predicted', 'ground_truth'):
        values = boundary(row.get(side + '_boundary_per_step'), completed)
        metrics.update({side + '_boundary_' + k: v if complete else None for k, v in values.items()})
        if not complete: prefix.update({side + '_boundary_' + k: v for k, v in values.items()})
    return metrics, prefix


def row_path(directory, cell):
    path = directory / Path(cell['path']).name
    require(path.parent == directory and path.is_file() and not path.is_symlink() and sha(path) == cell['sha256'], 'Saved row missing/changed')
    return path


def collect(queue_root, cohort_path):
    root = Path(queue_root).resolve(); cohort = read(cohort_path); cohort_sha = sha(cohort_path)
    quota = import_frozen('_goop_scalar_quota', HERE / 'supervise_graph_support_evaluation_quota_v2.py', QUOTA_SHA)
    evaluator = import_frozen('_goop_scalar_final', HERE / 'evaluate_goop_graph_support_final.py', EVALUATOR_SHA)
    release, status, ledger, process = [read(root / name) for name in ('release_snapshot.json', 'queue_status.json', 'coverage_ledger.json', 'process_outcomes.json')]
    require(release.get('schema') == 'adaptgns_graph_support_evaluation_quota_release_v2' and release.get('issued_by') == 'root'
            and release.get('dataset') == ledger.get('dataset') == 'Goop' and release.get('host_role') in ('A', 'B'), 'Root Goop queue release required')
    require(status.get('schema') == ledger.get('schema') == process.get('schema') == Q_SCHEMA
            and status.get('state') in ('allocation_finished', 'stopped_requires_review')
            and status.get('coverage_ledger_sha256') == sha(root / 'coverage_ledger.json')
            and status.get('unreaped_owned_children') == ledger.get('unreaped_owned_children') == process.get('unreaped_owned_children') == [],
            'Only stopped/reaped evaluation queues may be collected')
    require(cohort.get('schema') == 'adaptgns_goop_graph_support_final_cohort_v1' and cohort.get('status') == 'frozen_for_final_evaluation'
            and cohort.get('protocol_sha256') == PROTOCOL_SHA and cohort.get('updates') == 100000, 'Frozen exact100k Goop cohort required')
    role = release['host_role']; stream_ids = [f'{a}_seed{s}' for a, s, _ in quota.SCHEDULE[role]]
    require(len(release.get('streams', [])) == len(stream_ids) and {r.get('id') for r in release['streams']} == set(stream_ids), 'Queue streams differ')
    entries = ledger.get('stages', [])
    require(len(entries) == 4 * len(stream_ids) and len({(e.get('stream'), e.get('stage')) for e in entries}) == len(entries), 'Duplicate/missing coverage stage')
    collected, files = [], {}
    for stream in release['streams']:
        arm, seed = stream['arm'], stream['seed']
        model = next(r for r in cohort['models'] if (r['arm'], r['seed']) == (arm, seed))
        require(stream['id'] == f'{arm}_seed{seed}' and len(stream['commands']) == 4, 'Invalid stream identity/commands')
        for command, (stage, mode, split) in zip(stream['commands'], STAGES):
            options = quota.flags(command)
            require(options.get('--mode') == mode and options.get('--split') == split and options.get('--arm') == arm
                    and options.get('--seed') == str(seed) and options.get('--checkpoint-sha256') == model['checkpoint_sha256'], 'Released model/stage differs')
            required_inputs = {'--cohort': cohort_sha, '--protocol': PROTOCOL_SHA, '--trainer-source': TRAINER_SHA,
                               '--benchmark-helper': BENCH_SHA, '--checkpoint': model['checkpoint_sha256']}
            require(all(release['files_sha256'].get(options.get(k)) == v for k, v in required_inputs.items())
                    and release['files_sha256'].get(command[1]) == EVALUATOR_SHA, 'Released final input hash differs')
            directory = root / 'jobs' / stream['id'] / stage
            entry = next(e for e in entries if (e['stream'], e['stage']) == (stream['id'], stage))
            require(entry.get('coverage_audit_state') == 'row_coverage_checked', 'Coverage audit failed/deferred; inspect original receipts')
            recomputed = quota.account_stage('Goop', stage, directory, entry['outcome'])
            cells = entry['cells']; wanted = expected(stage)
            require(len(cells) == len(wanted) and [identity(c, stage) for c in cells] == wanted
                    and all(c.get('state') in C_STATES for c in cells), 'Coverage grid differs')
            def normalized(cell): return {k: (Path(v).name if k == 'path' else v) for k, v in cell.items()}
            require([normalized(c) for c in cells] == [normalized(c) for c in recomputed['cells']], 'Fresh committed row coverage differs from ledger')
            rows = []
            protocol_path = directory / 'protocol.json'
            protocol = read(protocol_path) if protocol_path.exists() else None
            if protocol is not None:
                require(protocol.get('schema') == E_SCHEMA and protocol.get('mode') == mode and protocol.get('split') == split
                        and protocol.get('source_frame_count') == 401 and protocol.get('model', {}).get('kind') == 'preselected_checkpoint'
                        and protocol['model'].get('completed_updates') == 100000 and protocol['model'].get('arm') == arm
                        and protocol['model'].get('seed') == seed and protocol['model'].get('checkpoint_sha256') == model['checkpoint_sha256'], 'Saved evaluation protocol/model differs')
                require(all(protocol.get('input_files_sha256', {}).get(options[k]) == v for k, v in required_inputs.items()), 'Saved evaluation input hashes differ')
                plan = [(r['source_index'], r['target_frame']) for r in protocol['schedule']] if mode != 'full-rollout' else [r['source_index'] for r in protocol['schedule']]
                require(plan == (wanted if mode != 'full-rollout' else list(range(30))), 'Fixed diagnostic/trajectory schedule differs')
                require(protocol.get('policies') == list(POLICIES if mode == 'full-rollout' else DIAG_POLICIES), 'Fixed policies differ')
            for cell in cells:
                if cell['state'] not in ('completed_required_outcome', 'recorded_failed_outcome'): continue
                require(protocol is not None, 'Committed row requires bound evaluation protocol')
                path = row_path(directory, cell); row = read(path)
                require(row.get('protocol_sha256') == sha(protocol_path), 'Row protocol hash differs')
                artifact_key = 'trace_file' if mode == 'full-rollout' else 'artifact_file'
                artifact = directory / row[artifact_key]
                require(artifact.parent == directory and artifact.is_file() and not artifact.is_symlink()
                        and sha(artifact) == row['trace_sha256' if mode == 'full-rollout' else 'artifact_sha256'], 'Referenced numeric artifact bytes differ')
                if mode == 'full-rollout': validate_rollout(row, arm, seed)
                rows.append(row)
            snapshot = {'state': 'absent', 'recorded_rows': 0, 'committed_rows': len(rows)}
            if mode == 'full-rollout' and (directory / 'result.json').exists():
                result = read(directory / 'result.json')
                require(result.get('schema') == E_SCHEMA and result.get('protocol_sha256') == sha(protocol_path)
                        and isinstance(result.get('rows'), list) and len(result['rows']) <= len(rows)
                        and result['rows'] == rows[:len(result['rows'])], 'Result snapshot is not an exact committed prefix')
                snapshot.update(state='current' if len(result['rows']) == len(rows) else 'stale_valid_prefix', recorded_rows=len(result['rows']))
            diag = None
            if mode != 'full-rollout':
                schedule = [{'source_index': i, 'target_frame': t} for i, t in wanted]
                diag = evaluator.summarize(schedule, rows, mode)
                if (directory / 'summary.json').exists():
                    saved = read(directory / 'summary.json'); count = saved.get('returned_frames')
                    require(type(count) is int and 0 <= count <= len(rows)
                            and saved == evaluator.summarize(schedule, rows[:count], mode), 'Stored diagnostic summary is not an exact committed prefix')
                    snapshot.update(state='current' if count == len(rows) else 'stale_valid_prefix', recorded_rows=count)
            collected.append({'arm': arm, 'seed': seed, 'stage': stage, 'mode': mode, 'split': split,
                'cells': cells, 'rows': rows, 'diagnostic_summary': diag, 'outcome': entry['outcome'],
                'protocol_sha256': sha(protocol_path) if protocol_path.exists() else None, 'aggregate_snapshot': snapshot})
    # Preserve every ordinary output and unsuccessful artifact, including temporary files.
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Evaluation snapshot must not contain symlinks')
        if path.is_file(): files[str(path.relative_to(root))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    require(sha(root / 'coverage_ledger.json') == status['coverage_ledger_sha256'], 'Coverage changed during collection')
    return {'schema': COLLECTION_SCHEMA, 'status': 'stopped_outputs_collected', 'issued_by': 'root', 'host_role': role,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'collector_sha256': sha(__file__), 'cohort_sha256': cohort_sha,
            'protocol_sha256': PROTOCOL_SHA, 'stages': collected, 'files': files, 'queue_status': status,
            'process_outcomes': process, 'coverage_audit_errors': ledger.get('audit_errors'), 'abort_reason': ledger.get('abort_reason'),
            'scope': 'committed row identity/arithmetic, complete coverage, diagnostic aggregation and numeric artifact byte hashes; no fresh source/array recomputation'}


def full_summary(stages):
    rows = {}; coverage = []; prefixes = []; timing = []
    truth = {}
    for stage in stages:
        arm, seed = stage['arm'], stage['seed']
        for row in stage['rows']:
            key = (arm, seed, row['source_index'], row['policy'])
            require(key not in rows, 'Duplicate rollout row')
            metrics, prefix = validate_rollout(row, arm, seed); rows[key] = metrics
            for step, value in enumerate(row['ground_truth_boundary_per_step']):
                k = (row['source_index'], step)
                require(k not in truth or truth[k] == value, 'Same-source boundary truth differs across model/policy')
                truth[k] = value
            if row['status'] == 'failed':
                prefixes.append({'arm': arm, 'seed': seed, 'source_index': row['source_index'], 'policy': row['policy'],
                                 'completed_steps': row['completed_steps'], 'failure': row['failure'], 'accepted_prefix_boundary': prefix})
        counts = dict(Counter(c['state'] for c in stage['cells']))
        failures = dict(Counter(c.get('failure', {}).get('category', 'unspecified') for c in stage['cells'] if c['state'] == 'recorded_failed_outcome'))
        coverage.append({'arm': arm, 'seed': seed, 'counts': counts, 'guard_or_execution_failure_categories': failures})
        timing.append({'arm': arm, 'seed': seed, 'whole_invocation': stage['outcome'],
            'committed_call_seconds': sum(r['synchronized_call_seconds'] for r in stage['rows']),
            'committed_publication_seconds': sum(r['trace_publication_seconds'] for r in stage['rows']),
            'per_policy': {p: {'committed_cases': sum(r['policy'] == p for r in stage['rows']),
                'committed_call_seconds': sum(r['synchronized_call_seconds'] for r in stage['rows'] if r['policy'] == p)} for p in POLICIES}})
    metric_names = ['mean_rollout_mse', 'mse_forecast200', 'mse_forecast395'] + [s + '_boundary_' + k for s in ('predicted', 'ground_truth')
        for k in (*['mean_' + k for k in B_KEYS], 'trajectory_maximum_excursion')]
    absolute, within, arm_change, interactions = {}, {}, {}, {}
    seed_means = {}
    for metric in metric_names:
        seed_means[metric] = {(a, s, p): mean_complete([rows.get((a, s, i, p), {}).get(metric) for i in range(30)])
                             for a in ('base', 'mix') for s in range(3) for p in POLICIES}
        values = seed_means[metric]
        absolute[metric] = {a: {p: seed_summary([values[a, s, p] for s in range(3)]) for p in POLICIES} for a in ('base', 'mix')}
        arm_change[metric] = {p: seed_summary([contrast(values['mix', s, p], values['base', s, p]) for s in range(3)]) for p in POLICIES}
        within[metric] = {a: {p + '_minus_' + ref: seed_summary([contrast(values[a, s, p], values[a, s, ref]) for s in range(3)])
                             for ref in ('base', 'random25') for p in POLICIES if p != ref} for a in ('base', 'mix')}
        interactions[metric] = seed_summary([contrast(contrast(values['mix', s, 'laggedrisk25'], values['mix', s, 'random25']),
                                                       contrast(values['base', s, 'laggedrisk25'], values['base', s, 'random25'])) for s in range(3)])
    all_counts = Counter(c['state'] for stage in stages for c in stage['cells'])
    require(sum(all_counts.values()) == 1080, 'Exactly1080 required full-rollout outcomes')
    return {'required_outcomes': 1080, 'coverage': dict(all_counts), 'coverage_by_model': coverage,
        'all_required_outcomes_recorded': all_counts['completed_required_outcome'] + all_counts['recorded_failed_outcome'] == 1080,
        'all_required_outcomes_complete': all_counts['completed_required_outcome'] == 1080,
        'absolute': absolute, 'within_arm_policy_contrasts': within, 'mix_minus_base': arm_change,
        'risk_minus_random_mix_minus_base_interaction': interactions, 'failed_accepted_prefixes': prefixes, 'runtime': timing,
        'runtime_scope': 'committed synchronized calls include parity; parent whole-invocation time includes imports/load/publication and interrupted work; fixed policy order does not establish causal speedup',
        'endpoint_scope': 'forecast200/395 are pointwise step errors;200 may remain defined for a committed accepted prefix whose later full horizon failed',
        'boundary_scope': 'full395 accepted steps only for all-sample means; failed accepted-prefix scalars retained separately; boundaries are geometric diagnostics'}


def diagnostic_leaves(value, prefix=()):
    if isinstance(value, dict):
        if 'equal_trajectory_mean' in value: return {'/'.join(prefix): value['equal_trajectory_mean']}
        result = {}
        for k, v in value.items(): result.update(diagnostic_leaves(v, prefix + (k,)))
        return result
    return {}


def diagnostic_summary(stages, stage_name):
    chosen = [s for s in stages if s['stage'] == stage_name]
    require(len(chosen) == 6 and {(s['arm'], s['seed']) for s in chosen} == {(a, s) for a in ('base', 'mix') for s in range(3)}, 'Complete diagnostic model grid required')
    leaf = {(s['arm'], s['seed']): diagnostic_leaves(s['diagnostic_summary']) for s in chosen}
    keys = sorted(set().union(*(r.keys() for r in leaf.values())))
    absolute = {a: {k: seed_summary([leaf[a, s].get(k) for s in range(3)]) for k in keys} for a in ('base', 'mix')}
    change = {k: seed_summary([contrast(leaf['mix', s].get(k), leaf['base', s].get(k)) for s in range(3)]) for k in keys}
    result = {'absolute': absolute, 'mix_minus_base': change,
        'coverage': {f'{s["arm"]}_seed{s["seed"]}': dict(Counter(c['state'] for c in s['cells'])) for s in chosen},
        'model_summaries': {f'{s["arm"]}_seed{s["seed"]}': s['diagnostic_summary'] for s in chosen},
        'runtime': {f'{s["arm"]}_seed{s["seed"]}': s['outcome'] for s in chosen},
        'aggregation': 'same frozen within-frame/equal-frame-within-source/equal-source aggregation, then three paired seed values; no undefined-value dropping'}
    if stage_name != 'clean_validation':
        risk = 'accuracy/previous-observed-base-risk25/position_coordinate_mse'; random = 'accuracy/random25/position_coordinate_mse'
        within = {a: [contrast(leaf[a, s].get(risk), leaf[a, s].get(random)) for s in range(3)] for a in ('base', 'mix')}
        result['previous_observed_risk_minus_random_position_mse'] = {a: seed_summary(v) for a, v in within.items()}
        result['risk_minus_random_mix_minus_base_interaction'] = seed_summary([contrast(within['mix'][s], within['base'][s]) for s in range(3)])
    return result


def summarize(collection_paths, root_release):
    require(root_release.get('schema') == 'adaptgns_goop_paired_scalar_analysis_release_quota_v2'
            and root_release.get('issued_by') == 'root' and root_release.get('status') == 'approved_for_fixed_scalar_aggregation'
            and root_release.get('summarizer_sha256') == sha(__file__), 'Exact root analysis release required')
    collections = []
    for role in ('A', 'B'):
        path = collection_paths[role]; value = read(path)
        require(sha(path) == root_release.get('collection_sha256', {}).get(role)
                and value.get('schema') == COLLECTION_SCHEMA and value.get('status') == 'stopped_outputs_collected'
                and value.get('collector_sha256') == sha(__file__) and value.get('host_role') == role
                and value.get('cohort_sha256') == root_release.get('cohort_sha256') and value.get('protocol_sha256') == PROTOCOL_SHA,
                'Root-bound collection differs')
        require(value.get('queue_status', {}).get('all_pinned_inputs_reverified') is True,
                'Input integrity was not reverified; preserved collection cannot enter scientific aggregation')
        collections.append(value)
    stages = [s for c in collections for s in c['stages']]
    grid = {(a, s, stage) for a in ('base', 'mix') for s in range(3) for stage, _, _ in STAGES}
    require(len(stages) == 24 and {(s['arm'], s['seed'], s['stage']) for s in stages} == grid, 'All24 model-stage receipts required, including missing work')
    return {'schema': SCHEMA, 'status': 'fixed_scalar_aggregation_complete', 'cohort_sha256': root_release['cohort_sha256'],
        'protocol_sha256': PROTOCOL_SHA, 'summarizer_sha256': sha(__file__), 'collection_sha256': root_release['collection_sha256'],
        'full_rollout': full_summary([s for s in stages if s['stage'] == 'full_rollout_test']),
        'diagnostics': {name: diagnostic_summary(stages, name) for name, _, _ in STAGES[1:]},
        'queue_accounting': {c['host_role']: {'status': c['queue_status'], 'abort_reason': c['abort_reason'], 'audit_errors': c['coverage_audit_errors']} for c in collections},
        'scope': 'all three training seeds; null when required values missing/failed; no survivor means, p-values, replacement seeds or shortened-horizon substitution',
        'numeric_artifact_validation': 'referenced binary hashes verified during collection; scalar arithmetic and saved diagnostic aggregation recomputed; independent per-array/source recomputation remains separate',
        'interpretation': 'exploratory native-CUDA Goop graph-support extension informed by prior results; not pristine independent confirmation; negative interaction alone does not establish risk superiority'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true'); p.add_argument('--mode', choices=('collect', 'summarize'))
    for name in ('queue-root', 'cohort', 'collection-a', 'collection-b', 'root-release', 'output'): p.add_argument('--' + name, type=Path)
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'status': 'description_only', 'required_rollout_outcomes': 1080, 'required_model_stages': 24,
            'scope': 'scalar arithmetic, coverage, saved diagnostics and paired-seed aggregation; hashes of numeric artifacts, no independent array/source recomputation'}, indent=2)); return 0
    require(a.output is not None and not a.output.exists(), 'Fresh explicit output required')
    if a.mode == 'collect':
        require(a.queue_root and a.cohort and a.queue_root.resolve() not in a.output.resolve().parents, 'Queue/cohort and separate output required')
        result = collect(a.queue_root, a.cohort)
    elif a.mode == 'summarize':
        require(a.collection_a and a.collection_b and a.root_release, 'Both collections and root release required')
        result = summarize({'A': a.collection_a, 'B': a.collection_b}, read(a.root_release))
    else: p.error('Explicit mode required')
    write_new(a.output, result); return 0


if __name__ == '__main__': raise SystemExit(main())
