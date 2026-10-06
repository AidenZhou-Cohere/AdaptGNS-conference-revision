#!/usr/bin/env python3
"""Scalar-only Sand collection with one explicit scoped-supervisor version bridge.

No research execution or array deserialization. Scientific scalar primitives are imported by exact hash from quota-v2.
Sand endpoint/grid adapters are explicit; paired aggregation/null rules retain
the same equations. Operational lineage/snapshot handling follows reviewed Goop v3. Pending/rejected GPU observations and every partial output are retained.
"""
import argparse
from collections import Counter
import statistics
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_sand_graph_support_paired_scalar_summary_scoped_v1'
COLLECTION_SCHEMA = 'adaptgns_sand_evaluation_collection_scoped_v1'
COLLECTION_RELEASE_SCHEMA = 'adaptgns_sand_scalar_collection_release_scoped_v1'
ANALYSIS_RELEASE_SCHEMA = 'adaptgns_sand_paired_scalar_analysis_release_scoped_v1'
Q_SCHEMA = 'adaptgns_sand_evaluation_gpu_scoped_v1'
Q_RELEASE_SCHEMA = 'adaptgns_sand_evaluation_gpu_scoped_release_v1'
AMENDMENT_SHA = '411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738'
SOURCE_PINS = {
    'summarize_goop_graph_support_quota_v2.py': 'da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c',
    'supervise_sand_evaluation_gpu_scoped_v1.py': '0' * 64,  # Pending reviewed operational adapter; blocks production admission.
    'supervise_goop_evaluation_gpu_scoped_v3.py': 'a13ca1bc30161f89cf79b6c0b632c65a272913302ef0cc37dcfd7ed762c6aa21',
    'evaluate_sand_graph_support_final.py': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58',
}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def json_snapshot(path, expected=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Ordinary JSON input required: ' + str(path))
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, 'Parsed JSON bytes differ from snapshot: ' + str(path))
    return json.loads(raw), digest


def tree_snapshot(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Ordinary queue directory required')
    entries, files = [], {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Unsupported queue entry')
        relative = str(path.relative_to(root)); is_file = path.is_file()
        entries.append([relative, 'file' if is_file else 'directory'])
        if is_file: files[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return {'entries': entries, 'files': files}


def inventory_sha256(snapshot):
    return hashlib.sha256((json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()).hexdigest()


def import_pinned(filename, label):
    path = HERE / filename
    require(sha(path) == SOURCE_PINS[filename], 'Frozen scalar/operational source differs: ' + filename)
    spec = importlib.util.spec_from_file_location(label, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


PURE = import_pinned('summarize_goop_graph_support_quota_v2.py', '_scoped_sand_frozen_v2_science')
PROTOCOL_SHA = 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d'
TRAINER_SHA = 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
BENCH_SHA = '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13'
EVALUATOR_SHA = SOURCE_PINS['evaluate_sand_graph_support_final.py']
E_SCHEMA = 'adaptgns_sand_graph_support_final_evaluation_v1'
STAGES, POLICIES, DIAG_POLICIES, C_STATES = PURE.STAGES, PURE.POLICIES, PURE.DIAG_POLICIES, PURE.C_STATES
identity, diagnostic_summary = PURE.identity, PURE.diagnostic_summary
finite, close, mean_complete, contrast, seed_summary, boundary, B_KEYS = (PURE.finite, PURE.close, PURE.mean_complete,
    PURE.contrast, PURE.seed_summary, PURE.boundary, PURE.B_KEYS)
SCHEDULE = {'A': [('base',1,0),('mix',1,1),('base',2,2),('mix',2,3)], 'B': [('base',0,2),('mix',0,3)]}
HORIZON, FRAMES = 314, 320


def expected(stage):
    if stage == 'full_rollout_test': return [(i,p) for i in range(30) for p in POLICIES]
    if stage == 'clean_validation':
        return [(n//314,n%314+6) for n in (i*(9420-1)//127 for i in range(128))]
    return [(i,t) for i in range(30) for t in (7,85,163,241,319)]


def validate_rollout(row, arm, seed):
    require(row.get('arm') == arm and row.get('training_seed') == seed and type(row.get('training_seed')) is int
            and row.get('objective') == 'faithful' and row.get('horizon') == 314
            and row.get('policy') in POLICIES and type(row.get('source_index')) is int and 0 <= row['source_index'] < 30,
            'Rollout identity/horizon differs')
    completed = row.get('completed_steps')
    require(type(completed) is int and 0 <= completed <= 314 and row.get('status') in ('complete', 'failed'), 'Invalid rollout endpoint/status')
    mse = row.get('mse_per_step')
    require(isinstance(mse, list) and len(mse) == completed and all(finite(v, True) for v in mse), 'Invalid accepted MSE prefix')
    complete = row['status'] == 'complete'
    require(complete == (completed == 314) and (row.get('failure') is None if complete else isinstance(row.get('failure'), dict)),
            'Complete/failed outcome inconsistent')
    if complete:
        require(close(row.get('mean_rollout_mse'), statistics.fmean(mse)) and close(row.get('mse_at_final_horizon'), mse[-1]), 'Saved full-horizon MSE arithmetic differs')
    else:
        require(row.get('mean_rollout_mse') is None and row.get('mse_at_final_horizon') is None
                and isinstance(row['failure'].get('category'), str), 'Failed full-horizon MSE must remain null')
    for step in (1, 10, 50, 200, 314):
        v = row.get('mse_at_declared_trace_steps', {}).get(str(step))
        require(close(v, mse[step - 1]) if completed >= step else v is None, 'Declared trace MSE differs from pointwise forecast')
    require(finite(row.get('synchronized_call_seconds'), True) and finite(row.get('trace_publication_seconds'), True), 'Invalid measured duration')
    metrics = {'mean_rollout_mse': statistics.fmean(mse) if complete else None,
               'mse_forecast200': mse[199] if completed >= 200 else None, 'mse_forecast314': mse[394] if complete else None}
    prefix = {}
    for side in ('predicted', 'ground_truth'):
        values = boundary(row.get(side + '_boundary_per_step'), completed)
        metrics.update({side + '_boundary_' + k: v if complete else None for k, v in values.items()})
        if not complete: prefix.update({side + '_boundary_' + k: v for k, v in values.items()})
    return metrics, prefix


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
    metric_names = ['mean_rollout_mse', 'mse_forecast200', 'mse_forecast314'] + [s + '_boundary_' + k for s in ('predicted', 'ground_truth')
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
        'endpoint_scope': 'forecast200/314 are pointwise step errors;200 may remain defined for a committed accepted prefix whose later full horizon failed',
        'boundary_scope': 'full314 accepted steps only for all-sample means; failed accepted-prefix scalars retained separately; boundaries are geometric diagnostics'}


def source_bindings():
    values = {str(HERE / name): pin for name, pin in SOURCE_PINS.items()}
    values[str(Path(__file__).resolve())] = sha(__file__)
    verify_bindings(values)
    return values


def verify_bindings(values):
    require(all(Path(p).is_file() and not Path(p).is_symlink() and sha(p) == h for p, h in values.items()),
            'Original input/source/release bytes changed')


def load_root(path, schema, status, bindings):
    release, digest = json_snapshot(path); bindings[str(Path(path).resolve())] = digest
    require(release.get('schema') == schema and release.get('issued_by') == 'root'
            and release.get('status') == status and release.get('collector_sha256') == sha(__file__)
            and release.get('source_sha256') == SOURCE_PINS, 'Exact scoped root scalar release required')
    require(isinstance(release.get('cohort_sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', release['cohort_sha256']), 'Explicit cohort digest required')
    if schema == ANALYSIS_RELEASE_SCHEMA:
        pins = release.get('collection_sha256', {})
        require(isinstance(pins, dict) and set(pins) == {'A','B'} and all(isinstance(v,str) and re.fullmatch(r'[0-9a-f]{64}',v) for v in pins.values()), 'Both explicit collection digests required')
    return release, digest


def canonical_uuid(value):
    require(isinstance(value, str) and re.fullmatch(r'(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', value),
            'Malformed full GPU UUID')
    return value.removeprefix('GPU-').lower()


def validate_scope(release, process, observations, quota):
    wanted = SCHEDULE[release['host_role']]
    scope = {'owned_indices': sorted({gpu for _, _, gpu in wanted}), 'unassigned_devices': 'observe_without_control',
             'timing_scope': 'shared_host_operational_measurement', 'live_training_handoff': False}
    require(release.get('gpu_scope') == process.get('gpu_scope') == scope, 'Released/process device scope differs')
    require([(s.get('arm'), s.get('seed'), s.get('gpu')) for s in release['streams']] == wanted, 'Fixed stream/device mapping differs')
    uuids = [canonical_uuid(v) for v in release.get('gpu_uuids', [])]
    require(len(uuids) == len(set(uuids)) == 4, 'Four physical device UUIDs required')
    require(observations.get('schema') == Q_SCHEMA and isinstance(observations.get('observations'), list), 'GPU observation schema differs')
    require(isinstance(process.get('all_children'), list) and isinstance(process.get('streams'), list), 'Process ownership inventory required')
    expected_streams = {s['id'] for s in release['streams']}
    require(len(process['streams']) == len(expected_streams) and {s.get('id') for s in process['streams']} == expected_streams, 'Process stream inventory differs')
    for stream in process['streams']:
        outcomes = stream.get('outcomes')
        require(isinstance(outcomes, list) and len({o.get('stage') for o in outcomes}) == len(outcomes)
                and all(o.get('stage') in {s[0] for s in STAGES} for o in outcomes), 'Duplicate/unknown process outcome')
        require(all(o.get('state') in ('exited', 'quota_expired', 'aborted_by_supervisor')
                    and type(o.get('exit_code')) is int for o in outcomes), 'Owned evaluator must have a reaped exit code')
    historical = {}
    for child in process['all_children']:
        matches = [(s, n) for s in release['streams'] for n, command in enumerate(s['commands']) if child.get('command') == command]
        require(len(matches) == 1, 'Child command is not uniquely released')
        stream, index = matches[0]
        require(child.get('stage') == STAGES[index][0] and child.get('output_dir') == quota.flags(child['command'])['--output-dir'], 'Child stage/output differs')
        ident = child.get('identity')
        if ident is None: continue  # Failed identity capture remains an unsuccessful outcome.
        require(type(ident.get('pid')) is int and ident['pid'] > 0 and ident.get('argv') == child['command'], 'Owned child identity differs')
        historical.setdefault(ident['pid'], []).append((stream['gpu'], ident))
    counts = {'passed': 0, 'pending': 0}
    for observation in observations['observations']:
        validation = observation.get('validation')
        require(validation in counts and isinstance(observation.get('raw_gpu_processes'), list), 'Malformed retained GPU observation')
        counts[validation] += 1
        if validation == 'pending': continue  # Preserve rejected raw inventory without laundering it as passed.
        require(observation.get('gpu_scope') == scope, 'Passed observation scope differs')
        annotated = observation.get('all_gpu_processes')
        require(isinstance(annotated, list) and len(annotated) == len(observation['raw_gpu_processes']), 'Passed GPU annotation length differs')
        for row, observed in zip(observation['raw_gpu_processes'], annotated):
            pid = row.get('pid'); uuid = canonical_uuid(row.get('gpu_uuid'))
            require(type(pid) is int and pid > 0 and uuid in uuids, 'Passed GPU identity differs')
            index = uuids.index(uuid); check = observed.get('identity_check'); current = observed.get('current_identity')
            extra = {'physical_index', 'scope', 'identity_check', 'current_identity'}
            require({k:v for k,v in observed.items() if k not in extra} == row
                    and observed.get('physical_index') == index
                    and observed.get('scope') == ('owned_device' if index in scope['owned_indices'] else 'observed_unassigned_device'), 'Passed GPU annotation differs')
            known = historical.get(pid, []); keys = ('pid','ppid','start_ticks','argv','executable')
            matching = [(gpu,ident) for gpu,ident in known if isinstance(current,dict) and all(k in ident and current.get(k) == ident[k] for k in keys)]
            if check == 'matched_owned_child':
                require(index in scope['owned_indices'] and any(gpu == index for gpu,ident in matching), 'Matched owned identity/device differs')
            elif check == 'exited_since_inventory':
                require(current is None and index in scope['owned_indices'] and any(gpu == index for gpu,ident in known), 'Exited owned snapshot differs')
            elif check == 'different_process_identity':
                require(index not in scope['owned_indices'] and known and isinstance(current,dict) and not matching
                        and current.get('pid') == pid and all(k in current for k in keys), 'Reused identity must remain unassigned/uncontrolled')
            elif check == 'not_owned':
                require(index not in scope['owned_indices'] and current is None, 'Foreign work on assigned GPU')
            else: raise ValueError('Unknown passed GPU identity annotation')
    return counts


def account_snapshot_stage(stage, directory, outcome, snapshot, root):
    """Exact v3 coverage arithmetic on initial immutable JSON snapshots."""
    wanted = expected(stage); rows, current = {}, None
    relative = directory.relative_to(root)
    pattern = 'trajectory_*_*.json' if stage == 'full_rollout_test' else 'trajectory_*_target_*.json'
    paths = [root / name for name in snapshot['files'] if Path(name).parent == relative and Path(name).match(pattern)]
    def saved(path): return json_snapshot(path, snapshot['files'][str(path.relative_to(root))]['sha256'])[0]
    for path in paths:
        row = saved(path); key = identity(row, stage)
        require(key in wanted and key not in rows and row.get('status') in ('complete', 'failed'), 'Unexpected/duplicate saved evaluation row')
        if stage == 'full_rollout_test' and row['status'] == 'complete':
            require(row.get('completed_steps') == row.get('horizon') == 314, 'Completed rollout must cover declared full horizon')
        rows[key] = {'state': 'completed_required_outcome' if row['status'] == 'complete' else 'recorded_failed_outcome',
                     'path': str(path), 'sha256': snapshot['files'][str(path.relative_to(root))]['sha256'],
                     'worker_status': row['status'], 'failure': row.get('failure')}
    status_path = directory / 'status.json'
    if str(status_path.relative_to(root)) in snapshot['files']:
        status = saved(status_path); candidate = status.get('case' if stage == 'full_rollout_test' else 'current')
        if candidate is not None: current = identity(candidate, stage)
    if outcome.get('current_before_stop') is not None: current = identity(outcome['current_before_stop'], stage)
    cells = []
    for source, target in wanted:
        value = rows.get((source, target))
        if value is None:
            state = 'timed_out_current' if outcome.get('quota_expired') and (source, target) == current else 'not_completed_before_invocation_end'
            if outcome.get('state') == 'never_started': state = 'never_started'
            value = {'state': state, 'missing': True}
        cells.append({'source_index': source, 'policy' if stage == 'full_rollout_test' else 'target_frame': target, **value})
    return {'cells': cells}


def collect(queue_root, cohort_path, amendment_path, root_release_path):
    bindings = source_bindings(); root = Path(queue_root).resolve(); snapshot = tree_snapshot(root)
    approval, approval_sha = load_root(root_release_path, COLLECTION_RELEASE_SCHEMA, 'approved_for_stopped_scalar_collection', bindings)
    original_root = Path(approval.get('original_queue_root', ''))
    require(original_root.is_absolute() and '..' not in original_root.parts
            and approval.get('local_queue_inventory_sha256') == inventory_sha256(snapshot), 'Original queue path and exact local copy inventory required')
    cohort, cohort_sha = json_snapshot(cohort_path, approval.get('cohort_sha256'))
    amendment_sha = sha(amendment_path)
    require(amendment_sha == AMENDMENT_SHA, 'Frozen operational amendment differs')
    bindings.update({str(Path(cohort_path).resolve()): cohort_sha, str(Path(amendment_path).resolve()): amendment_sha})
    def saved_sha(path):
        name = str(Path(path).relative_to(root)); require(name in snapshot['files'], 'Output absent from initial snapshot')
        return snapshot['files'][name]['sha256']
    def saved_json(path): return json_snapshot(path, saved_sha(path))[0]
    quota = import_pinned('supervise_goop_evaluation_gpu_scoped_v3.py', '_scoped_sand_pure_argv')
    evaluator = import_pinned('evaluate_sand_graph_support_final.py', '_scoped_sand_evaluator')
    release, status, ledger, process = [saved_json(root / name) for name in
        ('release_snapshot.json', 'queue_status.json', 'coverage_ledger.json', 'process_outcomes.json')]
    observation_file_state = 'present' if 'gpu_observations.json' in snapshot['files'] else 'absent_before_any_child'
    if observation_file_state == 'present':
        observations = saved_json(root / 'gpu_observations.json')
        require(process.get('gpu_observations_sha256') == saved_sha(root / 'gpu_observations.json'), 'GPU observation byte hash differs')
    else:
        require(process.get('gpu_observations_sha256') is None and process.get('all_children') == []
                and all(s.get('outcomes') == [] for s in process.get('streams', []))
                and isinstance(process.get('abort_reason'), str) and process['abort_reason'], 'Missing GPU inventory only admissible for pre-child abort')
        observations = {'schema': Q_SCHEMA, 'observations': [], 'observation_file_state': observation_file_state}
    require(release.get('schema') == Q_RELEASE_SCHEMA and release.get('issued_by') == 'root'
            and release.get('status') == 'admitted_for_execution_allocation' and release.get('dataset') == ledger.get('dataset') == 'Sand'
            and release.get('host_role') in ('A', 'B'), 'Root scoped Sand queue release required')
    require(approval.get('queue_release_sha256') == saved_sha(root / 'release_snapshot.json')
            and approval.get('operational_amendment_sha256') == amendment_sha
            and release.get('operational_amendment', {}).get('sha256') == amendment_sha
            and release.get('files_sha256', {}).get(release['operational_amendment']['path']) == amendment_sha
            and SOURCE_PINS['supervise_sand_evaluation_gpu_scoped_v1.py'] in release['files_sha256'].values(), 'Root-bound queue/amendment/supervisor differs')
    require(status.get('schema') == ledger.get('schema') == process.get('schema') == Q_SCHEMA
            and status.get('state') in ('allocation_finished', 'stopped_requires_review')
            and status.get('coverage_ledger_sha256') == saved_sha(root / 'coverage_ledger.json')
            and status.get('unreaped_owned_children') == ledger.get('unreaped_owned_children') == process.get('unreaped_owned_children') == [],
            'Only stopped/reaped bound scoped queues may be collected')
    require(cohort.get('schema') == 'adaptgns_sand_graph_support_final_cohort_v1' and cohort.get('status') == 'frozen_for_final_evaluation'
            and cohort.get('protocol_sha256') == PROTOCOL_SHA and cohort.get('updates') == 100000, 'Frozen exact100k Sand cohort required')
    require(len(cohort.get('models', [])) == 6 and {(r['arm'], r['seed']) for r in cohort['models']} == {(a,s) for a in ('base','mix') for s in range(3)}, 'Complete cohort grid required')
    role = release['host_role']; stream_ids = [f'{a}_seed{s}' for a, s, _ in SCHEDULE[role]]
    require(len(release.get('streams', [])) == len(stream_ids) and {r.get('id') for r in release['streams']} == set(stream_ids), 'Queue streams differ')
    observation_counts = validate_scope(release, process, observations, quota)
    entries = ledger.get('stages', [])
    require(len(entries) == 4 * len(stream_ids) and len({(e.get('stream'), e.get('stage')) for e in entries}) == len(entries), 'Duplicate/missing coverage stage')
    collected, files = [], {}
    for stream in release['streams']:
        arm, seed = stream['arm'], stream['seed']
        require(type(seed) is int, 'Integer training seed required')
        model = next(r for r in cohort['models'] if (r['arm'], r['seed']) == (arm, seed))
        require(stream['id'] == f'{arm}_seed{seed}' and len(stream['commands']) == 4, 'Invalid stream identity/commands')
        for command, (stage, mode, split) in zip(stream['commands'], STAGES):
            options = quota.flags(command)
            expected_outcome = next((o for o in next(s for s in process['streams'] if s['id'] == stream['id'])['outcomes'] if o['stage'] == stage), {'stage': stage, 'state': 'never_started'})
            require(options.get('--mode') == mode and options.get('--split') == split and options.get('--arm') == arm
                    and options.get('--seed') == str(seed) and options.get('--checkpoint-sha256') == model['checkpoint_sha256'], 'Released model/stage differs')
            required_inputs = {'--cohort': cohort_sha, '--protocol': PROTOCOL_SHA, '--trainer-source': TRAINER_SHA,
                               '--benchmark-helper': BENCH_SHA, '--checkpoint': model['checkpoint_sha256']}
            require(all(release['files_sha256'].get(options.get(k)) == v for k, v in required_inputs.items())
                    and release['files_sha256'].get(command[1]) == EVALUATOR_SHA, 'Released final input hash differs')
            directory = root / 'jobs' / stream['id'] / stage
            require(options.get('--output-dir') == str(original_root / 'jobs' / stream['id'] / stage), 'Released original output directory differs')
            entry = next(e for e in entries if (e['stream'], e['stage']) == (stream['id'], stage))
            require(entry['outcome'] == expected_outcome, 'Coverage/process outcome differs')
            require(entry.get('coverage_audit_state') == 'row_coverage_checked', 'Coverage audit failed/deferred; inspect original receipts')
            recomputed = account_snapshot_stage(stage, directory, entry['outcome'], snapshot, root)
            cells = entry['cells']; wanted = expected(stage)
            require(len(cells) == len(wanted) and [identity(c, stage) for c in cells] == wanted
                    and all(c.get('state') in C_STATES for c in cells), 'Coverage grid differs')
            require(all('path' not in c or Path(c['path']) == original_root / 'jobs' / stream['id'] / stage / Path(c['path']).name for c in cells), 'Original coverage path escapes released stage')
            def normalized(cell): return {k: (Path(v).name if k == 'path' else v) for k, v in cell.items()}
            require([normalized(c) for c in cells] == [normalized(c) for c in recomputed['cells']], 'Fresh committed row coverage differs from ledger')
            rows = []
            protocol_path = directory / 'protocol.json'
            protocol = saved_json(protocol_path) if protocol_path.exists() else None
            if protocol is not None:
                require(protocol.get('schema') == E_SCHEMA and protocol.get('mode') == mode and protocol.get('split') == split
                        and protocol.get('source_frame_count') == 320 and protocol.get('model', {}).get('kind') == 'preselected_checkpoint'
                        and protocol['model'].get('completed_updates') == 100000 and protocol['model'].get('arm') == arm
                        and protocol['model'].get('seed') == seed and protocol['model'].get('checkpoint_sha256') == model['checkpoint_sha256'], 'Saved evaluation protocol/model differs')
                require(all(protocol.get('input_files_sha256', {}).get(options[k]) == v for k, v in required_inputs.items()), 'Saved evaluation input hashes differ')
                plan = [(r['source_index'], r['target_frame']) for r in protocol['schedule']] if mode != 'full-rollout' else [r['source_index'] for r in protocol['schedule']]
                require(plan == (wanted if mode != 'full-rollout' else list(range(30))), 'Fixed diagnostic/trajectory schedule differs')
                require(protocol.get('policies') == list(POLICIES if mode == 'full-rollout' else DIAG_POLICIES), 'Fixed policies differ')
            for cell in cells:
                if cell['state'] not in ('completed_required_outcome', 'recorded_failed_outcome'): continue
                require(protocol is not None, 'Committed row requires bound evaluation protocol')
                path = directory / Path(cell['path']).name
                require(saved_sha(path) == cell['sha256'], 'Saved row missing/changed')
                row = saved_json(path)
                require(row.get('protocol_sha256') == saved_sha(protocol_path), 'Row protocol hash differs')
                artifact_key = 'trace_file' if mode == 'full-rollout' else 'artifact_file'
                require(isinstance(row.get(artifact_key), str) and row[artifact_key] == Path(row[artifact_key]).name, 'Artifact must be a local basename')
                artifact = directory / row[artifact_key]
                require(artifact.parent == directory and artifact.is_file() and not artifact.is_symlink()
                        and saved_sha(artifact) == row['trace_sha256' if mode == 'full-rollout' else 'artifact_sha256'], 'Referenced numeric artifact bytes differ')
                if mode == 'full-rollout': validate_rollout(row, arm, seed)
                rows.append(row)
            aggregate_snapshot = {'state': 'absent', 'recorded_rows': 0, 'committed_rows': len(rows)}
            if mode == 'full-rollout' and (directory / 'result.json').exists():
                result = saved_json(directory / 'result.json')
                require(result.get('schema') == E_SCHEMA and result.get('protocol_sha256') == saved_sha(protocol_path)
                        and isinstance(result.get('rows'), list) and len(result['rows']) <= len(rows)
                        and result['rows'] == rows[:len(result['rows'])], 'Result snapshot is not an exact committed prefix')
                aggregate_snapshot.update(state='current' if len(result['rows']) == len(rows) else 'stale_valid_prefix', recorded_rows=len(result['rows']))
            diag = None
            if mode != 'full-rollout':
                schedule = [{'source_index': i, 'target_frame': t} for i, t in wanted]
                diag = evaluator.summarize(schedule, rows, mode)
                if (directory / 'summary.json').exists():
                    saved = saved_json(directory / 'summary.json'); count = saved.get('returned_frames')
                    require(type(count) is int and 0 <= count <= len(rows)
                            and saved == evaluator.summarize(schedule, rows[:count], mode), 'Stored diagnostic summary is not an exact committed prefix')
                    aggregate_snapshot.update(state='current' if count == len(rows) else 'stale_valid_prefix', recorded_rows=count)
            collected.append({'arm': arm, 'seed': seed, 'stage': stage, 'mode': mode, 'split': split,
                'cells': cells, 'rows': rows, 'diagnostic_summary': diag, 'outcome': entry['outcome'],
                'protocol_sha256': saved_sha(protocol_path) if protocol_path.exists() else None, 'aggregate_snapshot': aggregate_snapshot})
    result = {'schema': COLLECTION_SCHEMA, 'status': 'stopped_outputs_collected', 'issued_by': 'root', 'host_role': role,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'collector_sha256': sha(__file__), 'cohort_sha256': cohort_sha,
        'protocol_sha256': PROTOCOL_SHA, 'source_sha256': SOURCE_PINS, 'stages': collected, 'files': snapshot['files'],
        'queue_root': str(root), 'original_queue_root': str(original_root),
        'local_queue_inventory_sha256': inventory_sha256(snapshot), 'output_tree_state': snapshot, 'input_sha256': bindings, 'collection_release_sha256': approval_sha,
        'queue_release_sha256': saved_sha(root / 'release_snapshot.json'), 'queue_release': release,
        'queue_status': status, 'process_outcomes': process, 'gpu_observations': observations,
        'gpu_observation_counts': observation_counts, 'gpu_observation_file_state': observation_file_state, 'operational_amendment_sha256': amendment_sha,
        'coverage_audit_errors': ledger.get('audit_errors'), 'abort_reason': ledger.get('abort_reason'),
        'timing_scope': 'shared_host_operational_measurement',
        'scientific_arithmetic': 'exact-hash quota-v2 scalar primitives; explicit Sand H314 and fixed Sand grids; no old outcome recomputation',
        'scope': 'committed row identity/arithmetic, complete coverage, diagnostic aggregation and numeric artifact byte hashes; no fresh source/array recomputation'}
    verify_collection_snapshot(result)
    return result


def verify_collection_snapshot(collection):
    verify_bindings(collection['input_sha256'])
    require(tree_snapshot(collection['queue_root']) == collection['output_tree_state'], 'Stopped queue entry/hash/size snapshot changed')


def summarize(collection_paths, root_release_path):
    bindings = source_bindings()
    approval, approval_sha = load_root(root_release_path, ANALYSIS_RELEASE_SCHEMA, 'approved_for_fixed_scalar_aggregation', bindings)
    collections = []
    for role in ('A', 'B'):
        path = Path(collection_paths[role]).resolve()
        value, digest = json_snapshot(path, approval.get('collection_sha256', {}).get(role)); bindings[str(path)] = digest
        require(value.get('schema') == COLLECTION_SCHEMA and value.get('status') == 'stopped_outputs_collected'
                and value.get('collector_sha256') == sha(__file__) and value.get('source_sha256') == SOURCE_PINS
                and value.get('host_role') == role and value.get('cohort_sha256') == approval.get('cohort_sha256')
                and value.get('protocol_sha256') == PROTOCOL_SHA and value.get('timing_scope') == 'shared_host_operational_measurement', 'Root-bound scoped collection differs')
        require(value.get('queue_status', {}).get('all_pinned_inputs_reverified') is True,
                'Input integrity was not reverified; preserved collection cannot enter scientific aggregation')
        collections.append(value)
    stages = [s for c in collections for s in c['stages']]
    grid = {(a,s,n) for a in ('base','mix') for s in range(3) for n,_,_ in STAGES}
    require(len(stages) == 24 and {(s['arm'],s['seed'],s['stage']) for s in stages} == grid, 'All24 model-stage receipts required, including missing work')
    result = {'schema': SCHEMA, 'status': 'fixed_scalar_aggregation_complete', 'cohort_sha256': approval['cohort_sha256'],
        'protocol_sha256': PROTOCOL_SHA, 'summarizer_sha256': sha(__file__), 'source_sha256': SOURCE_PINS,
        'analysis_release_sha256': approval_sha, 'collection_sha256': approval['collection_sha256'], 'input_sha256': bindings,
        'full_rollout': full_summary([s for s in stages if s['stage'] == 'full_rollout_test']),
        'diagnostics': {name: diagnostic_summary(stages, name) for name,_,_ in STAGES[1:]},
        'queue_accounting': {c['host_role']: {k: c[k] for k in ('queue_status', 'queue_release_sha256', 'operational_amendment_sha256',
            'abort_reason', 'coverage_audit_errors', 'gpu_observations', 'gpu_observation_counts', 'gpu_observation_file_state', 'process_outcomes')} for c in collections},
        'timing_scope': 'shared_host_operational_measurement; fixed policy order and concurrent unrelated work do not establish dedicated-host or causal speedup',
        'scientific_arithmetic': 'exact-hash quota-v2 scalar primitives plus explicit Sand H314 validator/aggregation; unchanged equal-source then three-seed means/sampleSD and paired contrasts',
        'scope': 'all three training seeds; null when required values missing/failed; no survivor means, p-values, replacement seeds or shortened-horizon substitution',
        'numeric_artifact_validation': 'referenced binary hashes verified during collection; scalar arithmetic and saved diagnostic aggregation recomputed; independent per-array/source recomputation remains separate',
        'interpretation': 'exploratory native-CUDA Sand graph-support extension informed by prior results; not pristine independent confirmation; negative interaction alone does not establish risk superiority'}
    verify_bindings(bindings)
    return result


def encode(value): return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def publish(path, value):
    raw = encode(value)  # Serialization is inside the checked interval.
    if value['schema'] == COLLECTION_SCHEMA: verify_collection_snapshot(value)
    else: verify_bindings(value['input_sha256'])
    with Path(path).open('xb') as stream: stream.write(raw)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true'); p.add_argument('--mode', choices=('collect','summarize'))
    for name in ('queue-root','cohort','operational-amendment','collection-a','collection-b','root-release','output'): p.add_argument('--' + name, type=Path)
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'status': 'description_only', 'schema': SCHEMA, 'required_rollout_outcomes': 1080,
            'required_model_stages': 24, 'source_sha256': SOURCE_PINS, 'scientific_arithmetic': 'frozen quota-v2 primitives plus explicit Sand H314 functions'}, indent=2)); return 0
    require(a.output is not None and not a.output.exists() and a.root_release is not None, 'Fresh output and root release required')
    if a.mode == 'collect':
        require(a.queue_root and a.cohort and a.operational_amendment and a.queue_root.resolve() not in a.output.resolve().parents,
                'Queue/cohort/amendment and separate output required')
        result = collect(a.queue_root, a.cohort, a.operational_amendment, a.root_release)
    elif a.mode == 'summarize':
        require(a.collection_a and a.collection_b, 'Both stopped collections required')
        result = summarize({'A': a.collection_a, 'B': a.collection_b}, a.root_release)
    else: p.error('Explicit mode required')
    publish(a.output, result); return 0


if __name__ == '__main__': raise SystemExit(main())
