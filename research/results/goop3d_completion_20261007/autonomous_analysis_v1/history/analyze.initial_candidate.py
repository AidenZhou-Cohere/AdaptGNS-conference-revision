#!/usr/bin/env python3
"""Incremental D3 H295 completion collection, saved-array audit and paired analysis.

No model, source trajectory, CUDA, spatial search or old evaluator CLI is run.
A snapshot retains every original cell/outcome; only final worker commit markers
admit new rows. Audit checkpoints survive explicit --resume and skip completed
checks. All 2160 declared cells stay in denominators even before completion.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PLAN_SHA = '838e7d4c1afc22fc4757a56ec0c755d5d303902d0ed600767d95c507d0402e08'
WORKER_SHA = 'a018ed62c6a51511347194d36341c690faa3d8893552f46a7a825b7b90195034'
WORKER_SCHEMA = 'goop3d_missing_autonomous_worker_v1'
LEDGER_SHA = 'ef0673b69180c0c648d7220a300ac526c115482b0aa26ddab39a2ddd9146939b'
METADATA_SHA = '727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'
SCIENCE_PINS = {
    'audit_goop3d_saved_arrays_v1.py': '633d7034d687acdc4e743e78c67fb5bcf1c81352ed524829d5ba01b163158491',
    'goop3d_saved_diagnostic_audit_v1.py': 'd772313e7a475897cc2dc7d5a893d10c89a8cad67fee9c9b5fe2fa3683006c2a',
    'audit_goop3d_paired_arrays_v1.py': 'a520fdf0766038b99ad075b27cbc227721745bce9c5cd5e0cb1416535f6897f9',
    'summarize_goop3d_graph_support_v1.py': 'd7096bef018ac7e3d28cc429a4711d88209e48a1b812dda0aad995d6a41de5fd',
}
SCIENCE = None


def need(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def strict(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def bad(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def atomic(path, value):
    path = Path(path)
    need(not path.is_symlink(), 'refuse output symlink')
    raw = encode(value)
    fd, name = tempfile.mkstemp(prefix=path.name + '.pending.', dir=path.parent)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return digest(raw)


def regular(path):
    path = Path(path)
    need(path.is_file() and all(not p.is_symlink() for p in (path, *path.parents)), 'regular canonical input required: ' + str(path))
    return path


def read_bound(path, pin, limit=64 << 20):
    path = regular(path)
    need(path.stat().st_size <= limit, 'input exceeds explicit byte bound: ' + str(path))
    raw = path.read_bytes()
    need(digest(raw) == pin, 'input hash differs: ' + str(path))
    return strict(raw)


def relative_file(root, name):
    relative = Path(name)
    path = Path(root) / relative
    need(not relative.is_absolute() and '..' not in relative.parts and path.resolve().is_relative_to(Path(root).resolve()), 'input escapes bound root')
    return regular(path)


def cell_id(arm, seed, split, cell):
    return f'{arm}_seed{seed}__{split}__{cell["source_index"]:06d}__{cell["policy"]}'


def schedule(plan, split):
    return [{'source_index': r['source_index'], 'trajectory_id': r['id'],
             'particles': r['positions']['shape'][1], 'size_group': 'fixed_source_grid'}
            for r in plan['splits'][split]['records']]


def load_science(frozen):
    global SCIENCE
    frozen = Path(frozen).resolve()
    for name, pin in SCIENCE_PINS.items():
        need(sha(frozen / name) == pin, 'frozen arithmetic source differs: ' + name)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(frozen))
    modules = tuple(importlib.import_module(name) for name in (
        'audit_goop3d_saved_arrays_v1', 'summarize_goop3d_graph_support_v1', 'audit_goop3d_paired_arrays_v1'))
    for module in (*modules, modules[0].diagnostic):
        need(Path(module.__file__).resolve().parent == frozen, 'unexpected arithmetic module path')
    SCIENCE = modules
    return modules


class Checkpoints:
    """Commit audited rows by a durable hash index; preserve all failed attempts."""
    def __init__(self, output, identity, resume):
        self.output, self.identity = Path(output), identity
        self.rows = self.output / 'rows'
        self.rows.mkdir(exist_ok=True)
        self.path = self.output / 'checkpoint_index.json'
        if self.path.exists():
            need(resume, 'existing checkpoints require --resume')
            self.index = strict(regular(self.path).read_bytes())
            need(self.index['schema'] == 'goop3d_autonomous_audit_checkpoints_v1' and self.index['identity'] == identity,
                 'checkpoint source/input/runtime identity differs')
        else:
            self.index = {'schema': 'goop3d_autonomous_audit_checkpoints_v1', 'identity': identity,
                          'completed': {}, 'failed': {}}
            atomic(self.path, self.index)

    def read(self, key, task_sha):
        entry = self.index['completed'].get(key)
        if entry is None:
            return None
        need(entry['task_sha256'] == task_sha, 'audited cell binding changed')
        record = read_bound(self.rows / (key + '.json'), entry['sha256'])
        need(record['key'] == key and record['task_sha256'] == task_sha and
             record['identity_sha256'] == digest(encode(self.identity)), 'audit checkpoint identity differs')
        return record['result']

    def commit(self, key, task_sha, result):
        need(key not in self.index['completed'], 'duplicate audited cell')
        record = {'key': key, 'task_sha256': task_sha, 'identity_sha256': digest(encode(self.identity)), 'result': result}
        path = self.rows / (key + '.json')
        if path.exists():
            need(regular(path).read_bytes() == encode(record), 'unindexed audit row differs; retained for review')
            pin = sha(path)
        else:
            pin = atomic(path, record)
        self.index = {**self.index,
                      'completed': {**self.index['completed'], key: {'sha256': pin, 'task_sha256': task_sha}},
                      'failed': {k: v for k, v in self.index['failed'].items() if k != key}}
        atomic(self.path, self.index)

    def failure(self, key, task_sha, error, attempt):
        record = {'key': key, 'task_sha256': task_sha, 'error_type': type(error).__name__,
                  'error': str(error), 'traceback': ''.join(traceback.format_exception(error)), 'scientific_admission': False}
        path = attempt / (key + '.failure.json')
        pin = atomic(path, record)
        self.index = {**self.index, 'failed': {**self.index['failed'], key: {
            'path': str(path.relative_to(self.output)), 'sha256': pin, 'task_sha256': task_sha}}}
        atomic(self.path, self.index)


def collect(args, plan, roots):
    """Read scalar metadata; every NPZ byte is verified in its parallel audit task."""
    ledger = read_bound(args.ledger, LEDGER_SHA)
    metadata = read_bound(args.metadata, METADATA_SHA)
    need(ledger['schema'] == 'adaptgns_goop3d_final_evaluation_ledger_v1'
         and ledger['state'] == 'stopped_all_owned_processes_reaped'
         and ledger['all_pinned_inputs_reverified'] is True and ledger['unreaped_owned_children'] == [], 'original stopped ledger identity')
    original_stages = [s for s in ledger['stages'] if s['mode'] == 'full-rollout']
    need(len(original_stages) == 12, 'original twelve autonomous stages required')
    originals = {}
    for stage in original_stages:
        for c in stage['cells']:
            key = cell_id(stage['arm'], stage['seed'], stage['split'], c)
            need(key not in originals, 'duplicate original cell')
            originals[key] = (stage, c)
    wanted = {c['cell_id']: c for c in plan['retained_original_outcomes']}
    missing = {c['cell_id']: (w, c) for w in plan['workers'] for c in w['cells']}
    need(len(originals) == 2160 and len(wanted) == 331 and len(missing) == 1829
         and not wanted.keys() & missing.keys() and originals.keys() == wanted.keys() | missing.keys(), 'full original/missing grid differs')
    for key, (stage, c) in originals.items():
        planned = wanted[key] if key in wanted else missing[key][1]
        need(c['state'] == planned['state'] and all(c[k] == planned[k] for k in (
            'source_index', 'trajectory_id', 'particles', 'size_group', 'policy')), 'original fixed cell identity differs')
        if key in wanted:
            need(c['state'] == 'completed_required_outcome' and c['row_sha256'] == planned['row_sha256']
                 and c['artifact_sha256'] == planned['artifact_sha256'], 'retained original outcome pin differs')
    registry = {str(args.ledger): LEDGER_SHA, str(args.metadata): METADATA_SHA, str(args.plan): PLAN_SHA}
    rows, tasks, worker_snapshots = {}, {}, []
    def bound_json(path, pin):
        doc = read_bound(path, pin)
        registry[str(path)] = pin
        return doc
    def add(key, cell, row, row_path, row_pin, artifact, artifact_pin, protocol_pin, origin):
        item = {k: cell[k] for k in ('source_index', 'trajectory_id', 'particles', 'size_group', 'policy')}
        need(all(row.get(k) == v for k, v in item.items()) and row.get('arm') == cell['arm']
             and row.get('training_seed') == cell['seed'] and row.get('objective') == 'faithful'
             and row.get('horizon') == 295 and row.get('rng_seed') == 93000 + 1000 * cell['seed'] + cell['source_index']
             and row.get('protocol_sha256') == protocol_pin and row.get('artifact_sha256') == artifact_pin
             and row.get('artifact_file') == artifact.name and row.get('status') in ('complete', 'failed'), 'committed row identity')
        need(key not in rows, 'duplicate admitted cell')
        regular(artifact)
        registry[str(artifact)] = artifact_pin
        rows[key] = row
        task = {'key': key, 'arm': cell['arm'], 'seed': cell['seed'], 'split': cell['split'],
                'item': item, 'row_file': str(row_path), 'row_sha256': row_pin,
                'artifact_file': str(artifact), 'artifact_sha256': artifact_pin,
                'artifact_bytes': artifact.stat().st_size, 'protocol_sha256': protocol_pin,
                'bounds': metadata['bounds'], 'origin': origin}
        tasks[key] = (digest(encode(task)), task)
    original_queue = Path(plan['original_remote_repo']) / 'goop3d_final_evaluation_20261006_v1'
    for key, cell in wanted.items():
        old_directory = Path(cell['original_directory'])
        directory = args.original_queue_root / old_directory.relative_to(original_queue)
        row_path = relative_file(directory, cell['row_file'])
        row = bound_json(row_path, cell['row_sha256'])
        protocol_path = directory / 'protocol.json'
        protocol = bound_json(protocol_path, row['protocol_sha256'])
        need((protocol['arm'], protocol['seed'], protocol['split'], protocol['mode'], protocol['horizon'], protocol['checkpoint_updates'])
             == (cell['arm'], cell['seed'], cell['split'], 'full-rollout', 295, 25000), 'original row protocol identity')
        artifact = relative_file(directory, row['artifact_file'])
        add(key, cell, row, row_path, cell['row_sha256'], artifact, cell['artifact_sha256'], row['protocol_sha256'], 'retained_original')
    input_pins = {r['path']: r for r in plan['files']}
    for worker in plan['workers']:
        root = roots[worker['worker_index']]
        expected = {c['cell_id']: c for c in worker['cells']}
        identity_path = root / 'worker_identity.json'
        if not identity_path.exists():
            need(not (root / 'cells').exists(), 'worker cell files without identity')
            worker_snapshots.append({'worker_index': worker['worker_index'], 'path': str(root), 'identity_available': False})
            continue
        identity_pin = sha(regular(identity_path))
        identity = bound_json(identity_path, identity_pin)
        need(identity['schema'] == WORKER_SCHEMA + '_identity' and identity['plan_sha256'] == PLAN_SHA
             and identity['worker_source_sha256'] == WORKER_SHA and identity['worker_index'] == worker['worker_index']
             and (identity['arm'], identity['seed']) == (worker['arm'], worker['seed'])
             and identity['cell_ids'] == [c['cell_id'] for c in worker['cells']], 'worker identity differs')
        for original, entry in identity['input_files'].items():
            need(original in input_pins and entry['sha256'] == input_pins[original]['sha256']
                 and entry['original_path'] == original, 'worker numerical input pin differs')
        checkpoint = next(m['checkpoint_path'] for m in plan['models'] if (m['arm'], m['seed']) == (worker['arm'], worker['seed']))
        need(checkpoint in identity['input_files'], 'worker exact model pin missing')
        snapshot = {'worker_index': worker['worker_index'], 'path': str(root), 'identity_sha256': identity_pin,
                    'identity': identity, 'committed_cell_ids': []}
        worker_snapshots.append(snapshot)
        protocol_path = root / 'protocol.json'
        if not protocol_path.exists():
            need(not (root / 'cells').exists(), 'worker cells without protocol')
            continue
        protocol_pin = sha(regular(protocol_path))
        protocol = bound_json(protocol_path, protocol_pin)
        need(protocol['schema'] == WORKER_SCHEMA and protocol['identity_sha256'] == identity_pin
             and protocol['worker_source_sha256'] == WORKER_SHA and protocol['plan_sha256'] == PLAN_SHA
             and protocol['cell_ids'] == identity['cell_ids'] and protocol['input_files'] == identity['input_files']
             and (protocol['arm'], protocol['seed'], protocol['horizon'], protocol['trace_steps'])
                 == (worker['arm'], worker['seed'], 295, [1, 10, 50, 200, 295]), 'worker protocol differs')
        snapshot['protocol_sha256'], snapshot['protocol'] = protocol_pin, protocol
        cellroot = root / 'cells'
        if not cellroot.exists():
            continue
        need(cellroot.is_dir() and not cellroot.is_symlink(), 'regular worker cells root required')
        for directory in sorted(cellroot.iterdir()):
            need(directory.is_dir() and not directory.is_symlink() and directory.name in expected, 'unexpected worker cell')
            marker_path = directory / 'commit.json'
            if not marker_path.exists():
                continue
            marker_pin = sha(regular(marker_path))
            marker = bound_json(marker_path, marker_pin)
            key, cell = directory.name, expected[directory.name]
            need(marker['schema'] == WORKER_SCHEMA + '_cell_commit' and marker['cell_id'] == key
                 and marker['plan_sha256'] == PLAN_SHA and marker['identity_sha256'] == identity_pin
                 and marker['protocol_sha256'] == protocol_pin, 'worker commit identity differs')
            row_path = relative_file(directory, marker['row_file'])
            artifact = relative_file(directory, marker['artifact_file'])
            row = bound_json(row_path, marker['row_sha256'])
            need(row['completion_cell_id'] == key and row['original_cell_state'] == cell['state']
                 and row['original_directory'] == cell['original_directory'], 'new row original provenance differs')
            add(key, cell, row, row_path, marker['row_sha256'], artifact, marker['artifact_sha256'], protocol_pin, 'completion_worker')
            snapshot['committed_cell_ids'].append(key)
    stages = []
    for original in original_stages:
        arm, seed, split = original['arm'], original['seed'], original['split']
        cells, stage_rows = [], []
        for c in original['cells']:
            key = cell_id(arm, seed, split, c)
            normalized = {**c, 'cell_id': key, 'original_state': c['state']}
            if key in rows:
                row = rows[key]
                normalized.update(state='completed_required_outcome' if row['status'] == 'complete' else 'recorded_failed_outcome',
                                  failure=row.get('failure'), origin=tasks[key][1]['origin'])
                stage_rows.append(row)
            else:
                normalized.update(state='not_completed_before_invocation_end',
                                  reason='no_final_completion_worker_commit_in_this_snapshot', origin='pending_completion')
            cells.append(normalized)
        stages.append({'arm': arm, 'seed': seed, 'split': split, 'stage': original['stage'], 'mode': 'full-rollout',
                       'cells': cells, 'rows': stage_rows, 'outcome': {
                           'scope': 'merged immutable original outcomes and committed completion-worker rows',
                           'original_invocation_outcome': original['outcome'], 'worker_invocations_separate': True}})
    need(sum(len(s['cells']) for s in stages) == 2160 and len(rows) >= 331, 'full merged denominator')
    return {'schema': 'goop3d_autonomous_completion_collection_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': PLAN_SHA, 'worker_source_sha256': WORKER_SHA, 'original_ledger_sha256': LEDGER_SHA,
            'required_outcomes': 2160, 'retained_original_outcomes': 331, 'new_committed_outcomes': len(rows) - 331,
            'still_missing_outcomes': 2160 - len(rows), 'cohort_sha256': ledger['cohort_sha256'],
            'source_manifest_sha256': ledger['source_manifest_sha256'], 'physical_reference': {
                'dataset': 'Goop-3D', 'dimension': 3, 'frames': 301, 'horizon': 295, 'particle_type': 7,
                'metadata_sha256': METADATA_SHA, 'metadata': metadata},
            'original_autonomous_invocations': original_stages, 'worker_snapshots': worker_snapshots,
            'stages': stages, 'verified_json_and_declared_artifact_sha256': registry}, tasks, registry


def audit_cell(task):
    saved, scalar, _ = SCIENCE
    checks = saved.Checks()
    checks.context = task['key']
    row = read_bound(task['row_file'], task['row_sha256'])
    artifact = regular(task['artifact_file'])
    need(artifact.stat().st_size == task['artifact_bytes'], 'artifact byte length changed')
    raw = artifact.read_bytes()
    need(digest(raw) == task['artifact_sha256'], 'artifact hash differs')
    with saved.np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        checks.equal(len(archive.files), len(set(archive.files)), 'unique saved archive members')
        arrays = {k: archive[k] for k in archive.files}
    checks.require(all(isinstance(v, saved.np.ndarray) and not v.dtype.hasobject for v in arrays.values()), 'numeric-only saved archive')
    expected, prefix = scalar.validate_rollout(row, task['arm'], task['seed'], task['item'])
    metrics, info = saved.audit_rollout(row, arrays, saved.np.asarray(task['bounds'], dtype=saved.np.float64), checks)
    checks.close(metrics, expected, 'independent array/scalar metrics')
    checks.close(info['accepted_prefix_boundary'], prefix if row['status'] == 'failed' else None, 'independent failure prefix')
    pairings = {f'{task["split"]}/initial/{task["item"]["source_index"]}': info['initial_history_sha256']}
    pairings.update({f'{task["split"]}/truth_float32/{task["item"]["source_index"]}/{int(step)+5}': pin
                    for step, pin in info['truth_hashes'].items()})
    need(sha(task['row_file']) == task['row_sha256'] and sha(artifact) == task['artifact_sha256'], 'input changed during saved audit')
    return {'key': task['key'], 'metrics': metrics, 'info': info, 'pairings': pairings,
            'status': row['status'], 'failure': row.get('failure'), 'checks': checks.count}


def merge(plan, collection, tasks, store):
    saved, scalar, paired = SCIENCE
    checks = saved.Checks()
    models, details, shared, summaries = [], [], {}, {}
    for stage in collection['stages']:
        arm, seed, split = stage['arm'], stage['seed'], stage['split']
        values = {}
        for cell in stage['cells']:
            key = cell['cell_id']
            if key not in tasks:
                continue
            result = store.read(key, tasks[key][0])
            need(result is not None, 'required available row not audited')
            checks.count += result['checks']
            checks.equal(result['key'], key, 'audited key')
            for label, pin in result['pairings'].items():
                if label in shared:
                    checks.equal(shared[label], pin, 'cross-model/policy same-source identity')
                shared[label] = pin
            values[cell['source_index'], cell['policy']] = result['metrics']
            details.append({'arm': arm, 'seed': seed, 'stage': stage['stage'], 'unit': [cell['source_index'], cell['policy']],
                            'status': result['status'], 'failure': result['failure'], **result['info']})
        means = {metric + '/' + policy: saved.mean([values.get((item['source_index'], policy), {}).get(metric)
                    for item in schedule(plan, split)]) for metric in saved.FULL_METRICS for policy in saved.POLICIES}
        models.append({'arm': arm, 'seed': seed, 'stage': stage['stage'], 'split': split, 'mode': 'full-rollout',
                       'metrics': means, 'cells': stage['cells'], 'coverage': dict(Counter(c['state'] for c in stage['cells']))})
    for split in ('valid', 'test'):
        selected = [s for s in collection['stages'] if s['split'] == split]
        result = scalar.full_summary(selected, schedule(plan, split))
        values = {(m['arm'], m['seed'], p): {k: m['metrics'][k + '/' + p] for k in saved.FULL_METRICS}
                  for m in models if m['split'] == split for p in saved.POLICIES}
        independent = paired.paired_metrics(values, saved.POLICIES, 'laggedrisk25')
        checks.close(result['paired'], independent, 'independent paired three-seed arithmetic/' + split)
        expected_prefixes = [{'arm': r['arm'], 'seed': r['seed'], 'source_index': r['unit'][0], 'policy': r['unit'][1],
                             'completed_steps': r['completed_steps'], 'failure': r['failure'], 'accepted_prefix': r['accepted_prefix_boundary']}
                            for r in details if r['stage'] == 'full_rollout_' + split and r['status'] == 'failed']
        checks.close(result['failed_accepted_prefixes'], expected_prefixes, 'complete failure-prefix accounting/' + split)
        summaries[split] = result
    return {'schema': 'goop3d_autonomous_completion_saved_audit_v1', 'status': 'passed_supported_checks',
            'checks': checks.count, 'required_outcomes': 2160, 'audited_outcomes': len(tasks),
            'models': models, 'row_checks': details, 'shared_saved_state_hashes': shared,
            'scope': {'arrays': 'Retained forecasts1/10/50/200/295 and saved overlapping histories/graph actions only.',
                      'scalar_series': 'Recorded complete/prefix MSE, physical-boundary and graph aggregates.',
                      'unsupported': ['Unsaved trajectory errors or full temporal replay.', 'Independent official source truth or model normalization.',
                                      'Fresh model/parity inference, spatial candidate completeness, hardware isolation or causal speedup.']}}, {
            'schema': 'goop3d_autonomous_completion_paired_summary_v1', 'endpoint_updates': 25000,
            'required_outcomes': 2160, 'retained_original_outcomes': 331, 'new_committed_outcomes': collection['new_committed_outcomes'],
            'still_missing_outcomes': collection['still_missing_outcomes'], 'stages': summaries,
            'all_recorded_guards_and_original_invocation_failures_retained': True,
            'survivor_means_computed': False, 'all_three_seed_denominators_required': True,
            'scope': 'Fixed source-grid, equal-source means within each model, then three paired training seeds; sample SD is not a confidence interval.'}


def event(progress, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False)
    progress.write(raw + '\n')
    progress.flush()
    print(raw, flush=True)


def execute(args):
    plan = read_bound(args.plan, PLAN_SHA)
    roots = {}
    for item in args.worker_root:
        index, name = item.split('=', 1)
        need(int(index) not in roots, 'duplicate worker root')
        roots[int(index)] = Path(name).resolve()
    need(set(roots) == set(range(12)), 'all twelve explicit worker roots required')
    saved, _, _ = load_science(args.frozen_dir)
    output = args.output.resolve()
    protected = [args.original_queue_root.resolve(), args.frozen_dir.resolve(), *roots.values()]
    need(all(output != p and not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected), 'output overlaps inputs')
    if args.resume:
        need(output.is_dir() and not output.is_symlink(), 'resume requires existing audit output')
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempts = output / 'attempts'
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f'{len(list(attempts.iterdir())) + 1:06d}'
    attempt.mkdir()
    progress = (attempt / 'progress.jsonl').open('x')
    started, source_sha = time.monotonic(), sha(__file__)
    try:
        identity = {'schema': 'goop3d_autonomous_completion_analysis_identity_v1', 'source_sha256': source_sha,
                    'plan_sha256': PLAN_SHA, 'worker_source_sha256': WORKER_SHA, 'original_ledger_sha256': LEDGER_SHA,
                    'metadata_sha256': METADATA_SHA, 'frozen_arithmetic_sha256': SCIENCE_PINS,
                    'original_queue_root': str(args.original_queue_root.resolve()), 'worker_roots': {str(k): str(v) for k, v in roots.items()},
                    'python': platform.python_version(), 'numpy': saved.np.__version__,
                    'python_executable_sha256': sha(Path(sys.executable).resolve())}
        store = Checkpoints(output, identity, args.resume)
        collection, tasks, registry = collect(args, plan, roots)
        expected = set(tasks)
        need(set(store.index['completed']) <= expected and set(store.index['failed']) <= expected,
             'previously observed committed rows disappeared')
        registry_path = output / 'input_registry.json'
        if registry_path.exists():
            prior = strict(regular(registry_path).read_bytes())
            need(prior.items() <= registry.items(), 'previously observed input binding changed or disappeared')
        atomic(registry_path, registry)
        collection_pin = atomic(attempt / 'collection.json', collection)
        pending = []
        for key, (pin, task) in tasks.items():
            if store.read(key, pin) is not None:
                continue
            if key in store.index['failed'] and not args.retry_failed:
                need(store.index['failed'][key]['task_sha256'] == pin, 'previous failed audit binding changed')
                continue
            pending.append((key, pin, task))
        event(progress, {'state': 'prepared', 'available_rows': len(tasks), 'missing_rows': 2160 - len(tasks),
                         'cached_rows': len(store.index['completed']), 'pending_audits': len(pending), 'workers': args.workers})
        if pending:
            iterator = iter(pending)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=load_science, initargs=(str(args.frozen_dir),)) as pool:
                inflight = {}
                def fill():
                    while len(inflight) < 2 * args.workers:
                        try:
                            key, pin, task = next(iterator)
                        except StopIteration:
                            return
                        inflight[pool.submit(audit_cell, task)] = (key, pin)
                fill()
                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for future in done:
                        key, pin = inflight.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:
                            store.failure(key, pin, error, attempt)
                            status = 'audit_failed_retained'
                        else:
                            store.commit(key, pin, result)
                            status = 'audit_checkpoint_committed'
                        event(progress, {'state': status, 'cell_id': key, 'completed': len(store.index['completed']),
                                         'failed': len(store.index['failed']), 'elapsed_seconds': round(time.monotonic() - started, 3)})
                    fill()
        need(not store.index['failed'] and set(store.index['completed']) == expected, 'available row audit incomplete; durable successes and all errors retained')
        audit, summary = merge(plan, collection, tasks, store)
        for path, pin in registry.items():
            need(sha(regular(path)) == pin, 'final saved input bytes changed: ' + path)
        for name, pin in SCIENCE_PINS.items():
            need(sha(args.frozen_dir / name) == pin, 'final arithmetic source changed')
        need(sha(__file__) == source_sha, 'analysis source changed')
        for product in (audit, summary):
            product.update(collection_sha256=collection_pin, plan_sha256=PLAN_SHA, analysis_source_sha256=source_sha,
                           cohort_sha256=collection['cohort_sha256'], physical_reference=collection['physical_reference'],
                           source_manifest_sha256=collection['source_manifest_sha256'])
        audit_pin = atomic(attempt / 'audit.json', audit)
        summary['audit_sha256'] = audit_pin
        summary_pin = atomic(attempt / 'summary.json', summary)
        receipt = {'schema': 'goop3d_autonomous_completion_analysis_receipt_v1',
                   'status': 'all2160_rows_audited_and_paired' if len(tasks) == 2160 else 'available_rows_audited_completion_pending',
                   'required_outcomes': 2160, 'audited_outcomes': len(tasks), 'missing_outcomes': 2160 - len(tasks),
                   'analysis_source_sha256': source_sha, 'frozen_arithmetic_sha256': SCIENCE_PINS,
                   'checkpoint_index_sha256': sha(store.path), 'input_registry_sha256': sha(registry_path),
                   'products_sha256': {'collection.json': collection_pin, 'audit.json': audit_pin, 'summary.json': summary_pin},
                   'attempt': attempt.name, 'elapsed_seconds': time.monotonic() - started,
                   'no_model_execution': True, 'arbitrary_elapsed_cutoff': False, 'scientific_admission_requires_review': True}
        receipt_pin = atomic(attempt / 'receipt.json', receipt)
        atomic(output / 'latest.json', {'attempt': attempt.name, 'receipt_sha256': receipt_pin, 'status': receipt['status']})
        event(progress, {'state': receipt['status'], 'receipt_sha256': receipt_pin})
    except BaseException as error:
        atomic(attempt / 'failure.json', {'error_type': type(error).__name__, 'error': str(error),
               'traceback': traceback.format_exc(), 'all_successful_checkpoints_and_prior_attempts_retained': True})
        raise
    finally:
        progress.close()
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-failed', action='store_true', help='Explicit repeat of reviewed failed saved-array checks; never inference.')
    for name in ('plan', 'ledger', 'metadata', 'original-queue-root', 'frozen-dir', 'output'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--worker-root', action='append', default=[], help='INDEX=/absolute/path/to/worker_XX; all twelve required')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'description_only', 'cells': 2160, 'retained_original': 331,
                          'missing_completion_grid': 1829, 'model_execution': False, 'incremental_audit_resume': True}))
        return
    need(all(getattr(args, name) is not None for name in ('plan', 'ledger', 'metadata', 'original_queue_root', 'frozen_dir', 'output')),
         'explicit inputs/output required')
    need(1 <= args.workers <= 128, 'CPU worker count must be1..128')
    execute(args)


if __name__ == '__main__':
    main()
