#!/usr/bin/env python3
"""New-version missing-cell scheduler over unchanged frozen D3 numerical code.

No execution without --execute. One model is loaded once per worker. The old
evaluation CLI, supervisor, releases, deadlines and result directories are not
modified or invoked. Root owns cross-host worker identity and process control.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import uuid
from types import SimpleNamespace

sys.dont_write_bytecode = True
SCHEMA = 'goop3d_missing_autonomous_worker_v1'
PLAN_SHA = '838e7d4c1afc22fc4757a56ec0c755d5d303902d0ed600767d95c507d0402e08'
WRITE_ATTEMPT = 'pre_owner_pid_' + str(os.getpid())
REQUIRED_ENV = {
    'LD_LIBRARY_PATH': '/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_directory(path, exist_ok=False):
    path = Path(path)
    path.mkdir(exist_ok=exist_ok)
    need(path.is_dir() and not path.is_symlink(), 'Nonregular output directory: ' + str(path))
    fsync_directory(path.parent)


def save(path, doc, replace=False):
    raw = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.' + WRITE_ATTEMPT + '.' + uuid.uuid4().hex)
    need(replace or not path.exists(), 'Refuse existing output: ' + str(path))
    with temp.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()



def bind_json(path, expected, resume):
    path = Path(path)
    if path.exists():
        need(resume and path.is_file() and not path.is_symlink(), 'Existing identity requires explicit resume')
        need(read(path) == expected, 'Saved worker/source/plan/input identity differs: ' + str(path))
        return sha(path)
    return save(path, expected)


def process_start(pid):
    need(type(pid) is int and pid > 0, 'Invalid prior owner PID')
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[19])
    except FileNotFoundError:
        return None


def owner_identity():
    start = process_start(os.getpid())
    need(start is not None, 'Linux process start identity unavailable')
    return {'pid': os.getpid(), 'start_ticks': start,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'hostname': socket.gethostname(), 'started_utc': datetime.now(timezone.utc).isoformat()}


def prior_owner_is_live(prior, now):
    need(prior.get('hostname') == now['hostname'], 'Cannot verify a prior owner from another host')
    need(isinstance(prior.get('boot_id'), str) and type(prior.get('start_ticks')) is int,
         'Malformed prior owner identity')
    if prior['boot_id'] != now['boot_id']:
        return False
    return process_start(prior['pid']) == prior['start_ticks']


@contextmanager
def own_worker(out, resume):
    global WRITE_ATTEMPT
    out = Path(out)
    if resume:
        need(out.is_dir() and not out.is_symlink(), 'Explicit resume requires an existing regular worker directory')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        make_directory(out)
    lock = out / 'worker.lock'
    need(not lock.is_symlink(), 'Worker lock is a symlink')
    with lock.open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Worker has a live kernel lock owner; refuse duplicate execution') from error
        fsync_directory(out)
        try:
            now = owner_identity()
            previous = read(out / 'owner.json') if (out / 'owner.json').exists() else None
            if previous is not None:
                need(resume, 'Existing owner requires explicit resume')
                need(not prior_owner_is_live(previous, now), 'Prior PID/start identity is still live; root must verify closure')
            attempts = out / 'attempts'
            make_directory(attempts, exist_ok=True)
            indices = [int(p.name.split('_')[1]) for p in attempts.glob('attempt_*') if p.is_dir()]
            attempt = attempts / f'attempt_{max(indices, default=0) + 1:06d}'
            make_directory(attempt)
            WRITE_ATTEMPT = attempt.name
            now['attempt'] = attempt.name
            now['explicit_resume'] = resume
            save(attempt / 'owner.json', now)
            if previous is not None:
                save(attempt / 'previous_owner.json', previous)
            save(out / 'owner.json', now, replace=True)
            yield attempt, now
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def committed_cells(out, cells, identity_sha, protocol_sha):
    """Verify final commits without interpreting accuracy or numerical success."""
    out = Path(out)
    allowed = {cell['cell_id']: cell for cell in cells}
    committed = {}
    root = out / 'cells'
    if not root.exists():
        return committed
    need(root.is_dir() and not root.is_symlink(), 'Invalid cell root')
    for directory in root.iterdir():
        need(directory.name in allowed and directory.is_dir() and not directory.is_symlink(), 'Unassigned/nonregular cell directory')
        marker = directory / 'commit.json'
        if not marker.exists():
            continue  # All partial files remain untouched in their old attempt directory.
        need(marker.is_file() and not marker.is_symlink(), 'Invalid commit marker')
        doc = read(marker)
        cell = allowed[directory.name]
        need(doc.get('schema') == SCHEMA + '_cell_commit'
             and doc.get('cell_id') == cell['cell_id'] and doc.get('plan_sha256') == PLAN_SHA
             and doc.get('identity_sha256') == identity_sha and doc.get('protocol_sha256') == protocol_sha,
             'Cell commit identity differs')
        paths = {}
        for kind in ('row', 'artifact'):
            relative = Path(doc[kind + '_file'])
            path = directory / relative
            need(not relative.is_absolute() and '..' not in relative.parts
                 and path.resolve().is_relative_to(directory.resolve()) and path.is_file()
                 and not path.is_symlink(), 'Invalid committed artifact path')
            need(sha(path) == doc[kind + '_sha256'], 'Committed artifact bytes differ: ' + str(path))
            paths[kind] = path
        row = read(paths['row'])
        need(row.get('completion_cell_id') == cell['cell_id']
             and row.get('protocol_sha256') == protocol_sha
             and row.get('artifact_sha256') == doc['artifact_sha256']
             and row.get('artifact_file') == paths['artifact'].name
             and row.get('training_seed') == cell['seed'] and row.get('arm') == cell['arm']
             and row.get('split') == cell['split'] and row.get('source_index') == cell['source_index']
             and row.get('policy') == cell['policy'], 'Committed row identity differs')
        committed[cell['cell_id']] = row
    return committed


def commit_cell(out, cell, attempt_name, row, trace, np, identity_sha, protocol_sha, io_started=None):
    directory = Path(out) / 'cells'
    make_directory(directory, exist_ok=True)
    directory /= cell['cell_id']
    make_directory(directory, exist_ok=True)
    need(not (directory / 'commit.json').exists(), 'Refuse duplicate committed cell')
    attempt = directory / attempt_name
    make_directory(attempt)
    artifact = attempt / 'trace.npz'
    temporary = artifact.with_name(artifact.name + '.tmp.' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **trace)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(artifact)
    fsync_directory(attempt)
    artifact_sha = sha(artifact)
    row.update(artifact_file=artifact.name, artifact_sha256=artifact_sha)
    if io_started is not None:
        row["publication_seconds"] = time.perf_counter() - io_started
    row_file = attempt / 'row.json'
    row_sha = save(row_file, row)
    save(directory / 'commit.json', {'schema': SCHEMA + '_cell_commit',
         'cell_id': cell['cell_id'], 'plan_sha256': PLAN_SHA, 'identity_sha256': identity_sha,
         'protocol_sha256': protocol_sha, 'row_file': str(row_file.relative_to(directory)),
         'row_sha256': row_sha, 'artifact_file': str(artifact.relative_to(directory)),
         'artifact_sha256': artifact_sha, 'attempt': attempt_name})
    return row

def load(path, pin, name):
    need(sha(path) == pin, 'Frozen source differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def until_deadline(deadline):
    seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
    need(seconds > 0, 'Submission deadline reached')
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Unexpected existing alarm')
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Submission deadline reached')))
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Operator-verified dead-owner resume of the exact same frozen worker/plan/inputs; never numerical retry')
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('plan.json'))
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--cuda-index', type=int)
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'schema': SCHEMA, 'status': 'source_candidate_no_execution',
                          'workers': 12, 'missing_cells': 1829, 'horizon': 295,
                          'resume_semantics': 'explicit verified resume skips every committed outcome and restarts only uncommitted cells from six initial frames; no midtrajectory resume'}))
        return
    need(all(getattr(args, k) is not None for k in ('runtime_root', 'output_root', 'worker_index', 'cuda_index', 'gpu_uuid')), 'All worker/runtime/output/device arguments required')
    need(sha(args.plan) == PLAN_SHA, 'Exact reviewed missing-cell plan required')
    plan = read(args.plan)
    need(plan['schema'] == 'goop3d_missing_autonomous_completion_plan_v1' and plan['horizon'] == 295 and plan['endpoint_updates'] == 25000, 'Study identity differs')
    need(0 <= args.worker_index < 12 and args.cuda_index >= 0, 'Invalid worker/device')
    worker = plan['workers'][args.worker_index]
    need(worker['worker_index'] == args.worker_index, 'Worker mapping differs')
    all_cells = [c['cell_id'] for w in plan['workers'] for c in w['cells']]
    retained = {c['cell_id'] for c in plan['retained_original_outcomes']}
    need(len(all_cells) == len(set(all_cells)) == 1829 and len(retained) == 331 and not retained.intersection(all_cells), 'Missing/retained coverage overlaps or differs')
    need(all(c['state'] in ('timed_out_current', 'not_completed_before_invocation_end', 'never_started') for c in worker['cells']), 'Only originally missing cells may run')
    runtime_root = args.runtime_root.resolve()
    original_root = Path(plan['original_remote_repo'])
    file_rows = {row['path']: row for row in plan['files']}

    def mapped(original):
        relative = Path(original).relative_to(original_root)
        target = runtime_root / relative
        need(target.resolve().is_relative_to(runtime_root), 'Mapped input escapes runtime root')
        return target

    checked = {}

    def verify(original):
        row = file_rows[original]
        path = mapped(original)
        need(path.is_file() and not path.is_symlink(), 'Missing/nonregular mapped input: ' + str(path))
        if row['size_bytes'] is not None:
            need(path.stat().st_size == row['size_bytes'], 'Input size differs: ' + str(path))
        need(sha(path) == row['sha256'], 'Input bytes differ: ' + str(path))
        checked[original] = {'original_path': original, 'execution_path': str(path),
                             'sha256': row['sha256'], 'size_bytes': path.stat().st_size}
        return path

    out = args.output_root.resolve() / f'worker_{args.worker_index:02d}'
    model_info = next(m for m in plan['models'] if (m['arm'], m['seed']) == (worker['arm'], worker['seed']))
    protected = [runtime_root / 'adaptive-gns', runtime_root / 'research', runtime_root / 'cuda_preparation',
                 mapped(model_info['checkpoint_path']).parent,
                 *(mapped(info['manifest_path']).parent for info in plan['splits'].values())]
    need(all(out != p and not out.is_relative_to(p) and not p.is_relative_to(out) for p in protected), 'Output overlaps frozen inputs')
    need(all('/' not in c['cell_id'] and c['cell_id'] not in ('', '.', '..') for c in worker['cells']), 'Invalid cell identity')
    source_sha = sha(__file__)
    with own_worker(out, args.resume) as (attempt, owner):
        started = time.perf_counter()
        committed, current = {}, None
        try:
            need(all(os.environ.get(k) == v for k, v in REQUIRED_ENV.items()),
                 'Original LD_LIBRARY_PATH and CUBLAS_WORKSPACE_CONFIG are required before importing Torch: ' + str(REQUIRED_ENV))
            # Pin the numerical/source closure before importing any model module.
            for relative in plan['source_pins']:
                verify(str(original_root / relative))
            prep = runtime_root / 'cuda_preparation'
            E = load(prep / 'evaluate_goop3d_graph_support_v1.py', plan['source_pins']['cuda_preparation/evaluate_goop3d_graph_support_v1.py'], '_d3_completion_frozen_evaluator')
            T = load(prep / 'train_goop3d_graph_support_cuda_v2.py', plan['source_pins']['cuda_preparation/train_goop3d_graph_support_cuda_v2.py'], '_d3_completion_frozen_trainer')
            helpers, _ = T.load_helpers(runtime_root)
            native = load(prep / 'goop3d_native_evaluation_v1.py', plan['source_pins']['cuda_preparation/goop3d_native_evaluation_v1.py'], '_d3_completion_frozen_native')
            need(list(E.POLICIES) == list(native.POLICIES) == plan['policies'], 'Frozen policy set/order differs')
            cohort_path, audit_path = verify(plan['cohort_path']), verify(plan['cohort_audit_path'])
            cohort, audit = read(cohort_path), read(audit_path)
            model_info = next(m for m in plan['models'] if (m['arm'], m['seed']) == (worker['arm'], worker['seed']))
            checkpoint = verify(model_info['checkpoint_path'])
            model_args = SimpleNamespace(purpose='final_evaluation', arm=worker['arm'], seed=worker['seed'],
                checkpoint=checkpoint, checkpoint_sha256=model_info['checkpoint_sha256'], checkpoint_updates=25000,
                protocol=prep / 'goop3d_scientific_protocol_v1.md', trainer_source=prep / 'train_goop3d_graph_support_cuda_v2.py',
                cohort_audit=audit_path, cuda_index=args.cuda_index, gpu_uuid=args.gpu_uuid, threads=2)
            E.cohort_gate(cohort, audit, {'scientific_endpoint_updates': 25000}, model_args)
            manifests, arrays = {}, {}
            for split in sorted({c['split'] for c in worker['cells']}):
                split_info = plan['splits'][split]
                manifest_path = verify(split_info['manifest_path'])
                manifest = read(manifest_path)
                metadata_path = verify(str(Path(split_info['manifest_path']).parent / 'metadata.json'))
                need(read(metadata_path) == manifest['metadata'], 'Metadata bytes/value differ')
                need(E.schedules(manifest['records'], 'full-rollout', 'final_evaluation') == [
                    {'source_index': r['source_index'], 'trajectory_id': r['id'],
                     'particles': r['positions']['shape'][1], 'size_group': 'fixed_source_grid'}
                    for r in split_info['records']
                ], 'Complete original source grid differs')
                manifests[split] = manifest
                for index in sorted({c['source_index'] for c in worker['cells'] if c['split'] == split}):
                    record = manifest['records'][index]
                    need(record == next(r for r in split_info['records'] if r['source_index'] == index), 'Selected source descriptor differs')
                    for key in ('positions', 'particle_types', 'step_context'):
                        verify(str(Path(split_info['manifest_path']).parent / record[key]['path']))
                    positions = helpers.data_loader._manifest_array(manifest_path.parent.resolve(), record['positions'])
                    types = helpers.data_loader._manifest_array(manifest_path.parent.resolve(), record['particle_types'])
                    need(positions.dtype.str == '<f4' and positions.shape == (301, record['positions']['shape'][1], 3)
                         and helpers.np.isfinite(positions).all() and types.shape == (positions.shape[1],) and helpers.np.all(types == 7), 'Frozen D3 array contract differs')
                    arrays[split, index] = positions, types
            identity = {'schema': SCHEMA + '_identity', 'plan_sha256': PLAN_SHA,
                        'worker_source_sha256': source_sha, 'worker_index': args.worker_index,
                        'arm': worker['arm'], 'seed': worker['seed'], 'cell_ids': [c['cell_id'] for c in worker['cells']],
                        'original_repo_root': str(original_root), 'execution_runtime_root': str(runtime_root),
                        'cuda_index': args.cuda_index, 'gpu_uuid': args.gpu_uuid,
                        'hostname': socket.gethostname(), 'required_environment': REQUIRED_ENV, 'input_files': checked}
            identity_path = out / 'worker_identity.json'
            protocol_path = out / 'protocol.json'
            need(identity_path.exists() or not protocol_path.exists(), 'Protocol without durable worker identity')
            identity_sha = bind_json(identity_path, identity, args.resume)
            save(attempt / 'identity.json', {'identity_sha256': identity_sha, 'worker_source_sha256': source_sha,
                                          'plan_sha256': PLAN_SHA})
            protocol = None
            if protocol_path.exists():
                need(args.resume and not protocol_path.is_symlink(), 'Existing protocol requires explicit resume')
                protocol = read(protocol_path)
                need(protocol['identity_sha256'] == identity_sha and protocol['plan_sha256'] == PLAN_SHA
                     and protocol['worker_source_sha256'] == source_sha, 'Saved protocol identity differs')
                protocol_sha = sha(protocol_path)
                committed = committed_cells(out, worker['cells'], identity_sha, protocol_sha)
            else:
                need(not (out / 'cells').exists(), 'Cell outputs without a durable protocol')
            save(attempt / 'resume_verification.json', {
                 'committed_cell_ids': list(committed), 'committed_outcomes_skipped_regardless_of_status': True,
                 'uncommitted_cells_restart_from_initial_six_frames': True, 'old_attempts_retained': True})
            need(not any((r.get('failure') or {}).get('category') in ('native_parity_failure', 'execution_error')
                         for r in committed.values()),
                 'Committed implementation/parity failure requires source review; same-source resume cannot bypass it')
            deadline = datetime.fromisoformat(plan['deadline_utc'])
            if len(committed) < len(worker['cells']):
                with until_deadline(deadline):
                    device, runtime = T.configure_cuda(helpers, model_args)
                    E.runtime_gate(model_args, {'gpu_uuid': args.gpu_uuid}, runtime)
                    if protocol is None:
                        protocol = {'schema': SCHEMA, 'plan_sha256': PLAN_SHA, 'worker_index': args.worker_index,
                                    'arm': worker['arm'], 'seed': worker['seed'], 'runtime': runtime,
                                    'hostname': socket.gethostname(), 'worker_source_sha256': source_sha,
                                    'identity_sha256': identity_sha,
                                    'original_repo_root': str(original_root), 'execution_runtime_root': str(runtime_root),
                                    'input_files': checked, 'horizon': 295, 'trace_steps': [1, 10, 50, 200, 295],
                                    'original_ledger_sha256': plan['original_ledger_sha256'],
                                    'restart_semantics': 'initial_six_frames_no_midtrajectory_resume',
                                    'cell_ids': [c['cell_id'] for c in worker['cells']]}
                        protocol_sha = save(protocol_path, protocol)
                    else:
                        need(runtime == protocol['runtime'], 'Resumed runtime differs from saved protocol')
                    save(attempt / 'runtime.json', runtime)
                    metadata = next(iter(manifests.values()))['metadata']
                    need(all(m['metadata'] == metadata for m in manifests.values()), 'Split physics differs')
                    model = E.prepare_model(model_args, {}, T, helpers, metadata, device)
                    for cell in worker['cells']:
                        if cell['cell_id'] in committed:
                            continue
                        current = cell['cell_id']
                        save(out / 'status.json', {'state': 'running', 'attempt': attempt.name,
                             'current_cell_id': current, 'committed_cell_ids': list(committed)}, replace=True)
                        positions, types = arrays[cell['split'], cell['source_index']]
                        with helpers.torch.no_grad():
                            call = time.perf_counter()
                            helpers.synchronize(device)
                            row, trace = native.rollout(model, positions, types, manifests[cell['split']]['metadata'], cell['policy'],
                                                       295, 93000 + 1000 * worker['seed'] + cell['source_index'], device,
                                                       trace_steps=(1, 10, 50, 200, 295))
                            helpers.synchronize(device)
                            call_seconds = time.perf_counter() - call
                        need(all(isinstance(v, helpers.np.ndarray) and not v.dtype.hasobject for v in trace.values()), 'Non-numeric artifact')
                        io = time.perf_counter()
                        row.update(**{k: cell[k] for k in ('source_index', 'trajectory_id', 'particles', 'size_group')},
                                   arm=worker['arm'], training_seed=worker['seed'], objective='faithful', split=cell['split'],
                                   protocol_sha256=protocol_sha, synchronized_call_seconds=call_seconds,
                                   completion_cell_id=current, original_cell_state=cell['state'], original_directory=cell['original_directory'])
                        row['mse_at_declared_trace_steps'] = {str(s): row['mse_per_step'][s-1] if row['completed_steps'] >= s else None for s in (1, 10, 50, 200, 295)}
                        # Publication timing includes durable NPZ serialization/hash, before row JSON.
                        row = commit_cell(out, cell, attempt.name, row, trace, helpers.np, identity_sha, protocol_sha, io)
                        committed[current] = row
                        if (row.get('failure') or {}).get('category') in ('native_parity_failure', 'execution_error'):
                            raise RuntimeError('Frozen numerical implementation/parity failure retained; stop this worker')
            # Recheck exactly the data/model/source inputs used by this worker, including an all-committed resume.
            for original in list(checked):
                verify(original)
            need(sha(args.plan) == PLAN_SHA, 'Plan changed during execution')
            need(sha(__file__) == source_sha, 'Worker source changed during execution')
            outcome = {'state': 'all_assigned_outcomes_committed', 'attempt': attempt.name,
                       'committed_cell_ids': list(committed), 'assigned_cells': len(worker['cells']),
                       'all_used_inputs_reverified': True, 'elapsed_seconds': time.perf_counter() - started,
                       'scientific_admission_requires_separate_review': True}
            save(attempt / 'outcome.json', outcome)
            save(out / 'status.json', outcome, replace=True)
        except BaseException as error:
            failure = {'schema': SCHEMA, 'state': 'interrupted_or_failed', 'attempt': attempt.name,
                       'owner': owner, 'current_cell_id': current, 'committed_cell_ids': list(committed),
                       'plan_sha256': PLAN_SHA, 'worker_source_sha256': source_sha,
                       'error_type': type(error).__name__, 'error': str(error),
                       'elapsed_seconds': time.perf_counter() - started, 'all_outputs_retained': True,
                       'automatic_retry': False, 'midtrajectory_resume_available': False}
            save(attempt / 'failed_attempt.json', failure)
            save(out / 'status.json', failure, replace=True)
            raise


if __name__ == '__main__':
    main()
