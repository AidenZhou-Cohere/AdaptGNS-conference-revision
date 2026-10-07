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
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
SCHEMA = 'goop3d_missing_autonomous_worker_v1'
PLAN_SHA = '838e7d4c1afc22fc4757a56ec0c755d5d303902d0ed600767d95c507d0402e08'


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


def save(path, doc, replace=False):
    raw = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    need(replace or not path.exists(), 'Refuse existing output: ' + str(path))
    with temp.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    return hashlib.sha256(raw).hexdigest()


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
                          'resume_semantics': 'restart original incomplete cell from six initial frames; no midtrajectory resume'}))
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
    out = args.output_root.resolve() / f'worker_{args.worker_index:02d}'
    protected = [runtime_root / 'adaptive-gns', runtime_root / 'research', prep, checkpoint.parent,
                 *(mapped(info['manifest_path']).parent for info in plan['splits'].values())]
    need(all(out != p and not out.is_relative_to(p) and not p.is_relative_to(out) for p in protected), 'Output overlaps frozen inputs')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    committed, current = [], None
    deadline = datetime.fromisoformat(plan['deadline_utc'])
    try:
        with until_deadline(deadline):
            device, runtime = T.configure_cuda(helpers, model_args)
            E.runtime_gate(model_args, {'gpu_uuid': args.gpu_uuid}, runtime)
            metadata = next(iter(manifests.values()))['metadata']
            need(all(m['metadata'] == metadata for m in manifests.values()), 'Split physics differs')
            model = E.prepare_model(model_args, {}, T, helpers, metadata, device)
            protocol = {'schema': SCHEMA, 'plan_sha256': PLAN_SHA, 'worker_index': args.worker_index,
                        'arm': worker['arm'], 'seed': worker['seed'], 'runtime': runtime,
                        'hostname': socket.gethostname(), 'pid': os.getpid(), 'worker_source_sha256': sha(__file__),
                        'original_repo_root': str(original_root), 'execution_runtime_root': str(runtime_root),
                        'input_files': checked, 'horizon': 295, 'trace_steps': [1, 10, 50, 200, 295],
                        'original_ledger_sha256': plan['original_ledger_sha256'],
                        'restart_semantics': 'initial_six_frames_no_midtrajectory_resume',
                        'cell_ids': [c['cell_id'] for c in worker['cells']]}
            protocol_sha = save(out / 'protocol.json', protocol)
            for cell in worker['cells']:
                current = cell['cell_id']
                save(out / 'status.json', {'state': 'running', 'current_cell_id': current, 'committed_cell_ids': committed}, replace=True)
                positions, types = arrays[cell['split'], cell['source_index']]
                helpers.synchronize(device)
                call = time.perf_counter()
                with helpers.torch.no_grad():
                    row, trace = native.rollout(model, positions, types, manifests[cell['split']]['metadata'], cell['policy'],
                                               295, 93000 + 1000 * worker['seed'] + cell['source_index'], device,
                                               trace_steps=(1, 10, 50, 200, 295))
                helpers.synchronize(device)
                call_seconds = time.perf_counter() - call
                need(all(isinstance(v, helpers.np.ndarray) and not v.dtype.hasobject for v in trace.values()), 'Non-numeric artifact')
                directory = out / cell['split']
                directory.mkdir(exist_ok=True)
                artifact = directory / f'trajectory_{cell["source_index"]:06d}_{cell["policy"]}.npz'
                need(not artifact.exists(), 'Refuse a duplicate new cell')
                io = time.perf_counter()
                temporary = artifact.with_suffix('.npz.tmp')
                with temporary.open('xb') as stream:
                    helpers.np.savez_compressed(stream, **trace)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(artifact)
                row.update(**{k: cell[k] for k in ('source_index', 'trajectory_id', 'particles', 'size_group')},
                           arm=worker['arm'], training_seed=worker['seed'], objective='faithful', split=cell['split'],
                           protocol_sha256=protocol_sha, artifact_file=artifact.name, artifact_sha256=sha(artifact),
                           synchronized_call_seconds=call_seconds, publication_seconds=time.perf_counter() - io,
                           completion_cell_id=current, original_cell_state=cell['state'], original_directory=cell['original_directory'])
                row['mse_at_declared_trace_steps'] = {str(s): row['mse_per_step'][s-1] if row['completed_steps'] >= s else None for s in (1, 10, 50, 200, 295)}
                save(artifact.with_suffix('.json'), row)
                committed.append(current)
                if (row.get('failure') or {}).get('category') in ('native_parity_failure', 'execution_error'):
                    raise RuntimeError('Frozen numerical implementation/parity failure retained; stop this worker')
            # Recheck exactly the data/model/source inputs used by this worker.
            for original in list(checked):
                verify(original)
            need(sha(args.plan) == PLAN_SHA, 'Plan changed during execution')
            need(sha(__file__) == protocol['worker_source_sha256'], 'Worker source changed during execution')
            save(out / 'status.json', {'state': 'all_assigned_outcomes_committed', 'committed_cell_ids': committed,
                 'assigned_cells': len(worker['cells']), 'all_used_inputs_reverified': True,
                 'elapsed_seconds': time.perf_counter() - started, 'scientific_admission_requires_separate_review': True}, replace=True)
    except BaseException as error:
        failure = {'schema': SCHEMA, 'state': 'interrupted_or_failed', 'current_cell_id': current,
                   'committed_cell_ids': committed, 'error_type': type(error).__name__, 'error': str(error),
                   'elapsed_seconds': time.perf_counter() - started, 'all_outputs_retained': True,
                   'automatic_retry': False, 'midtrajectory_resume_available': False}
        save(out / 'failed_attempt.json', failure)
        save(out / 'status.json', failure, replace=True)
        raise


if __name__ == '__main__':
    main()
