#!/usr/bin/env python3
"""Explicit Sand runtime-migration adapter; no numerical trainer changes.

The frozen run_config remains historical origin lineage. A separately pinned
receipt describes the actual replacement runtime. Only its UUID may differ.
Replay artifacts never initialize scientific training; training restores the
original immutable 50000-update parent again through the frozen restore code.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
SCHEMA = 'adaptgns_sand_runtime_migration_recovery_release_v1'
TRAINER_SHA = 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
PROTOCOL_SHA = 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d'
TRAINING_SECONDS = 27404.671591931674
POSTTRAINING_SECONDS = 18960
LATEST = datetime(2026, 10, 6, 15, 5, tzinfo=timezone.utc)
STOP = datetime(2026, 10, 6, 22, 44, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 10, 7, 4, tzinfo=timezone.utc)
JOBS = [('base_seed1', 'base', 1, 0), ('mix_seed1', 'mix', 1, 1), ('base_seed2', 'base', 2, 2), ('mix_seed2', 'mix', 2, 3)]
PARENT_FILES = {'protocol.json': 'origin_protocol_sha256', 'latest.json': 'origin_pointer_sha256',
                'checkpoint-000050000.pt': 'origin_checkpoint_sha256', 'run.lock': 'origin_lock_sha256'}
REPLAY_STEPS = 100
LEGACY_FIELDS = ('completed_steps', 'loss', 'lr', 'frame_ids', 'particles')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    p = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with p.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def stamp(value):
    t = datetime.fromisoformat(value)
    require(t.tzinfo is not None, 'Timezone-aware clock required')
    return t.astimezone(timezone.utc)


def absolute(value):
    p = Path(value)
    require(p.is_absolute() and str(p) == str(p.resolve()), 'Canonical absolute nonsymlink path required')
    return p


def runtime_contract(original, actual):
    require(isinstance(original, dict) and isinstance(actual, dict) and set(original) == set(actual), 'Complete runtime field sets required')
    require('uuid' in original and original['uuid'] != actual['uuid'], 'Explicit changed physical UUID required')
    require(json.dumps({k: v for k, v in original.items() if k != 'uuid'}, sort_keys=True, allow_nan=False)
            == json.dumps({k: v for k, v in actual.items() if k != 'uuid'}, sort_keys=True, allow_nan=False),
            'Only UUID may differ; no platform, package, device, precision or determinism waiver')
    require('GB200' in str(actual.get('name')) and actual.get('torch') == '2.13.0+cu129' and actual.get('cuda') == '12.9',
            'Original validated GB200 stack required')
    require(actual.get('deterministic_algorithms') is True and actual.get('deterministic_warn_only') is False
            and all(actual.get(k) is False for k in ('tf32', 'amp', 'compile', 'ddp')) and actual.get('threads') == 2
            and actual.get('cublas_workspace_config') == ':4096:8', 'Original strict deterministic runtime required')


def validate_release(path, expected_sha, mode):
    require(mode in ('replay', 'train'), 'Explicit mode required')
    path = absolute(path)
    require(sha(path) == expected_sha, 'Exact root release bytes required')
    r = read(path)
    require(r.get('schema') == SCHEMA and r.get('status') == 'approved_for_sand_migration_' + ('training' if mode == 'train' else 'replay')
            and r.get('issued_by') == 'root' and r.get('host') == socket.gethostname()
            and isinstance(r.get('original_host'), str) and r['original_host'] != r['host'], 'Explicit root-reviewed host migration release required')
    require(r.get('adapter_sha256') == sha(__file__) and r.get('original_trainer_sha256') == TRAINER_SHA
            and sha(r['original_trainer_path']) == TRAINER_SHA and sha(r['protocol']) == PROTOCOL_SHA,
            'Frozen numerical trainer, original scientific protocol and separately pinned adapter required')
    require(sha(absolute(r['migration_protocol_path'])) == r.get('migration_protocol_sha256'),
            'Separately reviewed recovery protocol amendment required')
    require(r.get('automatic_retry_resume_or_promotion') is False and r.get('required_models') == 6 and r.get('required_policies') == 6,
            'Complete original study and no automatic retry/promotion required')
    require([(j.get('id'), j.get('arm'), j.get('seed'), j.get('gpu')) for j in r.get('jobs', [])] == JOBS,
            'All four original A models with original GPU-index mapping required')
    now = datetime.now(timezone.utc)
    error = r.get('clock_error_bound_seconds')
    require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 5, 'Bounded actual clock required')
    require(stamp(r['training_stop_utc']) == STOP and stamp(r['compute_analysis_deadline_utc']) == DEADLINE
            and stamp(r['latest_start_utc']) <= LATEST and r.get('training_allocation_seconds') == TRAINING_SECONDS
            and r.get('post_training_allocation_seconds') == POSTTRAINING_SECONDS, 'Unchanged whole original allocation required')
    require(-5 <= (now - stamp(r['clock_checked_utc'])).total_seconds() <= 300
            and -5 <= (now - stamp(r['process_checked_utc'])).total_seconds() <= 300
            and now + timedelta(seconds=error) <= stamp(r['latest_start_utc'])
            and now + timedelta(seconds=TRAINING_SECONDS + error + 15) <= STOP, 'Fresh evidence and original launch window required')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and os.environ.get('CUDA_VISIBLE_DEVICES') is None
            and os.environ.get('LD_LIBRARY_PATH') == '/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
            'Original deterministic environment and unmapped CUDA indexing required')
    env = r.get('python_environment', {})
    require(env.get('lexical_path') == sys.executable and env.get('resolved_binary_path') == str(Path(sys.executable).resolve())
            and env.get('binary_sha256') == sha(sys.executable) and env.get('sys_prefix') == sys.prefix
            and env.get('sys_base_prefix') == sys.base_prefix, 'Actual exact lexical interpreter provenance required')
    cfg = Path(sys.executable).parent.parent / 'pyvenv.cfg'
    require(env.get('pyvenv_config_path') == (str(cfg) if cfg.exists() else None)
            and env.get('pyvenv_config_sha256') == (sha(cfg) if cfg.exists() else None), 'Actual virtual-environment bytes differ')
    pins = r.get('files_sha256')
    require(isinstance(pins, dict) and pins, 'Complete absolute input bindings required')
    for p, digest in pins.items():
        require((str(Path(p).resolve()) == p or p == sys.executable) and Path(p).is_absolute() and sha(p) == digest, 'Pinned bytes differ: ' + p)
    for p in (str(Path(__file__).resolve()), r['original_trainer_path'], r['train_manifest'], r['admission'], r['structural_report'], r['protocol'], r['migration_protocol_path']):
        require(pins.get(p) == sha(p), 'Required source/data-control binding missing: ' + p)
    r['_release_path'] = str(path)
    r['_release_sha256'] = expected_sha
    return r


def select_job(release, identity):
    rows = [j for j in release['jobs'] if j['id'] == identity]
    require(len(rows) == 1, 'One exact original A job required')
    return rows[0]


def verify_origin(release, job):
    origin = absolute(job['origin_directory'])
    for name, key in PARENT_FILES.items():
        path = absolute(str(origin / name))
        require(sha(path) == job[key] == release['files_sha256'].get(str(path)), 'Immutable original binding differs: ' + name)
    log = absolute(job['origin_stdout_path'])
    require(log == origin.parent.parent / 'logs' / (job['id'] + '.stdout.jsonl')
            and sha(log) == job['origin_stdout_sha256'] == release['files_sha256'].get(str(log)), 'Exact retained original stdout required')
    config = read(origin / 'protocol.json')
    pointer = read(origin / 'latest.json')
    lock = read(origin / 'run.lock')
    require(lock.get('host') == release['original_host'], 'Original retained lock must identify old host')
    require(pointer.get('completed_steps') == 50000 and type(pointer['completed_steps']) is int
            and pointer.get('path') == 'checkpoint-000050000.pt' and pointer.get('sha256') == job['origin_checkpoint_sha256'],
            'Only the original committed 50000-update state may initialize recovery')
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(pointer.get('run_config_sha256') == canonical and config.get('runtime') == job['original_runtime']
            and config.get('seed') == job['seed'] and config.get('arm') == job['arm'] and config.get('updates') == 100000
            and config.get('research_protocol_sha256') == PROTOCOL_SHA, 'Original checkpoint lineage/configuration differs')
    runtime_contract(job['original_runtime'], job['actual_runtime'])
    require(job['actual_runtime']['device'] == 'cuda:' + str(job['gpu']), 'Original CUDA-index mapping required')
    return origin, config, pointer


def legacy_replay_endpoint(job):
    path = Path(job['origin_stdout_path'])
    require(sha(path) == job['origin_stdout_sha256'], 'Original log changed before comparison')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get('completed_steps') == 50000 + REPLAY_STEPS]
    require(len(selected) == 1 and all(k in selected[0] for k in LEGACY_FIELDS), 'One exact original 50100 scalar record required')
    require(sha(path) == job['origin_stdout_sha256'], 'Original log changed during comparison')
    return {k: selected[0][k] for k in LEGACY_FIELDS}


def load_trainer(release):
    spec = importlib.util.spec_from_file_location('_sand_frozen_numerical_trainer_for_explicit_migration', release['original_trainer_path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trainer_args(release, job, output):
    return SimpleNamespace(repo=Path(release['repo']), train_manifest=Path(release['train_manifest']), admission=Path(release['admission']),
        structural_report=Path(release['structural_report']), protocol=Path(release['protocol']), output_dir=Path(output),
        objective='faithful', arm=job['arm'], seed=job['seed'], updates=100000, cuda_index=job['gpu'], threads=2,
        checkpoint_every=10000, log_every=100, stop_after=None, resume=True, clear_stale_lock=False)


def identity_record(release, job, config, runtime):
    return {'schema': 'adaptgns_sand_explicit_runtime_migration_v1', 'original_host': release['original_host'], 'actual_host': release['host'],
        'job': job['id'], 'origin_directory': job['origin_directory'], 'origin_checkpoint_sha256': job['origin_checkpoint_sha256'],
        'origin_protocol_sha256': job['origin_protocol_sha256'], 'origin_pointer_sha256': job['origin_pointer_sha256'],
        'origin_stdout_sha256': job['origin_stdout_sha256'],
        'original_runtime': config['runtime'], 'actual_runtime': runtime, 'allowed_runtime_differences': ['uuid'],
        'origin_run_config_is_historical_lineage': True, 'original_frozen_trainer_sha256': TRAINER_SHA,
        'adapter_sha256': sha(__file__), 'root_release_sha256': release['_release_sha256'],
        'parent_completed_steps': 50000, 'target_completed_steps': 100000, 'model_initialization_from_replay_artifact': False}


def protected_output(release, job, key):
    output = absolute(job[key])
    protected = [absolute(job['origin_directory']).parent.parent, absolute(str(Path(release['train_manifest']).parent)), absolute(str(Path(release['repo']) / 'adaptive-gns'))]
    for row in release['jobs']:
        protected.append(absolute(row['origin_directory']).parent.parent)
        protected.extend(absolute(row[k]) for k in ('origin_directory', 'output_directory', 'replay_output_directory')
                         if not (row['id'] == job['id'] and k == key))
    require(not output.exists() and all(output != p and p not in output.parents and output not in p.parents for p in protected),
            'Fresh disjoint recovery output required; originals and prior attempts are immutable')
    return output


def tree_digest(torch, value):
    h = hashlib.sha256()
    def walk(v):
        if isinstance(v, torch.Tensor):
            x = v.detach().cpu().contiguous()
            h.update(b'tensor:' + str(x.dtype).encode() + b':' + repr(tuple(x.shape)).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k, x in v.items():
                walk(k)
                walk(x)
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for x in v:
                walk(x)
        else:
            h.update(type(v).__name__.encode() + b':' + repr(v).encode() + b';')
    walk(value)
    return h.hexdigest()


def replay(release, job, T, helpers, support, args, device, runtime, origin, config, pointer, output):
    torch = helpers.torch
    train, metadata, data_info = T.load_admitted_dataset(args, helpers)
    require(data_info == config['data'], 'Exact original complete data/configuration required')
    require(config['source_sha256'] == {**T.SOURCE_PINS, 'train_sand_graph_support_cuda.py': TRAINER_SHA}, 'Exact original numerical closure required')
    payload = torch.load(origin / pointer['path'], map_location='cpu', weights_only=True)
    require(sha(origin / pointer['path']) == job['origin_checkpoint_sha256'], 'Parent changed during safe load')
    model = helpers.build_simulator(metadata, helpers.NOISE, helpers.NOISE, device, connectivity_radius=.015,
        nmessage_passing_steps=10, uncertainty_parameterization='variance', variance_floor=1e-6,
        detach_variance_features=True, radius_backend='scipy_host').to(device)
    model._training_config = {'loss': 'faithful', 'cuda_sand_graph_support_run': config}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
    reference = None
    digests = []
    legacy = legacy_replay_endpoint(job)
    for repeat in range(2):
        # Optimizer loading may retain a CPU step tensor by reference. Clone its
        # input so replay one cannot mutate the immutable in-memory parent used
        # for replay two. This changes neither parent bytes nor restore checks.
        restoration_payload = {**payload, 'optimizer_state': copy.deepcopy(payload['optimizer_state'])}
        completed, history = T.restore_payload(helpers, restoration_payload, model, optimizer, config, device)
        require(completed == 50000 and len(history['graph_updates']) == 50000, 'Exact parent restoration required')
        ledger = []
        for step in range(50000, 50000 + REPLAY_STEPS):
            indices = helpers.sample_indices(args.seed, step, len(train), 2)
            batch = helpers.unpack_batch([train[index] for index in indices])
            noise = helpers.host_noise(batch[0].shape, batch[1], args.seed, step)
            frame_ids = [helpers.frame_identity(train, data_info['trajectory_ids'], index) for index in indices]
            rate = helpers.learning_rate(step)
            for group in optimizer.param_groups:
                group['lr'] = rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred, head, target, graph_ledger = support.forward_batch(model, batch, noise, device, args.seed, step, args.arm)
            mask = (batch[1] != helpers.KINEMATIC).to(device)
            loss = T.guarded_update(helpers, model, optimizer, pred, head, target, mask, 'faithful', step + 1)
            helpers.synchronize(device)
            ledger.append({'completed_steps': step + 1, 'absolute_schedule_step': step, 'frame_ids': frame_ids,
                           'noise_sha256': support.bridge.full.state_hash(noise.numpy()), 'examples': graph_ledger,
                           'loss': float(loss.detach().cpu()), 'lr': rate, 'particles': len(batch[0])})
        observed = {k: ledger[-1][k] for k in LEGACY_FIELDS}
        require(observed == legacy, 'Exact original 50100 loss/LR/frame/particle comparison failed; no tolerance or retry')
        result = {'state_dict': helpers.cpu_tree(model.state_dict()), 'optimizer_state': helpers.cpu_tree(optimizer.state_dict()),
                  'rng_states': T.capture_rng(torch, device), 'replay_ledger': ledger}
        digests.append(tree_digest(torch, result))
        if reference is None:
            reference = result
        else:
            require(T.tree_equal(torch, reference, result), 'Repeated same-runtime replay differs in model/Adam/RNG/schedule/loss/graph state')
    verify_origin(release, job)
    require(sha(release['_release_path']) == release['_release_sha256'], 'Replay release changed')
    receipt = {**identity_record(release, job, config, runtime), 'schema': 'adaptgns_sand_migration_parent_replay_receipt_v1',
        'status': 'passed_exact_restore_and_repeated_current_runtime_replay', 'restored_parent_steps': 50000,
        'replay_steps_per_repeat': REPLAY_STEPS, 'repeats': 2, 'absolute_schedule_steps': list(range(50000, 50000 + REPLAY_STEPS)),
        'exact_model_optimizer_rng_restoration': True, 'exact_repeated_state_and_schedule': True,
        'exact_original_50100_scalar_match': True, 'original_50100_scalar': legacy, 'replayed_50100_scalar': observed,
        'original_50100_model_optimizer_rng_not_saved': True,
        'replay_state_sha256': digests, 'no_scientific_checkpoint_written': True,
        'cross_physical_device_bitwise_equivalence_claim': False, 'completed_utc': datetime.now(timezone.utc).isoformat()}
    write(output / 'receipt.json', receipt)
    return receipt


def check_replay_receipt(release, job):
    path = absolute(job['replay_receipt_path'])
    require(sha(path) == job['replay_receipt_sha256'] == release['files_sha256'].get(str(path)), 'Exact separately completed replay receipt required')
    r = read(path)
    require(r.get('schema') == 'adaptgns_sand_migration_parent_replay_receipt_v1'
            and r.get('status') == 'passed_exact_restore_and_repeated_current_runtime_replay'
            and r.get('job') == job['id'] and r.get('origin_checkpoint_sha256') == job['origin_checkpoint_sha256']
            and r.get('actual_runtime') == job['actual_runtime'] and r.get('original_runtime') == job['original_runtime']
            and r.get('actual_host') == release['host'] and r.get('adapter_sha256') == sha(__file__)
            and r.get('origin_stdout_sha256') == job['origin_stdout_sha256']
            and r.get('restored_parent_steps') == 50000 and r.get('replay_steps_per_repeat') == REPLAY_STEPS and r.get('repeats') == 2
            and r.get('absolute_schedule_steps') == list(range(50000, 50000 + REPLAY_STEPS))
            and r.get('exact_original_50100_scalar_match') is True and r.get('original_50100_scalar') == r.get('replayed_50100_scalar') == legacy_replay_endpoint(job)
            and r.get('exact_model_optimizer_rng_restoration') is True and r.get('exact_repeated_state_and_schedule') is True
            and r.get('no_scientific_checkpoint_written') is True and r.get('cross_physical_device_bitwise_equivalence_claim') is False
            and r.get('original_50100_model_optimizer_rng_not_saved') is True
            and len(r.get('replay_state_sha256', [])) == 2 and len(set(r['replay_state_sha256'])) == 1,
            'Replay must test exact original state on this admitted actual runtime without scientific promotion')
    return r


def execute(release, job, mode):
    origin, config, pointer = verify_origin(release, job)
    output = protected_output(release, job, 'replay_output_directory' if mode == 'replay' else 'output_directory')
    if mode == 'train':
        check_replay_receipt(release, job)
    T = load_trainer(release)
    args = trainer_args(release, job, output)
    helpers, support = T.load_helpers(args.repo)
    device, runtime = T.configure_cuda(helpers, args)
    require(runtime == job['actual_runtime'], 'Actual configured runtime differs from root-pinned migration mapping')
    runtime_contract(config['runtime'], runtime)
    output.mkdir(parents=True, exist_ok=False)
    record = identity_record(release, job, config, runtime)
    try:
        if mode == 'replay':
            write(output / 'runtime_migration.json', record)
            with T.RunLock(output, False):
                return replay(release, job, T, helpers, support, args, device, runtime, origin, config, pointer, output)
        # Copies retain exactly the original lineage; no old lock or attempt is moved.
        for name in ('protocol.json', 'latest.json', 'checkpoint-000050000.pt'):
            with (origin / name).open('rb') as source, (output / name).open('xb') as target:
                shutil.copyfileobj(source, target, 1 << 20)
                target.flush()
                os.fsync(target.fileno())
            require(sha(output / name) == job[PARENT_FILES[name]], 'Recovery origin copy differs')
        write(output / 'runtime_migration.json', record)
        with T.RunLock(output, False) as process:
            # Explicit historical runtime parameter preserves every frozen config,
            # restore, RNG, Adam, schedule, finite-check and training-loop predicate.
            # The actual runtime was checked above and is mandatory separate lineage.
            T.run_training(args, helpers, support, process, device, config['runtime'])
        verify_origin(release, job)
        require(sha(release['_release_path']) == release['_release_sha256'], 'Training release changed')
        status = read(output / 'status.json')
        require(status.get('state') == 'complete' and status.get('completed_steps') == status.get('committed_steps') == 100000,
                'Only original complete 100000-update endpoint may complete recovery')
        result = {**record, 'schema': 'adaptgns_sand_runtime_migration_completed_attempt_v1', 'status': 'training_complete_pending_independent_endpoint_audit',
                  'final_pointer': read(output / 'latest.json'), 'replay_receipt_sha256': job['replay_receipt_sha256'],
                  'completed_utc': datetime.now(timezone.utc).isoformat(), 'original_artifacts_reverified': True}
        write(output / 'migration_receipt.json', result)
        return result
    except BaseException as error:
        write(output / 'migration_failure.json', {**record, 'error_type': type(error).__name__, 'error': str(error), 'all_partial_artifacts_retained': True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=('replay', 'train'))
    parser.add_argument('--release', type=Path)
    parser.add_argument('--release-sha256')
    parser.add_argument('--job')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status': 'description_only', 'schema': SCHEMA, 'root_release_required': True,
                          'origin_steps': 50000, 'endpoint_steps': 100000, 'no_numerical_source_changes': True}))
        return
    require(all((args.mode, args.release, args.release_sha256, args.job)), 'Complete explicit execution arguments required')
    release = validate_release(args.release, args.release_sha256, args.mode)
    execute(release, select_job(release, args.job), args.mode)


if __name__ == '__main__':
    main()
