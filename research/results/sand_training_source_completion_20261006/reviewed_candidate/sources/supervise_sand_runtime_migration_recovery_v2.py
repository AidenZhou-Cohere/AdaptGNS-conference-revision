#!/usr/bin/env python3
"""Own the four root-released Sand A runtime-migration jobs; describe by default.

The original trainer, scoped GPU predicates and child lifecycle remain frozen.
Old-host identities are closure evidence only: only registered local children
can be signalled. Replay outputs never become scientific endpoints.
"""
import time
ENTRY_MONOTONIC = time.perf_counter()

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
SCHEMA = 'adaptgns_sand_runtime_migration_recovery_owner_v1'
RELEASE_SCHEMA = 'adaptgns_sand_runtime_migration_recovery_release_v1'
SCOPED_SHA = '2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13'
LIFECYCLE_SHA = 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd'
TRAINER_SHA = 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
TRAINING_SECONDS = 27404.671591931674
POSTTRAINING_SECONDS = 18960
LATEST_START = datetime(2026, 10, 6, 15, 5, tzinfo=timezone.utc)
TRAINING_STOP = datetime(2026, 10, 6, 22, 44, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 10, 7, 4, tzinfo=timezone.utc)
REPLAY_STOP = datetime(2026, 10, 6, 15, 2, tzinfo=timezone.utc)
REPLAY_LATEST = datetime(2026, 10, 6, 14, 51, 55, tzinfo=timezone.utc)
REPLAY_SECONDS = 600
REPLAY_HANDOFF = 180
CLEANUP = 15
JOBS = [('base_seed1', 'base', 1, 0), ('mix_seed1', 'mix', 1, 1),
        ('base_seed2', 'base', 2, 2), ('mix_seed2', 'mix', 2, 3)]
ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'LD_LIBRARY_PATH': '/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'Timezone-aware timestamp required')
    return result.astimezone(timezone.utc)


def utc():
    return datetime.now(timezone.utc)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(v in '0123456789abcdef' for v in value)


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and str(p.resolve()) == str(p), 'Canonical absolute path required')
    return p


def snapshot(path, expected_sha):
    p = canonical(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= 1048576,
            'Bounded ordinary root release required')
    raw = p.read_bytes()
    require(digest(expected_sha) and hashlib.sha256(raw).hexdigest() == expected_sha,
            'Root release bytes differ')
    return json.loads(raw)


def private_import(path, expected_sha, name):
    p = canonical(path)
    require(digest(expected_sha) and sha(p) == expected_sha, 'Pinned source differs: ' + str(p))
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_runtime_except_uuid(original, actual):
    require(isinstance(original, dict) and isinstance(actual, dict) and set(original) == set(actual)
            and 'uuid' in original, 'Complete original and actual runtime receipts required')
    require(encode({k: v for k, v in original.items() if k != 'uuid'}) ==
            encode({k: v for k, v in actual.items() if k != 'uuid'}),
            'Runtime migration may change only UUID; all other runtime fields stay exact')


def validate_contract(release, mode, now, scoped, lifecycle):
    """Pure control validation; no model, array, manifest or trainer import."""
    require(release.get('schema') == RELEASE_SCHEMA and release.get('issued_by') == 'root'
            and release.get('status') == 'approved_for_sand_migration_' + ('replay' if mode == 'replay' else 'training')
            and release.get('host') == socket.gethostname() and isinstance(release.get('original_host'), str)
            and release['original_host'] != release['host'], 'Exact root runtime-migration release required')
    require(release.get('training_allocation_seconds') == TRAINING_SECONDS
            and release.get('post_training_allocation_seconds') == POSTTRAINING_SECONDS
            and stamp(release['training_stop_utc']) == TRAINING_STOP
            and stamp(release['compute_analysis_deadline_utc']) == DEADLINE,
            'Full original training and analysis allocations are fixed')
    latest = stamp(release['latest_start_utc'])
    error = release.get('clock_error_bound_seconds')
    require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 5,
            'Bounded clock error required')
    cleanup = TRAINING_STOP - timedelta(seconds=error + CLEANUP)
    require(latest <= LATEST_START and latest + timedelta(seconds=TRAINING_SECONDS) <= cleanup
            and now + timedelta(seconds=error) <= latest, 'Full unchanged training allocation no longer fits')
    jobs = release.get('jobs')
    require(isinstance(jobs, list) and [(j.get('id'), j.get('arm'), j.get('seed'), j.get('gpu')) for j in jobs] == JOBS,
            'All four original A models with fixed seed/GPU mapping required')
    uuids = []
    for job in jobs:
        same_runtime_except_uuid(job['original_runtime'], job['actual_runtime'])
        actual = job['actual_runtime']
        require(actual.get('device') == 'cuda:' + str(job['gpu']) and 'GB200' in actual.get('name', ''),
                'Exact GB200 device assignment required')
        uuids.append(actual['uuid'])
    require(len({lifecycle.canonical_gpu_uuid(v) for v in uuids}) == 4
            and release.get('gpu_uuids') == uuids and release.get('owned_gpu_indices') == [0, 1, 2, 3],
            'Four distinct current UUIDs and exact scoped ownership required')
    require(release.get('six_model_freeze_before_test') is True
            and release.get('automatic_retry_resume_or_promotion') is False
            and release.get('required_models') == 6 and release.get('required_policies') == 6,
            'Original six-model study and no automatic retry/promotion required')
    process = read(release['process_check_path'])
    clock = read(release['clock_check_path'])
    require(process.get('schema') == 'adaptgns_sand_runtime_migration_process_check_v1'
            and process.get('issued_by') == 'root' and process.get('host') == release['host']
            and process.get('original_host') == release['original_host']
            and process.get('all_original_owned_processes_absent') is True
            and process.get('new_host_idle') is True and process.get('gpu_uuids') == uuids
            and process.get('gpu_processes') == [], 'Fresh original closure and idle new host evidence required')
    prior = process.get('original_owned_processes')
    require(isinstance(prior, list) and len(prior) >= 5
            and all(type(v.get('pid')) is int and v['pid'] > 0
                    and type(v.get('start_ticks')) is int and v['start_ticks'] > 0
                    and isinstance(v.get('argv'), list) and v['argv']
                    and isinstance(v.get('executable'), str) and v['executable']
                    and v.get('absent') is True for v in prior),
            'Original owner and all four worker identity-absence receipts required')
    require(len({(v['pid'], v['start_ticks']) for v in prior}) == len(prior)
            and sum(scoped.executed_python_script(v['argv']) == 'train_sand_graph_support_cuda.py' for v in prior) == 4
            and any(scoped.executed_python_script(v['argv']) == 'supervise_sand_scoped_science_v1.py' for v in prior),
            'Distinct original scoped owner and all four trainer identities required')
    require(clock.get('schema') == 'adaptgns_sand_runtime_migration_clock_check_v1'
            and clock.get('issued_by') == 'root' and clock.get('host') == release['host']
            and clock.get('root_host_samples_reviewed') is True
            and clock.get('clock_error_bound_seconds') == error, 'Root bounded-clock evidence required')
    sample = clock['clock_sample']
    start, end, remote = (stamp(sample[k]) for k in ('root_start_utc', 'root_end_utc', 'remote_sample_utc'))
    require(start <= end and sample['offset_lower_seconds'] == (remote - end).total_seconds()
            and sample['offset_upper_seconds'] == (remote - start).total_seconds()
            and -error <= sample['offset_lower_seconds'] <= sample['offset_upper_seconds'] <= error
            and stamp(clock['checked_utc']) == remote, 'Bounded clock arithmetic differs')
    for key, evidence in [('process_checked_utc', process), ('clock_checked_utc', clock)]:
        require(release[key] == evidence['checked_utc']
                and -error <= (now - stamp(evidence['checked_utc'])).total_seconds() <= 300,
                'Fresh exact process and clock timestamps required')
    scoped.check_gpu(process['gpu_processes'], {}, lifecycle, uuids, [0, 1, 2, 3], lambda pid: None)
    require(release.get('replay_allocation_seconds') == REPLAY_SECONDS
            and stamp(release['replay_stop_utc']) == REPLAY_STOP
            and stamp(release['replay_latest_start_utc']) <= REPLAY_LATEST
            and release.get('replay_handoff_reserve_seconds') == REPLAY_HANDOFF
            and REPLAY_STOP + timedelta(seconds=REPLAY_HANDOFF) <= latest,
            'Fixed replay quota, stop and three-minute training handoff reserve required')
    replay_seconds = None
    if mode == 'replay':
        replay_seconds = release.get('replay_allocation_seconds')
        require(type(replay_seconds) is int and replay_seconds == REPLAY_SECONDS
                and now + timedelta(seconds=error) <= stamp(release['replay_latest_start_utc'])
                and now + timedelta(seconds=replay_seconds + error) <= REPLAY_STOP,
                'Whole fixed replay quota must fit its launch window and stop')
    return SimpleNamespace(release=release, jobs=jobs, mode=mode, latest=latest, stop=TRAINING_STOP,
                           cleanup=cleanup, error=error, uuids=uuids, process=process,
                           replay_seconds=replay_seconds, scoped=scoped, B=lifecycle)


def fixed_command(context, release_path, release_sha, job):
    r = context.release
    return [r['python_environment']['lexical_path'], '-B', r['adapter_path'], '--execute',
            '--mode', context.mode, '--release', str(release_path), '--release-sha256', release_sha, '--job', job['id']]


def output_paths(context):
    r = context.release
    output = canonical(r['owner_output_directory'])
    require(not output.exists(), 'Fresh owner directory required; preserve earlier attempts')
    origins = [canonical(j['origin_directory']) for j in context.jobs]
    training = [canonical(j['output_directory']) for j in context.jobs]
    replay = [canonical(j['replay_output_directory']) for j in context.jobs]
    destinations = training if context.mode == 'train' else replay
    require(all(not p.exists() and not p.is_symlink() for p in destinations), 'Every owned output must be fresh')
    paths = [output] + origins + training + replay
    require(len(set(paths)) == len(paths) and all(a != b and a not in b.parents and b not in a.parents
            for i, a in enumerate(paths) for b in paths[i + 1:]), 'Origin, replay, training and owner paths must be disjoint')
    original_trees = {p.parent.parent for p in origins}
    require(all(p != tree and tree not in p.parents and p not in tree.parents
                for p in [output] + training + replay for tree in original_trees),
            'Entire original scientific trees, including siblings, are immutable')
    protected = [canonical(r[k]) for k in ('repo',)]
    # Outputs may be inside the repository, but may never contain the repository.
    require(all(output != p and output not in p.parents for p in protected), 'Owner output contains protected repository')
    for p in (canonical(r['train_manifest']).parent, canonical(r['repo']) / 'adaptive-gns'):
        require(output != p and output not in p.parents and p not in output.parents,
                'Owner output overlaps protected scientific source/data')
    return output


def operational_bindings(release, release_path, release_sha, scoped):
    paths = ['adapter_path', 'scoped_owner_path', 'lifecycle_source_path', 'auditor_path',
             'process_check_path', 'clock_check_path', 'original_trainer_path', 'migration_protocol_path']
    pins = release['files_sha256']
    require(isinstance(pins, dict) and all(Path(k).is_absolute() and digest(v) for k, v in pins.items()),
            'Absolute source/control pin map required')
    selected = {str(canonical(release[k])): pins[release[k]] for k in paths}
    own = str(Path(__file__).resolve())
    require(pins.get(own) == release['owner_sha256'] == sha(own), 'Owner source pin differs')
    selected[own] = pins[own]
    require(selected[release['scoped_owner_path']] == SCOPED_SHA
            and selected[release['lifecycle_source_path']] == LIFECYCLE_SHA
            and selected[release['original_trainer_path']] == TRAINER_SHA
            and selected[release['adapter_path']] == release['adapter_sha256']
            and selected[release['auditor_path']] == release['auditor_sha256']
            and selected[release['migration_protocol_path']] == release['migration_protocol_sha256'],
            'Operational source and amendment bindings differ')
    environment = scoped.python_environment(Path(release['python_environment']['lexical_path']))
    require(environment == release['python_environment'], 'Exact lexical Python/venv environment required')
    for key in ('lexical_path', 'resolved_binary_path', 'pyvenv_config_path'):
        path = environment.get(key)
        if path is not None:
            require(path in pins, 'Runtime missing from pin map')
            selected[path] = pins[path]
    selected[str(release_path)] = release_sha
    scoped.verify_files(selected)
    return selected


def record_reaped_only(directory, job, external, manifest, protocol_sha, gpu_uuid):
    require(external['exit_code'] == 0 and external['signals'] == [], 'Recovery child failed or needed signals')
    return {'id': job['id'], 'directory': job['output_directory'], 'external': external}


def verify_replay_receipt(release, job, external):
    """Operational receipt only; replay state never becomes a training input."""
    path = canonical(job['replay_output_directory']) / 'receipt.json'
    receipt = read(path)
    require(receipt.get('schema') == 'adaptgns_sand_migration_parent_replay_receipt_v1'
            and receipt.get('status') == 'passed_exact_restore_and_repeated_current_runtime_replay',
            'Complete exact-restore repeated-replay receipt required')
    expected = {'job': job['id'], 'original_host': release['original_host'], 'actual_host': release['host'],
        'origin_directory': job['origin_directory'], 'origin_checkpoint_sha256': job['origin_checkpoint_sha256'],
        'origin_protocol_sha256': job['origin_protocol_sha256'], 'origin_pointer_sha256': job['origin_pointer_sha256'],
        'origin_stdout_sha256': job['origin_stdout_sha256'],
        'original_runtime': job['original_runtime'], 'actual_runtime': job['actual_runtime'],
        'adapter_sha256': release['adapter_sha256'], 'root_release_sha256': release['_release_sha256'],
        'original_frozen_trainer_sha256': TRAINER_SHA, 'allowed_runtime_differences': ['uuid'],
        'origin_run_config_is_historical_lineage': True, 'model_initialization_from_replay_artifact': False,
        'parent_completed_steps': 50000, 'target_completed_steps': 100000, 'restored_parent_steps': 50000,
        'replay_steps_per_repeat': 100, 'repeats': 2, 'absolute_schedule_steps': list(range(50000, 50100)),
        'exact_model_optimizer_rng_restoration': True, 'exact_repeated_state_and_schedule': True,
        'no_scientific_checkpoint_written': True, 'cross_physical_device_bitwise_equivalence_claim': False}
    require(all(encode(receipt.get(k)) == encode(v) for k, v in expected.items()), 'Replay job/runtime/parent/release lineage differs')
    hashes = receipt.get('replay_state_sha256')
    require(isinstance(hashes, list) and len(hashes) == 2 and all(digest(v) for v in hashes) and hashes[0] == hashes[1],
            'Two identical complete replay state digests required')
    require(receipt.get('exact_original_50100_scalar_match') is True
            and isinstance(receipt.get('original_50100_scalar'), dict)
            and encode(receipt['original_50100_scalar']) == encode(receipt.get('replayed_50100_scalar'))
            and receipt.get('original_50100_model_optimizer_rng_not_saved') is True,
            'Exact recorded original 50100 scalar match required without invented original state')
    require(stamp(external['started_utc']) <= stamp(receipt['completed_utc']) <= stamp(external['reaped_utc'])
            and not any(p.suffix == '.pt' for p in path.parent.iterdir()),
            'Replay must finish within owned lifetime without scientific checkpoints')
    return {'id': job['id'], 'replay_receipt_path': str(path), 'replay_receipt_sha256': sha(path),
            'replay_receipt': receipt, 'external': external, 'scientific_promotion_from_replay': False}


def runtime_type(scoped):
    class Runtime(scoped.Runtime):
        def capture(self, args, child):
            pass  # No fabricated step-zero capture: immutable origin 50k is adapter-bound.

        def process_inventory(self):
            names = scoped.OLD_BROAD_SOURCES | scoped.TRAINING_SOURCES | {
                'supervise_sand_scoped_science_v1.py', 'train_sand_runtime_migration_recovery_v1.py',
                'supervise_sand_runtime_migration_recovery_v1.py', 'supervise_sand_runtime_migration_recovery_v2.py'}
            rows = []
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():
                    continue
                try:
                    row = self.identity(int(path.name))
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if scoped.executed_python_script(row['argv']) in names:
                    rows.append(row)
            return rows
    return Runtime


def check_processes(context, rt, children, owner_identity, record=None, path=None):
    """Strict identities; never scan across our own GPU query fork/exec window."""
    require(rt.future is None, 'Process scan requires completed and consumed GPU query')
    own = {c['process'].pid: c for c in children}
    rows = rt.process_inventory()
    def reject(message, row):
        if record is not None:
            record['rejected_process_inventory'] = {'checked_utc': rt.now().isoformat(),
                'rejected_row': row, 'full_inventory': rows, 'reason': message}
            if path is not None:
                context.scoped.write(path, record)
        raise ValueError(message)
    for row in rows:
        if row['pid'] == owner_identity['pid']:
            if not (row['start_ticks'] == owner_identity['start_ticks'] and row['argv'] == owner_identity['argv']):
                reject('Owner identity changed', row)
            continue
        child = own.get(row['pid'])
        if not (child is not None and child['process'].returncode is None
                and row['ppid'] == owner_identity['pid'] and row['argv'] == child['command']
                and child['identity'] is not None and row['start_ticks'] == child['identity']['start_ticks']):
            reject('Foreign, duplicate or old scientific owner/worker exists on recovery host', row)
    return rows


def run_owned(args, context, rt, auditor):
    S, B = context.scoped, context.B
    children, reaped, abort = [], [], [None]
    started = rt.mono()
    replay_end = getattr(context, 'entry_monotonic', ENTRY_MONOTONIC) + context.replay_seconds if context.mode == 'replay' else None
    def work_check():
        require(not abort[0], abort[0] or 'Owner interrupted')
        require(rt.now() < context.cleanup, 'Training cleanup boundary reached')
        if replay_end is not None:
            require(rt.mono() < replay_end - CLEANUP
                    and rt.now() + timedelta(seconds=context.error + CLEANUP) < REPLAY_STOP,
                    'Replay reserved cleanup window reached')
    def audit_check():
        require(rt.now() + timedelta(seconds=context.error) < context.stop, 'Endpoint audit reached training stop')
        if replay_end is not None:
            require(rt.mono() < replay_end and rt.now() + timedelta(seconds=context.error) <= REPLAY_STOP,
                    'Replay publication deadline reached')
    record = {'schema': SCHEMA, 'wave': 'A', 'mode': context.mode, 'state': 'preflight', 'jobs': [],
              'observations': [], 'started_utc': rt.now().isoformat(), 'automatic_retry': False,
              'old_host_pids_never_controlled': True, 'original_six_model_study_preserved': True}
    path = args.output_dir / 'wave_A.json'
    launch = {'gpu_uuids': context.uuids, 'files_sha256': {'protocol': context.release['files_sha256'][context.release['protocol']]}}
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(sig, frame):
        abort[0] = abort[0] or 'Owner received ' + signal.Signals(sig).name
    for s in previous:
        signal.signal(s, interrupted)
    try:
        S.verify_files(context.bindings)
        owner = rt.identity(os.getpid())
        expected = [context.release['python_environment']['lexical_path'], '-B', str(Path(__file__).resolve()),
                    '--execute', '--mode', context.mode, '--release', str(args.release), '--release-sha256', args.release_sha256]
        require(owner['argv'] == expected and owner['executable'] == context.release['python_environment']['resolved_binary_path'],
                'Exact root owner invocation/native interpreter required')
        record['owner_identity'] = owner
        record['initial_processes'] = check_processes(context, rt, children, owner, record, path)
        rows = rt.gpu_now()
        record['initial_gpu_inventory'] = rows
        record['initial_gpu_classification'] = S.check_gpu(rows, {}, B, context.uuids, [0, 1, 2, 3], rt.gpu_identity)
        S.write(path, record)
        for job in context.jobs:
            work_check()
            S.verify_metadata(context)
            check_processes(context, rt, children, owner, record, path)
            require(rt.now() + timedelta(seconds=context.error) <= context.latest
                    and rt.now() + timedelta(seconds=TRAINING_SECONDS) <= context.cleanup,
                    'Every spawn must retain the full unchanged training allocation')
            require(all(-context.error <= (rt.now() - stamp(context.release[k])).total_seconds() <= 300
                        for k in ('clock_checked_utc', 'process_checked_utc')),
                    'Process and clock evidence expired before child spawn')
            if context.mode == 'replay':
                require(rt.now() + timedelta(seconds=context.error) <= stamp(context.release['replay_latest_start_utc']),
                        'Replay child launch window closed')
            command = fixed_command(context, args.release, args.release_sha256, job)
            begin, at = rt.mono(), rt.now().isoformat()
            process, handles, out, err = rt.launch(args, job, command)
            child = {'job': job, 'process': process, 'identity': None, 'command': command, 'handles': handles,
                     'signals': [], 'started': begin, 'started_utc': at, 'stdout_file': out, 'stderr_file': err,
                     'initial_pointer': None}
            children.append(child)  # Register before identity/publication can fail.
            identity = rt.identity(process.pid)
            child['identity'] = identity
            require(identity['pid'] == process.pid and identity['ppid'] == owner['pid']
                    and identity['argv'] == command and identity['executable'] == owner['executable']
                    and type(identity['start_ticks']) is int and identity['start_ticks'] > 0,
                    'Spawned recovery child native identity differs')
            S.write(args.output_dir / 'logs' / (job['id'] + '.launch.json'),
                    {'job': job, 'identity': identity, 'command': command, 'started_utc': at})
        record['state'] = 'running'
        last_query = last_status = -math.inf
        while any(c['process'].returncode is None for c in children):
            work_check()
            for child in children:
                if child['process'].returncode is None:
                    failure = rt.reap(args, child, record, reaped, launch, {})
                    require(failure is None, failure or 'Recovery child failed')
            observed = rt.gpu_result()
            if observed is not None:
                rows, known = observed
                entry = {'checked_utc': rt.now().isoformat(), 'gpu_processes': rows, 'classification': 'pending'}
                record['observations'].append(entry)
                S.write(path, record)
                entry['classification'] = S.check_gpu(rows, known, B, context.uuids, [0, 1, 2, 3], rt.gpu_identity)
            # gpu_result clears a future only after subprocess.run returned and
            # reaped nvidia-smi. Scan before the next query; while a query is
            # pending, keep reaping/checking deadlines without process-scanning.
            if rt.future is None and rt.mono() - last_status >= 30:
                S.verify_metadata(context)
                record['latest_processes'] = check_processes(context, rt, children, owner, record, path)
                record['latest_observed_utc'] = rt.now().isoformat()
                S.write(path, record)
                last_status = rt.mono()
            if rt.future is None and rt.mono() - last_query >= 30:
                rt.gpu_submit({c['process'].pid: c for c in children if c['process'].returncode is None})
                last_query = rt.mono()
            rt.sleep(.2)
        require(len(reaped) == 4, 'All four original A recovery jobs must be reaped')
    except BaseException as error:
        abort[0] = abort[0] or type(error).__name__ + ': ' + str(error)
    finally:
        # A preflight/work alarm must not interrupt the bounded owned-child cleanup.
        # The frozen cleanup loop itself spends at most 15 seconds.
        if context.mode == 'replay':
            signal.setitimer(signal.ITIMER_REAL, 0)
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        try:
            S.cleanup(args, context, rt, children, record, reaped, launch, {})
        except BaseException as error:
            abort[0] = abort[0] or 'Cleanup: ' + str(error)
            record['unreaped_owned_children'] = [{'pid': c['process'].pid, 'identity': c['identity']}
                for c in children if c['process'].returncode is None]
        for child in children:
            for handle in child['handles']:
                handle.close()
        try:
            rt.close()
            observed = rt.gpu_result()
            if observed is not None:
                rows, known = observed
                record['final_pending_gpu_query'] = {'gpu_processes': rows,
                    'classification': S.check_gpu(rows, known, B, context.uuids, [0, 1, 2, 3], rt.gpu_identity)}
        except BaseException as error:
            abort[0] = abort[0] or 'Runtime close: ' + str(error)
        for s, handler in previous.items():
            signal.signal(s, handler)
        if context.mode == 'replay' and replay_end is not None:
            remaining = min(replay_end - rt.mono(), (REPLAY_STOP - rt.now()).total_seconds() - context.error)
            if remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining)
    record.update(state='stopped_pending_audit' if abort[0] is None else 'failed', error=abort[0],
                  all_children=[{'job': c['job'], 'pid': c['process'].pid, 'identity': c['identity'],
                      'command': c['command'], 'exit_code': c['process'].returncode, 'signals': c['signals']} for c in children],
                  never_started=[j['id'] for j in context.jobs if j['id'] not in {c['job']['id'] for c in children}],
                  ended_utc=rt.now().isoformat(), elapsed_seconds=rt.mono() - started)
    S.write(path, record)
    try:
        require(abort[0] is None and not record['unreaped_owned_children'], 'Incomplete recovery retained without retry')
        audit_check()
        rows = rt.gpu_now()
        record['final_gpu_inventory'] = rows
        record['final_gpu_classification'] = S.check_gpu(rows, {}, B, context.uuids, [0, 1, 2, 3], rt.gpu_identity)
        record['final_processes'] = check_processes(context, rt, [], owner, record, path)
        S.verify_files(context.bindings)
        audit_check()
        closure = {'verified': True, 'checked_utc': rt.now().isoformat(),
                   'owned_children_reaped': True, 'rows': rows}
        verified = []
        for result in reaped:
            job = next(j for j in context.jobs if j['id'] == result['id'])
            audit_check()
            external = dict(result['external'], reaped_utc=result['external']['ended_utc'],
                            stdout_sha256=sha(result['external']['stdout_file']),
                            stderr_sha256=sha(result['external']['stderr_file']), scoped_gpu_closure=closure,
                            owner_release_sha256=args.release_sha256, owner_identity=owner,
                            training_stop_utc=context.stop.isoformat(), clock_error_bound_seconds=context.error)
            if context.mode == 'replay':
                verified.append(verify_replay_receipt(context.release, job, external))
            else:
                verified.append(auditor.audit_recovery_job(context.release, job, external=external))
        if context.mode == 'train':
            audit_check()
            record['pairing'] = auditor.audit_recovery_pairing(verified)
        S.verify_files(context.bindings)
        audit_check()
        record.update(state='verified_replay_only' if context.mode == 'replay' else 'verified_recovery_endpoints',
                      verified_jobs=verified, scientific_promotion_from_replay=False,
                      whole_six_model_cohort_admitted=False, audit_ended_utc=rt.now().isoformat())
        S.write(path, record)
        return record
    except BaseException as error:
        record.update(state='failed', error=record['error'] or type(error).__name__ + ': ' + str(error))
        S.write(path, record)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=['replay', 'train'])
    parser.add_argument('--release', type=Path)
    parser.add_argument('--release-sha256')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema': SCHEMA, 'status': 'description_only', 'launches': 0,
                          'original_A_jobs': JOBS, 'training_allocation_seconds': TRAINING_SECONDS,
                          'latest_start_utc': LATEST_START.isoformat(), 'training_stop_utc': TRAINING_STOP.isoformat(),
                          'compute_analysis_deadline_utc': DEADLINE.isoformat()}))
        return 0
    return execute(args)


def execute(args):
    require(args.mode is not None and args.release is not None and digest(args.release_sha256), 'Explicit mode/release/hash required')
    require(sys.platform.startswith('linux') and hasattr(os, 'wait4') and os.environ.get('CUDA_VISIBLE_DEVICES') is None,
            'Unremapped Linux runtime required')
    args.release = canonical(args.release)
    release = snapshot(args.release, args.release_sha256)
    previous_alarm = signal.getsignal(signal.SIGALRM)
    if args.mode == 'replay':
        require(release.get('replay_allocation_seconds') == REPLAY_SECONDS
                and stamp(release['replay_stop_utc']) == REPLAY_STOP
                and release.get('replay_handoff_reserve_seconds') == REPLAY_HANDOFF,
                'Exact replay bounds required before any preflight work')
        remaining = min(ENTRY_MONOTONIC + REPLAY_SECONDS - CLEANUP - time.perf_counter(),
                        (REPLAY_STOP - utc()).total_seconds() - CLEANUP - 5)
        require(remaining > 0, 'Whole replay preflight already reached cleanup reserve')
        def expired(sig, frame):
            raise TimeoutError('Whole replay entry-based deadline reached; cleanup reserve retained')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        return execute_validated(args, release)
    finally:
        if args.mode == 'replay':
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)


def execute_validated(args, release):
    scoped = private_import(release['scoped_owner_path'], SCOPED_SHA, '_recovery_frozen_scoped_owner')
    lifecycle = private_import(release['lifecycle_source_path'], LIFECYCLE_SHA, '_recovery_frozen_lifecycle')
    bindings = operational_bindings(release, args.release, args.release_sha256, scoped)
    context = validate_contract(release, args.mode, utc(), scoped, lifecycle)
    os.environ.update(ENVIRONMENT)
    adapter = private_import(release['adapter_path'], release['adapter_sha256'], '_recovery_adapter')
    validated = adapter.validate_release(args.release, args.release_sha256, args.mode)
    require(validated == dict(release, _release_path=str(args.release), _release_sha256=args.release_sha256),
            'Adapter release contract differs')
    context.release = validated
    for job in context.jobs:
        require(adapter.select_job(release, job['id']) == job, 'Adapter fixed job selection differs')
    args.output_dir = output_paths(context)
    # Root source is imported after exact release validation; description never imports it.
    audit = private_import(release['auditor_path'], release['auditor_sha256'], '_recovery_endpoint_auditor') if args.mode == 'train' else None
    context.bindings = bindings
    context.metadata = {p: scoped.file_identity(p) for p in bindings}
    args.output_dir.mkdir(mode=0o700)
    for name in ('logs', 'jobs'):
        (args.output_dir / name).mkdir()
    scoped.write(args.output_dir / 'owner_started.json', {'schema': SCHEMA, 'owner_pid': os.getpid(),
        'owner_release_sha256': args.release_sha256, 'mode': args.mode, 'started_utc': utc().isoformat(),
        'entry_monotonic': ENTRY_MONOTONIC, 'training_allocation_seconds': TRAINING_SECONDS,
        'replay_allocation_seconds': context.replay_seconds, 'operational_input_sha256': bindings})
    lifecycle.verify_job = record_reaped_only
    rt = runtime_type(scoped)(SimpleNamespace(B=lifecycle))
    try:
        record = run_owned(args, context, rt, audit)
        scoped.write(args.output_dir / 'owner_terminal.json', {'schema': SCHEMA, 'status': 'complete_stopped_and_reaped',
            'mode': args.mode, 'owner_release_sha256': args.release_sha256,
            'wave_sha256': sha(args.output_dir / 'wave_A.json'), 'terminal_utc': utc().isoformat(),
            'scientific_promotion_from_replay': False, 'whole_six_model_cohort_admitted': False})
        print(json.dumps({'status': 'complete_stopped_and_reaped', 'wrapper_exit_code': 0, 'owner_output_directory': str(args.output_dir)}))
        return 0
    except BaseException as error:
        scoped.write(args.output_dir / 'owner_terminal.json', {'schema': SCHEMA, 'status': 'failed_requires_root_review',
            'mode': args.mode, 'error_type': type(error).__name__, 'error': str(error),
            'owner_release_sha256': args.release_sha256, 'terminal_utc': utc().isoformat(),
            'automatic_retry': False, 'all_outcomes_retained': True})
        print(json.dumps({'status': 'failed_requires_root_review', 'wrapper_exit_code': 1}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
