"""Synthetic scalar control/process fixtures only; no CUDA, data or checkpoints."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('migration_owner_tested', HERE / 'supervise_sand_runtime_migration_recovery_v2.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
S = M.private_import(HERE / 'supervise_sand_scoped_science_v1.py', M.SCOPED_SHA, '_test_frozen_scoped')
B = M.private_import(HERE / 'measure_sand_cuda_capacity_v2.py', M.LIFECYCLE_SHA, '_test_frozen_lifecycle')


class FakeRuntime:
    def __init__(self, context, args, now, behavior='complete'):
        self.context, self.args, self.start, self.behavior = context, args, now, behavior
        self.t, self.created, self.future, self.stopped = 0., [], None, []
        self.closed = False
        self.owner_command = [sys.executable, '-B', str(HERE / 'supervise_sand_runtime_migration_recovery_v2.py'),
            '--execute', '--mode', context.mode, '--release', str(args.release), '--release-sha256', args.release_sha256]

    def now(self): return self.start + timedelta(seconds=self.t)
    def mono(self): return self.t
    def sleep(self, seconds):
        self.t += seconds
        if self.behavior in ('hang', 'refuse') and self.t < 1:
            self.t = (self.context.cleanup - self.start).total_seconds() + 1

    def identity(self, pid):
        if pid == os.getpid():
            return {'pid': pid, 'ppid': 1, 'start_ticks': 42, 'argv': self.owner_command,
                    'state': 'R', 'executable': str(Path(sys.executable).resolve())}
        for row in self.created:
            if row['process'].pid == pid:
                if row['process'].returncode is not None: raise FileNotFoundError()
                return {'pid': pid, 'ppid': os.getpid(), 'start_ticks': 100 + pid,
                        'argv': row['command'], 'state': 'R', 'executable': str(Path(sys.executable).resolve())}
        raise FileNotFoundError()

    def gpu_identity(self, pid): return self.identity(pid)
    def process_inventory(self):
        if self.behavior == 'late': self.t = (self.context.latest - self.start).total_seconds() + 1
        return [self.identity(os.getpid())] + [self.identity(c['process'].pid) for c in self.created if c['process'].returncode is None]
    def gpu_now(self):
        return [{'pid': 987654, 'gpu_uuid': self.context.uuids[0]}] if self.behavior == 'foreign' else []
    def gpu_submit(self, owned): self.future = (self.gpu_now(), dict(owned))
    def gpu_result(self): result, self.future = self.future, None; return result
    def launch(self, args, job, command):
        if self.behavior == 'launch_failure' and self.created: raise RuntimeError('synthetic launch failure')
        process = SimpleNamespace(pid=5000 + len(self.created), returncode=None)
        out, err = args.output_dir / 'logs' / (job['id'] + '.stdout'), args.output_dir / 'logs' / (job['id'] + '.stderr')
        handles = out.open('xb'), err.open('xb')
        self.created.append({'job': job, 'command': command, 'process': process, 'started': self.t})
        return process, handles, str(out), str(err)
    def capture(self, args, child): pass
    def reap(self, args, child, record, reaped, launch, manifest):
        if self.behavior == 'refuse': return None
        if self.behavior == 'hang' and not any(s['signal'] == 'SIGKILL' for s in child['signals']): return None
        if self.behavior == 'launch_failure' and not child['signals']: return None
        if self.behavior == 'complete' and self.t - child['started'] < .3: return None
        code = -9 if child['signals'] else 0
        child['process'].returncode = code
        for handle in child['handles']: handle.close()
        external = {'pid': child['process'].pid, 'identity': child['identity'], 'command': child['command'],
            'started_utc': child['started_utc'], 'ended_utc': self.now().isoformat(), 'exit_code': code,
            'signals': child['signals'], 'stdout_file': child['stdout_file'], 'stderr_file': child['stderr_file']}
        record['jobs'].append({'id': child['job']['id'], 'external': external})
        if code: return 'synthetic nonzero child'
        reaped.append({'id': child['job']['id'], 'external': external})
    def stop(self, child, sig):
        self.stopped.append(child['process'].pid)
        child['signals'].append({'signal': signal.Signals(sig).name, 'result': 'synthetic_owned_only'})
    def close(self): self.closed = True


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.now = datetime(2026, 10, 6, 14, 40, tzinfo=timezone.utc)
        self.uuids = [f'00000000-0000-0000-0000-{i:012x}' for i in range(4)]
        jobs = []
        for identity, arm, seed, gpu in M.JOBS:
            actual = {'uuid': self.uuids[gpu], 'device': 'cuda:' + str(gpu), 'name': 'NVIDIA GB200', 'tf32': False}
            jobs.append({'id': identity, 'arm': arm, 'seed': seed, 'gpu': gpu,
                'original_runtime': dict(actual, uuid='old-' + str(gpu)), 'actual_runtime': actual,
                'origin_directory': str(self.root / 'original' / 'jobs' / identity),
                'output_directory': str(self.root / ('train-' + identity)),
                'replay_output_directory': str(self.root / ('replay-' + identity))})
        process = {'schema': 'adaptgns_sand_runtime_migration_process_check_v1', 'issued_by': 'root',
            'host': M.socket.gethostname(), 'original_host': 'synthetic-original-host', 'checked_utc': self.now.isoformat(),
            'all_original_owned_processes_absent': True, 'new_host_idle': True, 'gpu_uuids': self.uuids, 'gpu_processes': [],
            'original_owned_processes': [{'pid': 100 + i, 'start_ticks': 200 + i, 'executable': '/synthetic/python',
                'argv': ['python3', '/synthetic/' + ('supervise_sand_scoped_science_v1.py' if i == 0 else 'train_sand_graph_support_cuda.py')],
                'absent': True} for i in range(5)]}
        clock = {'schema': 'adaptgns_sand_runtime_migration_clock_check_v1', 'issued_by': 'root',
            'host': M.socket.gethostname(), 'checked_utc': self.now.isoformat(), 'root_host_samples_reviewed': True,
            'clock_error_bound_seconds': 1, 'clock_sample': {'root_start_utc': (self.now - timedelta(seconds=.5)).isoformat(),
                'root_end_utc': (self.now + timedelta(seconds=.5)).isoformat(), 'remote_sample_utc': self.now.isoformat(),
                'offset_lower_seconds': -.5, 'offset_upper_seconds': .5}}
        self.process_path, self.clock_path = self.root / 'process.json', self.root / 'clock.json'
        self.process_path.write_bytes(M.encode(process)); self.clock_path.write_bytes(M.encode(clock))
        self.release = {'schema': M.RELEASE_SCHEMA, 'issued_by': 'root', 'status': 'approved_for_sand_migration_training',
            'host': M.socket.gethostname(), 'original_host': 'synthetic-original-host', 'jobs': jobs,
            'training_allocation_seconds': M.TRAINING_SECONDS, 'post_training_allocation_seconds': M.POSTTRAINING_SECONDS,
            'training_stop_utc': M.TRAINING_STOP.isoformat(), 'compute_analysis_deadline_utc': M.DEADLINE.isoformat(),
            'latest_start_utc': M.LATEST_START.isoformat(), 'clock_error_bound_seconds': 1,
            'clock_checked_utc': self.now.isoformat(), 'process_checked_utc': self.now.isoformat(),
            'gpu_uuids': self.uuids, 'owned_gpu_indices': [0, 1, 2, 3], 'six_model_freeze_before_test': True,
            'required_models': 6, 'required_policies': 6, 'automatic_retry_resume_or_promotion': False,
            'process_check_path': str(self.process_path), 'clock_check_path': str(self.clock_path),
            'replay_allocation_seconds': 600, 'replay_stop_utc': M.REPLAY_STOP.isoformat(),
            'replay_latest_start_utc': M.REPLAY_LATEST.isoformat(), 'replay_handoff_reserve_seconds': 180,
            'adapter_path': str(self.root / 'train_sand_runtime_migration_recovery_v1.py'),
            'python_environment': {'lexical_path': sys.executable, 'resolved_binary_path': str(Path(sys.executable).resolve())},
            'protocol': str(self.root / 'protocol.md'), 'files_sha256': {str(self.root / 'protocol.md'): 'f' * 64}}

    def tearDown(self): self.temp.cleanup()

    def context(self, behavior='complete', mode='train'):
        r = copy.deepcopy(self.release)
        if mode == 'replay': r['status'] = 'approved_for_sand_migration_replay'
        c = M.validate_contract(r, mode, self.now, S, B)
        c.entry_monotonic = 0.
        c.bindings = {str(self.process_path): M.sha(self.process_path)}
        c.metadata = {p: S.file_identity(p) for p in c.bindings}
        output = self.root / 'owner'; output.mkdir(); (output / 'logs').mkdir(); (output / 'jobs').mkdir()
        release_path = self.root / 'release.json'; release_path.write_bytes(M.encode(r))
        args = SimpleNamespace(output_dir=output, release=release_path, release_sha256=M.sha(release_path))
        return args, c, FakeRuntime(c, args, self.now, behavior)

    def test_description_never_imports_or_launches(self):
        with patch.object(M, 'private_import', side_effect=AssertionError('not allowed')):
            self.assertEqual(M.main([]), 0)

    def test_runtime_only_uuid_may_differ_including_types(self):
        original = {'uuid': 'old', 'precision': False}
        M.same_runtime_except_uuid(original, {'uuid': 'new', 'precision': False})
        for other in ({'uuid': 'new', 'precision': 0}, {'uuid': 'new', 'precision': False, 'extra': 1}):
            with self.assertRaises(ValueError): M.same_runtime_except_uuid(original, other)

    def test_fixed_study_schedule_and_full_training_allocation(self):
        for field, value in [('required_models', 4), ('required_policies', 5), ('training_allocation_seconds', 13000),
                             ('replay_allocation_seconds', 599), ('replay_allocation_seconds', 601),
                             ('replay_handoff_reserve_seconds', 179),
                             ('replay_stop_utc', (M.REPLAY_STOP + timedelta(seconds=1)).isoformat()),
                             ('replay_latest_start_utc', (M.REPLAY_LATEST + timedelta(seconds=1)).isoformat())]:
            bad = dict(self.release, **{field: value})
            with self.assertRaises(ValueError): M.validate_contract(bad, 'train', self.now, S, B)
        bad = copy.deepcopy(self.release); bad['jobs'][0]['gpu'] = 1
        with self.assertRaises(ValueError): M.validate_contract(bad, 'train', self.now, S, B)

    def test_replay_deadline_and_clock_freshness(self):
        r = dict(self.release, status='approved_for_sand_migration_replay')
        self.assertEqual(M.validate_contract(r, 'replay', self.now, S, B).replay_seconds, 600)
        with self.assertRaises(ValueError): M.validate_contract(r, 'replay', M.REPLAY_LATEST, S, B)
        with self.assertRaises(ValueError): M.validate_contract(r, 'replay', self.now + timedelta(seconds=301), S, B)

    def test_four_jobs_audited_only_after_reap_and_scoped_closure(self):
        args, c, rt = self.context(); audited = []
        def audit(release, job, *, external):
            self.assertTrue(all(row['process'].returncode == 0 for row in rt.created))
            self.assertTrue(external['scoped_gpu_closure']['owned_children_reaped'])
            self.assertEqual(external['scoped_gpu_closure']['rows'], [])
            self.assertEqual(external['stdout_sha256'], M.sha(external['stdout_file']))
            self.assertEqual(external['owner_release_sha256'], args.release_sha256)
            audited.append(job['id']); return {'id': job['id']}
        auditor = SimpleNamespace(audit_recovery_job=audit, audit_recovery_pairing=lambda jobs: {'count': len(jobs)})
        with patch.object(S.subprocess, 'Popen', side_effect=AssertionError('real launch forbidden')):
            result = M.run_owned(args, c, rt, auditor)
        self.assertEqual(result['state'], 'verified_recovery_endpoints')
        self.assertEqual(len(audited), 4); self.assertTrue(rt.closed); self.assertFalse(rt.stopped)
        self.assertTrue(all(row['command'][1] == '-B' for row in rt.created))

    def test_partial_launch_failure_cleans_only_registered_local_child(self):
        args, c, rt = self.context('launch_failure')
        with self.assertRaises(ValueError): M.run_owned(args, c, rt, None)
        result = M.read(args.output_dir / 'wave_A.json')
        self.assertEqual(len(rt.created), 1); self.assertEqual(set(rt.stopped), {5000})
        self.assertEqual(len(result['never_started']), 3); self.assertEqual(result['state'], 'failed')

    def test_foreign_gpu_prevents_launch_without_signals(self):
        args, c, rt = self.context('foreign')
        with self.assertRaises(ValueError): M.run_owned(args, c, rt, None)
        self.assertFalse(rt.created); self.assertFalse(rt.stopped)

    def test_late_prelaunch_scan_blocks_all_jobs(self):
        args, c, rt = self.context('late')
        with self.assertRaises(ValueError): M.run_owned(args, c, rt, None)
        self.assertFalse(rt.created)

    def test_hung_children_reaped_or_retained_after_bounded_cleanup(self):
        for behavior in ('hang', 'refuse'):
            with self.subTest(behavior=behavior):
                if (self.root / 'owner').exists():
                    (self.root / 'owner').rename(self.root / 'previous-owner')
                args, c, rt = self.context(behavior)
                with self.assertRaises(ValueError): M.run_owned(args, c, rt, None)
                result = M.read(args.output_dir / 'wave_A.json')
                self.assertEqual(len(rt.created), 4)
                self.assertEqual(bool(result['unreaped_owned_children']), behavior == 'refuse')
                self.assertTrue(set(rt.stopped) <= {5000, 5001, 5002, 5003})

    def test_frozen_lifecycle_refuses_pid_reuse_without_signalling(self):
        child = {'process': SimpleNamespace(pid=4321, returncode=None), 'identity': {'start_ticks': 1},
                 'command': ['python', 'owned.py'], 'signals': []}
        current = {'pid': 4321, 'ppid': os.getpid(), 'start_ticks': 2, 'argv': child['command'], 'state': 'R'}
        with patch.object(B, 'process_identity', return_value=current), patch.object(B.os, 'killpg') as kill:
            B.stop_owned([child], signal.SIGTERM)
        kill.assert_not_called(); self.assertEqual(child['signals'][0]['result'], 'refused_or_failed')

    def test_preflight_timeout_counts_time_spent_since_entry(self):
        args = SimpleNamespace(mode='replay', release=self.root / 'release.json', release_sha256='a' * 64)
        args.release.write_text('{}')
        with patch.object(M.sys, 'platform', 'linux'), patch.object(M, 'snapshot', return_value=self.release), \
             patch.object(M, 'ENTRY_MONOTONIC', 100.), patch.object(M.time, 'perf_counter', return_value=200.), \
             patch.object(M, 'utc', return_value=self.now), patch.object(M.signal, 'setitimer') as timer, \
             patch.object(M, 'execute_validated', return_value=0):
            self.assertEqual(M.execute(args), 0)
        self.assertEqual(timer.call_args_list[0].args, (signal.ITIMER_REAL, 485.))
        with patch.object(M.sys, 'platform', 'linux'), patch.object(M, 'snapshot', return_value=self.release), \
             patch.object(M, 'ENTRY_MONOTONIC', 100.), patch.object(M.time, 'perf_counter', return_value=686.), \
             patch.object(M, 'utc', return_value=self.now), patch.object(M, 'execute_validated') as preflight:
            with self.assertRaises(ValueError): M.execute(args)
        preflight.assert_not_called()

    def test_original_scientific_tree_sibling_cannot_be_any_output(self):
        r = copy.deepcopy(self.release)
        r.update(owner_output_directory=str(self.root / 'owner'), repo=str(self.root / 'repo'),
                 train_manifest=str(self.root / 'dataset' / 'train.json'))
        context = SimpleNamespace(release=r, jobs=r['jobs'], mode='train')
        self.assertEqual(M.output_paths(context), self.root / 'owner')
        for target in ('owner', 'training', 'replay'):
            with self.subTest(target=target):
                bad = copy.deepcopy(r)
                path = str(self.root / 'original' / ('new-' + target))
                if target == 'owner': bad['owner_output_directory'] = path
                else: bad['jobs'][0]['output_directory' if target == 'training' else 'replay_output_directory'] = path
                with self.assertRaises(ValueError):
                    M.output_paths(SimpleNamespace(release=bad, jobs=bad['jobs'], mode='train'))

    def test_replay_receipt_rejects_wrong_lineage_reduced_steps_and_mismatch(self):
        r = copy.deepcopy(self.release)
        r.update(_release_sha256='b' * 64, adapter_sha256='c' * 64)
        job = r['jobs'][0]
        for key in ('origin_checkpoint_sha256', 'origin_protocol_sha256', 'origin_pointer_sha256', 'origin_stdout_sha256'):
            job[key] = 'a' * 64
        receipt = {'schema': 'adaptgns_sand_migration_parent_replay_receipt_v1',
            'status': 'passed_exact_restore_and_repeated_current_runtime_replay', 'job': job['id'],
            'original_host': r['original_host'], 'actual_host': r['host'], 'origin_directory': job['origin_directory'],
            **{key: job[key] for key in ('origin_checkpoint_sha256', 'origin_protocol_sha256', 'origin_pointer_sha256', 'origin_stdout_sha256')},
            'original_runtime': job['original_runtime'], 'actual_runtime': job['actual_runtime'],
            'adapter_sha256': r['adapter_sha256'], 'root_release_sha256': r['_release_sha256'],
            'original_frozen_trainer_sha256': M.TRAINER_SHA, 'allowed_runtime_differences': ['uuid'],
            'origin_run_config_is_historical_lineage': True, 'model_initialization_from_replay_artifact': False,
            'parent_completed_steps': 50000, 'target_completed_steps': 100000, 'restored_parent_steps': 50000,
            'replay_steps_per_repeat': 100, 'repeats': 2, 'absolute_schedule_steps': list(range(50000, 50100)),
            'exact_model_optimizer_rng_restoration': True, 'exact_repeated_state_and_schedule': True,
            'no_scientific_checkpoint_written': True, 'cross_physical_device_bitwise_equivalence_claim': False,
            'replay_state_sha256': ['d' * 64, 'd' * 64], 'exact_original_50100_scalar_match': True,
            'original_50100_scalar': {'loss': .1}, 'replayed_50100_scalar': {'loss': .1},
            'original_50100_model_optimizer_rng_not_saved': True, 'completed_utc': self.now.isoformat()}
        directory = Path(job['replay_output_directory']); directory.mkdir()
        path = directory / 'receipt.json'; path.write_bytes(M.encode(receipt))
        external = {'started_utc': (self.now - timedelta(seconds=1)).isoformat(),
                    'reaped_utc': (self.now + timedelta(seconds=1)).isoformat()}
        self.assertFalse(M.verify_replay_receipt(r, job, external)['scientific_promotion_from_replay'])
        for field, value in [('job', 'mix_seed2'), ('replay_steps_per_repeat', 99),
                ('exact_original_50100_scalar_match', False), ('replayed_50100_scalar', {'loss': .2}),
                ('replay_state_sha256', ['d' * 64, 'e' * 64])]:
            with self.subTest(field=field):
                path.write_bytes(M.encode(dict(receipt, **{field: value})))
                with self.assertRaises(ValueError): M.verify_replay_receipt(r, job, external)

    def test_replay_alarm_is_suspended_for_cleanup_then_rearmed_to_original_end(self):
        args, c, rt = self.context(mode='replay')
        events = []
        cleanup = S.cleanup
        def observed_cleanup(*args, **kwargs):
            self.assertEqual(events[-1], ('timer', 0))
            events.append(('cleanup', rt.mono()))
            return cleanup(*args, **kwargs)
        with patch.object(M.signal, 'setitimer', side_effect=lambda _, seconds: events.append(('timer', seconds))), \
             patch.object(S, 'cleanup', side_effect=observed_cleanup), \
             patch.object(M, 'verify_replay_receipt', side_effect=lambda release, job, external: {'id': job['id']}):
            result = M.run_owned(args, c, rt, None)
        self.assertEqual(result['state'], 'verified_replay_only')
        self.assertEqual(events[0], ('timer', 0))
        self.assertEqual(events[1][0], 'cleanup')
        self.assertEqual(events[2][0], 'timer')
        self.assertAlmostEqual(events[2][1], 600 - rt.mono())
        self.assertFalse(result['scientific_promotion_from_replay'])


if __name__ == '__main__': unittest.main()
