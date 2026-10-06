"""In-memory synthetic issuance fixtures; no native/clock/array accesses."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'review_actual_sand_final24_issuance_code_audit_v1.py'
spec = importlib.util.spec_from_file_location('synthetic_issuance_auditor', SOURCE)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
T = datetime(2026, 10, 6, 20, tzinfo=timezone.utc)


class Fixture:
    def __init__(self):
        self.pk = Path('/synthetic/package'); self.base = Path('/synthetic/base'); self.state = Path('/synthetic/state')
        self.files = {}; self.reads = []; self.hashes = []; self.symlinks = set()
        self.files[str(SOURCE)] = b'# synthetic reviewer identity binding\n'
        self.sources = {}
        for i in range(23):
            remote = M.R + '/cuda_preparation/frozen_' + str(i).zfill(2) + '.py'
            path = self.pk / 'sources' / (str(i).zfill(2) + '.py')
            self.files[str(path)] = ('# inert source ' + str(i) + '\n').encode()
            self.sources[remote] = self.sha(path)
        roots = (M.R + '/sand_numeric_train_valid_20261006_v1/valid', M.R + '/sand_reserved_preparation_20261006_v1/numeric/test')
        self.arrays = {root + '/' + kind + '_' + str(i).zfill(6) + '.npy': 'd' * 64
                       for root in roots for kind in ('position', 'type') for i in range(30)}
        census = {'status': 'complete_no_duplicates', 'counts': {'train': 1000, 'valid': 30, 'test': 30},
                  'numeric_files_sha256': {**self.arrays, **{M.R + '/train/' + kind + '_' + str(i) + '.npy': 'e' * 64
                  for kind in ('position', 'type') for i in range(1000)}}}
        self.census = Path('/synthetic/evidence/census/all_split_census.json'); self.put(self.census, census)
        self.extra_evidence = Path('/synthetic/evidence/accepted_review.json'); self.put(self.extra_evidence, {'status': 'synthetic accepted'})
        self.originals = {}; self.resolved = {}; commands = {'hosts': {}}
        self.pyenv = {'lexical_path': M.R + '/.venv/bin/python', 'resolved_binary_path': '/usr/bin/python3.12',
                      'binary_sha256': 'a' * 64, 'sys_prefix': M.R + '/.venv', 'sys_base_prefix': '/usr',
                      'pyvenv_config_path': M.R + '/.venv/pyvenv.cfg', 'pyvenv_config_sha256': 'b' * 64}
        self.roles = {}
        for role, count in [('A', 44), ('B', 42)]:
            original_pins = dict(self.sources)
            original_pins.update({M.R + '/inputs/' + role + '_' + str(i) + '.json': 'c' * 64 for i in range(count - 25)})
            pf = {split: {'path': M.R + '/preflight_' + role + '_' + split + '.json', 'sha256': None} for split in ('valid', 'test')}
            original_pins.update({row['path']: None for row in pf.values()})
            owner = [self.pyenv['lexical_path'], M.R + '/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py', '--execute', '--release', M.R + '/controls/' + role + '.json']
            commands['hosts'][role] = {'owner_argv': owner}
            original = {'files_sha256': original_pins, 'split_preflight': pf, 'environment': M.ENV,
                        'stage_quotas_seconds': M.QUOTAS, 'cleanup_seconds_per_invocation': 15,
                        'outer_processing_reserve_seconds': 3600, 'python_environment': self.pyenv,
                        'gpu_uuids': ['GPU-synthetic-' + str(i) for i in range(4)],
                        'gpu_scope': {'owned_indices': [0, 1, 2, 3] if role == 'A' else [2, 3]},
                        'streams': [{'seed': 0, 'arm': 'base', 'commands': [['python', 'evaluator', '--execute']]}],
                        'candidate_metadata': {'execution_authority': False}, 'status': 'unadmitted_concrete_candidate',
                        'issued_by': None, 'preparation_and_cohort_reserves_already_accounted_before_this_queue': False,
                        'clock_error_bound_seconds': None, 'latest_start_utc': None, 'process_clock_checked_utc': None}
            self.originals[role] = original; self.put(self.base / (role + '.evaluation_release.candidate.json'), original)
            pins = {k: v or 'f' * 64 for k, v in original_pins.items()}; pins.update(self.arrays)
            resolved = copy.deepcopy(original); resolved['files_sha256'] = pins
            for split in pf: resolved['split_preflight'][split]['sha256'] = pins[pf[split]['path']]
            candidate_path = role + '.resolved.json'; self.put(self.pk / candidate_path, resolved); self.resolved[role] = resolved
            self.roles[role] = {'files_sha256': pins, 'hostname': 'synthetic-' + role, 'historical_pids': [101, 102],
                                'expected_boot_id': 'synthetic-boot', 'owner_argv': owner,
                                'resolved_candidate': candidate_path, 'resolved_candidate_sha256': self.sha(self.pk / candidate_path)}
        self.put(self.base / 'commands.json', commands)
        self.put(self.base / 'frozen_source_closure.json', {'files_sha256': self.sources})
        self.put(self.base / 'manifest.json', {'files_sha256': {str(Path(path).relative_to(self.base)): self.sha(path) for path in self.files if path.startswith(str(self.base) + '/')}})
        self.base_pin = self.sha(self.base / 'manifest.json')
        prepared = {'candidate_manifest_sha256': self.base_pin, 'root_B_completion_sha256': '9' * 64,
                    'original_local_evidence_sha256': {str(self.census): self.sha(self.census), str(self.extra_evidence): self.sha(self.extra_evidence)},
                    'source_files': {remote: {'sha256': h, 'package_path': 'sources/' + str(i).zfill(2) + '.py'} for i, (remote, h) in enumerate(self.sources.items())},
                    'array_files_sha256': self.arrays, 'roles': self.roles}
        self.put(self.pk / 'prepared_inputs.json', prepared); self.put(self.state / 'prepared_inputs.json', prepared)
        self.prepared_pin = self.sha(self.pk / 'prepared_inputs.json')
        self.put(self.pk / 'manifest.json', {'files_sha256': {str(Path(path).relative_to(self.pk)): self.sha(path) for path in self.files if path.startswith(str(self.pk) + '/')}})
        self.package_pin = self.sha(self.pk / 'manifest.json')
        self.put(self.state / 'operator_binding.json', {'manifest_sha256': self.package_pin, 'local_preparation_only': True, 'clock_read_or_issued': False})
        self.stop = T + timedelta(seconds=600)
        phase = {'schema': 'sand_final24_original_control_phase_v1', 'status': 'approved_one_original_final24_control_phase',
                 'issued_by': 'root', 'prepared_inputs_sha256': self.prepared_pin, 'clock_restarted': False,
                 'control_seconds': 600, 'evaluation_seconds': 11760, 'analysis_seconds': 3600,
                 'started_utc': T.isoformat(), 'stop_utc': self.stop.isoformat(), 'global_analysis_stop_utc': M.GLOBAL_STOP}
        self.put(self.state / 'control_phase.json', phase); self.phase_pin = self.sha(self.state / 'control_phase.json')
        self.put(self.state / 'control_anchor.json', {'phase_sha256': self.phase_pin, 'utc': T.isoformat(), 'monotonic_seconds': 1000.0})
        for role in ('A', 'B'):
            original = self.originals[role]; spec = self.roles[role]
            capture = {'schema': 'sand_final24_remote_control_observation_v1', 'action': 'verify_stage', 'role': role,
                       'hostname': spec['hostname'], 'same_control_stop_utc': self.stop.isoformat(),
                       'input_sha256': spec['files_sha256'], 'native_absent': {'101': True, '102': True},
                       'historical_absence': {'101': True, '102': True}, 'owner': None, 'children': [],
                       'other_matching_science_processes': [], 'gpu_uuids': original['gpu_uuids'], 'gpu_processes': [],
                       'runtime': {'hostname': spec['hostname'], 'machine': 'aarch64', 'libc': ['glibc', 'synthetic'],
                                   'python_environment': {**self.pyenv, 'sys_prefix': '/usr'}, 'isolated': True, 'no_site': True,
                                   'dont_write_bytecode': True, 'optimize': 0},
                       'runtime_sha256': {self.pyenv['lexical_path']: 'a' * 64, self.pyenv['resolved_binary_path']: 'a' * 64,
                                          self.pyenv['pyvenv_config_path']: 'b' * 64, '/usr/bin/timeout': M.TIMEOUT_SHA},
                       'required_launch_environment': M.ENV, 'required_launch_variables_absent': ['CUDA_VISIBLE_DEVICES'],
                       'release_sha256': None, 'scientific_execution_performed': False, 'scientific_admission': False,
                       'clock': {'host_utc': (T + timedelta(seconds=20)).isoformat(), 'host_boot_id': spec['expected_boot_id'], 'host_monotonic_seconds': 1020.0},
                       'root_reference_utc': (T + timedelta(seconds=21)).isoformat()}
            capture_path = self.state / (role + '.verify_stage.json'); self.put(capture_path, capture); cap_pin = self.sha(capture_path)
            self.put(self.state / (role + '.verify_stage.external.json'), {'exit_code': 0, 'local_transport_reaped': True, 'local_transport_timeout': False,
                     'signals_to_own_local_group': [], 'failure': None, 'started_utc': (T + timedelta(seconds=5)).isoformat(), 'observed_utc': (T + timedelta(seconds=22)).isoformat()})
            self.put(self.state / (role + '.verify_stage.command.json'), {'stop_utc': self.stop.isoformat()})
            release = copy.deepcopy(self.resolved[role]); release.update(status='admitted_for_execution_allocation', issued_by='root',
                     preparation_and_cohort_reserves_already_accounted_before_this_queue=True, clock_error_bound_seconds=5,
                     latest_start_utc=(self.stop - timedelta(seconds=5)).isoformat(), process_clock_checked_utc=capture['clock']['host_utc'])
            release['candidate_metadata'].update(actual_release_created=True, fresh_process_closure_attestation={'control_phase_sha256': self.phase_pin, 'capture_sha256': cap_pin},
                     fresh_gpu_inventory_attestation={'capture_sha256': cap_pin}, fresh_clock_review_attestation={'capture_sha256': cap_pin, 'control_stop_utc': self.stop.isoformat()},
                     fresh_output_absence_attestation={'capture_sha256': cap_pin}, actual_complete_immutable_array_map=self.arrays, prepared_inputs_sha256=self.prepared_pin)
            self.put(self.state / (role + '.evaluation_release.json'), release)

    def put(self, path, obj): self.files[str(path)] = (json.dumps(obj, sort_keys=True, allow_nan=False) + '\n').encode()
    def get(self, path): return json.loads(self.files[str(path)])
    def read(self, path): self.reads.append(str(path)); return self.get(path)
    def sha(self, path):
        if Path(path).suffix in ('.npy', '.npz', '.pt', '.pth'): raise AssertionError('Numeric/model read attempted')
        self.hashes.append(str(path)); return hashlib.sha256(self.files[str(path)]).hexdigest()
    def contexts(self):
        return [patch.object(M, 'PK', self.pk), patch.object(M, 'BASE', self.base), patch.object(M, 'BASE_SHA', self.base_pin),
                patch.object(M, 'PREPARED_SHA', self.prepared_pin), patch.object(M, 'B_COMPLETION_SHA', '9' * 64),
                patch.object(M, 'read', self.read), patch.object(M, 'sha', self.sha),
                patch.object(Path, 'is_file', lambda path: str(path) in self.files),
                patch.object(Path, 'is_symlink', lambda path: str(path) in self.symlinks),
                patch.object(Path, 'resolve', lambda path: path)]


class IssuanceTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture(); self.patches = self.f.contexts()
        for p in self.patches: p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
    def audit(self, roles=('A', 'B'), **kw): return M.audit(self.f.state, roles, self.f.package_pin, checked_at=kw.get('checked_at', T + timedelta(seconds=30)), monotonic=kw.get('monotonic', 1030.0))
    def mutate(self, name, change):
        path = self.f.state / name; obj = self.f.get(path); change(obj); self.f.put(path, obj)

    def test_valid_both_roles_and_full_scalar_evidence_closure(self):
        r = self.audit(); self.assertEqual(set(r['release_sha256']), {'A', 'B'})
        expected = set(self.f.files) - {str(SOURCE)}
        self.assertEqual(set(r['evidence_sha256']), expected)
        self.assertTrue(all(not p.endswith(('.npy', '.pt', '.npz', '.pth')) for p in self.f.reads + self.f.hashes))
        self.assertFalse(r['scientific_execution_or_completion_asserted'])

    def test_valid_single_role(self):
        self.assertEqual(set(self.audit(('B',))['release_sha256']), {'B'})

    def test_scientific_delta_and_boolean_integer_substitution_rejected(self):
        changes = [lambda x: x['streams'][0].update(seed=False), lambda x: x['streams'][0].update(seed=2),
                   lambda x: x['candidate_metadata'].update(execution_authority=0), lambda x: x['stage_quotas_seconds'].update(full_rollout_test=7199)]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                saved = self.f.files[str(self.f.state / 'A.evaluation_release.json')]
                self.mutate('A.evaluation_release.json', change)
                with self.assertRaisesRegex(ValueError, 'exact independent actual-release delta'): self.audit()
                self.f.files[str(self.f.state / 'A.evaluation_release.json')] = saved

    def test_changed_actual_input_hash_rejected(self):
        self.mutate('A.verify_stage.json', lambda x: x['input_sha256'].update({next(iter(self.f.arrays)): '1' * 64}))
        with self.assertRaisesRegex(ValueError, 'every actual input hash'): self.audit()

    def test_stale_clock_rejected(self):
        self.mutate('A.verify_stage.json', lambda x: x['clock'].update(host_utc=(T - timedelta(seconds=400)).isoformat()))
        with self.assertRaisesRegex(ValueError, 'fresh root-host clock'): self.audit()

    def test_reset_phase_rejected(self):
        self.mutate('control_phase.json', lambda x: x.update(clock_restarted=True))
        with self.assertRaisesRegex(ValueError, 'allocations and no reset'): self.audit()

    def test_nonmatching_monotonic_clock_rejected(self):
        with self.assertRaisesRegex(ValueError, 'UTC and monotonic'): self.audit(monotonic=1400.0)

    def test_review_margin_exhausted_rejected(self):
        with self.assertRaises(ValueError): self.audit(checked_at=T + timedelta(seconds=590), monotonic=1590)

    def test_native_absence_mismatch_rejected(self):
        self.mutate('A.verify_stage.json', lambda x: x['native_absent'].update({'101': False}))
        with self.assertRaisesRegex(ValueError, 'historical native absence'): self.audit()

    def test_foreign_assigned_compute_rejected(self):
        self.mutate('A.verify_stage.json', lambda x: x.update(gpu_processes=[{'pid': 999, 'gpu_uuid': x['gpu_uuids'][0]}]))
        with self.assertRaisesRegex(ValueError, 'assigned devices'): self.audit()

    def test_transport_timeout_rejected(self):
        self.mutate('A.verify_stage.external.json', lambda x: x.update(local_transport_timeout=True))
        with self.assertRaisesRegex(ValueError, 'bounded control transport'): self.audit()

    def test_wrong_architecture_rejected(self):
        self.mutate('A.verify_stage.json', lambda x: x['runtime'].update(machine='x86_64'))
        with self.assertRaisesRegex(ValueError, 'architecture and libc'): self.audit()

    def test_changed_source_rejected(self):
        source = self.f.pk / 'sources/00.py'; self.f.files[str(source)] += b'# changed\n'
        with self.assertRaisesRegex(ValueError, 'Frozen operator member'): self.audit()

    def test_main_rechecks_source_closure_before_publication(self):
        original_audit = M.audit; output = self.f.state / 'receipt.json'
        def audited(*args, **kwargs):
            result = original_audit(*args, **kwargs)
            self.f.files[str(self.f.extra_evidence)] += b' '
            return result
        with patch.object(M, 'audit', audited), patch.object(M, 'now', side_effect=[T + timedelta(seconds=30), T + timedelta(seconds=40)]), patch.object(M.time, 'monotonic', side_effect=[1030.0, 1040.0]), patch.object(sys, 'argv', ['auditor', '--state', str(self.f.state), '--role', 'both', '--package-sha256', self.f.package_pin, '--output', str(output)]):
            with self.assertRaisesRegex(ValueError, 'Evidence changed before receipt publication'): M.main()
        self.assertNotIn(str(output), self.f.files)

    def test_main_final_time_guard_rejects_delayed_completion(self):
        output = self.f.state / 'receipt.json'
        with patch.object(M, 'now', side_effect=[T + timedelta(seconds=30), T + timedelta(seconds=590)]), patch.object(M.time, 'monotonic', side_effect=[1030.0, 1590.0]), patch.object(sys, 'argv', ['auditor', '--state', str(self.f.state), '--role', 'both', '--package-sha256', self.f.package_pin, '--output', str(output)]):
            with self.assertRaisesRegex(ValueError, 'deadline reached before review publication'): M.main()
        self.assertNotIn(str(output), self.f.files)

    def test_strict_json_type_identity(self):
        self.assertFalse(M.exact_json({'seed': 0}, {'seed': False}))
        self.assertFalse(M.exact_json({'flag': True}, {'flag': 1}))
        self.assertFalse(M.exact_json({'value': 1}, {'value': 1.0}))


if __name__ == '__main__': unittest.main()
