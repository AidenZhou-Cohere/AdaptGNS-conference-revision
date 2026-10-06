"""Decoded synthetic fixtures only; RealRuntime is never instantiated."""
import base64
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import PurePosixPath
import unittest
from unittest.mock import patch
import remote_control as M

BOOT = '11111111-2222-3333-4444-555555555555'
T = datetime(2026, 10, 6, 20, 0, tzinfo=timezone.utc)
SYNTHETIC_SOURCE = b'# synthetic pinned source; never executed\n'
SYNTHETIC_SHA = hashlib.sha256(SYNTHETIC_SOURCE).hexdigest()


def process(pid, argv, parent=1, group=None):
    return {'pid': pid, 'ppid': parent, 'pgid': pid if group is None else group,
            'sid': pid if group is None else group, 'start_ticks': pid * 10,
            'executable': M.PYTHON_REAL, 'argv': copy.deepcopy(argv)}


def payload(role='B', action='verify_stage'):
    pins = {p: h or 'e' * 64 for p, h in M.BASE_PINS[role].items()}
    pins.update({p: 'd' * 64 for p in M.ARRAY_PATHS})
    result = {'action': action, 'role': role, 'stop_utc': (T + timedelta(seconds=600)).isoformat(),
              'files_sha256': pins, 'historical_pids': [101, 102],
              'forbidden_program_tokens': sorted(M.MANDATORY_FORBIDDEN), 'expected_boot_id': BOOT}
    if action == 'verify_stage':
        result['sources'] = {p: base64.b64encode(SYNTHETIC_SOURCE).decode() for p in M.SOURCES}
    else:
        release = {'schema': 'adaptgns_sand_evaluation_gpu_scoped_release_v1',
                   'status': 'admitted_for_execution_allocation', 'issued_by': 'root',
                   'dataset': 'Sand', 'host_role': role, 'host': M.HOSTS[role],
                   'files_sha256': pins, 'environment': M.ENV, 'python_environment': M.PYENV,
                   'gpu_uuids': M.GPU_UUIDS[role], 'streams': [{'commands': M.CHILD_ARGV[role]}]}
        raw = M.encoded(release); digest = hashlib.sha256(raw).hexdigest()
        if action == 'stage_release':
            result['release'] = {'path': M.release_path(role), 'sha256': digest,
                                 'base64': base64.b64encode(raw).decode()}
        else:
            result['release_sha256'] = digest
            result['expected_owner_argv'] = M.OWNER_ARGV[role]
    return result


class FakeRuntime:
    def __init__(self, p):
        self.role = p['role']; self.boot = BOOT; self.seconds = 0
        self.hashes = dict(p['files_sha256']); self.hashes[M.TIMEOUT] = M.TIMEOUT_SHA
        self.files = {}; self.directories = {M.R, str(PurePosixPath(M.R).parent), M.CTRL}
        self.directories.update(str(PurePosixPath(path).parent) for path in self.hashes)
        self.symlinks = set(); self.noncanonical = set(); self.active = []; self.pids = set()
        self.gpus = list(M.GPU_UUIDS[self.role]); self.apps = []; self.created = []; self.writes = []
        self.native_calls = 0; self.hash_calls = 0; self.outer_checked = False
        self.outer_valid = True; self.env_valid = True; self.mutate_on_hash = None
        self.info = {'hostname': M.HOSTS[self.role], 'machine': 'aarch64', 'libc': ['glibc', 'synthetic'],
                     'python_environment': {**M.PYENV, 'sys_prefix': '/usr'},
                     'isolated': True, 'no_site': True, 'dont_write_bytecode': True, 'optimize': 0}
        self.identity_overrides = {}
        if p['action'] == 'observe_owner':
            staged = payload(self.role, 'stage_release')['release']
            raw = base64.b64decode(staged['base64']); self.files[M.release_path(self.role)] = raw
            self.hashes[M.release_path(self.role)] = hashlib.sha256(raw).hexdigest()
            self.active = [process(201, M.OWNER_ARGV[self.role], group=199),
                           process(202, M.CHILD_ARGV[self.role][0], parent=201)]
            self.pids.update([201, 202]); self.directories.add(M.output_path(self.role))
            command = self.active[1]['argv']; index = int(command[command.index('--cuda-index') + 1])
            self.apps = [{'pid': 202, 'gpu_uuid': self.gpus[index]}]

    def clock(self):
        return {'host_utc': (T + timedelta(seconds=self.seconds)).isoformat(),
                'host_monotonic_seconds': 1000 + self.seconds, 'host_boot_id': self.boot}

    def canonical(self, path): return path not in self.noncanonical
    def kind(self, path):
        if path in self.symlinks: return 'symlink'
        if path in self.directories: return 'directory'
        if path in self.hashes: return 'file'
        return 'absent'
    def python_link_valid(self): return M.PYTHON not in self.noncanonical
    def hash_file(self, path, check):
        check(); self.hash_calls += 1
        if self.mutate_on_hash: self.mutate_on_hash(self, path)
        check(); return self.hashes[path]
    def read_bytes(self, path, limit):
        value = self.files[path]; M.need(len(value) <= limit, 'fake size'); return value
    def mkdir(self, path):
        M.need(self.kind(path) == 'absent', 'fake mkdir collision'); self.directories.add(path); self.created.append(path)
    def write_new(self, path, raw):
        M.need(self.kind(path) == 'absent', 'fake exclusive write'); self.files[path] = raw
        self.hashes[path] = hashlib.sha256(raw).hexdigest(); self.writes.append(path)
    def runtime_identity(self): return copy.deepcopy(self.info)
    def pid_exists(self, pid): return pid in self.pids
    def processes(self): self.native_calls += 1; return copy.deepcopy(self.active)
    def identity(self, pid):
        if pid in self.identity_overrides: return self.identity_overrides[pid]
        return copy.deepcopy(next(row for row in self.active if row['pid'] == pid))
    def require_environment(self, pid, required, absent): M.need(self.env_valid, 'fake environment differs')
    def gpu_snapshot(self, guard): guard.check(); return copy.deepcopy(self.gpus), copy.deepcopy(self.apps)
    def require_outer_timeout(self, guard):
        M.need(self.outer_valid, 'fake invalid native outer'); guard.check(); self.outer_checked = True


class RemoteControlTests(unittest.TestCase):
    def setUp(self):
        self.sources = {p: SYNTHETIC_SHA for p in M.SOURCES}
        base = copy.deepcopy(M.BASE_PINS)
        for role in base:
            for path in self.sources: base[role][path] = SYNTHETIC_SHA
        self.patches = [patch.object(M, 'SOURCES', self.sources), patch.object(M, 'BASE_PINS', base)]
        for item in self.patches: item.start()
        self.addCleanup(lambda: [item.stop() for item in reversed(self.patches)])

    def test_all_three_actions_accept_synthetic_a_and_b(self):
        for role in ('A', 'B'):
            for action in ('verify_stage', 'stage_release', 'observe_owner'):
                with self.subTest(role=role, action=action):
                    p = payload(role, action); rt = FakeRuntime(p); original = copy.deepcopy(p)
                    result = M.run(p, rt)
                    self.assertEqual(result['input_sha256'], p['files_sha256'])
                    self.assertEqual(result['hostname'], M.HOSTS[role]); self.assertTrue(rt.outer_checked)
                    self.assertEqual(result['clock']['host_boot_id'], BOOT)
                    self.assertFalse(result['scientific_execution_performed']); self.assertEqual(p, original)
                    self.assertEqual(len(result['input_sha256']), 164 if role == 'A' else 162)

    def test_missing_sources_staged_exclusively_and_only_control_parent_created(self):
        p = payload(); rt = FakeRuntime(p); rt.directories.remove(M.CTRL)
        for path in M.SOURCES: del rt.hashes[path]
        result = M.run(p, rt)
        self.assertEqual(set(rt.writes), set(M.SOURCES)); self.assertEqual(rt.created, [M.CTRL])
        self.assertEqual(len(result['files_staged']), 24)

    def test_existing_different_source_not_overwritten(self):
        p = payload(); rt = FakeRuntime(p); bad = next(iter(M.SOURCES)); rt.hashes[bad] = 'a' * 64
        with self.assertRaises(ValueError): M.run(p, rt)
        self.assertEqual(rt.writes, [])

    def test_source_blob_mismatch_and_extra_source_rejected(self):
        for mutation in ('mismatch', 'extra'):
            p = payload()
            if mutation == 'mismatch': p['sources'][next(iter(p['sources']))] = base64.b64encode(b'wrong').decode()
            else: p['sources'][M.CTRL + '/extra.py'] = ''
            with self.assertRaises(ValueError): M.run(p, FakeRuntime(p))

    def test_exact_input_keys_and_nonnull_pins(self):
        for change in ('missing', 'extra', 'null', 'fixed'):
            p = payload(); path = next(iter(p['files_sha256']))
            if change == 'missing': del p['files_sha256'][path]
            elif change == 'extra': p['files_sha256'][M.R + '/other.npy'] = 'a' * 64
            elif change == 'null': p['files_sha256'][path] = None
            else: p['files_sha256'][path] = 'a' * 64
            with self.assertRaises(ValueError): M.validate_payload(p)

    def test_historical_pid_and_forbidden_process_rejected(self):
        for choice in ('historical', 'forbidden'):
            p = payload(); rt = FakeRuntime(p)
            if choice == 'historical': rt.pids.add(101)
            else: rt.active = [process(999, [M.PYTHON, M.PREP + '/train_sand_graph_support_cuda.py'])]
            with self.assertRaises(ValueError): M.run(p, rt)

    def test_wrong_host_runtime_boot_or_outer_rejected_before_writes(self):
        for choice in ('host', 'architecture', 'glibc', 'boot', 'outer', 'venv'):
            p = payload(); rt = FakeRuntime(p); rt.directories.remove(M.CTRL)
            if choice == 'host': rt.info['hostname'] = 'wrong'
            elif choice == 'architecture': rt.info['machine'] = 'x86_64'
            elif choice == 'glibc': rt.info['libc'] = ['musl', '1']
            elif choice == 'boot': rt.boot = 'aaaaaaaa-2222-3333-4444-555555555555'
            elif choice == 'outer': rt.outer_valid = False
            else: rt.info['python_environment']['pyvenv_config_sha256'] = 'a' * 64
            with self.assertRaises(ValueError): M.run(p, rt)
            self.assertEqual(rt.created, [])

    def test_role_output_or_actual_release_preexistence_rejected(self):
        for path in (M.output_path('B'), M.release_path('B')):
            p = payload(); rt = FakeRuntime(p); rt.directories.add(path)
            with self.assertRaises(ValueError): M.run(p, rt)

    def test_symlink_or_noncanonical_input_rejected(self):
        for choice in ('symlink', 'canonical'):
            p = payload(); rt = FakeRuntime(p); path = next(iter(M.ARRAY_PATHS))
            (rt.symlinks if choice == 'symlink' else rt.noncanonical).add(path)
            with self.assertRaises(ValueError): M.run(p, rt)

    def test_gpu_mapping_and_foreign_assigned_compute_rejected(self):
        for choice in ('mapping', 'foreign'):
            p = payload(); rt = FakeRuntime(p)
            if choice == 'mapping': rt.gpus[0], rt.gpus[1] = rt.gpus[1], rt.gpus[0]
            else: rt.apps = [{'pid': 900, 'gpu_uuid': rt.gpus[2]}]
            with self.assertRaises(ValueError): M.run(p, rt)

    def test_foreign_unassigned_b_compute_retained_without_control(self):
        p = payload(); rt = FakeRuntime(p); rt.apps = [{'pid': 900, 'gpu_uuid': rt.gpus[0]}]
        self.assertEqual(M.run(p, rt)['gpu_processes'], rt.apps)

    def test_staged_release_path_hash_and_scientific_argv_rejected(self):
        for choice in ('path', 'hash', 'argv'):
            p = payload(action='stage_release')
            if choice == 'path': p['release']['path'] = M.release_path('A')
            elif choice == 'hash': p['release']['sha256'] = 'a' * 64
            else:
                d = M.decode_json(base64.b64decode(p['release']['base64'])); d['streams'][0]['commands'][0][0] = '/other/python'
                raw = M.encoded(d); p['release']['base64'] = base64.b64encode(raw).decode(); p['release']['sha256'] = hashlib.sha256(raw).hexdigest()
            with self.assertRaises(ValueError): M.run(p, FakeRuntime(p))

    def test_observed_release_digest_and_owner_cardinality(self):
        for choice in ('digest', 'missing', 'duplicate'):
            p = payload(action='observe_owner'); rt = FakeRuntime(p)
            if choice == 'digest': p['release_sha256'] = 'a' * 64
            elif choice == 'missing': rt.active = []
            else: rt.active.append(process(203, M.OWNER_ARGV['B']))
            with self.assertRaises(ValueError): M.run(p, rt)

    def test_observed_owner_environment_and_child_parent_or_gpu(self):
        for choice in ('environment', 'parent', 'gpu', 'identity'):
            p = payload(action='observe_owner'); rt = FakeRuntime(p)
            if choice == 'environment': rt.env_valid = False
            elif choice == 'parent': rt.active[1]['ppid'] = 999
            elif choice == 'gpu': rt.apps[0]['gpu_uuid'] = rt.gpus[0]
            else: rt.identity_overrides[201] = {**rt.active[0], 'start_ticks': 999}
            with self.assertRaises(ValueError): M.run(p, rt)

    def test_remaining_deadline_and_complete_future_reserve(self):
        p = payload(); rt = FakeRuntime(p); rt.seconds = 598
        with self.assertRaises(ValueError): M.run(p, rt)
        p['stop_utc'] = '2026-10-06T23:43:51+00:00'
        with self.assertRaises(ValueError): M.validate_payload(p)

    def test_final_rehash_detects_input_mutation(self):
        p = payload(); rt = FakeRuntime(p); path = next(iter(M.ARRAY_PATHS)); seen = []
        def mutate(runtime, name):
            if name == path:
                seen.append(name)
                if len(seen) == 2: runtime.hashes[name] = 'a' * 64
        rt.mutate_on_hash = mutate
        with self.assertRaises(ValueError): M.run(p, rt)
        self.assertEqual(len(seen), 2)

    def test_decode_rejects_duplicate_and_nonfinite_json(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError): M.decode_json(value)
        self.assertEqual(json.loads(M.encoded({'a': 1})), {'a': 1})

    def test_native_timeout_grammar_uses_only_synthetic_process_identities(self):
        for mutation in ('valid', 'overlong', 'kill_tail', 'wrong_parent', 'wrong_child'):
            with self.subTest(mutation=mutation):
                p = payload(); rt = FakeRuntime(p); guard = M.Guard(rt, p)
                command = [M.PYTHON, '-I', '-S', '-B', '-c', '# synthetic inert helper']
                own = process(201, command, parent=200)
                parent = process(200, [M.TIMEOUT, '--signal=TERM', '--kill-after=5s', '120s'] + command)
                parent['executable'] = M.TIMEOUT
                if mutation == 'overlong': parent['argv'][3] = '121s'
                elif mutation == 'kill_tail': parent['argv'][2] = '--kill-after=15s'
                elif mutation == 'wrong_parent': parent['executable'] = '/other/timeout'
                elif mutation == 'wrong_child': parent['argv'][-1] = '# other helper'
                rt.identity_overrides = {201: own, 200: parent}
                with patch.object(M.sys, 'platform', 'linux'), patch.object(M.os, 'getpid', return_value=201), patch.object(M.os, 'getppid', return_value=200):
                    if mutation == 'valid': M.RealRuntime.require_outer_timeout(rt, guard)
                    else:
                        with self.assertRaises(ValueError): M.RealRuntime.require_outer_timeout(rt, guard)


if __name__ == '__main__': unittest.main()
