"""Independent local synthetic tests; no live clocks, network, or scientific data."""
from pathlib import Path
import base64
import contextlib
import datetime as DT
import hashlib
import importlib.util
import io
import json
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

K = Path(__file__).resolve().parent
STOP = '2026-10-06T23:33:01.226132+00:00'
NOW = DT.datetime.fromisoformat('2026-10-06T23:00:00+00:00')

def load(name):
    spec = importlib.util.spec_from_file_location('synthetic_' + name, K / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, 'path', [str(K), *sys.path]), patch('subprocess.Popen', side_effect=AssertionError('No subprocess on import')), patch('subprocess.check_output', side_effect=AssertionError('No subprocess on import')), patch('time.monotonic', side_effect=AssertionError('No clock on import')):
        spec.loader.exec_module(module)
    return module

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()

class FixedDateTime(DT.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW

class RemoteFixture:
    def __init__(self, directory, name):
        self.root = Path(directory)
        self.module = m = load(name)
        self.a = self.root / 'analysis'
        self.a.mkdir()
        m.A = str(self.a)
        self.phase = dump(self.a / 'controls/analysis_phase.json', {'synthetic': True})
        m.PH = self.phase
        self.proc = self.root / 'proc'
        boot = self.proc / 'sys/kernel/random/boot_id'
        boot.parent.mkdir(parents=True)
        boot.write_text(m.BOOT + '\n')
        def path(*args):
            p = Path(*args)
            if p == Path('/proc') or Path('/proc') in p.parents:
                return self.proc / p.relative_to('/proc')
            return p
        m.pathlib = types.SimpleNamespace(Path=path)
        m.D = types.SimpleNamespace(datetime=FixedDateTime, timedelta=DT.timedelta, timezone=DT.timezone)
        m.time = types.SimpleNamespace(monotonic=lambda: 100.0)
        self.stdin = io.TextIOWrapper(io.BytesIO(b''), encoding='utf8')
        self.payload = {'phase_sha256': self.phase, 'stop_utc': STOP}
        self.calls = []
    def run(self, raw=None):
        m = self.module
        if raw is None:
            raw = json.dumps(self.payload).encode()
        m.sys = types.SimpleNamespace(flags=types.SimpleNamespace(optimize=0), argv=['synthetic', base64.b64encode(json.dumps(self.payload).encode()).decode()], stdin=io.TextIOWrapper(io.BytesIO(raw), encoding='utf8'))
        output = io.StringIO()
        with patch.object(socket, 'gethostname', return_value=m.HOST), contextlib.redirect_stdout(output):
            m.main()
        return json.loads(output.getvalue())
    def transfer(self, data=b'original scalar bytes'):
        self.data = data
        self.target = self.a / 'B.saved_array_audit.json'
        self.partial = self.a / 'scalar_transfer_recovery1/B.saved_array_audit.json.partial'
        self.payload.update(target=str(self.target), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), observed_fate='absent')
    def fate(self):
        m = self.module
        self.frozen_probe = (K.parent / 'sand_stopped_analysis_operator_UNADMITTED_code_audit_v1/remote_probe.py').read_bytes()
        self.payload['original_probe_sha256'] = hashlib.sha256(self.frozen_probe).hexdigest()
        self.old_shell_commands = [b'synthetic old exact capture command', b'synthetic old exact transfer command']
        self.payload['original_failed_remote_command_sha256'] = [hashlib.sha256(x).hexdigest() for x in self.old_shell_commands]
        self.owner = self.a / 'owners/sand_saved_A'
        self.release = {'command': ['synthetic_worker'], 'invocation_origin_utc': '2026-10-06T22:40:00+00:00', 'hard_deadline_monotonic_ns': 2000}
        pin = dump(self.a / 'controls/sand_saved_A.cpu_release.json', self.release)
        self.saved = self.a / 'A.saved_array_audit.json'
        self.saved.write_bytes(b'synthetic audit')
        owner = {'pid': 202, 'start_ticks': 2}
        outer = {'pid': 201, 'start_ticks': 1}
        child = {'pid': 203, 'start_ticks': 3}
        self.terminal = {'schema': 'adaptgns_stopped_analysis_cpu_terminal_v1', 'status': 'complete', 'failure': None, 'release_sha256': pin, 'child_native_absent': True, 'owner_identity': owner, 'outer_timeout_identity': outer, 'child': {'reaped': True, 'exit_code': 0, 'signals': [], 'cleanup_errors': [], 'command': self.release['command'], 'identity': child, 'pid': 203}, 'publication_utc': '2026-10-06T22:50:00+00:00', 'native_hard_guard': {'absolute_deadline_ns': 2000}, 'output_sha256': {str(self.saved): hashlib.sha256(self.saved.read_bytes()).hexdigest()}}
        dump(self.owner / 'owner_terminal.json', self.terminal)
        dump(self.owner / 'owner_started.json', {'owner_identity': owner, 'outer_timeout_identity': outer})
        dump(self.owner / 'child_registered.json', {'identity': child, 'owner_identity': owner})
        self.payload.update(saved_A_release_sha256=pin, historical_pids=[200], B_targets={str(self.a / n): {'bytes': 4, 'sha256': hashlib.sha256(b'test').hexdigest()} for n in ('B.stopped_collection.json', 'B.saved_array_audit.json')})
        self.apps = ''
        def check_output(argv, **kwargs):
            self.calls.append(argv)
            if argv == ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits']:
                return '\n'.join(str(i) + ', ' + gpu for i, gpu in enumerate(m.GPU))
            if argv == ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits']:
                return self.apps
            raise AssertionError('Unexpected subprocess')
        m.subprocess = types.SimpleNamespace(check_output=check_output)
    def native_cmdline(self, raw):
        directory = self.proc / '300'; directory.mkdir()
        (directory / 'cmdline').write_bytes(raw)
        (directory / 'stat').write_text('300 (synthetic) ' + ' '.join(['S', '1', '300', '300'] + ['0'] * 15 + ['123']))

class RecoverySourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sand_recovery_synthetic_')
        self.root = Path(self.temp.name).resolve()
    def tearDown(self):
        self.temp.cleanup()
    def fixture(self, name):
        return RemoteFixture(self.root, name)
    def test_all_imports_inert(self):
        for name in ('recover', 'remote_fate', 'remote_transfer', 'continue_original'):
            load(name)
    def test_exact_transfer_and_retained_partial(self):
        f = self.fixture('remote_transfer'); f.transfer()
        result = f.run(f.data)
        self.assertEqual(result['status'], 'published_exact_stream_exclusively')
        self.assertEqual(f.target.read_bytes(), f.data)
        self.assertEqual(f.partial.read_bytes(), f.data)
        self.assertEqual(f.target.stat().st_ino, f.partial.stat().st_ino)
    def test_truncated_stream_retains_partial(self):
        f = self.fixture('remote_transfer'); f.transfer()
        with self.assertRaisesRegex(ValueError, 'Truncated'): f.run(f.data[:-1])
        self.assertFalse(f.target.exists()); self.assertEqual(f.partial.read_bytes(), f.data[:-1])
    def test_extra_stream_retains_partial(self):
        f = self.fixture('remote_transfer'); f.transfer()
        with self.assertRaisesRegex(ValueError, 'Unexpected bytes'): f.run(f.data + b'x')
        self.assertFalse(f.target.exists()); self.assertEqual(f.partial.read_bytes(), f.data)
    def test_wrong_stream_hash_retains_partial(self):
        f = self.fixture('remote_transfer'); f.transfer()
        wrong = b'x' * len(f.data)
        with self.assertRaisesRegex(ValueError, 'hash differs'): f.run(wrong)
        self.assertFalse(f.target.exists()); self.assertEqual(f.partial.read_bytes(), wrong)
    def test_conflicting_final_preserved(self):
        f = self.fixture('remote_transfer'); f.transfer(); f.target.write_bytes(b'conflict')
        with self.assertRaisesRegex(ValueError, 'conflict'): f.run(f.data)
        self.assertEqual(f.target.read_bytes(), b'conflict'); self.assertFalse(f.partial.exists())
    def test_identical_final_reused_only_from_exact_fate(self):
        f = self.fixture('remote_transfer'); f.transfer(); f.target.write_bytes(f.data)
        with self.assertRaisesRegex(ValueError, 'changed after'): f.run(f.data)
        f.payload['observed_fate'] = 'exact'
        self.assertEqual(f.run(b'')['status'], 'reused_exact_existing_bytes')
        self.assertFalse(f.partial.exists())
    def test_existing_partial_never_overwritten(self):
        f = self.fixture('remote_transfer'); f.transfer(); f.partial.parent.mkdir(); f.partial.write_bytes(b'earlier failed attempt')
        with self.assertRaises(FileExistsError): f.run(f.data)
        self.assertEqual(f.partial.read_bytes(), b'earlier failed attempt'); self.assertFalse(f.target.exists())
    def test_exact_fate_disappearance_rejected(self):
        f = self.fixture('remote_transfer'); f.transfer(); f.payload['observed_fate'] = 'exact'
        with self.assertRaisesRegex(ValueError, 'disappeared'): f.run(f.data)
        self.assertFalse(f.partial.exists())
    def test_only_two_fixed_target_names(self):
        f = self.fixture('remote_transfer'); f.transfer(); f.payload['target'] = str(f.a / 'new-science.json')
        with self.assertRaisesRegex(ValueError, 'Exact B scalar target'): f.run(f.data)
    def test_wrong_original_phase_rejected(self):
        f = self.fixture('remote_transfer'); f.transfer(); f.payload['phase_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'original phase'): f.run(f.data)
    def test_wrong_original_stop_rejected(self):
        f = self.fixture('remote_transfer'); f.transfer(); f.payload['stop_utc'] = '2026-10-07T00:33:01.226132+00:00'
        with self.assertRaisesRegex(ValueError, 'Original stop'): f.run(f.data)
    def test_expired_phase_rejected_with_synthetic_clock(self):
        f = self.fixture('remote_transfer'); f.transfer()
        with patch(__name__ + '.NOW', DT.datetime.fromisoformat(STOP) - DT.timedelta(seconds=8)):
            with self.assertRaisesRegex(ValueError, 'phase exhausted'): f.run(f.data)
        self.assertFalse(f.partial.exists())
    def test_fate_observes_original_native_and_scalar(self):
        f = self.fixture('remote_fate'); f.fate(); result = f.run()
        self.assertEqual(set(result['native_absent']), {'200', '201', '202', '203'})
        self.assertTrue(all(result['native_absent'].values()))
        self.assertEqual({x['state'] for x in result['B_targets'].values()}, {'absent'})
        self.assertEqual(len(f.calls), 2)
    def test_fate_reports_exact_and_conflicting_bytes_without_overwrite(self):
        f = self.fixture('remote_fate'); f.fate()
        first, second = map(Path, f.payload['B_targets'])
        first.write_bytes(b'test'); second.write_bytes(b'conflict')
        result = f.run()
        self.assertEqual(result['B_targets'][str(first)]['state'], 'exact')
        self.assertEqual(result['B_targets'][str(second)]['state'], 'conflicting_retained_bytes')
        self.assertEqual(second.read_bytes(), b'conflict')
    def test_fate_original_process_present_rejected(self):
        f = self.fixture('remote_fate'); f.fate(); (f.proc / '200').mkdir()
        with self.assertRaisesRegex(ValueError, 'native process remains'): f.run()
    def test_fate_failed_original_worker_rejected(self):
        f = self.fixture('remote_fate'); f.fate(); f.terminal['child']['exit_code'] = 1; dump(f.owner / 'owner_terminal.json', f.terminal)
        with self.assertRaisesRegex(ValueError, 'worker not cleanly complete'): f.run()
    def test_fate_saved_product_hash_rejected(self):
        f = self.fixture('remote_fate'); f.fate(); f.saved.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'scalar hash differs'): f.run()
    def test_fate_nonempty_gpu_rejected(self):
        f = self.fixture('remote_fate'); f.fate(); f.apps = '300, GPU-synthetic'
        with self.assertRaisesRegex(ValueError, 'GPUs nonempty'): f.run()
    def test_fate_live_old_frozen_probe_rejected(self):
        f = self.fixture('remote_fate'); f.fate()
        f.native_cmdline(b'python\0-I\0-S\0-B\0-c\0' + f.frozen_probe + b'\0')
        with self.assertRaisesRegex(ValueError, '[Pp]robe'): f.run()
    def test_fate_live_old_probe_timeout_ancestor_rejected(self):
        f = self.fixture('remote_fate'); f.fate()
        f.native_cmdline(b'/usr/bin/timeout\0--signal=TERM\0--kill-after=3s\090s\0python\0-I\0-S\0-B\0-c\0' + f.frozen_probe + b'\0')
        with self.assertRaisesRegex(ValueError, '[Pp]robe'): f.run()
    def test_fate_live_old_remote_shell_command_rejected(self):
        f = self.fixture('remote_fate'); f.fate()
        f.native_cmdline(b'/bin/sh\0-c\0' + f.old_shell_commands[1] + b'\0')
        with self.assertRaisesRegex(ValueError, '[Pp]robe'): f.run()
    def test_fate_unrelated_inline_python_not_misidentified(self):
        f = self.fixture('remote_fate'); f.fate()
        f.native_cmdline(b'python\0-c\0print("unrelated synthetic source")\0')
        self.assertEqual(f.run()['status'], 'metadata_observed')
    def test_fate_old_probe_appearing_after_hashes_rejected(self):
        f = self.fixture('remote_fate'); f.fate()
        previous = f.module.subprocess.check_output
        def delayed(argv, **kwargs):
            value = previous(argv, **kwargs)
            if '--query-compute-apps=pid,gpu_uuid' in argv:
                f.native_cmdline(b'python\0-c\0' + f.frozen_probe + b'\0')
            return value
        f.module.subprocess = types.SimpleNamespace(check_output=delayed)
        with self.assertRaisesRegex(ValueError, '[Pp]robe'): f.run()
    def route(self):
        m = load('recover'); m.S = self.root
        phase = {'original_analysis_started_utc': '2026-10-06T22:33:01.226132+00:00', 'original_analysis_stop_utc': STOP}
        dump(self.root / 'analysis_phase.json', phase); dump(self.root / 'local_phase_anchor.json', {'monotonic_seconds': 1000})
        ready = {'schema': 'coder_sand_original_phase_recovery_route_proxy_v1', 'analysis_phase_sha256': m.PH, 'local_phase_anchor_sha256': m.AN, **phase, 'original_monotonic_stop': 4600, 'new_or_restarted_clock_granted': False, 'authority': 'coder.internal.cohere.com:443', 'upstream': ['100.106.33.61', 443], 'opaque_tls': True, 'no_settings_changed': True, 'url': 'http://127.0.0.1:12345'}
        ready.update(analysis_phase_path=str(m.S / 'analysis_phase.json'), connection_cap=2048, concurrency_cap=8, per_direction_buffer_bytes=262144, payload_backpressure='nonblocking_partial_send', original_route_package_sha256='292cc31d54e822d6eb1a843da3ce55427f9f89201c1fb9a2dd7293d0660dfd80')
        return m, ready
    def test_controller_exact_original_route(self):
        m, ready = self.route(); path = self.root / 'ready.json'; pin = dump(path, ready)
        with patch.dict('os.environ', {'NO_PROXY': 'everything', 'no_proxy': 'everything'}):
            env = m.route_environment(path, pin)
        self.assertEqual(env['HTTPS_PROXY'], ready['url']); self.assertNotIn('NO_PROXY', env); self.assertNotIn('no_proxy', env)
    def test_controller_route_binding_mutations_rejected(self):
        m, ready = self.route(); path = self.root / 'ready.json'
        for key, bad in [('original_monotonic_stop', 8200), ('original_analysis_stop_utc', '2026-10-07T01:00:00+00:00'), ('new_or_restarted_clock_granted', True), ('upstream', ['127.0.0.1', 443]), ('url', 'http://name:password@127.0.0.1:12345'), ('url', 'http://127.0.0.1:12345/path'), ('analysis_phase_path', '/different/phase'), ('connection_cap', 4096), ('concurrency_cap', 9), ('per_direction_buffer_bytes', 524288), ('payload_backpressure', 'blocking_sendall'), ('original_route_package_sha256', '0' * 64)]:
            with self.subTest(key=key, bad=bad):
                pin = dump(path, {**ready, key: bad})
                with self.assertRaises(ValueError): m.route_environment(path, pin)
    def test_controller_strict_metadata_json(self):
        m = load('recover'); path = self.root / 'metadata.json'
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            path.write_text(raw)
            with self.assertRaises(ValueError): m.read(path)
    def stream_fixture(self, timeouts):
        m = load('recover'); m.S = self.root
        m.time = types.SimpleNamespace(monotonic=lambda: 100.0)
        stop = NOW + DT.timedelta(seconds=60)
        operator = types.SimpleNamespace(now=lambda: NOW, budget=lambda state, tail: ({}, stop, 160.0, 30.0))
        path = self.root / 'synthetic_scalar.json'; path.write_bytes(b'exact synthetic scalar')
        calls, signals = [], []
        class Process:
            pid = 42001
            returncode = None
            def __init__(self):
                self.count = 0
            def poll(self):
                return self.returncode
            def communicate(self, **kwargs):
                calls.append(kwargs)
                self.count += 1
                if self.count <= timeouts:
                    raise subprocess.TimeoutExpired('synthetic', kwargs['timeout'])
                self.returncode = 0 if timeouts < 2 else -9
                return b'{"synthetic":true}', b'synthetic stderr'
        process = Process()
        def popen(argv, **kwargs):
            self.assertEqual(argv, ['synthetic-ssh'])
            self.assertTrue(kwargs['start_new_session'])
            self.assertNotIn('text', kwargs)
            self.assertEqual(kwargs['stdin'].read(), path.read_bytes())
            return process
        m.subprocess = types.SimpleNamespace(Popen=popen, PIPE=-1, TimeoutExpired=subprocess.TimeoutExpired)
        m.os = types.SimpleNamespace(killpg=lambda pid, sig: signals.append((pid, sig)))
        return m, operator, path, calls, signals
    def test_controller_stream_binary_stdin_and_reaping(self):
        m, operator, path, calls, signals = self.stream_fixture(0)
        self.assertEqual(m.stream(operator, 'synthetic-transfer', ['synthetic-ssh'], path), {'synthetic': True})
        receipt = json.loads((self.root / 'synthetic-transfer.external.json').read_text())
        self.assertEqual(calls, [{'timeout': 18.0}]); self.assertEqual(signals, [])
        self.assertTrue(receipt['local_transport_reaped']); self.assertFalse(receipt['local_transport_timeout'])
    def test_controller_stream_timeout_records_scoped_term(self):
        m, operator, path, calls, signals = self.stream_fixture(1)
        with self.assertRaisesRegex(ValueError, 'Stream transport failed'):
            m.stream(operator, 'synthetic-transfer', ['synthetic-ssh'], path)
        receipt = json.loads((self.root / 'synthetic-transfer.external.json').read_text())
        self.assertEqual([p for p, _ in signals], [42001])
        self.assertEqual(receipt['signals_to_own_local_group'], ['SIGTERM'])
        self.assertTrue(receipt['local_transport_timeout']); self.assertTrue(receipt['local_transport_reaped'])
    def test_controller_stream_timeout_records_scoped_kill(self):
        m, operator, path, calls, signals = self.stream_fixture(2)
        with self.assertRaisesRegex(ValueError, 'Stream transport failed'):
            m.stream(operator, 'synthetic-transfer', ['synthetic-ssh'], path)
        receipt = json.loads((self.root / 'synthetic-transfer.external.json').read_text())
        self.assertEqual([p for p, _ in signals], [42001, 42001])
        self.assertEqual(receipt['signals_to_own_local_group'], ['SIGTERM', 'SIGKILL'])
        self.assertTrue(receipt['local_transport_timeout']); self.assertTrue(receipt['local_transport_reaped'])
        self.assertEqual(receipt['exit_code'], -9)
    def continuation_args(self, **changes):
        return types.SimpleNamespace(**{'action': 'run', 'operation': 'sand_summarize', 'session_id': None, 'exit_code': None, 'observed_utc': None, 'evidence_file': None, **changes})
    def test_continuation_exact_frozen_argv_and_operation_allowlist(self):
        m = load('continue_original')
        for action in ('run', 'close-product'):
            for operation in m.OPS:
                self.assertEqual(m.argv_for(self.continuation_args(action=action, operation=operation)), [sys.executable, '-B', str(m.F / 'root_operator.py'), action, '--root-action', '--state', str(m.S), '--package-sha256', m.FROZEN, '--operation', operation])
        self.assertEqual(m.argv_for(self.continuation_args(action='complete', operation=None))[-2:], ['--package-sha256', m.FROZEN])
    def test_continuation_preserves_genuine_nonzero_exit(self):
        m = load('continue_original')
        result = m.argv_for(self.continuation_args(action='record-original-exit', session_id=42, exit_code=130, observed_utc='synthetic root UTC', evidence_file=Path('/synthetic/tool.json')))
        self.assertEqual(result[-8:], ['--session-id', '42', '--exit-code', '130', '--observed-utc', 'synthetic root UTC', '--evidence-file', '/synthetic/tool.json'])
    def test_continuation_rejects_other_actions_operations_and_overrides(self):
        m = load('continue_original')
        for changes in ({'action': 'begin'}, {'operation': 'sand_saved_A'}, {'operation': 'sand_collect_B'}, {'action': 'complete'}, {'action': 'complete', 'operation': None, 'exit_code': 0}, {'session_id': 10}, {'action': 'record-original-exit'}, {'action': 'record-original-exit', 'session_id': True, 'exit_code': 0, 'observed_utc': 'synthetic', 'evidence_file': Path('/synthetic/tool.json')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): m.argv_for(self.continuation_args(**changes))
    def test_continuation_run_refuses_any_prior_attempt_artifact(self):
        m = load('continue_original'); m.S = self.root
        (self.root / 'sand_summarize.failed-attempt').write_text('preserved')
        m.fixed_preconditions = lambda *args: types.SimpleNamespace(budget=lambda *args: None)
        m.route_environment = lambda *args: {'SYNTHETIC': 'ONLY'}
        m.os = types.SimpleNamespace(execve=lambda *args: self.fail('No exec after prior attempt'))
        with patch.object(sys, 'argv', ['continue_original', 'run', '--root-execute', '--operation', 'sand_summarize']):
            with self.assertRaisesRegex(ValueError, 'already exists'): m.main()
    def test_continuation_parser_forbids_forwarded_state_or_command(self):
        m = load('continue_original')
        m.fixed_preconditions = lambda *args: self.fail('No admission after parser rejection')
        for flag in ('--state', '--command', '--worker', '--dataset'):
            with self.subTest(flag=flag), patch.object(sys, 'argv', ['continue_original', 'run', '--root-execute', '--operation', 'sand_summarize', flag, 'arbitrary']), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught: m.main()
                self.assertEqual(caught.exception.code, 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
