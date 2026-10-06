"""Independent pure synthetic startup audit tests; no native or live clock access."""
import copy
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'review_sand_final24_startup_transition_v1.py'
SOURCE_SHA = '7e3343c8eb07dc7221de60d9b67c3f2978814845ab6d632dd73c0e6c721f1877'
SOURCE_BYTES = SOURCE.read_bytes()
assert hashlib.sha256(SOURCE_BYTES).hexdigest() == SOURCE_SHA
spec = importlib.util.spec_from_file_location('startup_subject', SOURCE)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
fixture_path = HERE.parent / 'sand_final24_issuance_audit_review_owner_v1/test_issuance_audit.py'
spec = importlib.util.spec_from_file_location('issuance_fixture_only', fixture_path)
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
T = F.T


class Fixture(F.Fixture):
    def __init__(self):
        super().__init__()
        self.prep = Path('/synthetic/preparation')
        self.files[str(SOURCE)] = SOURCE_BYTES
        self.files[str(self.prep / 'review_actual_sand_final24_issuance_code_audit_v1.py')] = b'# inert issuance source\n'
        self.issuer_pin = self.sha(self.prep / 'review_actual_sand_final24_issuance_code_audit_v1.py')
        self.files[str(self.pk / 'remote_control.py')] = b'# inert remote helper\n'
        manifest = self.get(self.pk / 'manifest.json')
        manifest['files_sha256']['remote_control.py'] = self.sha(self.pk / 'remote_control.py')
        self.put(self.pk / 'manifest.json', manifest); self.package_pin = self.sha(self.pk / 'manifest.json')
        self.put(self.state / 'operator_binding.json', {'manifest_sha256': self.package_pin})
        for role, index in [('A', 0), ('B', 2)]:
            release = self.get(self.state / (role + '.evaluation_release.json'))
            release['streams'][0]['commands'] = [[M.PYTHON, M.R + '/evaluate.py', '--execute', '--cuda-index', str(index)]]
            self.put(self.state / (role + '.evaluation_release.json'), release)
        self.issue = self.state / 'accepted_issue_review.json'
        self.put(self.issue, {'status': 'passed_actual_final24_issuance', 'control_phase_sha256': self.phase_pin,
                 'prepared_inputs_sha256': self.prepared_pin, 'operator_manifest_sha256': self.package_pin,
                 'review_source_sha256': self.issuer_pin, 'checked_utc': self.at(30), 'completed_utc': self.at(40),
                 'release_sha256': {r: self.sha(self.state / (r + '.evaluation_release.json')) for r in ('A', 'B')},
                 'evidence_sha256': {str(self.extra_evidence): self.sha(self.extra_evidence)}})
        self.issue_pin = self.sha(self.issue)
        completion = {'status': 'both_original_owners_observed_inside_same_control_window', 'control_phase_sha256': self.phase_pin,
                      'scientific_completion': False, 'checked_utc': self.at(100), 'owner_observation_sha256': {}}
        for role, pid, session_id, index in [('A', 201, 10001, 0), ('B', 301, 10002, 2)]:
            spec = self.roles[role]; release = self.get(self.state / (role + '.evaluation_release.json'))
            release_pin = self.sha(self.state / (role + '.evaluation_release.json'))
            release_path = spec['owner_argv'][spec['owner_argv'].index('--release') + 1]
            shell = 'test "$(sha256sum -- ' + shlex.quote(release_path) + ')" = ' + shlex.quote(release_pin + '  ' + release_path) + ' && exec ' + shlex.join(['/usr/bin/env', '-u', 'CUDA_VISIBLE_DEVICES'] + [k + '=' + v for k, v in M.ENV.items()] + spec['owner_argv'])
            self.put(self.state / (role + '.stage_release.json'), {'action': 'stage_release', 'role': role,
                     'release_sha256': release_pin, 'same_control_stop_utc': self.stop.isoformat()})
            self.put(self.state / (role + '.staged.json'), {'release_sha256': release_pin,
                     'independent_issue_review': {'path': str(self.issue), 'sha256': self.issue_pin},
                     'capture_sha256': self.sha(self.state / (role + '.stage_release.json'))})
            self.put(self.state / (role + '.stage_release.external.json'), self.external(45, 55))
            self.put(self.state / (role + '.launch.json'), {'exact_owner_argv': spec['owner_argv'], 'remote_launch_shell': shell,
                     'release_sha256': release_pin, 'control_phase_sha256': self.phase_pin,
                     'original_tool_session_must_be_recorded': True, 'dispatched_utc': self.at(60)})
            self.put(self.state / (role + '.evaluation.command.json'), {'argv': ['ssh', '-T', '-F', str(self.prep / 'ssh_config'), M.ALIASES[role], shell],
                     'stop_utc': '2026-10-07T03:00:00+00:00', 'remote_closure_separately_required': True})
            tool = self.state / (role + '.tool_response.json'); self.put(tool, {'session_id': session_id, 'exit_code': None})
            self.put(self.state / (role + '.original_session.json'), {'original_tool_session': session_id, 'recorded_by': 'root',
                     'launch_sha256': self.sha(self.state / (role + '.launch.json')), 'observed_utc': self.at(65),
                     'evidence_file': str(tool), 'evidence_sha256': self.sha(tool)})
            capture = self.get(self.state / (role + '.verify_stage.json'))
            capture.update(action='observe_owner', release_sha256=release_pin,
                           clock={'host_utc': self.at(80), 'host_boot_id': spec['expected_boot_id'], 'host_monotonic_seconds': 1080.0},
                           root_reference_utc=self.at(81), owner=self.process(pid, 150, spec['owner_argv']),
                           children=[self.process(pid + 1, pid, release['streams'][0]['commands'][0])],
                           gpu_processes=[{'pid': pid + 1, 'gpu_uuid': release['gpu_uuids'][index]}])
            self.put(self.state / (role + '.observe_owner.json'), capture)
            raw = copy.deepcopy(capture); raw.pop('root_reference_utc'); self.put(self.state / (role + '.observe_owner.stdout'), raw)
            self.files[str(self.state / (role + '.observe_owner.stderr'))] = b''
            self.put(self.state / (role + '.observe_owner.external.json'), self.external(70, 85))
            self.put(self.state / (role + '.observe_owner.command.json'), {'stop_utc': self.stop.isoformat(),
                     'argv': ['ssh', '-T', '-F', str(self.prep / 'ssh_config'), M.ALIASES[role], shlex.join(['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '120s', M.PYTHON, '-I', '-S', '-B', '-c', self.files[str(self.pk / 'remote_control.py')].decode()])]})
            self.put(self.state / (role + '.owner_observed.json'), {'status': 'observed_original_final24_owner',
                     'control_phase_sha256': self.phase_pin, 'release_sha256': release_pin, 'original_tool_session': session_id,
                     'native_capture_sha256': self.sha(self.state / (role + '.observe_owner.json')),
                     'checked_utc': self.at(90), 'scientific_completion': False})
            completion['owner_observation_sha256'][role] = self.sha(self.state / (role + '.owner_observed.json'))
        self.put(self.state / 'control_completion.json', completion)
    def at(self, seconds): return (T + timedelta(seconds=seconds)).isoformat()
    def external(self, start, end): return {'exit_code': 0, 'local_transport_reaped': True, 'local_transport_timeout': False,
        'signals_to_own_local_group': [], 'failure': None, 'started_utc': self.at(start), 'observed_utc': self.at(end)}
    def process(self, pid, ppid, argv): return {'pid': pid, 'ppid': ppid, 'pgid': pid, 'sid': pid, 'start_ticks': 9000 + pid,
        'executable': self.pyenv['resolved_binary_path'], 'argv': argv}
    def contexts(self):
        return [patch.object(M, 'P', self.prep), patch.object(M, 'PK', self.pk), patch.object(M, 'PACKAGE_SHA', self.package_pin),
                patch.object(M, 'PREPARED_SHA', self.prepared_pin), patch.object(M, 'ISSUER_SHA', self.issuer_pin),
                patch.object(M, 'sha', self.sha), patch.object(M, 'read', self.read),
                patch.object(Path, 'is_file', lambda p: str(p) in self.files), patch.object(Path, 'is_symlink', lambda p: str(p) in self.symlinks),
                patch.object(Path, 'exists', lambda p: str(p) in self.files), patch.object(Path, 'resolve', lambda p: p),
                patch.object(Path, 'read_bytes', lambda p: self.files[str(p)]), patch.object(Path, 'read_text', lambda p: self.files[str(p)].decode())]


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture(); self.patches = self.f.contexts()
        for p in self.patches: p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
    def audit(self, seconds=150, monotonic=1150):
        return M.audit(self.f.state, self.f.issue, self.f.issue_pin, checked_at=T + timedelta(seconds=seconds), monotonic=monotonic)
    def mutate(self, name, change):
        path = self.f.state / name; obj = self.f.get(path); change(obj); self.f.put(path, obj)
    def capture(self, change, role='A'):
        self.mutate(role + '.observe_owner.json', change)
        obj = self.f.get(self.f.state / (role + '.observe_owner.json')); obj.pop('root_reference_utc')
        self.f.put(self.f.state / (role + '.observe_owner.stdout'), obj)
        self.mutate(role + '.owner_observed.json', lambda x: x.update(native_capture_sha256=self.f.sha(self.f.state / (role + '.observe_owner.json'))))
        self.refresh_observed(role)
    def refresh_observed(self, role='A'):
        self.mutate('control_completion.json', lambda x: x['owner_observation_sha256'].update({role: self.f.sha(self.f.state / (role + '.owner_observed.json'))}))
    def refresh_issue(self):
        self.f.issue_pin = self.f.sha(self.f.issue)
        for role in ('A', 'B'):
            self.mutate(role + '.staged.json', lambda x: x['independent_issue_review'].update(sha256=self.f.issue_pin))
    def test_valid_both_owners_children_and_scalar_only_evidence(self):
        result = self.audit(); self.assertEqual(set(result['roles']), {'A', 'B'})
        self.assertEqual(result['roles']['A']['original_tool_session'], 10001)
        self.assertFalse(result['scientific_completion']); self.assertFalse(result['all24_stage_completion_asserted'])
        self.assertTrue(result['eventual_original_tool_exits_and_stopped_native_closure_still_required'])
        self.assertTrue(all(not p.endswith(('.npy', '.npz', '.pt', '.pth')) for p in self.f.reads + self.f.hashes))
    def test_valid_B_unassigned_foreign_compute(self):
        self.capture(lambda x: x['gpu_processes'].append({'pid': 999, 'gpu_uuid': x['gpu_uuids'][0]}), 'B'); self.audit()
    def test_issuance_completion_after_stage_rejected(self):
        self.mutate('accepted_issue_review.json', lambda x: x.update(completed_utc=self.f.at(50))); self.refresh_issue()
        with self.assertRaisesRegex(ValueError, 'chronology'): self.audit()
    def test_launch_before_stage_completion_rejected(self):
        self.mutate('A.stage_release.external.json', lambda x: x.update(observed_utc=self.f.at(61)))
        with self.assertRaisesRegex(ValueError, 'chronology'): self.audit()
    def test_original_session_mismatch_rejected(self):
        self.mutate('A.tool_response.json', lambda x: x.update(session_id=10003))
        self.mutate('A.original_session.json', lambda x: x.update(evidence_sha256=self.f.sha(self.f.state / 'A.tool_response.json')))
        with self.assertRaisesRegex(ValueError, 'actual original execution-tool'): self.audit()
    def test_boolean_session_rejected(self):
        self.mutate('A.original_session.json', lambda x: x.update(original_tool_session=True))
        with self.assertRaisesRegex(ValueError, 'unique original root tool session'): self.audit()
    def test_duplicate_session_rejected(self):
        self.mutate('B.original_session.json', lambda x: x.update(original_tool_session=10001))
        with self.assertRaisesRegex(ValueError, 'unique original root tool session'): self.audit()
    def test_original_tool_exit_rejected(self):
        self.mutate('A.tool_response.json', lambda x: x.update(exit_code=0))
        self.mutate('A.original_session.json', lambda x: x.update(evidence_sha256=self.f.sha(self.f.state / 'A.tool_response.json')))
        with self.assertRaisesRegex(ValueError, 'actual original execution-tool'): self.audit()
    def test_all_completed_evaluation_artifacts_rejected(self):
        for suffix in ('evaluation.external.json', 'evaluation.stdout', 'evaluation.stderr', 'original_exit.json'):
            with self.subTest(suffix=suffix):
                path = str(self.f.state / ('A.' + suffix)); self.f.files[path] = b''
                with self.assertRaisesRegex(ValueError, 'not yet completed'): self.audit()
                del self.f.files[path]
    def test_wrong_owner_argv_rejected(self):
        self.capture(lambda x: x['owner']['argv'].append('--extra'))
        with self.assertRaisesRegex(ValueError, 'frozen owner argv'): self.audit()
    def test_child_parent_mismatch_rejected(self):
        self.capture(lambda x: x['children'][0].update(ppid=999))
        with self.assertRaisesRegex(ValueError, 'owner and native session'): self.audit()
    def test_child_native_session_mismatch_rejected(self):
        self.capture(lambda x: x['children'][0].update(sid=999))
        with self.assertRaisesRegex(ValueError, 'owner and native session'): self.audit()
    def test_child_gpu_mismatch_rejected(self):
        self.capture(lambda x: x['gpu_processes'][0].update(gpu_uuid=x['gpu_uuids'][1]))
        with self.assertRaisesRegex(ValueError, 'assigned physical GPU'): self.audit()
    def test_foreign_assigned_gpu_rejected(self):
        self.capture(lambda x: x['gpu_processes'].append({'pid': 999, 'gpu_uuid': x['gpu_uuids'][0]}))
        with self.assertRaisesRegex(ValueError, 'foreign compute'): self.audit()
    def test_boolean_native_identity_rejected(self):
        self.capture(lambda x: x['owner'].update(start_ticks=True))
        with self.assertRaisesRegex(ValueError, 'positive native process identity'): self.audit()
    def test_boolean_native_absence_rejected(self):
        self.capture(lambda x: x['native_absent'].update({'101': 1}))
        with self.assertRaisesRegex(ValueError, 'original owned processes absent'): self.audit()
    def test_raw_stdout_capture_mismatch_rejected(self):
        self.mutate('A.observe_owner.stdout', lambda x: x.update(hostname='other'))
        with self.assertRaisesRegex(ValueError, 'matches original helper stdout'): self.audit()
    def test_nonempty_stderr_rejected(self):
        self.f.files[str(self.f.state / 'A.observe_owner.stderr')] = b'warning\n'
        with self.assertRaisesRegex(ValueError, 'stderr empty'): self.audit()
    def test_altered_remote_helper_command_rejected(self):
        self.mutate('A.observe_owner.command.json', lambda x: x['argv'].__setitem__(-1, x['argv'][-1] + ' extra'))
        with self.assertRaisesRegex(ValueError, 'frozen native helper command'): self.audit()
    def test_remote_timeout_over120_rejected(self):
        self.mutate('A.observe_owner.command.json', lambda x: x['argv'].__setitem__(-1, x['argv'][-1].replace('120s', '121s')))
        with self.assertRaisesRegex(ValueError, 'frozen native helper command'): self.audit()
    def test_observation_timeout_rejected(self):
        self.mutate('A.observe_owner.external.json', lambda x: x.update(local_transport_timeout=True))
        with self.assertRaisesRegex(ValueError, 'transport completed successfully'): self.audit()
    def test_different_control_stop_rejected(self):
        self.capture(lambda x: x.update(same_control_stop_utc=self.f.at(601)))
        with self.assertRaisesRegex(ValueError, 'exact host release'): self.audit()
    def test_reset_control_phase_rejected(self):
        self.mutate('control_phase.json', lambda x: x.update(clock_restarted=True))
        with self.assertRaisesRegex(ValueError, 'Same original full600phase'): self.audit()
    def test_utc_monotonic_divergence_rejected(self):
        with self.assertRaisesRegex(ValueError, 'UTC/monotonic'): self.audit(monotonic=1200)
    def test_original_control_margin_rejected(self):
        with self.assertRaisesRegex(ValueError, 'UTC/monotonic'): self.audit(seconds=595, monotonic=1595)
    def test_stale_native_clock_rejected(self):
        self.capture(lambda x: x['clock'].update(host_utc=self.f.at(-400)))
        with self.assertRaisesRegex(ValueError, 'fresh same-boot'): self.audit()
    def test_changed_prior_accepted_evidence_rejected(self):
        self.f.files[str(self.f.extra_evidence)] += b' '
        with self.assertRaisesRegex(ValueError, 'Exact evidence bytes'): self.audit()
    def run_main(self, audit_override=None, finished=160):
        output = self.f.state / 'new_review.json'
        with patch.object(M, 'audit', audit_override or M.audit), patch.object(M, 'now', side_effect=[T + timedelta(seconds=150), T + timedelta(seconds=finished)]), patch.object(M.time, 'monotonic', side_effect=[1150, 1000 + finished]), patch.object(sys, 'argv', ['startup', '--state', str(self.f.state), '--issue-review', str(self.f.issue), '--issue-review-sha256', self.f.issue_pin, '--output', str(output)]):
            M.main()
    def test_main_rehashes_prior_evidence(self):
        original = M.audit
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs); self.f.files[str(self.f.extra_evidence)] += b' '; return result
        with self.assertRaisesRegex(ValueError, 'Evidence changed before receipt publication'): self.run_main(wrapped)
    def test_main_rejects_completed_transport_appearing_during_audit(self):
        original = M.audit
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs); self.f.files[str(self.f.state / 'B.original_exit.json')] = b'{}'; return result
        with self.assertRaisesRegex(ValueError, 'Original evaluation completed during'): self.run_main(wrapped)
    def test_main_rejects_final_control_margin_exhaustion(self):
        with self.assertRaisesRegex(ValueError, 'Original control stop exhausted'): self.run_main(finished=595)


if __name__ == '__main__': unittest.main(verbosity=2)
