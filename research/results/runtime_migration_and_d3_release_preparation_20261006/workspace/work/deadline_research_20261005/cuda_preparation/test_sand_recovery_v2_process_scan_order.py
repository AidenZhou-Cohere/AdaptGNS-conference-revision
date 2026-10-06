"""Independent scalar regressions; no OS children, CUDA, arrays, or checkpoints."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = local_module('_sand_v2_independent_scalar_fixtures', HERE / 'test_supervise_sand_runtime_migration_recovery_v1.py')
M = local_module('_sand_v2_independent_owner', HERE / 'supervise_sand_runtime_migration_recovery_v2.py')
F.M = M


class PendingQueryRuntime(F.FakeRuntime):
    def __init__(self, context, args, now, inject=None):
        super().__init__(context, args, now, 'complete')
        self.owner_command[2] = str(Path(M.__file__).resolve())
        self.inject = inject
        self.scans, self.submissions, self.overlaps = [], [], []
        self.rejected = None

    def process_inventory(self):
        self.scans.append(self.t)
        rows = super().process_inventory()
        # A pending query represents fork-to-exec argv inheritance. Its child
        # looks exactly like this owner until the query completes and is reaped.
        if self.future is not None:
            self.overlaps.append(self.t)
            clone = dict(self.identity(os.getpid()), pid=90001, ppid=os.getpid(), start_ticks=90001)
            rows.append(clone)
        if self.inject and len(self.created) == 4:
            exemplar = self.identity(self.created[0]['process'].pid)
            parent = {'foreign': 17, 'duplicate': os.getpid(), 'descendant': exemplar['pid']}[self.inject]
            self.rejected = dict(exemplar, pid=90002, ppid=parent, start_ticks=90002)
            rows.append(self.rejected)
        return rows

    def gpu_submit(self, owned):
        if self.future is not None:
            raise AssertionError('A prior query was neither completed nor consumed')
        self.submissions.append(self.t)
        self.future = {'complete_at': self.t + 2, 'owned': dict(owned)}

    def gpu_result(self):
        if self.future is None or self.t < self.future['complete_at']:
            return None
        result = [], self.future['owned']
        self.future = None
        return result

    def reap(self, args, child, record, reaped, launch, manifest):
        if self.t < 65 and not child['signals']:
            return None
        return super().reap(args, child, record, reaped, launch, manifest)


def contains_exact_row(value, wanted):
    if isinstance(value, dict):
        return value == wanted or any(contains_exact_row(v, wanted) for v in value.values())
    if isinstance(value, list):
        return any(contains_exact_row(v, wanted) for v in value)
    return False


class ProcessScanRegressions(unittest.TestCase):
    def setUp(self):
        self.fixture = F.OwnerTests('test_runtime_only_uuid_may_differ_including_types')
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def prepare(self, inject=None):
        args, context, _ = self.fixture.context()
        runtime = PendingQueryRuntime(context, args, self.fixture.now, inject)
        audit = SimpleNamespace(audit_recovery_job=lambda release, job, **kw: {'id': job['id']},
                                audit_recovery_pairing=lambda jobs: {'count': len(jobs)})
        return args, context, runtime, audit

    def test_pending_query_fork_window_never_overlaps_process_scan(self):
        args, context, runtime, audit = self.prepare()
        with patch.object(F.S.subprocess, 'Popen', side_effect=AssertionError('OS launch forbidden')):
            result = M.run_owned(args, context, runtime, audit)
        self.assertEqual(result['state'], 'verified_recovery_endpoints')
        self.assertEqual(runtime.overlaps, [])
        self.assertGreaterEqual(len(runtime.submissions), 3)
        periodic_scans = [t for t in runtime.scans if t > 0]
        self.assertGreaterEqual(len(periodic_scans), 3)
        self.assertEqual(runtime.stopped, [])
        self.assertTrue(all(x['process'].returncode == 0 for x in runtime.created))
        for submitted in runtime.submissions:
            self.assertFalse(any(submitted < scanned < submitted + 2 for scanned in runtime.scans))

    def assert_rejected_without_exemption(self, kind):
        args, context, runtime, audit = self.prepare(kind)
        with patch.object(F.S.subprocess, 'Popen', side_effect=AssertionError('OS launch forbidden')):
            with self.assertRaises(ValueError):
                M.run_owned(args, context, runtime, audit)
        retained = json.loads((args.output_dir / 'wave_A.json').read_text())
        self.assertEqual(retained['state'], 'failed')
        self.assertIn('Foreign, duplicate or old scientific', retained['error'])
        self.assertIsNotNone(runtime.rejected)
        self.assertTrue(contains_exact_row(retained, runtime.rejected), 'Rejected exact native identity was not retained')
        diagnostic = retained['rejected_process_inventory']
        self.assertEqual(diagnostic['rejected_row'], runtime.rejected)
        self.assertEqual({x['pid'] for x in diagnostic['full_inventory']},
                         {os.getpid(), 5000, 5001, 5002, 5003, 90002})
        self.assertEqual(diagnostic['checked_utc'], self.fixture.now.isoformat())
        self.assertEqual(diagnostic['reason'], 'Foreign, duplicate or old scientific owner/worker exists on recovery host')
        self.assertTrue(set(runtime.stopped) <= {5000, 5001, 5002, 5003})
        self.assertNotIn(90002, runtime.stopped)
        self.assertEqual(retained['unreaped_owned_children'], [])

    def test_foreign_scientific_peer_still_fails_and_persists_identity(self):
        self.assert_rejected_without_exemption('foreign')

    def test_unregistered_direct_duplicate_still_fails_and_persists_identity(self):
        self.assert_rejected_without_exemption('duplicate')

    def test_descendant_scientific_process_still_fails_and_persists_identity(self):
        self.assert_rejected_without_exemption('descendant')

    def test_direct_scan_guard_rejects_active_query_before_inventory(self):
        args, context, runtime, audit = self.prepare()
        owner = runtime.identity(os.getpid())
        runtime.gpu_submit({})
        with self.assertRaisesRegex(ValueError, 'completed and consumed GPU query'):
            M.check_processes(context, runtime, [], owner)
        self.assertEqual(runtime.scans, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
