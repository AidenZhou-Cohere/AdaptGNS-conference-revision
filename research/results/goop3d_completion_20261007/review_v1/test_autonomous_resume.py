"""Independent synthetic-only commit/restart tests. No real model/data imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

PATH = Path(__file__).resolve().parents[1]/'autonomous_v1/run_worker.py'
spec = importlib.util.spec_from_file_location('d3_test_worker', PATH)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


CELL = {'cell_id': 'base_seed0__valid__000000__base', 'seed': 0, 'arm': 'base',
        'split': 'valid', 'source_index': 0, 'policy': 'base'}
IDENTITY, PROTOCOL = 'a'*64, 'b'*64


def row(guard=False):
    return {'completion_cell_id': CELL['cell_id'], 'protocol_sha256': PROTOCOL,
            'training_seed': CELL['seed'], **{k: CELL[k] for k in ('arm', 'split', 'source_index', 'policy')},
            'status': 'failed' if guard else 'complete',
            'failure': {'category': 'candidate_pair_resource_guard'} if guard else None}


def trace():
    return {'synthetic_only': np.arange(12, dtype=np.float32).reshape(4, 3)}


class CommitTests(unittest.TestCase):
    def test_committed_guard_and_success_reused_without_selection(self):
        for guarded in (False, True):
            with self.subTest(guarded=guarded), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                saved = W.commit_cell(out, CELL, 'attempt_000001', row(guarded), trace(), np, IDENTITY, PROTOCOL)
                with patch.object(W, 'commit_cell', side_effect=AssertionError('no science rerun')):
                    recovered = W.committed_cells(out, [CELL], IDENTITY, PROTOCOL)
                self.assertEqual(recovered, {CELL['cell_id']: saved})
                self.assertEqual(recovered[CELL['cell_id']]['failure'], row(guarded)['failure'])

    def test_npz_and_row_without_marker_preserved_then_new_attempt_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); save = W.save
            def fail_marker(path, doc, replace=False):
                if Path(path).name == 'commit.json':
                    raise OSError('synthetic commit publication failure')
                return save(path, doc, replace)
            with patch.object(W, 'save', side_effect=fail_marker):
                with self.assertRaises(OSError):
                    W.commit_cell(out, CELL, 'attempt_000001', row(), trace(), np, IDENTITY, PROTOCOL)
            old = out/'cells'/CELL['cell_id']/'attempt_000001'
            originals = {p.name: p.read_bytes() for p in old.iterdir()}
            self.assertEqual(set(originals), {'trace.npz', 'row.json'})
            self.assertEqual(W.committed_cells(out, [CELL], IDENTITY, PROTOCOL), {})
            saved = W.commit_cell(out, CELL, 'attempt_000002', row(), trace(), np, IDENTITY, PROTOCOL)
            self.assertEqual({p.name: p.read_bytes() for p in old.iterdir()}, originals)
            self.assertEqual(W.committed_cells(out, [CELL], IDENTITY, PROTOCOL), {CELL['cell_id']: saved})

    def test_committed_cell_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            W.commit_cell(tmp, CELL, 'attempt_000001', row(), trace(), np, IDENTITY, PROTOCOL)
            files = {str(p): p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError, 'duplicate committed'):
                W.commit_cell(tmp, CELL, 'attempt_000002', row(), trace(), np, IDENTITY, PROTOCOL)
            self.assertEqual({str(p): p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}, files)

    def test_modified_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            W.commit_cell(tmp, CELL, 'attempt_000001', row(), trace(), np, IDENTITY, PROTOCOL)
            archive = next(Path(tmp).rglob('trace.npz')); archive.write_bytes(b'changed synthetic')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                W.committed_cells(tmp, [CELL], IDENTITY, PROTOCOL)

    def test_marker_identity_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            W.commit_cell(tmp, CELL, 'attempt_000001', row(), trace(), np, IDENTITY, PROTOCOL)
            marker = next(Path(tmp).rglob('commit.json')); original = W.read(marker)
            for field, value in [('identity_sha256', 'c'*64), ('row_file', '../../outside.json')]:
                with self.subTest(field=field):
                    corrupt = copy.deepcopy(original); corrupt[field] = value
                    marker.write_text(json.dumps(corrupt))
                    with self.assertRaises(ValueError):
                        W.committed_cells(tmp, [CELL], IDENTITY, PROTOCOL)

    def test_row_identity_rejected_even_after_row_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            W.commit_cell(tmp, CELL, 'attempt_000001', row(), trace(), np, IDENTITY, PROTOCOL)
            marker = next(Path(tmp).rglob('commit.json')); record = W.read(marker)
            path = marker.parent/record['row_file']; wrong = W.read(path); wrong['source_index'] = 1
            path.write_text(json.dumps(wrong)); record['row_sha256'] = W.sha(path)
            marker.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'row identity differs'):
                W.committed_cells(tmp, [CELL], IDENTITY, PROTOCOL)

    def test_json_directory_fsync_follows_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'data.json'; calls = []
            def sync(path):
                calls.append(Path(path)); self.assertTrue(target.is_file())
                self.assertEqual(W.read(target), {'synthetic': 1})
            with patch.object(W, 'fsync_directory', side_effect=sync):
                W.save(target, {'synthetic': 1})
            self.assertEqual(calls, [target.parent])


class OwnerTests(unittest.TestCase):
    def test_identity_requires_explicit_resume_and_exact_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'identity.json'; identity = {'source': 'a'}
            pin = W.bind_json(p, identity, False)
            self.assertEqual(W.bind_json(p, identity, True), pin)
            with self.assertRaises(ValueError): W.bind_json(p, identity, False)
            with self.assertRaises(ValueError): W.bind_json(p, {'source': 'b'}, True)

    def test_prior_owner_exact_start_ticks_not_pid_only(self):
        prior = {'pid': 123, 'start_ticks': 99, 'boot_id': 'old', 'hostname': 'synthetic'}
        now = {'boot_id': 'old', 'hostname': 'synthetic'}
        with patch.object(W, 'process_start', return_value=99): self.assertTrue(W.prior_owner_is_live(prior, now))
        with patch.object(W, 'process_start', return_value=100): self.assertFalse(W.prior_owner_is_live(prior, now))
        with patch.object(W, 'process_start', return_value=None): self.assertFalse(W.prior_owner_is_live(prior, now))
        with patch.object(W, 'process_start', side_effect=AssertionError('different boot')):
            self.assertFalse(W.prior_owner_is_live(prior, {**now, 'boot_id': 'new'}))

    def test_kernel_lock_refuses_duplicate_and_dead_owner_resume_preserves_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'worker'; native = {'pid': 123, 'start_ticks': 99, 'boot_id': 'same', 'hostname': 'synthetic', 'started_utc': 'synthetic'}
            with patch.object(W, 'owner_identity', return_value=copy.deepcopy(native)):
                with W.own_worker(out, False) as (attempt, _):
                    W.save(attempt/'failed_attempt.json', {'synthetic': 'preserve'})
                    with self.assertRaisesRegex(RuntimeError, 'live kernel lock'):
                        with W.own_worker(out, True): pass
            first = {str(p): p.read_bytes() for p in (out/'attempts'/'attempt_000001').iterdir() if p.is_file()}
            with patch.object(W, 'owner_identity', return_value={**native, 'pid': 124, 'start_ticks': 100}), patch.object(W, 'process_start', return_value=None):
                with W.own_worker(out, True) as (attempt, _):
                    self.assertEqual(attempt.name, 'attempt_000002')
                    self.assertEqual(W.read(attempt/'previous_owner.json')['pid'], 123)
            self.assertEqual({str(p): p.read_bytes() for p in (out/'attempts'/'attempt_000001').iterdir() if p.is_file()}, first)


if __name__ == '__main__':
    unittest.main(verbosity=2)
