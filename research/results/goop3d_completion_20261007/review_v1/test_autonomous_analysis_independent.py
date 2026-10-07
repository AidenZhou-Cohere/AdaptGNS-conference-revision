"""Extra synthetic-only parallelism, transaction fault and semantic-corruption cases."""
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

FOLDER = Path(__file__).resolve().parents[1]/'autonomous_analysis_v1'
sys.path.insert(0, str(FOLDER))
import test_analysis as fixtures
a = fixtures.a


def task_for(folder):
    row, arrays = fixtures.fixture()
    archive = folder/'trace.npz'
    with archive.open('wb') as stream:
        fixtures.np.savez_compressed(stream, **arrays)
    row_path = folder/'row.json'; row_pin = a.atomic(row_path, row)
    return {'key':'synthetic_parallel', 'arm':'base', 'seed':0, 'split':'test',
            'item':{k:row[k] for k in ('source_index','trajectory_id','particles','size_group','policy')},
            'row_file':str(row_path), 'row_sha256':row_pin, 'artifact_file':str(archive),
            'artifact_sha256':a.sha(archive), 'artifact_bytes':archive.stat().st_size,
            'bounds':fixtures.namespace['BOUNDS'].tolist()}


class IndependentCases(unittest.TestCase):
    def test_fault_between_row_and_index_preserves_recoverable_orphan(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as name:
            out = Path(name); store = a.Checkpoints(out, {}, False); original = a.atomic
            def fail_index(path, value):
                if Path(path) == store.path:
                    raise OSError('synthetic index publication failure')
                return original(path, value)
            with patch.object(a, 'atomic', side_effect=fail_index):
                with self.assertRaises(OSError): store.commit('cell0', 'pin', {'synthetic':1})
            self.assertEqual(store.index['completed'], {})
            self.assertEqual(store.index['failed'], {})
            orphan = (out/'rows/cell0.json').read_bytes()
            resumed = a.Checkpoints(out, {}, True)
            self.assertIsNone(resumed.read('cell0','pin'))
            resumed.commit('cell0', 'pin', {'synthetic':1})
            self.assertEqual((out/'rows/cell0.json').read_bytes(), orphan)
            self.assertEqual(resumed.read('cell0','pin'), {'synthetic':1})

    def test_actual_spawn_dispatch_matches_serial_result(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as name:
            task = task_for(Path(name)); serial = a.audit_cell(task)
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=a.load_science, initargs=(str(fixtures.PREP),)) as pool:
                got = list(pool.map(a.audit_cell, [task, copy.deepcopy(task)]))
            self.assertEqual(got, [serial, serial])

    def test_semantic_mismatch_rejected_after_json_repin(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as name:
            task = task_for(Path(name)); path = Path(task['row_file'])
            row = a.strict(path.read_bytes()); row['mse_per_step'][0] += 1.
            task['row_sha256'] = a.atomic(path, row)
            with self.assertRaises(ValueError): a.audit_cell(task)


if __name__ == '__main__':
    unittest.main(verbosity=2)
