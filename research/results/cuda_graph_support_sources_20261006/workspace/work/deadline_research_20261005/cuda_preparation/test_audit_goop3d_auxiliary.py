"""Synthetic auxiliary-only tests; no official trajectory/model/GPU access."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import audit_goop3d_auxiliary as audit


class Goop3DAuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='goop3d_context_synthetic_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'report.json'

    def record(self, split='train', index=0, present=True):
        record = {'id': f'{split}:{index:06d}', 'source_index': index,
                  'positions': {'shape': [301, 2, 3]}}
        if present:
            (self.root / split).mkdir(exist_ok=True)
            bits = np.resize(np.array([0, 0x80000000, 0x7fc12345, 0x7f800000, 0xff800000], dtype='<u4'), 301)
            values = bits.view('<f4').reshape(301, 1)
            path = self.root / split / f'step_context_{index:06d}.npy'
            np.save(path, values, allow_pickle=False)
            record['step_context'] = {'path': f'{split}/{path.name}', 'shape': [301, 1], 'dtype': '<f4',
                                      'sha256': audit.sha(path), 'size_bytes': path.stat().st_size,
                                      'use': 'unreviewed'}
        return record

    def fixture(self):
        metadata_path = Path(__file__).parent.parent / 'extension_feasibility_20261006/Goop-3D_metadata_json.json'
        raw = metadata_path.read_bytes()
        (self.root / 'metadata.json').write_bytes(raw)
        metadata = json.loads(raw)
        structural = {'schema': 'official_goop3d_numeric_preparation_v1', 'status': 'complete_structural_only',
                      'dataset': 'Goop-3D', 'wrapper_sha256': audit.CONVERTER_SHA,
                      'metadata_sha256': audit.METADATA_SHA, 'splits': {}}
        for split, records in [('train', [self.record(), self.record(index=1, present=False)]),
                                ('valid', [self.record(split='valid')])]:
            manifest = {'dataset': 'Goop-3D', 'split': split, 'metadata': metadata,
                        'metadata_sha256': audit.METADATA_SHA, 'converter_sha256': audit.CONVERTER_SHA,
                        'record_count': len(records), 'records': records}
            path = self.root / (split + '.json')
            path.write_text(json.dumps(manifest))
            structural['splits'][split] = {'manifest_sha256': audit.sha(path), 'record_count': len(records)}
        path = self.root / 'structural.json'
        path.write_text(json.dumps(structural))
        return ['--execute', '--numeric-root', str(self.root), '--structural-report', str(path),
                '--structural-sha256', audit.sha(path), '--output', str(self.output)]

    def test_default_does_not_read_arrays(self):
        with mock.patch.object(audit, 'census_record', side_effect=AssertionError('no arrays')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit.main([]), 0)
        self.assertFalse(json.loads(output.getvalue())['execution'])

    def test_raw_nan_negativezero_infinity_bits_and_counts(self):
        record = self.record()
        row, path = audit.census_record(self.root, 'train', record, np)
        self.assertTrue(row['present'])
        self.assertEqual(row['elements'], 301)
        self.assertEqual(row['finite_count'], 121)
        self.assertEqual(row['nan_count'], 60)
        self.assertEqual(row['positive_infinity_count'], 60)
        self.assertEqual(row['negative_infinity_count'], 60)
        self.assertEqual(set(row['unique_float32_bits_hex']), {'00000000', '80000000', '7fc12345', '7f800000', 'ff800000'})
        self.assertEqual(sum(item['count'] for item in row['float32_bit_counts']), 301)
        self.assertEqual(audit.sha(path), record['step_context']['sha256'])

    def test_absent_auxiliary_is_explicit(self):
        row, path = audit.census_record(self.root, 'train', self.record(present=False), np)
        self.assertEqual(row, {'id': 'train:000000', 'source_index': 0, 'present': False})
        self.assertIsNone(path)

    def test_bad_hash_shape_and_path_rejected(self):
        for field, value in [('sha256', '0' * 64), ('shape', [300, 1]), ('path', '../outside.npy')]:
            record = self.record()
            record['step_context'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.census_record(self.root, 'train', record, np)

    def test_complete_counts_are_observed_and_both_splits_covered(self):
        argv = self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(argv), 0)
        result = json.loads(self.output.read_text())
        self.assertEqual(result['status'], 'all_preserved_auxiliary_bytes_verified')
        self.assertEqual(result['splits']['train']['record_count'], 2)
        self.assertEqual(result['splits']['valid']['record_count'], 1)
        self.assertEqual(result['splits']['train']['auxiliary_present_count'], 1)
        self.assertEqual(result['splits']['train']['auxiliary_absent_count'], 1)
        self.assertFalse(result['scientific_training_admitted'])
        self.assertFalse(result['test_accessed'])

    def test_failed_array_retains_partial_census_and_source(self):
        argv = self.fixture()
        path = self.root / 'valid/step_context_000000.npy'
        with path.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Auxiliary bytes differ'):
            audit.main(argv)
        result = json.loads(self.output.read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['partial_census_retained'])
        self.assertEqual(len(result['splits']['train']['records']), 2)
        self.assertTrue(path.exists())

    def test_existing_output_is_not_replaced(self):
        argv = self.fixture()
        self.output.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'Existing census'):
            audit.main(argv)
        self.assertEqual(self.output.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main(verbosity=2)
