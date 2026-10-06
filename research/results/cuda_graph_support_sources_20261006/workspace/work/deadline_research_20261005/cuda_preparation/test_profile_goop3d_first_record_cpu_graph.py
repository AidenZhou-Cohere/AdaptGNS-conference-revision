"""Synthetic CPU-only checks; no official arrays, model import or CUDA init."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import profile_goop3d_first_record_cpu_graph as profile


class CpuGraphProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench, cls.graph = profile.load_pinned_helpers(Path(__file__).parent)

    def fixture(self):
        g = self.graph
        points = g.np.array(list(itertools.product((.4, .422), repeat=3)), dtype=g.np.float32)
        histories = g.np.repeat(points[:, None, :], 6, axis=1)
        position = g.torch.from_numpy(g.np.concatenate((histories, histories)))
        types = g.torch.full((16,), 7, dtype=g.torch.long)
        data = (position, types, g.torch.tensor([8, 8]), position[:, -1].clone())
        noise = g.host_noise(position.shape, types, 0, 0)
        return data, noise

    def test_default_has_no_helper_or_data_work(self):
        with mock.patch.object(profile, 'load_pinned_helpers', side_effect=AssertionError('no helper')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(profile.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result['cuda_calls'])
        self.assertFalse(result['model_or_optimizer'])

    def test_no_gns_import_and_no_cuda_initialization_in_graph_work(self):
        self.assertFalse(any(name == 'gns' or name.startswith('gns.') for name in sys.modules))
        data, noise = self.fixture()
        with mock.patch.object(self.graph.torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
            row = profile.graph_stages(data, noise, 'expanded25', self.graph)
        self.assertEqual(len(row['graph_ledger']), 2)
        self.assertTrue(all(item['expanded'] for item in row['graph_ledger']))
        self.assertFalse(any(name == 'gns' or name.startswith('gns.') for name in sys.modules))

    def test_literal_native_algorithm_matches_strict_cap_reference_in_3d(self):
        g = self.graph
        for points in (g.np.zeros((130, 3), dtype=g.np.float32),
                       g.np.array([[0, 0, 0], [0, 0, .024], [0, 0, .025], [0, 0, .03]], dtype=g.np.float32)):
            expected = g.ordered_edges(points, g.strict_pairs(points, .025).base, True, 128)
            native = profile.native_graph_cpu(points, [len(points)], g.np, g.cKDTree)
            g.np.testing.assert_array_equal(native, expected)

    def test_native_batch_partitions_never_cross(self):
        g = self.graph
        points = g.np.zeros((10, 3), dtype=g.np.float32)
        native = profile.native_graph_cpu(points, [5, 5], g.np, g.cKDTree)
        g.np.testing.assert_array_equal(native[0] // 5, native[1] // 5)
        self.assertEqual(native.shape[1], 50)

    def test_stages_preserve_exact_repeated_graph_hashes_and_budget(self):
        data, noise = self.fixture()
        for case in ('base', 'expanded25'):
            first = profile.graph_stages(data, noise, case, self.graph)
            second = profile.graph_stages(data, noise, case, self.graph)
            self.assertEqual(first['graph_ledger'], second['graph_ledger'])
            self.assertEqual(first['final_batch_edge_sha256'], second['final_batch_edge_sha256'])
            self.assertTrue(all(value >= 0 for value in first['stage_seconds'].values()))
            expected = sum(row['native_directed_edges'] + 2 * row['selected_optional_pairs'] for row in first['graph_ledger'])
            self.assertEqual(first['final_directed_edges'], expected)

    def test_changed_pinned_helpers_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'benchmark_goop3d_first_record_v2.py').write_text('raise RuntimeError()')
            with self.assertRaisesRegex(ValueError, 'helper bytes'):
                profile.load_pinned_helpers(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
