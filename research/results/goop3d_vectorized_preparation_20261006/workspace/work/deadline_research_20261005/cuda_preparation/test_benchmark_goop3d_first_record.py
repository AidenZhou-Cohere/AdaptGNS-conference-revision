"""Tiny CPU/synthetic tests only; never open official arrays or use CUDA."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import benchmark_goop3d_first_record as bench

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / 'outputs/AdaptGNS'
GRAPH_PATH = Path(__file__).with_name('goop3d_graph_support.py')
graph, model_io, losses = bench.load_helpers(REPO, GRAPH_PATH)
torch, np = graph.torch, graph.np
torch.set_num_threads(1)
META = {'dim': 3, 'bounds': [[.2, .8]] * 3, 'default_connectivity_radius': .025,
        'vel_mean': [0., 0., 0.], 'vel_std': [.01, .012, .014],
        'acc_mean': [0., 0., 0.], 'acc_std': [.001, .0012, .0014]}


def batch():
    points = np.array(list(itertools.product((.4, .422), repeat=3)), dtype=np.float32)
    positions = np.repeat(points[:, None, :], 6, axis=1)
    positions[:, :, 2] += np.arange(6, dtype=np.float32)[None] * .0001
    positions = torch.from_numpy(np.concatenate((positions, positions)))
    types = torch.full((16,), 7, dtype=torch.long)
    labels = positions[:, -1] + torch.tensor([0., 0., .00011])
    return positions, types, torch.tensor([8, 8]), labels


def model():
    torch.manual_seed(0)
    return model_io.build_simulator(META, graph.NOISE, graph.NOISE, torch.device('cpu'),
        detach_variance_features=True, radius_backend='scipy').train()


class Goop3DFirstRecordTests(unittest.TestCase):
    def test_default_describes_without_helpers_inputs_or_cuda(self):
        with mock.patch.object(bench, 'load_helpers', side_effect=AssertionError('no import')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(bench.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['total_optimizer_updates'], 8)
        self.assertEqual(result['likelihood_log_coefficient'], 1.5)
        self.assertEqual(result['vector_risk'], '3q')
        self.assertFalse(result['test_accessed'])
        self.assertFalse(result['scientific_training_admitted'])

    def test_default_execution_is_explicit_and_has_no_test_or_resume_switch(self):
        with self.assertRaisesRegex(ValueError, 'Explicit repository'):
            bench.parse_args(['--execute'])
        for option in ('--test', '--resume', '--updates'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.parse_args([option])

    def test_changed_adapter_refused_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'bad.py'
            bad.write_text("raise RuntimeError('must not import')")
            with self.assertRaisesRegex(ValueError, 'adapter hash'):
                bench.load_helpers(REPO, bad)

    def test_strict_pairs_use_third_coordinate_and_match_brute_force(self):
        points = np.array([[0, 0, 0], [0, 0, .024], [0, 0, .025], [0, 0, .030]], dtype=np.float32)
        result = graph.strict_pairs(points, .025)
        expected_base, expected_extra = [], []
        for a, b in itertools.combinations(range(len(points)), 2):
            distance = np.linalg.norm(points[a] - points[b])
            if distance < .025:
                expected_base.append((a, b))
            elif distance < .025 * 1.267:
                expected_extra.append((a, b))
        self.assertEqual(graph.pair_set(result.base), set(expected_base))
        self.assertEqual(graph.pair_set(result.extra), set(expected_extra))
        self.assertTrue(len(result.extra))
        self.assertNotEqual(len(result.base), 6)

    def test_2d_nonfinite_and_bad_radius_refused(self):
        for points, radius in [(np.zeros((4, 2)), .025), (np.full((4, 3), np.nan), .025),
                               (np.zeros((4, 3)), float('nan')), (np.zeros((4, 3)), 0)]:
            with self.subTest(shape=points.shape, radius=radius), self.assertRaises(ValueError):
                graph.strict_pairs(points, radius)

    def test_native_cap_and_tie_order_in_three_dimensions(self):
        points = np.zeros((130, 3), dtype=np.float32)
        pairs = graph.strict_pairs(points, .025).base
        ordered = graph.ordered_edges(points, pairs, True, 128)
        simulator = model()
        sources, targets = simulator._compute_graph_connectivity(torch.from_numpy(points), torch.tensor([130]), .025)
        native = torch.stack((sources, targets)).numpy()
        np.testing.assert_array_equal(ordered, native)
        self.assertEqual(ordered.shape[1], 130 * 128)

    def test_forced_optional_exact_budget_prefix_and_batch_partition(self):
        data, simulator = batch(), model()
        noisy = data[0]
        _, native, _ = simulator._encoder_preprocessor(noisy, data[2], data[1])
        for arm in ('base', 'mix', 'expanded25'):
            torch_state = torch.get_rng_state().clone()
            edges, ledger = graph.append_optional_edges(noisy, data[2], native, .025, 0, 0, arm)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            self.assertTrue(torch.equal(edges[:, :native.shape[1]], native))
            self.assertTrue(torch.equal(edges[0] // 8, edges[1] // 8))
            for row in ledger:
                active = arm == 'expanded25' or (arm == 'mix' and row['exposure_coin'])
                self.assertEqual(row['selected_optional_pairs'], row['annulus_pairs'] // 4 if active else 0)
                self.assertEqual(row['expanded'], active)
                if arm == 'expanded25':
                    self.assertGreater(row['selected_optional_pairs'], 0)
            self.assertEqual(edges.shape[1], native.shape[1] + 2 * sum(row['selected_optional_pairs'] for row in ledger))

    def test_candidate_guard_precedes_native_neighbor_materialization(self):
        data = batch()
        simulator = mock.Mock(_connectivity_radius=.025)
        with mock.patch.object(graph, 'MAX_PAIRS', 0), self.assertRaisesRegex(ValueError, 'candidate_pair_resource_guard'):
            graph.forward_batch(simulator, data, torch.zeros_like(data[0]), torch.device('cpu'), 0, 0, 'base')
        simulator._encoder_preprocessor.assert_not_called()

    def test_noise_is_dimension_generic_kinematic_masked_and_rng_private(self):
        data = batch()
        data[1][0] = 3
        before = torch.get_rng_state().clone()
        noise = graph.host_noise(data[0].shape, data[1], 0, 0)
        self.assertEqual(noise.shape, (16, 6, 3))
        self.assertTrue(torch.equal(noise[0], torch.zeros_like(noise[0])))
        self.assertTrue(torch.equal(noise[:, 0], torch.zeros_like(noise[:, 0])))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(noise, graph.host_noise(data[0].shape, data[1], 0, 0)))

    def test_base_forward_matches_native_full_architecture_in_3d(self):
        data, simulator = batch(), model()
        noise = graph.host_noise(data[0].shape, data[1], 0, 0)
        prediction, head, target, _ = graph.forward_batch(simulator, data, noise, torch.device('cpu'), 0, 0, 'base')
        ref = simulator(next_positions=data[3], position_sequence_noise=noise, position_sequence=data[0],
                        nparticles_per_example=data[2], particle_types=data[1], material_property=None,
                        augment_radius_prob=0., augment_radius_factor=1.267)
        for actual, expected in zip((prediction, head, target), ref):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(prediction.shape, (16, 3))
        self.assertEqual(simulator._checkpoint_config['nnode_in'], 37)
        self.assertEqual(simulator._checkpoint_config['nedge_in'], 4)

    def test_3d_likelihood_coefficient_and_faithful_gradient(self):
        prediction = torch.tensor([[1., 2., 3.], [9., 9., 9.]], requires_grad=True)
        target = torch.zeros_like(prediction)
        q = torch.tensor([2., 4.], requires_grad=True)
        mask = torch.tensor([True, False])
        nll = losses.acceleration_loss(prediction, target, mask, q, 'nll')
        self.assertAlmostEqual(float(nll.detach()), 14 / 4 + 1.5 * np.log(2), places=6)
        faithful = losses.acceleration_loss(prediction, target, mask, q, 'faithful')
        faithful.backward()
        np.testing.assert_array_equal(prediction.grad.numpy(), [[2., 4., 6.], [0., 0., 0.]])
        self.assertAlmostEqual(float(q.grad[0]), -14 / 8 + 1.5 / 2, places=6)

    def test_vector_risk_is_three_times_coordinate_variance(self):
        prediction = torch.tensor([[1., 2., 3.]])
        result = graph.normalized_statistics(prediction, torch.tensor([2.]), torch.zeros_like(prediction), torch.tensor([True]))
        self.assertEqual(result['vector_mse'], 14.)
        self.assertAlmostEqual(result['coordinate_mse'], 14 / 3, places=6)
        self.assertEqual(result['mean_vector_risk_3q'], 6.)
        self.assertAlmostEqual(result['constant_free_gaussian_nll'], 14 / 4 + 1.5 * np.log(2), places=6)

    def test_expanded_cpu_update_has_finite_3d_gradients_and_adam(self):
        data, simulator = batch(), model()
        noise = graph.host_noise(data[0].shape, data[1], 0, 0)
        optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4, foreach=False, fused=False)
        context = {}
        loss, ledger, diagnostics = bench.step_once(simulator, optimizer, data, noise, torch.device('cpu'),
                                                  'expanded25', graph, losses, context)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(context['optimizer_step_returned'])
        self.assertTrue(all(row['expanded'] and row['selected_optional_pairs'] > 0 for row in ledger))
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
        self.assertTrue(all(float(value['step']) == 1 for value in optimizer.state.values()))

    def test_fresh_output_refusal_precedes_helper_import(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / 'preserve'
            marker.write_text('unchanged')
            argv = ['--execute', '--repo', str(REPO), '--inspection-dir', 'not-opened', '--metadata', 'not-opened',
                    '--official-reading-utils', 'not-opened', '--output-dir', str(output), '--gpu-uuid', 'root-selected']
            with mock.patch.object(bench, 'load_helpers', side_effect=AssertionError('no import')), self.assertRaises(FileExistsError):
                bench.main(argv)
            self.assertEqual(marker.read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main(verbosity=2)
