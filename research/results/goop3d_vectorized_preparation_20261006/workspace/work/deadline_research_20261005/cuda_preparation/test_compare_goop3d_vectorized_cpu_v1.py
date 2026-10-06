"""Synthetic CPU-only comparison harness tests; no real arrays or GNS model."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import unittest
from unittest import mock

import compare_goop3d_vectorized_cpu_v1 as compare


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile, cls.bench, cls.original, cls.candidate = compare.load_helpers(Path(__file__).parent)

    def fixture(self):
        g = self.original
        points = g.np.array(list(itertools.product((.4, .422), repeat=3)), dtype=g.np.float32)
        history = g.np.repeat(points[:, None, :], 6, axis=1)
        position = g.torch.from_numpy(g.np.concatenate((history, history)))
        types = g.torch.full((16,), 7, dtype=g.torch.long)
        batch = (position, types, g.torch.tensor([8, 8]), position[:, -1].clone())
        return batch, g.host_noise(position.shape, types, 0, 0)

    def test_default_has_no_import_or_data_work(self):
        with mock.patch.object(compare, "load_helpers", side_effect=AssertionError("No helper work")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(compare.main([]), 0)
        self.assertFalse(json.loads(output.getvalue())["execution"])

    def test_cpu_proof_matches_all_edges_ledgers_and_rng_without_cuda(self):
        batch, noise = self.fixture()
        with mock.patch.object(self.original.torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            for case in ("base", "expanded25"):
                proof = compare.prove_exact(batch, noise, case, self.profile, self.original, self.candidate)
                self.assertTrue(proof["all_ordered_edges_exact"])
                self.assertTrue(proof["all_ledger_fields_exact"])
                self.assertTrue(proof["global_cpu_rng_unchanged"])

    def test_wrong_candidate_edges_fail_equality_gate(self):
        batch, noise = self.fixture()
        actual = self.candidate.append_optional_edges
        def bad(*args, **kwargs):
            edges, ledger = actual(*args, **kwargs)
            return edges.flip(1), ledger
        with mock.patch.object(self.candidate, "append_optional_edges", side_effect=bad):
            with self.assertRaisesRegex(ValueError, "edges/ledger differ"):
                compare.prove_exact(batch, noise, "expanded25", self.profile, self.original, self.candidate)

    def test_full_stages_match_both_cases(self):
        batch, noise = self.fixture()
        for case in ("base", "expanded25"):
            a = self.profile.graph_stages(batch, noise, case, self.original)
            b = self.profile.graph_stages(batch, noise, case, self.candidate)
            for key in ("graph_ledger", "native_batch_edge_sha256", "final_batch_edge_sha256", "final_directed_edges"):
                self.assertEqual(a[key], b[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
