"""Synthetic numerical/decision checks; never accesses scientific data/models."""
import copy
import json
import unittest
import numpy as np
import goop_global_action_gate_core_v1 as gate


class GateCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = gate.row_ids("train")
        cls.x = np.full((8000, 11), 3.0)
        cls.x[:, 0] = np.tile([-1.0, 1.0], 4000)
        cls.y = .003 + .002 * cls.x[:, 0]
        cls.heads = gate.fit_candidates(cls.x, cls.y, cls.ids, 0, {"synthetic_only": "0" * 64})

    def test_features_have_hand_calculated_units_and_quantiles(self):
        h = np.full((6, 2, 2), .5)
        h[3:, 0, 0] = [.25, .5, .75]
        h[3:, 1, 0] = [.5, .75, 1.0]
        expected = [np.log(3), .25, .25, .25, 0, 0, 0, .025, -.075, .5, .1]
        np.testing.assert_allclose(gate.features(h), expected, atol=1e-15, rtol=1e-14)
        h[:3] = -2.0
        np.testing.assert_allclose(gate.features(h), expected, atol=1e-15, rtol=1e-14)

    def test_features_reject_nonfinite_shape_and_guard_breach(self):
        for h in (np.zeros((5, 1, 2)), np.zeros((6, 0, 2)), np.full((6, 1, 2), np.nan),
                  np.zeros((6, 1, 3)), np.full((6, 1, 2), 10.01)):
            with self.subTest(shape=h.shape), self.assertRaises(ValueError):
                gate.features(h)

    def test_signed_label_uses_coordinate_mse_and_whole_actions(self):
        result = gate.action_errors([[1., 2.]], [[0., 1.]], [[0., 0.]])
        self.assertEqual(result, {"base_mse": 2.5, "random25_mse": .5, "signed_benefit": 2.0})
        reverse = gate.action_errors([[0., 1.]], [[1., 2.]], [[0., 0.]])
        self.assertEqual(reverse["signed_benefit"], -2.0)
        with self.assertRaises(ValueError):
            gate.action_errors([[0., 1.]], [[1., np.nan]], [[0., 0.]])

    def test_ridge_matches_closed_form_and_preserves_constant_columns(self):
        for penalty, head in self.heads.items():
            expected = .003 + .002 * self.x[:, 0] / (1.0 + penalty)
            np.testing.assert_allclose(gate.predict(head, self.x), expected, atol=2e-16, rtol=2e-13)
            self.assertEqual(head["feature_scale"][1:], [1e-12] * 10)
            np.testing.assert_allclose(head["coefficients"][1:], 0, atol=1e-15)
            restored = json.loads(json.dumps(head, allow_nan=False))
            np.testing.assert_array_equal(gate.predict(head, self.x), gate.predict(restored, self.x))

    def test_fit_refuses_incomplete_or_duplicate_scheduled_labels(self):
        bad = self.ids.copy(); bad[-1] = bad[-2]
        for ids, x, y in ((bad, self.x, self.y), (self.ids[:-1], self.x[:-1], self.y[:-1]),
                          (self.ids, self.x, np.full(8000, np.nan))):
            with self.assertRaises(ValueError):
                gate.fit_candidates(x, y, ids, 0, {"synthetic_only": "0" * 64})

    def fixture_selection(self, signs):
        candidates, validation = {}, {}
        for seed in (0, 1, 2):
            candidates[seed] = {}
            for penalty in gate.LAMBDAS:
                head = copy.deepcopy(self.heads[penalty])
                head.update(model_seed=seed, coefficients=[0.] * 11, label_mean=signs[penalty][seed] * 1e-4,
                            label_std=0., label_scale=1e-12)
                candidates[seed][penalty] = head
            validation[seed] = {"row_ids": gate.row_ids("valid"), "features": np.zeros((150, 11)),
                "base_mse": np.full(150, 0. if seed == 0 else 4.),
                "random25_mse": np.full(150, 9. if seed == 0 else 0.)}
        return candidates, validation

    def test_shared_lambda_uses_all_seeds_and_actual_decision_loss(self):
        candidates, validation = self.fixture_selection({.001: [1, 1, 1], .1: [-1, 1, 1], 10.: [-1, -1, -1]})
        result = gate.select_common_lambda(candidates, validation)
        self.assertEqual(result["selected_lambda"], .1)
        self.assertEqual(result["candidate_equal_seed_losses"][.001], 3.)
        self.assertEqual(result["candidate_equal_seed_losses"][.1], 0.)
        self.assertEqual([result["heads"][s]["validation_requested_expansion_fraction"] for s in (0, 1, 2)], [0., 1., 1.])
        del validation[2]
        with self.assertRaises(ValueError):
            gate.select_common_lambda(candidates, validation)

    def test_exact_ties_select_larger_penalty_and_zero_means_stay(self):
        candidates, validation = self.fixture_selection({p: [0, 0, 0] for p in gate.LAMBDAS})
        result = gate.select_common_lambda(candidates, validation)
        self.assertEqual(result["selected_lambda"], 10.)
        self.assertTrue(all(head["validation_requested_expansion_fraction"] == 0 for head in result["heads"].values()))
        validation[0]["row_ids"][-1] = (0, 7)
        with self.assertRaises(ValueError):
            gate.select_common_lambda(candidates, validation)

    def test_rng_is_per_step_independent_of_skipped_actions_and_policy(self):
        expected = gate.pair_rng("test", 2, 11, 200).choice(83, 17, replace=False)
        gate.pair_rng("test", 2, 11, 199).random(500)
        np.testing.assert_array_equal(expected, gate.pair_rng("test", 2, 11, 200).choice(83, 17, replace=False))
        self.assertEqual(gate.rng_material("test", 2, 11, 200), [20261006, 27103, 2, 11, 200, 1701])
        self.assertFalse(gate.independent_gate(0., "test", 2, 11, 200))
        self.assertTrue(gate.independent_gate(1., "test", 2, 11, 200))
        self.assertNotEqual(gate.rng_material("test", 2, 11, 200), gate.rng_material("train_rollout", 2, 11, 200))
        self.assertNotEqual(gate.rng_material("test", 2, 11, 200), gate.rng_material("test", 2, 11, 200, gate=True))
        with self.assertRaises(ValueError):
            gate.independent_gate(.5, "valid", 2, 11, 200)

    def test_capacity_schedule_is_fixed_and_covers_extremes(self):
        indices = gate.capacity_label_indices()
        self.assertEqual(len(set(indices)), 64)
        self.assertEqual((indices[0], indices[-1]), (0, 7999))
        self.assertEqual([gate.row_ids("train")[i] for i in (0, 7999)], [(0, 6), (999, 400)])

    def test_selection_rejects_boolean_seed_aliases_and_nonfinite_diagnostics(self):
        candidates, validation = self.fixture_selection({p: [0, 0, 0] for p in gate.LAMBDAS})
        bad = {False: candidates[0], 1: candidates[1], 2: candidates[2]}
        with self.assertRaises(ValueError):
            gate.select_common_lambda(bad, validation)
        candidates[0][.001]["model_seed"] = False
        with self.assertRaises(ValueError):
            gate.select_common_lambda(candidates, validation)
        candidates[0][.001]["model_seed"] = 0
        validation[0]["random25_mse"][:] = 1e300
        with self.assertRaises((ValueError, FloatingPointError)):
            gate.select_common_lambda(candidates, validation)


if __name__ == "__main__":
    unittest.main()
