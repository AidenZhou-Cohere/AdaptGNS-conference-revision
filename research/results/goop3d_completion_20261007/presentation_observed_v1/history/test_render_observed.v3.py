"""Synthetic schema, arithmetic and presentation checks; no live products."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('observed_renderer', HERE / 'render_observed.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture():
    lineage = {'required_all_cells': 4728, 'required_observed_cells': 2568, 'collection_sha256': r.COLLECTION_SHA,
               'finalizer_revision': 2, 'comparison_tolerance_unchanged': True, 'saved_row_array_audit_calls': 0,
               'old_checker_source_sha256': r.OLD_CHECKER_SHA, 'cohort_sha256': r.COHORT_SHA, 'phase_sha256': None}
    original, models, families = [], [], {}
    for stage in r.FAMILIES:
        n = 128 if stage == 'clean_validation' else 150
        family = {'full_family_complete': True, 'required_cells': 6 * n,
                  'coverage_by_model': {f'{a}_seed{s}': {'completed_required_outcome': n} for a in r.ARMS for s in range(3)},
                  'absolute': {}, 'mix_minus_base_training': {}}
        keys = sorted(r.expected_keys(stage))
        for arm in r.ARMS:
            absolute = {}
            for key_index, key in enumerate(keys):
                values = []
                for seed in range(3):
                    if key.startswith('graph/'):
                        value = 52078.3  # Identical non-integer large inputs; exact SD must be zero.
                    elif key == 'correlations/previous_risk_vs_' + r.RISK + '_benefit':
                        value = None
                    else:
                        value = (key_index + 1) * 1e-9 + seed * 2e-10 + ((seed - 1) * 1e-11 if arm == 'mix' else 0)
                    values.append(value)
                absolute[key] = r.stats(values)
            family['absolute'][arm] = absolute
            for seed in range(3):
                cells = [{'source_index': i // 5, 'target_frame': i % 5 + 6, 'state': 'completed_required_outcome'} for i in range(n)]
                slot = {'arm': arm, 'seed': seed, 'stage': stage, 'cells': cells}
                original.append(slot)
                models.append({**slot, 'coverage': {'completed_required_outcome': n},
                               'aggregates': {key: {'equal_trajectory_mean': value['seed_values'][str(seed)]} for key, value in absolute.items()}})
        for key in keys:
            family['mix_minus_base_training'][key] = r.stats([r.delta(x, y) for x, y in zip(r.ordered(family['absolute']['mix'][key]), r.ordered(family['absolute']['base'][key]))])
        if stage != 'clean_validation':
            family['accuracy_policy_contrasts'] = {}
            for metric in r.METRICS:
                within = {}
                for arm in r.ARMS:
                    within[arm] = {}
                    for policy in r.POLICIES:
                        for ref in r.REFERENCES:
                            if policy != ref:
                                a = r.ordered(family['absolute'][arm]['accuracy/' + policy + '/' + metric])
                                b = r.ordered(family['absolute'][arm]['accuracy/' + ref + '/' + metric])
                                within[arm][policy + '_minus_' + ref] = r.stats([r.delta(x, y) for x, y in zip(a, b)])
                a, b = [r.ordered(within[arm][r.RISK + '_minus_random25']) for arm in ('mix', 'base')]
                family['accuracy_policy_contrasts'][metric] = {'within_arm': within, 'risk_minus_random_mix_minus_base': r.stats([r.delta(x, y) for x, y in zip(a, b)])}
        families[stage] = family
    states = ['completed_required_outcome'] * 331 + ['not_completed_before_invocation_end'] * 1817 + ['timed_out_current'] * 12
    k = 0
    for arm in r.ARMS:
        for seed in range(3):
            for stage in r.STAGES[:2]:
                original.append({'arm': arm, 'seed': seed, 'stage': stage, 'cells': [{'state': v} for v in states[k:k + 180]]})
                k += 180
    audit = {**lineage, 'schema': 'goop3d_observed_history_saved_audit_v1', 'status': 'passed_scoped_observed_history_checks',
             'dataset': 'Goop-3D', 'endpoint_updates': 25000, 'all_original_accounting': original, 'models': models}
    summary = {**lineage, 'schema': 'goop3d_observed_history_summary_v1', 'status': 'passed_scoped_scalar_aggregation',
               'all_original_accounting': original, 'families': families}
    check = {**lineage, 'schema': 'goop3d_observed_history_arithmetic_check_v1', 'status': 'passed_independent_observed_arithmetic',
             'checker_revision': 2, 'sample_sd_method': 'exact_rational_pairwise_variance_decimal_sqrt'}
    completion = {**lineage, 'schema': 'goop3d_observed_cache_finalization_v2',
                  'status': 'all2568_cached_rows_merged_and_exact_variance_check_passed', 'cache_rows_consumed': 2568,
                  'original_arrays_decoded': 0, 'original_cache_and_failed_attempt_unchanged': True}
    mismatch = {'path_components': ['families', 'same_state_test', 'absolute', 'mix', 'graph/base/candidate_pairs', 'sample_sd'],
                'exact_rational_oracle': {'all_three_input_floats_identical': True, 'exact_unbiased_variance_ratio': '0'}, 'summarizer_value': 0}
    diagnosis = {'status': 'diagnosis_only_not_scientific_admission', 'cached_rows_verified': 2568,
                 'mismatch_count': 36, 'mismatches': [copy.deepcopy(mismatch) for _ in range(36)],
                 'rows_or_arrays_reaudited': 0, 'tolerances_changed': False}
    products = {'audit': audit, 'summary': summary, 'arithmetic_check': check, 'completion': completion,
                'retained_original_failure': {'error_type': 'ValueError', 'error': 'independent arithmetic differs'}, 'diagnosis': diagnosis}
    pins = {k: str(i) * 64 for i, k in enumerate(r.PRODUCTS)}
    summary['audit_sha256'] = pins['audit']
    check.update(audit_sha256=pins['audit'], summary_sha256=pins['summary'])
    completion['products_sha256'] = {k + '.json': pins[k] for k in ('audit', 'summary', 'arithmetic_check', 'retained_original_failure')}
    return products, pins


class Rendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products, cls.pins = fixture()

    def test_complete_family_and_all_seeds_render_without_mutation(self):
        before = r.encode(self.products['summary'])
        index = r.validate(self.products, self.pins)
        manifest = {'products': {k: {'sha256': v} for k, v in self.pins.items()}}
        fragments, claims = r.render(self.products['summary'], self.products['audit'], index, manifest)
        self.assertEqual(r.encode(self.products['summary']), before)
        text = fragments['appendix_goop3d_observed.tex']
        self.assertIn('GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT', text)
        self.assertNotIn('laggedrisk25', text)
        self.assertEqual(text.count(r'\begin{table*}'), 13)
        self.assertEqual(text.count(r'\begin{table*}'), text.count(r'\end{table*}'))
        self.assertGreater(len(index), 1000)
        self.assertTrue(all(c['summary_sha256'] == self.pins['summary'] for c in claims))
        self.assertEqual(len([k for k in index if '/clean_validation/' in k]), 12)
        for stage in r.FAMILIES[:2]:
            for arm in r.ARMS:
                for policy in r.POLICIES:
                    path = ['families', stage, 'absolute', arm, 'accuracy/' + policy + '/position_coordinate_mse']
                    self.assertTrue(any(c['json_path'] == path for c in claims))

    def test_large_identical_means_and_tiny_signs(self):
        record = r.stats([52078.3] * 3)
        self.assertEqual(record['sample_sd'], 0)
        self.assertEqual(r.ordered(record), [52078.3] * 3)
        self.assertTrue(r.tex_number(-1e-120).startswith('-1'))
        self.assertIn('10^{-120}', r.tex_number(-1e-120))
        bad = copy.deepcopy(record)
        bad['sample_sd'] = 8.9e-12
        with self.assertRaisesRegex(ValueError, 'zero statistic'):
            r.ordered(bad)

    def test_missing_seed_and_survivor_mean_rejected(self):
        record = r.stats([1, None, 3])
        self.assertEqual(r.pm(record), '---')
        self.assertEqual(r.seed_cells(record)[1], '---')
        record['mean'] = 2
        with self.assertRaisesRegex(ValueError, 'incomplete seed family'):
            r.ordered(record)

    def test_missing_policy_rejected(self):
        products = copy.deepcopy(self.products)
        del products['summary']['families']['same_state_valid']['absolute']['base']['accuracy/base/position_coordinate_mse']
        with self.assertRaisesRegex(ValueError, 'all diagnostic'):
            r.validate(products, self.pins)

    def test_original_denominator_or_observed_coverage_rejected(self):
        products = copy.deepcopy(self.products)
        products['summary']['required_all_cells'] = 4727
        with self.assertRaisesRegex(ValueError, 'fixed denominator'):
            r.validate(products, self.pins)
        products = copy.deepcopy(self.products)
        products['summary']['families']['same_state_test']['full_family_complete'] = False
        with self.assertRaisesRegex(ValueError, 'full observed family'):
            r.validate(products, self.pins)

    def test_bad_linkage_and_mixed_protocol_rejected(self):
        products = copy.deepcopy(self.products)
        products['arithmetic_check']['summary_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'product linkage'):
            r.validate(products, self.pins)
        products = copy.deepcopy(self.products)
        products['summary']['families']['full_rollout_test'] = {}
        with self.assertRaisesRegex(ValueError, 'cannot be mixed'):
            r.validate(products, self.pins)

    def test_diagnosis_cannot_hide_accuracy_error(self):
        products = copy.deepcopy(self.products)
        products['diagnosis']['mismatches'][0]['path_components'][4] = 'accuracy/base/position_coordinate_mse'
        with self.assertRaisesRegex(ValueError, 'only false nonzero graph'):
            r.validate(products, self.pins)

    def test_input_hash_path_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'input.json'
            path.write_bytes(b'{}\n')
            self.assertEqual(r.read_bound(path, r.digest(path.read_bytes())), b'{}\n')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                r.read_bound(path, 'f' * 64)
            link = path.parent / 'link.json'
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'canonical ordinary'):
                r.read_bound(link, r.digest(path.read_bytes()))
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            r.strict(b'{"a":1,"a":2}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
