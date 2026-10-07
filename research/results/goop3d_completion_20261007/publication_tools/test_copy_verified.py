"""Synthetic publication safety checks; no scientific source, model, or array execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('copy_verified', Path(__file__).with_name('copy_verified.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        self.entries = []
        self.products = {}
        for name in ('audit.json', 'summary.json', 'arithmetic_check.json', 'completion.json', 'retained_original_failure.json'):
            self.products[name] = self.add('observed/' + name, {'synthetic': name}, 'verified_observed_product')
        self.review = self.add('review.json', {
            'status': 'passed_independent_actual_product_scalar_review',
            'actual_original_exit': 0, 'cached_rows_bound_exactly': 2568,
            'all_original_accounting_states': 4728,
            'original36rounding_mismatches_preserved_and_resolved': True,
            'products_sha256': {k: v['sha256'] for k, v in self.products.items() if k != 'completion.json'},
            'completion_sha256': self.products['completion.json']['sha256'],
        })
        self.receipt = self.add('transfer.json', {
            'actual_status': {'status': 'exited', 'returncode': 0}, 'remaining_group_members': [],
            'files': {'observed_finalized_v2/' + Path(e['destination']).name: {'bytes': e['bytes'], 'sha256': e['sha256']} for e in self.entries if e['role'] == 'verified_observed_product'},
        })
        self.admission = self.add('admission.json', {
            'status': 'admitted_observed_history_only', 'independent_review_sha256': self.review['sha256'],
            'autonomous_scientific_admission': False,
            'completion_sha256': self.products['completion.json']['sha256'],
            'summary_sha256': self.products['summary.json']['sha256'],
        })
        self.plan = {'schema': 'goop3d_completion_publication_plan_v1',
                     'package': 'research/results/goop3d_completion_20261007', 'files': self.entries,
                     'pending': ['autonomous numerical admission', 'manuscript compilation'],
                     'numerical_product_gate': {'status': 'reviewed_for_publication', 'review': self.review,
                         'transfer_manifest': self.receipt, 'admission': self.admission, 'products': self.products}}
        self.plan_path = self.workspace / 'plan.json'
        self.destination = self.root / 'fork' / self.plan['package']

    def add(self, name, doc, role='provenance'):
        path = self.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(c.encode(doc))
        entry = {'source': name, 'destination': name, 'bytes': path.stat().st_size, 'sha256': c.sha(path), 'role': role}
        self.entries.append(entry)
        return {'destination': name, 'sha256': entry['sha256']}

    def run_copy(self, do_copy=False):
        self.plan_path.write_bytes(c.encode(self.plan))
        return c.execute(self.plan_path, c.sha(self.plan_path), self.workspace, self.destination, do_copy)

    def test_verified_copy_rechecks_all_bytes_and_refuses_existing(self):
        self.assertTrue(self.run_copy()['copy_ready'])
        result = self.run_copy(True)
        self.assertEqual(result['status'], 'copied_verified_package')
        for e in self.entries:
            self.assertEqual(c.sha(self.destination / e['destination']), e['sha256'])
        manifest = json.loads((self.destination / 'publication_manifest.json').read_text())
        self.assertEqual(len(manifest['files']), len(self.entries))
        self.assertEqual((self.destination / 'curation_plan.json').read_bytes(), self.plan_path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'fresh package'):
            self.run_copy(True)

    def test_changed_source_rejected(self):
        (self.workspace / 'observed/audit.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'source bytes changed'):
            self.run_copy()

    def test_wrong_plan_pin_rejected(self):
        self.plan_path.write_bytes(c.encode(self.plan))
        with self.assertRaisesRegex(ValueError, 'exact reviewed plan hash'):
            c.execute(self.plan_path, '0' * 64, self.workspace, None, False)

    def test_product_pin_substitution_rejected(self):
        self.plan['numerical_product_gate']['products']['audit.json']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'review/product pin differs'):
            self.run_copy()

    def test_launch_receipt_cannot_establish_completion(self):
        path = self.workspace / 'transfer.json'
        value = json.loads(path.read_text())
        value['actual_status']['status'] = 'running'
        path.write_bytes(c.encode(value))
        entry = next(e for e in self.entries if e['source'] == 'transfer.json')
        entry.update(bytes=path.stat().st_size, sha256=c.sha(path))
        self.receipt['sha256'] = entry['sha256']
        with self.assertRaisesRegex(ValueError, 'actual process completion'):
            self.run_copy()

    def test_credential_scan_reports_pattern_without_value(self):
        secret = 'gh' + 'p_' + 'Z' * 40
        self.add('unsafe.json', {'value': secret})
        with self.assertRaises(ValueError) as error:
            self.run_copy()
        self.assertIn('github_key', str(error.exception))
        self.assertNotIn(secret, str(error.exception))

    def test_destination_traversal_and_duplicate_rejected(self):
        self.entries[0]['destination'] = '../escape.json'
        with self.assertRaisesRegex(ValueError, 'unsafe or reserved'):
            self.run_copy()
        self.entries[0]['destination'] = self.entries[1]['destination']
        with self.assertRaisesRegex(ValueError, 'duplicate destinations'):
            self.run_copy()

    def test_parent_symlink_rejected(self):
        (self.workspace / 'alias').symlink_to(self.workspace / 'observed', target_is_directory=True)
        self.entries[0]['source'] = 'alias/audit.json'
        with self.assertRaisesRegex(ValueError, 'symlink source parent'):
            self.run_copy()

    def test_unreviewed_additional_product_rejected(self):
        self.add('extra_product.json', {'synthetic': True}, 'verified_observed_product')
        with self.assertRaisesRegex(ValueError, 'unreviewed numerical product'):
            self.run_copy()


if __name__ == '__main__':
    unittest.main(verbosity=2)
