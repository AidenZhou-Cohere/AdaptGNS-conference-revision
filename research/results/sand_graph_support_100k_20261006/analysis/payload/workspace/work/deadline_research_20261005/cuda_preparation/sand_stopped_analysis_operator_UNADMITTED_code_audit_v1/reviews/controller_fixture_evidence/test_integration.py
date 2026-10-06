#!/usr/bin/env python3
"""Independent synthetic integration checks; no host, science or clock access."""
import base64
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PREP = Path(__file__).resolve().parent.parent
PACKAGE = PREP / 'sand_stopped_analysis_operator_UNADMITTED_code_audit_v1'
sys.path.insert(0, str(PACKAGE))
import root_operator as O
import review_products as V

def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()

def pin(raw):
    return hashlib.sha256(raw).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))

def blob(path, raw):
    return dict(path=path, sha256=pin(raw), bytes=len(raw), base64=base64.b64encode(raw).decode())

class ProductIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name).resolve()
        self.now = datetime.datetime(2026, 10, 6, 23, tzinfo=datetime.timezone.utc)
        self.calls = []
        self.captured_payloads = []
        self.predecessors = {}
        write(self.state / 'analysis_phase.json', {'synthetic_phase': True})
        write(self.state / 'summary_release.json', {'synthetic_summary_release': True})
        for role in O.ALIASES:
            write(self.state / (role + '.evaluation_closure.json'), {'native_absent': {'90': True}})
            write(self.state / (role + '.collection_release.json'), {'synthetic_collection_release': role})
        for op in O.OPS:
            self.predecessors[op] = ({'product_sha256': pin(encoded({'operation': op})),
                                      'semantic_review': {'eligible_for_aggregation': True}}, {'operation': op})
        def semantic(name):
            def validate(*args):
                self.calls.append((name, args))
                return {'eligible_for_aggregation': True, 'synthetic_validator': name}
            return validate
        for name in ('validate_collection', 'validate_saved_audit', 'validate_summary', 'validate_paired_audit'):
            p = patch.object(V, name, semantic(name)); p.start(); self.addCleanup(p.stop)
        def probe(state, role, payload, tag):
            self.captured_payloads.append((role, payload, tag))
            return copy.deepcopy(self.capture)
        replacements = {
            'probe': probe,
            'runtime_ready': lambda *args: None,
            'original_exit': lambda *args: None,
            'budget': lambda *args: ({}, self.now + datetime.timedelta(seconds=100), 5000, 100),
            'now': lambda: self.now,
            'accepted': lambda state, op: self.predecessors[op],
        }
        for name, value in replacements.items():
            p = patch.object(O, name, value); p.start(); self.addCleanup(p.stop)

    def fixture(self, op='sand_collect_A', gpu=True):
        self.op = op; self.role = O.role_of(op); spec = O.spec(op)
        self.release = O.read(PACKAGE / 'candidates' / (op + '.cpu_release.candidate.json'))
        release = self.release
        release.update(invocation_origin_utc=(self.now - datetime.timedelta(seconds=20)).isoformat(),
                       publication_deadline_utc=(self.now + datetime.timedelta(seconds=100)).isoformat(),
                       hard_deadline_monotonic_ns=123456789, clock_sample={'error_bound_seconds': 5},
                       protected_tree_entries={}, inputs_sha256={p: h or 'e' * 64 for p, h in spec['inputs_sha256'].items()})
        release['analysis_phase']['sha256'] = O.sha(self.state / 'analysis_phase.json')
        release['command'] = [release['inputs_sha256'][x['sha256_of_input']] if isinstance(x, dict) else x for x in spec['argv']]
        self.docs = {}
        if op.startswith('sand_collect_'):
            q = O.bindings()['roles'][self.role]['queue_root']
            names = ['release_snapshot.json', 'queue_status.json', 'coverage_ledger.json', 'process_outcomes.json',
                     'jobs/synthetic/protocol.json'] + (['gpu_observations.json'] if gpu else [])
            rows = []
            for name in names:
                path = q + '/' + name; raw = encoded({'synthetic_document': name})
                self.docs[path] = json.loads(raw)
                local = self.state / (op + '.snapshot.documents') / path.removeprefix(O.R + '/')
                local.parent.mkdir(parents=True, exist_ok=True); local.write_bytes(raw)
                rows.append(blob(path, raw)); release['inputs_sha256'][path] = pin(raw)
            inventory = {'files': {row['path'].removeprefix(q + '/'): {'sha256': row['sha256'], 'bytes': row['bytes']} for row in rows},
                         'entries': sorted([[name, 'file'] for name in names])}
            release['protected_tree_entries'] = {q: inventory['entries']}
            write(self.state / (op + '.snapshot.json'), {'inputs_sha256': release['inputs_sha256'],
                  'protected_tree_entries': release['protected_tree_entries'], 'documents': rows, 'queue_inventories': {q: inventory}})
        write(self.state / (op + '.cpu_release.json'), release)
        write(self.state / (op + '.external.json'), {'observed_utc': (self.now - datetime.timedelta(seconds=1)).isoformat()})
        write(self.state / (op + '.root_original_external_exit.json'), {'synthetic_root_exit': op})
        self.owner_dir = release['owner_output_dir']; self.output = spec['outputs'][0]
        owner = {'pid': 101, 'ppid': 100}; outer = {'pid': 100}; identity = {'pid': 102, 'ppid': 101}
        guard = {'absolute_deadline_ns': release['hard_deadline_monotonic_ns']}
        timing = {'synthetic_native_timing': True}
        self.started = dict(owner_identity=owner, outer_timeout_identity=outer, release_sha256=O.sha(self.state / (op + '.cpu_release.json')),
                            native_hard_guard=guard, scientific_admission=False, original_clock_sample=release['clock_sample'], native_outer_timing=timing)
        self.registered = dict(identity=identity, pid=102, owner_identity=owner)
        child = dict(identity=identity, pid=102, reaped=True, exit_code=0, signals=[], cleanup_errors=[], command=release['command'])
        expected = O.merge(release['inputs_sha256'], O.bindings()['environment_expected_sha256'],
                           {O.A + '/controls/' + op + '.cpu_release.json': O.sha(self.state / (op + '.cpu_release.json'))},
                           {O.A + '/controls/' + role + '.evaluation_closure.json': O.sha(self.state / (role + '.evaluation_closure.json')) for role in O.ALIASES},
                           {release[key]['path']: release[key]['sha256'] for key in ('operation_spec', 'analysis_phase', 'owner_source', 'bootstrap_source')})
        self.terminal = dict(schema='adaptgns_stopped_analysis_cpu_terminal_v1', status='complete', failure=None,
                             release_sha256=self.started['release_sha256'], child_native_absent=True,
                             root_original_tool_exit_and_timeout_owner_child_native_closure_required=True,
                             child=child, input_sha256=expected, output_sha256={self.output: pin(encoded({'synthetic_product': op}))},
                             owner_identity=owner, outer_timeout_identity=outer, native_hard_guard=guard, scientific_admission=False,
                             native_outer_timing=timing, publication_utc=(self.now - datetime.timedelta(seconds=2)).isoformat())
        self.refresh_capture()

    def refresh_capture(self):
        files = {self.owner_dir + '/owner_started.json': encoded(self.started),
                 self.owner_dir + '/child_registered.json': encoded(self.registered),
                 self.owner_dir + '/child.stdout': b'synthetic stdout\n', self.owner_dir + '/child.stderr': b'',
                 self.output: encoded({'synthetic_product': self.op})}
        self.terminal['evidence_sha256'] = {p: pin(raw) for p, raw in files.items() if p != self.output}
        files[self.owner_dir + '/owner_terminal.json'] = encoded(self.terminal)
        self.capture = {'files': [blob(p, raw) for p, raw in files.items()],
                        'native_absent': {'90': True, '100': True, '101': True, '102': True},
                        'final_input_and_output_hashes_verified': True}

    def run_product(self):
        O.close_product(self.state, self.op)
        return O.read(self.state / (self.op + '.review.json'))

    def check_product(self, op):
        self.fixture(op); receipt = self.run_product()
        self.assertEqual(receipt['status'], 'accepted_stopped_product')
        self.assertEqual(receipt['native_absent'], self.capture['native_absent'])
        self.assertEqual(self.captured_payloads[0][1]['historical_pids'], [90])
        self.assertEqual(len(self.calls), 1)
        name, args = self.calls[0]
        self.assertEqual(args[0], {'synthetic_product': op})
        self.assertEqual(args[1], self.terminal['output_sha256'][self.output])
        expected = args[2]
        if op.startswith('sand_collect_'):
            q = O.bindings()['roles'][self.role]['queue_root']
            self.assertEqual(name, 'validate_collection')
            for field, file in [('queue_release', 'release_snapshot.json'), ('ledger', 'coverage_ledger.json'),
                                ('queue_status', 'queue_status.json'), ('process_outcomes', 'process_outcomes.json'),
                                ('gpu_observations', 'gpu_observations.json')]:
                self.assertEqual(expected[field], self.docs[q + '/' + file])
            self.assertEqual(expected['protocols'], {'jobs/synthetic/protocol.json': self.docs[q + '/jobs/synthetic/protocol.json']})
        elif op.startswith('sand_saved_'):
            self.assertEqual(args[3], self.predecessors['sand_collect_' + self.role][1])
        else:
            self.assertEqual(expected['collection_sha256'], {r: self.predecessors['sand_collect_' + r][0]['product_sha256'] for r in O.ALIASES})
            if op == 'sand_paired':
                self.assertEqual(expected['summary_sha256'], self.predecessors['sand_summarize'][0]['product_sha256'])
                self.assertEqual(expected['audit_sha256'], {r: self.predecessors['sand_saved_' + r][0]['product_sha256'] for r in O.ALIASES})

    def test_collection_raw_document_mutation_rejected(self):
        self.fixture(); path = next(iter(self.docs))
        local = self.state / (self.op + '.snapshot.documents') / path.removeprefix(O.R + '/')
        local.write_bytes(encoded({'changed': True}))
        with self.assertRaisesRegex(ValueError, 'transported scalar bytes changed'): self.run_product()
        self.assertEqual(self.calls, [])

    def test_collection_raw_size_mismatch_rejected(self):
        self.fixture(); path = self.state / (self.op + '.snapshot.json'); snap = O.read(path)
        snap['documents'][0]['bytes'] += 1; write(path, snap)
        with self.assertRaisesRegex(ValueError, 'transported scalar bytes changed'): self.run_product()

    def test_collection_duplicate_document_rejected(self):
        self.fixture(); path = self.state / (self.op + '.snapshot.json'); snap = O.read(path)
        snap['documents'].append(copy.deepcopy(snap['documents'][0])); write(path, snap)
        with self.assertRaisesRegex(ValueError, 'Duplicate original snapshot'): self.run_product()

    def test_no_gpu_observations_passes_explicit_none_to_validator(self):
        self.fixture(gpu=False); self.run_product(); self.assertIsNone(self.calls[0][1][2]['gpu_observations'])

    def test_missing_owner_evidence_rejected(self):
        self.fixture(); self.capture['files'] = [row for row in self.capture['files'] if not row['path'].endswith('/child.stderr')]
        with self.assertRaisesRegex(ValueError, 'Complete original owner evidence'): self.run_product()

    def test_extra_owner_evidence_rejected(self):
        self.fixture(); self.capture['files'].append(blob(self.owner_dir + '/extra.json', b'{}'))
        with self.assertRaisesRegex(ValueError, 'Unexpected/duplicate'): self.run_product()

    def test_wrong_owner_input_map_rejected(self):
        self.fixture(); self.terminal['input_sha256'].pop(next(iter(self.terminal['input_sha256']))); self.refresh_capture()
        with self.assertRaisesRegex(ValueError, 'full input/product byte pins'): self.run_product()

    def test_wrong_registered_identity_rejected(self):
        self.fixture(); self.registered['identity'] = {'pid': 909}; self.refresh_capture()
        with self.assertRaisesRegex(ValueError, 'identities differ'): self.run_product()

    def test_wrong_guard_rejected(self):
        self.fixture(); self.terminal['native_hard_guard'] = {'absolute_deadline_ns': 1}; self.refresh_capture()
        with self.assertRaisesRegex(ValueError, 'absolute native guard'): self.run_product()

    def test_capture_final_rehash_must_pass(self):
        self.fixture(); self.capture['final_input_and_output_hashes_verified'] = False
        with self.assertRaisesRegex(ValueError, 'evidence hash set'): self.run_product()

    def test_final_budget_refusal_leaves_no_review(self):
        self.fixture()
        with patch.object(O, 'budget', side_effect=ValueError('synthetic exhausted original phase')):
            with self.assertRaisesRegex(ValueError, 'exhausted original phase'): self.run_product()
        self.assertFalse((self.state / (self.op + '.review.json')).exists())

    def test_semantic_product_mutation_rejected_at_final_rehash(self):
        self.fixture()
        def mutate(*args):
            path = self.state / (self.op + '.collected') / self.output.removeprefix(O.R + '/')
            path.write_bytes(encoded({'changed_after_semantic': True})); return {'eligible_for_aggregation': True}
        with patch.object(V, 'validate_collection', mutate):
            with self.assertRaisesRegex(ValueError, 'Product changed during review'): self.run_product()

for operation in O.OPS:
    setattr(ProductIntegrationTests, 'test_close_' + operation, lambda self, op=operation: self.check_product(op))

class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.state = Path(self.temp.name)
        write(self.state / 'analysis_phase.json', {'synthetic_phase': True})
        self.receipts = {op: {'product_sha256': pin(encoded(op)), 'semantic_review': {'eligible_for_aggregation': True}} for op in O.OPS}
        self.calls = []
        def accept(state, op): self.calls.append(op); return self.receipts[op], {'synthetic': op}
        p = patch.object(O, 'accepted', accept); p.start(); self.addCleanup(p.stop)
        self.transfer = {'analysis_phase_sha256': O.sha(self.state / 'analysis_phase.json'),
                         'transferred_sha256': {O.spec(op)['outputs'][0]: self.receipts[op]['product_sha256'] for op in ('sand_collect_B', 'sand_saved_B')}}
        self.save()
    def save(self): write(self.state / 'B.scalars_transferred_to_A.json', self.transfer)
    def test_six_exact_dependency_sets(self):
        expected = {'sand_collect_A': (), 'sand_collect_B': (), 'sand_saved_A': ('sand_collect_A',),
                    'sand_saved_B': ('sand_collect_B',), 'sand_summarize': ('sand_collect_A', 'sand_collect_B'),
                    'sand_paired': ('sand_collect_A', 'sand_collect_B', 'sand_saved_A', 'sand_saved_B', 'sand_summarize')}
        self.assertEqual(O.DEPS, expected)
        for op, parents in expected.items():
            self.calls.clear(); O.require_predecessors(self.state, op)
            self.assertEqual(self.calls[:len(parents)], list(parents))
    def test_ineligible_collection_refuses_downstream(self):
        self.receipts['sand_collect_A']['semantic_review']['eligible_for_aggregation'] = False
        with self.assertRaisesRegex(ValueError, 'unresolved original input integrity'): O.require_predecessors(self.state, 'sand_saved_A')
    def test_b_saved_transfer_required_for_summary(self):
        self.transfer['transferred_sha256'].pop(O.spec('sand_saved_B')['outputs'][0]); self.save()
        with self.assertRaises(KeyError): O.require_predecessors(self.state, 'sand_summarize')
    def test_changed_b_transfer_rejected(self):
        self.transfer['transferred_sha256'][O.spec('sand_collect_B')['outputs'][0]] = 'e' * 64; self.save()
        with self.assertRaisesRegex(ValueError, 'exact accepted bytes'): O.require_predecessors(self.state, 'sand_paired')
    def test_wrong_phase_transfer_rejected(self):
        self.transfer['analysis_phase_sha256'] = 'e' * 64; self.save()
        with self.assertRaisesRegex(ValueError, 'Same-phase'): O.require_predecessors(self.state, 'sand_summarize')
    def test_native_history_unions_same_role_accepted_history(self):
        for r in O.ALIASES: write(self.state / (r + '.evaluation_closure.json'), {'native_absent': {'1': True, '2': True}})
        for op, pid in [('sand_collect_A', 3), ('sand_saved_A', 4), ('sand_collect_B', 5)]:
            write(self.state / (op + '.review.json'), {'analysis_phase_sha256': O.sha(self.state / 'analysis_phase.json'), 'native_absent': {str(pid): True}})
        self.assertEqual(O.historical_pids(self.state, 'A'), [1, 2, 3, 4])
    def test_native_history_refuses_nonabsence(self):
        write(self.state / 'A.evaluation_closure.json', {'native_absent': {'1': True}})
        write(self.state / 'sand_collect_A.review.json', {'analysis_phase_sha256': O.sha(self.state / 'analysis_phase.json'), 'native_absent': {'2': False}})
        with self.assertRaisesRegex(ValueError, 'Prior operation closure changed'): O.historical_pids(self.state, 'A')

if __name__ == '__main__': unittest.main(verbosity=2)
