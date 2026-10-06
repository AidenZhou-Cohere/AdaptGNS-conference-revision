#!/usr/bin/env python3
"""Local synthetic recovery-audit guards; no real model/data/runtime access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('audit', HERE / 'audit_sand_runtime_migration_endpoints_v1.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def emit(path, value):
    path.write_text(json.dumps(value))
    return A.sha(path)


class ExternalEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        out, err = root / 'stdout', root / 'stderr'
        out.write_text('{}\n'); err.write_text('')
        self.release = {'python_environment': {'lexical_path': '/bin/python'}, 'adapter_path': '/a.py',
                        '_release_path': '/r.json', '_release_sha256': 'a' * 64,
                        'latest_start_utc': '2026-10-06T15:05:00+00:00'}
        self.job = {'id': 'base_seed1'}
        command = ['/bin/python', '-B', '/a.py', '--execute', '--mode', 'train', '--release', '/r.json',
                   '--release-sha256', 'a' * 64, '--job', 'base_seed1']
        self.external = {'pid': 10, 'identity': {'pid': 10, 'argv': command}, 'command': command,
            'exit_code': 0, 'signals': [], 'owner_release_sha256': 'a' * 64,
            'started_utc': '2026-10-06T15:00:00+00:00', 'reaped_utc': '2026-10-06T18:00:00+00:00',
            'clock_error_bound_seconds': 2, 'training_stop_utc': '2026-10-06T22:44:00+00:00',
            'stdout_file': str(out), 'stdout_sha256': A.sha(out), 'stderr_file': str(err), 'stderr_sha256': A.sha(err),
            'scoped_gpu_closure': {'verified': True, 'owned_children_reaped': True, 'rows': [],
                                   'checked_utc': '2026-10-06T18:00:01+00:00'}}

    def check(self):
        # Clock alone is injected so these deterministic tests remain runnable later.
        with patch.object(A, 'deadline'):
            A.external_contract(self.release, self.job, self.external)

    def test_exact_owner_evidence(self): self.check()
    def test_wrong_command(self):
        self.external['command'] = ['python', 'something_else.py']
        with self.assertRaisesRegex(ValueError, 'command'): self.check()
    def test_foreign_gpu_work(self):
        self.external['scoped_gpu_closure']['rows'] = [{'pid': 20}]
        with self.assertRaisesRegex(ValueError, 'scope'): self.check()
    def test_unreaped(self):
        self.external['scoped_gpu_closure']['owned_children_reaped'] = False
        with self.assertRaisesRegex(ValueError, 'scope'): self.check()
    def test_closure_before_reap(self):
        self.external['scoped_gpu_closure']['checked_utc'] = '2026-10-06T17:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'chronology'): self.check()
    def test_mutated_stdout(self):
        Path(self.external['stdout_file']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'stdout'): self.check()
    def test_signalled_child(self):
        self.external['signals'] = ['SIGTERM']
        with self.assertRaisesRegex(ValueError, 'Clean'): self.check()
    def test_late_launch(self):
        self.external['started_utc'] = '2026-10-06T15:05:00+00:00'
        with self.assertRaisesRegex(ValueError, 'latest'): self.check()
    def test_wrong_owner_release(self):
        self.external['owner_release_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'Owner release'): self.check()


class OriginalInventory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        inventory = {}
        for step in range(0, 50001, 10000):
            p = self.root / f'checkpoint-{step:09d}.pt'
            p.write_bytes(('synthetic:' + str(step)).encode())
            inventory[p.name] = A.sha(p)
        self.release = {'files_sha256': {str(self.root / n): d for n, d in inventory.items()}}
        self.job = {'origin_directory': str(self.root), 'origin_checkpoint_inventory': inventory,
                    'origin_checkpoint_sha256': inventory['checkpoint-000050000.pt']}

    def test_complete_inventory(self): A.original_inventory(self.release, self.job)
    def test_missing_step_zero_pin(self):
        self.job['origin_checkpoint_inventory'].pop('checkpoint-000000000.pt')
        with self.assertRaisesRegex(ValueError, 'Every original'): A.original_inventory(self.release, self.job)
    def test_changed_origin_bytes(self):
        (self.root / 'checkpoint-000020000.pt').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Original checkpoint changed'): A.original_inventory(self.release, self.job)
    def test_unlisted_committed_checkpoint(self):
        (self.root / 'checkpoint-000060000.pt').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'inventory changed'): A.original_inventory(self.release, self.job)
    def test_prior_partial_preserved(self):
        (self.root / 'checkpoint-000060000.pt.tmp').write_bytes(b'prior interrupted artifact')
        A.original_inventory(self.release, self.job)


class PrefixAndSuffix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # New prefix/suffix predicates are tested at their actual 50k/100k sizes.
        cls.parent = {'training': [{'completed_steps': step, 'elapsed_seconds': float(step)} for step in [1, *range(100, 50001, 100)]],
                      'graph_updates': [{'completed_steps': step} for step in range(1, 50001)], 'elapsed_seconds': 50000.5}
        cls.full = {'training': cls.parent['training'] + [{'completed_steps': step, 'elapsed_seconds': float(step)} for step in range(50100, 100001, 100)],
                    'graph_updates': cls.parent['graph_updates'] + [{'completed_steps': step} for step in range(50001, 100001)],
                    'elapsed_seconds': 100000.5}

    def setUp(self):
        self.history = dict(self.full)
        self.stdout = self.full['training'][501:]
        self.status = {'last_training': self.stdout[-1]}

    def check(self): A.check_recovery_history(self.parent, self.history, self.stdout, self.status)
    def test_exact_prefix_new_stdout_only(self): self.check()
    def test_changed_old_graph(self):
        self.history['graph_updates'] = [dict(self.full['graph_updates'][0], extra='changed')] + self.full['graph_updates'][1:]
        with self.assertRaisesRegex(ValueError, 'prefix'): self.check()
    def test_changed_old_scalar(self):
        self.history['training'] = [dict(self.full['training'][0], elapsed_seconds=.5)] + self.full['training'][1:]
        with self.assertRaisesRegex(ValueError, 'prefix'): self.check()
    def test_old_stdout_replayed(self):
        self.stdout = self.full['training']
        with self.assertRaisesRegex(ValueError, 'stdout'): self.check()
    def test_missing_new_row(self):
        self.history['training'] = self.full['training'][:-1]
        self.stdout = self.history['training'][501:]
        self.status = {'last_training': self.stdout[-1]}
        with self.assertRaisesRegex(ValueError, 'stdout'): self.check()
    def test_elapsed_reset(self):
        self.history['elapsed_seconds'] = 2.
        with self.assertRaisesRegex(ValueError, 'elapsed'): self.check()
    def test_first_resumed_time_reset(self):
        self.history['training'] = self.full['training'][:501] + [dict(self.full['training'][501], elapsed_seconds=100.)] + self.full['training'][502:]
        self.stdout = self.history['training'][501:]
        with self.assertRaisesRegex(ValueError, 'elapsed'): self.check()
    def test_status_last_row_mismatch(self):
        self.status = {'last_training': self.stdout[-2]}
        with self.assertRaisesRegex(ValueError, 'stdout'): self.check()


class FrozenPayloadPredicates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        paths = {'verifier_source_path': HERE / 'supervise_sand_graph_support_science.py',
                 'capacity_source_path': HERE / 'measure_sand_graph_support_capacity.py',
                 'lifecycle_source_path': HERE / 'measure_sand_cuda_capacity_v2.py',
                 'original_trainer_path': HERE / 'train_sand_graph_support_cuda.py'}
        release = {k: str(v) for k, v in paths.items()}
        release['files_sha256'] = {str(v): A.sha(v) for v in paths.values()}
        cls.S = A.configure(release)

    def payload(self, completed):
        t, S = self.torch, self.S
        config = {'historical_runtime_uuid': 'old'}
        history = {'training': [], 'graph_updates': [None] * completed, 'elapsed_seconds': float(completed)}
        state = {'weight': t.zeros(1)}
        groups = {'params': [0], 'lr': 1e-4 * (.1 ** (max(completed - 1, 0) / 99999)),
                  'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False, 'maximize': False,
                  'foreach': False, 'capturable': False, 'differentiable': False, 'fused': False}
        moments = {} if not completed else {0: {'step': t.tensor(float(completed)), 'exp_avg': t.zeros(1), 'exp_avg_sq': t.ones(1)}}
        simulator = {'particle_dimensions': 2, 'nnode_in': 30, 'nedge_in': 3, 'latent_dim': 128, 'nmessage_passing_steps': 10,
                'nmlp_layers': 2, 'mlp_hidden_dim': 128, 'connectivity_radius': .015, 'nparticle_types': 9,
                'particle_type_embedding_size': 16, 'uncertainty_parameterization': 'variance', 'variance_floor': 1e-6,
                'max_num_neighbors': 128, 'detach_variance_features': True, 'radius_backend': 'scipy_host',
                'boundaries': [[.1, .9], [.1, .9]], 'boundary_clamp_limit': 1.,
                'normalization_stats': {k: {'mean': t.zeros(2), 'std': t.ones(2)} for k in ('velocity', 'acceleration')}}
        payload = {'format_version': 2, 'cuda_sand_graph_support_schema': S.T.SCHEMA, 'completed_steps': completed,
                   'run_config': config, 'run_config_sha256': S.B.canonical_hash(config),
                   'training_config': {'loss': 'faithful', 'cuda_sand_graph_support_run': config, 'completed_optimizer_updates': completed},
                   'history': history, 'simulator_config': simulator, 'state_dict': state,
                   'optimizer_state': {'param_groups': [groups], 'state': moments},
                   'rng_states': {'cpu': t.ones(10, dtype=t.uint8), 'cuda': t.ones(10, dtype=t.uint8)}}
        return payload, config, history

    def test_original_zero_parent50k_final100k_predicates(self):
        for step in (0, 50000, 100000):
            p, c, h = self.payload(step)
            self.assertEqual(self.S.check_payload(self.torch, p, c, step, h)['completed_steps'], step)
    def test_nonfinite_model_rejected(self):
        p, c, h = self.payload(100000); p['state_dict']['weight'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite float32'): self.S.check_payload(self.torch, p, c, 100000, h)
    def test_adam_step_rejected(self):
        p, c, h = self.payload(100000); p['optimizer_state']['state'][0]['step'] -= 1
        with self.assertRaisesRegex(ValueError, 'Adam endpoint'): self.S.check_payload(self.torch, p, c, 100000, h)
    def test_rng_missing_rejected(self):
        p, c, h = self.payload(50000); p['rng_states'].pop('cuda')
        with self.assertRaisesRegex(ValueError, 'RNG serialization'): self.S.check_payload(self.torch, p, c, 50000, h)
    def test_normalization_zero_rejected(self):
        p, c, h = self.payload(50000); p['simulator_config']['normalization_stats']['velocity']['std'][0] = 0
        with self.assertRaisesRegex(ValueError, 'positive-scale'): self.S.check_payload(self.torch, p, c, 50000, h)
    def test_historical_config_change_rejected(self):
        p, c, h = self.payload(50000); c = dict(c, historical_runtime_uuid='new')
        with self.assertRaisesRegex(ValueError, 'lineage'): self.S.check_payload(self.torch, p, c, 50000, h)



class NumericBytes(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        metadata = {'dim': 2}
        md_sha = emit(self.root / 'metadata.json', metadata)
        records = []
        for i in range(2):
            record = {}
            for name in ('positions', 'particle_types'):
                path = self.root / (name + str(i) + '.npy')
                path.write_bytes((name + str(i)).encode())
                record[name] = {'path': path.name, 'size_bytes': path.stat().st_size, 'sha256': A.sha(path)}
            records.append(record)
        manifest = {'metadata': metadata, 'records': records}
        m_sha = emit(self.root / 'manifest.json', manifest)
        structure_sha = emit(self.root / 'structural.json', {'synthetic': True})
        emit(self.root / 'admission.json', {'manifest_sha256': m_sha, 'structural_report_sha256': structure_sha})
        self.release = {'train_manifest': str(self.root / 'manifest.json'), 'admission': str(self.root / 'admission.json'),
                        'structural_report': str(self.root / 'structural.json')}
        # Frozen manifest validation is outside this helper-specific byte fixture.
        self.S = SimpleNamespace(T=SimpleNamespace(validate_manifest_contract=lambda m, a: None, METADATA_SHA=md_sha))

    def test_all_numeric_bytes(self):
        self.assertEqual(A.verify_training_bytes(self.S, self.release)['numeric_files'], 4)
    def test_numeric_mutation(self):
        (self.root / 'positions0.npy').write_bytes(b'different')
        with self.assertRaisesRegex(ValueError, 'array bytes changed'): A.verify_training_bytes(self.S, self.release)
    def test_metadata_mutation(self):
        emit(self.root / 'metadata.json', {'dim': 3})
        with self.assertRaisesRegex(ValueError, 'metadata'): A.verify_training_bytes(self.S, self.release)
    def test_manifest_not_readmitted(self):
        emit(self.root / 'manifest.json', {'metadata': {}, 'records': []})
        with self.assertRaisesRegex(ValueError, 'admission byte'): A.verify_training_bytes(self.S, self.release)


class Original50100Continuity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / 'original.stdout.jsonl'
        self.row = {'completed_steps': 50100, 'loss': .005, 'lr': .00003, 'frame_ids': ['train:000000:6', 'train:000001:7'],
                    'particles': 1000, 'elapsed_seconds': 12000., 'guarded_update_seconds': .1}
        self.path.write_text(json.dumps(self.row) + '\n')
        self.job = {'origin_stdout_path': str(self.path), 'origin_stdout_sha256': A.sha(self.path)}
        self.adapter = A.load(HERE / 'train_sand_runtime_migration_recovery_v1.py', '_test_read_only_migration_adapter')

    def test_same_scalar_with_different_timing(self):
        A.check_resumed_legacy(self.adapter, self.job, [dict(self.row, elapsed_seconds=14000., guarded_update_seconds=.2)])
    def test_different_loss_fails(self):
        with self.assertRaisesRegex(ValueError, 'scientific50100'): A.check_resumed_legacy(self.adapter, self.job, [dict(self.row, loss=.006)])
    def test_different_lr_fails(self):
        with self.assertRaisesRegex(ValueError, 'scientific50100'): A.check_resumed_legacy(self.adapter, self.job, [dict(self.row, lr=.00004)])
    def test_old_log_changed(self):
        self.path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'log changed'): A.check_resumed_legacy(self.adapter, self.job, [self.row])


class PairingAdmission(unittest.TestCase):
    def test_missing_arm(self):
        with self.assertRaisesRegex(ValueError, 'All four'): A.audit_recovery_pairing([])
    def test_duplicate_arm(self):
        row = {'id': 'base_seed1', 'arm': 'base', 'seed': 1, 'gpu': 0}
        with self.assertRaisesRegex(ValueError, 'All four'): A.audit_recovery_pairing([row] * 4)
    def test_unverified_endpoint(self):
        rows = [dict(zip(('id', 'arm', 'seed', 'gpu'), row)) for row in A.JOBS]
        with self.assertRaisesRegex(ValueError, 'Endpoint audits'): A.audit_recovery_pairing(rows)


if __name__ == '__main__': unittest.main(verbosity=2)
