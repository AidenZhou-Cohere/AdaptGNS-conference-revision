"""Synthetic, CPU-only contract checks. No original payload or data is read."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name('train_sand_runtime_migration_recovery_v1.py')
SPEC = importlib.util.spec_from_file_location('sand_runtime_migration_adapter_under_test', SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MigrationContractTests(unittest.TestCase):
    def runtimes(self):
        old = {'uuid': 'old-uuid', 'name': 'NVIDIA GB200', 'torch': '2.13.0+cu129', 'cuda': '12.9',
               'numpy': '2.5.3', 'scipy': '1.17.1', 'python': '3.12.3', 'platform': 'same-platform',
               'device': 'cuda:0', 'capability': [10, 0], 'threads': 2, 'tf32': False,
               'amp': False, 'compile': False, 'ddp': False, 'float32_matmul_precision': 'highest',
               'deterministic_algorithms': True, 'deterministic_warn_only': False, 'cublas_workspace_config': ':4096:8'}
        return old, {**old, 'uuid': 'new-uuid'}

    def test_only_uuid_migration_is_accepted(self):
        old, new = self.runtimes()
        M.runtime_contract(old, new)
        for key in old:
            if key == 'uuid':
                continue
            changed = copy.deepcopy(new)
            changed[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.runtime_contract(old, changed)

    def test_missing_extra_and_unchanged_uuid_fail(self):
        old, new = self.runtimes()
        for candidate in [old, {k: v for k, v in new.items() if k != 'platform'}, {**new, 'waiver': True}]:
            with self.assertRaises(ValueError):
                M.runtime_contract(old, candidate)

    def make_origin(self, root):
        origin = root / 'original/jobs/base_seed1'
        origin.mkdir(parents=True)
        logs = origin.parent.parent / 'logs'
        logs.mkdir()
        log = logs / 'base_seed1.stdout.jsonl'
        log.write_text(json.dumps({'completed_steps': 50100, 'loss': -1.2, 'lr': .00003, 'frame_ids': ['train:1:7', 'train:2:8'], 'particles': 15}) + '\n')
        old, new = self.runtimes()
        config = {'runtime': old, 'seed': 1, 'arm': 'base', 'updates': 100000, 'research_protocol_sha256': M.PROTOCOL_SHA}
        (origin / 'protocol.json').write_text(json.dumps(config))
        (origin / 'checkpoint-000050000.pt').write_bytes(b'opaque synthetic bytes, never deserialized')
        checkpoint_sha = M.sha(origin / 'checkpoint-000050000.pt')
        canonical = M.hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        pointer = {'path': 'checkpoint-000050000.pt', 'completed_steps': 50000, 'sha256': checkpoint_sha, 'run_config_sha256': canonical}
        (origin / 'latest.json').write_text(json.dumps(pointer))
        (origin / 'run.lock').write_text(json.dumps({'host': 'old-host', 'pid': 123}))
        job = {'id': 'base_seed1', 'arm': 'base', 'seed': 1, 'gpu': 0, 'origin_directory': str(origin),
               'output_directory': str(root / 'new'), 'replay_output_directory': str(root / 'replay'),
               'origin_stdout_path': str(log), 'origin_stdout_sha256': M.sha(log),
               'original_runtime': old, 'actual_runtime': new}
        pins = {}
        for name, key in M.PARENT_FILES.items():
            job[key] = M.sha(origin / name)
            pins[str(origin / name)] = job[key]
        pins[str(log)] = job['origin_stdout_sha256']
        release = {'original_host': 'old-host', 'files_sha256': pins, 'jobs': [job],
                   'train_manifest': str(root / 'data/train.json'), 'repo': str(root / 'repo')}
        return release, job, origin

    def test_exact_parent_guard_and_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, j, origin = self.make_origin(Path(tmp).resolve())
            before = {p.name: p.read_bytes() for p in origin.iterdir()}
            M.verify_origin(r, j)
            self.assertEqual(before, {p.name: p.read_bytes() for p in origin.iterdir()})
            (origin / 'checkpoint-000050000.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                M.verify_origin(r, j)

    def test_no_parent_or_existing_output_can_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, j, origin = self.make_origin(Path(tmp).resolve())
            self.assertEqual(M.protected_output(r, j, 'output_directory'), Path(j['output_directory']))
            for output in [origin, origin / 'child', origin.parent, origin.parent.parent / 'new_sibling', Path(j['replay_output_directory'])]:
                variant = {**j, 'output_directory': str(output)}
                with self.subTest(output=str(output)), self.assertRaises(ValueError):
                    M.protected_output(r, variant, 'output_directory')
            Path(j['output_directory']).mkdir()
            with self.assertRaises(ValueError):
                M.protected_output(r, j, 'output_directory')

    def test_original_numerical_source_stays_byte_pinned(self):
        frozen = SOURCE.with_name('train_sand_graph_support_cuda.py')
        self.assertEqual(M.sha(frozen), M.TRAINER_SHA)
        text = SOURCE.read_text()
        self.assertIn("T.run_training(args, helpers, support, process, device, config['runtime'])", text)
        self.assertNotIn('exec(', text)
        self.assertNotIn('T.restore_payload =', text)

    def test_legacy_50100_comparison_is_exact_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, j, _ = self.make_origin(Path(tmp).resolve())
            self.assertEqual(M.legacy_replay_endpoint(j)['completed_steps'], 50100)
            path = Path(j['origin_stdout_path'])
            path.write_text(path.read_text() + path.read_text())
            j['origin_stdout_sha256'] = M.sha(path)
            with self.assertRaises(ValueError):
                M.legacy_replay_endpoint(j)


if __name__ == '__main__':
    unittest.main()
