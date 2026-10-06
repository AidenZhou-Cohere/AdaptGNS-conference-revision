"""Goop preparation/release/saved-batch CPU tests, with no real data or model."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('goop_validation_test', HERE/'validate_goop_graph_support_cuda.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_default_imports_no_scientific_dependencies():
    code = """import builtins,runpy,sys
old=builtins.__import__
def restricted(name,*a,**kw):
 if name.split('.')[0] in ('torch','numpy','scipy','research','gns'): raise AssertionError(name)
 return old(name,*a,**kw)
builtins.__import__=restricted
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-I', '-c', code, str(HERE/'validate_goop_graph_support_cuda.py')], capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    assert value['max_optimizer_updates'] == 15 and value['scientific_training_admitted'] is False


def test_pinned_private_helpers_and_numeric_ast_are_unchanged():
    assert m.shared_numeric_check()
    mods = m.modules()
    assert mods['train_goop_graph_support_cuda.py'].SCHEMA == 'adaptgns_goop_graph_support_cuda_training_v1'
    assert mods['validate_sand_graph_support_cuda.py'].BRANCHES == m.BRANCHES


def synthetic_manifest():
    return {'dataset': 'Goop', 'split': 'train', 'record_count': 1000,
            'records': [{'id': f'train:{i:06d}', 'source_index': i, 'positions': {'shape': [401, 1000-i, 2]}} for i in range(1000)]}


def test_prospective_schedule_uses_source_structure_only():
    selector = m.modules()['validate_sand_cuda.py']; manifest = synthetic_manifest()
    schedule = m.expected_schedule(manifest, selector)
    assert [r['trajectory_indices'] for r in schedule] == [[999, 998], [499, 498], [1, 0]]
    assert all(r['target_frames'] == [6, 200] for r in schedule)
    assert schedule[0]['dataset_indices'] == [999*395, 998*395+194]
    manifest['records'][1]['positions']['shape'][0] = 400
    with pytest.raises(ValueError): m.expected_schedule(manifest, selector)


def test_both_modes_require_explicit_releases_and_source_paths():
    with pytest.raises(SystemExit): m.parse_args(['--execute'])
    with pytest.raises(SystemExit): m.parse_args(['--prepare-batches'])
    with pytest.raises(SystemExit): m.parse_args(['--execute', '--prepare-batches'])
    with pytest.raises(SystemExit): m.parse_args(['--threads', '1'])


def test_selection_release_cannot_be_selected_from_outcomes(tmp_path, monkeypatch):
    trainer = m.modules()['train_goop_graph_support_cuda.py']
    args = SimpleNamespace(admission=tmp_path/'admission.json', protocol=tmp_path/'protocol.md')
    args.protocol.write_text('prospective protocol')
    real_sha = m.sha
    monkeypatch.setattr(m, 'sha', lambda p: m.ADMISSION_SHA if Path(p) == args.admission else real_sha(p))
    schedule = [{'case': 'synthetic'}]
    release = {'schema': m.SELECTION_RELEASE, 'status': 'admitted_for_batch_preparation', 'issued_by': 'root',
               'scientific_training_admitted': False, 'train_manifest_sha256': trainer.MANIFEST_SHA,
               'training_admission_sha256': m.ADMISSION_SHA, 'trainer_sha256': m.TRAINER_SHA,
               'preparation_source_sha256': real_sha(HERE/'validate_goop_graph_support_cuda.py'),
               'protocol_sha256': real_sha(args.protocol), 'schedule': schedule}
    m.validate_selection_release(release, trainer, args, schedule)
    for key, value in [('issued_by', 'worker'), ('scientific_training_admitted', True), ('schedule', []), ('trainer_sha256', '0'*64)]:
        altered = {**release, key: value}
        with pytest.raises(ValueError): m.validate_selection_release(altered, trainer, args, schedule)


@pytest.fixture
def saved(tmp_path):
    trainer = m.modules()['train_goop_graph_support_cuda.py']
    h, _ = trainer.load_helpers(HERE.parents[2]/'outputs'/'AdaptGNS')
    types = torch.full((5,), 7, dtype=torch.int64)
    arrays = {'position_sequence': np.zeros((5, 6, 2), dtype=np.float32),
              'next_positions': np.ones((5, 2), dtype=np.float32), 'particle_types': types.numpy(),
              'nparticles_per_example': np.asarray([2, 3], dtype=np.int64),
              'position_sequence_noise': h.host_noise((5, 6, 2), types, 0, 0).numpy()}
    path = tmp_path/'small_batch.npz'; np.savez(path, **arrays)
    case = {'case': 'small', 'noise_seed': 0, 'noise_step': 0, 'particle_counts': [2, 3],
            'batch_file': path.name, 'batch_sha256': m.sha(path)}
    return h, arrays, path, case


def test_saved_goop_types_and_exact_prospective_host_noise(saved):
    h, arrays, path, case = saved
    batch, returned = m.check_saved_batch(h, case, path.parent, 0)
    assert returned == path and torch.equal(batch['particle_types'], torch.full((5,), 7))
    arrays['position_sequence_noise'][0, -1, 0] += .1
    np.savez(path, **arrays); case['batch_sha256'] = m.sha(path)
    with pytest.raises(ValueError, match='noise differs'): m.check_saved_batch(h, case, path.parent, 0)


def test_saved_sand_types_and_nonfinite_positions_are_rejected(saved):
    h, arrays, path, case = saved
    arrays['particle_types'][:] = 6
    np.savez(path, **arrays); case['batch_sha256'] = m.sha(path)
    with pytest.raises(ValueError, match='type7'): m.check_saved_batch(h, case, path.parent, 0)
    arrays['particle_types'][:] = 7; arrays['position_sequence'][0, 0, 0] = np.nan
    np.savez(path, **arrays); case['batch_sha256'] = m.sha(path)
    with pytest.raises(ValueError, match='finiteness'): m.check_saved_batch(h, case, path.parent, 0)


def test_batch_report_requires_exact_complete_prospective_schedule():
    mods = m.modules(); trainer = mods['train_goop_graph_support_cuda.py']
    schedule = m.expected_schedule(synthetic_manifest(), mods['validate_sand_cuda.py'])
    report = {'schema': m.BATCH_SCHEMA, 'status': 'complete_saved_training_batches', 'source_sha256': m.sha(HERE/'validate_goop_graph_support_cuda.py'),
              'helper_source_sha256': m.PINS, 'train_manifest_sha256': trainer.MANIFEST_SHA, 'metadata_sha256': trainer.METADATA_SHA,
              'training_admission_sha256': m.ADMISSION_SHA, 'selection_uses_outcomes': False, 'new_optimizer_updates': 0,
              'test_accessed': False, 'schedule': schedule, 'cases': copy.deepcopy(schedule)}
    m.validate_batch_report(report, trainer, schedule)
    report['cases'][1]['trajectory_indices'] = [0, 1]
    with pytest.raises(ValueError, match='schedule'): m.validate_batch_report(report, trainer, schedule)


def test_reused_completion_gate_refuses_missing_cases_or_replay():
    core = m.modules()['validate_sand_graph_support_cuda.py']
    assert core.completion_gate({'shared_trainer_functions_exact': True, 'cases': [], 'replay': {}}) is False
