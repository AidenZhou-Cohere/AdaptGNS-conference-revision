"""Pure scalar fixtures; no numerical libraries, actual arrays or device work."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import goop3d_deadline_worksheet_v1 as W


def test_description_does_not_import_numerical_libraries():
    code = """import builtins,runpy,sys
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','numpy','scipy','gns','research'):raise AssertionError(name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-I', '-c', code, W.__file__], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['test_accessed'] is False


def capacity():
    jobs = [{'arm': a, 'seed': s, 'wave': 'A' if s < 2 else 'B', 'status': 'verified_capacity_only',
        'final_pointer': {'completed_steps': 512, 'sha256': str(s) * 64},
        'steady_wall_seconds_per_update': 2. if s < 2 else 1.,
        'nonnegative_external_minus_all_guarded_seconds': 10. if s < 2 else 5.}
        for a in ('base', 'mix') for s in range(3)]
    return {'schema': 'adaptgns_goop3d_vectorized_capacity_v1', 'status': 'all_six_verified_capacity_only',
        'scientific_training_admitted': False, 'scientific_endpoint_selected': False, 'jobs': jobs,
        'q4_q2': {'A': 2., 'B': 1.}, 'r4_r2': {'A': 10., 'B': 5.},
        'pairing': [{'seed': s, 'initial_model_rng_Adam_exact': True, 'all512_sample_noise_lr_rows_exact': True} for s in range(3)]}


def test_capacity_recomputes_wave_maxima_and_requires_every_pair():
    source = capacity(); q, r, pointers = W.capacity_inputs(source)
    assert q == {'A': 2., 'B': 1.} and r == {'A': 10., 'B': 5.} and len(pointers) == 6
    for mutate in (lambda x: x['jobs'].pop(), lambda x: x['q4_q2'].update(A=1.),
                   lambda x: x['pairing'][0].update(all512_sample_noise_lr_rows_exact=False)):
        changed = copy.deepcopy(source); mutate(changed)
        with pytest.raises(ValueError): W.capacity_inputs(changed)


def test_forecast_keeps_two_training_waves_six_models_two_splits_and_no_overlap_credit():
    q, residual, _ = W.capacity_inputs(capacity())
    plan = {'schema': 'adaptgns_goop3d_deadline_planning_config_v1', 'selected_endpoint': None,
        'evaluation_overlap_credit': False, 'candidate_endpoints': [1000, 10000], 'checkpoint_every': 1000,
        'runtime_multiplier': 1.5, 'transfer_review_analysis_reserve_seconds': 600,
        'planning_start_utc': '2026-10-06T20:00:00+00:00'}
    timing = [{'arm': a, 'mode': m, 'per_unit_seconds': 2., 'setup_and_other_seconds': 3.}
              for a in ('base', 'mix') for m in W.MODES]
    rows = W.forecast(plan, q, residual, timing)
    evaluation = 6 * (60 * 2 + 300 * 2 + 128 * 2 + 5 * 3)
    assert rows[0]['evaluation_serial_seconds'] == evaluation
    assert rows[0]['training_4_plus_2_wave_core_seconds'] == 3000
    assert rows[0]['conservative_checkpoint_and_setup_seconds'] == 30
    assert rows[0]['projected_total_seconds'] == 1.5 * (3000 + 30 + evaluation) + 600
    assert rows[0]['fits_stated_planning_assumptions'] is True
    assert rows[1]['fits_stated_planning_assumptions'] is False
    with pytest.raises(ValueError): W.forecast({**plan, 'evaluation_overlap_credit': True}, q, residual, timing)
    with pytest.raises(ValueError): W.forecast({**plan, 'selected_endpoint': 1000}, q, residual, timing)
    with pytest.raises(ValueError): W.forecast(plan, q, residual, timing[:-1])


@pytest.fixture
def timed(tmp_path):
    evaluator = W.load_evaluator()
    manifest = {'records': [{'id': f'valid:{i:06d}', 'source_index': i, 'positions': {'shape': [301, 100 + i, 3]}} for i in range(100)]}
    expected = evaluator.schedules(manifest['records'], 'full-rollout', 'capacity_timing')
    p = {'schema': evaluator.SCHEMA, 'purpose': 'capacity_timing', 'split': 'valid', 'arm': 'base', 'seed': 0,
        'mode': 'full-rollout', 'checkpoint_updates': 512, 'checkpoint_sha256': 'a' * 64, 'horizon': 295,
        'policies': list(evaluator.POLICIES), 'schedule': expected, 'source_population_count': 100,
        'prospective_source_grid': evaluator.grid_indices(100)}
    bindings = {}
    def save(name, value):
        path = tmp_path / name; path.write_text(json.dumps(value)); bindings[str(path)] = W.sha(path); return path
    protocol = save('protocol.json', p)
    save('status.json', {'state': 'complete', 'all_inputs_reverified': True, 'total_execution_seconds': 40,
        'committed_rows': 18, 'expected_rows': 18})
    for item in expected:
        for policy in evaluator.POLICIES:
            name = f'trajectory_{item["source_index"]:06d}_{policy}'
            array = save(name + '.npz', {'placeholder_bytes': True})
            save(name + '.json', {**item, 'status': 'complete', 'failure': None, 'arm': 'base', 'training_seed': 0,
                'protocol_sha256': W.sha(protocol), 'artifact_sha256': W.sha(array), 'artifact_file': array.name,
                'synchronized_call_seconds': 1., 'publication_seconds': .5, 'policy': policy,
                'completed_steps': 295, 'mse_per_step': [0.] * 295})
    entry = {'directory': str(tmp_path), 'arm': 'base', 'mode': 'full-rollout', 'files_sha256': bindings,
        'external': {'stopped_and_reaped': True, 'exit_code': 0, 'pid': 1, 'elapsed_seconds': 45}}
    return entry, manifest, evaluator, save


def test_full_h_timing_uses_all_six_policy_maxima_and_whole_process_overhead(timed):
    entry, manifest, evaluator, _ = timed
    row = W.collect_timing(entry, manifest, 'a' * 64, evaluator)
    assert row['complete_rows'] == 18 and row['per_unit_seconds'] == 9
    assert row['setup_and_other_seconds'] == 18


@pytest.mark.parametrize('failure', ['prefix', 'failed_status', 'missing_policy', 'unreaped', 'artifact_changed'])
def test_failed_or_incomplete_full_h_timing_cannot_be_extrapolated(timed, failure):
    entry, manifest, evaluator, save = timed
    path = next(Path(entry['directory']).glob('trajectory*_base.json'))
    row = W.read(path)
    if failure == 'prefix':
        row.update(completed_steps=294, mse_per_step=[0.] * 294); save(path.name, row)
    elif failure == 'failed_status':
        row.update(status='failed', failure={'category': 'candidate_pair_resource_guard'}); save(path.name, row)
    elif failure == 'missing_policy':
        entry['files_sha256'].pop(str(path))
    elif failure == 'unreaped':
        entry['external']['stopped_and_reaped'] = False
    else:
        next(Path(entry['directory']).glob('*.npz')).write_text('changed')
    with pytest.raises(ValueError): W.collect_timing(entry, manifest, 'a' * 64, evaluator)


@pytest.mark.parametrize('mutate_after_collection', [False, True])
def test_main_rehashes_collected_timing_before_admitting_worksheet(tmp_path, monkeypatch, mutate_after_collection):
    # Main control uses synthetic scalar files and an isolated collector; this
    # checks that a late artifact mutation cannot leave successful forecasts.
    paths = {k: tmp_path / (k + '.json') for k in ('release', 'capacity_summary', 'valid_manifest', 'timing_inventory', 'planning_config')}
    output = tmp_path / 'result.json'; artifact = tmp_path / 'synthetic_timing.npz'; artifact.write_text('not real arrays')
    paths['capacity_summary'].write_text(json.dumps(capacity()))
    paths['valid_manifest'].write_text(json.dumps({'records': []}))
    monkeypatch.setattr(W, 'VALID_MANIFEST_SHA', W.sha(paths['valid_manifest']))
    entries = [{'arm': a, 'mode': m} for a in ('base', 'mix') for m in W.MODES]
    paths['timing_inventory'].write_text(json.dumps({'schema': 'adaptgns_goop3d_capacity_timing_inventory_v1',
        'issued_by': 'root', 'status': 'all_required_processes_stopped', 'entries': entries}))
    paths['planning_config'].write_text('{}')
    paths['release'].write_text(json.dumps({'schema': 'adaptgns_goop3d_deadline_worksheet_release_v1',
        'status': 'admitted_for_scalar_planning', 'issued_by': 'root', 'scientific_training_admitted': False,
        'files_sha256': {str(p.resolve()): W.sha(p) for p in [v for k, v in paths.items() if k != 'release'] + [Path(W.__file__)]}}))
    monkeypatch.setattr(W, 'collect_timing', lambda entry, *args: {**entry, 'per_unit_seconds': 1.,
        'setup_and_other_seconds': 0., 'files_sha256': {str(artifact): W.sha(artifact)}})
    def finish(*args):
        if mutate_after_collection: artifact.write_text('late mutation')
        return [{'synthetic_estimate': True}]
    monkeypatch.setattr(W, 'forecast', finish)
    argv = ['--execute', '--output-file', str(output)]
    for key, path in paths.items(): argv += ['--' + key.replace('_', '-'), str(path)]
    assert W.main(argv) == (2 if mutate_after_collection else 0)
    result = W.read(output)
    assert str(artifact) in result['input_sha256']
    if mutate_after_collection:
        assert result['status'] == 'forecast_unavailable' and result['forecasts'] == []
        assert result['unavailable_reasons'][0]['reason'] == 'Planning input changed'
    else:
        assert result['status'] == 'engineering_worksheet_complete_not_execution_admission'
