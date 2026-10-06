"""Independent synthetic hierarchy/admission checks; never load real assets."""
import copy
import json

import pytest

from research import summarize_continuation_evaluation as summary


@pytest.fixture(autouse=True)
def no_checkpoint_or_model_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent tests must not load checkpoints or run inference')
    monkeypatch.setattr(summary.torch, 'load', forbidden)
    monkeypatch.setattr(summary.evaluation, 'load_verified_cohort', forbidden)
    monkeypatch.setattr(summary.full, 'load_for_evaluation', forbidden)


def population(policies=summary.evaluation.POLICIES):
    # Three frames in A and one in B deliberately distinguish trajectory
    # weighting from frame weighting. Expected effects below are hand-derived.
    units = [{'trajectory_id': 'A', 'unit_id': str(i)} for i in (1, 2, 3)]
    units.append({'trajectory_id': 'B', 'unit_id': '1'})
    nominal, shift, risk_shift = [1, 3, 5, 9], [0, 2, 4, 10], [-3, 0, 3, None]
    offsets = dict(zip(policies, [0, -1, 2, 3, 4]))
    records = []
    for arm in ['base', 'mix']:
        for seed in range(3):
            for index, unit in enumerate(units):
                base = 10 + seed + nominal[index]
                values = {policy: base + offsets[policy] for policy in policies}
                if arm == 'mix':
                    values = {policy: value + shift[index] * (seed + 1) for policy, value in values.items()}
                    values[policies[-1]] += risk_shift[index] if index < 3 else 4 * (seed - 1)
                records.append({'arm': arm, 'seed': seed, **unit, 'policy_values': values})
    return list(reversed(records)), units


@pytest.mark.parametrize('policies', [summary.evaluation.POLICIES, summary.native.POLICIES])
def test_independent_unequal_frame_hierarchy_and_unit_first_interaction(policies):
    rows, units = population(policies)
    result = summary.aggregate_metric(rows, units, policies)
    base = result['absolute']['base']['base']
    mix = result['absolute']['mix']['base']
    assert base['seed_values'] == [16., 17., 18.]
    assert (base['mean'], base['sample_sd']) == (17., 1.)
    assert mix['seed_values'] == [22., 29., 36.]
    assert (mix['mean'], mix['sample_sd']) == (29., 7.)
    change = result['paired']['mix_minus_base__base']
    assert change['seed_values'] == [6., 12., 18.]
    assert (change['mean'], change['sample_sd']) == (12., 6.)
    interaction = result['paired']['risk_minus_random_interaction']
    assert interaction['seed_values'] == [-2., 0., 2.]
    assert (interaction['mean'], interaction['sample_sd']) == (0., 2.)
    assert result['paired']['base__risk_minus_random']['seed_values'] == [2., 2., 2.]
    assert result['paired']['mix__risk_minus_random']['seed_values'] == [0., 2., 4.]
    assert result['paired']['mix_minus_base__' + policies[-1]]['seed_values'] == [4., 12., 20.]
    assert len(result['paired']) == 12
    first = interaction['seeds'][0]
    assert [row['value'] for row in first['trajectories']] == [0., -4.]
    assert [row['value'] for row in first['trajectories'][0]['units']] == [-3., 0., 3.]
    assert first['required_units'] == first['defined_units'] == 4


def test_one_undefined_risk_metric_preserves_other_policies_and_null_reason():
    rows, units = population()
    risk = summary.evaluation.POLICIES[-1]
    selected = next(row for row in rows if (row['arm'], row['seed'], row['trajectory_id'], row['unit_id']) == ('base', 1, 'A', '2'))
    selected['policy_values'][risk] = None
    selected['policy_null_reasons'] = {risk: 'constant signed-gain rank'}
    result = summary.aggregate_metric(rows, units)
    affected = result['paired']['risk_minus_random_interaction']
    assert affected['seed_values'] == [-2., None, 2.]
    assert affected['mean'] is affected['sample_sd'] is None
    assert affected['defined_seeds'] == 2
    failed_unit = affected['seeds'][1]['trajectories'][0]['units'][1]
    assert failed_unit['value'] is None
    assert 'constant signed-gain rank' in failed_unit['null_reason']
    assert result['paired']['mix_minus_base__base']['mean'] == 12.
    assert result['absolute']['mix'][risk]['defined_seeds'] == 3


def test_missing_unit_does_not_become_a_survivor_mean():
    rows, units = population()
    rows = [row for row in rows if (row['arm'], row['seed'], row['trajectory_id'], row['unit_id']) != ('mix', 2, 'B', '1')]
    result = summary.aggregate_metric(rows, units)
    effect = result['paired']['mix_minus_base__base']
    assert effect['seed_values'] == [6., 12., None]
    assert effect['mean'] is effect['sample_sd'] is None
    assert effect['seeds'][2]['null_units'] == 1
    assert effect['seeds'][2]['trajectories'][1]['units'][0]['null_reason'] == 'mix/base: missing committed unit'
    assert result['paired']['base__dense_minus_base']['seed_values'] == [-1., -1., -1.]


@pytest.mark.parametrize('defect', ['duplicate_row', 'unknown_unit', 'duplicate_expected', 'boolean_seed', 'boolean_value', 'nan', 'infinity', 'missing_policy', 'reordered_family'])
def test_unknown_or_noncanonical_metric_identity_rejected(defect):
    rows, units = population()
    policies = summary.evaluation.POLICIES
    if defect == 'duplicate_row':
        rows.append(copy.deepcopy(rows[0]))
    elif defect == 'unknown_unit':
        rows[0]['unit_id'] = 'unexpected'
    elif defect == 'duplicate_expected':
        units.append(copy.deepcopy(units[0]))
    elif defect == 'boolean_seed':
        rows[0]['seed'] = True
    elif defect == 'boolean_value':
        rows[0]['policy_values']['base'] = False
    elif defect in ('nan', 'infinity'):
        rows[0]['policy_values']['base'] = float('nan' if defect == 'nan' else 'inf')
    elif defect == 'missing_policy':
        rows[0]['policy_values'].pop('base')
    else:
        policies = tuple(reversed(policies))
    with pytest.raises(ValueError):
        summary.aggregate_metric(rows, units, policies)


def test_failed_autonomous_record_retains_early_horizons_failure_and_elapsed_time():
    boundary = {name: None for name in summary.scalar_audit.BOUNDARY_METRICS}
    native = {name: None for name in summary.native_audit.EXTRA_METRICS}
    for name in native:
        if 'seconds' in name or 'passes' in name:
            native[name] = 2.
    row = {'status': 'failed', 'completed_steps': 333, 'forecast_network_passes': 334,
           'total_network_passes': 335, 'total_wall_seconds': 12.5,
           'mean_rollout_mse': None, 'mean_directed_edges': None,
           'mse_at_steps': {'1': 2.1, '10': 2.2, '50': 2.3, '200': 2.4, '500': None, '995': None},
           'boundary_diagnostics': {'full_horizon': boundary}, 'native_metrics': native}
    values = summary.autonomous_values(row)
    assert values['mse_at_200'] == 2.4
    assert values['mse_at_500'] is values['mse_at_995'] is values['mean_rollout_mse'] is None
    assert values['failure_fraction'] == 1.
    assert values['mean_rollout_wall_seconds'] == 12.5
    assert values['completed_steps'] == 333 and values['forecast_network_passes'] == 334
    assert all(values[name] is None for name in boundary)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def jobs(tmp_path):
    for arm in ('base', 'mix'):
        for seed in range(3):
            directory = tmp_path / f'{arm}_{seed}'
            directory.mkdir()
            write_json(directory / 'protocol.json', {'mode': 'observed', 'arm': arm, 'seed': seed})
    return tmp_path


def test_discovery_uses_protocol_identity_and_requires_exact_cohort(tmp_path):
    root = jobs(tmp_path)
    assert set(summary.discover_jobs(root, 'observed')) == summary.evaluation.COHORT
    (root / 'base_0').rename(root / 'arbitrary_directory_name')
    assert set(summary.discover_jobs(root, 'observed')) == summary.evaluation.COHORT


@pytest.mark.parametrize('defect', ['duplicate_arm', 'wrong_mode', 'boolean_seed', 'missing_job', 'extra_file', 'symlink_job'])
def test_discovery_refuses_selection_from_invalid_or_incomplete_family(tmp_path, defect):
    root = jobs(tmp_path)
    target = root / 'mix_0'
    if defect == 'duplicate_arm':
        write_json(target / 'protocol.json', {'mode': 'observed', 'arm': 'base', 'seed': 0})
    elif defect == 'wrong_mode':
        write_json(target / 'protocol.json', {'mode': 'autonomous', 'arm': 'mix', 'seed': 0})
    elif defect == 'boolean_seed':
        write_json(target / 'protocol.json', {'mode': 'observed', 'arm': 'mix', 'seed': False})
    elif defect == 'missing_job':
        (target / 'protocol.json').unlink(); target.rmdir()
    elif defect == 'extra_file':
        (root / 'unexpected.json').write_text('{}')
    else:
        (root / 'alias').symlink_to(target, target_is_directory=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        summary.discover_jobs(root, 'observed')


def committed_fixture(tmp_path, mode):
    protocol = {'schema': 1, 'scope': summary.EVALUATION_SCOPE, 'mode': mode, 'arm': 'mix', 'seed': 1,
                'original_seed': 1, 'objective': 'faithful', 'checkpoint_sha256': 'a' * 64,
                'parent_checkpoint_sha256': 'b' * 64, 'continuation_config_sha256': 'c' * 64,
                'completed_total_updates': 110000, 'completed_additional_updates': 10000}
    write_json(tmp_path / 'protocol.json', protocol)
    result = {**protocol, 'state': 'complete', 'protocol_sha256': summary.full.sha256(tmp_path / 'protocol.json'),
              'invocation_wall_seconds': 3.0}
    write_json(tmp_path / 'result.json', result)
    status = {'state': 'complete', 'arm': 'mix', 'seed': 1, 'result_sha256': summary.full.sha256(tmp_path / 'result.json')}
    if mode == 'observed':
        status['committed_frames'] = 425
    write_json(tmp_path / 'status.json', status)
    if mode == 'autonomous':
        write_json(tmp_path / 'lineage_identity.json', {key: protocol[key] for key in summary.IDENTITY_FIELDS})
    return protocol, result, status


@pytest.mark.parametrize('mode', ['observed', 'autonomous'])
def test_committed_publication_requires_status_hash_and_lineage(tmp_path, mode):
    protocol, result, status = committed_fixture(tmp_path, mode)
    actual, _, _ = summary.committed_job(tmp_path, mode, summary.Audit())
    assert actual == protocol
    status['state'] = 'running'
    write_json(tmp_path / 'status.json', status)
    with pytest.raises(ValueError, match='uncommitted'):
        summary.committed_job(tmp_path, mode, summary.Audit())


@pytest.mark.parametrize('defect', ['stale_result_hash', 'rehashed_wrong_seed', 'rehashed_wrong_parent', 'rehashed_wrong_arm'])
def test_completed_observed_publication_cannot_hide_changed_identity(tmp_path, defect):
    _, result, status = committed_fixture(tmp_path, 'observed')
    if defect == 'stale_result_hash':
        result['invocation_wall_seconds'] = 4.
    else:
        result[{'rehashed_wrong_seed': 'seed', 'rehashed_wrong_parent': 'parent_checkpoint_sha256', 'rehashed_wrong_arm': 'arm'}[defect]] = {'rehashed_wrong_seed': 0, 'rehashed_wrong_parent': 'd' * 64, 'rehashed_wrong_arm': 'base'}[defect]
    write_json(tmp_path / 'result.json', result)
    if defect != 'stale_result_hash':
        status['result_sha256'] = summary.full.sha256(tmp_path / 'result.json')
        write_json(tmp_path / 'status.json', status)
    with pytest.raises(ValueError):
        summary.committed_job(tmp_path, 'observed', summary.Audit())


def test_native_completed_result_still_needs_exact_arm_identity(tmp_path):
    protocol, _, _ = committed_fixture(tmp_path, 'autonomous')
    identity = {key: protocol[key] for key in summary.IDENTITY_FIELDS}
    identity['arm'] = 'base'
    write_json(tmp_path / 'lineage_identity.json', identity)
    with pytest.raises(ValueError, match='lineage identity'):
        summary.committed_job(tmp_path, 'autonomous', summary.Audit())


@pytest.mark.parametrize('payload', ['{"seed":0,"seed":1}', '{"x":NaN}', '{"x":Infinity}'])
def test_json_reader_rejects_ambiguous_or_nonfinite_payload(tmp_path, payload):
    path = tmp_path / 'record.json'
    path.write_text(payload)
    with pytest.raises(ValueError):
        summary.read_json(path)
