"""Synthetic record checks; never loads experimental models or datasets."""
import json
import statistics

import pytest

from research import summarize_training_validation as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def make_point(config, step=0, multiplier=1.):
    rows = []
    for index, identity in enumerate(config['validation_frames']):
        mse = multiplier * (1 + index / 10)
        rows.append({**identity, 'dynamic_particles': 1 + index % 7,
                     'coordinate_mse': mse, 'realized_vector_se': 2 * mse,
                     'predicted_vector_se': 3 * mse, 'constant_free_gaussian_nll': mse - 10})
    trajectories = {name: {metric: statistics.fmean(row[metric] for row in rows if row['trajectory'] == name)
                           for metric in report.METRICS[:4]}
                    for name in {row['trajectory'] for row in rows}}
    means = {metric: statistics.fmean(row[metric] for row in trajectories.values()) for metric in report.METRICS[:4]}
    return {'run_config_sha256': report.canonical(config), 'completed_steps': step, 'frames': rows,
            'trajectories': trajectories, 'equal_trajectory_mean': means,
            'calibration_bins': [{'index': 0, 'particles': sum(row['dynamic_particles'] for row in rows),
                                  'weight': 1., 'predicted_vector_se': means['predicted_vector_se'],
                                  'realized_vector_se': means['realized_vector_se']}],
            'binned_vector_se_calibration_gap': means['predicted_vector_se'] - means['realized_vector_se']}


@pytest.fixture
def config():
    # Deliberately unequal frames/trajectory and particles/frame.
    frames = []
    for trajectory in range(30):
        for local in range(5 if trajectory < 8 else 4):
            target = 6 + local
            name = f'valid:{trajectory:06d}'
            frames.append({'dataset_index': trajectory * 995 + local, 'trajectory': name,
                           'target_frame': target, 'id': f'{name}:{target}'})
    return {'scope': 'bounded_full_data_100k', 'objective': 'faithful', 'seed': 0,
            'steps': 100000, 'validation_interval': 5000, 'checkpoint_interval': 10000,
            'batch_size': 2, 'history': 6, 'noise_std': 6.7e-4,
            'architecture': {'width': 128, 'message_passing_blocks': 10, 'mlp_layers': 2},
            'graph': {'radius': .015, 'radius_backend': 'scipy_host', 'max_neighbors': 128,
                      'self_loops': True, 'training': 'base only; no radius augmentation'},
            'optimizer': {'kind': 'Adam', 'initial_lr': 1e-4, 'final_lr': 1e-5, 'decay_updates': 100000,
                          'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'gradient_clipping': None},
            'validation_noise': 0., 'validation_normalization': 'training noise-adjusted scales',
            'train': {'n_trajectories': 1000}, 'valid': {'n_trajectories': 30},
            'runtime': {'device': 'mps', 'mps_fallback_environment': '0'},
            'research_protocol_sha256': 'a' * 64, 'validation_frames': frames}


def save_run(root, config, points=(0,), state='running', multiplier=1.):
    folder = root / f"{config['objective']}_seed{config['seed']}"
    write(folder / 'protocol.json', config)
    write(folder / 'status.json', {'state': state, 'completed_steps': max(points),
                                  'run_config_sha256': report.canonical(config), 'error': None})
    for step in points:
        write(folder / 'validation' / f'step-{step:06d}.json', make_point(config, step, multiplier))
    return folder


def test_equal_trajectory_weighting_differs_from_flat_frames_and_particles(config):
    point = make_point(config)
    result = report.validate_point(point, config, 0)
    rows = point['frames']
    flat = statistics.fmean(row['coordinate_mse'] for row in rows)
    particles = sum(row['coordinate_mse'] * row['dynamic_particles'] for row in rows) / sum(row['dynamic_particles'] for row in rows)
    assert result['coordinate_mse'] == point['equal_trajectory_mean']['coordinate_mse']
    assert result['coordinate_mse'] != pytest.approx(flat)
    assert result['coordinate_mse'] != pytest.approx(particles)


@pytest.mark.parametrize('corruption', ['frame_id', 'frame_count', 'vector_units', 'trajectory_mean',
                                     'global_mean', 'bin_count', 'bin_weight', 'bin_risk', 'gap',
                                     'nonfinite', 'negative', 'bin_index', 'run_hash', 'step'])
def test_corrupted_saved_metrics_are_rejected(config, corruption):
    point = make_point(config)
    if corruption == 'frame_id': point['frames'][0]['id'] = 'other'
    elif corruption == 'frame_count': point['frames'].pop()
    elif corruption == 'vector_units': point['frames'][0]['realized_vector_se'] *= 2
    elif corruption == 'trajectory_mean': point['trajectories']['valid:000000']['coordinate_mse'] += 1
    elif corruption == 'global_mean': point['equal_trajectory_mean']['coordinate_mse'] += 1
    elif corruption == 'bin_count': point['calibration_bins'][0]['particles'] += 1
    elif corruption == 'bin_weight': point['calibration_bins'][0]['weight'] = .5
    elif corruption == 'bin_risk': point['calibration_bins'][0]['predicted_vector_se'] += 1
    elif corruption == 'gap': point['binned_vector_se_calibration_gap'] += 1
    elif corruption == 'nonfinite': point['frames'][0]['coordinate_mse'] = float('nan')
    elif corruption == 'negative': point['frames'][0]['predicted_vector_se'] = -1
    elif corruption == 'bin_index': point['calibration_bins'][0]['index'] = True
    elif corruption == 'run_hash': point['run_config_sha256'] = 'b' * 64
    elif corruption == 'step': point['completed_steps'] = 5000
    with pytest.raises(ValueError):
        report.validate_point(point, config, 0)


def test_missing_seeds_are_not_replaced_by_available_mean(tmp_path, config):
    save_run(tmp_path, config, (0, 5000))
    result = report.summarize(tmp_path, 'a' * 64)
    assert [run['state'] for run in result['runs']] == ['running'] + ['not_started'] * 5
    assert len(result['runs'][0]['points']) == 2
    assert result['runs'][0]['missing_scheduled_updates'] == list(range(10000, 100001, 5000))
    for point in result['three_seed_curves']['faithful']:
        assert point['available_seeds'] == 1
        for metric in point['statistics'].values():
            assert metric['mean'] is metric['sample_sd'] is None
            assert metric['seed_values'][1:] == [None, None]


def test_three_seed_mean_and_sample_sd_and_failure_retention(tmp_path, config):
    for seed in report.SEEDS:
        cfg = {**config, 'seed': seed}
        save_run(tmp_path, cfg, multiplier=seed + 1, state='failed' if seed == 2 else 'running')
    result = report.summarize(tmp_path, 'a' * 64)
    stats = result['three_seed_curves']['faithful'][0]['statistics']['coordinate_mse']
    assert stats['mean'] == statistics.fmean(stats['seed_values'])
    assert stats['sample_sd'] == statistics.stdev(stats['seed_values'])
    assert result['runs'][2]['state'] == 'failed'


@pytest.mark.parametrize('field,value', [('steps', 50000), ('batch_size', 1), ('noise_std', 0.),
                                        ('validation_noise', 1.), ('research_protocol_sha256', 'bad')])
def test_config_changes_rejected(tmp_path, config, field, value):
    config[field] = value
    save_run(tmp_path, config)
    with pytest.raises(ValueError, match='fixed full-data'):
        report.summarize(tmp_path, 'a' * 64)


def test_common_source_configuration_must_match(tmp_path, config):
    save_run(tmp_path, {**config, 'metadata_sha256': 'a' * 64})
    save_run(tmp_path, {**config, 'seed': 1, 'metadata_sha256': 'b' * 64})
    with pytest.raises(ValueError, match='differs across models'):
        report.summarize(tmp_path, 'a' * 64)


def test_complete_run_requires_all_scheduled_points(tmp_path, config):
    save_run(tmp_path, config, (100000,), 'complete')
    with pytest.raises(ValueError, match='missing scheduled'):
        report.summarize(tmp_path, 'a' * 64)


def test_noncanonical_or_duplicate_step_is_rejected(tmp_path, config):
    folder = save_run(tmp_path, config)
    write(folder / 'validation/step-0.json', make_point(config))
    with pytest.raises(ValueError, match='noncanonical'):
        report.summarize(tmp_path, 'a' * 64)


def test_source_hash_verification_does_not_touch_source(tmp_path, config):
    source = tmp_path / 'research/full_training.py'
    source.parent.mkdir()
    source.write_text('original')
    config['source_sha256'] = {'research/full_training.py': report.sha(source)}
    save_run(tmp_path / 'results', config)
    assert report.summarize(tmp_path / 'results', 'a' * 64, tmp_path)['current_frozen_source_files_verified']
    source.write_text('changed')
    with pytest.raises(ValueError, match='Frozen training source'):
        report.summarize(tmp_path / 'results', 'a' * 64, tmp_path)
    assert source.read_text() == 'changed'


def test_unrelated_source_hash_does_not_verify_frozen_sources(tmp_path, config):
    source = tmp_path / 'unrelated.txt'
    source.write_text('original')
    config['source_sha256'] = {'unrelated.txt': report.sha(source)}
    save_run(tmp_path / 'results', config)
    with pytest.raises(ValueError, match='path set differs'):
        report.summarize(tmp_path / 'results', 'a' * 64, tmp_path)


@pytest.mark.parametrize('keep_status', [True, False])
def test_missing_protocol_does_not_hide_saved_evidence(tmp_path, config, keep_status):
    folder = save_run(tmp_path, config)
    (folder / 'protocol.json').unlink()
    if not keep_status:
        (folder / 'status.json').unlink()
    with pytest.raises(ValueError, match='Existing run evidence has no protocol'):
        report.summarize(tmp_path, 'a' * 64)


def test_read_hash_uses_the_exact_parsed_bytes(tmp_path):
    path = tmp_path / 'record.json'
    path.write_bytes(b'{"value": 1}\n')
    value, digest = report.read_hashed(path)
    assert value == {'value': 1}
    assert digest == report.sha(path)
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError, match='Nonfinite'):
        report.read_hashed(path)
