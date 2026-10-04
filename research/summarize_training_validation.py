"""Read-only, provenance-checked clean-validation curves for the fixed six models.

No model or dataset is loaded. Every scheduled validation point is retained;
these diagnostics never select a checkpoint or replace final test evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

OBJECTIVES = ('faithful', 'nll')
SEEDS = (0, 1, 2)
STEPS = tuple(range(0, 100001, 5000))
METRICS = ('coordinate_mse', 'constant_free_gaussian_nll', 'realized_vector_se',
           'predicted_vector_se', 'binned_vector_se_calibration_gap')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_hashed(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    raw = path.read_bytes()
    return json.loads(raw, parse_constant=invalid), hashlib.sha256(raw).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def close(a, b):
    return type(a) in (float, int) and math.isfinite(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def validate_config(config, objective, seed, protocol_hash):
    require(config.get('scope') == 'bounded_full_data_100k' and config.get('objective') == objective
            and type(config.get('seed')) is int and config['seed'] == seed
            and config.get('steps') == 100000 and config.get('validation_interval') == 5000
            and config.get('checkpoint_interval') == 10000 and config.get('batch_size') == 2
            and config.get('history') == 6 and config.get('noise_std') == 6.7e-4
            and config.get('architecture') == {'width': 128, 'message_passing_blocks': 10, 'mlp_layers': 2}
            and config.get('graph') == {'radius': .015, 'radius_backend': 'scipy_host', 'max_neighbors': 128,
                                        'self_loops': True, 'training': 'base only; no radius augmentation'}
            and config.get('optimizer') == {'kind': 'Adam', 'initial_lr': 1e-4, 'final_lr': 1e-5,
                                            'decay_updates': 100000, 'betas': [.9, .999], 'eps': 1e-8,
                                            'weight_decay': 0., 'gradient_clipping': None}
            and config.get('validation_noise') == 0.
            and config.get('validation_normalization') == 'training noise-adjusted scales'
            and config.get('train', {}).get('n_trajectories') == 1000
            and config.get('valid', {}).get('n_trajectories') == 30
            and config.get('runtime', {}).get('device') == 'mps'
            and config.get('runtime', {}).get('mps_fallback_environment') == '0'
            and config.get('research_protocol_sha256') == protocol_hash,
            'Validation does not belong to the fixed full-data experiment')
    frames = config.get('validation_frames', [])
    require(len(frames) == 128 and len({x['trajectory'] for x in frames}) == 30
            and len({x['id'] for x in frames}) == len(frames)
            and len({x['dataset_index'] for x in frames}) == len(frames),
            'Fixed validation schedule count or uniqueness differs')
    for frame in frames:
        require(type(frame['dataset_index']) is int and 0 <= frame['dataset_index'] < 29850
                and type(frame['target_frame']) is int and 6 <= frame['target_frame'] <= 1000
                and frame['trajectory'] in {f'valid:{i:06d}' for i in range(30)}
                and frame['id'] == f"{frame['trajectory']}:{frame['target_frame']}",
                'Invalid fixed validation frame identity')


def validate_point(point, config, step):
    """Recompute equal-trajectory and weighted-bin means from saved records."""
    require(point['run_config_sha256'] == canonical(config)
            and type(point['completed_steps']) is int and point['completed_steps'] == step,
            'Validation run hash or update identity differs')
    expected = config['validation_frames']
    rows = point['frames']
    require(len(rows) == len(expected) and len({row['id'] for row in rows}) == len(rows),
            'Validation frame count or uniqueness differs')
    for row, identity in zip(rows, expected):
        require(all(row.get(key) == value for key, value in identity.items()), 'Fixed validation frame identity differs')
        require(type(row['dynamic_particles']) is int and row['dynamic_particles'] > 0, 'Invalid particle count')
        for name in METRICS[:4]:
            value = row[name]
            require(type(value) in (int, float) and math.isfinite(value)
                    and (name == 'constant_free_gaussian_nll' or value >= 0), 'Invalid validation metric')
        require(close(row['realized_vector_se'], 2 * row['coordinate_mse']), 'Coordinate/vector risk units differ')
    trajectories = sorted({row['trajectory'] for row in rows})
    require(set(point['trajectories']) == set(trajectories), 'Trajectory summary identity differs')
    recomputed = {}
    for metric in METRICS[:4]:
        values = []
        for trajectory in trajectories:
            value = statistics.fmean(row[metric] for row in rows if row['trajectory'] == trajectory)
            require(close(point['trajectories'][trajectory][metric], value), 'Saved trajectory mean differs')
            values.append(value)
        value = statistics.fmean(values)
        require(close(point['equal_trajectory_mean'][metric], value), 'Saved equal-trajectory mean differs')
        recomputed[metric] = value
    bins = point['calibration_bins']
    require(bool(bins) and len({b['index'] for b in bins}) == len(bins)
            and all(type(b['index']) is int and 0 <= b['index'] < 10 for b in bins),
            'Missing, invalid or duplicate calibration bins')
    for b in bins:
        require(type(b['particles']) is int and b['particles'] > 0
                and type(b['weight']) in (int, float) and math.isfinite(b['weight']) and b['weight'] > 0,
                'Invalid bin mass/count')
        require(all(type(b[name]) in (int, float) and math.isfinite(b[name]) and b[name] >= 0
                    for name in ('predicted_vector_se', 'realized_vector_se')), 'Invalid bin risk')
    require(sum(b['particles'] for b in bins) == sum(row['dynamic_particles'] for row in rows),
            'Calibration-bin particle count differs')
    require(close(sum(b['weight'] for b in bins), 1.), 'Calibration-bin weights do not sum to one')
    for metric in ('predicted_vector_se', 'realized_vector_se'):
        require(close(sum(b['weight'] * b[metric] for b in bins), recomputed[metric]),
                'Calibration bins use a different weighting or risk total')
    gap = sum(b['weight'] * abs(b['predicted_vector_se'] - b['realized_vector_se']) for b in bins)
    require(close(gap, point['binned_vector_se_calibration_gap']), 'Binned risk gap differs')
    recomputed['binned_vector_se_calibration_gap'] = gap
    return recomputed


def summarize(root, protocol_hash, repo=None):
    runs, common = [], set()
    for objective in OBJECTIVES:
        for seed in SEEDS:
            folder = root / f'{objective}_seed{seed}'
            row = {'objective': objective, 'seed': seed, 'state': 'not_started', 'points': []}
            runs.append(row)
            if not (folder / 'protocol.json').exists():
                require(not (folder / 'status.json').exists() and not list((folder / 'validation').glob('step-*.json')),
                        'Existing run evidence has no protocol; cannot classify it as not started')
                if folder.exists() and any(folder.iterdir()):
                    row['state'] = 'initializing_without_protocol'
                continue
            config, config_file_hash = read_hashed(folder / 'protocol.json')
            validate_config(config, objective, seed, protocol_hash)
            if repo is not None:
                expected_sources = {'research/full_training.py'} | {
                    path.relative_to(repo).as_posix() for path in (repo / 'adaptive-gns/gns').glob('*.py')}
                require(set(config.get('source_sha256', {})) == expected_sources, 'Frozen training source path set differs')
                for relative, digest in config['source_sha256'].items():
                    path = repo / relative
                    require(path.resolve().is_relative_to(repo.resolve()) and path.is_file()
                            and sha(path) == digest, 'Frozen training source differs')
            common.add(canonical({key: value for key, value in config.items() if key not in ('objective', 'seed')}))
            status, status_hash = read_hashed(folder / 'status.json') if (folder / 'status.json').exists() else (None, None)
            if status:
                require(status.get('run_config_sha256') == canonical(config), 'Status run hash differs')
                require(type(status.get('completed_steps')) is int and 0 <= status['completed_steps'] <= 100000,
                        'Invalid recorded update count')
                require(status.get('state') != 'complete' or status['completed_steps'] == 100000,
                        'Complete status has not reached fixed target')
                row.update(state=status['state'], last_recorded_updates=status['completed_steps'], error=status.get('error'))
                row['status_file_sha256'] = status_hash
            else:
                row['state'] = 'initializing_without_status'
            row['run_config_sha256'] = canonical(config)
            row['protocol_file_sha256'] = config_file_hash
            for path in sorted((folder / 'validation').glob('step-*.json')):
                step = int(path.stem.split('-')[1])
                require(step in STEPS and path.name == f'step-{step:06d}.json', 'Unscheduled or noncanonical validation update')
                record, record_hash = read_hashed(path)
                row['points'].append({'completed_steps': step, 'record_file': path.name,
                                      'record_sha256': record_hash, 'metrics': validate_point(record, config, step)})
            row['missing_scheduled_updates'] = [step for step in STEPS if step not in {p['completed_steps'] for p in row['points']}]
            require(row['state'] != 'complete' or not row['missing_scheduled_updates'],
                    'Completed run is missing scheduled validation records')
    require(len(common) <= 1, 'Common training/data/source/frame configuration differs across models')
    grouped = {}
    for objective in OBJECTIVES:
        by_seed = {row['seed']: {p['completed_steps']: p for p in row['points']}
                   for row in runs if row['objective'] == objective}
        grouped[objective] = []
        for step in STEPS:
            values = [by_seed[seed].get(step) for seed in SEEDS]
            available = sum(value is not None for value in values)
            if not available:
                continue
            grouped[objective].append({'completed_steps': step, 'available_seeds': available, 'required_seeds': 3,
                'statistics': {metric: {'seed_values': [v['metrics'][metric] if v else None for v in values],
                    'mean': statistics.fmean(v['metrics'][metric] for v in values) if available == 3 else None,
                    'sample_sd': statistics.stdev(v['metrics'][metric] for v in values) if available == 3 else None}
                    for metric in METRICS}})
    return {'schema': 1, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Fixed clean-validation learning diagnostics; never checkpoint selection or test evidence',
            'research_protocol_sha256': protocol_hash, 'source_sha256': sha(Path(__file__)),
            'current_frozen_source_files_verified': repo is not None,
            'units': 'normalized acceleration using stored noise-adjusted training scales; not position MSE or compact-pilot units',
            'aggregation': 'particles within each frame, equal frames within trajectory, equal trajectories; exactly three seeds for group mean/sample SD',
            'verification_limit': 'Checks saved frame/trajectory means, arithmetic consistency of bin counts/masses/risk totals and configuration hashes. Bin membership cannot be recovered from frame means; no checkpoint/data is loaded or predictions recomputed.',
            'runs': runs, 'three_seed_curves': grouped}


def plot(result, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    colors = {'faithful': '#246583', 'nll': '#b95839'}
    for row in result['runs']:
        if not row['points']:
            continue
        for ax, metric in zip(axes, ('coordinate_mse', 'binned_vector_se_calibration_gap')):
            ax.plot([p['completed_steps'] for p in row['points']], [p['metrics'][metric] for p in row['points']],
                    marker='o', ms=3, lw=1.3, alpha=.85, color=colors[row['objective']],
                    linestyle=('-', '--', ':')[row['seed']], label=f"{row['objective']} seed {row['seed']}")
    for ax, title in zip(axes, ('Clean acceleration coordinate MSE', 'Binned vector-risk gap')):
        ax.set(title=title, xlabel='Optimizer updates', ylabel='Normalized squared-acceleration units')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle('Interim validation — every recorded point; no checkpoint selection', fontsize=11)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix('.png'), dpi=180)
    fig.savefig(prefix.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, default=Path('research/results/full_waterdrop_100k'))
    parser.add_argument('--protocol', type=Path, default=Path(__file__).parent / 'protocols/full_waterdrop_100k.md')
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    result = summarize(args.result_dir, sha(args.protocol), Path(__file__).resolve().parents[1])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.plot:
        plot(result, args.output_prefix)
    print(json.dumps({'recorded_validation_points': sum(len(row['points']) for row in result['runs']),
                      'models_with_points': sum(bool(row['points']) for row in result['runs'])}))


if __name__ == '__main__':
    main()
