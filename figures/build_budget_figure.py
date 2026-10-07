"""Render all published observed-test policies/seeds at their budget classes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from picture import Picture, COLORS

DATA_SHA = 'a054ab4d130f8e04e9013d57f1cb7f5ecff68108f397a06f8f63656e7f1c008f'
COLORS.update(purple='#866196')
MATERIALS = ('Goop', 'WaterDrop', 'Sand')
POLICIES = ('base', 'random25', 'speed25', 'previous-observed-base-risk25', 'relative-velocity-RMS25', 'dense')
STYLE = {'base': ('N', 'ink'), 'dense': ('D', 'native'), 'random25': ('R', 'teal'),
         'speed25': ('S', 'blue'), 'previous-observed-base-risk25': ('Q', 'rust'),
         'relative-velocity-RMS25': ('V', 'purple')}
CONFIG = {'Goop': (1e8, (1.10, 1.63), (1.2, 1.4, 1.6)),
          'WaterDrop': (1e9, (3.2, 4.8), (3.4, 4.0, 4.6)),
          'Sand': (1e8, (2.7, 7.5), (3, 5, 7))}


def draw(data, output):
    assert data['rows_required'] == 34 and data['seed_values_required'] == 102
    assert len(data['rows']) == 34 and data['autonomous_data_included'] is False
    rows = {(r['material'], r['training_arm'], r['policy']): r for r in data['rows']}
    assert len(rows) == 34
    p = Picture(470, 280)
    p.text(8, 270, 'Accuracy with fewer added messages', size=12, bold=True)
    points, means, undefined = [], [], []
    for material, left in zip(MATERIALS, (32, 185, 338)):
        scale, ylim, ticks = CONFIG[material]
        policies = [policy for policy in POLICIES if (material, 'base', policy) in rows]
        positions = ((9, 30, 50, 70, 90, 113) if len(policies) == 6 else (9, 35, 60, 85, 113))
        xs = dict(zip(policies, [left + x for x in positions]))
        p.text(left + 61, 252, material, size=10.5, bold=True, align='center')
        context = '100k from scratch' if material != 'WaterDrop' else '100k to 110k paired'
        p.text(left + 61, 239, context, '100k from scratch' if material != 'WaterDrop' else r'100k $\to$ 110k paired',
               size=7.1, color='muted', align='center')
        exponent = '-9' if material == 'WaterDrop' else '-8'
        p.text(left, 227, 'Position MSE ×10' + ('⁻⁹' if material == 'WaterDrop' else '⁻⁸'), r'Position MSE $\times10^{' + exponent + '}$', size=7.1, color='muted')
        for arm, bottom, top in (('base', 152, 220), ('mix', 64, 132)):
            def y(value):
                scaled = value * scale
                assert ylim[0] <= scaled <= ylim[1], (material, arm, value)
                return bottom + (scaled - ylim[0]) / (ylim[1] - ylim[0]) * (top - bottom)
            for tick in ticks:
                yy = bottom + (tick - ylim[0]) / (ylim[1] - ylim[0]) * (top - bottom)
                p.line(left, yy, left + 122, yy, color='grid', width=.4)
                p.text(left - 4, yy, str(tick), size=6.8, color='muted', align='right')
            p.line(left, bottom, left, top, color='zero', width=.65)
            native_mean = rows[material, arm, 'base']['statistic_unchanged']['mean']
            if native_mean is not None:
                p.line(left, y(native_mean), left + 122, y(native_mean), color='zero', width=.8, dashed=True)
            p.text(left + 4, top - 6, 'Base-only training' if arm == 'base' else 'Mixed training', size=7.0, bold=True)
            for policy in policies:
                row = rows[material, arm, policy]
                x, color = xs[policy], STYLE[policy][1]
                values = row['ordered_seed_values']
                original = row['statistic_unchanged']['seed_values']
                assert values == ([original[str(i)] for i in range(3)] if isinstance(original, dict) else original)
                for seed, value in enumerate(values):
                    xx = x + (seed - 1) * 2.8
                    if value is None:
                        p.cross(xx, bottom + 4, r=1.6, color=color)
                        undefined.append({'material': material, 'arm': arm, 'policy': policy, 'seed': seed})
                    else:
                        yy = y(value)
                        if seed == 0:
                            p.circle(xx, yy, r=1.35, color=color)
                        elif seed == 1:
                            p.circle(xx, yy, r=1.35, color=color, filled=False, width=.65)
                        else:
                            p.cross(xx, yy, r=1.4, color=color, width=.65)
                        points.append({'material': material, 'training_arm': arm, 'policy': policy, 'seed': seed,
                                       'raw_mse': value, 'display_multiplier': scale, 'x': xx, 'y': yy,
                                       'source_key': row['source_key']})
                mean = row['statistic_unchanged']['mean']
                if mean is not None:
                    p.line(x - 4.4, y(mean), x + 4.4, y(mean), color=color, width=1.15)
                    means.append({'material': material, 'training_arm': arm, 'policy': policy, 'raw_mse': mean,
                                  'source_key': row['source_key']})
            if arm == 'mix':
                for policy in policies:
                    p.text(xs[policy], 54, STYLE[policy][0], size=7.5, bold=True, align='center', color=STYLE[policy][1])
                sparse = [xs[policy] for policy in policies if policy not in ('base', 'dense')]
                for x in (min(sparse) - 5, max(sparse) + 5):
                    p.line(x, 43, x, 47, color='zero', width=.6)
                p.line(min(sparse) - 5, 43, max(sparse) + 5, 43, color='zero', width=.6)
                p.text(xs['base'], 39, '0', size=7.2, align='center', color='muted')
                p.text((min(sparse) + max(sparse)) / 2, 35, '≤25%', r'$\le25\%$', size=7.2, align='center', color='muted')
                p.text(xs['dense'], 39, '100%', r'$100\%$', size=7.2, align='center', color='muted')
    p.text(235, 23, 'Budget classes: fraction of annulus messages added; native messages retained',
           size=7.1, align='center', color='muted')
    p.circle(15, 10, r=1.5);p.text(20, 10, 'seed 0', size=6.8)
    p.circle(63, 10, r=1.5, filled=False);p.text(68, 10, 'seed 1', size=6.8)
    p.cross(111, 10, r=1.5);p.text(116, 10, 'seed 2', size=6.8)
    p.line(163, 10, 173, 10, width=1.1);p.text(179, 10, 'three-seed mean', size=6.8)
    p.line(284, 10, 303, 10, color='zero', dashed=True);p.text(309, 10, 'native mean; lower MSE is better', size=6.8, color='muted')
    caption = (r'\textbf{Smaller expansions can outperform dense connectivity.} '
               r'Common observed test histories, all declared policies and both training arms. '
               r'N: native; R: random; S: speed; Q: preceding-observed-base risk; V: relative-velocity RMS; D: dense. '
               r'WaterDrop omits V. Symbols show seeds 0--2, short bars their mean, and dashed lines the native mean. '
               r'Brackets group policies by categorical fractions of added messages, with native messages retained; '
               r'this is not an empirical total-edge axis. Goop and Sand use 150 histories from 30 sources per model; '
               r'WaterDrop uses 297 from 27.')
    assert len(points) + len(undefined) == 102 and len(means) == 34
    product = p.save(output, 'observed_accuracy_budget', caption, 'fig:observed-accuracy-budget')
    figure_data = {'schema': 'observed_accuracy_budget_figure_data_v1', 'display_data_sha256': DATA_SHA,
               'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'picture_helper_sha256': hashlib.sha256((Path(__file__).parent / 'picture.py').read_bytes()).hexdigest(),
               'dimensions_pt': [470, 280], 'seed_points': points, 'mean_marks': means, 'undefined_slots': undefined,
               'source_row_count': 34, 'expected_seed_count': 102, 'all_declared_policies_retained': True,
               'product_sha256': product['sha256']}
    (output / 'observed_accuracy_budget_data.json').write_text(json.dumps(figure_data, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    packaged_data = here.parent / 'results/cross_material/budget_display_data.json'
    default_data = packaged_data if packaged_data.is_file() else here / 'display_data.json'
    default_output = here.parent / 'generated' if packaged_data.is_file() else here
    parser.add_argument('--data', type=Path, default=default_data)
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get('REPRODUCTION_OUTPUT', default_output)))
    args = parser.parse_args()
    raw = args.data.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(json.loads(raw), args.output_dir)


if __name__ == '__main__':
    main()
