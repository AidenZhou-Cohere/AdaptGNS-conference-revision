"""Plot saved complete-seed correlations; no model calls or statistical refits."""
from pathlib import Path
import json
import os
import hashlib
from picture import Picture

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get('REPRODUCTION_OUTPUT', ROOT / 'generated'))
OUTPUT.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / 'results/controls/uncertainty_associations.json').read_text())
assert data['models'] == 6 and data['seeds'] == [0, 1, 2]

p = Picture(225, 211)
p.text(112.5, 199, 'An error signal is not a benefit signal', size=10, bold=True, align='center')
p.text(112.5, 184, 'WaterDrop · six models trained for 100k updates',
       r'WaterDrop: six models trained for 100k updates', size=7.4, align='center', color='muted')
ymin, ymax = -.12, .5
bottom, top = 44, 158
def y(value):
    assert ymin <= value <= ymax
    return bottom + (value - ymin) / (ymax - ymin) * (top - bottom)

p.text(6, 156, 'Spearman ρ', r'Spearman $\rho$', size=7.4, color='muted')
for value in (0., .2, .4):
    p.line(27, y(value), 216, y(value), color='zero' if value == 0 else 'grid',
           width=.7 if value == 0 else .4, dashed=value == 0)
    p.text(21, y(value), f'{value:.1f}', size=7.2, align='right', color='muted')

points = []
for index, row in enumerate(data['records']):
    left, right = ((44, 99), (145, 202))[index]
    p.text((left+right)/2, 169, 'Faithful' if row['objective'] == 'faithful' else 'NLL',
           size=8.3, bold=True, align='center')
    errors = row['base_error']['seed_values']
    benefits = row['own_action_benefit']['seed_values']
    assert [v['seed'] for v in errors] == [0, 1, 2] and len(benefits) == 3
    for seed, (error, benefit) in enumerate(zip(errors, benefits)):
        a = error['value']
        offset = (seed - 1) * 3.5
        p.line(left+offset, y(a), right+offset, y(benefit), color='native', width=.65)
        p.circle(left+offset, y(a), r=2.25, color='teal')
        p.circle(right+offset, y(benefit), r=2.25, color='rust')
        points.append({'objective': row['objective'], 'seed': seed,
                       'base_error': a, 'own_action_benefit': benefit})
    p.text(left, 31, 'Error', size=8, align='center', color='teal')
    p.text(right, 31, 'Benefit', size=8, align='center', color='rust')
p.text(112.5, 13, 'Each line connects the two correlations for one seed.',
       size=7.1, align='center', color='muted')

caption = (r'\textbf{An error signal is not a benefit signal.} '
           r'Each line shows one WaterDrop 100k model: the preceding observed residual score '
           r'correlates with base error but weakly with the signed improvement under its actual Risk25 graph. '
           r'This score is one observed step older than the current scores in Figure~\ref{fig:residual-times}. '
           r'All six models use 27 test trajectories and 11 observed histories per trajectory; '
           r'frame correlations average within trajectory and seed. '
           r'This diagnostic uses uncapped graphs without self-messages and is separate from '
           r'the paired 110k continuation study.')
p.save(OUTPUT, 'uncertainty_associations', caption, 'fig:uncertainty-associations')
figure = OUTPUT / 'uncertainty_associations.tex'
figure.write_text(figure.read_text().replace('figure*', 'figure'))
record_path = OUTPUT / 'uncertainty_associations_primitives.json'
record = json.loads(record_path.read_text())
record['sha256']['uncertainty_associations.tex'] = hashlib.sha256(figure.read_bytes()).hexdigest()
record['categorical_seed_offsets_pt'] = [-3.5, 0, 3.5]
record_path.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
(OUTPUT / 'uncertainty_associations_points.json').write_text(
    json.dumps({'points': points, 'statistical_recomputation': False}, sort_keys=True, indent=2) + '\n')
print('Uncertainty associations: all 12 saved correlations from six models')
