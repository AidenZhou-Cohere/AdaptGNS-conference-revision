"""Editorial selection from current paper tables; no result reaggregation."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
snapshot_path = HERE / 'source_table_snapshot.json'
snapshot = json.loads(snapshot_path.read_text())
stats_path = ROOT / 'work/goop3d_completion_20261007/report_addendum_recipe_v1/displayed_observed_statistics.json'
stats = json.loads(stats_path.read_text())
fidelity = {'source_body_sha256': snapshot['source_body_sha256'],
            'source_table_snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            'prepared_d3_objects_sha256': hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            'selected_table_cells': []}


def source_table(label):
    text = snapshot['table_blocks'][label]
    rows = []
    for line in text.splitlines():
        if re.match(r'^(base|mix)\s*&', line):
            cells = [c.strip() for c in line.removesuffix('\\\\').split('&')]
            rows.append(cells)
    return text, rows


def policy(label):
    return label.lower().replace('previous ', '').replace('cached ', '').replace(' ', '')


def selected(label, columns):
    raw, rows = source_table(label)
    result = {}
    for cells in rows:
        selected_cells = [cells[i] for i in columns]
        key = (cells[0], policy(cells[1]))
        assert key not in result
        result[key] = selected_cells
        fidelity['selected_table_cells'].append({'source_label': label, 'source_row': cells,
                                                  'column_indices': columns, 'selected_cells': selected_cells})
    return result


def table(caption, labels, headers, rows, spec):
    return '\n'.join([
        r'\begin{table}[!ht]', r'\centering\small',
        r'\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}',
        '\\caption{' + caption + '}', *['\\label{' + x + '}' for x in labels],
        '\\begin{tabular}{' + spec + '}', r'\toprule',
        ' & '.join(headers) + r'\\', r'\midrule',
        *[' & '.join(row) + r'\\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


policies = ['base', 'dense', 'random25', 'speed25', 'risk25', 'rms25']
names = {'base': 'Base', 'dense': 'Dense', 'random25': 'Random25',
         'speed25': 'Speed25', 'risk25': 'Risk25', 'rms25': 'RMS25'}
arms = {'base': 'Base-only', 'mix': 'Mixed'}
go = selected('tab:goop-all-observed', [2])
gr = selected('tab:goop-all-rollouts', [2, 4])
wo = selected('tab:waterdrop110k-observed', [2, 3])
wr = selected('tab:waterdrop110k-rollout', [2])
so = selected('tab:sand-all-observed', [2, 3])
sr = selected('tab:sand-all-rollouts', [2])
tables = {}
tables['goop'] = table(
    r'Goop absolute errors for all policies and both training arms. Observed-test position-coordinate MSE uses $10^{-8}$ units; H395 averages all forecasts in coordinate-squared units. Each observed row has 150 histories per model, and each rollout row requires 90 trajectories. A guard failure leaves its full-horizon three-seed mean undefined. Risk25 uses preceding-observed-base scores in the observed column and its own cached selected-graph scores in rollouts.',
    ['tab:goop-all-observed', 'tab:goop-all-rollouts'],
    ['Training', 'Policy', r'Observed test ($10^{-8}$)', 'H395', 'Failed/90'],
    [[arms[a], names[p], *go[a, p], *gr[a, p]] for a in arms for p in policies], 'llrrr')
for material, observed, rollout, scale, horizon, count in (
        ('waterdrop', wo, wr, '-9', '995', '81'), ('sand', so, sr, '-8', '314', '90')):
    present = [p for p in policies if ('base', p) in observed]
    labels = ['tab:waterdrop110k-observed', 'tab:waterdrop110k-rollout'] if material == 'waterdrop' else ['tab:sand-all-observed', 'tab:sand-all-rollouts']
    title = 'WaterDrop continuation' if material == 'waterdrop' else 'Sand'
    tables[material] = table(
        title + r' absolute position-coordinate errors. Observed validation/test columns use $10^{' + scale + r'}$ units; H' + horizon + r' averages every autonomous forecast in coordinate-squared units. Every rollout row contains all ' + count + r' required trajectories with no failures. Means and sample SDs retain all three seeds. The risk score uses the preceding observed base graph for observed histories and its own selected-graph cache for rollouts.' + (' RMS25 was not a declared WaterDrop policy.' if material == 'waterdrop' else ''),
        labels, ['Training', 'Policy', r'Obs. validation ($10^{' + scale + r'}$)', r'Obs. test ($10^{' + scale + r'}$)', 'H' + horizon],
        [[arms[a], names[p], *observed[a, p], *rollout[a, p]] for a in arms for p in present], 'llrrr')

costs = {}
for material, label, columns, truth, scope in (
    ('goop', 'tab:goop-all-cost', [2, 4, 6], r'$0.0632\%$ outside; $0.000722$ excursion',
     'Time pools committed cases, including guard attempts and failed prefixes; no across-seed SD was recorded for this pooled quantity. Full-horizon geometry remains undefined for failed groups.'),
    ('waterdrop', 'tab:waterdrop110k-cost', [2, 3, 4], r'$2.678\%$ outside; $0.00782$ excursion',
     'Time is recorded policy wall time per trajectory, before the separately measured parity overhead.'),
    ('sand', 'tab:sand-all-cost', [2, 3, 4], r'$0.061825\%$ outside; $0.00079537$ excursion',
     'Time is committed-call time per trajectory, including native parity.')):
    values = selected(label, columns)
    present = [p for p in policies if ('base', p) in values]
    title = 'WaterDrop' if material == 'waterdrop' else material.title()
    costs[material] = table(title + r' full-horizon physical diagnostics and descriptive cost. For complete groups, matching truth is ' + truth + '. ' + scope,
        [label], ['Training', 'Policy', r'Outside ($\%$)', 'Excursion', 'Seconds/case'],
        [[arms[a], names[p], *values[a, p]] for a in arms for p in present], 'llrrr')
tables.update({name + '_cost': content for name, content in costs.items()})

fraw, unused_rows = source_table('tab:goop-all-failures')
failure_tabular = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', fraw, re.S).group()
tables['failures'] = '\n'.join([r'\begin{table}[!ht]\centering\small',
    r'\caption{All three Goop guard failures, each in mixed-training seed 2. The candidate-pair limit is 100,000. Accepted prefixes exclude the rejected forecast and do not replace full-H395 statistics.}',
    r'\label{tab:goop-all-failures}', failure_tabular, r'\end{table}', ''])


def d3_cell(key, signs=False):
    entry = stats[key]
    record = entry['statistic']
    text = f"${record['mean']*1e9:+.3f}\\pm{record['sample_sd']*1e9:.3f}$"
    if signs:
        text += ' [' + '/'.join('+' if record['seed_values'][str(i)] > 0 else '-' if record['seed_values'][str(i)] < 0 else '0' for i in range(3)) + ']'
    fidelity.setdefault('d3_prepared_transcriptions', []).append({'key': key, 'statistic': record, 'tex': text})
    return text

d3policies = ['base', 'dense', 'random25', 'speed25', 'previous-observed-base-risk25', 'relative-velocity-RMS25']
tables['d3_training'] = table(
    r'Goop3D 25k observed training effects, mixed minus base-only, for every policy ($10^{-9}$ position-coordinate MSE). Negative favors mixed training. Brackets preserve the signs of seeds 0/1/2; mean $\pm$ sample SD describes all three seeds.',
    ['tab:goop3d25k-observed-position-training'], ['Policy', 'Validation', 'Test'],
    [[names[p], d3_cell('same_state_valid/mix_minus_base/' + q, True), d3_cell('same_state_test/mix_minus_base/' + q, True)]
     for p, q in zip(policies, d3policies)], 'lrr')
tables['d3_gap'] = table(
    r'Goop3D 25k observed risk-minus-random gaps ($10^{-9}$ position-coordinate MSE). Negative within-arm gaps favor risk. Change in gap is mixed minus base-only; it need not imply a negative within-arm gap.',
    ['tab:goop3d25k-observed-position-contrasts'], ['Training/contrast', 'Validation', 'Test'],
    [[label, d3_cell('same_state_valid/' + key), d3_cell('same_state_test/' + key)]
     for label, key in [('Base-only', 'risk_minus_random/base'), ('Mixed', 'risk_minus_random/mix'), ('Change in gap', 'risk_interaction')]], 'lrr')

template = (HERE / 'curated_appendix_template.tex').read_text()
for name, content in tables.items():
    marker = '% TABLE_' + name.upper()
    pattern = re.compile(r'^' + re.escape(marker) + r'$', re.M)
    assert len(pattern.findall(template)) == 1, marker
    template = pattern.sub(lambda match: content, template)
assert '% TABLE_' not in template
(HERE / 'curated_appendix.tex').write_text(template)
fidelity['candidate_sha256'] = hashlib.sha256(template.encode()).hexdigest()
fidelity['scientific_values_reaggregated'] = False
fidelity['row_selection'] = 'Every declared policy and both arms retained for each primary material; selected columns copied verbatim. Goop failure table copied verbatim. D3 only formats the previously reviewed 18 objects.'
(HERE / 'table_transcription.json').write_text(json.dumps(fidelity, indent=2, sort_keys=True) + '\n')
print(json.dumps({'candidate_sha256': fidelity['candidate_sha256'], 'primary_policy_rows': len(gr)+len(wr)+len(sr), 'prepared_d3_cells':len(fidelity['d3_prepared_transcriptions'])}))
