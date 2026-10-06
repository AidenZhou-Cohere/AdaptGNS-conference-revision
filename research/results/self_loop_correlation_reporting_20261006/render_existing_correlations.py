"""Report already-computed bridge correlations; read no raw arrays or models."""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import argparse
_parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
_parser.add_argument('--output-dir', type=Path, required=True)
_args = _parser.parse_args()
ROOT = Path(__file__).resolve().parents[3]
OUT = _args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
PUB = ROOT / 'research/results/graph_convention_bridge_20261005'
PINS = {
    'PUBLICATION_MANIFEST.json': 'c1df28fb661a945bfd3b69ffcb8c69fbaa9047d14112e44fb2af69535c69736e',
    'PUBLICATION_AUDIT.json': 'cecf70dae18b5bb6c6ed42dc80906ec6362a68e3c316d9483037175aef5d47f0',
    'graph_bridge_aggregate_audit.json': '283fca2d9a529f1c3a62702999ad4da582e28fccf9ba12051886cf71b037a24c',
    'graph_convention_bridge.json.gz': '97b64ce8835c7734132200d292383f071610e6faa6b8f026760a7562563ebea5',
    'results.json': '9a179e343adafe87fcc7769c183eb6789736ecaf0e5e88fd00f444e0f96568ff',
}
SOURCE_PINS = {
    'research/summarize_graph_convention_bridge.py': '67558f70cc3774eca5134b8d900f7312b2f2091b01e1870ab9954915b56855c9',
    'research/graph_convention_bridge.py': '2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d',
    'research/full_same_state.py': 'ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592',
}
INFLATED_SHA = '1623cab0517dd4a20ceb668e322aa98ff064a6dc4e54b62b63d46d7c05ff27d5'
RULES = ('previous-observed-base-risk25', 'current-base-risk25')
LABELS = ('base_residual', 'own_sparse_benefit')
OBJECTIVES = ('faithful', 'nll')
SPLITS = ('valid', 'test')
checks = []


def check(condition, name):
    if not condition:
        raise ValueError(name)
    checks.append(name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs():
    for name, expected in PINS.items():
        check(sha(PUB / name) == expected, 'public hash: ' + name)
    for name, expected in SOURCE_PINS.items():
        check(sha(ROOT / name) == expected, 'source hash: ' + name)


def complete_mean(values):
    check(all(v is None or math.isfinite(v) for v in values), 'finite scalar or explicit null')
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def equal_scalar(actual, expected, name):
    check((actual is None and expected is None) or
          (actual is not None and expected is not None and
           math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14)), name)


verify_inputs()
manifest = json.loads((PUB / 'PUBLICATION_MANIFEST.json').read_text())
publication_audit = json.loads((PUB / 'PUBLICATION_AUDIT.json').read_text())
aggregate_audit = json.loads((PUB / 'graph_bridge_aggregate_audit.json').read_text())
check(publication_audit['passed'] is True and publication_audit['manifest_sha256'] == PINS['PUBLICATION_MANIFEST.json'],
      'publication audit binds exact manifest')
for name in ('graph_convention_bridge.json.gz', 'results.json', 'graph_bridge_aggregate_audit.json'):
    check(manifest['files'][name]['sha256'] == PINS[name], 'manifest binds ' + name)
raw = gzip.decompress((PUB / 'graph_convention_bridge.json.gz').read_bytes())
check(hashlib.sha256(raw).hexdigest() == INFLATED_SHA, 'inflated original summary hash')
check(manifest['files']['graph_convention_bridge.json.gz']['source_sha256'] == INFLATED_SHA,
      'manifest binds inflated original summary')
check(aggregate_audit['passed'] is True and aggregate_audit['summary_sha256'] == INFLATED_SHA,
      'independent scalar audit binds original summary')
summary = json.loads(raw)
compact = json.loads((PUB / 'results.json').read_text())
check(summary['audit']['passed'] is True, 'strict source/array audit passed')
check(summary['summary_source_sha256'] == SOURCE_PINS['research/summarize_graph_convention_bridge.py'],
      'strict summarizer source unchanged')
check(summary['groups'] == compact['groups'], 'compact groups equal original summary exactly')
runs = {(r['objective'], r['seed']): r for r in summary['runs']}
check(len(summary['runs']) == len(runs) == 6 and
      set(runs) == {(o, s) for o in OBJECTIVES for s in range(3)}, 'complete six-model cohort')

rows = []
coverage = []
for objective in OBJECTIVES:
    for split in SPLITS:
        expected_count = 128 if split == 'valid' else 297
        expected_sources = set(range(30)) if split == 'valid' else set(range(3, 30))
        for seed in range(3):
            run = runs[objective, seed]
            frames = [r for r in run['frames'] if r['split'] == split]
            expected = [r for r in run['source_protocol']['expected_frames'] if r['split'] == split]
            keys = lambda records: {(r['source_index'], r['target_frame'], r['trajectory_id']) for r in records}
            check(len(frames) == len(keys(frames)) == len(expected) == expected_count and keys(frames) == keys(expected),
                  f'{objective}/{split}/{seed}: full fixed history schedule')
            check({r['source_index'] for r in frames} == expected_sources, f'{objective}/{split}/{seed}: sources')
            check(all(r['status'] == 'complete' for r in frames), f'{objective}/{split}/{seed}: all saved frames complete')
            coverage.append(dict(objective=objective, split=split, seed=seed, frames=len(frames),
                                 trajectories=len(expected_sources)))
        for rule in RULES:
            for label in LABELS:
                metric = f'risk_correlation__{rule}__loops1__{label}'
                saved = summary['groups'][objective][split]['measures'][metric]
                seed_details = []
                for seed in range(3):
                    run = runs[objective, seed]
                    frames = [r for r in run['frames'] if r['split'] == split]
                    grouped = defaultdict(list)
                    for frame in frames:
                        value = frame['measures'][metric]
                        check(value is None or -1 <= value <= 1, 'correlation range')
                        grouped[frame['trajectory_id']].append(value)
                    trajectory_means = {t: complete_mean(v) for t, v in sorted(grouped.items())}
                    value = complete_mean(list(trajectory_means.values()))
                    original_split = run['splits'][split]['measures'][metric]
                    equal_scalar(value, original_split['equal_trajectory_mean'], 'recomputed equal-trajectory mean')
                    equal_scalar(value, saved['seed_values'][seed], 'group seed mean corresponds')
                    for trajectory, tr_value in trajectory_means.items():
                        tr_saved = original_split['trajectories'][trajectory]
                        equal_scalar(tr_value, tr_saved['value'], 'trajectory mean corresponds')
                        check(tr_saved['required_frames'] == len(grouped[trajectory]) and
                              tr_saved['undefined_frames'] == sum(v is None for v in grouped[trajectory]),
                              'trajectory required/undefined frame counts')
                    seed_details.append(dict(seed=seed, value=value,
                        undefined_frames=sum(frame['measures'][metric] is None for frame in frames),
                        trajectory_means=trajectory_means))
                values = [r['value'] for r in seed_details]
                equal_scalar(complete_mean(values), saved['mean'], 'three-seed mean corresponds')
                sd = statistics.stdev(values) if all(v is not None for v in values) else None
                equal_scalar(sd, saved['sample_sd'], 'three-seed sample SD corresponds')
                check(saved['required_seeds'] == 3 and saved['defined_seeds'] == sum(v is not None for v in values),
                      'defined seed count corresponds')
                rows.append(dict(objective=objective, split=split, rule=rule, label=label, metric=metric,
                                 saved_group=saved, seeds=seed_details))
check(sum(r['frames'] for r in coverage) == 2550, 'all 2,550 observed histories covered')
verify_inputs()

result = dict(schema='adaptgns_existing_graph_bridge_correlation_reporting_v1',
    generated_utc=datetime.now(timezone.utc).isoformat(),
    scope='Reporting of already audited loops-on scalar correlations; no raw-array analysis or inference',
    original_summary_sha256=INFLATED_SHA, source_pins=SOURCE_PINS, public_input_pins=PINS,
    generator_sha256=sha(Path(__file__)),
    definitions={
        'score': 'selection_score__{rule}_uncapped_loops1; previous observed base or current observed base, as named',
        'residual': 'sum over 2 coordinates of normalized_residual__base_uncapped_loops1 squared',
        'benefit': 'base normalized vector squared error minus own selected-graph normalized vector squared error; signed',
        'normalization': 'position prediction residual divided coordinatewise by saved acceleration_std; decoder-equivalent normalized acceleration error',
        'correlation': 'within-frame particle Spearman using average ranks, no mask; dimensionless',
        'aggregation': 'equal frame means within trajectory, equal trajectory means within seed, then mean and sample SD of exactly three seeds',
        'undefined': 'fewer than two particles or constant rank vector gives null; required null propagates, no available-case averaging',
        'graph': 'self-loops on, uncapped selected graph globally ordered; native cap inactive on measured histories; distinct from native prefix-plus-suffix autonomous evaluator',
        'interpretation': 'whole selected graph action, not marginal edge utility; not a calibration guarantee, causal concentration result or independent confirmation',
    },
    coverage=coverage, rows=rows, check_count=len(checks), checks=checks,
    new_inference=False, raw_particle_arrays_read=False)
(OUT / 'correlations.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

def fmt(value, signed=False):
    return 'undefined' if value is None else f'{value:+.6f}' if signed else f'{value:.6f}'

md = ['# Existing self-loop bridge correlations', '',
      'These are already-audited results, newly extracted for reporting. All 2,550 histories remain represented. '
      'Values are dimensionless within-frame particle Spearman correlations, averaged equally within trajectory '
      'and then equally across trajectories; the final mean and sample SD use three training seeds. '
      'Own benefit is signed base-minus-selected-action normalized vector squared error. No new inference or raw-array calculation occurred.', '',
      '| Objective | Split | Risk rule | Label | Seed 0 | Seed 1 | Seed 2 | Mean +/- sample SD | Undefined frames |',
      '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
    stats = r['saved_group']
    cells = [r['objective'], r['split'], 'Previous' if r['rule'] == RULES[0] else 'Current', r['label']]
    cells += [fmt(v, True) for v in stats['seed_values']]
    cells += [fmt(stats['mean'], True) + ' +/- ' + fmt(stats['sample_sd']),
              str(sum(s['undefined_frames'] for s in r['seeds']))]
    md.append('| ' + ' | '.join(cells) + ' |')
md += ['', f"Verification: {len(checks):,} source/summary/coverage/scalar correspondence checks passed. "
       'The counts measure bookkeeping consistency, not scientific replication. '
       'The original strict array audit and separate scalar audit remain the computational evidence.', '',
       'The bridge globally orders each selected graph. These correlations do not measure the autonomous '
       'cached-own-graph policy, and the inspected test histories do not supply independent confirmation.', '']
(OUT / 'table.md').write_text('\n'.join(md))
tex = [r'\begin{table}[ht]', r'\centering\small',
       r'\caption{Previously computed graph-bridge correlations with self-loops on. Dimensionless particle Spearman correlations are averaged over frames within trajectory, trajectories within seed, then three seeds (mean $\pm$ sample SD). Benefit is signed base-minus-own-selected-action normalized vector squared error. Both risk rules use observed histories; these are exploratory results on inspected data.}',
       r'\label{tab:graph-bridge-risk-benefit}', r'\begin{tabular}{lllrr}', r'\toprule',
       r'Split & Objective & Score & Base residual & Own sparse benefit\\', r'\midrule']
for split in SPLITS:
    for objective in OBJECTIVES:
        for rule in RULES:
            selected = [next(r for r in rows if (r['split'], r['objective'], r['rule'], r['label']) ==
                             (split, objective, rule, label)) for label in LABELS]
            cells = ['Validation' if split == 'valid' else 'Test', 'Faithful' if objective == 'faithful' else 'NLL',
                     'Previous' if rule == RULES[0] else 'Current']
            for r in selected:
                g = r['saved_group']
                cells.append('undefined' if g['mean'] is None else
                             f"${g['mean']:.3f}\\pm{g['sample_sd']:.3f}$")
            tex.append('&'.join(cells) + r'\\')
tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
(OUT / 'proposed_appendix_table.tex').write_text('\n'.join(tex))
print('\n'.join(md))
