"""Explicit hash-bound presentation extraction; inert until root supplies inputs.

Produces only a fresh candidate directory. Never edits canonical manuscript files.
No model, trajectory, checkpoint, analysis worker or network operation is called.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import re

from render_tables import MATERIALS, POLICIES, ordered, observed_table, full_table, replace_tables

ROOT = Path(__file__).resolve().parents[3]
PLAN_SHA = 'b89a3e5341f4db983a55c20645b127636f508261409ccfbd5ad372db6b4056f8'
MAIN_SHA = '9a8a2e46f2afede6b8b919f8e97cf5a76f4d22fc0ddcc291e728f2b9d69fa02a'
PAPER_SHA = '9240888d8c1a632e68065d7a64c0fe84cb89e3a4280d74e34ca6077196d5a3be'
KNOWN_SUMMARY_PINS = {
    'Goop': {'cec653b7a0bf657c2c21402f14fd50a2dda934cf1affcd2ab4c603dbced8bbb0',
             'bb3c3f54cdeaf2b1198f49b8437dc711d80ce5e8347ab70363558ab515852978'},
    'WaterDrop': {'082bdc9c33614fea6445e3373e94eeafcfdfc516d4a27d04096f380ad7781e64'},
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(binding):
    path = Path(binding['path'])
    assert path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink()
    raw = path.read_bytes()
    assert sha(raw) == binding['sha256']
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def validate_stats(value, found=None, trail=()):
    found = {} if found is None else found
    if isinstance(value, dict):
        if {'seed_values', 'mean', 'sample_sd'} <= set(value):
            seeds = ordered(value)
            assert value.get('required_seed_pairs', 3) == 3
            assert value.get('defined_seed_pairs', sum(v is not None for v in seeds)) == sum(v is not None for v in seeds)
            found['/'.join(trail)] = value
        else:
            for key, child in value.items():
                validate_stats(child, found, trail + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_stats(child, found, trail + (str(index),))
    return found


def same_stats(left, right):
    for a, b in zip(ordered(left) + [left['mean'], left['sample_sd']],
                    ordered(right) + [right['mean'], right['sample_sd']]):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-18)


def goop_style(summary, material):
    diag, full = summary['diagnostics']['same_state_test'], summary['full_rollout']
    result = {'observed_display_exponent': -9 if material == 'Sand' else -10, 'horizon': {'Goop': 395, 'Sand': 314}[material],
              'observed': {
                  'mix_minus_base_at_base': diag['mix_minus_base']['accuracy/base/position_coordinate_mse'],
                  'mix_minus_base_at_random25': diag['mix_minus_base']['accuracy/random25/position_coordinate_mse'],
                  'risk_minus_random_base_training': diag['previous_observed_risk_minus_random_position_mse']['base'],
                  'risk_minus_random_mixed_training': diag['previous_observed_risk_minus_random_position_mse']['mix'],
                  'risk_minus_random_training_interaction': diag['risk_minus_random_mix_minus_base_interaction']},
              'full': {}}
    assert set(full['mix_minus_base']['mean_rollout_mse']) == set(POLICIES)
    for policy in POLICIES:
        failures = {arm: sum(row['arm'] == arm and row['policy'] == policy for row in full['failed_accepted_prefixes'])
                    for arm in ('base', 'mix')}
        missing = {}
        for arm in ('base', 'mix'):
            invocations = [row for row in full['runtime'] if row['arm'] == arm]
            assert sorted(row['seed'] for row in invocations) == [0, 1, 2]
            committed = sum(row['per_policy'][policy]['committed_cases'] for row in invocations)
            assert 0 <= failures[arm] <= committed <= 90
            missing[arm] = 90 - committed
        result['full'][policy] = {'state': 'predeclared', 'required_per_arm': 90,
            'training_effect': full['mix_minus_base']['mean_rollout_mse'][policy],
            'absolute': {arm: full['absolute']['mean_rollout_mse'][arm][policy] for arm in ('base', 'mix')},
            'unsuccessful': failures, 'not_completed': missing}
    return result


def waterdrop(summary):
    assert summary['state'] == 'complete' and summary['source_and_input_reverified_after_analysis'] is True
    assert summary['coverage'] == {'endpoints': 6, 'jobs': 12, 'observed_frames': 2550,
                                   'observed_policy_slots': 12750, 'autonomous_outcomes': 810}
    assert all(job['failed_records'] == 0 for job in summary['jobs'])
    diag = summary['populations']['observed_test']['metrics']['position_coordinate_mse']['paired']
    full = summary['populations']['autonomous_test']['metrics']['mean_rollout_mse']
    result = {'observed_display_exponent': -10, 'horizon': 995, 'observed': {}, 'full': {}}
    for key, source in zip(('mix_minus_base_at_base', 'mix_minus_base_at_random25',
                           'risk_minus_random_base_training', 'risk_minus_random_mixed_training',
                           'risk_minus_random_training_interaction'),
                          ('mix_minus_base__base', 'mix_minus_base__random25', 'base__risk_minus_random',
                           'mix__risk_minus_random', 'risk_minus_random_interaction')):
        result['observed'][key] = diag[source]
    assert set(full['absolute']['base']) == set(POLICIES[:-1])
    for policy in POLICIES:
        if policy == 'relative-velocity-RMS25':
            result['full'][policy] = {'state': 'not_predeclared'}
        else:
            result['full'][policy] = {'state': 'predeclared', 'required_per_arm': 81,
                'training_effect': full['paired']['mix_minus_base__' + policy],
                'absolute': {arm: full['absolute'][arm][policy] for arm in ('base', 'mix')},
                'unsuccessful': {'base': 0, 'mix': 0}, 'not_completed': {'base': 0, 'mix': 0}}
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--inputs', type=Path); parser.add_argument('--inputs-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'inert_presentation_preparation', 'scientific_products_read': False}))
        return
    assert args.inputs and args.output and not args.output.exists()
    supplied = load({'path': str(args.inputs), 'sha256': args.inputs_sha256})
    assert supplied['schema'] == 'root_admitted_cross_material_presentation_inputs_v1'
    assert set(supplied['materials']) == set(MATERIALS)
    assert supplied['sand_numerical_admission_complete'] is True
    sources = {}
    for material in MATERIALS:
        binding = supplied['materials'][material]['summary']
        if material in KNOWN_SUMMARY_PINS:
            assert binding['sha256'] in KNOWN_SUMMARY_PINS[material]
        sources[material] = load(binding)
    sand = sources['Sand']
    assert (sand['schema'], sand['status']) == ('adaptgns_sand_graph_support_paired_scalar_summary_scoped_v1', 'fixed_scalar_aggregation_complete')
    paired = load(supplied['materials']['Sand']['paired_audit'])
    assert (paired['schema'], paired['status'], paired['audit_revision']) == ('sand_paired_saved_array_audit_v1', 'passed_supported_checks', 2)
    assert paired['summary_sha256'] == supplied['materials']['Sand']['summary']['sha256']
    assert paired['cohort_sha256'] == sand['cohort_sha256']
    admission = load(supplied['materials']['Sand']['admission'])
    mapped = {material: (waterdrop(sources[material]) if material == 'WaterDrop' else goop_style(sources[material], material)) for material in MATERIALS}
    all_stats = {material: validate_stats(sources[material]) for material in MATERIALS}
    paired_stats = validate_stats(paired['verified_paired_scalars'])
    assert all(key in all_stats['Sand'] for key in paired_stats)
    for key, value in paired_stats.items():
        expected = all_stats['Sand'][key]
        # The admitted independent audit already verifies all scalar families;
        # retain its exact reconstruction alongside the presentation inputs.
        same_stats(value, expected)
    main_path = ROOT / 'work/conference_experiments_main.tex'
    original = main_path.read_text()
    assert sha(original.encode()) == MAIN_SHA
    assert sha((ROOT / 'outputs/revised_manuscript.tex').read_bytes()) == PAPER_SHA
    plan_path = ROOT / 'work/conference_presentation_20261006/cross_material_presentation_plan_before_sand_admission_statistics_v1.md'
    assert sha(plan_path.read_bytes()) == PLAN_SHA
    observed, full = observed_table(mapped), full_table(mapped)
    replacement = replace_tables(original, observed, full)
    figure_pattern = r'\\begin\{figure\*\}\[t\].*?\\end\{figure\*\}'
    assert re.findall(figure_pattern, original, re.S) == re.findall(figure_pattern, replacement, re.S)
    args.output.mkdir()
    def write(name, value):
        with (args.output / name).open('x') as stream:
            if isinstance(value, str):
                stream.write(value)
            else:
                json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    write('main_observed_table.tex', observed)
    write('main_full_horizon_table.tex', full)
    write('main_tables_only_candidate.tex', replacement)
    write('printed_statistic_map.json', mapped)
    write('all_scalar_statistic_map.json', all_stats)
    write('sand_admitted_companion.json', {'summary': sand, 'paired_audit': paired, 'admission': admission})
    write('preflight.json', {'status': 'table_candidate_only_interpretation_and_native_layout_pending',
        'root_supplied_inputs': supplied, 'root_supplied_inputs_sha256': args.inputs_sha256,
        'frozen_plan_sha256': PLAN_SHA, 'canonical_main_sha256': MAIN_SHA,
        'manuscript_sha256': PAPER_SHA, 'scalar_statistic_counts': {m: len(v) for m, v in all_stats.items()},
        'all_fixed_rows_and_policy_columns_retained': True, 'goop_figure_byte_unchanged': True,
        'title_abstract_or_canonical_manuscript_edited': False, 'new_plot_or_experiment': False})
    # Recheck supplied numerical bytes before exposing the candidate to root.
    for material in MATERIALS:
        binding = supplied['materials'][material]['summary']
        assert sha(Path(binding['path']).read_bytes()) == binding['sha256']
    for name in ('paired_audit', 'admission'):
        binding = supplied['materials']['Sand'][name]
        assert sha(Path(binding['path']).read_bytes()) == binding['sha256']
    print(json.dumps({'status': 'table_candidate_prepared', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
