#!/usr/bin/env python3
"""Format hash-bound, fully checked D3 observed products; no scientific reruns.

Outputs reviewable fragments and complete scalar evidence only. Does not edit
the manuscript, pool materials, read autonomous products, or choose policies.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

COLLECTION_SHA = '616c612724f62d21bcb521c6163e83d9038b34fdd8012980d6ed55f4b05bed5f'
COHORT_SHA = '645343fc2a1c6ef0a82e212e351b03b1a4d081702d3c8e2187b244a62c6b9bef'
OLD_CHECKER_SHA = 'd266da3d5a437586c407aba20898374f62d459b2b9c2980232d241e4a56bf5ad'
FAMILIES = ('same_state_valid', 'same_state_test', 'clean_validation')
STAGES = ('full_rollout_valid', 'full_rollout_test', *FAMILIES)
ARMS, SEEDS = ('base', 'mix'), ('0', '1', '2')
RISK = 'previous-observed-base-risk25'
POLICIES = ('base', 'dense', 'random25', 'speed25', RISK, 'relative-velocity-RMS25')
TIMING_CASES = (*POLICIES, 'natural_base_reference')
REFERENCES = ('base', 'random25', 'speed25', 'relative-velocity-RMS25')
NAMES = dict(zip(POLICIES, ('Base', 'Dense', 'Random25', 'Speed25', 'Previous risk25', 'RMS25')))
NAMES['natural_base_reference'] = 'Natural base reference'
METRICS = ('position_coordinate_mse', 'normalized_coordinate_mse')
CLEAN = ('normalized_acceleration_coordinate_mse', 'constant_free_gaussian_nll',
         'predicted_normalized_vector_se', 'realized_normalized_vector_se')
CLEAN_NAMES = ('Normalized coordinate MSE', 'Constant-free NLL', 'Predicted vector squared error', 'Realized vector squared error')
TIMES = ('end_to_end_seconds', 'score_generation_seconds', 'score_graph_seconds',
         'score_forward_seconds', 'current_graph_and_selection_seconds', 'current_forward_seconds')
BOUNDARY = ('fraction_particles_outside', 'fraction_particles_outside_by_more_than_1e-6',
            'maximum_coordinate_excursion', 'mean_particle_maximum_excursion')
GRAPH = ('candidate_pairs', 'geometric_base_pairs', 'available_annulus_pairs', 'optional_pair_budget',
         'retained_optional_pairs', 'directed_edges', 'native_base_directed_edges', 'native_base_self_edges',
         'native_base_receivers_above_cap_before_capping', 'native_base_edges_removed_by_cap',
         'native_base_max_receiver_degree', 'native_base_asymmetric_directed_edges')
PRODUCTS = ('audit', 'summary', 'arithmetic_check', 'completion', 'retained_original_failure', 'diagnosis')
ORIGINAL_COUNTS = {'completed_required_outcome': 2899, 'not_completed_before_invocation_end': 1817, 'timed_out_current': 12}


def need(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def read_bound(path, pin):
    need(isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin), 'explicit SHA256 required')
    path = Path(path)
    need(path.is_absolute() and path == path.resolve() and path.is_file()
         and all(not p.is_symlink() for p in (path, *path.parents)), 'canonical ordinary input path required')
    need(path.stat().st_size <= 512 << 20, 'bounded saved JSON product required')
    raw = path.read_bytes()
    need(digest(raw) == pin, 'input SHA differs: ' + str(path))
    return raw


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def delta(a, b):
    return a - b if finite(a) and finite(b) else None


def stats(values):
    """Exact binary-input pairwise variance; constant large means have zero SD."""
    need(len(values) == 3, 'all three ordered seeds required')
    defined = sum(finite(v) for v in values)
    mean = sd = None
    if defined == 3:
        mean = math.fsum(values) / 3
        exact = [Fraction(v) for v in values]
        variance = sum((exact[i] - exact[j]) ** 2 for i in range(3) for j in range(i + 1, 3)) / 6
        with localcontext() as context:
            context.prec = 80
            sd = float((Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt())
    return {'seed_values': dict(zip(SEEDS, values)), 'required_seed_pairs': 3,
            'defined_seed_pairs': defined, 'mean': mean, 'sample_sd': sd}


def ordered(record):
    need(set(record) == {'seed_values', 'required_seed_pairs', 'defined_seed_pairs', 'mean', 'sample_sd'}, 'exact statistic object keys')
    need(type(record['seed_values']) is dict and set(record['seed_values']) == set(SEEDS), 'exact ordered seed identities')
    values = [record['seed_values'][s] for s in SEEDS]
    need(all(v is None or finite(v) for v in values), 'invalid statistic seed')
    expected = stats(values)
    for key in ('required_seed_pairs', 'defined_seed_pairs'):
        need(type(record[key]) is int and record[key] == expected[key], 'seed denominator differs')
    for key in ('mean', 'sample_sd'):
        if expected[key] is None:
            need(record[key] is None, 'incomplete seed family must have null summaries')
        elif expected[key] == 0:
            need(finite(record[key]) and record[key] == 0, 'zero statistic differs')
        else:
            need(finite(record[key]) and math.isclose(record[key], expected[key], rel_tol=1e-12, abs_tol=1e-300), 'statistic reduction differs')
    return values


def same_values(record, expected):
    need(ordered(record) == expected, 'paired values differ from saved absolute seed values')


def expected_keys(stage):
    if stage == 'clean_validation':
        return {'metrics/' + k for k in CLEAN}
    keys = {'accuracy/' + p + '/' + m for p in POLICIES for m in METRICS}
    benefit = ('mean_position_vector_benefit', 'mean_normalized_vector_benefit', 'positive_fraction', 'negative_fraction', 'zero_fraction')
    for p in POLICIES[1:]:
        names = benefit + (() if p == 'dense' else ('dense_sparse_sign_disagreement_fraction', 'dense_positive_sparse_nonpositive_fraction'))
        keys.update('benefit/' + p + '/' + k for k in names)
    keys.update('timing/' + p + '/' + k for p in TIMING_CASES for k in TIMES)
    keys.add('correlations/previous_risk_vs_base_error')
    keys.update('correlations/previous_risk_vs_' + p + '_benefit' for p in POLICIES[1:])
    keys.update('correlations/dense_vs_' + p + '_benefit' for p in POLICIES if p not in ('base', 'dense'))
    keys.update('boundary/' + p + '/' + k for p in TIMING_CASES for k in BOUNDARY)
    keys.update('truth_boundary/' + k for k in BOUNDARY)
    keys.update('graph/' + p + '/' + k for p in POLICIES for k in GRAPH)
    return keys


def statistic_index(summary):
    result = {}
    def walk(node, path):
        if isinstance(node, dict):
            if 'seed_values' in node:
                values = ordered(node)
                result['/'.join(path)] = {'json_path': path, 'statistic': node,
                                         'seed_signs': ['null' if v is None else '+' if v > 0 else '-' if v < 0 else '0' for v in values]}
            else:
                for key, value in node.items():
                    walk(value, path + [key])
    walk(summary['families'], ['families'])
    return result


def validate(products, pins):
    audit, summary, checked, completion = [products[k] for k in PRODUCTS[:4]]
    need(audit['schema'] == 'goop3d_observed_history_saved_audit_v1' and audit['status'] == 'passed_scoped_observed_history_checks', 'passing observed audit required')
    need(audit['dataset'] == 'Goop-3D' and audit['endpoint_updates'] == 25000, 'distinct 25k Goop3D endpoint required')
    need(summary['schema'] == 'goop3d_observed_history_summary_v1' and summary['status'] == 'passed_scoped_scalar_aggregation', 'unchanged observed summary required')
    need(checked['schema'] == 'goop3d_observed_history_arithmetic_check_v1' and checked['status'] == 'passed_independent_observed_arithmetic'
         and checked['checker_revision'] == 2 and checked['sample_sd_method'] == 'exact_rational_pairwise_variance_decimal_sqrt', 'passing exact-variance successor check required')
    need(completion['schema'] == 'goop3d_observed_cache_finalization_v2'
         and completion['status'] == 'all2568_cached_rows_merged_and_exact_variance_check_passed', 'completed cache-only finalizer required')
    for product in (audit, summary, checked, completion):
        need(product['required_all_cells'] == 4728 and product['required_observed_cells'] == 2568, 'fixed denominator differs')
        need(product['collection_sha256'] == COLLECTION_SHA, 'original collection differs')
        need(product['finalizer_revision'] == 2 and product['comparison_tolerance_unchanged'] is True
             and product['saved_row_array_audit_calls'] == 0 and product['old_checker_source_sha256'] == OLD_CHECKER_SHA, 'exact finalization lineage differs')
    for product in (audit, summary, checked):
        need(product['cohort_sha256'] == COHORT_SHA and product['phase_sha256'] is None, 'cohort/successor phase identity differs')
    need(completion['cache_rows_consumed'] == 2568 and completion['original_arrays_decoded'] == 0
         and completion['original_cache_and_failed_attempt_unchanged'] is True, 'all cached rows and immutable history required')
    need(summary['audit_sha256'] == checked['audit_sha256'] == pins['audit'] and checked['summary_sha256'] == pins['summary'], 'product linkage differs')
    need(completion['products_sha256'] == {k + '.json': pins[k] for k in ('audit', 'summary', 'arithmetic_check', 'retained_original_failure')}, 'completion product hashes differ')
    need(products['retained_original_failure']['error_type'] == 'ValueError'
         and products['retained_original_failure']['error'] == 'independent arithmetic differs', 'original failed checker evidence missing')
    diagnosis = products['diagnosis']
    need(diagnosis['status'] == 'diagnosis_only_not_scientific_admission' and diagnosis['cached_rows_verified'] == 2568
         and diagnosis['mismatch_count'] == len(diagnosis['mismatches']) == 36
         and diagnosis['rows_or_arrays_reaudited'] == 0 and diagnosis['tolerances_changed'] is False,
         'exact retained graph-SD diagnosis required')
    for mismatch in diagnosis['mismatches']:
        path = mismatch['path_components']
        need(len(path) == 6 and path[:1] == ['families'] and path[1] in FAMILIES[:2]
             and path[2] == 'absolute' and path[3] in ARMS and path[4].startswith('graph/') and path[5] == 'sample_sd'
             and mismatch['exact_rational_oracle']['all_three_input_floats_identical'] is True
             and mismatch['exact_rational_oracle']['exact_unbiased_variance_ratio'] == '0'
             and mismatch['summarizer_value'] == 0, 'diagnosis must preserve only false nonzero graph SDs')
    need(audit['all_original_accounting'] == summary['all_original_accounting'], 'original accounting differs')
    original = summary['all_original_accounting']
    grid = {(a, s, stage) for a in ARMS for s in range(3) for stage in STAGES}
    need(len(original) == 30 and {(r['arm'], r['seed'], r['stage']) for r in original} == grid, 'all 30 original stages required')
    need(sum(len(r['cells']) for r in original) == 4728
         and dict(Counter(c['state'] for r in original for c in r['cells'])) == ORIGINAL_COUNTS, 'all 4728 historical states required')
    models = audit['models']
    need(len(models) == 18 and {(m['arm'], m['seed'], m['stage']) for m in models}
         == {(a, s, stage) for a in ARMS for s in range(3) for stage in FAMILIES}, 'all 18 observed model/stages required')
    need(set(summary['families']) == set(FAMILIES), 'observed and autonomous schemas cannot be mixed')
    for stage in FAMILIES:
        family = summary['families'][stage]
        count = 128 if stage == 'clean_validation' else 150
        need(family['full_family_complete'] is True and family['required_cells'] == 6 * count, 'full observed family required')
        need(family['coverage_by_model'] == {f'{a}_seed{s}': {'completed_required_outcome': count} for a in ARMS for s in range(3)}, 'complete six-model family coverage required')
        keys = expected_keys(stage)
        need(set(family['absolute']) == set(ARMS) and set(family['mix_minus_base_training']) == keys, 'complete metric family required')
        for arm in ARMS:
            need(set(family['absolute'][arm]) == keys, 'all diagnostic/clean metric keys required')
            for seed in range(3):
                model = next(m for m in models if (m['arm'], m['seed'], m['stage']) == (arm, seed, stage))
                historical = next(r for r in original if (r['arm'], r['seed'], r['stage']) == (arm, seed, stage))
                need(len(model['cells']) == count and model['cells'] == historical['cells']
                     and model['coverage'] == {'completed_required_outcome': count}
                     and all(c['state'] == 'completed_required_outcome' for c in model['cells']), 'model/historical observed coverage differs')
                need(set(model['aggregates']) == keys, 'audit metric family differs')
                for key in keys:
                    need(family['absolute'][arm][key]['seed_values'][str(seed)] == model['aggregates'][key]['equal_trajectory_mean'], 'saved audit/summary model mean differs')
        for key in keys:
            same_values(family['mix_minus_base_training'][key], [delta(a, b) for a, b in zip(ordered(family['absolute']['mix'][key]), ordered(family['absolute']['base'][key]))])
        if stage != 'clean_validation':
            need(set(family['accuracy_policy_contrasts']) == set(METRICS), 'both accuracy metrics required')
            for metric in METRICS:
                contrasts = family['accuracy_policy_contrasts'][metric]
                need(set(contrasts) == {'within_arm', 'risk_minus_random_mix_minus_base'} and set(contrasts['within_arm']) == set(ARMS), 'all paired contrast arms required')
                for arm in ARMS:
                    expected = {p + '_minus_' + ref for p in POLICIES for ref in REFERENCES if p != ref}
                    need(set(contrasts['within_arm'][arm]) == expected, 'all fixed policy contrasts required')
                    for p in POLICIES:
                        for ref in REFERENCES:
                            if p != ref:
                                a = ordered(family['absolute'][arm]['accuracy/' + p + '/' + metric])
                                b = ordered(family['absolute'][arm]['accuracy/' + ref + '/' + metric])
                                same_values(contrasts['within_arm'][arm][p + '_minus_' + ref], [delta(x, y) for x, y in zip(a, b)])
                a = ordered(contrasts['within_arm']['mix'][RISK + '_minus_random25'])
                b = ordered(contrasts['within_arm']['base'][RISK + '_minus_random25'])
                same_values(contrasts['risk_minus_random_mix_minus_base'], [delta(x, y) for x, y in zip(a, b)])
    return statistic_index(summary)


def tex_number(value, scale=1, signed=False):
    if value is None:
        return r'---'
    need(finite(value) and finite(scale) and scale > 0, 'finite display value/scale required')
    value *= scale
    rendered = format(value, '+.6g' if signed else '.6g')
    if 'e' in rendered:
        mantissa, exponent = rendered.split('e')
        return mantissa + r'\!\times\!10^{' + str(int(exponent)) + '}'
    return rendered


def pm(record, scale=1, signed=False):
    ordered(record)
    if record['mean'] is None:
        return '---'
    return '$' + tex_number(record['mean'], scale, signed) + r'\pm' + tex_number(record['sample_sd'], scale) + '$'


def seed_cells(record, scale=1, signed=False):
    return ['---' if v is None else '$' + tex_number(v, scale, signed) + '$' for v in ordered(record)]


def table(caption, label, headers, rows):
    return '\n'.join([r'\begin{table*}[p]', r'\centering\scriptsize', r'\caption{' + caption + '}',
                      r'\label{' + label + '}', r'\setlength{\tabcolsep}{3pt}',
                      r'\begin{tabular}{' + 'l' * len(headers) + '}', r'\toprule',
                      ' & '.join(headers) + r'\\', r'\midrule']
                     + [' & '.join(row) + r'\\' for row in rows]
                     + [r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def resolve(root, path):
    for key in path:
        root = root[key]
    return root


def render(summary, audit, index, input_manifest):
    claims = []
    def rec(path, table_label, scale=1):
        record = resolve(summary, path)
        ordered(record)
        claims.append({'table_label': table_label, 'summary_sha256': input_manifest['products']['summary']['sha256'],
                       'json_path': path, 'display_multiplier': scale, 'seed_signs': index['/'.join(path)]['seed_signs']})
        return record
    def signs(record):
        values = ordered(record)
        need(all(v is not None for v in values), 'complete accuracy seeds required for prose')
        return {'negative': sum(v < 0 for v in values), 'positive': sum(v > 0 for v in values), 'zero': sum(v == 0 for v in values)}
    def count_word(count):
        return ('zero', 'one', 'two', 'three')[count]
    prose_label = 'prose:goop3d25k-observed-findings'
    random = {stage: rec(['families', stage, 'mix_minus_base_training', 'accuracy/random25/position_coordinate_mse'], prose_label, 1e9) for stage in FAMILIES[:2]}
    gaps = {stage: {arm: rec(['families', stage, 'accuracy_policy_contrasts', METRICS[0], 'within_arm', arm, RISK + '_minus_random25'], prose_label, 1e9) for arm in ARMS} for stage in FAMILIES[:2]}
    interactions = {stage: rec(['families', stage, 'accuracy_policy_contrasts', METRICS[0], 'risk_minus_random_mix_minus_base'], prose_label, 1e9) for stage in FAMILIES[:2]}
    test_mix = gaps['same_state_test']['mix']
    mean_gap = 'positive' if test_mix['mean'] > 0 else 'negative' if test_mix['mean'] < 0 else 'zero'
    narrowed = signs(interactions['same_state_test'])['negative']
    narrowed_text = 'all three seeds' if narrowed == 3 else count_word(narrowed) + ' of three seeds'
    valid_signs = signs(interactions['same_state_valid'])
    findings_prose = ('Fixed-random exposure improves observed-history position MSE in '
                     + count_word(signs(random['same_state_valid'])['negative']) + ' of three validation seeds and '
                     + count_word(signs(random['same_state_test'])['negative']) + ' of three test seeds. '
                     + 'On test, the risk-minus-random gap narrows in ' + narrowed_text + ', from '
                     + pm(gaps['same_state_test']['base'], 1e9) + ' to ' + pm(test_mix, 1e9)
                     + r' in $10^{-9}$ coordinate-squared units. Mixed risk loses to random in '
                     + count_word(signs(test_mix)['positive']) + ' of three seeds, and its mean gap is ' + mean_gap + '. '
                     + 'Validation differs: risk beats random in ' + count_word(signs(gaps['same_state_valid']['base'])['negative'])
                     + ' base-training seeds and ' + count_word(signs(gaps['same_state_valid']['mix'])['negative'])
                     + ' mixed-training seeds; the interaction is negative in ' + count_word(valid_signs['negative'])
                     + ' seeds and positive in ' + count_word(valid_signs['positive']) + '.\n\n')
    fragments = {}
    intro = r'''\section{Goop3D: exploratory 25k graph-exposure extension}
\label{sec:goop3d-25k-exposure}
This separate exploratory study retains six faithful 25,000-update endpoints: base-only and mixed-graph training at three paired seeds. It uses three-dimensional Goop, a fixed source grid, and 150 validation histories, 150 test histories and 128 clean-validation frames per model. All 2,568 required observed cells are retained. Frames average within trajectories, then trajectories within each model; paired seed values precede the three-seed mean and sample SD. The SD is not a confidence interval. These endpoints are distinct from Goop/Sand 100k training and WaterDrop 100k-to-110k continuation; no material pooling or superiority claim follows.

Previous risk25 uses scores from the preceding observed base history. Matched histories and optional-pair budgets do not require learned risk selections to match across models. Autonomous H295 uses cached scores from its own preceding selected graph and is a separate comparison. Observed results alone do not establish autonomous accuracy, stability or speedup. A negative mixed-minus-base interaction narrows the risk-minus-random gap; it does not by itself show that risk beats random or that expansion beats the native graph.

'''
    fragments['observed_intro.tex'] = intro + findings_prose
    accuracy = []
    for metric, short, label_name in ((METRICS[0], 'position', 'Position-coordinate MSE'), (METRICS[1], 'normalized', 'Normalized-acceleration coordinate MSE')):
        scale = 1e9 if metric == METRICS[0] else 1
        for stage, split in (('same_state_valid', 'validation'), ('same_state_test', 'test')):
            label = 'tab:goop3d25k-observed-' + short + '-' + split
            rows = []
            for arm in ARMS:
                for policy in POLICIES:
                    r = rec(['families', stage, 'absolute', arm, 'accuracy/' + policy + '/' + metric], label, scale)
                    rows.append([arm, NAMES[policy], pm(r, scale), *seed_cells(r, scale)])
            title = (r'Position-coordinate MSE in $10^{-9}$ coordinate-squared units' if scale == 1e9 else label_name)
            accuracy.append(table(title + ', Goop3D 25k observed ' + split + r'. Every row retains 150 declared histories per model and all three ordered training seeds. Previous risk25 uses the preceding observed base history.', label,
                                  ['Training', 'Policy', r'Mean $\pm$ SD', 'Seed 0', 'Seed 1', 'Seed 2'], rows))
    fragments['observed_accuracy.tex'] = '\n'.join(accuracy)
    contrasts = []
    for metric, short in ((METRICS[0], 'position'), (METRICS[1], 'normalized')):
        scale = 1e9 if metric == METRICS[0] else 1
        label = 'tab:goop3d25k-observed-' + short + '-contrasts'
        rows = []
        for stage, split in (('same_state_valid', 'Validation'), ('same_state_test', 'Test')):
            base = ['families', stage]
            c = base + ['accuracy_policy_contrasts', metric]
            choices = [
                ('Mix $-$ base (base)', base + ['mix_minus_base_training', 'accuracy/base/' + metric]),
                ('Mix $-$ base (random)', base + ['mix_minus_base_training', 'accuracy/random25/' + metric]),
                ('Risk $-$ random (base)', c + ['within_arm', 'base', RISK + '_minus_random25']),
                ('Risk $-$ random (mix)', c + ['within_arm', 'mix', RISK + '_minus_random25']),
                ('Training interaction', c + ['risk_minus_random_mix_minus_base'])]
            for text, path in choices:
                r = rec(path, label, scale)
                rows.append([split, text, pm(r, scale, signed=True), *seed_cells(r, scale, signed=True)])
        caption = ('Position-coordinate MSE' if short == 'position' else 'Normalized-acceleration coordinate MSE')
        if scale == 1e9:
            caption += r' in $10^{-9}$ coordinate-squared units'
        contrasts.append(table(caption + r' contrasts for all paired Goop3D 25k seeds, with validation and test separate. The first two rows per split compare training arms at fixed policies; the next two compare preceding-observed-base risk with random within each arm. The interaction is the mixed-minus-base change in this gap. Negative favors the first term; a negative interaction alone does not establish risk superiority.', label,
                               ['Split', 'Contrast', r'Mean $\pm$ SD', 'Seed 0', 'Seed 1', 'Seed 2'], rows))
    fragments['observed_contrasts.tex'] = '\n'.join(contrasts)
    training = []
    for metric, short in ((METRICS[0], 'position'), (METRICS[1], 'normalized')):
        scale = 1e9 if metric == METRICS[0] else 1
        label = 'tab:goop3d25k-observed-' + short + '-training'
        rows = []
        for stage, split in (('same_state_valid', 'Validation'), ('same_state_test', 'Test')):
            for policy in POLICIES:
                r = rec(['families', stage, 'mix_minus_base_training', 'accuracy/' + policy + '/' + metric], label, scale)
                rows.append([split, NAMES[policy], pm(r, scale, signed=True), *seed_cells(r, scale, signed=True)])
        scale_caption = r' Position-coordinate MSE is in $10^{-9}$ coordinate-squared units.' if scale == 1e9 else ''
        training.append(table('Every Goop3D 25k observed ' + short + r'-coordinate MSE training effect, mixed minus base, at the fixed policy.' + scale_caption + r' Both splits and all policies/seeds are retained. Negative favors mixed training; it does not establish expansion superiority over the native graph.', label,
                              ['Split', 'Policy', r'Mean $\pm$ SD', 'Seed 0', 'Seed 1', 'Seed 2'], rows))
    fragments['observed_training.tex'] = '\n'.join(training)
    label = 'tab:goop3d25k-clean'
    rows = []
    for arm in (*ARMS, 'mix-minus-base'):
        for key, name in zip(CLEAN, CLEAN_NAMES):
            path = ['families', 'clean_validation'] + (['mix_minus_base_training'] if arm == 'mix-minus-base' else ['absolute', arm]) + ['metrics/' + key]
            r = rec(path, label)
            rows.append([arm, name, pm(r, signed=arm == 'mix-minus-base'), *seed_cells(r, signed=arm == 'mix-minus-base')])
    fragments['observed_clean.tex'] = table(r'Goop3D 25k clean validation, all 128 declared frames per model under the base graph. Absolute quantities and paired training changes use the saved normalization. These residual-scale checks do not establish calibration of action benefit or autonomous stability.', label,
                                           ['Training/contrast', 'Metric', r'Mean $\pm$ SD', 'Seed 0', 'Seed 1', 'Seed 2'], rows)
    diagnostics = []
    for stage, split in (('same_state_valid', 'validation'), ('same_state_test', 'test')):
        label = 'tab:goop3d25k-observed-timing-' + split
        rows = []
        for arm in ARMS:
            for policy in TIMING_CASES:
                cells = [pm(rec(['families', stage, 'absolute', arm, 'timing/' + policy + '/' + key], label)) for key in ('end_to_end_seconds', 'score_generation_seconds', 'current_graph_and_selection_seconds', 'current_forward_seconds')]
                rows.append([arm, NAMES[policy], *cells])
        diagnostics.append(table('Goop3D 25k observed ' + split + r' descriptive timing in seconds (three-seed mean $\pm$ SD). Previous risk includes score generation; natural base reference is a separate timing control. Saved timing arithmetic does not establish isolated hardware timing or causal speedup. Every timing component and all seed values remain in the companion.', label,
                                 ['Training', 'Method', 'End-to-end', 'Score', 'Graph/select', 'Forward'], rows))
    label = 'tab:goop3d25k-observed-risk-correlations'
    rows = []
    for stage, split in (('same_state_valid', 'Validation'), ('same_state_test', 'Test')):
        for arm in ARMS:
            for key, name in [('previous_risk_vs_base_error', 'Risk vs. base error'), ('previous_risk_vs_' + RISK + '_benefit', 'Risk vs. its action benefit')]:
                r = rec(['families', stage, 'absolute', arm, 'correlations/' + key], label)
                rows.append([split, arm, name, pm(r), *seed_cells(r)])
    diagnostics.append(table(r'Goop3D 25k observed residual-risk association with base prediction error and with the benefit of its own sparse action. Saved frame Spearman values average within trajectory and then across trajectories; all three seeds and both splits are shown. Undefined correlations remain null. The companion retains every other fixed benefit, correlation, graph and boundary statistic.', label,
                             ['Split', 'Training', 'Association', r'Mean $\pm$ SD', 'Seed 0', 'Seed 1', 'Seed 2'], rows))
    fragments['observed_diagnostics.tex'] = '\n'.join(diagnostics)
    rows = []
    for stage in STAGES:
        cells = [c for r in summary['all_original_accounting'] if r['stage'] == stage for c in r['cells']]
        counts = Counter(c['state'] for c in cells)
        rows.append([stage.replace('_', r'\_'), str(len(cells)), str(counts['completed_required_outcome']), str(counts['recorded_failed_outcome']), str(counts['timed_out_current']), str(counts['not_completed_before_invocation_end']), str(counts['never_started'])])
    fragments['observed_accounting.tex'] = table(r'Historical all-stage Goop3D accounting retained by the observed audit: 4,728 original cells. Autonomous rows describe the original stopped attempt, not current successor rollout results. Observed rows retain 900 validation histories, 900 test histories and 768 clean frames. Cell types are not interchangeable rollout denominators.', 'tab:goop3d25k-original-accounting',
                                               ['Original stage', 'Required', 'Completed', 'Failed', 'Timed out', 'Not completed', 'Never started'], rows)
    fragments['appendix_goop3d_observed.tex'] = '\n'.join(fragments.values()) + r'''
% GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT
% A separately verified H295 fragment may follow here; never insert its values
% into observed statistics, captions, denominators, or the observed companion.
'''
    lines = ['# Goop3D 25k observed-only presentation', '',
             'All values below retain seeds 0, 1, 2. “+” means a positive contrast, “−” a negative contrast, and null remains undefined. These are descriptions of the fixed cohort, not tests of significance.', '',
             '| Split | Contrast | Seed signs (0, 1, 2) |', '|---|---|---|']
    for stage, split in (('same_state_valid', 'Validation'), ('same_state_test', 'Test')):
        for policy in POLICIES:
            path = ['families', stage, 'mix_minus_base_training', 'accuracy/' + policy + '/position_coordinate_mse']
            lines.append('| ' + split + ' | Mix−base at ' + NAMES[policy] + ' | ' + ', '.join(index['/'.join(path)]['seed_signs']) + ' |')
        for arm in ARMS:
            path = ['families', stage, 'accuracy_policy_contrasts', 'position_coordinate_mse', 'within_arm', arm, RISK + '_minus_random25']
            lines.append('| ' + split + ' | Risk−random, ' + arm + ' | ' + ', '.join(index['/'.join(path)]['seed_signs']) + ' |')
        path = ['families', stage, 'accuracy_policy_contrasts', 'position_coordinate_mse', 'risk_minus_random_mix_minus_base']
        lines.append('| ' + split + ' | Risk−random interaction | ' + ', '.join(index['/'.join(path)]['seed_signs']) + ' |')
    lines.extend(['', 'A negative interaction does not imply either within-arm risk−random gap is negative. Validation/test disagreement and heterogeneous signs remain visible. Fixed-policy exposure effects do not establish expansion benefit relative to native base; the complete within-arm comparisons are retained in the companion.', '',
                  'Historical 4,728-cell status accounting does not report current autonomous completion. Autonomous H295 results must enter through a separately verified product, namespace, companion and appendix fragment. The observed input summary remains unchanged.'])
    fragments['findings_observed.md'] = '\n'.join(lines) + '\n'
    return fragments, claims


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw_manifest = read_bound(args.inputs, args.inputs_sha256)
    manifest = strict(raw_manifest)
    need(manifest['schema'] == 'goop3d_observed_presentation_inputs_v1' and set(manifest['products']) == set(PRODUCTS), 'explicit complete observed input set required')
    products, raw_products, pins = {}, {}, {}
    for name in PRODUCTS:
        entry = manifest['products'][name]
        need(set(entry) == {'path', 'sha256'}, 'exact product path/hash required')
        raw_products[name] = read_bound(entry['path'], entry['sha256'])
        products[name], pins[name] = strict(raw_products[name]), entry['sha256']
    index = validate(products, pins)
    fragments, claims = render(products['summary'], products['audit'], index, manifest)
    companion = {'schema': 'goop3d25k_observed_complete_companion_v1', 'dataset': 'Goop-3D', 'endpoint_updates': 25000,
                 'namespace': 'observed', 'inputs': manifest, 'inputs_manifest_sha256': args.inputs_sha256,
                 'summary_unchanged': products['summary'], 'statistic_index': index,
                 'arithmetic_check': products['arithmetic_check'], 'completion': products['completion'],
                 'retained_original_failure': products['retained_original_failure'],
                 'retained_checker_diagnosis': products['diagnosis'],
                 'audit_provenance': {k: v for k, v in products['audit'].items() if k != 'models'},
                 'autonomous_numerical_products_included': False,
                 'scope': 'Every observed scalar object, seed, null, fixed policy contrast, clean value, timing/benefit/correlation/graph/boundary metric and historical cell retained. Full audited row details remain in the hash-bound audit input.'}
    raw_companion = encode(companion)
    payloads = {name: text.encode() for name, text in fragments.items()}
    payloads.update({'inputs.json': raw_manifest, 'source_summary_unchanged.json': raw_products['summary'],
                     'claim_source_map.json': encode({'schema': 'goop3d25k_observed_claim_source_map_v1', 'table_values': claims}),
                     'complete_observed_companion.json.gz': gzip.compress(raw_companion, mtime=0)})
    output = args.output
    need(output.is_absolute() and output == output.resolve() and not output.exists()
         and all(not p.is_symlink() for p in output.parents), 'fresh canonical output directory required')
    for entry in manifest['products'].values():
        need(not Path(entry['path']).is_relative_to(output), 'output must not contain inputs')
        read_bound(entry['path'], entry['sha256'])
    read_bound(args.inputs, args.inputs_sha256)
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        with (output / name).open('xb') as stream:
            stream.write(raw)
    receipt = {'schema': 'goop3d25k_observed_presentation_receipt_v1', 'status': 'observed_fragments_and_complete_companion_created',
               'renderer_sha256': digest(Path(__file__).read_bytes()), 'inputs_manifest_sha256': args.inputs_sha256,
               'required_observed_cells': 2568, 'required_all_cells': 4728, 'statistic_objects': len(index),
               'source_summary_bytes_unchanged': True, 'source_summary_sha256': pins['summary'],
               'manuscript_edited': False, 'autonomous_products_read': False, 'science_executed': False,
               'companion_uncompressed_sha256': digest(raw_companion), 'companion_uncompressed_bytes': len(raw_companion),
               'files': {name: {'sha256': digest(raw), 'bytes': len(raw)} for name, raw in payloads.items()},
               'layout': 'Reviewable fragments only; native compilation and visual checking remain required at integration.'}
    (output / 'receipt.json').write_bytes(encode(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
