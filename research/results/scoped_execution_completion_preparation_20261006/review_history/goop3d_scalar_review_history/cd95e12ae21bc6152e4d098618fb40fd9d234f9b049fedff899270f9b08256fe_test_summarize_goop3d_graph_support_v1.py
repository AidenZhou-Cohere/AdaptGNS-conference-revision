import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d3_scalar_summary_tests', HERE / 'summarize_goop3d_graph_support_v1.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n'); return path


def manifest(n=1, split='valid'):
    return {'record_count': n, 'metadata': {'dim': 3, 'bounds': [[.2, .8]] * 3,
            'dt': .0025, 'default_connectivity_radius': .025},
            'records': [{'source_index': i, 'id': f'{split}:{i:06d}', 'positions': {'shape': [301, 4 + i, 3]}} for i in range(n)]}


def boundary():
    return {'fraction_particles_outside': .25, 'fraction_particles_outside_by_more_than_1e-6': .25,
            'maximum_coordinate_excursion': .01, 'mean_particle_maximum_excursion': .0025,
            'coordinate_minimum': [.19, .2, .2], 'coordinate_maximum': [.7, .7, .7]}


def graph(policy='base', particles=4):
    base, extra = 3, 4
    budget = 0 if policy == 'base' else extra if policy == 'dense' else 1
    native = 2 * base + particles
    return {'candidate_pairs': base + extra, 'geometric_base_pairs': base, 'available_annulus_pairs': extra,
            'optional_pair_budget': 1, 'retained_optional_pairs': budget, 'directed_edges': native + 2 * budget,
            'native_base_directed_edges': native, 'native_base_self_edges': particles,
            'native_base_receivers_above_cap_before_capping': 0, 'native_base_edges_removed_by_cap': 0,
            'native_base_max_receiver_degree': 4, 'native_base_asymmetric_directed_edges': 0,
            'native_base_sha256': 'a' * 64, 'selected_optional_pair_sha256': hashlib.sha256(policy.encode()).hexdigest(),
            'directed_edge_sha256': hashlib.sha256((policy + 'edges').encode()).hexdigest(), 'native_base_prefix_preserved': True}


def rollout(arm, seed, item, value=1., completed=295):
    g = graph(item['policy'], item['particles']); b = boundary()
    attempts = [{**g, 'forecast_step': i + 1, 'accepted': True, 'coordinate_mse': value,
                 'predicted_boundary': b, 'ground_truth_boundary': b} for i in range(completed)]
    if completed < 295: attempts.append({'forecast_step': completed + 1, 'accepted': False, 'failure': {'category': 'coordinate_guard'}})
    return {**item, 'arm': arm, 'training_seed': seed, 'objective': 'faithful', 'horizon': 295,
            'status': 'complete' if completed == 295 else 'failed', 'completed_steps': completed,
            'failure': None if completed == 295 else {'category': 'coordinate_guard'}, 'mse_per_step': [value] * completed,
            'mean_rollout_mse': value if completed == 295 else None, 'mse_at_final_horizon': value if completed == 295 else None,
            'mse_at_declared_trace_steps': {str(k): value if completed >= k else None for k in (1, 10, 50, 200, 295)},
            'synchronized_call_seconds': 2., 'publication_seconds': .1,
            'predicted_boundary_per_step': [b] * completed, 'ground_truth_boundary_per_step': [b] * completed,
            'attempts': attempts, 'directed_edges_per_step': [g['directed_edges']] * completed,
            'retained_optional_pairs_per_step': [g['retained_optional_pairs']] * completed,
            'candidate_pairs_per_step': [g['candidate_pairs']] * completed,
            'base_pairs_per_step': [g['geometric_base_pairs']] * completed,
            'mean_normalized_acceleration_variance_per_step': [.5] * completed,
            'mean_directed_edges': g['directed_edges'] if completed == 295 else None}


def full_stages(n=1, value=None):
    schedule = M.expected(manifest(n), 'full-rollout'); stages = []
    for arm, seed in M.MODELS:
        rows = []; cells = []
        for item in schedule:
            for policy in M.POLICIES:
                x = {**item, 'policy': policy}
                rows.append(rollout(arm, seed, x, value(arm, seed, item['source_index'], policy) if value else 1.))
                cells.append({**x, 'state': 'completed_required_outcome'})
        stages.append({'arm': arm, 'seed': seed, 'rows': rows, 'cells': cells, 'stage': 'full_rollout_valid', 'outcome': {}})
    return stages, schedule


def diagnostic_row(arm='base', seed=0, item=None, split='valid', failed=False):
    item = item or M.expected(manifest(), 'same-state')[0]
    row = {**item, 'arm': arm, 'training_seed': seed, 'objective': 'faithful', 'status': 'failed' if failed else 'complete',
           'failure': {'category': 'native_parity_failure'} if failed else None,
           'synchronized_call_seconds': 20., 'publication_seconds': .1, 'risk_source': 'previous observed native-base history',
           'random_seed_material': [20261006, 93000, seed, 0 if split == 'valid' else 1, item['source_index'], item['target_frame']],
           'native_parity': {'current': {'passed': True}, 'previous': {'passed': True}},
           'natural_shared_base': {'ordered_edges_exact': True, 'outputs_within_fixed_parity_tolerance': True},
           'warmup_calls': [], 'timed_calls': [], 'policies': {}, 'benefit': {}, 'correlations': {}}
    if failed: return row
    D = M.load_scalar('goop3d_diagnostic_metrics_v1.py')
    for r in range(-1, 7):
        for slot, method in enumerate(D.timing_order(item['source_index'], item['schedule_index'], max(r, 0))):
            times = {k: (.01 * (r + 2) if k not in ('score_graph_seconds', 'score_forward_seconds', 'score_generation_seconds')
                          or method == M.DIAG_POLICIES[-1] else 0.) for k in M.T_KEYS}
            times['network_passes'] = 2 if method == M.DIAG_POLICIES[-1] else 1
            g = graph(method, 4 + item['source_index'])
            if method == 'natural_base_reference': g = {'retained_optional_pairs': 0, 'native_base_prefix_preserved': True, 'directed_edge_sha256': graph('base')['directed_edge_sha256']}
            c = {'method': method, 'round': r, 'slot': slot, 'status': 'complete', 'failure': None, 'timing': times,
                 'graph': g, 'prediction_sha256': 'b' * 64}
            if r > 0:
                c['repeat_consistency'] = {k: True for k in ('ordered_edges_exact', 'outputs_within_fixed_parity_tolerance',
                    'previous_score_ordered_edges_exact', 'previous_score_outputs_within_fixed_parity_tolerance', 'physical_scores_exact')}
            row['warmup_calls' if r < 0 else 'timed_calls'].append(c)
    errors = {'base': 4., 'dense': 3., 'random25': 2., 'speed25': 1.5, 'relative-velocity-RMS25': 1.8,
              'previous-observed-base-risk25': 2.5, 'natural_base_reference': 4.}
    for method, error in errors.items():
        calls = [c for c in row['timed_calls'] if c['method'] == method]
        row['policies'][method] = {'status': 'complete', 'repeat_consistent': True, 'completed_repetitions': 7, 'expected_repetitions': 7,
            'metrics': {'position_coordinate_mse': error, 'normalized_coordinate_mse': 2 * error},
            'timing': {k: M.timing_stats([c['timing'][k] for c in calls]) for k in M.T_KEYS},
            'graph': calls[0]['graph'], 'prediction_boundary': boundary()}
    row['truth_boundary'] = boundary()
    for method in M.DIAG_POLICIES[1:]:
        row['benefit'][method] = {'mean_position_vector_benefit': 3 * (4 - errors[method]),
            'mean_normalized_vector_benefit': 6 * (4 - errors[method]), 'positive_fraction': 1., 'negative_fraction': 0., 'zero_fraction': 0.}
        if method != 'dense':
            row['benefit'][method].update(dense_sparse_sign_disagreement_fraction=0., dense_positive_sparse_nonpositive_fraction=0.)
    names = ('previous_risk_vs_base_error',) + tuple('previous_risk_vs_' + p + '_benefit' for p in M.DIAG_POLICIES[1:]) + tuple('dense_vs_' + p + '_benefit' for p in M.DIAG_POLICIES[2:])
    row['correlations'] = {k: {'value': .2, 'reason': None, 'particles': 4 + item['source_index']} for k in names}
    return row


@pytest.mark.parametrize('n', [1, 2, 29, 30, 31, 100])
def test_actual_population_grid_and_all_schedules(n):
    m = manifest(n); grid = list(range(n)) if n <= 30 else [j * (n - 1) // 29 for j in range(30)]
    assert [r['source_index'] for r in M.expected(m, 'full-rollout')] == grid
    same = M.expected(m, 'same-state')
    assert [(r['source_index'], r['target_frame']) for r in same] == [(i, t) for i in grid for t in (7, 80, 153, 226, 300)]
    clean = M.expected(m, 'clean-validation')
    assert len(clean) == len({M.cell_key(r, 'clean-validation') for r in clean}) == 128
    assert clean[0]['target_frame'] == 6 and clean[-1]['target_frame'] == 300


@pytest.mark.parametrize('bad', ['count', 'dimension', 'frames', 'order'])
def test_manifest_rejects_wrong_source_contract(bad):
    m = manifest(2)
    if bad == 'count': m['record_count'] = 3
    if bad == 'dimension': m['records'][0]['positions']['shape'][2] = 2
    if bad == 'frames': m['records'][0]['positions']['shape'][0] = 401
    if bad == 'order': m['records'].reverse()
    with pytest.raises(ValueError): M.expected(m, 'full-rollout')


@pytest.mark.parametrize('n', [0, 199, 200, 294, 295])
def test_failed_prefix_vs_pointwise200_and_full295(n):
    item = {**M.expected(manifest(), 'full-rollout')[0], 'policy': 'base'}
    values, prefix = M.validate_rollout(rollout('base', 0, item, completed=n), 'base', 0, item)
    assert values['mean_rollout_mse'] == (1. if n == 295 else None)
    assert values['mse_forecast200'] == (1. if n >= 200 else None)
    assert values['mse_forecast295'] == (1. if n == 295 else None)
    assert bool(prefix) is (n < 295)


@pytest.mark.parametrize('bad', ['wrong_horizon', 'D2_boundary', 'NaN', 'mean', 'budget', 'cap', 'step_graph', 'attempt', 'negative_clock', 'failed_mean', 'trace'])
def test_rollout_rejects_scalar_inconsistency(bad):
    item = {**M.expected(manifest(), 'full-rollout')[0], 'policy': 'random25'}
    row = copy.deepcopy(rollout('base', 0, item, completed=294 if bad == 'failed_mean' else 295))
    if bad == 'wrong_horizon': row['horizon'] = 395
    if bad == 'D2_boundary': row['predicted_boundary_per_step'][0]['coordinate_minimum'] = [.2, .2]
    if bad == 'NaN': row['mse_per_step'][0] = float('nan')
    if bad == 'mean': row['mean_rollout_mse'] = 2
    if bad == 'budget': row['attempts'][0]['retained_optional_pairs'] += 1
    if bad == 'cap': row['attempts'][0]['native_base_max_receiver_degree'] = 129
    if bad == 'step_graph': row['directed_edges_per_step'][0] += 1
    if bad == 'attempt': row['attempts'][0]['accepted'] = False
    if bad == 'negative_clock': row['publication_seconds'] = -1
    if bad == 'failed_mean': row['mean_rollout_mse'] = 1
    if bad == 'trace': row['mse_at_declared_trace_steps']['200'] = 2
    with pytest.raises(ValueError): M.validate_rollout(row, 'base', 0, item)


def test_equal_source_weighting_and_explicit_training_interaction():
    def values(arm, seed, source, policy):
        offset = [0., 8.][source]
        if arm == 'base': return offset + {'base': 1., 'random25': 2., 'laggedrisk25': 3., 'speed25': 1.5, 'relative-velocity-RMS25': 1.7, 'dense': 4.}[policy]
        return offset + {'base': .5, 'random25': 1.8, 'laggedrisk25': 1.8, 'speed25': 1.4, 'relative-velocity-RMS25': 1.6, 'dense': 3.}[policy]
    stages, schedule = full_stages(2, values); out = M.full_summary(stages, schedule)['paired']
    assert out['absolute']['mean_rollout_mse']['base']['base']['mean'] == 5.
    assert out['risk_minus_random_mix_minus_base_interaction']['mean_rollout_mse']['mean'] == pytest.approx(-1.)
    assert out['within_arm_policy_contrasts']['mean_rollout_mse']['mix']['laggedrisk25_minus_base']['mean'] == pytest.approx(1.3)
    assert 'laggedrisk25_minus_speed25' in out['within_arm_policy_contrasts']['mean_rollout_mse']['mix']
    assert 'laggedrisk25_minus_relative-velocity-RMS25' in out['within_arm_policy_contrasts']['mean_rollout_mse']['mix']


def test_one_missing_seed_never_yields_survivor_mean():
    stages, schedule = full_stages(); stage = stages[-1]
    stage['rows'] = [r for r in stage['rows'] if r['policy'] != 'laggedrisk25']
    next(c for c in stage['cells'] if c['policy'] == 'laggedrisk25')['state'] = 'timed_out_current'
    out = M.full_summary(stages, schedule)
    x = out['paired']['absolute']['mean_rollout_mse']['mix']['laggedrisk25']
    assert x['defined_seed_pairs'] == 2 and x['mean'] is x['sample_sd'] is None
    assert out['required_outcomes'] == 36 and out['coverage']['timed_out_current'] == 1


def test_cross_model_truth_mismatch_is_rejected():
    stages, schedule = full_stages(); row = copy.deepcopy(stages[1]['rows'][0])
    row['ground_truth_boundary_per_step'][0]['coordinate_minimum'][0] = .18
    stages[1]['rows'][0] = row
    with pytest.raises(ValueError, match='truth boundary'): M.full_summary(stages, schedule)


def test_same_state_complete_and_failed_call_paths():
    item = M.expected(manifest(), 'same-state')[0]
    M.validate_diagnostic(diagnostic_row(item=item), 'base', 0, item, 'valid', 'same-state', 4)
    M.validate_diagnostic(diagnostic_row(item=item, failed=True), 'base', 0, item, 'valid', 'same-state', 4)


@pytest.mark.parametrize('bad', ['score_overhead', 'draw', 'call_order', 'unequal_budget', 'repeated', 'timing_mean', 'D2_benefit', 'correlation', 'native_parity', 'D2_truth'])
def test_same_state_rejects_bad_allocation_timing_and_units(bad):
    item = M.expected(manifest(), 'same-state')[0]; row = diagnostic_row(item=item)
    if bad == 'score_overhead': next(c for c in row['timed_calls'] if c['method'] == M.DIAG_POLICIES[-1])['timing']['network_passes'] = 1
    if bad == 'draw': row['random_seed_material'][-1] += 1
    if bad == 'call_order': row['timed_calls'][0]['slot'] = 6
    if bad == 'unequal_budget': row['timed_calls'][0]['graph']['retained_optional_pairs'] += 1
    if bad == 'repeated': row['timed_calls'][8]['repeat_consistency']['ordered_edges_exact'] = False
    if bad == 'timing_mean': row['policies']['base']['timing']['end_to_end_seconds']['mean'] += 1
    if bad == 'D2_benefit': row['benefit']['dense']['mean_position_vector_benefit'] = 2.
    if bad == 'correlation': row['correlations']['previous_risk_vs_base_error']['value'] = 2.
    if bad == 'native_parity': row['native_parity']['previous']['passed'] = False
    if bad == 'D2_truth': row['truth_boundary']['coordinate_minimum'] = [.2, .2]
    with pytest.raises(ValueError): M.validate_diagnostic(row, 'base', 0, item, 'valid', 'same-state', 4)


def test_clean_vector_coordinate_consistency_and_negative_nll_allowed():
    item = M.expected(manifest(), 'clean-validation')[0]
    row = {**item, 'arm': 'base', 'training_seed': 0, 'objective': 'faithful', 'status': 'complete', 'failure': None,
           'synchronized_call_seconds': 1., 'publication_seconds': .1, 'graph': graph(),
           'metrics': {'normalized_acceleration_coordinate_mse': 2., 'realized_normalized_vector_se': 6.,
                       'predicted_normalized_vector_se': 3., 'constant_free_gaussian_nll': -.5}}
    M.validate_diagnostic(row, 'base', 0, item, 'valid', 'clean-validation', 4)
    row['metrics']['realized_normalized_vector_se'] = 4.
    with pytest.raises(ValueError): M.validate_diagnostic(row, 'base', 0, item, 'valid', 'clean-validation', 4)


def test_unequal_clean_counts_still_weight_sources_equally():
    D = M.load_scalar('goop3d_diagnostic_metrics_v1.py')
    schedule = [{'source_index': 0, 'target_frame': 6}, {'source_index': 0, 'target_frame': 7}, {'source_index': 1, 'target_frame': 6}]
    rows = [{**r, 'value': 2. if r['source_index'] == 0 else 8.} for r in schedule]
    out = D.aggregate(schedule, rows, lambda r: r['value'])
    assert out['equal_trajectory_mean'] == 5.
    assert D.aggregate(schedule, rows[:-1], lambda r: r['value'])['equal_trajectory_mean'] is None


def test_undefined_correlations_are_retained_not_dropped():
    D = M.load_scalar('goop3d_diagnostic_metrics_v1.py'); schedule = M.expected(manifest(), 'same-state')
    rows = [diagnostic_row(item=r) for r in schedule]
    rows[0]['correlations']['previous_risk_vs_base_error'] = {'value': None, 'reason': 'constant_rank_vector', 'particles': 4}
    out = D.summarize(schedule, rows, 'same-state')['correlations']['previous_risk_vs_base_error']
    assert out['equal_trajectory_mean'] is None and out['undefined_returned_frame_reasons'] == {'constant_rank_vector': 1}


def fixture(tmp_path, monkeypatch, completed_row=False):
    valid = put(tmp_path / 'valid.json', manifest()); test = put(tmp_path / 'test.json', manifest(split='test'))
    monkeypatch.setattr(M, 'VALID_SHA', M.sha(valid))
    protocol = tmp_path / 'protocol.md'; protocol.write_text('Synthetic prospective37-update D3 fixture, not a scientific endpoint.\n')
    models = [{'arm': a, 'seed': s, 'completed_steps': 37, 'objective': 'faithful',
               'checkpoint_sha256': hashlib.sha256(f'{a}{s}'.encode()).hexdigest()} for a, s in M.MODELS]
    audit = {'schema': 'adaptgns_goop3d_graph_support_complete_cohort_audit_v2', 'issued_by': 'root',
        'status': 'all_six_endpoints_and_pairing_verified', 'training_schema': 'adaptgns_goop3d_graph_support_cuda_training_v2',
        'endpoint_updates': 37, 'protocol_sha256': M.sha(protocol), 'trainer_sha256': M.SOURCE_PINS['train_goop3d_graph_support_cuda_v2.py'],
        'graph_sha256': M.SOURCE_PINS['goop3d_graph_support_vectorized_v1.py'],
        'models': [{**m, 'graph_history_updates': 37, **{k: True for k in ('all_optimizer_steps_equal_endpoint', 'all_state_and_moments_finite',
                    'source_data_protocol_verified', 'checkpoint_bytes_verified')}} for m in models],
        'paired_seeds': [{'seed': s, **{k: True for k in ('initial_model_tensor_identity', 'initial_cpu_cuda_rng_identity',
                         'all_frame_noise_lr_schedules_equal', 'all_graph_budgets_and_rng_material_verified')}} for s in range(3)]}
    audit_path = put(tmp_path / 'cohort_audit.json', audit)
    cohort = {k: audit[k] for k in ('training_schema', 'endpoint_updates', 'protocol_sha256', 'trainer_sha256', 'graph_sha256')}
    cohort.update(schema='adaptgns_goop3d_graph_support_final_cohort_v2', status='frozen_for_final_evaluation', issued_by='root',
                  cohort_audit_sha256=M.sha(audit_path), models=models)
    cohort_path = put(tmp_path / 'cohort.json', cohort); stages = []
    for model in models:
        a, s = model['arm'], model['seed']
        for name, mode, split in M.STAGES:
            directory = tmp_path / 'outputs' / f'{a}_seed{s}' / name
            child_path = tmp_path / 'contracts' / f'{a}{s}_{name}.json'
            options = {'--purpose': 'final_evaluation', '--mode': mode, '--split': split, '--arm': a, '--seed': str(s),
                '--checkpoint-updates': '37', '--checkpoint-sha256': model['checkpoint_sha256'], '--output-dir': str(directory),
                '--cohort': str(cohort_path), '--cohort-audit': str(audit_path), '--protocol': str(protocol),
                '--manifest': str(valid if split == 'valid' else test), '--trainer-source': str(HERE / 'train_goop3d_graph_support_cuda_v2.py'),
                '--checkpoint': str(tmp_path / f'omitted_{a}{s}.pt'), '--release': str(child_path)}
            command = ['/synthetic/python', str(HERE / 'evaluate_goop3d_graph_support_v1.py'), '--execute'] + [v for k, value in options.items() for v in (k, value)]
            input_hashes = {options[k]: v for k, v in {'--cohort': M.sha(cohort_path), '--cohort-audit': M.sha(audit_path),
                '--protocol': M.sha(protocol), '--manifest': M.sha(valid if split == 'valid' else test),
                '--trainer-source': M.SOURCE_PINS['train_goop3d_graph_support_cuda_v2.py'], '--checkpoint': model['checkpoint_sha256']}.items()}
            child = {'schema': 'adaptgns_goop3d_evaluation_release_v1', 'issued_by': 'root', 'status': 'admitted_for_execution',
                'purpose': 'final_evaluation', 'mode': mode, 'split': split, 'arm': a, 'seed': s, 'checkpoint_updates': 37,
                'scientific_endpoint_updates': 37,
                'checkpoint_sha256': model['checkpoint_sha256'], 'evaluator_sha256': M.SOURCE_PINS['evaluate_goop3d_graph_support_v1.py'],
                'graph_sha256': M.SOURCE_PINS['goop3d_graph_support_vectorized_v1.py'], 'files_sha256': input_hashes}
            child_path = put(tmp_path / 'contracts' / f'{a}{s}_{name}.json', child)
            cells = [{**r, 'state': 'never_started', 'reason': 'synthetic_allocation_not_started'} for r in M.cell_schedule(manifest(split=split), mode)]
            entry = {'arm': a, 'seed': s, 'stage': name, 'mode': mode, 'split': split, 'endpoint_updates': 37,
                'checkpoint_sha256': model['checkpoint_sha256'], 'directory': str(directory), 'command': command,
                'release_file': str(child_path), 'release_sha256': M.sha(child_path), 'cells': cells,
                'outcome': {'started': False, 'pid': None, 'state': 'never_started', 'termination_reason': 'synthetic_allocation_not_started'}}
            if completed_row and (a, s, name) == ('base', 0, 'full_rollout_valid'):
                saved_protocol = {'schema': 'adaptgns_goop3d_graph_support_evaluation_v1', 'purpose': 'final_evaluation',
                    'mode': mode, 'split': split, 'arm': a, 'seed': s, 'checkpoint_updates': 37, 'checkpoint_sha256': model['checkpoint_sha256'],
                    'frames': 301, 'horizon': 295, 'policies': list(M.POLICIES), 'schedule': M.expected(manifest(), mode),
                    'source_population_count': 1, 'prospective_source_grid': [0], 'selected_source_indices': [0],
                    'release_sha256': M.sha(child_path), 'input_files_sha256': input_hashes}
                pp = put(directory / 'protocol.json', saved_protocol); item = M.cell_schedule(manifest(), mode)[0]
                row = rollout(a, s, item); artifact = directory / M.filename(item, mode).replace('.json', '.npz')
                artifact.write_bytes(b'opaque synthetic numeric artifact bytes; not a real array')
                row.update(protocol_sha256=M.sha(pp), artifact_file=artifact.name, artifact_sha256=M.sha(artifact))
                rp = put(directory / M.filename(item, mode), row)
                cells[0].update(state='completed_required_outcome', row_file=rp.name, row_sha256=M.sha(rp), artifact_sha256=M.sha(artifact), failure=None)
                for cell in cells[1:]: cell.update(state='not_completed_before_invocation_end', reason='synthetic_stop')
                entry['outcome'] = {'started': True, 'state': 'exited', 'termination_reason': 'synthetic_stop', 'stopped_and_reaped': True,
                    'pid': 123, 'start_ticks': 456, 'command': command, 'hostname': 'synthetic', 'exit_code': 0, 'elapsed_seconds': 5.,
                    'signals': [], 'outer_timeout_seconds': 10., 'absolute_stop_utc': '2026-10-07T00:00:00+00:00'}
            stages.append(entry)
    ledger = {'schema': M.LEDGER_SCHEMA, 'producer': 'supervisor_candidate', 'dataset': 'Goop-3D',
        'state': 'stopped_all_owned_processes_reaped', 'unreaped_owned_children': [], 'all_pinned_inputs_reverified': True,
        'cohort_sha256': M.sha(cohort_path), 'cohort_audit_sha256': M.sha(audit_path), 'protocol_sha256': M.sha(protocol),
        'source_manifest_sha256': {'valid': M.sha(valid), 'test': M.sha(test)}, 'endpoint_updates': 37, 'stages': stages}
    ledger_path = put(tmp_path / 'ledger.json', ledger)
    paths = [ledger_path, cohort_path, audit_path, valid, test, protocol]
    root = {'schema': 'adaptgns_goop3d_scalar_collection_release_v1', 'issued_by': 'root', 'status': 'approved_stopped_scalar_collection',
            'collector_sha256': M.sha(M.__file__), 'files_sha256': {str(p): M.sha(p) for p in paths}}
    return paths, root


def test_collector_preserves_all30_never_started_stages(tmp_path, monkeypatch):
    paths, release = fixture(tmp_path, monkeypatch); value = M.collect(*paths, release)
    assert len(value['stages']) == 30 and value['endpoint_updates'] == 37
    assert all(not s['rows'] for s in value['stages'])
    assert value['physical_reference']['dimension'] == 3


def test_collector_reads_scalar_row_and_hashes_opaque_artifact_only(tmp_path, monkeypatch):
    paths, release = fixture(tmp_path, monkeypatch, completed_row=True); value = M.collect(*paths, release)
    assert len(value['stages'][0]['rows']) == 1
    assert value['stages'][0]['aggregate_snapshot']['state'] == 'absent'
    assert 'opaque NPZ' in value['numeric_validation_scope']


@pytest.mark.parametrize('bad', ['unreaped', 'unverified', 'endpoint', 'duplicate_stage', 'duplicate_cell', 'artifact', 'row_hash', 'never_started_row', 'short_clock', 'D2_protocol', 'capacity_purpose', 'release_path', 'duplicate_flag', 'cohort_path', 'truth_metadata'])
def test_collector_rejects_changed_or_wrong_scope_outputs(tmp_path, monkeypatch, bad):
    paths, release = fixture(tmp_path, monkeypatch, completed_row=True); ledger = M.read(paths[0]); stage = ledger['stages'][0]
    if bad == 'unreaped': ledger['unreaped_owned_children'] = [123]
    if bad == 'unverified': ledger['all_pinned_inputs_reverified'] = False
    if bad == 'endpoint': ledger['endpoint_updates'] = 512
    if bad == 'duplicate_stage': ledger['stages'][-1] = stage
    if bad == 'duplicate_cell': stage['cells'][-1] = stage['cells'][0]
    if bad == 'artifact': (Path(stage['directory']) / 'trajectory_000000_base.npz').write_bytes(b'changed')
    if bad == 'row_hash': stage['cells'][0]['row_sha256'] = 'f' * 64
    if bad == 'never_started_row': stage['outcome'] = {'started': False, 'state': 'never_started', 'pid': None, 'termination_reason': 'synthetic'}
    if bad == 'short_clock': stage['outcome']['elapsed_seconds'] = 1.
    if bad == 'D2_protocol':
        pp = Path(stage['directory']) / 'protocol.json'; prot = M.read(pp); prot['horizon'] = 395; put(pp, prot)
    if bad == 'capacity_purpose':
        stage['command'][stage['command'].index('--purpose') + 1] = 'capacity_timing'
    if bad == 'release_path': stage['command'][stage['command'].index('--release') + 1] = str(tmp_path / 'unrelated.json')
    if bad == 'duplicate_flag': stage['command'] += ['--purpose', 'final_evaluation']
    if bad == 'cohort_path': stage['command'][stage['command'].index('--cohort') + 1] = str(tmp_path / 'remapped_cohort.json')
    if bad == 'truth_metadata':
        m = M.read(paths[4]); m['metadata']['bounds'][0][0] = .1; put(paths[4], m)
        release['files_sha256'][str(paths[4])] = M.sha(paths[4])
    put(paths[0], ledger); release['files_sha256'][str(paths[0])] = M.sha(paths[0])
    with pytest.raises(ValueError): M.collect(*paths, release)


def test_stale_valid_full_result_prefix_is_preserved(tmp_path, monkeypatch):
    paths, release = fixture(tmp_path, monkeypatch, completed_row=True); stage = M.read(paths[0])['stages'][0]; directory = Path(stage['directory'])
    put(directory / 'result.json', {'schema': 'adaptgns_goop3d_graph_support_evaluation_v1',
        'protocol_sha256': M.sha(directory / 'protocol.json'), 'rows': [], 'expected_outcomes': 6})
    value = M.collect(*paths, release)
    assert value['stages'][0]['aggregate_snapshot'] == {'state': 'stale_valid_prefix', 'recorded_rows': 0, 'committed_rows': 1}


def test_all_missing_analysis_keeps_nulls_and_explicit_population(tmp_path, monkeypatch):
    paths, release = fixture(tmp_path, monkeypatch); collection = M.collect(*paths, release)
    cp = put(tmp_path / 'collection.json', collection)
    root = {'schema': 'adaptgns_goop3d_scalar_analysis_release_v1', 'issued_by': 'root', 'status': 'approved_fixed_scalar_aggregation',
            'summarizer_sha256': M.sha(M.__file__), 'collection_sha256': M.sha(cp), 'cohort_sha256': collection['cohort_sha256'], 'endpoint_updates': 37}
    value = M.summarize(cp, root)
    assert len(value['required_cell_accounting']) == 30
    assert value['stages']['full_rollout_test']['required_outcomes'] == 36
    assert value['stages']['full_rollout_test']['paired']['absolute']['mean_rollout_mse']['base']['base']['mean'] is None
    assert value['stages']['full_rollout_test']['paired']['absolute']['predicted_boundary_trajectory_maximum_excursion']['base']['base']['mean'] is None
    assert value['stages']['full_rollout_test']['paired']['absolute']['graph_fraction_steps_cap_active']['base']['base']['mean'] is None
    assert 'Goop2D100k' in value['scope']


def test_failed_committed_row_is_retained_and_failure_ledger_agrees(tmp_path, monkeypatch):
    paths, release = fixture(tmp_path, monkeypatch, completed_row=True); ledger = M.read(paths[0]); stage = ledger['stages'][0]
    path = Path(stage['directory']) / stage['cells'][0]['row_file']; previous = M.read(path)
    row = rollout('base', 0, M.cell_schedule(manifest(), 'full-rollout')[0], completed=200)
    row.update({k: previous[k] for k in ('protocol_sha256', 'artifact_file', 'artifact_sha256')}); put(path, row)
    stage['cells'][0].update(state='recorded_failed_outcome', failure=row['failure'], row_sha256=M.sha(path))
    put(paths[0], ledger); release['files_sha256'][str(paths[0])] = M.sha(paths[0])
    value = M.collect(*paths, release)
    assert value['stages'][0]['rows'][0]['completed_steps'] == 200
    assert value['stages'][0]['rows'][0]['mean_rollout_mse'] is None
    stage['cells'][0]['failure'] = {'category': 'different_guard'}
    put(paths[0], ledger); release['files_sha256'][str(paths[0])] = M.sha(paths[0])
    with pytest.raises(ValueError, match='failure'): M.collect(*paths, release)


def test_same_state_paired_model_independent_graphs_checked():
    D = M.load_scalar('goop3d_diagnostic_metrics_v1.py'); schedule = M.expected(manifest(), 'same-state')
    stages = []
    for arm, seed in M.MODELS:
        rows = [diagnostic_row(arm, seed, item=r) for r in schedule]
        stages.append({'arm': arm, 'seed': seed, 'stage': 'same_state_valid', 'rows': rows, 'outcome': {},
            'cells': [{**r, 'state': 'completed_required_outcome'} for r in schedule],
            'diagnostic_summary': D.summarize(schedule, rows, 'same-state')})
    value = M.diagnostic_summary(stages, schedule, 'same-state')
    assert len(value['paired_observed_graph_selection']) == 75
    assert all(x['state'] == 'same_graph_and_draw_verified' for x in value['paired_observed_graph_selection'])
    assert value['accuracy_policy_contrasts']['within_arm_policy_contrasts']['position_coordinate_mse']['base']['previous-observed-base-risk25_minus_random25']['mean'] == .5
    stages[-1]['rows'][0]['policies']['random25']['graph']['selected_optional_pair_sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='model-independent'): M.diagnostic_summary(stages, schedule, 'same-state')


def test_description_does_not_touch_data(capsys):
    assert M.main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['status'] == 'description_only' and out['scientific_endpoint_selected'] is False


def test_collection_output_cannot_modify_stopped_stage(tmp_path, monkeypatch):
    paths, release = fixture(tmp_path, monkeypatch, completed_row=True)
    rp = put(tmp_path / 'root_collection_release.json', release)
    stage = M.read(paths[0])['stages'][0]
    argv = ['--execute', '--mode', 'collect', '--root-release', str(rp), '--output', str(Path(stage['directory']) / 'new_analysis.json')]
    for flag, path in zip(('ledger', 'cohort', 'cohort-audit', 'valid-manifest', 'test-manifest', 'protocol'), paths): argv += ['--' + flag, str(path)]
    with pytest.raises(ValueError, match='outside stopped'): M.main(argv)


def analysis_fixture(tmp_path, monkeypatch):
    paths, release = fixture(tmp_path, monkeypatch)
    collection = M.collect(*paths, release); cp = put(tmp_path / 'collection.json', collection)
    root = {'schema': 'adaptgns_goop3d_scalar_analysis_release_v1', 'issued_by': 'root', 'status': 'approved_fixed_scalar_aggregation',
            'summarizer_sha256': M.sha(M.__file__), 'collection_sha256': M.sha(cp), 'cohort_sha256': collection['cohort_sha256'], 'endpoint_updates': 37}
    return cp, root


@pytest.mark.parametrize('bad', ['row', 'new_file', 'new_directory', 'child_release', 'ledger', 'executed_source_alias'])
def test_collection_terminal_snapshot_rejects_changes_after_row_validation(tmp_path, monkeypatch, bad):
    paths, release = fixture(tmp_path, monkeypatch, completed_row=True)
    ledger = M.read(paths[0]); stage = ledger['stages'][0]; directory = Path(stage['directory'])
    source_alias = tmp_path / 'executed_source' / 'evaluate_goop3d_graph_support_v1.py'
    if bad == 'executed_source_alias':
        source_alias.parent.mkdir(); source_alias.write_bytes((HERE / source_alias.name).read_bytes())
        stage['command'][1] = str(source_alias)
        stage['outcome']['command'][1] = str(source_alias)
        put(paths[0], ledger); release['files_sha256'][str(paths[0])] = M.sha(paths[0])
    original = M.validate_rollout; changed = []
    def mutate_after_validation(*args):
        result = original(*args)
        if not changed:
            if bad == 'row':
                path = directory / stage['cells'][0]['row_file']; path.write_text(path.read_text() + '\n')
            if bad == 'new_file': (directory / 'new_unaccounted.log').write_text('late synthetic bytes')
            if bad == 'new_directory': (directory / 'new_empty_directory').mkdir()
            if bad == 'child_release':
                path = Path(stage['release_file']); path.write_text(path.read_text() + '\n')
            if bad == 'ledger': paths[0].write_text(paths[0].read_text() + '\n')
            if bad == 'executed_source_alias': source_alias.write_text(source_alias.read_text() + '\n')
            changed.append(True)
        return result
    monkeypatch.setattr(M, 'validate_rollout', mutate_after_validation)
    with pytest.raises(ValueError, match='changed'): M.collect(*paths, release)
    assert changed


def test_collection_bytes_cannot_change_during_aggregation(tmp_path, monkeypatch):
    cp, release = analysis_fixture(tmp_path, monkeypatch)
    original = M.full_summary; changed = []
    def mutate_after_summary(*args):
        result = original(*args)
        if not changed:
            collection = M.read(cp); collection['status'] = 'not_approved_collection'; put(cp, collection); changed.append(True)
        return result
    monkeypatch.setattr(M, 'full_summary', mutate_after_summary)
    with pytest.raises(ValueError, match='changed during aggregation'): M.summarize(cp, release)
    assert changed


@pytest.mark.parametrize('mode', ['collect', 'summarize'])
@pytest.mark.parametrize('bad', ['own_source', 'root_release', 'bound_input'])
def test_cli_checks_source_release_and_inputs_after_serialization(tmp_path, monkeypatch, mode, bad):
    own_copy = tmp_path / 'synthetic_collector_source.py'; own_copy.write_bytes(Path(M.__file__).read_bytes())
    monkeypatch.setattr(M, '__file__', str(own_copy))
    output = tmp_path / 'fresh_output.json'; argv = ['--execute', '--mode', mode, '--output', str(output)]
    if mode == 'collect':
        paths, release = fixture(tmp_path, monkeypatch, completed_row=True)
        for flag, path in zip(('ledger', 'cohort', 'cohort-audit', 'valid-manifest', 'test-manifest', 'protocol'), paths): argv += ['--' + flag, str(path)]
        stage = M.read(paths[0])['stages'][0]; bound = Path(stage['directory']) / stage['cells'][0]['row_file']
    else:
        bound, release = analysis_fixture(tmp_path, monkeypatch); argv += ['--collection', str(bound)]
    rp = put(tmp_path / 'root_release.json', release); argv += ['--root-release', str(rp)]
    target = {'own_source': own_copy, 'root_release': rp, 'bound_input': bound}[bad]
    original = M.json.dumps; changed = []
    def mutate_after_serialization(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(value, dict) and 'root_release_sha256' in value and not changed:
            target.write_bytes(target.read_bytes() + b'\n'); changed.append(True)
        return result
    monkeypatch.setattr(M.json, 'dumps', mutate_after_serialization)
    with pytest.raises(ValueError, match='changed'): M.main(argv)
    assert changed and not output.exists()


@pytest.mark.parametrize('mode', ['collect', 'summarize'])
def test_cli_success_publishes_captured_release_hash(tmp_path, monkeypatch, mode):
    output = tmp_path / 'fresh_output.json'; argv = ['--execute', '--mode', mode, '--output', str(output)]
    if mode == 'collect':
        paths, release = fixture(tmp_path, monkeypatch)
        for flag, path in zip(('ledger', 'cohort', 'cohort-audit', 'valid-manifest', 'test-manifest', 'protocol'), paths): argv += ['--' + flag, str(path)]
    else:
        cp, release = analysis_fixture(tmp_path, monkeypatch); argv += ['--collection', str(cp)]
    rp = put(tmp_path / 'root_release.json', release); argv += ['--root-release', str(rp)]
    assert M.main(argv) == 0
    assert M.read(output)['root_release_sha256'] == M.sha(rp)


@pytest.mark.parametrize('bad', ['graph_reference', 'missing_truth', 'missing_benefit', 'missing_correlation',
                                'unknown_correlation', 'dense_fraction_range', 'dense_fraction_subset', 'missing_dense_fraction'])
def test_same_state_requires_complete_bound_scalar_grids(bad):
    item = M.expected(manifest(), 'same-state')[0]; row = diagnostic_row(item=item)
    if bad == 'graph_reference':
        row['policies']['base']['graph'] = {**row['policies']['base']['graph'], 'directed_edge_sha256': 'f' * 64}
    if bad == 'missing_truth': row.pop('truth_boundary')
    if bad == 'missing_benefit': row['benefit'].pop('dense')
    if bad == 'missing_correlation': row['correlations'].pop('previous_risk_vs_base_error')
    if bad == 'unknown_correlation': row['correlations']['invented'] = {'value': .2, 'reason': None}
    if bad == 'dense_fraction_range': row['benefit']['random25'][M.DENSE_FRACTIONS[0]] = 1.5
    if bad == 'dense_fraction_subset': row['benefit']['random25'][M.DENSE_FRACTIONS[1]] = .5
    if bad == 'missing_dense_fraction': row['benefit']['random25'].pop(M.DENSE_FRACTIONS[0])
    with pytest.raises(ValueError): M.validate_diagnostic(row, 'base', 0, item, 'valid', 'same-state', 4)


def failed_policy_row():
    item = M.expected(manifest(), 'same-state')[0]; row = diagnostic_row(item=item)
    row['status'] = 'failed'; row['failure'] = {'category': 'policy_guard_or_repeat_failure'}
    method = 'random25'; record = row['policies'][method]
    record.update(status='failed', repeat_consistent=False, metrics=None, graph=None, prediction_boundary=None)
    row['benefit'].pop(method)
    for key in ('previous_risk_vs_random25_benefit', 'dense_vs_random25_benefit'): row['correlations'].pop(key)
    next(c for c in row['timed_calls'] if c['method'] == method and c['round'] == 1)['repeat_consistency']['ordered_edges_exact'] = False
    return item, row


@pytest.mark.parametrize('field', ['metrics', 'graph', 'prediction_boundary'])
def test_failed_policy_cannot_supply_stale_aggregate_values(field):
    item, row = failed_policy_row(); row['policies']['random25'][field] = diagnostic_row(item=item)['policies']['random25'][field]
    with pytest.raises(ValueError, match='remain null'): M.validate_diagnostic(row, 'base', 0, item, 'valid', 'same-state', 4)


def test_nullable_failed_policy_graph_boundary_remain_missing_in_summary():
    D = M.load_scalar('goop3d_diagnostic_metrics_v1.py'); item, row = failed_policy_row(); schedule = [item]
    M.validate_diagnostic(row, 'base', 0, item, 'valid', 'same-state', 4)
    stages = []
    for arm, seed in M.MODELS:
        r = copy.deepcopy(row); r.update(arm=arm, training_seed=seed); r['random_seed_material'][2] = seed
        stages.append({'arm': arm, 'seed': seed, 'stage': 'same_state_valid', 'rows': [r], 'outcome': {},
                       'cells': [{**item, 'state': 'recorded_failed_outcome'}], 'diagnostic_summary': D.summarize(schedule, [r], 'same-state')})
    value = M.diagnostic_summary(stages, schedule, 'same-state')['absolute']['base']
    assert value['graph/random25/directed_edges']['mean'] is None
    assert value['boundary/random25/maximum_coordinate_excursion']['mean'] is None


def test_failed_clean_metrics_must_be_null():
    item = M.expected(manifest(), 'clean-validation')[0]
    row = {**item, 'arm': 'base', 'training_seed': 0, 'objective': 'faithful', 'status': 'failed',
           'failure': {'category': 'synthetic'}, 'synchronized_call_seconds': 1., 'publication_seconds': .1, 'metrics': None}
    M.validate_diagnostic(row, 'base', 0, item, 'valid', 'clean-validation', 4)
    row['metrics'] = {'normalized_acceleration_coordinate_mse': 2.}
    with pytest.raises(ValueError, match='remain null'): M.validate_diagnostic(row, 'base', 0, item, 'valid', 'clean-validation', 4)


@pytest.mark.parametrize('seed', [0., False])
def test_exact_integer_diagnostic_seed(seed):
    item = M.expected(manifest(), 'same-state')[0]; row = diagnostic_row(item=item); row['training_seed'] = seed
    with pytest.raises(ValueError, match='identity'): M.validate_diagnostic(row, 'base', 0, item, 'valid', 'same-state', 4)


@pytest.mark.parametrize('target', ['cohort', 'stage', 'audit_model', 'audit_pair'])
def test_exact_integer_collection_seed(tmp_path, monkeypatch, target):
    paths, release = fixture(tmp_path, monkeypatch)
    index = 1 if target == 'cohort' else 0 if target == 'stage' else 2
    value = M.read(paths[index]); key = 'stages' if target == 'stage' else 'paired_seeds' if target == 'audit_pair' else 'models'
    value[key][0]['seed'] = 0.; put(paths[index], value)
    # Refresh all synthetic input bindings so type validation, not a stale hash,
    # rejects the semantically invalid cohort/receipt.
    if index == 2:
        cohort = M.read(paths[1]); cohort['cohort_audit_sha256'] = M.sha(paths[2]); put(paths[1], cohort)
    ledger = M.read(paths[0]); ledger.update(cohort_sha256=M.sha(paths[1]), cohort_audit_sha256=M.sha(paths[2])); put(paths[0], ledger)
    for path in paths: release['files_sha256'][str(path)] = M.sha(path)
    with pytest.raises(ValueError, match='integer seed'): M.collect(*paths, release)
