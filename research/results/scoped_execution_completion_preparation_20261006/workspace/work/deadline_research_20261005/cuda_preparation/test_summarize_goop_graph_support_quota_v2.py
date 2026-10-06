"""Synthetic scalar receipts and byte placeholders only; no scientific data/GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('goop_scalar_summary', HERE / 'summarize_goop_graph_support_quota_v2.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
B = {'fraction_particles_outside': 0., 'fraction_particles_outside_by_more_than_1e-6': 0.,
     'maximum_coordinate_excursion': 0., 'mean_particle_maximum_excursion': 0., 'coordinate_minimum': [.2, .2], 'coordinate_maximum': [.8, .8]}


def row(arm, seed, source, policy, value=1., completed=395):
    mse = [float(value)] * completed
    complete = completed == 395
    return {'arm': arm, 'training_seed': seed, 'objective': 'faithful', 'source_index': source, 'policy': policy,
            'horizon': 395, 'completed_steps': completed, 'status': 'complete' if complete else 'failed',
            'failure': None if complete else {'category': 'coordinate_guard', 'phase': 'predicted_state'},
            'mse_per_step': mse, 'mean_rollout_mse': float(value) if complete else None,
            'mse_at_final_horizon': float(value) if complete else None,
            'mse_at_declared_trace_steps': {str(s): float(value) if s <= completed else None for s in (1, 10, 50, 200, 395)},
            'synchronized_call_seconds': 2., 'trace_publication_seconds': .1,
            'predicted_boundary_per_step': [B] * completed, 'ground_truth_boundary_per_step': [B] * completed}


def complete_stages():
    result = []
    for arm in ('base', 'mix'):
        for seed in range(3):
            rows, cells = [], []
            for source, policy in M.expected('full_rollout_test'):
                value = (10 if arm == 'base' else 20) + (2 if arm == 'base' else 1) * (policy == 'laggedrisk25') + seed
                rows.append(row(arm, seed, source, policy, value))
                cells.append({'source_index': source, 'policy': policy, 'state': 'completed_required_outcome'})
            result.append({'arm': arm, 'seed': seed, 'stage': 'full_rollout_test', 'rows': rows, 'cells': cells,
                           'outcome': {'whole_invocation_seconds': 100., 'exit_code': 0}})
    return result


def test_exact_paired_interaction_not_absolute_superiority():
    result = M.full_summary(complete_stages())
    interaction = result['risk_minus_random_mix_minus_base_interaction']['mean_rollout_mse']
    assert interaction['mean'] == -1 and interaction['sample_sd'] == 0
    assert result['absolute']['mean_rollout_mse']['base']['laggedrisk25']['mean'] == 13
    assert result['absolute']['mean_rollout_mse']['mix']['laggedrisk25']['mean'] == 22
    assert result['mix_minus_base']['mean_rollout_mse']['random25']['mean'] == 10
    assert result['within_arm_policy_contrasts']['mean_rollout_mse']['mix']['laggedrisk25_minus_random25']['mean'] == 1
    assert result['coverage']['completed_required_outcome'] == 1080


def test_missing_required_policy_null_not_survivor_mean():
    stages = complete_stages(); stage = stages[0]
    stage['rows'] = [r for r in stage['rows'] if (r['source_index'], r['policy']) != (7, 'random25')]
    next(c for c in stage['cells'] if (c['source_index'], c['policy']) == (7, 'random25'))['state'] = 'timed_out_current'
    result = M.full_summary(stages)
    value = result['risk_minus_random_mix_minus_base_interaction']['mean_rollout_mse']
    assert value['mean'] is None and value['sample_sd'] is None and value['defined_seed_pairs'] == 2
    assert result['absolute']['mean_rollout_mse']['base']['random25']['mean'] is None
    assert result['coverage']['timed_out_current'] == 1


def test_unrelated_dense_failure_does_not_destroy_primary():
    stages = complete_stages(); stage = stages[0]
    old = next(r for r in stage['rows'] if (r['source_index'], r['policy']) == (7, 'dense'))
    stage['rows'][stage['rows'].index(old)] = row('base', 0, 7, 'dense', completed=3)
    cell = next(c for c in stage['cells'] if (c['source_index'], c['policy']) == (7, 'dense'))
    cell.update(state='recorded_failed_outcome', failure={'category': 'coordinate_guard'})
    result = M.full_summary(stages)
    assert not result['all_required_outcomes_complete'] and result['all_required_outcomes_recorded']
    assert result['risk_minus_random_mix_minus_base_interaction']['mean_rollout_mse']['mean'] == -1
    assert result['absolute']['mean_rollout_mse']['base']['dense']['mean'] is None
    assert len(result['failed_accepted_prefixes']) == 1


def test_forecast_is_pointwise_not_prefix_mean():
    r = row('base', 0, 0, 'base'); r['mse_per_step'] = list(range(395)); r['mean_rollout_mse'] = 197.; r['mse_at_final_horizon'] = 394
    r['mse_at_declared_trace_steps'] = {str(s): s - 1 for s in (1, 10, 50, 200, 395)}
    metrics, _ = M.validate_rollout(r, 'base', 0)
    assert metrics['mse_forecast200'] == 199 and metrics['mse_forecast395'] == 394
    assert metrics['mean_rollout_mse'] == 197


def test_later_failed_committed_prefix_keeps_forecast200_only():
    r = row('base', 0, 0, 'base', 3., completed=201)
    metrics, prefix = M.validate_rollout(r, 'base', 0)
    assert metrics['mse_forecast200'] == 3 and metrics['mse_forecast395'] is None and metrics['mean_rollout_mse'] is None
    assert metrics['predicted_boundary_mean_fraction_particles_outside'] is None
    assert prefix['predicted_boundary_mean_fraction_particles_outside'] == 0


@pytest.mark.parametrize('bad', ['mse_nan', 'wrong_mean', 'bad_boundary', 'wrong_seed', 'short_complete', 'negative_timing'])
def test_bad_scalar_rejected(bad):
    r = row('base', 0, 0, 'base')
    if bad == 'mse_nan': r['mse_per_step'][0] = float('nan')
    if bad == 'wrong_mean': r['mean_rollout_mse'] = 4
    if bad == 'bad_boundary': r['predicted_boundary_per_step'] = [dict(B, fraction_particles_outside=2.)] * 395
    if bad == 'wrong_seed': r['training_seed'] = 2
    if bad == 'short_complete': r['completed_steps'] = 10
    if bad == 'negative_timing': r['synchronized_call_seconds'] = -2
    with pytest.raises(ValueError): M.validate_rollout(r, 'base', 0)


def test_sample_sd_uses_three_seeds():
    value = M.seed_summary([1, 2, 3])
    assert value['mean'] == 2 and value['sample_sd'] == 1
    with pytest.raises(ValueError): M.seed_summary([1, 2])
    assert M.seed_summary([1, None, 3])['mean'] is None


def test_duplicate_row_and_changed_truth_rejected():
    stages = complete_stages(); stages[0]['rows'].append(stages[0]['rows'][0])
    with pytest.raises(ValueError, match='Duplicate'): M.full_summary(stages)
    stages = complete_stages(); r = stages[0]['rows'][1]
    r['ground_truth_boundary_per_step'] = [dict(B, coordinate_minimum=[.21, .21])] * 395
    with pytest.raises(ValueError, match='truth differs'): M.full_summary(stages)


def test_diagnostic_risk_estimand_and_missing_seed():
    evaluator = M.import_frozen('_diag_for_test', HERE / 'evaluate_goop_graph_support_final.py', M.EVALUATOR_SHA)
    stages = []
    schedule = [{'source_index': i, 'target_frame': t} for i, t in M.expected('same_state_test')]
    for arm in ('base', 'mix'):
        for seed in range(3):
            delta = 2. if arm == 'base' else 1.
            rows = [dict(item, status='complete', policies={
                'previous-observed-base-risk25': {'metrics': {'position_coordinate_mse': 10. + delta}},
                'random25': {'metrics': {'position_coordinate_mse': 10.}}}) for item in schedule]
            summary = evaluator.summarize(schedule, rows, 'same-state')
            stages.append({'arm': arm, 'seed': seed, 'stage': 'same_state_test', 'diagnostic_summary': summary,
                           'cells': [dict(item, state='completed_required_outcome') for item in schedule], 'outcome': {}})
    result = M.diagnostic_summary(stages, 'same_state_test')
    assert result['risk_minus_random_mix_minus_base_interaction']['mean'] == -1
    stages[0]['diagnostic_summary']['accuracy']['random25']['position_coordinate_mse']['equal_trajectory_mean'] = None
    assert M.diagnostic_summary(stages, 'same_state_test')['risk_minus_random_mix_minus_base_interaction']['mean'] is None


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2) + '\n'); return path


@pytest.fixture
def synthetic_queue(tmp_path):
    root = tmp_path / 'queue'; root.mkdir()
    cohort = {'schema': 'adaptgns_goop_graph_support_final_cohort_v1', 'status': 'frozen_for_final_evaluation',
              'protocol_sha256': M.PROTOCOL_SHA, 'updates': 100000,
              'models': [{'arm': a, 'seed': s, 'checkpoint_sha256': str(s + 1 if a == 'base' else s + 4) * 64}
                         for a in ('base', 'mix') for s in range(3)]}
    cohort_path = put(tmp_path / 'cohort.json', cohort)
    streams, entries = [], []
    quota = M.import_frozen('_quota_test_fixture', HERE / 'supervise_graph_support_evaluation_quota_v2.py', M.QUOTA_SHA)
    file_pins = {'/fake/cohort': M.sha(cohort_path), '/fake/protocol': M.PROTOCOL_SHA,
                 '/fake/trainer': M.TRAINER_SHA, '/fake/benchmark': M.BENCH_SHA, '/fake/evaluator': M.EVALUATOR_SHA}
    for arm, seed, gpu in quota.SCHEDULE['B']:
        ident = f'{arm}_seed{seed}'; commands = []
        cp_sha = next(r['checkpoint_sha256'] for r in cohort['models'] if (r['arm'], r['seed']) == (arm, seed))
        file_pins['/fake/' + ident] = cp_sha
        for stage, mode, split in M.STAGES:
            directory = root / 'jobs' / ident / stage
            opts = {'--mode': mode, '--split': split, '--arm': arm, '--seed': str(seed), '--checkpoint-sha256': cp_sha,
                    '--cohort': '/fake/cohort', '--protocol': '/fake/protocol', '--trainer-source': '/fake/trainer',
                    '--benchmark-helper': '/fake/benchmark', '--checkpoint': '/fake/' + ident}
            command = ['/fake/python', '/fake/evaluator', '--execute'] + [v for kv in opts.items() for v in kv]
            commands.append(command)
            outcome = {'state': 'never_started'}
            if (arm, stage) == ('base', 'full_rollout_test'):
                directory.mkdir(parents=True)
                protocol = {'schema': M.E_SCHEMA, 'mode': mode, 'split': split, 'source_frame_count': 401,
                    'model': {'kind': 'preselected_checkpoint', 'completed_updates': 100000, 'arm': arm, 'seed': seed, 'checkpoint_sha256': cp_sha},
                    'input_files_sha256': {opts[k]: file_pins[opts[k]] for k in ('--cohort', '--protocol', '--trainer-source', '--benchmark-helper', '--checkpoint')},
                    'schedule': [{'source_index': i} for i in range(30)], 'policies': list(M.POLICIES)}
                pp = put(directory / 'protocol.json', protocol)
                artifact = directory / 'trajectory_000000_base.npz'; artifact.write_bytes(b'placeholder byte audit only')
                r = row(arm, seed, 0, 'base'); r.update(protocol_sha256=M.sha(pp), trace_file=artifact.name, trace_sha256=M.sha(artifact))
                put(artifact.with_suffix('.json'), r)
                put(directory / 'result.json', {'schema': M.E_SCHEMA, 'protocol_sha256': M.sha(pp), 'rows': [r]})
                outcome = {'state': 'exited', 'quota_expired': True, 'current_before_stop': {'source_index': 0, 'policy': 'dense'}}
            entry = {'stream': ident, 'stage': stage, 'outcome': outcome, 'coverage_audit_state': 'row_coverage_checked',
                     **quota.account_stage('Goop', stage, directory, outcome)}
            entries.append(entry)
        streams.append({'id': ident, 'arm': arm, 'seed': seed, 'gpu': gpu, 'commands': commands})
    put(root / 'release_snapshot.json', {'schema': 'adaptgns_graph_support_evaluation_quota_release_v2', 'issued_by': 'root',
                                       'dataset': 'Goop', 'host_role': 'B', 'streams': streams, 'files_sha256': file_pins})
    ledger = put(root / 'coverage_ledger.json', {'schema': M.Q_SCHEMA, 'dataset': 'Goop', 'stages': entries,
                                              'unreaped_owned_children': [], 'audit_errors': [], 'abort_reason': None})
    put(root / 'process_outcomes.json', {'schema': M.Q_SCHEMA, 'unreaped_owned_children': []})
    put(root / 'queue_status.json', {'schema': M.Q_SCHEMA, 'state': 'allocation_finished',
                                   'coverage_ledger_sha256': M.sha(ledger), 'unreaped_owned_children': [], 'all_pinned_inputs_reverified': True})
    return root, cohort_path


def test_collector_preserves_quota_missing_and_binary_hashes(synthetic_queue):
    root, cohort = synthetic_queue
    result = M.collect(root, cohort)
    assert len(result['stages']) == 8
    stage = result['stages'][0]
    assert len(stage['rows']) == 1 and stage['cells'][1]['state'] == 'timed_out_current'
    assert result['files']['jobs/base_seed2/full_rollout_test/trajectory_000000_base.npz']['bytes'] > 0
    assert result['stages'][1]['diagnostic_summary']['accuracy']['base']['position_coordinate_mse']['equal_trajectory_mean'] is None


def test_collector_rejects_changed_array_bytes(synthetic_queue):
    root, cohort = synthetic_queue
    (root / 'jobs/base_seed2/full_rollout_test/trajectory_000000_base.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact bytes'): M.collect(root, cohort)


def test_collector_rejects_changed_ledger_row(synthetic_queue):
    root, cohort = synthetic_queue
    path = root / 'jobs/base_seed2/full_rollout_test/trajectory_000000_base.json'
    r = M.read(path); r['mean_rollout_mse'] = 7; put(path, r)
    with pytest.raises(ValueError, match='coverage differs'): M.collect(root, cohort)


def test_exact_stale_rollout_prefix_is_recovered_from_committed_rows(synthetic_queue):
    root, cohort = synthetic_queue
    path = root / 'jobs/base_seed2/full_rollout_test/result.json'
    result = M.read(path); result['rows'] = []; put(path, result)
    collected = M.collect(root, cohort)
    stage = collected['stages'][0]
    assert len(stage['rows']) == 1 and stage['aggregate_snapshot']['state'] == 'stale_valid_prefix'


def test_invalid_rollout_snapshot_is_rejected(synthetic_queue):
    root, cohort = synthetic_queue
    path = root / 'jobs/base_seed2/full_rollout_test/result.json'
    result = M.read(path); result['rows'][0]['mean_rollout_mse'] = 123.; put(path, result)
    with pytest.raises(ValueError, match='exact committed prefix'): M.collect(root, cohort)


def test_unverified_inputs_retained_but_scientific_summary_refused(synthetic_queue, tmp_path):
    root, cohort = synthetic_queue
    path = root / 'queue_status.json'; status = M.read(path); status['all_pinned_inputs_reverified'] = False; put(path, status)
    collected = M.collect(root, cohort)
    collected['host_role'] = 'A'
    collection = put(tmp_path / 'collection.json', collected)
    release = {'schema': 'adaptgns_goop_paired_scalar_analysis_release_quota_v2', 'issued_by': 'root',
        'status': 'approved_for_fixed_scalar_aggregation', 'summarizer_sha256': M.sha(M.__file__),
        'collection_sha256': {'A': M.sha(collection)}, 'cohort_sha256': M.sha(cohort)}
    with pytest.raises(ValueError, match='Input integrity'): M.summarize({'A': collection}, release)


def test_exact_stale_diagnostic_prefix_is_recovered(synthetic_queue):
    root, cohort = synthetic_queue
    directory = root / 'jobs/base_seed2/same_state_valid'; directory.mkdir()
    original = M.read(root / 'jobs/base_seed2/full_rollout_test/protocol.json')
    schedule = [{'source_index': i, 'target_frame': t} for i, t in M.expected('same_state_valid')]
    protocol = {**original, 'mode': 'same-state', 'split': 'valid', 'policies': list(M.DIAG_POLICIES), 'schedule': schedule}
    pp = put(directory / 'protocol.json', protocol)
    artifact = directory / 'trajectory_000000_target_007.npz'; artifact.write_bytes(b'synthetic numeric placeholder')
    r = {'source_index': 0, 'target_frame': 7, 'status': 'complete', 'policies': {}, 'protocol_sha256': M.sha(pp),
         'artifact_file': artifact.name, 'artifact_sha256': M.sha(artifact)}
    put(artifact.with_suffix('.json'), r)
    evaluator = M.import_frozen('_stale_diag_test', HERE / 'evaluate_goop_graph_support_final.py', M.EVALUATOR_SHA)
    put(directory / 'summary.json', evaluator.summarize(schedule, [], 'same-state'))
    quota = M.import_frozen('_stale_diag_quota', HERE / 'supervise_graph_support_evaluation_quota_v2.py', M.QUOTA_SHA)
    ledger_path = root / 'coverage_ledger.json'; ledger = M.read(ledger_path)
    entry = next(e for e in ledger['stages'] if e['stream'] == 'base_seed2' and e['stage'] == 'same_state_valid')
    outcome = {'state': 'exited', 'quota_expired': True, 'current_before_stop': {'source_index': 0, 'target_frame': 105}}
    entry.update(outcome=outcome, **quota.account_stage('Goop', 'same_state_valid', directory, outcome)); put(ledger_path, ledger)
    status_path = root / 'queue_status.json'; status = M.read(status_path); status['coverage_ledger_sha256'] = M.sha(ledger_path); put(status_path, status)
    result = M.collect(root, cohort)
    assert result['stages'][1]['aggregate_snapshot']['state'] == 'stale_valid_prefix'
    assert result['stages'][1]['diagnostic_summary']['returned_frames'] == 1
