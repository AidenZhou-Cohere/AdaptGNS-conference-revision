"""Pure synthetic Sand product metadata tests; no scientific IO or imports."""
from collections import Counter
import copy
import hashlib
import json
import unittest

import review_products as V


def pin(label): return hashlib.sha256(('synthetic-only:' + label).encode()).hexdigest()
def seed(values=None):
    values = [None] * 3 if values is None else values
    return {'seed_values': {str(i): x for i, x in enumerate(values)}, 'required_seed_pairs': 3,
            'defined_seed_pairs': sum(x is not None for x in values), 'mean': None, 'sample_sd': None}
def units(name):
    if name == 'full_rollout_test': return [(i, p) for i in range(30) for p in ('base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25')]
    if name == 'clean_validation': return [(n // 314, n % 314 + 6) for n in (j * (30 * 314 - 1) // 127 for j in range(128))]
    return [(i, t) for i in range(30) for t in (7, 85, 163, 241, 319)]
def place(tree, key, value):
    parts = key.split('/')
    for item in parts[:-1]: tree = tree.setdefault(item, {})
    tree[parts[-1]] = value
def diag(stage):
    name = stage['stage']; wanted = units(name); rows = stage['rows']
    result = {'expected_frames': len(wanted), 'returned_frames': len(rows),
              'complete_frames': sum(x['status'] == 'complete' for x in rows), 'failed_frames': sum(x['status'] == 'failed' for x in rows)}
    lookup = {(r['source_index'], r['target_frame']): r for r in rows}
    for metric in V.diagnostic_keys(name):
        per = {}
        for source in range(30):
            group = [u for u in wanted if u[0] == source]
            if not group: continue
            n = sum(V._finite(V._row_values(lookup.get(u, {}), metric)) for u in group)
            per[str(source)] = {'expected': len(group), 'defined': n, 'mean': None}
        place(result, metric, {'expected_frames': len(wanted), 'defined_frames': sum(x['defined'] for x in per.values()),
            'expected_trajectories': len(per), 'defined_trajectories': 0, 'equal_trajectory_mean': None, 'trajectories': per})
    return result


def fixtures(with_rows=False):
    collections = {}; collect_expected = {}; audits = {}; audit_expected = {}
    collection_pins = {r: pin(r + ' collection') for r in ('A', 'B')}
    audit_pins = {r: pin(r + ' saved audit') for r in ('A', 'B')}
    for role in ('A', 'B'):
        queue = V.REMOTE + '/sand_final_evaluation_' + role + '_20261006_v1'
        root_pin = pin(role + ' collection release'); release_pin = pin(role + ' evaluation release')
        files = {name: {'sha256': pin(role + '/' + name), 'bytes': 100} for name in ('release_snapshot.json', 'queue_status.json', 'coverage_ledger.json', 'process_outcomes.json')}
        files['release_snapshot.json']['sha256'] = release_pin
        scope = {'owned_indices': list(V.GPUS[role]), 'unassigned_devices': 'observe_without_control', 'timing_scope': 'shared_host_operational_measurement', 'live_training_handoff': False}
        release = {'schema': 'adaptgns_sand_evaluation_gpu_scoped_release_v1', 'issued_by': 'root', 'status': 'admitted_for_execution_allocation',
                   'dataset': 'Sand', 'host_role': role, 'streams': [], 'gpu_scope': scope, 'files_sha256': {V.PREP + '/evaluate_sand_graph_support_final.py': V.SOURCES['evaluate_sand_graph_support_final.py']}}
        ledger = {'schema': V.Q_SCHEMA, 'dataset': 'Sand', 'unreaped_owned_children': [], 'stages': [], 'audit_errors': [], 'abort_reason': 'synthetic stopped fixture'}
        process = {'schema': V.Q_SCHEMA, 'unreaped_owned_children': [], 'streams': [], 'all_children': [], 'gpu_observations_sha256': None,
                   'gpu_scope': scope, 'abort_reason': 'synthetic stopped fixture'}
        status = {'schema': V.Q_SCHEMA, 'state': 'stopped_requires_review', 'unreaped_owned_children': [], 'all_pinned_inputs_reverified': True,
                  'coverage_ledger_sha256': files['coverage_ledger.json']['sha256']}
        stages = []; protocols = {}; row_checks = []; models = []; verified = {}; shared = {}
        for (arm, training_seed), gpu in zip(V.MODELS[role], V.GPUS[role]):
            stream_id = arm + '_seed' + str(training_seed)
            released = {'id': stream_id, 'arm': arm, 'seed': training_seed, 'gpu': gpu, 'commands': []}; release['streams'].append(released)
            stream = {'id': stream_id, 'outcomes': []}; process['streams'].append(stream)
            for name, (mode, split) in V.STAGES.items():
                rel = 'jobs/' + stream_id + '/' + name
                required = {'--cohort': (V.COHORT_PATH, V.COHORT), '--protocol': (V.PREP + '/sand_graph_support_100k_protocol_v1.md', V.PROTOCOL),
                    '--trainer-source': (V.PREP + '/train_sand_graph_support_cuda.py', V.TRAINER), '--benchmark-helper': (V.PREP + '/benchmark_sand_graph_support_rollout.py', V.BENCHMARK),
                    '--checkpoint': (V.REMOTE + '/synthetic_checkpoint_' + stream_id + '.pt', pin('checkpoint ' + stream_id))}
                release['files_sha256'].update({path: value for path, value in required.values()})
                options = {'--mode': mode, '--split': split, '--arm': arm, '--seed': str(training_seed), '--objective': 'faithful',
                           '--cuda-index': str(gpu), '--output-dir': queue + '/' + rel,
                           **{key: path for key, (path, _) in required.items()}, '--checkpoint-sha256': required['--checkpoint'][1]}
                released['commands'].append([V.REMOTE + '/.venv/bin/python', V.PREP + '/evaluate_sand_graph_support_final.py', '--execute'] + [item for pair in options.items() for item in pair])
                cells = [{'source_index': i, 'policy' if mode == 'full-rollout' else 'target_frame': target, 'state': 'never_started', 'missing': True} for i, target in units(name)]
                stage = {'arm': arm, 'seed': training_seed, 'stage': name, 'mode': mode, 'split': split, 'cells': cells,
                         'rows': [], 'outcome': {'stage': name, 'state': 'never_started'}, 'protocol_sha256': None,
                         'aggregate_snapshot': {'state': 'absent', 'recorded_rows': 0, 'committed_rows': 0}, 'diagnostic_summary': None}
                chosen = with_rows and arm == 'base' and ((role == 'A' and training_seed == 1 and name in ('full_rollout_test', 'same_state_valid')) or (role == 'B' and name == 'full_rollout_test'))
                if chosen:
                    n = 201 if role == 'A' else 314; failed = role == 'A'; row_status = 'failed' if failed else 'complete'
                    failure = {'category': 'synthetic_late_guard', 'forecast_step': 202} if mode == 'full-rollout' and failed else {'category': 'synthetic_early_failure'} if failed else None
                    row_file = 'trajectory_000000_base.json' if mode == 'full-rollout' else 'trajectory_000000_target_007.json'
                    artifact = row_file[:-5] + '.npz'; protocol_pin = pin(role + rel + 'protocol')
                    row = {'source_index': 0, 'status': row_status, 'failure': failure, 'protocol_sha256': protocol_pin}
                    detail = {'arm': arm, 'seed': training_seed, 'stage': name, 'unit': [0, 'base' if mode == 'full-rollout' else 7], 'status': row_status, 'failure': failure}
                    if mode == 'full-rollout':
                        trace = [1, 10, 50, 200, 314]
                        row.update(policy='base', arm=arm, training_seed=training_seed, objective='faithful', horizon=314,
                            completed_steps=n, requested_trace_steps=trace, mse_per_step=[1.0] * n,
                            mean_rollout_mse=None if failed else 1.0, mse_at_final_horizon=None if failed else 1.0,
                            mse_at_declared_trace_steps={str(s): 1.0 if s <= n else None for s in trace},
                            predicted_boundary_per_step=[{} for _ in range(n)], ground_truth_boundary_per_step=[{} for _ in range(n)],
                            trace_file=artifact, trace_sha256=pin(role + rel + 'artifact'))
                        support = [s for s in trace if s <= n]
                        detail.update(completed_steps=n, accepted_prefix_boundary={k: None for k in V.FULL_METRICS[3:]} if failed else None,
                            array_recomputed_mse_forecasts=support,
                            array_recomputed_prediction_boundary_forecasts=sorted({i for s in support for i in [s] + list(range(s - 6, s)) if i > 0}),
                            full_and_prefix_MSE_verification='aggregate arithmetic over complete recorded scalar series; unsaved prediction errors not array-recomputed',
                            initial_history_sha256=pin('initial'), truth_hashes={str(s): pin('truth' + str(s)) for s in support})
                    else:
                        row.update(target_frame=7, artifact_file=artifact, artifact_sha256=pin(role + rel + 'artifact'),
                                   policies={'base': {'status': 'failed', 'metrics': None, 'timing': {k: {'mean': None} for k in ('end_to_end_seconds', 'score_generation_seconds', 'score_graph_seconds', 'score_forward_seconds', 'current_graph_and_selection_seconds', 'current_forward_seconds')}}},
                                   correlations={'previous_risk_vs_base_error': {'value': None, 'reason': 'constant'}})
                        detail['array_recomputed_diagnostic_metrics'] = V._diagnostic_supported_keys(row, name)
                    cells[0] = {k: v for k, v in cells[0].items() if k != 'missing'}
                    cells[0].update(state='recorded_failed_outcome' if failed else 'completed_required_outcome', path=queue + '/' + rel + '/' + row_file,
                                    sha256=pin(role + rel + 'row'), worker_status=row_status, failure=failure)
                    stage.update(rows=[row], protocol_sha256=protocol_pin, outcome={'stage': name, 'state': 'exited', 'exit_code': 1 if failed else 0},
                                 aggregate_snapshot={'state': 'stale_valid_prefix', 'recorded_rows': 0, 'committed_rows': 1})
                    stream['outcomes'].append(copy.deepcopy(stage['outcome']))
                    files.update({rel + '/protocol.json': {'sha256': protocol_pin, 'bytes': 100},
                                  rel + '/' + row_file: {'sha256': cells[0]['sha256'], 'bytes': 100},
                                  rel + '/' + artifact: {'sha256': row['trace_sha256' if mode == 'full-rollout' else 'artifact_sha256'], 'bytes': 100}})
                    protocols[rel + '/protocol.json'] = {'schema': 'adaptgns_sand_graph_support_final_evaluation_v1', 'mode': mode, 'split': split,
                        'source_frame_count': 320, 'model': {'kind': 'preselected_checkpoint', 'completed_updates': 100000, 'arm': arm, 'seed': training_seed, 'checkpoint_sha256': required['--checkpoint'][1]},
                        'input_files_sha256': {path: value for path, value in required.values()},
                        'policies': list(V.POLICIES if mode == 'full-rollout' else V.DIAG_POLICIES),
                        'schedule': [{'source_index': i} for i in range(30)] if mode == 'full-rollout' else [{'source_index': i, 'target_frame': t} for i, t in units(name)]}
                    for file in ('protocol.json', row_file, artifact): verified[queue + '/' + rel + '/' + file] = files[rel + '/' + file]['sha256']
                    row_checks.append(detail)
                if name != 'full_rollout_test': stage['diagnostic_summary'] = diag(stage)
                stages.append(stage)
                ledger['stages'].append({'stream': stream_id, 'stage': name, 'coverage_audit_state': 'row_coverage_checked', 'cells': copy.deepcopy(cells), 'outcome': copy.deepcopy(stage['outcome'])})
                keys = [metric + '/' + policy for metric in V.FULL_METRICS for policy in V.POLICIES] if mode == 'full-rollout' else V.diagnostic_keys(name)
                models.append({'arm': arm, 'seed': training_seed, 'stage': name, 'cells': copy.deepcopy(cells), 'coverage': dict(Counter(x['state'] for x in cells)), 'metrics': {key: None for key in keys}})
        observations = None
        if with_rows:
            observations = {'schema': V.Q_SCHEMA, 'observations': [{'validation': 'pending', 'raw_gpu_processes': []}]}
            files['gpu_observations.json'] = {'sha256': pin(role + 'gpu observations'), 'bytes': 100}
            process['gpu_observations_sha256'] = files['gpu_observations.json']['sha256']
        entries = [[name, 'file'] for name in files]
        dirs = {str(V.PurePosixPath(name).parent) for name in files if '/' in name}
        for directory in list(dirs): dirs.update(str(p) for p in V.PurePosixPath(directory).parents if str(p) != '.')
        entries += [[name, 'directory'] for name in dirs]; entries.sort()
        inventory = {'entries': entries, 'files': files}
        inventory_pin = hashlib.sha256((json.dumps(inventory, sort_keys=True, indent=2) + '\n').encode()).hexdigest()
        root = {'schema': 'adaptgns_sand_scalar_collection_release_scoped_v1', 'status': 'approved_for_stopped_scalar_collection', 'issued_by': 'root',
            'cohort_sha256': V.COHORT, 'collector_sha256': V.COLLECTOR, 'source_sha256': copy.deepcopy(V.SOURCES),
            'original_queue_root': queue, 'local_queue_inventory_sha256': inventory_pin, 'queue_release_sha256': release_pin, 'operational_amendment_sha256': V.AMENDMENT}
        collection = {'schema': V.COLLECTION_SCHEMA, 'status': 'stopped_outputs_collected', 'issued_by': 'root', 'host_role': role,
            'collector_sha256': V.COLLECTOR, 'cohort_sha256': V.COHORT, 'protocol_sha256': V.PROTOCOL, 'source_sha256': copy.deepcopy(V.SOURCES),
            'operational_amendment_sha256': V.AMENDMENT, 'timing_scope': 'shared_host_operational_measurement', 'queue_root': queue,
            'original_queue_root': queue, 'stages': stages, 'collection_release_sha256': root_pin, 'queue_release_sha256': release_pin,
            'output_tree_state': inventory, 'files': files, 'local_queue_inventory_sha256': inventory_pin,
            'input_sha256': {**V.SOURCE_INPUTS, V.COHORT_PATH: V.COHORT, V.PREP + '/sand_scoped_operational_amendment_v1.md': V.AMENDMENT,
                V.ANALYSIS + '/controls/' + role + '.collection_release.json': root_pin},
            'queue_release': release, 'queue_status': status, 'process_outcomes': process, 'coverage_audit_errors': [], 'abort_reason': ledger['abort_reason'],
            'gpu_observations': observations if observations is not None else {'schema': V.Q_SCHEMA, 'observations': [], 'observation_file_state': 'absent_before_any_child'},
            'gpu_observation_counts': {'passed': 0, 'pending': 1 if with_rows else 0}, 'gpu_observation_file_state': 'present' if with_rows else 'absent_before_any_child'}
        collections[role] = collection
        collect_expected[role] = copy.deepcopy({'artifact_sha256': collection_pins[role], 'host_role': role, 'root_release_sha256': root_pin, 'root_release': root,
            'queue_release_sha256': release_pin, 'queue_release': release, 'inventory': inventory, 'ledger': ledger, 'queue_status': status,
            'process_outcomes': process, 'gpu_observations': observations, 'protocols': protocols})
        audits[role] = {'schema': 'sand_saved_array_audit_v1', 'status': 'passed_supported_checks', 'audit_revision': 2, 'host_role': role,
            'collection_sha256': collection_pins[role], 'cohort_sha256': V.COHORT, 'checks': 100, 'models': models, 'row_checks': row_checks,
            'verified_input_sha256': verified, 'shared_saved_state_hashes': shared, 'auditor_sha256': V.SAVED,
            'diagnostic_helper_sha256': V.DIAGNOSTIC, 'audit_source_sha256': copy.deepcopy(V.SAVED_SOURCES), 'scope': copy.deepcopy(V.SAVED_SCOPE)}
        audit_expected[role] = {'artifact_sha256': audit_pins[role], 'collection_sha256': collection_pins[role]}
    full = {k: {} for k in ('absolute', 'within_arm_policy_contrasts', 'mix_minus_base', 'risk_minus_random_mix_minus_base_interaction')}
    for metric in V.FULL_METRICS:
        full['absolute'][metric] = {a: {p: seed() for p in V.POLICIES} for a in ('base', 'mix')}
        full['within_arm_policy_contrasts'][metric] = {a: {p + '_minus_' + r: seed() for r in ('base', 'random25') for p in V.POLICIES if p != r} for a in ('base', 'mix')}
        full['mix_minus_base'][metric] = {p: seed() for p in V.POLICIES}
        full['risk_minus_random_mix_minus_base_interaction'][metric] = seed()
    all_stages = [s for col in collections.values() for s in col['stages']]
    full_stages = [s for s in all_stages if s['stage'] == 'full_rollout_test']
    counts = Counter(c['state'] for stage in full_stages for c in stage['cells'])
    full.update(required_outcomes=1080, coverage=dict(counts), all_required_outcomes_recorded=False, all_required_outcomes_complete=False,
        coverage_by_model=[{'arm': s['arm'], 'seed': s['seed'], 'counts': dict(Counter(c['state'] for c in s['cells'])),
            'guard_or_execution_failure_categories': dict(Counter(c['failure']['category'] for c in s['cells'] if c['state'] == 'recorded_failed_outcome'))} for s in full_stages],
        failed_accepted_prefixes=[{**{k: row[k] for k in ('arm', 'seed', 'completed_steps', 'failure', 'accepted_prefix_boundary')}, 'source_index': row['unit'][0], 'policy': row['unit'][1]} for audit in audits.values() for row in audit['row_checks'] if row['stage'] == 'full_rollout_test' and row['status'] == 'failed'])
    diagnostics = {}
    for name in list(V.STAGES)[1:]:
        selected = [s for s in all_stages if s['stage'] == name]
        item = {'absolute': {a: {k: seed() for k in V.diagnostic_keys(name)} for a in ('base', 'mix')},
            'mix_minus_base': {k: seed() for k in V.diagnostic_keys(name)},
            'coverage': {s['arm'] + '_seed' + str(s['seed']): dict(Counter(c['state'] for c in s['cells'])) for s in selected},
            'model_summaries': {s['arm'] + '_seed' + str(s['seed']): copy.deepcopy(s['diagnostic_summary']) for s in selected}}
        if name != 'clean_validation':
            item.update(previous_observed_risk_minus_random_position_mse={a: seed() for a in ('base', 'mix')}, risk_minus_random_mix_minus_base_interaction=seed())
        diagnostics[name] = item
    root_pin = pin('summary release'); summary_pin = pin('summary')
    fields = ('queue_status', 'queue_release_sha256', 'operational_amendment_sha256', 'abort_reason', 'coverage_audit_errors', 'gpu_observations', 'gpu_observation_counts', 'gpu_observation_file_state', 'process_outcomes')
    summary = {'schema': V.SUMMARY_SCHEMA, 'status': 'fixed_scalar_aggregation_complete', 'cohort_sha256': V.COHORT, 'protocol_sha256': V.PROTOCOL,
        'summarizer_sha256': V.COLLECTOR, 'source_sha256': copy.deepcopy(V.SOURCES), 'analysis_release_sha256': root_pin, 'collection_sha256': collection_pins,
        'input_sha256': {**V.SOURCE_INPUTS, V.ANALYSIS + '/controls/summary_release.json': root_pin, **{V.ANALYSIS + '/' + r + '.stopped_collection.json': p for r, p in collection_pins.items()}},
        'full_rollout': full, 'diagnostics': diagnostics, 'queue_accounting': {r: {k: copy.deepcopy(col[k]) for k in fields} for r, col in collections.items()},
        'scope': 'all three training seeds; null when required values missing/failed; no survivor means, p-values, replacement seeds or shortened-horizon substitution'}
    summary_expected = {'artifact_sha256': summary_pin, 'root_release_sha256': root_pin, 'collection_sha256': collection_pins}
    computed = {'full_rollout': copy.deepcopy(full), 'diagnostics': copy.deepcopy(diagnostics)}
    for value in computed['diagnostics'].values(): value.pop('model_summaries')
    paired = {'schema': 'sand_paired_saved_array_audit_v1', 'status': 'passed_supported_checks', 'audit_revision': 2, 'checks': 100,
        'cohort_sha256': V.COHORT, 'summary_sha256': summary_pin, 'verified_paired_scalars': computed, 'scope': V.PAIRED_SCOPE,
        'verified_input_sha256': {**V.PAIRED_SOURCES, **{V.ANALYSIS + '/' + r + '.saved_array_audit.json': p for r, p in audit_pins.items()}, V.ANALYSIS + '/paired_scalar_summary.json': summary_pin}}
    paired_expected = {'artifact_sha256': pin('paired'), 'collection_sha256': collection_pins, 'audit_sha256': audit_pins, 'summary_sha256': summary_pin}
    return collections, collect_expected, audits, audit_expected, summary, summary_expected, paired, paired_expected


class SandProductTests(unittest.TestCase):
    def setUp(self):
        self.collections, self.ce, self.audits, self.ae, self.summary, self.se, self.paired, self.pe = fixtures(with_rows=True)
    def collect(self, role='A'): return V.validate_collection(self.collections[role], self.ce[role]['artifact_sha256'], self.ce[role])
    def saved(self, role='A'): return V.validate_saved_audit(self.audits[role], self.ae[role]['artifact_sha256'], self.ae[role], self.collections[role])
    def summarize(self): return V.validate_summary(self.summary, self.se['artifact_sha256'], self.se, self.collections)
    def paired_audit(self): return V.validate_paired_audit(self.paired, self.pe['artifact_sha256'], self.pe, self.collections, self.summary, self.audits)
    def test_valid_six_operations_with_failed_completed_and_unstarted_cells(self):
        for role, cells in [('A', 2432), ('B', 1216)]:
            self.assertEqual(self.collect(role)['required_cells'], cells); self.assertEqual(self.saved(role)['required_cells'], cells)
        for result in (self.summarize(), self.paired_audit()):
            self.assertEqual(result['required_cells'], 3648); self.assertFalse(result['all_required_cells_completed']); self.assertFalse(result['scientific_admission']); self.assertFalse(result['statistics_recomputed'])
    def test_prechild_abort_without_gpu_file_is_valid(self):
        col, ex, audits, ae, summary, se, paired, pe = fixtures()
        for role in ('A', 'B'): V.validate_collection(col[role], ex[role]['artifact_sha256'], ex[role])
        V.validate_paired_audit(paired, pe['artifact_sha256'], pe, col, summary, audits)
    def test_partial_diagnostic_keeps_undefined_correlation_and_timing_support(self):
        row = self.audits['A']['row_checks'][1]
        self.assertIn('correlations/previous_risk_vs_base_error', row['array_recomputed_diagnostic_metrics']); self.saved()
    def test_wrong_artifact_pin_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Artifact byte pin'): V.validate_collection(self.collections['A'], pin('wrong'), self.ce['A'])
    def test_missing_model_stage_rejected(self):
        self.collections['B']['stages'].pop()
        with self.assertRaisesRegex(ValueError, 'model stages'): self.summarize()
    def test_boolean_training_seed_rejected(self):
        self.collections['A']['stages'][0]['seed'] = True
        with self.assertRaises(ValueError): self.collect()
    def test_shortened_horizon_rejected(self):
        self.collections['A']['stages'][0]['rows'][0]['horizon'] = 200
        with self.assertRaisesRegex(ValueError, 'H314'): self.collect()
    def test_wrong_source_frames_rejected(self):
        next(iter(self.ce['A']['protocols'].values()))['source_frame_count'] = 301
        with self.assertRaisesRegex(ValueError, 'source_frame_count'): self.collect()
    def test_wrong_endpoint_updates_rejected(self):
        next(iter(self.ce['A']['protocols'].values()))['model']['completed_updates'] = 25000
        with self.assertRaisesRegex(ValueError, '100k'): self.collect()
    def test_wrong_same_state_target_rejected(self):
        self.collections['A']['stages'][1]['cells'][1]['target_frame'] = 80
        with self.assertRaisesRegex(ValueError, 'target grid'): self.collect()
    def test_wrong_clean_floor_grid_rejected(self):
        self.collections['A']['stages'][3]['cells'][-1]['target_frame'] = 318
        with self.assertRaisesRegex(ValueError, 'target grid'): self.collect()
    def test_dropped_policy_denominator_rejected(self):
        self.collections['A']['stages'][0]['cells'].pop()
        with self.assertRaisesRegex(ValueError, 'cell denominator'): self.collect()
    def test_original_ledger_state_mismatch_rejected(self):
        self.ce['A']['ledger']['stages'][0]['outcome']['exit_code'] = 0
        with self.assertRaisesRegex(ValueError, 'stage outcome'): self.collect()
    def test_nonzero_original_exit_is_preserved_and_valid(self):
        self.assertEqual(self.collections['A']['process_outcomes']['streams'][0]['outcomes'][0]['exit_code'], 1); self.collect()
    def test_dropped_pending_gpu_observation_rejected(self):
        self.collections['A']['gpu_observation_counts']['pending'] = 0
        with self.assertRaisesRegex(ValueError, 'observation counts'): self.collect()
    def test_gpu_absence_after_child_outcome_rejected(self):
        self.ce['A']['gpu_observations'] = None
        with self.assertRaisesRegex(ValueError, 'only before any child'): self.collect()
    def test_boolean_exit_code_rejected(self):
        for doc in (self.ce['A']['process_outcomes'], self.collections['A']['process_outcomes']): doc['streams'][0]['outcomes'][0]['exit_code'] = False
        with self.assertRaisesRegex(ValueError, 'evaluator outcome'): self.collect()
    def test_snapshot_membership_changed_rejected(self):
        self.collections['A']['output_tree_state']['entries'].append(['rogue.npy', 'file'])
        with self.assertRaisesRegex(ValueError, 'stopped snapshot'): self.collect()
    def test_diagnostic_survivor_source_mean_rejected(self):
        leaf = self.collections['A']['stages'][1]['diagnostic_summary']['accuracy']['base']['position_coordinate_mse']
        leaf['trajectories']['0']['mean'] = 1.0
        with self.assertRaisesRegex(ValueError, 'survivor mean within source'): self.collect()
    def test_diagnostic_survivor_trajectory_mean_rejected(self):
        self.collections['A']['stages'][3]['diagnostic_summary']['metrics']['constant_free_gaussian_nll']['equal_trajectory_mean'] = 1.0
        with self.assertRaisesRegex(ValueError, 'survivor mean across sources'): self.collect()
    def test_summary_survivor_seed_mean_rejected(self):
        self.summary['full_rollout']['absolute']['mean_rollout_mse']['base']['base']['mean'] = 1.0
        with self.assertRaisesRegex(ValueError, 'survivor mean or SD'): self.summarize()
    def test_contrast_cannot_make_missing_operands_defined(self):
        value = self.summary['full_rollout']['mix_minus_base']['mean_rollout_mse']['base']
        value.update(seed_values={'0': 1.0, '1': 1.0, '2': 1.0}, defined_seed_pairs=3, mean=1.0, sample_sd=0.0)
        with self.assertRaisesRegex(ValueError, 'operand nulls'): self.summarize()
    def test_diagnostic_interaction_cannot_make_missing_operands_defined(self):
        value = self.summary['diagnostics']['same_state_valid']['risk_minus_random_mix_minus_base_interaction']
        value.update(seed_values={'0': 1.0, '1': 1.0, '2': 1.0}, defined_seed_pairs=3, mean=1.0, sample_sd=0.0)
        with self.assertRaisesRegex(ValueError, 'operand nulls'): self.summarize()
    def test_protocol_checkpoint_substitution_rejected(self):
        next(iter(self.ce['A']['protocols'].values()))['model']['checkpoint_sha256'] = pin('replacement model')
        with self.assertRaisesRegex(ValueError, 'protocol checkpoint'): self.collect()
    def test_protocol_source_substitution_rejected(self):
        next(iter(self.ce['A']['protocols'].values()))['input_files_sha256'][V.PREP + '/train_sand_graph_support_cuda.py'] = pin('replacement trainer')
        with self.assertRaisesRegex(ValueError, 'protocol input pin'): self.collect()
    def test_summary_survivor_model_mean_rejected(self):
        self.summary['full_rollout']['absolute']['mse_forecast200']['base']['base']['seed_values']['1'] = 1.0
        with self.assertRaisesRegex(ValueError, 'survivor model/policy'): self.summarize()
    def test_false_scientific_completion_rejected(self):
        self.summary['full_rollout']['all_required_outcomes_complete'] = True
        with self.assertRaisesRegex(ValueError, 'denominator/state'): self.summarize()
    def test_failed_prefix_dropped_rejected(self):
        self.summary['full_rollout']['failed_accepted_prefixes'] = []
        with self.assertRaisesRegex(ValueError, 'Every failed accepted prefix'): self.summarize()
    def test_saved_audit_predecessor_changed_rejected(self):
        self.audits['A']['collection_sha256'] = pin('different collection')
        with self.assertRaisesRegex(ValueError, 'lineage/scope'): self.saved()
    def test_saved_revision1_rejected(self):
        self.audits['A']['audit_revision'] = 1
        with self.assertRaisesRegex(ValueError, 'audit_revision'): self.saved()
    def test_audited_sparse_errors_cannot_claim_unsaved_step(self):
        self.audits['A']['row_checks'][0]['array_recomputed_mse_forecasts'].append(201)
        with self.assertRaisesRegex(ValueError, 'exact saved forecasts'): self.saved()
    def test_sparse_save_limitation_cannot_be_removed(self):
        self.audits['A']['scope']['unsupported'].pop(0)
        with self.assertRaisesRegex(ValueError, 'scope'): self.saved()
    def test_late_failed_prefix_keeps_H200_but_not_H314_trace(self):
        self.assertEqual(self.audits['A']['row_checks'][0]['array_recomputed_mse_forecasts'], [1, 10, 50, 200]); self.saved()
    def test_saved_row_audit_missing_rejected(self):
        self.audits['A']['row_checks'].pop()
        with self.assertRaisesRegex(ValueError, 'every committed row'): self.saved()
    def test_saved_failure_category_changed_rejected(self):
        self.audits['A']['row_checks'][0]['failure'] = {'category': 'replacement'}
        with self.assertRaisesRegex(ValueError, 'Audited row failure'): self.saved()
    def test_paired_source_pin_changed_rejected(self):
        self.paired['verified_input_sha256'][next(iter(V.PAIRED_SOURCES))] = pin('wrong source')
        with self.assertRaisesRegex(ValueError, 'predecessor/source map'): self.paired_audit()
    def test_cross_host_saved_truth_mismatch_rejected(self):
        self.audits['A']['shared_saved_state_hashes']['same source'] = pin('A truth')
        self.audits['B']['shared_saved_state_hashes']['same source'] = pin('B truth')
        with self.assertRaisesRegex(ValueError, 'Cross-host'): self.paired_audit()
    def test_paired_denominator_mismatch_rejected(self):
        self.paired['verified_paired_scalars']['full_rollout']['required_outcomes'] = 900
        with self.assertRaisesRegex(ValueError, 'denominator/state'): self.paired_audit()
    def test_nonfinite_product_rejected(self):
        self.summary['unexpected'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Finite JSON'): self.summarize()
    def test_corrupt_original_input_integrity_blocks_aggregation(self):
        self.collections['A']['queue_status']['all_pinned_inputs_reverified'] = False
        self.ce['A']['queue_status']['all_pinned_inputs_reverified'] = False
        self.assertFalse(self.collect()['eligible_for_aggregation'])
        with self.assertRaisesRegex(ValueError, 'must be reverified'): self.summarize()


if __name__ == '__main__': unittest.main(verbosity=2)
