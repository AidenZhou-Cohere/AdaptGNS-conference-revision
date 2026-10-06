#!/usr/bin/env python3
"""Scalar-only deadline worksheet from completed capacity and validation timing.

Never chooses or launches a scientific endpoint. No numerical imports, model
deserialization, test access or network. Missing/failed full-H timing is retained
as an unavailable forecast rather than scaled from a successful prefix.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop3d_deadline_worksheet_v1'
EVALUATOR_SHA = '9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de'
VALID_MANIFEST_SHA = 'f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef'
TRAINER_SHA = '8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc'
GRAPH_SHA = 'ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50'
MODES = ('full-rollout', 'same-state', 'clean-validation')
DEADLINE = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def load_evaluator():
    path = HERE / 'evaluate_goop3d_graph_support_v1.py'
    require(sha(path) == EVALUATOR_SHA, 'Reviewed D3 evaluator differs')
    spec = importlib.util.spec_from_file_location('_goop3d_worksheet_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capacity_inputs(report):
    require(report.get('schema') == 'adaptgns_goop3d_vectorized_capacity_v1'
            and report.get('status') == 'all_six_verified_capacity_only'
            and report.get('scientific_training_admitted') is False
            and report.get('scientific_endpoint_selected') is False, 'Complete distinct six-model capacity report required')
    jobs = report.get('jobs', [])
    expected = {(a, s) for a in ('base', 'mix') for s in range(3)}
    require(len(jobs) == 6 and {(r.get('arm'), r.get('seed')) for r in jobs} == expected,
            'All six capacity jobs required')
    q, residual, pointers = {}, {}, {}
    for row in jobs:
        arm, seed = row['arm'], row['seed']
        wave = 'A' if seed < 2 else 'B'
        require(row.get('status') == 'verified_capacity_only' and row.get('wave') == wave
                and row.get('final_pointer', {}).get('completed_steps') == 512
                and finite(row.get('steady_wall_seconds_per_update'), True)
                and finite(row.get('nonnegative_external_minus_all_guarded_seconds')),
                'Incomplete capacity duration/checkpoint coverage')
        q[wave] = max(q.get(wave, 0), row['steady_wall_seconds_per_update'])
        residual[wave] = max(residual.get(wave, 0), row['nonnegative_external_minus_all_guarded_seconds'])
        pointers[arm, seed] = row['final_pointer']['sha256']
    require(q == report.get('q4_q2') and residual == report.get('r4_r2'), 'Capacity wave arithmetic differs')
    pairs = report.get('pairing', [])
    require(len(pairs) == 3 and {r.get('seed') for r in pairs} == {0, 1, 2}
            and all(r.get('initial_model_rng_Adam_exact') is True and r.get('all512_sample_noise_lr_rows_exact') is True for r in pairs),
            'Verified paired capacity schedules required')
    return q, residual, pointers


def collect_timing(entry, manifest, checkpoint_sha, evaluator):
    directory = Path(entry['directory']).resolve()
    require(not list(directory.rglob('*.tmp')) and not (directory / 'run.lock').exists(), 'Unfinished timing artifacts')
    bindings = entry['files_sha256']
    require(all(Path(p).resolve().is_relative_to(directory) and sha(p) == value for p, value in bindings.items()),
            'Timing receipt bytes differ or leave its directory')
    protocol_path, status_path = directory / 'protocol.json', directory / 'status.json'
    require(all(str(p) in bindings for p in (protocol_path, status_path)), 'Bound timing protocol/status required')
    p, status = read(protocol_path), read(status_path)
    arm, mode = entry['arm'], entry['mode']
    require(p.get('schema') == evaluator.SCHEMA and p.get('purpose') == 'capacity_timing'
            and p.get('split') == 'valid' and p.get('arm') == arm and p.get('seed') == 0
            and p.get('mode') == mode and p.get('checkpoint_updates') == 512
            and p.get('checkpoint_sha256') == checkpoint_sha and p.get('horizon') == 295
            and p.get('policies') == list(evaluator.POLICIES), 'Distinct seed0 capacity timing contract required')
    expected = evaluator.schedules(manifest['records'], mode, 'capacity_timing')
    require(p.get('schedule') == expected and p.get('source_population_count') == 100
            and p.get('prospective_source_grid') == evaluator.grid_indices(100), 'Prospective timing grid differs')
    require(status.get('state') == 'complete' and status.get('all_inputs_reverified') is True
            and not (directory / 'failed_attempt.json').exists(), 'Full timing invocation did not complete')
    external = entry['external']
    require(external.get('stopped_and_reaped') is True and external.get('exit_code') == 0
            and type(external.get('pid')) is int and external['pid'] > 0
            and finite(external.get('elapsed_seconds'), True)
            and finite(status.get('total_execution_seconds'), True)
            and external['elapsed_seconds'] + 1e-6 >= status['total_execution_seconds'],
            'Whole stopped-process timing receipt required')
    rows, costs = [], []
    for item in expected:
        methods = evaluator.POLICIES if mode == 'full-rollout' else (None,)
        for policy in methods:
            name = f'trajectory_{item["source_index"]:06d}_' + (policy if policy else f'target_{item["target_frame"]:03d}')
            row_path, array_path = directory / (name + '.json'), directory / (name + '.npz')
            require(all(str(path) in bindings for path in (row_path, array_path)), 'Every timed row and numeric artifact must be bound')
            row = read(row_path)
            require(row.get('status') == 'complete' and row.get('failure') is None
                    and all(row.get(k) == v for k, v in item.items()) and row.get('arm') == arm and row.get('training_seed') == 0
                    and row.get('protocol_sha256') == sha(protocol_path) and row.get('artifact_sha256') == sha(array_path)
                    and row.get('artifact_file') == array_path.name
                    and finite(row.get('synchronized_call_seconds'), True) and finite(row.get('publication_seconds')),
                    'Complete exact timing row required')
            if policy:
                require(row.get('policy') == policy and row.get('completed_steps') == 295 and len(row.get('mse_per_step', [])) == 295,
                        'All H295 steps required; no successful-prefix extrapolation')
            rows.append(row)
            costs.append(row['synchronized_call_seconds'] + row['publication_seconds'])
    require(status.get('committed_rows') == status.get('expected_rows') == len(rows)
            and external['elapsed_seconds'] + 1e-6 >= sum(costs), 'Timing coverage or whole-process duration differs')
    # Scalar outcome values are neither used for selection nor recomputed here.
    if mode == 'full-rollout':
        per_unit = sum(max(c for r, c in zip(rows, costs) if r['policy'] == method) for method in evaluator.POLICIES)
    else:
        per_unit = max(costs)
    return {'arm': arm, 'mode': mode, 'complete_rows': len(rows), 'per_unit_seconds': per_unit,
            'setup_and_other_seconds': max(0, external['elapsed_seconds'] - sum(costs)),
            'checkpoint_sha256': checkpoint_sha, 'files_sha256': bindings,
            'unit': 'one source, all six full-H policies' if mode == 'full-rollout' else 'one diagnostic frame'}


def forecast(plan, q, residual, timing):
    require(plan.get('schema') == 'adaptgns_goop3d_deadline_planning_config_v1'
            and plan.get('selected_endpoint') is None and plan.get('evaluation_overlap_credit') is False,
            'Explicit planning-only configuration required')
    candidates = plan.get('candidate_endpoints', [])
    require(candidates and len(set(candidates)) == len(candidates) and all(type(x) is int and x >= 2 for x in candidates),
            'Distinct candidate endpoints required')
    every = plan.get('checkpoint_every')
    require(type(every) is int and every > 0 and finite(plan.get('runtime_multiplier'), True)
            and plan['runtime_multiplier'] >= 1 and finite(plan.get('transfer_review_analysis_reserve_seconds')),
            'Prospective checkpoint/risk/reserve assumptions required')
    start = datetime.fromisoformat(plan['planning_start_utc'])
    require(start.tzinfo is not None and start < DEADLINE, 'Planning must precede compute cutoff')
    available = (DEADLINE - start).total_seconds()
    by = {(r['arm'], r['mode']): r for r in timing}
    require(set(by) == {(a, m) for a in ('base', 'mix') for m in MODES} and len(timing) == 6,
            'Both seed0 arms and all three full timing modes required')
    evaluation = 0.
    for arm in ('base', 'mix'):
        full, same, clean = (by[arm, mode] for mode in MODES)
        # Budget 30 validation and up to 30 future test sources without opening test.
        evaluation += 3 * (60 * full['per_unit_seconds'] + 300 * same['per_unit_seconds']
            + 128 * clean['per_unit_seconds'] + 2 * full['setup_and_other_seconds']
            + 2 * same['setup_and_other_seconds'] + clean['setup_and_other_seconds'])
    rows = []
    for updates in candidates:
        saves = math.ceil(updates / every) + 1
        core = updates * (q['A'] + q['B'])
        # Charge each final save the entire observed capacity non-kernel residual.
        # This deliberately overcharges startup; it remains an estimate, not a bound.
        checkpoint_and_setup = saves * (residual['A'] + residual['B'])
        total = plan['runtime_multiplier'] * (core + checkpoint_and_setup + evaluation) + plan['transfer_review_analysis_reserve_seconds']
        rows.append({'candidate_endpoint_updates': updates, 'training_4_plus_2_wave_core_seconds': core,
            'conservative_checkpoint_and_setup_seconds': checkpoint_and_setup, 'evaluation_serial_seconds': evaluation,
            'projected_total_seconds': total, 'available_seconds': available, 'projected_slack_seconds': available - total,
            'fits_stated_planning_assumptions': total <= available})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    for name in ('release', 'capacity-summary', 'valid-manifest', 'timing-inventory', 'planning-config', 'output-file'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema': SCHEMA, 'status': 'description_only', 'training_selected_or_launched': False,
            'timing_sample': 'base0 and mix0 distinct512 capacity; fullH295 three fixed-grid size ranks, target7 same-state,128 clean frames',
            'test_accessed': False, 'evaluation_overlap_credit': False})); return 0
    require(all(getattr(args, key) is not None for key in ('release', 'capacity_summary', 'valid_manifest', 'timing_inventory', 'planning_config', 'output_file')),
            'All root-bound scalar planning inputs required')
    release = read(args.release)
    paths = [args.capacity_summary, args.valid_manifest, args.timing_inventory, args.planning_config, Path(__file__)]
    require(release.get('schema') == 'adaptgns_goop3d_deadline_worksheet_release_v1'
            and release.get('status') == 'admitted_for_scalar_planning' and release.get('issued_by') == 'root'
            and release.get('scientific_training_admitted') is False
            and all(release.get('files_sha256', {}).get(str(path.resolve())) == sha(path) for path in paths),
            'Exact scalar planning release required')
    require(sha(args.valid_manifest) == VALID_MANIFEST_SHA, 'Original complete validation manifest required')
    bindings = {str(path.resolve()): sha(path) for path in paths + [args.release]}
    output = {'schema': SCHEMA, 'scientific_training_admitted': False, 'scientific_endpoint_selected': False,
        'test_accessed': False, 'input_sha256': bindings, 'forecasts': [], 'unavailable_reasons': [],
        'limits': ['Engineering estimates are not timing guarantees or statistical confidence bounds.',
            'Only seed0 infrastructure checkpoints are timed; final model states, other seeds, future test geometry and contention can change costs.',
            'No evaluation overlap credit. Scalar maxima include full H295 work and artifact publication; no failed-prefix extrapolation.',
            'All final models must start fresh after a separate root prospective endpoint/protocol admission.']}
    try:
        evaluator = load_evaluator()
        q, residual, checkpoints = capacity_inputs(read(args.capacity_summary))
        inventory = read(args.timing_inventory)
        require(inventory.get('schema') == 'adaptgns_goop3d_capacity_timing_inventory_v1' and inventory.get('issued_by') == 'root'
                and inventory.get('status') == 'all_required_processes_stopped', 'Root stopped timing inventory required')
        entries = inventory.get('entries', [])
        require(len(entries) == 6 and {(r.get('arm'), r.get('mode')) for r in entries} == {(a, m) for a in ('base', 'mix') for m in MODES},
                'Prospective six timing invocation sample required')
        manifest = read(args.valid_manifest)
        timing = [collect_timing(row, manifest, checkpoints[row['arm'], 0], evaluator) for row in entries]
        bindings = evaluator.merge_bindings(bindings, {HERE / 'evaluate_goop3d_graph_support_v1.py': EVALUATOR_SHA},
            *(row['files_sha256'] for row in timing))
        output['input_sha256'] = bindings
        output.update(status='engineering_worksheet_complete_not_execution_admission', timing=timing,
            forecasts=forecast(read(args.planning_config), q, residual, timing))
        require(all(sha(path) == value for path, value in bindings.items()), 'Planning input changed')
    except Exception as error:
        output.update(status='forecast_unavailable', forecasts=[], unavailable_reasons=[{'type': type(error).__name__, 'reason': str(error)}])
    with args.output_file.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write('\n')
    return 0 if output['forecasts'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
