#!/usr/bin/env python3
"""Independent local byte/operational audit of root-issued final24 controls.

Never calls SSH, issues a phase, imports scientific modules, or reads arrays or
checkpoints. Actual input bytes are established by the root's completed bounded
remote capture; this audit checks the frozen capture and exact release delta.
"""
from pathlib import Path
import argparse
import datetime as D
import hashlib
import json
import math
import time

P = Path(__file__).resolve().parent
PK = P / 'sand_final24_operator_UNADMITTED_transition_v1'
BASE = P / 'sand_final24_B_amendment_paths_UNADMITTED_transition_v1'
R = '/root/repos/AdaptGNS-cuda-20261006'
BASE_SHA = '0c438762373ea714ba1f415f509c756bd58f2297c8887e8c4a230b3e066472d0'
PREPARED_SHA = 'dfd1cb9c210196abd87dafc2d7164487e0aba18586e34eb098d92f85acfdf604'
B_COMPLETION_SHA = 'db7b953ccee7c8a661ebe7f2016065791f3dcdd17a99c4ea6749a0dc213bb1d8'
GLOBAL_STOP = '2026-10-07T04:00:00+00:00'
TIMEOUT_SHA = '2db30bc57746c940a0643581cd5e2671e505442bea16164143f0ea8ff4e36453'
QUOTAS = {'clean_validation': 900, 'full_rollout_test': 7200, 'same_state_test': 1800, 'same_state_valid': 1800}
ENV = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'LD_LIBRARY_PATH': '/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}


def read(path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise ValueError('Duplicate JSON key: ' + key)
            out[key] = value
        return out
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=reject)


def sha(path):
    path = Path(path)
    if path.suffix in ('.npy', '.npz', '.pt', '.pth'):
        raise ValueError('Numeric/model reading is outside this audit')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dt(value):
    out = D.datetime.fromisoformat(value)
    if out.tzinfo is None:
        raise ValueError('Aware UTC required')
    return out


def now():
    return D.datetime.now(D.timezone.utc)


def exact_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False, separators=(',', ':')) == json.dumps(right, sort_keys=True, allow_nan=False, separators=(',', ':'))


def audit(state, roles, package_pin, checked_at=None, monotonic=None):
    checked_at = now() if checked_at is None else checked_at
    monotonic = time.monotonic() if monotonic is None else monotonic
    checks = []
    evidence = {}

    def check(condition, message):
        if not condition:
            raise ValueError(message)
        checks.append(message)

    def fingerprint(path):
        path = Path(path)
        check(path.is_file() and not path.is_symlink(), 'Ordinary scalar/source: ' + str(path))
        pin = sha(path)
        evidence[str(path.resolve())] = pin
        return pin

    def document(path, expected=None):
        path = Path(path)
        pin = fingerprint(path)
        if expected is not None:
            check(pin == expected, 'Exact bytes: ' + str(path))
        return read(path)

    package = document(PK / 'manifest.json', package_pin)
    for name, pin in package['files_sha256'].items():
        check(fingerprint(PK / name) == pin, 'Frozen operator member: ' + name)
    base_manifest = document(BASE / 'manifest.json', BASE_SHA)
    for name, pin in base_manifest['files_sha256'].items():
        check(fingerprint(BASE / name) == pin, 'Frozen original candidate member: ' + name)
    prepared = document(state / 'prepared_inputs.json', PREPARED_SHA)
    check(fingerprint(PK / 'prepared_inputs.json') == PREPARED_SHA, 'Local and root prepared inputs identical')
    check(prepared['candidate_manifest_sha256'] == BASE_SHA and prepared['root_B_completion_sha256'] == B_COMPLETION_SHA, 'Exact original candidate and accepted B amendment')
    for path, pin in prepared['original_local_evidence_sha256'].items():
        check(fingerprint(path) == pin, 'Previously accepted evidence bytes: ' + path)
    sources = document(BASE / 'frozen_source_closure.json')['files_sha256']
    check(len(sources) == 23 and {path: row['sha256'] for path, row in prepared['source_files'].items()} == sources, 'Complete unchanged 23-source closure')
    for path, row in prepared['source_files'].items():
        check(fingerprint(PK / row['package_path']) == sources[path], 'Packaged frozen source: ' + path)
    census_paths = [path for path in prepared['original_local_evidence_sha256'] if path.endswith('/census/all_split_census.json')]
    check(len(census_paths) == 1, 'Exactly one accepted original numeric census')
    census = document(census_paths[0], prepared['original_local_evidence_sha256'][census_paths[0]])
    check(census['status'] == 'complete_no_duplicates' and census['counts'] == {'train': 1000, 'valid': 30, 'test': 30} and len(census['numeric_files_sha256']) == 2120, 'Original census covers all 2120 numeric files')
    paths = {root + '/' + kind + '_' + str(i).zfill(6) + '.npy' for root in (R + '/sand_numeric_train_valid_20261006_v1/valid', R + '/sand_reserved_preparation_20261006_v1/numeric/test') for kind in ('position', 'type') for i in range(30)}
    arrays = {path: census['numeric_files_sha256'][path] for path in sorted(paths)}
    check(len(arrays) == 120 and prepared['array_files_sha256'] == arrays, 'Exact 120 original valid/test array pins without decoding')
    binding = document(state / 'operator_binding.json')
    check(binding['manifest_sha256'] == package_pin and binding['local_preparation_only'] is True and binding['clock_read_or_issued'] is False, 'Root state bound before control clock issuance')
    phase = document(state / 'control_phase.json')
    phase_pin = sha(state / 'control_phase.json')
    anchor = document(state / 'control_anchor.json')
    start, stop = dt(phase['started_utc']), dt(phase['stop_utc'])
    check(phase['schema'] == 'sand_final24_original_control_phase_v1' and phase['status'] == 'approved_one_original_final24_control_phase' and phase['issued_by'] == 'root', 'Actual root-issued final24 control phase')
    check(phase['prepared_inputs_sha256'] == PREPARED_SHA and phase['clock_restarted'] is False and phase['control_seconds'] == 600 and phase['evaluation_seconds'] == 11760 and phase['analysis_seconds'] == 3600, 'Unchanged 600/11760/3600 allocations and no reset')
    check((stop - start).total_seconds() == 600 and phase['global_analysis_stop_utc'] == GLOBAL_STOP and stop + D.timedelta(seconds=11760 + 3600 + 15) <= dt(GLOBAL_STOP), 'Full fixed work and analysis fit by global deadline')
    check(anchor['phase_sha256'] == phase_pin and anchor['utc'] == start.isoformat() and type(anchor['monotonic_seconds']) in (int, float) and math.isfinite(anchor['monotonic_seconds']), 'Original phase has exact finite local anchor')
    elapsed = monotonic - anchor['monotonic_seconds']
    check(0 <= elapsed < 600 and abs((checked_at - start).total_seconds() - elapsed) <= 5 and start <= checked_at < stop - D.timedelta(seconds=10), 'Review completed inside original UTC and monotonic control window')
    commands = document(BASE / 'commands.json')
    release_pins = {}
    for role in roles:
        spec = prepared['roles'][role]
        candidate = document(PK / spec['resolved_candidate'], spec['resolved_candidate_sha256'])
        original = document(BASE / (role + '.evaluation_release.candidate.json'))
        resolved_expected = json.loads(json.dumps(original))
        resolved_expected['files_sha256'] = spec['files_sha256']
        for split in ('valid', 'test'):
            path = original['split_preflight'][split]['path']
            resolved_expected['split_preflight'][split]['sha256'] = spec['files_sha256'][path]
        check(exact_json(candidate, resolved_expected), role + ': preparation changes only original null pins and exact array map')
        original_pins = original['files_sha256']
        check(set(spec['files_sha256']) == set(original_pins) | paths and all(h is None or spec['files_sha256'][path] == h for path, h in original_pins.items()), role + ': no original immutable pin changed')
        check(len(spec['files_sha256']) == (164 if role == 'A' else 162) and {path: h for path, h in spec['files_sha256'].items() if path.endswith('.npy')} == arrays, role + ': complete exact role input map')
        check(spec['owner_argv'] == commands['hosts'][role]['owner_argv'] and candidate['environment'] == ENV and candidate['stage_quotas_seconds'] == QUOTAS and candidate['cleanup_seconds_per_invocation'] == 15 and candidate['outer_processing_reserve_seconds'] == 3600, role + ': owner argv and full scientific allocation unchanged')
        capture_path = state / (role + '.verify_stage.json')
        capture = document(capture_path)
        capture_pin = sha(capture_path)
        external = document(state / (role + '.verify_stage.external.json'))
        command = document(state / (role + '.verify_stage.command.json'))
        check(external['exit_code'] == 0 and external['local_transport_reaped'] is True and external['local_transport_timeout'] is False and external['signals_to_own_local_group'] == [] and external['failure'] is None, role + ': completed bounded control transport exited zero')
        check(start <= dt(external['started_utc']) <= dt(external['observed_utc']) <= checked_at < stop and command['stop_utc'] == stop.isoformat(), role + ': transport consumed the original control window')
        check(capture['schema'] == 'sand_final24_remote_control_observation_v1' and capture['action'] == 'verify_stage' and capture['role'] == role and capture['hostname'] == spec['hostname'] and capture['same_control_stop_utc'] == stop.isoformat(), role + ': exact bounded native verification capture')
        check(capture['input_sha256'] == spec['files_sha256'], role + ': every actual input hash matches the frozen input map')
        check(capture['native_absent'] == {str(pid): True for pid in spec['historical_pids']} and capture['historical_absence'] == capture['native_absent'] and capture['owner'] is None and capture['children'] == [] and capture['other_matching_science_processes'] == [], role + ': complete historical native absence and no active matching science')
        check(capture['gpu_uuids'] == candidate['gpu_uuids'], role + ': all four GPU indices retain original UUIDs')
        owned = candidate['gpu_scope']['owned_indices']
        check(all(row['gpu_uuid'] in capture['gpu_uuids'] and capture['gpu_uuids'].index(row['gpu_uuid']) not in owned for row in capture['gpu_processes']), role + ': assigned devices have no compute processes')
        runtime = capture['runtime']; pyenv = candidate['python_environment']
        check(runtime['hostname'] == spec['hostname'] and runtime['machine'] == 'aarch64' and runtime['libc'][0] == 'glibc' and bool(runtime['libc'][1]), role + ': exact native host architecture and libc')
        check(all(runtime['python_environment'][k] == v for k, v in pyenv.items() if k != 'sys_prefix') and runtime['python_environment']['sys_prefix'] in ('/usr', R + '/.venv') and runtime['isolated'] is True and runtime['no_site'] is True and runtime['dont_write_bytecode'] is True and runtime['optimize'] == 0, role + ': exact isolated lexical Python and venv')
        check(capture['runtime_sha256'] == {pyenv['lexical_path']: pyenv['binary_sha256'], pyenv['resolved_binary_path']: pyenv['binary_sha256'], pyenv['pyvenv_config_path']: pyenv['pyvenv_config_sha256'], '/usr/bin/timeout': TIMEOUT_SHA}, role + ': exact four runtime binary/config pins')
        check(capture['required_launch_environment'] == ENV and capture['required_launch_variables_absent'] == ['CUDA_VISIBLE_DEVICES'] and capture['release_sha256'] is None and capture['scientific_execution_performed'] is False and capture['scientific_admission'] is False, role + ': operational-only capture before release or science')
        clock = capture['clock']; sample = dt(clock['host_utc']); reference = dt(capture['root_reference_utc'])
        check(clock['host_boot_id'] == spec['expected_boot_id'] and type(clock['host_monotonic_seconds']) in (int, float) and math.isfinite(clock['host_monotonic_seconds']), role + ': unchanged native boot and finite monotonic sample')
        check(start - D.timedelta(seconds=5) <= sample < stop and dt(external['started_utc']) - D.timedelta(seconds=5) <= sample <= dt(external['observed_utc']) + D.timedelta(seconds=5) and abs((reference - sample).total_seconds()) <= 5 and 0 <= (checked_at - sample).total_seconds() <= 300, role + ': fresh root-host clock inside original control interval')
        release_path = state / (role + '.evaluation_release.json')
        release = document(release_path)
        expected = json.loads(json.dumps(candidate))
        expected.update(status='admitted_for_execution_allocation', issued_by='root', preparation_and_cohort_reserves_already_accounted_before_this_queue=True, clock_error_bound_seconds=5, latest_start_utc=(stop - D.timedelta(seconds=5)).isoformat(), process_clock_checked_utc=clock['host_utc'])
        expected['candidate_metadata'].update(actual_release_created=True, fresh_process_closure_attestation={'control_phase_sha256': phase_pin, 'capture_sha256': capture_pin}, fresh_gpu_inventory_attestation={'capture_sha256': capture_pin}, fresh_clock_review_attestation={'capture_sha256': capture_pin, 'control_stop_utc': stop.isoformat()}, fresh_output_absence_attestation={'capture_sha256': capture_pin}, actual_complete_immutable_array_map=arrays, prepared_inputs_sha256=PREPARED_SHA)
        check(exact_json(release, expected), role + ': exact independent actual-release delta; every scientific field unchanged')
        check(checked_at + D.timedelta(seconds=15) < dt(release['latest_start_utc']), role + ': same-control staging/launch margin remains')
        release_pins[role] = sha(release_path)
    return {'schema': 'sand_final24_actual_issuance_independent_review_v1', 'status': 'passed_actual_final24_issuance', 'reviewer': 'code_audit', 'checked_utc': checked_at.isoformat(), 'control_phase_sha256': phase_pin, 'prepared_inputs_sha256': PREPARED_SHA, 'operator_manifest_sha256': package_pin, 'release_sha256': release_pins, 'checks': checks, 'evidence_sha256': evidence, 'scientific_execution_or_completion_asserted': False, 'remote_native_observation_still_required_after_original_launch': True, 'review_source_sha256': sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--role', choices=('A', 'B', 'both'), required=True)
    parser.add_argument('--package-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roles = ('A', 'B') if args.role == 'both' else (args.role,)
    result = audit(args.state.resolve(), roles, args.package_sha256)
    for path, pin in result['evidence_sha256'].items():
        if sha(path) != pin:
            raise ValueError('Evidence changed before receipt publication: ' + path)
    finished = now()
    phase = read(args.state / 'control_phase.json')
    anchor = read(args.state / 'control_anchor.json')
    if not (finished + D.timedelta(seconds=20) < dt(phase['stop_utc'])
            and time.monotonic() + 20 < anchor['monotonic_seconds'] + 600):
        raise ValueError('Original control deadline reached before review publication')
    result['completed_utc'] = finished.isoformat()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'checks': len(result['checks']), 'release_sha256': result['release_sha256'], 'receipt': str(args.output.resolve()), 'receipt_sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
