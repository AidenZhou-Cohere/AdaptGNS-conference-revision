#!/usr/bin/env python3
"""Unadmitted Sand recovery-to-final-cohort bridge; description by default.

After root-verified completion only, inventory original A and recovered A as
distinct retained trees. Build joins that inventory to an unchanged original B
inventory and emits the existing frozen final-evaluator interface. No tensor
loading, inference, test access, process control, relocation or source edits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
INVENTORY_SCHEMA = 'adaptgns_sand_recovered_A_completed_byte_inventory_v1'
RELEASE_SCHEMA = 'adaptgns_sand_recovered_final_cohort_bridge_release_v1'
PROCESS_SCHEMA = 'adaptgns_sand_recovered_A_completed_process_check_v1'
OWNER_SCHEMA = 'adaptgns_sand_runtime_migration_recovery_owner_v1'
AUDIT_SCHEMA = 'adaptgns_sand_runtime_migration_endpoint_audit_v1'
RECOVERY_RELEASE_SHA = '194437c62c992a061e2b3e4f3f7d12d487d7edcca2a2c701e185d3d05d3bd9ca'
COHORT_ID = 'sand_graph_support_scoped_100k_20261006_v1'
SOURCE_PINS = {
    'prepare_sand_final_cohort_scoped_v1.py': 'ebad4e30edbdb9f444ed07dabf50b31540ecf7f85b134ee059c150d1ff68ca5f',
    'supervise_sand_runtime_migration_recovery_v2.py': '58625b827484b46b9c3d852bc4d71d16e48766b1c416050cc8c3dc5640bd21fe',
    'train_sand_runtime_migration_recovery_v1.py': '0faf1bb8ff25ac1487506781dba9722361682c725eeb7915874c90f3b66499e6',
    'audit_sand_runtime_migration_endpoints_v1.py': '7fc6a6000ecb00168beeb00c695c7eb8feb25db28c6436afd7a3aceb563ea7db',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())
def encode(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
def digest(value): return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
def now(): return datetime.now(timezone.utc)
def stamp(value):
    t = datetime.fromisoformat(value)
    require(t.tzinfo is not None, 'Timezone-aware time required')
    return t.astimezone(timezone.utc)


def absolute(value):
    p = Path(value)
    require(p.is_absolute() and str(p.resolve()) == str(p), 'Canonical absolute path required')
    return p


def helper():
    path = HERE / 'prepare_sand_final_cohort_scoped_v1.py'
    require(sha(path) == SOURCE_PINS[path.name], 'Frozen original B adapter differs')
    spec = importlib.util.spec_from_file_location('_sand_original_cohort_scalar', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge(*groups):
    result = {}
    for group in groups:
        for path, value in group.items():
            absolute(path)
            require(digest(value) and (path not in result or result[path] == value), 'Conflicting byte binding')
            result[path] = value
    return result


def verify(bindings):
    require(all(sha(path) == value for path, value in bindings.items()), 'Retained evidence changed')


def tree_paths(root):
    root = absolute(root)
    paths = sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
    require(paths and all(not p.is_symlink() for p in paths), 'Empty or symlinked retained tree')
    return paths


def tree_inventory(roots):
    trees, files = {}, {}
    for root in roots:
        paths = tree_paths(root)
        trees[str(root)] = [str(p.relative_to(root)) for p in paths]
        for path in paths:
            before = path.stat()
            value = {'sha256': sha(path), 'bytes': before.st_size}
            after = path.stat()
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    'Retained file changed while hashing')
            require(str(path) not in files or files[str(path)] == value, 'Conflicting retained tree overlap')
            files[str(path)] = value
    return trees, files


def verify_trees(inv):
    require(all([str(p.relative_to(root)) for p in tree_paths(root)] == names
                for root, names in inv['trees'].items()), 'Retained tree membership changed')
    verify(inv['inventory_evidence_bindings'])


def check_a(receipts, files, H):
    """Scalar evidence checks only; no checkpoint/history deserialization."""
    release, started, wave, terminal, process = [receipts[k] for k in
        ('release', 'started', 'wave', 'terminal', 'process')]
    release_path = receipts['release_path']
    owner = Path(release['owner_output_directory'])
    require(files[release_path]['sha256'] == RECOVERY_RELEASE_SHA
            and release.get('schema') == 'adaptgns_sand_runtime_migration_recovery_release_v1'
            and release.get('status') == 'approved_for_sand_migration_training'
            and release.get('issued_by') == 'root' and release.get('cohort_id') == COHORT_ID,
            'Exact original manually admitted recovery release required')
    require(started.get('schema') == wave.get('schema') == terminal.get('schema') == OWNER_SCHEMA
            and started.get('mode') == wave.get('mode') == terminal.get('mode') == 'train'
            and started.get('owner_release_sha256') == terminal.get('owner_release_sha256') == RECOVERY_RELEASE_SHA,
            'Recovery owner lineage differs')
    require(wave.get('state') == 'verified_recovery_endpoints' and wave.get('error') is None
            and terminal.get('status') == 'complete_stopped_and_reaped'
            and terminal.get('wave_sha256') == files[str(owner / 'wave_A.json')]['sha256']
            and wave.get('unreaped_owned_children') == [] and wave.get('never_started') == []
            and wave.get('final_gpu_inventory') == [] and wave.get('whole_six_model_cohort_admitted') is False
            and wave.get('scientific_promotion_from_replay') is False,
            'Only fully audited clean recovery may join the cohort')
    require(process.get('schema') == PROCESS_SCHEMA and process.get('status') == 'verified_completed_recovered_A'
            and process.get('issued_by') == 'root' and process.get('hostname') == release['host']
            and process.get('owner_pid') == started.get('owner_pid') == wave['owner_identity']['pid']
            and process.get('owner_release_sha256') == RECOVERY_RELEASE_SHA
            and process.get('owner_exited') is True and process.get('all_children_reaped') is True
            and process.get('matching_training_processes') == [] and process.get('scoped_gpu_inventory') == [],
            'Separate root post-exit native closure required')
    required_control = {release_path: RECOVERY_RELEASE_SHA,
                        **{str(owner / name): files[str(owner / name)]['sha256'] for name in
                           ('owner_started.json', 'wave_A.json', 'owner_terminal.json')}}
    require(process.get('files_sha256') == required_control, 'Root closure must bind exact original owner products')
    require(stamp(wave['ended_utc']) <= stamp(wave['audit_ended_utc']) <= stamp(terminal['terminal_utc'])
            <= stamp(process['checked_utc']) and stamp(wave['audit_ended_utc']) <= stamp(release['training_stop_utc']),
            'Recovery reaping/audit/closure chronology differs')
    expected = H.schedule('A')
    jobs = wave.get('verified_jobs', [])
    children = wave.get('all_children', [])
    require(len(jobs) == len(children) == 4
            and {j.get('id') for j in jobs} == {j['id'] for j in expected}
            and {c.get('job', {}).get('id') for c in children} == {j['id'] for j in expected}
            and all(c.get('exit_code') == 0 and c.get('signals') == [] for c in children),
            'Four original A jobs must cleanly finish')
    for plan in expected:
        job = next(j for j in jobs if j['id'] == plan['id'])
        original = next(j for j in release['jobs'] if j['id'] == plan['id'])
        require(all(job.get(k) == v for k, v in original.items())
                and all(job.get(k) == plan[k] for k in ('id', 'arm', 'seed', 'gpu'))
                and type(job['seed']) is int and job.get('schema') == AUDIT_SCHEMA
                and job.get('status') == 'verified_recovered_scientific100k_endpoint'
                and job.get('recovery_release_path') == release_path
                and job.get('recovery_release_sha256') == RECOVERY_RELEASE_SHA,
                'Exact recovered endpoint identity required')
        require(job.get('parent_completed_steps') == job.get('new_completed_updates') == 50000
                and job.get('parent_history_prefix_exact') is True and job.get('original_lock_retained') is True
                and job.get('historical_runtime_is_original_lineage') is True
                and job.get('cross_physical_device_bitwise_equivalence_claim') is False
                and job.get('recovery_is_fresh_from_scratch') is False
                and job.get('original_initialization_from_scratch_verified') is True,
                'Recovery lineage must remain explicit')
        config, status, pointer = [receipts['jobs'][job['id']][k] for k in ('config', 'status', 'pointer')]
        require(H.canonical(config) == job['config_sha256'] and config.get('schema') == H.TRAIN_SCHEMA
                and config.get('arm') == job['arm'] and config.get('seed') == job['seed']
                and config.get('updates') == 100000 and config.get('research_protocol_sha256') == H.PINS['protocol']
                and config.get('checkpoint_every') == 10000 and config.get('log_every') == 100
                and config['runtime'] == job['runtime'] == original['original_runtime'],
                'Historical training configuration differs')
        require(status.get('schema') == H.TRAIN_SCHEMA and status.get('state') == 'complete'
                and status.get('error') is None and status.get('completed_steps') == status.get('committed_steps')
                == status.get('requested_steps') == 100000 and status.get('run_config_sha256') == job['config_sha256']
                and status.get('latest_checkpoint') == pointer == job['final_pointer'], 'Incomplete recovered status/pointer')
        require(job.get('history', {}).get('graph_updates') == 100000
                and job['history'].get('logged_rows') == 1001, 'Complete original100k history audit required')
        for key, steps, has_adam in (('initial_checkpoint', 0, False), ('parent_checkpoint', 50000, True), ('endpoint', 100000, True)):
            evidence = job.get(key, {})
            require(evidence.get('completed_steps') == steps and type(evidence.get('finite_model_tensors')) is int
                    and evidence['finite_model_tensors'] > 0
                    and evidence.get('adam_parameter_states') == (evidence['finite_model_tensors'] if has_adam else 0)
                    and evidence.get('cpu_cuda_rng_serialized') is True and evidence.get('checkpoint_schema_verified') is True,
                    'Original/parent/final payload audit incomplete')
        origin, recovered = Path(job['origin_directory']), Path(job['directory'])
        require(recovered == Path(original['output_directory']) and origin != recovered
                and job['combined_scientific_checkpoint_steps'] == list(range(0, 100001, 10000)), 'Checkpoint lineage differs')
        for directory, inventory, steps in ((origin, job['origin_checkpoint_inventory'], range(0, 50001, 10000)),
                                            (recovered, job['recovered_checkpoint_inventory'], range(50000, 100001, 10000))):
            names = {f'checkpoint-{step:09d}.pt' for step in steps}
            require(set(inventory) == names and {Path(path).name for path in files
                    if Path(path).parent == directory and Path(path).name.startswith('checkpoint-') and Path(path).suffix == '.pt'} == names,
                    'Exact split checkpoint inventory required')
            require(all(files[str(directory / name)]['sha256'] == value and files[str(directory / name)]['bytes'] > 0
                        for name, value in inventory.items()), 'Audited checkpoint bytes changed')
        require(job['origin_checkpoint_inventory'] == original['origin_checkpoint_inventory']
                and job['recovered_checkpoint_inventory']['checkpoint-000050000.pt'] == original['origin_checkpoint_sha256'],
                'Copied50k parent differs')
        require(all(files[str(path)]['sha256'] == value for path, value in
                    ((origin / 'protocol.json', original['origin_protocol_sha256']),
                     (origin / 'latest.json', original['origin_pointer_sha256']),
                     (origin / 'run.lock', original['origin_lock_sha256']),
                     (Path(original['origin_stdout_path']), original['origin_stdout_sha256']),
                     (recovered / 'protocol.json', original['origin_protocol_sha256']),
                     (Path(original['replay_receipt_path']), original['replay_receipt_sha256']))),
                'Original protocol/pointer/lock/log or replay receipt changed')
        for key, directory, step in (('initial_pointer', origin, 0), ('final_pointer', recovered, 100000)):
            name = f'checkpoint-{step:09d}.pt'
            require(job[key] == {'path': name, 'completed_steps': step, 'run_config_sha256': job['config_sha256'],
                                'sha256': files[str(directory / name)]['sha256']}, 'Original/final pointer differs')
        require(set(job['artifact_sha256']) == {'protocol.json', 'status.json', 'history.json', 'latest.json',
                                               'runtime_migration.json', 'migration_receipt.json'}, 'Recovery sidecars missing')
        require(all(files[str(recovered / name)]['sha256'] == value for name, value in job['artifact_sha256'].items()),
                'Recovery artifact bytes changed')
        external = job['external']
        command = [release['python_environment']['lexical_path'], '-B', release['adapter_path'], '--execute', '--mode',
                   'train', '--release', release_path, '--release-sha256', RECOVERY_RELEASE_SHA, '--job', job['id']]
        require(external.get('exit_code') == 0 and external.get('signals') == [] and external.get('command') == command
                and external['identity']['argv'] == command and external['identity']['pid'] == external['pid']
                and external['identity']['ppid'] == process['owner_pid']
                and external.get('owner_release_sha256') == RECOVERY_RELEASE_SHA,
                'Exact clean owned recovery execution required')
        require(all(files[external[k + '_file']]['sha256'] == external[k + '_sha256'] for k in ('stdout', 'stderr')),
                'Recovery execution log bytes changed')
    pairs = wave.get('pairing', [])
    require(len(pairs) == 2 and {p.get('seed') for p in pairs} == {1, 2}
            and all(type(p['seed']) is int and all(p.get(k) is True for k in
                ('initial_model_adam_cpu_cuda_rng_exact', 'all100k_frame_noise_native_graph_rng_exact',
                 'all_logged_saved_lr_exact', 'original50k_prefix_preserved_in_each_arm'))
                and p.get('cross_physical_device_bitwise_equivalence_claim') is False
                and p.get('initial_audit_ordering') == 'verified after recovery training; no before-step1 barrier claimed' for p in pairs),
            'Full original100k paired recovery audit required')
    return jobs, pairs


def inventory_a(recovery_release, process_check):
    H = helper()
    require(sha(recovery_release) == RECOVERY_RELEASE_SHA, 'Only the actual admitted recovery release is supported')
    release = read(recovery_release);owner = absolute(release['owner_output_directory'])
    process = read(process_check)
    # Read only small control/owner JSON until the completion gate is proven.
    owner_controls = {name: owner / name for name in ('owner_started.json', 'wave_A.json', 'owner_terminal.json')}
    control_pins = {str(absolute(recovery_release)): RECOVERY_RELEASE_SHA,
                    **{str(path): sha(path) for path in owner_controls.values()}}
    started, wave, terminal = [read(owner_controls[name]) for name in owner_controls]
    require(process.get('schema') == PROCESS_SCHEMA and process.get('status') == 'verified_completed_recovered_A'
            and process.get('issued_by') == 'root' and process.get('files_sha256') == control_pins
            and process.get('hostname') == release['host'] and process.get('owner_release_sha256') == RECOVERY_RELEASE_SHA
            and process.get('owner_pid') == started.get('owner_pid') == wave.get('owner_identity', {}).get('pid')
            and process.get('owner_exited') is True and process.get('all_children_reaped') is True
            and process.get('matching_training_processes') == [] and process.get('scoped_gpu_inventory') == []
            and started.get('mode') == wave.get('mode') == terminal.get('mode') == 'train'
            and wave.get('state') == 'verified_recovery_endpoints' and wave.get('error') is None
            and wave.get('unreaped_owned_children') == [] and wave.get('never_started') == []
            and wave.get('final_gpu_inventory') == [] and len(wave.get('verified_jobs', [])) == 4
            and len(wave.get('pairing', [])) == 2 and terminal.get('status') == 'complete_stopped_and_reaped'
            and terminal.get('wave_sha256') == control_pins[str(owner / 'wave_A.json')]
            and stamp(terminal['terminal_utc']) <= stamp(process['checked_utc'])
            and 0 <= (now() - stamp(process['checked_utc'])).total_seconds() <= 1800,
            'No payload-byte inventory before root completed-owner/native-closure gate')
    require(all(sha(HERE / name) == value for name, value in SOURCE_PINS.items()), 'Frozen source closure differs')
    roots = [owner] + [absolute(j['origin_directory']) for j in release['jobs']] + [absolute(j['output_directory']) for j in release['jobs']]
    trees, files = tree_inventory(roots)
    extra = [absolute(recovery_release), absolute(process_check), Path(__file__).resolve()]
    extra += [HERE / name for name in SOURCE_PINS]
    extra += [absolute(j['origin_stdout_path']) for j in release['jobs']]
    extra += [absolute(j['replay_receipt_path']) for j in release['jobs']]
    extra += [absolute(value) for key, value in release.items() if key.endswith('_path') and isinstance(value, str)]
    for path in extra: files[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    require(all(files[str(HERE / name)]['sha256'] == value for name, value in SOURCE_PINS.items()), 'Frozen source closure differs')
    require(all(files[path]['sha256'] == value for path, value in control_pins.items()), 'Owner controls changed after completion gate')
    for key, value in release.items():
        if key.endswith('_path') and isinstance(value, str):
            expected = release.get(key[:-5] + '_sha256', release['files_sha256'].get(value))
            require(digest(expected) and files[value]['sha256'] == expected, 'Archived source/receipt binding differs: ' + key)
    for directory in [str(owner)] + [job['output_directory'] for job in release['jobs']]:
        names = trees[directory]
        require(not any(Path(name).name == 'run.lock' or name.endswith('.tmp')
                        or Path(name).name.startswith(('unsuccessful', 'state_preservation_error'))
                        or Path(name).name == 'migration_failure.json' for name in names),
                'Recovered output contains a failure, lock or partial file; preserve without promotion')
    def saved(path):
        raw = Path(path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == files[str(path)]['sha256'], 'Receipt changed during inventory')
        return json.loads(raw)
    receipts = {'release_path': str(absolute(recovery_release)), 'release': saved(recovery_release), 'process': saved(process_check),
                **{key: saved(owner / name) for key, name in (('started', 'owner_started.json'), ('wave', 'wave_A.json'),
                                                             ('terminal', 'owner_terminal.json'))},
                'jobs': {j['id']: {key: saved(Path(j['output_directory']) / name) for key, name in
                    (('config', 'protocol.json'), ('status', 'status.json'), ('pointer', 'latest.json'))} for j in release['jobs']}}
    check_a(receipts, files, H)
    require(0 <= (now() - stamp(process['checked_utc'])).total_seconds() <= 1800, 'Fresh post-completion native closure required')
    result = {'schema': INVENTORY_SCHEMA, 'status': 'completed_recovered_A_bytes_verified', 'issued_by': 'root',
              'source_sha256': sha(__file__), 'source_pins': SOURCE_PINS, 'cohort_id': COHORT_ID, 'host_role': 'A',
              'created_utc': now().isoformat(), 'trees': trees, 'files': files, 'receipts': receipts,
              'process_check_sha256': sha(process_check),
              'inventory_evidence_bindings': {path: row['sha256'] for path, row in files.items()},
              'fresh_tensor_deserialization': False, 'reserved_test_accessed': False,
              'audit_basis': 'Frozen recovery owner/auditor verified original0,parent50k,final100k payloads and complete paired histories; this inventory rehashes retained bytes only.',
              'historical_failed_attempts_remain_failed': True,
              'cross_physical_device_bitwise_equivalence_claim': False}
    verify_trees(result)
    return result


def build(inventory_a_path, inventory_b_path, release, sources):
    H = helper()
    require(release.get('schema') == RELEASE_SCHEMA and release.get('status') == 'approved_complete_recovery_cohort_bridge'
            and release.get('issued_by') == 'root' and release.get('adapter_sha256') == sha(__file__)
            and release.get('cohort_id') == COHORT_ID and release.get('recovery_release_sha256') == RECOVERY_RELEASE_SHA
            and release.get('original_B_adapter_sha256') == SOURCE_PINS['prepare_sand_final_cohort_scoped_v1.py']
            and isinstance(release.get('review_rationale'), str) and release['review_rationale'].strip(),
            'Explicit root-reviewed completed all-six bridge release required')
    require(set(sources) == set(H.FINAL_PINS) and release.get('final_source_sha256') == H.FINAL_PINS
            and all(sha(sources[k]) == value for k, value in H.FINAL_PINS.items()), 'Frozen final sources differ')
    a, b = read(inventory_a_path), read(inventory_b_path)
    require(release.get('inventory_sha256') == {'A': sha(inventory_a_path), 'B': sha(inventory_b_path)}, 'Exact root-bound inventories required')
    require(a.get('schema') == INVENTORY_SCHEMA and a.get('status') == 'completed_recovered_A_bytes_verified'
            and a.get('issued_by') == 'root' and a.get('source_sha256') == sha(__file__) and a.get('source_pins') == SOURCE_PINS
            and a.get('cohort_id') == COHORT_ID and a.get('host_role') == 'A', 'Completed recovered A inventory required')
    jobs_a, pairs_a = check_a(a['receipts'], a['files'], H)
    require(b.get('schema') == H.INVENTORY_SCHEMA and b.get('status') == 'completed_host_bytes_verified'
            and b.get('issued_by') == 'root' and b.get('host_role') == 'B' and b.get('cohort_id') == COHORT_ID
            and b.get('source_sha256') == SOURCE_PINS['prepare_sand_final_cohort_scoped_v1.py']
            and H.check_receipts(b['receipts'], b['files']) == 'B', 'Unchanged original completed B inventory required')
    jobs_b, pairs_b = b['receipts']['summary']['jobs'], b['receipts']['summary']['pairing']
    models, audited, pairs = [], [], []
    for role, jobs in (('A', jobs_a), ('B', jobs_b)):
        for job in jobs:
            row = {'objective': 'faithful', 'arm': job['arm'], 'seed': job['seed'], 'completed_steps': 100000,
                   'checkpoint_sha256': job['final_pointer']['sha256'], 'host_role': role,
                   'checkpoint_path': str(Path(job['directory']) / job['final_pointer']['path']), 'config_sha256': job['config_sha256']}
            models.append(row)
            audited.append({**row, 'graph_history_updates': 100000, 'checkpoint_every': 10000, 'log_every': 100,
                            'all_optimizer_steps_equal_100000': True, 'all_state_and_moments_finite': True,
                            'source_data_protocol_verified': True, 'checkpoint_bytes_verified': True,
                            'initial_checkpoint_sha256': job['initial_pointer']['sha256'], 'source_endpoint_report': job['endpoint'],
                            'payload_audit_basis': 'frozen recovery auditor original/parent/final checks' if role == 'A' else 'frozen original scoped supervisor initial/final checks',
                            'runtime_migration': role == 'A', 'recovery_is_fresh_from_scratch': False if role == 'A' else None})
    for pair in pairs_a + pairs_b:
        pairs.append({'seed': pair['seed'], 'initial_model_tensor_identity': True, 'initial_cpu_cuda_rng_identity': True,
                      'all_frame_noise_lr_schedules_equal': True, 'all_graph_budgets_and_rng_material_verified': True,
                      'source_pairing_report': pair})
    require(len(models) == 6 and {(m['arm'], m['seed']) for m in models} == {(a, s) for a in ('base', 'mix') for s in range(3)}
            and len({m['checkpoint_sha256'] for m in models}) == 6 and len(pairs) == 3 and {p['seed'] for p in pairs} == {0, 1, 2},
            'Original complete six-model/three-pair cohort required')
    bindings = merge({str(Path(sources[k]).resolve()): value for k, value in H.FINAL_PINS.items()},
                     {str(Path(inventory_a_path).resolve()): sha(inventory_a_path), str(Path(inventory_b_path).resolve()): sha(inventory_b_path),
                      str(Path(__file__).resolve()): sha(__file__), str(HERE / 'prepare_sand_final_cohort_scoped_v1.py'): SOURCE_PINS['prepare_sand_final_cohort_scoped_v1.py']})
    common = {'issued_by': 'root', 'dataset': 'Sand', 'training_schema': H.TRAIN_SCHEMA, 'protocol_sha256': H.FINAL_PINS['protocol'],
              'training_admission_sha256': H.FINAL_PINS['train_admission'], 'trainer_source_sha256': H.FINAL_PINS['trainer_source'],
              'cohort_id': COHORT_ID, 'adapter_sha256': sha(__file__), 'cost_basis': H.COST_BASIS, 'created_utc': now().isoformat(),
              'schedule_plan_sha256': H.PINS['schedule_plan'], 'operational_amendment_sha256': H.PINS['amendment'],
              'runtime_migration_recovery_release_sha256': RECOVERY_RELEASE_SHA}
    audit = {**common, 'schema': 'adaptgns_sand_graph_support_complete_cohort_audit_v1', 'status': 'all_six_endpoints_and_pairing_verified',
             'models': audited, 'paired_seeds': sorted(pairs, key=lambda p: p['seed']),
             'provenance': {'A': {'kind': 'original50k_plus_reviewed_recovered50k', 'inventory_sha256': sha(inventory_a_path),
                                  'receipts': a['receipts'], 'retained_file_bytes': a['files']},
                            'B': {'kind': 'original_uninterrupted100k', 'inventory_sha256': sha(inventory_b_path),
                                  'receipts': b['receipts'], 'retained_file_bytes': b['files']}},
             'input_binding_sha256': bindings, 'fresh_tensor_deserialization_by_adapter': False,
             'reserved_test_accessed_by_adapter': False, 'intermediate_payloads_deserialized': False,
             'cross_physical_device_bitwise_equivalence_claim': False,
             'audit_scope': 'Original six100k models. Recovered A retains original0..50k and full100k pairing; original B remains unchanged. Byte inventories inherit frozen payload audits without a second tensor audit.'}
    cohort = {**common, 'schema': 'adaptgns_sand_graph_support_final_cohort_v1', 'status': 'frozen_for_final_evaluation',
              'updates': 100000, 'models': models, 'policies': H.POLICIES, 'deterministic_algorithms': True,
              'cublas_workspace_config': ':4096:8', 'benchmark_helper_sha256': H.FINAL_PINS['benchmark_helper'],
              'diagnostic_source_sha256': H.FINAL_PINS['diagnostic_source']}
    return audit, cohort


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true');p.add_argument('--mode', choices=('inventory-a', 'build'))
    for name in ('recovery-release', 'process-check', 'output', 'inventory-a', 'inventory-b', 'root-release', 'output-dir',
                 'protocol', 'train-admission', 'trainer-source', 'benchmark-helper', 'diagnostic-source'):
        p.add_argument('--' + name, type=Path)
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'status': 'description_only_unadmitted', 'tensor_or_test_access': False,
                          'operations': ['inventory completed recovered A', 'join reviewed A and unchanged B inventories'],
                          'requires': 'root post-exit closure, all four recovery endpoints/pairing and original two B endpoints'}, indent=2));return 0
    if a.mode == 'inventory-a':
        require(a.recovery_release and a.process_check and a.output, 'Explicit recovered release/process check/new output required')
        inv = inventory_a(a.recovery_release.resolve(), a.process_check.resolve())
        require(all(Path(root) not in a.output.resolve().parents for root in inv['trees']), 'Inventory must lie outside retained trees')
        raw = encode(inv);verify_trees(inv)
        require(0 <= (now() - stamp(inv['receipts']['process']['checked_utc'])).total_seconds() <= 1800, 'Native closure expired')
        with a.output.open('xb') as f:f.write(raw)
    elif a.mode == 'build':
        require(a.inventory_a and a.inventory_b and a.root_release and a.output_dir, 'Both inventories/root release/new output required')
        H = helper();sources = {k: getattr(a, k) for k in H.FINAL_PINS}
        require(all(sources.values()), 'All frozen final source paths required')
        raw = a.root_release.read_bytes();release = json.loads(raw)
        audit, cohort = build(a.inventory_a, a.inventory_b, release, sources)
        release_sha = hashlib.sha256(raw).hexdigest();audit['root_release_sha256'] = release_sha
        audit['input_binding_sha256'][str(a.root_release.resolve())] = release_sha
        audit_raw = encode(audit);cohort['cohort_audit_sha256'] = hashlib.sha256(audit_raw).hexdigest()
        cohort['root_release_sha256'] = release_sha;cohort_raw = encode(cohort)
        a.output_dir.mkdir(exist_ok=False)
        (a.output_dir / 'root_release.json').write_bytes(raw)
        try:
            verify(audit['input_binding_sha256'])
            require(sha(a.output_dir / 'root_release.json') == release_sha, 'Copied root release changed')
            with (a.output_dir / 'cohort_audit.json').open('xb') as f:f.write(audit_raw)
            verify(audit['input_binding_sha256'])
            require(sha(a.output_dir / 'cohort_audit.json') == cohort['cohort_audit_sha256']
                    and sha(a.output_dir / 'root_release.json') == release_sha, 'Published audit/root receipt changed')
            with (a.output_dir / 'cohort.json').open('xb') as f:f.write(cohort_raw)
        except BaseException as error:
            with (a.output_dir / 'failed_publication.json').open('xb') as f:
                f.write(encode({'error_type': type(error).__name__, 'error': str(error), 'all_existing_outputs_retained': True}))
            raise
    else:
        p.error('Execution requires explicit mode')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
