#!/usr/bin/env python3
"""Read-only CPU endpoint audit for explicit Sand 50k-to-100k recovery.

The original frozen check_history/check_payload numerical predicates are reused
unchanged. Historical run_config is retained; current runtime is separate.
No CUDA, model construction, training, source changes, control issuance or writes.
Only an owner that has reaped every owned child and closed its GPU scope may call.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
SCHEMA = 'adaptgns_sand_runtime_migration_endpoint_audit_v1'
RELEASE_SCHEMA = 'adaptgns_sand_runtime_migration_recovery_release_v1'
PARENT = 50000
ENDPOINT = 100000
VERIFIER_SHA = 'a22a59c46231c328e4939a0f021b3bb2fab9e2f93c0848bb1abd39aad2d5d305'
TRAINER_SHA = 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
CAPACITY_SHA = 'ca9b5ed45455b575179464d35c42d0d03e87edb7508f4deea2451f9f3fb50878'
LIFECYCLE_SHA = 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd'
PROTOCOL_SHA = 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d'
JOBS = [('base_seed1', 'base', 1, 0), ('mix_seed1', 'mix', 1, 1), ('base_seed2', 'base', 2, 2), ('mix_seed2', 'mix', 2, 3)]
SOURCE_FIELDS = {'verifier_source_path': VERIFIER_SHA, 'capacity_source_path': CAPACITY_SHA,
                 'lifecycle_source_path': LIFECYCLE_SHA, 'original_trainer_path': TRAINER_SHA}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def absolute(value):
    p = Path(value)
    require(p.is_absolute() and str(p) == str(p.resolve()), 'Canonical absolute nonsymlink path required')
    return p


def stamp(value):
    t = datetime.fromisoformat(value)
    require(t.tzinfo is not None, 'Timezone-aware external evidence required')
    return t.astimezone(timezone.utc)


def deadline(external):
    import math
    error = external.get('clock_error_bound_seconds')
    require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 5,
            'Bounded owner clock required for endpoint audit')
    stop = stamp(external['training_stop_utc'])
    require(stop == datetime(2026, 10, 6, 22, 44, tzinfo=timezone.utc)
            and datetime.now(timezone.utc) + timedelta(seconds=error) < stop,
            'Original training/endpoint-audit stop has passed')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(release):
    pins = release['files_sha256']
    for key, expected in SOURCE_FIELDS.items():
        path = absolute(release[key])
        require(sha(path) == pins.get(str(path)) == expected, 'Frozen audit source differs: ' + key)
    S = load(release['verifier_source_path'], '_sand_recovery_frozen_endpoint_verifier')
    S.configure(SimpleNamespace(capacity_source=Path(release['capacity_source_path']),
        lifecycle_source=Path(release['lifecycle_source_path']), trainer=Path(release['original_trainer_path'])))
    return S


def bound_release(release):
    path = absolute(release['_release_path'])
    require(sha(path) == release['_release_sha256'], 'Root release bytes changed')
    require(read(path) == {k: v for k, v in release.items() if not k.startswith('_')}, 'Supplied release differs from issued bytes')
    require(release.get('schema') == RELEASE_SCHEMA and release.get('issued_by') == 'root'
            and release.get('status') == 'approved_for_sand_migration_training'
            and release.get('required_models') == 6 and release.get('required_policies') == 6
            and release.get('automatic_retry_resume_or_promotion') is False, 'Explicit complete-study migration release required')
    require([(j.get('id'), j.get('arm'), j.get('seed'), j.get('gpu')) for j in release['jobs']] == JOBS,
            'All four exact original A jobs required')
    require(absolute(release['auditor_path']) == Path(__file__).resolve()
            and release['auditor_sha256'] == sha(__file__), 'Exact reviewed endpoint auditor required')
    pins = release['files_sha256']
    require(isinstance(pins, dict) and pins, 'Absolute release input inventory required')
    for p, digest in pins.items():
        # The lexical interpreter may intentionally name a virtualenv symlink.
        require(Path(p).is_absolute() and (str(Path(p).resolve()) == p or p == release['python_environment']['lexical_path'])
                and sha(p) == digest, 'Pinned original/source/control changed: ' + p)
    for key in (*SOURCE_FIELDS, 'adapter_path', 'auditor_path', 'train_manifest', 'admission', 'structural_report', 'protocol'):
        require(pins.get(release[key]) == sha(release[key]), 'Required release input omitted: ' + key)
    require(sha(release['protocol']) == PROTOCOL_SHA, 'Scientific protocol changed')


def external_contract(release, job, external):
    deadline(external)
    require(external.get('exit_code') == 0 and external.get('signals') == []
            and type(external.get('pid')) is int and external['pid'] > 0, 'Clean reaped owned child required')
    command = [release['python_environment']['lexical_path'], '-B', release['adapter_path'], '--execute', '--mode', 'train',
               '--release', release['_release_path'], '--release-sha256', release['_release_sha256'], '--job', job['id']]
    require(external.get('command') == command and external.get('identity', {}).get('argv') == command
            and external['identity'].get('pid') == external['pid'], 'Exact owned recovery command/process identity required')
    require(external.get('owner_release_sha256') == release['_release_sha256'], 'Owner release identity differs')
    closure = external.get('scoped_gpu_closure', {})
    require(closure.get('verified') is True and closure.get('owned_children_reaped') is True and closure.get('rows') == [],
            'Every A GPU scope must be empty after all owned children were reaped')
    require(stamp(external['started_utc']) <= stamp(external['reaped_utc']) <= stamp(closure['checked_utc']),
            'Owner launch/reap/closure chronology differs')
    require(stamp(external['started_utc']) + timedelta(seconds=external['clock_error_bound_seconds']) <= stamp(release['latest_start_utc']),
            'Recovery launch exceeded original latest-start gate')
    for name in ('stdout', 'stderr'):
        p = absolute(external[name + '_file'])
        require(sha(p) == external[name + '_sha256'], 'Owned ' + name + ' bytes changed')


def original_inventory(release, job):
    origin = absolute(job['origin_directory'])
    expected = {f'checkpoint-{step:09d}.pt' for step in range(0, PARENT + 1, 10000)}
    inventory = job.get('origin_checkpoint_inventory')
    require(isinstance(inventory, dict) and set(inventory) == expected, 'Every original 0..50k checkpoint must be pinned')
    require({p.name for p in origin.glob('checkpoint-*.pt')} == expected, 'Original committed checkpoint inventory changed')
    for name, digest in inventory.items():
        path = absolute(str(origin / name))
        require(sha(path) == digest == release['files_sha256'].get(str(path)), 'Original checkpoint changed: ' + name)
    require(inventory['checkpoint-000050000.pt'] == job['origin_checkpoint_sha256'], 'Parent checkpoint inventory differs')
    return origin, inventory


def recipe(S, release, job, config):
    T, C = S.T, S.C
    require(config.get('schema') == T.SCHEMA and config.get('dataset') == 'Sand' and config.get('objective') == 'faithful'
            and config.get('arm') == job['arm'] and config.get('seed') == job['seed'] and config.get('updates') == ENDPOINT
            and config.get('checkpoint_every') == 10000 and config.get('log_every') == 100
            and config.get('research_protocol_sha256') == PROTOCOL_SHA
            and config.get('initialization') == 'from scratch; paired seed across arms; empty Adam; no parent checkpoint'
            and config.get('selection') == 'fixed final requested update; no validation/test selection'
            and config.get('graph_exposure') == T.graph_exposure_config(job['arm'])
            and config.get('source_sha256') == {**T.SOURCE_PINS, 'train_sand_graph_support_cuda.py': TRAINER_SHA},
            'Historical scientific lineage/recipe differs')
    require(config.get('batch_size') == 2 and config.get('history') == 6 and config.get('noise_std') == 6.7e-4
            and config.get('architecture') == {'width': 128, 'message_passing_blocks': 10, 'mlp_layers': 2}
            and config.get('graph') == {'radius': .015, 'backend': 'scipy_host', 'cap': 128, 'self_candidates': True, 'augmentation_probability': 0.}
            and config.get('optimizer') == {'name': 'Adam', 'initial_lr': 1e-4, 'final_lr': 1e-5, 'decay_updates': 100000,
                'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'foreach': False, 'fused': False, 'gradient_clipping': None},
            'Scientific architecture/noise/optimizer metadata differs')
    data = config.get('data', {})
    require(all(data.get(k) == C.DATA_PINS[v] == sha(release[v]) for k, v in
                (('manifest_sha256', 'train_manifest'), ('admission_sha256', 'admission'), ('structural_report_sha256', 'structural_report')))
            and data.get('frames_per_trajectory') == 320 and data.get('particle_type_ids') == [6], 'Scientific data lineage differs')
    for relative, digest in T.SOURCE_PINS.items():
        require(sha(Path(release['repo']) / relative) == digest, 'Frozen numerical source changed: ' + relative)



def verify_training_bytes(S, release):
    manifest_path = absolute(release['train_manifest'])
    root = manifest_path.parent
    manifest, admission = read(manifest_path), read(release['admission'])
    S.T.validate_manifest_contract(manifest, admission)
    require(sha(manifest_path) == admission['manifest_sha256']
            and sha(release['structural_report']) == admission['structural_report_sha256'], 'Training admission byte identity differs')
    metadata = root / 'metadata.json'
    require(sha(metadata) == S.T.METADATA_SHA and read(metadata) == manifest['metadata'], 'Actual metadata changed')
    total = 0
    for record in manifest['records']:
        for name in ('positions', 'particle_types'):
            descriptor = record[name]
            path = absolute(str(root / descriptor['path']))
            require(root in path.parents and path.stat().st_size == descriptor['size_bytes']
                    and sha(path) == descriptor['sha256'], 'Actual training array bytes changed: ' + str(path))
            total += path.stat().st_size
    return {'numeric_files': 2 * len(manifest['records']), 'numeric_bytes': total, 'metadata_sha256': sha(metadata)}


def check_recovery_history(parent_history, history, stdout, status):
    suffix = [row for row in history['training'] if row['completed_steps'] > PARENT]
    require(len(suffix) == 500 and stdout == suffix and status.get('last_training') == suffix[-1],
            'Recovery stdout/status must equal only the500 new logged updates')
    require([row['completed_steps'] for row in parent_history['training']] == [1, *range(100, PARENT + 1, 100)]
            and history['graph_updates'][:PARENT] == parent_history['graph_updates']
            and history['training'][:len(parent_history['training'])] == parent_history['training']
            and history['elapsed_seconds'] >= parent_history['elapsed_seconds']
            and suffix[0]['elapsed_seconds'] > parent_history['elapsed_seconds'],
            'Complete50k history prefix or cumulative elapsed time changed')


def check_resumed_legacy(adapter, job, stdout):
    require({k: stdout[0][k] for k in adapter.LEGACY_FIELDS} == adapter.legacy_replay_endpoint(job),
            'Resumed scientific50100 scalar differs from original/replay50100 evidence')


def sidecars(A, release, job, config, pointer):
    expected = A.identity_record(release, job, config, job['actual_runtime'])
    directory = Path(job['output_directory'])
    require(read(directory / 'runtime_migration.json') == expected, 'Pretraining migration identity differs')
    replay = A.check_replay_receipt(release, job)
    require(Path(job['replay_receipt_path']) == Path(job['replay_output_directory']) / 'receipt.json'
            and not list(Path(job['replay_output_directory']).rglob('checkpoint-*.pt'))
            and not (Path(job['replay_output_directory']) / 'run.lock').exists()
            and not (Path(job['replay_output_directory']) / 'migration_failure.json').exists(), 'Replay must remain closed and non-scientific')
    require(replay.get('origin_protocol_sha256') == job['origin_protocol_sha256']
            and replay.get('origin_pointer_sha256') == job['origin_pointer_sha256']
            and replay.get('original_host') == release['original_host']
            and replay.get('origin_directory') == job['origin_directory']
            and replay.get('parent_completed_steps') == PARENT and replay.get('target_completed_steps') == ENDPOINT
            and replay.get('allowed_runtime_differences') == ['uuid']
            and replay.get('model_initialization_from_replay_artifact') is False
            and replay.get('absolute_schedule_steps') == list(range(PARENT, PARENT + 100)), 'Replay provenance differs')
    receipt = read(directory / 'migration_receipt.json')
    expected = {**expected, 'schema': 'adaptgns_sand_runtime_migration_completed_attempt_v1',
        'status': 'training_complete_pending_independent_endpoint_audit', 'final_pointer': pointer,
        'replay_receipt_sha256': job['replay_receipt_sha256'], 'completed_utc': receipt.get('completed_utc'),
        'original_artifacts_reverified': True}
    require(receipt == expected and stamp(replay['completed_utc']) <= stamp(receipt['completed_utc']), 'Completed migration receipt differs')
    return receipt


def audit_recovery_job(release, job, *, external):
    external_contract(release, job, external)
    bound_release(release)
    require(job == next(j for j in release['jobs'] if j['id'] == job['id']), 'Job differs from root release')
    S = configure(release)
    A = load(release['adapter_path'], '_sand_recovery_reviewed_migration_adapter')
    require(sha(release['adapter_path']) == release['adapter_sha256'], 'Reviewed adapter differs')
    origin, config, parent_pointer = A.verify_origin(release, job)
    origin, inventory = original_inventory(release, job)
    recipe(S, release, job, config)
    numeric_inputs = verify_training_bytes(S, release)
    deadline(external)
    directory = absolute(job['output_directory'])
    require(directory != origin and origin not in directory.parents and directory not in origin.parents, 'Recovery output overlaps historical origin')
    require(not (directory / 'run.lock').exists() and not list(directory.rglob('*.tmp'))
            and not list(directory.rglob('unsuccessful*')) and not list(directory.rglob('state_preservation_error*'))
            and not (directory / 'migration_failure.json').exists(), 'Failed, locked or partial recovery cannot pass')
    require(sha(directory / 'protocol.json') == job['origin_protocol_sha256'] and read(directory / 'protocol.json') == config,
            'Historical protocol was changed in recovery')
    config_sha = S.B.canonical_hash(config)
    status, history, pointer = [read(directory / name) for name in ('status.json', 'history.json', 'latest.json')]
    require(status.get('schema') == S.T.SCHEMA and status.get('state') == 'complete' and status.get('error') is None
            and status.get('completed_steps') == status.get('committed_steps') == status.get('requested_steps') == ENDPOINT
            and status.get('run_config_sha256') == config_sha and status.get('arm') == job['arm'] and status.get('seed') == job['seed']
            and status.get('objective') == 'faithful' and status.get('process', {}).get('pid') == external['pid'],
            'Only a clean original 100k endpoint passes')
    attempts = list((directory / 'attempts').iterdir())
    require(len(attempts) == 1 and attempts[0].is_dir() and attempts[0].name == status.get('attempt_id')
            and read(attempts[0] / 'status.json') == status, 'Exactly one consistent new recovery attempt required')
    require(stamp(external['started_utc']) <= stamp(status['updated_utc']) <= stamp(external['reaped_utc']), 'Completed status lies outside owned execution')
    expected_files = {f'checkpoint-{step:09d}.pt' for step in range(PARENT, ENDPOINT + 1, 10000)}
    require({p.name for p in directory.glob('checkpoint-*.pt')} == expected_files, 'Recovery checkpoint set must be exact copied50k plus60..100k')
    recovered_inventory = {name: sha(directory / name) for name in sorted(expected_files)}
    require(recovered_inventory['checkpoint-000050000.pt'] == job['origin_checkpoint_sha256'], 'Copied parent changed')
    final_name = 'checkpoint-000100000.pt'
    expected_pointer = {'path': final_name, 'completed_steps': ENDPOINT, 'run_config_sha256': config_sha, 'sha256': recovered_inventory[final_name]}
    require(pointer == expected_pointer and status.get('latest_checkpoint') == pointer, 'Final committed pointer differs')
    receipt = sidecars(A, release, job, config, pointer)
    require(stamp(status['updated_utc']) <= stamp(receipt['completed_utc']) <= stamp(external['reaped_utc']), 'Migration completion lies outside owned execution')
    artifact_names = ['protocol.json', 'status.json', 'history.json', 'latest.json', 'runtime_migration.json', 'migration_receipt.json']
    artifacts = {name: sha(directory / name) for name in artifact_names}
    deadline(external)
    details = S.check_history(history, job, read(release['train_manifest']))
    stdout = [json.loads(line) for line in Path(external['stdout_file']).read_text().splitlines() if line.strip()]
    import torch  # CPU weights_only deserialization; no CUDA API call or model construction.
    parent = torch.load(origin / parent_pointer['path'], map_location='cpu', weights_only=True)
    parent_history = parent.get('history', {})
    S.T.validate_graph_history(parent_history, PARENT, job['arm'], job['seed'])
    parent_check = S.check_payload(torch, parent, config, PARENT, parent_history)
    check_recovery_history(parent_history, history, stdout, status)
    check_resumed_legacy(A, job, stdout)
    parent_simulator = parent['simulator_config']
    del parent, parent_history
    deadline(external)
    final = torch.load(directory / final_name, map_location='cpu', weights_only=True)
    endpoint = S.check_payload(torch, final, config, ENDPOINT, history)
    require(S.T.tree_equal(torch, parent_simulator, final['simulator_config']), 'Restored/final simulator normalization/configuration changed')
    final_simulator = final['simulator_config']
    del final
    initial_name = 'checkpoint-000000000.pt'
    initial = torch.load(origin / initial_name, map_location='cpu', weights_only=True)
    initial_history = initial.get('history', {})
    require(initial_history == {'training': [], 'graph_updates': [], 'elapsed_seconds': initial_history.get('elapsed_seconds')},
            'Historical initial checkpoint must precede every update')
    initial_check = S.check_payload(torch, initial, config, 0, initial_history)
    require(S.T.tree_equal(torch, initial['simulator_config'], final_simulator), 'Original initialization/final normalization differs')
    del initial
    # All checks are read-only and repeated byte checks surround deserialization.
    deadline(external)
    bound_release(release)
    original_inventory(release, job)
    recipe(S, release, job, config)
    require(verify_training_bytes(S, release) == numeric_inputs, 'Numeric input census changed during audit')
    deadline(external)
    require(all(sha(directory / name) == digest for name, digest in recovered_inventory.items())
            and all(sha(directory / name) == digest for name, digest in artifacts.items()), 'Recovery artifacts changed during audit')
    external_contract(release, job, external)
    return {**job, 'schema': SCHEMA, 'status': 'verified_recovered_scientific100k_endpoint', 'directory': str(directory),
        'recovery_release_path': release['_release_path'], 'recovery_release_sha256': release['_release_sha256'],
        'config_sha256': config_sha, 'runtime': config['runtime'], 'actual_runtime': job['actual_runtime'],
        'historical_runtime_is_original_lineage': True, 'cross_physical_device_bitwise_equivalence_claim': False,
        'recovery_is_fresh_from_scratch': False, 'original_initialization_from_scratch_verified': True,
        'parent_completed_steps': PARENT, 'new_completed_updates': ENDPOINT - PARENT, 'parent_history_prefix_exact': True,
        'original_lock_retained': True, 'origin_checkpoint_inventory': inventory, 'recovered_checkpoint_inventory': recovered_inventory,
        'combined_scientific_checkpoint_steps': list(range(0, ENDPOINT + 1, 10000)), 'external': external,
        'initial_pointer': {'path': initial_name, 'completed_steps': 0, 'run_config_sha256': config_sha, 'sha256': inventory[initial_name]},
        'final_pointer': pointer, 'history': details, 'endpoint': endpoint, 'parent_checkpoint': parent_check,
        'initial_checkpoint': initial_check, 'numeric_input_bytes_verified_before_and_after': numeric_inputs, 'artifact_sha256': artifacts, 'audit_completed_utc': datetime.now(timezone.utc).isoformat()}


def audit_recovery_pairing(verified_jobs):
    require(len(verified_jobs) == 4 and {(j.get('id'), j.get('arm'), j.get('seed'), j.get('gpu')) for j in verified_jobs} == set(JOBS),
            'All four verified original recovery jobs required')
    require(all(j.get('status') == 'verified_recovered_scientific100k_endpoint' and j.get('schema') == SCHEMA for j in verified_jobs),
            'Endpoint audits must precede pairing')
    paths = {(j['recovery_release_path'], j['recovery_release_sha256']) for j in verified_jobs}
    require(len(paths) == 1, 'One exact recovery release required')
    path, digest = next(iter(paths))
    require(sha(path) == digest, 'Pairing release changed')
    release = {**read(path), '_release_path': path, '_release_sha256': digest}
    bound_release(release)
    S = configure(release)
    import torch
    by_id = {j['id']: j for j in verified_jobs}
    results = []
    for seed in (1, 2):
        jobs = [by_id[f'{arm}_seed{seed}'] for arm in ('base', 'mix')]
        for job in jobs:
            deadline(job['external'])
            require(all(sha(Path(job['directory']) / name) == digest for name, digest in job['artifact_sha256'].items()),
                    'Verified history/control changed before pairing')
            require(sha(Path(job['origin_directory']) / job['initial_pointer']['path']) == job['initial_pointer']['sha256'],
                    'Original paired initial bytes changed')
        initial = [torch.load(Path(j['origin_directory']) / j['initial_pointer']['path'], map_location='cpu', weights_only=True) for j in jobs]
        require(all(S.T.tree_equal(torch, initial[0][key], initial[1][key]) for key in ('state_dict', 'simulator_config', 'optimizer_state', 'rng_states')),
                'Original paired model/Adam/CPU-CUDA RNG tensors differ')
        del initial
        histories = [read(Path(j['directory']) / 'history.json') for j in jobs]
        require(all(len(h['graph_updates']) == ENDPOINT and len(h['training']) == 1001 for h in histories), 'Complete paired100k histories required')
        for left, right in zip(histories[0]['training'], histories[1]['training']):
            require(all(left[k] == right[k] for k in ('completed_steps', 'frame_ids', 'particles', 'lr')), 'Paired saved LR/frame schedule differs')
        for left, right in zip(histories[0]['graph_updates'], histories[1]['graph_updates']):
            require(all(left[k] == right[k] for k in ('completed_steps', 'absolute_schedule_step', 'frame_ids', 'noise_sha256')),
                    'Paired full100k frame/noise schedule differs')
            require(len(left['examples']) == len(right['examples']) == 2, 'Two paired examples required')
            for x, y in zip(left['examples'], right['examples']):
                require(all(x[k] == y[k] for k in ('example_slot', 'n_particles', 'exposure_coin', 'coin_seed_material', 'pair_seed_material',
                    'native_directed_edges', 'native_self_edges', 'receivers_above_native_cap', 'annulus_pairs', 'optional_budget_if_exposed',
                    'native_edge_sha256', 'noisy_current_sha256')), 'Paired native graph/RNG evidence differs')
        for job in jobs:
            deadline(job['external'])
            require(sha(Path(job['directory']) / 'history.json') == job['artifact_sha256']['history.json']
                    and sha(Path(job['origin_directory']) / job['initial_pointer']['path']) == job['initial_pointer']['sha256'],
                    'Paired history/initial checkpoint changed during inspection')
        results.append({'seed': seed, 'initial_model_adam_cpu_cuda_rng_exact': True,
            'all100k_frame_noise_native_graph_rng_exact': True, 'all_logged_saved_lr_exact': True,
            'original50k_prefix_preserved_in_each_arm': True,
            'initial_audit_ordering': 'verified after recovery training; no before-step1 barrier claimed',
            'cross_physical_device_bitwise_equivalence_claim': False})
    bound_release(release)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args()
    print(json.dumps({'schema': SCHEMA, 'status': 'description_only', 'read_only_cpu_audit': True,
        'callable': ['audit_recovery_job(release, job, *, external)', 'audit_recovery_pairing(verified_jobs)'],
        'requires_all_owned_children_reaped_and_scoped_gpu_closure': True,
        'original_frozen_numerical_predicates': ['check_history', 'check_payload'],
        'origin_steps': PARENT, 'endpoint_steps': ENDPOINT}))


if __name__ == '__main__':
    main()
