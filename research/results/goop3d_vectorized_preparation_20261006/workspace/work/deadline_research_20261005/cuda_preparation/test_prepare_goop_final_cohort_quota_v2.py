"""Tiny synthetic bytes only. Never read real checkpoints, data or CUDA."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cohort_adapter', HERE / 'prepare_goop_final_cohort_quota_v2.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


@pytest.fixture
def cohort_fixture(tmp_path, monkeypatch):
    pins = {k: M.sha(put(tmp_path / 'sources' / k, {'synthetic': k})) for k in M.PINS}
    # This unit fixture substitutes only fake source bytes. Production pins stay strict.
    monkeypatch.setattr(M, 'PINS', pins)
    final = {k: M.sha(put(tmp_path / 'final_sources' / k, {'synthetic_final': k})) for k in M.FINAL_PINS}
    final.update(protocol=pins['protocol'], train_admission=pins['admission'], trainer_source=pins['trainer'])
    monkeypatch.setattr(M, 'FINAL_PINS', final)
    sources = {k: tmp_path / 'final_sources' / k for k in final}
    for dest, source in [('protocol', 'protocol'), ('train_admission', 'admission'), ('trainer_source', 'trainer')]:
        sources[dest] = tmp_path / 'sources' / source
    roots, inventories = {}, {}
    for role in ('A', 'B'):
        root = tmp_path / role; root.mkdir(); roots[role] = root
        snapshots = {}
        hashes = {**pins, 'python': 'a' * 64}
        for key in pins:
            target = root / 'inputs' / (key + '.json'); target.parent.mkdir(exist_ok=True)
            target.write_bytes((tmp_path / 'sources' / key).read_bytes()); snapshots[key] = str(target.relative_to(root))
        jobs, receipts = [], {}
        for plan in M.schedule(role):
            d = root / 'jobs' / plan['id']; d.mkdir(parents=True)
            config = {'schema': M.TRAIN_SCHEMA, 'research_protocol_sha256': pins['protocol'], 'checkpoint_every': 10000,
                      'log_every': 100, 'updates': 100000, 'arm': plan['arm'], 'seed': plan['seed']}
            config_sha = M.canonical(config)
            for step in range(0, 100001, 10000):
                (d / f'checkpoint-{step:09d}.pt').write_bytes(f'synthetic {role} {plan["id"]} step{step}'.encode())
            initial, pointer = [{'path': f'checkpoint-{s:09d}.pt', 'completed_steps': s, 'run_config_sha256': config_sha,
                                 'sha256': M.sha(d / f'checkpoint-{s:09d}.pt')} for s in (0, 100000)]
            status = {'schema': M.TRAIN_SCHEMA, 'state': 'complete', 'error': None, 'completed_steps': 100000,
                      'committed_steps': 100000, 'requested_steps': 100000, 'run_config_sha256': config_sha, 'latest_checkpoint': pointer}
            for name, value in [('protocol.json', config), ('status.json', status), ('history.json', {'synthetic': True}), ('latest.json', pointer)]:
                put(d / name, value)
            jobs.append({**plan, 'status': 'verified_scientific100k_endpoint', 'external': {'exit_code': 0, 'signals': []},
                         'history': {'graph_updates': 100000, 'logged_rows': 1001}, 'config_sha256': config_sha,
                         'runtime': {'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8'},
                         'endpoint': {'completed_steps': 100000, 'finite_model_tensors': 2, 'adam_parameter_states': 2,
                                      'cpu_cuda_rng_serialized': True, 'checkpoint_schema_verified': True},
                         'initial_checkpoint': {'completed_steps': 0, 'finite_model_tensors': 2, 'adam_parameter_states': 0,
                                                'cpu_cuda_rng_serialized': True, 'checkpoint_schema_verified': True},
                         'initial_pointer': initial, 'final_pointer': pointer, 'directory': str(d),
                         'artifact_sha256': {name: M.sha(d / name) for name in ('protocol.json', 'status.json', 'history.json', 'latest.json')}})
        pairs = [{'seed': s, 'initial_model_adam_cpu_cuda_rng_exact': True, 'all100k_frame_noise_native_graph_rng_exact': True,
                  'all_logged_saved_lr_exact': True, 'initial_audit_ordering': 'verified after training; no before-step1 barrier claimed'}
                 for s in sorted({j['seed'] for j in jobs})]
        release = {'schema': 'adaptgns_goop_graph_support_scientific_release_quota_v2', 'status': 'admitted_for_scientific_training',
                   'issued_by': 'root', 'cost_basis': M.COST_BASIS, 'files_sha256': hashes, 'cohort_id': 'synthetic_cohort',
                   'host_role': role, 'host': 'synthetic_' + role, 'schedule': M.schedule('A') + M.schedule('B')}
        put(root / 'inputs/release.json', release)
        launch = {'schema': M.SUPERVISOR_SCHEMA, 'host_role': role, 'cohort_id': 'synthetic_cohort', 'cost_basis': M.COST_BASIS,
                  'hostname': 'synthetic_' + role, 'files_sha256': hashes, 'release_sha256': M.sha(root / 'inputs/release.json'),
                  'input_snapshots': snapshots, 'environment': {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
                  'schedule': M.schedule('A') + M.schedule('B'), 'pid': 999}
        summary = {'schema': M.SUPERVISOR_SCHEMA, 'status': 'verified_host_scientific_endpoints', 'host_role': role,
                   'cohort_id': 'synthetic_cohort', 'cost_basis': M.COST_BASIS, 'all_inputs_reverified': True,
                   'whole_six_model_cohort_admitted': False, 'v1_timing_gate_reinterpreted': False,
                   'all_required_evaluation_outcomes_promised': False, 'jobs': jobs, 'pairing': pairs}
        put(root / 'launch.json', launch); put(root / 'worker_summary.json', summary)
        t = M.now()
        put(root / f'wave_{role}.json', {'schema': M.SUPERVISOR_SCHEMA, 'wave': role, 'state': 'verified', 'pairing': pairs,
                                       'all_children_reaped_utc': t, 'ended_utc': t})
        put(root / 'status.json', {'state': 'complete_host_scientific_training', 'host_role': role,
                                 'whole_six_model_cohort_admitted': False, 'summary_sha256': M.sha(root / 'worker_summary.json'), 'ended_utc': t})
        process = put(tmp_path / ('process_' + role + '.json'), {'schema': 'adaptgns_goop_completed_host_process_check_v1', 'issued_by': 'root',
                      'matching_training_processes': [], 'supervisor_exited': True, 'all_children_reaped': True, 'checked_utc': M.now(),
                      'host_role': role, 'hostname': launch['hostname'], 'supervisor_pid': 999, 'cohort_id': 'synthetic_cohort'})
        inventories[role] = put(tmp_path / ('inventory_' + role + '.json'), M.inventory(root, process))
    release = {'schema': M.RELEASE_SCHEMA, 'status': 'approved_for_cohort_interface_mapping', 'issued_by': 'root',
               'adapter_sha256': M.sha(M.__file__), 'review_rationale': 'Synthetic fixture only', 'final_source_sha256': final,
               'cohort_id': 'synthetic_cohort', 'inventory_sha256': {r: M.sha(p) for r, p in inventories.items()}}
    return roots, inventories, release, sources


def test_complete_maps_interface(cohort_fixture):
    _, inv, release, sources = cohort_fixture
    audit, cohort = M.build(inv, release, sources)
    assert len(cohort['models']) == 6 and len(audit['paired_seeds']) == 3
    assert not audit['fresh_tensor_deserialization_by_adapter']
    assert sum(len([p for p in role['all_retained_files'] if p.endswith('.pt')]) for role in audit['provenance'].values()) == 66


@pytest.mark.parametrize('mutation', ['incomplete', 'wrong_supervisor', 'wrong_protocol', 'missing_pairing', 'duplicate_jobs', 'missing_initial', 'wrong_endpoint', 'changed_input', 'wrong_cadence'])
def test_receipt_failures(cohort_fixture, mutation):
    _, inv, _, _ = cohort_fixture
    data = M.read(inv['A']); r, f = data['receipts'], data['files']
    if mutation == 'incomplete': r['terminal']['state'] = 'running_host_scientific_training'
    if mutation == 'wrong_supervisor': r['launch']['files_sha256']['supervisor'] = 'b' * 64
    if mutation == 'wrong_protocol': r['launch']['files_sha256']['protocol'] = 'b' * 64
    if mutation == 'missing_pairing': r['summary']['pairing'][0]['initial_model_adam_cpu_cuda_rng_exact'] = False
    if mutation == 'duplicate_jobs': r['summary']['jobs'][1] = r['summary']['jobs'][0]
    if mutation == 'missing_initial': del f['jobs/base_seed0/checkpoint-000000000.pt']
    if mutation == 'wrong_endpoint': r['summary']['jobs'][0]['endpoint']['completed_steps'] = 512
    if mutation == 'changed_input': f[r['launch']['input_snapshots']['trainer']]['sha256'] = 'b' * 64
    if mutation == 'wrong_cadence': r['jobs']['base_seed0']['config']['log_every'] = 1
    with pytest.raises(ValueError): M.check_receipts(r, f)


def test_changed_inventory_rejected(cohort_fixture):
    _, inv, release, sources = cohort_fixture
    data = M.read(inv['A']); data['created_utc'] = M.now(); put(inv['A'], data)
    with pytest.raises(ValueError, match='inventory differs'): M.build(inv, release, sources)


def test_failed_artifact_preserved_and_rejected(cohort_fixture, tmp_path):
    roots, _, _, _ = cohort_fixture
    fail = put(roots['A'] / 'jobs/base_seed0/unsuccessful.json', {'failure': 'synthetic'})
    with pytest.raises(ValueError, match='Unsuccessful/partial/locked'): M.inventory(roots['A'], tmp_path / 'process_A.json')
    assert fail.exists()


def test_symlink_rejected(cohort_fixture, tmp_path):
    roots, _, _, _ = cohort_fixture
    (roots['A'] / 'linked').symlink_to(tmp_path / 'process_A.json')
    with pytest.raises(ValueError, match='Symlink'): M.inventory(roots['A'], tmp_path / 'process_A.json')


def test_changed_checkpoint_rejected(cohort_fixture, tmp_path):
    roots, _, _, _ = cohort_fixture
    (roots['A'] / 'jobs/base_seed0/checkpoint-000100000.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint bytes'): M.inventory(roots['A'], tmp_path / 'process_A.json')


def test_frozen_evaluator_contract(cohort_fixture, tmp_path, monkeypatch):
    roots, inv, release, sources = cohort_fixture
    spec = importlib.util.spec_from_file_location('frozen_eval', HERE / 'evaluate_goop_graph_support_final.py')
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    # Frozen check_cohort expects actual helper-adjacent contract and own source bytes.
    sources['diagnostic_source'] = HERE / 'evaluate_goop_graph_support_final.py'
    sources['benchmark_helper'] = HERE / 'benchmark_goop_graph_support_rollout.py'
    sources['data_contract'] = HERE / 'goop_evaluation_contract.py'
    final = {k: M.sha(p) for k, p in sources.items()}
    monkeypatch.setattr(M, 'FINAL_PINS', final); release['final_source_sha256'] = final
    audit, cohort = M.build(inv, release, sources)
    audit_path = put(tmp_path / 'cohort_audit.json', audit); cohort['cohort_audit_sha256'] = M.sha(audit_path)
    cp = roots['A'] / 'jobs/base_seed0/checkpoint-000100000.pt'
    args = SimpleNamespace(protocol=sources['protocol'], train_admission=sources['train_admission'],
        trainer_source=sources['trainer_source'], cohort_audit=audit_path, benchmark_helper=sources['benchmark_helper'],
        benchmark_sha256=M.sha(sources['benchmark_helper']), arm='base', seed=0, checkpoint=cp, checkpoint_sha256=M.sha(cp))
    admission = {'schema': 'adaptgns_goop_training_admission_v1', 'status': 'admitted', 'dataset': 'Goop',
                 'frames_per_trajectory': 401, 'particle_type_ids': [7]}
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    assert evaluator.check_cohort(cohort, args, admission)['arm'] == 'base'
