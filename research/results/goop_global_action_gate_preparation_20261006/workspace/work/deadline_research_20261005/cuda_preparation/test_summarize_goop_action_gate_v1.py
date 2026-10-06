"""Synthetic arrays, opaque bytes and fake process identities only."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location('_collector_test_' + name, HERE / (name + '.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


M = module('summarize_goop_action_gate_v1')
T = module('test_goop_action_gate_scalar_core_v1')
NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.encode(value))
    return M.sha(path)


def absent(pid):
    raise ProcessLookupError(pid)


def process_fixture(started=(0, 1, 2)):
    parent_command = [sys.executable, str(HERE / 'supervise_goop_action_gate_scoped_v1.py'), '--execute']
    parent = {'pid': 111, 'ppid': 1, 'start_ticks': 1234, 'argv': parent_command, 'executable': sys.executable}
    process = {'schema': M.SUPERVISOR_SCHEMA, 'jobs': [], 'owned_registry': [], 'unreaped_owned_children': [], 'abort_reason': None}
    closure = {'schema': M.CLOSURE_SCHEMA, 'issued_by': 'root', 'status': 'all_gate_processes_stopped_and_reaped',
        'hostname': socket.gethostname(), 'queue_release_sha256': 'a' * 64, 'process_outcomes_sha256': 'b' * 64,
        'checked_utc': NOW.isoformat(), 'supervisor': {'identity': parent, 'reaped': True, 'exit_code': 0}, 'children': []}
    for seed in range(3):
        command = [sys.executable, str(HERE / 'run_goop_action_gate_v1.py'), '--execute', '--seed', str(seed)]
        entry = {'seed': seed, 'command': command}
        if seed in started:
            identity = {'pid': 222 + seed, 'ppid': 111, 'start_ticks': 2345 + seed, 'argv': command, 'executable': sys.executable}
            outcome = {'started': True, 'state': 'stopped_and_reaped', 'stopped_and_reaped': True,
                'pid': identity['pid'], 'start_ticks': identity['start_ticks'], 'command': command, 'initial_process_identity': identity, 'exit_code': 0}
            process['owned_registry'].append({'entry': entry, 'pid': identity['pid'], 'identity': identity, 'returncode': 0})
            closure['children'].append({'seed': seed, 'identity': identity, 'reaped': True, 'exit_code': 0})
        else:
            outcome = {'started': False, 'state': 'never_started', 'pid': None}
        process['jobs'].append({'entry': entry, 'outcome': outcome})
    mods = SimpleNamespace(B=SimpleNamespace(process_identity=absent), S=SimpleNamespace(executed_python_script=lambda argv: Path(argv[1]).name))
    return closure, process, mods


def closed(closure, process, mods, **kwargs):
    return M.check_closed(closure, process, 'a' * 64, 'b' * 64, mods, NOW, inventory_reader=lambda: [], **kwargs)


@pytest.mark.parametrize('started', [(), (0,), (0, 1), (0, 1, 2)])
def test_exact_stopped_and_never_started_inventory(started):
    c, p, mods = process_fixture(started)
    assert len(closed(c, p, mods)) == 1 + len(started)


@pytest.mark.parametrize('change', ['live_parent', 'live_child', 'missing_registry', 'unreaped', 'not_reaped', 'wrong_returncode',
    'missing_parent', 'stale', 'future', 'wrong_host', 'wrong_hash', 'wrong_command', 'wrong_start', 'wrong_parent', 'duplicate_registry'])
def test_invalid_closure_rejected_before_arrays(change):
    c, p, mods = process_fixture(); read = absent
    if change == 'live_parent': read = lambda pid: c['supervisor']['identity'] if pid == 111 else None
    elif change == 'live_child': read = lambda pid: c['children'][0]['identity'] if pid == 222 else None
    elif change == 'missing_registry': p['owned_registry'].pop(); c['children'].pop()
    elif change == 'unreaped': p['unreaped_owned_children'] = [222]
    elif change == 'not_reaped': p['jobs'][0]['outcome']['stopped_and_reaped'] = False
    elif change == 'wrong_returncode': p['owned_registry'][0]['returncode'] = 1
    elif change == 'missing_parent': c['supervisor'] = {}
    elif change == 'stale': c['checked_utc'] = (NOW - timedelta(seconds=1801)).isoformat()
    elif change == 'future': c['checked_utc'] = (NOW + timedelta(seconds=6)).isoformat()
    elif change == 'wrong_host': c['hostname'] = 'other-host'
    elif change == 'wrong_hash': c['process_outcomes_sha256'] = '0' * 64
    elif change == 'wrong_command': c['children'][0]['identity'] = {**c['children'][0]['identity'], 'argv': ['foreign']}
    elif change == 'wrong_start': c['children'][0]['identity'] = {**c['children'][0]['identity'], 'start_ticks': 9876}
    elif change == 'wrong_parent': c['children'][0]['identity'] = {**c['children'][0]['identity'], 'ppid': 9876}
    else: p['owned_registry'].append(copy.deepcopy(p['owned_registry'][0])); c['children'].append(copy.deepcopy(c['children'][0]))
    with pytest.raises(ValueError):
        closed(c, p, mods, identity_reader=read)


def test_reused_pid_is_observed_without_signals():
    c, p, mods = process_fixture()
    result = closed(c, p, mods, identity_reader=lambda pid: {'pid': pid, 'start_ticks': 99999, 'argv': ['foreign']})
    assert all(v['current_identity']['start_ticks'] == 99999 for v in result)


def test_launched_unclassified_child_remains_started():
    c, p, mods = process_fixture((0,))
    outcome = p['jobs'][0]['outcome']; outcome['state'] = 'stopped_and_reaped_unclassified'; outcome.pop('start_ticks')
    outcome['initial_process_identity'] = None; p['owned_registry'][0]['identity'] = None
    assert len(closed(c, p, mods)) == 2


def test_foreign_gate_process_still_blocks_collection():
    c, p, mods = process_fixture()
    with pytest.raises(ValueError, match='Another gate'):
        M.check_closed(c, p, 'a' * 64, 'b' * 64, mods, NOW, inventory_reader=lambda: [{'pid': 9876}])


def stage_fixture(tmp_path, monkeypatch, policy='learned_global_gate', failure_step=None, category='synthetic_guard', collection=False):
    root = tmp_path.resolve() / 'queue'; directory = root / 'jobs' / 'mix_seed0'; directory.mkdir(parents=True)
    row, arrays, expected, positions, types, head = T.numeric_fixture(policy, failure_step, category)
    data = tmp_path.resolve() / 'data'; data.mkdir()
    records = []
    for name, values in (('positions', positions), ('types', types)):
        np.save(data / (name + '.npy'), values)
    descriptors = {name: {'path': file, 'shape': list(values.shape), 'dtype': values.dtype.str,
        'size_bytes': (data / file).stat().st_size, 'sha256': M.sha(data / file)} for name, file, values in
        (('positions', 'positions.npy', positions), ('particle_types', 'types.npy', types))}
    records = [{**descriptors, 'trajectory_content_sha256': 'e' * 64} for _ in range(30)]
    protocol = {'schema': T.D.SCHEMA, 'model': {'seed': 0}, 'sentinel': 'synthetic protocol, no actual source admission'}
    protocol_pin = write(directory / 'protocol.json', protocol); row['protocol_sha256'] = protocol_pin
    name = f'source_000000_{policy}'
    with (directory / (name + '.npz')).open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    row.update(artifact_file=name + '.npz', artifact_sha256=M.sha(directory / (name + '.npz')))
    row_pin = write(directory / (name + '.json'), row)
    write(directory / 'status.json', {'current': {'source_index': 0, 'policy': policy}})
    args = SimpleNamespace(manifest=data / 'manifest.json', checkpoint_sha256='a' * 64, head=data / 'head.json', selection=data / 'selection.json')
    context = SimpleNamespace(args=args, manifest={'records': records}, head=head)
    bindings = {str(args.manifest): 'd' * 64, str(args.head): 'b' * 64, str(args.selection): 'c' * 64}
    entry = {'seed': 0, 'directory': str(directory)}
    outcome = {'started': True, 'state': 'stopped_and_reaped', 'stopped_and_reaped': True, 'exit_code': 1}
    mods = SimpleNamespace(D=T.D, C=T.C, P=T.P, V=T.S)
    monkeypatch.setattr(M, 'validate_protocol', lambda *a: None)
    if collection:
        committed = {**row, 'row_file': name + '.json', 'row_sha256': row_pin}
        value = {'schema': T.D.ROLLOUT_COLLECTION, 'mode': 'test-rollout', 'split': 'test', 'driver_protocol_sha256': protocol_pin,
            'model': protocol['model'], 'expected_cells': M.EXPECTED, 'required_rows': 120, 'committed_rows': 1,
            'rows': [committed], 'case_timings': [{'source_index': 0, 'policy': policy, 'wall_seconds': 123.}],
            'runtime': {'setup_seconds': 1}, 'status': 'stopped_incomplete_or_failed', 'complete_rows': 1,
            'all_inputs_reverified': True, 'model_state_verified_unchanged': True, 'all_required_outcomes_complete': False,
            'coverage': T.D.collection_coverage(M.EXPECTED, [committed]), 'abort_reason': {'category': 'synthetic_interruption'},
            'current_at_stop': {'source_index': 0, 'policy': policy}}
        status = {'schema': T.D.SCHEMA, 'state': value['status'], 'current': value['current_at_stop'],
            **{k: value[k] for k in ('required_rows', 'committed_rows', 'complete_rows', 'all_inputs_reverified',
                                    'model_state_verified_unchanged', 'abort_reason')}}
        write(directory / 'status.json', status)
        value.update(M.tree_snapshot(directory))
        write(directory / 'rollout_collection.json', value)
    return root, directory, entry, outcome, context, bindings, mods


def stage(args):
    root, directory, entry, outcome, context, bindings, mods = args
    return M.collect_stage(entry, outcome, context, root, M.tree_snapshot(root), bindings, mods)


@pytest.mark.parametrize('policy', M.POLICIES)
def test_synthetic_driver_row_array_core_composition(tmp_path, monkeypatch, policy):
    args = stage_fixture(tmp_path, monkeypatch, policy)
    result = stage(args)
    assert len(result['cells']) == 120 and len(result['rows']) == 1
    assert result['rows'][0]['validated']['complete'] and result['rows'][0]['case_wall_seconds'] is None
    assert sum(c['state'] == 'uncompleted_after_started_invocation' for c in result['cells']) == 119


@pytest.mark.parametrize('category,state', [('synthetic_guard', 'committed_guard_failed'), ('execution_error', 'committed_execution_failed'), ('execution_budget', 'timed_out_current')])
def test_failed_prefix_categories_and_h200_are_preserved(tmp_path, monkeypatch, category, state):
    args = stage_fixture(tmp_path, monkeypatch, failure_step=201, category=category)
    result = stage(args); row = result['rows'][0]['validated']
    assert row['completed_steps'] == 200 and row['metrics']['mse_forecast200'] is not None
    assert row['metrics']['mean_rollout_mse'] is None
    assert next(c for c in result['cells'] if c['policy'] == 'learned_global_gate' and c['source_index'] == 0)['state'] == state


@pytest.mark.parametrize('change', ['artifact_hash', 'invalid_npz', 'object_npz', 'json', 'missing_artifact', 'wrong_truth', 'scalar_types', 'missing_protocol'])
def test_invalid_rows_are_retained_without_metrics(tmp_path, monkeypatch, change):
    args = stage_fixture(tmp_path, monkeypatch); root, directory, entry, outcome, context, bindings, mods = args
    row_path = directory / 'source_000000_learned_global_gate.json'; artifact = row_path.with_suffix('.npz')
    if change == 'artifact_hash': artifact.write_bytes(b'changed opaque bytes')
    elif change in ('invalid_npz', 'object_npz'):
        if change == 'invalid_npz': artifact.write_bytes(b'not a numeric archive')
        else: np.savez(artifact, malicious=np.array([{'opaque': True}], dtype=object))
        row = json.loads(row_path.read_bytes()); row['artifact_sha256'] = M.sha(artifact); write(row_path, row)
    elif change == 'json': row_path.write_bytes(b'not json')
    elif change == 'missing_artifact': artifact.unlink()
    elif change == 'wrong_truth':
        path = context.args.manifest.parent / 'positions.npy'; values = np.load(path); values[10, 0, 0] += .1; np.save(path, values)
        context.manifest['records'][0]['positions']['sha256'] = M.sha(path)
    elif change == 'scalar_types':
        path = context.args.manifest.parent / 'types.npy'; np.save(path, np.asarray(7, dtype=np.int64)); descriptor = context.manifest['records'][0]['particle_types']
        descriptor.update(shape=[], sha256=M.sha(path), size_bytes=path.stat().st_size)
    else: (directory / 'protocol.json').unlink()
    result = stage(args)
    assert result['rows'] == [] and len(result['evidence']['invalid_records']) == 1
    assert all(c['state'] == 'uncompleted_after_started_invocation' for c in result['cells'])


def test_actual_case_wall_receipt_separate_from_component_timers(tmp_path, monkeypatch):
    result = stage(stage_fixture(tmp_path, monkeypatch, policy='base', collection=True))
    assert result['rows'][0]['case_wall_seconds'] == 123.
    assert result['runtime'] == {'setup_seconds': 1} and result['evidence']['collection_valid']
    assert result['scientific_verification_passed']


@pytest.mark.parametrize('change', ['timing_identity', 'timing_missing', 'timing_nan', 'row_sha', 'tree_partial'])
def test_invalid_collection_cannot_supply_case_timing(tmp_path, monkeypatch, change):
    args = stage_fixture(tmp_path, monkeypatch, policy='base', collection=True); directory = args[1]
    path = directory / 'rollout_collection.json'; value = json.loads(path.read_bytes())
    if change == 'timing_identity': value['case_timings'][0]['source_index'] = 1
    elif change == 'timing_missing': value['case_timings'] = []
    elif change == 'timing_nan': value['case_timings'][0]['wall_seconds'] = -1
    elif change == 'row_sha': value['rows'][0]['row_sha256'] = '0' * 64
    else: (directory / 'opaque_partial.npz.tmp').write_bytes(b'opaque unsuccessful output retained')
    write(path, value); result = stage(args)
    assert result['rows'][0]['case_wall_seconds'] is None
    assert not result['evidence']['collection_valid'] and result['runtime'] is None


def test_partial_files_and_missing_timed_out_current_preserved(tmp_path, monkeypatch):
    args = stage_fixture(tmp_path, monkeypatch); directory = args[1]; outcome = args[3]
    (directory / 'source_000001_base.npz.tmp').write_bytes(b'opaque interrupted write')
    outcome.update(inner_timeout_reported=True, current_before_stop={'source_index': 1, 'policy': 'base'})
    result = stage(args)
    assert next(c for c in result['cells'] if (c['source_index'], c['policy']) == (1, 'base'))['state'] == 'timed_out_current'
    assert (directory / 'source_000001_base.npz.tmp').read_bytes() == b'opaque interrupted write'


def test_never_started_must_have_no_stream_directory(tmp_path):
    root = tmp_path.resolve() / 'queue'; root.mkdir(); directory = root / 'jobs' / 'mix_seed2'
    entry = {'seed': 2, 'directory': str(directory)}; outcome = {'started': False}
    result = M.collect_stage(entry, outcome, None, root, M.tree_snapshot(root), {}, None)
    assert len(result['cells']) == 120 and all(c['state'] == 'never_started' for c in result['cells'])
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match='Never-started'):
        M.collect_stage(entry, outcome, None, root, M.tree_snapshot(root), {}, None)


def test_symlink_queue_entry_rejected(tmp_path):
    root = tmp_path.resolve() / 'queue'; root.mkdir(); target = tmp_path / 'target'; target.write_bytes(b'opaque')
    (root / 'alias').symlink_to(target)
    with pytest.raises(ValueError, match='Symlink'):
        M.tree_snapshot(root)


@pytest.mark.parametrize('path', ['../escape.npy', '/absolute.npy', 'array.pkl'])
def test_source_path_cannot_escape_or_pickle(tmp_path, path):
    with pytest.raises(ValueError): M.array_path(tmp_path.resolve(), {'path': path})


def test_description_is_inert(capsys, monkeypatch):
    monkeypatch.setattr(M, 'modules', lambda: pytest.fail('Description imported execution modules'))
    assert M.main([]) == 0
    assert json.loads(capsys.readouterr().out)['new_research_execution'] is False


def collector_fixture(tmp_path, monkeypatch, ledger=False):
    """Whole publication scaffold with frozen semantic gates replaced by fakes.

    Individual tests above compose actual synthetic numeric rows/core. This
    fixture checks process-before-array ordering, final bindings and publication
    races without touching model/data or purporting to exercise original gates.
    """
    base = tmp_path.resolve(); root = base / 'queue'; root.mkdir(); output = base / 'summary.json'
    supervisor = HERE / 'supervise_goop_action_gate_scoped_v1.py'
    source_pins = {supervisor.name: M.sha(supervisor)}
    monkeypatch.setattr(M, 'SOURCE_PINS', source_pins)
    closure, process, mods = process_fixture(())
    parent_release = base / 'original_phase.json'
    closure['supervisor']['identity']['argv'] += ['--release', str(parent_release), '--output-dir', str(root)]
    environment = {'lexical_path': sys.executable}
    mods.S.python_environment = lambda path: environment
    mods.S.ENVIRONMENT = {}
    mods.B.canonical_gpu_uuid = lambda value: value
    mods.D = SimpleNamespace(SOURCE_PINS={})
    mods.V = T.S; mods.P = T.P
    monkeypatch.setattr(M, 'runtime_process_inventory', lambda m: [])
    for seed, job in enumerate(process['jobs']):
        job['entry'].update(stream=f'mix_seed{seed}', stage='test-rollout', arm='mix', gpu=1 if seed == 1 else 0,
            gpu_uuid=f'uuid{1 if seed == 1 else 0}', directory=str(root / 'jobs' / f'mix_seed{seed}'), expected_cells=M.EXPECTED,
            outer_timeout_seconds=100)
    phase = {'schema': 'adaptgns_goop_action_gate_scoped_release_v1', 'issued_by': 'root', 'status': 'approved_for_one_fixed_gate_phase',
        'mode': 'test-rollout', 'hostname': socket.gethostname(), 'host_role': 'B', 'output_dir': str(root),
        'all_six_original_goop_models_frozen': True, 'no_retry_or_resume': True, 'does_not_displace_original_studies': True,
        'python_environment': environment, 'files_sha256': {str(supervisor): M.sha(supervisor)}, 'gpu_scope': {
            'owned_indices': [0, 1], 'unassigned_devices': 'observe_without_control', 'live_handoff': False},
        'gpu_uuids': [f'uuid{i}' for i in range(4)], 'environment': {},
        'jobs': [{'seed': seed, 'gpu': 1 if seed == 1 else 0, 'command': process['jobs'][seed]['entry']['command']} for seed in range(3)]}
    phase_sha = write(parent_release, phase); write(root / 'release_snapshot.json', phase)
    process_sha = write(root / 'process_outcomes.json', process)
    closure.update(queue_release_sha256=phase_sha, process_outcomes_sha256=process_sha)
    closure_path = base / 'closure.json'; closure_sha = write(closure_path, closure)
    manifest = base / 'manifest.json'; selection = base / 'selection.json'
    input_pins = {str(parent_release): phase_sha, str(closure_path): closure_sha, str(supervisor): M.sha(supervisor),
                  str(manifest): write(manifest, {'synthetic': 'data unavailable in this fixture'}),
                  str(selection): write(selection, {'synthetic': 'frozen selection placeholder'})}
    ledger_pin = None
    if ledger:
        ledger_pin = write(root / 'phase_ledger.json', {'schema': M.SUPERVISOR_SCHEMA, 'mode': 'test-rollout',
            'state': 'stopped_requires_review', 'jobs': process['jobs'], 'abort_reason': None,
            'all_pinned_inputs_reverified': True, 'numerical_arrays_audited_by_supervisor': False,
            'input_sha256': {str(parent_release): phase_sha},
            'supervisor_files': M.tree_snapshot(root)['files'], 'elapsed_seconds': 1.5})
    release = {'schema': M.RELEASE_SCHEMA, 'issued_by': 'root', 'status': 'approved_for_stopped_numeric_collection',
        'hostname': socket.gethostname(), 'queue_root': str(root), 'output': str(output), 'collector_sha256': M.sha(M.__file__),
        'source_sha256': source_pins, 'files_sha256': input_pins, 'queue_release_sha256': phase_sha,
        'process_outcomes_sha256': process_sha, 'phase_ledger_sha256': ledger_pin,
        'process_closure': {'path': str(closure_path), 'sha256': closure_sha},
        'queue_inventory_sha256': M.inventory_sha256(M.tree_snapshot(root))}
    release_path = base / 'release.json'; write(release_path, release)
    monkeypatch.setattr(M, 'driver_context', lambda *args: SimpleNamespace(cohort_sha='c' * 64,
        args=SimpleNamespace(manifest=manifest, selection=selection), selection={'synthetic': True}))
    args = SimpleNamespace(queue=root, release=release_path, output=output)
    return args, mods, release, closure, process


@pytest.mark.parametrize('ledger', [False, True])
def test_whole_collection_missing_ledger_and_empty_queue_retains_all360(tmp_path, monkeypatch, ledger):
    args, mods, *_ = collector_fixture(tmp_path, monkeypatch, ledger)
    value = M.collect(args, mods, NOW)
    assert args.output.is_file() and json.loads(args.output.read_bytes()) == value
    assert value['summary']['coverage'] == {'never_started': 360}
    assert value['phase_ledger_present'] is ledger
    assert value['summary']['primary_full_H395_contrasts']['learned_global_gate_minus_base']['mean'] is None


def test_closure_failure_precedes_any_queue_array_hashing(tmp_path, monkeypatch):
    args, mods, release, closure, process = collector_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(M, 'tree_snapshot', lambda root: pytest.fail('Queue array tree was accessed before process closure'))
    mods.B.process_identity = lambda pid: closure['supervisor']['identity']
    with pytest.raises(ValueError, match='still present'):
        M.collect(args, mods, NOW)
    assert not args.output.exists()


@pytest.mark.parametrize('change', ['input', 'source', 'queue_file', 'queue_directory', 'scalar_result', 'serialized_result', 'closure'])
def test_mutation_during_serialization_never_publishes(tmp_path, monkeypatch, change):
    args, mods, release, closure, process = collector_fixture(tmp_path, monkeypatch)
    original = M.encode
    def serialize(value):
        if value.get('schema') == M.SCHEMA:
            if change == 'input': (args.queue.parent / 'manifest.json').write_bytes(b'changed')
            elif change == 'source':
                # Mutate the recorded own-source binding, never actual source.
                value['input_sha256'][str(Path(M.__file__).resolve())] = '0' * 64
            elif change == 'queue_file': (args.queue / 'opaque_added_partial').write_bytes(b'partial')
            elif change == 'queue_directory': (args.queue / 'new_directory').mkdir()
            elif change == 'scalar_result': value['summary']['coverage'] = {'never_started': 0}
            elif change == 'serialized_result':
                replacement = copy.deepcopy(value); replacement['status'] = 'forged'; return original(replacement)
            elif change == 'closure': (args.queue.parent / 'closure.json').write_bytes(b'changed')
        return original(value)
    monkeypatch.setattr(M, 'encode', serialize)
    with pytest.raises(ValueError): M.collect(args, mods, NOW)
    assert not args.output.exists()


def test_restarted_gate_at_final_process_check_prevents_publication(tmp_path, monkeypatch):
    args, mods, *_ = collector_fixture(tmp_path, monkeypatch)
    calls = []
    def inventory(m):
        calls.append(1)
        return [] if len(calls) == 1 else [{'pid': 99999, 'argv': ['gate']}]
    monkeypatch.setattr(M, 'runtime_process_inventory', inventory)
    with pytest.raises(ValueError, match='Another gate'): M.collect(args, mods, NOW)
    assert not args.output.exists()


def protocol_fixture(tmp_path):
    root = tmp_path.resolve(); manifest_path = root / 'manifest.json'; release_path = root / 'release.json'
    head_path = root / 'head.json'; selection_path = root / 'selection.json'; metadata_path = root / 'metadata.json'
    position = root / 'positions.npy'; types = root / 'types.npy'; auxiliary = root / 'auxiliary.npy'
    for p in (manifest_path, release_path, head_path, selection_path, metadata_path, position, types, auxiliary):
        p.write_bytes(b'opaque synthetic pin only; not a real data file')
    bindings = {str(p): M.sha(p) for p in root.iterdir()}
    args = SimpleNamespace(manifest=manifest_path, release=release_path, head=head_path, selection=selection_path,
                          seed=0, checkpoint_sha256='a' * 64)
    context = SimpleNamespace(args=args, cohort_sha='b' * 64, release={'files_sha256': bindings}, execution_inputs=set(), model={'config_sha256': 'c' * 64},
        manifest={'records': [{'positions': {'path': position.name}, 'particle_types': {'path': types.name}, 'step_context': {'path': auxiliary.name}}]})
    protocol = {'schema': T.D.SCHEMA, 'mode': 'test-rollout', 'split': 'test', 'gate_protocol_sha256': T.D.GATE_PROTOCOL_SHA,
        'original_training_protocol_sha256': T.D.TRAINING_PROTOCOL_SHA, 'core_sha256': T.D.CORE_SHA,
        'driver_sha256': M.SOURCE_PINS['run_goop_action_gate_v1.py'], 'cohort_sha256': context.cohort_sha,
        'source_manifest_sha256': bindings[str(manifest_path)], 'head_sha256': bindings[str(head_path)],
        'selection_sha256': bindings[str(selection_path)], 'schedule': M.EXPECTED, 'features': list(T.C.FEATURES),
        'policies': list(M.POLICIES), 'selected_model_parameters_trainable': False, 'input_sha256': copy.deepcopy(bindings),
        'model': {'kind': 'preselected_checkpoint', 'seed': 0, 'arm': 'mix', 'objective': 'faithful', 'completed_updates': 100000,
                  'checkpoint_sha256': 'a' * 64, 'run_config_sha256': 'c' * 64, 'training_source_sha256': T.D.SOURCE_PINS['train_goop_graph_support_cuda.py']}}
    return protocol, context, bindings, SimpleNamespace(D=T.D, C=T.C)


def test_original_protocol_closure_validates_without_model_load(tmp_path):
    value, context, pins, mods = protocol_fixture(tmp_path)
    assert M.validate_protocol(value, 'f' * 64, context, pins, mods) == value['model']


@pytest.mark.parametrize('change', ['mode', 'split', 'cohort', 'head', 'selection', 'schedule', 'features', 'policies',
    'trainable', 'modelseed', 'checkpoint', 'config', 'otherconfig', 'source', 'missinginput', 'foreigninput'])
def test_original_protocol_rejects_wrong_scientific_or_input_lineage(tmp_path, change):
    value, context, pins, mods = protocol_fixture(tmp_path)
    if change == 'mode': value['mode'] = 'train-rollout-capacity'
    elif change == 'split': value['split'] = 'valid'
    elif change in ('cohort', 'head', 'selection'): value[change + '_sha256'] = '0' * 64
    elif change in ('schedule', 'features', 'policies'): value[change] = value[change][:-1]
    elif change == 'trainable': value['selected_model_parameters_trainable'] = True
    elif change == 'modelseed': value['model']['seed'] = False
    elif change == 'checkpoint': value['model']['checkpoint_sha256'] = '0' * 64
    elif change == 'config': value['model']['run_config_sha256'] = None
    elif change == 'otherconfig': value['model']['run_config_sha256'] = 'd' * 64
    elif change == 'source': value['model']['training_source_sha256'] = '0' * 64
    elif change == 'missinginput': value['input_sha256'].pop(str(context.args.manifest))
    else: value['input_sha256']['/foreign/source'] = '0' * 64
    with pytest.raises(ValueError): M.validate_protocol(value, 'f' * 64, context, pins, mods)


def update_collection(directory, mutate):
    path = directory / 'rollout_collection.json'; value = json.loads(path.read_bytes()); mutate(value)
    status = {'schema': T.D.SCHEMA, 'state': value['status'], 'current': value['current_at_stop'],
        **{k: value[k] for k in ('required_rows', 'committed_rows', 'complete_rows', 'all_inputs_reverified',
                                'model_state_verified_unchanged', 'abort_reason')}}
    write(directory / 'status.json', status)
    tree = M.tree_snapshot(directory)
    tree['files'].pop('rollout_collection.json')
    tree['output_tree_entries'] = [r for r in tree['output_tree_entries'] if r['path'] != 'rollout_collection.json']
    value.update(tree); write(path, value)


@pytest.mark.parametrize('field', ['all_inputs_reverified', 'model_state_verified_unchanged'])
def test_unverified_original_model_or_inputs_keep_diagnostics_only(tmp_path, monkeypatch, field):
    args = stage_fixture(tmp_path, monkeypatch, policy='base', collection=True)
    update_collection(args[1], lambda value: value.update({field: False}))
    result = stage(args)
    assert result['rows'][0]['validated']['complete'] and result['rows'][0]['case_wall_seconds'] == 123.
    assert result['evidence']['collection_valid'] and not result['scientific_verification_passed']
    assert result['evidence']['scientific_verification']['reasons']


def test_original_supervisor_collection_pin_cannot_be_replaced_by_new_inventory(tmp_path, monkeypatch):
    args = stage_fixture(tmp_path, monkeypatch, policy='base', collection=True)
    args[3]['complete_collection'] = {'path': str(args[1] / 'rollout_collection.json'), 'sha256': '0' * 64,
        'status': 'complete', 'required_rows': 120, 'committed_rows': 120}
    result = stage(args)
    assert result['rows'][0]['validated']['complete']
    assert result['rows'][0]['case_wall_seconds'] is None and not result['scientific_verification_passed']
    assert 'Original supervisor' in result['evidence']['invalid_collection']


def test_final_status_must_match_original_collection_flags(tmp_path, monkeypatch):
    args = stage_fixture(tmp_path, monkeypatch, policy='base', collection=True)
    path = args[1] / 'status.json'; value = json.loads(path.read_bytes()); value['model_state_verified_unchanged'] = False; write(path, value)
    # Refresh only the collection inventory to isolate semantic status mismatch.
    path = args[1] / 'rollout_collection.json'; value = json.loads(path.read_bytes()); value['files']['status.json']['sha256'] = M.sha(args[1] / 'status.json')
    value['files']['status.json']['bytes'] = (args[1] / 'status.json').stat().st_size; write(path, value)
    result = stage(args)
    assert not result['scientific_verification_passed'] and 'terminal status' in result['evidence']['invalid_collection']


def complete_scalar_grid_fixture(tmp_path, monkeypatch):
    """All120 scalar commitments; numeric validation mocked only in this fixture.

    Actual numeric driver/core composition is separately exercised for each
    policy and failure prefix; these cases target the complete-stage gate.
    """
    args = stage_fixture(tmp_path, monkeypatch, policy='base', collection=True)
    root, directory, entry, outcome, context, bindings, mods = args
    numeric_bytes = (directory / 'source_000000_base.npz').read_bytes()
    rows = []
    for item in M.EXPECTED:
        name = f'source_{item["source_index"]:06d}_{item["policy"]}'
        (directory / (name + '.npz')).write_bytes(numeric_bytes)
        row = {**item, 'status': 'complete', 'artifact_file': name + '.npz', 'artifact_sha256': M.sha(directory / (name + '.npz'))}
        pin = write(directory / (name + '.json'), row)
        rows.append({**row, 'row_file': name + '.json', 'row_sha256': pin})
    def change(value):
        value.update(rows=rows, committed_rows=120, complete_rows=120, status='complete', all_required_outcomes_complete=True,
            abort_reason=None, current_at_stop=None, coverage=T.D.collection_coverage(M.EXPECTED, rows),
            case_timings=[{**item, 'wall_seconds': 123.} for item in M.EXPECTED])
    update_collection(directory, change)
    outcome['exit_code'] = 0
    outcome['complete_collection'] = {'path': str(directory / 'rollout_collection.json'),
        'sha256': M.sha(directory / 'rollout_collection.json'), 'status': 'complete', 'required_rows': 120, 'committed_rows': 120}
    mods.V = SimpleNamespace(finite=T.S.finite, OPERATIONAL=T.S.OPERATIONAL,
        validate_row=lambda *args: {'complete': True, 'failure': None})
    return args


def test_complete120_stage_requires_and_accepts_original_commitment(tmp_path, monkeypatch):
    result = stage(complete_scalar_grid_fixture(tmp_path, monkeypatch))
    assert len(result['rows']) == 120 and result['scientific_verification_passed']
    assert result['evidence']['collection_valid'] and result['evidence']['scientific_verification']['reasons'] == []


def test_complete120_missing_original_receipt_never_gets_scientific_admission(tmp_path, monkeypatch):
    args = complete_scalar_grid_fixture(tmp_path, monkeypatch); args[3].pop('complete_collection')
    result = stage(args)
    assert len(result['rows']) == 120 and not result['scientific_verification_passed']
    assert 'original supervisor proof' in result['evidence']['invalid_collection']
    assert all(row['case_wall_seconds'] is None for row in result['rows'])


def test_complete120_numeric_rows_without_model_reverification_remain_diagnostics(tmp_path, monkeypatch):
    args = complete_scalar_grid_fixture(tmp_path, monkeypatch)
    def change(value):
        value.update(model_state_verified_unchanged=False, status='stopped_incomplete_or_failed', all_required_outcomes_complete=False,
                     abort_reason={'category': 'execution_error', 'error': 'synthetic final model audit failure'})
    update_collection(args[1], change); args[3].pop('complete_collection'); args[3]['exit_code'] = 1
    result = stage(args)
    assert len(result['rows']) == 120 and result['evidence']['collection_valid'] and not result['scientific_verification_passed']
    assert result['evidence']['scientific_verification']['reasons'] == ['original_model_state_reverification_not_passed']
