#!/usr/bin/env python3
"""Audit stopped Goop gate arrays on their original host; emit scalar evidence.

Description only by default. No simulator, checkpoint deserialization, launch,
signals, fitting, acquisition, or inference. An explicit root release and fresh
independent process-closure receipt precede every queue/data array read.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop_action_gate_stopped_numeric_collection_v1'
RELEASE_SCHEMA = 'adaptgns_goop_action_gate_stopped_collection_release_v1'
CLOSURE_SCHEMA = 'adaptgns_goop_action_gate_process_closure_v1'
SUPERVISOR_SCHEMA = 'adaptgns_goop_action_gate_scoped_supervisor_v1'
SOURCE_PINS = {
    'run_goop_action_gate_v1.py': 'af51bbe073e52a22d4394918dbce4bc9c34a99645dc7f302b0cd5c48ebba5079',
    'supervise_goop_action_gate_scoped_v1.py': '4cda31a408c097944e5f15b3502d05bcfab0d3db5b8ee25dcc5fdc1968fd61df',
    'goop_global_action_gate_core_v1.py': 'd3986c2ac3786e9fc0d36d76339bfeded5448497f7dad9b0559efaaaa53a43ca',
    'goop_action_gate_scalar_core_v1.py': 'a0c0098e72c073b3468a2cc9a0d7a23c723250148f1dc1a550f35e02425081b1',
    'summarize_goop_graph_support_quota_v2.py': 'da85058ea2ce0fc0f1b67e6cad442369835dfe1e8926d8f26c595d143f1ec33c',
    'measure_sand_cuda_capacity_v2.py': 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd',
    'supervise_sand_scoped_science_v1.py': '2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13',
}
POLICIES = ('base', 'random25', 'learned_global_gate', 'validation_rate_random_gate')
EXPECTED = [{'source_index': i, 'policy': p} for i in range(30) for p in POLICIES]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ordinary(path, directory=False, lexical_binary=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute unambiguous path required')
    require(path.is_dir() if directory else path.is_file(), 'Existing ordinary input required: ' + str(path))
    # The interpreter is the one deliberate symlink exception. Its exact
    # lexical path and resolved binary are independently pinned by the parent.
    require(str(path) == lexical_binary or path.resolve() == path, 'Symlink/alias input forbidden: ' + str(path))
    return path


def snapshot(path, pin=None):
    path = ordinary(path)
    raw = path.read_bytes(); actual = hashlib.sha256(raw).hexdigest()
    require(pin is None or actual == pin, 'JSON bytes differ: ' + str(path))
    return json.loads(raw), actual


def bind(target, incoming):
    require(isinstance(incoming, dict), 'Input hash map required')
    for path, pin in incoming.items():
        require(isinstance(path, str) and Path(path).is_absolute() and '..' not in Path(path).parts
                and digest(pin) and (path not in target or target[path] == pin), 'Invalid/conflicting input binding')
        target[path] = pin


def verify(bindings, lexical_binary=None):
    for path, pin in bindings.items():
        ordinary(path, lexical_binary=lexical_binary)
        require(sha(path) == pin, 'Original input bytes changed: ' + path)


def tree_snapshot(root):
    root = ordinary(root, directory=True); entries = []; files = {}
    for path in sorted(root.rglob('*')):
        ordinary(path, directory=path.is_dir())
        name = str(path.relative_to(root)); kind = 'directory' if path.is_dir() else 'file'
        entries.append({'path': name, 'kind': kind})
        if kind == 'file':
            files[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return {'files': files, 'output_tree_entries': entries}


def inventory_sha256(value):
    return hashlib.sha256(encode(value)).hexdigest()


def load(name):
    path = ordinary(HERE / name)
    require(sha(path) == SOURCE_PINS[name], 'Reviewed source differs: ' + name)
    spec = importlib.util.spec_from_file_location('_gate_collect_' + name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def modules():
    return SimpleNamespace(D=load('run_goop_action_gate_v1.py'),
        C=load('goop_global_action_gate_core_v1.py'), V=load('goop_action_gate_scalar_core_v1.py'),
        P=load('summarize_goop_graph_support_quota_v2.py'),
        B=load('measure_sand_cuda_capacity_v2.py'), S=load('supervise_sand_scoped_science_v1.py'))


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'Timezone-aware timestamp required')
    return result.astimezone(timezone.utc)


def key(item):
    require(type(item.get('source_index')) is int and 0 <= item['source_index'] < 30
            and item.get('policy') in POLICIES, 'Unknown planned test cell')
    return item['source_index'], item['policy']


def identity(value, command=None):
    require(isinstance(value, dict) and type(value.get('pid')) is int and value['pid'] > 0
            and type(value.get('start_ticks')) is int and value['start_ticks'] > 0
            and type(value.get('ppid')) is int and isinstance(value.get('argv'), list)
            and (command is None or value['argv'] == command), 'Complete independently captured process identity required')
    return value


def runtime_process_inventory(mods):
    require(sys.platform.startswith('linux') and Path('/proc').is_dir(), 'Collect on the original Linux host')
    rows = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            value = mods.B.process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if mods.S.executed_python_script(value['argv']) in ('run_goop_action_gate_v1.py', 'supervise_goop_action_gate_scoped_v1.py'):
            rows.append(value)
    return rows


def check_closed(closure, process, queue_release_sha, process_sha, mods, at=None,
                 identity_reader=None, inventory_reader=None, require_fresh=True):
    at = datetime.now(timezone.utc) if at is None else at
    read = mods.B.process_identity if identity_reader is None else identity_reader
    inventory = (lambda: runtime_process_inventory(mods)) if inventory_reader is None else inventory_reader
    require(closure.get('schema') == CLOSURE_SCHEMA and closure.get('issued_by') == 'root'
            and closure.get('status') == 'all_gate_processes_stopped_and_reaped'
            and closure.get('hostname') == socket.gethostname()
            and closure.get('queue_release_sha256') == queue_release_sha
            and closure.get('process_outcomes_sha256') == process_sha,
            'Independent root process closure must bind this exact stopped queue')
    if require_fresh:
        require(-5 <= (at - stamp(closure['checked_utc'])).total_seconds() <= 1800, 'Fresh root closure check required')
    require(process.get('schema') == SUPERVISOR_SCHEMA and process.get('unreaped_owned_children') == [], 'Unreaped owned children prohibit collection')
    registry = process.get('owned_registry'); jobs = process.get('jobs'); proofs = closure.get('children')
    require(isinstance(registry, list) and isinstance(jobs, list) and len(jobs) == 3
            and [j.get('entry', {}).get('seed') for j in jobs] == [0, 1, 2]
            and isinstance(proofs, list) and len(proofs) == len(registry), 'Complete three-stream/owned launch closure required')
    parent = closure.get('supervisor', {}); parent_identity = identity(parent.get('identity'))
    require(parent.get('reaped') is True and type(parent.get('exit_code')) is int
            and mods.S.executed_python_script(parent_identity['argv']) == 'supervise_goop_action_gate_scoped_v1.py',
            'Root supervisor reaping evidence required even without a final ledger')
    identities = [parent_identity]; seen = set()
    for owned, proof in zip(registry, proofs):
        entry = owned.get('entry', {}); seed = entry.get('seed')
        require(type(seed) is int and seed in (0, 1, 2) and seed not in seen, 'Each planned child may launch only once')
        seen.add(seed); job = jobs[seed]; outcome = job.get('outcome', {})
        require(entry == job['entry'] and outcome.get('started') is True
                and outcome.get('stopped_and_reaped') is True
                and outcome.get('state') in ('stopped_and_reaped', 'stopped_and_reaped_unclassified')
                and type(owned.get('returncode')) is int and type(outcome.get('exit_code')) is int and outcome.get('exit_code') == owned['returncode']
                and outcome.get('pid') == owned.get('pid') and outcome.get('command') == entry.get('command'),
                'Owned registry and reaped outcome differ')
        ident = identity(proof.get('identity'), entry['command'])
        require(type(proof.get('seed')) is int and proof.get('seed') == seed and proof.get('reaped') is True
                and type(proof.get('exit_code')) is int and proof.get('exit_code') == owned['returncode'] and ident['pid'] == owned['pid']
                and ident['ppid'] == parent_identity['pid'], 'Independent child closure differs from recorded invocation')
        for captured in (owned.get('identity'), outcome.get('initial_process_identity')):
            if captured is not None:
                require(all(captured.get(k) == ident.get(k) for k in ('pid', 'ppid', 'start_ticks', 'argv', 'executable')),
                        'Root identity differs from original launch capture')
        if outcome.get('start_ticks') is not None:
            require(outcome['start_ticks'] == ident['start_ticks'], 'Outcome start identity differs')
        identities.append(ident)
    for job in jobs:
        outcome = job.get('outcome', {}); seed = job['entry']['seed']
        if seed not in seen:
            require(outcome.get('started') is False and outcome.get('state') == 'never_started'
                    and outcome.get('pid') is None, 'Missing registry cannot become an unproved stopped child')
    require(len({(v['pid'], v['start_ticks']) for v in identities}) == len(identities), 'Duplicate process identity')
    observed = []
    for ident in identities:
        try:
            current = read(ident['pid'])
        except (FileNotFoundError, ProcessLookupError):
            current = None
        require(current is None or current.get('start_ticks') != ident['start_ticks'], 'An owned process identity is still present')
        observed.append({'captured_identity': ident, 'current_identity': current})
    require(inventory() == [], 'Another gate driver or supervisor is live; collection is not stable')
    return observed


def array_path(root, descriptor):
    relative = Path(descriptor['path'])
    require(not relative.is_absolute() and '..' not in relative.parts and relative.suffix == '.npy', 'Safe numeric source path required')
    return ordinary(Path(root) / relative)


def byte_array(path, descriptor):
    raw = ordinary(path).read_bytes()
    require(len(raw) == descriptor['size_bytes'] and hashlib.sha256(raw).hexdigest() == descriptor['sha256'], 'Source array bytes differ')
    value = np.load(io.BytesIO(raw), allow_pickle=False)
    require(isinstance(value, np.ndarray) and list(value.shape) == descriptor['shape'] and value.dtype.str == descriptor['dtype'], 'Source numeric descriptor differs')
    return value


def driver_context(entry, phase, bindings, mods):
    command = entry['command']; D = mods.D
    require(command[:3] == [sys.executable, str(HERE / 'run_goop_action_gate_v1.py'), '--execute']
            and len(command[3:]) % 2 == 0 and len(set(command[3::2])) == len(command[3::2]), 'Exact original lexical driver command required')
    args = D.parse_args(command[2:]); release, release_sha = snapshot(args.release, bindings.get(str(args.release)))
    # Revalidate the immutable original admission at its recorded admission
    # timestamp. This performs no fresh execution admission and never invokes
    # release_gate, whose nonexistent-output requirement is intentionally retained.
    pins = D.validate_release(args, release, stamp(phase['process_clock_checked_utc']))
    require(args.mode == 'test-rollout' and args.seed == entry['seed'] and args.cuda_index == entry['gpu']
            and str(args.output_dir) == entry['directory'] and args.max_seconds == entry['outer_timeout_seconds']
            and release.get('complete_capacity_report') == phase.get('complete_capacity_report')
            and release.get('absolute_stop_utc') == phase.get('absolute_stop_utc'), 'Original driver/parent lineage differs')
    for path, pin in {**pins, str(args.release): release_sha}.items():
        require(bindings.get(path) == pin, 'Root collection must bind every original driver input')
    args.split = 'test'; args.arm = 'mix'; args.objective = 'faithful'; args.benchmark_sha256 = D.SOURCE_PINS['benchmark_goop_graph_support_rollout.py']
    numerical = D.modules()
    cohort, cohort_sha = snapshot(args.cohort, bindings[str(args.cohort)])
    train, _ = snapshot(args.train_admission, bindings[str(args.train_admission)])
    numerical.final.check_cohort(cohort, args, train)
    manifest, admission, structural = [snapshot(getattr(args, n), bindings[str(getattr(args, n))])[0]
                                       for n in ('manifest', 'admission', 'structural_report')]
    numerical.final.check_split(numerical.bench, manifest, admission, structural, args, cohort_sha)
    metadata = args.manifest.parent / 'metadata.json'
    require(bindings.get(str(metadata)) == numerical.final.METADATA_SHA
            and snapshot(metadata, numerical.final.METADATA_SHA)[0] == manifest['metadata'], 'Original metadata bytes differ')
    for path, pin in numerical.bench.SOURCE_PINS.items():
        require(bindings.get(str(args.repo / path)) == pin, 'Original simulator source closure required without importing simulator')
    helpers = SimpleNamespace(np=np, data_loader=SimpleNamespace(_manifest_array_path=array_path))
    evidence = numerical.bench.load_split_evidence(args, manifest, admission, helpers)
    require(all(bindings.get(p) == h for p, h in evidence.items()), 'Root must bind complete preserved source/context evidence')
    for record in manifest['records']:
        for field in ('positions', 'particle_types', 'step_context'):
            path = array_path(args.manifest.parent, record[field])
            require(bindings.get(str(path)) == record[field]['sha256'], 'All original source arrays must be root-bound')
    head, _ = snapshot(args.head, bindings[str(args.head)]); selection, _ = snapshot(args.selection, bindings[str(args.selection)])
    head_inputs = head.get('training_input_hashes', {})
    require(set(head_inputs) == {'train_collection_sha256', 'checkpoint_sha256', 'cohort_sha256', 'fit_release_sha256'}
            and all(digest(v) for v in head_inputs.values()) and head_inputs['cohort_sha256'] == cohort_sha,
            'Fitted head training/cohort lineage differs')
    return SimpleNamespace(args=args, release=release, release_sha=release_sha, manifest=manifest, head=head,
        selection=selection, cohort_sha=cohort_sha, execution_inputs=set(evidence) | {str(args.repo / p) for p in numerical.bench.SOURCE_PINS},
        model=next(m for m in cohort['models'] if (m['arm'], m['seed']) == ('mix', args.seed)))


def timeout_cell(outcome, status, collection, failed):
    budget = bool(outcome.get('quota_stop_initiated') or outcome.get('quota_expired_at_observation') or outcome.get('inner_timeout_reported'))
    candidates = []
    for value in (failed, outcome.get('worker_failed_attempt'), collection.get('abort_reason')):
        while isinstance(value, dict):
            if value.get('category') == 'execution_budget':
                budget = True
            for field in ('current', 'case'):
                if isinstance(value.get(field), dict):
                    candidates.append(value[field])
            value = value.get('failure')
    candidates += [collection.get('current_at_stop'), status.get('current'), outcome.get('current_before_stop')]
    actual = [key(v) for v in candidates if isinstance(v, dict)]
    return actual[-1] if budget and actual else None


def validate_protocol(value, pin, context, bindings, mods):
    a = context.args; D = mods.D
    require(value.get('schema') == D.SCHEMA and value.get('mode') == 'test-rollout' and value.get('split') == 'test'
            and value.get('gate_protocol_sha256') == D.GATE_PROTOCOL_SHA and value.get('original_training_protocol_sha256') == D.TRAINING_PROTOCOL_SHA
            and value.get('core_sha256') == D.CORE_SHA and value.get('driver_sha256') == SOURCE_PINS['run_goop_action_gate_v1.py']
            and value.get('cohort_sha256') == context.cohort_sha and value.get('source_manifest_sha256') == bindings[str(a.manifest)]
            and value.get('head_sha256') == bindings[str(a.head)] and value.get('selection_sha256') == bindings[str(a.selection)]
            and value.get('schedule') == EXPECTED and value.get('features') == list(mods.C.FEATURES)
            and value.get('policies') == list(POLICIES) and value.get('selected_model_parameters_trainable') is False,
            'Exact committed driver protocol required')
    model = value.get('model', {})
    require(model.get('kind') == 'preselected_checkpoint' and model.get('seed') == a.seed and type(model.get('seed')) is int
            and model.get('arm') == 'mix' and model.get('objective') == 'faithful' and model.get('completed_updates') == 100000
            and model.get('checkpoint_sha256') == a.checkpoint_sha256 and digest(model.get('run_config_sha256'))
            and model.get('run_config_sha256') == context.model.get('config_sha256')
            and model.get('training_source_sha256') == D.SOURCE_PINS['train_goop_graph_support_cuda.py'], 'Committed model identity differs')
    incoming = value.get('input_sha256', {})
    require(isinstance(incoming, dict) and incoming and all(bindings.get(p) == h for p, h in incoming.items()), 'Every original protocol input must be root-bound')
    required = set(context.release['files_sha256']) | context.execution_inputs | {str(a.release), str(a.manifest.parent / 'metadata.json')}
    for record in context.manifest['records']:
        required.update(str(array_path(a.manifest.parent, record[field])) for field in ('positions', 'particle_types', 'step_context'))
    require(all(incoming.get(p) == bindings[p] for p in required), 'Committed protocol lacks required original input closure')
    return model


def collect_stage(entry, outcome, context, root, tree, bindings, mods):
    directory = Path(entry['directory']); prefix = str(directory.relative_to(root)); evidence = {'invalid_records': []}
    def present(name):
        return prefix + '/' + name in tree['files']
    def saved(name):
        return snapshot(directory / name, tree['files'][prefix + '/' + name]['sha256'])[0]
    if outcome['started'] is False:
        require(not directory.exists() and not any(v['path'] == prefix or v['path'].startswith(prefix + '/') for v in tree['output_tree_entries']),
                'Never-started stream contains output artifacts')
        evidence['scientific_verification'] = {'passed': False, 'reasons': ['invocation_never_started']}
        return {'seed': entry['seed'], 'outcome': outcome, 'runtime': None, 'rows': [], 'scientific_verification_passed': False,
                'cells': [{**v, 'state': 'never_started', 'failure': None} for v in EXPECTED], 'evidence': evidence}
    protocol = saved('protocol.json') if present('protocol.json') else None
    protocol_pin = tree['files'].get(prefix + '/protocol.json', {}).get('sha256')
    if protocol is not None:
        validate_protocol(protocol, protocol_pin, context, bindings, mods)
    def optional(name, field):
        if not present(name):
            return {}
        try:
            return saved(name)
        except (ValueError, TypeError, KeyError, OSError) as error:
            evidence[field] = str(error)
            return {}
    collection = optional('rollout_collection.json', 'invalid_collection')
    status = optional('status.json', 'invalid_status')
    failed = optional('failed_attempt.json', 'invalid_failed_attempt')
    scientific_verified = False; scientific_reasons = []
    proof = outcome.get('complete_collection')
    if proof is not None and not collection:
        evidence['invalid_collection'] = 'Original supervisor complete-collection commitment has no readable collection'
    timings = {}; collection_rows = {}; runtime = None
    if collection:
        try:
            require(protocol is not None and collection.get('schema') == mods.D.ROLLOUT_COLLECTION
                    and collection.get('mode') == 'test-rollout' and collection.get('split') == 'test'
                    and collection.get('driver_protocol_sha256') == protocol_pin and collection.get('model') == protocol['model']
                    and collection.get('expected_cells') == EXPECTED and collection.get('required_rows') == 120,
                    'Collection original grid/protocol/model differs')
            for field in ('gate_protocol_sha256', 'original_training_protocol_sha256', 'core_sha256', 'driver_sha256', 'cohort_sha256',
                          'source_manifest_sha256', 'head_sha256', 'selection_sha256', 'input_sha256'):
                require(collection.get(field) == protocol.get(field), 'Collection lineage differs: ' + field)
            rows = collection.get('rows'); cases = collection.get('case_timings')
            require(isinstance(rows, list) and isinstance(cases, list) and len(rows) == len(cases) == collection.get('committed_rows')
                    and [key(v) for v in rows] == [key(v) for v in cases] == [key(v) for v in EXPECTED[:len(rows)]],
                    'Exact one-to-one ordered collection timing receipt required')
            for row, case in zip(rows, cases):
                require(mods.V.finite(case.get('wall_seconds'), nonnegative=True), 'Finite whole-case wall time required')
                key_value = key(row); name = row.get('row_file')
                require(name == f'source_{key_value[0]:06d}_{key_value[1]}.json' and present(name)
                        and row.get('row_sha256') == tree['files'][prefix + '/' + name]['sha256']
                        and saved(name) == {k: v for k, v in row.items() if k not in ('row_file', 'row_sha256')}, 'Collection differs from committed row bytes')
            # Includes every opaque partial artifact and directory, excludes only
            # the self-describing final collection just as the frozen publisher.
            subtree = {'files': {k[len(prefix)+1:]: v for k, v in tree['files'].items() if k.startswith(prefix + '/') and k != prefix + '/rollout_collection.json'},
                       'output_tree_entries': [{'path': v['path'][len(prefix)+1:], 'kind': v['kind']} for v in tree['output_tree_entries']
                           if v['path'].startswith(prefix + '/') and v['path'] != prefix + '/rollout_collection.json']}
            require(collection.get('files') == subtree['files'] and collection.get('output_tree_entries') == subtree['output_tree_entries'], 'Final child collection tree differs')
            if proof is not None:
                require(isinstance(proof, dict) and set(proof) == {'path', 'sha256', 'status', 'required_rows', 'committed_rows'}
                        and proof['path'] == str(directory / 'rollout_collection.json')
                        and proof['sha256'] == tree['files'][prefix + '/rollout_collection.json']['sha256']
                        and proof['status'] == collection.get('status') and proof['required_rows'] == proof['committed_rows'] == 120,
                        'Original supervisor complete-collection commitment differs')
            complete_count = sum(r.get('status') == 'complete' for r in rows)
            all_inputs = collection.get('all_inputs_reverified'); model_unchanged = collection.get('model_state_verified_unchanged')
            require(type(all_inputs) is bool and type(model_unchanged) is bool
                    and collection.get('complete_rows') == complete_count and collection.get('coverage') == mods.D.collection_coverage(EXPECTED, rows),
                    'Collection completion/coverage audit differs')
            good = len(rows) == 120 and all_inputs and model_unchanged and collection.get('abort_reason') is None
            successful = good and complete_count == 120
            expected_status = 'complete' if successful else 'complete_with_guard_failures' if good else 'stopped_incomplete_or_failed'
            require(collection.get('status') == expected_status and collection.get('all_required_outcomes_complete') is successful
                    and status.get('schema') == mods.D.SCHEMA and status.get('state') == expected_status
                    and status.get('current') == collection.get('current_at_stop')
                    and all(status.get(k) == collection.get(k) for k in ('required_rows', 'committed_rows', 'complete_rows',
                        'all_inputs_reverified', 'model_state_verified_unchanged', 'abort_reason')),
                    'Original terminal status and collection outcome disagree')
            require(outcome.get('started') is True and outcome.get('stopped_and_reaped') is True
                    and type(outcome.get('exit_code')) is int, 'Stopped invocation outcome required')
            if expected_status in ('complete', 'complete_with_guard_failures'):
                require(outcome['exit_code'] == 0 and proof is not None, 'Completed collection requires original supervisor proof and zero exit')
            else:
                require(outcome['exit_code'] != 0 and proof is None, 'Incomplete collection must retain unsuccessful invocation outcome')
            collection_rows = {key(v): v for v in rows}; timings = {key(v): v['wall_seconds'] for v in cases}; runtime = collection.get('runtime')
            scientific_verified = all_inputs and model_unchanged
            if not all_inputs:
                scientific_reasons.append('original_input_reverification_not_passed')
            if not model_unchanged:
                scientific_reasons.append('original_model_state_reverification_not_passed')
        except (ValueError, TypeError, KeyError) as error:
            evidence['invalid_collection'] = str(error)
    if not collection:
        scientific_reasons.append('no_readable_final_collection')
    if 'invalid_collection' in evidence:
        scientific_reasons.append('final_collection_or_original_commitment_invalid')
    timeout = timeout_cell(outcome, status, collection, failed)
    rows = []; cells = []; source_cache = {}; planned_names = {f'source_{v["source_index"]:06d}_{v["policy"]}.json' for v in EXPECTED}
    unexpected = [k for k in tree['files'] if k.startswith(prefix + '/source_') and k.endswith('.json') and k[len(prefix)+1:] not in planned_names]
    require(not unexpected, 'Unexpected committed source/policy output: ' + repr(unexpected))
    for item in EXPECTED:
        source, policy = key(item); name = f'source_{source:06d}_{policy}.json'; value = None; error = None
        if present(name):
            try:
                require(protocol is not None, 'Committed row lacks original driver protocol')
                row = saved(name); artifact_name = name.removesuffix('.json') + '.npz'
                require(row.get('artifact_file') == artifact_name and present(artifact_name)
                        and row.get('artifact_sha256') == tree['files'][prefix + '/' + artifact_name]['sha256'], 'Row numeric artifact bytes differ')
                raw = (directory / artifact_name).read_bytes()
                require(hashlib.sha256(raw).hexdigest() == row['artifact_sha256'], 'Numeric bytes changed before deserialization')
                with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                    require(len(archive.files) == len(set(archive.files)), 'Duplicate NPZ array names')
                    arrays = {k: archive[k] for k in archive.files}
                if source not in source_cache:
                    record = context.manifest['records'][source]
                    source_cache[source] = tuple(byte_array(array_path(context.args.manifest.parent, record[f]), record[f]) for f in ('positions', 'particle_types'))
                positions, types = source_cache[source]
                expected = {'model_seed': entry['seed'], **item, 'checkpoint_sha256': context.args.checkpoint_sha256,
                    'head_sha256': bindings[str(context.args.head)], 'selection_sha256': bindings[str(context.args.selection)],
                    'source_manifest_sha256': bindings[str(context.args.manifest)],
                    'source_trajectory_content_sha256': context.manifest['records'][source]['trajectory_content_sha256'], 'protocol_sha256': protocol_pin}
                value = mods.V.validate_row(row, arrays, expected, positions, types, context.head, mods.C, mods.P)
                rows.append({**item, 'row_file': str(directory / name), 'row_sha256': tree['files'][prefix + '/' + name]['sha256'],
                    'artifact_file': str(directory / artifact_name), 'artifact_sha256': row['artifact_sha256'], 'validated': value,
                    'case_wall_seconds': timings.get((source, policy))})
            except (ValueError, TypeError, KeyError, OSError, EOFError) as failure:
                error = {'category': 'collector_validation_error', 'error_type': type(failure).__name__, 'error': str(failure)}
                evidence['invalid_records'].append({**item, 'row_file': str(directory / name), 'failure': error})
        if value is not None:
            category = (value.get('failure') or {}).get('category')
            state = 'committed_complete' if value['complete'] else 'timed_out_current' if category == 'execution_budget' else (
                'committed_execution_failed' if category in mods.V.OPERATIONAL else 'committed_guard_failed')
            failure = value.get('failure')
        else:
            state = 'timed_out_current' if timeout == (source, policy) else 'uncompleted_after_started_invocation'; failure = error
        cells.append({**item, 'state': state, 'failure': failure})
    if evidence['invalid_records'] or 'invalid_status' in evidence or 'invalid_failed_attempt' in evidence:
        scientific_verified = False
        scientific_reasons.append('invalid_committed_record_or_terminal_evidence')
    evidence.update(collection_present=bool(collection), collection_valid=bool(collection) and 'invalid_collection' not in evidence,
        committed_collection_row_count=len(collection_rows), protocol_present=protocol is not None,
        status=status, failed_attempt=failed, all_partial_artifacts_retained=True,
        scientific_verification={'passed': scientific_verified, 'reasons': scientific_reasons,
            'basis': 'Original collection/status/outcome/commitment consistent; original model and input reverification explicit.'})
    return {'seed': entry['seed'], 'outcome': outcome, 'runtime': runtime, 'rows': rows, 'cells': cells, 'evidence': evidence,
            'scientific_verification_passed': scientific_verified}


def collect(args, mods=None, at=None):
    mods = modules() if mods is None else mods
    root = ordinary(args.queue, directory=True); output = Path(args.output)
    ordinary(output.parent, directory=True)
    require(output.is_absolute() and '..' not in output.parts and not output.exists()
            and output != root and root not in output.parents, 'Fresh scalar output outside immutable queue required')
    approval, approval_sha = snapshot(args.release)
    require(approval.get('schema') == RELEASE_SCHEMA and approval.get('issued_by') == 'root'
            and approval.get('status') == 'approved_for_stopped_numeric_collection' and approval.get('hostname') == socket.gethostname()
            and approval.get('queue_root') == str(root) and approval.get('output') == str(output)
            and approval.get('collector_sha256') == sha(__file__) and approval.get('source_sha256') == SOURCE_PINS
            and all(digest(approval.get(k)) for k in ('queue_release_sha256', 'process_outcomes_sha256', 'queue_inventory_sha256')),
            'Exact root collector release required')
    bindings = {}; bind(bindings, approval.get('files_sha256'))
    bind(bindings, {str(Path(args.release)): approval_sha, str(Path(__file__).resolve()): sha(__file__)})
    bind(bindings, {str(HERE / p): h for p, h in SOURCE_PINS.items()})
    phase, phase_sha = snapshot(root / 'release_snapshot.json', approval.get('queue_release_sha256'))
    process, process_sha = snapshot(root / 'process_outcomes.json', approval.get('process_outcomes_sha256'))
    require(phase.get('schema') == 'adaptgns_goop_action_gate_scoped_release_v1' and phase.get('issued_by') == 'root'
            and phase.get('status') == 'approved_for_one_fixed_gate_phase' and phase.get('mode') == 'test-rollout'
            and phase.get('hostname') == socket.gethostname() and phase.get('host_role') == 'B'
            and phase.get('output_dir') == str(root) and phase.get('all_six_original_goop_models_frozen') is True
            and phase.get('no_retry_or_resume') is True and phase.get('does_not_displace_original_studies') is True,
            'Original final fixed gate release required')
    closed_pin = approval.get('process_closure', {})
    require(set(closed_pin) == {'path', 'sha256'} and bindings.get(closed_pin['path']) == closed_pin['sha256'], 'Byte-bound independent process closure required')
    closure, _ = snapshot(closed_pin['path'], closed_pin['sha256'])
    closure_observations = check_closed(closure, process, phase_sha, process_sha, mods, at)
    parent_command = closure['supervisor']['identity']['argv']
    require(len(parent_command) == 7 and parent_command[:3] == [sys.executable, str(HERE / 'supervise_goop_action_gate_scoped_v1.py'), '--execute']
            and len(set(parent_command[3::2])) == 2, 'Exact original supervisor command required')
    parent_flags = dict(zip(parent_command[3::2], parent_command[4::2]))
    require(set(parent_flags) == {'--release', '--output-dir'} and parent_flags['--output-dir'] == str(root)
            and bindings.get(parent_flags['--release']) == phase_sha, 'Root must bind original supervisor release path and output')
    # Only now may we hash the entire queue or open any numeric data artifact.
    tree = tree_snapshot(root)
    require(inventory_sha256(tree) == approval.get('queue_inventory_sha256'), 'Root-bound complete original queue inventory differs')
    lexical = phase.get('python_environment', {}).get('lexical_path')
    require(lexical == sys.executable and phase['python_environment'] == mods.S.python_environment(sys.executable), 'Original lexical interpreter and venv required')
    require(all(bindings.get(p) == h for p, h in phase.get('files_sha256', {}).items()), 'Root collection must bind complete parent input closure')
    require(phase.get('gpu_scope') == {'owned_indices': [0, 1], 'unassigned_devices': 'observe_without_control', 'live_handoff': False}
            and isinstance(phase.get('gpu_uuids'), list) and len(phase['gpu_uuids']) == 4
            and len({mods.B.canonical_gpu_uuid(v) for v in phase['gpu_uuids']}) == 4
            and phase.get('environment') == mods.S.ENVIRONMENT
            and phase['files_sha256'].get(str(HERE / 'supervise_goop_action_gate_scoped_v1.py')) == SOURCE_PINS['supervise_goop_action_gate_scoped_v1.py'],
            'Original physical scope/environment/reviewed supervisor differs')
    bind(bindings, {str(HERE / p): h for p, h in mods.D.SOURCE_PINS.items()})
    verify(bindings, lexical)
    ledger = None; ledger_name = root / 'phase_ledger.json'; ledger_pin = approval.get('phase_ledger_sha256')
    if ledger_name.exists():
        require(digest(ledger_pin), 'Existing final ledger must be bound')
        ledger, _ = snapshot(ledger_name, ledger_pin)
        require(ledger.get('schema') == SUPERVISOR_SCHEMA and ledger.get('mode') == 'test-rollout'
                and ledger.get('state') in ('complete_fixed_phase', 'stopped_requires_review')
                and ledger.get('jobs') == process['jobs'] and ledger.get('abort_reason') == process.get('abort_reason')
                and ledger.get('all_pinned_inputs_reverified') is True
                and ledger.get('numerical_arrays_audited_by_supervisor') is False,
                'Stopped final ledger contradicts original outcomes')
        require(all(bindings.get(p) == h or p == str(root / 'release_snapshot.json') and h == phase_sha
                    for p, h in ledger.get('input_sha256', {}).items()), 'Final ledger input closure differs')
        for name, descriptor in ledger.get('supervisor_files', {}).items():
            require(tree['files'].get(name) == descriptor, 'Retained supervisor evidence changed')
    else:
        require(ledger_pin is None, 'Missing ledger must be explicitly recorded as absent')
    phase_jobs = phase.get('jobs', [])
    require(len(phase_jobs) == 3 and [j.get('seed') for j in phase_jobs] == [0, 1, 2], 'All three original seed commands required')
    contexts = []; stages = []; shared = None
    for job, recorded in zip(phase_jobs, process['jobs']):
        entry = recorded['entry']; seed = job['seed']; gpu = 1 if seed == 1 else 0
        require(type(seed) is int and type(entry.get('seed')) is int and entry.get('stream') == f'mix_seed{seed}' and entry.get('seed') == seed
                and entry.get('stage') == 'test-rollout' and entry.get('arm') == 'mix' and entry.get('gpu') == job.get('gpu') == gpu
                and entry.get('gpu_uuid') == phase.get('gpu_uuids', [])[gpu] and entry.get('command') == job.get('command')
                and entry.get('directory') == str(root / 'jobs' / f'mix_seed{seed}') and entry.get('expected_cells') == EXPECTED,
                'Fixed ordered120-cell invocation mapping differs')
        context = driver_context(entry, phase, bindings, mods)
        current_shared = (context.cohort_sha, bindings[str(context.args.manifest)], bindings[str(context.args.selection)])
        require(shared is None or shared == current_shared, 'All three models must use one original cohort/test/selection')
        shared = current_shared; contexts.append(context)
        stages.append(collect_stage(entry, recorded['outcome'], context, root, tree, bindings, mods))
    summary = mods.V.summarize(stages, mods.P)
    invalid = any(s['evidence']['invalid_records'] or any(k in s['evidence'] for k in ('invalid_collection', 'invalid_status', 'invalid_failed_attempt')) for s in stages)
    result = {'schema': SCHEMA, 'status': 'stopped_collection_contains_invalid_evidence' if invalid else 'stopped_numeric_collection_complete',
        'collector_release_sha256': approval_sha, 'queue_release_sha256': phase_sha, 'process_outcomes_sha256': process_sha,
        'phase_ledger_sha256': ledger_pin, 'phase_ledger_present': ledger is not None, 'process_closure': closed_pin,
        'fresh_process_observations': closure_observations, 'queue_root': str(root), 'queue_inventory': tree,
        'queue_inventory_sha256': inventory_sha256(tree), 'source_sha256': SOURCE_PINS, 'input_sha256': bindings,
        'stages': stages, 'summary': summary, 'selection': contexts[0].selection,
        'selection_scope': 'Frozen train/validation selection diagnostics only; oracle labels are nondeployable.',
        'all_available_committed_numeric_rows_validated': not invalid,
        'all_three_stage_scientific_verifications_passed': all(s['scientific_verification_passed'] for s in stages),
        'array_retention_scope': 'Original numeric source/row bytes remain on original host; this export contains only scalar evidence.',
        'supervisor_runtime': None if ledger is None else ledger.get('elapsed_seconds'),
        'automatic_retry': False, 'new_test_or_training_executed': False, 'conference_readiness_assigned': False}
    result_snapshot = copy.deepcopy(result); raw = encode(result)
    require(result == result_snapshot and json.loads(raw) == result_snapshot, 'Scalar evidence mutated during serialization')
    verify(bindings, lexical)
    require(tree_snapshot(root) == tree, 'Stopped queue changed during array audit or scalar serialization')
    check_closed(closure, process, phase_sha, process_sha, mods, require_fresh=False)
    with output.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--collect', action='store_true')
    for name in ('queue', 'release', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if not args.collect:
        print(json.dumps({'schema': SCHEMA, 'status': 'description_only', 'required_cells': 360,
                          'array_audit_requires_root_stopped_release': True, 'new_research_execution': False}))
        return 0
    require(all(getattr(args, k) is not None for k in ('queue', 'release', 'output')), 'Explicit original queue/release/output required')
    collect(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
