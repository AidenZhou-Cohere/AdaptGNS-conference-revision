#!/usr/bin/env python3
"""Read-only scoped Sand receipts -> existing frozen final-cohort interface.

No tensor loading, CUDA, test access, process control or source modification.
Inventory hashes all retained files on a completed host. Build requires an
explicit root release binding the two inventory bytes and frozen source pins.
The payload/Adam/pairing conclusions are inherited from the reviewed completed
supervisor; this adapter does not claim a second tensor audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
SUPERVISOR_SCHEMA = 'adaptgns_sand_scoped_scientific_supervisor_v1'
TRAIN_SCHEMA = 'adaptgns_sand_graph_support_cuda_training_v1'
INVENTORY_SCHEMA = 'adaptgns_sand_completed_host_byte_inventory_scoped_v1'
RELEASE_SCHEMA = 'adaptgns_sand_final_cohort_release_scoped_v1'
COST_BASIS = 'fixed_execution_allocation_not_empirical_full_horizon_forecast'
PINS = {
    'supervisor': '2970922593f41f74713aff9208e4c6964bb1b3f351fee906724bbf52f11d4d13',
    'verifier_source': 'a22a59c46231c328e4939a0f021b3bb2fab9e2f93c0848bb1abd39aad2d5d305',
    'trainer': 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124',
    'capacity_source': 'ca9b5ed45455b575179464d35c42d0d03e87edb7508f4deea2451f9f3fb50878',
    'lifecycle_source': 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd',
    'train_manifest': 'f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f',
    'admission': 'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73',
    'structural_report': 'bc63fabfc07a663ab866f454c07e984dd57b8d4b32aa06a760838c4f84da72e6',
    'protocol': 'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d',
    'schedule_plan': '403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b',
    'amendment': '411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738',
}
FINAL_PINS = {
    'protocol': PINS['protocol'], 'train_admission': PINS['admission'], 'trainer_source': PINS['trainer'],
    'benchmark_helper': '8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13',
    'diagnostic_source': '952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58',
}
POLICIES = ['base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def verify_bindings(bindings):require(all(sha(p)==d for p,d in bindings.items()),'Bound source/inventory/control bytes changed before publication')


def verify_inventory_tree(inv):
    root=Path(inv['root']);paths=sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
    require(all(not p.is_symlink() for p in paths) and {str(p.relative_to(root)) for p in paths}==set(inv['files']),
        'Completed host tree changed before inventory publication')
    verify_bindings(inv['inventory_evidence_bindings'])


def stamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, 'Timezone-aware timestamp required')
    return parsed


def schedule(role):
    mapping={'A':[('base',1,0),('mix',1,1),('base',2,2),('mix',2,3)],'B':[('base',0,2),('mix',0,3)]}
    require(role in mapping,'Unknown scoped host role')
    return [{'id':f'{arm}_seed{seed}','wave':role,'gpu':gpu,'objective':'faithful','arm':arm,'seed':seed}
            for arm,seed,gpu in mapping[role]]


def check_receipts(receipts, files):
    launch, summary, terminal, wave, release = [receipts[k] for k in ('launch', 'summary', 'terminal', 'wave', 'release')]
    role = launch.get('host_role')
    require(role in ('A', 'B'), 'Unknown host role')
    expected = schedule(role)
    require(launch.get('schema') == summary.get('schema') == wave.get('schema') == SUPERVISOR_SCHEMA,
            'Wrong scientific supervisor schema')
    require(summary.get('host_role') == wave.get('wave') == role,
            'Host role receipts differ')
    require(launch.get('cohort_id') == summary.get('cohort_id') == release.get('cohort_id')
            and isinstance(launch.get('cohort_id'), str) and bool(launch['cohort_id'].strip()), 'Cohort identity differs')
    require(summary.get('status') == 'verified_host_scientific_endpoints'
            and terminal.get('state') == 'complete_host_scoped_training' and wave.get('state') == 'verified',
            'Only completed, verified hosts may be inventoried')
    require(summary.get('all_inputs_reverified') is True and summary.get('whole_six_model_cohort_admitted') is False
            and terminal.get('whole_six_model_cohort_admitted') is False, 'Incomplete host verification')
    require(summary.get('scientific_protocol_sha256')==PINS['protocol']
            and summary.get('schedule_plan_sha256')==PINS['schedule_plan']
            and summary.get('amendment_sha256')==PINS['amendment']
            and summary.get('endpoint_verifier_sha256')==PINS['verifier_source'], 'Scoped scientific/operational lineage differs')
    require(terminal.get('summary_sha256') == files['worker_summary.json']['sha256'], 'Terminal summary digest differs')
    require(release.get('schema') == 'adaptgns_sand_scoped_scientific_release_v1'
            and release.get('status') == 'admitted_fresh_scoped_training' and release.get('issued_by') == 'root'
            and release.get('files_sha256') == launch.get('files_sha256')
            and release.get('host_role') == role and release.get('host') == launch.get('hostname')
            and files['inputs/release.json']['sha256'] == launch.get('release_sha256'), 'Root training release differs')
    require(all(launch.get('files_sha256', {}).get(k) == v for k, v in PINS.items()), 'Frozen training source/data/protocol hash differs')
    require(launch.get('environment', {}).get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic environment differs')
    require(launch.get('schedule') == schedule('B') + schedule('A')
            and release.get('schedule') == launch['schedule'], 'Scientific schedule differs')
    snapshot_keys=set(launch['files_sha256'])-{'python','python_target'}
    actual={Path(name).stem for name in files if Path(name).parent==Path('inputs') and name!='inputs/release.json'}
    require(actual==snapshot_keys,'Complete scoped input snapshots required')
    for key in snapshot_keys:
        matches=[name for name in files if Path(name).parent==Path('inputs') and Path(name).stem==key]
        require(len(matches)==1 and files[matches[0]]['sha256']==launch['files_sha256'][key],'Changed bound input snapshot: '+key)
    require(wave.get('unreaped_owned_children')==[] and len(wave.get('all_children',[]))==len(expected)
            and all(c.get('exit_code')==0 and c.get('signals')==[] for c in wave['all_children'])
            and wave.get('never_started')==[],'Every scoped child must have cleanly finished')
    require(stamp(wave['ended_utc']) <= stamp(wave['endpoint_audit_finished_utc']) <= stamp(terminal['ended_utc'])
            and stamp(wave['endpoint_audit_finished_utc']) <= stamp(release['training_stop_utc']),
            'Scoped reaping/endpoint audit chronology differs')
    jobs = summary.get('jobs', [])
    require(len(jobs) == len(expected) and {j.get('id') for j in jobs} == {j['id'] for j in expected}, 'Missing or duplicate host job')
    for plan in expected:
        job = next(j for j in jobs if j['id'] == plan['id'])
        require(all(job.get(k) == v for k, v in plan.items()) and type(job.get('seed')) is int
                and job.get('status') == 'verified_scientific100k_endpoint', 'Scientific job identity differs')
        require(job.get('external', {}).get('exit_code') == 0 and job['external'].get('signals') == [], 'Unsuccessful child retained; cannot admit cohort')
        require(job.get('history', {}).get('graph_updates') == 100000 and job['history'].get('logged_rows') == 1001,
                'Incomplete graph/log history audit')
        for key, steps, states in [('endpoint', 100000, True), ('initial_checkpoint', 0, False)]:
            e = job.get(key, {})
            require(e.get('completed_steps') == steps and type(e.get('finite_model_tensors')) is int and e['finite_model_tensors'] > 0
                    and e.get('adam_parameter_states') == (e['finite_model_tensors'] if states else 0)
                    and e.get('cpu_cuda_rng_serialized') is True and e.get('checkpoint_schema_verified') is True,
                    'Incomplete model/Adam/RNG endpoint audit')
        prefix = 'jobs/' + job['id'] + '/'
        config, status, pointer = [receipts['jobs'][job['id']][k] for k in ('config', 'status', 'pointer')]
        require(canonical(config) == job.get('config_sha256') and config.get('schema') == TRAIN_SCHEMA
                and config.get('research_protocol_sha256') == PINS['protocol'] and config.get('checkpoint_every') == 10000
                and config.get('log_every') == 100 and config.get('updates') == 100000
                and config.get('arm') == plan['arm'] and config.get('seed') == plan['seed'], 'Job configuration differs')
        require(status.get('schema') == TRAIN_SCHEMA and status.get('state') == 'complete' and status.get('error') is None
                and status.get('completed_steps') == status.get('committed_steps') == status.get('requested_steps') == 100000
                and status.get('run_config_sha256') == job['config_sha256'] and status.get('latest_checkpoint') == pointer,
                'Job terminal/pointer/config differs')
        require(job.get('runtime', {}).get('deterministic_algorithms') is True
                and job['runtime'].get('cublas_workspace_config') == ':4096:8', 'Job runtime determinism differs')
        require(set(job.get('artifact_sha256', {})) == {'protocol.json', 'status.json', 'history.json', 'latest.json'}, 'Missing job artifact hash')
        for name, digest in job['artifact_sha256'].items():
            require(files.get(prefix + name, {}).get('sha256') == digest, 'Changed verified job artifact: ' + prefix + name)
        checkpoints = {prefix + f'checkpoint-{step:09d}.pt' for step in range(0, 100001, 10000)}
        actual = {name for name in files if name.startswith(prefix) and re.search(r'/checkpoint-[^/]+\.pt$', name)}
        require(actual == checkpoints and all(files[name]['bytes'] > 0 for name in checkpoints), 'All eleven checkpoints must be retained')
        for key, steps in [('initial_pointer', 0), ('final_pointer', 100000)]:
            p = job[key]
            require(p == {'path': f'checkpoint-{steps:09d}.pt', 'completed_steps': steps, 'run_config_sha256': job['config_sha256'],
                          'sha256': files[prefix + f'checkpoint-{steps:09d}.pt']['sha256']}, 'Changed initial/final checkpoint bytes')
        require(pointer == job['final_pointer'], 'Latest checkpoint differs from verified endpoint')
    pairs = summary.get('pairing', [])
    require(pairs == wave.get('pairing') and len(pairs) == len(expected) // 2
            and {p.get('seed') for p in pairs} == {j['seed'] for j in expected}, 'Pairing coverage differs')
    for pair in pairs:
        require(type(pair.get('seed')) is int and all(pair.get(key) is True for key in
            ('initial_model_adam_cpu_cuda_rng_exact', 'all100k_frame_noise_native_graph_rng_exact', 'all_logged_saved_lr_exact'))
            and pair.get('initial_audit_ordering') == 'verified after training; no before-step1 barrier claimed', 'Incomplete paired tensor/schedule audit')
    return role


def inventory(root, process_check):
    root = Path(root).resolve()
    process_raw=Path(process_check).read_bytes();process=json.loads(process_raw);process_sha=hashlib.sha256(process_raw).hexdigest()
    adapter_sha=sha(__file__)
    require(process.get('schema') == 'adaptgns_sand_completed_host_process_check_v1' and process.get('issued_by') == 'root'
            and process.get('matching_training_processes') == [] and process.get('supervisor_exited') is True
            and process.get('all_children_reaped') is True, 'Root completed-host process check required')
    paths = sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
    require(paths and all(not p.is_symlink() for p in paths), 'Symlink or empty host tree is not admissible')
    require(not any(p.name.endswith('.tmp') or p.name == 'run.lock' or p.name.startswith(('unsuccessful', 'state_preservation_error')) for p in paths),
            'Unsuccessful/partial/locked host tree cannot be promoted; retain it separately')
    stamps = {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    files = {str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths}
    def saved(name):
        raw=(root/name).read_bytes();require(hashlib.sha256(raw).hexdigest()==files[name]['sha256'],'Receipt bytes changed after inventory hash');return json.loads(raw)
    receipts = {k: saved(name) for k, name in [('launch', 'launch.json'), ('summary', 'worker_summary.json'),
                ('terminal', 'status.json'), ('release', 'inputs/release.json')]}
    role = receipts['launch']['host_role']
    receipts['wave'] = saved(f'wave_{role}.json')
    receipts['jobs'] = {j['id']: {key: saved('jobs/'+j['id']+'/'+name) for key, name in
        [('config', 'protocol.json'), ('status', 'status.json'), ('pointer', 'latest.json')]} for j in schedule(role)}
    require(process.get('host_role') == role and process.get('hostname') == receipts['launch']['hostname']
            and process.get('supervisor_pid') == receipts['launch']['pid']
            and process.get('cohort_id') == receipts['launch']['cohort_id'], 'Root process check refers to another training host')
    require(stamp(process['checked_utc']) >= stamp(receipts['terminal']['ended_utc'])
            and 0 <= (datetime.now(timezone.utc) - stamp(process['checked_utc'])).total_seconds() <= 1800,
            'Fresh post-completion root process check required')
    check_receipts(receipts, files)
    after = sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
    require(after == paths and all(not p.is_symlink() and (p.stat().st_size, p.stat().st_mtime_ns) == stamps[str(p.relative_to(root))] for p in paths),
            'Host tree changed during byte inventory')
    return {'schema': INVENTORY_SCHEMA, 'status': 'completed_host_bytes_verified', 'issued_by': 'root', 'host_role': role,
            'cohort_id': receipts['launch']['cohort_id'], 'source_sha256': adapter_sha, 'root': str(root), 'created_utc': now(),
            'process_check_sha256': process_sha, 'process_check': process, 'files': files, 'receipts': receipts,
            'inventory_evidence_bindings':{**{str(root/name):d['sha256'] for name,d in files.items()},
                str(Path(process_check).resolve()):process_sha,str(Path(__file__).resolve()):adapter_sha},
            'tensor_audit_basis': 'reviewed scoped supervisor verified initial/final payloads and every paired history before publication',
            'intermediate_checkpoints': 'all eleven byte hashes retained; nine intermediate payloads not deserialized by this adapter or supervisor'}


def build(inventories, release, source_files):
    require(set(inventories) == {'A', 'B'}, 'Both host inventories required')
    require(release.get('schema') == RELEASE_SCHEMA and release.get('status') == 'approved_for_cohort_interface_mapping'
            and release.get('issued_by') == 'root' and release.get('adapter_sha256') == sha(__file__)
            and isinstance(release.get('review_rationale'), str) and bool(release['review_rationale'].strip()), 'Exact root adapter release required')
    require(set(source_files) == set(FINAL_PINS) and all(sha(source_files[k]) == v for k, v in FINAL_PINS.items())
            and release.get('final_source_sha256') == FINAL_PINS, 'Frozen final evaluation sources differ')
    models, audited, pairs, provenance = [], [], [], {}
    bindings={str(Path(p).resolve()):FINAL_PINS[k] for k,p in source_files.items()};bindings[str(Path(__file__).resolve())]=release['adapter_sha256']
    for role in ('A', 'B'):
        path = Path(inventories[role]);raw=path.read_bytes();inv=json.loads(raw);inventory_sha=hashlib.sha256(raw).hexdigest()
        bindings[str(path.resolve())]=inventory_sha
        require(inventory_sha == release.get('inventory_sha256', {}).get(role) and inv.get('source_sha256') == release['adapter_sha256']
                and inv.get('schema') == INVENTORY_SCHEMA and inv.get('status') == 'completed_host_bytes_verified'
                and inv.get('issued_by') == 'root' and inv.get('host_role') == role
                and inv.get('cohort_id') == release.get('cohort_id'), 'Root-bound host inventory differs')
        require(check_receipts(inv['receipts'], inv['files']) == role, 'Wrong inventory role')
        provenance[role] = {'inventory_sha256': inventory_sha, 'summary_sha256': inv['files']['worker_summary.json']['sha256'],
            'launch_sha256': inv['files']['launch.json']['sha256'], 'process_check_sha256': inv['process_check_sha256'],
            'all_retained_files': inv['files'], 'inventory_created_utc': inv['created_utc']}
        for job in inv['receipts']['summary']['jobs']:
            row = {'objective': 'faithful', 'arm': job['arm'], 'seed': job['seed'], 'completed_steps': 100000,
                   'checkpoint_sha256': job['final_pointer']['sha256'], 'host_role': role,
                   'checkpoint_path': str(Path(job['directory']) / job['final_pointer']['path']), 'config_sha256': job['config_sha256']}
            models.append(row)
            audited.append({**row, 'graph_history_updates': 100000, 'checkpoint_every': 10000, 'log_every': 100,
                'all_optimizer_steps_equal_100000': True, 'all_state_and_moments_finite': True,
                'source_data_protocol_verified': True, 'checkpoint_bytes_verified': True,
                'payload_audit_basis': 'reviewed supervisor endpoint/initial checks; current hash matches completed audit',
                'initial_checkpoint_sha256': job['initial_pointer']['sha256'], 'source_endpoint_report': job['endpoint']})
        for pair in inv['receipts']['summary']['pairing']:
            pairs.append({'seed': pair['seed'], 'initial_model_tensor_identity': True, 'initial_cpu_cuda_rng_identity': True,
                'all_frame_noise_lr_schedules_equal': True, 'all_graph_budgets_and_rng_material_verified': True,
                'source_pairing_report': pair})
    require(len({r['checkpoint_sha256'] for r in models}) == 6, 'Six distinct endpoint hashes required')
    common = {'issued_by': 'root', 'dataset': 'Sand', 'training_schema': TRAIN_SCHEMA,
              'protocol_sha256': FINAL_PINS['protocol'], 'training_admission_sha256': FINAL_PINS['train_admission'],
              'trainer_source_sha256': FINAL_PINS['trainer_source'], 'cohort_id': release['cohort_id'],
              'adapter_sha256': sha(__file__), 'cost_basis': COST_BASIS, 'created_utc': now(),
              'schedule_plan_sha256':PINS['schedule_plan'],'operational_amendment_sha256':PINS['amendment']}
    audit = {**common, 'schema': 'adaptgns_sand_graph_support_complete_cohort_audit_v1',
             'status': 'all_six_endpoints_and_pairing_verified', 'models': audited, 'paired_seeds': sorted(pairs, key=lambda p: p['seed']),
             'provenance': provenance, 'fresh_tensor_deserialization_by_adapter': False,
             'audit_scope': 'reviewed supervisor initial/final tensor+Adam checks and full100k schedule pairing; fresh hashes of all retained files',
             'intermediate_payloads_deserialized': False, 'reserved_test_accessed_by_adapter': False}
    audit['input_binding_sha256']=bindings
    cohort = {**common, 'schema': 'adaptgns_sand_graph_support_final_cohort_v1', 'status': 'frozen_for_final_evaluation',
              'updates': 100000, 'models': models, 'policies': POLICIES, 'deterministic_algorithms': True,
              'cublas_workspace_config': ':4096:8', 'benchmark_helper_sha256': FINAL_PINS['benchmark_helper'],
              'diagnostic_source_sha256': FINAL_PINS['diagnostic_source']}
    return audit, cohort


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--mode', choices=('inventory', 'build'))
    for name in ('host-root', 'process-check', 'output', 'inventory-a', 'inventory-b', 'root-release', 'output-dir', *FINAL_PINS):
        p.add_argument('--' + name.replace('_', '-'), type=Path)
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'status': 'description_only', 'operations': ['inventory complete host bytes', 'build root-released final cohort interface'],
                          'payload_audit': 'inherit reviewed supervisor; no second tensor audit', 'test_access': False}, indent=2)); return 0
    if a.mode == 'inventory':
        require(a.host_root and a.process_check and a.output, 'Host root/process check/new output required')
        require(a.host_root.resolve() not in a.output.resolve().parents, 'Inventory output must be outside host tree')
        value=inventory(a.host_root,a.process_check);raw=encoded(value)
        verify_inventory_tree(value)
        require(0 <= (datetime.now(timezone.utc)-stamp(value['process_check']['checked_utc'])).total_seconds() <= 1800,'Completed-host process evidence expired')
        with a.output.open('xb') as f:f.write(raw)
    elif a.mode == 'build':
        require(a.inventory_a and a.inventory_b and a.root_release and a.output_dir, 'Both inventory paths/root release/new output directory required')
        sources = {key: getattr(a, key) for key in FINAL_PINS}
        require(all(sources.values()), 'Every frozen evaluation source path required')
        release_raw=a.root_release.read_bytes();release=json.loads(release_raw);release_sha=hashlib.sha256(release_raw).hexdigest()
        audit, cohort = build({'A': a.inventory_a, 'B': a.inventory_b}, release, sources)
        a.output_dir.mkdir(exist_ok=False)
        (a.output_dir / 'root_release.json').write_bytes(release_raw)
        audit['root_release_sha256']=release_sha;audit['input_binding_sha256'][str(a.root_release.resolve())]=release_sha
        audit_raw=encoded(audit);cohort['cohort_audit_sha256']=hashlib.sha256(audit_raw).hexdigest();cohort['root_release_sha256']=release_sha
        cohort_raw=encoded(cohort)
        try:
            verify_bindings(audit['input_binding_sha256'])
            require(sha(a.output_dir/'root_release.json')==release_sha,'Copied root release changed')
            with (a.output_dir/'cohort_audit.json').open('xb') as f:f.write(audit_raw)
            verify_bindings(audit['input_binding_sha256'])
            require(sha(a.output_dir/'cohort_audit.json')==cohort['cohort_audit_sha256'],'Audit changed before final cohort publication')
            require(sha(a.output_dir/'root_release.json')==release_sha,'Copied root release changed before final cohort publication')
            with (a.output_dir/'cohort.json').open('xb') as f:f.write(cohort_raw)
        except BaseException as error:
            write_new(a.output_dir/'failed_publication.json',{'error_type':type(error).__name__,'error':str(error),'all_existing_outputs_retained':True})
            raise
    else:
        p.error('Execution requires explicit mode')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
