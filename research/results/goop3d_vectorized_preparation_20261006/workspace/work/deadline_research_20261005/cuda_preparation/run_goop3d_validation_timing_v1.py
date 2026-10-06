#!/usr/bin/env python3
"""Six bounded validation timing calls around the unchanged reviewed evaluator.

Root-gated Linux supervisor; three two-GPU mode waves, no training/test/retry.
The whole child lifetime is timed and reaped with wait4. Default is inert.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop3d_bounded_validation_timing_supervisor_v1'
RELEASE_SCHEMA = 'adaptgns_goop3d_bounded_validation_timing_release_v1'
EVALUATOR_SHA = '9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de'
WORKSHEET_SHA = '8a4946e813a6b9ef3d0cb86c5649d614d2bc89fd08b9962bf3ec877e94a20db6'
TRAINER_SHA = '8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc'
LIFECYCLE_SHA = 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd'
MODES = ('full-rollout', 'same-state', 'clean-validation')
SCHEDULE = [dict(id=f'{arm}_seed0_{mode}', arm=arm, seed=0, mode=mode, gpu=gpu)
            for mode in MODES for gpu, arm in enumerate(('base', 'mix'))]
ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'LD_LIBRARY_PATH': '/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
COMMON = ('repo', 'manifest', 'split_admission', 'structural_report', 'acquisition_report',
          'context_semantics', 'auxiliary_report', 'trainer_source', 'protocol')
DEADLINE = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)
POLL, CLEANUP = .1, 15.


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc)


def write(path, value):
    path = Path(path); temp = path.with_name(path.name + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def load(name, digest):
    path = HERE / name
    require(sha(path) == digest, 'Reviewed supervisor dependency differs: ' + name)
    spec = importlib.util.spec_from_file_location('_timing_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def modules():
    return (load('evaluate_goop3d_graph_support_v1.py', EVALUATOR_SHA),
            load('goop3d_deadline_worksheet_v1.py', WORKSHEET_SHA),
            load('measure_sand_cuda_capacity_v2.py', LIFECYCLE_SHA),
            load('train_goop3d_graph_support_cuda_v2.py', TRAINER_SHA))


def child_arguments(root, output, job):
    values = {key: Path(root['common'][key]).resolve() for key in COMMON}
    values.update(purpose='capacity_timing', mode=job['mode'], split='valid', arm=job['arm'], seed=0,
        checkpoint=Path(root['checkpoints'][job['arm']]['path']).resolve(),
        checkpoint_sha256=root['checkpoints'][job['arm']]['sha256'], checkpoint_updates=512,
        cuda_index=job['gpu'], gpu_uuid=root['gpu_uuids'][job['gpu']], threads=2,
        max_seconds=root['inner_seconds'][job['mode']],
        release=Path(root['evaluator_releases'][job['id']]).resolve(), output_dir=output/'jobs'/job['id'])
    return SimpleNamespace(**values)


def command(root, args):
    # Preserve the venv symlink path for execution; byte bindings resolve it.
    result = [str(Path(root['python']).absolute()), '-B', '-u', str(HERE/'evaluate_goop3d_graph_support_v1.py'), '--execute']
    for key in ('purpose', 'mode', 'split', 'arm', 'seed', 'checkpoint_sha256', 'checkpoint_updates', 'cuda_index',
                'gpu_uuid', 'threads', 'max_seconds', 'release', *COMMON, 'checkpoint', 'output_dir'):
        result += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    return result


def preflight(args, root, E, W, B, T):
    output = args.output_dir.resolve()
    require(root.get('schema') == RELEASE_SCHEMA and root.get('status') == 'admitted_for_six_validation_timing'
            and root.get('issued_by') == 'root' and root.get('scientific_training_admitted') is False
            and root.get('scientific_endpoint_selected') is False and root.get('test_access_allowed') is False
            and root.get('supervisor_sha256') == sha(__file__) and root.get('schedule') == SCHEDULE
            and root.get('environment') == ENVIRONMENT and root.get('output_dir') == str(output)
            and root.get('hostname') == socket.gethostname() and root.get('evaluation_overlap_credit') is False,
            'Exact root six-call capacity-only timing release required')
    require(now() < DEADLINE and os.environ.get('CUDA_VISIBLE_DEVICES') is None, 'Cutoff/unremapped device environment required')
    checked = datetime.fromisoformat(root['process_identity_checked_utc'])
    require(checked.tzinfo is not None and -60 <= (now()-checked).total_seconds() <= 300, 'Fresh root process inventory required')
    require(len(root.get('gpu_uuids', [])) == 2 and len({B.canonical_gpu_uuid(v) for v in root['gpu_uuids']}) == 2,
            'Exactly two distinct released physical GPUs required')
    require(set(root.get('common', {})) == set(COMMON) and set(root.get('checkpoints', {})) == {'base', 'mix'}
            and set(root.get('evaluator_releases', {})) == {j['id'] for j in SCHEDULE}, 'Exact six-call input mapping required')
    require(set(root.get('inner_seconds', {})) == set(root.get('outer_seconds', {})) == set(MODES), 'Exactly three declared mode budgets required')
    for mode in MODES:
        inner, outer = root['inner_seconds'][mode], root['outer_seconds'][mode]
        require(type(inner) is int and 1 <= inner <= 7200 and type(outer) is int and inner < outer <= 7500,
                'Explicit inner and larger whole-child wall-clock limits required')
    require(root.get('cleanup_seconds') == CLEANUP and root.get('whole_supervisor_outer_timeout_required') is True,
            'Fifteen-second owned cleanup and external supervisor watchdog required')
    capacity, receipt = read(root['capacity_summary']), read(root['capacity_stop_receipt'])
    _, _, pointers = W.capacity_inputs(capacity)
    require(receipt.get('schema') == 'adaptgns_goop3d_capacity_stopped_receipt_v1' and receipt.get('issued_by') == 'root'
            and receipt.get('status') == 'supervisor_and_all_six_workers_stopped'
            and receipt.get('capacity_summary_sha256') == sha(root['capacity_summary'])
            and receipt.get('all_owned_processes_reaped_or_independently_verified_absent') is True,
            'Completed and root-reviewed stopped capacity queue required')
    for arm in ('base', 'mix'):
        require(root['checkpoints'][arm]['sha256'] == pointers[arm, 0]
                and sha(root['checkpoints'][arm]['path']) == pointers[arm, 0], 'Exact verified capacity seed0 endpoint required')
    required = [Path(root[k]).resolve() for k in ('python', 'capacity_summary', 'capacity_stop_receipt')]
    required += [Path(v).resolve() for k, v in root['common'].items() if k != 'repo']
    required += [Path(v).resolve() for v in root['evaluator_releases'].values()]
    required += [Path(v['path']).resolve() for v in root['checkpoints'].values()]
    required += [HERE/name for name in ('evaluate_goop3d_graph_support_v1.py', 'goop3d_native_evaluation_v1.py',
        'goop3d_diagnostic_metrics_v1.py', 'goop3d_graph_support_vectorized_v1.py', 'goop3d_deadline_worksheet_v1.py',
        'measure_sand_cuda_capacity_v2.py', 'audit_goop3d_auxiliary.py')]
    require(all(root.get('files_sha256', {}).get(str(path)) == sha(path) for path in required), 'Every launch input must be root-byte-bound')
    bindings = E.merge_bindings(root['files_sha256'], {args.release:sha(args.release), __file__:sha(__file__)},
        {Path(root['common']['repo'])/key:value for key,value in T.SOURCE_PINS.items()})
    children = []
    for job in SCHEDULE:
        child = child_arguments(root, output, job)
        release = E.release_gate(child)
        require(release.get('scientific_training_admitted') is False, 'Child scope must remain capacity only')
        bindings = E.merge_bindings(bindings, E.execution_bindings(child, release, T, sha(child.release)))
        children.append((job, child, command(root, child)))
    protected = [Path(root['common']['manifest']).resolve().parent, HERE,
        (Path(root['common']['repo'])/'adaptive-gns').resolve()]
    protected += [Path(v['path']).resolve().parent for v in root['checkpoints'].values()]
    require(not output.exists() and all(output != p and p not in output.parents and output not in p.parents for p in protected),
            'Fresh timing output must be separate from data, checkpoints and sources')
    E.verify_bindings(bindings)
    return children, bindings


def other_research_processes(excluded=()):
    names = {Path(__file__).name, 'evaluate_goop3d_graph_support_v1.py', 'measure_goop3d_vectorized_capacity_v1.py',
             'train_goop3d_graph_support_cuda_v2.py', 'validate_goop3d_vectorized_cuda_v1.py'}
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) in excluded: continue
        try:
            argv = [v.decode() for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError, ProcessLookupError): continue
        if any(Path(v).name in names for v in argv): found.append({'pid':int(directory.name), 'argv':argv})
    return found


def ancestor_pids():
    # An external timeout legitimately contains this script in its argv.
    # Exclude only this process and its actual ancestors, never siblings.
    result, pid = set(), os.getpid()
    while pid and pid not in result:
        result.add(pid)
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        pid = int(fields[1])
    return result


def spawn(output, job, child_args, argv, B):
    stdout_path, stderr_path = (output/'logs'/(job['id']+'.'+suffix) for suffix in ('stdout.log', 'stderr.log'))
    stdout = stdout_path.open('xb'); stderr = stderr_path.open('xb')
    started, started_utc = time.perf_counter(), now().isoformat()
    try:
        proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env={**os.environ, **ENVIRONMENT}, start_new_session=True)
    except BaseException:
        stdout.close(); stderr.close(); raise
    child = dict(job=job, args=child_args, process=proc, identity=None, command=argv, handles=(stdout, stderr), signals=[],
        started=started, started_utc=started_utc, stdout_file=str(stdout_path), stderr_file=str(stderr_path))
    # Return the owned Popen first; caller registers it before identity validation.
    return child


def capture_identity(child, B):
    try: identity = B.process_identity(child['process'].pid)
    except (FileNotFoundError, ProcessLookupError): return
    require(identity['ppid'] == os.getpid() and identity['argv'] == child['command'], 'Spawned child identity differs')
    child['identity'] = identity


def reap(child):
    proc = child['process']
    if proc.returncode is not None: return None
    pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
    if not pid: return None
    proc.returncode = os.waitstatus_to_exitcode(status)
    for handle in child['handles']: handle.close()
    return {'pid':pid, 'identity':child['identity'], 'command':child['command'], 'started_utc':child['started_utc'],
        'ended_utc':now().isoformat(), 'elapsed_seconds':time.perf_counter()-child['started'], 'exit_code':proc.returncode,
        'stopped_and_reaped':True, 'signals':child['signals'], 'stdout_file':child['stdout_file'], 'stderr_file':child['stderr_file'],
        'peak_host_rss_bytes':usage.ru_maxrss*1024, 'peak_host_rss_source_units':'Linux ru_maxrss KiB, multiplied by1024',
        'user_cpu_seconds':usage.ru_utime, 'system_cpu_seconds':usage.ru_stime,
        'nominal_poll_interval_seconds':POLL, 'elapsed_includes_exit_observer_lag':True}


def cleanup(children, B, record):
    began, sent = time.perf_counter(), set()
    while any(c['process'].returncode is None for c in children):
        elapsed = time.perf_counter()-began
        for threshold, signum in ((0, signal.SIGINT), (5, signal.SIGTERM), (10, signal.SIGKILL)):
            if elapsed >= threshold and signum not in sent:
                B.stop_owned(children, signum); sent.add(signum)
        for child in children:
            try:
                external = reap(child)
                if external: record(child, external)
            except Exception as error:
                child.setdefault('cleanup_errors', []).append({'type':type(error).__name__, 'error':str(error)})
        if elapsed >= CLEANUP: break
        time.sleep(POLL)
    return [{'id':c['job']['id'], 'pid':c['process'].pid, 'identity':c['identity'], 'command':c['command'], 'signals':c['signals']}
            for c in children if c['process'].returncode is None]


def supervise(args, root, specs, bindings, E, B):
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'jobs').mkdir(); (output/'logs').mkdir()
    lock = output/'run.lock'; write(lock, {'pid':os.getpid(), 'created_utc':now().isoformat(), 'release_sha256':sha(args.release)})
    entries = {job['id']:{**job, 'directory':str(child.output_dir), 'state':'not_started', 'external':None, 'files_sha256':{}}
               for job, child, argv in specs}
    start = time.perf_counter(); observations = []; children = []; unclosed = []; failure = None
    write(output/'launch.json', {'schema':SCHEMA, 'release_sha256':sha(args.release), 'files_sha256':bindings,
        'commands':{job['id']:argv for job, child, argv in specs}, 'schedule':SCHEDULE, 'environment':ENVIRONMENT,
        'gpu_uuids':root['gpu_uuids'], 'inner_seconds':root['inner_seconds'], 'outer_seconds':root['outer_seconds'],
        'evaluation_overlap_credit':False, 'scientific_training_admitted':False})
    def snapshot(state):
        write(output/'status.json', {'schema':SCHEMA, 'state':state, 'elapsed_seconds':time.perf_counter()-start,
            'entries':list(entries.values()), 'unreaped_owned_children':unclosed, 'failure':failure})
    def record(child, external):
        entry = entries[child['job']['id']]; entry.update(external=external, state='stopped')
        write(output/'logs'/(child['job']['id']+'.external.json'), external)
        snapshot('running')
    def interrupted(signum, frame): raise InterruptedError('Supervisor signal '+str(signum))
    previous = {s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        for mode in MODES:
            require(now() < DEADLINE, 'Compute cutoff reached before next timing wave')
            require(not other_research_processes(ancestor_pids()) and not B.gpu_processes(), 'Existing research/GPU process; no duplicate timing launch')
            E.verify_bindings(bindings)
            children = []
            for job, child_args, argv in specs:
                if job['mode'] != mode: continue
                child = spawn(output, job, child_args, argv, B); children.append(child)
                entries[job['id']].update(state='running', pid=child['process'].pid)
                capture_identity(child, B); snapshot('running')
            last_inventory = 0.
            while any(c['process'].returncode is None for c in children):
                require(now() < DEADLINE, 'Compute cutoff during timing wave')
                for child in children:
                    external = reap(child)
                    if external:
                        record(child, external)
                        require(external['exit_code'] == 0, 'Timing child failed: '+child['job']['id'])
                        require(external['elapsed_seconds'] <= root['outer_seconds'][mode],
                                'Observed child completion exceeds whole-child timeout: '+child['job']['id'])
                    if child['process'].returncode is None:
                        require(time.perf_counter()-child['started'] <= root['outer_seconds'][mode], 'Whole-child timeout: '+child['job']['id'])
                if time.perf_counter()-last_inventory >= 2:
                    visible = B.gpu_processes(); known = {c['process'].pid:c for c in children}
                    require(all(v['pid'] in known and B.canonical_gpu_uuid(v['gpu_uuid']) == B.canonical_gpu_uuid(root['gpu_uuids'][known[v['pid']]['job']['gpu']]) for v in visible),
                            'External GPU process or wrong physical assignment contaminated timing')
                    observations.append({'mode':mode, 'utc':now().isoformat(), 'gpu_processes':visible})
                    write(output/'gpu_observations.json', observations); last_inventory = time.perf_counter()
                time.sleep(POLL)
            # Guard failures exit zero and retain complete planned attempts. Do not
            # infer successful full-H timing from the exit code alone.
            for child in children:
                status = read(child['args'].output_dir/'status.json')
                require(status.get('state') in ('complete','complete_with_guard_failures')
                        and status.get('all_inputs_reverified') is True, 'Child implementation/input verification failed')
                entries[child['job']['id']]['evaluator_state'] = status['state']
        E.verify_bindings(bindings)
    except BaseException as error:
        failure = {'type':type(error).__name__, 'error':str(error), 'all_prior_outputs_retained':True}
        write(output/'failed_attempt.json', failure)
    finally:
        # Suppress additional SIGINT/TERM during bounded owned cleanup only.
        for sig in previous: signal.signal(sig, signal.SIG_IGN)
        unclosed = cleanup(children, B, record)
        for sig, handler in previous.items(): signal.signal(sig, handler)
        try:
            for entry in entries.values():
                directory = Path(entry['directory'])
                if directory.exists() and entry['external'] is not None:
                    entry['files_sha256'] = {str(path.resolve()):sha(path) for path in sorted(directory.rglob('*')) if path.is_file()}
                elif directory.exists(): entry['artifact_inventory_status'] = 'owned_process_unreaped_no_final_hash_claim'
                if entry['state'] == 'not_started': entry['unattempted_reason'] = failure or {'error':'prior wave did not complete'}
            E.verify_bindings(bindings)
        except BaseException as error:
            terminal = {'type':type(error).__name__,'error':str(error),'phase':'stopped_artifact_inventory_and_terminal_binding_check'}
            failure = {'terminal_error':terminal,'prior_failure':failure,'all_prior_outputs_retained':True}
            write(output/'failed_attempt.json',failure)
        all_stopped = all(e['external'] is not None and e['external']['stopped_and_reaped'] for e in entries.values()) and not unclosed
        inventory = {'schema':'adaptgns_goop3d_capacity_timing_inventory_v1', 'issued_by':'root',
            'status':'all_required_processes_stopped' if all_stopped and failure is None else 'failed_or_incomplete_timing_collection',
            'all_planned_processes_stopped':all_stopped,
            'root_authorization_sha256':sha(args.release), 'supervisor_sha256':sha(__file__), 'entries':list(entries.values()),
            'failure':failure, 'unreaped_owned_children':unclosed, 'evaluation_overlap_credit':False,
            'scientific_training_admitted':False, 'elapsed_supervisor_seconds':time.perf_counter()-start}
        write(output/'timing_inventory.json', inventory)
        snapshot('complete_stopped_timing_collection' if all_stopped and failure is None else 'failed_timing_collection')
        if not unclosed: lock.unlink()
    return 0 if failure is None and not unclosed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--release', type=Path); parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA, 'status':'description_only', 'schedule':SCHEDULE,
            'scientific_training_admitted':False, 'test_access_allowed':False, 'evaluation_overlap_credit':False,
            'whole_supervisor_outer_timeout_required':True})); return 0
    require(args.release and args.output_dir and sys.platform.startswith('linux') and hasattr(os,'wait4'), 'Explicit fresh Linux root timing invocation required')
    root = read(args.release)
    E, W, B, T = modules()
    specs, bindings = preflight(args, root, E, W, B, T)
    return supervise(args, root, specs, bindings, E, B)


if __name__ == '__main__': raise SystemExit(main())
