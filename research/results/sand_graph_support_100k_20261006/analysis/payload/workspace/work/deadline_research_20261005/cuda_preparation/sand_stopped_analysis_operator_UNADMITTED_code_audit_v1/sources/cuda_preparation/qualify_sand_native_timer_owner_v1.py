#!/usr/bin/env python3
"""One root-run, model-free Linux timer/PDEATHSIG qualification; inert by default.

Only the new pinned v2 operational owner is imported. The controller is a
qualification-only subreaper for one timer fixture and its one sleeping child.
GNU timeout 12s +3s KILL grace bounds the original invocation to 15 seconds.
This program does not issue releases or access scientific inputs.
"""
import time
ENTRY = time.monotonic()
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

OWNER_SHA = '43708e9d5c62f8eb20ab2bb19777946485909383c0bf6df91de0f448fdf97d8a'
PYTHON = '/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python'
SCHEMA = 'sand_model_free_native_timer_pdeathsig_qualification_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def event(directory, kind, **values):
    with (directory / 'timer_fixture.events.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(kind=kind, monotonic_ns=time.monotonic_ns(), **values), sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def identity(pid):
    root = Path('/proc') / str(pid)
    try:
        first = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
        argv = [v.decode(errors='surrogateescape') for v in (root / 'cmdline').read_bytes().split(b'\0') if v]
        second = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    require(first[19] == second[19], 'Fixture PID changed while recording identity')
    return dict(pid=pid, ppid=int(second[1]), pgid=int(second[2]), sid=int(second[3]),
                state=second[0], start_ticks=int(second[19]), argv=argv)


def load_owner(path):
    require(path.is_absolute() and path.name == 'supervise_sand_post_completion_cpu_v2.py'
            and not path.is_symlink() and sha(path) == OWNER_SHA, 'Exact new operational source required')
    spec = importlib.util.spec_from_file_location('_sand_native_timer_qualification_only', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(sha(path) == OWNER_SHA, 'Operational source changed during import')
    return module


def native_sleep(module, seconds, nanoseconds=0):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.nanosleep.argtypes = [ctypes.POINTER(module.TimeSpec), ctypes.POINTER(module.TimeSpec)]
    libc.nanosleep.restype = ctypes.c_int
    request = module.TimeSpec(seconds, nanoseconds)
    require(libc.nanosleep(ctypes.byref(request), None) == 0, 'Native sleep unexpectedly interrupted')


def fixture_argv(args, role):
    return [PYTHON, '-I', '-S', '-B', str(Path(__file__).resolve()), '--execute',
            '--owner-source', str(args.owner_source), '--output-dir', str(args.output_dir), '--fixture', role]


def worker_fixture(args, module):
    libc = ctypes.CDLL(None, use_errno=True)
    parent_signal = ctypes.c_int()
    require(libc.prctl(2, ctypes.byref(parent_signal), 0, 0, 0) == 0 and parent_signal.value == signal.SIGKILL,
            'Actual PR_GET_PDEATHSIG did not report SIGKILL')
    write_new(args.output_dir / 'sleeping_fixture.ready.json', dict(
        identity=identity(os.getpid()), parent_death_signal=parent_signal.value,
        native_sleep_seconds=10, native_sleep_entered_monotonic_ns=time.monotonic_ns()))
    native_sleep(module, 10)
    raise RuntimeError('Sleeping fixture survived parent death')


def timer_fixture(args, module):
    # Cancellation exists only in this qualification; production never disarms.
    cancelled_deadline = time.monotonic_ns() + 500_000_000
    receipt = module.arm_absolute_guard(cancelled_deadline)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.timer_delete.argtypes = [ctypes.c_void_p]
    libc.timer_delete.restype = ctypes.c_int
    require(libc.timer_delete(ctypes.c_void_p(receipt['timer_id'])) == 0, 'timer_delete qualification failed')
    event(args.output_dir, 'timer_cancelled', original_receipt=receipt)
    native_sleep(module, 0, 750_000_000)
    require(time.monotonic_ns() > cancelled_deadline, 'Cancellation survival interval too short')
    event(args.output_dir, 'cancelled_deadline_survived')
    with (args.output_dir / 'sleeping_fixture.stdout.txt').open('xb') as stdout, (args.output_dir / 'sleeping_fixture.stderr.txt').open('xb') as stderr:
        worker = subprocess.Popen(fixture_argv(args, 'worker'), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                  close_fds=True, start_new_session=False,
                                  preexec_fn=module.parent_death_hook(os.getpid()))
        event(args.output_dir, 'sleeping_fixture_spawned', pid=worker.pid, identity=identity(worker.pid))
        ready_path = args.output_dir / 'sleeping_fixture.ready.json'
        ready_limit = time.monotonic() + 1.5
        while not ready_path.exists() and time.monotonic() < ready_limit:
            time.sleep(0.01)
        require(ready_path.exists(), 'Sleeping fixture did not reach native sleep')
        # The file is published once and may be briefly visible before close.
        ready = None
        while time.monotonic() < ready_limit:
            try:
                ready = json.loads(ready_path.read_bytes())
                break
            except json.JSONDecodeError:
                time.sleep(0.01)
        require(ready and ready['identity']['pid'] == worker.pid and ready['identity']['ppid'] == os.getpid(),
                'Sleeping fixture ownership differs')
        deadline = time.monotonic_ns() + 2_000_000_000
        armed = module.arm_absolute_guard(deadline)
        event(args.output_dir, 'absolute_timer_armed', receipt=armed, sleeping_fixture_pid=worker.pid)
        event(args.output_dir, 'native_sleep_entered', seconds=10)
        native_sleep(module, 10)
    raise RuntimeError('Timer fixture survived absolute SIGKILL deadline')


def direct_children():
    text = Path('/proc/self/task/' + str(os.getpid()) + '/children').read_text().strip()
    return [int(v) for v in text.split()] if text else []


def controller(args, module):
    require(not args.output_dir.exists(), 'Fresh model-free qualification directory required')
    parent = identity(os.getppid())
    require(parent and parent['argv'][:4] == ['/usr/bin/timeout', '--signal=TERM', '--kill-after=3s', '12s'],
            'Exact independent GNU timeout 12s plus3s required')
    args.output_dir.mkdir(mode=0o700)
    libc = ctypes.CDLL(None, use_errno=True)
    prior = ctypes.c_int()
    require(libc.prctl(37, ctypes.byref(prior), 0, 0, 0) == 0 and prior.value == 0,
            'Fresh qualification controller must not already be subreaper')
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'Qualification-only subreaper setup failed')
    own = identity(os.getpid())
    write_new(args.output_dir / 'qualification_started.json', dict(
        schema=SCHEMA, controller_identity=own, outer_timeout_identity=parent,
        owner_source=str(args.owner_source), owner_source_sha256=OWNER_SHA,
        qualification_source_sha256=sha(__file__), entry_monotonic=ENTRY,
        glibc=os.confstr('CS_GNU_LIBC_VERSION'), machine=os.uname().machine,
        qualification_only_subreaper=True, production_owner_changed=False))
    records, exits, cleanup_signals, failures = {}, {}, [], []
    process = None
    try:
        with (args.output_dir / 'timer_fixture.stdout.txt').open('xb') as stdout, (args.output_dir / 'timer_fixture.stderr.txt').open('xb') as stderr:
            process = subprocess.Popen(fixture_argv(args, 'owner'), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       close_fds=True, start_new_session=False,
                                       preexec_fn=module.parent_death_hook(os.getpid()))
            records[process.pid] = identity(process.pid)
            require(records[process.pid] is not None, 'Timer fixture identity capture failed')
            limit = ENTRY + 10.0
            while time.monotonic() < limit:
                for pid in direct_children():
                    if pid not in records:
                        records[pid] = identity(pid)
                events_path = args.output_dir / 'timer_fixture.events.jsonl'
                if events_path.exists():
                    for line in events_path.read_text().splitlines():
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if entry['kind'] == 'sleeping_fixture_spawned':
                            # Preserve the original parent identity even if the
                            # subreaper first observed this PID after adoption.
                            previous = records.get(entry['pid'])
                            if previous is not None and entry['identity'] is not None:
                                require(previous['start_ticks'] == entry['identity']['start_ticks'], 'Fixture identity changed after adoption')
                            records[entry['pid']] = entry['identity']
                while True:
                    try:
                        pid, status = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        pid = 0
                    if not pid:
                        break
                    exits[pid] = dict(wait_status=status, exit_code=os.waitstatus_to_exitcode(status),
                                      observed_monotonic_ns=time.monotonic_ns())
                if len(exits) >= 2:
                    break
                time.sleep(0.01)
            require(len(records) == len(exits) == 2 and set(records) == set(exits), 'Exactly two fixture identities/exits required')
            require(all(value is not None for value in records.values()), 'Every fixture needs a captured native identity')
            require(all(e['exit_code'] == -signal.SIGKILL for e in exits.values()), 'Both fixtures must terminate with SIGKILL')
            entries = [json.loads(line) for line in events_path.read_text().splitlines()]
            require([e['kind'] for e in entries] == ['timer_cancelled', 'cancelled_deadline_survived',
                    'sleeping_fixture_spawned', 'absolute_timer_armed', 'native_sleep_entered'], 'Exact native API progression required')
            armed = entries[3]['receipt']
            require(armed['absolute_deadline_ns'] <= exits[process.pid]['observed_monotonic_ns']
                    <= armed['absolute_deadline_ns'] + 1_500_000_000, 'Timer exit observation outside declared1.5s tolerance')
            worker_pid = entries[3]['sleeping_fixture_pid']
            require(worker_pid != process.pid and records[worker_pid]['ppid'] == process.pid,
                    'PDEATHSIG fixture did not belong to timer fixture')
            ready = json.loads((args.output_dir / 'sleeping_fixture.ready.json').read_bytes())
            require(ready['parent_death_signal'] == signal.SIGKILL and ready['identity']['pid'] == worker_pid,
                    'Native PDEATHSIG setup receipt differs')
            require(abs(exits[worker_pid]['observed_monotonic_ns'] - exits[process.pid]['observed_monotonic_ns']) <= 1_500_000_000,
                    'PDEATHSIG child exit was not promptly observed')
            require(all(identity(pid) is None for pid in records) and direct_children() == [], 'Fixture native closure gap')
            process.returncode = exits[process.pid]['exit_code']
    except BaseException as error:
        failures.append(dict(type=type(error).__name__, message=str(error)))
    finally:
        # Only direct, captured fixtures of this dedicated subreaper are signaled.
        cleanup_limit = ENTRY + 11.5
        while time.monotonic() < cleanup_limit:
            children = direct_children()
            if not children:
                break
            for pid in children:
                row = identity(pid)
                if row is None:
                    continue
                records.setdefault(pid, row)
                if row['state'] != 'Z':
                    os.kill(pid, signal.SIGKILL)
                    cleanup_signals.append(dict(pid=pid, signal='SIGKILL', identity=row))
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    pid = 0
                if not pid:
                    break
                exits[pid] = dict(wait_status=status, exit_code=os.waitstatus_to_exitcode(status),
                                  observed_monotonic_ns=time.monotonic_ns(), cleanup=True)
            time.sleep(0.01)
        closure = {str(pid): identity(pid) is None for pid in records}
        if direct_children() or not all(closure.values()) or cleanup_signals:
            failures.append(dict(type='closure_or_cleanup_failure', message='Clean original exits and complete native closure are required'))
        require(sha(args.owner_source) == OWNER_SHA, 'Operational source changed during qualification')
        files = {str(p): sha(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}
        cancelled = False
        event_path = args.output_dir / 'timer_fixture.events.jsonl'
        if event_path.exists():
            for line in event_path.read_text().splitlines():
                try:
                    cancelled = cancelled or json.loads(line).get('kind') == 'timer_cancelled'
                except json.JSONDecodeError:
                    pass
        write_new(args.output_dir / 'qualification_terminal.json', dict(
            schema=SCHEMA, status='passed_model_free_native_api_qualification' if not failures else 'failed_preserved',
            failures=failures, fixture_identities={str(k): v for k, v in records.items()}, fixture_exits={str(k): v for k, v in exits.items()},
            fixture_native_absence=closure, cleanup_signals=cleanup_signals, files_sha256=files,
            outer_timeout_identity=parent, controller_identity=own, elapsed_seconds=time.monotonic() - ENTRY,
            scientific_inputs_opened=False, scientific_execution=False, production_timer_disarmed=False,
            qualification_timer_cancelled=cancelled, root_original_tool_exit_and_outer_controller_native_closure_required=True))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--owner-source', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--fixture', choices=['owner', 'worker'])
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(schema=SCHEMA, status='inert_requires_root_review_and_execution', maximum_outer_seconds=15)))
        return 0
    require(sys.platform.startswith('linux') and os.uname().machine == 'aarch64'
            and ctypes.sizeof(ctypes.c_long) == ctypes.sizeof(ctypes.c_void_p) == 8,
            'Linux aarch64 LP64 qualification only')
    require(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode, 'Exact isolated Python flags required')
    require(args.owner_source is not None and args.output_dir is not None and args.output_dir.is_absolute(), 'Explicit source/fresh output required')
    module = load_owner(args.owner_source)
    if args.fixture == 'owner':
        return timer_fixture(args, module)
    if args.fixture == 'worker':
        return worker_fixture(args, module)
    return controller(args, module)


if __name__ == '__main__':
    raise SystemExit(main())
