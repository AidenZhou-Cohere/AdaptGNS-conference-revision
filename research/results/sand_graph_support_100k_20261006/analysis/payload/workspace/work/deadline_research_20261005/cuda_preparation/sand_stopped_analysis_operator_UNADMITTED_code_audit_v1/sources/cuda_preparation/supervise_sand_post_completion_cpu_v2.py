#!/usr/bin/env python3
"""One exact Sand CPU worker under a separately pinned GNU timeout.

Description only unless --execute is supplied with a root-issued operational
release. No model, array, benchmark, or frozen scientific module is imported.
The original tool exit and timeout/supervisor/worker native closure still require
root verification; this program cannot certify its own later disappearance.
"""
import time
ENTRY_MONOTONIC = time.monotonic()
import ctypes
import os
import signal
import sys


class SigEvent(ctypes.Structure):
    # Linux glibc LP64: sigval_t8, signo4, notify4, union padding48.
    _fields_=[('value',ctypes.c_void_p),('signo',ctypes.c_int),('notify',ctypes.c_int),('padding',ctypes.c_byte*48)]
class TimeSpec(ctypes.Structure):
    _fields_=[('seconds',ctypes.c_long),('nanoseconds',ctypes.c_long)]
class ITimerSpec(ctypes.Structure):
    _fields_=[('interval',TimeSpec),('value',TimeSpec)]


def arm_absolute_guard(deadline_ns, libc=None):
    """Kernel SIGKILL timer, independent of Python/worker SIGALRM handlers.

    Never delete this timer: the kernel removes it only when this owner exits.
    Literal timing is checked against the hash-bound release after arming.
    """
    if not (sys.platform.startswith('linux') and ctypes.sizeof(ctypes.c_long)==8
            and ctypes.sizeof(ctypes.c_void_p)==8 and ctypes.sizeof(SigEvent)==64
            and ctypes.sizeof(TimeSpec)==16 and ctypes.sizeof(ITimerSpec)==32):
        raise ValueError('Linux LP64 native timer ABI required')
    if not (type(deadline_ns) is int and 0<deadline_ns<2**63):
        raise ValueError('Positive finite signed64-bit absolute deadline required')
    current=time.monotonic_ns()
    if not 0<deadline_ns-current<=3600*10**9:
        raise ValueError('Original absolute deadline expired or exceeds one hour')
    lib=libc if libc is not None else ctypes.CDLL(None,use_errno=True)
    lib.timer_create.argtypes=[ctypes.c_int,ctypes.POINTER(SigEvent),ctypes.POINTER(ctypes.c_void_p)]
    lib.timer_create.restype=ctypes.c_int
    lib.timer_settime.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ITimerSpec),ctypes.c_void_p]
    lib.timer_settime.restype=ctypes.c_int
    event=SigEvent();event.signo=signal.SIGKILL;event.notify=0  # SIGEV_SIGNAL
    timer=ctypes.c_void_p()
    if lib.timer_create(time.CLOCK_MONOTONIC,ctypes.byref(event),ctypes.byref(timer))!=0:
        raise OSError(ctypes.get_errno(),'timer_create failed')
    when=ITimerSpec();when.value.seconds,when.value.nanoseconds=divmod(deadline_ns,10**9)
    if lib.timer_settime(timer,1,ctypes.byref(when),None)!=0:  # TIMER_ABSTIME
        raise OSError(ctypes.get_errno(),'absolute timer_settime failed')
    return dict(clock='CLOCK_MONOTONIC',signal='SIGKILL',absolute_deadline_ns=deadline_ns,
                armed_monotonic_ns=time.monotonic_ns(),timer_id=timer.value or 0,deleted_before_exit=False)


NATIVE_GUARD=None
if __name__=='__main__' and '--execute' in sys.argv:
    # No release, scientific source or output file has been opened at this point.
    if sys.argv.count('--hard-deadline-monotonic-ns')!=1:
        raise ValueError('One literal absolute native deadline required')
    index=sys.argv.index('--hard-deadline-monotonic-ns')
    NATIVE_GUARD=arm_absolute_guard(int(sys.argv[index+1]))
    if sys.argv.count('--outer-computed-monotonic-ns')!=1:
        raise ValueError('Literal original bootstrap computation time required')
    computed=int(sys.argv[sys.argv.index('--outer-computed-monotonic-ns')+1])
    if not 0<=NATIVE_GUARD['armed_monotonic_ns']-computed<=5*10**9:
        raise ValueError('Explicit5-second bootstrap allowance exceeded; guard remains armed')

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess

SCHEMA = 'adaptgns_sand_post_completion_cpu_operational_release_v2'
REMOTE = '/root/repos/AdaptGNS-cuda-20261006'
PREP = REMOTE + '/cuda_preparation'
PYTHON = REMOTE + '/.venv/bin/python'
PYTHON_REAL = '/usr/bin/python3.12'
PYTHON_SHA = '6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a'
VENV_SHA = 'b7d360e62970717794ce0ab4f1e1398a49a8a73a138d34ca920b1eb251fe2f3b'
COMMANDS_SHA = '9d05860204929791306485ef99e1b4e1fb72423dbad631e025de9252e5ce2e13'
GLOBAL_STOP = datetime(2026, 10, 7, 4, tzinfo=timezone.utc)
RECOVERY_SHA = '194437c62c992a061e2b3e4f3f7d12d487d7edcca2a2c701e185d3d05d3bd9ca'
COHORT = 'sand_graph_support_scoped_100k_20261006_v1'
QUOTAS = {'inventory_A': 900, 'inventory_B': 900, 'build_cohort': 300}
SOURCES = {
    'prepare_sand_recovered_final_cohort_v1.py':'41bef815edf151fddd9ef9d667a339b5cb0f46ae405d8e4ed6bf00df69e75ced',
    'prepare_sand_final_cohort_scoped_v1.py':'ebad4e30edbdb9f444ed07dabf50b31540ecf7f85b134ee059c150d1ff68ca5f',
    'audit_sand_runtime_migration_endpoints_v1.py':'7fc6a6000ecb00168beeb00c695c7eb8feb25db28c6436afd7a3aceb563ea7db',
    'supervise_sand_runtime_migration_recovery_v2.py':'58625b827484b46b9c3d852bc4d71d16e48766b1c416050cc8c3dc5640bd21fe',
    'train_sand_runtime_migration_recovery_v1.py':'0faf1bb8ff25ac1487506781dba9722361682c725eeb7915874c90f3b66499e6',
    'train_sand_graph_support_cuda.py':'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124',
    'benchmark_sand_graph_support_rollout.py':'8586025552ee632f4dd28419f52e1f6bfeab79cc043068ef14abffa597a70e13',
    'evaluate_sand_graph_support_final.py':'952c65d6d5d41be1383dc85ad499d68e8ee1846bf05b739066f8d11b17626d58',
    'sand_graph_support_100k_protocol_v1.md':'e003abe1673018bbfa6eaffcf151ca3e46f1f7a4da8ca47ce887b71ec01c6e0d',
    'sand_train_admission.json':'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73'}


def require(ok, message):
    if not ok: raise ValueError(message)


def utc(): return datetime.now(timezone.utc)
def digest(v): return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
def stamp(v):
    t = datetime.fromisoformat(v.replace('Z', '+00:00'))
    require(t.tzinfo is not None, 'Timezone required')
    return t.astimezone(timezone.utc)
def encoded(v): return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def canonical(v):
    p = Path(v)
    require(p.is_absolute() and str(p.resolve()) == str(p), 'Canonical absolute path required')
    return p


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            check(); block = f.read(1 << 20); check()
            if not block: return h.hexdigest()
            h.update(block)


def read_json(binding, check=lambda: None):
    p = canonical(binding['path'])
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= 16*2**20, 'Bound ordinary JSON <=16 MiB required')
    check(); raw = p.read_bytes(); check()
    require(digest(binding['sha256']) and hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Control bytes changed')
    return json.loads(raw)


def publish(path, value, check, verify_after_encode=lambda: None):
    check(); raw = encoded(value); check()
    verify_after_encode(); check()
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    check()
    return hashlib.sha256(raw).hexdigest()


def native(pid):
    root = Path('/proc')/str(pid)
    try:
        a = (root/'stat').read_text().rsplit(') ', 1)[1].split()
        argv = [p.decode(errors='surrogateescape') for p in (root/'cmdline').read_bytes().split(b'\0') if p]
        try: exe = os.readlink(root/'exe')
        except FileNotFoundError: exe = None
        b = (root/'stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError): return None
    require(a[19] == b[19] and a[1:4] == b[1:4], 'Native process identity changed during read')
    return dict(pid=pid, ppid=int(b[1]), pgid=int(b[2]), sid=int(b[3]), start_id=b[19],
                state=b[0], argv=argv, executable=exe)


def stable(row): return {k: row[k] for k in ('pid', 'ppid', 'pgid', 'sid', 'start_id', 'argv', 'executable')}


def parent_death_hook(parent_pid):
    # Popen is single threaded. Setting PDEATHSIG before the second parent check
    # closes the parent-exit race; exec retains the setting for this interpreter.
    def hook():
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0: os._exit(126)
        if os.getppid() != parent_pid: os._exit(126)
    return hook


class Runtime:
    mono = staticmethod(time.monotonic)
    now = staticmethod(utc)
    sleep = staticmethod(time.sleep)
    identity = staticmethod(native)
    def boot(self): return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def bootstrap_start_bounds(self,row):
        # /proc start ticks use BOOTTIME. Measure both sides of MONOTONIC to
        # bound the offset rather than assuming the two domains are equal.
        boot_before=time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        mono=time.monotonic_ns()
        boot_after=time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        hz=os.sysconf('SC_CLK_TCK');ticks=int(row['start_id'])
        lower=ticks*10**9//hz-(boot_after-mono)
        upper=((ticks+1)*10**9+hz-1)//hz-(boot_before-mono)
        return dict(lower_ns=lower,upper_ns=upper,ticks=ticks,ticks_per_second=hz,
                    offset_lower_ns=boot_before-mono,offset_upper_ns=boot_after-mono)
    def open_pidfd(self, pid): return os.pidfd_open(pid, 0)
    def close_pidfd(self, fd): os.close(fd)
    def launch(self, argv, stdout, stderr):
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''; env['PYTHONDONTWRITEBYTECODE'] = '1'
        env.update({k:'2' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            env=env, close_fds=True, start_new_session=False, preexec_fn=parent_death_hook(os.getpid()))
    def reap(self, c):
        if c['reaped']: return True
        pid, status, usage = os.wait4(c['pid'], os.WNOHANG)
        if not pid: return False
        require(pid == c['pid'], 'Wrong child reaped')
        code = os.waitstatus_to_exitcode(status); c['process'].returncode = code
        c.update(reaped=True, exit_code=code, wait_status=status, reaped_utc=self.now().isoformat(),
                 reaped_monotonic=self.mono(), usage=dict(user_seconds=usage.ru_utime,
                 system_seconds=usage.ru_stime, peak_rss_kib=usage.ru_maxrss))
        return True
    def send(self, c, sig): signal.pidfd_send_signal(c['pidfd'], sig, None, 0)


class Budget:
    def __init__(self, r, rt, entry):
        self.rt, self.release, self.abort = rt, r, None
        clock = r['clock_sample']; sample = stamp(clock['host_utc'])
        require(clock['host_boot_id'] == rt.boot(), 'Host boot changed')
        m = clock['host_monotonic_seconds']; age = rt.mono()-m
        require(type(m) in (int,float) and math.isfinite(m) and 0 <= age <= 300
                and abs((rt.now()-sample).total_seconds()-age) <= 5, 'Original clock is stale/inconsistent')
        require(clock['source'] == 'root_fresh_tool_and_host_clock_evidence' and clock['error_bound_seconds'] == 5
                and abs((sample-stamp(clock['root_reference_utc'])).total_seconds()) <= 5, 'Clock uncertainty not bound')
        require(stamp(r['invocation_origin_utc']) == sample, 'Per-invocation root anchor differs')
        quota = r['whole_invocation_seconds']
        self.ends = {k:min(entry+quota, m+(stamp(r[k])-sample).total_seconds()-5) for k in
                     ('work_stop_utc', 'cleanup_deadline_utc', 'publication_deadline_utc')}
        self.check()
    def check(self, key='work_stop_utc', ignore_abort=False):
        require(ignore_abort or self.abort is None, self.abort or 'Cancelled')
        require(self.rt.mono() < self.ends[key] and self.rt.now()+timedelta(seconds=5) < stamp(self.release[key]),
                'Reached '+key)
    def cap_for_actual_outer(self,earliest_kill_ns):
        end=earliest_kill_ns/10**9
        for key,tail in (('work_stop_utc',15),('cleanup_deadline_utc',5),('publication_deadline_utc',0)):
            self.ends[key]=min(self.ends[key],end-tail)
        self.check()


def structure(r):
    require(sys.platform.startswith('linux'), 'Linux only')
    require(r.get('schema') == SCHEMA and r.get('issued_by') == 'root'
            and r.get('status') == 'approved_one_bounded_cpu_invocation', 'Unadmitted operational candidate')
    op = r.get('operation'); require(op in QUOTAS and r.get('whole_invocation_seconds') == QUOTAS[op], 'Exact900/300 quota required')
    require(r.get('hostname') == socket.gethostname() and r.get('new_or_restarted_clock_granted') is False
            and r.get('automatic_retry') is False and r.get('cleanup_and_publication_seconds') == 15, 'Host/tail/retry mismatch')
    origin = stamp(r['invocation_origin_utc']); end = stamp(r['publication_deadline_utc'])
    require((end-origin).total_seconds() == QUOTAS[op]
            and stamp(r['cleanup_deadline_utc']) == end-timedelta(seconds=5)
            and stamp(r['work_stop_utc']) == end-timedelta(seconds=15), 'Fixed included15s tail required')
    require(stamp(r['issued_utc']) >= origin and (stamp(r['issued_utc'])-origin).total_seconds() <= 300,
            'Fresh original issuance required')


def process_check(value, role):
    require(value.get('issued_by') == 'root' and value.get('all_children_reaped') is True
            and value.get('matching_training_processes') == [], 'Root completed process receipt required')
    if role == 'A':
        require(value.get('schema') == 'adaptgns_sand_recovered_A_completed_process_check_v1'
                and value.get('status') == 'verified_completed_recovered_A' and value.get('owner_exited') is True
                and value.get('scoped_gpu_inventory') == [] and value.get('owner_release_sha256') == RECOVERY_SHA,
                'Recovered A closure differs')
    else:
        require(value.get('schema') == 'adaptgns_sand_completed_host_process_check_v1' and value.get('host_role') == 'B'
                and value.get('cohort_id') == COHORT
                and value.get('supervisor_exited') is True, 'Original B closure differs')
    return stamp(value['checked_utc'])


def verify(r, path, pin, budget):
    structure(r); budget.check()
    require(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,'Owner must use -I -S -B')
    hard_ns=r['hard_deadline_monotonic_ns'];computed_ns=r['_outer_computed_monotonic_ns'];seconds=r['_outer_term_seconds']
    require(type(hard_ns) is int and hard_ns==int(budget.ends['publication_deadline_utc']*10**9)
            and r['bootstrap_allowance_seconds']==5 and r['_native_guard'] is not None
            and r['_native_guard']['absolute_deadline_ns']==hard_ns
            and 0<=r['_native_guard']['armed_monotonic_ns']-computed_ns<=5*10**9,
            'Literal native deadline/bootstrap must bind original root clock')
    require(type(seconds) is int and seconds>0 and type(computed_ns) is int
            and seconds==(hard_ns-computed_ns)//10**9-15-5
            and computed_ns+(seconds+15+5)*10**9<=hard_ns,'Outer duration must use actual original remaining time')
    timeout=r['outer_timeout']
    expected_outer=[timeout['path'],'--signal=TERM','--kill-after=15s',str(seconds)+'s',PYTHON,'-I','-S','-B',
        r['owner_source']['path'],'--execute','--hard-deadline-monotonic-ns',str(hard_ns),
        '--outer-term-seconds',str(seconds),'--outer-computed-monotonic-ns',str(computed_ns),
        '--release',str(path),'--release-sha256',pin]
    parent=budget.rt.identity(os.getppid());own=budget.rt.identity(os.getpid())
    require(parent is not None and parent['argv']==expected_outer and parent['executable']==str(Path(timeout['path']).resolve())
            and own is not None and own['pgid']==parent['pgid'] and own['sid']==parent['sid'], 'Exact native derived-duration outer timeout required')
    started=budget.rt.bootstrap_start_bounds(parent)
    require(started['lower_ns']<=computed_ns and started['upper_ns']<=r['_native_guard']['armed_monotonic_ns']
            and computed_ns<=r['_native_guard']['armed_monotonic_ns'], 'Native bootstrap clock-domain identity differs')
    # exec preserves the bootstrap PID/start ticks. These are NOT falsely
    # labelled as the moment GNU timeout arms its timer. Actual exec lies after
    # the literal calculation and before owner guard arming, within the5s bound.
    r['_native_outer_timing']=dict(bootstrap_process_start=started,computed_monotonic_ns=computed_ns,
        owner_guard_armed_monotonic_ns=r['_native_guard']['armed_monotonic_ns'],term_seconds=seconds,
        earliest_relative_kill_monotonic_ns=computed_ns+(seconds+15)*10**9,
        original_absolute_hard_deadline_ns=hard_ns,bootstrap_allowance_seconds=5)
    budget.cap_for_actual_outer(computed_ns+(seconds+15)*10**9)
    commands = read_json(r['reviewed_commands'], budget.check)
    require(r['reviewed_commands']['sha256'] == COMMANDS_SHA and r['command'] == commands[r['operation']]['argv']
            and r['hostname'] == commands[r['operation']]['host'], 'Reviewed exact worker argv/host differs')
    require(sys.executable == PYTHON and str(Path(sys.executable).resolve()) == PYTHON_REAL
            and r['owner_source']['path'] == PREP+'/supervise_sand_post_completion_cpu_v2.py'
            and str(Path(__file__).resolve()) == r['owner_source']['path'], 'Exact interpreter/owner path required')
    phase = read_json(r['verification_phase'], budget.check)
    require(phase.get('schema') == 'adaptgns_sand_original_posttraining_phase_v1' and phase.get('issued_by') == 'root'
            and phase.get('status') == 'approved_original_phase' and phase.get('original_verification_seconds') == 2700
            and phase.get('original_shared_source_seconds') == 900 and phase.get('original_evaluation_seconds') == 11760
            and phase.get('original_analysis_seconds') == 3600 and phase.get('global_analysis_deadline_utc') == GLOBAL_STOP.isoformat()
            and phase.get('new_or_restarted_clock_granted') is False, 'Original shared phase required')
    start, end = stamp(phase['verification_origin_utc']), stamp(phase['verification_deadline_utc'])
    require((end-start).total_seconds() == 2700 and start <= stamp(r['invocation_origin_utc'])
            and stamp(r['publication_deadline_utc']) <= end
            and end+timedelta(seconds=900+11760+3600) <= GLOBAL_STOP, 'Original verification/downstream reserves exceeded')
    closure = read_json(r['original_closure'], budget.check)
    require(closure.get('issued_by') == 'root' and closure.get('status') == 'original_exit_and_native_absence_verified'
            and closure.get('complete_registered_child_ledger_checked') is True and closure.get('gpu_processes') == []
            and type(closure.get('original_exit_code')) is int and closure.get('original_exit_code') == 0
            and closure.get('native_absent') and all(v is True for v in closure['native_absent'].values()),
            'Root original tool/native closure is required')
    role = 'B' if r['operation'] == 'inventory_B' else 'A'
    require(closure.get('host_role') == role and closure.get('hostname') == r['hostname']
            and (role != 'A' or closure.get('original_tool_session') == 63707), 'Wrong original closure')
    require(stamp(closure['checked_utc']) <= stamp(r['issued_utc']), 'Issue predates closure')
    flags = dict(zip(r['command'][4::2], r['command'][5::2]))
    controls = r['inputs_sha256']; require(isinstance(controls, dict) and all(digest(v) for v in controls.values()), 'Exact input hashes required')
    if r['operation'].startswith('inventory_'):
        expected_inputs = {flags['--process-check']}
        if role == 'A': expected_inputs.add(flags['--recovery-release'])
        require(set(controls) == expected_inputs, 'Exact inventory control set required')
        checked = [process_check(read_json(dict(path=flags['--process-check'], sha256=controls[flags['--process-check']]), budget.check), role)]
        require(closure.get('process_check_sha256') == controls[flags['--process-check']], 'Native/core process receipts not bound')
        if role == 'A': require(controls[flags['--recovery-release']] == RECOVERY_SHA, 'Original recovery release differs')
        output_paths = [flags['--output']]; output_root = None
    else:
        require(set(controls) == {flags[k] for k in ('--inventory-a','--inventory-b','--root-release')}, 'Exact bridge controls required')
        a, b, release = [read_json(dict(path=flags[k], sha256=controls[flags[k]]), budget.check)
                         for k in ('--inventory-a','--inventory-b','--root-release')]
        checked = [process_check(a['receipts']['process'], 'A'), process_check(b['process_check'], 'B')]
        require(release.get('issued_by') == 'root' and release.get('status') == 'approved_complete_recovery_cohort_bridge'
                and release.get('inventory_sha256') == {'A':controls[flags['--inventory-a']], 'B':controls[flags['--inventory-b']]},
                'Exact separately admitted all-six bridge required')
        output_root = flags['--output-dir']; output_paths = [output_root+'/'+n for n in ('root_release.json','cohort_audit.json','cohort.json')]
    if r['operation'].startswith('inventory_'):
        require(all(0 <= (budget.rt.now()-t).total_seconds() <= 1800
                    and stamp(r['publication_deadline_utc'])+timedelta(seconds=5) <= t+timedelta(seconds=1800) for t in checked),
                'Original native checks must remain fresh through inventory publication')
    expected = {str(path):pin, **controls, **{PREP+'/'+name:value for name,value in SOURCES.items()},
                PYTHON:PYTHON_SHA, PYTHON_REAL:PYTHON_SHA, REMOTE+'/.venv/pyvenv.cfg':VENV_SHA}
    for key in ('reviewed_commands','verification_phase','original_closure','owner_source','bootstrap_source'):
        expected[r[key]['path']] = r[key]['sha256']
    require(r['bootstrap_source']['path']==PREP+'/launch_sand_post_completion_cpu_v2.py','Exact isolated bootstrap required')
    require(timeout['path'] == '/usr/bin/timeout' and digest(timeout['sha256']), 'Pinned GNU timeout required')
    expected[timeout['path']] = timeout['sha256']
    require(timeout['duration_rule']=='floor((original_hard_deadline_ns-actual_bootstrap_monotonic_ns)/1e9)-15-5'
            and 'argv' not in timeout,'Static885/285 timeout argv is not admitted in v2')
    out = canonical(r['owner_output_dir']); protected = [canonical(p) for p in expected if p != PYTHON]
    protected += [canonical(flags['--host-root'])] if '--host-root' in flags else []
    all_outputs = [out]+[canonical(output_root)] if output_root else [out]+[canonical(p) for p in output_paths]
    require(all(a != b and a not in b.parents and b not in a.parents for i,a in enumerate(all_outputs) for b in all_outputs[i+1:]), 'Output overlap')
    require(all(o != p and o not in p.parents and p not in o.parents for o in all_outputs for p in protected), 'Protected path/output overlap')
    require(all(not p.exists() for p in all_outputs), 'Outputs must be fresh')
    for p, value in expected.items(): require(sha(p,budget.check) == value, 'Bound source/control bytes changed: '+p)
    return expected, output_paths, output_root, parent, own


def public(c): return {k:v for k,v in c.items() if k not in ('process','pidfd')}
def register(process, rt, command, owner):
    return dict(process=process,pid=process.pid,pidfd=None,identity=None,registered=True,reaped=False,
                command=command,owner_identity=owner,signals=[],cleanup_errors=[],exit_code=None,
                started_utc=rt.now().isoformat(),started_monotonic=rt.mono())
def capture(c, rt):
    # Never poll or reap before acquiring the fd: it is the cleanup capability
    # even when the subsequent argv/native capture or evidence publication fails.
    c['pidfd'] = rt.open_pidfd(c['pid'])
    row = rt.identity(c['pid']); own = c['owner_identity']
    require(row is not None and row['state'] != 'Z' and row['ppid'] == own['pid']
            and row['pgid'] == own['pgid'] and row['sid'] == own['sid']
            and row['argv'] == c['command'] and row['executable'] == PYTHON_REAL, 'Exact owned worker identity unavailable')
    c['identity'] = row
def send(c, sig, rt):
    if rt.reap(c): return False
    require(c['pidfd'] is not None, 'No owned pidfd; outer timeout/PDEATHSIG remains required')
    row = rt.identity(c['pid'])
    if c['identity'] is not None and row is not None and row['state'] != 'Z':
        require(stable(row) == stable(c['identity']), 'Owned identity changed; refusing signal')
    rt.send(c,sig); c['signals'].append(dict(signal=signal.Signals(sig).name,utc=rt.now().isoformat(),monotonic=rt.mono()))
    return True
def cleanup(c, rt, budget):
    if c is None or c['reaped']: return
    start = rt.mono(); end = min(start+10,budget.ends['cleanup_deadline_utc']); attempted=set()
    while rt.mono() < end:
        if rt.reap(c): return
        for delay,sig in ((0,signal.SIGINT),(3,signal.SIGTERM),(6,signal.SIGKILL)):
            if rt.mono() >= start+delay and sig not in attempted:
                attempted.add(sig)
                try: send(c,sig,rt)
                except BaseException as e: c['cleanup_errors'].append(type(e).__name__+': '+str(e))
        rt.sleep(min(.05,max(0,end-rt.mono())))
    rt.reap(c)


def output_hashes(paths, root, check):
    if root is not None:
        p=canonical(root); check()
        require(p.is_dir() and not p.is_symlink() and sorted(str(x) for x in p.iterdir()) == sorted(paths), 'Partial/failed/unexpected cohort publication')
    result={}
    for path in paths:
        check(); p=canonical(path); require(p.is_file() and not p.is_symlink(), 'Output is missing/not ordinary')
        result[path]=sha(p,check)
    return result


def execute(r, path, pin, rt, budget):
    expected, outputs, output_root, outer, own = verify(r,path,pin,budget)
    require(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, 'Default SIGCHLD/no automatic reaper required')
    directory=Path(r['owner_output_dir']); budget.check(); directory.mkdir(mode=0o700,exist_ok=False)
    child=None; failure=None; evidence={}; result={}
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def stop(sig,frame): budget.abort=budget.abort or signal.Signals(sig).name
    for s in previous: signal.signal(s,stop)
    final_check=lambda:budget.check('publication_deadline_utc',True)
    try:
        evidence[str(directory/'owner_started.json')]=publish(directory/'owner_started.json',dict(
            owner_identity=own,outer_timeout_identity=outer,release_sha256=pin,entry_monotonic=ENTRY_MONOTONIC,
            original_clock_sample=r['clock_sample'],native_hard_guard=r.get('_native_guard'),
            native_outer_timing=r.get('_native_outer_timing'),scientific_admission=False),budget.check)
        with (directory/'child.stdout').open('xb') as out,(directory/'child.stderr').open('xb') as err:
            budget.check(); child=register(rt.launch(r['command'],out,err),rt,r['command'],own)
            capture(child,rt)
            evidence[str(directory/'child_registered.json')]=publish(directory/'child_registered.json',public(child),budget.check)
            while not rt.reap(child): budget.check();rt.sleep(.05)
            require(child['exit_code']==0 and child['signals']==[], 'Worker failed/interrupted')
        result=output_hashes(outputs,output_root,budget.check)
        for p,value in expected.items(): require(sha(p,budget.check)==value,'Inputs changed after worker: '+p)
    except BaseException as e: failure=dict(type=type(e).__name__,error=str(e),utc=rt.now().isoformat())
    finally:
        try: cleanup(child,rt,budget)
        except BaseException as e: failure=failure or dict(type=type(e).__name__,error=str(e))
        if child is not None and child['pidfd'] is not None:
            try: rt.close_pidfd(child['pidfd']);child['pidfd']=None
            except BaseException as e: failure=failure or dict(type=type(e).__name__,error=str(e))
    try:
        final_check()
        absent=child is None or child['reaped'] and rt.identity(child['pid']) is None
        require(absent,'Worker native closure incomplete')
        for name in ('child.stdout','child.stderr'):
            p=directory/name
            if p.exists(): evidence[str(p)]=sha(p,final_check)
        for p,value in {**expected,**evidence}.items(): require(sha(p,final_check)==value,'Final control/evidence bytes changed: '+p)
        complete=failure is None and child is not None and child['identity'] is not None and child['exit_code']==0 and not child['signals']
        if complete: require(output_hashes(outputs,output_root,final_check)==result,'Final output bytes changed')
        def publication_bindings():
            # Encode first, then verify every terminal binding immediately before
            # the exclusive write. Large JSON serialization cannot hide mutation.
            final_check()
            for p,value in {**expected,**evidence}.items():
                require(sha(p,final_check)==value,'Publication control/evidence bytes changed: '+p)
            if complete: require(output_hashes(outputs,output_root,final_check)==result,'Publication output bytes changed')
            require(child is None or child['reaped'] and rt.identity(child['pid']) is None,'Publication native closure changed')
            final_check()
        # All hashes and closure are checked inside the same original envelope.
        publish(directory/'owner_terminal.json',dict(schema='adaptgns_sand_post_completion_cpu_terminal_v2',
            status='complete' if complete else 'failed',failure=failure,release_sha256=pin,
            child=None if child is None else public(child),child_native_absent=absent,
            input_sha256=expected,evidence_sha256=evidence,output_sha256=result,
            publication_utc=rt.now().isoformat(),publication_monotonic=rt.mono(),
            outer_timeout_identity=outer,owner_identity=own,scientific_admission=False,
            native_hard_guard=r.get('_native_guard'),native_outer_timing=r.get('_native_outer_timing'),
            root_original_tool_exit_and_timeout_owner_child_native_closure_required=True),final_check,publication_bindings)
        return 0 if complete else 1
    except BaseException as e:
        # Never write after the deadline merely to make failure evidence look
        # complete. The raw original timeout/tool exit remains authoritative.
        try: publish(directory/'publication_failed.json',dict(error=type(e).__name__+': '+str(e),failure=failure,
            child=None if child is None else public(child),all_outputs_preserved=True),final_check)
        except BaseException: pass
        return 1
    finally:
        for s,handler in previous.items(): signal.signal(s,handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--execute',action='store_true');p.add_argument('--release',type=Path);p.add_argument('--release-sha256')
    p.add_argument('--hard-deadline-monotonic-ns',type=int);p.add_argument('--outer-term-seconds',type=int)
    p.add_argument('--outer-computed-monotonic-ns',type=int)
    a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status='description_only',execution_authority=False,operations=QUOTAS,
            original_shared_verification_seconds=2700,no_model_or_array_imports=True)));return 0
    require(sys.flags.optimize==0 and a.release is not None and digest(a.release_sha256),'Exact nonoptimized release invocation required')
    path=canonical(a.release);r=read_json(dict(path=str(path),sha256=a.release_sha256));structure(r)
    require(NATIVE_GUARD is not None and a.hard_deadline_monotonic_ns==r['hard_deadline_monotonic_ns'],
            'Previously armed literal native guard required')
    r['_native_guard']=NATIVE_GUARD;r['_outer_term_seconds']=a.outer_term_seconds
    r['_outer_computed_monotonic_ns']=a.outer_computed_monotonic_ns
    rt=Runtime();budget=Budget(r,rt,ENTRY_MONOTONIC)
    return execute(r,path,a.release_sha256,rt,budget)


if __name__=='__main__': raise SystemExit(main())
