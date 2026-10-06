#!/usr/bin/env python3
"""One frozen Sand source-preparation worker in the original shared900s window.

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

SCHEMA = 'adaptgns_sand_reserved_source_cpu_operational_release_v1'
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
VERIFY_PHASE_SHA='61df824a2bb23d5270f07682ba01ba3028d5019997a9cf6ba47c82ed54241914'
PREPARER_SHA='6a40fe9196157c0666ffe556eb0e92781e9fc3b84f0c45b5ff67e09be0fce1b8'
OPERATIONS={'acquire':('acquire',None),'convert':('convert',None),'census':('census',None),
    'candidate_valid':('candidate','valid'),'candidate_test':('candidate','test'),
    'preflight_valid':('preflight','valid'),'preflight_test':('preflight','test')}
COMMON_FLAGS=('--cohort','--cohort-audit','--root-release','--output-dir')
SPLIT_FLAGS=('--train-manifest','--valid-manifest','--test-manifest','--census-report',
            '--structural-report','--protocol','--train-admission')
MODE_FLAGS={'acquire':(), 'convert':('--input-dir','--acquisition-report'),
    'census':('--train-manifest','--valid-manifest','--test-manifest'),
    'candidate':SPLIT_FLAGS,'preflight':SPLIT_FLAGS+('--admission',)}
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
    'sand_train_admission.json':'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73',
    'prepare_sand_reserved_test_recovered_v1.py':PREPARER_SHA,
    'download_sand_public_v2.py':'864d9c974deabceb19b5f3b8e88ced1ea7281595a0d3a526e8d39f382aa4605a',
    'repackage_designsafe_sand.py':'37c135a9eaa484a5fdbed2176780913b71cd3be4df602983fcd3b644c1a57ff9',
    'sand_scoped_operational_amendment_v1.md':'411547c3355859906f59792b3b493841030b0afd2e105e17e466114df6743738',
    'sand_scoped_schedule_fixed_spec_v2.json':'403a96c1304341ea1899c994e0ab7474280add44ea5986ec41570e08d16c337b'}
COHORT_ADAPTER_SHA=SOURCES['prepare_sand_recovered_final_cohort_v1.py']
PROTOCOL_SHA=SOURCES['sand_graph_support_100k_protocol_v1.md']
TRAIN_ADMISSION_SHA=SOURCES['sand_train_admission.json']
TRAINER_SHA=SOURCES['train_sand_graph_support_cuda.py']
AMENDMENT_SHA=SOURCES['sand_scoped_operational_amendment_v1.md']
PLAN_SHA=SOURCES['sand_scoped_schedule_fixed_spec_v2.json']
TRAIN_MANIFEST_SHA='f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f'


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
        quota = r['remaining_shared_source_seconds']
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
            and r.get('status') == 'approved_one_bounded_source_invocation', 'Unadmitted source operational candidate')
    require(r.get('operation') in OPERATIONS and r.get('hostname') == socket.gethostname(), 'Exact source mode/host required')
    require(r.get('new_or_restarted_clock_granted') is False and r.get('automatic_retry') is False
            and r.get('cleanup_and_publication_seconds') == 15, 'Original shared source/tail/retry policy differs')
    origin=stamp(r['invocation_origin_utc']);end=stamp(r['publication_deadline_utc'])
    remaining=(end-origin).total_seconds()
    require(type(r.get('remaining_shared_source_seconds')) in (int,float) and math.isfinite(remaining)
            and 0<remaining<=900 and r['remaining_shared_source_seconds']==remaining,
            'Only the actual remaining original shared source window is available')
    require(stamp(r['cleanup_deadline_utc'])==end-timedelta(seconds=5)
            and stamp(r['work_stop_utc'])==end-timedelta(seconds=15), 'One included15-second final tail required')
    require(origin<=stamp(r['issued_utc'])<=origin+timedelta(seconds=300),'Fresh operation issuance required')


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
    return verify_source_contract(r,path,pin,budget,parent,own)


def cohort_gate(cohort,audit):
    models=cohort.get('models',[]);expected={(a,s) for a in ('base','mix') for s in range(3)}
    require(cohort.get('schema')=='adaptgns_sand_graph_support_final_cohort_v1' and cohort.get('status')=='frozen_for_final_evaluation'
        and cohort.get('issued_by')=='root' and cohort.get('dataset')=='Sand' and cohort.get('updates')==100000
        and cohort.get('protocol_sha256')==PROTOCOL_SHA and cohort.get('training_admission_sha256')==TRAIN_ADMISSION_SHA
        and cohort.get('trainer_source_sha256')==TRAINER_SHA and cohort.get('adapter_sha256')==COHORT_ADAPTER_SHA
        and cohort.get('operational_amendment_sha256')==AMENDMENT_SHA and cohort.get('schedule_plan_sha256')==PLAN_SHA,
        'Complete reviewed Sand scoped cohort required before test access')
    require(len(models)==6 and {(m.get('arm'),m.get('seed')) for m in models}==expected
        and all(type(m.get('seed')) is int and m.get('objective')=='faithful' and m.get('completed_steps')==100000 for m in models)
        and all(isinstance(m.get('checkpoint_sha256'),str) and len(m['checkpoint_sha256'])==64
            and all(c in '0123456789abcdef' for c in m['checkpoint_sha256']) for m in models)
        and len({m['checkpoint_sha256'] for m in models})==6
        and cohort.get('policies')==['base','dense','random25','speed25','laggedrisk25','relative-velocity-RMS25'],
        'All six distinct fixed100k endpoints and six declared policies required')
    require(audit.get('schema')=='adaptgns_sand_graph_support_complete_cohort_audit_v1'
        and audit.get('status')=='all_six_endpoints_and_pairing_verified' and audit.get('adapter_sha256')==COHORT_ADAPTER_SHA
        and audit.get('cohort_id')==cohort.get('cohort_id') and audit.get('trainer_source_sha256')==TRAINER_SHA
        and audit.get('issued_by')=='root' and audit.get('protocol_sha256')==PROTOCOL_SHA
        and audit.get('training_admission_sha256')==TRAIN_ADMISSION_SHA,
        'Matching complete source/tensor/pairing audit required')
    rows=audit.get('models',[]);pairs=audit.get('paired_seeds',[])
    require(len(rows)==6 and {(m.get('arm'),m.get('seed')) for m in rows}==expected,'Complete endpoint audit grid required')
    for model in models:
        ev=next(m for m in rows if (m['arm'],m['seed'])==(model['arm'],model['seed']))
        require(ev.get('checkpoint_sha256')==model['checkpoint_sha256'] and ev.get('completed_steps')==ev.get('graph_history_updates')==100000
            and ev.get('checkpoint_every')==10000 and ev.get('log_every')==100 and all(ev.get(k) is True for k in
            ('all_optimizer_steps_equal_100000','all_state_and_moments_finite','source_data_protocol_verified','checkpoint_bytes_verified')),
            'Incomplete endpoint proof')
    require(len(pairs)==3 and {p.get('seed') for p in pairs}=={0,1,2} and all(type(p.get('seed')) is int and all(p.get(k) is True for k in
        ('initial_model_tensor_identity','initial_cpu_cuda_rng_identity','all_frame_noise_lr_schedules_equal','all_graph_budgets_and_rng_material_verified')) for p in pairs),
        'Complete three-seed initialization/schedule pairing required')


def source_command(r):
    mode,split=OPERATIONS[r['operation']]
    flags=COMMON_FLAGS+MODE_FLAGS[mode];paths=r['paths']
    require(set(paths)==set(flags),'Exact source-mode path flag set required')
    for value in paths.values():canonical(value)
    command=[PYTHON,'-B',PREP+'/prepare_sand_reserved_test_recovered_v1.py','--execute','--mode',mode]
    if split is not None:command+=['--split',split]
    for flag in flags:command += [flag,paths[flag]]
    return command


def source_outputs(mode,root):
    if mode=='acquire':names=['metadata.json','test.npz','download_report.json','root_preparation_receipt.json']
    elif mode=='convert':
        names=['metadata.json','structural_report.json','test.json']
        names += ['test/'+kind+'_%06d.npy'%i for i in range(30) for kind in ('position','type')]
    else:names=[{'census':'all_split_census.json','candidate':'split_admission_candidate.json',
                 'preflight':'split_preflight.json'}[mode]]
    return [root+'/'+name for name in names]


def numeric_bindings(paths,pins,check):
    result={}
    for split,count in (('train',1000),('valid',30),('test',30)):
        path=paths['--'+split+'-manifest']
        manifest=read_json(dict(path=path,sha256=pins[path]),check)
        require(manifest.get('dataset')=='Sand' and manifest.get('split')==split
                and manifest.get('record_count')==len(manifest.get('records',[]))==count,
                'Complete original1000/30/30 manifests required')
        for index,record in enumerate(manifest['records']):
            check()
            require(type(record.get('source_index')) is int and record['source_index']==index
                    and record.get('id')==f'{split}:{index:06d}','Exact complete manifest source order required')
            for key in ('positions','particle_types'):
                desc=record[key];relative=Path(desc['path']);p=Path(path).parent/relative
                require(not relative.is_absolute() and '..' not in relative.parts and p.is_file() and not p.is_symlink()
                        and str(p.resolve())==str(p) and p.resolve().is_relative_to(Path(path).parent)
                        and str(p) not in result and digest(desc.get('sha256'))
                        and type(desc.get('size_bytes')) is int and desc['size_bytes']>0
                        and p.stat().st_size==desc['size_bytes'],'Safe distinct original numeric file binding required')
                result[str(p)]=desc['sha256']
    require(len(result)==2120,'All2120 position/type file bindings required')
    return result


def verify_source_contract(r,path,pin,budget,parent,own):
    require(sys.executable==PYTHON and str(Path(sys.executable).resolve())==PYTHON_REAL
            and r['owner_source']['path']==PREP+'/supervise_sand_reserved_source_cpu_v1.py'
            and str(Path(__file__).resolve())==r['owner_source']['path'], 'Exact source owner/interpreter required')
    require(r['bootstrap_source']['path']==PREP+'/launch_sand_reserved_source_cpu_v1.py','Exact source bootstrap required')
    require(r['command']==source_command(r),'Exact allowlisted source worker argv required')
    mode,split=OPERATIONS[r['operation']];paths=r['paths']
    require(r['core_release']['path']==paths['--root-release'],'Separate original mode release required')
    release=read_json(r['core_release'],budget.check)
    pins=release.get('files_sha256',{})
    require(isinstance(pins,dict) and all(digest(v) for v in pins.values()),'Exact original worker byte bindings required')
    cohort_binding=dict(path=paths['--cohort'],sha256=pins.get(paths['--cohort']))
    audit_binding=dict(path=paths['--cohort-audit'],sha256=pins.get(paths['--cohort-audit']))
    # Only root controls/cohort scalar JSON are read before this complete gate.
    cohort=read_json(cohort_binding,budget.check);audit=read_json(audit_binding,budget.check)
    cohort_gate(cohort,audit)
    require(cohort.get('cohort_audit_sha256')==audit_binding['sha256'],'Cohort/audit bytes differ')
    require(release.get('schema')=='adaptgns_sand_reserved_test_preparation_release_scoped_v1'
            and release.get('issued_by')=='root' and release.get('status')=='approved_for_'+mode
            and release.get('preparation_source_sha256')==PREPARER_SHA
            and release.get('cohort_sha256')==cohort_binding['sha256']
            and release.get('cohort_audit_sha256')==audit_binding['sha256']
            and release.get('output_dir')==paths['--output-dir'] and release.get('no_retry_or_checkpoint_selection') is True,
            'Separate frozen mode authority/cohort bindings differ')
    phase=read_json(r['source_phase'],budget.check)
    require(phase.get('schema')=='adaptgns_sand_original_reserved_source_phase_v1' and phase.get('issued_by')=='root'
            and phase.get('status')=='approved_original_shared_source_phase'
            and phase.get('original_verification_phase_sha256')==VERIFY_PHASE_SHA
            and phase.get('cohort_sha256')==cohort_binding['sha256'] and phase.get('cohort_audit_sha256')==audit_binding['sha256']
            and phase.get('original_shared_source_seconds')==900 and phase.get('original_evaluation_seconds')==11760
            and phase.get('original_analysis_seconds')==3600 and phase.get('global_analysis_deadline_utc')==GLOBAL_STOP.isoformat()
            and phase.get('new_or_restarted_clock_granted') is False,'One original root-issued source phase required')
    start=stamp(phase['preparation_started_utc']);stop=stamp(phase['preparation_stop_utc'])
    require(0<(stop-start).total_seconds()<=900 and start<=stamp(r['invocation_origin_utc'])<stop
            and stamp(r['publication_deadline_utc'])==stop and stop+timedelta(seconds=11760+3600)<=GLOBAL_STOP,
            'Shared900-second source or full downstream reserve changed')
    require(stamp(release['preparation_started_utc'])==start and stamp(release['preparation_stop_utc'])==stop
            and release.get('clock_error_bound_seconds')==5
            and stamp(cohort['created_utc'])<=stamp(release['issued_utc'])<=stamp(r['issued_utc']),
            'Original mode alarm/phase/issuance differs; no per-mode clock restart')
    closure=read_json(phase['cohort_build_external_closure'],budget.check)
    require(closure.get('issued_by')=='root' and closure.get('status')=='complete_cohort_inputs_and_native_closure_verified'
            and closure.get('original_tool_exit_code')==0 and closure.get('owner_terminal_status')=='complete'
            and closure.get('native_absent') and all(x is True for x in closure['native_absent'].values())
            and closure.get('all_published_input_output_hashes_verified') is True
            and closure.get('cohort_sha256')==cohort_binding['sha256'] and closure.get('cohort_audit_sha256')==audit_binding['sha256']
            and stamp(closure['checked_utc'])<=start,'Actual stopped all-six cohort closure must precede source access')
    require(r['verification_phase']['sha256']==VERIFY_PHASE_SHA,'Original verification phase cannot be replaced')
    verification=read_json(r['verification_phase'],budget.check)
    require(verification.get('issued_by')=='root' and verification.get('original_verification_seconds')==2700
            and stamp(r['issued_utc'])>=start,'Original verification lineage differs')
    fixed_names=('prepare_sand_recovered_final_cohort_v1.py','download_sand_public_v2.py','repackage_designsafe_sand.py',
        'evaluate_sand_graph_support_final.py','benchmark_sand_graph_support_rollout.py','sand_graph_support_100k_protocol_v1.md',
        'sand_scoped_operational_amendment_v1.md','sand_scoped_schedule_fixed_spec_v2.json','prepare_sand_reserved_test_recovered_v1.py')
    required={PREP+'/'+name:SOURCES[name] for name in fixed_names}
    required.update({cohort_binding['path']:cohort_binding['sha256'],audit_binding['path']:audit_binding['sha256']})
    input_flags=[f for f in MODE_FLAGS[mode] if f!='--input-dir']
    for flag in input_flags:
        require(paths[flag] in pins,'Explicit mode input lacks original root binding: '+flag)
        required[paths[flag]]=pins[paths[flag]]
    if mode=='convert':
        for name in ('metadata.json','test.npz'):
            p=paths['--input-dir']+'/'+name;require(p in pins,'Acquired source bytes must be root bound');required[p]=pins[p]
    if mode in ('census','candidate','preflight'):
        require(pins[paths['--train-manifest']]==TRAIN_MANIFEST_SHA,'Original full1000 training manifest required')
    if mode in ('candidate','preflight'):
        require(paths['--protocol']==PREP+'/sand_graph_support_100k_protocol_v1.md'
                and pins[paths['--protocol']]==PROTOCOL_SHA and pins[paths['--train-admission']]==TRAIN_ADMISSION_SHA,
                'Original protocol/admission required')
    require(pins==required,'Exact minimal mode source/control closure required')
    require(r['inputs_sha256']=={**pins,r['core_release']['path']:r['core_release']['sha256']},
            'Operational and original mode bindings differ')
    numeric={}
    if mode in ('census','candidate','preflight'):
        numeric=numeric_bindings(paths,pins,budget.check)
        if mode in ('candidate','preflight'):
            census=read_json(dict(path=paths['--census-report'],sha256=pins[paths['--census-report']]),budget.check)
            require(census.get('numeric_files_sha256')==numeric,'Exact complete census numeric binding map required')
    r['_numeric_input_files_count']=len(numeric)
    expected={str(path):pin,**r['inputs_sha256'],**{PREP+'/'+name:value for name,value in SOURCES.items()},
        PYTHON:PYTHON_SHA,PYTHON_REAL:PYTHON_SHA,REMOTE+'/.venv/pyvenv.cfg':VENV_SHA}
    require(all(p not in expected or expected[p]==value for p,value in numeric.items()),'Numeric/control binding conflict')
    expected.update(numeric)
    for key in ('verification_phase','source_phase','owner_source','bootstrap_source'):
        expected[r[key]['path']]=r[key]['sha256']
    expected[phase['cohort_build_external_closure']['path']]=phase['cohort_build_external_closure']['sha256']
    timeout=r['outer_timeout']
    require(timeout['path']=='/usr/bin/timeout' and digest(timeout['sha256'])
            and timeout['duration_rule']=='floor((original_hard_deadline_ns-actual_bootstrap_monotonic_ns)/1e9)-15-5'
            and 'argv' not in timeout,'Derived remaining-time GNU timeout required')
    expected[timeout['path']]=timeout['sha256']
    outputs=[canonical(r['owner_output_dir']),canonical(paths['--output-dir'])]
    require(outputs[0]!=outputs[1] and outputs[0] not in outputs[1].parents and outputs[1] not in outputs[0].parents,
            'Separate worker and owner output trees required')
    protected=[canonical(p) for p in expected if p!=PYTHON]
    if mode=='convert':protected.append(canonical(paths['--input-dir']))
    require(all(o!=p and o not in p.parents and p not in o.parents for o in outputs for p in protected),
            'Source outputs overlap protected inputs')
    require(all(not o.exists() for o in outputs),'Fresh separate output trees required')
    # The full six-model gate above precedes all source/data byte hashing here.
    for p,value in expected.items():require(sha(p,budget.check)==value,'Source/control bytes changed: '+p)
    return expected,source_outputs(mode,paths['--output-dir']),paths['--output-dir'],parent,own


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
        require(p.is_dir() and not p.is_symlink(),'Source output directory missing or symlinked')
        actual=[];directories=[]
        for child in p.rglob('*'):
            check();require(not child.is_symlink(),'Source output symlink')
            if child.is_file():actual.append(str(child))
            elif child.is_dir():directories.append(str(child))
            else:raise ValueError('Unexpected non-file source output')
        expected_dirs=sorted({str(Path(x).parent) for x in paths if Path(x).parent!=p})
        require(sorted(actual)==sorted(paths) and sorted(directories)==expected_dirs,
                'Partial/failed/unexpected source output tree; preserve all files')
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
        publish(directory/'owner_terminal.json',dict(schema='adaptgns_sand_reserved_source_cpu_terminal_v1',
            status='complete' if complete else 'failed',failure=failure,release_sha256=pin,
            child=None if child is None else public(child),child_native_absent=absent,
            input_sha256=expected,evidence_sha256=evidence,output_sha256=result,
            publication_utc=rt.now().isoformat(),publication_monotonic=rt.mono(),
            outer_timeout_identity=outer,owner_identity=own,scientific_admission=False,
            native_hard_guard=r.get('_native_guard'),native_outer_timing=r.get('_native_outer_timing'),
            numeric_input_files_bound=r.get('_numeric_input_files_count',0),
            output_validation_scope='Exact regular-file tree membership and final SHA256. Output JSON/array semantics are inherited from the frozen worker clean exit and require root/downstream review.',
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
        print(json.dumps(dict(status='description_only',execution_authority=False,operations=list(OPERATIONS),
            original_shared_source_seconds=900,owner_has_no_model_or_array_imports=True)));return 0
    require(sys.flags.optimize==0 and a.release is not None and digest(a.release_sha256),'Exact nonoptimized release invocation required')
    path=canonical(a.release);r=read_json(dict(path=str(path),sha256=a.release_sha256));structure(r)
    require(NATIVE_GUARD is not None and a.hard_deadline_monotonic_ns==r['hard_deadline_monotonic_ns'],
            'Previously armed literal native guard required')
    r['_native_guard']=NATIVE_GUARD;r['_outer_term_seconds']=a.outer_term_seconds
    r['_outer_computed_monotonic_ns']=a.outer_computed_monotonic_ns
    rt=Runtime();budget=Budget(r,rt,ENTRY_MONOTONIC)
    return execute(r,path,a.release_sha256,rt,budget)


if __name__=='__main__': raise SystemExit(main())
