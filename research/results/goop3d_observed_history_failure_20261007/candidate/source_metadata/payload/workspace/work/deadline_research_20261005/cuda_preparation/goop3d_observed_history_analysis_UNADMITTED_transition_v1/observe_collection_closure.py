"""One bounded metadata-only observation of all47 original pipeline/collector PIDs."""
import argparse
import ctypes
import datetime as D
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import sys
import time

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

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(b):return hashlib.sha256(b).hexdigest()

def stamp(s):return D.datetime.fromisoformat(s).timestamp()

def read_file(path,cap,check,pin=None):
    check();p=Path(path);require(p.is_absolute() and str(p)==str(p.resolve()),'canonical retained file')
    for parent in p.parents:require(not parent.is_symlink(),'no retained parent symlinks')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd);require(stat.S_ISREG(a.st_mode) and a.st_size<=cap,'bounded ordinary retained file')
        data=bytearray()
        while True:
            check();block=os.read(fd,1<<20);check()
            if not block:break
            data.extend(block);require(len(data)<=cap,'retained file size cap')
        b=os.fstat(fd);require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns),'retained file changed')
        raw=bytes(data);require(pin is None or sha(raw)==pin,'retained file hash differs');return raw
    finally:os.close(fd)

ROOT='/root/repos/AdaptGNS-cuda-20261006/goop3d_observed_history_analysis_20261006_v1'

HOST='aidenzhou-teal-rat-80-5c78ffbffc-g8djt'

BOOT='4419f0f5-8f54-4cfd-9306-f4a9fc0dd5e6'

HISTORICAL={42898,42899,42911,42912,42913,42914,43568,43632,43633,43760,46156,46220,46221,46351,48776,48851,48979,49043,51739,51807,51946,52010,54713,54714,55094,55158,57372,57438,59710,59776,62415,62481,65470,65471,65472,65473,65892,65893,65928,66518,66519,66558}

def expected_identities(report):
    require(report['schema']=='goop3d_observed_products_collection_v1' and report['status']=='four_products_and_owner_closure_collected' and report['host']==HOST and report['boot_id']==BOOT,'exact preceding collection report')
    identities={**report['new_owner_identities'],'collector':report['collector_identity'],'collector_outer':report['collector_parent_identity']}
    require(set(identities)=={'outer','owner','worker','collector','collector_outer'},'five original native identities')
    for row in identities.values():
        require(type(row)is dict and all(type(row[k])is int and row[k]>0 for k in ('pid','ppid','pgid','sid')) and type(row['start_id'])is str and row['start_id'].isdigit(),'original native PID/start identity')
    own,parent=identities['collector'],identities['collector_outer']
    require(own['ppid']==parent['pid'] and own['pgid']==parent['pgid'] and own['sid']==parent['sid'] and parent['executable']=='/usr/bin/timeout','original collector timeout ancestry')
    expected=HISTORICAL|{row['pid'] for row in identities.values()}
    require(len(expected)==47,'exact old42 plus pipeline3 and collector2')
    return identities,expected

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--observe',action='store_true');p.add_argument('--deadline-monotonic-ns',type=int);p.add_argument('--phase-sha256');p.add_argument('--report-sha256');p.add_argument('--observer-sha256');args=p.parse_args()
    if not args.observe:return
    guard=arm_absolute_guard(args.deadline_monotonic_ns)
    require(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and not sys.flags.optimize,'isolated metadata observer required')
    require(socket.gethostname()==HOST and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'exact original host/boot')
    def initial_check():require(time.monotonic_ns()<args.deadline_monotonic_ns,'observer hard deadline')
    read_file(Path(__file__).resolve(),16<<20,initial_check,args.observer_sha256)
    phase_raw=read_file(ROOT+'/controls/analysis_phase.json',16<<20,initial_check,args.phase_sha256);phase=json.loads(phase_raw)
    require(phase['schema']=='goop3d_observed_history_analysis_phase_v1' and phase['status']=='approved_bounded_observed_history_analysis' and phase['issued_by']=='root' and phase['seconds']==3600,'distinct admitted original phase')
    start=stamp(phase['started_utc']);m=phase['clock_sample']['host_monotonic_seconds']
    require(args.deadline_monotonic_ns==int((m+3480-5)*1e9),'fixed phase-relative observer guard')
    def check():
        initial_check();wall=time.time()-start;mono=time.monotonic()-m
        require(0<=wall<3470 and 0<=mono<3470 and abs(wall-mono)<=5,'fixed post-collection observation deadline')
    check();raw=sys.stdin.buffer.read((1<<20)+1);check()
    require(len(raw)<=1<<20 and sha(raw)==args.report_sha256,'exact bounded original collection report')
    report=json.loads(raw);require(report['phase_sha256']==args.phase_sha256 and stamp(report['finished_utc'])<=time.time(),'preceding same-phase collection')
    identities,expected=expected_identities(report)
    def absence():
        check();values={str(pid):native(pid) for pid in sorted(expected)}
        require(all(value is None for value in values.values()),'original historical/pipeline/collector native PID remains; no reuse inference')
        return {pid:True for pid in values}
    first=absence();read_file(ROOT+'/controls/analysis_phase.json',16<<20,check,args.phase_sha256);second=absence()
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'final original boot unchanged');check()
    print(json.dumps({'schema':'goop3d_observed_postcollection_native_closure_v1','status':'all47_original_pids_absent_twice','phase_sha256':args.phase_sha256,'collection_report_sha256':args.report_sha256,'release_sha256':report['release_sha256'],'host':HOST,'boot_id':BOOT,'original_identities':identities,'native_first_absent':first,'native_second_absent':second,'native_guard':guard,'finished_utc':D.datetime.now(D.timezone.utc).isoformat(),'finished_monotonic':time.monotonic(),'no_files_written':True,'scientific_program_run':False,'original_result_queue_opened':False},sort_keys=True))

if __name__=='__main__':main()
