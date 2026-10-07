"""Bounded stopped-product collection. No models, arrays, or original queue reads."""
import ctypes
import os
import signal
import sys
import time
from pathlib import Path


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

import argparse
import datetime as D
import hashlib
import io
import json
import socket
import stat
import tarfile

ROOT='/root/repos/AdaptGNS-cuda-20261006/goop3d_observed_history_analysis_20261006_v1'
OWNER=ROOT+'/owners/d3_observed_history_pipeline'
HOST='aidenzhou-teal-rat-80-5c78ffbffc-g8djt'
BOOT='4419f0f5-8f54-4cfd-9306-f4a9fc0dd5e6'
HISTORICAL={42898,42899,42911,42912,42913,42914,43568,43632,43633,43760,46156,46220,46221,46351,48776,48851,48979,49043,51739,51807,51946,52010,54713,54714,55094,55158,57372,57438,59710,59776,62415,62481,65470,65471,65472,65473,65892,65893,65928,66518,66519,66558}
PRODUCTS=('audit.json','summary.json','arithmetic_check.json','pipeline_receipt.json')
REGISTRY=('owner_started.json','child_registered.json','child.stdout','child.stderr','owner_terminal.json')

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

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--collect',action='store_true');p.add_argument('--deadline-monotonic-ns',type=int);p.add_argument('--phase-sha256');p.add_argument('--release-sha256');p.add_argument('--collector-sha256')
    args=p.parse_args()
    if not args.collect:return
    # Arm the unchanged kernel guard before opening any controls or products.
    guard=arm_absolute_guard(args.deadline_monotonic_ns)
    require(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,'isolated collector required')
    require(socket.gethostname()==HOST and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'exact host and boot')
    def initial_check():require(time.monotonic_ns()<args.deadline_monotonic_ns,'collector hard deadline')
    own=native(os.getpid());parent=native(os.getppid())
    require(own is not None and parent is not None and own['ppid']==parent['pid'] and own['pgid']==parent['pgid'] and own['sid']==parent['sid'],'collector native ancestry')
    invocation=[ROOT.rsplit('/',1)[0]+'/.venv/bin/python','-I','-S','-B',str(Path(__file__).resolve()),*sys.argv[1:]]
    require(own['argv']==invocation and own['executable']=='/usr/bin/python3.12' and parent['executable']=='/usr/bin/timeout' and parent['argv'][3:]==invocation and parent['argv'][:2]==['/usr/bin/timeout','--signal=KILL'] and parent['argv'][2].endswith('s') and 0<int(parent['argv'][2][:-1])<=360,'exact original collector and native outer timeout')
    read_file(Path(__file__).resolve(),16<<20,initial_check,args.collector_sha256)
    phase_raw=read_file(ROOT+'/controls/analysis_phase.json',16<<20,initial_check,args.phase_sha256);phase=json.loads(phase_raw)
    require(phase['schema']=='goop3d_observed_history_analysis_phase_v1' and phase['status']=='approved_bounded_observed_history_analysis' and phase['issued_by']=='root' and phase['seconds']==3600,'distinct admitted phase')
    start=stamp(phase['started_utc']);m=phase['clock_sample']['host_monotonic_seconds']
    require(args.deadline_monotonic_ns==int((m+3420-5)*1e9),'collector fixed phase-relative native deadline')
    def check():
        initial_check();wall=time.time()-start;mono=time.monotonic()-m
        require(0<=wall<3400 and 0<=mono<3400 and abs(wall-mono)<=5,'fixed collection deadline or clock disagreement')
    release_raw=read_file(ROOT+'/controls/pipeline.cpu_release.json',16<<20,check,args.release_sha256);release=json.loads(release_raw)
    require(release['analysis_phase']['sha256']==args.phase_sha256 and release['operation']=='d3_observed_history_pipeline','same pipeline release')
    raw_registry={n:read_file(OWNER+'/'+n,16<<20,check) for n in REGISTRY};records={n:json.loads(b) for n,b in raw_registry.items() if n.endswith('.json')}
    terminal=records['owner_terminal.json'];child=terminal['child'];outer=terminal['outer_timeout_identity'];owner=terminal['owner_identity']
    require(terminal['schema']=='adaptgns_stopped_analysis_cpu_terminal_v1' and terminal['status']=='complete' and terminal['failure'] is None and terminal['release_sha256']==args.release_sha256,'original complete owner terminal')
    require(child['exit_code']==0 and child['reaped'] is True and child['signals']==[] and child['cleanup_errors']==[] and child['identity'] is not None and terminal['child_native_absent'] is True,'original registered worker complete and reaped')
    require(records['owner_started.json']['owner_identity']==owner and records['owner_started.json']['outer_timeout_identity']==outer and records['owner_started.json']['release_sha256']==args.release_sha256,'owner start matches terminal')
    registered=records['child_registered.json'];require(registered['pid']==child['pid'] and registered['identity']==child['identity'] and child['command']==release['command'],'exact registered worker and command')
    require(owner['ppid']==outer['pid'] and child['identity']['ppid']==owner['pid'] and owner['pgid']==outer['pgid']==child['identity']['pgid'],'exact new native ancestry')
    expected=HISTORICAL|{outer['pid'],owner['pid'],child['pid']};require(len(expected)==45,'all old42 plus new3 identities')
    def absence():
        check();observed={str(pid):native(pid) for pid in sorted(expected)}
        require(all(v is None for v in observed.values()),'historical or new worker native process remains');return {k:True for k in observed}
    first=absence()
    hashes=terminal['output_sha256'];require(set(hashes)=={ROOT+'/'+n for n in PRODUCTS},'exact four terminal product paths')
    for n,b in raw_registry.items():
        if n!='owner_terminal.json':require(terminal['evidence_sha256'][OWNER+'/'+n]==sha(b),'terminal binds original registry')
    require(set(terminal['evidence_sha256'])=={OWNER+'/'+n for n in REGISTRY if n!='owner_terminal.json'},'complete exact owner evidence map')
    require(terminal['input_sha256'][ROOT+'/controls/pipeline.cpu_release.json']==args.release_sha256,'terminal binds exact operational release')
    def add(tar,name,raw):
        check();info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;info.mtime=0;tar.addfile(info,io.BytesIO(raw));check()
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
        for name in PRODUCTS:
            raw=read_file(ROOT+'/'+name,256<<20,check,hashes[ROOT+'/'+name]);add(tar,name,raw)
        for name,raw in raw_registry.items():add(tar,name,raw)
        for name in PRODUCTS:read_file(ROOT+'/'+name,256<<20,check,hashes[ROOT+'/'+name])
        for name,raw in raw_registry.items():read_file(OWNER+'/'+name,16<<20,check,sha(raw))
        second=absence();require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'final boot differs')
        stable=lambda row:{k:row[k] for k in ('pid','ppid','pgid','sid','start_id','argv','executable')}
        require(stable(native(os.getpid()))==stable(own) and stable(native(os.getppid()))==stable(parent),'collector native identities changed')
        require(len(expected|{own['pid'],parent['pid']})==47,'all old42 plus pipeline3 plus collector2 are distinct')
        report={'schema':'goop3d_observed_products_collection_v1','status':'four_products_and_owner_closure_collected','phase_sha256':args.phase_sha256,'release_sha256':args.release_sha256,'host':HOST,'boot_id':BOOT,'native_first_absent':first,'native_second_absent':second,'new_owner_identities':{'outer':outer,'owner':owner,'worker':child['identity']},'product_sha256':{n:hashes[ROOT+'/'+n] for n in PRODUCTS},'registry_sha256':{n:sha(b) for n,b in raw_registry.items()},'native_guard':guard,'collector_identity':own,'collector_parent_identity':parent,'finished_utc':D.datetime.now(D.timezone.utc).isoformat(),'finished_monotonic':time.monotonic(),'scientific_admission':False,'original_result_queue_opened':False}
        add(tar,'collection.json',(json.dumps(report,indent=2,sort_keys=True)+'\n').encode())
    check()

if __name__=='__main__':main()
