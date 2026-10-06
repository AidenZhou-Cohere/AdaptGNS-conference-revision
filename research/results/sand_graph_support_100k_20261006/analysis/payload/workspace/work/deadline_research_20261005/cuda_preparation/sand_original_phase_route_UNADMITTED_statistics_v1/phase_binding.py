"""Read-only binding to the existing original Sand phase; never grants a clock."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import os
import stat
import time

PREP=Path(__file__).resolve().parent.parent
STATE=PREP/'sand_stopped_analysis_released_root_v1'
PHASE=STATE/'analysis_phase.json'
ANCHOR=STATE/'local_phase_anchor.json'
GLOBAL_STOP='2026-10-07T04:00:00+00:00'
FROZEN_PACKAGE='914e6e2df11836079f721a675bdbfcd9673b6bada3074fe766c52d085a37640b'
OPERATOR=PREP/'sand_stopped_analysis_operator_UNADMITTED_code_audit_v1/root_operator.py'
OPERATOR_SHA='cc6b22b41a11d6bcb5481bbd3b420267f3f776f9e0b5bef83da6ca94a7763795'


def need(ok,message):
    if not ok: raise ValueError(message)

def sha(raw): return hashlib.sha256(raw).hexdigest()
def pin(value): return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'duplicate JSON key'); out[k]=v
        return out
    def reject(value): raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)
def stamp(value):
    need(type(value) is str,'UTC timestamp required')
    dt=datetime.datetime.fromisoformat(value)
    need(dt.tzinfo is not None and dt.utcoffset()==datetime.timedelta(0),'UTC timestamp required')
    return dt

def read_pinned(path,expected,cap=32768):
    need(pin(expected) and path.is_absolute() and path.resolve()==path and not path.is_symlink(),'canonical existing pinned file required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd); need(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'bounded regular file required')
        raw=bytearray()
        while True:
            block=os.read(fd,min(65536,cap+1-len(raw)))
            if not block: break
            raw.extend(block); need(len(raw)<=cap,'file read cap exceeded')
        after=os.fstat(fd)
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'file changed while read')
    finally: os.close(fd)
    need(sha(raw)==expected,'original raw file SHA differs'); return bytes(raw)

class OriginalPhase:
    def __init__(self,phase_raw,anchor_raw,phase_sha,anchor_sha):
        need(pin(phase_sha) and pin(anchor_sha) and sha(phase_raw)==phase_sha and sha(anchor_raw)==anchor_sha,'exact phase and original anchor bytes required')
        p,a=strict(phase_raw),strict(anchor_raw)
        need(type(p) is dict and p.get('schema')=='adaptgns_sand_original_analysis_phase_v1' and p.get('issued_by')=='root' and p.get('status')=='approved_original_analysis_phase' and p.get('dataset')=='Sand','actual admitted original Sand phase required')
        need(type(p.get('original_analysis_seconds')) is int and p['original_analysis_seconds']==3600 and p.get('new_or_restarted_clock_granted') is False,'exact original hour; no new clock')
        start,stop=stamp(p['original_analysis_started_utc']),stamp(p['original_analysis_stop_utc'])
        need((stop-start).total_seconds()==3600 and p.get('global_analysis_deadline_utc')==GLOBAL_STOP and stop<=stamp(GLOBAL_STOP),'original full hour and global stop required')
        sessions=p.get('original_evaluation_sessions'); need(type(sessions) is dict and set(sessions)=={'A','B'} and all(type(sessions[r]) is int and sessions[r]==v for r,v in [('A',51595),('B',12559)]),'original Sand sessions required')
        closures=p.get('original_evaluation_closures'); need(type(closures) is dict and set(closures)=={'A','B'},'original closure bindings required')
        for role,row in closures.items():
            need(type(row) is dict and row.get('path')=='/root/repos/AdaptGNS-cuda-20261006/sand_final_analysis_20261006_v1/controls/'+role+'.evaluation_closure.json' and pin(row.get('sha256')),'original closure pin required')
        need(type(a) is dict and a.get('phase_sha256')==phase_sha and a.get('utc')==start.isoformat(),'original phase/anchor link required')
        tick=a.get('monotonic_seconds'); need(type(tick) in (int,float) and math.isfinite(tick) and tick>=0,'original monotonic anchor required')
        self.start=start.timestamp(); self.stop=stop.timestamp(); self.mono_start=tick; self.mono_stop=tick+3600
        self.metadata={'analysis_phase_path':str(PHASE),'analysis_phase_sha256':phase_sha,'local_phase_anchor_sha256':anchor_sha,'original_analysis_started_utc':start.isoformat(),'original_analysis_stop_utc':stop.isoformat(),'original_monotonic_stop':self.mono_stop,'new_or_restarted_clock_granted':False}
    def remaining(self):
        wall=time.time()-self.start; mono=time.monotonic()-self.mono_start
        need(math.isfinite(wall) and math.isfinite(mono) and 0<=wall<3600 and 0<=mono<3600 and abs(wall-mono)<=5,'original phase expired or UTC/monotonic disagreement')
        remaining=min(self.stop-time.time(),self.mono_stop-time.monotonic())
        need(0<remaining<=3600,'no original phase time remains'); return remaining
    def expired(self):
        try: self.remaining(); return False
        except (ValueError,OverflowError): return True

def load_original_phase(phase_sha,anchor_sha):
    phase=OriginalPhase(read_pinned(PHASE,phase_sha),read_pinned(ANCHOR,anchor_sha),phase_sha,anchor_sha)
    phase.remaining(); return phase
