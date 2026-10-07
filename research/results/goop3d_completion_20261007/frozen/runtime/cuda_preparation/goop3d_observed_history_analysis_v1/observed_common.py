"""Controls for one separately admitted D3 observed-history allocation."""
from pathlib import Path
import datetime as D
import hashlib
import json
import math
import os
import stat
import time

REMOTE='/root/repos/AdaptGNS-cuda-20261006'
SOURCE=REMOTE+'/cuda_preparation/goop3d_observed_history_analysis_v1'
OUTPUT=REMOTE+'/goop3d_observed_history_analysis_20261006_v1'
QUEUE=REMOTE+'/goop3d_final_evaluation_20261006_v1'
COLLECTION=REMOTE+'/goop3d_final_analysis_20261006_v1/stopped_collection.json'
COLLECTION_SHA='616c612724f62d21bcb521c6163e83d9038b34fdd8012980d6ed55f4b05bed5f'
COLLECTION_BYTES=659091312
COHORT_SHA='645343fc2a1c6ef0a82e212e351b03b1a4d081702d3c8e2187b244a62c6b9bef'
FATE_REVIEW_SHA='2c90d522f903746f0151be03932e86c0cd1a36365233bd19f34bb0b816474203'
OLD_PHASE_SHA='46a0e4c5dd7e35e1233735171eaf3df094da1e2f6236a9928c8ee911d21768df'
HOST='aidenzhou-teal-rat-80-5c78ffbffc-g8djt'
BOOT='4419f0f5-8f54-4cfd-9306-f4a9fc0dd5e6'
GLOBAL_STOP='2026-10-07T01:00:00+00:00'
PHASE_SCHEMA='goop3d_observed_history_analysis_phase_v1'
AUDIT_SCHEMA='goop3d_observed_history_saved_audit_v1'
SUMMARY_SCHEMA='goop3d_observed_history_summary_v1'
CHECK_SCHEMA='goop3d_observed_history_arithmetic_check_v1'
OBSERVED=('same_state_valid','same_state_test','clean_validation')
CAP_COLLECTION=1<<30
CAP_JSON=16<<20
CAP_ARCHIVE=8<<30
CAP_OUTPUT=256<<20

def need(ok,message):
    if not ok:raise ValueError(message)

def digest(raw):return hashlib.sha256(raw).hexdigest()
def is_pin(x):return type(x)is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def strict(raw):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    def bad(v):raise ValueError('nonfinite JSON '+v)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
def stamp(x):
    d=D.datetime.fromisoformat(x)
    need(d.tzinfo is not None and d.utcoffset()==D.timedelta(0),'UTC time required')
    return d
def metadata(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def open_regular(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts and str(p)==str(p.resolve()),'canonical exact path required')
    directory=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
            os.close(directory);directory=nxt
        fd=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
    finally:os.close(directory)
    if not stat.S_ISREG(os.fstat(fd).st_mode):os.close(fd);raise ValueError('ordinary regular file required')
    return fd

def read_bound(path,pin,cap,check=lambda:None,expected_bytes=None,retain=True):
    need(is_pin(pin),'exact file pin required');check();fd=open_regular(path)
    try:
        before=os.fstat(fd);need(0<=before.st_size<=cap,'file size cap')
        need(expected_bytes is None or before.st_size==expected_bytes,'exact accepted byte count')
        h=hashlib.sha256();parts=[];size=0
        while True:
            check();block=os.read(fd,1<<20);check()
            if not block:break
            size+=len(block);need(size<=cap,'file grew past cap');h.update(block)
            if retain:parts.append(block)
        need(size==before.st_size and metadata(before)==metadata(os.fstat(fd)),'file mutated while read')
        need(h.hexdigest()==pin,'accepted file SHA differs');check()
        return b''.join(parts) if retain else size
    finally:os.close(fd)

def validate_phase(p):
    need(p.get('schema')==PHASE_SCHEMA and p.get('issued_by')=='root' and p.get('status')=='approved_bounded_observed_history_analysis','admitted distinct observed-history phase required')
    need(p.get('dataset')=='Goop3D' and p.get('scope')=='all_2568_observed_cells_and_all_4728_accounting','exact scoped population required')
    start,end=stamp(p['started_utc']),stamp(p['stop_utc'])
    need((end-start).total_seconds()==3600 and p.get('seconds')==3600 and end<=stamp(GLOBAL_STOP),'one full hour before fixed global stop required')
    need(p.get('global_stop_utc')==GLOBAL_STOP and p.get('original_phase_sha256')==OLD_PHASE_SHA and p.get('prior_fate_review_sha256')==FATE_REVIEW_SHA,'original expired phase/fate provenance required')
    need(p.get('original_phase_modified') is False and p.get('automatic_retry') is False,'no old phase reset or retry')
    c=p['clock_sample'];need(c['host_boot_id']==BOOT and p['hostname']==HOST,'original host boot required')
    need(c['source']=='root_fresh_tool_and_host_clock_evidence' and c['error_bound_seconds']==5,'root clock provenance required')
    sample=stamp(c['host_utc']);m=c['host_monotonic_seconds']
    need(type(m)in(int,float) and math.isfinite(m) and m>0 and sample==start,'single exact host time origin required')
    need(abs((sample-stamp(c['root_reference_utc'])).total_seconds())<=5,'fresh root/host time disagreement')
    need(p['fresh_native_closure']['all_42_absent'] is True and is_pin(p['fresh_native_closure']['evidence_sha256']),'fresh root native closure pin required')
    return start,end,m

class WorkerBudget:
    def __init__(self,phase,offset):
        start,end,m=validate_phase(phase)
        need(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'worker boot differs')
        self.start=start.timestamp();self.mono_start=m;self.offset=offset
        self.check()
    def check(self):
        wall=time.time()-self.start;mono=time.monotonic()-self.mono_start
        need(0<=wall<self.offset-20 and 0<=mono<self.offset-20 and abs(wall-mono)<=5,'scoped work deadline or clock disagreement')

def load_phase(path,pin):
    need(str(path)==OUTPUT+'/controls/analysis_phase.json','exact new phase path required')
    p=strict(read_bound(path,pin,CAP_JSON));validate_phase(p);return p

def publish(path,value,check=lambda:None):
    check();raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();check()
    need(len(raw)<=CAP_OUTPUT,'bounded output required')
    with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    check();return digest(raw)
