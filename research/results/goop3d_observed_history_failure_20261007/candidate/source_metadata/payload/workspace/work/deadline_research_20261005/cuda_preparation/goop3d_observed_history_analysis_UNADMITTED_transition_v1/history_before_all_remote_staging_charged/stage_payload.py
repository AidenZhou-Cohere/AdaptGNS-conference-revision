"""Isolated bounded stdin staging of only the new exact D3 source/control paths."""
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT='/root/repos/AdaptGNS-cuda-20261006'
SOURCE=ROOT+'/cuda_preparation/goop3d_observed_history_analysis_v1'
OUTPUT=ROOT+'/goop3d_observed_history_analysis_20261006_v1'
SOURCE_NAMES={'observed_common.py','goop3d_observed_history_arithmetic_v1.py','goop3d_saved_diagnostic_audit_v1.py','audit_goop3d_observed_histories_v1.py','summarize_goop3d_observed_histories_v1.py','check_goop3d_observed_history_summary_v1.py','run_observed_history_analysis_v1.py','worker_source_bindings.json','observed_operations.json','supervise_observed_history_cpu_v1.py','launch_observed_history_cpu_v1.py','collect_observed_products.py','observe_collection_closure.py'}
CONTROL_NAMES={'analysis_phase.json','fresh_native_observation.json','prior_fate_review.json','pipeline.cpu_release.json'}
def need(ok,message):
    if not ok:raise ValueError(message)
def ordinary_dir(path):
    p=Path(path);need(p.is_absolute() and str(p.resolve())==str(p),'canonical directory required')
    if not p.exists():p.mkdir(mode=0o700)
    need(p.is_dir() and not p.is_symlink(),'ordinary directory required')
    for parent in p.parents:need(not parent.is_symlink(),'no symlink parent')
def main():
    need(sys.argv[1:2]==['--stage'] and len(sys.argv)==4 and sys.argv[2]=='--payload-sha256','explicit exact staging invocation')
    need(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode,'isolated staging required')
    need(socket.gethostname()=='aidenzhou-teal-rat-80-5c78ffbffc-g8djt' and Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='4419f0f5-8f54-4cfd-9306-f4a9fc0dd5e6','original host and boot required')
    raw=sys.stdin.buffer.read((4<<20)+1);need(len(raw)<=4<<20 and hashlib.sha256(raw).hexdigest()==sys.argv[3],'bounded exact payload')
    payload=json.loads(raw);need(payload['schema']=='goop3d_observed_history_staging_payload_v1','scoped staging schema')
    kind=payload['kind'];need(kind in ('sources','controls'),'exact staging kind')
    base=SOURCE if kind=='sources' else OUTPUT+'/controls';names=SOURCE_NAMES if kind=='sources' else CONTROL_NAMES
    rows=payload['files'];need(len(rows)==len(names) and {r['path'] for r in rows}=={base+'/'+n for n in names},'complete exact new file allowlist')
    blobs=[]
    for row in rows:
        blob=base64.b64decode(row['base64'],validate=True);need(len(blob)<=1<<20 and hashlib.sha256(blob).hexdigest()==row['sha256'],'exact bounded staged bytes')
        blobs.append((Path(row['path']),row['sha256'],blob))
    if kind=='controls':ordinary_dir(OUTPUT)
    ordinary_dir(base)
    if kind=='controls':ordinary_dir(OUTPUT+'/owners')
    result={}
    for path,pin,blob in blobs:
        if path.exists():need(path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==pin,'refuse changed existing staged file')
        else:
            with path.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
        need(hashlib.sha256(path.read_bytes()).hexdigest()==pin,'staged bytes differ');result[str(path)]=pin
    print(json.dumps({'status':'exact_new_files_staged','kind':kind,'files_sha256':result,'worker_started':False}))
if __name__=='__main__':main()
