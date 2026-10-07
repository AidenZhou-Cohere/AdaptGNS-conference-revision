#!/usr/bin/env python3
"""Incrementally copy committed D3 cells over existing SSH; root executes only.

No model, arrays, live status/owner files, or uncommitted cell contents are read.
Final worker-closure collection is separate. Every invocation needs a fresh
local receipt label; interrupted remote staging and all receipts are retained.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import zlib

REMOTE = r'''
import base64,fcntl,hashlib,json,os,re,socket,stat,sys,tarfile,time,traceback,zlib
from pathlib import Path,PurePosixPath

SCHEMA='goop3d_missing_autonomous_worker_v1'
PLAN_SHA='838e7d4c1afc22fc4757a56ec0c755d5d303902d0ed600767d95c507d0402e08'
BASE='/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007'
ATTEMPT_FILES={'owner.json','previous_owner.json','identity.json','resume_verification.json','runtime.json','outcome.json','failed_attempt.json'}
JSON_LIMIT=64*1024*1024
BLOCK=8*1024*1024

def need(ok,message):
 if not ok:raise ValueError(message)

def encode(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def strict(raw):
 def pairs(items):
  result={}
  for key,value in items:
   need(key not in result,'duplicate JSON key');result[key]=value
  return result
 def bad(value):raise ValueError('nonfinite JSON: '+value)
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
def pin(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def component(value):
 need(isinstance(value,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',value) is not None,'unsafe path component')
 return value
def relative(value):
 need(isinstance(value,str) and '\\' not in value,'invalid relative path')
 p=PurePosixPath(value)
 need(not p.is_absolute() and str(p)==value and all(x not in ('','.','..') for x in p.parts),'unsafe relative path')
 for part in p.parts:component(part)
 return p
def safe(path,kind=None,missing=False):
 path=Path(path)
 need(path.is_absolute(),'absolute path required')
 for p in reversed((path,*path.parents)):
  try:st=p.lstat()
  except FileNotFoundError:
   need(missing,'missing file: '+str(p));continue
  need(not stat.S_ISLNK(st.st_mode),'symlink path: '+str(p))
  if p!=path:need(stat.S_ISDIR(st.st_mode),'non-directory ancestor')
 if kind is not None and path.exists():
  need(path.is_file() if kind=='file' else path.is_dir(),'wrong file type: '+str(path))
 return path
def bounded(path):
 path=safe(path,'file');need(path.stat().st_size<=JSON_LIMIT,'JSON too large')
 before=path.stat();raw=path.read_bytes();after=path.stat()
 need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'published JSON changed during read')
 return raw
def file_hash(path):
 path=safe(path,'file');before=path.stat();h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(BLOCK),b''):h.update(block)
 after=path.stat()
 need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'published file changed during hashing')
 return h.hexdigest()
def fsync_dir(path):
 fd=os.open(str(path),os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try:os.fsync(fd)
 finally:os.close(fd)
def mkdir(path,exclusive=False):
 path=safe(path,'dir',True)
 if not path.exists():
  mkdir(path.parent)
  try:path.mkdir();fsync_dir(path.parent)
  except FileExistsError:safe(path,'dir');need(not exclusive,'receipt/staging directory concurrently created: '+str(path))
 else:need(not exclusive,'receipt/staging directory already exists: '+str(path))
 return path
def write_new(path,value):
 path=safe(path,'file',True);mkdir(path.parent)
 with path.open('xb') as stream:stream.write(encode(value));stream.flush();os.fsync(stream.fileno())
 fsync_dir(path.parent)
def machine_identity():return {'hostname':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
def verify_host(expected):
 actual=machine_identity();need(actual==expected,'host/boot identity mismatch');return actual
def descriptor(root,path,kind,expected=None,verify=True):
 path=safe(path,'file');name=str(path.relative_to(root));relative(name)
 value=file_hash(path) if expected is None or verify else expected
 need(pin(value) and (expected is None or value==expected),'file SHA differs: '+name)
 return {'path':name,'sha256':value,'size':path.stat().st_size,'kind':kind}
def check_marker(doc,cell,identity_sha,protocol_sha):
 need(doc.get('schema')==SCHEMA+'_cell_commit' and doc.get('cell_id')==cell and doc.get('plan_sha256')==PLAN_SHA
      and doc.get('identity_sha256')==identity_sha and doc.get('protocol_sha256')==protocol_sha,'commit identity mismatch')
 attempt=doc.get('attempt');need(isinstance(attempt,str) and re.fullmatch('attempt_[0-9]{6}',attempt),'invalid attempt')
 for kind,name in [('row','row.json'),('artifact','trace.npz')]:
  need(doc.get(kind+'_file')==attempt+'/'+name and pin(doc.get(kind+'_sha256')),'invalid marker reference')
 return doc
def inspect_worker(root,index,source_host,verify_artifacts):
 worker=f'worker_{index:02d}';directory=safe(root/worker,'dir',True)
 result={'worker_index':index,'metadata':[],'cells':{}}
 if not directory.exists():return result
 ip=directory/'worker_identity.json';pp=directory/'protocol.json'
 identity=protocol=None;identity_sha=protocol_sha=None
 if ip.exists():
  raw=bounded(ip);identity=strict(raw);identity_sha=digest(raw)
  need(identity.get('schema')==SCHEMA+'_identity' and identity.get('plan_sha256')==PLAN_SHA
       and identity.get('worker_index')==index and identity.get('hostname')==source_host['hostname']
       and pin(identity.get('worker_source_sha256')),'worker identity mismatch')
  ids=identity.get('cell_ids');need(isinstance(ids,list) and len(ids)==len(set(ids)),'invalid cell list')
  for cell in ids:component(cell)
  result['metadata'].append(descriptor(root,ip,'metadata',identity_sha))
 if pp.exists():
  need(identity is not None,'protocol without worker identity');raw=bounded(pp);protocol=strict(raw);protocol_sha=digest(raw)
  need(protocol.get('schema')==SCHEMA and protocol.get('plan_sha256')==PLAN_SHA and protocol.get('identity_sha256')==identity_sha
       and protocol.get('worker_index')==index and protocol.get('hostname')==source_host['hostname']
       and protocol.get('worker_source_sha256')==identity['worker_source_sha256'] and protocol.get('cell_ids')==identity['cell_ids'],'protocol identity mismatch')
  result['metadata'].append(descriptor(root,pp,'metadata',protocol_sha))
 attempts=safe(directory/'attempts','dir',True)
 if attempts.exists():
  for attempt in sorted(attempts.iterdir()):
   need(re.fullmatch('attempt_[0-9]{6}',attempt.name),'unexpected attempt directory');safe(attempt,'dir')
   # These final names are published once by run_worker.save(). Never glob/read temporary names.
   for name in sorted(ATTEMPT_FILES):
    path=attempt/name
    if path.exists():
     raw=bounded(path);strict(raw)
     result['metadata'].append(descriptor(root,path,'metadata',digest(raw)))
 cells=safe(directory/'cells','dir',True)
 if cells.exists():
  need(protocol is not None,'cells without immutable protocol')
  for directory in sorted(cells.iterdir()):
   need(directory.name in identity['cell_ids'],'unassigned cell directory');safe(directory,'dir')
   marker=directory/'commit.json'
   if not marker.exists():continue
   raw=bounded(marker);doc=check_marker(strict(raw),directory.name,identity_sha,protocol_sha)
   rowpath=directory/doc['row_file'];row_raw=bounded(rowpath);row=strict(row_raw)
   need(digest(row_raw)==doc['row_sha256'] and row.get('completion_cell_id')==directory.name
        and row.get('protocol_sha256')==protocol_sha and row.get('artifact_sha256')==doc['artifact_sha256']
        and row.get('artifact_file')=='trace.npz','committed row/marker differs')
   files=[descriptor(root,directory/doc['artifact_file'],'artifact',doc['artifact_sha256'],verify_artifacts),
          descriptor(root,rowpath,'row',doc['row_sha256']),descriptor(root,marker,'commit',digest(raw))]
   result['cells'][directory.name]={'marker_sha256':digest(raw),'files':files}
 return result
def inventory(root,workers,source_host,verify_artifacts):
 root=safe(root,'dir',True)
 return {'schema':'goop3d_committed_transfer_inventory_v1','root':str(root),'workers':[
  inspect_worker(root,i,source_host,verify_artifacts) for i in workers]}
def validate_manifest(manifest,spec):
 need(manifest.get('schema')=='goop3d_committed_transfer_manifest_v1' and manifest.get('binding')==spec['binding'],'transfer manifest binding mismatch')
 files=manifest.get('files');need(isinstance(files,list),'file list required');seen={}
 for entry in files:
  need(set(entry)=={'path','sha256','size','kind'} and pin(entry['sha256']) and type(entry['size']) is int and entry['size']>=0,'invalid file descriptor')
  parts=relative(entry['path']).parts;need(parts[0] in {f'worker_{i:02d}' for i in spec['binding']['workers']},'worker outside scope')
  need(entry['path'] not in seen,'duplicate transfer file')
  if entry['kind']=='metadata':
   need((len(parts)==2 and parts[1] in ('worker_identity.json','protocol.json')) or
        (len(parts)==4 and parts[1]=='attempts' and re.fullmatch('attempt_[0-9]{6}',parts[2]) and parts[3] in ATTEMPT_FILES),'unsafe metadata path')
  elif entry['kind']=='commit':
   need(len(parts)==4 and parts[1]=='cells' and parts[3]=='commit.json','unsafe marker path')
   prefix='/'.join(parts[:3])+'/'
   refs=[v for k,v in seen.items() if k.startswith(prefix)]
   need(len(refs)==2 and {v['kind'] for v in refs}=={'row','artifact'},'marker must follow exactly its row and artifact')
  else:
   need(entry['kind'] in ('row','artifact') and len(parts)==5 and parts[1]=='cells'
        and re.fullmatch('attempt_[0-9]{6}',parts[3]) and parts[4]==('row.json' if entry['kind']=='row' else 'trace.npz'),'unsafe cell path')
  seen[entry['path']]=entry
 for entry in files:
  if entry['kind'] in ('row','artifact'):
   need('/'.join(relative(entry['path']).parts[:3])+'/commit.json' in seen,'cell files without final marker')
 return files
class HashReader:
 def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256();self.count=0
 def read(self,size=-1):
  raw=self.stream.read(size);self.hash.update(raw);self.count+=len(raw);return raw
def pack(manifest,root,output,spec):
 files=validate_manifest(manifest,spec);root=safe(root,'dir');output.write(encode(manifest));output.flush()
 with tarfile.open(fileobj=output,mode='w|') as tar:
  for entry in files:
   path=safe(root/entry['path'],'file');before=path.stat();need(before.st_size==entry['size'],'source size changed')
   info=tarfile.TarInfo(entry['path']);info.size=entry['size'];info.mode=0o444;info.mtime=0
   with path.open('rb') as stream:
    reader=HashReader(stream);tar.addfile(info,reader)
   after=path.stat()
   need(reader.count==entry['size'] and reader.hash.hexdigest()==entry['sha256'],'source SHA changed while streaming: '+entry['path'])
   need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'source changed while streaming')
def marker_ready(root,path,raw,files_by_path):
 parts=relative(path).parts;worker=root/parts[0];cell=root/Path(*parts[:3])
 identity_raw=bounded(worker/'worker_identity.json');protocol_raw=bounded(worker/'protocol.json')
 doc=check_marker(strict(raw),parts[2],digest(identity_raw),digest(protocol_raw))
 identity=strict(identity_raw);protocol=strict(protocol_raw)
 need(parts[2] in identity['cell_ids'] and protocol['identity_sha256']==digest(identity_raw),'destination worker/marker binding')
 for kind in ('artifact','row'):
  rel='/'.join(parts[:3])+'/'+doc[kind+'_file'];entry=files_by_path.get(rel)
  need(entry is not None and entry['kind']==kind and entry['sha256']==doc[kind+'_sha256'],'marker/manifest reference differs')
  target=safe(cell/doc[kind+'_file'],'file');need(target.stat().st_size==entry['size'] and file_hash(target)==entry['sha256'],'destination marker reference not verified')
 row=strict(bounded(cell/doc['row_file']))
 need(row.get('completion_cell_id')==parts[2] and row.get('protocol_sha256')==digest(protocol_raw)
      and row.get('artifact_sha256')==doc['artifact_sha256'] and row.get('artifact_file')=='trace.npz','destination row/commit mismatch')
def receive(manifest,root,stream,attempt,spec):
 files=validate_manifest(manifest,spec);by_path={e['path']:e for e in files};records=[]
 with tarfile.open(fileobj=stream,mode='r|') as tar:
  for entry in files:
   member=tar.next();need(member is not None and member.name==entry['path'] and member.isfile() and member.size==entry['size'],'unexpected/out-of-order tar member')
   target=safe(root/entry['path'],'file',True);temp=attempt/'staging'/(entry['path']+'.part');mkdir(temp.parent)
   h=hashlib.sha256();count=0
   with tar.extractfile(member) as incoming,temp.open('xb') as outgoing:
    try:
     while True:
      block=incoming.read(BLOCK)
      if not block:break
      outgoing.write(block);h.update(block);count+=len(block)
    finally:outgoing.flush();os.fsync(outgoing.fileno())
   fsync_dir(temp.parent)
   need(count==entry['size'] and h.hexdigest()==entry['sha256'],'received file SHA/size differs: '+entry['path'])
   if entry['kind']=='commit':marker_ready(root,entry['path'],bounded(temp),by_path)
   mkdir(target.parent)
   if target.exists():
    need(target.stat().st_size==entry['size'] and file_hash(target)==entry['sha256'],'different destination bytes: '+entry['path']);state='same_hash_retained'
   else:
    # Atomic no-clobber publication. Retain staging (hardlink; no second data copy).
    try:os.link(temp,target);fsync_dir(target.parent);state='published'
    except FileExistsError:
     need(file_hash(target)==entry['sha256'],'concurrent differing destination bytes');state='same_hash_retained'
   record={**entry,'state':state};records.append(record)
   write_new(attempt/'files'/f'{len(records):06d}.json',record)
  need(tar.next() is None,'unexpected extra tar member')
 return records
def remote_main():
 action=sys.argv[1];spec=strict(zlib.decompress(base64.b64decode(sys.argv[2])))
 binding=spec['binding'];component(binding['label'])
 need(binding['source_alias'] in ('yellow-worm-77','aquamarine-toad-75') and binding['destination_alias']=='teal-rat-80','unsupported host pair')
 need(binding['workers']==(list(range(4,8)) if binding['source_alias']=='yellow-worm-77' else list(range(8,12))),'wrong worker range')
 need(binding['source_root']==BASE+'/autonomous_results_v1' and binding['destination_root']==BASE+'/collected_autonomous_v1','wrong roots')
 side='source' if action in ('source_inventory','pack') else 'destination'
 actual=verify_host(binding[side+'_identity'])
 if action in ('source_inventory','destination_inventory'):
  result=inventory(Path(binding[side+'_root']),binding['workers'],binding['source_identity'],side=='destination')
  result['host_identity']=actual;print(json.dumps(result,sort_keys=True));return
 if action=='pack':
  raw=sys.stdin.buffer.read(JSON_LIMIT+1);need(len(raw)<=JSON_LIMIT and digest(raw)==spec['manifest_sha256'],'source manifest bytes differ')
  pack(strict(raw),Path(binding['source_root']),sys.stdout.buffer,spec);return
 need(action=='receive','unknown action')
 root=mkdir(Path(binding['destination_root']));attempt=mkdir(root/'_transfer_attempts'/(binding['source_alias']+'_'+binding['label']),exclusive=True)
 write_new(attempt/'request.json',spec);write_new(attempt/'host_identity.json',actual)
 lock_path=root/'_transfer_locks'/(binding['source_alias']+'.lock');mkdir(lock_path.parent);safe(lock_path,'file',True)
 try:
  with lock_path.open('a+b') as lock:
   fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   raw=sys.stdin.buffer.readline(JSON_LIMIT+1);need(len(raw)<=JSON_LIMIT and digest(raw)==spec['manifest_sha256'],'destination manifest bytes differ')
   manifest=strict(raw);write_new(attempt/'manifest.json',manifest)
   records=receive(manifest,root,sys.stdin.buffer,attempt,spec)
   result={'status':'complete','host_identity':actual,'manifest_sha256':spec['manifest_sha256'],'files':len(records),'bytes':sum(e['size'] for e in records),'commits':sum(e['kind']=='commit' for e in records)}
 except BaseException as error:
  write_new(attempt/'result.json',{'status':'failed','host_identity':actual,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'all_staging_and_file_receipts_retained':True})
  raise
 write_new(attempt/'result.json',result);print(json.dumps(result,sort_keys=True))
'''


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def choose_manifest(source, destination, binding):
    """No metrics: select missing immutable files using verified inventories."""
    wanted = binding["workers"]
    if [w["worker_index"] for w in source["workers"]] != wanted or [w["worker_index"] for w in destination["workers"]] != wanted:
        raise ValueError("inventory worker ordering differs")
    files, skipped_cells = [], 0
    for src, dst in zip(source["workers"], destination["workers"]):
        old = {e["path"]: e for e in dst["metadata"]}
        for entry in src["metadata"]:
            if entry["path"] in old:
                if entry != old[entry["path"]]:
                    raise ValueError("different destination metadata: " + entry["path"])
            else:
                files.append(entry)
        for cell, record in src["cells"].items():
            if cell in dst["cells"]:
                if record != dst["cells"][cell]:
                    raise ValueError("different destination committed cell: " + cell)
                skipped_cells += 1
            else:
                files.extend(record["files"])
    return {"schema": "goop3d_committed_transfer_manifest_v1", "binding": binding,
            "files": files, "already_verified_cells_skipped": skipped_cells,
            "source_committed_cells": sum(len(w["cells"]) for w in source["workers"]),
            "scope": "Immutable published metadata and marker-selected row/NPZ only; final closure collection is separate."}


def initial_identity(observations, key):
    entry = observations[key]
    if entry["status"] != "fulfilled" or entry["value"]["exit_code"] != 0:
        raise ValueError("initial host observation failed")
    value = json.loads(entry["value"]["output"])
    return {"hostname": value["host"], "boot_id": value["boot_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--host", required=True, choices=("yellow-worm-77", "aquamarine-toad-75"))
    parser.add_argument("--label", required=True, help="Fresh receipt label, letters/digits/underscore/hyphen")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.label) is None:
        parser.error("invalid fresh receipt label")
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    out = here / ("transfer_committed_" + args.label)
    out.mkdir(exist_ok=False)  # Check fresh receipt before any network action.
    observation_path = here / "initial_host_observations.json"
    raw_observations = observation_path.read_bytes()
    observations = json.loads(raw_observations)
    remote_base = "/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007"
    binding = {"source_alias": args.host, "destination_alias": "teal-rat-80", "label": args.label,
               "workers": list(range(4, 8)) if args.host == "yellow-worm-77" else list(range(8, 12)),
               "source_root": remote_base + "/autonomous_results_v1", "destination_root": remote_base + "/collected_autonomous_v1",
               "source_identity": initial_identity(observations, "yellow" if args.host == "yellow-worm-77" else "aquamarine"),
               "destination_identity": initial_identity(observations, "teal"),
               "initial_observations_sha256": hashlib.sha256(raw_observations).hexdigest(),
               "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "binding.json").write_bytes(encode(binding))
    (out / "initial_host_observations.json").write_bytes(raw_observations)
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
           "-F", str(root / "work/deadline_research_20261005/cuda_preparation/ssh_config")]

    def command(host, action, spec):
        token = base64.b64encode(zlib.compress(encode(spec))).decode()
        return ssh + [host + ".coder", shlex.join(["python3", "-c", REMOTE + "\nremote_main()\n", action, token])]

    processes = []
    started = time.time()
    result = {"status": "failed", "binding": binding}
    try:
        inventories = {}
        for side, host in (("destination", "teal-rat-80"), ("source", args.host)):
            run = subprocess.run(command(host, side + "_inventory", {"binding": binding}), capture_output=True)
            (out / (side + "_inventory.stdout")).write_bytes(run.stdout)
            (out / (side + "_inventory.stderr")).write_bytes(run.stderr)
            (out / (side + "_inventory.exit.json")).write_bytes(encode({"exit_code": run.returncode}))
            if run.returncode:
                raise RuntimeError(side + " inventory failed; inspect receipt")
            inventories[side] = json.loads(run.stdout)
        manifest = choose_manifest(inventories["source"], inventories["destination"], binding)
        raw = encode(manifest)
        (out / "manifest.json").write_bytes(raw)
        spec = {"binding": binding, "manifest_sha256": hashlib.sha256(raw).hexdigest()}
        (out / "request.json").write_bytes(encode(spec))
        with (out / "source.stderr").open("xb") as source_err, (out / "destination.stderr").open("xb") as dest_err, (out / "destination.stdout").open("xb") as dest_out:
            source = subprocess.Popen(command(args.host, "pack", spec), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=source_err)
            processes.append(source)
            destination = subprocess.Popen(command("teal-rat-80", "receive", spec), stdin=source.stdout, stdout=dest_out, stderr=dest_err)
            processes.append(destination)
            source.stdout.close()
            (out / "processes.json").write_bytes(encode({"source_ssh_pid": source.pid, "destination_ssh_pid": destination.pid, "started": started}))
            source.stdin.write(raw)
            source.stdin.close()
            destination_rc = destination.wait()
            source_rc = source.wait()
        result.update(source_exit=source_rc, destination_exit=destination_rc, manifest_sha256=spec["manifest_sha256"],
                      transferred_files=len(manifest["files"]), transferred_cells=sum(e["kind"] == "commit" for e in manifest["files"]),
                      already_verified_cells_skipped=manifest["already_verified_cells_skipped"])
        if source_rc or destination_rc:
            raise RuntimeError("transfer subprocess failed; inspect preserved receipts/staging")
        remote_result = json.loads((out / "destination.stdout").read_bytes())
        if (remote_result.get("status") != "complete" or remote_result.get("manifest_sha256") != spec["manifest_sha256"]
                or remote_result.get("host_identity") != binding["destination_identity"]
                or remote_result.get("files") != len(manifest["files"])
                or remote_result.get("commits") != result["transferred_cells"]
                or remote_result.get("bytes") != sum(e["size"] for e in manifest["files"])):
            raise RuntimeError("destination completion receipt differs")
        result.update(status="complete", destination_receipt=remote_result)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        # Only this helper's local SSH children; never signal remote research owners.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        result.update(seconds=time.time() - started, child_exit_codes=[p.returncode for p in processes])
        (out / "result.json").write_bytes(encode(result))
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
