"""Copy pinned D3 runtime inputs over existing SSH, into a fresh directory."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[3]
SSH_CONFIG = ROOT / "work/deadline_research_20261005/cuda_preparation/ssh_config"
SOURCE = r'''
import base64,json,sys,tarfile,zlib
from pathlib import Path
x=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for entry in x['files']:
  p=Path(entry['path']); rel=entry['relative_path']
  assert p.is_file() and not p.is_symlink(),str(p)
  size=p.stat().st_size
  assert entry['size_bytes'] is None or size==entry['size_bytes'],str(p)
  info=tarfile.TarInfo(rel);info.size=size;info.mode=0o644;info.mtime=0
  with p.open('rb') as f:tar.addfile(info,f)
'''
DESTINATION = r'''
import base64,hashlib,json,os,socket,sys,tarfile,time,zlib
from pathlib import Path,PurePosixPath
x=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
root=Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=False)
expected={e['relative_path']:e for e in x['files']};assert len(expected)==len(x['files'])
seen={};started=time.time()
try:
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
  for member in tar:
   name=member.name; rel=PurePosixPath(name)
   assert name in expected and name not in seen and not rel.is_absolute() and '..' not in rel.parts
   assert member.isfile() and 0<=member.size<=2**30
   entry=expected[name];assert entry['size_bytes'] is None or member.size==entry['size_bytes']
   target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
   h=hashlib.sha256();count=0
   with tar.extractfile(member) as src,target.open('xb') as out:
    while True:
     block=src.read(8*1024*1024)
     if not block:break
     out.write(block);h.update(block);count+=len(block)
    out.flush();os.fsync(out.fileno())
   assert count==member.size and h.hexdigest()==entry['sha256'],name
   seen[name]={'bytes':count,'sha256':h.hexdigest()}
 assert set(seen)==set(expected)
 record={'status':'complete','host':socket.gethostname(),'root':str(root),'started':started,'finished':time.time(),'files':seen,'manifest_sha256':sys.argv[3]}
except BaseException as e:
 record={'status':'failed','host':socket.gethostname(),'root':str(root),'started':started,'finished':time.time(),'verified_files':seen,'error':repr(e),'manifest_sha256':sys.argv[3]}
 (root/'transfer_status.json').write_text(json.dumps(record,indent=2)+'\n')
 raise
(root/'transfer_status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':'complete','host':record['host'],'root':str(root),'files':len(seen),'bytes':sum(v['bytes'] for v in seen.values()),'seconds':record['finished']-started,'manifest_sha256':sys.argv[3]}))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);args=p.parse_args()
    assert args.host in {'yellow-worm-77','aquamarine-toad-75'}
    source_path=ROOT/'work/goop3d_completion_20261007/autonomous_v1/transfer_files.json'
    raw=source_path.read_bytes();manifest=json.loads(raw)
    for e in manifest['files']:
        assert e['path']=='/root/repos/AdaptGNS-cuda-20261006/'+e['relative_path']
        assert '..' not in Path(e['relative_path']).parts and not Path(e['relative_path']).is_absolute()
        assert len(e['sha256'])==64
    token=base64.b64encode(zlib.compress(raw)).decode();pin=hashlib.sha256(raw).hexdigest()
    dest='/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007/runtime_v1'
    out=Path(__file__).parent/('transfer_'+args.host);out.mkdir(exist_ok=False)
    (out/'manifest.json').write_bytes(raw)
    ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-F',str(SSH_CONFIG)]
    source_cmd=ssh+['teal-rat-80.coder',shlex.join(['python3','-c',SOURCE,token])]
    dest_cmd=ssh+[args.host+'.coder',shlex.join(['python3','-c',DESTINATION,token,dest,pin])]
    started=time.time()
    with (out/'source.stderr').open('wb') as source_err,(out/'destination.stderr').open('wb') as dest_err,(out/'destination.stdout').open('wb') as dest_out:
        source=subprocess.Popen(source_cmd,stdout=subprocess.PIPE,stderr=source_err)
        destination=subprocess.Popen(dest_cmd,stdin=source.stdout,stdout=dest_out,stderr=dest_err)
        source.stdout.close()
        (out/'processes.json').write_text(json.dumps({'source_pid':source.pid,'destination_pid':destination.pid,'started':started,'manifest_sha256':pin})+'\n')
        dest_rc=destination.wait();source_rc=source.wait()
    result={'host':args.host,'source_exit':source_rc,'destination_exit':dest_rc,'seconds':time.time()-started,'manifest_sha256':pin,'remote_runtime':dest}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    print((out/'destination.stdout').read_text(),end='')
    if source_rc or dest_rc:
        print((out/'source.stderr').read_text(),file=sys.stderr)
        print((out/'destination.stderr').read_text(),file=sys.stderr)
        raise SystemExit(1)

if __name__=='__main__':main()
