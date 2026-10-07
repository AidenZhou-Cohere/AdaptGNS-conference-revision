"""Collect a closed job with its explicitly expected exit; preserve exact bytes."""
import argparse,hashlib,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
REMOTE=r'''
import hashlib,io,json,os,socket,sys,tarfile,time
from pathlib import Path
req=json.loads(sys.stdin.read());base=Path('/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007')
assert socket.gethostname()==req['hostname']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==req['boot_id']
job=base/'jobs'/req['job'];status=json.loads((job/'status.json').read_text());assert status['status']=='exited' and status['returncode']==req['expected_exit']
native=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
  if int(f[2])==status['owner']['pgid']:native.append({'pid':int(proc.name),'state':f[0],'start_ticks':int(f[19]),'pgid':int(f[2])})
 except (FileNotFoundError,ProcessLookupError):pass
assert all(p['state']=='Z' for p in native),native
files={}
for relative in ['jobs/'+req['job']]+req['outputs']:
 p=base/relative;assert p.resolve().is_relative_to(base) and p.is_dir() and not p.is_symlink()
 for item in sorted(p.rglob('*')):
  assert not item.is_symlink()
  if item.is_file():
   rel=str(item.relative_to(base));assert rel not in files
   h=hashlib.sha256()
   with item.open('rb') as stream:
    for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
   files[rel]={'bytes':item.stat().st_size,'sha256':h.hexdigest()}
manifest={'host':socket.gethostname(),'boot_id':req['boot_id'],'job':req['job'],'actual_status':status,'remaining_group_members':native,'collection_utc_seconds':time.time(),'files':files}
raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 info=tarfile.TarInfo('transfer_manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 for rel in files:tar.add(base/rel,arcname=rel,recursive=False)
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--outputs',nargs='+',required=True);p.add_argument('--expected-exit',type=int,default=0);a=p.parse_args();a.dest.mkdir(parents=True,exist_ok=False)
 s=json.loads(a.spec.read_text());req={k:s[k] for k in ['hostname','boot_id']};req.update(job=s['job_id'],outputs=a.outputs,expected_exit=a.expected_exit)
 cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(ROOT/'work/deadline_research_20261005/cuda_preparation/ssh_config'),s['host_alias']+'.coder',shlex.join(['python3','-c',REMOTE])]
 archive=a.dest/'original_archive.tar.gz'
 with archive.open('xb') as stdout:r=subprocess.run(cmd,input=json.dumps(req).encode(),stdout=stdout,stderr=subprocess.PIPE)
 receipt={'exit_code':r.returncode,'stderr':r.stderr.decode(errors='replace'),'request':req,'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()}
 (a.dest/'original_transfer.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
 dest=a.dest/'files';dest.mkdir();members=[]
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   target=dest/member.name;assert member.isfile() and target.resolve().is_relative_to(dest.resolve()) and not target.exists()
   target.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(member) as src,target.open('xb') as out:
    for block in iter(lambda:src.read(1<<20),b''):out.write(block)
   members.append(member.name)
 manifest=json.loads((dest/'transfer_manifest.json').read_text());assert set(members)==set(manifest['files'])|{'transfer_manifest.json'}
 for rel,pin in manifest['files'].items():
  target=dest/rel;assert target.stat().st_size==pin['bytes'];assert hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==pin['sha256']
 result={'verified_files':len(manifest['files']),'verified_bytes':sum(x['bytes'] for x in manifest['files'].values()),'manifest_sha256':hashlib.sha256((dest/'transfer_manifest.json').read_bytes()).hexdigest(),'actual_child_returncode':manifest['actual_status']['returncode'],'remaining_group_members':manifest['remaining_group_members']}
 (a.dest/'verified.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
