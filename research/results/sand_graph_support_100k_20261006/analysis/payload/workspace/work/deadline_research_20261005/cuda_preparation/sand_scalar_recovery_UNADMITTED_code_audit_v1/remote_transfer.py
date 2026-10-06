"""One exact Sand B scalar, streamed to a fresh retained partial then exclusive publication."""
import base64,datetime as D,hashlib,json,os,pathlib,sys,time
R='/root/repos/AdaptGNS-cuda-20261006';A=R+'/sand_final_analysis_20261006_v1'
HOST='aidenzhou-yellow-worm-77-78fff65d5-62zgv';BOOT='cbe68c69-6a30-47f9-a9f8-b42a80785a1f'
PH='0ac7a368dd8afa921c8b973c5a8a760af7f10eb9f3413bceb592fff0c0ad7a5c'
def need(v,m):
    if not v:raise ValueError(m)
def main():
    import socket
    p=json.loads(base64.b64decode(sys.argv[1],validate=True));need(not sys.flags.optimize and p['phase_sha256']==PH,'Exact original phase required')
    stop=D.datetime.fromisoformat(p['stop_utc']);need(stop.isoformat()=='2026-10-06T23:33:01.226132+00:00','Original stop required');mono_stop=time.monotonic()+(stop-D.datetime.now(D.timezone.utc)).total_seconds()
    def check():need(D.datetime.now(D.timezone.utc)+D.timedelta(seconds=8)<stop and time.monotonic()+8<mono_stop,'Original phase exhausted')
    def ordinary(x):need(x.is_absolute() and x.resolve()==x and not x.is_symlink(),'Canonical ordinary path required')
    def digest(x):
        ordinary(x);need(x.is_file(),'Regular scalar required');before=x.stat();h=hashlib.sha256()
        with x.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):check();h.update(block)
        after=x.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Scalar changed while hashing');return {'sha256':h.hexdigest(),'bytes':before.st_size}
    check();need(socket.gethostname()==HOST and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'Original A host and boot')
    phase=pathlib.Path(A+'/controls/analysis_phase.json');ordinary(phase);need(hashlib.sha256(phase.read_bytes()).hexdigest()==PH,'Remote original phase differs')
    target=pathlib.Path(p['target']);need(str(target) in (A+'/B.stopped_collection.json',A+'/B.saved_array_audit.json'),'Exact B scalar target required');ordinary(target)
    expected={'sha256':p['sha256'],'bytes':p['bytes']};need(type(p['bytes']) is int and 0<p['bytes']<2**31 and type(p['sha256']) is str and len(p['sha256'])==64 and all(c in '0123456789abcdef' for c in p['sha256']),'Exact bounded scalar hash and length')
    if target.exists():
        need(digest(target)==expected,'Existing scalar bytes conflict and are preserved');need(p['observed_fate']=='exact','Target changed after original fate observation');status='reused_exact_existing_bytes';partial=None
    else:
        need(p['observed_fate']=='absent','Target disappeared after original fate observation')
        parent=pathlib.Path(A+'/scalar_transfer_recovery1');ordinary(parent);parent.mkdir(exist_ok=True);need(parent.is_dir(),'Recovery parent must be directory');partial=parent/(target.name+'.partial');ordinary(partial)
        h=hashlib.sha256();n=0
        with partial.open('xb') as f:
            while n<p['bytes']:
                check();block=sys.stdin.buffer.read(min(1<<20,p['bytes']-n));need(block,'Truncated original scalar stream');n+=len(block);h.update(block);f.write(block)
            check();need(sys.stdin.buffer.read(1)==b'','Unexpected bytes beyond exact scalar length');need(n==p['bytes'] and h.hexdigest()==p['sha256'],'Scalar stream hash differs');f.flush();os.fsync(f.fileno())
        check();need(digest(partial)==expected,'Retained partial bytes differ');os.link(partial,target,follow_symlinks=False);status='published_exact_stream_exclusively'
    check();need(digest(target)==expected and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'Final scalar or original boot changed')
    print(json.dumps({'schema':'sand_original_scalar_stream_transfer_v1','status':status,'phase_sha256':PH,'hostname':HOST,'host_boot_id':BOOT,'target':str(target),'sha256':p['sha256'],'bytes':p['bytes'],'retained_partial':str(partial) if partial else None,'checked_utc':D.datetime.now(D.timezone.utc).isoformat(),'no_original_files_overwritten_or_deleted':True,'no_scientific_program_run':True},sort_keys=True))
if __name__=='__main__':main()
