"""Root-only bounded metadata observation of retained original Sand scalar files."""
import base64,datetime as D,hashlib,json,os,pathlib,stat,subprocess,sys,time
R='/root/repos/AdaptGNS-cuda-20261006';A=R+'/sand_final_analysis_20261006_v1'
HOST='aidenzhou-yellow-worm-77-78fff65d5-62zgv';BOOT='cbe68c69-6a30-47f9-a9f8-b42a80785a1f'
PH='0ac7a368dd8afa921c8b973c5a8a760af7f10eb9f3413bceb592fff0c0ad7a5c'
GPU=['GPU-b6685200-7eaf-b46c-89b4-53e8715b8e21','GPU-38f0a7dd-8a4a-a461-2de3-747ebc7a73c6','GPU-af93e07a-ed08-8d07-25c4-4a51969065e0','GPU-8e9d199c-b92d-57f0-57af-ffea9caeace7']
def need(v,m):
    if not v:raise ValueError(m)
def main():
    import socket
    p=json.load(sys.stdin);need(not sys.flags.optimize and p['phase_sha256']==PH,'Exact original phase required')
    stop=D.datetime.fromisoformat(p['stop_utc']);need(stop.isoformat()=='2026-10-06T23:33:01.226132+00:00','Original stop required');mono_stop=time.monotonic()+(stop-D.datetime.now(D.timezone.utc)).total_seconds()
    def check():need(D.datetime.now(D.timezone.utc)+D.timedelta(seconds=8)<stop and time.monotonic()+8<mono_stop,'Original phase exhausted')
    def ordinary(path):
        x=pathlib.Path(path);need(x.is_absolute() and x.resolve()==x and not x.is_symlink() and x.is_file(),'Canonical regular file required');return x
    def raw(path):
        check();x=ordinary(path);need(x.stat().st_size<8*1024*1024,'Metadata cap');b=x.read_bytes();check();return b
    def digest(path):
        check();x=ordinary(path);before=x.stat();h=hashlib.sha256()
        with x.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):check();h.update(block)
        after=x.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'File changed while hashed')
        return {'sha256':h.hexdigest(),'bytes':before.st_size}
    def absent(pids):
        need(all(type(i) is int and i>0 for i in pids),'Positive native PIDs');v={str(i):not pathlib.Path('/proc',str(i)).exists() for i in sorted(set(pids))};need(v and all(v.values()),'Original native process remains');return v
    def old_probes():
        need(p['original_probe_sha256']=='d420d740bcec4fefce4189cddb15722effce46c2cc82902f1fc8b1c125b0a415','Exact frozen old probe source required');matches=[]
        command_pins=p['original_failed_remote_command_sha256'];need(type(command_pins) is list and len(command_pins)==2 and all(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in command_pins),'Both preserved failed remote shell commands required');match_pins=set(command_pins)|{p['original_probe_sha256']}
        for d in pathlib.Path('/proc').iterdir():
            if not d.name.isdigit():continue
            check()
            try:
                cmd=(d/'cmdline').read_bytes();args=cmd.split(b'\0')
                if not any(hashlib.sha256(a).hexdigest() in match_pins for a in args):continue
                status=(d/'stat').read_text();tail=status[status.rfind(')')+2:].split()
                matches.append({'pid':int(d.name),'ppid':int(tail[1]),'pgid':int(tail[2]),'sid':int(tail[3]),'start_ticks':int(tail[19]),'cmdline_sha256':hashlib.sha256(cmd).hexdigest(),'executable':str((d/'exe').resolve()),'inline_source_sha256':p['original_probe_sha256']})
            except (FileNotFoundError,ProcessLookupError):continue
        need(not matches,'Original frozen probe/timeout still present; file fate not closed: '+json.dumps(matches,sort_keys=True));return matches
    check();need(socket.gethostname()==HOST and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'Original A host and boot')
    old_probes()
    need(hashlib.sha256(raw(A+'/controls/analysis_phase.json')).hexdigest()==PH,'Remote phase bytes differ')
    owner=A+'/owners/sand_saved_A';release=A+'/controls/sand_saved_A.cpu_release.json';rr=raw(release);need(hashlib.sha256(rr).hexdigest()==p['saved_A_release_sha256'],'Original saved_A release differs')
    release_doc=json.loads(rr);terminal_raw=raw(owner+'/owner_terminal.json');terminal=json.loads(terminal_raw);started=json.loads(raw(owner+'/owner_started.json'));registered=json.loads(raw(owner+'/child_registered.json'))
    need(terminal['schema']=='adaptgns_stopped_analysis_cpu_terminal_v1' and terminal['status']=='complete' and terminal['failure'] is None and terminal['release_sha256']==p['saved_A_release_sha256'] and terminal['child_native_absent'] is True,'Original saved_A terminal incomplete')
    child=terminal['child'];need(child['reaped'] is True and type(child['exit_code']) is int and child['exit_code']==0 and child['signals']==child['cleanup_errors']==[] and child['command']==release_doc['command'],'Original saved_A worker not cleanly complete')
    need(started['owner_identity']==terminal['owner_identity'] and started['outer_timeout_identity']==terminal['outer_timeout_identity'] and registered['identity']==child['identity'] and registered['owner_identity']==terminal['owner_identity'],'Original native identity chain differs')
    pids=p['historical_pids']+[terminal['outer_timeout_identity']['pid'],terminal['owner_identity']['pid'],child['pid']];first=absent(pids)
    publication=D.datetime.fromisoformat(terminal['publication_utc']);need(D.datetime.fromisoformat(release_doc['invocation_origin_utc'])<=publication<stop and terminal['native_hard_guard']['absolute_deadline_ns']==release_doc['hard_deadline_monotonic_ns'],'Original publication and guard bounds differ')
    ap=A+'/A.saved_array_audit.json';need(set(terminal['output_sha256'])=={ap},'Exact original saved_A scalar output only');saved=digest(ap);need(saved['sha256']==terminal['output_sha256'][ap],'Retained original saved_A scalar hash differs')
    wanted=p['B_targets'];need(set(wanted)=={A+'/B.stopped_collection.json',A+'/B.saved_array_audit.json'},'Exactly two B scalar targets')
    fates={}
    for path,expected in wanted.items():
        x=pathlib.Path(path);need(x.resolve()==x and not x.is_symlink(),'Canonical scalar fate path')
        if not x.exists():fates[path]={'state':'absent'}
        else:
            value=digest(path);fates[path]={'state':'exact' if value==expected else 'conflicting_retained_bytes',**value}
    mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=5)
    need({int(i.strip()):u.strip() for i,u in (r.split(',') for r in mapping.splitlines() if r.strip())}==dict(enumerate(GPU)),'Original A GPU mapping differs')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=5);need(not apps.strip(),'Assigned A GPUs nonempty')
    check();need(absent(pids)==first and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT,'Original native state changed')
    old_probes()
    need(hashlib.sha256(raw(owner+'/owner_terminal.json')).hexdigest()==hashlib.sha256(terminal_raw).hexdigest(),'Original terminal changed')
    print(json.dumps({'schema':'sand_original_scalar_recovery_fate_v1','status':'metadata_observed','phase_sha256':PH,'hostname':HOST,'host_boot_id':BOOT,'checked_utc':D.datetime.now(D.timezone.utc).isoformat(),'native_absent':first,'original_frozen_probe_and_timeout_absent_twice':True,'assigned_devices_empty':True,'saved_A_release_sha256':p['saved_A_release_sha256'],'saved_A_terminal_sha256':hashlib.sha256(terminal_raw).hexdigest(),'saved_A_product':saved,'B_targets':fates,'no_files_written':True,'no_scientific_program_run':True},sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
