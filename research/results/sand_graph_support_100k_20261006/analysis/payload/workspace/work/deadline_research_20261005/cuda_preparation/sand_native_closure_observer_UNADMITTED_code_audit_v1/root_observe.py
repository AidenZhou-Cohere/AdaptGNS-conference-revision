#!/usr/bin/env python3
"""Root-only bounded transport for the exact Sand original native-closure metadata observer; no science.

No work without --root-observe. Every attempt uses a fresh output directory.
Only this invocation's newly created local SSH process group may be cleaned up.
"""
from pathlib import Path
import argparse, datetime as D, hashlib, json, os, selectors, shlex, signal, subprocess, time
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
PREP = HERE.parent
SSH_PIN = 'f04af81d2b926b04f7a8063df5c8535da1d8c9aa637e15cf235c261d6d0725b2'
PYTHON = '/root/repos/AdaptGNS-cuda-20261006/.venv/bin/python'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return D.datetime.now(D.timezone.utc)
def put(path,value):
    with Path(path).open('x') as stream: json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
def need(ok,message):
    if not ok: raise ValueError(message)

def command(source, contract_pin, role):
    remote = ['/usr/bin/timeout','--signal=KILL','20s',PYTHON,'-I','-S','-B','-c',source,'--observe','--contract-sha256',contract_pin]
    return ['ssh','-T','-F',str(PREP/'ssh_config'),'-o','ConnectTimeout=8','-o','ConnectionAttempts=1','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=1',{'A':'yellow-worm-77.coder','B':'aquamarine-toad-75.coder'}[role],shlex.join(remote)]

def proxy_environment(value):
    env=os.environ.copy()
    if value is not None:
        parsed=urlsplit(value)
        need(parsed.scheme=='http' and parsed.hostname=='127.0.0.1' and parsed.username is None and parsed.password is None and parsed.path=='' and not parsed.query and not parsed.fragment and parsed.port is not None and 0<parsed.port<65536 and value=='http://127.0.0.1:'+str(parsed.port),'exact uncredentialed loopback HTTP proxy required')
        env['HTTPS_PROXY']=value;env['https_proxy']=value
        env.pop('NO_PROXY',None);env.pop('no_proxy',None)
    return env

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--role',choices=('A','B')); parser.add_argument('--root-observe',action='store_true'); parser.add_argument('--package-sha256'); parser.add_argument('--output',type=Path)
    parser.add_argument('--https-proxy',help='Optional root-owned ephemeral http://127.0.0.1:PORT proxy; applied only to SSH subprocess')
    args=parser.parse_args()
    if not args.root_observe:
        print(json.dumps({'status':'inert_metadata_observer','scientific_execution':False,'clock_issued':False})); return
    started=utc(); tick=time.monotonic(); overall=35
    def remaining(): return min(overall-(time.monotonic()-tick),overall-(utc()-started).total_seconds())
    def check():
        elapsed=time.monotonic()-tick; wall=(utc()-started).total_seconds()
        need(0 <= elapsed and 0 <= wall and abs(wall-elapsed) <= 5 and remaining()>5,'local observation bound/clock disagreement')
    check(); need(args.role in ('A','B'),'explicit role required'); need(args.output and args.output.is_absolute() and args.output.resolve()==args.output and not args.output.exists(),'fresh canonical output directory required')
    manifest=HERE/'manifest.json'; need(args.package_sha256 and sha(manifest)==args.package_sha256,'exact independently reviewed package required')
    files=json.loads(manifest.read_bytes())['files_sha256']
    for name,pin in files.items():
        path=HERE/name; need(path.resolve().is_relative_to(HERE) and path.is_file() and not path.is_symlink() and sha(path)==pin,'package bytes differ'); check()
    need(sha(PREP/'ssh_config')==SSH_PIN,'original SSH configuration changed'); check()
    contract=(HERE/('contract_'+args.role+'.json')).read_bytes(); source_bytes=(HERE/'observe_metadata.py').read_bytes()
    contract_pin=hashlib.sha256(contract).hexdigest(); source_pin=hashlib.sha256(source_bytes).hexdigest()
    need(contract_pin==files['contract_'+args.role+'.json'] and source_pin==files['observe_metadata.py'],'exact in-memory payload bytes changed')
    source=source_bytes.decode();argv=command(source,contract_pin,args.role);child_env=proxy_environment(args.https_proxy)
    args.output.mkdir()
    put(args.output/'intent.json',{'schema':'sand_original_native_closure_metadata_observation_intent_v1','package_sha256':args.package_sha256,'role':args.role,'original_session':{'A':51595,'B':12559}[args.role],'contract_sha256':contract_pin,'observer_source_sha256':source_pin,'started_utc':started.isoformat(),'started_monotonic':tick,'whole_local_observation_seconds':overall,'remote_outer_timeout_seconds':20,'remote_internal_read_seconds':12,'argv':argv,'ephemeral_proxy_applied_to_ssh_child_only':args.https_proxy is not None,'scope':'original registered native absence and filename availability only; never original transport exit or admission; no target signals or scientific contents','new_analysis_clock_or_release':False})
    out=bytearray(); err=bytearray(); proc=None; failure=None; signals=[]; expired=False; cleanup_errors=[]
    def cleanup():
        if proc is None or proc.poll() is not None:return
        for sig,seconds in ((signal.SIGTERM,2),(signal.SIGKILL,1)):
            if proc.poll() is not None:return
            try:os.killpg(proc.pid,sig);signals.append(signal.Signals(sig).name)
            except ProcessLookupError:pass
            except OSError as exc:cleanup_errors.append(type(exc).__name__+': '+str(exc));return
            try:proc.wait(timeout=max(0,min(seconds,remaining()-2)))
            except subprocess.TimeoutExpired:pass
    try:
        check()
        proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env=child_env)
        # Contract is small and the helper begins reading it immediately; a
        # nonblocking stdin prevents stalled DNS/SSH from blocking publication.
        for stream in (proc.stdin,proc.stdout,proc.stderr):os.set_blocking(stream.fileno(),False)
        sel=selectors.DefaultSelector(); sel.register(proc.stdin,selectors.EVENT_WRITE,'stdin'); sel.register(proc.stdout,selectors.EVENT_READ,'stdout'); sel.register(proc.stderr,selectors.EVENT_READ,'stderr'); offset=0
        try:
            while sel.get_map():
                check()
                for key,_ in sel.select(timeout=min(.1,max(0,remaining()-5))):
                    if key.data=='stdin':
                        try:n=os.write(key.fileobj.fileno(),contract[offset:offset+65536]);offset+=n
                        except BrokenPipeError:offset=len(contract)
                        except BlockingIOError:continue
                        if offset==len(contract):sel.unregister(key.fileobj);key.fileobj.close()
                    else:
                        try:block=os.read(key.fileobj.fileno(),65536)
                        except BlockingIOError:continue
                        if not block:sel.unregister(key.fileobj);key.fileobj.close();continue
                        target=out if key.data=='stdout' else err;target.extend(block)
                        need(len(target)<=(1250000 if key.data=='stdout' else 262144),'bounded metadata transport output exceeded')
            proc.wait(timeout=max(0,min(1,remaining()-4)));check()
        finally:sel.close()
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc);expired=remaining()<=5
    finally:
        cleanup()
        if proc is not None:proc.poll()
        code=None if proc is None else proc.returncode
        if proc is not None:
            for stream in (proc.stdin,proc.stdout,proc.stderr):
                if stream is not None:
                    try:stream.close()
                    except OSError as exc:cleanup_errors.append(type(exc).__name__+': '+str(exc))
        for name,raw in (('stdout',out),('stderr',err)):
            with (args.output/name).open('xb') as stream:stream.write(raw)
        result={'schema':'sand_original_native_closure_metadata_transport_v1','status':'remote_observation_unavailable','role':args.role,'original_session':{'A':51595,'B':12559}[args.role],'started_utc':started.isoformat(),'observed_utc':utc().isoformat(),'elapsed_monotonic_seconds':time.monotonic()-tick,'elapsed_wall_seconds':(utc()-started).total_seconds(),'exit_code':code,'local_transport_pid':None if proc is None else proc.pid,'local_transport_reaped':code is not None,'local_transport_descendant_closure_established':False,'signals_to_own_new_local_transport_group':signals,'cleanup_errors':cleanup_errors,'failure':failure,'local_bound_reached':expired,'stdout_sha256':hashlib.sha256(out).hexdigest(),'stderr_sha256':hashlib.sha256(err).hexdigest(),'scientific_admission':False,'new_analysis_clock_or_release':False,'remote_execution_success_established':False,'remote_native_closure_established':False}
        if failure is None and code==0 and not signals and not cleanup_errors:
            try:
                # Import is local source-only and invokes no Runtime or clocks.
                from observe_metadata import strict
                report=strict(out);need(report['schema']=='sand_original_native_closure_metadata_observation_v1' and report['role']==args.role and report['original_session']=={'A':51595,'B':12559}[args.role] and report['scientific_admission'] is False and report['new_analysis_clock_or_release'] is False and report['remote_execution_success_established'] is False,'unexpected observer response')
                result.update(status='metadata_observation_returned_requires_root_review',remote_report=report)
            except BaseException as exc:result['response_error']=type(exc).__name__+': '+str(exc)
        result['pre_publication_local_budget_preserved']=0<=result['elapsed_monotonic_seconds']<overall and 0<=result['elapsed_wall_seconds']<overall and abs(result['elapsed_monotonic_seconds']-result['elapsed_wall_seconds'])<=5
        put(args.output/'transport.json',result)
    need(remaining()>0 and abs((utc()-started).total_seconds()-(time.monotonic()-tick))<=5,'whole observation bound exceeded during publication')
    print(json.dumps({'status':result['status'],'output':str(args.output),'transport_sha256':sha(args.output/'transport.json'),'scientific_admission':False}))

if __name__=='__main__':main()
