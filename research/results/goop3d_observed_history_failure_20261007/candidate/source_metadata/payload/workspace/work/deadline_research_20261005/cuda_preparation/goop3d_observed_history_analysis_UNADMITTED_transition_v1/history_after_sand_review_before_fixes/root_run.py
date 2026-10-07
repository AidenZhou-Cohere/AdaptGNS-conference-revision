"""One root-owned SSH invocation; closure and scientific admission remain external."""
import argparse
import datetime as D
from pathlib import Path
import os
import shlex
import signal
import subprocess
import time
from urllib.parse import urlsplit
from observed_common import *
from phase_binding import STATE,load_original_phase
from root_prepare import package

PREP=Path(__file__).resolve().parent.parent
PYTHON=REMOTE+'/.venv/bin/python'
def proxy_environment(value):
    parsed=urlsplit(value)
    need(parsed.scheme=='http' and parsed.hostname=='127.0.0.1' and parsed.username is None and parsed.password is None and parsed.path=='' and not parsed.query and not parsed.fragment and parsed.port is not None and 0<parsed.port<65536 and value=='http://127.0.0.1:'+str(parsed.port),'exact uncredentialed loopback proxy')
    env=os.environ.copy();env['HTTPS_PROXY']=value;env['https_proxy']=value;env.pop('NO_PROXY',None);env.pop('no_proxy',None);return env

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-run',action='store_true')
    for name in ('package-sha256','phase-sha256','anchor-sha256','release-sha256','route-ready-sha256'):p.add_argument('--'+name)
    p.add_argument('--route-ready',type=Path);args=p.parse_args()
    if not args.root_run:print(json.dumps({'status':'inert_observed_transport'}));return
    package(args.package_sha256);phase=load_original_phase(args.phase_sha256,args.anchor_sha256)
    release=strict(read_bound(STATE/'pipeline.cpu_release.json',args.release_sha256,CAP_JSON))
    need(release['analysis_phase']['sha256']==args.phase_sha256 and release['operation']=='d3_observed_history_pipeline','one exact current phase operation')
    route=strict(read_bound(args.route_ready,args.route_ready_sha256,CAP_JSON))
    need(route['schema']=='coder_goop3d_observed_phase_route_proxy_v1' and all(route.get(k)==v for k,v in phase.metadata.items()),'route binds this distinct phase and unchanged anchor')
    need(route['authority']=='coder.internal.cohere.com:443' and route['upstream']==['100.106.33.61',443] and route['opaque_tls'] is True and route['no_settings_changed'] is True,'exact opaque route')
    env=proxy_environment(route['url']);read_bound(PREP/'ssh_config','f04af81d2b926b04f7a8063df5c8535da1d8c9aa637e15cf235c261d6d0725b2',CAP_JSON)
    remote=[PYTHON,'-I','-S','-B',SOURCE+'/launch_observed_history_cpu_v1.py','--hard-deadline-monotonic-ns',str(release['hard_deadline_monotonic_ns']),'--release',OUTPUT+'/controls/pipeline.cpu_release.json','--release-sha256',args.release_sha256]
    argv=['ssh','-T','-F',str(PREP/'ssh_config'),'-o','ConnectTimeout=8','-o','ConnectionAttempts=1','teal-rat-80.coder',shlex.join(remote)]
    transport=STATE/'pipeline.transport';need(not transport.exists(),'one original transport only');transport.mkdir()
    phase.remaining();need(time.monotonic()<phase.mono_start+100 and time.time()<phase.start+100,'pipeline launch must fit fixed initial scope gate')
    need(signal.getsignal(signal.SIGCHLD)==signal.SIG_DFL,'default SIGCHLD required')
    publish(transport/'intent.json',{'phase_sha256':args.phase_sha256,'release_sha256':args.release_sha256,'argv':argv,'ephemeral_proxy_child_only':True,'original_phase_modified':False})
    proc=None;failure=None;signals=[];started=D.datetime.now(D.timezone.utc).isoformat()
    try:
        with (transport/'stdout').open('xb') as out,(transport/'stderr').open('xb') as err:
            proc=subprocess.Popen(argv,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
            while proc.poll() is None:
                phase.remaining();need(time.monotonic()<phase.mono_start+3020 and time.time()<phase.start+3020,'fixed transport closure deadline')
                need(out.tell()<=1<<20 and err.tell()<=1<<20,'bounded transport logs')
                time.sleep(.1)
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        if proc is not None and proc.poll() is None:
            for sig,wait in ((signal.SIGTERM,2),(signal.SIGKILL,1)):
                if proc.poll() is not None:break
                try:os.killpg(proc.pid,sig);signals.append(signal.Signals(sig).name)
                except ProcessLookupError:pass
                try:proc.wait(timeout=wait)
                except subprocess.TimeoutExpired:pass
        result={'schema':'goop3d_observed_history_local_transport_v1','started_utc':started,'finished_utc':D.datetime.now(D.timezone.utc).isoformat(),'exit_code':None if proc is None else proc.poll(),'local_transport_pid':None if proc is None else proc.pid,'failure':failure,'signals_to_new_local_group_only':signals,'local_transport_descendant_closure_established':False,'remote_native_closure_established':False,'scientific_admission':False,'phase_sha256':args.phase_sha256,'release_sha256':args.release_sha256}
        publish(transport/'transport.json',result)
    print(json.dumps(result));need(failure is None and result['exit_code']==0 and not signals,'original observed transport failed or interrupted')

if __name__=='__main__':main()
