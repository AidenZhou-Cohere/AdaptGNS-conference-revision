"""One bounded exact stopped-product transfer after actual pipeline tool exit."""
import argparse
import datetime as D
from pathlib import Path
import os
import shlex
import signal
import subprocess
import tarfile
import time
from observed_common import *
from phase_binding import STATE,load_original_phase
from root_prepare import package
from root_run import proxy_environment
from operational_checks import completed_tool,local_absence

HERE=Path(__file__).resolve().parent
PREP=HERE.parent
NAMES=('audit.json','summary.json','arithmetic_check.json','pipeline_receipt.json','owner_started.json','child_registered.json','owner_terminal.json','collection.json')

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-collect',action='store_true')
    for n in ('package-sha256','phase-sha256','anchor-sha256','release-sha256','original-tools-sha256','route-ready-sha256'):p.add_argument('--'+n)
    p.add_argument('--original-tools',type=Path);p.add_argument('--route-ready',type=Path);args=p.parse_args()
    if not args.root_collect:print(json.dumps({'status':'inert_exact_product_collector'}));return
    package(args.package_sha256);phase=load_original_phase(args.phase_sha256,args.anchor_sha256)
    release=strict(read_bound(STATE/'pipeline.cpu_release.json',args.release_sha256,CAP_JSON))
    original_raw=read_bound(args.original_tools,args.original_tools_sha256,CAP_JSON);original=strict(original_raw);session,done=completed_tool(original)
    transport_path=STATE/'pipeline.transport/transport.json';transport=strict(transport_path.read_bytes())
    need(strict(done['output'])==transport and transport['exit_code']==0 and transport['failure'] is None and transport['signals_to_new_local_group_only']==[] and transport['phase_sha256']==args.phase_sha256 and transport['release_sha256']==args.release_sha256,'actual pipeline tool exit binds recorded transport')
    before_local=local_absence({transport['local_transport_pid']})
    route=strict(read_bound(args.route_ready,args.route_ready_sha256,CAP_JSON));need(route['schema']=='coder_goop3d_observed_phase_route_proxy_v1' and all(route.get(k)==v for k,v in phase.metadata.items()),'same existing phase route')
    need(route['authority']=='coder.internal.cohere.com:443' and route['upstream']==['100.106.33.61',443] and route['opaque_tls'] is True and route['no_settings_changed'] is True,'exact opaque route')
    env=proxy_environment(route['url']);phase.remaining()
    remaining=min(phase.start+3380-time.time(),phase.mono_start+3380-time.monotonic())
    need(remaining>20,'collection reserve exhausted');seconds=min(360,int(remaining)-5)
    hard=int((release['clock_sample']['host_monotonic_seconds']+3420-5)*1e9)
    remote=['/usr/bin/timeout','--signal=KILL',str(seconds)+'s',REMOTE+'/.venv/bin/python','-I','-S','-B',SOURCE+'/collect_observed_products.py','--collect','--deadline-monotonic-ns',str(hard),'--phase-sha256',args.phase_sha256,'--release-sha256',args.release_sha256,'--collector-sha256',digest((HERE/'collect_observed_products.py').read_bytes())]
    read_bound(PREP/'ssh_config','f04af81d2b926b04f7a8063df5c8535da1d8c9aa637e15cf235c261d6d0725b2',CAP_JSON)
    argv=['ssh','-T','-F',str(PREP/'ssh_config'),'-o','ConnectTimeout=8','-o','ConnectionAttempts=1','teal-rat-80.coder',shlex.join(remote)]
    directory=STATE/'product_collection';need(not directory.exists(),'one fresh product collection only');directory.mkdir()
    publish(directory/'intent.json',{'phase_sha256':args.phase_sha256,'release_sha256':args.release_sha256,'original_pipeline_tool_session':session,'original_pipeline_tools_sha256':args.original_tools_sha256,'prior_local_transport_closure':before_local,'argv':argv,'remote_outer_seconds':seconds,'remote_absolute_deadline_ns':hard})
    (directory/'pipeline_original_tools.json').write_bytes(original_raw)
    proc=None;failure=None;signals=[]
    try:
        with (directory/'payload.tar.gz').open('xb') as out,(directory/'stderr').open('xb') as err:
            proc=subprocess.Popen(argv,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
            while proc.poll() is None:
                phase.remaining();need(time.time()<phase.start+3400 and time.monotonic()<phase.mono_start+3400,'fixed local collection deadline')
                need(out.tell()<=1200<<20 and err.tell()<=1<<20,'bounded collection transport')
                time.sleep(.1)
        need(proc.returncode==0,'remote product collection failed')
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        if proc is not None and proc.poll() is None:
            for sig,wait in ((signal.SIGTERM,2),(signal.SIGKILL,1)):
                if proc.poll() is not None:break
                try:os.killpg(proc.pid,sig);signals.append(signal.Signals(sig).name)
                except ProcessLookupError:pass
                try:proc.wait(timeout=wait)
                except subprocess.TimeoutExpired:pass
        tr={'schema':'goop3d_observed_product_collection_transport_v1','exit_code':None if proc is None else proc.poll(),'local_transport_pid':None if proc is None else proc.pid,'failure':failure,'signals_to_new_local_group_only':signals,'phase_sha256':args.phase_sha256,'release_sha256':args.release_sha256,'finished_utc':D.datetime.now(D.timezone.utc).isoformat(),'scientific_admission':False}
        publish(directory/'transport.json',tr)
    need(failure is None and tr['exit_code']==0 and not signals,'incomplete collection transport')
    phase.remaining();seen=set();files={}
    with tarfile.open(directory/'payload.tar.gz','r:gz') as tar:
        for item in tar:
            need(time.time()<phase.start+3420 and time.monotonic()<phase.mono_start+3420,'fixed local unpack deadline')
            need(item.name in NAMES and item.name not in seen and item.isfile() and item.size<=CAP_OUTPUT,'exact bounded unique archive outputs')
            raw=tar.extractfile(item).read(CAP_OUTPUT+1);need(len(raw)==item.size,'complete archive member')
            with (directory/item.name).open('xb') as f:f.write(raw)
            seen.add(item.name);files[item.name]=digest(raw)
    need(seen==set(NAMES),'complete exact eight-member product envelope')
    report=strict((directory/'collection.json').read_bytes())
    need(report['schema']=='goop3d_observed_products_collection_v1' and report['status']=='four_products_and_owner_closure_collected' and report['phase_sha256']==args.phase_sha256 and report['release_sha256']==args.release_sha256,'typed original collection receipt')
    need({**report['product_sha256'],**report['registry_sha256']}=={k:v for k,v in files.items() if k!='collection.json'},'exact collected product/registry hashes')
    local=local_absence({tr['local_transport_pid'],transport['local_transport_pid']})
    receipt={'schema':'goop3d_observed_local_collection_receipt_v1','status':'collected_requires_independent_semantic_review','phase_sha256':args.phase_sha256,'release_sha256':args.release_sha256,'files_sha256':files,'original_pipeline_tools_sha256':args.original_tools_sha256,'original_pipeline_tool_session':session,'local_native_closure':local,'transport_sha256':digest((directory/'transport.json').read_bytes()),'scientific_admission':False}
    pin=publish(directory/'local_receipt.json',receipt);print(json.dumps({'status':receipt['status'],'local_receipt_sha256':pin,'scientific_admission':False}))

if __name__=='__main__':main()
