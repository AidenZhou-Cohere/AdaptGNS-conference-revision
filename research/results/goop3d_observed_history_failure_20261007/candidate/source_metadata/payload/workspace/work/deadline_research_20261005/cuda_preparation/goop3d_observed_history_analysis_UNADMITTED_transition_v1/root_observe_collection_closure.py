"""One bounded post-collection metadata observation after actual collector exit."""
import argparse
import datetime as D
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from observed_common import *
from phase_binding import STATE,load_original_phase
from root_prepare import package
from root_run import route_environment
from operational_checks import completed_tool,local_absence
from observe_collection_closure import expected_identities

HERE=Path(__file__).resolve().parent
PREP=HERE.parent

def validate_closure(report,collected,phase_pin,collection_pin):
    identities,expected=expected_identities(collected)
    need(report['schema']=='goop3d_observed_postcollection_native_closure_v1' and report['status']=='all47_original_pids_absent_twice' and report['phase_sha256']==phase_pin and report['collection_report_sha256']==collection_pin and report['release_sha256']==collected['release_sha256'],'exact post-collection closure binding')
    need(report['host']==HOST and report['boot_id']==BOOT and report['original_identities']==identities,'exact original host/boot and five native start identities')
    for key in ('native_first_absent','native_second_absent'):
        need(report[key]=={str(pid):True for pid in expected},'all47 original native PIDs absent twice')
    need(stamp(report['finished_utc'])>=stamp(collected['finished_utc']) and report['no_files_written'] is True and report['scientific_program_run'] is False and report['original_result_queue_opened'] is False,'metadata-only closure follows collection')
    return expected

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-observe',action='store_true')
    for name in ('package-sha256','phase-sha256','anchor-sha256','collection-receipt-sha256','collection-tools-sha256','route-ready-sha256'):p.add_argument('--'+name)
    p.add_argument('--collection-tools',type=Path);p.add_argument('--route-ready',type=Path);args=p.parse_args()
    if not args.root_observe:print(json.dumps({'status':'inert_postcollection_observer','clock_granted':False}));return
    package(args.package_sha256);phase=load_original_phase(args.phase_sha256,args.anchor_sha256)
    def check():
        phase.remaining();need(time.time()<phase.start+3480 and time.monotonic()<phase.mono_start+3480,'fixed local post-collection closure cutoff')
    check();need(time.time()<phase.start+3440 and time.monotonic()<phase.mono_start+3440,'full post-collection observation reserve required')
    collected_dir=STATE/'product_collection';receipt=strict(read_bound(collected_dir/'local_receipt.json',args.collection_receipt_sha256,CAP_JSON,check))
    original_raw=read_bound(args.collection_tools,args.collection_tools_sha256,CAP_JSON,check);session,done=completed_tool(strict(original_raw))
    result=strict(done['output']);need(result['local_receipt_sha256']==args.collection_receipt_sha256 and result['scientific_admission'] is False,'actual original collection tool exit before new observation')
    tr=strict(read_bound(collected_dir/'transport.json',receipt['transport_sha256'],CAP_JSON,check))
    need(tr['exit_code']==0 and tr['failure'] is None and tr['signals_to_new_local_group_only']==[],'complete original collector transport')
    prior_local=local_absence({tr['local_transport_pid']})
    collection_pin=receipt['files_sha256']['collection.json'];raw=read_bound(collected_dir/'collection.json',collection_pin,CAP_JSON,check);collected=strict(raw)
    expected_identities(collected);need(collected['phase_sha256']==args.phase_sha256,'same-phase original collection')
    route=strict(read_bound(args.route_ready,args.route_ready_sha256,CAP_JSON,check));env=route_environment(route,phase)
    read_bound(PREP/'ssh_config','f04af81d2b926b04f7a8063df5c8535da1d8c9aa637e15cf235c261d6d0725b2',CAP_JSON,check)
    phase_doc=strict(read_bound(STATE/'analysis_phase.json',args.phase_sha256,CAP_JSON,check));hard=int((phase_doc['clock_sample']['host_monotonic_seconds']+3480-5)*1e9)
    remote=['/usr/bin/timeout','--signal=KILL','20s',REMOTE+'/.venv/bin/python','-I','-S','-B',SOURCE+'/observe_collection_closure.py','--observe','--deadline-monotonic-ns',str(hard),'--phase-sha256',args.phase_sha256,'--report-sha256',collection_pin,'--observer-sha256',digest((HERE/'observe_collection_closure.py').read_bytes())]
    argv=['ssh','-T','-F',str(PREP/'ssh_config'),'-o','ConnectTimeout=8','-o','ConnectionAttempts=1','teal-rat-80.coder',shlex.join(remote)]
    directory=STATE/'postcollection_closure';need(not directory.exists(),'one fresh post-collection observer only');directory.mkdir()
    publish(directory/'intent.json',{'phase_sha256':args.phase_sha256,'collection_receipt_sha256':args.collection_receipt_sha256,'collection_tools_sha256':args.collection_tools_sha256,'original_collection_session':session,'prior_local_native_closure':prior_local,'argv':argv,'collection_report_sha256':collection_pin})
    (directory/'collection_original_tools.json').write_bytes(original_raw)
    proc=None;out=b'';err=b'';failure=None;signals=[];started=D.datetime.now(D.timezone.utc).isoformat();tick=time.monotonic()
    try:
        check();proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True)
        out,err=proc.communicate(input=raw,timeout=25)
        need(len(out)<=1<<20 and len(err)<=1<<20,'bounded closure metadata output');check()
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        if proc is not None and proc.poll() is None:
            for sig,wait in ((signal.SIGTERM,2),(signal.SIGKILL,1)):
                if proc.poll() is not None:break
                try:os.killpg(proc.pid,sig);signals.append(signal.Signals(sig).name)
                except ProcessLookupError:pass
                try:out,err=proc.communicate(timeout=wait)
                except subprocess.TimeoutExpired:pass
        for name,value in (('stdout',out),('stderr',err)):
            with (directory/name).open('xb') as f:f.write(value)
        transport={'schema':'goop3d_postcollection_closure_transport_v1','phase_sha256':args.phase_sha256,'collection_report_sha256':collection_pin,'exit_code':None if proc is None else proc.poll(),'local_transport_pid':None if proc is None else proc.pid,'failure':failure,'signals_to_new_local_group_only':signals,'started_utc':started,'finished_utc':D.datetime.now(D.timezone.utc).isoformat(),'elapsed_monotonic_seconds':time.monotonic()-tick,'scientific_admission':False}
        publish(directory/'transport.json',transport)
    check();need(failure is None and transport['exit_code']==0 and not signals and 0<=transport['elapsed_monotonic_seconds']<30,'complete bounded original post-collection observation')
    report=strict(out);validate_closure(report,collected,args.phase_sha256,collection_pin)
    need(report['native_guard']['absolute_deadline_ns']==hard,'original fixed observer native guard')
    pin=publish(directory/'report.json',report,check);local=local_absence({tr['local_transport_pid'],transport['local_transport_pid']});check()
    final={'schema':'goop3d_postcollection_closure_local_receipt_v1','status':'all47_remote_original_pids_absent_twice_requires_actual_tool_exit','phase_sha256':args.phase_sha256,'report_sha256':pin,'collection_receipt_sha256':args.collection_receipt_sha256,'collection_tools_sha256':args.collection_tools_sha256,'original_collection_session':session,'transport_sha256':digest((directory/'transport.json').read_bytes()),'local_native_closure':local,'scientific_admission':False}
    print(json.dumps({'status':final['status'],'local_receipt_sha256':publish(directory/'local_receipt.json',final,check),'scientific_admission':False}))

if __name__=='__main__':main()
