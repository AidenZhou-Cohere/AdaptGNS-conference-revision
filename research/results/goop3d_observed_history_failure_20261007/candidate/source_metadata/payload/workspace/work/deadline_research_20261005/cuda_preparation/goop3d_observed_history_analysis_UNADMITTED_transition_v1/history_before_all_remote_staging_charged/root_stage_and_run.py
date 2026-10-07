"""Preassembled control staging and one exact pipeline launch in the first100s."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
from observed_common import *
from phase_binding import STATE,load_original_phase
from root_prepare import package
from root_run import route_environment
from operational_checks import local_absence
import root_stage

HERE=Path(__file__).resolve().parent

def run_argv(python,package_pin,phase_pin,anchor_pin,release_pin,ready_path,ready_pin):
    return [python,'-B',str(HERE/'root_run.py'),'--root-run','--package-sha256',package_pin,'--phase-sha256',phase_pin,'--anchor-sha256',anchor_pin,'--release-sha256',release_pin,'--route-ready',str(ready_path),'--route-ready-sha256',ready_pin]

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-start',action='store_true')
    for n in ('package-sha256','phase-sha256','anchor-sha256','release-sha256'):p.add_argument('--'+n)
    p.add_argument('--route-directory',type=Path);args=p.parse_args()
    if not args.root_start:print(json.dumps({'status':'inert_control_stage_and_original_launch','clock_granted':False}));return
    package(args.package_sha256);phase=load_original_phase(args.phase_sha256,args.anchor_sha256)
    ready_path=args.route_directory/'ready.json';ready_raw=read_bound(ready_path,None,CAP_JSON);ready_pin=digest(ready_raw);route=strict(ready_raw);route_environment(route,phase)
    read_bound(STATE/'pipeline.cpu_release.json',args.release_sha256,CAP_JSON)
    payload_path=STATE/'control_payload.json';payload_raw=read_bound(payload_path,None,CAP_JSON);payload_pin=digest(payload_raw)
    need(strict(payload_raw)['kind']=='controls','only fixed current controls')
    directory=STATE/'controls_stage';need(not directory.exists() and not (STATE/'pipeline.transport').exists(),'no previous stage or pipeline attempt')
    argv=['root_stage.py','--root-stage','--package-sha256',args.package_sha256,'--payload',str(payload_path),'--payload-sha256',payload_pin,'--output',str(directory),'--phase-sha256',args.phase_sha256,'--anchor-sha256',args.anchor_sha256,'--route-ready',str(ready_path),'--route-ready-sha256',ready_pin]
    previous=sys.argv;captured=io.StringIO()
    try:
        sys.argv=argv
        with contextlib.redirect_stdout(captured):root_stage.main()
    finally:sys.argv=previous
    actual=captured.getvalue();result=strict(actual)
    with (directory/'controller_stdout').open('x') as stream:stream.write(actual)
    transport=strict(read_bound(directory/'transport.json',result['transport_sha256'],CAP_JSON))
    need(result['status']=='staging_returned_requires_root_review' and transport['exit_code']==0 and transport['failure'] is None and transport['cleanup_errors']==[] and transport['signals_to_own_new_local_transport_group']==[] and transport['local_transport_reaped'] is True and transport['pre_publication_local_budget_preserved'] is True,'complete bounded original control staging')
    payload=strict(payload_raw);need(transport['remote_report']['files_sha256']=={row['path']:row['sha256'] for row in payload['files']},'all exact controls staged')
    local=local_absence({transport['local_transport_pid']});phase.remaining()
    publish(STATE/'control_stage_admission.json',{'schema':'goop3d_control_stage_local_admission_v1','phase_sha256':args.phase_sha256,'payload_sha256':payload_pin,'transport_sha256':result['transport_sha256'],'local_native_closure':local,'source_manifest_sha256':args.package_sha256,'scientific_admission':False})
    command=run_argv(sys.executable,args.package_sha256,args.phase_sha256,args.anchor_sha256,args.release_sha256,ready_path,ready_pin)
    # Replace this same tool process. Its sole stdout is the original pipeline
    # transport result, so actual tool launch/completion envelopes stay genuine.
    os.execve(sys.executable,command,os.environ.copy())

if __name__=='__main__':main()
