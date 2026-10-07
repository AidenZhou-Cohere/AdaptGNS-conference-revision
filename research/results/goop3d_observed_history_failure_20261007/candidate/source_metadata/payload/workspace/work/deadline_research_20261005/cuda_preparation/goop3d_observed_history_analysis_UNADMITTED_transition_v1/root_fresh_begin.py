"""Assemble exact fresh operational evidence, then invoke the fixed phase issuer."""
import argparse
import contextlib
import datetime as D
import io
from pathlib import Path
import shlex
import sys
from observed_common import *
from operational_checks import completed_tool,local_absence
from root_prepare import package,fresh_report
import root_prepare
from staging_plan import source_payload
from prephase_sources import PROXY_OUTPUT,PROXY,check_sources

HERE=Path(__file__).resolve().parent
PREP=HERE.parent

def start_commands(python,package_pin,issued,proxy_directory,source_path,source_pin):
    common=['--package-sha256',package_pin,'--phase-sha256',issued['phase_sha256'],'--anchor-sha256',issued['anchor_sha256']]
    proxy=[python,'-B',str(HERE/'phase_proxy.py'),'--root-serve','--output',str(proxy_directory),'--phase-sha256',issued['phase_sha256'],'--anchor-sha256',issued['anchor_sha256']]
    pipeline=[python,'-B',str(HERE/'root_stage_and_run.py'),'--root-start',*common,'--release-sha256',issued['release_sha256'],'--control-payload-sha256',issued['control_payload_sha256'],'--source-payload',str(source_path),'--source-payload-sha256',source_pin,'--route-directory',str(proxy_directory)]
    return {'phase_proxy_argv':proxy,'stage_and_pipeline_argv':pipeline,'phase_proxy_command':shlex.join(proxy),'stage_and_pipeline_command':shlex.join(pipeline)}

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-begin',action='store_true');p.add_argument('--package-sha256')
    for n in ('observer-tools','observation-proxy-tools','root-clock-json','source-payload'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-sha256')
    p.add_argument('--observer-directory',type=Path);p.add_argument('--phase-proxy-directory',type=Path);args=p.parse_args()
    if not args.root_begin:print(json.dumps({'status':'inert_fresh_admission_assembler','clock_granted':False}));return
    manifest=package(args.package_sha256);check_sources()
    sources=read_bound(args.source_payload,args.source_payload_sha256,CAP_JSON)
    bindings=strict(read_bound(HERE/'remote_source_bindings.json',manifest['files_sha256']['remote_source_bindings.json'],CAP_JSON));source_payload(sources,bindings)
    need(args.observer_directory is not None and args.observer_directory.is_absolute() and args.observer_directory.resolve()==args.observer_directory,'exact canonical observer directory')
    need(args.phase_proxy_directory is not None and args.phase_proxy_directory.is_absolute() and args.phase_proxy_directory.resolve()==args.phase_proxy_directory and not args.phase_proxy_directory.exists(),'fresh canonical phase proxy output')
    original_raw=read_bound(args.observer_tools,args.observer_tools_sha256,CAP_JSON);original=strict(original_raw);session,done=completed_tool(original)
    result=strict(done['output']);transport_path=args.observer_directory/'transport.json';transport_pin=result['transport_sha256']
    transport=strict(read_bound(transport_path,transport_pin,CAP_JSON));need(result['scientific_admission'] is False,'actual original observer tool scope')
    raw=read_bound(args.observer_directory/'stdout',transport['stdout_sha256'],CAP_JSON);report=fresh_report(raw)
    need(report==transport['remote_report'] and transport['exit_code']==0 and transport['failure'] is None and transport['cleanup_errors']==[] and transport['signals_to_own_new_local_transport_group']==[] and transport['local_transport_reaped'] is True,'exact original successful observer transport/report')
    # This reads only the original local transport group, not any scientific data.
    proxy_raw=read_bound(args.observation_proxy_tools,args.observation_proxy_tools_sha256,CAP_JSON);proxy_doc=strict(proxy_raw);proxy_session,proxy_done=completed_tool(proxy_doc)
    terminal_binding=proxy_doc['terminal'];need(terminal_binding['path']==str(PROXY_OUTPUT/'terminal.json'),'exact prerequisite proxy terminal path')
    terminal=strict(read_bound(terminal_binding['path'],terminal_binding['sha256'],CAP_JSON))
    need(terminal['lifetime_seconds']==40 and terminal['remaining_worker_threads']==0 and terminal['payloads_logged'] is False and terminal['remote_processes_signaled'] is False,'original40s metadata proxy complete before phase issuance')
    local=local_absence({transport['local_transport_pid']},PROXY_OUTPUT,PROXY.name);local.update(utc=D.datetime.now(D.timezone.utc).isoformat(),new_observation_pgid=transport['local_transport_pid'])
    output=PREP/'goop3d_observed_fresh_entry_root_v1';need(not output.exists(),'one fresh entry assembly only');output.mkdir()
    local_pin=publish(output/'local_native_closure.json',local)
    report_pin=digest(raw)
    with (output/'fresh_report.json').open('xb') as stream:stream.write(raw)
    with (output/'original_tools.json').open('xb') as stream:stream.write(original_raw)
    admission={'schema':'goop3d_fresh_observation_admission_v1','launch':original['launch'],'completion':original['completion'],'transport':{'path':str(transport_path),'sha256':transport_pin},'local_native_closure':{'path':str(output/'local_native_closure.json'),'sha256':local_pin},'metadata_proxy':{'original_tools':{'path':str(args.observation_proxy_tools),'sha256':args.observation_proxy_tools_sha256},'terminal':terminal_binding,'original_session':proxy_session}}
    admission_pin=publish(output/'admission.json',admission)
    argv=['root_prepare.py','begin','--root-prepare','--package-sha256',args.package_sha256,'--fresh-observation',str(output/'fresh_report.json'),'--fresh-observation-sha256',report_pin,'--root-clock-json',str(args.root_clock_json),'--root-clock-json-sha256',args.root_clock_json_sha256,'--fresh-admission',str(output/'admission.json'),'--fresh-admission-sha256',admission_pin]
    previous=sys.argv;captured=io.StringIO()
    try:
        sys.argv=argv
        with contextlib.redirect_stdout(captured):root_prepare.main()
    finally:sys.argv=previous
    issued=strict(captured.getvalue());need(issued['status']=='distinct_observed_phase_issued_worker_not_started','exact fixed phase issuer result')
    commands=start_commands(sys.executable,args.package_sha256,issued,args.phase_proxy_directory,args.source_payload,args.source_payload_sha256)
    plan={'schema':'goop3d_observed_fixed_start_commands_v1','issued':issued,'source_manifest_sha256':args.package_sha256,'original_fresh_observer_tool_session':session,'fresh_admission_sha256':admission_pin,**commands}
    pin=publish(output/'start_commands.json',plan)
    print(json.dumps({'status':issued['status'],'start_commands_path':str(output/'start_commands.json'),'start_commands_sha256':pin,**issued,**commands},sort_keys=True))

if __name__=='__main__':main()
