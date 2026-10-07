"""Local-only verification of actual tool envelopes and scoped native closure."""
from pathlib import Path
import os
import subprocess
from observed_common import *

def completed_tool(doc):
    launch,done=doc['launch'],doc['completion']
    if done is None:
        need('session_id' not in launch and type(launch.get('exit_code'))is int and launch['exit_code']==0,'actual synchronous original tool completion exit zero required')
        return None,launch
    need(type(launch.get('session_id'))is int and launch['session_id']>0,'actual original launch session required')
    need(type(done.get('exit_code'))is int and done['exit_code']==0 and 'session_id' not in done,'actual original tool completion exit zero required')
    return launch['session_id'],done

def verify_fresh_admission(doc,report,now):
    need(doc['schema']=='goop3d_fresh_observation_admission_v1','explicit fresh observation admission')
    session,done=completed_tool(doc)
    tbind=doc['transport'];traw=read_bound(tbind['path'],tbind['sha256'],CAP_JSON);t=strict(traw)
    output=strict(done['output'])
    need(output['transport_sha256']==tbind['sha256'] and output['scientific_admission'] is False,'actual tool exit binds fresh transport')
    need(t['remote_report']==report and t['exit_code']==0 and t['failure'] is None and t['cleanup_errors']==[] and t['signals_to_own_new_local_transport_group']==[],'exact successful fresh report transport')
    need(t['local_transport_reaped'] is True and t['pre_publication_local_budget_preserved'] is True and 0<=t['elapsed_monotonic_seconds']<35 and 0<=t['elapsed_wall_seconds']<35,'bounded original fresh observer transport')
    binding=doc['local_native_closure'];local=strict(read_bound(binding['path'],binding['sha256'],CAP_JSON))
    need(local['ps_exit_code']==0 and local.get('parser_self_verified',local.get('parser_verified_against_self_argv')) is True and local['new_observation_pgid']==t['local_transport_pid'] and local.get('matching_rows',local.get('matching_native_rows'))==[],'fresh observer full local transport absence')
    need(stamp(t['observed_utc'])<=stamp(local['utc'])<=now,'actual local closure follows observer and precedes issuance')
    from prephase_sources import PROXY_OUTPUT,PROXY
    proxy=doc['metadata_proxy'];pbind=proxy['original_tools'];proxy_doc=strict(read_bound(pbind['path'],pbind['sha256'],CAP_JSON));proxy_session,proxy_done=completed_tool(proxy_doc)
    need(proxy['original_session']==proxy_session and proxy['terminal']==proxy_doc['terminal'] and proxy['terminal']['path']==str(PROXY_OUTPUT/'terminal.json'),'actual original metadata proxy tool/terminal binding')
    terminal=strict(read_bound(proxy['terminal']['path'],proxy['terminal']['sha256'],CAP_JSON))
    ready=strict(proxy_doc['launch']['output']);need(ready['schema']=='coder_exact_route_proxy_v1' and ready['lifetime_seconds']==terminal['lifetime_seconds']==40 and terminal['remaining_worker_threads']==0 and terminal['payloads_logged'] is False and terminal['remote_processes_signaled'] is False,'completed exact40s prerequisite proxy')
    need(local['proxy_directory']==str(PROXY_OUTPUT) and local['proxy_program']==PROXY.name and stamp(terminal['ended_utc'])<=stamp(local['utc']),'completed prerequisite proxy native closure before phase')
    return {'original_tool_session':session,'tool_envelopes_sha256':digest(json.dumps(doc,sort_keys=True).encode()),'transport_sha256':tbind['sha256'],'local_native_closure_sha256':binding['sha256'],'original_metadata_proxy_tool_session':proxy_session,'metadata_proxy_tools_sha256':pbind['sha256'],'metadata_proxy_terminal_sha256':proxy['terminal']['sha256']}

def local_absence(pgids,proxy_directory=None,proxy_program='phase_proxy.py'):
    """Filter full argv from one self-verified ps; never signal any process."""
    result=subprocess.run(['ps','-ww','-axo','pid=,ppid=,pgid=,args='],capture_output=True,text=True,timeout=5,check=False)
    need(result.returncode==0,'local native ps failed');rows=[];self_found=False
    for line in result.stdout.splitlines():
        parts=line.strip().split(None,3)
        if len(parts)!=4:continue
        pid,ppid,pgid=map(int,parts[:3]);argv=parts[3]
        if pid==os.getpid():self_found=True
        if pid in pgids or pgid in pgids or proxy_directory is not None and proxy_program in argv and str(proxy_directory) in argv:
            rows.append({'pid':pid,'ppid':ppid,'pgid':pgid,'argv':argv})
    need(self_found,'local ps parser must find self')
    need(not rows,'scoped original transport or phase proxy remains native')
    return {'ps_exit_code':0,'parser_self_verified':True,'matching_rows':rows,'observed_pgids':sorted(pgids),'proxy_directory':None if proxy_directory is None else str(proxy_directory),'proxy_program':proxy_program,'no_signals':True}
