"""Root-only original proxy completion and scoped local closure, no scientific work."""
import datetime as D, hashlib, json, sys, time
from pathlib import Path
PREP=Path(__file__).resolve().parent
FROZEN=PREP/'goop3d_observed_history_analysis_UNADMITTED_transition_v1'
sys.path.insert(0,str(FROZEN))
from root_prepare import package
from observed_common import read_bound,strict,CAP_JSON,publish,need,stamp
from phase_binding import ObservedPhase
from root_run import route_environment
from operational_checks import completed_tool,local_absence

def main():
    package('1dee2ec7c67639fab665e9fe88ce2cd6fb5862ac9eb9289fafb29446c9165e62')
    plan=strict(read_bound(PREP/'goop3d_original_postpipeline_command_plan_transition_v1.json','564b898c97a0e3efb00e49ca112530ff37548ad4bd61c6c66e15cf43002720e6',CAP_JSON))
    tools_path=PREP/'goop3d_observed_phase_proxy_original_tools_root_v1.json'
    raw=tools_path.read_bytes();doc=strict(raw);session,done=completed_tool(doc)
    need(session==78014,'actual original proxy session only')
    original=strict(read_bound(plan['phase_proxy_original_launch_path'],plan['phase_proxy_original_launch_sha256'],CAP_JSON))
    need(doc['launch']==original and set(doc)=={'launch','completion'},'exact original raw proxy launch/completion')
    state=PREP/'goop3d_observed_history_analysis_released_root_v1'
    phase=ObservedPhase(read_bound(state/'analysis_phase.json',plan['phase_sha256'],CAP_JSON),read_bound(state/'local_phase_anchor.json',plan['anchor_sha256'],CAP_JSON),plan['phase_sha256'],plan['anchor_sha256'])
    ready=strict(original['output']);route_environment(ready,phase)
    directory=Path(plan['route_ready_path']).parent;terminal_path=directory/'terminal.json';terminal_raw=terminal_path.read_bytes();terminal=strict(terminal_raw)
    need(all(terminal.get(k)==v for k,v in phase.metadata.items()),'exact original phase terminal')
    need(terminal['remaining_worker_threads']==0 and terminal['payloads_logged'] is False and terminal['remote_processes_signaled'] is False,'completed original route')
    need(terminal['connection_cap']==2048 and terminal['concurrency_cap']==8 and terminal['per_direction_buffer_bytes']==262144,'unchanged route limits')
    need(stamp(terminal['ended_utc'])>=stamp(ready['started_utc']),'terminal chronology')
    wall=time.time()-phase.start;mono=time.monotonic()-phase.mono_start
    need(0<=wall<3600 and 0<=mono<3600 and abs(wall-mono)<=5,'closure within original hour')
    close_tools=PREP/'goop3d_failed_metadata_close_original_tools_root_v1.json';close_raw=close_tools.read_bytes();close_doc=strict(close_raw);close_session,close_done=completed_tool(close_doc);need(close_session==93447,'actual original two-PID closure tool')
    close_path=PREP/'goop3d_failed_metadata_observer_closure_root_v1/transport.json';close_data=read_bound(close_path,'c5afb8f4da8dd808464a8e27126b8627bcee8bfc40700e268e1dc14ba54c23dd',CAP_JSON);close=strict(close_data)
    need(strict(close_done['output'])['transport_sha256']==hashlib.sha256(close_data).hexdigest() and close['exit_code']==0 and close['failure'] is None and not close['signals_to_own_new_local_transport_group'] and not close['cleanup_errors'],'original successful closure metadata')
    report=close['remote_report'];need(report['both_exact_prior_metadata_identities_absent_twice'] and all(v['status']=='absent' for key in ('first','second') for v in report[key].values()),'both prior metadata identities absent twice')
    local=local_absence({51576,55854,56202},directory)
    now=D.datetime.now(D.timezone.utc);need(now<stamp(plan['UTC_deadlines']['phase_stop']),'original phase closure stop')
    result={'schema':'goop3d_failed_study_original_proxy_closure_root_v1','status':'failed_study_preserved_original_proxy_and_local_transports_closed','original_proxy_session':78014,'actual_proxy_exit_code':done['exit_code'],'proxy_tools_sha256':hashlib.sha256(raw).hexdigest(),'proxy_terminal_sha256':hashlib.sha256(terminal_raw).hexdigest(),'terminal':terminal,'local_native_closure':local,'prior_metadata_closure_session':93447,'prior_metadata_closure_tools_sha256':hashlib.sha256(close_raw).hexdigest(),'prior_metadata_closure_transport_sha256':hashlib.sha256(close_data).hexdigest(),'prior_metadata_two_PIDs_absent_twice':True,'original45_remote_closure_review_sha256':'3d4c8b75e117f741c146022673c48db0d76f6feb28768ae3626804e98f1f934e','phase_sha256':plan['phase_sha256'],'observed_utc':now.isoformat(),'scientific_admission':False,'original_success_collector_or_validator_executed':False,'new_phase_or_scientific_retry':False}
    output=PREP/'goop3d_failed_study_final_proxy_closure_root_v1.json';pin=publish(output,result)
    print(json.dumps({'path':str(output),'sha256':pin,'original_proxy_exit':0,'local_transports_and_proxy_absent':True,'scientific_admission':False}))
if __name__=='__main__':main()
