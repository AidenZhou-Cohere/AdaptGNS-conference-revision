"""Inert local preassembly and exact final-validator exec; no live work on import."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PREP=HERE.parent
FROZEN=PREP/'goop3d_observed_history_analysis_UNADMITTED_transition_v1'
STATE=PREP/'goop3d_observed_history_analysis_released_root_v1'
PLAN=PREP/'goop3d_original_postpipeline_command_plan_transition_v1.json'
PLAN_SHA='564b898c97a0e3efb00e49ca112530ff37548ad4bd61c6c66e15cf43002720e6'
PACKAGE_SHA='1dee2ec7c67639fab665e9fe88ce2cd6fb5862ac9eb9289fafb29446c9165e62'
sys.path.insert(0,str(FROZEN))
from observed_common import need,strict,read_bound,digest,publish,CAP_JSON,stamp
from operational_checks import completed_tool
from root_prepare import package
from root_stage_and_run import read_new_metadata
from phase_binding import ObservedPhase
from root_run import route_environment
from root_collect import NAMES

LAST_TWO={'phase_proxy_tools_sha256','phase_proxy_terminal_sha256'}
BOUND_FIELDS={'collection_receipt_sha256','collection_tools_path','collection_tools_sha256','closure_receipt_sha256','closure_tools_path','closure_tools_sha256','phase_proxy_tools_path'}

def fill(template,values):
    result=[]
    for item in template:
        if isinstance(item,dict):
            need(set(item)=={'required_future_actual_binding'},'exact original placeholder')
            key=item['required_future_actual_binding'];result.append(values.get(key,item))
        else:need(type(item)is str,'original argv string');result.append(item)
    return result

def outstanding(argv):
    return [x['required_future_actual_binding'] for x in argv if isinstance(x,dict)]

def load_reviewed(pin):
    m=strict(read_bound(HERE/'manifest.json',pin,CAP_JSON))
    need(m['plan_sha256']==PLAN_SHA and m['source_manifest_sha256']==PACKAGE_SHA,'exact reviewed helper scope')
    for name,sha in m['files_sha256'].items():
        p=HERE/name;need(p.resolve().is_relative_to(HERE),'helper relative source');read_bound(p,sha,CAP_JSON,retain=False)
    package(PACKAGE_SHA)
    plan=strict(read_bound(PLAN,PLAN_SHA,CAP_JSON))
    need(plan['source_manifest_sha256']==PACKAGE_SHA and plan['pipeline_original_session']==33822 and plan['phase_proxy_original_session']==78014,'original control plan and sessions')
    need(plan['final_admission_argv_template'][2]==str(FROZEN/'root_validate_products.py'),'only original final validator')
    return plan

def preassembly(plan,helper_pin,collection_path,collection_pin,closure_path,closure_pin,proxy_path):
    files={}
    def read(path,pin):
        path=Path(path);raw=read_bound(path,pin,CAP_JSON);files[str(path)]=pin;return strict(raw)
    need(all(Path(p).is_absolute() and Path(p).resolve()==Path(p) for p in (collection_path,closure_path,proxy_path)),'canonical original evidence paths')
    col=read(collection_path,collection_pin);col_session,col_done=completed_tool(col);col_output=strict(col_done['output'])
    need(col_output['scientific_admission'] is False,'actual collection tool scope')
    col_receipt_pin=col_output['local_receipt_sha256'];col_receipt=read(STATE/'product_collection/local_receipt.json',col_receipt_pin)
    need(col_receipt['schema']=='goop3d_observed_local_collection_receipt_v1' and col_receipt['phase_sha256']==plan['phase_sha256'] and col_receipt['release_sha256']==plan['release_sha256'] and col_receipt['scientific_admission'] is False and set(col_receipt['files_sha256'])==set(NAMES),'exact current collection metadata')
    pipeline=read(STATE/'product_collection/pipeline_original_tools.json',col_receipt['original_pipeline_tools_sha256']);pipeline_session,pipeline_done=completed_tool(pipeline)
    original_launch=read(plan['pipeline_original_launch_path'],plan['pipeline_original_launch_sha256'])
    need(pipeline_session==col_receipt['original_pipeline_tool_session']==33822 and pipeline['launch']==original_launch,'exact original33822 launch and actual completion')
    pipeline_transport_raw,pipeline_transport_pin=read_new_metadata(STATE/'pipeline.transport/transport.json');files[str(STATE/'pipeline.transport/transport.json')]=pipeline_transport_pin
    pipeline_transport=strict(pipeline_transport_raw)
    need(strict(pipeline_done['output'])==pipeline_transport and pipeline_transport['exit_code']==0 and pipeline_transport['failure'] is None and pipeline_transport['signals_to_new_local_group_only']==[] and pipeline_transport['phase_sha256']==plan['phase_sha256'] and pipeline_transport['release_sha256']==plan['release_sha256'],'actual pipeline completion binds exact current transport')
    col_transport=read(STATE/'product_collection/transport.json',col_receipt['transport_sha256'])
    need(col_transport['exit_code']==0 and col_transport['failure'] is None and col_transport['signals_to_new_local_group_only']==[],'completed original collection transport')
    clo=read(closure_path,closure_pin);clo_session,clo_done=completed_tool(clo);clo_output=strict(clo_done['output'])
    need(clo_output['scientific_admission'] is False,'actual closure tool scope')
    clo_receipt_pin=clo_output['local_receipt_sha256'];clo_receipt=read(STATE/'postcollection_closure/local_receipt.json',clo_receipt_pin)
    need(clo_receipt['schema']=='goop3d_postcollection_closure_local_receipt_v1' and clo_receipt['phase_sha256']==plan['phase_sha256'] and clo_receipt['collection_receipt_sha256']==col_receipt_pin and clo_receipt['collection_tools_sha256']==collection_pin and clo_receipt['original_collection_session']==col_session and clo_receipt['scientific_admission'] is False,'actual closure receipt binds preceding original collection')
    clo_transport=read(STATE/'postcollection_closure/transport.json',clo_receipt['transport_sha256'])
    need(clo_transport['exit_code']==0 and clo_transport['failure'] is None and clo_transport['signals_to_new_local_group_only']==[],'completed original closure transport')
    original_proxy=read(plan['phase_proxy_original_launch_path'],plan['phase_proxy_original_launch_sha256'])
    need(original_proxy['session_id']==78014 and 'exit_code' not in original_proxy,'unchanged original78014 launch')
    values=dict(collection_receipt_sha256=col_receipt_pin,collection_tools_path=str(collection_path),collection_tools_sha256=collection_pin,closure_receipt_sha256=clo_receipt_pin,closure_tools_path=str(closure_path),closure_tools_sha256=closure_pin,phase_proxy_tools_path=str(proxy_path))
    argv=fill(plan['final_admission_argv_template'],values)
    need(set(outstanding(argv))==LAST_TWO and len(outstanding(argv))==2,'only final two genuine hashes unresolved')
    return {'schema':'goop3d_original_final_admission_preassembly_v1','plan_sha256':PLAN_SHA,'helper_manifest_sha256':helper_pin,'source_manifest_sha256':PACKAGE_SHA,'resolved_bindings':values,'metadata_files_sha256':files,'actual_pipeline_session':pipeline_session,'actual_collection_session':col_session,'actual_closure_session':clo_session,'original_phase_proxy_session':78014,'final_argv_template':argv,'scientific_products_read':False,'final_validator_run':False}

def proxy_completion(doc,original_launch):
    need(set(doc)=={'launch','completion'} and doc['launch']==original_launch,'raw exact original proxy tool envelope')
    session,done=completed_tool(doc)
    need(session==78014,'actual original78014 completion only')
    return done

def final_argv(plan,pre,proxy_raw,terminal_raw):
    original_launch=strict(read_bound(plan['phase_proxy_original_launch_path'],plan['phase_proxy_original_launch_sha256'],CAP_JSON))
    proxy_completion(strict(proxy_raw),original_launch)
    phase_raw=read_bound(STATE/'analysis_phase.json',plan['phase_sha256'],CAP_JSON);anchor_raw=read_bound(STATE/'local_phase_anchor.json',plan['anchor_sha256'],CAP_JSON)
    phase=ObservedPhase(phase_raw,anchor_raw,plan['phase_sha256'],plan['anchor_sha256'])
    ready=strict(original_launch['output']);route_environment(ready,phase);terminal=strict(terminal_raw)
    need(all(terminal.get(k)==v for k,v in phase.metadata.items()) and terminal['remaining_worker_threads']==0 and terminal['payloads_logged'] is False and terminal['remote_processes_signaled'] is False,'actual completed terminal with original phase metadata')
    need(terminal['connection_cap']==2048 and terminal['concurrency_cap']==8 and terminal['per_direction_buffer_bytes']==262144,'original terminal route bounds')
    need(stamp(terminal['ended_utc'])<=stamp(plan['UTC_deadlines']['final_root_admission']) and stamp(terminal['started_utc'])>=stamp(ready['started_utc']),'terminal within original final reserve')
    argv=fill(pre['final_argv_template'],{'phase_proxy_tools_sha256':digest(proxy_raw),'phase_proxy_terminal_sha256':digest(terminal_raw)})
    need(not outstanding(argv) and argv[:3]==plan['final_admission_argv_template'][:3],'only two hashes filled into original final command')
    return argv

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('action',choices=('prepare','finalize'),nargs='?');p.add_argument('--root-final-entry',action='store_true');p.add_argument('--helper-sha256')
    for n in ('collection-tools','closure-tools','preassembly'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-sha256')
    p.add_argument('--proxy-tools',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if not args.root_final_entry:print(json.dumps({'status':'inert_original_final_entry','validator_run':False,'clock_granted':False}));return
    plan=load_reviewed(args.helper_sha256)
    if args.action=='prepare':
        need(args.proxy_tools is not None and not args.proxy_tools.exists(),'fresh future original proxy evidence path')
        result=preassembly(plan,args.helper_sha256,args.collection_tools,args.collection_tools_sha256,args.closure_tools,args.closure_tools_sha256,args.proxy_tools)
        need(args.output is not None and args.output.is_absolute() and args.output.resolve()==args.output and not args.output.exists(),'fresh canonical local preassembly output')
        pin=publish(args.output,result);print(json.dumps({'status':'actual_metadata_preassembled_final_two_hashes_pending','path':str(args.output),'sha256':pin}));return
    need(args.action=='finalize','explicit final action')
    pre=strict(read_bound(args.preassembly,args.preassembly_sha256,CAP_JSON));values=pre['resolved_bindings'];need(set(values)==BOUND_FIELDS,'exact preassembled binding fields')
    rebuilt=preassembly(plan,args.helper_sha256,Path(values['collection_tools_path']),values['collection_tools_sha256'],Path(values['closure_tools_path']),values['closure_tools_sha256'],Path(values['phase_proxy_tools_path']))
    need(pre==rebuilt,'every actual preassembled metadata binding unchanged')
    need(args.proxy_tools==Path(values['phase_proxy_tools_path']),'exact preselected actual proxy tools path')
    proxy_raw,_=read_new_metadata(args.proxy_tools);terminal_raw,_=read_new_metadata(Path(plan['route_ready_path']).parent/'terminal.json')
    argv=final_argv(plan,pre,proxy_raw,terminal_raw)
    # Replace the helper so it cannot remain as a matching local wrapper.
    # The original validator still performs every scientific and deadline check.
    os.execv(argv[0],argv)

if __name__=='__main__':main()
