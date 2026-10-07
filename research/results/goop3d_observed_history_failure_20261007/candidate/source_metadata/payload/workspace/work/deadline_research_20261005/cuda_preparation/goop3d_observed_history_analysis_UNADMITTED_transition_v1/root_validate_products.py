"""Root-only final four-product semantics plus actual tool and route closure."""
import argparse
import datetime as D
from pathlib import Path
import shlex
import time
from observed_common import *
from phase_binding import STATE,PHASE,ANCHOR,ObservedPhase
from root_prepare import package,PIDS
from operational_checks import completed_tool,local_absence
from owner_evidence import validate_owner,native_key
from root_observe_collection_closure import validate_closure
from root_run import route_environment
from root_collect import NAMES
import check_goop3d_observed_history_summary_v1 as checker

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-review',action='store_true')
    for n in ('package-sha256','phase-sha256','anchor-sha256','collection-receipt-sha256','collection-tools-sha256','closure-receipt-sha256','closure-tools-sha256','proxy-tools-sha256','proxy-terminal-sha256'):p.add_argument('--'+n)
    for n in ('collection-tools','closure-tools','proxy-tools','proxy-directory'):p.add_argument('--'+n,type=Path)
    args=p.parse_args()
    if not args.root_review:print(json.dumps({'status':'inert_scoped_product_validator'}));return
    manifest=package(args.package_sha256)
    phase_raw=read_bound(PHASE,args.phase_sha256,CAP_JSON);anchor_raw=read_bound(ANCHOR,args.anchor_sha256,CAP_JSON)
    route=ObservedPhase(phase_raw,anchor_raw,args.phase_sha256,args.anchor_sha256)
    def check():
        wall=time.time()-route.start;mono=time.monotonic()-route.mono_start
        need(0<=wall<3580 and 0<=mono<3580 and abs(wall-mono)<=5,'fixed final review cutoff or clock disagreement')
    check();directory=STATE/'product_collection'
    receipt=strict(read_bound(directory/'local_receipt.json',args.collection_receipt_sha256,CAP_JSON,check))
    need(receipt['schema']=='goop3d_observed_local_collection_receipt_v1' and receipt['phase_sha256']==args.phase_sha256 and receipt['scientific_admission'] is False,'exact local collection receipt')
    original=strict(read_bound(args.collection_tools,args.collection_tools_sha256,CAP_JSON,check));collection_session,done=completed_tool(original)
    tool_result=strict(done['output']);need(tool_result['local_receipt_sha256']==args.collection_receipt_sha256 and tool_result['scientific_admission'] is False,'actual original collection tool exit binds receipt')
    files=receipt['files_sha256'];need(set(files)==set(NAMES),'exact ten-member collection envelope')
    raw_data={name:read_bound(directory/name,pin,CAP_OUTPUT,check) for name,pin in files.items()};data={name:strict(raw) for name,raw in raw_data.items() if name.endswith('.json')}
    audit,summary,arithmetic,pipeline=(data[n+'.json'] for n in ('audit','summary','arithmetic_check','pipeline_receipt'))
    need(pipeline['schema']=='goop3d_observed_history_pipeline_receipt_v1' and pipeline['status']=='all_three_scoped_stages_completed' and pipeline['phase_sha256']==args.phase_sha256,'complete distinct pipeline')
    need(pipeline['audit_sha256']==summary['audit_sha256']==arithmetic['audit_sha256']==files['audit.json'] and pipeline['summary_sha256']==arithmetic['summary_sha256']==files['summary.json'] and pipeline['arithmetic_check_sha256']==files['arithmetic_check.json'],'four product hash chain')
    need(all(x['phase_sha256']==args.phase_sha256 for x in (audit,summary,arithmetic,pipeline)),'one exact phase for all products')
    recomputed=checker.verify(audit,summary)
    checker.equal({k:arithmetic[k] for k in recomputed},recomputed)
    need(audit['whole_collection_parsed_for_accounting'] is True and audit['autonomous_queue_files_or_arrays_opened'] is False and audit['autonomous_numerical_aggregation_performed'] is False,'precise observed-only scope')
    report=data['collection.json'];need(report['host']==HOST and report['boot_id']==BOOT,'original host/boot collection')
    identities=report['new_owner_identities'];expected={str(p) for p in PIDS}|{str(x['pid']) for x in identities.values()}
    need(len(expected)==45 and set(identities)=={'outer','owner','worker'} and set(report['native_first_absent'])==set(report['native_second_absent'])==expected and all(x is True for x in report['native_first_absent'].values()) and all(x is True for x in report['native_second_absent'].values()),'all exact old42 plus new3 absent twice')
    release_pin=receipt['release_sha256'];release=strict(read_bound(STATE/'pipeline.cpu_release.json',release_pin,CAP_JSON,check));phase_doc=strict(phase_raw)
    spec=strict(read_bound(Path(__file__).resolve().parent/'observed_operations.json',manifest['files_sha256']['observed_operations.json'],CAP_JSON,check))['operations']['d3_observed_history_pipeline']
    pipeline_tools=strict(read_bound(directory/'pipeline_original_tools.json',receipt['original_pipeline_tools_sha256'],CAP_JSON,check));pipeline_session,pipeline_done=completed_tool(pipeline_tools)
    need(pipeline_session==receipt['original_pipeline_tool_session'] and strict(pipeline_done['output'])==strict((STATE/'pipeline.transport/transport.json').read_bytes()),'actual original pipeline exit retained through final admission')
    pipeline_transport=strict(pipeline_done['output'])
    launch_intent=strict(read_bound(STATE/'pipeline.transport/intent.json',pipeline_transport['launch_intent_sha256'],CAP_JSON,check))
    stage_admission=strict(read_bound(STATE/'combined_stage_admission.json',launch_intent['combined_stage_admission_sha256'],CAP_JSON,check))
    need(stage_admission['phase_sha256']==args.phase_sha256 and stage_admission['source_manifest_sha256']==args.package_sha256 and stage_admission['all13_sources_and4_controls_charged_to_phase'] is True and route.start<=stamp(stage_admission['started_utc']).timestamp()<=stamp(stage_admission['finished_utc']).timestamp()<route.start+100,'all remote source/control setup charged to original phase')
    exact_identities=validate_owner(data,files,release,phase_doc,args.phase_sha256,release_pin,spec,manifest['files_sha256'],pipeline_transport)
    need({k:native_key(v) for k,v in identities.items()}==exact_identities,'reported collection identities match full owner evidence')
    intent=strict(read_bound(directory/'intent.json',receipt['intent_sha256'],CAP_JSON,check))
    collector=native_key(report['collector_identity']);collector_outer=native_key(report['collector_parent_identity'])
    need(collector_outer['argv']==shlex.split(intent['argv'][-1]) and collector['argv']==collector_outer['argv'][3:] and collector['ppid']==collector_outer['pid'] and collector['pgid']==collector_outer['pgid'] and collector['sid']==collector_outer['sid'],'exact original collector and timeout command/start identities')
    closure_dir=STATE/'postcollection_closure'
    closure_receipt=strict(read_bound(closure_dir/'local_receipt.json',args.closure_receipt_sha256,CAP_JSON,check))
    closure_original=strict(read_bound(args.closure_tools,args.closure_tools_sha256,CAP_JSON,check));closure_session,closure_done=completed_tool(closure_original)
    need(strict(closure_done['output'])['local_receipt_sha256']==args.closure_receipt_sha256 and closure_receipt['phase_sha256']==args.phase_sha256 and closure_receipt['collection_receipt_sha256']==args.collection_receipt_sha256 and closure_receipt['collection_tools_sha256']==args.collection_tools_sha256 and closure_receipt['original_collection_session']==collection_session,'actual original postcollection tool exit and exact preceding collection')
    closure_report=strict(read_bound(closure_dir/'report.json',closure_receipt['report_sha256'],CAP_JSON,check))
    all47=validate_closure(closure_report,report,args.phase_sha256,files['collection.json'])
    need(len(all47)==47 and stamp(closure_report['finished_utc']).timestamp()<route.start+3480,'all47 closure inside fixed original reserve')
    closure_transport=strict(read_bound(closure_dir/'transport.json',closure_receipt['transport_sha256'],CAP_JSON,check))
    need(closure_transport['exit_code']==0 and closure_transport['failure'] is None and closure_transport['signals_to_new_local_group_only']==[] and 0<=closure_transport['elapsed_monotonic_seconds']<30,'completed bounded original closure transport')
    proxy_raw=read_bound(args.proxy_tools,args.proxy_tools_sha256,CAP_JSON,check);proxy_doc=strict(proxy_raw);proxy_session,proxy_done=completed_tool(proxy_doc)
    ready=strict(proxy_doc['launch']['output']);route_environment(ready,route)
    terminal=strict(read_bound(args.proxy_directory/'terminal.json',args.proxy_terminal_sha256,CAP_JSON,check))
    need(all(terminal.get(k)==v for k,v in route.metadata.items()) and terminal['remaining_worker_threads']==0 and terminal['payloads_logged'] is False and terminal['remote_processes_signaled'] is False,'completed phase route without remaining workers')
    need(terminal['connection_cap']==2048 and terminal['concurrency_cap']==8 and terminal['per_direction_buffer_bytes']==262144,'same bounded nonblocking route through completion')
    need(stamp(terminal['ended_utc']).timestamp()<=route.start+3580 and stamp(terminal['started_utc'])>=stamp(ready['started_utc']),'route lifetime inside final review reserve')
    collect_transport=strict(read_bound(directory/'transport.json',receipt['transport_sha256'],CAP_JSON,check))
    closure=local_absence({pipeline_transport['local_transport_pid'],collect_transport['local_transport_pid'],closure_transport['local_transport_pid']},args.proxy_directory)
    check()
    for name,pin in files.items():read_bound(directory/name,pin,CAP_OUTPUT,check,retain=False)
    result={'schema':'goop3d_observed_history_root_product_admission_v1','status':'accepted_complete_scoped_observed_history_evidence','phase_sha256':args.phase_sha256,'original_expired_phase_sha256':OLD_PHASE_SHA,'package_sha256':args.package_sha256,'collection_receipt_sha256':args.collection_receipt_sha256,'postcollection_closure_receipt_sha256':args.closure_receipt_sha256,'original_pipeline_tool_session':pipeline_session,'original_collection_tool_session':collection_session,'original_postcollection_observer_tool_session':closure_session,'original_proxy_tool_session':proxy_session,'all47_remote_original_pids_absent_twice':True,'complete_owner_input_and_evidence_sha_maps_verified':True,'four_product_sha256':{n:files[n] for n in ('audit.json','summary.json','arithmetic_check.json','pipeline_receipt.json')},'full_original_cells_retained':4728,'fixed_observed_cells':2568,'local_native_closure':closure,'conditional_clean_normalization':True,'model_dependent_risk_graphs':True,'full_autonomous_cohort_not_completed':True,'autonomous_numerical_result_admitted':False,'reviewed_utc':D.datetime.now(D.timezone.utc).isoformat(),'limitations':audit['limitations']}
    print(json.dumps({'status':result['status'],'admission_sha256':publish(STATE/'product_admission.json',result,check)}))

if __name__=='__main__':main()
