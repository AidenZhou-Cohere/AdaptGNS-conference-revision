"""Root-only final four-product semantics plus actual tool and route closure."""
import argparse
import datetime as D
from pathlib import Path
import time
from observed_common import *
from phase_binding import STATE,PHASE,ANCHOR,ObservedPhase
from root_prepare import package,PIDS
from operational_checks import completed_tool,local_absence
import check_goop3d_observed_history_summary_v1 as checker

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-review',action='store_true')
    for n in ('package-sha256','phase-sha256','anchor-sha256','collection-receipt-sha256','collection-tools-sha256','proxy-tools-sha256','proxy-terminal-sha256'):p.add_argument('--'+n)
    for n in ('collection-tools','proxy-tools','proxy-directory'):p.add_argument('--'+n,type=Path)
    args=p.parse_args()
    if not args.root_review:print(json.dumps({'status':'inert_scoped_product_validator'}));return
    package(args.package_sha256)
    phase_raw=read_bound(PHASE,args.phase_sha256,CAP_JSON);anchor_raw=read_bound(ANCHOR,args.anchor_sha256,CAP_JSON)
    route=ObservedPhase(phase_raw,anchor_raw,args.phase_sha256,args.anchor_sha256)
    def check():
        wall=time.time()-route.start;mono=time.monotonic()-route.mono_start
        need(0<=wall<3580 and 0<=mono<3580 and abs(wall-mono)<=5,'fixed final review cutoff or clock disagreement')
    check();directory=STATE/'product_collection'
    receipt=strict(read_bound(directory/'local_receipt.json',args.collection_receipt_sha256,CAP_JSON,check))
    original=strict(read_bound(args.collection_tools,args.collection_tools_sha256,CAP_JSON,check));collection_session,done=completed_tool(original)
    tool_result=strict(done['output']);need(tool_result['local_receipt_sha256']==args.collection_receipt_sha256 and tool_result['scientific_admission'] is False,'actual original collection tool exit binds receipt')
    files=receipt['files_sha256'];data={name:strict(read_bound(directory/name,pin,CAP_OUTPUT,check)) for name,pin in files.items()}
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
    owner=data['owner_terminal.json'];need(owner['status']=='complete' and owner['failure'] is None and owner['child']['exit_code']==0 and owner['child']['signals']==[] and owner['child_native_absent'] is True,'complete original owner/worker evidence')
    need(owner['output_sha256']=={OUTPUT+'/'+n:files[n] for n in ('audit.json','summary.json','arithmetic_check.json','pipeline_receipt.json')},'owner binds all exact product bytes')
    pipeline_tools=strict(read_bound(directory/'pipeline_original_tools.json',receipt['original_pipeline_tools_sha256'],CAP_JSON,check));pipeline_session,pipeline_done=completed_tool(pipeline_tools)
    need(pipeline_session==receipt['original_pipeline_tool_session'] and strict(pipeline_done['output'])==strict((STATE/'pipeline.transport/transport.json').read_bytes()),'actual original pipeline exit retained through final admission')
    proxy_raw=read_bound(args.proxy_tools,args.proxy_tools_sha256,CAP_JSON,check);proxy_doc=strict(proxy_raw);proxy_session,proxy_done=completed_tool(proxy_doc)
    ready=strict(proxy_doc['launch']['output']);need(ready['schema']=='coder_goop3d_observed_phase_route_proxy_v1' and all(ready.get(k)==v for k,v in route.metadata.items()),'actual original proxy tool bound to phase')
    terminal=strict(read_bound(args.proxy_directory/'terminal.json',args.proxy_terminal_sha256,CAP_JSON,check))
    need(all(terminal.get(k)==v for k,v in route.metadata.items()) and terminal['remaining_worker_threads']==0 and terminal['payloads_logged'] is False and terminal['remote_processes_signaled'] is False,'completed phase route without remaining workers')
    need(stamp(terminal['ended_utc']).timestamp()<=route.start+3580 and stamp(terminal['started_utc'])>=stamp(ready['started_utc']),'route lifetime inside final review reserve')
    pipeline_transport=strict((STATE/'pipeline.transport/transport.json').read_bytes());collect_transport=strict((directory/'transport.json').read_bytes())
    closure=local_absence({pipeline_transport['local_transport_pid'],collect_transport['local_transport_pid']},args.proxy_directory)
    check()
    for name,pin in files.items():read_bound(directory/name,pin,CAP_OUTPUT,check,retain=False)
    result={'schema':'goop3d_observed_history_root_product_admission_v1','status':'accepted_complete_scoped_observed_history_evidence','phase_sha256':args.phase_sha256,'original_expired_phase_sha256':OLD_PHASE_SHA,'package_sha256':args.package_sha256,'collection_receipt_sha256':args.collection_receipt_sha256,'original_pipeline_tool_session':pipeline_session,'original_collection_tool_session':collection_session,'original_proxy_tool_session':proxy_session,'four_product_sha256':{n:files[n] for n in ('audit.json','summary.json','arithmetic_check.json','pipeline_receipt.json')},'full_original_cells_retained':4728,'fixed_observed_cells':2568,'local_native_closure':closure,'conditional_clean_normalization':True,'model_dependent_risk_graphs':True,'full_autonomous_cohort_not_completed':True,'autonomous_numerical_result_admitted':False,'reviewed_utc':D.datetime.now(D.timezone.utc).isoformat(),'limitations':audit['limitations']}
    print(json.dumps({'status':result['status'],'admission_sha256':publish(STATE/'product_admission.json',result,check)}))

if __name__=='__main__':main()
