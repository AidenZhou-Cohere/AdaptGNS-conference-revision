"""One CPU worker, three fixed scientific stages, four distinct final products."""
import argparse
from pathlib import Path
import resource
import traceback
from types import SimpleNamespace
from observed_common import *
import audit_goop3d_observed_histories_v1 as audited
import summarize_goop3d_observed_histories_v1 as summarized
import check_goop3d_observed_history_summary_v1 as checked

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--phase',type=Path);parser.add_argument('--phase-sha256');parser.add_argument('--source-bindings-sha256')
    args=parser.parse_args()
    if not args.execute:print(json.dumps({'status':'inert_single_observed_pipeline','new_allowance':False}));return
    phase=load_phase(args.phase,args.phase_sha256)
    resource.setrlimit(resource.RLIMIT_AS,(32<<30,32<<30))
    paths={name:Path(OUTPUT)/(name+'.json') for name in ('audit','summary','arithmetic_check','pipeline_receipt')}
    need(all(not p.exists() for p in paths.values()),'four fresh scoped outputs required')
    active='audit';final_budget=WorkerBudget(phase,2980)
    try:
        audit_budget=WorkerBudget(phase,2700)
        audit_args=SimpleNamespace(collection=Path(COLLECTION),collection_sha256=COLLECTION_SHA,queue_root=Path(QUEUE),output=paths['audit'],phase_sha256=args.phase_sha256,source_bindings_sha256=args.source_bindings_sha256)
        audit=audited.audit(audit_args,phase,audit_budget)
        audit_pin=publish(paths['audit'],audit,audit_budget.check)
        active='summary';summary_budget=WorkerBudget(phase,2820)
        summary=summarized.summarize(audit);summary['audit_sha256']=audit_pin
        summary_pin=publish(paths['summary'],summary,summary_budget.check)
        active='independent_arithmetic';check_budget=WorkerBudget(phase,2940)
        result=checked.verify(audit,summary);result.update(audit_sha256=audit_pin,summary_sha256=summary_pin)
        check_pin=publish(paths['arithmetic_check'],result,check_budget.check)
        active='final_bindings';final_budget.check()
        for path,pin in [(paths['audit'],audit_pin),(paths['summary'],summary_pin),(paths['arithmetic_check'],check_pin)]:read_bound(path,pin,CAP_OUTPUT,final_budget.check,retain=False)
        for path,pin in audit['source_sha256'].items():read_bound(path,pin,CAP_JSON,final_budget.check,retain=False)
        read_bound(args.phase,args.phase_sha256,CAP_JSON,final_budget.check,retain=False)
        terminal={'schema':'goop3d_observed_history_pipeline_receipt_v1','status':'all_three_scoped_stages_completed','phase_sha256':args.phase_sha256,'collection_sha256':COLLECTION_SHA,'audit_sha256':audit_pin,'summary_sha256':summary_pin,'arithmetic_check_sha256':check_pin,'required_all_cells':4728,'required_observed_cells':2568,'single_worker_process':True,'scientific_admission':False,'original_audit_retried':False,'old_phase_reset':False,'conditional_clean_normalization':True,'risk_policy_graph_is_model_dependent':True}
        print(json.dumps({'status':terminal['status'],'pipeline_receipt_sha256':publish(paths['pipeline_receipt'],terminal,final_budget.check),'scientific_admission':False}))
    except BaseException as error:
        try:publish(Path(OUTPUT)/'pipeline.failure.json',{'schema':'goop3d_observed_history_pipeline_failure_v1','status':'failed_or_incomplete','stage':active,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'partial_products_not_admitted':True,'automatic_retry':False,'scientific_admission':False},final_budget.check)
        except BaseException:pass
        raise

if __name__=='__main__':main()
