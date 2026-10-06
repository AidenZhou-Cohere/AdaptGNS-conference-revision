#!/usr/bin/env python3
"""Bind completed original Sand analysis to six accepted independent receipts. No reruns/clocks."""
from pathlib import Path
import argparse,datetime as D,json
from review_sand_actual_phase_runtime_code_audit_v1 import P,S,PH,AN,sha,read,exact,dt
OPS={'sand_collect_A':(84433,2432,16,'8638a839266c6e6e874b87e609e13d7429b1a519c0b9d154c7d2724e466b7897','09d30c70cdd83244019f201fe958487003d68526446989a70583e15a31913346'),'sand_collect_B':(5729,1216,8,'a18a6f50a6d704820741155b66d87266dba2be4e71bd3162a4dfc1fbe7119be9','a2d26c4c28bdb836601d251b99706ee2df1c809b6ff635ea751d0439fee73df3'),'sand_saved_A':(66211,2432,16,'6d07b58342747b6c716ebebeb2407d7ec72bd6100ffb547e8c4af06e80786d42','46fcdb24eb84bd67467f9c221c2d1a4eac74fffd84c7866996a08140b20c787e'),'sand_saved_B':(29214,1216,8,'6edcfcf7fd1db5e00c77d48cb5037240db5aefc3b53e4bbbdde8ce2340f62e9a','173f1b93524b47d5c6f01fef311418ee56c0eb1da6fa609f37831ee93fa52516'),'sand_summarize':(19303,3648,24,'9acac8293702fb26974bb1a286f01eb6beb7451867d5b4e0e07acc024bacef36','8f902aa9cfb6b85b0b125d9cab6288332ac5f237cfe27494e234089bbde76926')}
def audit(paired_root_sha,paired_independent_sha,completion_sha,completion_tool_file,completion_session):
    checks=[];evidence={}
    def ck(ok,n):
        if not ok:raise ValueError(n)
        checks.append(n)
    def pin(p,h=None):
        p=Path(p);ck(p.is_file() and not p.is_symlink() and p.suffix not in ('.npy','.npz','.pt','.pth'),'Ordinary completed metadata '+p.name);v=sha(p);evidence[str(p.resolve())]=v
        if h is not None:ck(v==h,'Exact completed evidence bytes '+p.name)
        return v
    def doc(p,h=None):pin(p,h);return read(p)
    pin(__file__);phase=doc(S/'analysis_phase.json',PH);anchor=doc(S/'local_phase_anchor.json',AN);start=dt(phase['original_analysis_started_utc']);stop=dt(phase['original_analysis_stop_utc'])
    ck(phase['original_analysis_seconds']==3600 and (stop-start).total_seconds()==3600 and phase['new_or_restarted_clock_granted'] is False and anchor['phase_sha256']==PH and anchor['utc']==start.isoformat(),'Exactly one original Sand3600second phase retained')
    done=doc(S/'analysis_completion.json',completion_sha);finished=dt(done['completed_utc']);ck(done['status']=='all_six_original_stopped_analysis_operations_reviewed' and done['analysis_phase_sha256']==PH and start<=finished<stop and done['all_original_failures_and_unexecuted_denominators_preserved'] is True and done['scientific_success_or_speedup_not_implied'] is True,'Original frozen completion inside original hour with limited claim')
    tool=doc(completion_tool_file)
    if 'launch' in tool and 'completion' in tool:ck(tool['launch']['session_id']==completion_session,'Original completion tool launch session');tool=tool['completion']
    elif 'original_launch' in tool and 'original_completion' in tool:ck(tool['original_launch']['session_id']==completion_session,'Original completion tool launch session');tool=tool['original_completion']
    elif 'launch' in tool and 'exit' in tool:ck(tool['launch']['session_id']==completion_session,'Original completion tool launch session');tool=tool['exit']
    ck(type(tool['exit_code']) is int and tool['exit_code']==0 and (tool.get('session_id') is None or tool['session_id']==completion_session),'Genuine original complete action exit0; null/omitted session means synchronous completion')
    all_ops=dict(OPS,sand_paired=(92386,3648,24,paired_root_sha,paired_independent_sha));ck(set(done['operation_review_sha256'])==set(all_ops),'Exactly all six original operations completed')
    summaries={}
    for op,(session,cells,stages,root_sha,ind_sha) in all_ops.items():
        root=doc(S/(op+'.review.json'),root_sha);ind=doc(P/(op+'_completed_independent_review_code_audit_v1.json'),ind_sha)
        ck(done['operation_review_sha256'][op]==root_sha and root['status']=='accepted_stopped_product' and root['analysis_phase_sha256']==PH and start<=dt(root['checked_utc'])<=finished,'Completion binds earlier accepted original operation '+op)
        ck(ind['status']=='passed_completed_original_Sand_product_evidence' and ind['operation']==op and ind['original_tool_session']==session and ind['analysis_phase_sha256']==PH and ind['accepted_root_review_sha256']==root_sha and ind['product_sha256']==root['product_sha256'],'Independent evidence receipt binds exact original operation/product '+op)
        ck(ind['required_cells']==cells and ind['required_model_stages']==stages and ind['coverage']=={'completed_required_outcome':cells} and root['semantic_review']['coverage']==ind['coverage'] and root['semantic_review']['all_required_cells_completed'] is True,'Complete fixed original cell/seed/stage denominator '+op)
        ck(ind['statistics_recomputed'] is False and ind['arrays_or_models_opened'] is False and ind['clocks_read_or_issued'] is False and ind['live_or_remote_action'] is False and ind['scientific_admission'] is False,'Independent check scope remains metadata/counting '+op)
        ck(all(x is True for x in root['native_absent'].values()),'All original operation native identities absent '+op)
        summaries[op]={'original_tool_session':session,'root_review_sha256':root_sha,'independent_review_sha256':ind_sha,'product_sha256':root['product_sha256'],'required_cells':cells,'required_model_stages':stages,'native_absent_count':len(root['native_absent'])}
    pin(P/'sand_recovered_transfer_independent_review_code_audit_v1.json','92ef54b554c68b87542754f1caf2558ba3967bd237d16308a06d73817581d2d1')
    for folder,manifest_pin in [('sand_scalar_recovery_UNADMITTED_code_audit_v1','71c50f558082d127757cc84dff97f7451c7c5e73ff4b5ef48ff41be1af851a79'),('sand_summary_capture_recovery_UNADMITTED_code_audit_v1','2b8b0b9a8a0a025533d6291f401f1e99fa6be9cb70b8e7b0bb27029f00fc429c')]:
        manifest=doc(P/folder/'manifest.json',manifest_pin)
        for n,h in manifest['files_sha256'].items():pin(P/folder/n,h)
        for p,h in doc(P/folder/'failure_bindings.json')['files_sha256'].items():pin(p,h)
    for p,h in list(evidence.items()):ck(sha(p)==h,'Completed original evidence stable '+Path(p).name)
    return {'schema':'sand_original_analysis_completion_independent_code_audit_v1','status':'passed_all_six_original_completed_analysis_bindings','analysis_phase_sha256':PH,'local_phase_anchor_sha256':AN,'original_started_utc':start.isoformat(),'original_stop_utc':stop.isoformat(),'root_completed_utc':done['completed_utc'],'analysis_completion_sha256':completion_sha,'operations':summaries,'distinct_required_models':6,'distinct_required_model_stages':24,'distinct_required_cells':3648,'distinct_completed_required_cells':3648,'original_transport_and_recovery_failures_preserved':True,'checks':checks,'check_count':len(checks),'evidence_sha256':evidence,'models_or_arrays_or_numerical_programs_reopened':False,'clock_read_or_issued':False,'scientific_accuracy_or_speedup_admission':False,'proxy_cleanup_separate_from_product_completion':True}
if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--paired-root-sha256',required=True);p.add_argument('--paired-independent-sha256',required=True);p.add_argument('--completion-sha256',required=True);p.add_argument('--completion-tool-evidence',type=Path,required=True);p.add_argument('--completion-session',type=int);a=p.parse_args();r=audit(a.paired_root_sha256,a.paired_independent_sha256,a.completion_sha256,a.completion_tool_evidence,a.completion_session);out=P/'sand_original_analysis_completion_independent_review_code_audit_v1.json'
    with out.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':r['status'],'checks':r['check_count'],'receipt_sha256':sha(out)}))
