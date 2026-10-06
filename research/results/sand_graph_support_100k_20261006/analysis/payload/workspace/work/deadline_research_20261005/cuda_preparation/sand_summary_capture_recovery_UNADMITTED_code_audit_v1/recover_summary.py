#!/usr/bin/env python3
"""Retrieve already-completed original Sand summary; no worker retry or new clock."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
K=Path(__file__).resolve().parent;P=K.parent;S=P/'sand_stopped_analysis_released_root_v1';BASE=P/'sand_scalar_recovery_UNADMITTED_code_audit_v1'
BASE_SHA='71c50f558082d127757cc84dff97f7451c7c5e73ff4b5ef48ff41be1af851a79';BASE_REVIEW=P/'sand_scalar_recovery_independent_transition_v1.json';BASE_REVIEW_SHA='e8e4085ffb2847187928e6c716b3a24bee5bce2c3ea901b304adc72b81efe593'
PH='0ac7a368dd8afa921c8b973c5a8a760af7f10eb9f3413bceb592fff0c0ad7a5c';OP='sand_summarize';TAG='sand_summarize.capture_recovery1'
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);need(p.resolve()==p and p.is_file() and not p.is_symlink(),'Canonical regular source/evidence required');return json.loads(p.read_bytes())
def put(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load_base():
    need(sha(BASE/'manifest.json')==BASE_SHA and sha(BASE_REVIEW)==BASE_REVIEW_SHA,'Exact already-reviewed base recovery required')
    for n,h in read(BASE/'manifest.json')['files_sha256'].items():need(sha(BASE/n)==h,'Base source changed '+n)
    spec=importlib.util.spec_from_file_location('sand_reviewed_scalar_recovery_base',BASE/'recover.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def prerequisites(package_sha,review_path,review_sha):
    need(not sys.flags.optimize and sha(K/'manifest.json')==package_sha,'Exact summary capture recovery package required')
    for n,h in read(K/'manifest.json')['files_sha256'].items():need(sha(K/n)==h,'Summary capture recovery source changed '+n)
    need(sha(review_path)==review_sha,'Independent review bytes changed');review=read(review_path);need(review['schema']=='sand_summary_capture_recovery_independent_review_v1' and review['status']=='passed_source_and_synthetic_review' and review['source_manifest_sha256']==package_sha,'Exact independent summary capture source review required')
    for p,h in read(K/'failure_bindings.json')['files_sha256'].items():need(sha(p)==h,'Original summary failure/exit evidence changed')
    failed=read(S/(OP+'.capture.external.json'));error=(S/(OP+'.capture.stderr')).read_text()
    need(failed['exit_code']==255 and failed['local_transport_reaped'] is True and failed['local_transport_timeout'] is False and failed['signals_to_own_local_group']==[] and 'proxyconnect tcp: dial tcp 127.0.0.1:56069: connect: operation not permitted' in error and (S/(OP+'.capture.stdout')).read_bytes()==b'','Exact local pre-connection sandbox failure required')
    base=load_base();o=base.fixed_preconditions(BASE_SHA,BASE_REVIEW,BASE_REVIEW_SHA);record=o.original_exit(S,OP)
    need(record['original_tool_session']==19303 and record['original_tool_exit_code']==0 and sha(S/'analysis_phase.json')==PH,'Genuine original summary19303exit0 inside unchanged phase required')
    need(not (S/(OP+'.collected')).exists() and not (S/(OP+'.review.json')).exists(),'Original decoded summary and review must remain fresh');o.budget(S,20);return base,o
def capture_original(o,package_sha,review_sha):
    put(S/(TAG+'.invocation.json'),{'phase_sha256':PH,'source_manifest_sha256':package_sha,'source_review_sha256':review_sha,'original_worker_session':19303,'original_failed_tag':OP+'.capture','new_capture_tag':TAG,'original_summary_worker_rerun':False,'new_or_restarted_clock_granted':False})
    original=o.probe
    def remapped(state,role,payload,tag,stop=None,mono_stop=None):
        need(state==S and role=='A' and payload['action']=='capture' and tag==OP+'.capture','Only original completed summary capture can be remapped')
        return original(state,role,payload,TAG,stop,mono_stop)
    o.probe=remapped;o.close_product(S,OP);o.budget(S,10)
    put(S/(OP+'.recovery_closure_receipt.json'),{'phase_sha256':PH,'accepted_root_review_sha256':sha(S/(OP+'.review.json')),'actual_capture_tag':TAG,'capture_sha256':sha(S/(TAG+'.json')),'source_manifest_sha256':package_sha,'source_review_sha256':review_sha,'original_worker_session':19303,'original_failed_capture_preserved':True,'new_clock_granted':False})
def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--root-execute',action='store_true');p.add_argument('--package-sha256');p.add_argument('--source-review',type=Path);p.add_argument('--source-review-sha256');p.add_argument('--route-ready',type=Path);p.add_argument('--route-ready-sha256');a=p.parse_args()
    if not a.root_execute:print(json.dumps({'status':'inert_completed_summary_capture_recovery','new_clock_granted':False}));return
    base,o=prerequisites(a.package_sha256,a.source_review,a.source_review_sha256);env=base.route_environment(a.route_ready,a.route_ready_sha256);os.environ.clear();os.environ.update(env);capture_original(o,a.package_sha256,a.source_review_sha256)
if __name__=='__main__':main()
