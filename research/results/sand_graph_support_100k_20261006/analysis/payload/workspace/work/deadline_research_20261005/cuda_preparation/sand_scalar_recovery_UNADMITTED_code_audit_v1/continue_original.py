#!/usr/bin/env python3
"""Narrow environment-only continuation of the unchanged original Sand operator."""
from pathlib import Path
import argparse,json,os,sys
from recover import S,F,FROZEN,fixed_preconditions,route_environment,need
ACTIONS=('run','record-original-exit','close-product','complete')
OPS=('sand_summarize','sand_paired')
def argv_for(args):
    need(args.action in ACTIONS,'Only original final actions')
    out=[sys.executable,'-B',str(F/'root_operator.py'),args.action,'--root-action','--state',str(S),'--package-sha256',FROZEN]
    extras=(args.session_id,args.exit_code,args.observed_utc,args.evidence_file)
    if args.action=='complete':need(args.operation is None and all(v is None for v in extras),'No complete overrides');return out
    need(args.operation in OPS,'Only never-run original summary/paired operations');out+=['--operation',args.operation]
    if args.action=='record-original-exit':
        need(type(args.session_id) is int and args.session_id>0 and type(args.exit_code) is int and args.observed_utc is not None and args.evidence_file is not None,'Actual original tool exit fields required')
        out+=['--session-id',str(args.session_id),'--exit-code',str(args.exit_code),'--observed-utc',args.observed_utc,'--evidence-file',str(args.evidence_file)]
    else:need(all(v is None for v in extras),'Unexpected original action overrides')
    return out
def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('action',choices=ACTIONS);p.add_argument('--root-execute',action='store_true');p.add_argument('--package-sha256');p.add_argument('--source-review',type=Path);p.add_argument('--source-review-sha256');p.add_argument('--route-ready',type=Path);p.add_argument('--route-ready-sha256');p.add_argument('--operation',choices=OPS);p.add_argument('--session-id',type=int);p.add_argument('--exit-code',type=int);p.add_argument('--observed-utc');p.add_argument('--evidence-file',type=Path);a=p.parse_args()
    if not a.root_execute:print(json.dumps({'status':'inert_original_final_actions_wrapper','new_clock_granted':False}));return
    o=fixed_preconditions(a.package_sha256,a.source_review,a.source_review_sha256);env=route_environment(a.route_ready,a.route_ready_sha256);argv=argv_for(a)
    if a.action=='run':need(not list(S.glob(a.operation+'.*')),'An original summary/paired invocation already exists; no retry')
    o.budget(S,15);os.execve(sys.executable,argv,env)
if __name__=='__main__':main()
