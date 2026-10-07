"""Root-only path adaptation for a second, distinct metadata observation.

No phase, staging, worker, clock, network, or native observation is executed on
import. All operational functions remain from the exact frozen D3 package.
"""
import argparse
import importlib
import json
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
PREP=HERE.parent
PACKAGE=PREP/'goop3d_observed_history_analysis_UNADMITTED_transition_v1'
PACKAGE_SHA='1dee2ec7c67639fab665e9fe88ce2cd6fb5862ac9eb9289fafb29446c9165e62'
PROXY_OUTPUT=PREP/'goop3d_observed_fresh_route_attempt2_root_v1'
OBSERVER_OUTPUT=PREP/'goop3d_observed_fresh_metadata_attempt2_root_v1'
ENTRY_PARENT=PREP/'goop3d_observed_fresh_entry_attempt2_root_v1'
ENTRY_OUTPUT=ENTRY_PARENT/'goop3d_observed_fresh_entry_root_v1'
PLAN=PREP/'goop3d_observed_metadata_attempt2_plan_root_v1.json'
FIRST_PLAN=PREP/'goop3d_observed_local_prephase_plan_root_v1.json'
FIRST_PLAN_SHA='dca8a7843ffa56deef24d4c21c124c2bba3ac1dda641642147488363f73d39e5'

def load_original():
    # Original imports resolve within the frozen package, never this wrapper.
    sys.path.insert(0,str(PACKAGE))
    common=importlib.import_module('observed_common')
    prepare=importlib.import_module('root_prepare')
    common.need(Path(prepare.__file__).resolve()==PACKAGE/'root_prepare.py','exact original phase issuer module')
    prepare.package(PACKAGE_SHA)
    return common,prepare

def verify_wrapper(common,pin):
    manifest=common.strict(common.read_bound(HERE/'manifest.json',pin,common.CAP_JSON))
    common.need(manifest['source_manifest_sha256']==PACKAGE_SHA and manifest['scope']=='metadata_attempt2_path_constants_only','exact metadata-only path adaptation')
    for name,sha in manifest['files_sha256'].items():
        path=HERE/name
        common.need(path.resolve().is_relative_to(HERE),'wrapper relative path')
        common.read_bound(path,sha,common.CAP_JSON,retain=False)
    for binding in manifest['preserved_attempt1'].values():
        common.read_bound(binding['path'],binding['sha256'],common.CAP_JSON,retain=False)
    return manifest

def bind_paths(action):
    prephase=importlib.import_module('prephase_sources')
    prephase.PROXY_OUTPUT=PROXY_OUTPUT
    prephase.OBSERVER_OUTPUT=OBSERVER_OUTPUT
    module=importlib.import_module('root_fresh_observe' if action=='observe' else 'root_fresh_begin')
    # The entry assembler's PREP is used only for its local output path. Its
    # HERE, original root_prepare.PREP, phase STATE, and start commands stay fixed.
    if action=='begin':module.PREP=ENTRY_PARENT
    return module

def original_argv(action,remaining):
    return ['root_fresh_observe.py' if action=='observe' else 'root_fresh_begin.py',*remaining]

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('plan','observe','begin'),nargs='?')
    p.add_argument('--root-attempt2',action='store_true')
    p.add_argument('--wrapper-sha256')
    args,remaining=p.parse_known_args()
    if not args.root_attempt2:
        print(json.dumps({'status':'inert_metadata_attempt2_wrapper','clock_granted':False}));return
    common,prepare=load_original();verify_wrapper(common,args.wrapper_sha256)
    prephase=importlib.import_module('prephase_sources');prephase.check_sources()
    if args.action=='plan':
        common.need(not remaining,'no unrecognized plan arguments')
        common.need(not PLAN.exists() and not ENTRY_PARENT.exists() and not PROXY_OUTPUT.exists() and not OBSERVER_OUTPUT.exists(),'fresh distinct metadata attempt2 paths')
        common.need(not importlib.import_module('phase_binding').STATE.exists(),'no phase has been issued')
        prior=common.strict(common.read_bound(FIRST_PLAN,FIRST_PLAN_SHA,common.CAP_JSON))
        common.need(prior['source_manifest_sha256']==PACKAGE_SHA and prior['remote_staging_performed'] is False and prior['clock_issued'] is False,'reuse only original local source payload')
        source=common.read_bound(prior['source_payload_path'],prior['source_payload_sha256'],common.CAP_JSON)
        bindings=common.strict(common.read_bound(PACKAGE/'remote_source_bindings.json',prepare.package(PACKAGE_SHA)['files_sha256']['remote_source_bindings.json'],common.CAP_JSON))
        importlib.import_module('staging_plan').source_payload(source,bindings)
        proxy=[sys.executable,'-B',str(prephase.PROXY),'--output',str(PROXY_OUTPUT),'--seconds','40']
        observe=[sys.executable,'-B',str(HERE/'attempt2.py'),'observe','--root-attempt2','--wrapper-sha256',args.wrapper_sha256,'--root-observe','--package-sha256',PACKAGE_SHA]
        ENTRY_PARENT.mkdir()
        plan={'schema':'goop3d_metadata_attempt2_command_plan_v1','source_manifest_sha256':PACKAGE_SHA,'wrapper_manifest_sha256':args.wrapper_sha256,'preserved_local_source_plan_sha256':FIRST_PLAN_SHA,'source_payload_path':prior['source_payload_path'],'source_payload_sha256':prior['source_payload_sha256'],'metadata_proxy_argv':proxy,'metadata_observe_argv':observe,'metadata_proxy_command':shlex.join(proxy),'metadata_observe_command':shlex.join(observe),'metadata_proxy_directory':str(PROXY_OUTPUT),'fresh_observer_directory':str(OBSERVER_OUTPUT),'fresh_entry_directory':str(ENTRY_OUTPUT),'remote_staging_performed':False,'clock_issued':False}
        pin=common.publish(PLAN,plan)
        print(json.dumps({'plan_path':str(PLAN),'plan_sha256':pin,**plan},sort_keys=True));return
    common.need(args.action in ('observe','begin'),'explicit original operational action')
    plan=common.strict(PLAN.read_bytes())
    common.need(plan['wrapper_manifest_sha256']==args.wrapper_sha256 and plan['source_manifest_sha256']==PACKAGE_SHA and plan['metadata_proxy_directory']==str(PROXY_OUTPUT) and plan['fresh_observer_directory']==str(OBSERVER_OUTPUT) and plan['fresh_entry_directory']==str(ENTRY_OUTPUT),'original attempt2 local plan binding')
    common.need('--package-sha256' in remaining and remaining[remaining.index('--package-sha256')+1]==PACKAGE_SHA,'unchanged original package argument')
    if args.action=='observe':
        common.need(remaining==['--root-observe','--package-sha256',PACKAGE_SHA],'exact original observer arguments')
    else:
        common.need('--root-begin' in remaining and '--observer-directory' in remaining and Path(remaining[remaining.index('--observer-directory')+1])==OBSERVER_OUTPUT,'only original begin with exact attempt2 observer')
    module=bind_paths(args.action)
    previous=sys.argv
    try:
        sys.argv=original_argv(args.action,remaining)
        module.main()
    finally:sys.argv=previous

if __name__=='__main__':main()
