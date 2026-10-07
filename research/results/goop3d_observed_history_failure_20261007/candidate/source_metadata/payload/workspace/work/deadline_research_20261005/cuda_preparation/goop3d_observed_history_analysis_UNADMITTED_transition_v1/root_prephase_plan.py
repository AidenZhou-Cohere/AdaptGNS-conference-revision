"""Prepare local sources and exact metadata commands; no remote work or clock."""
import argparse
import contextlib
import io
from pathlib import Path
import shlex
import sys
from observed_common import strict,publish
from root_prepare import package
import root_prepare
from prephase_sources import *

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-prepare',action='store_true');p.add_argument('--package-sha256');args=p.parse_args()
    if not args.root_prepare:print('{"status":"inert_local_prephase_plan","clock_granted":false}');return
    package(args.package_sha256);check_sources()
    previous=sys.argv;captured=io.StringIO()
    try:
        sys.argv=['root_prepare.py','source-payload','--root-prepare','--package-sha256',args.package_sha256]
        with contextlib.redirect_stdout(captured):root_prepare.main()
    finally:sys.argv=previous
    source=strict(captured.getvalue())
    proxy=[sys.executable,'-B',str(PROXY),'--output',str(PROXY_OUTPUT),'--seconds','40']
    observe=[sys.executable,'-B',str(HERE/'root_fresh_observe.py'),'--root-observe','--package-sha256',args.package_sha256]
    plan={'schema':'goop3d_local_prephase_command_plan_v1','source_manifest_sha256':args.package_sha256,'source_payload_path':source['payload'],'source_payload_sha256':source['sha256'],'metadata_proxy_source_sha256':PROXY_SHA,'metadata_observer_manifest_sha256':OBSERVER_SHA,'metadata_proxy_argv':proxy,'metadata_observe_argv':observe,'metadata_proxy_command':shlex.join(proxy),'metadata_observe_command':shlex.join(observe),'metadata_proxy_directory':str(PROXY_OUTPUT),'fresh_observer_directory':str(OBSERVER_OUTPUT),'remote_staging_performed':False,'clock_issued':False}
    path=PREP/'goop3d_observed_local_prephase_plan_root_v1.json';pin=publish(path,plan)
    import json
    print(json.dumps({'plan_path':str(path),'plan_sha256':pin,**plan},sort_keys=True))

if __name__=='__main__':main()
