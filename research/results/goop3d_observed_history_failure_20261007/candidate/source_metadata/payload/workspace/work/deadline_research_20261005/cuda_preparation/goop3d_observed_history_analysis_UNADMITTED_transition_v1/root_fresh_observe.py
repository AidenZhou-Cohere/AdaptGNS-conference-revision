"""Exec the exact previously reviewed42-PID observer through its40s route."""
import argparse
import os
import sys
from observed_common import need,strict
from root_prepare import package
from root_run import proxy_environment
from root_stage_and_run import read_new_metadata
from prephase_sources import *

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root-observe',action='store_true');p.add_argument('--package-sha256');args=p.parse_args()
    if not args.root_observe:print('{"status":"inert_exact_fresh_observer","clock_granted":false}');return
    package(args.package_sha256);check_sources()
    raw,pin=read_new_metadata(PROXY_OUTPUT/'ready.json');ready=strict(raw)
    need(ready['schema']=='coder_exact_route_proxy_v1' and ready['lifetime_seconds']==40 and ready['authority']=='coder.internal.cohere.com:443' and ready['upstream']==['100.106.33.61',443] and ready['opaque_tls'] is True and ready['no_settings_changed'] is True,'exact40second metadata route')
    proxy_environment(ready['url'])
    argv=[sys.executable,'-B',str(OBSERVER/'root_observe.py'),'--root-observe','--package-sha256',OBSERVER_SHA,'--output',str(OBSERVER_OUTPUT),'--https-proxy',ready['url']]
    os.execve(sys.executable,argv,os.environ.copy())

if __name__=='__main__':main()
