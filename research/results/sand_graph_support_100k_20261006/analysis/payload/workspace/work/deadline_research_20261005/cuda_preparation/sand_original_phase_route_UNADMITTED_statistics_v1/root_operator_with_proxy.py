"""Root-only CLI/environment wrapper; execs the unchanged original Sand operator.

Never begins a phase or starts a proxy. Existing phase time pays for every action.
"""
from pathlib import Path
import argparse
import json
import os
import sys
from urllib.parse import urlsplit
from phase_binding import (STATE,OPERATOR,OPERATOR_SHA,FROZEN_PACKAGE,load_original_phase,
                           need,read_pinned,strict,stamp)

ACTIONS=('stage-runtime','run','record-original-exit','close-product','transfer-B','complete')

def proxy_environment(value):
    parsed=urlsplit(value)
    need(parsed.scheme=='http' and parsed.hostname=='127.0.0.1' and parsed.username is None and parsed.password is None and parsed.path=='' and not parsed.query and not parsed.fragment and parsed.port is not None and 0<parsed.port<65536 and value=='http://127.0.0.1:'+str(parsed.port),'exact uncredentialed loopback HTTP proxy required')
    env=os.environ.copy(); env['HTTPS_PROXY']=value; env['https_proxy']=value
    env.pop('NO_PROXY',None); env.pop('no_proxy',None)
    return env

def operator_argv(action,extra):
    need(action in ACTIONS,'only unchanged post-begin actions permitted')
    if extra[:1]==['--']: extra=extra[1:]
    need(not any(x.split('=',1)[0] in ('--state','--package-sha256','--root-action') for x in extra),'fixed original operator state/package/root flag cannot be overridden')
    return [sys.executable,'-B',str(OPERATOR),action,'--root-action','--state',str(STATE),'--package-sha256',FROZEN_PACKAGE,*extra]

def ready_environment(raw,phase):
    ready=strict(raw)
    need(type(ready) is dict and ready.get('schema')=='coder_sand_original_phase_route_proxy_v1','reviewed successor ready record required')
    need(all(ready.get(k)==v for k,v in phase.metadata.items()),'route must bind the exact existing phase and anchor')
    need(ready.get('authority')=='coder.internal.cohere.com:443' and ready.get('upstream')==['100.106.33.61',443] and ready.get('opaque_tls') is True and ready.get('no_settings_changed') is True,'exact original route and opaque TLS required')
    need(phase.start<=stamp(ready['started_utc']).timestamp()<phase.stop,'proxy startup outside original phase')
    phase.remaining()
    return proxy_environment(ready['url'])

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--root-execute',action='store_true'); parser.add_argument('--phase-sha256'); parser.add_argument('--anchor-sha256')
    parser.add_argument('--route-ready',type=Path); parser.add_argument('--route-ready-sha256')
    parser.add_argument('action',choices=ACTIONS,nargs='?'); parser.add_argument('operator_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.root_execute:
        print(json.dumps({'status':'inert_original_operator_proxy_wrapper','new_clock_granted':False})); return
    need(args.action is not None and args.route_ready is not None,'explicit post-begin action and existing route ready record required')
    phase=load_original_phase(args.phase_sha256,args.anchor_sha256)
    read_pinned(OPERATOR,OPERATOR_SHA,256*1024)
    read_pinned(OPERATOR.parent/'manifest.json',FROZEN_PACKAGE,256*1024)
    env=ready_environment(read_pinned(args.route_ready,args.route_ready_sha256),phase)
    argv=operator_argv(args.action,args.operator_args)
    phase.remaining()
    # Replaces this wrapper: no background process, new proxy, new clock or retry.
    os.execve(sys.executable,argv,env)

if __name__=='__main__': main()
