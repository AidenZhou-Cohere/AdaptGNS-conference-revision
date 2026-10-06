#!/usr/bin/env python3
"""Root-only same-hour retrieval/transfer recovery; never launches a scientific worker."""
from pathlib import Path
import argparse,base64,datetime as D,hashlib,importlib.util,json,math,os,shlex,signal,subprocess,sys,time
K=Path(__file__).resolve().parent;P=K.parent;S=P/'sand_stopped_analysis_released_root_v1'
F=P/'sand_stopped_analysis_operator_UNADMITTED_code_audit_v1'
R='/root/repos/AdaptGNS-cuda-20261006';A=R+'/sand_final_analysis_20261006_v1';PY=R+'/.venv/bin/python'
PH='0ac7a368dd8afa921c8b973c5a8a760af7f10eb9f3413bceb592fff0c0ad7a5c';AN='06cee5e7a2372fb5799678cfa9493f91a0e88fc3ba45ebc1a70bc15944295fd1';FROZEN='914e6e2df11836079f721a675bdbfcd9673b6bada3074fe766c52d085a37640b'
OPS=('observe','close-saved-A','transfer-B-collection','transfer-B-audit','complete-transfer')
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'Duplicate JSON key');out[k]=v
        return out
    def reject(x):raise ValueError('Nonfinite JSON')
    p=Path(path);need(p.resolve()==p and not p.is_symlink() and p.is_file(),'Canonical regular local file');return json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=reject)
def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def pinned(path,expected):need(sha(path)==expected,'Pinned source/evidence changed: '+str(path));return read(path)
def load_operator():
    sys.path.insert(0,str(F));spec=importlib.util.spec_from_file_location('sand_frozen_recovery_operator',F/'root_operator.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def fixed_preconditions(package_sha,review_path,review_sha):
    need(not sys.flags.optimize,'Nonoptimized recovery required');mf=pinned(K/'manifest.json',package_sha)
    for n,h in mf['files_sha256'].items():need(sha(K/n)==h,'Recovery source changed '+n)
    review=pinned(review_path,review_sha);need(review['schema']=='sand_scalar_recovery_independent_review_v1' and review['status']=='passed_source_and_synthetic_review' and review['source_manifest_sha256']==package_sha,'Exact independently reviewed recovery package required')
    need(sha(F/'manifest.json')==FROZEN,'Frozen operator changed');original=read(F/'manifest.json')
    for n,h in original['files_sha256'].items():need(sha(F/n)==h,'Frozen operator payload changed '+n)
    need(sha(S/'analysis_phase.json')==PH and sha(S/'local_phase_anchor.json')==AN,'Original phase/anchor changed')
    failures=read(K/'failure_bindings.json')
    for path,h in failures['files_sha256'].items():need(sha(path)==h,'Original failure/closure evidence changed')
    closure=read(P/'sand_original_phase_proxy_and_failed_transports_closed_root_v1.json');need(closure['parser_self_verified'] is True and closure['matching_rows']==[] and closure['checked_pids_and_groups']==[46711,47926,48109] and closure['proxy_terminal']['remaining_worker_threads']==0 and closure['proxy_terminal']['analysis_phase_sha256']==PH and closure['proxy_terminal']['local_phase_anchor_sha256']==AN,'Exact old proxy/failed local native closure required')
    need(read(P/'sand_original_phase_proxy_original_exit_root_v1.json')['exit_code']==130,'Genuine original proxy tool exit required')
    for tag in ('sand_saved_A.capture','B.scalars_to_A'):
        ext=read(S/(tag+'.external.json'));need(ext['exit_code']==255 and ext['local_transport_reaped'] is True and ext['local_transport_timeout'] is False and ext['signals_to_own_local_group']==[],'Original failed transport must remain genuine255 and reaped')
    o=load_operator();o.package_check(FROZEN);o.original_exit(S,'sand_saved_A');record=read(S/'sand_saved_A.root_original_external_exit.json');need(record['original_tool_session']==66211 and record['original_tool_exit_code']==0,'Genuine original saved_A66211exit0 required');o.budget(S,25);return o
def route_environment(ready_path,ready_sha):
    from urllib.parse import urlsplit
    ready=pinned(ready_path,ready_sha);phase=read(S/'analysis_phase.json');anchor=read(S/'local_phase_anchor.json')
    need(ready['schema']=='coder_sand_original_phase_recovery_route_proxy_v1' and ready['analysis_phase_sha256']==PH and ready['local_phase_anchor_sha256']==AN and ready['original_analysis_started_utc']==phase['original_analysis_started_utc'] and ready['original_analysis_stop_utc']==phase['original_analysis_stop_utc'] and ready['original_monotonic_stop']==anchor['monotonic_seconds']+3600 and ready['new_or_restarted_clock_granted'] is False,'Reviewed successor ready must bind original phase')
    need(ready['authority']=='coder.internal.cohere.com:443' and ready['upstream']==['100.106.33.61',443] and ready['opaque_tls'] is True and ready['no_settings_changed'] is True,'Exact opaque successor route required')
    need(ready['analysis_phase_path']==str(S/'analysis_phase.json') and ready['connection_cap']==2048 and ready['concurrency_cap']==8 and ready['per_direction_buffer_bytes']==262144 and ready['payload_backpressure']=='nonblocking_partial_send' and ready['original_route_package_sha256']=='292cc31d54e822d6eb1a843da3ce55427f9f89201c1fb9a2dd7293d0660dfd80','Exact separately reviewed successor route limits and original phase path')
    value=ready['url'];u=urlsplit(value);need(u.scheme=='http' and u.hostname=='127.0.0.1' and u.username is None and u.password is None and not u.path and not u.query and not u.fragment and u.port is not None and value=='http://127.0.0.1:'+str(u.port),'Exact uncredentialed loopback proxy URL')
    env=os.environ.copy();env['HTTPS_PROXY']=env['https_proxy']=value;env.pop('NO_PROXY',None);env.pop('no_proxy',None);return env
def targets(o):
    result={}
    for op in ('sand_collect_B','sand_saved_B'):
        receipt,_=o.accepted(S,op);path=Path(receipt['local_product']);result[o.spec(op)['outputs'][0]]={'sha256':receipt['product_sha256'],'bytes':path.stat().st_size,'local':str(path),'review_sha256':sha(S/(op+'.review.json'))}
    return result
def fate_payload(o):
    return {'phase_sha256':PH,'stop_utc':read(S/'analysis_phase.json')['original_analysis_stop_utc'],'historical_pids':o.historical_pids(S,'A'),'original_probe_sha256':sha(F/'remote_probe.py'),'original_failed_remote_command_sha256':[hashlib.sha256(read(S/(tag+'.command.json'))['argv'][-1].encode()).hexdigest() for tag in ('sand_saved_A.capture','B.scalars_to_A')],'saved_A_release_sha256':sha(S/'sand_saved_A.cpu_release.json'),'B_targets':{p:{k:v for k,v in row.items() if k in ('sha256','bytes')} for p,row in targets(o).items()}}
def fate_probe(o,tag):
    _,stop,mono,remaining=o.budget(S,30);seconds=math.floor(remaining)-10;need(seconds>0,'No bounded fate allowance');payload=fate_payload(o)
    argv=o.ssh_argv('A',['/usr/bin/timeout','--signal=TERM','--kill-after=3s',str(seconds)+'s',PY,'-I','-S','-B','-c',(K/'remote_fate.py').read_text()])
    raw=o.bounded_local(S,argv,tag,stop,mono,json.dumps(payload));value=json.loads(raw);need(value['schema']=='sand_original_scalar_recovery_fate_v1' and value['status']=='metadata_observed' and value['phase_sha256']==PH and value['saved_A_release_sha256']==payload['saved_A_release_sha256'] and value['no_files_written'] is True and value['no_scientific_program_run'] is True and value['assigned_devices_empty'] is True and all(v is True for v in value['native_absent'].values()),'Retained original metadata/fate observation required');put(S/(tag+'.json'),value);o.budget(S,15);return value
def accepted_fate(o,path,pin):
    review=pinned(path,pin);need(review['schema']=='sand_original_scalar_fate_independent_review_v1' and review['status']=='passed_original_saved_A_closure_and_B_file_fates' and review['phase_sha256']==PH and review['fate_capture_sha256']==sha(S/'sand_recovery1.fate.json'),'Actual independent fate review required')
    value=read(S/'sand_recovery1.fate.json');need(value['phase_sha256']==PH and value['saved_A_release_sha256']==sha(S/'sand_saved_A.cpu_release.json') and all(v is True for v in value['native_absent'].values()),'Original fate binding changed')
    phase=read(S/'analysis_phase.json');need(o.dt(phase['original_analysis_started_utc'])<=o.dt(value['checked_utc'])<=o.now()<o.dt(phase['original_analysis_stop_utc']),'Fate observation must belong to this original hour')
    expected=targets(o);need(set(value['B_targets'])==set(expected),'Exact B targets required')
    for p,row in value['B_targets'].items():need(row['state'] in ('absent','exact') and (row['state']=='absent' or {k:row[k] for k in ('sha256','bytes')}=={k:expected[p][k] for k in ('sha256','bytes')}),'Conflicting retained target cannot be overwritten')
    return value
def stream(o,tag,argv,path):
    _,stop,mono,remaining=o.budget(S,25);start=o.now();tick=time.monotonic();timeout=remaining-12;need(timeout>0,'No stream transport allowance');put(S/(tag+'.command.json'),{'argv':argv,'timeout_seconds':timeout,'stop_utc':stop.isoformat(),'stdin_file':str(path),'stdin_sha256':sha(path),'streamed_binary':True})
    timed=False;signals=[];failure=None;out=b'';err=b'';remaining=lambda:min((stop-o.now()).total_seconds(),mono-time.monotonic())
    with Path(path).open('rb') as source:
        proc=subprocess.Popen(argv,stdin=source,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        def send(sig):
            try:os.killpg(proc.pid,sig);signals.append(signal.Signals(sig).name)
            except ProcessLookupError:pass
        def tail():return max(0,min(3,remaining()-5))
        try:
            try:out,err=proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed=True;send(signal.SIGTERM)
                try:out,err=proc.communicate(timeout=tail())
                except subprocess.TimeoutExpired:send(signal.SIGKILL);out,err=proc.communicate(timeout=tail())
        except BaseException as e:
            failure=type(e).__name__+': '+str(e)
            if proc.poll() is None:
                send(signal.SIGTERM)
                try:out,err=proc.communicate(timeout=tail())
                except subprocess.TimeoutExpired:
                    send(signal.SIGKILL)
                    try:out,err=proc.communicate(timeout=tail())
                    except subprocess.TimeoutExpired:pass
            raise
        finally:
            with (S/(tag+'.stdout')).open('xb') as f:f.write(out)
            with (S/(tag+'.stderr')).open('xb') as f:f.write(err)
            put(S/(tag+'.external.json'),{'started_utc':start.isoformat(),'observed_utc':o.now().isoformat(),'elapsed_seconds':time.monotonic()-tick,'exit_code':proc.returncode,'failure':failure,'local_transport_timeout':timed,'signals_to_own_local_group':signals,'local_transport_pid':proc.pid,'local_transport_reaped':proc.poll() is not None,'remote_closure_separately_required':True})
    need(not timed and proc.returncode==0 and not signals and proc.poll() is not None,'Stream transport failed; preserve original attempt and do not retry');need(remaining()>5,'Original hour expired during stream');return json.loads(out)
def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('action',choices=OPS);p.add_argument('--root-execute',action='store_true');p.add_argument('--package-sha256');p.add_argument('--source-review',type=Path);p.add_argument('--source-review-sha256');p.add_argument('--route-ready',type=Path);p.add_argument('--route-ready-sha256');p.add_argument('--fate-review',type=Path);p.add_argument('--fate-review-sha256');args=p.parse_args()
    if not args.root_execute:print(json.dumps({'status':'inert_recovery_preparation','new_clock_granted':False}));return
    o=fixed_preconditions(args.package_sha256,args.source_review,args.source_review_sha256);env=route_environment(args.route_ready,args.route_ready_sha256);os.environ.clear();os.environ.update(env)
    if args.action=='observe':
        need(not (S/'sand_recovery1.fate.json').exists(),'Original recovery fate already exists');print(json.dumps(fate_probe(o,'sand_recovery1.fate')));return
    fate=accepted_fate(o,args.fate_review,args.fate_review_sha256)
    if args.action=='close-saved-A':
        need(not (S/'sand_saved_A.collected').exists() and not (S/'sand_saved_A.review.json').exists(),'Original saved_A decoded product/review must remain fresh')
        put(S/'sand_saved_A.capture_recovery1.invocation.json',{'phase_sha256':PH,'package_sha256':args.package_sha256,'source_review_sha256':args.source_review_sha256,'fate_review_sha256':args.fate_review_sha256,'original_failed_capture_preserved':True,'original_worker_rerun':False})
        original=o.probe
        def remapped(state,role,payload,tag,stop=None,mono_stop=None):
            need(state==S and role=='A' and payload['action']=='capture' and tag=='sand_saved_A.capture','Only original saved_A completed capture may be remapped')
            return original(state,role,payload,'sand_saved_A.capture_recovery1',stop,mono_stop)
        o.probe=remapped;o.close_product(S,'sand_saved_A');o.budget(S,10)
        put(S/'sand_saved_A.recovery_closure_receipt.json',{'phase_sha256':PH,'accepted_root_review_sha256':sha(S/'sand_saved_A.review.json'),'actual_capture_tag':'sand_saved_A.capture_recovery1','capture_sha256':sha(S/'sand_saved_A.capture_recovery1.json'),'original_worker_session':66211,'original_failed_capture_preserved':True,'new_clock_granted':False});return
    if args.action.startswith('transfer-B-'):
        kind='collection' if args.action=='transfer-B-collection' else 'audit';name='B.stopped_collection.json' if kind=='collection' else 'B.saved_array_audit.json';target=A+'/'+name;row=targets(o)[target];observed=fate['B_targets'][target]['state'];tag='B.transfer_'+kind+'_recovery1'
        need(not (S/'B.scalars_transferred_to_A.json').exists(),'Original transfer final receipt must remain fresh');o.budget(S,30)
        payload={'phase_sha256':PH,'stop_utc':read(S/'analysis_phase.json')['original_analysis_stop_utc'],'target':target,'sha256':row['sha256'],'bytes':row['bytes'],'observed_fate':observed};encoded=base64.b64encode(json.dumps(payload).encode()).decode();_,stop,mono,remaining=o.budget(S,30);seconds=math.floor(remaining)-10
        argv=o.ssh_argv('A',['/usr/bin/timeout','--signal=TERM','--kill-after=3s',str(seconds)+'s',PY,'-I','-S','-B','-c',(K/'remote_transfer.py').read_text(),encoded])
        if observed=='exact':value=json.loads(o.bounded_local(S,argv,tag,stop,mono))
        else:value=stream(o,tag,argv,Path(row['local']))
        need(value['schema']=='sand_original_scalar_stream_transfer_v1' and value['phase_sha256']==PH and value['target']==target and value['sha256']==row['sha256'] and value['bytes']==row['bytes'] and value['no_original_files_overwritten_or_deleted'] is True,'Exact scalar recovery result required');put(S/(tag+'.json'),value);o.budget(S,10);return
    need(args.action=='complete-transfer','No other recovery action')
    expected=targets(o)
    for kind in ('collection','audit'):
        value=read(S/('B.transfer_'+kind+'_recovery1.json'));ext=read(S/('B.transfer_'+kind+'_recovery1.external.json'));need(ext['exit_code']==0 and ext['local_transport_reaped'] is True and ext['local_transport_timeout'] is False and ext['signals_to_own_local_group']==[] and value['phase_sha256']==PH,'Both original recovery transfers must complete')
    verified=fate_probe(o,'B.transfer_verify_recovery1');need(all(row['state']=='exact' and {k:row[k] for k in ('sha256','bytes')}=={k:expected[path][k] for k in ('sha256','bytes')} for path,row in verified['B_targets'].items()),'Both exact target files must be independently reobserved')
    o.budget(S,10);put(S/'B.scalars_transferred_to_A.json',{'analysis_phase_sha256':PH,'transferred_sha256':{path:row['sha256'] for path,row in expected.items()},'capture_sha256':sha(S/'B.transfer_verify_recovery1.json'),'original_B_review_sha256':{op:sha(S/(op+'.review.json')) for op in ('sand_collect_B','sand_saved_B')},'recovery_metadata':{'original_failed_transport_tag':'B.scalars_to_A','verification_tag':'B.transfer_verify_recovery1','stream_tags':['B.transfer_collection_recovery1','B.transfer_audit_recovery1'],'package_sha256':args.package_sha256,'fate_review_sha256':args.fate_review_sha256,'original_failures_preserved':True,'new_or_restarted_clock_granted':False}})
if __name__=='__main__':main()
