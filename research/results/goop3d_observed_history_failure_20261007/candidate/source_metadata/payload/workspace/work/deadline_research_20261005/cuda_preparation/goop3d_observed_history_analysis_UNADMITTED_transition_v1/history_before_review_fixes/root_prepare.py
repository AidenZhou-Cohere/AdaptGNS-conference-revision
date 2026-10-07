"""Explicit root-only local issuance; no SSH, worker, or route is started here."""
import argparse
import base64
import datetime as D
from pathlib import Path
import time
from observed_common import *
from phase_binding import STATE

HERE=Path(__file__).resolve().parent
PREP=HERE.parent
PIDS={42898,42899,42911,42912,42913,42914,43568,43632,43633,43760,46156,46220,46221,46351,48776,48851,48979,49043,51739,51807,51946,52010,54713,54714,55094,55158,57372,57438,59710,59776,62415,62481,65470,65471,65472,65473,65892,65893,65928,66518,66519,66558}

def package(pin):
    manifest=strict(read_bound(HERE/'manifest.json',pin,CAP_JSON))
    for name,h in manifest['files_sha256'].items():
        path=HERE/name;need(path.resolve().is_relative_to(HERE),'package relative path');read_bound(path,h,CAP_OUTPUT,retain=False)
    return manifest

def fresh_report(raw):
    report=strict(raw)
    need(report['schema']=='goop3d_post_expiry_metadata_observation_v1' and report['host']==HOST and report['boot_id']==report['final_boot_id']==BOOT,'exact original native observer host/boot')
    need(report['native_closure_observed'] is True and report['observed_pid_count']==42 and report['registry_bytes_unchanged'] is True and report['registry_identity_coverage_complete'] is True and report['registry_errors']==[],'complete stable original native closure')
    for name in ('native_first','native_second'):
        need(set(report[name])=={str(x) for x in PIDS} and all(report[name][str(x)]=={'pid':x,'status':'absent'} for x in PIDS),'all42 scoped PIDs absent twice')
    need(report['original_phase_sha256']==OLD_PHASE_SHA and report['original_session']==29448,'original expired identity retained')
    return report

def payload(rows,kind):
    return {'schema':'goop3d_observed_history_staging_payload_v1','kind':kind,'files':[{'path':destination,'sha256':digest(raw),'base64':base64.b64encode(raw).decode()} for destination,raw in rows]}

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('action',choices=('source-payload','begin'),nargs='?')
    p.add_argument('--root-prepare',action='store_true');p.add_argument('--package-sha256')
    for n in ('fresh-observation','root-clock-json'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-sha256')
    args=p.parse_args()
    if not args.root_prepare:print(json.dumps({'status':'inert_root_preparation','allocation_issued':False}));return
    m=package(args.package_sha256)
    if args.action=='source-payload':
        stage=strict((HERE/'remote_source_bindings.json').read_bytes());rows=[]
        for name,pin in stage['files_sha256'].items():rows.append((SOURCE+'/'+name,read_bound(HERE/name,pin,CAP_JSON)))
        result=payload(rows,'sources');out=PREP/'goop3d_observed_history_source_payload_root_v1.json'
        print(json.dumps({'payload':str(out),'sha256':publish(out,result),'allocation_issued':False}));return
    need(args.action=='begin' and not STATE.exists(),'one explicit fresh observed-history state required')
    raw=read_bound(args.fresh_observation,args.fresh_observation_sha256,CAP_JSON);report=fresh_report(raw)
    rootraw=read_bound(args.root_clock_json,args.root_clock_json_sha256,CAP_JSON);clock_doc=strict(rootraw)
    root_time=stamp(clock_doc['current_time'].replace(' UTC','+00:00'))
    now=D.datetime.now(D.timezone.utc);tick=time.monotonic();start=stamp(report['finished']['utc']);end=start+D.timedelta(seconds=3600)
    need(abs((now-root_time).total_seconds())<=5 and 0<=(now-start).total_seconds()<=5 and abs((start-root_time).total_seconds())<=5,'fresh independent root clock and host observation required')
    need(end<=stamp(GLOBAL_STOP),'a full distinct hour no longer fits')
    prior=read_bound(PREP/'goop3d_completed_retained_fate_observation_independent_transition_v1.json',FATE_REVIEW_SHA,CAP_JSON)
    clock={'source':'root_fresh_tool_and_host_clock_evidence','error_bound_seconds':5,'host_utc':start.isoformat(),'host_monotonic_seconds':report['finished']['monotonic_seconds'],'host_boot_id':BOOT,'root_reference_utc':root_time.isoformat()}
    evidence={'prior_fate_review':{'path':OUTPUT+'/controls/prior_fate_review.json','sha256':FATE_REVIEW_SHA},'fresh_native_observation':{'path':OUTPUT+'/controls/fresh_native_observation.json','sha256':args.fresh_observation_sha256}}
    phase={'schema':PHASE_SCHEMA,'issued_by':'root','status':'approved_bounded_observed_history_analysis','dataset':'Goop3D','scope':'all_2568_observed_cells_and_all_4728_accounting','seconds':3600,'started_utc':start.isoformat(),'stop_utc':end.isoformat(),'global_stop_utc':GLOBAL_STOP,'original_phase_sha256':OLD_PHASE_SHA,'prior_fate_review_sha256':FATE_REVIEW_SHA,'original_phase_modified':False,'automatic_retry':False,'hostname':HOST,'clock_sample':clock,'fresh_native_closure':{'all_42_absent':True,'evidence_sha256':args.fresh_observation_sha256},'operational_evidence':evidence,'reviewed_package_sha256':args.package_sha256,'independent_root_clock_record_sha256':args.root_clock_json_sha256}
    validate_phase(phase);STATE.mkdir()
    phase_pin=publish(STATE/'analysis_phase.json',phase)
    anchor={'phase_sha256':phase_pin,'utc':start.isoformat(),'monotonic_seconds':tick-(now-start).total_seconds(),'recorded_root_utc':now.isoformat(),'root_clock_record_sha256':args.root_clock_json_sha256}
    anchor_pin=publish(STATE/'local_phase_anchor.json',anchor)
    ops_raw=(HERE/'observed_operations.json').read_bytes();ops=strict(ops_raw);spec=ops['operations']['d3_observed_history_pipeline']
    inputs=dict(spec['inputs_sha256']);inputs[OUTPUT+'/controls/analysis_phase.json']=phase_pin
    cmd=[inputs[x['sha256_of_input']] if isinstance(x,dict) else x for x in spec['argv']]
    finish=start+D.timedelta(seconds=3000)
    owner_pin=digest((HERE/'supervise_observed_history_cpu_v1.py').read_bytes());bootstrap_pin=digest((HERE/'launch_observed_history_cpu_v1.py').read_bytes())
    release={'schema':'goop3d_observed_history_cpu_release_v1','issued_by':'root','status':'approved_one_bounded_stopped_analysis_invocation','operation':'d3_observed_history_pipeline','dataset':'Goop3D','host_role':'D3','hostname':HOST,'original_phase_modified':False,'automatic_retry':False,'cleanup_and_publication_seconds':15,'issued_utc':now.isoformat(),'invocation_origin_utc':start.isoformat(),'clock_sample':clock,'whole_invocation_seconds':3000,'work_stop_utc':(finish-D.timedelta(seconds=15)).isoformat(),'cleanup_deadline_utc':(finish-D.timedelta(seconds=5)).isoformat(),'publication_deadline_utc':finish.isoformat(),'hard_deadline_monotonic_ns':int((clock['host_monotonic_seconds']+3000-5)*1e9),'bootstrap_allowance_seconds':5,'operation_spec':{'path':SOURCE+'/observed_operations.json','sha256':digest(ops_raw)},'analysis_phase':{'path':OUTPUT+'/controls/analysis_phase.json','sha256':phase_pin},'owner_source':{'path':SOURCE+'/supervise_observed_history_cpu_v1.py','sha256':owner_pin},'bootstrap_source':{'path':SOURCE+'/launch_observed_history_cpu_v1.py','sha256':bootstrap_pin},'inputs_sha256':inputs,'protected_tree_entries':{},'command':cmd,'owner_output_dir':OUTPUT+'/owners/d3_observed_history_pipeline','outer_timeout':{'path':'/usr/bin/timeout','sha256':'2db30bc57746c940a0643581cd5e2671e505442bea16164143f0ea8ff4e36453','version':'timeout (GNU coreutils) 9.4','duration_rule':'floor((original_hard_deadline_ns-actual_bootstrap_monotonic_ns)/1e9)-15-5'}}
    release_pin=publish(STATE/'pipeline.cpu_release.json',release)
    (STATE/'fresh_native_observation.json').write_bytes(raw);(STATE/'prior_fate_review.json').write_bytes(prior);(STATE/'root_clock.json').write_bytes(rootraw)
    rows=[(OUTPUT+'/controls/'+name,(STATE/name).read_bytes()) for name in ('analysis_phase.json','fresh_native_observation.json','prior_fate_review.json','pipeline.cpu_release.json')]
    controls_pin=publish(STATE/'control_payload.json',payload(rows,'controls'))
    print(json.dumps({'status':'distinct_observed_phase_issued_worker_not_started','state':str(STATE),'phase_sha256':phase_pin,'anchor_sha256':anchor_pin,'release_sha256':release_pin,'control_payload_sha256':controls_pin,'original_phase_modified':False}))

if __name__=='__main__':main()
