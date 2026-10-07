"""Pure reconstruction of the exact owner input, evidence and native contract."""
from observed_common import need,stamp,REMOTE,SOURCE,OUTPUT,HOST

PYTHON=REMOTE+'/.venv/bin/python'
PYTHON_REAL='/usr/bin/python3.12'
PYTHON_SHA='6242e0e8650d7dbdebbc25e08bf4c9359ddaf65f54fcae0e57fa99395fa5357a'
VENV_SHA='b7d360e62970717794ce0ab4f1e1398a49a8a73a138d34ca920b1eb251fe2f3b'
TIMEOUT_SHA='2db30bc57746c940a0643581cd5e2671e505442bea16164143f0ea8ff4e36453'
PRODUCTS=('audit.json','summary.json','arithmetic_check.json','pipeline_receipt.json')
EVIDENCE=('owner_started.json','child_registered.json','child.stdout','child.stderr')

def merge(*maps):
    out={}
    for mapping in maps:
        for path,pin in mapping.items():
            need(path not in out or out[path]==pin,'conflicting owner input binding')
            out[path]=pin
    return out

def native_key(identity):
    need(type(identity)is dict,'native identity required')
    for key in ('pid','ppid','pgid','sid'):
        need(type(identity[key])is int and identity[key]>0,'positive native identity')
    need(type(identity['start_id'])is str and identity['start_id'].isdigit(),'native start identity')
    need(type(identity['argv'])is list and all(type(v)is str for v in identity['argv']),'exact native argv')
    return {k:identity[k] for k in ('pid','ppid','pgid','sid','start_id','argv','executable')}

def expected_inputs(release,phase,phase_pin,release_pin,spec,source_pins):
    need(release['schema']=='goop3d_observed_history_cpu_release_v1' and release['operation']=='d3_observed_history_pipeline' and release['dataset']=='Goop3D' and release['host_role']=='D3' and release['hostname']==HOST,'exact scoped owner release')
    bindings={
        'operation_spec':{'path':SOURCE+'/observed_operations.json','sha256':source_pins['observed_operations.json']},
        'analysis_phase':{'path':OUTPUT+'/controls/analysis_phase.json','sha256':phase_pin},
        'owner_source':{'path':SOURCE+'/supervise_observed_history_cpu_v1.py','sha256':source_pins['supervise_observed_history_cpu_v1.py']},
        'bootstrap_source':{'path':SOURCE+'/launch_observed_history_cpu_v1.py','sha256':source_pins['launch_observed_history_cpu_v1.py']}}
    need(all(release[k]==v for k,v in bindings.items()),'exact local package/release bindings')
    controls={path:(phase_pin if pin is None else pin) for path,pin in spec['inputs_sha256'].items()}
    need([path for path,pin in spec['inputs_sha256'].items() if pin is None]==[OUTPUT+'/controls/analysis_phase.json'],'only original phase is a late input binding')
    need(release['inputs_sha256']==controls and release['protected_tree_entries']=={} and spec['protected_tree_roots']==[],'complete fixed input set and no implicit tree inputs')
    command=[controls[x['sha256_of_input']] if isinstance(x,dict) else x for x in spec['argv']]
    need(release['command']==command and release['clock_sample']==phase['clock_sample'],'exact command and clock')
    need(release['outer_timeout']=={'path':'/usr/bin/timeout','sha256':TIMEOUT_SHA,'version':'timeout (GNU coreutils) 9.4','duration_rule':'floor((original_hard_deadline_ns-actual_bootstrap_monotonic_ns)/1e9)-15-5'},'exact native timeout binding')
    need(release['owner_output_dir']==OUTPUT+'/owners/d3_observed_history_pipeline' and spec['outputs']==[OUTPUT+'/'+n for n in PRODUCTS],'exact output and owner paths')
    return merge({OUTPUT+'/controls/pipeline.cpu_release.json':release_pin},controls,
        {PYTHON:PYTHON_SHA,PYTHON_REAL:PYTHON_SHA,REMOTE+'/.venv/pyvenv.cfg':VENV_SHA,'/usr/bin/timeout':TIMEOUT_SHA},
        {v['path']:v['sha256'] for v in bindings.values()},
        {v['path']:v['sha256'] for v in phase['operational_evidence'].values()})

def validate_owner(data,files,release,phase,phase_pin,release_pin,spec,source_pins,transport):
    owner=data['owner_terminal.json'];started=data['owner_started.json'];registered=data['child_registered.json'];child=owner['child']
    need(owner['schema']=='adaptgns_stopped_analysis_cpu_terminal_v1' and owner['status']=='complete' and owner['failure'] is None and owner['release_sha256']==release_pin and owner['scientific_admission'] is False and owner['root_original_tool_exit_and_timeout_owner_child_native_closure_required'] is True,'complete exact owner terminal')
    expected=expected_inputs(release,phase,phase_pin,release_pin,spec,source_pins)
    need(owner['input_sha256']==expected,'complete exact owner input SHA map')
    need(owner['evidence_sha256']=={release['owner_output_dir']+'/'+n:files[n] for n in EVIDENCE},'complete exact owner evidence SHA map')
    need(owner['output_sha256']=={OUTPUT+'/'+n:files[n] for n in PRODUCTS},'complete exact four-product SHA map')
    need(type(child['exit_code'])is int and child['exit_code']==0 and child['reaped'] is True and child['signals']==[] and child['cleanup_errors']==[] and owner['child_native_absent'] is True and child['command']==release['command'],'original worker cleanly completed')
    identities={'outer':native_key(owner['outer_timeout_identity']),'owner':native_key(owner['owner_identity']),'worker':native_key(child['identity'])}
    need(started['owner_identity']==owner['owner_identity'] and started['outer_timeout_identity']==owner['outer_timeout_identity'] and registered['identity']==child['identity'] and registered['owner_identity']==owner['owner_identity'] and registered['pid']==child['pid']==identities['worker']['pid'],'exact original registered native identity chain')
    need(identities['owner']['ppid']==identities['outer']['pid'] and identities['worker']['ppid']==identities['owner']['pid'] and len({x['pgid'] for x in identities.values()})==1 and len({x['sid'] for x in identities.values()})==1,'exact outer/owner/worker ancestry')
    need(started['release_sha256']==release_pin and started['scientific_admission'] is False and started['original_clock_sample']==release['clock_sample'],'original owner release and clock')
    guard=owner['native_hard_guard'];timing=owner['native_outer_timing'];hard=release['hard_deadline_monotonic_ns']
    need(started['native_hard_guard']==guard and started['native_outer_timing']==timing and guard['absolute_deadline_ns']==hard and guard['clock']=='CLOCK_MONOTONIC' and guard['signal']=='SIGKILL' and guard['deleted_before_exit'] is False,'unchanged original native guard')
    need(hard==int((phase['clock_sample']['host_monotonic_seconds']+2995)*1e9) and timing['original_absolute_hard_deadline_ns']==hard and timing['owner_guard_armed_monotonic_ns']==guard['armed_monotonic_ns'] and 0<=guard['armed_monotonic_ns']-timing['computed_monotonic_ns']<=5*10**9,'fixed absolute owner guard and bootstrap allowance')
    seconds=timing['term_seconds'];computed=timing['computed_monotonic_ns']
    need(type(seconds)is int and seconds>0 and type(computed)is int and seconds==(hard-computed)//10**9-20 and timing['earliest_relative_kill_monotonic_ns']==computed+(seconds+15)*10**9 and timing['bootstrap_allowance_seconds']==5,'original bounded native outer timing')
    bounds=timing['bootstrap_process_start'];need(bounds['lower_ns']<=computed<=guard['armed_monotonic_ns'] and bounds['upper_ns']<=guard['armed_monotonic_ns'],'original bootstrap start bounds')
    outer_argv=['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(seconds)+'s',PYTHON,'-I','-S','-B',release['owner_source']['path'],'--execute','--hard-deadline-monotonic-ns',str(hard),'--outer-term-seconds',str(seconds),'--outer-computed-monotonic-ns',str(computed),'--release',OUTPUT+'/controls/pipeline.cpu_release.json','--release-sha256',release_pin]
    need(identities['outer']['argv']==outer_argv and identities['owner']['argv']==outer_argv[4:] and identities['worker']['argv']==release['command'] and identities['outer']['executable']=='/usr/bin/timeout' and identities['owner']['executable']==identities['worker']['executable']==PYTHON_REAL,'exact original native timeout/owner/worker argv and executables')
    need(phase['clock_sample']['host_monotonic_seconds']<=owner['publication_monotonic']<hard/1e9,'publication inside original monotonic guard')
    need(stamp(phase['started_utc'])<=stamp(owner['publication_utc'])<stamp(release['publication_deadline_utc'])<=stamp(phase['stop_utc']) and stamp(owner['publication_utc'])<=stamp(transport['finished_utc']),'original owner publication before transport exit and phase stop')
    return identities
