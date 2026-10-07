"""Pure validation and combination of local prepared sources and issued controls."""
import base64
from observed_common import need,strict,digest,SOURCE,OUTPUT
from stage_payload import SOURCE_NAMES,CONTROL_NAMES

def source_payload(raw,bindings):
    data=strict(raw);need(data['schema']=='goop3d_observed_history_staging_payload_v1' and data['kind']=='sources','local source preparation payload only')
    rows=data['files'];expected={SOURCE+'/'+name:pin for name,pin in bindings['files_sha256'].items()}
    need(set(bindings['files_sha256'])==SOURCE_NAMES and len(rows)==len(expected) and {row['path']:row['sha256'] for row in rows}==expected,'all exact13 local prepared sources')
    for row in rows:
        raw=base64.b64decode(row['base64'],validate=True);need(len(raw)<=1<<20 and digest(raw)==row['sha256'],'exact bounded source payload bytes')
    return data

def combined_payload(source_raw,control_raw,bindings,phase_pin,release_pin):
    sources=source_payload(source_raw,bindings);controls=strict(control_raw)
    need(controls['schema']==sources['schema'] and controls['kind']=='controls','issued control preparation payload')
    rows=controls['files'];need(len(rows)==len(CONTROL_NAMES) and {row['path'] for row in rows}=={OUTPUT+'/controls/'+name for name in CONTROL_NAMES},'all exact4 issued controls')
    expected={OUTPUT+'/controls/analysis_phase.json':phase_pin,OUTPUT+'/controls/pipeline.cpu_release.json':release_pin}
    for row in rows:
        raw=base64.b64decode(row['base64'],validate=True);need(len(raw)<=1<<20 and digest(raw)==row['sha256'],'exact bounded control payload bytes')
        if row['path'] in expected:need(row['sha256']==expected[row['path']],'exact current phase/release controls')
    return {'schema':sources['schema'],'kind':'combined','files':sources['files']+rows}
