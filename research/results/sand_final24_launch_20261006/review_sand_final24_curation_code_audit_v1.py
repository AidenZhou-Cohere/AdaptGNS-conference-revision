"""Review only the completed startup publication candidate; no live result reads."""
from pathlib import Path
import datetime,hashlib,json,re
P=Path(__file__).resolve().parent
W=P.parent.parent
D=W/'sand_final24_launch_publication_candidate_transition_v1'
PIN='b382d5e03e8f8408c4fe81a86d7881a63c1712fa8800695519c6fd03cbe943df'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
checks=[]
def check(value,label):
    if not value:raise ValueError(label)
    checks.append(label)
def run():
    check(sha(D/'manifest.json')==PIN,'exact original candidate manifest')
    manifest=read(D/'manifest.json');inventory=read(D/'source_inventory.json');exclusions=read(D/'exclusions.json')
    members=manifest['files_sha256']
    actual={str(p.relative_to(D)) for p in D.rglob('*') if p.is_file()}
    check(actual==set(members)|{'manifest.json'} and len(members)==158,'exact158 payloads and manifest, no extra files')
    for rel,pin in members.items():
        path=D/rel;check(path.resolve()==path and not path.is_symlink() and sha(path)==pin,'candidate payload bytes: '+rel)
    selected=inventory['selected'];check(len(selected)==155 and inventory['copied_source_file_count']==155,'155 exact copied sources')
    check(len({r['destination'] for r in selected})==155,'unique copied destinations')
    for row in selected:
        src=Path(row['source_path']);dst=D/row['destination']
        check(src.is_file() and not src.is_symlink() and src.read_bytes()==dst.read_bytes() and sha(dst)==row['sha256'],'original preserved byte-for-byte: '+row['destination'])
    check(set(members)-{r['destination'] for r in selected}=={'source_inventory.json','exclusions.json','README.md'},'only three new curation documents')
    packages={'operator':('6898c6d6dcaf0421b6b64c37de9e4d5505e059e8cd2311a81bb203ab897ef942',35),'references/path_candidate':('0c438762373ea714ba1f415f509c756bd58f2297c8887e8c4a230b3e066472d0',35),'references/original_candidate':('49f1f655303d5af180758d681495f4fae7335f4ed800da1543333741857029e9',32)}
    for label,(pin,count) in packages.items():
        check(sha(D/label/'manifest.json')==pin,'unchanged package manifest: '+label);inner=read(D/label/'manifest.json')
        check(len(inner['files_sha256'])==count,'complete original member count: '+label)
        check(all(members[label+'/'+name]==h for name,h in inner['files_sha256'].items()),'all original members retained: '+label)
    names={'prepared_inputs.json','operator_binding.json','control_phase.json','control_anchor.json','control_completion.json'}
    for role in ('A','B'):
        names|={role+'.'+suffix for suffix in ('evaluation_release.json','launch.json','original_session.json','owner_observed.json','staged.json','verify_stage.json','stage_release.json','observe_owner.json','verify_stage.external.json','stage_release.external.json','observe_owner.external.json')}
    check(len(names)==27 and {p.name for p in (D/'completed_startup').iterdir()}==names,'exact27 closed control/startup whitelist')
    issue=read(D/'reviews/sand_final24_actual_issuance_independent_review_code_audit_v1.json');startup=read(D/'reviews/sand_final24_startup_independent_review_transition_v1.json')
    check(sha(D/'reviews/sand_final24_actual_issuance_independent_review_code_audit_v1.json')=='0276b168cabff772e525e4f2c3b9f3786e513e0fb1bd7a69cb41a822ce583c6a','original accepted340check issuance review')
    check(sha(D/'reviews/sand_final24_startup_independent_review_transition_v1.json')=='80c080375ceb3ef9b444e18ba719a83e4cfc1119fc51896fc86c31f609dbd442','original accepted440check native startup review')
    check(len(issue['checks'])==340 and len(startup['checks'])==440,'README review counts agree with closed receipts')
    check(issue['status']=='passed_actual_final24_issuance' and startup['status']=='passed_both_original_sessions_and_native_owner_startup','completed accepted review statuses')
    check(startup['scientific_completion'] is False and startup['all24_stage_completion_asserted'] is False,'startup has no numerical completion claim')
    check({role:value['original_tool_session'] for role,value in startup['roles'].items()}=={'A':51595,'B':12559},'both exact original tool sessions')
    evidence={**issue['evidence_sha256'],**startup['evidence_sha256']}
    prepared=read(D/'operator/prepared_inputs.json');evidence.update(prepared['original_local_evidence_sha256'])
    for row in selected:
        if row['destination'].startswith('completed_startup/'):
            check(evidence[row['source_path']]==row['sha256'],'startup copy bound by prior accepted evidence: '+row['destination'])
    phase=read(D/'completed_startup/control_phase.json');dt=datetime.datetime.fromisoformat
    check((dt(phase['stop_utc'])-dt(phase['started_utc'])).total_seconds()==phase['control_seconds']==600 and phase['clock_restarted'] is False,'one original600second control phase')
    check(phase['analysis_seconds']==3600 and phase['evaluation_seconds']==11760,'unmodified original analysis/stream allocations')
    check(read(D/'completed_startup/control_completion.json')['scientific_completion'] is False,'control completion not scientific completion')
    forbidden={'.npy','.npz','.pt','.pth','.pkl','.pickle','.pem','.key','.log','.stdout','.stderr','.pyc'}
    for rel in members:
        p=Path(rel);check(p.suffix not in forbidden and not p.name.endswith('.command.json') and p.name!='ssh_config' and not set(p.parts)&{'.ssh','.aws','.env','__pycache__'},'excluded sensitive/raw category absent: '+rel)
        raw=(D/rel).read_bytes();check(re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9_-]{32,}',raw) is None,'selected payload has no credential-shaped literal: '+rel)
    check(exclusions['forbidden_contents_opened_or_hashed'] is False,'curator declares exclusion without reading omitted bytes')
    check(all(row['included'] is False and row['sha256_obtained_by_this_curation'] is False for row in exclusions['enumerated_state_exclusions']),'omitted artifacts recorded without rereading mutable transport')
    check(all(row['source_path'] not in {r['source_path'] for r in selected} for row in exclusions['enumerated_state_exclusions']),'excluded sources not silently copied')
    for flag in ('scientific_results_included','scientific_completion_asserted','execution_authority','push_or_commit_performed','source_originals_modified'):
        check(manifest[flag] is False,'publication candidate scope: '+flag)
    readme=(D/'README.md').read_text()
    check('eventual exits and stopped-analysis admission remain separate' in readme and 'not a self-contained authority to rerun' in readme,'historical provenance and completion limitations explicit')
    # Review time-of-check binding only; this is not a repeat scientific audit.
    check(sha(D/'manifest.json')==PIN and all(sha(D/name)==pin for name,pin in members.items()),'final entire candidate byte reverification')
    receipt={'schema':'sand_final24_curation_independent_review_v1','status':'passed_completed_startup_curation_only','reviewer':'code_audit','curator':'transition_review','candidate_path':str(D),'candidate_manifest_sha256':PIN,'payload_count':158,'source_copy_count':155,'checks':checks,'check_count':len(checks),'review_source_sha256':sha(__file__),'scientific_products_or_arrays_or_models_read':False,'completed_scientific_audit_repeated':False,'new_control_clock_or_execution_or_publication':False}
    out=P/'sand_final24_curation_independent_review_code_audit_v1.json'
    with out.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':receipt['status'],'checks':len(checks),'receipt_sha256':sha(out)}))
if __name__=='__main__':run()
