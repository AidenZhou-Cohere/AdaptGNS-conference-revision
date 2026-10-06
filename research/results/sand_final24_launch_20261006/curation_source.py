"""Explicit immutable startup curation. Never reads evaluation products or secrets."""
from pathlib import Path
import hashlib,json,re

WORK=Path(__file__).resolve().parent
P=WORK/'deadline_research_20261005/cuda_preparation'
DEST=WORK/'sand_final24_launch_publication_candidate_transition_v1'
STATE=P/'sand_final24_released_root_v1'
OP=P/'sand_final24_operator_UNADMITTED_transition_v1'
REVIEW_PINS={
 'sand_final24_operator_independent_source_review_code_audit_v1.json':'082f521f2fbbfd30f7e63bd01267a7b8b4bbc6ad07fa1bd10c28e635808e5f6b',
 'sand_final24_actual_issuance_auditor_independent_owner_review_v1.json':'0232b0d890960ce3271b54fd3e9c0f43581c238028a4e77fc6238dce5fde59ae',
 'sand_final24_actual_issuance_independent_review_code_audit_v1.json':'0276b168cabff772e525e4f2c3b9f3786e513e0fb1bd7a69cb41a822ce583c6a',
 'sand_final24_startup_auditor_independent_owner_review_v1.json':'bb58c8b470d0edf88e42e8044acc5f65da0fc365862f806998d9a1fba4ef6ddb',
 'sand_final24_startup_independent_review_transition_v1.json':'80c080375ceb3ef9b444e18ba719a83e4cfc1119fc51896fc86c31f609dbd442',
 'sand_final24_B_amendment_path_delta_independent_owner_review_v1.json':'51b470067aa79268aca42f31c77170405c656c26c195966a4707e00ef2fbcbb4',
 'sand_final24_evaluation_candidates_independent_review_code_audit_v1.json':'2f97d62f2e7145cfc1b0607c0b9a6c583f895f90595a173cedf1b82b8ec840ec',
 'sand_final_evaluation_scoped_independent_review_code_audit_v1.json':'b3553d1fc8058373e04834856bf1cad174cce3231f4371328cbb08eb554432d5',
 'sand_final24_runtime_launch_prerequisites_independent_owner_v1.json':'3217b0e48de441b378e4122bbad39604b19b30b792c24251aa853f222d4d937f',
}
PACKAGES={
 'operator':(OP,'6898c6d6dcaf0421b6b64c37de9e4d5505e059e8cd2311a81bb203ab897ef942'),
 'references/path_candidate':(P/'sand_final24_B_amendment_paths_UNADMITTED_transition_v1','0c438762373ea714ba1f415f509c756bd58f2297c8887e8c4a230b3e066472d0'),
 'references/original_candidate':(P/'sand_final_evaluation_UNADMITTED_owner_v1','49f1f655303d5af180758d681495f4fae7335f4ed800da1543333741857029e9'),
}
FORBIDDEN_SUFFIX={'.npy','.npz','.pt','.pth','.pkl','.pickle','.pem','.key','.log','.stdout','.stderr','.pyc'}
def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def need(value,message):
 if not value:raise ValueError(message)
def source_bytes(path):
 path=Path(path);need(path.is_file() and not path.is_symlink(),'Ordinary source required')
 need(path.suffix not in FORBIDDEN_SUFFIX and not path.name.endswith('.command.json') and path.name!='ssh_config'
      and not any(x in path.parts for x in ('.ssh','.aws','.env')),'Excluded source category: '+str(path))
 return path.read_bytes()
def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
 need(not DEST.exists(),'Preserve existing candidate; no overwrite')
 selected={};inventory=[];reference_pins={};observed_manifest_counts={}
 def add(path,destination,expected=None,category='frozen_source_or_metadata'):
  path=Path(path);raw=source_bytes(path);pin=sha_bytes(raw)
  if expected is not None:need(pin==expected,'Reviewed source changed: '+str(path))
  destination=str(destination);need(destination not in selected,'Duplicate destination')
  selected[destination]=raw;inventory.append({'source_path':str(path.resolve()),'destination':destination,'sha256':pin,'category':category})
  return read(path) if path.suffix=='.json' else None
 for label,(root,pin) in PACKAGES.items():
  manifest=add(root/'manifest.json',label+'/manifest.json',pin,'frozen_package_manifest')
  observed_manifest_counts[label]=len(manifest['files_sha256'])
  for name,h in manifest['files_sha256'].items():add(root/name,label+'/'+name,h,'exact_frozen_package_member')
 need(observed_manifest_counts=={'operator':35,'references/path_candidate':35,'references/original_candidate':32},'Exact package member counts')
 review_documents={}
 for name,pin in REVIEW_PINS.items():
  review_documents[name]=add(P/name,'reviews/'+name,pin,'independent_source_or_completed_control_review')
 issue=review_documents['sand_final24_actual_issuance_independent_review_code_audit_v1.json']
 startup=review_documents['sand_final24_startup_independent_review_transition_v1.json']
 need(issue['status']=='passed_actual_final24_issuance' and startup['status']=='passed_both_original_sessions_and_native_owner_startup'
      and startup['scientific_completion'] is False and startup['all24_stage_completion_asserted'] is False,'Startup only, no scientific result admission')
 need({role:row['original_tool_session'] for role,row in startup['roles'].items()}=={'A':51595,'B':12559},'Exact original sessions')
 for document in (issue,startup):reference_pins.update(document['evidence_sha256'])
 prepared=read(OP/'prepared_inputs.json');reference_pins.update(prepared['original_local_evidence_sha256'])
 known={name:reference_pins.get(str((STATE/name).resolve())) for name in ['prepared_inputs.json','operator_binding.json','control_phase.json','control_anchor.json','control_completion.json']}
 suffixes=('evaluation_release.json','launch.json','original_session.json','owner_observed.json','staged.json','verify_stage.json','stage_release.json','observe_owner.json','verify_stage.external.json','stage_release.external.json','observe_owner.external.json')
 for role in ('A','B'):
  for suffix in suffixes:
   name=role+'.'+suffix;known[name]=reference_pins.get(str((STATE/name).resolve()))
 need(len(known)==27 and all(known.values()),'All27 completed control files must be byte-bound by accepted reviews')
 for name,pin in known.items():add(STATE/name,'completed_startup/'+name,pin,'completed_original_control_or_native_startup_receipt')
 for filename,pin in (
  ('review_actual_sand_final24_issuance_code_audit_v1.py','c2540e6bdbb202de8c8f4685661fd606ea6401cf426d89bcd82fa668469b904f'),
  ('review_sand_final24_startup_transition_v1.py','7e3343c8eb07dc7221de60d9b67c3f2978814845ab6d632dd73c0e6c721f1877')):
  add(P/filename,'auditors/'+filename,pin,'reviewed_inert_auditor_source')
 issuer_review=review_documents['sand_final24_actual_issuance_auditor_independent_owner_review_v1.json']
 start_review=review_documents['sand_final24_startup_auditor_independent_owner_review_v1.json']
 for name in ('sand_final24_issuance_audit_review_owner_v1/test_issuance_audit.py','sand_final24_startup_audit_review_owner_v1/test_startup_audit.py'):
  path=P/name
  if path.name=='test_startup_audit.py':expected=start_review['synthetic_test_sha256']
  else:expected=start_review['synthetic_fixture_dependency_sha256']
  add(path,'auditors/'+name,expected,'reviewed_synthetic_tests')
 source_closure=read(P/'sand_final24_B_amendment_paths_UNADMITTED_transition_v1/frozen_source_closure.json')
 add(Path(source_closure['profile_path']),'references/source_manifests/approved_runtime_dependency_closure_code_audit_v1.json',source_closure['profile_sha256'],'frozen_runtime_and_source_manifest')
 add(P/'sand_B_preflight_amendment_UNADMITTED_transition_v1/manifest.json','references/input_manifests/B_amendment_manifest.json','72cfe26b2d8c13ea50d24b8d85e450ef228a4bf9ef681fb18a2de525f8640e6d','frozen_input_reference_manifest')
 amend=read(P/'sand_B_preflight_amendment_UNADMITTED_transition_v1/manifest.json')
 add(P/'sand_B_preflight_amendment_UNADMITTED_transition_v1/immutable_original_transfer_bindings.json','references/input_manifests/immutable_original_transfer_bindings.json',amend['files_sha256']['immutable_original_transfer_bindings.json'],'frozen_input_reference_manifest')
 add(P/'sand_B_amendment_root_completion_admission_v1.json','references/input_manifests/B_amendment_root_completion.json',prepared['root_B_completion_sha256'],'completed_input_preparation_admission')
 # These are closed data-input census/preflight manifests, not arrays or evaluation products.
 for path,pin in prepared['original_local_evidence_sha256'].items():
  path=Path(path)
  if path.name in ('all_split_census.json','split_preflight.json'):
   add(path,'references/input_manifests/'+str(path.relative_to(P)),pin,'completed_input_manifest_no_array_contents')
 add(Path(__file__),'curation_source.py',category='curation_script_exact_bytes')
 # Inventory excluded files without opening logs, actual command captures,
 # raw tool responses, credentials, model files, arrays, or mutable outputs.
 excluded=[]
 for path in sorted(STATE.iterdir()):
  if path.name in known:continue
  excluded.append({'source_path':str(path.resolve()),'included':False,'sha256_from_accepted_review':reference_pins.get(str(path.resolve())),
   'sha256_obtained_by_this_curation':False,'reason':'Raw transport log, command capture or original tool response; completed structured receipts are retained.'})
 excluded.append({'source_path':str(P/'ssh_config'),'included':False,'sha256_from_accepted_review':'f04af81d2b926b04f7a8063df5c8535da1d8c9aa637e15cf235c261d6d0725b2','sha256_obtained_by_this_curation':False,'reason':'Credential/transport configuration excluded without reading.'})
 selected_sources={row['source_path'] for row in inventory}
 unresolved=[{'source_path':path,'sha256_from_accepted_review':pin,'included':False} for path,pin in sorted(reference_pins.items()) if path not in selected_sources]
 for name,raw in selected.items():
  need(re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9_-]{32,}',raw) is None,'Credential-shaped content in selected source: '+name)
 DEST.mkdir()
 for name,raw in selected.items():
  path=DEST/name;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(raw)
  need(sha_bytes(path.read_bytes())==sha_bytes(raw),'Copy verification failed')
 put(DEST/'source_inventory.json',{'schema':'sand_final24_publication_source_inventory_v1','selected':inventory,'frozen_package_member_counts':observed_manifest_counts,'copied_source_file_count':len(selected),'byte_identical_copies':True,'originals_modified':False})
 put(DEST/'exclusions.json',{'schema':'sand_final24_publication_exclusions_v1','enumerated_state_exclusions':excluded,'referenced_but_uncopied_evidence':unresolved,'excluded_categories':['raw stdout/stderr and logs','actual .command.json captures','raw original execution-tool responses','direct human messages/transcripts','SSH/credential configuration and secret files','mutable evaluation output trees, owner logs and eventual exits','scientific result files, checkpoint/model bytes and arrays','Python caches'],'source_command_templates_and_launch_receipts_retained_as_provenance':True,'forbidden_contents_opened_or_hashed':False})
 text='''# Sand final24 launch publication candidate

This candidate preserves the exact tested launch package and completed startup evidence. It is a local review candidate, with no commit, push, execution authority or scientific-result admission.

The operator retains all 35 frozen members and its original manifest. The original evaluation candidate and the B path-amendment candidate are preserved with their exact manifests and members. Input manifests identify the original data/preflight provenance and fixed six-model cohort by hashes; no checkpoint or array contents are included.

Completed startup receipts bind original sessions A51595 and B12559, their actual releases, one original 600-second control phase, native owner/child identities and assigned GPUs. The accepted issuance review passed 340 checks; the independent startup review passed 440. Evaluation results, eventual exits and stopped-analysis admission remain separate and are not represented here.

`source_inventory.json` identifies every copied source and SHA256. `exclusions.json` records omitted evidence references by hashes already present in accepted reviews. Raw logs, actual transport command captures, raw tool responses, direct human messages, credential configuration and mutable evaluation products are omitted. Source command definitions and structured launch receipts remain because they bind the reviewed argv.

Historical files and their absolute paths are preserved byte-for-byte. Some references intentionally point outside this candidate to omitted original evidence; this archive is not a self-contained authority to rerun the historical controls. All original files remain in place. The shared stopped-analysis operator is outside this candidate.
'''
 with (DEST/'README.md').open('x') as f:f.write(text)
 files={str(path.relative_to(DEST)):sha_bytes(path.read_bytes()) for path in sorted(DEST.rglob('*')) if path.is_file()}
 put(DEST/'manifest.json',{'schema':'sand_final24_launch_publication_candidate_v1','status':'frozen_local_candidate_pending_independent_curation_review','files_sha256':files,'operator_manifest_sha256':PACKAGES['operator'][1],'original_sessions':{'A':51595,'B':12559},'actual_startup_review_sha256':REVIEW_PINS['sand_final24_startup_independent_review_transition_v1.json'],'scientific_results_included':False,'scientific_completion_asserted':False,'execution_authority':False,'push_or_commit_performed':False,'source_originals_modified':False})
 print(json.dumps({'manifest_sha256':sha_bytes((DEST/'manifest.json').read_bytes()),'payload_count':len(files),'copied_source_file_count':len(selected),'destination':str(DEST)}))

if __name__=='__main__':main()
