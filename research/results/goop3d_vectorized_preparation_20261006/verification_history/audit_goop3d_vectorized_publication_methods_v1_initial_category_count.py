"""Independent package audit; reads only text and original source archives."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[3]
PREP=Path(__file__).resolve().parent
P=ROOT/'outputs/AdaptGNS/research/results/goop3d_vectorized_preparation_20261006'
checks=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(v,label):
 global checks
 checks+=1
 if not v:raise AssertionError(label)
pins={'README.md':'9dec6f82cbf2dd46a07806b850cbbff74cb53d7716b266441ea214f8aa42f2e5',
'verify_publication.py':'a02042184028b48dd94a07dc649d94255790c4241c980d9016c9b52f3b6ff1cb',
'manifest.json':'890896aa99a19b27872d5610588cde2aaa2165d0bc078f0421b7a60526f0235d',
'provenance.json':'2cbd46b841566d6d83bb0dcc3b9143ed62878fd41c17aca65a228d3a693ef574'}
for name,d in pins.items():check(sha(P/name)==d,'Reviewed publication pin '+name)
m=read(P/'manifest.json');prov=read(P/'provenance.json');omitted=read(P/'omissions.json');catalogs=read(P/'archive_catalogs.json')
files=m['files'];check(len(files)==274,'Expected pre-review manifest274')
actual={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='manifest.json'}
# The package includes predecessor manifest fixture names only under explicit paths.
actual={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p!=P/'manifest.json'}
check(actual==set(files),'Exact manifest coverage')
for name,row in files.items():
 p=P/name;check(not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Manifest byte identity '+name)
for name,row in prov.items():
 source=ROOT/row['source_workspace_path'];check(source.is_file() and source.stat().st_size==row['bytes'] and sha(source)==row['sha256']==sha(P/name),'Independent original byte comparison '+name)
check(len(prov)==264,'264 exact original copies')
closure=[(n,r) for n,r in prov.items() if r['category']=='predecessor_source_closure']
check(len(closure)==110,'Complete110-file support closure')
predecessors=read(P/'predecessors.json');predecessor_checks=0
for family,entries in predecessors.items():
 for name,row in entries.items():
  original=P.parent/family/name;check(sha(original)==row['sha256'] and original.stat().st_size==row['bytes'],'Predecessor record unchanged');predecessor_checks+=1
check(predecessor_checks==6,'Six predecessor records')
regular=directories=0;archive_blobs={}
for name,cat in catalogs.items():
 with tarfile.open(P/name) as archive:
  members=archive.getmembers();check(len(members)==len(cat['members']),'No duplicate/missing archive member')
  for member in members:
   descriptor=cat['members'][member.name];check(not member.name.startswith('/') and '..' not in Path(member.name).parts,'Relative archive member')
   check(member.isfile() or member.isdir(),'No link or special archive member')
   if member.isdir():check(member.size==0 and descriptor['type']=='directory','Directory entry');directories+=1;continue
   data=archive.extractfile(member).read();data.decode('utf-8')
   check(member.size==descriptor['bytes'] and hashlib.sha256(data).hexdigest()==descriptor['sha256'] and descriptor['type']=='file','Archive original byte identity')
   check(Path(member.name).suffix.lower() not in {'.pt','.pth','.ckpt','.npy','.npz','.tfrecord','.pkl','.pickle','.pyc'},'No omitted model/data payload');regular+=1
   archive_blobs[(name,member.name)]=data
check(len(catalogs)==6 and regular==139 and directories==10,'Six archives139files10directories')
archive='archives/goop3d_vectorized_capacity_v1_scalar_snapshot.tar.gz';prefix='goop3d_vectorized_capacity_20261006_v1/'
cap={name[len(prefix):]:data for (arc,name),data in archive_blobs.items() if arc==archive};summary=json.loads(cap['summary.json'])
check(len(cap)==74 and len(omitted['capacity_checkpoints'])==12 and len(omitted['numerical_payloads'])==32,'Scalar/payload omission counts')
check(hashlib.sha256(cap['summary.json']).hexdigest()=='39f048a7257428b0ffe342e52ed2f956e90cd5dec4463835b40a9d75855a47e1','Exact actual capacity summary')
full=read(P/'records/goop3d_capacity_final_artifact_inventory_v1.json')['files'];check(set(cap).isdisjoint(omitted['capacity_checkpoints']) and set(cap)|set(omitted['capacity_checkpoints'])==set(full),'Complete86-artifact capacity coverage')
for name,row in omitted['capacity_checkpoints'].items():check(row==full[name],'Capacity omitted byte ref')
check(omitted['numerical_payloads']==read(P/'records/goop3d_numerical_remote_artifact_audit_v1.json')['artifacts'],'32 omitted numerical byte refs')
review=read(P/'records/goop3d_vectorized_capacity_v1_independent_review_methods.json')
check(review['checks_passed']==47096 and review['q4_q2']==summary['q4_q2'] and review['r4_r2']==summary['r4_r2'],'Actual audit/q/r correspondence')
check(all(j['lr_cross_platform_max_float64_ulp']==1 and j['lr_cross_platform_nonexact_rows']==2 for j in review['jobs']),'Explicit1ULP/two-row LR limitation')
prep=P/'workspace/work/deadline_research_20261005/cuda_preparation'
required=['audit_goop3d_vectorized_capacity_report_methods_v1.py','audit_goop3d_vectorized_capacity_report_methods_v1_initial_crossplatform_lr.py',
'audit_goop3d_vectorized_numerical_report_methods_v1.py','supervise_goop3d_science_v1.py','test_supervise_goop3d_science_v1.py',
'goop3d_valid_timing_split_admission.template.initial_missing_source_sha.json']
for n in required:check((prep/n).is_file(),'Reviewed/failed source retained '+n)
for n in ['goop3d_capacity_methods_scalar_audit_attempt1.log','goop3d_capacity_methods_scalar_audit_attempt2.log']+[f'goop3d_science_synthetic_tests_attempt{i}.log' for i in range(1,6)]:check((P/'records'/n).is_file(),'Failed/passing history retained '+n)
check((P/'records/third_vm_pre_capacity_inventory_v1.json').read_bytes()==b'','InvalidJSON failed artifact preserved')
failure=read(P/'records/goop3d_capacity_preflight_failure_record_v1.json');check(failure['output_is_valid_json'] is False and failure['capacity_workers_launched'] is False,'Failure scope retained')
for n in ('goop3d_scientific_training_admission.template.json','goop3d_scientific_release.template.json','goop3d_deadline_worksheet_release.template.json'):
 obj=read(prep/n);check(obj['status']=='not_admitted' and obj['scientific_training_admitted'] is False,'Inert public admission')
plan=read(prep/'goop3d_scientific_endpoint_plan.template.json');planning=read(prep/'goop3d_deadline_planning_config.template.json')
check(plan['endpoint_updates'] is None and plan['status']=='not_selected' and planning['planning_start_utc'] is None and planning['selected_endpoint'] is None,'Unset endpoint and decision time')
check(not any(Path(n).name in {'goop3d_validation_timing_v1.stdout.log','goop3d_validation_timing_v1.stderr.log','goop3d_timing_first_identity_v1.json'} for n in files),'Live timing/counter evidence excluded')
check(not any('/goop3d_validation_timing_20261006_v1/' in n for n in files),'No timing result directory copied')
contracts=P/'launch_declarations/goop3d_timing_contracts_local_v1';timing=read(contracts/'supervisor_release.json')
check(timing['scientific_training_admitted'] is False and timing['scientific_endpoint_selected'] is False and timing['test_access_allowed'] is False,'Fixed timing declaration scope')
tested=read(P/'verification_history/tested_manifest_v1.json')['files'];tests=read(P/'test_inventory.json');run=read(P/'packaging_verification.json')['actual_replay']['cpu_tests']
check(tests['total']==run['passed_count']==260 and run['status']=='passed' and len(tests['files'])==len(run['files'])==10,'Reported ten-process260-test replay')
for name,row in files.items():
 if name.startswith('workspace/'):
  check(name in tested and row==tested[name],'Research source/test bytes unchanged since isolated replay')
readme=(P/'README.md').read_text();check('Steady wall seconds per update, q' in readme and 'Steady guarded seconds' not in readme,'q wall-time label corrected')
check('no selected scientific endpoint, admitted scientific' in readme and 'no forecast has been run from these templates' in readme,'Scope prose')
for name,d in pins.items():check(sha(P/name)==d,'Terminal publication pin '+name)
out={'schema':'adaptgns_goop3d_vectorized_publication_independent_review_methods_v1','reviewer':'conference_critique_methods',
'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed_after_two_publication_corrections','checks_passed':checks,
'source_sha256':sha(__file__),'reviewed_package':str(P),'reviewed_package_pins':pins,'manifest_file_count_before_review_inclusion':274,
'original_files_independently_compared':264,'predecessor_records':6,'support_closure_files':110,'archives':6,'regular_archive_members':139,'directory_archive_members':10,
'packaged_tests':{'reported_passed':260,'separate_test_processes':10,'research_source_and_test_bytes_unchanged_since_replay':True,'rerun_by_this_review':False},
'scope':'Independent byte/original/archive/omission/source-closure and scientific-boundary review only. No official numeric arrays, model deserialization, remote commands, GPU jobs or actual efficacy calculation.',
'resolved_findings':[{'finding':'README called q guarded seconds.','resolution':'Changed to steady wall seconds; exact capacity numbers unchanged.'},
{'finding':'Statistics review found frozen timing first-identity snapshot included progress counters.','resolution':'Public copy/current provenance removed; original preserved and omission recorded; prior tested manifest retained as historical attempt metadata.'}],
'remaining_blocking_findings':[],
'confirmed':['Capacity6x512 and numerical26optimizer-call results remain implementation/capacity evidence, not scientific endpoints or simulation efficacy.','Capacity archive contains all74 scalar/source artifacts and exactly accounts for12 omitted checkpoints; numerical32payload omissions match root byte receipts.','Actual capacity q/r and47096-check audit match original summary; first strict-LR audit and repaired1ULP cross-platform formula limit are disclosed without relaxing exact paired-arm learning rates.','Zero-byte invalidJSON inventory and labelled failure transcription remain distinct from corrected inventory.','All110 predecessor support files and six predecessor index records retain original bytes; current science/trainer/evaluator pins unchanged.','Scientific/planning templates remain inert; timing includes fixed declarations only, with no live outcomes or completed timing/forecast claim.','Historical audit scripts are parsed, not imported; their original-layout and exclusive-output replay restrictions are disclosed.'],
'publication_inclusion_rule':'Adding this review and its byte-check log requires manifest refresh; all already-reviewed files must retain the pinned content. Review records intentionally describe the pre-inclusion manifest to avoid self-hash recursion.',
'limits':['No independent reread of omitted remote model/checkpoint bytes in this review; their separately audited hashes/sizes are preserved.','CPU packaged tests do not establish actual CUDA/scientific outcome or endpoint feasibility.','No conference-readiness claim is established.']}
path=PREP/'goop3d_vectorized_publication_independent_review_methods_v1.json'
with path.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'checks_passed':checks,'review_sha256':sha(path),'review':str(path)},indent=2))
