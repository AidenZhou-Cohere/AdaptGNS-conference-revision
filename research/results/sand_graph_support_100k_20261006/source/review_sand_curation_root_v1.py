from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re
W=Path.cwd();P=W/'work/sand_completed_analysis_publication_candidate_UNADMITTED_20261006_v2';PREP=W/'work/deadline_research_20261005/cuda_preparation'; checks=[]
def need(v,s):
 if not v: raise AssertionError(s)
 checks.append(s)
def h(raw):return hashlib.sha256(raw).hexdigest()
def doc(p):return json.loads(p.read_bytes())
manifest=P/'candidate_manifest.json';need(h(manifest.read_bytes())=='16b9f66a04230fbb83b410cc007b1bce14b9bbc05ce93ef9b98b8780fd749029','exact candidate manifest');m=doc(manifest)
files={x.relative_to(P).as_posix():x for x in P.rglob('*') if x.is_file()};need(set(files)==set(m['files'])|set(m['manifest_exclusions']),'complete file set, including explicit seal exclusions');need(not any(x.is_symlink() for x in P.rglob('*')),'no symlinks')
patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9_.-]{16,}',rb'gh[pousr]_[A-Za-z0-9]{25,}',rb'sk-[A-Za-z0-9_-]{30,}'];matches=[]
for name,p in files.items():
 raw=p.read_bytes()
 if name in m['files']:need(len(raw)==m['files'][name]['bytes'] and h(raw)==m['files'][name]['sha256'],'manifest bytes '+name)
 need(len(raw)<20_000_000,'compact per-file size '+name)
 need(p.suffix.lower() not in ('.npz','.pt','.pth','.ckpt') and p.name not in ('ssh_config','credentials','id_rsa'),'no excluded large model or credential file '+name)
 for i,rx in enumerate(patterns):
  if re.search(rx,raw):matches.append({'file':name,'pattern':i})
need(not matches,'no credential signatures in public payload')
inv=doc(P/'source_inventory.json');need(len(inv)==133,'all133 original copies')
for name,row in inv.items():
 p=files[name];src=W/row['source'];need(row['transformation']=='none' and src.is_file() and not src.is_symlink(),'ordinary unchanged source '+name)
 need(p.stat().st_size==src.stat().st_size==row['bytes'] and h(p.read_bytes())==h(src.read_bytes())==row['sha256'],'exact original copy '+name)
old=W/'work/sand_completed_analysis_publication_candidate_UNADMITTED_20261006_v1';need(h((old/'candidate_manifest.json').read_bytes())=='d547f432548872555b4b4ebbff68bd4539f03ee8939139bd5dd6dd8d33f1e8a8','original predecessor manifest preserved')
om=doc(old/'candidate_manifest.json')
for name,row in om['files'].items():
 original=old/name;need(h(original.read_bytes())==row['sha256'],'predecessor stable '+name)
 target=P/name if name.startswith('payload/') else P/'history/source_provenance_candidate_v1'/name
 need(target.read_bytes()==original.read_bytes(),'predecessor retained '+name)
refs=doc(P/'large_local_product_references.json');need(set(refs)=={'sand_collect_A','sand_collect_B','sand_saved_A','sand_saved_B'},'four large originals remain references')
for op,row in refs.items():
 source=W/row['source'];need(source.stat().st_size==row['bytes'],'large product size reference '+op)
 need(not any(Path(x['source']).resolve()==source.resolve() for x in inv.values()),'large product excluded from copies '+op)
 rr=doc(W/row['root_review_source']);need(rr['product_sha256']==row['sha256'] and h((W/row['root_review_source']).read_bytes())==row['root_review_sha256'] and h((W/row['independent_review_source']).read_bytes())==row['independent_review_sha256'],'accepted large product identity reference '+op)
admission=doc(P/'payload/workspace/work/deadline_research_20261005/cuda_preparation/sand_complete_cohort_interpretation_admission_root_v2.json');need(m['interpretation_admission_sha256']=='bab36c91925e8368417206f258b8829f0cda78ec9e423b551b769d48332eca64','root interpretation admission pin')
for op,row in doc(P/'final_product_pins.json').items():
 need(op in ('sand_summarize','sand_paired') and row['sha256']==admission['products'][op]['sha256']==h((P/row['payload']).read_bytes()),'exact admitted compact product '+op)
need(admission['completed_mixed_stage_cells']==3648 and admission['completed_autonomous_outcomes']==1080 and admission['required_models']==6 and admission['horizon']==314,'complete fixed scope')
need(admission['policies']==['base','dense','random25','speed25','laggedrisk25','relative-velocity-RMS25'],'all exact policies')
failures=doc(P/'failure_inventory.json');need(len(failures['entries'])==7,'all seven current failure/history classes retained');need(m['retrieval_proxy_cleanup_pending'] is True and m['statistics_recomputed'] is False,'cleanup pending, no statistics rerun')
need(sum(row['bytes'] for row in m['files'].values())==m['bytes_excluding_manifest_and_verification'],'total manifest bytes')
result={'schema':'sand_completed_curation_independent_root_review_v1','status':'passed_exact_compact_publication_scope_with_proxy_addendum_pending','reviewer':'root (independent of curation author)','reviewed_utc':datetime.now(timezone.utc).isoformat(),'candidate_manifest_sha256':h(manifest.read_bytes()),'check_count':len(checks),'file_count':len(files),'exact_original_copies':len(inv),'reference_count':len(doc(P/'referenced_evidence_inventory.json')['files']),'no_credential_signatures':True,'all_source_and_failure_versions_retained_or_exactly_referenced':True,'large_products_read_or_rehashed':False,'statistical_or_scientific_programs_rerun':False,'candidate_files_modified':False,'scientific_admission_sha256':m['interpretation_admission_sha256'],'proxy_cleanup_addendum_pending':True,'limitations':['Curation checks preserve accepted numerical evidence; they do not rerun science.','Reference-only products retain their prior accepted hashes and current size, without reopening their bytes.','Original proxy62558 exit and local closure remain a separate required follow-up.'],'checks':checks}
out=PREP/'sand_completed_curation_independent_root_review_v1.json'
with out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'path':str(out),'sha256':h(out.read_bytes()),'checks':len(checks),'files':len(files),'exact_copies':len(inv)}))
