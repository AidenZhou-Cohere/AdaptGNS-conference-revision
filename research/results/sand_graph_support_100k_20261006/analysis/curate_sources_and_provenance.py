"""Copy only frozen source/provenance; reference scientific products without reading them."""
import hashlib
import json
from pathlib import Path
import re
import shutil

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
PREP=ROOT/'work/deadline_research_20261005/cuda_preparation'
STATE=PREP/'sand_stopped_analysis_released_root_v1'
PACKAGES={
 'sand_stopped_analysis_operator_UNADMITTED_code_audit_v1':'914e6e2df11836079f721a675bdbfcd9673b6bada3074fe766c52d085a37640b',
 'sand_native_closure_observer_UNADMITTED_code_audit_v1':'2b1c7fdbc547baf8c36e39508ec84d0b1189a83856b2e62fb85a9638cca1364e',
 'sand_original_phase_route_UNADMITTED_statistics_v1':'292cc31d54e822d6eb1a843da3ce55427f9f89201c1fb9a2dd7293d0660dfd80',
 'sand_scalar_recovery_route_UNADMITTED_statistics_v1':'d580a222cae4f1844861c38b47212e27e77c198fc8466e65dc05c528d54ca92b',
 'sand_scalar_recovery_UNADMITTED_code_audit_v1':'71c50f558082d127757cc84dff97f7451c7c5e73ff4b5ef48ff41be1af851a79',
 'sand_summary_capture_recovery_UNADMITTED_code_audit_v1':'2b8b0b9a8a0a025533d6291f401f1e99fa6be9cb70b8e7b0bb27029f00fc429c',
}
REVIEWS=(
 'sand_authoritative_evaluation_closures_independent_review_code_audit_v1.json',
 'sand_original_transport_cleanup_independent_review_code_audit_v1.json',
 'sand_actual_phase_runtime_independent_review_code_audit_v1.json',
 'sand_collect_A_completed_independent_review_code_audit_v1.json',
 'sand_collect_B_completed_independent_review_code_audit_v1.json',
 'sand_saved_A_completed_independent_review_code_audit_v1.json',
 'sand_saved_B_completed_independent_review_code_audit_v1.json',
 'sand_scalar_recovery_route_independent_review_code_audit_v1.json',
 'sand_scalar_recovery_independent_transition_v1.json',
 'sand_recovery_fate_independent_review_code_audit_v1.json',
 'sand_recovered_transfer_independent_review_code_audit_v1.json',
 'sand_summary_capture_recovery_independent_transition_v1.json',
)
PRODUCTS={
 'sand_collect_A':('A.stopped_collection.json','a4cc77129921c8ad123f487df229a9840cbd0865742858d7bbf8678df4c2fd65',1143012275),
 'sand_collect_B':('B.stopped_collection.json','acd5bc51262459c35f69898282a7faa64fb56a73c0e972cf1dc0d1731ac118ba',571777697),
 'sand_saved_A':('A.saved_array_audit.json','cbaf475998224b324ee68a0648e0f6e3e447790f4d0be981c7375915bbb9c304',11580671),
 'sand_saved_B':('B.saved_array_audit.json','f43624ba6bc7c17da079878007cf83fc365d9ee8def9af951883cad334e443f1',6109443),
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
inventory={};references={};omissions={};product_refs={};review_pins={}
def reference(path,pin,reason,authority,rehash=False):
 name=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
 old=references.get(name)
 if old:assert old['sha256']==pin,name
 # Metadata only, never read a referenced result/capture/array payload.
 size=path.stat().st_size if path.is_relative_to(PREP) and path.is_file() else None
 references[name]={'source':name,'sha256':pin,'bytes_from_local_stat':size,
  'identity_basis':authority,'identity_rehashed_by_reference_step':rehash,
  'content_parsed_by_reference_step':False,'copied_by_reference_step':False,'reason':reason}
def copy(path,role,pin=None):
 assert path.is_file() and not path.is_symlink() and path.stat().st_size<5_000_000,str(path)
 relative='payload/workspace/'+str(path.relative_to(ROOT))
 if relative in inventory:return
 raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
 assert pin is None or actual==pin,str(path)
 text=raw.decode()
 assert not re.search(r'-----BEGIN .*PRIVATE KEY-----|\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b',text),str(path)
 target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
 shutil.copyfile(path,target);assert sha(target)==actual
 inventory[relative]={'source':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':actual,'role':role,'transformation':'none'}
def refs_from(record,authority):
 for name,pin in record.get('evidence_sha256',{}).items():
  if not isinstance(pin,str) or len(pin)!=64:continue
  path=Path(name)
  if not path.is_absolute():path=PREP/path
  if any(s in str(path) for s in ('paired_scalar_summary.json','paired_saved_array_audit.json','sand_paired.','sand_summarize.collected')):continue
  reference(path,pin,'bound_completed_evidence_reference_not_bulk_copied',authority)

for family,pin in PACKAGES.items():
 directory=PREP/family;manifest=directory/'manifest.json';assert sha(manifest)==pin,family
 value=json.loads(manifest.read_text());files=value.get('files_sha256')
 assert isinstance(files,dict),family
 copy(manifest,'exact_frozen_source_package_manifest',pin)
 for name,source_pin in files.items():
  path=directory/name
  if path.suffix=='.log':
   reference(path,source_pin,'retained_raw_source_test_log',family+'/manifest.json')
  else:copy(path,'frozen_operator_scientific_source_template_test_or_review',source_pin)
for name in REVIEWS:
 path=PREP/name;copy(path,'completed_source_or_provenance_review_not_numeric_summary')
 review_pins[name]=sha(path);refs_from(json.loads(path.read_text()),name)

for family in ('sand_scalar_recovery_UNADMITTED_code_audit_v1','sand_summary_capture_recovery_UNADMITTED_code_audit_v1'):
 binding=json.loads((PREP/family/'failure_bindings.json').read_text())
 for name,pin in binding['files_sha256'].items():
  reference(Path(name),pin,'original_transport_failure_or_closed_route_retained_locally',family+'/failure_bindings.json')

provenance=json.loads((PREP/'sand_original_phase_route_UNADMITTED_statistics_v1/source_provenance.json').read_text())
for key in ('original_metadata_proxy','original_fake_fixtures'):
 item=provenance[key];copy(ROOT/item['path'],'unchanged_ancestor_route_source_or_tests',item['sha256'])
copy(PREP/'sand_scalar_recovery_development_history_code_audit_v1/retained_generated_cache_inventory.json','retained_source_development_history_reference')

for operation,(filename,pin,size) in PRODUCTS.items():
 review_path=STATE/(operation+'.review.json');review=json.loads(review_path.read_text())
 assert review['status']=='accepted_stopped_product' and review['operation']==operation and review['product_sha256']==pin
 path=Path(review['local_product']);assert path.name==filename and path.is_relative_to(STATE)
 assert path.stat().st_size==size
 independent=PREP/(operation+'_completed_independent_review_code_audit_v1.json')
 independent_record=json.loads(independent.read_text())
 assert independent_record['product_sha256']==pin and independent_record['accepted_root_review_sha256']==sha(review_path)
 copy(review_path,'accepted_original_product_root_review_no_scientific_success_implied')
 reference(path,pin,'large_original_collection_or_audit_retained_locally_not_read_or_copied',str(review_path.relative_to(PREP)))
 product_refs[operation]={'source':str(path.relative_to(ROOT)),'sha256':pin,'bytes':size,
  'root_review_source':str(review_path.relative_to(ROOT)),'root_review_sha256':sha(review_path),
  'independent_review_source':str(independent.relative_to(ROOT)),'independent_review_sha256':sha(independent),
  'raw_content_read':False,'raw_hash_recomputed':False,'identity_basis':'accepted original root and independent review; current local size agrees',
  'copy_or_compression_performed':False}

for name in ('analysis_phase.json','local_phase_anchor.json','B.scalars_transferred_to_A.json','sand_saved_A.recovery_closure_receipt.json'):
 copy(STATE/name,'completed_original_phase_or_transfer_recovery_receipt')
# Bind top-level original tool envelopes without publishing their transport text.
tool_names=(
 'sand_A_original_51595_exit_root_v1.json','sand_B_original_12559_exit_root_v1.json',
 'sand_saved_A_original_tool_exit_root_v1.json','sand_saved_A_close_failure_original_tool_root_v1.json',
 'sand_B_transfer_failure_original_tool_root_v1.json','sand_original_phase_proxy_original_exit_root_v1.json',
 'sand_original_phase_proxy_scoped_interrupt_root_v1.json','sand_original_phase_proxy_and_failed_transports_closed_root_v1.json',
 'sand_recovery_fate_original_tools_root_v1.json','sand_recovery_close-saved-A_original_tools_root_v1.json',
 'sand_recovery_transfer-B-collection_original_tools_root_v1.json','sand_recovery_transfer-B-audit_original_tools_root_v1.json',
 'sand_recovery_complete_transfer_original_tools_root_v1.json','sand_summarize_original_tool_exit_root_v1.json',
 'sand_summarize_close_failure_original_tool_root_v1.json','sand_summarize_capture_recovery_original_tools_root_v1.json')
for name in tool_names:
 path=PREP/name
 assert path.stat().st_size<5_000_000,name
 pin=sha(path);reference(path,pin,'original_raw_tool_transport_envelope_local_reference_only','exact local byte hash, content not parsed',rehash=True)

write(OUT/'source_inventory.json',inventory)
write(OUT/'source_package_pins.json',PACKAGES)
write(OUT/'completed_review_pins.json',review_pins)
write(OUT/'large_local_product_references.json',product_refs)
write(OUT/'referenced_evidence_inventory.json',{'schema':'sand_completed_evidence_reference_inventory_v1','files':references,
 'scope':'Only fixed completed source/provenance/failure inputs. Numeric summary, final paired products and final admission were not read.'})
write(OUT/'candidate_status.json',{'status':'UNADMITTED_SOURCE_PROVENANCE_AND_FAILURE_CANDIDATE',
 'scientific_accuracy_admission':False,'publication_admission':False,'numeric_summary_read':False,
 'statistics_derived':False,'scientific_workers_or_tests_rerun':False,'large_products_copied_or_compressed':False,
 'original_sources_changed':False,'remote_probes_or_clocks_used':False,
 'pending_root_supplied_items':['accepted numeric summary and its completed independent review','accepted final paired audit and review','final full-cohort admission','final successor route exit and cleanup evidence'],
 'preserved_original_analysis_stop_utc':'2026-10-06T23:33:01.226132+00:00'})
print(json.dumps({'source_and_provenance_copies':len(inventory),'copied_bytes':sum(v['bytes'] for v in inventory.values()),
 'referenced_originals':len(references),'large_product_reference_count':len(product_refs),'numeric_products_read':False},indent=2))
