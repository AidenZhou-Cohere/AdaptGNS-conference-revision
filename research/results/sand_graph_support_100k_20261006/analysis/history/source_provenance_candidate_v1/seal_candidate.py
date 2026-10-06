"""Hash only this compact candidate and its copied source/provenance originals."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
inventory=json.loads((OUT/'source_inventory.json').read_text())
for name,record in inventory.items():
 assert (OUT/name).stat().st_size==record['bytes']
 assert sha(OUT/name)==record['sha256']==sha(ROOT/record['source'])
assert not any(p.is_symlink() for p in OUT.rglob('*'))
names={str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file() and p.name not in ('candidate_manifest.json','FILELIST.txt','sealing_verification.json')}
names.add('FILELIST.txt')
(OUT/'FILELIST.txt').write_text(''.join(n+'\n' for n in sorted(names|{'candidate_manifest.json','sealing_verification.json'})))
files={n:{'bytes':(OUT/n).stat().st_size,'sha256':sha(OUT/n)} for n in sorted(names)}
manifest={'schema':'sand_completed_analysis_source_provenance_candidate_manifest_v1','status':'UNADMITTED',
 'files':files,'file_count_excluding_manifest_and_verification':len(files),
 'bytes_excluding_manifest_and_verification':sum(v['bytes'] for v in files.values()),
 'scientific_accuracy_admission':False,'numeric_summary_or_paired_product_read':False,
 'large_original_collections_audits_models_and_arrays_not_copied':True,
 'manifest_exclusions':['candidate_manifest.json','sealing_verification.json']}
write(OUT/'candidate_manifest.json',manifest)
result={'status':'passed_source_and_provenance_byte_preservation_only','candidate_manifest_sha256':sha(OUT/'candidate_manifest.json'),
 'candidate_files_including_manifest_and_verification':len(files)+2,'source_and_provenance_original_copies':len(inventory),
 'all_copied_originals_unchanged':True,'candidate_bytes_excluding_verification':sum((OUT/n).stat().st_size for n in files)+(OUT/'candidate_manifest.json').stat().st_size,
 'scientific_numeric_products_read':False,'tests_or_scientific_workers_rerun':False,'independent_final_curation_review_pending':True}
write(OUT/'sealing_verification.json',result);print(json.dumps(result,indent=2))
