"""Independent sealed publication-plan review; no copy/build/scientific work."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
P=BASE/'publication_plan_v2'
checks=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def check(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
plan=load(P/'plan.json');seal=load(P/'input_seal.json');prior=load(BASE/'publication_plan_v1/plan.json')
check(sha(P/'plan.json')=='cce18e8dd2d7c970e2b12590ebf47b10fd5723431c3ed8670acc4dbd0003e15b','exact successorplan')
check(sha(P/'input_seal.json')=='577b69db4cb5482da271399e639432814039f7fddf5870110b88357b3df1c592','exact inputseal')
check(plan['prior_plan_sha256']==sha(BASE/'publication_plan_v1/plan.json')==seal['prior_plan']['sha256'],'preserved old plan lineage')
check(plan['integration_seal_sha256']==sha(P/'input_seal.json'),'plan binds exact seal')
entries={e['destination']:e for e in plan['files']}
check(len(entries)==len(plan['files'])==393 and sum(e['bytes'] for e in entries.values())==43504255,'393unique files43504255bytes')
expected={e['destination']:e for e in prior['files']}
for replacement in seal['replace']:
    old=expected.pop(replacement['destination']);expected[replacement['preserve_old_as']]=dict(old,destination=replacement['preserve_old_as'],role='superseded_curation_history')
for e in seal['add']:
    check(e['destination'] not in expected,'explicit unique successor addition')
    expected[e['destination']]=e
seal_entry=entries['publication_tools/input_seal.json'];expected[seal_entry['destination']]=seal_entry
check(expected==entries,'successor exactly equals preserved plan plus explicit sealed changes')
for e in entries.values():
    path=Path(e['source'])
    check(path.is_file() and not path.is_symlink() and path.stat().st_size==e['bytes'] and sha(path)==e['sha256'],'every selected source exact '+e['destination'])
    check(path.suffix not in ('.npz','.npy','.pt','.pkl','.gz','.tar','.zip','.pdf'),'no rawarray model archive PDF '+e['destination'])
def included(name):return load(Path(entries[name]['source']))
def binding(name,pin):check(name in entries and entries[name]['sha256']==pin,'exact included review/input '+name)
gate=plan['numerical_product_gate'];review=included(gate['review']['destination'])
binding(gate['review']['destination'],'107e11a40f174f069bf73016e21f8dfdffae2c88e2d74212c99e806421fb3fd5')
check(plan['numerical_product_gate']==prior['numerical_product_gate'],'actual observed admission gate unchanged')
check({e['destination'] for e in entries.values() if e['role']=='verified_observed_product'}=={b['destination'] for b in gate['products'].values()},'only five admitted observed numerical products')
for name,b in gate['products'].items():
    expected=review['completion_sha256'] if name=='completion.json' else review['products_sha256'][name]
    binding(b['destination'],expected)
lineage=plan['presentation_lineage'];v4=included(lineage['review']['destination']);v3=included(lineage['prior_editorial_review']['destination']);v2=included(lineage['original_presentation_review']['destination'])
binding(lineage['review']['destination'],'38bfd09d86f1806d435ab2766c386cef82ac3d2b87884bac7448dbf676770102')
binding(lineage['prior_editorial_review']['destination'],v4['prior_editorial_review_sha256'])
binding(lineage['original_presentation_review']['destination'],v3['prior_presentation_review_sha256'])
check(v2['actual_product_review_sha256']==gate['review']['sha256'],'presentation lineage ends at admitted actualproducts')
binding(lineage['renderer']['destination'],v4['renderer_sha256']);binding(lineage['receipt']['destination'],v4['receipt_sha256']);binding(lineage['combined_appendix']['destination'],v4['combined_appendix_sha256'])
integration=plan['integration'];ir=included(integration['review']);im=included(integration['manifest']);compile=included(integration['compile_receipt'])
binding(integration['review'],'f7bf49f2cbe34a851f35b93f94ea05a09ff484f3a53f5a50bf559df542c0b571')
binding(integration['manuscript'],'2e6b43c8178ead7d1fef331d8f6437f4d156d73b7cf07b305385ba66b21c7291')
check(ir['manuscript_sha256']==compile['source_sha256']==entries[integration['manuscript']]['sha256'] and compile['native_compile_result']['kind']=='success' and compile['pdf_export_performed'] is False,'exact alreadycompiledsource notPDFexport')
check(ir['prior_editorial_v4_review_sha256']==lineage['review']['sha256'] and ir['actual_product_review_sha256']==gate['review']['sha256'],'integration review bound to finalv4andactualscience')
for name,dest in integration['canonical_sources'].items():
    check(im['outputs'][name]=={k:entries[dest][k] for k in ('bytes','sha256')},'all five canonical source bindings')
builder=load(P/'manuscript_builder_inputs.json')
check(len(builder['files'])==18 and builder['absent_optional_inputs']==['work/goop3d_autonomous_appendix.tex'],'all18present builderinputs and explicitabsentautonomous')
for source,value in builder['files'].items():
    check({k:entries['manuscript/'+source][k] for k in ('bytes','sha256')}==value,'included exact builder input '+source)
check(not Path('work/goop3d_autonomous_appendix.tex').exists() and builder['build_or_scientific_code_executed'] is False,'no autonomous inserted or rebuild')
scope=load(P/'status_document_scope.json')
for original,entry in scope['author_facing_absolute_path_copies'].items():
    check({k:entries[entry['destination']][k] for k in ('bytes','sha256')}=={k:entry[k] for k in ('bytes','sha256')},'sealed authorstatus snapshot '+original)
    check(Path(entries[entry['destination']]['source']).resolve().is_relative_to(P/'author_status_snapshot'),'author documents stored in frozen snapshot')
for path,pin in scope['repository_root_documents_staged_separately_by_owner'].items():
    check(sha(Path(path))==pin['sha256'] and Path(path).stat().st_size==pin['bytes'],'separatelystaged repo doc pin '+path)
readme=(P/'PACKAGE_README.md').read_text()
for phrase in ('reviewed final v4','12 scientific tables','224 table claims and eight','native compiler-success transcription','Autonomous completion is not admitted','reference','fresh PDF export','local absolute-path links','unadmitted first curation candidate'):
    check(phrase in readme,'corrected public scope '+phrase)
check(sha(BASE/'publication_plan_v1/copy_verified.py')=='9c0bf98b0a35aa582830e4c5de57da9b0b28d89ea719dce6383357940c57c250','previouslyreviewed testedcopier unchanged')
tree=ast.parse((P/'prepare_successor.py').read_text());calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
check(not any(x.endswith(('.glob','.rglob','.iterdir')) for x in calls),'successor builder performs no directorydiscovery')
check(not any('subprocess' in x or x in ('eval','exec') for x in calls),'successor has no launch/dynamicexecution')
check('Autonomous completion and independent actual-product review' in plan['pending'] and 'Fresh PDF export and final visual/page inspection' in plan['pending'],'unfinished scope remains explicit')
report={'status':'passed_explicit_successor_publication_plan_review','checks':len(checks),'plan_sha256':sha(P/'plan.json'),'input_seal_sha256':sha(P/'input_seal.json'),'selected_files':393,'selected_bytes':43504255,'copier_sha256':sha(BASE/'publication_plan_v1/copy_verified.py'),'copier_prior_source_review_sha256':sha(HERE/'publication_copier_source_review.json'),'actual_copy_performed':False,'scientific_model_array_or_builder_execution':False,'reviewed_scope':'Five exact admitted observed products; all original source/failure history; finalv4presentation and exact nativecompiled2e6b43c8manuscript/inputclosure; frozen authorstatus copies. No autonomous numerical product admission, rawarrays/model/cache/archive/PDFcopy, Git or network.', 'preserved_failures':'Unadmitted v1 dynamic snapshot/README scope error; successor initial README duplicate-selection expectation; all scientific and reviewer failures remain included.', 'remaining':'Root may copy once using this exact reviewed plan and unchanged copier; exact stagedbytes/fork-only publication verification follow. This receipt is separate from the sealed package and does not amend its inventory.'}
out=HERE/'publication_plan_v2_review.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'review':str(out),'sha256':sha(out),'checks':len(checks),'status':report['status']}))
