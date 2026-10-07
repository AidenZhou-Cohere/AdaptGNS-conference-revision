"""Independent formatting/byte review; no scientific or renderer execution."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PRESENTATION=BASE/'presentation_observed_v1'
CANDIDATE=PRESENTATION/'generated_candidate_v2'
PRODUCT=BASE/'root/observed_finalized_v2_collection/files/observed_finalized_v2'
checks=0
def require(value,label):
    global checks
    if not value:raise AssertionError(label)
    checks+=1
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def get(value,path):
    for key in path:value=value[key]
    return value
def number(value,scale,signed=False):
    if value is None:return '---'
    text=format(value*scale,'+.6g' if signed else '.6g')
    if 'e' in text:
        mantissa,exponent=text.split('e');text=mantissa+r'\!\times\!10^{'+str(int(exponent))+'}'
    return text
def cells(record,scale,signed=False):
    pm='---' if record['mean'] is None else '$'+number(record['mean'],scale,signed)+r'\pm'+number(record['sample_sd'],scale)+'$'
    seeds=['---' if record['seed_values'][str(s)] is None else '$'+number(record['seed_values'][str(s)],scale,signed)+'$' for s in range(3)]
    return [pm,*seeds]

receipt=load(CANDIDATE/'receipt.json')
require(receipt['renderer_sha256']==sha(PRESENTATION/'render_observed.py')=='88eb4d966c0a2915cca598e8df091d70fb2c1a1249b233cf1bb8756f9e30914a','reviewed renderer unchanged')
require(receipt['statistic_objects']==1382 and receipt['required_observed_cells']==2568 and receipt['required_all_cells']==4728,'full observed/statistic counts')
require(receipt['source_summary_bytes_unchanged'] and not receipt['manuscript_edited'] and not receipt['autonomous_products_read'] and not receipt['science_executed'],'formatting-only scope')
require(set(receipt['files'])|{'receipt.json'}=={p.name for p in CANDIDATE.iterdir() if p.is_file()},'exact candidate file set')
for name,entry in receipt['files'].items():
    path=CANDIDATE/name
    require(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'candidate byte hash '+name)
summary=load(PRODUCT/'summary.json')
require((CANDIDATE/'source_summary_unchanged.json').read_bytes()==(PRODUCT/'summary.json').read_bytes(),'original summary bytes exact')
inputs=load(CANDIDATE/'inputs.json')
require(sha(CANDIDATE/'inputs.json')==receipt['inputs_manifest_sha256'],'input manifest pinned')
require('107e11a40f174f069bf73016e21f8dfdffae2c88e2d74212c99e806421fb3fd5' in (CANDIDATE/'inputs.json').read_text(),'actual product review bound')
for name,entry in inputs['products'].items():
    require(sha(Path(entry['path']))==entry['sha256'],'exact input '+name)
raw=gzip.decompress((CANDIDATE/'complete_observed_companion.json.gz').read_bytes())
require(len(raw)==receipt['companion_uncompressed_bytes'] and hashlib.sha256(raw).hexdigest()==receipt['companion_uncompressed_sha256'],'complete companion expansion hash')
companion=json.loads(raw)
require(companion['summary_unchanged']==summary,'companion complete summary exact')
require(companion['namespace']=='observed' and companion['autonomous_numerical_products_included'] is False,'separate observed namespace')
expected_index={}
def walk(value,path):
    if type(value)is dict:
        if 'seed_values' in value:
            expected_index['/'.join(path)]={'json_path':path,'statistic':value,
                'seed_signs':['null' if value['seed_values'][str(s)]is None else '+' if value['seed_values'][str(s)]>0 else '-' if value['seed_values'][str(s)]<0 else '0' for s in range(3)]}
        else:
            for key,v in value.items():walk(v,path+[key])
walk(summary['families'],['families'])
require(len(expected_index)==1382 and companion['statistic_index']==expected_index,'all1382 complete statistic objects retained exactly')
require(companion['audit_provenance']=={k:v for k,v in load(PRODUCT/'audit.json').items() if k!='models'},'full audit provenance and original accounting retained')
for name in ('arithmetic_check','completion','retained_original_failure'):
    require(companion[name]==load(PRODUCT/(name+'.json')),'original companion product '+name)
require(companion['retained_checker_diagnosis']==load(HERE/'observed_arithmetic_diagnosis_v1/diagnosis.json'),'original36SDdiagnosis retained')

claims=load(CANDIDATE/'claim_source_map.json')['table_values']
require(len(claims)==224,'all224 rendered statistic references')
grouped=defaultdict(list)
for claim in claims:
    require(claim['summary_sha256']==sha(PRODUCT/'summary.json'),'claim exact source summary')
    obj=expected_index['/'.join(claim['json_path'])]
    require(claim['seed_signs']==obj['seed_signs'],'claim ordered seed signs')
    require(claim['display_multiplier']==(1e9 if 'position_coordinate_mse' in claim['json_path'][-1] or 'position_coordinate_mse' in claim['json_path'] else 1),'claim display scale')
    grouped[claim['table_label']].append(claim)
text=(CANDIDATE/'appendix_goop3d_observed.tex').read_text()
blocks=re.findall(r'\\begin\{table\*\}\[p\].*?\\end\{table\*\}',text,re.S)
require(len(blocks)==13,'all13 tables')
seen=set();displayed_numbers=0
for block in blocks:
    label=re.search(r'\\label\{([^}]+)\}',block).group(1)
    require(label not in seen,'unique table label');seen.add(label)
    body=block.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
    rows=[[x.strip() for x in line.strip().removesuffix(r'\\').split('&')] for line in body.splitlines() if line.strip()]
    if label=='tab:goop3d25k-original-accounting':
        require(len(rows)==5,'all five historical stages')
        for row in rows:
            name=row[0].replace(r'\_','_')
            actual=[c for stage in summary['all_original_accounting'] if stage['stage']==name for c in stage['cells']]
            expected=[len(actual)]+[sum(c['state']==s for c in actual) for s in ('completed_required_outcome','recorded_failed_outcome','timed_out_current','not_completed_before_invocation_end','never_started')]
            require(list(map(int,row[1:]))==expected,'historical accounting row exact')
        continue
    mapped=grouped[label]
    if '-timing-' in label:
        require(len(rows)*4==len(mapped),'four fixed timing statistics per row')
        for i,row in enumerate(rows):
            expected=[cells(get(summary,c['json_path']),c['display_multiplier'])[0] for c in mapped[i*4:i*4+4]]
            require(row[2:]==expected,'timing display values exact')
            displayed_numbers+=8
    else:
        require(len(rows)==len(mapped),'one fixed statistic per row')
        for row,claim in zip(rows,mapped):
            signed=('-training' in label or '-contrasts' in label or (label=='tab:goop3d25k-clean' and row[0]=='mix-minus-base'))
            expected=cells(get(summary,claim['json_path']),claim['display_multiplier'],signed)
            require(row[-4:]==expected,'mean SD allseeds display exact')
            displayed_numbers+=5
require(set(grouped)<=seen,'all claimed tables present')
require(r'10^9' in text and r'10^{-9}' in text,'position units explicit')
require('GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT' in text,'autonomous insertion remains distinct')
require('isolated hardware timing or causal speedup' in text and 'SD is not a confidence interval' in text,'limits retained')
require('native base' in (CANDIDATE/'findings_observed.md').read_text(),'exposure and native expansion distinct')
review={'status':'passed_observed_presentation_source_and_actual_byte_review','checks':checks,
    'renderer_sha256':receipt['renderer_sha256'],'candidate_receipt_sha256':sha(CANDIDATE/'receipt.json'),
    'candidate_inputs_sha256':receipt['inputs_manifest_sha256'],'actual_product_review_sha256':sha(HERE/'observed_final_products_v2_review.json'),
    'complete_statistic_objects':1382,'rendered_statistic_claims':224,'latex_tables':13,'formatted_numeric_fields_checked':displayed_numbers,
    'independent_synthetic_tests':8,'independent_test_tool':'0f60a5','independent_test_exit':0,
    'all_six_policies_both_splits_all_seeds_retained':True,'summary_bytes_unchanged':True,
    'original_arrays_models_or_science_executed':False,'manuscript_edited':False,
    'editorial_notes':['Add punctuation between the short accuracy caption title and its following sentence during manuscript integration.',
        'Keep detailed checker-operation history in companion/supporting provenance; the manuscript should lead with scientific findings.',
        'Native compilation and visual PDF inspection remain separate integration checks; formatting review does not certify page layout.'],
    'scientific_interpretation':'Observed test risk-minus-random remains positive on mean under both arms, though graph exposure narrows that gap in all three seeds. Fixed-random exposure improves observed-test mean but only one of three seeds. Validation signs differ. No autonomous claim follows.'}
path=HERE/'observed_presentation_v2_review.json';path.write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'review':str(path),'sha256':sha(path),'checks':checks,'status':review['status']}))
