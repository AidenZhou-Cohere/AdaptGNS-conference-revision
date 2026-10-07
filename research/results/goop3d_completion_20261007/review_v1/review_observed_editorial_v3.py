"""Narrow editorial delta review; old numerical review is inherited unchanged."""
import ast
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
P=HERE.parent/'presentation_observed_v1'
V2=P/'generated_candidate_v2';V3=P/'generated_candidate_v3'
checks=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def need(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
need(sha(P/'history/render_observed.reviewed_v2.py')=='88eb4d966c0a2915cca598e8df091d70fb2c1a1249b233cf1bb8756f9e30914a','original renderer retained exactly')
need(sha(P/'render_observed.py')=='5e697b824d241f260c9ea7cb838bebc2b77cdac59ce7900f86282b23013422b4','new renderer exact')
need(sha(V3/'receipt.json')=='69d6046f000ae25f7253f23a10eb8b13eb5f4f40505283703fd2fff570db303c','new receipt exact')
old_ast=ast.parse((P/'history/render_observed.reviewed_v2.py').read_text())
new_ast=ast.parse((P/'render_observed.py').read_text())
need(ast.dump(ast.Module(body=[n for n in old_ast.body if not (isinstance(n,ast.FunctionDef) and n.name=='render')],type_ignores=[]))==ast.dump(ast.Module(body=[n for n in new_ast.body if not (isinstance(n,ast.FunctionDef) and n.name=='render')],type_ignores=[])),'all non-render functions/constants/imports unchanged')
for name in ('source_summary_unchanged.json','complete_observed_companion.json.gz','findings_observed.md','inputs.json','observed_clean.tex','observed_diagnostics.tex','observed_accounting.tex'):
    need((V2/name).read_bytes()==(V3/name).read_bytes(),'exact retained payload '+name)
before=(V2/'appendix_goop3d_observed.tex').read_text();after=(V3/'appendix_goop3d_observed.tex').read_text()
tables=lambda text:re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',text,re.S)
need(len(tables(before))==13 and tables(before)==tables(after),'all13 table bodies byte-identical')
v2claims=load(V2/'claim_source_map.json')['table_values'];v3claims=load(V3/'claim_source_map.json')['table_values']
need([c for c in v3claims if c['table_label'].startswith('tab:')]==v2claims,'all224 original table claims byte-equivalent')
prose=[c for c in v3claims if c['table_label'].startswith('prose:')]
need(len(prose)==8 and len(v3claims)==232,'exactly eight added prose source claims')
summary=load(V3/'source_summary_unchanged.json')
values={}
for claim in prose:
    obj=summary
    for key in claim['json_path']:obj=obj[key]
    seeds=[obj['seed_values'][str(s)] for s in range(3)]
    need(claim['seed_signs']==['+' if v>0 else '-' if v<0 else '0' for v in seeds],'prose seed signs exact')
    need(claim['summary_sha256']==sha(V3/'source_summary_unchanged.json') and claim['display_multiplier']==1e9,'prose source and scale exact')
    values[tuple(claim['json_path'])]=obj
intro=(V3/'observed_intro.tex').read_text()
def root(stage):return ['families',stage]
def negatives(obj):return sum(v<0 for v in obj['seed_values'].values())
def positives(obj):return sum(v>0 for v in obj['seed_values'].values())
for stage,count in (('same_state_valid',2),('same_state_test',1)):
    need(negatives(values[tuple(root(stage)+['mix_minus_base_training','accuracy/random25/position_coordinate_mse'])])==count,'fixedrandom exact improvement count '+stage)
need('two of three validation seeds and one of three test seeds' in intro,'exposure heterogeneity correctly stated')
risk='previous-observed-base-risk25_minus_random25'
gaps={}
for stage in ('same_state_valid','same_state_test'):
    prefix=root(stage)+['accuracy_policy_contrasts','position_coordinate_mse']
    gaps[stage]={a:values[tuple(prefix+['within_arm',a,risk])] for a in ('base','mix')}
    interaction=values[tuple(prefix+['risk_minus_random_mix_minus_base'])]
    need(negatives(interaction)==(2 if stage=='same_state_valid' else 3),'interaction signs '+stage)
need(positives(gaps['same_state_test']['mix'])==2 and gaps['same_state_test']['mix']['mean']>0,'mixedtest adverse gap exact')
need(all(negatives(gaps['same_state_valid'][a])==2 for a in ('base','mix')),'validation withinarm signs exact')
def pm(obj):return '$'+format(obj['mean']*1e9,'.6g')+r'\pm'+format(obj['sample_sd']*1e9,'.6g')+'$'
need('from '+pm(gaps['same_state_test']['base'])+' to '+pm(gaps['same_state_test']['mix']) in intro,'new prose gap means and SD exact')
need('all three seeds' in intro and 'mean gap is positive' in intro and 'negative in two seeds and positive in one' in intro,'new prose adverse findings exact')
need(all(word not in intro for word in ('checker','cache were','failure record','hashes')),'operational checker history removed from manuscript prose')
need('preceding observed base history' in intro and 'preceding selected graph' in intro,'observed and autonomous risk definitions distinct')
need('do not establish autonomous accuracy, stability or speedup' in intro,'no observed to autonomous extrapolation')
need('scientific notation is explicit' not in after and 'No values are selected by sign' not in after,'implementation boilerplate removed')
need('observed validation. Every row' in after and 'observed test. Every row' in after,'accuracy caption punctuation corrected')
need(r'in $10^{-9}$ coordinate-squared units' in after and 'entries multiplied' not in after,'unit caption simplified without data scaling change')
receipt=load(V3/'receipt.json')
for name,entry in receipt['files'].items():
    path=V3/name
    need(sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes'],'new candidate receipt binding '+name)
report={'status':'passed_narrow_editorial_delta_review','checks':len(checks),'check_descriptions':checks,
    'renderer_sha256':sha(P/'render_observed.py'),'receipt_sha256':sha(V3/'receipt.json'),
    'prior_presentation_review_sha256':sha(HERE/'observed_presentation_v2_review.json'),
    'actual_product_review_sha256':sha(HERE/'observed_final_products_v2_review.json'),
    'all13tabular_blocks_byte_identical':True,'all224tableclaims_unchanged':True,'new_prose_claims':8,
    'summary_and_complete_companion_unchanged':True,'scientific_reexecution':False,
    'verdict':'The revised prose reports seed heterogeneity, positive test risk gaps and validation disagreement correctly. It removes operational checker history, retains original supporting evidence and distinct observed/autonomous meanings, and changes no numeric table value.',
    'remaining':'Root integration and native compilation; PDF export/visual inspection is a separate author/app step.'}
out=HERE/'observed_editorial_v3_review.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'review':str(out),'sha256':sha(out),'checks':len(checks),'status':report['status']}))
