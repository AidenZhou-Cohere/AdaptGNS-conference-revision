"""Focused integration delta check; no compilation or scientific rerun."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
BACKUP=BASE/'integration_root_v1'
checks=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def check(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
original=load(BACKUP/'original_source_manifest.json')
manifest=load(BACKUP/'integration_manifest.json')
compilation=load(BACKUP/'compile_and_exact_delta.json')
for name,entry in original.items():
    p=Path(entry['preserved_as'])
    check(sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],'original backup exact '+name)
for name,entry in manifest['outputs'].items():
    p=Path(name)
    check(sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],'current source exact '+name)
old=(BACKUP/'revised_manuscript.tex').read_text();new=Path('outputs/revised_manuscript.tex').read_text()
fragment=Path('work/goop3d_observed_appendix.tex').read_text()
check(fragment==(BASE/'presentation_observed_v1/generated_candidate_v4/appendix_goop3d_observed.tex').read_text(),'exact reviewedv4 standalone fragment')
check(not Path('work/goop3d_autonomous_appendix.tex').exists(),'no autonomous insertion source exists')
insert=fragment.replace('% GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT','')
removed=manifest['removed_attempted_paragraph'];addition=manifest['main_addition']
check(old.count(removed)==1 and new.count(addition)==1,'exact one replacement and one main addition')
expected=old.replace(removed,insert)
anchor='A better interaction therefore need not mean better allocation than random.'
check(expected.count(anchor)==1,'unique main insertion anchor')
expected=expected.replace(anchor,anchor+addition)
check(expected==new,'entire generated source exact declared two-part delta')
check((BACKUP/'manuscript_body.tex').read_text().replace(removed,'% GOOP3D_OBSERVED_APPENDIX_INSERT')==Path('work/manuscript_body.tex').read_text(),'canonical body changes only to new insert marker')
check((BACKUP/'conference_experiments_main.tex').read_text().replace(anchor,anchor+addition)==Path('work/conference_experiments_main.tex').read_text(),'canonical main insert only one new sentence')
old_builder=(BACKUP/'build_manuscript.py').read_text();new_builder=Path('work/build_manuscript.py').read_text()
lines="                     ('% GOOP3D_OBSERVED_APPENDIX_INSERT', 'goop3d_observed_appendix.tex'),\n                     ('% GOOP3D_AUTONOMOUS_H295_SEPARATE_INSERT', 'goop3d_autonomous_appendix.tex'),\n"
check(new_builder.count(lines)==1 and new_builder.replace(lines,'')==old_builder,'builder onlyadds two distinct insertion mappings')
def tables(text):
    return re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',text,re.S)
old_tables=tables(old);new_tables=tables(new);added_tables=tables(insert)
check(len(old_tables)==42 and len(added_tables)==12 and len(new_tables)==54,'all42old plus12new complete floating-table blocks')
check(len(re.findall(r'\\begin\{table\*?\}',old))==42 and len(re.findall(r'\\begin\{table\*?\}',new))==54,'all42old plus12new floating table environments')
check(Counter(new_tables)==Counter(old_tables)+Counter(added_tables),'every original and new table body exact')
figures=lambda text:re.findall(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',text,re.S)
check(figures(old)==figures(new),'all prior figure blocks unchanged')
check(re.search(r'\\aistatstitle\{[^\n]+',old).group()==re.search(r'\\aistatstitle\{[^\n]+',new).group(),'actual title unchanged')
abstract=lambda text:re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}',text,re.S).group()
check(abstract(old)==abstract(new),'actual abstract unchanged')
guard_lines=[line for line in old.splitlines() if ('page' in line.lower() and ('8' in line or 'PackageError' in line))]
check(all(line in new for line in guard_lines),'all original eightpage guard lines retained')
check('tab:goop3d25k-original-accounting' not in insert and 'An earlier observed audit stopped' not in insert,'operational history excluded from scientific fragment')
check(new.count(r'\label{sec:goop3d-25k-exposure}')==1 and r'Appendix~\ref{sec:goop3d-25k-exposure}' in addition,'new main reference resolves exactly once')
summary=load(BASE/'root/observed_finalized_v2_collection/files/observed_finalized_v2/summary.json')
test=summary['families']['same_state_test'];random=test['mix_minus_base_training']['accuracy/random25/position_coordinate_mse']
interaction=test['accuracy_policy_contrasts']['position_coordinate_mse']['risk_minus_random_mix_minus_base']
check(sum(v<0 for v in random['seed_values'].values())==1,'new main oneofthree exposure claim exact')
check(all(v<0 for v in interaction['seed_values'].values()),'new main observedtest riskgap narrowing exact')
check('observed-test' in addition and '25k' in addition,'new main claim scoped to endpoint and observed test')
check(compilation['source_sha256']==sha(Path('outputs/revised_manuscript.tex')) and compilation['native_compile_result']['kind']=='success','root original native compile success tied to current source')
check(compilation['pdf_export_performed'] is False,'PDFexport remains pending')
report={'status':'passed_focused_observed_manuscript_integration_review','checks':len(checks),'check_descriptions':checks,
    'manuscript_sha256':sha(Path('outputs/revised_manuscript.tex')),'integration_manifest_sha256':sha(BACKUP/'integration_manifest.json'),
    'root_compile_transcription_sha256':sha(BACKUP/'compile_and_exact_delta.json'),
    'prior_editorial_v4_review_sha256':sha(HERE/'observed_editorial_v4_review.json'),
    'actual_product_review_sha256':sha(HERE/'observed_final_products_v2_review.json'),
    'all42preexisting_tables_figures_title_abstract_preserved':True,'additional_scientific_tables':12,
    'compilation_scope':'Root supplied original native compiler success transcription for this exact source; this reviewer did not repeat compilation.',
    'scientific_array_model_execution':False,'pdf_export_or_visual_inspection_performed':False,
    'initial_reviewer_count_error':'Three reviewer variants conflated42complete floating tables with tabular tokens, then attempted regex/balanced parsing across style macro definitions. All failed sources/tools retained. Final review uses original claimed unit of42complete floating-table blocks; whole-source exact delta passed throughout. Actual manuscript unchanged.',
    'verdict':'Only the completed observed-scope insertion and one accurate scoped main sentence changed the generated manuscript. All prior scientific sources and values remain exact. New main claims match all three actual seed contrasts. Autonomous results remain separately pending.'}
out=HERE/'observed_manuscript_integration_review.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'checks':len(checks),'status':report['status']}))
