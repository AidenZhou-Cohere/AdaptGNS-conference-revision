"""Read-only independent arithmetic/allocation audit of final same-state files.

No model loading or inference. The reviewed strict loader is an additional
provenance check; all numeric metrics below are independently recomputed from
saved arrays and measured calls, without invoking the aggregation writer.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "outputs/AdaptGNS"
sys.path.insert(0, str(REPO))
from research import summarize_full_same_state as strict

OUT = Path(__file__).with_suffix('.json')
EVAL = ROOT / 'work/full-evaluation/same_state'
SUMMARY = ROOT / 'work/full-evaluation/reports/full_same_state.json'
DATA = ROOT / 'work/full-data/converted'
POLICIES = ('base', 'dense', 'random25', 'speed25', 'previous-observed-base-risk25')
CASES = POLICIES + ('base_shared_superset',)
TARGETS = (7,106,205,304,403,502,601,700,799,898,1000)
CORRELATIONS = ('previous_risk_vs_base_error','previous_risk_vs_dense_benefit','current_base_risk_vs_base_error','current_base_risk_vs_dense_benefit')
TFIELDS = ('end_to_end_seconds','score_generation_seconds','score_graph_seconds','score_forward_seconds','current_graph_and_selection_seconds','current_forward_seconds')
CHECKS = 0
HASHES = {}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def remember(path):
    path=Path(path); h=digest(path); HASHES[str(path.relative_to(ROOT))]=h; return h

def read(path):
    remember(path); return json.loads(Path(path).read_text())

def check(ok, label):
    global CHECKS
    CHECKS += 1
    if not ok: raise AssertionError(label)

def close(x,y,label,atol=1e-15):
    if x is None or y is None: check(x is None and y is None,label); return
    check(math.isclose(float(x),float(y),rel_tol=2e-12,abs_tol=atol),f'{label}: {x!r} != {y!r}')

def arrhash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def canonical(a):
    a=np.asarray(a,dtype=np.int64).reshape(-1,2)
    return a[np.lexsort((a[:,1],a[:,0]))]
def pset(a): return set(map(tuple,np.asarray(a).tolist()))
def same(a,b,label): check(np.array_equal(a,b),label)
def mean(v): return math.fsum(map(float,v))/len(v)
def eqtraj(rows,name):
    vals=[mean([r['metrics'][name] for r in rows if r['source']==s]) for s in range(3,30)]
    return mean(vals)
def across(v): return {'seed_values':v,'mean':mean(v),'sample_sd':statistics.stdev(v)}
def compare_across(saved,values,label):
    check(saved['defined_seed_count']==3,label+' seed count')
    for i,value in enumerate(values):
        check(saved['seed_values'][i]['seed']==i,label+' seed index')
        close(saved['seed_values'][i]['value'],value,label+' seed value')
    close(saved['mean'],mean(values),label+' mean'); close(saved['sample_sd'],statistics.stdev(values),label+' sd')

def run():
    summary=read(SUMMARY)
    check(summary['state']=='complete' and summary['complete_models']==6 and summary['aggregation_eligible_models']==6,'six complete')
    check(summary['consistency_errors']==[],'no consistency errors')
    training=remember(REPO/'research/protocols/full_waterdrop_100k.md')
    companion=remember(REPO/'research/protocols/full_same_state_diagnostic.md')
    evaluator=remember(REPO/'research/full_same_state.py')
    remember(REPO/'research/summarize_full_same_state.py')
    manifest=read(DATA/'test.json')
    sources={}; common={}; geometry={}; random_identity={}; allruns={}; strict_runs=[]
    counts={'frames':0,'timed_predictor_calls':0,'warmup_predictor_calls':0,'network_passes':0,'failed_frames':0,'undefined_correlations':0,'nonidentical_repeat_pair_sets':0}
    repeat_max={case:0. for case in CASES}; natural_shared_max=0.
    for objective,seed in itertools.product(('faithful','nll'),range(3)):
        name=f'{objective}_seed{seed}'; directory=EVAL/name
        checked=strict.load_run(directory,objective,seed,training,companion,evaluator)
        check(checked['state']=='complete' and checked['eligible'] and not checked['errors'],f'{name} strict loader')
        strict_runs.append({k:checked[k] for k in ('objective','seed','state','eligible','result_sha256','protocol_sha256','checkpoint_sha256')})
        protocol=read(directory/'protocol.json'); index=read(directory/'result.json'); status=read(directory/'status.json')
        check(protocol['manifest_sha256']==HASHES[str((DATA/'test.json').relative_to(ROOT))],name+' manifest hash')
        rows=[]; keys=[]
        for compact in index['records']:
            row=read(directory/compact['record_file']); apath=directory/row['array_file']; ahash=remember(apath)
            check(ahash==row['array_sha256'],name+' array hash')
            source,target=row['source_index'],row['target_frame']; key=(source,target); keys.append(key)
            check(row['status']=='complete' and row['failure'] is None,f'{name} {key} complete')
            with np.load(apath,allow_pickle=False) as z: a={k:z[k] for k in z.files}
            current,previous,target_position,types=(a[k] for k in ('current_observed_history','previous_observed_history','target_position','particle_types'))
            n=len(types); check(current.shape==previous.shape==(6,n,2) and target_position.shape==(n,2),name+' shapes')
            same(current[:-1],previous[1:],name+' consecutive observed history')
            for array,field in ((current,'observed_history_sha256'),(previous,'previous_observed_history_sha256'),(target_position,'target_sha256')):
                check(arrhash(array)==row[field],name+' history/target checksum')
            identity=tuple(arrhash(x) for x in (current,previous,target_position,types))
            if key in common: check(common[key]==identity,name+' paired observed arrays')
            else: common[key]=identity
            desc=protocol['trajectory_records'][source-3]
            if source not in sources:
                pospath=DATA/desc['positions']['path']; typepath=DATA/desc['particle_types']['path']
                check(remember(pospath)==desc['positions']['sha256'],'source position hash')
                check(remember(typepath)==desc['particle_types']['sha256'],'source type hash')
                sources[source]=(np.load(pospath,mmap_mode='r'),np.load(typepath,mmap_mode='r'))
            pos,original_types=sources[source]
            same(current,pos[target-6:target],name+' official current history')
            same(previous,pos[target-7:target-1],name+' official previous history')
            same(target_position,pos[target].astype(np.float64),name+' official target')
            same(types,original_types,name+' official particle types')
            if key not in geometry:
                p=current[-1].astype(np.float64); pairs=canonical(cKDTree(p).query_pairs(.015*1.267,output_type='ndarray'))
                d2=np.sum((p[pairs[:,0]]-p[pairs[:,1]])**2,axis=1); mask=d2<=.015**2
                d32=np.sum((current[-1,pairs[:,0]]-current[-1,pairs[:,1]])**2,axis=1); mask32=d32<=.015**2
                natural=canonical(cKDTree(p).query_pairs(.015,output_type='ndarray'))
                base,annulus=pairs[mask],pairs[~mask]; same(natural,base,'direct vs filtered float64')
                geometry[key]={'base':base,'annulus':annulus,'frozen_base':pairs[mask32],'frozen_annulus':pairs[~mask32]}
            g=geometry[key]; base,annulus=g['base'],g['annulus']; budget=len(annulus)//4
            same(a['candidate_base_pairs_float64'],base,name+' float64 base reconstruction')
            same(a['candidate_annulus_pairs_float64'],annulus,name+' float64 annulus reconstruction')
            same(a['frozen_float32_base_pairs'],g['frozen_base'],name+' float32 base reconstruction')
            same(a['frozen_float32_annulus_pairs'],g['frozen_annulus'],name+' float32 annulus reconstruction')
            bs,es=pset(base),pset(annulus); selected={}; metrics={}
            for case in CASES:
                pairs=a['pairs_'+case]; same(pairs,canonical(pairs),name+' canonical '+case)
                selected[case]=pset(pairs)-bs
                check(len(pairs)==len(pset(pairs)) and bs<=pset(pairs)<=bs|es,name+' unique legal pairs '+case)
                check(all(i<j for i,j in pset(pairs)),name+' no self loops '+case)
                expected=0 if case in ('base','base_shared_superset') else len(annulus) if case=='dense' else budget
                check(len(selected[case])==expected,name+' exact budget '+case)
                ga=row['graph_audit']['policies'][case]
                check(ga['retained_pairs']==len(pairs) and ga['directed_edges']==2*len(pairs) and ga['retained_optional_pairs']==expected,name+' saved edge counts '+case)
                first=next(c for c in row['timed_calls'] if c['method']==case)
                check(arrhash(pairs)==first['pair_sha256'],name+' first timed pair identity '+case)
            same(a['pairs_base'],a['pairs_base_shared_superset'],name+' natural/shared exact base')
            speed=np.linalg.norm(current[-1]-current[-2],axis=-1)
            scores={'speed25':speed,'previous-observed-base-risk25':a['previous_observed_base_q']}
            for case,score in scores.items():
                values=np.maximum(score[annulus[:,0]],score[annulus[:,1]])
                order=np.lexsort((annulus[:,1],annulus[:,0],-values))[:budget]
                same(a['pairs_'+case],canonical(np.concatenate((base,annulus[order]))),name+' independently ranked '+case)
                cutoff=float(np.sort(values)[-budget]) if budget else None
                tied=int(np.sum(values==cutoff)) if budget else 0
                admitted=budget-int(np.sum(values>cutoff)) if budget else 0
                expected_ties={'applicable':True,'cutoff_score':cutoff,'tied_pairs':tied,'selected_tied_pairs':admitted,'rejected_tied_pairs':tied-admitted,'boundary_tie':0<admitted<tied}
                check(row['graph_audit']['policies'][case]['cutoff_ties']==expected_ties,name+' cutoff '+case)
            rng=np.random.default_rng(np.random.SeedSequence([93000,seed,source,target]))
            random_pairs=canonical(np.concatenate((base,annulus[rng.permutation(len(annulus))[:budget]])))
            same(a['pairs_random25'],random_pairs,name+' random stream')
            rid=(seed,source,target)
            if rid in random_identity: check(random_identity[rid]==arrhash(random_pairs),name+' paired objective random subset')
            else: random_identity[rid]=arrhash(random_pairs)
            for left,right in itertools.combinations(POLICIES,2):
                ls,rs=selected[left],selected[right]; union=ls|rs
                expected={'optional_intersection':len(ls&rs),'optional_union':len(union),'optional_jaccard':len(ls&rs)/len(union) if union else 1.}
                check(row['graph_audit']['overlaps'][left+'__'+right]==expected,name+' overlap')
            comparison=row['graph_audit']['frozen_float32_comparison']
            check(comparison=={'base_symmetric_difference':len(bs^pset(g['frozen_base'])),'annulus_symmetric_difference':len(es^pset(g['frozen_annulus'])),'frozen_base_pairs':len(g['frozen_base']),'frozen_annulus_pairs':len(g['frozen_annulus'])},name+' saved float32 difference')
            scale=np.array(row['saved_acceleration_normalization']['std'])
            for policy in POLICIES:
                residual=a['prediction_'+policy]-target_position
                pvec=np.sum(residual**2,axis=1); nvec=np.sum((residual/scale)**2,axis=1)
                same(a['position_vector_se_'+policy],pvec,name+' position squared residual '+policy)
                same(a['normalized_vector_se_'+policy],nvec,name+' normalized squared residual '+policy)
                for field,value in (('position_coordinate_mse',mean(pvec)/2),('normalized_coordinate_mse',mean(nvec)/2)):
                    close(row['accuracy'][policy][field],value,name+' accuracy '+policy)
                    metrics[f'accuracy/{policy}/{field}']=value
            bn=a['normalized_vector_se_base']-a['normalized_vector_se_dense']; bp=a['position_vector_se_base']-a['position_vector_se_dense']
            same(a['signed_dense_benefit_normalized'],bn,name+' signed normalized benefit')
            same(a['signed_dense_benefit_position'],bp,name+' signed position benefit')
            for field,value in (('mean_normalized_vector_benefit',mean(bn)),('mean_position_vector_benefit',mean(bp)),('positive_fraction',mean(bn>0)),('negative_fraction',mean(bn<0)),('zero_fraction',mean(bn==0))):
                close(row['benefit'][field],value,name+' benefit '+field); metrics['benefit/'+field]=value
            for field,x,y in zip(CORRELATIONS,(a['previous_observed_base_q'],)*2+(a['current_base_q'],)*2,(a['normalized_vector_se_base'],bn)*2):
                value=float(spearmanr(x,y).statistic)
                if not math.isfinite(value): value=None; counts['undefined_correlations']+=1
                close(row['correlations'][field]['value'],value,name+' independent scipy Spearman '+field,atol=2e-12)
                metrics['correlation/'+field]=value
            for warmup,calls in ((True,row['warmup_calls']),(False,row['timed_calls'])):
                expected_calls=[]
                for r in (-1,) if warmup else range(6):
                    start=(source*11+TARGETS.index(target)+max(r,0))%6
                    expected_calls += [(r,slot,case) for slot,case in enumerate(CASES[start:]+CASES[:start])]
                check([(c['round'],c['slot'],c['method']) for c in calls]==expected_calls,name+' balanced runtime order')
                for call in calls:
                    risk=call['method']==POLICIES[-1]
                    check(call['network_passes']==call['candidate_builds']==1+int(risk),name+' full scoring pass accounting')
                    for field in TFIELDS: check(call[field]>=0 and math.isfinite(call[field]),name+' finite measured time')
                    check(call['end_to_end_seconds']>=call['score_generation_seconds']+call['current_graph_and_selection_seconds']+call['current_forward_seconds']-1e-12,name+' timed required phases')
                    check(call['score_generation_seconds']>=call['score_graph_seconds']+call['score_forward_seconds']-1e-12,name+' nested score phases')
                counts['warmup_predictor_calls' if warmup else 'timed_predictor_calls']+=len(calls)
            for case in CASES:
                calls=[c for c in row['timed_calls'] if c['method']==case]
                distinct=len({c['pair_sha256'] for c in calls}); check(row['repeat_consistency'][case]['distinct_pair_hashes']==distinct,name+' repeat pair count')
                if distinct>1: counts['nonidentical_repeat_pair_sets']+=1
                repeat_max[case]=max(repeat_max[case],row['repeat_consistency'][case]['maximum_absolute_prediction_difference'])
                for field in TFIELDS:
                    values=[c[field] for c in calls]; stat=row['timing_summary'][case][field]
                    close(stat['mean'],mean(values),name+' repeated mean');close(stat['median'],statistics.median(values),name+' repeated median');close(stat['sample_sd'],statistics.stdev(values),name+' repeated sd')
                    metrics[f'timing/{case}/{field}']=mean(values)
            metrics['warmup/total_seconds']=math.fsum(c['end_to_end_seconds'] for c in row['warmup_calls'])
            metrics['failure_fraction']=0.
            natural_shared_max=max(natural_shared_max,row['natural_shared_base_maximum_prediction_difference'])
            check(row['completed_network_passes']==49,name+' 49 passes per frame')
            counts['network_passes']+=49; counts['frames']+=1
            rows.append({'source':source,'target':target,'n':n,'metrics':metrics,'graph':row['graph_audit'],'repeat':row['repeat_consistency'],'sign_disagreements':int(np.sum(np.sign(bn)!=np.sign(bp)))})
        check(set(keys)==set(itertools.product(range(3,30),TARGETS)) and len(keys)==297,name+' exact unique coverage')
        saved=summary['objectives'][objective]['per_seed'][seed]
        means={metric:eqtraj(rows,metric) for metric in rows[0]['metrics']}
        for metric,value in means.items(): close(saved['equal_trajectory_metrics'][metric],value,name+' independently aggregated '+metric)
        weighted={metric:math.fsum(r['metrics'][metric]*r['n'] for r in rows)/sum(r['n'] for r in rows) for metric in saved['particle_weighted_error_and_benefit']}
        for metric,value in weighted.items(): close(saved['particle_weighted_error_and_benefit'][metric],value,name+' particle-weighted '+metric)
        for field in CORRELATIONS: check(saved['correlation_counts'][field]=={'defined_completed_frames':297,'undefined_completed_frames':0},name+' correlation counts')
        allruns[name]={'rows':rows,'means':means,'weighted':weighted,'runtime':protocol['runtime'],'software':protocol['software'],'threads':protocol['threads'],'result_sha256':checked['result_sha256']}
        print(name, 'verified',len(rows),'frames',flush=True)
    for objective in ('faithful','nll'):
        for metric,saved in summary['objectives'][objective]['metrics'].items(): compare_across(saved,[allruns[f'{objective}_seed{s}']['means'][metric] for s in range(3)],objective+' '+metric)
        for metric,saved in summary['objectives'][objective]['particle_weighted_error_and_benefit'].items(): compare_across(saved,[allruns[f'{objective}_seed{s}']['weighted'][metric] for s in range(3)],objective+' weighted '+metric)
    derived={}
    for label,comparison in summary['paired_comparisons'].items():
        if label=='nll-minus-faithful': leftobj,rightobj='nll','faithful'; method=None
        else:
            leftobj,method=label.split(':',1); rightobj=leftobj
        derived[label]={}
        for metric,saved in comparison['metrics'].items():
            if method is None: lm=rm=metric
            elif method=='current-minus-previous-risk-correlation': lm=f'correlation/current_base_risk_vs_{metric}'; rm=f'correlation/previous_risk_vs_{metric}'
            else:
                left,right=method.split('-minus-',1); prefix,field=metric.split('/',1); lm=f'{prefix}/{left}/{field}';rm=f'{prefix}/{right}/{field}'
            values=[]
            for seed in range(3):
                l=allruns[f'{leftobj}_seed{seed}']['rows']; r=allruns[f'{rightobj}_seed{seed}']['rows']
                differences=[{'source':x['source'],'metrics':{'delta':x['metrics'][lm]-y['metrics'][rm]}} for x,y in zip(l,r)]
                check([(x['source'],x['target']) for x in l]==[(x['source'],x['target']) for x in r],label+' paired keys')
                values.append(eqtraj(differences,'delta'))
            compare_across(saved,values,label+' '+metric); derived[label][metric]=across(values)
    geo_rows=allruns['faithful_seed0']['rows']
    graph_summary={field:{'mean':mean([r['graph'][field] for r in geo_rows]),'min':min(r['graph'][field] for r in geo_rows),'max':max(r['graph'][field] for r in geo_rows)} for field in ('base_pairs','annulus_pairs','candidate_pairs','optional_budget')}
    graph_summary['float32_differing_frames']=[{'source_index':r['source'],'target_frame':r['target'],**r['graph']['frozen_float32_comparison']} for r in geo_rows if r['graph']['frozen_float32_comparison']['base_symmetric_difference'] or r['graph']['frozen_float32_comparison']['annulus_symmetric_difference']]
    graph_summary['mean_budgeted_retained_pairs']=mean([r['graph']['base_pairs']+r['graph']['optional_budget'] for r in geo_rows])
    graph_summary['mean_budgeted_fraction_of_dense_pairs']=mean([(r['graph']['base_pairs']+r['graph']['optional_budget'])/r['graph']['candidate_pairs'] for r in geo_rows])
    run_report={}
    for name,v in allruns.items():
        rs=v['rows']; ms=v['means']; risk=POLICIES[-1]
        run_report[name]={k:v[k] for k in ('means','weighted','runtime','software','threads','result_sha256')}
        run_report[name]['extra_descriptive']={'negative_mean_dense_benefit_frames':sum(r['metrics']['benefit/mean_normalized_vector_benefit']<0 for r in rs),'position_normalized_particle_benefit_sign_disagreements':sum(r['sign_disagreements'] for r in rs),'particles_across_frames':sum(r['n'] for r in rs),'boundary_tie_frame_counts':{p:sum(r['graph']['policies'][p]['cutoff_ties']['boundary_tie'] for r in rs) for p in ('speed25',risk)},'mean_optional_jaccard':{label:mean([r['graph']['overlaps'][label]['optional_jaccard'] for r in rs]) for label in ('random25__speed25','random25__'+risk,'speed25__'+risk)},'risk_vs_base_runtime_ratio':ms[f'timing/{risk}/end_to_end_seconds']/ms['timing/base/end_to_end_seconds'],'risk_score_fraction_of_end_to_end':ms[f'timing/{risk}/score_generation_seconds']/ms[f'timing/{risk}/end_to_end_seconds'],'policy_position_mse_percent_change_vs_base':{p:100*(ms[f'accuracy/{p}/position_coordinate_mse']/ms['accuracy/base/position_coordinate_mse']-1) for p in POLICIES[1:]}}
    for path,expected in HASHES.items(): check(digest(ROOT/path)==expected,'input unchanged '+path)
    return {'schema':1,'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':CHECKS,'scope':'All 1782 saved diagnostic arrays and raw calls; 297 common observed states; independent array arithmetic and aggregation, plus strict loader provenance; no inference','counts':counts,'strict_loader_runs':strict_runs,'graph_summary':graph_summary,'maximum_recorded_repeated_prediction_difference':repeat_max,'maximum_recorded_natural_shared_base_prediction_difference':natural_shared_max,'per_run':run_report,'paired_comparisons':derived,'input_sha256':HASHES,'limitations':['Teacher-forced previous-observed-base risk is not autonomous own-graph cached risk.','No new inference was run; repetition prediction differences are recorded summaries, not independently recomputed without per-repeat prediction arrays.','All policies execute the variance head; optimized mean-only baseline is unmeasured.','Repeated runtime calls are nested within observed frames; training-model replicate count is three, and host was shared.','Same-state graph construction uses float64 classification; recorded float32 graph differences are preserved.','Whole dense intervention is not marginal edge value; normalized and position benefit signs can differ.']}

if __name__=='__main__':
    try:
        result=run()
    except BaseException as e:
        result={'schema':1,'status':'failed','verified_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':CHECKS,'error':f'{type(e).__name__}: {e}','input_sha256':HASHES}
        OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        raise
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_passed','counts')},indent=2))
