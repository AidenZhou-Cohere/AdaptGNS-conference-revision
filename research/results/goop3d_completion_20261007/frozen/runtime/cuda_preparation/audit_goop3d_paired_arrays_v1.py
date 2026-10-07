#!/usr/bin/env python3
"""Independent paired D3 H295 arithmetic over one SHA-bound saved-array receipt.

No model, original evaluator/aggregator, source arrays or result queue is read.
All source and training-seed denominators remain fixed; missing values propagate.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import traceback
import audit_goop3d_saved_arrays_v1 as saved
sys.dont_write_bytecode=True


def seed_stats(values):
    if len(values)!=3: raise ValueError('Exactly three ordered seed values required')
    complete=all(saved.finite(v) for v in values)
    average=math.fsum(values)/3 if complete else None
    return {'seed_values':{str(i):v for i,v in enumerate(values)},'required_seed_pairs':3,
            'defined_seed_pairs':sum(saved.finite(v) for v in values),'mean':average,
            'sample_sd':math.sqrt(math.fsum((v-average)**2 for v in values)/2) if complete else None}


def subtract(left,right):
    return left-right if saved.finite(left) and saved.finite(right) else None


def paired_metrics(values,policies,risk):
    names=sorted({name for metrics in values.values() for name in metrics})
    result={k:{} for k in ('absolute','mix_minus_base_training','within_arm_policy_contrasts','risk_minus_random_mix_minus_base_interaction')}
    for name in names:
        def get(arm,seed,policy):return values[arm,seed,policy].get(name)
        result['absolute'][name]={arm:{policy:seed_stats([get(arm,seed,policy) for seed in range(3)]) for policy in policies} for arm in ('base','mix')}
        result['mix_minus_base_training'][name]={policy:seed_stats([subtract(get('mix',seed,policy),get('base',seed,policy)) for seed in range(3)]) for policy in policies}
        result['within_arm_policy_contrasts'][name]={arm:{policy+'_minus_'+ref:seed_stats([subtract(get(arm,seed,policy),get(arm,seed,ref)) for seed in range(3)]) for ref in ('base','random25','speed25','relative-velocity-RMS25') for policy in policies if policy!=ref} for arm in ('base','mix')}
        result['risk_minus_random_mix_minus_base_interaction'][name]=seed_stats([subtract(subtract(get('mix',seed,risk),get('mix',seed,'random25')),subtract(get('base',seed,risk),get('base',seed,'random25'))) for seed in range(3)])
    result['interaction_formula']='(mix-trained risk - mix-trained random) - (base-trained risk - base-trained random)'
    return result


def verify(audit,summary,checks):
    checks.equal((audit['schema'],audit['status'],audit['dimension'],audit['horizon'],audit['endpoint_updates']),('goop3d_saved_array_audit_v1','passed_supported_checks',3,295,25000),'D3 passed saved audit')
    checks.equal((summary['schema'],summary['status'],summary['dataset'],summary['endpoint_updates']),('adaptgns_goop3d_paired_scalar_summary_v1','fixed_scalar_aggregation_complete','Goop-3D',25000),'D3 paired summary')
    for k in ('collection_sha256','cohort_sha256','physical_reference','source_manifest_sha256'):checks.equal(summary[k],audit[k],'same original '+k)
    checks.equal(summary['summarizer_sha256'],saved.COLLECTOR_SHA,'frozen scalar summarizer')
    checks.equal(audit['auditor_sha256'],saved.file_hash(saved.__file__),'saved auditor current bytes')
    checks.equal(audit['diagnostic_helper_sha256'],saved.file_hash(saved.diagnostic.__file__),'saved helper current bytes')
    grid={(a,s,n) for a,s in saved.MODELS for n,_,_ in saved.STAGES}
    models=audit['models'];checks.equal(len(models),30,'all30 audited model stages')
    checks.equal({(r['arm'],r['seed'],r['stage']) for r in models},grid,'exact D3 model-stage grid')
    schedules,_=saved.schedules_from_collection(audit,checks)
    checks.equal(set(summary['stages']),{n for n,_,_ in saved.STAGES},'all five paired stages')
    accounting={f"{r['arm']}_seed{r['seed']}_{r['stage']}":r['cells'] for r in models}
    checks.equal(summary['required_cell_accounting'],accounting,'exact all30 cell states and reasons')
    outputs={}
    for name,mode,split in saved.STAGES:
        checks.context=name;selected=[r for r in models if r['stage']==name];by={(r['arm'],r['seed']):r for r in selected};actual=summary['stages'][name]
        wanted=saved.expected_cells(schedules[split][mode],mode)
        for model in selected:
            checks.equal((model['mode'],model['split']),(mode,split),'audit stage mode/split')
            checks.equal(len(model['cells']),len(wanted),'full stage denominator')
            checks.require(all(all(c.get(k)==v for k,v in w.items()) and c.get('state') in saved.CELL_STATES for c,w in zip(model['cells'],wanted)),'audited fixed cell identities')
            checks.equal(model['coverage'],dict(Counter(c['state'] for c in model['cells'])),'audited cell counts')
        coverage={f"{r['arm']}_seed{r['seed']}":r['coverage'] for r in selected}
        checks.equal(actual['coverage_by_model'],coverage,'coverage per model')
        if mode=='full-rollout':
            counts=dict(Counter(c['state'] for r in selected for c in r['cells']))
            checks.equal(actual['coverage'],counts,'all failure and timeout categories')
            checks.equal(actual['required_outcomes'],6*len(wanted),'full required outcomes')
            checks.equal(actual['all_required_outcomes_complete'],counts.get('completed_required_outcome',0)==6*len(wanted),'full completion status')
            checks.equal(actual['failure_categories_by_model'],{f"{r['arm']}_seed{r['seed']}":r['failure_categories'] for r in selected},'failure category retention')
            values={}
            for (arm,seed),model in by.items():
                checks.equal(set(model['metrics']),{k+'/'+p for k in saved.FULL_METRICS for p in saved.POLICIES},'fixed full metric grid')
                for p in saved.POLICIES:values[arm,seed,p]={k:model['metrics'][k+'/'+p] for k in saved.FULL_METRICS}
            recomputed=paired_metrics(values,saved.POLICIES,'laggedrisk25');checks.close(actual['paired'],recomputed,'paired full arrays/series means')
            # The saved auditor has independently checked accepted-prefix scalar
            # arithmetic. Compare those values and preserve every failure record.
            prefixes=[{'arm':r['arm'],'seed':r['seed'],'source_index':r['unit'][0],'policy':r['unit'][1],
                       'completed_steps':r['completed_steps'],'failure':r['failure'],'accepted_prefix':r['accepted_prefix_boundary']}
                      for r in audit['row_checks'] if r['stage']==name and r['status']=='failed']
            checks.close(actual['failed_accepted_prefixes'],prefixes,'all failed accepted prefixes')
            outputs[name]={'paired':recomputed,'coverage':counts}
        else:
            keys=saved.diagnostic_keys(mode)
            for model in selected:
                checks.equal(set(model['metrics']),set(keys),'fixed diagnostic metric grid')
                checks.equal(set(model['aggregates']),set(keys),'fixed diagnostic aggregate grid')
                modelname=f"{model['arm']}_seed{model['seed']}"
                for key in keys:
                    computed=model['aggregates'][key]
                    checks.close(model['metrics'][key],computed['equal_trajectory_mean'],'audited source mean')
                    tree=actual['model_summaries'][modelname] if key in saved.diagnostic.expected_metric_keys(mode) else actual['augmented_graph_and_boundary'][modelname]
                    leaf=saved.tree_leaf(tree,key);checks.close({k:leaf[k] for k in computed},computed,'complete diagnostic hierarchy '+key)
            absolute={arm:{k:seed_stats([by[arm,s]['metrics'][k] for s in range(3)]) for k in sorted(keys)} for arm in ('base','mix')}
            change={k:seed_stats([subtract(by['mix',s]['metrics'][k],by['base',s]['metrics'][k]) for s in range(3)]) for k in sorted(keys)}
            checks.close(actual['absolute'],absolute,'diagnostic absolute mean and SD')
            checks.close(actual['mix_minus_base_training'],change,'diagnostic paired training changes')
            outputs[name]={'absolute':absolute,'mix_minus_base_training':change,'coverage_by_model':coverage}
            if mode=='same-state':
                values={(a,s,p):{k:by[a,s]['metrics']['accuracy/'+p+'/'+k] for k in ('position_coordinate_mse','normalized_coordinate_mse')} for a,s in saved.MODELS for p in saved.diagnostic.POLICIES}
                recomputed=paired_metrics(values,saved.diagnostic.POLICIES,'previous-observed-base-risk25')
                checks.close(actual['accuracy_policy_contrasts'],recomputed,'diagnostic policy and interaction contrasts')
                states=[]
                for seed in range(3):
                    for item in schedules[split][mode]:
                        ident=str((item['source_index'],item['target_frame']))
                        for policy in saved.diagnostic.POLICIES[:-1]:
                            complete=all(policy in by[a,seed]['policy_completion'].get(ident,[]) for a in ('base','mix'))
                            states.append({'seed':seed,'source_index':item['source_index'],'target_frame':item['target_frame'],'policy':policy,'state':'same_graph_and_draw_verified' if complete else 'required_policy_missing_or_failed'})
                checks.equal(actual['paired_observed_graph_selection'],states,'paired available graph identity states')
                outputs[name]['accuracy_policy_contrasts']=recomputed
    return outputs


def audit_pair(audit_path,audit_sha,summary_path,summary_sha):
    checks=saved.Checks();audit=saved.load_json(audit_path,audit_sha);summary=saved.load_json(summary_path,summary_sha)
    sources={str(Path(__file__).resolve()):saved.file_hash(__file__),str(Path(saved.__file__).resolve()):saved.file_hash(saved.__file__),str(Path(saved.diagnostic.__file__).resolve()):saved.file_hash(saved.diagnostic.__file__)}
    results=verify(audit,summary,checks)
    for p,h in {**sources,str(audit_path):audit_sha,str(summary_path):summary_sha}.items():checks.equal(saved.file_hash(p),h,'input/source unchanged')
    return {'schema':'goop3d_paired_array_audit_v1','status':'passed_supported_checks','dimension':3,'horizon':295,'endpoint_updates':25000,'checks':checks.count,
            'collection_sha256':audit['collection_sha256'],'saved_array_audit_sha256':audit_sha,'paired_summary_sha256':summary_sha,'source_sha256':sources,'stages':results,
            'scope':{'verified':'All five stages, every fixed source/cell and three seed values; independent equal-source means, sample SD, training and policy contrasts, risk/random interaction, available paired graph states and accepted failure prefixes.',
                     'inherited':'Only supported saved-array and recorded scalar-series arithmetic from the exact passed D3 saved-array audit.',
                     'unsupported':['No fresh model, source trajectory, spatial search, parity inference or unsaved rollout replay.','No independent runtime truth, hardware isolation or causal speedup.','Ground-truth boundary summary fields retain the original scalar collector provenance; this paired verifier does not separately re-derive them.'],
                     'all_missing_failed_timeout_unexecuted_cells_retained':True,'survivor_means_computed':False}}


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for name in ('audit','summary','output'):p.add_argument('--'+name,type=Path)
    for name in ('audit-sha256','summary-sha256'):p.add_argument('--'+name)
    a=p.parse_args()
    if not a.execute:print(json.dumps({'status':'description_only','dimension':3,'horizon':295,'stages':5,'seed_pairs':3}));return 0
    if not all((a.audit,a.audit_sha256,a.summary,a.summary_sha256,a.output)):p.error('Exact input hashes and fresh output required')
    if a.output.exists():p.error('Fresh output required')
    try:
        result=audit_pair(a.audit,a.audit_sha256,a.summary,a.summary_sha256)
        raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        for path,h in {**result['source_sha256'],str(a.audit):a.audit_sha256,str(a.summary):a.summary_sha256}.items():
            if saved.file_hash(path)!=h:raise ValueError('Input changed before publication: '+path)
    except Exception as error:
        with a.output.open('x') as f:json.dump({'schema':'goop3d_paired_array_audit_v1','status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()},f,indent=2)
        raise
    with a.output.open('xb') as f:f.write(raw)
    print(json.dumps({'status':result['status'],'checks':result['checks']}));return 0

if __name__=='__main__':raise SystemExit(main())
