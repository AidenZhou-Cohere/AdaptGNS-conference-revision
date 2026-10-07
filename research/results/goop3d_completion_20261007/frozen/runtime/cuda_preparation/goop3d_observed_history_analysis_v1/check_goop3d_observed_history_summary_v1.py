"""Independent stdlib arithmetic reconstruction from audited per-row values."""
import argparse
from collections import Counter
import math
from pathlib import Path
from observed_common import *

def valid(x):return type(x)in(int,float) and math.isfinite(x)
def average(items):return math.fsum(items)/len(items) if items and all(valid(x) for x in items) else None
def difference(x,y):return None if not valid(x) or not valid(y) else x-y
def seed_values(items):
    need(len(items)==3,'three required seed values');m=average(items)
    return {'seed_values':{str(s):items[s] for s in range(3)},'required_seed_pairs':3,'defined_seed_pairs':sum(valid(x) for x in items),'mean':m,'sample_sd':None if m is None else math.sqrt(math.fsum((x-m)**2 for x in items)/2)}
def equal(actual,expected):
    if isinstance(expected,dict):
        need(type(actual)is dict and set(actual)==set(expected),'exact arithmetic object keys')
        for key in expected:equal(actual[key],expected[key])
    elif isinstance(expected,list):
        need(type(actual)is list and len(actual)==len(expected),'exact arithmetic list size')
        for x,y in zip(actual,expected):equal(x,y)
    elif expected is None or isinstance(expected,(str,bool)):need(actual==expected,'exact arithmetic label/null')
    else:need(valid(actual) and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),'independent arithmetic differs')

def verify(audit,summary):
    need(audit['schema']==AUDIT_SCHEMA and audit['status']=='passed_scoped_observed_history_checks','distinct passing audit required')
    need(summary['schema']==SUMMARY_SCHEMA and summary['status']=='passed_scoped_scalar_aggregation','distinct completed summary required')
    for k,v in [('collection_sha256',COLLECTION_SHA),('cohort_sha256',COHORT_SHA),('required_all_cells',4728),('required_observed_cells',2568)]:need(audit[k]==summary[k]==v,'fixed '+k)
    need(audit['phase_sha256']==summary['phase_sha256'],'same new phase')
    equal(summary['all_original_accounting'],audit['all_original_accounting'])
    allcells=audit['all_original_accounting'];need(len(allcells)==30 and sum(len(s['cells']) for s in allcells)==4728,'full original cell accounting')
    equal(dict(Counter(c['state'] for s in allcells for c in s['cells'])),{'completed_required_outcome':2899,'not_completed_before_invocation_end':1817,'timed_out_current':12})
    models=audit['models'];need(len(models)==18 and {(r['arm'],r['seed'],r['stage']) for r in models}=={(a,s,n) for a in ('base','mix') for s in range(3) for n in OBSERVED},'all six models all observed families')
    recomputed={};checks=0
    for model in models:
        name=model['stage'];split=model['split'];mode=model['mode'];schedule=audit['source_schedules'][split][mode]
        expected=[(r['source_index'],r['target_frame']) for r in schedule]
        need(len(expected)==(128 if name=='clean_validation' else 150) and len(set(expected))==len(expected),'exact observed schedule cardinality')
        cells=model['cells'];need(len(cells)==len(expected) and [(c['source_index'],c['target_frame']) for c in cells]==expected,'exact declared observed cells')
        rows={tuple(r['unit']):r for r in model['rows']};need(len(rows)==len(model['rows']),'unique audited row identities')
        committed={(c['source_index'],c['target_frame']) for c in cells if c['state'] in ('completed_required_outcome','recorded_failed_outcome')}
        need(set(rows)==committed,'audited row presence equals original cell accounting')
        equal(model['coverage'],dict(Counter(c['state'] for c in cells)))
        per={}
        for metric in model['aggregates']:
            trajectories={}
            for source in sorted({u[0] for u in expected}):
                entries=[rows.get(u,{}).get('metrics',{}).get(metric) for u in expected if u[0]==source]
                trajectories[str(source)]={'expected':len(entries),'defined':sum(valid(x) for x in entries),'mean':average(entries)}
            agg={'expected_frames':len(expected),'defined_frames':sum(x['defined'] for x in trajectories.values()),'expected_trajectories':len(trajectories),'defined_trajectories':sum(x['mean'] is not None for x in trajectories.values()),'equal_trajectory_mean':average([x['mean'] for x in trajectories.values()]),'trajectories':trajectories}
            equal(model['aggregates'][metric],agg);per[metric]=agg['equal_trajectory_mean'];checks+=1
        recomputed[model['arm'],model['seed'],name]=per
    need(set(summary['families'])==set(OBSERVED),'exact summary family keys')
    policies=('base','dense','random25','speed25','relative-velocity-RMS25','previous-observed-base-risk25')
    for name in OBSERVED:
        group=[m for m in models if m['stage']==name];complete=all(c['state']=='completed_required_outcome' for m in group for c in m['cells']);actual=summary['families'][name]
        need(actual['full_family_complete']==complete and actual['required_cells']==sum(len(m['cells']) for m in group),'full family completeness and denominator')
        equal(actual['coverage_by_model'],{f"{m['arm']}_seed{m['seed']}":m['coverage'] for m in group})
        keys=set(group[0]['aggregates']);need(all(set(m['aggregates'])==keys for m in group),'common fixed metric keys')
        def x(arm,seed,key):return recomputed[arm,seed,name][key] if complete else None
        absolute={arm:{key:seed_values([x(arm,s,key) for s in range(3)]) for key in sorted(keys)} for arm in ('base','mix')}
        changes={key:seed_values([difference(x('mix',s,key),x('base',s,key)) for s in range(3)]) for key in sorted(keys)}
        equal(actual['absolute'],absolute);equal(actual['mix_minus_base_training'],changes)
        if name!='clean_validation':
            expected_contrasts={}
            for metric in ('position_coordinate_mse','normalized_coordinate_mse'):
                def loss(arm,seed,p):return x(arm,seed,'accuracy/'+p+'/'+metric)
                within={arm:{p+'_minus_'+ref:seed_values([difference(loss(arm,s,p),loss(arm,s,ref)) for s in range(3)]) for ref in ('base','random25','speed25','relative-velocity-RMS25') for p in policies if p!=ref} for arm in ('base','mix')}
                interaction=[]
                for s in range(3):
                    mixed=difference(loss('mix',s,policies[-1]),loss('mix',s,'random25'));base=difference(loss('base',s,policies[-1]),loss('base',s,'random25'));interaction.append(difference(mixed,base))
                expected_contrasts[metric]={'within_arm':within,'risk_minus_random_mix_minus_base':seed_values(interaction)}
            equal(actual['accuracy_policy_contrasts'],expected_contrasts)
    return {'schema':CHECK_SCHEMA,'status':'passed_independent_observed_arithmetic','checked_model_metric_aggregates':checks,'required_all_cells':4728,'required_observed_cells':2568,'collection_sha256':COLLECTION_SHA,'cohort_sha256':COHORT_SHA,'phase_sha256':audit['phase_sha256'],'scientific_admission':False,'model_array_or_original_result_queue_read':False,'survivor_means_computed':False}

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for n in ('audit','summary','output','phase'):p.add_argument('--'+n,type=Path)
    for n in ('audit-sha256','summary-sha256','phase-sha256'):p.add_argument('--'+n)
    args=p.parse_args()
    if not args.execute:print(json.dumps({'status':'inert_independent_observed_checker'}));return
    need(str(args.audit)==OUTPUT+'/audit.json' and str(args.summary)==OUTPUT+'/summary.json' and str(args.output)==OUTPUT+'/arithmetic_check.json' and not args.output.exists(),'exact fresh scoped check paths')
    phase=load_phase(args.phase,args.phase_sha256);budget=WorkerBudget(phase,2940)
    audit=strict(read_bound(args.audit,args.audit_sha256,CAP_OUTPUT,budget.check));summary=strict(read_bound(args.summary,args.summary_sha256,CAP_OUTPUT,budget.check))
    need(audit['phase_sha256']==args.phase_sha256 and summary['audit_sha256']==args.audit_sha256,'exact same audit and phase')
    result=verify(audit,summary);result.update(audit_sha256=args.audit_sha256,summary_sha256=args.summary_sha256)
    for path,pin in [(args.audit,args.audit_sha256),(args.summary,args.summary_sha256)]:read_bound(path,pin,CAP_OUTPUT,budget.check,retain=False)
    print(json.dumps({'status':result['status'],'sha256':publish(args.output,result,budget.check)}))

if __name__=='__main__':main()
