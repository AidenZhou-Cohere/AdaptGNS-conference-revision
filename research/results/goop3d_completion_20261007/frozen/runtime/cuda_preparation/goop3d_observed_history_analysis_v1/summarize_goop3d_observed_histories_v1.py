"""Fixed paired aggregation of the distinct observed-history audit receipt."""
import argparse
import math
from pathlib import Path
import statistics
from observed_common import *

POLICIES=('base','dense','random25','speed25','relative-velocity-RMS25','previous-observed-base-risk25')
RISK=POLICIES[-1]
def finite(x):return type(x)in(int,float) and math.isfinite(x)
def delta(x,y):return x-y if finite(x) and finite(y) else None
def stats(values):
    need(len(values)==3,'all three ordered seed values required');ok=all(finite(x) for x in values)
    return {'seed_values':dict(zip(('0','1','2'),values)),'required_seed_pairs':3,'defined_seed_pairs':sum(finite(x) for x in values),'mean':statistics.fmean(values) if ok else None,'sample_sd':statistics.stdev(values) if ok else None}

def summarize(audit):
    need(audit['schema']==AUDIT_SCHEMA and audit['status']=='passed_scoped_observed_history_checks','passing distinct observed audit required')
    need(audit['collection_sha256']==COLLECTION_SHA and audit['cohort_sha256']==COHORT_SHA and audit['required_all_cells']==4728 and audit['required_observed_cells']==2568,'exact observed cohort and denominator required')
    models=audit['models'];need(len(models)==18,'all18 observed models/stages')
    need({(r['arm'],r['seed'],r['stage']) for r in models}=={(a,s,n) for a in ('base','mix') for s in range(3) for n in OBSERVED},'exact observed stage grid')
    families={}
    for name in OBSERVED:
        selected=[x for x in models if x['stage']==name];by={(x['arm'],x['seed']):x for x in selected}
        complete=all(c['state']=='completed_required_outcome' for x in selected for c in x['cells'])
        keys=set(selected[0]['aggregates']);need(all(set(x['aggregates'])==keys for x in selected),'fixed metric key grid')
        def value(arm,seed,key):return by[arm,seed]['aggregates'][key]['equal_trajectory_mean'] if complete else None
        absolute={a:{k:stats([value(a,s,k) for s in range(3)]) for k in sorted(keys)} for a in ('base','mix')}
        changes={k:stats([delta(value('mix',s,k),value('base',s,k)) for s in range(3)]) for k in sorted(keys)}
        row={'full_family_complete':complete,'required_cells':sum(len(x['cells']) for x in selected),'coverage_by_model':{f"{x['arm']}_seed{x['seed']}":x['coverage'] for x in selected},'absolute':absolute,'mix_minus_base_training':changes}
        if name.startswith('same_state'):
            contrasts={}
            for metric in ('position_coordinate_mse','normalized_coordinate_mse'):
                def v(arm,seed,p):return value(arm,seed,'accuracy/'+p+'/'+metric)
                contrasts[metric]={
                    'within_arm':{a:{p+'_minus_'+ref:stats([delta(v(a,s,p),v(a,s,ref)) for s in range(3)]) for ref in ('base','random25','speed25','relative-velocity-RMS25') for p in POLICIES if p!=ref} for a in ('base','mix')},
                    'risk_minus_random_mix_minus_base':stats([delta(delta(v('mix',s,RISK),v('mix',s,'random25')),delta(v('base',s,RISK),v('base',s,'random25'))) for s in range(3)])}
            row['accuracy_policy_contrasts']=contrasts
        families[name]=row
    return {'schema':SUMMARY_SCHEMA,'status':'passed_scoped_scalar_aggregation','collection_sha256':COLLECTION_SHA,'cohort_sha256':COHORT_SHA,'phase_sha256':audit['phase_sha256'],'all_original_accounting':audit['all_original_accounting'],'required_all_cells':4728,'required_observed_cells':2568,'families':families,'scientific_admission':False,'original_full_audit_rerun':False,'limitations':audit['limitations']+['Entire incomplete families have null full-family comparisons; no survivor means.','Three paired training seeds; no p-values or independent-frame interpretation.']}

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for n in ('audit','output','phase'):p.add_argument('--'+n,type=Path)
    for n in ('audit-sha256','phase-sha256'):p.add_argument('--'+n)
    args=p.parse_args()
    if not args.execute:print(json.dumps({'status':'inert_observed_summary'}));return
    need(str(args.audit)==OUTPUT+'/audit.json' and str(args.output)==OUTPUT+'/summary.json' and not args.output.exists(),'exact fresh scoped summary paths')
    phase=load_phase(args.phase,args.phase_sha256);budget=WorkerBudget(phase,2820)
    audit=strict(read_bound(args.audit,args.audit_sha256,CAP_OUTPUT,budget.check));need(audit['phase_sha256']==args.phase_sha256,'same new phase')
    result=summarize(audit);result['audit_sha256']=args.audit_sha256
    read_bound(args.audit,args.audit_sha256,CAP_OUTPUT,budget.check,retain=False)
    print(json.dumps({'status':result['status'],'sha256':publish(args.output,result,budget.check)}))

if __name__=='__main__':main()
