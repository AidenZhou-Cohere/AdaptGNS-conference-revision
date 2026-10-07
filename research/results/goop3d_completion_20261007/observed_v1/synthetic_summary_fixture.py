import observed_common as c
import audit_goop3d_observed_histories_v1 as audit
import goop3d_observed_history_arithmetic_v1 as arithmetic
import summarize_goop3d_observed_histories_v1 as summarize

def synthetic_audit(incomplete=False):
    same=[{'source_index':i,'target_frame':t} for i in range(30) for t in (7,80,153,226,300)]
    clean=[{'source_index':j//5,'target_frame':6+j%5} for j in range(128)]
    schedules={s:{'same-state':same,'clean-validation':clean} for s in ('valid','test')}
    models=[];all_accounting=[]
    policies=summarize.POLICIES
    for arm in ('base','mix'):
        for seed in range(3):
            for stage in c.OBSERVED:
                mode='clean-validation' if stage=='clean_validation' else 'same-state';split='test' if stage=='same_state_test' else 'valid'
                schedule=schedules[split][mode];cells=[];rows=[]
                keys=['metrics/test_metric'] if mode=='clean-validation' else ['accuracy/'+p+'/'+m for p in policies for m in ('position_coordinate_mse','normalized_coordinate_mse')]
                for n,item in enumerate(schedule):
                    missing=incomplete and (arm,seed,stage,n)==('base',0,'same_state_valid',0)
                    cells.append({**item,'state':'not_completed_before_invocation_end' if missing else 'completed_required_outcome'})
                    if missing:continue
                    metrics={}
                    for key in keys:
                        v=(2 if arm=='mix' else 0)+seed+item['source_index']/1000+item['target_frame']/10000
                        if mode=='same-state':
                            policy=key.split('/')[1];v+=policies.index(policy)/100
                            if policy==policies[-1] and arm=='mix':v+=.5
                        metrics[key]=v
                    rows.append({'unit':[item['source_index'],item['target_frame']],'status':'complete','metrics':metrics})
                values={tuple(r['unit']):r['metrics'] for r in rows}
                aggregates={k:arithmetic.aggregate([(x['source_index'],x['target_frame']) for x in schedule],{u:m[k] for u,m in values.items()}) for k in keys}
                model={'arm':arm,'seed':seed,'stage':stage,'mode':mode,'split':split,'cells':cells,'coverage':dict(audit.Counter(x['state'] for x in cells)),'rows':rows,'aggregates':aggregates}
                models.append(model);all_accounting.append({'arm':arm,'seed':seed,'stage':stage,'cells':cells})
    states=['completed_required_outcome']*(331+int(incomplete))+['timed_out_current']*12+['not_completed_before_invocation_end']*(1817-int(incomplete))
    for k in range(12):all_accounting.append({'arm':'base' if k<6 else 'mix','seed':(k//2)%3,'stage':'full_rollout_valid' if k%2==0 else 'full_rollout_test','cells':[{'state':x} for x in states[k*180:(k+1)*180]]})
    return {'schema':c.AUDIT_SCHEMA,'status':'passed_scoped_observed_history_checks','collection_sha256':c.COLLECTION_SHA,'cohort_sha256':c.COHORT_SHA,'phase_sha256':'b'*64,'required_all_cells':4728,'required_observed_cells':2568,'source_schedules':schedules,'models':models,'all_original_accounting':all_accounting,'limitations':[]}
