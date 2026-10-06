#!/usr/bin/env python3
"""Pure prospective cost arithmetic; candidates do not authorize test access.

Input JSON names three complete driver collections, three reaped parent outcomes
and the original train manifest with SHA256s. No models or numeric arrays loaded.
"""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path

SCHEMA='adaptgns_goop_global_action_gate_cost_candidate_v1'
PROTOCOL_SHA='bc9235fae89d32ede8d1e7f846bff07bdcdd85f4671d5922d688a9535f7fe024'
CORE_SHA='d3986c2ac3786e9fc0d36d76339bfeded5448497f7dad9b0559efaaaa53a43ca'
TRAINING_PROTOCOL_SHA='c8690d0da209c66557e3dbb0cbccd55d660b4cc270683913f74d381516591851'
POLICIES=('base','random25','learned_global_gate','validation_rate_random_gate')
TARGETS=(6,62,118,174,231,287,343,400)
DEADLINE=datetime(2026,10,7,4,tzinfo=timezone.utc)
MARGIN=1.35
CLEANUP=15


def require(value,message):
    if not value:raise ValueError(message)


def finite(value,positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)


def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def stamp(value):
    d=datetime.fromisoformat(value);require(d.tzinfo is not None,'Timezone-aware clock required');return d.astimezone(timezone.utc)


def schedule(mode,manifest):
    records=manifest.get('records',[])
    require(manifest.get('split')=='train' and manifest.get('record_count')==len(records)==1000
        and all(r.get('source_index')==i and type(r['positions']['shape'][1]) is int and r['positions']['shape'][1]>0 for i,r in enumerate(records)),
        'Complete ordered1000-source training manifest required')
    if mode=='train-label-capacity':
        flat=[(source,target) for source in range(1000) for target in TARGETS]
        return [flat[i*7999//63] for i in range(64)]
    require(mode=='train-rollout-capacity','Only the two fixed capacity phases are arithmetic inputs')
    ordered=sorted(range(1000),key=lambda i:(records[i]['positions']['shape'][1],i))
    return [(ordered[i],policy) for i in (0,499,999) for policy in POLICIES]


def checked_measurement(collection,outcome,seed,mode,expected,bindings):
    kind='label' if mode=='train-label-capacity' else 'rollout'
    require(collection.get('schema')==f'adaptgns_goop_global_action_gate_{kind}_collection_v1'
        and collection.get('status')=='complete' and collection.get('mode')==mode and collection.get('split')=='train'
        and collection.get('gate_protocol_sha256')==PROTOCOL_SHA and collection.get('core_sha256')==CORE_SHA
        and collection.get('original_training_protocol_sha256')==TRAINING_PROTOCOL_SHA
        and collection.get('driver_sha256')==bindings['driver_sha256']
        and collection.get('cohort_sha256')==bindings['cohort_sha256']
        and collection.get('source_manifest_sha256')==bindings['train_manifest_sha256']
        and collection.get('all_inputs_reverified') is True and collection.get('model_state_verified_unchanged') is True
        and collection.get('abort_reason') is None and collection.get('all_required_outcomes_complete') is True,
        'Complete source-bound train-only capacity collection required')
    model=collection.get('model',{})
    require(model.get('seed')==seed and type(model.get('seed')) is int and model.get('arm')=='mix'
        and model.get('objective')=='faithful' and model.get('completed_updates')==100000 and digest(model.get('checkpoint_sha256')),
        'Exact fixed mix100k model required')
    rows=collection.get('rows',[]);key='target_frame' if kind=='label' else 'policy'
    require(len(rows)==len(expected) and [(r.get('source_index'),r.get(key)) for r in rows]==expected
        and collection.get('required_rows')==collection.get('committed_rows')==collection.get('complete_rows')==len(expected)
        and all(r.get('status')=='complete' for r in rows),'Every fixed capacity outcome must be complete, without guard failures')
    if kind=='rollout':
        require(collection.get('selection_sha256')==bindings.get('selection_sha256') and digest(collection.get('head_sha256'))
            and all(r.get('selection_sha256')==bindings['selection_sha256'] and r.get('head_sha256')==collection['head_sha256'] for r in rows),
            'Capacity policies must use the final frozen head/common selection')
    timings=collection.get('case_timings',[])
    require(len(timings)==len(expected) and [(r.get('source_index'),r.get(key)) for r in timings]==expected
        and all(finite(r.get('wall_seconds'),True) for r in timings),'Every ordered case needs complete measured publication-inclusive wall time')
    setup=collection.get('runtime',{}).get('setup_seconds')
    require(finite(setup) and outcome.get('started') is True and outcome.get('stopped_and_reaped') is True
        and outcome.get('exit_code')==0 and outcome.get('signals')==[] and outcome.get('quota_expired_at_observation') is False
        and outcome.get('inner_timeout_reported') is False and outcome.get('worker_failed_attempt') is None
        and finite(outcome.get('elapsed_seconds'),True),'Complete measured reaped parent invocation required')
    command=outcome.get('command',[])
    require(command.count('--seed')==1 and command[command.index('--seed')+1]==str(seed)
        and command.count('--mode')==1 and command[command.index('--mode')+1]==mode
        and command.count('--checkpoint-sha256')==1 and command[command.index('--checkpoint-sha256')+1]==model['checkpoint_sha256']
        and command.count('--cuda-index')==1 and command[command.index('--cuda-index')+1]==str(1 if seed==1 else 0),
        'Parent observation/model/mode/owned-GPU identity differs')
    measured=sum(t['wall_seconds'] for t in timings);elapsed=outcome['elapsed_seconds']
    require(elapsed+1e-6>=setup+measured,'Parent wall duration cannot omit setup or serial cases')
    overhead=max(0.,elapsed-setup-measured)
    return model,timings,{'setup_seconds':setup,'measured_cases_seconds':measured,
        'parent_whole_invocation_seconds':elapsed,'finalization_and_other_parent_overhead_seconds':overhead}


def build_plan(bundle,collections,outcomes,manifest,now=None,completed_receipts=None):
    now=datetime.now(timezone.utc) if now is None else now
    mode=bundle.get('mode');expected=schedule(mode,manifest)
    require(set(collections)==set(outcomes)=={0,1,2} and bundle.get('protocol_sha256')==PROTOCOL_SHA
        and all(digest(bundle.get(k)) for k in ('driver_sha256','cohort_sha256','train_manifest_sha256')),
        'All three fixed seed observations and lineage hashes required')
    models=[];plans=[]
    for seed in range(3):
        model,timings,measured=checked_measurement(collections[seed],outcomes[seed],seed,mode,expected,bundle);models.append(model)
        if mode=='train-label-capacity':
            slow=max(t['wall_seconds'] for t in timings)
            amounts={label:measured['setup_seconds']+MARGIN*n*slow+measured['finalization_and_other_parent_overhead_seconds']
                for label,n in [('train_labels',8000),('validation_labels',150)]}
            plans.append({'seed':seed,'checkpoint_sha256':model['checkpoint_sha256'],**measured,'slowest_paired_state_wall_seconds':slow,
                'whole_invocation_allocations_seconds':{k:math.ceil(v) for k,v in amounts.items()},'unrounded_allocations_seconds':amounts})
        else:
            worst={p:max(t['wall_seconds'] for t in timings if t['policy']==p) for p in POLICIES}
            amount=measured['setup_seconds']+MARGIN*30*sum(worst.values())+measured['finalization_and_other_parent_overhead_seconds']
            plans.append({'seed':seed,'checkpoint_sha256':model['checkpoint_sha256'],**measured,'slowest_full_rollout_seconds_per_policy':worst,
                'test_allocation_seconds':math.ceil(amount),'unrounded_test_allocation_seconds':amount})
    require(len({m['checkpoint_sha256'] for m in models})==3,'Three distinct immutable mean models required')
    result={'schema':SCHEMA,'status':'candidate_requires_root_review','issued_by':'cost_calculator_not_root','mode':mode,
        'protocol_sha256':PROTOCOL_SHA,'driver_sha256':bundle['driver_sha256'],'cohort_sha256':bundle['cohort_sha256'],
        'train_manifest_sha256':bundle['train_manifest_sha256'],'margin':MARGIN,'margin_is_engineering_allowance_not_runtime_bound':True,
        'per_seed':plans,'mapping':{'GPU0':[0,2],'GPU1':[1]},'cleanup_seconds_per_invocation':CLEANUP,
        'failed_or_missing_capacity_outcomes_dropped':False,'test_admitted':False,'compute_analysis_deadline_utc':DEADLINE.isoformat()}
    if mode=='train-label-capacity':
        result['concurrent_allocations_seconds']={k:max(plans[0]['whole_invocation_allocations_seconds'][k]+plans[2]['whole_invocation_allocations_seconds'][k]+2*CLEANUP,
            plans[1]['whole_invocation_allocations_seconds'][k]+CLEANUP) for k in ('train_labels','validation_labels')}
        result['complete_test_cost_remains_unmeasured']=True
    else:
        require(digest(bundle.get('selection_sha256')),'Common selected head receipt required')
        result['selection_sha256']=bundle['selection_sha256'];result['all_36_rollouts_complete']=True
        allocation=max(plans[0]['test_allocation_seconds']+plans[2]['test_allocation_seconds']+2*CLEANUP,plans[1]['test_allocation_seconds']+CLEANUP)
        reserves={'input_preflight':900,'scalar_array_collection':2700,'independent_analysis_manuscript':3600}
        spent=bundle.get('completed_outer_stages',{})
        require(isinstance(spent,dict) and set(spent)<={'input_preflight'},'Only already completed input/preflight work may be excluded')
        completed_receipts={} if completed_receipts is None else completed_receipts
        require(set(completed_receipts)==set(spent),'Actual completed-stage receipt bytes must be loaded before reducing reserves')
        for name,reference in spent.items():
            receipt,receipt_sha=completed_receipts[name]
            require(set(reference)=={'path','sha256'} and Path(reference['path']).is_absolute() and reference['sha256']==receipt_sha
                and digest(receipt_sha) and receipt.get('schema')=='adaptgns_goop_action_gate_preflight_completion_v1'
                and receipt.get('status')=='complete' and receipt.get('issued_by')=='root'
                and receipt.get('protocol_sha256')==PROTOCOL_SHA and receipt.get('cohort_sha256')==bundle['cohort_sha256']
                and receipt.get('selection_sha256')==bundle['selection_sha256'] and finite(receipt.get('actual_elapsed_seconds'))
                and stamp(receipt['completed_utc'])<=now,'Completed outside-work charges need bound dated root completion evidence')
        remaining={k:v for k,v in reserves.items() if k not in spent};error=bundle.get('clock_error_bound_seconds')
        require(finite(error) and error<=5,'Bounded root clock uncertainty required')
        latest=DEADLINE-timedelta(seconds=allocation+sum(remaining.values())+error)
        result.update(test_concurrent_allocation_seconds=allocation,outer_reserves_seconds=reserves,completed_outer_stages=spent,
            completed_outer_stage_evidence={k:v[0] for k,v in completed_receipts.items()},remaining_outer_reserves_seconds=remaining,latest_test_start_utc=latest.isoformat(),
            complete_remaining_work_fits=now<=latest,all_train_validation_labels_complete=False,
            complete_fit_collection_review_still_required=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','training_or_test_execution':False,'capacity_failures_admitted':False}));return 0
    require(args.input and args.output and not args.output.exists(),'Explicit source bundle and new candidate output required')
    raw=args.input.read_bytes();bundle=json.loads(raw);bindings={str(args.input):hashlib.sha256(raw).hexdigest(),str(Path(__file__).resolve()):sha(__file__)}
    def load(entry):
        p=Path(entry['path']);data=p.read_bytes();h=hashlib.sha256(data).hexdigest();require(h==entry['sha256'],'Bound capacity evidence bytes differ')
        bindings[str(p)]=h;return json.loads(data)
    manifest=load(bundle['train_manifest']);require(bundle['train_manifest']['sha256']==bundle['train_manifest_sha256'],'Original train manifest identity differs')
    collections={s:load(bundle['collections'][str(s)]) for s in range(3)};outcomes={s:load(bundle['process_outcomes'][str(s)]) for s in range(3)}
    for seed,outcome in outcomes.items():
        command=outcome.get('command',[])
        require(command.count('--output-dir')==1 and Path(command[command.index('--output-dir')+1]).resolve()
            ==Path(bundle['collections'][str(seed)]['path']).resolve().parent,'Capacity collection does not belong to observed child output')
    completed={k:(load(v),v['sha256']) for k,v in bundle.get('completed_outer_stages',{}).items()}
    result=build_plan(bundle,collections,outcomes,manifest,completed_receipts=completed);result['input_sha256']=bindings;ready=encode(result)
    require(all(sha(p)==h for p,h in bindings.items()),'Capacity evidence changed before candidate publication')
    with args.output.open('xb') as f:f.write(ready)
    return 0


if __name__=='__main__':raise SystemExit(main())
