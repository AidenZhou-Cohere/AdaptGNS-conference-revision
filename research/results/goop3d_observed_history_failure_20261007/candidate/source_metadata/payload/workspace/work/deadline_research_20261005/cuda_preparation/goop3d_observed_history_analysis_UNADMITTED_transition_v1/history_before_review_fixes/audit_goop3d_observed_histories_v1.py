"""New scoped driver; never invokes the original saved-array audit."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import resource
import traceback
import zipfile
import numpy as np
import goop3d_observed_history_arithmetic_v1 as a
from observed_common import *

def numeric_archive(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries=z.infolist();names=[v.filename for v in entries]
        need(len(entries)<=4096 and len(names)==len(set(names)),'bounded unique archive members')
        need(all('/' not in n and '\\' not in n and n.endswith('.npy') for n in names),'flat NPY members only')
        need(sum(v.file_size for v in entries)<=CAP_ARCHIVE,'decoded archive cap')
        need(all(not v.flag_bits&1 for v in entries),'unencrypted numeric archive required')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        need(len(archive.files)==len(set(archive.files)),'unique array keys')
        arrays={k:archive[k] for k in archive.files}
    need(all(isinstance(v,np.ndarray) and not v.dtype.hasobject for v in arrays.values()),'numeric arrays only')
    need(sum(v.nbytes for v in arrays.values())<=CAP_ARCHIVE,'decoded array cap')
    return arrays

def accounting(collection,checks):
    checks.equal(collection['schema'],a.COLLECTION_SCHEMA,'accepted collection schema')
    checks.equal(collection['status'],'stopped_outputs_collected','stopped collection required')
    checks.equal(collection['collector_sha256'],a.COLLECTOR_SHA,'original collector')
    checks.equal(collection['endpoint_updates'],25000,'fixed endpoint')
    checks.equal(collection['cohort_sha256'],COHORT_SHA,'fixed six-model cohort')
    checks.equal(collection['source_manifest_sha256'],{'valid':'f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef','test':'08edf3282267c05a427fc264a1a3aa547e062bc9b5ccb8ab7f982cfb3a9fb7cb'},'frozen source manifests')
    ledger=collection['ledger'];checks.equal(ledger['state'],'stopped_all_owned_processes_reaped','original stopped ledger')
    checks.equal(ledger['unreaped_owned_children'],[],'original child reaping')
    checks.equal(ledger['protocol_sha256'],a.PROTOCOL_SHA,'original protocol')
    checks.equal(ledger['root_release_sha256'],'4ecf5617b903f7ae0fa662152c8730fdd68b64ffb717e265d6b4a5ca5809b145','original evaluation release')
    schedules,populations=a.schedules_from_collection(collection,checks)
    checks.equal([len(schedules[s]['full-rollout']) for s in ('valid','test')],[30,30],'fixed two 30-source grids')
    grid={(arm,seed,name) for arm,seed in a.MODELS for name,_,_ in a.STAGES}
    stages=collection['stages'];entries=ledger['stages']
    checks.equal(len(stages),30,'all 30 stage records');checks.equal(len(entries),30,'all 30 ledger stages')
    checks.equal({(x['arm'],x['seed'],x['stage']) for x in stages},grid,'complete model-stage grid')
    checks.equal({(x['arm'],x['seed'],x['stage']) for x in entries},grid,'complete ledger grid')
    result=[];coverage=Counter();observed=0
    for stage in stages:
        arm,seed,name=stage['arm'],stage['seed'],stage['stage'];mode,split=next((m,s) for n,m,s in a.STAGES if n==name)
        checks.require(type(seed)is int,'integer seed identity')
        entry=next(x for x in entries if (x['arm'],x['seed'],x['stage'])==(arm,seed,name))
        checks.equal(entry['directory'],QUEUE+'/jobs/'+arm+'_seed'+str(seed)+'/'+name,'exact original stage directory')
        checks.equal(stage['cells'],entry['cells'],'full original cell accounting')
        checks.equal(stage['outcome'],entry['outcome'],'original invocation outcome')
        checks.equal((stage['mode'],stage['split']),(mode,split),'stage mode and split')
        wanted=a.expected_cells(schedules[split][mode],mode);cells=stage['cells']
        checks.equal(len(cells),len(wanted),'complete fixed stage denominator')
        checks.require(all(all(c.get(k)==v for k,v in w.items()) and c.get('state') in a.CELL_STATES for c,w in zip(cells,wanted)),'fixed source and cell state identities')
        coverage.update(c['state'] for c in cells)
        if name in OBSERVED:observed+=len(cells)
        result.append({'arm':arm,'seed':seed,'stage':name,'mode':mode,'split':split,'cells':cells,'outcome':stage['outcome']})
    checks.equal(sum(coverage.values()),4728,'all 4728 original cells')
    checks.equal(observed,2568,'all 2568 observed cells')
    checks.equal(dict(coverage),{'completed_required_outcome':2899,'not_completed_before_invocation_end':1817,'timed_out_current':12},'accepted global cell state totals')
    return result,schedules,populations

def audit(args,phase,budget):
    checks=a.Checks();check=budget.check
    need(str(args.collection)==COLLECTION and args.collection_sha256==COLLECTION_SHA and str(args.queue_root)==QUEUE,'exact original accepted collection/queue required')
    raw=read_bound(COLLECTION,COLLECTION_SHA,CAP_COLLECTION,check,COLLECTION_BYTES)
    collection=strict(raw);del raw
    full_accounting,schedules,populations=accounting(collection,checks)
    ref=collection['physical_reference']
    checks.equal((ref['dataset'],ref['frames'],ref['horizon'],ref['dimension'],ref['particle_type'],ref['metadata_sha256']),('Goop-3D',301,295,3,7,a.METADATA_SHA),'original physical reference')
    bounds=np.asarray(ref['metadata']['bounds'],dtype=np.float64)
    checks.require(bounds.shape==(3,2) and np.isfinite(bounds).all() and np.all(bounds[:,1]>bounds[:,0]),'valid saved bounds')
    bindings_path=Path(__file__).parent/'worker_source_bindings.json'
    bindings=strict(read_bound(bindings_path,args.source_bindings_sha256,CAP_JSON,check))
    sources=bindings['files_sha256'];need(type(sources)is dict and sources,'source bindings required')
    for path,pin in sources.items():read_bound(path,pin,CAP_JSON,check,retain=False)
    files=collection['files_sha256'];allowed={};prepared=[]
    for stage in collection['stages']:
        if stage['stage'] not in OBSERVED:continue
        arm,seed,name=stage['arm'],stage['seed'],stage['stage'];mode,split=stage['mode'],stage['split']
        directory=QUEUE+'/jobs/'+arm+'_seed'+str(seed)+'/'+name
        rows={a.unit(r,mode):r for r in stage['rows']}
        checks.equal(len(rows),len(stage['rows']),'unique committed observed rows')
        committed={a.unit(c,mode):c for c in stage['cells'] if c['state'] in ('completed_required_outcome','recorded_failed_outcome')}
        checks.equal(set(rows),set(committed),'committed rows equal required observed cell states')
        protocol=directory+'/protocol.json'
        if rows:checks.require(protocol in files,'committed observed rows require protocol');allowed[protocol]=files[protocol]
        for ident,row in rows.items():
            c=committed[ident];basename=f'trajectory_{ident[0]:06d}_target_{ident[1]:03d}'
            checks.equal(c['row_file'],basename+'.json','exact observed row basename')
            checks.equal(row['artifact_file'],basename+'.npz','exact observed archive basename')
            checks.equal(c['row_sha256'],files[directory+'/'+basename+'.json']['sha256'],'row inventory pin')
            checks.equal(row['artifact_sha256'],c['artifact_sha256'],'row and cell archive pins')
            checks.equal(row['artifact_sha256'],files[directory+'/'+basename+'.npz']['sha256'],'archive inventory pin')
            for suffix in ('.json','.npz'):
                path=directory+'/'+basename+suffix;allowed[path]=files[path]
        prepared.append((stage,directory,rows,committed))
    need(len(prepared)==18,'complete 18 observed stages')
    need(all(type(v['bytes'])is int and 0<=v['bytes']<=(CAP_ARCHIVE if p.endswith('.npz') else CAP_JSON) and is_pin(v['sha256']) for p,v in allowed.items()),'all selected file bounds/pins')
    # The initial inventory has a fixed 120-second subdeadline, with no error-value selection.
    initial=WorkerBudget(phase,120);initial.check();budget.check()
    progress=Path(args.output).with_name('audit.progress.jsonl')
    with progress.open('x') as f:
        f.write(json.dumps({'status':'scope_bound','required_observed_cells':2568,'required_all_cells':4728,'selected_files':len(allowed),'selected_bytes':sum(v['bytes'] for v in allowed.values()),'global_coverage':dict(Counter(c['state'] for s in full_accounting for c in s['cells']))})+'\n');f.flush()
    reads={}
    def bound(path):
        need(path in allowed,'path outside exact observed allowlist')
        entry=allowed[path];raw=read_bound(path,entry['sha256'],CAP_ARCHIVE if path.endswith('.npz') else CAP_JSON,check,entry['bytes'])
        reads[path]=entry['sha256'];return raw
    models=[];shared={}
    for stage,directory,rows,cells in prepared:
        check();arm,seed,name=stage['arm'],stage['seed'],stage['stage'];mode,split=stage['mode'],stage['split'];values={};details=[]
        if rows:
            protocol=strict(bound(directory+'/protocol.json'))
            checks.equal((protocol['schema'],protocol['purpose'],protocol['mode'],protocol['split'],protocol['arm'],protocol['seed'],protocol['horizon'],protocol['frames'],protocol['checkpoint_updates']),(a.EVALUATION_SCHEMA,'final_evaluation',mode,split,arm,seed,295,301,25000),'observed protocol identity')
            checks.equal(protocol['schedule'],schedules[split][mode],'exact observed protocol schedule')
            checks.equal(protocol['source_population_count'],populations[split],'original source population count')
        for c in stage['cells']:
            check();ident=a.unit(c,mode)
            if ident not in rows:
                checks.require(isinstance(c.get('reason'),str) and bool(c['reason']),'missing observed cell reason');continue
            row=rows[ident];checks.context=f'{arm}/seed{seed}/{name}/{ident}'
            checks.equal(strict(bound(directory+'/'+c['row_file'])),row,'exact original observed row JSON')
            checks.equal((row['arm'],row['training_seed'],row['objective']),(arm,seed,'faithful'),'observed model identity')
            checks.equal(row['status'],'complete' if c['state']=='completed_required_outcome' else 'failed','observed cell status')
            checks.equal(row.get('failure'),c.get('failure'),'original observed failure')
            checks.equal(row['protocol_sha256'],allowed[directory+'/protocol.json']['sha256'],'row protocol pin')
            arrays=numeric_archive(bound(directory+'/'+row['artifact_file']));check()
            source=next(x for x in schedules[split]['full-rollout'] if x['source_index']==ident[0])
            checks.equal(arrays['particle_types'].shape,(source['particles'],),'original source particle count')
            metrics=a.diagnostic.audit_row(row,arrays,bounds,checks,mode=mode)
            if mode=='clean-validation' and row['status']=='complete':
                a.graph_scalar(row['graph'],'base',len(arrays['particle_types']),checks)
                a.graph_arrays(arrays['native_edges'],np.empty((0,2),dtype=np.int64),row['graph'],len(arrays['particle_types']),checks)
            if mode=='same-state':
                checks.equal(row['random_seed_material'],[20261006,93000,seed,0 if split=='valid' else 1,ident[0],ident[1]],'fixed policy RNG material')
                metrics.update(a.extra_diagnostic_metrics(row,arrays,bounds,checks))
            pairings=a.diagnostic_shared_hashes(arrays,split,ident,mode,checks)
            if mode=='same-state':
                for method in ('base','dense','random25','speed25','relative-velocity-RMS25','natural_base_reference'):
                    for kind in ('edges','selected_optional_pairs'):
                        key=f'r1_{method}__{kind}'
                        if key in arrays:pairings[f'{split}/graph/{method}/{kind}/{ident}'+(f'/seed{seed}' if method=='random25' else '')]=a.array_hash(arrays[key])
            for key,value in pairings.items():
                if key in shared:checks.equal(shared[key],value,'observed shared input or model-independent graph identity')
                shared[key]=value
            values[ident]=metrics;details.append({'unit':list(ident),'status':row['status'],'failure':row.get('failure'),'metrics':metrics,'policy_completion':[p for p,v in row.get('policies',{}).items() if v.get('status')=='complete']})
            del arrays;check()
            with progress.open('a') as f:f.write(json.dumps({'stage':name,'arm':arm,'seed':seed,'unit':list(ident),'status':'row_checked_not_family_admitted'})+'\n')
        expected=[a.unit(x,mode) for x in schedules[split][mode]];aggregates={}
        for metric in a.diagnostic_keys(mode):
            computed=a.aggregate(expected,{k:v.get(metric) for k,v in values.items()});aggregates[metric]=computed
            if metric in a.diagnostic.expected_metric_keys(mode):
                original=a.tree_leaf(stage['diagnostic_summary'],metric)
                checks.close({k:original[k] for k in computed},computed,'original diagnostic aggregation arithmetic')
        models.append({'arm':arm,'seed':seed,'stage':name,'mode':mode,'split':split,'cells':stage['cells'],'coverage':dict(Counter(c['state'] for c in stage['cells'])),'rows':details,'aggregates':aggregates})
    checks.context='final exact observed byte recheck'
    need(set(reads)==set(allowed),'every selected observed file read and checked')
    for path,pin in reads.items():read_bound(path,pin,CAP_ARCHIVE if path.endswith('.npz') else CAP_JSON,check,allowed[path]['bytes'],retain=False)
    read_bound(COLLECTION,COLLECTION_SHA,CAP_COLLECTION,check,COLLECTION_BYTES,retain=False)
    for path,pin in sources.items():read_bound(path,pin,CAP_JSON,check,retain=False)
    read_bound(bindings_path,args.source_bindings_sha256,CAP_JSON,check,retain=False)
    return {'schema':AUDIT_SCHEMA,'status':'passed_scoped_observed_history_checks','dataset':'Goop-3D','endpoint_updates':25000,'collection_sha256':COLLECTION_SHA,'cohort_sha256':COHORT_SHA,'phase_sha256':args.phase_sha256,'original_expired_phase_sha256':OLD_PHASE_SHA,'source_manifest_sha256':collection['source_manifest_sha256'],'physical_reference':ref,'checks':checks.count,'source_schedules':schedules,'all_original_accounting':full_accounting,'required_all_cells':4728,'required_observed_cells':2568,'models':models,'shared_identity_sha256':shared,'verified_observed_files_sha256':reads,'source_sha256':sources,'source_bindings_sha256':args.source_bindings_sha256,'original_full_audit_rerun':False,'autonomous_array_or_result_read':False,'scientific_admission':False,'limitations':['Conditional observed-history arithmetic only; no autonomous rollout conclusion.','Saved normalization/source truth inherited; model and original source arrays not reopened.','Fixed source-order grid is not a probability sample; all missing values retain their denominators.','Diagnostic timing arithmetic does not establish clock truth or end-to-end speedup.']}

def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for n in ('collection','queue-root','output','phase'):p.add_argument('--'+n,type=Path)
    for n in ('collection-sha256','phase-sha256','source-bindings-sha256'):p.add_argument('--'+n)
    args=p.parse_args()
    if not args.execute:print(json.dumps({'status':'inert_observed_history_audit','allocation_issued':False}));return
    need(str(args.output)==OUTPUT+'/audit.json' and not args.output.exists(),'fresh scoped audit output required')
    phase=load_phase(args.phase,args.phase_sha256);budget=WorkerBudget(phase,2700)
    resource.setrlimit(resource.RLIMIT_AS,(32<<30,32<<30))
    try:
        result=audit(args,phase,budget);pin=publish(args.output,result,budget.check)
        print(json.dumps({'status':result['status'],'sha256':pin,'scientific_admission':False}))
    except BaseException as error:
        try:publish(args.output.with_name('audit.failure.json'),{'schema':AUDIT_SCHEMA,'status':'failed_or_incomplete','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'original_inputs_modified':False,'scientific_admission':False},budget.check)
        except BaseException:pass
        raise

if __name__=='__main__':main()
