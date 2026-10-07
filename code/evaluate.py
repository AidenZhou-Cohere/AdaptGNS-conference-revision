"""Evaluate native graph policies with explicit data, checkpoint and output paths.

Metadata-only checks verify the requested study configuration. --execute loads
the model and numeric trajectories, preserves every unsuccessful outcome and
uses the packaged rollout or observed-history numerical functions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
SPECS = {
    'goop': {'frames':401,'horizon':395,'indices':list(range(30)),'updates':100000},
    'sand': {'frames':320,'horizon':314,'indices':list(range(30)),'updates':100000},
    'waterdrop': {'frames':1001,'horizon':995,'indices':list(range(3,30)),'updates':110000},
}
POLICIES=('base','dense','random25','speed25','laggedrisk25','relative-velocity-RMS25')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--study',choices=SPECS,required=True)
    p.add_argument('--mode',choices=('rollout','observed','clean-validation'),default='rollout')
    p.add_argument('--split',choices=('valid','test'),default='test')
    p.add_argument('--arm',choices=('base','mix'),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    for name in ('manifest','metadata','checkpoint','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest','metadata','checkpoint'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--parent-sha256',help='Required separately for a portable WaterDrop continuation endpoint')
    p.add_argument('--device',default='cpu',help='cpu, mps, or cuda:N (no automatic fallback)')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--execute',action='store_true',help='Without this flag, validate metadata/configuration only')
    a=p.parse_args(argv)
    require(re.fullmatch(r'cpu|mps|cuda:[0-9]+',a.device) is not None,'Explicit cpu/mps/cuda:N device required')
    require(a.threads==2,'The study setting is two CPU threads')
    require(a.mode!='clean-validation' or a.split=='valid','Clean validation requires valid split')
    require(a.study!='waterdrop' or a.mode=='rollout','WaterDrop observed-history selection differs; use its original full_same_state routines')
    for name in ('manifest','metadata','checkpoint'):
        require(re.fullmatch(r'[0-9a-f]{64}',getattr(a,name+'_sha256')) is not None,'Expected SHA256 for '+name)
    return a

def configuration(args, check_files=True):
    spec=SPECS[args.study]
    result={'adapter':'portable_native_evaluation_v1','study':args.study,'mode':args.mode,
            'split':args.split,'arm':args.arm,'seed':args.seed,'device':args.device,'threads':args.threads,
            **spec,'policies':list(POLICIES[:-1] if args.study=='waterdrop' else POLICIES),
            'rng_seed':('93000+1000*seed+source_index' if args.mode=='rollout' else 'SeedSequence([20261006,93000,seed,split_index,source_index,target_frame])' if args.mode=='observed' else 'No stochastic policy; native base only'),
            'input_sha256':{n:getattr(args,n+'_sha256') for n in ('manifest','metadata','checkpoint')},
            'expected_parent_checkpoint_sha256':args.parent_sha256}
    if not check_files:return result
    for name in ('manifest','metadata','checkpoint'):
        require(sha(getattr(args,name))==getattr(args,name+'_sha256'),'Input hash mismatch: '+name)
    manifest=json.loads(args.manifest.read_text());metadata=json.loads(args.metadata.read_text())
    require(manifest.get('format')=='gns-trajectory-manifest' and manifest.get('version')==1,'Numeric manifest v1 required')
    require(manifest.get('split')==args.split,'Manifest split differs')
    require(manifest.get('metadata')==metadata,'Explicit metadata differs from manifest')
    require(manifest.get('metadata_sha256')==args.metadata_sha256,'Manifest metadata hash differs')
    records=manifest.get('records',[])
    require(len(records)==30 and manifest.get('record_count')==30,'Complete official 30-record split required')
    require(all(r.get('source_index')==i and r.get('id')==f'{args.split}:{i:06d}' for i,r in enumerate(records)),'Ordered original split source indices and IDs required')
    require(all(r['positions'].get('dtype')=='<f4' for r in records),'Original float32 position descriptors required')
    allowed_types=('<i8','<i4') if args.study=='sand' else ('<i8',)
    require(all(r['particle_types'].get('dtype') in allowed_types for r in records),'Original integer particle type descriptors required')
    require(all(r['positions']['shape'][0]==spec['frames'] and r['positions']['shape'][-1]==2 for r in records),'Study trajectory dimensions differ')
    require(metadata.get('dim')==2 and metadata.get('default_connectivity_radius')==.015 and metadata.get('dt')==.0025,'Study dimension/radius/timestep differs')
    require(args.metadata.resolve()==args.manifest.resolve().parent/'metadata.json','Manifest requires adjacent matching metadata.json')
    require(not args.output_dir.exists(),'Fresh output directory required; existing attempts are never overwritten')
    out=args.output_dir.resolve()
    for protected in (ROOT,args.manifest.resolve().parent,args.checkpoint.resolve().parent):
        require(out!=protected and protected not in out.parents and out not in protected.parents,'Output must be separate from code, data and checkpoints')
    return result

def module(name):
    path=ROOT/'studies'/name
    spec=importlib.util.spec_from_file_location('_portable_'+path.stem,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    temp.replace(path)

def execute(args,config):
    # Imports and array/model access occur only inside this explicit execution path.
    benchmark=module('benchmark_'+('sand' if args.study=='sand' else 'goop')+'_graph_support_rollout.py')
    native,helpers=benchmark.load_helpers(ROOT)
    torch,np=helpers.torch,helpers.np
    torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type=='cuda':
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Start CUDA execution with CUBLAS_WORKSPACE_CONFIG=:4096:8')
        require(torch.cuda.is_available(),'Requested CUDA unavailable')
        torch.cuda.set_device(device);torch.use_deterministic_algorithms(True,warn_only=False)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    elif device.type=='mps':
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0','MPS execution requires PYTORCH_ENABLE_MPS_FALLBACK=0')
        require(torch.backends.mps.is_available(),'Requested MPS unavailable')
    trajectories=native.full.load_manifest_data(args.manifest,verify_hashes=True)
    manifest=json.loads(args.manifest.read_text());metadata=json.loads(args.metadata.read_text())
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if args.study=='waterdrop':
        from research import faithful_graph_support as continuation
        if payload.get('graph_support_schema')!='1' and payload.get('graph_support_schema')!=1:
            spec=importlib.util.spec_from_file_location('_portable_waterdrop_identity',ROOT/'portable/continue_waterdrop.py')
            portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)
            require(args.parent_sha256 is not None,'Portable WaterDrop evaluation requires explicit --parent-sha256')
            portable.validate_portable_identity(payload,args.seed,args.parent_sha256)
            portable.bind_parent_for_validation(continuation,args.seed,args.parent_sha256)
        saved,_=continuation.validate_continuation_payload(payload,require_complete=True)
        require(payload['completed_total_updates']==110000,'Continuation total endpoint differs')
    else:
        trainer=module('train_'+args.study+'_graph_support_cuda.py')
        allowed=(trainer.SCHEMA,trainer.SCHEMA.replace('_training_v1','_portable_training_v1'))
        require(payload.get('cuda_'+args.study+'_graph_support_schema') in allowed,'Unknown checkpoint schema')
        saved=payload.get('run_config',{})
        require(payload.get('completed_steps')==100000,'Checkpoint endpoint differs')
        require(payload.get('run_config_sha256')==trainer.config_hash(saved),'Checkpoint config hash differs')
        require(saved.get('schema') in allowed,'Checkpoint/config schema differs')
        require(saved.get('dataset')==('Goop' if args.study=='goop' else 'Sand'),'Checkpoint material differs')
        trainer.validate_graph_history(payload['history'],100000,args.arm,args.seed)
    require(saved.get('seed')==args.seed and saved.get('objective')=='faithful','Checkpoint seed/objective differs')
    require(saved.get('arm')==args.arm,'Checkpoint training arm differs')
    model,_=native.full.load_for_evaluation(args.checkpoint,metadata,device,radius_backend='scipy_host')
    require(model._max_num_neighbors==128,'Native receiver cap differs')
    args.output_dir.mkdir(parents=True)
    config['runtime']={'torch':str(torch.__version__),'numpy':np.__version__,'device':str(device),'threads':2}
    config['numerical_sources']={k:v for k,v in benchmark.SOURCE_PINS.items()}
    atomic(args.output_dir/'configuration.json',config)
    rows=[]
    try:
        if args.mode=='rollout':
            for index in config['indices']:
                positions,types=trajectories[index]
                require(np.all(types==(7 if args.study=='goop' else 6)) if args.study in ('goop','sand') else np.all(types!=3),'Study particle types differ')
                for policy in config['policies']:
                    native.full.synchronize(device);started=time.perf_counter()
                    row,traces=native.rollout(model,positions,types,metadata,policy,config['horizon'],93000+1000*args.seed+index,device,
                        trace_steps=tuple(sorted({1,10,50,200,config['horizon']})))
                    native.full.synchronize(device)
                    row.update(source_index=index,trajectory_id=manifest['records'][index]['id'],arm=args.arm,training_seed=args.seed,
                               synchronized_call_seconds=time.perf_counter()-started)
                    stem=f'trajectory_{index:06d}_{policy}'
                    require(all(isinstance(v,np.ndarray) and not v.dtype.hasobject for v in traces.values()),'Trace arrays must be numeric')
                    with (args.output_dir/(stem+'.npz')).open('xb') as f:np.savez_compressed(f,**traces)
                    atomic(args.output_dir/(stem+'.json'),row);rows.append(row)
                    atomic(args.output_dir/'results.json',{'records':rows,'required_records':len(config['indices'])*len(config['policies'])})
                    if row.get('failure',{}):require(row['failure'].get('category') not in ('native_parity_failure','execution_error'),'Retained parity/implementation failure; stop without retry')
        else:
            diagnostic=module('evaluate_'+args.study+'_graph_support_final.py')
            schedule=diagnostic.schedules(manifest['records'],'same-state' if args.mode=='observed' else 'clean-validation')
            for item in schedule:
                positions,types=trajectories[item['source_index']]
                require(np.all(types==(7 if args.study=='goop' else 6)),'Study particle types differ')
                if args.mode=='observed':row,traces=diagnostic.same_state(native,model,positions,types,metadata,item,args.split,args.seed,device)
                else:row,traces=diagnostic.clean_validation(native,helpers,model,positions,types,item,device)
                stem=f'history_{item["schedule_index"]:06d}'
                require(all(isinstance(v,np.ndarray) and not v.dtype.hasobject for v in traces.values()),'Trace arrays must be numeric')
                with (args.output_dir/(stem+'.npz')).open('xb') as f:np.savez_compressed(f,**traces)
                atomic(args.output_dir/(stem+'.json'),row);rows.append(row)
                atomic(args.output_dir/'results.json',{'records':rows,'required_records':len(schedule)})
                if row.get('failure',{}):require(row['failure'].get('category') not in ('native_parity_failure','execution_error'),'Retained parity/implementation failure; stop without retry')
        for name in ('manifest','metadata','checkpoint'):require(sha(getattr(args,name))==getattr(args,name+'_sha256'),'Input changed during execution: '+name)
        native.full.load_manifest_data(args.manifest,verify_hashes=True)
        atomic(args.output_dir/'completion.json',{'complete_grid':True,'returned_records':len(rows),'unsuccessful_records':sum(r.get('status')!='complete' for r in rows),'full_horizon_means':'Compute only when all required trajectories in a group complete; retain failed prefixes separately'})
    except BaseException as exc:
        atomic(args.output_dir/'failure.json',{'type':type(exc).__name__,'message':str(exc),'returned_records':len(rows),'retried':False})
        raise

def main(argv=None):
    args=parse_args(argv);config=configuration(args)
    if args.execute:execute(args,config)
    else:print(json.dumps(config,indent=2,sort_keys=True))

if __name__=='__main__':main()
