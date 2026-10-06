#!/usr/bin/env python3
"""Separate fixed Goop global-action-gate labels and autonomous rollouts.

Description only by default. No acquisition, model training, policy-global
mutation or automatic retry. Root admissions and a scoped parent are required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop_global_action_gate_driver_v1'
RELEASE_SCHEMA = 'adaptgns_goop_global_action_gate_driver_release_v1'
LABEL_SCHEMA = 'adaptgns_goop_global_action_gate_label_row_v1'
LABEL_COLLECTION = 'adaptgns_goop_global_action_gate_label_collection_v1'
ROLLOUT_SCHEMA = 'adaptgns_goop_global_action_gate_rollout_row_v1'
ROLLOUT_COLLECTION = 'adaptgns_goop_global_action_gate_rollout_collection_v1'
GATE_PROTOCOL_SHA = 'bc9235fae89d32ede8d1e7f846bff07bdcdd85f4671d5922d688a9535f7fe024'
TRAINING_PROTOCOL_SHA = 'c8690d0da209c66557e3dbb0cbccd55d660b4cc270683913f74d381516591851'
CORE_SHA = 'd3986c2ac3786e9fc0d36d76339bfeded5448497f7dad9b0559efaaaa53a43ca'
SOURCE_PINS = {
    'goop_global_action_gate_core_v1.py': CORE_SHA,
    'benchmark_goop_graph_support_rollout.py': '2e0ef0b84635c8430102cd797648d24cec14b457e1df900eb784d5e167966f8f',
    'evaluate_goop_graph_support_final.py': 'cd970e04012930d896b31944271ccaf7da2b0881f22fe2f903533a8d5911dc6d',
    'goop_evaluation_contract.py': '4ee1e33dbe666804d1827b88565620a435d04836a170671914a71acb818be090',
    'sand_graph_support_policy.py': '4a1db45f742043ba261c06273c1538fa0195dfb7f203423510b9a353c5ddb03a',
    'train_goop_graph_support_cuda.py': 'dd9ef01a16116f04a578bbfe9ed01774e66fdc08d60b913072a33ad211b319c1',
}
MODES = {'train-label-capacity':'train', 'train-labels':'train', 'validation-labels':'valid',
         'train-rollout-capacity':'train', 'test-rollout':'test'}
DEADLINE = datetime(2026,10,7,4,tzinfo=timezone.utc)
PATH_ARGS = ('repo','benchmark-helper','cohort','cohort-audit','protocol','gate-protocol','trainer-source',
    'train-admission','manifest','admission','structural-report','acquisition-report','context-semantics','auxiliary-report',
    'cross-split-audit','checkpoint','output-dir','head','selection')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def json_snapshot(path, digest=None):
    raw=Path(path).read_bytes(); actual=hashlib.sha256(raw).hexdigest()
    require(digest is None or actual==digest, 'JSON bytes differ: '+str(path))
    return json.loads(raw),actual


def stamp(value):
    result=datetime.fromisoformat(value);require(result.tzinfo is not None,'Timezone-aware stop required');return result.astimezone(timezone.utc)


def write(path,value,exclusive=False):
    raw=encode(value);path=Path(path)
    if exclusive:
        with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    else:
        tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def tree_snapshot(root, exclude=()):
    root=Path(root);files={};entries=[]
    for path in sorted(root.rglob('*')):
        name=str(path.relative_to(root))
        if name in exclude:continue
        require(not path.is_symlink() and (path.is_file() or path.is_dir()),'Ordinary output entries required')
        kind='file' if path.is_file() else 'directory';entries.append({'path':name,'kind':kind})
        if kind=='file':files[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    return {'files':files,'output_tree_entries':entries}


def verify_inputs(bindings):
    require(all(Path(p).is_file() and not Path(p).is_symlink() and sha(p)==h for p,h in bindings.items()),'Original input/source bytes changed')


def bind_inputs(bindings,incoming):
    for path,pin in incoming.items():
        resolved=str(Path(path).resolve())
        require(Path(path).is_absolute() and not Path(path).is_symlink() and isinstance(pin,str) and len(pin)==64
            and all(c in '0123456789abcdef' for c in pin) and (resolved not in bindings or bindings[resolved]==pin),
            'Conflicting/nonabsolute input binding')
        bindings[resolved]=pin


def failure_record(error,phase,**context):
    return {'category':'execution_budget' if isinstance(error,TimeoutError) else 'execution_error',
        'phase':phase,'error_type':type(error).__name__,'error':str(error),**context}


def load_source(name,label):
    path=HERE/name;require(sha(path)==SOURCE_PINS[name],'Immutable numerical/core source differs: '+name)
    spec=importlib.util.spec_from_file_location(label,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def modules():
    return SimpleNamespace(core=load_source('goop_global_action_gate_core_v1.py','_gate_core'),
        bench=load_source('benchmark_goop_graph_support_rollout.py','_gate_benchmark'),
        final=load_source('evaluate_goop_graph_support_final.py','_gate_final_contract'))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--execute',action='store_true');p.add_argument('--mode',choices=tuple(MODES));p.add_argument('--release',type=Path)
    for name in PATH_ARGS:p.add_argument('--'+name,type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--checkpoint-sha256');p.add_argument('--cuda-index',type=int);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-seconds',type=int)
    return p.parse_args(argv)


def validate_release(args, release, now=None):
    """Scalar argument/source authority gate; no model/array deserialization."""
    now=datetime.now(timezone.utc) if now is None else now
    require(args.mode in MODES and type(args.seed) is int and args.seed in (0,1,2) and type(args.cuda_index) is int and args.cuda_index in (0,1)
            and args.threads==2 and type(args.max_seconds) is int and args.max_seconds>0,'Fixed mode/seed/B-device/whole-invocation quota required')
    required=set(PATH_ARGS)-{'head','selection','cross-split-audit'}
    if 'rollout' in args.mode:required|={'head','selection'}
    if args.mode=='test-rollout':required.add('cross-split-audit')
    required|={k for k in PATH_ARGS if getattr(args,k.replace('-','_')) is not None}
    require(all(getattr(args,k.replace('-','_')) is not None for k in required) and args.release is not None,'Every fixed input argument required')
    require(all(Path(getattr(args,k.replace('-','_'))).is_absolute() for k in required),'Absolute input/output paths required')
    own=sha(__file__)
    require(release.get('schema')==RELEASE_SCHEMA and release.get('issued_by')=='root'
        and release.get('status')=='approved_for_fixed_gate_driver' and release.get('host')==socket.gethostname()
        and release.get('mode')==args.mode and release.get('seed')==args.seed and type(release.get('seed')) is int
        and release.get('arm')=='mix' and release.get('checkpoint_updates')==100000 and release.get('cuda_index')==args.cuda_index
        and type(release.get('cuda_index')) is int and release.get('threads')==2
        and release.get('max_seconds')==args.max_seconds and release.get('output_dir')==str(args.output_dir)
        and isinstance(release.get('gpu_uuid'),str) and bool(release.get('gpu_uuid'))
        and release.get('source_sha256')=={**SOURCE_PINS,Path(__file__).name:own}
        and release.get('protocol_sha256')==GATE_PROTOCOL_SHA and release.get('original_training_protocol_sha256')==TRAINING_PROTOCOL_SHA,
        'Exact root driver release required')
    stop=stamp(release['absolute_stop_utc']);require(now<stop<=DEADLINE,'Gate study stop reached or too late')
    pins=release.get('files_sha256',{});require(isinstance(pins,dict),'Complete root file pins required')
    bind_inputs({},pins)
    for name in required-{'repo','output-dir'}:
        path=Path(getattr(args,name.replace('-','_')))
        require(str(path) in pins and sha(path)==pins[str(path)],'Root input pin differs: '+name)
    require(sha(args.gate_protocol)==GATE_PROTOCOL_SHA and sha(args.protocol)==TRAINING_PROTOCOL_SHA
        and sha(args.checkpoint)==args.checkpoint_sha256,'Fixed protocol/checkpoint bytes required')
    require(args.benchmark_helper==HERE/'benchmark_goop_graph_support_rollout.py'
        and args.trainer_source==HERE/'train_goop_graph_support_cuda.py','Exact immutable numerical helper aliases required')
    if 'rollout' in args.mode:
        head,_=json_snapshot(args.head,pins[str(args.head)]);selection,_=json_snapshot(args.selection,pins[str(args.selection)])
        require(head.get('schema')=='adaptgns_goop_global_action_gate_head_v1' and head.get('model_seed')==args.seed
            and head.get('protocol_sha256')==GATE_PROTOCOL_SHA and head.get('training_states')==8000
            and head.get('training_input_hashes',{}).get('checkpoint_sha256')==args.checkpoint_sha256,
            'Selected head model/training identity differs')
        require(selection.get('schema')=='adaptgns_goop_global_action_gate_selection_v1'
            and selection.get('protocol_sha256')==GATE_PROTOCOL_SHA and selection.get('heads',{}).get(str(args.seed))==head
            and set(selection.get('heads',{}))=={'0','1','2'},'Common selection/all three selected heads required')
        probability=head.get('validation_requested_expansion_fraction')
        require(type(probability) in (int,float) and np.isfinite(probability) and 0<=probability<=1,
            'Frozen validation expansion probability required')
    if args.mode=='test-rollout':
        entry=release.get('complete_capacity_report',{})
        require(isinstance(entry,dict) and set(entry)=={'path','sha256'} and pins.get(entry['path'])==entry['sha256'],'Separate bound complete-capacity report required')
        capacity,_=json_snapshot(entry['path'],entry['sha256'])
        require(capacity.get('schema')=='adaptgns_goop_global_action_gate_capacity_budget_v1'
            and capacity.get('status')=='complete_measured_capacity_passed' and capacity.get('protocol_sha256')==GATE_PROTOCOL_SHA
            and capacity.get('selection_sha256')==pins[str(args.selection)] and capacity.get('all_36_rollouts_complete') is True
            and capacity.get('all_train_validation_labels_complete') is True and now<=stamp(capacity['latest_test_start_utc']),
            'Complete fixed-study measured capacity has not passed/does not fit')
    return pins


def release_gate(args, now=None):
    """Complete scalar cohort/split/source admission; no simulator/array loading."""
    release,release_sha=json_snapshot(args.release);pins=validate_release(args,release,now);mods=modules()
    args.split=MODES[args.mode];args.arm='mix';args.objective='faithful';args.benchmark_sha256=SOURCE_PINS['benchmark_goop_graph_support_rollout.py']
    cohort,cohort_sha=json_snapshot(args.cohort,pins[str(args.cohort)]);admission_train,_=json_snapshot(args.train_admission,pins[str(args.train_admission)])
    mods.final.check_cohort(cohort,args,admission_train)
    manifest,admission,structural=[json_snapshot(getattr(args,k),pins[str(getattr(args,k))])[0] for k in ('manifest','admission','structural_report')]
    if args.split=='train':
        require(sha(args.manifest)==admission.get('manifest_sha256') and sha(args.structural_report)==admission.get('structural_report_sha256'),'Bound train scalar evidence differs')
        mods.bench.load_data_contract().validate_split(manifest,admission,structural,'train')
    else:mods.final.check_split(mods.bench,manifest,admission,structural,args,cohort_sha)
    metadata_path=args.manifest.parent/'metadata.json'
    require(sha(metadata_path)==mods.final.METADATA_SHA and json_snapshot(metadata_path)[0]==manifest['metadata'],'Actual metadata differs')
    protected=[args.manifest.parent.resolve(),args.checkpoint.parent.resolve(),(args.repo/'adaptive-gns').resolve()]
    out=args.output_dir.resolve();require(not out.exists() and all(out!=p and out not in p.parents and p not in out.parents for p in protected),'Fresh separate output directory required')
    bindings={};bind_inputs(bindings,pins)
    bind_inputs(bindings,{str(args.release.resolve()):release_sha})
    bind_inputs(bindings,{str(HERE/name):h for name,h in SOURCE_PINS.items()})
    bind_inputs(bindings,{str(Path(__file__).resolve()):sha(__file__),str(metadata_path.resolve()):mods.final.METADATA_SHA})
    verify_inputs(bindings)
    return SimpleNamespace(args=args,release=release,release_sha=release_sha,mods=mods,cohort=cohort,cohort_sha=cohort_sha,
        train_admission=admission_train,manifest=manifest,admission=admission,metadata_path=metadata_path,bindings=bindings)


def capacity_sources(records):
    ordered=sorted(range(len(records)),key=lambda i:(records[i]['positions']['shape'][1],i));require(len(ordered)==1000,'All1000 train source descriptors required')
    chosen=[]
    for rank in (0,(len(ordered)-1)//2,len(ordered)-1):
        while ordered[rank] in chosen:rank=(rank+1)%len(ordered)
        chosen.append(ordered[rank])
    return chosen


def expected_schedule(core,mode,manifest):
    if mode=='train-label-capacity':
        rows=core.row_ids('train');return [{'source_index':rows[i][0],'target_frame':rows[i][1]} for i in core.capacity_label_indices()]
    if mode in ('train-labels','validation-labels'):
        return [{'source_index':i,'target_frame':t} for i,t in core.row_ids(MODES[mode])]
    sources=capacity_sources(manifest['records']) if mode=='train-rollout-capacity' else range(30)
    return [{'source_index':i,'policy':p} for i in sources for p in core.POLICIES]


def array_descriptors(arrays):
    return {k:{'shape':list(v.shape),'dtype':v.dtype.str,'value_sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in arrays.items()}


def publish_row(root,name,row,arrays):
    require(all(isinstance(v,np.ndarray) and v.dtype.kind in 'fiub' for v in arrays.values()),'Numeric-only artifacts required')
    before=array_descriptors(arrays)
    begin=time.perf_counter();artifact=Path(root)/(name+'.npz');tmp=artifact.with_suffix('.npz.tmp')
    with tmp.open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    raw=tmp.read_bytes()
    with np.load(io.BytesIO(raw),allow_pickle=False) as saved:
        require(len(saved.files)==len(set(saved.files)) and array_descriptors({k:saved[k] for k in saved.files})==before,
            'Serialized numeric arrays changed during publication')
    require(array_descriptors(arrays)==before,'Original numeric arrays changed during publication')
    tmp.replace(artifact)
    artifact_sha=hashlib.sha256(raw).hexdigest()
    require(sha(artifact)==artifact_sha,'Artifact bytes changed during rename')
    row=dict(row,artifact_file=artifact.name,artifact_sha256=artifact_sha,numeric_arrays=before)
    row['timing']['publication_seconds']=time.perf_counter()-begin
    row_raw=encode(row)
    require(sha(artifact)==artifact_sha and array_descriptors(arrays)==before,'Numeric bytes changed during row serialization')
    row_path=artifact.with_suffix('.json')
    with row_path.open('xb') as stream:stream.write(row_raw);stream.flush();os.fsync(stream.fileno())
    row_sha=hashlib.sha256(row_raw).hexdigest()
    return {**row,'row_file':row_path.name,'row_sha256':row_sha}


def pair_actions(native,core,model,history,types,seed,source,target,split,device):
    """No target values accepted; return both actions before label scoring."""
    arrays={'history':np.array(history,dtype=np.float32,copy=True),'particle_types':np.array(types,dtype=np.int64,copy=True)}
    timing={};graphs={};started=time.perf_counter();phase='state';passes=0
    try:
        native.full.check_state(history,'observed_state',native.bridge.MAX_ABS)
        at=time.perf_counter();arrays['features']=core.features(history);timing['feature_seconds']=time.perf_counter()-at
        for policy in ('base','random25'):
            phase=policy+'_graph';at=time.perf_counter()
            _,edges,optional,audit=native.native_graph(history,policy,None,core.pair_rng(split,seed,source,target))
            timing[policy+'_graph_seconds']=time.perf_counter()-at;graphs[policy]=audit;arrays[policy+'_edges']=edges
            if policy=='random25':arrays['random25_selected_optional_pairs']=optional
            phase=policy+'_forward';at=time.perf_counter();passes+=1
            output=native.bridge.supplied(model,history,types,edges,device)
            for key in ('prediction','risk','raw_risk'):arrays[policy+'_'+key]=output[key]
            native.bridge.validate_output(output,len(types));native.full.synchronize(device)
            timing[policy+'_forward_seconds']=time.perf_counter()-at
        require(np.array_equal(arrays['random25_edges'][:,:arrays['base_edges'].shape[1]],arrays['base_edges']),'Random graph changed native base prefix')
        timing['paired_action_seconds']=time.perf_counter()-started
        return {'status':'complete','failure':None,'timing':timing,'network_passes':passes,'base_graph':graphs['base'],'random25_graph':graphs['random25']},arrays
    except native.full.RolloutGuard as error:
        failure={**error.details,'phase':phase}
    except Exception as error:failure=failure_record(error,phase)
    timing['paired_action_seconds']=time.perf_counter()-started
    return {'status':'failed','failure':failure,'timing':timing,'network_passes':passes,'base_graph':graphs.get('base'),'random25_graph':graphs.get('random25')},arrays


def label_row(native,core,model,positions,types,item,seed,split,device):
    source,target=item['source_index'],item['target_frame'];history=np.array(positions[target-6:target],dtype=np.float32,copy=True)
    row,arrays=pair_actions(native,core,model,history,types,seed,source,target,split,device)
    row.update(schema=LABEL_SCHEMA,model_seed=seed,arm='mix',checkpoint_updates=100000,split=split,**item,
        features=None,base_mse=None,random25_mse=None,signed_benefit=None,pair_rng_material=core.rng_material(split,seed,source,target),
        history_sha256=native.full.state_hash(history),prediction_sha256={p:native.full.state_hash(arrays[p+'_prediction']) for p in ('base','random25') if p+'_prediction' in arrays})
    if row['status']=='complete':
        try:
            at=time.perf_counter();truth=np.array(positions[target],dtype=np.float32,copy=True);arrays['target']=truth
            errors=core.action_errors(arrays['base_prediction'],arrays['random25_prediction'],truth)
            arrays['target']=truth;arrays['action_errors']=np.asarray([errors[k] for k in ('base_mse','random25_mse','signed_benefit')],dtype=np.float64)
            row.update(errors,features=arrays['features'].tolist());row['timing']['scoring_seconds']=time.perf_counter()-at
        except Exception as error:row.update(status='failed',failure=failure_record(error,'label_scoring'))
    return row,arrays


def choose_action(core,history,policy,head,domain,seed,source,step):
    require(policy in core.POLICIES,'Unknown frozen gate policy')
    at=time.perf_counter();features=None;predicted=None
    if policy=='learned_global_gate':features=core.features(history);predicted=core.predict(head,features);expand=predicted>0
    elif policy=='validation_rate_random_gate':expand=core.independent_gate(head['validation_requested_expansion_fraction'],domain,seed,source,step)
    else:expand=policy=='random25'
    return {'requested_expand':bool(expand),'predicted_benefit':predicted,'feature_and_gate_seconds':time.perf_counter()-at},features


def rollout_row(native,core,model,positions,types,metadata,item,seed,head,domain,device):
    """One real forward/forecast; target read only after action and prediction."""
    history=np.array(positions[:6],dtype=np.float32,copy=True);source,policy=item['source_index'],item['policy']
    arrays={'initial_history':history.copy(),'particle_types':np.asarray(types,dtype=np.int64)}
    steps=[];predictions=[];truths=[];feature_trace=[];failure=None;phase='initial_state';passes=0;parity_passes=0
    started=time.perf_counter();parity=None
    try:
        native.full.check_state(history,'initial_state',native.bridge.MAX_ABS)
        phase='initial_graph'
        _,base_edges,_,_=native.native_graph(history,'base',None,core.pair_rng(domain,seed,source,1))
        phase='native_parity';parity_passes=None
        parity,parity_arrays,parity_output=native.bridge.native_parity(model,history,types,base_edges,device)
        parity_passes=2
        arrays.update({'initial_parity_'+k:v for k,v in parity_arrays.items()})
        arrays.update({'initial_parity_supplied_'+k:parity_output[k] for k in ('prediction','raw_risk','risk')})
        require(parity.get('passed') is True,'Native parity failed')
        parity_seconds=time.perf_counter()-started
        for step in range(1,396):
            phase='state';native.full.check_state(history,'state',native.bridge.MAX_ABS)
            phase='gate';decision,features=choose_action(core,history,policy,head,domain,seed,source,step)
            at=time.perf_counter();phase='current_graph'
            _,edges,optional,audit=native.native_graph(history,'random25' if decision['requested_expand'] else 'base',None,core.pair_rng(domain,seed,source,step))
            graph_seconds=time.perf_counter()-at;phase='current_forward';at=time.perf_counter();passes+=1
            output=native.bridge.supplied(model,history,types,edges,device)
            try:native.bridge.validate_output(output,len(types))
            except Exception:
                arrays.update({'rejected_'+k:output[k] for k in ('prediction','risk','raw_risk')});arrays['rejected_edges']=edges;raise
            native.full.synchronize(device);forward_seconds=time.perf_counter()-at;prediction=output['prediction']
            phase='scoring';truth=np.array(positions[5+step],dtype=np.float32,copy=True)
            arrays['current_prediction']=prediction.copy();arrays['current_target']=truth.copy();arrays['current_edges']=edges.copy()
            require(np.isfinite(truth).all(),'Nonfinite source truth')
            mse=float(np.mean((prediction.astype(np.float64)-truth.astype(np.float64))**2));require(np.isfinite(mse),'Nonfinite rollout error')
            steps.append({'forecast_step':step,**decision,'effective_expansion':audit['retained_optional_pairs']>0,
                'pair_rng_material':core.rng_material(domain,seed,source,step),
                'gate_rng_material':core.rng_material(domain,seed,source,step,gate=True) if policy=='validation_rate_random_gate' else None,
                'graph':audit,'graph_seconds':graph_seconds,'forward_seconds':forward_seconds,'coordinate_mse':mse,
                'predicted_boundary':native.full.boundary_metrics(prediction,metadata['bounds']),
                'ground_truth_boundary':native.full.boundary_metrics(truth,metadata['bounds'])})
            predictions.append(prediction.copy());truths.append(truth)
            if features is not None:feature_trace.append(features)
            if step in (1,10,50,200,395):arrays[f'edges_forecast_{step:04d}']=edges;arrays[f'optional_pairs_forecast_{step:04d}']=optional
            history=np.concatenate((history[1:],prediction[None]),axis=0)
    except native.full.RolloutGuard as error:
        failure={**error.details,'phase':phase,'forecast_step':0 if phase in ('initial_state','initial_graph','native_parity') else len(steps)+1}
    except Exception as error:
        failure=failure_record(error,phase,forecast_step=0 if phase in ('initial_state','initial_graph','native_parity') else len(steps)+1)
        if phase=='native_parity' and not isinstance(error,TimeoutError):failure['category']='native_parity_failure'
    if failure:arrays['failed_input_history']=history.copy()
    completed=len(steps);complete=completed==395 and failure is None;n=len(types)
    arrays.update(prediction=np.stack(predictions) if predictions else np.empty((0,n,2),dtype=np.float32),
        ground_truth=np.stack(truths) if truths else np.empty((0,n,2),dtype=np.float32),
        feature_trace=np.stack(feature_trace) if feature_trace else np.empty((0,11),dtype=np.float64))
    mse=[s['coordinate_mse'] for s in steps]
    row={'schema':ROLLOUT_SCHEMA,**item,'model_seed':seed,'arm':'mix','checkpoint_updates':100000,'horizon':395,
        'status':'complete' if complete else 'failed','failure':failure,'completed_steps':completed,'steps':steps,'mse_per_step':mse,
        'mean_rollout_mse':float(np.mean(mse)) if complete else None,'mse_at_final_horizon':mse[-1] if complete else None,
        'mse_at_declared_trace_steps':{str(s):mse[s-1] if completed>=s else None for s in (1,10,50,200,395)},
        'network_passes':passes,'native_parity_network_passes':parity_passes,'native_parity':parity,
        'network_pass_count_basis':'deployed supplied-forward attempts; parity calls separate; failed parity count unknown',
        'feature_trace_scope':'accepted learned-gate steps only; zero rows for other policies',
        'timing':{'trajectory_processing_seconds':time.perf_counter()-started,'native_parity_seconds':locals().get('parity_seconds')},
        'requested_expansion_fraction':float(np.mean([s['requested_expand'] for s in steps])) if complete else None,
        'effective_expansion_fraction':float(np.mean([s['effective_expansion'] for s in steps])) if complete else None,
        'zero_budget_fraction':float(np.mean([s['graph']['optional_pair_budget']==0 for s in steps])) if complete else None}
    return row,arrays


def remaining(context,started):
    seconds=min(context.args.max_seconds-(time.perf_counter()-started),
                (stamp(context.release['absolute_stop_utc'])-datetime.now(timezone.utc)).total_seconds())
    if seconds<=0:raise TimeoutError('Whole-invocation gate allocation exhausted')
    return seconds


def collection_coverage(expected,rows):
    key=lambda r:(r['source_index'],r.get('target_frame',r.get('policy')))
    committed={key(r):r for r in rows};require(len(committed)==len(rows),'Duplicate committed gate row')
    return [{**item,'state':('committed_complete' if committed[key(item)]['status']=='complete' else 'committed_failed')
             if key(item) in committed else 'never_started'} for item in expected]


def publish_collection(output,name,result,bindings):
    snapshot=tree_snapshot(output,(name,));result={**result,**snapshot}
    # Compare current bytes with the originally committed row/artifact pins,
    # rather than silently adopting a mutation into the final inventory.
    committed=list(result.get('rows',[]))
    if result.get('initial_native_parity') is not None:committed.append(result['initial_native_parity'])
    for row in committed:
        for field,pin in (('row_file','row_sha256'),('artifact_file','artifact_sha256')):
            require(snapshot['files'].get(row[field],{}).get('sha256')==row[pin],
                'Previously committed row/artifact bytes changed')
    if result.get('driver_protocol_sha256') is not None:
        require(snapshot['files'].get('protocol.json',{}).get('sha256')==result['driver_protocol_sha256'],
            'Previously committed protocol bytes changed')
    raw=encode(result)
    verify_inputs(bindings)
    require(tree_snapshot(output,(name,))==snapshot,'Output entry/hash/size snapshot changed before collection publication')
    with (Path(output)/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return result


def execute(context,started):
    args=context.args;mods=context.mods;core=mods.core;bench=mods.bench;output=args.output_dir
    output.mkdir(mode=0o700,exist_ok=False)
    expected=expected_schedule(core,args.mode,context.manifest);rows=[];case_timings=[];current=None;abort=None
    identity=None;runtime={};protocol_sha=None;setup_seconds=None;all_verified=False;model_unchanged=False;model=None;helpers=None;initial_parity=None
    try:
        with bench.deadline_alarm(remaining(context,started)):
            head=json_snapshot(args.head,context.bindings[str(args.head.resolve())])[0] if args.head else None
            if head is not None:core.predict(head,np.zeros(11))
            native,helpers=bench.load_helpers(args.repo)
            bind_inputs(context.bindings,{str((args.repo/path).resolve()):h for path,h in bench.SOURCE_PINS.items()})
            bind_inputs(context.bindings,bench.load_split_evidence(args,context.manifest,context.admission,helpers))
            trajectories=helpers.data_loader.load_manifest_data(args.manifest,verify_hashes=True)
            required_count=1000 if args.split=='train' else 30
            require(len(trajectories)==required_count,'Complete admitted split required')
            for record,(positions,types) in zip(context.manifest['records'],trajectories):
                require(positions.dtype.str=='<f4' and tuple(positions.shape)==tuple(record['positions']['shape'])
                    and positions.shape[0]==401 and positions.shape[2]==2 and np.isfinite(positions).all()
                    and types.dtype.str=='<i8' and types.shape==(positions.shape[1],) and np.all(types==7),'Actual Goop source shape/type/values differ')
                for field in ('positions','particle_types'):
                    path=helpers.data_loader._manifest_array_path(args.manifest.parent,record[field])
                    require(path.stat().st_size==record[field]['size_bytes'],'Actual source byte length differs')
                    bind_inputs(context.bindings,{str(path.resolve()):record[field]['sha256']})
            device,runtime=bench.configure_cuda(helpers,args)
            require(str(runtime['uuid']).lower().removeprefix('gpu-')==str(context.release['gpu_uuid']).lower().removeprefix('gpu-'),'Actual physical CUDA UUID differs')
            model_args=SimpleNamespace(**vars(args),model_kind='checkpoint',checkpoint_updates=100000,training_protocol=args.protocol)
            model,identity=bench.prepare_model(helpers,model_args,context.manifest['metadata'],context.train_admission,device)
            initial_state=helpers.cpu_tree(model.state_dict())
            if 'labels' in args.mode or args.mode=='train-label-capacity':
                item=expected[0];positions,types=trajectories[item['source_index']]
                history=np.array(positions[item['target_frame']-6:item['target_frame']],dtype=np.float32,copy=True)
                _,edges,_,_=native.native_graph(history,'base',None,core.pair_rng(args.split,args.seed,item['source_index'],item['target_frame']))
                at=time.perf_counter();parity,arrays,returned=native.bridge.native_parity(model,history,types,edges,device)
                arrays.update({'returned_'+k:returned[k] for k in ('prediction','risk','raw_risk')})
                passed=parity.get('passed') is True
                initial_parity=publish_row(output,'initial_native_parity',{'schema':SCHEMA,'status':'complete' if passed else 'failed',
                    'failure':None if passed else {'category':'native_parity_failure'},'parity':parity,'network_passes':2,
                    'source_index':item['source_index'],'target_frame':item['target_frame'],'timing':{'processing_seconds':time.perf_counter()-at}},arrays)
                require(passed,'Initial native parity failed; raw returned arrays preserved')
            protocol={'schema':SCHEMA,'mode':args.mode,'split':args.split,'model':identity,'gate_protocol_sha256':GATE_PROTOCOL_SHA,
                'original_training_protocol_sha256':TRAINING_PROTOCOL_SHA,'core_sha256':CORE_SHA,'driver_sha256':sha(__file__),
                'source_manifest_sha256':sha(args.manifest),'cohort_sha256':context.cohort_sha,'schedule':expected,
                'features':list(core.FEATURES),'policies':list(core.POLICIES),'runtime':runtime,'input_sha256':context.bindings,
                'initial_native_parity':initial_parity,'head_sha256':sha(args.head) if args.head else None,
                'selection_sha256':sha(args.selection) if args.selection else None,'selected_model_parameters_trainable':False}
            protocol_sha=write(output/'protocol.json',protocol,exclusive=True)
            setup_seconds=time.perf_counter()-started
        with helpers.torch.no_grad():
            for item in expected:
                current=item;write(output/'status.json',{'schema':SCHEMA,'state':'running','current':item,'committed_rows':len(rows),'required_rows':len(expected)})
                case_started=time.perf_counter()
                with bench.deadline_alarm(remaining(context,started)):
                    positions,types=trajectories[item['source_index']]
                    if 'labels' in args.mode or args.mode=='train-label-capacity':
                        row,arrays=label_row(native,core,model,positions,types,item,args.seed,args.split,device)
                        name=f"source_{item['source_index']:06d}_target_{item['target_frame']:03d}"
                    else:
                        domain='test' if args.mode=='test-rollout' else 'train_rollout'
                        row,arrays=rollout_row(native,core,model,positions,types,context.manifest['metadata'],item,args.seed,head,domain,device)
                        name=f"source_{item['source_index']:06d}_{item['policy']}"
                # An expired calculation returns its accepted prefix. Publish it
                # outside the one-shot alarm; finalization is charged by parent.
                row.update(gate_protocol_sha256=GATE_PROTOCOL_SHA,protocol_sha256=protocol_sha,
                    checkpoint_sha256=args.checkpoint_sha256,source_manifest_sha256=context.bindings[str(args.manifest.resolve())],
                    source_trajectory_content_sha256=context.manifest['records'][item['source_index']]['trajectory_content_sha256'])
                if 'rollout' in args.mode:row.update(head_sha256=context.bindings[str(args.head.resolve())],selection_sha256=context.bindings[str(args.selection.resolve())])
                committed=publish_row(output,name,row,arrays)
                rows.append(committed);case_timings.append({**item,'wall_seconds':time.perf_counter()-case_started})
                print(json.dumps({**item,'status':row['status'],'committed_rows':len(rows)}),flush=True)
                if row['status']!='complete':
                    category=(row.get('failure') or {}).get('category')
                    if args.mode!='test-rollout' or category in ('execution_error','native_parity_failure','execution_budget'):
                        abort={'category':'recorded_case_failure','case':item,'failure':row.get('failure')};break
                current=None
        remaining(context,started)
        model_unchanged=bench.tree_equal(helpers.torch,initial_state,helpers.cpu_tree(model.state_dict()))
        require(model_unchanged,'Frozen simulator state changed during gate work')
        verify_inputs(context.bindings);all_verified=True
    except BaseException as error:
        abort={'category':'execution_budget' if isinstance(error,TimeoutError) else 'execution_error','error_type':type(error).__name__,'error':str(error),'current':current}
        write(output/'failed_attempt.json',{'schema':SCHEMA,'failure':abort,'committed_rows':len(rows),'all_existing_outputs_retained':True})
    complete_count=sum(row['status']=='complete' for row in rows);all_rows=len(rows)==len(expected)
    successful=all_rows and complete_count==len(expected) and all_verified and model_unchanged and abort is None
    guard_complete=args.mode=='test-rollout' and all_rows and all_verified and model_unchanged and abort is None
    status='complete' if successful else 'complete_with_guard_failures' if guard_complete else 'stopped_incomplete_or_failed'
    write(output/'status.json',{'schema':SCHEMA,'state':status,'current':current,'required_rows':len(expected),'committed_rows':len(rows),'complete_rows':complete_count,
        'all_inputs_reverified':all_verified,'model_state_verified_unchanged':model_unchanged,'abort_reason':abort})
    result={'schema':LABEL_COLLECTION if 'labels' in args.mode or args.mode=='train-label-capacity' else ROLLOUT_COLLECTION,
        'status':status,'mode':args.mode,'split':args.split,'model':identity,'gate_protocol_sha256':GATE_PROTOCOL_SHA,
        'original_training_protocol_sha256':TRAINING_PROTOCOL_SHA,'core_sha256':CORE_SHA,'driver_sha256':sha(__file__),
        'cohort_sha256':context.cohort_sha,'cohort_audit_sha256':context.bindings[str(args.cohort_audit.resolve())],
        'training_admission_sha256':context.bindings[str(args.train_admission.resolve())],
        'source_manifest_sha256':context.bindings[str(args.manifest.resolve())],'metadata_sha256':mods.final.METADATA_SHA,'driver_protocol_sha256':protocol_sha,
        'expected_row_ids':[[e['source_index'],e['target_frame']] for e in expected] if 'target_frame' in expected[0] else None,
        'expected_cells':expected,'required_rows':len(expected),'committed_rows':len(rows),'complete_rows':complete_count,
        'all_required_labels_complete':successful if 'target_frame' in expected[0] else None,
        'all_required_outcomes_complete':successful,'rows':rows,'coverage':collection_coverage(expected,rows),
        'initial_native_parity':initial_parity,
        'case_timings':case_timings,'input_sha256':context.bindings,'all_inputs_reverified':all_verified,
        'model_state_verified_unchanged':model_unchanged,'abort_reason':abort,'current_at_stop':current,
        'runtime':{**runtime,'setup_seconds':setup_seconds,'elapsed_before_collection_seconds':time.perf_counter()-started,'max_seconds':args.max_seconds,
            'final_collection_publication_seconds':'measured externally by scoped parent; not known before writing own bytes'},
        'head_sha256':context.bindings[str(args.head.resolve())] if args.head else None,
        'selection_sha256':context.bindings[str(args.selection.resolve())] if args.selection else None,
        'array_scope':'actual numeric bytes retained locally to driver; no schema/endpoint/policy replacement'}
    name='label_collection.json' if result['schema']==LABEL_COLLECTION else 'rollout_collection.json'
    try:publish_collection(output,name,result,context.bindings)
    except BaseException as error:
        write(output/'failed_collection_publication.json',{'schema':SCHEMA,'error_type':type(error).__name__,'error':str(error),'all_existing_outputs_retained':True})
        return 1
    return 0 if successful or guard_complete else 1


def main(argv=None):
    started=time.perf_counter();args=parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','modes':MODES,'gate_protocol_sha256':GATE_PROTOCOL_SHA,
            'source_sha256':SOURCE_PINS,'no_scientific_execution':True},indent=2));return 0
    context=release_gate(args)
    return execute(context,started)


if __name__=='__main__':raise SystemExit(main())
