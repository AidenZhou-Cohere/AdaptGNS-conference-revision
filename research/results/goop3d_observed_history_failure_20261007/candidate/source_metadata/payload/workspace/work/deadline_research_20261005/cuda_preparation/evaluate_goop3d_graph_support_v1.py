#!/usr/bin/env python3
"""Root-gated D3 capacity timing/final evaluation; description by default.

SeparateD3/.025 implementation and schemas. A capacity checkpoint never passes
final cohort admission. Fixed source-order grid and metric schedules are set
before outcomes. No acquisition, training, tuning, promotion or automatic retry.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop3d_graph_support_evaluation_v1'
TRAIN_SCHEMA = 'adaptgns_goop3d_graph_support_cuda_training_v2'
TRAINER_SHA = '8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc'
PROBE_SCHEMA = 'adaptgns_goop3d_vectorized_capacity_checkpoint_v1'
METADATA_SHA = '727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'
TRAIN_MANIFEST_SHA = '0f0ce1802e202b9faaf86f0a6ce059c286ed454ceb53273cb926980614b0f864'
VALID_MANIFEST_SHA = 'f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef'
GRAPH_SHA = 'ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50'
CONVERTER_SHA = '49dece28bf4494591379b8a667e5366b5bdf1609c4dc0757ceb31664c28f9b74'
READER_SHA = 'ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33'
CONTEXT_SHA = '5eb6818ae2699c57e62e80f63724248e8573e4536195eb471df0c647122fb1a5'
CENSUS_SOURCE_SHA = '1b5a8c24fde395b2633ee197557fde3e5bf30ca74908a77f1fb0e011b85e41fb'
NUMERICAL_SOURCE_PINS = {
    'research/graph_convention_bridge.py': '2c589c3c762631de5d3b3d60b986cc71178b97b3a76d0ce0d02132247b0be42d',
    'research/full_rollout.py': 'b0a37ee47619e699298b86865649e63c402cc1cb5dc2fa3dfd4055f7fa966eb8',
    'research/full_same_state.py': 'ff0f9b428791453a1592c23e0c1a4f31653418f74654c859701f52a70f32f592',
    'research/budget_graph.py': '951f0d13863672f1dd248febf8cc960ba500103ca95361e06c9b0577913a8187',
}
POLICIES = ('base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25')
FRAMES, HORIZON = 301, 295
DEADLINE = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def digest(v): return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)


def atomic_json(path, value):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    with temp.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def load(path, expected, name):
    require(sha(path)==expected, 'Bound source bytes differ: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module); return module


def merge_bindings(*groups):
    result={}
    for group in groups:
        for path,value in group.items():
            path=str(Path(path).resolve())
            require(digest(value) and (path not in result or result[path]==value),'Conflicting or invalid input binding: '+path)
            result[path]=value
    return result


def verify_bindings(bindings):
    require(all(sha(path)==value for path,value in bindings.items()),'Inputs changed during D3 evaluation')


def execution_bindings(args,release,trainer,release_sha):
    numerical=release.get('numerical_source_sha256',{})
    require(all(numerical.get(k)==v for k,v in NUMERICAL_SOURCE_PINS.items()),'Pinned transitive numerical sources required')
    result=merge_bindings(release['files_sha256'],
        {args.release:release_sha,__file__:release['evaluator_sha256']},
        {args.repo/path:value for path,value in numerical.items()},
        {args.repo/path:value for path,value in trainer.SOURCE_PINS.items()})
    verify_bindings(result)
    return result


def runtime_gate(args,release,runtime):
    normalize=lambda value:str(value).lower().removeprefix('gpu-')
    require(runtime.get('device')==f'cuda:{args.cuda_index}'
            and normalize(runtime.get('uuid'))==normalize(args.gpu_uuid)==normalize(release['gpu_uuid']),
            'Released physical CUDA device differs from runtime')


def grid_indices(count):
    require(type(count) is int and count>0, 'Complete positive source record count required')
    return list(range(count)) if count<=30 else [j*(count-1)//29 for j in range(30)]


def schedules(records, mode, purpose):
    require(all(r['source_index']==i and r['positions']['shape'][0]==FRAMES and r['positions']['shape'][2]==3 for i,r in enumerate(records)), 'Complete orderedT301/D3 records required')
    indices=grid_indices(len(records))
    if purpose=='capacity_timing':
        ranked=sorted(indices,key=lambda i:(records[i]['positions']['shape'][1],i))
        indices=sorted({ranked[0],ranked[(len(ranked)-1)//2],ranked[-1]})
    if mode=='full-rollout':
        return [{'source_index':i,'trajectory_id':records[i]['id'],'particles':records[i]['positions']['shape'][1],
                 'size_group':'fixed_source_grid' if purpose=='final_evaluation' else 'bounded_size_timing'} for i in indices]
    if mode=='clean-validation':
        eligible=len(indices)*HORIZON; flat=[i*(eligible-1)//127 for i in range(128)]
        require(len(set(flat))==128,'128 distinct clean validation histories required')
        pairs=[(indices[i//HORIZON],i%HORIZON+6) for i in flat]
    else:
        targets=(7,80,153,226,300) if purpose=='final_evaluation' else (7,)
        pairs=[(i,t) for i in indices for t in targets]
    return [{'schedule_index':k,'source_index':i,'target_frame':t,'trajectory_id':records[i]['id']} for k,(i,t) in enumerate(pairs)]


def cohort_gate(cohort, audit, release, args):
    require(cohort.get('schema')=='adaptgns_goop3d_graph_support_final_cohort_v2' and cohort.get('status')=='frozen_for_final_evaluation'
            and cohort.get('issued_by')=='root' and cohort.get('training_schema')==TRAIN_SCHEMA
            and type(cohort.get('endpoint_updates')) is int and cohort['endpoint_updates']>=2
            and cohort['endpoint_updates']==release.get('scientific_endpoint_updates')
            and cohort.get('protocol_sha256')==sha(args.protocol) and cohort.get('trainer_sha256')==sha(args.trainer_source)
            and cohort.get('graph_sha256')==GRAPH_SHA and cohort.get('cohort_audit_sha256')==sha(args.cohort_audit), 'Frozen prospectiveD3v2 cohort required')
    models=cohort.get('models',[]); expected={(a,s) for a in ('base','mix') for s in range(3)}
    require(len(models)==6 and {(r.get('arm'),r.get('seed')) for r in models}==expected
            and all(r.get('completed_steps')==cohort['endpoint_updates'] and r.get('objective')=='faithful' and digest(r.get('checkpoint_sha256')) for r in models)
            and len({r['checkpoint_sha256'] for r in models})==6,'Six distinctD3 final checkpoints required')
    require(audit.get('schema')=='adaptgns_goop3d_graph_support_complete_cohort_audit_v2' and audit.get('issued_by')=='root'
            and audit.get('status')=='all_six_endpoints_and_pairing_verified' and audit.get('training_schema')==TRAIN_SCHEMA
            and audit.get('endpoint_updates')==cohort['endpoint_updates'] and audit.get('protocol_sha256')==cohort['protocol_sha256']
            and audit.get('trainer_sha256')==cohort['trainer_sha256'] and audit.get('graph_sha256')==GRAPH_SHA,'CompleteD3 endpoint/paired audit required')
    checked=audit.get('models',[])
    require(len(checked)==6 and {(r.get('arm'),r.get('seed')) for r in checked}==expected,'FullD3 endpoint audit coverage required')
    for model in models:
        row=next(r for r in checked if (r['arm'],r['seed'])==(model['arm'],model['seed']))
        require(row.get('checkpoint_sha256')==model['checkpoint_sha256'] and row.get('completed_steps')==row.get('graph_history_updates')==cohort['endpoint_updates']
                and all(row.get(k) is True for k in ('all_optimizer_steps_equal_endpoint','all_state_and_moments_finite','source_data_protocol_verified','checkpoint_bytes_verified')),'IncompleteD3 endpoint audit')
    pairs=audit.get('paired_seeds',[])
    require(len(pairs)==3 and {r.get('seed') for r in pairs}=={0,1,2}
            and all(all(r.get(k) is True for k in ('initial_model_tensor_identity','initial_cpu_cuda_rng_identity','all_frame_noise_lr_schedules_equal','all_graph_budgets_and_rng_material_verified')) for r in pairs),'FullD3 pairing audit required')
    selected=next(r for r in models if (r['arm'],r['seed'])==(args.arm,args.seed))
    require(selected['checkpoint_sha256']==args.checkpoint_sha256,'SelectedD3 checkpoint differs from frozen cohort')
    return cohort['endpoint_updates']


def release_gate(args):
    r=read(args.release)
    require(r.get('schema')=='adaptgns_goop3d_evaluation_release_v1' and r.get('status')=='admitted_for_execution'
            and r.get('issued_by')=='root' and r.get('purpose')==args.purpose and r.get('mode')==args.mode and r.get('split')==args.split
            and r.get('arm')==args.arm and r.get('seed')==args.seed and r.get('checkpoint_sha256')==args.checkpoint_sha256
            and r.get('checkpoint_updates')==args.checkpoint_updates and r.get('graph_sha256')==GRAPH_SHA
            and r.get('evaluator_sha256')==sha(__file__) and r.get('whole_invocation_outer_timeout_required') is True
            and r.get('cuda_index')==args.cuda_index and r.get('gpu_uuid')==args.gpu_uuid
            and r.get('max_seconds')==args.max_seconds,'Exact reviewedD3 execution release required')
    require(sha(args.trainer_source)==TRAINER_SHA,'Exact reviewedD3v2 trainer source required')
    if args.purpose=='final_evaluation':
        require(all(r.get('files_sha256',{}).get(str(p.resolve()))==sha(p) for p in (args.cohort,args.cohort_audit)), 'Root-bound cohort/audit files required before test reads')
        endpoint=cohort_gate(read(args.cohort),read(args.cohort_audit),r,args)
        require(endpoint==args.checkpoint_updates,'Scientific endpoint differs')
    fields=('manifest','split_admission','structural_report','acquisition_report','context_semantics','auxiliary_report','trainer_source','protocol','checkpoint')
    paths=[getattr(args,k).resolve() for k in fields]+[HERE/'goop3d_native_evaluation_v1.py',HERE/'goop3d_diagnostic_metrics_v1.py',HERE/'goop3d_graph_support_vectorized_v1.py']
    if args.purpose=='final_evaluation': paths += [args.cohort.resolve(),args.cohort_audit.resolve()]
    if args.split=='test': paths.append(args.cross_split_audit.resolve())
    require(all(r.get('files_sha256',{}).get(str(p))==sha(p) for p in paths),'Exact execution file bindings required')
    require(datetime.now(timezone.utc)<DEADLINE and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
            and os.environ.get('CUDA_VISIBLE_DEVICES') is None,'Cutoff/deterministic environment required')
    if args.purpose=='capacity_timing':
        require(args.split=='valid' and args.checkpoint_updates==512 and r.get('scientific_training_admitted') is False,
                'Only distinct512 probe schema on validation is supported for timing')
    return r


def check_split(args, release, helpers):
    # All final cohort gates run before this function first opens a test manifest.
    m,a,report=[read(getattr(args,k)) for k in ('manifest','split_admission','structural_report')]
    require(a.get('schema')=='adaptgns_goop3d_evaluation_split_admission_v1' and a.get('status')=='admitted' and a.get('issued_by')=='root'
            and a.get('split')==args.split and a.get('dataset')=='Goop-3D' and a.get('manifest_sha256')==sha(args.manifest)
            and a.get('structural_report_sha256')==sha(args.structural_report) and a.get('metadata_sha256')==METADATA_SHA
            and a.get('context_semantics_sha256')==CONTEXT_SHA and a.get('acquisition_report_sha256')==sha(args.acquisition_report)
            and a.get('auxiliary_report_sha256')==sha(args.auxiliary_report) and a.get('frames')==301 and a.get('dimension')==3,
            'Separate completeD3 split admission required')
    if args.split=='valid': require(sha(args.manifest)==VALID_MANIFEST_SHA,'Original completeD3 validation manifest differs')
    else:
        require(args.purpose=='final_evaluation' and a.get('cohort_sha256')==sha(args.cohort)
                and a.get('reserved_test_acquired_after_cohort_freeze') is True
                and a.get('cross_split_audit_sha256')==sha(args.cross_split_audit),'ReservedD3 test requires future complete source/cohort/overlap admission')
        cross=read(args.cross_split_audit)
        require(cross.get('schema')=='adaptgns_goop3d_all_split_integrity_audit_v1' and cross.get('issued_by')=='root'
                and cross.get('status')=='all_required_splits_verified' and cross.get('duplicate_pairs')==[]
                and cross.get('manifest_sha256')=={'train':TRAIN_MANIFEST_SHA,'valid':VALID_MANIFEST_SHA,'test':sha(args.manifest)},'CompleteD3 split integrity required')
    require(m.get('format')=='gns-trajectory-manifest' and m.get('version')==1 and m.get('dataset')=='Goop-3D'
            and m.get('split')==args.split and m.get('metadata_sha256')==METADATA_SHA and m.get('reader_sha256')==READER_SHA
            and m.get('converter_sha256')==a.get('converter_sha256') and len(m.get('records',[]))==m.get('record_count')==a.get('record_count')
            and a.get('particle_type_ids')==[7],'Complete actualD3 source/count/type lineage required')
    root=args.manifest.resolve().parent; metadata_path=root/'metadata.json'
    require(sha(metadata_path)==METADATA_SHA and read(metadata_path)==m['metadata'],'ExactD3 metadata bytes required')
    metadata=m['metadata']; source=m['source']; detail=report.get('splits',{}).get(args.split,{})
    require(metadata.get('dim')==3 and metadata.get('sequence_length')==300 and metadata.get('default_connectivity_radius')==.025
            and metadata.get('dt')==.0025 and 'context_mean' not in metadata and 'context_std' not in metadata,'D3 geometry/context differs')
    require(report.get('schema')=='official_goop3d_numeric_preparation_v1' and report.get('status')=='complete_structural_only'
            and report.get('wrapper_sha256')==a['converter_sha256'] and detail.get('manifest_sha256')==sha(args.manifest)
            and detail.get('record_count')==len(m['records']) and detail.get('frame_lengths')==[301]
            and detail.get('particle_type_ids')==[7] and all(detail.get(k) is True for k in ('source_EOF_SHA256_verified','whole_object_crc32c_verified',
                'TFRecord_length_and_payload_CRC32C_verified','all_match_official_parser_frame_count','all_source_array_bytes_preserved_exact')),'CompleteD3 source preservation evidence required')
    receipt=read(args.acquisition_report); selected=[v for v in receipt.get('files',[]) if v.get('name')==args.split+'.tfrecord']
    require(receipt.get('status')=='complete' and receipt.get('dataset')=='Goop-3D' and len(selected)==1,'Complete acquiredD3 split required')
    received=selected[0]
    require(source.get('sha256')==a.get('source_sha256')==received.get('sha256') and source.get('generation')==received.get('generation')
            and source.get('size_bytes')==received.get('received_bytes') and source.get('crc32c_base64')==received.get('crc32c_base64')
            and source.get('CRC_verified') is True and received.get('crc32c_verified') is True,'ReceivedD3 source bytes differ')
    aux=read(args.auxiliary_report); context=read(args.context_semantics)
    require(sha(args.context_semantics)==CONTEXT_SHA and aux.get('schema')=='adaptgns_goop3d_auxiliary_census_v1'
            and aux.get('status')=='all_preserved_auxiliary_bytes_verified' and aux.get('splits',{}).get(args.split,{}).get('manifest_sha256')==sha(args.manifest),'CompleteD3 context/census evidence required')
    context_root=args.context_semantics.resolve().parent/'goop_context_semantics_sources'
    require(all(sha(context_root/name)==row['sha256'] for name,row in context['sources'].items()),'Official context source bytes differ')
    census_source=load(HERE/'audit_goop3d_auxiliary.py',CENSUS_SOURCE_SHA,'_goop3d_eval_auxiliary')
    auxrows=aux['splits'][args.split]['records']; require(len(auxrows)==len(m['records']),'Complete auxiliary census rows required')
    offset=0; files=merge_bindings({metadata_path:METADATA_SHA,HERE/'audit_goop3d_auxiliary.py':CENSUS_SOURCE_SHA},
        {context_root/name:row['sha256'] for name,row in context['sources'].items()})
    for i,(record,expected_aux) in enumerate(zip(m['records'],auxrows)):
        require(record.get('id')==f'{args.split}:{i:06d}' and record.get('source_index')==i and record.get('source_offset_bytes')==offset
                and record.get('positions',{}).get('shape',[None])[0]==301 and record['positions']['shape'][2]==3,'D3 source record order/shape differs')
        offset += record['record_payload_bytes']+16
        actual,p=census_source.census_record(root,args.split,record,helpers.np); require(actual==expected_aux,'PreservedD3 auxiliary bytes/census differ')
        if p is not None: files[str(p)]=record['step_context']['sha256']
        for key in ('positions','particle_types'):
            d=record[key]; path=helpers.data_loader._manifest_array_path(root,d)
            require(path.stat().st_size==d['size_bytes'] and sha(path)==d['sha256'],'D3 numeric bytes differ')
            files[str(path)]=d['sha256']
    require(offset==source['size_bytes'],'AllD3 source offsets must cover complete object')
    trajectories=helpers.data_loader.load_manifest_data(args.manifest,verify_hashes=True)
    require(len(trajectories)==len(m['records']) and all(p.shape==tuple(r['positions']['shape']) and p.dtype.str=='<f4'
            and helpers.np.isfinite(p).all() and t.shape==(p.shape[1],) and helpers.np.all(t==7)
            for r,(p,t) in zip(m['records'],trajectories)),'ActualD3 trajectory/type contract differs')
    return m,trajectories,files


def check_payload(payload,args,release,trainer):
    if args.purpose=='capacity_timing':
        require(payload.get('schema')==PROBE_SCHEMA and payload.get('purpose')=='bounded_capacity_only_never_promote'
                and payload.get('scientific_training_admitted') is False and payload.get('completed_steps')==512
                and payload.get('probe_lr_horizon')==100000 and payload.get('arm')==args.arm and payload.get('seed')==args.seed
                and payload.get('trainer_sha256')==sha(args.trainer_source) and payload.get('graph_sha256')==GRAPH_SHA
                and payload.get('train_manifest_sha256')==TRAIN_MANIFEST_SHA
                and not any(k in payload for k in ('cuda_goop3d_graph_support_schema','training_config','run_config')),
                'Distinct exact512 capacity checkpoint required; no scientific checkpoint promotion')
        return 100000
    config=payload.get('run_config',{})
    require(payload.get('format_version')==2 and payload.get('cuda_goop3d_graph_support_schema')==TRAIN_SCHEMA
            and not any(k in payload for k in ('schema','cuda_goop_graph_support_schema','cuda_sand_graph_support_schema','graph_support_schema','full_training_schema'))
            and payload.get('completed_steps')==args.checkpoint_updates and config.get('schema')==TRAIN_SCHEMA
            and config.get('dataset')=='Goop-3D' and config.get('arm')==args.arm and config.get('seed')==args.seed
            and config.get('objective')=='faithful' and config.get('updates')==args.checkpoint_updates
            and config.get('research_protocol_sha256')==sha(args.protocol)
            and payload.get('run_config_sha256')==trainer.config_hash(config)
            and config.get('source_sha256',{}).get('train_goop3d_graph_support_cuda_v2.py')==sha(args.trainer_source)
            and config['source_sha256'].get('goop3d_graph_support_vectorized_v1.py')==GRAPH_SHA
            and config.get('data',{}).get('manifest_sha256')==TRAIN_MANIFEST_SHA,'Exact prospectiveD3v2 scientific checkpoint required')
    trainer.validate_graph_history(payload['history'],args.checkpoint_updates,args.arm,args.seed)
    return args.checkpoint_updates


def prepare_model(args,release,trainer,helpers,metadata,device):
    torch=helpers.torch; require(sha(args.checkpoint)==args.checkpoint_sha256,'Selected checkpoint bytes differ')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True); endpoint=check_payload(payload,args,release,trainer)
    torch.manual_seed(args.seed)
    if device.type=='cuda':
        with torch.cuda.device(device): torch.cuda.manual_seed(args.seed)
    model=helpers.build_simulator(metadata,6.7e-4,6.7e-4,device,connectivity_radius=.025,nmessage_passing_steps=10,
        uncertainty_parameterization='variance',variance_floor=1e-6,detach_variance_features=True,radius_backend='scipy_host').to(device)
    require(model._checkpoint_config['particle_dimensions']==3 and model._checkpoint_config['nnode_in']==37 and model._checkpoint_config['nedge_in']==4
            and trainer.tree_equal(torch,payload['simulator_config'],helpers.cpu_tree(model._checkpoint_config)),'ExactD3 simulator/normalization configuration required')
    model.load_state_dict(payload['state_dict'],strict=True)
    require(trainer.tree_equal(torch,payload['state_dict'],helpers.cpu_tree(model.state_dict())) and helpers.tensors_are_finite(model.state_dict().values()),'SelectedD3 model tensors differ/nonfinite')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    optimizer.load_state_dict(payload['optimizer_state']); trainer.assert_adam(torch,model,optimizer,args.checkpoint_updates,endpoint)
    require(sha(args.checkpoint)==args.checkpoint_sha256,'Checkpoint changed during load')
    return model.eval()


@contextmanager
def alarm(seconds):
    require(seconds>0,'Invocation budget exhausted')
    require(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'Existing alarm conflicts')
    previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('D3 invocation budget exhausted')))
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try: yield
    finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    p.add_argument('--purpose',choices=('capacity_timing','final_evaluation'));p.add_argument('--mode',choices=('full-rollout','same-state','clean-validation'))
    p.add_argument('--split',choices=('valid','test'));p.add_argument('--arm',choices=('base','mix'));p.add_argument('--seed',type=int,choices=(0,1,2))
    p.add_argument('--checkpoint-sha256');p.add_argument('--checkpoint-updates',type=int);p.add_argument('--cuda-index',type=int,default=0);p.add_argument('--gpu-uuid')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--max-seconds',type=int,default=900)
    names=('release','repo','manifest','split-admission','structural-report','acquisition-report','context-semantics','auxiliary-report',
           'trainer-source','protocol','checkpoint','cohort','cohort-audit','cross-split-audit','output-dir')
    for name in names:p.add_argument('--'+name,type=Path)
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','purpose':['capacity_timing','final_evaluation'],
            'grid':'allN ifN<=30 elsefloor(j*(N-1)/29); fixedsourceorder; notrepresentativeprobabilitysample',
            'horizon':HORIZON,'policies':POLICIES,'same_state_targets':[7,80,153,226,300],
            'outer_owned_process_timeout_required':True,'scientific_endpoint_selected':False},indent=2));return 0
    required=('purpose','mode','split','arm','seed','checkpoint_sha256','checkpoint_updates','gpu_uuid',*(n.replace('-','_') for n in names if n not in ('cohort','cohort-audit','cross-split-audit')))
    require(all(getattr(args,k) is not None for k in required),'Every execution/data/source argument required')
    require(digest(args.checkpoint_sha256) and 1<=args.max_seconds<=7200 and 1<=args.threads<=16 and args.cuda_index>=0,'Invalid execution bound')
    require(args.purpose!='final_evaluation' or (args.cohort and args.cohort_audit),'FinalD3cohort/audit required')
    require(args.split!='test' or args.cross_split_audit,'ReservedD3test requires fullsplitintegrityaudit')
    require(args.mode!='clean-validation' or args.split=='valid','Cleanvalidationneverusestest')
    release_sha=sha(args.release);release=release_gate(args)
    trainer=load(args.trainer_source,release['files_sha256'][str(args.trainer_source.resolve())],'_goop3d_eval_trainer_v2')
    require(trainer.SCHEMA==TRAIN_SCHEMA,'Only explicitD3v2trainer accepted')
    helpers,_=trainer.load_helpers(args.repo)
    helpers=SimpleNamespace(**vars(helpers));helpers.unpack_batch=lambda examples:trainer.unpack_batch(helpers,examples)
    # Retain every source and release binding through the terminal check.
    bindings=execution_bindings(args,release,trainer,release_sha)
    native=load(HERE/'goop3d_native_evaluation_v1.py',release['files_sha256'][str(HERE/'goop3d_native_evaluation_v1.py')],'_goop3d_eval_native')
    diagnostic=load(HERE/'goop3d_diagnostic_metrics_v1.py',release['files_sha256'][str(HERE/'goop3d_diagnostic_metrics_v1.py')],'_goop3d_eval_metrics')
    output=args.output_dir.resolve(); require(not output.exists(),'Fresh output only; no recovery or retry')
    protected=(args.manifest.resolve().parent,args.checkpoint.resolve().parent,(args.repo/'adaptive-gns').resolve())
    require(all(output!=x and x not in output.parents and output not in x.parents for x in protected),'Outputmustbeseparatefromdata/model/source')
    output.mkdir(parents=True);started=time.perf_counter();rows=[];current=None;extra_files={}
    try:
        manifest,trajectories,extra_files=check_split(args,release,helpers)
        bindings=merge_bindings(bindings,extra_files)
        device,runtime=trainer.configure_cuda(helpers,args)
        runtime_gate(args,release,runtime)
        model=prepare_model(args,release,trainer,helpers,manifest['metadata'],device)
        expected=schedules(manifest['records'],args.mode,args.purpose)
        protocol={'schema':SCHEMA,'purpose':args.purpose,'mode':args.mode,'split':args.split,'arm':args.arm,'seed':args.seed,
            'checkpoint_sha256':args.checkpoint_sha256,'checkpoint_updates':args.checkpoint_updates,'schedule':expected,'policies':POLICIES,
            'source_population_count':len(manifest['records']),'prospective_source_grid':grid_indices(len(manifest['records'])),'selected_source_indices':sorted({i['source_index'] for i in expected}),
            'horizon':HORIZON,'frames':FRAMES,'runtime':runtime,'release_sha256':release_sha,'input_files_sha256':bindings,
            'setup_seconds':time.perf_counter()-started,'scope':'infrastructureonly' if args.purpose=='capacity_timing' else 'fixedsourcegridpairedscientificstudy'}
        atomic_json(output/'protocol.json',protocol);protocol_sha=sha(output/'protocol.json')
        for item in expected:
            policies=POLICIES if args.mode=='full-rollout' else (None,)
            for policy in policies:
                current={**item,**({'policy':policy,'horizon':HORIZON} if policy else {})};atomic_json(output/'status.json',{'state':'running','current':current,'committed_rows':len(rows)})
                remaining=min(args.max_seconds-(time.perf_counter()-started),(DEADLINE-datetime.now(timezone.utc)).total_seconds())
                positions,types=trajectories[item['source_index']]
                with alarm(remaining),helpers.torch.no_grad():
                    call=time.perf_counter();helpers.synchronize(device)
                    if policy:
                        row,arrays=native.rollout(model,positions,types,manifest['metadata'],policy,HORIZON,93000+1000*args.seed+item['source_index'],device,trace_steps=(1,10,50,200,HORIZON))
                    elif args.mode=='same-state':row,arrays=diagnostic.same_state(native,model,positions,types,manifest['metadata'],item,args.split,args.seed,device)
                    else:row,arrays=diagnostic.clean_validation(native,helpers,model,positions,types,item,device)
                    helpers.synchronize(device);call_seconds=time.perf_counter()-call
                require(all(isinstance(a,helpers.np.ndarray) and not a.dtype.hasobject for a in arrays.values()),'Numericonlyartifactsrequired')
                name=f'trajectory_{item["source_index"]:06d}_'+(policy if policy else f'target_{item["target_frame"]:03d}')
                artifact=output/(name+'.npz');temporary=artifact.with_suffix('.npz.tmp');io=time.perf_counter()
                with temporary.open('xb') as f:helpers.np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
                temporary.replace(artifact)
                row.update(**item,arm=args.arm,training_seed=args.seed,objective='faithful',protocol_sha256=protocol_sha,
                    artifact_file=artifact.name,artifact_sha256=sha(artifact),synchronized_call_seconds=call_seconds,publication_seconds=time.perf_counter()-io)
                if policy:row['mse_at_declared_trace_steps']={str(s):row['mse_per_step'][s-1] if row['completed_steps']>=s else None for s in (1,10,50,200,HORIZON)}
                atomic_json(artifact.with_suffix('.json'),row);rows.append(row)
                if not policy:atomic_json(output/'summary.json',diagnostic.summarize(expected,rows,args.mode))
                else:atomic_json(output/'result.json',{'schema':SCHEMA,'protocol_sha256':protocol_sha,'rows':rows,'expected_outcomes':len(expected)*6})
                if (row.get('failure') or {}).get('category') in ('native_parity_failure','execution_error'):
                    raise RuntimeError('D3implementation/parityfailure retained; stopfurtherinference')
        verify_bindings(bindings)
        state='complete' if all(r['status']=='complete' for r in rows) else 'complete_with_guard_failures'
        atomic_json(output/'status.json',{'schema':SCHEMA,'state':state,'committed_rows':len(rows),'expected_rows':len(expected)*(6 if args.mode=='full-rollout' else 1),
            'all_inputs_reverified':True,'total_execution_seconds':time.perf_counter()-started,'full_horizon_forecast_admitted':False})
    except BaseException as error:
        atomic_json(output/'failed_attempt.json',{'schema':SCHEMA,'current':current,'committed_rows':len(rows),'error_type':type(error).__name__,'error':str(error),'all_outputs_retained':True})
        atomic_json(output/'status.json',{'schema':SCHEMA,'state':'error','error_type':type(error).__name__,'committed_rows':len(rows),'total_execution_seconds':time.perf_counter()-started})
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
