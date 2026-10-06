#!/usr/bin/env python3
"""Synthetic terminal-ledger I/O cost only; never writes a training checkpoint.

One root-released worker per assigned GPU; root owns concurrent4+2 launches.
No model construction, forward/backward/update, trajectory data or resume.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import traceback

sys.dont_write_bytecode=True
SCHEMA='adaptgns_sand_synthetic_terminal_cost_v1'
RELEASE_SCHEMA='adaptgns_sand_terminal_cost_release_v1'
TRAINER_SHA='fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
REPETITIONS=3
TERMINAL_RECORDS=100000
MAX_SECONDS=900
HASH_FIELDS=('native_edge_sha256','optional_pair_sha256','noisy_current_sha256')
ENVIRONMENT={'CUBLAS_WORKSPACE_CONFIG':':4096:8','LD_LIBRARY_PATH':'/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
def synthetic_digest(*parts):return hashlib.sha256(('SYNTHETIC_TERMINAL_IO_ONLY:'+':'.join(map(str,parts))).encode()).hexdigest()
def fresh_string(value):return value.encode().decode()

def synthetic_history(probe_history,seed,arm,count=TERMINAL_RECORDS):
    """Repeat shape/count patterns, never evidence of scientific updates.

    Fresh unique hashes prevent pickle identity memoization from understating
    the terminal ledger. Coins/pair hashes are explicitly synthetic metadata.
    """
    require(type(count) is int and count>0 and arm in ('base','mix'),'Invalid synthetic metadata request')
    originals=probe_history['graph_updates'];scalars=probe_history['training']
    require(len(originals)==len(scalars)==512,'Only complete512probe histories accepted')
    graphs=[];training=[]
    for index in range(count):
        old=originals[index%512]
        row={'completed_steps':index+1,'absolute_schedule_step':index,
             'frame_ids':[fresh_string(x) for x in old['frame_ids']],
             'noise_sha256':synthetic_digest(index,'noise'),'examples':[], 'SYNTHETIC_METADATA_ONLY':True}
        for slot,example in enumerate(old['examples']):
            value=copy.deepcopy(example)
            value['coin_seed_material']=[20261005,seed,index,slot,4409]
            value['pair_seed_material']=[20261005,seed,index,slot,5501]
            value['expanded']=arm=='mix' and value['exposure_coin']
            value['selected_optional_pairs']=value['optional_budget_if_exposed'] if value['expanded'] else 0
            for key in HASH_FIELDS:value[key]=synthetic_digest(index,slot,key)
            row['examples'].append(value)
        graphs.append(row)
        if index==0 or (index+1)%100==0 or index+1==count:
            value=copy.deepcopy(scalars[index%512]);value.update(completed_steps=index+1,
                frame_ids=list(row['frame_ids']),lr=1e-4*(1e-5/1e-4)**(index/99999),elapsed_seconds=float(index+1))
            training.append(value)
    return {'training':training,'graph_updates':graphs,'elapsed_seconds':float(count)}

def tensor_tree(torch,value,device):
    if torch.is_tensor(value):return value.detach().to(device)
    if isinstance(value,dict):return {k:tensor_tree(torch,v,device) for k,v in value.items()}
    if isinstance(value,list):return [tensor_tree(torch,v,device) for v in value]
    if isinstance(value,tuple):return tuple(tensor_tree(torch,v,device) for v in value)
    return value

def tensor_inventory(torch,value,path='root'):
    if torch.is_tensor(value):
        raw=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return [{'path':path,'dtype':str(value.dtype),'shape':list(value.shape),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]
    if isinstance(value,dict):return [r for k,v in value.items() for r in tensor_inventory(torch,v,path+'/'+str(k))]
    if isinstance(value,(list,tuple)):return [r for k,v in enumerate(value) for r in tensor_inventory(torch,v,path+'/'+str(k))]
    return []

def envelope(cpu_tensors,history,probe_sha):
    return {'schema':SCHEMA,'NOT_A_TRAINING_CHECKPOINT':True,'synthetic_metadata_records':len(history['graph_updates']),
            'actual_probe_completed_optimizer_updates':512,'source_probe_checkpoint_sha256':probe_sha,
            'fixed_probe_tensor_bytes':cpu_tensors,'synthetic_graph_history':history}

def publish_synthetic_artifacts(torch,out,rep,cpu_tensors,history,probe_sha):
    """Time the synthetic write/hash/pointer/status/history path, including fsync.

    Artifact schemas/names are deliberately incompatible with training resume.
    Model/Adam predicates and live RNG capture are outside this measured scope.
    """
    binary=out/f'synthetic_terminal_envelope_{rep}.bin';temporary=binary.with_suffix('.bin.tmp')
    require(not binary.exists(),'Refusing to replace a synthetic envelope')
    t0=time.perf_counter()
    with temporary.open('xb') as stream:
        torch.save(envelope(cpu_tensors,history,probe_sha),stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(binary);t1=time.perf_counter()
    binary_sha=sha(binary);t2=time.perf_counter()
    pointer_path=out/f'synthetic_envelope_pointer_{rep}.json'
    synthetic_pointer={'schema':SCHEMA,'NOT_A_TRAINING_CHECKPOINT':True,'binary_file':binary.name,'binary_sha256':binary_sha,
                       'synthetic_metadata_records':len(history['graph_updates']),'actual_probe_completed_optimizer_updates':512}
    atomic(pointer_path,synthetic_pointer);t3=time.perf_counter()
    synthetic_status={'schema':SCHEMA,'state':'synthetic_publication_only','NOT_A_TRAINING_CHECKPOINT':True,'pointer':synthetic_pointer}
    status_paths=[out/f'synthetic_publication_status_{rep}.json',out/f'synthetic_attempt_status_{rep}.json']
    for path in status_paths:atomic(path,synthetic_status)
    t4=time.perf_counter();history_path=out/f'synthetic_terminal_history_{rep}.json';atomic(history_path,history);t5=time.perf_counter()
    return {'envelope_serialization_fsync_seconds':t1-t0,'binary_sha256_seconds':t2-t1,
            'synthetic_pointer_fsync_seconds':t3-t2,'synthetic_status_fsync_seconds':t4-t3,'history_json_fsync_seconds':t5-t4,
            'artifact_publication_seconds':t5-t0,'binary_file':binary.name,'binary_sha256':binary_sha,
            'history_file':history_path.name,'synthetic_pointer_file':pointer_path.name,
            'synthetic_status_files':[path.name for path in status_paths]}

def probe_inputs(directory):
    names={'probe_protocol':'protocol.json','probe_status':'status.json','probe_history':'history.json','probe_latest':'latest.json',
           'checkpoint':'checkpoint-000000512.pt'}
    return {key:directory/name for key,name in names.items()}

def verify_probe_metadata(directory,trainer):
    cfg,status,history,pointer=[read(directory/name) for name in ('protocol.json','status.json','history.json','latest.json')]
    digest=trainer.config_hash(cfg)
    require(cfg.get('schema')==trainer.SCHEMA and cfg.get('arm') in ('base','mix') and cfg.get('seed') in (0,1,2)
            and cfg.get('objective')=='faithful' and cfg.get('updates')==100000 and cfg.get('log_every')==1
            and cfg.get('source_sha256')=={**trainer.SOURCE_PINS,'train_sand_graph_support_cuda.py':TRAINER_SHA},'Expected fixed graph-support infrastructure recipe')
    require(status.get('state')=='planned_stop_incomplete' and status.get('completed_steps')==status.get('committed_steps')==512
            and status.get('error') is None and status.get('run_config_sha256')==digest and not(directory/'run.lock').exists(),
            'Only completed fresh512 infrastructure accepted')
    require(pointer=={'path':'checkpoint-000000512.pt','sha256':sha(directory/'checkpoint-000000512.pt'),'completed_steps':512,'run_config_sha256':digest}
            and status.get('latest_checkpoint')==pointer,'Probe checkpoint/pointer/configuration differs')
    trainer.validate_graph_history(history,512,cfg['arm'],cfg['seed'])
    require(len(history['training'])==512 and [r['completed_steps'] for r in history['training']]==list(range(1,513)),'Complete512scalar history required')
    return cfg,history,pointer

def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for name in ('probe-dir','trainer','protocol','release','output-dir'):p.add_argument('--'+name,type=Path)
    p.add_argument('--cuda-index',type=int)
    a=p.parse_args(argv)
    if a.execute and any(getattr(a,k) is None for k in ('probe_dir','trainer','protocol','release','output_dir','cuda_index')):p.error('Explicit probe/trainer/protocol/root-release/output/GPU required')
    if a.cuda_index is not None and a.cuda_index<0:p.error('Nonnegative GPU index required')
    return a

def main(argv=None):
    a=parse(argv)
    if not a.execute:
        print(json.dumps({'schema':SCHEMA,'execute':False,'synthetic_records':TERMINAL_RECORDS,'repetitions':REPETITIONS,
                          'scientific_training_admitted':False,'real_optimizer_updates':0}));return 0
    require(sys.platform.startswith('linux'),'Linux host RSS/CUDA execution required')
    require(datetime.now(timezone.utc)<datetime(2026,10,7,1,tzinfo=timezone.utc),'Compute/analysis deadline reached')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') is None and all(os.environ.get(k)==v for k,v in ENVIRONMENT.items()),'Required native CUDA environment differs')
    require(sha(a.trainer)==TRAINER_SHA,'Frozen trainer definitions changed')
    spec=importlib.util.spec_from_file_location('_terminal_cost_trainer',a.trainer);trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    directory=a.probe_dir.resolve();files={**probe_inputs(directory),'trainer':a.trainer.resolve(),'protocol':a.protocol.resolve(),'supervisor':Path(__file__).resolve(),'python':Path(sys.executable).resolve()}
    hashes={k:sha(v) for k,v in files.items()};release=read(a.release)
    require(release.get('schema')==RELEASE_SCHEMA and release.get('status')=='admitted_for_infrastructure_io' and release.get('issued_by')=='root'
            and release.get('scientific_training_admitted') is False and release.get('files_sha256')==hashes
            and release.get('cuda_index')==a.cuda_index and release.get('repetitions')==REPETITIONS and release.get('synthetic_records')==TERMINAL_RECORDS
            and isinstance(release.get('cohort_id'),str) and release['cohort_id'] and release.get('expected_host_workers') in (2,4), 'Exact root I/O-only release required')
    checked=datetime.fromisoformat(release['process_identity_checked_utc'])
    require(checked.tzinfo is not None and -60<=(datetime.now(timezone.utc)-checked).total_seconds()<=300,'Fresh root process check required')
    cfg,probe_history,pointer=verify_probe_metadata(directory,trainer)
    out=a.output_dir.resolve();require(out!=directory and directory not in out.parents and out not in directory.parents,'Separate fresh output required')
    out.mkdir(mode=0o700,exist_ok=False)
    (out/'release.json').write_bytes(a.release.read_bytes())
    (out/'protocol.md').write_bytes(a.protocol.read_bytes())
    report={'schema':SCHEMA,'state':'preparing','started_utc':utc(),'pid':os.getpid(),'hostname':socket.gethostname(),
            'cohort_id':release['cohort_id'],'expected_host_workers':release['expected_host_workers'],'files_sha256':hashes,
            'release_sha256':sha(a.release),'environment':ENVIRONMENT,'arm':cfg['arm'],'seed':cfg['seed'],'repetitions':[],
            'synthetic_only':True,'actual_probe_completed_optimizer_updates':512,'new_optimizer_updates':0,'scientific_training_admitted':False,
            'omitted_publication_work_requires_separate_reserve':['model-state finite predicates','live Adam parameter/order/hyperparameter and moment predicates',
                'live CPU/CUDA RNG capture','training configuration/payload construction beyond fixed probe tensor tree'],
            'complete_scientific_checkpoint_bound':False,
            'concurrency':'Root must verify intended concurrent worker/process/storage coverage; this worker does not certify it.'}
    atomic(out/'status.json',report);started=time.perf_counter()
    try:
        import torch
        require(str(torch.__version__)=='2.13.0+cu129' and str(torch.version.cuda)=='12.9' and torch.cuda.is_available(),'Pinned Torch/CUDA required')
        torch.set_num_threads(2);device=torch.device(f'cuda:{a.cuda_index}');torch.cuda.set_device(device)
        props=torch.cuda.get_device_properties(device)
        require('GB200' in props.name and str(props.uuid)==release['gpu_uuid'].removeprefix('GPU-'),'GPU identity differs')
        torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        payload=torch.load(files['checkpoint'],map_location='cpu',weights_only=True)
        require(payload.get('cuda_sand_graph_support_schema')==trainer.SCHEMA and payload.get('completed_steps')==512 and payload.get('run_config')==cfg
                and payload.get('run_config_sha256')==trainer.config_hash(cfg) and payload.get('history')==probe_history,'Loaded probe schema/bytes differ')
        source_tensors={key:payload[key] for key in ('state_dict','optimizer_state','rng_states','simulator_config')}
        expected_tensor_inventory=tensor_inventory(torch,source_tensors);require(expected_tensor_inventory,'Fixed probe tensor bytes required')
        synthetic=synthetic_history(probe_history,cfg['seed'],cfg['arm']);trainer.validate_graph_history(synthetic,TERMINAL_RECORDS,cfg['arm'],cfg['seed'])
        gpu_tree=tensor_tree(torch,source_tensors,device);torch.cuda.synchronize(device)
        del payload,source_tensors
        report.update(state='measuring',preparation_seconds=time.perf_counter()-started,synthetic_training_log_rows=len(synthetic['training']),
            fixed_tensor_inventory=expected_tensor_inventory,fixed_tensor_bytes=sum(r['bytes'] for r in expected_tensor_inventory),
            runtime={'torch':str(torch.__version__),'cuda':str(torch.version.cuda),'device':str(device),'gpu_uuid':str(props.uuid),'name':props.name,'threads':2})
        atomic(out/'status.json',report)
        for rep in range(REPETITIONS):
            require(time.perf_counter()-started<MAX_SECONDS,'I/O measurement time budget reached')
            torch.cuda.synchronize(device);t0=time.perf_counter();rep_started=utc()
            trainer.validate_graph_history(synthetic,TERMINAL_RECORDS,cfg['arm'],cfg['seed']);t1=time.perf_counter()
            cpu_tensors=tensor_tree(torch,gpu_tree,'cpu');torch.cuda.synchronize(device);t2=time.perf_counter()
            artifacts=publish_synthetic_artifacts(torch,out,rep,cpu_tensors,synthetic,pointer['sha256'])
            t3=time.perf_counter();rep_ended=utc();binary=out/artifacts['binary_file'];history_path=out/artifacts['history_file']
            # Verification is outside the measured publication path and is retained.
            require(tensor_inventory(torch,cpu_tensors)==expected_tensor_inventory,'Fixed tensor bytes changed during copy')
            row={'repetition':rep,'started_utc':rep_started,'ended_utc':rep_ended,'ledger_validation_seconds':t1-t0,'gpu_to_cpu_copy_seconds':t2-t1,
                 **artifacts,'synthetic_publication_seconds':t3-t0,
                 'binary_bytes':binary.stat().st_size,'history_bytes':history_path.stat().st_size,'history_sha256':sha(history_path),'verification_ended_utc':utc()}
            report['repetitions'].append(row);atomic(out/'status.json',report);del cpu_tensors
        require(all(sha(path)==hashes[key] for key,path in files.items()) and sha(a.release)==report['release_sha256'], 'Pinned input/release changed during cost measurement')
        maximum=max(r['synthetic_publication_seconds'] for r in report['repetitions'])
        report.update(state='complete_synthetic_io_measurement',ended_utc=utc(),elapsed_seconds=time.perf_counter()-started,
            maximum_synthetic_publication_seconds=maximum,proposed_eleven_synthetic_path_allowance_seconds=1.35*11*maximum,
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_host_rss_units='Linux ru_maxrss KiB converted to bytes',
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),all_inputs_reverified=True,
            interpretation='A planning measurement, not an upper confidence bound. Root must confirm concurrency, storage scope, and an additional reserve for every listed omission; factor1.35 does not establish coverage of omitted work.')
        atomic(out/'result.json',report);atomic(out/'status.json',report);print(json.dumps({'state':report['state'],'maximum_synthetic_publication_seconds':maximum,'result_sha256':sha(out/'result.json')}));return 0
    except BaseException as error:
        report.update(state='failed_synthetic_io_measurement',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            ended_utc=utc(),elapsed_seconds=time.perf_counter()-started,all_existing_outputs_retained=True)
        atomic(out/'status.json',report);raise

if __name__=='__main__':raise SystemExit(main())
