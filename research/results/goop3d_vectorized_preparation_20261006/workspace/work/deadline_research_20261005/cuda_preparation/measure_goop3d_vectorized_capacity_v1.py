#!/usr/bin/env python3
"""Separate root-gated 4+2-wave Goop3D capacity; no scientific endpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA = 'adaptgns_goop3d_vectorized_capacity_v1'
RELEASE_SCHEMA = 'adaptgns_goop3d_vectorized_capacity_release_v1'
CHECKPOINT_SCHEMA = 'adaptgns_goop3d_vectorized_capacity_checkpoint_v1'
PURPOSE = 'bounded_capacity_only_never_promote'
TRAINER_SHA = '8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc'
GRAPH_SHA = 'ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50'
VALIDATOR_SHA = '984a358a07e6b3e7bc80d7e0c64a2b23d4076fdf1a278c0f5d03121616fd3dd1'
LIFECYCLE_SHA = 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd'
STOP, WARMUP, PROBE_LR_HORIZON = 512, 64, 100000
SCHEDULE = [dict(id=f'{a}_seed{s}', wave=w, gpu=g, objective='faithful', arm=a, seed=s)
    for w, g, a, s in [('A',0,'base',0),('A',1,'mix',0),('A',2,'base',1),('A',3,'mix',1),('B',0,'base',2),('B',1,'mix',2)]]
ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'LD_LIBRARY_PATH': '/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
DATA_FIELDS = ('train_manifest','structural_report','acquisition_report','context_semantics','auxiliary_report','data_review','cpu_proof')
B = V = T = H = None


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def private_import(path, digest, name):
    require(sha(path) == digest, 'Frozen helper differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def configure():
    global B, V, T, H
    H = None
    B = private_import(HERE/'measure_sand_cuda_capacity_v2.py', LIFECYCLE_SHA, '_goop3d_capacity_lifecycle')
    V = private_import(HERE/'validate_goop3d_vectorized_cuda_v1.py', VALIDATOR_SHA, '_goop3d_capacity_data')
    T = private_import(HERE/'train_goop3d_graph_support_cuda_v2.py', TRAINER_SHA, '_goop3d_capacity_trainer')
    B.SCHEDULE = SCHEDULE
    B.fixed_command, B.verify_job, B.verify_pairing = fixed_command, verify_job, verify_pairing


def input_paths(args):
    paths = {k: getattr(args, k).resolve() for k in (*DATA_FIELDS, 'protocol', 'numerical_report', 'python')}
    paths.update(supervisor=Path(__file__).resolve(), trainer=HERE/'train_goop3d_graph_support_cuda_v2.py',
        graph=HERE/'goop3d_graph_support_vectorized_v1.py', validator=HERE/'validate_goop3d_vectorized_cuda_v1.py',
        lifecycle=HERE/'measure_sand_cuda_capacity_v2.py')
    return paths


def validate_release(args, release):
    paths = input_paths(args); hashes = {k: sha(p) for k, p in paths.items()}
    require(all(hashes[k] == v for k, v in V.DATA_PINS.items()), 'Frozen complete source evidence differs')
    require(all(sha(args.repo/k)==v for k,v in T.SOURCE_PINS.items()), 'Frozen numerical core differs')
    require(release.get('schema') == RELEASE_SCHEMA and release.get('status') == 'admitted_for_capacity_only'
        and release.get('issued_by') == 'root' and release.get('purpose') == PURPOSE
        and release.get('scientific_training_admitted') is False and release.get('scientific_endpoint_selected') is False
        and release.get('files_sha256') == hashes and release.get('schedule') == SCHEDULE
        and release.get('environment') == ENVIRONMENT and release.get('updates_per_job') == STOP
        and release.get('warmup_updates') == WARMUP and release.get('probe_lr_horizon') == PROBE_LR_HORIZON
        and release.get('numerical_report_independently_reviewed') is True
        and isinstance(release.get('review_rationale'), str) and release['review_rationale'].strip(), 'Exact root capacity-only release required')
    uuids = release.get('gpu_uuids', [])
    require(isinstance(uuids,list) and len(uuids) == 4 and all(isinstance(x,str) and x.startswith('GPU-') for x in uuids)
        and len({B.canonical_gpu_uuid(x) for x in uuids}) == 4, 'Four distinct raw nvidia-smi GPU identities required')
    require(release.get('hostname') == socket.gethostname(), 'Wrong released host')
    contract=release.get('data_contract',{})
    require(contract.get('scope')=='bounded_implementation_data_only' and contract.get('scientific_training_admitted') is False
        and not any(k in contract for k in ('prospective_endpoint_updates','endpoint_selection_basis','selection_evidence_sha256')),
        'Bounded source-only data contract required')
    T.validate_manifest_contract(read(args.train_manifest),contract)
    report = read(args.numerical_report)
    require(report.get('schema') == V.SCHEMA and report.get('status') == 'implementation_passed'
        and report.get('scientific_training_admitted') is False and report.get('scientific_endpoint_selected') is False
        and report.get('test_accessed') is False and report.get('prospective_batches_saved_before_cuda') is True
        and report.get('source_sha256') == VALIDATOR_SHA and report.get('trainer_sha256') == TRAINER_SHA
        and report.get('candidate_graph_sha256') == GRAPH_SHA and report.get('original_graph_sha256') == V.ORIGINAL_SHA
        and report.get('all_inputs_reverified') is True and report.get('schedule') == V.expected_schedule(read(args.train_manifest))
        and report.get('probe_lr_horizon') == PROBE_LR_HORIZON and report.get('attempted_update') is None
        and report.get('completed_optimizer_calls') == 26 and set(report.get('replay', {})) == {'base','mix'}
        and all(row.get('passed') is True and row.get('final_payload_exact') is True
            and row.get('checkpoint_serialization_exact') is True for row in report['replay'].values()),
        'Complete reviewed same-CUDA numerical/replay proof required')
    cases=report.get('cases',[])
    require(len(cases)==6 and {(row.get('case'),row.get('arm')) for row in cases}=={(c,a) for c in V.CASES for a in V.ARMS}
        and all([s.get('completed_steps') for s in row.get('steps',[])]==[1,2]
            and all(s.get('passed') is True and s.get('bytewise_component_checks')
                and all(v is True for v in s['bytewise_component_checks'].values()) for s in row['steps'])
            and len(row.get('graph_gates',[]))==(5 if row['case']=='small' else 4)
            and all(all(g.get(k) is True for k in ('passed','edges_exact','ledger_exact','rng_unchanged')) for g in row['graph_gates'])
            for row in cases)
        and any(e.get('selected_optional_pairs',0)>0 for row in cases for g in row['graph_gates'] for e in g.get('graph_ledger',[])),
        'Complete nonvacuous numerical and graph gates required')
    proof_runtime=report.get('runtime',{})
    require(proof_runtime.get('torch')=='2.13.0+cu129' and proof_runtime.get('cuda')=='12.9'
        and 'GB200' in proof_runtime.get('name','') and proof_runtime.get('threads')==2
        and proof_runtime.get('deterministic_algorithms') is True and proof_runtime.get('deterministic_warn_only') is False
        and proof_runtime.get('cublas_workspace_config')==ENVIRONMENT['CUBLAS_WORKSPACE_CONFIG']
        and proof_runtime.get('float32_matmul_precision')=='highest'
        and all(proof_runtime.get(k) is False for k in ('tf32','amp','compile','ddp')),
        'Numerical proof CUDA contract differs')
    require(datetime.now(timezone.utc) < B.COMPUTE_DEADLINE, 'Compute cutoff reached')
    if args.mode == 'supervise':
        checked = datetime.fromisoformat(release['process_identity_checked_utc'])
        require(checked.tzinfo is not None and -60 <= (datetime.now(timezone.utc)-checked).total_seconds() <= 300,
                'Fresh root host/device process inventory required')
    return paths, hashes


def fixed_command(args, job):
    command = [str(args.python), '-u', str(Path(__file__).resolve()), '--execute', '--mode', 'worker',
               '--repo', str(args.repo), '--release', str(args.release), '--output-dir', str(args.output_dir/'jobs'/job['id']),
               '--job', job['id'], '--python', str(args.python)]
    for name in (*DATA_FIELDS, 'protocol', 'numerical_report'):
        command += ['--'+name.replace('_','-'), str(getattr(args,name))]
    return command


def tree_digest(torch, value):
    def normalize(v):
        if torch.is_tensor(v):
            a = v.detach().cpu().contiguous().numpy()
            return {'tensor_dtype': str(v.dtype), 'shape': list(v.shape), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
        if isinstance(v, dict): return {str(k): normalize(w) for k,w in v.items()}
        if isinstance(v, (list,tuple)): return [normalize(w) for w in v]
        return v
    return T.config_hash(normalize(value))


def checkpoint_payload(h, model, optimizer, config, completed, history, device):
    T.assert_adam(h.torch, model, optimizer, completed, PROBE_LR_HORIZON)
    T.validate_graph_history(history, completed, config['arm'], config['seed'])
    require(h.tensors_are_finite(model.state_dict().values()), 'Nonfinite capacity model')
    return {'schema': CHECKPOINT_SCHEMA, 'purpose': PURPOSE, 'scientific_training_admitted': False,
        'completed_steps': completed, 'probe_lr_horizon': PROBE_LR_HORIZON, 'arm': config['arm'], 'seed': config['seed'],
        'trainer_sha256': TRAINER_SHA, 'graph_sha256': GRAPH_SHA, 'train_manifest_sha256': V.DATA_PINS['train_manifest'],
        'capacity_config': config, 'capacity_config_sha256': T.config_hash(config),
        'state_dict': h.cpu_tree(model.state_dict()), 'simulator_config': h.cpu_tree(model._checkpoint_config),
        'optimizer_state': h.cpu_tree(optimizer.state_dict()), 'rng_states': T.capture_rng(h.torch, device), 'history': history}


def interrupted(signum, frame): raise InterruptedError('Capacity worker interrupted by signal ' + str(signum))


def worker(args, release, hashes):
    job = next(j for j in SCHEDULE if j['id'] == args.job)
    global T, H
    T, H, _, graph = V.load_modules(args.repo)
    dataset, metadata, info = V.load_data(args, H, T, release)
    args.cuda_index, args.threads = job['gpu'], 2
    device, runtime = T.configure_cuda(H, args)
    require(B.canonical_gpu_uuid(runtime['uuid']) == B.canonical_gpu_uuid(release['gpu_uuids'][job['gpu']]), 'Released GPU differs')
    output = args.output_dir.resolve()
    require(all(output != p and p not in output.parents and output not in p.parents for p in
        (args.train_manifest.resolve().parent, (args.repo/'adaptive-gns').resolve())), 'Separate fresh capacity output required')
    output.mkdir(parents=True, exist_ok=False)
    config = {'schema': SCHEMA, 'purpose': PURPOSE, 'scientific_training_admitted': False,
        'scientific_endpoint_selected': False, 'dataset': 'Goop-3D', **job, 'updates_per_job': STOP,
        'probe_lr_horizon': PROBE_LR_HORIZON, 'warmup_updates': WARMUP, 'batch_size': 2, 'history': 6,
        'architecture': {'width':128,'message_passing_blocks':10,'mlp_layers':2}, 'radius':.025, 'noise_std':6.7e-4,
        'graph_exposure': T.graph_exposure_config(job['arm']), 'source_and_input_sha256': hashes, 'repo':str(args.repo.resolve()),
        'release_sha256': sha(args.release), 'data':info, 'runtime':runtime,
        'initialization':'fresh_paired_seed_empty_Adam_no_resume_or_parent_checkpoint'}
    T.atomic_json(output/'protocol.json', config)
    H.torch.manual_seed(job['seed'])
    with H.torch.cuda.device(device): H.torch.cuda.manual_seed(job['seed'])
    model, optimizer = V.make_model(H, metadata, device)
    completed, latest = 0, None
    history = {'training':[], 'graph_updates':[], 'elapsed_seconds':0.}
    started = time.perf_counter(); current = None
    def status(state, error=None):
        T.atomic_json(output/'status.json', {'schema':SCHEMA,'state':state,'purpose':PURPOSE,'scientific_training_admitted':False,
            'completed_steps':completed,'committed_steps':latest['completed_steps'] if latest else None,
            'arm':job['arm'],'seed':job['seed'],'capacity_config_sha256':T.config_hash(config),
            'process':{'pid':os.getpid()},'latest_checkpoint':latest,'last_training':history['training'][-1] if history['training'] else None,
            'error':error,'updated_utc':T.utc_now()})
    def save():
        nonlocal latest
        H.synchronize(device); history['elapsed_seconds'] = time.perf_counter()-started
        payload = checkpoint_payload(H,model,optimizer,config,completed,history,device)
        path = output/f'checkpoint-{completed:09d}.pt'; temporary=path.with_suffix('.pt.tmp')
        require(not path.exists(), 'Never overwrite a capacity checkpoint')
        with temporary.open('xb') as f: H.torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        temporary.replace(path)
        pointer={'path':path.name,'sha256':sha(path),'completed_steps':completed,'capacity_config_sha256':T.config_hash(config)}
        T.atomic_json(output/'latest.json',pointer); latest=pointer
    previous = {s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with T.RunLock(output):
            status('running'); save(); status('running')
            while completed < STOP:
                require(datetime.now(timezone.utc)<B.COMPUTE_DEADLINE,'Compute cutoff reached')
                H.synchronize(device); update_started=time.perf_counter()
                indices=H.sample_indices(job['seed'],completed,len(dataset),2)
                batch=T.unpack_batch(H,[dataset[i] for i in indices]); noise=H.host_noise(batch[0].shape,batch[1],job['seed'],completed)
                ids=[H.frame_identity(dataset,info['trajectory_ids'],i) for i in indices]
                current={'completed_before':completed,'frame_ids':ids,'noise_sha256':graph.state_hash(noise.numpy())}
                rate=T.learning_rate(completed,PROBE_LR_HORIZON)
                optimizer.param_groups[0]['lr']=rate; model.train(); optimizer.zero_grad(set_to_none=True)
                pred,head,target,ledger=graph.forward_batch(model,batch,noise,device,job['seed'],completed,job['arm'])
                current['examples']=ledger
                loss=T.guarded_update(H,model,optimizer,pred,head,target,(batch[1]!=3).to(device),'faithful',completed+1,PROBE_LR_HORIZON)
                H.synchronize(device); seconds=time.perf_counter()-update_started; completed+=1
                history['graph_updates'].append({'completed_steps':completed,'absolute_schedule_step':completed-1,
                    'frame_ids':ids,'noise_sha256':current['noise_sha256'],'examples':ledger})
                row={'completed_steps':completed,'loss':float(loss.detach().cpu()),'lr':rate,'frame_ids':ids,'particles':len(batch[0]),
                     'guarded_update_seconds':seconds,'elapsed_seconds':time.perf_counter()-started}
                history['training'].append(row); print(json.dumps(row,allow_nan=False),flush=True); status('running')
            save()
            V.verify_data_evidence(args,H,T,release)
            H.data_loader.load_manifest_data(args.train_manifest,verify_hashes=True)
            require(all(sha(p)==hashes[k] for k,p in input_paths(args).items()) and sha(args.release)==config['release_sha256']
                and all(sha(args.repo/k)==v for k,v in T.SOURCE_PINS.items()), 'Source/input/release changed')
            T.atomic_json(output/'history.json',history); status('complete_capacity_only')
    except BaseException as error:
        history['elapsed_seconds']=time.perf_counter()-started
        T.atomic_json(output/'history.json',history)
        try:
            with (output/'unsuccessful_state.pt').open('xb') as f:
                H.torch.save({'schema':CHECKPOINT_SCHEMA,'purpose':'unsuccessful_capacity_state_never_promote',
                    'scientific_training_admitted':False,'capacity_config':config,'completed_steps':completed,
                    'state_dict':H.cpu_tree(model.state_dict()),'optimizer_state':H.cpu_tree(optimizer.state_dict()),
                    'gradients':{name:H.cpu_tree(p.grad) for name,p in model.named_parameters()},
                    'rng_states':T.capture_rng(H.torch,device),'current':current,'history':history},f)
                f.flush();os.fsync(f.fileno())
        except BaseException as preservation_error:
            T.atomic_json(output/'state_preservation_error.json',{'original_error_type':type(error).__name__,
                'original_error':str(error),'preservation_error_type':type(preservation_error).__name__,
                'preservation_error':str(preservation_error),'all_existing_outputs_retained':True})
        T.atomic_json(output/'failed_attempt.json',{'error_type':type(error).__name__,'error':str(error),'current':current,
            'completed_steps':completed,'latest_committed_checkpoint':latest,'purpose':PURPOSE,'all_existing_outputs_retained':True})
        status('failed',f'{type(error).__name__}: {error}'); raise
    finally:
        for s,handler in previous.items():signal.signal(s,handler)


def validate_rows(history, job, manifest):
    require(H is not None,'Source-only helper required to recompute exact sample schedule')
    T.validate_graph_history(history,STOP,job['arm'],job['seed'])
    rows=history['training'];require(len(rows)==STOP,'All512 scalar rows required')
    counts={r['id']:r['positions']['shape'][1] for r in manifest['records']}; previous=0.
    for step,(row,graph) in enumerate(zip(rows,history['graph_updates']),1):
        require(row['completed_steps']==step and row['frame_ids']==graph['frame_ids'] and len(row['frame_ids'])==2,
                'Exact scalar/graph schedule rows required')
        indices=H.sample_indices(job['seed'],step-1,len(manifest['records'])*295,2)
        require(row['frame_ids']==[f"{manifest['records'][int(i)//295]['id']}:{int(i)%295+6}" for i in indices],
                'Fixed host sample schedule differs')
        require(math.isfinite(row['loss']) and B.finite(row['guarded_update_seconds'],True)
            and B.finite(row['elapsed_seconds'],True) and row['elapsed_seconds']>previous
            and row['lr']==T.learning_rate(step-1,PROBE_LR_HORIZON),'Invalid timing/loss/LR row')
        total=0
        for slot,identity in enumerate(row['frame_ids']):
            trajectory,frame=identity.rsplit(':',1)
            require(trajectory in counts and frame.isdigit() and 6<=int(frame)<301,'Out-of-source frame')
            require(graph['examples'][slot]['n_particles']==counts[trajectory],'Graph particle count differs')
            total+=counts[trajectory]
        require(row['particles']==total,'Scalar particle count differs');previous=row['elapsed_seconds']
    guarded=[r['guarded_update_seconds'] for r in rows]
    q=(rows[-1]['elapsed_seconds']-rows[WARMUP-1]['elapsed_seconds'])/(STOP-WARMUP)
    require(q>0 and q*(STOP-WARMUP)+1e-6>=sum(guarded[WARMUP:]),'Steady wall excludes guarded work')
    return {'steady_wall_seconds_per_update':q,'sum_all512_guarded_seconds':sum(guarded)}


def verify_checkpoint(path, config, expected_steps, history, metadata):
    global H
    if H is None:
        _,H,_,_=V.load_modules(Path(config['repo']))
    payload=H.torch.load(path,map_location='cpu',weights_only=True)
    require(payload.get('schema')==CHECKPOINT_SCHEMA and payload.get('purpose')==PURPOSE
        and payload.get('scientific_training_admitted') is False and payload.get('completed_steps')==expected_steps
        and payload.get('probe_lr_horizon')==PROBE_LR_HORIZON and payload.get('capacity_config')==config
        and payload.get('capacity_config_sha256')==T.config_hash(config)
        and payload.get('trainer_sha256')==TRAINER_SHA and payload.get('graph_sha256')==GRAPH_SHA
        and payload.get('train_manifest_sha256')==V.DATA_PINS['train_manifest']
        and payload.get('arm')==config['arm'] and payload.get('seed')==config['seed'], 'Capacity payload lineage differs')
    require('training_config' not in payload and 'run_config' not in payload and payload['history']==history,'Payload history or scientific schema contamination')
    T.validate_graph_history(history,expected_steps,config['arm'],config['seed'])
    model,optimizer=V.make_model(H,metadata,'cpu')
    require(T.tree_equal(H.torch,H.cpu_tree(model._checkpoint_config),payload['simulator_config']),'Simulator config differs')
    model.load_state_dict(payload['state_dict'],strict=True);optimizer.load_state_dict(payload['optimizer_state'])
    require(T.tree_equal(H.torch,H.cpu_tree(model.state_dict()),payload['state_dict'])
        and T.tree_equal(H.torch,H.cpu_tree(optimizer.state_dict()),payload['optimizer_state']), 'Checkpoint tensor representation changed on restore')
    T.assert_adam(H.torch,model,optimizer,expected_steps,PROBE_LR_HORIZON)
    require(H.tensors_are_finite(model.state_dict().values()),'Nonfinite saved model')
    rng=payload['rng_states']
    require(set(rng)=={'cpu','cuda'} and all(H.torch.is_tensor(x) and x.device.type=='cpu' and x.dtype==H.torch.uint8 and x.ndim==1 and x.numel()>0 for x in rng.values()),'Saved RNG payload differs')
    return {'model_tensor_sha256':tree_digest(H.torch,payload['state_dict']),'rng_sha256':tree_digest(H.torch,rng),
            'all_model_and_Adam_values_finite':True,'all_Adam_steps_exact':True}


def verify_job(directory, job, external, manifest, protocol_sha, gpu_uuid):
    global H
    directory=Path(directory)
    require(external.get('exit_code')==0 and B.finite(external.get('elapsed_seconds'),True),'Incomplete worker process')
    require(not list(directory.rglob('*.tmp')) and not any((directory/n).exists() for n in
        ('run.lock','failed_attempt.json','unsuccessful_state.pt','state_preservation_error.json')),'Failed/partial capacity output')
    config,status,history,pointer=[read(directory/n) for n in ('protocol.json','status.json','history.json','latest.json')]
    launch=read(directory.parent.parent/'launch.json')
    require(launch.get('schema')==SCHEMA and launch.get('purpose')==PURPOSE
        and launch.get('schedule')==SCHEDULE and launch.get('environment')==ENVIRONMENT
        and config.get('source_and_input_sha256')==launch.get('files_sha256')
        and config.get('release_sha256')==launch.get('release_sha256') and config.get('repo')==launch.get('repo'),
        'Worker configuration differs from supervisor release/input binding')
    require(config.get('schema')==SCHEMA and config.get('purpose')==PURPOSE and config.get('scientific_training_admitted') is False
        and config.get('scientific_endpoint_selected') is False and all(config.get(k)==v for k,v in job.items())
        and config.get('updates_per_job')==STOP and config.get('probe_lr_horizon')==PROBE_LR_HORIZON
        and config.get('warmup_updates')==WARMUP and config.get('batch_size')==2 and config.get('history')==6
        and config.get('architecture')=={'width':128,'message_passing_blocks':10,'mlp_layers':2}
        and config.get('radius')==.025 and config.get('noise_std')==6.7e-4
        and config.get('graph_exposure')==T.graph_exposure_config(job['arm'])
        and config.get('source_and_input_sha256',{}).get('protocol')==protocol_sha,'Capacity configuration differs')
    expected_pins={**V.DATA_PINS,'supervisor':sha(__file__),'trainer':TRAINER_SHA,'graph':GRAPH_SHA,'validator':VALIDATOR_SHA,'lifecycle':LIFECYCLE_SHA}
    require(all(config['source_and_input_sha256'].get(k)==v for k,v in expected_pins.items()), 'Recorded source/data pins differ')
    if H is None: _,H,_,_=V.load_modules(Path(config['repo']))
    runtime=config['runtime']
    require(runtime.get('device')==f"cuda:{job['gpu']}" and B.canonical_gpu_uuid(runtime['uuid'])==B.canonical_gpu_uuid(gpu_uuid)
        and runtime.get('deterministic_algorithms') is True and runtime.get('deterministic_warn_only') is False
        and runtime.get('torch')=='2.13.0+cu129' and runtime.get('cuda')=='12.9' and runtime.get('threads')==2
        and 'GB200' in runtime.get('name','') and runtime.get('float32_matmul_precision')=='highest'
        and runtime.get('cublas_workspace_config')==ENVIRONMENT['CUBLAS_WORKSPACE_CONFIG']
        and all(runtime.get(k) is False for k in ('tf32','amp','compile','ddp')),'Runtime/GPU contract differs')
    require(status.get('state')=='complete_capacity_only' and status.get('completed_steps')==status.get('committed_steps')==STOP
        and status.get('capacity_config_sha256')==T.config_hash(config) and status.get('process',{}).get('pid')==external['pid']
        and status.get('latest_checkpoint')==pointer and status.get('error') is None,'Capacity completion status differs')
    pointers=[]
    for step in (0,STOP):
        expected={'path':f'checkpoint-{step:09d}.pt','sha256':sha(directory/f'checkpoint-{step:09d}.pt'),
                  'completed_steps':step,'capacity_config_sha256':T.config_hash(config)}
        require((external.get('initial_pointer') if step==0 else pointer)==expected,'Observed pointer/checkpoint differs');pointers.append(expected)
    stats=validate_rows(history,job,manifest)
    require([json.loads(line) for line in Path(external['stdout_file']).read_text().splitlines() if line.strip()]==history['training']
        and status.get('last_training')==history['training'][-1],'Stdout/history/status rows differ')
    initial=H.torch.load(directory/pointers[0]['path'],map_location='cpu',weights_only=True)
    first=verify_checkpoint(directory/pointers[0]['path'],config,0,initial['history'],manifest['metadata'])
    final=verify_checkpoint(directory/pointers[1]['path'],config,STOP,history,manifest['metadata'])
    residual=external['elapsed_seconds']-stats['sum_all512_guarded_seconds'];require(residual>=-1e-6,'External time excludes guarded work')
    return {**job,'status':'verified_capacity_only','initial':first,'final':final,'initial_pointer':pointers[0],'final_pointer':pointers[1],
        **stats,'nonnegative_external_minus_all_guarded_seconds':max(0.,residual),'pairing_rows':history['training'],
        'graph_rows':history['graph_updates'],'external':external,'artifact_sha256':{n:sha(directory/n) for n in ('protocol.json','status.json','history.json','latest.json')}}


def verify_pairing(jobs):
    by={(j['arm'],j['seed']):j for j in jobs};require(len(by)==len(jobs),'Duplicate job identity')
    results=[]
    for seed in sorted({j['seed'] for j in jobs}):
        require(('base',seed) in by and ('mix',seed) in by,'Missing paired arm')
        a,b=by['base',seed],by['mix',seed]
        require(a['initial']==b['initial'],'Initial model/RNG/Adam pairing differs')
        require(len(a['pairing_rows'])==len(b['pairing_rows'])==len(a['graph_rows'])==len(b['graph_rows'])==STOP,'All512 paired rows required')
        for x,y,g,h in zip(a['pairing_rows'],b['pairing_rows'],a['graph_rows'],b['graph_rows']):
            require(all(x[k]==y[k] for k in ('completed_steps','frame_ids','particles','lr'))
                and g['noise_sha256']==h['noise_sha256'],'Paired sample/noise/LR differs')
            require(len(g.get('examples',[]))==len(h.get('examples',[]))==2,'Two paired graph examples required')
            for first,second in zip(g['examples'],h['examples']):
                require(all(k in first and k in second and first[k]==second[k] for k in ('example_slot','n_particles','exposure_coin',
                    'coin_seed_material','pair_seed_material','native_directed_edges','native_self_edges','receivers_above_native_cap',
                    'annulus_pairs','optional_budget_if_exposed','native_edge_sha256','noisy_current_sha256')),
                    'Paired native graph/noisy state/coin/material/annulus budget differs')
        results.append({'seed':seed,'initial_model_rng_Adam_exact':True,'all512_sample_noise_lr_rows_exact':True,
                        'all_native_graph_noisy_state_coin_material_annulus_budget_exact':True})
    return results


def supervise(args, release, paths, hashes):
    require(sys.platform.startswith('linux') and hasattr(os,'wait4') and os.environ.get('CUDA_VISIBLE_DEVICES') is None,
            'Unremapped Linux CUDA host required')
    output=args.output_dir.resolve()
    require(all(output!=p and p not in output.parents and output not in p.parents for p in
        (args.train_manifest.resolve().parent,(args.repo/'adaptive-gns').resolve())), 'Separate fresh supervisor output required')
    output.mkdir(parents=True,exist_ok=False);(output/'jobs').mkdir();(output/'logs').mkdir();(output/'inputs').mkdir()
    copies={}
    for key,path in paths.items():
        if key=='python':continue
        target=output/'inputs'/f'{key}{path.suffix}';target.write_bytes(path.read_bytes());copies[key]=str(target.relative_to(output))
    (output/'inputs/release.json').write_bytes(args.release.read_bytes())
    launch={'schema':SCHEMA,'purpose':PURPOSE,'scientific_training_admitted':False,'scientific_endpoint_selected':False,
        'schedule':SCHEDULE,'environment':ENVIRONMENT,'gpu_uuids':release['gpu_uuids'],'files_sha256':hashes,'input_snapshots':copies,
        'release_sha256':sha(args.release),'repo':str(args.repo.resolve()),'commands':{j['id']:fixed_command(args,j) for j in SCHEDULE}}
    B.atomic_json(output/'launch.json',launch)
    try:
        manifest=read(args.train_manifest)
        jobs=B.run_waves(lambda wave:B.run_wave(args,wave,launch,manifest))
        require(all(sha(path)==hashes[key] for key,path in paths.items()) and sha(args.release)==launch['release_sha256']
            and all(sha(args.repo/k)==v for k,v in T.SOURCE_PINS.items()),'Input/release/numerical core changed')
        require(read(output/'launch.json')==launch and sha(output/'inputs/release.json')==launch['release_sha256']
            and all(sha(output/relative)==hashes[key] for key,relative in copies.items()),'Copied launch/input snapshot changed')
        for wave in ('A','B'):B.verify_wave_record(output,wave,launch)
        for job in jobs:
            directory=output/'jobs'/job['id']
            require(all(sha(directory/n)==v for n,v in job['artifact_sha256'].items())
                and all(sha(directory/job[p]['path'])==job[p]['sha256'] for p in ('initial_pointer','final_pointer')),
                'Verified worker artifacts changed before final summary')
        q={wave:max(j['steady_wall_seconds_per_update'] for j in jobs if j['wave']==wave) for wave in ('A','B')}
        residual={wave:max(j['nonnegative_external_minus_all_guarded_seconds'] for j in jobs if j['wave']==wave) for wave in ('A','B')}
        B.atomic_json(output/'summary.json',{'schema':SCHEMA,'status':'all_six_verified_capacity_only','purpose':PURPOSE,
            'scientific_training_admitted':False,'scientific_endpoint_selected':False,'jobs':jobs,'pairing':verify_pairing(jobs),
            'all_inputs_and_worker_artifacts_reverified':True,'q4_q2':q,'r4_r2':residual,
            'scientific_endpoint_updates':None,'full_study_runtime_forecast_seconds':None,
            'interpretation':'Sequential4+2 waves with observed steady overlap; engineering inputs only, no selected endpoint or full-evaluation forecast.'})
        B.atomic_json(output/'status.json',{'schema':SCHEMA,'state':'complete_capacity_only','summary_sha256':sha(output/'summary.json'),
            'scientific_training_admitted':False,'scientific_endpoint_selected':False,'ended_utc':T.utc_now()})
    except BaseException as error:
        B.atomic_json(output/'status.json',{'schema':SCHEMA,'state':'failed_capacity_only','error_type':type(error).__name__,'error':str(error),
            'wave_B_released':(output/'wave_B.json').exists(),'scientific_training_admitted':False,
            'scientific_endpoint_selected':False,'all_existing_outputs_retained':True,'ended_utc':T.utc_now()})
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    p.add_argument('--mode',choices=('supervise','worker'));p.add_argument('--job',choices=[j['id'] for j in SCHEDULE])
    for name in ('repo','release','output-dir','python','protocol','numerical-report',*[k.replace('_','-') for k in DATA_FIELDS]):p.add_argument('--'+name,type=Path)
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','schedule':SCHEDULE,'updates_per_job':STOP,'warmup_updates':WARMUP,
            'probe_lr_horizon':PROBE_LR_HORIZON,'scientific_endpoint_selected':False,'scientific_training_admitted':False,'purpose':PURPOSE},indent=2));return 0
    require(args.mode and all(getattr(args,k) is not None for k in ('repo','release','output_dir','python','protocol','numerical_report',*DATA_FIELDS)), 'Explicit input/release/mode/fresh output required')
    require(args.mode!='worker' or args.job is not None,'Worker fixed job identity required')
    configure();release=read(args.release);paths,hashes=validate_release(args,release)
    if args.mode=='worker':worker(args,release,hashes)
    else:supervise(args,release,paths,hashes)
    return 0


if __name__=='__main__':raise SystemExit(main())
