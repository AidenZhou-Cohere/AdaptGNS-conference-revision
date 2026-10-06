#!/usr/bin/env python3
"""Fresh six-cell Goop3D science; inert until root fixes endpoint and time budget.

The trainer and numerical sources remain byte-frozen. The process loop derives
from the reviewed scientific supervisor; wait4, identity checks and cleanup are
called directly from the pinned lifecycle. No automatic resume or test access.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
SCHEMA='adaptgns_goop3d_scientific_supervisor_v1'
RELEASE_SCHEMA='adaptgns_goop3d_scientific_release_v1'
TRAINER_SHA='8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc'
GRAPH_SHA='ca5898ec3f487d8138adf7479743156874a7796df09841f8ff7711b97536ad50'
LIFECYCLE_SHA='c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd'
VALIDATOR_SHA='984a358a07e6b3e7bc80d7e0c64a2b23d4076fdf1a278c0f5d03121616fd3dd1'
WORKSHEET_SHA='8a4946e813a6b9ef3d0cb86c5649d614d2bc89fd08b9962bf3ec877e94a20db6'
PROCESS_LOOP_DERIVED_FROM_SHA='a22a59c46231c328e4939a0f021b3bb2fab9e2f93c0848bb1abd39aad2d5d305'
DEADLINE=datetime(2026,10,7,1,tzinfo=timezone.utc)
SCHEDULE=[dict(id=f'{a}_seed{s}',wave=w,gpu=g,objective='faithful',arm=a,seed=s) for w,g,a,s in
    [('A',0,'base',0),('A',1,'mix',0),('A',2,'base',1),('A',3,'mix',1),('B',0,'base',2),('B',1,'mix',2)]]
ENVIRONMENT={'CUBLAS_WORKSPACE_CONFIG':':4096:8','LD_LIBRARY_PATH':'/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
FIELDS=('repo','python','train_manifest','admission','structural_report','acquisition_report','context_semantics',
    'auxiliary_report','protocol','capacity_summary','worksheet','planning_config','endpoint_plan','process_check','clock_check')
B=T=V=W=H=None
ACTIVE=None


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def stamp(value):
    result=datetime.fromisoformat(value)
    require(result.tzinfo is not None,'Explicit timezone required')
    return result.astimezone(timezone.utc)


def load(name,digest,label):
    path=HERE/name;require(sha(path)==digest,'Frozen helper differs: '+name)
    spec=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(spec)
    sys.modules[label]=m;spec.loader.exec_module(m);return m


def configure(args):
    global B,T,V,W,H,ACTIVE
    B=load('measure_sand_cuda_capacity_v2.py',LIFECYCLE_SHA,'_goop3d_science_lifecycle')
    T=load('train_goop3d_graph_support_cuda_v2.py',TRAINER_SHA,'_goop3d_science_trainer')
    V=load('validate_goop3d_vectorized_cuda_v1.py',VALIDATOR_SHA,'_goop3d_science_model_factory')
    W=load('goop3d_deadline_worksheet_v1.py',WORKSHEET_SHA,'_goop3d_science_worksheet')
    B.verify_job=record_reaped_job;B.fixed_command=fixed_command;B.SCHEDULE=SCHEDULE
    H=None;ACTIVE=args


def input_paths(args):
    paths={key:getattr(args,key).resolve() for key in FIELDS if key!='repo'}
    paths.update(supervisor=Path(__file__).resolve(),trainer=HERE/'train_goop3d_graph_support_cuda_v2.py',
        graph=HERE/'goop3d_graph_support_vectorized_v1.py',lifecycle=HERE/'measure_sand_cuda_capacity_v2.py',
        validator=HERE/'validate_goop3d_vectorized_cuda_v1.py',worksheet_source=HERE/'goop3d_deadline_worksheet_v1.py')
    paths.update(metadata=args.train_manifest.resolve().parent/'metadata.json',
        auxiliary_validator=HERE/'audit_goop3d_auxiliary.py',original_graph=HERE/'goop3d_graph_support.py')
    paths.update({'core:'+key:args.repo/key for key in T.SOURCE_PINS})
    paths.update({'context:'+key:args.context_semantics.resolve().parent/'goop_context_semantics_sources'/key for key in T.CONTEXT_SOURCE_PINS})
    return paths


def fixed_command(args,job):
    c=[str(args.python),'-u',str(HERE/'train_goop3d_graph_support_cuda_v2.py'),'--execute','--repo',str(args.repo)]
    for key in ('train_manifest','admission','structural_report','acquisition_report','context_semantics','auxiliary_report','protocol'):
        c+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return c+['--output-dir',str(args.output_dir/'jobs'/job['id']),'--objective','faithful','--arm',job['arm'],
        '--seed',str(job['seed']),'--cuda-index',str(job['gpu']),'--updates',str(args.updates),'--threads','2',
        '--checkpoint-every',str(args.checkpoint_every),'--log-every',str(args.log_every)]


def validate_plan(args,hashes):
    plan,worksheet,planning=read(args.endpoint_plan),read(args.worksheet),read(args.planning_config)
    q,residual,_=W.capacity_inputs(read(args.capacity_summary))
    require(plan.get('schema')=='adaptgns_goop3d_prospective_endpoint_plan_v1' and plan.get('status')=='prospectively_selected'
        and plan.get('issued_by')=='root' and plan.get('endpoint_updates')==args.updates
        and plan.get('checkpoint_every')==args.checkpoint_every and plan.get('log_every')==args.log_every
        and plan.get('selection_basis')==T.ENDPOINT_BASIS and plan.get('selection_used_validation_or_test_accuracy') is False
        and isinstance(plan.get('rationale'),str) and plan['rationale'].strip(), 'Explicit timing-only prospective endpoint plan required')
    require(planning.get('schema')=='adaptgns_goop3d_deadline_planning_config_v1'
        and planning.get('selected_endpoint') is None and planning.get('evaluation_overlap_credit') is False
        and args.updates in planning.get('candidate_endpoints',[]) and planning.get('checkpoint_every')==args.checkpoint_every,
        'Endpoint must be a prospectively timed candidate with matching checkpoint cadence')
    require(worksheet.get('schema')==W.SCHEMA and worksheet.get('scientific_endpoint_selected') is False
        and worksheet.get('scientific_training_admitted') is False and worksheet.get('test_accessed') is False
        and all(worksheet.get('input_sha256',{}).get(str(getattr(args,k).resolve()))==hashes[k]
            for k in ('capacity_summary','planning_config')), 'Worksheet/capacity/planning byte lineage differs')
    basis=plan.get('cost_basis')
    if basis=='measured_full_horizon_engineering_estimate':
        require(worksheet.get('status')=='engineering_worksheet_complete_not_execution_admission'
            and worksheet.get('forecasts')==W.forecast(planning,q,residual,worksheet.get('timing',[])), 'Complete measured worksheet arithmetic required')
        selected=[r for r in worksheet['forecasts'] if r['candidate_endpoint_updates']==args.updates]
        require(len(selected)==1 and selected[0]['fits_stated_planning_assumptions'] is True,'Selected endpoint does not fit measured planning assumptions')
        require(plan.get('runtime_multiplier',0)>=planning['runtime_multiplier']
            and plan.get('evaluation_and_analysis_reserve_seconds',0)>=selected[0]['evaluation_serial_seconds']*plan['runtime_multiplier']+planning['transfer_review_analysis_reserve_seconds'],
            'Scientific windows must retain the measured evaluation and analysis reserve')
    elif basis=='fixed_operational_quota_after_failed_full_horizon_timing':
        require(worksheet.get('status')=='forecast_unavailable' and worksheet.get('forecasts')==[]
            and worksheet.get('unavailable_reasons') and plan.get('all_required_evaluation_outcomes_promised') is False
            and plan.get('full_study_measured_runtime_claim') is False
            and plan.get('complete_full_horizon_timing_attempt_failed') is True
            and isinstance(plan.get('preserved_failed_timing_sha256'),list) and plan['preserved_failed_timing_sha256']
            and all(T.digest_string(x) for x in plan['preserved_failed_timing_sha256']), 'Explicit adverse-timing/quota branch required; no full-runtime claim')
    else: raise ValueError('Unknown prospective cost basis')
    require(B.finite(plan.get('runtime_multiplier'),True) and plan['runtime_multiplier']>=1
        and B.finite(plan.get('evaluation_and_analysis_reserve_seconds'),True)
        and set(plan.get('wave_audit_reserve_seconds',{}))=={'A','B'}
        and all(B.finite(x,True) for x in plan['wave_audit_reserve_seconds'].values()),'Explicit conservative reserves required')
    saves=math.ceil(args.updates/args.checkpoint_every)+1
    costs={w:plan['runtime_multiplier']*(args.updates*q[w]+saves*residual[w])+plan['wave_audit_reserve_seconds'][w] for w in ('A','B')}
    return plan,costs


def validate_release(args,release,hashes,now=None):
    now=B.now() if now is None else now
    plan,costs=validate_plan(args,hashes)
    admission=read(args.admission);T.validate_prospective_endpoint(args,admission)
    require(admission.get('scientific_training_admitted') is True and 'scope' not in admission,
        'Scientific admission must explicitly replace the bounded implementation scope')
    T.validate_manifest_contract(read(args.train_manifest),admission)
    require({hashes[k] for k in ('capacity_summary','worksheet','endpoint_plan')}<=set(admission['selection_evidence_sha256']), 'Admission must bind capacity, worksheet and endpoint decision')
    require(release.get('schema')==RELEASE_SCHEMA and release.get('status')=='admitted_for_scientific_training'
        and release.get('issued_by')=='root' and release.get('scientific_training_admitted') is True
        and release.get('files_sha256')==hashes and release.get('schedule')==SCHEDULE and release.get('environment')==ENVIRONMENT
        and release.get('endpoint_updates')==args.updates and release.get('hostname')==socket.gethostname()
        and release.get('output_dir')==str(args.output_dir.resolve())
        and release.get('cost_basis')==plan['cost_basis'] and release.get('compute_analysis_deadline_utc')==DEADLINE.isoformat()
        and isinstance(release.get('cohort_id'),str) and release['cohort_id'].strip()
        and release.get('independent_source_and_feasibility_reviews_complete') is True,'Exact reviewed root scientific release required')
    uuids=release.get('gpu_uuids',[])
    require(len(uuids)==4 and all(isinstance(u,str) and u.startswith('GPU-') for u in uuids)
        and len({B.canonical_gpu_uuid(u) for u in uuids})==4,'Four raw distinct physical GPU UUIDs required')
    process,clock=read(args.process_check),read(args.clock_check)
    require(process.get('schema')=='adaptgns_goop3d_scientific_process_check_v1' and process.get('issued_by')=='root'
        and process.get('hostname')==release['hostname'] and process.get('gpu_uuids')==uuids
        and process.get('matching_training_processes')==[] and process.get('gpu_processes')==[], 'Idle dedicated third-host process evidence required')
    require(clock.get('schema')=='adaptgns_goop3d_scientific_clock_check_v1' and clock.get('issued_by')=='root'
        and clock.get('hostname')==release['hostname'] and clock.get('root_host_samples_reviewed') is True
        and B.finite(clock.get('clock_error_bound_seconds')) and clock['clock_error_bound_seconds']<=5, 'Root-reviewed clock bound required')
    error=clock['clock_error_bound_seconds']
    for record,key in ((process,'process_identity_checked_utc'),(clock,'clock_checked_utc')):
        require(-5<=(now-stamp(record['checked_utc'])).total_seconds()<=300
            and release.get(key)==record['checked_utc'],'Fresh root process/clock receipt required')
    require(release.get('clock_error_bound_seconds')==error and set(release.get('waves',{}))=={'A','B'},'Exact wave clock contract required')
    windows={w:{k:stamp(release['waves'][w][k]) for k in ('latest_start_utc','stop_utc')} for w in ('A','B')}
    for w,x in windows.items():
        require(x['latest_start_utc']+timedelta(seconds=costs[w]+error+B.CLEANUP_SECONDS)<=x['stop_utc']<=DEADLINE,
            'Wave window lacks measured training/audit/cleanup reserve')
    require(now+timedelta(seconds=error)<=windows['A']['latest_start_utc']
        and windows['A']['stop_utc']<=windows['B']['latest_start_utc']
        and windows['B']['stop_utc']+timedelta(seconds=plan['evaluation_and_analysis_reserve_seconds'])<=DEADLINE,
        'Sequential four-plus-two training/evaluation windows do not fit')
    return plan,windows


def record_reaped_job(directory,job,external,manifest,protocol_sha,gpu_uuid):
    external['nominal_poll_interval_seconds']=.2
    require(external.get('exit_code')==0 and not external.get('signals'),'Scientific child did not exit cleanly')
    status=read(Path(directory)/'status.json')
    require(status.get('state')=='complete' and status.get('error') is None
        and status.get('completed_steps')==status.get('committed_steps')==status.get('requested_steps')==ACTIVE.updates,
        'Complete selected endpoint required')
    return {**job,'status':'reaped_pending_endpoint_verification','directory':str(directory),'external':external}


@contextmanager
def audit_alarm(stop):
    remaining=(stop-B.now()).total_seconds()
    require(remaining>0 and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'Audit time budget exhausted or conflicting alarm')
    previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Scientific audit budget exhausted')))
    signal.setitimer(signal.ITIMER_REAL,remaining)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def checkpoint_steps(endpoint,every):return sorted({0,endpoint,*range(every,endpoint+1,every)})
def logged_steps(endpoint,every):return sorted({1,endpoint,*range(every,endpoint+1,every)})


def check_history(history,job,manifest):
    T.validate_graph_history(history,ACTIVE.updates,job['arm'],job['seed'])
    rows=history['training'];require([r['completed_steps'] for r in rows]==logged_steps(ACTIVE.updates,ACTIVE.log_every),'Exact selected-endpoint logging cadence required')
    previous=0.;counts={r['id']:r['positions']['shape'][1] for r in manifest['records']}
    for row in rows:
        require(B.finite(row['guarded_update_seconds'],True) and B.finite(row['elapsed_seconds'],True)
            and row['elapsed_seconds']>previous and math.isfinite(row['loss'])
            and row['lr']==T.learning_rate(row['completed_steps']-1,ACTIVE.updates),'Invalid scalar duration/loss/LR')
        g=history['graph_updates'][row['completed_steps']-1]
        require(row['frame_ids']==g['frame_ids'] and row['particles']==sum(e['n_particles'] for e in g['examples']),'Scalar/graph record disagreement')
        previous=row['elapsed_seconds']
    for i,g in enumerate(history['graph_updates']):
        indices=H.sample_indices(job['seed'],i,len(manifest['records'])*295,2)
        expected=[f"{manifest['records'][int(x)//295]['id']}:{int(x)%295+6}" for x in indices]
        require(g['frame_ids']==expected,'Absolute host sample schedule differs')
        for slot,(identity,e) in enumerate(zip(g['frame_ids'],g['examples'])):
            name,target=identity.rsplit(':',1)
            coin=bool(H.np.random.default_rng(H.np.random.SeedSequence([20261005,job['seed'],i,slot,4409])).random()<.5)
            require(e['n_particles']==counts[name] and 6<=int(target)<301 and e['exposure_coin'] is coin,'Particle/graph-coin schedule differs')
    require(history['elapsed_seconds']>=previous,'Final history clock precedes last update')
    return {'graph_updates':ACTIVE.updates,'logged_rows':len(rows),'logged_guarded_seconds':sum(r['guarded_update_seconds'] for r in rows),
        'logged_guarded_seconds_scope':'Only logged updates; never a sum over all updates unless log_every=1.'}


def check_checkpoint(path,config,completed,history,metadata):
    payload=H.torch.load(path,map_location='cpu',weights_only=True)
    require(payload.get('format_version')==2 and payload.get('cuda_goop3d_graph_support_schema')==T.SCHEMA
        and not any(k in payload for k in ('schema','cuda_goop_graph_support_schema','cuda_sand_graph_support_schema','full_training_schema'))
        and payload.get('completed_steps')==completed and payload.get('run_config')==config
        and payload.get('run_config_sha256')==T.config_hash(config)
        and payload.get('training_config')=={'loss':'faithful','cuda_goop3d_graph_support_run':config,'completed_optimizer_updates':completed}
        and payload.get('history')==history,'Scientific endpoint/checkpoint lineage differs')
    T.validate_graph_history(history,completed,config['arm'],config['seed'])
    model,optimizer=V.make_model(H,metadata,'cpu')
    require(T.tree_equal(H.torch,payload['simulator_config'],H.cpu_tree(model._checkpoint_config)),'D3 model metadata/normalization differs')
    model.load_state_dict(payload['state_dict'],strict=True);optimizer.load_state_dict(payload['optimizer_state'])
    require(T.tree_equal(H.torch,payload['state_dict'],H.cpu_tree(model.state_dict()))
        and T.tree_equal(H.torch,payload['optimizer_state'],H.cpu_tree(optimizer.state_dict())),'Saved tensor representation differs')
    T.assert_adam(H.torch,model,optimizer,completed,ACTIVE.updates)
    require(H.tensors_are_finite(model.state_dict().values()),'Nonfinite checkpoint model')
    rng=payload['rng_states']
    require(set(rng)=={'cpu','cuda'} and all(H.torch.is_tensor(v) and v.device.type=='cpu'
        and v.dtype==H.torch.uint8 and v.ndim==1 and v.numel()>0 for v in rng.values()),'Serialized CPU/CUDA RNG differs')
    return {'completed_steps':completed,'all_optimizer_steps_equal_endpoint':True,'all_state_and_moments_finite':True,
        'cpu_cuda_rng_serialized':True,'simulator_config_verified':True}


def verify_job(directory,job,external,manifest,protocol_sha,gpu_uuid):
    global H
    if H is None:H,_=T.load_helpers(ACTIVE.repo)
    directory=Path(directory);launch=read(ACTIVE.output_dir/'launch.json')
    require(external.get('exit_code')==0 and not external.get('signals') and B.finite(external.get('elapsed_seconds'),True),'Clean stopped child required')
    require(not list(directory.rglob('*.tmp')) and not (directory/'run.lock').exists()
        and not list(directory.rglob('unsuccessful*')) and not list(directory.rglob('state_preservation_error*')),'Unsuccessful/partial artifacts retained, never admitted')
    config,status,history,pointer=[read(directory/n) for n in ('protocol.json','status.json','history.json','latest.json')]
    hashes=launch['files_sha256'];admission=read(ACTIVE.admission)
    require(config.get('schema')==T.SCHEMA and config.get('dataset')=='Goop-3D'
        and all(config.get(k)==job[k] for k in ('objective','arm','seed'))
        and config.get('updates')==config.get('prospective_endpoint_updates')==ACTIVE.updates
        and config.get('checkpoint_every')==ACTIVE.checkpoint_every and config.get('log_every')==ACTIVE.log_every
        and config.get('research_protocol_sha256')==protocol_sha and config.get('selection')==T.ENDPOINT_BASIS
        and config.get('initialization')=='from scratch; paired seed across arms; empty Adam; no parent checkpoint'
        and config.get('source_sha256')=={**T.SOURCE_PINS,'train_goop3d_graph_support_cuda_v2.py':TRAINER_SHA,'goop3d_graph_support_vectorized_v1.py':GRAPH_SHA}
        and config.get('graph_exposure')==T.graph_exposure_config(job['arm']),'Exact fresh scientific configuration required')
    require(config.get('batch_size')==2 and config.get('history')==6 and config.get('noise_std')==6.7e-4
        and config.get('architecture')=={'width':128,'message_passing_blocks':10,'mlp_layers':2}
        and config.get('graph')=={'radius':.025,'backend':'scipy_host','cap':128,'self_candidates':True,'augmentation_probability':0.}
        and config.get('optimizer')=={'name':'Adam','initial_lr':1e-4,'final_lr':1e-5,'decay_updates':ACTIVE.updates,
            'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'foreach':False,'fused':False,'gradient_clipping':None},'Recipe metadata differs')
    data=config['data']
    require(data.get('manifest_sha256')==hashes['train_manifest'] and data.get('admission_sha256')==hashes['admission']
        and data.get('source')==manifest['source'] and data.get('trajectory_ids')==[r['id'] for r in manifest['records']]
        and data.get('n_trajectories')==1000 and data.get('eligible_frames')==295000 and data.get('frames_per_trajectory')==301
        and data.get('dimension')==3 and data.get('particle_type_ids')==[7] and data.get('forecast_horizon_after_six_frames')==295
        and data.get('context_source_sha256')==T.CONTEXT_SOURCE_PINS and data.get('auxiliary_policy')==admission['auxiliary_policy']
        and all(data.get(k)==admission[k] for k in ('structural_report_sha256','converter_sha256','reader_sha256','metadata_sha256',
            'acquisition_report_sha256','context_semantics_sha256','auxiliary_report_sha256','auxiliary_validator_sha256')),'Complete train/context/auxiliary lineage differs')
    runtime=config['runtime']
    require(runtime.get('device')==f"cuda:{job['gpu']}" and B.canonical_gpu_uuid(runtime.get('uuid'))==B.canonical_gpu_uuid(gpu_uuid)
        and runtime.get('torch')=='2.13.0+cu129' and runtime.get('cuda')=='12.9' and runtime.get('threads')==2
        and 'GB200' in runtime.get('name','') and runtime.get('float32_matmul_precision')=='highest'
        and runtime.get('deterministic_algorithms') is True and runtime.get('deterministic_warn_only') is False
        and runtime.get('cublas_workspace_config')==ENVIRONMENT['CUBLAS_WORKSPACE_CONFIG']
        and all(runtime.get(k) is False for k in ('tf32','amp','compile','ddp')),'CUDA runtime/device contract differs')
    require(status.get('schema')==T.SCHEMA and status.get('state')=='complete' and status.get('error') is None
        and status.get('completed_steps')==status.get('committed_steps')==status.get('requested_steps')==ACTIVE.updates
        and status.get('run_config_sha256')==T.config_hash(config) and status.get('process',{}).get('pid')==external['pid']
        and status.get('latest_checkpoint')==pointer and all(status.get(k)==job[k] for k in ('objective','arm','seed')),'Final scientific status differs')
    attempts=list((directory/'attempts').iterdir())
    require(len(attempts)==1 and attempts[0].name==status['attempt_id'] and read(attempts[0]/'status.json')==status,'One fresh complete attempt required')
    details=check_history(history,job,manifest)
    stdout=[json.loads(s) for s in Path(external['stdout_file']).read_text().splitlines() if s.strip()]
    require(stdout==history['training'] and status['last_training']==stdout[-1],'Saved stdout/history/status differs')
    expected={f'checkpoint-{n:09d}.pt' for n in checkpoint_steps(ACTIVE.updates,ACTIVE.checkpoint_every)}
    require({p.name for p in directory.glob('checkpoint-*.pt')}==expected,'Complete arbitrary-endpoint checkpoint cadence required')
    checkpoint_hashes={n:sha(directory/n) for n in sorted(expected)}
    pointers={n:{'path':f'checkpoint-{n:09d}.pt','completed_steps':n,'run_config_sha256':T.config_hash(config),
        'sha256':checkpoint_hashes[f'checkpoint-{n:09d}.pt']} for n in (0,ACTIVE.updates)}
    require(pointer==pointers[ACTIVE.updates] and (external.get('initial_pointer') is None or external['initial_pointer']==pointers[0]),'Checkpoint pointer mismatch')
    final=check_checkpoint(directory/pointer['path'],config,ACTIVE.updates,history,manifest['metadata'])
    initial_payload=H.torch.load(directory/pointers[0]['path'],map_location='cpu',weights_only=True)
    initial_history=initial_payload['history'];del initial_payload
    require(initial_history['training']==[] and initial_history['graph_updates']==[],'Initial checkpoint must precede all updates')
    initial=check_checkpoint(directory/pointers[0]['path'],config,0,initial_history,manifest['metadata'])
    require(all(sha(directory/n)==d for n,d in checkpoint_hashes.items()),'Checkpoint changed during audit')
    return {**job,'status':'verified_scientific_endpoint','directory':str(directory),'config_sha256':T.config_hash(config),
        'runtime':runtime,'external':external,'initial_pointer':pointers[0],'final_pointer':pointer,'history':details,
        'initial_checkpoint':initial,'endpoint':final,'checkpoint_sha256':checkpoint_hashes,
        'initial_pointer_observed_while_running':external.get('initial_pointer') is not None,
        'artifact_sha256':{n:sha(directory/n) for n in ('protocol.json','status.json','history.json','latest.json')}}


def verify_pairing(jobs):
    by={(j['arm'],j['seed']):j for j in jobs};require(len(by)==len(jobs),'Duplicate scientific cell')
    pairs=[]
    for seed in sorted({j['seed'] for j in jobs}):
        require(('base',seed) in by and ('mix',seed) in by,'Both paired arms required')
        a,b=by['base',seed],by['mix',seed]
        left,right=[H.torch.load(Path(j['directory'])/j['initial_pointer']['path'],map_location='cpu',weights_only=True) for j in (a,b)]
        require(all(T.tree_equal(H.torch,left[k],right[k]) for k in ('state_dict','simulator_config','optimizer_state','rng_states')),'Initial paired model/Adam/RNG differs')
        del left,right
        lh,rh=[read(Path(j['directory'])/'history.json') for j in (a,b)]
        require(len(lh['graph_updates'])==len(rh['graph_updates'])==ACTIVE.updates
            and len(lh['training'])==len(rh['training'])==len(logged_steps(ACTIVE.updates,ACTIVE.log_every)),'Complete pairing histories required')
        for x,y in zip(lh['training'],rh['training']):
            require(all(x[k]==y[k] for k in ('completed_steps','frame_ids','particles','lr')),'Paired scalar schedule differs')
        for x,y in zip(lh['graph_updates'],rh['graph_updates']):
            require(all(x[k]==y[k] for k in ('completed_steps','absolute_schedule_step','frame_ids','noise_sha256')),'Paired sample/noise schedule differs')
            for p,q in zip(x['examples'],y['examples']):
                require(all(p[k]==q[k] for k in ('example_slot','n_particles','exposure_coin','coin_seed_material','pair_seed_material',
                    'native_directed_edges','native_self_edges','receivers_above_native_cap','annulus_pairs','optional_budget_if_exposed',
                    'native_edge_sha256','noisy_current_sha256')),'Paired native/noisy/coin/annulus state differs')
        pairs.append({'seed':seed,'initial_model_tensor_identity':True,'initial_cpu_cuda_rng_identity':True,
            'initial_empty_adam_identity':True,'all_frame_noise_lr_schedules_equal':True,'all_graph_budgets_and_rng_material_verified':True,
            'initial_audit_ordering':'verified after training; no before-step1 barrier claimed'})
    return pairs


def run_wave(args, launch, manifest, stop):
    jobs = [j for j in SCHEDULE if j["wave"] == args.host_role]
    require(not B.gpu_processes(), "GPU work exists; root must review before launch")
    children, reaped = [], []
    cleanup_trigger = stop - timedelta(seconds=launch["clock_error_bound_seconds"] + B.CLEANUP_SECONDS)
    started = time.perf_counter()
    record = {"wave": args.host_role, "schema": SCHEMA, "state": "running", "jobs": [], "observations": [],
              "started_utc": B.now().isoformat(), "observation_interval_seconds": 30,
              "cleanup_trigger_utc": cleanup_trigger.isoformat(),
              "child_outcome_verification_scope": "clean selected-endpoint status only; full endpoint audits after every child is reaped"}
    path = args.output_dir / f"wave_{args.host_role}.json"
    B.atomic_json(path, record)
    try:
        for job in jobs:
            require(B.now() + timedelta(seconds=launch["clock_error_bound_seconds"]) <= stamp(launch["latest_start_utc"])
                    and B.now() < cleanup_trigger, "Agreed launch-fit window closed before next child")
            command = fixed_command(args, job)
            stdout_path = args.output_dir / "logs" / (job["id"] + ".stdout.jsonl")
            stderr_path = args.output_dir / "logs" / (job["id"] + ".stderr.txt")
            stdout, stderr = stdout_path.open("xb"), stderr_path.open("xb")
            external_started = time.perf_counter()
            try:
                proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env={**os.environ, **ENVIRONMENT}, start_new_session=True)
            except BaseException:
                stdout.close(); stderr.close()
                raise
            child = {"job": job, "process": proc, "identity": None, "command": command, "handles": (stdout, stderr), "signals": [],
                     "started": external_started, "started_utc": B.now().isoformat(), "stdout_file": str(stdout_path),
                     "stderr_file": str(stderr_path), "initial_pointer": None}
            children.append(child)
            identity = B.process_identity(proc.pid)
            require(identity["argv"] == command and identity["ppid"] == os.getpid(), "Launched scientific child identity differs")
            child["identity"] = identity
            B.atomic_json(args.output_dir / "logs" / (job["id"] + ".launch.json"),
                          {"job": job, "pid": proc.pid, "identity": identity, "command": command, "started_utc": child["started_utc"]})
        record["launch_skew_seconds"] = max(c["started"] for c in children) - min(c["started"] for c in children)
        last_observation = -math.inf
        while any(c["process"].returncode is None for c in children):
            require(B.now() < cleanup_trigger, "Scientific training cleanup cutoff reached; evaluation reserve retained")
            for child in children:
                if child["process"].returncode is None:
                    B.capture_initial(args, child)
                    failure = B.reap_child(args, child, record, reaped, launch, manifest)
                    require(failure is None, failure or "Scientific child failed")
            elapsed = time.perf_counter() - started
            if elapsed - last_observation >= 30:
                last_observation = elapsed
                known = {c["process"].pid: c for c in children}
                processes = B.gpu_processes()
                require(all(p["pid"] in known and B.canonical_gpu_uuid(p["gpu_uuid"]) ==
                    B.canonical_gpu_uuid(launch["gpu_uuids"][known[p["pid"]]["job"]["gpu"]]) for p in processes),
                    "Foreign GPU work or changed GPU assignment detected")
                states = {}
                for child in children:
                    status = args.output_dir / "jobs" / child["job"]["id"] / "status.json"
                    if status.exists():
                        value = read(status)
                        states[str(child["process"].pid)] = {k: value.get(k) for k in ("state", "completed_steps", "committed_steps")}
                        require(value.get("state") not in ("failed", "interrupted"), "Scientific child reported failure")
                record["observations"].append({"utc": B.now().isoformat(), "elapsed_seconds": elapsed,
                                               "gpu_processes": processes, "trainer_states": states})
                B.atomic_json(path, record)
            time.sleep(.2)
        require(len(reaped) == len(jobs), "Every assigned scientific child must be reaped")
        record.update(state="verifying_endpoints", all_children_reaped_utc=B.now().isoformat())
        B.atomic_json(path, record)
        with audit_alarm(cleanup_trigger):
            verified = [verify_job(item["directory"], next(j for j in jobs if j["id"] == item["id"]), item["external"], manifest,
                                   launch["files_sha256"]["protocol"], launch["gpu_uuids"][item["gpu"]]) for item in reaped]
            pairing = verify_pairing(verified)
        record.update(state="verified", pairing=pairing)
        return verified, pairing
    except BaseException as error:
        record.update(state="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        with cleanup_signals_suppressed():
            B.cleanup_owned(args, children, record, reaped, launch, manifest)
        raise
    finally:
        record.update(ended_utc=B.now().isoformat(), wave_elapsed_seconds=time.perf_counter() - started)
        B.atomic_json(path, record)


@contextmanager
def cleanup_signals_suppressed():
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        yield
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def verify_bound_inputs(args,paths,hashes,snapshots,launch,release):
    require(all(sha(p)==hashes[k] for k,p in paths.items())
        and all(sha(args.output_dir/relative)==hashes[k] for k,relative in snapshots.items())
        and sha(args.release)==sha(args.output_dir/'inputs/release.json')==launch['release_sha256']
        and read(args.release)==release and read(args.output_dir/'launch.json')==launch,'Source/input/release/snapshot changed')


def verify_publication_inputs(args,paths,hashes,snapshots,launch,release,jobs):
    for job in jobs:
        directory=Path(job['directory'])
        require(all(sha(directory/n)==d for n,d in {**job['checkpoint_sha256'],**job['artifact_sha256']}.items()),
            'Verified worker artifact changed before cohort publication')
    verify_bound_inputs(args,paths,hashes,snapshots,launch,release)


def verify_wave_receipt(args,wave,launch,jobs):
    record=read(args.output_dir/f'wave_{wave}.json');expected=[j for j in SCHEDULE if j['wave']==wave]
    require(record.get('state')=='verified' and not record.get('unreaped_owned_children')
        and len(record.get('jobs',[]))==len(expected) and {j['id'] for j in record['jobs']}=={j['id'] for j in expected}
        and stamp(record['all_children_reaped_utc'])<=stamp(record['ended_utc']),'Complete reaped wave receipt required')
    for outcome in record['jobs']:
        job=next(j for j in jobs if j['id']==outcome['id']);external=outcome['external']
        require(outcome['verification']=='passed' and external==job['external']
            and external['command']==launch['commands'][job['id']] and not external['signals']
            and read(args.output_dir/'logs'/f"{job['id']}.outcome.json")==outcome,'Wave outcome differs')
        observed=read(args.output_dir/'logs'/f"{job['id']}.launch.json")
        require(observed['job']=={k:job[k] for k in expected[0]} and observed['pid']==external['pid']
            and observed['identity']==external['identity'] and observed['command']==external['command'],'Captured launch identity differs')
    return record


def final_candidates(args,launch,jobs,pairs):
    require(len(jobs)==6 and {(j['arm'],j['seed']) for j in jobs}=={(a,s) for a in ('base','mix') for s in range(3)}
        and len(pairs)==3 and {p['seed'] for p in pairs}=={0,1,2},'Full six-cell cohort required')
    models=[{'objective':'faithful','arm':j['arm'],'seed':j['seed'],'completed_steps':args.updates,
        'graph_history_updates':j['history']['graph_updates'],'checkpoint_sha256':j['final_pointer']['sha256'],
        'checkpoint_path':str(Path(j['directory'])/j['final_pointer']['path']), 'config_sha256':j['config_sha256'],
        'all_optimizer_steps_equal_endpoint':True,'all_state_and_moments_finite':True,
        'source_data_protocol_verified':True,'checkpoint_bytes_verified':True} for j in jobs]
    require(len({m['checkpoint_sha256'] for m in models})==6,'Six distinct completed scientific checkpoints required')
    common={'training_schema':T.SCHEMA,'endpoint_updates':args.updates,'protocol_sha256':launch['files_sha256']['protocol'],
        'trainer_sha256':TRAINER_SHA,'graph_sha256':GRAPH_SHA,'cohort_id':launch['cohort_id'],'models':models}
    audit={**common,'schema':'adaptgns_goop3d_graph_support_complete_cohort_audit_v2',
        'status':'all_six_verified_candidate_requires_root_review','issued_by':'scientific_supervisor_not_root',
        'paired_seeds':pairs,'evaluation_admitted':False}
    B.atomic_json(args.output_dir/'cohort_audit_candidate.json',audit)
    cohort={**common,'schema':'adaptgns_goop3d_graph_support_final_cohort_v2',
        'status':'candidate_requires_root_freeze','issued_by':'scientific_supervisor_not_root',
        'cohort_audit_sha256':sha(args.output_dir/'cohort_audit_candidate.json'),'evaluation_admitted':False}
    B.atomic_json(args.output_dir/'cohort_candidate.json',cohort)
    return {'cohort_candidate.json':sha(args.output_dir/'cohort_candidate.json'),
            'cohort_audit_candidate.json':sha(args.output_dir/'cohort_audit_candidate.json')}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--execute',action='store_true')
    for name in (*FIELDS,'release','output_dir'):p.add_argument('--'+name.replace('_','-'),type=Path)
    p.add_argument('--updates',type=int);p.add_argument('--checkpoint-every',type=int);p.add_argument('--log-every',type=int)
    args=p.parse_args(argv)
    if args.execute:
        require(all(getattr(args,k) is not None for k in (*FIELDS,'release','output_dir','updates','checkpoint_every','log_every'))
            and args.updates>=2 and args.checkpoint_every>0 and args.log_every>0,'Explicit endpoint/cadence and every root-bound input required')
    return args


def main(argv=None):
    args=parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'status':'description_only','scientific_endpoint_selected':False,
            'endpoint_updates':None,'schedule':SCHEDULE,'scientific_training_admitted':False,'test_accessed':False,
            'automatic_resume_retry_or_promotion':False,'process_lifecycle_sha256':LIFECYCLE_SHA},indent=2));return 0
    require(sys.platform.startswith('linux') and hasattr(os,'wait4') and os.environ.get('CUDA_VISIBLE_DEVICES') is None,
        'Dedicated unremapped Linux host required')
    started=time.perf_counter();configure(args);args.output_dir=args.output_dir.resolve()
    paths=input_paths(args);hashes={k:sha(p) for k,p in paths.items()}
    require(hashes['python']==sha(sys.executable),'Supervisor/worker interpreter bytes differ')
    require(all(hashes[k]==v for k,v in V.DATA_PINS.items() if k in hashes)
        and hashes['metadata']==T.METADATA_SHA and hashes['auxiliary_validator']==T.AUXILIARY_VALIDATOR_SHA
        and hashes['original_graph']==V.ORIGINAL_SHA
        and all(hashes['core:'+k]==v for k,v in T.SOURCE_PINS.items())
        and all(hashes['context:'+k]==v for k,v in T.CONTEXT_SOURCE_PINS.items()),'Frozen full training evidence/core/context differs')
    release=read(args.release);plan,windows=validate_release(args,release,hashes)
    T.verify_goop3d_evidence(args,read(args.train_manifest),read(args.admission))
    protected=(args.train_manifest.resolve().parent,(args.repo/'adaptive-gns').resolve())
    require(all(args.output_dir!=p and p not in args.output_dir.parents and args.output_dir not in p.parents for p in protected),
        'Separate fresh scientific output required')
    args.output_dir.mkdir(mode=0o700,parents=True,exist_ok=False)
    for name in ('inputs','jobs','logs'):(args.output_dir/name).mkdir()
    snapshots={}
    for i,(key,path) in enumerate(paths.items()):
        if key=='python':continue
        relative=f'inputs/{i:03d}_{key.replace("/","_").replace(":","_")}{path.suffix}'
        (args.output_dir/relative).write_bytes(path.read_bytes());snapshots[key]=relative
    (args.output_dir/'inputs/release.json').write_bytes(args.release.read_bytes())
    launch={'schema':SCHEMA,'cohort_id':release['cohort_id'],'hostname':socket.gethostname(),'repo':str(args.repo.resolve()),
        'started_utc':B.now().isoformat(),'pid':os.getpid(),'files_sha256':hashes,'input_snapshots':snapshots,
        'release_sha256':sha(args.release),'gpu_uuids':release['gpu_uuids'],'schedule':SCHEDULE,'environment':ENVIRONMENT,
        'clock_error_bound_seconds':release['clock_error_bound_seconds'],'waves':release['waves'],
        'endpoint_updates':args.updates,'cost_basis':plan['cost_basis'],'commands':{j['id']:fixed_command(args,j) for j in SCHEDULE}}
    B.atomic_json(args.output_dir/'launch.json',launch)
    B.atomic_json(args.output_dir/'status.json',{'schema':SCHEMA,'state':'running_scientific_training','endpoint_updates':args.updates,
        'whole_cohort_frozen_for_evaluation':False,'test_accessed':False})
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(signum,frame):raise KeyboardInterrupt('Scientific supervisor received '+signal.Signals(signum).name)
    jobs=[];pairs=[]
    try:
        for sig in previous:signal.signal(sig,interrupted)
        validate_release(args,release,hashes)
        verify_bound_inputs(args,paths,hashes,snapshots,launch,release)
        for wave in ('A','B'):
            require(B.now()<windows[wave]['latest_start_utc'],'Next scientific wave no longer fits; no retry')
            wave_args=SimpleNamespace(**vars(args),host_role=wave)
            wave_launch={**launch,'latest_start_utc':windows[wave]['latest_start_utc'].isoformat()}
            result,pairing=run_wave(wave_args,wave_launch,read(args.train_manifest),windows[wave]['stop_utc'])
            jobs.extend(result);pairs.extend(pairing)
            with audit_alarm(windows[wave]['stop_utc']):
                verify_bound_inputs(args,paths,hashes,snapshots,launch,release)
                verify_wave_receipt(args,wave,launch,result)
        with audit_alarm(windows['B']['stop_utc']):
            verify_publication_inputs(args,paths,hashes,snapshots,launch,release,jobs)
            candidates=final_candidates(args,launch,jobs,pairs)
            summary={'schema':SCHEMA,'status':'all_six_scientific_endpoints_verified','endpoint_updates':args.updates,
                'cohort_id':release['cohort_id'],'cost_basis':plan['cost_basis'],'jobs':jobs,'paired_seeds':pairs,
                'all_inputs_and_worker_artifacts_reverified':True,'candidate_files_sha256':candidates,
                'wall_seconds_from_prelaunch_validation_through_audits':time.perf_counter()-started,
                'sum_observed_child_lifetimes_seconds':sum(j['external']['elapsed_seconds'] for j in jobs),
                'timing_limits':'Child lifetimes overlap within each wave and include observer lag; summing them is not elapsed wall time. Logged guarded times cover only logged updates.',
                'whole_cohort_frozen_for_evaluation':False,'test_accessed':False}
            B.atomic_json(args.output_dir/'summary.json',summary)
            B.atomic_json(args.output_dir/'status.json',{'schema':SCHEMA,'state':'complete_scientific_training',
                'summary_sha256':sha(args.output_dir/'summary.json'),'ended_utc':B.now().isoformat(),
                'whole_cohort_frozen_for_evaluation':False,'test_accessed':False})
        return 0
    except BaseException as error:
        B.atomic_json(args.output_dir/'status.json',{'schema':SCHEMA,'state':'failed_scientific_training',
            'error_type':type(error).__name__,'error':str(error),'ended_utc':B.now().isoformat(),
            'wave_B_released':(args.output_dir/'wave_B.json').exists(),'all_outcomes_retained':True,
            'automatic_retry_resume_or_promotion':False,'whole_cohort_frozen_for_evaluation':False,'test_accessed':False})
        raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':raise SystemExit(main())
