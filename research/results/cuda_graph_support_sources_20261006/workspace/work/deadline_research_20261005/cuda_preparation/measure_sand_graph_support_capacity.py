#!/usr/bin/env python3
"""Root-released per-host Sand base/mix capacity; default describes only.

Uses the unchanged, hash-pinned v2 process lifecycle in a private module.
No SSH, data acquisition, scientific checkpoint promotion or automatic retry.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import traceback

sys.dont_write_bytecode = True
SCHEMA = 'adaptgns_sand_graph_support_capacity_v1'
RELEASE_SCHEMA = 'adaptgns_sand_graph_support_capacity_release_v1'
TRAINING_SCHEMA = 'adaptgns_sand_graph_support_cuda_training_v1'
LIFECYCLE_SHA = 'c5ef3fd3d1df547f5b860c0e01304a48f796058676518a0681735d8936f117cd'
TRAINER_SHA = 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124'
DATA_PINS = {
 'train_manifest':'f133a629a0d94b67875267245ad7bda67464731caf83e2411743a5a85a71bb1f',
 'admission':'fbc8ddd94d90adea6e8f4b07caa73931dfcea41e86c8f2f830e3f01f6a177b73',
 'structural_report':'bc63fabfc07a663ab866f454c07e984dd57b8d4b32aa06a760838c4f84da72e6'}
SCHEDULE = [dict(id=f'{arm}_seed{seed}',wave=role,gpu=gpu,objective='faithful',arm=arm,seed=seed)
 for role,gpu,arm,seed in [('A',0,'base',0),('A',1,'mix',0),('A',2,'base',1),('A',3,'mix',1),('B',0,'base',2),('B',1,'mix',2)]]
ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG':':4096:8','LD_LIBRARY_PATH':'/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
STOP, UPDATES = 512, 100000
COMPUTE_DEADLINE = datetime(2026,10,7,1,tzinfo=timezone.utc)
B = T = None
BASE_FIXED_COMMAND = None


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def private_import(path, expected, name):
    require(sha(path)==expected,'Frozen helper hash differs: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(lifecycle, trainer):
    global B,T,BASE_FIXED_COMMAND
    B=private_import(lifecycle,LIFECYCLE_SHA,'_private_graph_capacity_lifecycle')
    T=private_import(trainer,TRAINER_SHA,'_private_graph_capacity_trainer_definitions')
    BASE_FIXED_COMMAND=B.fixed_command
    B.SCHEDULE=SCHEDULE
    B.verify_job=verify_job
    B.verify_pairing=verify_pairing
    B.fixed_command=fixed_command
    require(T.SCHEMA==TRAINING_SCHEMA,'Unexpected graph trainer schema')


def fixed_command(args,job):return BASE_FIXED_COMMAND(args,job)+['--arm',job['arm']]


def validate_training_rows(rows,counts):
    # Linux/ARM and macOS pow can differ by one float64 ULP. Preserve raw rows
    # and exact paired-arm LR equality; this is only a verifier recomputation bound.
    normalized=[];discrepancies=[]
    for step,row in enumerate(rows,1):
        expected=1e-4*(1e-5/1e-4)**((step-1)/99999)
        value=row.get('lr')
        require(B.finite(value,True) and abs(value-expected)<=math.ulp(expected),'Learning-rate recomputation differs beyond1float64ULP')
        if value!=expected:
            discrepancies.append({'completed_steps':step,'saved_lr':value,'local_recomputed_lr':expected,'float64_ulps':abs(value-expected)/math.ulp(expected)})
        normalized.append({**row,'lr':expected})
    stats=B.validate_rows(normalized,counts)
    return {**stats,'local_lr_recomputation_discrepancies':discrepancies,'maximum_lr_recomputation_float64_ulps':1}


def verify_pairing(jobs):
    by_id={j['id']:j for j in jobs}
    require(len(by_id)==len(jobs),'Duplicate job identity')
    checked=[]
    for seed in sorted({j['seed'] for j in jobs}):
        a,b=by_id.get(f'base_seed{seed}'),by_id.get(f'mix_seed{seed}')
        require(a is not None and b is not None,'Both arms required for each paired seed')
        require(len(a['pairing_rows'])==len(b['pairing_rows'])==STOP,'Complete paired512 rows required')
        for left,right in zip(a['pairing_rows'],b['pairing_rows']):
            require(all(left[k]==right[k] for k in ('completed_steps','frame_ids','particles','lr')),'Paired frame/particle/LR schedule differs')
        require(len(a['graph_rows'])==len(b['graph_rows'])==STOP,'Complete paired graph ledgers required')
        for left,right in zip(a['graph_rows'],b['graph_rows']):
            require(all(left[k]==right[k] for k in ('completed_steps','absolute_schedule_step','frame_ids','noise_sha256')),'Paired graph/frame/noise schedule differs')
            for x,y in zip(left['examples'],right['examples']):
                require(all(x[k]==y[k] for k in ('example_slot','n_particles','exposure_coin','coin_seed_material','pair_seed_material',
                        'native_directed_edges','native_self_edges','receivers_above_native_cap','annulus_pairs','optional_budget_if_exposed',
                        'native_edge_sha256','noisy_current_sha256')),'Paired native graph/noise/exposure evidence differs')
        checked.append({'seed':seed,'rows':STOP,'exact_frame_noise_native_graph_coin_lr_pairing':True,
                        'initial_tensor_pairing':'checkpoint0 retained; separate root tensor audit required before scientific admission'})
    return checked


def verify_job(directory,job,external,manifest,protocol_sha,gpu_uuid):
    directory=Path(directory)
    require(external.get('exit_code')==0 and B.finite(external.get('elapsed_seconds'),True),'Process did not finish successfully')
    require(not(directory/'run.lock').exists() and not list(directory.rglob('*.tmp')),'Lock/partial artifact remains')
    config,status,history,pointer=[read_json(directory/n) for n in ('protocol.json','status.json','history.json','latest.json')]
    config_sha=B.canonical_hash(config)
    require(config.get('schema')==TRAINING_SCHEMA and config.get('dataset')=='Sand' and config.get('objective')=='faithful'
            and config.get('arm')==job['arm'] and type(config.get('seed')) is int and config['seed']==job['seed']
            and config.get('updates')==UPDATES and config.get('batch_size')==2 and config.get('history')==6
            and config.get('checkpoint_every')==10000 and config.get('log_every')==1
            and config.get('research_protocol_sha256')==protocol_sha,'Graph training identity/configuration differs')
    require(config.get('source_sha256')=={**T.SOURCE_PINS,'train_sand_graph_support_cuda.py':TRAINER_SHA},'Trainer/core lineage differs')
    require(config.get('architecture')=={'width':128,'message_passing_blocks':10,'mlp_layers':2} and config.get('noise_std')==6.7e-4
            and config.get('graph_exposure')==T.graph_exposure_config(job['arm'])
            and config.get('initialization')=='from scratch; paired seed across arms; empty Adam; no parent checkpoint'
            and config.get('graph')=={'radius':.015,'backend':'scipy_host','cap':128,'self_candidates':True,'augmentation_probability':0.}
            and config.get('optimizer')=={'name':'Adam','initial_lr':1e-4,'final_lr':1e-5,'decay_updates':100000,'betas':[.9,.999],
                'eps':1e-8,'weight_decay':0.,'foreach':False,'fused':False,'gradient_clipping':None},'Graph-support recipe differs')
    data=config.get('data',{})
    require(data.get('manifest_sha256')==DATA_PINS['train_manifest'] and data.get('admission_sha256')==DATA_PINS['admission']
            and data.get('structural_report_sha256')==DATA_PINS['structural_report'] and data.get('frames_per_trajectory')==320
            and data.get('particle_type_ids')==[6],'Training data lineage differs')
    runtime=config.get('runtime',{})
    require(runtime.get('device')==f"cuda:{job['gpu']}" and B.canonical_gpu_uuid(runtime.get('uuid'))==B.canonical_gpu_uuid(gpu_uuid)
            and runtime.get('torch')=='2.13.0+cu129' and runtime.get('cuda')=='12.9' and runtime.get('threads')==2
            and runtime.get('deterministic_algorithms') is True and runtime.get('deterministic_warn_only') is False
            and runtime.get('cublas_workspace_config')==ENVIRONMENT['CUBLAS_WORKSPACE_CONFIG']
            and all(runtime.get(k) is False for k in ('tf32','amp','compile','ddp')),'CUDA runtime/GPU mapping differs')
    require(status.get('schema')==TRAINING_SCHEMA and status.get('state')=='planned_stop_incomplete'
            and type(status.get('completed_steps')) is int and status['completed_steps']==STOP
            and type(status.get('committed_steps')) is int and status['committed_steps']==STOP
            and status.get('requested_steps')==UPDATES and status.get('error') is None and status.get('objective')=='faithful'
            and status.get('arm')==job['arm'] and status.get('seed')==job['seed'] and status.get('run_config_sha256')==config_sha
            and status.get('process',{}).get('pid')==external['pid'],'Only committed planned_stop_incomplete512 is accepted')
    attempts=list((directory/'attempts').iterdir())
    require(len(attempts)==1 and attempts[0].is_dir() and attempts[0].name==status.get('attempt_id')
            and read_json(attempts[0]/'status.json')==status,'Exactly one consistent fresh attempt required')
    require(not list(directory.rglob('failed_attempt*')) and not list(directory.rglob('unsuccessful*'))
            and not list(directory.rglob('state_preservation_error*')),'A failed attempt cannot be promoted')
    expected={'path':f'checkpoint-{STOP:09d}.pt','completed_steps':STOP,'run_config_sha256':config_sha,'sha256':sha(directory/f'checkpoint-{STOP:09d}.pt')}
    require(pointer==expected and status.get('latest_checkpoint')==pointer,'Final pointer/checkpoint/status differ')
    initial=external.get('initial_pointer')
    require(initial=={'path':'checkpoint-000000000.pt','completed_steps':0,'run_config_sha256':config_sha,'sha256':sha(directory/'checkpoint-000000000.pt')},'Initial pointer/checkpoint differs')
    T.validate_graph_history(history,STOP,job['arm'],job['seed'])
    rows=history['training'];graph_rows=history['graph_updates']
    counts={r['id']:r['positions']['shape'][1] for r in manifest['records']}
    stats=validate_training_rows(rows,counts)
    for training,graph in zip(rows,graph_rows):
        require(training['completed_steps']==graph['completed_steps'] and training['frame_ids']==graph['frame_ids'],'Scalar and graph history identity differs')
        require(sum(e['n_particles'] for e in graph['examples'])==training['particles'],'Graph particle counts differ')
        for slot,example in enumerate(graph['examples']):
            trajectory=training['frame_ids'][slot].rsplit(':',1)[0]
            require(example['n_particles']==counts[trajectory],'Per-example particle count differs')
    stdout=[json.loads(line) for line in Path(external['stdout_file']).read_text().splitlines() if line.strip()]
    require(stdout==rows and status.get('last_training')==rows[-1] and history['elapsed_seconds']>=rows[-1]['elapsed_seconds'],'Scalar stdout/history/status differ')
    residual=external['elapsed_seconds']-stats['sum_all512_guarded_seconds']
    require(residual>=-1e-6,'External timing shorter than guarded work')
    return {**job,'status':'verified_512_graph_timing_probe','config_sha256':config_sha,'gpu_uuid':gpu_uuid,'runtime':runtime,
            'external':external,'initial_pointer':initial,'final_pointer':pointer,
            'artifact_sha256':{n:sha(directory/n) for n in ('protocol.json','status.json','history.json','latest.json')},
            **stats,'external_minus_all_guarded_seconds':residual,'nonnegative_external_minus_all_guarded_seconds':max(0.,residual),
            'pairing_rows':rows,'graph_rows':graph_rows}


def validate_release(release,hashes,role):
    require(release.get('schema')==RELEASE_SCHEMA and release.get('status')=='admitted_for_timing' and release.get('issued_by')=='root'
            and release.get('scientific_training_admitted') is False and release.get('files_sha256')==hashes
            and release.get('schedule')==SCHEDULE and release.get('environment')==ENVIRONMENT and release.get('host_role')==role
            and isinstance(release.get('cohort_id'),str) and bool(release['cohort_id'].strip())
            and isinstance(release.get('review_rationale'),str) and bool(release['review_rationale'].strip()),'Exact root timing release required')
    uuids=release.get('gpu_uuids')
    require(isinstance(uuids,list) and len(uuids)==4 and len({B.canonical_gpu_uuid(u) for u in uuids})==4,'Four unique physical UUIDs required per host')
    require(B.finite(release.get('clock_error_bound_seconds')) and release['clock_error_bound_seconds']<=5,'Root-attested host UTC error bound0..5seconds required')
    return uuids


def verify_inputs(root):
    launch=read_json(root/'launch.json')
    require(launch.get('schema')==SCHEMA and launch.get('schedule')==SCHEDULE and launch.get('environment')==ENVIRONMENT,'Launch contract differs')
    hashes=launch['files_sha256']; snapshots=launch['input_snapshots']
    required={'trainer','lifecycle_source','train_manifest','admission','structural_report','protocol','numerical_gate','python','supervisor'}
    require(set(hashes)==required and set(snapshots)==required-{'python'},'Complete input snapshot required')
    for key,relative in snapshots.items():
        path=(root/relative).resolve()
        require(path.parent==(root/'inputs').resolve() and sha(path)==hashes[key],'Copied input hash/path differs: '+key)
    require(hashes['supervisor']==sha(__file__) and hashes['trainer']==TRAINER_SHA and hashes['lifecycle_source']==LIFECYCLE_SHA
            and all(hashes[k]==v for k,v in DATA_PINS.items()),'Frozen source/data differs')
    require(sha(root/'inputs/release.json')==launch.get('release_sha256'),'Release snapshot differs')
    release=read_json(root/'inputs/release.json')
    require(validate_release(release,hashes,launch['host_role'])==launch['gpu_uuids'] and release['cohort_id']==launch['cohort_id'],'Launch/release identity differs')
    gate=read_json(root/snapshots['numerical_gate'])
    require(gate.get('schema')=='adaptgns_sand_graph_support_cuda_numerical_admission_v1' and gate.get('status')=='admitted_for_timing_only'
            and gate.get('issued_by')=='root' and gate.get('trainer_sha256')==TRAINER_SHA and gate.get('scientific_training_admitted') is False,
            'Distinct root graph-support numerical timing admission required')
    return launch,release


def steady_interval(root,role,clock_error):
    record=read_json(root/f'wave_{role}.json')
    pids={str(o['external']['pid']) for o in record['jobs']}
    valid=[]
    for obs in record['observations']:
        states=obs['trainer_states']
        if set(states)==pids and {str(p['pid']) for p in obs['gpu_processes']}==pids and all(s['state']=='running' and type(s['completed_steps']) is int and 64<=s['completed_steps']<STOP for s in states.values()):
            value=datetime.fromisoformat(obs['utc'])
            require(value.tzinfo is not None,'UTC-aware observations required')
            valid.append(value)
    require(len(valid)>=2,'At least two complete steady observations required for cross-host overlap')
    return min(valid)+timedelta(seconds=clock_error),max(valid)-timedelta(seconds=clock_error)


def verify_worker_completion(root,role):
    status=read_json(root/'status.json')
    require(status.get('state')=='complete_host_capacity_measurement' and status.get('host_role')==role
            and status.get('scientific_training_admitted') is False
            and status.get('summary_sha256')==sha(root/'worker_summary.json'),'Worker did not complete final integrity checks')
    summary=read_json(root/'worker_summary.json')
    require(summary.get('schema')==SCHEMA and summary.get('status')=='verified_static_host_wave' and summary.get('host_role')==role
            and summary.get('scientific_training_admitted') is False
            and len(summary.get('jobs',[]))==len([j for j in SCHEDULE if j['wave']==role])
            and {j['id'] for j in summary['jobs']}=={j['id'] for j in SCHEDULE if j['wave']==role},'Worker summary membership/state differs')
    return summary


def cost_contract(path):
    if path is None:return None
    c=read_json(path); policies=c.get('policy_count')
    require(c.get('schema')=='adaptgns_sand_graph_support_cost_estimate_v1' and c.get('status')=='complete_workload_estimated'
            and policies==6 and c.get('rollout_outcomes')==1080 and c.get('horizon')==314
            and c.get('diagnostic_forward_calls')==124704
            and all(B.finite(c.get(k),True) for k in ('full_rollout_seconds','diagnostics_execution_seconds','ledger_total_reserve_seconds'))
            and isinstance(c.get('ledger_bound_rationale'),str) and bool(c['ledger_bound_rationale'].strip())
            and isinstance(c.get('timing_evidence_sha256'),list) and bool(c['timing_evidence_sha256'])
            and all(T.digest_string(v) for v in c['timing_evidence_sha256']),'Complete frozen rollout/diagnostic/ledger cost estimate required')
    return c


def forecast(jobs,costs=None,at=None):
    require(len(jobs)==6 and {j['id'] for j in jobs}=={j['id'] for j in SCHEDULE},'Complete six-job capacity required')
    q6=max(j['steady_wall_seconds_per_update'] for j in jobs);r6=max(j['nonnegative_external_minus_all_guarded_seconds'] for j in jobs)
    require(B.finite(q6,True) and B.finite(r6),'Invalid measured timing')
    core=1.35*(100000*q6+12*r6)
    training=None if costs is None else core+costs['ledger_total_reserve_seconds']
    total=None if costs is None else training+costs['full_rollout_seconds']+costs['diagnostics_execution_seconds']+3600
    at=B.now() if at is None else at
    require(at.tzinfo is not None,'Timezone-aware forecast origin required')
    finish=None if total is None else at+timedelta(seconds=total)
    return {'formula':'1.35*(100000*q6+12*r6)+ledger_total_reserve_seconds','q6':q6,'r6':r6,
            'core_training_and_short_probe_overhead_seconds':core,'training_including_terminal_ledger_seconds':training,
            'costs':costs,'verification_allowance_seconds':3600,'total_compute_analysis_seconds':total,
            'forecast_origin_utc':at.isoformat(),'forecast_finish_utc':None if finish is None else finish.isoformat(),
            'deadline_utc':COMPUTE_DEADLINE.isoformat(),'fits_before_writing_reserve':None if finish is None else finish<=COMPUTE_DEADLINE,
            'interpretation':'Nominal six-job measured concurrency, not staged q4+q2. Contingencies are planning allowances, not guarantees. No scientific admission.'}


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--combine',type=Path,nargs=2)
    parser.add_argument('--host-role',choices=('A','B'))
    for name in ('release','repo','trainer','lifecycle-source','train-manifest','admission','structural-report','protocol','numerical-gate','output-dir','python','costs','summary-output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args(argv)
    if args.execute and any(getattr(args,k) is None for k in ('host_role','release','repo','trainer','lifecycle_source','train_manifest','admission','structural_report','protocol','numerical_gate','output_dir','python')):
        parser.error('Execution requires explicit role, release and every input/source/interpreter/fresh-output path')
    if args.combine and (args.lifecycle_source is None or args.trainer is None):parser.error('Combination requires pinned lifecycle-source and trainer definitions')
    return args


def main(argv=None):
    args=parse_args(argv)
    if not args.execute and args.combine is None:
        print(json.dumps({'schema':SCHEMA,'execute':False,'schedule':SCHEDULE,'stop_after':STOP,'updates':UPDATES,
                          'scientific_training_admitted':False,'formula':'1.35*(100000*q6+12*r6)+ledger_total_reserve_seconds'},indent=2));return 0
    configure(args.lifecycle_source,args.trainer)
    if args.combine:
        roots=[p.resolve() for p in args.combine];workers=[];jobs=[];intervals=[]
        require(roots[0]!=roots[1],'Two distinct worker roots required')
        for root in roots:
            launch,release=verify_inputs(root);role=launch['host_role']
            completion=verify_worker_completion(root,role)
            require(completion.get('cohort_id')==launch['cohort_id'],'Worker completion cohort differs')
            outcomes=B.verify_wave_record(root,role,launch)
            manifest=read_json(root/launch['input_snapshots']['train_manifest'])
            values=[]
            for outcome in outcomes:
                job=next(j for j in SCHEDULE if j['id']==outcome['id']);external=dict(outcome['external'])
                external['stdout_file']=str(root/'logs'/(job['id']+'.stdout.jsonl'))
                values.append(verify_job(root/'jobs'/job['id'],job,external,manifest,launch['files_sha256']['protocol'],launch['gpu_uuids'][job['gpu']]))
            verify_pairing(values);jobs.extend(values)
            intervals.append(steady_interval(root,role,release['clock_error_bound_seconds']))
            workers.append({'root':str(root),'launch':launch,'release':release,'launch_sha256':sha(root/'launch.json'),'wave_sha256':sha(root/f'wave_{role}.json')})
        require({w['launch']['host_role'] for w in workers}=={'A','B'} and len({w['launch']['cohort_id'] for w in workers})==1,'One hostA/B of one frozen cohort required')
        require(len({w['launch']['hostname'] for w in workers})==2,'Distinct host identities required')
        require(len({B.canonical_gpu_uuid(u) for w in workers for u in w['launch']['gpu_uuids']})==8,'Two distinct four-GPU inventories required')
        for key in ('trainer','lifecycle_source','train_manifest','admission','structural_report','protocol','numerical_gate','supervisor'):
            require(len({w['launch']['files_sha256'][key] for w in workers})==1,'Cross-host input mismatch: '+key)
        for key in ('torch','cuda','numpy','scipy','python','capability','float32_matmul_precision'):
            require(len({json.dumps(j['runtime'].get(key),sort_keys=True) for j in jobs})==1,'Cross-host runtime differs: '+key)
        begin=max(i[0] for i in intervals);end=min(i[1] for i in intervals)
        require(end>begin,'No conservative cross-host steady overlap; do not claim six-job capacity')
        costs=cost_contract(args.costs)
        summary={'schema':SCHEMA,'status':'all_six_verified','scientific_training_admitted':False,'workers':workers,'jobs':jobs,
                 'pairing':verify_pairing(jobs),'conservative_cross_host_overlap_utc':[begin.isoformat(),end.isoformat()],
                 'forecast':forecast(jobs,costs),'cost_estimate_sha256':sha(args.costs) if args.costs else None}
        if args.summary_output:
            require(not args.summary_output.exists(),'Preserve previous forecast snapshots; choose fresh output')
            B.atomic_json(args.summary_output,summary);print(json.dumps({'summary':str(args.summary_output),'sha256':sha(args.summary_output),'forecast':summary['forecast']},indent=2))
        else:print(json.dumps(summary,indent=2))
        return 0
    require(sys.platform.startswith('linux') and hasattr(os,'wait4'),'Per-host execution requires Linux')
    require(B.now()<COMPUTE_DEADLINE and os.environ.get('CUDA_VISIBLE_DEVICES') is None,'Deadline/remapping violation')
    paths={k:getattr(args,k).resolve() for k in ('trainer','lifecycle_source','train_manifest','admission','structural_report','protocol','numerical_gate','python')}
    paths['supervisor']=Path(__file__).resolve();hashes={k:sha(p) for k,p in paths.items()}
    require(all(hashes[k]==v for k,v in DATA_PINS.items()),'Frozen data/admission differs')
    require(all(sha(args.repo/r)==v for r,v in T.SOURCE_PINS.items()),'Frozen graph numerical source differs')
    release=read_json(args.release);uuids=validate_release(release,hashes,args.host_role)
    checked=datetime.fromisoformat(release['process_identity_checked_utc'])
    require(checked.tzinfo is not None and -60<=(B.now()-checked).total_seconds()<=300,'Fresh root process check required')
    args.output_dir=args.output_dir.resolve()
    protected=[args.train_manifest.resolve().parent,(args.repo/'adaptive-gns').resolve()]
    require(all(args.output_dir!=p and p not in args.output_dir.parents and args.output_dir not in p.parents for p in protected),'Output must be separate from protected input trees')
    args.output_dir.mkdir(mode=0o700,exist_ok=False)
    for name in ('inputs','jobs','logs'):(args.output_dir/name).mkdir()
    snapshots={}
    for key,path in paths.items():
        if key!='python':
            snapshots[key]='inputs/'+key+path.suffix;(args.output_dir/snapshots[key]).write_bytes(path.read_bytes())
    (args.output_dir/'inputs/release.json').write_bytes(args.release.read_bytes())
    launch={'schema':SCHEMA,'host_role':args.host_role,'cohort_id':release['cohort_id'],'files_sha256':hashes,'input_snapshots':snapshots,
            'schedule':SCHEDULE,'environment':ENVIRONMENT,'gpu_uuids':uuids,'release_sha256':sha(args.release),'hostname':socket.gethostname(),
            'pid':os.getpid(),'started_utc':B.now().isoformat(),'commands':{j['id']:fixed_command(args,j) for j in SCHEDULE}}
    B.atomic_json(args.output_dir/'launch.json',launch)
    try:
        verify_inputs(args.output_dir)
        values,record=B.run_wave(args,args.host_role,launch,read_json(args.train_manifest))
        require(all(sha(p)==hashes[k] for k,p in paths.items()) and all(sha(args.repo/r)==v for r,v in T.SOURCE_PINS.items()),'Inputs/core changed during measurement')
        verify_inputs(args.output_dir);B.verify_wave_record(args.output_dir,args.host_role,launch)
        B.atomic_json(args.output_dir/'worker_summary.json',{'schema':SCHEMA,'status':'verified_static_host_wave','host_role':args.host_role,
            'cohort_id':release['cohort_id'],'jobs':values,'pairing':verify_pairing(values),'scientific_training_admitted':False,
            'requires':'Other host complete and cross-host overlap check before six-job forecast'})
        B.atomic_json(args.output_dir/'status.json',{'state':'complete_host_capacity_measurement','host_role':args.host_role,'ended_utc':B.now().isoformat(),
            'summary_sha256':sha(args.output_dir/'worker_summary.json'),'scientific_training_admitted':False});return 0
    except BaseException as error:
        B.atomic_json(args.output_dir/'status.json',{'state':'failed_host_capacity_measurement','host_role':args.host_role,'at_utc':B.now().isoformat(),
            'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'all_outcomes_retained':True,'scientific_training_admitted':False})
        raise


if __name__=='__main__':raise SystemExit(main())
