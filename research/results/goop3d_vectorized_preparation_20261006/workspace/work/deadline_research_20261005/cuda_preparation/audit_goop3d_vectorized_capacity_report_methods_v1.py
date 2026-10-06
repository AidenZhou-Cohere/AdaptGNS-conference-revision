"""Independent actual capacity scalar/byte audit; no arrays, models or remote calls."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import re
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent
checks=0

def check(v,label):
    global checks
    checks+=1
    if not v:raise AssertionError(label)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def digest(v):return isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) is not None
def stamp(v):return datetime.fromisoformat(v)
def finite(v,positive=False):return type(v) in (int,float) and math.isfinite(v) and (v>0 if positive else v>=0)
def integer(v,lower=0):return type(v) is int and v>=lower

def constants(name):
    out={}
    for node in ast.parse((ROOT/name).read_text()).body:
        if isinstance(node,ast.Assign):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            for target in node.targets:
                if isinstance(target,ast.Name):out[target.id]=value
    return out

summary_name='goop3d_vectorized_capacity_v1_summary.json'
release_name='goop3d_vectorized_capacity_release_v1.json'
archive_name='goop3d_vectorized_capacity_v1_scalar_snapshot.tar.gz'
inputs={summary_name:'39f048a7257428b0ffe342e52ed2f956e90cd5dec4463835b40a9d75855a47e1',
 release_name:'d9c9d5c717cc6695da04f4f0a98f465c06eaeea8d6431c82d3ba228bf5a50441',
 archive_name:'0886efa92a4cd08f427ff45ba369bbc52eef1d0bfd206421f0640ae6897162b8',
 'goop3d_capacity_stopped_receipt_v1.json':'40f13d2e66d94cf607e3bf21b69dac135fd6458693bb9ef2e2cfb367cece9fa5',
 'goop3d_capacity_final_artifact_inventory_v1.json':'df741af7c0e69d6050d2de7d3fc1c82a1628d1f0b6efa4cb8ed07b298416f248',
 'third_vm_capacity_stopped_valid_bytes_v1.json':'d886dffb43c98832b922901f2fe399b7ffa98d81f4bfd92385c048f13fd9d9a1',
 'third_vm_pre_capacity_inventory_v2.json':'068e555dfe3d0bbef57f0592111598b557f7d18167627e394fb6b5ac70651f90'}
s,r=read(summary_name),read(release_name)
source_files={'train_manifest':'goop3d_structural_metadata_v1/train.json',
 'structural_report':'goop3d_structural_metadata_v1/structural_report.json','acquisition_report':'goop3d_acquisition_v1_report.json',
 'context_semantics':'goop3d_context_semantics_review.json','auxiliary_report':'goop3d_auxiliary_v1_report.json',
 'data_review':'goop3d_data_compatibility_root_review_v1.json','cpu_proof':'goop3d_vectorized_cpu_comparison_v1_report.json',
 'protocol':'goop3d_vectorized_capacity_protocol_v1.md','numerical_report':'goop3d_vectorized_numerical_v1_report.json',
 'supervisor':'measure_goop3d_vectorized_capacity_v1.py','trainer':'train_goop3d_graph_support_cuda_v2.py',
 'graph':'goop3d_graph_support_vectorized_v1.py','validator':'validate_goop3d_vectorized_cuda_v1.py','lifecycle':'measure_sand_cuda_capacity_v2.py'}
inputs.update({name:r['files_sha256'][k] for k,name in source_files.items()})
inputs['goop3d_vectorized_numerical_v1_independent_review_methods.json']=r['independent_review_sha256']
inputs['goop3d_numerical_remote_artifact_audit_v1.json']=r['numerical_artifact_audit_sha256']
for name,value in inputs.items():check(sha(ROOT/name)==value,'Input hash '+name)
# Check frozen core sources separately, without importing trainer or numerical libraries.
t=constants(source_files['trainer']);repo=ROOT.parents[2]/'outputs'/'AdaptGNS'
core={str(repo/name):value for name,value in t['SOURCE_PINS'].items()}
for name,value in core.items():check(sha(name)==value,'Core source '+name)
manifest=read(source_files['train_manifest']);records=manifest['records']
check(len(records)==manifest['record_count']==1000 and manifest['split']=='train','Complete train source')
check(manifest['source']['sha256']==r['data_contract']['source_sha256'],'Train original source SHA')
counts={x['id']:x['positions']['shape'][1] for x in records}
check(len(counts)==1000 and all(x['positions']['shape'][0::2]==[301,3] for x in records),'Train D3 full frames')
expected=[dict(id=f'{a}_seed{seed}',wave=w,gpu=g,objective='faithful',arm=a,seed=seed)
 for w,g,a,seed in [('A',0,'base',0),('A',1,'mix',0),('A',2,'base',1),('A',3,'mix',1),('B',0,'base',2),('B',1,'mix',2)]]
check(r['schedule']==expected and r['updates_per_job']==512 and r['warmup_updates']==64 and r['probe_lr_horizon']==100000,'Fixed capacity release')
check(r['issued_by']=='root' and r['status']=='admitted_for_capacity_only' and r['purpose']=='bounded_capacity_only_never_promote','Root bounded release')
check(s['schema']=='adaptgns_goop3d_vectorized_capacity_v1' and s['status']=='all_six_verified_capacity_only','Complete summary')
for obj in (s,r):check(obj['scientific_training_admitted'] is False and obj['scientific_endpoint_selected'] is False,'No science admission')
check(s['scientific_endpoint_updates'] is None and s['full_study_runtime_forecast_seconds'] is None,'No endpoint/forecast')
check(s['all_inputs_and_worker_artifacts_reverified'] is True and len(s['jobs'])==6,'Complete verified six cells')
inv=read('goop3d_capacity_final_artifact_inventory_v1.json');files=inv['files']
check(inv['status']=='all_saved_artifacts_rehashed' and inv['summary_sha256']==inputs[summary_name],'Remote byte inventory summary binding')
archive_prefix='goop3d_vectorized_capacity_20261006_v1/'
with tarfile.open(ROOT/archive_name) as archive:
    blobs={m.name[len(archive_prefix):]:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile() and m.name.startswith(archive_prefix)}
check(len(blobs)==74 and len(files)==86 and set(files)-set(blobs)=={f'jobs/{j["id"]}/checkpoint-{n:09d}.pt' for j in expected for n in (0,512)},'74 scalar artifacts and12 omitted checkpoints')
for name,data in blobs.items():
    check(name in files and hashlib.sha256(data).hexdigest()==files[name]['sha256'] and len(data)==files[name]['size_bytes'],'Archived bytes '+name)
def saved(name):return json.loads(blobs[name])
launch=saved('launch.json')
check(saved('summary.json')==s and saved('inputs/release.json')==r,'Exact archived summary/release')
check(launch['schedule']==expected and launch['files_sha256']==r['files_sha256'] and launch['release_sha256']==inputs[release_name],'Launch input bindings')
check(launch['gpu_uuids']==r['gpu_uuids'] and launch['environment']==r['environment'],'Launch devices/environment')
for key,path in launch['input_snapshots'].items():check(hashlib.sha256(blobs[path]).hexdigest()==r['files_sha256'][key],'Input snapshot '+key)
check(saved('status.json')['state']=='complete_capacity_only' and saved('status.json')['summary_sha256']==inputs[summary_name],'Final status')
computed=[]
for job,exp in zip(s['jobs'],expected):
    check(all(job[k]==v for k,v in exp.items()) and job['status']=='verified_capacity_only','Scientific cell identity')
    name=job['id'];base=f'jobs/{name}/';config=saved(base+'protocol.json');status=saved(base+'status.json');history=saved(base+'history.json')
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    check(config['source_and_input_sha256']==r['files_sha256'] and config['release_sha256']==inputs[release_name],'Config inputs')
    check(all(config[k]==v for k,v in exp.items()) and config['updates_per_job']==512 and config['probe_lr_horizon']==100000 and config['warmup_updates']==64,'Config schedule')
    check(config['batch_size']==2 and config['history']==6 and config['radius']==.025 and config['noise_std']==6.7e-4
        and config['architecture']=={'width':128,'message_passing_blocks':10,'mlp_layers':2},'Full architecture')
    data=config['data'];check(data['source']==manifest['source'] and data['trajectory_ids']==[x['id'] for x in records]
        and data['n_trajectories']==1000 and data['eligible_frames']==295000 and data['manifest_sha256']==r['files_sha256']['train_manifest'],'Complete data lineage')
    rt=config['runtime'];check(rt['device']==f'cuda:{job["gpu"]}' and 'GPU-'+rt['uuid']==r['gpu_uuids'][job['gpu']]
        and rt['torch']=='2.13.0+cu129' and rt['cuda']=='12.9' and rt['threads']==2 and rt['name']=='NVIDIA GB200','Runtime device')
    check(rt['deterministic_algorithms'] is True and rt['deterministic_warn_only'] is False and rt['float32_matmul_precision']=='highest'
        and rt['cublas_workspace_config']==':4096:8' and all(rt[k] is False for k in ('tf32','amp','compile','ddp')),'Determinism runtime')
    rows,graphs=job['pairing_rows'],job['graph_rows'];check(rows==history['training'] and graphs==history['graph_updates'] and len(rows)==len(graphs)==512,'Complete scalar/graph ledger')
    check([json.loads(x) for x in blobs[f'logs/{name}.stdout.jsonl'].decode().splitlines() if x.strip()]==rows,'Stdout scalar identity')
    for key,value in job['artifact_sha256'].items():check(files[base+key]['sha256']==value,'Worker artifact inventory '+key)
    for label,step in [('initial_pointer',0),('final_pointer',512)]:
        ptr=job[label];check(ptr['path']==f'checkpoint-{step:09d}.pt' and ptr['completed_steps']==step and ptr['capacity_config_sha256']==config_sha
            and digest(ptr['sha256']) and ptr['sha256']==files[base+ptr['path']]['sha256'],'Checkpoint pointer/config/remote-byte receipt')
    for endpoint in ('initial','final'):
        check(all(job[endpoint][flag] is True for flag in ('all_model_and_Adam_values_finite','all_Adam_steps_exact'))
            and all(digest(job[endpoint][key]) for key in ('model_tensor_sha256','rng_sha256')),'Endpoint reported finite/tensor digest')
    ex=job['external'];identity=ex['identity'];command=ex['command']
    check(ex['exit_code']==0 and ex['signals']==[] and finite(ex['elapsed_seconds'],True),'Stopped clean process')
    check(ex['pid']==identity['pid']==status['process']['pid'] and identity['ppid']==1963 and integer(identity['start_ticks'],1)
        and identity['argv']==command==launch['commands'][name] and identity['executable']=='/usr/bin/python3.12','PID/parent/start/argv identity')
    check(command[command.index('--job')+1]==name and command[command.index('--output-dir')+1]==archive_prefix+'jobs/'+name
        and command[command.index('--mode')+1]=='worker','Bounded worker command')
    check(stamp(ex['ended_utc'])>stamp(ex['started_utc']) and abs((stamp(ex['ended_utc'])-stamp(ex['started_utc'])).total_seconds()-ex['elapsed_seconds'])<.1,'External elapsed timestamp')
    check(integer(ex['peak_host_rss_bytes'],1) and ex['peak_host_rss_bytes']%1024==0 and ex['peak_host_rss_source_units']=='Linux ru_maxrss KiB, multiplied by1024','Linux RSS units')
    check(ex['initial_pointer']==job['initial_pointer'] and saved(base+'latest.json')==job['final_pointer'],'Observed checkpoint pointers')
    check(status['state']=='complete_capacity_only' and status['error'] is None and status['completed_steps']==status['committed_steps']==512
        and status['capacity_config_sha256']==config_sha and status['latest_checkpoint']==job['final_pointer'] and status['last_training']==rows[-1],'Worker final status')
    observed=saved(f'logs/{name}.launch.json');outcome=saved(f'logs/{name}.outcome.json')
    check(observed['identity']==identity and observed['command']==command and observed['pid']==ex['pid'] and outcome['external']==ex and outcome['verification']=='passed','Launch/outcome receipt')
    previous=0.;expanded_count=0;lr_ulps=[]
    for step,(row,graph) in enumerate(zip(rows,graphs)):
        rng=np.random.default_rng(np.random.SeedSequence([job['seed'],step,1103]));indices=rng.integers(295000,size=2,dtype=np.int64).tolist()
        ids=[records[x//295]['id']+':'+str(x%295+6) for x in indices]
        check(row['completed_steps']==graph['completed_steps']==step+1 and graph['absolute_schedule_step']==step,'Exact update sequence')
        check(row['frame_ids']==graph['frame_ids']==ids,'Independent host sample schedule')
        check(finite(row['guarded_update_seconds'],True) and finite(row['elapsed_seconds'],True) and row['elapsed_seconds']>previous
            and math.isfinite(row['loss']),'Finite monotone timing/loss')
        expected_lr=1e-4*(.1**(step/99999));lr_ulps.append(abs(row['lr']-expected_lr)/math.ulp(expected_lr))
        check(lr_ulps[-1]<=1,'Cross-platform LR formula differs by more than1 float64 ULP')
        check(len(graph['examples'])==2 and row['particles']==sum(counts[x.rsplit(':',1)[0]] for x in ids) and digest(graph['noise_sha256']),'Scalar particles/noise digest')
        for slot,e in enumerate(graph['examples']):
            coin=bool(np.random.default_rng(np.random.SeedSequence([20261005,job['seed'],step,slot,4409])).random()<.5)
            active=job['arm']=='mix' and coin
            check(e['example_slot']==slot and e['n_particles']==counts[ids[slot].rsplit(':',1)[0]],'Graph slot/count')
            check(e['exposure_coin'] is coin and e['expanded'] is active and e['coin_seed_material']==[20261005,job['seed'],step,slot,4409]
                and e['pair_seed_material']==[20261005,job['seed'],step,slot,5501],'Independent graph RNG schedule')
            check(integer(e['annulus_pairs']) and e['optional_budget_if_exposed']==e['annulus_pairs']//4
                and e['selected_optional_pairs']==(e['annulus_pairs']//4 if active else 0),'Exact annulus budget/exposure')
            check(integer(e['native_directed_edges']) and e['native_self_edges']==e['n_particles'] and integer(e['receivers_above_native_cap'])
                and all(digest(e[k]) for k in ('native_edge_sha256','optional_pair_sha256','noisy_current_sha256')),'Graph counts/digests')
            expanded_count+=active
        previous=row['elapsed_seconds']
    q=(rows[-1]['elapsed_seconds']-rows[63]['elapsed_seconds'])/448;guard=sum(x['guarded_update_seconds'] for x in rows);residual=max(0.,ex['elapsed_seconds']-guard)
    check(q==job['steady_wall_seconds_per_update'] and guard==job['sum_all512_guarded_seconds'] and residual==job['nonnegative_external_minus_all_guarded_seconds'],'Independent q/r recomputation')
    check(q>0 and q*448+1e-6>=sum(x['guarded_update_seconds'] for x in rows[64:]) and history['elapsed_seconds']>=previous,'Steady wall covers guarded work')
    computed.append({'id':name,'wave':job['wave'],'q_seconds_per_update':q,'all512_guarded_seconds':guard,'r_external_minus_all_guarded_seconds':residual,
        'external_seconds':ex['elapsed_seconds'],'lr_cross_platform_max_float64_ulp':max(lr_ulps),'lr_cross_platform_nonexact_rows':sum(x>0 for x in lr_ulps),'expanded_examples':expanded_count,'checkpoint_sha256':job['final_pointer']['sha256']})
by={(j['arm'],j['seed']):j for j in s['jobs']}
for seed in range(3):
    a,b=by['base',seed],by['mix',seed];check(a['initial']==b['initial'],'Initial model/RNG/Adam scalar digest pairing')
    for x,y,g,h in zip(a['pairing_rows'],b['pairing_rows'],a['graph_rows'],b['graph_rows']):
        check(all(x[k]==y[k] for k in ('completed_steps','frame_ids','particles','lr')),'Paired scalar schedule')
        check(all(g[k]==h[k] for k in ('completed_steps','absolute_schedule_step','frame_ids','noise_sha256')),'Paired frame/noise schedule')
        for p,q in zip(g['examples'],h['examples']):
            check(all(p[k]==q[k] for k in ('example_slot','n_particles','exposure_coin','coin_seed_material','pair_seed_material','native_directed_edges','native_self_edges',
                'receivers_above_native_cap','annulus_pairs','optional_budget_if_exposed','native_edge_sha256','noisy_current_sha256')),'Paired native/noisy/coin/annulus invariants')
    pair=next(x for x in s['pairing'] if x['seed']==seed)
    check(all(pair[k] is True for k in ('initial_model_rng_Adam_exact','all512_sample_noise_lr_rows_exact','all_native_graph_noisy_state_coin_material_annulus_budget_exact')),'Summary pairing flags')
q={w:max(j['q_seconds_per_update'] for j in computed if j['wave']==w) for w in ('A','B')}
rmax={w:max(j['r_external_minus_all_guarded_seconds'] for j in computed if j['wave']==w) for w in ('A','B')}
check(q==s['q4_q2'] and rmax==s['r4_r2'],'Wave maxima')
waves={w:saved('wave_'+w+'.json') for w in ('A','B')}
for w,receipt in waves.items():
    assigned=[j for j in s['jobs'] if j['wave']==w];pids={j['external']['pid']:j for j in assigned}
    check(receipt['state']=='verified' and receipt['full_steady_overlap_observed'] is True and len(receipt['jobs'])==len(assigned),'Verified wave receipt')
    check(all(x==saved('logs/'+x['id']+'.outcome.json') for x in receipt['jobs']),'Wave/outcome correspondence')
    observed_overlap=False
    for ob in receipt['observations']:
        check(all(p['pid'] in pids and p['gpu_uuid']==r['gpu_uuids'][pids[p['pid']]['gpu']] for p in ob['gpu_processes']),'Observed owned device workload')
        states=ob['trainer_states'];present={p['pid'] for p in ob['gpu_processes']}
        observed_overlap |= present==set(pids) and all(str(pid) in states and states[str(pid)]['state']=='running' and 64<=states[str(pid)]['completed_steps']<512 for pid in pids)
    check(observed_overlap,'Actual steady overlap observation')
    check(stamp(receipt['started_utc'])<=min(stamp(j['external']['started_utc']) for j in assigned)
        and stamp(receipt['ended_utc'])>=max(stamp(j['external']['ended_utc']) for j in assigned),'Wave/child chronology')
check(stamp(waves['A']['ended_utc'])<stamp(waves['B']['started_utc']),'Sequential four then two waves')
stop=read('goop3d_capacity_stopped_receipt_v1.json');process=read('third_vm_capacity_stopped_valid_bytes_v1.json')
check(stop['status']=='supervisor_and_all_six_workers_stopped' and stop['issued_by']=='root' and stop['capacity_summary_sha256']==inputs[summary_name]
    and stop['artifact_inventory_sha256']==inputs['goop3d_capacity_final_artifact_inventory_v1.json'] and stop['process_inventory_sha256']==inputs['third_vm_capacity_stopped_valid_bytes_v1.json'],'Root stopped receipt bindings')
check(stop['all_owned_processes_reaped_or_independently_verified_absent'] is True and stop['all_twelve_checkpoint_hashes_rechecked'] is True
    and stop['all_six_external_exit_codes']==[0]*6 and stop['supervisor_unified_exit_code']==0,'Root stopped/checkpoint receipt')
check(process['matching_processes']==[] and process['gpu_apps']=='' and process['hostname']==r['hostname']
    and stamp(process['utc'])>stamp(waves['B']['ended_utc']),'Root postrun idle process evidence')
for name,value in inputs.items():check(sha(ROOT/name)==value,'Terminal input hash '+name)
for name,value in core.items():check(sha(name)==value,'Terminal core source '+name)
output={'schema':'adaptgns_goop3d_capacity_independent_scalar_audit_v1','reviewer':'conference_critique_methods',
 'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed_actual_capacity_scalar_and_snapshot_audit','checks_passed':checks,
 'source_sha256':sha(__file__),'inputs_sha256':inputs,'core_source_sha256':core,'jobs':computed,'q4_q2':q,'r4_r2':rmax,
 'scope':'Independent local recomputation from all3072 scalar updates,3072 graph rows/6144 examples,74 actual scalar artifacts and pinned source/data manifests. No official numeric arrays, checkpoint deserialization, remote commands or GPU actions.',
 'remote_bytes_scope':'12 actual checkpoint hashes consumed from the root remote final-artifact inventory and stopped receipt; this local audit did not read their bytes or independently reproduce tensor digests.',
 'scientific_training_admitted':False,'scientific_endpoint_selected':False,'test_accessed':False,'remaining_blocking_findings':[],
 'resolved_audit_portability_finding':{'initial_log':'goop3d_capacity_methods_scalar_audit_attempt1.log','reason':'Strict cross-platform exponentiation equality failed at2/512 steps,1float64 ULP in all six jobs. Paired-arm LR equality remains exact; formula reproduction is explicitly bounded by1ULP. Frozen training code and artifacts were unchanged.'},
 'limits':['Observed q/r include CPU audit contention and process-exit observer lag; they are engineering measurements, not ideal throughput.',
           'Steady simultaneous GPU occupancy was observed; continuous occupancy between observations was not measured.',
           'Fresh scientific models require a separately selected endpoint and root release. Capacity512 checkpoints are never science models.',
           'Complete full-horizon validation timing is still required for the preferred full-study forecast; q/r alone select no endpoint.']}
path=ROOT/'goop3d_vectorized_capacity_v1_independent_review_methods.json'
with path.open('x') as stream:json.dump(output,stream,indent=2);stream.write('\n')
print(json.dumps({'checks_passed':checks,'output':str(path),'sha256':sha(path),'q4_q2':q,'r4_r2':rmax},indent=2))
