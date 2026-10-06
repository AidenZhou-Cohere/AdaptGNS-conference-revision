"""Synthetic scalar cohort/split contracts and opaque checkpoints; no inference."""
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sand_final_ops',HERE/'supervise_sand_final_evaluation_scoped_v1.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
NOW=datetime(2026,10,6,22,tzinfo=timezone.utc)


def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n');return p
def h(v):return hashlib.sha256(str(v).encode()).hexdigest()


def split_fixture(split,root,cohort_sha):
    B=M.modules()[3];records=[];details=[]
    for i in range(30):
        p={'path':f'{split}/p{i}.npy','shape':[320,1,2],'dtype':'<f4','size_bytes':128,'sha256':h((split,i,'p'))}
        t={'path':f'{split}/t{i}.npy','shape':[],'dtype':'<i8','size_bytes':80,'sha256':h((split,i,'t'))}
        pv,tv=h((split,i,'pv')),h((split,i,'tv'));member=f'simulation_trajectory_{i}.npy'
        records.append({'id':f'{split}:{i:06d}','source_index':i,'source_member':member,'positions':p,'particle_types':t,
            'trajectory_content_sha256':h(p['sha256']+':'+t['sha256']),'logical_content_sha256':h(pv+':'+tv)})
        details.append({'source_index':i,'source_member':member,'frames':320,'particles':1,'position_dtype':'<f4','particle_type_dtype':'<i8',
            'particle_type_shape':[],'numeric_dtype_shape_values_verified_exact':True,'source_position_value_sha256':pv,'source_type_value_sha256':tv})
    source_sha=B.SOURCES['valid'][2] if split=='valid' else h('synthetic-test')
    converter='37c135a9eaa484a5fdbed2176780913b71cd3be4df602983fcd3b644c1a57ff9'
    manifest={'format':'gns-trajectory-manifest','version':1,'dataset':'Sand','split':split,'record_count':30,'records':records,
        'metadata':{'dim':2,'bounds':[[.1,.9],[.1,.9]]},'metadata_sha256':B.METADATA_SHA,'converter_sha256':converter,
        'source':{'family':'designsafe_published_npz','dataset':'Sand','file':split+'.npz','size_bytes':82712898 if split=='valid' else 85825802,
            'sha256':source_sha,'ZIP_CRC_verified':True,'member_count':30,'acquisition_report_sha256':h('synthetic-acq')}}
    mp=put(root/(split+'.json'),manifest)
    structural={'schema':'designsafe_sand_numeric_repackage_v1','status':'complete_structural_only','dataset':'Sand','source_family':'designsafe_published_npz',
        'metadata_sha256':B.METADATA_SHA,'converter_sha256':converter,'acquisition_report_sha256':h('synthetic-acq'),
        'splits':{split:{'manifest_sha256':M.sha(mp),'record_count':30,'frame_lengths':[320],'position_dtypes':['<f4'],'particle_type_ids':[6],
            'kinematic_type3_particles':0,'ZIP_CRC_verified':True,'all_numeric_values_preserved_exact':True,'records':details}}}
    sp=put(root/(split+'_structural.json'),structural)
    admission={'schema':'adaptgns_sand_graph_support_final_evaluation_admission_v1','status':'admitted_for_final_evaluation','dataset':'Sand','split':split,
        'cohort_manifest_sha256':cohort_sha,'training_admission_sha256':M.TRAIN_ADMISSION_SHA,'protocol_sha256':M.PROTOCOL_SHA,
        'manifest_sha256':M.sha(mp),'structural_report_sha256':M.sha(sp),'frames_per_trajectory':320,'record_count':30,'particle_type_ids':[6],
        'position_dtype':'<f4','source_sha256':source_sha,'metadata_sha256':B.METADATA_SHA,'converter_sha256':converter}
    ap=put(root/(split+'_admission.json'),admission)
    pf=put(root/(split+'_preflight.json'),{'schema':'adaptgns_sand_final_split_preflight_v1','status':'complete_split_contract_passed','split':split,
        'cohort_sha256':cohort_sha,'manifest_sha256':M.sha(mp),'admission_sha256':M.sha(ap),'structural_report_sha256':M.sha(sp),
        'all_split_numeric_duplicates_checked':True,'evaluator_sha256':M.EVALUATOR_SHA,'evaluation_executed':False})
    return mp,sp,ap,pf


def fixture(tmp_path,monkeypatch,role='B'):
    G,S,E,B=M.modules();monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    a=SimpleNamespace(output_dir=tmp_path/'queue',lifecycle_source=HERE/'measure_sand_cuda_capacity_v2.py')
    cohort_path=tmp_path/'cohort.json';audit_path=tmp_path/'audit.json';models=[]
    for arm in ('base','mix'):
        for seed in range(3):
            cp=tmp_path/f'{arm}{seed}.pt';cp.write_bytes(f'opaque-synthetic-{arm}{seed}'.encode())
            models.append({'objective':'faithful','arm':arm,'seed':seed,'completed_steps':100000,'checkpoint_sha256':M.sha(cp),'checkpoint_path':str(cp)})
    common={'training_schema':'adaptgns_sand_graph_support_cuda_training_v1','protocol_sha256':M.PROTOCOL_SHA,
        'training_admission_sha256':M.TRAIN_ADMISSION_SHA,'trainer_source_sha256':M.TRAINER_SHA}
    audit={**common,'schema':'adaptgns_sand_graph_support_complete_cohort_audit_v1','status':'all_six_endpoints_and_pairing_verified',
        'models':[{**m,'graph_history_updates':100000,'checkpoint_every':10000,'log_every':100,**{k:True for k in
            ('all_optimizer_steps_equal_100000','all_state_and_moments_finite','source_data_protocol_verified','checkpoint_bytes_verified')}} for m in models],
        'paired_seeds':[{'seed':s,**{k:True for k in ('initial_model_tensor_identity','initial_cpu_cuda_rng_identity',
            'all_frame_noise_lr_schedules_equal','all_graph_budgets_and_rng_material_verified')}} for s in range(3)]}
    put(audit_path,audit);cohort={**common,'schema':'adaptgns_sand_graph_support_final_cohort_v1','status':'frozen_for_final_evaluation',
        'dataset':'Sand','updates':100000,'models':models,'cohort_audit_sha256':M.sha(audit_path),'benchmark_helper_sha256':M.BENCH_SHA,
        'diagnostic_source_sha256':M.EVALUATOR_SHA,'deterministic_algorithms':True,'cublas_workspace_config':':4096:8','policies':list(G.POLICIES)}
    put(cohort_path,cohort);splits={s:split_fixture(s,tmp_path/'data',M.sha(cohort_path)) for s in ('valid','test')}
    r={'schema':M.RELEASE_SCHEMA,'status':'admitted_for_execution_allocation','issued_by':'root','dataset':'Sand','host_role':role,
        'host':M.socket.gethostname(),'environment':G.ENVIRONMENT,'cost_basis':G.COST_BASIS,'all_six100k_models_frozen_before_test':True,
        'compute_analysis_deadline_utc':M.DEADLINE.isoformat(),'outer_processing_reserve_seconds':3600,
        'stage_quotas_seconds':{n:q for n,_,_,q in G.STAGES},'cleanup_seconds_per_invocation':15,
        'preparation_and_cohort_reserves_already_accounted_before_this_queue':True,'clock_error_bound_seconds':1,
        'latest_start_utc':(NOW+timedelta(hours=1)).isoformat(),'process_clock_checked_utc':NOW.isoformat(),
        'python_environment':S.python_environment(sys.executable),'gpu_uuids':[f'00000000-0000-0000-0000-{i:012d}' for i in range(4)],
        'gpu_scope':{'owned_indices':sorted({g for _,_,g in M.SCHEDULE[role]}),'unassigned_devices':'observe_without_control',
            'timing_scope':'shared_host_operational_measurement','live_training_handoff':False},'streams':[],
        'operational_amendment':{'path':str(HERE/'sand_scoped_operational_amendment_v1.md'),'sha256':M.AMENDMENT_SHA},
        'split_preflight':{s:{'path':str(v[3]),'sha256':M.sha(v[3])} for s,v in splits.items()}}
    source_names=('supervise_sand_final_evaluation_scoped_v1.py','supervise_goop_evaluation_gpu_scoped_v3.py','supervise_sand_scoped_science_v1.py',
        'evaluate_sand_graph_support_final.py','benchmark_sand_graph_support_rollout.py','train_sand_graph_support_cuda.py',
        'sand_graph_support_100k_protocol_v1.md','sand_scoped_schedule_fixed_spec_v2.json','sand_scoped_operational_amendment_v1.md',
        'measure_sand_cuda_capacity_v2.py','sand_train_admission.json')
    files=[HERE/n for n in source_names]+[cohort_path,audit_path]+[Path(m['checkpoint_path']) for m in models]+[p for v in splits.values() for p in v]
    repo=HERE.parents[2]/'outputs/AdaptGNS';files += [repo/n for n in B.SOURCE_PINS]
    env=r['python_environment'];files += [Path(env['lexical_path']),Path(env['resolved_binary_path'])]
    if env['pyvenv_config_path']:files.append(Path(env['pyvenv_config_path']))
    r['files_sha256']={str(p):M.sha(p) for p in files}
    for arm,seed,gpu in M.SCHEDULE[role]:
        model=next(m for m in models if (m['arm'],m['seed'])==(arm,seed));sid=f'{arm}_seed{seed}';commands=[]
        for n,mode,split,quota in G.STAGES:
            mp,sp,ap,_=splits[split]
            options={'--repo':str(repo),'--benchmark-helper':str(HERE/'benchmark_sand_graph_support_rollout.py'),'--benchmark-sha256':M.BENCH_SHA,
                '--cohort':str(cohort_path),'--manifest':str(mp),'--admission':str(ap),'--structural-report':str(sp),
                '--train-admission':str(HERE/'sand_train_admission.json'),'--trainer-source':str(HERE/'train_sand_graph_support_cuda.py'),
                '--protocol':str(HERE/'sand_graph_support_100k_protocol_v1.md'),'--checkpoint':model['checkpoint_path'],'--checkpoint-sha256':model['checkpoint_sha256'],
                '--cohort-audit':str(audit_path),'--output-dir':str(a.output_dir/'jobs'/sid/n),'--mode':mode,'--split':split,
                '--objective':'faithful','--arm':arm,'--seed':str(seed),'--cuda-index':str(gpu),'--threads':'2','--max-seconds':str(quota)}
            commands.append([sys.executable,str(HERE/'evaluate_sand_graph_support_final.py'),'--execute',*[x for kv in options.items() for x in kv]])
        r['streams'].append({'id':sid,'arm':arm,'seed':seed,'gpu':gpu,'commands':commands})
    return a,r,(G,S,E,B)


def test_default_no_runtime_or_source_loading(monkeypatch,capsys):
    monkeypatch.setattr(M,'modules',lambda:pytest.fail('default must not load'))
    assert M.main([])==0 and json.loads(capsys.readouterr().out)['horizon']==314


@pytest.mark.parametrize('role',['A','B'])
def test_frozen_sand_cohort_split_and_scope_contracts(tmp_path,monkeypatch,role):
    a,r,mods=fixture(tmp_path,monkeypatch,role)
    M.validate_release(a,r,mods,NOW)
    assert mods[0].DEADLINE==M.DEADLINE and len(mods[0].expected_cells('Sand','full_rollout_test'))==180
    assert mods[0].expected_cells('Sand','same_state_test')[:5]==[(0,t) for t in (7,85,163,241,319)]


@pytest.mark.parametrize('bad',['mapping','deadline','reserve','cohort','preflight','python'])
def test_admission_refuses_incomplete_or_changed_study(tmp_path,monkeypatch,bad):
    a,r,mods=fixture(tmp_path,monkeypatch)
    if bad=='mapping':r['streams'][0]['gpu']=0
    elif bad=='deadline':r['compute_analysis_deadline_utc']='2026-10-07T08:00:00+00:00'
    elif bad=='reserve':r['outer_processing_reserve_seconds']=6300
    elif bad=='cohort':r['all_six100k_models_frozen_before_test']=False
    elif bad=='preflight':r['split_preflight'].pop('test')
    else:r['streams'][0]['commands'][0][0]=str(Path(sys.executable).resolve())
    with pytest.raises(ValueError):M.validate_release(a,r,mods,NOW)
