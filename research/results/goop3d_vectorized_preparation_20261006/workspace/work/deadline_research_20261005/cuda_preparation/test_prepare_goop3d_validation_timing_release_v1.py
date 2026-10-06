"""Scalar contract fixtures only; never executes a child or reads real data."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

import prepare_goop3d_validation_timing_release_v1 as G
import run_goop3d_validation_timing_v1 as S


def test_default_is_inert_and_has_no_numerical_imports():
    code="""import builtins,runpy,sys
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','numpy','scipy','gns','research'):raise AssertionError(name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-I','-c',code,G.__file__],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['processes_launched']==0


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    prep=tmp_path/'prep';prep.mkdir();monkeypatch.setattr(G,'HERE',prep)
    source=tmp_path/'source';source.mkdir();capacity=tmp_path/'capacity';capacity.mkdir()
    common={k:str(source/k) for k in S.COMMON if k!='split_admission'}
    Path(common['repo']).mkdir()
    for key,path in common.items():
        if key!='repo':Path(path).write_text(key)
    manifest={'record_count':100,'split':'valid','source':{'sha256':G.VALID_SOURCE_SHA}}
    Path(common['manifest']).write_text(json.dumps(manifest))
    E=SimpleNamespace(VALID_MANIFEST_SHA=S.sha(common['manifest']),METADATA_SHA='a'*64,
        CONTEXT_SHA=S.sha(common['context_semantics']),CONVERTER_SHA='b'*64,GRAPH_SHA='c'*64,
        NUMERICAL_SOURCE_PINS={'synthetic.py':'d'*64},merge_bindings=lambda *groups:{str(k):v for group in groups for k,v in group.items()})
    admission={'schema':'adaptgns_goop3d_evaluation_split_admission_v1','status':'not_admitted','issued_by':'root_required',
        'metadata_sha256':E.METADATA_SHA,'source_sha256':G.VALID_SOURCE_SHA,'converter_sha256':E.CONVERTER_SHA,
        'split':'valid','dataset':'Goop-3D','frames':301,'dimension':3,'record_count':100,'particle_type_ids':[7]}
    for field in ('manifest','structural_report','acquisition_report','context_semantics','auxiliary_report'):
        admission[field+'_sha256']=S.sha(common[field])
    template=prep/'goop3d_valid_timing_split_admission.template.json';template.write_text(json.dumps(admission))
    for name in ('evaluate_goop3d_graph_support_v1.py','goop3d_native_evaluation_v1.py','goop3d_diagnostic_metrics_v1.py',
        'goop3d_graph_support_vectorized_v1.py','goop3d_deadline_worksheet_v1.py','measure_sand_cuda_capacity_v2.py','audit_goop3d_auxiliary.py'):
        (prep/name).write_text(name)
    monkeypatch.setattr(G,'PINNED_SIBLINGS',{name:S.sha(prep/name) for name in G.PINNED_SIBLINGS})
    pointers={}
    for arm in ('base','mix'):
        path=capacity/'jobs'/f'{arm}_seed0'/'checkpoint-000000512.pt';path.parent.mkdir(parents=True);path.write_text(arm)
        pointers[arm,0]=S.sha(path)
    (capacity/'summary.json').write_text('{}')
    python=tmp_path/'python';python.write_text('python bytes');stopped=tmp_path/'stopped';stopped.write_text('{}')
    root={'schema':'adaptgns_goop3d_timing_root_preparation_spec_v1',
        'status':'approved_for_contract_generation_and_six_bounded_validation_calls','issued_by':'root',
        'scientific_training_admitted':False,'scientific_endpoint_selected':False,'test_access_allowed':False,
        'root_approved_valid_source_admission':True,'reviewed_generator_sha256':S.sha(G.__file__),
        'reviewed_supervisor_sha256':G.SUPERVISOR_SHA,'reviewed_valid_admission_template_sha256':S.sha(template),
        'common':common,'capacity_root':str(capacity),'output_dir':str(tmp_path/'output'),'capacity_stop_receipt':str(stopped),
        'python':str(python),'hostname':'synthetic-host','process_identity_checked_utc':S.now().isoformat(),
        'gpu_uuids':['GPU-first','GPU-second'],'inner_seconds':{m:100 for m in S.MODES},'outer_seconds':{m:200 for m in S.MODES}}
    root_path=tmp_path/'spec.json';root_path.write_text(json.dumps(root))
    helper=SimpleNamespace(**vars(S))
    helper.modules=lambda:(E,SimpleNamespace(capacity_inputs=lambda r:({}, {}, pointers)),None,None)
    def preflight(args,parent,*modules):
        assert not (tmp_path/'contracts'/'supervisor_release.json').exists()
        assert args.release==root_path
        specs=[(job,S.child_arguments(parent,args.output_dir,job),['synthetic',job['id']]) for job in S.SCHEDULE]
        return specs,copy.deepcopy(parent['files_sha256'])
    helper.preflight=preflight
    return root,root_path,tmp_path/'contracts',helper,template,capacity


def test_one_root_spec_generates_all_absolute_byte_bindings(prepared):
    root,path,output,H,template,capacity=prepared
    report=G.prepare(path,output,H)
    assert report['status']=='six_exact_contracts_prepared_no_process_launched'
    parent=S.read(output/'supervisor_release.json')
    assert len(parent['evaluator_releases'])==6 and parent['evaluation_overlap_credit'] is False
    assert all(Path(k).is_absolute() and S.sha(k)==v for k,v in parent['files_sha256'].items())
    for job in S.SCHEDULE:
        release=S.read(parent['evaluator_releases'][job['id']])
        assert release['purpose']=='capacity_timing' and release['split']=='valid' and release['checkpoint_updates']==512
        assert release['seed']==0 and release['cuda_index']==job['gpu']
        assert release['checkpoint_sha256']==parent['checkpoints'][job['arm']]['sha256']
    admission=S.read(parent['common']['split_admission'])
    assert admission['source_sha256']==G.VALID_SOURCE_SHA and admission['scope']=='bounded_capacity_validation_only'
    assert not Path(root['output_dir']).exists()
    with pytest.raises(FileExistsError):G.prepare(path,output,H)


@pytest.mark.parametrize('failure',['scope','template_source','checkpoint','preflight','changed_sibling'])
def test_failed_preparation_preserves_files_without_publishing_parent_release(prepared,failure):
    root,path,output,H,template,capacity=prepared
    if failure=='scope':root['test_access_allowed']=True
    elif failure=='template_source':
        value=S.read(template);value.pop('source_sha256');template.write_text(json.dumps(value))
        root['reviewed_valid_admission_template_sha256']=S.sha(template)
    elif failure=='checkpoint':(capacity/'jobs'/'base_seed0'/'checkpoint-000000512.pt').write_text('changed')
    elif failure=='changed_sibling':(template.parent/'goop3d_native_evaluation_v1.py').write_text('changed source')
    else:H.preflight=lambda *args:(_ for _ in ()).throw(ValueError('synthetic preflight failure'))
    path.write_text(json.dumps(root))
    with pytest.raises(ValueError):G.prepare(path,output,H)
    assert not (output/'supervisor_release.json').exists()
    if output.exists():assert S.read(output/'failed_preparation.json')['processes_launched']==0


def test_real_scalar_preflight_composes_with_pinned_local_source_evidence(tmp_path,monkeypatch):
    # Actual scalar manifests/source files only. Capacity summary, stop receipt
    # and checkpoint byte files below are explicitly synthetic; no torch load,
    # numerical arrays, process check or child execution occurs.
    E,W,B,T=S.modules();monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    prep=Path(S.__file__).parent;repo=prep.parents[2]/'outputs'/'AdaptGNS'
    common={'repo':str(repo),'manifest':str(prep/'goop3d_structural_metadata_v1'/'valid.json'),
        'structural_report':str(prep/'goop3d_structural_metadata_v1'/'structural_report.json'),
        'acquisition_report':str(prep/'goop3d_acquisition_v1_report.json'),
        'context_semantics':str(prep/'goop3d_context_semantics_review.json'),
        'auxiliary_report':str(prep/'goop3d_auxiliary_v1_report.json'),
        'trainer_source':str(prep/'train_goop3d_graph_support_cuda_v2.py'),
        'protocol':str(prep/'goop3d_validation_timing_protocol_v1.md')}
    cap=tmp_path/'synthetic_capacity';cap.mkdir();jobs=[]
    for arm in ('base','mix'):
        for seed in range(3):
            cp=cap/'jobs'/f'{arm}_seed{seed}'/'checkpoint-000000512.pt';cp.parent.mkdir(parents=True)
            cp.write_bytes(f'synthetic bytes never deserialized: {arm} {seed}'.encode())
            jobs.append({'arm':arm,'seed':seed,'wave':'A' if seed<2 else 'B','status':'verified_capacity_only',
                'final_pointer':{'completed_steps':512,'sha256':S.sha(cp)},'steady_wall_seconds_per_update':1.,
                'nonnegative_external_minus_all_guarded_seconds':1.})
    summary={'schema':'adaptgns_goop3d_vectorized_capacity_v1','status':'all_six_verified_capacity_only',
        'scientific_training_admitted':False,'scientific_endpoint_selected':False,'jobs':jobs,
        'q4_q2':{'A':1.,'B':1.},'r4_r2':{'A':1.,'B':1.},
        'pairing':[{'seed':s,'initial_model_rng_Adam_exact':True,'all512_sample_noise_lr_rows_exact':True} for s in range(3)]}
    S.write(cap/'summary.json',summary)
    stop=tmp_path/'synthetic_stop.json';S.write(stop,{'schema':'adaptgns_goop3d_capacity_stopped_receipt_v1',
        'issued_by':'root','status':'supervisor_and_all_six_workers_stopped','capacity_summary_sha256':S.sha(cap/'summary.json'),
        'all_owned_processes_reaped_or_independently_verified_absent':True,'synthetic_fixture':True})
    root={'schema':'adaptgns_goop3d_timing_root_preparation_spec_v1',
        'status':'approved_for_contract_generation_and_six_bounded_validation_calls','issued_by':'root',
        'scientific_training_admitted':False,'scientific_endpoint_selected':False,'test_access_allowed':False,
        'root_approved_valid_source_admission':True,'reviewed_generator_sha256':S.sha(G.__file__),
        'reviewed_supervisor_sha256':G.SUPERVISOR_SHA,
        'reviewed_valid_admission_template_sha256':S.sha(prep/'goop3d_valid_timing_split_admission.template.json'),
        'python':sys.executable,'capacity_root':str(cap),'capacity_stop_receipt':str(stop),'output_dir':str(tmp_path/'never_launched'),
        'hostname':S.socket.gethostname(),'process_identity_checked_utc':S.now().isoformat(),
        'gpu_uuids':['GPU-00000000-0000-0000-0000-000000000001','GPU-00000000-0000-0000-0000-000000000002'],
        'inner_seconds':{m:100 for m in S.MODES},'outer_seconds':{m:200 for m in S.MODES},'common':common,'synthetic_fixture':True}
    path=tmp_path/'synthetic_root_spec.json';S.write(path,root)
    output=tmp_path/'contracts';report=G.prepare(path,output,G.supervisor())
    assert len(report['commands'])==6 and report['test_accessed'] is False
    assert not Path(root['output_dir']).exists()
    assert (output/'supervisor_release.json').is_file()
