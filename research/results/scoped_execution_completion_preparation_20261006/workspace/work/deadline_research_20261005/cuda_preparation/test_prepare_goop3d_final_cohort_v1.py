"""Synthetic stopped receipts/checkpoint bytes only; no model, official arrays or CUDA."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import prepare_goop3d_final_cohort_v1 as M
import prepare_goop3d_science_contracts_v1 as G
from test_prepare_goop3d_science_contracts_v1 import fixture as generator_fixture


def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))


def cohort_fixture(tmp_path):
    S,a,spec,root_spec,n=generator_fixture(tmp_path)
    G.prepare(root_spec,Path(spec['contracts_dir']),S,lambda:n)
    root=a.output_dir;root.mkdir();(root/'inputs').mkdir();(root/'jobs').mkdir();(root/'logs').mkdir()
    paths=S.input_paths(a);hashes={k:M.sha(p) for k,p in paths.items()};snapshots={}
    for i,(k,p) in enumerate(paths.items()):
        if k=='python':continue
        relative='inputs/'+str(i)+p.suffix;(root/relative).write_bytes(p.read_bytes());snapshots[k]=relative
    (root/'inputs/release.json').write_bytes(a.release.read_bytes())
    launch={'schema':S.SCHEMA,'schedule':S.SCHEDULE,'endpoint_updates':37,'environment':S.ENVIRONMENT,'files_sha256':hashes,
        'input_snapshots':snapshots,'release_sha256':M.sha(a.release),'hostname':spec['hostname'],'cohort_id':spec['cohort_id'],
        'gpu_uuids':spec['gpu_uuids'],'pid':100,'commands':{j['id']:S.fixed_command(a,j) for j in S.SCHEDULE}}
    write(root/'launch.json',launch);jobs=[]
    for i,j in enumerate(S.SCHEDULE):
        d=root/'jobs'/j['id'];d.mkdir();pid=200+i
        config={'schema':S.T.SCHEMA,'dataset':'Goop-3D',**{k:j[k] for k in ('objective','arm','seed')},
            'updates':37,'prospective_endpoint_updates':37,'research_protocol_sha256':hashes['protocol'],
            'source_sha256':{**S.T.SOURCE_PINS,'train_goop3d_graph_support_cuda_v2.py':M.TRAINER_SHA,'goop3d_graph_support_vectorized_v1.py':M.GRAPH_SHA},
            'data':{'manifest_sha256':hashes['train_manifest'],'admission_sha256':hashes['admission']}}
        config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for step in (0,10,20,30,37):(d/f'checkpoint-{step:09d}.pt').write_bytes(('synthetic-only-no-payload-'+str(i)+'-'+str(step)).encode())
        checkpoints={p.name:M.sha(p) for p in d.glob('checkpoint-*.pt')}
        pointers={step:{'path':f'checkpoint-{step:09d}.pt','sha256':checkpoints[f'checkpoint-{step:09d}.pt'],
            'completed_steps':step,'run_config_sha256':config_sha} for step in (0,37)}
        external={'pid':pid,'identity':{'pid':pid,'ppid':100,'start_ticks':1000+i,'argv':launch['commands'][j['id']]},
            'command':launch['commands'][j['id']],'exit_code':0,'signals':[],'elapsed_seconds':1000.}
        status={'state':'complete','process':{'pid':pid},'completed_steps':37,'committed_steps':37,'requested_steps':37,'run_config_sha256':config_sha,'latest_checkpoint':pointers[37],'error':None}
        for name,obj in [('protocol.json',config),('status.json',status),('history.json',{'synthetic_history':i}),('latest.json',pointers[37])]:write(d/name,obj)
        flags={'all_optimizer_steps_equal_endpoint':True,'all_state_and_moments_finite':True,'cpu_cuda_rng_serialized':True,'simulator_config_verified':True}
        jobs.append({**j,'status':'verified_scientific_endpoint','directory':str(d),'config_sha256':config_sha,'history':{'graph_updates':37},
            'initial_checkpoint':{**flags,'completed_steps':0},'endpoint':{**flags,'completed_steps':37},'checkpoint_sha256':checkpoints,
            'initial_pointer':pointers[0],'final_pointer':pointers[37],'external':external,
            'artifact_sha256':{name:M.sha(d/name) for name in ('protocol.json','status.json','history.json','latest.json')}})
        outcome={**j,'external':external,'verification':'passed'}
        write(root/'logs'/(j['id']+'.outcome.json'),outcome)
        write(root/'logs'/(j['id']+'.launch.json'),{'job':j,'pid':pid,'identity':external['identity'],'command':external['command']})
    pairs=[{'seed':s,**{k:True for k in M.PAIR_FLAGS}} for s in range(3)]
    for wave,begin,end in [('A',1,20),('B',21,40)]:
        write(root/f'wave_{wave}.json',{'state':'verified','jobs':[{**{k:j[k] for k in S.SCHEDULE[0]},'external':j['external'],'verification':'passed'} for j in jobs if j['wave']==wave],
            'started_utc':(n+timedelta(minutes=begin)).isoformat(),'all_children_reaped_utc':(n+timedelta(minutes=end)).isoformat(),
            'ended_utc':(n+timedelta(minutes=end,seconds=1)).isoformat()})
    S.final_candidates(a,launch,jobs,pairs)
    summary={'schema':S.SCHEMA,'status':'all_six_scientific_endpoints_verified','endpoint_updates':37,'all_inputs_and_worker_artifacts_reverified':True,
        'whole_cohort_frozen_for_evaluation':False,'test_accessed':False,'cohort_id':spec['cohort_id'],'jobs':jobs,'paired_seeds':pairs,
        'candidate_files_sha256':{name:M.sha(root/name) for name in ('cohort_candidate.json','cohort_audit_candidate.json')}}
    write(root/'summary.json',summary);write(root/'status.json',{'state':'complete_scientific_training','summary_sha256':M.sha(root/'summary.json'),
        'whole_cohort_frozen_for_evaluation':False,'test_accessed':False})
    inventory=tmp_path/'completed_inventory.json';M.inventory(root,inventory)
    stopped=tmp_path/'stopped.json';checked=n+timedelta(minutes=41)
    stop={'schema':M.STOP_SCHEMA,'issued_by':'root','status':'all_scientific_processes_stopped','science_dir':str(root),
        'summary_sha256':M.sha(root/'summary.json'),'hostname':spec['hostname'],'matching_owned_processes':[],'gpu_processes':[],
        'all_owned_processes_reaped_or_independently_verified_absent':True,'known_owned_pids':[100,*range(200,206)],'checked_utc':checked.isoformat()}
    write(stopped,stop);out=tmp_path/'frozen_cohort';release=tmp_path/'freeze_release.json'
    r={'schema':M.RELEASE_SCHEMA,'status':'approved_for_byte_checked_cohort_freeze','issued_by':'root','source_sha256':M.sha(M.__file__),
        'science_dir':str(root),'output_dir':str(out),'inventory_sha256':M.sha(inventory),'stopped_receipt_sha256':M.sha(stopped),
        'supervisor_sha256':M.SUPERVISOR_SHA,'evaluator_sha256':M.EVALUATOR_SHA,'independent_completed_cohort_review':True,'endpoint_updates':37}
    write(release,r)
    return SimpleNamespace(science_dir=root,inventory=inventory,stopped_receipt=stopped,root_release=release,output_dir=out),checked+timedelta(seconds=1),S


def test_full_synthetic_frozen_evaluator_composition_and_opaque_byte_audit(tmp_path):
    a,n,S=cohort_fixture(tmp_path);r=M.freeze(a,now=lambda:n)
    cohort=M.read(a.output_dir/'cohort.json');audit=M.read(a.output_dir/'cohort_audit.json')
    assert cohort['status']=='frozen_for_final_evaluation' and audit['issued_by']=='root' and cohort['endpoint_updates']==37
    assert r['all_six_frozen_evaluator_scalar_gates_passed'] and not r['model_deserialization_performed']
    assert M.sha(a.output_dir/'cohort.json')==r['cohort_sha256']


@pytest.mark.parametrize('mutation',['checkpoint','source_snapshot','pairing','capacity_schema','missing_cell','pointer','wave','candidate'])
def test_incomplete_or_changed_science_refuses_freeze(tmp_path,mutation):
    a,n,S=cohort_fixture(tmp_path);root=a.science_dir
    if mutation=='checkpoint':(root/'jobs/base_seed0/checkpoint-000000037.pt').write_bytes(b'changed')
    elif mutation=='source_snapshot':
        launch=M.read(root/'launch.json');(root/launch['input_snapshots']['trainer']).write_text('changed')
    elif mutation=='pairing':
        p=root/'summary.json';s=M.read(p);s['paired_seeds'][0]['initial_cpu_cuda_rng_identity']=False;write(p,s)
    elif mutation=='capacity_schema':
        p=root/'summary.json';s=M.read(p);s['schema']='adaptgns_goop3d_vectorized_capacity_v1';write(p,s)
    elif mutation=='missing_cell':
        p=root/'summary.json';s=M.read(p);s['jobs'].pop();write(p,s)
    elif mutation=='pointer':
        p=root/'summary.json';s=M.read(p);s['jobs'][0]['final_pointer']['completed_steps']=512;write(p,s)
    elif mutation=='wave':
        p=root/'wave_B.json';s=M.read(p);s['state']='failed';write(p,s)
    else:
        p=root/'cohort_candidate.json';s=M.read(p);s['models'][0]['checkpoint_sha256']='f'*64;write(p,s)
    with pytest.raises(ValueError):M.freeze(a,now=lambda:n)
    assert not (a.output_dir/'cohort.json').exists()


def test_fresh_stopped_receipt_required_at_publication(tmp_path):
    a,n,S=cohort_fixture(tmp_path)
    with pytest.raises(ValueError,match='stale'):M.freeze(a,now=lambda:n+timedelta(minutes=6))
    assert (a.output_dir/'failed_freeze.json').exists() and not (a.output_dir/'cohort.json').exists()


def test_root_freeze_cannot_change_endpoint_or_output(tmp_path):
    a,n,S=cohort_fixture(tmp_path);r=M.read(a.root_release);r['endpoint_updates']=100000;write(a.root_release,r)
    with pytest.raises(ValueError,match='endpoint'):M.freeze(a,now=lambda:n)
    assert not (a.output_dir/'cohort.json').exists()


def test_inventory_refuses_symlink_and_preserves_bytes(tmp_path):
    root=tmp_path/'science';root.mkdir();p=root/'checkpoint.pt';p.write_bytes(b'synthetic')
    (root/'alias.pt').symlink_to(p)
    with pytest.raises(ValueError,match='symlink'):M.inventory(root,tmp_path/'inventory.json')
    assert p.read_bytes()==b'synthetic'


def test_terminal_science_mutation_prevents_active_cohort(tmp_path,monkeypatch):
    a,n,S=cohort_fixture(tmp_path);original=M.write
    def mutate(path,obj):
        original(path,obj)
        if Path(path).name=='freeze_report.json':(a.science_dir/'jobs/base_seed0/checkpoint-000000037.pt').write_bytes(b'changed')
    monkeypatch.setattr(M,'write',mutate)
    with pytest.raises(ValueError,match='changed'):M.freeze(a,now=lambda:n)
    assert not (a.output_dir/'cohort.json').exists()


def rebind_synthetic_inventory(a):
    """Rebind synthetic receipts so rejection exercises semantics, not stale SHA."""
    root=a.science_dir
    status=M.read(root/'status.json');status['summary_sha256']=M.sha(root/'summary.json');write(root/'status.json',status)
    stop=M.read(a.stopped_receipt);stop['summary_sha256']=M.sha(root/'summary.json');write(a.stopped_receipt,stop)
    inv=M.read(a.inventory);inv['files']=M.tree_inventory(root);write(a.inventory,inv)
    release=M.read(a.root_release);release['inventory_sha256']=M.sha(a.inventory);release['stopped_receipt_sha256']=M.sha(a.stopped_receipt);write(a.root_release,release)


@pytest.mark.parametrize('mutation,message',[
    ('pairing','initial/RNG/frame/graph'),('schema','endpoint summary'),('missing','six-cell'),
    ('pointer','pointer/configuration'),('wave','wave receipt'),('stopped','stopped-process')])
def test_rebound_synthetic_receipts_still_require_complete_semantics(tmp_path,mutation,message):
    a,n,S=cohort_fixture(tmp_path);root=a.science_dir;s=M.read(root/'summary.json')
    if mutation=='pairing':s['paired_seeds'][0]['initial_empty_adam_identity']=False
    elif mutation=='schema':s['schema']='adaptgns_goop3d_vectorized_capacity_v1'
    elif mutation=='missing':s['jobs'].pop()
    elif mutation=='pointer':s['jobs'][0]['final_pointer']['completed_steps']=512
    elif mutation=='wave':
        p=root/'wave_B.json';w=M.read(p);w['state']='failed';write(p,w)
    else:
        stop=M.read(a.stopped_receipt);stop['known_owned_pids'].pop();write(a.stopped_receipt,stop)
    write(root/'summary.json',s);rebind_synthetic_inventory(a)
    with pytest.raises(ValueError,match=message):M.freeze(a,now=lambda:n)
    assert not (a.output_dir/'cohort.json').exists()


def test_rebound_candidate_authority_refused(tmp_path):
    a,n,S=cohort_fixture(tmp_path);root=a.science_dir
    p=root/'cohort_candidate.json';c=M.read(p);c['issued_by']='root';write(p,c)
    p=root/'summary.json';s=M.read(p);s['candidate_files_sha256']['cohort_candidate.json']=M.sha(root/'cohort_candidate.json');write(p,s)
    rebind_synthetic_inventory(a)
    with pytest.raises(ValueError,match='unadmitted candidates'):M.freeze(a,now=lambda:n)
    assert not (a.output_dir/'cohort.json').exists()


@pytest.mark.parametrize('kind',['original_graph','structural_report','acquisition_report','context_semantics','auxiliary_report','cohort_id'])
def test_independently_reproduced_rebound_lineage_gaps_refused(tmp_path,kind):
    a,n,S=cohort_fixture(tmp_path);root=a.science_dir
    if kind=='cohort_id':
        audit=M.read(root/'cohort_audit_candidate.json');candidate=M.read(root/'cohort_candidate.json')
        audit['cohort_id']=candidate['cohort_id']='different_synthetic_cohort';write(root/'cohort_audit_candidate.json',audit)
        candidate['cohort_audit_sha256']=M.sha(root/'cohort_audit_candidate.json');write(root/'cohort_candidate.json',candidate)
        summary=M.read(root/'summary.json');summary['candidate_files_sha256']={n:M.sha(root/n) for n in ('cohort_candidate.json','cohort_audit_candidate.json')};write(root/'summary.json',summary)
    else:
        launch=M.read(root/'launch.json');snapshot=root/launch['input_snapshots'][kind];snapshot.write_text('synthetic altered '+kind)
        launch['files_sha256'][kind]=M.sha(snapshot);r=M.read(root/'inputs/release.json');r['files_sha256']=launch['files_sha256'];write(root/'inputs/release.json',r)
        launch['release_sha256']=M.sha(root/'inputs/release.json');write(root/'launch.json',launch)
    rebind_synthetic_inventory(a)
    with pytest.raises(ValueError,match='cohort correspondence' if kind=='cohort_id' else 'data-support/original-graph'):M.freeze(a,now=lambda:n)
    assert not (a.output_dir/'cohort.json').exists()


@pytest.mark.parametrize('where',['job','pair','candidate_model','audit_model','audit_pair','worker_config'])
def test_bool_seed_never_aliases_integer_zero(tmp_path,where):
    a,n,S=cohort_fixture(tmp_path);root=a.science_dir
    summary=M.read(root/'summary.json');audit=M.read(root/'cohort_audit_candidate.json');candidate=M.read(root/'cohort_candidate.json')
    if where=='job':summary['jobs'][0]['seed']=False
    elif where=='pair':summary['paired_seeds'][0]['seed']=False
    elif where=='candidate_model':candidate['models'][0]['seed']=False
    elif where=='audit_model':audit['models'][0]['seed']=False
    elif where=='audit_pair':audit['paired_seeds'][0]['seed']=False
    else:
        job=summary['jobs'][0];p=root/'jobs'/job['id']/'protocol.json';config=M.read(p);config['seed']=False;write(p,config)
        job['config_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        job['artifact_sha256']['protocol.json']=M.sha(p)
        for key in ('initial_pointer','final_pointer'):job[key]['run_config_sha256']=job['config_sha256']
    write(root/'cohort_audit_candidate.json',audit);candidate['cohort_audit_sha256']=M.sha(root/'cohort_audit_candidate.json');write(root/'cohort_candidate.json',candidate)
    summary['candidate_files_sha256']={name:M.sha(root/name) for name in ('cohort_candidate.json','cohort_audit_candidate.json')};write(root/'summary.json',summary)
    rebind_synthetic_inventory(a)
    with pytest.raises(ValueError):M.freeze(a,now=lambda:n)
    assert not (a.output_dir/'cohort.json').exists()
