"""Synthetic scalar receipts and opaque byte placeholders; no data/model/GPU work."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent

def load(name, label):
    spec = importlib.util.spec_from_file_location(label, HERE / name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

M = load('summarize_goop_graph_support_scoped_v3.py', '_scoped_test')
F = load('test_summarize_goop_graph_support_quota_v2.py', '_scoped_old_fixture')
Q = M.import_pinned('supervise_goop_evaluation_gpu_scoped_v3.py', '_scoped_fixture_quota')
put = F.put
@pytest.fixture(autouse=True)
def synthetic_amendment_pin(monkeypatch):
    monkeypatch.setattr(M, 'AMENDMENT_SHA', hashlib.sha256(b'Synthetic scoped operational amendment\n').hexdigest())


UUIDS = ['GPU-00000000-0000-0000-0000-00000000000' + str(i) for i in range(4)]


def approval(path, **extra):
    return put(path, dict(schema=M.COLLECTION_RELEASE_SCHEMA, issued_by='root', status='approved_for_stopped_scalar_collection',
        collector_sha256=M.sha(M.__file__), source_sha256=M.SOURCE_PINS, **extra))


def refresh(root, cohort, amendment, release):
    pp = root / 'process_outcomes.json'; process = M.json_snapshot(pp)[0]
    process['gpu_observations_sha256'] = M.sha(root / 'gpu_observations.json'); put(pp, process)
    sp = root / 'queue_status.json'; status = M.json_snapshot(sp)[0]
    status['coverage_ledger_sha256'] = M.sha(root / 'coverage_ledger.json'); put(sp, status)
    approval(release, cohort_sha256=M.sha(cohort), queue_release_sha256=M.sha(root / 'release_snapshot.json'), operational_amendment_sha256=M.sha(amendment))


def make_queue(tmp_path, role='B', completed=395):
    tmp_path.mkdir(parents=True, exist_ok=True); root = tmp_path / 'queue'; root.mkdir()
    cohort = {'schema': 'adaptgns_goop_graph_support_final_cohort_v1', 'status': 'frozen_for_final_evaluation',
        'protocol_sha256': M.PROTOCOL_SHA, 'updates': 100000,
        'models': [{'arm':a,'seed':s,'checkpoint_sha256': str(s + 1 if a == 'base' else s + 4)*64} for a in ('base','mix') for s in range(3)]}
    cp = put(tmp_path/'cohort.json',cohort); amendment=tmp_path/'amendment.md'; amendment.write_text('Synthetic scoped operational amendment\n')
    pins = {'/fake/cohort':M.sha(cp),'/fake/protocol':M.PROTOCOL_SHA,'/fake/trainer':M.TRAINER_SHA,
        '/fake/benchmark':M.BENCH_SHA,'/fake/evaluator':M.EVALUATOR_SHA,'/fake/amendment':M.sha(amendment),
        '/fake/supervisor':M.SOURCE_PINS['supervise_goop_evaluation_gpu_scoped_v3.py']}
    streams, entries, proc_streams, children = [],[],[],[]
    for arm, seed, gpu in Q.SCHEDULE[role]:
        ident=f'{arm}_seed{seed}'; commands=[]; outcomes=[]
        cp_sha=next(r['checkpoint_sha256'] for r in cohort['models'] if (r['arm'],r['seed'])==(arm,seed));pins['/fake/'+ident]=cp_sha
        for stage,mode,split in M.STAGES:
            directory=root/'jobs'/ident/stage
            opts={'--mode':mode,'--split':split,'--arm':arm,'--seed':str(seed),'--checkpoint-sha256':cp_sha,
                '--cohort':'/fake/cohort','--protocol':'/fake/protocol','--trainer-source':'/fake/trainer',
                '--benchmark-helper':'/fake/benchmark','--checkpoint':'/fake/'+ident,'--output-dir':str(directory)}
            command=['/fake/python','/fake/evaluator','--execute']+[v for kv in opts.items() for v in kv];commands.append(command)
            outcome={'stage':stage,'state':'never_started'}
            if arm=='base' and stage=='full_rollout_test':
                directory.mkdir(parents=True)
                protocol={'schema':M.E_SCHEMA,'mode':mode,'split':split,'source_frame_count':401,
                    'model':{'kind':'preselected_checkpoint','completed_updates':100000,'arm':arm,'seed':seed,'checkpoint_sha256':cp_sha},
                    'input_files_sha256':{opts[k]:pins[opts[k]] for k in ('--cohort','--protocol','--trainer-source','--benchmark-helper','--checkpoint')},
                    'schedule':[{'source_index':i} for i in range(30)],'policies':list(M.POLICIES)}
                pp=put(directory/'protocol.json',protocol);artifact=directory/'trajectory_000000_base.npz';artifact.write_bytes(b'opaque synthetic bytes')
                row=F.row(arm,seed,0,'base',completed=completed);row.update(protocol_sha256=M.sha(pp),trace_file=artifact.name,trace_sha256=M.sha(artifact))
                put(artifact.with_suffix('.json'),row);put(directory/'result.json',{'schema':M.E_SCHEMA,'protocol_sha256':M.sha(pp),'rows':[row]})
                outcome={'stage':stage,'state':'quota_expired','exit_code':0,'pid':100+gpu,'output_dir':str(directory),'signals':[],
                    'quota_expired':True,'current_before_stop':{'source_index':0,'policy':'dense'}};outcomes.append(outcome)
                children.append({'identity':{'pid':100+gpu,'ppid':44,'start_ticks':1000+gpu,'argv':command,'executable':'/fake/python'},'command':command,'signals':[],'stage':stage,'quota_expired':True,'output_dir':str(directory)})
            entries.append({'stream':ident,'stage':stage,'outcome':outcome,'coverage_audit_state':'row_coverage_checked',**Q.account_stage('Goop',stage,directory,outcome)})
        streams.append({'id':ident,'arm':arm,'seed':seed,'gpu':gpu,'commands':commands});proc_streams.append({'id':ident,'outcomes':outcomes})
    scope={'owned_indices':sorted({gpu for _,_,gpu in Q.SCHEDULE[role]}),'unassigned_devices':'observe_without_control',
        'timing_scope':'shared_host_operational_measurement','live_training_handoff':False}
    put(root/'release_snapshot.json',{'schema':M.Q_RELEASE_SCHEMA,'issued_by':'root','status':'admitted_for_execution_allocation',
        'dataset':'Goop','host_role':role,'streams':streams,'files_sha256':pins,'gpu_scope':scope,'gpu_uuids':UUIDS,
        'operational_amendment':{'path':'/fake/amendment','sha256':M.sha(amendment)}})
    raw=[{'pid':100,'gpu_uuid':UUIDS[0]}]
    if role=='B':raw.append({'pid':900,'gpu_uuid':UUIDS[2],'label':'unrelated declared Sand'})
    annotated=[]
    for r in raw:
        index=UUIDS.index(r['gpu_uuid']); known=next((c['identity'] for c in children if c['identity']['pid']==r['pid']),None)
        annotated.append(dict(r,physical_index=index,scope='owned_device' if index in scope['owned_indices'] else 'observed_unassigned_device',
            identity_check='matched_owned_child' if known else 'not_owned',current_identity=known))
    put(root/'gpu_observations.json',{'schema':M.Q_SCHEMA,'observations':[{'utc':'synthetic','raw_gpu_processes':raw,
        'validation':'passed','gpu_scope':scope,'all_gpu_processes':annotated}]})
    put(root/'process_outcomes.json',{'schema':M.Q_SCHEMA,'unreaped_owned_children':[],'streams':proc_streams,'all_children':children,'gpu_scope':scope})
    put(root/'coverage_ledger.json',{'schema':M.Q_SCHEMA,'dataset':'Goop','stages':entries,'unreaped_owned_children':[],'audit_errors':[],'abort_reason':None})
    put(root/'queue_status.json',{'schema':M.Q_SCHEMA,'state':'allocation_finished','unreaped_owned_children':[],'all_pinned_inputs_reverified':True})
    release=tmp_path/'collection_release.json';refresh(root,cp,amendment,release)
    return root,cp,amendment,release


@pytest.fixture
def queue(tmp_path):return make_queue(tmp_path)


def test_scoped_collection_preserves_unassigned_activity_and_all_stage_cells(queue):
    result=M.collect(*queue)
    assert len(result['stages'])==8 and result['gpu_observation_counts']=={'passed':1,'pending':0}
    assert result['gpu_observations']['observations'][0]['all_gpu_processes'][1]['scope']=='observed_unassigned_device'
    assert result['stages'][0]['cells'][1]['state']=='timed_out_current'
    assert result['timing_scope']=='shared_host_operational_measurement'
    assert result['files']['jobs/base_seed2/full_rollout_test/trajectory_000000_base.npz']['bytes']>0


@pytest.mark.parametrize('bad',['old_schema','wrong_supervisor','wrong_amendment','unreaped','wrong_process_schema','forged_annotation','wrong_owned_device','foreign_owned','outcome_mismatch','missing_stage','bad_scalar','bad_binary'])
def test_scoped_rejects_invalid_lineage_and_evidence(queue,bad):
    root,cohort,amendment,release=queue
    path=root/'release_snapshot.json';r=M.json_snapshot(path)[0]
    if bad=='old_schema':r['schema']='adaptgns_graph_support_evaluation_quota_release_v2'
    if bad=='wrong_supervisor':r['files_sha256']['/fake/supervisor']='0'*64
    if bad=='wrong_amendment':r['operational_amendment']['sha256']='0'*64
    put(path,r)
    path=root/'process_outcomes.json';r=M.json_snapshot(path)[0]
    if bad=='unreaped':r['unreaped_owned_children']=[100]
    if bad=='wrong_process_schema':r['schema']='old'
    if bad=='outcome_mismatch':r['streams'][0]['outcomes'][0]['exit_code']=1
    put(path,r)
    path=root/'gpu_observations.json';r=M.json_snapshot(path)[0];o=r['observations'][0]
    if bad=='forged_annotation':o['all_gpu_processes'][1]['scope']='owned_device'
    if bad=='wrong_owned_device':o['raw_gpu_processes'][0]['gpu_uuid']=UUIDS[1]
    if bad=='foreign_owned':o['raw_gpu_processes'][0]['pid']=888
    put(path,r)
    if bad=='missing_stage':
        path=root/'coverage_ledger.json';r=M.json_snapshot(path)[0];r['stages'].pop();put(path,r)
    if bad=='bad_scalar':
        path=root/'jobs/base_seed2/full_rollout_test/trajectory_000000_base.json';r=M.json_snapshot(path)[0];r['mean_rollout_mse']=123;put(path,r)
    if bad=='bad_binary':(root/'jobs/base_seed2/full_rollout_test/trajectory_000000_base.npz').write_bytes(b'changed')
    refresh(*queue)
    with pytest.raises(ValueError):M.collect(*queue)


def test_rejected_raw_inventory_preserved_as_pending(queue):
    root=queue[0];path=root/'gpu_observations.json';r=M.json_snapshot(path)[0]
    r['observations'].append({'utc':'synthetic-rejected','validation':'pending','raw_gpu_processes':[{'pid':888,'gpu_uuid':UUIDS[0]}]})
    put(path,r);refresh(*queue);result=M.collect(*queue)
    assert result['gpu_observation_counts']=={'passed':1,'pending':1}
    assert result['gpu_observations']['observations'][1]['raw_gpu_processes'][0]['pid']==888


def test_guarded_prefix_partial_and_stale_aggregate_retained(tmp_path):
    queue=make_queue(tmp_path,completed=200);root=queue[0];directory=root/'jobs/base_seed2/full_rollout_test'
    path=directory/'result.json';r=M.json_snapshot(path)[0];r['rows']=[];put(path,r)
    (directory/'unfinished.npz.tmp').write_bytes(b'unfinished evidence')
    result=M.collect(*queue);stage=result['stages'][0]
    assert stage['cells'][0]['state']=='recorded_failed_outcome' and stage['rows'][0]['completed_steps']==200
    assert stage['aggregate_snapshot']['state']=='stale_valid_prefix'
    assert 'jobs/base_seed2/full_rollout_test/unfinished.npz.tmp' in result['files']
    metrics,_=M.validate_rollout(stage['rows'][0],'base',2)
    assert metrics['mse_forecast200']==1 and metrics['mean_rollout_mse'] is None


@pytest.mark.parametrize('when',['during_parse','during_aggregation','during_serialization'])
@pytest.mark.parametrize('target',['row','new_file','new_directory','amendment','root_release','own_source','frozen_source'])
def test_collection_mutation_blocks_publication(queue,tmp_path,monkeypatch,when,target):
    # Source-race tests inject hash changes; never touch frozen source bytes.
    root,cohort,amendment,release=queue;output=tmp_path/'published.json';changed=False
    original_sha=M.sha
    def mutate():
        nonlocal changed
        if changed:return
        changed=True
        if target=='row':
            p=root/'jobs/base_seed2/full_rollout_test/trajectory_000000_base.json';p.write_bytes(p.read_bytes()+b' ')
        if target=='new_file':(root/'late.txt').write_text('late')
        if target=='new_directory':(root/'late_empty').mkdir()
        if target=='amendment':amendment.write_text('mutated')
        if target=='root_release':release.write_bytes(release.read_bytes()+b' ')
        if target in ('own_source','frozen_source'):
            name=Path(M.__file__) if target=='own_source' else HERE/'summarize_goop_graph_support_quota_v2.py'
            monkeypatch.setattr(M,'sha',lambda p:'0'*64 if Path(p)==name else original_sha(p))
    if when=='during_parse':
        original=M.json_snapshot
        def intercept(path,expected=None):
            value=original(path,expected)
            if str(path).endswith('trajectory_000000_base.json'):mutate()
            return value
        monkeypatch.setattr(M,'json_snapshot',intercept)
    if when=='during_aggregation':
        original=M.validate_rollout
        def intercept(*a):
            result=original(*a);mutate();return result
        monkeypatch.setattr(M,'validate_rollout',intercept)
    if when=='during_serialization':
        original=M.encode
        def intercept(value):
            raw=original(value);mutate();return raw
        monkeypatch.setattr(M,'encode',intercept)
    with pytest.raises(ValueError):M.publish(output,M.collect(*queue))
    assert not output.exists() and changed


def make_analysis(tmp_path):
    collections={};receipts=[]
    for role in ('A','B'):
        queue=make_queue(tmp_path/role,role);c=M.collect(*queue);receipts.append(c)
        collections[role]=put(tmp_path/f'{role}.json',c)
    release=put(tmp_path/'analysis_release.json',{'schema':M.ANALYSIS_RELEASE_SCHEMA,'issued_by':'root','status':'approved_for_fixed_scalar_aggregation',
        'collector_sha256':M.sha(M.__file__),'source_sha256':M.SOURCE_PINS,'collection_sha256':{role:M.sha(p) for role,p in collections.items()},'cohort_sha256':receipts[0]['cohort_sha256']})
    return collections,release,receipts


def test_all24_and_scientific_golden_equal_frozen_v2(tmp_path):
    collections,release,receipts=make_analysis(tmp_path);result=M.summarize(collections,release)
    stages=[s for c in receipts for s in c['stages']]
    assert len(stages)==24 and result['full_rollout']['required_outcomes']==1080
    assert result['full_rollout']==F.M.full_summary([s for s in stages if s['stage']=='full_rollout_test'])
    assert result['diagnostics']=={name:F.M.diagnostic_summary(stages,name) for name,_,_ in M.STAGES[1:]}
    assert result['full_rollout']['absolute']['mean_rollout_mse']['base']['base']['mean'] is None
    M.publish(tmp_path/'summary.json',result)


@pytest.mark.parametrize('adverse',[False,True])
def test_complete_and_adverse_pure_scientific_functions_identical(adverse):
    stages=F.complete_stages()
    if adverse:
        stages[0]['rows'][0]=F.row('base',0,0,'base',completed=201)
        stages[0]['cells'][0].update(state='recorded_failed_outcome',failure={'category':'coordinate_guard'})
    assert M.full_summary(stages)==F.M.full_summary(stages)


@pytest.mark.parametrize('when',['aggregation','serialization'])
@pytest.mark.parametrize('target',['collection','release','source'])
def test_analysis_mutation_blocks_publication(tmp_path,monkeypatch,when,target):
    collections,release,_=make_analysis(tmp_path);changed=False
    def mutate():
        nonlocal changed
        if changed:return
        changed=True
        if target=='collection':collections['A'].write_bytes(collections['A'].read_bytes()+b' ')
        if target=='release':release.write_bytes(release.read_bytes()+b' ')
        if target=='source':
            original=M.sha;monkeypatch.setattr(M,'sha',lambda p:'0'*64 if Path(p)==Path(M.__file__) else original(p))
    if when=='aggregation':
        original=M.full_summary
        def intercept(*a):v=original(*a);mutate();return v
        monkeypatch.setattr(M,'full_summary',intercept)
    else:
        original=M.encode
        def intercept(v):raw=original(v);mutate();return raw
        monkeypatch.setattr(M,'encode',intercept)
    output=tmp_path/'summary.json'
    with pytest.raises(ValueError):M.publish(output,M.summarize(collections,release))
    assert changed and not output.exists()


def test_unverified_inputs_block_analysis(tmp_path):
    collections,release,_=make_analysis(tmp_path)
    r=M.json_snapshot(collections['A'])[0];r['queue_status']['all_pinned_inputs_reverified']=False;put(collections['A'],r)
    r=M.json_snapshot(release)[0];r['collection_sha256']['A']=M.sha(collections['A']);put(release,r)
    with pytest.raises(ValueError,match='Input integrity'):M.summarize(collections,release)

@pytest.mark.parametrize('case,accepted',[
    ('matched_current',True),('exited_since_inventory',True),('reused_unassigned',True),
    ('reused_assigned',False),('wrong_native_identity',False),('matched_wrong_unassigned',False),
    ('exited_with_identity',False),('forged_different_identity',False)])
def test_native_identity_scope_receipts(queue,case,accepted):
    root=queue[0];path=root/'gpu_observations.json';value=M.json_snapshot(path)[0]
    observation=value['observations'][0];raw=observation['raw_gpu_processes'][0];row=observation['all_gpu_processes'][0]
    if case=='exited_since_inventory':row.update(identity_check='exited_since_inventory',current_identity=None)
    if case in ('reused_unassigned','reused_assigned','forged_different_identity'):
        row['identity_check']='different_process_identity'
        if case!='forged_different_identity':row['current_identity']['start_ticks']+=1
        if case!='reused_assigned':
            raw['gpu_uuid']=UUIDS[3];row.update(gpu_uuid=UUIDS[3],physical_index=3,scope='observed_unassigned_device')
    if case=='wrong_native_identity':row['current_identity']['argv']=['/fake/foreign']
    if case=='matched_wrong_unassigned':
        raw['gpu_uuid']=UUIDS[3];row.update(gpu_uuid=UUIDS[3],physical_index=3,scope='observed_unassigned_device')
    if case=='exited_with_identity':row['identity_check']='exited_since_inventory'
    put(path,value);refresh(*queue)
    if accepted:
        result=M.collect(*queue);assert result['gpu_observations']==value
    else:
        with pytest.raises(ValueError):M.collect(*queue)


def test_snapshot_coverage_matches_final_supervisor(queue):
    root=queue[0];result=M.collect(*queue)
    for stage in result['stages']:
        directory=root/'jobs'/f'{stage["arm"]}_seed{stage["seed"]}'/stage['stage']
        fresh=Q.account_stage('Goop',stage['stage'],directory,stage['outcome'])
        assert fresh['cells']==stage['cells']


def test_collection_publication_exclusive_and_source_stable(queue,tmp_path):
    pins={name:M.sha(HERE/name) for name in M.SOURCE_PINS}
    result=M.collect(*queue);output=tmp_path/'published.json';M.publish(output,result);original=output.read_bytes()
    with pytest.raises(FileExistsError):M.publish(output,result)
    assert output.read_bytes()==original and pins=={name:M.sha(HERE/name) for name in M.SOURCE_PINS}
