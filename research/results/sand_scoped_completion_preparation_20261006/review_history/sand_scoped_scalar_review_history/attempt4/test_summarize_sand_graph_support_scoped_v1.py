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

M = load('summarize_sand_graph_support_scoped_v1.py', '_scoped_test')
F = load('test_summarize_goop_graph_support_quota_v2.py', '_scoped_old_fixture')
G = M.import_pinned('supervise_goop_evaluation_gpu_scoped_v3.py', '_scoped_fixture_quota')
from types import SimpleNamespace
Q = SimpleNamespace(SCHEDULE=M.SCHEDULE, account_stage=G.account_stage, flags=G.flags)
put = F.put
@pytest.fixture(autouse=True)
def synthetic_amendment_pin(monkeypatch):
    monkeypatch.setattr(M, 'AMENDMENT_SHA', hashlib.sha256(b'Synthetic scoped operational amendment\n').hexdigest())


UUIDS = ['GPU-00000000-0000-0000-0000-00000000000' + str(i) for i in range(4)]


B = F.B
def rollout_row(arm, seed, source, policy, value=1., completed=314):
    mse = [float(value)] * completed
    complete = completed == 314
    return {'arm': arm, 'training_seed': seed, 'objective': 'faithful', 'source_index': source, 'policy': policy,
            'horizon': 314, 'completed_steps': completed, 'status': 'complete' if complete else 'failed',
            'failure': None if complete else {'category': 'coordinate_guard', 'phase': 'predicted_state'},
            'mse_per_step': mse, 'mean_rollout_mse': float(value) if complete else None,
            'mse_at_final_horizon': float(value) if complete else None,
            'mse_at_declared_trace_steps': {str(s): float(value) if s <= completed else None for s in (1, 10, 50, 200, 314)},
            'synchronized_call_seconds': 2., 'trace_publication_seconds': .1,
            'predicted_boundary_per_step': [B] * completed, 'ground_truth_boundary_per_step': [B] * completed}


def complete_stages():
    result = []
    for arm in ('base', 'mix'):
        for seed in range(3):
            rows, cells = [], []
            for source, policy in M.expected('full_rollout_test'):
                value = (10 if arm == 'base' else 20) + (2 if arm == 'base' else 1) * (policy == 'laggedrisk25') + seed
                rows.append(rollout_row(arm, seed, source, policy, value))
                cells.append({'source_index': source, 'policy': policy, 'state': 'completed_required_outcome'})
            result.append({'arm': arm, 'seed': seed, 'stage': 'full_rollout_test', 'rows': rows, 'cells': cells,
                           'outcome': {'whole_invocation_seconds': 100., 'exit_code': 0}})
    return result


def as_goop_stages(stages):
    stages=copy.deepcopy(stages)
    for stage in stages:
        for r in stage['rows']:
            r['horizon']=395
            if r['status']=='complete':
                r['completed_steps']=395
                r['mse_per_step'] += [r['mse_per_step'][-1]]*81
                for side in ('predicted','ground_truth'):r[side+'_boundary_per_step'] += [r[side+'_boundary_per_step'][-1]]*81
            r['mse_at_declared_trace_steps']['395']=r['mse_at_declared_trace_steps'].pop('314')
    return stages


def as_goop_names(value):
    if isinstance(value,dict):return {k.replace('314','395'):as_goop_names(v) for k,v in value.items()}
    if isinstance(value,list):return [as_goop_names(v) for v in value]
    if isinstance(value,str) and value.startswith(('forecast200/314','full314')):return value.replace('314','395')
    return value


@pytest.fixture(autouse=True)
def pending_operational_source_fixture(tmp_path,monkeypatch):
    # Unit-only source closure until methods freezes the new operational adapter.
    # This branch is removed before final review; no execution admission is possible.
    name='supervise_sand_evaluation_gpu_scoped_v1.py'
    if M.SOURCE_PINS[name] != '0'*64:return
    import shutil
    closure=tmp_path/'synthetic_source_closure';closure.mkdir()
    pins=dict(M.SOURCE_PINS)
    for filename in pins:
        if filename != name:shutil.copy2(HERE/filename,closure/filename)
    (closure/name).write_text('# Synthetic pending operational source only\n')
    pins[name]=M.sha(closure/name)
    monkeypatch.setattr(M,'HERE',closure);monkeypatch.setattr(M,'SOURCE_PINS',pins)


def approval(path, **extra):
    return put(path, dict(schema=M.COLLECTION_RELEASE_SCHEMA, issued_by='root', status='approved_for_stopped_scalar_collection',
        collector_sha256=M.sha(M.__file__), source_sha256=M.SOURCE_PINS, **extra))


def refresh(root, cohort, amendment, release):
    pp = root / 'process_outcomes.json'; process = M.json_snapshot(pp)[0]
    process['gpu_observations_sha256'] = M.sha(root / 'gpu_observations.json'); put(pp, process)
    sp = root / 'queue_status.json'; status = M.json_snapshot(sp)[0]
    status['coverage_ledger_sha256'] = M.sha(root / 'coverage_ledger.json'); put(sp, status)
    approval(release, cohort_sha256=M.sha(cohort), queue_release_sha256=M.sha(root / 'release_snapshot.json'), operational_amendment_sha256=M.sha(amendment),
        original_queue_root=str(root),local_queue_inventory_sha256=M.inventory_sha256(M.tree_snapshot(root)))


def make_queue(tmp_path, role='B', completed=314):
    tmp_path.mkdir(parents=True, exist_ok=True); root = tmp_path / 'queue'; root.mkdir()
    cohort = {'schema': 'adaptgns_sand_graph_support_final_cohort_v1', 'status': 'frozen_for_final_evaluation',
        'protocol_sha256': M.PROTOCOL_SHA, 'updates': 100000,
        'models': [{'arm':a,'seed':s,'checkpoint_sha256': str(s + 1 if a == 'base' else s + 4)*64} for a in ('base','mix') for s in range(3)]}
    cp = put(tmp_path/'cohort.json',cohort); amendment=tmp_path/'amendment.md'; amendment.write_text('Synthetic scoped operational amendment\n')
    pins = {'/fake/cohort':M.sha(cp),'/fake/protocol':M.PROTOCOL_SHA,'/fake/trainer':M.TRAINER_SHA,
        '/fake/benchmark':M.BENCH_SHA,'/fake/evaluator':M.EVALUATOR_SHA,'/fake/amendment':M.sha(amendment),
        '/fake/supervisor':M.SOURCE_PINS['supervise_sand_evaluation_gpu_scoped_v1.py']}
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
                protocol={'schema':M.E_SCHEMA,'mode':mode,'split':split,'source_frame_count':320,
                    'model':{'kind':'preselected_checkpoint','completed_updates':100000,'arm':arm,'seed':seed,'checkpoint_sha256':cp_sha},
                    'input_files_sha256':{opts[k]:pins[opts[k]] for k in ('--cohort','--protocol','--trainer-source','--benchmark-helper','--checkpoint')},
                    'schedule':[{'source_index':i} for i in range(30)],'policies':list(M.POLICIES)}
                pp=put(directory/'protocol.json',protocol);artifact=directory/'trajectory_000000_base.npz';artifact.write_bytes(b'opaque synthetic bytes')
                row=rollout_row(arm,seed,0,'base',completed=completed);row.update(protocol_sha256=M.sha(pp),trace_file=artifact.name,trace_sha256=M.sha(artifact))
                put(artifact.with_suffix('.json'),row);put(directory/'result.json',{'schema':M.E_SCHEMA,'protocol_sha256':M.sha(pp),'rows':[row]})
                outcome={'stage':stage,'state':'quota_expired','exit_code':0,'pid':100+gpu,'output_dir':str(directory),'signals':[],
                    'quota_expired':True,'current_before_stop':{'source_index':0,'policy':'dense'}};outcomes.append(outcome)
                children.append({'identity':{'pid':100+gpu,'ppid':44,'start_ticks':1000+gpu,'argv':command,'executable':'/fake/python'},'command':command,'signals':[],'stage':stage,'quota_expired':True,'output_dir':str(directory)})
            entries.append({'stream':ident,'stage':stage,'outcome':outcome,'coverage_audit_state':'row_coverage_checked',**Q.account_stage('Sand',stage,directory,outcome)})
        streams.append({'id':ident,'arm':arm,'seed':seed,'gpu':gpu,'commands':commands});proc_streams.append({'id':ident,'outcomes':outcomes})
    scope={'owned_indices':sorted({gpu for _,_,gpu in Q.SCHEDULE[role]}),'unassigned_devices':'observe_without_control',
        'timing_scope':'shared_host_operational_measurement','live_training_handoff':False}
    put(root/'release_snapshot.json',{'schema':M.Q_RELEASE_SCHEMA,'issued_by':'root','status':'admitted_for_execution_allocation',
        'dataset':'Sand','host_role':role,'streams':streams,'files_sha256':pins,'gpu_scope':scope,'gpu_uuids':UUIDS,
        'operational_amendment':{'path':'/fake/amendment','sha256':M.sha(amendment)}})
    raw=[{'pid':100+Q.SCHEDULE[role][0][2],'gpu_uuid':UUIDS[Q.SCHEDULE[role][0][2]]}]
    if role=='B':raw.append({'pid':900,'gpu_uuid':UUIDS[0],'label':'unrelated declared Sand'})
    annotated=[]
    for r in raw:
        index=UUIDS.index(r['gpu_uuid']); known=next((c['identity'] for c in children if c['identity']['pid']==r['pid']),None)
        annotated.append(dict(r,physical_index=index,scope='owned_device' if index in scope['owned_indices'] else 'observed_unassigned_device',
            identity_check='matched_owned_child' if known else 'not_owned',current_identity=known))
    put(root/'gpu_observations.json',{'schema':M.Q_SCHEMA,'observations':[{'utc':'synthetic','raw_gpu_processes':raw,
        'validation':'passed','gpu_scope':scope,'all_gpu_processes':annotated}]})
    put(root/'process_outcomes.json',{'schema':M.Q_SCHEMA,'unreaped_owned_children':[],'streams':proc_streams,'all_children':children,'gpu_scope':scope})
    put(root/'coverage_ledger.json',{'schema':M.Q_SCHEMA,'dataset':'Sand','stages':entries,'unreaped_owned_children':[],'audit_errors':[],'abort_reason':None})
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
    assert result['files']['jobs/base_seed0/full_rollout_test/trajectory_000000_base.npz']['bytes']>0


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
    if bad=='wrong_owned_device':o['raw_gpu_processes'][0]['gpu_uuid']=UUIDS[3]
    if bad=='foreign_owned':o['raw_gpu_processes'][0]['pid']=888
    put(path,r)
    if bad=='missing_stage':
        path=root/'coverage_ledger.json';r=M.json_snapshot(path)[0];r['stages'].pop();put(path,r)
    if bad=='bad_scalar':
        path=root/'jobs/base_seed0/full_rollout_test/trajectory_000000_base.json';r=M.json_snapshot(path)[0];r['mean_rollout_mse']=123;put(path,r)
    if bad=='bad_binary':(root/'jobs/base_seed0/full_rollout_test/trajectory_000000_base.npz').write_bytes(b'changed')
    refresh(*queue)
    with pytest.raises(ValueError):M.collect(*queue)


def test_rejected_raw_inventory_preserved_as_pending(queue):
    root=queue[0];path=root/'gpu_observations.json';r=M.json_snapshot(path)[0]
    r['observations'].append({'utc':'synthetic-rejected','validation':'pending','raw_gpu_processes':[{'pid':888,'gpu_uuid':UUIDS[2]}]})
    put(path,r);refresh(*queue);result=M.collect(*queue)
    assert result['gpu_observation_counts']=={'passed':1,'pending':1}
    assert result['gpu_observations']['observations'][1]['raw_gpu_processes'][0]['pid']==888


def test_guarded_prefix_partial_and_stale_aggregate_retained(tmp_path):
    queue=make_queue(tmp_path,completed=200);root=queue[0];directory=root/'jobs/base_seed0/full_rollout_test'
    path=directory/'result.json';r=M.json_snapshot(path)[0];r['rows']=[];put(path,r)
    (directory/'unfinished.npz.tmp').write_bytes(b'unfinished evidence')
    refresh(*queue)
    result=M.collect(*queue);stage=result['stages'][0]
    assert stage['cells'][0]['state']=='recorded_failed_outcome' and stage['rows'][0]['completed_steps']==200
    assert stage['aggregate_snapshot']['state']=='stale_valid_prefix'
    assert 'jobs/base_seed0/full_rollout_test/unfinished.npz.tmp' in result['files']
    metrics,_=M.validate_rollout(stage['rows'][0],'base',0)
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
            p=root/'jobs/base_seed0/full_rollout_test/trajectory_000000_base.json';p.write_bytes(p.read_bytes()+b' ')
        if target=='new_file':(root/'late.txt').write_text('late')
        if target=='new_directory':(root/'late_empty').mkdir()
        if target=='amendment':amendment.write_text('mutated')
        if target=='root_release':release.write_bytes(release.read_bytes()+b' ')
        if target in ('own_source','frozen_source'):
            name=Path(M.__file__) if target=='own_source' else M.HERE/'summarize_goop_graph_support_quota_v2.py'
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
    assert as_goop_names(result['full_rollout'])==F.M.full_summary(as_goop_stages([s for s in stages if s['stage']=='full_rollout_test']))
    assert result['diagnostics']=={name:F.M.diagnostic_summary(stages,name) for name,_,_ in M.STAGES[1:]}
    assert result['full_rollout']['absolute']['mean_rollout_mse']['base']['base']['mean'] is None
    M.publish(tmp_path/'summary.json',result)


@pytest.mark.parametrize('adverse',[False,True])
def test_complete_and_adverse_pure_scientific_functions_identical(adverse):
    stages=complete_stages()
    if adverse:
        stages[0]['rows'][0]=rollout_row('base',0,0,'base',completed=201)
        stages[0]['cells'][0].update(state='recorded_failed_outcome',failure={'category':'coordinate_guard'})
    assert as_goop_names(M.full_summary(stages))==F.M.full_summary(as_goop_stages(stages))


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
            raw['gpu_uuid']=UUIDS[1];row.update(gpu_uuid=UUIDS[1],physical_index=1,scope='observed_unassigned_device')
    if case=='wrong_native_identity':row['current_identity']['argv']=['/fake/foreign']
    if case=='matched_wrong_unassigned':
        raw['gpu_uuid']=UUIDS[1];row.update(gpu_uuid=UUIDS[1],physical_index=1,scope='observed_unassigned_device')
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
        fresh=Q.account_stage('Sand',stage['stage'],directory,stage['outcome'])
        assert fresh['cells']==stage['cells']


def test_collection_publication_exclusive_and_source_stable(queue,tmp_path):
    pins={name:M.sha(M.HERE/name) for name in M.SOURCE_PINS}
    result=M.collect(*queue);output=tmp_path/'published.json';M.publish(output,result);original=output.read_bytes()
    with pytest.raises(FileExistsError):M.publish(output,result)
    assert output.read_bytes()==original and pins=={name:M.sha(M.HERE/name) for name in M.SOURCE_PINS}


def test_prechild_gpu_query_failure_has_explicit_absent_inventory(queue):
    import shutil
    root=queue[0];shutil.rmtree(root/'jobs');(root/'gpu_observations.json').unlink()
    path=root/'process_outcomes.json';process=M.json_snapshot(path)[0]
    process.update(all_children=[],gpu_observations_sha256=None,abort_reason='synthetic initial GPU query failure')
    for stream in process['streams']:stream['outcomes']=[]
    put(path,process)
    path=root/'coverage_ledger.json';ledger=M.json_snapshot(path)[0]
    for entry in ledger['stages']:
        outcome={'stage':entry['stage'],'state':'never_started'}
        entry.update(outcome=outcome,**Q.account_stage('Sand',entry['stage'],root/'jobs'/entry['stream']/entry['stage'],outcome))
    put(path,ledger)
    path=root/'queue_status.json';status=M.json_snapshot(path)[0];status['state']='stopped_requires_review';status['coverage_ledger_sha256']=M.sha(root/'coverage_ledger.json');put(path,status)
    receipt=M.json_snapshot(queue[3])[0];receipt['local_queue_inventory_sha256']=M.inventory_sha256(M.tree_snapshot(root));put(queue[3],receipt)
    result=M.collect(*queue)
    assert result['gpu_observation_file_state']=='absent_before_any_child'
    assert result['gpu_observations']['observations']==[]
    assert all(c['state']=='never_started' for s in result['stages'] for c in s['cells'])


def test_missing_inventory_after_child_is_rejected(queue):
    (queue[0]/'gpu_observations.json').unlink()
    receipt=M.json_snapshot(queue[3])[0];receipt['local_queue_inventory_sha256']=M.inventory_sha256(M.tree_snapshot(queue[0]));put(queue[3],receipt)
    with pytest.raises(ValueError,match='Missing GPU inventory'):M.collect(*queue)


@pytest.mark.parametrize('field',['cohort_sha256','collection_sha256'])
def test_omitted_analysis_binding_rejected(tmp_path,field):
    collections,release,_=make_analysis(tmp_path);r=M.json_snapshot(release)[0];r.pop(field);put(release,r)
    with pytest.raises(ValueError):M.summarize(collections,release)


def test_relocated_copy_preserves_original_paths_and_bytes(queue,tmp_path):
    import shutil
    original,cohort,amendment,release=queue
    copy_root=tmp_path/'relocated';shutil.copytree(original,copy_root)
    original_snapshot=M.tree_snapshot(original);copy_snapshot=M.tree_snapshot(copy_root)
    assert original_snapshot==copy_snapshot
    collected=M.collect(copy_root,cohort,amendment,release)
    assert collected['original_queue_root']==str(original) and collected['queue_root']==str(copy_root)
    assert collected['stages'][0]['cells'][0]['path'].startswith(str(original))
    assert collected['output_tree_state']==original_snapshot
    assert M.tree_snapshot(original)==original_snapshot
    M.publish(tmp_path/'relocated_collection.json',collected)


@pytest.mark.parametrize('bad',['origin','inventory','cell_escape','artifact_escape'])
def test_origin_inventory_and_artifact_escapes_rejected(queue,bad):
    root,cohort,amendment,release=queue
    if bad=='cell_escape':
        path=root/'coverage_ledger.json';value=M.json_snapshot(path)[0];value['stages'][0]['cells'][0]['path']='/unreleased/trajectory_000000_base.json';put(path,value)
    if bad=='artifact_escape':
        path=root/'jobs/base_seed0/full_rollout_test/trajectory_000000_base.json';value=M.json_snapshot(path)[0]
        value['trace_file']='../full_rollout_test/trajectory_000000_base.npz';put(path,value)
        lp=root/'coverage_ledger.json';ledger=M.json_snapshot(lp)[0];ledger['stages'][0].update(Q.account_stage('Sand','full_rollout_test',path.parent,ledger['stages'][0]['outcome']));put(lp,ledger)
    refresh(*queue)
    value=M.json_snapshot(release)[0]
    if bad=='origin':value['original_queue_root']='/wrong/origin'
    if bad=='inventory':value['local_queue_inventory_sha256']='0'*64
    put(release,value)
    with pytest.raises(ValueError):M.collect(*queue)


def test_explicit_sand_grid_and_nonconstant_h314_endpoint():
    assert M.expected('same_state_test')[:5]==[(0,t) for t in (7,85,163,241,319)]
    assert len(M.expected('same_state_valid'))==150 and len(M.expected('clean_validation'))==128
    assert M.expected('clean_validation')[0]==(0,6) and M.expected('clean_validation')[-1]==(29,319)
    assert len(set(M.expected('clean_validation')))==128
    r=rollout_row('mix',2,29,'base');r['mse_per_step']=list(range(314));r['mean_rollout_mse']=156.5;r['mse_at_final_horizon']=313
    r['mse_at_declared_trace_steps']={str(s):s-1 for s in (1,10,50,200,314)}
    value,prefix=M.validate_rollout(r,'mix',2)
    assert value['mean_rollout_mse']==156.5 and value['mse_forecast200']==199 and value['mse_forecast314']==313
    assert 'mse_forecast395' not in value and prefix=={}
    r['horizon']=395
    with pytest.raises(ValueError,match='horizon'):M.validate_rollout(r,'mix',2)


def test_sand_late_failure_keeps_h200_only():
    r=rollout_row('base',0,7,'random25',completed=201)
    values,prefix=M.validate_rollout(r,'base',0)
    assert values['mse_forecast200']==1 and values['mse_forecast314'] is None and values['mean_rollout_mse'] is None
    assert values['predicted_boundary_mean_fraction_particles_outside'] is None
    assert prefix['predicted_boundary_mean_fraction_particles_outside']==0


def test_sand_same_state_equal_source_then_paired_seed_interaction():
    evaluator=M.import_pinned('evaluate_sand_graph_support_final.py','_sand_diags_fixture')
    schedule=[{'source_index':i,'target_frame':t} for i,t in M.expected('same_state_test')]
    stages=[]
    for arm in ('base','mix'):
        for seed in range(3):
            delta=2. if arm=='base' else 1.
            rows=[dict(item,status='complete',policies={
                'previous-observed-base-risk25':{'metrics':{'position_coordinate_mse':10.+delta}},
                'random25':{'metrics':{'position_coordinate_mse':10.}}}) for item in schedule]
            stages.append({'arm':arm,'seed':seed,'stage':'same_state_test','diagnostic_summary':evaluator.summarize(schedule,rows,'same-state'),
                'cells':[dict(item,state='completed_required_outcome') for item in schedule],'outcome':{}})
    result=M.diagnostic_summary(stages,'same_state_test')
    assert result['risk_minus_random_mix_minus_base_interaction']['mean']==-1
    stages[0]['diagnostic_summary']['accuracy']['random25']['position_coordinate_mse']['equal_trajectory_mean']=None
    assert M.diagnostic_summary(stages,'same_state_test')['risk_minus_random_mix_minus_base_interaction']['mean'] is None
