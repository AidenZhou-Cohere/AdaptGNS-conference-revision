"""Synthetic supervisor tests: no real preflight, MPS job or source mutation."""
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).with_name("run_continuation_evaluation_queue.py")
spec = importlib.util.spec_from_file_location("endpoint_queue", SOURCE)
queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
EV = queue.evaluator()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "REPO", tmp_path / "outputs/AdaptGNS")
    for name, rel in [("OUTPUT", "work/new-evaluation"), ("TRAINING", "work/training"), ("SAVED", "work/full-evaluation/same_state"),
        ("NATIVE", "work/native"), ("DATA", "work/data"), ("FREEZE", "work/queue_freeze.json")]:
        monkeypatch.setattr(queue, name, tmp_path / rel)
    helper = tmp_path / "work/run_continuation_evaluation_queue.py"; helper.parent.mkdir(parents=True); helper.write_text("synthetic helper")
    monkeypatch.setattr(queue, "__file__", str(helper))
    monkeypatch.setattr(queue, "evaluator", lambda: EV)
    monkeypatch.setattr(queue, "now", lambda: datetime(2026, 10, 6, 12, tzinfo=timezone.utc))


def training_fixture():
    jobs = []
    for arm, seed in queue.COHORT:
        directory = queue.TRAINING / f"{arm}_seed{seed}"; directory.mkdir(parents=True)
        checkpoint = directory / "checkpoint-extra-10000.pt"; checkpoint.write_bytes(f"synthetic-{arm}-{seed}".encode())
        latest = {"path": checkpoint.name, "sha256": queue.sha(checkpoint), "completed_additional_updates": 10000, "completed_total_updates": 110000}
        write(directory / "latest.json", latest)
        write(directory / "protocol.json", {"arm": arm, "seed": seed, "objective": "faithful"})
        write(directory / "status.json", {"state": "complete", "latest_checkpoint": latest, "completed_additional_updates": 10000, "completed_total_updates": 110000})
        jobs.append({"arm": arm, "seed": seed, "name": f"{arm}_seed{seed}", "state": "complete", "returncode": 0})
    write(queue.TRAINING / "queue_status.json", {"state": "complete", "jobs": jobs})


def test_training_gate_builds_only_six_fixed_checkpoints():
    training_fixture(); manifest, pins = queue.completed_training_manifest()
    assert [(r['arm'], r['seed']) for r in manifest['endpoints']] == list(queue.COHORT)
    assert all(Path(r['checkpoint']).name == 'checkpoint-extra-10000.pt' for r in manifest['endpoints'])
    assert len(pins) == 25 and not queue.OUTPUT.exists()


@pytest.mark.parametrize('defect', ['queue_running','missing_job','duplicate_job','bad_returncode','child_lock','recovery_lock','temp',
    'status_running','wrong_additional','wrong_total','intermediate_pointer','bad_checkpoint_hash','wrong_arm'])
def test_training_completion_refusals(defect):
    training_fixture(); directory = queue.TRAINING / 'base_seed0'
    queue_file = queue.TRAINING / 'queue_status.json'; q = queue.read(queue_file)
    status = queue.read(directory/'status.json'); latest = queue.read(directory/'latest.json')
    if defect == 'queue_running': q['state']='running'
    elif defect == 'missing_job': q['jobs'].pop()
    elif defect == 'duplicate_job': q['jobs'][0]=q['jobs'][1]
    elif defect == 'bad_returncode': q['jobs'][0]['returncode']=1
    elif defect in ('child_lock','recovery_lock','temp'):
        (directory / {'child_lock':'run.lock','recovery_lock':'run.lock.recovery','temp':'checkpoint.pt.tmp'}[defect]).write_text('preserve')
    elif defect == 'status_running': status['state']='running'
    elif defect == 'wrong_additional': status['completed_additional_updates']=9999
    elif defect == 'wrong_total': status['completed_total_updates']=100000
    elif defect == 'intermediate_pointer': latest['path']='checkpoint-extra-7500.pt'; status['latest_checkpoint']=latest
    elif defect == 'bad_checkpoint_hash': (directory/'checkpoint-extra-10000.pt').write_bytes(b'changed')
    elif defect == 'wrong_arm': write(directory/'protocol.json',{'arm':'mix','seed':0,'objective':'faithful'})
    write(queue_file,q);write(directory/'status.json',status);write(directory/'latest.json',latest)
    before={str(p):p.read_bytes() for p in queue.TRAINING.rglob('*') if p.is_file()}
    with pytest.raises(RuntimeError):queue.completed_training_manifest()
    assert {str(p):p.read_bytes() for p in queue.TRAINING.rglob('*') if p.is_file()} == before


def test_exact_heavy_modules_and_between_job_supervisors():
    text='\n'.join([
        '10 /python -m research.native_graph_rollout',
        '11 /python -u -m research.continuation_evaluation --mode observed',
        '12 /python /work/run_continuation_evaluation_queue.py --check',
        '13 /python /work/run_graph_support_queue.py --launch',
        '14 /python /work/run_native_rollout_queue.py',
        '15 /python -m research.continuation_evaluation_extra',
        '16 /bin/echo research.full_training',
        '17 /python /work/run_continuation_evaluation_queue.py --launch'])
    assert [p['pid'] for p in queue.heavy_processes(text,17)]==[10,11,12,13,14]
    assert queue.heavy_processes('19 /python -m research.full_training "unterminated',0)[0]['pid']==19
    with pytest.raises(RuntimeError):queue.heavy_processes('unparseable',0)


def test_fixed_order_parent_paths_and_mps_commands():
    assert len(queue.JOBS)==12 and [j[0] for j in queue.JOBS]==['observed']*6+['autonomous']*6
    for mode,arm,seed in queue.JOBS:
        cmd=queue.command(mode,arm,seed,executable='/fake/python')
        assert cmd[:4]==['/fake/python','-u','-m','research.continuation_evaluation']
        args=dict(zip(cmd[4::2],cmd[5::2]))
        assert args['--arm']==arm and args['--seed']==str(seed) and args['--device']=='mps' and args['--threads']=='2'
        assert ('--saved-same-state-dir' in args)==(mode=='observed')
        if mode=='observed':assert args['--saved-same-state-dir']==str(queue.SAVED/f'faithful_seed{seed}')
        assert '--resume' not in cmd and '--clear-stale-lock' not in cmd


def test_forecast_includes_all_remaining_work_and_never_claims_guarantee():
    early=datetime(2026,10,6,14,59,59,tzinfo=timezone.utc)
    value=queue.forecast(queue.JOBS,queue.INITIAL_ALLOWANCES,early,initial=True)
    assert value['remaining_allowance_seconds']==36000 and 'not a guaranteed' in value['scope']
    with pytest.raises(RuntimeError,match='latest launch'):queue.forecast(queue.JOBS,queue.INITIAL_ALLOWANCES,queue.LATEST_START,initial=True)
    with pytest.raises(RuntimeError,match='no longer fits'):queue.forecast([('autonomous','base',0)],queue.INITIAL_ALLOWANCES,queue.FINISH_TARGET-timedelta(seconds=5699))
    assert queue.forecast([('autonomous','base',0)],queue.INITIAL_ALLOWANCES,queue.FINISH_TARGET-timedelta(seconds=5700))
    with pytest.raises(RuntimeError,match='Hard research cutoff'):queue.forecast([],queue.INITIAL_ALLOWANCES,queue.HARD_CUTOFF)


def test_exact_freeze_inventory_and_static_pins(monkeypatch):
    monkeypatch.setattr(queue,'PINNED',{'research/continuation_evaluation.py':hashlib.sha256(b'fixed').hexdigest()})
    for relative in queue.required_frozen_files():
        p=queue.ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixed')
    frozen={'schema':1,'scope':'reviewed_continuation_evaluation_queue_v1','evaluation_commit':queue.EVALUATOR_COMMIT,
        'latest_launch_utc':queue.LATEST_START.isoformat(),'finish_target_utc':queue.FINISH_TARGET.isoformat(),
        'initial_job_allowances_seconds':queue.INITIAL_ALLOWANCES,
        'files_sha256':{name:queue.sha(queue.ROOT/name) for name in queue.required_frozen_files()}}
    queue.verify_freeze(frozen)
    for defect in ('missing','extra','changed'):
        bad=copy.deepcopy(frozen)
        if defect=='missing':bad['files_sha256'].pop(next(iter(bad['files_sha256'])))
        elif defect=='extra':bad['files_sha256']['extra.py']='x'
        else:bad['files_sha256']['outputs/AdaptGNS/research/continuation_evaluation.py']='0'*64
        with pytest.raises(RuntimeError):queue.verify_freeze(bad)


def test_actual_frozen_loader_checks_synthetic_six_endpoints_three_parents_and_schedule(tmp_path,monkeypatch):
    from research.tests.test_continuation_evaluation import endpoint_cohort
    fixture=tmp_path/'tiny';fixture.mkdir()
    _,document,_=endpoint_cohort(fixture,monkeypatch)
    calls=[];original=EV.full.check_checkpoint
    def spy(payload,*args,**kwargs):
        calls.append(payload['completed_steps']);return original(payload,*args,**kwargs)
    monkeypatch.setattr(EV.full,'check_checkpoint',spy)
    entries,configs,pins=queue.verify_cohort_document(document,'synthetic-metadata')
    assert len(entries)==len(configs)==6 and calls==[100000]*3
    assert not any('continuation-cohort-check-' in p for p in pins)
    assert not queue.OUTPUT.exists()
    # The helper must use the loader's paired-history gate, not trust publication counters alone.
    torch=EV.torch;path=Path(document['endpoints'][0]['checkpoint']);payload=torch.load(path,weights_only=True)
    payload['history']['graph_updates'][0]['noise_sha256']='different paired schedule'
    torch.save(payload,path);new_sha=EV.full.sha256(path);document['endpoints'][0]['sha256']=new_sha
    latest=queue.read(path.parent/'latest.json');latest['sha256']=new_sha;write(path.parent/'latest.json',latest)
    status=queue.read(path.parent/'status.json');status['latest_checkpoint']=latest;write(path.parent/'status.json',status)
    with pytest.raises(ValueError,match='Paired frame/noise'):queue.verify_cohort_document(document,'synthetic-metadata')


def publication_fixture(mode='observed'):
    queue.OUTPUT.mkdir(parents=True,exist_ok=True)
    endpoint={'arm':'base','seed':0,'checkpoint':'synthetic.pt','sha256':'synthetic_endpoint'}
    manifest={'schema':1,'scope':'faithful_graph_support_110k_endpoints','endpoints':[endpoint]}
    (queue.OUTPUT/'cohort_manifest.json').write_bytes(queue.json_bytes(manifest))
    config={'parent_checkpoint_sha256':'parent','seed':0,'arm':'base'}
    expected=[{'split':'valid','source_index':i//8,'target_frame':7+i%8,'trajectory_id':f'valid:{i//8}'} for i in range(128)]
    expected += [{'split':'test','source_index':s,'target_frame':t,'trajectory_id':f'test:{s}'} for s in range(3,30) for t in (7,106,205,304,403,502,601,700,799,898,1000)]
    test={'records':[{'id':f'test:{i}'} for i in range(30)],'source':{'sha256':'official'}}
    prepared={'manifest':manifest,'configs':{('base',0):config},'entries':{('base',0):endpoint},'expected':expected,'test':test,
        'base_pins':{},'saved_pins':{0:{}},'saved':{0:{'parent':'saved'}},'immutable':{}}
    directory=queue.OUTPUT/mode/'base_seed0';directory.mkdir(parents=True)
    lineage={'arm':'base','seed':0,'original_seed':0,'checkpoint_sha256':endpoint['sha256'],'parent_checkpoint_sha256':'parent',
        'continuation_config_sha256':EV.support.original.config_hash(config),'completed_total_updates':110000,'completed_additional_updates':10000}
    protocol={**lineage,'mode':mode,'objective':'faithful','cohort':manifest['endpoints'],'source_indices':list(range(3,30)),
        'trajectory_ids':[f'test:{i}' for i in range(3,30)],'source_tfrecord':test['source'],
        'policies':list(queue.OBSERVED_POLICIES if mode=='observed' else queue.AUTONOMOUS_POLICIES),'horizon':1 if mode=='observed' else 995,
        'guards':{'max_candidate_pairs':EV.bridge.MAX_PAIRS,'max_abs_coordinate':EV.bridge.MAX_ABS},'threads':2,'runtime':{'device':'mps','radius_backend':'scipy_host','mps_fallback_environment':'0'},'deadline_utc':queue.HARD_CUTOFF.isoformat(),
        'input_files_sha256':queue.job_input_pins(prepared,mode,0),'expected_frames':expected if mode=='observed' else None,
        'saved_parent_test_input_archive':prepared['saved'][0] if mode=='observed' else None}
    write(directory/'protocol.json',protocol);digest=queue.sha(directory/'protocol.json');records=[]
    if mode=='observed':
        for item in expected:
            stem=EV.bridge.stem(item);array=directory/(stem+'.npz');array.write_bytes(b'synthetic arrays')
            row={**item,**{k:lineage[k] for k in ('arm','original_seed','checkpoint_sha256','parent_checkpoint_sha256')},
                'random_seed_material':EV.random_material(0,item),'status':'complete','failure':None,'array_file':array.name,'array_sha256':queue.sha(array),'protocol_sha256':digest,
                'cases':{p:{'status':'complete','failure':None,'metrics':{'coordinate_mse':1.}} for p in queue.OBSERVED_POLICIES}}
            path=directory/(stem+'.json');write(path,row);records.append(EV.bridge.record_index(row,path))
    else:
        write(directory/'lineage_identity.json',lineage)
        for i in range(3,30):
            for policy in queue.AUTONOMOUS_POLICIES:
                stem=f'trajectory_{i:06d}_{policy}';trace=directory/(stem+'.npz');trace.write_bytes(b'synthetic traces')
                row={'trajectory_id':f'test:{i}','source_index':i,'policy':policy,'status':'complete','failure':None,'horizon':995,
                    'n_particles':2,'rng_seed':93000+i,'completed_steps':995,'mse_at_steps':{str(t):1. for t in EV.full.TRACE_STEPS},
                    'mse_at_final_horizon':1.,'mean_rollout_mse':1.,'mean_directed_edges':4.,'total_wall_seconds':.01,
                    'total_network_passes':995+(policy=='laggedrisk25'),'forecast_network_passes':995,'trace_file':trace.name,
                    'trace_sha256':queue.sha(trace),'protocol_sha256':digest,'mse_per_step':[1.]*995}
                path=directory/(stem+'.json');write(path,row);records.append(EV.full.compact_record(row,path))
    result={'state':'complete','objective':'faithful','arm':'base','seed':0,'checkpoint_sha256':endpoint['sha256'],
        'parent_checkpoint_sha256':'parent','protocol_sha256':digest,'records':records,'required_frames':425,'complete_frames':425,'failed_frames':0}
    if mode=='autonomous':result['summary']=EV.full.summarize(records,protocol['trajectory_ids'])
    write(directory/'result.json',result)
    write(directory/'status.json',{'state':'complete','result_sha256':queue.sha(directory/'result.json'),'committed_frames':425,'arm':'base','seed':0})
    return directory,prepared


def reseal(directory,result):
    write(directory/'result.json',result);status=queue.read(directory/'status.json');status['result_sha256']=queue.sha(directory/'result.json');write(directory/'status.json',status)


@pytest.mark.parametrize('mode',['observed','autonomous'])
def test_completed_job_recovery_verifies_full_population(mode):
    directory,prepared=publication_fixture(mode)
    result=queue.verify_job(directory,mode,'base',0,prepared)
    assert result['records']==(425 if mode=='observed' else 135) and result['scientific_failed_policy_outcomes']==0


@pytest.mark.parametrize('mode',['observed','autonomous'])
def test_scientific_guard_failure_is_preserved_and_counts_as_completed_outcome(mode):
    directory,prepared=publication_fixture(mode);result=queue.read(directory/'result.json');compact=result['records'][0]
    path=directory/compact['record_file'];row=queue.read(path)
    if mode=='observed':
        failure={'category':'coordinate_guard'};row['status']='failed';row['failure']={'category':'case_failures','cases':[{'case':'dense','failure':failure}]}
        row['cases']['dense']={'status':'failed','failure':failure,'metrics':None}
        write(path,row);result['records'][0]=EV.bridge.record_index(row,path);result['complete_frames']=424;result['failed_frames']=1
    else:
        row.update(status='failed',failure={'category':'coordinate_guard','forecast_step':3},completed_steps=2,mse_per_step=[1.,1.],
            mse_at_final_horizon=None,mean_rollout_mse=None,mean_directed_edges=None)
        write(path,row);result['records'][0]=EV.full.compact_record(row,path)
        result['summary']=EV.full.summarize(result['records'],[f'test:{i}' for i in range(3,30)])
    reseal(directory,result)
    assert queue.verify_job(directory,mode,'base',0,prepared)['scientific_failed_policy_outcomes']==1


@pytest.mark.parametrize('mode,defect',[('observed','missing'),('observed','duplicate'),('observed','bad_lineage'),('observed','orphan'),('observed','temp'),
    ('observed','bad_policy'),('observed','native_parity'),('autonomous','missing'),('autonomous','bad_lineage'),('autonomous','native_parity'),('autonomous','wrong_rng')])
def test_checksum_consistent_invalid_publications_are_refused(mode,defect):
    directory,prepared=publication_fixture(mode);result=queue.read(directory/'result.json');compact=result['records'][0];path=directory/compact['record_file'];row=queue.read(path)
    if defect=='missing':result['records'].pop()
    elif defect=='duplicate':result['records'][0]=result['records'][1]
    elif defect=='orphan':(directory/'unexpected.npz').write_bytes(b'preserve orphan')
    elif defect=='temp':(directory/'unfinished.tmp').write_bytes(b'preserve temporary')
    elif defect=='bad_lineage' and mode=='autonomous':
        identity=queue.read(directory/'lineage_identity.json');identity['arm']='mix';write(directory/'lineage_identity.json',identity)
    else:
        if defect=='bad_lineage':row['arm']='mix'
        elif defect=='bad_policy':row['cases'].pop('dense')
        elif defect=='wrong_rng':row['rng_seed']+=1
        elif defect=='native_parity':
            row.update(status='failed',failure={'category':'native_parity_failure'})
            if mode=='observed':row['cases']={p:{'status':'failed','failure':row['failure'],'metrics':None} for p in queue.OBSERVED_POLICIES};result['complete_frames']=424;result['failed_frames']=1
            else:row.update(completed_steps=0,mse_per_step=[],mse_at_final_horizon=None,mean_rollout_mse=None,mean_directed_edges=None)
        write(path,row);result['records'][0]=EV.bridge.record_index(row,path) if mode=='observed' else EV.full.compact_record(row,path)
        if mode=='autonomous':result['summary']=EV.full.summarize(result['records'],[f'test:{i}' for i in range(3,30)])
    reseal(directory,result)
    with pytest.raises((RuntimeError,ValueError)):queue.verify_job(directory,mode,'base',0,prepared)


def test_check_is_explicit_and_does_not_write_output_or_spawn(monkeypatch,capsys):
    prepared={'manifest':{'endpoints':[]},'timing':{'synthetic':True}}
    monkeypatch.setattr(queue,'preflight',lambda:prepared)
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('check spawned')))
    assert queue.main(['--check'])==0 and not queue.OUTPUT.exists()
    assert json.loads(capsys.readouterr().out)['outputtree_written'] is False
    with pytest.raises(SystemExit):queue.main([])
    with pytest.raises(SystemExit):queue.main(['--check','--launch'])


def test_preflight_refuses_conflict_before_loading_or_output(monkeypatch):
    monkeypatch.setattr(queue,'assert_idle',lambda:(_ for _ in ()).throw(RuntimeError('heavy process')))
    monkeypatch.setattr(queue,'evaluator',lambda:(_ for _ in ()).throw(AssertionError('loaded while busy')))
    with pytest.raises(RuntimeError,match='heavy process'):queue.preflight()
    assert not queue.OUTPUT.exists()


def test_existing_output_is_never_entered_or_retried(monkeypatch):
    queue.OUTPUT.mkdir(parents=True);(queue.OUTPUT/'preserved.log').write_text('partial')
    monkeypatch.setattr(queue,'assert_idle',lambda:(_ for _ in ()).throw(AssertionError('process scan after existing tree')))
    with pytest.raises(RuntimeError,match='Existing outputtree'):queue.preflight()
    assert (queue.OUTPUT/'preserved.log').read_text()=='partial'


def test_full_preflight_check_glue_is_nonwriting_and_checks_all_parent_archives(monkeypatch):
    # Small metadata-only injected cohort; the real six-payload loader is tested above.
    write(queue.FREEZE, {'files_sha256':{}})
    for name in ('metadata.json','valid.json','test.json'):
        write(queue.DATA/name, {})
    doc={'schema':1,'scope':'faithful_graph_support_110k_endpoints','endpoints':[]}
    configs={(arm,seed):{'parent_checkpoint_sha256':f'parent{seed}'} for arm,seed in queue.COHORT}
    expected=[{'split':'valid','source_index':0,'target_frame':i+7,'trajectory_id':'v'} for i in range(128)]
    expected += [{'split':'test','source_index':3+i//11,'target_frame':i%11+7,'trajectory_id':'t'} for i in range(297)]
    valid={'metadata':{},'records':[]};test={'metadata':{},'records':[]}
    checked=[]
    for seed in range(3):
        d=queue.SAVED/f'faithful_seed{seed}'
        write(d/'result.json',{});write(d/'protocol.json',{})
        write(d/'status.json',{'state':'complete','result_sha256':queue.sha(d/'result.json')})
    monkeypatch.setattr(queue,'assert_idle',lambda:checked.append('idle'))
    monkeypatch.setattr(queue,'verify_freeze',lambda frozen:None)
    monkeypatch.setattr(queue,'completed_training_manifest',lambda:(doc,{}))
    monkeypatch.setattr(queue,'verify_cohort_document',lambda manifest,metadata:({},configs,{}))
    monkeypatch.setattr(EV.bridge,'expected_frames',lambda *args:(expected,valid,test))
    monkeypatch.setattr(EV.full,'select_records',lambda *args:list(range(3,30)))
    def saved(directory,parent,expected):
        checked.append((directory.name,parent));return {},{'parent':parent}
    monkeypatch.setattr(EV.bridge,'verify_saved_test',saved)
    monkeypatch.setattr(queue,'evaluation_source_paths',lambda:[])
    monkeypatch.setattr(queue,'manifest_array_pins',lambda *args:{})
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('preflight spawned')))
    before={str(p):p.read_bytes() for p in queue.ROOT.rglob('*') if p.is_file()}
    assert queue.preflight()['saved']=={s:{'parent':f'parent{s}'} for s in range(3)}
    assert checked==['idle',('faithful_seed0','parent0'),('faithful_seed1','parent1'),('faithful_seed2','parent2'),'idle']
    assert not queue.OUTPUT.exists() and before=={str(p):p.read_bytes() for p in queue.ROOT.rglob('*') if p.is_file()}


def launch_fixture(monkeypatch, *, fail_code=0, verify_error=False, duration=10, interrupt=False, mutate=False):
    write(queue.FREEZE,{'synthetic':'frozen'});write(queue.TRAINING/'queue_status.json',{'state':'complete'})
    pin=queue.ROOT/'immutable';pin.write_bytes(b'fixed')
    prepared={'manifest':{'schema':1,'endpoints':[]},'expected':[], 'timing':{},'immutable':{str(pin):queue.sha(pin)}}
    timer={'seconds':0};calls=[]
    base=datetime(2026,10,6,12,tzinfo=timezone.utc)
    monkeypatch.setattr(queue,'now',lambda:base+timedelta(seconds=timer['seconds']))
    monkeypatch.setattr(queue.time,'perf_counter',lambda:timer['seconds'])
    monkeypatch.setattr(queue,'assert_idle',lambda:None)
    def verify(*args):
        if verify_error:raise RuntimeError('invalid committed output')
        return {'passed':True,'scientific_failed_policy_outcomes':1,'failures':[{'category':'coordinate_guard'}]}
    monkeypatch.setattr(queue,'verify_job',verify)
    class Child:
        def __init__(self,pid):self.pid=pid
        def wait(self):
            if interrupt:raise KeyboardInterrupt('synthetic interruption')
            timer['seconds']+=duration
            if mutate:pin.write_bytes(b'changed')
            return fail_code
        def terminate(self):raise AssertionError('helper must not auto-interrupt child')
    def popen(command,**kwargs):
        calls.append((command,kwargs))
        return Child(500+len(calls))
    monkeypatch.setattr(queue.subprocess,'Popen',popen)
    return prepared,calls


def test_fake_launch_is_sequential_all12_and_keeps_scientific_failures(monkeypatch):
    prepared,calls=launch_fixture(monkeypatch)
    assert queue.launch(prepared)==0
    status=queue.read(queue.OUTPUT/'queue_status.json')
    assert status['state']=='complete' and not status['remaining_work_is_incomplete']
    assert len(calls)==13 and calls[0][0][0]=='caffeinate'
    assert all(j['state']=='complete' and j['verification']['scientific_failed_policy_outcomes']==1 for j in status['jobs'])
    assert [c[0] for c in calls[1:]]==[queue.command(*job) for job in queue.JOBS]
    assert all(c[1]['env']['PYTORCH_ENABLE_MPS_FALLBACK']=='0' for c in calls[1:])
    assert not (queue.OUTPUT/'run.lock').exists()
    assert queue.read(queue.OUTPUT/'launch.json')['jobs'][0]['state']=='unstarted'
    assert len(list(queue.OUTPUT.glob('*.log')))==12
    with pytest.raises(RuntimeError,match='Never overwrite'):queue.launch(prepared)
    assert len(calls)==13


@pytest.mark.parametrize('kind',['nonzero','bad_output','immutable_change'])
def test_fake_launch_stops_once_on_operational_failure_and_retains_log_pid(monkeypatch,kind):
    prepared,calls=launch_fixture(monkeypatch,fail_code=2 if kind=='nonzero' else 0,verify_error=kind=='bad_output',mutate=kind=='immutable_change')
    assert queue.launch(prepared)!=0
    status=queue.read(queue.OUTPUT/'queue_status.json')
    assert status['state']=='error' and status['remaining_work_is_incomplete']
    assert len(calls)==2 and status['jobs'][0]['state']=='error' and status['jobs'][0]['pid']==502
    assert status['jobs'][0]['log_sha256']==queue.sha(queue.OUTPUT/'observed_base_seed0.log')
    assert all(j['state']=='unstarted' for j in status['jobs'][1:])


def test_measured_slow_child_increases_remaining_forecast_and_stops_new_jobs(monkeypatch):
    prepared,calls=launch_fixture(monkeypatch,duration=4000)
    assert queue.launch(prepared)==2
    status=queue.read(queue.OUTPUT/'queue_status.json')
    assert len(calls)==2 and status['state']=='forecast_stopped' and status['jobs'][0]['state']=='complete'
    assert status['remaining_work_is_incomplete'] and all(j['state']=='unstarted' for j in status['jobs'][1:])
    assert 'no longer fits' in status['stop_reason']


def test_interrupted_supervisor_preserves_current_child_identity_without_killing(monkeypatch):
    prepared,calls=launch_fixture(monkeypatch,interrupt=True)
    with pytest.raises(KeyboardInterrupt):queue.launch(prepared)
    status=queue.read(queue.OUTPUT/'queue_status.json')
    assert len(calls)==2 and status['state']=='interrupted' and status['jobs'][0]['pid']==502
    assert (queue.OUTPUT/'queue_error.json').is_file() and (queue.OUTPUT/'observed_base_seed0.log').is_file()


def test_process_matching_ignores_shell_echo_and_python_code_mentions():
    text='\n'.join([
        '30 /bin/echo -m research.continuation_evaluation',
        '31 /bin/zsh -lc "/python /work/run_continuation_evaluation_queue.py --check"',
        '32 /python -c "print(\'research.native_graph_rollout\')"',
        '33 /python -W ignore -X dev -m research.continuation_evaluation',
        '34 /python3.12 -u -mresearch.native_graph_rollout',
        '35 /work/run_continuation_evaluation_queue.py --launch'])
    assert [p['pid'] for p in queue.heavy_processes(text,0)]==[33,34,35]


@pytest.mark.parametrize('text',['{"state":"complete","state":"running"}','{"seconds":NaN}','{"seconds":Infinity}','{"seconds":-Infinity}'])
def test_ambiguous_or_nonfinite_json_metadata_is_rejected(tmp_path,text):
    path=tmp_path/'metadata.json';path.write_text(text)
    with pytest.raises(RuntimeError):queue.read(path)


def test_failed_lock_acquisition_does_not_write_competing_owner_state(monkeypatch):
    prepared,calls=launch_fixture(monkeypatch)
    class CompetingLock:
        def __init__(self,directory):self.directory=directory
        def __enter__(self):
            write(self.directory/'queue_status.json',{'state':'competing_owner'})
            (self.directory/'run.lock').write_text('another process lock')
            raise RuntimeError('Lock already acquired by another owner')
        def __exit__(self,*args):raise AssertionError('unowned lock release')
    monkeypatch.setattr(EV.full,'RunLock',CompetingLock)
    with pytest.raises(RuntimeError,match='another owner'):queue.launch(prepared)
    assert not calls and queue.read(queue.OUTPUT/'queue_status.json')=={'state':'competing_owner'}
    assert (queue.OUTPUT/'run.lock').read_text()=='another process lock'
    assert not (queue.OUTPUT/'queue_error.json').exists()
