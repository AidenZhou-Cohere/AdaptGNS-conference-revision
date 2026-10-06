"""Synthetic-only queue safety tests. Never run real --check or launch."""
import copy
from datetime import datetime,timedelta,timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE=Path(__file__).with_name('run_native_random_envelope_queue.py')
spec=importlib.util.spec_from_file_location('random_queue',SOURCE)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture(autouse=True)
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'ROOT',tmp_path);monkeypatch.setattr(q,'REPO',tmp_path/'repo')
    for key in ('OUTPUT','NATIVE','TRAINING','ENDPOINT','SAVED','DATA'):
        monkeypatch.setattr(q,key,tmp_path/key.lower())
    for key in ('FREEZE','RELEASE'):
        monkeypatch.setattr(q,key,tmp_path/(key.lower()+'.json'))
    monkeypatch.setattr(q,'now',lambda:datetime(2026,10,6,7,tzinfo=timezone.utc))
    helper=tmp_path/'helpers/run_native_random_envelope_queue.py';helper.parent.mkdir();helper.write_text('synthetic helper')
    monkeypatch.setattr(q,'__file__',str(helper))
    monkeypatch.setattr(q,'assert_idle',lambda:None)


def test_exact_six_original_commands_no_wrapper_or_retry():
    assert q.JOBS==(('faithful',0),('faithful',1),('faithful',2),('nll',0),('nll',1),('nll',2))
    for objective,seed in q.JOBS:
        command=q.command(objective,seed,'/synthetic/python')
        assert command[:4]==['/synthetic/python','-u','-m','research.native_random_envelope']
        assert command[-1]=='--launch'
        args=dict(zip(command[4:-1:2],command[5:-1:2]))
        assert args['--device']=='mps'
        assert args['--checkpoint']==str(q.REPO/f'research/results/full_waterdrop_100k/{objective}_seed{seed}/checkpoint-100000.pt')
        assert args['--saved-same-state-dir']==str(q.SAVED/f'{objective}_seed{seed}')
        assert args['--output-dir']==str(q.OUTPUT/f'{objective}_seed{seed}')
        assert args['--validation-manifest']==str(q.DATA/'valid.json')
        assert args['--test-manifest']==str(q.DATA/'test.json')
        assert not {'--resume','--clear-stale-lock','caffeinate'} & set(command)


def test_forecast_full45minutes_includes900seconds_reserve():
    at=q.LATEST_START-timedelta(seconds=1)
    result=q.forecast(q.JOBS,300,at,initial=True)
    assert result['remaining_allowance_seconds']==2700 and result['summary_audit_reserve_seconds']==900
    assert 'not a guarantee' in result['scope']
    assert q.forecast(q.JOBS[1:],300,at+timedelta(seconds=300),started=at)['forecast_execution_and_analysis_seconds']==2700


@pytest.mark.parametrize('kind',['late_start','too_slow','overrun','finish','lower','backward','cutoff'])
def test_forecast_refusals(kind):
    at=datetime(2026,10,6,7,tzinfo=timezone.utc)
    with pytest.raises(RuntimeError):
        if kind=='late_start':q.forecast(q.JOBS,300,q.LATEST_START,initial=True)
        elif kind=='too_slow':q.forecast(q.JOBS[1:],400,at)
        elif kind=='overrun':q.forecast(q.JOBS[1:],300,at+timedelta(seconds=301),started=at)
        elif kind=='finish':q.forecast([q.JOBS[-1]],300,q.FINISH_TARGET-timedelta(seconds=1199))
        elif kind=='lower':q.forecast(q.JOBS,299,at)
        elif kind=='backward':q.forecast(q.JOBS,300,at,started=at+timedelta(seconds=1))
        else:q.forecast([],300,q.HARD_CUTOFF)


@pytest.mark.parametrize('tree',['TRAINING','ENDPOINT'])
@pytest.mark.parametrize('state',['empty','complete','active','stale'])
def test_training_endpoint_any_tree_refuses_pretraining(tree,state):
    root=getattr(q,tree);root.mkdir()
    if state=='complete':write(root/'queue_status.json',{'state':'complete'})
    if state in ('active','stale'):(root/'run.lock').write_text(state)
    before={p:p.read_bytes() for p in root.iterdir() if p.is_file()}
    with pytest.raises(RuntimeError):q.pretraining_only()
    assert {p:p.read_bytes() for p in root.iterdir() if p.is_file()}==before


def test_process_inventory_exact_modules_and_supervisors():
    text='\n'.join(['10 /python -u -m research.native_random_envelope --launch','11 /python /x/run_graph_support_queue.py --launch',
        '12 /python /x/run_native_random_envelope_queue.py --launch','13 /python -m research.continuation_evaluation',
        '14 /python -m research.native_random_envelope_extra','15 /python -c "research.native_random_envelope"',
        '16 /python -m research.summarize_native_random_envelope','17 /bin/echo research.native_random_envelope'])
    assert [row['pid'] for row in q.heavy_processes(text,12)]==[10,11,13,16]
    assert q.heavy_processes('18 /python -m research.native_random_envelope "unterminated',0)[0]['pid']==18
    with pytest.raises(RuntimeError):q.heavy_processes('cannot parse',0)


def freeze_fixture(monkeypatch):
    source=q.ROOT/'frozen.py';source.write_text('released bytes')
    monkeypatch.setattr(q,'required_frozen_files',lambda:{'frozen.py'})
    value={'schema':1,'scope':q.SCOPE,'scientific_review_complete':True,'latest_launch_utc':q.LATEST_START.isoformat(),
        'finish_target_utc':q.FINISH_TARGET.isoformat(),'initial_job_allowance_seconds':300,'summary_audit_reserve_seconds':900,
        'total_budget_seconds':2700,'files_sha256':{'frozen.py':q.sha(source)}}
    write(q.FREEZE,value);return value


@pytest.mark.parametrize('kind',['unreviewed','missing','extra','changed','bool_reserve'])
def test_freeze_fails_closed_and_no_guessed_hash(kind,monkeypatch):
    frozen=freeze_fixture(monkeypatch);q.verify_freeze(frozen)
    if kind=='unreviewed':frozen['scientific_review_complete']=False
    elif kind=='missing':frozen['files_sha256']={}
    elif kind=='extra':frozen['files_sha256']['extra.py']='a'*64
    elif kind=='changed':(q.ROOT/'frozen.py').write_text('changed')
    else:frozen['summary_audit_reserve_seconds']=True
    with pytest.raises(RuntimeError):q.verify_freeze(frozen)


def release_fixture(monkeypatch):
    freeze_fixture(monkeypatch)
    summary=q.ROOT/'analysis/summary.json';audit=q.ROOT/'analysis/audit.json'
    native={'native_queue_sha256':'a'*64,'native_jobs':[{'name':n,'protocol_sha256':'b'*64,'result_sha256':'c'*64} for n in q.NAMES]}
    runs=[]
    for objective,seed in q.JOBS:
        name=f'{objective}_seed{seed}';directory=q.NATIVE/name
        checkpoint=q.REPO/f'research/results/full_waterdrop_100k/{name}/checkpoint-100000.pt'
        checkpoint.parent.mkdir(parents=True);checkpoint.write_text(name)
        runs.append({'objective':objective,'seed':seed,'state':'complete','eligible_for_aggregation':True,
            'checkpoint_sha256':q.sha(checkpoint),'protocol_sha256':'b'*64,'input_files_sha256':{
                str((directory/'result.json').resolve()):'c'*64,str((directory/'protocol.json').resolve()):'b'*64}})
    write(summary,{'scope':'post-inspection native convention follow-up on original six100k models',
        'audit':{'passed':True},'native_runs':runs,'original_runs':[{}]*6})
    write(audit,{'passed':True,'native_outcomes':810,'original_outcomes':810,'source_summary_sha256':q.sha(summary)})
    release={'schema':1,'scope':q.SCOPE,'root_native_identity_verified':True,'freeze_sha256':q.sha(q.FREEZE),**native,
        'recorded_utc':q.now().isoformat(),'root_process_check_utc':q.now().isoformat(),
        'strict_native_summary_path':str(summary),'strict_native_summary_sha256':q.sha(summary),
        'native_scalar_audit_path':str(audit),'native_scalar_audit_sha256':q.sha(audit)}
    write(q.RELEASE,release);return release,native


@pytest.mark.parametrize('kind',['stale','future','process_stale','root_false','freeze','native','audit_partial','summary_false','audit_binding'])
def test_release_fresh_identity_analysis_gates(kind,monkeypatch):
    release,native=release_fixture(monkeypatch);assert len(q.verify_release(release,native))==8
    if kind=='stale':release['recorded_utc']=(q.now()-timedelta(seconds=901)).isoformat()
    elif kind=='future':release['recorded_utc']=(q.now()+timedelta(seconds=1)).isoformat()
    elif kind=='process_stale':release['root_process_check_utc']=(q.now()-timedelta(seconds=901)).isoformat()
    elif kind=='root_false':release['root_native_identity_verified']=1
    elif kind=='freeze':release['freeze_sha256']='d'*64
    elif kind=='native':release['native_queue_sha256']='d'*64
    else:
        prefix='strict_native_summary' if kind=='summary_false' else 'native_scalar_audit'
        path=Path(release[prefix+'_path']);value=q.read(path)
        if kind=='audit_partial':value['native_outcomes']=809
        elif kind=='summary_false':value['audit']['passed']=False
        else:value['source_summary_sha256']='e'*64
        write(path,value);release[prefix+'_sha256']=q.sha(path)
    with pytest.raises(RuntimeError):q.verify_release(release,native)


def prior_fixture():
    entries=[]
    for objective,seed in q.JOBS:
        name=f'{objective}_seed{seed}';directory=q.NATIVE/name;directory.mkdir(parents=True)
        checkpoint=q.REPO/f'research/results/full_waterdrop_100k/{name}/checkpoint-100000.pt';checkpoint.parent.mkdir(parents=True);checkpoint.write_text(name)
        protocol={'objective':objective,'seed':seed,'checkpoint_sha256':q.sha(checkpoint),'source_indices':list(range(3,30)),
            'trajectory_ids':[str(i) for i in range(3,30)]};write(directory/'protocol.json',protocol)
        ph=q.sha(directory/'protocol.json');records=[]
        for i in range(3,30):
            for policy in q.NATIVE_POLICIES:
                stem=f'trajectory_{i:06d}_{policy}';row={'source_index':i,'policy':policy,'trajectory_id':str(i),'status':'complete','failure':None}
                write(directory/(stem+'.json'),row);(directory/(stem+'.npz')).write_text('synthetic trace')
                records.append({**row,'record_file':stem+'.json','record_sha256':q.sha(directory/(stem+'.json')),
                    'trace_file':stem+'.npz','trace_sha256':q.sha(directory/(stem+'.npz'))})
        result={'state':'complete','objective':objective,'seed':seed,'checkpoint_sha256':q.sha(checkpoint),'protocol_sha256':ph,'records':records}
        write(directory/'result.json',result);write(directory/'status.json',{'state':'complete','result_sha256':q.sha(directory/'result.json')})
        entries.append({'name':name,'state':'complete','returncode':0})
    write(q.NATIVE/'queue_status.json',{'state':'complete','jobs':entries})
    calls=[]
    def recover(*args):calls.append(args[1:4]);return args[-1]
    return SimpleNamespace(full=SimpleNamespace(recover_record=recover)),calls


def test_all_native810_rows_and_traces_verified():
    ev,calls=prior_fixture();identity,pins=q.prior_complete(ev)
    assert len(calls)==810 and len(pins)==1639
    assert [row['name'] for row in identity['native_jobs']]==list(q.NAMES)


@pytest.mark.parametrize('kind',['stale_lock','partial','wrong_identity','trace_hash','duplicate','nonzero','boolean_seed'])
def test_native_prerequisite_refusals_preserve_files(kind):
    ev,_=prior_fixture();directory=q.NATIVE/q.NAMES[0]
    if kind=='stale_lock':(directory/'run.lock').write_text('stale')
    elif kind=='trace_hash':next(directory.glob('*.npz')).write_text('changed')
    else:
        path=q.NATIVE/'queue_status.json' if kind in ('partial','nonzero') else directory/'result.json'
        value=q.read(path)
        if kind=='partial':value['state']='running'
        elif kind=='nonzero':value['jobs'][0]['returncode']=1
        elif kind=='wrong_identity':value['objective']='nll'
        elif kind=='boolean_seed':value['seed']=False
        else:value['records'][0]=value['records'][1]
        write(path,value)
        if path.name=='result.json':write(directory/'status.json',{'state':'complete','result_sha256':q.sha(path)})
    with pytest.raises(RuntimeError):q.prior_complete(ev)
    assert not q.OUTPUT.exists()


def post_fixture(monkeypatch):
    expected=[{'split':'valid','source_index':i//8,'target_frame':7+i%8,'trajectory_id':f'v{i//8}'} for i in range(128)]
    expected += [{'split':'test','source_index':i,'target_frame':7+j,'trajectory_id':f't{i}'} for i in range(3,30) for j in range(11)]
    directory=q.OUTPUT/'faithful_seed0';directory.mkdir(parents=True)
    protocol_file=q.ROOT/'protocol.md';protocol_file.write_text('synthetic reviewed protocol')
    model={'checkpoint_sha256':'a'*64,'expected':expected,'saved':{},'test_source':{},'input_pins':{}}
    prepared={'models':{('faithful',0):model}}
    protocol={'objective':'faithful','seed':0,'scope':'synthetic','checkpoint_sha256':'a'*64,'cases':list(q.CASES),'random_draw_ids':list(range(8)),
        'expected_frames':expected,'saved_test_inputs':{},'source_tfrecord':{},'input_files_sha256':{},'envelope_protocol_sha256':q.sha(protocol_file),
        'threads':2,'runtime':{'device':'mps','radius_backend':'scipy_host','mps_fallback_environment':'0'}}
    write(directory/'protocol.json',protocol);ph=q.sha(directory/'protocol.json');records=[]
    stem=lambda item:f"{item['split']}_{item['source_index']:06d}_{item['target_frame']:04d}"
    for item in expected:
        name=stem(item);(directory/(name+'.npz')).write_text('synthetic array')
        row={**item,'objective':'faithful','seed':0,'checkpoint_sha256':'a'*64,'protocol_sha256':ph,'status':'complete','failure':None,
            'cases':{case:{'status':'complete','failure':None,'metrics':{'mse':1.}} for case in q.CASES},
            'array_file':name+'.npz','array_sha256':q.sha(directory/(name+'.npz'))}
        write(directory/(name+'.json'),row)
        records.append({**{key:row[key] for key in ('split','source_index','target_frame','trajectory_id','status','failure','array_file','array_sha256')},
            'record_file':name+'.json','record_sha256':q.sha(directory/(name+'.json'))})
    result={'state':'complete','objective':'faithful','seed':0,'scope':'synthetic','checkpoint_sha256':'a'*64,'protocol_sha256':ph,'required_frames':425,
        'complete_frames':425,'failed_frames':0,'records':records};write(directory/'result.json',result)
    write(directory/'status.json',{'state':'complete','objective':'faithful','seed':0,'committed_frames':425,'result_sha256':q.sha(directory/'result.json')})
    recovered={row['record_file']:row for row in records}
    ev=SimpleNamespace(SCOPE='synthetic',PROTOCOL=protocol_file,bridge=SimpleNamespace(stem=stem,recover=lambda directory,item,digest:recovered[stem(item)+'.json']))
    monkeypatch.setattr(q,'science',lambda:ev)
    return directory,prepared,recovered


def test_postjob_complete425_and_raw853_hashes(monkeypatch):
    directory,prepared,_=post_fixture(monkeypatch)
    result=q.verify_job(directory,'faithful',0,prepared)
    assert result['records']==425 and result['case_slots']==5525 and len(result['files_sha256'])==853


def test_postjob_valid_scientific_failure_is_retained_not_integrity_error(monkeypatch):
    directory,prepared,recovered=post_fixture(monkeypatch)
    record=next(iter(recovered.values()));path=directory/record['record_file'];row=q.read(path)
    failure={'category':'nonfinite_prediction','phase':'forward'}
    row['cases']['dense']={'status':'failed','failure':failure,'metrics':None}
    row.update(status='failed',failure={'category':'case_failures','cases':[{'case':'dense','failure':failure}]})
    write(path,row);record.update(status=row['status'],failure=row['failure'],record_sha256=q.sha(path))
    result=q.read(directory/'result.json');result['records'][0]=record;result.update(complete_frames=424,failed_frames=1)
    write(directory/'result.json',result)
    status=q.read(directory/'status.json');status['result_sha256']=q.sha(directory/'result.json');write(directory/'status.json',status)
    verified=q.verify_job(directory,'faithful',0,prepared)
    assert verified['passed'] is True and verified['records']==425 and verified['failed_case_count']==1
    assert verified['scientific_failures'][0]['failure']==failure


@pytest.mark.parametrize('kind',['array_hash','wrong_identity','missing_case','bad_count','source_change','temp','parity'])
def test_postjob_refusal(kind,monkeypatch):
    directory,prepared,recovered=post_fixture(monkeypatch)
    record=next(iter(recovered.values()));path=directory/record['record_file'];row=q.read(path)
    if kind=='array_hash':(directory/record['array_file']).write_text('changed')
    elif kind=='wrong_identity':row['objective']='nll'
    elif kind=='missing_case':row['cases'].pop('base')
    elif kind=='bad_count':
        status=q.read(directory/'status.json');status['committed_frames']=424;write(directory/'status.json',status)
    elif kind=='source_change':q.science().PROTOCOL.write_text('changed')
    elif kind=='temp':(directory/'orphan.npz.tmp').write_text('preserve')
    elif kind=='parity':row['failure']={'category':'native_parity_failure'}
    if kind in ('wrong_identity','missing_case','parity'):
        write(path,row);record['record_sha256']=q.sha(path)
        result=q.read(directory/'result.json');result['records'][0]=record;write(directory/'result.json',result)
        status=q.read(directory/'status.json');status['result_sha256']=q.sha(directory/'result.json');write(directory/'status.json',status)
    with pytest.raises(RuntimeError):q.verify_job(directory,'faithful',0,prepared)


def launch_fixture(monkeypatch,failed=False,slow=False,lock_failure=False,interrupt=False,nonzero=False):
    write(q.FREEZE,{});write(q.RELEASE,{})
    prepared={'immutable':{},'release':{},'native_identity':{},'planning':{}}
    monkeypatch.setattr(q,'verify_release',lambda *a,**k:{})
    monkeypatch.setattr(q,'verify_pins',lambda *a:None)
    monkeypatch.setattr(q,'no_locks',lambda *a:None)
    verify_calls=[]
    def verify(*args):verify_calls.append(args);return {'passed':True,'failed_case_count':int(failed)}
    monkeypatch.setattr(q,'verify_job',verify)
    class Lock:
        def __init__(self,*a):pass
        def __enter__(self):
            if lock_failure:
                write(q.OUTPUT/'queue_status.json',{'state':'other_owner'})
                raise RuntimeError('another owner')
        def __exit__(self,*a):pass
    ev=SimpleNamespace(full=SimpleNamespace(RunLock=Lock,atomic_json=write))
    monkeypatch.setattr(q,'science',lambda:ev)
    processes=[]
    class Process:
        def __init__(self,command,**kwargs):
            self.command=command;self.kwargs=kwargs;self.pid=100+len(processes);self.code=None;self.terminated=False;processes.append(self)
        def wait(self,timeout=None):
            if interrupt and self.command[0]!='caffeinate':raise KeyboardInterrupt()
            self.code=1 if nonzero and self.command[0]!='caffeinate' else 0;return self.code
        def poll(self):return self.code
        def terminate(self):self.terminated=True;self.code=0
    monkeypatch.setattr(q.subprocess,'Popen',Process)
    ticks=iter(range(0,10000,400 if slow else 1))
    monkeypatch.setattr(q.time,'perf_counter',lambda:next(ticks))
    return prepared,processes,verify_calls


def test_launch_exact_children_fallback_and_owned_keepawake(monkeypatch):
    prepared,processes,calls=launch_fixture(monkeypatch)
    assert q.launch(prepared)==0 and len(calls)==6 and len(processes)==7
    assert processes[0].command==['caffeinate','-i','-w',str(q.os.getpid())] and processes[0].terminated
    for process in processes[1:]:
        assert process.kwargs['env']['PYTORCH_ENABLE_MPS_FALLBACK']=='0' and process.kwargs['cwd']==q.REPO and not process.terminated
    status=q.read(q.OUTPUT/'queue_status.json')
    assert status['state']=='complete' and status['scientific_summary_and_audit_pending'] is True


def test_complete_scientific_failed_cases_continue_fixed_cohort_without_retry(monkeypatch):
    prepared,processes,calls=launch_fixture(monkeypatch,failed=True)
    assert q.launch(prepared)==0 and len(calls)==6 and len(processes)==7
    status=q.read(q.OUTPUT/'queue_status.json')
    assert status['state']=='complete'
    assert all(job['verification']['failed_case_count']==1 for job in status['jobs'])
    assert len({tuple(process.command) for process in processes[1:]})==6


def test_admission_setup_and_execution_budget_scopes_remain_separate(monkeypatch):
    prepared,processes,calls=launch_fixture(monkeypatch)
    prepared.update(admission_started_utc=(q.now()-timedelta(seconds=20)).isoformat(),admission_seconds=20.)
    baseline=q.now();ticks=[0]
    def advancing():
        ticks[0]+=1
        return baseline+timedelta(milliseconds=ticks[0])
    monkeypatch.setattr(q,'now',advancing)
    assert q.launch(prepared)==0
    status=q.read(q.OUTPUT/'queue_status.json')
    assert status['admission_seconds']==20 and status['initialization_seconds_before_first_evaluator']>0
    assert datetime.fromisoformat(status['execution_budget_started_utc'])>datetime.fromisoformat(status['initialization_started_utc'])
    assert status['supervisor_seconds_since_initialization']>=status['final_analysis_reserve_check']['elapsed_execution_budget_seconds']
    assert status['analysis_reserve_available'] is True and status['inference_complete'] is True
    assert 'excluded and separately measured' in status['budget_scope']
    assert 'forecast_whole_study_seconds' not in status['planning']


@pytest.mark.parametrize('kind',['nonzero','slow','existing','competing_owner','interrupt'])
def test_launch_stops_preserves_and_never_kills_or_retries(kind,monkeypatch):
    prepared,processes,calls=launch_fixture(monkeypatch,nonzero=kind=='nonzero',slow=kind=='slow',lock_failure=kind=='competing_owner',interrupt=kind=='interrupt')
    if kind=='existing':
        q.OUTPUT.mkdir();(q.OUTPUT/'preserve').write_text('existing')
        with pytest.raises(RuntimeError):q.launch(prepared)
        assert not processes and (q.OUTPUT/'preserve').read_text()=='existing';return
    if kind=='competing_owner':
        with pytest.raises(RuntimeError):q.launch(prepared)
        assert q.read(q.OUTPUT/'queue_status.json')=={'state':'other_owner'} and not processes;return
    if kind=='interrupt':
        with pytest.raises(KeyboardInterrupt):q.launch(prepared)
        assert q.read(q.OUTPUT/'queue_status.json')['live_child_pid']==processes[1].pid
        assert not any(p.terminated for p in processes);return
    assert q.launch(prepared)==(1 if kind=='nonzero' else 2)
    assert len(processes)==2 and len(calls)==1 and processes[0].terminated and not processes[1].terminated
    status=q.read(q.OUTPUT/'queue_status.json')
    assert status['state']==('error' if kind=='nonzero' else 'gate_stopped')
    assert all(job['state']=='unstarted' for job in status['jobs'][1:])


def test_check_requires_explicit_mode_and_never_creates_output(monkeypatch,capsys):
    monkeypatch.setattr(q,'preflight',lambda:{'planning':{'forecast':'synthetic'}})
    assert q.main(['--check'])==0 and not q.OUTPUT.exists()
    assert json.loads(capsys.readouterr().out)['outputtree_written'] is False
    with pytest.raises(SystemExit):q.main([])
    with pytest.raises(SystemExit):q.main(['--check','--launch'])


@pytest.mark.parametrize('text',['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'])
def test_strict_json(text):
    path=q.ROOT/'bad.json';path.write_text(text)
    with pytest.raises(RuntimeError):q.read(path)
