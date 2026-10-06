"""Synthetic operational tests only: no network, real clock issuance, arrays or workers."""
import contextlib,copy,datetime,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import root_operator as O
import remote_probe as P

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')

def closure_fixture(role):
    cfg=P.EVALUATIONS[role];release=O.read(O.K/'candidates'/(role+'.original_evaluation_release.json'));Q=cfg['queue_root']
    schema='adaptgns_sand_evaluation_gpu_scoped_v1';metadata={Q+'/coverage_ledger.json':'a'*64,Q+'/gpu_observations.json':'b'*64}
    status=dict(schema=schema,state='allocation_finished',coverage_ledger_sha256='a'*64,unreaped_owned_children=[])
    ledger=dict(schema=schema,dataset='Sand',unreaped_owned_children=[],stages=[])
    process=dict(schema=schema,unreaped_owned_children=[],gpu_observations_sha256='b'*64,all_children=[],streams=[],abort_reason=None,gpu_scope=copy.deepcopy(release['gpu_scope']))
    names=('full_rollout_test','same_state_valid','same_state_test','clean_validation')
    for i,stream in enumerate(release['streams']):
        outcomes=[]
        for j,command in enumerate(stream['commands']):
            identity=copy.deepcopy(cfg['initial_children'][i])
            if j:identity.update(pid=900000+i*10+j,start_ticks=800000+i*10+j,argv=command)
            output=command[command.index('--output-dir')+1]
            outcome=dict(stage=names[j],state='exited',exit_code=0,pid=identity['pid'],output_dir=output,signals=[],quota_expired=False)
            child=dict(identity=identity,command=command,stage=names[j],output_dir=output,signals=[],quota_expired=False)
            process['all_children'].append(child);outcomes.append(outcome);ledger['stages'].append(dict(stream=stream['id'],stage=names[j],outcome=copy.deepcopy(outcome)))
        process['streams'].append(dict(id=stream['id'],outcomes=outcomes))
    return [role,release,copy.deepcopy(release),status,ledger,process,{},metadata]

class ClosureTests(unittest.TestCase):
    def setUp(self):self.args=closure_fixture('A')
    def run_fixture(self):return P.validate_evaluation_closure(*self.args)
    def rejects(self):
        with self.assertRaises((ValueError,KeyError)):self.run_fixture()
    def test_A_full_16(self):self.assertEqual(len(self.run_fixture()['registered_children']),16)
    def test_B_full_8(self):self.assertEqual(len(P.validate_evaluation_closure(*closure_fixture('B'))['registered_children']),8)
    def test_quota_failure_preserved(self):
        child=self.args[5]['all_children'][1];out=self.args[5]['streams'][0]['outcomes'][1]
        child.update(quota_expired=True,signals=[{'signal':'SIGINT'}]);out.update(state='quota_expired',exit_code=-2,quota_expired=True,signals=child['signals'])
        self.args[4]['stages'][1]['outcome']=copy.deepcopy(out)
        self.assertEqual(self.run_fixture()['original_stage_outcomes'][1]['outcome']['exit_code'],-2)
    def test_never_started_preserved(self):
        self.args[5]['all_children'].pop();self.args[5]['streams'][-1]['outcomes'].pop();self.args[4]['stages'][-1]['outcome']={'stage':'clean_validation','state':'never_started'}
        self.assertEqual(len(self.run_fixture()['original_stage_outcomes']),16)
    def test_abort_preserved(self):
        self.args[3]['state']='stopped_requires_review';self.args[5]['abort_reason']='synthetic worker failure'
        self.assertEqual(self.run_fixture()['original_abort_reason'],'synthetic worker failure')
    def test_snapshot_changed(self):self.args[2]['dataset']='Water';self.rejects()
    def test_ledger_pin_changed(self):self.args[3]['coverage_ledger_sha256']='c'*64;self.rejects()
    def test_unreaped(self):self.args[5]['unreaped_owned_children']=[1];self.rejects()
    def test_missing_gpu_after_children(self):self.args[6]=None;self.rejects()
    def test_gpu_scope_changed(self):self.args[5]['gpu_scope']['owned_indices']=[2];self.rejects()
    def test_missing_stage(self):self.args[4]['stages'].pop();self.rejects()
    def test_duplicate_stage(self):self.args[4]['stages'][-1]=copy.deepcopy(self.args[4]['stages'][0]);self.rejects()
    def test_extra_stream(self):self.args[5]['streams'].append(copy.deepcopy(self.args[5]['streams'][0]));self.rejects()
    def test_missing_registered_child(self):self.args[5]['all_children'].pop();self.rejects()
    def test_missing_outcome(self):self.args[5]['streams'][0]['outcomes'].pop();self.rejects()
    def test_startup_identity_changed(self):self.args[5]['all_children'][0]['identity']['start_ticks']+=1;self.rejects()
    def test_parent_changed(self):self.args[5]['all_children'][1]['identity']['ppid']+=1;self.rejects()
    def test_pid_bool(self):self.args[5]['all_children'][1]['identity']['pid']=True;self.rejects()
    def test_outcome_exit_bool(self):
        self.args[5]['streams'][0]['outcomes'][0]['exit_code']=False;self.args[4]['stages'][0]['outcome']['exit_code']=False;self.rejects()
    def test_signal_history_changed(self):self.args[5]['all_children'][0]['signals']=[{'signal':'SIGTERM'}];self.rejects()
    def test_quota_history_changed(self):self.args[5]['all_children'][0]['quota_expired']=True;self.rejects()
    def test_duplicate_native_pid(self):self.args[5]['all_children'][1]['identity']['pid']=self.args[5]['all_children'][0]['identity']['pid'];self.rejects()

class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.state=Path(self.temp.name).resolve();self.eval=self.state/'original_evaluation';self.eval.mkdir()
        self.now=datetime.datetime(2026,10,6,23,tzinfo=datetime.timezone.utc)
        for role,cfg in O.bindings()['roles'].items():
            ext=self.eval/(role+'.evaluation.external.json');write(ext,{'local_transport_reaped':True,'observed_utc':(self.now-datetime.timedelta(seconds=5)).isoformat()})
            ev=self.state/(role+'.synthetic_tool.json');write(ev,{'synthetic':True,'session':cfg['session_id']})
            record=dict(recorded_by='root',original_tool_session=cfg['session_id'],original_tool_exit_code=0,observed_utc=(self.now-datetime.timedelta(seconds=4)).isoformat(),evidence_path=str(ev),evidence_sha256=O.sha(ev),original_transport_sha256=O.sha(ext))
            rp=self.state/(role+'.evaluation.root_original_external_exit.json');write(rp,record)
            c=O.read(O.K/'candidates'/('Sand.'+role+'.evaluation_closure.candidate.json'));c.update(issued_by='root',status='original_exit_and_native_absence_verified',original_tool_session=cfg['session_id'],original_exit_code=0,all_children_reaped=True,complete_registered_child_ledger_checked=True,native_absent={str(cfg['owner']['pid']):True},checked_utc=(self.now-datetime.timedelta(seconds=3)).isoformat(),original_tool_exit_receipt_sha256=O.sha(rp))
            write(self.state/(role+'.evaluation_closure.json'),c)
        for name,value in [('EVAL_STATE',self.eval),('now',lambda:self.now)]:
            p=patch.object(O,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(O.time,'monotonic',return_value=1000);p.start();self.addCleanup(p.stop)
    def begin(self):
        with contextlib.redirect_stdout(io.StringIO()):O.begin(self.state)
    def test_whole3600(self):
        self.begin();v,stop,mono=O.phase(self.state);self.assertEqual((stop-self.now).total_seconds(),3600);self.assertEqual(mono,4600)
    def test_single_original_phase(self):
        self.begin()
        with self.assertRaises(FileExistsError):self.begin()
    def test_nonzero_evaluation_exit_remains_stopped_evidence(self):
        rp=self.state/'A.evaluation.root_original_external_exit.json';record=O.read(rp);record['original_tool_exit_code']=1;write(rp,record)
        path=self.state/'A.evaluation_closure.json';c=O.read(path);c.update(original_exit_code=1,original_tool_exit_receipt_sha256=O.sha(rp));write(path,c);self.begin()
    def test_incomplete_closure_rejected(self):
        path=self.state/'B.evaluation_closure.json';c=O.read(path);c['all_children_reaped']=False;write(path,c)
        with self.assertRaises(ValueError):self.begin()
    def test_empty_native_closure_rejected(self):
        path=self.state/'B.evaluation_closure.json';c=O.read(path);c['native_absent']={};write(path,c)
        with self.assertRaises(ValueError):self.begin()
    def test_transport_changed_rejected(self):
        write(self.eval/'A.evaluation.external.json',{'changed':True})
        with self.assertRaises(ValueError):self.begin()
    def test_full_hour_must_fit_global_stop(self):
        self.now=datetime.datetime(2026,10,7,3,1,tzinfo=datetime.timezone.utc)
        with self.assertRaises(ValueError):self.begin()
    def test_no_budget_after_original_stop(self):
        self.begin();self.now+=datetime.timedelta(seconds=3600)
        with patch.object(O.time,'monotonic',return_value=4600),self.assertRaises(ValueError):O.budget(self.state)
    def test_clock_drift_rejected(self):
        self.begin();self.now+=datetime.timedelta(seconds=10)
        with self.assertRaises(ValueError):O.budget(self.state)
    def test_closure_transport_rechecked_before_native_probe(self):
        write(self.eval/'A.evaluation.external.json',{'local_transport_reaped':True,'observed_utc':self.now.isoformat()})
        with patch.object(O,'probe') as native,self.assertRaises(ValueError):O.close_evaluation(self.state,'A',(self.now+datetime.timedelta(seconds=50)).isoformat())
        native.assert_not_called()

class OriginalExitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.state=Path(self.temp.name).resolve();self.op='sand_collect_A'
        self.now=datetime.datetime(2026,10,6,23,tzinfo=datetime.timezone.utc);iso=lambda s:(self.now+datetime.timedelta(seconds=s)).isoformat()
        self.ext=dict(exit_code=0,failure=None,local_transport_reaped=True,local_transport_timeout=False,signals_to_own_local_group=[],started_utc=iso(-20),observed_utc=iso(-5));self.evidence=self.state/'synthetic_tool_exit.json';write(self.evidence,{'exit_code':0})
        self.record=dict(recorded_by='root',original_tool_session=123456,original_tool_exit_code=0,evidence_path=str(self.evidence),evidence_sha256=O.sha(self.evidence),observed_utc=iso(-4))
        for name,value in [('now',lambda:self.now),('budget',lambda s:({'original_analysis_started_utc':iso(-30)},self.now+datetime.timedelta(seconds=100),0,100))]:
            p=patch.object(O,name,value);p.start();self.addCleanup(p.stop)
        self.save()
    def save(self):
        path=self.state/(self.op+'.external.json');write(path,self.ext);self.record['original_transport_sha256']=O.sha(path);write(self.state/(self.op+'.root_original_external_exit.json'),self.record)
    def test_original_exit_accepted(self):self.assertEqual(O.original_exit(self.state,self.op)['original_tool_session'],123456)
    def test_failure_preserved_not_accepted(self):
        self.ext['failure']='synthetic transport failure';self.save()
        with self.assertRaises(ValueError):O.original_exit(self.state,self.op)
    def test_original_nonzero_rejected_for_product(self):
        self.record['original_tool_exit_code']=1;self.save()
        with self.assertRaises(ValueError):O.original_exit(self.state,self.op)
    def test_false_not_exit_zero(self):
        self.record['original_tool_exit_code']=False;self.save()
        with self.assertRaises(ValueError):O.original_exit(self.state,self.op)
    def test_changed_transport_rejected(self):
        self.ext['observed_utc']=self.now.isoformat();write(self.state/(self.op+'.external.json'),self.ext)
        with self.assertRaises(ValueError):O.original_exit(self.state,self.op)
    def test_changed_tool_evidence_rejected(self):
        write(self.evidence,{'changed':True})
        with self.assertRaises(ValueError):O.original_exit(self.state,self.op)
    def test_timeout_rejected(self):
        self.ext['local_transport_timeout']=True;self.save()
        with self.assertRaises(ValueError):O.original_exit(self.state,self.op)
    def test_unreaped_rejected(self):
        self.ext['local_transport_reaped']=False;self.save()
        with self.assertRaises(ValueError):O.original_exit(self.state,self.op)

class ParentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()/'analysis';self.root.mkdir();self.p=patch.object(P,'A',str(self.root));self.p.start();self.addCleanup(self.p.stop)
    def test_create_only_parent(self):
        result=P.ensure_empty_owner_parent(lambda:None);self.assertTrue(result['created']);self.assertEqual(list((self.root/'owners').iterdir()),[])
    def test_empty_existing_parent(self):
        (self.root/'owners').mkdir();self.assertFalse(P.ensure_empty_owner_parent(lambda:None)['created'])
    def test_existing_owner_rejected(self):
        (self.root/'owners').mkdir();(self.root/'owners'/'old').mkdir()
        with self.assertRaises(ValueError):P.ensure_empty_owner_parent(lambda:None)
    def test_symlink_parent_rejected(self):
        target=self.root/'elsewhere';target.mkdir();(self.root/'owners').symlink_to(target)
        with self.assertRaises(ValueError):P.ensure_empty_owner_parent(lambda:None)

class PublicationClockTests(unittest.TestCase):
    def setUp(self):
        self.t=datetime.datetime(2026,10,6,23,tzinfo=datetime.timezone.utc)
        self.stamp=lambda n:(self.t+datetime.timedelta(seconds=n)).isoformat()
        self.release={'invocation_origin_utc':self.stamp(0),'publication_deadline_utc':self.stamp(100),'clock_sample':{'error_bound_seconds':5}}
        self.external={'observed_utc':self.stamp(20)}
    def test_exact_clock(self):O.publication_order(self.release,{'publication_utc':self.stamp(20)},self.external)
    def test_host_leading_within_bound(self):O.publication_order(self.release,{'publication_utc':self.stamp(24)},self.external)
    def test_exact_five_second_boundary(self):O.publication_order(self.release,{'publication_utc':self.stamp(25)},self.external)
    def test_beyond_five_seconds_rejected(self):
        with self.assertRaises(ValueError):O.publication_order(self.release,{'publication_utc':self.stamp(25.000001)},self.external)
    def test_no_skew_extension_of_same_host_deadline(self):
        with self.assertRaises(ValueError):O.publication_order(self.release,{'publication_utc':self.stamp(100)},{'observed_utc':self.stamp(101)})
    def test_no_skew_before_same_host_origin(self):
        with self.assertRaises(ValueError):O.publication_order(self.release,{'publication_utc':self.stamp(-.001)},self.external)
    def test_no_wider_clock_tolerance(self):
        self.release['clock_sample']['error_bound_seconds']=6
        with self.assertRaises(ValueError):O.publication_order(self.release,{'publication_utc':self.stamp(20)},self.external)

class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.state=Path(self.temp.name).resolve()
        self.now=datetime.datetime(2026,10,6,23,tzinfo=datetime.timezone.utc);self.stop=self.now+datetime.timedelta(seconds=1000)
        write(self.state/'analysis_phase.json',{'synthetic_original_phase':True})
        self.receipts={};self.captures=[];self.stages=[];self.launches=[];self.mutate_tree=False;self.mutate_metadata=False
        for role in ('A','B'):
            Q=O.bindings()['roles'][role]['queue_root'];inv={'entries':[['release_snapshot.json','file']],'files':{'release_snapshot.json':{'sha256':'1'*64,'bytes':10}}}
            write(self.state/(role+'.evaluation_closure.json'),{'closure_metadata_sha256':{Q+'/release_snapshot.json':'1'*64}})
            write(self.state/(role+'.stage_runtime.json'),{'timeout_version':'timeout synthetic'})
            write(self.state/(role+'.runtime_ready.json'),{'synthetic_ready':role})
            write(self.state/('sand_collect_'+role+'.snapshot.json'),{'queue_inventories':{Q:inv}})
        for index,op in enumerate(O.OPS):
            path=self.state/(op+'.synthetic_product.json');write(path,{'synthetic':op});receipt={'product_sha256':O.sha(path),'local_product':str(path),'semantic_review':{'eligible_for_aggregation':True}}
            self.receipts[op]=receipt;write(self.state/(op+'.review.json'),receipt)
        def snapshot(state,op,pins,tag,with_documents=False):
            s=O.spec(op);trees={};inputs=dict(pins)
            for Q in s['protected_tree_roots']:
                inv={'entries':[['release_snapshot.json','file']],'files':{'release_snapshot.json':{'sha256':'1'*64,'bytes':10}}}
                if self.mutate_tree and tag.endswith('.snapshot'):inv['entries'].append(['added','directory'])
                trees[Q]=inv;inputs[Q+'/release_snapshot.json']='2'*64 if self.mutate_metadata else '1'*64
            result={'queue_inventories':trees,'inputs_sha256':inputs,'protected_tree_entries':{q:v['entries'] for q,v in trees.items()}}
            # Fixture predecessor snapshots are distinct from this invocation's captures.
            if state.joinpath(tag+'.json').exists():state.joinpath(tag+'.json').unlink()
            O.put(state/(tag+'.json'),result);self.captures.append(tag);return result
        def stage(state,role,files,tag):self.stages.append((role,files,tag))
        def launch(state,argv,tag,stop,mono):self.launches.append((argv,tag,stop,mono))
        clock={'host_utc':self.now.isoformat(),'host_monotonic_seconds':5000,'host_boot_id':'synthetic','root_reference_utc':self.now.isoformat(),'source':'root_fresh_tool_and_host_clock_evidence','error_bound_seconds':5}
        replacements={'now':lambda:self.now,'runtime_ready':lambda *a:None,'require_predecessors':lambda *a:None,'accepted':lambda s,op:(self.receipts[op],{'synthetic':op}),'snapshot':snapshot,'stage_controls':stage,'fresh_clock':lambda *a:(copy.deepcopy(clock),self.stop),'budget':lambda *a:({},self.stop,6000,1000),'bounded_local':launch}
        for name,value in replacements.items():
            p=patch.object(O,name,value);p.start();self.addCleanup(p.stop)
    def check_operation(self,op):
        O.issue_and_run(self.state,op);r=O.read(self.state/(op+'.cpu_release.json'));s=O.spec(op)
        expected=[r['inputs_sha256'][x['sha256_of_input']] if isinstance(x,dict) else x for x in s['argv']]
        self.assertEqual(r['command'],expected);self.assertEqual(r['host_role'],s['host_role']);self.assertEqual(r['whole_invocation_seconds'],1000)
        self.assertEqual(r['publication_deadline_utc'],self.stop.isoformat());self.assertEqual(r['hard_deadline_monotonic_ns'],5995*10**9)
        self.assertEqual(r['analysis_phase']['sha256'],O.sha(self.state/'analysis_phase.json'));self.assertFalse(r['automatic_retry']);self.assertFalse(r['new_or_restarted_clock_granted'])
        self.assertEqual(len(self.launches),1);self.assertEqual(self.launches[0][0][4],O.ALIASES[s['host_role']])
        if op.startswith('sand_collect_'):self.assertEqual(self.captures,[op+'.inventory',op+'.snapshot'])
        self.assertEqual(self.stages[-1][2],op+'.stage_cpu')
    def test_collection_tree_changed_refuses_launch(self):
        self.mutate_tree=True
        with self.assertRaises(ValueError):O.issue_and_run(self.state,'sand_collect_A')
        self.assertEqual(self.launches,[])
    def test_closure_metadata_changed_refuses_launch(self):
        self.mutate_metadata=True
        with self.assertRaises(ValueError):O.issue_and_run(self.state,'sand_saved_A')
        self.assertEqual(self.launches,[])
    def test_preserved_release_cannot_be_reissued(self):
        self.check_operation('sand_collect_A')
        with self.assertRaises(FileExistsError):O.issue_and_run(self.state,'sand_collect_A')
        self.assertEqual(len(self.launches),1)

for _op in O.OPS:
    setattr(OrchestrationTests,'test_'+_op,lambda self,op=_op:self.check_operation(op))

if __name__=='__main__':unittest.main(verbosity=2)
