"""Synthetic metadata only. No actual hosts, queues, GPU commands or science."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys,tempfile,unittest
from unittest.mock import patch
import observe_metadata as M
import root_observe as T
S=Path(__file__).resolve().parent.parent/'sand_stopped_analysis_operator_UNADMITTED_code_audit_v1'
sys.path.insert(0,str(S))
from test_operator import closure_fixture

def raw(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(value).hexdigest()
class Fake:
    def __init__(self,role='A'):
        self.c=copy.deepcopy(M.CONTRACTS[role]);self.cfg=self.c['evaluation'];self.Q=self.cfg['queue_root'];self.allowed_pids=set();self.stage_paths=set();self.reads=[];self.natives=[];self.listed=[];self.count={};self.host_value=self.cfg['hostname'];self.boot_value=self.cfg['boot_id'];self.mutation=None;self.present=None;self.changed_pid=None;self.gpu_busy=False;self.availability_changes=False
        args=closure_fixture(role);self.release=args[1];self.status=args[3];self.ledger=args[4];self.process=args[5];self.refresh()
    def refresh(self):
        gpu=raw({});self.process['gpu_observations_sha256']=digest(gpu);ledger=raw(self.ledger);self.status['coverage_ledger_sha256']=digest(ledger)
        self.data={self.cfg['release_path']:raw(self.release),self.Q+'/release_snapshot.json':raw(self.release),self.Q+'/queue_status.json':raw(self.status),self.Q+'/coverage_ledger.json':ledger,self.Q+'/process_outcomes.json':raw(self.process),self.Q+'/gpu_observations.json':gpu}
        # Exact original release byte pin in production remains immutable. This
        # fixture preserves its real byte representation from the frozen copy.
        exact=(S/'candidates'/(self.c['role']+'.original_evaluation_release.json')).read_bytes();self.data[self.cfg['release_path']]=exact;self.data[self.Q+'/release_snapshot.json']=exact
    def check(self):pass
    def clocks(self):return {'utc':'2026-10-06T22:20:00+00:00','monotonic_seconds':1000}
    def host(self):return self.host_value
    def boot(self):return self.boot_value
    def read_raw(self,path,cap):
        self.reads.append(path);self.count[path]=self.count.get(path,0)+1
        if path not in self.data:raise FileNotFoundError(path)
        if path==self.mutation and self.count[path]>1:return self.data[path]+b' '
        return self.data[path]
    def native(self,pid):
        self.natives.append(pid)
        if pid==self.present or pid==self.changed_pid and self.natives.count(pid)>1:return {'status':'present','stat':{'pid':pid,'start_ticks':123}}
        return {'status':'absent'}
    def availability(self,path):
        self.listed.append(path);entries=[{'name':'trajectory_000000_static_native.npz','kind':'file','bytes':100},{'name':'status.json','kind':'file','bytes':42}]
        if self.availability_changes and self.listed.count(path)>1:entries.append({'name':'new.json','kind':'file','bytes':1})
        return {'state':'present','entries':entries,'scientific_file_contents_read':False}
    def gpu(self):return {'gpu_uuids':self.cfg['gpu_uuids'],'assigned_gpu_indices':self.cfg['owned_gpu_indices'],'assigned_devices_empty':not self.gpu_busy,'observed_compute_processes':[],'unassigned_devices':'observe_without_control'}
class ObserverTests(unittest.TestCase):
    def setUp(self):self.f=Fake()
    def run_fixture(self):return M.observe(self.f.c,self.f)
    def test_full_A_history(self):
        r=self.run_fixture();self.assertTrue(r['native_closure_observed']);self.assertEqual(len(r['registered_children']),16);self.assertEqual(len(r['availability_first']),16);self.assertFalse(r['original_tool_exit_observed']);self.assertIsNone(r['original_tool_exit_code'])
    def test_full_B_history(self):
        f=Fake('B');r=M.observe(f.c,f);self.assertTrue(r['native_closure_observed']);self.assertEqual(len(r['registered_children']),8);self.assertEqual(len(r['availability_first']),8)
    def test_no_result_or_array_content_reads(self):
        self.run_fixture();self.assertEqual(set(self.f.reads),set(self.f.data));self.assertFalse(any('/jobs/' in p for p in self.f.reads));self.assertTrue(self.f.listed)
    def test_full_historical_and_registered_native_union(self):
        r=self.run_fixture();expected=set(self.f.cfg['historical_pids'])|{self.f.cfg['owner']['pid']}|{int(p) for p in r['registered_children']};self.assertEqual(set(map(int,r['native_first'])),expected);self.assertTrue(all(self.f.natives.count(p)==2 for p in expected))
    def test_wrong_host_no_queue_or_native_reads(self):
        self.f.host_value='wrong';r=self.run_fixture();self.assertFalse(r['native_closure_observed']);self.assertEqual(self.f.reads,[]);self.assertEqual(self.f.natives,[])
    def test_wrong_boot_no_queue_reads(self):
        self.f.boot_value='wrong';self.assertFalse(self.run_fixture()['native_closure_observed']);self.assertEqual(self.f.reads,[])
    def test_missing_process_registry(self):
        del self.f.data[self.f.Q+'/process_outcomes.json'];r=self.run_fixture();self.assertFalse(r['native_closure_observed']);self.assertEqual(self.f.natives,[])
    def test_missing_initial_child(self):
        self.f.process['all_children'].pop(0);self.f.refresh();self.assertFalse(self.run_fixture()['native_closure_observed'])
    def test_unknown_child_command(self):
        self.f.process['all_children'][0]['command']=['unknown'];self.f.refresh();self.assertFalse(self.run_fixture()['native_closure_observed'])
    def test_nonzero_child_failure_preserved(self):
        out=self.f.process['streams'][0]['outcomes'][1];out['exit_code']=1;self.f.ledger['stages'][1]['outcome']=copy.deepcopy(out);self.f.refresh();r=self.run_fixture();self.assertTrue(r['native_closure_observed']);self.assertEqual(r['original_stage_outcomes'][1]['outcome']['exit_code'],1);self.assertFalse(r['remote_execution_success_established'])
    def test_unexecuted_stage_preserved(self):
        self.f.process['all_children'].pop();self.f.process['streams'][-1]['outcomes'].pop();self.f.ledger['stages'][-1]['outcome']={'stage':'clean_validation','state':'never_started'};self.f.refresh();r=self.run_fixture();self.assertTrue(r['native_closure_observed']);self.assertEqual(len(r['original_stage_outcomes']),16)
    def test_original_pid_still_present(self):self.f.present=self.f.cfg['owner']['pid'];self.assertFalse(self.run_fixture()['native_closure_observed'])
    def test_pid_appears_before_final_publication(self):self.f.changed_pid=self.f.cfg['owner']['pid'];self.assertFalse(self.run_fixture()['native_closure_observed'])
    def test_metadata_changed_rejected(self):self.f.mutation=self.f.Q+'/process_outcomes.json';self.assertFalse(self.run_fixture()['native_closure_observed'])
    def test_busy_assigned_gpu_rejected(self):self.f.gpu_busy=True;self.assertFalse(self.run_fixture()['native_closure_observed'])
    def test_availability_changes_are_explicit_not_numeric_verdict(self):
        self.f.availability_changes=True;r=self.run_fixture();self.assertFalse(r['stage_filename_metadata_unchanged']);self.assertTrue(r['availability_is_not_numeric_completeness']);self.assertFalse(r['evaluation_closure_or_analysis_admission'])
    def test_contract_cannot_change_session(self):
        self.f.c['original_session']=1
        with self.assertRaises(ValueError):self.run_fixture()
    def test_validator_exact_frozen_ast(self):
        def find(path):return next(ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_evaluation_closure')
        self.assertEqual(find(Path(M.__file__)),find(S/'remote_probe.py'))
    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError):M.strict(b'{"a":1,"a":2}')
    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ValueError):M.strict(b'{"a":NaN}')
class BoundaryTests(unittest.TestCase):
    def test_exact_loopback_proxy_only(self):
        self.assertEqual(T.proxy_environment('http://127.0.0.1:4321')['HTTPS_PROXY'],'http://127.0.0.1:4321')
        for url in ('http://localhost:4','https://127.0.0.1:4','http://u:p@127.0.0.1:4','http://127.0.0.1:4/','http://127.0.0.2:4'):
            with self.assertRaises(ValueError):T.proxy_environment(url)
    def test_remote_command_fixed_scope(self):
        argv=T.command('fake source','a'*64,'B');self.assertEqual(argv[-2],'aquamarine-toad-75.coder');self.assertIn('--signal=KILL 20s',argv[-1]);self.assertNotIn('--execute',argv[-1])
    def test_arbitrary_metadata_read_refused_before_open(self):
        rt=M.Runtime(M.CONTRACTS['A'])
        with patch.object(M.os,'open') as opened,self.assertRaises(ValueError):rt.read_raw('/tmp/scientific.npz',10)
        opened.assert_not_called()
    def test_arbitrary_stage_scan_refused(self):
        rt=M.Runtime(M.CONTRACTS['A'])
        with self.assertRaises(ValueError):rt.availability('/root/repos/other')
    def test_symlink_metadata_not_followed(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d).resolve();target=parent/'target';target.write_text('{}');link=parent/'metadata.json';link.symlink_to(target);rt=M.Runtime(M.CONTRACTS['A']);rt.metadata_paths={str(link)}
            with self.assertRaises(OSError):rt.read_raw(str(link),100)
    def test_existing_metadata_cap(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve()/'metadata.json';path.write_text('x'*100);rt=M.Runtime(M.CONTRACTS['A']);rt.metadata_paths={str(path)}
            with self.assertRaises(ValueError):rt.read_raw(str(path),10)
if __name__=='__main__':unittest.main(verbosity=2)
