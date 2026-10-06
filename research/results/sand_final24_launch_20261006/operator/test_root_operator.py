"""Synthetic root-control tests; no host clocks, probes, launches or data reads."""
from pathlib import Path
import base64,datetime,hashlib,importlib.util,json,subprocess,tempfile,unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('final24_root',Path(__file__).with_name('root_operator.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
helper_spec=importlib.util.spec_from_file_location('final24_helper',Path(__file__).with_name('remote_control.py'))
H=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(H)
T=datetime.datetime(2026,10,6,20,tzinfo=datetime.timezone.utc)
class FakeProcess:
 pid=12345
 def __init__(self,values):self.values=iter(values);self.returncode=None
 def communicate(self,**kwargs):
  value=next(self.values)
  if isinstance(value,BaseException):raise value
  self.returncode=value;return 'synthetic stdout','synthetic stderr'
 def poll(self):return self.returncode
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.addCleanup(patch.stopall);patch.object(M,'now',return_value=T).start();patch.object(M.time,'monotonic',return_value=1000.).start()
 def phase(self,seconds=600,start=T):
  value={'status':'approved_one_original_final24_control_phase','issued_by':'root','started_utc':start.isoformat(),'stop_utc':(start+datetime.timedelta(seconds=seconds)).isoformat(),'control_seconds':seconds,'evaluation_seconds':11760,'analysis_seconds':3600,'clock_restarted':False}
  M.put(self.root/'control_phase.json',value);M.put(self.root/'control_anchor.json',{'utc':start.isoformat(),'monotonic_seconds':1000.,'phase_sha256':M.sha(self.root/'control_phase.json')});return value
 def test_original600(self):self.phase();self.assertEqual(M.phase(self.root)[2],1600)
 def test_no_missing_phase_fallback(self):
  with self.assertRaises(FileNotFoundError):M.phase(self.root)
 def test_short_phase_rejected(self):
  self.phase(599)
  with self.assertRaises(ValueError):M.phase(self.root)
 def test_monotonic_reset_rejected(self):
  self.phase()
  with patch.object(M.time,'monotonic',return_value=999.):
   with self.assertRaises(ValueError):M.phase(self.root)
 def test_utc_jump_rejected(self):
  self.phase()
  with patch.object(M,'now',return_value=T+datetime.timedelta(seconds=8)):
   with self.assertRaises(ValueError):M.phase(self.root)
 def test_expired_control_rejected(self):
  self.phase()
  with patch.object(M,'now',return_value=T+datetime.timedelta(seconds=601)),patch.object(M.time,'monotonic',return_value=1601.):
   with self.assertRaises(ValueError):M.phase(self.root)
 def test_full_downstream_must_fit(self):
  start=datetime.datetime(2026,10,6,23,40,tzinfo=datetime.timezone.utc);self.phase(start=start)
  with patch.object(M,'now',return_value=start):
   with self.assertRaises(ValueError):M.phase(self.root)
 def test_phase_mutation_rejected(self):
  self.phase();v=M.read(self.root/'control_phase.json');v['unused']=True;(self.root/'control_phase.json').write_text(json.dumps(v))
  with self.assertRaises(ValueError):M.phase(self.root)
 def test_resolver_preserves_scientific_and_allocation_identity(self):
  control=self.phase();candidate={'status':'unadmitted','issued_by':None,'files_sha256':{'x.npy':'a'*64,'source.py':'b'*64},'streams':[{'commands':[['exact','scientific','argv']]}],'stage_quotas_seconds':{'one':7200},'outer_processing_reserve_seconds':3600,'candidate_metadata':{'kept':'original'}}
  cap={'input_sha256':dict(candidate['files_sha256']),'clock':{'host_utc':T.isoformat()}}
  r=M.resolve_release(candidate,cap,control,'c'*64,'d'*64,'e'*64)
  self.assertEqual(r['streams'],candidate['streams']);self.assertEqual(r['files_sha256'],candidate['files_sha256']);self.assertEqual(r['stage_quotas_seconds'],candidate['stage_quotas_seconds']);self.assertEqual(r['outer_processing_reserve_seconds'],3600)
  self.assertEqual(r['latest_start_utc'],(T+datetime.timedelta(seconds=595)).isoformat());self.assertEqual(r['candidate_metadata']['actual_complete_immutable_array_map'],{'x.npy':'a'*64});self.assertEqual(candidate['status'],'unadmitted')
 def test_resolver_rejects_input_divergence(self):
  with self.assertRaises(ValueError):M.resolve_release({'files_sha256':{'p':'a'}},{'input_sha256':{'p':'b'}},{'stop_utc':T.isoformat()},'c','d','e')
 def test_transport_preserves_nonzero_evaluation(self):
  proc=FakeProcess([1])
  with patch.object(M.subprocess,'Popen',return_value=proc):
   _,v=M.bounded_local(self.root,['synthetic'],'eval',T+datetime.timedelta(seconds=60),1060.,require_zero=False)
  self.assertEqual(v['exit_code'],1);self.assertTrue(v['remote_closure_separately_required'])
 def test_control_transport_rejects_nonzero(self):
  with patch.object(M.subprocess,'Popen',return_value=FakeProcess([1])):
   with self.assertRaises(ValueError):M.bounded_local(self.root,['synthetic'],'control',T+datetime.timedelta(seconds=60),1060.)
 def test_timeout_cleanup_and_no_relabel(self):
  with patch.object(M.subprocess,'Popen',return_value=FakeProcess([subprocess.TimeoutExpired('fake',1),subprocess.TimeoutExpired('fake',1),-9])),patch.object(M.os,'killpg'):
   with self.assertRaises(ValueError):M.bounded_local(self.root,['synthetic'],'timed',T+datetime.timedelta(seconds=60),1060.)
  v=M.read(self.root/'timed.external.json');self.assertEqual(v['signals_to_own_local_group'],['SIGTERM','SIGKILL']);self.assertTrue(v['local_transport_timeout'])
 def test_attempt_cannot_overwrite(self):
  with patch.object(M.subprocess,'Popen',return_value=FakeProcess([0])):M.bounded_local(self.root,['synthetic'],'one',T+datetime.timedelta(seconds=60),1060.)
  with self.assertRaises(FileExistsError):M.bounded_local(self.root,['synthetic'],'one',T+datetime.timedelta(seconds=60),1060.)
 def test_launch_exact_owner_argv_with_release_gate(self):
  argv=[M.PYTHON,M.R+'/cuda_preparation/supervise_sand_final_evaluation_scoped_v1.py','--execute','--release',M.CTRL+'/A.evaluation_release.json']
  value=M.launch_shell(argv,'a'*64)
  self.assertIn('sha256sum -- '+M.CTRL+'/A.evaluation_release.json',value);self.assertIn('&& exec /usr/bin/env -u CUDA_VISIBLE_DEVICES ',value);self.assertTrue(value.endswith(' '.join(argv)));self.assertNotIn('timeout',value)
 def test_issue_review_binds_exact_phase_release_and_inputs(self):
  self.phase();M.put(self.root/'prepared_inputs.json',{'synthetic':True});M.put(self.root/'A.evaluation_release.json',{'synthetic_release':True})
  r={'status':'passed_actual_final24_issuance','control_phase_sha256':M.sha(self.root/'control_phase.json'),'prepared_inputs_sha256':M.sha(self.root/'prepared_inputs.json'),'release_sha256':{'A':M.sha(self.root/'A.evaluation_release.json')}}
  path=self.root/'review.json';M.put(path,r);M.accepted_issue_review(self.root,'A',path,M.sha(path))
  (self.root/'A.evaluation_release.json').write_text('{}')
  with self.assertRaises(ValueError):M.accepted_issue_review(self.root,'A',path,M.sha(path))
 def test_actual_prepared_maps_integrate_with_pure_helper(self):
  prepared=M.read(M.PK/'prepared_inputs.json');control=self.phase()
  for role,row in prepared['roles'].items():
   with self.subTest(role=role):
    common={'role':role,'stop_utc':control['stop_utc'],'files_sha256':row['files_sha256'],'historical_pids':row['historical_pids'],'forbidden_program_tokens':prepared['forbidden_program_tokens'],'expected_boot_id':row['expected_boot_id']}
    sources={remote:base64.b64encode((M.PK/info['package_path']).read_bytes()).decode() for remote,info in prepared['source_files'].items()}
    H.validate_payload(dict(common,action='verify_stage',sources=sources))
    capture={'input_sha256':row['files_sha256'],'clock':{'host_utc':T.isoformat()}}
    release=M.resolve_release(M.read(M.PK/row['resolved_candidate']),capture,control,M.sha(M.PK/'prepared_inputs.json'),'d'*64,'e'*64)
    raw=json.dumps(release).encode();pin=hashlib.sha256(raw).hexdigest()
    H.validate_payload(dict(common,action='stage_release',release={'path':H.release_path(role),'sha256':pin,'base64':base64.b64encode(raw).decode()}))
    H.validate_payload(dict(common,action='observe_owner',release_sha256=pin,expected_owner_argv=row['owner_argv']))
 def test_prepared_forbidden_token_delta_only(self):
  previous=M.read(M.PK/'prepared_inputs.before_integration.json');current=M.read(M.PK/'prepared_inputs.json')
  added=M.R+'/cuda_preparation/train_sand_graph_support_cuda.py'
  self.assertEqual(set(current['forbidden_program_tokens'])-set(previous['forbidden_program_tokens']),{added})
  previous['forbidden_program_tokens']=current['forbidden_program_tokens'];self.assertEqual(previous,current)
if __name__=='__main__':unittest.main()
