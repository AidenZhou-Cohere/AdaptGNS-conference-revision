import datetime,hashlib,importlib.util,json,os,socket,subprocess,tempfile,unittest
from pathlib import Path
from unittest import mock
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sand_transfer',P/'transfer_sand_reserved_A_to_B_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TransferTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.r=Path(self.temp.name)/'repo';self.r.mkdir();self.numeric=self.r/'reserved/numeric';self.numeric.mkdir(parents=True)
  self.files={}
  names=['metadata.json','test.json','structural_report.json']+[f'test/{s}_{i:06d}.npy' for i in range(30) for s in ('position','type')]
  for i,name in enumerate(names):
   p=self.numeric/name;p.parent.mkdir(exist_ok=True);p.write_bytes(('opaque-toy-'+str(i)).encode());self.files[str(p)]=m.sha(p)
  self.meta=self.r/'existing/metadata.json';self.meta.parent.mkdir();self.meta.write_bytes(b'toy-metadata');self.files[str(self.meta)]=m.sha(self.meta)
  self.v={'repo':str(self.r),'staging':str(self.r/'stage'),'files':self.files,'numeric_tree':str(self.numeric),'existing_only_paths':[str(self.meta)],'hostname':socket.gethostname(),'stop_utc':(m.utc()+datetime.timedelta(seconds=900)).isoformat(),'action':'source'}
  patch=mock.patch('signal.setitimer');patch.start();self.addCleanup(patch.stop)
 def call(self,action,**kw):return m.remote_operation({**self.v,'action':action,**kw})
 def remove_numeric(self):
  for p in self.numeric.rglob('*'):
   if p.is_file():p.unlink()
  (self.numeric/'test').rmdir();self.numeric.rmdir()
 def populate_stage(self,staged):
  for p in staged:
   q=Path(self.v['staging'])/'payload'/Path(p).relative_to(self.r);q.parent.mkdir(exist_ok=True,parents=True)
   i=list(self.files).index(p);q.write_bytes(('opaque-toy-'+str(i)).encode() if p!=str(self.meta) else b'toy-metadata')
 def test_source_exact_64(self):self.assertEqual(self.call('source')['files_sha256'],self.files)
 def test_source_extra_file_rejected(self):
  (self.numeric/'extra').write_bytes(b'x')
  with self.assertRaises(AssertionError):self.call('source')
 def test_source_extra_empty_directory_rejected(self):
  (self.numeric/'empty').mkdir()
  with self.assertRaises(AssertionError):self.call('source')
 def test_source_symlink_rejected(self):
  p=self.numeric/'metadata.json';p.unlink();p.symlink_to(self.meta)
  with self.assertRaises(AssertionError):self.call('source')
 def test_prepare_equal_no_copy(self):self.assertEqual(self.call('prepare')['staged'],[])
 def test_prepare_different_preserved(self):
  self.meta.write_bytes(b'different')
  with self.assertRaises(AssertionError):self.call('prepare')
  self.assertEqual(self.meta.read_bytes(),b'different');self.assertFalse(Path(self.v['staging']).exists())
 def test_missing_existing_only_rejected(self):
  self.meta.unlink()
  with self.assertRaises(AssertionError):self.call('prepare')
 def test_fresh_attempt_required(self):
  Path(self.v['staging']).mkdir()
  with self.assertRaises(AssertionError):self.call('prepare')
 def test_prepare_and_publish_63_opaque_files(self):
  self.remove_numeric();p=self.call('prepare');self.assertEqual(len(p['staged']),63);self.populate_stage(p['staged']);r=self.call('publish',staged=p['staged']);self.assertEqual(len(r['published']),63);self.assertTrue(Path(self.v['staging']).is_dir());self.assertEqual(self.call('source')['files_sha256'],self.files)
 def test_partial_numeric_stages_full_tree(self):
  (self.numeric/'metadata.json').unlink();p=self.call('prepare');self.assertEqual(len(p['missing']),1);self.assertEqual(len(p['staged']),63)
 def test_wrong_stage_hash_rejected_before_publish(self):
  self.remove_numeric();p=self.call('prepare');self.populate_stage(p['staged']);q=Path(self.v['staging'])/'payload'/self.numeric.relative_to(self.r)/'metadata.json';q.write_bytes(b'wrong')
  with self.assertRaises(AssertionError):self.call('publish',staged=p['staged'])
  self.assertFalse(self.numeric.exists());self.assertEqual(q.read_bytes(),b'wrong')
 def test_stage_extra_rejected(self):
  self.remove_numeric();p=self.call('prepare');self.populate_stage(p['staged']);(Path(self.v['staging'])/'payload/extra').write_bytes(b'x')
  with self.assertRaises(AssertionError):self.call('publish',staged=p['staged'])
 def test_publication_race_different_file_preserved(self):
  self.remove_numeric();p=self.call('prepare');self.populate_stage(p['staged']);written=[]
  def race(src,dst,**kw):Path(dst).write_bytes(b'racing-different');written.append(Path(dst));raise FileExistsError()
  with mock.patch('os.link',side_effect=race):
   with self.assertRaises(AssertionError):self.call('publish',staged=p['staged'])
  self.assertEqual(written[0].read_bytes(),b'racing-different')
 def test_publication_race_equal_file_retained(self):
  self.remove_numeric();p=self.call('prepare');self.populate_stage(p['staged'])
  def race(src,dst,**kw):Path(dst).write_bytes(Path(src).read_bytes());raise FileExistsError()
  with mock.patch('os.link',side_effect=race):r=self.call('publish',staged=p['staged'])
  self.assertEqual(r['published'],[]);self.assertEqual(len(r['retained']),64)
 def test_deadline_no_new_staging(self):
  with self.assertRaises(AssertionError):self.call('prepare',stop_utc=(m.utc()-datetime.timedelta(seconds=1)).isoformat())
  self.assertFalse(Path(self.v['staging']).exists())
 def test_numeric_only_recursive_command(self):
  plan=m.read(P/'sand_reserved_transfer_UNADMITTED_transition_v1/exact_map.json');commands=m.build_copies(plan,list(plan['files_sha256']),P/'ssh_config');recursive=[c for c in commands if '-r' in c];self.assertEqual(len(recursive),1);self.assertEqual(recursive[0][-2],plan['source_alias']+':'+plan['numeric_tree']);self.assertTrue(all(c[:3]==['scp','-3','-p'] for c in commands));self.assertNotIn('test.npz',json.dumps(commands))
class TransportTests(unittest.TestCase):
 def fixture(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);p=Path(tmp.name);phase=p/'phase.json';phase.write_text('{}');out=p/'receipts';out.mkdir();return m.Controller(phase,m.sha(phase),out,m.utc()+datetime.timedelta(seconds=90)),out
 def fake(self,effects,code=0):
  p=mock.Mock();p.pid=424242;p.returncode=code;p.communicate.side_effect=effects;p.poll.return_value=code;return p
 def test_clean_exit(self):
  c,out=self.fixture();p=self.fake([('ok','')])
  with mock.patch.object(m.subprocess,'Popen',return_value=p):self.assertEqual(c.bounded(['toy'],'clean',limit=900),'ok')
  t=p.communicate.call_args.kwargs['timeout'];self.assertGreater(t,70);self.assertLessEqual(t,78);self.assertTrue(m.read(next(out.glob('*.external.json')))['local_transport_reaped'])
 def test_nonzero_rejected(self):
  c,out=self.fixture();p=self.fake([('','fail')],code=7)
  with mock.patch.object(m.subprocess,'Popen',return_value=p):
   with self.assertRaises(AssertionError):c.bounded(['toy'],'nonzero')
 def test_timeout_term_reaped_rejected(self):
  c,out=self.fixture();p=self.fake([subprocess.TimeoutExpired(['toy'],1),('partial','')],code=-15)
  with mock.patch.object(m.subprocess,'Popen',return_value=p),mock.patch.object(m.os,'killpg') as kill:
   with self.assertRaises(AssertionError):c.bounded(['toy'],'timed')
  self.assertEqual(kill.call_args.args[0],p.pid);r=m.read(next(out.glob('*.external.json')));self.assertTrue(r['local_transport_timeout']);self.assertEqual(r['signals_to_own_local_group'],['SIGTERM']);self.assertTrue(r['remote_closure_separately_required'])
 def test_timeout_kill_reaped_rejected(self):
  c,out=self.fixture();p=self.fake([subprocess.TimeoutExpired(['toy'],1),subprocess.TimeoutExpired(['toy'],3),('partial','')],code=-9)
  with mock.patch.object(m.subprocess,'Popen',return_value=p),mock.patch.object(m.os,'killpg'):
   with self.assertRaises(AssertionError):c.bounded(['toy'],'kill')
  self.assertEqual(m.read(next(out.glob('*.external.json')))['signals_to_own_local_group'],['SIGTERM','SIGKILL'])
 def test_interruption_term_reaps_before_reraise(self):
  c,out=self.fixture();p=self.fake([KeyboardInterrupt(),('partial','interrupted')],code=-15);p.poll.side_effect=[None,-15]
  with mock.patch.object(m.subprocess,'Popen',return_value=p),mock.patch.object(m.os,'killpg') as kill:
   with self.assertRaises(KeyboardInterrupt):c.bounded(['toy'],'interrupt')
  self.assertEqual(kill.call_args.args,(p.pid,m.signal.SIGTERM));r=m.read(next(out.glob('*.external.json')));self.assertTrue(r['local_transport_reaped']);self.assertEqual(next(out.glob('*.stdout')).read_text(),'partial')
 def test_interruption_kill_reaps_before_reraise(self):
  c,out=self.fixture();p=self.fake([KeyboardInterrupt(),subprocess.TimeoutExpired(['toy'],3),('partial','interrupted')],code=-9);p.poll.side_effect=[None,-9]
  with mock.patch.object(m.subprocess,'Popen',return_value=p),mock.patch.object(m.os,'killpg'):
   with self.assertRaises(KeyboardInterrupt):c.bounded(['toy'],'interruptkill')
  r=m.read(next(out.glob('*.external.json')));self.assertTrue(r['local_transport_reaped']);self.assertEqual(r['signals_to_own_local_group'],['SIGTERM','SIGKILL'])
 def test_timeout_then_interruption_kills_and_reaps(self):
  c,out=self.fixture();p=self.fake([subprocess.TimeoutExpired(['toy'],1),KeyboardInterrupt(),('partial','timeout interrupted')],code=-9);p.poll.side_effect=[None,-9]
  with mock.patch.object(m.subprocess,'Popen',return_value=p),mock.patch.object(m.os,'killpg'):
   with self.assertRaises(KeyboardInterrupt):c.bounded(['toy'],'timeoutinterrupt')
  r=m.read(next(out.glob('*.external.json')));self.assertTrue(r['local_transport_reaped']);self.assertTrue(r['local_transport_timeout']);self.assertEqual(r['signals_to_own_local_group'],['SIGTERM','SIGKILL'])
 def test_optimized_local_interpreter_rejected(self):
  p=subprocess.run([m.sys.executable,'-O',m.__file__],capture_output=True,text=True)
  self.assertNotEqual(p.returncode,0);self.assertIn('Optimized interpreter forbidden',p.stderr)
 def test_expired_before_transport(self):
  c,out=self.fixture();c.mono_stop=m.time.monotonic()-1
  with mock.patch.object(m.subprocess,'Popen') as popen:
   with self.assertRaises(AssertionError):c.bounded(['toy'],'expired')
  popen.assert_not_called()
 def test_mutated_phase_before_transport(self):
  c,out=self.fixture();c.phase_path.write_text('changed')
  with mock.patch.object(m.subprocess,'Popen') as popen:
   with self.assertRaises(AssertionError):c.bounded(['toy'],'changed')
  popen.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
