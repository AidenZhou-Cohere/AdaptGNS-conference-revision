"""Synthetic-only capacity contract tests: no CUDA, arrays, Torch or processes."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('capacity', HERE/'measure_sand_cuda_capacity.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rows():
    return [{'completed_steps': s, 'loss': -.1, 'lr': 1e-4*(1e-5/1e-4)**((s-1)/99999),
             'frame_ids': ['train:000000:6', 'train:000001:319'], 'particles': 300,
             'guarded_update_seconds': .5, 'elapsed_seconds': float(s)} for s in range(1,513)]


def measured():
    return [{**j, 'steady_wall_seconds_per_update': .2 if j['wave']=='A' else .1,
             'nonnegative_external_minus_all_guarded_seconds': 10. if j['wave']=='A' else 5.,
             'pairing_rows': rows()} for j in m.SCHEDULE]


def write(path, value):
    path.write_text(json.dumps(value))


class CapacityTests(unittest.TestCase):
    def test_default_does_not_execute(self):
        with patch.object(m, 'run_wave') as run, patch('builtins.print'):
            self.assertEqual(m.main([]), 0)
            run.assert_not_called()

    def test_exact_schedule_and_commands(self):
        self.assertEqual([(j['wave'],j['gpu'],j['objective'],j['seed']) for j in m.SCHEDULE],
                         [('A',0,'faithful',0),('A',1,'nll',0),('A',2,'faithful',1),('A',3,'nll',1),('B',0,'faithful',2),('B',1,'nll',2)])
        args = SimpleNamespace(**{k: Path('/'+k) for k in ('python','trainer','repo','train_manifest','admission','structural_report','protocol','output_dir')})
        for job in m.SCHEDULE:
            cmd=m.fixed_command(args,job)
            for flag, value in [('--updates','100000'),('--stop-after','512'),('--threads','2'),('--checkpoint-every','10000'),('--log-every','1')]:
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertNotIn('--resume',cmd)

    def test_failed_a_never_starts_b(self):
        fn=Mock(return_value=([], {'state':'failed'}))
        with self.assertRaises(ValueError): m.run_waves(fn)
        fn.assert_called_once_with('A')

    def test_raised_a_never_starts_b(self):
        fn=Mock(side_effect=RuntimeError('failed'))
        with self.assertRaises(RuntimeError): m.run_waves(fn)
        fn.assert_called_once_with('A')

    def test_success_a_then_b(self):
        jobs=measured()
        fn=Mock(side_effect=lambda wave: ([j for j in jobs if j['wave']==wave],{'state':'verified'}))
        self.assertEqual(len(m.run_waves(fn)),6)
        self.assertEqual([c.args[0] for c in fn.call_args_list],['A','B'])

    def test_rows_wall_q_and_fixed_window(self):
        result=m.validate_rows(rows(),{'train:000000':100,'train:000001':200})
        self.assertEqual(result['steady_wall_seconds_per_update'],1.)
        self.assertEqual(result['steady_guarded_seconds'],{'mean':.5,'median':.5,'p90_linear':.5,'maximum':.5})
        self.assertEqual(result['sum_all512_guarded_seconds'],256.)
        self.assertEqual(len(result['all512_timings']),512)

    def test_rows_reject_incomplete_nonfinite_order_count_rate_frame(self):
        corruptions=[lambda r:r.pop(),lambda r:r[9].update(loss=float('nan')),lambda r:r[1].update(completed_steps=1),
                     lambda r:r[0].update(particles=1),lambda r:r[0].update(lr=1.),
                     lambda r:r[0].update(frame_ids=['train:000000:5','train:000001:319']),
                     lambda r:r[0].update(guarded_update_seconds=0.)]
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt):
                r=rows();corrupt(r)
                with self.assertRaises(ValueError):m.validate_rows(r,{'train:000000':100,'train:000001':200})

    def test_paired_schedule_exact(self):
        jobs=measured();self.assertEqual(len(m.verify_pairing(jobs)),3)
        jobs[1]['pairing_rows'][0]['frame_ids']=['train:000001:319','train:000000:6']
        with self.assertRaises(ValueError):m.verify_pairing(jobs)

    def test_fixed_formula_and_complete_cost_gate(self):
        result=m.forecast(measured(),at=datetime(2026,10,6,tzinfo=timezone.utc))
        self.assertAlmostEqual(result['training_seconds'],1.35*(100000*.3+12*15))
        self.assertIsNone(result['fits_before_seven_hour_writing_reserve'])
        self.assertIsNone(m.forecast(measured(),100.,at=datetime(2026,10,6,tzinfo=timezone.utc))['forecast_finish_utc'])
        r=m.forecast(measured(),100.,200.,at=datetime(2026,10,6,tzinfo=timezone.utc))
        self.assertAlmostEqual(r['total_compute_analysis_seconds'],r['training_seconds']+100+200+3600)
        self.assertTrue(r['fits_before_seven_hour_writing_reserve'])
        r=m.forecast(measured(),100.,200.,at=datetime(2026,10,7,tzinfo=timezone.utc))
        self.assertFalse(r['fits_before_seven_hour_writing_reserve'])
        with self.assertRaises(ValueError):m.forecast(measured()[:-1])

    def test_signal_skips_zombie_and_continues(self):
        children=[{'process':SimpleNamespace(pid=p,returncode=None),'identity':{'start_ticks':p},'command':['cmd'], 'signals':[]} for p in (1,2,3)]
        identities=[{'state':'Z'},FileNotFoundError(),{'state':'R','ppid':m.os.getpid(),'argv':['cmd'],'start_ticks':3}]
        with patch.object(m,'process_identity',side_effect=identities),patch.object(m.os,'killpg') as kill:
            m.stop_owned(children)
            kill.assert_called_once_with(3,signal.SIGINT)
        self.assertEqual(children[1]['signals'][0]['result'],'already_exited')

    def test_changed_identity_never_signalled(self):
        child={'process':SimpleNamespace(pid=3,returncode=None),'identity':{'start_ticks':3},'command':['cmd'],'signals':[]}
        with patch.object(m,'process_identity',return_value={'state':'R','ppid':m.os.getpid(),'argv':['different'],'start_ticks':3}),patch.object(m.os,'killpg') as kill:
            m.stop_owned([child]);kill.assert_not_called()
        self.assertEqual(child['signals'][0]['result'],'refused_or_failed')

    def test_cleanup_reaps_even_when_initial_pointer_read_fails(self):
        child={'process':SimpleNamespace(pid=1,returncode=None),'handles':[]}
        record={}
        def reap(*args):child['process'].returncode=1
        with patch.object(m,'capture_initial',side_effect=ValueError('bad pointer')),patch.object(m,'reap_child',side_effect=reap) as reaper,patch.object(m,'stop_owned'),patch.object(m.time,'sleep'):
            child['job']={'id':'fake'}
            m.cleanup_owned(None,[child],record,[],None,None)
            reaper.assert_called_once()
        self.assertEqual(record['unreaped_owned_children'],[])

    def test_timing_only_release(self):
        release={'schema':m.RELEASE_SCHEMA,'status':'admitted_for_timing','issued_by':'root','scientific_training_admitted':False,
                 'files_sha256':{},'environment':m.ENVIRONMENT,'schedule':m.SCHEDULE,'review_rationale':'reviewed','gpu_uuids':['a','b','c','d']}
        self.assertEqual(m.validate_release(release,{}),['a','b','c','d'])
        for key,value in [('status','not_admitted'),('scientific_training_admitted',True),('issued_by','agent')]:
            wrong=copy.deepcopy(release);wrong[key]=value
            with self.assertRaises(ValueError):m.validate_release(wrong,{})

    def test_spawn_identity_failure_registers_owned_child_for_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'logs').mkdir();(root/'jobs').mkdir()
            args=SimpleNamespace(**{k:Path('/'+k) for k in ('python','trainer','repo','train_manifest','admission','structural_report','protocol')},output_dir=root)
            proc=SimpleNamespace(pid=123,returncode=None)
            def cleanup(args, children, record, verified, launch, manifest):
                self.assertEqual(len(children),1)
                self.assertIs(children[0]['process'],proc)
                for handle in children[0]['handles']:handle.close()
                proc.returncode=1
            with patch.object(m,'gpu_processes',return_value=[]),patch.object(m.subprocess,'Popen',return_value=proc),patch.object(m,'process_identity',side_effect=ValueError('identity failed')),patch.object(m,'cleanup_owned',side_effect=cleanup) as clean:
                with self.assertRaisesRegex(ValueError,'identity failed'):m.run_wave(args,'A',{}, {})
                clean.assert_called_once()
            self.assertEqual(m.read_json(root/'wave_A.json')['state'],'failed')

    def test_copied_input_integrity_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'inputs').mkdir()
            keys=('trainer','train_manifest','admission','structural_report','protocol','mechanism_review','supervisor')
            snapshots={key:'inputs/'+key+'.txt' for key in keys}
            for key,path in snapshots.items():(root/path).write_bytes(Path(m.__file__).read_bytes() if key=='supervisor' else key.encode())
            hashes={key:m.sha(root/path) for key,path in snapshots.items()};hashes['python']='fake-interpreter-hash'
            release={'schema':m.RELEASE_SCHEMA,'status':'admitted_for_timing','issued_by':'root','scientific_training_admitted':False,'files_sha256':hashes,'environment':m.ENVIRONMENT,'schedule':m.SCHEDULE,'review_rationale':'reviewed','gpu_uuids':['a','b','c','d']}
            write(root/'inputs/release.json',release)
            launch={'schema':m.SCHEMA,'schedule':m.SCHEDULE,'environment':m.ENVIRONMENT,'files_sha256':hashes,'input_snapshots':snapshots,'release_sha256':m.sha(root/'inputs/release.json'),'gpu_uuids':release['gpu_uuids']}
            write(root/'launch.json',launch)
            with patch.object(m,'PINS',{key:hashes[key] for key in m.PINS}):
                self.assertEqual(m.verify_snapshot(root),launch)
                (root/snapshots['trainer']).write_text('tampered')
                with self.assertRaisesRegex(ValueError,'Copied input hash'):m.verify_snapshot(root)

    def test_strict_job_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=m.SCHEDULE[0];r=rows()
            config={'schema':m.TRAINING_SCHEMA,'dataset':'Sand','objective':'faithful','seed':0,'updates':100000,'batch_size':2,'history':6,
                    'architecture':{'width':128,'message_passing_blocks':10,'mlp_layers':2},'noise_std':6.7e-4,
                    'graph':{'radius':.015,'backend':'scipy_host','cap':128,'self_candidates':True,'augmentation_probability':0.},
                    'optimizer':{'name':'Adam','initial_lr':1e-4,'final_lr':1e-5,'decay_updates':100000,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'foreach':False,'fused':False,'gradient_clipping':None},
                    'checkpoint_every':10000,'log_every':1,'research_protocol_sha256':'protocol',
                    'source_sha256':{**m.CORE_PINS,'train_sand_cuda_deterministic.py':m.PINS['trainer']},
                    'data':{'manifest_sha256':m.PINS['train_manifest'],'admission_sha256':m.PINS['admission'],'structural_report_sha256':m.PINS['structural_report'],'frames_per_trajectory':320,'particle_type_ids':[6]},
                    'runtime':{'device':'cuda:0','uuid':'uuid','torch':'2.13.0+cu129','cuda':'12.9','threads':2,'deterministic_algorithms':True,'deterministic_warn_only':False,'cublas_workspace_config':m.ENVIRONMENT['CUBLAS_WORKSPACE_CONFIG'],'tf32':False,'amp':False,'compile':False,'ddp':False}}
            config_hash=m.canonical_hash(config)
            (root/'checkpoint-000000000.pt').write_bytes(b'fake0');(root/'checkpoint-000000512.pt').write_bytes(b'fake512')
            pointers=[{'path':f'checkpoint-{s:09d}.pt','completed_steps':s,'run_config_sha256':config_hash,'sha256':m.sha(root/f'checkpoint-{s:09d}.pt')} for s in (0,512)]
            status={'schema':m.TRAINING_SCHEMA,'state':'planned_stop_incomplete','completed_steps':512,'committed_steps':512,'requested_steps':100000,'error':None,'objective':'faithful','seed':0,'run_config_sha256':config_hash,'process':{'pid':123},'latest_checkpoint':pointers[1],'last_training':r[-1]}
            for name,value in [('protocol.json',config),('status.json',status),('history.json',{'training':r,'elapsed_seconds':513.}),('latest.json',pointers[1])]:write(root/name,value)
            (root/'stdout.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in r))
            external={'exit_code':0,'elapsed_seconds':520.,'pid':123,'initial_pointer':pointers[0],'stdout_file':str(root/'stdout.jsonl')}
            manifest={'records':[{'id':'train:000000','positions':{'shape':[320,100,2]}},{'id':'train:000001','positions':{'shape':[320,200,2]}}]}
            result=m.verify_job(root,job,external,manifest,'protocol','uuid')
            self.assertEqual(result['nonnegative_external_minus_all_guarded_seconds'],264.)
            for change in [lambda:write(root/'status.json',{**status,'state':'complete'}),lambda:write(root/'latest.json',{**pointers[1],'sha256':'bad'}),lambda:(root/'run.lock').write_text('locked')]:
                change()
                with self.assertRaises(ValueError):m.verify_job(root,job,external,manifest,'protocol','uuid')
                write(root/'status.json',status);write(root/'latest.json',pointers[1]);(root/'run.lock').unlink(missing_ok=True)
            (root/'checkpoint-000000512.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):m.verify_job(root,job,external,manifest,'protocol','uuid')


if __name__=='__main__':unittest.main()
