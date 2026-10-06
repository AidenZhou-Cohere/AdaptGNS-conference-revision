"""Synthetic/scalar tests only. No CUDA, process launch, model or dataset load."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('graph_capacity',HERE/'measure_sand_graph_support_capacity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
UUIDS=[f'00000000-0000-0000-0000-{i:012x}' for i in range(4)]


def write(path,value):path.write_text(json.dumps(value))


def rows(seed=0,arm='base'):
    scalar=[];graphs=[]
    for s in range(1,513):
        ids=['train:000000:6','train:000001:319']
        scalar.append({'completed_steps':s,'loss':-.1,'lr':1e-4*(1e-5/1e-4)**((s-1)/99999),'frame_ids':ids,'particles':300,
                       'guarded_update_seconds':.5,'elapsed_seconds':float(s)})
        examples=[]
        for slot,n in enumerate((100,200)):
            active=arm=='mix'
            examples.append({'example_slot':slot,'n_particles':n,'exposure_coin':True,'expanded':active,
                'coin_seed_material':[20261005,seed,s-1,slot,4409],'pair_seed_material':[20261005,seed,s-1,slot,5501],
                'native_directed_edges':n,'native_self_edges':n,'receivers_above_native_cap':0,'annulus_pairs':8,
                'optional_budget_if_exposed':2,'selected_optional_pairs':2 if active else 0,
                'native_edge_sha256':'a'*64,'optional_pair_sha256':('b' if active else 'c')*64,'noisy_current_sha256':'d'*64})
        graphs.append({'completed_steps':s,'absolute_schedule_step':s-1,'frame_ids':ids,'noise_sha256':'e'*64,'examples':examples})
    return scalar,graphs


def measured():
    result=[]
    for job in m.SCHEDULE:
        scalar,graph=rows(job['seed'],job['arm'])
        result.append({**job,'pairing_rows':scalar,'graph_rows':graph,'steady_wall_seconds_per_update':.4 if job['wave']=='A' else .2,
                       'nonnegative_external_minus_all_guarded_seconds':20. if job['wave']=='A' else 10.})
    return result


def config(job):
    return {'schema':m.TRAINING_SCHEMA,'dataset':'Sand','objective':'faithful','arm':job['arm'],'seed':job['seed'],'updates':100000,'batch_size':2,'history':6,
            'initialization':'from scratch; paired seed across arms; empty Adam; no parent checkpoint',
            'architecture':{'width':128,'message_passing_blocks':10,'mlp_layers':2},'noise_std':6.7e-4,
            'graph':{'radius':.015,'backend':'scipy_host','cap':128,'self_candidates':True,'augmentation_probability':0.},
            'graph_exposure':m.T.graph_exposure_config(job['arm']),
            'optimizer':{'name':'Adam','initial_lr':1e-4,'final_lr':1e-5,'decay_updates':100000,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'foreach':False,'fused':False,'gradient_clipping':None},
            'checkpoint_every':10000,'log_every':1,'research_protocol_sha256':'protocol',
            'source_sha256':{**m.T.SOURCE_PINS,'train_sand_graph_support_cuda.py':m.TRAINER_SHA},
            'data':{'manifest_sha256':m.DATA_PINS['train_manifest'],'admission_sha256':m.DATA_PINS['admission'],'structural_report_sha256':m.DATA_PINS['structural_report'],'frames_per_trajectory':320,'particle_type_ids':[6]},
            'runtime':{'device':f"cuda:{job['gpu']}",'uuid':UUIDS[job['gpu']],'torch':'2.13.0+cu129','cuda':'12.9','threads':2,'deterministic_algorithms':True,'deterministic_warn_only':False,'cublas_workspace_config':m.ENVIRONMENT['CUBLAS_WORKSPACE_CONFIG'],'tf32':False,'amp':False,'compile':False,'ddp':False}}


def fixture(root,job):
    cfg=config(job);cfgsha=m.B.canonical_hash(cfg);r,g=rows(job['seed'],job['arm'])
    for step in (0,512):(root/f'checkpoint-{step:09d}.pt').write_bytes(f'fake{step}'.encode())
    pointers=[{'path':f'checkpoint-{step:09d}.pt','completed_steps':step,'run_config_sha256':cfgsha,'sha256':m.sha(root/f'checkpoint-{step:09d}.pt')} for step in (0,512)]
    status={'schema':m.TRAINING_SCHEMA,'state':'planned_stop_incomplete','completed_steps':512,'committed_steps':512,'requested_steps':100000,'error':None,'objective':'faithful','arm':job['arm'],'seed':job['seed'],'run_config_sha256':cfgsha,'process':{'pid':123},'latest_checkpoint':pointers[1],'last_training':r[-1],'attempt_id':'fake-attempt'}
    (root/'attempts/fake-attempt').mkdir(parents=True);write(root/'attempts/fake-attempt/status.json',status)
    for name,value in [('protocol.json',cfg),('status.json',status),('history.json',{'training':r,'graph_updates':g,'elapsed_seconds':513.}),('latest.json',pointers[1])]:write(root/name,value)
    (root/'stdout.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in r))
    external={'exit_code':0,'elapsed_seconds':520.,'pid':123,'initial_pointer':pointers[0],'stdout_file':str(root/'stdout.jsonl')}
    manifest={'records':[{'id':'train:000000','positions':{'shape':[320,100,2]}},{'id':'train:000001','positions':{'shape':[320,200,2]}}]}
    return external,manifest


class GraphCapacityTests(unittest.TestCase):
    def setUp(self):m.configure(HERE/'measure_sand_cuda_capacity_v2.py',HERE/'train_sand_graph_support_cuda.py')

    def test_default_never_imports_helpers_or_executes(self):
        with patch.object(m,'configure') as configure,patch('builtins.print'):
            self.assertEqual(m.main([]),0);configure.assert_not_called()

    def test_private_lifecycle_preserved_and_fixed_schedule(self):
        self.assertEqual(Path(m.B.run_wave.__code__.co_filename).name,'measure_sand_cuda_capacity_v2.py')
        self.assertEqual(m.sha(m.B.__file__),m.LIFECYCLE_SHA)
        self.assertEqual([j['id'] for j in m.B.SCHEDULE if j['wave']=='A'],['base_seed0','mix_seed0','base_seed1','mix_seed1'])
        self.assertEqual([j['id'] for j in m.B.SCHEDULE if j['wave']=='B'],['base_seed2','mix_seed2'])
        self.assertIs(m.B.verify_job,m.verify_job);self.assertIs(m.B.verify_pairing,m.verify_pairing)

    def test_fixed_commands_cannot_resume_or_change_objective(self):
        args=SimpleNamespace(**{k:Path('/'+k) for k in ('python','trainer','repo','train_manifest','admission','structural_report','protocol','output_dir')})
        for job in m.SCHEDULE:
            command=m.fixed_command(args,job)
            for flag,value in [('--arm',job['arm']),('--objective','faithful'),('--stop-after','512'),('--updates','100000'),('--log-every','1'),('--checkpoint-every','10000'),('--threads','2')]:self.assertEqual(command[command.index(flag)+1],value)
            self.assertNotIn('--resume',command)

    def test_exact_seed_arm_noise_graph_pairing(self):
        jobs=measured();self.assertEqual(len(m.verify_pairing(jobs)),3)
        for change in [('noise_sha256','f'*64),('absolute_schedule_step',99)]:
            wrong=copy.deepcopy(jobs);wrong[1]['graph_rows'][0][change[0]]=change[1]
            with self.assertRaises(ValueError):m.verify_pairing(wrong)
        wrong=copy.deepcopy(jobs);wrong[1]['graph_rows'][0]['examples'][0]['native_edge_sha256']='f'*64
        with self.assertRaises(ValueError):m.verify_pairing(wrong)
        with self.assertRaises(ValueError):m.verify_pairing(jobs[:-1])

    def test_lr_portability_preserves_raw_and_bounds_one_ulp(self):
        scalar,_=rows()
        original=scalar[15]['lr'];scalar[15]['lr']=math.nextafter(original,math.inf)
        result=m.validate_training_rows(scalar,{'train:000000':100,'train:000001':200})
        self.assertEqual(len(result['local_lr_recomputation_discrepancies']),1)
        self.assertEqual(scalar[15]['lr'],math.nextafter(original,math.inf))
        scalar[15]['lr']=math.nextafter(scalar[15]['lr'],math.inf)
        with self.assertRaisesRegex(ValueError,'beyond1float64ULP'):m.validate_training_rows(scalar,{'train:000000':100,'train:000001':200})

    def test_failed_worker_cannot_be_combined_even_if_wave_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'status.json',{'state':'failed_host_capacity_measurement','host_role':'A'})
            with self.assertRaisesRegex(ValueError,'final integrity'):m.verify_worker_completion(root,'A')

    def test_graph_job_passes_strict_byte_and_ledger_checks(self):
        for job in (m.SCHEDULE[0],m.SCHEDULE[1]):
            with self.subTest(arm=job['arm']),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);external,manifest=fixture(root,job)
                result=m.verify_job(root,job,external,manifest,'protocol','GPU-'+UUIDS[job['gpu']])
                self.assertEqual(len(result['graph_rows']),512)
                self.assertEqual(result['nonnegative_external_minus_all_guarded_seconds'],264.)

    def test_missing_graph_rows_bad_budget_and_failed_attempt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=m.SCHEDULE[1];external,manifest=fixture(root,job)
            history=m.read_json(root/'history.json')
            for mutate in [lambda h:h['graph_updates'].pop(),lambda h:h['graph_updates'][0]['examples'][0].update(selected_optional_pairs=1),lambda h:h['graph_updates'][0]['examples'][0].update(n_particles=101),lambda h:h['graph_updates'][0].update(noise_sha256='bad')]:
                changed=copy.deepcopy(history);mutate(changed);write(root/'history.json',changed)
                with self.assertRaises(ValueError):m.verify_job(root,job,external,manifest,'protocol','GPU-'+UUIDS[1])
            write(root/'history.json',history)
            (root/'attempts/fake-attempt/unsuccessful_history.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'failed attempt'):m.verify_job(root,job,external,manifest,'protocol','GPU-'+UUIDS[1])

    def test_checkpoint_and_attempt_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=m.SCHEDULE[0];external,manifest=fixture(root,job)
            (root/'checkpoint-000000512.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):m.verify_job(root,job,external,manifest,'protocol',UUIDS[0])
            (root/'attempts/second').mkdir()
            with self.assertRaisesRegex(ValueError,'one consistent fresh attempt'):m.verify_job(root,job,external,manifest,'protocol',UUIDS[0])

    def test_root_release_host_role_and_physical_uuid_gate(self):
        release={'schema':m.RELEASE_SCHEMA,'status':'admitted_for_timing','issued_by':'root','scientific_training_admitted':False,
                 'files_sha256':{},'schedule':m.SCHEDULE,'environment':m.ENVIRONMENT,'host_role':'A','cohort_id':'cohort','review_rationale':'reviewed','gpu_uuids':UUIDS,'clock_error_bound_seconds':1.}
        self.assertEqual(m.validate_release(release,{},'A'),UUIDS)
        for key,value in [('host_role','B'),('clock_error_bound_seconds',6.),('scientific_training_admitted',True),('gpu_uuids',[UUIDS[0],'GPU-'+UUIDS[0],UUIDS[2],UUIDS[3]])]:
            wrong={**release,key:value}
            with self.assertRaises(ValueError):m.validate_release(wrong,{},'A')

    def test_six_concurrent_max_formula_and_missing_costs(self):
        jobs=measured();at=datetime(2026,10,6,tzinfo=timezone.utc)
        result=m.forecast(jobs,at=at)
        self.assertAlmostEqual(result['core_training_and_short_probe_overhead_seconds'],1.35*(100000*.4+12*20))
        self.assertIsNone(result['fits_before_writing_reserve'])
        costs={'ledger_total_reserve_seconds':1000.,'full_rollout_seconds':2000.,'diagnostics_execution_seconds':3000.}
        result=m.forecast(jobs,costs,at)
        self.assertAlmostEqual(result['total_compute_analysis_seconds'],1.35*(100000*.4+12*20)+1000+2000+3000+3600)
        self.assertTrue(result['fits_before_writing_reserve'])
        self.assertFalse(m.forecast(jobs,costs,datetime(2026,10,7,tzinfo=timezone.utc))['fits_before_writing_reserve'])
        with self.assertRaises(ValueError):m.forecast(jobs[:-1])

    def test_full_cost_requires_six_policy_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'costs.json'
            costs={'schema':'adaptgns_sand_graph_support_cost_estimate_v1','status':'complete_workload_estimated','policy_count':5,
                   'rollout_outcomes':900,'horizon':314,'diagnostic_forward_calls':97704,'full_rollout_seconds':100.,'diagnostics_execution_seconds':100.,
                   'ledger_total_reserve_seconds':100.,'ledger_bound_rationale':'explicit bound','timing_evidence_sha256':['a'*64]}
            write(path,costs)
            with self.assertRaises(ValueError):m.cost_contract(path)
            costs.update(policy_count=6,rollout_outcomes=1080,diagnostic_forward_calls=124704);write(path,costs);self.assertEqual(m.cost_contract(path),costs)
            costs['diagnostic_forward_calls']=97704;write(path,costs)
            with self.assertRaises(ValueError):m.cost_contract(path)

    def test_cross_host_steady_interval_shrinks_clock_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            observations=[{'utc':f'2026-10-06T00:00:{s:02d}+00:00','gpu_processes':[{'pid':123}],
                           'trainer_states':{'123':{'state':'running','completed_steps':step}}} for s,step in [(0,1),(10,64),(20,200),(30,500),(40,512)]]
            write(root/'wave_A.json',{'jobs':[{'external':{'pid':123}}],'observations':observations})
            start,end=m.steady_interval(root,'A',2.)
            self.assertEqual((start.second,end.second),(12,28))
            write(root/'wave_A.json',{'jobs':[{'external':{'pid':123}}],'observations':observations[:2]})
            with self.assertRaises(ValueError):m.steady_interval(root,'A',2.)


if __name__=='__main__':unittest.main()
