"""Small synthetic metadata/tensor tests; no real probe, model or CUDA."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from test_measure_sand_graph_support_capacity import rows

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('terminal_cost',HERE/'measure_sand_terminal_ledger_cost.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('terminal_trainer',HERE/'train_sand_graph_support_cuda.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

class TerminalCostTests(unittest.TestCase):
    def test_default_has_no_execution(self):
        with patch.object(m,'probe_inputs') as p,patch('builtins.print'):
            self.assertEqual(m.main([]),0);p.assert_not_called()
        self.assertEqual(m.REPETITIONS,3);self.assertEqual(m.TERMINAL_RECORDS,100000)

    def test_explicit_paths_and_gpu_required(self):
        with self.assertRaises(SystemExit):m.parse(['--execute'])
        with self.assertRaises(SystemExit):m.parse(['--cuda-index','-1'])

    def test_synthetic_wrap_schedule_budget_and_unique_hashes(self):
        for arm in ('base','mix'):
            scalar,graph=rows(2,arm);probe={'training':scalar,'graph_updates':graph,'elapsed_seconds':512.}
            synthetic=m.synthetic_history(probe,2,arm,600)
            t.validate_graph_history(synthetic,600,arm,2)
            self.assertEqual(len(synthetic['graph_updates']),600)
            self.assertEqual([r['completed_steps'] for r in synthetic['training']],[1,100,200,300,400,500,600])
            self.assertEqual(len({r['noise_sha256'] for r in synthetic['graph_updates']}),600)
            self.assertEqual(synthetic['graph_updates'][512]['frame_ids'],graph[0]['frame_ids'])
            self.assertIsNot(synthetic['graph_updates'][512]['frame_ids'][0],graph[0]['frame_ids'][0])
            self.assertNotEqual(synthetic['graph_updates'][0]['examples'][0]['native_edge_sha256'],synthetic['graph_updates'][512]['examples'][0]['native_edge_sha256'])
            self.assertTrue(all(r['SYNTHETIC_METADATA_ONLY'] for r in synthetic['graph_updates']))
            self.assertEqual(probe['graph_updates'][0]['noise_sha256'],'e'*64)

    def test_envelope_cannot_load_as_any_graph_training_checkpoint(self):
        scalar,graph=rows();synthetic=m.synthetic_history({'training':scalar,'graph_updates':graph},0,'base',2)
        value=m.envelope({'state_dict':{'w':torch.tensor([1.])}},synthetic,'a'*64)
        self.assertNotIn('format_version',value);self.assertNotIn('state_dict',value);self.assertNotIn('completed_steps',value)
        self.assertTrue(value['NOT_A_TRAINING_CHECKPOINT']);self.assertEqual(value['actual_probe_completed_optimizer_updates'],512)
        with self.assertRaisesRegex(ValueError,'lineage'):
            t.restore_payload(SimpleNamespace(torch=torch),value,None,None,{'objective':'faithful','updates':100000},'cpu')

    def test_fixed_tensor_tree_and_exact_byte_inventory(self):
        data={'state_dict':{'w':torch.tensor([1.,-0.])},'optimizer_state':{'step':torch.tensor(512.)},'rng_states':[torch.tensor([1,2],dtype=torch.uint8)]}
        copied=m.tensor_tree(torch,data,'cpu')
        self.assertEqual(m.tensor_inventory(torch,data),m.tensor_inventory(torch,copied))
        inventory=m.tensor_inventory(torch,data)
        self.assertEqual(sum(r['bytes'] for r in inventory),14)
        changed={'state_dict':{'w':torch.tensor([1.,0.])},'optimizer_state':data['optimizer_state'],'rng_states':data['rng_states']}
        self.assertNotEqual(m.tensor_inventory(torch,data),m.tensor_inventory(torch,changed))

    def test_only_fixed_probe_artifact_names(self):
        values=m.probe_inputs(Path('/probe'))
        self.assertEqual(values['checkpoint'],Path('/probe/checkpoint-000000512.pt'))
        self.assertEqual(set(values),{'checkpoint','probe_protocol','probe_status','probe_history','probe_latest'})

    def test_publication_measures_hash_pointer_and_status(self):
        history={'training':[],'graph_updates':[{'SYNTHETIC_METADATA_ONLY':True}]}
        with TemporaryDirectory() as tmp,patch.object(m.time,'perf_counter',side_effect=[0.,1.,4.,6.,9.,13.]):
            out=Path(tmp)
            result=m.publish_synthetic_artifacts(torch,out,0,{'w':torch.tensor([1.,-0.])},history,'a'*64)
            self.assertEqual(result['binary_sha256_seconds'],3.)
            self.assertEqual(result['synthetic_pointer_fsync_seconds'],2.)
            self.assertEqual(result['synthetic_status_fsync_seconds'],3.)
            phase_keys=['envelope_serialization_fsync_seconds','binary_sha256_seconds','synthetic_pointer_fsync_seconds',
                        'synthetic_status_fsync_seconds','history_json_fsync_seconds']
            self.assertEqual(sum(result[k] for k in phase_keys),result['artifact_publication_seconds'])
            self.assertEqual(result['binary_sha256'],m.sha(out/result['binary_file']))
            self.assertFalse((out/'latest.json').exists());self.assertFalse((out/'status.json').exists())
            pointer=m.read(out/result['synthetic_pointer_file'])
            self.assertTrue(pointer['NOT_A_TRAINING_CHECKPOINT']);self.assertEqual(pointer['actual_probe_completed_optimizer_updates'],512)
            for name in result['synthetic_status_files']:self.assertTrue(m.read(out/name)['NOT_A_TRAINING_CHECKPOINT'])
            saved=torch.load(out/result['binary_file'],weights_only=True)
            self.assertEqual(m.tensor_inventory(torch,saved['fixed_probe_tensor_bytes']),m.tensor_inventory(torch,{'w':torch.tensor([1.,-0.])}))

if __name__=='__main__':unittest.main()
