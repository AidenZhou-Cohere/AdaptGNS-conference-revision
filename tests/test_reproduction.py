"""Saved-result, syntax and synthetic metadata checks; no simulation execution."""
import ast,hashlib,importlib.util,json,math,statistics,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('portable_evaluation',ROOT/'code/evaluate.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class SavedResults(unittest.TestCase):
 def test_all_python_syntax(self):
  for path in ROOT.rglob('*.py'):ast.parse(path.read_text(),filename=str(path.relative_to(ROOT)))
 def test_native_figure_coordinates_and_text(self):
  expected=json.loads((ROOT/'tests/expected_figures.json').read_text())
  for name,sha in expected.items():self.assertEqual(digest(ROOT/'generated'/name),sha,name)
 def test_complete_rows_and_nulls(self):
  counts=json.loads((ROOT/'generated/table_counts.json').read_text())
  self.assertEqual(counts['primary_policy_rows'],34);self.assertEqual(counts['physical_policy_rows'],34)
  self.assertEqual(counts['prepared_d3_cells'],18)
  counts=json.loads((ROOT/'generated/figure_counts.json').read_text())
  self.assertEqual(counts['undefined_three_seed_means'],4);self.assertEqual(counts['not_predeclared_slots'],1)
  failures=json.loads((ROOT/'results/failure_accounting.json').read_text())
  self.assertEqual(len(failures['goop2d']['failed_cases']),3)
  self.assertEqual(sum(failures['waterdrop_objective_control']['nll_seed2_failed_counts'].values()),8)
 def test_seed_statistics_and_full_denominators(self):
  def walk(value):
   if isinstance(value,dict):
    if 'seed_values' in value and 'mean' in value:
     vals=value['seed_values'];vals=list(vals.values()) if isinstance(vals,dict) else vals
     if len(vals)==3:
      if any(v is None for v in vals):
       self.assertIsNone(value['mean']);self.assertIsNone(value.get('sample_sd'))
      else:
       self.assertTrue(math.isclose(statistics.mean(vals),value['mean'],rel_tol=1e-11,abs_tol=1e-15))
       if 'sample_sd' in value:self.assertTrue(math.isclose(statistics.stdev(vals),value['sample_sd'],rel_tol=1e-9,abs_tol=1e-15))
    for child in value.values():walk(child)
   elif isinstance(value,list):
    for child in value:walk(child)
  for name in ('results/cross_material/printed_statistic_map.json','results/goop3d/observed_statistics.json'):walk(json.loads((ROOT/name).read_text()))
 def test_d3_uses_100_source_split_grid_and_3d_guards(self):
  settings=json.loads((ROOT/'protocols/studies.json').read_text())['studies']['goop3d']
  expected=[j*99//29 for j in range(30)]
  self.assertEqual(settings['source_indices_by_split'],{'valid':expected,'test':expected})
  self.assertEqual(settings['graph']['base_radius'],.025)
  self.assertEqual(settings['evaluation']['guards']['max_candidate_pairs'],2000000)
  self.assertEqual(settings['evaluation']['guards']['max_directed_edges'],5000000)

class EvaluationConfig(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
  data=root/'data';data.mkdir();models=root/'models';models.mkdir()
  self.metadata={'dim':2,'default_connectivity_radius':.015,'dt':.0025}
  mp=data/'metadata.json';mp.write_text(json.dumps(self.metadata))
  cp=models/'model.pt';cp.write_bytes(b'synthetic metadata fixture, not a model')
  self.manifest={'format':'gns-trajectory-manifest','version':1,'dataset':'Goop','split':'test','metadata':self.metadata,'metadata_sha256':digest(mp),'record_count':30,
   'records':[{'id':f'test:{i:06d}','source_index':i,'positions':{'shape':[401,5,2],'dtype':'<f4'},'particle_types':{'shape':[5],'dtype':'<i8'}} for i in range(30)]}
  self.manifest_path=data/'test.json';self.manifest_path.write_text(json.dumps(self.manifest))
  self.argv=['--study','goop','--arm','base','--seed','0','--manifest',str(self.manifest_path),'--metadata',str(mp),'--checkpoint',str(cp),
   '--output-dir',str(root/'result'),'--manifest-sha256',digest(self.manifest_path),'--metadata-sha256',digest(mp),'--checkpoint-sha256',digest(cp)]
 def check(self):
  self.manifest_path.write_text(json.dumps(self.manifest));self.argv[self.argv.index('--manifest-sha256')+1]=digest(self.manifest_path)
  return evaluation.configuration(evaluation.parse_args(self.argv))
 def test_original_source_order(self):self.assertEqual(self.check()['indices'],list(range(30)))
 def test_reordered_sources_rejected(self):
  self.manifest['records'].reverse()
  with self.assertRaisesRegex(ValueError,'Ordered original'):self.check()
 def test_float64_positions_rejected(self):
  self.manifest['records'][0]['positions']['dtype']='<f8'
  with self.assertRaisesRegex(ValueError,'float32'):self.check()
 def test_missing_split_record_rejected(self):
  self.manifest['records'].pop()
  with self.assertRaisesRegex(ValueError,'30-record'):self.check()
 def test_observed_rng_is_separate(self):
  self.argv.extend(['--mode','observed']);self.assertIn('SeedSequence',self.check()['rng_seed'])
 def test_wrong_metadata_hash_rejected(self):
  self.argv[self.argv.index('--metadata-sha256')+1]='0'*64
  with self.assertRaisesRegex(ValueError,'hash mismatch'):self.check()

if __name__=='__main__':unittest.main()
