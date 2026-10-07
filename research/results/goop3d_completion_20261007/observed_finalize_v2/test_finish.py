"""Synthetic-only tests of corrected checker and immutable cache reader."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FROZEN = HERE.parents[1]/'deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1'
sys.path[:0] = [str(HERE), str(FROZEN), str(BASE/'observed_v1')]
import finish_from_cache as f
import check_observed_exact_variance_v2 as checker
import check_goop3d_observed_history_summary_v1 as old
import summarize_goop3d_observed_histories_v1 as summary
import goop3d_observed_history_arithmetic_v1 as arithmetic
from synthetic_summary_fixture import synthetic_audit

spec = importlib.util.spec_from_file_location('runner', BASE/'observed_v1/run_observed_resumable.py')
runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
TEMP = str(Path(tempfile.gettempdir()).resolve())


class ExactCheckerTests(unittest.TestCase):
    def test_old_false_sd_and_new_exact_variance(self):
        for xs in ([102089.95333333334]*3, [100000.1]*3, [1651982.6]*3,
                   [100000.1,100000.10000000002,100000.10000000004],
                   [1.,2.,3.],[-1.,0.,1.],[0.,0.,0.],[None,1.,2.],
                   [1e-250,2e-250,3e-250],[1e155,2e155,3e155]):
            actual=checker.seed_values(xs);expected=summary.stats(xs)
            self.assertEqual(actual.keys(),expected.keys())
            for k in actual:
                if k=='sample_sd' and expected[k] is not None:
                    self.assertTrue(math.isclose(actual[k],expected[k],rel_tol=1e-10,abs_tol=1e-12))
                else:self.assertEqual(actual[k],expected[k])
        self.assertEqual(checker.seed_values([100000.1]*3)['sample_sd'],0.)
        self.assertGreater(old.seed_values([100000.1]*3)['sample_sd'],1e-12)

    def test_full_checker_regression_without_accuracy_or_tolerance_change(self):
        audit=synthetic_audit()
        for model in audit['models']:
            name='graph/base/candidate_pairs'
            for row in model['rows']:row['metrics'][name]=102089.95333333334
            expected=[tuple(row['unit']) for row in model['rows']]
            model['aggregates'][name]=arithmetic.aggregate(expected,{k:102089.95333333334 for k in expected})
        result=summary.summarize(audit)
        with self.assertRaisesRegex(ValueError,'independent arithmetic differs'):old.verify(audit,result)
        passed=checker.verify(audit,result)
        self.assertEqual(passed['checker_revision'],2)
        changed=deepcopy(result)
        changed['families']['same_state_valid']['absolute']['base']['accuracy/base/position_coordinate_mse']['mean']+=.01
        with self.assertRaisesRegex(ValueError,'independent arithmetic differs'):checker.verify(audit,changed)
        self.assertEqual(checker.equal.__code__.co_consts,old.equal.__code__.co_consts)

    def test_original_verify_body_only_adds_revision_metadata(self):
        before=ast.parse((FROZEN/'check_goop3d_observed_history_summary_v1.py').read_text())
        after=ast.parse((HERE/'check_observed_exact_variance_v2.py').read_text())
        bf={n.name:n for n in before.body if isinstance(n,ast.FunctionDef)}
        af={n.name:n for n in after.body if isinstance(n,ast.FunctionDef)}
        for name in ('valid','average','difference','equal'):
            self.assertEqual(ast.dump(bf[name]),ast.dump(af[name]))
        self.assertEqual(ast.dump(ast.Module(body=bf['verify'].body[:-1],type_ignores=[])),
                         ast.dump(ast.Module(body=af['verify'].body[:-1],type_ignores=[])))
        # No row-audit worker, executor, process launch or NumPy decoding call in the finisher.
        tree=ast.parse((HERE/'finish_from_cache.py').read_text())
        forbidden={'audit_cell','initialize_worker','ProcessPoolExecutor','np.load','numpy.load','subprocess','Popen'}
        calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(any(name in forbidden or name.rsplit('.',1)[-1] in forbidden for name in calls))
        self.assertIn('runner.prepare',calls);self.assertIn('runner.merge',calls)
        self.assertIn('summarizer.summarize',calls);self.assertIn('checker.verify',calls)


class CacheReaderTests(unittest.TestCase):
    def fixture(self, root):
        (root/'rows').mkdir()
        identity={'synthetic':True}
        result={'detail':{'metrics':{'test':1.}},'pairings':{},'checks':1}
        record={'key':'cell0','task_sha256':'task0','identity_sha256':runner.digest(runner.encode(identity)),'result':result}
        raw=runner.encode(record);(root/'rows/cell0.json').write_bytes(raw)
        index={'schema':'goop3d_observed_row_checkpoints_v1','identity':identity,'failed':{},
               'completed':{f'cell{i}':{'task_sha256':f'task{i}','sha256':hashlib.sha256(raw).hexdigest()} for i in range(2568)}}
        (root/'checkpoint_index.json').write_bytes(runner.encode(index))
        return identity,result,index

    def test_read_exact_cache_without_mutation(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as name:
            root=Path(name);identity,result,index=self.fixture(root)
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            reader=f.ReadOnlyCache(root,index,identity,runner)
            self.assertEqual(reader.read('cell0','task0'),result)
            self.assertEqual(before,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(ValueError):reader.read('cell0','wrong-task')
            with self.assertRaises(ValueError):reader.read('../cell0','task0')
            with self.assertRaises(ValueError):reader.read('cell9999','task9999')

    def test_corruption_and_incomplete_or_failed_cache_refused(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as name:
            root=Path(name);identity,result,index=self.fixture(root)
            reader=f.ReadOnlyCache(root,index,identity,runner)
            (root/'rows/cell0.json').write_text('{}')
            with self.assertRaises(ValueError):reader.read('cell0','task0')
            for mutation in ('missing','failed','identity'):
                bad=deepcopy(index)
                if mutation=='missing':del bad['completed']['cell1']
                elif mutation=='failed':bad['failed']['cell1']={'error':'synthetic'}
                else:bad['identity']={'changed':True}
                with self.assertRaises(ValueError):f.ReadOnlyCache(root,bad,identity,runner)


if __name__=='__main__':unittest.main(verbosity=2)
