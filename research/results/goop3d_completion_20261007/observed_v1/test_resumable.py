"""Synthetic fixtures only: durable resume, fixed denominators, and real worker dispatch."""
import copy
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_observed_resumable as R

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / 'deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1'
PINS = json.loads((HERE/'source_pins.json').read_bytes())
C,AUD,A,S,V = R.load_science(FROZEN,PINS)
R.SCIENCE=(C,AUD,A,S,V)
import synthetic_fixtures as FIX
from synthetic_summary_fixture import synthetic_audit


def clean_task(directory):
    directory=directory.resolve()
    row,arrays=FIX.clean_fixture()
    graph=dict.fromkeys(A.G_KEYS,0)
    graph.update(directed_edges=4,native_base_directed_edges=4,native_base_self_edges=4,
                 native_base_max_receiver_degree=1,native_base_prefix_preserved=True)
    row.update(source_index=0,target_frame=6,arm='base',training_seed=0,objective='faithful',
               protocol_sha256='a'*64,artifact_file='row.npz',graph=graph)
    buffer=io.BytesIO();AUD.np.savez_compressed(buffer,**arrays)
    (directory/'row.npz').write_bytes(buffer.getvalue())
    row['artifact_sha256']=R.file_hash(directory/'row.npz')
    (directory/'row.json').write_bytes(R.encode(row))
    files={str(directory/name):{'sha256':R.file_hash(directory/name),'bytes':(directory/name).stat().st_size}
           for name in ('row.json','row.npz')}
    return {'key':'synthetic_clean','arm':'base','seed':0,'stage':'clean_validation','mode':'clean-validation',
            'split':'valid','directory':str(directory),'row':row,
            'cell':{'source_index':0,'target_frame':6,'row_file':'row.json','state':'completed_required_outcome','failure':None},
            'bounds':FIX.BOUNDS.tolist(),'source':{'particles':4},'protocol_sha256':'a'*64,'files':files}


class ResumeTests(unittest.TestCase):
    def test_completed_run_does_not_reaudit(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d).resolve()
            R.atomic_write(output/'checkpoint_index.json',{})
            for name in ('audit.json','summary.json','arithmetic_check.json'):R.atomic_write(output/name,{})
            R.atomic_write(output/'completion.json',{
                'status':'all_required_observed_rows_audited_and_arithmetic_verified',
                'source_pins_sha256':'src','protocol_sha256':'proto',
                'checkpoint_index_sha256':R.file_hash(output/'checkpoint_index.json'),
                'products_sha256':{name:R.file_hash(output/name) for name in ('audit.json','summary.json','arithmetic_check.json')}})
            with patch.object(R,'prepare',side_effect=AssertionError('must not redo audit')),patch('sys.stdout',new_callable=io.StringIO) as stdout:
                R.execute(SimpleNamespace(frozen_dir=FROZEN,output=output),PINS,'proto','src')
            self.assertEqual(json.loads(stdout.getvalue())['status'],'already_completed')
            self.assertFalse((output/'attempts').exists())

    def test_durable_restart_and_identity_checks(self):
        with tempfile.TemporaryDirectory() as d:
            store=R.Checkpoints(d,{'source':'a'})
            store.commit('cell','b'*64,{'value':3})
            self.assertEqual(R.Checkpoints(d,{'source':'a'}).read('cell','b'*64),{'value':3})
            with self.assertRaisesRegex(ValueError,'identity differs'):R.Checkpoints(d,{'source':'changed'})
            with self.assertRaisesRegex(ValueError,'pin differs'):store.read('cell','c'*64)

    def test_corrupt_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            store=R.Checkpoints(d,{})
            store.commit('cell','b'*64,{'value':3})
            (Path(d)/'rows/cell.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'bytes differ'):store.read('cell','b'*64)

    def test_index_commit_failure_leaves_recoverable_orphan(self):
        with tempfile.TemporaryDirectory() as d:
            store=R.Checkpoints(d,{})
            original=R.atomic_write
            def fail_index(path,value):
                if Path(path)==store.path:raise OSError('synthetic index fsync failure')
                return original(path,value)
            with patch.object(R,'atomic_write',side_effect=fail_index):
                with self.assertRaises(OSError):store.commit('cell','b'*64,{'value':3})
            self.assertNotIn('cell',store.index['completed'])
            self.assertFalse(store.index['failed'])
            resumed=R.Checkpoints(d,{})
            self.assertIsNone(resumed.read('cell','b'*64))
            orphan=(Path(d)/'rows/cell.json').read_bytes()
            resumed.commit('cell','b'*64,{'value':3})
            self.assertEqual((Path(d)/'rows/cell.json').read_bytes(),orphan)
            self.assertEqual(resumed.read('cell','b'*64),{'value':3})

    def test_conflicting_orphan_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            store=R.Checkpoints(d,{})
            path=Path(d)/'rows/cell.json';path.write_text('preserve me')
            with self.assertRaisesRegex(ValueError,'unindexed row conflicts'):store.commit('cell','b'*64,{})
            self.assertEqual(path.read_text(),'preserve me')

    def test_failed_row_history_survives_explicit_success(self):
        with tempfile.TemporaryDirectory() as d:
            store=R.Checkpoints(d,{})
            attempt=Path(d)/'attempt';attempt.mkdir()
            store.failure('cell','b'*64,ValueError('synthetic scientific mismatch'),attempt)
            failure=(attempt/'cell.failure.json').read_bytes()
            store.commit('cell','b'*64,{'fixed':True})
            self.assertFalse(store.index['failed'])
            self.assertEqual((attempt/'cell.failure.json').read_bytes(),failure)


class FrozenRowTests(unittest.TestCase):
    def test_clean_row_frozen_hand_oracle(self):
        with tempfile.TemporaryDirectory() as d:
            task=clean_task(Path(d));result=R.audit_cell(task)
            for key,value in task['row']['metrics'].items():
                self.assertAlmostEqual(result['detail']['metrics']['metrics/'+key],value)
            self.assertGreater(result['checks'],20)
            self.assertEqual(result['pairings']['valid/truth_float32/0/6'],A.array_hash(AUD.np.zeros((4,3),dtype=AUD.np.float32)))

    def test_semantic_numeric_corruption_rejected_after_repinning(self):
        with tempfile.TemporaryDirectory() as d:
            task=clean_task(Path(d));task['row']['metrics']['normalized_acceleration_coordinate_mse']+=1
            path=Path(task['directory'])/'row.json';path.write_bytes(R.encode(task['row']))
            task['files'][str(path)]={'sha256':R.file_hash(path),'bytes':path.stat().st_size}
            with self.assertRaises(ValueError):R.audit_cell(task)

    def test_spawn_workers_match_serial_frozen_results(self):
        with tempfile.TemporaryDirectory() as d:
            task=clean_task(Path(d));serial=R.audit_cell(task)
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=R.initialize_worker,initargs=(str(FROZEN),PINS)) as pool:
                results=list(pool.map(R.audit_cell,[task,copy.deepcopy(task)]))
            self.assertEqual(results,[serial,serial])


class MergeTests(unittest.TestCase):
    def fixture(self,incomplete=False,bad_pairing=False):
        original=synthetic_audit(incomplete)
        prepared=[];records={};metric_keys={}
        for model in original['models']:
            stage=copy.deepcopy(model);stage['diagnostic_summary']={};keys={}
            metric_keys[model['mode']]=list(model['aggregates'])
            for metric,value in model['aggregates'].items():
                parent=stage['diagnostic_summary'];parts=metric.split('/')
                for part in parts[:-1]:parent=parent.setdefault(part,{})
                parent[parts[-1]]=value
            for row in model['rows']:
                ident=tuple(row['unit']);key=f"{model['arm']}_{model['seed']}_{model['stage']}_{ident}"
                keys[ident]=(key,'x')
                records[key]={'detail':{**row,'failure':None,'policy_completion':[]},'pairings':{'shared_synthetic':key if bad_pairing else 'same'},'checks':2}
            prepared.append((stage,keys))
        class Store:
            def read(self,key,pin):return copy.deepcopy(records[key])
        identity={'protocol_sha256':'p','source_pins_sha256':'s','original_source_manifest_sha256':{}}
        with patch.object(A,'diagnostic_keys',side_effect=lambda mode:metric_keys[mode]):
            merged=R.merge(C,A,A.Checks(),original['all_original_accounting'],original['source_schedules'],{}, {},{},prepared,Store(),identity,PINS)
        return original,merged

    def test_fixed_order_merging_matches_complete_summary(self):
        original,merged=self.fixture()
        expected=S.summarize(original);actual=S.summarize(merged)
        self.assertEqual(actual['families'],expected['families'])
        self.assertEqual(V.verify(merged,actual)['status'],'passed_independent_observed_arithmetic')
        self.assertIsNone(merged['phase_sha256'])

    def test_conflicting_cross_row_identity_rejected(self):
        with self.assertRaisesRegex(ValueError,'observed shared input'):
            self.fixture(bad_pairing=True)

    def test_missing_cell_retains_denominator_and_null_family(self):
        original,merged=self.fixture(True)
        actual=S.summarize(merged);V.verify(merged,actual)
        self.assertEqual(actual['families'],S.summarize(original)['families'])
        self.assertFalse(actual['families']['same_state_valid']['full_family_complete'])
        self.assertTrue(all(row['mean'] is None for row in actual['families']['same_state_valid']['mix_minus_base_training'].values()))
        self.assertEqual(sum(len(x['cells']) for x in merged['all_original_accounting']),4728)


if __name__=='__main__':unittest.main(verbosity=2)
