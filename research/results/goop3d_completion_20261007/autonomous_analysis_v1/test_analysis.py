"""Small invented arrays only; exercise existing arithmetic plus new checkpoint/merge glue."""
import ast
from copy import deepcopy
import importlib.util
import io
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PREP = Path(os.environ.get('GOOP3D_ANALYSIS_FROZEN_DIR', HERE.parents[1] / 'deadline_research_20261005' / 'cuda_preparation'))
if not PREP.is_dir():
    PREP = HERE.parents[1] / 'cuda_preparation'
TEST_TEMP = str(Path(tempfile.gettempdir()).resolve())
sys.path.insert(0, str(HERE))
import analyze as a
saved, scalar, paired = a.load_science(PREP)
np = saved.np
source = ast.parse((PREP / 'test_audit_goop3d_saved_arrays_v1.py').read_text())
namespace = {'np': np, 'math': math, 'audit': saved, 'BOUNDS': np.asarray([[.1,.9]] * 3)}
for node in source.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('fixed_boundary', 'make_rollout'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen synthetic fixture>', 'exec'), namespace)
make = namespace['make_rollout']


def fixture(n=295, arm='base', seed=0, index=0, policy='base'):
    row, arrays = make(n)
    row.update(arm=arm, training_seed=seed, objective='faithful', source_index=index,
               trajectory_id=f'test:{index:06d}', size_group='fixed_source_grid', policy=policy,
               synchronized_call_seconds=1., publication_seconds=.001)
    for attempt in row['attempts'][:n]:
        attempt['selected_optional_pair_sha256'] = saved.array_hash(np.empty((0, 2), dtype=np.int64))
        attempt['directed_edge_sha256'] = saved.array_hash(np.asarray([[0, 1], [0, 1]], dtype=np.int64))
    if policy == 'laggedrisk25' and len(arrays['cached_risk_before_forecast']):
        arrays['cached_risk_before_forecast'][0] = 1.
    return row, arrays


class CheckpointTests(unittest.TestCase):
    def test_commit_resume_and_identity_guard(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as name:
            root = Path(name)
            store = a.Checkpoints(root, {'source': 'frozen'}, False)
            store.commit('cell0', 'task', {'status': 'failed', 'guard': 'retained'})
            resumed = a.Checkpoints(root, {'source': 'frozen'}, True)
            self.assertEqual(resumed.read('cell0', 'task')['guard'], 'retained')
            with self.assertRaises(ValueError):
                resumed.read('cell0', 'changed')
            with self.assertRaises(ValueError):
                a.Checkpoints(root, {'source': 'changed'}, True)
            with self.assertRaises(ValueError):
                resumed.commit('cell0', 'task', {})

    def test_orphan_recovery_and_corruption(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as name:
            root = Path(name)
            store = a.Checkpoints(root, {}, False)
            record = {'key': 'cell0', 'task_sha256': 't', 'identity_sha256': a.digest(a.encode({})), 'result': {'ok': True}}
            a.atomic(root / 'rows/cell0.json', record)
            self.assertIsNone(store.read('cell0', 't'))
            store.commit('cell0', 't', {'ok': True})
            (root / 'rows/cell0.json').write_text('{}')
            with self.assertRaises(ValueError):
                store.read('cell0', 't')

    def test_strict_json_and_path_rejection(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                a.strict(raw)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as name:
            with self.assertRaises(ValueError):
                a.relative_file(name, '../escape')

    def test_failure_record_survives_success(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as name:
            root = Path(name)
            store = a.Checkpoints(root, {}, False)
            attempt = root / 'attempt'; attempt.mkdir()
            store.failure('cell0', 't', ValueError('invented error'), attempt)
            self.assertIn('cell0', store.index['failed'])
            store.commit('cell0', 't', {'ok': True})
            self.assertFalse(store.index['failed'])
            self.assertTrue((attempt / 'cell0.failure.json').exists())


    def test_index_publication_failure_keeps_memory_and_orphan_reusable(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as name:
            root = Path(name);store = a.Checkpoints(root, {}, False)
            original = a.atomic
            def failing(path, value):
                if Path(path) == store.path:
                    raise OSError('invented storage interruption')
                return original(path, value)
            with patch.object(a, 'atomic', failing), self.assertRaises(OSError):
                store.commit('cell0', 't', {'ok': True})
            self.assertIsNone(store.read('cell0', 't'))
            self.assertTrue((root/'rows/cell0.json').exists())
            store.commit('cell0', 't', {'ok': True})
            self.assertTrue(store.read('cell0', 't')['ok'])


class CollectorTests(unittest.TestCase):
    def test_original331_plus_one_marker_and_missing_input_rejected(self):
        plan = a.strict((HERE.parent / 'autonomous_v1/plan.json').read_bytes())
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as name:
            root = Path(name);original = root/'original';original.mkdir()
            metadata_path = root/'metadata.json'
            metadata_pin = a.atomic(metadata_path, {'bounds': namespace['BOUNDS'].tolist()})
            queue = Path(plan['original_remote_repo'])/'goop3d_final_evaluation_20261006_v1'
            for cell in plan['retained_original_outcomes']:
                directory = original/Path(cell['original_directory']).relative_to(queue);directory.mkdir(parents=True,exist_ok=True)
                protocol = {'arm':cell['arm'],'seed':cell['seed'],'split':cell['split'],'mode':'full-rollout','horizon':295,'checkpoint_updates':25000}
                protocol_pin = a.atomic(directory/'protocol.json', protocol)
                artifact = directory/(Path(cell['row_file']).stem+'.npz');artifact.write_bytes(b'invented-not-loaded-array')
                row = {k:cell[k] for k in ('source_index','trajectory_id','particles','size_group','policy','arm')}
                row.update(training_seed=cell['seed'],objective='faithful',horizon=295,
                           rng_seed=93000+1000*cell['seed']+cell['source_index'],protocol_sha256=protocol_pin,
                           artifact_file=artifact.name,artifact_sha256=a.sha(artifact),status='complete',failure=None)
                cell['row_sha256']=a.atomic(directory/cell['row_file'],row);cell['artifact_sha256']=row['artifact_sha256']
            allcells=plan['retained_original_outcomes']+[c for w in plan['workers'] for c in w['cells']]
            stages=[]
            for arm,seed in saved.MODELS:
                for split in ('valid','test'):
                    cells=sorted([c for c in allcells if (c['arm'],c['seed'],c['split'])==(arm,seed,split)],
                                 key=lambda c:(c['source_index'],plan['policies'].index(c['policy'])))
                    stages.append({'arm':arm,'seed':seed,'split':split,'mode':'full-rollout','stage':'full_rollout_'+split,
                                   'cells':cells,'outcome':{'reason':'original invented interruption'}})
            ledger={'schema':'adaptgns_goop3d_final_evaluation_ledger_v1','state':'stopped_all_owned_processes_reaped',
                    'all_pinned_inputs_reverified':True,'unreaped_owned_children':[],'stages':stages,
                    'cohort_sha256':'invented','source_manifest_sha256':{'valid':'invented','test':'invented'}}
            ledger_path=root/'ledger.json';ledger_pin=a.atomic(ledger_path,ledger)
            plan_path=root/'plan.json';plan_pin=a.atomic(plan_path,plan)
            roots={i:root/f'worker_{i:02d}' for i in range(12)}
            worker=plan['workers'][0];cell=worker['cells'][0];wroot=roots[0];wroot.mkdir()
            source_root=Path(plan['original_remote_repo']);runtime=Path('/synthetic/runtime')
            checkpoint=next(m['checkpoint_path'] for m in plan['models'] if (m['arm'],m['seed'])==(worker['arm'],worker['seed']))
            expected={str(source_root/p) for p in plan['source_pins']}|{plan['cohort_path'],plan['cohort_audit_path'],checkpoint}
            for split in {c['split'] for c in worker['cells']}:
                info=plan['splits'][split];manifest=Path(info['manifest_path']);expected.update((str(manifest),str(manifest.parent/'metadata.json')))
                for index in {c['source_index'] for c in worker['cells'] if c['split']==split}:
                    rec=next(r for r in info['records'] if r['source_index']==index)
                    expected.update(str(manifest.parent/rec[k]['path']) for k in ('positions','particle_types','step_context'))
            filepins={f['path']:f for f in plan['files']}
            inputs={p:{'original_path':p,'execution_path':str(runtime/Path(p).relative_to(source_root)),
                       'sha256':filepins[p]['sha256'],'size_bytes':filepins[p]['size_bytes'] or 17} for p in expected}
            identity={'schema':a.WORKER_SCHEMA+'_identity','plan_sha256':plan_pin,'worker_source_sha256':a.WORKER_SHA,
                      'worker_index':0,'arm':worker['arm'],'seed':worker['seed'],'cell_ids':[c['cell_id'] for c in worker['cells']],
                      'original_repo_root':str(source_root),'execution_runtime_root':str(runtime),'input_files':inputs}
            identity_pin=a.atomic(wroot/'worker_identity.json',identity)
            protocol={'schema':a.WORKER_SCHEMA,'identity_sha256':identity_pin,'worker_source_sha256':a.WORKER_SHA,
                      'plan_sha256':plan_pin,'cell_ids':identity['cell_ids'],'input_files':inputs,
                      'arm':worker['arm'],'seed':worker['seed'],'horizon':295,'trace_steps':[1,10,50,200,295]}
            protocol_pin=a.atomic(wroot/'protocol.json',protocol)
            directory=wroot/'cells'/cell['cell_id'];attempt=directory/'attempt_000001';attempt.mkdir(parents=True)
            artifact=attempt/'trace.npz';artifact.write_bytes(b'new-invented-unopened-array')
            row={k:cell[k] for k in ('source_index','trajectory_id','particles','size_group','policy','arm')}
            row.update(training_seed=cell['seed'],objective='faithful',horizon=295,status='failed',failure={'category':'invented'},
                       rng_seed=93000+1000*cell['seed']+cell['source_index'],protocol_sha256=protocol_pin,
                       artifact_file=artifact.name,artifact_sha256=a.sha(artifact),completion_cell_id=cell['cell_id'],
                       original_cell_state=cell['state'],original_directory=cell['original_directory'])
            row_pin=a.atomic(attempt/'row.json',row)
            marker={'schema':a.WORKER_SCHEMA+'_cell_commit','cell_id':cell['cell_id'],'plan_sha256':plan_pin,
                    'identity_sha256':identity_pin,'protocol_sha256':protocol_pin,'row_file':'attempt_000001/row.json',
                    'row_sha256':row_pin,'artifact_file':'attempt_000001/trace.npz','artifact_sha256':a.sha(artifact)}
            a.atomic(directory/'commit.json',marker)
            args=SimpleNamespace(plan=plan_path,ledger=ledger_path,metadata=metadata_path,original_queue_root=original)
            with patch.object(a,'PLAN_SHA',plan_pin),patch.object(a,'LEDGER_SHA',ledger_pin),patch.object(a,'METADATA_SHA',metadata_pin):
                collection,tasks,_=a.collect(args,plan,roots)
                self.assertEqual(len(tasks),332);self.assertEqual(collection['still_missing_outcomes'],1828)
                self.assertEqual(collection['new_committed_outcomes'],1)
                self.assertEqual(sum(len(s['cells']) for s in collection['stages']),2160)
                self.assertEqual(collection['original_autonomous_invocations'],stages)
                selected=next(c for s in collection['stages'] for c in s['cells'] if c['cell_id']==cell['cell_id'])
                self.assertEqual(selected['state'],'recorded_failed_outcome')
                del identity['input_files'][next(k for k in inputs if k!=checkpoint)]
                a.atomic(wroot/'worker_identity.json',identity)
                with self.assertRaisesRegex(ValueError,'complete source/model/data input key set'):
                    a.collect(args,plan,roots)


class ArithmeticTests(unittest.TestCase):
    def test_sparse_saved_full_and_guard_rows(self):
        for n in (0, 10, 250, 295):
            with self.subTest(n=n), tempfile.TemporaryDirectory(dir=TEST_TEMP) as name:
                root = Path(name)
                row, arrays = fixture(n)
                archive = root / 'trace.npz'
                with archive.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                row_path = root / 'row.json'; row_pin = a.atomic(row_path, row)
                task = {'key': 'synthetic', 'arm': 'base', 'seed': 0, 'split': 'test',
                        'item': {k: row[k] for k in ('source_index','trajectory_id','particles','size_group','policy')},
                        'row_file': str(row_path), 'row_sha256': row_pin, 'artifact_file': str(archive),
                        'artifact_sha256': a.sha(archive), 'artifact_bytes': archive.stat().st_size, 'bounds': namespace['BOUNDS'].tolist()}
                result = a.audit_cell(task)
                self.assertEqual(result['status'], 'complete' if n == 295 else 'failed')
                self.assertEqual(result['info']['array_recomputed_mse_forecasts'], [s for s in (1,10,50,200,295) if s <= n])
                self.assertEqual(result['metrics']['mean_rollout_mse'] is None, n < 295)
                self.assertEqual(result['metrics']['mse_forecast200'] is None, n < 200)
                archive.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    a.audit_cell(task)

    def test_full_grid_then_missing_then_guard(self):
        plan = {'splits': {split: {'records': [{'source_index': i, 'id': f'{split}:{i:06d}',
                   'positions': {'shape': [301, 2, 3]}} for i in range(30)]} for split in ('valid', 'test')}}
        stages, tasks, results = [], {}, {}
        base_row, arrays = fixture()
        base_metrics, base_info = saved.audit_rollout(base_row, arrays, namespace['BOUNDS'], saved.Checks())
        for arm, seed in saved.MODELS:
            for split in ('valid', 'test'):
                stage = {'arm': arm, 'seed': seed, 'split': split, 'mode': 'full-rollout',
                         'stage': 'full_rollout_' + split, 'cells': [], 'rows': [], 'outcome': {'synthetic': True}}
                for item in a.schedule(plan, split):
                    for policy in saved.POLICIES:
                        row = {**base_row, **item, 'arm': arm, 'training_seed': seed, 'policy': policy}
                        cell = {**item, 'policy': policy, 'state': 'completed_required_outcome'}
                        key = a.cell_id(arm, seed, split, cell); cell['cell_id'] = key
                        stage['cells'].append(cell);stage['rows'].append(row)
                        tasks[key] = ('pin', {})
                        results[key] = {'key': key, 'status': 'complete', 'failure': None, 'metrics': base_metrics,
                                        'info': base_info, 'pairings': {}, 'checks': 1}
                stages.append(stage)
        class Store:
            def read(self, key, pin):
                return results[key]
        collection = {'stages': stages, 'new_committed_outcomes': 1829, 'still_missing_outcomes': 0}
        audit, summary = a.merge(plan, collection, tasks, Store())
        self.assertEqual(audit['audited_outcomes'], 2160)
        for split in ('valid', 'test'):
            self.assertEqual(summary['stages'][split]['required_outcomes'], 1080)
            self.assertTrue(summary['stages'][split]['all_required_outcomes_complete'])
            contrast = summary['stages'][split]['paired']['mix_minus_base_training']['mean_rollout_mse']['base']
            self.assertEqual(contrast['mean'], 0.)
            self.assertEqual(contrast['defined_seed_pairs'], 3)
        stage = stages[0]; cell = stage['cells'][0]; row = stage['rows'].pop(0); key = cell['cell_id']
        cell['state'] = 'not_completed_before_invocation_end';del tasks[key]
        collection['still_missing_outcomes'] = 1;collection['new_committed_outcomes'] = 1828
        _, missing = a.merge(plan, collection, tasks, Store())
        stats = missing['stages']['valid']['paired']['absolute']['mean_rollout_mse']['base']['base']
        self.assertIsNone(stats['mean']);self.assertEqual(stats['required_seed_pairs'], 3);self.assertEqual(stats['defined_seed_pairs'], 2)
        failed, arr = fixture(250)
        failed.update({k: row[k] for k in ('arm','training_seed','source_index','trajectory_id','policy','size_group')})
        metrics, info = saved.audit_rollout(failed, arr, namespace['BOUNDS'], saved.Checks())
        stage['rows'].insert(0, failed);cell['state'] = 'recorded_failed_outcome';cell['failure'] = failed['failure']
        tasks[key] = ('pin', {});results[key] = {'key': key, 'status': 'failed', 'failure': failed['failure'],
            'metrics': metrics, 'info': info, 'pairings': {}, 'checks': 1}
        collection['still_missing_outcomes'] = 0;collection['new_committed_outcomes'] = 1829
        _, guard = a.merge(plan, collection, tasks, Store())
        self.assertEqual(guard['stages']['valid']['coverage']['recorded_failed_outcome'], 1)
        self.assertEqual(len(guard['stages']['valid']['failed_accepted_prefixes']), 1)
        self.assertIsNone(guard['stages']['valid']['paired']['absolute']['mean_rollout_mse']['base']['base']['mean'])
        self.assertIsNotNone(guard['stages']['valid']['paired']['absolute']['mse_forecast200']['base']['base']['mean'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
