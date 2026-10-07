"""Focused synthetic boundaries; no worker/clock/native probe/science file access."""
import ast
import copy
import datetime as D
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import observed_common as c
import audit_goop3d_observed_histories_v1 as audit
import goop3d_observed_history_arithmetic_v1 as arithmetic
import summarize_goop3d_observed_histories_v1 as summarize
import check_goop3d_observed_history_summary_v1 as checker
import phase_binding

def phase_fixture():
    return {'schema':c.PHASE_SCHEMA,'issued_by':'root','status':'approved_bounded_observed_history_analysis','dataset':'Goop3D','scope':'all_2568_observed_cells_and_all_4728_accounting','seconds':3600,'started_utc':'2026-10-06T23:00:00+00:00','stop_utc':'2026-10-07T00:00:00+00:00','global_stop_utc':c.GLOBAL_STOP,'original_phase_sha256':c.OLD_PHASE_SHA,'prior_fate_review_sha256':c.FATE_REVIEW_SHA,'original_phase_modified':False,'automatic_retry':False,'hostname':c.HOST,'clock_sample':{'host_boot_id':c.BOOT,'source':'root_fresh_tool_and_host_clock_evidence','error_bound_seconds':5,'host_utc':'2026-10-06T23:00:00+00:00','host_monotonic_seconds':10000.,'root_reference_utc':'2026-10-06T23:00:00+00:00'},'fresh_native_closure':{'all_42_absent':True,'evidence_sha256':'a'*64}}

def synthetic_audit(incomplete=False):
    same=[{'source_index':i,'target_frame':t} for i in range(30) for t in (7,80,153,226,300)]
    clean=[{'source_index':j//5,'target_frame':6+j%5} for j in range(128)]
    schedules={s:{'same-state':same,'clean-validation':clean} for s in ('valid','test')}
    models=[];all_accounting=[]
    policies=summarize.POLICIES
    for arm in ('base','mix'):
        for seed in range(3):
            for stage in c.OBSERVED:
                mode='clean-validation' if stage=='clean_validation' else 'same-state';split='test' if stage=='same_state_test' else 'valid'
                schedule=schedules[split][mode];cells=[];rows=[]
                keys=['metrics/test_metric'] if mode=='clean-validation' else ['accuracy/'+p+'/'+m for p in policies for m in ('position_coordinate_mse','normalized_coordinate_mse')]
                for n,item in enumerate(schedule):
                    missing=incomplete and (arm,seed,stage,n)==('base',0,'same_state_valid',0)
                    cells.append({**item,'state':'not_completed_before_invocation_end' if missing else 'completed_required_outcome'})
                    if missing:continue
                    metrics={}
                    for key in keys:
                        v=(2 if arm=='mix' else 0)+seed+item['source_index']/1000+item['target_frame']/10000
                        if mode=='same-state':
                            policy=key.split('/')[1];v+=policies.index(policy)/100
                            if policy==policies[-1] and arm=='mix':v+=.5
                        metrics[key]=v
                    rows.append({'unit':[item['source_index'],item['target_frame']],'status':'complete','metrics':metrics})
                values={tuple(r['unit']):r['metrics'] for r in rows}
                aggregates={k:arithmetic.aggregate([(x['source_index'],x['target_frame']) for x in schedule],{u:m[k] for u,m in values.items()}) for k in keys}
                model={'arm':arm,'seed':seed,'stage':stage,'mode':mode,'split':split,'cells':cells,'coverage':dict(audit.Counter(x['state'] for x in cells)),'rows':rows,'aggregates':aggregates}
                models.append(model);all_accounting.append({'arm':arm,'seed':seed,'stage':stage,'cells':cells})
    states=['completed_required_outcome']*(331+int(incomplete))+['timed_out_current']*12+['not_completed_before_invocation_end']*(1817-int(incomplete))
    for k in range(12):all_accounting.append({'arm':'base' if k<6 else 'mix','seed':(k//2)%3,'stage':'full_rollout_valid' if k%2==0 else 'full_rollout_test','cells':[{'state':x} for x in states[k*180:(k+1)*180]]})
    return {'schema':c.AUDIT_SCHEMA,'status':'passed_scoped_observed_history_checks','collection_sha256':c.COLLECTION_SHA,'cohort_sha256':c.COHORT_SHA,'phase_sha256':'b'*64,'required_all_cells':4728,'required_observed_cells':2568,'source_schedules':schedules,'models':models,'all_original_accounting':all_accounting,'limitations':[]}

class NumericScope(unittest.TestCase):
    def test_complete_fixed_grid_independent_arithmetic(self):
        a=synthetic_audit();s=summarize.summarize(a);r=checker.verify(a,s)
        self.assertEqual(r['status'],'passed_independent_observed_arithmetic')
        for name in ('same_state_valid','same_state_test'):
            v=s['families'][name]['accuracy_policy_contrasts']['position_coordinate_mse']['risk_minus_random_mix_minus_base']
            self.assertAlmostEqual(v['mean'],.5);self.assertAlmostEqual(v['sample_sd'],0)
    def test_missing_cell_nulls_whole_family(self):
        a=synthetic_audit(True);s=summarize.summarize(a);checker.verify(a,s)
        f=s['families']['same_state_valid'];self.assertFalse(f['full_family_complete'])
        self.assertTrue(all(v['mean'] is None and v['defined_seed_pairs']==0 for v in f['mix_minus_base_training'].values()))
        self.assertTrue(s['families']['same_state_test']['full_family_complete'])
    def test_summary_tampering_refused(self):
        a=synthetic_audit();s=summarize.summarize(a);next(iter(s['families']['clean_validation']['mix_minus_base_training'].values()))['mean']+=1
        with self.assertRaises(ValueError):checker.verify(a,s)
    def test_missing_seed_refused(self):
        a=synthetic_audit();a['models'].pop()
        with self.assertRaises(ValueError):summarize.summarize(a)
    def test_old_full_audit_schema_refused(self):
        a=synthetic_audit();a['schema']='goop3d_saved_array_audit_v1'
        with self.assertRaises(ValueError):summarize.summarize(a)
    def test_undefined_seed_propagates(self):
        self.assertIsNone(summarize.stats([1,None,3])['mean'])
        self.assertIsNone(checker.seed_values([1,None,3])['sample_sd'])

class FileBounds(unittest.TestCase):
    def test_collection_known_bytes_fit(self):self.assertLess(c.COLLECTION_BYTES,c.CAP_COLLECTION);self.assertEqual(c.CAP_COLLECTION,1<<30)
    def test_pinned_regular_read(self):
        with tempfile.TemporaryDirectory() as d:
            p=(Path(d)/'x').resolve();p.write_bytes(b'abc');h=hashlib.sha256(b'abc').hexdigest();self.assertEqual(c.read_bound(p,h,3,expected_bytes=3),b'abc')
    def test_wrong_size_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=(Path(d)/'x').resolve();p.write_bytes(b'abc')
            with self.assertRaises(ValueError):c.read_bound(p,hashlib.sha256(b'abc').hexdigest(),10,expected_bytes=2)
    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=(Path(d)/'x').resolve();p.write_bytes(b'abc');q=p.with_name('link');q.symlink_to(p)
            with self.assertRaises(ValueError):c.read_bound(q,hashlib.sha256(b'abc').hexdigest(),10)
    def test_numeric_archive(self):
        b=io.BytesIO();np.savez(b,x=np.arange(3));self.assertEqual(audit.numeric_archive(b.getvalue())['x'].tolist(),[0,1,2])
    def test_object_archive_refused(self):
        b=io.BytesIO();np.savez(b,x=np.array([object()],dtype=object))
        with self.assertRaises(ValueError):audit.numeric_archive(b.getvalue())
    def test_archive_subpath_refused(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('../x.npy',b'abc')
        with self.assertRaises(ValueError):audit.numeric_archive(b.getvalue())
    def test_decoded_archive_cap(self):
        b=io.BytesIO();np.savez(b,x=np.arange(3))
        with patch.object(audit,'CAP_ARCHIVE',1):
            with self.assertRaises(ValueError):audit.numeric_archive(b.getvalue())

class PhaseBounds(unittest.TestCase):
    def test_distinct_full_hour(self):self.assertEqual((c.validate_phase(phase_fixture())[1]-c.validate_phase(phase_fixture())[0]).total_seconds(),3600)
    def test_partial_hour_refused(self):
        p=phase_fixture();p['seconds']=100
        with self.assertRaises(ValueError):c.validate_phase(p)
    def test_global_extension_refused(self):
        p=phase_fixture();p['started_utc']='2026-10-07T00:01:00+00:00';p['stop_utc']='2026-10-07T01:01:00+00:00'
        with self.assertRaises(ValueError):c.validate_phase(p)
    def test_old_phase_reset_refused(self):
        p=phase_fixture();p['original_phase_modified']=True
        with self.assertRaises(ValueError):c.validate_phase(p)
    def test_route_has_sixty_second_tail(self):
        p=phase_fixture();raw=json.dumps(p).encode();ph=c.digest(raw);anchor=json.dumps({'phase_sha256':ph,'utc':p['started_utc'],'monotonic_seconds':200.}).encode();route=phase_binding.ObservedPhase(raw,anchor,ph,c.digest(anchor))
        with patch.object(phase_binding.time,'time',return_value=c.stamp(p['started_utc']).timestamp()+100),patch.object(phase_binding.time,'monotonic',return_value=300.):self.assertEqual(route.remaining(),3440)
        with patch.object(phase_binding.time,'time',return_value=c.stamp(p['started_utc']).timestamp()+3540),patch.object(phase_binding.time,'monotonic',return_value=3740.):self.assertTrue(route.expired())
    def test_route_clock_disagreement_refused(self):
        p=phase_fixture();raw=json.dumps(p).encode();ph=c.digest(raw);anchor=json.dumps({'phase_sha256':ph,'utc':p['started_utc'],'monotonic_seconds':200.}).encode();route=phase_binding.ObservedPhase(raw,anchor,ph,c.digest(anchor))
        with patch.object(phase_binding.time,'time',return_value=c.stamp(p['started_utc']).timestamp()+100),patch.object(phase_binding.time,'monotonic',return_value=500.):self.assertTrue(route.expired())

if __name__=='__main__':unittest.main()
