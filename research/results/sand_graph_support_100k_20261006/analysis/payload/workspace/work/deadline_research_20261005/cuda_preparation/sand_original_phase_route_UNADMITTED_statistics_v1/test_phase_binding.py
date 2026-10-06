"""Synthetic phase/anchor and wrapper boundaries; no actual phase or live action."""
import copy
import datetime as D
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import phase_binding as B
import root_operator_with_proxy as W

ORIGIN=1791327600.0
MONO=10000.

def records():
    start=D.datetime.fromtimestamp(ORIGIN,D.timezone.utc)
    p={'schema':'adaptgns_sand_original_analysis_phase_v1','issued_by':'root','status':'approved_original_analysis_phase','dataset':'Sand','original_analysis_seconds':3600,'new_or_restarted_clock_granted':False,'original_analysis_started_utc':start.isoformat(),'original_analysis_stop_utc':(start+D.timedelta(seconds=3600)).isoformat(),'global_analysis_deadline_utc':B.GLOBAL_STOP,'original_evaluation_sessions':{'A':51595,'B':12559},'original_evaluation_closures':{role:{'path':'/root/repos/AdaptGNS-cuda-20261006/sand_final_analysis_20261006_v1/controls/'+role+'.evaluation_closure.json','sha256':'a'*64} for role in ('A','B')}}
    a={'utc':start.isoformat(),'monotonic_seconds':MONO}
    return p,a

def build(p=None,a=None):
    if p is None: p,a=records()
    raw=json.dumps(p).encode(); a=copy.deepcopy(a); a.setdefault('phase_sha256',B.sha(raw)); anchor=json.dumps(a).encode()
    return B.OriginalPhase(raw,anchor,B.sha(raw),B.sha(anchor))

def ready(phase):
    return {'schema':'coder_sand_original_phase_route_proxy_v1',**phase.metadata,'authority':'coder.internal.cohere.com:443','upstream':['100.106.33.61',443],'opaque_tls':True,'no_settings_changed':True,'started_utc':D.datetime.fromtimestamp(ORIGIN+20,D.timezone.utc).isoformat(),'url':'http://127.0.0.1:34567'}

class PhaseBoundaries(unittest.TestCase):
    def test_original_remaining_consumes_startup_elapsed(self):
        phase=build()
        with patch.object(B.time,'time',return_value=ORIGIN+123),patch.object(B.time,'monotonic',return_value=MONO+123):
            self.assertEqual(phase.remaining(),3477)
        with patch.object(B.time,'time',return_value=ORIGIN+500),patch.object(B.time,'monotonic',return_value=MONO+500):
            self.assertEqual(phase.remaining(),3100)
            self.assertEqual(phase.metadata['original_monotonic_stop'],MONO+3600)
    def test_expired_future_and_clock_disagreement_refused(self):
        phase=build()
        for wall,mono in [(3600,3600),(3601,3601),(-1,-1),(1,7),(7,1),(float('inf'),1)]:
            with self.subTest(wall=wall,mono=mono),patch.object(B.time,'time',return_value=ORIGIN+wall),patch.object(B.time,'monotonic',return_value=MONO+mono):
                with self.assertRaises(ValueError): phase.remaining()
                self.assertTrue(phase.expired())
    def test_phase_schema_original_hour_and_session_guards(self):
        for field,value in [('status','template_not_admitted'),('issued_by',None),('dataset','Goop'),('original_analysis_seconds',7200),('original_analysis_seconds',True),('new_or_restarted_clock_granted',True),('original_evaluation_sessions',{'A':True,'B':12559})]:
            p,a=records();p[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError): build(p,a)
        p,a=records();p['original_analysis_stop_utc']=D.datetime.fromtimestamp(ORIGIN+3601,D.timezone.utc).isoformat()
        with self.assertRaises(ValueError):build(p,a)
        p,a=records();p['original_analysis_started_utc']='2026-10-07T04:00:00+00:00';p['original_analysis_stop_utc']='2026-10-07T05:00:00+00:00';a['utc']=p['original_analysis_started_utc']
        with self.assertRaises(ValueError):build(p,a)
    def test_phase_anchor_raw_hash_and_link_guards(self):
        p,a=records();raw=json.dumps(p).encode();a['phase_sha256']=B.sha(raw);anchor=json.dumps(a).encode()
        for phasepin,anchorpin in [('f'*64,B.sha(anchor)),(B.sha(raw),'f'*64)]:
            with self.assertRaises(ValueError):B.OriginalPhase(raw,anchor,phasepin,anchorpin)
        for field,value in [('phase_sha256','f'*64),('utc','2026-10-06T00:00:00+00:00'),('monotonic_seconds',True),('monotonic_seconds',float('inf'))]:
            p,a=records();a[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):build(p,a)
    def test_missing_actual_phase_refused_without_other_reads(self):
        with patch.object(B,'read_pinned',side_effect=FileNotFoundError()) as read:
            with self.assertRaises(FileNotFoundError):B.load_original_phase('a'*64,'b'*64)
            read.assert_called_once_with(B.PHASE,'a'*64)
    def test_wrapper_fixed_original_operator_and_postbegin_only(self):
        argv=W.operator_argv('run',['--','--operation','sand_collect_A'])
        self.assertEqual(argv[:4],[W.sys.executable,'-B',str(B.OPERATOR),'run']);self.assertNotIn('-O',argv);self.assertNotIn('-OO',argv);self.assertIn(B.FROZEN_PACKAGE,argv);self.assertIn(str(B.STATE),argv)
        for action in ['begin','prepare','close-evaluation']:
            with self.assertRaises(ValueError):W.operator_argv(action,[])
        for extra in [['--state','/tmp/other'],['--state=/tmp/other'],['--root-action'],['--package-sha256','a'*64]]:
            with self.assertRaises(ValueError):W.operator_argv('run',extra)
    def test_wrapper_route_phase_pin_and_environment_scope(self):
        phase=build();r=ready(phase)
        with patch.object(B.time,'time',return_value=ORIGIN+30),patch.object(B.time,'monotonic',return_value=MONO+30),patch.dict(W.os.environ,{'KEEP':'synthetic','NO_PROXY':'*'},clear=True):
            env=W.ready_environment(json.dumps(r).encode(),phase)
            self.assertEqual(env['HTTPS_PROXY'],'http://127.0.0.1:34567');self.assertEqual(env['KEEP'],'synthetic');self.assertNotIn('NO_PROXY',env);self.assertEqual(W.os.environ['NO_PROXY'],'*')
            for field,value in [('analysis_phase_sha256','f'*64),('upstream',['127.0.0.1',443]),('url','http://u:p@127.0.0.1:34567'),('url','http://other:34567'),('new_or_restarted_clock_granted',True)]:
                changed=copy.deepcopy(r);changed[field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):W.ready_environment(json.dumps(changed).encode(),phase)
        with patch.object(B.time,'time',return_value=ORIGIN+3600),patch.object(B.time,'monotonic',return_value=MONO+3600):
            with self.assertRaises(ValueError):W.ready_environment(json.dumps(r).encode(),phase)

if __name__=='__main__':unittest.main(verbosity=2)
