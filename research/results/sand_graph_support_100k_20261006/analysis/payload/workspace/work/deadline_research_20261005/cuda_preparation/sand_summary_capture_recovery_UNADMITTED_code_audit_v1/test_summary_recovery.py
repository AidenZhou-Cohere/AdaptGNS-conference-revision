"""Local synthetic wrapper tests; no real clocks, network, outputs or science."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def load():
    spec=importlib.util.spec_from_file_location('synthetic_summary_capture_recovery',HERE/'recover_summary.py')
    m=importlib.util.module_from_spec(spec)
    with patch('subprocess.Popen',side_effect=AssertionError('No process on import')),patch('time.monotonic',side_effect=AssertionError('No clock on import')):
        spec.loader.exec_module(m)
    return m

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()

class SummaryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='sand_summary_capture_synthetic_')
        self.root=Path(self.temporary.name).resolve();self.m=load();self.m.S=self.root
    def tearDown(self):self.temporary.cleanup()
    def fake_operator(self,mutation=None,failure=False):
        m=self.m;calls=[]
        def probe(state,role,payload,tag,stop=None,mono_stop=None):
            calls.append((state,role,payload,tag,stop,mono_stop));dump(state/(tag+'.json'),{'synthetic':'capture'});return {'synthetic':'capture'}
        o=SimpleNamespace(probe=probe,budget=lambda state,tail:calls.append(('budget',state,tail)))
        def close(state,op):
            calls.append(('close_product',state,op));args=[state,'A',{'action':'capture'},op+'.capture','original-stop',123]
            if mutation:mutation(args)
            o.probe(*args)
            if failure:raise ValueError('synthetic frozen validation failure')
            dump(state/(op+'.review.json'),{'synthetic':'validated by frozen close_product'})
        o.close_product=close
        return o,calls
    def test_import_and_description_inert(self):
        self.m.prerequisites=lambda *args:self.fail('No prerequisites without explicit root execution')
        out=io.StringIO()
        with patch.object(sys,'argv',['recover_summary.py']),contextlib.redirect_stdout(out):self.m.main()
        self.assertEqual(json.loads(out.getvalue())['status'],'inert_completed_summary_capture_recovery')
    def test_only_original_summary_capture_is_remapped(self):
        o,calls=self.fake_operator();self.m.capture_original(o,'package','review')
        self.assertEqual(calls[0],('close_product',self.root,'sand_summarize'))
        self.assertEqual(calls[1],(self.root,'A',{'action':'capture'},'sand_summarize.capture_recovery1','original-stop',123))
        receipt=json.loads((self.root/'sand_summarize.recovery_closure_receipt.json').read_text())
        self.assertEqual(receipt['original_worker_session'],19303);self.assertFalse(receipt['new_clock_granted'])
        self.assertEqual(receipt['accepted_root_review_sha256'],self.m.sha(self.root/'sand_summarize.review.json'))
    def test_other_state_role_action_and_tag_are_rejected(self):
        for index,value in ((0,self.root/'other-state'),(1,'B'),(2,{'action':'run'}),(3,'sand_saved_A.capture')):
            with self.subTest(index=index):
                self.m.S=self.root/str(index);self.m.S.mkdir()
                o,calls=self.fake_operator(lambda args:args.__setitem__(index,value))
                with self.assertRaisesRegex(ValueError,'Only original completed summary'):self.m.capture_original(o,'package','review')
                self.assertFalse((self.m.S/'sand_summarize.recovery_closure_receipt.json').exists())
                self.assertEqual(len(calls),1)
    def test_frozen_validation_failure_is_not_success(self):
        original=self.root/'sand_summarize.capture.stderr';original.write_bytes(b'preserved original error')
        o,calls=self.fake_operator(failure=True)
        with self.assertRaisesRegex(ValueError,'frozen validation failure'):self.m.capture_original(o,'package','review')
        self.assertTrue((self.root/'sand_summarize.capture_recovery1.invocation.json').exists())
        self.assertFalse((self.root/'sand_summarize.recovery_closure_receipt.json').exists())
        self.assertEqual(original.read_bytes(),b'preserved original error')
    def test_second_attempt_cannot_overwrite(self):
        o,calls=self.fake_operator();self.m.capture_original(o,'package','review')
        before=(self.root/'sand_summarize.capture_recovery1.invocation.json').read_bytes()
        with self.assertRaises(FileExistsError):self.m.capture_original(o,'newpackage','newreview')
        self.assertEqual((self.root/'sand_summarize.capture_recovery1.invocation.json').read_bytes(),before)
        self.assertEqual(sum(row[0]=='close_product' for row in calls),1)
    def test_parser_has_no_operation_state_action_or_command_override(self):
        self.m.prerequisites=lambda *args:self.fail('No prerequisites after invalid parser input')
        for flag in ('--operation','--state','--action','--command'):
            with self.subTest(flag=flag),patch.object(sys,'argv',['recover_summary.py','--root-execute',flag,'arbitrary']),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:self.m.main()
                self.assertEqual(result.exception.code,2)
    def preconditions(self):
        m=self.m;m.K=self.root/'package';m.K.mkdir();(m.K/'recover_summary.py').write_bytes((HERE/'recover_summary.py').read_bytes())
        package_pin=dump(m.K/'manifest.json',{'files_sha256':{'recover_summary.py':m.sha(m.K/'recover_summary.py')}})
        review=self.root/'independent-review.json';review_pin=dump(review,{'schema':'sand_summary_capture_recovery_independent_review_v1','status':'passed_source_and_synthetic_review','source_manifest_sha256':package_pin})
        failed={'exit_code':255,'local_transport_reaped':True,'local_transport_timeout':False,'signals_to_own_local_group':[]}
        dump(self.root/'sand_summarize.capture.external.json',failed)
        (self.root/'sand_summarize.capture.stderr').write_text('proxyconnect tcp: dial tcp 127.0.0.1:56069: connect: operation not permitted')
        (self.root/'sand_summarize.capture.stdout').write_bytes(b'')
        m.PH=dump(self.root/'analysis_phase.json',{'synthetic':'original phase'})
        self.bind_failure()
        self.record={'original_tool_session':19303,'original_tool_exit_code':0}
        self.budgets=[];o=SimpleNamespace(original_exit=lambda *args:self.record,budget=lambda *args:self.budgets.append(args))
        base=SimpleNamespace(fixed_preconditions=lambda *args:o)
        m.load_base=lambda:base
        return package_pin,review,review_pin
    def bind_failure(self):
        dump(self.m.K/'failure_bindings.json',{'files_sha256':{str(self.root/('sand_summarize.capture.'+n)):self.m.sha(self.root/('sand_summarize.capture.'+n)) for n in ('external.json','stderr','stdout')}})
    def test_exact_original_summary_exit_and_failure_preconditions(self):
        args=self.preconditions();self.m.prerequisites(*args)
        self.assertEqual(self.budgets,[(self.root,20)])
    def test_wrong_original_summary_session_rejected(self):
        args=self.preconditions();self.record['original_tool_session']=19304
        with self.assertRaisesRegex(ValueError,'Genuine original summary19303'):self.m.prerequisites(*args)
    def test_prior_decoded_summary_refuses_retrieval(self):
        args=self.preconditions();(self.root/'sand_summarize.collected').mkdir()
        with self.assertRaisesRegex(ValueError,'remain fresh'):self.m.prerequisites(*args)
    def test_nonempty_original_stdout_not_preconnection_failure(self):
        args=self.preconditions();(self.root/'sand_summarize.capture.stdout').write_bytes(b'partial remote response');self.bind_failure()
        with self.assertRaisesRegex(ValueError,'pre-connection sandbox failure'):self.m.prerequisites(*args)
    def test_changed_original_failure_bytes_rejected(self):
        args=self.preconditions();(self.root/'sand_summarize.capture.stderr').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Original summary failure/exit evidence changed'):self.m.prerequisites(*args)

if __name__=='__main__':unittest.main(verbosity=2)
