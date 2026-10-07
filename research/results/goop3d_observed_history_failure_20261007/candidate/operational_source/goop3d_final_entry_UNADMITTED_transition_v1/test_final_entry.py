"""Synthetic-only helper tests. No real tools, clocks, processes or products."""
import ast
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import final_entry as f

class FinalEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='d3_final_entry_synthetic_');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.state=self.root/'state'
        for part in ('product_collection','pipeline.transport','postcollection_closure'):(self.state/part).mkdir(parents=True)
        self.state_patch=patch.object(f,'STATE',self.state);self.state_patch.start();self.addCleanup(self.state_patch.stop)
        self.plan=json.loads(f.PLAN.read_bytes())
        self.helper_pin='b'*64;self.proxy_path=self.root/'future_tools.json'
        self.pipeline_launch={'session_id':33822,'output':''};p=self.root/'pipeline_launch.json';pin=self.save(p,self.pipeline_launch)
        self.plan.update(pipeline_original_launch_path=str(p),pipeline_original_launch_sha256=pin)
        self.proxy_launch={'session_id':78014,'output':json.dumps({'synthetic_ready':True})};p=self.root/'proxy_launch.json';pin=self.save(p,self.proxy_launch)
        self.plan.update(phase_proxy_original_launch_path=str(p),phase_proxy_original_launch_sha256=pin)
        transport={'exit_code':0,'failure':None,'signals_to_new_local_group_only':[],'phase_sha256':self.plan['phase_sha256'],'release_sha256':self.plan['release_sha256']}
        self.save(self.state/'pipeline.transport/transport.json',transport)
        ptools={'launch':self.pipeline_launch,'completion':{'exit_code':0,'output':json.dumps(transport)}}
        ppin=self.save(self.state/'product_collection/pipeline_original_tools.json',ptools)
        ctpin=self.save(self.state/'product_collection/transport.json',transport)
        self.colreceipt={'schema':'goop3d_observed_local_collection_receipt_v1','phase_sha256':self.plan['phase_sha256'],'release_sha256':self.plan['release_sha256'],'scientific_admission':False,'files_sha256':{n:'a'*64 for n in f.NAMES},'original_pipeline_tools_sha256':ppin,'original_pipeline_tool_session':33822,'transport_sha256':ctpin}
        cpin=self.save(self.state/'product_collection/local_receipt.json',self.colreceipt)
        self.colpath=self.root/'collection_tools.json';self.coltools={'launch':{'session_id':4001},'completion':{'exit_code':0,'output':json.dumps({'scientific_admission':False,'local_receipt_sha256':cpin})}};self.colpin=self.save(self.colpath,self.coltools)
        ctpin=self.save(self.state/'postcollection_closure/transport.json',transport)
        self.cloreceipt={'schema':'goop3d_postcollection_closure_local_receipt_v1','phase_sha256':self.plan['phase_sha256'],'collection_receipt_sha256':cpin,'collection_tools_sha256':self.colpin,'original_collection_session':4001,'scientific_admission':False,'transport_sha256':ctpin}
        cpin=self.save(self.state/'postcollection_closure/local_receipt.json',self.cloreceipt)
        self.clopath=self.root/'closure_tools.json';self.clotools={'launch':{'session_id':4002},'completion':{'exit_code':0,'output':json.dumps({'scientific_admission':False,'local_receipt_sha256':cpin})}};self.clopin=self.save(self.clopath,self.clotools)

    def save(self,path,value):
        raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw);return f.digest(raw)

    def assemble(self):return f.preassembly(self.plan,self.helper_pin,self.colpath,self.colpin,self.clopath,self.clopin,self.proxy_path)

    def test_preassembly_reads_metadata_only_and_leaves_exact_two(self):
        pre=self.assemble()
        self.assertEqual(set(f.outstanding(pre['final_argv_template'])),f.LAST_TWO)
        self.assertEqual(set(pre['resolved_bindings']),f.BOUND_FIELDS)
        self.assertEqual(pre['actual_pipeline_session'],33822)
        self.assertFalse(pre['scientific_products_read']);self.assertFalse(pre['final_validator_run'])
        self.assertFalse(any((self.state/'product_collection'/n).exists() for n in ('audit.json','summary.json','arithmetic_check.json','pipeline_receipt.json','collection.json')))

    def test_nonzero_collection_exit_rejected(self):
        self.coltools['completion']['exit_code']=1;self.colpin=self.save(self.colpath,self.coltools)
        with self.assertRaises(ValueError):self.assemble()

    def test_running_closure_rejected(self):
        self.clotools['completion']['session_id']=4002;self.clopin=self.save(self.clopath,self.clotools)
        with self.assertRaises(ValueError):self.assemble()

    def test_changed_pipeline_launch_rejected(self):
        path=self.state/'product_collection/pipeline_original_tools.json';doc=json.loads(path.read_bytes());doc['launch']['session_id']=999
        self.colreceipt['original_pipeline_tools_sha256']=self.save(path,doc)
        pin=self.save(self.state/'product_collection/local_receipt.json',self.colreceipt)
        self.coltools['completion']['output']=json.dumps({'scientific_admission':False,'local_receipt_sha256':pin});self.colpin=self.save(self.colpath,self.coltools)
        with self.assertRaises(ValueError):self.assemble()

    def test_altered_bound_metadata_rejected(self):
        pre=self.assemble();path=self.state/'postcollection_closure/transport.json';path.write_text('{}')
        with self.assertRaises(ValueError):self.assemble()

    def test_preassembly_rebuild_identical(self):self.assertEqual(self.assemble(),self.assemble())

    def test_proxy_actual_exit_and_launch_required(self):
        doc={'launch':self.proxy_launch,'completion':{'exit_code':0,'output':''}}
        self.assertEqual(f.proxy_completion(doc,self.proxy_launch),doc['completion'])
        for update in ({'exit_code':None},{'exit_code':1},{'exit_code':0,'session_id':78014}):
            with self.assertRaises(ValueError):f.proxy_completion({'launch':self.proxy_launch,'completion':update},self.proxy_launch)
        changed=copy.deepcopy(doc);changed['launch']['session_id']=78015
        with self.assertRaises(ValueError):f.proxy_completion(changed,self.proxy_launch)

    def test_no_synthetic_synchronous_proxy_substitution(self):
        with self.assertRaises(ValueError):f.proxy_completion({'launch':{'exit_code':0},'completion':None},self.proxy_launch)

    def test_fill_changes_only_two_positions(self):
        pre=self.assemble();template=pre['final_argv_template'];filled=f.fill(template,{k:'c'*64 for k in f.LAST_TWO})
        self.assertEqual(sum(a!=b for a,b in zip(template,filled)),2)
        self.assertTrue(all(type(x)is str for x in filled))
        self.assertEqual(filled[:3],self.plan['final_admission_argv_template'][:3])

    def test_unknown_placeholder_structure_rejected(self):
        with self.assertRaises(ValueError):f.fill([{'required_future_actual_binding':'x','extra':1}],{})

    def final_fixture(self):
        import observed_common as c
        start='2026-10-06T23:31:18.235220+00:00';end='2026-10-07T00:31:18.235220+00:00'
        phase={'schema':c.PHASE_SCHEMA,'issued_by':'root','status':'approved_bounded_observed_history_analysis','dataset':'Goop3D','scope':'all_2568_observed_cells_and_all_4728_accounting','started_utc':start,'stop_utc':end,'seconds':3600,'global_stop_utc':c.GLOBAL_STOP,'original_phase_sha256':c.OLD_PHASE_SHA,'prior_fate_review_sha256':c.FATE_REVIEW_SHA,'original_phase_modified':False,'automatic_retry':False,'hostname':c.HOST,'clock_sample':{'host_boot_id':c.BOOT,'source':'root_fresh_tool_and_host_clock_evidence','error_bound_seconds':5,'host_utc':start,'host_monotonic_seconds':1000.,'root_reference_utc':start},'fresh_native_closure':{'all_42_absent':True,'evidence_sha256':'a'*64}}
        old_phase=self.plan['phase_sha256'];old_anchor=self.plan['anchor_sha256']
        self.plan['phase_sha256']=self.save(self.state/'analysis_phase.json',phase)
        anchor={'phase_sha256':self.plan['phase_sha256'],'utc':start,'monotonic_seconds':1000.}
        self.plan['anchor_sha256']=self.save(self.state/'local_phase_anchor.json',anchor)
        self.plan['final_admission_argv_template']=[self.plan['phase_sha256'] if x==old_phase else self.plan['anchor_sha256'] if x==old_anchor else x for x in self.plan['final_admission_argv_template']]
        route=f.ObservedPhase((self.state/'analysis_phase.json').read_bytes(),(self.state/'local_phase_anchor.json').read_bytes(),self.plan['phase_sha256'],self.plan['anchor_sha256'])
        ready={'schema':'coder_goop3d_observed_phase_route_proxy_v1',**route.metadata,'started_utc':'2026-10-06T23:31:40+00:00','authority':'coder.internal.cohere.com:443','upstream':['100.106.33.61',443],'opaque_tls':True,'no_settings_changed':True,'connection_cap':2048,'concurrency_cap':8,'per_direction_buffer_bytes':262144,'payload_backpressure':'nonblocking_partial_send','url':'http://127.0.0.1:12345'}
        launch={'session_id':78014,'output':json.dumps(ready)}
        self.plan['phase_proxy_original_launch_sha256']=self.save(Path(self.plan['phase_proxy_original_launch_path']),launch)
        proxy={'launch':launch,'completion':{'exit_code':0,'output':''}}
        terminal={**route.metadata,'started_utc':ready['started_utc'],'ended_utc':'2026-10-07T00:30:18.3+00:00','remaining_worker_threads':0,'payloads_logged':False,'remote_processes_signaled':False,'connection_cap':2048,'concurrency_cap':8,'per_direction_buffer_bytes':262144}
        pre={'final_argv_template':f.fill(self.plan['final_admission_argv_template'],{k:'d'*64 for k in f.BOUND_FIELDS})}
        return pre,json.dumps(proxy).encode(),terminal

    def test_final_argv_checks_actual_synthetic_route_and_only_fills_two_hashes(self):
        pre,proxy,terminal=self.final_fixture();raw=json.dumps(terminal).encode();argv=f.final_argv(self.plan,pre,proxy,raw)
        self.assertTrue(all(type(x)is str for x in argv));self.assertEqual(sum(a!=b for a,b in zip(pre['final_argv_template'],argv)),2)
        self.assertEqual(argv[argv.index('--proxy-tools-sha256')+1],f.digest(proxy))
        self.assertEqual(argv[argv.index('--proxy-terminal-sha256')+1],f.digest(raw))

    def test_remaining_proxy_worker_or_late_terminal_rejected(self):
        pre,proxy,terminal=self.final_fixture();terminal['remaining_worker_threads']=1
        with self.assertRaises(ValueError):f.final_argv(self.plan,pre,proxy,json.dumps(terminal).encode())
        terminal['remaining_worker_threads']=0;terminal['ended_utc']='2026-10-07T00:30:58.235221+00:00'
        with self.assertRaises(ValueError):f.final_argv(self.plan,pre,proxy,json.dumps(terminal).encode())

    def test_inert_main_without_explicit_root_flag(self):
        with patch.object(f.sys,'argv',['final_entry.py']),patch('sys.stdout',new_callable=io.StringIO) as out,patch.object(f,'load_reviewed',side_effect=AssertionError('must remain inert')):
            f.main();self.assertEqual(json.loads(out.getvalue())['status'],'inert_original_final_entry')

    def test_source_no_clocks_processes_network_and_only_exec_replacement(self):
        source=Path(f.__file__).read_text();tree=ast.parse(source)
        calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls&{'time.time','time.monotonic','datetime.now','subprocess.run','subprocess.Popen','os.system','socket.socket'})
        self.assertIn('os.execv',calls)
        self.assertNotIn('phase_proxy.py',source)

if __name__=='__main__':unittest.main(verbosity=2)
