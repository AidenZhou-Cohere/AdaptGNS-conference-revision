"""Synthetic exact owner maps, collector closure, original tools and route gates."""
import copy
import datetime as D
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch
import owner_evidence as e
import observe_collection_closure as observer
from root_observe_collection_closure import validate_closure
from root_run import route_environment
from operational_checks import completed_tool,verify_fresh_admission
import prephase_sources

HERE=Path(__file__).resolve().parent

def native(pid,ppid,argv,executable='/usr/bin/python3.12',pgid=70001):
    return {'pid':pid,'ppid':ppid,'pgid':pgid,'sid':pgid,'start_id':str(pid*10),'state':'S','argv':argv,'executable':executable}

def fixture():
    start=D.datetime.fromisoformat('2026-10-06T23:00:00+00:00');iso=lambda sec:(start+D.timedelta(seconds=sec)).isoformat()
    phase_pin,release_pin='1'*64,'2'*64
    pins={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('observed_operations.json','supervise_observed_history_cpu_v1.py','launch_observed_history_cpu_v1.py')}
    spec=json.loads((HERE/'observed_operations.json').read_bytes())['operations']['d3_observed_history_pipeline']
    phase={'started_utc':iso(0),'stop_utc':iso(3600),'clock_sample':{'host_monotonic_seconds':1000},'operational_evidence':{'synthetic':{'path':e.OUTPUT+'/controls/synthetic.json','sha256':'3'*64}}}
    controls={p:phase_pin if h is None else h for p,h in spec['inputs_sha256'].items()}
    command=[controls[v['sha256_of_input']] if isinstance(v,dict) else v for v in spec['argv']]
    release={'schema':'goop3d_observed_history_cpu_release_v1','operation':'d3_observed_history_pipeline','dataset':'Goop3D','host_role':'D3','hostname':e.HOST,'operation_spec':{'path':e.SOURCE+'/observed_operations.json','sha256':pins['observed_operations.json']},'analysis_phase':{'path':e.OUTPUT+'/controls/analysis_phase.json','sha256':phase_pin},'owner_source':{'path':e.SOURCE+'/supervise_observed_history_cpu_v1.py','sha256':pins['supervise_observed_history_cpu_v1.py']},'bootstrap_source':{'path':e.SOURCE+'/launch_observed_history_cpu_v1.py','sha256':pins['launch_observed_history_cpu_v1.py']},'inputs_sha256':controls,'protected_tree_entries':{},'command':command,'clock_sample':phase['clock_sample'],'outer_timeout':{'path':'/usr/bin/timeout','sha256':e.TIMEOUT_SHA,'version':'timeout (GNU coreutils) 9.4','duration_rule':'floor((original_hard_deadline_ns-actual_bootstrap_monotonic_ns)/1e9)-15-5'},'owner_output_dir':e.OUTPUT+'/owners/d3_observed_history_pipeline','hard_deadline_monotonic_ns':3995*10**9,'publication_deadline_utc':iso(3000)}
    hard=release['hard_deadline_monotonic_ns'];computed=1000*10**9;seconds=2975
    guard={'clock':'CLOCK_MONOTONIC','signal':'SIGKILL','absolute_deadline_ns':hard,'armed_monotonic_ns':1001*10**9,'deleted_before_exit':False}
    timing={'original_absolute_hard_deadline_ns':hard,'owner_guard_armed_monotonic_ns':guard['armed_monotonic_ns'],'computed_monotonic_ns':computed,'term_seconds':seconds,'earliest_relative_kill_monotonic_ns':computed+(seconds+15)*10**9,'bootstrap_allowance_seconds':5,'bootstrap_process_start':{'lower_ns':computed-10,'upper_ns':computed}}
    outer_argv=['/usr/bin/timeout','--signal=TERM','--kill-after=15s',str(seconds)+'s',e.PYTHON,'-I','-S','-B',release['owner_source']['path'],'--execute','--hard-deadline-monotonic-ns',str(hard),'--outer-term-seconds',str(seconds),'--outer-computed-monotonic-ns',str(computed),'--release',e.OUTPUT+'/controls/pipeline.cpu_release.json','--release-sha256',release_pin]
    outer=native(70001,1,outer_argv,'/usr/bin/timeout');own=native(70002,70001,outer_argv[4:]);worker=native(70003,70002,command)
    child={'pid':70003,'identity':worker,'command':command,'exit_code':0,'reaped':True,'signals':[],'cleanup_errors':[]}
    files={n:hashlib.sha256(n.encode()).hexdigest() for n in (*e.PRODUCTS,*e.EVIDENCE,'owner_terminal.json')}
    terminal={'schema':'adaptgns_stopped_analysis_cpu_terminal_v1','status':'complete','failure':None,'release_sha256':release_pin,'scientific_admission':False,'root_original_tool_exit_and_timeout_owner_child_native_closure_required':True,'input_sha256':e.expected_inputs(release,phase,phase_pin,release_pin,spec,pins),'evidence_sha256':{release['owner_output_dir']+'/'+n:files[n] for n in e.EVIDENCE},'output_sha256':{e.OUTPUT+'/'+n:files[n] for n in e.PRODUCTS},'child':child,'child_native_absent':True,'owner_identity':own,'outer_timeout_identity':outer,'native_hard_guard':guard,'native_outer_timing':timing,'publication_utc':iso(2900),'publication_monotonic':3900}
    started={'owner_identity':own,'outer_timeout_identity':outer,'release_sha256':release_pin,'scientific_admission':False,'original_clock_sample':phase['clock_sample'],'native_hard_guard':guard,'native_outer_timing':timing}
    registered={'pid':70003,'identity':worker,'owner_identity':own}
    return {'data':{'owner_terminal.json':terminal,'owner_started.json':started,'child_registered.json':registered},'files':files,'release':release,'phase':phase,'phase_pin':phase_pin,'release_pin':release_pin,'spec':spec,'source_pins':pins,'transport':{'finished_utc':iso(2910)}}

class EvidenceTests(unittest.TestCase):
    def test_exact_complete_owner_maps_and_native_contract(self):
        self.assertEqual(set(e.validate_owner(**fixture())),{'outer','owner','worker'})
    def test_missing_or_extra_input_and_evidence_bindings_fail(self):
        for field,path in [('input_sha256',e.PYTHON_REAL),('evidence_sha256',e.OUTPUT+'/owners/d3_observed_history_pipeline/child.stderr')]:
            for mode in ('missing','changed','extra'):
                with self.subTest(field=field,mode=mode):
                    f=fixture();mapping=f['data']['owner_terminal.json'][field]
                    if mode=='missing':del mapping[path]
                    elif mode=='changed':mapping[path]='f'*64
                    else:mapping['/unexpected']='f'*64
                    with self.assertRaises(ValueError):e.validate_owner(**f)
    def test_changed_native_command_or_guard_fails(self):
        f=fixture();f['data']['owner_terminal.json']['outer_timeout_identity']['argv'][3]='3600s'
        with self.assertRaises(ValueError):e.validate_owner(**f)
        f=fixture();f['data']['owner_terminal.json']['native_hard_guard']['absolute_deadline_ns']+=1
        with self.assertRaises(ValueError):e.validate_owner(**f)
    def test_changed_release_spec_or_clock_fails(self):
        f=fixture();f['release']['operation_spec']['sha256']='f'*64
        with self.assertRaises(ValueError):e.validate_owner(**f)
        f=fixture();f['release']['clock_sample']={'host_monotonic_seconds':1001}
        with self.assertRaises(ValueError):e.validate_owner(**f)
    def closure(self):
        f=fixture();owner=f['data']['owner_terminal.json'];ids={'outer':owner['outer_timeout_identity'],'owner':owner['owner_identity'],'worker':owner['child']['identity']}
        outer=native(70004,1,['/usr/bin/timeout','--signal=KILL','20s','synthetic-collector'],'/usr/bin/timeout',70004);collector=native(70005,70004,['synthetic-collector'],pgid=70004)
        collected={'schema':'goop3d_observed_products_collection_v1','status':'four_products_and_owner_closure_collected','host':e.HOST,'boot_id':observer.BOOT,'phase_sha256':f['phase_pin'],'release_sha256':f['release_pin'],'new_owner_identities':ids,'collector_identity':collector,'collector_parent_identity':outer,'finished_utc':'2026-10-06T23:55:00+00:00'}
        identities,expected=observer.expected_identities(collected)
        report={'schema':'goop3d_observed_postcollection_native_closure_v1','status':'all47_original_pids_absent_twice','phase_sha256':f['phase_pin'],'collection_report_sha256':'4'*64,'release_sha256':f['release_pin'],'host':e.HOST,'boot_id':observer.BOOT,'original_identities':copy.deepcopy(identities),'native_first_absent':{str(p):True for p in expected},'native_second_absent':{str(p):True for p in expected},'finished_utc':'2026-10-06T23:56:00+00:00','no_files_written':True,'scientific_program_run':False,'original_result_queue_opened':False}
        return collected,report,f['phase_pin']
    def test_exact47_postcollection_closure(self):
        collected,report,pin=self.closure();self.assertEqual(len(validate_closure(report,collected,pin,'4'*64)),47)
    def test_missing_collector_or_changed_start_identity_fails(self):
        collected,report,pin=self.closure();del report['native_second_absent']['70005']
        with self.assertRaises(ValueError):validate_closure(report,collected,pin,'4'*64)
        collected,report,pin=self.closure();report['original_identities']['collector']['start_id']='new'
        with self.assertRaises(ValueError):validate_closure(report,collected,pin,'4'*64)
    def test_reused_collector_pid_or_premature_observation_fails(self):
        collected,report,pin=self.closure();collected['collector_identity']['pid']=next(iter(observer.HISTORICAL))
        with self.assertRaises(ValueError):observer.expected_identities(collected)
        collected,report,pin=self.closure();report['finished_utc']='2026-10-06T23:54:00+00:00'
        with self.assertRaises(ValueError):validate_closure(report,collected,pin,'4'*64)
    def test_original_tools_require_actual_completion(self):
        self.assertEqual(completed_tool({'launch':{'session_id':1},'completion':{'exit_code':0}})[0],1)
        for done in ({'exit_code':None},{'exit_code':0,'session_id':1},{'exit_code':1}):
            with self.assertRaises(ValueError):completed_tool({'launch':{'session_id':1},'completion':done})
        self.assertEqual(completed_tool({'launch':{'exit_code':0,'output':'original complete output'},'completion':None}),(None,{'exit_code':0,'output':'original complete output'}))
        for launch in ({'session_id':1},{'session_id':1,'exit_code':0},{'exit_code':1},{'exit_code':None}):
            with self.assertRaises(ValueError):completed_tool({'launch':launch,'completion':None})
    def test_exact_d3_route_limits(self):
        phase=SimpleNamespace(metadata={'synthetic_phase_binding':'fixed'})
        route={'schema':'coder_goop3d_observed_phase_route_proxy_v1',**phase.metadata,'authority':'coder.internal.cohere.com:443','upstream':['100.106.33.61',443],'opaque_tls':True,'no_settings_changed':True,'connection_cap':2048,'concurrency_cap':8,'per_direction_buffer_bytes':262144,'payload_backpressure':'nonblocking_partial_send','url':'http://127.0.0.1:12345'}
        self.assertEqual(route_environment(route,phase)['HTTPS_PROXY'],route['url'])
        for key,value in [('connection_cap',128),('per_direction_buffer_bytes',524288),('synthetic_phase_binding','changed')]:
            with self.assertRaises(ValueError):route_environment({**route,key:value},phase)
    def test_fresh_admission_requires_actual_metadata_proxy_exit_and_closure(self):
        with tempfile.TemporaryDirectory(prefix='d3_fresh_admission_synthetic_') as directory:
            root=Path(directory).resolve();proxy_dir=root/'proxy';proxy_dir.mkdir()
            def saved(path,value):
                raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw)
                return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
            now=D.datetime.fromisoformat('2026-10-06T23:00:41+00:00');report={'synthetic':'exact native report'}
            transport={'remote_report':report,'exit_code':0,'failure':None,'cleanup_errors':[],'signals_to_own_new_local_transport_group':[],'local_transport_reaped':True,'pre_publication_local_budget_preserved':True,'elapsed_monotonic_seconds':1.0,'elapsed_wall_seconds':1.0,'local_transport_pid':80001,'observed_utc':'2026-10-06T23:00:01+00:00'}
            tbind=saved(root/'transport.json',transport)
            terminal={'lifetime_seconds':40,'remaining_worker_threads':0,'payloads_logged':False,'remote_processes_signaled':False,'ended_utc':'2026-10-06T23:00:40+00:00'}
            terminal_bind=saved(proxy_dir/'terminal.json',terminal)
            proxy_tools={'launch':{'session_id':12,'output':json.dumps({'schema':'coder_exact_route_proxy_v1','lifetime_seconds':40})},'completion':{'exit_code':0},'terminal':terminal_bind}
            proxy_bind=saved(root/'proxy_tools.json',proxy_tools)
            local={'ps_exit_code':0,'parser_self_verified':True,'new_observation_pgid':80001,'matching_rows':[],'utc':'2026-10-06T23:00:40.5+00:00','proxy_directory':str(proxy_dir),'proxy_program':'synthetic_proxy.py'}
            local_bind=saved(root/'local.json',local)
            doc={'schema':'goop3d_fresh_observation_admission_v1','launch':{'session_id':11},'completion':{'exit_code':0,'output':json.dumps({'transport_sha256':tbind['sha256'],'scientific_admission':False})},'transport':tbind,'local_native_closure':local_bind,'metadata_proxy':{'original_tools':proxy_bind,'terminal':terminal_bind,'original_session':12}}
            with patch.object(prephase_sources,'PROXY_OUTPUT',proxy_dir),patch.object(prephase_sources,'PROXY',root/'synthetic_proxy.py'):
                accepted=verify_fresh_admission(doc,report,now)
                self.assertEqual(accepted['original_metadata_proxy_tool_session'],12)
                proxy_tools['completion']={'exit_code':1};doc['metadata_proxy']['original_tools']=saved(root/'proxy_tools.json',proxy_tools)
                with self.assertRaises(ValueError):verify_fresh_admission(doc,report,now)

if __name__=='__main__':unittest.main(verbosity=2)
