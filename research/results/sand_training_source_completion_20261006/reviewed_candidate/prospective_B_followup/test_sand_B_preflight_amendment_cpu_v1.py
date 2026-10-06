"""Only NEW source-owner contracts and fake processes; no scientific execution."""
import ast
import copy
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'supervise_sand_B_preflight_amendment_cpu_v1.py'
spec=importlib.util.spec_from_file_location('new_reserved_cpu_test',SOURCE)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
boot_spec=importlib.util.spec_from_file_location('new_reserved_boot_test',HERE/'launch_sand_B_preflight_amendment_cpu_v1.py')
BOOT=importlib.util.module_from_spec(boot_spec);boot_spec.loader.exec_module(BOOT)
BASE=datetime(2026,10,6,18,tzinfo=timezone.utc)


def declaration(path,name):
    tree=ast.parse(Path(path).read_text())
    return next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)


# Reuse only the already-reviewed fake process classes as AST, with the NEW
# source module injected; importing/executing old test modules is unnecessary.
ns={'M':M,'BASE':BASE,'copy':copy,'signal':signal,'timedelta':timedelta}
for name in ('FakeRuntime','FakeBudget'):
    node=declaration(HERE/'test_sand_post_completion_cpu_owner_transition_v2.py',name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<fake-runtime-only>','exec'),ns)
FakeRuntime=ns['FakeRuntime'];FakeBudget=ns['FakeBudget']


class SourceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
    def test_native_lifecycle_matches_qualified_v2(self):
        old=HERE/'supervise_sand_post_completion_cpu_v2.py'
        names=('SigEvent','TimeSpec','ITimerSpec','arm_absolute_guard','native','stable','parent_death_hook',
               'Runtime','public','register','capture','send','cleanup','publish','sha','read_json')
        for name in names:
            self.assertEqual(ast.dump(declaration(SOURCE,name),include_attributes=False),
                ast.dump(declaration(old,name),include_attributes=False),name)
    def test_cohort_gate_is_frozen_scalar_gate(self):
        self.assertEqual(ast.dump(declaration(SOURCE,'cohort_gate'),include_attributes=False),
            ast.dump(declaration(HERE/'prepare_sand_reserved_test_recovered_v1.py','cohort_gate'),include_attributes=False))
    def test_two_exact_worker_argv(self):
        for operation,(mode,split) in M.OPERATIONS.items():
            flags=M.COMMON_FLAGS+M.MODE_FLAGS[mode]
            paths={f:str(self.root/f.lstrip('-')) for f in flags}
            r={'operation':operation,'paths':paths};argv=M.source_command(r)
            self.assertEqual(argv[:6],[M.PYTHON,'-B',M.PREP+'/prepare_sand_reserved_test_recovered_v1.py','--execute','--mode',mode])
            pairs=dict(zip(argv[6::2],argv[7::2]));self.assertEqual(set(pairs),set(flags)|({'--split'} if split else set()))
            if split:self.assertEqual(pairs['--split'],split)
            for f in flags:self.assertEqual(pairs[f],paths[f])
            r['paths']['--unexpected']=str(self.root/'other')
            with self.assertRaises(ValueError):M.source_command(r)
    def make_outputs(self,mode):
        out=self.root/'outputs';paths=M.source_outputs(mode,str(out));out.mkdir()
        for p in paths:Path(p).parent.mkdir(exist_ok=True);Path(p).write_bytes(b'opaque synthetic bytes')
        return out,paths
    def test_missing_output_directory_rejected(self):
        with self.assertRaises(ValueError):M.output_hashes([str(self.root/'no/file')],str(self.root/'no'),lambda:None)
    def test_output_symlink_rejected(self):
        out,paths=self.make_outputs('preflight');p=Path(paths[0]);p.unlink();p.symlink_to(self.root/'missing')
        with self.assertRaisesRegex(ValueError,'symlink'):M.output_hashes(paths,str(out),lambda:None)
    def release(self,at):
        origin=BASE+timedelta(seconds=at);end=BASE+timedelta(seconds=600)
        return dict(schema=M.SCHEMA,issued_by='root',status='approved_one_bounded_B_preflight_amendment_invocation',operation='preflight_valid',
            hostname=M.B_HOST,remaining_shared_amendment_seconds=600-at,cleanup_and_publication_seconds=15,
            new_or_restarted_clock_granted=True,original_source_clock_restarted=False,automatic_retry=False,invocation_origin_utc=origin.isoformat(),
            work_stop_utc=(end-timedelta(seconds=15)).isoformat(),cleanup_deadline_utc=(end-timedelta(seconds=5)).isoformat(),
            publication_deadline_utc=end.isoformat(),issued_utc=origin.isoformat(),clock_sample=dict(
                source='root_fresh_tool_and_host_clock_evidence',error_bound_seconds=5,host_utc=origin.isoformat(),
                root_reference_utc=origin.isoformat(),host_boot_id='fake-boot',host_monotonic_seconds=float(at)))
    def test_later_fresh_sample_maps_same_original_stop(self):
        ends=[]
        for at in (0,300,500):
            rt=FakeRuntime();rt.t=float(at);r=self.release(at)
            with patch.object(M.sys,'platform','linux'),patch.object(M.socket,'gethostname',return_value=M.B_HOST):M.structure(r)
            b=M.Budget(r,rt,rt.t);ends.append(b.ends['publication_deadline_utc'])
        self.assertEqual(ends,[595.,595.,595.])
    def test_attempt_to_restart900_at_later_mode_rejected(self):
        r=self.release(500);r['remaining_shared_amendment_seconds']=600
        with patch.object(M.sys,'platform','linux'),patch.object(M.socket,'gethostname',return_value=M.B_HOST):
            with self.assertRaisesRegex(ValueError,'remaining'):M.structure(r)
    def test_direct_bootstrap_consumes_same_remaining_clock(self):
        argv=BOOT.command(595*10**9,'/r','a'*64,500*10**9)
        self.assertEqual(argv[3],'75s');self.assertEqual(argv[8],M.PREP+'/supervise_sand_B_preflight_amendment_cpu_v1.py')
        with self.assertRaises(ValueError):BOOT.command(595*10**9,'/r','a'*64,580*10**9)
    def test_cohort_gate_precedes_numeric_and_data_hashing(self):
        node=declaration(SOURCE,'verify_source_contract')
        calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        line=lambda name:min(n.lineno for n in calls if n.func.id==name)
        self.assertLess(line('cohort_gate'),line('numeric_bindings'));self.assertLess(line('cohort_gate'),line('sha'))


class NumericBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve();cls.paths={};cls.pins={}
        for split,count in (('train',1000),('valid',30),('test',30)):
            root=cls.root/split;root.mkdir();records=[]
            for i in range(count):
                row={'id':f'{split}:{i:06d}','source_index':i}
                for key,prefix in (('positions','position'),('particle_types','type')):
                    name=f'{prefix}_{i:06d}.npy';p=root/name;raw=(split+':'+key+':'+str(i)).encode();p.write_bytes(raw)
                    row[key]={'path':name,'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                records.append(row)
            path=root/(split+'.json');path.write_text(json.dumps({'dataset':'Sand','split':split,'record_count':count,'records':records}))
            cls.paths['--'+split+'-manifest']=str(path);cls.pins[str(path)]=M.sha(path)
        cls.numeric=M.numeric_bindings(cls.paths,cls.pins,lambda:None)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_exact2120_opaque_byte_bindings(self):
        self.assertEqual(len(self.numeric),2120)
        self.assertTrue(all(M.sha(path)==pin for path,pin in self.numeric.items()))
    def test_duplicate_or_escaping_manifest_paths_refused(self):
        p=Path(self.paths['--test-manifest']);original=p.read_bytes()
        try:
            for changed in ('../escape.npy','/absolute.npy'):
                m=json.loads(original);m['records'][0]['positions']['path']=changed;p.write_text(json.dumps(m))
                pins={**self.pins,str(p):M.sha(p)}
                with self.assertRaises(ValueError):M.numeric_bindings(self.paths,pins,lambda:None)
            m=json.loads(original);m['records'][1]['positions']=m['records'][0]['positions'];p.write_text(json.dumps(m))
            with self.assertRaises(ValueError):M.numeric_bindings(self.paths,{**self.pins,str(p):M.sha(p)},lambda:None)
        finally:p.write_bytes(original)
    def test_numeric_mutation_after_worker_prevents_terminal_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();output=root/'result.json';control=root/'control.json';control.write_text('{}')
            r={'owner_output_dir':str(root/'owner'),'command':['fake-python','-B','frozen-source.py'],
               'clock_sample':{'synthetic':True},'_numeric_input_files_count':2120}
            own=dict(pid=9000,ppid=8000,pgid=8000,sid=8000,start_id='90',state='S',argv=['owner'],executable='owner')
            expected={**self.numeric,str(control):M.sha(control)}
            rt=FakeRuntime([output]);budget=FakeBudget(rt);original_encode=M.encoded
            changed=Path(next(iter(self.numeric)));original=changed.read_bytes()
            def mutate(value):
                raw=original_encode(value)
                if isinstance(value,dict) and value.get('schema')=='adaptgns_sand_B_preflight_amendment_cpu_terminal_v1':changed.write_bytes(b'mutated after worker')
                return raw
            try:
                with patch.object(M,'verify',return_value=(expected,[str(output)],None,{'pid':8000},own)),patch.object(M,'encoded',side_effect=mutate):
                    code=M.execute(r,control,M.sha(control),rt,budget)
                self.assertEqual(code,1);self.assertFalse((root/'owner/owner_terminal.json').exists())
                self.assertIn('Publication control',json.loads((root/'owner/publication_failed.json').read_text())['error'])
            finally:changed.write_bytes(original)





class AmendmentGateTests(unittest.TestCase):
    def fixture(self):
        original=json.loads((HERE/'sand_reserved_released_1808_root_v1/source_phase.json').read_text())
        histlocal={
            'original_source_disposition':'sand_original_source_phase_incomplete_root_v1.json',
            'independent_source_disposition':'sand_original_source_window_disposition_independent_review_code_audit_v1.json',
            'transfer_products_review':'sand_completed_transfer_independent_review_code_audit_v1.json',
            'transfer_target_publication':'sand_reserved_transfer_AB_20261006_root_v1/target_publish.json',
            'transfer_original_external_exit':'sand_reserved_transfer_AB_20261006_root_v1/root_original_external_exit.json'}
        controls={M.HISTORY_BINDINGS[k]['path']:json.loads((HERE/v).read_text()) for k,v in histlocal.items()}
        start=datetime(2026,10,6,19,tzinfo=timezone.utc);stop=start+timedelta(seconds=600)
        r=dict(source_phase=dict(path=M.PREP+'/sand_original_reserved_source_phase_20261006_v1.json',sha256=M.ORIGINAL_SOURCE_SHA),
            history=M.HISTORY_BINDINGS,amendment_phase=dict(path=M.AMENDMENT_CONTROLS+'/amendment_phase.json',sha256='a'*64),
            human_approval=dict(path=M.AMENDMENT_CONTROLS+'/human_approval.json',sha256='b'*64),
            invocation_origin_utc=start.isoformat(),publication_deadline_utc=stop.isoformat(),issued_utc=start.isoformat(),
            owner_source={'sha256':'c'*64},bootstrap_source={'sha256':'d'*64},reviewed_candidate_manifest_sha256='e'*64)
        phase=dict(schema='adaptgns_sand_B_preflight_completion_amendment_phase_v1',issued_by='root',
            status='approved_new_shared_B_preflight_amendment_phase',amendment_id=M.AMENDMENT_ID,hostname=M.B_HOST,
            operations=list(M.OPERATIONS),new_shared_amendment_seconds=600,new_or_restarted_clock_granted=True,
            original_source_clock_restarted=False,original_source_phase=r['source_phase'],history=M.HISTORY_BINDINGS,
            human_approval=r['human_approval'],cohort_sha256=original['cohort_sha256'],cohort_audit_sha256=original['cohort_audit_sha256'],
            automatic_retry=False,includes_both_operations_external_exits_native_closure_and_reviews=True,
            subsequent_gpu_control_seconds=600,original_evaluation_seconds=11760,original_analysis_seconds=3600,
            full_evaluation_stage_denominator=24,global_analysis_deadline_utc=M.GLOBAL_STOP.isoformat(),
            amendment_started_utc=start.isoformat(),amendment_stop_utc=stop.isoformat(),issued_utc=start.isoformat(),
            reviewed_candidate_manifest_sha256='e'*64)
        approval=dict(schema='adaptgns_sand_B_preflight_explicit_human_approval_v1',recorded_by='root',
            status='explicit_human_approval_received',authorization_source='direct_human_message',amendment_id=M.AMENDMENT_ID,
            human_message_reference='SYNTHETIC_TEST_ONLY',human_message_text='SYNTHETIC_TEST_ONLY: approve the exact test fixture',
            approved_new_shared_seconds=600,hostname=M.B_HOST,operations=list(M.OPERATIONS),
            original_source_phase_sha256=M.ORIGINAL_SOURCE_SHA,preserves_original_incomplete_history=True,automatic_retry=False,
            owner_source_sha256='c'*64,bootstrap_source_sha256='d'*64,preparation_source_sha256=M.PREPARER_SHA,
            reviewed_candidate_manifest_sha256='e'*64,subsequent_gpu_control_seconds=600,original_evaluation_seconds=11760,
            original_analysis_seconds=3600,global_analysis_deadline_utc=M.GLOBAL_STOP.isoformat(),
            human_approval_observed_utc=(start-timedelta(seconds=60)).isoformat(),recorded_utc=(start-timedelta(seconds=30)).isoformat())
        controls[r['source_phase']['path']]=original;controls[r['amendment_phase']['path']]=phase;controls[r['human_approval']['path']]=approval
        return r,controls,phase,approval,{'sha256':original['cohort_sha256']},{'sha256':original['cohort_audit_sha256']}
    def check_gate(self,fixture):
        r,controls,phase,approval,cohort,audit=fixture
        with patch.object(M,'read_json',side_effect=lambda b,c:controls[b['path']]):
            return M.amendment_gate(r,cohort,audit,lambda:None)
    def test_synthetic_exact600_passes_and_returns_every_history_binding(self):
        f=self.fixture();out=self.check_gate(f)
        self.assertEqual((out[3]-out[2]).total_seconds(),600);self.assertEqual(len(out[4]),7)
    def test_reject_approval_failures(self):
        for key,value in [('status','template_not_admitted'),('authorization_source','other_model'),
             ('human_message_reference',None),('human_message_text',' '),('approved_new_shared_seconds',900),
             ('hostname','A'),('operations',['preflight_valid']),('owner_source_sha256','f'*64),
             ('bootstrap_source_sha256','f'*64),('preparation_source_sha256','f'*64),('reviewed_candidate_manifest_sha256',None),
             ('preserves_original_incomplete_history',False),('automatic_retry',True),('subsequent_gpu_control_seconds',0),
             ('original_evaluation_seconds',100),('original_analysis_seconds',100),
             ('human_approval_observed_utc','2026-10-06T19:00:01+00:00')]:
            with self.subTest(key=key):
                f=self.fixture();f[3][key]=value
                with self.assertRaises(ValueError):self.check_gate(f)
    def test_reject_phase_failures(self):
        for key,value in [('status','template_not_admitted'),('hostname','A'),('operations',['acquire']),
             ('new_shared_amendment_seconds',599),('new_shared_amendment_seconds',900),('original_source_clock_restarted',True),
             ('new_or_restarted_clock_granted',False),('automatic_retry',True),('subsequent_gpu_control_seconds',0),
             ('original_evaluation_seconds',11759),('original_analysis_seconds',3599),('full_evaluation_stage_denominator',12),
             ('includes_both_operations_external_exits_native_closure_and_reviews',False),
             ('amendment_stop_utc','2026-10-06T19:09:59+00:00'),('amendment_stop_utc','2026-10-06T19:10:01+00:00')]:
            with self.subTest(key=key,value=value):
                f=self.fixture();f[2][key]=value
                with self.assertRaises(ValueError):self.check_gate(f)
    def test_reject_later_mode_clock_reset(self):
        f=self.fixture();f[0]['invocation_origin_utc']='2026-10-06T19:05:00+00:00'
        f[0]['publication_deadline_utc']='2026-10-06T19:15:00+00:00'
        with self.assertRaisesRegex(ValueError,'reserve'):self.check_gate(f)
    def test_reject_full_reserve_without_five_second_margin(self):
        f=self.fixture();start=M.GLOBAL_STOP-timedelta(seconds=600+600+11760+3600)
        f[2]['amendment_started_utc']=start.isoformat();f[2]['amendment_stop_utc']=(start+timedelta(seconds=600)).isoformat()
        f[0]['invocation_origin_utc']=start.isoformat();f[0]['publication_deadline_utc']=f[2]['amendment_stop_utc']
        with self.assertRaisesRegex(ValueError,'reserve'):self.check_gate(f)
    def test_original_external_exit_must_precede_new_anchor(self):
        f=self.fixture();start=datetime(2026,10,6,18,35,40,tzinfo=timezone.utc);stop=start+timedelta(seconds=600)
        f[2].update(amendment_started_utc=start.isoformat(),amendment_stop_utc=stop.isoformat(),issued_utc=start.isoformat())
        f[0].update(invocation_origin_utc=start.isoformat(),publication_deadline_utc=stop.isoformat(),issued_utc=start.isoformat())
        f[3].update(human_approval_observed_utc='2026-10-06T18:35:35+00:00',recorded_utc='2026-10-06T18:35:39+00:00')
        with self.assertRaisesRegex(ValueError,'Original transfer external exit'):self.check_gate(f)
    def test_reject_original_history_rewrite(self):
        for k,field,value in [('original_source_disposition','status','complete'),
            ('original_source_disposition','source_preparation_unexecuted',0),
            ('independent_source_disposition','original_history_must_remain_incomplete',False),
            ('transfer_original_external_exit','observed_utc',M.ORIGINAL_SOURCE_STOP),
            ('transfer_original_external_exit','original_session',123),
            ('transfer_target_publication','hostname','A')]:
            with self.subTest(k=k,field=field):
                f=self.fixture();f[1][M.HISTORY_BINDINGS[k]['path']][field]=value
                with self.assertRaises(ValueError):self.check_gate(f)
    def test_reject_transfer_map_missing_or_mutated_file(self):
        for remove in (True,False):
            f=self.fixture();files=f[1][M.HISTORY_BINDINGS['transfer_target_publication']['path']]['files_sha256'];key=next(iter(files))
            if remove:del files[key]
            else:files[key]='f'*64
            with self.assertRaises(ValueError):self.check_gate(f)
    def test_only_preflight_operations_and_outputs(self):
        self.assertEqual(M.OPERATIONS,{'preflight_valid':('preflight','valid'),'preflight_test':('preflight','test')})
        for mode in ('acquire','convert','census','candidate'):
            with self.assertRaises(ValueError):M.source_outputs(mode,'/synthetic')
        self.assertEqual(M.source_outputs('preflight','/synthetic'),['/synthetic/split_preflight.json'])
    def test_null_bound_approval_rejected_by_real_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'approval.json';path.write_text('{}')
            with self.assertRaises(ValueError):M.read_json({'path':str(path),'sha256':None})
    def test_bootstrap_refuses_over600(self):
        with self.assertRaises(ValueError):BOOT.command(601*10**9,'/r','a'*64,0)
    def test_original_qualified_owner_and_frozen_worker_unmodified(self):
        self.assertEqual(M.sha(HERE/'supervise_sand_reserved_source_cpu_v1.py'),'d7880125e43ea6eeb047e7b3170a06b2ff1b189b9051978234adcf4d2b297955')
        self.assertEqual(M.sha(HERE/'prepare_sand_reserved_test_recovered_v1.py'),M.PREPARER_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
