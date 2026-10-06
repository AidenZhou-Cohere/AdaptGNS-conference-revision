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
SOURCE=HERE/'supervise_sand_reserved_source_cpu_v1.py'
spec=importlib.util.spec_from_file_location('new_reserved_cpu_test',SOURCE)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
boot_spec=importlib.util.spec_from_file_location('new_reserved_boot_test',HERE/'launch_sand_reserved_source_cpu_v1.py')
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
    def test_all_seven_exact_worker_argv(self):
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
    def test_exact_output_sets(self):
        for mode,count in (('acquire',4),('convert',63),('census',1),('candidate',1),('preflight',1)):
            paths=M.source_outputs(mode,str(self.root))
            self.assertEqual(len(paths),count);self.assertEqual(len(set(paths)),count)
        self.assertIn(str(self.root/'test/position_000029.npy'),M.source_outputs('convert',str(self.root)))
        self.assertIn(str(self.root/'test/type_000000.npy'),M.source_outputs('convert',str(self.root)))
    def make_outputs(self,mode):
        out=self.root/'outputs';paths=M.source_outputs(mode,str(out));out.mkdir()
        for p in paths:Path(p).parent.mkdir(exist_ok=True);Path(p).write_bytes(b'opaque synthetic bytes')
        return out,paths
    def test_convert_exact63_success_and_staging_extra_refusal(self):
        out,paths=self.make_outputs('convert')
        self.assertEqual(len(M.output_hashes(paths,str(out),lambda:None)),63)
        (out/'.test.staging').mkdir()
        with self.assertRaisesRegex(ValueError,'source output tree'):M.output_hashes(paths,str(out),lambda:None)
    def test_acquire_failure_marker_rejected_preserved(self):
        out,paths=self.make_outputs('acquire');marker=out/'failed_preparation.json';marker.write_text('{}')
        with self.assertRaisesRegex(ValueError,'source output tree'):M.output_hashes(paths,str(out),lambda:None)
        self.assertTrue(marker.exists())
    def test_missing_output_directory_rejected(self):
        with self.assertRaises(ValueError):M.output_hashes([str(self.root/'no/file')],str(self.root/'no'),lambda:None)
    def test_output_symlink_rejected(self):
        out,paths=self.make_outputs('census');p=Path(paths[0]);p.unlink();p.symlink_to(self.root/'missing')
        with self.assertRaisesRegex(ValueError,'symlink'):M.output_hashes(paths,str(out),lambda:None)
    def release(self,at):
        origin=BASE+timedelta(seconds=at);end=BASE+timedelta(seconds=900)
        return dict(schema=M.SCHEMA,issued_by='root',status='approved_one_bounded_source_invocation',operation='census',
            hostname='fake',remaining_shared_source_seconds=900-at,cleanup_and_publication_seconds=15,
            new_or_restarted_clock_granted=False,automatic_retry=False,invocation_origin_utc=origin.isoformat(),
            work_stop_utc=(end-timedelta(seconds=15)).isoformat(),cleanup_deadline_utc=(end-timedelta(seconds=5)).isoformat(),
            publication_deadline_utc=end.isoformat(),issued_utc=origin.isoformat(),clock_sample=dict(
                source='root_fresh_tool_and_host_clock_evidence',error_bound_seconds=5,host_utc=origin.isoformat(),
                root_reference_utc=origin.isoformat(),host_boot_id='fake-boot',host_monotonic_seconds=float(at)))
    def test_later_fresh_sample_maps_same_original_stop(self):
        ends=[]
        for at in (0,400,800):
            rt=FakeRuntime();rt.t=float(at);r=self.release(at)
            with patch.object(M.sys,'platform','linux'),patch.object(M.socket,'gethostname',return_value='fake'):M.structure(r)
            b=M.Budget(r,rt,rt.t);ends.append(b.ends['publication_deadline_utc'])
        self.assertEqual(ends,[895.,895.,895.])
    def test_attempt_to_restart900_at_later_mode_rejected(self):
        r=self.release(800);r['remaining_shared_source_seconds']=900
        with patch.object(M.sys,'platform','linux'),patch.object(M.socket,'gethostname',return_value='fake'):
            with self.assertRaisesRegex(ValueError,'remaining'):M.structure(r)
    def test_direct_bootstrap_consumes_same_remaining_clock(self):
        argv=BOOT.command(895*10**9,'/r','a'*64,800*10**9)
        self.assertEqual(argv[3],'75s');self.assertEqual(argv[8],M.PREP+'/supervise_sand_reserved_source_cpu_v1.py')
        with self.assertRaises(ValueError):BOOT.command(895*10**9,'/r','a'*64,880*10**9)
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
                if isinstance(value,dict) and value.get('schema')=='adaptgns_sand_reserved_source_cpu_terminal_v1':changed.write_bytes(b'mutated after worker')
                return raw
            try:
                with patch.object(M,'verify',return_value=(expected,[str(output)],None,{'pid':8000},own)),patch.object(M,'encoded',side_effect=mutate):
                    code=M.execute(r,control,M.sha(control),rt,budget)
                self.assertEqual(code,1);self.assertFalse((root/'owner/owner_terminal.json').exists())
                self.assertIn('Publication control',json.loads((root/'owner/publication_failed.json').read_text())['error'])
            finally:changed.write_bytes(original)


if __name__=='__main__':unittest.main(verbosity=2)
