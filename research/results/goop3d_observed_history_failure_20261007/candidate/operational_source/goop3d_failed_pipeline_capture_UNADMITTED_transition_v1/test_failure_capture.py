"""Focused inherited synthetic identity tests; never instantiate live Runtime."""
import ast,copy,datetime,importlib.util,json,os,shlex,stat,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import observe_failure as O
import root_capture as R
C=json.loads((Path(__file__).resolve().parent/'contract.json').read_bytes())
def records():
    hard = C['hard_deadline_monotonic_ns']
    computed = hard - 140 * 10**9
    owner_argv = [O.BASE+'/.venv/bin/python','-I','-S','-B',C['owner_source']['path'],
                  '--execute','--hard-deadline-monotonic-ns',str(hard),
                  '--outer-term-seconds','120','--outer-computed-monotonic-ns',str(computed),
                  '--release',O.ROOT+'/controls/pipeline.cpu_release.json',
                  '--release-sha256',C['release_sha256']]
    def row(pid, parent, argv):
        return dict(pid=pid,ppid=parent,pgid=70001,sid=70001,start_id=str(900000000+pid),argv=argv)
    outer = row(70001,39,['/usr/bin/timeout','--signal=TERM','--kill-after=15s','120s']+owner_argv)
    owner = row(70002,70001,owner_argv)
    worker = row(70003,70002,copy.deepcopy(C['worker_argv']))
    started = dict(release_sha256=C['release_sha256'],
                   native_hard_guard=dict(absolute_deadline_ns=hard,clock='CLOCK_MONOTONIC',signal='SIGKILL'),
                   native_outer_timing=dict(computed_monotonic_ns=computed,term_seconds=120),
                   outer_timeout_identity=outer,owner_identity=owner,original_clock_sample=copy.deepcopy(C['clock_sample']))
    child = dict(registered=True,pid=70003,identity=worker,owner_identity=copy.deepcopy(owner),command=copy.deepcopy(C['worker_argv']))
    terminal = copy.deepcopy(started)
    terminal.pop('original_clock_sample')
    terminal.update(child=copy.deepcopy(child),status='complete',failure=None,child_native_absent=True,
                    publication_utc='synthetic-only',publication_monotonic=1)
    terminal['child'].update(reaped=True,exit_code=0)
    return dict(zip(O.NAMES,[started,child,terminal]))

def stat_raw(pid, start='900070003', state='S', ppid=70002, pgid=70001, sid=70001, comm='python'):
    fields = [state,str(ppid),str(pgid),str(sid)] + ['0']*15 + [str(start)]
    return (str(pid)+' ('+comm+') '+' '.join(fields)+'\n').encode('ascii')

class FakeRuntime:
    def __init__(self, docs=None, proc=None, host=None, boots=None):
        docs = records() if docs is None else docs
        self.docs = {k:(json.dumps(v).encode() if not isinstance(v,(bytes,Exception,list)) else v) for k,v in docs.items()}
        self.process = proc or {}
        self.host_value = O.HOST if host is None else host
        self.boot_values = boots or [O.BOOT]
        self.allowed_pids = set(O.HISTORICAL)
        self.native_reads = []
        self.registry_reads = []
        self.clock_calls = 0
    def check(self): pass
    def clocks(self):
        self.clock_calls += 1
        return dict(utc='synthetic-only',monotonic_seconds=self.clock_calls)
    def host(self): return self.host_value
    def boot(self): return self.boot_values.pop(0) if len(self.boot_values)>1 else self.boot_values[0]
    @staticmethod
    def value(values,key):
        if key not in values: raise FileNotFoundError('synthetic absent')
        value = values[key]
        if isinstance(value,list):
            value = value.pop(0) if len(value)>1 else value[0]
        if isinstance(value,Exception): raise value
        return value
    def registry(self,name):
        self.registry_reads.append(name)
        return self.value(self.docs,name)
    def proc(self,pid,name):
        assert pid in self.allowed_pids
        assert name in ('stat','cmdline')
        self.native_reads.append((pid,name))
        return self.value(self.process,(pid,name))

class RegistryTests(unittest.TestCase):
    def test_complete_without_terminal(self):
        d=records();d.pop('owner_terminal.json')
        roles,claims,errors=O.registry_identities(d,C)
        self.assertEqual(set(roles),{'outer','owner','worker'});self.assertFalse(errors);self.assertFalse(claims)
    def test_terminal_alone_supplies_three_identities(self):
        d=records();roles,_,errors=O.registry_identities({'owner_terminal.json':d['owner_terminal.json']},C)
        self.assertEqual(set(roles),{'outer','owner','worker'});self.assertFalse(errors)
    def test_registration_alone_is_incomplete(self):
        d=records();roles,_,errors=O.registry_identities({'child_registered.json':d['child_registered.json']},C)
        self.assertEqual(set(roles),{'owner','worker'});self.assertFalse(errors)
    def test_conflicting_top_level_owner(self):
        d=records();d['owner_terminal.json']['owner_identity']['start_id']='123'
        self.assertTrue(O.registry_identities(d,C)[2])
    def test_conflicting_terminal_child_owner(self):
        d=records();d['owner_terminal.json']['child']['owner_identity']['start_id']='123'
        self.assertTrue(O.registry_identities(d,C)[2])
    def test_wrong_release_clock_deadline_and_argv(self):
        for mutation in ('release','clock','deadline','argv'):
            with self.subTest(mutation=mutation):
                d=records();s=d['owner_started.json']
                if mutation=='release':s['release_sha256']='0'*64
                elif mutation=='clock':s['original_clock_sample']['host_boot_id']='wrong'
                elif mutation=='deadline':s['native_hard_guard']['absolute_deadline_ns']+=1
                else:s['owner_identity']['argv']=['wrong']
                self.assertTrue(O.registry_identities(d,C)[2])
    def test_wrong_worker_argv(self):
        d=records();d['child_registered.json']['identity']['argv']=['wrong']
        self.assertTrue(O.registry_identities(d,C)[2])
    def test_role_overlap_historical(self):
        d=records();d['owner_started.json']['outer_timeout_identity']['pid']=O.HISTORICAL[0]
        self.assertTrue(O.registry_identities(d,C)[2])
    def test_wrong_ancestry(self):
        d=records();d.pop('owner_terminal.json');d['child_registered.json']['identity']['ppid']=80000
        self.assertTrue(O.registry_identities(d,C)[2])
    def test_invalid_pid_types_and_range(self):
        for bad in (True,0,-1,1.5,'70001',2**40):
            with self.subTest(bad=bad):
                d=records()['owner_started.json']['owner_identity'];d['pid']=bad
                with self.assertRaises(ValueError):O.identity(d)
    def test_invalid_start_ids(self):
        for bad in ('0','-1','abc','١٢٣','0900070002',3):
            with self.subTest(bad=bad):
                d=records()['owner_started.json']['owner_identity'];d['start_id']=bad
                with self.assertRaises(ValueError):O.identity(d)
    def test_duplicate_and_nonfinite_json(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):O.strict(raw)

class NativeTests(unittest.TestCase):
    def setUp(self):self.expected=records()['child_registered.json']['identity']
    def sample(self,process):
        rt=FakeRuntime(proc=process);rt.allowed_pids.add(self.expected['pid'])
        return O.sample_native(rt,self.expected)
    def test_absent(self):self.assertEqual(self.sample({})['status'],'absent')
    def test_permission_is_not_absence(self):self.assertEqual(self.sample({(70003,'stat'):PermissionError()})['status'],'cannot_observe')
    def test_same_identity_present(self):
        x=self.sample({(70003,'stat'):stat_raw(70003),(70003,'cmdline'):b'\0'.join(s.encode() for s in self.expected['argv'])+b'\0'})
        self.assertEqual(x['status'],'present');self.assertFalse(x['recorded_field_mismatches'])
    def test_zombie_present(self):
        x=self.sample({(70003,'stat'):stat_raw(70003,state='Z'),(70003,'cmdline'):b''})
        self.assertEqual(x['status'],'present');self.assertTrue(x['zombie'])
    def test_orphan_present(self):
        x=self.sample({(70003,'stat'):stat_raw(70003,ppid=1),(70003,'cmdline'):b'changed\0'})
        self.assertEqual(x['status'],'present');self.assertIn('ppid',x['recorded_field_mismatches'])
    def test_stable_reuse_no_unrelated_cmdline_read(self):
        rt=FakeRuntime(proc={(70003,'stat'):stat_raw(70003,start='999')})
        rt.allowed_pids.add(self.expected['pid'])
        x=O.sample_native(rt,self.expected)
        self.assertEqual(x['status'],'pid_reused');self.assertNotIn((70003,'cmdline'),rt.native_reads)
    def test_reuse_race_unknown(self):
        x=self.sample({(70003,'stat'):[stat_raw(70003,start='999'),stat_raw(70003,start='1000')]})
        self.assertEqual(x['status'],'cannot_observe')
    def test_cmdline_disappearance(self):
        x=self.sample({(70003,'stat'):stat_raw(70003)})
        self.assertEqual(x['status'],'changed_during_observation')
    def test_final_stat_disappearance(self):
        x=self.sample({(70003,'stat'):[stat_raw(70003),FileNotFoundError()],(70003,'cmdline'):b'x\0'})
        self.assertEqual(x['status'],'changed_during_observation')
    def test_ancestry_change_during_sample(self):
        x=self.sample({(70003,'stat'):[stat_raw(70003),stat_raw(70003,ppid=1)],(70003,'cmdline'):b'x\0'})
        self.assertEqual(x['status'],'cannot_observe')
    def test_comm_parentheses_and_spaces(self):
        x=O.parse_stat(stat_raw(70003,comm='x ) ( embedded ) name'),70003)
        self.assertEqual(x['start_id'],self.expected['start_id'])
    def test_malformed_native_metadata_not_reuse(self):
        for field,value in [('start','nonnumeric'),('start','0'),('start','-1'),('start','0900070003'),('state','BAD'),('ppid',-1),('pgid',-1),('sid',-1)]:
            with self.subTest(field=field,value=value):
                x=self.sample({(70003,'stat'):stat_raw(70003,**{field:value})})
                self.assertEqual(x['status'],'cannot_observe')
    def test_wrong_pid_and_short_stat(self):
        for raw in (stat_raw(70004),b'70003 (x) S 1 2'):
            with self.subTest(raw=raw):
                self.assertEqual(self.sample({(70003,'stat'):raw})['status'],'cannot_observe')
    def test_invalid_utf8_preserved_as_mismatch(self):
        x=self.sample({(70003,'stat'):stat_raw(70003),(70003,'cmdline'):b'\xff\0'})
        self.assertEqual(x['status'],'present');self.assertIn('argv',x['recorded_field_mismatches'])
class ScopeTests(unittest.TestCase):
    def test_exact_failed_contract(self):
        O.validate_contract(C)
        d=copy.deepcopy(C);d['original_pipeline_exit_code']=0
        with self.assertRaises(ValueError):O.validate_contract(d)
    def test_exact42historical(self):
        self.assertEqual(len(C['historical']),42)
        self.assertEqual(tuple(x['pid'] for x in C['historical']),O.HISTORICAL)
    def test_remote_timeout_only_new_observer(self):
        argv=R.command('SYNTHETIC_SOURCE','a'*64);remote=shlex.split(argv[-1])
        self.assertEqual(remote[:4],['/usr/bin/timeout','--signal=KILL','20s',O.BASE+'/.venv/bin/python'])
        self.assertEqual(remote[4:],['-I','-S','-B','-c','SYNTHETIC_SOURCE','--observe','--contract-sha256','a'*64])
    def test_remote_source_has_no_target_signals_or_subprocesses(self):
        tree=ast.parse(Path(O.__file__).read_text());calls={ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)}
        self.assertFalse(calls&{'os.kill','os.killpg','subprocess.run','subprocess.Popen','os.system'})
    def test_unreadable_registry_blocks_complete_gate(self):
        source=Path(O.__file__).read_text()
        self.assertIn("all(item['status'] in ('read','missing') for item in metadata.values())",source)
    def test_products_hash_only(self):
        source=Path(O.__file__).read_text()
        self.assertIn('products={path:rt.metadata(path,256<<20,True) for path in PRODUCTS}',source)
        self.assertNotIn('strict(products',source)
    def test_inert_local_entry(self):
        import io
        with patch.object(sys,'argv',['root_capture.py']),patch('sys.stdout',new_callable=io.StringIO) as out:
            R.main();self.assertEqual(json.loads(out.getvalue())['status'],'inert_metadata_observer')

if __name__=='__main__':unittest.main(verbosity=2)
