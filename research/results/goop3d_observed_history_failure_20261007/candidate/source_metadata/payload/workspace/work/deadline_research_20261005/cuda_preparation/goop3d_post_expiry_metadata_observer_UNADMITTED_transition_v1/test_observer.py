"""Independent synthetic-only observer tests. Never instantiate live Runtime."""
import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
PACKAGE = BASE if (BASE / 'observe_metadata.py').is_file() else BASE / 'goop3d_post_expiry_metadata_observer_UNADMITTED_transition_v1'
SPEC = importlib.util.spec_from_file_location('closure_observer_independent_subject', PACKAGE / 'observe_metadata.py')
O = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(O)
C = json.loads((PACKAGE / 'contract.json').read_bytes())
ROOT_SPEC = importlib.util.spec_from_file_location('closure_observer_independent_transport', PACKAGE / 'root_observe.py')
R = importlib.util.module_from_spec(ROOT_SPEC)
ROOT_SPEC.loader.exec_module(R)


def records():
    hard = C['hard_deadline_monotonic_ns']
    computed = hard - 140 * 10**9
    owner_argv = [O.BASE+'/.venv/bin/python','-I','-S','-B',C['owner_source']['path'],
                  '--execute','--hard-deadline-monotonic-ns',str(hard),
                  '--outer-term-seconds','120','--outer-computed-monotonic-ns',str(computed),
                  '--release',O.BASE+'/goop3d_final_analysis_20261006_v1/controls/d3_saved_array_audit.cpu_release.json',
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


class ObservationTests(unittest.TestCase):
    def test_all_absent_42_without_terminal_and_no_admission(self):
        d=records();d.pop('owner_terminal.json');rt=FakeRuntime(d)
        x=O.observe(C,rt)
        self.assertTrue(x['native_closure_observed']);self.assertTrue(x['all_recorded_identity_absence_observed']);self.assertEqual(x['observed_pid_count'],42)
        self.assertFalse(x['scientific_admission']);self.assertFalse(x['remote_execution_success_established']);self.assertFalse(x['new_analysis_clock_or_release'])
        self.assertEqual(rt.native_reads.count((70003,'stat')),2)
    def test_missing_registry_unknown_not_closed(self):
        x=O.observe(C,FakeRuntime({}))
        self.assertFalse(x['native_closure_observed']);self.assertFalse(x['registry_identity_coverage_complete']);self.assertEqual(x['observed_pid_count'],39)
    def test_conflicting_registry_cannot_expand_allowlist(self):
        d=records();d['owner_terminal.json']['owner_identity']['pid']=777777;rt=FakeRuntime(d)
        x=O.observe(C,rt)
        self.assertFalse(x['registry_identity_coverage_complete']);self.assertEqual(rt.allowed_pids,set(O.HISTORICAL));self.assertEqual(x['observed_pid_count'],39)
    def test_partial_json_unknown(self):
        d=records();d['owner_started.json']=b'{'
        x=O.observe(C,FakeRuntime(d));self.assertFalse(x['native_closure_observed'])
    def test_nondict_registry_preserves_unknown_result(self):
        for raw in (b'null',b'[]',b'1',b'"text"'):
            with self.subTest(raw=raw):
                d=records();d['owner_started.json']=raw
                x=O.observe(C,FakeRuntime(d));self.assertFalse(x['native_closure_observed'])
    def test_registry_permission_unknown(self):
        d=records();d['owner_started.json']=PermissionError('synthetic')
        x=O.observe(C,FakeRuntime(d));self.assertFalse(x['native_closure_observed'])
    def test_registry_changed_after_native_sample(self):
        d=records();raw=json.dumps(d['owner_started.json']).encode();d['owner_started.json']=[raw,raw+b' ']
        x=O.observe(C,FakeRuntime(d));self.assertFalse(x['registry_bytes_unchanged']);self.assertFalse(x['native_closure_observed'])
    def test_terminal_appears_after_first_read(self):
        d=records();raw=json.dumps(d['owner_terminal.json']).encode();d['owner_terminal.json']=[FileNotFoundError(),raw]
        x=O.observe(C,FakeRuntime(d));self.assertFalse(x['registry_bytes_unchanged']);self.assertFalse(x['native_closure_observed'])
    def test_wrong_host_stops_native_reads(self):
        rt=FakeRuntime(host='wrong');x=O.observe(C,rt)
        self.assertFalse(x['native_closure_observed']);self.assertFalse(rt.native_reads);self.assertFalse(rt.registry_reads)
    def test_wrong_initial_boot_stops_native_reads(self):
        rt=FakeRuntime(boots=['wrong']);x=O.observe(C,rt)
        self.assertFalse(x['native_closure_observed']);self.assertFalse(rt.native_reads)
    def test_changed_final_boot_blocks_closure(self):
        x=O.observe(C,FakeRuntime(boots=[O.BOOT,'wrong']))
        self.assertFalse(x['native_closure_observed'])
    def test_present_between_two_absence_rounds(self):
        pid=O.HISTORICAL[0]
        rt=FakeRuntime(proc={(pid,'stat'):[FileNotFoundError(),stat_raw(pid,start=C['historical'][0]['start_id']),stat_raw(pid,start=C['historical'][0]['start_id'])],(pid,'cmdline'):b'x\0'})
        self.assertFalse(O.observe(C,rt)['native_closure_observed'])
    def test_contract_wrong_scope_rejected(self):
        d=copy.deepcopy(C);d['historical'][0]['pid']=123
        with self.assertRaises(ValueError):O.observe(d,FakeRuntime())
    def test_16_mib_cap_does_not_reject_normal_terminal(self):
        d=records();d['owner_terminal.json']['ignored_metadata']='x'*1400000
        x=O.observe(C,FakeRuntime(d));self.assertTrue(x['native_closure_observed'])


class ReadOnlyFakeOS:
    O_RDONLY=os.O_RDONLY;O_DIRECTORY=os.O_DIRECTORY;O_NOFOLLOW=os.O_NOFOLLOW;O_NONBLOCK=os.O_NONBLOCK
    def __init__(self,raw=b'{}',mode=stat.S_IFREG|0o600,change=False):
        self.raw=raw;self.size=len(raw);self.mode=mode;self.change=change;self.calls=[];self.nextfd=10;self.fstat_calls=0
    def open(self,path,flags,dir_fd=None):
        self.calls.append(('open',path,flags,dir_fd));self.nextfd+=1;return self.nextfd
    def close(self,fd):self.calls.append(('close',fd))
    def fstat(self,fd):
        self.fstat_calls+=1
        return SimpleNamespace(st_mode=self.mode,st_size=self.size,st_dev=1,st_ino=1,
                               st_mtime_ns=int(self.change and self.fstat_calls>1),st_ctime_ns=0)
    def read(self,fd,size):
        self.calls.append(('read',fd,size));raw=self.raw[:size];self.raw=self.raw[size:];return raw


class ReadSurfaceTests(unittest.TestCase):
    def runtime(self):
        rt=object.__new__(O.Runtime);rt.allowed_pids=set(O.HISTORICAL);rt.check=lambda:None
        return rt
    def test_allowlist_rejects_product_and_arbitrary_proc_before_open(self):
        fs=ReadOnlyFakeOS()
        with patch.object(O,'os',fs):
            for path in (O.BASE+'/saved_array_audit.json','/proc/12345/stat','/etc/passwd'):
                with self.subTest(path=path):
                    with self.assertRaises(ValueError):self.runtime().read_raw(path,100)
        self.assertFalse(fs.calls)
    def test_every_nonroot_open_nofollow_and_final_nonblocking(self):
        fs=ReadOnlyFakeOS(raw=b'')
        with patch.object(O,'os',fs):self.runtime().read_raw(O.OWNER+'/owner_started.json',100)
        opens=[x for x in fs.calls if x[0]=='open']
        self.assertTrue(all(x[2]&os.O_NOFOLLOW for x in opens[1:]));self.assertTrue(opens[-1][2]&os.O_NONBLOCK)
        self.assertEqual(len([x for x in fs.calls if x[0]=='close']),len(opens))
    def test_nonregular_and_oversize_refused_without_read(self):
        for fs in (ReadOnlyFakeOS(mode=stat.S_IFIFO),ReadOnlyFakeOS(raw=b'x'*101)):
            with self.subTest(mode=fs.mode):
                with patch.object(O,'os',fs):
                    with self.assertRaises(ValueError):self.runtime().read_raw(O.OWNER+'/owner_started.json',100)
                self.assertFalse(any(x[0]=='read' for x in fs.calls))
    def test_changed_file_rejected(self):
        fs=ReadOnlyFakeOS(raw=b'',change=True)
        with patch.object(O,'os',fs):
            with self.assertRaises(ValueError):self.runtime().read_raw(O.OWNER+'/owner_started.json',100)
    def test_regular_bytes_return_unchanged(self):
        fs=ReadOnlyFakeOS(raw=b'{"x":1}')
        with patch.object(O,'os',fs):
            self.assertEqual(self.runtime().read_raw(O.OWNER+'/owner_started.json',100),b'{"x":1}')
    def test_zero_stat_size_cannot_bypass_read_cap(self):
        fs=ReadOnlyFakeOS(raw=b'x'*101);fs.size=0
        with patch.object(O,'os',fs):
            with self.assertRaises(ValueError):self.runtime().read_raw(O.OWNER+'/owner_started.json',100)
    def test_observation_clock_guard_uses_only_synthetic_clocks(self):
        ref=datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc)
        rt=object.__new__(O.Runtime);rt.start=0;rt.start_utc=ref
        for elapsed,wall,passes in ((1,1,True),(11.9,11.9,True),(12,12,False),(2,8,False),(2,-1,False)):
            with self.subTest(elapsed=elapsed,wall=wall):
                fakeD=SimpleNamespace(datetime=SimpleNamespace(now=lambda tz:ref+datetime.timedelta(seconds=wall)),timezone=datetime.timezone)
                with patch.object(O,'time',SimpleNamespace(monotonic=lambda:elapsed)),patch.object(O,'D',fakeD):
                    if passes:O.Runtime.check(rt)
                    else:
                        with self.assertRaises(ValueError):O.Runtime.check(rt)


class TransportPureTests(unittest.TestCase):
    def test_root_wrapper_default_is_inert_without_clocks_or_subprocess(self):
        def prohibited(*args,**kwargs):raise AssertionError('live action prohibited in synthetic test')
        with patch.object(sys,'argv',['root_observe.py']),patch.object(R,'utc',prohibited),patch.object(R,'time',SimpleNamespace(monotonic=prohibited)),patch.object(R,'subprocess',SimpleNamespace(Popen=prohibited)),patch('builtins.print') as printed:
            R.main()
        self.assertEqual(json.loads(printed.call_args.args[0]),{'status':'inert_metadata_observer','scientific_execution':False,'clock_issued':False})
    def test_exact_command_bounds_and_quote_roundtrip(self):
        source='print("synthetic only")\n# quotes \' \" ` $() ; \\n'
        argv=R.command(source,'a'*64)
        self.assertEqual(argv[:4],['ssh','-T','-F',str(R.PREP/'ssh_config')])
        self.assertEqual(argv[-2],'teal-rat-80.coder')
        self.assertIn('ConnectTimeout=8',argv);self.assertIn('ConnectionAttempts=1',argv)
        remote=shlex.split(argv[-1])
        self.assertEqual(remote,['/usr/bin/timeout','--signal=KILL','20s',R.PYTHON,'-I','-S','-B','-c',source,'--observe','--contract-sha256','a'*64])
    def test_proxy_none_preserves_copy_only(self):
        original={'KEEP':'synthetic','HTTPS_PROXY':'existing'}
        with patch.object(R,'os',SimpleNamespace(environ=original)):
            result=R.proxy_environment(None)
        self.assertEqual(result,original);self.assertIsNot(result,original)
    def test_proxy_changes_only_requested_child_environment_keys(self):
        original={'KEEP':'synthetic','HTTPS_PROXY':'old','https_proxy':'synthetic-existing','NO_PROXY':'old-upper','no_proxy':'old-lower'}
        before=copy.deepcopy(original)
        with patch.object(R,'os',SimpleNamespace(environ=original)):
            result=R.proxy_environment('http://127.0.0.1:8765')
        self.assertEqual(result,{'KEEP':'synthetic','HTTPS_PROXY':'http://127.0.0.1:8765','https_proxy':'http://127.0.0.1:8765'})
        self.assertEqual(original,before)
    def test_invalid_proxy_scopes(self):
        bad=('https://127.0.0.1:8765','http://localhost:8765','http://127.0.0.2:8765',
             'http://a@127.0.0.1:8765','http://a:b@127.0.0.1:8765',
             'http://127.0.0.1:8765/','http://127.0.0.1:8765?x','http://127.0.0.1:8765#x',
             'http://127.0.0.1','http://127.0.0.1:0','http://127.0.0.1:65536','http://127.0.0.1:08765',
             ' HTTP://127.0.0.1:8765','HTTP://127.0.0.1:8765','http://127.0.0.1:\n8765',
             'http://127.0.0.1:\t8765')
        for value in bad:
            with self.subTest(value=value),patch.object(R,'os',SimpleNamespace(environ={})):
                with self.assertRaises(ValueError):R.proxy_environment(value)


if __name__=='__main__':unittest.main()
