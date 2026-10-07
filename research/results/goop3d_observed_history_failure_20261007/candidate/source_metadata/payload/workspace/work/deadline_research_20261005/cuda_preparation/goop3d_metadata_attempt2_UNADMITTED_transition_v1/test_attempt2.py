"""Pure source and synthetic checks; no live clocks, probes or operational main."""
import ast
import datetime as D
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import attempt2 as a

class Attempt2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common,cls.prepare=a.load_original()

    def test_original_package_all_hashes(self):
        self.assertEqual(self.prepare.package(a.PACKAGE_SHA)['files_sha256']['root_prepare.py'],self.common.digest((a.PACKAGE/'root_prepare.py').read_bytes()))

    def test_only_three_path_assignments(self):
        tree=ast.parse(Path(a.__file__).read_text())
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='bind_paths')
        writes=[ast.unparse(x.targets[0]) for x in ast.walk(function) if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Attribute)]
        self.assertEqual(writes,['prephase.PROXY_OUTPUT','prephase.OBSERVER_OUTPUT','module.PREP'])

    def test_original_entry_prep_used_only_for_output(self):
        tree=ast.parse((a.PACKAGE/'root_fresh_begin.py').read_text())
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        uses=[x for x in ast.walk(main) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load) and x.id=='PREP']
        self.assertEqual(len(uses),1)

    def test_observe_calls_original_module_and_paths(self):
        module=a.bind_paths('observe')
        self.assertEqual(Path(module.__file__).resolve(),a.PACKAGE/'root_fresh_observe.py')
        self.assertEqual(module.PROXY_OUTPUT,a.PROXY_OUTPUT)
        self.assertEqual(module.OBSERVER_OUTPUT,a.OBSERVER_OUTPUT)

    def test_begin_keeps_original_issuer_state_and_commands(self):
        module=a.bind_paths('begin')
        self.assertEqual(Path(module.__file__).resolve(),a.PACKAGE/'root_fresh_begin.py')
        self.assertIs(module.root_prepare,self.prepare)
        self.assertEqual(module.HERE,a.PACKAGE)
        self.assertEqual(module.PREP,a.ENTRY_PARENT)
        self.assertEqual(self.prepare.PREP,a.PREP)
        self.assertEqual(self.prepare.STATE,a.PREP/'goop3d_observed_history_analysis_released_root_v1')
        issued={n:n for n in ('phase_sha256','anchor_sha256','release_sha256','control_payload_sha256')}
        command=module.start_commands('python',a.PACKAGE_SHA,issued,Path('/phase'),Path('/sources'),'source_pin')
        self.assertEqual(command['phase_proxy_argv'][2],str(a.PACKAGE/'phase_proxy.py'))
        self.assertEqual(command['stage_and_pipeline_argv'][2],str(a.PACKAGE/'root_stage_and_run.py'))

    def test_attempt1_clock_rejected(self):
        start=D.datetime.fromisoformat('2026-10-06T23:24:52.247114+00:00')
        clock=D.datetime.fromisoformat('2026-10-06T23:25:02+00:00')
        with self.assertRaises(ValueError):self.prepare.fresh_times(start,clock,clock)

    def test_unchanged_freshness_and_agreement_boundaries(self):
        start=D.datetime.fromisoformat('2026-10-06T23:24:00+00:00')
        self.prepare.fresh_times(start,start+D.timedelta(seconds=5),start+D.timedelta(seconds=60))
        for clock,now in ((5.001,60),(0,60.001),(-.001,60),(0,-.001)):
            with self.assertRaises(ValueError):self.prepare.fresh_times(start,start+D.timedelta(seconds=clock),start+D.timedelta(seconds=now))

    def test_argument_forwarding(self):
        args=['--root-begin','--package-sha256',a.PACKAGE_SHA]
        self.assertEqual(a.original_argv('begin',args),['root_fresh_begin.py',*args])

    def test_no_phase_or_route_actions_in_wrapper_source(self):
        tree=ast.parse(Path(a.__file__).read_text())
        calls={ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)}
        self.assertFalse(calls&{'time.time','time.monotonic','subprocess.run','subprocess.Popen','os.execve','D.datetime.now'})

if __name__=='__main__':unittest.main()
