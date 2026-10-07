"""Synthetic combined staging, unchanged launch gates, and prepared argv."""
import base64
import contextlib
import datetime as D
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import stage_payload
import staging_plan
import root_fresh_begin
import root_stage_and_run
from root_prepare import fresh_times

HERE=Path(__file__).resolve().parent
START=D.datetime.fromisoformat('2026-10-06T23:00:00+00:00')

def row(path,raw):
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}

class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='d3_combined_stage_synthetic_');self.root=Path(self.temp.name).resolve()
    def tearDown(self):self.temp.cleanup()
    def remote(self,kind='combined',elapsed=1):
        spec=importlib.util.spec_from_file_location('synthetic_combined_stage',HERE/'stage_payload.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        m.ROOT=str(self.root);m.SOURCE=str(self.root/'sources');m.OUTPUT=str(self.root/'outputs')
        boot=self.root/'boot';boot.write_text('4419f0f5-8f54-4cfd-9306-f4a9fc0dd5e6')
        m.Path=lambda p:boot if str(p)=='/proc/sys/kernel/random/boot_id' else Path(p)
        m.socket=SimpleNamespace(gethostname=lambda:'aidenzhou-teal-rat-80-5c78ffbffc-g8djt')
        class FixedDateTime(D.datetime):
            @classmethod
            def now(cls,tz=None):return START+D.timedelta(seconds=elapsed)
        m.D=SimpleNamespace(datetime=FixedDateTime,timezone=D.timezone)
        m.time=SimpleNamespace(monotonic=lambda:1000+elapsed)
        phase={'schema':'goop3d_observed_history_analysis_phase_v1','issued_by':'root','status':'approved_bounded_observed_history_analysis','seconds':3600,'started_utc':START.isoformat(),'clock_sample':{'host_monotonic_seconds':1000}}
        rows=[row(Path(m.SOURCE)/name,('synthetic source '+name).encode()) for name in sorted(m.SOURCE_NAMES)]
        rows += [row(Path(m.OUTPUT)/'controls'/name,json.dumps(phase if name=='analysis_phase.json' else {'synthetic':name}).encode()) for name in sorted(m.CONTROL_NAMES)]
        payload={'schema':'goop3d_observed_history_staging_payload_v1','kind':kind,'files':rows}
        return m,payload
    def execute(self,m,payload):
        raw=json.dumps(payload).encode();m.sys=SimpleNamespace(argv=['synthetic','--stage','--payload-sha256',hashlib.sha256(raw).hexdigest()],flags=SimpleNamespace(isolated=True,no_site=True,dont_write_bytecode=True),stdin=SimpleNamespace(buffer=io.BytesIO(raw)))
        out=io.StringIO()
        with contextlib.redirect_stdout(out):m.main()
        return json.loads(out.getvalue())
    def test_all13_sources_and4_controls_stage_together(self):
        m,payload=self.remote();result=self.execute(m,payload)
        self.assertEqual(result['kind'],'combined');self.assertEqual(len(result['files_sha256']),17)
        self.assertFalse(result['worker_started'])
        for r in payload['files']:self.assertEqual(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(),r['sha256'])
    def test_prephase_source_only_staging_refused(self):
        m,payload=self.remote(kind='sources')
        with self.assertRaisesRegex(ValueError,'admitted phase'):self.execute(m,payload)
        self.assertFalse(Path(m.SOURCE).exists());self.assertFalse(Path(m.OUTPUT).exists())
    def test_remote_staging_at100seconds_refused_before_writes(self):
        m,payload=self.remote(elapsed=100)
        with self.assertRaisesRegex(ValueError,'initial100seconds'):self.execute(m,payload)
        self.assertFalse(Path(m.SOURCE).exists());self.assertFalse(Path(m.OUTPUT).exists())
    def test_missing_combined_member_refused(self):
        m,payload=self.remote();payload['files'].pop()
        with self.assertRaisesRegex(ValueError,'combined source/control allowlist'):self.execute(m,payload)
    def test_conflicting_staged_source_preserved(self):
        m,payload=self.remote();path=Path(payload['files'][0]['path']);path.parent.mkdir();path.write_bytes(b'preserve conflicting source')
        with self.assertRaisesRegex(ValueError,'changed existing staged file'):self.execute(m,payload)
        self.assertEqual(path.read_bytes(),b'preserve conflicting source')
    def local_payloads(self):
        rows=[row(staging_plan.SOURCE+'/'+name,('synthetic source '+name).encode()) for name in sorted(stage_payload.SOURCE_NAMES)]
        sources={'schema':'goop3d_observed_history_staging_payload_v1','kind':'sources','files':rows}
        bindings={'files_sha256':{Path(r['path']).name:r['sha256'] for r in rows}}
        controls={'schema':sources['schema'],'kind':'controls','files':[row(staging_plan.OUTPUT+'/controls/'+name,json.dumps({'synthetic':name}).encode()) for name in sorted(stage_payload.CONTROL_NAMES)]}
        pins={Path(r['path']).name:r['sha256'] for r in controls['files']}
        return json.dumps(sources).encode(),json.dumps(controls).encode(),bindings,pins['analysis_phase.json'],pins['pipeline.cpu_release.json']
    def test_local_combination_binds_exact_current_phase_and_release(self):
        args=self.local_payloads();result=staging_plan.combined_payload(*args)
        self.assertEqual(result['kind'],'combined');self.assertEqual(len(result['files']),17)
        with self.assertRaises(ValueError):staging_plan.combined_payload(*args[:3],'f'*64,args[4])
    def test_prepared_start_commands_preserve_pins_and_no_arbitrary_operation(self):
        issued={'phase_sha256':'1'*64,'anchor_sha256':'2'*64,'release_sha256':'3'*64,'control_payload_sha256':'4'*64}
        result=root_fresh_begin.start_commands('/synthetic/python','5'*64,issued,self.root/'phase proxy',self.root/'source payload','6'*64)
        proxy=result['phase_proxy_argv'];start=result['stage_and_pipeline_argv']
        self.assertEqual(proxy[2],str(HERE/'phase_proxy.py'));self.assertEqual(start[2],str(HERE/'root_stage_and_run.py'))
        self.assertIn('--root-start',start);self.assertNotIn('--operation',start)
        for key,pin in issued.items():self.assertEqual(start[start.index('--'+key.replace('_','-'))+1],pin)
        self.assertEqual(start[start.index('--source-payload')+1],str(self.root/'source payload'))
        run=root_stage_and_run.run_argv('/synthetic/python','a','b','c','d',self.root/'ready','e')
        self.assertEqual(run[2],str(HERE/'root_run.py'));self.assertIn('--root-run',run);self.assertNotIn('--operation',run)
    def test_new_ready_metadata_rejects_symlink(self):
        path=self.root/'ready';path.write_bytes(b'{"synthetic":true}')
        raw,pin=root_stage_and_run.read_new_metadata(path)
        self.assertEqual(pin,hashlib.sha256(raw).hexdigest())
        link=self.root/'link';link.symlink_to(path)
        with self.assertRaises((ValueError,OSError)):root_stage_and_run.read_new_metadata(link)
    def test_fresh_age60_is_distinct_from_clock_agreement5(self):
        fresh_times(START,START+D.timedelta(seconds=1),START+D.timedelta(seconds=59))
        with self.assertRaises(ValueError):fresh_times(START,START,START+D.timedelta(seconds=61))
        with self.assertRaises(ValueError):fresh_times(START,START+D.timedelta(seconds=6),START+D.timedelta(seconds=40))
        with self.assertRaises(ValueError):fresh_times(START,START,START-D.timedelta(seconds=1))

if __name__=='__main__':unittest.main(verbosity=2)
