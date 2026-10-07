import ast,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import observe_closure as O
import root_close as R
C=json.loads((Path(__file__).parent/'contract.json').read_bytes())
class Fake:
 def __init__(self,values):self.values=values;self.calls=[]
 def proc(self,pid,name):
  self.calls.append((pid,name));value=self.values.get(name,FileNotFoundError())
  if isinstance(value,Exception):raise value
  return value
def stat(start):return ('79395 (python) S 79394 79394 79394 '+'0 '*15+start+'\n').encode()
class Tests(unittest.TestCase):
 def test_absent(self):self.assertEqual(O.sample_native(Fake({}),C['expected'][1])['status'],'absent')
 def test_reuse_no_cmdline(self):
  f=Fake({'stat':stat('999')});self.assertEqual(O.sample_native(f,C['expected'][1])['status'],'pid_reused');self.assertNotIn((79395,'cmdline'),f.calls)
 def test_permission_unknown(self):self.assertEqual(O.sample_native(Fake({'stat':PermissionError()}),C['expected'][1])['status'],'cannot_observe')
 def test_disappeared_during_read(self):self.assertEqual(O.sample_native(Fake({'stat':stat('831917341')}),C['expected'][1])['status'],'changed_during_observation')
 def test_exact_present(self):
  e=C['expected'][1];f=Fake({'stat':stat(e['start_id']),'cmdline':b'\0'.join(x.encode() for x in e['argv'])+b'\0'});r=O.sample_native(f,e);self.assertEqual(r['status'],'present');self.assertFalse(r['recorded_field_mismatches'])
 def test_inert_wrapper(self):
  with patch.object(sys,'argv',['root_close.py']),patch('sys.stdout',new_callable=io.StringIO) as out:R.main();self.assertEqual(json.loads(out.getvalue())['status'],'inert_metadata_observer')
 def test_no_remote_signals_or_subprocesses(self):
  tree=ast.parse(Path(O.__file__).read_text());calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)};self.assertFalse(calls&{'os.kill','os.killpg','subprocess.Popen','subprocess.run','os.system'})
if __name__=='__main__':unittest.main(verbosity=2)
