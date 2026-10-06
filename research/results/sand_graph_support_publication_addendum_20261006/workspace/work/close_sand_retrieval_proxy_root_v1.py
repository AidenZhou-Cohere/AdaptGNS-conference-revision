from pathlib import Path
import datetime as D
import hashlib
import json
import sys

PREP = Path(__file__).resolve().parent / 'deadline_research_20261005/cuda_preparation'
sys.path.insert(0, str(PREP / 'goop3d_observed_history_analysis_UNADMITTED_transition_v1'))
from operational_checks import local_absence

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

launch_path = PREP / 'sand_recovery_proxy_original_launch_root_v1.json'
exit_path = PREP / 'sand_recovery_proxy_original_exit_root_v1.json'
launch, done = [json.loads(p.read_text()) for p in (launch_path, exit_path)]
assert launch['session_id'] == 62558 and done['exit_code'] == 0 and 'session_id' not in done
directory = PREP / 'sand_scalar_recovery_route_root_v1'
terminal_path = directory / 'terminal.json'
terminal = json.loads(terminal_path.read_text())
assert terminal['analysis_phase_sha256'] == '0ac7a368dd8afa921c8b973c5a8a760af7f10eb9f3413bceb592fff0c0ad7a5c'
assert terminal['remaining_worker_threads'] == 0
assert terminal['new_or_restarted_clock_granted'] is False
assert terminal['remote_processes_signaled'] is False
native = local_absence({48605}, directory)
report = {'schema':'sand_retrieval_proxy_final_closure_addendum_v1', 'utc':D.datetime.now(D.timezone.utc).isoformat(), 'original_session':62558, 'original_native_pid_pgid':48605, 'launch':{'path':str(launch_path),'sha256':sha(launch_path)},'original_exit':{'path':str(exit_path),'sha256':sha(exit_path)},'terminal':{'path':str(terminal_path),'sha256':sha(terminal_path)},'terminal_record':terminal,'native_closure':native,'scientific_operations_rerun':False,'status':'original_proxy_exited_zero_and_native_closed'}
out = PREP / 'sand_retrieval_proxy_final_closure_addendum_root_v1.json'
with out.open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'status':report['status']}))
