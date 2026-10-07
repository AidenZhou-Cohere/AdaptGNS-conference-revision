"""Review saved root startup metadata only; never probes a host or executes science."""
import hashlib
import json
from pathlib import Path
import shlex

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE/'root'
R = '/root/repos/AdaptGNS-cuda-20261006'
REMOTE = R+'/goop3d_completion_20261007'
plan = json.loads((BASE/'autonomous_v1/plan.json').read_text())
initial = json.loads((ROOT/'initial_host_observations.json').read_text())
checks, identities, pins = [], [], {}


def check(ok, what):
    if not ok:
        raise AssertionError(what)
    checks.append(what)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for host_index, key in enumerate(('teal', 'yellow', 'aquamarine')):
    path = ROOT/f'gpu_startup_host{host_index}.json'; pins[str(path)] = sha(path)
    envelope = json.loads(path.read_text())
    check(envelope['exit_code'] == 0 and envelope['stderr'] == '', 'actual metadata capture exit0 '+key)
    capture = json.loads(envelope['stdout'])
    prior = json.loads(initial[key]['value']['output'])
    check((capture['hostname'], capture['boot_id']) == (prior['host'], prior['boot_id']), 'original supplied host/boot '+key)
    gpu_rows = [line.split(', ') for line in prior['gpus'].splitlines() if line]
    gpu_map = {int(parts[0]): parts[2] for parts in gpu_rows}
    apps = [line.split(', ') for line in capture['gpu_processes'].splitlines() if line]
    check(len(apps) == 4, 'four actual GPU applications '+key)
    expected_apps = set()
    expected_jobs = {f'autonomous_worker{i:02d}_attempt1' for i in range(4*host_index, 4*host_index+4)}
    check(set(capture['jobs']) == expected_jobs, 'exact four launched jobs '+key)
    for name, job in capture['jobs'].items():
        spec = job['spec.json']; local = ROOT/(name+'.json'); launch_path = ROOT/(name+'.launch.json')
        launch = json.loads(launch_path.read_text()); response = json.loads(launch['stdout'])
        pins[str(local)] = sha(local); pins[str(launch_path)] = sha(launch_path)
        check(spec == json.loads(local.read_text()), 'local and observed spec match '+name)
        canonical = (json.dumps(spec, sort_keys=True, indent=2)+'\n').encode()
        spec_sha = hashlib.sha256(canonical).hexdigest()
        status = job['status.json']; owner = job['native']['owner']; child = job['native']['child']
        check(spec_sha == status['spec_sha256'] == response['spec_sha256'], 'exact published spec pin '+name)
        check(launch['exit_code'] == 0 and launch['stderr'] == '' and response['launched'] is True, 'original successful launch '+name)
        check(launch['sources']['autonomous_v1/run_worker.py'] == 'a018ed62c6a51511347194d36341c690faa3d8893552f46a7a825b7b90195034'
              and launch['sources']['autonomous_v1/plan.json'] == '838e7d4c1afc22fc4757a56ec0c755d5d303902d0ed600767d95c507d0402e08',
              'exact reviewed worker and plan deployment '+name)
        check(spec['hostname'] == response['hostname'] == capture['hostname'] and
              spec['boot_id'] == response['boot_id'] == capture['boot_id'], 'host/boot binding '+name)
        args = spec['argv']; index = int(args[args.index('--worker-index')+1]); cuda = int(args[args.index('--cuda-index')+1])
        device = args[args.index('--gpu-uuid')+1]
        check(index//4 == host_index and cuda == index%4 and device == gpu_map[cuda], 'physical GPU and worker association '+name)
        expected = [R+'/.venv/bin/python', '-B', '-u', REMOTE+'/autonomous_v1/run_worker.py', '--execute',
                    '--plan', REMOTE+'/autonomous_v1/plan.json', '--runtime-root', R if host_index == 0 else REMOTE+'/runtime_v1',
                    '--output-root', REMOTE+'/autonomous_results_v1', '--worker-index', str(index), '--cuda-index', str(cuda), '--gpu-uuid', device]
        check(args == expected and shlex.split(child['argv']) == args, 'exact initial worker argv; no resume '+name)
        check(spec['environment']['LD_LIBRARY_PATH'] == '/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'
              and spec['environment']['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
              and all(spec['environment'][k] == '1' for k in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
              'required deterministic driver and thread environment '+name)
        check(status['status'] == 'running' and owner['state'] not in ('Z', 'X') and child['state'] not in ('Z', 'X'), 'actual owner and child running '+name)
        check(owner['pid'] == response['owner_pid'] == status['owner']['pid'] and
              owner['start_ticks'] == response['owner_start_ticks'] == status['owner']['start_ticks'], 'original exact owner identity '+name)
        check(child['pid'] == status['child']['pid'] and child['start_ticks'] == status['child']['start_ticks']
              and child['ppid'] == owner['pid'] and child['pgid'] == owner['pgid'] == owner['pid'], 'owned child identity, PPID and group '+name)
        check(shlex.split(owner['argv']) == ['/usr/bin/python3', REMOTE+'/execution_v1/job_owner.py', REMOTE+'/jobs/'+name, spec_sha], 'exact native owner argv '+name)
        output = capture['outputs'][f'autonomous_results_v1/worker_{index:02d}']
        check(output['exists'] and output['status.json']['state'] == 'running'
              and output['status.json']['attempt'] == 'attempt_000001'
              and output['status.json']['committed_cell_ids'] == []
              and output['status.json']['current_cell_id'] == plan['workers'][index]['cells'][0]['cell_id'],
              'early status matches first assigned missing cell '+name)
        check(job['owner.stderr'] == '' and 'Traceback (most recent call last)' not in job['worker.stderr'], 'no captured traceback/owner error '+name)
        expected_apps.add((device, child['pid'], R+'/.venv/bin/python'))
        identities.append({'host': capture['hostname'], 'boot_id': capture['boot_id'], 'worker_index': index,
                           'owner_pid': owner['pid'], 'owner_start_ticks': owner['start_ticks'],
                           'child_pid': child['pid'], 'child_start_ticks': child['start_ticks'], 'gpu_uuid': device,
                           'first_cell': output['status.json']['current_cell_id'], 'source_spec_sha256': spec_sha})
    check({(parts[0], int(parts[1]), parts[2]) for parts in apps} == expected_apps, 'GPU application identities exactly equal four expected children '+key)

check({x['worker_index'] for x in identities} == set(range(12)), 'all12 worker indices exactly once')
check(len({(x['host'], x['child_pid'], x['child_start_ticks']) for x in identities}) == 12, 'unique12 actual child identities')
report = {'status': 'passed_saved_startup_association_review', 'checks': len(checks), 'workers': 12,
          'capture_sha256': pins, 'native_identities': identities, 'remote_actions': False,
          'scientific_data_or_model_read': False, 'scientific_result_admission': False,
          'limitations': ['Saved startup snapshot only, not continuous ownership or final closure.',
                         'Source pins in original launch records bind deployed worker/plan; root retains complete stage/runtime input checks.',
                         'Legacy invalid-escape and nonwritable-NumPy tensor warnings remain in saved stderr; no numerical-failure inference.']}
out = BASE/'review_v1/autonomous_startup_review.json'; out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps({'status': report['status'], 'checks':len(checks), 'review_sha256':sha(out)}))
