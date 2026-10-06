#!/usr/bin/env python3
"""Reconstruct exact published source dependencies and run CPU synthetic checks."""
import argparse, hashlib, json, pathlib, shutil, subprocess, sys, tempfile
P = pathlib.Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Fresh output required'
    args.output.mkdir(parents=True)
    registry = json.loads((P/'dependency_paths.json').read_text())['files']
    source_map = {}
    with tempfile.TemporaryDirectory(prefix='adaptgns_recovery_package_') as tmp:
        root = pathlib.Path(tmp)
        for rel, row in registry.items():
            source = (P/row['published_path']).resolve()
            assert source.stat().st_size == row['bytes'] and sha(source) == row['sha256'], rel
            target = root/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            source_map[rel] = row['sha256']
        prep = root/'work/deadline_research_20261005/cuda_preparation'
        files = ['test_sand_runtime_migration_adapter_v1.py', 'test_sand_runtime_migration_endpoint_audit_v1.py',
                 'test_supervise_sand_runtime_migration_recovery_v1.py', 'test_supervise_sand_runtime_migration_recovery_v2.py',
                 'test_sand_recovery_v2_process_scan_order.py']
        command = [args.python, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *[str(prep/f) for f in files]]
        r = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=120)
        (args.output/'cpu_tests.stdout').write_text(r.stdout)
        (args.output/'cpu_tests.stderr').write_text(r.stderr)
        assert r.returncode == 0, r.stdout+r.stderr
        defaults = {}
        for name in ['train_sand_runtime_migration_recovery_v1.py', 'audit_sand_runtime_migration_endpoints_v1.py',
                     'supervise_sand_runtime_migration_recovery_v1.py', 'supervise_sand_runtime_migration_recovery_v2.py']:
            r = subprocess.run([args.python, '-B', str(prep/name)], cwd=root, text=True, capture_output=True, timeout=30)
            assert r.returncode == 0, (name,r.stdout,r.stderr)
            defaults[name] = {'exit_code': r.returncode, 'stdout': json.loads(r.stdout), 'stderr': r.stderr}
        for rel, expected in source_map.items():
            assert sha(root/rel) == expected, rel
        (args.output/'verification.json').write_text(json.dumps({'status':'passed', 'reconstructed_file_count':len(registry),
            'all_input_bytes_unchanged':True, 'cpu_test_stdout_sha256':sha(args.output/'cpu_tests.stdout'),
            'inert_cli_defaults':defaults, 'models_data_cuda_remote_access':False}, indent=2)+'\n')
        print(json.dumps({'status':'passed','reconstructed_files':len(registry),'tests':(args.output/'cpu_tests.stdout').read_text().strip()}))
if __name__ == '__main__': main()
