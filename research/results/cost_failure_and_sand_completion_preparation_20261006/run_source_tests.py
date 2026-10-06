#!/usr/bin/env python3
"""Extract the package into a temporary tree and run only CPU synthetic/source tests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MODULES = [
    'test_goop2d_validation_cost_v1', 'test_goop2d_validation_cost_v2',
    'test_goop2d_validation_cost_v3', 'test_goop2d_cost_independent_code_audit_v1',
    'test_goop2d_cost_independent_code_audit_v2', 'test_goop2d_validation_cost_summary_v1',
    'test_goop2d_cost_summary_independent_code_audit_v1',
    'test_goop2d_cost_reporting_integration_code_audit_v1',
    'test_goop2d_cost_reporting_integration_code_audit_v3',
    'test_goop2d_v3_pilot_lifecycle_transition_v1',
    'test_sand_recovered_cohort_bridge_independent_code_audit_v1',
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    verifier = load(package / 'verify_package.py', '_package_verifier')
    replay = load(package / 'replay_failure_evidence.py', '_failure_replay')
    with tempfile.TemporaryDirectory(prefix='cost_failure_package_') as temporary:
        workspace = Path(temporary) / 'extracted'
        verification = verifier.verify(package, workspace)
        evidence = replay.replay(workspace)
        prep = workspace / 'work/deadline_research_20261005/cuda_preparation'
        env = os.environ.copy()
        env.pop('PYTHONOPTIMIZE', None)
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(prep))
        command = [sys.executable, '-B', '-m', 'unittest', '-v', *MODULES]
        completed = subprocess.run(command, cwd=prep, env=env, capture_output=True, text=True)
        (args.output / 'source_tests.log').write_text(completed.stdout + completed.stderr)
        result = dict(status='passed' if completed.returncode == 0 else 'failed',
                      exit_code=completed.returncode, test_modules=MODULES,
                      archive_verification=verification, failure_evidence_replay=evidence,
                      git_mutation=False, remote_calls=False, numerical_arrays_loaded=False)
        (args.output / 'test_receipt.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
