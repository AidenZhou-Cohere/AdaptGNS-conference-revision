#!/usr/bin/env python3
"""Verify this frozen package and optionally replay ten isolated CPU test files."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
PREP = Path('work/deadline_research_20261005/cuda_preparation')
CAPACITY = 'archives/goop3d_vectorized_capacity_v1_scalar_snapshot.tar.gz'
CAPACITY_PREFIX = 'goop3d_vectorized_capacity_20261006_v1/'
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.tfrecord', '.tfrecords', '.pkl', '.pickle', '.pyc'}
LIVE_NAMES = {'goop3d_validation_timing_v1.stdout.log', 'goop3d_validation_timing_v1.stderr.log'}
# High-confidence credential markers only; values are never printed on failure.
SECRET_PATTERNS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE' + r' KEY-----'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{25,}\b'),
    re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b'),
    re.compile(r'https?://[^\s/@:]+:[^\s/@]+@'),
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_info(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': digest(data)}


def safe_path(name):
    p = Path(name)
    require(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name,
            'Unsafe relative path: ' + name)
    require(not any(part in ('.git', '.ssh', '__pycache__', '.pytest_cache') for part in p.parts),
            'Forbidden path: ' + name)
    require(p.suffix.lower() not in FORBIDDEN_SUFFIXES and p.name not in LIVE_NAMES,
            'Excluded payload or live output: ' + name)


def safe_text(name, data):
    safe_path(name)
    try:
        value = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError('Unexpected nontext content: ' + name) from exc
    require('\x00' not in value, 'Unexpected binary content: ' + name)
    require(not any(pattern.search(value) for pattern in SECRET_PATTERNS),
            'Credential-like content requires private inspection: ' + name)


def verify_package(check_originals=None, check_predecessors=False):
    manifest = read(HERE / 'manifest.json')
    paths = list(HERE.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Symlink in public package')
    actual = {str(p.relative_to(HERE)) for p in paths if p.is_file() and p != HERE / 'manifest.json'}
    require(actual == set(manifest['files']), 'Published file set differs from manifest')
    catalogs = read(HERE / 'archive_catalogs.json')
    text_count = 0
    for name, expected in manifest['files'].items():
        safe_path(name)
        require(file_info(HERE / name) == expected, 'Changed published bytes: ' + name)
        if name not in catalogs:
            safe_text(name, (HERE / name).read_bytes())
            text_count += 1
    safe_text('manifest.json', (HERE / 'manifest.json').read_bytes())
    archive_files, archive_dirs, archived_bytes = 0, 0, 0
    for name, catalog in catalogs.items():
        require(file_info(HERE / name) == catalog['archive'], 'Changed original archive: ' + name)
        seen = set()
        with tarfile.open(HERE / name, 'r:*') as archive:
            for member in archive:
                safe_path(member.name)
                require(member.name not in seen and member.name in catalog['members'], 'Duplicate/unlisted archive member')
                expected = catalog['members'][member.name]
                require(member.isfile() or member.isdir(), 'Archive links/special files are forbidden')
                require(member.size == expected['bytes'], 'Changed archive member size')
                if member.isdir():
                    require(expected['type'] == 'directory' and member.size == 0, 'Bad directory member')
                    archive_dirs += 1
                else:
                    with archive.extractfile(member) as stream:
                        data = stream.read()
                    require(expected['type'] == 'file' and digest(data) == expected['sha256'], 'Changed archive member bytes')
                    safe_text(member.name, data)
                    archive_files += 1
                    archived_bytes += len(data)
                seen.add(member.name)
        require(seen == set(catalog['members']), 'Missing archive members')
    provenance = read(HERE / 'provenance.json')
    original_checks = 0
    for name, row in provenance.items():
        require(file_info(HERE / name) == {k: row[k] for k in ('bytes', 'sha256')}, 'Copied provenance mismatch: ' + name)
        if check_originals is not None:
            origin = check_originals / row['source_workspace_path']
            require(file_info(origin) == file_info(HERE / name), 'Original provenance mismatch: ' + name)
            original_checks += 1
    predecessor_checks = 0
    if check_predecessors:
        for family, files in read(HERE / 'predecessors.json').items():
            for name, expected in files.items():
                require(file_info(HERE.parent / family / name) == expected, 'Predecessor changed: ' + family + '/' + name)
                predecessor_checks += 1
    # Exact public scalar archive equals the stopped full inventory after omitting only tensor files.
    full = read(HERE / 'records/goop3d_capacity_final_artifact_inventory_v1.json')['files']
    omission = read(HERE / 'omissions.json')
    capmembers = catalogs[CAPACITY]['members']
    regular = {n[len(CAPACITY_PREFIX):]: v for n, v in capmembers.items() if v['type'] == 'file'}
    require(len(regular) == 74 and len(omission['capacity_checkpoints']) == 12, 'Capacity evidence count changed')
    require(set(regular) | set(omission['capacity_checkpoints']) == set(full), 'Capacity inventory coverage changed')
    require(not set(regular) & set(omission['capacity_checkpoints']), 'Omitted payload included')
    for name, row in regular.items():
        require(row['sha256'] == full[name]['sha256'] and row['bytes'] == full[name]['size_bytes'], 'Capacity artifact mismatch: ' + name)
    for name, row in omission['capacity_checkpoints'].items():
        require(row == full[name], 'Capacity omitted reference changed')
    numerical = read(HERE / 'records/goop3d_numerical_remote_artifact_audit_v1.json')['artifacts']
    require(numerical == omission['numerical_payloads'] and len(numerical) == 32, 'Numerical omitted references changed')
    summary = omission['capacity_summary']
    require(regular['summary.json']['sha256'] == summary['sha256'] and regular['summary.json']['bytes'] == summary['bytes'], 'Summary location mismatch')
    failed = HERE / 'records/third_vm_pre_capacity_inventory_v1.json'
    require(failed.read_bytes() == b'', 'Historical failed inventory must remain opaque zero bytes')
    failure_record = read(HERE / 'records/goop3d_capacity_preflight_failure_record_v1.json')
    require(failure_record['output_is_valid_json'] is False and failure_record['capacity_workers_launched'] is False, 'Historical failure scope changed')
    # Fixed launch declarations remain distinct from scientific admission.
    contracts = HERE / 'launch_declarations/goop3d_timing_contracts_local_v1'
    review = read(HERE / 'records/goop3d_timing_root_contract_review_v1.json')
    for name, key in (('supervisor_release.json', 'supervisor_release_sha256'), ('preparation_report.json', 'preparation_report_sha256')):
        require(file_info(contracts / name)['sha256'] == review[key], 'Timing contract review pin changed')
    template_root = HERE / 'workspace' / PREP
    endpoint = read(template_root / 'goop3d_scientific_endpoint_plan.template.json')
    require(endpoint['endpoint_updates'] is None and endpoint['status'] == 'not_selected',
            'Scientific endpoint template is no longer inert')
    for name in ('goop3d_scientific_training_admission.template.json',
                 'goop3d_scientific_release.template.json', 'goop3d_deadline_worksheet_release.template.json'):
        template = read(template_root / name)
        require(template['scientific_training_admitted'] is False and template['status'] == 'not_admitted',
                'Scientific/planning admission template is no longer inert: ' + name)
    planning = read(template_root / 'goop3d_deadline_planning_config.template.json')
    require(planning['selected_endpoint'] is None and planning['planning_start_utc'] is None
            and planning['scientific_training_admitted'] is False, 'Decision-time planning template changed scope')
    timing = read(contracts / 'supervisor_release.json')
    require(timing['scientific_training_admitted'] is False and timing['scientific_endpoint_selected'] is False
            and timing['test_access_allowed'] is False, 'Timing contract acquired scientific or test scope')
    parsed = 0
    for source in (HERE / 'workspace').rglob('*.py'):
        ast.parse(source.read_text(), filename=str(source))
        parsed += 1
    return {'status': 'passed', 'files_verified': len(actual), 'text_files_scanned': text_count + 1,
            'copied_provenance_files': len(provenance), 'originals_rechecked': original_checks,
            'predecessor_records_rechecked': predecessor_checks, 'archives_verified': len(catalogs),
            'archive_regular_files_verified': archive_files, 'archive_directories_verified': archive_dirs,
            'archived_text_bytes': archived_bytes, 'python_sources_parsed_without_import': parsed,
            'scientific_endpoint_selected': False, 'mutable_timing_results_included': False}


def run_tests():
    inventory = read(HERE / 'test_inventory.json')
    report = {'files': [], 'expected_test_count': inventory['total'],
              'scope': 'separate fresh CPU processes; synthetic/mock/generated fixtures only; no official numeric data, remote calls, or CUDA execution'}
    with tempfile.TemporaryDirectory(prefix='adaptgns_goop3d_vectorized_cpu_') as directory:
        workspace = Path(directory) / 'workspace'
        shutil.copytree(HERE / 'workspace', workspace)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        # No CUDA mask override: reviewed positive-gate fixtures control this variable.
        for name, expected_count in inventory['files'].items():
            command = [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '-q', str(PREP / name)]
            completed = subprocess.run(command, cwd=workspace, env=environment, capture_output=True, text=True, timeout=180)
            counts = re.findall(r'(\d+) passed', completed.stdout)
            report['files'].append({'name': name, 'expected_count': expected_count, 'returncode': completed.returncode,
                                    'stdout': completed.stdout, 'stderr': completed.stderr,
                                    'passed_count': int(counts[-1]) if counts else 0})
        report['passed_count'] = sum(x['passed_count'] for x in report['files'])
        report['status'] = 'passed' if all(x['returncode'] == 0 and x['passed_count'] == x['expected_count'] for x in report['files']) else 'failed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests', action='store_true')
    parser.add_argument('--check-originals', type=Path, help='Optional original workspace root; never accesses remote paths')
    parser.add_argument('--check-predecessors', action='store_true')
    args = parser.parse_args()
    report = verify_package(args.check_originals, args.check_predecessors)
    report['created_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.run_tests:
        report['cpu_tests'] = run_tests()
        report['status'] = report['cpu_tests']['status']
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
