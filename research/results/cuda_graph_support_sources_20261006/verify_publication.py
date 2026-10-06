#!/usr/bin/env python3
"""Verify four preparation families and optionally run their copied CPU tests."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
FAMILIES = ('cuda_graph_support_sources_20261006', 'goop2d_cuda_preparation_20261006',
            'goop3d_cuda_preparation_20261006', 'sand_graph_support_cuda_preparation_20261006')


def require(condition, message):
    if not condition: raise ValueError(message)


def stream_sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def sha(path):
    with path.open('rb') as f: return stream_sha(f)


def ignored(path):
    return any(part in ('__pycache__', '.pytest_cache') for part in path.parts)


def verify_family(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and p.name != 'manifest.json' and not ignored(p.relative_to(root))}
    require(actual == set(manifest['files']), 'Published file set differs: ' + root.name)
    for name, expected in manifest['files'].items():
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Unsafe published path')
        require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Changed file: ' + name)
    archive_count = 0
    if (root / 'archive_members.json').exists():
        members = json.loads((root / 'archive_members.json').read_text()); seen = set()
        with tarfile.open(root / 'raw_evidence.tar.gz', mode='r|gz') as archive:
            for member in archive:
                require(member.isfile() and member.name in members and member.name not in seen,
                        'Unexpected/duplicate/nonfile archive member')
                relative = Path(member.name)
                require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe archive member path')
                expected = members[member.name]
                with archive.extractfile(member) as stream:
                    require(member.size == expected['bytes'] and stream_sha(stream) == expected['sha256'],
                            'Archive member bytes differ: ' + member.name)
                seen.add(member.name)
        require(seen == set(members), 'Missing archive members')
        archive_count = len(seen)
        for name, provenance in json.loads((root / 'copy_provenance.json').read_text()).items():
            require(provenance['sha256'] == members[provenance['source_preparation_path']]['sha256'] == sha(root / name),
                    'Readable copy differs from archived source bytes')
    if (root / 'provenance.json').exists():
        for name, expected in json.loads((root / 'provenance.json').read_text()).items():
            require(sha(root / name) == expected['sha256'], 'Copied source provenance differs')
    return {'family': root.name, 'published_files_verified': len(actual), 'archive_members_verified': archive_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests', action='store_true')
    args = parser.parse_args()
    result = {'status': 'passed', 'families': [verify_family(HERE.parent / name) for name in FAMILIES]}
    sources = list((HERE / 'workspace').rglob('*.py'))
    for path in sources: ast.parse(path.read_text(), filename=str(path))
    result['python_sources_parsed'] = len(sources)
    if args.run_tests:
        with tempfile.TemporaryDirectory(prefix='adaptgns_cuda_preparation_cpu_') as tmp:
            workspace = Path(tmp) / 'workspace'
            shutil.copytree(HERE / 'workspace', workspace, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
            result['cpu_tests'] = {'files': [], 'scope': 'separate processes preserve import-isolation assertions; copied source, tiny synthetic tensors, generated records, mock HTTP/receipts; no real dataset/checkpoint/CUDA/remote work'}
            for test in sorted((workspace / 'work/deadline_research_20261005/cuda_preparation').glob('test_*.py')):
                command = [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '-q', str(test.relative_to(workspace))]
                completed = subprocess.run(command, cwd=workspace, env=environment, capture_output=True, text=True, timeout=120)
                result['cpu_tests']['files'].append({'test_file': test.name, 'returncode': completed.returncode,
                    'stdout': completed.stdout, 'stderr': completed.stderr})
                require(completed.returncode == 0, 'Packaged CPU tests failed:\n' + completed.stdout + completed.stderr)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
