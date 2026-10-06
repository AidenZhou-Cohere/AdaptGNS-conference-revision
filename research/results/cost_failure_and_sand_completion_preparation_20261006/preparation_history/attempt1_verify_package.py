#!/usr/bin/env python3
"""Verify exact archive members and optionally extract reviewed text artifacts."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def verify(package, extract_to=None):
    manifest = json.loads((package / 'archive_manifest.json').read_text())
    if extract_to is not None:
        extract_to.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    for row in manifest['archives'].values():
        path = package / row['file']
        if path.stat().st_size != row['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Archive hash or size differs: ' + row['file'])
        expected = {x['path']: x for x in row['members']}
        if len(expected) != row['member_count']:
            raise ValueError('Duplicate manifest members')
        observed = set()
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if not member.isfile() or relative.is_absolute() or '..' in relative.parts or member.name not in expected:
                    raise ValueError('Unexpected or unsafe archive member')
                if member.name in seen or relative.suffix in ('.pt', '.npy', '.npz', '.tar'):
                    raise ValueError('Duplicate member or raw numerical payload')
                raw = archive.extractfile(member).read()
                raw.decode('utf-8')
                pin = expected[member.name]
                if len(raw) != pin['size_bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
                    raise ValueError('Archive member hash differs: ' + member.name)
                if extract_to is not None:
                    target = extract_to.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as output:
                        output.write(raw)
                observed.add(member.name)
                seen.add(member.name)
                total += len(raw)
        if observed != set(expected):
            raise ValueError('Missing archive members')
    return dict(status='all_archive_bytes_verified', archives=len(manifest['archives']),
                text_files=len(seen), original_text_bytes=total, raw_numerical_payloads=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--extract-to', type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.package.resolve(), a.extract_to.resolve() if a.extract_to else None), indent=2))
