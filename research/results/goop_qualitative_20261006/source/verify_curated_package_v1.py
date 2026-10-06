#!/usr/bin/env python3
"""Verify a curated package and reconstruct its canonical TeX entirely in memory."""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json


def verify(package):
    sha = lambda b: hashlib.sha256(b).hexdigest()
    manifest = json.loads((package / 'manifest.json').read_text())
    listed = {item['path']: item for item in manifest['files']}
    actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    assert actual == set(listed) | {'manifest.json'}
    assert not any(p.is_symlink() for p in package.rglob('*'))
    for name, item in listed.items():
        data = (package / name).read_bytes()
        assert len(data) == item['bytes'] and sha(data) == item['sha256'], name
        assert not name.endswith(('.npz', '.tar', '.tar.gz', '.pem', '.key'))
    provenance = json.loads((package / 'copy_provenance.json').read_text())
    for copy in provenance['exact_copies']:
        assert listed[copy['package_path']]['sha256'] == copy['source_sha256']
        assert listed[copy['package_path']]['bytes'] == copy['source_bytes']

    receipt = json.loads((package / 'verification/applied_integration_independent_check_v1.json').read_text())
    figure = (package / 'presentation/goop_fixed_source12_inline_v1.tex').read_bytes()
    note = (package / 'presentation/integration_note_v1.tex').read_bytes()
    manuscript = (package / 'presentation/revised_manuscript.tex').read_bytes()
    before = (package / 'canonical/manuscript_body.before.tex').read_bytes()
    body = before.replace(b'\\end{document}', note + b'\n' + figure + b'\n\\end{document}')
    assert sha(body) == receipt['canonical_body_sha256']
    assert sha(manuscript) == receipt['manuscript_sha256']
    assert manuscript.count(figure) == manuscript.count(note) == 1
    for item in provenance['exact_snippets']:
        source = (package / item['package_source']).read_bytes()
        start, stop = item['source_byte_range_half_open']
        assert (package / item['package_path']).read_bytes() == source[start:stop]

    # Snapshot bytes first, then prevent the builder from reading or writing
    # outside this in-memory mapping. No TeX compiler or network is involved.
    canonical = {}
    for name, expected in receipt['canonical_inputs_sha256'].items():
        data = body if name == 'work/manuscript_body.tex' else (package / 'canonical' / name).read_bytes()
        assert sha(data) == expected, name
        canonical[name] = data.decode('utf-8')
    builder = (package / 'canonical/work/build_manuscript.py').read_bytes()
    assert sha(builder) == receipt['canonical_builder_sha256']
    captured = {}

    def get_text(path, *args, **kwargs):
        assert str(path) in canonical, path
        return canonical[str(path)]

    def exists(path):
        return str(path) in canonical

    def write_text(path, text, *args, **kwargs):
        assert str(path) == 'outputs/revised_manuscript.tex'
        assert not captured
        captured['bytes'] = text.encode('utf-8')
        return len(text)

    with patch.object(Path, 'read_text', get_text), patch.object(Path, 'exists', exists), patch.object(Path, 'write_text', write_text):
        exec(compile(builder, 'canonical/work/build_manuscript.py', 'exec'), {'__name__': '__main__'})
    assert captured['bytes'] == manuscript
    return {
        'passed': True,
        'manifest_sha256': sha((package / 'manifest.json').read_bytes()),
        'verified_manifest_files': len(listed),
        'payload_bytes': sum(item['bytes'] for item in listed.values()),
        'exact_copy_provenance_records': len(provenance['exact_copies']),
        'exact_snippets_checked': len(provenance['exact_snippets']),
        'canonical_source_reconstruction_exact': True,
        'canonical_build_matches_packaged_standalone_exact_bytes': True,
        'manuscript_sha256': sha(manuscript),
        'raw_arrays_required_for_reexport': True,
        'native_compilation_performed': False,
        'filesystem_mutations': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.package.resolve()), indent=2))
