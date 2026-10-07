#!/usr/bin/env python3
"""Verify an explicit package plan; --copy writes a fresh package, never Git/network."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(doc):
    return (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def credential_patterns(raw):
    text = raw.decode('utf-8')
    patterns = {
        'private_key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
        'openai_key': r'\bsk-[A-Za-z0-9_-]{24,}',
        'github_key': r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}',
        'aws_access_key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        'bearer_credential': r'(?i)authorization["\\\s:]+bearer\s+[A-Za-z0-9._~+/-]{12,}',
        'literal_password': r'(?i)["\'](?:password|api_key|access_token|secret_access_key)["\']\s*:\s*["\'][^"\']{8,}["\']',
    }
    return [label for label, pattern in patterns.items() if re.search(pattern, text)]


def validate(plan, workspace):
    need(plan['schema'] == 'goop3d_completion_publication_plan_v1', 'plan schema differs')
    need(plan['package'] == 'research/results/goop3d_completion_20261007', 'unexpected package target')
    entries = plan['files']
    need(entries and len({e['destination'] for e in entries}) == len(entries), 'empty or duplicate destinations')
    for entry in entries:
        relative = Path(entry['source'])
        target = Path(entry['destination'])
        need(not relative.is_absolute() and '..' not in relative.parts, 'source escapes workspace')
        need(target.parts and not target.is_absolute() and '..' not in target.parts and str(target) not in ('.', 'publication_manifest.json', 'curation_plan.json'), 'unsafe or reserved destination')
        path = workspace / relative
        need(path.resolve().is_relative_to(workspace) and path.is_file() and not path.is_symlink(), 'ordinary workspace source required')
        need(all(not parent.is_symlink() for parent in path.parents if parent.is_relative_to(workspace)), 'symlink source parent prohibited')
        need(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'source bytes changed: ' + entry['source'])
        need(path.suffix not in ('.npz', '.npy', '.pt', '.pkl', '.gz', '.tar', '.zip', '.pdf'), 'raw arrays/models/archives/PDF outside this package plan')
        need(entry['bytes'] <= (100 << 20) if entry['role'] == 'verified_observed_product' else entry['bytes'] <= (2 << 20), 'per-file publication limit')
        matches = credential_patterns(path.read_bytes())
        need(not matches, 'credential-pattern match in ' + entry['source'] + ': ' + ','.join(matches))
    return entries


def validate_gate(plan, workspace):
    gate = plan['numerical_product_gate']
    need(gate['status'] == 'reviewed_for_publication', 'observed numerical products await exact independent review')
    entries = {entry['destination']: entry for entry in plan['files']}

    def bound(binding):
        entry = entries[binding['destination']]
        need(entry['sha256'] == binding['sha256'], 'gate pin differs from included file')
        return json.loads((workspace / entry['source']).read_bytes())

    review = bound(gate['review'])
    receipt = bound(gate['transfer_manifest'])
    admission = bound(gate['admission'])
    need(review['status'] == 'passed_independent_actual_product_scalar_review', 'actual product review did not pass')
    need(review['actual_original_exit'] == 0 and review['cached_rows_bound_exactly'] == 2568
         and review['all_original_accounting_states'] == 4728
         and review['original36rounding_mismatches_preserved_and_resolved'] is True, 'observed review scope or outcome differs')
    need(receipt['actual_status']['status'] == 'exited' and receipt['actual_status']['returncode'] == 0
         and receipt['remaining_group_members'] == [], 'actual process completion not established')
    need(admission['status'] == 'admitted_observed_history_only'
         and admission['independent_review_sha256'] == gate['review']['sha256']
         and admission['autonomous_scientific_admission'] is False, 'observed-only admission required')
    expected = dict(review['products_sha256'], **{'completion.json': review['completion_sha256']})
    need(set(expected) == {'audit.json', 'summary.json', 'arithmetic_check.json', 'completion.json', 'retained_original_failure.json'}, 'observed product set differs')
    need(set(gate['products']) == set(expected), 'gate product set differs')
    for name, digest in expected.items():
        binding = gate['products'][name]
        need(binding['sha256'] == digest, 'review/product pin differs: ' + name)
        entry = entries[binding['destination']]
        need(entry['sha256'] == digest and entry['role'] == 'verified_observed_product', 'included product differs')
        need(receipt['files']['observed_finalized_v2/' + name] == {'bytes': entry['bytes'], 'sha256': digest}, 'transport/product binding differs')
    need(admission['completion_sha256'] == expected['completion.json'] and admission['summary_sha256'] == expected['summary.json'], 'admission/product binding differs')
    need({e['destination'] for e in plan['files'] if e['role'] == 'verified_observed_product'}
         == {b['destination'] for b in gate['products'].values()}, 'unreviewed numerical product role')
    return gate


def durable_write(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute(plan_path, plan_sha, workspace, destination, do_copy):
    need(sha(plan_path) == plan_sha, 'exact reviewed plan hash required')
    plan = json.loads(plan_path.read_bytes())
    workspace = workspace.resolve()
    entries = validate(plan, workspace)
    gate = validate_gate(plan, workspace)
    if not do_copy:
        return {'status': 'verified_sources', 'copy_ready': True, 'files': len(entries),
                'bytes': sum(e['bytes'] for e in entries), 'numerical_product_gate': gate['status']}
    dest = destination.resolve()
    need(destination == dest and dest.name == 'goop3d_completion_20261007'
         and dest.parent.name == 'results' and dest.parent.parent.name == 'research', 'canonical research/results package destination required')
    need(not dest.exists(), 'fresh package only; preserve existing publication')
    need(not plan_path.resolve().is_relative_to(dest) and all(not (workspace / e['source']).resolve().is_relative_to(dest) for e in entries), 'source/destination overlap prohibited')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=dest.name + '.partial.', dir=dest.parent))
    try:
        for entry in entries:
            source, target = workspace / entry['source'], temporary / entry['destination']
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
                dst.flush(); os.fsync(dst.fileno())
            need(target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'], 'copied bytes differ')
        raw_plan = plan_path.read_bytes()
        need(hashlib.sha256(raw_plan).hexdigest() == plan_sha, 'plan changed during copy')
        durable_write(temporary / 'curation_plan.json', raw_plan)
        manifest = {'schema': 'goop3d_completion_publication_manifest_v1', 'plan_sha256': plan_sha,
                    'package': plan['package'], 'status': 'reviewed_code_and_observed_products_copied',
                    'files': {e['destination']: {'sha256': e['sha256'], 'bytes': e['bytes'], 'role': e['role']} for e in entries},
                    'curation_plan': {'sha256': plan_sha, 'bytes': len(raw_plan)},
                    'numerical_product_gate': gate, 'pending': plan['pending'],
                    'no_git_or_network_operation': True, 'raw_arrays_models_and_caches_not_copied': True}
        durable_write(temporary / 'publication_manifest.json', encode(manifest))
        # Recheck every input immediately before making the complete directory visible.
        validate(plan, workspace)
        validate_gate(plan, workspace)
        for directory in sorted((p for p in temporary.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            sync_directory(directory)
        sync_directory(temporary)
        os.rename(temporary, dest)
        sync_directory(dest.parent)
    except BaseException:
        # Keep a failed partial copy as explicit evidence; never remove previous work.
        raise
    return {'status': 'copied_verified_package', 'destination': str(dest), 'files': len(entries),
            'bytes': sum(e['bytes'] for e in entries), 'publication_manifest_sha256': sha(dest / 'publication_manifest.json'),
            'plan_sha256': plan_sha}


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--destination', type=Path)
    p.add_argument('--copy', action='store_true')
    args = p.parse_args()
    need(not args.copy or args.destination is not None, '--copy requires explicit destination')
    print(json.dumps(execute(args.plan, args.plan_sha256, args.workspace, args.destination, args.copy), sort_keys=True))


if __name__ == '__main__':
    main()
