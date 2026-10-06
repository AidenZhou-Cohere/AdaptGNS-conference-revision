"""Package complete Goop scalar results only after both supported array audits.

No model execution or remote access. Exact allowlisted files only; large source
collections and raw arrays stay at their original locations. Outputs are fresh.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / 'goop_paired_scalar_summary_1153_v1.json'
SUMMARY_SHA = 'cec653b7a0bf657c2c21402f14fd50a2dda934cf1affcd2ab4c603dbced8bbb0'
MAIN_SHA = '45254f29966b91b8d5ced16133d84d18d983b89f34f00aa39fa8792384e6ce1c'
HELPER_SHA = '615ce93877e8612f8e70a557d87d324b05e15b2b1fe659bc8c63034011007110'
COLLECTIONS = {
    'A': ('goop_scalar_collection_A_1141_v1.json', '41ce909f2caab840857d42a4ff4a19557bd855abd3dd3c06c9035ae6a7337072', 1362002800),
    'B': ('goop_scalar_collection_B_1148_v1.json', 'bbdd333a101107f2a41b441684bff799b85ef4ea20e2179a5f0a4c8b171ca3fa', 680927748),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-a', 'audit-b', 'paired-audit', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('output must be a fresh directory')
    payload = {}
    origins = {}

    def add(path, name, compressed=False):
        require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(), 'Missing or linked source: ' + str(path))
        raw = path.read_bytes()
        payload[name] = gzip.compress(raw, mtime=0) if compressed else raw
        origins[name] = {'source': str(path), 'uncompressed_sha256': sha(raw), 'uncompressed_bytes': len(raw)}
        return raw

    raw = add(SUMMARY, 'paired_scalar_summary.json.gz', True)
    require(sha(raw) == SUMMARY_SHA, 'frozen scalar summary changed')
    summary = json.loads(raw)
    audits = {}
    audit_hashes = {}
    for role, path in (('A', a.audit_a), ('B', a.audit_b)):
        raw = add(path, 'array_audit_' + role + '.json.gz', True)
        audit_hashes[role] = sha(raw)
        audit = audits[role] = json.loads(raw)
        require((audit['schema'], audit['status'], audit['host_role'], audit['audit_revision']) == (
            'goop_saved_array_audit_v1', 'passed_supported_checks', role, 3), 'Passed role-specific revision3 audit required')
        require(audit['auditor_sha256'] == MAIN_SHA and audit['diagnostic_helper_sha256'] == HELPER_SHA, 'Auditor source mismatch')
        require(audit['collection_sha256'] == COLLECTIONS[role][1] == summary['collection_sha256'][role], 'Collection mismatch')
        require(audit['cohort_sha256'] == summary['cohort_sha256'], 'Cohort mismatch')
    paired = json.loads(add(a.paired_audit, 'paired_array_audit.json.gz', True))
    require((paired['schema'], paired['status'], paired['audit_revision']) == (
        'goop_paired_saved_array_audit_v1', 'passed_supported_checks', 3), 'Passed paired revision3 audit required')
    require(paired['summary_sha256'] == SUMMARY_SHA and paired['cohort_sha256'] == summary['cohort_sha256'], 'Paired summary/cohort mismatch')
    bindings = paired['verified_input_sha256']
    for role, path in (('A', a.audit_a), ('B', a.audit_b)):
        require(path.is_absolute() and bindings.get(str(path)) == audit_hashes[role], 'Exact paired receipt path/hash mismatch')
    require(bindings.get(str(SUMMARY)) == SUMMARY_SHA, 'Exact paired summary binding mismatch')
    for name, expected in {'audit_goop_saved_arrays_v1.py': MAIN_SHA,
            'goop_saved_diagnostic_audit_v1.py': HELPER_SHA,
            'audit_goop_paired_arrays_v1.py': 'c74df670fec03a534269311077148e64db10ad1da23fd0baabbb9e4ef9b7c10c'}.items():
        require(bindings.get(str(HERE / 'goop_saved_array_audit_review_v3' / name)) == expected, 'Paired source binding mismatch')
    for revision in (1, 2, 3):
        directory = HERE / ('goop_saved_array_audit_review_v' + str(revision))
        pins = json.loads((directory / 'source_sha256.json').read_text())
        for name, expected in pins.items():
            require(Path(name).name == name and sha((directory / name).read_bytes()) == expected, 'snapshot source changed')
        expected_sources = {'audit_goop_saved_arrays_v1.py', 'audit_goop_paired_arrays_v1.py',
            'goop_saved_diagnostic_audit_v1.py', 'test_audit_goop_saved_arrays_v1.py', 'test_goop_saved_diagnostic_audit_v1.py'}
        require(set(pins) == expected_sources, 'Unexpected snapshot source list')
        allowed = expected_sources | {'source_sha256.json', 'README.md', 'development_failure_history.json', 'synthetic_verification.json'}
        if revision == 3:
            allowed |= {'seal_sha256.json', 'actual_v2_failure_history.json',
                'preserved_v2_failure/goop_array_audit_A_v2_1156_v1.log',
                'preserved_v2_failure/goop_saved_array_audit_v2_target_namespace_diagnosis_code_audit_v1.json',
                'preserved_v2_failure/goop_target_pair_failure_evidence_1158_v1.json'}
        for name in sorted(allowed):
            add(directory / name, 'auditor_history/v' + str(revision) + '/' + name)
        name = 'goop_saved_array_audit_independent_review_code_audit_v' + str(revision) + '.json'
        add(HERE / name, 'reviews/' + name)
    omitted = {'collections': {role: {'workspace_path': name, 'sha256': pin, 'bytes': size}
               for role, (name, pin, size) in COLLECTIONS.items()},
               'raw_arrays': 'Remain in original stopped A/B queues. Exact read-array/file hashes are retained in the array audit receipts.',
               'models_and_dataset': 'Unchanged original checkpoints and official data are referenced by the complete scalar summary; they are not duplicated here.'}
    payload['omissions.json'] = encode(omitted)
    payload['copy_provenance.json'] = encode(origins)
    payload['README.md'] = (
        '# Paired Goop 2D graph-exposure study\n\n'
        'Six fresh models compare base-only and mixed-graph training at 100,000 updates, '
        'with three paired seeds and six evaluation policies. The compact scalar summary '
        'retains all 1,080 planned rollout outcomes: 1,077 complete and three recorded '
        'candidate-pair guards. Failed full-horizon groups remain undefined. All original '
        'same-state and clean-validation stages are retained.\n\n'
        'The two array receipts and paired companion independently check saved forecast '
        'positions, available diagnostic arrays, recorded scalar-series aggregates and '
        'paired seed summaries. They do not replay unsaved full trajectories, reload '
        'official source truth or checkpoint normalization, rerun models, or establish '
        'isolated hardware timing. Read each receipt for its exact supported scope. '
        'Shared-host wall times must not be interpreted as a causal speedup.\n\n'
        'Auditor versions and their unsuccessful checks are retained in `auditor_history/`; '
        'revision 3 is the approved numerical-identity repair for float32/float64 target '
        'storage. Frozen scientific trainers, evaluators and scalar aggregators were not '
        'changed. Their preparation and dependency layout are in the neighboring '
        '`scoped_execution_completion_preparation_20261006` package.\n\n'
        'JSON compression is lossless. `manifest.json` binds packaged bytes; '
        '`copy_provenance.json` binds their original uncompressed bytes. Large scalar '
        'collections, raw arrays, datasets and checkpoints are referenced in '
        '`omissions.json`. Credentials, SSH configuration and native process inventories '
        'are not packaged.\n'
    ).encode()
    payload['manifest.json'] = encode({name: {'sha256': sha(raw), 'bytes': len(raw)} for name, raw in sorted(payload.items())})
    a.output.mkdir(parents=True)
    for name, raw in payload.items():
        path = a.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    print(json.dumps({'files': len(payload), 'bytes': sum(map(len, payload.values())), 'output': str(a.output)}))


if __name__ == '__main__':
    main()
