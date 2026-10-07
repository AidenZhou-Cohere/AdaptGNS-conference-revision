#!/usr/bin/env python3
"""Numerical particle-simulation helpers. Use code/evaluate.py for evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'adaptgns_goop3d_auxiliary_census_v1'
METADATA_SHA = '727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'
CONVERTER_SHA = '49dece28bf4494591379b8a667e5366b5bdf1609c4dc0757ceb31664c28f9b74'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def census_record(root, split, record, np):
    root = Path(root).resolve()
    row = {'id': record['id'], 'source_index': record['source_index'], 'present': 'step_context' in record}
    shape = record['positions']['shape']
    require(len(shape) == 3 and shape[0] == 301 and shape[1] > 0 and shape[2] == 3,
            'Expected structurally verified T301/D3 position descriptor')
    if not row['present']:
        return row, None
    desc = record['step_context']
    require(desc['path'] == f'{split}/step_context_{record["source_index"]:06d}.npy', 'Unexpected auxiliary path')
    raw_path = root / desc['path']
    path = raw_path.resolve()
    require(not raw_path.is_symlink() and path.is_relative_to(root) and path.is_file(), 'Unsafe auxiliary path')
    require(path.stat().st_size == desc['size_bytes'] and sha(path) == desc['sha256'], 'Auxiliary bytes differ')
    values = np.load(path, allow_pickle=False)
    require(list(values.shape) == desc['shape'] and values.ndim >= 2 and values.shape[0] == shape[0]
            and all(v > 0 for v in values.shape[1:]) and values.dtype.str == desc['dtype'] == '<f4',
            'Auxiliary representation differs')
    bits, counts = np.unique(values.view('<u4'), return_counts=True)
    row.update(sha256=desc['sha256'],
               descriptor_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
               shape=list(values.shape), dtype=values.dtype.str, elements=int(values.size),
               finite_count=int(np.isfinite(values).sum()), nan_count=int(np.isnan(values).sum()),
               positive_infinity_count=int(np.isposinf(values).sum()), negative_infinity_count=int(np.isneginf(values).sum()),
               unique_float32_bits_hex=[f'{int(value):08x}' for value in bits],
               float32_bit_counts=[{'bits_hex': f'{int(value):08x}', 'count': int(count)} for value, count in zip(bits, counts)])
    return row, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    for name in ('numeric-root', 'structural-report', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--structural-sha256')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema': SCHEMA, 'execution': False, 'splits': ['train', 'valid'],
                          'scientific_training_admitted': False, 'test_accessed': False}))
        return 0
    require(all(getattr(args, name) is not None for name in ('numeric_root', 'structural_report', 'structural_sha256', 'output')),
            'Explicit numeric root, completed structural report/hash and fresh output required')
    require(not args.output.exists(), 'Existing census output must be preserved')
    root = args.numeric_root.resolve()
    report = {'schema': SCHEMA, 'status': 'running', 'scientific_training_admitted': False, 'test_accessed': False,
              'source_sha256': sha(__file__), 'structural_report_sha256': args.structural_sha256,
              'metadata_sha256': METADATA_SHA, 'splits': {}}
    write(args.output, report)
    try:
        import numpy as np
        require(sha(args.structural_report) == args.structural_sha256, 'Structural report changed')
        structural = json.loads(args.structural_report.read_text())
        require(structural['schema'] == 'official_goop3d_numeric_preparation_v1'
                and structural['status'] == 'complete_structural_only' and structural['dataset'] == 'Goop-3D'
                and structural['wrapper_sha256'] == CONVERTER_SHA and structural['metadata_sha256'] == METADATA_SHA
                and set(structural['splits']) == {'train', 'valid'}, 'Complete exact train/valid structural report required')
        require(sha(root / 'metadata.json') == METADATA_SHA, 'Metadata bytes differ')
        metadata = json.loads((root / 'metadata.json').read_text())
        require('context_mean' not in metadata and 'context_std' not in metadata, 'Context normalization exists')
        pins = {str(args.structural_report): args.structural_sha256, str(root / 'metadata.json'): METADATA_SHA}
        for split in ('train', 'valid'):
            path = root / (split + '.json')
            digest = sha(path)
            require(digest == structural['splits'][split]['manifest_sha256'], 'Published manifest differs')
            manifest = json.loads(path.read_text())
            require(manifest['dataset'] == 'Goop-3D' and manifest['split'] == split
                    and manifest['metadata'] == metadata and manifest['metadata_sha256'] == METADATA_SHA
                    and manifest['converter_sha256'] == CONVERTER_SHA
                    and len(manifest['records']) == manifest['record_count'] == structural['splits'][split]['record_count'],
                    'Manifest lineage or complete record count differs')
            pins[str(path)] = digest
            split_report = {'manifest_sha256': digest, 'record_count': len(manifest['records']),
                            'context_mean_absent': True, 'context_std_absent': True, 'records': []}
            report['splits'][split] = split_report
            for index, record in enumerate(manifest['records']):
                require(record['source_index'] == index and record['id'] == f'{split}:{index:06d}', 'Source order differs')
                row, array_path = census_record(root, split, record, np)
                split_report['records'].append(row)
                if array_path is not None:
                    pins[str(array_path)] = row['sha256']
            split_report.update(auxiliary_present_count=sum(row['present'] for row in split_report['records']),
                                auxiliary_absent_count=sum(not row['present'] for row in split_report['records']),
                                all_context_descriptors_and_bytes_verified=True)
            write(args.output, report)
        require(all(sha(path) == digest for path, digest in pins.items()) and sha(__file__) == report['source_sha256'],
                'Census input or source changed during audit')
        report.update(status='all_preserved_auxiliary_bytes_verified',
                      semantics='No semantic inference from NaN, constancy or missing arrays; omission requires separate official-parser/metadata review.')
        write(args.output, report)
        print(json.dumps({'status': report['status'], 'report_sha256': sha(args.output)}))
        return 0
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), partial_census_retained=True)
        write(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit("Use code/evaluate.py or code/portable/ entry points.")
