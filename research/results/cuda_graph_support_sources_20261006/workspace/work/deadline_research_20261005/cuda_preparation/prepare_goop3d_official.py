#!/usr/bin/env python3
"""Prepare exact official Goop-3D train/valid TFRecords; description only by default.

No acquisition, test access or model admission. Reuses the unchanged reviewed
TFRecord reader/decoder and preserves all source arrays plus failed staging.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = 'official_goop3d_numeric_preparation_v1'
RECEIPT_SCHEMA = 'official_goop3d_train_valid_acquisition_v1'
READER_SHA = 'ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33'
BASE_URL = 'https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop-3D/'
METADATA_SHA = '727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'
SOURCES = {
    'metadata.json': {'bytes': 471, 'generation': '1599153912104523', 'crc32c': 'odsy4Q=='},
    'train.tfrecord': {'bytes': 27448425232, 'generation': '1599159646798459', 'crc32c': 'gBeM1w=='},
    'valid.tfrecord': {'bytes': 2857385340, 'generation': '1599154482348408', 'crc32c': 'rw9odw=='},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_reader(path):
    require(sha(path) == READER_SHA, 'Frozen TFRecord reader/decoder SHA256 differs')
    spec = importlib.util.spec_from_file_location('_goop3d_pinned_tfrecord_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_receipt(receipt, splits):
    require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('status') == 'complete'
            and receipt.get('dataset') == 'Goop-3D' and receipt.get('source_family') == 'official_gns_tfrecord',
            'Completed exact official Goop-3D acquisition receipt required')
    rows = receipt.get('files', [])
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), 'Receipt files must be records')
    names = [row.get('name') for row in rows]
    required = {'metadata.json'} | {split + '.tfrecord' for split in splits}
    require(len(names) == len(set(names)) and required <= set(names) <= set(SOURCES), 'Receipt has missing, duplicate or unreviewed files')
    by_name = {row['name']: row for row in rows}
    for name, row in by_name.items():
        expected = SOURCES[name]
        require(row.get('status') == 'complete' and row.get('url') == BASE_URL + name
                and row.get('saved_name') == name and row.get('received_bytes') == expected['bytes']
                and row.get('generation') == expected['generation'] and row.get('crc32c_base64') == expected['crc32c']
                and row.get('crc32c_verified') is True and digest(row.get('sha256')),
                'Acquired source identity/checksum differs: ' + name)
    require(by_name['metadata.json']['sha256'] == METADATA_SHA, 'Acquired metadata SHA differs')
    return by_name


def verify_source(reader, path, row):
    require(path.is_file() and not path.is_symlink(), 'Source must be a regular non-symlink file')
    initial = path.stat()
    value, crc, count = hashlib.sha256(), 0, 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
            crc = reader.crc32c.crc32c(block, crc)
            count += len(block)
    encoded = base64.b64encode(crc.to_bytes(4, 'big')).decode()
    final = path.stat()
    require((initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns), 'Source changed while hashing')
    require(count == row['received_bytes'] and value.hexdigest() == row['sha256'] and encoded == row['crc32c_base64'],
            'Complete source length/SHA256/CRC32C differs from receipt')
    return initial


def verify_arrays(reader, payload, record, staging):
    """Compare saved element bytes to every original protobuf array byte string."""
    np = reader.np
    example = reader.example_pb2.SequenceExample()
    example.ParseFromString(payload)
    context, sequences = example.context.feature, example.feature_lists.feature_list
    expected_fields = {'positions', 'particle_types'} | ({'step_context'} if 'step_context' in sequences else set())
    require(expected_fields <= set(record), 'Decoder omitted a source array')
    positions = np.load(staging / Path(record['positions']['path']).name, mmap_mode='r', allow_pickle=False)
    types = np.load(staging / Path(record['particle_types']['path']).name, mmap_mode='r', allow_pickle=False)
    require(positions.dtype.str == '<f4' and types.dtype.str == '<i8' and len(positions) >= 7,
            'Expected source float32/int64 and at least seven actual frames')
    require(types.ndim == 1 and len(types) > 0 and positions.shape == (len(positions), len(types), 3),
            'Decoded source array shapes differ')
    require(list(context['key'].int64_list.value) == record['source_key'], 'Source key changed')
    require(types.tobytes(order='C') == reader.one_bytes(context['particle_type'], 'particle_type'), 'Particle type bytes changed')
    source_frames = sequences['position'].feature
    require(len(source_frames) == len(positions), 'Actual frame count changed')
    for index, frame in enumerate(source_frames):
        require(positions[index].tobytes(order='C') == reader.one_bytes(frame, 'position'), 'Position element bytes changed')
    detail = {'frames': len(positions), 'particles': positions.shape[1], 'dimension': 3, 'position_dtype': positions.dtype.str,
              'particle_type_dtype': types.dtype.str, 'particle_type_ids': [int(v) for v in np.unique(types)],
              'all_source_array_bytes_preserved_exact': True, 'auxiliary_fields': []}
    if 'step_context' in sequences:
        aux = np.load(staging / Path(record['step_context']['path']).name, mmap_mode='r', allow_pickle=False)
        require(aux.dtype.str == '<f4' and len(aux) == len(positions), 'Auxiliary dtype/frame count changed')
        for index, frame in enumerate(sequences['step_context'].feature):
            require(aux[index].tobytes(order='C') == reader.one_bytes(frame, 'step_context'), 'Auxiliary element bytes changed')
        detail['auxiliary_fields'].append({'name': 'step_context', 'shape': list(aux.shape),
                                           'all_finite': bool(np.isfinite(aux).all())})
        record['step_context']['use'] = 'Preserved source auxiliary data; semantics unreviewed and model admission required'
        del aux
    del positions, types
    return detail


def convert_one(reader, source, split, row, output, metadata, seen):
    initial = verify_source(reader, source, row)
    staging = output / ('.' + split + '.staging')
    staging.mkdir(mode=0o700)
    records, details, current = [], [], None
    stream = reader.VerifiedTFRecords(source, max_record_bytes=128 * 1024 * 1024)
    try:
        for index, offset, payload in stream:
            current = {'source_index': index, 'source_offset_bytes': offset}
            record = reader.decode_record(payload, index, offset, split, staging, dimension=3, expected_frames=None)
            detail = verify_arrays(reader, payload, record, staging)
            identity = record['trajectory_content_sha256']
            require(identity not in seen, 'Exact position/type trajectory duplicate within/across selected splits')
            seen.add(identity)
            records.append(record)
            details.append({**current, **detail})
            if len(records) % 50 == 0:
                write_json(staging / 'progress.json', {'completed_records': len(records), 'current': current})
        require(records and stream.sha256 == row['sha256'] and stream.size_bytes == row['received_bytes']
                and stream.record_count == len(records), 'Incomplete source EOF/hash/record coverage')
        final = source.stat()
        require((initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns), 'Source changed during conversion')
        particle_counts = sorted(d['particles'] for d in details)
        summary = {'record_count': len(records), 'dimension': 3,
                   'particle_count_min': particle_counts[0], 'particle_count_max': particle_counts[-1],
                   'particle_counts_sorted': particle_counts,
                   'eligible_six_frame_histories': sum(d['frames'] - 6 for d in details),
                   'frame_lengths': sorted({d['frames'] for d in details}),
                   'forecast_horizons_after_six_frames': sorted({d['frames'] - 6 for d in details}),
                   'metadata_sequence_length': metadata['sequence_length'],
                   'official_parser_sequence_length_plus_one': metadata['sequence_length'] + 1,
                   'all_match_official_parser_frame_count': all(d['frames'] == metadata['sequence_length'] + 1 for d in details),
                   'particle_type_ids': sorted({v for d in details for v in d['particle_type_ids']}),
                   'any_auxiliary_fields': any(d['auxiliary_fields'] for d in details),
                   'all_source_array_bytes_preserved_exact': True, 'whole_object_crc32c_verified': True,
                   'TFRecord_length_and_payload_CRC32C_verified': True, 'source_EOF_SHA256_verified': True, 'records': details}
        return records, summary
    except BaseException as error:
        write_json(staging / 'failure.json', {'state': 'failed', 'current': current, 'completed_records': len(records),
                   'error_type': type(error).__name__, 'reason': str(error), 'staging_and_source_retained': True})
        write_json(staging / 'completed_record_details.json', details)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    for name in ('input-dir', 'output-dir', 'acquisition-report', 'reader'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--splits', nargs='+', choices=('train', 'valid'), default=['train', 'valid'])
    args = parser.parse_args(argv)
    require(len(args.splits) == len(set(args.splits)), 'Duplicate splits refused')
    args.splits = [s for s in ('train', 'valid') if s in args.splits]
    if args.execute:
        require(all(getattr(args, name) is not None for name in ('input_dir','output_dir','acquisition_report','reader')),
                'Explicit input/output/acquisition receipt/pinned reader paths required')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema': SCHEMA, 'execution': False, 'dataset': 'Goop-3D', 'splits': ['train','valid'],
                          'test_access': False, 'acquisition': False, 'scientific_training_admission': False}))
        return 0
    output, source = args.output_dir.resolve(), args.input_dir.resolve()
    require(output != source and output not in source.parents and source not in output.parents, 'Fresh output must be separate from acquisition tree')
    output.mkdir(mode=0o700, exist_ok=False)
    report = {'schema': SCHEMA, 'status': 'admitting', 'dataset': 'Goop-3D', 'source_family': 'official_gns_tfrecord',
              'test_accessed': False, 'scientific_training_admission': False, 'splits': {}}
    report_path = output / 'structural_report.json'
    write_json(report_path, report)
    started = time.perf_counter()
    try:
        receipt_bytes = args.acquisition_report.read_bytes()
        receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
        rows = check_receipt(json.loads(receipt_bytes), args.splits)
        metadata_bytes = (source / 'metadata.json').read_bytes()
        require(hashlib.sha256(metadata_bytes).hexdigest() == METADATA_SHA, 'Exact metadata bytes differ')
        metadata = json.loads(metadata_bytes)
        require(metadata['dim'] == 3 and metadata['sequence_length'] == 300 and metadata['default_connectivity_radius'] == .025,
                'Only the verified 3D Goop-3D metadata is supported')
        reader = load_reader(args.reader)
        verify_source(reader, source / 'metadata.json', rows['metadata.json'])
        wrapper_sha = sha(__file__)
        report.update(wrapper_sha256=wrapper_sha, reader_sha256=READER_SHA, metadata_sha256=METADATA_SHA,
                      acquisition_report_sha256=receipt_hash, numpy_version=reader.np.__version__)
        manifests, seen = {}, set()
        for split in args.splits:
            records, summary = convert_one(reader, source / (split + '.tfrecord'), split, rows[split + '.tfrecord'], output, metadata, seen)
            report['splits'][split] = summary
            row = rows[split + '.tfrecord']
            manifests[split] = {'format': reader.FORMAT, 'version': 1, 'dataset': 'Goop-3D', 'split': split,
                'metadata': metadata, 'metadata_sha256': METADATA_SHA, 'record_count': len(records), 'records': records,
                'source': {'family': 'official_gns_tfrecord', 'dataset': 'Goop-3D', 'file': split + '.tfrecord',
                    'size_bytes': row['received_bytes'], 'sha256': row['sha256'], 'generation': row['generation'],
                    'crc32c_base64': row['crc32c_base64'], 'CRC_verified': True, 'record_count': len(records),
                    'acquisition_report_sha256': receipt_hash}, 'converter_sha256': wrapper_sha, 'reader_sha256': READER_SHA}
            write_json(report_path, report)
        require(sha(args.acquisition_report) == receipt_hash and sha(source / 'metadata.json') == METADATA_SHA
                and sha(args.reader) == READER_SHA and sha(__file__) == wrapper_sha, 'Receipt/metadata/source changed during conversion')
        (output / 'metadata.json').write_bytes(metadata_bytes)
        for split, manifest in manifests.items():
            (output / ('.' + split + '.staging')).replace(output / split)
            write_json(output / (split + '.json'), manifest)
            report['splits'][split]['manifest_sha256'] = sha(output / (split + '.json'))
        report.update(status='complete_structural_only', elapsed_seconds=time.perf_counter()-started,
                      duplicate_definition='exact stored position/type dtype,shape,bytes; auxiliaries are preserved but excluded from duplicate identity')
        write_json(report_path, report)
        print(json.dumps({'status':report['status'], 'scientific_training_admission':False}))
        return 0
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, reason=str(error),
                      all_existing_outputs_retained=True, elapsed_seconds=time.perf_counter()-started)
        write_json(report_path, report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
