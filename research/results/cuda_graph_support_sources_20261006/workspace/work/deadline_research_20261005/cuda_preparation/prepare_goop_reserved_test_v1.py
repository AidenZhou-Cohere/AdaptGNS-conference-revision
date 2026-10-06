#!/usr/bin/env python3
"""Separate root-gated reserved Goop test preparation; description by default.

No test request or file read before complete six100k cohort/audit release.
inspect obtains headers only; acquire requires those exact root-pinned headers.
convert reuses the unchanged reviewed TFRecord/array-preservation helper.
census verifies all-split numeric bytes/duplicates and test auxiliaries; emits
an admission candidate requiring root review. preflight uses the frozen final
split contract. No training, checkpoint selection, model import or inference.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = 'https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop/'
URL = BASE + 'test.tfrecord'
READER_SHA = 'ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33'
CONVERTER_SHA = 'fa4b7d883d7c360500fc3c603c9cd14538c916f1ddd7435aedfc8ec2420f2985'
CONTRACT_SHA = '4ee1e33dbe666804d1827b88565620a435d04836a170671914a71acb818be090'
METADATA_SHA = '565d6e13be91be6a6b0fbc31a9aa19ed70a5505228411d0928288fcf874ca3dd'
PROTOCOL_SHA = 'c8690d0da209c66557e3dbb0cbccd55d660b4cc270683913f74d381516591851'
TRAIN_ADMISSION_SHA = 'c2a12ef0c55b47f4c9493027450f8b51f52dbbd6043915bb09c1648b6cb9edeb'
TRAIN_MANIFEST_SHA = '5ef43daf9bac961a69bd460a539891624b79c8a13f80ca851e85802197982256'
VALID_MANIFEST_SHA = '3227e03c4c7fcee9f99c4104010774cabc74c34bfda667d76c5894590dcfc415'
CONTEXT_SHA = 'c81ae2f1565e61542bcc406c4a9d71b67135620857ad62292eaef0e29b407de9'
ACQ_SCHEMA = 'official_goop_reserved_test_acquisition_v1'
GATE_SCHEMA = 'adaptgns_goop_reserved_test_preparation_release_v1'
HEAD_SCHEMA = 'official_goop_reserved_test_source_metadata_v1'
POLICIES = ['base', 'dense', 'random25', 'speed25', 'laggedrisk25', 'relative-velocity-RMS25']
CUTOFF = datetime(2026, 10, 7, 1, tzinfo=timezone.utc)


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(v in '0123456789abcdef' for v in value)


def now(): return datetime.now(timezone.utc).isoformat()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def load(path, expected, name):
    require(sha(path) == expected, 'Frozen source mismatch: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


def stamped(value):
    t = datetime.fromisoformat(value); require(t.tzinfo is not None, 'Timezone-aware chronology required'); return t


def gate(args):
    # These three non-test authorization files are the only reads before admission.
    cohort, audit, release = [read(getattr(args, key)) for key in ('cohort', 'cohort_audit', 'root_release')]
    required = {(a, s) for a in ('base', 'mix') for s in range(3)}
    require(cohort.get('schema') == 'adaptgns_goop_graph_support_final_cohort_v1' and cohort.get('status') == 'frozen_for_final_evaluation'
            and cohort.get('issued_by') == 'root' and cohort.get('dataset') == 'Goop' and cohort.get('updates') == 100000
            and cohort.get('protocol_sha256') == PROTOCOL_SHA and cohort.get('training_admission_sha256') == TRAIN_ADMISSION_SHA
            and cohort.get('cohort_audit_sha256') == sha(args.cohort_audit) and cohort.get('policies') == POLICIES,
            'Complete frozen six100k Goop cohort required before test access')
    models = cohort.get('models', [])
    require(len(models) == 6 and {(r.get('arm'), r.get('seed')) for r in models} == required
            and all(r.get('objective') == 'faithful' and r.get('completed_steps') == 100000 and digest(r.get('checkpoint_sha256')) for r in models)
            and len({r['checkpoint_sha256'] for r in models}) == 6, 'All six distinct100k checkpoint identities required')
    require(audit.get('schema') == 'adaptgns_goop_graph_support_complete_cohort_audit_v1'
            and audit.get('status') == 'all_six_endpoints_and_pairing_verified' and audit.get('issued_by') == 'root'
            and audit.get('training_schema') == 'adaptgns_goop_graph_support_cuda_training_v1'
            and audit.get('protocol_sha256') == PROTOCOL_SHA and audit.get('training_admission_sha256') == TRAIN_ADMISSION_SHA,
            'Root complete endpoint/pairing audit required')
    checked = audit.get('models', [])
    require(len(checked) == 6 and {(r.get('arm'), r.get('seed')) for r in checked} == required, 'Incomplete endpoint audit')
    for row in models:
        ev = next(r for r in checked if (r['arm'], r['seed']) == (row['arm'], row['seed']))
        require(ev.get('checkpoint_sha256') == row['checkpoint_sha256'] and ev.get('completed_steps') == ev.get('graph_history_updates') == 100000
                and ev.get('checkpoint_every') == 10000 and ev.get('log_every') == 100
                and all(ev.get(k) is True for k in ('all_optimizer_steps_equal_100000', 'all_state_and_moments_finite',
                    'source_data_protocol_verified', 'checkpoint_bytes_verified')), 'Incomplete endpoint verification')
    pairs = audit.get('paired_seeds', [])
    require(len(pairs) == 3 and {p.get('seed') for p in pairs} == {0, 1, 2}
            and all(all(p.get(k) is True for k in ('initial_model_tensor_identity', 'initial_cpu_cuda_rng_identity',
                'all_frame_noise_lr_schedules_equal', 'all_graph_budgets_and_rng_material_verified')) for p in pairs), 'Incomplete paired schedules/initialization')
    status = 'approved_for_source_metadata' if args.mode == 'inspect' else 'approved_for_acquisition_preparation'
    require(release.get('schema') == GATE_SCHEMA and release.get('status') == status and release.get('issued_by') == 'root'
            and release.get('cohort_sha256') == sha(args.cohort) and release.get('cohort_audit_sha256') == sha(args.cohort_audit)
            and release.get('preparation_source_sha256') == sha(__file__) and release.get('source_url') == URL
            and release.get('all_six_checkpoint_hashes_verified') is True
            and release.get('test_converter_independently_reviewed') is True
            and isinstance(release.get('review_rationale'), str) and bool(release['review_rationale'].strip()), 'Exact root reserved-test release required')
    require(stamped(cohort['created_utc']) <= stamped(release['issued_utc']) <= datetime.now(timezone.utc) < CUTOFF,
            'Test access must follow cohort freeze and precede compute cutoff')
    return cohort, release


def source_metadata(args, release):
    require(args.source_metadata and sha(args.source_metadata) == release.get('source_metadata_sha256'), 'Root-pinned source header receipt required')
    value = read(args.source_metadata)
    require(value.get('schema') == HEAD_SCHEMA and value.get('status') == 'headers_verified' and value.get('url') == URL
            and value.get('cohort_sha256') == sha(args.cohort) and value.get('response_status') == 200
            and type(value.get('size_bytes')) is int and value['size_bytes'] > 0
            and isinstance(value.get('generation'), str) and value['generation'].isdigit()
            and len(base64.b64decode(value.get('crc32c_base64', ''), validate=True)) == 4,
            'Complete exact source generation/length/CRC header receipt required')
    return value


def header_values(response):
    require(response.status_code == 200, 'Expected publisher200 without redirect')
    headers = {k.lower(): v for k, v in response.headers.items()}
    require(headers.get('content-encoding', 'identity') == 'identity', 'Source must use identity transfer encoding')
    hashes = {v.strip().split('=', 1)[0]: v.strip().split('=', 1)[1] for v in headers.get('x-goog-hash', '').split(',') if '=' in v}
    size, generation, crc = int(headers.get('content-length', '-1')), headers.get('x-goog-generation'), hashes.get('crc32c', '')
    require(size > 0 and isinstance(generation, str) and generation.isdigit() and len(base64.b64decode(crc, validate=True)) == 4,
            'Publisher length/generation/CRC metadata missing')
    return size, generation, crc


def inspect_source(args, release, requests):
    require(args.output and not args.output.exists(), 'Fresh metadata receipt output required')
    report = {'schema': HEAD_SCHEMA, 'status': 'started', 'url': URL, 'cohort_sha256': sha(args.cohort),
              'root_release_sha256': sha(args.root_release), 'preparation_source_sha256': sha(__file__), 'started_utc': now(), 'payload_downloaded': False}
    write(args.output, report)
    try:
        with requests.head(URL, headers={'Accept-Encoding': 'identity'}, timeout=(20, 30), allow_redirects=False) as response:
            report.update(response_status=response.status_code, response_headers=dict(response.headers))
            write(args.output, report)
            size, generation, crc = header_values(response)
        report.update(status='headers_verified', size_bytes=size, generation=generation, crc32c_base64=crc, ended_utc=now())
        write(args.output, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), ended_utc=now()); write(args.output, report); raise


def acquire(args, release, metadata, requests, crc32c):
    require(args.output_dir and args.metadata and sha(args.metadata) == METADATA_SHA, 'Fresh acquisition root and original metadata bytes required')
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'acquisition_report.json'
    report = {'schema': ACQ_SCHEMA, 'status': 'running', 'dataset': 'Goop', 'source_family': 'official_gns_tfrecord', 'test_accessed': True,
              'cohort_sha256': sha(args.cohort), 'cohort_freeze_utc': read(args.cohort)['created_utc'], 'root_release_sha256': sha(args.root_release),
              'source_metadata_sha256': sha(args.source_metadata), 'source_sha256': sha(__file__), 'started_utc': now(), 'files': []}
    write(report_path, report); partial = output / 'test.tfrecord.partial'; current = None
    started = time.perf_counter()
    try:
        require(crc32c.crc32c(b'123456789') == 0xe3069283, 'CRC32C implementation self-test failed')
        raw = args.metadata.read_bytes(); require(hashlib.sha256(raw).hexdigest() == METADATA_SHA, 'Metadata changed')
        (output / 'metadata.json').write_bytes(raw)
        report['files'].append({'name': 'metadata.json', 'saved_name': 'metadata.json', 'url': BASE + 'metadata.json',
            'status': 'complete', 'received_bytes': len(raw), 'generation': '1599153763201175', 'sha256': METADATA_SHA,
            'crc32c_base64': base64.b64encode(crc32c.crc32c(raw).to_bytes(4, 'big')).decode(), 'crc32c_verified': True,
            'retrieval': 'local_hash_verified_copy_of_frozen_official_metadata'})
        current = {'name': 'test.tfrecord', 'saved_name': 'test.tfrecord', 'url': URL, 'status': 'started', 'received_bytes': 0,
                   'requested_generation': metadata['generation']}; report['files'].append(current); write(report_path, report)
        with requests.get(URL, params={'generation': metadata['generation']}, headers={'Accept-Encoding': 'identity'},
                          stream=True, timeout=(20, 120), allow_redirects=False) as response:
            current.update(response_status=response.status_code, response_headers=dict(response.headers)); write(report_path, report)
            require(header_values(response) == (metadata['size_bytes'], metadata['generation'], metadata['crc32c_base64']), 'GET differs from root-pinned publisher headers')
            h, crc, count = hashlib.sha256(), 0, 0
            with partial.open('xb') as f:
                for block in response.iter_content(chunk_size=1 << 20):
                    if not block: continue
                    f.write(block); h.update(block); crc = crc32c.crc32c(block, crc); count += len(block); current['received_bytes'] = count
                    require(count <= metadata['size_bytes'], 'Received more than pinned source length')
                f.flush(); os.fsync(f.fileno())
            encoded = base64.b64encode(crc.to_bytes(4, 'big')).decode()
            current.update(sha256=h.hexdigest(), generation=metadata['generation'], crc32c_base64=encoded,
                           crc32c_verified=encoded == metadata['crc32c_base64'])
            require(count == metadata['size_bytes'] and encoded == metadata['crc32c_base64'], 'Complete source checksum/length differs')
        partial.rename(output / 'test.tfrecord'); current['status'] = 'complete'
        require(sha(__file__) == report['source_sha256'], 'Preparation source changed')
        report.update(status='complete', ended_utc=now(), elapsed_seconds=time.perf_counter() - started); write(report_path, report)
    except BaseException as error:
        if partial.exists():
            current.update(retained_partial_sha256=sha(partial), retained_partial_bytes=partial.stat().st_size)
        if current is not None: current['status'] = 'failed'
        report.update(status='failed', error_type=type(error).__name__, error=str(error), ended_utc=now(), all_existing_outputs_retained=True)
        write(report_path, report); raise


def receipt_check(receipt, metadata, cohort_sha):
    require(receipt.get('schema') == ACQ_SCHEMA and receipt.get('status') == 'complete' and receipt.get('dataset') == 'Goop'
            and receipt.get('source_family') == 'official_gns_tfrecord' and receipt.get('cohort_sha256') == cohort_sha
            and receipt.get('test_accessed') is True, 'Complete post-cohort test acquisition receipt required')
    rows = receipt.get('files', [])
    require(len(rows) == 2 and {r.get('name') for r in rows} == {'metadata.json', 'test.tfrecord'}, 'Exactly metadata andtest acquired inputs required')
    by_name = {r['name']: r for r in rows}
    for name, row in by_name.items():
        require(row.get('status') == 'complete' and row.get('saved_name') == name and row.get('url') == BASE + name
                and row.get('crc32c_verified') is True and digest(row.get('sha256')), 'Incomplete source identity/checksum')
    source = by_name['test.tfrecord']
    require((source['received_bytes'], source['generation'], source['crc32c_base64']) ==
            (metadata['size_bytes'], metadata['generation'], metadata['crc32c_base64']) and by_name['metadata.json']['sha256'] == METADATA_SHA,
            'Acquired source differs from frozen publisher metadata')
    return by_name


def convert(args, release, metadata):
    require(all((args.input_dir, args.output_dir, args.acquisition_report, args.reader)), 'Explicit acquired source/output/receipt/reader required')
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    require(output != source and output not in source.parents and source not in output.parents, 'Separate fresh numeric output required')
    output.mkdir(parents=True, exist_ok=False)
    helper = load(HERE / 'prepare_goop_official.py', CONVERTER_SHA, '_reserved_test_converter')
    report = {'schema': 'official_goop_numeric_preparation_v1', 'status': 'admitting', 'dataset': 'Goop',
              'source_family': 'official_gns_tfrecord', 'test_accessed': True, 'scientific_training_admission': False,
              'cohort_sha256': sha(args.cohort), 'root_release_sha256': sha(args.root_release), 'splits': {}}
    report_path = output / 'structural_report.json'; write(report_path, report)
    try:
        receipt_hash = sha(args.acquisition_report); receipt = read(args.acquisition_report)
        require(receipt.get('source_metadata_sha256') == sha(args.source_metadata), 'Acquisition source metadata binding differs')
        rows = receipt_check(receipt, metadata, sha(args.cohort))
        metadata_bytes = (source / 'metadata.json').read_bytes(); require(hashlib.sha256(metadata_bytes).hexdigest() == METADATA_SHA, 'Metadata bytes differ')
        numeric_metadata = json.loads(metadata_bytes)
        reader = helper.load_reader(args.reader); helper.verify_source(reader, source / 'metadata.json', rows['metadata.json'])
        wrapper_sha = sha(__file__)
        report.update(wrapper_sha256=wrapper_sha, converter_helper_sha256=CONVERTER_SHA, reader_sha256=READER_SHA,
                      metadata_sha256=METADATA_SHA, acquisition_report_sha256=receipt_hash, numpy_version=reader.np.__version__)
        records, summary = helper.convert_one(reader, source / 'test.tfrecord', 'test', rows['test.tfrecord'], output, numeric_metadata, set())
        report['splits']['test'] = summary; write(report_path, report)
        # Preserve any discrepant source fully. No subset or representation repair.
        require(summary['record_count'] == 30 and summary['frame_lengths'] == [401]
                and summary['forecast_horizons_after_six_frames'] == [395] and summary['particle_type_ids'] == [7]
                and all(r['source_key'] == [i] and r.get('step_context', {}).get('shape') == [401, 1] for i, r in enumerate(records)),
                'Observed test structure differs from frozen final-evaluation contract; preserve and review')
        r = rows['test.tfrecord']
        manifest = {'format': reader.FORMAT, 'version': 1, 'dataset': 'Goop', 'split': 'test', 'metadata': numeric_metadata,
            'metadata_sha256': METADATA_SHA, 'record_count': len(records), 'records': records,
            'source': {'family': 'official_gns_tfrecord', 'dataset': 'Goop', 'file': 'test.tfrecord', 'size_bytes': r['received_bytes'],
                'sha256': r['sha256'], 'generation': r['generation'], 'crc32c_base64': r['crc32c_base64'], 'CRC_verified': True,
                'record_count': len(records), 'acquisition_report_sha256': receipt_hash}, 'converter_sha256': wrapper_sha, 'reader_sha256': READER_SHA}
        require(sha(args.acquisition_report) == receipt_hash and sha(args.reader) == READER_SHA and sha(__file__) == wrapper_sha
                and sha(HERE / 'prepare_goop_official.py') == CONVERTER_SHA, 'Bound source/receipt changed during conversion')
        (output / 'metadata.json').write_bytes(metadata_bytes); (output / '.test.staging').replace(output / 'test')
        write(output / 'test.json', manifest); summary['manifest_sha256'] = sha(output / 'test.json')
        report.update(status='complete_structural_only', completed_utc=now()); write(report_path, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, reason=str(error), all_existing_outputs_retained=True)
        write(report_path, report); raise


def safe_array(root, descriptor):
    root = Path(root).resolve(); raw = root / descriptor['path']; path = raw.resolve()
    require(not raw.is_symlink() and path.is_relative_to(root) and path.is_file(), 'Unsafe numeric path')
    require(path.stat().st_size == descriptor['size_bytes'] and sha(path) == descriptor['sha256'], 'Numeric array bytes differ')
    return path


def numeric_identity(root, record, np):
    h = hashlib.sha256()
    for key in ('positions', 'particle_types'):
        d = record[key]; p = safe_array(root, d); a = np.load(p, mmap_mode='r', allow_pickle=False)
        require(a.dtype.str == d['dtype'] and list(a.shape) == d['shape'] and a.flags.c_contiguous, 'Numeric shape/dtype/layout differs')
        header = json.dumps({'field': key, 'dtype': a.dtype.str, 'shape': list(a.shape)}, sort_keys=True).encode()
        h.update(len(header).to_bytes(8, 'big')); h.update(header)
        raw = memoryview(a).cast('B')
        for start in range(0, len(raw), 1 << 20): h.update(raw[start:start + (1 << 20)])
        del raw, a
    return h.hexdigest()


def auxiliary_census(root, manifest, np):
    rows = []
    for index, record in enumerate(manifest['records']):
        d = record['step_context']; p = safe_array(root, d); values = np.load(p, allow_pickle=False)
        require(record['id'] == f'test:{index:06d}' and record['source_index'] == index and d['path'] == f'test/step_context_{index:06d}.npy'
                and list(values.shape) == d['shape'] == [401, 1] and values.dtype.str == d['dtype'] == '<f4', 'Auxiliary representation/source identity differs')
        bits, counts = np.unique(values.view('<u4'), return_counts=True)
        row = {'id': record['id'], 'source_index': index, 'sha256': d['sha256'], 'shape': list(values.shape), 'dtype': values.dtype.str,
               'elements': int(values.size), 'finite_count': int(np.isfinite(values).sum()), 'nan_count': int(np.isnan(values).sum()),
               'positive_infinity_count': int(np.isposinf(values).sum()), 'negative_infinity_count': int(np.isneginf(values).sum()),
               'unique_float32_bits_hex': [f'{int(v):08x}' for v in bits],
               'float32_bit_counts': [{'bits_hex': f'{int(v):08x}', 'count': int(c)} for v, c in zip(bits, counts)]}
        rows.append(row)
    return {'schema': 'adaptgns_goop_auxiliary_census_v1', 'status': 'all_preserved_auxiliary_bytes_verified', 'test_accessed': True,
            'source_sha256': sha(__file__), 'metadata_sha256': METADATA_SHA, 'splits': {'test': {'record_count': len(rows),
                'context_mean_absent': 'context_mean' not in manifest['metadata'], 'context_std_absent': 'context_std' not in manifest['metadata'],
                'manifest_sha256': sha(Path(root) / 'test.json'), 'all_context_descriptors_and_bytes_verified': True, 'records': rows}},
            'semantics': 'Omission follows exact official parser without context_mean; no inference from auxiliary values or constancy'}


def census(args, release, metadata):
    require(all((args.numeric_root, args.train_manifest, args.valid_manifest, args.acquisition_report, args.output_dir)), 'Numeric root/all three manifests/acquisition/fresh output required')
    require(sha(args.train_manifest) == TRAIN_MANIFEST_SHA and sha(args.valid_manifest) == VALID_MANIFEST_SHA, 'Frozen complete train/valid manifest differs')
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    root = args.numeric_root.resolve(); manifest_path = root / 'test.json'; structural_path = root / 'structural_report.json'
    status = {'status': 'running', 'cohort_sha256': sha(args.cohort), 'test_accessed': True, 'started_utc': now()}; write(output / 'preparation_status.json', status)
    try:
        import numpy as np
        manifest, structural = read(manifest_path), read(structural_path)
        require(sha(root / 'metadata.json') == METADATA_SHA and manifest.get('metadata') == read(root / 'metadata.json')
                and manifest.get('split') == 'test' and manifest.get('dataset') == 'Goop' and manifest.get('record_count') == len(manifest['records']) == 30
                and structural.get('schema') == 'official_goop_numeric_preparation_v1' and structural.get('status') == 'complete_structural_only'
                and structural.get('wrapper_sha256') == sha(__file__) and structural.get('cohort_sha256') == sha(args.cohort)
                and structural.get('splits', {}).get('test', {}).get('manifest_sha256') == sha(manifest_path), 'Matching complete test conversion required')
        receipt_check(read(args.acquisition_report), metadata, sha(args.cohort))
        aux = auxiliary_census(root, manifest, np); aux['structural_report_sha256'] = sha(structural_path); write(output / 'auxiliary_report.json', aux)
        manifests = {'train': args.train_manifest, 'valid': args.valid_manifest, 'test': manifest_path}
        seen, duplicates, records, counts = {}, [], {}, {}
        for split, path in manifests.items():
            value = read(path); counts[split] = len(value['records']); records[split] = []
            require(value.get('dataset') == 'Goop' and value.get('split') == split and value.get('record_count') == counts[split], 'Wrong split identity/count')
            for record in value['records']:
                key = numeric_identity(path.resolve().parent, record, np)
                identity = {'split': split, 'id': record['id'], 'source_index': record['source_index']}
                if key in seen: duplicates.append({'left': seen[key], 'right': identity, 'numeric_content_sha256': key})
                else: seen[key] = identity
                records[split].append({**identity, 'numeric_content_sha256': key})
        require(counts == {'train': 1000, 'valid': 30, 'test': 30}, 'Complete all-split counts required')
        overlap = {'schema': 'adaptgns_goop_all_split_integrity_audit_v1', 'issued_by': 'root',
            'status': 'all_required_splits_verified' if not duplicates else 'failed_duplicates_retained', 'duplicate_pairs': duplicates,
            'manifest_sha256': {s: sha(p) for s, p in manifests.items()}, 'record_counts': counts,
            'definition': 'exact stored position/type dtype,shape,bytes; auxiliaries preserved separately',
            'array_hash_scope': 'all numeric position/type file hashes, then exactCelement bytes+dtypes+shapes independent ofNPYheaders',
            'records': records, 'source_sha256': sha(__file__), 'cohort_sha256': sha(args.cohort)}
        write(output / 'cross_split_audit.json', overlap)
        require(not duplicates, 'Exact cross/within-split duplicates retained; root review required')
        contract = load(HERE / 'goop_evaluation_contract.py', CONTRACT_SHA, '_reserved_test_final_contract')
        candidate = {'schema': 'adaptgns_goop_graph_support_final_evaluation_admission_v1', 'status': 'prepared_requires_root_review',
            'issued_by': 'root', 'dataset': 'Goop', 'split': 'test', 'record_count': 30, 'frames_per_trajectory': 401, 'particle_type_ids': [7],
            'position_dtype': '<f4', 'particle_type_dtype': '<i8', 'auxiliary_policy': contract.OMISSION,
            'context_semantics_sha256': CONTEXT_SHA, 'metadata_sha256': METADATA_SHA, 'reader_sha256': READER_SHA,
            'converter_sha256': sha(__file__), 'acquisition_report_sha256': sha(args.acquisition_report), 'source_sha256': manifest['source']['sha256'],
            'manifest_sha256': sha(manifest_path), 'structural_report_sha256': sha(structural_path),
            'auxiliary_report_sha256': sha(output / 'auxiliary_report.json'), 'cross_split_audit_sha256': sha(output / 'cross_split_audit.json'),
            'reserved_test_acquired_after_cohort_freeze': True, 'test_converter_independently_reviewed': True,
            'cohort_manifest_sha256': sha(args.cohort), 'training_admission_sha256': TRAIN_ADMISSION_SHA, 'protocol_sha256': PROTOCOL_SHA,
            'preparation_release_sha256': sha(args.root_release)}
        # Pure structural check with an explicitly temporary in-memory status;
        # no admitted file is published and no model/test evaluation runs.
        contract.validate_split(manifest, {**candidate, 'status': 'admitted_for_final_evaluation'}, structural, 'test', final=True)
        observed = aux['splits']['test']['records']
        require(all(r['elements'] == 401 and r['nan_count'] == 1 and r['finite_count'] == 400 and r['positive_infinity_count'] == r['negative_infinity_count'] == 0
                    and r['unique_float32_bits_hex'] == ['00000000', '7fc00000'] for r in observed),
                'Actual auxiliary census differs from frozen evaluator expectations; preserve and review')
        write(output / 'final_test_admission.candidate.json', candidate)
        status.update(status='ready_for_root_admission_review', completed_utc=now()); write(output / 'preparation_status.json', status)
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error), all_existing_outputs_retained=True)
        write(output / 'preparation_status.json', status); raise


def preflight(args, release, metadata):
    require(all((args.numeric_root, args.admission, args.acquisition_report, args.context_semantics, args.auxiliary_report, args.cross_split_audit, args.output)),
            'Root admission and every final split/evidence path plus fresh report required')
    require(not args.output.exists(), 'Fresh preflight output required')
    contract = load(HERE / 'goop_evaluation_contract.py', CONTRACT_SHA, '_reserved_test_preflight_contract')
    admission = read(args.admission); root = args.numeric_root.resolve(); manifest_path = root / 'test.json'; structural_path = root / 'structural_report.json'
    require(admission.get('cohort_manifest_sha256') == sha(args.cohort) and admission.get('protocol_sha256') == PROTOCOL_SHA
            and admission.get('training_admission_sha256') == TRAIN_ADMISSION_SHA
            and admission.get('manifest_sha256') == sha(manifest_path) and admission.get('structural_report_sha256') == sha(structural_path), 'Root final split admission hash differs')
    manifest, structural = read(manifest_path), read(structural_path)
    contract.validate_split(manifest, admission, structural, 'test', final=True)
    import numpy as np
    evidence_args = SimpleNamespace(acquisition_report=args.acquisition_report, context_semantics=args.context_semantics,
        auxiliary_report=args.auxiliary_report, cross_split_audit=args.cross_split_audit, manifest=manifest_path, split='test')
    helpers = SimpleNamespace(np=np, data_loader=SimpleNamespace(_manifest_array_path=safe_array))
    files = contract.verify_evidence(evidence_args, manifest, admission, helpers)
    write(args.output, {'schema': 'adaptgns_goop_reserved_test_preflight_v1', 'status': 'frozen_split_and_evidence_contract_passed',
        'cohort_sha256': sha(args.cohort), 'admission_sha256': sha(args.admission), 'manifest_sha256': sha(manifest_path),
        'structural_report_sha256': sha(structural_path), 'verified_evidence_sha256': files, 'test_evaluation_executed': False,
        'source_sha256': sha(__file__), 'completed_utc': now()})


def interrupted(signum, frame): raise InterruptedError('Reserved-test preparation interrupted by signal' + str(signum))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False); p.add_argument('--execute', action='store_true')
    p.add_argument('--mode', choices=('inspect', 'acquire', 'convert', 'census', 'preflight'))
    for name in ('cohort', 'cohort-audit', 'root-release', 'source-metadata', 'metadata', 'output', 'output-dir', 'input-dir',
                 'acquisition-report', 'reader', 'numeric-root', 'train-manifest', 'valid-manifest', 'admission', 'context-semantics', 'auxiliary-report', 'cross-split-audit'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status': 'description_only', 'test_accessed': False, 'modes': ['inspect', 'acquire', 'convert', 'census', 'preflight'],
            'gate': 'all six exact100k checkpoints and root paired audit before evenHEAD; no invented test source checksum',
            'automatic_training_or_evaluation': False}, indent=2)); return 0
    require(args.mode and args.cohort and args.cohort_audit and args.root_release, 'Explicit mode/cohort/audit/root release required')
    cohort, release = gate(args)
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.mode == 'inspect':
            import requests
            inspect_source(args, release, requests)
        else:
            metadata = source_metadata(args, release)
            if args.mode == 'acquire':
                import requests, crc32c
                acquire(args, release, metadata, requests, crc32c)
            elif args.mode == 'convert': convert(args, release, metadata)
            elif args.mode == 'census': census(args, release, metadata)
            else: preflight(args, release, metadata)
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)
    return 0


if __name__ == '__main__': raise SystemExit(main())
