#!/usr/bin/env python3
"""Separate root-gated reserved Goop-3D test preparation; description by default.

No test request or file read before complete D3v2 scientific cohort/audit release.
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
BASE = 'https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop-3D/'
URL = BASE + 'test.tfrecord'
READER_SHA = 'ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33'
CONVERTER_SHA = '49dece28bf4494591379b8a667e5366b5bdf1609c4dc0757ceb31664c28f9b74'
EVALUATOR_SHA = '9364dbbdd44d78212979f9cc6b381a545d0d875b9463d84d33155b3787c6a1de'
TRAINER_SHA = '8de9b7d1de9435c1042151be49d0f6521b0a562ae9094d9e50060262ecad8fdc'
AUXILIARY_SHA = '1b5a8c24fde395b2633ee197557fde3e5bf30ca74908a77f1fb0e011b85e41fb'
LOADER_SHA = '287f26902068d84ea1fd74cccbd37459e415d046a10453f1011c6fecb1330bef'
METADATA_SHA = '727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'
PROTOCOL_SHA = '5010f9023a3eee45f85bee80b35fc8e506ca145667daf75027b32c92658faa70'
TRAIN_MANIFEST_SHA = '0f0ce1802e202b9faaf86f0a6ce059c286ed454ceb53273cb926980614b0f864'
VALID_MANIFEST_SHA = 'f79a20101e3926a9aa9ba91c060f694d673a9f58ef8b565ebf895f7dc2793cef'
CONTEXT_SHA = '5eb6818ae2699c57e62e80f63724248e8573e4536195eb471df0c647122fb1a5'
ACQ_SCHEMA = 'official_goop3d_reserved_test_acquisition_v1'
GATE_SCHEMA = 'adaptgns_goop3d_reserved_test_preparation_release_v1'
HEAD_SCHEMA = 'official_goop3d_reserved_test_source_metadata_v1'
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


def input_paths(args):
    paths = {k: getattr(args, k) for k in ('cohort', 'cohort_audit', 'protocol', 'trainer_source')}
    if args.mode != 'inspect': paths['source_metadata'] = args.source_metadata
    if args.mode == 'acquire': paths['metadata'] = args.metadata
    if args.mode == 'convert':
        paths.update(acquisition_report=args.acquisition_report, reader=args.reader,
                     source=args.input_dir/'test.tfrecord', metadata=args.input_dir/'metadata.json')
    if args.mode in ('census', 'preflight'):
        paths.update(acquisition_report=args.acquisition_report, context_semantics=args.context_semantics,
            manifest=args.numeric_root/'test.json', structural_report=args.numeric_root/'structural_report.json',
            metadata=args.numeric_root/'metadata.json')
    if args.mode == 'census': paths.update(train_manifest=args.train_manifest, valid_manifest=args.valid_manifest)
    if args.mode == 'preflight':
        paths.update(admission=args.admission, auxiliary_report=args.auxiliary_report,
            cross_split_audit=args.cross_split_audit, data_loader=args.data_loader)
    require(all(p is not None for p in paths.values()), 'Every mode-specific explicit input is required')
    return {k: p.resolve() for k, p in paths.items()}


def verify_stable(args):
    require(all(sha(p) == d for p, d in args._bindings.items()), 'Root-bound source or preparation input changed')
    require(datetime.now(timezone.utc) < CUTOFF, 'Preparation cutoff reached')


def gate(args):
    # Capture the authorizing bytes before parsing. No test file/HTTP access yet.
    raw = {k: getattr(args, k).read_bytes() for k in ('cohort', 'cohort_audit', 'root_release')}
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
    cohort, audit, release = [json.loads(raw[k]) for k in ('cohort', 'cohort_audit', 'root_release')]
    source_sha = sha(__file__)
    require(release.get('schema') == GATE_SCHEMA and release.get('status') == 'approved_for_' + args.mode
            and release.get('mode') == args.mode and release.get('issued_by') == 'root'
            and release.get('cohort_sha256') == hashes['cohort'] and release.get('cohort_audit_sha256') == hashes['cohort_audit']
            and release.get('preparation_source_sha256') == source_sha and release.get('source_url') == URL
            and release.get('all_six_checkpoint_hashes_verified') is True
            and release.get('test_converter_independently_reviewed') is True
            and release.get('whole_invocation_outer_timeout_required') is True
            and isinstance(release.get('review_rationale'), str) and bool(release['review_rationale'].strip()),
            'Exact mode-specific root reserved-test release required')
    require(sha(args.protocol) == PROTOCOL_SHA and sha(args.trainer_source) == TRAINER_SHA,
            'Frozen scientific protocol/trainer required before test access')
    E = load(HERE/'evaluate_goop3d_graph_support_v1.py', EVALUATOR_SHA, '_reserved3d_cohort_contract')
    for model in cohort.get('models', []):
        E.cohort_gate(cohort, audit, {'scientific_endpoint_updates': release.get('endpoint_updates')},
            SimpleNamespace(protocol=args.protocol, trainer_source=args.trainer_source, cohort_audit=args.cohort_audit,
                arm=model.get('arm'), seed=model.get('seed'), checkpoint_sha256=model.get('checkpoint_sha256')))
    require(len(cohort.get('models', [])) == 6 and cohort.get('evaluation_admitted') is True
            and audit.get('evaluation_admitted') is True and all(p.get('initial_empty_adam_identity') is True for p in audit['paired_seeds']),
            'Complete root-frozen six-endpoint D3 cohort required before test access')
    require(stamped(cohort['created_utc']) <= stamped(release['issued_utc']) <= datetime.now(timezone.utc) < CUTOFF,
            'Test access must follow cohort freeze and precede compute cutoff')
    output = args.output if args.mode in ('inspect', 'preflight') else args.output_dir
    require(output is not None and release.get('output_path') == str(output.resolve()) and not output.exists(), 'Root-bound fresh output required')
    paths = input_paths(args)
    require(set(release.get('files_sha256', {})) == {str(p) for p in paths.values()}, 'Exact mode-specific file closure required')
    require(all(release['files_sha256'][str(getattr(args,k).resolve())] == hashes[k] for k in ('cohort','cohort_audit')), 'Cohort file-map hashes differ')
    # Only now may any released test input be opened/hashed.
    bindings = {str(p): release['files_sha256'][str(p)] for p in paths.values()}
    bindings.update({str(getattr(args,k).resolve()): hashes[k] for k in raw})
    bindings.update({str(Path(__file__).resolve()): source_sha, str(HERE/'evaluate_goop3d_graph_support_v1.py'): EVALUATOR_SHA,
        str(HERE/'prepare_goop3d_official.py'): CONVERTER_SHA, str(HERE/'audit_goop3d_auxiliary.py'): AUXILIARY_SHA})
    require(all(digest(d) for d in bindings.values()), 'Valid input SHA256 bindings required')
    args._bindings = bindings; verify_stable(args)
    return cohort, release


def source_metadata(args, release):
    require(args.source_metadata and sha(args.source_metadata) == release.get('source_metadata_sha256'), 'Root-pinned source header receipt required')
    value = read(args.source_metadata)
    require(value.get('schema') == HEAD_SCHEMA and value.get('status') == 'headers_verified' and value.get('url') == URL
            and value.get('cohort_sha256') == sha(args.cohort) and value.get('response_status') == 200
            and value.get('preparation_source_sha256') == sha(__file__)
            and type(value.get('size_bytes')) is int and value['size_bytes'] > 0
            and isinstance(value.get('generation'), str) and value['generation'].isdigit()
            and len(base64.b64decode(value.get('crc32c_base64', ''), validate=True)) == 4,
            'Complete exact source generation/length/CRC header receipt required')
    require(stamped(read(args.cohort)['created_utc']) <= stamped(value['started_utc']) <= stamped(value['ended_utc']) <= stamped(release['issued_utc']), 'Publisher header chronology must follow freeze and precede consuming release')
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
        verify_stable(args)
        with requests.head(URL, headers={'Accept-Encoding': 'identity'}, timeout=(20, 30), allow_redirects=False) as response:
            report.update(response_status=response.status_code, response_headers=dict(response.headers))
            write(args.output, report)
            size, generation, crc = header_values(response)
        verify_stable(args)
        report.update(status='headers_verified', size_bytes=size, generation=generation, crc32c_base64=crc, ended_utc=now())
        write(args.output, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), ended_utc=now()); write(args.output, report); raise


def acquire(args, release, metadata, requests, crc32c):
    require(args.output_dir and args.metadata and sha(args.metadata) == METADATA_SHA, 'Fresh acquisition root and original metadata bytes required')
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'acquisition_report.json'
    report = {'schema': ACQ_SCHEMA, 'status': 'running', 'dataset': 'Goop-3D', 'source_family': 'official_gns_tfrecord', 'test_accessed': True,
              'cohort_sha256': sha(args.cohort), 'cohort_freeze_utc': read(args.cohort)['created_utc'], 'root_release_sha256': sha(args.root_release),
              'source_metadata_sha256': sha(args.source_metadata), 'source_sha256': sha(__file__), 'started_utc': now(), 'files': []}
    write(report_path, report); partial = output / 'test.tfrecord.partial'; current = None
    started = time.perf_counter()
    try:
        verify_stable(args)
        require(crc32c.crc32c(b'123456789') == 0xe3069283, 'CRC32C implementation self-test failed')
        raw = args.metadata.read_bytes(); require(hashlib.sha256(raw).hexdigest() == METADATA_SHA, 'Metadata changed')
        (output / 'metadata.json').write_bytes(raw)
        report['files'].append({'name': 'metadata.json', 'saved_name': 'metadata.json', 'url': BASE + 'metadata.json',
            'status': 'complete', 'received_bytes': len(raw), 'generation': '1599153912104523', 'sha256': METADATA_SHA,
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
                    require(datetime.now(timezone.utc) < CUTOFF, 'Acquisition cutoff reached')
                    f.write(block); h.update(block); crc = crc32c.crc32c(block, crc); count += len(block); current['received_bytes'] = count
                    require(count <= metadata['size_bytes'], 'Received more than pinned source length')
                f.flush(); os.fsync(f.fileno())
            encoded = base64.b64encode(crc.to_bytes(4, 'big')).decode()
            current.update(sha256=h.hexdigest(), generation=metadata['generation'], crc32c_base64=encoded,
                           crc32c_verified=encoded == metadata['crc32c_base64'])
            require(count == metadata['size_bytes'] and encoded == metadata['crc32c_base64'], 'Complete source checksum/length differs')
        partial.rename(output / 'test.tfrecord'); current['status'] = 'complete'
        verify_stable(args)
        report.update(status='complete', ended_utc=now(), elapsed_seconds=time.perf_counter() - started); write(report_path, report)
    except BaseException as error:
        if partial.exists():
            current.update(retained_partial_sha256=sha(partial), retained_partial_bytes=partial.stat().st_size)
        if current is not None: current['status'] = 'failed'
        report.update(status='failed', error_type=type(error).__name__, error=str(error), ended_utc=now(), all_existing_outputs_retained=True)
        write(report_path, report); raise


def receipt_check(receipt, metadata, cohort_sha, freeze_utc):
    require(receipt.get('schema') == ACQ_SCHEMA and receipt.get('status') == 'complete' and receipt.get('dataset') == 'Goop-3D'
            and receipt.get('source_family') == 'official_gns_tfrecord' and receipt.get('cohort_sha256') == cohort_sha
            and receipt.get('test_accessed') is True and receipt.get('source_sha256') == sha(__file__), 'Complete post-cohort test acquisition receipt required')
    require(stamped(freeze_utc) <= stamped(receipt['started_utc']) <= stamped(receipt['ended_utc']), 'Acquisition must follow cohort freeze')
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
    helper = load(HERE / 'prepare_goop3d_official.py', CONVERTER_SHA, '_reserved_test_converter')
    report = {'schema': 'official_goop3d_numeric_preparation_v1', 'status': 'admitting', 'dataset': 'Goop-3D',
              'source_family': 'official_gns_tfrecord', 'test_accessed': True, 'scientific_training_admission': False,
              'cohort_sha256': sha(args.cohort), 'root_release_sha256': sha(args.root_release), 'splits': {}}
    report_path = output / 'structural_report.json'; write(report_path, report)
    try:
        verify_stable(args)
        receipt_hash = sha(args.acquisition_report); receipt = read(args.acquisition_report)
        require(receipt.get('source_metadata_sha256') == sha(args.source_metadata), 'Acquisition source metadata binding differs')
        rows = receipt_check(receipt, metadata, sha(args.cohort), read(args.cohort)['created_utc'])
        require(stamped(receipt['ended_utc']) <= stamped(release['issued_utc']), 'Conversion release must follow acquired source')
        metadata_bytes = (source / 'metadata.json').read_bytes(); require(hashlib.sha256(metadata_bytes).hexdigest() == METADATA_SHA, 'Metadata bytes differ')
        numeric_metadata = json.loads(metadata_bytes)
        reader = helper.load_reader(args.reader); helper.verify_source(reader, source / 'metadata.json', rows['metadata.json'])
        wrapper_sha = sha(__file__)
        report.update(wrapper_sha256=wrapper_sha, converter_helper_sha256=CONVERTER_SHA, reader_sha256=READER_SHA,
                      metadata_sha256=METADATA_SHA, acquisition_report_sha256=receipt_hash, numpy_version=reader.np.__version__)
        records, summary = helper.convert_one(reader, source / 'test.tfrecord', 'test', rows['test.tfrecord'], output, numeric_metadata, set())
        report['splits']['test'] = summary; write(report_path, report)
        write(output / '.test.staging' / 'preserved_records.json', records)
        # Preserve any discrepant source fully. No subset or representation repair.
        require(summary['record_count'] > 0 and summary['frame_lengths'] == [301]
                and summary['forecast_horizons_after_six_frames'] == [295] and summary['particle_type_ids'] == [7]
                and summary['dimension'] == 3,
                'Observed test structure differs from frozen final-evaluation contract; preserve and review')
        r = rows['test.tfrecord']
        manifest = {'format': reader.FORMAT, 'version': 1, 'dataset': 'Goop-3D', 'split': 'test', 'metadata': numeric_metadata,
            'metadata_sha256': METADATA_SHA, 'record_count': len(records), 'records': records,
            'source': {'family': 'official_gns_tfrecord', 'dataset': 'Goop-3D', 'file': 'test.tfrecord', 'size_bytes': r['received_bytes'],
                'sha256': r['sha256'], 'generation': r['generation'], 'crc32c_base64': r['crc32c_base64'], 'CRC_verified': True,
                'record_count': len(records), 'acquisition_report_sha256': receipt_hash}, 'converter_sha256': wrapper_sha, 'reader_sha256': READER_SHA}
        require(sha(args.acquisition_report) == receipt_hash and sha(args.reader) == READER_SHA and sha(__file__) == wrapper_sha
                and sha(HERE / 'prepare_goop3d_official.py') == CONVERTER_SHA, 'Bound source/receipt changed during conversion')
        verify_stable(args)
        (output / 'metadata.json').write_bytes(metadata_bytes); (output / '.test.staging').replace(output / 'test')
        write(output / 'test.json', manifest); summary['manifest_sha256'] = sha(output / 'test.json')
        report.update(status='complete_structural_only', completed_utc=now()); write(report_path, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, reason=str(error), all_existing_outputs_retained=True)
        write(report_path, report); raise


def safe_array(root, descriptor):
    root = Path(root).resolve(); relative = Path(descriptor['path']); require(not relative.is_absolute() and '..' not in relative.parts, 'Relative numeric path required')
    raw = root / relative; path = raw.resolve()
    require(not raw.is_symlink() and path.is_relative_to(root) and path.is_file() and path.suffix == '.npy' and not any(p.is_symlink() for p in raw.parents if p != root and p.is_relative_to(root)), 'Unsafe numeric path')
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


def all_split_census(manifests, np):
    helper = load(HERE/'audit_goop3d_auxiliary.py', AUXILIARY_SHA, '_reserved3d_auxiliary')
    auxiliary = {'schema': 'adaptgns_goop3d_auxiliary_census_v1', 'status': 'running', 'test_accessed': True,
        'source_sha256': sha(__file__), 'census_helper_sha256': AUXILIARY_SHA, 'metadata_sha256': METADATA_SHA, 'splits': {}}
    seen, duplicates, records, counts, files = {}, [], {}, {}, {}
    for split, path in manifests.items():
        value = read(path); count = len(value['records']); counts[split] = count; records[split] = []
        require(value.get('dataset') == 'Goop-3D' and value.get('split') == split and count > 0 and value.get('record_count') == count
            and value.get('metadata_sha256') == METADATA_SHA and sha(path.parent/'metadata.json') == METADATA_SHA,
            'Complete all-split metadata/source identity required')
        files[str(path.resolve())] = sha(path); files[str(path.parent/'metadata.json')] = METADATA_SHA
        rows=[]
        for index, record in enumerate(value['records']):
            require(record['id'] == f'{split}:{index:06d}' and record['source_index'] == index, 'Complete source order required')
            key = numeric_identity(path.resolve().parent, record, np)
            require(hashlib.sha256((record['positions']['sha256']+':'+record['particle_types']['sha256']).encode()).hexdigest() == record['trajectory_content_sha256'], 'Original exact-content manifest identity differs')
            for field in ('positions', 'particle_types'):
                d=record[field]; p=safe_array(path.parent,d); files[str(p)] = d['sha256']
                a=np.load(p,mmap_mode='r',allow_pickle=False)
                require((field!='positions' or (a.shape[0]==301 and a.shape[2]==3 and np.isfinite(a).all()))
                    and (field!='particle_types' or (a.ndim==1 and np.all(a==7))), 'Actual T301D3/type7 numeric representation required')
                del a
            row, p = helper.census_record(path.parent,split,record,np); rows.append(row)
            if p is not None: files[str(p)] = record['step_context']['sha256']
            identity={'split':split,'id':record['id'],'source_index':index}
            if key in seen: duplicates.append({'left':seen[key],'right':identity,'numeric_content_sha256':key})
            else: seen[key]=identity
            records[split].append({**identity,'numeric_content_sha256':key})
        auxiliary['splits'][split]={'manifest_sha256':sha(path),'record_count':count,'records':rows,
            'context_mean_absent':'context_mean' not in value['metadata'],'context_std_absent':'context_std' not in value['metadata'],
            'auxiliary_present_count':sum(r['present'] for r in rows),'auxiliary_absent_count':sum(not r['present'] for r in rows),
            'all_context_descriptors_and_bytes_verified':True}
    require(counts['train']==1000 and counts['valid']==100 and counts['test']>0, 'Complete1000train/100valid and actual positive testN required')
    auxiliary.update(status='all_preserved_auxiliary_bytes_verified',
        semantics='Preserved byte census only; omission follows separately pinned official parser without context_mean; no inference from constancy or NaN.')
    overlap={'schema':'adaptgns_goop3d_all_split_integrity_audit_v1','issued_by':'preparation_wrapper_not_root',
        'status':'all_required_splits_verified_requires_root_review' if not duplicates else 'failed_duplicates_retained',
        'duplicate_pairs':duplicates,'manifest_sha256':{s:sha(p) for s,p in manifests.items()},'record_counts':counts,'records':records,
        'definition':'exact stored position/type dtype,shape,bytes; auxiliaries preserved separately',
        'array_hash_scope':'All position/type file hashes and exact element bytes, dtypes and shapes independent of NPY headers',
        'source_sha256':sha(__file__)}
    return auxiliary,overlap,files


def census(args, release, metadata):
    require(sha(args.train_manifest)==TRAIN_MANIFEST_SHA and sha(args.valid_manifest)==VALID_MANIFEST_SHA, 'Frozen complete train/valid manifests required')
    root=args.numeric_root.resolve(); manifest_path=root/'test.json'; structural_path=root/'structural_report.json'
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    status={'status':'running','cohort_sha256':sha(args.cohort),'test_accessed':True,'started_utc':now()}
    write(output/'preparation_status.json',status)
    try:
        verify_stable(args)
        import numpy as np
        manifest,structural=read(manifest_path),read(structural_path)
        require(sha(root/'metadata.json')==METADATA_SHA and manifest.get('metadata')==read(root/'metadata.json')
            and manifest.get('split')=='test' and manifest.get('dataset')=='Goop-3D'
            and manifest.get('record_count')==len(manifest['records'])>0
            and structural.get('schema')=='official_goop3d_numeric_preparation_v1' and structural.get('status')=='complete_structural_only'
            and structural.get('wrapper_sha256')==sha(__file__) and structural.get('converter_helper_sha256')==CONVERTER_SHA
            and structural.get('cohort_sha256')==sha(args.cohort)
            and structural.get('splits',{}).get('test',{}).get('manifest_sha256')==sha(manifest_path), 'Complete D3 test conversion required')
        receipt=read(args.acquisition_report)
        rows=receipt_check(receipt,metadata,sha(args.cohort),read(args.cohort)['created_utc'])
        require(stamped(receipt['ended_utc']) <= stamped(release['issued_utc']), 'Census release must follow acquired source')
        require(receipt.get('source_metadata_sha256')==sha(args.source_metadata)
            and manifest['source']['sha256']==rows['test.tfrecord']['sha256']
            and manifest['source']['acquisition_report_sha256']==sha(args.acquisition_report), 'Acquisition/manifest source binding differs')
        require(sha(args.context_semantics)==CONTEXT_SHA, 'Frozen D3 context semantics required')
        context=read(args.context_semantics); context_root=args.context_semantics.parent/'goop_context_semantics_sources'
        context_files={str(context_root/name):r['sha256'] for name,r in context['sources'].items()}
        require(all(sha(p)==d for p,d in context_files.items()),'Official context source bytes differ')
        aux,cross,files=all_split_census({'train':args.train_manifest,'valid':args.valid_manifest,'test':manifest_path},np)
        aux.update(structural_report_sha256=sha(structural_path),cohort_sha256=sha(args.cohort))
        cross.update(cohort_sha256=sha(args.cohort))
        write(output/'auxiliary_report.json',aux);write(output/'cross_split_audit.candidate.json',cross)
        require(not cross['duplicate_pairs'],'Exact within/cross-split duplicates retained; root review required')
        candidate={'schema':'adaptgns_goop3d_evaluation_split_admission_v1','status':'prepared_requires_root_review',
            'issued_by':'preparation_wrapper_not_root','dataset':'Goop-3D','split':'test','record_count':manifest['record_count'],
            'frames':301,'dimension':3,'particle_type_ids':[7],'metadata_sha256':METADATA_SHA,'reader_sha256':READER_SHA,
            'converter_sha256':sha(__file__),'converter_helper_sha256':CONVERTER_SHA,'context_semantics_sha256':CONTEXT_SHA,
            'acquisition_report_sha256':sha(args.acquisition_report),'source_sha256':manifest['source']['sha256'],
            'manifest_sha256':sha(manifest_path),'structural_report_sha256':sha(structural_path),
            'auxiliary_report_sha256':sha(output/'auxiliary_report.json'),'cross_split_audit_sha256':None,
            'cross_split_audit_candidate_sha256':sha(output/'cross_split_audit.candidate.json'),
            'cohort_sha256':sha(args.cohort),'reserved_test_acquired_after_cohort_freeze':True,
            'test_converter_independently_reviewed':True,'preparation_release_sha256':sha(args.root_release)}
        files.update(context_files)
        require(all(sha(p)==d for p,d in files.items()), 'All-split numeric/context bytes changed during census')
        verify_stable(args)
        write(output/'verified_input_bytes.json',files)
        write(output/'final_test_admission.candidate.json',candidate)
        status.update(status='ready_for_root_admission_review',completed_utc=now(),test_evaluation_executed=False,
            root_must_issue_separate_cross_split_audit_and_test_admission=True);write(output/'preparation_status.json',status)
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error),all_existing_outputs_retained=True)
        write(output/'preparation_status.json',status);raise


def preflight(args, release, metadata):
    output=args.output;report={'schema':'adaptgns_goop3d_reserved_test_preflight_v1','status':'running',
        'cohort_sha256':sha(args.cohort),'started_utc':now(),'test_evaluation_executed':False,'source_sha256':sha(__file__)}
    write(output,report)
    try:
        verify_stable(args)
        import numpy as np
        E=load(HERE/'evaluate_goop3d_graph_support_v1.py',EVALUATOR_SHA,'_reserved3d_split_contract')
        loader=load(args.data_loader,LOADER_SHA,'_reserved3d_numeric_loader')
        receipt_check(read(args.acquisition_report),metadata,sha(args.cohort),read(args.cohort)['created_utc'])
        root=args.numeric_root.resolve()
        evidence=SimpleNamespace(manifest=root/'test.json',structural_report=root/'structural_report.json',split_admission=args.admission,
            acquisition_report=args.acquisition_report,context_semantics=args.context_semantics,auxiliary_report=args.auxiliary_report,
            cross_split_audit=args.cross_split_audit,cohort=args.cohort,split='test',purpose='final_evaluation')
        manifest,trajectories,files=E.check_split(evidence,{},SimpleNamespace(np=np,data_loader=loader))
        schedules={mode:E.schedules(manifest['records'],mode,'final_evaluation') for mode in ('full-rollout','same-state')}
        del trajectories
        E.verify_bindings(files);verify_stable(args)
        report.update(status='frozen_split_and_evidence_contract_passed',admission_sha256=sha(args.admission),
            manifest_sha256=sha(root/'test.json'),structural_report_sha256=sha(root/'structural_report.json'),
            verified_evidence_sha256=files,source_order_grid=schedules['full-rollout'],
            same_state_scheduled_histories=len(schedules['same-state']),
            clean_validation_is_not_run_on_test=True,evaluator_sha256=EVALUATOR_SHA,data_loader_sha256=LOADER_SHA,completed_utc=now())
        write(output,report)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),all_existing_outputs_retained=True)
        write(output,report);raise


def interrupted(signum, frame): raise InterruptedError('Reserved-test preparation interrupted by signal' + str(signum))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False); p.add_argument('--execute', action='store_true')
    p.add_argument('--mode', choices=('inspect', 'acquire', 'convert', 'census', 'preflight'))
    for name in ('cohort', 'cohort-audit', 'root-release', 'source-metadata', 'metadata', 'output', 'output-dir', 'input-dir',
                 'acquisition-report', 'reader', 'numeric-root', 'train-manifest', 'valid-manifest', 'admission', 'context-semantics', 'auxiliary-report', 'cross-split-audit', 'protocol', 'trainer-source', 'data-loader'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status': 'description_only', 'test_accessed': False, 'modes': ['inspect', 'acquire', 'convert', 'census', 'preflight'],
            'gate': 'all six prospective D3 scientific endpoints and root paired audit before even HEAD; no invented checksum/count',
            'automatic_training_or_evaluation': False}, indent=2)); return 0
    require(args.mode and args.cohort and args.cohort_audit and args.root_release and args.protocol and args.trainer_source, 'Explicit mode/cohort/audit/root release required')
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
