"""Synthetic TFRecords/mock HTTP only; no reserved source request or model use."""
import base64
import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('reserved_test_prep', HERE / 'prepare_goop_reserved_test_v1.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
reader = M.load(ROOT / 'outputs/AdaptGNS/research/prepare_full_waterdrop.py', M.READER_SHA, '_synthetic_test_reader')
np = reader.np
METADATA = HERE.parent / 'extension_feasibility_20261006/Goop_metadata_json.json'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2) + '\n'); return path


def args(**values):
    defaults = {k: None for k in ('cohort', 'cohort_audit', 'root_release', 'source_metadata', 'metadata', 'output', 'output_dir',
        'input_dir', 'acquisition_report', 'reader', 'numeric_root', 'train_manifest', 'valid_manifest', 'admission',
        'context_semantics', 'auxiliary_report', 'cross_split_audit')}
    return SimpleNamespace(**{**defaults, **values})


@pytest.fixture
def gate_args(tmp_path):
    models = [{'arm': a, 'seed': s, 'objective': 'faithful', 'completed_steps': 100000,
               'checkpoint_sha256': str(1 + s + (3 if a == 'mix' else 0)) * 64} for a in ('base', 'mix') for s in range(3)]
    audit = {'schema': 'adaptgns_goop_graph_support_complete_cohort_audit_v1', 'status': 'all_six_endpoints_and_pairing_verified',
        'issued_by': 'root', 'training_schema': 'adaptgns_goop_graph_support_cuda_training_v1', 'protocol_sha256': M.PROTOCOL_SHA,
        'training_admission_sha256': M.TRAIN_ADMISSION_SHA,
        'models': [{**m, 'graph_history_updates': 100000, 'checkpoint_every': 10000, 'log_every': 100,
                    'all_optimizer_steps_equal_100000': True, 'all_state_and_moments_finite': True,
                    'source_data_protocol_verified': True, 'checkpoint_bytes_verified': True} for m in models],
        'paired_seeds': [{'seed': s, 'initial_model_tensor_identity': True, 'initial_cpu_cuda_rng_identity': True,
                         'all_frame_noise_lr_schedules_equal': True, 'all_graph_budgets_and_rng_material_verified': True} for s in range(3)]}
    ap = put(tmp_path / 'audit.json', audit)
    cohort = {'schema': 'adaptgns_goop_graph_support_final_cohort_v1', 'status': 'frozen_for_final_evaluation', 'issued_by': 'root',
        'dataset': 'Goop', 'updates': 100000, 'protocol_sha256': M.PROTOCOL_SHA, 'training_admission_sha256': M.TRAIN_ADMISSION_SHA,
        'cohort_audit_sha256': M.sha(ap), 'policies': M.POLICIES, 'models': models,
        'created_utc': (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    cp = put(tmp_path / 'cohort.json', cohort)
    release = {'schema': M.GATE_SCHEMA, 'status': 'approved_for_source_metadata', 'issued_by': 'root', 'issued_utc': M.now(),
        'cohort_sha256': M.sha(cp), 'cohort_audit_sha256': M.sha(ap), 'preparation_source_sha256': M.sha(M.__file__),
        'source_url': M.URL, 'all_six_checkpoint_hashes_verified': True, 'test_converter_independently_reviewed': True,
        'review_rationale': 'Synthetic authorization fixture only'}
    rp = put(tmp_path / 'release.json', release)
    return args(mode='inspect', cohort=cp, cohort_audit=ap, root_release=rp, output=tmp_path / 'headers.json')


def test_default_describes_without_access(monkeypatch):
    monkeypatch.setattr(M, 'gate', lambda _: (_ for _ in ()).throw(AssertionError('gate must not run')))
    out = io.StringIO()
    with contextlib.redirect_stdout(out): assert M.main([]) == 0
    assert not json.loads(out.getvalue())['test_accessed']


def test_complete_gate(gate_args):
    cohort, release = M.gate(gate_args)
    assert len(cohort['models']) == 6 and release['source_url'] == M.URL


@pytest.mark.parametrize('change', ['incomplete', 'bad_pairing', 'wrong_source', 'unreviewed', 'wrong_protocol'])
def test_gate_rejects_before_source_metadata_or_http(gate_args, change, monkeypatch):
    if change in ('incomplete', 'wrong_protocol'):
        c = M.read(gate_args.cohort)
        if change == 'incomplete': c['models'][0]['completed_steps'] = 99999
        else: c['protocol_sha256'] = '0' * 64
        put(gate_args.cohort, c)
    elif change == 'bad_pairing':
        a = M.read(gate_args.cohort_audit); a['paired_seeds'][0]['initial_model_tensor_identity'] = False; put(gate_args.cohort_audit, a)
    else:
        r = M.read(gate_args.root_release)
        if change == 'wrong_source': r['source_url'] = M.BASE + 'valid.tfrecord'
        else: r['test_converter_independently_reviewed'] = False
        put(gate_args.root_release, r)
    monkeypatch.setattr(M, 'inspect_source', lambda *a: (_ for _ in ()).throw(AssertionError('must not request')))
    with pytest.raises(ValueError):
        M.main(['--execute', '--mode', 'inspect', '--cohort', str(gate_args.cohort), '--cohort-audit', str(gate_args.cohort_audit),
                '--root-release', str(gate_args.root_release), '--output', str(gate_args.output)])
    assert not gate_args.output.exists()


class Response:
    def __init__(self, raw, generation='123456', bad_crc=False, fail=False):
        self.raw, self.fail = raw, fail; self.status_code = 200
        crc = base64.b64encode(reader.crc32c.crc32c(raw).to_bytes(4, 'big')).decode()
        self.headers = {'Content-Length': str(len(raw)), 'x-goog-generation': generation,
                        'x-goog-hash': 'crc32c=' + ('AAAAAA==' if bad_crc else crc)}
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_content(self, chunk_size):
        yield self.raw[:len(self.raw)//2]
        if self.fail: raise OSError('synthetic stream interruption')
        yield self.raw[len(self.raw)//2:]


class Requests:
    def __init__(self, response): self.response, self.calls = response, []
    def head(self, url, **kwargs): self.calls.append(('HEAD', url, kwargs)); return self.response
    def get(self, url, **kwargs): self.calls.append(('GET', url, kwargs)); return self.response


def test_headers_pinned_no_payload_download(gate_args):
    req = Requests(Response(b'fake source bytes'))
    M.inspect_source(gate_args, M.read(gate_args.root_release), req)
    value = M.read(gate_args.output)
    assert value['status'] == 'headers_verified' and not value['payload_downloaded']
    assert req.calls[0][0] == 'HEAD' and req.calls[0][1] == M.URL and req.calls[0][2]['allow_redirects'] is False


def acquisition_args(gate_args, tmp_path, response):
    a = args(**vars(gate_args)); a.mode = 'acquire'; a.source_metadata = tmp_path / 'source_metadata.json'; a.metadata = METADATA
    a.output_dir = tmp_path / 'acquired'
    size, generation, crc = M.header_values(response)
    meta = {'schema': M.HEAD_SCHEMA, 'status': 'headers_verified', 'url': M.URL, 'cohort_sha256': M.sha(a.cohort),
            'response_status': 200, 'size_bytes': size, 'generation': generation, 'crc32c_base64': crc}
    put(a.source_metadata, meta)
    r = M.read(a.root_release); r.update(status='approved_for_acquisition_preparation', source_metadata_sha256=M.sha(a.source_metadata)); put(a.root_release, r)
    return a, r, meta


def test_generation_pinned_acquisition_and_exact_partial_hash(gate_args, tmp_path):
    response = Response(b'synthetic source bytes' * 100)
    a, release, meta = acquisition_args(gate_args, tmp_path, response); req = Requests(response)
    M.acquire(a, release, meta, req, reader.crc32c)
    report = M.read(a.output_dir / 'acquisition_report.json')
    assert report['status'] == 'complete' and report['files'][1]['sha256'] == hashlib.sha256(response.raw).hexdigest()
    assert req.calls[0][2]['params'] == {'generation': meta['generation']}
    assert report['files'][0]['retrieval'] == 'local_hash_verified_copy_of_frozen_official_metadata'


def test_interrupted_acquisition_retains_partial_bytes_and_hash(gate_args, tmp_path):
    response = Response(b'synthetic source bytes' * 100, fail=True)
    a, release, meta = acquisition_args(gate_args, tmp_path, response)
    with pytest.raises(OSError): M.acquire(a, release, meta, Requests(response), reader.crc32c)
    partial = a.output_dir / 'test.tfrecord.partial'; report = M.read(a.output_dir / 'acquisition_report.json')
    assert report['status'] == 'failed' and report['files'][1]['retained_partial_sha256'] == M.sha(partial)
    assert partial.read_bytes() == response.raw[:len(response.raw)//2]


def test_get_generation_mismatch_fails_before_body(gate_args, tmp_path):
    response = Response(b'x' * 100)
    a, release, meta = acquisition_args(gate_args, tmp_path, response); response.headers['x-goog-generation'] = '789'
    with pytest.raises(ValueError, match='root-pinned'): M.acquire(a, release, meta, Requests(response), reader.crc32c)
    assert not (a.output_dir / 'test.tfrecord').exists()
    assert M.read(a.output_dir / 'acquisition_report.json')['status'] == 'failed'


def payload(index, frames=401, bits=0x7fc00000):
    ex = reader.example_pb2.SequenceExample(); ex.context.feature['key'].int64_list.value.append(index)
    ex.context.feature['particle_type'].bytes_list.value.append(np.array([7, 7], dtype='<i8').tobytes())
    positions = (np.arange(frames * 4, dtype='<f4').reshape(frames, 2, 2) / 10000 + .2 + index / 1000).astype('<f4')
    aux = np.zeros((frames, 1), dtype='<f4'); aux[-1, 0] = np.array([bits], dtype='<u4').view('<f4')[0]
    for p, a in zip(positions, aux):
        ex.feature_lists.feature_list['position'].feature.add().bytes_list.value.append(p.tobytes())
        ex.feature_lists.feature_list['step_context'].feature.add().bytes_list.value.append(a.tobytes())
    return ex.SerializeToString()


def encoded_records(values):
    raw = b''
    for p in values:
        size = struct.pack('<Q', len(p))
        raw += size + struct.pack('<I', reader.masked_crc(size)) + p + struct.pack('<I', reader.masked_crc(p))
    return raw


@pytest.fixture
def converted(gate_args, tmp_path):
    response = Response(encoded_records([payload(i) for i in range(30)]))
    a, release, meta = acquisition_args(gate_args, tmp_path, response)
    M.acquire(a, release, meta, Requests(response), reader.crc32c)
    a.input_dir = a.output_dir; a.acquisition_report = a.input_dir / 'acquisition_report.json'
    a.output_dir = tmp_path / 'numeric'; a.reader = ROOT / 'outputs/AdaptGNS/research/prepare_full_waterdrop.py'; a.mode = 'convert'
    M.convert(a, release, meta)
    return a, release, meta


def test_complete_conversion_preserves_all30_and_matches_frozen_split_contract(converted, tmp_path):
    a, release, meta = converted; root = a.output_dir
    manifest, structural = M.read(root / 'test.json'), M.read(root / 'structural_report.json')
    assert manifest['record_count'] == 30 and structural['status'] == 'complete_structural_only'
    assert manifest['source']['generation'] == meta['generation'] and manifest['converter_sha256'] == M.sha(M.__file__)
    census = M.auxiliary_census(root, manifest, np)
    assert census['splits']['test']['records'][0]['nan_count'] == 1
    assert census['splits']['test']['records'][0]['unique_float32_bits_hex'] == ['00000000', '7fc00000']
    contract = M.load(HERE / 'goop_evaluation_contract.py', M.CONTRACT_SHA, '_test_frozen_contract')
    admission = {'schema': 'adaptgns_goop_graph_support_final_evaluation_admission_v1', 'status': 'admitted_for_final_evaluation',
        'issued_by': 'root', 'dataset': 'Goop', 'split': 'test', 'record_count': 30, 'frames_per_trajectory': 401,
        'particle_type_ids': [7], 'position_dtype': '<f4', 'particle_type_dtype': '<i8', 'auxiliary_policy': contract.OMISSION,
        'context_semantics_sha256': M.CONTEXT_SHA, 'metadata_sha256': M.METADATA_SHA, 'reader_sha256': M.READER_SHA,
        'converter_sha256': M.sha(M.__file__), 'source_sha256': manifest['source']['sha256'],
        'manifest_sha256': M.sha(root / 'test.json'), 'structural_report_sha256': M.sha(root / 'structural_report.json'),
        'acquisition_report_sha256': M.sha(a.acquisition_report), 'auxiliary_report_sha256': 'b'*64,
        'cross_split_audit_sha256': 'c'*64, 'reserved_test_acquired_after_cohort_freeze': True, 'test_converter_independently_reviewed': True}
    assert contract.validate_split(manifest, admission, structural, 'test', final=True) == 401


def test_duplicate_numeric_identity_ignores_id_and_npy_header(converted):
    a, _, _ = converted; manifest = M.read(a.output_dir / 'test.json')
    first, second = manifest['records'][:2]
    assert M.numeric_identity(a.output_dir, first, np) != M.numeric_identity(a.output_dir, second, np)
    copied = dict(first, id='another:000000', source_index=999)
    assert M.numeric_identity(a.output_dir, first, np) == M.numeric_identity(a.output_dir, copied, np)


def test_census_preserves_noncanonical_nan_for_review(converted):
    a, _, _ = converted; manifest = M.read(a.output_dir / 'test.json'); record = manifest['records'][0]
    d = record['step_context']; path = a.output_dir / d['path']; values = np.load(path); values.view('<u4')[-1, 0] = 0x7fc12345
    np.save(path, values, allow_pickle=False); d['sha256'] = M.sha(path); d['size_bytes'] = path.stat().st_size
    census = M.auxiliary_census(a.output_dir, manifest, np)
    assert census['splits']['test']['records'][0]['unique_float32_bits_hex'] == ['00000000', '7fc12345']
    assert values.view('<u4')[-1, 0] == 0x7fc12345


def test_changed_numeric_bytes_refused(converted):
    a, _, _ = converted; record = M.read(a.output_dir / 'test.json')['records'][0]
    path = a.output_dir / record['particle_types']['path']; path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='array bytes'): M.numeric_identity(a.output_dir, record, np)


def test_partial_test_conversion_retained_not_subsampled(gate_args, tmp_path):
    response = Response(encoded_records([payload(0)]))
    a, release, meta = acquisition_args(gate_args, tmp_path, response); M.acquire(a, release, meta, Requests(response), reader.crc32c)
    a.input_dir = a.output_dir; a.acquisition_report = a.input_dir / 'acquisition_report.json'; a.output_dir = tmp_path / 'numeric'
    a.reader = ROOT / 'outputs/AdaptGNS/research/prepare_full_waterdrop.py'
    with pytest.raises(ValueError, match='Observed test structure'): M.convert(a, release, meta)
    assert (a.output_dir / '.test.staging').exists() and not (a.output_dir / 'test.json').exists()
    assert M.read(a.output_dir / 'structural_report.json')['status'] == 'failed'


def synthetic_prior_split(root, split, count, shift):
    records = []
    for i in range(count):
        values = {'positions': np.array([[[shift + i / 10000, .4]]], dtype='<f4'), 'particle_types': np.array([7], dtype='<i8')}
        record = {'id': f'{split}:{i:06d}', 'source_index': i}
        for key, value in values.items():
            path = root / split / f'{key}_{i:06d}.npy'; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, value, allow_pickle=False)
            record[key] = {'path': str(path.relative_to(root)), 'shape': list(value.shape), 'dtype': value.dtype.str,
                           'size_bytes': path.stat().st_size, 'sha256': M.sha(path)}
        records.append(record)
    return put(root / f'{split}.json', {'dataset': 'Goop', 'split': split, 'record_count': count, 'records': records})


def test_all_split_census_candidate_and_frozen_evidence_preflight(converted, tmp_path, monkeypatch):
    a, release, meta = converted; numeric = a.output_dir
    train = synthetic_prior_split(tmp_path / 'prior', 'train', 1000, .1)
    valid = synthetic_prior_split(tmp_path / 'prior', 'valid', 30, .6)
    original_train, original_valid = M.TRAIN_MANIFEST_SHA, M.VALID_MANIFEST_SHA
    monkeypatch.setattr(M, 'TRAIN_MANIFEST_SHA', M.sha(train)); monkeypatch.setattr(M, 'VALID_MANIFEST_SHA', M.sha(valid))
    a.numeric_root = numeric; a.train_manifest = train; a.valid_manifest = valid; a.output_dir = tmp_path / 'evidence'
    M.census(a, release, meta)
    candidate_path = a.output_dir / 'final_test_admission.candidate.json'; candidate = M.read(candidate_path)
    assert candidate['status'] == 'prepared_requires_root_review'
    assert M.read(a.output_dir / 'cross_split_audit.json')['record_counts'] == {'train': 1000, 'valid': 30, 'test': 30}
    # The frozen preflight consumes a root audit with real fixed prior manifest
    # pins; this synthetic receipt only exercises its interface, never prior data.
    cross = a.output_dir / 'cross_split_audit.json'; value = M.read(cross)
    value['manifest_sha256'].update(train=original_train, valid=original_valid); put(cross, value)
    candidate.update(status='admitted_for_final_evaluation', cross_split_audit_sha256=M.sha(cross))
    a.admission = put(tmp_path / 'root_final_admission.json', candidate)
    a.context_semantics = HERE / 'goop_context_semantics_review.json'
    a.auxiliary_report = a.output_dir / 'auxiliary_report.json'; a.cross_split_audit = cross; a.output = tmp_path / 'preflight.json'
    M.preflight(a, release, meta)
    assert M.read(a.output)['status'] == 'frozen_split_and_evidence_contract_passed'
    assert not M.read(a.output)['test_evaluation_executed']
