"""Bounded first training record inspection only; never full-data admission."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import struct
import time
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'goop3d_first_record_20261006_v1'
MAX_RECORD_BYTES = 64 * 1024 * 1024
READER = HERE.parents[1] / 'outputs/AdaptGNS/research/prepare_full_waterdrop.py'
READER_SHA = 'ab077f11240a7296a2a15679ae8a57032c91280fdc94c23b63155e86793d2f33'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


inventory = HERE / 'extension_feasibility_20261006/public_metadata_inventory.json'
source = next(row for row in json.loads(inventory.read_text())['requests']
              if row['dataset'] == 'Goop-3D' and row['resource'] == 'train.tfrecord')
headers = {key.lower(): value for key, value in source['headers'].items()}
generation, total = headers['x-goog-generation'], int(headers['content-length'])
require(source['url'] == 'https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop-3D/train.tfrecord', 'Unexpected source')
require(sha(READER) == READER_SHA, 'Reader differs')
spec = importlib.util.spec_from_file_location('_crc_reader', READER)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
OUT.mkdir(exist_ok=False)
report = {'schema': 'goop3d_first_training_record_inspection_v1', 'status': 'running',
          'scope': 'First training record by source order only; not a full dataset, representative sample, model admission or timing benchmark',
          'test_accessed': False, 'whole_object_checksum_verified': False,
          'scientific_training_admitted': False, 'source_url': source['url'],
          'generation': generation, 'object_bytes': total, 'requests': [],
          'source_sha256': sha(Path(__file__)), 'reader_sha256': READER_SHA,
          'input_inventory_sha256': sha(inventory), 'max_record_bytes': MAX_RECORD_BYTES,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def fetch(start, end, name):
    expected = end - start + 1
    with requests.get(source['url'], params={'generation': generation},
                      headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                      stream=True, timeout=(20, 90), allow_redirects=False) as response:
        require(response.status_code == 206, 'Range response must be206; refusing whole object')
        require(response.headers.get('Content-Range') == f'bytes {start}-{end}/{total}', 'Range identity differs')
        require(response.headers.get('x-goog-generation') == generation, 'Generation differs')
        require(int(response.headers.get('Content-Length', '-1')) == expected, 'Range length differs')
        require(response.headers.get('Content-Encoding', 'identity') == 'identity', 'Encoded response refused')
        path = OUT / name
        count = 0
        with path.open('xb') as target:
            for block in response.iter_content(1 << 20):
                count += len(block)
                require(count <= expected, 'Range exceeded requested bytes')
                target.write(block)
        require(count == expected, 'Truncated range')
        report['requests'].append({'range': [start, end], 'received_bytes': count, 'file': name, 'sha256': sha(path)})
        return path.read_bytes()


started = time.perf_counter()
try:
    header = fetch(0, 11, 'record0_header.bin')
    require(struct.unpack('<I', header[8:])[0] == reader.masked_crc(header[:8]), 'Length CRC mismatch')
    length = struct.unpack('<Q', header[:8])[0]
    require(0 < length <= MAX_RECORD_BYTES and length + 16 <= total, 'Record exceeds declared guard')
    tail = fetch(12, 12 + length + 3, 'record0_payload_and_crc.bin')
    require(struct.unpack('<I', tail[-4:])[0] == reader.masked_crc(tail[:-4]), 'Payload CRC mismatch')
    staging = OUT / 'numeric'
    staging.mkdir()
    record = reader.decode_record(tail[:-4], 0, 0, 'train', staging, 3, 301)
    report.update(status='first_training_record_verified', record=record,
                  length_crc_verified=True, payload_crc_verified=True,
                  stored_frames=record['positions']['shape'][0],
                  particles=record['positions']['shape'][1],
                  dimension=record['positions']['shape'][2],
                  declared_full_stored_horizon=record['positions']['shape'][0] - 6)
except BaseException as error:
    report.update(status='failed_inspection', error_type=type(error).__name__, error=str(error))
    raise
finally:
    report['elapsed_seconds'] = time.perf_counter() - started
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
