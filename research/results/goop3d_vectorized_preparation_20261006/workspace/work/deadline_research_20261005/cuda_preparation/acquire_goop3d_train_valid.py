#!/usr/bin/env python3
"""Acquire the three pinned Goop-3D train/valid objects; no test access or resume.

Separate adaptation of reviewed acquire_goop_train_valid.py. Description only by
default. Keep partial bytes and failure receipts; root owns execution/inventory.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

BASE = 'https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Goop-3D/'
SOURCES = {
    'metadata.json': (471, '1599153912104523', 'odsy4Q=='),
    'train.tfrecord': (27448425232, '1599159646798459', 'gBeM1w=='),
    'valid.tfrecord': (2857385340, '1599154482348408', 'rw9odw=='),
}
META_SHA = '727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def interrupted(signum, _frame):
    raise InterruptedError('Acquisition interrupted by signal ' + str(signum))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'execute': False, 'dataset': 'Goop-3D', 'sources': SOURCES,
                          'test_access': False, 'scientific_training_admission': False}))
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required')
    import requests
    import crc32c
    if crc32c.crc32c(b'123456789') != 0xe3069283:
        raise ValueError('CRC32C implementation self-test failed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'schema': 'official_goop3d_train_valid_acquisition_v1',
              'status': 'running', 'dataset': 'Goop-3D', 'source_family': 'official_gns_tfrecord',
              'files': [], 'test_accessed': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scientific_training_admission': False}
    report_path = args.output_dir / 'acquisition_report.json'
    current = None
    old_handlers = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name, (size, generation, expected_crc) in SOURCES.items():
            url = BASE + name
            current = {'name': name, 'saved_name': name, 'url': url,
                       'requested_generation': generation, 'status': 'started', 'received_bytes': 0}
            report['files'].append(current)
            write(report_path, report)
            with requests.get(url, params={'generation': generation},
                              headers={'Accept-Encoding': 'identity'}, stream=True,
                              timeout=(20, 120), allow_redirects=False) as response:
                current['response_status'] = response.status_code
                current['response_headers'] = dict(response.headers)
                write(report_path, report)
                if response.status_code != 200:
                    raise ValueError('Expected 200 without redirect: ' + name)
                if (response.headers.get('x-goog-generation') != generation
                        or int(response.headers.get('Content-Length', '-1')) != size
                        or response.headers.get('Content-Encoding', 'identity') != 'identity'):
                    raise ValueError('Generation, size or encoding mismatch: ' + name)
                server_hashes = {part.strip().split('=', 1)[0]: part.strip().split('=', 1)[1]
                                 for part in response.headers.get('x-goog-hash', '').split(',') if '=' in part}
                if server_hashes.get('crc32c') != expected_crc:
                    raise ValueError('Publisher CRC mismatch: ' + name)
                digest, crc, count = hashlib.sha256(), 0, 0
                partial = args.output_dir / (name + '.partial')
                last_progress = time.perf_counter()
                with partial.open('xb') as stream:
                    for block in response.iter_content(chunk_size=1 << 20):
                        if not block:
                            continue
                        stream.write(block)
                        digest.update(block)
                        crc = crc32c.crc32c(block, crc)
                        count += len(block)
                        current['received_bytes'] = count
                        if count > size:
                            raise ValueError('Received more than pinned bytes: ' + name)
                        if time.perf_counter() - last_progress >= 10:
                            report['elapsed_seconds'] = time.perf_counter() - started
                            write(report_path, report)
                            last_progress = time.perf_counter()
                    stream.flush()
                    os.fsync(stream.fileno())
                encoded = base64.b64encode(crc.to_bytes(4, 'big')).decode()
                current.update(generation=generation, sha256=digest.hexdigest(),
                               crc32c_base64=encoded, crc32c_verified=encoded == expected_crc)
                if (count != size or encoded != expected_crc
                        or (name == 'metadata.json' and digest.hexdigest() != META_SHA)):
                    raise ValueError('Complete source checksum mismatch: ' + name)
                final = args.output_dir / name
                if final.exists():
                    raise FileExistsError(final)
                partial.rename(final)
                current['status'] = 'complete'
                write(report_path, report)
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != report['source_sha256']:
            raise ValueError('Acquisition source changed during execution')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        if current is not None and current['status'] != 'complete':
            current['status'] = 'failed'
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        report['elapsed_seconds'] = time.perf_counter() - started
        report['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(report_path, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
