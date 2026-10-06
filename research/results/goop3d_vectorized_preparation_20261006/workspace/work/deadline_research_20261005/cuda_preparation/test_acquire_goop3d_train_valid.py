"""Small mocked HTTP tests; never contact or acquire an official source."""
import base64
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import crc32c
from requests.structures import CaseInsensitiveDict
import acquire_goop3d_train_valid as acquire


class Response:
    def __init__(self, body, expected, status=200, fail_after_first=False):
        self.body = body
        self.status_code = status
        self.fail_after_first = fail_after_first
        size, generation, crc = expected
        self.headers = CaseInsensitiveDict({'Content-Length': str(size),
                                            'x-goog-generation': generation,
                                            'x-goog-hash': 'crc32c=' + crc})

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, chunk_size):
        yield self.body[:3]
        if self.fail_after_first:
            acquire.interrupted(signal.SIGTERM, None)
        yield self.body[3:]


class Goop3DAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='goop3d_acquisition_mock_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'fresh'
        self.bodies = {name: (name + ':synthetic bytes').encode() for name in acquire.SOURCES}
        self.sources = {name: (len(body), acquire.SOURCES[name][1],
                              base64.b64encode(crc32c.crc32c(body).to_bytes(4, 'big')).decode())
                        for name, body in self.bodies.items()}
        self.responses = [Response(self.bodies[name], expected) for name, expected in self.sources.items()]
        self.source_patch = mock.patch.multiple(acquire, SOURCES=self.sources,
                           META_SHA=hashlib.sha256(self.bodies['metadata.json']).hexdigest())

    def execute(self):
        with self.source_patch, mock.patch('requests.get', side_effect=self.responses) as get:
            value = acquire.main(['--execute', '--output-dir', str(self.output)])
        return value, get

    def receipt(self):
        return json.loads((self.output / 'acquisition_report.json').read_text())

    def test_default_description_performs_no_request(self):
        with mock.patch('requests.get', side_effect=AssertionError('no request')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(acquire.main([]), 0)
        value = json.loads(output.getvalue())
        self.assertFalse(value['execute'])
        self.assertFalse(value['test_access'])
        self.assertEqual(set(value['sources']), {'metadata.json', 'train.tfrecord', 'valid.tfrecord'})

    def test_exact_three_pinned_gets_and_stream_checks(self):
        value, get = self.execute()
        self.assertEqual(value, 0)
        self.assertEqual(get.call_count, 3)
        report = self.receipt()
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(report['scientific_training_admission'])
        self.assertFalse(report['test_accessed'])
        self.assertEqual(report['schema'], 'official_goop3d_train_valid_acquisition_v1')
        for call, row in zip(get.call_args_list, report['files']):
            name = row['name']
            self.assertEqual(call.args, (acquire.BASE + name,))
            self.assertEqual(call.kwargs['params'], {'generation': self.sources[name][1]})
            self.assertFalse(call.kwargs['allow_redirects'])
            self.assertEqual(call.kwargs['headers'], {'Accept-Encoding': 'identity'})
            self.assertEqual((self.output / name).read_bytes(), self.bodies[name])
            self.assertEqual(row['received_bytes'], len(self.bodies[name]))
            self.assertEqual(row['sha256'], hashlib.sha256(self.bodies[name]).hexdigest())
            self.assertTrue(row['crc32c_verified'])
            self.assertEqual(row['response_status'], 200)
            self.assertFalse((self.output / (name + '.partial')).exists())

    def test_failed_status_retains_response_evidence(self):
        self.responses[0].status_code = 503
        with self.assertRaisesRegex(ValueError, 'Expected 200'):
            self.execute()
        report = self.receipt()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['files'][0]['response_status'], 503)
        self.assertIn('x-goog-hash', report['files'][0]['response_headers'])

    def test_header_identity_failures_do_not_publish(self):
        for field, value in [('x-goog-generation', 'wrong'), ('Content-Length', '1'),
                             ('Content-Encoding', 'gzip'), ('x-goog-hash', 'crc32c=wrong')]:
            with self.subTest(field=field):
                self.output = self.root / field
                responses = copy.deepcopy(self.responses)
                responses[0].headers[field] = value
                with self.source_patch, mock.patch('requests.get', side_effect=responses):
                    with self.assertRaises(ValueError):
                        acquire.main(['--execute', '--output-dir', str(self.output)])
                report = self.receipt()
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['files'][0]['response_headers'][field], value)
                self.assertFalse((self.output / 'metadata.json').exists())

    def test_signal_failure_preserves_partial_and_received_count(self):
        before = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}
        self.responses[1].fail_after_first = True
        with self.assertRaisesRegex(InterruptedError, 'signal'):
            self.execute()
        report = self.receipt()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['files'][0]['status'], 'complete')
        self.assertEqual(report['files'][1]['received_bytes'], 3)
        self.assertEqual((self.output / 'train.tfrecord.partial').read_bytes(), self.bodies['train.tfrecord'][:3])
        self.assertEqual(before, {value: signal.getsignal(value) for value in before})
        self.assertFalse((self.output / 'train.tfrecord').exists())

    def test_corrupt_body_fails_whole_checksum_with_bytes_retained(self):
        self.responses[1].body = b'X' + self.responses[1].body[1:]
        with self.assertRaisesRegex(ValueError, 'Complete source checksum'):
            self.execute()
        report = self.receipt()
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['files'][1]['crc32c_verified'])
        self.assertEqual((self.output / 'train.tfrecord.partial').read_bytes(), self.responses[1].body)

    def test_fresh_output_refused_before_request(self):
        self.output.mkdir()
        marker = self.output / 'preserve'
        marker.write_text('unchanged')
        with mock.patch('requests.get', side_effect=AssertionError('no request')):
            with self.assertRaises(FileExistsError):
                acquire.main(['--execute', '--output-dir', str(self.output)])
        self.assertEqual(marker.read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main(verbosity=2)
