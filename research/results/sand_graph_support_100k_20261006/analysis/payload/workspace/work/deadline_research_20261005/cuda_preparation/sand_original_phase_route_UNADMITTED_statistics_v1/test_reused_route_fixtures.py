"""Independent deterministic proxy fixtures: fake sockets, clocks and workers only."""
from pathlib import Path
from unittest.mock import patch
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest

SUT = None

class FakePhase:
    metadata = {'synthetic_phase': True}
    def remaining(self):
        remaining = 60. - SUT.time.time()
        if remaining <= 0: raise ValueError('synthetic original phase expired')
        return remaining
    def expired(self):
        try: self.remaining(); return False
        except ValueError: return True

PAYLOAD = b'TLS_PAYLOAD_DO_NOT_LOG'
REPLY = b'TLS_REPLY_DO_NOT_LOG'
HEADER = b'CONNECT coder.internal.cohere.com:443 HTTP/1.1\r\nHost: coder.internal.cohere.com:443\r\n\r\n'


class Clock:
    wall = 0.
    mono = 0.

    def advance(self, value=61., divergence=False):
        self.wall = value
        if not divergence:
            self.mono = value


class FakeSocket:
    def __init__(self, clock, data=b'', after_recv=None, after_send=None):
        self.clock, self.data = clock, data
        self.after_recv, self.after_send = after_recv, after_send
        self.sent, self.timeouts, self.closed = [], [], False

    def recv(self, limit):
        value, self.data = self.data[:limit], self.data[limit:]
        if self.after_recv:
            self.after_recv(value)
        return value

    def sendall(self, value):
        self.sent.append((self.clock.wall, value))
        if self.after_send:
            self.after_send(value)

    def settimeout(self, value):
        self.timeouts.append(value)

    def shutdown(self, how):
        self.closed = True

    def close(self):
        self.closed = True


class FakeListener(FakeSocket):
    def __init__(self, clock, clients, peer, interrupted=False):
        super().__init__(clock)
        self.clients, self.peer, self.interrupted = list(clients), peer, interrupted
        self.bound = None

    def bind(self, address):
        self.bound = address

    def listen(self, backlog):
        self.backlog = backlog

    def getsockname(self):
        return ('127.0.0.1', 34567)

    def accept(self):
        if self.clients:
            return self.clients.pop(0), (self.peer, 12345)
        if self.interrupted:
            raise KeyboardInterrupt()
        self.clock.advance()
        raise SUT.socket.timeout()


class InlineWorker:
    def __init__(self, target, args, daemon):
        assert daemon is True
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)

    def is_alive(self):
        return False

    def join(self, timeout):
        pass


def exercise(request=HEADER + PAYLOAD, scenario='normal', count=1, peer='127.0.0.1', interrupted=False):
    clock = Clock()
    if scenario in ('select_expire', 'select_diverge', 'recv_expire'):
        request = HEADER
    clients = [FakeSocket(clock, request) for _ in range(count)]
    if scenario == 'initial_extra_expire':
        clients[0].after_send = lambda value: clock.advance() if value.startswith(b'HTTP/') else None
    if scenario == 'request_expire':
        clients[0].after_recv = lambda value: clock.advance()
    listener = FakeListener(clock, clients, peer, interrupted)
    upstreams, connections, select_calls = [], [], []

    def connect(address, timeout):
        assert address == ('100.106.33.61', 443)
        assert timeout == 3
        connections.append((address, timeout))
        if scenario == 'connect_error':
            raise OSError('synthetic connection refusal')
        upstream = FakeSocket(clock, REPLY)
        upstreams.append(upstream)
        if scenario == 'connect_expire':
            clock.advance()
        return upstream

    def select(read, write, error, timeout):
        assert write == error == [] and timeout == .2
        client, upstream = read
        select_calls.append(True)
        if scenario in ('select_expire', 'select_diverge', 'recv_expire'):
            client.data = PAYLOAD
            if scenario == 'recv_expire':
                client.after_recv = lambda value: clock.advance()
            else:
                clock.advance(61. if scenario == 'select_expire' else 3., scenario == 'select_diverge')
            return [client], [], []
        return ([upstream] if upstream.data else [client]), [], []

    stdout = io.StringIO()
    with tempfile.TemporaryDirectory(prefix='coder-route-fixture-') as temp:
        output = Path(temp).resolve() / 'new'
        with patch.object(SUT.socket, 'socket', return_value=listener), \
             patch.object(SUT.socket, 'create_connection', side_effect=connect), \
             patch.object(SUT.select, 'select', side_effect=select), \
             patch.object(SUT.threading, 'Thread', InlineWorker), \
             patch.object(SUT.time, 'time', side_effect=lambda: clock.wall), \
             patch.object(SUT.time, 'monotonic', side_effect=lambda: clock.mono), \
             patch.object(SUT, 'now', return_value='synthetic-time'), \
             contextlib.redirect_stdout(stdout):
            try:
                SUT.serve(output, FakePhase())
            except KeyboardInterrupt:
                assert interrupted
        terminal = json.loads((output / 'terminal.json').read_text())
        ready = json.loads((output / 'ready.json').read_text())
        output_text = ''.join(p.read_text() for p in output.iterdir()) + stdout.getvalue()
    return dict(clock=clock, clients=clients, listener=listener, upstreams=upstreams,
                connections=connections, ready=ready, terminal=terminal, output_text=output_text)


class ProxyFixtures(unittest.TestCase):
    def test_exact_connect_versions(self):
        for version in [b'HTTP/1.0', b'HTTP/1.1']:
            with self.subTest(version=version):
                SUT.parse_request(b'CONNECT ' + SUT.AUTHORITY + b' ' + version + b'\r\n\r\n')

    def test_refuse_other_routes_and_malformed_request(self):
        invalid = [
            b'GET / HTTP/1.1\r\n\r\n',
            b'CONNECT 100.106.33.61:443 HTTP/1.1\r\n\r\n',
            b'CONNECT coder.tail5566.ts.net:443 HTTP/1.1\r\n\r\n',
            b'CONNECT coder.internal.cohere.com:80 HTTP/1.1\r\n\r\n',
            b'CONNECT evil.example:443 HTTP/1.1\r\n\r\n',
            b'CONNECT coder.internal.cohere.com:443@evil:443 HTTP/1.1\r\n\r\n',
            b'CONNECT coder.internal.cohere.com:443 HTTP/2\r\n\r\n',
            b'CONNECT  coder.internal.cohere.com:443 HTTP/1.1\r\n\r\n',
            HEADER[:-1], HEADER + b'A' * 8192,
        ]
        for value in invalid:
            with self.subTest(request_number=invalid.index(value)):
                with self.assertRaises(ValueError):
                    SUT.parse_request(value)

    def test_only_fixed_upstream_opaque_relay_and_clean_metadata(self):
        r = exercise()
        self.assertEqual(r['listener'].bound, ('127.0.0.1', 0))
        self.assertEqual(r['connections'], [(('100.106.33.61', 443), 3)])
        self.assertIn(PAYLOAD, [v for _, v in r['upstreams'][0].sent])
        self.assertIn(REPLY, [v for _, v in r['clients'][0].sent])
        self.assertNotIn(PAYLOAD.decode(), r['output_text'])
        self.assertNotIn(REPLY.decode(), r['output_text'])
        self.assertTrue(all(s.closed for s in [r['listener'], *r['clients'], *r['upstreams']]))
        self.assertEqual(r['terminal']['remaining_worker_threads'], 0)

    def test_bad_peer_and_oversize_header_never_connect(self):
        for arguments in [{'peer': '127.0.0.2'}, {'request': b'A' * 9000}]:
            with self.subTest(arguments=list(arguments)):
                r = exercise(**arguments)
                self.assertFalse(r['connections'])
                self.assertTrue(r['clients'][0].closed)

    def test_connection_cap(self):
        r = exercise(count=129)
        self.assertEqual(len(r['connections']), 128)
        self.assertEqual(r['terminal']['counts']['accepted'], 128)
        self.assertTrue(all(s.closed for s in r['clients']))

    def test_request_or_connect_expiry_never_relays(self):
        for scenario in ['request_expire', 'connect_expire', 'connect_error']:
            with self.subTest(scenario=scenario):
                r = exercise(scenario=scenario)
                self.assertFalse([v for s in r['upstreams'] for _, v in s.sent])
                self.assertTrue(all(s.closed for s in [*r['clients'], *r['upstreams']]))

    def test_stop_or_clock_divergence_during_io_never_forwards(self):
        for scenario in ['select_expire', 'select_diverge', 'recv_expire', 'initial_extra_expire']:
            with self.subTest(scenario=scenario):
                r = exercise(scenario=scenario)
                self.assertFalse([v for s in r['upstreams'] for _, v in s.sent], scenario)

    def test_listener_interrupt_still_closes_and_records(self):
        r = exercise(interrupted=True)
        self.assertTrue(all(s.closed for s in [r['listener'], *r['clients'], *r['upstreams']]))
        self.assertFalse(r['terminal']['payloads_logged'])

    def test_expired_phase_rejects_before_output(self):
        with tempfile.TemporaryDirectory(prefix='coder-phase-fixture-') as temp:
            output = Path(temp).resolve() / 'absent'
            phase = FakePhase()
            with patch.object(phase,'remaining',side_effect=ValueError('expired')):
                with self.assertRaises(ValueError): SUT.serve(output,phase)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    before = args.source.read_bytes()
    spec = importlib.util.spec_from_file_location('coder_route_fixture_target', args.source)
    SUT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SUT)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProxyFixtures))
    assert args.source.read_bytes() == before
    receipt = {'source_sha256': hashlib.sha256(before).hexdigest(), 'tests_run': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors), 'passed': result.wasSuccessful(),
               'all_sockets_threads_and_clocks_faked': True, 'real_upstream_connections': False,
               'no_credentials_settings_models_or_remote_actions': True}
    with args.receipt.open('x') as f:
        f.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
