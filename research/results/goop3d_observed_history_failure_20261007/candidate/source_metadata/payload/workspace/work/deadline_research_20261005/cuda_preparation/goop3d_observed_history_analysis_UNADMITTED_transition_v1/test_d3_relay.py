"""Synthetic sockets/readiness only: no network, real phase, clocks or processes."""
import json
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bounded_relay import relay, BUFFER_CAP
import phase_proxy as proxy


class FakeSocket:
    def __init__(self, incoming=b'', chunk=65536, write_chunk=32768, blocked_writes=0):
        self.incoming = incoming
        self.offset = 0
        self.chunk = chunk
        self.write_chunk = write_chunk
        self.blocked_writes = blocked_writes
        self.sent = bytearray()
        self.max_pending = 0
        self.shutdowns = []
        self.nonblocking = False

    def setblocking(self, value):
        self.nonblocking = not value

    def recv(self, size):
        end = min(len(self.incoming), self.offset + min(size, self.chunk))
        result = self.incoming[self.offset:end]
        self.offset = end
        return result

    def send(self, value):
        self.max_pending = max(self.max_pending, len(value))
        if self.blocked_writes:
            self.blocked_writes -= 1
            raise BlockingIOError()
        count = min(len(value), self.write_chunk)
        self.sent.extend(value[:count])
        return count

    def shutdown(self, mode):
        self.shutdowns.append((mode, len(self.sent)))


class Ready:
    def __init__(self):
        self.calls = 0

    def __call__(self, reads, writes, errors, timeout):
        self.calls += 1
        if self.calls > 200000:
            raise AssertionError('synthetic relay failed to drain')
        return reads, writes, []


class RelayTests(unittest.TestCase):
    def test_large_bidirectional_partial_send_and_backpressure(self):
        forward = bytes(range(256)) * 8192
        backward = b'answer' * 170000
        client = FakeSocket(forward, chunk=7111, write_chunk=127, blocked_writes=800)
        upstream = FakeSocket(backward, chunk=8177, write_chunk=509, blocked_writes=1000)
        result = relay(client, upstream, lambda: False, select_ready=Ready())
        self.assertEqual(result, 'both_directions_drained')
        self.assertEqual(upstream.sent, forward)
        self.assertEqual(client.sent, backward)
        self.assertLessEqual(max(client.max_pending, upstream.max_pending), BUFFER_CAP)
        self.assertTrue(client.nonblocking and upstream.nonblocking)
        self.assertEqual(client.shutdowns, [(socket.SHUT_WR, len(backward))])
        self.assertEqual(upstream.shutdowns, [(socket.SHUT_WR, len(forward))])

    def test_initial_connect_tail_preserved(self):
        client, upstream = FakeSocket(b'later'), FakeSocket()
        relay(client, upstream, lambda: False, initial=b'early', select_ready=Ready())
        self.assertEqual(upstream.sent, b'earlylater')

    def test_original_deadline_interrupts_stalled_payload(self):
        client = FakeSocket(b'x' * (BUFFER_CAP * 3))
        upstream = FakeSocket(blocked_writes=10000000)
        ready = Ready()
        result = relay(client, upstream, lambda: ready.calls >= 20, select_ready=ready)
        self.assertEqual(result, 'original_phase_expired')
        self.assertEqual(ready.calls, 20)
        self.assertLessEqual(client.offset, BUFFER_CAP)

    def test_initially_expired_does_not_transfer(self):
        client, upstream = FakeSocket(b'forbidden'), FakeSocket()
        self.assertEqual(relay(client, upstream, lambda: True, select_ready=Ready()), 'original_phase_expired')
        self.assertEqual(client.offset, 0)
        self.assertEqual(upstream.sent, b'')

    def test_oversized_connect_tail_fails(self):
        with self.assertRaises(ValueError):
            relay(FakeSocket(), FakeSocket(), lambda: False, initial=b'x' * (BUFFER_CAP + 1))

    def test_zero_progress_write_fails(self):
        with self.assertRaises(OSError):
            relay(FakeSocket(b'a'), FakeSocket(write_chunk=0), lambda: False, select_ready=Ready())

    def test_exact_authority_only(self):
        proxy.parse_request(b'CONNECT coder.internal.cohere.com:443 HTTP/1.1\r\n\r\n')
        for header in (b'GET coder.internal.cohere.com:443 HTTP/1.1\r\n\r\n',
                       b'CONNECT other.invalid:443 HTTP/1.1\r\n\r\n',
                       b'CONNECT coder.internal.cohere.com:80 HTTP/1.1\r\n\r\n',
                       b'CONNECT coder.internal.cohere.com:443 HTTP/2\r\n\r\n'):
            with self.assertRaises(ValueError):
                proxy.parse_request(header)



if __name__ == '__main__':
    unittest.main()
