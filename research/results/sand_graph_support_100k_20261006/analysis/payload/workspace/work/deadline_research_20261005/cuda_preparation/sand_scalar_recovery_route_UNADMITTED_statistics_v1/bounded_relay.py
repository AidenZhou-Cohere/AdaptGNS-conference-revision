"""Bounded opaque duplex forwarding; pure injected readiness for fake tests."""
import select
import socket

BUFFER_CAP = 256 * 1024
CHUNK = 65536


def relay(client, upstream, expired, initial=b'', select_ready=select.select):
    """Forward partial writes, propagate EOF after drain, stop at original phase.

    Each direction has at most BUFFER_CAP pending bytes. No content is logged.
    Backpressure pauses reads and cannot cause a one-second sendall timeout.
    The caller closes both sockets after this function returns or raises.
    """
    if len(initial) > BUFFER_CAP:
        raise ValueError('initial payload exceeds fixed relay buffer')
    peers = {client: upstream, upstream: client}
    pending = {client: bytearray(initial), upstream: bytearray()}
    ended = {client: False, upstream: False}
    shutdown = {client: False, upstream: False}
    for sock in peers:
        sock.setblocking(False)
    while not expired():
        for source, target in peers.items():
            if ended[source] and not pending[source] and not shutdown[target]:
                try:
                    target.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                shutdown[target] = True
        if all(ended.values()) and not any(pending.values()):
            return 'both_directions_drained'
        reads = [s for s in peers if not ended[s] and len(pending[s]) < BUFFER_CAP]
        writes = [peers[s] for s in peers if pending[s]]
        readable, writable, _ = select_ready(reads, writes, [], 0.2)
        if expired():
            return 'original_phase_expired'
        for target in writable:
            if expired():
                return 'original_phase_expired'
            source = peers[target]
            data = pending[source]
            if not data:
                continue
            try:
                sent = target.send(data)
            except (BlockingIOError, InterruptedError):
                continue
            if sent <= 0:
                raise OSError('zero-progress socket write')
            del data[:sent]
        for source in readable:
            if expired():
                return 'original_phase_expired'
            try:
                block = source.recv(min(CHUNK, BUFFER_CAP - len(pending[source])))
            except (BlockingIOError, InterruptedError):
                continue
            if block:
                pending[source].extend(block)
            else:
                ended[source] = True
    return 'original_phase_expired'
