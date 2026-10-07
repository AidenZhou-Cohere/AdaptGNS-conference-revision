"""Bounded opaque route for one distinct D3 observed-history phase.

Reuses reviewed nonblocking duplex transport; grants no clock or execution.
The route ends at the D3 phase start plus3540seconds. No Sand closure gate.
"""
import argparse
import datetime
import json
from pathlib import Path
import socket
import threading
import time

from bounded_relay import relay, BUFFER_CAP
from phase_binding import load_original_phase, need

AUTHORITY = b'coder.internal.cohere.com:443'
UPSTREAM = ('100.106.33.61', 443)
MAX_HEADER = 8192
CONNECTION_CAP = 2048
CONCURRENCY_CAP = 8
SCHEMA = 'coder_goop3d_observed_phase_route_proxy_v1'


def parse_request(header):
    need(len(header) <= MAX_HEADER and header.endswith(b'\r\n\r\n'), 'bounded complete CONNECT header required')
    fields = header.split(b'\r\n', 1)[0].split(b' ')
    need(len(fields) == 3 and fields[0] == b'CONNECT' and fields[1] == AUTHORITY
         and fields[2] in (b'HTTP/1.0', b'HTTP/1.1'), 'only exact original CONNECT authority permitted')




def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def serve(output, phase):
    seconds = phase.remaining()
    need(output.is_absolute() and output.resolve() == output and not output.exists(), 'fresh canonical output required')
    output.mkdir(exist_ok=False)
    started = now()
    wall_start, mono_start = time.time(), time.monotonic()
    stop = threading.Event()
    lock = threading.Lock()
    sockets = set()
    workers = []
    counts = dict(accepted=0, routed=0, cap_rejected=0, concurrency_rejected=0,
                  nonloopback_rejected=0, handler_errors=0, drained=0, expired=0)

    def expired():
        wall, mono = time.time() - wall_start, time.monotonic() - mono_start
        return stop.is_set() or phase.expired() or wall >= seconds or mono >= seconds or abs(wall - mono) > 2

    def add_count(name):
        with lock:
            counts[name] += 1

    def remember(sock):
        with lock:
            sockets.add(sock)

    def close(sock):
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
        with lock:
            sockets.discard(sock)

    def handle(client):
        upstream = None
        try:
            client.settimeout(min(2, phase.remaining()))
            data = b''
            while b'\r\n\r\n' not in data:
                need(not expired() and len(data) < MAX_HEADER, 'request deadline/header limit')
                chunk = client.recv(min(1024, MAX_HEADER - len(data)))
                need(bool(chunk), 'closed CONNECT request')
                data += chunk
            header, extra = data.split(b'\r\n\r\n', 1)
            parse_request(header + b'\r\n\r\n')
            need(not expired(), 'expired before connect')
            upstream = socket.create_connection(UPSTREAM, timeout=min(3, phase.remaining()))
            remember(upstream)
            need(not expired(), 'expired during connect')
            client.settimeout(min(1, phase.remaining()))
            client.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            add_count('routed')
            outcome = relay(client, upstream, expired, initial=extra)
            add_count('drained' if outcome == 'both_directions_drained' else 'expired')
        except (OSError, ValueError):
            add_count('handler_errors')
        finally:
            close(upstream)
            close(client)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', 0))
        listener.listen(4)
        listener.settimeout(0.2)
        ready = dict(schema=SCHEMA, **phase.metadata, started_utc=started, lifetime_seconds=seconds,
                     url='http://127.0.0.1:' + str(listener.getsockname()[1]),
                     authority=AUTHORITY.decode(), upstream=list(UPSTREAM), opaque_tls=True,
                     no_settings_changed=True,
                     connection_cap=CONNECTION_CAP, concurrency_cap=CONCURRENCY_CAP,
                     per_direction_buffer_bytes=BUFFER_CAP, payload_backpressure='nonblocking_partial_send')
        with (output / 'ready.json').open('x') as stream:
            json.dump(ready, stream, indent=2); stream.write('\n')
        print(json.dumps(ready), flush=True)
        while not expired():
            try:
                client, address = listener.accept()
            except socket.timeout:
                continue
            remember(client)
            if address[0] != '127.0.0.1':
                add_count('nonloopback_rejected'); close(client); continue
            if counts['accepted'] >= CONNECTION_CAP:
                add_count('cap_rejected'); close(client); continue
            add_count('accepted')
            workers = [worker for worker in workers if worker.is_alive()]
            if len(workers) >= CONCURRENCY_CAP:
                add_count('concurrency_rejected'); close(client); continue
            worker = threading.Thread(target=handle, args=(client,), daemon=True)
            workers.append(worker)
            worker.start()
    finally:
        stop.set()
        listener.close()
        with lock:
            remaining = list(sockets)
        for sock in remaining:
            close(sock)
        for worker in workers:
            worker.join(timeout=0.1)
        terminal = dict(**phase.metadata, started_utc=started, ended_utc=now(), lifetime_seconds=seconds,
                        counts=counts, remaining_worker_threads=sum(w.is_alive() for w in workers),
                        wall_elapsed_seconds=time.time()-wall_start, monotonic_elapsed_seconds=time.monotonic()-mono_start,
                        payloads_logged=False, remote_processes_signaled=False,
                        connection_cap=CONNECTION_CAP,
                        concurrency_cap=CONCURRENCY_CAP, per_direction_buffer_bytes=BUFFER_CAP)
        with (output / 'terminal.json').open('x') as stream:
            json.dump(terminal, stream, indent=2); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root-serve', action='store_true')
    parser.add_argument('--phase-sha256'); parser.add_argument('--anchor-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.root_serve:
        print(json.dumps(dict(status='inert_observed_history_phase_route', new_clock_granted=False)))
        return
    phase = load_original_phase(args.phase_sha256, args.anchor_sha256)
    need(args.output is not None, 'explicit fresh output required')
    phase.remaining()
    serve(args.output, phase)


if __name__ == '__main__':
    main()
