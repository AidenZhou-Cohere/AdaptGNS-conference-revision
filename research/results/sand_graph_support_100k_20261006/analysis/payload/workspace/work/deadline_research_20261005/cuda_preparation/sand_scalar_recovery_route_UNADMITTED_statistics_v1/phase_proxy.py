"""One separately reviewed recovery route inside the unchanged original Sand hour.

No subprocess, clock grant, TLS termination, remote signal or content logging.
All live use requires a root-pinned prior-route/local-transport closure admission.
"""
import argparse
import datetime
import json
from pathlib import Path
import socket
import threading
import time

from bounded_relay import relay, BUFFER_CAP
from phase_binding import load_original_phase, need, read_pinned, strict, pin

AUTHORITY = b'coder.internal.cohere.com:443'
UPSTREAM = ('100.106.33.61', 443)
MAX_HEADER = 8192
CONNECTION_CAP = 2048
CONCURRENCY_CAP = 8
SCHEMA = 'coder_sand_original_phase_recovery_route_proxy_v1'
ORIGINAL_PACKAGE_SHA = '292cc31d54e822d6eb1a843da3ce55427f9f89201c1fb9a2dd7293d0660dfd80'
ORIGINAL_PHASE_SHA = '0ac7a368dd8afa921c8b973c5a8a760af7f10eb9f3413bceb592fff0c0ad7a5c'
ORIGINAL_ANCHOR_SHA = '06cee5e7a2372fb5799678cfa9493f91a0e88fc3ba45ebc1a70bc15944295fd1'


def parse_request(header):
    need(len(header) <= MAX_HEADER and header.endswith(b'\r\n\r\n'), 'bounded complete CONNECT header required')
    fields = header.split(b'\r\n', 1)[0].split(b' ')
    need(len(fields) == 3 and fields[0] == b'CONNECT' and fields[1] == AUTHORITY
         and fields[2] in (b'HTTP/1.0', b'HTTP/1.1'), 'only exact original CONNECT authority permitted')


def closure_admission(raw, phase):
    row = strict(raw)
    need(row.get('schema') == 'sand_original_phase_route_closure_admission_v1'
         and row.get('issued_by') == 'root'
         and row.get('status') == 'original_proxy_and_failed_local_transport_groups_closed', 'root closure admission required')
    need(row.get('analysis_phase_sha256') == ORIGINAL_PHASE_SHA == phase.metadata['analysis_phase_sha256']
         and row.get('local_phase_anchor_sha256') == ORIGINAL_ANCHOR_SHA == phase.metadata['local_phase_anchor_sha256']
         and row.get('original_route_package_sha256') == ORIGINAL_PACKAGE_SHA, 'unchanged original phase/anchor/route required')
    need(row.get('original_proxy_tool_session') == 23611 and row.get('original_proxy_pid') == 46711
         and row.get('original_proxy_pid_and_group_absent') is True
         and row.get('failed_local_transport_groups_absent') == [47926, 48109], 'exact previous local identities must be closed')
    need(type(row.get('original_proxy_tool_exit_code')) is int and row.get('scientific_workers_restarted') is False
         and row.get('new_or_restarted_clock_granted') is False, 'actual prior exit, no worker restart or clock required')
    for key in ('original_proxy_tool_exit', 'original_proxy_terminal', 'local_native_closure'):
        binding = row[key]
        need(type(binding) is dict and pin(binding.get('sha256')), 'exact closure evidence binding required')
        read_pinned(Path(binding['path']), binding['sha256'], 1024 * 1024)
    return row


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def serve(output, phase, closure_sha):
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
                     no_settings_changed=True, closure_admission_sha256=closure_sha,
                     original_route_package_sha256=ORIGINAL_PACKAGE_SHA,
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
                        closure_admission_sha256=closure_sha, connection_cap=CONNECTION_CAP,
                        concurrency_cap=CONCURRENCY_CAP, per_direction_buffer_bytes=BUFFER_CAP)
        with (output / 'terminal.json').open('x') as stream:
            json.dump(terminal, stream, indent=2); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root-serve', action='store_true')
    parser.add_argument('--phase-sha256'); parser.add_argument('--anchor-sha256')
    parser.add_argument('--prior-closure-admission', type=Path)
    parser.add_argument('--prior-closure-admission-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.root_serve:
        print(json.dumps(dict(status='inert_original_phase_recovery_route', new_clock_granted=False)))
        return
    phase = load_original_phase(args.phase_sha256, args.anchor_sha256)
    need(args.prior_closure_admission is not None and args.output is not None, 'explicit closure admission/output required')
    closure_admission(read_pinned(args.prior_closure_admission, args.prior_closure_admission_sha256), phase)
    phase.remaining()
    serve(args.output, phase, args.prior_closure_admission_sha256)


if __name__ == '__main__':
    main()
