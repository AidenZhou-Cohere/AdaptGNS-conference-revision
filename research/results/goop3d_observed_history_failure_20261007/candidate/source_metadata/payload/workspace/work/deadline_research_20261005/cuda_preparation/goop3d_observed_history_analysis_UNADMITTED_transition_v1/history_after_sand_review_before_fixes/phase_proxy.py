"""Bounded opaque CONNECT route to the original, DNS-verified Coder service.

No authentication, TLS termination, global settings, subprocesses or payload logs.
Route for one already admitted distinct D3 observed-history phase; no clock issuance.
Original metadata proxy remains unchanged. Cap128 covers fixed operations plus
Coder connection multiplicity; eight concurrent handlers/backlog4 are unchanged.
"""
import argparse
import datetime
import json
import select
import socket
import threading
import time
from pathlib import Path
from phase_binding import load_original_phase, need

AUTHORITY = b"coder.internal.cohere.com:443"
UPSTREAM = ("100.106.33.61", 443)
MAX_HEADER = 8192


def parse_request(header):
    if len(header) > MAX_HEADER or not header.endswith(b"\r\n\r\n"):
        raise ValueError("invalid bounded header")
    fields = header.split(b"\r\n", 1)[0].split(b" ")
    if len(fields) != 3 or fields[0] != b"CONNECT" or fields[1] != AUTHORITY:
        raise ValueError("only the exact original authority is permitted")
    if fields[2] not in (b"HTTP/1.0", b"HTTP/1.1"):
        raise ValueError("unsupported HTTP version")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def serve(output, phase):
    seconds = phase.remaining()
    need(output.is_absolute() and output.resolve() == output and not output.exists(), "fresh canonical output required")
    output.mkdir(exist_ok=False)
    started = now()
    wall_start, mono_start = time.time(), time.monotonic()
    stop = threading.Event()
    lock = threading.Lock()
    sockets = set()
    workers = []
    counts = {"accepted": 0, "routed": 0, "rejected_or_closed": 0}

    def expired():
        wall_elapsed = time.time() - wall_start
        mono_elapsed = time.monotonic() - mono_start
        return (stop.is_set() or phase.expired() or wall_elapsed >= seconds or mono_elapsed >= seconds
                or abs(wall_elapsed - mono_elapsed) > 2)

    def remember(sock):
        with lock:
            sockets.add(sock)

    def close(sock):
        if sock is not None:
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
            client.settimeout(2)
            data = b""
            while b"\r\n\r\n" not in data:
                if expired() or len(data) >= MAX_HEADER:
                    raise ValueError("request deadline or header limit")
                chunk = client.recv(min(1024, MAX_HEADER - len(data)))
                if not chunk:
                    raise ValueError("closed request")
                data += chunk
            header, extra = data.split(b"\r\n\r\n", 1)
            parse_request(header + b"\r\n\r\n")
            if expired():
                raise ValueError("expired before connect")
            upstream = socket.create_connection(UPSTREAM, timeout=3)
            remember(upstream)
            upstream.settimeout(1)
            client.settimeout(1)
            if expired():
                raise ValueError("expired during connect")
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            if extra:
                if expired():
                    raise ValueError("expired before initial forwarding")
                upstream.sendall(extra)
            with lock:
                counts["routed"] += 1
            while not expired():
                readable, _, _ = select.select([client, upstream], [], [], 0.2)
                if expired():
                    return
                for source in readable:
                    if expired():
                        return
                    payload = source.recv(65536)
                    if not payload:
                        return
                    if expired():
                        return
                    (upstream if source is client else client).sendall(payload)
        except (OSError, ValueError):
            with lock:
                counts["rejected_or_closed"] += 1
        finally:
            close(upstream)
            close(client)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(0.2)
        port = listener.getsockname()[1]
        ready = {"schema": "coder_goop3d_observed_phase_route_proxy_v1", **phase.metadata, "started_utc": started,
                 "lifetime_seconds": seconds, "url": f"http://127.0.0.1:{port}",
                 "authority": AUTHORITY.decode(), "upstream": list(UPSTREAM),
                 "opaque_tls": True, "no_settings_changed": True}
        (output / "ready.json").write_text(json.dumps(ready, indent=2) + "\n")
        print(json.dumps(ready), flush=True)
        while not expired():
            try:
                client, address = listener.accept()
            except socket.timeout:
                continue
            remember(client)
            if address[0] != "127.0.0.1" or counts["accepted"] >= 128:
                close(client)
                continue
            counts["accepted"] += 1
            workers = [w for w in workers if w.is_alive()]
            if len(workers) >= 8:
                close(client)
                continue
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
        terminal = {**phase.metadata, "started_utc": started, "ended_utc": now(),
                    "lifetime_seconds": seconds, "counts": counts,
                    "remaining_worker_threads": sum(w.is_alive() for w in workers),
                    "wall_elapsed_seconds": time.time() - wall_start,
                    "monotonic_elapsed_seconds": time.monotonic() - mono_start,
                    "payloads_logged": False, "remote_processes_signaled": False}
        (output / "terminal.json").write_text(json.dumps(terminal, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root-serve", action="store_true")
    parser.add_argument("--phase-sha256")
    parser.add_argument("--anchor-sha256")
    args = parser.parse_args()
    if not args.root_serve:
        print(json.dumps({"status": "inert_original_phase_route", "new_clock_granted": False}))
    else:
        need(args.output is not None, "fresh output required")
        phase = load_original_phase(args.phase_sha256, args.anchor_sha256)
        serve(args.output, phase)
