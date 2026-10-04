"""Run the six fixed local jobs sequentially, with a single queue lock and deadline.

This launcher never evaluates test data or modifies the scientific protocol.
A failed numerical run stops the queue for review; interruptions may resume from
an existing atomic checkpoint after the previous process is confirmed dead.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from research.full_training import atomic_json, sha256, config_hash


def utc_now():
    return datetime.now(timezone.utc)


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(path):
    """Serial recovery plus O_EXCL prevents two launchers stealing a new lock."""
    token = uuid.uuid4().hex
    payload = {'pid': os.getpid(), 'created_utc': utc_now().isoformat(), 'token': token}
    if path.exists():
        recovery = path.with_suffix('.lock.recovery')
        recovery_fd = os.open(recovery, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(recovery_fd, json.dumps(payload).encode())
            os.fsync(recovery_fd)
            if path.exists():
                previous = json.loads(path.read_text())
                if not isinstance(previous.get('pid'), int) or previous['pid'] <= 0:
                    raise RuntimeError('Unreadable queue PID; refusing stale-lock removal')
                if process_alive(previous['pid']):
                    raise RuntimeError(f"Queue already running as PID {previous['pid']}")
                path.unlink()
        finally:
            os.close(recovery_fd)
            recovery.unlink()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return token


def stop_child(child):
    if child.poll() is not None:
        return
    for action in (lambda: child.send_signal(signal.SIGINT), child.terminate, child.kill):
        action()
        try:
            child.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            continue
    # Ownership is retained until the OS reaps the child, even after SIGKILL.
    child.wait()


def verify_completed(output, job, args, previous):
    if previous.get('completed_steps') != 100000 or previous.get('requested_steps') != 100000:
        raise RuntimeError('Refusing to treat a different training budget as complete')
    latest = json.loads((output/'latest.json').read_text())
    if latest['completed_steps'] != 100000 or sha256(output/latest['path']) != latest['sha256']:
        raise RuntimeError('Completed checkpoint is missing or its hash differs')
    config = json.loads((output/'protocol.json').read_text())
    if previous.get('objective') != job['objective'] or previous.get('seed') != job['seed']:
        raise RuntimeError('Completed status has a different experiment identity')
    if config.get('objective') != job['objective'] or config.get('seed') != job['seed']:
        raise RuntimeError('Completed protocol has a different experiment identity')
    if config.get('scope') != 'bounded_full_data_100k' or config_hash(config) != previous.get('run_config_sha256'):
        raise RuntimeError('Completed protocol hash/scope differs from recorded status')
    if sha256(args.protocol) != config['research_protocol_sha256']:
        raise RuntimeError('Scientific protocol changed after this completed run')
    if sha256(args.data_dir/'metadata.json') != config['metadata_sha256']:
        raise RuntimeError('Metadata changed after this completed run')
    for split in ('train', 'valid'):
        if sha256(args.data_dir/(split+'.json')) != config[split]['manifest_sha256']:
            raise RuntimeError('Data manifest changed after this completed run')
    repo = Path(__file__).resolve().parents[1]
    for source, expected in config['source_sha256'].items():
        if sha256(repo/source) != expected:
            raise RuntimeError('Training source changed after this completed run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--deadline-utc', default='2026-10-07T08:00:00+00:00')
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.result_dir = args.result_dir.resolve()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        parser.error('deadline must include a UTC offset')
    args.result_dir.mkdir(parents=True, exist_ok=True)
    for name in ['train.json', 'valid.json', 'metadata.json']:
        if not (args.data_dir / name).is_file():
            raise FileNotFoundError(args.data_dir / name)
    lock = args.result_dir / 'queue.lock'
    lock_token = acquire_lock(lock)
    jobs = [{'objective': objective, 'seed': seed, 'name': f'{objective}_seed{seed}'}
            for seed in [0, 1, 2] for objective in ['faithful', 'nll']]
    started = utc_now().isoformat()
    child = None
    def status(state, **details):
        atomic_json(args.result_dir / 'queue_status.json', {
            'state': state, 'pid': os.getpid(), 'started_utc': started,
            'updated_utc': utc_now().isoformat(), 'deadline_utc': deadline.isoformat(),
            'jobs': jobs, **details})
    try:
        status('running')
        for job in jobs:
            output = args.result_dir / job['name']
            output.mkdir(parents=True, exist_ok=True)
            current_path = output / 'status.json'
            previous = json.loads(current_path.read_text()) if current_path.exists() else {}
            if previous.get('state') == 'complete':
                verify_completed(output, job, args, previous)
                job['state'] = 'complete'
                continue
            if previous.get('state') == 'failed':
                raise RuntimeError(f"{job['name']} failed and requires review: {previous.get('error')}")
            if utc_now() >= deadline:
                status('deadline_reached_incomplete', current_job=job['name'])
                return
            command = [sys.executable, '-m', 'research.full_training',
                '--train-manifest', str(args.data_dir/'train.json'),
                '--valid-manifest', str(args.data_dir/'valid.json'),
                '--metadata', str(args.data_dir/'metadata.json'),
                '--output-dir', str(output), '--objective', job['objective'],
                '--seed', str(job['seed']), '--device', 'mps', '--threads', '2',
                '--protocol', str(args.protocol.resolve()), '--clear-stale-lock']
            if (output/'protocol.json').exists():
                command.append('--resume')
            job['state'] = 'running'
            status('running', current_job=job['name'], command=command)
            environment = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0')
            with (output/'run.log').open('a') as log:
                log.write('\nQUEUE START '+utc_now().isoformat()+'\n')
                log.flush()
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                status('running', current_job=job['name'], child_pid=child.pid, command=command)
                while child.poll() is None:
                    if utc_now() >= deadline:
                        stop_child(child)
                        status('deadline_reached_incomplete', current_job=job['name'])
                        return
                    time.sleep(5)
            code = child.returncode
            child = None
            current = json.loads(current_path.read_text()) if current_path.exists() else {}
            if code != 0 or current.get('state') != 'complete' or current.get('completed_steps') != 100000:
                raise RuntimeError(f"{job['name']} exit={code}, state={current.get('state')}; inspect run.log")
            verify_completed(output, job, args, current)
            job['state'] = 'complete'
            status('running', completed_job=job['name'])
        status('complete')
    except BaseException as error:
        if child is not None and child.poll() is None:
            stop_child(child)
        status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text()).get('token') == lock_token:
            lock.unlink()


if __name__ == '__main__':
    main()
