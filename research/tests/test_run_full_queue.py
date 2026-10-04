import json
import os
import pytest
from research import run_full_queue


def test_queue_never_steals_a_live_lock(tmp_path):
    path = tmp_path/'queue.lock'
    run_full_queue.acquire_lock(path)
    with pytest.raises(RuntimeError, match='already running'):
        run_full_queue.acquire_lock(path)
    assert json.loads(path.read_text())['pid'] == os.getpid()


def test_queue_can_recover_only_a_confirmed_stale_lock(tmp_path, monkeypatch):
    path = tmp_path/'queue.lock'
    path.write_text(json.dumps({'pid': 12345}))
    observed = []
    monkeypatch.setattr(run_full_queue, 'process_alive', lambda pid: observed.append(pid) or False)
    run_full_queue.acquire_lock(path)
    assert observed == [12345]
    assert json.loads(path.read_text())['pid'] == os.getpid()


def test_expired_queue_starts_no_training(tmp_path, monkeypatch):
    data = tmp_path/'data'; data.mkdir()
    for name in ('train.json','valid.json','metadata.json'):
        (data/name).write_text('{}')
    result = tmp_path/'result'
    monkeypatch.setattr(run_full_queue.sys, 'argv', ['queue', '--data-dir', str(data),
        '--result-dir', str(result), '--protocol', str(tmp_path/'protocol.md'),
        '--deadline-utc', '2000-01-01T00:00:00+00:00'])
    def no_launch(*args, **kwargs):
        raise AssertionError('Expired queue must never launch a subprocess')
    monkeypatch.setattr(run_full_queue.subprocess, 'Popen', no_launch)
    run_full_queue.main()
    assert json.loads((result/'queue_status.json').read_text())['state'] == 'deadline_reached_incomplete'
    assert not (result/'queue.lock').exists()


def test_shutdown_reaps_child_after_escalating(monkeypatch):
    calls = []
    class Child:
        attempts = 0
        def poll(self): return None
        def send_signal(self, sig): calls.append('interrupt')
        def terminate(self): calls.append('terminate')
        def kill(self): calls.append('kill')
        def wait(self, timeout=None):
            self.attempts += 1
            calls.append('wait')
            if self.attempts < 3:
                raise run_full_queue.subprocess.TimeoutExpired('train', timeout)
            return -9
    run_full_queue.stop_child(Child())
    assert calls == ['interrupt','wait','terminate','wait','kill','wait']


def test_completed_smoke_cannot_be_relabelled_as_full_run(tmp_path):
    with pytest.raises(RuntimeError, match='different training budget'):
        run_full_queue.verify_completed(tmp_path, {}, None,
            {'completed_steps':100, 'requested_steps':100})
