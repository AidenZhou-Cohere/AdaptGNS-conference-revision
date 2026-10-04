from datetime import datetime, timezone
import json

import pytest

from research import monitor_training as monitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def running(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, 'process_liveness', lambda pid: 'present_command_not_verified')
    repo = tmp_path / 'repo'
    protocol = repo / 'research/protocols/full_waterdrop_100k.md'
    protocol.parent.mkdir(parents=True)
    protocol.write_text('fixed protocol')
    source = repo / 'training.py'
    source.write_text('frozen source')
    root = tmp_path / 'results'
    queue = {'state': 'running', 'pid': 101, 'child_pid': 102, 'current_job': 'faithful_seed0',
             'deadline_utc': '2026-10-07T08:00:00+00:00'}
    write(root / 'queue_status.json', queue)
    folder = root / 'faithful_seed0'
    config = {'source_sha256': {'training.py': monitor.sha(source)},
              'research_protocol_sha256': monitor.sha(protocol)}
    write(folder / 'protocol.json', config)
    status = {'state': 'running', 'pid': 102, 'seed': 0, 'objective': 'faithful',
              'requested_steps': 100000, 'completed_steps': 200,
              'updated_utc': '2026-10-04T18:00:00+00:00',
              'run_config_sha256': monitor.config_hash(config)}
    write(folder / 'status.json', status)
    (folder / 'run.log').write_text('\n'.join(json.dumps(x) for x in [
        {'completed_steps': 100, 'elapsed_seconds': 50},
        {'completed_steps': 200, 'elapsed_seconds': 100}]))
    return root, repo, datetime(2026, 10, 4, 18, 1, tzinfo=timezone.utc), status


def test_missing_models_are_pending_and_projection_is_conditional(running):
    root, repo, now, _ = running
    result = monitor.snapshot(root, repo, now)
    assert result['total_completed_updates'] == 200
    assert result['total_requested_updates'] == 600000
    assert [r['state'] for r in result['jobs']] == ['running'] + ['not_started'] * 5
    assert result['jobs'][0]['status_age_seconds'] == 60
    assert result['jobs'][0]['frozen_source_mismatches'] == []
    assert result['jobs'][0]['process_liveness'] == 'present_command_not_verified'
    assert result['conditional_projections']['recent_median']['after_cutoff'] is True
    assert result['conditional_projections']['recent_median']['remaining_training_hours'] == pytest.approx(599800 * .5 / 3600)


def test_stopped_queue_is_not_projected_as_running(running):
    root, repo, now, _ = running
    queue = json.loads((root / 'queue_status.json').read_text())
    queue['state'] = 'error'
    write(root / 'queue_status.json', queue)
    result = monitor.snapshot(root, repo, now)
    assert 'conditional_projections' not in result
    assert result['total_completed_updates'] == 200


def test_source_mutation_is_reported_without_touching_training(running):
    root, repo, now, _ = running
    (repo / 'training.py').write_text('changed')
    before = (root / 'faithful_seed0/status.json').read_bytes()
    result = monitor.snapshot(root, repo, now)
    assert result['jobs'][0]['frozen_source_mismatches'] == ['training.py']
    assert 'conditional_projections' not in result
    assert (root / 'faithful_seed0/status.json').read_bytes() == before


def test_saved_run_configuration_mutation_is_rejected(running):
    root, repo, now, _ = running
    path = root / 'faithful_seed0/protocol.json'
    config = json.loads(path.read_text())
    config['steps'] = 50000
    write(path, config)
    with pytest.raises(ValueError, match='changed saved run configuration'):
        monitor.snapshot(root, repo, now)


@pytest.mark.parametrize('completed', [-1, 100001, 5.5, True])
def test_invalid_progress_is_not_counted(running, completed):
    root, repo, now, status = running
    status['completed_steps'] = completed
    write(root / 'faithful_seed0/status.json', status)
    with pytest.raises(ValueError, match='completed-update count'):
        monitor.snapshot(root, repo, now)


def test_checkpoint_rollback_uses_only_new_log_segment(tmp_path):
    path = tmp_path / 'log'
    entries = [(100, 10), (200, 20), (100, 30), (200, 90), (300, 170)]
    path.write_text('\n'.join(json.dumps({'completed_steps': n, 'elapsed_seconds': t})
                              for n, t in entries) + '\n{"completed_steps": 400,')
    spans = monitor.recent_progress(path)
    assert [s['seconds_per_update'] for s in spans] == [.6, .8]


@pytest.mark.parametrize('fresh_progress', [[], [(20100, 8532)]])
def test_queue_start_withholds_projection_until_a_fresh_span(running, fresh_progress):
    root, repo, now, status = running
    folder = root / 'faithful_seed0'
    status['completed_steps'] = 20000
    write(folder / 'status.json', status)
    old = [(21400, 9058), (21500, 9075)]
    encode = lambda values: '\n'.join(json.dumps({'completed_steps': n, 'elapsed_seconds': t})
                                      for n, t in values)
    (folder / 'run.log').write_text(encode(old) + '\nQUEUE START 2026-10-04T19:00:00+00:00\n'
                                   + encode(fresh_progress))
    before = (folder / 'status.json').read_bytes()
    result = monitor.snapshot(root, repo, now)
    assert 'conditional_projections' not in result
    assert 'observed_average_seconds_per_update' not in result['jobs'][0]
    assert 'recent_median_seconds_per_update' not in result['jobs'][0]
    assert monitor.recent_progress(folder / 'run.log') == []
    assert (folder / 'status.json').read_bytes() == before


def test_queue_start_discards_old_spans_even_when_new_progress_is_monotone(tmp_path):
    path = tmp_path / 'log'
    encode = lambda n, t: json.dumps({'completed_steps': n, 'elapsed_seconds': t})
    path.write_text('\n'.join([encode(100, 10), encode(200, 20),
                               'QUEUE START 2026-10-04T19:00:00+00:00',
                               encode(300, 80), encode(400, 100)]))
    assert monitor.recent_progress(path) == [{'steps': 100, 'seconds': 20, 'seconds_per_update': .2}]


def test_permission_denial_does_not_claim_process_absence(monkeypatch):
    def denied(pid, signal):
        assert signal == 0
        raise PermissionError()
    monkeypatch.setattr(monitor.os, 'kill', denied)
    assert monitor.process_liveness(42) == 'unknown_permission'


def test_dead_or_replaced_queue_child_has_no_projection(running, monkeypatch):
    root, repo, now, _ = running
    monkeypatch.setattr(monitor, 'process_liveness', lambda pid: 'absent' if pid == 102 else 'present_command_not_verified')
    assert 'conditional_projections' not in monitor.snapshot(root, repo, now)
    monkeypatch.setattr(monitor, 'process_liveness', lambda pid: 'present_command_not_verified')
    queue = json.loads((root / 'queue_status.json').read_text())
    queue['child_pid'] = 103
    write(root / 'queue_status.json', queue)
    assert 'conditional_projections' not in monitor.snapshot(root, repo, now)


def test_stale_running_model_does_not_supply_the_current_child_rate(running):
    root, repo, now, status = running
    folder = root / 'nll_seed0'
    status = {**status, 'objective': 'nll', 'pid': 103}
    write(folder / 'status.json', status)
    write(folder / 'protocol.json', json.loads((root / 'faithful_seed0/protocol.json').read_text()))
    (folder / 'run.log').write_text('\n'.join(json.dumps({'completed_steps': n, 'elapsed_seconds': t})
                                              for n, t in [(100, 1), (200, 2)]))
    result = monitor.snapshot(root, repo, now)
    assert result['conditional_projections']['recent_median']['seconds_per_update'] == .5
