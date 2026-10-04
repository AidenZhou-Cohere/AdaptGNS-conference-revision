"""Read-only snapshots of the fixed training queue, with conditional timing estimates."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def recent_progress(log_path):
    """Use only the final monotone log segment of the current queue attempt.

    Repeated progress from earlier interrupted attempts must not count as new
    completed updates. A queue-start marker resets the segment before resumed
    progress arrives. A partly written final JSON line is ignored.
    """
    log = []
    for line in log_path.read_text().splitlines() if log_path.exists() else []:
        if line.startswith('QUEUE START '):
            log = []
            continue
        if not line.startswith('{'):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'completed_steps' not in item or 'elapsed_seconds' not in item:
            continue
        if log and (item['completed_steps'] <= log[-1]['completed_steps']
                    or item['elapsed_seconds'] <= log[-1]['elapsed_seconds']):
            log = []
        log.append(item)
    spans = []
    for a, b in zip(log, log[1:]):
        steps = b['completed_steps'] - a['completed_steps']
        seconds = b['elapsed_seconds'] - a['elapsed_seconds']
        spans.append({'steps': steps, 'seconds': seconds,
                      'seconds_per_update': seconds / steps})
    return spans


def process_liveness(pid):
    if not isinstance(pid, int) or pid <= 0:
        return 'unknown'
    try:
        os.kill(pid, 0)
        return 'present_command_not_verified'
    except ProcessLookupError:
        return 'absent'
    except PermissionError:
        return 'unknown_permission'


def snapshot(result_dir, repo, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    queue = read_json(result_dir/'queue_status.json')
    if queue is None:
        raise ValueError('No queue status found')
    rows, active = [], None
    for seed in range(3):
        for objective in ('faithful', 'nll'):
            name = f'{objective}_seed{seed}'
            folder = result_dir/name
            status = read_json(folder/'status.json')
            if status is None:
                rows.append({'name':name,'objective':objective,'seed':seed,'state':'not_started','completed_steps':0})
                continue
            config = read_json(folder/'protocol.json')
            if status.get('objective') != objective or status.get('seed') != seed or status.get('requested_steps') != 100000:
                raise ValueError(f'Unexpected experiment identity/budget: {name}')
            if type(status.get('completed_steps')) is not int or not 0 <= status['completed_steps'] <= 100000:
                raise ValueError(f'Invalid completed-update count: {name}')
            if config is None or config_hash(config) != status.get('run_config_sha256'):
                raise ValueError(f'Missing or changed saved run configuration: {name}')
            mismatch = [path for path,expected in config['source_sha256'].items()
                        if not (repo/path).is_file() or sha(repo/path)!=expected]
            portable = repo/'research/protocols/full_waterdrop_100k.md'
            if sha(portable) != config['research_protocol_sha256']:
                mismatch.append(str(portable.relative_to(repo)))
            row = {key:status.get(key) for key in ('state','objective','seed','completed_steps','requested_steps','pid',
                   'started_utc','updated_utc','elapsed_seconds','error','last_validation','latest_checkpoint')}
            row.update(name=name, frozen_source_mismatches=mismatch, process_liveness=process_liveness(status.get('pid')))
            row['status_age_seconds']=(now-datetime.fromisoformat(status['updated_utc'])).total_seconds()
            spans = recent_progress(folder/'run.log')
            if spans:
                row['observed_average_seconds_per_update']=sum(s['seconds'] for s in spans)/sum(s['steps'] for s in spans)
                recent=spans[-10:]
                row['recent_median_seconds_per_update']=statistics.median(s['seconds_per_update'] for s in recent)
                row['recent_interval_count']=len(recent)
                if name == queue.get('current_job'):
                    active = row
            rows.append(row)
    completed=sum(r['completed_steps'] for r in rows)
    cutoff=datetime.fromisoformat(queue['deadline_utc'])
    result={'recorded_utc':now.isoformat(),'queue_state':queue['state'],'queue_pid':queue['pid'],
            'queue_liveness':process_liveness(queue['pid']),'jobs':rows,'total_requested_updates':600000,
            'total_completed_updates':completed,'deadline_utc':cutoff.isoformat(),
            'total_recorded_checkpoint_updates':sum((r.get('latest_checkpoint') or {}).get('completed_steps', 0) for r in rows),
            'hours_to_cutoff':(cutoff-now).total_seconds()/3600,
            'interpretation':'Read-only last-recorded progress, not independently verified recoverable updates. Checkpoint bytes are not verified here. No checkpoint selection, training edits, process interruption or test evaluation. PID existence does not verify command identity.'}
    result['required_seconds_per_remaining_update_before_cutoff'] = (
        max(0., (cutoff-now).total_seconds()) / (600000-completed) if completed < 600000 else None)
    can_project = (active is not None and queue['state'] == 'running' and active['state'] == 'running'
                   and active['pid'] == queue.get('child_pid')
                   and result['queue_liveness'] != 'absent' and active['process_liveness'] != 'absent'
                   and not active['frozen_source_mismatches'])
    if can_project:
        rates={'observed_average':active['observed_average_seconds_per_update'],
               'recent_median':active['recent_median_seconds_per_update']}
        projections={}
        for label, rate in rates.items():
            seconds=(600000-completed)*rate
            finish=now+timedelta(seconds=seconds)
            projections[label]={'seconds_per_update':rate,'remaining_training_hours':seconds/3600,
                                'projected_training_finish_utc':finish.isoformat(),'after_cutoff':finish>cutoff}
        result['conditional_projections']=projections
        result['projection_limitations']='Uses the queue-designated child only. Process command identities require separate verification. Assumes the same future per-update throughput for all models; ignores future contention changes, startup and new evaluation time. Not a promised completion time.'
    else:
        result['projection_withheld_reason'] = 'No matching running queue child with recent progress, non-absent PIDs and intact frozen source hashes.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir',type=Path,default=Path('research/results/full_waterdrop_100k'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    result=snapshot(args.result_dir.resolve(),repo)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'},indent=2))


if __name__=='__main__':main()
