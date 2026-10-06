#!/usr/bin/env python3
"""Replay failure-accounting checks only; never summarize performance or load arrays."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError('Evidence replay must run with Python assertions enabled; optimization is unsupported')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(workspace):
    prep = workspace / 'work/deadline_research_20261005/cuda_preparation'
    for version, expected in [('v2', 13), ('v3', 62)]:
        directory = prep / ('goop2d_cost_pilot_failure_20261006_root_' + version)
        inventory = read(directory / 'collection_inventory.json')
        assert len(inventory['files']) == expected
        for name, row in inventory['files'].items():
            path = directory / name
            assert sha(path) == row['sha256'] and path.stat().st_size == row['size_bytes']
        owner = read(directory / 'owner_receipt.json')
        assert owner['status'] == 'failed' and owner['observed_complete_blocks'] == 0
        assert owner['all_children_reaped'] and owner['final_gpu_processes'] == []
    decline = read(prep / 'goop2d_validation_cost_v3_root_decline_v1.json')
    assert decline['status'] == 'full_study_declined'
    assert all(decline[k] is False for k in ('scientific_result_admitted', 'full_launch_authorized',
                                            'retry_authorized', 'further_repair_authorized'))
    contract = prep / 'goop2d_validation_cost_contract_v3.py'
    assert sha(contract) == 'f657f36e9e50e9d90ed45c99e1e1fee58ce87286a100503edc798acbf8785dde'
    spec = importlib.util.spec_from_file_location('_failure_evidence_contract', contract)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    directory = prep / 'goop2d_cost_pilot_failure_20261006_root_v3'
    inventory = read(directory / 'collection_inventory.json')
    owner = read(directory / 'owner_receipt.json')
    release = read(prep / 'goop2d_validation_cost_staged_root_v3/pilot_release_issued_pending_stage_review.json')
    manifest = directory / 'pinned_valid_manifest.json'
    assert sha(manifest) == release['controls']['manifest']['sha256']
    expected = C.schedule(read(manifest)['records'], 'pilot')
    assert len(expected) == 15 and len(owner['children']) == 3
    assert {(c['arm'], c['seed'], c['exit_code']) for c in owner['children']} == {
        ('base', 0, -2), ('mix', 1, 0), ('base', 2, 0)}
    states, opaque = 0, set()
    for child in owner['children']:
        assert child['verified'] is False and child['reaped'] is True
        local = directory / ('seed%d_%s' % (child['seed'], child['arm']))
        receipt, progress, protocol = [read(local / name) for name in ('receipt.json', 'progress.json', 'protocol.json')]
        assert receipt['schedule'] == protocol['schedule'] == child['schedule'] == expected
        assert receipt['committed'] == progress['committed'] and len(receipt['committed']) == 15
        gpu = next(g for g in release['gpus'] if g['index'] == child['seed'])
        for committed, item in zip(receipt['committed'], expected):
            assert committed['path'] == 'state_%03d.json' % item['schedule_index']
            path = local / committed['path']
            assert sha(path) == committed['sha256']
            result = read(path)
            C.scalar_block(result, item, child['seed'], gpu)
            name = str(path.with_suffix('.npz').relative_to(directory))
            assert result['artifact_sha256'] == inventory['opaque'][name]['sha256']
            opaque.add(name)
            states += 1
    assert states == 45 and opaque == set(inventory['opaque'])
    return dict(status='failure_accounting_replayed', original_text_files_checked=75,
                published_diagnostic_scalar_states=45, owner_verified_states=0,
                full_study_declined=True, performance_statistics_computed=False,
                numerical_arrays_loaded=False, scientific_result_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('workspace', type=Path)
    print(json.dumps(replay(parser.parse_args().workspace.resolve()), indent=2))
