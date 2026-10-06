#!/usr/bin/env python3
"""Independent exact-decimal arithmetic and scalar publication probes."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import goop_global_action_gate_core_v1 as CORE
import test_goop_action_gate_cost_plan_v1 as T

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return {'path': str(path), 'sha256': sha(path)}


def ceil(value):
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-sha', required=True)
    args = parser.parse_args()
    source = HERE / 'goop_action_gate_cost_plan_v1.py'
    assert sha(source) == args.expected_sha
    root = HERE / ('goop_action_gate_cost_independent_fixtures_' + args.expected_sha[:8])
    root.mkdir()
    snapshot = root / 'reviewed_source.py'
    snapshot.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location('_independent_cost_snapshot', snapshot)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    rows = []
    for mode in ('train-label-capacity', 'train-rollout-capacity'):
        for pattern in range(12):
            b, c, o, manifest = T.fixture(mode)
            for seed in range(3):
                setup = Decimal(seed + 1) * Decimal('1.25')
                overhead = Decimal(pattern + 1) * Decimal('.5')
                for i, timing in enumerate(c[seed]['case_timings']):
                    # Include cases where GPU1 is dominant and where each
                    # policy's slowest source occurs at a different rank.
                    mult = 20 if pattern % 3 == 0 and seed == 1 else seed + 1
                    timing['wall_seconds'] = float(Decimal(mult * (1 + (i * 7 + pattern) % 13)) * Decimal('.125'))
                c[seed]['runtime']['setup_seconds'] = float(setup)
                total = sum(Decimal(str(t['wall_seconds'])) for t in c[seed]['case_timings'])
                o[seed]['elapsed_seconds'] = float(setup + total + overhead)
            result = M.build_plan(b, c, o, manifest, T.NOW)
            allocations = {}
            for seed in range(3):
                setup = Decimal(str(c[seed]['runtime']['setup_seconds']))
                total = sum(Decimal(str(t['wall_seconds'])) for t in c[seed]['case_timings'])
                overhead = Decimal(str(o[seed]['elapsed_seconds'])) - setup - total
                if mode == 'train-rollout-capacity':
                    worst = [max(Decimal(str(t['wall_seconds'])) for t in c[seed]['case_timings'] if t['policy'] == p) for p in M.POLICIES]
                    expected = ceil(setup + overhead + Decimal('1.35') * 30 * sum(worst))
                    actual = result['per_seed'][seed]['test_allocation_seconds']
                    assert expected <= actual <= expected + 1
                    allocations[seed] = actual
                else:
                    slow = max(Decimal(str(t['wall_seconds'])) for t in c[seed]['case_timings'])
                    for phase, count in (('train_labels', 8000), ('validation_labels', 150)):
                        expected = ceil(setup + overhead + Decimal('1.35') * count * slow)
                        actual = result['per_seed'][seed]['whole_invocation_allocations_seconds'][phase]
                        assert expected <= actual <= expected + 1
            if mode == 'train-rollout-capacity':
                wall = max(allocations[0] + allocations[2] + 30, allocations[1] + 15)
                assert result['test_concurrent_allocation_seconds'] == wall
                latest = M.DEADLINE - timedelta(seconds=wall + 7200 + 1)
                assert M.stamp(result['latest_test_start_utc']) == latest
                assert M.build_plan(b, c, o, manifest, latest)['complete_remaining_work_fits']
                assert not M.build_plan(b, c, o, manifest, latest + timedelta(microseconds=1))['complete_remaining_work_fits']
            else:
                flat = CORE.row_ids('train')
                assert M.schedule(mode, manifest) == [flat[i] for i in CORE.capacity_label_indices()]
            rows.append({'case': mode + '_decimal_pattern_' + str(pattern), 'passed': True})

    b, c, o, manifest = T.fixture()
    b['completed_outer_stages'] = {'input_preflight': {'receipt_sha256': 'a' * 64, 'actual_elapsed_seconds': 0,
                                                      'completed_utc': T.NOW.isoformat()}}
    try:
        result = M.build_plan(b, c, o, manifest, T.NOW)
    except (ValueError, KeyError) as exc:
        rows.append({'case': 'unbacked_preflight_credit', 'passed': True, 'error': str(exc)})
    else:
        rows.append({'case': 'unbacked_preflight_credit', 'passed': result['remaining_outer_reserves_seconds'].get('input_preflight') == 900})

    for case in ('complete_cli_candidate', 'complete_bound_preflight_credit', 'input_mutation_during_serialization',
                 'receipt_mutation_during_serialization'):
        directory = root / case
        b, c, o, manifest = T.fixture()
        b['train_manifest'] = put(directory / 'train_manifest.json', manifest)
        b['train_manifest_sha256'] = b['train_manifest']['sha256']
        b['collections'], b['process_outcomes'] = {}, {}
        for seed in range(3):
            c[seed]['source_manifest_sha256'] = b['train_manifest_sha256']
            cp = directory / f'run{seed}' / 'rollout_collection.json'
            b['collections'][str(seed)] = put(cp, c[seed])
            o[seed]['command'] += ['--output-dir', str(cp.parent)]
            b['process_outcomes'][str(seed)] = put(directory / f'outcome{seed}.json', o[seed])
        receipt_path = directory / 'completed_preflight.json'
        if case in ('complete_bound_preflight_credit', 'receipt_mutation_during_serialization'):
            receipt = dict(schema='adaptgns_goop_action_gate_preflight_completion_v1', status='complete', issued_by='root',
                protocol_sha256=M.PROTOCOL_SHA, cohort_sha256=b['cohort_sha256'], selection_sha256=b['selection_sha256'],
                actual_elapsed_seconds=120, completed_utc=T.NOW.isoformat())
            b['completed_outer_stages'] = {'input_preflight': put(receipt_path, receipt)}
        bundle_path = Path(put(directory / 'bundle.json', b)['path'])
        output = directory / 'candidate.json'
        original_encode = M.encode
        mutations = []
        def encode(value):
            raw = original_encode(value)
            if case == 'input_mutation_during_serialization':
                Path(b['collections']['0']['path']).write_bytes(b'CHANGED DURING SERIALIZATION')
                mutations.append('collection0')
            elif case == 'receipt_mutation_during_serialization':
                receipt_path.write_bytes(b'CHANGED DURING SERIALIZATION')
                mutations.append('completed receipt')
            return raw
        class FixedDate(datetime):
            @classmethod
            def now(cls, tz=None):
                return T.NOW
        error = None
        with patch.object(M, 'encode', encode), patch.object(M, 'datetime', FixedDate):
            try:
                M.main(['--execute', '--input', str(bundle_path), '--output', str(output)])
            except ValueError as exc:
                error = str(exc)
        expected_success = case in ('complete_cli_candidate', 'complete_bound_preflight_credit')
        row = {'case': case, 'passed': output.exists() == expected_success and (error is None) == expected_success,
               'error': error, 'mutations': mutations, 'output_exists': output.exists()}
        if output.exists():
            value = json.loads(output.read_bytes())
            row['remaining_reserve_seconds'] = sum(value['remaining_outer_reserves_seconds'].values())
            row['passed'] &= row['remaining_reserve_seconds'] == (6300 if case == 'complete_bound_preflight_credit' else 7200)
            if case == 'complete_bound_preflight_credit':
                row['receipt_is_bound'] = value['input_sha256'].get(str(receipt_path)) == sha(receipt_path)
                row['passed'] &= row['receipt_is_bound']
        rows.append(row)
    report = {'schema': 'adaptgns_goop_action_gate_cost_independent_probes_code_audit_v1',
              'source_sha256': args.expected_sha, 'source_is_still_current': sha(source) == args.expected_sha,
              'synthetic_scalar_only': True, 'decimal_cases': 24, 'probes': rows, 'all_passed': all(r['passed'] for r in rows)}
    put(HERE / 'goop_action_gate_cost_independent_probes_code_audit_v1.json', report)
    print(json.dumps(report, indent=2))
    assert report['all_passed']


if __name__ == '__main__':
    main()
