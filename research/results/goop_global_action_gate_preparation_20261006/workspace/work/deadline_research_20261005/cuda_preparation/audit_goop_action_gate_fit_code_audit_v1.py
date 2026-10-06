#!/usr/bin/env python3
"""Synthetic orchestration probes; real core, stubbed already-checked collector.

The artifact reader is independently tested by its committed NPZ fixtures.
This file tests root release closure and terminal publication without creating
24,450 duplicated artifact files or opening scientific data/checkpoints.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import goop_global_action_gate_core_v1 as CORE

HERE = Path(__file__).resolve().parent
PIN = '85411dd07972fd555da86210c8bf1825d31801c6d0b168e9f8106f2492ea1ee3'
NOW = datetime(2026, 10, 6, 23, tzinfo=timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return path


def main():
    original = HERE / 'fit_goop_action_gate_v1.py'
    assert sha(original) == PIN
    root = HERE / 'goop_action_gate_fit_independent_fixtures_v1'
    root.mkdir()
    snapshot = root / 'reviewed_source.py'
    snapshot.write_bytes(original.read_bytes())
    spec = importlib.util.spec_from_file_location('_independent_fit_snapshot', snapshot)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    rng = np.random.default_rng(84731)
    numeric = {}
    for seed in range(3):
        for split in ('train', 'valid'):
            ids = CORE.row_ids(split)
            features = rng.normal(size=(len(ids), 11))
            benefit = .002 * features[:, 0] - .001 * features[:, 3]
            numeric[seed, split] = dict(features=features, benefits=benefit, row_ids=ids,
                base_mse=np.full(len(ids), .1), random25_mse=.1 - benefit)
    results = []
    scenarios = ('complete', 'missing_driver_binding', 'missing_collection_input', 'different_seed_manifest',
                 'wrong_cohort_checkpoint', 'mutate_input_on_serialization', 'mutate_label_tree_on_serialization',
                 'mutate_head_on_serialization', 'expire_on_serialization')
    for case in scenarios:
        directory = root / case
        directory.mkdir()
        lineage = {}
        for key in F.RELEASE_PATHS:
            if key != 'cohort_sha256':
                lineage[key] = str(save(directory / (key + '.json'), {'independent_fixture': key}))
        members = []
        models = []
        for arm in ('base', 'mix'):
            for seed in range(3):
                checkpoint = directory / f'{arm}{seed}.opaque'
                checkpoint.write_bytes(f'NOT A MODEL: {arm} {seed}'.encode())
                members.append(dict(arm=arm, seed=seed, objective='faithful', completed_steps=100000,
                                    checkpoint_sha256=sha(checkpoint)))
                if arm == 'mix':
                    models.append(dict(seed=seed, checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                                       run_config_sha256=h(('synthetic-config', seed))))
        cohort = dict(schema='adaptgns_goop_graph_support_final_cohort_v1', status='frozen_for_final_evaluation',
                      dataset='Goop', updates=100000, issued_by='root', models=members,
                      protocol_sha256=sha(lineage['original_training_protocol_sha256']),
                      training_admission_sha256=sha(lineage['training_admission_sha256']),
                      cohort_audit_sha256=sha(lineage['cohort_audit_sha256']))
        if case == 'wrong_cohort_checkpoint':
            cohort['models'][3]['checkpoint_sha256'] = h('different checkpoint')
        lineage['cohort_sha256'] = str(save(directory / 'cohort.json', cohort))
        core_path = HERE / 'goop_global_action_gate_core_v1.py'
        protocol_path = HERE / 'goop_global_action_gate_protocol_v1.md'
        inputs = {p: sha(p) for p in lineage.values()}
        inputs.update({m['checkpoint_path']: m['checkpoint_sha256'] for m in models})
        inputs.update({str(core_path): F.CORE_SHA, str(protocol_path): F.PROTOCOL_SHA})
        entries = []
        for seed in range(3):
            for split in ('train', 'valid'):
                label_root = directory / f'labels_{seed}_{split}'
                save(label_root / 'protocol.json', {'synthetic_stub': True})
                files, tree = F.inventory(label_root)
                collection = {key: inputs[lineage[key]] for key in F.LINEAGE if key != 'core_sha256'}
                collection.update(core_sha256=F.CORE_SHA,
                    source_manifest_sha256=inputs[lineage[split + '_manifest_sha256']],
                    model=dict(arm='mix', seed=seed, objective='faithful', completed_updates=100000,
                               checkpoint_sha256=models[seed]['checkpoint_sha256'], run_config_sha256=models[seed]['run_config_sha256']),
                    files=files, output_tree_entries=tree, input_sha256=dict(inputs))
                if case == 'missing_collection_input' and (seed, split) == (1, 'valid'):
                    del collection['input_sha256'][lineage['driver_sha256']]
                if case == 'different_seed_manifest' and (seed, split) == (1, 'valid'):
                    collection['source_manifest_sha256'] = h('different validation source')
                cp = save(label_root / 'label_collection.json', collection)
                entries.append(dict(seed=seed, split=split, path=str(cp), sha256=sha(cp)))
        if case == 'missing_driver_binding':
            del inputs[lineage['driver_sha256']]
        output = directory / 'fit'
        release = dict(schema='adaptgns_goop_global_action_gate_fit_release_v1', issued_by='root',
            status='approved_complete_labels_for_fitting', output_dir=str(output), core_sha256=F.CORE_SHA,
            gate_protocol_sha256=F.PROTOCOL_SHA, fit_source_sha256=PIN, driver_sha256=sha(lineage['driver_sha256']),
            max_seconds=60, cpu_threads=2, absolute_stop_utc='2026-10-07T04:00:00+00:00',
            files_sha256=inputs, collections=entries, lineage_paths=lineage, models=models)
        rp = save(directory / 'release.json', release)
        clock = [NOW]
        class FixedDate(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock[0]
        def checked_reader(entry, release, np, core, all_inputs, check_time):
            # Only this boundary is stubbed. Artifact-reader tests cover real
            # row JSON+NPZ checks, and main still runs full-sized core fits.
            value = F.snapshot(entry['path'], entry['sha256'])
            F.merge_inputs(all_inputs, value['input_sha256'])
            F.merge_inputs(all_inputs, {entry['path']: entry['sha256']})
            return value, numeric[entry['seed'], entry['split']]
        original_encode = F.encode
        serializations = []
        def encode(value):
            raw = original_encode(value)
            if value.get('status') == 'all_three_gates_fitted_and_selected':
                serializations.append('final result')
                if case == 'mutate_input_on_serialization':
                    Path(lineage['metadata_sha256']).write_bytes(b'CHANGED DURING SERIALIZATION')
                elif case == 'mutate_label_tree_on_serialization':
                    (Path(entries[0]['path']).parent / 'unexpected.tmp').write_bytes(b'changed')
                elif case == 'mutate_head_on_serialization':
                    (output / 'selected_seed0.json').write_bytes(b'CHANGED DURING SERIALIZATION')
                elif case == 'expire_on_serialization':
                    clock[0] = datetime(2026, 10, 7, 4, tzinfo=timezone.utc)
            return raw
        error = None
        with patch.object(F, 'datetime', FixedDate), patch.object(F, 'read_collection', checked_reader), patch.object(F, 'encode', encode):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    F.main(['--execute', '--release', str(rp), '--core', str(core_path),
                            '--gate-protocol', str(protocol_path), '--output-dir', str(output)])
            except ValueError as exc:
                error = str(exc)
        ready = output / 'fit_result.json'
        passed = (ready.exists() and error is None) if case == 'complete' else (not ready.exists() and error is not None)
        row = dict(case=case, passed=passed, error=error, success_output_exists=ready.exists(),
                   final_serialization_count=len(serializations), failure_receipt_exists=(output / 'fit_failure.json').exists())
        if ready.exists():
            result = json.loads(ready.read_bytes())
            row['output_files'] = len(result['output_sha256'])
            head = json.loads((output / 'selected_seed0.json').read_bytes())
            row['canonical_checkpoint_key_matches'] = head['training_input_hashes']['checkpoint_sha256'] == models[0]['checkpoint_sha256']
            row['all_outputs_hash_match'] = all(sha(output / name) == pin for name, pin in result['output_sha256'].items())
            row['test_admitted'] = result['test_admitted']
            row['passed'] &= row['canonical_checkpoint_key_matches'] and row['all_outputs_hash_match'] and not row['test_admitted']
        results.append(row)
    result = dict(schema='adaptgns_goop_action_gate_fit_independent_probes_code_audit_v1', source_sha256=PIN,
        source_is_still_current=sha(original) == PIN, python_executable=sys.executable,
        scope='Synthetic orchestration with stubbed artifact-reader boundary; unchanged full-size numerical core',
        no_scientific_data_or_models=True, probes=results, all_passed=all(row['passed'] for row in results))
    save(HERE / 'goop_action_gate_fit_independent_probes_code_audit_v1.json', result)
    print(json.dumps(result, indent=2))
    assert result['all_passed']


if __name__ == '__main__':
    main()
