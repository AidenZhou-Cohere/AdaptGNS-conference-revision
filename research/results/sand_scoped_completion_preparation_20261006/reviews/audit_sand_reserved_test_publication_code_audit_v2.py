#!/usr/bin/env python3
"""Full-manifest scalar closure and tiny-byte publication probes only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import test_prepare_sand_reserved_test_scoped_v1 as T

HERE = Path(__file__).resolve().parent
PIN = '98b41509fa253a86630cf44ebc86bd3f0b969e875713b736867727bdc392b0bd'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = HERE / 'prepare_sand_reserved_test_scoped_v1.py'
    assert digest(source) == PIN
    root = HERE / 'sand_reserved_test_independent_fixtures_final_v2'
    root.mkdir()
    snapshot = root / 'reviewed_source.py'
    snapshot.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location('_sand_reserved_final_audit', snapshot)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.HERE = HERE
    T.M = M
    results = []
    with pytest.MonkeyPatch.context() as patch:
        a = T.manual_args(root, patch, split='test', protocol=HERE / 'sand_graph_support_100k_protocol_v1.md',
                          train_admission=HERE / 'sand_train_admission.json')
        sources = {s: T.split_fixture(s, root / 'splits', a._cohort_sha) for s in ('valid', 'test')}
        train = T.put(root / 'splits/train.json', {'dataset': 'Sand', 'split': 'train', 'metadata_sha256': M.METADATA_SHA,
            'record_count': 1000, 'records': [{'id': f'train:{i:06d}', 'source_index': i,
            **{k: {'path': f'train/{k}{i}.npy'} for k in ('positions', 'particle_types')}} for i in range(1000)]})
        paths = {'train': train, **{s: v[0] for s, v in sources.items()}}
        numeric = {}
        for split, path in paths.items():
            manifest = M.read(path)
            for index, row in enumerate(manifest['records']):
                for key in ('positions', 'particle_types'):
                    p = path.parent / row[key]['path']
                    p.parent.mkdir(exist_ok=True)
                    p.write_bytes(f'INDEPENDENT SYNTHETIC {split} {key} {index}'.encode())
                    row[key]['sha256'] = digest(p)
                    numeric[str(p)] = digest(p)
                if split != 'train':
                    row['trajectory_content_sha256'] = T.h(row['positions']['sha256'] + ':' + row['particle_types']['sha256'])
            T.put(path, manifest)
            if split != 'train':
                evidence = M.read(sources[split][1])
                evidence['splits'][split]['manifest_sha256'] = digest(path)
                T.put(sources[split][1], evidence)
        a.train_manifest, a.valid_manifest, a.test_manifest = (paths[s] for s in ('train', 'valid', 'test'))
        a.structural_report = sources['test'][1]
        a.census_report = root / 'census.json'
        patch.setattr(M, 'TRAIN_MANIFEST_SHA', digest(train))
        original_report = {'schema': 'adaptgns_sand_complete_numeric_census_v1', 'status': 'complete_no_duplicates',
            'cohort_sha256': a._cohort_sha, 'counts': M.COUNTS, 'all_split_numeric_duplicates_checked': True,
            'preparation_source_sha256': PIN, 'manifests_sha256': {s: digest(p) for s, p in paths.items()},
            'numeric_files_sha256': numeric}
        selected_numeric = Path(next(iter(numeric)))
        original_numeric = selected_numeric.read_bytes()
        for case in ('complete_candidate', 'missing_numeric_map', 'one_opaque_file_map', 'missing_train_manifest',
                     'source_during_serialization', 'train_numeric_during_serialization', 'clock_during_serialization'):
            with pytest.MonkeyPatch.context() as inner:
                selected_numeric.write_bytes(original_numeric)
                report = json.loads(json.dumps(original_report))
                if case == 'missing_numeric_map':
                    del report['numeric_files_sha256']
                elif case == 'one_opaque_file_map':
                    report['numeric_files_sha256'] = {str(selected_numeric): digest(selected_numeric)}
                elif case == 'missing_train_manifest':
                    del report['manifests_sha256']['train']
                T.put(a.census_report, report)
                T.bind(a, *paths.values(), a.structural_report, a.census_report, a.protocol, a.train_admission)
                a.output_dir = root / case
                dumps = json.dumps
                touched = []
                def encode(value, **kwargs):
                    raw = dumps(value, **kwargs)
                    if value.get('status') == 'candidate_requires_root_review':
                        if case == 'source_during_serialization':
                            a.census_report.write_bytes(b'CHANGED DURING FINAL SERIALIZATION'); touched.append('census')
                        elif case == 'train_numeric_during_serialization':
                            selected_numeric.write_bytes(b'CHANGED DURING FINAL SERIALIZATION'); touched.append('train numeric')
                        elif case == 'clock_during_serialization':
                            inner.setattr(M, 'now', lambda: a._stop); touched.append('clock')
                    return raw
                inner.setattr(M.json, 'dumps', encode)
                error = None
                try:
                    M.candidate(a)
                except ValueError as exc:
                    error = str(exc)
                exists = (a.output_dir / 'split_admission_candidate.json').exists()
                expected = case == 'complete_candidate'
                results.append({'case': case, 'success_output_exists': exists, 'error': error, 'mutation_fired': touched,
                                'passed': exists == expected and (error is None) == expected})
        empty_root = root / 'empty_scalar_type'
        empty_root.mkdir()
        p = np.empty((320, 0, 2), dtype=np.float32)
        t = np.array(6, dtype=np.int64)
        row = {'positions': T.C.save_array(np, empty_root / 'p.npy', empty_root.name, p),
               'particle_types': T.C.save_array(np, empty_root / 't.npy', empty_root.name, t),
               'logical_content_sha256': T.h(T.C.value_hash(p) + ':' + T.C.value_hash(t))}
        try:
            M.numeric_identity(np, T.C, root, row)
        except ValueError as exc:
            results.append({'case': 'empty_particles_scalar_type6', 'passed': True, 'error': str(exc)})
        else:
            results.append({'case': 'empty_particles_scalar_type6', 'passed': False})
    result = {'schema': 'adaptgns_sand_reserved_test_independent_probes_code_audit_v2',
              'source_sha256': PIN, 'source_is_still_current': digest(source) == PIN, 'synthetic_only': True,
              'python_executable': sys.executable, 'numeric_manifest_file_count': len(numeric),
              'probes': results, 'all_passed': all(r['passed'] for r in results)}
    output = HERE / 'sand_reserved_test_independent_probes_code_audit_final_v2.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    assert result['all_passed']


if __name__ == '__main__':
    main()
