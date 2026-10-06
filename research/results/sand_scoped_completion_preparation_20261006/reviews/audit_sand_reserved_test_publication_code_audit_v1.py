#!/usr/bin/env python3
"""Independent tiny-fixture probes. No scientific data or model access."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import test_prepare_sand_reserved_test_scoped_v1 as T

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    source = HERE / 'prepare_sand_reserved_test_scoped_v1.py'
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.expected_sha
    directory = HERE / ('sand_reserved_test_independent_fixtures_' + args.run_name)
    directory.mkdir()
    snapshot = directory / 'reviewed_source.py'
    snapshot.write_bytes(raw)
    spec = importlib.util.spec_from_file_location('_sand_reserved_audit_snapshot', snapshot)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.HERE = HERE
    T.M = M
    rows = []

    def record(name, operation):
        try:
            operation()
        except (ValueError, AssertionError, FileNotFoundError) as error:
            rows.append({'case': name, 'rejected': True, 'error_type': type(error).__name__, 'error': str(error)})
        else:
            rows.append({'case': name, 'rejected': False})

    root = directory / 'empty_scalar_type'
    root.mkdir()
    p = np.empty((320, 0, 2), dtype=np.float32)
    t = np.array(6, dtype=np.int64)
    empty = {'positions': T.C.save_array(np, root / 'p.npy', root.name, p),
             'particle_types': T.C.save_array(np, root / 't.npy', root.name, t),
             'logical_content_sha256': T.h(T.C.value_hash(p) + ':' + T.C.value_hash(t))}
    record('empty_particles_with_scalar_type6', lambda: M.numeric_identity(np, T.C, directory, empty))

    for case in ('missing_numeric_map', 'one_opaque_file_map', 'source_during_serialization',
                 'numeric_during_serialization', 'clock_during_serialization'):
        root = directory / case
        root.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            a = T.manual_args(root, patch, split='test', protocol=HERE / 'sand_graph_support_100k_protocol_v1.md',
                              train_admission=HERE / 'sand_train_admission.json')
            mp, sp, _, _ = T.split_fixture('test', root / 'splits', a._cohort_sha)
            a.valid_manifest = None
            a.test_manifest = mp
            a.structural_report = sp
            numeric = root / 'opaque_numeric_bytes'
            numeric.write_bytes(b'tiny independent synthetic bytes')
            report = {'schema': 'adaptgns_sand_complete_numeric_census_v1', 'status': 'complete_no_duplicates',
                      'cohort_sha256': a._cohort_sha, 'counts': M.COUNTS, 'all_split_numeric_duplicates_checked': True,
                      'preparation_source_sha256': digest(snapshot), 'manifests_sha256': {'test': digest(mp)}}
            if case != 'missing_numeric_map':
                report['numeric_files_sha256'] = {str(numeric): digest(numeric)}
            a.census_report = T.put(root / 'census.json', report)
            T.bind(a, mp, sp, a.census_report, a.protocol, a.train_admission)
            original = json.dumps

            def encode(value, **kwargs):
                encoded = original(value, **kwargs)
                if value.get('status') == 'candidate_requires_root_review':
                    if case == 'source_during_serialization':
                        a.census_report.write_bytes(b'changed after candidate serialization')
                    elif case == 'numeric_during_serialization':
                        numeric.write_bytes(b'changed after candidate serialization')
                    elif case == 'clock_during_serialization':
                        patch.setattr(M, 'now', lambda: a._stop)
                return encoded

            patch.setattr(M.json, 'dumps', encode)
            record(case, lambda: M.candidate(a))
            rows[-1]['success_output_exists'] = (a.output_dir / 'split_admission_candidate.json').exists()

    output = {'schema': 'adaptgns_sand_reserved_test_independent_probes_code_audit_v1',
              'source_sha256': args.expected_sha, 'snapshot_sha256': digest(snapshot),
              'synthetic_only': True, 'python_executable': sys.executable, 'probes': rows,
              'source_is_still_current': digest(source) == args.expected_sha,
              'all_expected_refusals': all(row['rejected'] for row in rows)}
    target = HERE / ('sand_reserved_test_independent_probes_code_audit_' + args.run_name + '.json')
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
