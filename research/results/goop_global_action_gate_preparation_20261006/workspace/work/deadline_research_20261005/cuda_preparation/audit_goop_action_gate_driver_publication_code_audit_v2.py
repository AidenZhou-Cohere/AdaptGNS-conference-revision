#!/usr/bin/env python3
"""Independent synthetic driver publication replay and mathematical AST check."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
import pytest
import test_run_goop_action_gate_v1 as T

HERE = Path(__file__).resolve().parent
PIN = 'cbbb2d18ffd08aa6721b53e95d3adb2f7ab39ef1900e737266c70bd3147d6869'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(path).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    source = HERE / 'run_goop_action_gate_v1.py'
    assert sha(source) == PIN
    root = HERE / 'goop_action_gate_driver_independent_publication_fixtures_v2'
    root.mkdir()
    snapshot = root / 'reviewed_source.py'
    snapshot.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location('_driver_independent_final_snapshot', snapshot)
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    D.HERE = HERE
    T.D = D
    results = []
    for case in ('complete', 'scalar_row_before_encode', 'scalar_row_after_encode', 'aggregate_after_row_commit',
                 'scalar_collection_after_encode', 'original_checkpoint_after_row_commit'):
        directory = root / case
        directory.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            context, native, expected = T.execution_fixture(directory, patch)
            encode = D.encode
            publish = D.publish_row
            mutations = []
            def changed_encode(value):
                if value.get('schema') == D.LABEL_SCHEMA and case == 'scalar_row_before_encode':
                    value['features'][0] += 1
                    mutations.append('row before serialization')
                raw = encode(value)
                if value.get('schema') == D.LABEL_SCHEMA and case == 'scalar_row_after_encode':
                    value['features'][0] += 1
                    mutations.append('row after serialization')
                if value.get('schema') == D.LABEL_COLLECTION and case == 'scalar_collection_after_encode':
                    value['complete_rows'] += 1
                    mutations.append('collection after serialization')
                return raw
            def changed_publish(*args, **kwargs):
                row = publish(*args, **kwargs)
                if args[1].startswith('source') and case == 'aggregate_after_row_commit':
                    row['features'][0] += 1
                    mutations.append('aggregate after committed row')
                if args[1].startswith('source') and case == 'original_checkpoint_after_row_commit':
                    context.args.checkpoint.write_bytes(b'changed opaque synthetic checkpoint')
                    mutations.append('checkpoint after committed row')
                return row
            patch.setattr(D, 'encode', changed_encode)
            patch.setattr(D, 'publish_row', changed_publish)
            with contextlib.redirect_stdout(io.StringIO()):
                code = D.execute(context, time.perf_counter())
            collection_path = context.args.output_dir / 'label_collection.json'
            collection = json.loads(collection_path.read_bytes()) if collection_path.exists() else None
            complete = collection is not None and collection['status'] == 'complete'
            consistent = None
            if collection is not None:
                consistent = all(json.loads((context.args.output_dir / row['row_file']).read_bytes()) ==
                    {k: v for k, v in row.items() if k not in ('row_file', 'row_sha256')} for row in collection['rows'])
            good = (code == 0 and complete and consistent) if case == 'complete' else (code != 0 and not complete)
            results.append(dict(case=case, passed=good, exit_code=code,
                collection_status=collection['status'] if collection else None, all_committed_rows_match=consistent,
                mutations=mutations, failed_publication_receipt=(context.args.output_dir / 'failed_collection_publication.json').exists()))
    numerical_source = HERE / 'goop_action_gate_driver_row_snapshot_probe_v1/reviewed_source.py'
    assert sha(numerical_source) == '082fe7affa6231919adf4c65a5f1328d119bec8f42217c3ac0c7532d3317afcb'
    old, new = functions(numerical_source), functions(source)
    numerical = {name: old[name] == new[name] for name in ('pair_actions', 'label_row', 'choose_action', 'rollout_row',
        'remaining', 'collection_coverage', 'execute', 'release_gate', 'validate_release', 'capacity_sources', 'expected_schedule')}
    report = dict(schema='adaptgns_goop_action_gate_driver_independent_publication_code_audit_v2', source_sha256=PIN,
        source_is_still_current=sha(source) == PIN, synthetic_only=True, probes=results,
        inherited_numerical_review_sha256='db514517a3ddd71d7a85f23fe4b523d03658780a212fb14b3066a80a67033d49',
        unchanged_functions_from_independently_reviewed_082fe7af=numerical,
        changed_function_names=sorted(name for name in new if new[name] != old.get(name)),
        all_passed=all(r['passed'] for r in results) and all(numerical.values()))
    output = HERE / 'goop_action_gate_driver_independent_publication_code_audit_v2.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['all_passed']


if __name__ == '__main__':
    main()
