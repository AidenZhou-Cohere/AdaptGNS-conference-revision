#!/usr/bin/env python3
"""Build a deterministic local publication candidate; never mutate Git or launch work."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
ALLOWED = {'.py', '.json', '.md', '.log', '.diff', '.stdout', '.stderr', '.html'}
FROZEN_GOOP = [
    'evaluate_goop_graph_support_final.py', 'benchmark_goop_graph_support_rollout.py',
    'goop_evaluation_contract.py', 'train_goop_graph_support_cuda.py', 'sand_graph_support_policy.py',
]
CANDIDATE_METADATA = [
    'aquamarine_idle_locality_1522_root_v1.json', 'goop_evaluation_release_A_0947_v1.json',
    'goop_evaluation_release_B_0949_v1.json', 'goop_gate_train_labels_seed0_release_1155_v1.json',
    'goop_gate_train_labels_seed1_release_1155_v1.json',
]
SAND_DEPENDENCIES = [
    'prepare_sand_final_cohort_scoped_v1.py', 'prepare_sand_reserved_test_scoped_v1.py',
    'supervise_sand_runtime_migration_recovery_v2.py', 'train_sand_runtime_migration_recovery_v1.py',
    'audit_sand_runtime_migration_endpoints_v1.py', 'train_sand_graph_support_cuda.py',
    'evaluate_sand_graph_support_final.py', 'benchmark_sand_graph_support_rollout.py',
    'download_sand_public_v2.py', 'repackage_designsafe_sand.py',
    'sand_graph_support_100k_protocol_v1.md', 'sand_scoped_operational_amendment_v1.md',
    'sand_scoped_schedule_fixed_spec_v2.json', 'sand_completed_host_process_check.template.json',
    'sand_final_cohort_scoped_release.template.json', 'sand_final_evaluation_scoped_release.template.json',
    'goop2d_completion_sand_handoff_operator_code_audit_v1.md',
]
EXCLUDE = {
    'audit_goop2d_v3_actual_scalars_code_audit_v1.py':
        'Unexecuted preparatory success-only scalar checker; actual pilot failed and the complete-phase checker was never used.',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    if not path.is_file() or path.is_symlink() or path.suffix not in ALLOWED:
        raise ValueError('Only allowlisted regular text artifacts are packageable: ' + str(path))
    raw = path.read_bytes()
    raw.decode('utf-8')
    return raw


def add_tree(files, directory):
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix in ALLOWED and '__pycache__' not in path.parts:
            files.add(path)


def archive(output, paths):
    rows = []
    with output.open('xb') as raw_output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw_output, mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(paths):
                    raw = regular(path)
                    name = str(path.relative_to(WORKSPACE))
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(raw), 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    tar.addfile(info, io.BytesIO(raw))
                    rows.append(dict(path=name, size_bytes=len(raw), sha256=sha(raw)))
    return dict(file=output.name, sha256=sha(output.read_bytes()), size_bytes=output.stat().st_size,
                member_count=len(rows), uncompressed_member_bytes=sum(r['size_bytes'] for r in rows), members=rows)


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    cost = {p for p in HERE.iterdir() if p.is_file() and 'goop2d' in p.name
            and p.suffix in ALLOWED and p.name not in EXCLUDE
            and p.name != 'goop2d_completion_sand_handoff_operator_code_audit_v1.md'}
    cost.update(HERE / n for n in FROZEN_GOOP)
    cost.update(HERE / n for n in CANDIDATE_METADATA)
    add_tree(cost, HERE / 'goop2d_cost_independent_test_history')
    add_tree(cost, HERE / 'goop2d_memory_accounting_documentation_transition_v1')
    evidence = set()
    for name in (
        'goop2d_validation_cost_candidate_UNADMITTED_owner_v1',
        'goop2d_validation_cost_candidate_UNADMITTED_owner_v2',
        'goop2d_validation_cost_candidate_UNADMITTED_owner_v2r2',
        'goop2d_validation_cost_candidate_UNADMITTED_owner_v3',
        'goop2d_validation_cost_staged_root_v2', 'goop2d_validation_cost_staged_root_v3',
        'goop2d_cost_pilot_failure_20261006_root_v2', 'goop2d_cost_pilot_failure_20261006_root_v3',
    ):
        add_tree(evidence, HERE / name)
    sand = {p for p in HERE.iterdir() if p.is_file() and p.suffix in ALLOWED
            and ('sand_recovered' in p.name or 'sand_reserved_test_recovered' in p.name)}
    sand.update(HERE / n for n in SAND_DEPENDENCIES)
    add_tree(sand, HERE / 'sand_recovered_cohort_bridge_source_history')
    add_tree(sand, HERE / 'sand_recovered_completion_candidates_UNADMITTED_transition_v1')
    required = [
        HERE / 'goop2d_v3_failed_pilot_scientific_review_code_audit_v1.json',
        HERE / 'goop2d_validation_cost_v3_root_decline_v1.json',
        HERE / 'goop2d_v3_failed_pilot_lifecycle_review_transition_v1.json',
    ]
    for path in required:
        regular(path)
        cost.add(path)
    sets = {'goop2d_sources_and_reviews': cost, 'goop2d_candidates_and_failures': evidence,
            'sand_completion_bridge_preparation': sand}
    seen = set()
    for paths in sets.values():
        if seen & paths:
            raise ValueError('Artifact duplicated between package families')
        seen |= paths
    products = {name: archive(output / (name + '.tar.gz'), paths) for name, paths in sets.items()}
    raw_receipt = json.loads((HERE / 'goop2d_cost_pilot_failure_20261006_root_v3/numeric_archive_receipt.json').read_text())
    specification = dict(schema='adaptgns_cost_failure_and_sand_bridge_publication_package_v1',
        status='local_publication_candidate_no_git_changes', archives=products,
        omitted_unexecuted_preparation=EXCLUDE,
        numerical_payload_policy='All raw model, trajectory, and numeric reference arrays omitted uniformly; original scalar JSON/logs remain exact.',
        v3_original_numeric_archive_receipt=raw_receipt,
        scientific_scope=dict(goop2d_cost_result_admitted=False, goop2d_full_declined=True,
            partial_scalar_files_preserved=45, subset_cost_statistics=False, performance_aggregation=False,
            sand_scope='Reviewed completion bridge and unresolved handoff preparation only; no completed cohort or evaluation claimed.'),
        operational_authority=False, git_mutation=False, remote_calls=False)
    (output / 'archive_manifest.json').write_text(json.dumps(specification, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: {n: v for n, v in row.items() if n != 'members'} for k, row in products.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
