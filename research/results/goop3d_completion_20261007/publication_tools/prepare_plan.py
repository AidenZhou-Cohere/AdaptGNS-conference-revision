#!/usr/bin/env python3
"""Build a compact, explicit publication inventory; writes only fresh plan files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PACKAGE = 'research/results/goop3d_completion_20261007'
BASE = Path('work/goop3d_completion_20261007')


def pin(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def encode(doc):
    return (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    workspace = args.workspace.resolve()
    out = args.output_dir.resolve()
    assert out.is_relative_to(workspace), 'plan output must be in workspace'
    out.mkdir(parents=True, exist_ok=True)
    files = []
    destinations = set()
    def add(source, destination, role):
        source = Path(source)
        source = source if source.is_absolute() else workspace / source
        assert source.is_file() and not source.is_symlink()
        assert destination not in destinations, destination
        destinations.add(destination)
        entry = {'source': str(source.relative_to(workspace)), 'destination': destination, 'role': role, **pin(source)}
        files.append(entry)
        return {'destination': destination, 'sha256': entry['sha256']}
    def subtree(relative, role, excluded=()):
        for source in sorted((workspace / BASE / relative).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix not in excluded:
                add(source, str(source.relative_to(workspace / BASE)), role)
    def fresh(name, doc):
        path = out / name
        with path.open('xb') as stream:
            stream.write(doc.encode() if isinstance(doc, str) else encode(doc))
        return path

    for relative in ('autonomous_v1', 'autonomous_analysis_v1', 'observed_v1', 'observed_diagnosis_v1', 'observed_finalize_v2'):
        subtree(relative, 'reviewed_source_protocol_test_or_history')
    subtree('review_v1', 'independent_review_or_preserved_review_history')
    # Explicitly snapshot ordinary top-level operational files. Never recurse root/.
    for source in sorted((workspace / BASE / 'root').iterdir()):
        if source.is_file() and source.suffix in ('.py', '.json', '.md', '.log'):
            add(source, str(source.relative_to(workspace / BASE)), 'historical_operation_snapshot')
    for relative in ('root/status_before_resumption', 'root/transfer_yellow-worm-77',
                     'root/transfer_aquamarine-toad-75', 'root/transfer_committed_yellow_initial_0228',
                     'root/transfer_committed_aqua_initial_0228'):
        subtree(relative, 'historical_transfer_or_status_evidence')

    # Exact runtime source layout. Do not import or execute frozen scientific code.
    autonomous_plan = json.loads((workspace / BASE / 'autonomous_v1/plan.json').read_bytes())
    for relative, digest in autonomous_plan['source_pins'].items():
        prefix = Path('work/deadline_research_20261005') if relative.startswith('cuda_preparation/') else Path('outputs/AdaptGNS')
        binding = add(prefix / relative, 'frozen/runtime/' + relative, 'frozen_scientific_source')
        assert binding['sha256'] == digest, relative
    observed_pins = json.loads((workspace / BASE / 'observed_v1/source_pins.json').read_bytes())
    observed_original = Path('work/deadline_research_20261005/cuda_preparation/goop3d_observed_history_analysis_UNADMITTED_transition_v1')
    for relative, digest in observed_pins['frozen_files_sha256'].items():
        binding = add(observed_original / relative, 'frozen/runtime/cuda_preparation/goop3d_observed_history_analysis_v1/' + relative, 'frozen_scientific_source')
        assert binding['sha256'] == digest, relative
    analysis_pins = json.loads((workspace / BASE / 'autonomous_analysis_v1/source_manifest.json').read_bytes())
    for relative, record in analysis_pins['arithmetic_sources'].items():
        binding = add(Path('work/deadline_research_20261005/cuda_preparation') / relative,
                      'frozen/runtime/cuda_preparation/' + relative, 'frozen_scientific_source')
        assert binding['sha256'] == record['sha256'], relative

    # The original failed cache stays external; all its exact file pins remain in
    # its included transfer manifest. No per-row cache is copied into publication.
    collections = ('observed_failure_collection_v1', 'observed_finalized_v2_collection', 'autonomous_audit_snapshot001_collection')
    references = []
    for collection in collections:
        rel = BASE / 'root' / collection
        absolute = workspace / rel
        for name in ('original_transfer.json', 'verified.json'):
            if (absolute / name).is_file():
                add(rel / name, str(Path('root') / collection / name), 'actual_collection_receipt')
        transfer_destination = str(Path('root') / collection / 'files/transfer_manifest.json')
        transfer_binding = add(rel / 'files/transfer_manifest.json', transfer_destination, 'actual_collection_manifest')
        subtree(str(Path('root') / collection / 'files/jobs'), 'actual_job_completion_evidence')
        manifest = json.loads((absolute / 'files/transfer_manifest.json').read_bytes())
        original_transfer = json.loads((absolute / 'original_transfer.json').read_bytes())
        archive = absolute / 'original_archive.tar.gz'
        expected_archive = {'bytes': original_transfer['archive_bytes'], 'sha256': original_transfer['archive_sha256']}
        assert pin(archive) == expected_archive
        refs = {name: record for name, record in manifest['files'].items()
                if not name.startswith('jobs/') and not (collection == 'observed_finalized_v2_collection' and name.startswith('observed_finalized_v2/'))}
        references.append({
            'id': collection, 'publication_mode': 'reference_only_for_items_named_in_reference_inventory',
            'local_collection_root': str(rel / 'files'),
            'remote_completion_root': '/root/repos/AdaptGNS-cuda-20261006/goop3d_completion_20261007',
            'manifest': transfer_binding, 'referenced_file_count': len(refs),
            'referenced_files_bytes': sum(record['bytes'] for record in refs.values()),
            'exact_file_inventory_location': transfer_destination,
            'archive': {'local_path': str(archive.relative_to(workspace)), **expected_archive},
            'scope': ('Original failed observed cache and checkpoint index; all 2568 successful row audits retained, failed final checker retained.'
                      if collection == 'observed_failure_collection_v1' else
                      'Autonomous partial snapshot 001; no autonomous numerical admission. Preserve original 331 outcomes and full 2160 denominator.'
                      if collection == 'autonomous_audit_snapshot001_collection' else
                      'Transport archive only; five verified observed final products are included individually.'),
            'pin_basis': 'Included actual transfer manifest and separately checked local transport archive; no scientific recomputation.'})

    product_root = BASE / 'root/observed_finalized_v2_collection/files/observed_finalized_v2'
    products = {name: add(product_root / name, 'observed_products/' + name, 'verified_observed_product')
                for name in ('audit.json', 'summary.json', 'arithmetic_check.json', 'completion.json', 'retained_original_failure.json')}
    # Preserve rendered fragments, reviews, source and earlier candidate history.
    # The compressed companion and duplicate unchanged summaries are reference-only.
    for source in sorted((workspace / BASE / 'presentation_observed_v1').rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts:
            continue
        relative = str(source.relative_to(workspace / BASE))
        if source.suffix == '.gz' or source.name == 'source_summary_unchanged.json':
            references.append({'id': relative, 'publication_mode': 'reference_only',
                'local_path': str(source.relative_to(workspace)), **pin(source),
                'scope': 'Preserved presentation companion or duplicate summary. Exact admitted summary is included at observed_products/summary.json. Generated v1 remains historical; v2 has independent review.'})
        else:
            add(source, relative, 'reviewed_presentation_or_preserved_candidate')

    def existing_binding(destination):
        entry = next(e for e in files if e['destination'] == destination)
        return {'destination': destination, 'sha256': entry['sha256']}
    references.extend([
        {'id': 'original_runtime_inputs', 'publication_mode': 'reference_only',
         'manifest': existing_binding('autonomous_v1/transfer_files.json'),
         'expected_files': 208, 'scope': 'Official selected numeric data, fixed cohort, six frozen 25000-update checkpoints, original source/protocol/configuration. Source pins are reproduced under frozen/runtime. The original six checkpoint sizes are null; they are not inferred.',
         'actual_runtime_transfer_receipts': ['root/transfer_yellow-worm-77/destination.stdout', 'root/transfer_aquamarine-toad-75/destination.stdout'],
         'actual_runtime_transferred_bytes_per_host': 1861835983,
         'runtime_location': 'Staged runtime exists on yellow and aquamarine; teal uses original paths.'},
        {'id': 'original_scalar_collection', 'publication_mode': 'reference_only',
         'remote_path': '/root/repos/AdaptGNS-cuda-20261006/goop3d_final_analysis_20261006_v1/stopped_collection.json',
         'bytes': 659091312, 'sha256': '616c612724f62d21bcb521c6163e83d9038b34fdd8012980d6ed55f4b05bed5f',
         'scope': 'Original stopped collection, all observed and autonomous accounting states; inherited saved values, not new independent source truth.'},
        {'id': 'original_evaluation_ledger', 'publication_mode': 'reference_only',
         **{'remote_path': analysis_pins['original_ledger']['path'], 'bytes': analysis_pins['original_ledger']['bytes'], 'sha256': analysis_pins['original_ledger']['sha256']},
         'scope': 'Original full grid; no missing/failed outcomes dropped.'},
        {'id': 'original_teal_metadata', 'publication_mode': 'reference_only',
         **{'remote_path': analysis_pins['metadata']['path'], 'bytes': analysis_pins['metadata']['bytes'], 'sha256': analysis_pins['metadata']['sha256']},
         'scope': 'Correct original teal metadata path. Earlier incorrect handoff paths are preserved in history.'},
    ])
    inventory = {'schema': 'goop3d_completion_external_references_v1', 'references': references,
                 'no_remote_fetch_no_model_or_array_execution': True,
                 'autonomous_numerical_admission': False,
                 'note': 'Pins inherited from reviewed source/transfer manifests are labeled references; inclusion does not claim an independent reconstruction of data generation. Autonomous snapshots and original cache rows are excluded from Git.'}
    refs_file = fresh('external_references.json', inventory)
    add(refs_file, 'external_references.json', 'reference_inventory')
    readme = '''# Goop3D completion package, 7 October 2026

This package records the completed, independently reviewed observed-history comparison for six existing Goop3D models (base/mix, seeds 0, 1 and 2, 25,000 updates). It also preserves the frozen sources, reviewed continuation machinery and historical process evidence for autonomous completion. **Autonomous completion is not admitted by this package.** It is a separate 2,160-cell evaluation; the continuation plan preserves 331 original outcomes and fills 1,829 originally missing cells. No training was repeated.

The observed scope contains 2,568 audited rows: 768 clean-validation cells and 900 same-state cells for each of validation and test. The full original accounting grid retains 4,728 states. All six policies, both model arms and all three paired seeds remain in the results. Sample SD is over the three training seeds. Previous-observed-base-risk placement uses observed histories; these results do not establish autonomous feedback quality. Saved source/normalization provenance is inherited, and prior inspection is not presented as pristine independent confirmation. The separate WaterDrop experiments, compact pilot and historical arrays are not merged into this extension.

## Included evidence

- `observed_products/`: exact audit, summary, arithmetic check, completion record and retained original failed-checker outcome. The actual observed finisher exited 0, and its original process group was empty at collection. `review_v1/observed_final_products_v2_review.json` independently binds all products, 2,568 row caches, 2,436 model/metric aggregates and 1,382 statistic objects. `root/observed_numerical_admission.json` admits only this scope.
- `observed_v1/`, `observed_diagnosis_v1/`, `observed_finalize_v2/`: original resumable auditor, all tests/failures, the 36-case SD diagnosis and the separately versioned cache-only correction. The original 2,568 row audits and original-byte rehash passed before the first final scalar checker failed. All 36 discrepancies were graph-count SDs for bit-identical seed means; no accuracy mean or contrast mismatch was found. The new checker uses exact pairwise unbiased variance, leaving counts, means, contrasts, nulls, denominators and tolerances unchanged. The finisher performed zero new row-array audits and no model execution.
- `autonomous_v1/`, `autonomous_analysis_v1/`: frozen plan, explicit resume/source/lock protections, original versions, failed synthetic attempts and reviewed final tests. Analysis-source history preserves the corrected teal metadata path. These sources do not certify the result of work still running.
- `frozen/runtime/`: byte-identical scientific source closures and original protocols in their runtime-relative layout. The original failed observed scalar checker remains present. Source review, synthetic tests and actual-product review are distinct evidence stages.
- `root/`: dated launch, native identity, transfer and completion evidence. These files are immutable historical snapshots, not live status. Owner/child start ticks and boot identity distinguish actual process completion from a launch receipt. Earlier closed-status documents under `root/status_before_resumption/` remain history.
- `presentation_observed_v1/`: immutable renderer/tests, v1 candidate history and independently reviewed v2 generated fragments, all 13 tables and 224 rendered statistic claims. v2 receipt/review bind the exact admitted summary. Native manuscript compilation and page-layout checks remain separate integration work; these fragments are not a final manuscript.

## External inputs and large artifacts

`external_references.json` identifies official selected inputs, frozen checkpoints, original scalar collection/ledger, original observed row cache, autonomous snapshot and transport archives with scope and exact pins. Included transfer manifests retain each collected file's hash and byte count. Large raw arrays, checkpoints, per-row caches and archives are not duplicated here. The compressed presentation companion and duplicate summary files are referenced; the complete admitted summary is included once at `observed_products/summary.json`. Original checkpoint sizes missing from the source manifest remain null.

`../goop3d_observed_history_failure_20261007/`, `../goop3d_observed_history_failure_closure_20261007/` and `../goop3d_failure_publication_review_20261007/` preserve earlier published failure evidence. This package does not replace or erase those outcomes.

## Reproduction and verification

`curation_plan.json` records each original workspace-relative source path, destination, size, SHA-256 and role. `publication_manifest.json` is created only after the explicit copier verifies and copies every selected byte. The copy tool defaults to verification, requires the reviewed plan's exact SHA-256 and refuses an existing destination. It performs no Git/network operations or scientific execution. To verify original workspace inputs, run `publication_tools/copy_verified.py --workspace <original-workspace> --plan <reviewed-plan.json> --plan-sha256 <reviewed-sha256>`; the execution owner may add `--copy --destination <fork>/research/results/goop3d_completion_20261007` after review. Paths are provenance and must be mapped explicitly for a different machine; they are not download URLs.

Running tests of the copier uses only temporary synthetic files: `python3 -m unittest discover -s publication_tools -p test_copy_verified.py -v`. Scientific reproduction requires the pinned external artifacts and the exact source/environment/argument manifests; curation does not rerun completed scientific cells. No aggregate is recalculated by this package operation.

## Outstanding work at this publication snapshot

Autonomous workers and their subsequent complete-array/paired audit remain separately owned and unadmitted here. Manuscript prose integration, native LaTeX compilation, final page inspection and author verification remain outside this snapshot. Descriptive shared-host timing does not establish end-to-end speedup. This package does not claim conference readiness or submission.
'''
    readme_file = fresh('PACKAGE_README.md', readme)
    add(readme_file, 'README.md', 'package_scope_and_instructions')
    tools_dir = workspace / BASE / 'publication_plan_v1'
    for name in ('copy_verified.py', 'test_copy_verified.py', 'prepare_plan.py', 'synthetic_tests.log'):
        add(tools_dir / name, 'publication_tools/' + name, 'publication_tool_or_synthetic_test')
    gate = {'status': 'reviewed_for_publication',
            'scope': 'Observed-history only; no autonomous numerical admission',
            'review': existing_binding('review_v1/observed_final_products_v2_review.json'),
            'transfer_manifest': existing_binding('root/observed_finalized_v2_collection/files/transfer_manifest.json'),
            'admission': existing_binding('root/observed_numerical_admission.json'),
            'products': products}
    plan = {'schema': 'goop3d_completion_publication_plan_v1', 'package': PACKAGE,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'files': sorted(files, key=lambda e: e['destination']), 'numerical_product_gate': gate,
            'pending': ['Autonomous completion and independent actual-product review', 'Final manuscript integration and native LaTeX/page-layout verification', 'Author verification and submission decision'],
            'prohibited_actions': ['No training, evaluation, array audit, or scientific recomputation', 'No network, Git mutation, or process launch', 'No overwrite of existing package'],
            'scope': 'Reviewed observed-history completion; frozen sources and historical operational provenance; autonomous numerical products are reference-only.'}
    plan_file = fresh('plan.json', plan)
    print(json.dumps({'status': 'explicit_plan_prepared', 'plan': str(plan_file), **pin(plan_file),
                      'selected_files': len(files), 'selected_bytes': sum(e['bytes'] for e in files)}, sort_keys=True))


if __name__ == '__main__':
    main()
